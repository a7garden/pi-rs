//! [`FacetBundleModuleLoader`] implementation: evaluate facet bundle entries in
//! an embedded deno_core `JsRuntime`.
//!
//! One loader owns one long-lived JS thread. Each `load_entry` evaluates the
//! (integrity-verified, upstream-ported checks already done by pi-chord) CJS
//! artifact with the upstream `(exports, require, module, __filename,
//! __dirname)` shape, validates its export shape with pi-chord's
//! `facets_from_module` (exact upstream error messages), and hands back
//! [`Facet`] values whose `setup` dispatches into the same runtime.
//!
//! The JS surface a facet setup sees mirrors upstream `FacetEnvironment` for
//! the lifecycle callbacks (`own`, `onActivate`, `onDeactivate`); the typed
//! service operations (`use`, `observe`, `provide`, `provideMany`,
//! `replicatedState`) cannot cross an untyped JS boundary in the Rust port and
//! throw with a message pointing at the crate docs (documented deviation).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use deno_core::{ModuleSpecifier, PollEventLoopOptions};
use pi_chord::node::bundle_loader::{FacetBundleModuleLoader, facets_from_module};
use pi_chord::services::errors::{AnyError, error};
use pi_chord::types::{BoxFuture, Facet};
use serde_json::Value;

use crate::compiler::{BUNDLE_FORMAT, BUNDLE_FORMAT_VERSION};
use crate::host::{make_runtime, spawn_js_thread};
use crate::loader::{PiModuleLoader, VirtualSource};

/// How long a synchronous JS facet `setup` call may run before the host gives
/// up (upstream setups are synchronous by construction).
const SETUP_TIMEOUT: Duration = Duration::from_secs(30);

/// Options for [`FacetRuntimeModuleLoader`].
pub struct FacetRuntimeOptions {
    /// Upstream `pluginId` half of facet validation labels.
    pub plugin_id: String,
    /// Upstream `entryName` half of facet validation labels.
    pub entry_name: String,
    /// Modules the bundle's declared external imports resolve to, keyed by
    /// specifier. Values are the JSON exports of the module. Nothing is
    /// registered by default: the embedded runtime ships no node/npm shims.
    pub externals: HashMap<String, Value>,
}

impl FacetRuntimeOptions {
    pub fn new(plugin_id: impl Into<String>, entry_name: impl Into<String>) -> Self {
        Self {
            plugin_id: plugin_id.into(),
            entry_name: entry_name.into(),
            externals: HashMap::new(),
        }
    }
}

/// One lifecycle operation a JS setup performed on its environment shim.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct EnvOp {
    kind: String,
    slot: u32,
}

enum Request {
    Evaluate {
        module_path: PathBuf,
        external_imports: Vec<String>,
        reply: tokio::sync::oneshot::Sender<Result<Vec<(String, u32)>, String>>,
    },
    Setup {
        slot: u32,
        reply: mpsc::SyncSender<Result<Vec<EnvOp>, String>>,
    },
    Callback {
        slot: u32,
        reply: tokio::sync::oneshot::Sender<Result<(), String>>,
    },
}

/// Handle to the loader's JS thread, shared with the `Facet::setup` closures.
#[derive(Clone)]
struct FacetRuntimeHandle {
    requests: mpsc::Sender<Request>,
}

impl FacetRuntimeHandle {
    fn dispatch(&self, request: Request) -> Result<(), AnyError> {
        self.requests
            .send(request)
            .map_err(|_| error("Facet runtime is no longer running"))
    }
}

/// [`FacetBundleModuleLoader`] over an embedded `JsRuntime`.
pub struct FacetRuntimeModuleLoader {
    handle: FacetRuntimeHandle,
    /// Kept so the thread outlives the loader; dropping it detaches.
    _worker_detached: std::thread::JoinHandle<()>,
}

impl FacetRuntimeModuleLoader {
    pub fn new(options: FacetRuntimeOptions) -> Self {
        let options = Arc::new(options);
        let (requests_tx, requests_rx) = mpsc::channel::<Request>();
        // Bridge the std channel into the async loop on the JS thread.
        let (async_tx, mut async_rx) = tokio::sync::mpsc::unbounded_channel::<Request>();
        let _forwarder = std::thread::Builder::new()
            .name("pi-runtime-facet-requests".to_string())
            .spawn(move || {
                for request in requests_rx {
                    if async_tx.send(request).is_err() {
                        break;
                    }
                }
            })
            .expect("pi-runtime: failed to spawn facet request forwarder");
        // Dropping the forwarder handle detaches it; it ends when all request
        // senders are gone.

        let thread_options = Arc::clone(&options);
        let worker = spawn_js_thread(move || async move {
            facet_runtime_loop(thread_options, &mut async_rx).await;
        });
        // Dropping the worker handle detaches it; the loop ends when its
        // request channel closes.

        Self {
            handle: FacetRuntimeHandle { requests: requests_tx },
            _worker_detached: worker,
        }
    }
}

#[async_trait::async_trait]
impl FacetBundleModuleLoader for FacetRuntimeModuleLoader {
    async fn load_entry(
        &self,
        module_path: &Path,
        external_imports: &[String],
    ) -> Result<Vec<Facet>, AnyError> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.handle
            .dispatch(Request::Evaluate {
                module_path: module_path.to_path_buf(),
                external_imports: external_imports.to_vec(),
                reply: reply_tx,
            })?;
        let described = reply_rx
            .await
            .map_err(|_| error("Facet runtime stopped before answering"))?
            .map_err(error)?;
        let mut facets = Vec::with_capacity(described.len());
        for (id, slot) in described {
            let handle = self.handle.clone();
            facets.push(Facet::new(id, move |env| {
                let (reply_tx, reply_rx) = mpsc::sync_channel(1);
                handle.dispatch(Request::Setup { slot, reply: reply_tx })?;
                let ops = reply_rx
                    .recv_timeout(SETUP_TIMEOUT)
                    .map_err(|_| error("Facet setup timed out"))?
                    .map_err(error)?;
                for op in ops {
                    let disposal = js_callback_disposal(handle.clone(), op.slot);
                    match op.kind.as_str() {
                        "own" => env.own(disposal)?,
                        "on_activate" => env.on_activate(disposal)?,
                        "on_deactivate" => env.on_deactivate(disposal)?,
                        other => {
                            return Err(error(format!(
                                "Facet setup recorded an unknown environment operation {other}"
                            )));
                        }
                    }
                }
                Ok(())
            }));
        }
        Ok(facets)
    }
}

/// `FacetEnvironment` lifecycle callbacks bridged back into JS as async
/// disposals (`() => void | Promise<void>` upstream).
fn js_callback_disposal(handle: FacetRuntimeHandle, slot: u32) -> pi_chord::facets::host::Disposal {
    Arc::new(move || {
        let handle = handle.clone();
        Box::pin(async move {
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            handle
                .dispatch(Request::Callback { slot, reply: reply_tx })?;
            reply_rx
                .await
                .map_err(|_| error("Facet runtime stopped before the callback finished"))?
                .map_err(error)
        }) as BoxFuture<Result<(), AnyError>>
    })
}

/// The long-lived JS thread: evaluates artifacts, registers facet generations,
/// and runs lifecycle callbacks.
async fn facet_runtime_loop(
    options: Arc<FacetRuntimeOptions>,
    requests: &mut tokio::sync::mpsc::UnboundedReceiver<Request>,
) {
    let loader = PiModuleLoader::new();
    let mut runtime = make_runtime(vec![], Some(Rc::clone(&loader)));
    if let Err(e) = runtime.execute_script("pi-facet-bootstrap.js", FACET_BOOTSTRAP) {
        tracing::error!(target: "pi::js", "facet bootstrap failed: {e}");
        return;
    }
    let mut generation: u64 = 0;

    loop {
        tokio::select! {
            biased;
            request = requests.recv() => {
                let Some(request) = request else { break };
                match request {
                    Request::Evaluate { module_path, external_imports, reply } => {
                        let result = evaluate_entry(&mut runtime, &loader, &options, &module_path, &external_imports, generation).await;
                        generation += 1;
                        let _ = reply.send(result);
                    }
                    Request::Setup { slot, reply } => {
                        let result = run_setup(&mut runtime, slot);
                        let _ = reply.send(result);
                    }
                    Request::Callback { slot, reply } => {
                        let result = run_callback(&mut runtime, slot).await;
                        let _ = reply.send(result);
                    }
                }
            }
            // Keep the event loop alive for async disposals/activations.
            polled = runtime.run_event_loop(PollEventLoopOptions::default()) => {
                if let Err(e) = polled {
                    tracing::warn!(target: "pi::js", "facet runtime event loop error: {e}");
                }
            }
        }
    }
}

/// Evaluate a CJS facet bundle artifact, register its ESM modules for this
/// generation, and describe the facet exports.
async fn evaluate_entry(
    runtime: &mut deno_core::JsRuntime,
    loader: &Rc<PiModuleLoader>,
    options: &FacetRuntimeOptions,
    module_path: &Path,
    external_imports: &[String],
    generation: u64,
) -> Result<Vec<(String, u32)>, String> {
    let source = std::fs::read_to_string(module_path)
        .map_err(|e| format!("Could not read facet bundle entry {}: {e}", module_path.display()))?;

    // Evaluate the artifact's CJS shell with the upstream CommonJS shape.
    // The artifact declares no imports; `require` mirrors the upstream
    // "undeclared external import" failure if the shell ever calls it.
    let name = module_path.to_string_lossy().to_string();
    let dir = module_path
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let wrapped = format!(
        "globalThis.__piModule = {{ exports: {{}} }};\n\
         const __piRequire = (specifier) => {{ throw new Error(\"Facet bundle required undeclared external import: \" + specifier); }};\n\
         (function (exports, require, module, __filename, __dirname) {{\n{source}\n}}).call(\n\
         \x20   globalThis.__piModule.exports,\n\
         \x20   globalThis.__piModule.exports,\n\
         \x20   __piRequire,\n\
         \x20   globalThis.__piModule,\n\
         \x20   {filename},\n\
         \x20   {dirname},\n\
         );\n\
         globalThis.__piArtifact = globalThis.__piModule.exports;",
        filename = crate::host::js_string_literal(&name),
        dirname = crate::host::js_string_literal(&dir),
    );
    if let Err(e) = runtime.execute_script(name, wrapped) {
        return Err(format!("{e}"));
    }
    // Plain CommonJS facets (the upstream `executeCommonJsModule` shape) are
    // described directly; pi-runtime's own bundle artifacts go through the
    // ESM module map below.
    let artifact = match execute_json(runtime, "__piExtractArtifact()", "the facet bundle artifact")? {
        artifact_json if artifact_json != "null" => {
            let artifact_value: serde_json::Value = serde_json::from_str(&artifact_json)
                .map_err(|e| format!("the facet bundle artifact is not valid JSON: {e}"))?;
            Some(parse_artifact(&artifact_value)?)
        }
        _ => None,
    };
    let Some(artifact) = artifact else {
        return describe_plain_exports(runtime, options);
    };

    // Register every bundled module for this generation.
    let prefix = format!("pi-facet://g{generation}");
    let mut canonical_for: BTreeMap<String, String> = BTreeMap::new();
    for (key, module_source) in &artifact.modules {
        let canonical = format!("{prefix}{key}");
        loader.register_virtual(canonical.clone(), VirtualSource::Js(module_source.clone()));
        loader.register_path_aliases(Path::new(key), &canonical);
        canonical_for.insert(key.clone(), canonical);
    }

    // External imports resolve to facade modules: registered values verbatim,
    // otherwise a namespace with the names the bundle imports (undefined).
    for specifier in external_imports {
        let names = artifact.external_names.get(specifier).cloned().unwrap_or_default();
        let source = match options.externals.get(specifier) {
            Some(exports) => crate::host::esm_facade_from_json(exports, &names),
            None => crate::host::esm_facade_from_json(&Value::Null, &names),
        };
        loader.register_virtual(specifier.clone(), VirtualSource::Js(source));
    }

    // Load and evaluate the entry module.
    let entry_key = &artifact.entry;
    let entry_canonical = canonical_for
        .get(entry_key)
        .ok_or_else(|| format!("Facet bundle artifact entry {entry_key} is missing from its modules"))?;
    let entry_specifier = ModuleSpecifier::parse(entry_canonical)
        .map_err(|e| format!("invalid facet module URL {entry_canonical}: {e}"))?;
    let module_id = runtime
        .load_main_es_module(&entry_specifier)
        .await
        .map_err(|e| format!("load: {e}"))?;
    let evaluated = runtime.mod_evaluate(module_id);
    runtime
        .run_event_loop(PollEventLoopOptions::default())
        .await
        .map_err(|e| e.to_string())?;
    evaluated.await.map_err(|e| e.to_string())?;

    // Hand the module namespace (the exports record) to the descriptor.
    let namespace = runtime
        .get_module_namespace(module_id)
        .map_err(|e| format!("namespace: {e}"))?;
    {
        deno_core::scope!(scope, runtime);
        let namespace = deno_core::v8::Local::new(scope, namespace);
        let context = scope.get_current_context();
        let global_object = context.global(scope);
        let target_key = deno_core::v8::String::new(scope, "__piExports")
            .ok_or("Could not allocate the facet exports key")?
            .into();
        global_object.set(scope, target_key, namespace.into());
    }
    describe_exports(runtime, options)
}

/// Describe `globalThis.__piExports`-shaped facet candidates; pi-chord's
/// `facets_from_module` produces the upstream error messages. The descriptor
/// maps back onto the upstream export shape: a function setup becomes a
/// non-null placeholder carrying its callback slot.
fn describe_exports(
    runtime: &mut deno_core::JsRuntime,
    options: &FacetRuntimeOptions,
) -> Result<Vec<(String, u32)>, String> {
    let described_json = execute_json(runtime, "__piDescribeFacets()", "facet exports")?;
    let described: Value = serde_json::from_str(&described_json)
        .map_err(|e| format!("Facet exports description is not JSON: {e}"))?;
    let exports_value = described
        .get("exports")
        .cloned()
        .unwrap_or(Value::Null);
    facets_from_module(&exports_value, &options.plugin_id, &options.entry_name)
        .map_err(|e| e.to_string())?;
    let mut facets = Vec::new();
    for candidate in exports_value
        .get("default")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        let id = candidate.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
        let slot = candidate
            .pointer("/setup/__piSlot")
            .and_then(Value::as_u64)
            .unwrap_or(u64::MAX) as u32;
        facets.push((id, slot));
    }
    Ok(facets)
}

/// Plain CommonJS facet modules (`module.exports = { default: … }`): point the
/// descriptor at the evaluated exports and validate.
fn describe_plain_exports(
    runtime: &mut deno_core::JsRuntime,
    options: &FacetRuntimeOptions,
) -> Result<Vec<(String, u32)>, String> {
    runtime
        .execute_script(
            "pi-facet-plain.js",
            "globalThis.__piExports = globalThis.__piModule.exports;".to_string(),
        )
        .map_err(|e| e.to_string())?;
    describe_exports(runtime, options)
}

fn parse_artifact(artifact: &Value) -> Result<Artifact, String> {
    let object = artifact
        .as_object()
        .ok_or("Facet bundle artifact is not an object")?;
    if object.get("format").and_then(Value::as_str) != Some(BUNDLE_FORMAT)
        || object.get("formatVersion").and_then(Value::as_u64) != Some(BUNDLE_FORMAT_VERSION)
    {
        return Err(format!(
            "Facet bundle artifact is not a {BUNDLE_FORMAT} v{BUNDLE_FORMAT_VERSION} bundle: {}",
            serde_json::to_string(&artifact).unwrap_or_default()
        ));
    }
    let entry = object
        .get("entry")
        .and_then(Value::as_str)
        .ok_or("Facet bundle artifact has no entry")?
        .to_string();
    let mut modules = BTreeMap::new();
    for (key, source) in object
        .get("modules")
        .and_then(Value::as_object)
        .ok_or("Facet bundle artifact has no modules")?
    {
        modules.insert(
            key.clone(),
            source.as_str().ok_or("Facet bundle artifact module is not a string")?.to_string(),
        );
    }
    let mut external_names = BTreeMap::new();
    if let Some(names) = object.get("externalNames").and_then(Value::as_object) {
        for (specifier, list) in names {
            external_names.insert(
                specifier.clone(),
                list.as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
            );
        }
    }
    Ok(Artifact { entry, modules, external_names })
}

struct Artifact {
    entry: String,
    modules: BTreeMap<String, String>,
    external_names: BTreeMap<String, Vec<String>>,
}

fn run_setup(runtime: &mut deno_core::JsRuntime, slot: u32) -> Result<Vec<EnvOp>, String> {
    // Run setup once, stash its result, and report whether it was async
    // (upstream rejects asynchronous setups).
    let script = format!(
        "(function () {{\n\
         \x20   const __piResult = __piRunSetup({slot});\n\
         \x20   globalThis.__piSetupResult = __piResult;\n\
         \x20   return String(__piResult instanceof Promise);\n\
         }})()"
    );
    let is_promise = execute_json(runtime, &script, "the setup result check")? == "true";
    if is_promise {
        return Err("Facet bundle entry setup must be synchronous".to_string());
    }
    let ops_json = execute_json(runtime, "__piTakeEnvOps()", "setup environment operations")?;
    serde_json::from_str(&ops_json).map_err(|e| format!("Facet setup recorded invalid operations: {e}"))
}

async fn run_callback(runtime: &mut deno_core::JsRuntime, slot: u32) -> Result<(), String> {
    // Drive the callback promise explicitly: `resolve` only watches, the
    // event loop must be polled for microtasks to run.
    runtime
        .execute_script(
            "pi-facet-callback-start.js",
            format!("globalThis.__piCallbackDone = undefined; __piRunCallback({slot});"),
        )
        .map_err(|e| e.to_string())?;
    loop {
        let state = execute_json(runtime, "String(globalThis.__piCallbackDone)", "the callback state")?;
        match state.as_str() {
            "undefined" => {}
            "ok" => return Ok(()),
            message => return Err(message.to_string()),
        }
        runtime
            .run_event_loop(PollEventLoopOptions::default())
            .await
            .map_err(|e| e.to_string())?;
    }
}

/// Run a JSON-expression script and return its string result.
fn execute_json(
    runtime: &mut deno_core::JsRuntime,
    expression: &str,
    what: &str,
) -> Result<String, String> {
    let global = runtime
        .execute_script("pi-facet-json.js", expression.to_string())
        .map_err(|e| e.to_string())?;
    deno_core::scope!(scope, runtime);
    let local = deno_core::v8::Local::new(scope, global);
    let string = local
        .to_string(scope)
        .ok_or_else(|| format!("{what} did not produce a string"))?;
    Ok(string.to_rust_string_lossy(scope))
}

/// Bootstrap evaluated once per facet runtime: artifact extraction, facet
/// export description, and the setup environment shim.
const FACET_BOOTSTRAP: &str = r#"
const isRecord = (value) => typeof value === "object" && value !== null && !Array.isArray(value);

// The evaluated CJS artifact registers itself here. Plain CommonJS facet
// modules (no pi-runtime bundle marker) report `null`.
globalThis.__piArtifact = undefined;
globalThis.__piExtractArtifact = () => {
	const artifact = globalThis.__piArtifact;
	if (
		!artifact ||
		artifact.format !== "pi-facet-esm-bundle" ||
		artifact.formatVersion !== 1 ||
		typeof artifact.entry !== "string" ||
		typeof artifact.modules !== "object" ||
		artifact.modules === null
	) {
		return "null";
	}
	return JSON.stringify({
		format: artifact.format,
		formatVersion: artifact.formatVersion,
		entry: artifact.entry,
		modules: artifact.modules,
		externalNames: artifact.externalNames ?? {},
	});
};

// Function slots (facet setups, lifecycle callbacks).
globalThis.__piSlots = [];
globalThis.__piEnvOps = [];
globalThis.__piDescribeFacets = () => {
	const exportsValue = globalThis.__piExports;
	if (!isRecord(exportsValue)) return JSON.stringify({ exports: null });
	const exported = exportsValue.default;
	const candidates = Array.isArray(exported) ? exported : [exported];
	return JSON.stringify({
		exports: {
			default: candidates.map((candidate) => {
				if (!isRecord(candidate)) return null;
				let setupSlot = null;
				if (typeof candidate.setup === "function") {
					setupSlot = globalThis.__piSlots.push(candidate.setup) - 1;
				}
				return {
					id: typeof candidate.id === "string" ? candidate.id : null,
					setup: setupSlot === null ? null : { __piSlot: setupSlot },
				};
			}),
		},
	});
};

const unsupported = (operation) => () => {
	throw new Error(
		"Facet setup from a JavaScript bundle cannot use the typed service operation " +
		operation + " in the Rust port; see the pi-runtime crate docs",
	);
};

globalThis.__piRunSetup = (slot) => {
	globalThis.__piEnvOps = [];
	const env = {
		own: (fn) => { globalThis.__piEnvOps.push({ kind: "own", slot: globalThis.__piSlots.push(fn) - 1 }); },
		onActivate: (fn) => { globalThis.__piEnvOps.push({ kind: "on_activate", slot: globalThis.__piSlots.push(fn) - 1 }); },
		onDeactivate: (fn) => { globalThis.__piEnvOps.push({ kind: "on_deactivate", slot: globalThis.__piSlots.push(fn) - 1 }); },
		use: unsupported("use"),
		observe: unsupported("observe"),
		provide: unsupported("provide"),
		provideMany: unsupported("provideMany"),
		replicatedState: unsupported("replicatedState"),
	};
	const setup = globalThis.__piSlots[slot];
	const result = setup(env);
	globalThis.__piTakeEnvOps = () => {
		const ops = JSON.stringify(globalThis.__piEnvOps);
		globalThis.__piEnvOps = [];
		return ops;
	};
	return result;
};

globalThis.__piRunCallback = (slot) => {
	Promise.resolve(globalThis.__piSlots[slot]()).then(
		() => {
			globalThis.__piCallbackDone = "ok";
		},
		(error) => {
			globalThis.__piCallbackDone =
				(error instanceof Error ? error.message : String(error));
		},
	);
};
"#;
