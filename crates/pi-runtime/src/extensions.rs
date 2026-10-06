//! The `ExtensionHost` seam: loading and driving JS/TS extensions in an
//! embedded deno_core `JsRuntime`.
//!
//! Upstream (`packages/coding-agent/src/core/extensions/loader.ts` + `runner.ts`)
//! imports an extension entry with jiti, then calls its default-export factory
//! with an `ExtensionAPI` object (`pi`). Registration methods record handlers
//! and tool/command/flag definitions; every other method is an *action* that
//! delegates into the running agent. This module preserves that shape with a
//! JSON boundary:
//!
//! - **Registration** calls from JS land on [`ExtensionBackend::register`] with
//!   the method name (`on`, `registerTool`, `registerCommand`, …) and a
//!   JSON payload in which every function argument has been replaced by a
//!   slot reference `{"__piSlot": n}`. Slots are invoked later via
//!   [`ExtensionHost::call_slot`].
//! - **Actions** (`pi.sendMessage`, `ctx.sessionName`, … — everything that is
//!   not a registration method) land on [`ExtensionBackend::call`] with
//!   `ctx.*` methods prefixed `ctx.`.
//! - **Events** are dispatched with [`ExtensionHost::dispatch_event`]; each
//!   handler receives `(event, ctx)` where `ctx` is a live proxy over the
//!   backend, and its JSON result is reported per handler in registration
//!   order.
//!
//! Modules load through [`crate::loader::PiModuleLoader`]: TypeScript and JSX
//! entries transpile via swc; relative imports resolve against the entry;
//! bare specifiers resolve only to registered node-module shims (none by
//! default).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::mpsc;

use async_trait::async_trait;
use deno_core::{OpState, PollEventLoopOptions, op2};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::host::{make_runtime, spawn_js_thread};
use crate::loader::PiModuleLoader;

/// Errors raised by [`ExtensionHost`] implementations.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ExtensionHostError {
    pub message: String,
}

impl ExtensionHostError {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

/// Summary of one loaded extension, mirroring the upstream `Extension` record's
/// discoverable surface (paths, event subscriptions, tools, commands).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedExtension {
    /// Stable handle for further calls (distinct per load, so reloading the
    /// same path creates a fresh generation).
    pub key: String,
    /// The resolved entry path that was imported.
    pub path: PathBuf,
    /// Event names this extension subscribed to, in subscription order.
    pub events: Vec<String>,
    /// Registered tool names, in registration order.
    pub tools: Vec<String>,
    /// Registered command names, in registration order.
    pub commands: Vec<String>,
}

/// One handler's outcome inside a [`ExtensionHost::dispatch_event`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlerOutcome {
    /// JSON result when the handler returned a value (`null` for `undefined`).
    pub value: Option<Value>,
    /// Error message when the handler threw.
    pub error: Option<String>,
}

/// Rust side of the `pi` API surface. The coding-agent implements this; the
/// host routes JS calls to it.
#[async_trait]
pub trait ExtensionBackend: Send + Sync {
    /// A registration call from JS. `method` is the API method name without
    /// the `pi.` prefix (`on`, `registerTool`, `registerCommand`,
    /// `registerShortcut`, `registerFlag`, `registerMessageRenderer`,
    /// `registerEntryRenderer`, `registerMarkdownTransformer`,
    /// `registerVirtualModel`, `registerProvider`, `unsubscribe`). Function
    /// arguments in `payload` appear as `{"__piSlot": n}` slot references.
    fn register(&self, ext: &str, method: &str, payload: Value) -> Result<Value, String> {
        let _ = (ext, method, payload);
        Ok(Value::Null)
    }

    /// An action call from JS: any `pi.<method>` or `ctx.<method>` that is not
    /// a registration method. `ctx.*` methods arrive with the `ctx.` prefix.
    async fn call(&self, ext: &str, method: &str, args: Value) -> Result<Value, String>;
}

/// The extension loading and dispatch seam the coding-agent port codes against.
#[async_trait]
pub trait ExtensionHost: Send + Sync {
    /// Import a JS/TS extension entry from disk and run its default-export
    /// factory against a `pi` API object backed by `backend`.
    async fn load(&self, entry: &Path) -> Result<LoadedExtension, ExtensionHostError>;

    /// Invoke one extension's handlers for an event, in registration order.
    /// A handler's methods on `ctx` route to the backend with the `ctx.`
    /// prefix while it runs.
    async fn dispatch_event(
        &self,
        key: &str,
        event: &str,
        payload: Value,
    ) -> Result<Vec<HandlerOutcome>, ExtensionHostError>;

    /// Invoke a registered callback slot (a tool `execute`, command handler,
    /// …) with a JSON argument. The slot receives `(args, ctx)`.
    async fn call_slot(
        &self,
        key: &str,
        slot: u32,
        args: Value,
    ) -> Result<Option<Value>, ExtensionHostError>;

    /// Drop one extension's registrations and handlers.
    async fn dispose(&self, key: &str) -> Result<(), ExtensionHostError>;
}

/// Options for [`DenoExtensionHost`].
#[derive(Default)]
pub struct DenoExtensionHostOptions {
    /// Node-module shims exposed to extensions, keyed by specifier
    /// (`"node:path"`, an npm name, …). Values are the JSON exports. The
    /// runtime ships none by default — no node/npm compat beyond what the
    /// embedder registers.
    pub node_modules: BTreeMap<String, Value>,
}

enum Request {
    Load {
        entry: PathBuf,
        reply: tokio::sync::oneshot::Sender<Result<LoadedExtension, String>>,
    },
    Dispatch {
        key: String,
        event: String,
        payload: Value,
        reply: tokio::sync::oneshot::Sender<Result<Vec<HandlerOutcome>, String>>,
    },
    CallSlot {
        key: String,
        slot: u32,
        args: Value,
        reply: tokio::sync::oneshot::Sender<Result<Option<Value>, String>>,
    },
    Dispose {
        key: String,
        reply: tokio::sync::oneshot::Sender<Result<(), String>>,
    },
}

/// [`ExtensionHost`] over one embedded `JsRuntime` on a dedicated thread.
pub struct DenoExtensionHost {
    requests: mpsc::Sender<Request>,
    _worker_detached: std::thread::JoinHandle<()>,
}

impl DenoExtensionHost {
    pub fn new(backend: Arc<dyn ExtensionBackend>, options: DenoExtensionHostOptions) -> Self {
        let (requests_tx, requests_rx) = mpsc::channel::<Request>();
        let (async_tx, mut async_rx) = tokio::sync::mpsc::unbounded_channel::<Request>();
        let _forwarder = std::thread::Builder::new()
            .name("pi-runtime-ext-requests".to_string())
            .spawn(move || {
                for request in requests_rx {
                    if async_tx.send(request).is_err() {
                        break;
                    }
                }
            })
            .expect("pi-runtime: failed to spawn extension request forwarder");

        let worker = spawn_js_thread(move || async move {
            extension_runtime_loop(options.node_modules, backend, &mut async_rx).await;
        });

        Self {
            requests: requests_tx,
            _worker_detached: worker,
        }
    }
}

#[async_trait]
impl ExtensionHost for DenoExtensionHost {
    async fn load(&self, entry: &Path) -> Result<LoadedExtension, ExtensionHostError> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.requests
            .send(Request::Load { entry: entry.to_path_buf(), reply: reply_tx })
            .map_err(|_| ExtensionHostError::new("Extension host is no longer running"))?;
        reply_rx
            .await
            .map_err(|_| ExtensionHostError::new("Extension host stopped before answering"))?
            .map_err(ExtensionHostError::new)
    }

    async fn dispatch_event(
        &self,
        key: &str,
        event: &str,
        payload: Value,
    ) -> Result<Vec<HandlerOutcome>, ExtensionHostError> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.requests
            .send(Request::Dispatch {
                key: key.to_string(),
                event: event.to_string(),
                payload,
                reply: reply_tx,
            })
            .map_err(|_| ExtensionHostError::new("Extension host is no longer running"))?;
        reply_rx
            .await
            .map_err(|_| ExtensionHostError::new("Extension host stopped before answering"))?
            .map_err(ExtensionHostError::new)
    }

    async fn call_slot(
        &self,
        key: &str,
        slot: u32,
        args: Value,
    ) -> Result<Option<Value>, ExtensionHostError> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.requests
            .send(Request::CallSlot { key: key.to_string(), slot, args, reply: reply_tx })
            .map_err(|_| ExtensionHostError::new("Extension host is no longer running"))?;
        reply_rx
            .await
            .map_err(|_| ExtensionHostError::new("Extension host stopped before answering"))?
            .map_err(ExtensionHostError::new)
    }

    async fn dispose(&self, key: &str) -> Result<(), ExtensionHostError> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.requests
            .send(Request::Dispose { key: key.to_string(), reply: reply_tx })
            .map_err(|_| ExtensionHostError::new("Extension host is no longer running"))?;
        reply_rx
            .await
            .map_err(|_| ExtensionHostError::new("Extension host stopped before answering"))?
            .map_err(ExtensionHostError::new)
    }
}

async fn extension_runtime_loop(
    node_modules: BTreeMap<String, serde_json::Value>,
    backend: Arc<dyn ExtensionBackend>,
    requests: &mut tokio::sync::mpsc::UnboundedReceiver<Request>,
) {
    let loader = PiModuleLoader::new();
    for (specifier, exports) in node_modules {
        let names = exports
            .as_object()
            .map(|object| object.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        loader.register_virtual(
            specifier,
            crate::loader::VirtualSource::Js(crate::host::esm_facade_from_json(&exports, &names)),
        );
    }
    let mut runtime = make_runtime(vec![pi_extension_ops::init()], Some(Rc::clone(&loader)));
    {
        let op_state = runtime.op_state();
        op_state.borrow_mut().put(backend);
    }
    if let Err(e) = runtime.execute_script("pi-extension-bootstrap.js", EXTENSION_BOOTSTRAP) {
        tracing::error!(target: "pi::js", "extension bootstrap failed: {e}");
        return;
    }
    let mut next_key: u64 = 0;

    loop {
        tokio::select! {
            biased;
            request = requests.recv() => {
                let Some(request) = request else { break };
                match request {
                    Request::Load { entry, reply } => {
                        next_key += 1;
                        let result = load_extension(&mut runtime, &entry, next_key).await;
                        let _ = reply.send(result);
                    }
                    Request::Dispatch { key, event, payload, reply } => {
                        let result = dispatch_event(&mut runtime, &key, &event, &payload).await;
                        let _ = reply.send(result);
                    }
                    Request::CallSlot { key, slot, args, reply } => {
                        let result = call_slot(&mut runtime, &key, slot, &args).await;
                        let _ = reply.send(result);
                    }
                    Request::Dispose { key, reply } => {
                        let result = dispose_extension(&mut runtime, &key);
                        let _ = reply.send(result);
                    }
                }
            }
            polled = runtime.run_event_loop(PollEventLoopOptions::default()) => {
                if let Err(e) = polled {
                    tracing::warn!(target: "pi::js", "extension runtime event loop error: {e}");
                }
            }
        }
    }
}

async fn load_extension(
    runtime: &mut deno_core::JsRuntime,
    entry: &Path,
    key: u64,
) -> Result<LoadedExtension, String> {
    let key = format!("e{key}");
    // Import the entry dynamically from the runtime's bootstrap script: the
    // dynamic-import path drives every extension's dependency graph through
    // the event loop, for the first and all later loads.
    let entry_url = deno_core::resolve_path(
        &entry.to_string_lossy(),
        &std::env::current_dir().map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("Could not resolve extension entry {entry:?}: {e}"))?;
    runtime
        .execute_script(
            "pi-extension-import.js",
            format!(
                "__piDoImport({});",
                crate::host::js_string_literal(entry_url.as_str())
            ),
        )
        .map_err(|e| e.to_string())?;
    wait_for_done(runtime, "__piImportDone").await?;

    // Fetch the entry namespace's default export; check it is a factory
    // function.
    {
        deno_core::scope!(scope, runtime);
        let context = scope.get_current_context();
        let global_object = context.global(scope);
        let exports_key = deno_core::v8::String::new(scope, "__piImportNs")
            .ok_or("Could not allocate the imports key")?
            .into();
        let entry_namespace = global_object
            .get(scope, exports_key)
            .ok_or("Extension import produced no namespace")?;
        if !entry_namespace.is_object() {
            return Err(format!(
                "Extension module {} did not import as an ES module",
                entry.display()
            ));
        }
        let entry_object = deno_core::v8::Local::<deno_core::v8::Object>::try_from(entry_namespace)
            .map_err(|_| "Extension import namespace is not an object")?;
        let factory_key = deno_core::v8::String::new(scope, "default")
            .ok_or("Could not allocate the factory key")?
            .into();
        let default_value = entry_object
            .get(scope, factory_key)
            .ok_or("Extension module has no default export binding")?;
        if !default_value.is_function() {
            return Err(format!(
                "Extension module {} must default-export a factory function",
                entry.display()
            ));
        }
        let target_key = deno_core::v8::String::new(scope, "__piFactory")
            .ok_or("Could not allocate the factory key")?
            .into();
        global_object.set(scope, target_key, default_value);
    }

    // Build the pi API for this extension and invoke the factory. The factory
    // may be async: drive its completion through the event loop.
    let create = format!("globalThis.__piApi = __piCreateApi({});", crate::host::js_string_literal(&key));
    runtime
        .execute_script("pi-extension-api.js", create)
        .map_err(|e| e.to_string())?;
    runtime
        .execute_script(
            "pi-extension-factory.js",
            format!(
                "__piRunFactory({});",
                crate::host::js_string_literal(&key)
            ),
        )
        .map_err(|e| e.to_string())?;
    wait_for_done(runtime, "__piFactoryDone")
        .await
        .map_err(|e| format!("factory: {e}"))?;

    // Collect the summary record.
    let summary_json = execute_json(
        runtime,
        &format!("__piSummary({})", crate::host::js_string_literal(&key)),
        "the extension summary",
    )?;
    let mut summary: LoadedExtension = serde_json::from_str(&summary_json)
        .map_err(|e| format!("Extension summary is not valid JSON: {e}"))?;
    summary.key = key;
    summary.path = entry.to_path_buf();
    Ok(summary)
}

async fn dispatch_event(
    runtime: &mut deno_core::JsRuntime,
    key: &str,
    event: &str,
    payload: &Value,
) -> Result<Vec<HandlerOutcome>, String> {
    let script = format!(
        "__piDispatch({}, {}, {})",
        crate::host::js_string_literal(key),
        crate::host::js_string_literal(event),
        crate::host::js_string_literal(&crate::host::json_expr(payload)),
    );
    runtime
        .execute_script("pi-extension-dispatch.js", script)
        .map_err(|e| e.to_string())?;
    wait_for_done(runtime, "__piDispatchDone").await?;
    let json = execute_json(runtime, "globalThis.__piDispatchResult", "dispatch results")?;
    serde_json::from_str(&json).map_err(|e| format!("dispatch results are not valid JSON: {e}"))
}

async fn call_slot(
    runtime: &mut deno_core::JsRuntime,
    key: &str,
    slot: u32,
    args: &Value,
) -> Result<Option<Value>, String> {
    let script = format!(
        "__piCallSlot({}, {}, {})",
        crate::host::js_string_literal(key),
        slot,
        crate::host::js_string_literal(&crate::host::json_expr(args)),
    );
    runtime
        .execute_script("pi-extension-slot.js", script)
        .map_err(|e| e.to_string())?;
    wait_for_done(runtime, "__piCallSlotDone").await?;
    let json = execute_json(runtime, "globalThis.__piCallSlotResult", "the slot result")?;
    if json == "null" {
        return Ok(None);
    }
    serde_json::from_str(&json)
        .map(Some)
        .map_err(|e| format!("slot result is not valid JSON: {e}"))
}

fn dispose_extension(runtime: &mut deno_core::JsRuntime, key: &str) -> Result<(), String> {
    runtime
        .execute_script(
            "pi-extension-dispose.js",
            format!("__piDispose({})", crate::host::js_string_literal(key)),
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn execute_json(
    runtime: &mut deno_core::JsRuntime,
    expression: &str,
    what: &str,
) -> Result<String, String> {
    let global = runtime
        .execute_script("pi-extension-json.js", expression.to_string())
        .map_err(|e| e.to_string())?;
    deno_core::scope!(scope, runtime);
    let local = deno_core::v8::Local::new(scope, global);
    let string = local
        .to_string(scope)
        .ok_or_else(|| format!("{what} did not produce a string"))?;
    Ok(string.to_rust_string_lossy(scope))
}

/// Await a JS completion flag (`ok` or an error message) while driving the
/// event loop: promises only settle when microtasks run.
async fn wait_for_done(
    runtime: &mut deno_core::JsRuntime,
    done_global: &str,
) -> Result<(), String> {
    loop {
        let done = execute_json(
            runtime,
            &format!("String(globalThis.{done_global})"),
            done_global,
        )?;
        match done.as_str() {
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

/// Registration methods that carry handler callbacks to slotify.
#[allow(dead_code)] // mirrored in the bootstrap JS below
const REGISTER_METHODS: &[&str] = &[
    "registerTool",
    "registerCommand",
    "registerShortcut",
    "registerFlag",
    "registerMessageRenderer",
    "registerEntryRenderer",
    "registerMarkdownTransformer",
    "registerVirtualModel",
    "registerProvider",
];

#[op2]
#[serde]
fn op_pi_register(
    state: &mut OpState,
    #[string] key: String,
    #[string] method: String,
    #[serde] payload: serde_json::Value,
) -> Result<serde_json::Value, deno_error::JsErrorBox> {
    let backend = state.borrow::<Arc<dyn ExtensionBackend>>().clone();
    backend
        .register(&key, &method, payload)
        .map_err(deno_error::JsErrorBox::generic)
}

#[op2]
#[serde]
async fn op_pi_action(
    state: std::rc::Rc<std::cell::RefCell<OpState>>,
    #[string] key: String,
    #[string] method: String,
    #[serde] args: serde_json::Value,
) -> Result<serde_json::Value, deno_error::JsErrorBox> {
    let backend = state.borrow().borrow::<Arc<dyn ExtensionBackend>>().clone();
    backend
        .call(&key, &method, args)
        .await
        .map_err(deno_error::JsErrorBox::generic)
}

deno_core::extension!(
    pi_extension_ops,
    ops = [op_pi_register, op_pi_action],
);

/// Bootstrap evaluated once per extension runtime.
const EXTENSION_BOOTSTRAP: &str = r#"
const isRecord = (value) => typeof value === "object" && value !== null && !Array.isArray(value);
globalThis.__piSlots = [];
globalThis.__piExts = new Map();

// Replace function values with slot references so registrations stay JSON.
const slotify = (value, depth = 0) => {
	if (typeof value === "function") {
		return { __piSlot: globalThis.__piSlots.push(value) - 1 };
	}
	if (depth > 8 || !isRecord(value) && !Array.isArray(value)) return value;
	if (Array.isArray(value)) return value.map((item) => slotify(item, depth + 1));
	const out = {};
	for (const [key, item] of Object.entries(value)) out[key] = slotify(item, depth + 1);
	return out;
};

const callOp = (name, ...args) => Deno.core.ops.op_pi_register(...args.map((arg) => arg === undefined ? null : arg));

const makeProxy = (key, prefix) => new Proxy({}, {
	get(_target, method) {
		if (typeof method !== "string") return undefined;
		if (prefix === "" && method === "on") {
			return (event, handler) => {
				const slot = globalThis.__piSlots.push(handler) - 1;
				const record = globalThis.__piExts.get(key);
				record.events.push(event);
				(record.handlers[event] ??= []).push(slot);
				callOp("register", key, "on", { event, handler: { __piSlot: slot } });
				return () => {
					const slots = record.handlers[event] ?? [];
					const index = slots.indexOf(slot);
					if (index !== -1) slots.splice(index, 1);
					callOp("register", key, "unsubscribe", { event, handler: { __piSlot: slot } });
				};
			};
		}
		if (prefix === "" && REGISTER_METHOD_SET.has(method)) {
			return (...args) => {
				const record = globalThis.__piExts.get(key);
				const payload = args.map((arg) => slotify(arg));
				if (method === "registerTool") {
					const name = args[0] && typeof args[0].name === "string" ? args[0].name : null;
					if (name !== null) record.tools.push(name);
				} else if (method === "registerCommand") {
					if (typeof args[0] === "string") record.commands.push(args[0]);
				}
				return callOp("register", key, method, payload.length === 1 ? payload[0] : payload);
			};
		}
		return (...args) => {
			const json = args.map((arg) => arg === undefined ? null : arg);
			return Deno.core.ops.op_pi_action(key, prefix + method, json.length === 1 ? json[0] : json);
		};
	},
});

const REGISTER_METHOD_SET = new Set([
	"registerTool",
	"registerCommand",
	"registerShortcut",
	"registerFlag",
	"registerMessageRenderer",
	"registerEntryRenderer",
	"registerMarkdownTransformer",
	"registerVirtualModel",
	"registerProvider",
]);

globalThis.__piCreateApi = (key) => {
	globalThis.__piExts.set(key, {
		events: [],
		tools: [],
		commands: [],
		handlers: {},
	});
	return makeProxy(key, "");
};

// Context passed to handlers and slots: methods route to the backend with a
// `ctx.` prefix so the embedder can distinguish them from `pi.*` actions.
globalThis.__piCtx = (key) => makeProxy(key, "ctx.");

globalThis.__piDispatch = (key, event, payloadJson) => {
	globalThis.__piDispatchDone = undefined;
	(async () => {
		const record = globalThis.__piExts.get(key);
		if (!record) return "[]";
		const payload = JSON.parse(payloadJson);
		const outcomes = [];
		for (const slot of record.handlers[event] ?? []) {
			try {
				const result = await globalThis.__piSlots[slot](payload, globalThis.__piCtx(key));
				outcomes.push({ value: result === undefined ? null : JSON.parse(JSON.stringify(result)) });
			} catch (error) {
				outcomes.push({ error: error instanceof Error ? error.message : String(error) });
			}
		}
		return JSON.stringify(outcomes);
	})().then(
		(result) => {
			globalThis.__piDispatchResult = result;
			globalThis.__piDispatchDone = "ok";
		},
		(error) => {
			globalThis.__piDispatchDone = error instanceof Error ? error.message : String(error);
		},
	);
};

globalThis.__piCallSlot = (key, slot, argsJson) => {
	globalThis.__piCallSlotDone = undefined;
	(async () => {
		const record = globalThis.__piExts.get(key);
		if (!record) throw new Error("Extension " + key + " is not loaded");
		const args = JSON.parse(argsJson);
		const result = await globalThis.__piSlots[slot](args, globalThis.__piCtx(key));
		return result === undefined ? "null" : JSON.stringify(result);
	})().then(
		(result) => {
			globalThis.__piCallSlotResult = result;
			globalThis.__piCallSlotDone = "ok";
		},
		(error) => {
			globalThis.__piCallSlotDone = error instanceof Error ? error.message : String(error);
		},
	);
};

globalThis.__piDoImport = (specifier) => {
	globalThis.__piImportDone = undefined;
	import(specifier).then(
		(namespace) => {
			globalThis.__piImportNs = namespace;
			globalThis.__piImportDone = "ok";
		},
		(error) => {
			globalThis.__piImportDone = error instanceof Error ? error.message : String(error);
		},
	);
};

globalThis.__piRunFactory = (key) => {
	globalThis.__piFactoryDone = undefined;
	Promise.resolve(globalThis.__piFactory(globalThis.__piApi)).then(
		() => {
			globalThis.__piFactoryDone = "ok";
		},
		(error) => {
			globalThis.__piFactoryDone = error instanceof Error ? error.message : String(error);
		},
	);
};

globalThis.__piDispose = (key) => {
	globalThis.__piExts.delete(key);
};

globalThis.__piSummary = (key) => {
	const record = globalThis.__piExts.get(key);
	if (!record) throw new Error("Extension " + key + " is not loaded");
	return JSON.stringify({
		key,
		path: "",
		events: record.events,
		tools: record.tools,
		commands: record.commands,
	});
};
"#;
