//! Integration tests: the `ExtensionHost` seam over an embedded `JsRuntime` —
//! JS and TypeScript extensions, event dispatch, action routing, and
//! discovery.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use parking_lot::Mutex;

use async_trait::async_trait;
use pi_runtime::extensions::{ExtensionBackend, ExtensionHost};
use pi_runtime::{DenoExtensionHost, DenoExtensionHostOptions};
use serde_json::{json, Value};

/// Records registrations and actions, answering actions from a fixed table.
#[derive(Default)]
struct FakeBackend {
    registrations: Mutex<Vec<(String, String, Value)>>,
    action_calls: Mutex<Vec<(String, String, Value)>>,
    action_results: Mutex<std::collections::HashMap<String, Value>>,
}

impl FakeBackend {
    fn with_action(mut self, method: &str, result: Value) -> Self {
        self.action_results
            .lock()
            .insert(method.to_string(), result);
        self
    }

    fn registrations(&self) -> Vec<(String, String, Value)> {
        self.registrations.lock().clone()
    }

    fn action_calls(&self) -> Vec<(String, String, Value)> {
        self.action_calls.lock().clone()
    }
}

#[async_trait]
impl ExtensionBackend for FakeBackend {
    fn register(&self, ext: &str, method: &str, payload: Value) -> Result<Value, String> {
        self.registrations.lock().push((ext.to_string(), method.to_string(), payload));
        Ok(Value::Null)
    }

    async fn call(&self, ext: &str, method: &str, args: Value) -> Result<Value, String> {
        self.action_calls.lock().push((ext.to_string(), method.to_string(), args));
        match self.action_results.lock().get(method) {
            Some(result) => Ok(result.clone()),
            None => Ok(Value::Null),
        }
    }
}

const JS_EXTENSION: &str = r#"
export default function (pi) {
	pi.registerTool({
		name: "greet",
		description: "Greet someone",
		parameters: { type: "object", properties: { name: { type: "string" } } },
		execute(args) {
			return { text: "hello " + args.name };
		},
	});
	pi.on("session_start", (event, ctx) => {
		return {
			greeted: event.path,
			via: typeof ctx.workingDirectory === "function" ? "ctx" : "root",
		};
	});
	pi.on("session_start", () => undefined);
}
"#;

const TS_EXTENSION: &str = r#"
interface ToolDefinition {
	name: string;
	execute: (args: { n: number }) => Promise<{ doubled: number }>;
}

const tool: ToolDefinition = {
	name: "double",
	execute: async (args) => ({ doubled: args.n * 2 }),
};

export default async function (pi: {
	registerTool: (tool: ToolDefinition) => void;
	sendMessage: (message: string) => Promise<void>;
}): Promise<void> {
	pi.registerTool(tool);
	await pi.sendMessage("typescript extension ready");
}
"#;

fn host(backend: Arc<FakeBackend>) -> DenoExtensionHost {
    DenoExtensionHost::new(backend, DenoExtensionHostOptions::default())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn loads_a_js_extension_and_dispatches_events_and_tools() {
    let backend = Arc::new(FakeBackend::default());
    let host = host(Arc::clone(&backend));

    let dir = tempfile::tempdir().unwrap();
    let entry = dir.path().join("greet.js");
    std::fs::write(&entry, JS_EXTENSION).unwrap();

    let loaded = host.load(&entry).await.unwrap();
    assert_eq!(loaded.tools, vec!["greet".to_string()]);
    assert_eq!(loaded.events, vec!["session_start".to_string(); 2]);

    // Registrations crossed the backend; function values became slots.
    let registrations = backend.registrations();
    let tool_registration = registrations
        .iter()
        .find(|(_, method, _)| method == "registerTool")
        .expect("registerTool reached the backend");
    let slot = tool_registration
        .2
        .pointer("/execute/__piSlot")
        .and_then(Value::as_u64)
        .expect("execute replaced by a slot reference");

    // Tool execute via the slot.
    let result = host
        .call_slot(&loaded.key, slot as u32, json!({ "name": "pi" }))
        .await
        .unwrap();
    assert_eq!(result, Some(json!({ "text": "hello pi" })));

    // Event dispatch: both handlers ran, in registration order.
    let outcomes = host
        .dispatch_event(&loaded.key, "session_start", json!({ "path": "/tmp" }))
        .await
        .unwrap();
    assert_eq!(outcomes.len(), 2);
    assert_eq!(outcomes[0].error, None);
    assert_eq!(
        outcomes[0].value,
        Some(json!({ "greeted": "/tmp", "via": "ctx" }))
    );
    // A bare `return` yields undefined, reported as no value.
    assert_eq!(outcomes[1].value, None);

    // Disposal drops the handlers.
    host.dispose(&loaded.key).await.unwrap();
    let outcomes = host
        .dispatch_event(&loaded.key, "session_start", json!({}))
        .await
        .unwrap();
    assert!(outcomes.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn loads_a_typescript_extension_and_routes_actions() {
    let backend = Arc::new(
        FakeBackend::default().with_action("sendMessage", json!(true)),
    );
    let host = host(Arc::clone(&backend));

    let dir = tempfile::tempdir().unwrap();
    let entry = dir.path().join("double.ts");
    std::fs::write(&entry, TS_EXTENSION).unwrap();

    // TypeScript transpiles through the module loader chain (swc).
    let loaded = host.load(&entry).await.unwrap();
    assert_eq!(loaded.tools, vec!["double".to_string()]);

    let registrations = backend.registrations();
    let tool_registration = registrations
        .iter()
        .find(|(_, method, _)| method == "registerTool")
        .expect("registerTool reached the backend");
    let slot = tool_registration
        .2
        .pointer("/execute/__piSlot")
        .and_then(Value::as_u64)
        .expect("execute replaced by a slot reference");
    let result = host
        .call_slot(&loaded.key, slot as u32, json!({ "n": 21 }))
        .await
        .unwrap();
    assert_eq!(result, Some(json!({ "doubled": 42 })));

    // The async factory's action call routed through the backend; a
    // single-argument action passes the argument directly.
    let action_calls = backend.action_calls();
    assert_eq!(
        action_calls,
        vec![(
            loaded.key.clone(),
            "sendMessage".to_string(),
            json!("typescript extension ready")
        )]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handler_context_methods_route_with_the_ctx_prefix() {
    let backend = Arc::new(
        FakeBackend::default().with_action("ctx.sessionName", json!("my-session")),
    );
    let host = host(Arc::clone(&backend));

    let dir = tempfile::tempdir().unwrap();
    let entry = dir.path().join("ctx.js");
    std::fs::write(
        &entry,
        r#"
export default function (pi) {
	pi.on("session_start", async (event, ctx) => {
		return { name: await ctx.sessionName() };
	});
}
"#,
    )
    .unwrap();
    let loaded = host.load(&entry).await.unwrap();

    let outcomes = host
        .dispatch_event(&loaded.key, "session_start", json!({}))
        .await
        .unwrap();
    assert_eq!(outcomes[0].value, Some(json!({ "name": "my-session" })));

    let action_calls = backend.action_calls();
    assert_eq!(action_calls[0].1, "ctx.sessionName");
    assert_eq!(action_calls[0].2, json!([]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handler_errors_are_reported_per_handler() {
    let backend = Arc::new(FakeBackend::default());
    let host = host(Arc::clone(&backend));
    let dir = tempfile::tempdir().unwrap();
    let entry = dir.path().join("throwing.js");
    std::fs::write(
        &entry,
        r#"
export default function (pi) {
	pi.on("session_start", () => {
		throw new Error("handler exploded");
	});
	pi.on("session_start", () => "after");
}
"#,
    )
    .unwrap();
    let loaded = host.load(&entry).await.unwrap();

    let outcomes = host
        .dispatch_event(&loaded.key, "session_start", json!({}))
        .await
        .unwrap();
    assert_eq!(outcomes.len(), 2);
    assert_eq!(
        outcomes[0].error.as_deref(),
        Some("handler exploded"),
        "the error is isolated to its handler"
    );
    assert_eq!(outcomes[1].value, Some(json!("after")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_factory_exports_fail_the_load() {
    let backend = Arc::new(FakeBackend::default());
    let host = host(Arc::clone(&backend));
    let dir = tempfile::tempdir().unwrap();
    let entry = dir.path().join("bad.js");
    std::fs::write(&entry, "export default 42;").unwrap();
    let error = host.load(&entry).await.unwrap_err();
    assert!(
        error.to_string().contains("factory function"),
        "unexpected error: {error}"
    );
}

#[test]
fn discovery_finds_direct_files_and_package_entries() {
    let dir = tempfile::tempdir().unwrap();
    // Direct files.
    std::fs::write(dir.path().join("alpha.ts"), "export default () => {};").unwrap();
    // Subdirectory with index.
    let pkg = dir.path().join("pkg");
    std::fs::create_dir(&pkg).unwrap();
    std::fs::write(pkg.join("index.ts"), "export default () => {};").unwrap();
    // Subdirectory with a pi manifest.
    let declared = dir.path().join("declared");
    std::fs::create_dir(&declared).unwrap();
    std::fs::write(
        declared.join("package.json"),
        r#"{ "pi": { "extensions": ["./custom.ts"] } }"#,
    )
    .unwrap();
    std::fs::write(declared.join("custom.ts"), "export default () => {};").unwrap();
    // Ignored: non-extension files, deeper recursion.
    std::fs::write(dir.path().join("notes.txt"), "not an extension").unwrap();
    let nested = pkg.join("nested");
    std::fs::create_dir(&nested).unwrap();
    std::fs::write(nested.join("deep.ts"), "export default () => {};").unwrap();

    let discovered = pi_runtime::discover_extensions_in_dir(dir.path());
    let mut names: Vec<String> = discovered
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    names.sort();
    assert_eq!(names, vec!["alpha.ts", "custom.ts", "index.ts"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_module_shims_resolve_registered_bare_specifiers() {
    use std::collections::BTreeMap;
    let backend = Arc::new(FakeBackend::default());
    let mut shims = BTreeMap::new();
    shims.insert("node:path".to_string(), json!({ "join": "shimmed" }));
    let host = DenoExtensionHost::new(
        Arc::clone(&backend) as Arc<dyn ExtensionBackend>,
        DenoExtensionHostOptions { node_modules: shims },
    );

    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("shims")).unwrap();
    let entry = dir.path().join("shimmed.js");
    std::fs::write(
        &entry,
        r#"
import path from "node:path";
export default function (pi) {
	pi.on("session_start", () => path.join);
}
"#,
    )
    .unwrap();
    let loaded = host.load(&entry).await.unwrap();
    let outcomes = host
        .dispatch_event(&loaded.key, "session_start", json!({}))
        .await
        .unwrap();
    assert_eq!(outcomes[0].value, Some(json!("shimmed")));

    // Relative imports resolve and link within the entry's file graph.
    let backend2 = Arc::new(FakeBackend::default());
    let host2 = DenoExtensionHost::new(
        backend2 as Arc<dyn ExtensionBackend>,
        DenoExtensionHostOptions::default(),
    );
    let graphed = dir.path().join("graphed.js");
    std::fs::write(
        &graphed,
        "\nimport pkg from \"./helper.js\";\nexport default function (pi) {\n\tpi.on(\"session_start\", () => pkg.marker);\n}\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("helper.js"),
        "export const marker = 7;\nexport default { marker };",
    )
    .unwrap();
    let loaded2 = host2.load(&graphed).await.unwrap();
    let outcomes = host2
        .dispatch_event(&loaded2.key, "session_start", json!({}))
        .await
        .unwrap();
    assert_eq!(outcomes[0].value, Some(json!(7)));

    // A missing relative import either fails the load loudly (the loader
    // rejects unresolvable file imports at resolve time) or, racing deno_core
    // 0.412's recursion, links with the binding unbound (`pkg` stays
    // undefined). Both outcomes keep the unresolvable import visible.
    let miss = dir.path().join("missing.js");
    std::fs::write(
        &miss,
        "\nimport pkg from \"./does-not-exist.js\";\nexport default () => typeof pkg;",
    )
    .unwrap();
    match host2.load(&miss).await {
        Err(error) => assert!(
            error.to_string().contains("does-not-exist"),
            "unexpected: {error}"
        ),
        Ok(loaded) => {
            let outcomes = host2
                .dispatch_event(&loaded.key, "session_start", json!({}))
                .await
                .unwrap();
            assert_eq!(outcomes[0].value, Some(json!("undefined")));
        }
    }
}

// Keep the atomic import exercised if the tests above change.
#[allow(dead_code)]
fn _witness(counter: &AtomicU32) {
    counter.fetch_add(1, Ordering::SeqCst);
}
