//! Shared embedded-JsRuntime plumbing for every pi-runtime surface.
//!
//! deno_core's `JsRuntime` is `!Send` and single-threaded, so each JS surface
//! (codemode worker, facet loader, extension host) owns one dedicated OS thread
//! running a current-thread tokio runtime. This module holds the pieces they
//! all share: op registration, runtime construction, the tracing console
//! bridge, and JS-error shaping.

use std::future::Future;
use std::rc::Rc;

use deno_core::error::JsError;
use deno_core::{Extension, JsRuntime, RuntimeOptions};

use crate::loader::PiModuleLoader;

/// `console.*` bridge: extension and facet scripts log through `tracing`
/// (upstream: host `console` in the Node process). Codemode scripts do not use
/// this op — their `console` is the prelude's output-item channel.
#[deno_core::op2(fast)]
fn op_runtime_console(#[string] text: &str) {
    tracing::info!(target: "pi::js", "{text}");
}

deno_core::extension!(
    pi_runtime_console,
    ops = [op_runtime_console],
);

/// Ops every runtime gets (console bridge); consumers append their own.
pub(crate) fn base_extensions() -> Vec<Extension> {
    vec![pi_runtime_console::init()]
}

/// Build a runtime with the given extra extensions and module loader.
pub(crate) fn make_runtime(
    extra: Vec<Extension>,
    module_loader: Option<Rc<PiModuleLoader>>,
) -> JsRuntime {
    let mut extensions = base_extensions();
    extensions.extend(extra);
    let module_loader: Option<Rc<dyn deno_core::ModuleLoader>> =
        module_loader.map(|loader| loader as Rc<dyn deno_core::ModuleLoader>);
    JsRuntime::new(RuntimeOptions {
        extensions,
        module_loader,
        ..Default::default()
    })
}

/// Extract `(name, message, stack)` from a deno_core `JsError`, shaped the way
/// the upstream worker's `describeException` expects inputs.
pub(crate) fn js_error_parts(error: &JsError) -> (String, String, Option<String>) {
    let name = error.name.clone().unwrap_or_else(|| "Error".to_string());
    let message = error.message.clone().unwrap_or_default();
    let stack = error.stack.clone().map(|stack| stack.trim_end().to_string());
    (name, message, stack)
}

/// Shape a script error as the codemode protocol's JSON error string
/// (upstream `describeException`).
pub(crate) fn script_error_json_string(error: &JsError) -> String {
    let (name, message, stack) = js_error_parts(error);
    pi_codemode::runtime::worker::script_error_json_string(&name, &message, stack.as_deref())
}

/// Run an async body on a dedicated OS thread that owns a current-thread tokio
/// runtime. The future is created and driven on that thread, so it does not
/// need to be `Send` even though it holds a `JsRuntime`.
pub(crate) fn spawn_js_thread<F, Fut>(body: F) -> std::thread::JoinHandle<()>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = ()> + 'static,
{
    std::thread::Builder::new()
        .name("pi-runtime-js".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("pi-runtime: failed to build worker tokio runtime");
            runtime.block_on(body());
        })
        .expect("pi-runtime: failed to spawn worker thread")
}

/// Convert a `serde_json::Value` into a JS expression that evaluates to that
/// value (raw JSON is a valid JS expression; JSON has no `undefined`).
pub(crate) fn json_expr(value: &serde_json::Value) -> String {
    serde_json::to_string(value).expect("pi-runtime: JSON serialization cannot fail")
}

/// Convert a Rust string into a JS string literal expression.
pub(crate) fn js_string_literal(value: &str) -> String {
    json_expr(&serde_json::Value::String(value.to_string()))
}

/// SHA-256 as lowercase hex (esbuild-style output hashing).
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Build an ESM facade exposing JSON exports as `default` plus named exports
/// for each key that is a valid identifier (used for node-module shims and
/// facet external imports).
pub(crate) fn esm_facade_from_json(exports: &serde_json::Value, names: &[String]) -> String {
    let mut source = format!(
        "const __piExports = {};\nexport default __piExports;\n",
        json_expr(exports)
    );
    for name in names {
        if is_js_identifier(name) {
            source.push_str(&format!(
                "export const {name} = __piExports[{}];\n",
                js_string_literal(name)
            ));
        }
    }
    source
}

/// Whether `name` is a valid JavaScript identifier.
pub(crate) fn is_js_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(first) if first.is_alphabetic() || first == '_' || first == '$')
        && chars.all(|rest| rest.is_alphanumeric() || rest == '_' || rest == '$')
}
