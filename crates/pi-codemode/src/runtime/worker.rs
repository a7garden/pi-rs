//! Mirrors upstream `src/runtime/worker.ts` (worker-thread entry).
//!
//! Upstream, this module is the worker entry point: importing it starts a worker that runs one
//! script inside a fresh QuickJS VM (a separate wasm instance), relays tool calls and output to
//! the host, and reports the result. The host terminates the worker when the script settles,
//! times out, or is aborted; the worker exists so that a spinning script never blocks the host
//! thread.
//!
//! In the Rust core, the worker side is the [`crate::runtime::host::CodeModeWorkerFactory`] /
//! [`crate::runtime::host::WorkerHandle`] trait surface (pi-runtime implements it with a real
//! QuickJS/deno worker). What is portable here is the worker-side error-shaping helper used by
//! both sides of the protocol: script errors cross the wire as JSON-encoded
//! `{ name?, message, stack? }`.

use serde_json::Value;

/// JSON-encode a script exception the way the upstream worker does
/// (`describeException`): `{"name": ..., "message": ..., "stack": ...}` where `stack`, when
/// present, is `head\n{stack}` with `head` being `"Name: message"` (or `"Name"` without a
/// message), and the stack trimmed of trailing whitespace.
pub fn script_error_json(name: &str, message: &str, stack: Option<&str>) -> Value {
    let head = if message.is_empty() {
        name.to_string()
    } else {
        format!("{name}: {message}")
    };
    let stack = stack.map(|stack| stack.trim_end());
    let stack_value = match stack {
        Some(stack) if !stack.is_empty() => Value::String(format!("{head}\n{stack}")),
        _ => Value::String(head),
    };
    serde_json::json!({
        "name": name,
        "message": message,
        "stack": stack_value,
    })
}

/// Convenience: serialize [`script_error_json`] for the wire.
pub fn script_error_json_string(name: &str, message: &str, stack: Option<&str>) -> String {
    script_error_json(name, message, stack).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn shapes_errors_like_the_upstream_worker() {
        assert_eq!(
            script_error_json("TypeError", "boom 1", Some("  at codemode.js:2\n")),
            json!({ "name": "TypeError", "message": "boom 1", "stack": "TypeError: boom 1\n  at codemode.js:2" })
        );
        assert_eq!(
            script_error_json("Error", "", None),
            json!({ "name": "Error", "message": "", "stack": "Error" })
        );
    }
}
