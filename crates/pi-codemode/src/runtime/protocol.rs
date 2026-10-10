//! Mirrors upstream `src/runtime/protocol.ts`: messages between the host (main thread) and the
//! worker. Tool arguments, results, and values cross as JSON strings: the worker passes them
//! into and out of the QuickJS VM as strings and never builds structured values itself.

use std::sync::atomic::AtomicI32;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::CodemodeOutputItem;
use crate::wasm::CodemodeWasmModule;

/// Per-execution worker payload (upstream `workerData`).
#[derive(Serialize, Deserialize)]
pub struct WorkerData {
    pub code: String,
    /// `js_name` is the identifier the script uses; `description` is listed in `ALL_TOOLS`.
    pub tools: Vec<ToolInfo>,
    pub globals: Vec<GlobalInfo>,
    /// Compiled `quickjs-wasi` module. Opaque; the runtime that spawned the worker can interpret
    /// it (upstream passes it via structured clone, which shares the compiled code).
    #[serde(skip)]
    pub wasm: Option<CodemodeWasmModule>,
    pub memory_limit_bytes: Option<u64>,
    /// Snapshot for `load()`: key to JSON text.
    #[serde(default)]
    pub store: serde_json::Map<String, Value>,
    /// One atomic the host sets to non-zero before terminating the worker. The VM's interrupt
    /// handler polls it, because worker termination cannot stop a thread spinning in wasm.
    #[serde(skip)]
    pub interrupt: Arc<AtomicI32>,
}

/// `tools` entry of [`WorkerData`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolInfo {
    pub name: String,
    pub js_name: String,
    #[serde(default)]
    pub description: String,
}

/// `globals` entry of [`WorkerData`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GlobalInfo {
    pub name: String,
    #[serde(default)]
    pub spread: bool,
}

/// JSON-encoded `{ name?, message, stack? }` of an error thrown by the script.
pub type ScriptErrorJson = String;

/// The success half of a `done` message (upstream two variants of the `done` tag).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DoneOutcome {
    /// `writes` is a JSON array of `[key, json]` for `store()` and `[key]` for deletions.
    Ok {
        value: Option<String>,
        writes: String,
    },
    Err {
        error: ScriptErrorJson,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerToHostMessage {
    Call {
        id: u64,
        target: CallTarget,
        name: String,
        args: Option<String>,
    },
    Output {
        item: CodemodeOutputItem,
    },
    #[serde(rename = "done")]
    Done {
        ok: bool,
        #[serde(flatten)]
        outcome: DoneOutcome,
    },
    /// The VM failed outside the script's control, for example a wasm trap.
    Crash {
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallTarget {
    Tool,
    Global,
}

/// `payload` is the JSON result when `ok`, otherwise the error message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HostToWorkerMessage {
    Result {
        id: u64,
        ok: bool,
        payload: Option<String>,
    },
}

/// Structural guard matching upstream `isWorkerToHostMessage`: checks the `type` tag without
/// fully validating the payload shape.
pub fn is_worker_to_host_message(value: &Value) -> bool {
    value
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|ty| matches!(ty, "call" | "output" | "done" | "crash"))
}

/// Structural guard matching upstream `isHostToWorkerMessage`.
pub fn is_host_to_worker_message(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("result")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn worker_to_host_messages_round_trip_with_upstream_json_shape() {
        let call: WorkerToHostMessage = serde_json::from_value(json!({
            "type": "call", "id": 3, "target": "tool", "name": "echo", "args": "1"
        }))
        .unwrap();
        assert_eq!(
            call,
            WorkerToHostMessage::Call {
                id: 3,
                target: CallTarget::Tool,
                name: "echo".into(),
                args: Some("1".into())
            }
        );
        assert_eq!(
            serde_json::to_value(&call).unwrap(),
            json!({ "type": "call", "id": 3, "target": "tool", "name": "echo", "args": "1" })
        );

        let done: WorkerToHostMessage = serde_json::from_value(json!({
            "type": "done", "ok": true, "value": "\"x\"", "writes": "[]"
        }))
        .unwrap();
        assert_eq!(
            serde_json::to_value(&done).unwrap(),
            json!({ "type": "done", "ok": true, "value": "\"x\"", "writes": "[]" })
        );

        let failure: WorkerToHostMessage = serde_json::from_value(
            json!({ "type": "done", "ok": false, "error": "{\"message\":\"nope\"}" }),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&failure).unwrap(),
            json!({ "type": "done", "ok": false, "error": "{\"message\":\"nope\"}" })
        );

        let crash: WorkerToHostMessage =
            serde_json::from_value(json!({ "type": "crash", "message": "trap" })).unwrap();
        assert_eq!(
            serde_json::to_value(&crash).unwrap(),
            json!({ "type": "crash", "message": "trap" })
        );

        let output: WorkerToHostMessage = serde_json::from_value(json!({
            "type": "output", "item": { "type": "image", "data": "AAAA", "mimeType": "image/png" }
        }))
        .unwrap();
        assert_eq!(
            serde_json::to_value(&output).unwrap(),
            json!({ "type": "output", "item": { "type": "image", "data": "AAAA", "mimeType": "image/png" } })
        );
    }

    #[test]
    fn host_to_worker_messages_round_trip() {
        let message = HostToWorkerMessage::Result {
            id: 7,
            ok: true,
            payload: Some("null".into()),
        };
        assert_eq!(
            serde_json::to_value(&message).unwrap(),
            json!({ "type": "result", "id": 7, "ok": true, "payload": "null" })
        );
        assert!(is_host_to_worker_message(&json!({ "type": "result" })));
        assert!(!is_host_to_worker_message(&json!({ "type": "call" })));
    }

    #[test]
    fn tag_guards_reject_unknown_types() {
        assert!(is_worker_to_host_message(&json!({ "type": "call" })));
        assert!(is_worker_to_host_message(&json!({ "type": "output" })));
        assert!(is_worker_to_host_message(&json!({ "type": "done" })));
        assert!(is_worker_to_host_message(&json!({ "type": "crash" })));
        assert!(!is_worker_to_host_message(&json!({ "type": "result" })));
        assert!(!is_worker_to_host_message(&json!({ "other": true })));
        assert!(!is_worker_to_host_message(&json!(null)));
    }
}
