//! Wire-shape helpers for protocol messages (upstream constructs these object
//! literals inline in `server.ts`/`testing/client.ts`; pi-protocol keeps the
//! message type as [`pi_chord::JsonValue`], so the shapes are built as
//! JSON values with byte-compatible key order).

use pi_chord::JsonValue;
use serde_json::json;

/// Upstream `ServerHello`.
pub(crate) fn server_hello(server_id: &str) -> JsonValue {
    json!({ "type": "hello", "version": pi_protocol::PROTOCOL_VERSION, "serverId": server_id })
}

/// Upstream `ServerHelloError`.
pub(crate) fn hello_error(code: &str, message: &str) -> JsonValue {
    json!({ "type": "hello_error", "error": { "code": code, "message": message } })
}

pub(crate) fn response_ok_void(id: &str) -> JsonValue {
    json!({ "type": "response", "id": id, "ok": true })
}

pub(crate) fn response_ok_result(id: &str, result: JsonValue) -> JsonValue {
    json!({ "type": "response", "id": id, "ok": true, "result": result })
}

pub(crate) fn response_error(id: &str, code: &str, message: &str) -> JsonValue {
    json!({ "type": "response", "id": id, "ok": false, "error": { "code": code, "message": message } })
}

/// Upstream `service_update` envelope.
pub(crate) fn service_update(subscription_id: &str, update: JsonValue) -> JsonValue {
    json!({ "type": "service_update", "subscriptionId": subscription_id, "update": update })
}

/// Upstream `attachment` envelope (`SessionTarget | null`).
pub(crate) fn attachment(target: Option<JsonValue>) -> JsonValue {
    json!({ "type": "attachment", "attachment": target })
}

/// Upstream `SessionTarget`: `{ serverId, sessionId, attachmentId }`.
pub(crate) fn session_target(server_id: &str, session_id: &str, attachment_id: &str) -> JsonValue {
    json!({ "serverId": server_id, "sessionId": session_id, "attachmentId": attachment_id })
}

pub(crate) fn target_server_id(target: &JsonValue) -> &str {
    target
        .as_object()
        .and_then(|o| o.get("serverId"))
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
}

/// Upstream `"sessionId" in target` — session-scoped targets only.
pub(crate) fn target_session(target: &JsonValue) -> Option<(&str, &str)> {
    let object = target.as_object()?;
    Some((
        object.get("sessionId")?.as_str()?,
        object.get("attachmentId")?.as_str()?,
    ))
}

/// Upstream `hello.version`.
pub(crate) fn hello_version(hello: &JsonValue) -> Option<i64> {
    hello
        .as_object()
        .and_then(|o| o.get("version"))
        .and_then(JsonValue::as_i64)
}

/// Upstream envelope `id`.
pub(crate) fn envelope_id<'a>(message: &'a JsonValue) -> &'a str {
    message
        .as_object()
        .and_then(|o| o.get("id"))
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
}
