//! Port of `packages/protocol/src/protocol.ts`.
//!
//! Upstream validates with TypeBox schemas over dynamic JS values. In Rust the
//! wire type is [`pi_chord::json::JsonValue`] (strict JSON by construction), so
//! the schemas become hand-written structural validators over `serde_json`
//! objects with identical accept/reject behavior: literal/integer/string-shape
//! checks, strict (no additional properties) objects, and the canonical UUIDv4
//! server-id pattern. `OpaqueJsonValue` guards collapse into the value type.

use pi_chord::json::JsonValue;
use serde_json::Map;

pub const PROTOCOL_VERSION: i64 = 8;

/// Upstream `ProtocolValidationError`.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ProtocolValidationError(String);

impl ProtocolValidationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// Upstream `ServerId` pattern:
/// `^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`.
pub fn is_server_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    for (index, byte) in bytes.iter().enumerate() {
        let is_hex = byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase();
        match index {
            8 | 13 | 18 | 23 => {
                if *byte != b'-' {
                    return false;
                }
            }
            14 => {
                if *byte != b'4' {
                    return false;
                }
            }
            19 => {
                if !matches!(byte, b'8' | b'9' | b'a' | b'b') {
                    return false;
                }
            }
            _ => {
                if !is_hex {
                    return false;
                }
            }
        }
    }
    true
}

fn as_object<'a>(value: &'a JsonValue) -> Option<&'a Map<String, JsonValue>> {
    value.as_object()
}

/// Strict-object check: exactly `keys` present, no additional properties
/// (`StrictObject` with `additionalProperties: false`).
fn exact_keys<'a>(value: &'a JsonValue, keys: &[&str]) -> Option<&'a Map<String, JsonValue>> {
    let object = as_object(value)?;
    if object.len() != keys.len() || keys.iter().any(|key| !object.contains_key(*key)) {
        return None;
    }
    Some(object)
}

/// Strict-object check allowing optional keys (subset of `keys`, required keys
/// all present, nothing else).
fn keys_subset<'a>(value: &'a JsonValue, keys: &[&str]) -> Option<&'a Map<String, JsonValue>> {
    let object = as_object(value)?;
    if !object.keys().all(|key| keys.contains(&key.as_str())) {
        return None;
    }
    Some(object)
}

fn string_field<'a>(object: &'a Map<String, JsonValue>, key: &str) -> Option<&'a str> {
    object.get(key)?.as_str()
}

fn nonempty_string_field<'a>(object: &'a Map<String, JsonValue>, key: &str) -> Option<&'a str> {
    let value = string_field(object, key)?;
    if value.is_empty() {
        return None;
    }
    Some(value)
}

/// `Type.Integer({ minimum: 0 })`: integral, non-negative.
fn is_uint(value: &JsonValue) -> bool {
    match value {
        JsonValue::Number(number) => number.as_i64().map(|i| i >= 0).unwrap_or(false),
        _ => false,
    }
}

/// `Type.Literal(8)` (`===` semantics: `8.0` compares equal to `8`).
fn is_literal_version(value: &JsonValue) -> bool {
    value
        .as_f64()
        .map(|f| f == PROTOCOL_VERSION as f64)
        .unwrap_or(false)
}

/// `ProtocolErrorSchema`: `{ code: nonempty string, message: string }`.
fn validate_protocol_error(value: &JsonValue) -> bool {
    let Some(object) = exact_keys(value, &["code", "message"]) else {
        return false;
    };
    object
        .get("message")
        .map(JsonValue::is_string)
        .unwrap_or(false)
        && nonempty_string_field(object, "code").is_some()
}

/// `SessionTargetSchema`: `{ serverId, sessionId, attachmentId }`.
fn validate_session_target(value: &JsonValue) -> bool {
    let Some(object) = exact_keys(value, &["serverId", "sessionId", "attachmentId"]) else {
        return false;
    };
    is_server_id_wrapped(object)
        && nonempty_string_field(object, "sessionId").is_some()
        && nonempty_string_field(object, "attachmentId").is_some()
}

fn is_server_id_wrapped(object: &Map<String, JsonValue>) -> bool {
    string_field(object, "serverId")
        .map(is_server_id)
        .unwrap_or(false)
}

/// `ServerTargetSchema`: `{ serverId }`.
fn validate_server_target(value: &JsonValue) -> bool {
    let Some(object) = exact_keys(value, &["serverId"]) else {
        return false;
    };
    is_server_id_wrapped(object)
}

/// `RpcTargetSchema`: server target or session target.
fn validate_rpc_target(value: &JsonValue) -> bool {
    validate_server_target(value) || validate_session_target(value)
}

/// `ClientMessageSchema`: hello | request | cancel.
pub fn validate_client_message(value: &JsonValue) -> bool {
    let Some(object) = as_object(value) else {
        return false;
    };
    match string_field(object, "type") {
        Some("hello") => {
            let Some(object) = exact_keys(value, &["type", "version"]) else {
                return false;
            };
            is_uint(object.get("version").unwrap_or(&JsonValue::Null))
        }
        Some("request") => {
            let Some(object) = exact_keys(value, &["type", "id", "target", "call"]) else {
                return false;
            };
            nonempty_string_field(object, "id").is_some()
                && validate_rpc_target(object.get("target").unwrap_or(&JsonValue::Null))
            // `call` is `OpaqueJsonValueSchema` (unknown + strict JSON): any
            // `JsonValue` qualifies by construction.
        }
        Some("cancel") => {
            let Some(object) = exact_keys(value, &["type", "id", "target"]) else {
                return false;
            };
            nonempty_string_field(object, "id").is_some()
                && validate_rpc_target(object.get("target").unwrap_or(&JsonValue::Null))
        }
        _ => false,
    }
}

/// `ServerMessageSchema`: hello | hello_error | response | service_update | attachment.
pub fn validate_server_message(value: &JsonValue) -> bool {
    let Some(object) = as_object(value) else {
        return false;
    };
    match string_field(object, "type") {
        Some("hello") => {
            let Some(object) = exact_keys(value, &["type", "version", "serverId"]) else {
                return false;
            };
            is_literal_version(object.get("version").unwrap_or(&JsonValue::Null))
                && is_server_id_wrapped(object)
        }
        Some("hello_error") => {
            let Some(object) = exact_keys(value, &["type", "error"]) else {
                return false;
            };
            validate_protocol_error(object.get("error").unwrap_or(&JsonValue::Null))
        }
        Some("response") => {
            // `ok: true` branch: { type, id, ok, result? }.
            // `ok: false` branch: { type, id, ok: false, error }.
            let Some(object) = keys_subset(value, &["type", "id", "ok", "result", "error"]) else {
                return false;
            };
            if nonempty_string_field(object, "id").is_none() {
                return false;
            }
            match object.get("ok") {
                Some(JsonValue::Bool(true)) => {
                    // `error` belongs to the other union branch (additionalProperties: false).
                    object.get("error").is_none()
                }
                Some(JsonValue::Bool(false)) => {
                    object.len() == 4
                        && object.contains_key("error")
                        && validate_protocol_error(object.get("error").unwrap())
                }
                _ => false,
            }
        }
        Some("service_update") => {
            let Some(object) = exact_keys(value, &["type", "subscriptionId", "update"]) else {
                return false;
            };
            nonempty_string_field(object, "subscriptionId").is_some()
        }
        Some("attachment") => {
            let Some(object) = exact_keys(value, &["type", "attachment"]) else {
                return false;
            };
            let attachment = object.get("attachment").unwrap_or(&JsonValue::Null);
            attachment.is_null() || validate_session_target(attachment)
        }
        _ => false,
    }
}

pub fn parse_client_message(value: JsonValue) -> Result<JsonValue, ProtocolValidationError> {
    if !validate_client_message(&value) {
        return Err(ProtocolValidationError::new(
            "Invalid client protocol message",
        ));
    }
    Ok(value)
}

pub fn parse_server_message(value: JsonValue) -> Result<JsonValue, ProtocolValidationError> {
    if !validate_server_message(&value) {
        return Err(ProtocolValidationError::new(
            "Invalid server protocol message",
        ));
    }
    Ok(value)
}

pub fn is_supported_protocol_version(version: i64) -> bool {
    version == PROTOCOL_VERSION
}

#[cfg(test)]
mod tests {
    //! Port of `packages/protocol/test/protocol.test.ts`.
    //!
    //! Upstream's `rejects non-JSON opaque payloads` case (byte arrays, NaN,
    //! `undefined` properties, cycles) is unconstructible for
    //! [`pi_chord::json::JsonValue`], which is strict by construction; the
    //! remaining validation coverage is retained.
    use super::*;
    use serde_json::json;

    const SERVER_ID: &str = "00000000-0000-4000-8000-000000000001";

    fn client_hello(version: i64) -> JsonValue {
        json!({ "type": "hello", "version": version })
    }

    fn server_hello() -> JsonValue {
        json!({ "type": "hello", "version": PROTOCOL_VERSION, "serverId": SERVER_ID })
    }

    fn request_with(call: JsonValue) -> JsonValue {
        json!({
            "type": "request",
            "id": "request-1",
            "target": { "serverId": SERVER_ID },
            "call": call,
        })
    }

    fn expect_client_ok(message: JsonValue) -> JsonValue {
        match parse_client_message(message) {
            Ok(parsed) => parsed,
            Err(error) => panic!("expected client message to parse: {error}"),
        }
    }

    fn expect_client_err(message: JsonValue) {
        assert!(parse_client_message(message).is_err());
    }

    fn expect_server_ok(message: JsonValue) -> JsonValue {
        match parse_server_message(message) {
            Ok(parsed) => parsed,
            Err(error) => panic!("expected server message to parse: {error}"),
        }
    }

    fn expect_server_err(message: JsonValue) {
        assert!(parse_server_message(message).is_err());
    }

    #[test]
    fn negotiates_protocol_version_8() {
        assert_eq!(PROTOCOL_VERSION, 8);
        assert!(is_supported_protocol_version(8));
        assert!(!is_supported_protocol_version(7));
        // 8.5 is unrepresentable for i64; integer-only typing rejects it by
        // construction (upstream `isSupportedProtocolVersion(8.5)` is false).
    }

    #[test]
    fn accepts_integer_client_hello_versions_for_negotiation() {
        for version in [0, PROTOCOL_VERSION, PROTOCOL_VERSION + 1] {
            assert_eq!(
                expect_client_ok(client_hello(version)),
                client_hello(version)
            );
        }
    }

    #[test]
    fn rejects_an_invalid_client_hello() {
        expect_client_err(json!({ "type": "hello", "version": PROTOCOL_VERSION.to_string() }));
        expect_client_err(json!({ "type": "hello", "version": PROTOCOL_VERSION as f64 + 0.5 }));
        expect_client_err(json!({ "type": "hello", "version": PROTOCOL_VERSION, "extra": true }));
    }

    #[test]
    fn rejects_non_canonical_uuidv4_server_ids() {
        for server_id in [
            "",
            "server-1",
            "00000000-0000-7000-8000-000000000001",
            "00000000-0000-4000-7000-000000000001",
            "00000000-0000-4000-8000-00000000000A",
        ] {
            let mut message =
                request_with(json!({ "serviceId": "pi.models", "member": "list", "args": [] }));
            message["target"]["serverId"] = json!(server_id);
            expect_client_err(message);
        }
    }

    #[test]
    fn keeps_routed_request_and_event_payloads_opaque() {
        let request = json!({
            "type": "request",
            "id": "request-1",
            "target": {
                "serverId": SERVER_ID,
                "sessionId": "session-1",
                "attachmentId": "attachment-1",
            },
            "call": {
                "serviceId": "application.custom",
                "instance": { "key": "instance-1", "generation": 2 },
                "member": "invoke",
                "args": [{ "arbitrary": true }, ["opaque"]],
            },
        });
        assert_eq!(expect_client_ok(request.clone()), request);
        let arbitrary_call =
            json!({ "arbitrary": "strict JSON whose service meaning belongs to Chord" });
        let parsed = expect_client_ok(json!({
            "type": "request",
            "id": "request-1",
            "target": {
                "serverId": SERVER_ID,
                "sessionId": "session-1",
                "attachmentId": "attachment-1",
            },
            "call": arbitrary_call,
        }));
        assert_eq!(parsed["call"], arbitrary_call);
        let parsed = expect_server_ok(json!({
            "type": "service_update",
            "subscriptionId": "subscription-1",
            "update": { "applicationDefined": true },
        }));
        assert_eq!(parsed["update"], json!({ "applicationDefined": true }));
    }

    // Upstream `rejects non-JSON opaque payloads` is unconstructible: JsonValue
    // cannot contain byte arrays, non-finite numbers, undefined properties, or
    // cycles.

    #[test]
    fn validates_request_cancellation_envelopes() {
        let cancel =
            json!({ "type": "cancel", "id": "request-1", "target": { "serverId": SERVER_ID } });
        assert_eq!(expect_client_ok(cancel.clone()), cancel);
        expect_client_err(
            json!({ "type": "cancel", "id": "", "target": { "serverId": SERVER_ID } }),
        );
        expect_client_err(
            json!({ "type": "cancel", "id": "request-1", "target": { "serverId": SERVER_ID }, "extra": true }),
        );
    }

    #[test]
    fn validates_attachment_route_updates() {
        let attached = json!({
            "type": "attachment",
            "attachment": { "serverId": SERVER_ID, "sessionId": "session-1", "attachmentId": "attachment-1" },
        });
        let detached = json!({ "type": "attachment", "attachment": null });
        assert_eq!(expect_server_ok(attached.clone()), attached);
        assert_eq!(expect_server_ok(detached.clone()), detached);
        expect_server_err(json!({
            "type": "attachment",
            "attachment": { "serverId": SERVER_ID, "sessionId": "session-1", "attachmentId": "attachment-1", "extra": true },
        }));
        // Partial session target (upstream: `{ sessionId: "session-1" }`).
        expect_server_err(
            json!({ "type": "attachment", "attachment": { "sessionId": "session-1" } }),
        );
    }

    #[test]
    fn rejects_malformed_request_boundaries() {
        expect_client_err(json!({
            "type": "request",
            "id": "",
            "target": { "serverId": SERVER_ID },
            "call": { "serviceId": "pi.models", "member": "list", "args": [] },
        }));
        expect_client_err(json!({
            "type": "request",
            "id": "request-1",
            "target": { "serverId": SERVER_ID },
            "call": { "serviceId": "pi.models", "member": "list", "args": [] },
            "extra": true,
        }));
    }

    #[test]
    fn accepts_a_successful_void_response_without_a_result_field() {
        assert_eq!(
            expect_server_ok(json!({ "type": "response", "id": "request-1", "ok": true })),
            json!({ "type": "response", "id": "request-1", "ok": true }),
        );
    }

    #[test]
    fn rejects_malformed_server_boundaries() {
        expect_server_err(
            json!({ "type": "hello", "version": PROTOCOL_VERSION, "serverId": "server-1" }),
        );
        expect_server_err(
            json!({ "type": "response", "id": "request-1", "ok": true, "result": [], "extra": true }),
        );
        expect_server_err(
            json!({ "type": "response", "id": "request-1", "ok": false, "error": { "code": "", "message": "bad" } }),
        );
    }

    #[test]
    fn accepts_the_opaque_error_codes() {
        for code in [
            "wrong_server",
            "cancelled",
            "service_not_found",
            "application_error",
        ] {
            let message = json!({
                "type": "response",
                "id": "request-1",
                "ok": false,
                "error": { "code": code, "message": "safe" },
            });
            assert_eq!(expect_server_ok(message.clone()), message);
        }
    }

    #[test]
    fn rejects_unknown_messages_and_fields() {
        let mut snapshot = server_hello();
        snapshot["snapshot"] = json!({});
        expect_server_err(snapshot);
        expect_server_err(json!({ "type": "unknown", "event": {} }));
    }

    #[test]
    fn does_not_parse_json_strings_as_messages() {
        expect_client_err(json!(client_hello(PROTOCOL_VERSION).to_string()));
        expect_server_err(json!(server_hello().to_string()));
    }

    #[test]
    fn is_server_id_matches_the_canonical_uuidv4_pattern() {
        assert!(is_server_id(SERVER_ID));
        assert!(!is_server_id("server-1"));
    }
}
