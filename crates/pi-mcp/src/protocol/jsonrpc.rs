//! Port of `packages/mcp/src/protocol/jsonrpc.ts`.
//!
//! JSON-RPC 2.0 message types shared by the client and its transports, plus the
//! crate-wide [`McpError`] error type. Port deviation: upstream defines several
//! error *classes* across `protocol/jsonrpc.ts`, `transports/streamable-http.ts`
//! and `oauth/errors.ts`; Rust gets one enum ([`McpError`]) whose variants map
//! 1:1 to those classes (the HTTP and OAuth payloads stay in their upstream
//! modules and are wrapped here), so every fallible API is `Result`-based.

use std::fmt;

use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::oauth::errors::{
    McpOAuthAuthorizationRequiredError, OAuthError, OAuthInsecureEndpointError,
    OAuthIssuerMismatchError, OAuthRegistrationError,
};
use crate::transports::streamable_http::{AuthRequiredError, HttpError, SessionExpiredError};

/// JSON-RPC `"jsonrpc": "2.0"` version marker.
pub const JSONRPC_VERSION: &str = "2.0";

/// A JSON-RPC message id: a string or a (finite) number.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcId {
    String(String),
    Number(serde_json::Number),
}

impl JsonRpcId {
    pub fn number(value: u64) -> JsonRpcId {
        JsonRpcId::Number(value.into())
    }

    /// The id as `f64`, mirroring JS numeric ids (used by progress tokens).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonRpcId::Number(n) => n.as_f64(),
            JsonRpcId::String(_) => None,
        }
    }

    /// The id as a `u64` when it is a whole number.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            JsonRpcId::Number(n) => n.as_u64(),
            JsonRpcId::String(_) => None,
        }
    }
}

impl fmt::Display for JsonRpcId {
    /// `String(message.id)` from upstream.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonRpcId::String(s) => write!(f, "{s}"),
            JsonRpcId::Number(n) => write(f, n),
        }
    }
}

fn write(f: &mut fmt::Formatter<'_>, n: &serde_json::Number) -> fmt::Result {
    // Match JS `String(number)` for the ids this crate produces and parses.
    if let Some(v) = n.as_i64() {
        write!(f, "{v}")
    } else if let Some(v) = n.as_u64() {
        write!(f, "{v}")
    } else {
        write!(f, "{}", n.as_f64().unwrap_or(f64::NAN))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: JsonRpcId,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcErrorObject {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcSuccessResponse {
    pub jsonrpc: String,
    pub id: JsonRpcId,
    pub result: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcErrorResponse {
    pub jsonrpc: String,
    pub id: JsonRpcId,
    pub error: JsonRpcErrorObject,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JsonRpcResponse {
    Success(JsonRpcSuccessResponse),
    Error(JsonRpcErrorResponse),
}

impl JsonRpcResponse {
    pub fn id(&self) -> &JsonRpcId {
        match self {
            JsonRpcResponse::Success(r) => &r.id,
            JsonRpcResponse::Error(r) => &r.id,
        }
    }
}

impl Serialize for JsonRpcResponse {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            JsonRpcResponse::Success(r) => r.serialize(serializer),
            JsonRpcResponse::Error(r) => r.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for JsonRpcResponse {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        decode_response(value).map_err(|()| DeError::custom("Invalid JSON-RPC message"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum JsonRpcMessage {
    Request(JsonRpcRequest),
    Notification(JsonRpcNotification),
    Response(JsonRpcResponse),
}

impl Serialize for JsonRpcMessage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            JsonRpcMessage::Request(m) => m.serialize(serializer),
            JsonRpcMessage::Notification(m) => m.serialize(serializer),
            JsonRpcMessage::Response(m) => m.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for JsonRpcMessage {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        decode_message(value).map_err(|()| DeError::custom("Invalid JSON-RPC message"))
    }
}

impl From<JsonRpcRequest> for JsonRpcMessage {
    fn from(m: JsonRpcRequest) -> Self {
        JsonRpcMessage::Request(m)
    }
}

impl From<JsonRpcNotification> for JsonRpcMessage {
    fn from(m: JsonRpcNotification) -> Self {
        JsonRpcMessage::Notification(m)
    }
}

impl From<JsonRpcResponse> for JsonRpcMessage {
    fn from(m: JsonRpcResponse) -> Self {
        JsonRpcMessage::Response(m)
    }
}

pub const JSON_RPC_ERROR_CODES_PARSE_ERROR: i64 = -32700;
pub const JSON_RPC_ERROR_CODES_INVALID_REQUEST: i64 = -32600;
pub const JSON_RPC_ERROR_CODES_METHOD_NOT_FOUND: i64 = -32601;
pub const JSON_RPC_ERROR_CODES_INVALID_PARAMS: i64 = -32602;
pub const JSON_RPC_ERROR_CODES_INTERNAL_ERROR: i64 = -32603;

/// The crate-wide error. Variants correspond 1:1 to the upstream error classes:
/// [`McpError::JsonRpc`] is upstream `McpError`, [`McpError::ConnectionClosed`]
/// is `McpConnectionClosedError`, [`McpError::Timeout`] is `McpTimeoutError`,
/// [`McpError::Abort`] is `McpAbortError`, and the `Other`/`Network` variants
/// carry the message of upstream plain `Error`s (including `TypeError`s from
/// `fetch`).
#[derive(Debug, Clone, thiserror::Error)]
pub enum McpError {
    /// Upstream `McpError`: a JSON-RPC error with a code, message, and data.
    #[error("{message}")]
    JsonRpc {
        code: i64,
        message: String,
        data: Option<Value>,
    },
    /// Upstream `McpConnectionClosedError`.
    #[error("{message}")]
    ConnectionClosed { message: String },
    /// Upstream `McpTimeoutError`.
    #[error("MCP request timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
    /// Upstream `McpAbortError` (JS `name` is `"AbortError"`).
    #[error("{message}")]
    Abort { message: String },
    /// Upstream `McpHttpError`.
    #[error(transparent)]
    Http(#[from] HttpError),
    /// Upstream `McpAuthRequiredError`.
    #[error(transparent)]
    AuthRequired(#[from] AuthRequiredError),
    /// Upstream `McpSessionExpiredError`.
    #[error(transparent)]
    SessionExpired(#[from] SessionExpiredError),
    /// Upstream `OAuthError`.
    #[error(transparent)]
    OAuth(#[from] OAuthError),
    /// Upstream `OAuthIssuerMismatchError`.
    #[error(transparent)]
    IssuerMismatch(#[from] OAuthIssuerMismatchError),
    /// Upstream `OAuthInsecureEndpointError`.
    #[error(transparent)]
    InsecureEndpoint(#[from] OAuthInsecureEndpointError),
    /// Upstream `OAuthRegistrationError`.
    #[error(transparent)]
    Registration(#[from] OAuthRegistrationError),
    /// Upstream `McpOAuthAuthorizationRequiredError`.
    #[error(transparent)]
    OAuthAuthorizationRequired(#[from] McpOAuthAuthorizationRequiredError),
    /// A `fetch`-level failure (upstream `TypeError`); always retryable for SSE streams.
    #[error("{0}")]
    Network(String),
    /// Any other upstream `Error` (message-only).
    #[error("{0}")]
    Other(String),
}

impl McpError {
    /// Upstream `new McpError(code, message, data?)`.
    pub fn json_rpc(code: i64, message: impl Into<String>, data: Option<Value>) -> McpError {
        McpError::JsonRpc {
            code,
            message: message.into(),
            data,
        }
    }

    /// Upstream `new McpConnectionClosedError(message?)`.
    pub fn connection_closed(message: impl Into<String>) -> McpError {
        McpError::ConnectionClosed {
            message: message.into(),
        }
    }

    /// Upstream `new McpTimeoutError(timeoutMs)`.
    pub fn timeout(timeout_ms: u64) -> McpError {
        McpError::Timeout { timeout_ms }
    }

    /// Upstream `new McpAbortError(message?)`.
    pub fn abort(message: impl Into<String>) -> McpError {
        McpError::Abort {
            message: message.into(),
        }
    }

    /// Upstream `new Error(message)`.
    pub fn other(message: impl Into<String>) -> McpError {
        McpError::Other(message.into())
    }

    /// Upstream `invalidRequest`-coded `McpError` used by `invalid(...)`.
    pub fn invalid(message: impl Into<String>) -> McpError {
        McpError::json_rpc(JSON_RPC_ERROR_CODES_INVALID_REQUEST, message, None)
    }

    /// The JSON-RPC error code, when this is a JSON-RPC protocol error.
    pub fn code(&self) -> Option<i64> {
        match self {
            McpError::JsonRpc { code, .. } => Some(*code),
            _ => None,
        }
    }
}

/// Upstream `isObject`: an object but not an array.
pub fn is_object(value: &Value) -> bool {
    value.is_object()
}

/// Upstream `toError`: normalize any thrown value into a crate error.
pub fn to_error(value: impl Into<McpError>) -> McpError {
    value.into()
}

/// Upstream `isJsonRpcId`: a string or finite number.
pub fn is_json_rpc_id(value: &Value) -> bool {
    value.is_string() || value.as_f64().map(|n| n.is_finite()).unwrap_or(false)
}

/// Upstream `isJsonRpcRequest`.
pub fn is_json_rpc_request(message: &Value) -> bool {
    is_object(message)
        && message.get("jsonrpc").and_then(Value::as_str) == Some(JSONRPC_VERSION)
        && message.get("id").map(is_json_rpc_id).unwrap_or(false)
        && message.get("method").and_then(Value::as_str).is_some()
}

/// Upstream `isJsonRpcNotification`: no `id` field at all.
pub fn is_json_rpc_notification(message: &Value) -> bool {
    is_object(message)
        && message.get("jsonrpc").and_then(Value::as_str) == Some(JSONRPC_VERSION)
        && !message
            .as_object()
            .map(|m| m.contains_key("id"))
            .unwrap_or(false)
        && message.get("method").and_then(Value::as_str).is_some()
}

/// Upstream `isJsonRpcResponse`.
pub fn is_json_rpc_response(message: &Value) -> bool {
    if !is_object(message)
        || message.get("jsonrpc").and_then(Value::as_str) != Some(JSONRPC_VERSION)
        || !message.get("id").map(is_json_rpc_id).unwrap_or(false)
    {
        return false;
    }
    if message.get("result").is_some() {
        return message.get("error").is_none();
    }
    let Some(error) = message.get("error") else {
        return false;
    };
    is_object(error)
        && error.get("code").map(|c| c.is_number()).unwrap_or(false)
        && error.get("message").and_then(Value::as_str).is_some()
}

/// Upstream `parseJsonRpcMessage`.
pub fn parse_json_rpc_message(value: Value) -> Result<JsonRpcMessage, McpError> {
    decode_message(value).map_err(|()| McpError::invalid("Invalid JSON-RPC message"))
}

fn decode_message(value: Value) -> Result<JsonRpcMessage, ()> {
    if is_json_rpc_request(&value) {
        let request: JsonRpcRequest = serde_json::from_value(value).map_err(|_| ())?;
        return Ok(JsonRpcMessage::Request(request));
    }
    if is_json_rpc_notification(&value) {
        let notification: JsonRpcNotification = serde_json::from_value(value).map_err(|_| ())?;
        return Ok(JsonRpcMessage::Notification(notification));
    }
    decode_response(value).map(JsonRpcMessage::Response)
}

fn decode_response(value: Value) -> Result<JsonRpcResponse, ()> {
    if !is_json_rpc_response(&value) {
        return Err(());
    }
    if value.get("result").is_some() {
        let success: JsonRpcSuccessResponse = serde_json::from_value(value).map_err(|_| ())?;
        return Ok(JsonRpcResponse::Success(success));
    }
    let error_response: JsonRpcErrorResponse = serde_json::from_value(value).map_err(|_| ())?;
    Ok(JsonRpcResponse::Error(error_response))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // No upstream jsonrpc.test.ts; these guard the wire decode contract that the
    // client and transport tests rely on (upstream exercises the same predicates
    // indirectly through those suites).

    #[test]
    fn classifies_request_notification_and_response() {
        assert!(matches!(
            parse_json_rpc_message(json!({"jsonrpc": "2.0", "id": 1, "method": "ping"})).unwrap(),
            JsonRpcMessage::Request(_)
        ));
        assert!(matches!(
            parse_json_rpc_message(json!({"jsonrpc": "2.0", "method": "ping"})).unwrap(),
            JsonRpcMessage::Notification(_)
        ));
        assert!(matches!(
            parse_json_rpc_message(json!({"jsonrpc": "2.0", "id": 1, "result": {}})).unwrap(),
            JsonRpcMessage::Response(JsonRpcResponse::Success(_))
        ));
        assert!(matches!(
            parse_json_rpc_message(
                json!({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "x"}})
            )
            .unwrap(),
            JsonRpcMessage::Response(JsonRpcResponse::Error(_))
        ));
    }

    #[test]
    fn rejects_malformed_messages_like_upstream() {
        for value in [
            json!({}),
            json!({"jsonrpc": "1.0", "id": 1, "method": "x"}),
            json!({"jsonrpc": "2.0", "id": true, "method": "x"}),
            json!({"jsonrpc": "2.0", "id": 1, "result": {}, "error": {"code": 1, "message": "x"}}),
            json!({"jsonrpc": "2.0", "id": 1, "error": {"code": "x", "message": "x"}}),
        ] {
            let error = parse_json_rpc_message(value).unwrap_err();
            assert_eq!(error.to_string(), "Invalid JSON-RPC message");
            assert_eq!(error.code(), Some(JSON_RPC_ERROR_CODES_INVALID_REQUEST));
        }
    }

    #[test]
    fn round_trips_wire_shapes() {
        let message = json!({"jsonrpc": "2.0", "id": 1, "method": "m", "params": {"a": 1}});
        let parsed = parse_json_rpc_message(message.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), message);

        // Explicit `params: null` normalizes to absent (serde Option semantics);
        // JSON-RPC treats both as "no params".
        let notification = json!({"jsonrpc": "2.0", "method": "n", "params": null});
        let parsed = parse_json_rpc_message(notification).unwrap();
        assert_eq!(
            serde_json::to_value(&parsed).unwrap(),
            json!({"jsonrpc": "2.0", "method": "n"})
        );
        let notification = json!({"jsonrpc": "2.0", "method": "n", "params": [1, 2]});
        let parsed = parse_json_rpc_message(notification.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), notification);
    }
}
