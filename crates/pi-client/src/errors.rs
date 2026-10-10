//! Port of `packages/client/src/errors.ts`.
//!
//! Upstream error classes become Rust error structs with identical `Display`
//! messages. `DisconnectedError` keeps an optional cause chain (upstream
//! `cause`); identity checks (`error instanceof DisconnectedError`) become
//! [`AnyError::downcast_ref`] checks.

use pi_chord::json::JsonValue;
use pi_chord::services::errors::AnyError;

/// Upstream `ServerError`: a typed failure reported by the server. The upstream
/// constructor takes a validated `ProtocolError`; the Rust protocol port
/// represents protocol errors as strict-JSON `{ code, message }` objects, so
/// [`ServerError::from_protocol`] extracts the same fields.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ServerError {
    /// Upstream `ServerError.code`.
    pub code: String,
    /// Upstream inherited `Error.message`.
    pub message: String,
}

impl ServerError {
    /// Builds a `ServerError` from a protocol error object (`{ code, message }`).
    pub fn from_protocol(value: &JsonValue) -> Option<ServerError> {
        let object = value.as_object()?;
        let code = object.get("code")?.as_str()?.to_string();
        let message = object.get("message")?.as_str()?.to_string();
        Some(ServerError { code, message })
    }
}

/// Upstream `DisconnectedError`.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct DisconnectedError {
    /// Upstream `Error.message`.
    pub message: String,
    /// Upstream `Error.cause`.
    #[source]
    pub cause: Option<AnyError>,
}

impl DisconnectedError {
    pub fn new(message: impl Into<String>) -> DisconnectedError {
        DisconnectedError {
            message: message.into(),
            cause: None,
        }
    }

    pub fn with_cause(message: impl Into<String>, cause: AnyError) -> DisconnectedError {
        DisconnectedError {
            message: message.into(),
            cause: Some(cause),
        }
    }
}

/// Upstream `ClientDisposedError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("Client is disposed")]
pub struct ClientDisposedError;

/// Upstream `toError`: box any error value. Rust callers already hold owned
/// errors, so this is the identity-ish [`AnyError`] coercion point.
pub fn to_error(error: AnyError) -> AnyError {
    error
}

/// Upstream `toDisconnectedError`: pass through `DisconnectedError` identity,
/// otherwise wrap as its cause with the cause's message.
pub fn to_disconnected_error(error: AnyError) -> DisconnectedError {
    if let Some(disconnected) = error.downcast_ref::<DisconnectedError>() {
        // Re-box the same error value while satisfying the owned return type by
        // cloning the (cheap) message; the original is dropped.
        return DisconnectedError {
            message: disconnected.message.clone(),
            cause: disconnected
                .cause
                .as_ref()
                .map(crate::client::clone_any_error),
        };
    }
    DisconnectedError {
        message: error.to_string(),
        cause: Some(error),
    }
}
