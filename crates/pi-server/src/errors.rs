//! Port of `packages/server/src/errors.ts`.
//!
//! Upstream subclasses of `ServerError` become constructors on the single
//! [`ServerError`] struct carrying the same wire codes and default messages
//! (Rust downcasts operate on one type; tests assert wire code and message).

use std::fmt;

use pi_chord::RemoteServiceErrorCode;

/// Upstream `ServerOperationErrorCode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerOperationErrorCode {
    Remote(RemoteServiceErrorCode),
    WrongServer,
    SessionNotFound,
    SessionAmbiguous,
    SessionNotAttached,
    ServerDraining,
}

impl ServerOperationErrorCode {
    pub fn as_str(&self) -> &str {
        match self {
            ServerOperationErrorCode::Remote(code) => code.as_str(),
            ServerOperationErrorCode::WrongServer => "wrong_server",
            ServerOperationErrorCode::SessionNotFound => "session_not_found",
            ServerOperationErrorCode::SessionAmbiguous => "session_ambiguous",
            ServerOperationErrorCode::SessionNotAttached => "session_not_attached",
            ServerOperationErrorCode::ServerDraining => "server_draining",
        }
    }
}

pub const INTERNAL_SERVER_ERROR_MESSAGE: &str = "Internal server error";

/// A host or lifecycle error that can safely cross the protocol boundary
/// (upstream `ServerError` and its subclasses).
#[derive(Debug)]
pub struct ServerError {
    pub code: ServerOperationErrorCode,
    pub message: String,
}

impl ServerError {
    pub fn new(code: ServerOperationErrorCode, message: impl Into<String>) -> Self {
        ServerError {
            code,
            message: message.into(),
        }
    }

    /// Upstream `WrongServerError`.
    pub fn wrong_server() -> Self {
        Self::new(
            ServerOperationErrorCode::WrongServer,
            "Request was addressed to another server",
        )
    }

    /// Upstream `SessionNotFoundError` (default message).
    pub fn session_not_found() -> Self {
        Self::new(
            ServerOperationErrorCode::SessionNotFound,
            "Session was not found",
        )
    }

    /// Upstream `SessionNotFoundError` with a custom message.
    pub fn session_not_found_with(message: impl Into<String>) -> Self {
        Self::new(ServerOperationErrorCode::SessionNotFound, message)
    }

    /// Upstream `SessionAmbiguousError`.
    pub fn session_ambiguous() -> Self {
        Self::new(
            ServerOperationErrorCode::SessionAmbiguous,
            "Session ID matches more than one session",
        )
    }

    /// Upstream `SessionNotAttachedError`.
    pub fn session_not_attached() -> Self {
        Self::new(
            ServerOperationErrorCode::SessionNotAttached,
            "Session is not attached to this client",
        )
    }

    /// Upstream `ServerDrainingError`.
    pub fn server_draining() -> Self {
        Self::new(
            ServerOperationErrorCode::ServerDraining,
            "Server is draining",
        )
    }
}

impl fmt::Display for ServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ServerError {}
