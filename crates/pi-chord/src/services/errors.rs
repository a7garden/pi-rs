//! Port of `packages/chord/src/services/errors.ts`.

use std::fmt;

/// Owned error value crossing chord boundaries (upstream `Error`).
pub type AnyError = Box<dyn std::error::Error + Send + Sync>;

/// Build an owned error with a message (upstream `new Error(message)`).
pub fn error(message: impl Into<String>) -> AnyError {
    ChordError {
        message: message.into(),
    }
    .into()
}

/// Plain chord error carrying a message; `Display` matches upstream `Error.message`.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ChordError {
    pub message: String,
}

/// Upstream `TypeError` (message-compatible; Rust has no TypeError type).
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct TypeError {
    pub message: String,
}

pub fn type_error(message: impl Into<String>) -> AnyError {
    TypeError {
        message: message.into(),
    }
    .into()
}

/// Upstream `AggregateError` with the upstream `Display` of its message.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct AggregateError {
    pub message: String,
    pub errors: Vec<AnyError>,
}

impl AggregateError {
    pub fn new(message: impl Into<String>, errors: Vec<AnyError>) -> AnyError {
        AggregateError {
            message: message.into(),
            errors,
        }
        .into()
    }
}

pub const REMOTE_SERVICE_ERROR_CODES: [&str; 8] = [
    "service_not_allowed",
    "service_not_found",
    "service_mode_mismatch",
    "service_member_not_found",
    "service_member_mismatch",
    "service_instance_not_found",
    "service_stale_instance",
    "service_invalid_value",
];

pub fn is_remote_service_error_code(value: &str) -> bool {
    REMOTE_SERVICE_ERROR_CODES.contains(&value)
}

/// Protocol-level remote service failure carrying a stable machine-readable code.
#[derive(Debug)]
pub struct RemoteServiceError {
    pub code: RemoteServiceErrorCode,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteServiceErrorCode {
    ServiceNotAllowed,
    ServiceNotFound,
    ServiceModeMismatch,
    ServiceMemberNotFound,
    ServiceMemberMismatch,
    ServiceInstanceNotFound,
    ServiceStaleInstance,
    ServiceInvalidValue,
}

impl RemoteServiceErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            RemoteServiceErrorCode::ServiceNotAllowed => "service_not_allowed",
            RemoteServiceErrorCode::ServiceNotFound => "service_not_found",
            RemoteServiceErrorCode::ServiceModeMismatch => "service_mode_mismatch",
            RemoteServiceErrorCode::ServiceMemberNotFound => "service_member_not_found",
            RemoteServiceErrorCode::ServiceMemberMismatch => "service_member_mismatch",
            RemoteServiceErrorCode::ServiceInstanceNotFound => "service_instance_not_found",
            RemoteServiceErrorCode::ServiceStaleInstance => "service_stale_instance",
            RemoteServiceErrorCode::ServiceInvalidValue => "service_invalid_value",
        }
    }

    pub fn from_str(value: &str) -> Option<RemoteServiceErrorCode> {
        Some(match value {
            "service_not_allowed" => RemoteServiceErrorCode::ServiceNotAllowed,
            "service_not_found" => RemoteServiceErrorCode::ServiceNotFound,
            "service_mode_mismatch" => RemoteServiceErrorCode::ServiceModeMismatch,
            "service_member_not_found" => RemoteServiceErrorCode::ServiceMemberNotFound,
            "service_member_mismatch" => RemoteServiceErrorCode::ServiceMemberMismatch,
            "service_instance_not_found" => RemoteServiceErrorCode::ServiceInstanceNotFound,
            "service_stale_instance" => RemoteServiceErrorCode::ServiceStaleInstance,
            "service_invalid_value" => RemoteServiceErrorCode::ServiceInvalidValue,
            _ => return None,
        })
    }
}

impl fmt::Display for RemoteServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RemoteServiceError {}

pub fn remote_service_error(code: RemoteServiceErrorCode, message: impl Into<String>) -> AnyError {
    RemoteServiceError {
        code,
        message: message.into(),
    }
    .into()
}
