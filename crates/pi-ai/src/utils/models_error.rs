//! Port of `packages/ai/src/utils/models-error.ts`.

use serde::{Deserialize, Serialize};

use crate::utils::diagnostics::format_thrown_value;
use crate::utils::thrown::ThrownValue;

/// Machine-readable error codes (`ModelsErrorCode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelsErrorCode {
	#[serde(rename = "model_source")]
	ModelSource,
	#[serde(rename = "model_validation")]
	ModelValidation,
	#[serde(rename = "provider")]
	Provider,
	#[serde(rename = "stream")]
	Stream,
	#[serde(rename = "auth")]
	Auth,
	#[serde(rename = "oauth")]
	OAuth,
}

impl ModelsErrorCode {
	pub fn as_str(&self) -> &'static str {
		match self {
			ModelsErrorCode::ModelSource => "model_source",
			ModelsErrorCode::ModelValidation => "model_validation",
			ModelsErrorCode::Provider => "provider",
			ModelsErrorCode::Stream => "stream",
			ModelsErrorCode::Auth => "auth",
			ModelsErrorCode::OAuth => "oauth",
		}
	}
}

/// `ModelsError`: message-carrying error with a stable code. Callers surface
/// `error.message` only, so the underlying reason is folded into the message
/// (`withCauseDetail`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ModelsError {
	pub code: ModelsErrorCode,
	pub message: String,
}

impl ModelsError {
	pub fn new(code: ModelsErrorCode, message: impl Into<String>) -> ModelsError {
		ModelsError { code, message: message.into() }
	}

	/// Upstream constructor with `cause`: the cause detail is appended to the
	/// message when not already contained.
	pub fn with_cause(
		code: ModelsErrorCode,
		message: impl Into<String>,
		cause: Option<&ThrownValue>,
	) -> ModelsError {
		let message = message.into();
		let detail = match cause {
			None => message,
			Some(cause) => {
				let detail = format_thrown_value(cause).trim().to_string();
				if detail.is_empty() || message.contains(&detail) {
					message
				} else {
					format!("{message}: {detail}")
				}
			}
		};
		ModelsError { code, message: detail }
	}
}
