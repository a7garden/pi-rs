//! Thrown-value plumbing shared by error handling (upstream `unknown`).

use std::sync::Arc;

use crate::http::ProviderHttpError;
use crate::utils::models_error::ModelsError;

/// Cheap-to-clone shared error (upstream a thrown `Error` object).
pub type SharedError = Arc<dyn std::error::Error + Send + Sync>;

/// Upstream arbitrary thrown value: either an `Error` or a non-error value
/// (upstream strings are the only non-Error values pi throws itself).
#[derive(Clone)]
pub enum ThrownValue {
	Error(SharedError),
	Text(String),
}

impl ThrownValue {
	pub fn error(error: impl std::error::Error + Send + Sync + 'static) -> ThrownValue {
		ThrownValue::Error(Arc::new(error))
	}

	pub fn message(message: impl Into<String>) -> ThrownValue {
		ThrownValue::Text(message.into())
	}

	/// Upstream `error.name`: the crate's concrete error types map to their
	/// upstream names; anything else is `"Error"`.
	pub fn error_name(&self) -> Option<String> {
		match self {
			ThrownValue::Text(_) => None,
			ThrownValue::Error(error) => {
				let name = if error.is::<ModelsError>() {
					"ModelsError"
				} else if error.is::<ProviderHttpError>() {
					"ProviderError"
				} else {
					"Error"
				};
				Some(name.to_string())
			}
		}
	}

	/// Upstream `error instanceof Error`.
	pub fn is_error(&self) -> bool {
		matches!(self, ThrownValue::Error(_))
	}
}

impl From<SharedError> for ThrownValue {
	fn from(value: SharedError) -> Self {
		ThrownValue::Error(value)
	}
}

impl From<String> for ThrownValue {
	fn from(value: String) -> Self {
		ThrownValue::Text(value)
	}
}

impl From<&str> for ThrownValue {
	fn from(value: &str) -> Self {
		ThrownValue::Text(value.to_string())
	}
}

impl std::fmt::Debug for ThrownValue {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			ThrownValue::Error(error) => write!(f, "ThrownValue::Error({error})"),
			ThrownValue::Text(text) => write!(f, "ThrownValue::Text({text})"),
		}
	}
}
