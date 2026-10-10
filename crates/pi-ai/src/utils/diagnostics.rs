//! Port of `packages/ai/src/utils/diagnostics.ts` and
//! `packages/ai/src/utils/models-error.ts`.
//!
//! Upstream inspects arbitrary thrown values; Rust carries failures as
//! [`ThrownValue`] (`Error(SharedError)` or `Text(String)`), so
//! `extractDiagnosticError` works on that type.

use crate::types::{
    AssistantMessage, AssistantMessageDiagnostic, DiagnosticCode, DiagnosticErrorInfo, JsonObject,
};
use crate::utils::thrown::ThrownValue;

/// Upstream `formatThrownValue`.
pub fn format_thrown_value(value: &ThrownValue) -> String {
    match value {
        ThrownValue::Error(error) => {
            let message = error.to_string();
            // Upstream falls back to `error.name` for empty messages.
            if message.is_empty() {
                value.error_name().unwrap_or_else(|| "Error".to_string())
            } else {
                message
            }
        }
        ThrownValue::Text(text) => text.clone(),
    }
}

/// Upstream `extractDiagnosticError`.
pub fn extract_diagnostic_error(error: &ThrownValue) -> DiagnosticErrorInfo {
    if !error.is_error() {
        return DiagnosticErrorInfo {
            name: Some("ThrownValue".to_string()),
            message: format_thrown_value(error),
            stack: None,
            code: None,
        };
    }

    let code = match error {
        ThrownValue::Error(err) => err
            .downcast_ref::<CodedError>()
            .map(|coded| coded.code.clone()),
        ThrownValue::Text(_) => None,
    };
    let name = error.error_name().filter(|name| name != "Error");
    let message = format_thrown_value(error);
    DiagnosticErrorInfo {
        name,
        message: if message.is_empty() {
            "Error".to_string()
        } else {
            message
        },
        stack: None,
        code,
    }
}

/// Error carrying a provider-style `code` field (`error.code` upstream).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct CodedError {
    pub message: String,
    pub code: DiagnosticCode,
}

/// Upstream `createAssistantMessageDiagnostic`.
pub fn create_assistant_message_diagnostic(
    diagnostic_type: &str,
    error: &ThrownValue,
    details: Option<JsonObject>,
) -> AssistantMessageDiagnostic {
    AssistantMessageDiagnostic {
        diagnostic_type: diagnostic_type.to_string(),
        timestamp: now_ms(),
        error: Some(extract_diagnostic_error(error)),
        details,
    }
}

/// Upstream `appendAssistantMessageDiagnostic`.
pub fn append_assistant_message_diagnostic(
    message: &mut AssistantMessage,
    diagnostic: AssistantMessageDiagnostic,
) {
    message
        .diagnostics
        .get_or_insert_with(Vec::new)
        .push(diagnostic);
}

pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}
