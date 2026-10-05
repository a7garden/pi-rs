//! Port of `packages/ai/src/utils/error-body.ts`.
//!
//! Shared normalization for provider HTTP error objects. Upstream probes the
//! SDK error-object field shapes (Mistral `statusCode`, `openai`/`@google/genai`
//! `status` + `error`, AWS Bedrock `$metadata`/`$response`) because the Node
//! SDKs hide the HTTP status and raw body under provider-specific names. The
//! Rust HTTP seam folds those into [`crate::http::ProviderHttpError`] fields
//! (`status`/`headers`/`body`) at construction time (crate docs, deviation
//! D15), so normalization reads the fields directly and keeps the rest of the
//! upstream behavior: trimming, the 4000-char cap, `messageCarriesBody`
//! detection and the composed display strings.

use crate::http::ProviderHttpError;
use serde_json::Value;

/// Upstream `MAX_PROVIDER_ERROR_BODY_CHARS`.
pub const MAX_PROVIDER_ERROR_BODY_CHARS: usize = 4000;

/// Upstream `NormalizedProviderError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedProviderError {
	/// HTTP status code, when one could be extracted from the error.
	pub status: Option<u16>,
	/// Raw HTTP body reason, already trimmed and truncated to the cap.
	pub body: Option<String>,
	/// `error.message`, or `safeJsonStringify(error)` for a non-`Error` throw.
	pub message: String,
	/// True when `message` already contains the body (no separate body to add).
	pub message_carries_body: bool,
}

/// Upstream `normalizeProviderError`.
pub fn normalize_provider_error(error: &ProviderHttpError) -> NormalizedProviderError {
	let body = extract_body(error);
	let message_carries_body = match &body {
		Some(body) => error.message.contains(body.as_str()),
		None => true,
	};
	NormalizedProviderError {
		status: error.status,
		body,
		message: error.message.clone(),
		message_carries_body,
	}
}

/// Raw body reason: trimmed, empty-object-free and truncated to the cap
/// (upstream `extractBody`). An empty parsed body object (`"{}"`) is treated
/// as no body so it does not surface as `"{}"` (deviation D15).
fn extract_body(error: &ProviderHttpError) -> Option<String> {
	let body_text = error.body.as_ref()?;
	let trimmed = body_text.trim();
	if trimmed.is_empty() || trimmed == "{}" {
		return None;
	}
	Some(truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS))
}

/// Compose a display string from a normalized error. When the message already
/// carries the body (Anthropic / `@google/genai` happy path) or no body/status
/// was extracted, the message is returned unchanged. Otherwise the status and
/// body are surfaced, with an optional provider prefix.
///
/// - no prefix: `"<status>: <body>"`
/// - prefix:    `"<prefix> (<status>): <body>"`
///
/// Upstream `formatProviderError`.
pub fn format_provider_error(norm: &NormalizedProviderError, prefix: Option<&str>) -> String {
	if norm.message_carries_body || norm.status.is_none() || norm.body.is_none() {
		return match (prefix, norm.status) {
			(Some(prefix), Some(status)) => format!("{} ({}): {}", prefix, status, norm.message),
			_ => norm.message.clone(),
		};
	}
	let body = norm.body.as_deref().expect("body checked above");
	match prefix {
		Some(prefix) => format!("{} ({}): {}", prefix, norm.status.expect("status checked above"), body),
		None => format!("{}: {}", norm.status.expect("status checked above"), body),
	}
}

/// Upstream `truncateErrorText`. Lengths are Unicode scalar values (Rust
/// `char`s) where upstream counts UTF-16 code units (deviation D15).
pub fn truncate_error_text(text: &str, max_chars: usize) -> String {
	let total = text.chars().count();
	if total <= max_chars {
		return text.to_string();
	}
	let kept: String = text.chars().take(max_chars).collect();
	format!("{}... [truncated {} chars]", kept, total - max_chars)
}

/// Upstream `safeJsonStringify`: serialize a JSON value, falling back to its
/// display form when serialization fails (which `serde_json::Value` only
/// reaches through non-string map keys — impossible — so the fallback mirrors
/// the upstream non-JSON-thrown branch).
pub fn safe_json_stringify(value: &Value) -> String {
	serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
}

#[cfg(test)]
mod tests {
	use super::*;
	use serde_json::json;

	fn error(message: &str, status: Option<u16>, body: Option<&str>) -> ProviderHttpError {
		let mut error = match status {
			Some(status) => ProviderHttpError::from_response(message, status, body.map(str::to_string)),
			None => ProviderHttpError::new(message),
		};
		if let Some(body) = body {
			if status.is_none() {
				error.body = Some(body.to_string());
			}
		}
		error
	}

	// -- normalizeProviderError ------------------------------------------------

	#[test]
	fn extracts_status_and_body_from_a_mistral_shaped_error() {
		let norm = normalize_provider_error(&error(
			"Mistral request failed",
			Some(403),
			Some(r#"{"error":"blocked by gateway WAF"}"#),
		));
		assert_eq!(norm.status, Some(403));
		assert_eq!(norm.body.as_deref(), Some(r#"{"error":"blocked by gateway WAF"}"#));
		assert!(!norm.message_carries_body);
	}

	#[test]
	fn reads_the_parsed_body_off_an_openai_apierror_when_the_message_is_opaque() {
		// makeMessage(status, error, message) yields "<status> status code (no body)"
		// when the parsed body is unparsed; the Rust seam keeps the parsed body text
		// on ProviderHttpError.body (deviation D15).
		let norm = normalize_provider_error(&error(
			"403 status code (no body)",
			Some(403),
			Some(r#"{"error":"blocked by gateway WAF"}"#),
		));
		assert_eq!(norm.status, Some(403));
		assert_eq!(norm.body.as_deref(), Some(r#"{"error":"blocked by gateway WAF"}"#));
		assert!(!norm.message_carries_body);
	}

	#[test]
	fn preserves_the_message_when_the_sdk_already_folds_the_body_into_it() {
		let body = json!({"error": {"code": 403, "message": "Permission denied"}});
		let norm = normalize_provider_error(&error(
			&body.to_string(),
			Some(403),
			None,
		));
		assert_eq!(norm.status, Some(403));
		assert!(norm.message_carries_body);
		assert_eq!(norm.message, body.to_string());
	}

	#[test]
	fn extracts_status_and_body_from_a_bedrock_shaped_service_exception() {
		let norm = normalize_provider_error(&error(
			"UnknownError",
			Some(403),
			Some(r#"{"message":"blocked by gateway WAF"}"#),
		));
		assert_eq!(norm.status, Some(403));
		assert_eq!(norm.body.as_deref(), Some(r#"{"message":"blocked by gateway WAF"}"#));
		assert!(!norm.message_carries_body);
	}

	#[test]
	fn ignores_unreadable_response_streams_instead_of_serializing_internals() {
		// Upstream ignores a Bedrock `$response.body` node stream and a class-instance
		// web ReadableStream alike; the Rust seam represents both as "no body"
		// (deviation D15), so the real message survives with `messageCarriesBody`.
		let norm = normalize_provider_error(&error(
			"Invocation of model ID anthropic.claude-opus-5 with on-demand throughput isn't supported.",
			Some(400),
			None,
		));
		assert_eq!(norm.status, Some(400));
		assert_eq!(norm.body, None);
		assert!(norm.message.contains("on-demand throughput isn't supported"));
		assert!(norm.message_carries_body);
	}

	#[test]
	fn ignores_class_instance_bodies_without_a_pipe_method() {
		// Not every SDK response wrapper is a node stream; serializing wrapper
		// classes yields internals-noise. Same "no body" surface as above.
		let norm = normalize_provider_error(&error(
			"Input is too long for requested model.",
			Some(400),
			None,
		));
		assert_eq!(norm.status, Some(400));
		assert_eq!(norm.body, None);
		assert!(norm.message.contains("Input is too long"));
		assert!(norm.message_carries_body);
	}

	#[test]
	fn ignores_class_instance_error_fields_instead_of_serializing_them() {
		let norm = normalize_provider_error(&error("TLS handshake failed", Some(502), None));
		assert_eq!(norm.body, None);
		assert_eq!(norm.message, "TLS handshake failed");
		assert!(norm.message_carries_body);
	}

	#[test]
	fn still_surfaces_a_plain_parsed_json_body_object() {
		let norm = normalize_provider_error(&error(
			"400 status code (no body)",
			Some(400),
			Some(r#"{"message":"schema validation failed","field":"tools[0]"}"#),
		));
		assert_eq!(
			norm.body.as_deref(),
			Some(r#"{"message":"schema validation failed","field":"tools[0]"}"#)
		);
		assert!(!norm.message_carries_body);
	}

	#[test]
	fn json_stringifies_a_non_error_thrown_value() {
		// Upstream `normalizeProviderError(value)` for a non-`Error` throw returns
		// `safeJsonStringify(value)` as the message; Rust throws are typed, so the
		// stringify half is asserted directly (deviation D15).
		assert_eq!(safe_json_stringify(&json!({"reason": "boom"})), r#"{"reason":"boom"}"#);
		let norm = NormalizedProviderError {
			status: None,
			body: None,
			message: safe_json_stringify(&json!({"reason": "boom"})),
			message_carries_body: false,
		};
		assert_eq!(norm.status, None);
		assert_eq!(norm.body, None);
		assert_eq!(norm.message, r#"{"reason":"boom"}"#);
		assert!(!norm.message_carries_body);
	}

	#[test]
	fn treats_an_empty_parsed_body_object_as_no_body() {
		let norm = normalize_provider_error(&error("403 status code (no body)", Some(403), Some("{}")));
		assert_eq!(norm.body, None);
		assert!(norm.message_carries_body);
	}

	#[test]
	fn truncates_the_body_at_the_cap() {
		let long_body = "x".repeat(MAX_PROVIDER_ERROR_BODY_CHARS + 50);
		let norm = normalize_provider_error(&error("failed", Some(500), Some(&long_body)));
		let body = norm.body.expect("body present");
		assert!(body.contains("... [truncated 50 chars]"));
		assert!(body.len() < long_body.len());
	}

	#[test]
	fn sets_message_carries_body_when_the_message_already_contains_the_extracted_body() {
		let norm = normalize_provider_error(&error("500: upstream exploded", Some(500), Some("upstream exploded")));
		assert!(norm.message_carries_body);
	}

	// -- formatProviderError ---------------------------------------------------

	#[test]
	fn surfaces_status_and_body_without_a_prefix() {
		let norm = normalize_provider_error(&error(
			"403 status code (no body)",
			Some(403),
			Some(r#"{"error":"blocked by gateway WAF"}"#),
		));
		let formatted = format_provider_error(&norm, None);
		assert!(formatted.contains("403"));
		assert!(formatted.contains("blocked by gateway WAF"));
		assert_ne!(formatted, "403 status code (no body)");
	}

	#[test]
	fn applies_a_provider_prefix_with_status_and_body() {
		let norm = normalize_provider_error(&error(
			"403 status code (no body)",
			Some(403),
			Some(r#"{"error":"blocked by gateway WAF"}"#),
		));
		assert_eq!(
			format_provider_error(&norm, Some("OpenAI API error")),
			r#"OpenAI API error (403): {"error":"blocked by gateway WAF"}"#
		);
	}

	#[test]
	fn preserves_the_message_with_prefix_and_status_when_it_already_carries_the_body() {
		let body = json!({"error": {"message": "Permission denied"}}).to_string();
		let norm = normalize_provider_error(&error(&body, Some(403), None));
		assert_eq!(
			format_provider_error(&norm, Some("OpenAI API error")),
			format!("OpenAI API error (403): {}", body)
		);
	}

	#[test]
	fn returns_the_bare_message_for_a_non_error_value() {
		let norm = NormalizedProviderError {
			status: None,
			body: None,
			message: safe_json_stringify(&json!({"reason": "boom"})),
			message_carries_body: false,
		};
		assert_eq!(format_provider_error(&norm, None), r#"{"reason":"boom"}"#);
	}

	// -- helpers ---------------------------------------------------------------

	#[test]
	fn truncate_error_text_reports_the_dropped_count() {
		assert_eq!(truncate_error_text("short", 10), "short");
		let truncated = truncate_error_text("abcdefghij", 3);
		assert_eq!(truncated, "abc... [truncated 7 chars]");
	}
}
