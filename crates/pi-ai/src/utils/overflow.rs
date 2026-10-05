//! Port of `packages/ai/src/utils/overflow.ts`.
//!
//! Regex patterns to detect context overflow errors from different providers.
//!
//! These patterns match error messages returned when the input exceeds
//! the model's context window.
//!
//! Provider-specific patterns (with example error messages):
//!
//! - Anthropic: "prompt is too long: 213462 tokens > 200000 maximum"
//! - Anthropic: "413 {\"error\":{\"type\":\"request_too_large\",\"message\":\"Request exceeds the maximum size\"}}"
//! - OpenAI: "Your input exceeds the context window of this model"
//! - OpenAI/LiteLLM: "Requested token count exceeds the model's maximum context length of 131072 tokens"
//! - OpenAI-compatible: "Input length (265330) exceeds model's maximum context length (262144)."
//! - Google: "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)"
//! - xAI: "This model's maximum prompt length is 131072 but the request contains 537812 tokens"
//! - Groq: "Please reduce the length of the messages or completion"
//! - OpenRouter: "This endpoint's maximum context length is X tokens. However, you requested about Y tokens"
//! - OpenRouter/Poolside: "Input length X exceeds the maximum allowed input length of Y tokens."
//! - Together AI: "The input (X tokens) is longer than the model's context length (Y tokens)."
//! - llama.cpp: "the request exceeds the available context size, try increasing it"
//! - LM Studio: "tokens to keep from the initial prompt is greater than the context length"
//! - GitHub Copilot: "prompt token count of X exceeds the limit of Y"
//! - MiniMax: "invalid params, context window exceeds limit"
//! - Kimi For Coding: "Your request exceeded model token limit: X (requested: Y)"
//! - DS4: "Prompt has X tokens, but the configured context size is Y tokens"
//! - Cerebras: "400/413 status code (no body)"
//! - Mistral: "Prompt contains X tokens ... too large for model with Y maximum context length"
//! - z.ai: `{"code":"1261","message":"Prompt too long"}`, `{"code":"1261","message":"Prompt exceeds max length"}` (CN endpoint), or silent overflow via usage.input > contextWindow
//! - Xiaomi MiMo: Truncates input to fill contextWindow exactly, then returns finish_reason "length"
//!   with output=0 (no room left to generate). Detected via stopReason "length" + zero output +
//!   input filling the context window.
//! - DashScope/Qwen: "Range of input length should be [1, X]" (HTTP 400 invalid_parameter_error)
//! - Ollama: Some deployments truncate silently, others return errors like "prompt too long; exceeded max context length by X tokens"

use regex::Regex;

use crate::types::AssistantMessage;
use std::sync::LazyLock;

/// Upstream `OVERFLOW_PATTERNS`, carried over 1:1 and case-insensitive.
const OVERFLOW_PATTERN_SOURCES: [&str; 25] = [
	r"prompt (?:is )?too long",                                       // Anthropic and z.ai token overflow
	r"prompt exceeds max length",                                     // z.ai CN endpoint token overflow
	r"request_too_large",                                             // Anthropic request byte-size overflow (HTTP 413)
	r"input is too long for requested model",                         // Amazon Bedrock
	r"exceeds the context window",                                    // OpenAI (Completions & Responses API)
	r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))", // OpenAI-compatible proxies (LiteLLM)
	r"input token count.*exceeds the maximum",                        // Google (Gemini)
	r"maximum prompt length is \d+",                                  // xAI (Grok)
	r"reduce the length of the messages",                             // Groq
	r"maximum context length is \d+ tokens",                          // OpenRouter (most backends)
	r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?", // OpenRouter/Poolside
	r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)", // Together AI
	r"exceeds the limit of \d+",                                      // GitHub Copilot
	r"exceeds the available context size",                            // llama.cpp server
	r"greater than the context length",                               // LM Studio
	r"context window exceeds limit",                                  // MiniMax
	r"exceeded model token limit",                                    // Kimi For Coding
	r"too large for model with \d+ maximum context length",           // Mistral
	r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?", // DS4 server
	r"model_context_window_exceeded",                                 // z.ai non-standard finish_reason surfaced as error text
	r"prompt too long; exceeded (?:max )?context length",             // Ollama explicit overflow error
	r"range of input length should be",                               // DashScope / Qwen Token Plan
	r"context[_ ]length[_ ]exceeded",                                 // Generic fallback
	r"too many tokens",                                               // Generic fallback
	r"token limit exceeded",                                          // Generic fallback
];

static CEREBRAS_BODYLESS_OVERFLOW_PATTERN: LazyLock<Regex> =
	LazyLock::new(|| Regex::new(r"(?i)^4(?:00|13)\s*(?:status code)?\s*\(no body\)").expect("pattern compiles"));

/// Patterns that indicate non-overflow errors (e.g. rate limiting, server errors).
/// Error messages matching any of these are excluded from overflow detection
/// even if they also match an `OVERFLOW_PATTERN`.
///
/// Example: Bedrock formats throttling errors as "ThrottlingException: Too many tokens,
/// please wait before trying again." which would match the /too many tokens/i overflow
/// pattern without this exclusion.
const NON_OVERFLOW_PATTERN_SOURCES: [&str; 3] = [
	// AWS Bedrock non-overflow errors (human-readable prefixes from formatBedrockError)
	r"^(Throttling error|Service unavailable):",
	// Generic rate limiting
	r"rate limit",
	// Generic HTTP 429 style
	r"too many requests",
];

static OVERFLOW_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
	OVERFLOW_PATTERN_SOURCES
		.iter()
		.map(|source| Regex::new(&format!("(?i){source}")).expect("overflow pattern compiles"))
		.collect()
});

static NON_OVERFLOW_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
	NON_OVERFLOW_PATTERN_SOURCES
		.iter()
		.map(|source| Regex::new(&format!("(?i){source}")).expect("non-overflow pattern compiles"))
		.collect()
});

/// Check if an assistant message represents a context overflow error.
///
/// This handles three cases:
/// 1. Error-based overflow: Most providers return stopReason "error" with a
///    specific error message pattern.
/// 2. Silent overflow: Some providers accept overflow requests and return
///    successfully. For these, we check if usage.input exceeds the context window.
/// 3. Length-stop overflow: Xiaomi MiMo can return "length" with zero output when
///    the input fills the context window.
///
/// ## Reliability by Provider
///
/// **Reliable detection (returns error with detectable message):**
/// - Anthropic: "prompt is too long: X tokens > Y maximum" or "request_too_large"
/// - OpenAI (Completions & Responses): "exceeds the context window", "exceeds the model's maximum context length of X tokens", or "exceeds model's maximum context length (X)"
/// - Google Gemini: "input token count exceeds the maximum"
/// - xAI (Grok): "maximum prompt length is X but request contains Y"
/// - Groq: "reduce the length of the messages"
/// - Cerebras: 400/413 status code (no body)
/// - Mistral: "Prompt contains X tokens ... too large for model with Y maximum context length"
/// - OpenRouter (most backends): "maximum context length is X tokens"
/// - OpenRouter/Poolside: "Input length X exceeds the maximum allowed input length of Y tokens."
/// - Together AI: "The input (X tokens) is longer than the model's context length (Y tokens)."
/// - llama.cpp: "exceeds the available context size"
/// - LM Studio: "greater than the context length"
/// - Kimi For Coding: "exceeded model token limit: X (requested: Y)"
/// - DS4: "Prompt has X tokens, but the configured context size is Y tokens"
/// - DashScope/Qwen: "Range of input length should be [1, X]"
/// - z.ai: "Prompt too long"
///
/// **Unreliable detection:**
/// - z.ai: Sometimes accepts overflow silently (detectable via usage.input > contextWindow),
///   sometimes returns rate limit errors. Pass contextWindow param to detect silent overflow.
/// - Xiaomi MiMo: Truncates input to fit contextWindow then returns stopReason "length" with
///   output=0. Pass contextWindow param to detect via the "filled context + zero output" signal.
/// - Ollama: May truncate input silently for some setups, but may also return explicit
///   overflow errors that match the patterns above. Silent truncation still cannot be
///   detected here because we do not know the expected token count.
///
/// ## Custom Providers
///
/// If you've added custom models via settings.json, this function may not detect
/// overflow errors from those providers. To add support:
///
/// 1. Send a request that exceeds the model's context window
/// 2. Check the errorMessage in the response
/// 3. Create a regex pattern that matches the error
/// 4. The pattern should be added to `OVERFLOW_PATTERN_SOURCES` in this file, or
///    check the errorMessage yourself before calling this function
///
/// Upstream `isContextOverflow`. `contextWindow` maps upstream's optional
/// parameter; `None`/`Some(0)` both mean "unknown".
pub fn is_context_overflow(message: &AssistantMessage, context_window: Option<u64>) -> bool {
	// Case 1: Check error message patterns
	if message.stop_reason == crate::types::StopReason::Error {
		if let Some(error_message) = &message.error_message {
			// Skip messages matching known non-overflow patterns (e.g. throttling / rate-limit)
			let is_non_overflow = NON_OVERFLOW_PATTERNS.iter().any(|pattern| pattern.is_match(error_message));
			if !is_non_overflow {
				if OVERFLOW_PATTERNS.iter().any(|pattern| pattern.is_match(error_message)) {
					return true;
				}
				if message.provider == "cerebras" && CEREBRAS_BODYLESS_OVERFLOW_PATTERN.is_match(error_message) {
					return true;
				}
			}
		}
	}

	let Some(context_window) = context_window.filter(|window| *window > 0) else {
		return false;
	};

	// Case 2: Silent overflow (z.ai style) - successful but usage exceeds context
	if message.stop_reason == crate::types::StopReason::Stop {
		let input_tokens = message.usage.input + message.usage.cache_read;
		if input_tokens > context_window {
			return true;
		}
	}

	// Case 3: Length-stop overflow (Xiaomi MiMo style) - server truncates oversized input
	// to fit the context window, leaving no room for output. Returns stopReason "length"
	// with output=0 and input+cacheRead filling the context window.
	if message.stop_reason == crate::types::StopReason::Length && message.usage.output == 0 {
		let input_tokens = message.usage.input + message.usage.cache_read;
		if input_tokens as f64 >= context_window as f64 * 0.99 {
			return true;
		}
	}

	false
}

/// Check whether a length stop ended below the caller or model's intended output limit.
/// Such responses may be caused by context pressure or provider-side truncation, so callers
/// can make one bounded compact-and-retry attempt. `desiredMaxOutput` must be the original
/// limit before any context-based clamping.
///
/// Upstream `isRecoverableLength`.
pub fn is_recoverable_length(message: &AssistantMessage, desired_max_output: u64) -> bool {
	message.stop_reason == crate::types::StopReason::Length
		&& desired_max_output > 0
		&& message.usage.output < desired_max_output
}

/// Get the overflow patterns for testing purposes.
///
/// Upstream `getOverflowPatterns`.
pub fn get_overflow_patterns() -> Vec<Regex> {
	OVERFLOW_PATTERNS.clone()
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::{StopReason, Usage, UsageCost};

	fn zero_usage() -> Usage {
		Usage {
			input: 0,
			output: 0,
			cache_read: 0,
			cache_write: 0,
			cache_write_1h: None,
			reasoning: None,
			total_tokens: 0,
			cost: UsageCost::default(),
		}
	}

	fn create_error_message(error_message: &str, provider: &str) -> AssistantMessage {
		AssistantMessage {
			content: Vec::new(),
			api: "openai-completions".to_string(),
			provider: provider.to_string(),
			model: "qwen3.5:35b".to_string(),
			response_model: None,
			response_id: None,
			provider_thinking_level: None,
			thinking_level: None,
			diagnostics: None,
			usage: zero_usage(),
			stop_reason: StopReason::Error,
			deferred: None,
			error_message: Some(error_message.to_string()),
			raw_stop_reason: None,
			end_turn: None,
			timestamp: 0,
		}
	}

	fn create_length_stop_message(
		input: u64,
		cache_read: u64,
		output: u64,
		cache_write: u64,
		provider: &str,
		model: &str,
	) -> AssistantMessage {
		AssistantMessage {
			content: Vec::new(),
			api: "openai-completions".to_string(),
			provider: provider.to_string(),
			model: model.to_string(),
			response_model: None,
			response_id: None,
			provider_thinking_level: None,
			thinking_level: None,
			diagnostics: None,
			usage: Usage {
				input,
				output,
				cache_read,
				cache_write,
				total_tokens: input + cache_read + cache_write + output,
				..zero_usage()
			},
			stop_reason: StopReason::Length,
			deferred: None,
			error_message: None,
			raw_stop_reason: None,
			end_turn: None,
			timestamp: 0,
		}
	}

	#[test]
	fn detects_explicit_ollama_prompt_too_long_errors() {
		let message = create_error_message("400 `prompt too long; exceeded max context length by 100918 tokens`", "ollama");
		assert!(is_context_overflow(&message, Some(32768)));
	}

	#[test]
	fn detects_zai_prompt_too_long_errors() {
		// Regression for #9805.
		let message = create_error_message(r#"400 {"code":"1261","message":"Prompt too long"}"#, "zai");
		assert!(is_context_overflow(&message, Some(1048576)));
	}

	#[test]
	fn detects_zai_cn_endpoint_prompt_exceeds_max_length_errors() {
		// Regression for #10208.
		let message = create_error_message(r#"400 {"code":"1261","message":"Prompt exceeds max length"}"#, "zai");
		assert!(is_context_overflow(&message, Some(1048576)));
	}

	#[test]
	fn detects_together_ai_context_length_errors() {
		let message = create_error_message(
			"400 The input (516368 tokens) is longer than the model's context length (262144 tokens).",
			"ollama",
		);
		assert!(is_context_overflow(&message, Some(262144)));
	}

	#[test]
	fn detects_litellm_wrapped_openai_maximum_context_length_errors() {
		let message = create_error_message(
			"Error: 503 litellm.ServiceUnavailableError: litellm.MidStreamFallbackError: litellm.APIConnectionError: APIConnectionError: OpenAIException - Requested token count exceeds the model's maximum context length of 131072 tokens.",
			"ollama",
		);
		assert!(is_context_overflow(&message, Some(131072)));
	}

	#[test]
	fn detects_openai_compatible_parenthesized_maximum_context_length_errors() {
		let message = create_error_message(
			"Error: 400 Input length (265330) exceeds model's maximum context length (262144).",
			"ollama",
		);
		assert!(is_context_overflow(&message, Some(262144)));
	}

	#[test]
	fn detects_openrouter_poolside_maximum_allowed_input_length_errors() {
		let message = create_error_message(
			"Provider returned error: Input length 131393 exceeds the maximum allowed input length of 131040 tokens.",
			"ollama",
		);
		assert!(is_context_overflow(&message, Some(131072)));
	}

	#[test]
	fn detects_ds4_configured_context_size_errors() {
		let message = create_error_message(
			"400 Prompt has 256468 tokens, but the configured context size is 256000 tokens",
			"ollama",
		);
		assert!(is_context_overflow(&message, Some(256000)));

		let comma_message = create_error_message(
			"Prompt has 5,958,968 tokens, but the configured context size is 256,000 tokens",
			"ollama",
		);
		assert!(is_context_overflow(&comma_message, Some(256000)));
	}

	#[test]
	fn does_not_treat_generic_non_overflow_ollama_errors_as_overflow() {
		let message = create_error_message("500 `model runner crashed unexpectedly`", "ollama");
		assert!(!is_context_overflow(&message, Some(32768)));
	}

	#[test]
	fn only_treats_bodyless_400_and_413_errors_as_overflow_for_cerebras() {
		// Regression for #9482.
		for error_message in ["400 status code (no body)", "413 status code (no body)"] {
			assert!(is_context_overflow(&create_error_message(error_message, "cerebras"), Some(131072)));
			assert!(!is_context_overflow(&create_error_message(error_message, "opencode-go"), Some(1000000)));
		}
	}

	#[test]
	fn does_not_treat_bedrock_throttling_too_many_tokens_as_overflow() {
		// Bedrock returns this for HTTP 429 rate limiting, NOT context overflow.
		// formatBedrockError uses a human-readable prefix for ThrottlingException.
		let message = create_error_message("Throttling error: Too many tokens, please wait before trying again.", "ollama");
		assert!(!is_context_overflow(&message, Some(200000)));
	}

	#[test]
	fn does_not_treat_bedrock_service_unavailable_as_overflow() {
		let message = create_error_message("Service unavailable: The service is temporarily unavailable.", "ollama");
		assert!(!is_context_overflow(&message, Some(200000)));
	}

	#[test]
	fn does_not_treat_generic_rate_limit_errors_as_overflow() {
		let message = create_error_message("Rate limit exceeded, please retry after 30 seconds.", "ollama");
		assert!(!is_context_overflow(&message, Some(200000)));
	}

	#[test]
	fn does_not_treat_http_429_style_errors_as_overflow() {
		let message = create_error_message("Too many requests. Please slow down.", "ollama");
		assert!(!is_context_overflow(&message, Some(200000)));
	}

	#[test]
	fn detects_xiaomi_style_overflow_length_stop_with_zero_output_and_filled_context() {
		let message = create_length_stop_message(58, 1048512, 0, 0, "xiaomi", "mimo-v2.5-pro");
		assert!(is_context_overflow(&message, Some(1048576)));
	}

	#[test]
	fn treats_a_length_stop_below_the_desired_output_limit_as_recoverable() {
		let message = create_length_stop_message(3, 253584, 16, 25554, "openai", "gpt-5.6-sol");
		assert!(is_recoverable_length(&message, 128000));
	}

	#[test]
	fn does_not_recover_a_length_stop_that_reached_the_desired_output_limit() {
		let message = create_length_stop_message(4062, 0, 1024, 0, "test-provider", "test-model");
		assert!(!is_recoverable_length(&message, 1024));
	}

	#[test]
	fn treats_zero_output_length_stops_as_recoverable_without_context_metadata() {
		let message = create_length_stop_message(100, 0, 0, 0, "test-provider", "test-model");
		assert!(is_recoverable_length(&message, 128000));
	}

	#[test]
	fn does_not_treat_normal_length_stops_with_output_as_context_overflow() {
		let message = create_length_stop_message(1000, 0, 4096, 0, "test-provider", "test-model");
		assert!(!is_context_overflow(&message, Some(200000)));
	}

	#[test]
	fn does_not_treat_zero_output_length_stops_far_below_context_as_context_overflow() {
		let message = create_length_stop_message(100, 0, 0, 0, "test-provider", "test-model");
		assert!(!is_context_overflow(&message, Some(200000)));
	}

	#[test]
	fn exposes_the_full_pattern_list() {
		assert_eq!(get_overflow_patterns().len(), OVERFLOW_PATTERN_SOURCES.len());
	}
}
