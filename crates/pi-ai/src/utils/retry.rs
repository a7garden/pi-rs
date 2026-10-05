//! Port of `packages/ai/src/utils/retry.ts`.
//!
//! Bounded assistant-call retry with exponential backoff
//! (`baseDelayMs * 2^(attempt-1)`, capped by `maxAgentDelayMs`, default 60s),
//! the transient-provider-error classifier and its regex pattern lists, which
//! are carried over verbatim (case-insensitive, joined like the upstream
//! `new RegExp(patterns.join("|"), "i")`).
//!
//! Deviations (crate docs ledger):
//! - **D18 — `setTimeout`/`AbortSignal` sleeps become `tokio::time::sleep`
//!   raced against the pure-Rust [`crate::utils::abort::AbortSignal`]**; the
//!   `RetrySleepAbortError` sentinel is a private unit error, never surfaced.
//! - **D18a — `Math.random` jitter/backoff inputs use `getrandom`**, and
//!   overflow beyond `Number.MAX_SAFE_INTEGER` saturates instead of jumping to
//!   `2^53 - 1` (both are capped by `maxAgentDelayMs` in practice).
//! - **D18b — retry callbacks are sync-invoked async hooks** (`Arc`d closures
//!   returning boxed futures), one struct instead of an optional-fields object.

use std::future::Future;
use std::sync::{Arc, LazyLock};

use futures_core::future::BoxFuture;
use regex::Regex;

use crate::types::{AssistantMessage, StopReason};

/// Retry policy: bounded attempts with exponential backoff
/// (`baseDelayMs * 2^(attempt-1)`). `max_agent_delay_ms` caps each computed
/// delay and defaults to 60 seconds. Matches `settings.retry` (`enabled`,
/// `maxRetries`, `baseDelayMs`, `maxAgentDelayMs`) in coding-agent; kept here so
/// the classifier and the policy-driven retry loop live together and stay
/// reusable by the SDK and other callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryPolicy {
	pub enabled: bool,
	/// Max retry attempts (0 = no retries). The initial call never counts as a retry.
	pub max_retries: u32,
	/// Base delay in ms. Per-attempt delay is `baseDelayMs * 2^(attempt-1)` before jitter.
	pub base_delay_ms: u64,
	/// Optional cap for agent-level retry delays in ms. Defaults to 60 seconds.
	pub max_agent_delay_ms: Option<u64>,
}

/// Upstream `DEFAULT_MAX_AGENT_RETRY_DELAY_MS`.
pub const DEFAULT_MAX_AGENT_RETRY_DELAY_MS: u64 = 60_000;

fn non_retryable_provider_limit_error_pattern() -> &'static Regex {
	static PATTERN: LazyLock<Regex> = LazyLock::new(|| {
		Regex::new(&format!("(?i){}", [
			// OpenCode Go/free-tier limits returned as 429 JSON error types by OpenCode's
			// Zen API. These are subscription/account limits, not transient throttles.
			"GoUsageLimitError",
			"FreeUsageLimitError",
			// OpenCode Go subscription-limit text asks users to enable available-balance
			// usage after rolling/weekly/monthly limits are reached.
			"Monthly usage limit reached",
			"available balance",
			// Generic quota/budget/billing exhaustion. `insufficient_quota` is OpenAI's
			// quota/billing error code; the other strings cover common gateway wording.
			"insufficient_quota",
			"out of budget",
			"quota exceeded",
			"billing",
			// Sign in with ChatGPT: the subscription's shared usage limit, which resets
			// after hours rather than seconds.
			"subscription_sharing_usage_limit_exceeded",
		]
		.join("|")))
		.expect("non-retryable limit pattern compiles")
	});
	&PATTERN
}

fn retryable_provider_error_pattern() -> &'static Regex {
	static PATTERN: LazyLock<Regex> = LazyLock::new(|| {
		Regex::new(&format!("(?i){}", [
			// Generic provider load, HTTP status, and server-side transient failures.
			"overloaded",
			"currently experiencing high demand",
			"model is at capacity",
			"rate.?limit",
			"too many requests",
			"429",
			"500",
			"502",
			"503",
			"504",
			"520",
			"524",
			"service.?unavailable",
			"server.?error",
			"internal.?error",
			// Wrapper/provider text for transient upstream failures, including OpenRouter
			// "Provider returned error" responses (#2264).
			"provider.?returned.?error",
			"exceeded request buffer limit while retrying upstream",
			// Network, proxy, and fetch transport failures. This includes OpenAI Codex
			// raw-fetch failures such as "upstream connect", "connection refused", and
			// "reset before headers" (#733), plus OpenRouter connection drops (#3317).
			"network.?error",
			"connection.?error",
			"connection.?refused",
			"connection.?lost",
			"other side closed",
			"fetch failed",
			"getaddrinfo",
			"ENOTFOUND",
			"EAI_AGAIN",
			"upstream.?connect",
			"reset before headers",
			"socket hang up",
			"socket connection was closed",
			"timed? out",
			"timeout",
			"terminated",
			// WebSocket transports can report close/error text instead of HTTP/fetch text.
			"websocket.?closed",
			"websocket.?error",
			// Premature stream endings from SDKs and transports. Anthropic can throw
			// "stream ended without ..." and "Anthropic stream ended before message_stop"
			// (#4433); Bedrock/Smithy can throw an HTTP/2 no-response error (#3594).
			"ended without",
			"stream ended before message_stop",
			"stream ended before a terminal response event",
			"http2 request did not get a response",
			// Provider-requested retry delay cap failures should flow through the outer
			// retry policy so callers can surface/abort the backoff (#1123).
			"retry delay",
			// Explicit retry guidance emitted mid-stream by OpenAI Responses and Bedrock
			// stream exceptions (#6019).
			"you can retry your request",
			"try your request again",
			"please retry your request",
			// gRPC based providers (e.g. NVIDIA NIM)
			"ResourceExhausted",
			// Sign in with ChatGPT: usage or user data temporarily unavailable. Usage
			// failures can arrive mid-stream without an HTTP 503 in the message.
			"subscription_sharing_usage_unavailable",
			"subscription_sharing_user_unavailable",
		]
		.join("|")))
		.expect("retryable provider pattern compiles")
	});
	&PATTERN
}

/// Upstream `retryDelayMs`.
pub fn retry_delay_ms(policy: &RetryPolicy, attempt: u32) -> u64 {
	let shift = attempt.saturating_sub(1).min(63);
	let delay = policy.base_delay_ms.saturating_mul(1u64 << shift);
	delay.min(policy.max_agent_delay_ms.unwrap_or(DEFAULT_MAX_AGENT_RETRY_DELAY_MS))
}

/// Optional callbacks emitted by [`retry_assistant_call`] around each retry
/// (upstream `RetryCallbacks`; hooks receive owned data and may be async).
#[derive(Clone, Default)]
pub struct RetryCallbacks {
	/// Emitted before the backoff sleep of each retry attempt (1-indexed).
	pub on_retry_scheduled: Option<Arc<dyn Fn(u32, u32, u64, &str) -> BoxFuture<'static, ()> + Send + Sync>>,
	/// Emitted after the backoff sleep, immediately before the retried call starts.
	pub on_retry_attempt_start: Option<Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>>,
	/// Emitted once when the loop ends: success if a later call completed normally.
	pub on_retry_finished: Option<Arc<dyn Fn(bool, u32, Option<&str>) -> BoxFuture<'static, ()> + Send + Sync>>,
}

/// Sentinel for an abort during the backoff sleep (upstream
/// `RetrySleepAbortError`); never crosses this module's API.
struct RetrySleepAbort;

async fn retry_sleep(ms: u64, signal: Option<&crate::utils::abort::AbortSignal>) -> Result<(), RetrySleepAbort> {
	match signal {
		Some(signal) if signal.aborted() => Err(RetrySleepAbort),
		Some(signal) => {
			tokio::select! {
				biased;
				_ = tokio::time::sleep(std::time::Duration::from_millis(ms)) => Ok(()),
				_ = signal.aborted_future() => Err(RetrySleepAbort),
			}
		}
		None => {
			tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
			Ok(())
		}
	}
}

/// Run a single assistant-producing call with bounded retry on transient errors.
///
/// Behavior:
/// - A successful response is returned immediately. Aborts are terminal and never
///   retried, but reported as unsuccessful if they happen after a retry was scheduled.
///   Aborts during the backoff sleep are normalized to an aborted `AssistantMessage`
///   too, so callers do not need to care when cancellation happened.
/// - A non-retryable error (per [`is_retryable_assistant_error`], including quota/
///   billing exhaustion) is returned immediately so deterministic errors fail fast.
/// - Otherwise retries up to `policy.max_retries` times with exponential backoff,
///   emitting `on_retry_scheduled` before each sleep, `on_retry_attempt_start` after
///   each sleep before the retried call starts, and `on_retry_finished` once at the
///   end (whether the loop ends in success, exhausted retries, or an aborted backoff).
///
/// When `policy` is `None` or disabled, the first response is returned unchanged
/// (equivalent to calling `produce` directly).
///
/// Upstream `retryAssistantCall`.
pub async fn retry_assistant_call<P, Fut>(
	mut produce: P,
	policy: Option<&RetryPolicy>,
	signal: Option<&crate::utils::abort::AbortSignal>,
	callbacks: Option<&RetryCallbacks>,
) -> AssistantMessage
where
	P: FnMut() -> Fut,
	Fut: Future<Output = AssistantMessage>,
{
	let max_attempts = policy.filter(|policy| policy.enabled).map_or(0, |policy| policy.max_retries);

	let mut attempt: u32 = 0;
	let mut last_retry: Option<(u32, String)> = None;
	loop {
		let response = produce().await;

		// Abort: terminal but not successful. Never retry an aborted message.
		if response.stop_reason == StopReason::Aborted {
			if let Some((last_attempt, _)) = &last_retry {
				if let Some(on_retry_finished) = callbacks.and_then(|callbacks| callbacks.on_retry_finished.as_ref()) {
					on_retry_finished(false, *last_attempt, None).await;
				}
			}
			return response;
		}

		// Success: non-error, non-abort responses return as-is.
		if response.stop_reason != StopReason::Error {
			if let Some((last_attempt, _)) = &last_retry {
				if let Some(on_retry_finished) = callbacks.and_then(|callbacks| callbacks.on_retry_finished.as_ref()) {
					on_retry_finished(true, *last_attempt, None).await;
				}
			}
			return response;
		}

		// Non-retryable, or budget exhausted: return the final error message.
		if attempt >= max_attempts || !is_retryable_assistant_error(&response) {
			if let Some((last_attempt, _)) = &last_retry {
				if let Some(on_retry_finished) = callbacks.and_then(|callbacks| callbacks.on_retry_finished.as_ref()) {
					on_retry_finished(false, *last_attempt, response.error_message.as_deref()).await;
				}
			}
			return response;
		}

		attempt += 1;
		let error_message = response
			.error_message
			.clone()
			.filter(|message| !message.is_empty())
			.unwrap_or_else(|| "Unknown error".to_string());
		last_retry = Some((attempt, error_message.clone()));
		let delay_ms = retry_delay_ms(policy.expect("retrying requires an enabled policy"), attempt);
		if let Some(on_retry_scheduled) = callbacks.and_then(|callbacks| callbacks.on_retry_scheduled.as_ref()) {
			on_retry_scheduled(attempt, max_attempts, delay_ms, &error_message).await;
		}

		// Normalize aborts during retry backoff to the same AssistantMessage shape as
		// provider stream aborts, so callers do not need to care when cancellation happened.
		if retry_sleep(delay_ms, signal).await.is_err() {
			if let Some(on_retry_finished) = callbacks.and_then(|callbacks| callbacks.on_retry_finished.as_ref()) {
				on_retry_finished(false, attempt, Some(error_message.as_str())).await;
			}
			let mut aborted = response;
			aborted.stop_reason = StopReason::Aborted;
			aborted.error_message = None;
			return aborted;
		}
		if let Some(on_retry_attempt_start) = callbacks.and_then(|callbacks| callbacks.on_retry_attempt_start.as_ref())
		{
			on_retry_attempt_start().await;
		}
	}
}

/// Classifies whether a failed assistant message looks like a transient provider
/// or transport error, so callers can decide if the last assistant turn should be
/// restarted.
///
/// This does not implement retry policy. Callers should first handle context
/// overflow separately, then apply their own retry budget, backoff, and reporting
/// before restarting the assistant turn.
///
/// Upstream `isRetryableAssistantError`.
pub fn is_retryable_assistant_error(message: &AssistantMessage) -> bool {
	if message.stop_reason != StopReason::Error {
		return false;
	}
	let Some(error_message) = &message.error_message else {
		return false;
	};
	if non_retryable_provider_limit_error_pattern().is_match(error_message) {
		return false;
	}
	retryable_provider_error_pattern().is_match(error_message)
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::Mutex;
	use std::sync::atomic::{AtomicUsize, Ordering};

	use crate::types::{AssistantContent, TextContent, Usage, UsageCost};

	const OPEN_AI_EXPLICIT_RETRY_MESSAGE: &str = "An error occurred while processing your request. You can retry your request, or contact us through our help center at help.openai.com if the error persists. Please include the request ID req_******** in your message.";
	const BEDROCK_EXPLICIT_RETRY_MESSAGE: &str =
		r#"{"message":"The system encountered an unexpected error during processing. Try your request again."}"#;
	const NVIDIA_NIM_RESOURCE_EXHAUSTED_MESSAGE: &str =
		"ResourceExhausted: Worker local total request limit reached (288/48)";
	const BUN_FETCH_SOCKET_CLOSED_MESSAGE: &str = "The socket connection was closed unexpectedly. For more information, pass `verbose: true` in the second argument to fetch()";
	const OPEN_AI_RESPONSES_EARLY_EOF_MESSAGE: &str = "OpenAI Responses stream ended before a terminal response event";
	const WRAPPED_DNS_LOOKUP_ERROR: &str = "The pending stream has been canceled (caused by: getaddrinfo ENOTFOUND bedrock-runtime.us-east-1.amazonaws.com)";
	const AZURE_PEAK_LOAD_ERROR: &str = "The system is currently experiencing high demand and cannot process your request. Your request exceeds the maximum usage size allowed during peak load. For improved capacity reliability, consider switching to Provisioned Throughput.";

	/// Test helper mirroring `fauxAssistantMessage` (providers/faux.ts); the
	/// faux provider module itself is ported by the provider slices.
	fn faux_assistant_message(text: &str, stop_reason: StopReason, error_message: Option<&str>) -> AssistantMessage {
		AssistantMessage {
			content: if text.is_empty() && stop_reason != StopReason::Stop {
				Vec::new()
			} else {
				vec![AssistantContent::Text(TextContent::new(text))]
			},
			api: "faux".to_string(),
			provider: "faux".to_string(),
			model: "faux-1".to_string(),
			response_model: None,
			response_id: None,
			provider_thinking_level: None,
			thinking_level: None,
			diagnostics: None,
			usage: Usage {
				input: 0,
				output: 0,
				cache_read: 0,
				cache_write: 0,
				cache_write_1h: None,
				reasoning: None,
				total_tokens: 0,
				cost: UsageCost::default(),
			},
			stop_reason,
			deferred: None,
			error_message: error_message.map(str::to_string),
			raw_stop_reason: None,
			end_turn: None,
			timestamp: 0,
		}
	}

	fn error_message_message(error_message: &str) -> AssistantMessage {
		faux_assistant_message("", StopReason::Error, Some(error_message))
	}

	fn enabled_policy() -> RetryPolicy {
		RetryPolicy { enabled: true, max_retries: 3, base_delay_ms: 0, max_agent_delay_ms: None }
	}

	fn disabled_policy() -> RetryPolicy {
		RetryPolicy { enabled: false, max_retries: 3, base_delay_ms: 0, max_agent_delay_ms: None }
	}

	// -- provider retry classification -----------------------------------------

	#[test]
	fn matches_explicit_provider_retry_guidance() {
		assert!(is_retryable_assistant_error(&error_message_message(OPEN_AI_EXPLICIT_RETRY_MESSAGE)));
		assert!(is_retryable_assistant_error(&error_message_message(BEDROCK_EXPLICIT_RETRY_MESSAGE)));
		assert!(is_retryable_assistant_error(&error_message_message(NVIDIA_NIM_RESOURCE_EXHAUSTED_MESSAGE)));
	}

	#[test]
	fn matches_bun_fetch_socket_drop_wording() {
		assert!(is_retryable_assistant_error(&error_message_message(BUN_FETCH_SOCKET_CLOSED_MESSAGE)));
	}

	#[test]
	fn matches_upstream_request_buffer_exhaustion_wording() {
		assert!(is_retryable_assistant_error(&error_message_message(
			"Error: exceeded request buffer limit while retrying upstream"
		)));
	}

	#[test]
	fn matches_dns_transport_failure_wording() {
		for error_message in [
			WRAPPED_DNS_LOOKUP_ERROR,
			"connect ENOTFOUND api.example.com",
			"EAI_AGAIN api.example.com",
			"getaddrinfo failed for api.example.com",
		] {
			assert!(is_retryable_assistant_error(&error_message_message(error_message)), "{error_message}");
		}
	}

	#[test]
	fn matches_openai_responses_streams_that_end_before_terminal_events() {
		assert!(is_retryable_assistant_error(&error_message_message(OPEN_AI_RESPONSES_EARLY_EOF_MESSAGE)));
	}

	#[test]
	fn matches_azure_peak_load_capacity_errors() {
		// Regression for #9669.
		assert!(is_retryable_assistant_error(&error_message_message(AZURE_PEAK_LOAD_ERROR)));
	}

	#[test]
	fn keeps_provider_limit_errors_non_retryable() {
		assert!(!is_retryable_assistant_error(&error_message_message("429 quota exceeded")));
	}

	#[test]
	fn keeps_the_chatgpt_subscription_usage_limit_non_retryable() {
		let error_message =
			r#"OpenAI API error (429): {"code":"subscription_sharing_usage_limit_exceeded","message":"Usage limit reached."}"#;
		assert!(!is_retryable_assistant_error(&error_message_message(error_message)));
	}

	#[test]
	fn retries_temporary_chatgpt_subscription_errors() {
		for error_message in [
			"subscription_sharing_usage_unavailable: Usage cannot be checked.",
			"subscription_sharing_user_unavailable: User cannot be loaded.",
		] {
			assert!(is_retryable_assistant_error(&error_message_message(error_message)), "{error_message}");
		}
	}

	#[test]
	fn classifies_assistant_error_messages() {
		assert!(is_retryable_assistant_error(&error_message_message("overloaded_error")));
		// Regression for #9627.
		assert!(is_retryable_assistant_error(&error_message_message("520 status code (no body)")));
		assert!(is_retryable_assistant_error(&error_message_message("524 status code (no body)")));
		assert!(!is_retryable_assistant_error(&faux_assistant_message("not an error", StopReason::Stop, None)));
	}

	// -- retryDelayMs -----------------------------------------------------------

	#[test]
	fn caps_agent_retry_delay() {
		// Regression for #8826.
		let base = RetryPolicy { enabled: true, max_retries: 0, base_delay_ms: 2000, max_agent_delay_ms: None };
		assert_eq!(retry_delay_ms(&base, 6), 60000);
		let capped =
			RetryPolicy { enabled: true, max_retries: 0, base_delay_ms: 2000, max_agent_delay_ms: Some(5000) };
		assert_eq!(retry_delay_ms(&capped, 5), 5000);
		let zero_cap = RetryPolicy { enabled: true, max_retries: 0, base_delay_ms: 2000, max_agent_delay_ms: Some(0) };
		assert_eq!(retry_delay_ms(&zero_cap, 5), 0);
	}

	// -- retryAssistantCall -----------------------------------------------------

	type Produce = Box<dyn FnMut() -> futures_core::future::BoxFuture<'static, AssistantMessage> + Send>;

	fn counter_produce(
		messages: Arc<Mutex<Vec<AssistantMessage>>>,
	) -> (Produce, Arc<AtomicUsize>) {
		let calls = Arc::new(AtomicUsize::new(0));
		let produce_calls = calls.clone();
		let produce: Produce = Box::new(move || {
			let messages = messages.clone();
			let produce_calls = produce_calls.clone();
			Box::pin(async move {
				produce_calls.fetch_add(1, Ordering::SeqCst);
				let mut queue = messages.lock().expect("produce queue");
				if queue.len() > 1 {
					queue.remove(0)
				} else {
					queue[0].clone()
				}
			})
		});
		(produce, calls)
	}

	fn recorded_events() -> Arc<Mutex<Vec<String>>> {
		Arc::new(Mutex::new(Vec::new()))
	}

	fn callbacks_with(
		on_retry_scheduled: Option<Arc<dyn Fn(u32, u32, u64, &str) -> BoxFuture<'static, ()> + Send + Sync>>,
		on_retry_finished: Option<Arc<dyn Fn(bool, u32, Option<&str>) -> BoxFuture<'static, ()> + Send + Sync>>,
		on_retry_attempt_start: Option<Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>>,
	) -> RetryCallbacks {
		RetryCallbacks { on_retry_scheduled, on_retry_attempt_start, on_retry_finished }
	}

	#[tokio::test]
	async fn returns_a_successful_response_immediately_without_retrying() {
		let (produce, calls) = counter_produce(Arc::new(Mutex::new(vec![faux_assistant_message(
			"ok", StopReason::Stop, None,
		)])));
		let res = retry_assistant_call(produce, Some(&enabled_policy()), None, None).await;
		assert_eq!(
			res.content,
			vec![AssistantContent::Text(TextContent::new("ok"))]
		);
		assert_eq!(calls.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn does_not_retry_an_aborted_message() {
		let (produce, calls) = counter_produce(Arc::new(Mutex::new(vec![faux_assistant_message(
			"", StopReason::Aborted, None,
		)])));
		let scheduled = recorded_events();
		let on_retry_scheduled: Arc<dyn Fn(u32, u32, u64, &str) -> BoxFuture<'static, ()> + Send + Sync> = {
			let scheduled = scheduled.clone();
			Arc::new(move |attempt, _, _, _| {
				let scheduled = scheduled.clone();
				Box::pin(async move {
					scheduled.lock().expect("events").push(format!("retry:{attempt}"));
				})
			})
		};
		let res = retry_assistant_call(
			produce,
			Some(&enabled_policy()),
			None,
			Some(&callbacks_with(Some(on_retry_scheduled), None, None)),
		)
		.await;
		assert_eq!(res.stop_reason, StopReason::Aborted);
		assert_eq!(calls.load(Ordering::SeqCst), 1);
		assert!(scheduled.lock().expect("events").is_empty());
	}

	#[tokio::test]
	async fn does_not_retry_a_non_retryable_error() {
		let (produce, calls) = counter_produce(Arc::new(Mutex::new(vec![faux_assistant_message(
			"", StopReason::Error, Some("insufficient_quota"),
		)])));
		let finished: Arc<Mutex<Vec<(bool, u32, Option<String>)>>> = Arc::new(Mutex::new(Vec::new()));
		let on_retry_finished: Arc<dyn Fn(bool, u32, Option<&str>) -> BoxFuture<'static, ()> + Send + Sync> = {
			let finished = finished.clone();
			Arc::new(move |success, attempt, error| {
				let finished = finished.clone();
				let error = error.map(str::to_string);
				Box::pin(async move {
					finished.lock().expect("finished").push((success, attempt, error));
				})
			})
		};
		let res = retry_assistant_call(
			produce,
			Some(&enabled_policy()),
			None,
			Some(&callbacks_with(None, Some(on_retry_finished), None)),
		)
		.await;
		assert_eq!(res.stop_reason, StopReason::Error);
		assert_eq!(calls.load(Ordering::SeqCst), 1);
		assert!(finished.lock().expect("finished").is_empty());
	}

	#[tokio::test]
	async fn retries_a_transient_error_up_to_max_retries_then_returns_the_final_error() {
		let (produce, calls) = counter_produce(Arc::new(Mutex::new(vec![faux_assistant_message(
			"", StopReason::Error, Some("terminated"),
		)])));
		let finished: Arc<Mutex<Vec<(bool, u32, Option<String>)>>> = Arc::new(Mutex::new(Vec::new()));
		let on_retry_finished: Arc<dyn Fn(bool, u32, Option<&str>) -> BoxFuture<'static, ()> + Send + Sync> = {
			let finished = finished.clone();
			Arc::new(move |success, attempt, error| {
				let finished = finished.clone();
				let error = error.map(str::to_string);
				Box::pin(async move {
					finished.lock().expect("finished").push((success, attempt, error));
				})
			})
		};
		let res = retry_assistant_call(
			produce,
			Some(&enabled_policy()),
			None,
			Some(&callbacks_with(None, Some(on_retry_finished), None)),
		)
		.await;
		assert_eq!(res.stop_reason, StopReason::Error);
		assert_eq!(calls.load(Ordering::SeqCst), 4); // 1 initial + 3 retries
		assert_eq!(
			finished.lock().expect("finished").as_slice(),
			[(false, 3, Some("terminated".to_string()))]
		);
	}

	#[tokio::test]
	async fn reports_capped_retry_delays() {
		// Regression for #8826.
		let policy = RetryPolicy { enabled: true, max_retries: 4, base_delay_ms: 10, max_agent_delay_ms: Some(15) };
		let produce_calls = Arc::new(AtomicUsize::new(0));
		let counter = produce_calls.clone();
		let produce: Produce = Box::new(move || {
			let counter = counter.clone();
			Box::pin(async move {
				let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
				if n < 5 {
					faux_assistant_message("", StopReason::Error, Some("terminated"))
				} else {
					faux_assistant_message("recovered", StopReason::Stop, None)
				}
			})
		});
		let delays: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
		let on_retry_scheduled: Arc<dyn Fn(u32, u32, u64, &str) -> BoxFuture<'static, ()> + Send + Sync> = {
			let delays = delays.clone();
			Arc::new(move |_, _, delay_ms, _| {
				let delays = delays.clone();
				Box::pin(async move {
					delays.lock().expect("delays").push(delay_ms);
				})
			})
		};

		retry_assistant_call(produce, Some(&policy), None, Some(&callbacks_with(Some(on_retry_scheduled), None, None)))
			.await;

		assert_eq!(*delays.lock().expect("delays"), vec![10, 15, 15, 15]);
	}

	#[tokio::test]
	async fn stops_retrying_once_a_call_succeeds() {
		let produce_calls = Arc::new(AtomicUsize::new(0));
		let counter = produce_calls.clone();
		let produce: Produce = Box::new(move || {
			let counter = counter.clone();
			Box::pin(async move {
				let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
				if n < 3 {
					faux_assistant_message("", StopReason::Error, Some("terminated"))
				} else {
					faux_assistant_message("recovered", StopReason::Stop, None)
				}
			})
		});
		let finished: Arc<Mutex<Vec<(bool, u32, Option<String>)>>> = Arc::new(Mutex::new(Vec::new()));
		let on_retry_finished: Arc<dyn Fn(bool, u32, Option<&str>) -> BoxFuture<'static, ()> + Send + Sync> = {
			let finished = finished.clone();
			Arc::new(move |success, attempt, error| {
				let finished = finished.clone();
				let error = error.map(str::to_string);
				Box::pin(async move {
					finished.lock().expect("finished").push((success, attempt, error));
				})
			})
		};
		let res = retry_assistant_call(
			produce,
			Some(&enabled_policy()),
			None,
			Some(&callbacks_with(None, Some(on_retry_finished), None)),
		)
		.await;
		assert_eq!(res.content, vec![AssistantContent::Text(TextContent::new("recovered"))]);
		assert_eq!(produce_calls.load(Ordering::SeqCst), 3);
		assert_eq!(finished.lock().expect("finished").as_slice(), [(true, 2, None)]);
	}

	#[tokio::test]
	async fn reports_an_aborted_retried_call_as_unsuccessful() {
		let produce_calls = Arc::new(AtomicUsize::new(0));
		let counter = produce_calls.clone();
		let produce: Produce = Box::new(move || {
			let counter = counter.clone();
			Box::pin(async move {
				let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
				if n == 1 {
					faux_assistant_message("", StopReason::Error, Some("terminated"))
				} else {
					faux_assistant_message("", StopReason::Aborted, None)
				}
			})
		});
		let finished: Arc<Mutex<Vec<(bool, u32, Option<String>)>>> = Arc::new(Mutex::new(Vec::new()));
		let on_retry_finished: Arc<dyn Fn(bool, u32, Option<&str>) -> BoxFuture<'static, ()> + Send + Sync> = {
			let finished = finished.clone();
			Arc::new(move |success, attempt, error| {
				let finished = finished.clone();
				let error = error.map(str::to_string);
				Box::pin(async move {
					finished.lock().expect("finished").push((success, attempt, error));
				})
			})
		};
		let res = retry_assistant_call(
			produce,
			Some(&enabled_policy()),
			None,
			Some(&callbacks_with(None, Some(on_retry_finished), None)),
		)
		.await;
		assert_eq!(res.stop_reason, StopReason::Aborted);
		assert_eq!(produce_calls.load(Ordering::SeqCst), 2);
		assert_eq!(finished.lock().expect("finished").as_slice(), [(false, 1, None)]);
	}

	#[tokio::test]
	async fn does_not_retry_when_policy_is_disabled() {
		let (produce, calls) = counter_produce(Arc::new(Mutex::new(vec![faux_assistant_message(
			"", StopReason::Error, Some("terminated"),
		)])));
		let scheduled = recorded_events();
		let finished: Arc<Mutex<Vec<(bool, u32, Option<String>)>>> = Arc::new(Mutex::new(Vec::new()));
		let on_retry_scheduled: Arc<dyn Fn(u32, u32, u64, &str) -> BoxFuture<'static, ()> + Send + Sync> = {
			let scheduled = scheduled.clone();
			Arc::new(move |attempt, _, _, _| {
				let scheduled = scheduled.clone();
				Box::pin(async move {
					scheduled.lock().expect("events").push(format!("retry:{attempt}"));
				})
			})
		};
		let on_retry_finished: Arc<dyn Fn(bool, u32, Option<&str>) -> BoxFuture<'static, ()> + Send + Sync> = {
			let finished = finished.clone();
			Arc::new(move |success, attempt, error| {
				let finished = finished.clone();
				let error = error.map(str::to_string);
				Box::pin(async move {
					finished.lock().expect("finished").push((success, attempt, error));
				})
			})
		};
		let res = retry_assistant_call(
			produce,
			Some(&disabled_policy()),
			None,
			Some(&callbacks_with(Some(on_retry_scheduled), Some(on_retry_finished), None)),
		)
		.await;
		assert_eq!(res.stop_reason, StopReason::Error);
		assert_eq!(calls.load(Ordering::SeqCst), 1);
		assert!(scheduled.lock().expect("events").is_empty());
		assert!(finished.lock().expect("finished").is_empty());
	}

	#[tokio::test]
	async fn emits_on_retry_attempt_start_after_backoff_before_each_retried_call() {
		let events = recorded_events();
		let produce_calls = Arc::new(AtomicUsize::new(0));
		let counter = produce_calls.clone();
		let produce: Produce = {
			let events = events.clone();
			Box::new(move || {
				let counter = counter.clone();
				let events = events.clone();
				Box::pin(async move {
					let n = counter.fetch_add(1, Ordering::SeqCst);
					events.lock().expect("events").push(format!("produce:{n}"));
					if n < 2 {
						faux_assistant_message("", StopReason::Error, Some("terminated"))
					} else {
						faux_assistant_message("recovered", StopReason::Stop, None)
					}
				})
			})
		};
		let on_retry_scheduled: Arc<dyn Fn(u32, u32, u64, &str) -> BoxFuture<'static, ()> + Send + Sync> = {
			let events = events.clone();
			Arc::new(move |attempt, _, _, _| {
				let events = events.clone();
				Box::pin(async move {
					events.lock().expect("events").push(format!("retry:{attempt}"));
				})
			})
		};
		let on_retry_attempt_start: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync> = {
			let events = events.clone();
			Arc::new(move || {
				let events = events.clone();
				Box::pin(async move {
					events.lock().expect("events").push("attempt-start".to_string());
				})
			})
		};
		let res = retry_assistant_call(
			produce,
			Some(&enabled_policy()),
			None,
			Some(&callbacks_with(Some(on_retry_scheduled), None, Some(on_retry_attempt_start))),
		)
		.await;
		assert_eq!(res.content, vec![AssistantContent::Text(TextContent::new("recovered"))]);
		assert_eq!(
			*events.lock().expect("events"),
			vec![
				"produce:0".to_string(),
				"retry:1".to_string(),
				"attempt-start".to_string(),
				"produce:1".to_string(),
				"retry:2".to_string(),
				"attempt-start".to_string(),
				"produce:2".to_string(),
			]
		);
	}

	#[tokio::test(start_paused = true)]
	async fn aborts_backoff_sleep_via_signal_returns_aborted_message_and_finishes_unsuccessfully() {
		use crate::utils::abort::AbortController;

		let controller = AbortController::new();
		let signal = controller.signal();
		let (produce, calls) = counter_produce(Arc::new(Mutex::new(vec![faux_assistant_message(
			"", StopReason::Error, Some("terminated"),
		)])));
		let policy = RetryPolicy { enabled: true, max_retries: 5, base_delay_ms: 10_000, max_agent_delay_ms: None };
		let finished: Arc<Mutex<Vec<(bool, u32, Option<String>)>>> = Arc::new(Mutex::new(Vec::new()));
		let on_retry_finished: Arc<dyn Fn(bool, u32, Option<&str>) -> BoxFuture<'static, ()> + Send + Sync> = {
			let finished = finished.clone();
			Arc::new(move |success, attempt, error| {
				let finished = finished.clone();
				let error = error.map(str::to_string);
				Box::pin(async move {
					finished.lock().expect("finished").push((success, attempt, error));
				})
			})
		};
		let callbacks = callbacks_with(None, Some(on_retry_finished), None);
		let mut fut = std::pin::pin!(retry_assistant_call(
			produce,
			Some(&policy),
			Some(&signal),
			Some(&callbacks),
		));

		// Let one error call resolve and the first backoff sleep start, then abort.
		let _ = futures_util::poll!(fut.as_mut());
		assert_eq!(calls.load(Ordering::SeqCst), 1);
		controller.abort();

		let res = fut.await;
		assert_eq!(res.stop_reason, StopReason::Aborted);
		assert_eq!(res.error_message, None);
		assert_eq!(calls.load(Ordering::SeqCst), 1);
		assert_eq!(
			finished.lock().expect("finished").as_slice(),
			[(false, 1, Some("terminated".to_string()))]
		);
	}
}
