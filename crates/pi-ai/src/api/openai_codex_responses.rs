//! Port of `packages/ai/src/api/openai-codex-responses.ts` — the ChatGPT
//! Codex Responses stream adapter (request assembly, retry, SSE decoding,
//! Codex event mapping) over the [`crate::http::FetchFn`] seam.
//!
//! Deviations:
//! - **D25 — WebSocket transport omitted.** Upstream opens a cached
//!   per-session WebSocket (`responses_websockets=2026-02-06`) with
//!   continuation state, debug stats, and SSE fallback bookkeeping. The crate
//!   has no WebSocket dependency (and `transport: "sse"` upstream skips the
//!   whole path), so this port always takes the SSE path; the exported
//!   WebSocket debug-stat/teardown functions have no Rust counterpart. The
//!   `transport` option is accepted and `sse`/`auto` behave identically.
//! - **D26 — request-body zstd compression omitted.** Upstream compresses the
//!   SSE request body with Node `zlib.zstdCompressSync` when available and
//!   otherwise falls back to the uncompressed JSON; the Rust port always sends
//!   the uncompressed JSON (the upstream fallback path, identical semantics).
//! - **D19**: synchronous setup failures (`streamSimple` key check) surface as
//!   the stream `error` event (crate no-throw-across-stream rule).
//! - **D27 — codex-specific `"none"` reasoning effort is not representable**
//!   in the shared `options.openai_responses.reasoning_effort` group (which
//!   mirrors upstream `ThinkingLevel`); callers pass `None`, and the
//!   model-driven fallback (`reasoning` + `thinkingLevelMap.off`) is ported
//!   verbatim.
//! - Upstream `streamSimple` throws synchronously when the API key is missing;
//!   here it produces the standard error event (D1).

use std::pin::Pin;
use std::sync::Arc;

use base64::Engine as _;
use serde_json::{json, Map, Value};

use crate::api::constrained_sampling::create_grammar_tool_input_properties;
use crate::api::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::api::openai_responses_shared::{
    convert_responses_messages, convert_responses_tools, process_responses_stream,
    ConvertResponsesMessagesOptions, ConvertResponsesToolsOptions, OpenAIResponsesStreamOptions,
};
use crate::api::simple_options::build_base_options;
use crate::http::{FetchFn, HttpBody, HttpRequest, HttpResponse, ProviderHttpError};
use crate::models::{clamp_thinking_level, ProviderStreams};
use crate::types::{
    AssistantMessage, AssistantMessageEvent, CacheRetention, Model, ModelThinkingLevel,
    ProviderResponse, SimpleStreamOptions, StopReason, StreamDoneReason, StreamErrorReason,
    StreamOptions, ThinkingLevel, ThinkingLevelMap, TranscriptContext, Usage,
};
use crate::utils::diagnostics::now_ms;
use crate::utils::error_body::{format_provider_error, NormalizedProviderError};
use crate::utils::event_stream::{done_event, error_event, AssistantMessageEventStream};
use crate::utils::headers::headers_to_record;
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::transcript::{
    get_declared_tools, get_initial_system_message, resolve_transcript, resolve_transcript_tools,
};

const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
const DEFAULT_MAX_RETRIES: u32 = 0;
const BASE_DELAY_MS: u64 = 1000;
const DEFAULT_MAX_RETRY_DELAY_MS: u64 = 60_000;

fn codex_tool_call_providers() -> std::collections::BTreeSet<String> {
    ["openai", "openai-codex", "opencode"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// The exported API implementation (`openAICodexResponsesApi`).
pub fn openai_codex_responses_api() -> Arc<dyn ProviderStreams> {
    Arc::new(OpenAICodexResponsesApi)
}

struct OpenAICodexResponsesApi;

impl ProviderStreams for OpenAICodexResponsesApi {
    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        stream_codex_responses(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        stream_simple_codex_responses(model, context, options)
    }
}

// =============================================================================
// Retry helpers
// =============================================================================

fn is_terminal_rate_limit_error(error_text: &str) -> bool {
    // Upstream: /GoUsageLimitError|FreeUsageLimitError|Monthly usage limit
    // reached|available balance|insufficient_quota|out of budget|quota
    // exceeded|billing/i
    let lower = error_text.to_ascii_lowercase();
    [
        "gousagelimiterror",
        "freeusagelimiterror",
        "monthly usage limit reached",
        "available balance",
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn is_retryable_error(status: u16, error_text: &str) -> bool {
    if status == 429 && is_terminal_rate_limit_error(error_text) {
        return false;
    }
    if matches!(status, 429 | 500 | 502 | 503 | 504) {
        return true;
    }
    // /rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused/i
    let lower = error_text.to_ascii_lowercase();
    lower.contains("rate limit")
        || lower.contains("ratelimit")
        || lower.contains("rate-limit")
        || lower.contains("overloaded")
        || lower.contains("service unavailable")
        || lower.contains("service-unavailable")
        || lower.contains("upstream connect")
        || lower.contains("upstream-connect")
        || lower.contains("connection refused")
        || lower.contains("connection-refused")
}

fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

/// Upstream `getRetryAfterDelayMs` (headers of the failed response).
fn get_retry_after_delay_ms(headers: &[(String, String)]) -> Option<f64> {
    if let Some(retry_after_ms) = header_value(headers, "retry-after-ms") {
        if let Ok(millis) = retry_after_ms.parse::<f64>() {
            if millis.is_finite() {
                return Some(millis.max(0.0));
            }
        }
    }
    let retry_after = header_value(headers, "retry-after")?;
    if retry_after.is_empty() {
        return None;
    }
    if let Ok(seconds) = retry_after.parse::<f64>() {
        if seconds.is_finite() {
            return Some((seconds * 1000.0).max(0.0));
        }
    }
    // HTTP-date form: `Date.parse(retryAfter) - Date.now()`.
    if let Some(date) = httpdate_parse(retry_after) {
        return Some(((date - now_ms()) as f64).max(0.0));
    }
    None
}

/// Minimal RFC 7231 IMF-fixdate parser returning unix milliseconds
/// (`Date.parse` for the `Retry-After` HTTP-date form).
fn httpdate_parse(value: &str) -> Option<i64> {
    let mut day: Option<i64> = None;
    let mut month: Option<i64> = None;
    let mut year: Option<i64> = None;
    let mut time = (0i64, 0i64, 0i64);
    let month_names = [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ];
    for token in value.split_whitespace() {
        let clean = token.trim_end_matches(',');
        if let Ok(number) = clean.parse::<i64>() {
            if day.is_none() {
                day = Some(number);
            } else {
                // time h:m:s
                let parts: Vec<i64> = clean.split(':').filter_map(|p| p.parse().ok()).collect();
                if parts.len() == 3 {
                    time = (parts[0], parts[1], parts[2]);
                }
            }
        } else if let Some(index) = month_names
            .iter()
            .position(|name| clean.to_ascii_lowercase().starts_with(name))
        {
            month = Some(index as i64 + 1);
        } else if clean.len() == 4 && clean.chars().all(|c| c.is_ascii_digit()) {
            year = clean.parse().ok();
        }
    }
    let (day, month, year) = (day?, month?, year?);
    if !(1..=31).contains(&day) || time.0 > 23 || time.1 > 59 || time.2 > 60 {
        return None;
    }
    Some(
        days_from_civil(year, month, day) * 86_400_000
            + time.0 * 3_600_000
            + time.1 * 60_000
            + time.2 * 1000,
    )
}

/// Howard Hinnant's `days_from_civil` (proleptic Gregorian → unix days).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Upstream `RetryDelayExceededError`; not retried, thrown to the caller.
fn validate_retry_delay_ms(delay_ms: f64, options: Option<&StreamOptions>) -> Result<f64, String> {
    let max_retry_delay_ms = options
        .and_then(|o| o.base.max_retry_delay_ms)
        .unwrap_or(DEFAULT_MAX_RETRY_DELAY_MS) as f64;
    if max_retry_delay_ms > 0.0 && delay_ms > max_retry_delay_ms {
        return Err(format!(
            "Server requested {}s retry delay (max: {}s)",
            (delay_ms / 1000.0).ceil(),
            (max_retry_delay_ms / 1000.0).ceil()
        ));
    }
    Ok(delay_ms)
}

/// Upstream `sleep(ms, signal)`; `Err` when aborted while waiting.
async fn codex_sleep(
    ms: u64,
    signal: Option<&crate::utils::abort::AbortSignal>,
) -> Result<(), String> {
    if let Some(signal) = signal {
        if signal.aborted() {
            return Err("Request was aborted".to_string());
        }
        let wait = tokio::time::sleep(std::time::Duration::from_millis(ms));
        tokio::pin!(wait);
        tokio::select! {
            _ = &mut wait => Ok(()),
            _ = signal.aborted_future() => Err("Request was aborted".to_string()),
        }
    } else {
        tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
        Ok(())
    }
}

// =============================================================================
// Request building
// =============================================================================

/// Upstream `buildRequestBody` — ordered to match the upstream literal.
#[allow(clippy::too_many_arguments)]
fn build_request_body(
    model: &Model,
    context: &TranscriptContext,
    options: Option<&StreamOptions>,
    cache_session_id: Option<&str>,
    grammar_tool_input_properties: &std::collections::BTreeMap<String, String>,
) -> Result<Value, String> {
    let compat = model.compat.as_ref();
    let supports_strict_mode = compat.and_then(|c| c.supports_strict_mode).unwrap_or(true);
    let supports_openai_grammar_tools = compat
        .and_then(|c| c.supports_openai_grammar_tools)
        .unwrap_or(false);
    let supports_additional_tools = compat
        .and_then(|c| c.supports_additional_tools)
        .unwrap_or(false);
    let supports_tool_search = compat.and_then(|c| c.supports_tool_search).unwrap_or(false);
    let transcript_tools = resolve_transcript_tools(
        context.messages(),
        supports_additional_tools || supports_tool_search,
    );
    let messages = convert_responses_messages(
        model,
        context,
        &codex_tool_call_providers(),
        Some(&ConvertResponsesMessagesOptions {
            include_system_prompt: Some(false),
            grammar_tool_input_properties: Some(grammar_tool_input_properties.clone()),
            supports_mid_convo_system_messages: compat
                .and_then(|c| c.supports_mid_convo_system_messages)
                .unwrap_or(false),
            supports_additional_tools,
            supports_tool_search,
            tool_options: Some(ConvertResponsesToolsOptions {
                strict: None,
                supports_strict_mode: Some(supports_strict_mode),
                supports_openai_grammar_tools: Some(supports_openai_grammar_tools),
                tool_search_result: false,
            }),
        }),
    )?;

    let initial_system_message = get_initial_system_message(context.messages());
    let instructions = initial_system_message
        .map(crate::utils::text::get_system_message_text)
        .unwrap_or_default();
    let opts = options;
    let mut body = Map::new();
    body.insert("model".to_string(), json!(model.id()));
    body.insert("store".to_string(), json!(false));
    body.insert("stream".to_string(), json!(true));
    body.insert(
        "instructions".to_string(),
        json!(if instructions.is_empty() {
            "You are a helpful assistant."
        } else {
            instructions.as_str()
        }),
    );
    body.insert("input".to_string(), Value::Array(messages));
    // D27: `options.textVerbosity` has no shared options-group field (types.rs
    // is outside this slice); the upstream default "low" is used.
    body.insert("text".to_string(), json!({ "verbosity": "low" }));
    body.insert(
        "include".to_string(),
        json!(["reasoning.encrypted_content"]),
    );
    if let Some(key) = cache_session_id {
        body.insert("prompt_cache_key".to_string(), json!(key));
    }
    body.insert(
        "tool_choice".to_string(),
        opts.and_then(|o| o.openai_responses.tool_choice.clone())
            .unwrap_or_else(|| json!("auto")),
    );
    body.insert("parallel_tool_calls".to_string(), json!(true));

    if let Some(temperature) = opts.and_then(|o| o.temperature) {
        body.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(service_tier) = opts.and_then(|o| o.openai_responses.service_tier.as_deref()) {
        body.insert("service_tier".to_string(), json!(service_tier));
    }
    if !transcript_tools.request_tools.is_empty() {
        body.insert(
            "tools".to_string(),
            Value::Array(convert_responses_tools(
                &transcript_tools.request_tools,
                Some(&ConvertResponsesToolsOptions {
                    strict: None,
                    supports_strict_mode: Some(supports_strict_mode),
                    supports_openai_grammar_tools: Some(supports_openai_grammar_tools),
                    tool_search_result: false,
                }),
            )?),
        );
    }

    // Reasoning effort resolution (upstream inline block).
    let reasoning_effort = opts.and_then(|o| o.openai_responses.reasoning_effort);
    let reasoning_summary = opts.and_then(|o| o.openai_responses.reasoning_summary);
    let thinking_level_map = model.thinking_level_map.as_ref();
    if let Some(effort) = reasoning_effort {
        if let Some(effort) = resolve_reasoning_effort(thinking_level_map, effort) {
            let summary = match reasoning_summary {
                Some(Some(summary)) => serde_json::to_value(summary).unwrap_or(json!("auto")),
                _ => json!("auto"),
            };
            body.insert(
                "reasoning".to_string(),
                json!({ "effort": effort, "summary": summary }),
            );
        }
    } else if model.reasoning && !matches!(fallback_reasoning_effort(thinking_level_map), None) {
        // `else if (model.reasoning && model.thinkingLevelMap?.off !== null)`;
        // `fallback_reasoning_effort` returns None exactly when `off` is null.
        let effort =
            fallback_reasoning_effort(thinking_level_map).unwrap_or_else(|| "none".to_string());
        body.insert("reasoning".to_string(), json!({ "effort": effort }));
    }

    Ok(Value::Object(body))
}

/// `model.thinkingLevelMap[effort] ?? effort`, returning `None` when the map
/// marks the level unsupported (JSON `null` skips `body.reasoning`).
fn resolve_reasoning_effort(
    map: Option<&ThinkingLevelMap>,
    effort: ThinkingLevel,
) -> Option<String> {
    let key = match effort {
        ThinkingLevel::Minimal => ModelThinkingLevel::Minimal,
        ThinkingLevel::Low => ModelThinkingLevel::Low,
        ThinkingLevel::Medium => ModelThinkingLevel::Medium,
        ThinkingLevel::High => ModelThinkingLevel::High,
        ThinkingLevel::Xhigh => ModelThinkingLevel::Xhigh,
        ThinkingLevel::Max => ModelThinkingLevel::Max,
    };
    match map.and_then(|m| m.get(&key)) {
        Some(Some(value)) => Some(value.clone()),
        Some(None) => None,
        None => Some(serde_json_level(effort.into()).to_string()),
    }
}

fn serde_json_level(level: ModelThinkingLevel) -> &'static str {
    match level {
        ModelThinkingLevel::Off => "none",
        ModelThinkingLevel::Minimal => "minimal",
        ModelThinkingLevel::Low => "low",
        ModelThinkingLevel::Medium => "medium",
        ModelThinkingLevel::High => "high",
        ModelThinkingLevel::Xhigh => "xhigh",
        ModelThinkingLevel::Max => "max",
    }
}

/// The `else if (model.reasoning ...)` fallback: `effort: map.off ?? "none"`,
/// skipped entirely when `off` is `null`.
fn fallback_reasoning_effort(map: Option<&ThinkingLevelMap>) -> Option<String> {
    match map.and_then(|m| m.get(&ModelThinkingLevel::Off)) {
        Some(Some(value)) => Some(value.clone()),
        Some(None) => None,
        None => Some("none".to_string()),
    }
}

fn get_service_tier_cost_multiplier(model_id: &str, service_tier: Option<&Value>) -> f64 {
    match service_tier.and_then(Value::as_str) {
        Some("flex") => 0.5,
        Some("priority") => {
            if model_id == "gpt-5.5" {
                2.5
            } else {
                2.0
            }
        }
        _ => 1.0,
    }
}

fn apply_service_tier_pricing(usage: &mut Usage, service_tier: Option<&Value>, model_id: &str) {
    let multiplier = get_service_tier_cost_multiplier(model_id, service_tier);
    if multiplier == 1.0 {
        return;
    }
    usage.cost.input *= multiplier;
    usage.cost.output *= multiplier;
    usage.cost.cache_read *= multiplier;
    usage.cost.cache_write *= multiplier;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

/// Upstream `resolveCodexServiceTier`.
fn resolve_codex_service_tier(
    response_tier: Option<&Value>,
    request_tier: Option<&Value>,
) -> Option<Value> {
    if response_tier.and_then(Value::as_str) == Some("default")
        && matches!(
            request_tier.and_then(Value::as_str),
            Some("flex") | Some("priority")
        )
    {
        return request_tier.cloned();
    }
    response_tier.cloned().or_else(|| request_tier.cloned())
}

/// Upstream `resolveCodexUrl`.
fn resolve_codex_url(base_url: Option<&str>) -> String {
    let raw = match base_url.map(str::trim) {
        Some(trimmed) if !trimmed.is_empty() => trimmed.to_string(),
        _ => DEFAULT_CODEX_BASE_URL.to_string(),
    };
    let normalized = raw.trim_end_matches('/').to_string();
    if normalized.ends_with("/codex/responses") {
        return normalized;
    }
    if normalized.ends_with("/codex") {
        return format!("{normalized}/responses");
    }
    format!("{normalized}/codex/responses")
}

// =============================================================================
// SSE decoding + Codex event mapping
// =============================================================================

#[derive(Default)]
struct CodexSseDecoder {
    buffer: String,
}

impl CodexSseDecoder {
    /// Feed a chunk; returns parsed data payloads and any protocol error
    /// (invalid JSON → `CodexProtocolError` message).
    fn feed(&mut self, chunk: &str, done: bool) -> (Vec<Value>, Option<String>) {
        self.buffer.push_str(chunk);
        if done && !self.buffer.trim().is_empty() {
            self.buffer.push_str("\n\n");
        }
        let mut events = Vec::new();
        let mut error = None;
        while let Some(index) = self.buffer.find("\n\n") {
            let frame: String = self.buffer.drain(..index + 2).collect();
            let data_lines: Vec<&str> = frame
                .split('\n')
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|line| line.trim())
                .collect();
            if data_lines.is_empty() {
                continue;
            }
            let data = data_lines.join("\n").trim().to_string();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            match serde_json::from_str::<Value>(&data) {
                Ok(event) => events.push(event),
                Err(cause) => {
                    error = Some(format!("Invalid Codex SSE JSON: {cause}"));
                }
            }
        }
        (events, error)
    }
}

/// Result of draining the mapped Codex event stream: terminal metadata
/// hoisted off `response.completed` (upstream `mapCodexEvents` side effects).
#[derive(Default)]
struct CodexEventMapping {
    end_turn: Option<bool>,
}

/// Decode an SSE body into the mapped Codex Responses event stream
/// (`parseSSE` + `mapCodexEvents`). `error` / `response.failed` events become
/// stream errors carrying the upstream message; terminal events are rewritten
/// to `response.completed` with a normalized `response.status`.
fn decode_codex_events(
    body: HttpBody,
    mapping: Arc<parking_lot::Mutex<CodexEventMapping>>,
) -> Pin<Box<dyn futures_core::Stream<Item = Result<Value, ProviderHttpError>> + Send>> {
    use futures_util::StreamExt;
    let chunks: futures_core::stream::BoxStream<'static, Result<Vec<u8>, ProviderHttpError>> =
        match body {
            HttpBody::Empty => Box::pin(futures_util::stream::empty()),
            HttpBody::Bytes(bytes) => Box::pin(futures_util::stream::iter(vec![Ok(bytes)])),
            HttpBody::Stream(stream) => stream,
        };
    let mapper = CodexEventMapper { mapping };
    Box::pin(futures_util::stream::unfold(
        (
            chunks,
            CodexSseDecoder::default(),
            mapper,
            std::collections::VecDeque::new(),
            false,
        ),
        move |(mut chunks, mut decoder, mapper, mut pending, mut finished)| async move {
            loop {
                if let Some(mapped) = pending.pop_front() {
                    match mapped {
                        Ok(event) => {
                            return Some((Ok(event), (chunks, decoder, mapper, pending, finished)))
                        }
                        Err(message) => {
                            finished = true;
                            return Some((
                                Err(ProviderHttpError::new(message)),
                                (chunks, decoder, mapper, pending, finished),
                            ));
                        }
                    }
                }
                if finished {
                    return None;
                }
                match chunks.next().await {
                    Some(Ok(bytes)) => {
                        let done = false;
                        let text = String::from_utf8_lossy(&bytes).into_owned();
                        let (events, error) = decoder.feed(&text, done);
                        if let Some(message) = error {
                            finished = true;
                            return Some((
                                Err(ProviderHttpError::new(message)),
                                (chunks, decoder, mapper, pending, finished),
                            ));
                        }
                        for event in events {
                            match mapper.map_event(event) {
                                Ok(Some(mapped)) => pending.push_back(Ok(mapped)),
                                Ok(None) => {}
                                Err(message) => pending.push_back(Err(message)),
                            }
                        }
                    }
                    Some(Err(error)) => {
                        finished = true;
                        return Some((Err(error), (chunks, decoder, mapper, pending, finished)));
                    }
                    None => {
                        let (events, error) = decoder.feed("", true);
                        if let Some(message) = error {
                            finished = true;
                            return Some((
                                Err(ProviderHttpError::new(message)),
                                (chunks, decoder, mapper, pending, finished),
                            ));
                        }
                        for event in events {
                            match mapper.map_event(event) {
                                Ok(Some(mapped)) => pending.push_back(Ok(mapped)),
                                Ok(None) => {}
                                Err(message) => pending.push_back(Err(message)),
                            }
                        }
                        finished = true;
                    }
                }
            }
        },
    ))
}

struct CodexEventMapper {
    mapping: Arc<parking_lot::Mutex<CodexEventMapping>>,
}

impl CodexEventMapper {
    /// Upstream `mapCodexEvents` for a single event: `None` drops the event,
    /// `Err` fails the stream, `Some` forwards (terminal events rewritten).
    fn map_event(&self, event: Value) -> Result<Option<Value>, String> {
        let type_ = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if type_.is_empty() {
            return Ok(None);
        }
        if type_ == "error" {
            let (code, message) = extract_codex_event_error(&event);
            return Err(format!(
                "Codex error: {}",
                message
                    .or(code)
                    .map(str::to_string)
                    .unwrap_or_else(|| event.to_string())
            ));
        }
        if type_ == "response.failed" {
            let response = event.get("response");
            let error = response.and_then(|r| r.get("error"));
            let code = error.and_then(|e| e.get("code")).and_then(Value::as_str);
            let message = error.and_then(|e| e.get("message")).and_then(Value::as_str);
            return Err(message
                .or(code)
                .unwrap_or("Codex response failed")
                .to_string());
        }
        if type_ == "response.done"
            || type_ == "response.completed"
            || type_ == "response.incomplete"
        {
            let response = event.get("response").cloned();
            if let Some(end_turn) = response
                .as_ref()
                .and_then(|r| r.get("end_turn"))
                .and_then(Value::as_bool)
            {
                self.mapping.lock().end_turn = Some(end_turn);
            }
            let mut normalized = event.clone();
            normalized["type"] = json!("response.completed");
            if let Some(response) = response {
                let mut response = response;
                if let Some(status) = response.get("status") {
                    let normalized_status = normalize_codex_status(status);
                    response["status"] = match normalized_status {
                        Some(status) => json!(status),
                        None => Value::Null,
                    };
                }
                normalized["response"] = response;
            }
            return Ok(Some(normalized));
        }
        Ok(Some(event))
    }
}

fn extract_codex_event_error(event: &Value) -> (Option<&str>, Option<&str>) {
    let nested = event.get("error").filter(|e| e.is_object());
    let code = event
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| nested.and_then(|n| n.get("code")).and_then(Value::as_str));
    let message = event.get("message").and_then(Value::as_str).or_else(|| {
        nested
            .and_then(|n| n.get("message"))
            .and_then(Value::as_str)
    });
    (code, message)
}

/// Upstream `normalizeCodexStatus`.
fn normalize_codex_status(status: &Value) -> Option<&'static str> {
    let status = status.as_str()?;
    match status {
        "completed" => Some("completed"),
        "incomplete" => Some("incomplete"),
        "failed" => Some("failed"),
        "cancelled" => Some("cancelled"),
        "queued" => Some("queued"),
        "in_progress" => Some("in_progress"),
        _ => None,
    }
}

// =============================================================================
// Error parsing & auth
// =============================================================================

/// Upstream `parseErrorResponse`.
fn parse_error_response(raw: &str, status: u16, status_text: &str) -> (String, Option<String>) {
    let mut message = if raw.is_empty() {
        if status_text.is_empty() {
            "Request failed".to_string()
        } else {
            status_text.to_string()
        }
    } else {
        raw.to_string()
    };
    let mut friendly_message: Option<String> = None;
    if let Ok(parsed) = serde_json::from_str::<Value>(raw) {
        if let Some(err) = parsed.get("error").filter(|e| e.is_object()) {
            let code = err
                .get("code")
                .and_then(Value::as_str)
                .or_else(|| err.get("type").and_then(Value::as_str))
                .unwrap_or("");
            let code_matches = [
                "usage_limit_reached",
                "usage_not_included",
                "rate_limit_exceeded",
            ]
            .iter()
            .any(|needle| {
                code.to_ascii_lowercase()
                    .contains(&needle.to_ascii_lowercase())
            });
            if code_matches || status == 429 {
                let plan = err
                    .get("plan_type")
                    .and_then(Value::as_str)
                    .map(|plan| format!(" ({} plan)", plan.to_ascii_lowercase()))
                    .unwrap_or_default();
                let resets_at = err.get("resets_at").and_then(Value::as_f64);
                // `err.resets_at ? ...` — 0/absent yields no suffix.
                let when = match resets_at.filter(|resets_at| *resets_at != 0.0) {
                    Some(resets_at) => {
                        let mins = ((resets_at * 1000.0 - now_ms() as f64) / 60_000.0)
                            .round()
                            .max(0.0) as i64;
                        format!(" Try again in ~{mins} min.")
                    }
                    None => String::new(),
                };
                friendly_message = Some(
                    format!("You have hit your ChatGPT usage limit{plan}.{when}")
                        .trim()
                        .to_string(),
                );
            }
            if let Some(err_message) = err.get("message").and_then(Value::as_str) {
                message = err_message.to_string();
            } else if let Some(friendly) = &friendly_message {
                message = friendly.clone();
            }
        }
    }
    (message, friendly_message)
}

/// Upstream `extractAccountId` — decodes the ChatGPT account id from the JWT.
fn extract_account_id(token: &str) -> Result<String, String> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err("Failed to extract accountId from token".to_string());
    }
    let payload = base64::engine::general_purpose::STANDARD
        .decode(parts[1])
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .and_then(|text| serde_json::from_str::<Value>(&text).ok());
    let account_id = payload
        .as_ref()
        .and_then(|payload| payload.get(JWT_CLAIM_PATH))
        .and_then(|claim| claim.get("chatgpt_account_id"))
        .and_then(Value::as_str);
    match account_id {
        Some(account_id) => Ok(account_id.to_string()),
        None => Err("Failed to extract accountId from token".to_string()),
    }
}

/// Upstream `buildBaseCodexHeaders` + `buildSSEHeaders`.
fn build_sse_headers(
    model_headers: Option<&std::collections::BTreeMap<String, String>>,
    additional_headers: Option<&crate::types::ProviderHeaders>,
    account_id: &str,
    token: &str,
    session_id: Option<&str>,
) -> Vec<(String, String)> {
    // Upstream v1.1.0: the defaults are seeded first so model and caller
    // headers can override `originator` and `User-Agent`, matching the other
    // providers (#10429). Later entries use `Headers.set` semantics.
    let mut headers: Vec<(String, String)> = vec![
        ("originator".to_string(), "pi".to_string()),
        ("User-Agent".to_string(), get_pi_user_agent()),
    ];
    if let Some(model_headers) = model_headers {
        for (key, value) in model_headers {
            headers.retain(|(name, _)| !name.eq_ignore_ascii_case(key));
            headers.push((key.clone(), value.clone()));
        }
    }
    if let Some(additional) = additional_headers {
        for (key, value) in additional {
            headers.retain(|(name, _)| !name.eq_ignore_ascii_case(key));
            if let Some(value) = value {
                headers.push((key.clone(), value.clone()));
            }
        }
    }
    for (key, value) in [
        ("Authorization", format!("Bearer {token}")),
        ("chatgpt-account-id", account_id.to_string()),
        ("OpenAI-Beta", "responses=experimental".to_string()),
        ("accept", "text/event-stream".to_string()),
        ("content-type", "application/json".to_string()),
    ] {
        headers.retain(|(name, _)| !name.eq_ignore_ascii_case(key));
        headers.push((key.to_string(), value));
    }
    if let Some(session_id) = session_id {
        headers.retain(|(name, _)| name != "session-id" && name != "x-client-request-id");
        headers.push(("session-id".to_string(), session_id.to_string()));
        headers.push(("x-client-request-id".to_string(), session_id.to_string()));
    }
    headers
}

// =============================================================================
// Stream functions
// =============================================================================

fn new_output(model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api().to_string(),
        provider: model.provider().to_string(),
        model: model.id().to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_ms(),
        duration_ms: None,
    }
}

fn stream_codex_responses(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    let normalized_context = resolve_transcript(
        context,
        Some(
            model
                .compat
                .as_ref()
                .and_then(|c| c.supports_mid_convo_system_messages)
                .unwrap_or(false),
        ),
    );
    let model = model.clone();
    let options = options.clone();
    let handle = stream.clone();
    tokio::spawn(async move {
        let mut output = new_output(&model);
        let result = run_codex(&model, &normalized_context, &options, &handle, &mut output).await;
        match result {
            Ok(()) => {
                let reason = match output.stop_reason {
                    StopReason::Stop => StreamDoneReason::Stop,
                    StopReason::Length => StreamDoneReason::Length,
                    StopReason::ToolUse => StreamDoneReason::ToolUse,
                    StopReason::Deferred => StreamDoneReason::Deferred,
                    // Guarded by run(): error/aborted/pending never reach here.
                    _ => StreamDoneReason::Stop,
                };
                handle.push(done_event(output, reason));
            }
            Err(error_message) => {
                let aborted = options
                    .base
                    .signal
                    .as_ref()
                    .map(|s| s.aborted())
                    .unwrap_or(false);
                output.stop_reason = if aborted {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                output.error_message = Some(format_provider_error(
                    &NormalizedProviderError {
                        status: None,
                        body: None,
                        message: error_message,
                        message_carries_body: true,
                    },
                    None,
                ));
                handle.push(error_event(
                    output,
                    if aborted {
                        StreamErrorReason::Aborted
                    } else {
                        StreamErrorReason::Error
                    },
                ));
            }
        }
    });
    stream
}

/// The request/response/stream pipeline (`stream` body between stream
/// construction and the terminal event). Returns `Err` with the upstream
/// error text on failure.
async fn run_codex(
    model: &Model,
    normalized_context: &TranscriptContext,
    options: &StreamOptions,
    stream: &AssistantMessageEventStream,
    output: &mut AssistantMessage,
) -> Result<(), String> {
    let abort_message = || "Request was aborted".to_string();
    let is_aborted = || {
        options
            .base
            .signal
            .as_ref()
            .map(|s| s.aborted())
            .unwrap_or(false)
    };
    let Some(api_key) = options
        .base
        .api_key
        .as_deref()
        .map(str::to_string)
        .filter(|key| !key.is_empty())
    else {
        return Err(format!("No API key for provider: {}", model.provider()));
    };
    let account_id = extract_account_id(&api_key)?;
    let compat_grammar = model
        .compat
        .as_ref()
        .and_then(|c| c.supports_openai_grammar_tools)
        .unwrap_or(false);
    let declared_tools = get_declared_tools(normalized_context.messages());
    let grammar_tool_input_properties =
        create_grammar_tool_input_properties(&declared_tools, compat_grammar)?;
    let cache_session_id = if options.cache_retention == Some(CacheRetention::None) {
        None
    } else {
        options.session_id.as_deref()
    };
    let codex_session_id = clamp_openai_prompt_cache_key(cache_session_id);
    let mut body = build_request_body(
        model,
        normalized_context,
        Some(options),
        codex_session_id.as_deref(),
        &grammar_tool_input_properties,
    )?;
    if let Some(on_payload) = &options.base.on_payload {
        if let Some(next) = on_payload(body.clone(), model.clone()).await {
            body = next;
        }
    }
    let sse_headers = build_sse_headers(
        model.base.headers.as_ref(),
        options.base.headers.as_ref(),
        &account_id,
        &api_key,
        codex_session_id.as_deref(),
    );
    let body_json = serde_json::to_vec(&body).map_err(|error| error.to_string())?;
    let url = resolve_codex_url(Some(&model.base.base_url));

    let fetch: FetchFn = options.base.fetch.clone().unwrap_or_default();
    let max_retries = options.base.max_retries.unwrap_or(DEFAULT_MAX_RETRIES);

    let mut response: Option<HttpResponse> = None;
    let mut last_error: Option<String> = None;

    let mut attempt: u32 = 0;
    while attempt <= max_retries {
        if is_aborted() {
            return Err(abort_message());
        }
        let mut request = HttpRequest::new("POST", &url);
        request.headers = sse_headers.clone();
        request.body = HttpBody::Bytes(body_json.clone());
        request.timeout_ms = options.base.timeout_ms;
        request.signal = options.base.signal.clone();

        let send_result = fetch.send(request).await;
        match send_result {
            Ok(received) => {
                if let Some(on_response) = &options.base.on_response {
                    on_response(
                        ProviderResponse {
                            status: received.status,
                            headers: headers_to_record(&received.headers),
                        },
                        model.clone(),
                    )
                    .await;
                }
                if received.ok() {
                    response = Some(received);
                    break;
                }
                let error_text = received
                    .body
                    .collect()
                    .await
                    .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                    .unwrap_or_default();
                if attempt < max_retries && is_retryable_error(received.status, &error_text) {
                    let delay_ms = match get_retry_after_delay_ms(&received.headers) {
                        None => BASE_DELAY_MS as f64 * 2f64.powi(attempt as i32),
                        Some(retry_after) => validate_retry_delay_ms(retry_after, Some(options))?,
                    };
                    codex_sleep(delay_ms as u64, options.base.signal.as_ref()).await?;
                    attempt += 1;
                    continue;
                }
                let (message, friendly) = parse_error_response(&error_text, received.status, "");
                let _ = friendly;
                return Err(message);
            }
            Err(error) => {
                let message = if is_aborted() {
                    abort_message()
                } else {
                    error.message.clone()
                };
                if message == "Request was aborted" {
                    return Err(abort_message());
                }
                // Network errors are retryable.
                if attempt < max_retries && !message.contains("usage limit") {
                    let delay_ms = BASE_DELAY_MS * 2u64.pow(attempt);
                    codex_sleep(delay_ms, options.base.signal.as_ref()).await?;
                    attempt += 1;
                    last_error = Some(message);
                    continue;
                }
                return Err(message);
            }
        }
    }

    let Some(response) = response else {
        return Err(last_error.unwrap_or_else(|| "Failed after retries".to_string()));
    };

    stream.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });

    let mapping = Arc::new(parking_lot::Mutex::new(CodexEventMapping::default()));
    let mut event_stream = Box::pin(decode_codex_events(response.body, mapping.clone()));

    let model_id = model.id().to_string();
    let stream_options = OpenAIResponsesStreamOptions {
        on_provider_stream_event: options.on_provider_stream_event.clone(),
        service_tier: options
            .openai_responses
            .service_tier
            .as_deref()
            .map(|tier| json!(tier)),
        grammar_tool_input_properties,
        resolve_service_tier: Some(Arc::new(resolve_codex_service_tier)),
        apply_service_tier_pricing: Some(Arc::new(
            move |usage: &mut Usage, service_tier: Option<&Value>| {
                apply_service_tier_pricing(usage, service_tier, &model_id);
            },
        )),
    };

    process_responses_stream(
        &mut event_stream,
        output,
        stream,
        model,
        Some(&stream_options),
    )
    .await?;

    output.end_turn = mapping.lock().end_turn;

    if is_aborted() {
        return Err(abort_message());
    }
    // `assertSuccessfulOutput`.
    if output.stop_reason == StopReason::Pending {
        return Err("Codex stream ended without a stop reason".to_string());
    }
    if output.stop_reason == StopReason::Error || output.stop_reason == StopReason::Aborted {
        return Err(output
            .error_message
            .clone()
            .unwrap_or_else(|| "An unknown error occurred".to_string()));
    }
    Ok(())
}

fn stream_simple_codex_responses(
    model: &Model,
    context: &TranscriptContext,
    options: &SimpleStreamOptions,
) -> AssistantMessageEventStream {
    // Upstream throws synchronously before a stream exists; Rust surfaces the
    // failure as the stream error event (D19).
    let Some(api_key) = options
        .base
        .base
        .api_key
        .as_deref()
        .map(str::to_string)
        .filter(|key| !key.is_empty())
    else {
        let stream = AssistantMessageEventStream::new();
        let mut message = new_output(model);
        message.stop_reason = StopReason::Error;
        message.error_message = Some(format!("No API key for provider: {}", model.provider()));
        let reason = StreamErrorReason::Error;
        stream.push(error_event(message, reason));
        return stream;
    };

    let mut base = build_base_options(model, context, options, Some(&api_key));
    base.openai_responses.tool_choice = options.tool_choice.map(|choice| match choice {
        crate::types::ToolChoice::Auto => json!("auto"),
        crate::types::ToolChoice::None => json!("none"),
    });
    let clamped_reasoning = options
        .reasoning
        .map(|reasoning| clamp_thinking_level(model, ModelThinkingLevel::from(reasoning)));
    // `clampedReasoning === "off" ? undefined : clampedReasoning`.
    base.openai_responses.reasoning_effort = clamped_reasoning.and_then(|level| match level {
        ModelThinkingLevel::Off => None,
        ModelThinkingLevel::Minimal => Some(ThinkingLevel::Minimal),
        ModelThinkingLevel::Low => Some(ThinkingLevel::Low),
        ModelThinkingLevel::Medium => Some(ThinkingLevel::Medium),
        ModelThinkingLevel::High => Some(ThinkingLevel::High),
        ModelThinkingLevel::Xhigh => Some(ThinkingLevel::Xhigh),
        ModelThinkingLevel::Max => Some(ThinkingLevel::Max),
    });

    stream_codex_responses(model, context, &base)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Upstream test/openai-codex-stream.test.ts URL-resolution cases
    // (`resolveCodexUrl` is exercised through every mocked request URL).
    #[test]
    fn resolves_codex_urls() {
        assert_eq!(
            resolve_codex_url(None),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            resolve_codex_url(Some("https://chatgpt.com/backend-api")),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            resolve_codex_url(Some("https://chatgpt.com/backend-api/")),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            resolve_codex_url(Some("https://proxy.example/codex")),
            "https://proxy.example/codex/responses"
        );
        assert_eq!(
            resolve_codex_url(Some("https://proxy.example/codex/responses")),
            "https://proxy.example/codex/responses"
        );
        assert_eq!(
            resolve_codex_url(Some("  ")),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn extracts_account_id_from_jwt() {
        let payload = base64::engine::general_purpose::STANDARD
            .encode(r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acc_test"}}"#);
        let token = format!("aaa.{payload}.bbb");
        assert_eq!(extract_account_id(&token).unwrap(), "acc_test");
        assert_eq!(
            extract_account_id("aaa.bbb").unwrap_err(),
            "Failed to extract accountId from token"
        );
        assert_eq!(
            extract_account_id("aaa.aaa.bbb").unwrap_err(),
            "Failed to extract accountId from token"
        );
    }

    #[test]
    fn classifies_retryable_errors() {
        assert!(is_retryable_error(429, "slow down"));
        assert!(is_retryable_error(500, "oops"));
        assert!(is_retryable_error(503, "oops"));
        assert!(!is_retryable_error(429, "insufficient_quota"));
        assert!(!is_retryable_error(429, "Monthly usage limit reached"));
        assert!(is_retryable_error(400, "connection refused"));
        assert!(!is_retryable_error(400, "bad request"));
    }

    #[test]
    fn parses_friendly_usage_limit_errors() {
        let (message, friendly) = parse_error_response(
            r#"{"error":{"code":"usage_limit_reached","message":"Limit hit","plan_type":"PRO","resets_at":0}}"#,
            429,
            "",
        );
        assert_eq!(message, "Limit hit");
        assert_eq!(
            friendly.unwrap(),
            "You have hit your ChatGPT usage limit (pro plan)."
        );
    }

    #[test]
    fn lets_model_and_caller_headers_override_originator_and_user_agent() {
        // #10429: model and caller headers override the `originator` and
        // `User-Agent` defaults, matching the other providers.
        let mut model_headers = std::collections::BTreeMap::new();
        model_headers.insert("originator".to_string(), "my-app".to_string());
        let mut caller_headers: std::collections::BTreeMap<String, Option<String>> =
            std::collections::BTreeMap::new();
        caller_headers.insert("user-agent".to_string(), Some("my-app/1.0".to_string()));
        let headers = build_sse_headers(
            Some(&model_headers),
            Some(&caller_headers),
            "acc",
            "tok",
            None,
        );
        let get = |name: &str| {
            headers
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, value)| value.clone())
        };
        assert_eq!(get("originator").unwrap(), "my-app");
        assert_eq!(get("User-Agent").unwrap(), "my-app/1.0");
        assert_eq!(get("Authorization").unwrap(), "Bearer tok");
    }

    #[test]
    fn builds_sse_headers_with_session() {
        let mut model_headers = std::collections::BTreeMap::new();
        model_headers.insert(
            "Copilot-Integration-Id".to_string(),
            "vscode-chat".to_string(),
        );
        let headers =
            build_sse_headers(Some(&model_headers), None, "acc", "tok", Some("session-1"));
        let get = |name: &str| {
            headers
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, value)| value.clone())
        };
        assert_eq!(get("Authorization").unwrap(), "Bearer tok");
        assert_eq!(get("chatgpt-account-id").unwrap(), "acc");
        assert_eq!(get("originator").unwrap(), "pi");
        assert_eq!(get("OpenAI-Beta").unwrap(), "responses=experimental");
        assert_eq!(get("accept").unwrap(), "text/event-stream");
        assert_eq!(get("content-type").unwrap(), "application/json");
        assert_eq!(get("session-id").unwrap(), "session-1");
        assert_eq!(get("x-client-request-id").unwrap(), "session-1");
        // Model headers survive unless a static header overwrites them.
        assert_eq!(get("Copilot-Integration-Id").unwrap(), "vscode-chat");
        assert_eq!(get("User-Agent").unwrap(), get_pi_user_agent());
    }

    #[test]
    fn maps_codex_error_events() {
        let mapper = CodexEventMapper {
            mapping: Arc::default(),
        };
        let error = mapper
            .map_event(
                serde_json::json!({ "type": "error", "code": "boom", "message": "bad thing" }),
            )
            .unwrap_err();
        assert_eq!(error, "Codex error: bad thing");

        let error = mapper
            .map_event(serde_json::json!({
                "type": "response.failed",
                "response": { "error": { "code": "x", "message": "nope" } }
            }))
            .unwrap_err();
        assert_eq!(error, "nope");
    }

    #[test]
    fn normalizes_terminal_events_and_hoists_end_turn() {
        let mapping = Arc::new(parking_lot::Mutex::new(CodexEventMapping::default()));
        let mapper = CodexEventMapper {
            mapping: mapping.clone(),
        };
        let mapped = mapper
            .map_event(serde_json::json!({
                "type": "response.incomplete",
                "response": {
                    "status": "incomplete",
                    "end_turn": true,
                    "incomplete_details": { "reason": "max_output_tokens" },
                    "usage": {}
                }
            }))
            .unwrap()
            .unwrap();
        assert_eq!(mapped["type"], "response.completed");
        assert_eq!(mapped["response"]["status"], "incomplete");
        assert_eq!(mapping.lock().end_turn, Some(true));
    }

    #[test]
    fn applies_service_tier_pricing() {
        let mut usage = Usage {
            input: 10,
            output: 5,
            cache_read: 2,
            cache_write: 1,
            reasoning: None,
            total_tokens: 0,
            cache_write_1h: None,
            cost: crate::types::UsageCost {
                input: 4.0,
                output: 2.0,
                cache_read: 1.0,
                cache_write: 0.5,
                total: 7.5,
            },
        };
        apply_service_tier_pricing(&mut usage, Some(&json!("flex")), "gpt-5.2");
        assert!((usage.cost.total - 3.75).abs() < 1e-9);
        // gpt-5.5 priority is 2.5x.
        let mut usage = Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            reasoning: None,
            total_tokens: 0,
            cache_write_1h: None,
            cost: crate::types::UsageCost {
                input: 2.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 2.0,
            },
        };
        apply_service_tier_pricing(&mut usage, Some(&json!("priority")), "gpt-5.5");
        assert!((usage.cost.total - 5.0).abs() < 1e-9);
    }

    // Port of upstream test/openai-codex-stream.test.ts "streams SSE responses
    // and forwards raw provider events" (fetch-fake variant of the crate's
    // `options.fetch` seam).
    #[tokio::test]
    async fn streams_sse_responses() {
        use crate::types::{
            AssistantContent, AssistantMessageEvent, Context, Message, UserContentField,
            UserMessage,
        };
        use crate::utils::transcript::normalize_context;
        use parking_lot::Mutex;

        let payload = base64::engine::general_purpose::STANDARD
            .encode(r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acc_test"}}"#);
        let token = format!("aaa.{payload}.bbb");
        let sse = format!(
            "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: {}\n\ndata: {}\n\n",
            serde_json::json!({
                "type": "response.output_item.added",
                "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] }
            }),
            serde_json::json!({ "type": "response.content_part.added", "part": { "type": "output_text", "text": "" } }),
            serde_json::json!({ "type": "response.output_text.delta", "delta": "Hello" }),
            serde_json::json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                    "content": [{ "type": "output_text", "text": "Hello" }]
                }
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "end_turn": true,
                    "incomplete_details": null,
                    "usage": { "input_tokens": 5, "output_tokens": 3, "total_tokens": 8, "input_tokens_details": { "cached_tokens": 0 } }
                }
            })
        );

        let recorded: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let recorder = recorded.clone();
        let sse_body = Arc::new(sse.into_bytes());
        let fetch = FetchFn::new(move |request: HttpRequest| {
            let request_body = match request.body {
                HttpBody::Bytes(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                _ => String::new(),
            };
            let entry = serde_json::json!({
                "url": request.url,
                "headers": request.headers,
                "body": serde_json::from_str::<Value>(&request_body).ok(),
            });
            recorder.lock().push(entry);
            let response = HttpResponse {
                status: 200,
                headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
                body: HttpBody::Bytes((*sse_body).clone()),
            };
            Box::pin(async move { Ok(response) })
        });

        let model: Model = serde_json::from_value(serde_json::json!({
            "id": "gpt-5.2-codex",
            "name": "GPT-5.2 Codex",
            "api": "openai-codex-responses",
            "provider": "openai-codex",
            "baseUrl": "https://chatgpt.com/backend-api",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 1000,
            "maxTokens": 100,
        }))
        .expect("model parses");
        let context = normalize_context(&Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage {
                content: UserContentField::Text("hi".to_string()),
                timestamp: 0,
            })],
            tools: None,
        });
        let mut options = StreamOptions::default();
        options.base.api_key = Some(token);
        options.base.fetch = Some(fetch);

        let message = stream_codex_responses(&model, &context, &options)
            .result()
            .await;
        assert_eq!(message.stop_reason, StopReason::Stop);
        assert_eq!(message.end_turn, Some(true));
        let text: Vec<&str> = message
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text.join(""), "Hello");
        assert_eq!(message.usage.input, 5);
        assert_eq!(message.usage.output, 3);

        let calls = recorded.lock();
        assert_eq!(calls.len(), 1);
        let call = &calls[0];
        assert_eq!(
            call["url"],
            "https://chatgpt.com/backend-api/codex/responses"
        );
        let headers = call["headers"].as_array().expect("header pairs");
        let get = |name: &str| {
            headers
                .iter()
                .find(|pair| {
                    pair[0]
                        .as_str()
                        .unwrap_or_default()
                        .eq_ignore_ascii_case(name)
                })
                .and_then(|pair| pair[1].as_str())
                .unwrap_or_default()
                .to_string()
        };
        assert_eq!(get("chatgpt-account-id"), "acc_test");
        assert_eq!(get("OpenAI-Beta"), "responses=experimental");
        assert_eq!(get("accept"), "text/event-stream");
        let body = &call["body"];
        assert_eq!(body["model"], "gpt-5.2-codex");
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
        assert_eq!(body["instructions"], "You are a helpful assistant.");
        assert_eq!(body["text"]["verbosity"], "low");
        assert_eq!(body["include"][0], "reasoning.encrypted_content");
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], true);
    }

    #[tokio::test]
    async fn reports_missing_api_key_as_stream_error() {
        use crate::types::{Context, Message, StreamErrorReason, UserContentField, UserMessage};
        use crate::utils::event_stream::EventStream;
        use crate::utils::transcript::normalize_context;
        use futures_util::StreamExt;

        let model: Model = serde_json::from_value(serde_json::json!({
            "id": "gpt-5.2-codex",
            "name": "GPT-5.2 Codex",
            "api": "openai-codex-responses",
            "provider": "openai-codex",
            "baseUrl": "https://chatgpt.com/backend-api",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 1000,
            "maxTokens": 100,
        }))
        .expect("model parses");
        let context = normalize_context(&Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage {
                content: UserContentField::Text("hi".to_string()),
                timestamp: 0,
            })],
            tools: None,
        });
        let options = StreamOptions::default();
        let stream = stream_codex_responses(&model, &context, &options);
        let mut events = stream.clone();
        let mut saw_error = None;
        while let Some(event) = events.next().await {
            if let AssistantMessageEvent::Error { reason, error } = event {
                saw_error = Some((reason, error));
            }
        }
        let (reason, error) = saw_error.expect("error event");
        assert_eq!(reason, StreamErrorReason::Error);
        assert_eq!(
            error.error_message.as_deref(),
            Some("No API key for provider: openai-codex")
        );
        assert_eq!(error.stop_reason, StopReason::Error);
    }
}
