//! Port of `packages/ai/src/api/openai-responses.ts` — the OpenAI Responses
//! stream adapter: request assembly, retry, and the SSE event loop over the
//! [`crate::http::FetchFn`] seam.
//!
//! Deviations:
//! - **D17**: upstream constructs an OpenAI SDK client (`new OpenAI({...})`).
//!   Rust hand-rolls `POST ${baseUrl}/responses` with the same headers
//!   (Authorization bearer, Content-Type/Accept JSON, pi user agent, model
//!   headers, session-affinity headers, caller headers last) and a hand-rolled
//!   `event:`/`data:` SSE decoder over the response body stream.
//! - **D18**: `github-copilot-headers.ts` support (`buildCopilotDynamicHeaders`,
//!   `hasCopilotVisionInput`) is not ported here; the header block is omitted
//!   until that slice lands.
//! - **D19**: synchronous setup failures (`streamSimple` key check, missing API
//!   key) cannot throw in Rust; they surface as the stream `error` event per
//!   the crate's no-throw-across-stream rule.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::api::constrained_sampling::create_grammar_tool_input_properties;
use crate::api::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::api::openai_responses_shared::{
	convert_responses_messages, convert_responses_tools, process_responses_stream, OpenAIResponsesStreamOptions,
};
use crate::api::simple_options::{build_base_options, resolve_sampling_params};
use crate::http::{FetchFn, HttpBody, HttpRequest, HttpResponse, ProviderHttpError};
use crate::models::{clamp_thinking_level, ProviderStreams};
use crate::types::{
	AssistantMessage, AssistantMessageEvent, CacheRetention, Model, ModelThinkingLevel, ProviderResponse,
	SessionAffinityFormat, SimpleStreamOptions, StopReason, StreamDoneReason, StreamErrorReason, StreamOptions,
	ThinkingLevel, TranscriptContext, Usage,
};
use crate::utils::abort::{race_with_abort_signal, AbortReason};
use crate::utils::diagnostics::now_ms;
use crate::utils::error_body::{
	format_provider_error, truncate_error_text, NormalizedProviderError, MAX_PROVIDER_ERROR_BODY_CHARS,
};
use crate::utils::event_stream::{done_event, error_event, AssistantMessageEventStream};
use crate::utils::headers::headers_to_record;
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::provider_retry::{retry_provider_request, ProviderRetryOptions};
use crate::utils::transcript::{get_declared_tools, resolve_transcript, resolve_transcript_tools};

const OPENAI_TOOL_CALL_PROVIDERS: [&str; 3] = ["openai", "openai-codex", "opencode"];
// OpenAI Responses rejects max_output_tokens below 16 (issue #6265).
const OPENAI_RESPONSES_MIN_OUTPUT_TOKENS: u64 = 16;
const CHATGPT_USAGE_URL: &str = "https://chatgpt.com/settings/usage";

/// The exported API implementation (`openai_responses_api`).
pub fn openai_responses_api() -> Arc<dyn ProviderStreams> {
	Arc::new(OpenAIResponsesApi)
}

struct OpenAIResponsesApi;

impl ProviderStreams for OpenAIResponsesApi {
	fn stream(&self, model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
		stream_openai_responses(model, context, options)
	}

	fn stream_simple(
		&self,
		model: &Model,
		context: &TranscriptContext,
		options: &SimpleStreamOptions,
	) -> AssistantMessageEventStream {
		stream_simple_openai_responses(model, context, options)
	}
}

/// `Required<OpenAIResponsesCompat>`: compat flags with their defaults applied.
#[derive(Debug, Clone)]
struct ResponsesCompat {
	supports_developer_role: bool,
	supports_mid_convo_system_messages: bool,
	session_affinity_format: SessionAffinityFormat,
	supports_long_cache_retention: bool,
	supports_strict_mode: bool,
	supports_openai_grammar_tools: bool,
	supports_additional_tools: bool,
	supports_tool_search: bool,
	supports_explicit_prompt_cache_mode: bool,
	supports_max_output_tokens: bool,
}

fn detect_session_affinity_format(model: &Model) -> SessionAffinityFormat {
	if model.provider() == "openrouter" || model.base.base_url.contains("openrouter.ai") {
		SessionAffinityFormat::Openrouter
	} else {
		SessionAffinityFormat::Openai
	}
}

fn get_compat(model: &Model) -> ResponsesCompat {
	let compat = model.compat.as_ref();
	ResponsesCompat {
		supports_developer_role: compat.and_then(|c| c.supports_developer_role).unwrap_or(true),
		supports_mid_convo_system_messages: compat.and_then(|c| c.supports_mid_convo_system_messages).unwrap_or(false),
		session_affinity_format: compat
			.and_then(|c| c.session_affinity_format)
			.unwrap_or_else(|| detect_session_affinity_format(model)),
		supports_long_cache_retention: compat.and_then(|c| c.supports_long_cache_retention).unwrap_or(true),
		supports_strict_mode: compat.and_then(|c| c.supports_strict_mode).unwrap_or(false),
		supports_openai_grammar_tools: compat.and_then(|c| c.supports_openai_grammar_tools).unwrap_or(false),
		supports_additional_tools: compat.and_then(|c| c.supports_additional_tools).unwrap_or(false),
		supports_tool_search: compat.and_then(|c| c.supports_tool_search).unwrap_or(false),
		supports_explicit_prompt_cache_mode: compat
			.and_then(|c| c.supports_explicit_prompt_cache_mode)
			.unwrap_or(false),
		supports_max_output_tokens: compat.and_then(|c| c.supports_max_output_tokens).unwrap_or(true),
	}
}

/// OpenAI API keys start with `sk-`; a different credential sent directly to
/// OpenAI is a Sign in with ChatGPT access token.
fn is_chatgpt_sign_in(model: &Model, api_key: Option<&str>) -> bool {
	model.provider() == "openai"
		&& model.base.base_url == "https://api.openai.com/v1"
		&& api_key.is_some_and(|key| !key.starts_with("sk-"))
}

fn has_header(headers: Option<&BTreeMap<String, Option<String>>>, name: &str) -> bool {
	let Some(headers) = headers else { return false };
	let expected = name.to_ascii_lowercase();
	headers.iter().any(|(key, value)| {
		key.eq_ignore_ascii_case(&expected)
			&& value.as_ref().map(|v| !v.trim().is_empty()).unwrap_or(false)
	})
}

/// Resolve the API key, or the `"unused"` placeholder when headers carry auth
/// (`getClientApiKey`). `Err` carries the upstream error message.
fn get_client_api_key(
	provider: &str,
	api_key: Option<&str>,
	headers: Option<&BTreeMap<String, Option<String>>>,
) -> Result<String, String> {
	if let Some(api_key) = api_key {
		return Ok(api_key.to_string());
	}
	if has_header(headers, "authorization") || has_header(headers, "cf-aig-authorization") {
		return Ok("unused".to_string());
	}
	Err(format!("No API key for provider: {provider}"))
}

/// Resolve cache retention preference. Defaults to `Short` and uses
/// `PI_CACHE_RETENTION` for backward compatibility.
fn resolve_cache_retention(cache_retention: Option<CacheRetention>, env: Option<&BTreeMap<String, String>>) -> CacheRetention {
	if let Some(cache_retention) = cache_retention {
		return cache_retention;
	}
	if get_provider_env_value("PI_CACHE_RETENTION", env).as_deref() == Some("long") {
		CacheRetention::Long
	} else {
		CacheRetention::Short
	}
}

fn get_prompt_cache_retention(compat: &ResponsesCompat, cache_retention: CacheRetention) -> Option<&'static str> {
	if cache_retention == CacheRetention::Long
		&& compat.supports_long_cache_retention
		&& !compat.supports_explicit_prompt_cache_mode
	{
		Some("24h")
	} else {
		None
	}
}

fn get_prompt_cache_options(compat: &ResponsesCompat, cache_retention: CacheRetention) -> Option<Value> {
	if !compat.supports_explicit_prompt_cache_mode {
		return None;
	}
	if cache_retention == CacheRetention::None {
		return Some(json!({ "mode": "explicit" }));
	}
	if cache_retention == CacheRetention::Long && compat.supports_long_cache_retention {
		return Some(json!({ "ttl": "30m" }));
	}
	None
}

fn openai_tool_call_providers() -> BTreeSet<String> {
	OPENAI_TOOL_CALL_PROVIDERS.iter().map(|s| s.to_string()).collect()
}

// =============================================================================
// Request assembly
// =============================================================================

/// Default request headers plus session-affinity headers (`createClient`).
fn build_request_headers(
	model: &Model,
	compat: &ResponsesCompat,
	session_id: Option<&str>,
	options_headers: Option<&BTreeMap<String, Option<String>>>,
) -> Vec<(String, String)> {
	let mut headers: Vec<(String, String)> = Vec::new();
	headers.push(("User-Agent".to_string(), get_pi_user_agent()));
	if let Some(model_headers) = &model.base.headers {
		for (key, value) in model_headers {
			headers.push((key.clone(), value.clone()));
		}
	}
	// D18: github-copilot dynamic headers are not ported yet.
	if let Some(session_id) = session_id {
		match compat.session_affinity_format {
			SessionAffinityFormat::Openrouter => {
				headers.push(("x-session-id".to_string(), session_id.to_string()));
			}
			_ => {
				if compat.session_affinity_format == SessionAffinityFormat::Openai {
					headers.push(("session_id".to_string(), session_id.to_string()));
				}
				headers.push(("x-client-request-id".to_string(), session_id.to_string()));
			}
		}
	}
	// Merge options headers last so they can override defaults; `None` values
	// suppress a default header.
	if let Some(options_headers) = options_headers {
		for (key, value) in options_headers {
			match value {
				Some(value) => headers.push((key.clone(), value.clone())),
				// Case-insensitive: upstream SDK null-deletion removes the
				// default regardless of the casing the caller used.
				None => headers.retain(|(name, _)| !name.eq_ignore_ascii_case(key)),
			}
		}
	}
	headers
}

/// Build the Responses create payload (`buildParams`).
#[allow(clippy::too_many_arguments)]
fn build_params(
	model: &Model,
	messages: Vec<Value>,
	request_tools: &[crate::types::Tool],
	options: Option<&StreamOptions>,
	compat: &ResponsesCompat,
	_grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<Value, String> {
	let opts = options;
	let cache_retention = resolve_cache_retention(opts.and_then(|o| o.cache_retention), opts.and_then(|o| o.base.env.as_ref()));
	// Sign in with ChatGPT rejects these request fields.
	let omit_unsupported_fields = is_chatgpt_sign_in(model, opts.and_then(|o| o.base.api_key.as_deref()));

	let mut params = Map::new();
	params.insert("model".to_string(), json!(model.id()));
	params.insert("input".to_string(), Value::Array(messages));
	params.insert("stream".to_string(), json!(true));
	if cache_retention != CacheRetention::None {
		if let Some(key) = clamp_openai_prompt_cache_key(opts.and_then(|o| o.session_id.as_deref())) {
			params.insert("prompt_cache_key".to_string(), json!(key));
		}
	}
	if !omit_unsupported_fields {
		if let Some(retention) = get_prompt_cache_retention(compat, cache_retention) {
			params.insert("prompt_cache_retention".to_string(), json!(retention));
		}
		if let Some(cache_options) = get_prompt_cache_options(compat, cache_retention) {
			params.insert("prompt_cache_options".to_string(), cache_options);
		}
	}
	params.insert("store".to_string(), json!(false));

	if let (Some(max_tokens), true, false) = (
		opts.and_then(|o| o.max_tokens),
		compat.supports_max_output_tokens,
		omit_unsupported_fields,
	) {
		params.insert(
			"max_output_tokens".to_string(),
			json!(std::cmp::max(max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)),
		);
	}

	if let Some(temperature) = opts.and_then(|o| o.temperature).filter(|_| !omit_unsupported_fields) {
		params.insert("temperature".to_string(), json!(temperature));
	}

	if let Some(service_tier) = opts.and_then(|o| o.openai_responses.service_tier.clone()) {
		params.insert("service_tier".to_string(), json!(service_tier));
	}

	if !request_tools.is_empty() {
		let tool_options = crate::api::openai_responses_shared::ConvertResponsesToolsOptions {
			strict: None,
			supports_strict_mode: Some(compat.supports_strict_mode),
			supports_openai_grammar_tools: Some(compat.supports_openai_grammar_tools),
			tool_search_result: false,
		};
		params.insert("tools".to_string(), Value::Array(convert_responses_tools(request_tools, Some(&tool_options))?));
	}

	if let Some(tool_choice) = opts.and_then(|o| o.openai_responses.tool_choice.clone()) {
		params.insert("tool_choice".to_string(), tool_choice);
	}

	let reasoning_summary = opts.and_then(|o| o.openai_responses.reasoning_summary);
	let reasoning_effort = opts
		.and_then(|o| o.openai_responses.reasoning_effort)
		.or_else(|| reasoning_summary.filter(|s| s.is_some()).map(|_| ThinkingLevel::Medium));

	if model.reasoning {
		if let Some(effort_level) = reasoning_effort {
			let map_value = model
				.thinking_level_map
				.as_ref()
				.and_then(|map| map.get(&ModelThinkingLevel::from(effort_level)))
				.cloned()
				.flatten();
			// `model.thinkingLevelMap?.[effort] ?? effort`
			let effort = map_value
					.map(Value::String)
					.unwrap_or_else(|| serde_json::to_value(effort_level).expect("effort serializes"));
			let summary = match reasoning_summary {
				Some(Some(summary)) => json!(summary),
				_ => json!("auto"),
			};
			params.insert("reasoning".to_string(), json!({ "effort": effort, "summary": summary }));
			params.insert("include".to_string(), json!(["reasoning.encrypted_content"]));
		} else if model.provider() != "github-copilot"
			&& model
				.thinking_level_map
				.as_ref()
				.and_then(|map| map.get(&ModelThinkingLevel::Off))
				.cloned()
				!= Some(None)
		{
			// `model.thinkingLevelMap?.off ?? "none"`
			let effort = model
				.thinking_level_map
				.as_ref()
				.and_then(|map| map.get(&ModelThinkingLevel::Off))
				.cloned()
				.flatten()
				.unwrap_or_else(|| "none".to_string());
			params.insert("reasoning".to_string(), json!({ "effort": effort }));
		}
		if model.provider() == "xai" {
			params.insert("include".to_string(), json!(["reasoning.encrypted_content"]));
		}
	}

	// Last so model and request sampling parameters override named request fields.
	let effort_for_sampling = reasoning_effort
		.map(ModelThinkingLevel::from)
		.unwrap_or(ModelThinkingLevel::Off);
	if let Some(sampling_params) = resolve_sampling_params(model, effort_for_sampling, opts.and_then(|o| o.sampling_params.as_ref())) {
		if let Value::Object(merged) = Value::Object(sampling_params) {
			for (key, value) in merged {
				params.insert(key, value);
			}
		}
	}

	Ok(Value::Object(params))
}

fn get_service_tier_cost_multiplier(model_id: &str, service_tier: Option<&Value>) -> f64 {
	match service_tier.and_then(Value::as_str) {
		Some("flex") => 0.5,
		Some("priority") | Some("fast") => {
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
	usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

// =============================================================================
// SSE decoding (hand-rolled, upstream uses the SDK's decoder)
// =============================================================================

/// Incremental `text/event-stream` decoder producing the JSON payload of each
/// event. `data:` lines are joined with `\n`; blank lines dispatch; unknown
/// fields and comments are ignored.
#[derive(Default)]
struct SseDecoder {
	buffer: String,
	data_lines: Vec<String>,
}

impl SseDecoder {
	fn feed(&mut self, chunk: &str) -> Vec<Value> {
		self.buffer.push_str(chunk);
		let mut events = Vec::new();
		while let Some(break_at) = self.buffer.find('\n') {
			let line: String = self.buffer.drain(..break_at + 1).collect();
			let line = line.trim_end_matches(['\n', '\r']);
			if let Some(event) = self.handle_line(line) {
				events.push(event);
			}
		}
		events
	}

	/// Dispatch any trailing event at end of stream.
	fn finish(&mut self) -> Vec<Value> {
		let line = self.buffer.clone();
		self.buffer.clear();
		let mut events = self.handle_line(&line).into_iter().collect::<Vec<_>>();
		events.extend(self.flush().into_iter());
		events
	}

	fn handle_line(&mut self, line: &str) -> Option<Value> {
		if line.is_empty() {
			return self.flush();
		}
		if let Some(data) = line.strip_prefix("data:") {
			self.data_lines.push(data.strip_prefix(' ').unwrap_or(data).to_string());
		}
		// `event:`, `id:`, `retry:`, comments (`:`) — the Responses API encodes
		// the event type inside the data payload, so the SSE event name is
		// redundant and ignored, as in the SDK decoder.
		None
	}

	fn flush(&mut self) -> Option<Value> {
		if self.data_lines.is_empty() {
			return None;
		}
		let data = self.data_lines.join("\n");
		self.data_lines.clear();
		serde_json::from_str(&data).ok()
	}
}

/// Decode an SSE body into a stream of JSON event payloads.
fn decode_sse(body: HttpBody) -> futures_core::stream::BoxStream<'static, Result<Value, ProviderHttpError>> {
	use futures_util::StreamExt;
	let chunks: futures_core::stream::BoxStream<'static, Result<Vec<u8>, ProviderHttpError>> = match body {
		HttpBody::Empty => Box::pin(futures_util::stream::empty()),
		HttpBody::Bytes(bytes) => Box::pin(futures_util::stream::iter(vec![Ok(bytes)])),
		HttpBody::Stream(stream) => stream,
	};
	Box::pin(futures_util::stream::unfold(
		(chunks, SseDecoder::default(), std::collections::VecDeque::new(), false),
		|(mut chunks, mut decoder, mut pending, mut finished)| async move {
			loop {
				if let Some(event) = pending.pop_front() {
					return Some((Ok(event), (chunks, decoder, pending, finished)));
				}
				if finished {
					return None;
				}
				match chunks.next().await {
					Some(Ok(bytes)) => {
						let text = String::from_utf8_lossy(&bytes).into_owned();
						pending.extend(decoder.feed(&text));
					}
					Some(Err(error)) => {
						finished = true;
						return Some((Err(error), (chunks, decoder, pending, finished)));
					}
					None => {
						pending.extend(decoder.finish());
						finished = true;
					}
				}
			}
		},
	))
}

// =============================================================================
// Stream functions
// =============================================================================

fn stream_openai_responses(
	model: &Model,
	context: &TranscriptContext,
	options: &StreamOptions,
) -> AssistantMessageEventStream {
	let stream = AssistantMessageEventStream::new();
	let compat = get_compat(model);
	let normalized_context = resolve_transcript(context, Some(compat.supports_mid_convo_system_messages));

	let model = model.clone();
	let options = options.clone();
	let handle = stream.clone();
	tokio::spawn(async move {
		let mut output = AssistantMessage {
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
		};

		let result = run_openai_responses(&model, &normalized_context, &options, &handle, &mut output).await;
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
				let aborted = options.base.signal.as_ref().map(|s| s.aborted()).unwrap_or(false);
				output.stop_reason = if aborted { StopReason::Aborted } else { StopReason::Error };
				// Sign in with ChatGPT shares the subscription's usage limit with
				// other apps.
				output.error_message = Some(if error_message.contains("subscription_sharing_usage_limit_exceeded") {
					format!("{error_message}\nCheck your ChatGPT usage: {CHATGPT_USAGE_URL}")
				} else {
					error_message
				});
				handle.push(error_event(output, if aborted { StreamErrorReason::Aborted } else { StreamErrorReason::Error }));
			}
		}
	});
	stream
}

/// Request-level error mirroring the OpenAI SDK error objects: transport
/// failures have no `status`; completed non-2xx responses carry the status,
/// response headers, and raw body. Implements the same protocol
/// [`crate::utils::provider_retry::ProviderRequestError`] uses for retries.
enum ResponsesRequestError {
	Http { message: String, status: Option<u16>, headers: Vec<(String, String)>, body: Option<String> },
	Abort,
	Message(String),
}

impl crate::utils::provider_retry::ProviderRequestError for ResponsesRequestError {
	fn status(&self) -> Option<u16> {
		match self {
			ResponsesRequestError::Http { status, .. } => *status,
			_ => None,
		}
	}

	fn header(&self, name: &str) -> Option<String> {
		match self {
			ResponsesRequestError::Http { headers, .. } => headers
				.iter()
				.find(|(key, _)| key.eq_ignore_ascii_case(name))
				.map(|(_, value)| value.clone()),
			_ => None,
		}
	}

	fn message(&self) -> &str {
		match self {
			ResponsesRequestError::Http { message, .. } => message,
			ResponsesRequestError::Abort => "Request was aborted.",
			ResponsesRequestError::Message(message) => message,
		}
	}

	fn from_abort() -> Self {
		ResponsesRequestError::Abort
	}
}

/// `normalizeProviderError` + `formatProviderError` upstream.
fn format_responses_error(error: &ResponsesRequestError, prefix: &str) -> String {
	let (message, status, raw_body) = match error {
		ResponsesRequestError::Http { message, status, body, .. } => (message.clone(), *status, body.clone()),
		ResponsesRequestError::Abort => ("Request was aborted.".to_string(), None, None),
		ResponsesRequestError::Message(message) => (message.clone(), None, None),
	};
	let body = raw_body
		.as_deref()
		.map(str::trim)
		.filter(|trimmed| !trimmed.is_empty())
		.map(|trimmed| truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS));
	let message_carries_body = match &body {
		None => true,
		Some(body) => message.contains(body),
	};
	let normalized = NormalizedProviderError { status, body, message, message_carries_body };
	format_provider_error(&normalized, Some(prefix))
}

/// The request/response/stream pipeline between stream construction and the
/// terminal event. Returns `Err` with the error message on failure.
async fn run_openai_responses(
	model: &Model,
	normalized_context: &TranscriptContext,
	options: &StreamOptions,
	stream: &AssistantMessageEventStream,
	output: &mut AssistantMessage,
) -> Result<(), String> {
	let compat = get_compat(model);
	let error_prefix =
		format!("{} API error", if model.provider() == "openai" { "OpenAI" } else { model.provider() });
	let api_key = get_client_api_key(
		model.provider(),
		options.base.api_key.as_deref(),
		options.base.headers.as_ref(),
	)?;
	let cache_retention = resolve_cache_retention(options.cache_retention, options.base.env.as_ref());
	let cache_session_id = if cache_retention == CacheRetention::None { None } else { options.session_id.as_deref() };

	let declared_tools = get_declared_tools(normalized_context.messages());
	let grammar_tool_input_properties = create_grammar_tool_input_properties(&declared_tools, compat.supports_openai_grammar_tools)?;

	let messages = convert_responses_messages(
		model,
		normalized_context,
		&openai_tool_call_providers(),
		Some(&crate::api::openai_responses_shared::ConvertResponsesMessagesOptions {
			include_system_prompt: None,
			grammar_tool_input_properties: Some(grammar_tool_input_properties.clone()),
			supports_mid_convo_system_messages: compat.supports_mid_convo_system_messages,
			supports_additional_tools: compat.supports_additional_tools,
			supports_tool_search: compat.supports_tool_search,
			tool_options: Some(crate::api::openai_responses_shared::ConvertResponsesToolsOptions {
				strict: None,
				supports_strict_mode: Some(compat.supports_strict_mode),
				supports_openai_grammar_tools: Some(compat.supports_openai_grammar_tools),
				tool_search_result: false,
			}),
		}),
	)?;

	let transcript_tools = resolve_transcript_tools(
		normalized_context.messages(),
		compat.supports_additional_tools || compat.supports_tool_search,
	);

	let mut params = build_params(
		model,
		messages,
		&transcript_tools.request_tools,
		Some(options),
		&compat,
		&grammar_tool_input_properties,
	)?;
	if let Some(on_payload) = &options.base.on_payload {
		if let Some(next) = on_payload(params.clone(), model.clone()).await {
			params = next;
		}
	}

	// Build the HTTP request (D17): the SDK's `client.responses.create`.
	let url = format!("{}/responses", model.base.base_url.trim_end_matches('/'));
	let mut request = HttpRequest::new("POST", &url);
	request.headers.push(("Authorization".to_string(), format!("Bearer {api_key}")));
	request.headers.push(("Content-Type".to_string(), "application/json".to_string()));
	request.headers.push(("Accept".to_string(), "application/json".to_string()));
	for (key, value) in build_request_headers(model, &compat, cache_session_id, options.base.headers.as_ref()) {
		if key.eq_ignore_ascii_case("authorization") || key.eq_ignore_ascii_case("content-type") {
			// Caller values override the SDK defaults above.
			request.headers.retain(|(name, _)| !name.eq_ignore_ascii_case(&key));
		}
		request.headers.push((key, value));
	}
	// Plain request data captured by the retry factory (a `HttpRequest` can
	// carry a non-`Sync` stream body, so it cannot be captured by reference).
	let request_template = (
		request.method.clone(),
		request.url.clone(),
		request.headers.clone(),
		serde_json::to_vec(&params).expect("request body serializes"),
		options.base.timeout_ms,
		options.base.signal.clone(),
	);

	let fetch: FetchFn = options.base.fetch.clone().unwrap_or_default();
	let max_retries = options.base.max_retries;
	let max_retry_delay_ms = options.base.max_retry_delay_ms;
	let signal_for_retry = options.base.signal.clone();

	let response: HttpResponse = match race_with_abort_signal(
		retry_provider_request(
			|| {
				let fetch = fetch.clone();
				let request = HttpRequest {
					method: request_template.0.clone(),
					url: request_template.1.clone(),
					headers: request_template.2.clone(),
					body: HttpBody::Bytes(request_template.3.clone()),
					timeout_ms: request_template.4,
					signal: request_template.5.clone(),
				};
				async move {
					match fetch.send(request).await {
						Ok(response) if !response.ok() => {
							
							let body = response.body.collect().await.unwrap_or_default();
							let body = String::from_utf8_lossy(&body).to_string();
							Err(ResponsesRequestError::Http {
								message: response.status.to_string(),
								status: Some(response.status),
								headers: response.headers.clone(),
								body: Some(body),
							})
						}
						Ok(response) => Ok(response),
						Err(error) => Err(ResponsesRequestError::Http {
							message: error.message,
							status: error.status,
							headers: error.headers,
							body: error.body,
						}),
					}
				}
			},
			ProviderRetryOptions { max_retries, max_retry_delay_ms, signal: signal_for_retry },
		),
		options.base.signal.clone().unwrap_or_default(),
	)
	.await
	{
		Ok(result) => result.map_err(|error| format_responses_error(&error, &error_prefix))?,
		Err(reason) => {
			let error = match reason {
				AbortReason::Aborted => ResponsesRequestError::Abort,
				AbortReason::Message(message) => ResponsesRequestError::Message(message),
				AbortReason::Error(error) => ResponsesRequestError::Message(error.to_string()),
			};
			return Err(format_responses_error(&error, &error_prefix));
		}
	};

	if let Some(on_response) = &options.base.on_response {
		on_response(
			ProviderResponse { status: response.status, headers: headers_to_record(&response.headers) },
			model.clone(),
		)
		.await;
	}

	let mut event_stream = Box::pin(decode_sse(response.body));

	stream.push(AssistantMessageEvent::Start { partial: output.clone() });

	let model_id = model.id().to_string();
	let stream_options = OpenAIResponsesStreamOptions {
		on_provider_stream_event: options.on_provider_stream_event.clone(),
		service_tier: options.openai_responses.service_tier.as_deref().map(|tier| json!(tier)),
		grammar_tool_input_properties,
		resolve_service_tier: None,
		apply_service_tier_pricing: Some(Arc::new(move |usage: &mut Usage, service_tier: Option<&Value>| {
			apply_service_tier_pricing(usage, service_tier, &model_id);
		})),
	};

	process_responses_stream(&mut event_stream, output, stream, model, Some(&stream_options)).await?;

	if options.base.signal.as_ref().map(|s| s.aborted()).unwrap_or(false) {
		return Err("Request was aborted".to_string());
	}
	if output.stop_reason == StopReason::Pending {
		return Err("OpenAI Responses stream ended without a stop reason".to_string());
	}
	if output.stop_reason == StopReason::Aborted || output.stop_reason == StopReason::Error {
		return Err(output.error_message.clone().unwrap_or_else(|| "An unknown error occurred".to_string()));
	}
	Ok(())
}

fn abort_reason_message(reason: AbortReason) -> String {
	match reason {
		AbortReason::Aborted => "Request was aborted.".to_string(),
		AbortReason::Message(message) => message,
		AbortReason::Error(error) => error.to_string(),
	}
}

fn stream_simple_openai_responses(
	model: &Model,
	context: &TranscriptContext,
	options: &SimpleStreamOptions,
) -> AssistantMessageEventStream {
	// Upstream throws synchronously before a stream exists; Rust surfaces the
	// failure as the stream error event (D19).
	if let Err(message) = get_client_api_key(
		model.provider(),
		options.base.base.api_key.as_deref(),
		options.base.base.headers.as_ref(),
	) {
		let stream = AssistantMessageEventStream::new();
		stream.push(error_event(
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
				stop_reason: StopReason::Error,
				deferred: None,
				error_message: Some(message),
				raw_stop_reason: None,
				end_turn: None,
				timestamp: now_ms(),
			},
			StreamErrorReason::Error,
		));
		return stream;
	}

	let mut base = build_base_options(model, context, options, options.base.base.api_key.as_deref());
	base.openai_responses.tool_choice = options.tool_choice.map(|choice| match choice {
		crate::types::ToolChoice::Auto => json!("auto"),
		crate::types::ToolChoice::None => json!("none"),
	});
	let clamped_reasoning =
		options.reasoning.map(|reasoning| clamp_thinking_level(model, ModelThinkingLevel::from(reasoning)));
	// ModelThinkingLevel -> ThinkingLevel mapping without Off.
	let reasoning_effort = clamped_reasoning.and_then(|level| match level {
		ModelThinkingLevel::Off => None,
		ModelThinkingLevel::Minimal => Some(ThinkingLevel::Minimal),
		ModelThinkingLevel::Low => Some(ThinkingLevel::Low),
		ModelThinkingLevel::Medium => Some(ThinkingLevel::Medium),
		ModelThinkingLevel::High => Some(ThinkingLevel::High),
		ModelThinkingLevel::Xhigh => Some(ThinkingLevel::Xhigh),
		ModelThinkingLevel::Max => Some(ThinkingLevel::Max),
	});
	base.openai_responses.reasoning_effort = reasoning_effort;

	stream_openai_responses(model, context, &base)
}
