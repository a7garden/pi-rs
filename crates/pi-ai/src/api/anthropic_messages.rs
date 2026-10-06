//! Port of `packages/ai/src/api/anthropic-messages.ts`.
//!
//! Upstream drives the `@anthropic-ai/sdk` client (`client.beta.messages.create(...).asResponse()`)
//! and iterates the raw SSE body itself; the Rust port hand-rolls the HTTP request over the
//! [`crate::http::FetchFn`] seam (`POST {baseUrl}/v1/messages`, `anthropic-version: 2023-06-01`,
//! `x-api-key`/Bearer auth, [`crate::utils::pi_user_agent::get_pi_user_agent`] user agent) and
//! reuses the upstream SSE decoder verbatim (deviation D16). All wire shapes
//! (`BetaRawMessageStreamEvent`, `MessageCreateParamsStreaming`, content blocks, tools) are
//! mirrored as serde types in this module.
//!
//! # Deviations (crate ledger continues from D14)
//!
//! - **D15** — `AnthropicOptions.client` (pre-built SDK client injection, e.g. `AnthropicVertex`)
//!   is omitted; there is no SDK to inject. Tests inject a `FetchFn` fake instead.
//! - **D16** — the SDK HTTP layer is replaced by a hand-rolled request. Non-2xx responses become
//!   [`AnthropicRequestError`] with the status/headers the SDK error object carried upstream;
//!   the error message formatting is minimal (`status` + raw body) rather than the SDK's
//!   normalized APIError string.
//! - **D17** — Anthropic workload-identity federation (the `ANTHROPIC_*` env-api-key variables
//!   and the SDK's OIDC token exchange in `getAnthropicFederation`) is omitted; it is entirely
//!   an SDK runtime capability. `assertRequestAuth` therefore always applies for the anthropic
//!   provider.
//! - **D18** — `api/github-copilot-headers.ts` has no assigned slice; its three helpers are
//!   ported privately in this module (`infer_copilot_initiator`, `has_copilot_vision_input`,
//!   `build_copilot_dynamic_headers`).
//! - **D19** — SSE chunk decoding uses per-chunk `String::from_utf8_lossy` where upstream
//!   streams through a `TextDecoder` (split multibyte sequences across chunk boundaries would
//!   produce replacement characters instead of being joined; SSE payloads are ASCII in practice).
//! - **D20** — `retryProviderRequest` is consumed as
//!   `crate::utils::provider_retry::retry_provider_request(request, ProviderRetryOptions)` with
//!   `ProviderRetryOptions { max_retries: Option<u32>, max_retry_delay_ms: Option<u64>, signal:
//!   Option<AbortSignal> }`, generic over an error implementing the module's
//!   `ProviderRequestError` trait (`status() -> Option<u16>`, `header(name) -> Option<String>`),
//!   preserving the upstream SDK retry policy (x-should-retry, 408/409/429/5xx, Retry-After,
//!   abort, 60s default delay cap).
//! - **D21** — upstream `parseJsonWithRepair<RawMessageStreamEvent>` is a TS type assertion and
//!   never rejects unknown event shapes; the Rust event enums therefore accept unknown
//!   `content_block` / `delta` variants (`#[serde(other)]`) and default all optional fields so
//!   only malformed JSON reaches the parse-error path.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::http::{BodyStream, FetchFn, HttpBody, HttpRequest, HttpResponse};
use crate::models::{calculate_cost, ProviderStreams};
use crate::types::{
	AnthropicApiOptions, AnthropicBuiltInToolChoice, AnthropicEffort, AnthropicThinkingDisplay, AnthropicToolChoice,
	AnyModel, AssistantContent, AssistantMessage, AssistantMessageDiagnostic, AssistantMessageEvent,
	AssistantMessageEventStream, BaseModelFields, CacheRetention, Context, Message, Model, ModelCompat, ModelCost,
	ModelCostRates, ModelThinkingLevel, ProviderHeaders, ProviderResponse, SimpleStreamOptions, StopReason,
	StreamDoneReason, StreamErrorReason, StreamOptions, ThinkingContent, ThinkingLevel, Tool, ToolCall,
	ToolChoice, ToolResultMessage, TranscriptContext, UserContent,
};
use crate::utils::abort::AbortSignal;
use crate::utils::diagnostics::{append_assistant_message_diagnostic, now_ms};
use crate::utils::event_stream::{done_event, error_event};
use crate::utils::headers::headers_to_record;
use crate::utils::json_parse::{parse_json_with_repair, parse_streaming_json};
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::text::{get_system_message_text, render_system_message_update};
use crate::utils::transcript::{get_current_tools, get_initial_system_message, resolve_transcript};

// ---------------------------------------------------------------------------
// Cache retention
// ---------------------------------------------------------------------------

/// Resolve cache retention preference (`resolveCacheRetention`). Defaults to
/// "short" and uses `PI_CACHE_RETENTION` for backward compatibility.
fn resolve_cache_retention(cache_retention: Option<CacheRetention>, env: Option<&crate::types::ProviderEnv>) -> CacheRetention {
	if let Some(retention) = cache_retention {
		return retention;
	}
	if crate::utils::provider_env::get_provider_env_value("PI_CACHE_RETENTION", env).as_deref() == Some("long") {
		return CacheRetention::Long;
	}
	CacheRetention::Short
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct CacheControl {
	#[serde(rename = "type")]
	tag: &'static str,
	#[serde(skip_serializing_if = "Option::is_none")]
	ttl: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq)]
struct CacheControlResult {
	retention: CacheRetention,
	cache_control: Option<CacheControl>,
}

fn get_cache_control(model: &Model, cache_retention: Option<CacheRetention>, env: Option<&crate::types::ProviderEnv>) -> CacheControlResult {
	let retention = resolve_cache_retention(cache_retention, env);
	if retention == CacheRetention::None {
		return CacheControlResult { retention, cache_control: None };
	}
	let supports_long = model
		.compat
		.as_ref()
		.and_then(|c| c.supports_long_cache_retention)
		.unwrap_or(true);
	let ttl = if retention == CacheRetention::Long && supports_long { Some("1h") } else { None };
	CacheControlResult {
		retention,
		cache_control: Some(CacheControl { tag: "ephemeral", ttl }),
	}
}

// ---------------------------------------------------------------------------
// Stealth mode: Mimic Claude Code's tool naming exactly
// ---------------------------------------------------------------------------

const CLAUDE_CODE_VERSION: &str = "2.1.280";

/// Claude Code 2.x tool names (canonical casing).
const CLAUDE_CODE_TOOLS: &[&str] = &[
	"Read",
	"Write",
	"Edit",
	"Bash",
	"Grep",
	"Glob",
	"AskUserQuestion",
	"EnterPlanMode",
	"ExitPlanMode",
	"KillShell",
	"NotebookEdit",
	"Skill",
	"Task",
	"TaskOutput",
	"TodoWrite",
	"WebFetch",
	"WebSearch",
];

/// Convert tool name to CC canonical casing if it matches (case-insensitive).
fn to_claude_code_name(name: &str) -> String {
	let lower = name.to_lowercase();
	for tool in CLAUDE_CODE_TOOLS {
		if tool.to_lowercase() == lower {
			return (*tool).to_string();
		}
	}
	name.to_string()
}

fn from_claude_code_name(name: &str, tools: &[Tool]) -> String {
	if !tools.is_empty() {
		let lower = name.to_lowercase();
		if let Some(matched) = tools.iter().find(|tool| tool.name.to_lowercase() == lower) {
			return matched.name.clone();
		}
	}
	name.to_string()
}

/// Convert tool-result content blocks to the Anthropic API format
/// (`convertContentBlocks`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
enum ToolResultContent {
	Text(String),
	Blocks(Vec<BlockParam>),
}

fn convert_content_blocks(content: &[UserContent]) -> ToolResultContent {
	let has_images = content.iter().any(|c| matches!(c, UserContent::Image(_)));
	if !has_images {
		let text = content
			.iter()
			.map(|c| match c {
				UserContent::Text(text) => text.text.clone(),
				UserContent::Image(_) => String::new(),
			})
			.collect::<Vec<_>>()
			.join("\n");
		return ToolResultContent::Text(sanitize_surrogates(&text));
	}

	let mut blocks: Vec<BlockParam> = content
		.iter()
		.map(|block| match block {
			UserContent::Text(text) => BlockParam::Text {
				text: sanitize_surrogates(&text.text),
				cache_control: None,
			},
			UserContent::Image(image) => BlockParam::Image {
				source: ImageSource {
					source_type: "base64",
					media_type: image.mime_type.clone(),
					data: image.data.clone(),
				},
				cache_control: None,
			},
		})
		.collect();

	// If only images (no text), add placeholder text block
	let has_text = blocks
		.iter()
		.any(|b| matches!(b, BlockParam::Text { .. }));
	if !has_text {
		blocks.insert(
			0,
			BlockParam::Text {
				text: "(see attached image)".to_string(),
				cache_control: None,
			},
		);
	}

	ToolResultContent::Blocks(blocks)
}

// ---------------------------------------------------------------------------
// Betas
// ---------------------------------------------------------------------------

const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const SERVER_SIDE_FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
const MID_CONVERSATION_OUTPUT_CONFIG_BETA: &str = "mid-conversation-output-config-2026-07-01";
const THINKING_BINDING_CONTROLS_BETA: &str = "thinking-binding-controls-2026-08-01";
const INLINE_TOOLS_BETA: &str = "inline-tools-2026-09-15";

/// Stable deferred tool declared whenever native tool changes are in use
/// (`DEFERRED_TOOL_PLACEHOLDER`).
fn deferred_tool_placeholder() -> BetaTool {
	BetaTool {
		name: "__pi_deferred_placeholder__".to_string(),
		description: Some("Reserved placeholder. Never available. Never call this.".to_string()),
		eager_input_streaming: None,
		strict: None,
		input_schema: InputSchema::legacy(json!({}).as_object().expect("empty object").clone(), Vec::new()),
		cache_control: None,
		defer_loading: Some(true),
	}
}

fn should_use_server_side_fallback_beta(model: &Model) -> bool {
	model
		.compat
		.as_ref()
		.and_then(|c| c.allowed_fallback_models.as_ref())
		.map_or(false, |f| !f.is_empty())
}

/// Resolved Anthropic compat flags (`getAnthropicCompat`).
#[derive(Debug, Clone)]
struct AnthropicCompat {
	supports_eager_tool_input_streaming: bool,
	send_session_affinity_headers: bool,
	session_affinity_format: Option<crate::types::SessionAffinityFormat>,
	supports_cache_control_on_tools: bool,
	supports_temperature: bool,
	allow_empty_signature: bool,
	supports_strict_tools: bool,
	supports_mid_convo_system_messages: bool,
	supports_mid_convo_tool_changes: bool,
}

fn get_anthropic_compat(model: &Model) -> AnthropicCompat {
	let compat: Option<&ModelCompat> = model.compat.as_ref();
	let is_open_router = model.provider() == "openrouter" || model.base.base_url.contains("openrouter.ai");
	AnthropicCompat {
		supports_eager_tool_input_streaming: compat
			.and_then(|c| c.supports_eager_tool_input_streaming)
			.unwrap_or(true),
		send_session_affinity_headers: compat.and_then(|c| c.send_session_affinity_headers).unwrap_or(is_open_router),
		session_affinity_format: compat
			.and_then(|c| c.session_affinity_format)
			.or(if is_open_router { Some(crate::types::SessionAffinityFormat::Openrouter) } else { None }),
		supports_cache_control_on_tools: compat.and_then(|c| c.supports_cache_control_on_tools).unwrap_or(true),
		supports_temperature: compat.and_then(|c| c.supports_temperature).unwrap_or(true),
		allow_empty_signature: compat.and_then(|c| c.allow_empty_signature).unwrap_or(false),
		supports_strict_tools: compat.and_then(|c| c.supports_strict_tools).unwrap_or(false),
		supports_mid_convo_system_messages: compat.and_then(|c| c.supports_mid_convo_system_messages).unwrap_or(false),
		supports_mid_convo_tool_changes: compat.and_then(|c| c.supports_mid_convo_tool_changes).unwrap_or(false),
	}
}

fn force_adaptive_thinking(model: &Model) -> bool {
	model.compat.as_ref().and_then(|c| c.force_adaptive_thinking).unwrap_or(false)
}

fn supports_mid_convo_effort(model: &Model) -> bool {
	model.compat.as_ref().and_then(|c| c.supports_mid_convo_effort).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// GitHub Copilot dynamic headers (deviation D18 — port of
// `api/github-copilot-headers.ts`)
// ---------------------------------------------------------------------------

/// Copilot expects X-Initiator to indicate whether the request is
/// user-initiated or agent-initiated (`inferCopilotInitiator`).
fn infer_copilot_initiator(messages: &[Message]) -> &'static str {
	match messages.last() {
		Some(last) if last.role() == "user" => "user",
		_ => "agent",
	}
}

/// Copilot requires Copilot-Vision-Request header when sending images
/// (`hasCopilotVisionInput`).
fn has_copilot_vision_input(messages: &[Message]) -> bool {
	messages.iter().any(|msg| match msg {
		Message::User(user) => matches!(&user.content, crate::types::UserContentField::Blocks(blocks)
			if blocks.iter().any(|c| matches!(c, UserContent::Image(_)))),
		Message::ToolResult(result) => result.content.iter().any(|c| matches!(c, UserContent::Image(_))),
		_ => false,
	})
}

fn build_copilot_dynamic_headers(messages: &[Message], has_images: bool) -> Vec<(String, String)> {
	let mut headers = vec![
		("X-Initiator".to_string(), infer_copilot_initiator(messages).to_string()),
		("Openai-Intent".to_string(), "conversation-edits".to_string()),
	];
	if has_images {
		headers.push(("Copilot-Vision-Request".to_string(), "true".to_string()));
	}
	headers
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

fn has_header(headers: Option<&ProviderHeaders>, name: &str) -> bool {
	let Some(headers) = headers else { return false };
	let expected = name.to_lowercase();
	headers.iter().any(|(key, value)| {
		key.to_lowercase() == expected
			&& value.as_ref().map(|v| !v.trim().is_empty()).unwrap_or(false)
	})
}

fn has_request_auth(api_key: Option<&str>, headers: Option<&ProviderHeaders>) -> bool {
	api_key.is_some()
		|| has_header(headers, "authorization")
		|| has_header(headers, "x-api-key")
		|| has_header(headers, "cf-aig-authorization")
}

fn assert_request_auth(provider: &str, api_key: Option<&str>, headers: Option<&ProviderHeaders>) -> Result<(), String> {
	if has_request_auth(api_key, headers) {
		Ok(())
	} else {
		Err(format!("No API key for provider: {provider}"))
	}
}

fn is_oauth_token(api_key: &str) -> bool {
	api_key.contains("sk-ant-oat")
}

// ---------------------------------------------------------------------------
// SSE decoding (verbatim port of the upstream decoder)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct ServerSentEvent {
	event: Option<String>,
	data: String,
	raw: Vec<String>,
}

#[derive(Debug, Default)]
struct SseDecoderState {
	event: Option<String>,
	data: Vec<String>,
	raw: Vec<String>,
}

fn is_anthropic_message_event(event: &str) -> bool {
	matches!(
		event,
		"message_start"
			| "message_delta"
			| "message_stop"
			| "content_block_start"
			| "content_block_delta"
			| "content_block_stop"
	)
}

fn flush_sse_event(state: &mut SseDecoderState) -> Option<ServerSentEvent> {
	if state.event.is_none() && state.data.is_empty() {
		return None;
	}

	let event = ServerSentEvent {
		event: state.event.clone(),
		data: state.data.join("\n"),
		raw: state.raw.clone(),
	};
	state.event = None;
	state.data = Vec::new();
	state.raw = Vec::new();
	Some(event)
}

fn decode_sse_line(line: &str, state: &mut SseDecoderState) -> Option<ServerSentEvent> {
	if line.is_empty() {
		return flush_sse_event(state);
	}

	state.raw.push(line.to_string());
	if line.starts_with(':') {
		return None;
	}

	let delimiter_index = line.find(':');
	let (field_name, mut value) = match delimiter_index {
		None => (line, ""),
		Some(index) => (&line[..index], &line[index + 1..]),
	};
	if let Some(stripped) = value.strip_prefix(' ') {
		value = stripped;
	}

	if field_name == "event" {
		state.event = Some(value.to_string());
	} else if field_name == "data" {
		state.data.push(value.to_string());
	}

	None
}

fn next_line_break_index(text: &str) -> Option<usize> {
	let carriage_return_index = text.find('\r');
	let newline_index = text.find('\n');
	match (carriage_return_index, newline_index) {
		(None, newline) => newline,
		(cr, None) => cr,
		(Some(cr), Some(lf)) => Some(cr.min(lf)),
	}
}

fn consume_line(text: &str) -> Option<(String, String)> {
	let line_break_index = next_line_break_index(text)?;
	let bytes = text.as_bytes();
	let mut next_index = line_break_index + 1;
	if bytes[line_break_index] == b'\r' && next_index < bytes.len() && bytes[next_index] == b'\n' {
		next_index += 1;
	}

	Some((text[..line_break_index].to_string(), text[next_index..].to_string()))
}

// ---------------------------------------------------------------------------
// Raw Anthropic stream event types (`BetaRawMessageStreamEvent` mirror)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RawMessageStreamEvent {
	MessageStart { message: StreamMessage },
	ContentBlockStart { index: u32, content_block: ContentBlockStart },
	ContentBlockDelta { index: u32, delta: ContentBlockDelta },
	ContentBlockStop { index: u32 },
	MessageDelta {
		delta: MessageDeltaBody,
		#[serde(default)]
		usage: Option<MessageDeltaUsage>,
		// Read dynamically upstream (`event.input_transformations`).
		#[serde(default)]
		input_transformations: Option<Value>,
	},
	MessageStop {},
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StreamMessage {
	id: String,
	#[serde(default)]
	model: Option<String>,
	usage: StreamUsage,
	#[serde(default)]
	input_transformations: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StreamUsage {
	#[serde(default)]
	input_tokens: Option<u64>,
	#[serde(default)]
	output_tokens: Option<u64>,
	#[serde(default, rename = "cache_read_input_tokens")]
	cache_read_input_tokens: Option<u64>,
	#[serde(default, rename = "cache_creation_input_tokens")]
	cache_creation_input_tokens: Option<u64>,
	#[serde(default, rename = "cache_creation")]
	cache_creation: Option<CacheCreation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct CacheCreation {
	#[serde(default, rename = "ephemeral_1h_input_tokens")]
	ephemeral_1h_input_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockStart {
	Text { #[serde(default)] text: Option<String> },
	Thinking { #[serde(default)] thinking: Option<String>, #[serde(default)] signature: Option<String> },
	RedactedThinking { #[serde(default)] data: Option<String> },
	ToolUse {
		#[serde(default)]
		id: Option<String>,
		#[serde(default)]
		name: Option<String>,
		#[serde(default)]
		input: Option<Value>,
	},
	Fallback {},
	#[serde(other)]
	Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockDelta {
	TextDelta { #[serde(default)] text: Option<String> },
	ThinkingDelta { #[serde(default)] thinking: Option<String> },
	InputJsonDelta { #[serde(default, rename = "partial_json")] partial_json: Option<String> },
	SignatureDelta { #[serde(default)] signature: Option<String> },
	#[serde(other)]
	Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MessageDeltaBody {
	#[serde(default)]
	stop_reason: Option<String>,
	#[serde(default)]
	stop_details: Option<StopDetails>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StopDetails {
	#[serde(default)]
	explanation: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MessageDeltaUsage {
	#[serde(default)]
	input_tokens: Option<u64>,
	#[serde(default)]
	output_tokens: Option<u64>,
	#[serde(default, rename = "cache_read_input_tokens")]
	cache_read_input_tokens: Option<u64>,
	#[serde(default, rename = "cache_creation_input_tokens")]
	cache_creation_input_tokens: Option<u64>,
	#[serde(default, rename = "cache_creation")]
	cache_creation: Option<CacheCreation>,
	#[serde(default, rename = "output_tokens_details")]
	output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct OutputTokensDetails {
	#[serde(default)]
	thinking_tokens: Option<u64>,
}

/// Streaming SSE event source over the response body
/// (`iterateAnthropicEvents`). Yields parsed provider events; `Err` carries the
/// exact upstream error strings (abort, SSE `error` event, parse failure,
/// missing `message_stop`).
struct AnthropicEventStream {
	body: BodyStream,
	signal: Option<AbortSignal>,
	state: SseDecoderState,
	buffer: String,
	finished: bool,
	saw_message_start: bool,
	saw_message_end: bool,
	queue: VecDeque<Result<RawMessageStreamEvent, String>>,
}

impl AnthropicEventStream {
	fn new(body: BodyStream, signal: Option<AbortSignal>) -> AnthropicEventStream {
		AnthropicEventStream {
			body,
			signal,
			state: SseDecoderState::default(),
			buffer: String::new(),
			finished: false,
			saw_message_start: false,
			saw_message_end: false,
			queue: VecDeque::new(),
		}
	}

	fn handle_sse(&mut self, sse: ServerSentEvent) -> Result<Option<RawMessageStreamEvent>, String> {
		if sse.event.as_deref() == Some("error") {
			return Err(sse.data);
		}

		if !is_anthropic_message_event(sse.event.as_deref().unwrap_or("")) {
			return Ok(None);
		}

		match parse_json_with_repair::<RawMessageStreamEvent>(&sse.data) {
			Ok(event) => {
				if matches!(event, RawMessageStreamEvent::MessageStart { .. }) {
					self.saw_message_start = true;
				} else if matches!(event, RawMessageStreamEvent::MessageStop { .. }) {
					self.saw_message_end = true;
				}
				Ok(Some(event))
			}
			Err(error) => Err(format!(
				"Could not parse Anthropic SSE event {}: {}; data={}; raw={}",
				sse.event.unwrap_or_default(),
				error.message(),
				sse.data,
				sse.raw.join("\\n")
			)),
		}
	}

	fn finish(&mut self) -> Option<Result<RawMessageStreamEvent, String>> {
		// Flush the trailing partial line, then any buffered event.
		let last_line = std::mem::take(&mut self.buffer);
		if !last_line.is_empty() {
			if let Some(event) = decode_sse_line(&last_line, &mut self.state) {
				match self.handle_sse(event) {
					Ok(Some(event)) => self.queue.push_back(Ok(event)),
					Ok(None) => {}
					Err(message) => {
						self.finished = true;
						return Some(Err(message));
					}
				}
			}
		}
		if let Some(event) = flush_sse_event(&mut self.state) {
			match self.handle_sse(event) {
				Ok(Some(event)) => self.queue.push_back(Ok(event)),
				Ok(None) => {}
				Err(message) => {
					self.finished = true;
					return Some(Err(message));
				}
			}
		}

		self.finished = true;
		if self.saw_message_start && !self.saw_message_end {
			// Deliver any queued events first, then surface the error.
			self.queue.push_back(Err("Anthropic stream ended before message_stop".to_string()));
		}
		self.queue.pop_front()
	}
}

impl AnthropicEventStream {
	/// Consume complete lines already in the buffer, returning the first parsed
	/// event (or error). `None` means the buffer holds no complete line.
	fn pump(&mut self) -> Option<Result<RawMessageStreamEvent, String>> {
		while let Some((line, rest)) = consume_line(&self.buffer) {
			self.buffer = rest;
			if let Some(event) = decode_sse_line(&line, &mut self.state) {
				return match self.handle_sse(event) {
					Ok(Some(event)) => Some(Ok(event)),
					Ok(None) => continue,
					Err(message) => {
						self.finished = true;
						Some(Err(message))
					}
				};
			}
		}
		None
	}
}

impl futures_core::Stream for AnthropicEventStream {
	type Item = Result<RawMessageStreamEvent, String>;

	fn poll_next(self: std::pin::Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
		let this = self.get_mut();
		loop {
			if let Some(event) = this.queue.pop_front() {
				return Poll::Ready(Some(event));
			}
			if this.finished {
				return Poll::Ready(None);
			}

			if let Some(signal) = &this.signal {
				if signal.aborted() {
					this.finished = true;
					return Poll::Ready(Some(Err("Request was aborted".to_string())));
				}
			}

			// Resume any complete lines buffered by earlier chunks before
			// polling for more bytes.
			if let Some(item) = this.pump() {
				return Poll::Ready(Some(item));
			}

			match this.body.as_mut().poll_next(cx) {
				Poll::Ready(Some(Ok(chunk))) => {
					this.buffer.push_str(&String::from_utf8_lossy(&chunk));
				}
				Poll::Ready(Some(Err(error))) => {
					this.finished = true;
					return Poll::Ready(Some(Err(error.to_string())));
				}
				Poll::Ready(None) => {
					if let Some(item) = this.finish() {
						return Poll::Ready(Some(item));
					}
					return Poll::Ready(None);
				}
				Poll::Pending => return Poll::Pending,
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Request parameter types (`MessageCreateParamsStreaming` mirror)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
struct MessageCreateParams {
	model: String,
	messages: Vec<MessageParam>,
	max_tokens: u64,
	stream: bool,
	#[serde(skip_serializing_if = "Option::is_none")]
	betas: Option<Vec<String>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	system: Option<Vec<BlockParam>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	temperature: Option<f64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tools: Option<Vec<BetaTool>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	thinking: Option<ThinkingConfig>,
	#[serde(skip_serializing_if = "Option::is_none")]
	output_config: Option<OutputConfig>,
	#[serde(skip_serializing_if = "Option::is_none")]
	metadata: Option<MetadataParam>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_choice: Option<ToolChoiceParam>,
	#[serde(skip_serializing_if = "Option::is_none")]
	fallbacks: Option<Vec<FallbackParam>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct MetadataParam {
	user_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct FallbackParam {
	model: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum MessageRole {
	User,
	Assistant,
	System,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct MessageParam {
	role: MessageRole,
	#[serde(serialize_with = "serialize_message_content")]
	content: MessageContent,
	#[serde(skip_serializing_if = "Option::is_none")]
	output_config: Option<OutputConfig>,
}

#[derive(Debug, Clone, PartialEq)]
enum MessageContent {
	Text(String),
	Blocks(Vec<BlockParam>),
}

fn serialize_message_content<S: serde::Serializer>(content: &MessageContent, serializer: S) -> Result<S::Ok, S::Error> {
	match content {
		MessageContent::Text(text) => text.serialize(serializer),
		MessageContent::Blocks(blocks) => blocks.serialize(serializer),
	}
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
enum BlockParam {
	#[serde(rename = "text")]
	Text {
		text: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	#[serde(rename = "image")]
	Image {
		source: ImageSource,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	#[serde(rename = "thinking")]
	Thinking { thinking: String, signature: String },
	#[serde(rename = "redacted_thinking")]
	RedactedThinking { data: String },
	#[serde(rename = "tool_use")]
	ToolUse { id: String, name: String, input: crate::types::JsonObject },
	#[serde(rename = "tool_result")]
	ToolResult {
		tool_use_id: String,
		content: ToolResultContent,
		is_error: bool,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	#[serde(rename = "tool_addition")]
	ToolAddition { tool: ToolDefinitionRef },
	#[serde(rename = "tool_removal")]
	ToolRemoval {
		tool: ToolReferenceRef,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct ImageSource {
	#[serde(rename = "type")]
	source_type: &'static str,
	media_type: String,
	data: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct ToolDefinitionRef {
	#[serde(rename = "type")]
	tag: &'static str,
	definition: BetaTool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct ToolReferenceRef {
	#[serde(rename = "type")]
	tag: &'static str,
	name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct BetaTool {
	name: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	description: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	eager_input_streaming: Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	strict: Option<bool>,
	input_schema: InputSchema,
	#[serde(skip_serializing_if = "Option::is_none")]
	cache_control: Option<CacheControl>,
	#[serde(skip_serializing_if = "Option::is_none")]
	defer_loading: Option<bool>,
}

/// `input_schema` for a tool: `{ type: "object", properties, required }`, or
/// the strict-validated parameters merged under those pinned keys.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct InputSchema(Value);

impl InputSchema {
	fn legacy(properties: crate::types::JsonObject, required: Vec<String>) -> InputSchema {
		let mut object = crate::types::JsonObject::new();
		object.insert("type".to_string(), json!("object"));
		object.insert("properties".to_string(), Value::Object(properties));
		object.insert("required".to_string(), json!(required));
		InputSchema(Value::Object(object))
	}

	fn strict(parameters: Value, properties: crate::types::JsonObject, required: Vec<String>) -> InputSchema {
		let mut object = match parameters {
			Value::Object(map) => map,
			_ => crate::types::JsonObject::new(),
		};
		object.insert("type".to_string(), json!("object"));
		object.insert("properties".to_string(), Value::Object(properties));
		object.insert("required".to_string(), json!(required));
		InputSchema(Value::Object(object))
	}
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ThinkingConfig {
	Adaptive {
		#[serde(skip_serializing_if = "Option::is_none")]
		display: Option<AnthropicThinkingDisplay>,
		#[serde(skip_serializing_if = "Option::is_none")]
		block_binding: Option<BlockBinding>,
	},
	Enabled {
		budget_tokens: u64,
		#[serde(skip_serializing_if = "Option::is_none")]
		display: Option<AnthropicThinkingDisplay>,
	},
	Disabled,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct BlockBinding {
	prefix_mismatch_behavior: &'static str,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct OutputConfig {
	effort: AnthropicEffort,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ToolChoiceParam {
	Auto,
	Any,
	None,
	Tool { name: String },
}

impl AnthropicEffort {
	fn as_str(self) -> &'static str {
		match self {
			AnthropicEffort::Low => "low",
			AnthropicEffort::Medium => "medium",
			AnthropicEffort::High => "high",
			AnthropicEffort::Xhigh => "xhigh",
			AnthropicEffort::Max => "max",
		}
	}

	fn parse(value: &str) -> Option<AnthropicEffort> {
		match value {
			"low" => Some(AnthropicEffort::Low),
			"medium" => Some(AnthropicEffort::Medium),
			"high" => Some(AnthropicEffort::High),
			"xhigh" => Some(AnthropicEffort::Xhigh),
			"max" => Some(AnthropicEffort::Max),
			_ => None,
		}
	}
}

fn thinking_level_to_model_level(level: ThinkingLevel) -> ModelThinkingLevel {
	match level {
		ThinkingLevel::Minimal => ModelThinkingLevel::Minimal,
		ThinkingLevel::Low => ModelThinkingLevel::Low,
		ThinkingLevel::Medium => ModelThinkingLevel::Medium,
		ThinkingLevel::High => ModelThinkingLevel::High,
		ThinkingLevel::Xhigh => ModelThinkingLevel::Xhigh,
		ThinkingLevel::Max => ModelThinkingLevel::Max,
	}
}

// ---------------------------------------------------------------------------
// Stop reasons
// ---------------------------------------------------------------------------

fn map_stop_reason(
	reason: &str,
	stop_details: Option<&StopDetails>,
) -> Result<(StopReason, Option<String>), String> {
	match reason {
		"end_turn" => Ok((StopReason::Stop, None)),
		"max_tokens" => Ok((StopReason::Length, None)),
		"tool_use" => Ok((StopReason::ToolUse, None)),
		"refusal" => Ok((
			StopReason::Error,
			Some(
				stop_details
					.and_then(|d| d.explanation.clone())
					.filter(|e| !e.is_empty())
					.unwrap_or_else(|| "The model refused to complete the request".to_string()),
			),
		)),
		"pause_turn" => Ok((StopReason::Stop, None)), // Stop is good enough -> resubmit
		"stop_sequence" => Ok((StopReason::Stop, None)), // We don't supply stop sequences
		"sensitive" => Ok((StopReason::Error, Some("Provider stopped with: sensitive".to_string()))),
		// Handle unknown stop reasons gracefully (API may add new values)
		_ => Err(format!("Unhandled stop reason: {reason}")),
	}
}

// ---------------------------------------------------------------------------
// Betas + params
// ---------------------------------------------------------------------------

fn configured_beta_features(model: &Model, headers: Option<&ProviderHeaders>) -> Option<Vec<String>> {
	let mut configured: Option<String> = None;
	for source in [model.base.headers.as_ref(), options_headers_map(headers).as_ref()] {
		let Some(map) = source else { continue };
		for (name, value) in map.iter() {
			if name.eq_ignore_ascii_case("anthropic-beta") {
				configured = Some(value.clone());
			}
		}
	}
	match configured {
		// Explicit null suppresses all beta features.
		Some(value) if value.is_empty() => Some(Vec::new()),
		Some(value) => Some(
			value
				.split(',')
				.map(|feature| feature.trim().to_string())
				.filter(|feature| !feature.is_empty())
				.collect::<Vec<_>>(),
		),
		None => None,
	}
}

/// Options headers may carry `null` values (suppression); upstream iterates
/// `Object.entries(headers ?? {})` where `null` is a present-but-null entry.
fn options_headers_map(headers: Option<&ProviderHeaders>) -> Option<BTreeMap<String, String>> {
	headers.map(|map| {
		map.iter()
			.map(|(key, value)| (key.clone(), value.clone().unwrap_or_default()))
			.collect()
	})
}

/// Upstream distinguishes `null` (suppress all betas) from absent. The Rust
/// port represents the null case as `Some("")` in the header map.
fn get_beta_features(
	model: &Model,
	context: &TranscriptContext,
	is_oauth_token: bool,
	native_tool_changes: bool,
	options: Option<&AnthropicApiOptions>,
	headers: Option<&ProviderHeaders>,
) -> Vec<String> {
	let configured_features = configured_beta_features(model, headers);
	if let Some(features) = configured_features {
		return dedupe(features);
	}
	default_beta_features(model, context, is_oauth_token, native_tool_changes, options)
}

fn dedupe(features: Vec<String>) -> Vec<String> {
	let mut seen = std::collections::BTreeSet::new();
	features.into_iter().filter(|f| seen.insert(f.clone())).collect()
}

fn default_beta_features(
	model: &Model,
	context: &TranscriptContext,
	is_oauth_token: bool,
	native_tool_changes: bool,
	options: Option<&AnthropicApiOptions>,
) -> Vec<String> {
	let mut features: Vec<String> = Vec::new();
	if is_oauth_token {
		features.push("claude-code-20250219".to_string());
		features.push("oauth-2025-04-20".to_string());
	}
	if should_use_fine_grained_tool_streaming_beta(model, context) {
		features.push(FINE_GRAINED_TOOL_STREAMING_BETA.to_string());
	}
	let thinking_enabled = options.and_then(|o| o.thinking_enabled);
	if model.reasoning
		&& thinking_enabled == Some(true)
		&& options.and_then(|o| o.interleaved_thinking).unwrap_or(true)
		&& !force_adaptive_thinking(model)
	{
		features.push(INTERLEAVED_THINKING_BETA.to_string());
	}
	if should_use_server_side_fallback_beta(model) {
		features.push(SERVER_SIDE_FALLBACK_BETA.to_string());
	}
	if supports_mid_convo_effort(model) {
		features.push(MID_CONVERSATION_OUTPUT_CONFIG_BETA.to_string());
		features.push(THINKING_BINDING_CONTROLS_BETA.to_string());
	}
	if native_tool_changes {
		features.push(INLINE_TOOLS_BETA.to_string());
	}
	dedupe(features)
}

fn should_use_fine_grained_tool_streaming_beta(model: &Model, context: &TranscriptContext) -> bool {
	!get_current_tools(context.messages()).is_empty() && !get_anthropic_compat(model).supports_eager_tool_input_streaming
}

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

/// Normalize tool call IDs to match Anthropic's required pattern and length.
fn normalize_tool_call_id(id: &str) -> String {
	let cleaned: String = id
		.chars()
		.map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
		.collect();
	cleaned.chars().take(64).collect()
}

fn convert_tool_result(msg: &ToolResultMessage) -> BlockParam {
	BlockParam::ToolResult {
		tool_use_id: msg.tool_call_id.clone(),
		content: convert_content_blocks(&msg.content),
		is_error: msg.is_error,
		cache_control: None,
	}
}

struct ConvertedAnthropicMessages {
	messages: Vec<MessageParam>,
	assistant_levels: BTreeMap<usize, AnthropicEffort>,
}

#[allow(clippy::too_many_arguments)]
fn convert_messages(
	transformed_messages: &[Message],
	is_oauth_token: bool,
	cache_control: Option<&CacheControl>,
	allow_empty_signature: bool,
	managed_provider: Option<&str>,
	convert_tool_definitions: Option<&dyn Fn(&[Tool]) -> Result<Vec<BetaTool>, String>>,
) -> Result<ConvertedAnthropicMessages, String> {
	let mut params: Vec<MessageParam> = Vec::new();
	let mut assistant_levels: BTreeMap<usize, AnthropicEffort> = BTreeMap::new();
	// Later system messages are held back and emitted directly before the next
	// assistant message (or at the end of the transcript).
	let mut pending_system_messages: Vec<MessageParam> = Vec::new();
	fn flush_pending_system_messages(params: &mut Vec<MessageParam>, pending: &mut Vec<MessageParam>) {
		params.append(pending);
	}

	let mut i = 0;
	while i < transformed_messages.len() {
		let msg = &transformed_messages[i];

		match msg {
			Message::System(system) => {
				// Later system messages only reach this point when the model accepts
				// them natively.
				let text = render_system_message_update(system);
				let mut blocks: Vec<BlockParam> = Vec::new();
				if !text.is_empty() {
					blocks.push(BlockParam::Text {
						text: sanitize_surrogates(&text),
						cache_control: None,
					});
				}
				if let Some(convert_tool_definitions) = convert_tool_definitions {
					let added = system.tools_added.clone().unwrap_or_default();
					let redefined: std::collections::BTreeSet<&str> =
						added.iter().map(|tool| tool.name.as_str()).collect();
					for tool in system.tools_removed.clone().unwrap_or_default() {
						// A new definition under the same name replaces the old one.
						if redefined.contains(tool.name.as_str()) {
							continue;
						}
						let name = if is_oauth_token { to_claude_code_name(&tool.name) } else { tool.name.clone() };
						blocks.push(BlockParam::ToolRemoval {
							tool: ToolReferenceRef { tag: "tool_reference", name },
							cache_control: None,
						});
					}
					for definition in convert_tool_definitions(&added)? {
						blocks.push(BlockParam::ToolAddition {
							tool: ToolDefinitionRef { tag: "tool_definition", definition },
						});
					}
				}
				if !blocks.is_empty() {
					pending_system_messages.push(MessageParam {
						role: MessageRole::System,
						content: MessageContent::Blocks(blocks),
						output_config: None,
					});
				}
			}
			Message::User(user) => match &user.content {
				crate::types::UserContentField::Text(text) => {
					if !text.trim().is_empty() {
						params.push(MessageParam {
							role: MessageRole::User,
							content: MessageContent::Text(sanitize_surrogates(text)),
							output_config: None,
						});
					}
				}
				crate::types::UserContentField::Blocks(content) => {
					let blocks: Vec<BlockParam> = content
						.iter()
						.map(|item| match item {
							UserContent::Text(text) => BlockParam::Text {
								text: sanitize_surrogates(&text.text),
								cache_control: None,
							},
							UserContent::Image(image) => BlockParam::Image {
								source: ImageSource {
									source_type: "base64",
									media_type: image.mime_type.clone(),
									data: image.data.clone(),
								},
								cache_control: None,
							},
						})
						.filter(|block| match block {
							BlockParam::Text { text, .. } => !text.trim().is_empty(),
							_ => true,
						})
						.collect();
					if blocks.is_empty() {
						i += 1;
						continue;
					}
					params.push(MessageParam {
						role: MessageRole::User,
						content: MessageContent::Blocks(blocks),
						output_config: None,
					});
				}
			},
			Message::Assistant(assistant) => {
				flush_pending_system_messages(&mut params, &mut pending_system_messages);
				let mut blocks: Vec<BlockParam> = Vec::new();

				for block in &assistant.content {
					match block {
						AssistantContent::Text(text) => {
							if text.text.trim().is_empty() {
								continue;
							}
							blocks.push(BlockParam::Text {
								text: sanitize_surrogates(&text.text),
								cache_control: None,
							});
						}
						AssistantContent::Thinking(thinking) => {
							// Redacted thinking: pass the opaque payload back.
							if thinking.redacted == Some(true) {
								blocks.push(BlockParam::RedactedThinking {
									data: thinking.thinking_signature.clone().unwrap_or_default(),
								});
								continue;
							}
							let thinking_signature = thinking.thinking_signature.as_deref().unwrap_or("");
							let has_thinking_signature = !thinking_signature.trim().is_empty();
							if thinking.thinking.trim().is_empty() && !has_thinking_signature {
								continue;
							}
							// Missing/empty signature (e.g. aborted stream): convert to
							// plain text unless the model accepts empty signatures.
							if !has_thinking_signature {
								if allow_empty_signature {
									blocks.push(BlockParam::Thinking {
										thinking: sanitize_surrogates(&thinking.thinking),
										signature: String::new(),
									});
								} else {
									blocks.push(BlockParam::Text {
										text: sanitize_surrogates(&thinking.thinking),
										cache_control: None,
									});
								}
							} else {
								blocks.push(BlockParam::Thinking {
									thinking: sanitize_surrogates(&thinking.thinking),
									signature: thinking_signature.to_string(),
								});
							}
						}
						AssistantContent::ToolCall(tool_call) => {
							blocks.push(BlockParam::ToolUse {
								id: tool_call.id.clone(),
								name: if is_oauth_token {
									to_claude_code_name(&tool_call.name)
								} else {
									tool_call.name.clone()
								},
								input: tool_call.arguments.clone(),
							});
						}
					}
				}
				if blocks.is_empty() {
					i += 1;
					continue;
				}
				let message_index = params.len();
				params.push(MessageParam {
					role: MessageRole::Assistant,
					content: MessageContent::Blocks(blocks),
					output_config: None,
				});
				if let Some(managed_provider) = managed_provider {
					if assistant.api == "anthropic-messages"
						&& assistant.provider == managed_provider
						&& assistant
							.provider_thinking_level
							.as_deref()
							.and_then(AnthropicEffort::parse)
							.is_some()
					{
						assistant_levels.insert(
							message_index,
							assistant
								.provider_thinking_level
								.as_deref()
								.and_then(AnthropicEffort::parse)
								.expect("checked above"),
						);
					}
				}
			}
			Message::ToolResult(_) => {
				// Collect all consecutive toolResult messages, needed for z.ai
				// Anthropic endpoint.
				let mut tool_results: Vec<BlockParam> = Vec::new();
				let mut j = i;
				while j < transformed_messages.len() {
					match &transformed_messages[j] {
						Message::ToolResult(result) => tool_results.push(convert_tool_result(result)),
						_ => break,
					}
					j += 1;
				}

				// Skip the messages we've already processed.
				i = j - 1;

				params.push(MessageParam {
					role: MessageRole::User,
					content: MessageContent::Blocks(tool_results),
					output_config: None,
				});
			}
		}

		i += 1;
	}

	flush_pending_system_messages(&mut params, &mut pending_system_messages);

	// Add cache_control to the last user or system message to cache conversation history
	if let Some(cache_control) = cache_control {
		if let Some(last_message) = params.last_mut() {
			if matches!(last_message.role, MessageRole::User | MessageRole::System) {
				match &mut last_message.content {
					MessageContent::Blocks(blocks) => {
						if let Some(last_block) = blocks.last_mut() {
							apply_cache_control(last_block, cache_control);
						}
					}
					MessageContent::Text(text) => {
						let text = std::mem::take(text);
						last_message.content = MessageContent::Blocks(vec![BlockParam::Text {
							text,
							cache_control: Some(cache_control.clone()),
						}]);
					}
				}
			}
		}
	}

	Ok(ConvertedAnthropicMessages { messages: params, assistant_levels })
}

fn apply_cache_control(block: &mut BlockParam, cache_control: &CacheControl) {
	match block {
		BlockParam::Text { cache_control: slot, .. }
		| BlockParam::Image { cache_control: slot, .. }
		| BlockParam::ToolResult { cache_control: slot, .. } => *slot = Some(cache_control.clone()),
		BlockParam::ToolAddition { tool } => tool.definition.cache_control = Some(cache_control.clone()),
		BlockParam::ToolRemoval { cache_control: slot, .. } => *slot = Some(cache_control.clone()),
		_ => {}
	}
}

fn insert_thinking_level_messages(converted: ConvertedAnthropicMessages, active_effort: AnthropicEffort) -> Vec<MessageParam> {
	let mut messages: Vec<MessageParam> = Vec::new();
	for (index, message) in converted.messages.into_iter().enumerate() {
		if let Some(historical_effort) = converted.assistant_levels.get(&index) {
			messages.push(MessageParam {
				role: MessageRole::System,
				content: MessageContent::Blocks(Vec::new()),
				output_config: Some(OutputConfig { effort: *historical_effort }),
			});
		}
		messages.push(message);
	}
	messages.push(MessageParam {
		role: MessageRole::System,
		content: MessageContent::Blocks(Vec::new()),
		output_config: Some(OutputConfig { effort: active_effort }),
	});
	messages
}

// ---------------------------------------------------------------------------
// Strict tool schemas (Anthropic-specific keyword check)
// ---------------------------------------------------------------------------

/// Keywords Anthropic strict tool use rejects with a 400 for the whole request
/// (`ANTHROPIC_STRICT_UNSUPPORTED_KEYWORDS`).
const ANTHROPIC_STRICT_UNSUPPORTED_KEYWORDS: &[&str] = &[
	"minimum",
	"maximum",
	"exclusiveMinimum",
	"exclusiveMaximum",
	"multipleOf",
	"maxItems",
	"uniqueItems",
	"minContains",
	"maxContains",
	"minProperties",
	"maxProperties",
];

const ANTHROPIC_STRICT_STRING_FORMATS: &[&str] = &[
	"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid",
];

fn is_anthropic_strict_unsupported_keyword(key: &str, value: &Value) -> bool {
	if ANTHROPIC_STRICT_UNSUPPORTED_KEYWORDS.contains(&key) {
		return true;
	}
	if key == "minItems" {
		return *value != Value::from(0) && *value != Value::from(1);
	}
	if key == "format" {
		return !value.is_string() || value.as_str().map_or(true, |f| !ANTHROPIC_STRICT_STRING_FORMATS.contains(&f));
	}
	false
}

fn convert_tools(
	tools: &[Tool],
	is_oauth_token: bool,
	supports_eager_tool_input_streaming: bool,
	supports_strict_tools: bool,
	cache_control: Option<&CacheControl>,
) -> Result<Vec<BetaTool>, String> {
	let mut converted = Vec::with_capacity(tools.len());
	for (index, tool) in tools.iter().enumerate() {
		let strict = crate::api::constrained_sampling::resolve_json_schema_strict_sampling_with(
			tool,
			supports_strict_tools,
			Some(&is_anthropic_strict_unsupported_keyword),
		)?;
		let parameters = crate::api::constrained_sampling::get_json_schema_tool_parameters(tool, strict == Some(true))
			.map_err(|error| error.to_string())?;

		let (properties, required) = schema_properties_and_required(&parameters);
		let input_schema = if strict == Some(true) {
			InputSchema::strict(parameters, properties, required)
		} else {
			InputSchema::legacy(properties, required)
		};

		converted.push(BetaTool {
			name: if is_oauth_token { to_claude_code_name(&tool.name) } else { tool.name.clone() },
			description: Some(tool.description.clone()),
			eager_input_streaming: supports_eager_tool_input_streaming.then_some(true),
			strict: (strict == Some(true)).then_some(true),
			input_schema,
			cache_control: cache_control.and_then(|cc| (index == tools.len() - 1).then(|| cc.clone())),
			defer_loading: None,
		});
	}
	Ok(converted)
}

fn schema_properties_and_required(schema: &Value) -> (crate::types::JsonObject, Vec<String>) {
	let properties = schema
		.get("properties")
		.and_then(Value::as_object)
		.cloned()
		.unwrap_or_default();
	let required = schema
		.get("required")
		.and_then(Value::as_array)
		.map(|items| {
			items
				.iter()
				.filter_map(Value::as_str)
				.map(str::to_string)
				.collect()
		})
		.unwrap_or_default();
	(properties, required)
}

// ---------------------------------------------------------------------------
// Params assembly
// ---------------------------------------------------------------------------

fn build_params(
	model: &Model,
	context: &TranscriptContext,
	is_oauth_token: bool,
	options: &StreamOptions,
	anthropic: &AnthropicApiOptions,
) -> Result<MessageCreateParams, String> {
	let compat = get_anthropic_compat(model);
	let cache_control = get_cache_control(model, options.cache_retention, options.base.env.as_ref()).cache_control;
	let initial_system_message = get_initial_system_message(context.messages());
	let initial_system_text = initial_system_message.map(get_system_message_text).unwrap_or_default();
	let transformed_messages = crate::api::transform_messages::transform_messages(
		context.messages(),
		model,
		Some(&|id: &str, _model: &Model, _message: &AssistantMessage| normalize_tool_call_id(id)),
	);
	let conversation_messages: &[Message] = if initial_system_message.is_some() {
		&transformed_messages[1..]
	} else {
		&transformed_messages
	};
	// Native tool changes keep the request-level tool list fixed and define
	// every later tool by value in a `tool_addition` block.
	let initial_tools: Vec<Tool> = initial_system_message
		.and_then(|system| system.tools_added.clone())
		.unwrap_or_default();
	let native_tool_changes = compat.supports_mid_convo_system_messages
		&& compat.supports_mid_convo_tool_changes
		&& !initial_tools.is_empty();
	let managed = supports_mid_convo_effort(model).then(|| model.provider().to_string());
	let convert_tool_definitions: Option<&dyn Fn(&[Tool]) -> Result<Vec<BetaTool>, String>> = if native_tool_changes {
		Some(&|tools: &[Tool]| {
			convert_tools(
				tools,
				is_oauth_token,
				compat.supports_eager_tool_input_streaming,
				compat.supports_strict_tools,
				None,
			)
		})
	} else {
		None
	};
	let converted = convert_messages(
		conversation_messages,
		is_oauth_token,
		cache_control.as_ref(),
		compat.allow_empty_signature,
		managed.as_deref(),
		convert_tool_definitions,
	)?;
	let active_effort = anthropic.effort.unwrap_or(AnthropicEffort::High);
	let beta_features = get_beta_features(model, context, is_oauth_token, native_tool_changes, Some(anthropic), options.base.headers.as_ref());
	let mut params = MessageCreateParams {
		model: model.id().to_string(),
		messages: if supports_mid_convo_effort(model) {
			insert_thinking_level_messages(converted, active_effort)
		} else {
			converted.messages
		},
		max_tokens: options.max_tokens.unwrap_or(model.max_tokens),
		stream: true,
		betas: (!beta_features.is_empty()).then_some(beta_features),
		system: None,
		temperature: None,
		tools: None,
		thinking: None,
		output_config: None,
		metadata: None,
		tool_choice: None,
		fallbacks: None,
	};

	// For OAuth tokens, we MUST include Claude Code identity
	if is_oauth_token {
		let mut system = vec![BlockParam::Text {
			text: "You are Claude Code, Anthropic's official CLI for Claude.".to_string(),
			cache_control: cache_control.clone(),
		}];
		if !initial_system_text.is_empty() {
			system.push(BlockParam::Text {
				text: sanitize_surrogates(&initial_system_text),
				cache_control: cache_control.clone(),
			});
		}
		params.system = Some(system);
	} else if !initial_system_text.is_empty() {
		// Add cache control to system prompt for non-OAuth tokens
		params.system = Some(vec![BlockParam::Text {
			text: sanitize_surrogates(&initial_system_text),
			cache_control: cache_control.clone(),
		}]);
	}

	// Temperature is incompatible with extended thinking and unsupported on
	// Claude Opus 4.7+.
	if let Some(temperature) = options.temperature {
		if anthropic.thinking_enabled != Some(true)
			&& !supports_mid_convo_effort(model)
			&& compat.supports_temperature
		{
			params.temperature = Some(temperature);
		}
	}

	let tool_cache_control = compat.supports_cache_control_on_tools.then(|| cache_control.clone()).flatten();
	if native_tool_changes {
		// Initial tools stay active with the cache breakpoint on the last one,
		// followed by the placeholder.
		let mut tools = convert_tools(
			&initial_tools,
			is_oauth_token,
			compat.supports_eager_tool_input_streaming,
			compat.supports_strict_tools,
			tool_cache_control.as_ref(),
		)?;
		tools.push(deferred_tool_placeholder());
		params.tools = Some(tools);
	} else {
		let tools = get_current_tools(context.messages());
		if !tools.is_empty() {
			params.tools = Some(convert_tools(
				&tools,
				is_oauth_token,
				compat.supports_eager_tool_input_streaming,
				compat.supports_strict_tools,
				tool_cache_control.as_ref(),
			)?);
		}
	}

	// Managed effort models always use adaptive thinking so prefix mismatches
	// can be dropped instead of surfacing as persistent 400 responses.
	if supports_mid_convo_effort(model) {
		params.thinking = Some(ThinkingConfig::Adaptive {
			display: Some(anthropic.thinking_display.unwrap_or(AnthropicThinkingDisplay::Summarized)),
			block_binding: Some(BlockBinding { prefix_mismatch_behavior: "drop_block" }),
		});
		params.output_config = Some(OutputConfig { effort: AnthropicEffort::High });
	} else if model.reasoning {
		if anthropic.thinking_enabled == Some(true) {
			// Default to "summarized" so Opus 4.7 and Mythos Preview behave like
			// older Claude 4 models.
			let display = anthropic.thinking_display.unwrap_or(AnthropicThinkingDisplay::Summarized);
			if force_adaptive_thinking(model) {
				// Adaptive thinking: Claude decides when and how much to think.
				params.thinking = Some(ThinkingConfig::Adaptive { display: Some(display), block_binding: None });
				if let Some(effort) = anthropic.effort {
					params.output_config = Some(OutputConfig { effort });
				}
			} else {
				// Budget-based thinking for older models
				params.thinking = Some(ThinkingConfig::Enabled {
					budget_tokens: match anthropic.thinking_budget_tokens {
						Some(0) | None => 1024,
						Some(budget) => budget,
					},
					display: Some(display),
				});
			}
		} else if anthropic.thinking_enabled == Some(false)
			&& model
				.thinking_level_map
				.as_ref()
				.and_then(|map| map.get(&ModelThinkingLevel::Off))
				!= Some(&None)
		{
			params.thinking = Some(ThinkingConfig::Disabled);
		}
	}

	if let Some(metadata) = &options.metadata {
		if let Some(user_id) = metadata.get("user_id").and_then(Value::as_str) {
			params.metadata = Some(MetadataParam {
				user_id: user_id.to_string(),
			});
		}
	}

	if let Some(tool_choice) = &anthropic.tool_choice {
		params.tool_choice = Some(match tool_choice {
			AnthropicToolChoice::BuiltIn(AnthropicBuiltInToolChoice::Auto) => ToolChoiceParam::Auto,
			AnthropicToolChoice::BuiltIn(AnthropicBuiltInToolChoice::Any) => ToolChoiceParam::Any,
			AnthropicToolChoice::BuiltIn(AnthropicBuiltInToolChoice::None) => ToolChoiceParam::None,
			AnthropicToolChoice::Tool { name, .. } => ToolChoiceParam::Tool { name: name.clone() },
		});
	}

	let allowed_fallback_models = model
		.compat
		.as_ref()
		.and_then(|c| c.allowed_fallback_models.as_ref());
	if let Some(fallbacks) = allowed_fallback_models {
		if !fallbacks.is_empty() {
			params.fallbacks = Some(
				fallbacks
					.iter()
					.map(|fallback| FallbackParam {
						model: fallback.model.clone(),
					})
					.collect(),
			);
		}
	}

	Ok(params)
}

// ---------------------------------------------------------------------------
// HTTP request construction (deviation D16 — SDK client replacement)
// ---------------------------------------------------------------------------

/// Merged header sources (`mergeHeaders`): later sources win; `None` values
/// (from options headers) suppress the entry entirely.
struct HeaderBuilder {
	headers: Vec<(String, String)>,
}

impl HeaderBuilder {
	fn new() -> HeaderBuilder {
		HeaderBuilder { headers: Vec::new() }
	}

	fn set(&mut self, name: impl Into<String>, value: impl Into<String>) {
		let name = name.into();
		let value = value.into();
		if let Some(slot) = self
			.headers
			.iter_mut()
			.find(|(key, _)| key.eq_ignore_ascii_case(&name))
		{
			slot.1 = value;
		} else {
			self.headers.push((name, value));
		}
	}

	fn suppress(&mut self, name: &str) {
		self.headers.retain(|(key, _)| !key.eq_ignore_ascii_case(name));
	}

	fn extend_model_headers(&mut self, headers: Option<&BTreeMap<String, String>>) {
		if let Some(headers) = headers {
			for (key, value) in headers {
				self.set(key.clone(), value.clone());
			}
		}
	}

	fn extend_options_headers(&mut self, headers: Option<&ProviderHeaders>) {
		if let Some(headers) = headers {
			for (key, value) in headers {
				match value {
					Some(value) => self.set(key.clone(), value.clone()),
					None => self.suppress(key),
				}
			}
		}
	}

	fn has(&self, name: &str) -> bool {
		self.headers.iter().any(|(key, _)| key.eq_ignore_ascii_case(name))
	}

	fn finish(self) -> Vec<(String, String)> {
		self.headers
	}
}

/// Clonable request template; a fresh [`HttpRequest`] is built per retry
/// attempt (upstream the SDK client re-issues the same request).
#[derive(Clone)]
struct RequestTemplate {
	url: String,
	headers: Arc<Vec<(String, String)>>,
	body: Arc<Vec<u8>>,
	timeout_ms: Option<u64>,
	signal: Option<AbortSignal>,
}

impl RequestTemplate {
	fn to_request(&self) -> HttpRequest {
		let mut request = HttpRequest::new("POST", self.url.clone());
		request.headers = (*self.headers).clone();
		request.body = HttpBody::Bytes((*self.body).clone());
		request.timeout_ms = self.timeout_ms;
		request.signal = self.signal.clone();
		request
	}
}

/// Builds the streaming HTTP request (`createClient` + SDK request assembly).
#[allow(clippy::too_many_arguments)]
fn build_request(
	model: &Model,
	params: &Value,
	api_key: Option<&str>,
	options_headers: Option<&ProviderHeaders>,
	session_id: Option<&str>,
	is_oauth: bool,
	copilot_dynamic_headers: Option<&[(String, String)]>,
	signal: Option<AbortSignal>,
	timeout_ms: Option<u64>,
) -> RequestTemplate {
	let compat = get_anthropic_compat(model);

	let mut headers = HeaderBuilder::new();
	headers.set("User-Agent", get_pi_user_agent());
	headers.set("accept", "application/json");
	headers.set("anthropic-dangerous-direct-browser-access", "true");

	if is_oauth {
		// OAuth: Claude Code identity headers (mergeHeaders override order —
		// these replace the pi user agent).
		headers.set("user-agent", format!("claude-cli/{CLAUDE_CODE_VERSION}"));
		headers.set("x-app", "cli");
	}

	// Session affinity headers.
	if compat.send_session_affinity_headers {
		if let Some(session_id) = session_id {
			let header = if compat.session_affinity_format == Some(crate::types::SessionAffinityFormat::Openrouter) {
				"x-session-id"
			} else {
				"x-session-affinity"
			};
			headers.set(header, session_id);
		}
	}

	headers.extend_model_headers(model.base.headers.as_ref());
	if let Some(dynamic) = copilot_dynamic_headers {
		for (name, value) in dynamic {
			headers.set(name.clone(), value.clone());
		}
	}
	headers.extend_options_headers(options_headers);

	// Fixed transport headers (SDK always sends these; caller headers win).
	if !headers.has("content-type") {
		headers.set("content-type", "application/json");
	}
	if !headers.has("anthropic-version") {
		headers.set("anthropic-version", "2023-06-01");
	}

	// Auth headers are set after defaults; callers can still supply their own.
	if let Some(api_key) = api_key {
		if is_oauth || model.provider() == "github-copilot" {
			if !headers.has("authorization") {
				headers.set("authorization", format!("Bearer {api_key}"));
			}
		} else if !headers.has("x-api-key") {
			headers.set("x-api-key", api_key);
		}
	}

	let url = format!("{}/v1/messages", model.base.base_url.trim_end_matches('/'));
	RequestTemplate {
		url,
		headers: Arc::new(headers.finish()),
		body: Arc::new(serde_json::to_vec(params).expect("params serialize")),
		timeout_ms,
		signal,
	}
}

/// Converts a non-2xx response into the retry error (upstream SDK `APIError`).
async fn response_to_error(response: HttpResponse) -> crate::http::ProviderHttpError {
	let status = response.status;
	let headers = response.headers.clone();
	let body = response.body.collect().await.unwrap_or_default();
	let body_text = String::from_utf8_lossy(&body).to_string();
	let message = if body_text.is_empty() {
		format!("Request failed with status {status}")
	} else {
		format!("{body_text} (status {status})")
	};
	crate::http::ProviderHttpError {
		message,
		status: Some(status),
		headers,
		body: Some(body_text),
		source: None,
	}
}

// ---------------------------------------------------------------------------
// stream()
// ---------------------------------------------------------------------------

/// Upstream `stream` export for the anthropic-messages API.
pub fn stream(model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
	let stream = AssistantMessageEventStream::new();
	let compat = get_anthropic_compat(model);
	let normalized_context = resolve_transcript(context, Some(compat.supports_mid_convo_system_messages));
	let current_tools = get_current_tools(normalized_context.messages());

	let model = model.clone();
	let options = options.clone();
	let inner_stream = stream.clone();
	tokio::spawn(async move {
		run_stream(model, normalized_context, current_tools, options, inner_stream).await;
	});

	stream
}

struct BlockScratch {
	anthropic_index: u32,
	partial_json: String,
	/// Set once `content_block_stop` finalized the block (upstream deletes
	/// `index`, so later events no longer match the block).
	stopped: bool,
}

#[allow(clippy::too_many_lines)]
async fn run_stream(
	model: Model,
	normalized_context: TranscriptContext,
	current_tools: Vec<Tool>,
	options: StreamOptions,
	stream: AssistantMessageEventStream,
) {
	let anthropic = options.anthropic.clone();
	let provider_thinking_level = if supports_mid_convo_effort(&model) {
		Some(anthropic.effort.unwrap_or(AnthropicEffort::High).as_str().to_string())
	} else {
		None
	};
	let mut output = AssistantMessage {
		content: Vec::new(),
		api: model.api().to_string(),
		provider: model.provider().to_string(),
		model: model.id().to_string(),
		response_model: None,
		response_id: None,
		provider_thinking_level: provider_thinking_level.clone(),
		thinking_level: None,
		diagnostics: None,
		usage: Default::default(),
		stop_reason: StopReason::Pending,
		deferred: None,
		error_message: None,
		raw_stop_reason: None,
		end_turn: None,
		timestamp: now_ms(),
	};

	let result: Result<(), String> = async {
		let api_key = options.base.api_key.clone();
		assert_request_auth(model.provider(), api_key.as_deref(), options.base.headers.as_ref())?;

		// Copilot dynamic headers (deviation D18).
		let copilot_dynamic_headers = if model.provider() == "github-copilot" {
			Some(build_copilot_dynamic_headers(
				normalized_context.messages(),
				has_copilot_vision_input(normalized_context.messages()),
			))
		} else {
			None
		};

		let cache_retention = resolve_cache_retention(options.cache_retention, options.base.env.as_ref());
		let cache_session_id = if cache_retention == CacheRetention::None { None } else { options.session_id.clone() };

		let is_oauth = api_key.as_deref().map(is_oauth_token).unwrap_or(false);

		let params = build_params(&model, &normalized_context, is_oauth, &options, &anthropic)?;
		let mut params_value = serde_json::to_value(&params).expect("params serialize");
		if let Some(hook) = &options.base.on_payload {
			if let Some(next) = hook(params_value.clone(), model.clone()).await {
				params_value = next;
			}
		}
		// Streaming is always forced after an onPayload replacement.
		if let Some(object) = params_value.as_object_mut() {
			object.insert("stream".to_string(), Value::Bool(true));
		}

		let request = build_request(
			&model,
			&params_value,
			api_key.as_deref(),
			options.base.headers.as_ref(),
			cache_session_id.as_deref(),
			is_oauth,
			copilot_dynamic_headers.as_deref(),
			options.base.signal.clone(),
			options.base.timeout_ms,
		);

		let fetch: FetchFn = options.base.fetch.clone().unwrap_or_default();
		let response = crate::utils::provider_retry::retry_provider_request(
			|| {
				let request = request.to_request();
				let fetch = fetch.clone();
				async move {
					match fetch.send(request).await {
						Ok(response) if response.ok() => Ok(response),
						Ok(response) => Err(response_to_error(response).await),
						Err(error) => Err(error),
					}
				}
			},
			crate::utils::provider_retry::ProviderRetryOptions {
				max_retries: options.base.max_retries,
				max_retry_delay_ms: options.base.max_retry_delay_ms,
				signal: options.base.signal.clone(),
			},
		)
		.await
		.map_err(|error| error.message)?;

		if let Some(hook) = &options.base.on_response {
			hook(
				ProviderResponse {
					status: response.status,
					headers: headers_to_record(&response.headers),
				},
				model.clone(),
			)
			.await;
		}

		stream.push(AssistantMessageEvent::Start {
			partial: output.clone(),
		});

		// Parallel scratch state for the blocks in `output.content`
		// (upstream stores `index`/`partialJson` directly on the block).
		let mut blocks: Vec<BlockScratch> = Vec::new();
		let mut input_transformations: Option<Value> = None;
		let mut usage_model = model.clone();

		let body = match response.body {
			HttpBody::Stream(body_stream) => body_stream,
			other => Box::pin(futures_util::stream::once(async move { other.collect().await })),
		};
		let mut events = AnthropicEventStream::new(body, options.base.signal.clone());

		let events_result: Result<(), String> = 'events: {
			while let Some(event) = events.next().await {
				let event = match event {
					Ok(event) => event,
					Err(message) => break 'events Err(message),
				};

			if let Some(hook) = &options.on_provider_stream_event {
				hook(
					serde_json::to_value(&event).expect("event serializes"),
					model.clone(),
				)
				.await;
			}

			match event {
				RawMessageStreamEvent::MessageStart { message } => {
					output.response_id = Some(message.id.clone());
					if let Some(transformations) = message.input_transformations {
						if transformations.is_array() {
							input_transformations = Some(transformations);
						}
					}
					let response_model = message.model.clone();
					if response_model.as_deref() != Some(model.id()) {
						output.response_model = response_model.clone();
					}
					let fallback_cost = response_model
						.as_deref()
						.filter(|serving| *serving != model.id())
						.and_then(|serving| {
							model
								.compat
								.as_ref()
								.and_then(|compat| compat.allowed_fallback_models.as_ref())
								.and_then(|fallbacks| {
									fallbacks.iter().find(|fallback| {
										fallback.provider == model.provider() && fallback.model == serving
									})
								})
								.map(|fallback| fallback.cost.clone())
						});
					usage_model = match fallback_cost {
						Some(cost) => {
							let mut fallback_model = model.clone();
							fallback_model.base.id = response_model.expect("checked above");
							fallback_model.base.cost = cost;
							fallback_model
						}
						None => model.clone(),
					};
					// Capture initial token usage from message_start event.
					let usage = &message.usage;
					output.usage.input = usage.input_tokens.unwrap_or(0);
					output.usage.output = usage.output_tokens.unwrap_or(0);
					output.usage.cache_read = usage.cache_read_input_tokens.unwrap_or(0);
					output.usage.cache_write = usage.cache_creation_input_tokens.unwrap_or(0);
					output.usage.cache_write_1h =
						Some(usage.cache_creation.as_ref().and_then(|c| c.ephemeral_1h_input_tokens).unwrap_or(0));
					// Anthropic doesn't provide total_tokens, compute from components
					output.usage.total_tokens =
						output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write;
					calculate_cost(&AnyModel::Chat(usage_model.clone()), &mut output.usage);
				}
				RawMessageStreamEvent::ContentBlockStart {
					index,
					content_block,
				} => {
					match content_block {
						ContentBlockStart::Fallback { .. } => {
							if !output.content.is_empty() {
								break 'events Err(
									"Anthropic performed an unsupported mid-output model fallback".to_string(),
								);
							}
							continue;
						}
						ContentBlockStart::Text { text } => {
							output.content.push(AssistantContent::Text(crate::types::TextContent {
								text: text.unwrap_or_default(),
								text_signature: None,
							}));
							blocks.push(BlockScratch {
								anthropic_index: index,
								partial_json: String::new(),
								stopped: false,
							});
							stream.push(AssistantMessageEvent::TextStart {
								content_index: (output.content.len() - 1) as u32,
								partial: output.clone(),
							});
						}
						ContentBlockStart::Thinking { thinking, signature } => {
							output.content.push(AssistantContent::Thinking(ThinkingContent {
								thinking: thinking.unwrap_or_default(),
								thinking_signature: Some(signature.unwrap_or_default()),
								redacted: None,
							}));
							blocks.push(BlockScratch {
								anthropic_index: index,
								partial_json: String::new(),
								stopped: false,
							});
							stream.push(AssistantMessageEvent::ThinkingStart {
								content_index: (output.content.len() - 1) as u32,
								partial: output.clone(),
							});
						}
						ContentBlockStart::RedactedThinking { data } => {
							output.content.push(AssistantContent::Thinking(ThinkingContent {
								thinking: "[Reasoning redacted]".to_string(),
								thinking_signature: Some(data.unwrap_or_default()),
								redacted: Some(true),
							}));
							blocks.push(BlockScratch {
								anthropic_index: index,
								partial_json: String::new(),
								stopped: false,
							});
							stream.push(AssistantMessageEvent::ThinkingStart {
								content_index: (output.content.len() - 1) as u32,
								partial: output.clone(),
							});
						}
						ContentBlockStart::ToolUse { id, name, input } => {
							let name = name.unwrap_or_default();
							let name = if is_oauth {
								from_claude_code_name(&name, &current_tools)
							} else {
								name
							};
							let arguments = match input {
								Some(Value::Object(map)) => map,
								_ => crate::types::JsonObject::new(),
							};
							output.content.push(AssistantContent::ToolCall(ToolCall {
								id: id.unwrap_or_default(),
								name,
								arguments,
								thought_signature: None,
								namespace: None,
							}));
							blocks.push(BlockScratch {
								anthropic_index: index,
								partial_json: String::new(),
								stopped: false,
							});
							stream.push(AssistantMessageEvent::ToolcallStart {
								content_index: (output.content.len() - 1) as u32,
								partial: output.clone(),
							});
						}
						ContentBlockStart::Other => {
							// Unknown block types are ignored (TS type-assertion semantics).
						}
					}
				}
				RawMessageStreamEvent::ContentBlockDelta { index, delta } => match delta {
					ContentBlockDelta::TextDelta { text } => {
						if let Some(position) = blocks.iter().position(|b| b.anthropic_index == index && !b.stopped) {
							if let Some(AssistantContent::Text(block)) = output.content.get_mut(position) {
								let text = text.unwrap_or_default();
								block.text += &text;
								stream.push(AssistantMessageEvent::TextDelta {
									content_index: position as u32,
									delta: text,
									partial: output.clone(),
								});
							}
						}
					}
					ContentBlockDelta::ThinkingDelta { thinking } => {
						if let Some(position) = blocks.iter().position(|b| b.anthropic_index == index && !b.stopped) {
							if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(position) {
								let thinking = thinking.unwrap_or_default();
								block.thinking += &thinking;
								stream.push(AssistantMessageEvent::ThinkingDelta {
									content_index: position as u32,
									delta: thinking,
									partial: output.clone(),
								});
							}
						}
					}
					ContentBlockDelta::InputJsonDelta { partial_json } => {
						if let Some(position) = blocks.iter().position(|b| b.anthropic_index == index && !b.stopped) {
							if matches!(output.content.get(position), Some(AssistantContent::ToolCall(_))) {
								let delta = partial_json.unwrap_or_default();
								blocks[position].partial_json += &delta;
								let arguments = parse_streaming_json(Some(&blocks[position].partial_json));
								if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(position) {
									block.arguments = arguments
										.as_object()
										.cloned()
										.unwrap_or_default();
								}
								stream.push(AssistantMessageEvent::ToolcallDelta {
									content_index: position as u32,
									delta,
									partial: output.clone(),
								});
							}
						}
					}
					ContentBlockDelta::SignatureDelta { signature } => {
						if let Some(position) = blocks.iter().position(|b| b.anthropic_index == index && !b.stopped) {
							if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(position) {
								let signature = signature.unwrap_or_default();
								let slot = block.thinking_signature.get_or_insert_with(String::new);
								slot.push_str(&signature);
							}
						}
					}
					ContentBlockDelta::Other => {}
				},
				RawMessageStreamEvent::ContentBlockStop { index } => {
					if let Some(position) = blocks
						.iter()
						.position(|b| b.anthropic_index == index && !b.stopped)
					{
						blocks[position].stopped = true;
						let partial_json = blocks[position].partial_json.clone();
						match output.content.get_mut(position) {
							Some(AssistantContent::Text(block)) => {
								stream.push(AssistantMessageEvent::TextEnd {
									content_index: position as u32,
									content: block.text.clone(),
									partial: output.clone(),
								});
							}
							Some(AssistantContent::Thinking(block)) => {
								stream.push(AssistantMessageEvent::ThinkingEnd {
									content_index: position as u32,
									content: block.thinking.clone(),
									partial: output.clone(),
								});
							}
							Some(AssistantContent::ToolCall(block)) => {
								let arguments = parse_streaming_json(Some(&partial_json));
								block.arguments = arguments.as_object().cloned().unwrap_or_default();
								stream.push(AssistantMessageEvent::ToolcallEnd {
									content_index: position as u32,
									tool_call: block.clone(),
									partial: output.clone(),
								});
							}
							None => {}
						}
					}
				}
				RawMessageStreamEvent::MessageDelta {
					delta,
					usage,
					input_transformations: event_transformations,
				} => {
					if let Some(transformations) = event_transformations {
						if transformations.is_array() {
							input_transformations = Some(transformations);
						}
					}
					if let Some(stop_reason) = &delta.stop_reason {
						output.raw_stop_reason = Some(stop_reason.clone());
						match map_stop_reason(stop_reason, delta.stop_details.as_ref()) {
							Ok((stop_reason, error_message)) => {
								output.stop_reason = stop_reason;
								if let Some(error_message) = error_message {
									output.error_message = Some(error_message);
								}
							}
							Err(message) => break 'events Err(message),
						}
					}
					// Only update usage fields if present (not null). Preserves
					// input_tokens from message_start when proxies omit it in
					// message_delta.
					if let Some(usage) = usage {
						if let Some(input_tokens) = usage.input_tokens {
							output.usage.input = input_tokens;
						}
						if let Some(output_tokens) = usage.output_tokens {
							output.usage.output = output_tokens;
						}
						if let Some(cache_read) = usage.cache_read_input_tokens {
							output.usage.cache_read = cache_read;
						}
						if let Some(cache_write) = usage.cache_creation_input_tokens {
							output.usage.cache_write = cache_write;
						}
						// Vercel AI Gateway includes the TTL breakdown in deltas.
						if let Some(cache_write_1h) = usage
							.cache_creation
							.as_ref()
							.and_then(|c| c.ephemeral_1h_input_tokens)
						{
							output.usage.cache_write_1h = Some(cache_write_1h);
						}
						// Anthropic reports reasoning tokens as a subset of output tokens.
						if let Some(thinking_tokens) = usage
							.output_tokens_details
							.as_ref()
							.and_then(|d| d.thinking_tokens)
						{
							output.usage.reasoning = Some(thinking_tokens);
						}
					}
					// Anthropic doesn't provide total_tokens, compute from components
					output.usage.total_tokens =
						output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write;
					calculate_cost(&AnyModel::Chat(usage_model.clone()), &mut output.usage);
				}
				RawMessageStreamEvent::MessageStop {} => {}
			}
			}
			Ok(())
		};
		events_result?;

		if options.base.signal.as_ref().map(|s| s.aborted()).unwrap_or(false) {
			return Err("Request was aborted".to_string());
		}

		if output.stop_reason == StopReason::Pending {
			return Err("Anthropic stream ended without a stop reason".to_string());
		}
		if output.stop_reason == StopReason::Aborted || output.stop_reason == StopReason::Error {
			return Err(output
				.error_message
				.clone()
				.unwrap_or_else(|| "An unknown error occurred".to_string()));
		}

		if let Some(transformations) = &input_transformations {
			if transformations.is_array() {
				let details = json!({
					"transformations": transformations
						.as_array()
						.expect("checked array")
						.iter()
						.map(|transformation| {
							json!({
								"type": transformation.get("type").cloned().filter(|v| !v.is_null()),
								"path": transformation.get("path").cloned().filter(|v| !v.is_null()),
								"reason": transformation.get("reason").cloned().filter(|v| !v.is_null()),
							})
						})
						.collect::<Vec<_>>(),
				});
				append_assistant_message_diagnostic(
					&mut output,
					AssistantMessageDiagnostic {
						diagnostic_type: "anthropic_input_transformations".to_string(),
						timestamp: now_ms(),
						error: None,
						details: details.as_object().cloned(),
					},
				);
			}
		}

		let reason = done_reason(output.stop_reason);
		stream.push(done_event(output.clone(), reason));
		stream.end(Some(output.clone()));
		Ok(())
	}
	.await;

	if let Err(message) = result {
		let aborted = options.base.signal.as_ref().map(|s| s.aborted()).unwrap_or(false);
		output.stop_reason = if aborted { StopReason::Aborted } else { StopReason::Error };
		output.error_message = Some(message);
		let reason = if aborted { StreamErrorReason::Aborted } else { StreamErrorReason::Error };
		stream.push(error_event(output, reason));
		stream.end(None);
	}
}

fn done_reason(stop_reason: StopReason) -> StreamDoneReason {
	match stop_reason {
		StopReason::Stop => StreamDoneReason::Stop,
		StopReason::Length => StreamDoneReason::Length,
		StopReason::ToolUse => StreamDoneReason::ToolUse,
		StopReason::Deferred => StreamDoneReason::Deferred,
		StopReason::Pending | StopReason::Error | StopReason::Aborted => StreamDoneReason::Stop,
	}
}

// ---------------------------------------------------------------------------
// streamSimple()
// ---------------------------------------------------------------------------

/// Map ThinkingLevel to Anthropic effort levels for adaptive thinking
/// (`mapThinkingLevelToEffort`). Effort "max" is available on all
/// adaptive-thinking Claude models, while native "xhigh" is only available on
/// Opus 4.7/4.8, Sonnet 5, and Fable 5.
fn map_thinking_level_to_effort(model: &Model, level: Option<ThinkingLevel>) -> AnthropicEffort {
	if let Some(level) = level {
		if let Some(mapped) = model
			.thinking_level_map
			.as_ref()
			.and_then(|map| map.get(&thinking_level_to_model_level(level)))
			.cloned()
			.flatten()
		{
			if let Some(effort) = AnthropicEffort::parse(&mapped) {
				return effort;
			}
		}
	}

	match level {
		Some(ThinkingLevel::Minimal) | Some(ThinkingLevel::Low) => AnthropicEffort::Low,
		Some(ThinkingLevel::Medium) => AnthropicEffort::Medium,
		Some(ThinkingLevel::High) => AnthropicEffort::High,
		_ => AnthropicEffort::High,
	}
}

/// Upstream `streamSimple` export.
pub fn stream_simple(model: &Model, context: &TranscriptContext, options: &SimpleStreamOptions) -> AssistantMessageEventStream {
	if let Err(message) = assert_request_auth(
		model.provider(),
		options.base.base.api_key.as_deref(),
		options.base.base.headers.as_ref(),
	) {
		return error_stream(model, &message);
	}

	let api_key = options.base.base.api_key.clone();
	let mut base = crate::api::simple_options::build_base_options(model, context, options, api_key.as_deref());
	base.anthropic.tool_choice = options.tool_choice.map(|tool_choice| match tool_choice {
		ToolChoice::Auto => AnthropicToolChoice::BuiltIn(AnthropicBuiltInToolChoice::Auto),
		ToolChoice::None => AnthropicToolChoice::BuiltIn(AnthropicBuiltInToolChoice::None),
	});

	let Some(reasoning) = options.reasoning else {
		base.anthropic.thinking_enabled = Some(false);
		return stream(model, context, &base);
	};

	// For models with adaptive thinking: use an effort level.
	// For older models: use budget-based thinking.
	if force_adaptive_thinking(model) {
		let effort = map_thinking_level_to_effort(model, Some(reasoning));
		base.anthropic.thinking_enabled = Some(true);
		base.anthropic.effort = Some(effort);
		return stream(model, context, &base);
	}

	// Undefined means the caller did not request an output cap; let the helper
	// use the model cap. Do not coerce to 0 here, or the thinking budget would
	// become the entire max_tokens value.
	let adjusted = crate::api::simple_options::adjust_max_tokens_for_thinking(
		base.max_tokens,
		model.max_tokens,
		reasoning,
		options.thinking_budgets.as_ref(),
	);

	let max_tokens = crate::api::simple_options::clamp_max_tokens_to_context(model, context, adjusted.max_tokens);

	base.max_tokens = Some(max_tokens);
	base.anthropic.thinking_enabled = Some(true);
	base.anthropic.thinking_budget_tokens =
		Some(adjusted.thinking_budget.min(max_tokens.saturating_sub(1024)));
	stream(model, context, &base)
}

/// Synchronous setup failure surfaced as a terminated error stream.
fn error_stream(model: &Model, message: &str) -> AssistantMessageEventStream {
	let stream = AssistantMessageEventStream::new();
	let output = AssistantMessage {
		content: Vec::new(),
		api: model.api().to_string(),
		provider: model.provider().to_string(),
		model: model.id().to_string(),
		response_model: None,
		response_id: None,
		provider_thinking_level: None,
		thinking_level: None,
		diagnostics: None,
		usage: Default::default(),
		stop_reason: StopReason::Error,
		deferred: None,
		error_message: Some(message.to_string()),
		raw_stop_reason: None,
		end_turn: None,
		timestamp: now_ms(),
	};
	stream.push(error_event(output, StreamErrorReason::Error));
	stream.end(None);
	stream
}

// ---------------------------------------------------------------------------
// ProviderStreams export
// ---------------------------------------------------------------------------

/// The anthropic-messages API implementation
/// (`export const anthropicMessagesApi` upstream object). No deferred methods
/// are exported upstream, so both presence probes stay `false`.
pub struct AnthropicMessagesApi;

impl ProviderStreams for AnthropicMessagesApi {
	fn stream(&self, model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
		stream(model, context, options)
	}

	fn stream_simple(
		&self,
		model: &Model,
		context: &TranscriptContext,
		options: &SimpleStreamOptions,
	) -> AssistantMessageEventStream {
		stream_simple(model, context, options)
	}
}

pub fn anthropic_messages_api() -> Arc<dyn ProviderStreams> {
	Arc::new(AnthropicMessagesApi)
}

// ---------------------------------------------------------------------------
// Tests (port of test/anthropic-sse-parsing.test.ts)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;

	fn test_model(provider: &str, id: &str) -> Model {
		Model {
			base: BaseModelFields {
				id: id.to_string(),
				name: id.to_string(),
				api: "anthropic-messages".to_string(),
				provider: provider.to_string(),
				base_url: "https://api.anthropic.com".to_string(),
				input: vec![crate::types::ModelInput::Text],
				input_limits: None,
				cost: ModelCost {
					rates: ModelCostRates {
						input: 1.0,
						output: 5.0,
						cache_read: 0.1,
						cache_write: 0.0,
					},
					tiers: None,
				},
				headers: None,
			},
			model_type: None,
			reasoning: true,
			thinking_level_map: None,
			prompt_cache: None,
			context_window: 200_000,
			max_tokens: 8_192,
			sampling_params: None,
			sampling_params_by_thinking_level: None,
			compat: None,
		}
	}

	fn user_context(messages: Vec<Message>) -> Context {
		Context {
			system_prompt: None,
			messages,
			tools: None,
		}
	}

	fn hello_context() -> Context {
		user_context(vec![Message::User(crate::types::UserMessage {
			content: crate::types::UserContentField::Text("Hello".to_string()),
			timestamp: 1,
		})])
	}

	fn normalize(context: &Context) -> TranscriptContext {
		crate::utils::transcript::normalize_context(context)
	}

	fn base_options() -> StreamOptions {
		let mut options = StreamOptions::default();
		options.base.api_key = Some("sk-ant-api03-test".to_string());
		options
	}

	/// Recording FetchFn fake returning a constructed SSE body
	/// (`createSseResponse` + `createFakeAnthropicClient`).
	fn sse_fetch(body: String, recorded: Arc<parking_lot::Mutex<Vec<Value>>>) -> FetchFn {
		let body = Arc::new(body.into_bytes());
		FetchFn::new(move |request: HttpRequest| {
			let request_body = match request.body {
				HttpBody::Bytes(bytes) => String::from_utf8_lossy(&bytes).to_string(),
				_ => String::new(),
			};
			if let Ok(parsed) = serde_json::from_str::<Value>(&request_body) {
				recorded.lock().push(parsed);
			}
			let response = HttpResponse {
				status: 200,
				headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
				body: HttpBody::Bytes((*body).clone()),
			};
			Box::pin(async move { Ok(response) })
		})
	}

	fn sse_body(events: &[(&str, String)]) -> String {
		events
			.iter()
			.map(|(event, data)| format!("event: {event}\ndata: {data}\n"))
			.collect::<Vec<_>>()
			.join("\n")
	}

	fn event_json(value: Value) -> String {
		serde_json::to_string(&value).expect("event serializes")
	}

	fn minimal_anthropic_events() -> Vec<(&'static str, String)> {
		vec![
			(
				"message_start",
				event_json(json!({
					"type": "message_start",
					"message": {
						"id": "msg_test",
						"usage": {
							"input_tokens": 12,
							"output_tokens": 0,
							"cache_read_input_tokens": 0,
							"cache_creation_input_tokens": 0,
						},
					},
				})),
			),
			(
				"content_block_start",
				event_json(json!({
					"type": "content_block_start",
					"index": 0,
					"content_block": { "type": "text", "text": "" },
				})),
			),
			(
				"content_block_delta",
				event_json(json!({
					"type": "content_block_delta",
					"index": 0,
					"delta": { "type": "text_delta", "text": "Hello" },
				})),
			),
			(
				"content_block_stop",
				event_json(json!({ "type": "content_block_stop", "index": 0 })),
			),
			(
				"message_delta",
				event_json(json!({
					"type": "message_delta",
					"delta": { "stop_reason": "end_turn" },
					"usage": {
						"input_tokens": 12,
						"output_tokens": 5,
						"cache_read_input_tokens": 0,
						"cache_creation_input_tokens": 0,
					},
				})),
			),
			("message_stop", event_json(json!({ "type": "message_stop" }))),
		]
	}

	fn response_model_events(model: &str, content_block: Value) -> Vec<(&'static str, String)> {
		vec![
			(
				"message_start",
				event_json(json!({
					"type": "message_start",
					"message": { "id": "msg_response_model", "model": model, "usage": { "input_tokens": 100, "output_tokens": 0 } },
				})),
			),
			(
				"content_block_start",
				event_json(json!({
					"type": "content_block_start",
					"index": 0,
					"content_block": content_block,
				})),
			),
			(
				"content_block_stop",
				event_json(json!({ "type": "content_block_stop", "index": 0 })),
			),
			(
				"message_delta",
				event_json(json!({
					"type": "message_delta",
					"delta": { "stop_reason": "end_turn" },
					"usage": { "input_tokens": 100, "output_tokens": 20 },
				})),
			),
			("message_stop", event_json(json!({ "type": "message_stop" }))),
		]
	}

	#[tokio::test]
	async fn forwards_parsed_provider_stream_events_in_order() {
		let model = test_model("anthropic", "claude-haiku-4-5");
		let provider_events = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		let event_models = Arc::new(parking_lot::Mutex::new(Vec::<(String, String)>::new()));

		let mut options = base_options();
		let hook_events = provider_events.clone();
		let hook_models = event_models.clone();
		let hook_model = model.clone();
		options.on_provider_stream_event = Some(Arc::new(move |event: Value, event_model: Model| {
			let hook_events = hook_events.clone();
			let hook_models = hook_models.clone();
			let hook_model = hook_model.clone();
			Box::pin(async move {
				hook_events.lock().push(event);
				hook_models
					.lock()
					.push((event_model.provider().to_string(), event_model.id().to_string()));
				let _ = &hook_model;
			})
		}));
		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(sse_fetch(sse_body(&minimal_anthropic_events()), recorded));

		let result = stream(&model, &normalize(&hello_context()), &options)
			.into_result()
			.await;

		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let types: Vec<String> = provider_events
			.lock()
			.iter()
			.map(|event| event.get("type").and_then(Value::as_str).unwrap_or_default().to_string())
			.collect();
		assert_eq!(
			types,
			vec![
				"message_start",
				"content_block_start",
				"content_block_delta",
				"content_block_stop",
				"message_delta",
				"message_stop",
			]
		);
		let models = event_models.lock().clone();
		assert!(models.iter().all(|(provider, id)| provider == "anthropic" && id == model.id()));
	}

	#[tokio::test]
	async fn keeps_signed_thinking_replayable_when_a_proxy_relabels_the_model() {
		// Regression test for earendil-works/pi#9188.
		let model = test_model("anthropic", "claude-opus-5");
		let response_model = "kimi-for-coding";
		let initial_context = normalize(&hello_context());
		let mut options = base_options();
		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(sse_fetch(
			sse_body(&response_model_events(
				response_model,
				json!({ "type": "thinking", "thinking": "reasoning", "signature": "signature" }),
			)),
			recorded,
		));

		let first = stream(&model, &initial_context, &options).into_result().await;

		assert_eq!(first.model, model.id());
		assert_eq!(first.response_model.as_deref(), Some(response_model));

		let mut messages = initial_context.clone().into_messages();
		messages.push(Message::Assistant(first));
		let transformed = crate::api::transform_messages::transform_messages(&messages, &model, None);
		let replayed_assistant = transformed
			.iter()
			.find(|message| message.role() == "assistant")
			.and_then(|m| match m {
				Message::Assistant(a) => Some(a),
				_ => None,
			})
			.expect("assistant message present");
		let content = serde_json::to_value(&replayed_assistant.content).expect("content serializes");
		assert_eq!(
			content,
			json!([{ "type": "thinking", "thinking": "reasoning", "thinkingSignature": "signature" }])
		);
	}

	#[tokio::test]
	async fn uses_a_returned_fallback_model_for_cost_attribution() {
		let fallback_model = "fallback-model";
		let mut model = test_model("anthropic", "claude-opus-5");
		let mut compat = ModelCompat::default();
		compat.allowed_fallback_models = Some(vec![crate::types::AnthropicAllowedFallbackModel {
			provider: "anthropic".to_string(),
			model: fallback_model.to_string(),
			cost: ModelCost {
				rates: ModelCostRates {
					input: 3.0,
					output: 5.0,
					cache_read: 0.0,
					cache_write: 0.0,
				},
				tiers: None,
			},
		}]);
		model.compat = Some(compat);

		let mut options = base_options();
		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(sse_fetch(
			sse_body(&response_model_events(fallback_model, json!({ "type": "text", "text": "done" }))),
			recorded,
		));

		let result = stream(&model, &normalize(&hello_context()), &options)
			.into_result()
			.await;

		assert_eq!(result.model, model.id());
		assert_eq!(result.response_model.as_deref(), Some(fallback_model));
		assert!((result.usage.cost.input - 0.0003).abs() < 1e-10);
		assert!((result.usage.cost.output - 0.0001).abs() < 1e-10);
	}

	#[tokio::test]
	async fn fails_safely_when_anthropic_falls_back_after_output_begins() {
		let model = test_model("anthropic", "claude-opus-5");
		let events = vec![
			(
				"message_start",
				event_json(json!({
					"type": "message_start",
					"message": {
						"id": "msg_fallback",
						"model": "claude-opus-5",
						"usage": { "input_tokens": 1, "output_tokens": 0 },
					},
				})),
			),
			(
				"content_block_start",
				event_json(json!({
					"type": "content_block_start",
					"index": 0,
					"content_block": { "type": "text", "text": "partial" },
				})),
			),
			(
				"content_block_stop",
				event_json(json!({ "type": "content_block_stop", "index": 0 })),
			),
			(
				"content_block_start",
				event_json(json!({
					"type": "content_block_start",
					"index": 1,
					"content_block": {
						"type": "fallback",
						"from": { "model": "claude-opus-5" },
						"to": { "model": "claude-opus-4-8" },
					},
				})),
			),
		];

		let mut options = base_options();
		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(sse_fetch(sse_body(&events), recorded));

		let result = stream(&model, &normalize(&hello_context()), &options)
			.into_result()
			.await;

		assert_eq!(result.stop_reason, StopReason::Error);
		let error_message = result.error_message.expect("error message present");
		assert!(error_message.contains("unsupported mid-output model fallback"), "{error_message}");
	}

	#[tokio::test]
	async fn forces_streaming_after_an_on_payload_replacement() {
		let model = test_model("anthropic", "claude-fable-5-1");
		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		let mut options = base_options();
		options.base.fetch = Some(sse_fetch(sse_body(&minimal_anthropic_events()), recorded.clone()));
		options.base.on_payload = Some(Arc::new(|mut payload: Value, _model: Model| {
			Box::pin(async move {
				if let Some(object) = payload.as_object_mut() {
					object.insert("stream".to_string(), Value::Bool(false));
				}
				Some(payload)
			})
		}));

		stream(&model, &normalize(&hello_context()), &options)
			.into_result()
			.await;

		let requests = recorded.lock();
		assert_eq!(requests.len(), 1);
		assert_eq!(requests[0].get("stream"), Some(&Value::Bool(true)));
	}

	#[tokio::test]
	async fn omits_the_interleaved_thinking_beta_when_thinking_is_disabled() {
		let model = {
			let mut model = test_model("openrouter", "anthropic/claude-haiku-4.5");
			model.base.base_url = "https://openrouter.ai/api/v1".to_string();
			model
		};
		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		let mut options = base_options();
		options.base.fetch = Some(sse_fetch(sse_body(&minimal_anthropic_events()), recorded.clone()));
		options.anthropic.thinking_enabled = Some(false);

		stream(&model, &normalize(&hello_context()), &options)
			.into_result()
			.await;

		let requests = recorded.lock();
		assert_eq!(requests.len(), 1);
		let betas = requests[0]
			.get("betas")
			.and_then(Value::as_array)
			.cloned()
			.unwrap_or_default();
		assert!(!betas.contains(&json!("interleaved-thinking-2025-05-14")), "{betas:?}");
	}

	#[tokio::test]
	async fn passes_managed_beta_features_to_injected_clients() {
		let mut model = test_model("anthropic", "claude-fable-5-1");
		let mut compat = ModelCompat::default();
		compat.supports_mid_convo_effort = Some(true);
		model.compat = Some(compat);

		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		let mut options = base_options();
		options.base.fetch = Some(sse_fetch(sse_body(&minimal_anthropic_events()), recorded.clone()));

		let result = stream(&model, &normalize(&hello_context()), &options)
			.into_result()
			.await;

		assert_eq!(result.stop_reason, StopReason::Stop);
		let requests = recorded.lock();
		assert_eq!(requests.len(), 1);
		let betas = requests[0]
			.get("betas")
			.and_then(Value::as_array)
			.cloned()
			.unwrap_or_default();
		assert!(betas.contains(&json!("mid-conversation-output-config-2026-07-01")), "{betas:?}");
		assert!(betas.contains(&json!("thinking-binding-controls-2026-08-01")), "{betas:?}");
	}

	#[tokio::test]
	async fn uses_the_serving_model_input_transformations_from_the_final_stream_event() {
		let mut model = test_model("anthropic", "claude-fable-5-1");
		let mut compat = ModelCompat::default();
		compat.supports_mid_convo_effort = Some(true);
		model.compat = Some(compat);

		let mut events = minimal_anthropic_events();
		events[0].1 = event_json(json!({
			"type": "message_start",
			"message": {
				"id": "msg_transformations",
				"model": "claude-fable-5-1",
				"usage": { "input_tokens": 12, "output_tokens": 0 },
				"input_transformations": [
					{ "type": "thinking_dropped", "path": "messages.1.content.0", "reason": "prefix_binding_mismatch" },
				],
			},
		}));
		let mut delta =
			serde_json::from_str::<Value>(&events[4].1).expect("delta parses");
		delta["input_transformations"] = json!([
			{ "type": "thinking_dropped", "path": "messages.3.content.0", "reason": "model_binding_mismatch" },
		]);
		events[4].1 = event_json(delta);

		let mut options = base_options();
		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(sse_fetch(sse_body(&events), recorded));

		let result = stream(&model, &normalize(&hello_context()), &options)
			.into_result()
			.await;

		let diagnostics = serde_json::to_value(result.diagnostics).expect("diagnostics serialize");
		assert_eq!(
			diagnostics,
			json!([
				{
					"type": "anthropic_input_transformations",
					"timestamp": diagnostics[0].get("timestamp").cloned().expect("timestamp present"),
					"details": {
						"transformations": [
							{
								"type": "thinking_dropped",
								"path": "messages.3.content.0",
								"reason": "model_binding_mismatch",
							},
						],
					},
				},
			])
		);
	}

	#[tokio::test]
	async fn repairs_malformed_sse_json_and_malformed_streamed_tool_json() {
		let model = test_model("anthropic", "claude-haiku-4-5");
		let mut context = hello_context();
		context.tools = Some(vec![Tool {
			name: "edit".to_string(),
			description: "Edit a file.".to_string(),
			parameters: json!({
				"type": "object",
				"properties": {
					"path": { "type": "string" },
					"text": { "type": "string" },
				},
				"required": ["path", "text"],
			}),
			constrained_sampling: None,
		}]);

		let malformed_tool_json_delta = concat!(
			r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","#,
			r#""partial_json":"{\"path\":\"A\H\",\"text\":\"col1	col2\"}""#,
			r#"}}"#
		);

		let events = vec![
			(
				"message_start",
				event_json(json!({
					"type": "message_start",
					"message": {
						"id": "msg_test",
						"usage": {
							"input_tokens": 12,
							"output_tokens": 0,
							"cache_read_input_tokens": 0,
							"cache_creation_input_tokens": 0,
						},
					},
				})),
			),
			(
				"content_block_start",
				event_json(json!({
					"type": "content_block_start",
					"index": 0,
					"content_block": {
						"type": "tool_use",
						"id": "toolu_test",
						"name": "edit",
						"input": {},
					},
				})),
			),
			("content_block_delta", malformed_tool_json_delta.to_string()),
			(
				"content_block_stop",
				event_json(json!({ "type": "content_block_stop", "index": 0 })),
			),
			(
				"message_delta",
				event_json(json!({
					"type": "message_delta",
					"delta": { "stop_reason": "tool_use" },
					"usage": {
						"input_tokens": 12,
						"output_tokens": 5,
						"cache_read_input_tokens": 0,
						"cache_creation_input_tokens": 0,
					},
				})),
			),
			("message_stop", event_json(json!({ "type": "message_stop" }))),
		];

		let mut options = base_options();
		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(sse_fetch(sse_body(&events), recorded));

		let result = stream(&model, &normalize(&context), &options).into_result().await;

		assert_eq!(result.stop_reason, StopReason::ToolUse);
		assert_eq!(result.error_message, None);

		let tool_call = result
			.content
			.iter()
			.find_map(|block| match block {
				AssistantContent::ToolCall(call) => Some(call),
				_ => None,
			})
			.expect("tool call present");
		assert_eq!(
			serde_json::to_value(&tool_call.arguments).expect("arguments serialize"),
			json!({
				"path": "A\\H",
				"text": "col1\tcol2",
			})
		);
	}

	#[tokio::test]
	async fn preserves_content_from_content_block_start_events() {
		let model = test_model("anthropic", "claude-haiku-4-5");
		let events = vec![
			(
				"message_start",
				event_json(json!({
					"type": "message_start",
					"message": {
						"id": "msg_initial_content",
						"usage": {
							"input_tokens": 12,
							"output_tokens": 0,
							"cache_read_input_tokens": 0,
							"cache_creation_input_tokens": 0,
						},
					},
				})),
			),
			(
				"content_block_start",
				event_json(json!({
					"type": "content_block_start",
					"index": 0,
					"content_block": { "type": "text", "text": "Initial text" },
				})),
			),
			(
				"content_block_delta",
				event_json(json!({
					"type": "content_block_delta",
					"index": 0,
					"delta": { "type": "text_delta", "text": " plus delta" },
				})),
			),
			(
				"content_block_stop",
				event_json(json!({ "type": "content_block_stop", "index": 0 })),
			),
			(
				"content_block_start",
				event_json(json!({
					"type": "content_block_start",
					"index": 1,
					"content_block": {
						"type": "thinking",
						"thinking": "Initial thinking",
						"signature": "initial signature",
					},
				})),
			),
			(
				"content_block_delta",
				event_json(json!({
					"type": "content_block_delta",
					"index": 1,
					"delta": { "type": "thinking_delta", "thinking": " plus delta" },
				})),
			),
			(
				"content_block_delta",
				event_json(json!({
					"type": "content_block_delta",
					"index": 1,
					"delta": { "type": "signature_delta", "signature": " plus delta" },
				})),
			),
			(
				"content_block_stop",
				event_json(json!({ "type": "content_block_stop", "index": 1 })),
			),
			(
				"message_delta",
				event_json(json!({
					"type": "message_delta",
					"delta": { "stop_reason": "end_turn" },
					"usage": {
						"input_tokens": 12,
						"output_tokens": 5,
						"cache_read_input_tokens": 0,
						"cache_creation_input_tokens": 0,
					},
				})),
			),
			("message_stop", event_json(json!({ "type": "message_stop" }))),
		];

		let mut options = base_options();
		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(sse_fetch(sse_body(&events), recorded));

		let result = stream(&model, &normalize(&hello_context()), &options)
			.into_result()
			.await;

		let content = serde_json::to_value(&result.content).expect("content serializes");
		assert_eq!(
			content,
			json!([
				{ "type": "text", "text": "Initial text plus delta" },
				{
					"type": "thinking",
					"thinking": "Initial thinking plus delta",
					"thinkingSignature": "initial signature plus delta",
				},
			])
		);
	}

	#[tokio::test]
	async fn preserves_refusal_stop_details_from_message_delta() {
		let model = test_model("anthropic", "claude-fable-5");
		let context = normalize(&user_context(vec![Message::User(crate::types::UserMessage {
			content: crate::types::UserContentField::Text("blocked request".to_string()),
			timestamp: now_ms(),
		})]));
		let explanation = "This request triggered restrictions on violative cyber content and was blocked under Anthropic's Usage Policy. To learn more, provide feedback, or request an exemption based on how you use Claude, visit our help center: https://support.claude.com/en/articles/14604842-real-time-cyber-safeguards-on-claude.";
		let events = vec![
			(
				"message_start",
				event_json(json!({
					"type": "message_start",
					"message": {
						"id": "msg_01XFUDYJgAACzvnptvVoYEL",
						"usage": {
							"input_tokens": 412,
							"output_tokens": 0,
							"cache_read_input_tokens": 0,
							"cache_creation_input_tokens": 0,
						},
					},
				})),
			),
			(
				"message_delta",
				event_json(json!({
					"type": "message_delta",
					"delta": {
						"stop_reason": "refusal",
						"stop_details": {
							"type": "refusal",
							"category": "cyber",
							"explanation": explanation,
						},
					},
					"usage": {
						"input_tokens": 412,
						"output_tokens": 0,
						"cache_read_input_tokens": 0,
						"cache_creation_input_tokens": 0,
					},
				})),
			),
			("message_stop", event_json(json!({ "type": "message_stop" }))),
		];

		let mut options = base_options();
		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(sse_fetch(sse_body(&events), recorded));

		let result = stream(&model, &context, &options).into_result().await;

		assert_eq!(result.stop_reason, StopReason::Error);
		assert_eq!(result.raw_stop_reason.as_deref(), Some("refusal"));
		assert_eq!(result.error_message.as_deref(), Some(explanation));
	}

	#[tokio::test]
	async fn preserves_sensitive_stop_reasons_with_a_descriptive_error_message() {
		let model = test_model("anthropic", "claude-haiku-4-5");
		let context = normalize(&user_context(vec![Message::User(crate::types::UserMessage {
			content: crate::types::UserContentField::Text("blocked request".to_string()),
			timestamp: now_ms(),
		})]));
		let events = vec![
			(
				"message_start",
				event_json(json!({
					"type": "message_start",
					"message": {
						"id": "msg_sensitive",
						"usage": {
							"input_tokens": 12,
							"output_tokens": 0,
							"cache_read_input_tokens": 0,
							"cache_creation_input_tokens": 0,
						},
					},
				})),
			),
			(
				"message_delta",
				event_json(json!({
					"type": "message_delta",
					"delta": { "stop_reason": "sensitive" },
					"usage": {
						"input_tokens": 12,
						"output_tokens": 0,
						"cache_read_input_tokens": 0,
						"cache_creation_input_tokens": 0,
					},
				})),
			),
			("message_stop", event_json(json!({ "type": "message_stop" }))),
		];

		let mut options = base_options();
		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(sse_fetch(sse_body(&events), recorded));

		let result = stream(&model, &context, &options).into_result().await;

		assert_eq!(result.stop_reason, StopReason::Error);
		assert_eq!(result.raw_stop_reason.as_deref(), Some("sensitive"));
		assert_eq!(result.error_message.as_deref(), Some("Provider stopped with: sensitive"));
	}

	#[tokio::test]
	async fn treats_message_delta_without_usage_as_a_no_op_for_usage_accumulation() {
		let model = test_model("anthropic", "claude-haiku-4-5");
		let events: Vec<(&'static str, String)> = minimal_anthropic_events()
			.into_iter()
			.map(|(event, data)| {
				if event == "message_delta" {
					(
						event,
						event_json(json!({
							"type": "message_delta",
							"delta": { "stop_reason": "end_turn" },
						})),
					)
				} else {
					(event, data)
				}
			})
			.collect();

		let mut options = base_options();
		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(sse_fetch(sse_body(&events), recorded));

		let result = stream(&model, &normalize(&hello_context()), &options)
			.into_result()
			.await;

		assert_eq!(result.stop_reason, StopReason::Stop);
		assert_eq!(result.error_message, None);
		let content = serde_json::to_value(&result.content).expect("content serializes");
		assert_eq!(content, json!([{ "type": "text", "text": "Hello" }]));
		assert_eq!(result.usage.input, 12);
		assert_eq!(result.usage.total_tokens, 12);
	}

	#[tokio::test]
	async fn ignores_unknown_sse_events_after_message_stop() {
		let model = test_model("anthropic", "claude-haiku-4-5");
		let mut events = minimal_anthropic_events();
		events.push(("done", "[DONE]".to_string()));
		events.push(("proxy.stats", "not json".to_string()));

		let mut options = base_options();
		let recorded = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(sse_fetch(sse_body(&events), recorded));

		let result = stream(&model, &normalize(&hello_context()), &options)
			.into_result()
			.await;

		assert_eq!(result.stop_reason, StopReason::Stop);
		assert_eq!(result.error_message, None);
		let content = serde_json::to_value(&result.content).expect("content serializes");
		assert_eq!(content, json!([{ "type": "text", "text": "Hello" }]));
	}

	// ---- Unit tests for the ported SSE decoder -----------------------------

	#[test]
	fn sse_decoder_handles_crlf_comments_and_multiline_data() {
		let mut state = SseDecoderState::default();
		assert_eq!(decode_sse_line(": comment", &mut state), None);
		assert_eq!(decode_sse_line("event: message_start", &mut state), None);
		assert_eq!(decode_sse_line("data: line1", &mut state), None);
		assert_eq!(decode_sse_line("data:line2", &mut state), None);
		let event = decode_sse_line("", &mut state).expect("event flushed");
		assert_eq!(event.event.as_deref(), Some("message_start"));
		assert_eq!(event.data, "line1\nline2");
		assert_eq!(event.raw, vec![": comment", "event: message_start", "data: line1", "data:line2"]);
	}

	#[test]
	fn sse_decoder_consumes_crlf_line_breaks() {
		let text = "event: a\r\ndata: 1\r\n\r\ndata: tail";
		let first = consume_line(text).expect("line");
		assert_eq!(first.0, "event: a");
		let second = consume_line(&first.1).expect("line");
		assert_eq!(second.0, "data: 1");
		let third = consume_line(&second.1).expect("blank line");
		assert_eq!(third.0, "");
		assert_eq!(consume_line(&third.1), None);
	}
}
