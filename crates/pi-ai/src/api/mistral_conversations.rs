//! Port of `packages/ai/src/api/mistral-conversations.ts` — the native Mistral
//! Chat Completions streaming API.
//!
//! Upstream is hand-rolled over `fetch` (the task mention of the
//! `@mistralai/sdk` applies to older revisions; v1.0.3 builds the request and
//! decodes the SSE body itself), so this port mirrors it 1:1 over the
//! [`crate::http::FetchFn`] seam: `POST {baseUrl}/v1/chat/completions`,
//! `Authorization: Bearer`, `accept: text/event-stream`,
//! [`crate::utils::pi_user_agent::get_pi_user_agent`] user agent, optional
//! `x-affinity` session header, and the upstream SSE event framing
//! (`data:` lines, `[DONE]` sentinel, permissive line terminators). The
//! internal request payload is camelCase ("SDK-style"); `toMistralWirePayload`
//! remaps it to the Mistral snake_case wire format exactly as upstream.
//!
//! # Deviations (per-file ledger)
//!
//! - **D15 — per-API options cannot ride `StreamOptions`** (`types.rs` is
//!   frozen; deviation follows crate D2 pattern): upstream `MistralOptions`
//!   extends `StreamOptions`, so the extra fields (`toolChoice`, `promptMode`,
//!   `reasoningEffort`) live in [`MistralOptions`] here. `stream` builds the
//!   group `Default` and delegates to [`stream_with_options`]; `streamSimple`
//!   maps the simple fields and delegates to [`stream_simple_with_options`].
//!   Tests call `*_with_options`; final integration should add the nested
//!   `StreamOptions` group (reported in reexports).
//! - **D16 — no client-side retry.** Upstream mistral-conversations issues
//!   the request once (`options.fetch ?? globalThis.fetch`, no
//!   `retryProviderRequest` wiring), so the port does the same; transport
//!   failures surface as stream errors.
//! - **D17 — `AbortSignal.timeout` re-implemented locally**
//!   ([`timeout_signal`]): the pure-Rust signal (crate deviation D3) has no
//!   builtin timer, so a sleep task aborts a fresh signal with the DOM
//!   `TimeoutError` message; combined with the caller signal via
//!   `AbortSignal::any`.
//! - **D18 — streaming UTF-8 decoding** ([`MistralEventStream`]): bytes are
//!   buffered and only complete UTF-8 prefixes are decoded (upstream
//!   `TextDecoder(..., {stream: true})`); the mistral tests split multibyte
//!   sequences across transport chunks, so per-chunk lossy decoding (see
//!   anthropic_messages D19) would corrupt them.
//! - **D19 — malformed input handling.** A choice without `delta` is treated
//!   as an absent delta and events without a `choices` array are skipped
//!   (upstream would throw reading `delta.content` of undefined /
//!   `chunk.choices[0]`), non-string/non-array `delta.content` is ignored, and
//!   malformed SSE JSON surfaces the `serde_json` message instead of a JS
//!   `SyntaxError` message. All reach the same error-event path.
//! - **D20 — `statusText` is unavailable** on the Rust
//!   [`crate::http::HttpResponse`] seam; `MistralHttpError`'s message is
//!   always `Request failed with status <N>` (upstream prefers the reason
//!   phrase when present). Error bodies still travel verbatim.
//! - **D21 — `onPayload` replacement carries the payload as a JSON `Value`**
//!   (upstream casts it back to `MistralChatPayload`, trusting the shape), so
//!   extra fields a hook adds (e.g. `safePrompt`) survive into the wire
//!   payload, matching the upstream index-signature behavior.
//! - **D22 — `stripSymbolKeys` is the identity**: TypeBox symbol keys do not
//!   exist in the Rust JSON world (crate deviation D6), so tool parameters
//!   are passed through untouched.
//! - **D23 — mapped reasoning efforts are validated**: `streamSimple` maps
//!   `thinkingLevelMap` strings through [`MistralReasoningEffort::parse`],
//!   falling back to `high` (upstream casts arbitrary strings into the
//!   `MistralReasoningEffort` type; every catalog map only produces valid
//!   efforts, so behavior is identical for real models).

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::http::{BodyStream, FetchFn, HttpBody, HttpRequest};
use crate::models::{calculate_cost, clamp_thinking_level, ProviderStreams};
use crate::types::{
	AnyModel, AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream, CacheRetention,
	Message, Model, ModelCompat, ModelInput, ModelThinkingLevel, ProviderResponse, SimpleStreamOptions, StopReason,
	StreamDoneReason, StreamErrorReason, StreamOptions, TextContent, ThinkingContent, Tool, ToolCall, ToolChoice,
	TranscriptContext, UserContent, UserContentField,
};
use crate::utils::abort::{AbortReason, AbortSignal};
use crate::utils::diagnostics::now_ms;
use crate::utils::event_stream::{done_event, error_event};
use crate::utils::hash::short_hash;
use crate::utils::headers::headers_to_record;
use crate::utils::json_parse::parse_streaming_json;
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::text::{get_system_message_text, render_system_message_update};
use crate::utils::transcript::{get_current_tools, resolve_transcript};

const MISTRAL_TOOL_CALL_ID_LENGTH: usize = 9;
const MAX_MISTRAL_ERROR_BODY_CHARS: usize = 4000;

/// `AbortSignal.timeout` reason message (DOM `TimeoutError`).
const TIMEOUT_MESSAGE: &str = "The operation was aborted due to timeout";

// ---------------------------------------------------------------------------
// Options (deviation D15)
// ---------------------------------------------------------------------------

/// Upstream `MistralReasoningEffort`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MistralReasoningEffort {
	None,
	Low,
	Medium,
	High,
	Max,
}

impl MistralReasoningEffort {
	/// Parse a wire-level effort string (`"none" | "low" | "medium" | "high" | "max"`).
	pub fn parse(value: &str) -> Option<MistralReasoningEffort> {
		serde_json::from_value::<MistralReasoningEffort>(Value::String(value.to_string())).ok()
	}

	pub fn as_str(self) -> &'static str {
		match self {
			MistralReasoningEffort::None => "none",
			MistralReasoningEffort::Low => "low",
			MistralReasoningEffort::Medium => "medium",
			MistralReasoningEffort::High => "high",
			MistralReasoningEffort::Max => "max",
		}
	}
}

/// Upstream `MistralOptions["promptMode"]` (`"reasoning"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MistralPromptMode {
	Reasoning,
}

/// Built-in `toolChoice` strings (`"auto" | "none" | "any" | "required"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MistralBuiltinToolChoice {
	Auto,
	None,
	Any,
	Required,
}

/// The `type: "function"` discriminator of the forced-tool variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MistralFunctionTag {
	#[serde(rename = "function")]
	Function,
}

/// `function: { name }` of the forced-tool `toolChoice`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MistralNamedFunction {
	pub name: String,
}

/// Upstream `MistralOptions["toolChoice"]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MistralToolChoiceOption {
	BuiltIn(MistralBuiltinToolChoice),
	Function {
		#[serde(rename = "type")]
		tag: MistralFunctionTag,
		function: MistralNamedFunction,
	},
}

/// Upstream `MistralOptions` (deviation D15: base `StreamOptions` fields stay
/// on `StreamOptions`).
#[derive(Clone, Default)]
pub struct MistralOptions {
	/// `toolChoice`.
	pub tool_choice: Option<MistralToolChoiceOption>,
	/// `promptMode` (`"reasoning"`).
	pub prompt_mode: Option<MistralPromptMode>,
	/// `reasoningEffort`.
	pub reasoning_effort: Option<MistralReasoningEffort>,
}

// ---------------------------------------------------------------------------
// Request payload types ("SDK-style" camelCase, serialized to JSON)
// ---------------------------------------------------------------------------

/// `thinking: [{ type: "text", text }]` part of a thinking content chunk.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct MistralThinkingPart {
	#[serde(rename = "type")]
	tag: &'static str,
	text: String,
}

impl MistralThinkingPart {
	fn text(text: impl Into<String>) -> MistralThinkingPart {
		MistralThinkingPart { tag: "text", text: text.into() }
	}
}

/// `MistralContentChunk`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
enum MistralContentChunk {
	#[serde(rename = "text")]
	Text { text: String },
	#[serde(rename = "image_url")]
	ImageUrl {
		#[serde(rename = "image_url")]
		image_url: String,
	},
	#[serde(rename = "thinking")]
	Thinking { thinking: Vec<MistralThinkingPart> },
}

/// `MistralRequestToolCall`.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct MistralRequestToolCall {
	id: String,
	#[serde(rename = "type")]
	tag: &'static str,
	function: MistralRequestFunction,
	index: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct MistralRequestFunction {
	name: String,
	arguments: String,
}

/// `string | MistralContentChunk[]` content of [`MistralChatMessage`].
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
enum MistralMessageContent {
	Text(String),
	Chunks(Vec<MistralContentChunk>),
}

/// `MistralChatMessage`.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct MistralChatMessage {
	role: &'static str,
	#[serde(skip_serializing_if = "Option::is_none")]
	content: Option<MistralMessageContent>,
	#[serde(rename = "toolCalls", skip_serializing_if = "Option::is_none")]
	tool_calls: Option<Vec<MistralRequestToolCall>>,
	#[serde(rename = "toolCallId", skip_serializing_if = "Option::is_none")]
	tool_call_id: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	name: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	prefix: Option<bool>,
}

impl MistralChatMessage {
	fn new(role: &'static str) -> MistralChatMessage {
		MistralChatMessage { role, content: None, tool_calls: None, tool_call_id: None, name: None, prefix: None }
	}
}

/// `MistralFunctionTool`.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct MistralFunctionTool {
	#[serde(rename = "type")]
	tag: &'static str,
	function: MistralFunctionDef,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct MistralFunctionDef {
	name: String,
	description: String,
	parameters: Value,
	strict: bool,
}

/// `MistralChatPayload`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct MistralChatPayload {
	model: String,
	stream: bool,
	messages: Vec<MistralChatMessage>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tools: Option<Vec<MistralFunctionTool>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	temperature: Option<f64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	max_tokens: Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_choice: Option<MistralToolChoiceOption>,
	#[serde(skip_serializing_if = "Option::is_none")]
	prompt_mode: Option<MistralPromptMode>,
	#[serde(skip_serializing_if = "Option::is_none")]
	reasoning_effort: Option<MistralReasoningEffort>,
	#[serde(skip_serializing_if = "Option::is_none")]
	prompt_cache_key: Option<String>,
}

// ---------------------------------------------------------------------------
// Errors (`MistralHttpError`, `formatMistralError`)
// ---------------------------------------------------------------------------

/// Upstream `MistralHttpError`.
#[derive(Debug, Clone, PartialEq)]
struct MistralHttpError {
	status_code: u16,
	body: String,
	message: String,
}

/// Values thrown inside the stream task (upstream `Error` instances and abort
/// reasons; deviation D19 for the message text of parse failures).
#[derive(Debug, Clone, PartialEq)]
enum MistralStreamError {
	Http(MistralHttpError),
	Message(String),
}

/// Upstream `formatMistralError`.
fn format_mistral_error(error: &MistralStreamError) -> String {
	match error {
		MistralStreamError::Http(http_error) => {
			let body_text = http_error.body.trim();
			if !body_text.is_empty() {
				return format!(
					"Mistral API error ({}): {}",
					http_error.status_code,
					truncate_error_text(body_text, MAX_MISTRAL_ERROR_BODY_CHARS)
				);
			}
			format!("Mistral API error ({}): {}", http_error.status_code, http_error.message)
		}
		MistralStreamError::Message(message) => message.clone(),
	}
}

/// Upstream `truncateErrorText`.
fn truncate_error_text(text: &str, max_chars: usize) -> String {
	let length = text.chars().count();
	if length <= max_chars {
		return text.to_string();
	}
	let prefix: String = text.chars().take(max_chars).collect();
	format!("{prefix}... [truncated {} chars]", length - max_chars)
}

/// Upstream `createMistralToolCallIdNormalizer` (state in `RefCell` so the
/// `transform_messages` callback can borrow it immutably).
#[derive(Debug, Default)]
struct MistralToolCallIdNormalizer {
	id_map: RefCell<HashMap<String, String>>,
	reverse_map: RefCell<HashMap<String, String>>,
}

impl MistralToolCallIdNormalizer {
	fn normalize(&self, id: &str) -> String {
		if let Some(existing) = self.id_map.borrow().get(id) {
			return existing.clone();
		}
		let mut attempt = 0u32;
		loop {
			let candidate = derive_mistral_tool_call_id(id, attempt);
			let owner = self.reverse_map.borrow().get(&candidate).cloned();
			if owner.map_or(true, |owner| owner == id) {
				self.id_map.borrow_mut().insert(id.to_string(), candidate.clone());
				self.reverse_map.borrow_mut().insert(candidate.clone(), id.to_string());
				return candidate;
			}
			attempt += 1;
		}
	}
}

/// Upstream `deriveMistralToolCallId`.
fn derive_mistral_tool_call_id(id: &str, attempt: u32) -> String {
	let normalized: String = id.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
	if attempt == 0 && normalized.len() == MISTRAL_TOOL_CALL_ID_LENGTH {
		return normalized;
	}
	let seed_base = if normalized.is_empty() { id.to_string() } else { normalized };
	let seed = if attempt == 0 { seed_base } else { format!("{seed_base}:{attempt}") };
	short_hash(&seed)
		.chars()
		.filter(|c| c.is_ascii_alphanumeric())
		.take(MISTRAL_TOOL_CALL_ID_LENGTH)
		.collect()
}

// ---------------------------------------------------------------------------
// Request assembly (`buildMistralHeaders`, `requestMistralStream`, wire payload)
// ---------------------------------------------------------------------------

fn set_header(headers: &mut Vec<(String, String)>, name: &str, value: String) {
	if let Some(slot) = headers.iter_mut().find(|(key, _)| key.eq_ignore_ascii_case(name)) {
		slot.1 = value;
	} else {
		headers.push((name.to_string(), value));
	}
}

fn delete_header(headers: &mut Vec<(String, String)>, name: &str) {
	headers.retain(|(key, _)| !key.eq_ignore_ascii_case(name));
}

fn has_header_override(overrides: Option<&crate::types::ProviderHeaders>, target: &str) -> bool {
	overrides.is_some_and(|map| map.keys().any(|name| name.eq_ignore_ascii_case(target)))
}

/// `hasMistralHeaderOverride(model.headers, ...)` — model headers carry plain
/// string values.
fn has_model_header_override(overrides: Option<&std::collections::BTreeMap<String, String>>, target: &str) -> bool {
	overrides.is_some_and(|map| map.keys().any(|name| name.eq_ignore_ascii_case(target)))
}

/// `shouldUsePromptCaching`.
fn should_use_prompt_caching(options: &StreamOptions) -> bool {
	options.cache_retention != Some(CacheRetention::None) && options.session_id.is_some()
}

/// Upstream `buildMistralHeaders`.
fn build_mistral_headers(model: &Model, api_key: &str, options: &StreamOptions) -> Vec<(String, String)> {
	let mut headers: Vec<(String, String)> = Vec::new();
	set_header(&mut headers, "User-Agent", get_pi_user_agent());
	set_header(&mut headers, "accept", "text/event-stream".to_string());
	set_header(&mut headers, "authorization", format!("Bearer {api_key}"));
	set_header(&mut headers, "content-type", "application/json".to_string());

	// `applyMistralHeaderOverrides(headers, model.headers)`.
	if let Some(model_headers) = model.base.headers.as_ref() {
		for (name, value) in model_headers {
			set_header(&mut headers, name, value.clone());
		}
	}
	// `applyMistralHeaderOverrides(headers, options.headers)`.
	if let Some(overrides) = options.base.headers.as_ref() {
		for (name, value) in overrides {
			match value {
				Some(value) => set_header(&mut headers, name, value.clone()),
				None => delete_header(&mut headers, name),
			}
		}
	}

	let has_explicit_affinity = has_model_header_override(model.base.headers.as_ref(), "x-affinity")
		|| has_header_override(options.base.headers.as_ref(), "x-affinity");
	if should_use_prompt_caching(options) && !has_explicit_affinity {
		if let Some(session_id) = options.session_id.as_deref() {
			set_header(&mut headers, "x-affinity", session_id.to_string());
		}
	}

	headers
}

/// Move `record[source]` to `record[target]` (`remapMistralProperty`).
fn remap_property(record: &mut serde_json::Map<String, Value>, source: &str, target: &str) {
	if let Some(value) = record.remove(source) {
		record.insert(target.to_string(), value);
	}
}

/// Upstream `toMistralWirePayload` (deviation D21: operates on the post-hook
/// payload `Value`).
fn to_mistral_wire_payload(payload: Value) -> Value {
	let Some(mut wire) = payload.as_object().cloned() else {
		return payload;
	};
	for (source, target) in [
		("topP", "top_p"),
		("maxTokens", "max_tokens"),
		("randomSeed", "random_seed"),
		("responseFormat", "response_format"),
		("toolChoice", "tool_choice"),
		("presencePenalty", "presence_penalty"),
		("frequencyPenalty", "frequency_penalty"),
		("parallelToolCalls", "parallel_tool_calls"),
		("reasoningEffort", "reasoning_effort"),
		("promptMode", "prompt_mode"),
		("promptCacheKey", "prompt_cache_key"),
		("safePrompt", "safe_prompt"),
	] {
		remap_property(&mut wire, source, target);
	}
	if let Some(Value::Array(messages)) = wire.get("messages").cloned() {
		wire.insert(
			"messages".to_string(),
			Value::Array(messages.into_iter().map(to_mistral_wire_message).collect()),
		);
	}
	if let Some(response_format) = wire.get_mut("response_format") {
		if let Some(response_format) = response_format.as_object_mut() {
			remap_property(response_format, "jsonSchema", "json_schema");
			if let Some(json_schema) = response_format.get_mut("json_schema") {
				if let Some(json_schema) = json_schema.as_object_mut() {
					remap_property(json_schema, "schemaDefinition", "schema");
				}
			}
		}
	}
	Value::Object(wire)
}

/// Upstream `toMistralWireMessage`.
fn to_mistral_wire_message(message: Value) -> Value {
	let Some(mut wire) = message.as_object().cloned() else {
		return message;
	};
	remap_property(&mut wire, "toolCalls", "tool_calls");
	remap_property(&mut wire, "toolCallId", "tool_call_id");
	if let Some(Value::Array(chunks)) = wire.get("content").cloned() {
		wire.insert(
			"content".to_string(),
			Value::Array(chunks.into_iter().map(to_mistral_wire_content_chunk).collect()),
		);
	}
	Value::Object(wire)
}

/// Upstream `toMistralWireContentChunk`.
fn to_mistral_wire_content_chunk(chunk: Value) -> Value {
	let Some(mut wire) = chunk.as_object().cloned() else {
		return chunk;
	};
	for (source, target) in [
		("imageUrl", "image_url"),
		("documentUrl", "document_url"),
		("documentName", "document_name"),
		("fileId", "file_id"),
		("referenceIds", "reference_ids"),
		("inputAudio", "input_audio"),
	] {
		remap_property(&mut wire, source, target);
	}
	Value::Object(wire)
}

/// `AbortSignal.timeout(ms)` (deviation D17).
fn timeout_signal(ms: u64) -> AbortSignal {
	let signal = AbortSignal::new();
	let handle = signal.clone();
	tokio::spawn(async move {
		tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
		AbortSignal::abort_with(&handle, AbortReason::Message(TIMEOUT_MESSAGE.to_string()));
	});
	signal
}

fn abort_message(signal: &AbortSignal) -> String {
	signal
		.reason()
		.map(|reason| reason.to_string())
		.unwrap_or_else(|| "The operation was aborted".to_string())
}

/// Parsed SSE payload (`MistralCompletionEvent`): the raw `data` JSON object.
type MistralCompletionEvent = Value;

/// Upstream `requestMistralStream`.
async fn request_mistral_stream(
	model: &Model,
	payload: Value,
	api_key: &str,
	options: &StreamOptions,
) -> Result<MistralEventStream, MistralStreamError> {
	let base_url = model.base.base_url.trim_end_matches('/');
	let url = format!("{base_url}/v1/chat/completions");
	let headers = build_mistral_headers(model, api_key, options);
	let timeout_ms = options.base.timeout_ms.unwrap_or(60_000);
	let timeout_signal = timeout_signal(timeout_ms);
	let signal = match options.base.signal.as_ref() {
		Some(request_signal) => AbortSignal::any([request_signal.clone(), timeout_signal]),
		None => timeout_signal,
	};

	let mut request = HttpRequest::new("POST", url);
	request.headers = headers;
	request.body = HttpBody::Bytes(to_mistral_wire_payload(payload).to_string().into_bytes());
	request.signal = Some(signal.clone());

	let fetch: FetchFn = options.base.fetch.clone().unwrap_or_default();
	let response = fetch
		.send(request)
		.await
		.map_err(|error| MistralStreamError::Message(error.message))?;

	if let Some(hook) = options.base.on_response.as_ref() {
		hook(
			ProviderResponse { status: response.status, headers: headers_to_record(&response.headers) },
			model.clone(),
		)
		.await;
	}

	if !response.ok() {
		let body = response.body.collect().await.unwrap_or_default();
		return Err(MistralStreamError::Http(MistralHttpError {
			status_code: response.status,
			body: String::from_utf8_lossy(&body).to_string(),
			// Deviation D20: no `statusText` on the seam.
			message: format!("Request failed with status {}", response.status),
		}));
	}
	let body: BodyStream = match response.body {
		HttpBody::Stream(body_stream) => body_stream,
		HttpBody::Bytes(bytes) => Box::pin(futures_util::stream::once(async move { Ok(bytes) })),
		HttpBody::Empty => {
			return Err(MistralStreamError::Message("Mistral response has no body".to_string()));
		}
	};
	Ok(MistralEventStream::new(body, Some(signal)))
}

// ---------------------------------------------------------------------------
// SSE decoding (`readMistralEvents`, `parseMistralEvent`)
// ---------------------------------------------------------------------------

enum ParsedSse {
	/// Event with no `data:` payload (upstream returns `undefined`).
	Skip,
	/// `data: [DONE]`.
	Done,
	Event(Value),
}

/// Upstream `findMistralEventBoundary`: leftmost match of
/// `\r\n\r\n|\r\n\r|\r\n\n|\r\r\n|\n\r\n|\r\r|\n\r|\n\n`, returning
/// `(index, length)`.
fn find_mistral_event_boundary(buffer: &str) -> Option<(usize, usize)> {
	const PATTERNS: [&[u8]; 8] =
		[b"\r\n\r\n", b"\r\n\r", b"\r\n\n", b"\r\r\n", b"\n\r\n", b"\r\r", b"\n\r", b"\n\n"];
	let bytes = buffer.as_bytes();
	for index in 0..bytes.len() {
		for pattern in PATTERNS {
			if bytes[index..].starts_with(pattern) {
				return Some((index, pattern.len()));
			}
		}
	}
	None
}

/// Split on `/\r\n|\r|\n/u` (upstream `raw.split(...)`).
fn split_mistral_lines(raw: &str) -> Vec<&str> {
	let mut lines = Vec::new();
	let bytes = raw.as_bytes();
	let mut start = 0usize;
	let mut index = 0usize;
	while index < bytes.len() {
		match bytes[index] {
			b'\r' => {
				lines.push(&raw[start..index]);
				index += if bytes.get(index + 1) == Some(&b'\n') { 2 } else { 1 };
				start = index;
			}
			b'\n' => {
				lines.push(&raw[start..index]);
				index += 1;
				start = index;
			}
			_ => index += 1,
		}
	}
	lines.push(&raw[start..]);
	lines
}

/// Upstream `parseMistralEvent`.
fn parse_mistral_event(raw: &str) -> Result<ParsedSse, MistralStreamError> {
	let data = split_mistral_lines(raw)
		.into_iter()
		.filter(|line| line.starts_with("data:"))
		.map(|line| line[5..].trim_start())
		.collect::<Vec<_>>()
		.join("\n")
		.trim()
		.to_string();
	if data.is_empty() {
		return Ok(ParsedSse::Skip);
	}
	if data == "[DONE]" {
		return Ok(ParsedSse::Done);
	}
	let parsed: Value = serde_json::from_str(&data)
		.map_err(|error| MistralStreamError::Message(error.to_string()))?;
	match parsed {
		Value::Object(object) => {
			if !object.get("choices").is_some_and(Value::is_array) {
				return Err(MistralStreamError::Message("Invalid Mistral streaming event".to_string()));
			}
			Ok(ParsedSse::Event(Value::Object(object)))
		}
		_ => Err(MistralStreamError::Message("Invalid Mistral streaming event".to_string())),
	}
}

/// Streaming SSE event source over the response body
/// (`readMistralEvents`; deviation D18 for the UTF-8 decoding).
struct MistralEventStream {
	body: BodyStream,
	signal: Option<AbortSignal>,
	/// Bytes not yet decodable (incomplete UTF-8 sequence at the tail).
	pending_bytes: Vec<u8>,
	/// Decoded text not yet consumed by an event boundary.
	buffer: String,
	eof_seen: bool,
	finished: bool,
	queue: VecDeque<MistralCompletionEvent>,
}

impl MistralEventStream {
	fn new(body: BodyStream, signal: Option<AbortSignal>) -> MistralEventStream {
		MistralEventStream {
			body,
			signal,
			pending_bytes: Vec::new(),
			buffer: String::new(),
			eof_seen: false,
			finished: false,
			queue: VecDeque::new(),
		}
	}

	/// Decode complete UTF-8 prefixes of the pending bytes into the text
	/// buffer (`TextDecoder(..., {stream: true})`); flush lossily at EOF
	/// (final `decoder.decode()`).
	fn decode_buffered(&mut self, final_flush: bool) {
		if self.pending_bytes.is_empty() {
			return;
		}
		if final_flush {
			let bytes = std::mem::take(&mut self.pending_bytes);
			self.buffer.push_str(&String::from_utf8_lossy(&bytes));
			return;
		}
		let valid_up_to = match std::str::from_utf8(&self.pending_bytes) {
			Ok(_) => self.pending_bytes.len(),
			Err(error) => error.valid_up_to(),
		};
		if valid_up_to == 0 {
			return;
		}
		let bytes: Vec<u8> = self.pending_bytes.drain(..valid_up_to).collect();
		self.buffer.push_str(&String::from_utf8_lossy(&bytes));
	}

	/// Parse and queue every complete event currently in the buffer.
	/// `Ok(true)` means the `[DONE]` sentinel was seen.
	fn drain_boundaries(&mut self) -> Result<bool, MistralStreamError> {
		while let Some((index, length)) = find_mistral_event_boundary(&self.buffer) {
			let raw = self.buffer[..index].to_string();
			self.buffer.drain(..index + length);
			match parse_mistral_event(&raw)? {
				ParsedSse::Skip => {}
				ParsedSse::Done => return Ok(true),
				ParsedSse::Event(event) => self.queue.push_back(event),
			}
		}
		Ok(false)
	}
}

impl futures_core::Stream for MistralEventStream {
	type Item = Result<MistralCompletionEvent, MistralStreamError>;

	fn poll_next(self: std::pin::Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
		let this = self.get_mut();
		loop {
			if let Some(event) = this.queue.pop_front() {
				return Poll::Ready(Some(Ok(event)));
			}
			if this.finished {
				return Poll::Ready(None);
			}
			if this.signal.as_ref().is_some_and(|signal| signal.aborted()) {
				this.finished = true;
				let message = this
					.signal
					.as_ref()
					.map(|signal| abort_message(signal))
					.unwrap_or_else(|| "The operation was aborted".to_string());
				return Poll::Ready(Some(Err(MistralStreamError::Message(message))));
			}

			this.decode_buffered(this.eof_seen);
			match this.drain_boundaries() {
				Ok(true) => this.finished = true,
				Ok(false) => {}
				Err(error) => {
					this.finished = true;
					return Poll::Ready(Some(Err(error)));
				}
			}
			if this.eof_seen {
				// Trailing event without a terminator (`buffer.trim()`).
				if !this.buffer.trim().is_empty() {
					let raw = std::mem::take(&mut this.buffer);
					match parse_mistral_event(&raw) {
						Ok(ParsedSse::Event(event)) => this.queue.push_back(event),
						Ok(ParsedSse::Skip | ParsedSse::Done) => {}
						Err(error) => {
							this.finished = true;
							return Poll::Ready(Some(Err(error)));
						}
					}
				}
				this.finished = true;
				continue;
			}

			match this.body.as_mut().poll_next(cx) {
				Poll::Ready(Some(Ok(chunk))) => {
					this.pending_bytes.extend_from_slice(&chunk);
				}
				Poll::Ready(Some(Err(error))) => {
					this.finished = true;
					return Poll::Ready(Some(Err(MistralStreamError::Message(error.message))));
				}
				Poll::Ready(None) => {
					this.eof_seen = true;
				}
				Poll::Pending => {
					// Keep the abort waker registered so a concurrent abort or
					// the request timeout re-awaits this stream (upstream
					// cancels the reader from an abort listener).
					if let Some(signal) = this.signal.clone() {
						let aborted = signal.aborted_future();
						let _ = std::pin::pin!(aborted).poll(cx);
					}
					return Poll::Pending;
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Transcript conversion (`toChatMessages`, `buildToolResultText`, tools)
// ---------------------------------------------------------------------------

/// Upstream `toFunctionTools` (deviation D22: `stripSymbolKeys` is identity).
fn to_function_tools(tools: &[Tool]) -> Result<Vec<MistralFunctionTool>, String> {
	tools
		.iter()
		.map(|tool| {
			let strict = crate::api::constrained_sampling::resolve_json_schema_strict_sampling(tool, true)?;
			let parameters = crate::api::constrained_sampling::get_json_schema_tool_parameters(tool, strict.unwrap_or(false))
				.map_err(|error| error.to_string())?;
			Ok(MistralFunctionTool {
				tag: "function",
				function: MistralFunctionDef {
					name: tool.name.clone(),
					description: tool.description.clone(),
					parameters,
					strict: strict.unwrap_or(false),
				},
			})
		})
		.collect()
}

/// Upstream `buildToolResultText`.
fn build_tool_result_text(text: &str, has_images: bool, supports_images: bool, is_error: bool) -> String {
	let trimmed = text.trim();
	let error_prefix = if is_error { "[tool error] " } else { "" };

	if !trimmed.is_empty() {
		let image_suffix =
			if has_images && !supports_images { "\n[tool image omitted: model does not support images]" } else { "" };
		return format!("{error_prefix}{trimmed}{image_suffix}");
	}

	if has_images {
		if supports_images {
			return if is_error { "[tool error] (see attached image)".to_string() } else { "(see attached image)".to_string() };
		}
		return if is_error {
			"[tool error] (image omitted: model does not support images)".to_string()
		} else {
			"(image omitted: model does not support images)".to_string()
		};
	}

	if is_error { "[tool error] (no tool output)".to_string() } else { "(no tool output)".to_string() }
}

fn text_content_chunk(text: String) -> MistralContentChunk {
	MistralContentChunk::Text { text }
}

fn image_content_chunk(image: &crate::types::ImageContent) -> MistralContentChunk {
	MistralContentChunk::ImageUrl { image_url: format!("data:{};base64,{}", image.mime_type, image.data) }
}

/// Upstream `toChatMessages`.
fn to_chat_messages(messages: &[Message], supports_images: bool) -> Vec<MistralChatMessage> {
	let mut result: Vec<MistralChatMessage> = Vec::new();

	for (index, msg) in messages.iter().enumerate() {
		match msg {
			Message::System(message) => {
				let text = if index == 0 {
					get_system_message_text(message)
				} else {
					render_system_message_update(message)
				};
				if !text.is_empty() {
					let mut chat_message = MistralChatMessage::new("system");
					chat_message.content = Some(MistralMessageContent::Text(sanitize_surrogates(&text)));
					result.push(chat_message);
				}
			}
			Message::User(message) => match &message.content {
				UserContentField::Text(text) => {
					let mut chat_message = MistralChatMessage::new("user");
					chat_message.content = Some(MistralMessageContent::Text(sanitize_surrogates(text)));
					result.push(chat_message);
				}
				UserContentField::Blocks(blocks) => {
					let had_images = blocks.iter().any(|item| matches!(item, UserContent::Image(_)));
					let content: Vec<MistralContentChunk> = blocks
						.iter()
						.filter(|item| matches!(item, UserContent::Text(_)) || supports_images)
						.map(|item| match item {
							UserContent::Text(text) => text_content_chunk(sanitize_surrogates(&text.text)),
							UserContent::Image(image) => image_content_chunk(image),
						})
						.collect();
					if !content.is_empty() {
						let mut chat_message = MistralChatMessage::new("user");
						chat_message.content = Some(MistralMessageContent::Chunks(content));
						result.push(chat_message);
					} else if had_images && !supports_images {
						let mut chat_message = MistralChatMessage::new("user");
						chat_message.content =
							Some(MistralMessageContent::Text("(image omitted: model does not support images)".to_string()));
						result.push(chat_message);
					}
				}
			},
			Message::Assistant(message) => {
				let mut content_parts: Vec<MistralContentChunk> = Vec::new();
				let mut tool_calls: Vec<MistralRequestToolCall> = Vec::new();

				for block in &message.content {
					match block {
						AssistantContent::Text(block) => {
							if !block.text.trim().is_empty() {
								content_parts.push(text_content_chunk(sanitize_surrogates(&block.text)));
							}
						}
						AssistantContent::Thinking(block) => {
							if !block.thinking.trim().is_empty() {
								content_parts.push(MistralContentChunk::Thinking {
									thinking: vec![MistralThinkingPart::text(sanitize_surrogates(&block.thinking))],
								});
							}
						}
						AssistantContent::ToolCall(block) => {
							tool_calls.push(MistralRequestToolCall {
								id: block.id.clone(),
								tag: "function",
								function: MistralRequestFunction {
									name: block.name.clone(),
									arguments: Value::Object(block.arguments.clone()).to_string(),
								},
								index: 0,
							});
						}
					}
				}

				let mut assistant_message = MistralChatMessage::new("assistant");
				assistant_message.prefix = Some(false);
				if !content_parts.is_empty() {
					assistant_message.content = Some(MistralMessageContent::Chunks(content_parts));
				}
				if !tool_calls.is_empty() {
					assistant_message.tool_calls = Some(tool_calls);
				}
				if assistant_message.content.is_some() || assistant_message.tool_calls.is_some() {
					result.push(assistant_message);
				}
			}
			Message::ToolResult(message) => {
				let mut tool_content: Vec<MistralContentChunk> = Vec::new();
				let text_result = message
					.content
					.iter()
					.map(|part| match part {
						UserContent::Text(text) => sanitize_surrogates(&text.text),
						UserContent::Image(_) => String::new(),
					})
					.collect::<Vec<_>>()
					.join("\n");
				let has_images = message.content.iter().any(|part| matches!(part, UserContent::Image(_)));
				let tool_text = build_tool_result_text(&text_result, has_images, supports_images, message.is_error);
				tool_content.push(text_content_chunk(tool_text));
				for part in &message.content {
					if !supports_images {
						continue;
					}
					if let UserContent::Image(image) = part {
						tool_content.push(image_content_chunk(image));
					}
				}
				let mut chat_message = MistralChatMessage::new("tool");
				chat_message.tool_call_id = Some(message.tool_call_id.clone());
				chat_message.name = Some(message.tool_name.clone());
				chat_message.content = Some(MistralMessageContent::Chunks(tool_content));
				result.push(chat_message);
			}
		}
	}

	result
}

/// Upstream `mapToolChoice` (the typed option already carries the narrowed
/// wire shape, so mapping is the identity).
fn map_tool_choice(choice: &MistralToolChoiceOption) -> MistralToolChoiceOption {
	match choice {
		MistralToolChoiceOption::Function { tag, function } => MistralToolChoiceOption::Function {
			tag: *tag,
			function: MistralNamedFunction { name: function.name.clone() },
		},
		built_in => built_in.clone(),
	}
}

/// Upstream `buildChatPayload`.
fn build_chat_payload(
	model: &Model,
	context: &TranscriptContext,
	messages: &[Message],
	options: &StreamOptions,
	mistral: &MistralOptions,
) -> Result<MistralChatPayload, String> {
	let supports_images = model.base.input.iter().any(|input| *input == ModelInput::Image);
	let mut payload = MistralChatPayload {
		model: model.id().to_string(),
		stream: true,
		messages: to_chat_messages(messages, supports_images),
		tools: None,
		temperature: None,
		max_tokens: None,
		tool_choice: None,
		prompt_mode: None,
		reasoning_effort: None,
		prompt_cache_key: None,
	};

	let current_tools = get_current_tools(context.messages());
	if !current_tools.is_empty() {
		payload.tools = Some(to_function_tools(&current_tools)?);
	}
	if let Some(temperature) = options.temperature {
		payload.temperature = Some(temperature);
	}
	if let Some(max_tokens) = options.max_tokens {
		payload.max_tokens = Some(max_tokens);
	}
	if let Some(tool_choice) = mistral.tool_choice.as_ref() {
		payload.tool_choice = Some(map_tool_choice(tool_choice));
	}
	if let Some(prompt_mode) = mistral.prompt_mode {
		payload.prompt_mode = Some(prompt_mode);
	}
	if let Some(reasoning_effort) = mistral.reasoning_effort {
		payload.reasoning_effort = Some(reasoning_effort);
	}
	if should_use_prompt_caching(options) {
		payload.prompt_cache_key = options.session_id.clone();
	}

	Ok(payload)
}

/// Upstream `getMistralCachedPromptTokens`.
fn get_mistral_cached_prompt_tokens(usage: &Value, prompt_tokens: u64) -> u64 {
	let mut raw_cached: Option<&Value> = None;
	for (outer, inner) in [
		("promptTokensDetails", "cachedTokens"),
		("prompt_tokens_details", "cached_tokens"),
		("promptTokenDetails", "cachedTokens"),
		("prompt_token_details", "cached_tokens"),
	] {
		if let Some(value) = usage.get(outer).and_then(|outer| outer.get(inner)) {
			if !value.is_null() {
				raw_cached = Some(value);
				break;
			}
		}
	}
	let raw_cached = raw_cached
		.or_else(|| usage.get("numCachedTokens").filter(|value| !value.is_null()))
		.or_else(|| usage.get("num_cached_tokens").filter(|value| !value.is_null()));
	let cached_tokens = raw_cached.and_then(Value::as_f64).filter(|value| value.is_finite()).unwrap_or(0.0);
	((prompt_tokens as f64).min(cached_tokens.max(0.0))) as u64
}

/// Upstream `mapChatStopReason` (only called with a truthy reason upstream).
fn map_chat_stop_reason(reason: &str) -> (StopReason, Option<String>) {
	match reason {
		"stop" => (StopReason::Stop, None),
		"length" | "model_length" => (StopReason::Length, None),
		"tool_calls" => (StopReason::ToolUse, None),
		"error" => (StopReason::Error, Some("Provider stopped with: error".to_string())),
		other => (StopReason::Error, Some(format!("Provider stopped with: {other}"))),
	}
}

// ---------------------------------------------------------------------------
// Stream consumption (`consumeChatStream`)
// ---------------------------------------------------------------------------

/// Which block kind is currently open (upstream `currentBlock`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentBlock {
	Text,
	Thinking,
}

/// Upstream tool-block identity key: `toolCall.index ?? callId`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolBlockKey {
	Index(u64),
	Id(String),
}

fn finish_current_block(
	output: &mut AssistantMessage,
	stream: &AssistantMessageEventStream,
	current_block: Option<CurrentBlock>,
) {
	let Some(kind) = current_block else { return };
	let content_index = output.content.len().saturating_sub(1) as u32;
	match kind {
		CurrentBlock::Text => {
			let text = match output.content.last() {
				Some(AssistantContent::Text(block)) => block.text.clone(),
				_ => String::new(),
			};
			stream.push(AssistantMessageEvent::TextEnd { content_index, content: text, partial: output.clone() });
		}
		CurrentBlock::Thinking => {
			let thinking = match output.content.last() {
				Some(AssistantContent::Thinking(block)) => block.thinking.clone(),
				_ => String::new(),
			};
			stream.push(AssistantMessageEvent::ThinkingEnd { content_index, content: thinking, partial: output.clone() });
		}
	}
}

fn open_text_block(output: &mut AssistantMessage, stream: &AssistantMessageEventStream) {
	output.content.push(AssistantContent::Text(TextContent { text: String::new(), text_signature: None }));
	stream.push(AssistantMessageEvent::TextStart {
		content_index: (output.content.len() - 1) as u32,
		partial: output.clone(),
	});
}

fn open_thinking_block(output: &mut AssistantMessage, stream: &AssistantMessageEventStream) {
	output
		.content
		.push(AssistantContent::Thinking(ThinkingContent { thinking: String::new(), thinking_signature: None, redacted: None }));
	stream.push(AssistantMessageEvent::ThinkingStart {
		content_index: (output.content.len() - 1) as u32,
		partial: output.clone(),
	});
}

/// Upstream `consumeChatStream`. Tool-call `partialArgs` live in the local
/// `tool_blocks` scratch list, never on the output blocks (upstream deletes
/// them from the blocks before finishing).
async fn consume_chat_stream(
	model: &Model,
	output: &mut AssistantMessage,
	stream: &AssistantMessageEventStream,
	mistral_stream: MistralEventStream,
	on_provider_stream_event: Option<crate::types::ProviderStreamEventHook>,
) -> Result<(), MistralStreamError> {
	let mut current_block: Option<CurrentBlock> = None;
	let mut tool_blocks: Vec<(ToolBlockKey, usize, String)> = Vec::new();

	let mut events = std::pin::pin!(mistral_stream);
	while let Some(event) = events.next().await {
		let chunk = event?;
		if let Some(hook) = on_provider_stream_event.as_ref() {
			hook(chunk.clone(), model.clone()).await;
		}
		// Mistral's streamed CompletionChunk carries an id field. Keep the
		// first non-empty one.
		if output.response_id.as_ref().map_or(true, |id| id.is_empty()) {
			if let Some(id) = chunk.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()) {
				output.response_id = Some(id.to_string());
			}
		}

		if let Some(usage) = chunk.get("usage").filter(|usage| !usage.is_null()) {
			let prompt_tokens = usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
			let cached_prompt_tokens = get_mistral_cached_prompt_tokens(usage, prompt_tokens);

			output.usage.input = prompt_tokens.saturating_sub(cached_prompt_tokens);
			output.usage.output = usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0);
			output.usage.cache_read = cached_prompt_tokens;
			output.usage.cache_write = 0;
			output.usage.total_tokens = usage.get("total_tokens").and_then(Value::as_u64).unwrap_or(
				output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write,
			);
			calculate_cost(&AnyModel::Chat(model.clone()), &mut output.usage);
		}

		let Some(choice) = chunk.get("choices").and_then(Value::as_array).and_then(|choices| choices.first()) else {
			continue;
		};

		if let Some(finish_reason) =
			choice.get("finish_reason").and_then(Value::as_str).filter(|reason| !reason.is_empty())
		{
			output.raw_stop_reason = Some(finish_reason.to_string());
			let (stop_reason, error_message) = map_chat_stop_reason(finish_reason);
			output.stop_reason = stop_reason;
			if let Some(error_message) = error_message {
				output.error_message = Some(error_message);
			}
		}

		// Deviation D19: a missing `delta` behaves as an empty delta.
		let delta = choice.get("delta").cloned().unwrap_or(Value::Null);

		if !delta.get("content").map_or(true, Value::is_null) {
			let content_items: Vec<Value> = match delta.get("content") {
				Some(Value::String(text)) => vec![Value::String(text.clone())],
				Some(Value::Array(items)) => items.clone(),
				_ => Vec::new(),
			};
			for item in content_items {
				match item {
					Value::String(text) => {
						let text_delta = sanitize_surrogates(&text);
						// GLM models on Mistral send empty content deltas
						// around thinking and tool calls (#9674).
						if text_delta.is_empty() {
							continue;
						}
						if current_block != Some(CurrentBlock::Text) {
							finish_current_block(output, stream, current_block);
							open_text_block(output, stream);
							current_block = Some(CurrentBlock::Text);
						}
						if let Some(AssistantContent::Text(block)) = output.content.last_mut() {
							block.text.push_str(&text_delta);
						}
						stream.push(AssistantMessageEvent::TextDelta {
							content_index: (output.content.len() - 1) as u32,
							delta: text_delta,
							partial: output.clone(),
						});
					}
					Value::Object(item) => match item.get("type").and_then(Value::as_str) {
						Some("thinking") => {
							let delta_text = item
								.get("thinking")
								.and_then(Value::as_array)
								.map(|parts| {
									parts
										.iter()
										.filter_map(|part| part.get("text").and_then(Value::as_str))
										.filter(|text| !text.is_empty())
										.collect::<Vec<_>>()
										.join("")
								})
								.unwrap_or_default();
							let thinking_delta = sanitize_surrogates(&delta_text);
							if thinking_delta.is_empty() {
								continue;
							}
							if current_block != Some(CurrentBlock::Thinking) {
								finish_current_block(output, stream, current_block);
								open_thinking_block(output, stream);
								current_block = Some(CurrentBlock::Thinking);
							}
							if let Some(AssistantContent::Thinking(block)) = output.content.last_mut() {
								block.thinking.push_str(&thinking_delta);
							}
							stream.push(AssistantMessageEvent::ThinkingDelta {
								content_index: (output.content.len() - 1) as u32,
								delta: thinking_delta,
								partial: output.clone(),
							});
						}
						Some("text") => {
							let text_delta =
								sanitize_surrogates(item.get("text").and_then(Value::as_str).unwrap_or(""));
							if text_delta.is_empty() {
								continue;
							}
							if current_block != Some(CurrentBlock::Text) {
								finish_current_block(output, stream, current_block);
								open_text_block(output, stream);
								current_block = Some(CurrentBlock::Text);
							}
							if let Some(AssistantContent::Text(block)) = output.content.last_mut() {
								block.text.push_str(&text_delta);
							}
							stream.push(AssistantMessageEvent::TextDelta {
								content_index: (output.content.len() - 1) as u32,
								delta: text_delta,
								partial: output.clone(),
							});
						}
						_ => {}
					},
					_ => {}
				}
			}
		}

		let tool_calls = delta.get("tool_calls").and_then(Value::as_array).cloned().unwrap_or_default();
		for tool_call in tool_calls {
			if current_block.is_some() {
				finish_current_block(output, stream, current_block);
				current_block = None;
			}
			let function = tool_call.get("function").cloned().unwrap_or(Value::Null);
			let call_id = match tool_call.get("id").and_then(Value::as_str) {
				Some(id) if !id.is_empty() && id != "null" => id.to_string(),
				_ => derive_mistral_tool_call_id(
					&format!("toolcall:{}", tool_call.get("index").and_then(Value::as_u64).unwrap_or(0)),
					0,
				),
			};
			let key = match tool_call.get("index").and_then(Value::as_u64) {
				Some(index) => ToolBlockKey::Index(index),
				None => ToolBlockKey::Id(call_id.clone()),
			};
			let existing_position = tool_blocks
				.iter()
				.position(|(entry_key, _, _)| *entry_key == key)
				.filter(|position| {
					matches!(output.content.get(tool_blocks[*position].1), Some(AssistantContent::ToolCall(_)))
				});

			let block_index = match existing_position {
				Some(position) => tool_blocks[position].1,
				None => {
					let block = AssistantContent::ToolCall(ToolCall {
						id: call_id,
						name: function.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
						arguments: serde_json::Map::new(),
						thought_signature: None,
						namespace: None,
					});
					output.content.push(block);
					let index = output.content.len() - 1;
					match tool_blocks.iter().position(|(entry_key, _, _)| *entry_key == key) {
						Some(position) => tool_blocks[position].1 = index,
						None => tool_blocks.push((key.clone(), index, String::new())),
					}
					stream.push(AssistantMessageEvent::ToolcallStart {
						content_index: index as u32,
						partial: output.clone(),
					});
					index
				}
			};

			let args_delta = match function.get("arguments") {
				Some(Value::String(text)) => text.clone(),
				Some(Value::Null) | None => "{}".to_string(),
				Some(other) => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
			};
			let position = tool_blocks
				.iter()
				.position(|(entry_key, entry_index, _)| *entry_key == key && *entry_index == block_index)
				.expect("tool block registered");
			tool_blocks[position].2.push_str(&args_delta);
			let partial_args = tool_blocks[position].2.clone();
			let arguments = parse_streaming_json(Some(&partial_args));
			if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(block_index) {
				block.arguments = arguments.as_object().cloned().unwrap_or_default();
			}
			stream.push(AssistantMessageEvent::ToolcallDelta {
				content_index: block_index as u32,
				delta: args_delta,
				partial: output.clone(),
			});
		}
	}

	finish_current_block(output, stream, current_block);
	for (_, index, partial_args) in tool_blocks.iter() {
		let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(*index) else {
			continue;
		};
		block.arguments = parse_streaming_json(Some(partial_args)).as_object().cloned().unwrap_or_default();
		let tool_call = block.clone();
		stream.push(AssistantMessageEvent::ToolcallEnd {
			content_index: *index as u32,
			tool_call,
			partial: output.clone(),
		});
	}
	Ok(())
}

// ---------------------------------------------------------------------------
// stream()
// ---------------------------------------------------------------------------

fn create_output(model: &Model) -> AssistantMessage {
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
		usage: Default::default(),
		stop_reason: StopReason::Pending,
		deferred: None,
		error_message: None,
		raw_stop_reason: None,
		end_turn: None,
		timestamp: now_ms(),
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

/// Upstream `stream` export for the mistral-conversations API.
pub fn stream(model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
	let mistral = MistralOptions::default();
	stream_with_options(model, context, options, &mistral)
}

/// `stream` with the mistral option group (deviation D15).
pub fn stream_with_options(
	model: &Model,
	context: &TranscriptContext,
	options: &StreamOptions,
	mistral: &MistralOptions,
) -> AssistantMessageEventStream {
	let event_stream = AssistantMessageEventStream::new();
	let supports_mid_convo_system_messages = model
		.compat
		.as_ref()
		.and_then(|compat: &ModelCompat| compat.supports_mid_convo_system_messages);
	let normalized_context = resolve_transcript(context, supports_mid_convo_system_messages);

	let model = model.clone();
	let options = options.clone();
	let mistral = mistral.clone();
	let inner = event_stream.clone();
	tokio::spawn(async move {
		run_stream(model, normalized_context, options, mistral, inner).await;
	});
	event_stream
}

async fn run_stream(
	model: Model,
	normalized_context: TranscriptContext,
	options: StreamOptions,
	mistral: MistralOptions,
	stream: AssistantMessageEventStream,
) {
	let mut output = create_output(&model);

	let result: Result<(), MistralStreamError> = async {
		let api_key = options.base.api_key.clone().filter(|key| !key.is_empty());
		let Some(api_key) = api_key else {
			return Err(MistralStreamError::Message(format!("No API key for provider: {}", model.provider())));
		};

		let normalizer = MistralToolCallIdNormalizer::default();
		let transformed_messages = crate::api::transform_messages::transform_messages(
			normalized_context.messages(),
			&model,
			Some(&|id: &str, _model: &Model, _assistant: &AssistantMessage| normalizer.normalize(id)),
		);

		let payload =
			build_chat_payload(&model, &normalized_context, &transformed_messages, &options, &mistral).map_err(
				|error| MistralStreamError::Message(error),
			)?;
		let mut payload_value =
			serde_json::to_value(&payload).map_err(|error| MistralStreamError::Message(error.to_string()))?;
		if let Some(hook) = options.base.on_payload.as_ref() {
			if let Some(next) = hook(payload_value.clone(), model.clone()).await {
				payload_value = next;
			}
		}

		let mistral_stream = request_mistral_stream(&model, payload_value, &api_key, &options).await?;

		stream.push(AssistantMessageEvent::Start { partial: output.clone() });

		consume_chat_stream(&model, &mut output, &stream, mistral_stream, options.on_provider_stream_event.clone())
			.await?;

		if options.base.signal.as_ref().is_some_and(|signal| signal.aborted()) {
			return Err(MistralStreamError::Message("Request was aborted".to_string()));
		}
		if output.stop_reason == StopReason::Pending {
			return Err(MistralStreamError::Message(
				"Mistral stream ended without a finish reason".to_string(),
			));
		}
		if output.stop_reason == StopReason::Aborted || output.stop_reason == StopReason::Error {
			return Err(MistralStreamError::Message(
				output
					.error_message
					.clone()
					.unwrap_or_else(|| "An unknown error occurred".to_string()),
			));
		}

		let reason = done_reason(output.stop_reason);
		stream.push(done_event(output.clone(), reason));
		stream.end(Some(output.clone()));
		Ok(())
	}
	.await;

	if let Err(error) = result {
		let aborted = options.base.signal.as_ref().is_some_and(|signal| signal.aborted());
		output.stop_reason = if aborted { StopReason::Aborted } else { StopReason::Error };
		output.error_message = Some(format_mistral_error(&error));
		let reason = if aborted { StreamErrorReason::Aborted } else { StreamErrorReason::Error };
		stream.push(error_event(output, reason));
		stream.end(None);
	}
}

// ---------------------------------------------------------------------------
// streamSimple()
// ---------------------------------------------------------------------------

/// Upstream `streamSimple` export.
pub fn stream_simple(
	model: &Model,
	context: &TranscriptContext,
	options: &SimpleStreamOptions,
) -> AssistantMessageEventStream {
	let mistral = MistralOptions::default();
	stream_simple_with_options(model, context, options, &mistral)
}

/// `streamSimple` with the mistral option group (deviation D15: upstream
/// computes the group from the simple options, so the parameter is accepted
/// for signature symmetry and the computed values always win).
pub fn stream_simple_with_options(
	model: &Model,
	context: &TranscriptContext,
	options: &SimpleStreamOptions,
	_mistral: &MistralOptions,
) -> AssistantMessageEventStream {
	let api_key = options.base.base.api_key.clone().filter(|key| !key.is_empty());
	let Some(api_key) = api_key else {
		return error_stream(model, &format!("No API key for provider: {}", model.provider()));
	};

	let base = crate::api::simple_options::build_base_options(model, context, options, Some(&api_key));
	let clamped_reasoning = options.reasoning.map(|level| clamp_thinking_level(model, level.into()));
	let reasoning = clamped_reasoning.filter(|level| *level != ModelThinkingLevel::Off);
	// Models with a thinking level map use `reasoning_effort`; other reasoning
	// models use `prompt_mode` (deviation D23 for the mapped-value validation).
	let effort_map = if model.reasoning { model.thinking_level_map.as_ref() } else { None };
	let reasoning_effort: Option<MistralReasoningEffort> = effort_map.and_then(|map| match reasoning {
		Some(level) => {
			let mapped = map.get(&level);
			Some(match mapped {
				Some(Some(value)) => MistralReasoningEffort::parse(value).unwrap_or(MistralReasoningEffort::High),
				_ => MistralReasoningEffort::High,
			})
		}
		None => map
			.get(&ModelThinkingLevel::Off)
			.cloned()
			.flatten()
			.and_then(|value| MistralReasoningEffort::parse(&value)),
	});
	let prompt_mode = if model.reasoning && effort_map.is_none() && reasoning.is_some() {
		Some(MistralPromptMode::Reasoning)
	} else {
		None
	};

	let mistral_options = MistralOptions {
		tool_choice: options.tool_choice.map(|tool_choice| match tool_choice {
			ToolChoice::Auto => MistralToolChoiceOption::BuiltIn(MistralBuiltinToolChoice::Auto),
			ToolChoice::None => MistralToolChoiceOption::BuiltIn(MistralBuiltinToolChoice::None),
		}),
		prompt_mode,
		reasoning_effort,
	};
	stream_with_options(model, context, &base, &mistral_options)
}

/// Synchronous setup failure surfaced as a terminated error stream (crate
/// deviation D1).
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

/// The mistral-conversations API implementation (`export const
/// mistralConversationsApi` upstream; the `lazyApi` wrapper collapses under
/// crate deviation D10). No deferred methods are exported upstream, so both
/// presence probes stay `false`.
pub struct MistralConversationsApi;

impl ProviderStreams for MistralConversationsApi {
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

/// Upstream `mistralConversationsApi()`.
pub fn mistral_conversations_api() -> Arc<dyn ProviderStreams> {
	Arc::new(MistralConversationsApi)
}

// ---------------------------------------------------------------------------
// Tests (ports of test/mistral-http-transport.test.ts,
// test/mistral-raw-stop-reason.test.ts, test/mistral-reasoning-mode.test.ts
// and test/mistral-tool-schema.test.ts)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;
	use crate::http::{HttpResponse, ProviderHttpError};
	use crate::types::{
		BaseModelFields, ConstrainedSamplingConfig, ConstrainedSamplingSetting, Context, ImageContent,
		ModelCost, ModelCostRates, StrictSchemaMode, ToolResultMessage, UserMessage,
	};
	use serde_json::json;
	use parking_lot::Mutex;

	/// Upstream `getModel("mistral", id)`: a real catalog entry.
	fn catalog_model(id: &str) -> Model {
		crate::providers::mistral_models::mistral_models()
			.into_iter()
			.find(|model| model.id() == id)
			.unwrap_or_else(|| panic!("model {id} present in the mistral catalog"))
	}

	fn normalize(context: &Context) -> TranscriptContext {
		crate::utils::transcript::normalize_context(context)
	}

	fn user_context(text: &str) -> Context {
		Context {
			system_prompt: None,
			messages: vec![Message::User(UserMessage {
				content: UserContentField::Text(text.to_string()),
				timestamp: 1,
			})],
			tools: None,
		}
	}

	fn base_options(api_key: &str) -> StreamOptions {
		let mut options = StreamOptions::default();
		options.base.api_key = Some(api_key.to_string());
		options
	}

	/// Upstream `createSseResponse` (`\r\n\r\n` framing + `[DONE]`).
	fn sse_response(events: &[Value]) -> HttpResponse {
		let body = format!(
			"{}\r\n\r\ndata: [DONE]\r\n\r\n",
			events.iter().map(|event| format!("data: {event}")).collect::<Vec<_>>().join("\r\n\r\n")
		);
		HttpResponse {
			status: 200,
			headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
			body: HttpBody::Bytes(body.into_bytes()),
		}
	}

	/// Upstream `createBytewiseSseResponse`: one byte per transport chunk.
	fn bytewise_sse_response(event: &Value) -> HttpResponse {
		let body = format!("data: {event}\r\n\r\ndata: [DONE]\r\n\r\n");
		let bytes = body.into_bytes();
		HttpResponse {
			status: 200,
			headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
			body: HttpBody::Stream(Box::pin(futures_util::stream::iter(
				bytes.into_iter().map(|byte| Ok::<Vec<u8>, ProviderHttpError>(vec![byte])),
			))),
		}
	}

	fn never_yielding_response() -> HttpResponse {
		HttpResponse {
			status: 200,
			headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
			body: HttpBody::Stream(Box::pin(futures_util::stream::pending::<Result<Vec<u8>, ProviderHttpError>>())),
		}
	}

	/// Upstream `createTerminalEvent`.
	fn terminal_event(finish_reason: &str) -> Value {
		json!({
			"id": "mistral-response-id",
			"model": "mistral-large-latest",
			"choices": [{ "index": 0, "finish_reason": finish_reason, "delta": {} }],
			"usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
		})
	}

	fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
		headers
			.iter()
			.find(|(key, _)| key.eq_ignore_ascii_case(name))
			.map(|(_, value)| value.as_str())
	}

	fn has_header(headers: &[(String, String)], name: &str) -> bool {
		headers.iter().any(|(key, _)| key.eq_ignore_ascii_case(name))
	}

	/// Owned snapshot of the outgoing request (`HttpRequest` is not `Clone`).
	#[derive(Debug, Clone)]
	struct CapturedRequest {
		url: String,
		headers: Vec<(String, String)>,
		body: String,
	}

	fn capture_request_fetch(
		captured: Arc<Mutex<Option<CapturedRequest>>>,
		make_response: impl Fn() -> HttpResponse + Send + Sync + 'static,
	) -> FetchFn {
		FetchFn::new(move |request: HttpRequest| {
			let body = match &request.body {
				HttpBody::Bytes(bytes) => String::from_utf8_lossy(bytes).to_string(),
				_ => String::new(),
			};
			*captured.lock() = Some(CapturedRequest { url: request.url, headers: request.headers, body });
			let response = make_response();
			Box::pin(async move { Ok(response) })
		})
	}

	// -- test/mistral-http-transport.test.ts --------------------------------

	#[tokio::test]
	async fn serializes_sdk_style_payloads_to_the_mistral_wire_format() {
		let model = catalog_model("mistral-large-latest");
		let context = Context {
			system_prompt: Some("Be precise".to_string()),
			messages: vec![Message::User(UserMessage {
				content: UserContentField::Blocks(vec![
					TextContent::new("describe").into(),
					ImageContent::new("aGVsbG8=", "image/png").into(),
				]),
				timestamp: 1,
			})],
			tools: Some(vec![Tool {
				name: "lookup".to_string(),
				description: "Look something up".to_string(),
				parameters: json!({ "type": "object", "properties": { "query": { "type": "string" } } }),
				constrained_sampling: None,
			}]),
		};

		let callback_payload = Arc::new(Mutex::new(None::<Value>));
		let callback_response = Arc::new(Mutex::new(None::<ProviderResponse>));
		let request = Arc::new(Mutex::new(None::<CapturedRequest>));

		let mut options = base_options("secret");
		options.base.headers = Some(
			[("x-custom".to_string(), Some("value".to_string()))].into_iter().collect(),
		);
		options.max_tokens = Some(123);
		options.session_id = Some("session-1".to_string());

		let hook_payload = callback_payload.clone();
		options.base.on_payload = Some(Arc::new(move |payload: Value, _model: Model| {
			let hook_payload = hook_payload.clone();
			Box::pin(async move {
				*hook_payload.lock() = Some(payload.clone());
				let mut next = payload.as_object().cloned().expect("payload object");
				next.insert("topP".to_string(), json!(0.9));
				next.insert("randomSeed".to_string(), json!(42));
				next.insert(
					"responseFormat".to_string(),
					json!({
						"type": "json_schema",
						"jsonSchema": {
							"name": "result",
							"schemaDefinition": {
								"type": "object",
								"properties": { "maxTokens": { "type": "number" } },
							},
						},
					}),
				);
				next.insert("presencePenalty".to_string(), json!(0.1));
				next.insert("frequencyPenalty".to_string(), json!(0.2));
				next.insert("parallelToolCalls".to_string(), json!(true));
				next.insert("safePrompt".to_string(), json!(true));
				Some(Value::Object(next))
			})
		}));
		let hook_response = callback_response.clone();
		options.base.on_response = Some(Arc::new(move |response: ProviderResponse, _model: Model| {
			let hook_response = hook_response.clone();
			Box::pin(async move {
				*hook_response.lock() = Some(response);
			})
		}));

		let make_response = || {
			let mut response = sse_response(&[terminal_event("stop")]);
			response.headers.push(("x-request-id".to_string(), "request-1".to_string()));
			response
		};
		let captured = Arc::new(Mutex::new(None::<CapturedRequest>));
		options.base.fetch = Some(capture_request_fetch(captured.clone(), make_response));

		let mistral = MistralOptions {
			tool_choice: Some(MistralToolChoiceOption::Function {
				tag: MistralFunctionTag::Function,
				function: MistralNamedFunction { name: "lookup".to_string() },
			}),
			prompt_mode: Some(MistralPromptMode::Reasoning),
			reasoning_effort: Some(MistralReasoningEffort::High),
		};
		let message = stream_with_options(&model, &normalize(&context), &options, &mistral)
			.into_result()
			.await;

		assert_eq!(message.stop_reason, StopReason::Stop, "error: {:?}", message.error_message);

		let request = captured.lock().clone().expect("request captured");
		assert_eq!(request.url, "https://api.mistral.ai/v1/chat/completions");
		assert_eq!(header(&request.headers, "authorization"), Some("Bearer secret"));
		assert_eq!(header(&request.headers, "accept"), Some("text/event-stream"));
		assert_eq!(header(&request.headers, "x-affinity"), Some("session-1"));
		assert_eq!(header(&request.headers, "x-custom"), Some("value"));
		assert_eq!(header(&request.headers, "user-agent"), Some(get_pi_user_agent().as_str()));

		let payload = callback_payload.lock().clone().expect("onPayload ran");
		assert_eq!(payload.get("maxTokens"), Some(&json!(123)));
		assert_eq!(payload.get("promptMode"), Some(&json!("reasoning")));
		assert_eq!(payload.get("promptCacheKey"), Some(&json!("session-1")));
		assert_eq!(
			callback_response.lock().clone().expect("onResponse ran"),
			ProviderResponse {
				status: 200,
				headers: [
					("content-type".to_string(), "text/event-stream".to_string()),
					("x-request-id".to_string(), "request-1".to_string()),
				]
				.into_iter()
				.collect(),
			}
		);

		let wire_payload = serde_json::from_str::<Value>(&request.body).expect("request body is JSON");
		assert_eq!(wire_payload.get("max_tokens"), Some(&json!(123)));
		assert_eq!(wire_payload.get("prompt_mode"), Some(&json!("reasoning")));
		assert_eq!(wire_payload.get("reasoning_effort"), Some(&json!("high")));
		assert_eq!(
			wire_payload.get("tool_choice"),
			Some(&json!({ "type": "function", "function": { "name": "lookup" } }))
		);
		assert_eq!(wire_payload.get("prompt_cache_key"), Some(&json!("session-1")));
		assert_eq!(wire_payload.get("top_p"), Some(&json!(0.9)));
		assert_eq!(wire_payload.get("random_seed"), Some(&json!(42)));
		assert_eq!(wire_payload.get("presence_penalty"), Some(&json!(0.1)));
		assert_eq!(wire_payload.get("frequency_penalty"), Some(&json!(0.2)));
		assert_eq!(wire_payload.get("parallel_tool_calls"), Some(&json!(true)));
		assert_eq!(wire_payload.get("safe_prompt"), Some(&json!(true)));
		assert_eq!(
			wire_payload.get("response_format"),
			Some(&json!({
				"type": "json_schema",
				"json_schema": {
					"name": "result",
					"schema": {
						"type": "object",
						"properties": { "maxTokens": { "type": "number" } },
					},
				},
			}))
		);
		assert!(wire_payload.get("maxTokens").is_none());
		assert!(wire_payload.get("promptMode").is_none());
		assert!(wire_payload.get("promptCacheKey").is_none());
		assert_eq!(
			wire_payload.get("messages"),
			Some(&json!([
				{ "role": "system", "content": "Be precise" },
				{
					"role": "user",
					"content": [
						{ "type": "text", "text": "describe" },
						{ "type": "image_url", "image_url": "data:image/png;base64,aGVsbG8=" },
					],
				},
			]))
		);
	}

	#[tokio::test]
	async fn serializes_assistant_thinking_tool_calls_and_tool_results_for_replay() {
		let model = catalog_model("mistral-large-latest");
		let context = Context {
			system_prompt: None,
			messages: vec![
				Message::Assistant(AssistantMessage {
					content: vec![
						AssistantContent::Thinking(ThinkingContent::new("reason")),
						AssistantContent::Text(TextContent::new("answer")),
						AssistantContent::ToolCall(ToolCall {
							id: "abc123456".to_string(),
							name: "lookup".to_string(),
							arguments: serde_json::from_value(json!({ "query": "pi" })).expect("arguments"),
							thought_signature: None,
							namespace: None,
						}),
					],
					api: "mistral-conversations".to_string(),
					provider: "mistral".to_string(),
					model: model.id().to_string(),
					response_model: None,
					response_id: None,
					provider_thinking_level: None,
					thinking_level: None,
					diagnostics: None,
					usage: Default::default(),
					stop_reason: StopReason::ToolUse,
					deferred: None,
					error_message: None,
					raw_stop_reason: None,
					end_turn: None,
					timestamp: 1,
				}),
				Message::ToolResult(ToolResultMessage {
					tool_call_id: "abc123456".to_string(),
					tool_name: "lookup".to_string(),
					content: vec![
						TextContent::new("found").into(),
						ImageContent::new("aGVsbG8=", "image/png").into(),
					],
					details: None,
					usage: None,
					nested_calls: None,
					is_error: false,
					timestamp: 2,
				}),
			],
			tools: None,
		};

		let captured = Arc::new(Mutex::new(None::<CapturedRequest>));
		let mut options = base_options("test");
		options.base.fetch = Some(capture_request_fetch(captured.clone(), || sse_response(&[terminal_event("stop")])));

		let message = stream(&model, &normalize(&context), &options).into_result().await;

		assert_eq!(message.stop_reason, StopReason::Stop, "error: {:?}", message.error_message);
		let request = captured.lock().clone().expect("request captured");
		let wire_payload = serde_json::from_str::<Value>(&request.body).expect("request body is JSON");
		assert_eq!(
			wire_payload.get("messages"),
			Some(&json!([
				{
					"role": "assistant",
					"prefix": false,
					"content": [
						{ "type": "thinking", "thinking": [{ "type": "text", "text": "reason" }] },
						{ "type": "text", "text": "answer" },
					],
					"tool_calls": [
						{
							"id": "abc123456",
							"type": "function",
							"function": { "name": "lookup", "arguments": "{\"query\":\"pi\"}" },
							"index": 0,
						},
					],
				},
				{
					"role": "tool",
					"tool_call_id": "abc123456",
					"name": "lookup",
					"content": [
						{ "type": "text", "text": "found" },
						{ "type": "image_url", "image_url": "data:image/png;base64,aGVsbG8=" },
					],
				},
			]))
		);
	}

	#[tokio::test]
	async fn parses_native_thinking_text_fragmented_tool_calls_and_cached_token_usage() {
		let model = catalog_model("mistral-large-latest");
		let events = vec![
			json!({
				"id": "response-1",
				"model": model.id(),
				"choices": [{
					"index": 0,
					"finish_reason": null,
					"delta": { "content": [{ "type": "thinking", "thinking": [{ "type": "text", "text": "reason" }] }] },
				}],
			}),
			json!({
				"id": "response-1",
				"model": model.id(),
				"choices": [{
					"index": 0,
					"finish_reason": null,
					"delta": { "content": [{ "type": "text", "text": "answer" }] },
				}],
			}),
			json!({
				"id": "response-1",
				"model": model.id(),
				"choices": [{
					"index": 0,
					"finish_reason": null,
					"delta": {
						"tool_calls": [
							{ "id": "abc123456", "index": 0, "function": { "name": "lookup", "arguments": "{\"query\":" } },
						],
					},
				}],
			}),
			json!({
				"id": "response-1",
				"model": model.id(),
				"choices": [{
					"index": 0,
					"finish_reason": "tool_calls",
					"delta": {
						"tool_calls": [
							{ "index": 0, "function": { "name": "", "arguments": "\"pi\"}" } },
						],
					},
				}],
				"usage": {
					"prompt_tokens": 10,
					"completion_tokens": 4,
					"total_tokens": 14,
					"prompt_tokens_details": { "cached_tokens": 3 },
				},
			}),
		];
		let mut options = base_options("test");
		options.base.fetch = Some(FetchFn::new(move |_request: HttpRequest| {
			let response = sse_response(&events);
			Box::pin(async move { Ok(response) })
		}));

		let message = stream(&model, &normalize(&user_context("hello")), &options).into_result().await;

		assert_eq!(message.stop_reason, StopReason::ToolUse, "error: {:?}", message.error_message);
		assert_eq!(message.raw_stop_reason.as_deref(), Some("tool_calls"));
		assert_eq!(message.response_id.as_deref(), Some("response-1"));
		let content = serde_json::to_value(&message.content).expect("content serializes");
		assert_eq!(
			content,
			json!([
				{ "type": "thinking", "thinking": "reason" },
				{ "type": "text", "text": "answer" },
				{ "type": "toolCall", "id": "abc123456", "name": "lookup", "arguments": { "query": "pi" } },
			])
		);
		assert_eq!(message.usage.input, 7);
		assert_eq!(message.usage.output, 4);
		assert_eq!(message.usage.cache_read, 3);
		assert_eq!(message.usage.cache_write, 0);
		assert_eq!(message.usage.total_tokens, 14);
	}

	// #9674: GLM models on Mistral send empty content deltas at the start, around tool calls,
	// and sometimes mid-thinking. They must not open blocks or split thinking.
	#[tokio::test]
	async fn ignores_empty_content_deltas() {
		let model = catalog_model("zai-glm-5-3");
		let thinking = |text: &str| json!({ "type": "thinking", "thinking": [{ "type": "text", "text": text }] });
		let tool_call = |args: &str, first: bool| {
			let mut call = json!({
				"index": 0,
				"function": { "name": if first { "read" } else { "" }, "arguments": args },
			});
			if first {
				call["id"] = json!("abc123456");
			}
			call
		};
		let deltas = vec![
			json!({ "content": "" }),
			json!({ "content": [thinking("first part,")] }),
			json!({ "content": "" }),
			json!({ "content": [{ "type": "text", "text": "" }] }),
			json!({ "content": [thinking(" second part."), { "type": "text", "text": "Reading." }] }),
			json!({ "content": "", "tool_calls": [tool_call("", true)] }),
			json!({ "content": "", "tool_calls": [tool_call("{\"path\":", false)] }),
			json!({ "content": "", "tool_calls": [tool_call("\"a.txt\"}", false)] }),
			json!({ "content": "" }),
		];
		let events = deltas
			.iter()
			.enumerate()
			.map(|(index, delta)| {
				json!({
					"id": "response-1",
					"model": model.id(),
					"choices": [{
						"index": 0,
						"finish_reason": if index == deltas.len() - 1 { json!("tool_calls") } else { Value::Null },
						"delta": delta,
					}],
				})
			})
			.collect::<Vec<_>>();

		let mut options = base_options("test");
		options.base.fetch = Some(FetchFn::new(move |_request: HttpRequest| {
			let response = sse_response(&events);
			Box::pin(async move { Ok(response) })
		}));

		let message = stream(&model, &normalize(&user_context("hello")), &options).into_result().await;

		let content = serde_json::to_value(&message.content).expect("content serializes");
		assert_eq!(
			content,
			json!([
				{ "type": "thinking", "thinking": "first part, second part." },
				{ "type": "text", "text": "Reading." },
				{ "type": "toolCall", "id": "abc123456", "name": "read", "arguments": { "path": "a.txt" } },
			])
		);
	}

	#[tokio::test]
	async fn forwards_each_parsed_sse_payload_before_normalizing_it() {
		let model = catalog_model("mistral-large-latest");
		let events = vec![
			json!({
				"id": "response-1",
				"provider_metadata": { "request": "test" },
				"choices": [{ "index": 0, "finish_reason": null, "delta": { "content": "hello" } }],
			}),
			json!({
				"id": "response-1",
				"choices": [{ "index": 0, "finish_reason": "stop", "delta": {} }],
				"usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
			}),
		];
		let received = Arc::new(Mutex::new(Vec::<Value>::new()));
		let event_models = Arc::new(Mutex::new(Vec::<Model>::new()));

		let mut options = base_options("test");
		let hook_events = received.clone();
		let hook_models = event_models.clone();
		options.on_provider_stream_event = Some(Arc::new(move |event: Value, event_model: Model| {
			let hook_events = hook_events.clone();
			let hook_models = hook_models.clone();
			Box::pin(async move {
				hook_events.lock().push(event);
				hook_models.lock().push(event_model);
			})
		}));
		let response_events = events.clone();
		options.base.fetch = Some(FetchFn::new(move |_request: HttpRequest| {
			let response = sse_response(&response_events);
			Box::pin(async move { Ok(response) })
		}));

		let result = stream(&model, &normalize(&user_context("hello")), &options).into_result().await;

		assert_eq!(received.lock().clone(), events);
		let models = event_models.lock().clone();
		assert_eq!(models, vec![model.clone(), model.clone()]);
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let content = serde_json::to_value(&result.content).expect("content serializes");
		assert_eq!(content, json!([{ "type": "text", "text": "hello" }]));
	}

	#[tokio::test]
	async fn parses_sse_and_utf8_sequences_split_across_transport_chunks() {
		let model = catalog_model("mistral-large-latest");
		let event = json!({
			"id": "response-bytewise",
			"model": model.id(),
			"choices": [{ "index": 0, "finish_reason": "stop", "delta": { "content": "héllo 🌍" } }],
			"usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 },
		});
		let mut options = base_options("test");
		options.base.fetch = Some(FetchFn::new(move |_request: HttpRequest| {
			let response = bytewise_sse_response(&event);
			Box::pin(async move { Ok(response) })
		}));

		let message = stream(&model, &normalize(&user_context("hello")), &options).into_result().await;

		assert_eq!(message.stop_reason, StopReason::Stop, "error: {:?}", message.error_message);
		let content = serde_json::to_value(&message.content).expect("content serializes");
		assert_eq!(content, json!([{ "type": "text", "text": "héllo 🌍" }]));
	}

	#[tokio::test]
	async fn honors_case_insensitive_header_overrides_and_explicit_affinity_suppression() {
		let mut model = catalog_model("mistral-large-latest");
		model.base.headers = Some(
			[
				("Authorization".to_string(), "Bearer model-key".to_string()),
				("X-Affinity".to_string(), "model-affinity".to_string()),
			]
			.into_iter()
			.collect(),
		);
		let captured = Arc::new(Mutex::new(None::<CapturedRequest>));
		let mut options = base_options("request-key");
		options.session_id = Some("automatic-affinity".to_string());
		options.base.headers = Some(
			[
				("authorization".to_string(), None),
				("x-affinity".to_string(), None),
				("User-Agent".to_string(), Some("custom-agent".to_string())),
			]
			.into_iter()
			.collect(),
		);
		options.base.fetch = Some(capture_request_fetch(captured.clone(), || sse_response(&[terminal_event("stop")])));

		stream(&model, &normalize(&user_context("hello")), &options).into_result().await;

		let request = captured.lock().clone().expect("request captured");
		assert!(!has_header(&request.headers, "authorization"));
		assert!(!has_header(&request.headers, "x-affinity"));
		assert_eq!(header(&request.headers, "user-agent"), Some("custom-agent"));
	}

	#[tokio::test]
	async fn aborts_while_waiting_for_an_sse_chunk() {
		let model = catalog_model("mistral-large-latest");
		let controller = crate::utils::abort::AbortController::new();
		let mut options = base_options("test");
		options.base.signal = Some(controller.signal());
		options.base.fetch = Some(FetchFn::new(|_request: HttpRequest| {
			let response = never_yielding_response();
			Box::pin(async move { Ok(response) })
		}));

		let handle = stream(&model, &normalize(&user_context("hello")), &options);
		controller.abort();
		let message = handle.into_result().await;

		assert_eq!(message.stop_reason, StopReason::Aborted);
	}

	#[tokio::test]
	async fn applies_the_request_timeout_while_waiting_for_an_sse_chunk() {
		let model = catalog_model("mistral-large-latest");
		let mut options = base_options("test");
		options.base.timeout_ms = Some(5);
		options.base.fetch = Some(FetchFn::new(|_request: HttpRequest| {
			let response = never_yielding_response();
			Box::pin(async move { Ok(response) })
		}));

		let message = stream(&model, &normalize(&user_context("hello")), &options).into_result().await;

		assert_eq!(message.stop_reason, StopReason::Error);
		let error_message = message.error_message.expect("error message");
		assert!(
			error_message.to_lowercase().contains("timeout"),
			"expected timeout error, got: {error_message}"
		);
	}

	#[tokio::test]
	async fn preserves_http_status_and_response_bodies_in_errors() {
		let model = catalog_model("mistral-large-latest");
		let mut options = base_options("test");
		options.base.fetch = Some(FetchFn::new(|_request: HttpRequest| {
			let response = HttpResponse {
				status: 403,
				headers: Vec::new(),
				body: HttpBody::Bytes(br#"{"message":"blocked by gateway"}"#.to_vec()),
			};
			Box::pin(async move { Ok(response) })
		}));

		let message = stream(&model, &normalize(&user_context("hello")), &options).into_result().await;

		assert_eq!(message.stop_reason, StopReason::Error);
		assert_eq!(
			message.error_message.as_deref(),
			Some(r#"Mistral API error (403): {"message":"blocked by gateway"}"#)
		);
	}

	// -- test/mistral-raw-stop-reason.test.ts --------------------------------

	/// Upstream `createFetch(finishReason)` (`\n\n` framing).
	fn raw_stop_fetch(finish_reason: &'static str) -> FetchFn {
		FetchFn::new(move |_request: HttpRequest| {
			let body = format!(
				"data: {}\n\ndata: [DONE]\n\n",
				json!({
					"id": "mistral-response-id",
					"model": "devstral-medium-latest",
					"choices": [{ "index": 0, "finish_reason": finish_reason, "delta": {} }],
					"usage": { "prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1 },
				})
			);
			let response = HttpResponse {
				status: 200,
				headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
				body: HttpBody::Bytes(body.into_bytes()),
			};
			Box::pin(async move { Ok(response) })
		})
	}

	#[tokio::test]
	async fn preserves_raw_mistral_finish_reasons_for_successful_stops() {
		let model = catalog_model("devstral-medium-latest");
		let mut options = base_options("test");
		options.base.fetch = Some(raw_stop_fetch("stop"));

		let message = stream(&model, &normalize(&user_context("hello")), &options).into_result().await;

		assert_eq!(message.stop_reason, StopReason::Stop);
		assert_eq!(message.raw_stop_reason.as_deref(), Some("stop"));
		assert_eq!(message.error_message, None);
	}

	#[tokio::test]
	async fn preserves_raw_mistral_finish_reasons_for_provider_error_stops() {
		let model = catalog_model("devstral-medium-latest");
		let mut options = base_options("test");
		options.base.fetch = Some(raw_stop_fetch("error"));

		let message = stream(&model, &normalize(&user_context("hello")), &options).into_result().await;

		assert_eq!(message.stop_reason, StopReason::Error);
		assert_eq!(message.raw_stop_reason.as_deref(), Some("error"));
		assert_eq!(message.error_message.as_deref(), Some("Provider stopped with: error"));
	}

	#[tokio::test]
	async fn treats_unknown_mistral_finish_reasons_as_provider_error_stops() {
		let model = catalog_model("devstral-medium-latest");
		let mut options = base_options("test");
		options.base.fetch = Some(raw_stop_fetch("unmapped_error"));

		let message = stream(&model, &normalize(&user_context("hello")), &options).into_result().await;

		assert_eq!(message.stop_reason, StopReason::Error);
		assert_eq!(message.raw_stop_reason.as_deref(), Some("unmapped_error"));
		assert_eq!(message.error_message.as_deref(), Some("Provider stopped with: unmapped_error"));
	}

	// -- test/mistral-reasoning-mode.test.ts ---------------------------------

	fn level(name: &str) -> ModelThinkingLevel {
		match name {
			"off" => ModelThinkingLevel::Off,
			"minimal" => ModelThinkingLevel::Minimal,
			"low" => ModelThinkingLevel::Low,
			"medium" => ModelThinkingLevel::Medium,
			"high" => ModelThinkingLevel::High,
			"xhigh" => ModelThinkingLevel::Xhigh,
			"max" => ModelThinkingLevel::Max,
			other => panic!("unknown level {other}"),
		}
	}

	fn level_map(entries: &[(&str, Option<&str>)]) -> crate::types::ThinkingLevelMap {
		entries
			.iter()
			.map(|(name, value)| (level(name), value.map(str::to_string)))
			.collect()
	}

	const NONE_HIGH_LEVELS: &[(&str, Option<&str>)] = &[
		("off", Some("none")),
		("minimal", None),
		("low", None),
		("medium", None),
		("high", Some("high")),
		("xhigh", None),
		("max", None),
	];
	const GLM_5_2_LEVELS: &[(&str, Option<&str>)] = &[
		("off", Some("none")),
		("minimal", None),
		("low", None),
		("medium", None),
		("high", Some("high")),
		("xhigh", None),
		("max", Some("max")),
	];
	const GLM_5_3_LEVELS: &[(&str, Option<&str>)] = &[
		("off", None),
		("minimal", None),
		("low", Some("low")),
		("medium", None),
		("high", Some("high")),
		("xhigh", None),
		("max", Some("max")),
	];

	/// Upstream `makeModel` (points at an unroutable port; the payload is
	/// captured by the `onPayload` hook before the request fails).
	fn make_model(id: &str, reasoning: bool, thinking_levels: Option<crate::types::ThinkingLevelMap>) -> Model {
		Model {
			base: BaseModelFields {
				id: id.to_string(),
				name: id.to_string(),
				api: "mistral-conversations".to_string(),
				provider: "mistral".to_string(),
				base_url: "http://127.0.0.1:9".to_string(),
				input: vec![ModelInput::Text],
				input_limits: None,
				cost: ModelCost {
					rates: ModelCostRates { input: 0.0, output: 0.0, cache_read: 0.0, cache_write: 0.0 },
					tiers: None,
				},
				headers: None,
			},
			model_type: None,
			reasoning,
			thinking_level_map: thinking_levels,
			prompt_cache: None,
			context_window: 128_000,
			max_tokens: 16_384,
			sampling_params: None,
			sampling_params_by_thinking_level: None,
			compat: None,
		}
	}

	/// Upstream `capturePayload`: stream the request, capture the payload the
	/// hook saw, and return the reasoning-relevant fields.
	async fn capture_payload(model: &Model, options: &SimpleStreamOptions) -> Value {
		let captured = Arc::new(Mutex::new(None::<Value>));
		let mut options = options.clone();
		options.base.base.api_key = Some("fake-key".to_string());
		let hook_captured = captured.clone();
		options.base.base.on_payload = Some(Arc::new(move |payload: Value, _model: Model| {
			let hook_captured = hook_captured.clone();
			Box::pin(async move {
				*hook_captured.lock() = Some(payload.clone());
				Some(payload)
			})
		}));

		stream_simple(model, &normalize(&user_context("Hello")), &options).into_result().await;

		let payload = captured.lock().clone();
		payload.expect("payload captured before request failure")
	}

	#[tokio::test]
	async fn uses_prompt_mode_for_reasoning_models_without_a_thinking_level_map() {
		let model = make_model("magistral-medium-latest", true, None);
		let mut options = SimpleStreamOptions::default();
		options.reasoning = Some(crate::types::ThinkingLevel::Medium);

		let payload = capture_payload(&model, &options).await;

		assert_eq!(payload.get("promptMode"), Some(&json!("reasoning")));
		assert_eq!(payload.get("reasoningEffort"), None);
	}

	#[tokio::test]
	async fn omits_reasoning_controls_for_magistral_when_thinking_is_off() {
		let model = make_model("magistral-medium-latest", true, None);
		let options = SimpleStreamOptions::default();

		let payload = capture_payload(&model, &options).await;

		assert_eq!(payload.get("promptMode"), None);
		assert_eq!(payload.get("reasoningEffort"), None);
	}

	// Regression for #8700 and #9375: Medium and GLM-5.2 ignore Magistral's prompt_mode.
	#[tokio::test]
	async fn uses_reasoning_effort_when_thinking_is_enabled() {
		for (model_id, map) in [
			("mistral-small-2603", NONE_HIGH_LEVELS),
			("mistral-medium-latest", NONE_HIGH_LEVELS),
			("zai-glm-5-2", GLM_5_2_LEVELS),
		] {
			let model = make_model(model_id, true, Some(level_map(map)));
			let mut options = SimpleStreamOptions::default();
			options.reasoning = Some(crate::types::ThinkingLevel::High);

			let payload = capture_payload(&model, &options).await;

			assert_eq!(payload.get("reasoningEffort"), Some(&json!("high")), "model {model_id}");
			assert_eq!(payload.get("promptMode"), None, "model {model_id}");
		}
	}

	#[tokio::test]
	async fn clamps_unsupported_levels_to_a_supported_effort() {
		for (model_id, map) in [
			("mistral-small-2603", NONE_HIGH_LEVELS),
			("mistral-medium-latest", NONE_HIGH_LEVELS),
			("zai-glm-5-2", GLM_5_2_LEVELS),
		] {
			let model = make_model(model_id, true, Some(level_map(map)));
			let mut options = SimpleStreamOptions::default();
			options.reasoning = Some(crate::types::ThinkingLevel::Low);

			let payload = capture_payload(&model, &options).await;

			assert_eq!(payload.get("reasoningEffort"), Some(&json!("high")), "model {model_id}");
		}
	}

	#[tokio::test]
	async fn sends_reasoning_effort_none_when_thinking_is_off() {
		for (model_id, map) in [
			("mistral-small-2603", NONE_HIGH_LEVELS),
			("mistral-medium-latest", NONE_HIGH_LEVELS),
			("zai-glm-5-2", GLM_5_2_LEVELS),
		] {
			let model = make_model(model_id, true, Some(level_map(map)));
			let options = SimpleStreamOptions::default();

			let payload = capture_payload(&model, &options).await;

			assert_eq!(payload.get("reasoningEffort"), Some(&json!("none")), "model {model_id}");
			assert_eq!(payload.get("promptMode"), None, "model {model_id}");
		}
	}

	// Regression for #9678: requested levels must reach Mistral-hosted GLM models.
	#[tokio::test]
	async fn sends_max_for_glm_5_2() {
		let model = make_model("zai-glm-5-2", true, Some(level_map(GLM_5_2_LEVELS)));
		let mut options = SimpleStreamOptions::default();
		options.reasoning = Some(crate::types::ThinkingLevel::Max);

		let payload = capture_payload(&model, &options).await;

		assert_eq!(payload.get("reasoningEffort"), Some(&json!("max")));
	}

	#[tokio::test]
	async fn sends_reasoning_effort_levels_for_glm_5_3() {
		for request_level in [
			crate::types::ThinkingLevel::Low,
			crate::types::ThinkingLevel::High,
			crate::types::ThinkingLevel::Max,
		] {
			let expected = match request_level {
				crate::types::ThinkingLevel::Low => "low",
				crate::types::ThinkingLevel::High => "high",
				crate::types::ThinkingLevel::Max => "max",
				_ => unreachable!("covered levels"),
			};
			let model = make_model("zai-glm-5-3", true, Some(level_map(GLM_5_3_LEVELS)));
			let mut options = SimpleStreamOptions::default();
			options.reasoning = Some(request_level);

			let payload = capture_payload(&model, &options).await;

			assert_eq!(payload.get("reasoningEffort"), Some(&json!(expected)), "level {expected}");
			assert_eq!(payload.get("promptMode"), None, "level {expected}");
		}
	}

	#[tokio::test]
	async fn maps_medium_to_high_for_glm_5_3() {
		let model = make_model("zai-glm-5-3", true, Some(level_map(GLM_5_3_LEVELS)));
		let mut options = SimpleStreamOptions::default();
		options.reasoning = Some(crate::types::ThinkingLevel::Medium);

		let payload = capture_payload(&model, &options).await;

		assert_eq!(payload.get("reasoningEffort"), Some(&json!("high")));
	}

	// Regression for #8700: reasoning controls must respect the model's reasoning capability.
	#[tokio::test]
	async fn omits_reasoning_controls_for_non_reasoning_models() {
		let model = make_model("mistral-medium-2505", false, None);
		let mut options = SimpleStreamOptions::default();
		options.reasoning = Some(crate::types::ThinkingLevel::Medium);

		let payload = capture_payload(&model, &options).await;

		assert_eq!(payload.get("reasoningEffort"), None);
		assert_eq!(payload.get("promptMode"), None);
	}

	#[tokio::test]
	async fn uses_the_session_id_as_prompt_cache_key() {
		let model = make_model("mistral-large-latest", false, None);
		let mut options = SimpleStreamOptions::default();
		options.base.session_id = Some("session-123".to_string());

		let payload = capture_payload(&model, &options).await;

		assert_eq!(payload.get("promptCacheKey"), Some(&json!("session-123")));
	}

	#[tokio::test]
	async fn omits_prompt_cache_key_when_cache_retention_is_disabled() {
		let model = make_model("mistral-large-latest", false, None);
		let mut options = SimpleStreamOptions::default();
		options.base.session_id = Some("session-123".to_string());
		options.base.cache_retention = Some(CacheRetention::None);

		let payload = capture_payload(&model, &options).await;

		assert_eq!(payload.get("promptCacheKey"), None);
	}

	// -- test/mistral-tool-schema.test.ts ------------------------------------

	#[tokio::test]
	async fn strips_typebox_symbol_keys_before_the_sdk_validates_tool_schemas() {
		let mut model = catalog_model("devstral-medium-latest");
		model.base.base_url = "http://127.0.0.1:9".to_string();
		let context = Context {
			system_prompt: None,
			messages: vec![Message::User(UserMessage {
				content: UserContentField::Text("Hi".to_string()),
				timestamp: 1,
			})],
			tools: Some(vec![Tool {
				name: "inspect_schema".to_string(),
				description: "Inspect the schema".to_string(),
				parameters: json!({
					"type": "object",
					"properties": {
						"nested": {
							"type": "object",
							"properties": { "value": { "type": "string" } },
						},
					},
				}),
				constrained_sampling: Some(ConstrainedSamplingSetting::Config(
					ConstrainedSamplingConfig::JsonSchema { strict: StrictSchemaMode::Require },
				)),
			}]),
		};
		let captured = Arc::new(Mutex::new(None::<Value>));
		let mut options = base_options("fake-key");
		let hook_captured = captured.clone();
		options.base.on_payload = Some(Arc::new(move |payload: Value, _model: Model| {
			let hook_captured = hook_captured.clone();
			Box::pin(async move {
				*hook_captured.lock() = Some(payload.clone());
				Some(payload)
			})
		}));

		// Upstream `complete()` drives `stream` with plain stream options.
		let response = stream(&model, &normalize(&context), &options).into_result().await;

		let payload = captured.lock().clone().expect("payload captured");
		let tools = payload.get("tools").and_then(Value::as_array).expect("tools present");
		assert_eq!(tools.len(), 1);
		let function = &tools[0]["function"];
		assert_eq!(function.get("strict"), Some(&json!(true)));
		let parameters = function.get("parameters").expect("parameters present");
		// TypeBox symbol keys do not exist in the Rust JSON world (crate
		// deviation D6); the structural assertions cover what the upstream
		// `Object.getOwnPropertySymbols` checks verified.
		let properties = parameters.get("properties").expect("properties present");
		assert!(properties.get("nested").is_some());

		assert_eq!(response.stop_reason, StopReason::Error);
		let error_message = response.error_message.unwrap_or_default();
		assert!(!error_message.contains("Input validation failed"));
	}
}

