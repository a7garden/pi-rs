//! Port of `packages/ai/src/api/google-shared.ts` — shared utilities for the
//! Google Generative AI and Google Vertex API adapters.
//!
//! Wire shapes (`Content`/`Part`/`FunctionCall`/`FunctionResponse`/
//! `ThinkingConfig`/`GenerateContentParameters` and the streamed
//! `GenerateContentResponse`) are mirrored as serde types here, matching the
//! `@google/genai` JSON exactly (camelCase keys, optional members omitted).
//!
//! # Deviations (file ledger)
//!
//! - **D15** — `@google/genai` enum types (`ThinkingLevel`,
//!   `FunctionCallingConfigMode`, `FinishReason`) become serde enums with the
//!   identical string values; unknown finish reasons are carried as
//!   [`FinishReason::Unknown`] so `mapStopReason`'s exhaustive-throw
//!   (`Unhandled stop reason: …`) stays reachable.
//! - **D16** — `retryGoogleRequest`'s error normalization (attaching a missing
//!   `headers` property to SDK `ApiError`s) is a no-op here: the crate's
//!   [`crate::http::ProviderHttpError`] always carries `status` and `headers`,
//!   so [`retry_google_request`] delegates straight to
//!   [`crate::utils::provider_retry::retry_provider_request`] with the same
//!   observable retry policy.
//! - **D17** — SSE decoding reproduces `@google/genai`'s
//!   `processStreamResponse` (blank-line-delimited `data:` events, per-chunk
//!   inline `error` probe, "Incomplete JSON segment at the end") with one
//!   observable difference: a chunk payload that fails `JSON.parse` surfaces
//!   as `Failed to parse stream chunk: …` where the SDK rethrows the engine's
//!   `SyntaxError` message.

use std::collections::BTreeSet;
use std::future::Future;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::http::{BodyStream, ProviderHttpError};
use crate::models::clamp_thinking_level;
use crate::types::{
	AssistantMessage, Model, StopReason, StreamOptions, ThinkingBudgets, ThinkingLevel, Tool, TranscriptContext,
};
use crate::utils::provider_retry::{retry_provider_request, ProviderRetryOptions};
use crate::api::constrained_sampling::{
	get_json_schema_tool_parameters, resolve_json_schema_strict_sampling,
};
use crate::api::transform_messages::transform_messages;
use crate::utils::provider_retry::ProviderRequestError;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::transcript::{collapse_system_messages, without_initial_system_message};

// ---------------------------------------------------------------------------
// Thinking levels
// ---------------------------------------------------------------------------

/// Thinking level for Gemini 3 models (`GoogleApiThinkingLevel`); mirrors
/// Google's `ThinkingLevel` enum values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoogleApiThinkingLevel {
	#[serde(rename = "THINKING_LEVEL_UNSPECIFIED")]
	ThinkingLevelUnspecified,
	#[serde(rename = "MINIMAL")]
	Minimal,
	#[serde(rename = "LOW")]
	Low,
	#[serde(rename = "MEDIUM")]
	Medium,
	#[serde(rename = "HIGH")]
	High,
}

impl GoogleApiThinkingLevel {
	pub fn as_str(self) -> &'static str {
		match self {
			GoogleApiThinkingLevel::ThinkingLevelUnspecified => "THINKING_LEVEL_UNSPECIFIED",
			GoogleApiThinkingLevel::Minimal => "MINIMAL",
			GoogleApiThinkingLevel::Low => "LOW",
			GoogleApiThinkingLevel::Medium => "MEDIUM",
			GoogleApiThinkingLevel::High => "HIGH",
		}
	}
}

/// The `@google/genai` `ThinkingLevel` enum values the SDK sends on the wire
/// (`GoogleSdkThinkingLevel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoogleSdkThinkingLevel {
	#[serde(rename = "THINKING_LEVEL_UNSPECIFIED")]
	ThinkingLevelUnspecified,
	#[serde(rename = "MINIMAL")]
	Minimal,
	#[serde(rename = "LOW")]
	Low,
	#[serde(rename = "MEDIUM")]
	Medium,
	#[serde(rename = "HIGH")]
	High,
}

impl GoogleSdkThinkingLevel {
	pub fn as_str(self) -> &'static str {
		match self {
			GoogleSdkThinkingLevel::ThinkingLevelUnspecified => "THINKING_LEVEL_UNSPECIFIED",
			GoogleSdkThinkingLevel::Minimal => "MINIMAL",
			GoogleSdkThinkingLevel::Low => "LOW",
			GoogleSdkThinkingLevel::Medium => "MEDIUM",
			GoogleSdkThinkingLevel::High => "HIGH",
		}
	}
}

/// `ResolvedGoogleThinkingLevel` — `Exclude<ThinkingLevel, "xhigh" | "max">`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedGoogleThinkingLevel {
	Minimal,
	Low,
	Medium,
	High,
}

/// `GOOGLE_SDK_THINKING_LEVEL_MAP` (identity over the shared string values).
fn google_sdk_thinking_level_map(level: GoogleApiThinkingLevel) -> GoogleSdkThinkingLevel {
	match level {
		GoogleApiThinkingLevel::ThinkingLevelUnspecified => GoogleSdkThinkingLevel::ThinkingLevelUnspecified,
		GoogleApiThinkingLevel::Minimal => GoogleSdkThinkingLevel::Minimal,
		GoogleApiThinkingLevel::Low => GoogleSdkThinkingLevel::Low,
		GoogleApiThinkingLevel::Medium => GoogleSdkThinkingLevel::Medium,
		GoogleApiThinkingLevel::High => GoogleSdkThinkingLevel::High,
	}
}

/// Resolve a supported pi level or model-specific Google mapping to a standard
/// Google level (`resolveGoogleThinkingLevel`).
pub fn resolve_google_thinking_level(model: &Model, level: ThinkingLevel) -> Result<ResolvedGoogleThinkingLevel, String> {
	let mapped = model
		.thinking_level_map
		.as_ref()
		.and_then(|map| map.get(&crate::types::ModelThinkingLevel::from(level)));
	let resolved_level = match mapped {
		Some(Some(value)) => value.to_lowercase(),
		_ => level_name(level).to_string(),
	};
	match resolved_level.as_str() {
		"minimal" => Ok(ResolvedGoogleThinkingLevel::Minimal),
		"low" => Ok(ResolvedGoogleThinkingLevel::Low),
		"medium" => Ok(ResolvedGoogleThinkingLevel::Medium),
		"high" => Ok(ResolvedGoogleThinkingLevel::High),
		_ => Err(format!(
			"Unsupported Google thinking level mapping for {}/{}: {} -> {}",
			model.provider(),
			model.id(),
			level_name(level),
			mapped_js_string(mapped),
		)),
	}
}

/// `String(mapped)` where `mapped` is `level | undefined`.
fn mapped_js_string(mapped: Option<&Option<String>>) -> String {
	match mapped {
		None => "undefined".to_string(),
		Some(None) => "null".to_string(),
		Some(Some(value)) => value.clone(),
	}
}

/// Lowercase pi level name (`String(level)` for the string-literal union).
fn level_name(level: ThinkingLevel) -> &'static str {
	match level {
		ThinkingLevel::Minimal => "minimal",
		ThinkingLevel::Low => "low",
		ThinkingLevel::Medium => "medium",
		ThinkingLevel::High => "high",
		ThinkingLevel::Xhigh => "xhigh",
		ThinkingLevel::Max => "max",
	}
}

/// Whether this model uses Gemini's discrete `thinkingLevel` control instead
/// of the token-based `thinkingBudget` control (`usesGoogleThinkingLevel`).
/// Supported levels come from the model's `thinkingLevelMap`; this only
/// selects the Google wire format.
pub fn uses_google_thinking_level(model: &Model) -> bool {
	let id = model.id().to_lowercase();
	// Match Gemini 3 Pro/Flash IDs with or without a minor version, such as
	// gemini-3-flash-preview, gemini-3.1-pro-preview, and gemini-3.8-flash
	// (`/gemini-3(?:\.\d+)?-(?:pro|flash)/`).
	if let Some(rest) = id.strip_prefix("gemini-3") {
		let after_version = match rest.strip_prefix('.') {
			Some(after_dot) => {
				let digits: String = after_dot.chars().take_while(|c| c.is_ascii_digit()).collect();
				if digits.is_empty() {
					None
				} else {
					Some(&after_dot[digits.len()..])
				}
			}
			None => Some(rest),
		};
		if let Some(after) = after_version {
			if after.starts_with("-pro") || after.starts_with("-flash") {
				return true;
			}
		}
	}
	if id == "gemini-flash-latest" || id == "gemini-flash-lite-latest" {
		return true;
	}
	// Match both hosted Gemma 4 naming forms: gemma-4-* and gemma4-*.
	id.contains("gemma-4") || id.contains("gemma4")
}

/// `toGoogleThinkingLevel`.
pub fn to_google_thinking_level(level: ResolvedGoogleThinkingLevel) -> GoogleApiThinkingLevel {
	match level {
		ResolvedGoogleThinkingLevel::Minimal => GoogleApiThinkingLevel::Minimal,
		ResolvedGoogleThinkingLevel::Low => GoogleApiThinkingLevel::Low,
		ResolvedGoogleThinkingLevel::Medium => GoogleApiThinkingLevel::Medium,
		ResolvedGoogleThinkingLevel::High => GoogleApiThinkingLevel::High,
	}
}

/// `toGoogleSdkThinkingLevel`.
pub fn to_google_sdk_thinking_level(level: GoogleApiThinkingLevel) -> GoogleSdkThinkingLevel {
	google_sdk_thinking_level_map(level)
}

/// `getDisabledGoogleThinkingConfig`.
pub fn get_disabled_google_thinking_config(model: &Model) -> GoogleThinkingConfig {
	if !uses_google_thinking_level(model) {
		return GoogleThinkingConfig { thinking_budget: Some(0), ..Default::default() };
	}

	let fallback = clamp_thinking_level(model, crate::types::ModelThinkingLevel::Off);
	if fallback == crate::types::ModelThinkingLevel::Off {
		return GoogleThinkingConfig { thinking_budget: Some(0), ..Default::default() };
	}

	let resolved_level = match resolve_google_thinking_level(model, level_from_model_level(fallback)) {
		Ok(level) => level,
		// The clamp only returns levels the model supports; the mapping is
		// present by construction.
		Err(message) => unreachable!("clamped level must resolve: {message}"),
	};
	let api_level = to_google_thinking_level(resolved_level);
	GoogleThinkingConfig {
		thinking_level: Some(to_google_sdk_thinking_level(api_level)),
		..Default::default()
	}
}

fn level_from_model_level(level: crate::types::ModelThinkingLevel) -> ThinkingLevel {
	match level {
		crate::types::ModelThinkingLevel::Minimal => ThinkingLevel::Minimal,
		crate::types::ModelThinkingLevel::Low => ThinkingLevel::Low,
		crate::types::ModelThinkingLevel::Medium => ThinkingLevel::Medium,
		crate::types::ModelThinkingLevel::High => ThinkingLevel::High,
		crate::types::ModelThinkingLevel::Xhigh => ThinkingLevel::Xhigh,
		crate::types::ModelThinkingLevel::Max => ThinkingLevel::Max,
		crate::types::ModelThinkingLevel::Off => ThinkingLevel::Low,
	}
}

// ---------------------------------------------------------------------------
// Thought signatures
// ---------------------------------------------------------------------------

/// Determines whether a streamed Gemini `Part` should be treated as "thinking"
/// (`isThinkingPart`).
///
/// Protocol note (Gemini / Vertex AI thought signatures):
/// - `thought: true` is the definitive marker for thinking content (thought
///   summaries).
/// - `thoughtSignature` is an encrypted representation of the model's internal
///   thought process used to preserve reasoning context across multi-turn
///   interactions.
/// - `thoughtSignature` can appear on ANY part type (text, functionCall, etc.)
///   — it does NOT indicate the part itself is thinking content.
///
/// See: <https://ai.google.dev/gemini-api/docs/thought-signatures>
pub fn is_thinking_part(part: &StreamPart) -> bool {
	part.thought == Some(true)
}

/// Retain thought signatures during streaming (`retainThoughtSignature`). Some
/// backends only send `thoughtSignature` on the first delta for a given
/// block; later deltas may omit it. This preserves the last non-empty
/// signature for the current block without merging signatures across distinct
/// response parts.
pub fn retain_thought_signature(existing: Option<String>, incoming: Option<&str>) -> Option<String> {
	match incoming {
		Some(incoming) if !incoming.is_empty() => Some(incoming.to_string()),
		_ => existing,
	}
}

/// Thought signatures must be base64 for Google APIs (TYPE_BYTES).
fn is_valid_thought_signature(signature: Option<&str>) -> bool {
	let Some(signature) = signature else { return false };
	if signature.is_empty() {
		return false;
	}
	if signature.len() % 4 != 0 {
		return false;
	}
	let padding = signature.chars().rev().take_while(|c| *c == '=').count();
	if padding > 2 {
		return false;
	}
	let core = &signature[..signature.len() - padding];
	!core.is_empty() && core.chars().all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/')
}

/// Only keep signatures from the same provider/model and with valid base64.
fn resolve_thought_signature(is_same_provider_and_model: bool, signature: Option<&str>) -> Option<String> {
	if is_same_provider_and_model && is_valid_thought_signature(signature) {
		signature.map(str::to_string)
	} else {
		None
	}
}

// ---------------------------------------------------------------------------
// Tool call IDs
// ---------------------------------------------------------------------------

/// Models via Google APIs that require explicit tool call IDs in function
/// calls/responses (`requiresToolCallId`).
pub fn requires_tool_call_id(model_id: &str) -> bool {
	let gemini_major_version = get_gemini_major_version(model_id);
	model_id.starts_with("claude-")
		|| model_id.starts_with("gpt-oss-")
		|| gemini_major_version.is_some_and(|version| version >= 3)
}

fn get_gemini_major_version(model_id: &str) -> Option<u32> {
	// `^gemini(?:-live)?-(\d+)`.
	let lower = model_id.to_lowercase();
	let rest = lower.strip_prefix("gemini-live-").or_else(|| lower.strip_prefix("gemini-"))?;
	let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
	if digits.is_empty() {
		return None;
	}
	digits.parse().ok()
}

fn supports_multimodal_function_response(model_id: &str) -> bool {
	match get_gemini_major_version(model_id) {
		Some(version) => version >= 3,
		None => true,
	}
}

// ---------------------------------------------------------------------------
// convertMessages
// ---------------------------------------------------------------------------

/// Convert internal messages to Gemini `Content[]` format
/// (`convertMessages`).
pub fn convert_messages(model: &Model, context: &TranscriptContext) -> Vec<GoogleContent> {
	// Gemini has no mid-conversation system messages; the leading prompt is
	// sent as systemInstruction.
	let conversation = without_initial_system_message(collapse_system_messages(context).messages());

	let mut contents: Vec<GoogleContent> = Vec::new();
	let transformed_messages = transform_messages(&conversation, model, Some(&normalize_tool_call_id_fn(model.id())));

	for msg in transformed_messages {
		match msg {
			crate::types::Message::User(msg) => {
				let parts: Vec<GooglePart> = match msg.content {
					crate::types::UserContentField::Text(text) => {
						vec![GooglePart { text: Some(sanitize_surrogates(&text)), ..Default::default() }]
					}
					crate::types::UserContentField::Blocks(items) => items
						.into_iter()
						.map(|item| match item {
							crate::types::UserContent::Text(text) => {
								GooglePart { text: Some(sanitize_surrogates(&text.text)), ..Default::default() }
							}
							crate::types::UserContent::Image(image) => GooglePart {
								inline_data: Some(GoogleInlineData { mime_type: image.mime_type, data: image.data }),
								..Default::default()
							},
						})
						.collect(),
				};
				if parts.is_empty() {
					continue;
				}
				contents.push(GoogleContent { role: GoogleContentRole::User, parts });
			}
			crate::types::Message::Assistant(msg) => {
				let mut parts: Vec<GooglePart> = Vec::new();
				// Check if message is from same provider and model - only then keep thinking blocks
				let is_same_provider_and_model = msg.provider == model.provider() && msg.model == model.id();

				for block in msg.content {
					match block {
						crate::types::AssistantContent::Text(block) => {
							let thought_signature =
								resolve_thought_signature(is_same_provider_and_model, block.text_signature.as_deref());
							// Skip empty text blocks — unless they carry a thought signature. Gemini can
							// attach the signature to a part whose visible text is empty and requires it
							// echoed back; dropping it breaks the reasoning chain and the model
							// intermittently ends mid-task turns with a thought-only STOP (empty
							// completion, no tool call).
							if (block.text.is_empty() || block.text.trim().is_empty()) && thought_signature.is_none() {
								continue;
							}
							parts.push(GooglePart {
								text: Some(sanitize_surrogates(&block.text)),
								thought_signature,
								..Default::default()
							});
						}
						crate::types::AssistantContent::Thinking(block) => {
							// Only keep as thinking block if same provider AND same model.
							// Otherwise convert to plain text (no tags to avoid model mimicking them).
							if is_same_provider_and_model {
								let thought_signature = resolve_thought_signature(
									is_same_provider_and_model,
									block.thinking_signature.as_deref(),
								);
								// Same rule as text blocks: an empty thinking block is dropped only when
								// it carries no signature (mirrors the anthropic converter's handling).
								if (block.thinking.is_empty() || block.thinking.trim().is_empty())
									&& thought_signature.is_none()
								{
									continue;
								}
								parts.push(GooglePart {
									thought: Some(true),
									text: Some(sanitize_surrogates(&block.thinking)),
									thought_signature,
									..Default::default()
								});
							} else {
								// Cross-provider/model: the signature is unusable, empty blocks stay dropped.
								if block.thinking.is_empty() || block.thinking.trim().is_empty() {
									continue;
								}
								parts.push(GooglePart {
									text: Some(sanitize_surrogates(&block.thinking)),
									..Default::default()
								});
							}
						}
						crate::types::AssistantContent::ToolCall(block) => {
							let thought_signature = resolve_thought_signature(
								is_same_provider_and_model,
								block.thought_signature.as_deref(),
							);
							parts.push(GooglePart {
								function_call: Some(GoogleFunctionCall {
									name: block.name,
									args: block.arguments,
									id: requires_tool_call_id(model.id()).then(|| block.id),
								}),
								thought_signature,
								..Default::default()
							});
						}
					}
				}

				if parts.is_empty() {
					continue;
				}
				contents.push(GoogleContent { role: GoogleContentRole::Model, parts });
			}
			crate::types::Message::ToolResult(msg) => {
				// Extract text and image content
				let mut text_parts: Vec<&str> = Vec::new();
				for content in &msg.content {
					if let crate::types::UserContent::Text(text) = content {
						text_parts.push(&text.text);
					}
				}
				let text_result = text_parts.join("\n");
				let image_content: Vec<&crate::types::ImageContent> = if model.base.input.contains(&crate::types::ModelInput::Image) {
					msg.content
						.iter()
						.filter_map(|content| match content {
							crate::types::UserContent::Image(image) => Some(image),
							_ => None,
						})
						.collect()
				} else {
					Vec::new()
				};

				let has_text = !text_result.is_empty();
				let has_images = !image_content.is_empty();

				// Gemini 3+ models support multimodal function responses with images nested inside
				// functionResponse.parts. Claude and other non-Gemini models behind Cloud Code Assist /
				// Gemini < 3 still need a separate user image turn.
				let model_supports_multimodal_function_response = supports_multimodal_function_response(model.id());

				// Use "output" key for success, "error" key for errors as per SDK documentation
				let response_value = if has_text {
					sanitize_surrogates(&text_result)
				} else if has_images {
					"(see attached image)".to_string()
				} else {
					String::new()
				};

				let image_parts: Vec<GooglePart> = image_content
					.iter()
					.map(|image_block| GooglePart {
						inline_data: Some(GoogleInlineData {
							mime_type: image_block.mime_type.clone(),
							data: image_block.data.clone(),
						}),
						..Default::default()
					})
					.collect();

				let include_id = requires_tool_call_id(model.id());
				let mut response = serde_json::Map::new();
				if msg.is_error {
					response.insert("error".to_string(), Value::String(response_value));
				} else {
					response.insert("output".to_string(), Value::String(response_value));
				}
				let function_response = GoogleFunctionResponse {
					name: msg.tool_name.clone(),
					response,
					parts: (has_images && model_supports_multimodal_function_response)
						.then(|| image_parts.clone()),
					id: include_id.then(|| msg.tool_call_id.clone()),
				};
				let function_response_part = GooglePart {
					function_response: Some(function_response),
					..Default::default()
				};

				// Cloud Code Assist API requires all function responses to be in a single user turn.
				// Check if the last content is already a user turn with function responses and merge.
				let last_content = contents.last_mut();
				match last_content {
					Some(content)
						if content.role == GoogleContentRole::User
							&& content.parts.iter().any(|part| part.function_response.is_some()) =>
					{
						content.parts.push(function_response_part);
					}
					_ => {
						contents.push(GoogleContent { role: GoogleContentRole::User, parts: vec![function_response_part] });
					}
				}

				// For Gemini < 3, add images in a separate user message
				if has_images && !model_supports_multimodal_function_response {
					let mut parts = vec![GooglePart {
						text: Some("Tool result image:".to_string()),
						..Default::default()
					}];
					parts.extend(image_parts);
					contents.push(GoogleContent { role: GoogleContentRole::User, parts });
				}
			}
			crate::types::Message::System(_) => {}
		}
	}

	contents
}

/// Upstream `normalizeToolCallId`: keep `a-zA-Z0-9_-`, replace the rest with
/// `_`, cap at 64 chars — but only for models that require tool call IDs.
fn normalize_tool_call_id_fn(
	model_id: &str,
) -> impl Fn(&str, &Model, &AssistantMessage) -> String + '_ {
	let requires = requires_tool_call_id(model_id);
	move |id: &str, _model: &Model, _message: &AssistantMessage| {
		if !requires {
			return id.to_string();
		}
		let replaced: String = id
			.chars()
			.map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
			.collect();
		replaced.chars().take(64).collect()
	}
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

const JSON_SCHEMA_META_DECLARATIONS: [&str; 8] = [
	"$schema",
	"$id",
	"$anchor",
	"$dynamicAnchor",
	"$vocabulary",
	"$comment",
	"$defs",
	"definitions", // pre-draft-2019-09 equivalent of $defs
];

/// Strip meta-declarations from a schema obj (`sanitizeForOpenApi`).
fn sanitize_for_open_api(schema: &Value) -> Value {
	match schema {
		Value::Object(entries) => {
			let meta: BTreeSet<&str> = JSON_SCHEMA_META_DECLARATIONS.into_iter().collect();
			let mut result = serde_json::Map::new();
			for (key, value) in entries {
				if meta.contains(key.as_str()) {
					continue;
				}
				result.insert(key.clone(), sanitize_for_open_api(value));
			}
			Value::Object(result)
		}
		other => other.clone(),
	}
}

/// One `functionDeclarations` entry (`{ name, description, parameters |
/// parametersJsonSchema }`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleFunctionDeclaration {
	pub name: String,
	pub description: String,
	/// OpenAPI 3.03 subset (`useParameters`), meta keys stripped.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub parameters: Option<Value>,
	/// Full JSON Schema (default path).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub parameters_json_schema: Option<Value>,
}

/// Upstream `convertTools` return element: `{ functionDeclarations: [...] }`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GoogleToolsBlock {
	#[serde(rename = "functionDeclarations")]
	pub function_declarations: Vec<GoogleFunctionDeclaration>,
}

/// Convert tools to Gemini function declarations format (`convertTools`).
///
/// By default uses `parametersJsonSchema` which supports full JSON Schema
/// (including anyOf, oneOf, const, etc.). Set `use_parameters` to true to use
/// the legacy `parameters` field instead (OpenAPI 3.03 Schema). This is needed
/// for Cloud Code Assist with Claude models, where the API translates
/// `parameters` into Anthropic's `input_schema`.
pub fn convert_tools(
	tools: &[Tool],
	use_parameters: bool,
	supports_strict_mode: bool,
) -> Result<Option<Vec<GoogleToolsBlock>>, String> {
	if tools.is_empty() {
		return Ok(None);
	}
	let mut declarations = Vec::with_capacity(tools.len());
	for tool in tools {
		let strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)?;
		let parameters = get_json_schema_tool_parameters(tool, strict == Some(true))
			.map_err(|error| error.to_string())?;
		declarations.push(if use_parameters {
			GoogleFunctionDeclaration {
				name: tool.name.clone(),
				description: tool.description.clone(),
				parameters: Some(sanitize_for_open_api(&parameters)),
				parameters_json_schema: None,
			}
		} else {
			GoogleFunctionDeclaration {
				name: tool.name.clone(),
				description: tool.description.clone(),
				parameters: None,
				parameters_json_schema: Some(parameters),
			}
		});
	}
	Ok(Some(vec![GoogleToolsBlock { function_declarations: declarations }]))
}

/// Gemini 3+ enforces required function parameters in validated tool-calling
/// modes (`supportsGoogleStrictToolSampling`).
pub fn supports_google_strict_tool_sampling(model_id: &str) -> bool {
	match get_gemini_major_version(model_id) {
		Some(version) => version >= 3,
		None => false,
	}
}

/// Map tool choice string to Gemini `FunctionCallingConfigMode`
/// (`mapToolChoice`).
pub fn map_tool_choice(choice: GoogleToolChoice) -> FunctionCallingConfigMode {
	match choice {
		GoogleToolChoice::Auto => FunctionCallingConfigMode::Auto,
		GoogleToolChoice::None => FunctionCallingConfigMode::None,
		GoogleToolChoice::Any => FunctionCallingConfigMode::Any,
	}
}

/// `resolveGoogleFunctionCallingMode`.
pub fn resolve_google_function_calling_mode(
	tools: &[Tool],
	tool_choice: Option<GoogleToolChoice>,
	supports_strict_mode: bool,
) -> Result<Option<FunctionCallingConfigMode>, String> {
	let mut use_strict_mode = false;
	for tool in tools {
		if resolve_json_schema_strict_sampling(tool, supports_strict_mode)? == Some(true) {
			use_strict_mode = true;
			break;
		}
	}
	if matches!(tool_choice, Some(GoogleToolChoice::None) | Some(GoogleToolChoice::Any)) {
		return Ok(tool_choice.map(map_tool_choice));
	}
	if use_strict_mode {
		return Ok(Some(FunctionCallingConfigMode::Validated));
	}
	Ok(tool_choice.map(map_tool_choice))
}

/// Upstream `toolChoice` of the Google options (`"auto" | "none" | "any"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoogleToolChoice {
	Auto,
	None,
	Any,
}

/// `FunctionCallingConfigMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FunctionCallingConfigMode {
	#[serde(rename = "AUTO")]
	Auto,
	#[serde(rename = "NONE")]
	None,
	#[serde(rename = "ANY")]
	Any,
	#[serde(rename = "VALIDATED")]
	Validated,
}

// ---------------------------------------------------------------------------
// Stop reasons
// ---------------------------------------------------------------------------

/// `FinishReason` (`@google/genai` enum; unknown raw strings are carried).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
	Stop,
	MaxTokens,
	Blocklist,
	ProhibitedContent,
	Spii,
	Safety,
	ImageSafety,
	ImageProhibitedContent,
	ImageRecitation,
	ImageOther,
	Recitation,
	FinishReasonUnspecified,
	Other,
	Language,
	MalformedFunctionCall,
	UnexpectedToolCall,
	TooManyToolCalls,
	NoImage,
	/// A raw value outside the pinned SDK enum (deviation D15).
	Unknown(String),
}

impl FinishReason {
	pub fn parse(reason: &str) -> FinishReason {
		match reason {
			"STOP" => FinishReason::Stop,
			"MAX_TOKENS" => FinishReason::MaxTokens,
			"BLOCKLIST" => FinishReason::Blocklist,
			"PROHIBITED_CONTENT" => FinishReason::ProhibitedContent,
			"SPII" => FinishReason::Spii,
			"SAFETY" => FinishReason::Safety,
			"IMAGE_SAFETY" => FinishReason::ImageSafety,
			"IMAGE_PROHIBITED_CONTENT" => FinishReason::ImageProhibitedContent,
			"IMAGE_RECITATION" => FinishReason::ImageRecitation,
			"IMAGE_OTHER" => FinishReason::ImageOther,
			"RECITATION" => FinishReason::Recitation,
			"FINISH_REASON_UNSPECIFIED" => FinishReason::FinishReasonUnspecified,
			"OTHER" => FinishReason::Other,
			"LANGUAGE" => FinishReason::Language,
			"MALFORMED_FUNCTION_CALL" => FinishReason::MalformedFunctionCall,
			"UNEXPECTED_TOOL_CALL" => FinishReason::UnexpectedToolCall,
			"TOO_MANY_TOOL_CALLS" => FinishReason::TooManyToolCalls,
			"NO_IMAGE" => FinishReason::NoImage,
			other => FinishReason::Unknown(other.to_string()),
		}
	}

	/// The raw wire string (`candidate.finishReason`).
	pub fn raw(&self) -> &str {
		match self {
			FinishReason::Stop => "STOP",
			FinishReason::MaxTokens => "MAX_TOKENS",
			FinishReason::Blocklist => "BLOCKLIST",
			FinishReason::ProhibitedContent => "PROHIBITED_CONTENT",
			FinishReason::Spii => "SPII",
			FinishReason::Safety => "SAFETY",
			FinishReason::ImageSafety => "IMAGE_SAFETY",
			FinishReason::ImageProhibitedContent => "IMAGE_PROHIBITED_CONTENT",
			FinishReason::ImageRecitation => "IMAGE_RECITATION",
			FinishReason::ImageOther => "IMAGE_OTHER",
			FinishReason::Recitation => "RECITATION",
			FinishReason::FinishReasonUnspecified => "FINISH_REASON_UNSPECIFIED",
			FinishReason::Other => "OTHER",
			FinishReason::Language => "LANGUAGE",
			FinishReason::MalformedFunctionCall => "MALFORMED_FUNCTION_CALL",
			FinishReason::UnexpectedToolCall => "UNEXPECTED_TOOL_CALL",
			FinishReason::TooManyToolCalls => "TOO_MANY_TOOL_CALLS",
			FinishReason::NoImage => "NO_IMAGE",
			FinishReason::Unknown(raw) => raw,
		}
	}
}

/// Map Gemini `FinishReason` to our `StopReason` (`mapStopReason`). `Err`
/// reproduces the upstream exhaustive-match throw.
pub fn map_stop_reason(reason: &FinishReason) -> Result<StopReason, String> {
	match reason {
		FinishReason::Stop => Ok(StopReason::Stop),
		FinishReason::MaxTokens => Ok(StopReason::Length),
		FinishReason::Blocklist
		| FinishReason::ProhibitedContent
		| FinishReason::Spii
		| FinishReason::Safety
		| FinishReason::ImageSafety
		| FinishReason::ImageProhibitedContent
		| FinishReason::ImageRecitation
		| FinishReason::ImageOther
		| FinishReason::Recitation
		| FinishReason::FinishReasonUnspecified
		| FinishReason::Other
		| FinishReason::Language
		| FinishReason::MalformedFunctionCall
		| FinishReason::UnexpectedToolCall
		| FinishReason::TooManyToolCalls
		| FinishReason::NoImage => Ok(StopReason::Error),
		FinishReason::Unknown(raw) => Err(format!("Unhandled stop reason: {raw}")),
	}
}

/// Map string finish reason to our `StopReason` (for raw API responses,
/// `mapStopReasonString`).
pub fn map_stop_reason_string(reason: &str) -> StopReason {
	match reason {
		"STOP" => StopReason::Stop,
		"MAX_TOKENS" => StopReason::Length,
		_ => StopReason::Error,
	}
}

// ---------------------------------------------------------------------------
// Retry
// ---------------------------------------------------------------------------

/// Run a Google GenAI request with the shared provider retry policy
/// (408/409/429/5xx with backoff, honoring retry-after), mirroring how the
/// Anthropic and OpenAI adapters wrap their initial request in
/// `retryProviderRequest` (`retryGoogleRequest`).
///
/// Deviation D16: upstream normalizes SDK `ApiError`s by attaching a missing
/// `headers` property before rethrowing; the crate error type always carries
/// both fields, so the wrapper delegates directly.
pub async fn retry_google_request<T, F, Fut>(mut request: F, options: ProviderRetryOptions) -> Result<T, ProviderHttpError>
	where
	F: FnMut() -> Fut,
	Fut: Future<Output = Result<T, ProviderHttpError>>,
{
	retry_provider_request(|| request(), options).await
}

// ---------------------------------------------------------------------------
// Request wire shapes (`GenerateContentParameters` mirror)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum GoogleContentRole {
	User,
	Model,
}

#[derive(Debug, Clone, PartialEq, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GoogleInlineData {
	pub mime_type: String,
	pub data: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct GoogleFunctionCall {
	pub name: String,
	pub args: crate::types::JsonObject,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct GoogleFunctionResponse {
	pub name: String,
	pub response: crate::types::JsonObject,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub parts: Option<Vec<GooglePart>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub id: Option<String>,
}

/// One request `Part`.
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GooglePart {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub text: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub thought: Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub thought_signature: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub inline_data: Option<GoogleInlineData>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub function_call: Option<GoogleFunctionCall>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub function_response: Option<GoogleFunctionResponse>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GoogleContent {
	pub role: GoogleContentRole,
	pub parts: Vec<GooglePart>,
}

/// `ThinkingConfig`.
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GoogleThinkingConfig {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub include_thoughts: Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub thinking_level: Option<GoogleSdkThinkingLevel>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub thinking_budget: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GoogleFunctionCallingConfig {
	pub mode: FunctionCallingConfigMode,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleToolConfig {
	pub function_calling_config: GoogleFunctionCallingConfig,
}

/// `GenerateContentConfig` (pi only sets these members).
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GoogleConfig {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub temperature: Option<f64>,
	#[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
	pub max_output_tokens: Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub system_instruction: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tools: Option<Vec<GoogleToolsBlock>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tool_config: Option<GoogleToolConfig>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub thinking_config: Option<GoogleThinkingConfig>,
}

/// `GenerateContentParameters`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GoogleGenerateContentParameters {
	pub model: String,
	pub contents: Vec<GoogleContent>,
	pub config: GoogleConfig,
}

// ---------------------------------------------------------------------------
// Response wire shapes (`GenerateContentResponse` mirror)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
	#[serde(default)]
	pub prompt_token_count: Option<u64>,
	#[serde(default)]
	pub candidates_token_count: Option<u64>,
	#[serde(default)]
	pub total_token_count: Option<u64>,
	#[serde(default)]
	pub cached_content_token_count: Option<u64>,
	#[serde(default)]
	pub thoughts_token_count: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StreamFunctionCall {
	#[serde(default)]
	pub id: Option<String>,
	#[serde(default)]
	pub name: Option<String>,
	#[serde(default)]
	pub args: Option<crate::types::JsonObject>,
}

/// One streamed `Part`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct StreamPart {
	#[serde(default)]
	pub text: Option<String>,
	#[serde(default)]
	pub thought: Option<bool>,
	#[serde(default)]
	pub thought_signature: Option<String>,
	#[serde(default)]
	pub function_call: Option<StreamFunctionCall>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StreamContent {
	#[serde(default)]
	pub parts: Option<Vec<StreamPart>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StreamCandidate {
	#[serde(default)]
	pub content: Option<StreamContent>,
	#[serde(default, rename = "finishReason")]
	pub finish_reason: Option<String>,
}

/// Streamed `GenerateContentResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
	#[serde(default)]
	pub response_id: Option<String>,
	#[serde(default)]
	pub candidates: Option<Vec<StreamCandidate>>,
	#[serde(default)]
	pub usage_metadata: Option<UsageMetadata>,
}

// ---------------------------------------------------------------------------
// SSE decoding (`@google/genai` processStreamResponse)
// ---------------------------------------------------------------------------

/// Upstream `THINKING_LEVEL_UNSPECIFIED` is unused here; kept for parity via
/// the enums above.

/// Streaming SSE event source over the response body. Yields each parsed
/// `data:` payload as raw JSON (the hook receives it verbatim, like the SDK's
/// chunk objects); `Err` carries the SDK-observable error strings.
pub(crate) struct GoogleSseStream {
	body: BodyStream,
	signal: Option<crate::utils::abort::AbortSignal>,
	buffer: String,
	finished: bool,
}

impl GoogleSseStream {
	pub(crate) fn new(body: BodyStream, signal: Option<crate::utils::abort::AbortSignal>) -> GoogleSseStream {
		GoogleSseStream { body, signal, buffer: String::new(), finished: false }
	}

	/// Probe a raw chunk for an inline `error` object the way the SDK does
	/// before buffering (`got status: <status>. <json>`).
	fn probe_chunk_error(chunk: &str) -> Option<String> {
		let Ok(value) = serde_json::from_str::<Value>(chunk) else {
			return None;
		};
		let Some(error) = value.get("error") else { return None };
		let code = error.get("code").and_then(Value::as_i64)?;
		if (400..600).contains(&code) {
			let status = error.get("status").and_then(Value::as_str).unwrap_or("undefined");
			return Some(format!("got status: {status}. {chunk}"));
		}
		None
	}

	/// Decode one delimiter-separated event; `Ok(None)` skips non-data events.
	fn decode_event(event: &str) -> Option<Result<Value, String>> {
		let trimmed = event.trim();
		let payload = trimmed.strip_prefix("data:")?.trim();
		match serde_json::from_str::<Value>(payload) {
			Ok(response) => Some(Ok(response)),
			Err(error) => Some(Err(format!("Failed to parse stream chunk: {error}"))),
		}
	}

	fn pump(&mut self) -> Option<Result<Value, String>> {
		for delimiter in ["\r\n\r\n", "\n\n", "\r\r"] {
			while let Some(index) = self.buffer.find(delimiter) {
				let event = self.buffer[..index].to_string();
				self.buffer.drain(..index + delimiter.len());
				if let Some(result) = Self::decode_event(&event) {
					return Some(result);
				}
			}
		}
		None
	}

	fn finish(&mut self) -> Option<Result<Value, String>> {
		self.finished = true;
		if !self.buffer.trim().is_empty() {
			return Some(Err("Incomplete JSON segment at the end".to_string()));
		}
		None
	}
}

impl futures_core::Stream for GoogleSseStream {
	type Item = Result<Value, String>;

	fn poll_next(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
		let this = self.get_mut();
		loop {
			if this.finished {
				return std::task::Poll::Ready(None);
			}
			if let Some(signal) = &this.signal {
				if signal.aborted() {
					this.finished = true;
					return std::task::Poll::Ready(Some(Err("Request was aborted".to_string())));
				}
			}
			if let Some(item) = this.pump() {
				return std::task::Poll::Ready(Some(item));
			}
			match this.body.as_mut().poll_next(cx) {
				std::task::Poll::Ready(Some(Ok(chunk))) => {
					let chunk_string = String::from_utf8_lossy(&chunk).to_string();
					if let Some(message) = GoogleSseStream::probe_chunk_error(&chunk_string) {
						this.finished = true;
						return std::task::Poll::Ready(Some(Err(message)));
					}
					this.buffer.push_str(&chunk_string);
				}
				std::task::Poll::Ready(Some(Err(error))) => {
					this.finished = true;
					return std::task::Poll::Ready(Some(Err(error.message().to_string())));
				}
				std::task::Poll::Ready(None) => {
					if let Some(item) = this.finish() {
						return std::task::Poll::Ready(Some(item));
					}
					return std::task::Poll::Ready(None);
				}
				std::task::Poll::Pending => return std::task::Poll::Pending,
			}
		}
	}
}

/// Typed view over one raw stream chunk (`GenerateContentResponse`).
pub(crate) fn parse_chunk(chunk: &Value) -> Result<GenerateContentResponse, String> {
	serde_json::from_value::<GenerateContentResponse>(chunk.clone())
		.map_err(|error| format!("Failed to parse stream chunk: {error}"))
}

// ---------------------------------------------------------------------------
// Shared budget math (upstream: one private getGoogleBudget per API module)
// ---------------------------------------------------------------------------

/// Upstream `getGoogleBudget` for `google-generative-ai` (2.5-flash-lite tier
/// included).
pub(crate) fn get_google_generative_ai_budget(
	model: &Model,
	level: ResolvedGoogleThinkingLevel,
	custom_budgets: Option<&ThinkingBudgets>,
) -> i64 {
	if let Some(custom) = custom_budgets {
		let value = match level {
			ResolvedGoogleThinkingLevel::Minimal => custom.minimal,
			ResolvedGoogleThinkingLevel::Low => custom.low,
			ResolvedGoogleThinkingLevel::Medium => custom.medium,
			ResolvedGoogleThinkingLevel::High => custom.high,
		};
		if let Some(value) = value {
			return value as i64;
		}
	}

	if model.id().contains("2.5-pro") {
		return match level {
			ResolvedGoogleThinkingLevel::Minimal => 128,
			ResolvedGoogleThinkingLevel::Low => 2048,
			ResolvedGoogleThinkingLevel::Medium => 8192,
			ResolvedGoogleThinkingLevel::High => 32768,
		};
	}

	if model.id().contains("2.5-flash-lite") {
		return match level {
			ResolvedGoogleThinkingLevel::Minimal => 512,
			ResolvedGoogleThinkingLevel::Low => 2048,
			ResolvedGoogleThinkingLevel::Medium => 8192,
			ResolvedGoogleThinkingLevel::High => 24576,
		};
	}

	if model.id().contains("2.5-flash") {
		return match level {
			ResolvedGoogleThinkingLevel::Minimal => 128,
			ResolvedGoogleThinkingLevel::Low => 2048,
			ResolvedGoogleThinkingLevel::Medium => 8192,
			ResolvedGoogleThinkingLevel::High => 24576,
		};
	}

	-1
}

/// Upstream `getGoogleBudget` for `google-vertex` (no flash-lite tier).
pub(crate) fn get_google_vertex_budget(
	model: &Model,
	level: ResolvedGoogleThinkingLevel,
	custom_budgets: Option<&ThinkingBudgets>,
) -> i64 {
	if let Some(custom) = custom_budgets {
		let value = match level {
			ResolvedGoogleThinkingLevel::Minimal => custom.minimal,
			ResolvedGoogleThinkingLevel::Low => custom.low,
			ResolvedGoogleThinkingLevel::Medium => custom.medium,
			ResolvedGoogleThinkingLevel::High => custom.high,
		};
		if let Some(value) = value {
			return value as i64;
		}
	}

	if model.id().contains("2.5-pro") {
		return match level {
			ResolvedGoogleThinkingLevel::Minimal => 128,
			ResolvedGoogleThinkingLevel::Low => 2048,
			ResolvedGoogleThinkingLevel::Medium => 8192,
			ResolvedGoogleThinkingLevel::High => 32768,
		};
	}

	if model.id().contains("2.5-flash") {
		return match level {
			ResolvedGoogleThinkingLevel::Minimal => 128,
			ResolvedGoogleThinkingLevel::Low => 2048,
			ResolvedGoogleThinkingLevel::Medium => 8192,
			ResolvedGoogleThinkingLevel::High => 24576,
		};
	}

	-1
}

/// Retry option triple lifted from `StreamOptions` (upstream `Pick<
/// StreamOptions, "maxRetries" | "maxRetryDelayMs" | "signal">`).
pub(crate) fn google_retry_options(options: &StreamOptions) -> ProviderRetryOptions {
	ProviderRetryOptions {
		max_retries: options.base.max_retries,
		max_retry_delay_ms: options.base.max_retry_delay_ms,
		signal: options.base.signal.clone(),
	}
}

// ---------------------------------------------------------------------------
// Tests (ports of test/google-shared-*.test.ts and the shared parts of
// test/google-thinking-*.test.ts)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::{
		AssistantContent, ConstrainedSamplingConfig, ConstrainedSamplingSetting, Context, ImageContent,
		StrictSchemaMode, ThinkingContent, ToolCall, UserContent, UserContentField, UserMessage,
	};
	use serde_json::json;
	use std::sync::Mutex;

	fn make_tool(parameters: Value) -> Tool {
		Tool {
			name: "test_tool".to_string(),
			description: "A test tool".to_string(),
			parameters,
			constrained_sampling: None,
		}
	}

	fn gemini3_model(provider: &str, api: &str, id: &str) -> Model {
		Model {
			base: crate::types::BaseModelFields {
				id: id.to_string(),
				name: "Gemini 3 Pro Preview".to_string(),
				api: api.to_string(),
				provider: provider.to_string(),
				base_url: "https://example.com".to_string(),
				input: vec![crate::types::ModelInput::Text],
				input_limits: None,
				cost: Default::default(),
				headers: None,
			},
			model_type: None,
			reasoning: true,
			thinking_level_map: None,
			prompt_cache: None,
			context_window: 128_000,
			max_tokens: 8192,
			sampling_params: None,
			sampling_params_by_thinking_level: None,
			compat: None,
		}
	}

	fn google_model(id: &str, thinking_level_map: Value) -> Model {
		let mut model = gemini3_model("test-google", "google-generative-ai", id);
		model.base.base_url = "https://example.invalid/v1beta".to_string();
		model.thinking_level_map = Some(serde_json::from_value(thinking_level_map).expect("thinkingLevelMap parses"));
		model
	}

	fn normalize(context: &Context) -> TranscriptContext {
		crate::utils::transcript::normalize_context(context)
	}

	fn user_context(messages: Vec<crate::types::Message>) -> Context {
		Context { system_prompt: None, messages, tools: None }
	}

	fn text_user(text: &str) -> crate::types::Message {
		crate::types::Message::User(UserMessage { content: UserContentField::Text(text.to_string()), timestamp: 1 })
	}

	fn assistant_message(
		model: (&str, &str, &str),
		content: Vec<AssistantContent>,
	) -> AssistantMessage {
		AssistantMessage {
			content,
			api: model.0.to_string(),
			provider: model.1.to_string(),
			model: model.2.to_string(),
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
		}
	}

	fn tool_result(call_id: &str, tool_name: &str, content: Vec<UserContent>) -> crate::types::Message {
		crate::types::Message::ToolResult(crate::types::ToolResultMessage {
			tool_call_id: call_id.to_string(),
			tool_name: tool_name.to_string(),
			content,
			details: None,
			usage: None,
			nested_calls: None,
			is_error: false,
			timestamp: 1,
		})
	}

	fn contents_value(contents: &[GoogleContent]) -> Value {
		serde_json::to_value(contents).expect("contents serialize")
	}

	// -- convertTools ---------------------------------------------------------

	#[test]
	fn strips_json_schema_meta_keys_from_parameters_when_use_parameters_true() {
		let tools = [make_tool(json!({
			"$schema": "http://json-schema.org/draft-07/schema#",
			"$id": "urn:bash-tool",
			"$comment": "A bash tool for demonstration",
			"$defs": { "commandDef": { "type": "string" } },
			"definitions": { "legacyDef": { "type": "number" } },
			"type": "object",
			"properties": { "command": { "type": "string" } },
			"required": ["command"],
		}))];

		let result = convert_tools(&tools, true, true).expect("convertTools");
		let decl = serde_json::to_value(&result.as_ref().expect("some").first().expect("block").function_declarations[0])
			.expect("decl");
		assert_eq!(
			decl.get("parameters").expect("parameters"),
			&json!({
				"type": "object",
				"properties": { "command": { "type": "string" } },
				"required": ["command"],
			})
		);
		assert!(decl.get("parameters").expect("parameters").get("$schema").is_none());
		assert!(decl.get("parameters").expect("parameters").get("$id").is_none());
		assert!(decl.get("parameters").expect("parameters").get("$comment").is_none());
		assert!(decl.get("parameters").expect("parameters").get("$defs").is_none());
		assert!(decl.get("parameters").expect("parameters").get("definitions").is_none());
	}

	#[test]
	fn recursively_strips_nested_json_schema_meta_keys() {
		let tools = [make_tool(json!({
			"$schema": "http://json-schema.org/draft-07/schema#",
			"type": "object",
			"properties": {
				"deep": {
					"$schema": "http://json-schema.org/draft-07/schema#",
					"$id": "urn:nested",
					"type": "string",
				},
			},
		}))];

		let result = convert_tools(&tools, true, true).expect("convertTools");
		let decl = serde_json::to_value(&result.as_ref().expect("some").first().expect("block").function_declarations[0])
			.expect("decl");
		assert_eq!(
			decl.get("parameters").expect("parameters"),
			&json!({ "type": "object", "properties": { "deep": { "type": "string" } } })
		);
	}

	#[test]
	fn preserves_ref_while_stripping_meta_keys() {
		let tools = [make_tool(json!({
			"$schema": "http://json-schema.org/draft-07/schema#",
			"type": "object",
			"properties": {
				"refProp": { "$ref": "#/$defs/someDef", "type": "string" },
			},
		}))];

		let result = convert_tools(&tools, true, true).expect("convertTools");
		let decl = serde_json::to_value(&result.as_ref().expect("some").first().expect("block").function_declarations[0])
			.expect("decl");
		assert_eq!(
			decl.get("parameters").expect("parameters"),
			&json!({ "type": "object", "properties": { "refProp": { "$ref": "#/$defs/someDef", "type": "string" } } })
		);
	}

	#[test]
	fn does_not_mutate_the_original_tool_parameters() {
		let original_parameters = json!({
			"$schema": "http://json-schema.org/draft-07/schema#",
			"type": "object",
			"properties": { "command": { "type": "string" } },
			"required": ["command"],
		});
		let tools = [make_tool(original_parameters.clone())];

		let _ = convert_tools(&tools, true, true).expect("convertTools");

		assert_eq!(
			tools[0].parameters,
			json!({
				"$schema": "http://json-schema.org/draft-07/schema#",
				"type": "object",
				"properties": { "command": { "type": "string" } },
				"required": ["command"],
			})
		);
	}

	#[test]
	fn preserves_schema_in_parameters_json_schema_when_use_parameters_false() {
		let tools = [make_tool(json!({
			"$schema": "http://json-schema.org/draft-07/schema#",
			"type": "object",
			"properties": { "command": { "type": "string" } },
			"required": ["command"],
		}))];

		let result = convert_tools(&tools, false, true).expect("convertTools");
		let decl = serde_json::to_value(&result.as_ref().expect("some").first().expect("block").function_declarations[0])
			.expect("decl");
		assert_eq!(
			decl.get("parametersJsonSchema").expect("parametersJsonSchema"),
			&json!({
				"$schema": "http://json-schema.org/draft-07/schema#",
				"type": "object",
				"properties": { "command": { "type": "string" } },
				"required": ["command"],
			})
		);
	}

	#[test]
	fn handles_tools_without_schema_gracefully() {
		let tools = [make_tool(json!({
			"type": "object",
			"properties": { "path": { "type": "string" } },
			"required": ["path"],
		}))];

		let result = convert_tools(&tools, true, true).expect("convertTools");
		let decl = serde_json::to_value(&result.as_ref().expect("some").first().expect("block").function_declarations[0])
			.expect("decl");
		assert_eq!(
			decl.get("parameters").expect("parameters"),
			&json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] })
		);
	}

	#[test]
	fn uses_validated_function_calling_for_strict_tools_on_gemini_3() {
		let mut tool = make_tool(json!({ "type": "object", "properties": {} }));
		tool.constrained_sampling =
			Some(ConstrainedSamplingSetting::Config(ConstrainedSamplingConfig::JsonSchema {
				strict: StrictSchemaMode::Require,
			}));

		assert!(supports_google_strict_tool_sampling("gemini-3.1-pro-preview"));
		assert!(!supports_google_strict_tool_sampling("gemini-2.5-pro"));
		assert_eq!(
			resolve_google_function_calling_mode(&[tool.clone()], None, true).expect("mode"),
			Some(FunctionCallingConfigMode::Validated)
		);
		let error = resolve_google_function_calling_mode(&[tool], None, false).expect_err("throws");
		assert!(
			error.contains("Tool \"test_tool\" requires JSON-schema constrained sampling"),
			"unexpected error: {error}"
		);
	}

	#[test]
	fn returns_none_for_empty_tool_list() {
		assert_eq!(convert_tools(&[], false, true).expect("convertTools"), None);
		assert_eq!(convert_tools(&[], true, true).expect("convertTools"), None);
	}

	// -- thinking signature ---------------------------------------------------

	#[test]
	fn treats_part_thought_true_as_thinking() {
		assert!(is_thinking_part(&StreamPart { thought: Some(true), ..Default::default() }));
		assert!(is_thinking_part(&StreamPart {
			thought: Some(true),
			thought_signature: Some("opaque-signature".to_string()),
			..Default::default()
		}));
	}

	#[test]
	fn does_not_treat_thought_signature_alone_as_thinking() {
		// Per Google docs, thoughtSignature is for context replay and can appear
		// on any part type. Only thought === true indicates thinking content.
		assert!(!is_thinking_part(&StreamPart {
			thought: None,
			thought_signature: Some("opaque-signature".to_string()),
			..Default::default()
		}));
		assert!(!is_thinking_part(&StreamPart {
			thought: Some(false),
			thought_signature: Some("opaque-signature".to_string()),
			..Default::default()
		}));
	}

	#[test]
	fn does_not_treat_empty_or_missing_signatures_as_thinking_if_thought_is_not_set() {
		assert!(!is_thinking_part(&StreamPart { thought: None, ..Default::default() }));
		assert!(!is_thinking_part(&StreamPart {
			thought: Some(false),
			thought_signature: Some(String::new()),
			..Default::default()
		}));
	}

	#[test]
	fn preserves_the_existing_signature_when_subsequent_deltas_omit_thought_signature() {
		let first = retain_thought_signature(None, Some("sig-1"));
		assert_eq!(first.as_deref(), Some("sig-1"));

		let second = retain_thought_signature(first, None);
		assert_eq!(second.as_deref(), Some("sig-1"));

		let third = retain_thought_signature(second, Some(""));
		assert_eq!(third.as_deref(), Some("sig-1"));
	}

	#[test]
	fn updates_the_signature_when_a_new_non_empty_signature_arrives() {
		let updated = retain_thought_signature(Some("sig-1".to_string()), Some("sig-2"));
		assert_eq!(updated.as_deref(), Some("sig-2"));
	}

	// -- requiresToolCallId ---------------------------------------------------

	#[test]
	fn requires_tool_call_id_by_model_family() {
		assert!(!requires_tool_call_id("gemini-2.5-flash"));
		assert!(requires_tool_call_id("gemini-3.6-flash"));
		assert!(requires_tool_call_id("claude-sonnet-4-5"));
		assert!(requires_tool_call_id("gpt-oss-120b"));
	}

	// -- Gemini 3 unsigned tool calls ----------------------------------------

	fn unsigned_tool_call_context(
		model: (&str, &str, &str),
		thought_signature: Option<&str>,
	) -> Context {
		let first_tool_call = AssistantContent::ToolCall(ToolCall {
			id: "call_1".to_string(),
			name: "bash".to_string(),
			arguments: serde_json::from_value(json!({ "command": "echo hi" })).expect("args"),
			thought_signature: thought_signature.map(str::to_string),
			namespace: None,
		});
		let second_tool_call = AssistantContent::ToolCall(ToolCall {
			id: "call_2".to_string(),
			name: "bash".to_string(),
			arguments: serde_json::from_value(json!({ "command": "ls -la" })).expect("args"),
			thought_signature: None,
			namespace: None,
		});
		user_context(vec![
			text_user("Hi"),
			crate::types::Message::Assistant(assistant_message(
				model,
				vec![first_tool_call, second_tool_call],
			)),
			tool_result("call_1", "bash", vec![UserContent::Text(crate::types::TextContent::new("hi"))]),
			tool_result("call_2", "bash", vec![UserContent::Text(crate::types::TextContent::new("files"))]),
		])
	}

	fn function_call_values(contents: &[GoogleContent]) -> Vec<Value> {
		contents
			.iter()
			.flat_map(|content| &content.parts)
			.filter_map(|part| part.function_call.as_ref())
			.map(|call| serde_json::to_value(call).expect("functionCall"))
			.collect()
	}

	fn function_response_values(contents: &[GoogleContent]) -> Vec<Value> {
		contents
			.iter()
			.flat_map(|content| &content.parts)
			.filter_map(|part| part.function_response.as_ref())
			.map(|response| serde_json::to_value(response).expect("functionResponse"))
			.collect()
	}

	#[test]
	fn preserves_tool_call_ids_for_gemini3_history() {
		for model in [
			gemini3_model("google", "google-generative-ai", "gemini-3-pro-preview"),
			gemini3_model("google", "google-generative-ai", "gemini-3.6-flash"),
			gemini3_model("google-vertex", "google-vertex", "gemini-3-pro-preview"),
		] {
			let context = unsigned_tool_call_context(
				(model.api(), model.provider(), model.id()),
				None,
			);
			let contents = convert_messages(&model, &normalize(&context));
			let call_ids: Vec<Value> = function_call_values(&contents)
				.into_iter()
				.filter_map(|call| call.get("id").cloned())
				.collect();
			let response_ids: Vec<Value> = function_response_values(&contents)
				.into_iter()
				.filter_map(|response| response.get("id").cloned())
				.collect();

			assert_eq!(call_ids, vec![json!("call_1"), json!("call_2")]);
			assert_eq!(response_ids, vec![json!("call_1"), json!("call_2")]);
		}
	}

	#[test]
	fn does_not_add_skip_thought_signature_validator_for_unsigned_google_gen_ai_tool_calls() {
		let model = gemini3_model("google", "google-generative-ai", "gemini-3-pro-preview");
		let context = unsigned_tool_call_context((model.api(), model.provider(), "other-model"), None);
		let contents = convert_messages(&model, &normalize(&context));

		let model_turn = contents.iter().find(|content| content.role == GoogleContentRole::Model).expect("model turn");
		let serialized = serde_json::to_string(&model_turn.parts).expect("parts");
		let function_call_parts: Vec<&GooglePart> =
			model_turn.parts.iter().filter(|part| part.function_call.is_some()).collect();
		assert_eq!(function_call_parts.len(), 2);
		assert!(function_call_parts.iter().all(|part| part.thought_signature.is_none()));
		assert!(!serialized.contains("skip_thought_signature_validator"));

		let historical_text = model_turn
			.parts
			.iter()
			.filter(|part| part.text.is_some())
			.filter(|part| part.text.as_deref().is_some_and(|text| text.contains("Historical context")))
			.count();
		assert_eq!(historical_text, 0);
	}

	#[test]
	fn does_not_add_skip_thought_signature_validator_for_unsigned_vertex_tool_calls() {
		let model = gemini3_model("google-vertex", "google-vertex", "gemini-3-pro-preview");
		let context = unsigned_tool_call_context((model.api(), model.provider(), model.id()), None);
		let contents = convert_messages(&model, &normalize(&context));
		let model_turn = contents.iter().find(|content| content.role == GoogleContentRole::Model).expect("model turn");
		let serialized = serde_json::to_string(&model_turn.parts).expect("parts");
		let function_call_parts: Vec<&GooglePart> =
			model_turn.parts.iter().filter(|part| part.function_call.is_some()).collect();

		assert_eq!(function_call_parts.len(), 2);
		assert!(function_call_parts.iter().all(|part| part.thought_signature.is_none()));
		assert!(!serialized.contains("skip_thought_signature_validator"));
	}

	#[test]
	fn preserves_valid_thought_signature_when_present_for_the_same_provider_and_model() {
		let model = gemini3_model("google", "google-generative-ai", "gemini-3-pro-preview");
		let valid_sig = "AAAAAAAAAAAAAAAAAAAAAA==";
		let context = unsigned_tool_call_context((model.api(), model.provider(), model.id()), Some(valid_sig));
		let contents = convert_messages(&model, &normalize(&context));
		let model_turn = contents.iter().find(|content| content.role == GoogleContentRole::Model).expect("model turn");
		let function_call_parts: Vec<&GooglePart> =
			model_turn.parts.iter().filter(|part| part.function_call.is_some()).collect();

		assert_eq!(function_call_parts.len(), 2);
		assert_eq!(function_call_parts[0].thought_signature.as_deref(), Some(valid_sig));
		assert_eq!(function_call_parts[1].thought_signature, None);
	}

	#[test]
	fn does_not_add_a_thought_signature_for_non_gemini_3_models() {
		let model = gemini3_model("google", "google-generative-ai", "gemini-2.5-flash");
		let context = unsigned_tool_call_context((model.api(), model.provider(), "other-model"), None);
		let contents = convert_messages(&model, &normalize(&context));
		let model_turn = contents.iter().find(|content| content.role == GoogleContentRole::Model).expect("model turn");
		let function_call_parts: Vec<&GooglePart> =
			model_turn.parts.iter().filter(|part| part.function_call.is_some()).collect();
		let function_response_parts = function_response_values(&contents);

		assert_eq!(function_call_parts.len(), 2);
		assert!(function_call_parts.iter().all(|part| part.function_call.as_ref().is_some_and(|call| call.id.is_none())));
		assert!(function_call_parts.iter().all(|part| part.thought_signature.is_none()));
		assert_eq!(function_response_parts.len(), 2);
		assert!(function_response_parts.iter().all(|response| response.get("id").is_none()));
	}

	// -- signed empty blocks --------------------------------------------------

	const VALID_SIG: &str = "AAAAAAAAAAAAAAAAAAAAAA==";

	fn signed_blocks_context(model: (&str, &str, &str), content: Vec<AssistantContent>) -> Context {
		user_context(vec![
			text_user("Hi"),
			crate::types::Message::Assistant(assistant_message(model, content)),
		])
	}

	#[test]
	fn keeps_a_signed_empty_thinking_block_so_its_signature_is_echoed_back() {
		let model = gemini3_model("google", "google-generative-ai", "gemini-3-pro-preview");
		let contents = convert_messages(
			&model,
			&normalize(&signed_blocks_context(
				(model.api(), model.provider(), model.id()),
				vec![
					AssistantContent::Thinking(ThinkingContent {
						thinking: String::new(),
						thinking_signature: Some(VALID_SIG.to_string()),
						redacted: None,
					}),
					AssistantContent::ToolCall(ToolCall {
						id: "call_1".to_string(),
						name: "bash".to_string(),
						arguments: serde_json::from_value(json!({ "command": "ls" })).expect("args"),
						thought_signature: None,
						namespace: None,
					}),
				],
			)),
		);
		let value = contents_value(&contents);
		let signed: Vec<&Value> = value
			.as_array()
			.expect("contents")
			.iter()
			.find(|content| content.get("role") == Some(&json!("model")))
			.and_then(|turn| turn.get("parts"))
			.and_then(Value::as_array)
			.expect("parts")
			.iter()
			.filter(|part| part.get("thoughtSignature") == Some(&json!(VALID_SIG)))
			.collect();
		assert_eq!(signed.len(), 1);
		assert_eq!(signed[0].get("thought"), Some(&json!(true)));
	}

	#[test]
	fn keeps_a_signed_empty_text_block_the_same_way() {
		let model = gemini3_model("google", "google-generative-ai", "gemini-3-pro-preview");
		let contents = convert_messages(
			&model,
			&normalize(&signed_blocks_context(
				(model.api(), model.provider(), model.id()),
				vec![
					AssistantContent::Text(crate::types::TextContent {
						text: String::new(),
						text_signature: Some(VALID_SIG.to_string()),
					}),
					AssistantContent::ToolCall(ToolCall {
						id: "call_1".to_string(),
						name: "bash".to_string(),
						arguments: serde_json::from_value(json!({ "command": "ls" })).expect("args"),
						thought_signature: None,
						namespace: None,
					}),
				],
			)),
		);
		let value = contents_value(&contents);
		let signed: Vec<&Value> = value
			.as_array()
			.expect("contents")
			.iter()
			.find(|content| content.get("role") == Some(&json!("model")))
			.and_then(|turn| turn.get("parts"))
			.and_then(Value::as_array)
			.expect("parts")
			.iter()
			.filter(|part| part.get("thoughtSignature") == Some(&json!(VALID_SIG)))
			.collect();
		assert_eq!(signed.len(), 1);
	}

	#[test]
	fn still_drops_unsigned_empty_blocks() {
		let model = gemini3_model("google", "google-generative-ai", "gemini-3-pro-preview");
		let contents = convert_messages(
			&model,
			&normalize(&signed_blocks_context(
				(model.api(), model.provider(), model.id()),
				vec![
					AssistantContent::Thinking(ThinkingContent {
						thinking: String::new(),
						thinking_signature: None,
						redacted: None,
					}),
					AssistantContent::Text(crate::types::TextContent {
						text: "   ".to_string(),
						text_signature: None,
					}),
					AssistantContent::ToolCall(ToolCall {
						id: "call_1".to_string(),
						name: "bash".to_string(),
						arguments: serde_json::from_value(json!({ "command": "ls" })).expect("args"),
						thought_signature: None,
						namespace: None,
					}),
				],
			)),
		);
		let value = contents_value(&contents);
		let parts = value
			.as_array()
			.expect("contents")
			.iter()
			.find(|content| content.get("role") == Some(&json!("model")))
			.and_then(|turn| turn.get("parts"))
			.and_then(Value::as_array)
			.expect("parts");
		assert_eq!(parts.len(), 1);
		assert!(parts[0].get("functionCall").is_some());
	}

	#[test]
	fn still_drops_signed_empty_blocks_from_a_different_provider_or_model() {
		let model = gemini3_model("google", "google-generative-ai", "gemini-3-pro-preview");
		let contents = convert_messages(
			&model,
			&normalize(&signed_blocks_context(
				(model.api(), model.provider(), "other-model"),
				vec![
					AssistantContent::Thinking(ThinkingContent {
						thinking: String::new(),
						thinking_signature: Some(VALID_SIG.to_string()),
						redacted: None,
					}),
					AssistantContent::Text(crate::types::TextContent {
						text: String::new(),
						text_signature: Some(VALID_SIG.to_string()),
					}),
					AssistantContent::ToolCall(ToolCall {
						id: "call_1".to_string(),
						name: "bash".to_string(),
						arguments: serde_json::from_value(json!({ "command": "ls" })).expect("args"),
						thought_signature: None,
						namespace: None,
					}),
				],
			)),
		);
		let value = contents_value(&contents);
		let parts = value
			.as_array()
			.expect("contents")
			.iter()
			.find(|content| content.get("role") == Some(&json!("model")))
			.and_then(|turn| turn.get("parts"))
			.and_then(Value::as_array)
			.expect("parts");
		assert_eq!(parts.len(), 1);
		assert!(parts[0].get("functionCall").is_some());
		let serialized = serde_json::to_string(&value).expect("serialized");
		assert!(!serialized.contains(VALID_SIG));
	}

	// -- image tool result routing -------------------------------------------

	fn image_routing_context(model: (&str, &str, &str)) -> Context {
		let tool_call = |id: &str, path: &str| AssistantContent::ToolCall(ToolCall {
			id: id.to_string(),
			name: "read".to_string(),
			arguments: serde_json::from_value(json!({ "path": path })).expect("args"),
			thought_signature: None,
			namespace: None,
		});
		user_context(vec![
			text_user("read the files"),
			crate::types::Message::Assistant(assistant_message(
				model,
				vec![
					tool_call("call_a", "a.txt"),
					tool_call("call_img", "image.png"),
					tool_call("call_b", "b.txt"),
				],
			)),
			tool_result("call_a", "read", vec![UserContent::Text(crate::types::TextContent::new("alpha text"))]),
			tool_result(
				"call_img",
				"read",
				vec![UserContent::Image(ImageContent::new("abc", "image/png"))],
			),
			tool_result("call_b", "read", vec![UserContent::Text(crate::types::TextContent::new("beta text"))]),
		])
	}

	fn image_routing_model(id: &str) -> Model {
		let mut model = gemini3_model("google", "google-generative-ai", id);
		model.base.input = vec![crate::types::ModelInput::Text, crate::types::ModelInput::Image];
		model
	}

	#[test]
	fn keeps_separate_synthetic_image_turn_for_gemini_2_google_api_models() {
		let model = image_routing_model("gemini-2.5-flash");
		let contents = convert_messages(&model, &normalize(&image_routing_context((model.api(), model.provider(), model.id()))));

		let value = contents_value(&contents);
		let array = value.as_array().expect("contents");
		assert_eq!(array.len(), 5);
		let parts_of = |index: usize| array[index].get("parts").and_then(Value::as_array).expect("parts");
		assert!(parts_of(2).iter().all(|part| part.get("functionResponse").is_some()));
		assert_eq!(parts_of(3)[0].get("text"), Some(&json!("Tool result image:")));
		assert!(parts_of(3)[1].get("inlineData").is_some());
		assert!(parts_of(4)[0].get("functionResponse").is_some());
	}

	#[test]
	fn nests_image_tool_results_for_gemini_3_google_api_models() {
		let model = image_routing_model("gemini-3-pro-preview");
		let contents = convert_messages(&model, &normalize(&image_routing_context((model.api(), model.provider(), model.id()))));

		let value = contents_value(&contents);
		let array = value.as_array().expect("contents");
		assert_eq!(array.len(), 3);
		let parts = array[2].get("parts").and_then(Value::as_array).expect("parts");
		assert_eq!(parts.len(), 3);
		let image_response = parts[1].get("functionResponse").expect("image response");
		let nested = image_response.get("parts").and_then(Value::as_array).expect("nested parts");
		assert_eq!(nested.len(), 1);
		assert!(nested[0].get("inlineData").is_some());
	}

	// -- retryGoogleRequest ---------------------------------------------------

	fn google_api_error(status: u16) -> ProviderHttpError {
		ProviderHttpError {
			message: format!("got status: {status}"),
			status: Some(status),
			headers: Vec::new(),
			body: None,
			source: None,
		}
	}

	#[tokio::test]
	async fn retries_a_headers_less_sdk_error_with_a_retryable_status() {
		let calls = Mutex::new(0u32);
		let result = retry_google_request(
			|| {
				let mut calls = calls.lock().expect("lock");
				*calls += 1;
				let first = *calls == 1;
				async move {
					if first {
						Err(google_api_error(429))
					} else {
						Ok("ok")
					}
				}
			},
			crate::utils::provider_retry::ProviderRetryOptions {
				max_retries: Some(1),
				..Default::default()
			},
		)
		.await
		.expect("retries");

		assert_eq!(result, "ok");
		assert_eq!(*calls.lock().expect("lock"), 2);
	}

	#[tokio::test]
	async fn does_not_retry_when_max_retries_is_unset() {
		let calls = Mutex::new(0u32);
		let result = retry_google_request(
			|| {
				let mut calls = calls.lock().expect("lock");
				*calls += 1;
				async move { Err::<String, _>(google_api_error(429)) }
			},
			crate::utils::provider_retry::ProviderRetryOptions::default(),
		)
		.await
		.expect_err("rejects");

		assert_eq!(result.message, "got status: 429");
		assert_eq!(*calls.lock().expect("lock"), 1);
	}

	#[tokio::test]
	async fn does_not_retry_a_non_retryable_status() {
		let calls = Mutex::new(0u32);
		let result = retry_google_request(
			|| {
				let mut calls = calls.lock().expect("lock");
				*calls += 1;
				async move { Err::<String, _>(google_api_error(400)) }
			},
			crate::utils::provider_retry::ProviderRetryOptions {
				max_retries: Some(2),
				..Default::default()
			},
		)
		.await
		.expect_err("rejects");

		assert_eq!(result.message, "got status: 400");
		assert_eq!(*calls.lock().expect("lock"), 1);
	}

	// -- resolveGoogleThinkingLevel (shared exhaustive case) ------------------

	#[test]
	fn exhaustively_resolves_supported_logical_levels_and_mapping_values() {
		let model = google_model("gemini-3.7-flash", json!({}));
		for (level, expected) in [
			(ThinkingLevel::Minimal, ResolvedGoogleThinkingLevel::Minimal),
			(ThinkingLevel::Low, ResolvedGoogleThinkingLevel::Low),
			(ThinkingLevel::Medium, ResolvedGoogleThinkingLevel::Medium),
			(ThinkingLevel::High, ResolvedGoogleThinkingLevel::High),
		] {
			assert_eq!(resolve_google_thinking_level(&model, level), Ok(expected));
		}

		for mapped in ["minimal", "low", "medium", "high", "MINIMAL", "LOW", "MEDIUM", "HIGH"] {
			let model = google_model(
				"gemini-3.7-flash",
				json!({ "high": mapped, "xhigh": mapped, "max": mapped }),
			);
			let expected = match mapped.to_lowercase().as_str() {
				"minimal" => ResolvedGoogleThinkingLevel::Minimal,
				"low" => ResolvedGoogleThinkingLevel::Low,
				"medium" => ResolvedGoogleThinkingLevel::Medium,
				_ => ResolvedGoogleThinkingLevel::High,
			};
			assert_eq!(resolve_google_thinking_level(&model, ThinkingLevel::High), Ok(expected));
			assert_eq!(resolve_google_thinking_level(&model, ThinkingLevel::Xhigh), Ok(expected));
			assert_eq!(resolve_google_thinking_level(&model, ThinkingLevel::Max), Ok(expected));
		}

		let invalid_model = google_model("gemini-3.7-flash", json!({ "xhigh": "extreme" }));
		assert_eq!(
			resolve_google_thinking_level(&invalid_model, ThinkingLevel::Xhigh),
			Err("Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: xhigh -> extreme"
				.to_string())
		);
		assert_eq!(
			resolve_google_thinking_level(&google_model("gemini-3.7-flash", json!({})), ThinkingLevel::Max),
			Err("Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: max -> undefined"
				.to_string())
		);
	}
}

