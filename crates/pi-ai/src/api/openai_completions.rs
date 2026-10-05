//! Port of `packages/ai/src/api/openai-completions.ts`.
//!
//! The upstream module drives the pinned `openai` SDK (`client.chat.completions.create`
//! with `stream: true`). The Rust port hand-rolls the equivalent wire behavior:
//! `POST {baseUrl}/chat/completions` with `Authorization: Bearer`, SSE
//! `data:` framing and `[DONE]` termination, byte-compatible request bodies.
//!
//! Deviations:
//! - D15: The `openai` SDK is replaced by the [`crate::http`] seam. Non-2xx
//!   responses become a structured HTTP error (status + headers + body) so
//!   [`crate::utils::provider_retry::retry_provider_request`] and
//!   [`crate::utils::error_body::format_provider_error`] keep the upstream
//!   semantics (SDK errors carry `status`/`headers`/parsed body).
//! - D16: `api/github-copilot-headers.ts` has no Rust module yet, so its three
//!   tiny pure functions (`inferCopilotInitiator`, `hasCopilotVisionInput`,
//!   `buildCopilotDynamicHeaders`) are ported inline below, verbatim.
//! - D17: Cross-module dependencies expected from sibling slices (upstream
//!   signatures in parentheses):
//!   - `crate::api::transform_messages::transform_messages(messages: &[Message],
//!     model, normalize_tool_call_id: Option<&dyn Fn(&str, &AssistantMessage)
//!     -> String>) -> Vec<Message>` (`transformMessages`)
//!   - `crate::api::constrained_sampling::{GrammarToolInputJsonBuffer,
//!     GrammarConstrainedSampling (format: String "lark"|"regex"),
//!     create_grammar_tool_input_properties -> Result<BTreeMap, String>,
//!     get_grammar_tool_input, append_grammar_tool_input_json_delta,
//!     get_json_schema_tool_parameters(tool, strict: bool),
//!     resolve_grammar_constrained_sampling,
//!     resolve_json_schema_strict_sampling}` (same names upstream)
//!   - `crate::api::openai_prompt_cache::clamp_openai_prompt_cache_key`
//!     (`clampOpenAIPromptCacheKey`)
//!   - `crate::api::simple_options::{build_base_options,
//!     resolve_sampling_params, thinking_budget_for_level,
//!     clamp_thinking_budget_to_answer_room}` (same names upstream)
//!   - `crate::utils::provider_retry::{retry_provider_request,
//!     ProviderRetryOptions}` (`retryProviderRequest`); request failures are
//!     the status-aware [`crate::http::ProviderHttpError`].
//!   - `crate::utils::error_body::{normalize_provider_error,
//!     format_provider_error}` (`normalizeProviderError`,
//!     `formatProviderError`).
//! - D18: `parseChunkUsage` always materializes `usage.reasoning` upstream
//!   (`x || 0`), so the Rust port sets `Some(value)` including `Some(0)`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::http::{FetchFn, HttpBody, HttpRequest, ProviderHttpError};
use crate::models::{calculate_cost, ProviderStreams};
use crate::types::{
	AnyModel, AssistantContent, AssistantMessage, AssistantMessageEventStream, CacheControlFormat,
	CacheRetention, ChatTemplateKwargValue, ChatTemplateVar, JsonLiteral, MaxTokensField, Message,
	Model, ModelInput, ModelThinkingLevel, ProviderResponse, SimpleStreamOptions, StopReason,
	StreamDoneReason, StreamErrorReason, StreamOptions, TextContent, ThinkingContent, ThinkingFormat,
	ThinkingLevel, ThinkingTokenBudgetField, Tool, ToolCall,
	TranscriptContext, Usage, UserContent,
};
use crate::utils::error_body::{format_provider_error, normalize_provider_error};
use crate::utils::event_stream::{done_event, error_event};
use crate::utils::hash::short_hash;
use crate::utils::headers::headers_to_record;
use crate::utils::json_parse::parse_streaming_json;
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::provider_retry::{retry_provider_request, ProviderRetryOptions};
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::text::{get_system_message_text, render_system_message_update};
use crate::utils::transcript::{
	get_declared_tools, resolve_transcript, resolve_transcript_tools, TranscriptTools,
};

// ---------------------------------------------------------------------------
// Local helpers (module-level functions upstream)
// ---------------------------------------------------------------------------

fn now_ms() -> i64 {
	std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map(|duration| duration.as_millis() as i64)
		.unwrap_or(0)
}

fn has_header(headers: Option<&BTreeMap<String, Option<String>>>, name: &str) -> bool {
	let Some(headers) = headers else { return false };
	let expected = name.to_lowercase();
	headers.iter().any(|(key, value)| {
		key.to_lowercase() == expected && value.as_deref().is_some_and(|value| !value.trim().is_empty())
	})
}

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

fn has_tool_history(messages: &[Message]) -> bool {
	for message in messages {
		match message {
			Message::ToolResult(_) => return true,
			Message::Assistant(assistant) => {
				if assistant.content.iter().any(|block| matches!(block, AssistantContent::ToolCall(_))) {
					return true;
				}
			}
			_ => {}
		}
	}
	false
}

// ---------------------------------------------------------------------------
// OpenAI reasoning details (`reasoning_details` replay metadata)
// ---------------------------------------------------------------------------

const OPENAI_COMPLETIONS_REASONING_FIELDS: [&str; 3] = ["reasoning", "reasoning_content", "reasoning_text"];

fn is_reasoning_detail_object(detail: &Value) -> bool {
	detail.is_object()
}

fn has_valid_common_reasoning_detail_fields(candidate: &Map<String, Value>) -> bool {
	let id_ok = match candidate.get("id") {
		None | Some(Value::Null) => true,
		Some(Value::String(_)) => true,
		_ => false,
	};
	let format_ok = match candidate.get("format") {
		None => true,
		Some(Value::String(_)) => true,
		_ => false,
	};
	let index_ok = match candidate.get("index") {
		None => true,
		Some(Value::Number(_)) => true,
		_ => false,
	};
	id_ok && format_ok && index_ok
}

fn is_openai_reasoning_detail(detail: &Value) -> bool {
	if !is_reasoning_detail_object(detail) {
		return false;
	}
	let object = detail.as_object().expect("object");
	if !has_valid_common_reasoning_detail_fields(object) {
		return false;
	}
	match object.get("type").and_then(Value::as_str) {
		Some("reasoning.summary") => object.get("summary").is_some_and(Value::is_string),
		Some("reasoning.encrypted") => object.get("data").is_some_and(Value::is_string),
		Some("reasoning.text") => {
			object.get("text").is_some_and(Value::is_string)
				&& match object.get("signature") {
					None | Some(Value::Null) => true,
					Some(Value::String(_)) => true,
					_ => false,
				}
		}
		_ => false,
	}
}

fn parse_openai_reasoning_details(signature: Option<&str>) -> Option<Vec<Value>> {
	let signature = signature?;
	let parsed: Value = serde_json::from_str(signature).ok()?;
	let array = parsed.as_array()?;
	if array.is_empty() || !array.iter().all(is_openai_reasoning_detail) {
		return None;
	}
	Some(array.clone())
}

fn parse_legacy_encrypted_reasoning_detail(signature: Option<&str>) -> Option<Value> {
	let signature = signature?;
	let parsed: Value = serde_json::from_str(signature).ok()?;
	if !is_openai_reasoning_detail(&parsed) {
		return None;
	}
	let object = parsed.as_object()?;
	if object.get("type").and_then(Value::as_str) != Some("reasoning.encrypted") {
		return None;
	}
	let id_ok = object.get("id").and_then(Value::as_str).is_some_and(|id| !id.is_empty());
	let data_ok = object.get("data").and_then(Value::as_str).is_some_and(|data| !data.is_empty());
	if id_ok && data_ok {
		Some(parsed)
	} else {
		None
	}
}

fn json_value_or_clone(target: &mut Map<String, Value>, key: &str, source_key: &str, source: &Map<String, Value>) {
	if !target.contains_key(key) {
		if let Some(value) = source.get(source_key) {
			target.insert(key.to_string(), value.clone());
		}
	}
}

/// `appendOpenAIReasoningDetail`: merge consecutive text/summary deltas.
fn append_openai_reasoning_detail(details: &mut Vec<Value>, detail: Value) {
	let detail_object = detail.as_object().expect("reasoning detail object").clone();
	let detail_type = detail_object.get("type").and_then(Value::as_str).unwrap_or_default();
		if let Some(last) = details.last_mut() {
		let last_object = last.as_object_mut().expect("reasoning detail object");
		let last_type = last_object.get("type").and_then(Value::as_str).unwrap_or_default();
		if detail_type == "reasoning.text" && last_type == "reasoning.text" {
			if let (Some(Value::String(text)), Some(delta_text)) = (
				last_object.get_mut("text"),
				detail_object.get("text").and_then(Value::as_str),
			) {
				text.push_str(delta_text);
			}
			if last_object.get("signature").map(|value| value.is_null()).unwrap_or(true)
				|| !last_object.contains_key("signature")
			{
				if let Some(signature) = detail_object.get("signature") {
					if !signature.is_null() {
						last_object.insert("signature".to_string(), signature.clone());
					}
				}
			}
			json_value_or_clone(last_object, "id", "id", &detail_object);
			if let Some(format) = detail_object.get("format") {
				last_object
					.entry("format".to_string())
					.or_insert_with(|| format.clone());
			}
			json_value_or_clone(last_object, "index", "index", &detail_object);
			return;
		}
		if detail_type == "reasoning.summary" && last_type == "reasoning.summary" {
			if let (Some(Value::String(summary)), Some(delta_summary)) = (
				last_object.get_mut("summary"),
				detail_object.get("summary").and_then(Value::as_str),
			) {
				summary.push_str(delta_summary);
			}
			json_value_or_clone(last_object, "id", "id", &detail_object);
			if let Some(format) = detail_object.get("format") {
				last_object
					.entry("format".to_string())
					.or_insert_with(|| format.clone());
			}
			json_value_or_clone(last_object, "index", "index", &detail_object);
			return;
		}
	}
	details.push(detail);
}

// ---------------------------------------------------------------------------
// GitHub Copilot dynamic headers (D14 — inline port of
// `api/github-copilot-headers.ts`)
// ---------------------------------------------------------------------------


fn infer_copilot_initiator(messages: &[Message]) -> &'static str {
	match messages.last() {
		Some(message) if message.role() == "user" => "user",
		Some(_) => "agent",
		None => "user",
	}
}

fn has_copilot_vision_input(messages: &[Message]) -> bool {
	messages.iter().any(|message| match message {
		Message::User(user) => matches!(&user.content, crate::types::UserContentField::Blocks(blocks)
			if blocks.iter().any(|block| matches!(block, UserContent::Image(_)))),
		Message::ToolResult(tool_result) => {
			tool_result.content.iter().any(|block| matches!(block, UserContent::Image(_)))
		}
		_ => false,
	})
}

fn build_copilot_dynamic_headers(messages: &[Message]) -> Vec<(String, String)> {
	let mut headers = vec![
		("X-Initiator".to_string(), infer_copilot_initiator(messages).to_string()),
		("Openai-Intent".to_string(), "conversation-edits".to_string()),
	];
	if has_copilot_vision_input(messages) {
		headers.push(("Copilot-Vision-Request".to_string(), "true".to_string()));
	}
	headers
}

// ---------------------------------------------------------------------------
// Request error handling (D15)
// ---------------------------------------------------------------------------

/// OpenRouter attaches extra information at `error.error.metadata.raw`
/// upstream; the Rust port re-reads it from the structured body.
fn raw_metadata_suffix(error: &ProviderHttpError, error_message: &str) -> Option<String> {
	let Some(body) = &error.body else { return None };
	let parsed: Value = serde_json::from_str(body).ok()?;
	let raw = parsed.get("error")?.get("metadata")?.get("raw")?;
	let raw_string = match raw {
		Value::String(raw) => raw.clone(),
		other => other.to_string(),
	};
	if error_message.contains(&raw_string) {
		None
	} else {
		Some(raw_string)
	}
}

// ---------------------------------------------------------------------------
// Resolved compatibility settings
// ---------------------------------------------------------------------------

/// `ResolvedOpenAICompletionsCompat`: detection defaults overlaid with
/// explicit `model.compat` entries.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedCompat {
	supports_store: bool,
	supports_developer_role: bool,
	supports_reasoning_effort: bool,
	supports_usage_in_streaming: bool,
	supports_finish_reason: bool,
	max_tokens_field: MaxTokensField,
	requires_tool_result_name: bool,
	requires_assistant_after_tool_result: bool,
	requires_thinking_as_text: bool,
	requires_reasoning_content_on_assistant_messages: bool,
	thinking_format: ThinkingFormat,
	open_router_routing: Option<crate::types::OpenRouterRouting>,
	vercel_gateway_routing: crate::types::VercelGatewayRouting,
	chat_template_kwargs: BTreeMap<String, ChatTemplateKwargValue>,
	chat_template_args: BTreeMap<String, ChatTemplateKwargValue>,
	zai_tool_stream: bool,
	supports_thinking_token_budget: bool,
	thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
	supports_strict_mode: bool,
	supports_openai_grammar_tools: bool,
	supports_mid_convo_system_messages: bool,
	supports_mid_convo_tool_additions: bool,
	cache_control_format: Option<CacheControlFormat>,
	send_session_affinity_headers: bool,
	session_affinity_format: crate::types::SessionAffinityFormat,
	supports_long_cache_retention: bool,
	vllm_priority: Option<f64>,
}

/// Auto-detect compatibility settings from provider name and baseUrl
/// (`detectCompat`). Used as the base when `model.compat` is not set;
/// explicit entries override the detected values.
fn detect_compat(model: &Model) -> ResolvedCompat {
	let provider = model.provider();
	let base_url = &model.base.base_url;

	let is_zai = provider == "zai"
		|| provider == "zai-coding-cn"
		|| base_url.contains("api.z.ai")
		|| base_url.contains("open.bigmodel.cn");
	let is_together =
		provider == "together" || base_url.contains("api.together.ai") || base_url.contains("api.together.xyz");
	let is_moonshot = provider == "moonshotai" || provider == "moonshotai-cn" || base_url.contains("api.moonshot.");
	let is_openrouter = provider == "openrouter" || base_url.contains("openrouter.ai");
	let is_cloudflare_workers_ai = provider == "cloudflare-workers-ai" || base_url.contains("api.cloudflare.com");
	let is_cloudflare_ai_gateway =
		provider == "cloudflare-ai-gateway" || base_url.contains("gateway.ai.cloudflare.com");
	let is_nvidia = provider == "nvidia" || base_url.contains("integrate.api.nvidia.com");
	let is_ant_ling = provider == "ant-ling" || base_url.contains("api.ant-ling.com");
	let is_cerebras = provider == "cerebras" || base_url.contains("cerebras.ai");
	let is_deepseek = provider == "deepseek" || base_url.to_lowercase().contains("deepseek.com");

	let is_non_standard = is_nvidia
		|| is_cerebras
		|| provider == "xai"
		|| base_url.contains("api.x.ai")
		|| is_together
		|| base_url.contains("chutes.ai")
		|| is_deepseek
		|| is_zai
		|| is_moonshot
		|| provider == "opencode"
		|| base_url.contains("opencode.ai")
		|| is_cloudflare_workers_ai
		|| is_cloudflare_ai_gateway
		|| is_ant_ling;

	let use_max_tokens = base_url.contains("chutes.ai")
		|| is_deepseek
		|| is_moonshot
		|| is_cloudflare_ai_gateway
		|| is_together
		|| is_nvidia
		|| is_ant_ling
		|| is_zai;

	let is_grok = provider == "xai" || base_url.contains("api.x.ai");
	let is_openrouter_developer_role_model =
		is_openrouter && (model.id().starts_with("anthropic/") || model.id().starts_with("openai/"));
	let cache_control_format =
		if provider == "openrouter" && model.id().starts_with("anthropic/") {
			Some(CacheControlFormat::Anthropic)
		} else {
			None
		};

	ResolvedCompat {
		supports_store: !is_non_standard,
		supports_developer_role: is_openrouter_developer_role_model || (!is_non_standard && !is_openrouter),
		supports_reasoning_effort: !is_grok
			&& !is_zai
			&& !is_moonshot
			&& !is_together
			&& !is_cloudflare_ai_gateway
			&& !is_nvidia
			&& !is_ant_ling,
		supports_usage_in_streaming: true,
		supports_finish_reason: true,
		max_tokens_field: if use_max_tokens { MaxTokensField::MaxTokens } else { MaxTokensField::MaxCompletionTokens },
		requires_tool_result_name: false,
		requires_assistant_after_tool_result: false,
		requires_thinking_as_text: false,
		requires_reasoning_content_on_assistant_messages: is_deepseek,
		thinking_format: if is_deepseek {
			ThinkingFormat::Deepseek
		} else if is_zai {
			ThinkingFormat::Zai
		} else if is_together {
			ThinkingFormat::Together
		} else if is_ant_ling {
			ThinkingFormat::AntLing
		} else if is_openrouter {
			ThinkingFormat::Openrouter
		} else {
			ThinkingFormat::Openai
		},
		open_router_routing: Some(crate::types::OpenRouterRouting::default()),
		vercel_gateway_routing: crate::types::VercelGatewayRouting::default(),
		chat_template_kwargs: BTreeMap::new(),
		chat_template_args: BTreeMap::new(),
		zai_tool_stream: false,
		supports_thinking_token_budget: false,
		thinking_token_budget_field: None,
		// OpenAI compatibility alone does not imply strict JSON-schema tool support.
		supports_strict_mode: false,
		supports_openai_grammar_tools: false,
		supports_mid_convo_system_messages: false,
		supports_mid_convo_tool_additions: false,
		cache_control_format,
		send_session_affinity_headers: is_openrouter,
		session_affinity_format: if is_openrouter {
			crate::types::SessionAffinityFormat::Openrouter
		} else {
			crate::types::SessionAffinityFormat::Openai
		},
		supports_long_cache_retention: !(is_together
			|| is_cloudflare_workers_ai
			|| is_cloudflare_ai_gateway
			|| is_nvidia
			|| is_ant_ling),
		vllm_priority: None,
	}
}

/// Get resolved compatibility settings for a model (`getCompat`).
fn get_compat(model: &Model) -> ResolvedCompat {
	let detected = detect_compat(model);
	let Some(compat) = model.compat.as_ref() else { return detected };

	ResolvedCompat {
		supports_store: compat.supports_store.unwrap_or(detected.supports_store),
		supports_developer_role: compat.supports_developer_role.unwrap_or(detected.supports_developer_role),
		supports_reasoning_effort: compat.supports_reasoning_effort.unwrap_or(detected.supports_reasoning_effort),
		supports_usage_in_streaming: compat
			.supports_usage_in_streaming
			.unwrap_or(detected.supports_usage_in_streaming),
		supports_finish_reason: compat.supports_finish_reason.unwrap_or(detected.supports_finish_reason),
		max_tokens_field: compat.max_tokens_field.unwrap_or(detected.max_tokens_field),
		requires_tool_result_name: compat.requires_tool_result_name.unwrap_or(detected.requires_tool_result_name),
		requires_assistant_after_tool_result: compat
			.requires_assistant_after_tool_result
			.unwrap_or(detected.requires_assistant_after_tool_result),
		requires_thinking_as_text: compat.requires_thinking_as_text.unwrap_or(detected.requires_thinking_as_text),
		requires_reasoning_content_on_assistant_messages: compat
			.requires_reasoning_content_on_assistant_messages
			.unwrap_or(detected.requires_reasoning_content_on_assistant_messages),
		thinking_format: compat.thinking_format.unwrap_or(detected.thinking_format),
		open_router_routing: compat.open_router_routing.clone().or(detected.open_router_routing),
		vercel_gateway_routing: compat.vercel_gateway_routing.clone().unwrap_or(detected.vercel_gateway_routing),
		chat_template_kwargs: compat.chat_template_kwargs.clone().unwrap_or(detected.chat_template_kwargs),
		chat_template_args: compat.chat_template_args.clone().unwrap_or(detected.chat_template_args),
		zai_tool_stream: compat.zai_tool_stream.unwrap_or(detected.zai_tool_stream),
		supports_thinking_token_budget: compat
			.supports_thinking_token_budget
			.unwrap_or(detected.supports_thinking_token_budget),
		thinking_token_budget_field: compat
			.thinking_token_budget_field
			.or(detected.thinking_token_budget_field),
		supports_strict_mode: compat.supports_strict_mode.unwrap_or(detected.supports_strict_mode),
		supports_openai_grammar_tools: compat
			.supports_openai_grammar_tools
			.unwrap_or(detected.supports_openai_grammar_tools),
		supports_mid_convo_system_messages: compat
			.supports_mid_convo_system_messages
			.unwrap_or(detected.supports_mid_convo_system_messages),
		supports_mid_convo_tool_additions: compat
			.supports_mid_convo_tool_additions
			.unwrap_or(detected.supports_mid_convo_tool_additions),
		cache_control_format: compat.cache_control_format.or(detected.cache_control_format),
		send_session_affinity_headers: compat
			.send_session_affinity_headers
			.unwrap_or(detected.send_session_affinity_headers),
		session_affinity_format: compat.session_affinity_format.unwrap_or(detected.session_affinity_format),
		supports_long_cache_retention: compat
			.supports_long_cache_retention
			.unwrap_or(detected.supports_long_cache_retention),
		vllm_priority: compat.vllm_priority,
	}
}

// ---------------------------------------------------------------------------
// Usage + stop reasons
// ---------------------------------------------------------------------------

fn value_u64(value: Option<&Value>) -> u64 {
	value.and_then(Value::as_u64).unwrap_or(0)
}

fn parse_chunk_usage(raw_usage: &Value, model: &Model) -> Usage {
	let prompt_tokens = value_u64(raw_usage.get("prompt_tokens"));
	let prompt_tokens_details = raw_usage.get("prompt_tokens_details");
	let cache_read_tokens = prompt_tokens_details
		.and_then(|details| details.get("cached_tokens"))
		.and_then(Value::as_u64)
		.or_else(|| raw_usage.get("prompt_cache_hit_tokens").and_then(Value::as_u64))
		.or_else(|| raw_usage.get("cached_tokens").and_then(Value::as_u64))
		.unwrap_or(0);
	let cache_write_tokens = prompt_tokens_details
		.and_then(|details| details.get("cache_write_tokens"))
		.and_then(Value::as_u64)
		.unwrap_or(0);

	// Follow documented OpenAI/OpenRouter semantics: cached_tokens is
	// cache-read tokens (hits). Do not subtract writes from cached_tokens.
	let input = prompt_tokens.saturating_sub(cache_read_tokens).saturating_sub(cache_write_tokens);
	// OpenAI completion_tokens already includes reasoning_tokens.
	let output_tokens = value_u64(raw_usage.get("completion_tokens"));
	let reasoning = value_u64(raw_usage.get("completion_tokens_details").and_then(|details| details.get("reasoning_tokens")));
	let mut usage = Usage {
		input,
		output: output_tokens,
		cache_read: cache_read_tokens,
		cache_write: cache_write_tokens,
		cache_write_1h: None,
		reasoning: Some(reasoning),
		total_tokens: input + output_tokens + cache_read_tokens + cache_write_tokens,
		cost: Default::default(),
	};
	calculate_cost(&AnyModel::Chat(model.clone()), &mut usage);
	usage
}

fn map_stop_reason(reason: &str) -> (StopReason, Option<String>) {
	match reason {
		"stop" | "end" => (StopReason::Stop, None),
		"length" => (StopReason::Length, None),
		"function_call" | "tool_calls" => (StopReason::ToolUse, None),
		"content_filter" => (StopReason::Error, Some("Provider finish_reason: content_filter".to_string())),
		"network_error" => (StopReason::Error, Some("Provider finish_reason: network_error".to_string())),
		other => (StopReason::Error, Some(format!("Provider finish_reason: {other}"))),
	}
}

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
pub(crate) struct ConvertMessagesOptions {
	grammar_tool_input_properties: Option<BTreeMap<String, String>>,
}

fn normalize_tool_call_id(model: &Model, id: &str) -> String {
	// Handle pipe-separated IDs from OpenAI Responses API
	// (`{call_id}|{item_id}`): extract the call id and normalize.
	if id.contains('|') {
		let separator_index = id.find('|').expect("contains |");
		let sanitize = |text: &str| -> String {
			text.chars()
				.map(|character| {
					if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
						character
					} else {
						'_'
					}
				})
				.collect()
		};
		let call_id = sanitize(&id[..separator_index]);
		let item_id = sanitize(&id[separator_index + 1..]);
		let combined_id = if !item_id.is_empty() { format!("{call_id}_{item_id}") } else { call_id };
		if combined_id.chars().count() <= 40 {
			return combined_id;
		}
		let hash = short_hash(id);
		let hash = &hash[..hash.chars().count().min(8)];
		let prefix: String = combined_id
			.chars()
			.take((40usize).saturating_sub(hash.chars().count() + 1).max(1))
			.collect();
		return format!("{prefix}_{hash}");
	}

	if model.provider() == "openai" {
		return id.chars().take(40).collect();
	}
	id.to_string()
}

fn convert_tools(tools: &[Tool], compat: &ResolvedCompat) -> Result<Vec<Value>, String> {
	use crate::api::constrained_sampling::{
		get_json_schema_tool_parameters, resolve_grammar_constrained_sampling, resolve_json_schema_strict_sampling,
	};
	tools
		.iter()
		.map(|tool| {
			let grammar = resolve_grammar_constrained_sampling(tool, compat.supports_openai_grammar_tools)?;
			if let Some(grammar) = grammar {
				return Ok(json!({
					"type": "custom",
					"custom": {
						"name": tool.name,
						"description": tool.description,
						"format": {
							"type": "grammar",
							"grammar": {
								"syntax": grammar.format,
								"definition": grammar.definition,
							},
						},
					},
				}));
			}
			let strict = resolve_json_schema_strict_sampling(tool, compat.supports_strict_mode)?;
			let mut function = Map::new();
			function.insert("name".to_string(), json!(tool.name));
			function.insert("description".to_string(), json!(tool.description));
			// After a `Some(true)` strict resolution the conversion cannot
			// fail; the non-strict path always returns the schema unchanged.
			let parameters = get_json_schema_tool_parameters(tool, strict == Some(true))
				.map_err(|error| error.to_string())?;
			function.insert("parameters".to_string(), parameters);
			// Only include strict if provider supports it. Some reject unknown fields.
			if compat.supports_strict_mode {
				function.insert("strict".to_string(), json!(strict.unwrap_or(false)));
			}
			Ok(json!({
				"type": "function",
				"function": function,
			}))
		})
		.collect()
}

#[derive(Clone, PartialEq)]
struct OpenAICompatCacheControl {
	ttl: Option<&'static str>,
}

impl OpenAICompatCacheControl {
	fn to_json(&self) -> Value {
		let mut control = json!({ "type": "ephemeral" });
		if let Some(ttl) = self.ttl {
			control["ttl"] = json!(ttl);
		}
		control
	}
}

fn resolve_cache_retention(cache_retention: Option<CacheRetention>, env: Option<&crate::types::ProviderEnv>) -> CacheRetention {
	if let Some(cache_retention) = cache_retention {
		return cache_retention;
	}
	if get_provider_env_value("PI_CACHE_RETENTION", env).as_deref() == Some("long") {
		return CacheRetention::Long;
	}
	CacheRetention::Short
}

fn get_compat_cache_control(compat: &ResolvedCompat, cache_retention: CacheRetention) -> Option<OpenAICompatCacheControl> {
	if compat.cache_control_format != Some(CacheControlFormat::Anthropic) || cache_retention == CacheRetention::None {
		return None;
	}
	let ttl = if cache_retention == CacheRetention::Long && compat.supports_long_cache_retention {
		Some("1h")
	} else {
		None
	};
	Some(OpenAICompatCacheControl { ttl })
}

/// Replace string content (or annotate the last text part) with an
/// anthropic-style `cache_control` marker. Returns whether anything changed.
fn add_cache_control_to_text_content(message: &mut Value, cache_control: &OpenAICompatCacheControl) -> bool {
	let Some(content) = message.get_mut("content") else { return false };
	match content {
		Value::String(text) => {
			if text.is_empty() {
				return false;
			}
			let text = text.clone();
			message["content"] = json!([{ "type": "text", "text": text, "cache_control": cache_control.to_json() }]);
			true
		}
		Value::Array(parts) => {
			for part in parts.iter_mut().rev() {
				if part.get("type").and_then(Value::as_str) == Some("text") {
					part["cache_control"] = cache_control.to_json();
					return true;
				}
			}
			false
		}
		_ => false,
	}
}

fn add_cache_control_to_system_prompt(messages: &mut [Value], cache_control: &OpenAICompatCacheControl) {
	for message in messages.iter_mut() {
		if matches!(message.get("role").and_then(Value::as_str), Some("system") | Some("developer")) {
			add_cache_control_to_text_content(message, cache_control);
			return;
		}
	}
}

fn add_cache_control_to_last_conversation_message(messages: &mut [Value], cache_control: &OpenAICompatCacheControl) {
	for message in messages.iter_mut().rev() {
		if matches!(
			message.get("role").and_then(Value::as_str),
			Some("user") | Some("assistant") | Some("tool")
		) {
			if add_cache_control_to_text_content(message, cache_control) {
				return;
			}
		}
	}
}

fn add_cache_control_to_last_tool(tools: Option<&mut Vec<Value>>, cache_control: &OpenAICompatCacheControl) {
	let Some(tools) = tools else { return };
	if let Some(last_tool) = tools.last_mut() {
		last_tool["cache_control"] = cache_control.to_json();
	}
}

fn apply_anthropic_cache_control(
	messages: &mut [Value],
	tools: Option<&mut Vec<Value>>,
	cache_control: &OpenAICompatCacheControl,
) {
	add_cache_control_to_system_prompt(messages, cache_control);
	add_cache_control_to_last_tool(tools, cache_control);
	add_cache_control_to_last_conversation_message(messages, cache_control);
}

fn assistant_text_part(text: &str) -> Value {
	json!({ "type": "text", "text": sanitize_surrogates(text) })
}

/// `convertMessages`.
pub(crate) fn convert_messages(
	model: &Model,
	context: &TranscriptContext,
	compat: &ResolvedCompat,
	options: Option<&ConvertMessagesOptions>,
) -> Result<Vec<Value>, String> {
	let normalized_context = resolve_transcript(context, Some(compat.supports_mid_convo_system_messages));
	let mut params: Vec<Value> = Vec::new();

	let transformed_messages = crate::api::transform_messages::transform_messages(
		normalized_context.messages(),
		model,
		Some(&|id: &str, _model: &Model, _source: &crate::types::AssistantMessage| normalize_tool_call_id(model, id)),
	);
	let transcript_tools = resolve_transcript_tools(
		normalized_context.messages(),
		compat.supports_mid_convo_system_messages && compat.supports_mid_convo_tool_additions,
	);
	let instruction_role = if model.reasoning && compat.supports_developer_role {
		"developer"
	} else {
		"system"
	};

	let mut last_role: Option<&'static str> = None;

	let mut index = 0;
	while index < transformed_messages.len() {
		let message = &transformed_messages[index];
		// Some providers don't allow user messages directly after tool results;
		// insert a synthetic assistant message to bridge the gap.
		if compat.requires_assistant_after_tool_result && last_role == Some("toolResult") && message.role() == "user" {
			params.push(json!({ "role": "assistant", "content": "I have processed the tool results." }));
		}

		match message {
			Message::System(system) => {
				let added_tools = if index > 0 && transcript_tools.anchors_additions {
					system.tools_added.clone().unwrap_or_default()
				} else {
					Vec::new()
				};
				if !added_tools.is_empty() {
					let tools = convert_tools(&added_tools, compat)?;
					params.push(json!({ "role": "system", "tools": tools }));
				}
				let text = if index == 0 {
					get_system_message_text(system)
				} else {
					render_system_message_update(system)
				};
				if !text.is_empty() {
					params.push(json!({ "role": instruction_role, "content": sanitize_surrogates(&text) }));
				}
			}
			Message::User(user) => match &user.content {
				crate::types::UserContentField::Text(text) => {
					params.push(json!({ "role": "user", "content": sanitize_surrogates(text) }));
				}
				crate::types::UserContentField::Blocks(blocks) => {
					let content: Vec<Value> = blocks
						.iter()
						.filter(|item| !matches!(item, UserContent::Text(text) if text.text.is_empty()))
						.map(|item| match item {
							UserContent::Text(text) => assistant_text_part(&text.text),
							UserContent::Image(image) => json!({
								"type": "image_url",
								"image_url": { "url": format!("data:{};base64,{}", image.mime_type, image.data) },
							}),
						})
						.collect();
					if !content.is_empty() {
						params.push(json!({ "role": "user", "content": content }));
					}
				}
			},
			Message::Assistant(assistant) => {
				let mut assistant_msg = Map::new();
				assistant_msg.insert("role".to_string(), json!("assistant"));
				// Some providers don't accept null content; those get "".
				assistant_msg.insert(
					"content".to_string(),
					if compat.requires_assistant_after_tool_result { json!("") } else { Value::Null },
				);

				let assistant_text_parts: Vec<Value> = assistant
					.content
					.iter()
					.filter_map(|block| match block {
						AssistantContent::Text(text) if !text.text.trim().is_empty() => {
							Some(assistant_text_part(&text.text))
						}
						_ => None,
					})
					.collect();
				let assistant_text: String = assistant_text_parts
					.iter()
					.filter_map(|part| part.get("text").and_then(Value::as_str))
					.collect();

				let thinking_blocks: Vec<&ThinkingContent> = assistant
					.content
					.iter()
					.filter_map(|block| match block {
						AssistantContent::Thinking(thinking) => Some(thinking),
						_ => None,
					})
					.collect();
				let tool_calls: Vec<&ToolCall> = assistant
					.content
					.iter()
					.filter_map(|block| match block {
						AssistantContent::ToolCall(tool_call) => Some(tool_call),
						_ => None,
					})
					.collect();
				let signed_reasoning_details = thinking_blocks
					.iter()
					.filter_map(|block| parse_openai_reasoning_details(block.thinking_signature.as_deref()))
					.next();
				let legacy_reasoning_details: Vec<Value> = tool_calls
					.iter()
					.filter_map(|tool_call| parse_legacy_encrypted_reasoning_detail(tool_call.thought_signature.as_deref()))
					.collect();
				let preserved_reasoning_details =
					signed_reasoning_details.or(if legacy_reasoning_details.is_empty() {
						None
					} else {
						Some(legacy_reasoning_details)
					});

				let non_empty_thinking_blocks: Vec<&&ThinkingContent> = thinking_blocks
					.iter()
					.filter(|block| !block.thinking.trim().is_empty())
					.collect();
				if !non_empty_thinking_blocks.is_empty() {
					if compat.requires_thinking_as_text {
						// Convert thinking blocks to plain text (no tags to avoid
						// model mimicking them).
						let thinking_text = non_empty_thinking_blocks
							.iter()
							.map(|block| sanitize_surrogates(&block.thinking))
							.collect::<Vec<_>>()
							.join("\n\n");
						let mut content = vec![json!({ "type": "text", "text": thinking_text })];
						content.extend(assistant_text_parts);
						assistant_msg.insert("content".to_string(), Value::Array(content));
					} else {
						// Always send assistant content as a plain string (OpenAI
						// Chat Completions standard format).
						if !assistant_text.is_empty() {
							assistant_msg.insert("content".to_string(), json!(assistant_text));
						}

						// reasoning_details is the structured alternative to a raw
						// reasoning field.
						if preserved_reasoning_details.is_none() {
							// Use the signature from the first thinking block if
							// available (for llama.cpp server + gpt-oss).
							let mut signature = non_empty_thinking_blocks[0].thinking_signature.clone();
							if model.provider() == "opencode-go" && signature.as_deref() == Some("reasoning") {
								signature = Some("reasoning_content".to_string());
							}
							if let Some(signature) = signature {
								if OPENAI_COMPLETIONS_REASONING_FIELDS.contains(&signature.as_str()) {
									let joined = non_empty_thinking_blocks
										.iter()
										.map(|block| block.thinking.as_str())
										.collect::<Vec<_>>()
										.join("\n");
									assistant_msg.insert(signature, json!(joined));
								}
							}
						}
					}
				} else if !assistant_text.is_empty() {
					assistant_msg.insert("content".to_string(), json!(assistant_text));
				}

				if !tool_calls.is_empty() {
					let serialized: Vec<Value> = tool_calls
						.iter()
						.map(|tool_call| {
							let custom_input_property = options
								.and_then(|options| options.grammar_tool_input_properties.as_ref())
								.and_then(|properties| properties.get(&tool_call.name));
							if let Some(custom_input_property) = custom_input_property {
								let input = crate::api::constrained_sampling::get_grammar_tool_input(
									&tool_call.name,
									&tool_call.arguments,
									custom_input_property,
								)
								.unwrap_or_default();
								return json!({
									"id": tool_call.id,
									"type": "custom",
									"custom": {
										"name": tool_call.name,
										"input": sanitize_surrogates(&input),
									},
								});
							}
							json!({
								"id": tool_call.id,
								"type": "function",
								"function": {
									"name": tool_call.name,
									"arguments": Value::Object(tool_call.arguments.clone()).to_string(),
								},
							})
						})
						.collect();
					assistant_msg.insert("tool_calls".to_string(), Value::Array(serialized));
				}
				if let Some(details) = preserved_reasoning_details {
					assistant_msg.insert("reasoning_details".to_string(), Value::Array(details));
				}
				if compat.requires_reasoning_content_on_assistant_messages
					&& model.reasoning
					&& !assistant_msg.contains_key("reasoning_content")
				{
					assistant_msg.insert("reasoning_content".to_string(), json!(""));
				}
				// Skip assistant messages that have no content and no tool calls
				// (e.g. aborted assistant responses that got no content).
				let has_content = match assistant_msg.get("content") {
					Some(Value::String(text)) => !text.is_empty(),
					Some(Value::Array(parts)) => !parts.is_empty(),
					_ => false,
				};
				if !has_content && !assistant_msg.contains_key("tool_calls") {
					// Upstream `continue`s before `lastRole = msg.role`.
					index += 1;
					continue;
				}
				params.push(Value::Object(assistant_msg));
			}
			Message::ToolResult(_) => {
				let mut image_blocks: Vec<Value> = Vec::new();
				let mut cursor = index;

				while cursor < transformed_messages.len() && transformed_messages[cursor].role() == "toolResult" {
					let Message::ToolResult(tool_msg) = &transformed_messages[cursor] else {
						unreachable!("checked role");
					};

					// Extract text and image content.
					let text_result = tool_msg
						.content
						.iter()
						.filter_map(|block| match block {
							UserContent::Text(text) => Some(text.text.as_str()),
							_ => None,
						})
						.collect::<Vec<_>>()
						.join("\n");
					let has_images = tool_msg.content.iter().any(|block| matches!(block, UserContent::Image(_)));

					// Always send tool result with text (or placeholder if only
					// images).
					let has_text = !text_result.is_empty();
					let tool_result_text = if has_text {
						text_result
					} else if has_images {
						"(see attached image)".to_string()
					} else {
						"(no tool output)".to_string()
					};
					let mut tool_result_msg = json!({
						"role": "tool",
						"content": sanitize_surrogates(&tool_result_text),
						"tool_call_id": tool_msg.tool_call_id,
					});
					if compat.requires_tool_result_name && !tool_msg.tool_name.is_empty() {
						tool_result_msg["name"] = json!(tool_msg.tool_name);
					}
					params.push(tool_result_msg);

					if has_images && model.base.input.contains(&ModelInput::Image) {
						for block in &tool_msg.content {
							if let UserContent::Image(image) = block {
								image_blocks.push(json!({
									"type": "image_url",
									"image_url": { "url": format!("data:{};base64,{}", image.mime_type, image.data) },
								}));
							}
						}
					}
					cursor += 1;
				}

				index = cursor;

				if !image_blocks.is_empty() {
					if compat.requires_assistant_after_tool_result {
						params.push(json!({ "role": "assistant", "content": "I have processed the tool results." }));
					}
					let mut content = vec![json!({ "type": "text", "text": "Attached image(s) from tool result:" })];
					content.extend(image_blocks);
					params.push(json!({ "role": "user", "content": content }));
					last_role = Some("user");
					// Upstream `continue`s past the trailing `lastRole = msg.role`.
					index += 1;
					continue;
				}
				last_role = Some("toolResult");
				index += 1;
				continue;
			}
		}

		last_role = role_str(message);
		index += 1;
	}

	Ok(params)
}

fn role_str(message: &Message) -> Option<&'static str> {
	Some(message.role())
}

// ---------------------------------------------------------------------------
// Request parameters
// ---------------------------------------------------------------------------

fn resolve_thinking_token_budget_field(
	compat: &ResolvedCompat,
) -> Option<ThinkingTokenBudgetField> {
	if compat.thinking_token_budget_field.is_some() {
		return compat.thinking_token_budget_field;
	}
	if compat.supports_thinking_token_budget {
		return Some(ThinkingTokenBudgetField::ThinkingTokenBudget);
	}
	None
}

impl ThinkingTokenBudgetField {
	fn as_str(&self) -> &'static str {
		match self {
			ThinkingTokenBudgetField::ThinkingTokenBudget => "thinking_token_budget",
			ThinkingTokenBudgetField::ThinkingBudget => "thinking_budget",
			ThinkingTokenBudgetField::ThinkingBudgetTokens => "thinking_budget_tokens",
		}
	}
}

impl ThinkingLevel {
	fn as_str(&self) -> &'static str {
		match self {
			ThinkingLevel::Minimal => "minimal",
			ThinkingLevel::Low => "low",
			ThinkingLevel::Medium => "medium",
			ThinkingLevel::High => "high",
			ThinkingLevel::Xhigh => "xhigh",
			ThinkingLevel::Max => "max",
		}
	}
}

fn resolve_clamped_thinking_budget(
	model: &Model,
	options: Option<&StreamOptions>,
	params: &Map<String, Value>,
) -> Option<u64> {
	use crate::api::simple_options::{clamp_thinking_budget_to_answer_room, thinking_budget_for_level};
	let options = options?;
	let reasoning_effort = options.openai_completions.reasoning_effort?;
	if !model.reasoning {
		return None;
	}
	let ceiling = params
		.get("max_tokens")
		.and_then(Value::as_u64)
		.or_else(|| params.get("max_completion_tokens").and_then(Value::as_u64))
		.unwrap_or(model.max_tokens);
	let budget = clamp_thinking_budget_to_answer_room(
		thinking_budget_for_level(reasoning_effort, options.openai_completions.thinking_budgets.as_ref()),
		ceiling,
	);
	(budget > 0).then_some(budget)
}

/// Resolve `$var` template kwargs (`resolveChatTemplateKwargValue`). Returns
/// `None` when the value should be omitted.
fn resolve_chat_template_kwarg_value(
	model: &Model,
	reasoning_effort: Option<ThinkingLevel>,
	value: &ChatTemplateKwargValue,
	thinking_budget: Option<u64>,
) -> Option<Value> {
	match value {
		ChatTemplateKwargValue::Literal(literal) => Some(match literal {
			JsonLiteral::String(text) => json!(text),
			JsonLiteral::Number(number) => json!(number),
			JsonLiteral::Boolean(flag) => json!(flag),
			JsonLiteral::Null => Value::Null,
		}),
		ChatTemplateKwargValue::Var { var, omit_when_off } => {
			if reasoning_effort.is_none() && omit_when_off.unwrap_or(false) {
				return None;
			}
			match var {
				ChatTemplateVar::ThinkingEnabled => Some(json!(reasoning_effort.is_some())),
				ChatTemplateVar::ThinkingBudget => thinking_budget.map(|budget| json!(budget)),
				ChatTemplateVar::ThinkingEffort => {
					let level = reasoning_effort
						.map(ModelThinkingLevel::from)
						.unwrap_or(ModelThinkingLevel::Off);
					let mapped = model
						.thinking_level_map
						.as_ref()
						.and_then(|map| map.get(&level))
						.map(|mapped| mapped.clone());
					match mapped {
						None => reasoning_effort.map(|level| json!(level.as_str())),
						Some(Some(mapped)) => Some(json!(mapped)),
						Some(None) => None,
					}
				}
			}
		}
	}
}

fn build_chat_template_values(
	model: &Model,
	reasoning_effort: Option<ThinkingLevel>,
	values: &BTreeMap<String, ChatTemplateKwargValue>,
	thinking_budget: Option<u64>,
) -> Option<Value> {
	let mut resolved = Map::new();
	for (key, value) in values {
		if let Some(resolved_value) =
			resolve_chat_template_kwarg_value(model, reasoning_effort, value, thinking_budget)
		{
			resolved.insert(key.clone(), resolved_value);
		}
	}
	(!resolved.is_empty()).then_some(Value::Object(resolved))
}

/// `model.thinkingLevelMap?.[level]`: absent key vs explicit `null` vs string.
enum MappedLevel {
	Absent,
	Null,
	Value(String),
}

fn lookup_thinking_level(model: &Model, level: ModelThinkingLevel) -> MappedLevel {
	let Some(map) = model.thinking_level_map.as_ref() else { return MappedLevel::Absent };
	match map.get(&level) {
		None => MappedLevel::Absent,
		Some(None) => MappedLevel::Null,
		Some(Some(value)) => MappedLevel::Value(value.clone()),
	}
}

fn mapped_or<'a>(mapped: &'a MappedLevel, fallback: &'a str) -> Option<&'a str> {
	match mapped {
		MappedLevel::Absent => Some(fallback),
		MappedLevel::Value(value) => Some(value),
		MappedLevel::Null => None,
	}
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_params(
	model: &Model,
	context: &TranscriptContext,
	options: Option<&StreamOptions>,
	compat: &ResolvedCompat,
	cache_retention: CacheRetention,
	grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<Map<String, Value>, String> {
	let reasoning_effort = options.and_then(|options| options.openai_completions.reasoning_effort);

	let transcript_tools: TranscriptTools = resolve_transcript_tools(
		context.messages(),
		compat.supports_mid_convo_system_messages && compat.supports_mid_convo_tool_additions,
	);
	let mut messages = convert_messages(model, context, compat, Some(&ConvertMessagesOptions {
		grammar_tool_input_properties: Some(grammar_tool_input_properties.clone()),
	}))?;
	let cache_control = get_compat_cache_control(compat, cache_retention);

	let mut params = Map::new();
	params.insert("model".to_string(), json!(model.id()));
	params.insert("messages".to_string(), Value::Array(messages.clone()));
	params.insert("stream".to_string(), json!(true));
	let base_url = &model.base.base_url;
	let prompt_cache_key = (base_url.contains("api.openai.com") && cache_retention != CacheRetention::None)
		|| (cache_retention == CacheRetention::Long && compat.supports_long_cache_retention);
	if prompt_cache_key {
		if let Some(key) = crate::api::openai_prompt_cache::clamp_openai_prompt_cache_key(
			options.and_then(|options| options.session_id.as_deref()),
		) {
			params.insert("prompt_cache_key".to_string(), json!(key));
		}
	}
	if cache_retention == CacheRetention::Long && compat.supports_long_cache_retention {
		params.insert("prompt_cache_retention".to_string(), json!("24h"));
	}

	if compat.supports_usage_in_streaming {
		params.insert("stream_options".to_string(), json!({ "include_usage": true }));
	}

	if compat.supports_store {
		params.insert("store".to_string(), json!(false));
	}

	if let Some(max_tokens) = options.and_then(|options| options.max_tokens) {
		if compat.max_tokens_field == MaxTokensField::MaxTokens {
			// Deprecated by OpenAI, but some OpenAI-compatible providers only
			// accept max_tokens.
			params.insert("max_tokens".to_string(), json!(max_tokens));
		} else {
			params.insert("max_completion_tokens".to_string(), json!(max_tokens));
		}
	}

	if let Some(temperature) = options.and_then(|options| options.temperature) {
		params.insert("temperature".to_string(), json!(temperature));
	}

	let mut tools: Option<Vec<Value>> = None;
	if !transcript_tools.request_tools.is_empty() {
		let converted = convert_tools(&transcript_tools.request_tools, compat)?;
		if compat.zai_tool_stream {
			params.insert("tool_stream".to_string(), json!(true));
		}
		tools = Some(converted);
	} else if has_tool_history(context.messages()) {
		// Anthropic (via LiteLLM/proxy) requires tools param when conversation
		// has tool_calls/tool_results.
		tools = Some(Vec::new());
	}

	if let Some(cache_control) = &cache_control {
		apply_anthropic_cache_control(&mut messages, tools.as_mut(), cache_control);
		params.insert("messages".to_string(), Value::Array(messages.clone()));
	}
	if let Some(tools) = tools {
		if !tools.is_empty() {
			params.insert("tools".to_string(), Value::Array(tools));
		} else {
			params.insert("tools".to_string(), json!([]));
		}
	}

	if let Some(tool_choice) = options.and_then(|options| options.openai_completions.tool_choice.clone()) {
		params.insert("tool_choice".to_string(), tool_choice);
	}

	if let Some(priority) = compat.vllm_priority {
		params.insert("priority".to_string(), json!(priority));
	}

	let thinking_token_budget_field = resolve_thinking_token_budget_field(compat);
	let thinking_budget = resolve_clamped_thinking_budget(model, options, &params);

	if model.reasoning {
		match compat.thinking_format {
			ThinkingFormat::Zai => {
				let thinking = if reasoning_effort.is_some() {
					json!({ "type": "enabled", "clear_thinking": false })
				} else {
					json!({ "type": "disabled" })
				};
				params.insert("thinking".to_string(), thinking);
				if let Some(effort_level) = reasoning_effort {
					if compat.supports_reasoning_effort {
						let mapped = lookup_thinking_level(model, ModelThinkingLevel::from(effort_level));
						if let Some(effort) = mapped_or(&mapped, effort_level.as_str()) {
							params.insert("reasoning_effort".to_string(), json!(effort));
						}
					}
				}
			}
			ThinkingFormat::Qwen => {
				params.insert("enable_thinking".to_string(), json!(reasoning_effort.is_some()));
				if let Some(effort_level) = reasoning_effort {
					if compat.supports_reasoning_effort {
						let mapped = lookup_thinking_level(model, ModelThinkingLevel::from(effort_level));
						if let Some(effort) = mapped_or(&mapped, effort_level.as_str()) {
							params.insert("reasoning_effort".to_string(), json!(effort));
						}
					}
				}
			}
			ThinkingFormat::QwenChatTemplate => {
				params.insert(
					"chat_template_kwargs".to_string(),
					json!({
						"enable_thinking": reasoning_effort.is_some(),
						"preserve_thinking": true,
					}),
				);
			}
			ThinkingFormat::ChatTemplate => {
				if let Some(kwargs) = build_chat_template_values(
					model,
					reasoning_effort,
					&compat.chat_template_kwargs,
					thinking_budget,
				) {
					params.insert("chat_template_kwargs".to_string(), kwargs);
				}
			}
			ThinkingFormat::Baseten => {
				if let Some(args) =
					build_chat_template_values(model, reasoning_effort, &compat.chat_template_args, thinking_budget)
				{
					params.insert("chat_template_args".to_string(), args);
				}
				if compat.supports_reasoning_effort {
					let requested = reasoning_effort;
					let level = requested
						.map(ModelThinkingLevel::from)
						.unwrap_or(ModelThinkingLevel::Off);
					let mapped = lookup_thinking_level(model, level);
					if let Some(effort) = match (&mapped, requested) {
						(MappedLevel::Value(value), _) => Some(value.as_str()),
						(MappedLevel::Absent, Some(requested)) => Some(requested.as_str()),
						(MappedLevel::Absent, None) | (MappedLevel::Null, _) => None,
					} {
						params.insert("reasoning_effort".to_string(), json!(effort));
					}
				}
			}
			ThinkingFormat::Deepseek => {
				if reasoning_effort.is_some() {
					params.insert("thinking".to_string(), json!({ "type": "enabled" }));
				} else {
					// `model.thinkingLevelMap?.off !== null` — absent keys pass,
					// only an explicit `null` mapping suppresses the flag.
					let off_is_null = matches!(lookup_thinking_level(model, ModelThinkingLevel::Off), MappedLevel::Null);
					if !off_is_null {
						params.insert("thinking".to_string(), json!({ "type": "disabled" }));
					}
				}
				if let Some(effort_level) = reasoning_effort {
					if compat.supports_reasoning_effort {
						let mapped = lookup_thinking_level(model, ModelThinkingLevel::from(effort_level));
						if let Some(effort) = mapped_or(&mapped, effort_level.as_str()) {
							params.insert("reasoning_effort".to_string(), json!(effort));
						}
					}
				}
			}
			ThinkingFormat::Openrouter => {
				// OpenRouter normalizes reasoning across providers via a nested
				// reasoning object.
				if let Some(effort_level) = reasoning_effort {
					let mapped = lookup_thinking_level(model, ModelThinkingLevel::from(effort_level));
					let effort = mapped_or(&mapped, effort_level.as_str()).unwrap_or(effort_level.as_str());
					params.insert("reasoning".to_string(), json!({ "effort": effort }));
				} else {
					let mapped = lookup_thinking_level(model, ModelThinkingLevel::Off);
					let is_null = matches!(mapped, MappedLevel::Null);
					if !is_null {
						let effort = match &mapped {
							MappedLevel::Value(value) => value.as_str(),
							_ => "none",
						};
						params.insert("reasoning".to_string(), json!({ "effort": effort }));
					}
				}
			}
			ThinkingFormat::AntLing => {
				if let Some(effort_level) = reasoning_effort {
					let mapped = lookup_thinking_level(model, ModelThinkingLevel::from(effort_level));
					if let MappedLevel::Value(effort) = &mapped {
						params.insert("reasoning".to_string(), json!({ "effort": effort }));
					}
				}
			}
			ThinkingFormat::Together => {
				params.insert("reasoning".to_string(), json!({ "enabled": reasoning_effort.is_some() }));
				if let Some(effort_level) = reasoning_effort {
					if compat.supports_reasoning_effort {
						let mapped = lookup_thinking_level(model, ModelThinkingLevel::from(effort_level));
						if let Some(effort) = mapped_or(&mapped, effort_level.as_str()) {
							params.insert("reasoning_effort".to_string(), json!(effort));
						}
					}
				}
			}
			ThinkingFormat::StringThinking => {
				if let Some(effort_level) = reasoning_effort {
					let mapped = lookup_thinking_level(model, ModelThinkingLevel::from(effort_level));
					if let Some(effort) = mapped_or(&mapped, effort_level.as_str()) {
						params.insert("thinking".to_string(), json!(effort));
					}
				} else {
					let mapped = lookup_thinking_level(model, ModelThinkingLevel::Off);
					let is_null = matches!(mapped, MappedLevel::Null);
					if !is_null {
						let effort = match &mapped {
							MappedLevel::Value(value) => value.as_str(),
							_ => "none",
						};
						params.insert("thinking".to_string(), json!(effort));
					}
				}
			}
			ThinkingFormat::Openai => {
				if let Some(effort_level) = reasoning_effort {
					// OpenAI-style reasoning_effort
					if compat.supports_reasoning_effort {
						let mapped = lookup_thinking_level(model, ModelThinkingLevel::from(effort_level));
						if let Some(effort) = mapped_or(&mapped, effort_level.as_str()) {
							params.insert("reasoning_effort".to_string(), json!(effort));
						}
					}
				} else if compat.supports_reasoning_effort {
					if let MappedLevel::Value(off_value) = lookup_thinking_level(model, ModelThinkingLevel::Off) {
						params.insert("reasoning_effort".to_string(), json!(off_value));
					}
				}
			}
		}
	}

	// Cap reasoning with a top-level budget field. Independent of
	// thinkingFormat: reasoning and the answer share max_tokens here.
	if let Some(field) = thinking_token_budget_field {
		if let Some(budget) = thinking_budget {
			params.insert(field.as_str().to_string(), json!(budget));
		}
	}

	// OpenRouter provider routing preferences.
	if let Some(routing) = model.compat.as_ref().and_then(|compat| compat.open_router_routing.clone()) {
		params.insert("provider".to_string(), serde_json::to_value(&routing).expect("routing serializes"));
	}

	// Vercel AI Gateway provider routing preferences.
	if let Some(routing) = model.compat.as_ref().and_then(|compat| compat.vercel_gateway_routing.clone()) {
		if routing.only.is_some() || routing.order.is_some() {
			let mut gateway = Map::new();
			if let Some(only) = routing.only {
				gateway.insert("only".to_string(), json!(only));
			}
			if let Some(order) = routing.order {
				gateway.insert("order".to_string(), json!(order));
			}
			params.insert("providerOptions".to_string(), json!({ "gateway": gateway }));
		}
	}

	// Last so model and request sampling parameters override named request fields.
	let level = reasoning_effort
		.map(ModelThinkingLevel::from)
		.unwrap_or(ModelThinkingLevel::Off);
	if let Some(sampling_params) = crate::api::simple_options::resolve_sampling_params(
		model,
		level,
		options.and_then(|options| options.sampling_params.as_ref()),
	) {
		for (key, value) in sampling_params {
			params.insert(key, value);
		}
	}

	Ok(params)
}

// ---------------------------------------------------------------------------
// Streaming state
// ---------------------------------------------------------------------------

type GrammarBuffer = crate::api::constrained_sampling::GrammarToolInputJsonBuffer;

struct ToolCallScratch {
	partial_args: Option<String>,
	custom_input: Option<(String, GrammarBuffer)>,
}

struct StreamState {
	output: AssistantMessage,
	text_block: Option<usize>,
	thinking_block: Option<usize>,
	tool_blocks_by_index: HashMap<u64, usize>,
	tool_blocks_by_id: HashMap<String, usize>,
	scratch: HashMap<usize, ToolCallScratch>,
	streamed_reasoning_details: Option<Vec<Value>>,
	has_finish_reason: bool,
}

impl StreamState {
	fn new(model: &Model) -> StreamState {
		StreamState {
			output: AssistantMessage {
				content: Vec::new(),
				api: model.api().to_string(),
				provider: model.provider().to_string(),
				model: model.id().to_string(),
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
					cost: Default::default(),
				},
				stop_reason: StopReason::Pending,
				deferred: None,
				error_message: None,
				raw_stop_reason: None,
				end_turn: None,
				timestamp: now_ms(),
			},
			text_block: None,
			thinking_block: None,
			tool_blocks_by_index: HashMap::new(),
			tool_blocks_by_id: HashMap::new(),
			scratch: HashMap::new(),
			streamed_reasoning_details: None,
			has_finish_reason: false,
		}
	}

	/// Apply streamed `reasoning_details` as the thinking block signature.
	fn apply_streamed_reasoning_details(&self, block: &mut ThinkingContent) {
		if let Some(details) = &self.streamed_reasoning_details {
			block.thinking_signature = Some(Value::Array(details.clone()).to_string());
		}
	}
}

fn ensure_text_block(state: &mut StreamState, stream: &AssistantMessageEventStream) -> usize {
	if let Some(index) = state.text_block {
		return index;
	}
	state.output.content.push(AssistantContent::Text(TextContent {
		text: String::new(),
		text_signature: None,
	}));
	let index = state.output.content.len() - 1;
	state.text_block = Some(index);
	stream.push(crate::types::AssistantMessageEvent::TextStart {
		content_index: index as u32,
		partial: state.output.clone(),
	});
	index
}

fn ensure_thinking_block(
	state: &mut StreamState,
	stream: &AssistantMessageEventStream,
	thinking_signature: &str,
) -> usize {
	if let Some(index) = state.thinking_block {
		return index;
	}
	state.output.content.push(AssistantContent::Thinking(ThinkingContent {
		thinking: String::new(),
		thinking_signature: Some(thinking_signature.to_string()),
		redacted: None,
	}));
	let index = state.output.content.len() - 1;
	state.thinking_block = Some(index);
	stream.push(crate::types::AssistantMessageEvent::ThinkingStart {
		content_index: index as u32,
		partial: state.output.clone(),
	});
	index
}

fn get_custom_tool_call_input(block: &ToolCall, scratch: &ToolCallScratch) -> String {
	let Some((property, _)) = scratch.custom_input.as_ref() else { return String::new() };
	match block.arguments.get(property) {
		Some(Value::String(value)) => value.clone(),
		_ => String::new(),
	}
}

fn append_custom_tool_call_input(
	block: &mut ToolCall,
	scratch: &mut ToolCallScratch,
	next_input: &str,
	close: bool,
) -> Result<Option<String>, String> {
	let Some((property, buffer)) = scratch.custom_input.as_mut() else { return Ok(None) };
	let delta = crate::api::constrained_sampling::append_grammar_tool_input_json_delta(
		buffer,
		property,
		next_input,
		close,
	)?;
	let mut arguments = Map::new();
	arguments.insert(property.clone(), json!(next_input));
	block.arguments = arguments;
	Ok(delta)
}

#[allow(clippy::too_many_arguments)]
fn finish_block(
	state: &mut StreamState,
	index: usize,
	stream: &AssistantMessageEventStream,
) -> Result<(), String> {
	use crate::types::AssistantMessageEvent as Event;
	// Reasoning-detail signatures are computed up front so the per-block
	// mutable borrows below never overlap with `state`.
	let reasoning_signature = state
		.streamed_reasoning_details
		.as_ref()
		.map(|details| Value::Array(details.clone()).to_string());
	let partial = state.output.clone();
	let Some(block) = state.output.content.get_mut(index) else { return Ok(()) };
	match block {
		AssistantContent::Text(text) => {
			let content = text.text.clone();
			stream.push(Event::TextEnd { content_index: index as u32, content, partial });
		}
		AssistantContent::Thinking(thinking) => {
			if let Some(signature) = reasoning_signature {
				thinking.thinking_signature = Some(signature);
			}
			let content = thinking.thinking.clone();
			stream.push(Event::ThinkingEnd { content_index: index as u32, content, partial });
		}
		AssistantContent::ToolCall(tool_call) => {
			let is_custom = state.scratch.get(&index).is_some_and(|scratch| scratch.custom_input.is_some());
			if is_custom {
				// Disjoint field borrows: scratch buffer and the block itself.
				let Some(scratch) = state.scratch.get_mut(&index) else { unreachable!("checked") };
				let next_input = get_custom_tool_call_input(tool_call, scratch);
				let delta = append_custom_tool_call_input(tool_call, scratch, &next_input, true)?;
				if let Some(delta) = delta {
					stream.push(Event::ToolcallDelta { content_index: index as u32, delta, partial });
				}
			} else if let Some(scratch) = state.scratch.get_mut(&index) {
				let parsed = parse_streaming_json(scratch.partial_args.as_deref());
				tool_call.arguments = parsed.as_object().cloned().unwrap_or_default();
			}
			// Finalize in place and strip the scratch buffers so replay only
			// carries parsed arguments.
			state.scratch.remove(&index);
			let partial = state.output.clone();
			let tool_call = match state.output.content.get(index) {
				Some(AssistantContent::ToolCall(tool_call)) => tool_call.clone(),
				_ => unreachable!("tool call block"),
			};
			stream.push(Event::ToolcallEnd { content_index: index as u32, tool_call, partial });
		}
	}
	Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ensure_tool_call_block(
	state: &mut StreamState,
	stream: &AssistantMessageEventStream,
	tool_call: &Value,
	grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<usize, String> {
	let function = tool_call.get("function");
	let custom = tool_call.get("custom");
	let stream_index = tool_call.get("index").and_then(Value::as_u64);
	let name = function
		.and_then(|function| function.get("name"))
		.and_then(Value::as_str)
		.or(custom.and_then(|custom| custom.get("name")).and_then(Value::as_str))
		.unwrap_or_default()
		.to_string();

	let mut block_index = stream_index.and_then(|index| state.tool_blocks_by_index.get(&index).copied());
	if block_index.is_none() {
		if let Some(id) = tool_call.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()) {
			block_index = state.tool_blocks_by_id.get(id).copied();
		}
	}

	let block_index = match block_index {
		Some(index) => index,
		None => {
			// Note: the "input" fallback here should/must not be taken in case
			// the LLM makes up a tool we don't know about; we at least have a
			// place to stash our stuff.
			let has_custom = custom.is_some() && function.is_none();
			let custom_input_property = if has_custom {
				Some(
					grammar_tool_input_properties
						.get(&name)
						.cloned()
						.unwrap_or_else(|| "input".to_string()),
				)
			} else {
				None
			};
			let id = tool_call
				.get("id")
				.and_then(Value::as_str)
				.filter(|id| !id.is_empty())
				.unwrap_or_default()
				.to_string();
			let (arguments, scratch) = match &custom_input_property {
				Some(property) => {
					let mut arguments = Map::new();
					arguments.insert(property.clone(), json!(""));
					(
						arguments,
						ToolCallScratch {
							partial_args: None,
							custom_input: Some((
								property.clone(),
								GrammarBuffer { input: String::new(), started: false, closed: false },
							)),
						},
					)
				}
				None => (
					Map::new(),
					ToolCallScratch { partial_args: Some(String::new()), custom_input: None },
				),
			};
			state.output.content.push(AssistantContent::ToolCall(ToolCall {
				id,
				name: String::new(),
				arguments,
				thought_signature: None,
				namespace: None,
			}));
			let index = state.output.content.len() - 1;
			if let Some(stream_index) = stream_index {
				state.tool_blocks_by_index.insert(stream_index, index);
			}
			if let Some(id) = tool_call.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()) {
				state.tool_blocks_by_id.insert(id.to_string(), index);
			}
			state.scratch.insert(index, scratch);
			stream.push(crate::types::AssistantMessageEvent::ToolcallStart {
				content_index: index as u32,
				partial: state.output.clone(),
			});
			index
		}
	};

	if let Some(stream_index) = stream_index {
		state.tool_blocks_by_index.entry(stream_index).or_insert(block_index);
	}
	if let Some(id) = tool_call.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()) {
		state.tool_blocks_by_id.insert(id.to_string(), block_index);
	}
	{
		let AssistantContent::ToolCall(block) =
			state.output.content.get_mut(block_index).expect("block exists")
		else {
			unreachable!("tool call block");
		};
		if block.name.is_empty() && !name.is_empty() {
			block.name = name.clone();
		}
	}
	if custom.is_some() && function.is_none() {
		let needs_custom_input = matches!(
			state.scratch.get(&block_index).and_then(|scratch| scratch.custom_input.as_ref()),
			None
		);
		if needs_custom_input {
			let custom_input_property = grammar_tool_input_properties
				.get(&state_tool_call_name(state, block_index))
				.cloned()
				.unwrap_or_else(|| "input".to_string());
			let AssistantContent::ToolCall(block) =
				state.output.content.get_mut(block_index).expect("block exists")
			else {
				unreachable!("tool call block");
			};
			let mut arguments = Map::new();
			arguments.insert(custom_input_property.clone(), json!(""));
			block.arguments = arguments;
			state.scratch.insert(
				block_index,
				ToolCallScratch {
					partial_args: None,
					custom_input: Some((
						custom_input_property,
						GrammarBuffer { input: String::new(), started: false, closed: false },
					)),
				},
			);
		}
	}
	Ok(block_index)
}

fn state_tool_call_name(state: &StreamState, index: usize) -> String {
	match state.output.content.get(index) {
		Some(AssistantContent::ToolCall(tool_call)) => tool_call.name.clone(),
		_ => String::new(),
	}
}

// ---------------------------------------------------------------------------
// stream()
// ---------------------------------------------------------------------------

fn chunk_choice<'a>(chunk: &'a Value) -> Option<&'a Value> {
	chunk.get("choices").and_then(Value::as_array).and_then(|choices| choices.first())
}

fn process_chunk(
	state: &mut StreamState,
	stream: &AssistantMessageEventStream,
	chunk: &Value,
	model: &Model,
	compat: &ResolvedCompat,
	grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<(), String> {
	use crate::types::AssistantMessageEvent as Event;

	// OpenAI documents ChatCompletionChunk.id as the unique chat completion
	// identifier; each chunk carries the same id.
	if !state.output.response_id.is_some() {
		if let Some(id) = chunk.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()) {
			state.output.response_id = Some(id.to_string());
		}
	}
	if let Some(response_model) = chunk.get("model").and_then(Value::as_str).filter(|model| !model.is_empty()) {
		if response_model != model.id() && state.output.response_model.is_none() {
			state.output.response_model = Some(response_model.to_string());
		}
	}
	if chunk.get("usage").is_some_and(|usage| usage.is_object()) {
		state.output.usage = parse_chunk_usage(chunk.get("usage").expect("object"), model);
	}

	let Some(choice) = chunk_choice(chunk) else { return Ok(()) };

	// Fallback: some providers (e.g., Moonshot) return usage in choice.usage
	// instead of the standard chunk.usage.
	if !chunk.get("usage").is_some_and(|usage| usage.is_object()) && choice.get("usage").is_some_and(Value::is_object) {
		state.output.usage = parse_chunk_usage(choice.get("usage").expect("object"), model);
	}

	if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
		state.output.raw_stop_reason = Some(finish_reason.to_string());
		let (stop_reason, error_message) = map_stop_reason(finish_reason);
		state.output.stop_reason = stop_reason;
		if let Some(error_message) = error_message {
			state.output.error_message = Some(error_message);
		}
		state.has_finish_reason = true;
	}

	let Some(delta) = choice.get("delta") else { return Ok(()) };

	if let Some(content) = delta.get("content").and_then(Value::as_str).filter(|content| !content.is_empty()) {
		let index = ensure_text_block(state, stream);
		if let Some(AssistantContent::Text(block)) = state.output.content.get_mut(index) {
			block.text.push_str(content);
		}
		stream.push(Event::TextDelta {
			content_index: index as u32,
			delta: content.to_string(),
			partial: state.output.clone(),
		});
	}

	// Some endpoints return reasoning in reasoning_content (llama.cpp), or
	// reasoning / reasoning_text (other OpenAI-compatible endpoints). Use the
	// first non-empty reasoning field to avoid duplication.
	let mut found_reasoning_field: Option<&str> = None;
	for field in OPENAI_COMPLETIONS_REASONING_FIELDS {
		if let Some(value) = delta.get(field).and_then(Value::as_str) {
			if !value.is_empty() {
				found_reasoning_field = Some(field);
				break;
			}
		}
	}
	if let Some(field) = found_reasoning_field {
		let reasoning_signature = if model.provider() == "opencode-go" && field == "reasoning" {
			"reasoning_content"
		} else {
			field
		};
		let index = ensure_thinking_block(state, stream, reasoning_signature);
		let delta_text = delta.get(field).and_then(Value::as_str).unwrap_or_default();
		if !delta_text.is_empty() {
			if let Some(AssistantContent::Thinking(block)) = state.output.content.get_mut(index) {
				block.thinking.push_str(delta_text);
			}
			stream.push(Event::ThinkingDelta {
				content_index: index as u32,
				delta: delta_text.to_string(),
				partial: state.output.clone(),
			});
		}
	}

	if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
		for tool_call in tool_calls {
			let block_index = ensure_tool_call_block(state, stream, tool_call, grammar_tool_input_properties)?;
			if let Some(id) = tool_call.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()) {
				let needs_id = match state.output.content.get(block_index) {
					Some(AssistantContent::ToolCall(block)) => block.id.is_empty(),
					_ => false,
				};
				if needs_id {
					if let Some(AssistantContent::ToolCall(block)) = state.output.content.get_mut(block_index) {
						block.id = id.to_string();
					}
					state.tool_blocks_by_id.insert(id.to_string(), block_index);
				}
			}
			let name = tool_call
				.get("function")
				.and_then(|function| function.get("name"))
				.and_then(Value::as_str)
				.or(tool_call
					.get("custom")
					.and_then(|custom| custom.get("name"))
					.and_then(Value::as_str));
			if let Some(name) = name.filter(|name| !name.is_empty()) {
				let is_empty = match state.output.content.get(block_index) {
					Some(AssistantContent::ToolCall(block)) => block.name.is_empty(),
					_ => false,
				};
				if is_empty {
					if let Some(AssistantContent::ToolCall(block)) = state.output.content.get_mut(block_index) {
						block.name = name.to_string();
					}
				}
			}

			let mut delta_text = String::new();
			if let Some(arguments) = tool_call
				.get("function")
				.and_then(|function| function.get("arguments"))
				.and_then(Value::as_str)
				.filter(|arguments| !arguments.is_empty())
			{
				delta_text = arguments.to_string();
				let partial_args = {
					let Some(scratch) = state.scratch.get_mut(&block_index) else {
						return Err("missing tool call scratch state".to_string());
					};
					let buffer = scratch.partial_args.get_or_insert_with(String::new);
					buffer.push_str(arguments);
					buffer.clone()
				};
				let parsed = parse_streaming_json(Some(&partial_args));
				if let Some(AssistantContent::ToolCall(block)) = state.output.content.get_mut(block_index) {
					block.arguments = parsed.as_object().cloned().unwrap_or_default();
				}
			} else if let Some(custom_input) = tool_call
				.get("custom")
				.and_then(|custom| custom.get("input"))
				.and_then(Value::as_str)
				.filter(|input| !input.is_empty())
			{
				let (next_input, result) = {
					let Some(scratch) = state.scratch.get_mut(&block_index) else {
						return Err("missing tool call scratch state".to_string());
					};
					let current = {
						let Some(AssistantContent::ToolCall(block)) = state.output.content.get(block_index) else {
							unreachable!("tool call block");
						};
						get_custom_tool_call_input(block, scratch)
					};
					let next_input = format!("{current}{custom_input}");
					// borrow split: mutate below
					(next_input, ())
				};
				let _ = result;
				let delta_result = {
					let Some(scratch) = state.scratch.get_mut(&block_index) else {
						return Err("missing tool call scratch state".to_string());
					};
					let Some(AssistantContent::ToolCall(block)) = state.output.content.get_mut(block_index) else {
						unreachable!("tool call block");
					};
					append_custom_tool_call_input(block, scratch, &next_input, false)
				};
				delta_text = delta_result?.unwrap_or_default();
			}
			stream.push(Event::ToolcallDelta {
				content_index: block_index as u32,
				delta: delta_text,
				partial: state.output.clone(),
			});
		}
	}

	// `reasoning_details` are replay metadata, not user-visible deltas.
	if let Some(reasoning_details) = delta.get("reasoning_details").and_then(Value::as_array) {
		for detail in reasoning_details {
			if !is_openai_reasoning_detail(detail) {
				continue;
			}
			ensure_thinking_block(state, stream, "");
			state.streamed_reasoning_details.get_or_insert_with(Vec::new);
			// OpenRouter streams reasoning_details as deltas: consecutive
			// text/summary deltas merge; encrypted entries stay discrete.
			append_openai_reasoning_detail(state.streamed_reasoning_details.as_mut().expect("just set"), detail.clone());
		}
	}

	let _ = compat;
	Ok(())
}

// ---------------------------------------------------------------------------
// API entry points
// ---------------------------------------------------------------------------

/// The `stream` export of `openai-completions.ts`.
pub fn stream(model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
	let stream = AssistantMessageEventStream::new();
	let compat = get_compat(model);
	let normalized_context = resolve_transcript(context, Some(compat.supports_mid_convo_system_messages));

	let stream_handle = stream.clone();
	let model = model.clone();
	let options = options.clone();
	tokio::spawn(async move {
		let mut state = StreamState::new(&model);
		let result = stream_inner(&model, &normalized_context, &options, &compat, &stream_handle, &mut state).await;
		if let Err(error) = result {
			fail_stream(&model, &options, &state, &error, &stream_handle);
		}
	});
	stream
}

/// The upstream `catch`: finalize thinking signatures, mark the partial
/// output aborted/errored, and terminate with the error protocol.
fn fail_stream(
	_model: &Model,
	options: &StreamOptions,
	state: &StreamState,
	error: &ProviderHttpError,
	stream: &AssistantMessageEventStream,
) {
	let mut output = state.output.clone();
	// `reasoning_details` are replay metadata; serialize them into thinking
	// signatures even on the failure path.
	for block in output.content.iter_mut() {
		if let AssistantContent::Thinking(thinking) = block {
			state.apply_streamed_reasoning_details(thinking);
		}
	}
	output.stop_reason = if options.signal.as_ref().is_some_and(|signal| signal.aborted()) {
		StopReason::Aborted
	} else {
		StopReason::Error
	};
	let mut error_message = format_provider_error(&normalize_provider_error(error), None);
	// Some providers via OpenRouter give additional information in
	// error.error.metadata.raw; only append when not already present.
	if let Some(raw) = raw_metadata_suffix(error, &error_message) {
		error_message.push('\n');
		error_message.push_str(&raw);
	}
	output.error_message = Some(error_message);
	let reason = StreamErrorReason::from(output.stop_reason);
	stream.push(error_event(output, reason));
	stream.end(None);
}

impl From<StopReason> for StreamErrorReason {
	fn from(stop_reason: StopReason) -> Self {
		match stop_reason {
			StopReason::Aborted => StreamErrorReason::Aborted,
			_ => StreamErrorReason::Error,
		}
	}
}

/// Synchronously-failing `stream`/`streamSimple` setup (missing API key
/// upstream throws) becomes the stream error protocol (crate docs, models.rs).
fn setup_error_stream(model: &Model, message: String) -> AssistantMessageEventStream {
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
		usage: Usage {
			input: 0,
			output: 0,
			cache_read: 0,
			cache_write: 0,
			cache_write_1h: None,
			reasoning: None,
			total_tokens: 0,
			cost: Default::default(),
		},
		stop_reason: StopReason::Error,
		deferred: None,
		error_message: Some(message),
		raw_stop_reason: None,
		end_turn: None,
		timestamp: now_ms(),
	};
	stream.push(error_event(output, StreamErrorReason::Error));
	stream.end(None);
	stream
}

async fn stream_inner(
	model: &Model,
	context: &TranscriptContext,
	options: &StreamOptions,
	compat: &ResolvedCompat,
	stream: &AssistantMessageEventStream,
	state: &mut StreamState,
) -> Result<(), ProviderHttpError> {
	let api_key = get_client_api_key(model.provider(), options.api_key.as_deref(), options.headers.as_ref())
		.map_err(ProviderHttpError::new)?;
	let grammar_tool_input_properties =
		crate::api::constrained_sampling::create_grammar_tool_input_properties(
			&get_declared_tools(context.messages()),
			compat.supports_openai_grammar_tools,
		)
		.map_err(ProviderHttpError::new)?;
	let cache_retention = resolve_cache_retention(options.cache_retention, options.env.as_ref());
	let cache_session_id = if cache_retention == CacheRetention::None { None } else { options.session_id.clone() };

	let mut params = build_params(model, context, Some(options), compat, cache_retention, &grammar_tool_input_properties)
		.map_err(ProviderHttpError::new)?;
	if let Some(hook) = &options.on_payload {
		if let Some(next) = hook(Value::Object(params.clone()), model.clone()).await {
			params = next.as_object().cloned().unwrap_or_default();
		}
	}

	// Build request headers.
	let mut headers: Vec<(String, String)> = vec![("User-Agent".to_string(), get_pi_user_agent())];
	let set_header = |headers: &mut Vec<(String, String)>, name: &str, value: String| {
		if let Some(entry) = headers.iter_mut().find(|(key, _)| key.eq_ignore_ascii_case(name)) {
			entry.1 = value;
		} else {
			headers.push((name.to_string(), value));
		}
	};
	for (name, value) in model.base.headers.iter().flat_map(|headers| headers.iter()) {
		set_header(&mut headers, name, value.clone());
	}
	if model.provider() == "github-copilot" {
		for (name, value) in build_copilot_dynamic_headers(context.messages()) {
			set_header(&mut headers, &name, value);
		}
	}
	if let Some(session_id) = &cache_session_id {
		if compat.send_session_affinity_headers {
			if compat.session_affinity_format == crate::types::SessionAffinityFormat::Openrouter {
				set_header(&mut headers, "x-session-id", session_id.clone());
			} else {
				if compat.session_affinity_format == crate::types::SessionAffinityFormat::Openai {
					set_header(&mut headers, "session_id", session_id.clone());
				}
				set_header(&mut headers, "x-client-request-id", session_id.clone());
				set_header(&mut headers, "x-session-affinity", session_id.clone());
			}
		}
	}
	// Merge options headers last so they can override defaults; `None` values
	// suppress a default header.
	if let Some(options_headers) = &options.headers {
		for (name, value) in options_headers {
			match value {
				Some(value) => set_header(&mut headers, name, value.clone()),
				None => headers.retain(|(key, _)| !key.eq_ignore_ascii_case(name)),
			}
		}
	}
	set_header(&mut headers, "Authorization", format!("Bearer {api_key}"));

	let fetch: FetchFn = options.fetch.clone().unwrap_or_default();
	let url = format!("{}/chat/completions", model.base.base_url.trim_end_matches('/'));
	let mut request = HttpRequest::new("POST", url.clone());
	for (name, value) in &headers {
		request = request.header(name, value);
	}
	request = request.json_body(&Value::Object(params));
	request.timeout_ms = options.timeout_ms;
	request.signal = options.signal.clone();

	let retry_options = ProviderRetryOptions {
		max_retries: options.max_retries,
		max_retry_delay_ms: options.max_retry_delay_ms,
		signal: options.signal.clone(),
	};
	// The retry closure owns a stream-free copy of the request; the response
	// body stream is only held after the final attempt succeeds.
	let base_request = clone_request(&request);
	let mut response = retry_provider_request(
		move || {
			let fetch = fetch.clone();
			let request = clone_request(&base_request);
			async move {
				match fetch.send(request).await {
					Ok(response) if !response.ok() => {
						let body = response.body.collect().await.unwrap_or_default();
						let body = String::from_utf8_lossy(&body).to_string();
						Err(ProviderHttpError {
							message: response.status.to_string(),
							status: Some(response.status),
							headers: response.headers.clone(),
							body: Some(body),
							source: None,
						})
					}
					Ok(response) => Ok(response),
					Err(error) => Err(error),
				}
			}
		},
		retry_options,
	)
	.await?;

	if let Some(hook) = &options.on_response {
		let response_meta = ProviderResponse {
			status: response.status,
			headers: headers_to_record(&response.headers),
		};
		hook(response_meta, model.clone()).await;
	}

	stream.push(crate::types::AssistantMessageEvent::Start { partial: state.output.clone() });

	// SSE decode: buffer bytes, split lines, `data:` frames, `[DONE]` ends.
	let mut buffer: Vec<u8> = Vec::new();
	let mut body_stream = match std::mem::replace(&mut response.body, HttpBody::Empty) {
		HttpBody::Empty => None,
		HttpBody::Bytes(bytes) => {
			buffer.extend_from_slice(&bytes);
			None
		}
		HttpBody::Stream(inner) => Some(inner),
	};
	let mut aborted_mid_stream = false;
	'outer: loop {
		// Drain complete lines already buffered.
		while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
			let line: Vec<u8> = buffer.drain(..=position).collect();
			let mut line = String::from_utf8_lossy(&line[..line.len() - 1]).to_string();
			if line.ends_with('\r') {
				line.pop();
			}
			let Some(data) = line.strip_prefix("data:") else { continue };
			let data = data.strip_prefix(' ').unwrap_or(data);
			if data.trim() == "[DONE]" {
				break 'outer;
			}
			let Ok(chunk) = serde_json::from_str::<Value>(data) else { continue };
			if let Some(hook) = &options.on_provider_stream_event {
				hook(chunk.clone(), model.clone()).await;
			}
			if !chunk.is_object() {
				continue;
			}
			process_chunk(state, stream, &chunk, model, compat, &grammar_tool_input_properties)
				.map_err(ProviderHttpError::new)?;
			if options.signal.as_ref().is_some_and(|signal| signal.aborted()) {
				aborted_mid_stream = true;
				break 'outer;
			}
		}
		let Some(body) = body_stream.as_mut() else { break };
		let next = tokio::select! {
			biased;
			_ = async {
				if let Some(signal) = &options.signal {
					signal.aborted_future().await;
				} else {
					std::future::pending::<()>().await;
				}
			} => {
				aborted_mid_stream = true;
				break 'outer;
			}
			chunk = futures_util::StreamExt::next(body) => chunk,
		};
		match next {
			Some(Ok(bytes)) => {
				if bytes.is_empty() {
					continue;
				}
				buffer.extend_from_slice(&bytes);
			}
			Some(Err(error)) => return Err(ProviderHttpError::new(error.to_string())),
			None => break,
		}
	}

	if aborted_mid_stream {
		return Err(ProviderHttpError::new("Request was aborted".to_string()));
	}

	// Finalize blocks in order.
	for index in 0..state.output.content.len() {
		finish_block(state, index, stream).map_err(ProviderHttpError::new)?;
	}

	if options.signal.as_ref().is_some_and(|signal| signal.aborted()) {
		return Err(ProviderHttpError::new("Request was aborted".to_string()));
	}
	if state.output.stop_reason == StopReason::Aborted {
		return Err(ProviderHttpError::new("Request was aborted".to_string()));
	}
	if !state.has_finish_reason && !compat.supports_finish_reason {
		state.output.stop_reason =
			if state.output.content.iter().any(|block| matches!(block, AssistantContent::ToolCall(_))) {
				StopReason::ToolUse
			} else {
				StopReason::Stop
			};
	}
	if state.output.stop_reason == StopReason::Error {
		return Err(ProviderHttpError::new(
			state
				.output
				.error_message
				.clone()
				.unwrap_or_else(|| "Provider returned an error stop reason".to_string()),
		));
	}
	if (compat.supports_finish_reason && !state.has_finish_reason)
		|| state.output.stop_reason == StopReason::Pending
	{
		return Err(ProviderHttpError::new("Stream ended without finish_reason".to_string()));
	}

	let reason = match state.output.stop_reason {
		StopReason::Length => StreamDoneReason::Length,
		StopReason::ToolUse => StreamDoneReason::ToolUse,
		_ => StreamDoneReason::Stop,
	};
	stream.push(done_event(state.output.clone(), reason));
	stream.end(None);
	Ok(())
}

fn clone_request(request: &HttpRequest) -> HttpRequest {
	let mut cloned = HttpRequest::new(request.method.clone(), request.url.clone());
	cloned.headers = request.headers.clone();
	cloned.body = match &request.body {
		HttpBody::Empty => HttpBody::Empty,
		HttpBody::Bytes(bytes) => HttpBody::Bytes(bytes.clone()),
		HttpBody::Stream(_) => HttpBody::Empty,
	};
	cloned.timeout_ms = request.timeout_ms;
	cloned.signal = request.signal.clone();
	cloned
}

/// The `streamSimple` export of `openai-completions.ts`.
pub fn stream_simple(
	model: &Model,
	context: &TranscriptContext,
	options: &SimpleStreamOptions,
) -> AssistantMessageEventStream {
	if let Err(message) = get_client_api_key(model.provider(), options.api_key.as_deref(), options.headers.as_ref()) {
		return setup_error_stream(model, message);
	}

	let mut base = crate::api::simple_options::build_base_options(model, context, options, options.api_key.as_deref());
	let clamped_reasoning = options.reasoning.map(|level| crate::models::clamp_thinking_level(model, level.into()));
	let reasoning_effort = match clamped_reasoning {
		None | Some(ModelThinkingLevel::Off) => None,
		Some(ModelThinkingLevel::Minimal) => Some(ThinkingLevel::Minimal),
		Some(ModelThinkingLevel::Low) => Some(ThinkingLevel::Low),
		Some(ModelThinkingLevel::Medium) => Some(ThinkingLevel::Medium),
		Some(ModelThinkingLevel::High) => Some(ThinkingLevel::High),
		Some(ModelThinkingLevel::Xhigh) => Some(ThinkingLevel::Xhigh),
		Some(ModelThinkingLevel::Max) => Some(ThinkingLevel::Max),
	};
	base.openai_completions.tool_choice = options.tool_choice.as_ref().map(|choice| {
		serde_json::to_value(choice).expect("tool choice serializes")
	});
	base.openai_completions.reasoning_effort = reasoning_effort;
	base.openai_completions.thinking_budgets = options.thinking_budgets.clone();

	stream(model, context, &base)
}

/// `<api>_api()` factory (models.rs convention).
pub fn openai_completions_api() -> Arc<dyn ProviderStreams> {
	Arc::new(OpenAICompletionsApi)
}

/// The API implementation object (`export const stream`/`streamSimple`).
pub struct OpenAICompletionsApi;

impl ProviderStreams for OpenAICompletionsApi {
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

#[cfg(test)]
mod tests {
	// Port of test/openai-completions-provider-stream-event.test.ts.
	use super::*;
	use crate::http::HttpResponse;
	use crate::types::{Context, UserMessage};
	use crate::utils::transcript::normalize_context;
	use std::sync::Mutex;

	fn openrouter_model() -> Model {
		Model {
			base: crate::types::BaseModelFields {
				id: "openrouter/auto".to_string(),
				name: "OpenRouter Auto".to_string(),
				api: "openai-completions".to_string(),
				provider: "openrouter".to_string(),
				base_url: "https://openrouter.ai/api/v1".to_string(),
				input: vec![ModelInput::Text],
				input_limits: None,
				cost: Default::default(),
				headers: None,
			},
			model_type: None,
			reasoning: false,
			thinking_level_map: None,
			prompt_cache: None,
			context_window: 200_000,
			max_tokens: 8192,
			sampling_params: None,
			sampling_params_by_thinking_level: None,
			compat: None,
		}
	}

	fn sse_fetch(chunks: Vec<Value>) -> FetchFn {
		let mut body = String::new();
		for chunk in chunks {
			body.push_str(&format!("data: {}\n\n", chunk));
		}
		body.push_str("data: [DONE]\n\n");
		let body = body.into_bytes();
		FetchFn::new(move |_request| {
			let body = body.clone();
			Box::pin(async move {
				Ok(HttpResponse {
					status: 200,
					headers: vec![("x-request-id".to_string(), "req-1".to_string())],
					body: HttpBody::Bytes(body),
				})
			})
		})
	}

	// Regression test for #9784.
	#[tokio::test]
	async fn exposes_provider_chunks_including_openrouter_metadata() {
		let first_chunk = json!({
			"id": "chatcmpl-1",
			"model": "anthropic/claude-sonnet-4.6",
			"choices": [{ "index": 0, "delta": { "content": "hello" } }],
		});
		let final_chunk = json!({
			"id": "chatcmpl-1",
			"model": "anthropic/claude-sonnet-4.6",
			"choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
			"usage": {
				"prompt_tokens": 10,
				"completion_tokens": 2,
				"total_tokens": 12,
				"cost": 0.0012,
				"is_byok": false,
			},
			"openrouter_metadata": { "strategy": "direct", "region": "iad" },
		});
		let events: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
		let events_hook = events.clone();

		let mut options = SimpleStreamOptions::default();
		options.base.base.api_key = Some("test".to_string());
		options.base.base.fetch = Some(sse_fetch(vec![first_chunk.clone(), final_chunk.clone()]));
		options.base.on_provider_stream_event = Some(Arc::new(move |data: Value, _model: Model| {
			let events = events_hook.clone();
			Box::pin(async move { events.lock().expect("events lock").push(data) })
		}));

		let context = normalize_context(&Context {
			system_prompt: None,
			messages: vec![Message::User(UserMessage {
				content: crate::types::UserContentField::Text("hi".to_string()),
				timestamp: now_ms(),
			})],
			tools: None,
		});

		let message = stream_simple(&openrouter_model(), &context, &options).result().await;

		assert_eq!(
			message.content,
			vec![AssistantContent::Text(TextContent { text: "hello".to_string(), text_signature: None })]
		);
		assert_eq!(*events.lock().expect("events lock"), vec![first_chunk, final_chunk]);
	}
}
