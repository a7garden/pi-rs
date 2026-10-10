//! Port of `packages/ai/src/api/openai-responses-shared.ts` — the OpenAI
//! Responses wire conversion layer (input items, function/tool calls,
//! reasoning items) and the response SSE event processor.
//!
//! Deviations:
//! - **D13**: upstream leans on the OpenAI SDK's wire types
//!   (`ResponseInputItem`, `ResponseOutputItem`, `OpenAITool`, …). Rust
//!   builds those payloads as ordered [`serde_json::Value`] documents with
//!   exactly the upstream key order (the crate enables `preserve_order`), so
//!   wire bytes match and reasoning items round-trip verbatim through
//!   `thinkingSignature` (`JSON.stringify` upstream).
//! - **D14**: the SDK `AsyncIterable<ResponseStreamEvent>` becomes an
//!   async stream of `serde_json::Value` event payloads; unknown event types
//!   are ignored exactly as the SDK passes them through unhandled.
//! - **D15**: upstream throws from `processResponsesStream`; here the function
//!   returns `Err(String)` and the caller (`openai_responses`) converts that
//!   into the stream `error` event, per the crate's no-throw-across-stream
//!   rule.
//! - **D16**: invalid `thinkingSignature` JSON on replayed thinking blocks is
//!   skipped instead of throwing a `JSON.parse` SyntaxError (same net effect:
//!   the item is not replayed).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::api::constrained_sampling::{
    append_grammar_tool_input_json_delta, get_grammar_tool_input, get_json_schema_tool_parameters,
    resolve_grammar_constrained_sampling, resolve_json_schema_strict_sampling,
    GrammarToolInputJsonBuffer,
};
use crate::api::transform_messages::transform_messages;
use crate::models::calculate_cost;
use crate::types::{
    AnyModel, AssistantContent, AssistantMessage, AssistantMessageEvent, ImageContent, Message,
    Model, ModelInput, StopReason, SystemMessage, TextContent, TextSignaturePhase, TextSignatureV1,
    ThinkingContent, Tool, ToolCall, TranscriptContext, Usage, UsageCost, UserContent,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::hash::short_hash;
use crate::utils::json_parse::parse_streaming_json;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::text::{get_system_message_text, render_system_message_update};
use crate::utils::transcript::{resolve_transcript, resolve_transcript_tools};

// =============================================================================
// Utilities
// =============================================================================

fn encode_text_signature_v1(id: &str, phase: Option<TextSignaturePhase>) -> String {
    let payload = TextSignatureV1 {
        v: 1,
        id: id.to_string(),
        phase,
    };
    serde_json::to_string(&payload).expect("text signature serializes")
}

fn parse_text_signature(signature: Option<&str>) -> Option<(String, Option<TextSignaturePhase>)> {
    let signature = signature?;
    if signature.starts_with('{') {
        // Upstream tries JSON.parse and falls back to the legacy plain-string
        // form when the payload is not a v1 signature.
        if let Ok(parsed) = serde_json::from_str::<Value>(signature) {
            let v = parsed.get("v").and_then(Value::as_i64);
            let id = parsed.get("id").and_then(Value::as_str);
            if v == Some(1) {
                if let Some(id) = id {
                    let phase = parsed.get("phase").and_then(Value::as_str).and_then(|p| {
                        serde_json::from_str::<TextSignaturePhase>(&format!("\"{p}\"")).ok()
                    });
                    return Some((id.to_string(), phase));
                }
            }
        }
    }
    Some((signature.to_string(), None))
}

/// Output of a tool call sent back to the API: plain text or a text/image
/// array (`ToolResultOutputContent`).
fn convert_tool_result_output(model: &Model, content: &[UserContent]) -> Value {
    let text_result = content
        .iter()
        .filter_map(|block| match block {
            UserContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let images: Vec<&ImageContent> = content
        .iter()
        .filter_map(|block| match block {
            UserContent::Image(image) => Some(image),
            _ => None,
        })
        .collect();
    let has_text = !text_result.is_empty();

    if images.is_empty() || !model.base.input.contains(&ModelInput::Image) {
        let fallback = if has_text {
            text_result
        } else if !images.is_empty() {
            "(see attached image)".to_string()
        } else {
            "(no tool output)".to_string()
        };
        return json!(sanitize_surrogates(&fallback));
    }

    let mut output = Vec::new();
    if has_text {
        output.push(json!({ "type": "input_text", "text": sanitize_surrogates(&text_result) }));
    }
    for image in images {
        output.push(json!({
            "type": "input_image",
            "detail": "auto",
            "image_url": format!("data:{};base64,{}", image.mime_type, image.data),
        }));
    }
    Value::Array(output)
}

/// Upstream `OpenAIResponsesStreamOptions`.
#[derive(Clone, Default)]
pub struct OpenAIResponsesStreamOptions {
    pub on_provider_stream_event: Option<crate::types::ProviderStreamEventHook>,
    /// OpenAI service tier echoed on the request (`service_tier`).
    pub service_tier: Option<Value>,
    /// Grammar-tool input property per tool name (see
    /// [`create_grammar_tool_input_properties`]).
    pub grammar_tool_input_properties: BTreeMap<String, String>,
    /// Resolve the effective service tier from (response tier, request tier).
    pub resolve_service_tier:
        Option<Arc<dyn Fn(Option<&Value>, Option<&Value>) -> Option<Value> + Send + Sync>>,
    /// Apply service-tier pricing to the computed usage.
    pub apply_service_tier_pricing: Option<Arc<dyn Fn(&mut Usage, Option<&Value>) + Send + Sync>>,
}

/// Upstream `ConvertResponsesMessagesOptions`.
#[derive(Debug, Clone, Default)]
pub struct ConvertResponsesMessagesOptions {
    pub include_system_prompt: Option<bool>,
    pub grammar_tool_input_properties: Option<BTreeMap<String, String>>,
    /// Whether later system messages are sent in place; otherwise they are
    /// folded into the leading prompt.
    pub supports_mid_convo_system_messages: bool,
    pub supports_additional_tools: bool,
    pub supports_tool_search: bool,
    pub tool_options: Option<ConvertResponsesToolsOptions>,
}

/// Upstream `ConvertResponsesToolsOptions`.
#[derive(Debug, Clone, Default)]
pub struct ConvertResponsesToolsOptions {
    pub strict: Option<bool>,
    pub supports_strict_mode: Option<bool>,
    pub supports_openai_grammar_tools: Option<bool>,
    pub tool_search_result: bool,
}

// =============================================================================
// Message conversion
// =============================================================================

/// Build the Responses `input` array from a transcript
/// (`convertResponsesMessages`). Returns `Err` with the upstream thrown
/// message when grammar-tool arguments are malformed.
pub fn convert_responses_messages(
    model: &Model,
    context: &TranscriptContext,
    allowed_tool_call_providers: &std::collections::BTreeSet<String>,
    options: Option<&ConvertResponsesMessagesOptions>,
) -> Result<Vec<Value>, String> {
    let normalized_context = resolve_transcript(
        context,
        options.map(|o| o.supports_mid_convo_system_messages),
    );
    let mut messages: Vec<Value> = Vec::new();

    let normalize_id_part = |part: &str| -> String {
        let sanitized: String = part
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let truncated: String = sanitized.chars().take(64).collect();
        let trimmed = truncated.trim_end_matches('_');
        trimmed.to_string()
    };

    let build_foreign_responses_item_id = |item_id: &str| -> String {
        let normalized = format!("fc_{}", short_hash(item_id));
        if normalized.chars().count() > 64 {
            normalized.chars().take(64).collect()
        } else {
            normalized
        }
    };

    let normalize_tool_call_id = |id: &str, _model: &Model, source: &AssistantMessage| -> String {
        if !allowed_tool_call_providers.contains(model.provider()) {
            return normalize_id_part(id);
        }
        if !id.contains('|') {
            return normalize_id_part(id);
        }
        let (call_id, item_id) = id.split_once('|').expect("id contains |");
        let normalized_call_id = normalize_id_part(call_id);
        let is_foreign_tool_call = source.provider != model.provider() || source.api != model.api();
        let mut normalized_item_id = if is_foreign_tool_call {
            build_foreign_responses_item_id(item_id)
        } else {
            normalize_id_part(item_id)
        };
        // OpenAI Responses API requires item id to start with "fc"
        if !normalized_item_id.starts_with("fc_") {
            normalized_item_id = normalize_id_part(&format!("fc_{normalized_item_id}"));
        }
        format!("{normalized_call_id}|{normalized_item_id}")
    };

    let transformed_messages = transform_messages(
        normalized_context.messages(),
        model,
        Some(&normalize_tool_call_id),
    );
    let transcript_tools = resolve_transcript_tools(
        normalized_context.messages(),
        options
            .map(|o| o.supports_additional_tools || o.supports_tool_search)
            .unwrap_or(false),
    );

    let append_system_tool_additions =
        |messages: &mut Vec<Value>, message: &SystemMessage, seed: &str| -> Result<(), String> {
            let tools: &[Tool] = if transcript_tools.anchors_additions {
                message.tools_added.as_deref().unwrap_or(&[])
            } else {
                &[]
            };
            if tools.is_empty() {
                return Ok(());
            }
            let Some(opts) = options else { return Ok(()) };
            if opts.supports_additional_tools {
                messages.push(json!({
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": convert_responses_tools(tools, opts.tool_options.as_ref())?,
                }));
                return Ok(());
            }
            if !opts.supports_tool_search {
                return Ok(());
            }
            let names: Vec<&str> = tools.iter().map(|tool| tool.name.as_str()).collect();
            let call_id = format!(
                "pi_tool_load_{}",
                short_hash(&format!("{seed}:{}", names.join(",")))
            );
            messages.push(json!({
                "type": "tool_search_call",
                "call_id": call_id,
                "execution": "client",
                "status": "completed",
                "arguments": { "query": names.join(" "), "limit": names.len() },
            }));
            let mut tool_options = opts.tool_options.clone().unwrap_or_default();
            tool_options.tool_search_result = true;
            messages.push(json!({
                "type": "tool_search_output",
                "call_id": call_id,
                "execution": "client",
                "status": "completed",
                "tools": convert_responses_tools(tools, Some(&tool_options))?,
            }));
            Ok(())
        };

    let include_initial_system_message = options
        .map(|o| o.include_system_prompt)
        .unwrap_or(None)
        .unwrap_or(true);
    let supports_developer_role = model
        .compat
        .as_ref()
        .and_then(|c| c.supports_developer_role);
    let instruction_role = if model.reasoning && supports_developer_role != Some(false) {
        "developer"
    } else {
        "system"
    };

    let mut msg_index = 0usize;
    let mut source_index = 0usize;
    for msg in &transformed_messages {
        let is_leading_system_message = source_index == 0 && msg.role() == "system";
        source_index += 1;
        match msg {
            Message::System(system) => {
                if !is_leading_system_message {
                    append_system_tool_additions(
                        &mut messages,
                        system,
                        &format!("system:{msg_index}"),
                    )?;
                }
                if !is_leading_system_message || include_initial_system_message {
                    let text = if is_leading_system_message {
                        get_system_message_text(system)
                    } else {
                        render_system_message_update(system)
                    };
                    if !text.is_empty() {
                        messages.push(json!({
                            "role": instruction_role,
                            "content": sanitize_surrogates(&text),
                        }));
                    }
                }
            }
            Message::User(user) => match &user.content {
                crate::types::UserContentField::Text(text) => {
                    messages.push(json!({
							"role": "user",
							"content": [{ "type": "input_text", "text": sanitize_surrogates(text.as_str()) }],
						}));
                }
                crate::types::UserContentField::Blocks(blocks) => {
                    let content: Vec<Value> = blocks
							.iter()
							.map(|item| match item {
								UserContent::Text(text) => json!({
									"type": "input_text",
									"text": sanitize_surrogates(&text.text),
								}),
								UserContent::Image(image) => json!({
									"type": "input_image",
									"detail": "auto",
									"image_url": format!("data:{};base64,{}", image.mime_type, image.data),
								}),
							})
							.collect();
                    if content.is_empty() {
                        if !is_leading_system_message {
                            msg_index += 1;
                        }
                        continue;
                    }
                    messages.push(json!({
                        "role": "user",
                        "content": content,
                    }));
                }
            },
            Message::Assistant(assistant) => {
                let mut output: Vec<Value> = Vec::new();
                let is_same_provider_and_api =
                    assistant.provider == model.provider() && assistant.api == model.api();
                let is_same_model = is_same_provider_and_api && assistant.model == model.id();
                let is_different_model = is_same_provider_and_api && assistant.model != model.id();
                let mut text_block_index = 0usize;

                for block in &assistant.content {
                    match block {
                        AssistantContent::Thinking(thinking) => {
                            if let Some(signature) = &thinking.thinking_signature {
                                // Upstream JSON.parses the signature and replays the
                                // reasoning item verbatim (D15/D16: invalid JSON is
                                // skipped rather than thrown).
                                if let Ok(item) = serde_json::from_str::<Value>(signature) {
                                    output.push(item);
                                }
                            }
                        }
                        AssistantContent::Text(text_block) => {
                            let parsed_signature =
                                parse_text_signature(text_block.text_signature.as_deref());
                            let fallback_message_id = if text_block_index == 0 {
                                format!("msg_pi_{msg_index}")
                            } else {
                                format!("msg_pi_{msg_index}_{}", text_block_index)
                            };
                            text_block_index += 1;
                            // OpenAI requires id to be max 64 characters
                            let mut msg_id = parsed_signature.as_ref().map(|(id, _)| id.clone());
                            if msg_id.is_none() {
                                msg_id = Some(fallback_message_id);
                            } else if msg_id.as_ref().expect("some").chars().count() > 64 {
                                msg_id = Some(format!(
                                    "msg_{}",
                                    short_hash(msg_id.as_ref().expect("some"))
                                ));
                            }
                            let msg_id = msg_id.expect("message id");
                            let mut item = json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": sanitize_surrogates(&text_block.text),
                                    "annotations": [],
                                }],
                                "status": "completed",
                                "id": msg_id,
                            });
                            if let Some((_, Some(phase))) = &parsed_signature {
                                item["phase"] = json!(phase);
                            }
                            output.push(item);
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let (call_id, item_id_raw) = match tool_call.id.split_once('|') {
                                Some((call_id, item_id)) => (call_id, Some(item_id)),
                                None => (tool_call.id.as_str(), None),
                            };
                            let custom_input_property = options
                                .and_then(|o| o.grammar_tool_input_properties.as_ref())
                                .and_then(|props| props.get(&tool_call.name).cloned());
                            let mut item_id: Option<String> = item_id_raw.map(str::to_string);

                            // For different-model messages, set id to undefined to avoid
                            // pairing validation. Also drop ids that do not match the
                            // replayed item type: function_call ids must be fc_* and
                            // custom_tool_call ids must be ctc_*.
                            let item_id_prefix = if custom_input_property.is_none() {
                                "fc_"
                            } else {
                                "ctc_"
                            };
                            let has_prefix = item_id
                                .as_deref()
                                .map(|id| id.starts_with(item_id_prefix))
                                .unwrap_or(false);
                            if is_different_model || !has_prefix {
                                item_id = None;
                            }

                            let namespace = if is_same_model {
                                tool_call.namespace.clone()
                            } else {
                                None
                            };
                            if let Some(custom_input_property) = custom_input_property {
                                let input = get_grammar_tool_input(
                                    &tool_call.name,
                                    &tool_call.arguments,
                                    &custom_input_property,
                                )?;
                                let mut item = json!({
                                    "type": "custom_tool_call",
                                    "id": item_id,
                                    "call_id": call_id,
                                    "name": tool_call.name,
                                    "input": sanitize_surrogates(&input),
                                });
                                if let Some(namespace) = namespace {
                                    item["namespace"] = json!(namespace);
                                }
                                output.push(item);
                            } else {
                                let mut item = json!({
                                    "type": "function_call",
                                    "id": item_id,
                                    "call_id": call_id,
                                    "name": tool_call.name,
                                    "arguments": serde_json::to_string(&tool_call.arguments)
                                        .expect("arguments serialize"),
                                });
                                if let Some(namespace) = namespace {
                                    item["namespace"] = json!(namespace);
                                }
                                output.push(item);
                            }
                        }
                    }
                }
                if output.is_empty() {
                    if !is_leading_system_message {
                        msg_index += 1;
                    }
                    continue;
                }
                messages.extend(output);
            }
            Message::ToolResult(tool_result) => {
                let call_id = tool_result.tool_call_id.split('|').next().unwrap_or("");
                let tool_output = convert_tool_result_output(model, &tool_result.content);

                let is_custom = options
                    .and_then(|o| o.grammar_tool_input_properties.as_ref())
                    .map(|props| props.contains_key(&tool_result.tool_name))
                    .unwrap_or(false);
                if is_custom {
                    messages.push(json!({
                        "type": "custom_tool_call_output",
                        "call_id": call_id,
                        "output": tool_output,
                    }));
                } else {
                    messages.push(json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": tool_output,
                    }));
                }
            }
        }
        if !is_leading_system_message {
            msg_index += 1;
        }
    }

    Ok(messages)
}

// =============================================================================
// Tool conversion
// =============================================================================

/// Convert pi tools to the Responses wire tool list (`convertResponsesTools`).
pub fn convert_responses_tools(
    tools: &[Tool],
    options: Option<&ConvertResponsesToolsOptions>,
) -> Result<Vec<Value>, String> {
    let default_strict = options.and_then(|o| o.strict).unwrap_or(false);
    let supports_strict_mode = options.and_then(|o| o.supports_strict_mode).unwrap_or(true);
    let supports_openai_grammar_tools = options
        .and_then(|o| o.supports_openai_grammar_tools)
        .unwrap_or(false);
    let tool_search_result = options.map(|o| o.tool_search_result).unwrap_or(false);

    let mut converted = Vec::with_capacity(tools.len());
    for tool in tools {
        let grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)?;
        if let Some(grammar) = grammar {
            let mut item = json!({
                "type": "custom",
                "name": tool.name,
                "description": tool.description,
                "format": {
                    "type": "grammar",
                    "syntax": grammar.format,
                    "definition": grammar.definition,
                },
            });
            if tool_search_result {
                item["defer_loading"] = json!(true);
            }
            converted.push(item);
            continue;
        }

        let constrained_strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)?;
        let strict = constrained_strict.unwrap_or(default_strict);
        let mut function_tool = json!({
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": get_json_schema_tool_parameters(tool, strict).map_err(|error| error.to_string())?,
        });
        if tool_search_result {
            function_tool["defer_loading"] = json!(true);
        }
        if supports_strict_mode {
            function_tool["strict"] = json!(strict);
        }
        converted.push(function_tool);
    }
    Ok(converted)
}

// =============================================================================
// Stream processing
// =============================================================================

/// One SSE event of the Responses stream (`ResponseStreamEvent`). Only the
/// event shapes the adapter reacts to are modeled; unknown `type`s are parsed
/// to [`ResponseStreamEvent::Other`] and ignored (D14).
#[derive(Debug)]
pub enum ResponseStreamEvent {
    ResponseCreated {
        response: ResponseObject,
    },
    OutputItemAdded {
        output_index: u32,
        item: Value,
    },
    ReasoningSummaryTextDelta {
        output_index: u32,
        delta: String,
    },
    ReasoningSummaryPartDone {
        output_index: u32,
    },
    ReasoningTextDelta {
        output_index: u32,
        delta: String,
    },
    OutputTextDelta {
        output_index: u32,
        delta: String,
    },
    RefusalDelta {
        output_index: u32,
        delta: String,
    },
    FunctionCallArgumentsDelta {
        output_index: u32,
        delta: String,
    },
    FunctionCallArgumentsDone {
        output_index: u32,
        arguments: Option<String>,
    },
    CustomToolCallInputDelta {
        output_index: u32,
        delta: String,
    },
    CustomToolCallInputDone {
        output_index: u32,
        input: Option<String>,
    },
    OutputItemDone {
        output_index: u32,
        item: Value,
    },
    Completed {
        response: ResponseObject,
    },
    Incomplete {
        response: ResponseObject,
    },
    ResponseFailed {
        response: Option<ResponseObject>,
    },
    Error {
        code: Option<String>,
        message: Option<String>,
    },
    /// Any event type the adapter does not handle.
    Other,
}

/// The `response` object carried by terminal and created events.
#[derive(Debug, Default, Clone, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResponseObject {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub error: Option<ResponseErrorBody>,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    pub output: Option<Vec<Value>>,
    #[serde(default)]
    pub usage: Option<ResponseUsage>,
    #[serde(default)]
    pub service_tier: Option<Value>,
}

#[derive(Debug, Default, Clone, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResponseErrorBody {
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
}

#[derive(Debug, Default, Clone, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct IncompleteDetails {
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Default, Clone, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResponseUsage {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
    #[serde(default)]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(default)]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Default, Clone, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    #[serde(default)]
    pub cache_write_tokens: Option<u64>,
}

#[derive(Debug, Default, Clone, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}

fn output_index(raw: &Value) -> u32 {
    raw.get("output_index").and_then(Value::as_u64).unwrap_or(0) as u32
}

impl ResponseStreamEvent {
    /// Parse one SSE `data:` payload. Returns `None` for payloads that are not
    /// JSON objects (upstream would throw inside the SDK decoder and abort the
    /// stream; the caller turns that into a stream error, D14).
    pub fn parse(raw: &Value) -> Option<ResponseStreamEvent> {
        let event_type = raw.get("type").and_then(Value::as_str)?;
        let event = match event_type {
            "response.created" => ResponseStreamEvent::ResponseCreated {
                response: raw
                    .get("response")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .ok()
                    .flatten()?,
            },
            "response.output_item.added" => ResponseStreamEvent::OutputItemAdded {
                output_index: output_index(raw),
                item: raw.get("item").cloned().unwrap_or(Value::Null),
            },
            "response.reasoning_summary_text.delta" => {
                ResponseStreamEvent::ReasoningSummaryTextDelta {
                    output_index: output_index(raw),
                    delta: raw
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                }
            }
            "response.reasoning_summary_part.done" => {
                ResponseStreamEvent::ReasoningSummaryPartDone {
                    output_index: output_index(raw),
                }
            }
            "response.reasoning_text.delta" => ResponseStreamEvent::ReasoningTextDelta {
                output_index: output_index(raw),
                delta: raw
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
            "response.output_text.delta" => ResponseStreamEvent::OutputTextDelta {
                output_index: output_index(raw),
                delta: raw
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
            "response.refusal.delta" => ResponseStreamEvent::RefusalDelta {
                output_index: output_index(raw),
                delta: raw
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
            "response.function_call_arguments.delta" => {
                ResponseStreamEvent::FunctionCallArgumentsDelta {
                    output_index: output_index(raw),
                    delta: raw
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                }
            }
            "response.function_call_arguments.done" => {
                ResponseStreamEvent::FunctionCallArgumentsDone {
                    output_index: output_index(raw),
                    arguments: raw
                        .get("arguments")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                }
            }
            "response.custom_tool_call_input.delta" => {
                ResponseStreamEvent::CustomToolCallInputDelta {
                    output_index: output_index(raw),
                    delta: raw
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                }
            }
            "response.custom_tool_call_input.done" => {
                ResponseStreamEvent::CustomToolCallInputDone {
                    output_index: output_index(raw),
                    input: raw.get("input").and_then(Value::as_str).map(str::to_string),
                }
            }
            "response.output_item.done" => ResponseStreamEvent::OutputItemDone {
                output_index: output_index(raw),
                item: raw.get("item").cloned().unwrap_or(Value::Null),
            },
            "response.completed" => ResponseStreamEvent::Completed {
                response: raw
                    .get("response")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .ok()
                    .flatten()?,
            },
            "response.incomplete" => ResponseStreamEvent::Incomplete {
                response: raw
                    .get("response")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .ok()
                    .flatten()?,
            },
            "response.failed" => ResponseStreamEvent::ResponseFailed {
                response: raw
                    .get("response")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .ok()
                    .flatten(),
            },
            "error" => ResponseStreamEvent::Error {
                code: raw.get("code").map(|c| match c {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                }),
                message: raw.get("message").map(|m| match m {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                }),
            },
            _ => ResponseStreamEvent::Other,
        };
        Some(event)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum SlotKind {
    Thinking,
    Text,
    ToolCall,
}

/// Per-output-index streaming state (`ResponsesOutputSlot`). The streaming
/// scratch buffers (`partialJson`, grammar input buffer) live here rather than
/// on the content block; blocks are only finalized once the scratch is
/// dropped, matching upstream's delete-before-`toolcall_end` protocol.
#[derive(Clone)]
struct OutputSlot {
    kind: SlotKind,
    content_index: usize,
    partial_json: Option<String>,
    custom_input: Option<(String, GrammarToolInputJsonBuffer)>,
}

fn join_text_list(value: Option<&Value>, separator: &str) -> String {
    value
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .map(|part| part.get("text").and_then(Value::as_str).unwrap_or(""))
                .collect::<Vec<_>>()
                .join(separator)
        })
        .unwrap_or_default()
}

fn map_stop_reason(
    status: Option<&str>,
    incomplete_reason: Option<&str>,
) -> Result<(StopReason, Option<String>), String> {
    let Some(status) = status else {
        return Ok((StopReason::Stop, None));
    };
    match status {
        "completed" => Ok((StopReason::Stop, None)),
        "incomplete" => {
            if incomplete_reason == Some("max_output_tokens") {
                Ok((StopReason::Length, None))
            } else {
                Ok((
                    StopReason::Error,
                    Some(
                        incomplete_reason
                            .map(|reason| format!("Response incomplete: {reason}"))
                            .unwrap_or_else(|| {
                                "Response incomplete without a provider reason".to_string()
                            }),
                    ),
                ))
            }
        }
        "failed" | "cancelled" => Ok((StopReason::Error, None)),
        // These two are wonky ...
        "in_progress" | "queued" => Ok((StopReason::Stop, None)),
        other => Err(format!("Unhandled stop reason: {other}")),
    }
}

/// Consume the Responses event stream and drive the assistant message and
/// event protocol (`processResponsesStream`). Returns `Err` with the upstream
/// thrown message on failure (D15).
pub async fn process_responses_stream<S>(
    events: &mut S,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    model: &Model,
    options: Option<&OpenAIResponsesStreamOptions>,
) -> Result<(), String>
where
    S: futures_core::Stream<Item = Result<Value, crate::http::ProviderHttpError>> + Unpin,
{
    use futures_util::StreamExt;

    let default_grammar_properties: BTreeMap<String, String> = BTreeMap::new();
    let grammar_properties = options
        .map(|o| &o.grammar_tool_input_properties)
        .unwrap_or(&default_grammar_properties);
    let mut saw_terminal_response_event = false;
    let mut output_slots: HashMap<u32, OutputSlot> = HashMap::new();
    // reasoning item id -> content index of its thinking block
    let mut reasoning_blocks_by_id: HashMap<String, usize> = HashMap::new();

    let apply_message_phase_stop_reason = |output: &mut AssistantMessage, item: &Value| {
        if item.get("type").and_then(Value::as_str) == Some("message")
            && item.get("phase").and_then(Value::as_str) == Some("final_answer")
        {
            output.stop_reason = StopReason::Stop;
        }
    };

    fn finalize(
        output: &mut AssistantMessage,
        response: &ResponseObject,
        reasoning_blocks_by_id: &mut HashMap<String, usize>,
        model: &Model,
        options: Option<&OpenAIResponsesStreamOptions>,
    ) -> Result<bool, String> {
        let saw_terminal_response_event = true;
        // Azure OpenAI can omit reasoning.encrypted_content from
        // response.output_item.done and provide it only in
        // response.completed.response.output. Backfill the persisted reasoning
        // signature from the terminal response to keep store:false multi-turn
        // replay stateless (issue #6409).
        for item in response.output.as_deref().unwrap_or(&[]) {
            if item.get("type").and_then(Value::as_str) != Some("reasoning") {
                continue;
            }
            let Some(encrypted_content) = item.get("encrypted_content").and_then(Value::as_str)
            else {
                continue;
            };
            let Some(content_index) = item
                .get("id")
                .and_then(Value::as_str)
                .and_then(|id| reasoning_blocks_by_id.get(id))
            else {
                continue;
            };
            let Some(AssistantContent::Thinking(block)) = output.content.get(*content_index) else {
                continue;
            };
            let Some(signature) = &block.thinking_signature else {
                continue;
            };
            let Ok(mut stored) = serde_json::from_str::<Value>(signature) else {
                continue;
            };
            if stored
                .get("encrypted_content")
                .and_then(Value::as_str)
                .is_some()
            {
                continue;
            }
            stored["encrypted_content"] = json!(encrypted_content);
            if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(*content_index)
            {
                block.thinking_signature = Some(stored.to_string());
            }
        }
        if let Some(id) = &response.id {
            output.response_id = Some(id.clone());
        }
        if let Some(usage) = &response.usage {
            let input_details = usage.input_tokens_details.clone().unwrap_or_default();
            let cached_tokens = input_details.cached_tokens.unwrap_or(0);
            let cache_write_tokens = input_details.cache_write_tokens.unwrap_or(0);
            let input_tokens = usage.input_tokens.unwrap_or(0);
            // OpenAI includes cached and cache-write tokens in input_tokens, so subtract both.
            output.usage = Usage {
                input: input_tokens
                    .saturating_sub(cached_tokens)
                    .saturating_sub(cache_write_tokens),
                output: usage.output_tokens.unwrap_or(0),
                cache_read: cached_tokens,
                cache_write: cache_write_tokens,
                cache_write_1h: None,
                reasoning: Some(
                    usage
                        .output_tokens_details
                        .clone()
                        .unwrap_or_default()
                        .reasoning_tokens
                        .unwrap_or(0),
                ),
                total_tokens: usage.total_tokens.unwrap_or(0),
                cost: UsageCost::default(),
            };
        }
        calculate_cost(&AnyModel::Chat(model.clone()), &mut output.usage);
        if let Some(opts) = options {
            if let Some(apply_pricing) = &opts.apply_service_tier_pricing {
                let service_tier = if let Some(resolve) = &opts.resolve_service_tier {
                    resolve(response.service_tier.as_ref(), opts.service_tier.as_ref())
                } else {
                    response
                        .service_tier
                        .clone()
                        .or_else(|| opts.service_tier.clone())
                };
                apply_pricing(&mut output.usage, service_tier.as_ref());
            }
        }
        // Map status to stop reason. For incomplete responses, retain the
        // provider's specific reason so max-output truncation and content
        // filtering stay distinct.
        let status = response.status.as_deref();
        let incomplete_reason = response
            .incomplete_details
            .as_ref()
            .and_then(|details| details.reason.as_deref())
            .filter(|reason| !reason.is_empty());
        output.raw_stop_reason = Some(match incomplete_reason {
            Some(reason) => format!("{}.{}", status.unwrap_or("undefined"), reason),
            None => status.unwrap_or("undefined").to_string(),
        });
        let (stop_reason, error_message) = map_stop_reason(status, incomplete_reason)?;
        output.stop_reason = stop_reason;
        output.error_message = error_message;
        if output
            .content
            .iter()
            .any(|b| matches!(b, AssistantContent::ToolCall(_)))
            && output.stop_reason == StopReason::Stop
        {
            output.stop_reason = StopReason::ToolUse;
        }
        Ok(saw_terminal_response_event)
    }

    while let Some(item) = events.next().await {
        let raw = match item {
            Ok(raw) => raw,
            Err(error) => return Err(error.message),
        };
        if let Some(hook) = options.and_then(|o| o.on_provider_stream_event.as_ref()) {
            hook(raw.clone(), model.clone()).await;
        }
        let Some(event) = ResponseStreamEvent::parse(&raw) else {
            continue;
        };
        match event {
            ResponseStreamEvent::Other => {}
            ResponseStreamEvent::ResponseCreated { response } => {
                output.response_id = response.id;
            }
            ResponseStreamEvent::OutputItemAdded { output_index, item } => {
                create_slot(
                    output,
                    stream,
                    &mut output_slots,
                    grammar_properties,
                    output_index,
                    &item,
                );
            }
            ResponseStreamEvent::ReasoningSummaryTextDelta {
                output_index,
                delta,
            }
            | ResponseStreamEvent::ReasoningTextDelta {
                output_index,
                delta,
            } => {
                let Some(slot) = get_slot(&output_slots, output_index, &SlotKind::Thinking) else {
                    continue;
                };
                if let Some(AssistantContent::Thinking(block)) =
                    output.content.get_mut(slot.content_index)
                {
                    block.thinking += &delta;
                }
                stream.push(AssistantMessageEvent::ThinkingDelta {
                    content_index: slot.content_index as u32,
                    delta,
                    partial: output.clone(),
                });
            }
            ResponseStreamEvent::ReasoningSummaryPartDone { output_index } => {
                let Some(slot) = get_slot(&output_slots, output_index, &SlotKind::Thinking) else {
                    continue;
                };
                if let Some(AssistantContent::Thinking(block)) =
                    output.content.get_mut(slot.content_index)
                {
                    block.thinking += "\n\n";
                }
                stream.push(AssistantMessageEvent::ThinkingDelta {
                    content_index: slot.content_index as u32,
                    delta: "\n\n".to_string(),
                    partial: output.clone(),
                });
            }
            ResponseStreamEvent::OutputTextDelta {
                output_index,
                delta,
            }
            | ResponseStreamEvent::RefusalDelta {
                output_index,
                delta,
            } => {
                let Some(slot) = get_slot(&output_slots, output_index, &SlotKind::Text) else {
                    continue;
                };
                if let Some(AssistantContent::Text(block)) =
                    output.content.get_mut(slot.content_index)
                {
                    block.text += &delta;
                }
                stream.push(AssistantMessageEvent::TextDelta {
                    content_index: slot.content_index as u32,
                    delta,
                    partial: output.clone(),
                });
            }
            ResponseStreamEvent::FunctionCallArgumentsDelta {
                output_index,
                delta,
            } => {
                let Some(slot) = get_slot(&output_slots, output_index, &SlotKind::ToolCall) else {
                    continue;
                };
                let Some(partial_json) = &slot.partial_json else {
                    continue;
                };
                let args = parse_streaming_json(Some(partial_json));
                if let Some(AssistantContent::ToolCall(block)) =
                    output.content.get_mut(slot.content_index)
                {
                    block.arguments = match args {
                        Value::Object(map) => map,
                        _ => Map::new(),
                    };
                }
                stream.push(AssistantMessageEvent::ToolcallDelta {
                    content_index: slot.content_index as u32,
                    delta,
                    partial: output.clone(),
                });
            }
            ResponseStreamEvent::FunctionCallArgumentsDone {
                output_index,
                arguments,
            } => {
                let Some(mut slot) = get_slot(&output_slots, output_index, &SlotKind::ToolCall)
                else {
                    continue;
                };
                let Some(previous_partial_json) = slot.partial_json.clone() else {
                    continue;
                };
                let arguments = arguments.unwrap_or_default();
                slot.partial_json = Some(arguments.clone());
                output_slots.insert(output_index, slot.clone());
                let args = parse_streaming_json(Some(&arguments));
                if let Some(AssistantContent::ToolCall(block)) =
                    output.content.get_mut(slot.content_index)
                {
                    block.arguments = match args {
                        Value::Object(map) => map,
                        _ => Map::new(),
                    };
                }

                if arguments.starts_with(&previous_partial_json) {
                    let delta = &arguments[previous_partial_json.len()..];
                    if !delta.is_empty() {
                        stream.push(AssistantMessageEvent::ToolcallDelta {
                            content_index: slot.content_index as u32,
                            delta: delta.to_string(),
                            partial: output.clone(),
                        });
                    }
                }
            }
            ResponseStreamEvent::CustomToolCallInputDelta {
                output_index,
                delta,
            } => {
                let Some(slot) = get_slot(&output_slots, output_index, &SlotKind::ToolCall) else {
                    continue;
                };
                let Some(custom_input) = &slot.custom_input else {
                    continue;
                };
                let current = custom_input_value(output, slot.content_index, &custom_input.0);
                let next_input = format!("{current}{delta}");
                let delta_out = append_custom_tool_call_input(
                    output,
                    &mut output_slots,
                    output_index,
                    &next_input,
                    false,
                )?;
                if let Some(delta_out) = delta_out {
                    stream.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: slot.content_index as u32,
                        delta: delta_out,
                        partial: output.clone(),
                    });
                }
            }
            ResponseStreamEvent::CustomToolCallInputDone {
                output_index,
                input,
            } => {
                let Some(slot) = get_slot(&output_slots, output_index, &SlotKind::ToolCall) else {
                    continue;
                };
                if slot.custom_input.is_none() {
                    continue;
                }
                let next_input = match input {
                    Some(input) => input,
                    None => match &slot.custom_input {
                        Some((property, _)) => {
                            custom_input_value(output, slot.content_index, property)
                        }
                        None => continue,
                    },
                };
                let delta_out = append_custom_tool_call_input(
                    output,
                    &mut output_slots,
                    output_index,
                    &next_input,
                    true,
                )?;
                if let Some(delta_out) = delta_out {
                    stream.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: slot.content_index as u32,
                        delta: delta_out,
                        partial: output.clone(),
                    });
                }
            }
            ResponseStreamEvent::OutputItemDone { output_index, item } => {
                apply_message_phase_stop_reason(output, &item);
                let slot = match output_slots.get(&output_index) {
                    Some(slot) => slot.clone(),
                    None => create_slot(
                        output,
                        stream,
                        &mut output_slots,
                        grammar_properties,
                        output_index,
                        &item,
                    )
                    .unwrap_or_else(|| OutputSlot {
                        kind: SlotKind::Thinking,
                        content_index: usize::MAX,
                        partial_json: None,
                        custom_input: None,
                    }),
                };
                let item_type = item.get("type").and_then(Value::as_str);
                match (item_type, &slot.kind) {
                    (Some("reasoning"), SlotKind::Thinking) if slot.content_index != usize::MAX => {
                        let summary_text = join_text_list(item.get("summary"), "\n\n");
                        let content_text = join_text_list(item.get("content"), "\n\n");
                        let final_thinking = if !summary_text.is_empty() {
                            summary_text
                        } else if !content_text.is_empty() {
                            content_text
                        } else {
                            match output.content.get(slot.content_index) {
                                Some(AssistantContent::Thinking(block)) => block.thinking.clone(),
                                _ => String::new(),
                            }
                        };
                        if let Some(AssistantContent::Thinking(block)) =
                            output.content.get_mut(slot.content_index)
                        {
                            block.thinking = final_thinking.clone();
                            block.thinking_signature = Some(item.to_string());
                        }
                        if let Some(id) = item.get("id").and_then(Value::as_str) {
                            reasoning_blocks_by_id.insert(id.to_string(), slot.content_index);
                        }
                        stream.push(AssistantMessageEvent::ThinkingEnd {
                            content_index: slot.content_index as u32,
                            content: final_thinking,
                            partial: output.clone(),
                        });
                        output_slots.remove(&output_index);
                    }
                    (Some("message"), SlotKind::Text) if slot.content_index != usize::MAX => {
                        let text = item
                            .get("content")
                            .and_then(Value::as_array)
                            .map(|parts| {
                                parts
                                    .iter()
                                    .map(|part| {
                                        if part.get("type").and_then(Value::as_str)
                                            == Some("output_text")
                                        {
                                            part.get("text").and_then(Value::as_str).unwrap_or("")
                                        } else {
                                            part.get("refusal")
                                                .and_then(Value::as_str)
                                                .unwrap_or("")
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join("")
                            })
                            .unwrap_or_default();
                        let phase = item.get("phase").and_then(Value::as_str).and_then(|p| {
                            serde_json::from_str::<TextSignaturePhase>(&format!("\"{p}\"")).ok()
                        });
                        if let Some(AssistantContent::Text(block)) =
                            output.content.get_mut(slot.content_index)
                        {
                            block.text = text.clone();
                            block.text_signature = Some(encode_text_signature_v1(
                                item.get("id").and_then(Value::as_str).unwrap_or(""),
                                phase,
                            ));
                        }
                        stream.push(AssistantMessageEvent::TextEnd {
                            content_index: slot.content_index as u32,
                            content: text,
                            partial: output.clone(),
                        });
                        output_slots.remove(&output_index);
                    }
                    (Some("function_call"), SlotKind::ToolCall)
                        if slot.partial_json.is_some() && slot.content_index != usize::MAX =>
                    {
                        let args_source = item
                            .get("arguments")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .or_else(|| slot.partial_json.clone())
                            .unwrap_or_else(|| "{}".to_string());
                        let args = parse_streaming_json(Some(&args_source));
                        if let Some(AssistantContent::ToolCall(block)) =
                            output.content.get_mut(slot.content_index)
                        {
                            block.arguments = match args {
                                Value::Object(map) => map,
                                _ => Map::new(),
                            };
                            if let Some(namespace) = item.get("namespace").and_then(Value::as_str) {
                                block.namespace = Some(namespace.to_string());
                            }
                        }
                        let tool_call = match output.content.get(slot.content_index) {
                            Some(AssistantContent::ToolCall(block)) => block.clone(),
                            _ => continue,
                        };
                        // Finalize in place and strip the scratch buffer so replay
                        // only carries parsed arguments.
                        output_slots.remove(&output_index);
                        stream.push(AssistantMessageEvent::ToolcallEnd {
                            content_index: slot.content_index as u32,
                            tool_call,
                            partial: output.clone(),
                        });
                    }
                    (Some("custom_tool_call"), SlotKind::ToolCall)
                        if slot.custom_input.is_some() && slot.content_index != usize::MAX =>
                    {
                        let property = slot.custom_input.as_ref().expect("custom input").0.clone();
                        let current = custom_input_value(output, slot.content_index, &property);
                        let next_input = item
                            .get("input")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .unwrap_or(current);
                        let delta_out = append_custom_tool_call_input(
                            output,
                            &mut output_slots,
                            output_index,
                            &next_input,
                            true,
                        )?;
                        if let Some(delta_out) = delta_out {
                            stream.push(AssistantMessageEvent::ToolcallDelta {
                                content_index: slot.content_index as u32,
                                delta: delta_out,
                                partial: output.clone(),
                            });
                        }
                        if let Some(namespace) = item.get("namespace").and_then(Value::as_str) {
                            if let Some(AssistantContent::ToolCall(block)) =
                                output.content.get_mut(slot.content_index)
                            {
                                block.namespace = Some(namespace.to_string());
                            }
                        }
                        let tool_call = match output.content.get(slot.content_index) {
                            Some(AssistantContent::ToolCall(block)) => block.clone(),
                            _ => continue,
                        };
                        let content_index = slot.content_index;
                        output_slots.remove(&output_index);
                        stream.push(AssistantMessageEvent::ToolcallEnd {
                            content_index: content_index as u32,
                            tool_call,
                            partial: output.clone(),
                        });
                    }
                    _ => {}
                }
            }
            ResponseStreamEvent::Completed { response }
            | ResponseStreamEvent::Incomplete { response } => {
                saw_terminal_response_event = finalize(
                    output,
                    &response,
                    &mut reasoning_blocks_by_id,
                    model,
                    options,
                )?;
            }
            ResponseStreamEvent::ResponseFailed { response } => {
                // Upstream marks the terminal event seen, then throws below.
                let response = response.unwrap_or_default();
                output.raw_stop_reason = response.status.clone();
                let error = response.error.clone();
                let details = response.incomplete_details.clone();
                let msg = if let Some(error) = error {
                    format!(
                        "{}: {}",
                        error.code.unwrap_or_else(|| "unknown".to_string()),
                        error.message.unwrap_or_else(|| "no message".to_string())
                    )
                } else if let Some(reason) =
                    details.and_then(|d| d.reason).filter(|r| !r.is_empty())
                {
                    format!("incomplete: {reason}")
                } else {
                    "Unknown error (no error details in response)".to_string()
                };
                return Err(msg);
            }
            ResponseStreamEvent::Error { code, message } => {
                // Upstream: new Error(`Error Code ${event.code}: ${event.message}`) —
                // template-literal undefineds preserved for missing fields.
                return Err(format!(
                    "Error Code {}: {}",
                    code.unwrap_or_else(|| "undefined".to_string()),
                    message.unwrap_or_else(|| "undefined".to_string())
                ));
            }
        }
    }
    if !saw_terminal_response_event {
        return Err("OpenAI Responses stream ended before a terminal response event".to_string());
    }
    // The agent runs every tool call in the final message. Refuse to hand over
    // calls whose output_item.done never arrived: their arguments may be cut
    // off or mixed up, e.g. when a non-compliant server omits output_index.
    if output.stop_reason == StopReason::ToolUse {
        for slot in output_slots.values() {
            if slot.kind != SlotKind::ToolCall || slot.content_index == usize::MAX {
                continue;
            }
            if let Some(AssistantContent::ToolCall(block)) = output.content.get(slot.content_index)
            {
                return Err(format!(
                    "OpenAI Responses stream completed with an unfinished tool call: {} ({})",
                    block.name, block.id
                ));
            }
        }
    }
    Ok(())
}

fn get_slot(
    output_slots: &HashMap<u32, OutputSlot>,
    output_index: u32,
    kind: &SlotKind,
) -> Option<OutputSlot> {
    let slot = output_slots.get(&output_index)?;
    if kind_matches(&slot.kind, kind) {
        Some(slot.clone())
    } else {
        None
    }
}

fn kind_matches(actual: &SlotKind, expected: &SlotKind) -> bool {
    matches!(
        (actual, expected),
        (SlotKind::Thinking, SlotKind::Thinking)
            | (SlotKind::Text, SlotKind::Text)
            | (SlotKind::ToolCall, SlotKind::ToolCall)
    )
}

fn create_slot(
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    output_slots: &mut HashMap<u32, OutputSlot>,
    grammar_properties: &BTreeMap<String, String>,
    output_index: u32,
    item: &Value,
) -> Option<OutputSlot> {
    let item_type = item.get("type").and_then(Value::as_str)?;
    let slot = match item_type {
        "reasoning" => {
            let block = ThinkingContent::new("");
            output.content.push(AssistantContent::Thinking(block));
            let content_index = output.content.len() - 1;
            stream.push(AssistantMessageEvent::ThinkingStart {
                content_index: content_index as u32,
                partial: output.clone(),
            });
            OutputSlot {
                kind: SlotKind::Thinking,
                content_index,
                partial_json: None,
                custom_input: None,
            }
        }
        "message" => {
            if item.get("phase").and_then(Value::as_str) == Some("final_answer") {
                output.stop_reason = StopReason::Stop;
            }
            output
                .content
                .push(AssistantContent::Text(TextContent::new("")));
            let content_index = output.content.len() - 1;
            stream.push(AssistantMessageEvent::TextStart {
                content_index: content_index as u32,
                partial: output.clone(),
            });
            OutputSlot {
                kind: SlotKind::Text,
                content_index,
                partial_json: None,
                custom_input: None,
            }
        }
        "function_call" => {
            let call_id = item
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or("undefined");
            let item_id = item
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("undefined");
            let block = ToolCall {
                id: format!("{call_id}|{item_id}"),
                name: item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                arguments: Map::new(),
                thought_signature: None,
                namespace: item
                    .get("namespace")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            };
            output.content.push(AssistantContent::ToolCall(block));
            let content_index = output.content.len() - 1;
            stream.push(AssistantMessageEvent::ToolcallStart {
                content_index: content_index as u32,
                partial: output.clone(),
            });
            OutputSlot {
                kind: SlotKind::ToolCall,
                content_index,
                partial_json: Some(
                    item.get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                ),
                custom_input: None,
            }
        }
        "custom_tool_call" => {
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let input_property = grammar_properties
                .get(&name)
                .cloned()
                .unwrap_or_else(|| "input".to_string());
            let input = item
                .get("input")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let mut arguments = Map::new();
            arguments.insert(input_property.clone(), json!(input));
            output.content.push(AssistantContent::ToolCall(ToolCall {
                id: format!(
                    "{}|{}",
                    item.get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or("undefined"),
                    item.get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("undefined")
                ),
                name,
                arguments,
                thought_signature: None,
                namespace: item
                    .get("namespace")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            }));
            let content_index = output.content.len() - 1;
            stream.push(AssistantMessageEvent::ToolcallStart {
                content_index: content_index as u32,
                partial: output.clone(),
            });
            OutputSlot {
                kind: SlotKind::ToolCall,
                content_index,
                partial_json: None,
                custom_input: Some((
                    input_property,
                    GrammarToolInputJsonBuffer {
                        input: String::new(),
                        started: false,
                        closed: false,
                    },
                )),
            }
        }
        _ => return None,
    };
    output_slots.insert(output_index, slot.clone());
    Some(slot)
}

/// `getCustomToolCallInput`: the current string value of the custom input
/// property on the streamed tool call.
fn custom_input_value(output: &AssistantMessage, content_index: usize, property: &str) -> String {
    match output.content.get(content_index) {
        Some(AssistantContent::ToolCall(block)) => match block.arguments.get(property) {
            Some(Value::String(value)) => value.clone(),
            _ => String::new(),
        },
        _ => String::new(),
    }
}

/// `appendCustomToolCallInput`: feed the grammar buffer, replace the block's
/// arguments with `{ [property]: nextInput }`, return the wire delta.
fn append_custom_tool_call_input(
    output: &mut AssistantMessage,
    output_slots: &mut HashMap<u32, OutputSlot>,
    output_index: u32,
    next_input: &str,
    close: bool,
) -> Result<Option<String>, String> {
    let Some(slot) = output_slots.get_mut(&output_index) else {
        return Ok(None);
    };
    let Some((property, buffer)) = slot.custom_input.as_mut() else {
        return Ok(None);
    };
    let delta = append_grammar_tool_input_json_delta(buffer, property, next_input, close)?;
    if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(slot.content_index) {
        let mut arguments = Map::new();
        arguments.insert(property.clone(), json!(next_input));
        block.arguments = arguments;
    }
    Ok(delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_and_normalizes_foreign_item_ids() {
        // parseTextSignature round-trips
        assert_eq!(
            parse_text_signature(Some("{\"v\":1,\"id\":\"msg_1\"}")),
            Some(("msg_1".to_string(), None))
        );
        assert_eq!(
            parse_text_signature(Some("{\"v\":1,\"id\":\"m\",\"phase\":\"final_answer\"}")),
            Some(("m".to_string(), Some(TextSignaturePhase::FinalAnswer)))
        );
        assert_eq!(
            parse_text_signature(Some("legacy-id")),
            Some(("legacy-id".to_string(), None))
        );
        assert_eq!(parse_text_signature(None), None);
    }

    #[test]
    fn encodes_text_signature_v1() {
        assert_eq!(
            encode_text_signature_v1("m1", Some(TextSignaturePhase::Commentary)),
            "{\"v\":1,\"id\":\"m1\",\"phase\":\"commentary\"}"
        );
        assert_eq!(
            encode_text_signature_v1("m1", None),
            "{\"v\":1,\"id\":\"m1\"}"
        );
    }

    #[test]
    fn maps_stop_reasons() {
        assert_eq!(map_stop_reason(None, None), Ok((StopReason::Stop, None)));
        assert_eq!(
            map_stop_reason(Some("completed"), None),
            Ok((StopReason::Stop, None))
        );
        assert_eq!(
            map_stop_reason(Some("incomplete"), Some("max_output_tokens")),
            Ok((StopReason::Length, None))
        );
        assert_eq!(
            map_stop_reason(Some("incomplete"), Some("content_filter")),
            Ok((
                StopReason::Error,
                Some("Response incomplete: content_filter".to_string())
            ))
        );
        assert_eq!(
            map_stop_reason(Some("incomplete"), None),
            Ok((
                StopReason::Error,
                Some("Response incomplete without a provider reason".to_string())
            ))
        );
        assert_eq!(
            map_stop_reason(Some("failed"), None),
            Ok((StopReason::Error, None))
        );
        assert_eq!(
            map_stop_reason(Some("cancelled"), None),
            Ok((StopReason::Error, None))
        );
        assert_eq!(
            map_stop_reason(Some("in_progress"), None),
            Ok((StopReason::Stop, None))
        );
        assert_eq!(
            map_stop_reason(Some("queued"), None),
            Ok((StopReason::Stop, None))
        );
        assert!(map_stop_reason(Some("bogus"), None).is_err());
    }

    #[test]
    fn parses_stream_events() {
        let event = ResponseStreamEvent::parse(&serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 2,
            "delta": "hi"
        }));
        assert!(matches!(
            event,
            Some(ResponseStreamEvent::OutputTextDelta { output_index: 2, delta })
                if delta == "hi"
        ));
        assert!(matches!(
            ResponseStreamEvent::parse(&serde_json::json!({"type": "response.content_part.added"})),
            Some(ResponseStreamEvent::Other)
        ));
        assert!(ResponseStreamEvent::parse(&serde_json::json!("not-an-object-type")).is_none());
    }
}
