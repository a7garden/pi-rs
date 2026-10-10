//! Port of `packages/ai/src/api/transform-messages.ts`.
//!
//! Normalizes a transcript for a target model: downgrades images for
//! non-vision models, drops/converts thinking blocks across models, strips
//! foreign `thoughtSignature`s, normalizes tool call IDs, and synthesizes
//! empty tool results for orphaned tool calls.
//!
//! Deviations:
//! - D19 (see D13): upstream null/undefined `content` normalization for
//!   untyped callers happens at deserialization (foundation deviation D13);
//!   the map over `msg.content == null` is an identity here and is omitted.
//! - D19a: `Date.now()` timestamps of synthetic tool results use
//!   [`crate::utils::diagnostics::now_ms`].

use std::collections::{HashMap, HashSet};

use crate::types::{
    AssistantContent, AssistantMessage, Message, Model, ModelInput, StopReason, TextContent,
    ToolCall, ToolResultMessage, UserContent, UserContentField, UserMessage,
};
use crate::utils::diagnostics::now_ms;

const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str =
    "(tool image omitted: model does not support images)";

fn replace_images_with_placeholder(content: &[UserContent], placeholder: &str) -> Vec<UserContent> {
    let mut result: Vec<UserContent> = Vec::new();
    let mut previous_was_placeholder = false;

    for block in content {
        match block {
            UserContent::Image(_) => {
                if !previous_was_placeholder {
                    result.push(UserContent::Text(TextContent::new(placeholder)));
                }
                previous_was_placeholder = true;
            }
            UserContent::Text(text) => {
                previous_was_placeholder = text.text == placeholder;
                result.push(block.clone());
            }
        }
    }

    result
}

fn downgrade_unsupported_images(messages: &[Message], model: &Model) -> Vec<Message> {
    if model
        .base
        .input
        .iter()
        .any(|input| *input == ModelInput::Image)
    {
        return messages.to_vec();
    }

    messages
        .iter()
        .map(|msg| match msg {
            Message::User(user) => match &user.content {
                UserContentField::Blocks(blocks) => Message::User(UserMessage {
                    content: UserContentField::Blocks(replace_images_with_placeholder(
                        blocks,
                        NON_VISION_USER_IMAGE_PLACEHOLDER,
                    )),
                    ..user.clone()
                }),
                UserContentField::Text(_) => msg.clone(),
            },
            Message::ToolResult(tool_result) => Message::ToolResult(ToolResultMessage {
                content: replace_images_with_placeholder(
                    &tool_result.content,
                    NON_VISION_TOOL_IMAGE_PLACEHOLDER,
                ),
                ..tool_result.clone()
            }),
            _ => msg.clone(),
        })
        .collect()
}

/// Normalize tool call ID for cross-provider compatibility.
/// OpenAI Responses API generates IDs that are 450+ chars with special
/// characters like `|`. Anthropic APIs require IDs matching `^[a-zA-Z0-9_-]+$`
/// (max 64 chars).
///
/// `normalize_tool_call_id` mirrors the upstream optional callback
/// `(id, model, source) => string`; it only runs for assistant messages from a
/// different model, and remapped tool result IDs follow the new call IDs.
pub fn transform_messages(
    messages: &[Message],
    model: &Model,
    normalize_tool_call_id: Option<&dyn Fn(&str, &Model, &AssistantMessage) -> String>,
) -> Vec<Message> {
    // Build a map of original tool call IDs to normalized IDs
    let mut tool_call_id_map: HashMap<String, String> = HashMap::new();
    let image_aware_messages = downgrade_unsupported_images(messages, model);

    // First pass: transform messages (unsupported image downgrade, thinking
    // blocks, tool call ID normalization)
    let transformed: Vec<Message> = image_aware_messages
        .iter()
        .map(|msg| match msg {
            // System and user messages pass through unchanged
            Message::System(_) | Message::User(_) => msg.clone(),
            // Handle toolResult messages - normalize toolCallId if we have a mapping
            Message::ToolResult(tool_result) => {
                match tool_call_id_map.get(&tool_result.tool_call_id) {
                    Some(normalized_id) if normalized_id != &tool_result.tool_call_id => {
                        Message::ToolResult(ToolResultMessage {
                            tool_call_id: normalized_id.clone(),
                            ..tool_result.clone()
                        })
                    }
                    _ => msg.clone(),
                }
            }
            // Assistant messages need transformation check
            Message::Assistant(assistant) => {
                let is_same_model = assistant.provider == model.provider()
                    && assistant.api == model.api()
                    && assistant.model == model.id();

                let content: Vec<AssistantContent> = assistant
                    .content
                    .iter()
                    .filter_map(|block| {
                        match block {
                            AssistantContent::Thinking(thinking) => {
                                // Redacted thinking is opaque encrypted content, only valid for
                                // the same model. Drop it for cross-model to avoid API errors.
                                if thinking.redacted.unwrap_or(false) {
                                    return is_same_model.then(|| block.clone());
                                }
                                // For same model: keep thinking blocks with signatures (needed
                                // for replay) even if the thinking text is empty (OpenAI
                                // encrypted reasoning)
                                if is_same_model && thinking.thinking_signature.is_some() {
                                    return Some(block.clone());
                                }
                                // Skip empty thinking blocks, convert others to plain text
                                if thinking.thinking.is_empty()
                                    || thinking.thinking.trim().is_empty()
                                {
                                    return None;
                                }
                                if is_same_model {
                                    Some(block.clone())
                                } else {
                                    Some(AssistantContent::Text(TextContent::new(
                                        &thinking.thinking,
                                    )))
                                }
                            }
                            AssistantContent::Text(text) => {
                                if is_same_model {
                                    Some(block.clone())
                                } else {
                                    Some(AssistantContent::Text(TextContent::new(&text.text)))
                                }
                            }
                            AssistantContent::ToolCall(tool_call) => {
                                let mut normalized_tool_call: ToolCall = tool_call.clone();

                                if !is_same_model
                                    && normalized_tool_call.thought_signature.is_some()
                                {
                                    normalized_tool_call.thought_signature = None;
                                }

                                if !is_same_model {
                                    if let Some(normalize) = normalize_tool_call_id {
                                        let normalized_id =
                                            normalize(&tool_call.id, model, assistant);
                                        if normalized_id != tool_call.id {
                                            tool_call_id_map.insert(
                                                tool_call.id.clone(),
                                                normalized_id.clone(),
                                            );
                                            normalized_tool_call.id = normalized_id;
                                        }
                                    }
                                }

                                Some(AssistantContent::ToolCall(normalized_tool_call))
                            }
                        }
                    })
                    .collect();

                Message::Assistant(AssistantMessage {
                    content,
                    ..assistant.clone()
                })
            }
        })
        .collect();

    // Second pass: insert synthetic empty tool results for orphaned tool calls
    // This preserves thinking signatures and satisfies API requirements
    let mut result: Vec<Message> = Vec::new();
    let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
    let mut existing_tool_result_ids: HashSet<String> = HashSet::new();
    // System messages are transparent to tool-call accounting: one that lands
    // between a tool call and its results is held back and emitted after the
    // results (synthetic ones included), so it never causes a duplicate result
    // for a call that is answered later.
    let mut held_system_messages: Vec<Message> = Vec::new();

    fn close_pending_tool_calls(
        result: &mut Vec<Message>,
        pending_tool_calls: &mut Vec<ToolCall>,
        existing_tool_result_ids: &mut HashSet<String>,
        held_system_messages: &mut Vec<Message>,
    ) {
        if !pending_tool_calls.is_empty() {
            for tool_call in pending_tool_calls.iter() {
                if !existing_tool_result_ids.contains(&tool_call.id) {
                    result.push(Message::ToolResult(ToolResultMessage {
                        tool_call_id: tool_call.id.clone(),
                        tool_name: tool_call.name.clone(),
                        content: vec![UserContent::Text(TextContent::new("No result provided"))],
                        details: None,
                        usage: None,
                        nested_calls: None,
                        is_error: true,
                        timestamp: now_ms(),
                        duration_ms: None,
                    }));
                }
            }
            pending_tool_calls.clear();
            existing_tool_result_ids.clear();
        }
        result.append(held_system_messages);
    }

    for msg in &transformed {
        match msg {
            Message::Assistant(assistant) => {
                // If we have pending orphaned tool calls from a previous assistant,
                // insert synthetic results now
                close_pending_tool_calls(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_result_ids,
                    &mut held_system_messages,
                );

                // Skip errored/aborted assistant messages entirely.
                // These are incomplete turns that shouldn't be replayed:
                // - May have partial content (reasoning without message, incomplete tool calls)
                // - Replaying them can cause API errors (e.g., OpenAI "reasoning without following item")
                // - The model should retry from the last valid state
                if matches!(
                    assistant.stop_reason,
                    StopReason::Error | StopReason::Aborted
                ) {
                    continue;
                }

                // Track tool calls from this assistant message
                let tool_calls: Vec<ToolCall> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
                        _ => None,
                    })
                    .collect();
                if !tool_calls.is_empty() {
                    pending_tool_calls = tool_calls;
                    existing_tool_result_ids = HashSet::new();
                }

                result.push(msg.clone());
            }
            Message::ToolResult(tool_result) => {
                existing_tool_result_ids.insert(tool_result.tool_call_id.clone());
                result.push(msg.clone());
            }
            Message::System(_) => {
                if !pending_tool_calls.is_empty() {
                    held_system_messages.push(msg.clone());
                } else {
                    result.push(msg.clone());
                }
            }
            Message::User(_) => {
                // A new user turn interrupts tool flow - insert synthetic results for
                // orphaned calls
                close_pending_tool_calls(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_result_ids,
                    &mut held_system_messages,
                );
                result.push(msg.clone());
            }
        }
    }

    // If the conversation ends with unresolved tool calls, synthesize results now.
    close_pending_tool_calls(
        &mut result,
        &mut pending_tool_calls,
        &mut existing_tool_result_ids,
        &mut held_system_messages,
    );

    result
}

#[cfg(test)]
mod tests {
    // Port of test/transform-messages-copilot-openai-to-anthropic.test.ts.
    use super::*;
    use crate::types::{
        BaseModelFields, ModelCost, ModelCostRates, ThinkingContent, Usage, UsageCost,
    };

    // Normalize function matching what anthropic.ts uses
    fn anthropic_normalize_tool_call_id(
        id: &str,
        _model: &Model,
        _source: &AssistantMessage,
    ) -> String {
        let replaced: String = id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        replaced.chars().take(64).collect()
    }

    fn make_copilot_claude_model() -> Model {
        Model {
            base: BaseModelFields {
                id: "claude-sonnet-4.6".to_string(),
                name: "Claude Sonnet 4.6".to_string(),
                api: "anthropic-messages".to_string(),
                provider: "github-copilot".to_string(),
                base_url: "https://api.individual.githubcopilot.com".to_string(),
                input: vec![ModelInput::Text, ModelInput::Image],
                input_limits: None,
                cost: ModelCost {
                    rates: ModelCostRates {
                        input: 0.0,
                        output: 0.0,
                        cache_read: 0.0,
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
            context_window: 128_000,
            max_tokens: 16_000,
            sampling_params: None,
            sampling_params_by_thinking_level: None,
            compat: None,
        }
    }

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

    fn make_assistant_message(content: Vec<AssistantContent>) -> AssistantMessage {
        AssistantMessage {
            content,
            api: "openai-responses".to_string(),
            provider: "github-copilot".to_string(),
            model: "gpt-5".to_string(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            thinking_level: None,
            diagnostics: None,
            usage: zero_usage(),
            stop_reason: StopReason::ToolUse,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: now_ms(),
            duration_ms: None,
        }
    }

    fn user_message(text: &str) -> Message {
        Message::User(UserMessage {
            content: UserContentField::Text(text.to_string()),
            timestamp: now_ms(),
        })
    }

    fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> AssistantContent {
        AssistantContent::ToolCall(ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: arguments.as_object().expect("object arguments").clone(),
            thought_signature: None,
            namespace: None,
        })
    }

    fn tool_result(tool_call_id: &str, tool_name: &str, text: &str) -> Message {
        Message::ToolResult(ToolResultMessage {
            tool_call_id: tool_call_id.to_string(),
            tool_name: tool_name.to_string(),
            content: vec![UserContent::Text(TextContent::new(text))],
            details: None,
            usage: None,
            nested_calls: None,
            is_error: false,
            timestamp: now_ms(),
            duration_ms: None,
        })
    }

    #[test]
    fn converts_thinking_blocks_to_plain_text_when_source_model_differs() {
        let model = make_copilot_claude_model();
        let messages = vec![
            user_message("hello"),
            Message::Assistant(AssistantMessage {
                content: vec![
                    AssistantContent::Thinking(ThinkingContent {
                        thinking: "Let me think about this...".to_string(),
                        thinking_signature: Some("reasoning_content".to_string()),
                        redacted: None,
                    }),
                    AssistantContent::Text(TextContent::new("Hi there!")),
                ],
                api: "openai-completions".to_string(),
                provider: "github-copilot".to_string(),
                model: "gpt-4o".to_string(),
                response_model: None,
                response_id: None,
                provider_thinking_level: None,
                thinking_level: None,
                diagnostics: None,
                usage: zero_usage(),
                stop_reason: StopReason::Stop,
                deferred: None,
                error_message: None,
                raw_stop_reason: None,
                end_turn: None,
                timestamp: now_ms(),
                duration_ms: None,
            }),
        ];

        let result = transform_messages(&messages, &model, Some(&anthropic_normalize_tool_call_id));
        let assistant_msg = result
            .iter()
            .filter_map(|m| match m {
                Message::Assistant(assistant) => Some(assistant),
                _ => None,
            })
            .next()
            .expect("assistant message present");

        // Thinking block should be converted to text since models differ
        let text_blocks = assistant_msg
            .content
            .iter()
            .filter(|b| matches!(b, AssistantContent::Text(_)))
            .count();
        let thinking_blocks = assistant_msg
            .content
            .iter()
            .filter(|b| matches!(b, AssistantContent::Thinking(_)))
            .count();
        assert_eq!(thinking_blocks, 0);
        assert!(text_blocks >= 2);
    }

    #[test]
    fn removes_thought_signature_from_tool_calls_when_migrating_between_models() {
        let model = make_copilot_claude_model();
        let tool_call_block = ToolCall {
			id: "call_123".to_string(),
			name: "bash".to_string(),
			arguments: serde_json::json!({ "command": "ls" }).as_object().expect("object").clone(),
			thought_signature: Some(
				serde_json::json!({ "type": "reasoning.encrypted", "id": "call_123", "data": "encrypted" }).to_string(),
			),
			namespace: None,
		};
        let messages = vec![
            user_message("run a command"),
            Message::Assistant(make_assistant_message(vec![AssistantContent::ToolCall(
                tool_call_block,
            )])),
            tool_result("call_123", "bash", "output"),
        ];

        let result = transform_messages(&messages, &model, Some(&anthropic_normalize_tool_call_id));
        let assistant_msg = result
            .iter()
            .filter_map(|m| match m {
                Message::Assistant(assistant) => Some(assistant),
                _ => None,
            })
            .next()
            .expect("assistant message present");
        let call = assistant_msg
            .content
            .iter()
            .filter_map(|b| match b {
                AssistantContent::ToolCall(call) => Some(call),
                _ => None,
            })
            .next()
            .expect("tool call present");

        assert_eq!(call.thought_signature, None);
    }

    #[test]
    fn adds_synthetic_tool_results_for_trailing_orphaned_tool_calls() {
        let model = make_copilot_claude_model();
        let messages = vec![
            user_message("read the file"),
            Message::Assistant(make_assistant_message(vec![tool_call(
                "call_123|fc_123",
                "read",
                serde_json::json!({ "path": "README.md" }),
            )])),
        ];

        let result = transform_messages(&messages, &model, Some(&anthropic_normalize_tool_call_id));
        let last = result.last().expect("non-empty result");

        let Message::ToolResult(synthetic) = last else {
            panic!("expected trailing tool result, got {:?}", last.role());
        };
        assert_eq!(synthetic.tool_call_id, "call_123_fc_123");
        assert_eq!(synthetic.tool_name, "read");
        assert!(synthetic.is_error);
        assert_eq!(
            synthetic.content,
            vec![UserContent::Text(TextContent::new("No result provided"))]
        );
    }

    #[test]
    fn adds_synthetic_results_only_for_trailing_tool_calls_that_are_still_missing_results() {
        let model = make_copilot_claude_model();
        let messages = vec![
            user_message("run commands"),
            Message::Assistant(make_assistant_message(vec![
                tool_call(
                    "call_1|fc_1",
                    "read",
                    serde_json::json!({ "path": "README.md" }),
                ),
                tool_call(
                    "call_2|fc_2",
                    "bash",
                    serde_json::json!({ "command": "pwd" }),
                ),
            ])),
            tool_result("call_1|fc_1", "read", "done"),
        ];

        let result = transform_messages(&messages, &model, Some(&anthropic_normalize_tool_call_id));
        let synthetic_results: Vec<&ToolResultMessage> = result
            .iter()
            .filter_map(|message| match message {
                Message::ToolResult(tool_result) if tool_result.is_error => Some(tool_result),
                _ => None,
            })
            .collect();

        assert_eq!(synthetic_results.len(), 1);
        assert_eq!(synthetic_results[0].tool_call_id, "call_2_fc_2");
        assert_eq!(synthetic_results[0].tool_name, "bash");
        assert_eq!(
            synthetic_results[0].content,
            vec![UserContent::Text(TextContent::new("No result provided"))]
        );
    }
}
