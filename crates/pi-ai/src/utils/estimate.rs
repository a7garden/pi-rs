//! Port of `packages/ai/src/utils/estimate.ts`.
//!
//! Pure token-count estimation over transcript structures.
//!
//! Deviations (crate docs ledger):
//! - **D16 — the upstream `TranscriptContext | readonly Message[]` overload
//!   splits in two**: [`estimate_context_tokens`] takes the normalized
//!   [`TranscriptContext`], and [`estimate_context_tokens_for_messages`]
//!   accepts a raw message slice.
//! - **D16a — lengths are Unicode scalar values** (Rust `char`s) where
//!   upstream counts UTF-16 code units; identical for the BMP text these
//!   estimates target.

use serde::Serialize;

use crate::types::{AssistantContent, Message, StopReason, TranscriptContext, Usage, UserContent, UserContentField};
use crate::utils::error_body::safe_json_stringify;
use crate::utils::text::get_system_message_text;

/// Upstream `ContextUsageEstimate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextUsageEstimate {
	/// Estimated total context tokens.
	pub tokens: u64,
	/// Tokens reported by the most recent applicable assistant usage block.
	pub usage_tokens: u64,
	/// Estimated tokens after the most recent applicable assistant usage block.
	pub trailing_tokens: u64,
	/// Index of the applicable message that provided usage, or `None` when none exists.
	pub last_usage_index: Option<usize>,
}

const CHARS_PER_TOKEN: usize = 4;
const ESTIMATED_IMAGE_CHARS: usize = 4800;

/// Upstream `calculateContextTokens`.
pub fn calculate_context_tokens(usage: &Usage) -> u64 {
	if usage.total_tokens != 0 {
		usage.total_tokens
	} else {
		usage.input + usage.output + usage.cache_read + usage.cache_write
	}
}

/// `content.chars()` counted like upstream `content.length` (deviation D16a).
fn char_len(text: &str) -> usize {
	text.chars().count()
}

fn div_ceil_chars(chars: usize) -> u64 {
	chars.div_ceil(CHARS_PER_TOKEN) as u64
}

fn user_content_chars(content: &[UserContent]) -> usize {
	content
		.iter()
		.map(|block| match block {
			UserContent::Text(text) => char_len(&text.text),
			UserContent::Image(_) => ESTIMATED_IMAGE_CHARS,
		})
		.sum()
}

/// Upstream `estimateTextAndImageContentChars` over `string |
/// (TextContent | ImageContent)[]`.
fn text_and_image_content_chars(content: &UserContentField) -> usize {
	match content {
		UserContentField::Text(text) => char_len(text),
		UserContentField::Blocks(blocks) => user_content_chars(blocks),
	}
}

/// Upstream `estimateTextTokens`.
pub fn estimate_text_tokens(text: &str) -> u64 {
	div_ceil_chars(char_len(text))
}

/// Upstream `estimateTextAndImageContentTokens`.
pub fn estimate_text_and_image_content_tokens(content: &UserContentField) -> u64 {
	div_ceil_chars(text_and_image_content_chars(content))
}

/// Tokens for user/tool-result block lists (upstream overload over
/// `Array<TextContent | ImageContent>`).
pub fn estimate_user_content_tokens(content: &[UserContent]) -> u64 {
	div_ceil_chars(user_content_chars(content))
}

/// Upstream `estimateMessageTokens`.
pub fn estimate_message_tokens(message: &Message) -> u64 {
	match message {
		Message::System(system) => {
			estimate_text_tokens(&get_system_message_text(system))
				+ estimate_tools_tokens(system.tools_added.as_deref())
				+ estimate_tool_references_tokens(system.tools_removed.as_deref())
		}
		Message::User(user) => estimate_text_and_image_content_tokens(&user.content),
		Message::ToolResult(tool_result) => estimate_user_content_tokens(&tool_result.content),
		Message::Assistant(assistant) => {
			let mut chars = 0;
			for block in &assistant.content {
				match block {
					AssistantContent::Text(text) => chars += char_len(&text.text),
					AssistantContent::Thinking(thinking) => chars += char_len(&thinking.thinking),
					AssistantContent::ToolCall(tool_call) => {
						chars += char_len(&tool_call.name)
							+ char_len(&safe_json_stringify(&serde_json::Value::Object(
								tool_call.arguments.clone(),
							)));
					}
				}
			}
			div_ceil_chars(chars)
		}
	}
}

/// Upstream `getLastAssistantUsageInfo`.
fn get_last_assistant_usage_info(messages: &[Message]) -> Option<(&Usage, usize)> {
	let mut latest_prefix_timestamp = i64::MIN;
	let mut usage_info: Option<(&Usage, usize)> = None;

	for (index, message) in messages.iter().enumerate() {
		if let Message::Assistant(assistant) = message {
			// A newer prefix message was inserted after this response (for example, a
			// compaction summary), so its usage cannot describe the current prefix.
			let usage_applies_to_prefix = assistant.timestamp >= latest_prefix_timestamp;
			if usage_applies_to_prefix
				&& assistant.stop_reason != StopReason::Aborted
				&& assistant.stop_reason != StopReason::Error
				&& calculate_context_tokens(&assistant.usage) > 0
			{
				usage_info = Some((&assistant.usage, index));
			}
		}
		latest_prefix_timestamp = latest_prefix_timestamp.max(message_timestamp(message));
	}

	usage_info
}

fn message_timestamp(message: &Message) -> i64 {
	match message {
		Message::System(system) => system.timestamp,
		Message::User(user) => user.timestamp,
		Message::Assistant(assistant) => assistant.timestamp,
		Message::ToolResult(tool_result) => tool_result.timestamp,
	}
}

/// Upstream `estimateContextTokens`.
pub fn estimate_context_tokens(context: &TranscriptContext) -> ContextUsageEstimate {
	estimate_context_tokens_for_messages(context.messages())
}

/// Upstream `estimateContextTokens` over a raw message list (deviation D16).
pub fn estimate_context_tokens_for_messages(messages: &[Message]) -> ContextUsageEstimate {
	if let Some((usage, index)) = get_last_assistant_usage_info(messages) {
		let usage_tokens = calculate_context_tokens(usage);
		let mut trailing_tokens = 0;
		for message in &messages[index + 1..] {
			trailing_tokens += estimate_message_tokens(message);
		}
		return ContextUsageEstimate {
			tokens: usage_tokens + trailing_tokens,
			usage_tokens,
			trailing_tokens,
			last_usage_index: Some(index),
		};
	}

	let mut tokens = 0;
	for message in messages {
		tokens += estimate_message_tokens(message);
	}
	ContextUsageEstimate { tokens, usage_tokens: 0, trailing_tokens: tokens, last_usage_index: None }
}

/// Upstream `estimateToolsTokens` over tool declarations.
fn estimate_tools_tokens(tools: Option<&[crate::types::Tool]>) -> u64 {
	let Some(tools) = tools else { return 0 };
	if tools.is_empty() {
		return 0;
	}
	estimate_text_tokens(&serialize_json_text(tools))
}

/// Upstream `estimateToolsTokens` over tool references (`toolsRemoved`).
fn estimate_tool_references_tokens(references: Option<&[crate::types::ToolReference]>) -> u64 {
	let Some(references) = references else { return 0 };
	if references.is_empty() {
		return 0;
	}
	estimate_text_tokens(&serialize_json_text(references))
}

fn serialize_json_text<T: Serialize + ?Sized>(value: &T) -> String {
	match serde_json::to_value(value) {
		Ok(value) => safe_json_stringify(&value),
		Err(_) => "[unserializable]".to_string(),
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::{AssistantContent, AssistantMessage, Context, TextContent, UsageCost};
	use crate::utils::transcript::normalize_context;

	fn create_usage(total_tokens: u64) -> Usage {
		Usage {
			input: total_tokens,
			output: 0,
			cache_read: 0,
			cache_write: 0,
			cache_write_1h: None,
			reasoning: None,
			total_tokens,
			cost: UsageCost::default(),
		}
	}

	fn create_assistant(timestamp: i64, total_tokens: u64) -> AssistantMessage {
		AssistantMessage {
			content: vec![AssistantContent::Text(TextContent::new("kept"))],
			api: "openai-responses".to_string(),
			provider: "openai".to_string(),
			model: "test-model".to_string(),
			response_model: None,
			response_id: None,
			provider_thinking_level: None,
			thinking_level: None,
			diagnostics: None,
			usage: create_usage(total_tokens),
			stop_reason: StopReason::Stop,
			deferred: None,
			error_message: None,
			raw_stop_reason: None,
			end_turn: None,
			timestamp,
		}
	}

	fn user_message(content: &str, timestamp: i64) -> Message {
		Message::User(crate::types::UserMessage {
			content: UserContentField::Text(content.to_string()),
			timestamp,
		})
	}

	fn assistant_message(timestamp: i64, total_tokens: u64) -> Message {
		Message::Assistant(create_assistant(timestamp, total_tokens))
	}

	#[test]
	fn ignores_stale_assistant_usage_after_a_newer_message_is_inserted_before_it() {
		let context = normalize_context(&Context {
			system_prompt: Some("system".to_string()),
			messages: vec![
				user_message("summary", 200),
				assistant_message(100, 9_500),
				user_message(&"x".repeat(4_000), 300),
			],
			tools: None,
		});

		assert_eq!(
			estimate_context_tokens(&context),
			ContextUsageEstimate {
				tokens: 1_005,
				usage_tokens: 0,
				trailing_tokens: 1_005,
				last_usage_index: None,
			}
		);
		// The paired `buildBaseOptions(model, context).maxTokens === 4899` assertion
		// lives with the api/simple-options port (module owned by another slice).
	}

	#[test]
	fn uses_assistant_usage_again_after_a_response_to_the_inserted_context() {
		let context = normalize_context(&Context {
			system_prompt: None,
			messages: vec![
				user_message("summary", 200),
				assistant_message(100, 9_500),
				user_message("new prompt", 300),
				assistant_message(400, 2_000),
				user_message("tail", 500),
			],
			tools: None,
		});

		assert_eq!(
			estimate_context_tokens(&context),
			ContextUsageEstimate {
				tokens: 2_001,
				usage_tokens: 2_000,
				trailing_tokens: 1,
				last_usage_index: Some(3),
			}
		);
	}
}
