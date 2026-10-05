//! Port of `packages/ai/src/utils/text.ts`.

use crate::types::{AssistantContent, SystemBlock, SystemContent, SystemMessage, UserContent};

/// Extract and join text blocks from assistant content (`contentText` over
/// `AssistantMessage.content`).
pub fn content_blocks_text(content: &[AssistantContent], separator: &str) -> String {
	filter_join(content.iter().map(|block| match block {
		AssistantContent::Text(text) => Some(text.text.as_str()),
		_ => None,
	}), separator)
}

/// Extract and join text blocks from user/tool-result content.
pub fn user_content_text(content: &[UserContent], separator: &str) -> String {
	filter_join(content.iter().map(|block| match block {
		UserContent::Text(text) => Some(text.text.as_str()),
		_ => None,
	}), separator)
}

/// Extract text from system content (`contentText` over `string |
/// TextContent[]`).
pub fn system_content_text(content: &SystemContent) -> String {
	match content {
		SystemContent::Text(text) => text.clone(),
		SystemContent::Blocks(blocks) => filter_join(
			blocks.iter().map(|b| match b {
				SystemBlock::Text(text) => Some(text.text.as_str()),
			}),
			"\n",
		),
	}
}

/// Render a system message as a complete prompt: its content followed by its
/// sections (`getSystemMessageText`).
pub fn get_system_message_text(message: &SystemMessage) -> String {
	let mut parts = vec![system_content_text(&message.content)];
	if let Some(sections) = &message.sections {
		for text in sections.values() {
			if let Some(text) = text {
				parts.push(text.clone());
			}
		}
	}
	parts.into_iter().filter(|part| !part.is_empty()).collect::<Vec<_>>().join("\n\n")
}

/// Render a later system message for APIs that accept system messages
/// mid-conversation (`renderSystemMessageUpdate`). Section changes are framed
/// by name so the model can relate them to the leading prompt. This framing is
/// request-time only and may change between versions.
pub fn render_system_message_update(message: &SystemMessage) -> String {
	let mut parts: Vec<String> = Vec::new();
	let text = system_content_text(&message.content);
	if !text.is_empty() {
		parts.push(text);
	}
	if let Some(sections) = &message.sections {
		for (name, value) in sections {
			parts.push(match value {
				None => format!("Removed system prompt section \"{name}\"."),
				Some(value) => format!("Updated system prompt section \"{name}\":\n\n{value}"),
			});
		}
	}
	parts.join("\n\n")
}

fn filter_join<'a>(items: impl Iterator<Item = Option<&'a str>>, separator: &str) -> String {
	items.flatten().collect::<Vec<_>>().join(separator)
}

#[cfg(test)]
mod tests {
	// Port of test/text.test.ts (contentText / separators / tool-result text).
	use super::*;
	use crate::types::{AssistantContent, ImageContent, TextContent, UserContent};

	fn sample_content() -> Vec<AssistantContent> {
		vec![
			AssistantContent::Thinking(crate::types::ThinkingContent::new("reasoning")),
			AssistantContent::Text(TextContent::new("first")),
			AssistantContent::ToolCall(crate::types::ToolCall {
				id: "1".to_string(),
				name: "read".to_string(),
				arguments: Default::default(),
				thought_signature: None,
				namespace: None,
			}),
			AssistantContent::Text(TextContent::new("second")),
		]
	}

	#[test]
	fn extracts_assistant_text_blocks() {
		assert_eq!(content_blocks_text(&sample_content(), "\n"), "first\nsecond");
	}

	#[test]
	fn supports_custom_separators() {
		assert_eq!(content_blocks_text(&sample_content(), ""), "firstsecond");
	}

	#[test]
	fn passes_string_content_through() {
		assert_eq!(system_content_text(&SystemContent::Text("hello".to_string())), "hello");
	}

	#[test]
	fn extracts_text_from_tool_result_content() {
		let tool_result_content: Vec<UserContent> = vec![
			UserContent::Text(TextContent::new("first")),
			UserContent::Image(ImageContent::new("...", "image/png")),
			UserContent::Text(TextContent::new("second")),
		];

		assert_eq!(user_content_text(&tool_result_content, ""), "firstsecond");
	}
}
