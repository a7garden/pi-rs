//! Port of `packages/ai/src/api/github-copilot-headers.ts`.
//!
//! Deviations:
//! - D25: upstream returns plain records; Rust returns ordered header pairs
//!   (upstream insertion order).
//!
//! Note: upstream consumes these helpers from `api/anthropic-messages.ts`,
//! `api/openai-completions.ts`, and `api/openai-responses.ts`; the landed
//! Rust adapters currently carry private copies (see their D18/D25 notes).

use crate::types::{Message, UserContent, UserContentField};

/// Copilot expects X-Initiator to indicate whether the request is
/// user-initiated or agent-initiated (e.g. follow-up after assistant/tool
/// messages) (`inferCopilotInitiator`).
pub fn infer_copilot_initiator(messages: &[Message]) -> &'static str {
    let last = messages.last();
    match last {
        Some(message) if message.role() != "user" => "agent",
        _ => "user",
    }
}

fn blocks_have_image(blocks: &[UserContent]) -> bool {
    blocks
        .iter()
        .any(|part| matches!(part, UserContent::Image(_)))
}

/// Copilot requires Copilot-Vision-Request header when sending images
/// (`hasCopilotVisionInput`).
pub fn has_copilot_vision_input(messages: &[Message]) -> bool {
    messages.iter().any(|message| match message {
        Message::User(user) => match &user.content {
            UserContentField::Text(_) => false,
            UserContentField::Blocks(blocks) => blocks_have_image(blocks),
        },
        Message::ToolResult(tool_result) => blocks_have_image(&tool_result.content),
        _ => false,
    })
}

/// Parameter object of [`build_copilot_dynamic_headers`]
/// (upstream's `params` argument).
pub struct CopilotDynamicHeadersParams<'a> {
    pub messages: &'a [Message],
    pub has_images: bool,
}

/// `buildCopilotDynamicHeaders`.
pub fn build_copilot_dynamic_headers(
    params: CopilotDynamicHeadersParams<'_>,
) -> Vec<(String, String)> {
    let mut headers = vec![
        (
            "X-Initiator".to_string(),
            infer_copilot_initiator(params.messages).to_string(),
        ),
        (
            "Openai-Intent".to_string(),
            "conversation-edits".to_string(),
        ),
    ];
    if params.has_images {
        headers.push(("Copilot-Vision-Request".to_string(), "true".to_string()));
    }
    headers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AssistantContent, AssistantMessage, ImageContent, StopReason, TextContent, Usage,
        UserContentField, UserMessage,
    };

    fn user_message(content: UserContentField) -> Message {
        Message::User(UserMessage {
            content,
            timestamp: 0,
        })
    }

    fn assistant_message(text: &str) -> Message {
        Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::Text(TextContent::new(text))],
            api: "test".to_string(),
            provider: "test".to_string(),
            model: "test".to_string(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            thinking_level: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
            duration_ms: None,
        })
    }

    fn image_blocks() -> UserContentField {
        UserContentField::Blocks(vec![UserContent::Image(ImageContent::new(
            "data:image/png;base64,AAA".to_string(),
            "image/png".to_string(),
        ))])
    }

    #[test]
    fn infers_initiator_from_last_message_role() {
        let messages = vec![
            user_message(UserContentField::Text("hi".into())),
            assistant_message("hello"),
        ];
        assert_eq!(infer_copilot_initiator(&messages), "agent");
        let messages = vec![
            assistant_message("hello"),
            user_message(UserContentField::Text("hi".into())),
        ];
        assert_eq!(infer_copilot_initiator(&messages), "user");
        let empty: Vec<Message> = Vec::new();
        assert_eq!(infer_copilot_initiator(&empty), "user");
    }

    #[test]
    fn detects_image_input_on_user_and_tool_result_messages() {
        let messages = vec![user_message(UserContentField::Text("hi".into()))];
        assert!(!has_copilot_vision_input(&messages));
        assert!(has_copilot_vision_input(&[user_message(image_blocks())]));
    }

    #[test]
    fn builds_dynamic_headers_with_vision_flag() {
        let messages = vec![user_message(UserContentField::Text("hi".into()))];
        let headers = build_copilot_dynamic_headers(CopilotDynamicHeadersParams {
            messages: &messages,
            has_images: true,
        });
        assert_eq!(
            headers,
            vec![
                ("X-Initiator".to_string(), "user".to_string()),
                (
                    "Openai-Intent".to_string(),
                    "conversation-edits".to_string()
                ),
                ("Copilot-Vision-Request".to_string(), "true".to_string()),
            ]
        );
    }
}
