//! Port of `packages/coding-agent/src/core/compaction/utils.ts` — shared
//! utilities for compaction and branch summarization: file-operation
//! tracking, conversation serialization for summarization prompts, and the
//! summarization system prompt.

use std::collections::BTreeSet;

use pi_agent::AgentMessage;
use pi_ai::types::{AssistantContent, Message, UserContentField};
use pi_ai::utils::text::{content_blocks_text, user_content_text};
use serde_json::Value;

// ============================================================================
// File Operation Tracking
// ============================================================================

/// File paths touched by read/write/edit tool calls (`FileOperations`). JS
/// `Set`s; the ported `BTreeSet` yields the sorted order
/// [`compute_file_lists`] produces upstream.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FileOperations {
	pub read: BTreeSet<String>,
	pub written: BTreeSet<String>,
	pub edited: BTreeSet<String>,
}

pub fn create_file_ops() -> FileOperations {
	FileOperations::default()
}

/// Extract file operations from tool calls in an assistant message, or from
/// the nested calls recorded on a tool result (`extractFileOpsFromMessage`).
pub fn extract_file_ops_from_message(message: &AgentMessage, file_ops: &mut FileOperations) {
	match message {
		// Calls made from codemode scripts are recorded on the script's result.
		AgentMessage::Llm(Message::ToolResult(result)) => {
			for call in result.nested_calls.iter().flat_map(|nested| nested.calls.iter()) {
				add_file_op(&call.name, call.arguments.as_ref(), file_ops);
			}
		}
		AgentMessage::Llm(Message::Assistant(assistant)) => {
			for block in &assistant.content {
				if let AssistantContent::ToolCall(call) = block {
					add_file_op(&call.name, Some(&call.arguments), file_ops);
				}
			}
		}
		// Upstream duck-types custom payloads; mirror the defensive checks for
		// opaque messages (crate docs D4).
		AgentMessage::Custom(value) => {
			let role = value.get("role").and_then(|role| role.as_str()).unwrap_or("custom");
			match role {
				"toolResult" => {
					for call in value.pointer("/nestedCalls/calls").and_then(Value::as_array).into_iter().flatten() {
						if let Some(name) = call.get("name").and_then(Value::as_str) {
							let arguments = call.get("arguments").and_then(Value::as_object);
							add_file_op(name, arguments, file_ops);
						}
					}
				}
				"assistant" => {
					for block in value.get("content").and_then(Value::as_array).into_iter().flatten() {
						if block.get("type").and_then(Value::as_str) != Some("toolCall") {
							continue;
						}
						if let Some(name) = block.get("name").and_then(Value::as_str) {
							let arguments = block.get("arguments").and_then(Value::as_object);
							add_file_op(name, arguments, file_ops);
						}
					}
				}
				_ => {}
			}
		}
		_ => {}
	}
}

fn add_file_op(tool_name: &str, args: Option<&serde_json::Map<String, Value>>, file_ops: &mut FileOperations) {
	let path = args.and_then(|args| args.get("path")).and_then(Value::as_str);
	// Upstream `if (!path) return;` also drops the empty string.
	let Some(path) = path.filter(|path| !path.is_empty()) else { return };
	match tool_name {
		"read" => {
			file_ops.read.insert(path.to_string());
		}
		"write" => {
			file_ops.written.insert(path.to_string());
		}
		"edit" => {
			file_ops.edited.insert(path.to_string());
		}
		_ => {}
	}
}

/// Computed file lists ([`compute_file_lists`] return shape).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FileLists {
	/// Files only read, not modified.
	pub read_files: Vec<String>,
	pub modified_files: Vec<String>,
}

/// Compute final file lists from file operations (`computeFileLists`).
pub fn compute_file_lists(file_ops: &FileOperations) -> FileLists {
	let modified: BTreeSet<&String> = file_ops.edited.iter().chain(file_ops.written.iter()).collect();
	let read_files = file_ops
		.read
		.iter()
		.filter(|file| !modified.contains(*file))
		.cloned()
		.collect();
	FileLists { read_files, modified_files: modified.into_iter().cloned().collect() }
}

/// Format file operations as XML tags for summary (`formatFileOperations`).
pub fn format_file_operations(read_files: &[String], modified_files: &[String]) -> String {
	let mut sections: Vec<String> = Vec::new();
	if !read_files.is_empty() {
		sections.push(format!("<read-files>\n{}\n</read-files>", read_files.join("\n")));
	}
	if !modified_files.is_empty() {
		sections.push(format!("<modified-files>\n{}\n</modified-files>", modified_files.join("\n")));
	}
	if sections.is_empty() {
		return String::new();
	}
	format!("\n\n{}", sections.join("\n\n"))
}

// ============================================================================
// Message Serialization
// ============================================================================

/// Maximum characters for a tool result in serialized summaries.
const TOOL_RESULT_MAX_CHARS: usize = 2000;

/// Truncate text to a maximum character length for summarization. Keeps the
/// beginning and appends a truncation marker (`truncateForSummary`).
fn truncate_for_summary(text: &str, max_chars: usize) -> String {
	if text.chars().count() <= max_chars {
		return text.to_string();
	}
	let truncated_chars = text.chars().count() - max_chars;
	let truncated: String = text.chars().take(max_chars).collect();
	format!("{truncated}\n\n[... {truncated_chars} more characters truncated]")
}

fn user_content_field_text(content: &UserContentField, separator: &str) -> String {
	match content {
		UserContentField::Text(text) => text.clone(),
		UserContentField::Blocks(blocks) => user_content_text(blocks, separator),
	}
}

/// Serialize LLM messages to text for summarization
/// (`serializeConversation`). This prevents the model from treating it as a
/// conversation to continue. Call [`crate::core::messages::convert_to_llm`]
/// first to handle custom message types.
///
/// Tool results are truncated to keep the summarization request within
/// reasonable token budgets. Full content is not needed for summarization.
pub fn serialize_conversation(messages: &[Message]) -> String {
	let mut parts: Vec<String> = Vec::new();

	for message in messages {
		match message {
			Message::User(user) => {
				let content = user_content_field_text(&user.content, "");
				if !content.is_empty() {
					parts.push(format!("[User]: {content}"));
				}
			}
			Message::Assistant(assistant) => {
				let mut thinking_parts: Vec<&str> = Vec::new();
				let mut tool_calls: Vec<String> = Vec::new();

				for block in &assistant.content {
					match block {
						AssistantContent::Thinking(thinking) => thinking_parts.push(&thinking.thinking),
						AssistantContent::ToolCall(call) => {
							let args = call
								.arguments
								.iter()
								.map(|(key, value)| {
									let json = serde_json::to_string(value).unwrap_or_else(|_| "null".to_string());
									format!("{key}={json}")
								})
								.collect::<Vec<_>>()
								.join(", ");
							tool_calls.push(format!("{}({args})", call.name));
						}
						AssistantContent::Text(_) => {}
					}
				}

				if !thinking_parts.is_empty() {
					parts.push(format!("[Assistant thinking]: {}", thinking_parts.join("\n")));
				}
				if assistant.content.iter().any(|block| matches!(block, AssistantContent::Text(_))) {
					parts.push(format!("[Assistant]: {}", content_blocks_text(&assistant.content, "\n")));
				}
				if !tool_calls.is_empty() {
					parts.push(format!("[Assistant tool calls]: {}", tool_calls.join("; ")));
				}
			}
			Message::ToolResult(result) => {
				let content = user_content_text(&result.content, "");
				if !content.is_empty() {
					parts.push(format!("[Tool result]: {}", truncate_for_summary(&content, TOOL_RESULT_MAX_CHARS)));
				}
			}
			Message::System(_) => {}
		}
	}

	parts.join("\n\n")
}

// ============================================================================
// Summarization System Prompt
// ============================================================================

pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

#[cfg(test)]
mod utils_tests {
	//! Ports of upstream `test/compaction-nested-calls.test.ts` (1 case) and
	//! `test/compaction-serialization.test.ts` (3 cases).

	use super::*;
	use pi_ai::types::{
		AssistantMessage, NestedToolCallRecord, NestedToolCallStatus, NestedToolCalls, StopReason, TextContent,
		ToolResultMessage, UserContent,
	};

	fn json_map<const N: usize>(entries: [(&str, Value); N]) -> serde_json::Map<String, Value> {
		entries.into_iter().map(|(key, value)| (key.to_string(), value)).collect()
	}

	fn assistant_message_with_text(text: &str, timestamp: i64) -> AssistantMessage {
		AssistantMessage {
			content: vec![AssistantContent::Text(TextContent::new(text))],
			api: "anthropic".to_string(),
			provider: "anthropic".to_string(),
			model: "test".to_string(),
			response_model: None,
			response_id: None,
			provider_thinking_level: None,
			thinking_level: None,
			diagnostics: None,
			usage: pi_ai::types::Usage {
				input: 0,
				output: 0,
				cache_read: 0,
				cache_write: 0,
				cache_write_1h: None,
				reasoning: None,
				total_tokens: 0,
				cost: Default::default(),
			},
			stop_reason: StopReason::Stop,
			deferred: None,
			error_message: None,
			raw_stop_reason: None,
			end_turn: None,
			timestamp,
		}
	}

	// -- test/compaction-nested-calls.test.ts -----------------------------------

	#[test]
	fn include_files_touched_by_nested_calls_recorded_on_tool_results() {
		let result = AgentMessage::tool_result(ToolResultMessage {
			tool_call_id: "codemode-1".to_string(),
			tool_name: "codemode".to_string(),
			content: Vec::new(),
			details: None,
			usage: None,
			nested_calls: Some(NestedToolCalls {
				calls: vec![
					NestedToolCallRecord {
						id: "codemode-1/1".to_string(),
						name: "read".to_string(),
						arguments: Some(json_map([("path", Value::from("a.ts"))])),
						arguments_bytes: None,
						status: NestedToolCallStatus::Ok,
						duration_ms: None,
						error: None,
					},
					NestedToolCallRecord {
						id: "codemode-1/2".to_string(),
						name: "edit".to_string(),
						arguments: Some(json_map([("path", Value::from("b.ts")), ("edits", Value::Array(vec![]))])),
						arguments_bytes: None,
						status: NestedToolCallStatus::Ok,
						duration_ms: None,
						error: None,
					},
					NestedToolCallRecord {
						id: "codemode-1/3".to_string(),
						name: "write".to_string(),
						arguments: None,
						arguments_bytes: Some(40000),
						status: NestedToolCallStatus::Ok,
						duration_ms: None,
						error: None,
					},
				],
				complete: false,
			}),
			is_error: false,
			timestamp: 0,
		});
		let mut file_ops = create_file_ops();
		extract_file_ops_from_message(&result, &mut file_ops);
		assert_eq!(
			compute_file_lists(&file_ops),
			FileLists {
				read_files: vec!["a.ts".to_string()],
				modified_files: vec!["b.ts".to_string()],
			}
		);
	}

	// -- test/compaction-serialization.test.ts ----------------------------------

	fn tool_result_with_text(text: &str) -> Vec<Message> {
		vec![Message::ToolResult(ToolResultMessage {
			tool_call_id: "tc1".to_string(),
			tool_name: "read".to_string(),
			content: vec![UserContent::Text(TextContent::new(text))],
			details: None,
			usage: None,
			nested_calls: None,
			is_error: false,
			timestamp: crate::utils::time::now_ms(),
		})]
	}

	#[test]
	fn should_truncate_long_tool_results() {
		let long_content = "x".repeat(5000);
		let result = serialize_conversation(&tool_result_with_text(&long_content));

		assert!(result.contains("[Tool result]:"));
		assert!(result.contains("[... 3000 more characters truncated]"));
		assert!(!result.contains("x".repeat(3000).as_str()));
		// First 2000 chars should be present
		assert!(result.contains("x".repeat(2000).as_str()));
	}

	#[test]
	fn should_not_truncate_short_tool_results() {
		let short_content = "x".repeat(1500);
		let result = serialize_conversation(&tool_result_with_text(&short_content));

		assert_eq!(result, format!("[Tool result]: {short_content}"));
		assert!(!result.contains("truncated"));
	}

	#[test]
	fn should_not_truncate_assistant_or_user_messages() {
		let long_text = "y".repeat(5000);
		let messages = vec![
			Message::User(pi_ai::types::UserMessage {
				content: UserContentField::Blocks(vec![UserContent::Text(TextContent::new(long_text.clone()))]),
				timestamp: crate::utils::time::now_ms(),
			}),
			Message::Assistant(assistant_message_with_text(&long_text, crate::utils::time::now_ms())),
		];

		let result = serialize_conversation(&messages);

		assert!(!result.contains("truncated"));
		assert!(result.contains(long_text.as_str()));
	}
}
