//! Port of `packages/ai/src/utils/transcript.ts`.

use std::collections::BTreeMap;

use crate::types::{Message, SystemContent, SystemMessage, Tool, ToolReference, TranscriptContext};
use crate::utils::text::{get_system_message_text, system_content_text};

/// Build the leading system message for a prompt and tool set
/// (`createInitialSystemMessage`). Returns `None` when both are empty, so an
/// empty transcript stays empty.
pub fn create_initial_system_message(
	system_prompt: Option<&str>,
	tools: Option<&[Tool]>,
) -> Option<SystemMessage> {
	let has_system_prompt = system_prompt.map(|prompt| !prompt.is_empty()).unwrap_or(false);
	let has_tools = tools.map(|tools| !tools.is_empty()).unwrap_or(false);
	if !has_system_prompt && !has_tools {
		return None;
	}
	Some(SystemMessage {
		content: SystemContent::Text(system_prompt.unwrap_or("").to_string()),
		sections: None,
		tools_added: has_tools.then(|| tools.expect("has tools").to_vec()),
		tools_removed: None,
		timestamp: 0,
	})
}

/// Fold `Context.systemPrompt` and `Context.tools` into a leading system
/// message (`normalizeContext`). This is the only entry point that produces a
/// [`TranscriptContext`]; every provider-facing function expects the result.
pub fn normalize_context(context: &crate::types::Context) -> TranscriptContext {
	let initial_message =
		create_initial_system_message(context.system_prompt.as_deref(), context.tools.as_deref());
	let mut messages = Vec::with_capacity(context.messages.len() + 1);
	if let Some(initial) = initial_message {
		messages.push(Message::System(initial));
	}
	messages.extend(context.messages.iter().cloned());
	TranscriptContext::new(messages)
}

/// Any message list. The replay helpers only read entries whose role is
/// `"system"`, so agent transcripts that carry custom message roles can be
/// passed without filtering.
pub fn transcript_messages<'a>(messages: &'a [Message]) -> impl Iterator<Item = Option<&'a SystemMessage>> {
	messages
		.iter()
		.map(|message| match message {
			Message::System(system) => Some(system),
			_ => None,
		})
}

/// Return the leading system message, if the transcript starts with one
/// (`getInitialSystemMessage`).
pub fn get_initial_system_message(messages: &[Message]) -> Option<&SystemMessage> {
	match messages.first() {
		Some(Message::System(system)) => Some(system),
		_ => None,
	}
}

/// Drop the leading system message for APIs that carry the prompt outside the
/// message list (`withoutInitialSystemMessage`).
pub fn without_initial_system_message(messages: &[Message]) -> Vec<Message> {
	if get_initial_system_message(messages).is_some() {
		messages[1..].to_vec()
	} else {
		messages.to_vec()
	}
}

/// Resolve the tools available after applying every transcript delta in order
/// (`getCurrentTools`).
pub fn get_current_tools(messages: &[Message]) -> Vec<Tool> {
	let mut tools: BTreeMap<String, Tool> = BTreeMap::new();
	for message in messages {
		let Some(system) = as_system(message) else { continue };
		for tool in system.tools_removed.iter().flatten() {
			tools.remove(&tool.name);
		}
		for tool in system.tools_added.iter().flatten() {
			tools.insert(tool.name.clone(), tool.clone());
		}
	}
	tools.into_values().collect()
}

/// Replay every system message into one leading system message holding the
/// current prompt and tools (`getCurrentSystemMessage`). Later `content` is
/// appended to the base prompt, `sections` are patched by name, and tools are
/// resolved with [`get_current_tools`].
pub fn get_current_system_message(messages: &[Message]) -> Option<SystemMessage> {
	let mut content: Vec<String> = Vec::new();
	let mut sections: BTreeMap<String, Option<String>> = BTreeMap::new();
	let mut timestamp: Option<i64> = None;
	for message in messages {
		let Some(system) = as_system(message) else { continue };
		timestamp.get_or_insert_with(|| system.timestamp);
		let text = system_content_text(&system.content);
		if !text.is_empty() {
			content.push(text);
		}
		for (name, value) in system.sections.iter().flatten() {
			match value {
				None => {
					sections.remove(name);
				}
				Some(value) => {
					sections.insert(name.clone(), Some(value.clone()));
				}
			}
		}
	}
	let tools = get_current_tools(messages);
	if timestamp.is_none() && tools.is_empty() {
		return None;
	}
	Some(SystemMessage {
		content: SystemContent::Text(content.join("\n\n")),
		sections: (!sections.is_empty()).then(|| sections.into_iter().map(|(k, v)| (k, v)).collect()),
		tools_added: (!tools.is_empty()).then(|| tools),
		tools_removed: None,
		timestamp: timestamp.unwrap_or(0),
	})
}

/// Render the current system prompt text after replaying every system message
/// (`getCurrentSystemPrompt`).
pub fn get_current_system_prompt(messages: &[Message]) -> String {
	get_current_system_message(messages)
		.map(|message| get_system_message_text(&message))
		.unwrap_or_default()
}

/// Rebuild the transcript for APIs without mid-conversation system messages:
/// the replayed system message leads, and every later system message is
/// dropped (`collapseSystemMessages`).
pub fn collapse_system_messages(context: &TranscriptContext) -> TranscriptContext {
	let head = get_current_system_message(context.messages());
	let messages: Vec<Message> = context
		.messages()
		.iter()
		.filter(|message| message.role() != "system")
		.cloned()
		.collect();
	let mut out = Vec::with_capacity(messages.len() + 1);
	if let Some(head) = head {
		out.push(Message::System(head));
	}
	out.extend(messages);
	TranscriptContext::new(out)
}

/// Keep later system messages in place when the model accepts them; otherwise
/// collapse them (`resolveTranscript`).
pub fn resolve_transcript(
	context: &TranscriptContext,
	supports_mid_convo_system_messages: Option<bool>,
) -> TranscriptContext {
	if supports_mid_convo_system_messages.unwrap_or(false) {
		context.clone()
	} else {
		collapse_system_messages(context)
	}
}

/// Strip executable and display-only fields from a tool before transcript
/// comparison or persistence (`toToolDeclaration`).
pub fn to_tool_declaration(tool: &Tool) -> Tool {
	Tool {
		name: tool.name.clone(),
		description: tool.description.clone(),
		parameters: serde_json::from_value(serde_json::to_value(&tool.parameters).expect("schema is JSON"))
			.expect("schema round-trips"),
		constrained_sampling: tool.constrained_sampling.clone(),
	}
}

/// Whether two tools declare the same interface to the model
/// (`declarationsEqual`). Both sides go through [`to_tool_declaration`] first;
/// comparing the serialized declarations is exact.
pub fn declarations_equal(left: &Tool, right: &Tool) -> bool {
	serde_json::to_string(&to_tool_declaration(left)).expect("tool serializes")
		== serde_json::to_string(&to_tool_declaration(right)).expect("tool serializes")
}

/// Tool state changes between two complete states (`ToolStateChanges`).
pub struct ToolStateChanges {
	pub tools_added: Vec<Tool>,
	pub tools_removed: Vec<ToolReference>,
}

/// Compare two complete tool states. A changed definition is a removal
/// followed by an addition (`getToolStateChanges`).
pub fn get_tool_state_changes(previous: &[Tool], current: &[Tool]) -> ToolStateChanges {
	let previous_tools: BTreeMap<&str, &Tool> = previous.iter().map(|tool| (tool.name.as_str(), tool)).collect();
	let current_tools: BTreeMap<&str, &Tool> = current.iter().map(|tool| (tool.name.as_str(), tool)).collect();
	ToolStateChanges {
		tools_added: current
			.iter()
			.filter(|tool| {
				previous_tools
					.get(tool.name.as_str())
					.map(|previous_tool| !declarations_equal(previous_tool, tool))
					.unwrap_or(true)
			})
			.map(to_tool_declaration)
			.collect(),
		tools_removed: previous
			.iter()
			.filter(|tool| {
				current_tools
					.get(tool.name.as_str())
					.map(|current_tool| !declarations_equal(tool, current_tool))
					.unwrap_or(true)
			})
			.map(|tool| ToolReference { name: tool.name.clone() })
			.collect(),
	}
}

/// Every definition referenced by transcript tool state, in first-declaration
/// order (`getDeclaredTools`).
pub fn get_declared_tools(messages: &[Message]) -> Vec<Tool> {
	let mut definitions: BTreeMap<String, Tool> = BTreeMap::new();
	for message in messages {
		let Some(system) = as_system(message) else { continue };
		for tool in system.tools_added.iter().flatten() {
			definitions.insert(tool.name.clone(), tool.clone());
		}
	}
	definitions.into_values().collect()
}

/// Whether a tool name was declared twice with different definitions
/// (`hasToolRedefinitions`).
pub fn has_tool_redefinitions(messages: &[Message]) -> bool {
	let mut declared: BTreeMap<String, Tool> = BTreeMap::new();
	for message in messages {
		let Some(system) = as_system(message) else { continue };
		for tool in system.tools_added.iter().flatten() {
			if let Some(previous) = declared.get(&tool.name) {
				if !declarations_equal(previous, tool) {
					return true;
				}
			}
			declared.insert(tool.name.clone(), tool.clone());
		}
	}
	false
}

/// Whether tool history contains a removal or same-name redeclaration that an
/// addition-only transport cannot replay (`hasNonAdditiveToolChanges`).
pub fn has_non_additive_tool_changes(messages: &[Message]) -> bool {
	let mut declared: BTreeMap<String, ()> = BTreeMap::new();
	for message in messages {
		let Some(system) = as_system(message) else { continue };
		if system.tools_removed.as_ref().map(|t| !t.is_empty()).unwrap_or(false) {
			return true;
		}
		for tool in system.tools_added.iter().flatten() {
			if declared.contains_key(&tool.name) {
				return true;
			}
			declared.insert(tool.name.clone(), ());
		}
	}
	false
}

/// Tool declarations split between the top-level request field and in-place
/// additions (`TranscriptTools`).
pub struct TranscriptTools {
	/// Tools sent in the top-level request field.
	pub request_tools: Vec<Tool>,
	/// Whether later system messages carry their own `toolsAdded` as in-place
	/// additions. When false, `request_tools` already holds the complete
	/// current tool set.
	pub anchors_additions: bool,
}

/// Split tool declarations between the top-level request field and in-place
/// additions (`resolveTranscriptTools`).
pub fn resolve_transcript_tools(messages: &[Message], supports_tool_additions: bool) -> TranscriptTools {
	let anchors_additions = supports_tool_additions && !has_non_additive_tool_changes(messages);
	TranscriptTools {
		request_tools: if anchors_additions {
			get_initial_system_message(messages)
				.and_then(|system| system.tools_added.clone())
				.unwrap_or_default()
		} else {
			get_current_tools(messages)
		},
		anchors_additions,
	}
}

fn as_system(message: &Message) -> Option<&SystemMessage> {
	match message {
		Message::System(system) => Some(system),
		_ => None,
	}
}

#[cfg(test)]
mod tests {
	//! Port of `packages/ai/test/system-message-replay.test.ts`. The upstream
	//! `test/transcript-tool-changes.test.ts` cases are provider end-to-end
	//! (they capture HTTP payloads through `streamSimple` against the
	//! anthropic/openai API modules) and travel with those slices; see the
	//! port report for the skipped list.

	use std::collections::BTreeMap;

	use serde_json::json;

	use super::*;
	use crate::types::{
		AssistantContent, ConstrainedSamplingSetting, Context, TextContent, UserContentField, UserMessage,
	};
	use crate::utils::text::render_system_message_update;

	fn tool(name: &str) -> Tool {
		tool_with_description(name, &format!("{name} tool"))
	}

	fn tool_with_description(name: &str, description: &str) -> Tool {
		Tool {
			name: name.to_string(),
			description: description.to_string(),
			parameters: json!({ "type": "object", "properties": {} }),
			constrained_sampling: None,
		}
	}

	fn system_message(
		content: &str,
		sections: BTreeMap<String, Option<String>>,
		tools_added: Option<Vec<Tool>>,
		tools_removed: Option<Vec<ToolReference>>,
		timestamp: i64,
	) -> Message {
		Message::System(SystemMessage {
			content: SystemContent::Text(content.to_string()),
			sections: (!sections.is_empty()).then(|| sections),
			tools_added,
			tools_removed,
			timestamp,
		})
	}

	fn user_message(content: &str, timestamp: i64) -> Message {
		Message::User(UserMessage {
			content: UserContentField::Text(content.to_string()),
			timestamp,
		})
	}

	fn assistant_message(timestamp: i64) -> Message {
		Message::Assistant(crate::types::AssistantMessage {
			content: vec![AssistantContent::Text(TextContent::new("ok"))],
			api: "test".to_string(),
			provider: "test".to_string(),
			model: "test-model".to_string(),
			response_model: None,
			response_id: None,
			provider_thinking_level: None,
			thinking_level: None,
			diagnostics: None,
			usage: Default::default(),
			stop_reason: crate::types::StopReason::Stop,
			deferred: None,
			error_message: None,
			raw_stop_reason: None,
			end_turn: None,
			timestamp,
		})
	}

	fn transcript() -> TranscriptContext {
		let mut base_sections = BTreeMap::new();
		base_sections.insert("a".to_string(), Some("<a>1</a>".to_string()));
		base_sections.insert("b".to_string(), Some("<b>1</b>".to_string()));
		let mut patch_sections = BTreeMap::new();
		patch_sections.insert("a".to_string(), Some("<a>2</a>".to_string()));
		patch_sections.insert("b".to_string(), None);
		patch_sections.insert("c".to_string(), Some("<c>1</c>".to_string()));
		normalize_context(&Context {
			system_prompt: None,
			messages: vec![
				system_message(
					"base",
					base_sections,
					Some(vec![tool("first")]),
					None,
					10,
				),
				user_message("hello", 11),
				system_message("also do this", BTreeMap::new(), None, None, 12),
				assistant_message(13),
				system_message(
					"",
					patch_sections,
					Some(vec![tool("second")]),
					Some(vec![ToolReference { name: "first".to_string() }]),
					14,
				),
			],
			tools: None,
		})
	}

	fn sections(entries: &[(&str, &str)]) -> BTreeMap<String, Option<String>> {
		entries
			.iter()
			.map(|(name, value)| (name.to_string(), Some(value.to_string())))
			.collect()
	}

	#[test]
	fn replays_content_sections_and_tools_into_one_leading_message() {
		let messages = transcript().into_messages();
		let current = get_current_system_message(&messages).expect("system message");

		let mut expected_sections = BTreeMap::new();
		expected_sections.insert("a".to_string(), Some("<a>2</a>".to_string()));
		expected_sections.insert("c".to_string(), Some("<c>1</c>".to_string()));
		assert_eq!(
			current,
			SystemMessage {
				content: SystemContent::Text("base\n\nalso do this".to_string()),
				sections: Some(expected_sections),
				tools_added: Some(vec![tool("second")]),
				tools_removed: None,
				timestamp: 10,
			}
		);
		assert_eq!(
			get_current_system_prompt(&messages),
			"base\n\nalso do this\n\n<a>2</a>\n\n<c>1</c>"
		);
	}

	#[test]
	fn collapse_keeps_only_non_system_messages_after_the_replayed_head() {
		let collapsed = collapse_system_messages(&transcript());
		let roles: Vec<&str> = collapsed.messages().iter().map(|message| message.role()).collect();
		assert_eq!(roles, vec!["system", "user", "assistant"]);
		assert_eq!(collapse_system_messages(&collapsed), collapsed);
	}

	#[test]
	fn replay_of_a_transcript_without_system_messages_is_empty() {
		let context = normalize_context(&Context {
			system_prompt: None,
			messages: vec![user_message("hi", 1)],
			tools: None,
		});
		assert!(get_current_system_message(context.messages()).is_none());
		assert_eq!(get_current_system_prompt(context.messages()), "");
		let collapsed = collapse_system_messages(&context);
		assert_eq!(collapsed.messages(), context.messages());
	}

	#[test]
	fn a_late_full_patch_on_a_transcript_without_a_leading_message_replays_as_the_prompt() {
		let mut sections = BTreeMap::new();
		sections.insert("preamble".to_string(), Some("You are pi.".to_string()));
		let context = normalize_context(&Context {
			system_prompt: None,
			messages: vec![
				user_message("old session", 1),
				system_message("", sections, Some(vec![tool("x")]), None, 2),
			],
			tools: None,
		});
		assert_eq!(get_current_system_prompt(context.messages()), "You are pi.");
		let collapsed = collapse_system_messages(&context);
		let Message::System(head) = &collapsed.messages()[0] else {
			panic!("expected a system head");
		};
		assert_eq!(head.tools_added.as_deref(), Some(&[tool("x")][..]));
	}

	#[test]
	fn renders_complete_prompts_and_framed_updates() {
		let messages = transcript().into_messages();
		let Message::System(leading) = &messages[0] else {
			panic!("expected a leading system message");
		};
		let Message::System(update) = &messages[4] else {
			panic!("expected an update system message");
		};
		assert_eq!(get_system_message_text(leading), "base\n\n<a>1</a>\n\n<b>1</b>");
		assert_eq!(
			render_system_message_update(update),
			[
				"Updated system prompt section \"a\":\n\n<a>2</a>",
				"Removed system prompt section \"b\".",
				"Updated system prompt section \"c\":\n\n<c>1</c>",
			]
			.join("\n\n")
		);
	}

	#[test]
	fn normalizes_the_legacy_prompt_and_tool_fields_into_a_leading_system_message() {
		let messages = vec![user_message("hi", 1)];
		let plain = normalize_context(&Context { system_prompt: None, messages: messages.clone(), tools: None });
		assert_eq!(plain.messages(), messages.as_slice());

		let empty = normalize_context(&Context {
			system_prompt: Some(String::new()),
			messages: messages.clone(),
			tools: Some(Vec::new()),
		});
		assert_eq!(empty.messages(), messages.as_slice());

		let legacy = normalize_context(&Context {
			system_prompt: Some("be brief".to_string()),
			messages: messages.clone(),
			tools: Some(vec![tool("a")]),
		});
		let expected = [
			Message::System(SystemMessage {
				content: SystemContent::Text("be brief".to_string()),
				sections: None,
				tools_added: Some(vec![tool("a")]),
				tools_removed: None,
				timestamp: 0,
			}),
			messages[0].clone(),
		];
		assert_eq!(legacy.messages(), expected.as_slice());
	}

	#[test]
	fn compares_tool_declarations_without_executable_or_undefined_fields() {
		let mut executable = tool("a");
		// `constrainedSampling: undefined` (absent) matches a plain declaration.
		assert!(declarations_equal(&executable, &tool("a")));
		assert!(!declarations_equal(&tool("a"), &tool_with_description("a", "changed")));
		executable.constrained_sampling = Some(ConstrainedSamplingSetting::Disabled(false));
		assert!(!declarations_equal(&tool("a"), &executable));
	}

	#[test]
	fn tool_state_changes_treat_changed_definitions_as_removal_plus_addition() {
		let previous = vec![tool("a"), tool("b")];
		let current = vec![tool_with_description("b", "changed"), tool("c")];
		let changes = get_tool_state_changes(&previous, &current);
		assert_eq!(changes.tools_added, vec![tool_with_description("b", "changed"), tool("c")]);
		assert_eq!(
			changes.tools_removed,
			vec![ToolReference { name: "a".to_string() }, ToolReference { name: "b".to_string() }]
		);

		let unchanged = get_tool_state_changes(&[tool("a")], &[tool("a")]);
		assert!(unchanged.tools_added.is_empty());
		assert!(unchanged.tools_removed.is_empty());
	}

	#[test]
	fn detects_non_additive_tool_history_and_redefinitions() {
		assert!(has_non_additive_tool_changes(&transcript().into_messages()));
		assert!(!has_tool_redefinitions(&transcript().into_messages()));

		let additive = normalize_context(&Context {
			system_prompt: None,
			messages: vec![
				system_message("", BTreeMap::new(), Some(vec![tool("a")]), None, 1),
				system_message("", BTreeMap::new(), Some(vec![tool("b")]), None, 2),
			],
			tools: None,
		});
		assert!(!has_non_additive_tool_changes(additive.messages()));

		let redeclared = normalize_context(&Context {
			system_prompt: None,
			messages: vec![
				system_message("", BTreeMap::new(), Some(vec![tool("a")]), None, 1),
				system_message("", BTreeMap::new(), Some(vec![tool_with_description("a", "changed")]), None, 2),
			],
			tools: None,
		});
		assert!(has_non_additive_tool_changes(redeclared.messages()));
		assert!(has_tool_redefinitions(redeclared.messages()));
	}

	#[test]
	fn sections_helper_matches_btree_shape() {
		assert_eq!(sections(&[("a", "<a>1</a>")])["a"], Some("<a>1</a>".to_string()));
	}
}
