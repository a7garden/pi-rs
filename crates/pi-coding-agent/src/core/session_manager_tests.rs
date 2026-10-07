//! Port of upstream `test/session-manager/*.test.ts` (8 files, 117 cases):
//! build-context, custom-session-id, file-operations, labels, load-entries,
//! migration, save-entry, tree-traversal. Helpers mirror
//! `test/utilities.ts` (`userMsg`, `assistantMsg`, `readSessionFileRoles`).

use super::*;
use serde_json::json;
use std::io::{Seek, Write};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(label: &str) -> String {
	let dir = std::env::temp_dir().join(format!(
		"pi-session-manager-{}-{}-{}",
		label,
		std::process::id(),
		TEMP_COUNTER.fetch_add(1, Ordering::SeqCst)
	));
	std::fs::create_dir_all(&dir).expect("create temp dir");
	dir.to_string_lossy().into_owned()
}

fn cleanup_dir(dir: &str) {
	let _ = std::fs::remove_dir_all(dir);
}

// `test/utilities.ts` helpers ------------------------------------------------

fn user_msg(text: &str) -> AgentMessage {
	AgentMessage::user(pi_ai::types::UserMessage {
		content: pi_ai::types::UserContentField::Text(text.to_string()),
		timestamp: crate::utils::time::now_ms(),
	})
}

fn assistant_message(
	text: &str,
	provider: &str,
	model: &str,
	usage: pi_ai::types::Usage,
	timestamp: i64,
) -> pi_ai::types::AssistantMessage {
	pi_ai::types::AssistantMessage {
		content: vec![pi_ai::types::AssistantContent::Text(pi_ai::types::TextContent::new(text))],
		api: "anthropic-messages".to_string(),
		provider: provider.to_string(),
		model: model.to_string(),
		response_model: None,
		response_id: None,
		provider_thinking_level: None,
		thinking_level: None,
		diagnostics: None,
		usage,
		stop_reason: pi_ai::types::StopReason::Stop,
		deferred: None,
		error_message: None,
		raw_stop_reason: None,
		end_turn: None,
		timestamp,
	}
}

fn assistant_msg(text: &str) -> AgentMessage {
	AgentMessage::assistant(assistant_message(
		text,
		"anthropic",
		"test",
		test_usage(1, 1, 0, 0, 2),
		crate::utils::time::now_ms(),
	))
}

fn test_usage(input: u64, output: u64, cache_read: u64, cache_write: u64, total: u64) -> pi_ai::types::Usage {
	pi_ai::types::Usage {
		input,
		output,
		cache_read,
		cache_write,
		cache_write_1h: None,
		reasoning: None,
		total_tokens: total,
		cost: pi_ai::types::UsageCost::default(),
	}
}

fn big_usage() -> pi_ai::types::Usage {
	pi_ai::types::Usage {
		input: 10,
		output: 20,
		cache_read: 30,
		cache_write: 40,
		cache_write_1h: None,
		reasoning: None,
		total_tokens: 100,
		cost: pi_ai::types::UsageCost {
			input: 0.1,
			output: 0.2,
			cache_read: 0.3,
			cache_write: 0.4,
			total: 1.0,
		},
	}
}

/// `readSessionFileRoles` — one label per record: message role for message
/// entries, otherwise the entry type ("session" for the header).
fn read_session_file_roles(file: &str) -> Vec<String> {
	let content = std::fs::read_to_string(file).expect("read session file");
	content
		.trim()
		.split('\n')
		.filter(|line| !line.trim().is_empty())
		.map(|line| {
			let entry: FileEntry = serde_json::from_str(line).expect("parse session line");
			match entry.as_entry() {
				Some(SessionEntry::Message(message)) => message.message.role().to_string(),
				Some(entry) => entry.type_name().to_string(),
				None => "session".to_string(),
			}
		})
		.collect()
}

fn is_uuid_v7(id: &str) -> bool {
	let bytes = id.as_bytes();
	if bytes.len() != 36 {
		return false;
	}
	for (index, byte) in bytes.iter().enumerate() {
		let is_dash = matches!(index, 8 | 13 | 18 | 23);
		if is_dash {
			if *byte != b'-' {
				return false;
			}
		} else if !byte.is_ascii_hexdigit() {
			return false;
		}
	}
	bytes[14] == b'7' && matches!(bytes[19], b'8' | b'9' | b'a' | b'b')
}

/// `^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z$` (upstream filename regex).
fn is_file_timestamp(prefix: &str) -> bool {
	let bytes = prefix.as_bytes();
	if bytes.len() != 24 {
		return false;
	}
	for (index, byte) in bytes.iter().enumerate() {
		let expected: u8 = match index {
			10 => b'T',
			23 => b'Z',
			4 | 7 | 13 | 16 | 19 => b'-',
			_ => b'0',
		};
		if index == 10 || index == 23 || matches!(index, 4 | 7 | 13 | 16 | 19) {
			if *byte != expected {
				return false;
			}
		} else if !byte.is_ascii_digit() {
			return false;
		}
	}
	true
}

// Message assertion helpers --------------------------------------------------

fn message_role(message: &AgentMessage) -> String {
	message.role().to_string()
}

fn custom_str<'a>(message: &'a AgentMessage, field: &str) -> Option<&'a str> {
	match message {
		AgentMessage::Custom(value) => value.get(field).and_then(|value| value.as_str()),
		_ => None,
	}
}

fn user_content_text(message: &AgentMessage) -> Option<String> {
	match message {
		AgentMessage::Llm(pi_ai::types::Message::User(user)) => match &user.content {
			pi_ai::types::UserContentField::Text(text) => Some(text.clone()),
			pi_ai::types::UserContentField::Blocks(_) => None,
		},
		AgentMessage::Custom(value) => value.get("content").and_then(|value| value.as_str()).map(String::from),
		_ => None,
	}
}

fn assistant_block_text(message: &AgentMessage, index: usize) -> Option<String> {
	match message {
		AgentMessage::Llm(pi_ai::types::Message::Assistant(assistant)) => assistant.content.get(index).and_then(
			|block| match block {
				pi_ai::types::AssistantContent::Text(text) => Some(text.text.clone()),
				_ => None,
			},
		),
		AgentMessage::Custom(value) => value
			.get("content")
			.and_then(|value| value.as_array())
			.and_then(|blocks| blocks.get(index))
			.and_then(|block| block.get("text"))
			.and_then(|value| value.as_str())
			.map(String::from),
		_ => None,
	}
}

// ===========================================================================
// Upstream test/session-manager/build-context.test.ts
// ===========================================================================

fn bc_msg(id: &str, parent_id: Option<&str>, role: &str, text: &str) -> SessionEntry {
	let base = base_fields(id, parent_id);
	if role == "user" {
		SessionEntry::Message(SessionMessageEntry {
			base,
			message: AgentMessage::user(pi_ai::types::UserMessage {
				content: pi_ai::types::UserContentField::Text(text.to_string()),
				timestamp: 1,
			}),
		})
	} else {
		SessionEntry::Message(SessionMessageEntry {
			base,
			message: AgentMessage::assistant(assistant_message(text, "anthropic", "claude-test", test_usage(1, 1, 0, 0, 2), 1)),
		})
	}
}

fn base_fields(id: &str, parent_id: Option<&str>) -> SessionEntryBase {
	SessionEntryBase {
		id: id.to_string(),
		parent_id: parent_id.map(String::from),
		timestamp: "2025-01-01T00:00:00Z".to_string(),
	}
}

fn bc_compaction(id: &str, parent_id: Option<&str>, summary: &str, first_kept_entry_id: &str) -> SessionEntry {
	SessionEntry::Compaction(CompactionEntry {
		base: base_fields(id, parent_id),
		summary: summary.to_string(),
		first_kept_entry_id: first_kept_entry_id.to_string(),
		tokens_before: 1000,
		details: None,
		usage: None,
		from_hook: None,
		system_message: None,
		first_kept_entry_index: None,
	})
}

fn bc_branch_summary(id: &str, parent_id: Option<&str>, summary: &str, from_id: &str) -> SessionEntry {
	SessionEntry::BranchSummary(BranchSummaryEntry {
		base: base_fields(id, parent_id),
		summary: summary.to_string(),
		from_id: from_id.to_string(),
		details: None,
		usage: None,
		from_hook: None,
	})
}

fn bc_custom(id: &str, parent_id: Option<&str>, custom_type: &str, data: serde_json::Value) -> SessionEntry {
	SessionEntry::Custom(CustomEntry {
		base: base_fields(id, parent_id),
		custom_type: custom_type.to_string(),
		data: Some(data),
	})
}

fn bc_thinking_level(id: &str, parent_id: Option<&str>, level: &str) -> SessionEntry {
	SessionEntry::ThinkingLevelChange(ThinkingLevelChangeEntry {
		base: base_fields(id, parent_id),
		thinking_level: level.to_string(),
	})
}

fn bc_model_change(id: &str, parent_id: Option<&str>, provider: &str, model_id: &str) -> SessionEntry {
	SessionEntry::ModelChange(ModelChangeEntry {
		base: base_fields(id, parent_id),
		provider: provider.to_string(),
		model_id: model_id.to_string(),
	})
}

#[test]
fn build_context_empty_entries_returns_empty_context() {
	let ctx = build_session_context(&[], Leaf::Undefined, None);
	assert!(ctx.messages.is_empty());
	assert_eq!(ctx.thinking_level, "off");
	assert!(ctx.model.is_none());
}

#[test]
fn build_context_single_user_message() {
	let entries = vec![bc_msg("1", None, "user", "hello")];
	let ctx = build_session_context(&entries, Leaf::Undefined, None);
	assert_eq!(ctx.messages.len(), 1);
	assert_eq!(message_role(&ctx.messages[0]), "user");
}

#[test]
fn build_context_simple_conversation() {
	let entries = vec![
		bc_msg("1", None, "user", "hello"),
		bc_msg("2", Some("1"), "assistant", "hi there"),
		bc_msg("3", Some("2"), "user", "how are you"),
		bc_msg("4", Some("3"), "assistant", "great"),
	];
	let ctx = build_session_context(&entries, Leaf::Undefined, None);
	assert_eq!(ctx.messages.len(), 4);
	let roles: Vec<String> = ctx.messages.iter().map(message_role).collect();
	assert_eq!(roles, ["user", "assistant", "user", "assistant"]);
}

#[test]
fn build_context_tracks_thinking_level_changes() {
	let entries = vec![
		bc_msg("1", None, "user", "hello"),
		bc_thinking_level("2", Some("1"), "high"),
		bc_msg("3", Some("2"), "assistant", "thinking hard"),
	];
	let ctx = build_session_context(&entries, Leaf::Undefined, None);
	assert_eq!(ctx.thinking_level, "high");
	assert_eq!(ctx.messages.len(), 2);
}

#[test]
fn build_context_tracks_model_from_assistant_message() {
	let entries = vec![bc_msg("1", None, "user", "hello"), bc_msg("2", Some("1"), "assistant", "hi")];
	let ctx = build_session_context(&entries, Leaf::Undefined, None);
	assert_eq!(
		ctx.model,
		Some(SessionModelRef { provider: "anthropic".to_string(), model_id: "claude-test".to_string() })
	);
}

#[test]
fn build_context_tracks_model_from_model_change_entry() {
	let entries = vec![
		bc_msg("1", None, "user", "hello"),
		bc_model_change("2", Some("1"), "openai", "gpt-4"),
		bc_msg("3", Some("2"), "assistant", "hi"),
	];
	let ctx = build_session_context(&entries, Leaf::Undefined, None);
	// Assistant message overwrites model change
	assert_eq!(
		ctx.model,
		Some(SessionModelRef { provider: "anthropic".to_string(), model_id: "claude-test".to_string() })
	);
}

#[test]
fn build_context_includes_summary_before_kept_messages() {
	let entries = vec![
		bc_msg("1", None, "user", "first"),
		bc_msg("2", Some("1"), "assistant", "response1"),
		bc_msg("3", Some("2"), "user", "second"),
		bc_msg("4", Some("3"), "assistant", "response2"),
		bc_compaction("5", Some("4"), "Summary of first two turns", "3"),
		bc_msg("6", Some("5"), "user", "third"),
		bc_msg("7", Some("6"), "assistant", "response3"),
	];
	let ctx = build_session_context(&entries, Leaf::Undefined, None);

	// Should have: summary + kept (3,4) + after (6,7) = 5 messages
	assert_eq!(ctx.messages.len(), 5);
	assert!(custom_str(&ctx.messages[0], "summary").unwrap().contains("Summary of first two turns"));
	assert_eq!(user_content_text(&ctx.messages[1]).unwrap(), "second");
	assert_eq!(assistant_block_text(&ctx.messages[2], 0).unwrap(), "response2");
	assert_eq!(user_content_text(&ctx.messages[3]).unwrap(), "third");
	assert_eq!(assistant_block_text(&ctx.messages[4], 0).unwrap(), "response3");
}

#[test]
fn build_context_handles_compaction_keeping_from_first_message() {
	let entries = vec![
		bc_msg("1", None, "user", "first"),
		bc_msg("2", Some("1"), "assistant", "response"),
		bc_compaction("3", Some("2"), "Empty summary", "1"),
		bc_msg("4", Some("3"), "user", "second"),
	];
	let ctx = build_session_context(&entries, Leaf::Undefined, None);

	// Summary + all messages (1,2,4)
	assert_eq!(ctx.messages.len(), 4);
	assert!(custom_str(&ctx.messages[0], "summary").unwrap().contains("Empty summary"));
}

#[test]
fn build_context_multiple_compactions_uses_latest() {
	let entries = vec![
		bc_msg("1", None, "user", "a"),
		bc_msg("2", Some("1"), "assistant", "b"),
		bc_compaction("3", Some("2"), "First summary", "1"),
		bc_msg("4", Some("3"), "user", "c"),
		bc_msg("5", Some("4"), "assistant", "d"),
		bc_compaction("6", Some("5"), "Second summary", "4"),
		bc_msg("7", Some("6"), "user", "e"),
	];
	let ctx = build_session_context(&entries, Leaf::Undefined, None);

	// Should use second summary, keep from 4
	assert_eq!(ctx.messages.len(), 4);
	assert!(custom_str(&ctx.messages[0], "summary").unwrap().contains("Second summary"));
}

#[test]
fn build_context_entries_returns_compaction_aware_entries_including_custom_entries() {
	let entries = vec![
		bc_msg("1", None, "user", "first"),
		bc_custom("2", Some("1"), "old-state", json!({ "hidden": true })),
		bc_msg("3", Some("2"), "assistant", "response1"),
		bc_custom("4", Some("3"), "kept-card", json!({ "title": "Kept" })),
		bc_msg("5", Some("4"), "user", "second"),
		bc_compaction("6", Some("5"), "Summary", "4"),
		bc_custom("7", Some("6"), "after-card", json!({ "title": "After" })),
		bc_msg("8", Some("7"), "assistant", "response2"),
	];

	let context_entries = build_context_entries(&entries, Leaf::Undefined, None);
	let ids: Vec<&str> = context_entries.iter().map(|entry| entry.id()).collect();
	assert_eq!(ids, ["6", "4", "5", "7", "8"]);
	let ctx = build_session_context(&entries, Leaf::Undefined, None);
	let roles: Vec<String> = ctx.messages.iter().map(message_role).collect();
	assert_eq!(roles, ["compactionSummary", "user", "assistant"]);
}

#[test]
fn build_context_keeps_settings_from_the_full_path_after_compaction() {
	let entries = vec![
		bc_msg("1", None, "user", "first"),
		bc_thinking_level("2", Some("1"), "high"),
		bc_msg("3", Some("2"), "assistant", "response1"),
		bc_msg("4", Some("3"), "user", "second"),
		bc_compaction("5", Some("4"), "Summary", "4"),
	];

	let ctx = build_session_context(&entries, Leaf::Undefined, None);
	assert_eq!(ctx.thinking_level, "high");
	let roles: Vec<String> = ctx.messages.iter().map(message_role).collect();
	assert_eq!(roles, ["compactionSummary", "user"]);
}

#[test]
fn build_context_follows_path_to_specified_leaf() {
	// Tree:
	//   1 -> 2 -> 3 (branch A)
	//         \-> 4 (branch B)
	let entries = vec![
		bc_msg("1", None, "user", "start"),
		bc_msg("2", Some("1"), "assistant", "response"),
		bc_msg("3", Some("2"), "user", "branch A"),
		bc_msg("4", Some("2"), "user", "branch B"),
	];

	let ctx_a = build_session_context(&entries, Leaf::Id("3"), None);
	assert_eq!(ctx_a.messages.len(), 3);
	assert_eq!(user_content_text(&ctx_a.messages[2]).unwrap(), "branch A");

	let ctx_b = build_session_context(&entries, Leaf::Id("4"), None);
	assert_eq!(ctx_b.messages.len(), 3);
	assert_eq!(user_content_text(&ctx_b.messages[2]).unwrap(), "branch B");
}

#[test]
fn build_context_includes_branch_summary_in_path() {
	let entries = vec![
		bc_msg("1", None, "user", "start"),
		bc_msg("2", Some("1"), "assistant", "response"),
		bc_msg("3", Some("2"), "user", "abandoned path"),
		bc_branch_summary("4", Some("2"), "Summary of abandoned work", "3"),
		bc_msg("5", Some("4"), "user", "new direction"),
	];
	let ctx = build_session_context(&entries, Leaf::Id("5"), None);

	assert_eq!(ctx.messages.len(), 4);
	assert!(custom_str(&ctx.messages[2], "summary").unwrap().contains("Summary of abandoned work"));
	assert_eq!(user_content_text(&ctx.messages[3]).unwrap(), "new direction");
}

#[test]
fn build_context_complex_tree_with_multiple_branches_and_compaction() {
	// Tree:
	//   1 -> 2 -> 3 -> 4 -> compaction(5) -> 6 -> 7 (main path)
	//              \-> 8 -> 9 (abandoned branch)
	//                    \-> branchSummary(10) -> 11 (resumed from 3)
	let entries = vec![
		bc_msg("1", None, "user", "start"),
		bc_msg("2", Some("1"), "assistant", "r1"),
		bc_msg("3", Some("2"), "user", "q2"),
		bc_msg("4", Some("3"), "assistant", "r2"),
		bc_compaction("5", Some("4"), "Compacted history", "3"),
		bc_msg("6", Some("5"), "user", "q3"),
		bc_msg("7", Some("6"), "assistant", "r3"),
		// Abandoned branch from 3
		bc_msg("8", Some("3"), "user", "wrong path"),
		bc_msg("9", Some("8"), "assistant", "wrong response"),
		// Branch summary resuming from 3
		bc_branch_summary("10", Some("3"), "Tried wrong approach", "9"),
		bc_msg("11", Some("10"), "user", "better approach"),
	];

	// Main path to 7: summary + kept(3,4) + after(6,7)
	let ctx_main = build_session_context(&entries, Leaf::Id("7"), None);
	assert_eq!(ctx_main.messages.len(), 5);
	assert!(custom_str(&ctx_main.messages[0], "summary").unwrap().contains("Compacted history"));
	assert_eq!(user_content_text(&ctx_main.messages[1]).unwrap(), "q2");
	assert_eq!(assistant_block_text(&ctx_main.messages[2], 0).unwrap(), "r2");
	assert_eq!(user_content_text(&ctx_main.messages[3]).unwrap(), "q3");
	assert_eq!(assistant_block_text(&ctx_main.messages[4], 0).unwrap(), "r3");

	// Branch path to 11: 1,2,3 + branch_summary + 11
	let ctx_branch = build_session_context(&entries, Leaf::Id("11"), None);
	assert_eq!(ctx_branch.messages.len(), 5);
	assert_eq!(user_content_text(&ctx_branch.messages[0]).unwrap(), "start");
	assert_eq!(assistant_block_text(&ctx_branch.messages[1], 0).unwrap(), "r1");
	assert_eq!(user_content_text(&ctx_branch.messages[2]).unwrap(), "q2");
	assert!(custom_str(&ctx_branch.messages[3], "summary").unwrap().contains("Tried wrong approach"));
	assert_eq!(user_content_text(&ctx_branch.messages[4]).unwrap(), "better approach");
}

#[test]
fn build_context_uses_last_entry_when_leaf_id_not_found() {
	let entries = vec![bc_msg("1", None, "user", "hello"), bc_msg("2", Some("1"), "assistant", "hi")];
	let ctx = build_session_context(&entries, Leaf::Id("nonexistent"), None);
	assert_eq!(ctx.messages.len(), 2);
}

#[test]
fn build_context_handles_orphaned_entries_gracefully() {
	let entries = vec![
		bc_msg("1", None, "user", "hello"),
		bc_msg("2", Some("missing"), "assistant", "orphan"), // parent doesn't exist
	];
	let ctx = build_session_context(&entries, Leaf::Id("2"), None);
	// Should only get the orphan since parent chain is broken
	assert_eq!(ctx.messages.len(), 1);
}

// ===========================================================================
// Upstream test/session-manager/custom-session-id.test.ts
// ===========================================================================

#[test]
fn custom_id_uses_the_provided_id_instead_of_generating_one() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();
	session.new_session(Some(&NewSessionOptions { id: Some("my-custom-id".into()), ..Default::default() })).unwrap();
	assert_eq!(session.get_session_id(), "my-custom-id");
}

#[test]
fn custom_id_uses_the_provided_id_when_creating_an_in_memory_session() {
	let session = SessionManager::in_memory(
		None,
		Some(NewSessionOptions { id: Some("memory-session-id".into()), ..Default::default() }),
		None,
	)
	.unwrap();
	assert_eq!(session.get_session_id(), "memory-session-id");
	assert_eq!(session.get_header().unwrap().id, "memory-session-id");
	assert!(session.get_session_file().is_none());
}

#[test]
fn custom_id_allows_alphanumeric_session_ids_with_interior_punctuation() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();
	session
		.new_session(Some(&NewSessionOptions {
			id: Some("abc-123_def.456".into()),
			..Default::default()
		}))
		.unwrap();
	assert_eq!(session.get_session_id(), "abc-123_def.456");
}

#[test]
fn custom_id_rejects_invalid_custom_session_ids() {
	let invalid_ids = ["", "-abc", "abc-", "_abc", "abc_", ".abc", "abc.", "abc/def", "abc\\def", "abc def"];

	for id in invalid_ids {
		let mut session = SessionManager::in_memory(None, None, None).unwrap();
		let error = session
			.new_session(Some(&NewSessionOptions { id: Some(id.to_string()), ..Default::default() }))
			.unwrap_err();
		assert!(
			error.to_string().contains("Session id must be non-empty, contain only alphanumeric characters"),
			"unexpected error for {id:?}: {error}"
		);
	}
}

#[test]
fn custom_id_generates_uuidv7_when_no_id_is_provided() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();
	session.new_session(None).unwrap();
	let id = session.get_session_id().to_string();
	assert!(!id.is_empty());
	assert!(is_uuid_v7(&id), "not a uuidv7: {id}");
}

#[test]
fn custom_id_generates_uuidv7_when_options_is_provided_without_id() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();
	session
		.new_session(Some(&NewSessionOptions {
			parent_session: Some("parent.jsonl".into()),
			..Default::default()
		}))
		.unwrap();
	let id = session.get_session_id().to_string();
	assert!(!id.is_empty());
	assert!(is_uuid_v7(&id));
}

#[test]
fn custom_id_includes_the_custom_id_in_the_session_header() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();
	session
		.new_session(Some(&NewSessionOptions { id: Some("header-test-id".into()), ..Default::default() }))
		.unwrap();

	let header = session.get_header().expect("header");
	assert_eq!(header.id, "header-test-id");
}

#[test]
fn custom_id_generates_uuidv7_when_constructed_without_an_explicit_id() {
	let session = SessionManager::in_memory(None, None, None).unwrap();
	assert!(is_uuid_v7(session.get_session_id()));
	assert_eq!(session.get_header().unwrap().id, session.get_session_id());
}

#[test]
fn custom_id_uses_the_provided_id_when_creating_a_persisted_session() {
	let temp = temp_dir("custom-id-created");
	let session = SessionManager::create(
		&temp,
		Some(&temp),
		Some(NewSessionOptions { id: Some("created-session-id".into()), ..Default::default() }),
	)
	.unwrap();

	assert_eq!(session.get_session_id(), "created-session-id");
	assert_eq!(session.get_header().unwrap().id, "created-session-id");
	let session_file = session.get_session_file().unwrap().to_string();
	assert!(session_file.contains("created-session-id"));
	let basename = std::path::Path::new(&session_file).file_name().unwrap().to_string_lossy();
	let (prefix, suffix) = basename.split_once('_').expect("filename separator");
	assert!(is_file_timestamp(prefix), "bad filename prefix: {prefix}");
	assert_eq!(suffix, "created-session-id.jsonl");
	assert!(!std::path::Path::new(&session_file).exists());
	cleanup_dir(&temp);
}

#[test]
fn custom_id_generates_uuidv7_when_creating_a_branched_session() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();
	let first_id = session
		.append_message(AgentMessage::user(pi_ai::types::UserMessage {
			content: pi_ai::types::UserContentField::Blocks(vec![pi_ai::types::UserContent::Text(
				pi_ai::types::TextContent::new("hello"),
			)]),
			timestamp: crate::utils::time::now_ms(),
		}))
		.unwrap();

	session.create_branched_session(&first_id).unwrap();

	assert!(is_uuid_v7(session.get_session_id()));
	assert_eq!(session.get_header().unwrap().id, session.get_session_id());
}

fn write_fork_source(path: &str, lines: &[serde_json::Value]) {
	let content = lines
		.iter()
		.map(|line| serde_json::to_string(line).unwrap())
		.collect::<Vec<_>>()
		.join("\n")
		+ "\n";
	std::fs::write(path, content).unwrap();
}

fn fork_source_message_entry(text: &str) -> serde_json::Value {
	json!({
		"type": "message",
		"id": "entry-1",
		"parentId": null,
		"timestamp": "2026-01-01T00:00:00.000Z",
		"message": {
			"role": "assistant",
			"content": [{ "type": "text", "text": text }],
			"api": "openai-responses",
			"provider": "openai",
			"model": "gpt-5.4",
			"usage": {
				"input": 0,
				"output": 0,
				"cacheRead": 0,
				"cacheWrite": 0,
				"totalTokens": 0,
				"cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 }
			},
			"stopReason": "stop",
			"timestamp": 1_760_000_000_000i64
		}
	})
}

#[test]
fn custom_id_generates_uuidv7_when_forking_from_another_session_file() {
	let temp = temp_dir("custom-id-fork-v7");
	let source_path = format!("{temp}/source.jsonl");
	write_fork_source(
		&source_path,
		&[
			json!({
				"type": "session",
				"version": 3,
				"id": "legacy-session-id",
				"timestamp": "2026-01-01T00:00:00.000Z",
				"cwd": temp
			}),
			fork_source_message_entry("hello"),
		],
	);

	let forked = SessionManager::fork_from(&source_path, &temp, Some(&temp), None).unwrap();
	let header = forked.get_header().expect("header");
	assert!(is_uuid_v7(&header.id));
	assert_eq!(header.parent_session.as_deref(), Some(source_path.as_str()));
	cleanup_dir(&temp);
}

#[test]
fn custom_id_uses_the_provided_id_when_forking_from_another_session_file() {
	let temp = temp_dir("custom-id-fork-id");
	let source_path = format!("{temp}/source.jsonl");
	write_fork_source(
		&source_path,
		&[json!({
			"type": "session",
			"version": 3,
			"id": "source-session-id",
			"timestamp": "2026-01-01T00:00:00.000Z",
			"cwd": temp
		})],
	);

	let forked = SessionManager::fork_from(
		&source_path,
		&temp,
		Some(&temp),
		Some(NewSessionOptions { id: Some("forked-session-id".into()), ..Default::default() }),
	)
	.unwrap();
	let header = forked.get_header().expect("header");
	assert_eq!(header.id, "forked-session-id");
	assert_eq!(header.parent_session.as_deref(), Some(source_path.as_str()));
	let session_file = forked.get_session_file().unwrap().to_string();
	assert!(session_file.contains("forked-session-id"));
	let basename = std::path::Path::new(&session_file).file_name().unwrap().to_string_lossy();
	let (prefix, suffix) = basename.split_once('_').expect("filename separator");
	assert!(is_file_timestamp(prefix));
	assert_eq!(suffix, "forked-session-id.jsonl");
	cleanup_dir(&temp);
}

// ===========================================================================
// Upstream test/session-manager/file-operations.test.ts
// ===========================================================================

const HEADER_SCAN_LIMIT_BYTES: usize = 1024 * 1024;

fn write_session_header(file: &str, cwd: &str, id: &str, prefix: &str) {
	let header = json!({ "type": "session", "version": 3, "id": id, "timestamp": "2025-01-01T00:00:00Z", "cwd": cwd });
	std::fs::write(file, format!("{prefix}{}\n", serde_json::to_string(&header).unwrap())).unwrap();
}

#[test]
fn load_entries_returns_empty_array_for_non_existent_file() {
	let temp = temp_dir("load-nonexistent");
	let entries = load_entries_from_file(&format!("{temp}/nonexistent.jsonl")).unwrap();
	assert!(entries.is_empty());
	cleanup_dir(&temp);
}

#[test]
fn load_entries_returns_empty_array_for_empty_file() {
	let temp = temp_dir("load-empty");
	let file = format!("{temp}/empty.jsonl");
	std::fs::write(&file, "").unwrap();
	assert!(load_entries_from_file(&file).unwrap().is_empty());
	cleanup_dir(&temp);
}

#[test]
fn load_entries_returns_empty_array_for_file_without_valid_session_header() {
	let temp = temp_dir("load-no-header");
	let file = format!("{temp}/no-header.jsonl");
	std::fs::write(&file, "{\"type\":\"message\",\"id\":\"1\"}\n").unwrap();
	assert!(load_entries_from_file(&file).unwrap().is_empty());
	cleanup_dir(&temp);
}

#[test]
fn load_entries_returns_empty_array_for_malformed_json() {
	let temp = temp_dir("load-malformed");
	let file = format!("{temp}/malformed.jsonl");
	std::fs::write(&file, "not json\n").unwrap();
	assert!(load_entries_from_file(&file).unwrap().is_empty());
	cleanup_dir(&temp);
}

#[test]
fn load_entries_loads_valid_session_file() {
	let temp = temp_dir("load-valid");
	let file = format!("{temp}/valid.jsonl");
	std::fs::write(
		&file,
		"{\"type\":\"session\",\"id\":\"abc\",\"timestamp\":\"2025-01-01T00:00:00Z\",\"cwd\":\"/tmp\"}\n\
		 {\"type\":\"message\",\"id\":\"1\",\"parentId\":null,\"timestamp\":\"2025-01-01T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":1}}\n",
	)
	.unwrap();
	let entries = load_entries_from_file(&file).unwrap();
	assert_eq!(entries.len(), 2);
	assert!(entries[0].as_session_header().is_some());
	assert!(matches!(entries[1].as_entry(), Some(SessionEntry::Message(_))));
	cleanup_dir(&temp);
}

#[test]
fn load_entries_skips_malformed_lines_but_keeps_valid_ones() {
	let temp = temp_dir("load-mixed");
	let file = format!("{temp}/mixed.jsonl");
	std::fs::write(
		&file,
		"{\"type\":\"session\",\"id\":\"abc\",\"timestamp\":\"2025-01-01T00:00:00Z\",\"cwd\":\"/tmp\"}\n\
		 not valid json\n\
		 {\"type\":\"message\",\"id\":\"1\",\"parentId\":null,\"timestamp\":\"2025-01-01T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":1}}\n",
	)
	.unwrap();
	let entries = load_entries_from_file(&file).unwrap();
	assert_eq!(entries.len(), 2);
	cleanup_dir(&temp);
}

#[test]
fn load_entries_adds_a_newline_after_an_unterminated_valid_record() {
	let temp = temp_dir("load-unterminated");
	let file = format!("{temp}/unterminated.jsonl");
	let content =
		"{\"type\":\"session\",\"id\":\"abc\",\"timestamp\":\"2025-01-01T00:00:00Z\",\"cwd\":\"/tmp\"}\n\
		 {\"type\":\"message\",\"id\":\"1\",\"parentId\":null,\"timestamp\":\"2025-01-01T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":1}}";
	std::fs::write(&file, content).unwrap();

	assert_eq!(load_entries_from_file(&file).unwrap().len(), 2);
	assert_eq!(std::fs::read_to_string(&file).unwrap(), format!("{content}\n"));
	cleanup_dir(&temp);
}

#[test]
fn load_entries_adds_a_newline_after_an_unterminated_malformed_final_fragment() {
	let temp = temp_dir("load-malformed-tail");
	let file = format!("{temp}/malformed-tail.jsonl");
	let content = "{\"type\":\"session\",\"id\":\"abc\",\"timestamp\":\"2025-01-01T00:00:00Z\",\"cwd\":\"/tmp\"}\n\
		 {\"type\":\"message\"";
	std::fs::write(&file, content).unwrap();

	assert_eq!(load_entries_from_file(&file).unwrap().len(), 1);
	assert_eq!(std::fs::read_to_string(&file).unwrap(), format!("{content}\n"));
	cleanup_dir(&temp);
}

#[test]
fn load_entries_does_not_modify_an_unterminated_non_session_file() {
	let temp = temp_dir("load-invalid-unterminated");
	let file = format!("{temp}/invalid.jsonl");
	let content = "{\"type\":\"message\",\"id\":\"1\"}";
	std::fs::write(&file, content).unwrap();

	assert!(load_entries_from_file(&file).unwrap().is_empty());
	assert_eq!(std::fs::read_to_string(&file).unwrap(), content);
	cleanup_dir(&temp);
}

#[test]
fn load_entries_reads_cwd_from_a_session_with_leading_blank_lines() {
	let temp = temp_dir("load-leading-blank");
	let file = format!("{temp}/header.jsonl");
	let stored_cwd = format!("{temp}/stored-project");
	write_session_header(&file, &stored_cwd, "leading-blank", "\n  \n");

	let session_manager = SessionManager::open(&file, Some(&temp), None).unwrap();
	assert_eq!(session_manager.get_session_id(), "leading-blank");
	assert_eq!(session_manager.get_cwd(), stored_cwd);
	cleanup_dir(&temp);
}

#[test]
fn load_entries_reads_cwd_from_a_session_with_leading_malformed_lines() {
	let temp = temp_dir("load-leading-malformed");
	let file = format!("{temp}/header.jsonl");
	let stored_cwd = format!("{temp}/stored-project");
	write_session_header(&file, &stored_cwd, "leading-malformed", "not json\n{broken json\n");

	let session_manager = SessionManager::open(&file, Some(&temp), None).unwrap();
	assert_eq!(session_manager.get_session_id(), "leading-malformed");
	assert_eq!(session_manager.get_cwd(), stored_cwd);
	cleanup_dir(&temp);
}

#[test]
fn load_entries_reads_cwd_from_a_session_with_a_multi_buffer_header() {
	let temp = temp_dir("load-multi-buffer");
	let file = format!("{temp}/header.jsonl");
	let stored_cwd = format!("{temp}/stored-project");
	write_session_header(&file, &stored_cwd, &"a".repeat(8192), "");

	let session_manager = SessionManager::open(&file, Some(&temp), None).unwrap();
	assert_eq!(session_manager.get_session_id(), "a".repeat(8192));
	assert_eq!(session_manager.get_cwd(), stored_cwd);
	cleanup_dir(&temp);
}

#[test]
fn load_entries_opens_compatible_sessions_beyond_the_discovery_scan_limit() {
	let temp = temp_dir("load-scan-limit");
	let stored_cwd = format!("{temp}/stored-project");
	let override_cwd = format!("{temp}/override-project");
	let cases = [
		("large-header", "a".repeat(HEADER_SCAN_LIMIT_BYTES + 1), ""),
		("large-prefix", "large-prefix".to_string(), &format!("{}\n", "x".repeat(HEADER_SCAN_LIMIT_BYTES + 1))),
	];

	for (name, id, prefix) in cases {
		let file = format!("{temp}/{name}.jsonl");
		write_session_header(&file, &stored_cwd, &id, prefix);
		for cwd_override in [None, Some(override_cwd.as_str())] {
			let session_manager = SessionManager::open(&file, Some(&temp), cwd_override).unwrap();
			assert_eq!(session_manager.get_session_id(), id);
			assert_eq!(session_manager.get_cwd(), cwd_override.unwrap_or(stored_cwd.as_str()));
		}
	}
	cleanup_dir(&temp);
}

#[test]
fn load_entries_opens_session_files_larger_than_node_max_string_length() {
	let temp = temp_dir("load-huge");
	let file = format!("{temp}/large.jsonl");
	std::fs::write(
		&file,
		"{\"type\":\"session\",\"version\":3,\"id\":\"abc\",\"timestamp\":\"2025-01-01T00:00:00Z\",\"cwd\":\"/tmp\"}\n",
	)
	.unwrap();

	// Upstream writes newline sparsely up to Node's MAX_STRING_LENGTH
	// (~512 Mi); the port writes the same 512 Mi span.
	{
		let mut file_handle = std::fs::OpenOptions::new().write(true).open(&file).unwrap();
		let newline = b"\n";
		let stride: u64 = 16 * 1024 * 1024;
		let max_string_length: u64 = 512 * 1024 * 1024 - 24;
		let mut offset = stride;
		while offset <= max_string_length + stride {
			file_handle.seek(std::io::SeekFrom::Start(offset)).unwrap();
			file_handle.write_all(newline).unwrap();
			offset += stride;
		}
	}

	{
		let mut append = std::fs::OpenOptions::new().append(true).open(&file).unwrap();
		append
			.write_all(
				b"{\"type\":\"message\",\"id\":\"1\",\"parentId\":null,\"timestamp\":\"2025-01-01T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":1}}\n",
			)
			.unwrap();
	}

	let session_manager = SessionManager::open(&file, Some(&temp), None).unwrap();
	assert_eq!(session_manager.get_session_id(), "abc");
	assert_eq!(session_manager.get_entries().len(), 1);
	assert_eq!(
		session_manager.build_session_context().messages,
		vec![AgentMessage::user(pi_ai::types::UserMessage {
			content: pi_ai::types::UserContentField::Text("hi".to_string()),
			timestamp: 1,
		})]
	);
	cleanup_dir(&temp);
}

fn find_recent_header_line(id: &str, cwd: &str) -> String {
	format!(
		"{}\n",
		serde_json::to_string(&json!({ "type": "session", "id": id, "timestamp": "2025-01-01T00:00:00Z", "cwd": cwd }))
			.unwrap()
	)
}

#[test]
fn find_most_recent_returns_null_for_empty_directory() {
	let temp = temp_dir("recent-empty");
	assert_eq!(find_most_recent_session(&temp, None), None);
	cleanup_dir(&temp);
}

#[test]
fn find_most_recent_returns_null_for_non_existent_directory() {
	let temp = temp_dir("recent-nonexistent");
	assert_eq!(find_most_recent_session(&format!("{temp}/nonexistent"), None), None);
	cleanup_dir(&temp);
}

#[test]
fn find_most_recent_ignores_non_jsonl_files() {
	let temp = temp_dir("recent-non-jsonl");
	std::fs::write(format!("{temp}/file.txt"), "hello").unwrap();
	std::fs::write(format!("{temp}/file.json"), "{}").unwrap();
	assert_eq!(find_most_recent_session(&temp, None), None);
	cleanup_dir(&temp);
}

#[test]
fn find_most_recent_ignores_jsonl_files_without_valid_session_header() {
	let temp = temp_dir("recent-invalid");
	std::fs::write(format!("{temp}/invalid.jsonl"), "{\"type\":\"message\"}\n").unwrap();
	assert_eq!(find_most_recent_session(&temp, None), None);
	cleanup_dir(&temp);
}

#[test]
fn find_most_recent_returns_single_valid_session_file() {
	let temp = temp_dir("recent-single");
	let file = format!("{temp}/session.jsonl");
	std::fs::write(&file, find_recent_header_line("abc", "/tmp")).unwrap();
	assert_eq!(find_most_recent_session(&temp, None), Some(file));
	cleanup_dir(&temp);
}

#[test]
fn find_most_recent_returns_most_recently_modified_session() {
	let temp = temp_dir("recent-modified");
	let file1 = format!("{temp}/older.jsonl");
	let file2 = format!("{temp}/newer.jsonl");

	std::fs::write(&file1, find_recent_header_line("old", "/tmp")).unwrap();
	// Small delay to ensure different mtime
	std::thread::sleep(std::time::Duration::from_millis(20));
	std::fs::write(&file2, find_recent_header_line("new", "/tmp")).unwrap();

	assert_eq!(find_most_recent_session(&temp, None), Some(file2));
	cleanup_dir(&temp);
}

#[test]
fn find_most_recent_skips_invalid_files_and_returns_valid_one() {
	let temp = temp_dir("recent-skip-invalid");
	let invalid = format!("{temp}/invalid.jsonl");
	let valid = format!("{temp}/valid.jsonl");

	std::fs::write(&invalid, "{\"type\":\"not-session\"}\n").unwrap();
	std::thread::sleep(std::time::Duration::from_millis(20));
	std::fs::write(&valid, find_recent_header_line("abc", "/tmp")).unwrap();

	assert_eq!(find_most_recent_session(&temp, None), Some(valid));
	cleanup_dir(&temp);
}

#[test]
fn find_most_recent_skips_oversized_corrupt_files_and_returns_a_valid_session() {
	let temp = temp_dir("recent-oversized");
	let invalid = format!("{temp}/oversized.jsonl");
	let valid = format!("{temp}/valid.jsonl");
	std::fs::write(&invalid, "x".repeat(HEADER_SCAN_LIMIT_BYTES + 1)).unwrap();
	std::fs::write(&valid, find_recent_header_line("abc", "/tmp")).unwrap();

	assert_eq!(find_most_recent_session(&temp, None), Some(valid));
	cleanup_dir(&temp);
}

#[test]
fn find_most_recent_filters_most_recent_session_by_cwd() {
	let temp = temp_dir("recent-cwd");
	let project_a = format!("{temp}/project-a");
	let project_b = format!("{temp}/project-b");
	let file_a = format!("{temp}/a.jsonl");
	let file_b = format!("{temp}/b.jsonl");

	std::fs::write(&file_a, find_recent_header_line("a", &project_a)).unwrap();
	std::thread::sleep(std::time::Duration::from_millis(20));
	std::fs::write(&file_b, find_recent_header_line("b", &project_b)).unwrap();

	assert_eq!(find_most_recent_session(&temp, Some(&project_a)), Some(file_a));
	assert_eq!(find_most_recent_session(&temp, Some(&project_b)), Some(file_b));
	cleanup_dir(&temp);
}

fn create_persisted_session(cwd: &str, label: &str, session_dir: &str) -> String {
	let mut session = SessionManager::create(cwd, Some(session_dir), None).unwrap();
	session.append_message(AgentMessage::user(pi_ai::types::UserMessage {
		content: pi_ai::types::UserContentField::Text(label.to_string()),
		timestamp: crate::utils::time::now_ms(),
	})).unwrap();
	session.append_message(AgentMessage::assistant(assistant_message(
		&format!("reply to {label}"),
		"anthropic",
		"test",
		test_usage(1, 1, 0, 0, 2),
		crate::utils::time::now_ms(),
	)))
	.unwrap();
	session.get_session_file().expect("Expected persisted session file").to_string()
}

#[tokio::test]
async fn flat_session_dir_scopes_current_folder_apis_by_cwd_while_listing_all_flat_sessions() {
	let temp = temp_dir("flat-scope");
	let project_a = format!("{temp}/project-a");
	let project_b = format!("{temp}/project-b");
	std::fs::create_dir_all(&project_a).unwrap();
	std::fs::create_dir_all(&project_b).unwrap();

	let session_a = create_persisted_session(&project_a, "from A", &temp);
	std::thread::sleep(std::time::Duration::from_millis(20));
	let session_b = create_persisted_session(&project_b, "from B", &temp);

	let current_a = SessionManager::list(&project_a, Some(&temp), None, None).await.unwrap();
	let paths: Vec<String> = current_a.iter().map(|session| session.path.clone()).collect();
	assert_eq!(paths, [session_a.clone()]);

	let all = SessionManager::list_all_in_dir(&temp, None, None).await.unwrap();
	let mut all_paths: Vec<String> = all.iter().map(|session| session.path.clone()).collect();
	all_paths.sort();
	let mut expected = vec![session_a.clone(), session_b.clone()];
	expected.sort();
	assert_eq!(all_paths, expected);

	let continued_a = SessionManager::continue_recent(&project_a, Some(&temp)).unwrap();
	assert_eq!(continued_a.get_session_file(), Some(session_a.as_str()));
	cleanup_dir(&temp);
}

#[tokio::test]
async fn flat_session_dir_rejects_a_cancelled_session_listing() {
	let temp = temp_dir("flat-abort");
	let project_a = format!("{temp}/project-a");
	let project_b = format!("{temp}/project-b");
	std::fs::create_dir_all(&project_a).unwrap();
	std::fs::create_dir_all(&project_b).unwrap();
	create_persisted_session(&project_a, "from A", &temp);
	create_persisted_session(&project_b, "from B", &temp);

	let signal = pi_ai::utils::abort::AbortSignal::new();
	let abort_signal = signal.clone();
	let progress: SessionListProgress = Arc::new(move |_loaded, _total, partial_sessions| {
		if partial_sessions.is_some() {
			pi_ai::utils::abort::AbortSignal::abort(&abort_signal);
		}
	});

	let listing = SessionManager::list_all_in_dir(&temp, Some(progress), Some(signal.clone())).await;
	assert!(matches!(listing, Err(SessionListError::Aborted(_))), "expected abort, got {listing:?}");

	let listing = SessionManager::list_all(None, Some(signal)).await;
	assert!(matches!(listing, Err(SessionListError::Aborted(_))));
	cleanup_dir(&temp);
}

#[test]
fn corrupted_files_truncates_and_rewrites_empty_file_with_valid_header() {
	let temp = temp_dir("corrupt-empty");
	let empty_file = format!("{temp}/empty.jsonl");
	std::fs::write(&empty_file, "").unwrap();

	let sm = SessionManager::open(&empty_file, Some(&temp), None).unwrap();

	// Should have created a new session with valid header
	assert!(!sm.get_session_id().is_empty());
	assert!(sm.get_header().is_some());
	assert_eq!(sm.get_header().unwrap().header_type, SessionHeaderType::Session);

	// File should now contain a valid header
	let content = std::fs::read_to_string(&empty_file).unwrap();
	let lines: Vec<&str> = content.trim().split('\n').filter(|line| !line.is_empty()).collect();
	assert_eq!(lines.len(), 1);
	let header: FileEntry = serde_json::from_str(lines[0]).unwrap();
	assert_eq!(header.as_session_header().unwrap().header_type, SessionHeaderType::Session);
	assert_eq!(header.as_session_header().unwrap().id, sm.get_session_id());
	cleanup_dir(&temp);
}

#[test]
fn corrupted_files_throws_and_preserves_non_empty_file_without_valid_header() {
	let temp = temp_dir("corrupt-no-header");
	let no_header_file = format!("{temp}/no-header.jsonl");
	let original_content = "{\"type\":\"message\",\"id\":\"abc\",\"parentId\":\"orphaned\",\"timestamp\":\"2025-01-01T00:00:00Z\",\"message\":{\"role\":\"assistant\",\"content\":\"test\"}}\n";
	std::fs::write(&no_header_file, original_content).unwrap();

	let error = expect_open_error(&no_header_file, &temp);
	assert_eq!(
		error.to_string(),
		format!("Session file is not a valid pi session: {no_header_file}")
	);
	assert_eq!(std::fs::read_to_string(&no_header_file).unwrap(), original_content);
	cleanup_dir(&temp);
}

#[test]
fn corrupted_files_throws_and_preserves_non_session_jsonl_files() {
	let temp = temp_dir("corrupt-non-session");
	let non_session_file = format!("{temp}/not-a-session.log");
	let original_content = "{\"type\":\"event\",\"data\":\"not a session\"}\n";
	std::fs::write(&non_session_file, original_content).unwrap();

	let error = expect_open_error(&non_session_file, &temp);
	assert_eq!(error.to_string(), format!("Session file is not a valid pi session: {non_session_file}"));
	assert_eq!(std::fs::read_to_string(&non_session_file).unwrap(), original_content);
	cleanup_dir(&temp);
}

#[test]
fn corrupted_files_preserves_explicit_session_file_path_when_recovering_from_corrupted_file() {
	let temp = temp_dir("corrupt-explicit");
	let explicit_path = format!("{temp}/my-session.jsonl");
	std::fs::write(&explicit_path, "").unwrap();

	let sm = SessionManager::open(&explicit_path, Some(&temp), None).unwrap();

	// The session file path should be preserved
	assert_eq!(sm.get_session_file(), Some(explicit_path.as_str()));
	cleanup_dir(&temp);
}

#[test]
fn corrupted_files_subsequent_loads_of_initialized_empty_file_work_correctly() {
	let temp = temp_dir("corrupt-reopen");
	let empty_file = format!("{temp}/empty.jsonl");
	std::fs::write(&empty_file, "").unwrap();

	let sm1 = SessionManager::open(&empty_file, Some(&temp), None).unwrap();
	let session_id = sm1.get_session_id().to_string();

	let sm2 = SessionManager::open(&empty_file, Some(&temp), None).unwrap();
	assert_eq!(sm2.get_session_id(), session_id);
	assert_eq!(
		sm2.get_header().map(|header| header.header_type),
		Some(SessionHeaderType::Session)
	);
	cleanup_dir(&temp);
}

#[test]
fn file_creation_does_not_create_a_file_for_a_session_with_only_setup_entries() {
	let temp = temp_dir("creation-setup-only");
	let mut session = SessionManager::create(&temp, Some(&temp), None).unwrap();
	session.append_model_change("anthropic", "claude-sonnet-4-5").unwrap();
	session.append_thinking_level_change("off").unwrap();

	assert!(!std::path::Path::new(session.get_session_file().unwrap()).exists());
	cleanup_dir(&temp);
}

#[test]
fn file_creation_creates_the_file_when_the_first_user_message_is_appended() {
	let temp = temp_dir("creation-first-user");
	let mut session = SessionManager::create(&temp, Some(&temp), None).unwrap();
	session.append_model_change("anthropic", "claude-sonnet-4-5").unwrap();
	session.append_message(user_msg("first question")).unwrap();

	let file = session.get_session_file().unwrap().to_string();
	assert_eq!(read_session_file_roles(&file), ["session", "model_change", "user"]);
	assert_eq!(SessionManager::open(&file, Some(&temp), None).unwrap().build_session_context().messages.len(), 1);
	cleanup_dir(&temp);
}

#[test]
fn file_creation_appends_later_entries_to_the_file_without_rewriting_earlier_ones() {
	let temp = temp_dir("creation-append");
	let mut session = SessionManager::create(&temp, Some(&temp), None).unwrap();
	session.append_message(user_msg("first question")).unwrap();
	session.append_custom_entry("preset-state", Some(json!({ "name": "plan" }))).unwrap();
	session.append_message(assistant_msg("first answer")).unwrap();

	assert_eq!(
		read_session_file_roles(session.get_session_file().unwrap()),
		["session", "user", "custom", "assistant"]
	);
	cleanup_dir(&temp);
}

// ===========================================================================
// Upstream test/session-manager/labels.test.ts
// ===========================================================================

#[test]
fn labels_sets_and_gets_labels() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let msg_id = session.append_message(user_msg("hello")).unwrap();

	// No label initially
	assert_eq!(session.get_label(&msg_id), None);

	// Set a label
	let label_id = session.append_label_change(&msg_id, Some("checkpoint".to_string())).unwrap();
	assert_eq!(session.get_label(&msg_id), Some("checkpoint".to_string()));

	// Label entry should be in entries
	let entries = session.get_entries();
	let Some(SessionEntry::Label(label_entry)) = entries.iter().find(|entry| entry.type_name() == "label") else {
		panic!("label entry missing");
	};
	assert_eq!(label_entry.base.id, label_id);
	assert_eq!(label_entry.target_id, msg_id);
	assert_eq!(label_entry.label.as_deref(), Some("checkpoint"));
}

#[test]
fn labels_clears_labels_with_undefined() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let msg_id = session.append_message(user_msg("hello")).unwrap();

	session.append_label_change(&msg_id, Some("checkpoint".to_string())).unwrap();
	assert_eq!(session.get_label(&msg_id), Some("checkpoint".to_string()));

	// Clear the label
	session.append_label_change(&msg_id, None).unwrap();
	assert_eq!(session.get_label(&msg_id), None);
}

#[test]
fn labels_last_label_wins() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let msg_id = session.append_message(user_msg("hello")).unwrap();

	session.append_label_change(&msg_id, Some("first".to_string())).unwrap();
	session.append_label_change(&msg_id, Some("second".to_string())).unwrap();
	let last_label_id = session.append_label_change(&msg_id, Some("third".to_string())).unwrap();

	assert_eq!(session.get_label(&msg_id), Some("third".to_string()));

	let entries = session.get_entries();
	let last_label_entry = entries.iter().find(|entry| entry.id() == last_label_id).unwrap();
	let tree = session.get_tree();
	let msg_node = tree.iter().find(|node| node.entry.id() == msg_id).unwrap();
	assert_eq!(msg_node.label_timestamp.as_deref(), Some(last_label_entry.timestamp()));
}

#[test]
fn labels_are_included_in_tree_nodes() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let msg1_id = session.append_message(user_msg("hello")).unwrap();
	let msg2_id = session.append_message(assistant_msg("hi")).unwrap();

	let msg1_label_id = session.append_label_change(&msg1_id, Some("start".to_string())).unwrap();
	let msg2_label_id = session.append_label_change(&msg2_id, Some("response".to_string())).unwrap();

	let entries = session.get_entries();
	let msg1_label_entry = entries.iter().find(|entry| entry.id() == msg1_label_id).unwrap();
	let msg2_label_entry = entries.iter().find(|entry| entry.id() == msg2_label_id).unwrap();
	let tree = session.get_tree();

	// Find the message nodes (skip label entries)
	let msg1_node = tree.iter().find(|node| node.entry.id() == msg1_id).unwrap();
	assert_eq!(msg1_node.label.as_deref(), Some("start"));
	assert_eq!(msg1_node.label_timestamp.as_deref(), Some(msg1_label_entry.timestamp()));

	// msg2 is a child of msg1
	let msg2_node = msg1_node.children.iter().find(|node| node.entry.id() == msg2_id).unwrap();
	assert_eq!(msg2_node.label.as_deref(), Some("response"));
	assert_eq!(msg2_node.label_timestamp.as_deref(), Some(msg2_label_entry.timestamp()));
}

#[test]
fn labels_are_preserved_in_create_branched_session() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let msg1_id = session.append_message(user_msg("hello")).unwrap();
	let msg2_id = session.append_message(assistant_msg("hi")).unwrap();

	let msg1_label_id = session.append_label_change(&msg1_id, Some("important".to_string())).unwrap();
	let msg2_label_id = session.append_label_change(&msg2_id, Some("also-important".to_string())).unwrap();
	let original_entries = session.get_entries();
	let msg1_label_entry = original_entries.iter().find(|entry| entry.id() == msg1_label_id).unwrap();
	let msg2_label_entry = original_entries.iter().find(|entry| entry.id() == msg2_label_id).unwrap();

	// Branch from msg2 (in-memory mode returns None, but updates internal state)
	session.create_branched_session(&msg2_id).unwrap();

	// Labels should be preserved
	assert_eq!(session.get_label(&msg1_id), Some("important".to_string()));
	assert_eq!(session.get_label(&msg2_id), Some("also-important".to_string()));

	// New label entries should exist
	let entries = session.get_entries();
	let label_entries: Vec<&SessionEntry> =
		entries.iter().filter(|entry| entry.type_name() == "label").collect();
	assert_eq!(label_entries.len(), 2);

	let tree = session.get_tree();
	let msg1_node = tree.iter().find(|node| node.entry.id() == msg1_id).unwrap();
	let msg2_node = msg1_node.children.iter().find(|node| node.entry.id() == msg2_id).unwrap();
	assert_eq!(msg1_node.label_timestamp.as_deref(), Some(msg1_label_entry.timestamp()));
	assert_eq!(msg2_node.label_timestamp.as_deref(), Some(msg2_label_entry.timestamp()));
}

#[test]
fn labels_rewires_children_of_removed_labels_when_forking() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let msg1_id = session.append_message(user_msg("hello")).unwrap();
	session.append_label_change(&msg1_id, Some("checkpoint".to_string())).unwrap();
	let model_change_id = session.append_model_change("anthropic", "claude-test").unwrap();
	let msg2_id = session.append_message(user_msg("followup")).unwrap();

	session.create_branched_session(&msg2_id).unwrap();

	assert_eq!(session.get_entry(&model_change_id).unwrap().parent_id(), Some(msg1_id.as_str()));
}

#[test]
fn labels_not_on_path_are_not_preserved_in_create_branched_session() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let msg1_id = session.append_message(user_msg("hello")).unwrap();
	let msg2_id = session.append_message(assistant_msg("hi")).unwrap();
	let msg3_id = session.append_message(user_msg("followup")).unwrap();

	// Label all messages
	session.append_label_change(&msg1_id, Some("first".to_string())).unwrap();
	session.append_label_change(&msg2_id, Some("second".to_string())).unwrap();
	session.append_label_change(&msg3_id, Some("third".to_string())).unwrap();

	// Branch from msg2 (excludes msg3)
	session.create_branched_session(&msg2_id).unwrap();

	// Only labels for msg1 and msg2 should be preserved
	assert_eq!(session.get_label(&msg1_id), Some("first".to_string()));
	assert_eq!(session.get_label(&msg2_id), Some("second".to_string()));
	assert_eq!(session.get_label(&msg3_id), None);
}

#[test]
fn labels_are_not_included_in_build_session_context() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let msg_id = session.append_message(user_msg("hello")).unwrap();
	session.append_label_change(&msg_id, Some("checkpoint".to_string())).unwrap();

	let ctx = session.build_session_context();
	assert_eq!(ctx.messages.len(), 1);
	assert_eq!(message_role(&ctx.messages[0]), "user");
}

#[test]
fn labels_throws_when_labeling_non_existent_entry() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let error = session.append_label_change("non-existent", Some("label".to_string())).unwrap_err();
	assert_eq!(error.to_string(), "Entry non-existent not found");
}

// ===========================================================================
// Upstream test/session-manager/load-entries.test.ts
// ===========================================================================

fn load_entries_user_message(text: &str) -> AgentMessage {
	AgentMessage::user(pi_ai::types::UserMessage {
		content: pi_ai::types::UserContentField::Blocks(vec![pi_ai::types::UserContent::Text(
			pi_ai::types::TextContent::new(text),
		)]),
		timestamp: crate::utils::time::now_ms(),
	})
}

fn stored_entries(build: impl FnOnce(&mut SessionManager)) -> Vec<SessionEntry> {
	let mut source = SessionManager::in_memory(Some("/project"), None, None).unwrap();
	build(&mut source);
	source.get_entries()
}

fn as_file_entries(entries: Vec<SessionEntry>) -> Vec<FileEntry> {
	entries.into_iter().map(FileEntry::Entry).collect()
}

fn expect_open_error(path: &str, session_dir: &str) -> SessionManagerError {
	let Err(error) = SessionManager::open(path, Some(session_dir), None) else {
		panic!("expected open({path}) to fail");
	};
	error
}

#[test]
fn preloaded_adopts_entries_verbatim() {
	let entries = stored_entries(|source| {
		source.append_message(load_entries_user_message("hello")).unwrap();
		source.append_model_change("anthropic", "claude-opus-4-5").unwrap();
		source.append_message(load_entries_user_message("again")).unwrap();
	});

	let session = SessionManager::in_memory(Some("/project"), None, Some(as_file_entries(entries.clone()))).unwrap();

	assert_eq!(session.get_entries(), entries);
}

#[test]
fn preloaded_keeps_the_loaded_leaf_so_appends_continue_the_conversation() {
	let entries = stored_entries(|source| {
		source.append_message(load_entries_user_message("hello")).unwrap();
		source.append_message(load_entries_user_message("again")).unwrap();
	});
	let last_id = entries.last().unwrap().id().to_string();

	let mut session = SessionManager::in_memory(Some("/project"), None, Some(as_file_entries(entries))).unwrap();
	let appended_id = session.append_message(load_entries_user_message("continued")).unwrap();

	assert_eq!(session.get_leaf_id(), Some(appended_id.as_str()));
	assert_eq!(session.get_entry(&appended_id).unwrap().parent_id(), Some(last_id.as_str()));
}

#[test]
fn preloaded_never_mints_an_id_that_collides_with_a_loaded_entry() {
	let entries = stored_entries(|source| {
		for i in 0..50 {
			source.append_message(load_entries_user_message(&format!("message {i}"))).unwrap();
		}
	});

	let mut session = SessionManager::in_memory(Some("/project"), None, Some(as_file_entries(entries.clone()))).unwrap();
	let appended_id = session.append_message(load_entries_user_message("continued")).unwrap();

	assert!(!entries.iter().any(|entry| entry.id() == appended_id));
}

#[test]
fn preloaded_rebuilds_the_branch_structure_rather_than_a_flat_chain() {
	let entries = stored_entries(|source| {
		let first_id = source.append_message(load_entries_user_message("hello")).unwrap();
		source.append_message(load_entries_user_message("abandoned")).unwrap();
		source.branch(&first_id).unwrap();
		source.append_message(load_entries_user_message("kept")).unwrap();
	});

	let session = SessionManager::in_memory(Some("/project"), None, Some(as_file_entries(entries))).unwrap();
	let roots = session.get_tree();

	assert_eq!(roots.len(), 1);
	assert_eq!(roots[0].children.len(), 2);
}

#[test]
fn preloaded_rebuilds_labels() {
	let labelled_id = std::cell::RefCell::new(String::new());
	let entries = stored_entries(|source| {
		let id = source.append_message(load_entries_user_message("hello")).unwrap();
		source.append_label_change(&id, Some("checkpoint".to_string())).unwrap();
		*labelled_id.borrow_mut() = id;
	});

	let session = SessionManager::in_memory(Some("/project"), None, Some(as_file_entries(entries))).unwrap();

	assert_eq!(session.get_label(&labelled_id.borrow()), Some("checkpoint".to_string()));
}

#[test]
fn preloaded_resolves_a_compaction_against_the_entry_it_was_written_against() {
	let kept_id = std::cell::RefCell::new(String::new());
	let entries = stored_entries(|source| {
		source.append_message(load_entries_user_message("dropped")).unwrap();
		let id = source.append_message(load_entries_user_message("kept")).unwrap();
		source.append_compaction("summary so far", Some(&id), 1000, None, None, None).unwrap();
		*kept_id.borrow_mut() = id;
	});

	let session = SessionManager::in_memory(Some("/project"), None, Some(as_file_entries(entries))).unwrap();
	let context = session.build_context_entries();

	assert!(context.iter().any(|entry| entry.id() == kept_id.borrow().as_str()));
}

#[test]
fn preloaded_creates_a_header_from_the_options_when_the_entries_carry_none() {
	let entries = stored_entries(|source| {
		source.append_message(load_entries_user_message("hello")).unwrap();
	});

	let session = SessionManager::in_memory(
		Some("/project"),
		Some(NewSessionOptions { id: Some("restored-session".into()), ..Default::default() }),
		Some(as_file_entries(entries)),
	)
	.unwrap();

	assert_eq!(session.get_session_id(), "restored-session");
	assert_eq!(session.get_header().unwrap().id, "restored-session");
	assert_eq!(session.get_header().unwrap().cwd, "/project");
}

#[test]
fn preloaded_generates_a_session_id_when_the_options_carry_none() {
	let entries = stored_entries(|source| {
		source.append_message(load_entries_user_message("hello")).unwrap();
	});

	let session = SessionManager::in_memory(Some("/project"), None, Some(as_file_entries(entries))).unwrap();

	assert!(is_uuid_v7(session.get_session_id()));
	assert_eq!(session.get_header().unwrap().id, session.get_session_id());
}

#[test]
fn preloaded_stays_off_the_filesystem() {
	let entries = stored_entries(|source| {
		source.append_message(load_entries_user_message("hello")).unwrap();
	});

	let mut session = SessionManager::in_memory(Some("/project"), None, Some(as_file_entries(entries))).unwrap();
	session.append_message(load_entries_user_message("continued")).unwrap();

	assert!(session.get_session_file().is_none());
	assert!(!session.is_persisted());
}

#[test]
fn preloaded_starts_an_empty_session_when_the_entries_are_empty() {
	let session = SessionManager::in_memory(
		Some("/project"),
		Some(NewSessionOptions { id: Some("empty-session".into()), ..Default::default() }),
		Some(Vec::new()),
	)
	.unwrap();

	assert_eq!(session.get_session_id(), "empty-session");
	assert!(session.get_entries().is_empty());
	assert_eq!(session.get_leaf_id(), None);
}

#[test]
fn preloaded_takes_the_session_identity_from_a_header_among_the_entries() {
	let body = stored_entries(|source| {
		source.append_message(load_entries_user_message("hello")).unwrap();
	});
	let entries: Vec<FileEntry> = std::iter::once(FileEntry::Header(SessionHeader {
		header_type: SessionHeaderType::Session,
		version: Some(3),
		id: "stored-session".to_string(),
		timestamp: "2026-01-01T00:00:00Z".to_string(),
		cwd: "/stored".to_string(),
		parent_session: None,
	}))
	.chain(body.into_iter().map(FileEntry::Entry))
	.collect();

	let session = SessionManager::in_memory(
		Some("/project"),
		Some(NewSessionOptions { id: Some("ignored".into()), ..Default::default() }),
		Some(entries),
	)
	.unwrap();

	assert_eq!(session.get_session_id(), "stored-session");
	assert_eq!(session.get_header().unwrap().cwd, "/stored");
}

#[test]
fn preloaded_migrates_entries_restored_with_an_older_header() {
	let entries: Vec<FileEntry> = serde_json::from_value(json!([
		{ "type": "session", "version": 2, "id": "v2-session", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/project" },
		{
			"type": "message",
			"id": "abc12345",
			"parentId": null,
			"timestamp": "2026-01-01T00:00:01Z",
			"message": { "role": "hookMessage", "content": "from a hook", "timestamp": 1 }
		}
	]))
	.unwrap();

	let session = SessionManager::in_memory(Some("/project"), None, Some(entries)).unwrap();
	let restored = session.get_entries()[0].clone();

	assert_eq!(session.get_header().unwrap().version, Some(3));
	assert_eq!(restored.base().id, "abc12345");
	let SessionEntry::Message(message_entry) = restored else { panic!("expected message entry") };
	assert_eq!(message_entry.message.role(), "custom");
}

#[test]
fn preloaded_adopts_headerless_entries_as_current_version_without_migrating_them() {
	let entries: Vec<FileEntry> = serde_json::from_value(json!([
		{
			"type": "message",
			"id": "abc12345",
			"parentId": null,
			"timestamp": "2026-01-01T00:00:01Z",
			"message": { "role": "hookMessage", "content": "from a hook", "timestamp": 1 }
		}
	]))
	.unwrap();

	let session = SessionManager::in_memory(Some("/project"), None, Some(entries)).unwrap();
	let restored = session.get_entries()[0].clone();

	let SessionEntry::Message(message_entry) = restored else { panic!("expected message entry") };
	assert_eq!(message_entry.message.role(), "hookMessage");
}

// ===========================================================================
// Upstream test/session-manager/migration.test.ts
// ===========================================================================

#[test]
fn migrate_should_add_id_parent_id_to_v1_entries() {
	let mut entries: Vec<FileEntry> = serde_json::from_value(json!([
		{ "type": "session", "id": "sess-1", "timestamp": "2025-01-01T00:00:00Z", "cwd": "/tmp" },
		{ "type": "message", "timestamp": "2025-01-01T00:00:01Z", "message": { "role": "user", "content": "hi", "timestamp": 1 } },
		{
			"type": "message",
			"timestamp": "2025-01-01T00:00:02Z",
			"message": {
				"role": "assistant",
				"content": [{ "type": "text", "text": "hello" }],
				"api": "test",
				"provider": "test",
				"model": "test",
				"usage": { "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0 },
				"stopReason": "stop",
				"timestamp": 2
			}
		}
	]))
	.unwrap();

	migrate_session_entries(&mut entries);

	// Header should have version set (v3 is current after hookMessage->custom migration)
	assert_eq!(entries[0].as_session_header().unwrap().version, Some(3));

	// Entries should have id/parentId
	let msg1_id = entries[1].as_entry().unwrap().id().to_string();
	let msg1_parent = entries[1].as_entry().unwrap().parent_id().map(String::from);
	let msg2_id = entries[2].as_entry().unwrap().id().to_string();
	let msg2_parent = entries[2].as_entry().unwrap().parent_id().map(String::from);

	assert_eq!(msg1_id.len(), 8);
	assert_eq!(msg1_parent, None);

	assert_eq!(msg2_id.len(), 8);
	assert_eq!(msg2_parent, Some(msg1_id));
}

#[test]
fn migrate_should_be_idempotent_skip_already_migrated() {
	let mut entries: Vec<FileEntry> = serde_json::from_value(json!([
		{ "type": "session", "id": "sess-1", "version": 2, "timestamp": "2025-01-01T00:00:00Z", "cwd": "/tmp" },
		{
			"type": "message",
			"id": "abc12345",
			"parentId": null,
			"timestamp": "2025-01-01T00:00:01Z",
			"message": { "role": "user", "content": "hi", "timestamp": 1 }
		},
		{
			"type": "message",
			"id": "def67890",
			"parentId": "abc12345",
			"timestamp": "2025-01-01T00:00:02Z",
			"message": {
				"role": "assistant",
				"content": [{ "type": "text", "text": "hello" }],
				"api": "test",
				"provider": "test",
				"model": "test",
				"usage": { "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0 },
				"stopReason": "stop",
				"timestamp": 2
			}
		}
	]))
	.unwrap();

	migrate_session_entries(&mut entries);

	// IDs should be unchanged
	assert_eq!(entries[1].as_entry().unwrap().id(), "abc12345");
	assert_eq!(entries[2].as_entry().unwrap().id(), "def67890");
	assert_eq!(entries[2].as_entry().unwrap().parent_id(), Some("abc12345"));
}

// ===========================================================================
// Upstream test/session-manager/save-entry.test.ts
// ===========================================================================

#[test]
fn save_custom_entry_saves_custom_entries_and_includes_them_in_tree_traversal() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	// Save a message
	let msg_id = session.append_message(user_msg("hello")).unwrap();

	// Save a custom entry
	let custom_id = session.append_custom_entry("my_data", Some(json!({ "foo": "bar" }))).unwrap();

	// Save another message
	let msg2_id = session.append_message(assistant_msg("hi")).unwrap();

	// Custom entry should be in entries
	let entries = session.get_entries();
	assert_eq!(entries.len(), 3);

	let Some(SessionEntry::Custom(custom_entry)) = entries.iter().find(|entry| entry.type_name() == "custom")
	else {
		panic!("custom entry missing");
	};
	assert_eq!(custom_entry.custom_type, "my_data");
	assert_eq!(custom_entry.data, Some(json!({ "foo": "bar" })));
	assert_eq!(custom_entry.base.id, custom_id);
	assert_eq!(custom_entry.base.parent_id, Some(msg_id.clone()));

	// Tree structure should be correct
	let path = session.get_branch();
	assert_eq!(path.len(), 3);
	assert_eq!(path[0].id(), msg_id);
	assert_eq!(path[1].id(), custom_id);
	assert_eq!(path[2].id(), msg2_id);

	// buildSessionContext should work (custom entries skipped in messages)
	let ctx = session.build_session_context();
	assert_eq!(ctx.messages.len(), 2); // only message entries
}

// ===========================================================================
// Upstream test/session-manager/tree-traversal.test.ts
// ===========================================================================

#[test]
fn append_message_creates_entry_with_correct_parent_id_chain() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let id1 = session.append_message(user_msg("first")).unwrap();
	let id2 = session.append_message(assistant_msg("second")).unwrap();
	let id3 = session.append_message(user_msg("third")).unwrap();

	let entries = session.get_entries();
	assert_eq!(entries.len(), 3);

	assert_eq!(entries[0].id(), id1);
	assert_eq!(entries[0].parent_id(), None);
	assert_eq!(entries[0].type_name(), "message");

	assert_eq!(entries[1].id(), id2);
	assert_eq!(entries[1].parent_id(), Some(id1.as_str()));

	assert_eq!(entries[2].id(), id3);
	assert_eq!(entries[2].parent_id(), Some(id2.as_str()));
}

#[test]
fn append_thinking_level_change_integrates_into_tree() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let msg_id = session.append_message(user_msg("hello")).unwrap();
	let thinking_id = session.append_thinking_level_change("high").unwrap();
	let _msg2_id = session.append_message(assistant_msg("response")).unwrap();

	let entries = session.get_entries();
	assert_eq!(entries.len(), 3);

	let thinking_entry = entries.iter().find(|entry| entry.type_name() == "thinking_level_change").unwrap();
	assert_eq!(thinking_entry.id(), thinking_id);
	assert_eq!(thinking_entry.parent_id(), Some(msg_id.as_str()));

	assert_eq!(entries[2].parent_id(), Some(thinking_id.as_str()));
}

#[test]
fn append_model_change_integrates_into_tree() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let msg_id = session.append_message(user_msg("hello")).unwrap();
	let model_id = session.append_model_change("openai", "gpt-4").unwrap();
	let _msg2_id = session.append_message(assistant_msg("response")).unwrap();

	let entries = session.get_entries();
	let model_entry = entries.iter().find(|entry| entry.type_name() == "model_change").unwrap();
	assert_eq!(model_entry.id(), model_id);
	assert_eq!(model_entry.parent_id(), Some(msg_id.as_str()));
	if let SessionEntry::ModelChange(change) = model_entry {
		assert_eq!(change.provider, "openai");
		assert_eq!(change.model_id, "gpt-4");
	}

	assert_eq!(entries[2].parent_id(), Some(model_id.as_str()));
}

#[test]
fn append_compaction_integrates_into_tree() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let id1 = session.append_message(user_msg("1")).unwrap();
	let id2 = session.append_message(assistant_msg("2")).unwrap();
	let usage = big_usage();
	let compaction_id =
		session.append_compaction("summary", Some(&id1), 1000, None, Some(false), Some(usage.clone())).unwrap();
	let _id3 = session.append_message(user_msg("3")).unwrap();

	let entries = session.get_entries();
	let compaction_entry = entries.iter().find(|entry| entry.type_name() == "compaction").unwrap();
	assert_eq!(compaction_entry.id(), compaction_id);
	assert_eq!(compaction_entry.parent_id(), Some(id2.as_str()));
	if let SessionEntry::Compaction(compaction) = compaction_entry {
		assert_eq!(compaction.summary, "summary");
		assert_eq!(compaction.first_kept_entry_id, id1);
		assert_eq!(compaction.tokens_before, 1000);
		assert_eq!(compaction.usage, Some(usage));
	}

	assert_eq!(entries[3].parent_id(), Some(compaction_id.as_str()));
}

#[test]
fn append_custom_entry_integrates_into_tree() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let msg_id = session.append_message(user_msg("hello")).unwrap();
	let custom_id = session.append_custom_entry("my_data", Some(json!({ "key": "value" }))).unwrap();
	let _msg2_id = session.append_message(assistant_msg("response")).unwrap();

	let entries = session.get_entries();
	let Some(SessionEntry::Custom(custom_entry)) =
		entries.iter().find(|entry| entry.type_name() == "custom")
	else {
		panic!("custom entry missing");
	};
	assert_eq!(custom_entry.base.id, custom_id);
	assert_eq!(custom_entry.base.parent_id, Some(msg_id));
	assert_eq!(custom_entry.custom_type, "my_data");
	assert_eq!(custom_entry.data, Some(json!({ "key": "value" })));

	assert_eq!(entries[2].parent_id(), Some(custom_id.as_str()));
}

#[test]
fn leaf_pointer_advances_after_each_append() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	assert_eq!(session.get_leaf_id(), None);

	let id1 = session.append_message(user_msg("1")).unwrap();
	assert_eq!(session.get_leaf_id(), Some(id1.as_str()));

	let id2 = session.append_message(assistant_msg("2")).unwrap();
	assert_eq!(session.get_leaf_id(), Some(id2.as_str()));

	let id3 = session.append_thinking_level_change("high").unwrap();
	assert_eq!(session.get_leaf_id(), Some(id3.as_str()));
}

#[test]
fn get_path_returns_empty_array_for_empty_session() {
	let session = SessionManager::in_memory(None, None, None).unwrap();
	assert!(session.get_branch().is_empty());
}

#[test]
fn get_path_returns_single_entry_path() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();
	let id = session.append_message(user_msg("hello")).unwrap();

	let path = session.get_branch();
	assert_eq!(path.len(), 1);
	assert_eq!(path[0].id(), id);
}

#[test]
fn get_path_returns_full_path_from_root_to_leaf() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let id1 = session.append_message(user_msg("1")).unwrap();
	let id2 = session.append_message(assistant_msg("2")).unwrap();
	let id3 = session.append_thinking_level_change("high").unwrap();
	let id4 = session.append_message(user_msg("3")).unwrap();

	let path = session.get_branch();
	assert_eq!(path.len(), 4);
	let ids: Vec<&str> = path.iter().map(|entry| entry.id()).collect();
	assert_eq!(ids, [id1.as_str(), id2.as_str(), id3.as_str(), id4.as_str()]);
}

#[test]
fn get_path_returns_path_from_specified_entry_to_root() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let id1 = session.append_message(user_msg("1")).unwrap();
	let id2 = session.append_message(assistant_msg("2")).unwrap();
	let _id3 = session.append_message(user_msg("3")).unwrap();
	let _id4 = session.append_message(assistant_msg("4")).unwrap();

	let path = session.get_branch_from(&id2);
	assert_eq!(path.len(), 2);
	let ids: Vec<&str> = path.iter().map(|entry| entry.id()).collect();
	assert_eq!(ids, [id1.as_str(), id2.as_str()]);
}

#[test]
fn get_tree_returns_empty_array_for_empty_session() {
	let session = SessionManager::in_memory(None, None, None).unwrap();
	assert!(session.get_tree().is_empty());
}

#[test]
fn get_tree_returns_single_root_for_linear_session() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let id1 = session.append_message(user_msg("1")).unwrap();
	let id2 = session.append_message(assistant_msg("2")).unwrap();
	let id3 = session.append_message(user_msg("3")).unwrap();

	let tree = session.get_tree();
	assert_eq!(tree.len(), 1);

	let root = &tree[0];
	assert_eq!(root.entry.id(), id1);
	assert_eq!(root.children.len(), 1);
	assert_eq!(root.children[0].entry.id(), id2);
	assert_eq!(root.children[0].children.len(), 1);
	assert_eq!(root.children[0].children[0].entry.id(), id3);
	assert_eq!(root.children[0].children[0].children.len(), 0);
}

#[test]
fn get_tree_returns_tree_with_branches_after_branch() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	// Build: 1 -> 2 -> 3
	let id1 = session.append_message(user_msg("1")).unwrap();
	let id2 = session.append_message(assistant_msg("2")).unwrap();
	let id3 = session.append_message(user_msg("3")).unwrap();

	// Branch from id2, add new path: 2 -> 4
	session.branch(&id2).unwrap();
	let id4 = session.append_message(user_msg("4-branch")).unwrap();

	let tree = session.get_tree();
	assert_eq!(tree.len(), 1);

	let root = &tree[0];
	assert_eq!(root.entry.id(), id1);
	assert_eq!(root.children.len(), 1);

	let node2 = &root.children[0];
	assert_eq!(node2.entry.id(), id2);
	assert_eq!(node2.children.len(), 2); // id3 and id4 are siblings

	let mut child_ids: Vec<&str> = node2.children.iter().map(|child| child.entry.id()).collect();
	child_ids.sort_unstable();
	let mut expected = vec![id3.as_str(), id4.as_str()];
	expected.sort_unstable();
	assert_eq!(child_ids, expected);
}

#[test]
fn get_tree_handles_multiple_branches_at_same_point() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let _id1 = session.append_message(user_msg("root")).unwrap();
	let id2 = session.append_message(assistant_msg("response")).unwrap();

	// Branch A
	session.branch(&id2).unwrap();
	let id_a = session.append_message(user_msg("branch-A")).unwrap();

	// Branch B
	session.branch(&id2).unwrap();
	let id_b = session.append_message(user_msg("branch-B")).unwrap();

	// Branch C
	session.branch(&id2).unwrap();
	let id_c = session.append_message(user_msg("branch-C")).unwrap();

	let tree = session.get_tree();
	let node2 = &tree[0].children[0];
	assert_eq!(node2.entry.id(), id2);
	assert_eq!(node2.children.len(), 3);

	let mut branch_ids: Vec<&str> = node2.children.iter().map(|child| child.entry.id()).collect();
	branch_ids.sort_unstable();
	let mut expected = vec![id_a.as_str(), id_b.as_str(), id_c.as_str()];
	expected.sort_unstable();
	assert_eq!(branch_ids, expected);
}

#[test]
fn get_tree_handles_deep_branching() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	// Main path: 1 -> 2 -> 3 -> 4
	let _id1 = session.append_message(user_msg("1")).unwrap();
	let id2 = session.append_message(assistant_msg("2")).unwrap();
	let id3 = session.append_message(user_msg("3")).unwrap();
	let _id4 = session.append_message(assistant_msg("4")).unwrap();

	// Branch from 2: 2 -> 5 -> 6
	session.branch(&id2).unwrap();
	let id5 = session.append_message(user_msg("5")).unwrap();
	let _id6 = session.append_message(assistant_msg("6")).unwrap();

	// Branch from 5: 5 -> 7
	session.branch(&id5).unwrap();
	let _id7 = session.append_message(user_msg("7")).unwrap();

	let tree = session.get_tree();

	// Verify structure
	let node2 = &tree[0].children[0];
	assert_eq!(node2.children.len(), 2); // id3 and id5

	let node5 = node2.children.iter().find(|child| child.entry.id() == id5).unwrap();
	assert_eq!(node5.children.len(), 2); // id6 and id7

	let node3 = node2.children.iter().find(|child| child.entry.id() == id3).unwrap();
	assert_eq!(node3.children.len(), 1); // id4
}

#[test]
fn branch_moves_leaf_pointer_to_specified_entry() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let id1 = session.append_message(user_msg("1")).unwrap();
	let _id2 = session.append_message(assistant_msg("2")).unwrap();
	let id3 = session.append_message(user_msg("3")).unwrap();

	assert_eq!(session.get_leaf_id(), Some(id3.as_str()));

	session.branch(&id1).unwrap();
	assert_eq!(session.get_leaf_id(), Some(id1.as_str()));
}

#[test]
fn branch_throws_for_non_existent_entry() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();
	session.append_message(user_msg("hello")).unwrap();

	let error = session.branch("nonexistent").unwrap_err();
	assert_eq!(error.to_string(), "Entry nonexistent not found");
}

#[test]
fn branch_new_appends_become_children_of_branch_point() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let id1 = session.append_message(user_msg("1")).unwrap();
	let _id2 = session.append_message(assistant_msg("2")).unwrap();

	session.branch(&id1).unwrap();
	let id3 = session.append_message(user_msg("branched")).unwrap();

	let entries = session.get_entries();
	let branched_entry = entries.iter().find(|entry| entry.id() == id3).unwrap();
	assert_eq!(branched_entry.parent_id(), Some(id1.as_str())); // sibling of id2
}

#[test]
fn branch_with_summary_inserts_branch_summary_with_the_source_and_destination_and_advances_leaf() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let id1 = session.append_message(user_msg("1")).unwrap();
	let _id2 = session.append_message(assistant_msg("2")).unwrap();
	let id3 = session.append_message(user_msg("3")).unwrap();

	let usage = big_usage();
	let summary_id =
		session.branch_with_summary(Some(&id1), "Summary of abandoned work", None, Some(false), Some(usage.clone()))
			.unwrap();

	assert_eq!(session.get_leaf_id(), Some(summary_id.as_str()));

	let entries = session.get_entries();
	let summary_entry = entries.iter().find(|entry| entry.type_name() == "branch_summary").unwrap();
	assert_eq!(summary_entry.parent_id(), Some(id1.as_str()));
	if let SessionEntry::BranchSummary(summary) = summary_entry {
		assert_eq!(summary.from_id, id3);
		assert_eq!(summary.summary, "Summary of abandoned work");
		assert_eq!(summary.usage, Some(usage));
	}
}

#[test]
fn branch_with_summary_throws_for_non_existent_entry() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();
	session.append_message(user_msg("hello")).unwrap();

	let error =
		session.branch_with_summary(Some("nonexistent"), "summary", None, None, None).unwrap_err();
	assert_eq!(error.to_string(), "Entry nonexistent not found");
}

#[test]
fn get_leaf_entry_returns_undefined_for_empty_session() {
	let session = SessionManager::in_memory(None, None, None).unwrap();
	assert!(session.get_leaf_entry().is_none());
}

#[test]
fn get_leaf_entry_returns_current_leaf_entry() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	session.append_message(user_msg("1")).unwrap();
	let id2 = session.append_message(assistant_msg("2")).unwrap();

	let leaf = session.get_leaf_entry();
	assert!(leaf.is_some());
	assert_eq!(leaf.unwrap().id(), id2);
}

#[test]
fn get_entry_returns_undefined_for_non_existent_id() {
	let session = SessionManager::in_memory(None, None, None).unwrap();
	assert!(session.get_entry("nonexistent").is_none());
}

#[test]
fn get_entry_returns_entry_by_id() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	let id1 = session.append_message(user_msg("first")).unwrap();
	let id2 = session.append_message(assistant_msg("second")).unwrap();

	let entry1 = session.get_entry(&id1).unwrap();
	assert_eq!(entry1.type_name(), "message");
	if let SessionEntry::Message(message_entry) = entry1 {
		assert_eq!(user_content_text(&message_entry.message).as_deref(), Some("first"));
	}

	let entry2 = session.get_entry(&id2).unwrap();
	if let SessionEntry::Message(message_entry) = entry2 {
		assert_eq!(assistant_block_text(&message_entry.message, 0).as_deref(), Some("second"));
	}
}

#[test]
fn build_session_context_with_branches_returns_messages_from_current_branch_only() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	// Main: 1 -> 2 -> 3
	session.append_message(user_msg("msg1")).unwrap();
	let id2 = session.append_message(assistant_msg("msg2")).unwrap();
	session.append_message(user_msg("msg3")).unwrap();

	// Branch from 2: 2 -> 4
	session.branch(&id2).unwrap();
	session.append_message(assistant_msg("msg4-branch")).unwrap();

	let ctx = session.build_session_context();
	assert_eq!(ctx.messages.len(), 3); // msg1, msg2, msg4-branch (not msg3)

	assert_eq!(user_content_text(&ctx.messages[0]).as_deref(), Some("msg1"));
	assert_eq!(assistant_block_text(&ctx.messages[1], 0).as_deref(), Some("msg2"));
	assert_eq!(assistant_block_text(&ctx.messages[2], 0).as_deref(), Some("msg4-branch"));
}

#[test]
fn create_branched_session_throws_for_non_existent_entry() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();
	session.append_message(user_msg("hello")).unwrap();

	let error = session.create_branched_session("nonexistent").unwrap_err();
	assert_eq!(error.to_string(), "Entry nonexistent not found");
}

#[test]
fn create_branched_session_creates_new_session_with_path_to_specified_leaf_in_memory() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	// Build: 1 -> 2 -> 3 -> 4
	let id1 = session.append_message(user_msg("1")).unwrap();
	let id2 = session.append_message(assistant_msg("2")).unwrap();
	let id3 = session.append_message(user_msg("3")).unwrap();
	session.append_message(assistant_msg("4")).unwrap();

	// Branch from 3: 3 -> 5
	session.branch(&id3).unwrap();
	let _id5 = session.append_message(user_msg("5")).unwrap();

	// Create branched session from id2 (should only have 1 -> 2)
	let result = session.create_branched_session(&id2).unwrap();
	assert_eq!(result, None); // in-memory returns null

	// Session should now only have entries 1 and 2
	let entries = session.get_entries();
	assert_eq!(entries.len(), 2);
	assert_eq!(entries[0].id(), id1);
	assert_eq!(entries[1].id(), id2);
}

#[test]
fn create_branched_session_extracts_correct_path_from_branched_tree() {
	let mut session = SessionManager::in_memory(None, None, None).unwrap();

	// Build: 1 -> 2 -> 3
	let id1 = session.append_message(user_msg("1")).unwrap();
	let id2 = session.append_message(assistant_msg("2")).unwrap();
	session.append_message(user_msg("3")).unwrap();

	// Branch from 2: 2 -> 4 -> 5
	session.branch(&id2).unwrap();
	let id4 = session.append_message(user_msg("4")).unwrap();
	let id5 = session.append_message(assistant_msg("5")).unwrap();

	// Create branched session from id5 (should have 1 -> 2 -> 4 -> 5)
	session.create_branched_session(&id5).unwrap();

	let entries = session.get_entries();
	assert_eq!(entries.len(), 4);
	let ids: Vec<&str> = entries.iter().map(|entry| entry.id()).collect();
	assert_eq!(ids, [id1.as_str(), id2.as_str(), id4.as_str(), id5.as_str()]);
}

#[test]
fn create_branched_session_does_not_duplicate_entries_when_forking_from_before_the_first_user_message() {
	let temp = temp_dir("fork-dedup");

	// Create a persisted session with a couple of turns
	let mut session = SessionManager::create(&temp, Some(&temp), None).unwrap();
	let model_change_id = session.append_model_change("anthropic", "claude-sonnet-4-5").unwrap();
	session.append_message(user_msg("first question")).unwrap();
	session.append_message(assistant_msg("first answer")).unwrap();

	// Fork from a setup entry (no user or assistant message in the branched path)
	let new_file = session.create_branched_session(&model_change_id).unwrap().unwrap();
	assert!(!std::path::Path::new(&new_file).exists());

	// Nothing to save yet, so the file is created later by the first user message
	session.append_message(user_msg("new question")).unwrap();
	assert!(std::path::Path::new(&new_file).exists());

	// Simulate extension adding entry before assistant (like preset on turn_start)
	session.append_custom_entry("preset-state", Some(json!({ "name": "plan" }))).unwrap();
	session.append_message(assistant_msg("new answer")).unwrap();

	// Exactly one header and each entry written once
	assert_eq!(read_session_file_roles(&new_file), ["session", "model_change", "user", "custom", "assistant"]);
	cleanup_dir(&temp);
}

#[test]
fn create_branched_session_preserves_tool_and_summary_usage_across_a_file_backed_reload() {
	let temp = temp_dir("usage-roundtrip");

	let mut session = SessionManager::create(&temp, Some(&temp), None).unwrap();
	let root_id = session.append_message(user_msg("question")).unwrap();
	session.append_message(assistant_msg("answer")).unwrap();
	let usage = big_usage();
	session
		.append_message(AgentMessage::tool_result(pi_ai::types::ToolResultMessage {
			tool_call_id: "call-1".to_string(),
			tool_name: "nested-model".to_string(),
			content: vec![pi_ai::types::UserContent::Text(pi_ai::types::TextContent::new("result"))],
			details: None,
			usage: Some(usage.clone()),
			nested_calls: None,
			is_error: false,
			timestamp: crate::utils::time::now_ms(),
		}))
		.unwrap();
	session.append_compaction("summary", Some(&root_id), 100, None, Some(false), Some(usage.clone())).unwrap();
	session.branch_with_summary(Some(&root_id), "branch summary", None, Some(false), Some(usage.clone())).unwrap();

	let file = session.get_session_file().unwrap().to_string();
	let reopened = SessionManager::open(&file, Some(&temp), None).unwrap();
	let entries = reopened.get_entries();
	assert!(
		entries.iter().any(|entry| matches!(entry, SessionEntry::Compaction(compaction) if compaction.usage == Some(usage.clone()))),
		"compaction usage not preserved"
	);
	assert!(
		entries
			.iter()
			.any(|entry| matches!(entry, SessionEntry::BranchSummary(summary) if summary.usage == Some(usage.clone()))),
		"branch summary usage not preserved"
	);
	assert!(
		entries.iter().any(|entry| matches!(
			entry,
			SessionEntry::Message(message_entry)
				if matches!(&message_entry.message, AgentMessage::Llm(pi_ai::types::Message::ToolResult(tool_result))
					if tool_result.usage == Some(usage.clone()))
		)),
		"tool result usage not preserved"
	);
	cleanup_dir(&temp);
}

#[test]
fn create_branched_session_writes_file_immediately_when_forking_at_a_user_message() {
	let temp = temp_dir("fork-with-user");

	let mut session = SessionManager::create(&temp, Some(&temp), None).unwrap();
	let id1 = session.append_message(user_msg("first question")).unwrap();
	session.append_message(assistant_msg("first answer")).unwrap();

	let new_file = session.create_branched_session(&id1).unwrap().unwrap();
	assert!(std::path::Path::new(&new_file).exists());

	session.append_message(assistant_msg("new answer")).unwrap();
	assert_eq!(read_session_file_roles(&new_file), ["session", "user", "assistant"]);
	cleanup_dir(&temp);
}
