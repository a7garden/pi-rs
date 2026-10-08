//! Port of `packages/coding-agent/src/core/session-export.ts` — serialize the
//! current session branch (plus optional export-only entries) as JSONL.
//!
//! There is no upstream `session-export.test.ts`: consumers are
//! `AgentSession.exportSession` (`core/agent-session.ts`), the interactive
//! `session-share.ts`/`bug-report.ts`, all deferred modules (deviation D66).
//! The [`tests`] module pins the wire shape so the deferred consumers land
//! against a verified serializer.

use std::path::Path;

use serde_json::{json, Value};

use crate::core::session_manager::{SessionEntry, SessionEntryBase, SessionHeader, SessionHeaderType, SessionManager, CURRENT_SESSION_VERSION};
use crate::utils::paths::resolve_path;
use crate::utils::time::{now_ms, to_iso_timestamp};

/// Upstream `TrailingEntries`: given the last branch entry's id (`None` for an
/// empty branch) and the export timestamp, produce export-only entries.
/// Upstream returns arbitrary objects; here they are JSON values (deviation
/// D66).
pub type TrailingEntriesFn<'a> = dyn FnOnce(Option<String>, String) -> Vec<Value> + 'a;

/// Base fields of any session entry (all payloads carry `base`).
fn entry_base(entry: &SessionEntry) -> &SessionEntryBase {
	match entry {
		SessionEntry::Message(payload) => &payload.base,
		SessionEntry::ThinkingLevelChange(payload) => &payload.base,
		SessionEntry::ModelChange(payload) => &payload.base,
		SessionEntry::Usage(payload) => &payload.base,
		SessionEntry::Compaction(payload) => &payload.base,
		SessionEntry::BranchSummary(payload) => &payload.base,
		SessionEntry::Custom(payload) => &payload.base,
		SessionEntry::CustomMessage(payload) => &payload.base,
		SessionEntry::ContextEdit(payload) => &payload.base,
		SessionEntry::Label(payload) => &payload.base,
		SessionEntry::SessionInfo(payload) => &payload.base,
	}
}

/// `serializeSessionBranch`: serialize the current branch and optional
/// export-only entries as JSONL, rewiring each entry's `parentId` to its
/// predecessor.
pub fn serialize_session_branch(
	session_manager: &SessionManager,
	create_trailing_entries: Option<Box<TrailingEntriesFn<'_>>>,
) -> String {
	let timestamp = to_iso_timestamp(now_ms());
	let header = SessionHeader::new(session_manager.get_session_id(), timestamp.clone(), session_manager.get_cwd());
	let mut entries: Vec<Value> = vec![serde_json::to_value(&header).expect("session header serializes")];
	let mut parent_id: Option<String> = None;
	for entry in session_manager.get_branch() {
		let mut value = serde_json::to_value(&entry).expect("session entry serializes");
		value["parentId"] = match &parent_id {
			Some(id) => json!(id),
			None => Value::Null,
		};
		parent_id = Some(entry_base(&entry).id.clone());
		entries.push(value);
	}
	if let Some(create_trailing_entries) = create_trailing_entries {
		entries.extend(create_trailing_entries(parent_id, timestamp));
	}
	let lines: Vec<String> = entries
		.iter()
		.map(|entry| serde_json::to_string(entry).expect("entry serializes"))
		.collect();
	format!("{}\n", lines.join("\n"))
}

/// `exportSessionToJsonl`: write the current branch (plus optional
/// export-only entries) as JSONL, returning the written path. `output_path`
/// defaults to `session-<ISO timestamp>.jsonl` resolved against the current
/// directory.
pub fn export_session_to_jsonl(
	session_manager: &SessionManager,
	output_path: Option<&str>,
	create_trailing_entries: Option<Box<TrailingEntriesFn<'_>>>,
) -> std::io::Result<String> {
	let default_name = format!("session-{}.jsonl", to_iso_timestamp(now_ms()).replace([':', '.'], "-"));
	let file_path = resolve_path(output_path.unwrap_or(&default_name), &std::env::current_dir()?.to_string_lossy());
	let dir = Path::new(&file_path)
		.parent()
		.map(|parent| parent.to_string_lossy().to_string())
		.unwrap_or_else(|| ".".to_string());
	if !Path::new(&dir).exists() {
		std::fs::create_dir_all(&dir)?;
	}
	std::fs::write(&file_path, serialize_session_branch(session_manager, create_trailing_entries))?;
	Ok(file_path)
}

#[cfg(test)]
mod tests {
	use super::*;
	use pi_agent::types::AgentMessage;

	// No upstream suite (see module docs): shape test for the deferred
	// consumers, mirroring `AgentSession.exportSession` output — a session
	// header, the branch with rewired `parentId`s, then trailing entries.
	#[test]
	fn serializes_header_branch_parent_ids_and_trailing_entries() {
		let dir = std::env::temp_dir().join(format!("pi-session-export-{}", std::process::id()));
		let _ = std::fs::remove_dir_all(&dir);
		std::fs::create_dir_all(&dir).unwrap();
		let session_file = dir.join("session.jsonl");
		let mut manager = SessionManager::open(&session_file.to_string_lossy(), None, None).unwrap();
		manager
			.append_message(serde_json::from_value::<AgentMessage>(serde_json::json!({
				"role": "user", "content": "hi", "timestamp": 0
			}))
			.unwrap())
			.unwrap();
		manager
			.append_message(
				serde_json::from_value::<AgentMessage>(serde_json::json!({
					"role": "assistant",
					"content": [],
					"api": "anthropic-messages",
					"provider": "anthropic",
					"model": "claude-test",
					"usage": {
						"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
						"cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 }
					},
					"stopReason": "stop",
					"timestamp": 0
				}))
				.unwrap(),
			)
			.unwrap();

		let jsonl = serialize_session_branch(
			&manager,
			Some(Box::new(|parent_id, export_timestamp| {
				assert!(!export_timestamp.is_empty());
				vec![json!({ "type": "custom-message", "customType": "share", "parentId": parent_id })]
			})),
		);

		let lines: Vec<Value> = jsonl
			.trim_end_matches('\n')
			.split('\n')
			.map(|line| serde_json::from_str(line).unwrap())
			.collect();
		assert_eq!(lines.len(), 4);
		assert_eq!(lines[0]["type"], "session");
		assert_eq!(lines[0]["version"], CURRENT_SESSION_VERSION);
		assert_eq!(lines[0]["id"], manager.get_session_id());
		assert_eq!(lines[0]["cwd"], manager.get_cwd());
		assert!(lines[0]["parentId"].is_null());
		assert_eq!(lines[1]["message"]["role"], "user");
		assert!(lines[1]["parentId"].is_null());
		assert_eq!(lines[2]["message"]["role"], "assistant");
		assert_eq!(lines[2]["parentId"], lines[1]["id"]);
		// Trailing entries chain off the last branch entry.
		assert_eq!(lines[3]["customType"], "share");
		assert_eq!(lines[3]["parentId"], lines[2]["id"]);
		// File ends with a newline.
		assert!(jsonl.ends_with('\n'));

		let output_path = dir.join("exported").join("out.jsonl");
		let written =
			export_session_to_jsonl(&manager, Some(&output_path.to_string_lossy()), None).unwrap();
		assert_eq!(written, output_path.to_string_lossy());
		assert!(output_path.exists());

		let _ = std::fs::remove_dir_all(&dir);
	}
}
