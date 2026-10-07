//! Port of `packages/durable/src/harness/context.ts`.
//!
//! Deviations: D1 — bounds capture runs directly against
//! [`crate::types::Storage`] (upstream wraps reads in
//! `session.readOnLine`, a Session-line guarantee callers must keep: call
//! these on the Session line). D2 — `ContextView` is
//! [`crate::session::ContextView`] (session-owned).

use pi_chord::context::Context;
use serde_json::Value;
use pi_ai::types::{AssistantMessage, Message, StopReason, ToolCall, ToolResultMessage, UserContent};

use crate::types::{
	ContextEdit, ConversationId, Cursor, EntryId, EntryQuery, EntryRecord, Storage,
};

const SCAN_PAGE_SIZE: u64 = 256;

fn is_excluded_stop_reason(reason: StopReason) -> bool {
	matches!(reason, StopReason::Aborted | StopReason::Error | StopReason::Deferred)
}

const MISSING_RESULT_TEXT: &str = "Tool result unavailable: history ends before this call completed.";

/// Head marker and newest visible entry that fix one committed context range.
#[derive(Debug, Clone)]
pub struct ContextBounds {
	/// Newest visible entry that carries a `head`.
	pub head: Option<EntryRecord>,
	pub tail: EntryId,
}

/// Capture the bounds of the current context, or of the context cut off at
/// the visible entry `at`, with two O(1) reads. Run this on the Session
/// line; entries at or below the tail are immutable, so
/// [`derive_context`] can then scan them off the line.
pub async fn capture_context_bounds(
	storage: &dyn Storage,
	conversation_id: ConversationId,
	context: &Context,
	at: Option<EntryId>,
) -> Result<Option<ContextBounds>, String> {
	let tail: EntryId = match at {
		None => {
			let query = EntryQuery { conversation_id, min_entry_id: None, max_entry_id: None };
			let page = storage.scan_entries(&query, 1, None, context).await;
			match page.items.first() {
				Some(entry) => entry.id,
				None => return Ok(None),
			}
		}
		Some(at) => {
			if storage.entry_in(conversation_id, at, context).await.is_none() {
				return Err(format!("Entry {at} is not visible from conversation {conversation_id}"));
			}
			at
		}
	};
	let head = storage.find_latest_head_marker(conversation_id, Some(tail), context).await;
	Ok(Some(ContextBounds { head, tail }))
}

/// Derive the active transcript and model context of one conversation within
/// captured bounds.
///
/// H = newest visible head marker; the range runs from `H.head` (or
/// transcript start) through the tail. Per target, the newest edit in the
/// range wins. Context entries are H followed by the range's non-head
/// entries.
pub async fn derive_context(
	storage: &dyn Storage,
	conversation_id: ConversationId,
	bounds: Option<ContextBounds>,
	context: &Context,
) -> Result<crate::session::ContextView, String> {
	let Some(bounds) = bounds else {
		return Ok(crate::session::ContextView::default());
	};
	let head = bounds.head.clone();
	let range = scan_range(storage, conversation_id, &bounds, context).await?;
	let mut edits = std::collections::BTreeMap::<EntryId, ContextEdit>::new();
	// Edits of every entry in the range count, including older head markers
	// that `select_active` drops.
	for entry in &range {
		for edit in entry.edits.iter().flatten() {
			edits.insert(edit.target, edit.clone());
		}
	}

	let entries = select_active(head.as_ref(), &range);
	let mut contributions = Vec::<Vec<Message>>::with_capacity(entries.len());
	for entry in &entries {
		let edit = edits.get(&entry.id);
		if matches!(edit, Some(edit) if edit.action == crate::types::ContextEditAction::Omit) {
			contributions.push(Vec::new());
			continue;
		}
		let contributed: Vec<Message> = match edit {
			Some(edit) if matches!(edit.action, crate::types::ContextEditAction::Replace) => {
				edit.messages.clone().unwrap_or_default()
			}
			_ => entry.model.clone().unwrap_or_default(),
		};
		contributions.push(
			contributed
				.into_iter()
				.filter(|message| !matches!(message, Message::Assistant(a) if is_excluded_stop_reason(a.stop_reason)))
				.collect(),
		);
	}
	let flat: Vec<Message> = contributions.iter().flatten().cloned().collect();
	let messages = order_tool_results(flat);
	Ok(crate::session::ContextView {
		head,
		entries,
		contributions,
		messages,
	})
}

/// The raw active entries within captured bounds, without deriving model
/// context.
pub async fn active_entries(
	storage: &dyn Storage,
	conversation_id: ConversationId,
	bounds: Option<ContextBounds>,
	context: &Context,
) -> Result<Vec<EntryRecord>, String> {
	let Some(bounds) = bounds else { return Ok(Vec::new()) };
	let range = scan_range(storage, conversation_id, &bounds, context).await?;
	Ok(select_active(bounds.head.as_ref(), &range))
}

/// Visible entries from the head marker's head, or transcript start, through
/// the tail, oldest first.
async fn scan_range(
	storage: &dyn Storage,
	conversation_id: ConversationId,
	bounds: &ContextBounds,
	context: &Context,
) -> Result<Vec<EntryRecord>, String> {
	let query = EntryQuery {
		conversation_id,
		min_entry_id: bounds.head.as_ref().map(|head| {
			head.head.unwrap_or(head.id)
		}),
		max_entry_id: Some(bounds.tail),
	};
	let mut range = Vec::new();
	let mut cursor: Option<Cursor> = None;
	loop {
		let page = storage.scan_entries(&query, SCAN_PAGE_SIZE, cursor.as_ref(), context).await;
		range.extend(page.items);
		cursor = page.next;
		if cursor.is_none() {
			break;
		}
	}
	range.reverse();
	Ok(range)
}

/// The head marker followed by the range's non-head entries, or the whole
/// range without a marker.
fn select_active(head: Option<&EntryRecord>, range: &[EntryRecord]) -> Vec<EntryRecord> {
	match head {
		None => range.to_vec(),
		Some(head) => {
			let mut active = vec![head.clone()];
			active.extend(range.iter().filter(|entry| entry.head.is_none()).cloned());
			active
		}
	}
}

/// Place each assistant's tool results directly after it in call order.
/// Results are taken from the messages before the next assistant; a missing
/// result is synthesized and unmatched results are dropped.
pub fn order_tool_results(messages: Vec<Message>) -> Vec<Message> {
	let mut ordered = Vec::with_capacity(messages.len());
	let count = messages.len();
	for index in 0..count {
		let message = messages[index].clone();
		if matches!(message, Message::ToolResult(_)) {
			continue;
		}
		let calls: Vec<ToolCall> = match &message {
			Message::Assistant(assistant) => assistant
				.content
				.iter()
				.filter_map(|block| match block {
					pi_ai::types::AssistantContent::ToolCall(call) => Some(call.clone()),
					_ => None,
				})
				.collect(),
			_ => Vec::new(),
		};
		let is_assistant = matches!(message, Message::Assistant(_));
		let timestamp = message_timestamp(&message);
		ordered.push(message);
		if !is_assistant || calls.is_empty() {
			continue;
		}
		let mut results = std::collections::BTreeMap::<String, usize>::new();
		let mut next = index + 1;
		while next < count && !matches!(messages[next], Message::Assistant(_)) {
			if let Message::ToolResult(candidate) = &messages[next] {
				results.entry(candidate.tool_call_id.clone()).or_insert(next);
			}
			next += 1;
		}
		for call in calls {
			match results.get(&call.id) {
				Some(&result_index) => ordered.push(messages[result_index].clone()),
				None => ordered.push(Message::ToolResult(missing_result(&call, timestamp))),
			}
		}
	}
	ordered
}

fn message_timestamp(message: &Message) -> i64 {
	match message {
		Message::System(m) => m.timestamp,
		Message::User(m) => m.timestamp,
		Message::Assistant(m) => m.timestamp,
		Message::ToolResult(m) => m.timestamp,
	}
}

fn missing_result(call: &ToolCall, timestamp: i64) -> ToolResultMessage {
	ToolResultMessage {
		tool_call_id: call.id.clone(),
		tool_name: call.name.clone(),
		content: vec![UserContent::Text(pi_ai::types::TextContent::new(MISSING_RESULT_TEXT))],
		is_error: true,
		details: Some(serde_json::json!({ "reason": "missing_result" })),
		usage: None,
		nested_calls: None,
		timestamp,
	}
}

/// Sections in effect after replaying system messages in order: set in
/// place, `None` deletes, re-adding appends.
pub fn replay_sections(messages: &[Message]) -> serde_json::Map<String, Value> {
	let mut shown = serde_json::Map::new();
	for message in messages {
		let Message::System(system) = message else { continue };
		for (key, value) in system.sections.iter().flatten() {
			match value {
				None => {
					shown.shift_remove(key);
				}
				Some(value) => {
					if let Some(existing) = shown.get_mut(key) {
						*existing = Value::String(value.clone());
					} else {
						shown.insert(key.clone(), Value::String(value.clone()));
					}
				}
			}
		}
	}
	shown
}

// Keep AssistantMessage imported for the stop-reason filter above.
#[allow(unused)]
fn _assert(message: &AssistantMessage) -> StopReason {
	message.stop_reason
}
