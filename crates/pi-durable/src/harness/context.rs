//! Port of `packages/durable/src/harness/context.ts`.
//!
//! Deviations: D1 — bounds capture runs directly against
//! [`crate::types::Storage`] (upstream wraps reads in
//! `session.readOnLine`, a Session-line guarantee callers must keep: call
//! these on the Session line). D2 — `ContextView` is
//! [`crate::session::ContextView`] (session-owned). D3 — Rust has no
//! `freezeJson`: upstream freezes every scanned record and every returned
//! view deeply, so kept [`ContextRange`] entries and returned views cannot be
//! mutated through shared references. The port keeps ranges private to the
//! scheduler (which hands out cloned views), which preserves the observable
//! semantics without freezing.

use std::collections::HashSet;

use pi_ai::types::{
    AssistantMessage, Message, StopReason, ToolCall, ToolResultMessage, UserContent,
};
use pi_chord::context::Context;
use serde_json::Value;

use crate::ids::id_from_number;
use crate::types::{
    ContextEdit, ConversationId, Cursor, EntryId, EntryQuery, EntryRecord, Storage,
};

const SCAN_PAGE_SIZE: u64 = 256;

fn is_excluded_stop_reason(reason: StopReason) -> bool {
    matches!(
        reason,
        StopReason::Aborted | StopReason::Error | StopReason::Deferred
    )
}

const MISSING_RESULT_TEXT: &str =
    "Tool result unavailable: history ends before this call completed.";

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
            let query = EntryQuery {
                conversation_id,
                min_entry_id: None,
                max_entry_id: None,
                order: None,
            };
            let page = storage.scan_entries(&query, 1, None, context).await;
            match page.items.first() {
                Some(entry) => entry.id,
                None => return Ok(None),
            }
        }
        Some(at) => {
            if storage
                .entry_in(conversation_id, at, context)
                .await
                .is_none()
            {
                return Err(format!(
                    "Entry {at} is not visible from conversation {conversation_id}"
                ));
            }
            at
        }
    };
    let head = storage
        .find_latest_head_marker(conversation_id, Some(tail), context)
        .await;
    Ok(Some(ContextBounds { head, tail }))
}

/// One context read: the visible entries it scanned, from the head marker's
/// head, or transcript start, through `bounds.tail`, oldest first, and the
/// view derived from them. Entries at or below the tail never change, so a
/// later read with the same head marker extends both.
#[derive(Debug, Clone)]
pub struct ContextRange {
    pub bounds: ContextBounds,
    pub entries: Vec<EntryRecord>,
    pub view: crate::session::ContextView,
    /// Targets of the edits in `entries`.
    pub edited: HashSet<EntryId>,
    /// `view.messages` from the contributed messages before the last assistant
    /// message. Tool results are ordered within the messages up to the next
    /// assistant message, so later entries cannot change these.
    pub settled: Vec<Message>,
    /// Contributed messages from the last assistant message on, before tool
    /// result ordering.
    pub open: Vec<Message>,
}

/// `read_context()` that reuses `previous`, an earlier range of the same
/// conversation: with the same head marker, only entries after its tail are
/// scanned. Returns the view and the range to pass to the next read. The
/// range outlives the read; the returned view owns fresh clones of its
/// arrays (upstream `freezeJson`s the range and copies the view arrays).
pub async fn read_context_from(
    storage: &dyn Storage,
    conversation_id: ConversationId,
    context: &Context,
    at: Option<EntryId>,
    previous: Option<&ContextRange>,
) -> Result<(crate::session::ContextView, Option<ContextRange>), String> {
    let Some(bounds) = capture_context_bounds(storage, conversation_id, context, at).await? else {
        return Ok((crate::session::ContextView::default(), None));
    };
    let range: ContextRange = match previous {
        Some(previous)
            if previous.bounds.head.as_ref().map(|head| head.id)
                != bounds.head.as_ref().map(|head| head.id) =>
        {
            let scanned =
                scan_range(storage, conversation_id, &range_query(&bounds), context).await?;
            derive_range(bounds, scanned)
        }
        // The same head marker reuses `previous`; a different one (including
        // appearing or disappearing) reads the new range whole.
        None => {
            let scanned =
                scan_range(storage, conversation_id, &range_query(&bounds), context).await?;
            derive_range(bounds, scanned)
        }
        Some(previous) if bounds.tail == previous.bounds.tail => previous.clone(),
        Some(previous) if bounds.tail < previous.bounds.tail => {
            let tail = bounds.tail;
            derive_range(
                bounds,
                previous
                    .entries
                    .iter()
                    .filter(|entry| entry.id <= tail)
                    .cloned()
                    .collect(),
            )
        }
        Some(previous) => {
            let min_entry_id = id_from_number::<EntryId>(previous.bounds.tail.0 + 1);
            let added = scan_range(
                storage,
                conversation_id,
                &RangeQuery {
                    min_entry_id: Some(min_entry_id),
                    max_entry_id: Some(bounds.tail),
                },
                context,
            )
            .await?;
            extend_range(previous.clone(), bounds, added)
        }
    };
    Ok((range.view.clone(), Some(range)))
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
    let scanned = scan_range(storage, conversation_id, &range_query(&bounds), context).await?;
    Ok(derive_range(bounds, scanned).view)
}

/// The visible range of `bounds`: from the head marker's head, or transcript
/// start, through the tail.
#[derive(Debug, Clone, Copy)]
struct RangeQuery {
    min_entry_id: Option<EntryId>,
    max_entry_id: Option<EntryId>,
}

fn range_query(bounds: &ContextBounds) -> RangeQuery {
    match &bounds.head {
        None => RangeQuery {
            min_entry_id: None,
            max_entry_id: Some(bounds.tail),
        },
        Some(head) => RangeQuery {
            min_entry_id: Some(head.head.unwrap_or(head.id)),
            max_entry_id: Some(bounds.tail),
        },
    }
}

/// Derive the context view from the entries scanned within `bounds`.
fn derive_range(bounds: ContextBounds, entries: Vec<EntryRecord>) -> ContextRange {
    let mut edits = std::collections::BTreeMap::<EntryId, ContextEdit>::new();
    // Edits of every entry in the range count, including older head markers
    // that `select_active` drops.
    for entry in &entries {
        for edit in entry.edits.iter().flatten() {
            edits.insert(edit.target, edit.clone());
        }
    }
    let head = bounds.head.clone();
    let active = select_active(head.as_ref(), &entries);
    let contributions: Vec<Vec<Message>> = active
        .iter()
        .map(|entry| contribute(entry, edits.get(&entry.id)))
        .collect();
    let flat: Vec<Message> = contributions.iter().flatten().cloned().collect();
    let (settled, open) = settle(Vec::new(), flat);
    let messages = lead_with_system(
        settled
            .iter()
            .chain(order_tool_results(open.clone()).iter())
            .cloned()
            .collect(),
    );
    let view = crate::session::ContextView {
        head,
        entries: active,
        contributions,
        messages,
    };
    ContextRange {
        bounds,
        entries,
        view,
        edited: edits.into_keys().collect(),
        settled,
        open,
    }
}

/// `previous` extended by the entries `added` after its tail under the same
/// head marker: only their contributions and the open messages are derived.
/// An added edit can change an earlier entry, so it derives the whole range
/// again.
fn extend_range(
    previous: ContextRange,
    bounds: ContextBounds,
    added: Vec<EntryRecord>,
) -> ContextRange {
    if added.iter().any(|entry| {
        entry.edits.is_some() || entry.head.is_some() || previous.edited.contains(&entry.id)
    }) {
        let mut entries = previous.entries;
        entries.extend(added);
        return derive_range(bounds, entries);
    }
    let contributions: Vec<Vec<Message>> =
        added.iter().map(|entry| contribute(entry, None)).collect();
    let flat: Vec<Message> = contributions.iter().flatten().cloned().collect();
    let mut open = previous.open;
    open.extend(flat);
    let (settled, open) = settle(previous.settled, open);
    let view = crate::session::ContextView {
        head: bounds.head.clone(),
        entries: previous
            .view
            .entries
            .iter()
            .cloned()
            .chain(added.iter().cloned())
            .collect(),
        contributions: previous
            .view
            .contributions
            .iter()
            .cloned()
            .chain(contributions)
            .collect(),
        messages: lead_with_system(
            settled
                .iter()
                .chain(order_tool_results(open.clone()).iter())
                .cloned()
                .collect(),
        ),
    };
    ContextRange {
        bounds,
        entries: previous.entries.iter().cloned().chain(added).collect(),
        view,
        edited: previous.edited,
        settled,
        open,
    }
}

/// Move a system message that only user messages precede to the front. A
/// run's input is committed before generation renders the system prompt, so
/// a transcript, or the range after a compaction or reset, starts with user
/// messages followed by the baseline system message. Providers treat only a
/// leading system message as the initial prompt and tool set; without it, a
/// later tool change rewrites the request's tool list and invalidates the
/// whole prompt cache.
fn lead_with_system(messages: Vec<Message>) -> Vec<Message> {
    let index = match messages
        .iter()
        .position(|message| message_role(message) != "user")
    {
        Some(index) => index,
        None => return messages,
    };
    if index == 0 || message_role(&messages[index]) != "system" {
        return messages;
    }
    let mut led = Vec::with_capacity(messages.len());
    led.push(messages[index].clone());
    led.extend(messages[..index].iter().cloned());
    led.extend(messages[index + 1..].iter().cloned());
    led
}

fn message_role(message: &Message) -> &'static str {
    match message {
        Message::System(_) => "system",
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::ToolResult(_) => "toolResult",
    }
}

/// One active entry's model messages after its edit and excluded stop
/// reasons, before tool result ordering.
fn contribute(entry: &EntryRecord, edit: Option<&ContextEdit>) -> Vec<Message> {
    if matches!(edit, Some(edit) if edit.action == crate::types::ContextEditAction::Omit) {
        return Vec::new();
    }
    let contributed: Vec<Message> = match edit {
        Some(edit) if matches!(edit.action, crate::types::ContextEditAction::Replace) => {
            edit.messages.clone().unwrap_or_default()
        }
        _ => entry.model.clone().unwrap_or_default(),
    };
    contributed
		.into_iter()
		.filter(|message| !matches!(message, Message::Assistant(a) if is_excluded_stop_reason(a.stop_reason)))
		.collect()
}

/// Move the ordered messages before the last assistant message of `open` to
/// `settled`. `order_tool_results()` of a sequence equals the concatenation
/// over its parts when each later part starts with an assistant message.
fn settle(settled: Vec<Message>, open: Vec<Message>) -> (Vec<Message>, Vec<Message>) {
    let last = open
        .iter()
        .rposition(|message| message_role(message) == "assistant");
    match last {
        Some(last) if last > 0 => {
            let mut settled = settled;
            settled.extend(order_tool_results(open[..last].to_vec()));
            (settled, open[last..].to_vec())
        }
        _ => (settled, open),
    }
}

/// The raw active entries within captured bounds, without deriving model
/// context.
pub async fn active_entries(
    storage: &dyn Storage,
    conversation_id: ConversationId,
    bounds: Option<ContextBounds>,
    context: &Context,
) -> Result<Vec<EntryRecord>, String> {
    let Some(bounds) = bounds else {
        return Ok(Vec::new());
    };
    let range = scan_range(storage, conversation_id, &range_query(&bounds), context).await?;
    Ok(select_active(bounds.head.as_ref(), &range))
}

/// Visible entries within `range`, oldest first.
async fn scan_range(
    storage: &dyn Storage,
    conversation_id: ConversationId,
    range: &RangeQuery,
    context: &Context,
) -> Result<Vec<EntryRecord>, String> {
    let query = EntryQuery {
        conversation_id,
        min_entry_id: range.min_entry_id,
        max_entry_id: range.max_entry_id,
        order: None,
    };
    let mut range = Vec::new();
    let mut cursor: Option<Cursor> = None;
    loop {
        let page = storage
            .scan_entries(&query, SCAN_PAGE_SIZE, cursor.as_ref(), context)
            .await;
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
                results
                    .entry(candidate.tool_call_id.clone())
                    .or_insert(next);
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
        content: vec![UserContent::Text(pi_ai::types::TextContent::new(
            MISSING_RESULT_TEXT,
        ))],
        is_error: true,
        details: Some(serde_json::json!({ "reason": "missing_result" })),
        usage: None,
        nested_calls: None,
        duration_ms: None,
        timestamp,
    }
}

/// Sections in effect after replaying system messages in order: set in
/// place, `None` deletes, re-adding appends.
pub fn replay_sections(messages: &[Message]) -> serde_json::Map<String, Value> {
    let mut shown = serde_json::Map::new();
    for message in messages {
        let Message::System(system) = message else {
            continue;
        };
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
