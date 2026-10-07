//! Port of `packages/durable/src/harness/inbox.ts`.
//!
//! Deviations: D1 — `InboxState`/`InboxItem` are typed serde structs; the
//! commit glue converts the erased document draft like `live.rs` (mutations
//! are stored by one root set; ops derive from value diffing).
//! D2 — `Boundary` holds the read-only snapshot of the items plus the
//! mutable draft handle; item removal happens through
//! [`store_items`].

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::documents::{DocDefinition, DocToken};
use crate::types::{DocumentFork, DocumentHistory};
use crate::entries::USER_ENTRY;
use crate::harness::api::QueueMode;
use crate::harness::types::{ConversationId, EntryId, SubmissionId};
use crate::session::Tx;
use crate::types::{EntryDraft, EntryHead, SubmissionSettlement};

/// A queued submission: user input for a run, or a passive entry write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "camelCase")]
pub enum InboxItem {
	#[serde(rename = "steer")]
	Steer { id: SubmissionId, content: Value },
	#[serde(rename = "followUp")]
	FollowUp { id: SubmissionId, content: Value },
	/// `entry` is an `EntryDraft`, stored as plain JSON.
	#[serde(rename = "write")]
	Write { id: SubmissionId, entry: EntryDraft },
}

impl InboxItem {
	pub fn id(&self) -> SubmissionId {
		match self {
			InboxItem::Steer { id, .. }
			| InboxItem::FollowUp { id, .. }
			| InboxItem::Write { id, .. } => *id,
		}
	}
}

/// Built-in queue of one conversation's submissions waiting for a boundary,
/// in ID order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InboxState {
	pub items: Vec<InboxItem>,
}

fn inbox_definition() -> DocDefinition {
	DocDefinition::conversation(
		"pi.inbox",
		1,
		DocumentHistory::Latest,
		DocumentFork::Initial,
		|| serde_json::json!({ "items": [] }).as_object().cloned().unwrap_or_default(),
	)
	.with_checkpoint_when(|value, _, _| {
		value.get("items").and_then(|items| items.as_array()).is_some_and(|items| items.is_empty())
	})
}

static INBOX_DOC: std::sync::LazyLock<DocToken> =
	std::sync::LazyLock::new(|| DocToken::new(inbox_definition()));

/// The conversation inbox document.
pub fn inbox_doc() -> DocToken {
	INBOX_DOC.clone()
}

/// The settings a boundary reads, on the Session line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueModes {
	pub steering_mode: QueueMode,
	pub follow_up_mode: QueueMode,
}

/// Read the typed inbox of a conversation from the erased draft.
pub fn read_inbox(draft: &crate::session::DocDraft) -> InboxState {
	serde_json::from_value(Value::Object(draft.value().unwrap_or_default())).unwrap_or_default()
}

/// Store the typed inbox through the erased draft.
pub fn store_inbox(draft: &crate::session::DocDraft, state: &InboxState) -> Result<(), String> {
	use pi_chord::delta::Seg;
	let value = serde_json::to_value(state).expect("InboxState serializes");
	match value.get("items") {
		Some(items) => draft.set(&[Seg::key("items")], items.clone()),
		None if draft.has(&[], "items")? => draft.delete(&[Seg::key("items")]),
		None => Ok(()),
	}
}

/// What a boundary reads before the commit's first table write, and the
/// newest head it has seen so far.
pub struct Boundary {
	pub conversation_id: ConversationId,
	pub draft: crate::session::DocDraft,
	pub state: InboxState,
	pub steering_mode: QueueMode,
	pub follow_up_mode: QueueMode,
	/// Start of the active range, the newest head marker's `head`; advanced
	/// by heads written in this commit.
	pub head: Option<EntryId>,
}

/// Selected user items, in ID order, and whether a `head: "self"` write (a
/// reset) was placed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryResult {
	pub users: Vec<SubmissionId>,
	pub reset: bool,
}

/// Read what a boundary needs. Table reads must precede the commit's first
/// table write, so callers prepare the boundary at the start of their
/// commit.
pub fn prepare_boundary(
	tx: &Tx<'_>,
	conversation_id: ConversationId,
	modes: QueueModes,
) -> Result<Boundary, String> {
	let head = tx
		.latest_head_marker(conversation_id)
		.map_err(|e| e.to_string())?
		.and_then(|entry| entry.head);
	let draft = tx.doc(&inbox_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
	let state = read_inbox(&draft);
	Ok(Boundary {
		conversation_id,
		draft,
		state,
		steering_mode: modes.steering_mode,
		follow_up_mode: modes.follow_up_mode,
		head,
	})
}

/// Boundary placement point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryAt {
	PostTools,
	Final,
}

/// Place the queued items a boundary selects (spec §6): every write, the
/// first or all steers, and at `final` the first or all follow-ups. A
/// selected reset turns a `postTools` boundary into `final`. Writes are
/// placed first and user items after them, each in ID order, so user items
/// queued before a reset run in the new context. A write whose head targets
/// an entry before the active range, including a range started earlier in
/// this commit, is stale. Selected and stale items are removed positionally.
pub fn apply_boundary(
	tx: &Tx<'_>,
	boundary: &mut Boundary,
	at: BoundaryAt,
	now: i64,
) -> Result<BoundaryResult, String> {
	let conversation_id = boundary.conversation_id;
	let items = &boundary.state.items;
	let reset = items.iter().any(|item| {
		matches!(item, InboxItem::Write { entry, .. } if entry.head == Some(EntryHead::SelfMarker(crate::types::SelfMarker)))
	});
	let is_final = at == BoundaryAt::Final || reset;
	let pick = |mode: fn(&InboxItem) -> bool, queue_mode: QueueMode| -> Vec<usize> {
		let indexes: Vec<usize> =
			items.iter().enumerate().filter(|(_, item)| mode(item)).map(|(index, _)| index).collect();
		match queue_mode {
			QueueMode::All => indexes,
			QueueMode::OneAtATime => indexes.into_iter().take(1).collect(),
		}
	};
	let writes: Vec<usize> =
		items.iter().enumerate().filter(|(_, item)| matches!(item, InboxItem::Write { .. })).map(|(i, _)| i).collect();
	let mut users: Vec<usize> = pick(
		|item| matches!(item, InboxItem::Steer { .. }),
		boundary.steering_mode,
	)
	.into_iter()
	.chain(if is_final {
		pick(|item| matches!(item, InboxItem::FollowUp { .. }), boundary.follow_up_mode)
	} else {
		Vec::new()
	})
	.collect();
	users.sort_unstable();

	// `append_entry` copies the drafts' values; the items are removed only
	// afterwards.
	for index in &writes {
		let item = &boundary.state.items[*index];
		let InboxItem::Write { id, entry } = item else { continue };
		if is_stale(boundary, entry) {
			tx.settle_submission(
				*id,
				SubmissionSettlement::Unanswered { reason: "stale".to_string(), detail: None },
			)
			.map_err(|e| e.to_string())?;
			continue;
		}
		let entry_record = tx.append_entry(conversation_id, entry.clone()).map_err(|e| e.to_string())?;
		if let Some(head) = entry.head {
			boundary.head = Some(match head {
				EntryHead::SelfMarker(_) => entry_record.id,
				EntryHead::Entry(id) => id,
			});
		}
		tx.place_submission(*id, entry_record.id).map_err(|e| e.to_string())?;
	}
	let mut placed = Vec::new();
	for index in &users {
		let item = &boundary.state.items[*index];
		let (id, content) = match item {
			InboxItem::Steer { id, content } => (*id, content),
			InboxItem::FollowUp { id, content } => (*id, content),
			InboxItem::Write { .. } => continue,
		};
		let message = pi_ai::types::UserMessage {
			content: serde_json::from_value::<pi_ai::types::UserContentField>(content.clone())
				.map_err(|error| format!("Queued input is not UserInput: {error}"))?,
			timestamp: now,
		};
		let mut draft = EntryDraft { kind: Some(USER_ENTRY.kind.clone()), ..Default::default() };
		draft.model = Some(vec![pi_ai::Message::User(message)]);
		let entry = tx.append_entry(conversation_id, draft).map_err(|e| e.to_string())?;
		tx.place_submission(id, entry.id).map_err(|e| e.to_string())?;
		placed.push(id);
	}
	let mut removed: Vec<usize> = writes
		.iter()
		.copied()
		.chain(users.iter().copied())
		.collect();
	removed.sort_unstable_by(|a, b| b.cmp(a));
	for index in removed {
		boundary.state.items.remove(index);
	}
	store_inbox(&boundary.draft, &boundary.state)?;
	Ok(BoundaryResult { users: placed, reset })
}

/// Whether a head write targets an entry before the active range, so placing
/// it would bring back cut history.
pub fn is_stale(boundary: &Boundary, entry: &EntryDraft) -> bool {
	match entry.head {
		Some(EntryHead::Entry(head)) => boundary.head.is_some_and(|boundary_head| head < boundary_head),
		_ => false,
	}
}

/// Remove a withdrawn submission's item; the caller settles the submission.
pub fn remove_inbox_item(
	tx: &Tx<'_>,
	conversation_id: ConversationId,
	id: SubmissionId,
) -> Result<(), String> {
	let draft = tx.doc(&inbox_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
	let mut state = read_inbox(&draft);
	if let Some(index) = state.items.iter().position(|item| item.id() == id) {
		state.items.remove(index);
	}
	store_inbox(&draft, &state)
}

/// Withdraw every queued input of a conversation, as
/// `Conversation.abort()` and abort cascades do: each settles `unanswered`
/// with `aborted` and leaves the inbox; queued writes stay for later
/// placement.
pub fn withdraw_queued_inputs(tx: &Tx<'_>, conversation_id: ConversationId) -> Result<(), String> {
	let draft = tx.doc(&inbox_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
	let mut state = read_inbox(&draft);
	for index in (0..state.items.len()).rev() {
		let item = &state.items[index];
		if matches!(item, InboxItem::Write { .. }) {
			continue;
		}
		tx.settle_submission(
			item.id(),
			SubmissionSettlement::Unanswered { reason: "aborted".to_string(), detail: None },
		)
		.map_err(|e| e.to_string())?;
		state.items.remove(index);
	}
	store_inbox(&draft, &state)
}
