//! Port of `packages/durable/src/types.ts` (data model + `Storage` boundary).
//!
//! Upstream models documents/tasks with extensive generics (`Id<Kind, Type>`,
//! `TaskRecord<I, S, R>`, `DocToken<T, D>`); storage itself only ever sees the
//! erased `TaskRecord<JsonValue, JsonValue, JsonValue>` shapes (upstream
//! `types.ts`). Rust ports the erased forms as the canonical wire types —
//! payloads (`input`, `checkpoint`, `result`, document values) are
//! `serde_json::Value`/`JsonObject` — so every serialized shape matches the
//! upstream JSON byte-for-byte. Typed ergonomics (tokens, definitions) live in
//! [`crate::documents`]; the generic task state machine surface lives in
//! [`crate::harness`]. Branded IDs become distinct newtypes ([`DurableId`]).
//!
//! Deviations (D1): `Page<T, C>` collapses to `Page<T>` (`Cursor` is already a
//! JSON object). D2: `DocumentPoint` uses custom serde (untagged `Seq | "current"`).
//! D3: `WatchEnd.listener_error` carries the error message (Rust `Error` values
//! are not JSON-safe by construction); upstream serializes it through
//! `MessageError`-style snapshots where observed.

use std::fmt;

use async_trait::async_trait;
use std::future::Future;
use pi_chord::context::Context;
use pi_chord::delta::Op;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::StorageRejected;

/// JSON object used as the root of every durable document.
pub type JsonObject = serde_json::Map<String, Value>;

/// Shared marker trait for the erased nominal ID newtypes.
pub trait DurableId: Copy + Into<u64> + fmt::Debug + fmt::Display + Ord + Send + Sync + 'static {
	/// Apply the erased ID brand at a trusted numeric allocation or decoding boundary.
	fn from_number(value: u64) -> Self;
}

macro_rules! durable_id {
	($(#[$meta:meta])* $name:ident) => {
		$(#[$meta])*
		#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
		#[serde(transparent)]
		pub struct $name(pub u64);

		impl DurableId for $name {
			fn from_number(value: u64) -> Self {
				$name(value)
			}
		}
		impl From<u64> for $name {
			fn from(value: u64) -> Self {
				$name(value)
			}
		}
		impl From<$name> for u64 {
			fn from(value: $name) -> u64 {
				value.0
			}
		}
		impl fmt::Display for $name {
			fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
				write!(f, "{}", self.0)
			}
		}
	};
}

durable_id!(
	/// Erased nominal number identifying one conversation record.
	ConversationId
);
durable_id!(
	/// Erased nominal number identifying one entry record.
	EntryId
);
durable_id!(
	/// Erased nominal number identifying one durable task.
	TaskId
);
durable_id!(
	/// Erased nominal number identifying one submission record.
	SubmissionId
);
durable_id!(
	/// Erased nominal number identifying one document incarnation.
	DocumentId
);

/// Strictly increasing sequence assigned to one atomic storage commit; gaps are permitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct Seq(pub u64);

impl Seq {
	/// Apply the erased commit-sequence brand at a trusted storage boundary.
	pub fn from_number(value: u64) -> Self {
		Seq(value)
	}
}

impl From<u64> for Seq {
	fn from(value: u64) -> Self {
		Seq(value)
	}
}

impl From<Seq> for u64 {
	fn from(value: Seq) -> u64 {
		value.0
	}
}

impl fmt::Display for Seq {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{}", self.0)
	}
}

/// The root conversation always uses this reserved ID.
pub const ROOT_CONVERSATION_ID: ConversationId = ConversationId(1);

/// Which transcript scope owns a document.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum DocumentScope {
	/// Owned by the Session itself.
	Session,
	/// Owned by one conversation.
	Conversation {
		#[serde(rename = "conversationId")]
		conversation_id: ConversationId,
	},
	/// Owned by one task.
	Task {
		#[serde(rename = "taskId")]
		task_id: TaskId,
	},
}

/// How much history a conversation document retains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DocumentHistory {
	/// Retain only current state.
	Latest,
	/// Retain history needed for as-of reads.
	Rewindable,
}

/// How a fork initializes the new incarnation's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DocumentFork {
	/// Initialize a fork at its cutoff (rewindable only).
	AsOf,
	/// Initialize from current source state.
	Current,
	/// Initialize from the definition's initial value.
	Initial,
}

/// Stored replay state supplied to a document's checkpoint predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointInfo {
	/// Deltas already stored after the newest base, excluding the change being evaluated.
	pub deltas_since_base: u64,
}

/// Immutable identity, history ancestry, and task ownership of a transcript scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationRecord {
	pub id: ConversationId,
	/// Fork source and inclusive parent entry through which history is inherited.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub parent: Option<ConversationParent>,
	/// Creator edge used for attribution, subtree abort, and subtree idle waits.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub owner: Option<ConversationOwner>,
}

/// Fork ancestry edge of a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationParent {
	pub conversation_id: ConversationId,
	pub at: EntryId,
}

/// Creator edge of a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationOwner {
	pub conversation_id: ConversationId,
	pub task_id: TaskId,
}

/// Ownership selected explicitly whenever a conversation is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ConversationOwnership {
	/// No owning task.
	Ownerless,
	/// Owned by one task of the conversation.
	Task {
		#[serde(rename = "taskId")]
		task_id: TaskId,
	},
}

/// An immutable override of one visible entry's contribution to model context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextEdit {
	/// Entry whose model messages are omitted or replaced.
	pub target: EntryId,
	/// `"omit"` or `"replace"`.
	pub action: ContextEditAction,
	/// Messages contributed instead of the target entry's model messages (`replace` only).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub messages: Option<Vec<pi_ai::Message>>,
}

/// Action of a [`ContextEdit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ContextEditAction {
	Omit,
	Replace,
}

/// Immutable transcript event with separate model-facing and application-facing payloads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryRecord {
	pub id: EntryId,
	pub conversation_id: ConversationId,
	/// Application-defined entry discriminator.
	pub kind: String,
	/// Messages contributed to model context; absent for display or bookkeeping entries.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub model: Option<Vec<pi_ai::Message>>,
	/// JSON payload consumed by views, extensions, or bookkeeping logic.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub data: Option<Value>,
	/// First entry in the active context selected by this entry.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub head: Option<EntryId>,
	/// Context-only overrides of earlier visible entries.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub edits: Option<Vec<ContextEdit>>,
	/// Task that appended this entry, when it was produced by durable work.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub by_task_id: Option<TaskId>,
}

impl EntryRecord {
	/// True when `head == Some(self.id)` — this entry starts the active context.
	pub fn starts_context(&self) -> bool {
		self.head == Some(self.id)
	}
}

/// Entry content supplied before the Session assigns identity and task attribution.
///
/// Upstream `EntryDraft` also allows `head: "self"`; Rust models that with
/// [`EntryHead`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryDraft {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub kind: Option<String>,
	/// Messages contributed to model context.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub model: Option<Vec<pi_ai::Message>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub data: Option<Value>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub head: Option<EntryHead>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub edits: Option<Vec<ContextEdit>>,
}

/// `EntryId | "self"` for [`EntryDraft::head`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EntryHead {
	SelfMarker(SelfMarker),
	Entry(EntryId),
}

/// Marker for `"self"` — starts active context at the newly assigned entry ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelfMarker;

impl serde::Serialize for SelfMarker {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		serializer.serialize_str("self")
	}
}

impl<'de> Deserialize<'de> for SelfMarker {
	fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
		let s = String::deserialize(deserializer)?;
		if s == "self" {
			Ok(SelfMarker)
		} else {
			Err(serde::de::Error::custom("expected \"self\""))
		}
	}
}

/// Identity fields shared by every durable submission state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmissionRecord {
	pub id: SubmissionId,
	pub conversation_id: ConversationId,
	/// Host-provided deduplication key, scoped to the conversation.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub request_id: Option<String>,
	/// `"input"` or `"write"`.
	#[serde(rename = "type")]
	pub submission_type: SubmissionType,
	/// Lifecycle status (`queued`, `placed`, `done`, `unanswered`).
	pub status: SubmissionStatus,
	/// Entry placed or appended (`placed`/`done` inputs, `done` writes).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub entry: Option<EntryId>,
	/// Answering entry (`done` inputs only).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub answer: Option<EntryId>,
	/// Terminal reason (`unanswered` only).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reason: Option<String>,
	/// Optional structured detail (`unanswered` only).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub detail: Option<Value>,
}

/// Kind of a durable submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SubmissionType {
	Input,
	Write,
}

/// Lifecycle status of a durable submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SubmissionStatus {
	Queued,
	Placed,
	Done,
	Unanswered,
}

/// Submission fields supplied before the Session assigns an ID.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmissionCreate {
	pub conversation_id: ConversationId,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub request_id: Option<String>,
	#[serde(rename = "type")]
	pub submission_type: Option<SubmissionType>,
	pub status: Option<SubmissionStatus>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub entry: Option<EntryId>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub answer: Option<EntryId>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reason: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub detail: Option<Value>,
}

/// Terminal status staged for a submission; identity, type, and entry come from its current record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum SubmissionSettlement {
	/// Successfully answered user input.
	Done {
		/// Answering entry.
		answer: EntryId,
	},
	/// Terminal without an answer.
	Unanswered {
		reason: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		detail: Option<Value>,
	},
}

/// JSON-safe error snapshot persisted instead of a runtime `Error` object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskOutcomeError {
	pub message: String,
	/// Optional structured diagnostic data for inspection or recovery.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub detail: Option<Value>,
}

/// Durable reason recorded when a task becomes terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TaskOutcomeStatus {
	Completed,
	Failed,
	Aborted,
	Orphaned,
	Faulted,
}

/// Durable reason and optional result recorded when a task becomes terminal (erased `R = JsonValue`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskOutcome {
	pub status: TaskOutcomeStatus,
	/// Terminal result (`completed`/`failed`/`aborted`).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub result: Option<Value>,
	/// Error snapshot (`failed`/`faulted`).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub error: Option<TaskOutcomeError>,
	/// Cancellation or orphaning reason (`aborted`/`orphaned`).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reason: Option<String>,
}

impl TaskOutcome {
	/// Successfully completed with a result.
	pub fn completed(result: Value) -> Self {
		Self { status: TaskOutcomeStatus::Completed, result: Some(result), error: None, reason: None }
	}
	/// Expected task or domain failure explicitly committed by its implementation.
	pub fn failed(error: TaskOutcomeError) -> Self {
		Self { status: TaskOutcomeStatus::Failed, result: None, error: Some(error), reason: None }
	}
	/// Explicit cancellation handled by the task's abort protocol.
	pub fn aborted(reason: Option<String>) -> Self {
		Self { status: TaskOutcomeStatus::Aborted, result: None, error: None, reason }
	}
	/// Task that cannot resume because its definition or migration is unavailable.
	pub fn orphaned(reason: impl Into<String>) -> Self {
		Self { status: TaskOutcomeStatus::Orphaned, result: None, error: None, reason: Some(reason.into()) }
	}
	/// Runtime-detected contract failure, such as an uncaught throw or no durable progress.
	pub fn faulted(error: TaskOutcomeError) -> Self {
		Self { status: TaskOutcomeStatus::Faulted, result: None, error: Some(error), reason: None }
	}
}

/// Complete durable execution state of a task (erased `S = R = JsonValue`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskState {
	/// Lifecycle status.
	pub status: TaskStateStatus,
	/// Complete durable state from which execution resumes (`pending`/`running`/`waiting`).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub checkpoint: Option<Value>,
	/// Tasks this waiting task waits on (`waiting` only).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub on: Option<Vec<TaskId>>,
	/// How the wait treats its targets (`waiting` only).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub policy: Option<JoinPolicy>,
	/// Outcome (`completing`/`terminal` only).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub outcome: Option<TaskOutcome>,
}

/// Status of a durable task state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TaskStateStatus {
	/// Eligible for scheduling.
	Pending,
	/// Reserved by one in-memory task invocation.
	Running,
	/// Parked until every task in `on` is terminal.
	Waiting,
	/// Outcome decided; becomes terminal once no ordinary owned work below is live.
	Completing,
	/// Permanently settled durable result receipt.
	Terminal,
}

/// How a waiting task treats the tasks it waits on (spec §5.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JoinPolicy {
	FailFast,
	AllSettled,
}

/// Who owns a task: its conversation (a top-level task) or another task of the same conversation (a child task).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum TaskOwnership {
	/// The task's conversation owns it.
	Conversation,
	/// Another task of the same conversation owns it.
	Task {
		#[serde(rename = "taskId")]
		task_id: TaskId,
	},
}

/// Complete replacement record for one durable task state machine (erased generics).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskRecord {
	pub id: TaskId,
	pub conversation_id: ConversationId,
	/// Registered task definition name.
	pub kind: String,
	/// Definition version used to migrate live input and checkpoints.
	pub version: u64,
	/// Original task input retained while the task is live or terminal.
	pub input: Value,
	/// Owning task of a child task; absent for a task its conversation owns. Immutable.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub owner: Option<TaskId>,
	/// Whether this conversation-owned task is excluded from ordinary idle waits, conversation aborts, and cascades.
	pub background: bool,
	/// Durable abort mark checked before run-mode progress is committed.
	pub abort_requested: bool,
	/// Complete durable execution state.
	pub state: TaskState,
	/// Small first-writer-wins values retained while the task can run (`pending`/`running`/`waiting` only).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub memos: Option<JsonObject>,
}

/// Creation options for a durable task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskOptions {
	/// Required: a task always names its owner (spec §5.5).
	pub ownership: TaskOwnership,
	/// Default: the owner task's conversation, or the transaction's bound conversation.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub conversation_id: Option<ConversationId>,
	/// Conversation-owned tasks only: excluded from ordinary idle waits, conversation aborts, and cascades.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub background: Option<bool>,
}

/// Persisted lifecycle record for one create-to-retire document incarnation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentRecord {
	/// Unique incarnation ID; never reused when the same logical document is recreated.
	pub id: DocumentId,
	/// Stable document definition kind.
	pub kind: String,
	/// Family member key; absent for singleton documents.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub key: Option<String>,
	/// Commit that created the incarnation, stamped by storage.
	pub created_at: Seq,
	/// Commit that retired the incarnation; absent while it is current.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub retired_at: Option<Seq>,
	/// Owning scope.
	pub scope: DocumentScope,
	/// History retention (conversation documents only).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub history: Option<DocumentHistory>,
	/// Fork initialization policy (conversation documents only).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub fork: Option<DocumentFork>,
}

/// Fields supplied when storage creates and stamps a new [`DocumentRecord`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentCreate {
	pub id: DocumentId,
	pub kind: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub key: Option<String>,
	pub scope: DocumentScope,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub history: Option<DocumentHistory>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub fork: Option<DocumentFork>,
}

impl DocumentCreate {
	/// Stamp the creation and (optionally) retirement sequences.
	pub fn into_record(self, created_at: Seq, retired_at: Option<Seq>) -> DocumentRecord {
		DocumentRecord {
			id: self.id,
			kind: self.kind,
			key: self.key,
			created_at,
			retired_at,
			scope: self.scope,
			history: self.history,
			fork: self.fork,
		}
	}
}

/// Backend-owned JSON continuation state that callers only round-trip to the same scan.
pub type Cursor = JsonObject;

/// One ordered scan result and its optional continuation state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Page<T> {
	pub items: Vec<T>,
	/// Continuation state; absent when the scan is exhausted.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub next: Option<Cursor>,
}

impl<T> Page<T> {
	pub fn empty() -> Self {
		Self { items: Vec::new(), next: None }
	}
}

/// Optional filters for an ordered conversation scan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationQuery {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub owner_conversation_id: Option<ConversationId>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub owner_task_id: Option<TaskId>,
}

/// Inclusive ID bounds for a newest-first scan of one conversation's fork-aware history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryQuery {
	pub conversation_id: ConversationId,
	/// Oldest entry ID that may be returned.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub min_entry_id: Option<EntryId>,
	/// Newest entry ID that may be returned.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub max_entry_id: Option<EntryId>,
}

/// Optional filters for an ordered scan of durable task records.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskQuery {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub conversation_id: Option<ConversationId>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub kind: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub status: Option<TaskStateStatus>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub abort_requested: Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub background: Option<bool>,
}

/// Optional filters for an ordered scan of submission records.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmissionQuery {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub conversation_id: Option<ConversationId>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub status: Option<SubmissionStatus>,
}

/// Current state or one historical commit sequence used for document membership and content reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentPoint {
	/// One exact commit sequence.
	At(Seq),
	/// The current state.
	Current,
}

impl Serialize for DocumentPoint {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		match self {
			DocumentPoint::At(seq) => seq.serialize(serializer),
			DocumentPoint::Current => serializer.serialize_str("current"),
		}
	}
}

impl<'de> Deserialize<'de> for DocumentPoint {
	fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
		let value = Value::deserialize(deserializer)?;
		match value {
			Value::String(s) if s == "current" => Ok(DocumentPoint::Current),
			Value::Number(n) => n
				.as_u64()
				.map(|v| DocumentPoint::At(Seq(v)))
				.ok_or_else(|| serde::de::Error::custom("expected a non-negative sequence")),
			_ => Err(serde::de::Error::custom("expected a sequence or \"current\"")),
		}
	}
}

/// Exact logical identity of a singleton or one keyed family member.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentAddress {
	/// Stable document definition kind.
	pub kind: String,
	/// Owning scope.
	pub scope: DocumentScope,
	/// Absent selects the singleton; present selects one family member.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub key: Option<String>,
}

/// Ordered scan of document incarnations alive in one exact scope at one point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentQuery {
	pub scope: DocumentScope,
	pub at: DocumentPoint,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub kind: Option<String>,
}

/// Complete checkpoint or Chord operation batch selected by the owning Session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum DocumentContent {
	/// Complete base value.
	Base {
		version: u64,
		value: JsonObject,
	},
	/// Chord operation batch applied after the newest base.
	Delta {
		version: u64,
		ops: Vec<Op>,
	},
}

/// Exact persisted source selected for a definition-free document copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentCopySource {
	pub id: DocumentId,
	pub at: DocumentPoint,
}

/// Detached materialized value and stored definition version at a selected point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredDocument {
	pub record: DocumentRecord,
	/// Stored definition version of `value`.
	pub version: u64,
	/// Materialized detached value.
	pub value: JsonObject,
	/// Deltas replayed after the selected base to materialize `value`.
	pub deltas_since_base: u64,
}

/// One record or document mutation in an atomic storage commit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum StorageWrite {
	#[serde(rename = "conversation")]
	Conversation { value: ConversationRecord },
	#[serde(rename = "entry")]
	Entry { value: EntryRecord },
	#[serde(rename = "task")]
	Task { value: TaskRecord },
	#[serde(rename = "submission")]
	Submission { value: SubmissionRecord },
	#[serde(rename = "document.create")]
	DocumentCreate {
		record: DocumentCreate,
		content: DocumentBase,
	},
	#[serde(rename = "document.copy")]
	DocumentCopy {
		record: DocumentCreate,
		source: DocumentCopySource,
	},
	#[serde(rename = "document.change")]
	DocumentChange {
		id: DocumentId,
		content: DocumentContent,
	},
	#[serde(rename = "document.retire")]
	DocumentRetire { id: DocumentId },
}

/// Complete base content of a new document incarnation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentBase {
	pub version: u64,
	pub value: JsonObject,
}

/// Committed change of one document incarnation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum DocumentCommitChange {
	#[serde(rename = "document")]
	Document {
		record: DocumentRecord,
		/// Conversation owning the document; task documents derive it from their task record. `None` only for Session documents.
		conversation_id: Option<ConversationId>,
		/// Definition version of `value`; absent when this commit retired the incarnation.
		version: Option<u64>,
		/// Exact adopted immutable revision, or `None` (serialized `null`) when this commit retired the incarnation.
		value: Option<JsonObject>,
		/// Exact adopted operations for an ordinary update; empty for creation and retirement.
		ops: Vec<Op>,
	},
	/// Definition-free child initialization; consumers hydrate through state or watch acquisition.
	#[serde(rename = "document.copy")]
	DocumentCopy {
		record: DocumentRecord,
		conversation_id: ConversationId,
		source: DocumentCopySource,
	},
}

/// Complete table record committed without another publication copy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum TableCommitChange {
	#[serde(rename = "conversation")]
	Conversation { value: ConversationRecord },
	#[serde(rename = "entry")]
	Entry { value: EntryRecord },
	#[serde(rename = "task")]
	Task { value: TaskRecord },
	#[serde(rename = "submission")]
	Submission { value: SubmissionRecord },
}

/// Every immutable change from one successful Session commit. Change order is unspecified.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CommitChange {
	Table(TableCommitChange),
	Document(DocumentCommitChange),
}

/// Every immutable change from one successful Session commit, with its sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommitPublication {
	pub seq: Seq,
	pub changes: Vec<CommitChange>,
}

/// Terminal result of one document watch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "camelCase")]
pub enum WatchEnd {
	/// `stop()` was called.
	Stopped,
	/// The acquisition context was cancelled.
	Cancelled,
	/// The owning Session closed.
	SessionClosed,
	/// The document incarnation retired.
	Retired,
	/// The listener itself failed; the watch ends with the error snapshot.
	ListenerError {
		/// Serialized error message.
		error: String,
	},
}

impl WatchEnd {
	/// The `reason` discriminator value, for tests and logging.
	pub fn reason(&self) -> &'static str {
		match self {
			WatchEnd::Stopped => "stopped",
			WatchEnd::Cancelled => "cancelled",
			WatchEnd::SessionClosed => "session_closed",
			WatchEnd::Retired => "retired",
			WatchEnd::ListenerError { .. } => "listener_error",
		}
	}
}

/// Serialized exact-frame observation of an immutable value with bounded pending delivery.
///
/// Upstream is an interface implemented by the Session's watch machinery; the
/// Rust struct is created by [`crate::session`] and holds the sole listener.
pub struct WatchHandle<T> {
	inner: parking_lot::Mutex<watch_inner::Inner<T>>,
}

mod watch_inner {
	use super::WatchEnd;
	use std::future::Future;

	pub(super) type Listener<T> = std::sync::Arc<
		dyn Fn(T, Vec<pi_chord::delta::Op>, &pi_chord::context::Context) -> std::pin::Pin<Box<dyn Future<Output = ()> + Send>>
			+ Send
			+ Sync,
	>;

	pub(super) struct Shared<T> {
		pub value: T,
		pub ops: Vec<pi_chord::delta::Op>,
		pub listener: Option<Listener<T>>,
		pub pending: u64,
	}

	pub(super) enum Inner<T> {
		/// Created; `start` not yet called.
		Idle(Shared<T>),
		/// Running with the sole listener installed.
		Running(Shared<T>),
		/// Terminated; `end` carries the terminal result.
		Stopped { end: WatchEnd, value: T },
	}
}

use watch_inner::{Inner, Shared};

impl<T: Clone + Send + Sync + 'static> WatchHandle<T> {
	/// Create a handle observing `value` as of acquisition (not yet started).
	pub fn new(value: T) -> Self {
		Self {
			inner: parking_lot::Mutex::new(Inner::Idle(Shared {
				value,
				ops: Vec::new(),
				listener: None,
				pending: 0,
			})),
		}
	}

	/// Acquisition revision before start; latest delivered immutable revision afterward.
	pub fn value(&self) -> T {
		match &*self.inner.lock() {
			Inner::Idle(shared) | Inner::Running(shared) => shared.value.clone(),
			Inner::Stopped { value, .. } => value.clone(),
		}
	}

	/// Publish a committed revision. Never invokes the listener inline; unobserved
	/// frames coalesce into one pending delivery (bounded pending delivery).
	pub fn publish(&self, value: T, ops: Vec<pi_chord::delta::Op>) {
		let mut guard = self.inner.lock();
		match &mut *guard {
			Inner::Idle(shared) | Inner::Running(shared) => {
				shared.value = value;
				shared.ops = ops;
			}
			Inner::Stopped { .. } => {}
		}
		if let Inner::Running(shared) = &mut *guard {
			shared.pending += 1;
		}
	}

	/// Install the sole asynchronous listener. Never invokes it inline; the
	/// owning Session pumps frames through [`Self::take_pending_delivery`].
	pub fn start(&self, listener: watch_inner::Listener<T>) {
		let mut guard = self.inner.lock();
		match &mut *guard {
			Inner::Idle(shared) => {
				*guard = Inner::Running(Shared {
					value: shared.value.clone(),
					ops: std::mem::take(&mut shared.ops),
					listener: Some(listener),
					pending: 1,
				});
			}
			Inner::Running(_) => panic!("WatchHandle listener already installed"),
			Inner::Stopped { .. } => panic!("WatchHandle.start() after stop"),
		}
	}

	/// Idempotently stop future callbacks and return this watch's terminal result.
	pub async fn stop(&self) -> WatchEnd {
		let mut guard = self.inner.lock();
		if let Inner::Stopped { end, .. } = &*guard {
			return end.clone();
		}
		let value = match &*guard {
			Inner::Idle(shared) | Inner::Running(shared) => shared.value.clone(),
			Inner::Stopped { .. } => unreachable!(),
		};
		let end = WatchEnd::Stopped;
		*guard = Inner::Stopped { end: end.clone(), value };
		end
	}

	/// Settle when the watch terminates; an already-running callback remains caller-owned.
	pub async fn closed(&self) -> WatchEnd {
		loop {
			{
				let guard = self.inner.lock();
				if let Inner::Stopped { end, .. } = &*guard {
					return end.clone();
				}
			}
			tokio::time::sleep(std::time::Duration::from_millis(1)).await;
		}
	}

	/// Whether a delivery frame is pending (Session pump boundary).
	pub fn has_pending_delivery(&self) -> bool {
		match &*self.inner.lock() {
			Inner::Running(shared) => shared.pending > 0,
			_ => false,
		}
	}

	/// Take one pending frame for asynchronous delivery to the listener.
	pub fn take_pending_delivery(&self) -> Option<(T, Vec<pi_chord::delta::Op>)> {
		let mut guard = self.inner.lock();
		if let Inner::Running(shared) = &mut *guard {
			if shared.pending > 0 {
				shared.pending -= 1;
				return Some((shared.value.clone(), std::mem::take(&mut shared.ops)));
			}
		}
		None
	}

	/// Record a listener failure and terminate the watch with `listener_error`.
	pub fn fail(&self, error: &str) -> WatchEnd {
		let mut guard = self.inner.lock();
		let value = match &*guard {
			Inner::Idle(shared) | Inner::Running(shared) => shared.value.clone(),
			Inner::Stopped { value, .. } => value.clone(),
		};
		let end = WatchEnd::ListenerError { error: error.to_string() };
		*guard = Inner::Stopped { end: end.clone(), value };
		end
	}

	/// Terminate the watch with a terminal reason (`session_closed`, `retired`, `cancelled`).
	pub fn terminate(&self, end: WatchEnd) -> WatchEnd {
		let mut guard = self.inner.lock();
		if let Inner::Stopped { end, .. } = &*guard {
			return end.clone();
		}
		let value = match &*guard {
			Inner::Idle(shared) | Inner::Running(shared) => shared.value.clone(),
			Inner::Stopped { .. } => unreachable!(),
		};
		*guard = Inner::Stopped { end: end.clone(), value };
		end
	}
}

/// Atomic persistence boundary for Session records (upstream `interface Storage`).
///
/// Storage trusts the owning Session to supply semantically valid records, references,
/// ancestry, and transitions. Implementations enforce atomicity, global ID ownership,
/// immutable conversation/entry creation, document record consistency, and detached values;
/// Session serializes commits.
#[async_trait]
pub trait Storage: Send + Sync {
	/// Atomically persist one batch and return its sequence. Once resolved, later reads through this storage observe it.
	async fn commit(&self, writes: &[StorageWrite], context: &Context) -> Result<Seq, StorageRejected>;

	/// Return a fresh candidate from the Session-global numeric ID namespace.
	async fn mint_id(&self) -> u64;

	/// Look up one conversation by exact ID.
	async fn conversation(&self, id: ConversationId, context: &Context) -> Option<ConversationRecord>;

	/// Scan conversations in ascending ID order.
	async fn scan_conversations(
		&self,
		query: &ConversationQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<ConversationRecord>;

	/// Look up one global entry and the sequence of the commit that persisted it.
	async fn entry(&self, id: EntryId, context: &Context) -> Option<(EntryRecord, Seq)>;

	/// Look up one entry only when it is visible through the requested conversation's ancestry.
	async fn entry_in(
		&self,
		conversation_id: ConversationId,
		id: EntryId,
		context: &Context,
	) -> Option<(EntryRecord, Seq)>;

	/// Return the newest visible entry with a `head` at or below the optional inclusive cutoff.
	async fn find_latest_head_marker(
		&self,
		conversation_id: ConversationId,
		at_or_before_entry_id: Option<EntryId>,
		context: &Context,
	) -> Option<EntryRecord>;

	/// Scan the inclusive visible range newest-first, returning at most `limit` entries.
	async fn scan_entries(
		&self,
		query: &EntryQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<EntryRecord>;

	/// Look up the latest complete record for one task.
	async fn task(&self, id: TaskId, context: &Context) -> Option<TaskRecord>;

	/// Scan task records matching every supplied filter.
	async fn scan_tasks(
		&self,
		query: &TaskQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<TaskRecord>;

	/// Look up the latest complete record for one admitted submission.
	async fn submission(&self, id: SubmissionId, context: &Context) -> Option<SubmissionRecord>;

	/// Scan submissions matching every supplied filter in ascending ID order.
	async fn scan_submissions(
		&self,
		query: &SubmissionQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<SubmissionRecord>;

	/// Find a submission by its conversation-scoped host deduplication key.
	async fn submission_by_request(
		&self,
		conversation_id: ConversationId,
		request_id: &str,
		context: &Context,
	) -> Option<SubmissionRecord>;

	/// Resolve the incarnation occupying one exact logical address at the selected point.
	async fn find_document(
		&self,
		address: &DocumentAddress,
		at: DocumentPoint,
		context: &Context,
	) -> Option<DocumentRecord>;

	/// Materialize one specific incarnation by ID at the selected point without following a replacement at its address.
	async fn document(&self, id: DocumentId, at: DocumentPoint, context: &Context)
		-> Option<StoredDocument>;

	/// Scan incarnations alive in one exact scope at the selected point.
	async fn scan_documents(
		&self,
		query: &DocumentQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<DocumentRecord>;

	/// Release backend resources; all later operations must reject.
	async fn close(&self, context: &Context);
}
