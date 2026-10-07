//! Port of `packages/durable/src/session/transaction.ts` (the `Tx` commit
//! callback: table reads/writes, document drafts, submissions, task staging,
//! fork copies, ReadAfterWrite enforcement, and batch assembly/adoption).
//!
//! Deviations (the session module D1 sync-line facade applies — every method
//! takes `&self` over an interior `parking_lot::Mutex`, usable as `&mut Tx`
//! from `commit`/`commit_dyn` and as `&Tx` from async commit jobs):
//! - D-T1: pending-operation tracking (`#track`/`settleFailure` draining) is
//!   unreachable — no operation stays pending after a sync method returns;
//!   `settle_failure` only revokes drafts, and the "callback settled before its
//!   pending Tx operations" error cannot occur.
//! - D-T2: [`DocDraft`] records mutations as an ordered log and replays them
//!   through ONE chord overlay at settlement, so prepared values/ops match
//!   upstream exactly (empty batches suppress, structural no-ops stay
//!   replayable, oversized batches fold to a root replacement).
//! - D-T3: strict-JSON rejections are unrepresentable (`serde_json::Value` is
//!   always strict); read errors are [`TxError`] with upstream-verbatim text.
//! - D-T4: read-only transactions (session `read_on_line`) reject write-path
//!   methods with `Transaction is read-only` (upstream read-on-line jobs only
//!   read).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::Mutex;
use pi_chord::context::Context;
use pi_chord::delta::{Draft, Op, Seg, Tracker};
use serde_json::Value;

use crate::documents::{
	address_id, check_record_scope, check_record_version, document_create, materialize_document_value,
	resolve_address, DocDefinition, DocToken, DocumentAccessError, DocumentRecordRef, ResolvedAddress,
};
use crate::errors::ReadAfterWrite;
use crate::harness::types::AnyTask;
use crate::session::bridge::StorageBridge;
use crate::session::forks::prepare_fork_document_copies;
use crate::session::session::{LoadedDocument, LineState, SessionInner};
use crate::types::{
	ConversationId, ConversationOwnership, ConversationParent, ConversationRecord, Cursor, DocumentAddress,
	DocumentBase, DocumentCommitChange, DocumentContent, DocumentCopySource, DocumentCreate, DocumentFork,
	DocumentId, DocumentPoint, DocumentQuery, DocumentRecord, DocumentScope, EntryDraft, EntryHead, EntryId,
	EntryQuery, EntryRecord, JsonObject, Page, ROOT_CONVERSATION_ID, Seq, StorageWrite, StoredDocument,
	SubmissionCreate, SubmissionId, SubmissionRecord, SubmissionSettlement, SubmissionStatus, SubmissionType,
	TaskId, TaskOptions, TaskQuery, TaskRecord, TaskState, TaskStateStatus,
};

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Failure of one transaction operation; `Display` strings match upstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxError {
	/// Upstream `ReadAfterWrite` — carries the offending method name.
	ReadAfterWrite(ReadAfterWrite),
	/// Any other failure; message verbatim.
	Message(String),
}

impl TxError {
	pub(crate) fn message(message: impl Into<String>) -> Self {
		TxError::Message(message.into())
	}
}

impl std::fmt::Display for TxError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			TxError::ReadAfterWrite(error) => std::fmt::Display::fmt(error, f),
			TxError::Message(message) => f.write_str(message),
		}
	}
}

impl std::error::Error for TxError {}

impl From<DocumentAccessError> for TxError {
	fn from(error: DocumentAccessError) -> Self {
		TxError::Message(error.to_string())
	}
}

// ─── Drafts ──────────────────────────────────────────────────────────────────

/// One recorded draft mutation, replayed through a single chord overlay at
/// settlement (D-T2).
#[derive(Debug, Clone)]
enum Mutation {
	Set { path: Vec<Seg>, value: Value },
	Delete { path: Vec<Seg> },
	Push { path: Vec<Seg>, items: Vec<Value> },
	Pop { path: Vec<Seg> },
	Shift { path: Vec<Seg> },
	Unshift { path: Vec<Seg>, items: Vec<Value> },
	Splice { path: Vec<Seg>, start: usize, remove: Option<usize>, items: Vec<Value> },
	SetIndex { path: Vec<Seg>, index: usize, value: Value },
	SetLength { path: Vec<Seg>, len: usize },
	Reverse { path: Vec<Seg> },
	SortDefault { path: Vec<Seg> },
}

impl Mutation {
	fn apply(self, draft: &mut Draft<'_>) -> Result<(), pi_chord::delta::DeltaError> {
		match self {
			Mutation::Set { path, value } => match path.split_last() {
				Some((Seg::Key(key), parent)) => draft.set(parent, key, value),
				Some((Seg::Index(index), parent)) => draft.set_index(parent, *index, value),
				None => Err(pi_chord::delta::DeltaError::invalid(
					"Cannot replace the document root through a draft",
				)),
			},
			Mutation::Delete { path } => match path.split_last() {
				Some((Seg::Key(key), parent)) => draft.delete(parent, key),
				Some((Seg::Index(_), _)) => Err(pi_chord::delta::DeltaError::invalid("Overlay arrays cannot contain holes")),
				None => Err(pi_chord::delta::DeltaError::invalid(
					"Cannot delete the document root through a draft",
				)),
			},
			Mutation::Push { path, items } => draft.push(&path, items).map(|_| ()),
			Mutation::Pop { path } => draft.pop(&path).map(|_| ()),
			Mutation::Shift { path } => draft.shift(&path).map(|_| ()),
			Mutation::Unshift { path, items } => draft.unshift(&path, items).map(|_| ()),
			Mutation::Splice { path, start, remove, items } => draft.splice(&path, start, remove, items).map(|_| ()),
			Mutation::SetIndex { path, index, value } => draft.set_index(&path, index, value),
			Mutation::SetLength { path, len } => draft.set_length(&path, len),
			Mutation::Reverse { path } => draft.reverse(&path),
			Mutation::SortDefault { path } => draft.sort_default(&path),
		}
	}
}

struct DraftCore {
	base: JsonObject,
	/// Adopted after every call so reads observe the mutations so far.
	work: Tracker,
	log: Vec<Mutation>,
	settled: bool,
}

/// The settled-draft rejection — chord's own message, verbatim.
const SETTLED_DRAFT: &str = "Cannot use a settled overlay";

/// Erased document draft (upstream `Draft<JsonObject>` over the tx's open
/// change): a cloneable handle whose reads reflect the candidate and whose
/// mutations are replayed through one overlay at settlement (D-T2). Every
/// operation rejects after the transaction settled (upstream revokes escaped
/// drafts).
#[derive(Clone)]
pub struct DocDraft {
	core: Arc<Mutex<DraftCore>>,
}

/// Prepared candidate of one draft after settlement (upstream `Prepared`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedDraft {
	pub value: JsonObject,
	pub ops: Vec<Op>,
}

impl DocDraft {
	pub(crate) fn new(base: JsonObject) -> Self {
		let work = Tracker::new(Value::Object(base.clone()));
		Self {
			core: Arc::new(Mutex::new(DraftCore { base, work, log: Vec::new(), settled: false })),
		}
	}

	fn run_mutation(
		&self,
		apply: impl FnOnce(&mut Draft<'_>) -> Result<(), pi_chord::delta::DeltaError>,
		record: impl FnOnce() -> Mutation,
	) -> Result<(), String> {
		let mut core = self.core.lock();
		if core.settled {
			return Err(SETTLED_DRAFT.to_string());
		}
		let mut change = core.work.begin_change();
		apply(&mut change.state()).map_err(|error| error.to_string())?;
		let prepared = change.prepare().map_err(|error| error.to_string())?;
		core.work.adopt(prepared).map_err(|error| error.to_string())?;
		core.log.push(record());
		Ok(())
	}

	fn run_mutation_with<R>(
		&self,
		apply: impl FnOnce(&mut Draft<'_>) -> Result<R, pi_chord::delta::DeltaError>,
		record: impl FnOnce() -> Mutation,
	) -> Result<R, String> {
		let mut core = self.core.lock();
		if core.settled {
			return Err(SETTLED_DRAFT.to_string());
		}
		let mut change = core.work.begin_change();
		let result = apply(&mut change.state()).map_err(|error| error.to_string())?;
		let prepared = change.prepare().map_err(|error| error.to_string())?;
		core.work.adopt(prepared).map_err(|error| error.to_string())?;
		core.log.push(record());
		Ok(result)
	}

	fn simple(&self, mutation: Mutation) -> Result<(), String> {
		let apply = mutation.clone();
		self.run_mutation(move |draft| apply.apply(draft), || mutation)
	}

	/// Full-path write of one value; the final segment is the object key, or an
	/// array index for element writes (upstream proxy property/index writes).
	pub fn set(&self, path: &[Seg], value: Value) -> Result<(), String> {
		if path.is_empty() {
			return Err("Cannot replace the document root through a draft".to_string());
		}
		self.simple(Mutation::Set { path: path.to_vec(), value })
	}

	/// Full-path delete of one object key.
	pub fn delete(&self, path: &[Seg]) -> Result<(), String> {
		if path.is_empty() {
			return Err("Cannot delete the document root through a draft".to_string());
		}
		self.simple(Mutation::Delete { path: path.to_vec() })
	}

	/// Append to the array at `path`; returns the new length.
	pub fn push(&self, path: &[Seg], items: Vec<Value>) -> Result<usize, String> {
		let logged = items.clone();
		self.run_mutation_with(move |draft| draft.push(path, items), move || Mutation::Push {
			path: path.to_vec(),
			items: logged,
		})
	}

	/// Remove and return the last array element.
	pub fn pop(&self, path: &[Seg]) -> Result<Option<Value>, String> {
		self.run_mutation_with(|draft| draft.pop(path), || Mutation::Pop { path: path.to_vec() })
	}

	/// Remove and return the first array element.
	pub fn shift(&self, path: &[Seg]) -> Result<Option<Value>, String> {
		self.run_mutation_with(|draft| draft.shift(path), || Mutation::Shift { path: path.to_vec() })
	}

	/// Prepend to the array at `path`; returns the new length.
	pub fn unshift(&self, path: &[Seg], items: Vec<Value>) -> Result<usize, String> {
		let logged = items.clone();
		self.run_mutation_with(move |draft| draft.unshift(path, items), move || Mutation::Unshift {
			path: path.to_vec(),
			items: logged,
		})
	}

	/// JS `splice` clamping: `remove: None` removes through the end.
	pub fn splice(
		&self,
		path: &[Seg],
		start: usize,
		remove: Option<usize>,
		items: Vec<Value>,
	) -> Result<Vec<Value>, String> {
		let logged = items.clone();
		self.run_mutation_with(
			move |draft| draft.splice(path, start, remove, items),
			move || Mutation::Splice { path: path.to_vec(), start, remove, items: logged },
		)
	}

	/// Write one array element.
	pub fn set_index(&self, path: &[Seg], index: usize, value: Value) -> Result<(), String> {
		self.simple(Mutation::SetIndex { path: path.to_vec(), index, value })
	}

	/// Grow with explicit nulls / shrink (upstream array `length` writes).
	pub fn set_length(&self, path: &[Seg], len: usize) -> Result<(), String> {
		self.simple(Mutation::SetLength { path: path.to_vec(), len })
	}

	/// Reverse the array at `path`.
	pub fn reverse(&self, path: &[Seg]) -> Result<(), String> {
		self.simple(Mutation::Reverse { path: path.to_vec() })
	}

	/// JS default sort (string coercion), like `Draft::sort_default`.
	pub fn sort_default(&self, path: &[Seg]) -> Result<(), String> {
		self.simple(Mutation::SortDefault { path: path.to_vec() })
	}

	/// Overlay-aware read of one path (upstream `draft.get`).
	pub fn get(&self, path: &[Seg]) -> Result<Option<Value>, String> {
		let core = self.core.lock();
		check_settled(&core)?;
		Ok(resolve_json(core.work.value(), path).cloned())
	}

	/// Overlay-aware presence (upstream `draft.has`); arrays answer `"length"`
	/// and canonical indices.
	pub fn has(&self, path: &[Seg], key: &str) -> Result<bool, String> {
		let core = self.core.lock();
		check_settled(&core)?;
		let Some(container) = resolve_json(core.work.value(), path) else {
			return Ok(false);
		};
		Ok(match container {
			Value::Object(object) => object.contains_key(key),
			Value::Array(array) => match key {
				"length" => true,
				key => key.parse::<usize>().map(|index| index < array.len()).unwrap_or(false),
			},
			_ => false,
		})
	}

	/// Array length at `path`; an error for other containers
	/// (upstream `draft.length`).
	pub fn length(&self, path: &[Seg]) -> Result<usize, String> {
		let core = self.core.lock();
		check_settled(&core)?;
		match resolve_json(core.work.value(), path) {
			Some(Value::Array(array)) => Ok(array.len()),
			Some(_) => Err("Only arrays have a length".to_string()),
			None => Err("Cannot resolve the requested path".to_string()),
		}
	}

	/// Array element read (upstream `draft.get_index`).
	pub fn get_index(&self, path: &[Seg], index: usize) -> Result<Option<Value>, String> {
		let core = self.core.lock();
		check_settled(&core)?;
		match resolve_json(core.work.value(), path) {
			Some(Value::Array(array)) => Ok(array.get(index).cloned()),
			Some(_) => Err("Only arrays have indices".to_string()),
			None => Ok(None),
		}
	}

	/// Current candidate value (upstream: reading through the draft's root
	/// proxy).
	pub fn value(&self) -> Result<JsonObject, String> {
		let core = self.core.lock();
		check_settled(&core)?;
		Ok(match core.work.value() {
			Value::Object(object) => object.clone(),
			_ => unreachable!("document drafts always track objects"),
		})
	}

	/// Replay the mutation log through one overlay and revoke the draft
	/// (upstream `change.prepare()` at settlement).
	pub(crate) fn settle(&self) -> Result<PreparedDraft, String> {
		let mut core = self.core.lock();
		if core.settled {
			return Err(SETTLED_DRAFT.to_string());
		}
		core.settled = true;
		let mut tracker = Tracker::new(Value::Object(core.base.clone()));
		let mut change = tracker.begin_change();
		{
			let mut draft = change.state();
			for mutation in std::mem::take(&mut core.log) {
				mutation.apply(&mut draft).map_err(|error| error.to_string())?;
			}
		}
		let prepared = change.prepare().map_err(|error| error.to_string())?;
		let value = match prepared.value {
			Value::Object(object) => object,
			_ => unreachable!("document drafts always track objects"),
		};
		Ok(PreparedDraft { value, ops: prepared.ops })
	}

	/// Revoke the draft without preparing (upstream `change.abort()`).
	pub(crate) fn revoke(&self) {
		self.core.lock().settled = true;
	}
}

fn check_settled(core: &DraftCore) -> Result<(), String> {
	if core.settled {
		Err(SETTLED_DRAFT.to_string())
	} else {
		Ok(())
	}
}

fn resolve_json<'a>(root: &'a Value, path: &[Seg]) -> Option<&'a Value> {
	let mut current = root;
	for segment in path {
		current = match (current, segment) {
			(Value::Object(object), Seg::Key(key)) => object.get(key)?,
			(Value::Array(array), Seg::Index(index)) => array.get(*index)?,
			_ => return None,
		};
	}
	Some(current)
}

// ─── Submission changes ──────────────────────────────────────────────────────

/// A staged submission change: a settlement, or the placement of a queued
/// submission at its entry (upstream `SubmissionChange`).
#[derive(Debug, Clone, PartialEq)]
enum SubmissionChange {
	Settlement(SubmissionSettlement),
	Placed { entry: EntryId },
}

/// Complete record after applying one change (upstream `applySubmissionChange`).
fn apply_submission_change(current: &SubmissionRecord, change: &SubmissionChange) -> Result<SubmissionRecord, String> {
	if matches!(current.status, SubmissionStatus::Done | SubmissionStatus::Unanswered) {
		return Ok(current.clone());
	}
	match change {
		SubmissionChange::Placed { entry } => {
			if current.status != SubmissionStatus::Queued {
				return Err(format!("Submission {} is not queued", current.id));
			}
			let status = match current.submission_type {
				SubmissionType::Input => SubmissionStatus::Placed,
				SubmissionType::Write => SubmissionStatus::Done,
			};
			let mut next = current.clone();
			next.status = status;
			next.entry = Some(*entry);
			Ok(next)
		}
		SubmissionChange::Settlement(settlement) => {
			if matches!(settlement, SubmissionSettlement::Done { .. }) && current.status != SubmissionStatus::Placed {
				return Err(format!("Submission {} is not a placed input", current.id));
			}
			let mut next = current.clone();
			match settlement {
				SubmissionSettlement::Done { answer } => {
					next.status = SubmissionStatus::Done;
					next.answer = Some(*answer);
				}
				SubmissionSettlement::Unanswered { reason, detail } => {
					next.status = SubmissionStatus::Unanswered;
					next.reason = Some(reason.clone());
					next.detail = detail.clone();
				}
			}
			Ok(next)
		}
	}
}

// ─── Transaction state ───────────────────────────────────────────────────────

/// Defaults a commit binds to: `tx.create_task()` conversation and the task
/// attributed to appended entries (upstream `TransactionScope`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransactionScope {
	pub conversation_id: Option<ConversationId>,
	pub task_id: Option<TaskId>,
}

/// Committed and candidate state for one task touched by this transaction
/// (upstream `TransactionTask`).
#[derive(Default)]
struct TransactionTask {
	committed_read: Option<Option<TaskRecord>>,
	write: Option<TaskWrite>,
	publication_conversation_id: Option<ConversationId>,
}

#[derive(Debug, Clone)]
struct TaskWrite {
	kind: TaskWriteKind,
	record: TaskRecord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskWriteKind {
	Create,
	Replace,
}

/// Storage/cache provenance of one staged document incarnation
/// (upstream `DocumentTarget`).
enum DocumentTarget {
	Loaded { document: Arc<Mutex<LoadedDocument>> },
	Created { record: DocumentCreate, version: u64 },
	ForkCopy { record: DocumentCreate, source: DocumentCopySource },
	RetireOnly { record: DocumentRecord },
}

/// One document incarnation acquired, created, or retired by this transaction
/// (upstream `DocumentEntry`; the draft is a synchronous handle).
struct DocumentEntry {
	address_id: String,
	address: DocumentAddress,
	definition: Option<DocDefinition>,
	draft: Option<DocDraft>,
	target: Option<DocumentTarget>,
	prepared: Option<PreparedDraft>,
	retire_on_commit: bool,
}

/// What one staged incarnation writes and publishes, decided once before
/// Storage admission (upstream `DocumentPlan`).
struct DocumentPlan {
	address_id: String,
	record: PlanRecord,
	retire: bool,
	content: Option<PlanContent>,
	change: Option<PlanChange>,
	conversation_id: Option<ConversationId>,
}

enum PlanRecord {
	Create(DocumentCreate),
	Existing(DocumentRecord),
}

#[derive(Clone)]
enum PlanContent {
	Create { record: DocumentCreate, base: DocumentBase },
	Copy { record: DocumentCreate, source: DocumentCopySource },
	Change { id: DocumentId, content: DocumentContent },
}

struct PlanChange {
	prepared: PreparedDraft,
	version: u64,
	loaded: Option<Arc<Mutex<LoadedDocument>>>,
	definition: Option<DocDefinition>,
}

#[derive(Default)]
struct TxState {
	sealed: bool,
	has_table_write: bool,
	writes: Vec<StorageWrite>,
	created_conversation_ids: HashSet<ConversationId>,
	fork_source_conversation_ids: HashSet<ConversationId>,
	fork_source_document_ids: HashSet<DocumentId>,
	tasks: HashMap<TaskId, TransactionTask>,
	task_order: Vec<TaskId>,
	submissions: Vec<SubmissionRecord>,
	submission_changes: Vec<(SubmissionId, SubmissionChange)>,
	documents: Vec<DocumentEntry>,
	latest_document_by_address: HashMap<String, usize>,
	plans: Vec<DocumentPlan>,
}

fn task_entry<'s>(state: &'s mut TxState, id: TaskId) -> &'s mut TransactionTask {
	state.tasks.entry(id).or_default()
}

/// Memoized committed task read; not a caller table read
/// (upstream `#committedTask`).
fn committed_task_locked(
	state: &mut TxState,
	bridge: &StorageBridge,
	context: &Context,
	id: TaskId,
) -> Option<TaskRecord> {
	if let Some(cached) = state.tasks.get(&id).and_then(|task| task.committed_read.clone()) {
		return cached;
	}
	let record = bridge.task(id, context);
	task_entry(state, id).committed_read = Some(record.clone());
	record
}

/// Latest candidate task record, falling back to committed state; not a caller
/// table read (upstream `#currentTask`).
fn current_task_locked(
	state: &mut TxState,
	bridge: &StorageBridge,
	context: &Context,
	id: TaskId,
) -> Option<TaskRecord> {
	if let Some(write) = state.tasks.get(&id).and_then(|task| task.write.as_ref()) {
		return Some(write.record.clone());
	}
	committed_task_locked(state, bridge, context, id)
}

fn status_str(status: TaskStateStatus) -> &'static str {
	match status {
		TaskStateStatus::Pending => "pending",
		TaskStateStatus::Running => "running",
		TaskStateStatus::Waiting => "waiting",
		TaskStateStatus::Completing => "completing",
		TaskStateStatus::Terminal => "terminal",
	}
}

// ─── Transaction ─────────────────────────────────────────────────────────────

/// Transaction for one Session commit callback (upstream `Transaction`).
///
/// Every storage operation runs synchronously on the session line; settlement
/// prepares every open draft and assembles the atomic batch, and adoption
/// applies the committed sequence to the tracker cache.
pub struct Tx<'a> {
	host: &'a SessionInner,
	/// The line's kernel state; the session line job holds the async admission
	/// gate, so mutation through `&self` via the parking_lot lock is exclusive.
	line: &'a parking_lot::Mutex<LineState>,
	context: Context,
	scope: TransactionScope,
	state: Mutex<TxState>,
	read_only: bool,
}

impl<'a> Tx<'a> {
	pub(crate) fn new(
		host: &'a SessionInner,
		line: &'a parking_lot::Mutex<LineState>,
		context: Context,
		scope: TransactionScope,
		read_only: bool,
	) -> Self {
		Self { host, line, context, scope, state: Mutex::new(TxState::default()), read_only }
	}

	fn line(&self) -> parking_lot::MutexGuard<'a, LineState> {
		self.line.lock()
	}

	/// The commit's default scope (`tx.create_task()` conversation + entry
	/// attribution).
	pub fn scope(&self) -> &TransactionScope {
		&self.scope
	}

	// ─── Table reads ──────────────────────────────────────────────────────

	/// Read one conversation (upstream `tx.conversation`).
	pub fn conversation(&self, id: ConversationId) -> Result<Option<ConversationRecord>, TxError> {
		self.read("conversation", |tx| Ok(tx.host.bridge.conversation(id, &tx.context)))
	}

	/// Read one entry (upstream `tx.entry`).
	pub fn entry(&self, id: EntryId) -> Result<Option<EntryRecord>, TxError> {
		self.read("entry", |tx| Ok(tx.host.bridge.entry(id, &tx.context).map(|(entry, _)| entry)))
	}

	/// Read one entry of `kind`; `None` when absent or of another kind
	/// (upstream's typed `entry(token, id)` overload).
	pub fn entry_of_kind(&self, kind: &str, id: EntryId) -> Result<Option<EntryRecord>, TxError> {
		let kind = kind.to_string();
		self.read("entry", move |tx| {
			Ok(tx
				.host
				.bridge
				.entry(id, &tx.context)
				.map(|(entry, _)| entry)
				.filter(|entry| entry.kind == kind))
		})
	}

	/// Read the committed record of one task (upstream `tx.task`).
	pub fn task(&self, id: TaskId) -> Result<Option<TaskRecord>, TxError> {
		self.read("task", move |tx| {
			let mut state = tx.state.lock();
			Ok(committed_task_locked(&mut state, &tx.host.bridge, &tx.context, id))
		})
	}

	/// Scan conversations in ascending ID order (upstream `tx.scanConversations`).
	pub fn scan_conversations(
		&self,
		query: &crate::types::ConversationQuery,
		limit: u64,
		cursor: Option<&Cursor>,
	) -> Result<Page<ConversationRecord>, TxError> {
		let query = query.clone();
		let cursor = cursor.cloned();
		self.read("scanConversations", move |tx| {
			Ok(tx.host.bridge.scan_conversations(&query, limit, cursor.as_ref(), &tx.context))
		})
	}

	/// Scan the visible entry range newest-first (upstream `tx.scanEntries`).
	pub fn scan_entries(
		&self,
		query: &EntryQuery,
		limit: u64,
		cursor: Option<&Cursor>,
	) -> Result<Page<EntryRecord>, TxError> {
		let query = query.clone();
		let cursor = cursor.cloned();
		self.read("scanEntries", move |tx| {
			Ok(tx.host.bridge.scan_entries(&query, limit, cursor.as_ref(), &tx.context))
		})
	}

	/// Newest visible entry carrying a `head` (upstream `tx.latestHeadMarker`).
	pub fn latest_head_marker(&self, conversation_id: ConversationId) -> Result<Option<EntryRecord>, TxError> {
		self.read("latestHeadMarker", move |tx| {
			Ok(tx.host.bridge.find_latest_head_marker(conversation_id, None, &tx.context))
		})
	}

	/// Scan task records (upstream `tx.scanTasks`).
	pub fn scan_tasks(
		&self,
		query: &TaskQuery,
		limit: u64,
		cursor: Option<&Cursor>,
	) -> Result<Page<TaskRecord>, TxError> {
		let query = query.clone();
		let cursor = cursor.cloned();
		self.read("scanTasks", move |tx| {
			Ok(tx.host.bridge.scan_tasks(&query, limit, cursor.as_ref(), &tx.context))
		})
	}

	/// Committed submission record (internal upstream `tx.submission`).
	pub fn submission(&self, id: SubmissionId) -> Result<Option<SubmissionRecord>, TxError> {
		self.read("submission", move |tx| Ok(tx.host.bridge.submission(id, &tx.context)))
	}

	/// Committed submission with a conversation-scoped request ID.
	pub fn submission_by_request(
		&self,
		conversation_id: ConversationId,
		request_id: &str,
	) -> Result<Option<SubmissionRecord>, TxError> {
		let request_id = request_id.to_string();
		self.read("submissionByRequest", move |tx| {
			Ok(tx.host.bridge.submission_by_request(conversation_id, &request_id, &tx.context))
		})
	}

	// ─── Table writes ─────────────────────────────────────────────────────

	/// Create a conversation with explicitly selected ownership.
	pub fn create_conversation(&self, ownership: ConversationOwnership) -> Result<ConversationRecord, TxError> {
		self.write(|tx| tx.stage_conversation(None, ownership, None))
	}

	/// Internal final-form bootstrap path for the reserved root identity.
	pub fn create_root_conversation(&self) -> Result<ConversationRecord, TxError> {
		self.write(|tx| tx.stage_conversation(None, ConversationOwnership::Ownerless, Some(ROOT_CONVERSATION_ID)))
	}

	/// Create a history fork at one concrete visible entry.
	pub fn fork_conversation(
		&self,
		parent_conversation_id: ConversationId,
		at: EntryId,
		ownership: ConversationOwnership,
	) -> Result<ConversationRecord, TxError> {
		self.write(move |tx| {
			tx.stage_conversation(
				Some(ConversationParent { conversation_id: parent_conversation_id, at }),
				ownership,
				None,
			)
		})
	}

	fn stage_conversation(
		&self,
		parent: Option<ConversationParent>,
		ownership: ConversationOwnership,
		reserved_id: Option<ConversationId>,
	) -> Result<ConversationRecord, TxError> {
		let mut state = self.state.lock();
		let owner_task_id = match ownership {
			ConversationOwnership::Ownerless => None,
			ConversationOwnership::Task { task_id } => Some(task_id),
		};
		let id = match reserved_id {
			Some(id) => id,
			None => {
				let id = ConversationId::from(self.host.bridge.mint_id());
				self.assert_open(&state)?;
				id
			}
		};
		let owner = match owner_task_id {
			Some(owner_task_id) => {
				let task = current_task_locked(&mut state, &self.host.bridge, &self.context, owner_task_id);
				self.assert_open(&state)?;
				let Some(task) = task else {
					return Err(TxError::message(format!("Conversation owner task {owner_task_id} does not exist")));
				};
				Some(crate::types::ConversationOwner { conversation_id: task.conversation_id, task_id: owner_task_id })
			}
			None => None,
		};
		let record = ConversationRecord { id, parent: parent.clone(), owner };
		let copies = match &parent {
			Some(parent) => prepare_fork_document_copies(
				&self.host.bridge,
				parent.conversation_id,
				parent.at,
				id,
				&self.context,
			)
			.map_err(TxError::message)?,
			None => Vec::new(),
		};
		self.assert_open(&state)?;
		for copy in copies {
			state.fork_source_document_ids.insert(copy.source.id);
			let address = DocumentAddress {
				kind: copy.record.kind.clone(),
				scope: copy.record.scope.clone(),
				key: copy.record.key.clone(),
			};
			let entry = DocumentEntry {
				address_id: address_id(&address),
				address,
				definition: None,
				draft: None,
				target: Some(DocumentTarget::ForkCopy { record: copy.record, source: copy.source }),
				prepared: None,
				retire_on_commit: false,
			};
			let index = state.documents.len();
			state.latest_document_by_address.insert(entry.address_id.clone(), index);
			state.documents.push(entry);
		}
		if let Some(parent) = &parent {
			state.fork_source_conversation_ids.insert(parent.conversation_id);
		}
		state.created_conversation_ids.insert(id);
		state.writes.push(StorageWrite::Conversation { value: record.clone() });
		drop(state);
		self.host.conversation_created(self, &record)?;
		let state = self.state.lock();
		self.assert_open(&state)?;
		Ok(record)
	}

	/// Append an entry; `head: SelfMarker` becomes the assigned ID and the
	/// entry is attributed to the commit's task (upstream `tx.appendEntry`).
	pub fn append_entry(&self, conversation_id: ConversationId, value: EntryDraft) -> Result<EntryRecord, TxError> {
		self.write(move |tx| {
			let mut state = tx.state.lock();
			tx.require_conversation(&mut state, conversation_id)?;
			tx.assert_open(&state)?;
			let id = EntryId::from(tx.host.bridge.mint_id());
			tx.assert_open(&state)?;
			let kind = value.kind.ok_or_else(|| TxError::message("Entry draft requires a kind"))?;
			let record = EntryRecord {
				id,
				conversation_id,
				kind,
				model: value.model,
				data: value.data,
				head: match value.head {
					Some(EntryHead::SelfMarker(_)) => Some(id),
					Some(EntryHead::Entry(head)) => Some(head),
					None => None,
				},
				edits: value.edits,
				by_task_id: tx.scope.task_id,
			};
			state.writes.push(StorageWrite::Entry { value: record.clone() });
			Ok(record)
		})
	}

	/// Create a durable task (upstream `tx.createTask`).
	pub fn create_task(&self, task: &AnyTask, input: Value, options: TaskOptions) -> Result<TaskId, TxError> {
		self.write(move |tx| {
			let mut state = tx.state.lock();
			let mut owner: Option<TaskRecord> = None;
			if let crate::types::TaskOwnership::Task { task_id } = options.ownership {
				// Validated again against the owner's final candidate during assembly.
				owner = Some(current_task_locked(&mut state, &tx.host.bridge, &tx.context, task_id)
					.ok_or_else(|| TxError::message(format!("Task owner {task_id} does not exist")))?);
				let owner_conversation = owner.as_ref().expect("checked above").conversation_id;
				if options.background == Some(true) {
					return Err(TxError::message("A child task cannot be background"));
				}
				if options.conversation_id.is_some() && options.conversation_id != Some(owner_conversation) {
					return Err(TxError::message(format!(
						"A child task lives in its owner's conversation {owner_conversation}"
					)));
				}
			}
			let conversation_id = match &owner {
				Some(owner) => Some(owner.conversation_id),
				None => options.conversation_id.or(tx.scope.conversation_id),
			};
			let Some(conversation_id) = conversation_id else {
				return Err(TxError::message("Tx.createTask() requires options.conversationId"));
			};
			tx.require_conversation(&mut state, conversation_id)?;
			tx.assert_open(&state)?;
			let definition = &task.definition;
			let checkpoint = (definition.initial)(&input);
			let id = TaskId::from(tx.host.bridge.mint_id());
			tx.assert_open(&state)?;
			let record = TaskRecord {
				id,
				conversation_id,
				kind: definition.name.clone(),
				version: definition.version,
				input,
				owner: owner.as_ref().map(|owner| owner.id),
				background: options.background.unwrap_or(false),
				abort_requested: false,
				state: TaskState {
					status: TaskStateStatus::Pending,
					checkpoint: Some(checkpoint),
					on: None,
					policy: None,
					outcome: None,
				},
				memos: None,
			};
			state.task_order.push(id);
			state.tasks.insert(
				id,
				TransactionTask { write: Some(TaskWrite { kind: TaskWriteKind::Create, record }), ..Default::default() },
			);
			Ok(id)
		})
	}

	/// Create a raw submission record with a fresh ID; no admission rules apply
	/// (upstream `tx.createSubmission`).
	pub fn create_submission(&self, create: SubmissionCreate) -> Result<SubmissionRecord, TxError> {
		self.write(move |tx| {
			let mut state = tx.state.lock();
			tx.require_conversation(&mut state, create.conversation_id)?;
			tx.assert_open(&state)?;
			let id = SubmissionId::from(tx.host.bridge.mint_id());
			tx.assert_open(&state)?;
			let submission_type = create
				.submission_type
				.ok_or_else(|| TxError::message("Submission create requires a type"))?;
			let status = create
				.status
				.ok_or_else(|| TxError::message("Submission create requires a status"))?;
			let record = SubmissionRecord {
				id,
				conversation_id: create.conversation_id,
				request_id: create.request_id,
				submission_type,
				status,
				entry: create.entry,
				answer: create.answer,
				reason: create.reason,
				detail: create.detail,
			};
			state.submissions.push(record.clone());
			Ok(record)
		})
	}

	/// Settle a submission; resolved during assembly against the transaction's
	/// latest candidate record (upstream `tx.settleSubmission`).
	pub fn settle_submission(&self, id: SubmissionId, settlement: SubmissionSettlement) -> Result<(), TxError> {
		let mut state = self.state.lock();
		self.assert_open(&state)?;
		self.check_writable()?;
		state.has_table_write = true;
		state.submission_changes.push((id, SubmissionChange::Settlement(settlement)));
		Ok(())
	}

	/// Place a queued submission at `entry` (upstream `tx.placeSubmission`).
	pub fn place_submission(&self, id: SubmissionId, entry: EntryId) -> Result<(), TxError> {
		let mut state = self.state.lock();
		self.assert_open(&state)?;
		self.check_writable()?;
		state.has_table_write = true;
		state.submission_changes.push((id, SubmissionChange::Placed { entry }));
		Ok(())
	}

	/// Replace one task record completely (upstream `tx.setTask`).
	pub fn set_task(&self, value: TaskRecord) -> Result<(), TxError> {
		let mut state = self.state.lock();
		self.assert_open(&state)?;
		self.check_writable()?;
		state.has_table_write = true;
		let needs_order = state.tasks.get(&value.id).and_then(|task| task.write.as_ref()).is_none();
		if needs_order {
			state.task_order.push(value.id);
		}
		let entry = task_entry(&mut state, value.id);
		if let Some(candidate) = entry.write.as_ref().map(|write| &write.record) {
			if candidate.state.status == TaskStateStatus::Terminal {
				return Err(TxError::message(format!("Task {} already has a terminal candidate", value.id)));
			}
			if candidate.conversation_id != value.conversation_id {
				return Err(TxError::message(format!("Task {} cannot change conversations", value.id)));
			}
		}
		let kind = match entry.write.as_ref().map(|write| write.kind) {
			Some(TaskWriteKind::Create) => TaskWriteKind::Create,
			_ => TaskWriteKind::Replace,
		};
		entry.write = Some(TaskWrite { kind, record: value });
		Ok(())
	}

	/// Candidate records of the tasks this transaction created or replaced.
	pub fn staged_tasks(&self) -> Vec<TaskRecord> {
		let state = self.state.lock();
		state
			.task_order
			.iter()
			.filter_map(|id| state.tasks.get(id).and_then(|task| task.write.as_ref()).map(|write| write.record.clone()))
			.collect()
	}

	/// Conversations this transaction created or forked so far.
	pub fn staged_conversations(&self) -> Vec<ConversationRecord> {
		let state = self.state.lock();
		state
			.writes
			.iter()
			.filter_map(|write| match write {
				StorageWrite::Conversation { value } => Some(value.clone()),
				_ => None,
			})
			.collect()
	}

	// ─── Documents ────────────────────────────────────────────────────────

	/// Acquire the document draft at one address (upstream `tx.doc`; the TS
	/// overloads collapse to `(token, owner, key, seed)`).
	pub fn doc(
		&self,
		token: &DocToken,
		owner: Option<u64>,
		key: Option<&str>,
		seed: Option<Value>,
	) -> Result<DocDraft, TxError> {
		let mut state = self.state.lock();
		self.assert_open(&state)?;
		let definition = token.definition().clone();
		let resolved = resolve_address(&definition, owner, key)?;
		self.assert_task_documents_open(&state, &resolved)?;
		if let Some(&index) = state.latest_document_by_address.get(&resolved.id) {
			if !state.documents[index].retire_on_commit {
				if let Some(draft) = state.documents[index].draft.clone() {
					return Ok(draft);
				}
				if matches!(state.documents[index].target, Some(DocumentTarget::ForkCopy { .. })) {
					return self.acquire_fork_copy(&mut state, index, &definition);
				}
			}
		}
		let skip_load = state
			.latest_document_by_address
			.get(&resolved.id)
			.and_then(|&index| state.documents.get(index))
			.map(|entry| entry.retire_on_commit)
			.unwrap_or(false);
		let seed = if definition.family { seed } else { None };
		let entry = DocumentEntry {
			address_id: resolved.id.clone(),
			address: resolved.address.clone(),
			definition: Some(definition),
			draft: None,
			target: None,
			prepared: None,
			retire_on_commit: false,
		};
		let index = state.documents.len();
		state.latest_document_by_address.insert(entry.address_id.clone(), index);
		state.documents.push(entry);
		self.acquire(&mut state, index, seed, skip_load)
	}

	/// Retire the document at one address (upstream `tx.retireDoc`).
	pub fn retire_doc(&self, token: &DocToken, owner: Option<u64>, key: Option<&str>) -> Result<(), TxError> {
		let mut state = self.state.lock();
		self.assert_open(&state)?;
		let definition = token.definition().clone();
		let resolved = resolve_address(&definition, owner, key)?;
		if let Some(&index) = state.latest_document_by_address.get(&resolved.id) {
			if state.documents[index].retire_on_commit {
				return Ok(());
			}
			if let Some(DocumentTarget::ForkCopy { record, .. }) = &state.documents[index].target {
				check_record_scope(&definition, DocumentRecordRef::Create(record))?;
				state.documents[index].retire_on_commit = true;
				return Ok(());
			}
			if state.documents[index].draft.is_some() {
				// Retirement of an acquired draft persists its final content first.
				state.documents[index].retire_on_commit = true;
				return Ok(());
			}
		}
		let entry = DocumentEntry {
			address_id: resolved.id.clone(),
			address: resolved.address.clone(),
			definition: Some(definition),
			draft: None,
			target: None,
			prepared: None,
			retire_on_commit: true,
		};
		let index = state.documents.len();
		state.latest_document_by_address.insert(entry.address_id.clone(), index);
		state.documents.push(entry);
		self.find_retirement(&mut state, index)
	}

	fn acquire(
		&self,
		state: &mut TxState,
		index: usize,
		seed: Option<Value>,
		skip_load: bool,
	) -> Result<DocDraft, TxError> {
		let definition = state.documents[index]
			.definition
			.clone()
			.expect("document entries acquired through doc carry a definition");
		let loaded = if skip_load {
			None
		} else {
			let mut line = self.line();
			self.host.load_document(
				&mut line,
				&definition,
				&state.documents[index].address_id,
				&state.documents[index].address,
				&self.context,
			)?
		};
		self.assert_open(state)?;
		let draft = if let Some(loaded) = loaded {
			{
				let loaded_guard = loaded.lock();
				check_record_scope(&definition, DocumentRecordRef::Record(&loaded_guard.record))?;
				check_record_version(
					&definition,
					DocumentRecordRef::Record(&loaded_guard.record),
					loaded_guard.stored_version,
				)?;
			}
			state.documents[index].target = Some(DocumentTarget::Loaded { document: loaded.clone() });
			DocDraft::new(loaded.lock().tracker.value().as_object().expect("document values are objects").clone())
		} else {
			let scope = state.documents[index].address.scope.clone();
			match &scope {
				DocumentScope::Conversation { conversation_id } => {
					self.require_conversation(state, *conversation_id)?;
				}
				DocumentScope::Task { task_id } => {
					let task = current_task_locked(state, &self.host.bridge, &self.context, *task_id);
					let Some(task) = task else {
						return Err(TxError::message(format!("Task {} does not exist", task_id)));
					};
					if task.state.status == TaskStateStatus::Terminal {
						return Err(TxError::message(format!("Task {} is terminal", task_id)));
					}
				}
				DocumentScope::Session => {}
			}
			self.assert_open(state)?;
			let value = (definition.initial)(seed.as_ref());
			let id = DocumentId::from(self.host.bridge.mint_id());
			self.assert_open(state)?;
			let record = document_create(&definition, &state.documents[index].address, id);
			state.documents[index].target = Some(DocumentTarget::Created { record, version: definition.version });
			DocDraft::new(value)
		};
		state.documents[index].draft = Some(draft.clone());
		Ok(draft)
	}

	fn acquire_fork_copy(&self, state: &mut TxState, index: usize, definition: &DocDefinition) -> Result<DocDraft, TxError> {
		let (record, source) = match &state.documents[index].target {
			Some(DocumentTarget::ForkCopy { record, source }) => (record.clone(), *source),
			_ => unreachable!("acquire_fork_copy requires a fork-copy target"),
		};
		let stored: Option<StoredDocument> = self.host.bridge.document(source.id, source.at, &self.context);
		self.assert_open(state)?;
		let Some(stored) = stored else {
			return Err(TxError::message(format!("Fork source document {} cannot be read", source.id)));
		};
		if !matches!(stored.record.scope, DocumentScope::Conversation { .. })
			|| stored.record.kind != record.kind
			|| stored.record.key != record.key
			|| stored.record.history != record.history
			|| stored.record.fork != record.fork
		{
			return Err(TxError::message(format!(
				"Fork source document {} does not match the copied record",
				source.id
			)));
		}
		let value = materialize_document_value(
			definition,
			DocumentRecordRef::Record(&stored.record),
			stored.version,
			&stored.value,
		)?;
		state.documents[index].definition = Some(definition.clone());
		state.documents[index].target = Some(DocumentTarget::Created { record, version: definition.version });
		let draft = DocDraft::new(value);
		state.documents[index].draft = Some(draft.clone());
		Ok(draft)
	}

	fn find_retirement(&self, state: &mut TxState, index: usize) -> Result<(), TxError> {
		let definition = state.documents[index]
			.definition
			.clone()
			.expect("retirement entries carry a definition");
		let record = {
			let line = self.line();
			match line.documents.get(&state.documents[index].address_id) {
				Some(document) => Some(document.lock().record.clone()),
				None => self
					.host
					.bridge
					.find_document(&state.documents[index].address, DocumentPoint::Current, &self.context),
			}
		};
		self.assert_open(state)?;
		let Some(record) = record else {
			return Ok(());
		};
		check_record_scope(&definition, DocumentRecordRef::Record(&record))?;
		state.documents[index].target = Some(DocumentTarget::RetireOnly { record });
		Ok(())
	}

	// ─── Settlement ───────────────────────────────────────────────────────

	/// Seal after callback failure: revoke every draft (upstream
	/// `settleFailure`; pending-operation draining is unreachable, D-T1).
	pub(crate) fn settle_failure(&self) {
		let mut state = self.state.lock();
		state.sealed = true;
		for entry in &state.documents {
			if let Some(draft) = &entry.draft {
				draft.revoke();
			}
		}
	}

	/// Seal after callback success, prepare every open change, and assemble the
	/// atomic batch (upstream `settleSuccess`). Any failure revokes every draft
	/// before Storage admission.
	pub(crate) fn settle_success(&self) -> Result<Vec<StorageWrite>, TxError> {
		let mut state = self.state.lock();
		state.sealed = true;
		for index in 0..state.documents.len() {
			if state.documents[index].draft.is_some() && state.documents[index].prepared.is_none() {
				let draft = state.documents[index].draft.clone().expect("checked above");
				match draft.settle() {
					Ok(prepared) => state.documents[index].prepared = Some(prepared),
					Err(error) => {
						for entry in &state.documents {
							if let Some(draft) = &entry.draft {
								draft.revoke();
							}
						}
						return Err(TxError::message(error));
					}
				}
			}
		}
		self.assemble(&mut state)
	}

	/// Abort every prepared change (upstream `discard`).
	pub(crate) fn discard(&self) {
		self.settle_failure();
	}

	/// Adopt every prepared change after Storage success and describe the
	/// publication (upstream `adopt`).
	pub(crate) fn adopt(&self, seq: Seq) -> Result<Vec<DocumentCommitChange>, String> {
		let mut state = self.state.lock();
		let mut line = self.line();
		let plans = std::mem::take(&mut state.plans);
		let mut publications = Vec::new();
		for plan in plans {
			let DocumentPlan {
				address_id,
				record: plan_record,
				retire,
				content: plan_content,
				change: plan_change,
				conversation_id: plan_conversation_id,
			} = plan;
			let committed = matches!(plan_record, PlanRecord::Existing(_));
			let mut record = match plan_record {
				PlanRecord::Create(create) => create.into_record(seq, None),
				PlanRecord::Existing(record) => record,
			};
			if retire {
				record.retired_at = Some(seq);
			}
			if let Some(change) = &plan_change {
				let adopts = match &change.loaded {
					// A loaded incarnation adopts only when it changed.
					Some(_) => !change.prepared.ops.is_empty(),
					// A new incarnation is adopted unless it retires in the same commit.
					None => !retire,
				};
				if adopts {
					match &change.loaded {
						Some(loaded) => {
							let mut guard = loaded.lock();
							let prepared = guard
								.tracker
								.prepare_replace(Value::Object(change.prepared.value.clone()))
								.map_err(|error| error.to_string())?;
							guard.tracker.adopt(prepared).map_err(|error| error.to_string())?;
						}
						None => {
							let tracker = Tracker::new(Value::Object(change.prepared.value.clone()));
							line.documents.insert(
								address_id.clone(),
								Arc::new(Mutex::new(LoadedDocument {
									address_id: address_id.clone(),
									record: record.clone(),
									stored_version: change.version,
									value_version: change.version,
									deltas_since_base: 0,
									tracker,
								})),
							);
						}
					}
				}
				if let Some(loaded) = &change.loaded {
					let mut guard = loaded.lock();
					if guard.stored_version < change.version {
						guard.stored_version = change.version;
					}
					if let Some(PlanContent::Change { content, .. }) = &plan_content {
						match content {
							DocumentContent::Base { .. } => guard.deltas_since_base = 0,
							DocumentContent::Delta { .. } => guard.deltas_since_base += 1,
						}
					}
				}
			}
			let conversation_id = plan_conversation_id;
			if retire {
				if committed {
					let occupant_matches = line
						.documents
						.get(&address_id)
						.map(|document| document.lock().record.id == record.id)
						.unwrap_or(false);
					if occupant_matches {
						line.documents.remove(&address_id);
					}
				}
				publications.push(DocumentCommitChange::Document {
					record,
					conversation_id,
					version: None,
					value: None,
					ops: Vec::new(),
				});
			} else if let Some(PlanContent::Copy { source, .. }) = &plan_content {
				publications.push(DocumentCommitChange::DocumentCopy {
					record,
					conversation_id: conversation_id.expect("copy plans resolve their conversation"),
					source: *source,
				});
			} else if let Some(change) = &plan_change {
				if publishes(retire, &plan_change, &plan_content) {
					let ops = if change.loaded.is_some() { change.prepared.ops.clone() } else { Vec::new() };
					publications.push(DocumentCommitChange::Document {
						record,
						conversation_id,
						version: Some(change.version),
						value: Some(change.prepared.value.clone()),
						ops,
					});
				}
			}
		}
		Ok(publications)
	}

	fn assemble(&self, state: &mut TxState) -> Result<Vec<StorageWrite>, TxError> {
		let mut plans: Vec<DocumentPlan> = (0..state.documents.len()).filter_map(|index| plan_document(state, index)).collect();
		self.reject_fork_source_writes(state, &plans)?;
		self.validate_owners(state)?;
		for id in state.task_order.clone() {
			let Some(write) = state.tasks.get(&id).and_then(|task| task.write.as_ref()) else { continue };
			let (kind, record_conversation_id) = (write.kind, write.record.conversation_id);
			drop(write);
			if kind != TaskWriteKind::Replace {
				continue;
			}
			let committed = committed_task_locked(state, &self.host.bridge, &self.context, id)
				.ok_or_else(|| TxError::message(format!("Task {id} does not exist")))?;
			if committed.state.status == TaskStateStatus::Terminal {
				return Err(TxError::message(format!("Task {id} is already terminal")));
			}
			if committed.conversation_id != record_conversation_id {
				return Err(TxError::message(format!("Task {id} cannot change conversations")));
			}
		}

		// Terminal settlement retires every task document, including ones
		// created by this transaction.
		let terminal_task_ids: Vec<TaskId> = state
			.task_order
			.iter()
			.filter(|id| {
				state
					.tasks
					.get(*id)
					.and_then(|task| task.write.as_ref())
					.map(|write| write.record.state.status == TaskStateStatus::Terminal)
					.unwrap_or(false)
			})
			.copied()
			.collect();
		if !terminal_task_ids.is_empty() {
			let mut retiring: HashSet<DocumentId> = HashSet::new();
			for plan in &mut plans {
				let (record_id, scope) = match &plan.record {
					PlanRecord::Create(record) => (record.id, record.scope.clone()),
					PlanRecord::Existing(record) => (record.id, record.scope.clone()),
				};
				if let DocumentScope::Task { task_id } = &scope {
					if terminal_task_ids.contains(task_id) {
						plan.retire = true;
						retiring.insert(record_id);
					}
				}
			}
			for task_id in terminal_task_ids {
				if state
					.tasks
					.get(&task_id)
					.and_then(|task| task.write.as_ref())
					.map(|write| write.kind == TaskWriteKind::Create)
					.unwrap_or(false)
				{
					continue;
				}
				let mut cursor: Option<Cursor> = None;
				loop {
					let query = DocumentQuery {
						scope: DocumentScope::Task { task_id },
						at: DocumentPoint::Current,
						kind: None,
					};
					let page = self.host.bridge.scan_documents(&query, 256, cursor.as_ref(), &self.context);
					for record in page.items {
						if retiring.contains(&record.id) {
							continue;
						}
						let address = DocumentAddress {
							kind: record.kind.clone(),
							scope: record.scope.clone(),
							key: record.key.clone(),
						};
						retiring.insert(record.id);
						plans.push(DocumentPlan {
							address_id: address_id(&address),
							record: PlanRecord::Existing(record),
							retire: true,
							content: None,
							change: None,
							conversation_id: None,
						});
					}
					cursor = page.next;
					if cursor.is_none() {
						break;
					}
				}
			}
		}

		// Resolve publication ownership before Storage admission so adoption
		// remains synchronous.
		for plan in &mut plans {
			if !publishes(plan.retire, &plan.change, &plan.content) {
				continue;
			}
			let scope = match &plan.record {
				PlanRecord::Create(record) => record.scope.clone(),
				PlanRecord::Existing(record) => record.scope.clone(),
			};
			match &scope {
				DocumentScope::Conversation { conversation_id: owner } => plan.conversation_id = Some(*owner),
				DocumentScope::Task { task_id } => {
					let resolved =
						state.tasks.get(task_id).and_then(|task| task.publication_conversation_id);
					let resolved = match resolved {
						Some(resolved) => Some(resolved),
						None => current_task_locked(state, &self.host.bridge, &self.context, *task_id)
							.map(|current| current.conversation_id),
					};
					if resolved.is_some() {
						task_entry(state, *task_id).publication_conversation_id = resolved;
						plan.conversation_id = resolved;
					}
				}
				DocumentScope::Session => {}
			}
		}

		for (id, change) in std::mem::take(&mut state.submission_changes) {
			let position = state.submissions.iter().position(|record| record.id == id);
			let current = match position {
				Some(index) => state.submissions[index].clone(),
				None => self
					.host
					.bridge
					.submission(id, &self.context)
					.ok_or_else(|| TxError::message(format!("Submission {id} does not exist")))?,
			};
			let next = apply_submission_change(&current, &change).map_err(TxError::message)?;
			if next != current {
				match position {
					Some(index) => state.submissions[index] = next,
					None => state.submissions.push(next),
				}
			}
		}

		let mut writes = std::mem::take(&mut state.writes);
		for submission in &state.submissions {
			writes.push(StorageWrite::Submission { value: submission.clone() });
		}
		for id in &state.task_order {
			if let Some(write) = state.tasks.get(id).and_then(|task| task.write.as_ref()) {
				writes.push(StorageWrite::Task { value: write.record.clone() });
			}
		}
		for plan in &mut plans {
			// Checkpoint predicates run last, after every validation.
			let checkpoint_upgrade = match (&plan.content, &plan.change) {
				(
					Some(PlanContent::Change { content: DocumentContent::Delta { .. }, .. }),
					Some(PlanChange { prepared, version: _, loaded: Some(loaded), definition: Some(definition) }),
				) => {
					let info = crate::types::CheckpointInfo { deltas_since_base: loaded.lock().deltas_since_base };
					definition
						.checkpoint_when
						.as_ref()
						.map(|predicate| predicate(&prepared.value, &prepared.ops, info))
						.unwrap_or(false)
				}
				_ => false,
			};
			if checkpoint_upgrade {
				let change = plan.change.as_ref().expect("checked above");
				plan.content = Some(PlanContent::Change {
					id: plan_record_id(&plan.record),
					content: DocumentContent::Base {
						version: change.version,
						value: change.prepared.value.clone(),
					},
				});
			}
			if let Some(content) = plan.content.clone() {
				writes.push(plan_content_write(content));
			}
			if plan.retire {
				writes.push(StorageWrite::DocumentRetire { id: plan_record_id(&plan.record) });
			}
		}
		state.plans = plans;
		Ok(writes)
	}

	// ─── Helpers ──────────────────────────────────────────────────────────

	/// New owned work needs a live owner, judged on the owner's final candidate
	/// (upstream `#validateOwners`).
	fn validate_owners(&self, state: &mut TxState) -> Result<(), TxError> {
		let mut owners: Vec<(&'static str, TaskId)> = Vec::new();
		for write in &state.writes {
			if let StorageWrite::Conversation { value } = write {
				if let Some(owner) = &value.owner {
					owners.push(("Conversation owner task", owner.task_id));
				}
			}
		}
		for id in &state.task_order {
			if let Some(owner) = state
				.tasks
				.get(id)
				.and_then(|task| task.write.as_ref())
				.filter(|write| write.kind == TaskWriteKind::Create)
				.and_then(|write| write.record.owner)
			{
				owners.push(("Task owner", owner));
			}
		}
		for (what, task_id) in owners {
			let task = current_task_locked(state, &self.host.bridge, &self.context, task_id)
				.ok_or_else(|| TxError::message(format!("{what} {task_id} does not exist")))?;
			if matches!(task.state.status, TaskStateStatus::Terminal | TaskStateStatus::Completing) {
				return Err(TxError::message(format!(
					"{what} {task_id} is {}",
					status_str(task.state.status)
				)));
			}
			if task.abort_requested {
				return Err(TxError::message(format!("{what} {task_id} is abort-marked")));
			}
		}
		Ok(())
	}

	fn reject_fork_source_writes(&self, state: &TxState, plans: &[DocumentPlan]) -> Result<(), TxError> {
		for plan in plans {
			if plan.content.is_none() && !plan.retire {
				continue;
			}
			let (record_id, scope, fork) = match &plan.record {
				PlanRecord::Create(record) => (record.id, record.scope.clone(), record.fork),
				PlanRecord::Existing(record) => (record.id, record.scope.clone(), record.fork),
			};
			if state.fork_source_document_ids.contains(&record_id) {
				return Err(TxError::message(format!(
					"Cannot change fork source document {record_id} in the fork transaction"
				)));
			}
			if let DocumentScope::Conversation { conversation_id } = &scope {
				if state.fork_source_conversation_ids.contains(conversation_id)
					&& fork == Some(DocumentFork::Current)
				{
					return Err(TxError::message(format!(
						"Cannot fork conversation {conversation_id} while changing its current-policy documents"
					)));
				}
			}
		}
		Ok(())
	}

	fn assert_open(&self, state: &TxState) -> Result<(), TxError> {
		if state.sealed {
			Err(TxError::message("Transaction has settled"))
		} else {
			Ok(())
		}
	}

	fn check_writable(&self) -> Result<(), TxError> {
		if self.read_only {
			Err(TxError::message("Transaction is read-only"))
		} else {
			Ok(())
		}
	}

	fn assert_task_documents_open(&self, state: &TxState, resolved: &ResolvedAddress) -> Result<(), TxError> {
		if let DocumentScope::Task { task_id } = &resolved.address.scope {
			let terminal = state
				.tasks
				.get(task_id)
				.and_then(|task| task.write.as_ref())
				.map(|write| write.record.state.status == TaskStateStatus::Terminal)
				.unwrap_or(false);
			if terminal {
				return Err(TxError::message(format!("Task {task_id} is terminal")));
			}
		}
		Ok(())
	}

	fn read<T>(&self, method: &str, read: impl FnOnce(&Self) -> Result<T, TxError>) -> Result<T, TxError> {
		{
			let state = self.state.lock();
			self.assert_open(&state)?;
			if state.has_table_write {
				return Err(TxError::ReadAfterWrite(ReadAfterWrite::new(method)));
			}
		}
		read(self)
	}

	fn write<T>(&self, write: impl FnOnce(&Self) -> Result<T, TxError>) -> Result<T, TxError> {
		{
			let state = self.state.lock();
			self.assert_open(&state)?;
		}
		self.check_writable()?;
		{
			let mut state = self.state.lock();
			state.has_table_write = true;
		}
		write(self)
	}

	fn require_conversation(&self, state: &mut TxState, id: ConversationId) -> Result<(), TxError> {
		if state.created_conversation_ids.contains(&id) {
			return Ok(());
		}
		if self.host.bridge.conversation(id, &self.context).is_none() {
			return Err(TxError::message(format!("Conversation {id} does not exist")));
		}
		Ok(())
	}
}

fn plan_record_id(record: &PlanRecord) -> DocumentId {
	match record {
		PlanRecord::Create(create) => create.id,
		PlanRecord::Existing(record) => record.id,
	}
}

fn plan_content_write(content: PlanContent) -> StorageWrite {
	match content {
		PlanContent::Create { record, base } => StorageWrite::DocumentCreate { record, content: base },
		PlanContent::Copy { record, source } => StorageWrite::DocumentCopy { record, source },
		PlanContent::Change { id, content } => StorageWrite::DocumentChange { id, content },
	}
}

/// Plan of one staged document: its record, content write, and prepared change
/// (upstream `planDocument`).
fn plan_document(state: &TxState, index: usize) -> Option<DocumentPlan> {
	let entry = &state.documents[index];
	let target = entry.target.as_ref()?;
	let mut plan = DocumentPlan {
		address_id: entry.address_id.clone(),
		record: PlanRecord::Existing(DocumentRecord {
			id: DocumentId::from(0),
			kind: String::new(),
			key: None,
			created_at: Seq::from_number(0),
			retired_at: None,
			scope: DocumentScope::Session,
			history: None,
			fork: None,
		}),
		retire: entry.retire_on_commit,
		content: None,
		change: None,
		conversation_id: None,
	};
	match target {
		DocumentTarget::Created { record, version } => {
			let prepared = entry.prepared.clone().expect("created documents settle their draft");
			plan.record = PlanRecord::Create(record.clone());
			plan.content = Some(PlanContent::Create {
				record: record.clone(),
				base: DocumentBase { version: *version, value: prepared.value.clone() },
			});
			plan.change = Some(PlanChange {
				prepared,
				version: *version,
				loaded: None,
				definition: entry.definition.clone(),
			});
		}
		DocumentTarget::ForkCopy { record, source } => {
			plan.record = PlanRecord::Create(record.clone());
			plan.content = Some(PlanContent::Copy { record: record.clone(), source: *source });
		}
		DocumentTarget::RetireOnly { record } => {
			plan.record = PlanRecord::Existing(record.clone());
		}
		DocumentTarget::Loaded { document } => {
			let definition = entry.definition.as_ref().expect("loaded documents carry a definition");
			let prepared = entry.prepared.clone().expect("loaded documents settle their draft");
			let version = definition.version;
			let (record, stored_version) = {
				let guard = document.lock();
				(guard.record.clone(), guard.stored_version)
			};
			// A version change stores a base even without operations; otherwise
			// only a change stores a delta.
			let content = if stored_version < version {
				Some(PlanContent::Change {
					id: record.id,
					content: DocumentContent::Base { version, value: prepared.value.clone() },
				})
			} else if !prepared.ops.is_empty() {
				Some(PlanContent::Change {
					id: record.id,
					content: DocumentContent::Delta { version, ops: prepared.ops.clone() },
				})
			} else {
				None
			};
			plan.record = PlanRecord::Existing(record);
			plan.content = content;
			plan.change = Some(PlanChange {
				prepared,
				version,
				loaded: Some(document.clone()),
				definition: entry.definition.clone(),
			});
		}
	}
	Some(plan)
}

/// Whether adoption publishes the plan (upstream `publishes`): every creation,
/// copy, and retirement, and a loaded incarnation that writes content.
fn publishes(retire: bool, change: &Option<PlanChange>, content: &Option<PlanContent>) -> bool {
	retire
		|| change.as_ref().map(|change| change.loaded.is_none()).unwrap_or(false)
		|| content.is_some()
}
