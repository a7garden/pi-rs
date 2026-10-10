//! Port of `packages/durable/src/storage/memory.ts`.
//!
//! Detached in-memory reference implementation of [`Storage`]. Reads and
//! retained writes are cloned intentionally to match the ownership boundary of
//! serialization-backed stores; this is backend conformance, not validation.
//!
//! Deviations (this module ledger):
//! - D1: upstream `freeze`/`clone` detaching collapses into Rust ownership —
//!   `serde_json::Value` clones are already detached. The upstream test that
//!   asserts a frozen prepared write throws on mutation adapts to asserting
//!   the mutation cannot observe stored state.
//! - D2: upstream `prepareCommit` returns an object with an `apply()` closure;
//!   Rust returns [`PreparedMemoryCommit`] applied through
//!   [`PreparedMemoryCommit::apply`] (which re-locks the state mutex).
//! - D3: the shared [`Storage`] trait surface is infallible for reads, so
//!   upstream throws that cannot cross the trait — `Unknown conversation`,
//!   historical reads of current-only documents, missing required bases, and
//!   reads after `close()` — return `None` / empty pages instead. `commit()`
//!   carries every rejection as [`StorageRejected`] with the exact upstream
//!   message (upstream mixes plain `Error` and `StorageRejected`).
//! - D4: `mint_id` returns `u64`, so the upstream "ID space is exhausted"
//!   throw is unreachable; minting simply continues past `Number.MAX_SAFE_INTEGER`.

use std::collections::HashMap;

use pi_chord::context::Context;
use pi_chord::delta::apply_immutable_batches;
use serde_json::{json, Value};

use crate::errors::StorageRejected;
use crate::storage::scan::{next_cursor, scan_start};
use crate::types::{
    ConversationId, ConversationQuery, ConversationRecord, Cursor, DocumentAddress,
    DocumentContent, DocumentCreate, DocumentId, DocumentPoint, DocumentQuery, DocumentRecord,
    DocumentScope, EntryId, EntryQuery, EntryRecord, Page, ScanOrder, Seq, Storage, StorageWrite,
    StoredDocument, SubmissionId, SubmissionQuery, SubmissionRecord, SubmissionStatus, TaskId,
    TaskQuery, TaskRecord, TaskStateStatus,
};

/// Table owning a global record ID (upstream `TableName`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TableName {
    Conversation,
    Entry,
    Task,
    Submission,
    Document,
}

impl TableName {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            TableName::Conversation => "conversation",
            TableName::Entry => "entry",
            TableName::Task => "task",
            TableName::Submission => "submission",
            TableName::Document => "document",
        }
    }
}

/// One stored revision of a document incarnation (`DocumentContent & { seq }`).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DocumentRevision {
    pub seq: Seq,
    pub content: DocumentContent,
}

#[derive(Debug, Clone)]
struct StoredDocumentState {
    record: DocumentRecord,
    revisions: Vec<DocumentRevision>,
}

/// Fully validated mutation of one document in a prepared commit.
#[derive(Debug, Default, Clone)]
pub(super) struct DocumentAction {
    pub create: Option<DocumentCreate>,
    pub content: Option<DocumentContent>,
    pub retire: bool,
}

#[derive(Debug, Default)]
struct DocumentAddressIndex {
    ids: Vec<DocumentId>,
    current_id: Option<DocumentId>,
}

#[derive(Debug, Default)]
struct State {
    record_types: HashMap<u64, TableName>,
    conversations: HashMap<u64, ConversationRecord>,
    conversation_ids: Vec<ConversationId>,
    conversations_by_owner_conversation: HashMap<u64, Vec<ConversationId>>,
    conversations_by_owner_task: HashMap<u64, Vec<ConversationId>>,
    entries: HashMap<u64, EntryRecord>,
    entry_ids: HashMap<u64, Vec<EntryId>>,
    head_entry_ids: HashMap<u64, Vec<EntryId>>,
    entry_commit_seqs: HashMap<u64, Seq>,
    tasks: HashMap<u64, TaskRecord>,
    task_ids: Vec<TaskId>,
    tasks_by_status: HashMap<TaskStateStatus, Vec<TaskId>>,
    submissions: HashMap<u64, SubmissionRecord>,
    submission_ids: Vec<SubmissionId>,
    submissions_by_status: HashMap<&'static str, Vec<SubmissionId>>,
    submissions_by_request: HashMap<u64, HashMap<String, SubmissionId>>,
    documents: HashMap<u64, StoredDocumentState>,
    document_addresses: HashMap<String, DocumentAddressIndex>,
    documents_by_scope: HashMap<String, Vec<DocumentId>>,
}

#[derive(Debug, Default)]
struct Inner {
    state: State,
    next_id: u64,
    next_seq: u64,
    closed: bool,
}

/// Extract the `after` continuation ID from an opaque scan cursor.
///
/// Cursors are opaque JSON only round-tripped by callers; a malformed cursor is
/// a caller contract violation, so this panics with the upstream message.
pub(super) fn cursor_id(cursor: Option<&Cursor>) -> Option<u64> {
    let after = cursor?.get("after")?;
    Some(
        after
            .as_u64()
            .unwrap_or_else(|| panic!("Invalid storage cursor")),
    )
}

pub(super) fn lower_bound<T>(ids: &[T], target: u64) -> usize
where
    T: Copy + Into<u64>,
{
    ids.partition_point(|id| Into::<u64>::into(*id) < target)
}

pub(super) fn upper_bound<T>(ids: &[T], target: u64) -> usize
where
    T: Copy + Into<u64>,
{
    ids.partition_point(|id| Into::<u64>::into(*id) <= target)
}

/// Stable index key for a submission status (`SubmissionStatus` is not `Hash`).
fn submission_status_key(status: SubmissionStatus) -> &'static str {
    match status {
        SubmissionStatus::Queued => "queued",
        SubmissionStatus::Placed => "placed",
        SubmissionStatus::Done => "done",
        SubmissionStatus::Unanswered => "unanswered",
    }
}

pub(super) fn insert_sorted<T>(ids: &mut Vec<T>, id: T)
where
    T: Copy + Into<u64> + PartialOrd,
{
    if ids.is_empty() || *ids.last().unwrap() < id {
        ids.push(id);
    } else {
        ids.insert(lower_bound(ids, id.into()), id);
    }
}

pub(super) fn remove_sorted<T>(ids: &mut Vec<T>, id: T)
where
    T: Copy + Into<u64> + PartialEq,
{
    let index = lower_bound(ids, id.into());
    if ids.get(index) == Some(&id) {
        ids.remove(index);
    }
}

pub(super) fn insert_map_id<K, T>(index: &mut HashMap<K, Vec<T>>, key: K, id: T)
where
    K: std::hash::Hash + Eq,
    T: Copy + Into<u64> + PartialOrd + PartialEq,
{
    insert_sorted(index.entry(key).or_default(), id);
}

pub(super) fn scope_key(scope: &DocumentScope) -> Value {
    match scope {
        DocumentScope::Session => json!(["session"]),
        DocumentScope::Conversation { conversation_id } => {
            json!(["conversation", conversation_id.0])
        }
        DocumentScope::Task { task_id } => json!(["task", task_id.0]),
    }
}

pub(super) fn address_key(kind: &str, scope: &DocumentScope, key: Option<&str>) -> String {
    let key_part = match key {
        Some(key) => json!(["family", key]),
        None => json!(["singleton"]),
    };
    json!([kind, scope_key(scope), key_part]).to_string()
}

pub(super) fn record_address_key(record: &DocumentRecord) -> String {
    address_key(&record.kind, &record.scope, record.key.as_deref())
}

pub(super) fn create_address_key(record: &DocumentCreate) -> String {
    address_key(&record.kind, &record.scope, record.key.as_deref())
}

pub(super) fn is_alive_at(record: &DocumentRecord, at: DocumentPoint) -> bool {
    match at {
        DocumentPoint::Current => record.retired_at.is_none(),
        DocumentPoint::At(seq) => {
            record.created_at <= seq && record.retired_at.map_or(true, |retired| seq < retired)
        }
    }
}

pub(super) fn scope_kind(scope: &DocumentScope) -> &'static str {
    match scope {
        DocumentScope::Session => "session",
        DocumentScope::Conversation { .. } => "conversation",
        DocumentScope::Task { .. } => "task",
    }
}

pub(super) fn content_version(content: &DocumentContent) -> u64 {
    match content {
        DocumentContent::Base { version, .. } | DocumentContent::Delta { version, .. } => *version,
    }
}

pub(super) fn is_current_only(record: &DocumentRecord) -> bool {
    scope_kind(&record.scope) != "conversation"
        || record.history == Some(crate::types::DocumentHistory::Latest)
}

pub(super) fn create_is_current_only(record: &DocumentCreate) -> bool {
    scope_kind(&record.scope) != "conversation"
        || record.history == Some(crate::types::DocumentHistory::Latest)
}

fn page<T>(mut values: Vec<T>, id_of: impl Fn(&T) -> u64, limit: u64, order: ScanOrder) -> Page<T> {
    if values.len() as u64 <= limit {
        return Page {
            items: values,
            next: None,
        };
    }
    values.truncate(limit as usize);
    let after = id_of(values.last().unwrap());
    Page {
        items: values,
        next: Some(next_cursor(after, order)),
    }
}

/// Indexes of sorted `ids` in scan order, after the cursor's ID when there is one (upstream `scanIndexes`).
fn scan_indexes<T>(ids: &[T], start: super::scan::ScanStart) -> impl Iterator<Item = usize>
where
    T: Copy + Into<u64>,
{
    match start.order {
        ScanOrder::Ascending => {
            let first = start.after.map_or(0, |after| upper_bound(ids, after));
            Box::new(first..ids.len()) as Box<dyn Iterator<Item = usize>>
        }
        ScanOrder::Descending => {
            let end = start
                .after
                .map_or(ids.len(), |after| lower_bound(ids, after));
            Box::new((0..end).rev()) as Box<dyn Iterator<Item = usize>>
        }
    }
}

/// A fully validated, detached state mutation whose application performs no
/// fallible preparation (upstream `PreparedMemoryCommit`).
pub struct PreparedMemoryCommit {
    /// Sequence this commit occupies.
    pub seq: Seq,
    /// Deeply detached writes for persistence.
    pub writes: Vec<StorageWrite>,
    /// A second deep clone: `apply` must persist the state validated at
    /// prepare time even if `writes` is mutated afterwards (upstream froze
    /// prepared writes; D1 covers this with an untouched copy).
    detached: Vec<StorageWrite>,
    pub(super) actions: Vec<(DocumentId, DocumentAction)>,
    applied: bool,
}

impl PreparedMemoryCommit {
    /// Apply the prepared commit; idempotent, returns its sequence.
    pub fn apply(&mut self, storage: &MemoryStorage) -> Seq {
        if !self.applied {
            self.applied = true;
            let mut inner = storage.inner.lock();
            storage.apply_locked(&mut inner, &self.detached, &self.actions, self.seq);
        }
        self.seq
    }
}

/// Detached in-memory reference implementation of [`Storage`].
///
/// Shareable via `Arc`; every method takes `&self` and all state sits behind a
/// `parking_lot::Mutex`.
pub struct MemoryStorage {
    inner: parking_lot::Mutex<Inner>,
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStorage {
    /// Create an empty storage.
    pub fn new() -> Self {
        Self {
            inner: parking_lot::Mutex::new(Inner {
                next_id: 2,
                next_seq: 1,
                ..Default::default()
            }),
        }
    }

    /// Validate and detach one commit at the current sequence without changing
    /// observable state.
    pub fn prepare_commit(
        &self,
        writes: &[StorageWrite],
    ) -> Result<PreparedMemoryCommit, StorageRejected> {
        let seq = Seq::from_number(self.inner.lock().next_seq);
        self.prepare_commit_at(writes, seq)
    }

    /// Validate and detach one commit at an explicit sequence (recovery path).
    pub fn prepare_commit_at(
        &self,
        writes: &[StorageWrite],
        seq: Seq,
    ) -> Result<PreparedMemoryCommit, StorageRejected> {
        let mut inner = self.inner.lock();
        self.assert_open(&inner)?;
        if seq < Seq::from_number(inner.next_seq) {
            return Err(StorageRejected::new(format!(
                "Commit sequence {} does not strictly increase",
                seq
            )));
        }
        let detached_writes = Self::resolve_document_copies(&inner.state, writes.to_vec())?;
        Self::check_global_ids(&inner.state, &detached_writes)?;
        let actions = Self::prepare_document_actions(&detached_writes)?;
        Self::check_document_actions(&inner.state, &actions)?;
        Ok(PreparedMemoryCommit {
            detached: detached_writes.clone(),
            seq,
            writes: detached_writes,
            actions,
            applied: false,
        })
    }

    fn assert_open(&self, inner: &Inner) -> Result<(), StorageRejected> {
        if inner.closed {
            return Err(StorageRejected::new("MemoryStorage is closed"));
        }
        Ok(())
    }

    /// Resolve `document.copy` writes into `document.create` writes by
    /// materializing their sources (upstream `resolveDocumentCopies`).
    fn resolve_document_copies(
        state: &State,
        mut writes: Vec<StorageWrite>,
    ) -> Result<Vec<StorageWrite>, StorageRejected> {
        if !writes
            .iter()
            .any(|write| matches!(write, StorageWrite::DocumentCopy { .. }))
        {
            return Ok(writes);
        }
        let mut changed_document_ids = std::collections::HashSet::new();
        for write in &writes {
            match write {
                StorageWrite::DocumentCreate { record, .. }
                | StorageWrite::DocumentCopy { record, .. } => {
                    changed_document_ids.insert(record.id);
                }
                StorageWrite::DocumentChange { id, .. } | StorageWrite::DocumentRetire { id } => {
                    changed_document_ids.insert(*id);
                }
                _ => {}
            }
        }
        let resolved = writes
            .iter_mut()
            .map(|write| match write {
                StorageWrite::DocumentCopy { record, source } => {
                    let wrap = |message: String| {
                        StorageRejected::new(format!("Document copy {} was rejected", record.id.0))
                    };
                    if changed_document_ids.contains(&source.id) {
                        return Err(wrap("changed".into()));
                    }
                    let stored = Self::materialize_document(state, source.id, source.at)
                        .ok_or_else(|| wrap("unreadable".into()))?;
                    if scope_kind(&stored.record.scope) != "conversation"
                        || scope_kind(&record.scope) != "conversation"
                        || stored.record.kind != record.kind
                        || stored.record.key != record.key
                        || stored.record.history != record.history
                        || stored.record.fork != record.fork
                    {
                        return Err(wrap("history mismatch".into()));
                    }
                    *write = StorageWrite::DocumentCreate {
                        record: record.clone(),
                        content: crate::types::DocumentBase {
                            version: stored.version,
                            value: stored.value,
                        },
                    };
                    Ok(())
                }
                _ => Ok(()),
            })
            .collect::<Result<Vec<_>, StorageRejected>>();
        resolved?;
        Ok(writes)
    }

    fn apply_locked(
        &self,
        inner: &mut Inner,
        prepared: &[StorageWrite],
        actions: &[(DocumentId, DocumentAction)],
        seq: Seq,
    ) -> Seq {
        let state = &mut inner.state;
        for write in prepared {
            match write {
                StorageWrite::Conversation { value } => {
                    state
                        .record_types
                        .insert(value.id.0, TableName::Conversation);
                    state.conversations.insert(value.id.0, value.clone());
                    insert_sorted(&mut state.conversation_ids, value.id);
                    if let Some(owner) = &value.owner {
                        insert_map_id(
                            &mut state.conversations_by_owner_conversation,
                            owner.conversation_id.0,
                            value.id,
                        );
                        insert_map_id(
                            &mut state.conversations_by_owner_task,
                            owner.task_id.0,
                            value.id,
                        );
                    }
                    inner.next_id = inner.next_id.max(value.id.0 + 1);
                }
                StorageWrite::Entry { value } => {
                    state.record_types.insert(value.id.0, TableName::Entry);
                    state.entries.insert(value.id.0, value.clone());
                    state.entry_commit_seqs.insert(value.id.0, seq);
                    insert_sorted(
                        state.entry_ids.entry(value.conversation_id.0).or_default(),
                        value.id,
                    );
                    if value.head.is_some() {
                        insert_sorted(
                            state
                                .head_entry_ids
                                .entry(value.conversation_id.0)
                                .or_default(),
                            value.id,
                        );
                    }
                    inner.next_id = inner.next_id.max(value.id.0 + 1);
                }
                StorageWrite::Task { value } => {
                    state.record_types.insert(value.id.0, TableName::Task);
                    match state.tasks.get(&value.id.0) {
                        None => {
                            insert_sorted(&mut state.task_ids, value.id);
                            insert_sorted(
                                state.tasks_by_status.entry(value.state.status).or_default(),
                                value.id,
                            );
                        }
                        Some(previous) if previous.state.status != value.state.status => {
                            remove_sorted(
                                state
                                    .tasks_by_status
                                    .entry(previous.state.status)
                                    .or_default(),
                                value.id,
                            );
                            insert_sorted(
                                state.tasks_by_status.entry(value.state.status).or_default(),
                                value.id,
                            );
                        }
                        _ => {}
                    }
                    state.tasks.insert(value.id.0, value.clone());
                    inner.next_id = inner.next_id.max(value.id.0 + 1);
                }
                StorageWrite::Submission { value } => {
                    state.record_types.insert(value.id.0, TableName::Submission);
                    let previous = state.submissions.get(&value.id.0).cloned();
                    match &previous {
                        None => {
                            insert_sorted(&mut state.submission_ids, value.id);
                            insert_sorted(
                                state
                                    .submissions_by_status
                                    .entry(submission_status_key(value.status))
                                    .or_default(),
                                value.id,
                            );
                        }
                        Some(previous) if previous.status != value.status => {
                            remove_sorted(
                                state
                                    .submissions_by_status
                                    .entry(submission_status_key(previous.status))
                                    .or_default(),
                                value.id,
                            );
                            insert_sorted(
                                state
                                    .submissions_by_status
                                    .entry(submission_status_key(value.status))
                                    .or_default(),
                                value.id,
                            );
                        }
                        _ => {}
                    }
                    if let Some(previous) = &previous {
                        if let Some(request_id) = &previous.request_id {
                            let mut remove_key = false;
                            if let Some(requests) = state
                                .submissions_by_request
                                .get_mut(&previous.conversation_id.0)
                            {
                                if requests.get(request_id) == Some(&value.id) {
                                    requests.remove(request_id);
                                    remove_key = requests.is_empty();
                                }
                            }
                            if remove_key {
                                state
                                    .submissions_by_request
                                    .remove(&previous.conversation_id.0);
                            }
                        }
                    }
                    state.submissions.insert(value.id.0, value.clone());
                    if let Some(request_id) = &value.request_id {
                        state
                            .submissions_by_request
                            .entry(value.conversation_id.0)
                            .or_default()
                            .insert(request_id.clone(), value.id);
                    }
                    inner.next_id = inner.next_id.max(value.id.0 + 1);
                }
                StorageWrite::DocumentCopy { .. } => {
                    unreachable!("Prepared document copy was not resolved")
                }
                StorageWrite::DocumentCreate { .. }
                | StorageWrite::DocumentChange { .. }
                | StorageWrite::DocumentRetire { .. } => {}
            }
        }
        Self::apply_document_actions(state, actions, seq, &mut inner.next_id);
        inner.next_seq = seq.0 + 1;
        seq
    }

    /// Materialize one incarnation by ID at the selected point.
    ///
    /// Upstream throws when the document is current-only and a historical point
    /// is requested, or when the stored revision list is missing its base; the
    /// fixed read surface collapses both to `None` (D3) — callers cannot
    /// distinguish them, matching "no readable value".
    pub(super) fn materialize_document(
        state: &State,
        id: DocumentId,
        at: DocumentPoint,
    ) -> Option<StoredDocument> {
        let stored = state.documents.get(&id.0)?;
        if at != DocumentPoint::Current && is_current_only(&stored.record) {
            return None;
        }
        if !is_alive_at(&stored.record, at) {
            return None;
        }
        let upper = match at {
            DocumentPoint::Current => None,
            DocumentPoint::At(seq) => Some(seq),
        };
        let revisions: Vec<&DocumentRevision> = stored
            .revisions
            .iter()
            .filter(|revision| upper.map_or(true, |seq| revision.seq <= seq))
            .collect();
        let base_index = revisions
            .iter()
            .rposition(|revision| matches!(revision.content, DocumentContent::Base { .. }))?;
        let base = revisions[base_index].content.clone();
        let DocumentContent::Base { version, value } = &base else {
            return None;
        };
        let mut batches: Vec<&[pi_chord::delta::Op]> = Vec::new();
        for revision in &revisions[base_index + 1..] {
            match &revision.content {
                DocumentContent::Delta {
                    version: revision_version,
                    ops,
                } if *revision_version == *version => {
                    batches.push(ops);
                }
                _ => return None,
            }
        }
        let applied = apply_immutable_batches(&Value::Object(value.clone()), batches).ok()?;
        Some(StoredDocument {
            record: stored.record.clone(),
            version: *version,
            value: applied
                .as_object()
                .expect("document base is an object")
                .clone(),
            deltas_since_base: (revisions.len() - base_index - 1) as u64,
        })
    }

    fn check_global_ids(state: &State, writes: &[StorageWrite]) -> Result<(), StorageRejected> {
        let mut claimed: HashMap<u64, TableName> = HashMap::new();
        for write in writes {
            let (table, id) = match write {
                StorageWrite::DocumentChange { .. } | StorageWrite::DocumentRetire { .. } => {
                    continue
                }
                StorageWrite::DocumentCreate { record, .. }
                | StorageWrite::DocumentCopy { record, .. } => (TableName::Document, record.id.0),
                StorageWrite::Conversation { value } => (TableName::Conversation, value.id.0),
                StorageWrite::Entry { value } => (TableName::Entry, value.id.0),
                StorageWrite::Task { value } => (TableName::Task, value.id.0),
                StorageWrite::Submission { value } => (TableName::Submission, value.id.0),
            };
            let existing = claimed.get(&id).copied();
            // The global record-ID namespace is cross-commit: immutable tables
            // reject any stored ID; replaceable tables reject cross-table reuse.
            if let Some(stored) = state.record_types.get(&id) {
                if *stored != table || matches!(table, TableName::Conversation | TableName::Entry) {
                    return Err(StorageRejected::new(format!(
                        "ID {} already belongs to {}",
                        id,
                        stored.as_str()
                    )));
                }
            }
            if matches!(
                table,
                TableName::Conversation | TableName::Entry | TableName::Document
            ) {
                if let Some(existing) = existing {
                    return Err(StorageRejected::new(format!(
                        "ID {} already belongs to {}",
                        id,
                        existing.as_str()
                    )));
                }
                if existing.is_some() {
                    return Err(StorageRejected::new(format!(
                        "ID {} is written more than once",
                        id
                    )));
                }
            } else {
                if let Some(existing) = existing {
                    if existing != table {
                        return Err(StorageRejected::new(format!(
                            "ID {} already belongs to {}",
                            id,
                            existing.as_str()
                        )));
                    }
                }
                if let Some(existing) = existing {
                    if existing != table {
                        return Err(StorageRejected::new(format!(
                            "ID {} is written as two record types",
                            id
                        )));
                    }
                }
            }
            claimed.insert(id, table);
        }
        Ok(())
    }

    fn prepare_document_actions(
        writes: &[StorageWrite],
    ) -> Result<Vec<(DocumentId, DocumentAction)>, StorageRejected> {
        let mut order: Vec<DocumentId> = Vec::new();
        let mut actions: HashMap<DocumentId, DocumentAction> = HashMap::new();
        for write in writes {
            let (id, apply): (
                DocumentId,
                Box<dyn Fn(&mut DocumentAction) -> Result<(), StorageRejected>>,
            ) = match write {
                StorageWrite::DocumentCreate { record, content } => (
                    record.id,
                    Box::new(move |action| {
                        if action.create.is_some() || action.content.is_some() {
                            return Err(StorageRejected::new(format!(
                                "Document {} has more than one content command",
                                record.id.0
                            )));
                        }
                        action.create = Some(record.clone());
                        action.content = Some(DocumentContent::Base {
                            version: content.version,
                            value: content.value.clone(),
                        });
                        Ok(())
                    }),
                ),
                StorageWrite::DocumentChange { id, content } => (
                    *id,
                    Box::new(move |action| {
                        if action.content.is_some() {
                            return Err(StorageRejected::new(format!(
                                "Document {} has more than one content command",
                                id.0
                            )));
                        }
                        action.content = Some(content.clone());
                        Ok(())
                    }),
                ),
                StorageWrite::DocumentRetire { id } => (
                    *id,
                    Box::new(move |action| {
                        if action.retire {
                            return Err(StorageRejected::new(format!(
                                "Document {} is retired more than once",
                                id.0
                            )));
                        }
                        action.retire = true;
                        Ok(())
                    }),
                ),
                _ => continue,
            };
            if !actions.contains_key(&id) {
                order.push(id);
                actions.insert(id, DocumentAction::default());
            }
            apply(actions.get_mut(&id).unwrap())?;
        }
        Ok(order
            .into_iter()
            .map(|id| (id, actions.remove(&id).unwrap()))
            .collect())
    }

    fn check_document_actions(
        state: &State,
        actions: &[(DocumentId, DocumentAction)],
    ) -> Result<(), StorageRejected> {
        let mut live_counts: HashMap<String, i64> = HashMap::new();
        for (id, action) in actions {
            let existing = state.documents.get(&id.0);
            if action.create.is_none() && existing.is_none() {
                return Err(StorageRejected::new(format!("Unknown document: {}", id.0)));
            }
            if action.create.is_some() && existing.is_some() {
                return Err(StorageRejected::new(format!(
                    "Document {} already exists",
                    id.0
                )));
            }
            if existing.map_or(false, |stored| stored.record.retired_at.is_some()) {
                return Err(StorageRejected::new(format!(
                    "Document {} is retired",
                    id.0
                )));
            }
            if let Some(DocumentContent::Delta { version, .. }) = &action.content {
                let Some(previous) = existing.and_then(|stored| stored.revisions.last()) else {
                    return Err(StorageRejected::new(format!(
                        "Document {} delta has no base",
                        id.0
                    )));
                };
                if content_version(&previous.content) != *version {
                    return Err(StorageRejected::new(format!(
                        "Document {} version transition requires a base",
                        id.0
                    )));
                }
            }
            let key = match &action.create {
                None => record_address_key(&existing.unwrap().record),
                Some(create) => create_address_key(create),
            };
            let current_id = state
                .document_addresses
                .get(&key)
                .and_then(|index| index.current_id);
            let live = live_counts
                .entry(key)
                .or_insert(if current_id.is_none() { 0 } else { 1 });
            if action.retire && current_id == Some(*id) {
                *live -= 1;
            }
            if action.create.is_some() && !action.retire {
                *live += 1;
            }
        }
        for live in live_counts.values() {
            if *live > 1 {
                return Err(StorageRejected::new(
                    "Document address already has a current incarnation",
                ));
            }
        }
        Ok(())
    }

    fn apply_document_actions(
        state: &mut State,
        actions: &[(DocumentId, DocumentAction)],
        seq: Seq,
        next_id: &mut u64,
    ) {
        for (id, action) in actions {
            let mut stored = state.documents.get_mut(&id.0);
            if let Some(create) = &action.create {
                let record = create
                    .clone()
                    .into_record(seq, action.retire.then_some(seq));
                let revision = DocumentRevision {
                    seq,
                    content: action.content.clone().expect("create carries content"),
                };
                state.record_types.insert(id.0, TableName::Document);
                state.documents.insert(
                    id.0,
                    StoredDocumentState {
                        record: record.clone(),
                        revisions: vec![revision],
                    },
                );
                stored = state.documents.get_mut(&id.0);

                let key = record_address_key(&record);
                let address = state.document_addresses.entry(key).or_default();
                insert_sorted(&mut address.ids, *id);

                let scope = scope_key(&record.scope).to_string();
                insert_sorted(state.documents_by_scope.entry(scope).or_default(), *id);
                *next_id = (*next_id).max(id.0 + 1);
            } else if let Some(content) = &action.content {
                let stored = stored.as_mut().unwrap();
                let revision = DocumentRevision {
                    seq,
                    content: content.clone(),
                };
                if matches!(revision.content, DocumentContent::Base { .. })
                    && is_current_only(&stored.record)
                {
                    stored.revisions = vec![revision];
                } else {
                    stored.revisions.push(revision);
                }
            }

            let stored = state
                .documents
                .get_mut(&id.0)
                .expect("document action has a stored state");
            if action.retire && action.create.is_none() {
                stored.record.retired_at = Some(seq);
            }
            if action.retire && is_current_only(&stored.record) {
                stored.revisions.clear();
            }
            if action.create.is_some() || action.retire {
                let key = record_address_key(&stored.record);
                let address = state
                    .document_addresses
                    .get_mut(&key)
                    .expect("address exists");
                if action.retire && address.current_id == Some(*id) {
                    address.current_id = None;
                }
                if action.create.is_some() && !action.retire {
                    address.current_id = Some(*id);
                }
            }
        }
    }

    /// Fork-aware newest-first walk of the entries visible in one conversation.
    pub(super) fn visible_entries(
        state: &State,
        conversation_id: ConversationId,
        min_entry_id: u64,
        max_entry_id: u64,
        cap: Option<usize>,
    ) -> Result<Vec<EntryRecord>, StorageRejected> {
        if !state.conversations.contains_key(&conversation_id.0) {
            // Upstream throws `Unknown conversation`; the fixed read surface
            // cannot (D3) — callers treat unknown conversations as empty.
            return Ok(Vec::new());
        }
        let mut values = Vec::new();
        let mut current_id = conversation_id;
        let mut upper_entry_id = max_entry_id;
        loop {
            let ids = state
                .entry_ids
                .get(&current_id.0)
                .cloned()
                .unwrap_or_default();
            for index in (0..upper_bound(&ids, upper_entry_id)).rev() {
                let id = ids[index];
                if id.0 < min_entry_id {
                    break;
                }
                values.push(state.entries.get(&id.0).unwrap().clone());
                if let Some(cap) = cap {
                    if values.len() > cap {
                        return Ok(values);
                    }
                }
            }
            let conversation = state.conversations.get(&current_id.0).unwrap();
            let Some(parent) = &conversation.parent else {
                break;
            };
            upper_entry_id = upper_entry_id.min(parent.at.0);
            if upper_entry_id < min_entry_id {
                break;
            }
            current_id = parent.conversation_id;
        }
        Ok(values)
    }

    /// Visible entries oldest first: the fork chain's segments from the root conversation forward.
    pub(super) fn visible_entries_ascending(
        state: &State,
        conversation_id: ConversationId,
        min_entry_id: u64,
        max_entry_id: u64,
        cap: Option<usize>,
    ) -> Result<Vec<EntryRecord>, StorageRejected> {
        if !state.conversations.contains_key(&conversation_id.0) {
            // Upstream throws `Unknown conversation`; the fixed read surface
            // cannot (D3) — callers treat unknown conversations as empty.
            return Ok(Vec::new());
        }
        let mut segments: Vec<(ConversationId, u64)> = Vec::new();
        let mut current_id = conversation_id;
        let mut upper_entry_id = max_entry_id;
        loop {
            segments.push((current_id, upper_entry_id));
            let conversation = state.conversations.get(&current_id.0).unwrap();
            let Some(parent) = &conversation.parent else {
                break;
            };
            upper_entry_id = upper_entry_id.min(parent.at.0);
            if upper_entry_id < min_entry_id {
                break;
            }
            current_id = parent.conversation_id;
        }
        let mut values = Vec::new();
        for (segment_id, upper) in segments.iter().rev() {
            let ids = state
                .entry_ids
                .get(&segment_id.0)
                .cloned()
                .unwrap_or_default();
            for index in lower_bound(&ids, min_entry_id)..ids.len() {
                let id = ids[index];
                if id.0 > *upper {
                    break;
                }
                values.push(state.entries.get(&id.0).unwrap().clone());
                if let Some(cap) = cap {
                    if values.len() > cap {
                        return Ok(values);
                    }
                }
            }
        }
        Ok(values)
    }
}

#[async_trait::async_trait]
impl Storage for MemoryStorage {
    async fn commit(
        &self,
        writes: &[StorageWrite],
        _context: &Context,
    ) -> Result<Seq, StorageRejected> {
        // `prepare` + `apply` happen under one lock acquisition so concurrent
        // commits stay atomic, matching the synchronous upstream `commit()`.
        let mut inner_guard = self.inner.lock();
        let mut prepared = {
            let inner = &*inner_guard;
            if inner.closed {
                return Err(StorageRejected::new("MemoryStorage is closed"));
            }
            let seq = Seq::from_number(inner.next_seq);
            let detached_writes = Self::resolve_document_copies(&inner.state, writes.to_vec())?;
            Self::check_global_ids(&inner.state, &detached_writes)?;
            let actions = Self::prepare_document_actions(&detached_writes)?;
            Self::check_document_actions(&inner.state, &actions)?;
            PreparedMemoryCommit {
                detached: detached_writes.clone(),
                seq,
                writes: detached_writes,
                actions,
                applied: false,
            }
        };
        // `drop` of the immutable reborrow is implied; apply re-locks via
        // `inner`, so release the guard manually to keep one critical section.
        let seq = prepared.seq;
        let writes = std::mem::take(&mut prepared.writes);
        let actions = std::mem::take(&mut prepared.actions);
        self.apply_locked(&mut inner_guard, &writes, &actions, seq);
        Ok(seq)
    }

    async fn mint_id(&self) -> u64 {
        let mut inner = self.inner.lock();
        let id = inner.next_id;
        inner.next_id += 1;
        id
    }

    async fn conversation(
        &self,
        id: ConversationId,
        _context: &Context,
    ) -> Option<ConversationRecord> {
        let inner = self.inner.lock();
        if inner.closed {
            return None;
        }
        inner.state.conversations.get(&id.0).cloned()
    }

    async fn scan_conversations(
        &self,
        query: &ConversationQuery,
        limit: u64,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Page<ConversationRecord> {
        let inner = self.inner.lock();
        if inner.closed {
            return Page::empty();
        }
        let ids: &[ConversationId] = if let Some(owner_task_id) = query.owner_task_id {
            inner
                .state
                .conversations_by_owner_task
                .get(&owner_task_id.0)
                .map_or(&[], Vec::as_slice)
        } else if let Some(owner_conversation_id) = query.owner_conversation_id {
            inner
                .state
                .conversations_by_owner_conversation
                .get(&owner_conversation_id.0)
                .map_or(&[], Vec::as_slice)
        } else {
            &inner.state.conversation_ids
        };
        let start = scan_start(query.order, cursor, ScanOrder::Ascending);
        let mut values = Vec::new();
        for index in scan_indexes(ids, start) {
            if values.len() as u64 > limit {
                break;
            }
            let value = inner.state.conversations.get(&ids[index].0).unwrap();
            if let Some(owner_conversation_id) = query.owner_conversation_id {
                if value.owner.as_ref().map(|owner| owner.conversation_id)
                    != Some(owner_conversation_id)
                {
                    continue;
                }
            }
            values.push(value.clone());
        }
        page(values, |record| record.id.0, limit, start.order)
    }

    async fn entry(&self, id: EntryId, _context: &Context) -> Option<(EntryRecord, Seq)> {
        let inner = self.inner.lock();
        if inner.closed {
            return None;
        }
        let entry = inner.state.entries.get(&id.0)?;
        Some((entry.clone(), inner.state.entry_commit_seqs[&id.0]))
    }

    async fn entry_in(
        &self,
        conversation_id: ConversationId,
        id: EntryId,
        _context: &Context,
    ) -> Option<(EntryRecord, Seq)> {
        let inner = self.inner.lock();
        if inner.closed {
            return None;
        }
        let visible =
            Self::visible_entries(&inner.state, conversation_id, id.0, id.0, Some(0)).ok()?;
        let entry = visible.into_iter().next()?;
        Some((entry, inner.state.entry_commit_seqs[&id.0]))
    }

    async fn find_latest_head_marker(
        &self,
        conversation_id: ConversationId,
        at_or_before_entry_id: Option<EntryId>,
        _context: &Context,
    ) -> Option<EntryRecord> {
        let inner = self.inner.lock();
        if inner.closed || !inner.state.conversations.contains_key(&conversation_id.0) {
            return None;
        }
        let mut current_id = conversation_id;
        let mut upper_entry_id = at_or_before_entry_id.map_or(u64::MAX, |id| id.0);
        loop {
            let ids = inner
                .state
                .head_entry_ids
                .get(&current_id.0)
                .cloned()
                .unwrap_or_default();
            let index = upper_bound(&ids, upper_entry_id);
            if index > 0 {
                return inner.state.entries.get(&ids[index - 1].0).cloned();
            }
            let conversation = inner.state.conversations.get(&current_id.0).unwrap();
            let parent = conversation.parent.as_ref()?;
            upper_entry_id = upper_entry_id.min(parent.at.0);
            current_id = parent.conversation_id;
        }
    }

    async fn scan_entries(
        &self,
        query: &EntryQuery,
        limit: u64,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Page<EntryRecord> {
        let inner = self.inner.lock();
        if inner.closed {
            return Page::empty();
        }
        let start = scan_start(query.order, cursor, ScanOrder::Descending);
        // The cursor narrows the bound on the side the scan moves away from.
        let mut min_entry_id = query.min_entry_id.map_or(0, |id| id.0);
        let mut max_entry_id = query.max_entry_id.map_or(u64::MAX, |id| id.0);
        if let Some(after) = start.after {
            match start.order {
                ScanOrder::Descending => max_entry_id = max_entry_id.min(after.saturating_sub(1)),
                ScanOrder::Ascending => min_entry_id = min_entry_id.max(after.saturating_add(1)),
            }
        }
        let mut values = match start.order {
            ScanOrder::Descending => Self::visible_entries(
                &inner.state,
                query.conversation_id,
                min_entry_id,
                max_entry_id,
                Some(limit as usize + 1),
            )
            .unwrap_or_default(),
            ScanOrder::Ascending => Self::visible_entries_ascending(
                &inner.state,
                query.conversation_id,
                min_entry_id,
                max_entry_id,
                Some(limit as usize + 1),
            )
            .unwrap_or_default(),
        };
        if values.len() as u64 > limit {
            values.truncate(limit as usize);
            let after = values.last().unwrap().id.0;
            return Page {
                items: values,
                next: Some(next_cursor(after, start.order)),
            };
        }
        Page {
            items: values,
            next: None,
        }
    }

    async fn task(&self, id: TaskId, _context: &Context) -> Option<TaskRecord> {
        let inner = self.inner.lock();
        if inner.closed {
            return None;
        }
        inner.state.tasks.get(&id.0).cloned()
    }

    async fn scan_tasks(
        &self,
        query: &TaskQuery,
        limit: u64,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Page<TaskRecord> {
        let inner = self.inner.lock();
        if inner.closed {
            return Page::empty();
        }
        let ids: &[TaskId] = match query.status {
            None => &inner.state.task_ids,
            Some(status) => inner
                .state
                .tasks_by_status
                .get(&status)
                .map_or(&[], Vec::as_slice),
        };
        let start = scan_start(query.order, cursor, ScanOrder::Ascending);
        let mut values = Vec::new();
        for index in scan_indexes(ids, start) {
            if values.len() as u64 > limit {
                break;
            }
            let value = inner.state.tasks.get(&ids[index].0).unwrap();
            if query
                .conversation_id
                .is_some_and(|id| value.conversation_id != id)
            {
                continue;
            }
            if query.kind.as_deref().is_some_and(|kind| value.kind != kind) {
                continue;
            }
            if query
                .abort_requested
                .is_some_and(|abort| value.abort_requested != abort)
            {
                continue;
            }
            if query
                .background
                .is_some_and(|background| value.background != background)
            {
                continue;
            }
            values.push(value.clone());
        }
        page(values, |record| record.id.0, limit, start.order)
    }

    async fn submission(&self, id: SubmissionId, _context: &Context) -> Option<SubmissionRecord> {
        let inner = self.inner.lock();
        if inner.closed {
            return None;
        }
        inner.state.submissions.get(&id.0).cloned()
    }

    async fn scan_submissions(
        &self,
        query: &SubmissionQuery,
        limit: u64,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Page<SubmissionRecord> {
        let inner = self.inner.lock();
        if inner.closed {
            return Page::empty();
        }
        let ids: &[SubmissionId] = match query.status {
            None => &inner.state.submission_ids,
            Some(status) => inner
                .state
                .submissions_by_status
                .get(submission_status_key(status))
                .map_or(&[], Vec::as_slice),
        };
        let start = scan_start(query.order, cursor, ScanOrder::Ascending);
        let mut values = Vec::new();
        for index in scan_indexes(ids, start) {
            if values.len() as u64 > limit {
                break;
            }
            let value = inner.state.submissions.get(&ids[index].0).unwrap();
            if query
                .conversation_id
                .is_some_and(|id| value.conversation_id != id)
            {
                continue;
            }
            values.push(value.clone());
        }
        page(values, |record| record.id.0, limit, start.order)
    }

    async fn submission_by_request(
        &self,
        conversation_id: ConversationId,
        request_id: &str,
        _context: &Context,
    ) -> Option<SubmissionRecord> {
        let inner = self.inner.lock();
        if inner.closed {
            return None;
        }
        let id = inner
            .state
            .submissions_by_request
            .get(&conversation_id.0)?
            .get(request_id)?;
        inner.state.submissions.get(&id.0).cloned()
    }

    async fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        _context: &Context,
    ) -> Option<DocumentRecord> {
        let inner = self.inner.lock();
        if inner.closed {
            return None;
        }
        let key = address_key(&address.kind, &address.scope, address.key.as_deref());
        let index = inner.state.document_addresses.get(&key)?;
        match at {
            DocumentPoint::Current => Some(
                inner
                    .state
                    .documents
                    .get(&index.current_id?.0)?
                    .record
                    .clone(),
            ),
            DocumentPoint::At(_) => {
                for id in &index.ids {
                    let record = &inner.state.documents.get(&id.0)?.record;
                    if is_alive_at(record, at) {
                        return Some(record.clone());
                    }
                }
                None
            }
        }
    }

    async fn document(
        &self,
        id: DocumentId,
        at: DocumentPoint,
        _context: &Context,
    ) -> Option<StoredDocument> {
        let inner = self.inner.lock();
        if inner.closed {
            return None;
        }
        Self::materialize_document(&inner.state, id, at)
    }

    async fn scan_documents(
        &self,
        query: &DocumentQuery,
        limit: u64,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Page<DocumentRecord> {
        let inner = self.inner.lock();
        if inner.closed {
            return Page::empty();
        }
        let ids = inner
            .state
            .documents_by_scope
            .get(&scope_key(&query.scope).to_string())
            .cloned()
            .unwrap_or_default();
        let after = cursor_id(cursor);
        let start = after.map_or(0, |after| upper_bound(&ids, after));
        let mut values = Vec::new();
        for index in start..ids.len() {
            if values.len() as u64 > limit {
                break;
            }
            let record = &inner.state.documents.get(&ids[index].0).unwrap().record;
            if query
                .kind
                .as_deref()
                .is_some_and(|kind| record.kind != kind)
            {
                continue;
            }
            if is_alive_at(record, query.at) {
                values.push(record.clone());
            }
        }
        page(values, |record| record.id.0, limit, ScanOrder::Ascending)
    }

    async fn close(&self, _context: &Context) {
        self.inner.lock().closed = true;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pi_chord::context::BACKGROUND_CONTEXT;
    use serde_json::{json, Value};

    use super::*;
    use crate::storage::conformance::{entry_json, record, run_conformance, ROOT_CONVERSATION_ID};

    #[test]
    fn storage_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MemoryStorage>();
    }

    #[tokio::test]
    async fn storage_conformance() {
        run_conformance(|| Box::pin(async { Arc::new(MemoryStorage::new()) as Arc<dyn Storage> }))
            .await;
    }

    #[tokio::test]
    async fn rejects_preparing_a_commit_after_close() {
        let storage = MemoryStorage::new();
        storage.close(&BACKGROUND_CONTEXT).await;
        let error = storage
            .prepare_commit_at(&[], Seq(1))
            .err()
            .expect("closed storage must reject preparation");
        assert_eq!(error.to_string(), "MemoryStorage is closed");
    }

    /// Port of `memory-storage.test.ts` "does not expose retained state through
    /// a prepared commit" (freeze degrades to clone detachment, D1).
    #[tokio::test]
    async fn does_not_expose_retained_state_through_a_prepared_commit() {
        let storage = MemoryStorage::new();
        storage
            .commit(
                &[StorageWrite::Conversation {
                    value: record(conversation_json_value()),
                }],
                &BACKGROUND_CONTEXT,
            )
            .await
            .unwrap();
        let entry_id = EntryId(2);
        let mut prepared = storage
            .prepare_commit(&[StorageWrite::Entry {
                value: record(entry_json(
                    entry_id.0,
                    ROOT_CONVERSATION_ID.0,
                    "test",
                    json!({ "data": { "nested": [1] } }),
                )),
            }])
            .unwrap();
        // The prepared writes are detached: mutating them cannot leak into
        // storage (upstream asserts the frozen write throws on mutation).
        mutate_entry_data(&mut prepared.writes);
        assert_eq!(prepared.apply(&storage), Seq(2));
        assert_eq!(prepared.apply(&storage), Seq(2));
        let entry = storage
            .entry(entry_id, &BACKGROUND_CONTEXT)
            .await
            .unwrap()
            .0;
        assert_eq!(entry.data, Some(json!({ "nested": [1] })));
    }

    fn conversation_json_value() -> Value {
        json!({ "id": ROOT_CONVERSATION_ID.0 })
    }

    fn mutate_entry_data(writes: &mut [StorageWrite]) {
        if let Some(StorageWrite::Entry { value }) = writes.first_mut() {
            if let Some(Value::Object(data)) = value.data.as_mut() {
                if let Some(Value::Array(nested)) = data.get_mut("nested") {
                    nested.push(json!(2));
                }
            }
        }
    }
}
