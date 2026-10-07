//! Port of `packages/durable/src/session/session.ts` (`createSession` and the
//! `SessionImpl` kernel: the mutation line, the loaded-document tracker cache,
//! committed publication, snapshots/states/watches, close) plus the
//! session-owned harness surfaces re-exported by `crate::harness`
//! (`SettledTask`, `ContextView`, `ConversationHandle`,
//! `ConversationAbortOptions` — upstream `harness/types.ts` lines 86–128/494).
//!
//! Deviations (D1 — sync line facade): upstream serializes commit callbacks on
//! a promise chain and `await`s async callbacks; here the line is a tokio
//! async mutex held for the whole job and commit callbacks are synchronous
//! closures over `&mut Tx` (`crate::tasks` D3 convention), with
//! [`Session::commit_async`] for async jobs. Storage access goes through the
//! blocking [`StorageBridge`](bridge::StorageBridge).
//! - D2: `commitWith` is public as [`Session::commit_with`]; `readOnLine` and
//!   `conversationDocumentOnLine` are public internals with the same names.
//! - D3: protected overrides (`conversationCreated`/`beforeClose`) become the
//!   [`SessionHooks`] registry installed by the harness.
//! - D4: poison is reachable only through tracker-cache adoption failures;
//!   upstream's uncertain-storage-failure poison is unrepresentable because the
//!   Rust `Storage::commit` has the single `StorageRejected` error type.
//! - D5: the TS doc overloads collapse to `(token, owner: Option<u64>,
//!   key: Option<&str>)` — the erased addressing used by the harness
//!   `DocumentReader` seam.
//! - D6: close listeners fire after admission is sealed and before the
//!   asynchronous `beforeClose` hook joins owned work.
//! - D7: `documentState`/`watchDoc` per-observer commit subscriptions register
//!   through the same listener registry as `subscribeCommits`; a subscription
//!   failure after close panics exactly where upstream throws synchronously
//!   (`subscribeCommits`/`subscribeClose`).

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use pi_chord::context::{without_abort_signal, Context};
use pi_chord::delta::Tracker;
use pi_chord::{AttachedReplicatedState, ReplicatedStateSource, ReplicatedStateSourceOptions};
use serde_json::Value;
use std::pin::Pin;

use crate::documents::{
	check_record_scope, check_record_version, materialize_document, resolve_address, DocDefinition, DocToken,
	DocumentAccessError, DocumentRecordRef,
};
use crate::errors::StorageRejected;
use crate::harness::types::{DocumentReader, InputSubmissionDraft, Submission};
use crate::session::bridge::StorageBridge;
use crate::session::observation::{
	cancellation_message, observed_operations, CommittedStateSource, DocumentWatch, ObservedDocumentValue,
};
use crate::session::transaction::{TransactionScope, Tx, TxError};
use crate::types::{
	CommitChange, CommitPublication, ConversationId, ConversationRecord, DocumentAddress, DocumentCommitChange,
	DocumentId, DocumentPoint, DocumentRecord, DocumentScope, EntryId, EntryRecord, JsonObject, Seq, Storage,
	StorageWrite, TableCommitChange, TaskRecord,
};

/// Sendable boxed commit job (upstream `change: (tx: Tx) => T | Promise<T>`).
type SendCommitJob<T> = Box<
	dyn for<'tx> FnOnce(&'tx Tx<'tx>) -> Pin<Box<dyn Future<Output = Result<T, String>> + Send + 'tx>>
		+ Send
		+ 'static,
>;

// ─── Public session-owned harness surfaces ───────────────────────────────────

/// Task record whose state is terminal (upstream `SettledTask<R>` — the erased
/// narrowing is contractual; the record shape is [`TaskRecord`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SettledTask {
	pub record: TaskRecord,
}

impl SettledTask {
	/// Wrap a terminal record (the host narrows before wrapping).
	pub fn new(record: TaskRecord) -> Self {
		Self { record }
	}

	/// The terminal outcome (upstream `settled.state.outcome`).
	pub fn outcome(&self) -> Option<&crate::types::TaskOutcome> {
		self.record.state.outcome.as_ref()
	}
}

impl std::ops::Deref for SettledTask {
	type Target = TaskRecord;

	fn deref(&self) -> &TaskRecord {
		&self.record
	}
}

/// Raw active transcript and derived model context (upstream `ContextView`);
/// the harness computes the derived fields, the Session owns the shape.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ContextView {
	/// Newest applicable head marker, if any.
	pub head: Option<EntryRecord>,
	/// Raw active entries: the head marker followed by non-head entries from its
	/// head through the tail.
	pub entries: Vec<EntryRecord>,
	/// Per entry of `entries`, its model messages after edits and excluded stop
	/// reasons, before tool result ordering.
	pub contributions: Vec<Vec<pi_ai::Message>>,
	/// Model context for the next provider request.
	pub messages: Vec<pi_ai::Message>,
}

/// Options of conversation aborts (upstream `ConversationAbortOptions`).
///
/// Cross background boundaries: mark every live task reached ignoring the
/// background flag when the abort is admitted, withdraw the queued inputs of
/// every conversation reached, and wait until those tasks are terminal and the
/// conversation is ordinarily idle. Background work created afterwards is
/// neither marked nor awaited.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationAbortOptions {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub background: Option<bool>,
}

/// Invocation-bound conversation operations for tasks and tools (upstream
/// `ConversationHandle`). Rejects after the invocation ends; passive entries are
/// written with ordinary transaction writes instead. The harness implements the
/// instances.
#[async_trait::async_trait]
pub trait ConversationHandle: Send + Sync {
	/// The bound conversation.
	fn id(&self) -> ConversationId;

	/// Durably admit user input that may start a run.
	async fn submit(&self, submission: InputSubmissionDraft, context: &Context) -> Result<Arc<dyn Submission>, String>;

	/// `Conversation.abort()`: withdraw queued inputs, abort the ordinary
	/// ownership scope, and wait until it is idle.
	async fn abort(&self, context: &Context, options: Option<ConversationAbortOptions>) -> Result<(), String>;

	/// Resolve when the conversation's ordinary ownership scope has no live
	/// non-background task.
	async fn wait_for_idle(&self, context: &Context) -> Result<(), String>;
}

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Failure of one Session operation; `Display` strings match upstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
	/// Operation after close.
	Closed,
	/// A commit failed after Storage admission; the session must be reopened.
	Poisoned,
	/// Typed read-after-write rejection (upstream `ReadAfterWrite`).
	ReadAfterWrite(crate::errors::ReadAfterWrite),
	/// Storage rejected the batch; no effect committed.
	Storage(StorageRejected),
	/// Any other failure; message verbatim.
	Message(String),
}

impl std::fmt::Display for SessionError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			SessionError::Closed => f.write_str("Session is closed"),
			SessionError::Poisoned => f.write_str(
				"Session is poisoned by a failed commit after storage admission; reopen it",
			),
			SessionError::ReadAfterWrite(error) => std::fmt::Display::fmt(error, f),
			SessionError::Storage(error) => std::fmt::Display::fmt(error, f),
			SessionError::Message(message) => f.write_str(message),
		}
	}
}

impl std::error::Error for SessionError {}

impl From<DocumentAccessError> for SessionError {
	fn from(error: DocumentAccessError) -> Self {
		SessionError::Message(error.to_string())
	}
}

impl From<TxError> for SessionError {
	fn from(error: TxError) -> Self {
		match error {
			TxError::ReadAfterWrite(read_after_write) => SessionError::ReadAfterWrite(read_after_write),
			TxError::Message(message) => SessionError::Message(message),
		}
	}
}

// ─── Kernel state ────────────────────────────────────────────────────────────

/// One committed document incarnation owned by the Session tracker cache
/// (upstream `LoadedDocument`).
pub(crate) struct LoadedDocument {
	pub address_id: String,
	pub record: DocumentRecord,
	/// Persisted definition version; older while the tracked value is migrated
	/// only in memory.
	pub stored_version: u64,
	/// Definition version whose shape the tracked value has.
	pub value_version: u64,
	/// Stored deltas after the newest base; advanced by adoption.
	pub deltas_since_base: u64,
	pub tracker: Tracker,
}

/// Per-line kernel state: the loaded document tracker cache.
pub(crate) struct LineState {
	pub(crate) documents: HashMap<String, Arc<Mutex<LoadedDocument>>>,
}

impl LineState {
	pub(crate) fn new() -> Self {
		Self { documents: HashMap::new() }
	}
}

/// Internal hooks replacing upstream's protected overrides (D3).
#[derive(Default)]
pub(crate) struct SessionHooks {
	/// Runs after close seals admission and before the line closes Storage;
	/// must not reject.
	pub before_close: Option<Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>>,
	/// Runs inside every transaction that creates or forks a conversation,
	/// after the conversation record is staged.
	pub conversation_created: Option<Box<dyn Fn(&Tx<'_>, &ConversationRecord) -> Result<(), TxError> + Send + Sync>>,
}

/// Commit publication listener (upstream
/// `(publication: CommitPublication, context: Context) => void`).
pub type CommitListener = Arc<dyn Fn(&CommitPublication, &Context) + Send + Sync>;

/// Close listener (upstream `() => void`).
pub type CloseListener = Arc<dyn Fn() + Send + Sync>;

pub(crate) struct ListenerRegistry {
	commits: Vec<(u64, CommitListener)>,
	closes: Vec<(u64, CloseListener)>,
	next_id: u64,
}

impl ListenerRegistry {
	/// Drain every close listener (close drains them once, in order).
	fn take_closes(&mut self) -> Vec<CloseListener> {
		std::mem::take(&mut self.closes).into_iter().map(|(_, listener)| listener).collect()
	}

	/// Drop every commit subscription (upstream clears `#commitListeners`).
	fn clear_commits(&mut self) {
		self.commits.clear();
	}
}

/// Handle that removes one listener when called (upstream unsubscribe fns).
#[derive(Clone)]
pub struct Unsubscribe {
	registry: Arc<Mutex<ListenerRegistry>>,
	id: u64,
	commit: bool,
}

impl Unsubscribe {
	/// Remove the listener (idempotent; a second call does nothing).
	pub fn call(&self) {
		let mut registry = self.registry.lock();
		if self.commit {
			registry.commits.retain(|(id, _)| *id != self.id);
		} else {
			registry.closes.retain(|(id, _)| *id != self.id);
		}
	}
}

/// Owner of one mutation line, its records, and its tracked documents
/// (upstream `SessionImpl` internals).
pub(crate) struct SessionInner {
	pub(crate) storage: Arc<dyn Storage>,
	pub(crate) bridge: StorageBridge,
	/// Admission + serialization of line jobs (upstream `#tail` promise chain).
	line_gate: Arc<tokio::sync::Mutex<()>>,
	/// The line's kernel state; the guard is exclusive for the whole job.
	pub(crate) line: parking_lot::Mutex<LineState>,
	pub(crate) listeners: Arc<Mutex<ListenerRegistry>>,
	pub(crate) poison: Mutex<Option<String>>,
	pub(crate) closing: AtomicBool,
	close_lock: tokio::sync::Mutex<()>,
	pub(crate) hooks: Mutex<SessionHooks>,
}

impl SessionInner {
	/// Cold-load (and migrate) one document into the tracker cache
	/// (upstream `#loadDocument`).
	pub(crate) fn load_document(
		&self,
		line: &mut LineState,
		definition: &DocDefinition,
		address_id: &str,
		address: &DocumentAddress,
		context: &Context,
	) -> Result<Option<Arc<Mutex<LoadedDocument>>>, TxError> {
		// A tracker serves only tokens of the version its value was materialized
		// for; others reload from Storage.
		if let Some(cached) = line.documents.get(address_id) {
			if cached.lock().value_version == definition.version {
				return Ok(Some(cached.clone()));
			}
		}
		line.documents.remove(address_id);
		let Some(record) = self.bridge.find_document(address, DocumentPoint::Current, context) else {
			return Ok(None);
		};
		let stored = self
			.bridge
			.document(record.id, DocumentPoint::Current, context)
			.ok_or_else(|| TxError::message(format!("Current document {} ({}) cannot be read", record.id, record.kind)))?;
		let value = materialize_document(definition, &stored)?;
		let loaded = Arc::new(Mutex::new(LoadedDocument {
			address_id: address_id.to_string(),
			record: stored.record,
			stored_version: stored.version,
			value_version: definition.version,
			deltas_since_base: stored.deltas_since_base,
			tracker: Tracker::new(Value::Object(value)),
		}));
		line.documents.insert(address_id.to_string(), loaded.clone());
		Ok(Some(loaded))
	}

	/// Stage writes that belong to every newly created or forked conversation
	/// (upstream `conversationCreated`; a plain Session stages nothing).
	pub(crate) fn conversation_created(&self, tx: &Tx<'_>, record: &ConversationRecord) -> Result<(), TxError> {
		let hooks = self.hooks.lock();
		if let Some(hook) = hooks.conversation_created.as_ref() {
			hook(tx, record)?;
		}
		Ok(())
	}

	async fn before_close(&self) {
		let hook = self.hooks.lock().before_close.clone();
		if let Some(hook) = hook {
			hook().await;
		}
	}
}

fn document_value(tracker: &Tracker) -> JsonObject {
	tracker.value().as_object().expect("document values are objects").clone()
}

fn change_record_id(change: &DocumentCommitChange) -> Option<DocumentId> {
	match change {
		DocumentCommitChange::Document { record, .. } => Some(record.id),
		DocumentCommitChange::DocumentCopy { record, .. } => Some(record.id),
	}
}

/// Advance one attached state source with a committed change's frame. A
/// document state's frames carry no caller cancellation (upstream
/// `withoutAbortSignal(context)`); a migration-only base changes nothing for an
/// observer of the new version.
fn advance_state_source(
	source: &Arc<CommittedStateSource<ObservedDocumentValue>>,
	observed_version: &Mutex<u64>,
	change: &DocumentCommitChange,
	context: &Context,
) {
	let DocumentCommitChange::Document { value, .. } = change else { return };
	let ops = observed_operations(&mut observed_version.lock(), change);
	if ops.is_empty() {
		return;
	}
	let value = value.clone().map(Arc::new);
	source.advance(value, ops, without_abort_signal(context.clone()));
}

// ─── Session ─────────────────────────────────────────────────────────────────

/// Session kernel: one mutation line, the loaded document tracker cache, and
/// committed publication (upstream `Session` / `SessionImpl`).
///
/// Only committed state is observable. Every commit callback, preparation,
/// Storage settlement, adoption, and publication enqueue runs while the line is
/// held; listeners run during publication.
#[derive(Clone)]
pub struct Session {
	inner: Arc<SessionInner>,
}

/// Open a Session kernel over one storage backend (upstream `createSession`).
pub fn create_session(storage: Arc<dyn Storage>) -> Session {
	Session {
		inner: Arc::new(SessionInner {
			bridge: StorageBridge::start(storage.clone()),
			storage,
			line_gate: Arc::new(tokio::sync::Mutex::new(())),
			line: parking_lot::Mutex::new(LineState::new()),
			listeners: Arc::new(Mutex::new(ListenerRegistry { commits: Vec::new(), closes: Vec::new(), next_id: 0 })),
			poison: Mutex::new(None),
			closing: AtomicBool::new(false),
			close_lock: tokio::sync::Mutex::new(()),
			hooks: Mutex::new(SessionHooks::default()),
		}),
	}
}

/// Snapshot of one conversation document's current incarnation
/// (upstream `conversationDocumentOnLine`'s result).
#[derive(Debug, Clone, PartialEq)]
pub struct ConversationDocumentSnapshot {
	pub record: DocumentRecord,
	pub version: u64,
	pub value: JsonObject,
}

impl Session {
	/// Install the internal override hooks (upstream protected overrides; D3).
	pub(crate) fn install_hooks(&self, hooks: SessionHooks) {
		*self.inner.hooks.lock() = hooks;
	}

	/// Run one atomic transaction on the Session mutation line (upstream
	/// `commit`).
	pub async fn commit<T>(
		&self,
		change: impl FnOnce(&mut Tx<'_>) -> Result<T, String> + Send + 'static,
		context: &Context,
	) -> Result<T, SessionError>
	where
		T: Send + 'static,
	{
		self.run_commit_sync(change, context, None).await
	}

	/// Commit with a borrowed `Send` `FnMut` closure — the
	/// [`crate::harness::api`] `ToolExecutionApi::commit` shape (`&mut F where
	/// F: FnMut` implements `FnOnce`). The job runs inline on the caller's
	/// task; because the callback reference is `Send`, the future is `Send`.
	pub async fn commit_dyn(
		&self,
		change: &mut super::CommitValueCallback<'_>,
		context: &Context,
	) -> Result<Value, SessionError> {
		self.run_commit_sync_value(change, context, None).await
	}
	/// Commit a borrowed callback after a synchronous transaction guard.
	pub(crate) async fn commit_dyn_with_guard(
		&self,
		guard: &mut super::CommitGuardCallback<'_>,
		change: &mut super::CommitValueCallback<'_>,
		context: &Context,
	) -> Result<Value, SessionError> {
		self.run_commit_sync_value_guarded(guard, change, context, None).await
	}

	/// Internal commit exposing the transaction scope: the scope sets the
	/// default `tx.create_task()` conversation and the task attributed to
	/// appended entries (upstream `commitWith`).
	pub async fn commit_with<T>(
		&self,
		change: impl FnOnce(&mut Tx<'_>) -> Result<T, String> + Send + 'static,
		context: &Context,
		scope: TransactionScope,
	) -> Result<T, SessionError>
	where
		T: Send + 'static,
	{
		self.run_commit_sync(change, context, Some(scope)).await
	}

	/// Commit with an async callback (upstream async commit callbacks; the
	/// mutation line stays held across the await). `Box::pin` the callback's
	/// future, as `crate::tasks::commit_change` does; the transaction is shared
	/// because every `Tx` method takes `&self`.
	pub async fn commit_async<T, F>(&self, change: F, context: &Context) -> Result<T, SessionError>
	where
		T: Send + 'static,
		F: for<'tx> FnOnce(
				&'tx Tx<'tx>,
			) -> Pin<Box<dyn Future<Output = Result<T, String>> + Send + 'tx>>
			+ Send
			+ 'static,
	{
		let _admission = self.admit(context).await?;
		let tx = Tx::new(&self.inner, &self.inner.line, context.clone(), TransactionScope::default(), false);
		// The job holds the transaction by shared borrow; settlement shares it.
		match change(&tx).await {
			Ok(result) => self.settle_and_store(&tx, result, context).await,
			Err(message) => {
				tx.settle_failure();
				Err(SessionError::Message(message))
			}
		}
	}

	/// The line-held commit body for synchronous callbacks (the callback's
	/// `&mut Tx` borrow is transient; settlement reborrows the transaction).
	async fn run_commit_sync<T>(
		&self,
		change: impl FnOnce(&mut Tx<'_>) -> Result<T, String> + Send + 'static,
		context: &Context,
		scope: Option<TransactionScope>,
	) -> Result<T, SessionError>
	where
		T: Send + 'static,
	{
		let _admission = self.admit(context).await?;
		let mut tx = Tx::new(&self.inner, &self.inner.line, context.clone(), scope.unwrap_or_default(), false);
		match change(&mut tx) {
			Ok(result) => self.settle_and_store(&tx, result, context).await,
			Err(message) => {
				tx.settle_failure();
				Err(SessionError::Message(message))
			}
		}
	}

	/// `commit_dyn` body: the same transient shape over a borrowed `FnMut`.
	async fn run_commit_sync_value(
		&self,
		change: &mut super::CommitValueCallback<'_>,
		context: &Context,
		scope: Option<TransactionScope>,
	) -> Result<Value, SessionError> {
		let mut guard = |_tx: &mut Tx<'_>| Ok::<(), String>(());
		self.run_commit_sync_value_guarded(&mut guard, change, context, scope).await
	}

	async fn run_commit_sync_value_guarded(
		&self,
		guard: &mut super::CommitGuardCallback<'_>,
		change: &mut super::CommitValueCallback<'_>,
		context: &Context,
		scope: Option<TransactionScope>,
	) -> Result<Value, SessionError> {
		let _admission = self.admit(context).await?;
		let mut tx = Tx::new(&self.inner, &self.inner.line, context.clone(), scope.unwrap_or_default(), false);
		match guard(&mut tx).and_then(|()| change(&mut tx)) {
			Ok(result) => self.settle_and_store(&tx, result, context).await,
			Err(message) => {
				tx.settle_failure();
				Err(SessionError::Message(message))
			}
		}
	}

	/// Seal admission, check health and cancellation, and hold the line
	/// (upstream `#assertUsable` + `#runCommit` admission).
	///
	/// Returns only the `Send` gate guard: mutual exclusion for the whole job
	/// comes from the async gate, while the parking_lot line lock is taken for
	/// each brief cache access (a `MutexGuard` would make the commit futures
	/// `!Send`).
	async fn admit(
		&self,
		context: &Context,
	) -> Result<tokio::sync::OwnedMutexGuard<()>, SessionError> {
		self.assert_usable()?;
		let gate = self.inner.line_gate.clone().lock_owned().await;
		if let Some(signal) = context.abort_signal() {
			if signal.aborted() {
				return Err(SessionError::Message(cancellation_message(&signal)));
			}
		}
		Ok(gate)
	}

	/// Settle a successful callback, store the batch, adopt, and publish
	/// (upstream `#runCommit` settlement half).
	async fn settle_and_store<T>(
		&self,
		tx: &Tx<'_>,
		result: T,
		context: &Context,
	) -> Result<T, SessionError> {
		let writes = tx.settle_success()?;
		if writes.is_empty() {
			tx.discard();
			return Ok(result);
		}
		// Once admitted, caller cancellation does not interrupt Storage settlement.
		let seq = match self.inner.bridge.commit(&writes, context) {
			Ok(seq) => seq,
			// StorageRejected alone guarantees that no batch effect committed.
			Err(error) => {
				tx.discard();
				return Err(SessionError::Storage(error));
			}
		};
		let documents = match tx.adopt(seq) {
			Ok(documents) => documents,
			// Storage already committed; a failed adoption leaves memory behind
			// durable state.
			Err(error) => {
				*self.inner.poison.lock() = Some(error.clone());
				return Err(SessionError::Message(error));
			}
		};
		self.publish(seq, &writes, documents, context);
		Ok(result)
	}

	/// Internal: run a job on the mutation line so multi-read derivations
	/// observe one committed state (upstream `readOnLine`; the harness
	/// hydrations and inspections use this exclusion).
	pub async fn read_on_line<T, F>(&self, job: F) -> Result<T, String>
	where
		F: Future<Output = Result<T, String>>,
	{
		self.assert_healthy().map_err(|error| error.to_string())?;
		let _gate = self.inner.line_gate.lock().await;
		if self.inner.closing.load(Ordering::SeqCst) {
			return Err("Session is closed".to_string());
		}
		job.await
	}

	/// Internal: run a read-only transaction on the mutation line (the
	/// transactional shape of upstream `readOnLine`; writes are rejected —
	/// transaction D-T4).
	pub async fn read_on_line_tx<T>(
		&self,
		job: impl FnOnce(&mut Tx<'_>) -> Result<T, String> + Send,
		context: &Context,
	) -> Result<T, SessionError> {
		self.assert_healthy()?;
		let _gate = self.inner.line_gate.lock().await;
		let _line = self.inner.line.lock();
		if self.inner.closing.load(Ordering::SeqCst) {
			return Err(SessionError::Closed);
		}
		let mut tx = Tx::new(&self.inner, &self.inner.line, context.clone(), TransactionScope::default(), true);
		job(&mut tx).map_err(SessionError::Message)
	}

	/// Internal: read a conversation document while the caller holds the Session
	/// mutation line (upstream `conversationDocumentOnLine`); absent documents are `None`.
	pub(crate) fn conversation_document_on_line(
		&self,
		token: &DocToken,
		conversation_id: ConversationId,
		context: &Context,
	) -> Result<Option<ConversationDocumentSnapshot>, SessionError> {
		self.assert_healthy()?;
		let definition = token.definition();
		let resolved = resolve_address(definition, Some(conversation_id.0), None)?;
		let mut line = self.inner.line.lock();
		if self.inner.closing.load(Ordering::SeqCst) {
			return Err(SessionError::Closed);
		}
		let Some(loaded) = self
			.inner
			.load_document(&mut line, definition, &resolved.id, &resolved.address, context)
			.map_err(SessionError::from)?
		else {
			return Ok(None);
		};
		let guard = loaded.lock();
		check_record_scope(definition, DocumentRecordRef::Record(&guard.record))?;
		check_record_version(definition, DocumentRecordRef::Record(&guard.record), guard.stored_version)?;
		Ok(Some(ConversationDocumentSnapshot {
			record: guard.record.clone(),
			version: guard.value_version,
			value: document_value(&guard.tracker),
		}))
	}

	/// Committed state of one document at its address (upstream `snapshot`);
	/// never creates and returns `None` when absent.
	pub async fn snapshot(
		&self,
		token: &DocToken,
		owner: Option<u64>,
		key: Option<&str>,
		context: &Context,
	) -> Result<Option<Arc<JsonObject>>, SessionError> {
		self.assert_usable()?;
		let definition = token.definition();
		let resolved = resolve_address(definition, owner, key)?;
		{
			let line = self.inner.line.lock();
			if let Some(cached) = line.documents.get(&resolved.id) {
				let cached = cached.lock();
				if cached.value_version == definition.version {
					check_record_scope(definition, DocumentRecordRef::Record(&cached.record))?;
					check_record_version(
						definition,
						DocumentRecordRef::Record(&cached.record),
						cached.stored_version,
					)?;
					return Ok(Some(Arc::new(document_value(&cached.tracker))));
				}
			}
		}
		let _gate = self.inner.line_gate.lock().await;
		let mut line = self.inner.line.lock();
		self.assert_healthy()?;
		let Some(loaded) = self
			.inner
			.load_document(&mut line, definition, &resolved.id, &resolved.address, context)
			.map_err(SessionError::from)?
		else {
			return Ok(None);
		};
		let guard = loaded.lock();
		check_record_scope(definition, DocumentRecordRef::Record(&guard.record))?;
		check_record_version(definition, DocumentRecordRef::Record(&guard.record), guard.stored_version)?;
		Ok(Some(Arc::new(document_value(&guard.tracker))))
	}

	/// Rewound state of a conversation document as of an entry (upstream
	/// `snapshotAsOf`; requires a conversation document).
	pub async fn snapshot_as_of(
		&self,
		token: &DocToken,
		conversation_id: ConversationId,
		key: Option<&str>,
		at: EntryId,
		context: &Context,
	) -> Result<Option<JsonObject>, SessionError> {
		self.assert_usable()?;
		let definition = token.definition();
		let resolved = resolve_address(definition, Some(conversation_id.0), key)?;
		if !matches!(resolved.address.scope, DocumentScope::Conversation { .. }) {
			return Err(SessionError::Message(
				"Session.snapshotAsOf() requires a conversation document".to_string(),
			));
		}
		let _line = self.inner.line.lock();
		self.assert_healthy()?;
		let Some((stored_entry, commit_seq)) = self.inner.bridge.entry_in(conversation_id, at, context) else {
			return Err(SessionError::Message(format!(
				"Entry {at} is not visible from conversation {conversation_id}"
			)));
		};
		let address = DocumentAddress {
			kind: resolved.address.kind.clone(),
			scope: DocumentScope::Conversation { conversation_id: stored_entry.conversation_id },
			key: resolved.address.key.clone(),
		};
		let Some(record) = self.inner.bridge.find_document(&address, DocumentPoint::At(commit_seq), context) else {
			return Ok(None);
		};
		let Some(stored) = self.inner.bridge.document(record.id, DocumentPoint::At(commit_seq), context) else {
			return Err(SessionError::Message(format!(
				"Historical document {} ({}) cannot be read",
				record.id, record.kind
			)));
		};
		Ok(Some(materialize_document(definition, &stored)?))
	}

	/// Disposable read-only Chord state of one document incarnation (upstream
	/// `documentState`); `None` when absent.
	pub async fn document_state(
		&self,
		token: &DocToken,
		owner: Option<u64>,
		key: Option<&str>,
		context: &Context,
	) -> Result<Option<Arc<AttachedReplicatedState>>, SessionError> {
		self.assert_usable()?;
		let definition = token.definition();
		let resolved = resolve_address(definition, owner, key)?;
		let mut line = self.inner.line.lock();
		self.assert_healthy()?;
		let Some(loaded) = self
			.inner
			.load_document(&mut line, definition, &resolved.id, &resolved.address, context)
			.map_err(SessionError::from)?
		else {
			return Ok(None);
		};
		{
			let guard = loaded.lock();
			check_record_scope(definition, DocumentRecordRef::Record(&guard.record))?;
			check_record_version(definition, DocumentRecordRef::Record(&guard.record), guard.stored_version)?;
		}
		let record_id = loaded.lock().record.id;
		let observed_version = Arc::new(Mutex::new(loaded.lock().value_version));
		let initial_value: ObservedDocumentValue = Some(Arc::new(document_value(&loaded.lock().tracker)));
		// Attach an observer to one committed incarnation: check the definition,
		// then forward this incarnation's committed changes and close (upstream
		// `#attachDocument`); `detach` removes both subscriptions.
		let ids = self.allocate_listener_ids();
		let (source, _detach_alias) = CommittedStateSource::new(initial_value, ids.detach());
		let source_for_listener = source.clone();
		let observed_for_listener = observed_version.clone();
		ids.install(
			Arc::new(move |publication: &CommitPublication, context: &Context| {
				for change in &publication.changes {
					let CommitChange::Document(change) = change else { continue };
					if change_record_id(change) != Some(record_id) {
						continue;
					}
					advance_state_source(&source_for_listener, &observed_for_listener, change, context);
				}
			}),
			{
				let source_for_close = source.clone();
				Arc::new(move || source_for_close.close_session())
			},
		);
		let state = pi_chord::attach_replicated_state(
			Arc::clone(&source) as Arc<dyn ReplicatedStateSource>,
			ReplicatedStateSourceOptions { on_error: None },
		)
		.map_err(|error| {
			// Detach both subscriptions when Chord rejects the attachment.
			ids.detach()();
			SessionError::Message(error.to_string())
		})?;
		Ok(Some(state))
	}

	/// Non-creating document watch acquisition (upstream `watchDoc`); `None`
	/// when absent.
	pub async fn watch_doc(
		&self,
		token: &DocToken,
		owner: Option<u64>,
		key: Option<&str>,
		context: &Context,
	) -> Result<Option<DocumentWatch>, SessionError> {
		self.assert_usable()?;
		let definition = token.definition();
		let resolved = resolve_address(definition, owner, key)?;
		let signal = context.abort_signal();
		if let Some(signal) = &signal {
			if signal.aborted() {
				return Err(SessionError::Message(cancellation_message(signal)));
			}
		}
		let mut line = self.inner.line.lock();
		self.assert_healthy()?;
		let Some(loaded) = self
			.inner
			.load_document(&mut line, definition, &resolved.id, &resolved.address, context)
			.map_err(SessionError::from)?
		else {
			return Ok(None);
		};
		{
			let guard = loaded.lock();
			check_record_scope(definition, DocumentRecordRef::Record(&guard.record))?;
			check_record_version(definition, DocumentRecordRef::Record(&guard.record), guard.stored_version)?;
		}
		let record_id = loaded.lock().record.id;
		let observed_version = Arc::new(Mutex::new(loaded.lock().value_version));
		let initial_value: ObservedDocumentValue = Some(Arc::new(document_value(&loaded.lock().tracker)));
		let ids = self.allocate_listener_ids();
		let watch = DocumentWatch::new(initial_value, ids.detach());
		// Frames keep the producer context; the pump strips caller cancellation.
		let watch_for_listener = watch.clone();
		ids.install(
			Arc::new(move |publication: &CommitPublication, context: &Context| {
				for change in &publication.changes {
					let CommitChange::Document(change) = change else { continue };
					if change_record_id(change) != Some(record_id) {
						continue;
					}
					let ops = observed_operations(&mut observed_version.lock(), change);
					if ops.is_empty() {
						continue;
					}
					let DocumentCommitChange::Document { value, .. } = change else { continue };
					watch_for_listener.advance(value.clone().map(Arc::new), ops, context.clone());
				}
			}),
			{
				let watch_for_close = watch.clone();
				Arc::new(move || watch_for_close.close_session())
			},
		);
		if let Some(signal) = &signal {
			let _ = watch.observe_cancellation(signal);
			if signal.aborted() {
				return Err(SessionError::Message(cancellation_message(signal)));
			}
		}
		Ok(Some(watch))
	}

	/// Seal admission, settle admitted commits, then close storage (upstream
	/// `close`).
	pub async fn close(&self, context: &Context) -> Result<(), SessionError> {
		let _completion = self.inner.close_lock.lock().await;
		if self.inner.closing.swap(true, Ordering::SeqCst) {
			// Already closing; the first close settles Storage before returning.
			return Ok(());
		}
		// Fire close listeners synchronously after admission is sealed. The
		// harness listener stops scheduler admission before beforeClose joins.
		let listeners: Vec<CloseListener> = {
			let mut registry = self.inner.listeners.lock();
			registry.take_closes()
		};
		for listener in listeners {
			listener();
		}
		self.inner.before_close().await;
		let _line_gate = self.inner.line_gate.lock().await;
		let mut line = self.inner.line.lock();
		self.inner.listeners.lock().clear_commits();
		line.documents.clear();
		self.inner.bridge.close(&without_abort_signal(context.clone()));
		Ok(())
	}

	/// Observe complete commits synchronously after adoption (upstream
	/// `subscribeCommits`). The listener must not throw, block, or call Session
	/// APIs.
	pub fn subscribe_commits(&self, listener: CommitListener) -> Unsubscribe {
		self.assert_usable_sync();
		let registry = self.inner.listeners.clone();
		let id = next_listener_id(&registry);
		registry.lock().commits.push((id, listener));
		Unsubscribe { registry, id, commit: true }
	}

	/// Observe close synchronously when it begins (upstream `subscribeClose`).
	pub fn subscribe_close(&self, listener: CloseListener) -> Unsubscribe {
		self.assert_usable_sync();
		let registry = self.inner.listeners.clone();
		let id = next_listener_id(&registry);
		registry.lock().closes.push((id, listener));
		Unsubscribe { registry, id, commit: false }
	}

	/// Drop every loaded tracker on the mutation line; later access cold-loads
	/// from Storage (upstream `unloadDocuments`).
	pub async fn unload_documents(&self) {
		self.inner.line.lock().documents.clear();
	}

	/// Reserve the commit/close listener id pair of one attached observer and
	/// build its shared `detach` closure (upstream `#attachDocument`).
	fn allocate_listener_ids(&self) -> ListenerIds {
		let registry = self.inner.listeners.clone();
		let commit = next_listener_id(&registry);
		let close = next_listener_id(&registry);
		ListenerIds { registry, commit, close }
	}

	fn assert_usable(&self) -> Result<(), SessionError> {
		if self.inner.closing.load(Ordering::SeqCst) {
			return Err(SessionError::Closed);
		}
		self.assert_healthy()
	}

	fn assert_usable_sync(&self) {
		if let Err(error) = self.assert_usable() {
			panic!("{error}");
		}
	}

	fn assert_healthy(&self) -> Result<(), SessionError> {
		if self.inner.poison.lock().is_some() {
			Err(SessionError::Poisoned)
		} else {
			Ok(())
		}
	}

	fn publish(
		&self,
		seq: Seq,
		writes: &[StorageWrite],
		documents: Vec<DocumentCommitChange>,
		context: &Context,
	) {
		let registry = self.inner.listeners.lock();
		if registry.commits.is_empty() {
			return;
		}
		let mut changes: Vec<CommitChange> = Vec::new();
		for write in writes {
			match write {
				StorageWrite::Conversation { value } => {
					changes.push(CommitChange::Table(TableCommitChange::Conversation { value: value.clone() }))
				}
				StorageWrite::Entry { value } => {
					changes.push(CommitChange::Table(TableCommitChange::Entry { value: value.clone() }))
				}
				StorageWrite::Task { value } => {
					changes.push(CommitChange::Table(TableCommitChange::Task { value: value.clone() }))
				}
				StorageWrite::Submission { value } => {
					changes.push(CommitChange::Table(TableCommitChange::Submission { value: value.clone() }))
				}
				_ => {}
			}
		}
		for document in documents {
			changes.push(CommitChange::Document(document));
		}
		let publication = CommitPublication { seq, changes };
		for (_, listener) in &registry.commits {
			listener(&publication, context);
		}
	}
}

fn next_listener_id(registry: &Arc<Mutex<ListenerRegistry>>) -> u64 {
	let mut guard = registry.lock();
	let id = guard.next_id;
	guard.next_id += 1;
	id
}

fn unregister_ids(registry: &Arc<Mutex<ListenerRegistry>>, commit: u64, close: u64) {
	let mut guard = registry.lock();
	guard.commits.retain(|(id, _)| *id != commit);
	guard.closes.retain(|(id, _)| *id != close);
}

/// Reserved commit/close listener ids of one attached observer, with the shared
/// `detach` closure and the paired install (upstream `#attachDocument`
/// unsubscribe wiring).
struct ListenerIds {
	registry: Arc<Mutex<ListenerRegistry>>,
	commit: u64,
	close: u64,
}

impl ListenerIds {
	/// Remove both subscriptions; safe to call more than once.
	fn detach(&self) -> Box<dyn FnOnce() + Send> {
		let registry = self.registry.clone();
		let (commit, close) = (self.commit, self.close);
		Box::new(move || unregister_ids(&registry, commit, close))
	}

	/// Install the paired listeners under the reserved ids.
	fn install(&self, commit: CommitListener, close: CloseListener) {
		let mut guard = self.registry.lock();
		guard.commits.push((self.commit, commit));
		guard.closes.push((self.close, close));
	}
}

// ─── Harness DocumentReader seam ─────────────────────────────────────────────

// ─── Send assertions (the harness spawns these futures) ─────────────────────

mod send_assertions {
	use super::*;

	/// `commit` futures must be `Send` (scheduler `tokio::spawn`, async_trait
	/// seams).
	#[allow(dead_code)]
	pub fn commit_future_is_send<'a>(
		session: &'a Session,
		context: &'a Context,
	) -> impl Future<Output = Result<(), SessionError>> + Send + 'a {
		session.commit(|_tx| Ok(()), context)
	}

	/// `commit_async` futures must be `Send`.
	#[allow(dead_code)]
	pub fn commit_async_future_is_send<'a>(
		session: &'a Session,
		context: &'a Context,
	) -> impl Future<Output = Result<(), SessionError>> + Send + 'a {
		session.commit_async(
			|_tx: &Tx<'_>| {
				Box::pin(async { Ok(()) })
					as Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
			},
			context,
		)
	}

	/// `commit_dyn` futures must be `Send` (`ToolExecutionApi::commit`; the
	/// seam's callback now carries `+ Send`).
	#[allow(dead_code)]
	pub fn commit_dyn_future_is_send<'a>(
		session: &'a Session,
		change: &'a mut crate::session::CommitValueCallback<'a>,
		context: &'a Context,
	) -> impl Future<Output = Result<Value, SessionError>> + Send + 'a {
		session.commit_dyn(change, context)
	}
}

#[async_trait::async_trait]
impl DocumentReader for Session {
	async fn snapshot(
		&self,
		token: &DocToken,
		owner: Option<u64>,
		key: Option<&str>,
		context: &Context,
	) -> Result<Option<Arc<JsonObject>>, String> {
		Session::snapshot(self, token, owner, key, context).await.map_err(|error| error.to_string())
	}

	async fn snapshot_as_of(
		&self,
		token: &DocToken,
		conversation_id: ConversationId,
		key: Option<&str>,
		at: EntryId,
		context: &Context,
	) -> Result<Option<JsonObject>, String> {
		Session::snapshot_as_of(self, token, conversation_id, key, at, context)
			.await
			.map_err(|error| error.to_string())
	}
}
