//! Port of `packages/durable/test/jsonl-storage.test.ts` — JSONL publication,
//! recovery, corruption, and injected-I/O-failure behavior, plus the shared
//! backend conformance suite run fresh and across reopen.
//!
//! Deviations (module ledger, see `mod.rs`):
//! - D1: the "opens through the Node adapter" case exercises `JsonlStorage::open`
//!   directly (the Node adapter is folded into the default host).
//! - D2: read paths are infallible, so poisoned reads assert `None` instead of
//!   upstream rejections; open-path failures keep the upstream messages.
//! - D3: the "serializes the complete candidate before I/O" case uses a
//!   conversation rewrite as the preparation failure (upstream used a bigint
//!   that cannot be represented in `serde_json::Value`); the substance —
//!   preparation failure performs zero host I/O and leaves the storage usable —
//!   is identical.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use serde_json::{Value, json};

use super::host::{Failure, FailureMode, FailureOperation, Host, InstrumentedHost};
use super::{JsonlError, JsonlStorage, JsonlStorageOptions};
use crate::storage::conformance::{
	ROOT_CONVERSATION_ID, commit, commit_ok, ctx, record, run_conformance, session_document,
};
use crate::types::{
	ConversationId, Cursor, DocumentAddress, DocumentId, DocumentPoint, DocumentQuery, EntryId, EntryQuery, Page, Seq,
	Storage, StorageWrite, StoredDocument, SubmissionId, SubmissionQuery, TaskId, TaskRecord,
};

static DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_directory() -> PathBuf {
	let directory = std::env::temp_dir().join(format!(
		"pi-durable-jsonl-{}-{}",
		std::process::id(),
		DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed)
	));
	let _ = std::fs::remove_dir_all(&directory);
	directory
}

async fn open_storage_at(directory: &std::path::Path, host: Option<Arc<InstrumentedHost>>, fsync: bool) -> JsonlStorage {
	let options = JsonlStorageOptions { fsync };
	let result = match host {
		Some(host) => JsonlStorage::open_with_host(directory, host as Arc<dyn Host>, options).await,
		None => JsonlStorage::open(directory, options).await,
	};
	result.unwrap_or_else(|error| panic!("open failed: {error}"))
}

async fn open_storage(directory: &std::path::Path) -> JsonlStorage {
	open_storage_at(directory, None, false).await
}

async fn open_shared(directory: &std::path::Path) -> Arc<dyn Storage> {
	Arc::new(open_storage(directory).await)
}

async fn create_root(storage: &Arc<dyn Storage>) {
	commit_ok(storage, vec![json!({ "type": "conversation", "value": { "id": ROOT_CONVERSATION_ID.0 } })]).await;
}

fn pending_task_json(id: u64, phase: &str) -> Value {
	json!({
		"id": id,
		"conversationId": ROOT_CONVERSATION_ID.0,
		"kind": "test.task",
		"version": 1,
		"input": null,
		"state": { "status": "pending", "checkpoint": { "phase": phase } },
		"background": false,
		"abortRequested": false,
	})
}

fn terminal_task_json(id: u64) -> Value {
	json!({
		"id": id,
		"conversationId": ROOT_CONVERSATION_ID.0,
		"kind": "test.task",
		"version": 1,
		"input": null,
		"state": { "status": "terminal", "outcome": { "status": "completed", "result": null } },
		"background": false,
		"abortRequested": false,
	})
}

fn pending_task(id: u64, phase: &str) -> TaskRecord {
	record(pending_task_json(id, phase))
}

fn terminal_task(id: u64) -> TaskRecord {
	record(terminal_task_json(id))
}

fn document_create_write(id: DocumentId, kind: &str, value: Value) -> Value {
	json!({
		"type": "document.create",
		"record": session_document(id.0, kind),
		"content": { "kind": "base", "version": 1, "value": value },
	})
}

fn document_change_base(id: DocumentId, version: u64, value: Value) -> Value {
	json!({ "type": "document.change", "id": id.0, "content": { "kind": "base", "version": version, "value": value } })
}

fn document_change_delta(id: DocumentId, version: u64, ops: Value) -> Value {
	json!({ "type": "document.change", "id": id.0, "content": { "kind": "delta", "version": version, "ops": ops } })
}

fn document_retire_write(id: DocumentId) -> Value {
	json!({ "type": "document.retire", "id": id.0 })
}

fn task_write(id: u64, phase: &str) -> Value {
	json!({ "type": "task", "value": pending_task_json(id, phase) })
}

fn terminal_task_write(id: u64) -> Value {
	json!({ "type": "task", "value": terminal_task_json(id) })
}

fn read_lines(path: &std::path::Path) -> Vec<String> {
	let text = std::fs::read_to_string(path).unwrap();
	if text.is_empty() { Vec::new() } else { text.trim_end().split('\n').map(str::to_owned).collect() }
}

fn file_exists(path: &std::path::Path) -> bool {
	path.exists()
}

fn sidecar(directory: &std::path::Path, prefix: &str, id: u64) -> PathBuf {
	directory.join(format!("{prefix}-{id}.jsonl"))
}

fn append_bytes(path: &std::path::Path, bytes: &[u8]) {
	use std::io::Write as _;
	std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap().write_all(bytes).unwrap();
}

fn operations(host: &InstrumentedHost) -> Vec<String> {
	host.operations.lock().clone()
}

fn fail(host: &InstrumentedHost, operation: FailureOperation, call: usize, mode: FailureMode) {
	host.fail(Failure { operation, call, mode });
}

// ---------------------------------------------------------------------------
// Conformance (fresh + across reopen)
// ---------------------------------------------------------------------------

/// Storage wrapper that closes and reopens the underlying JSONL directory after
/// every commit (upstream `ReopeningStorage`).
struct ReopeningStorage {
	current: tokio::sync::Mutex<Arc<JsonlStorage>>,
	directory: PathBuf,
	closed: AtomicBool,
}

impl ReopeningStorage {
	async fn open(directory: PathBuf) -> Self {
		let current = open_storage(&directory).await;
		Self { current: tokio::sync::Mutex::new(Arc::new(current)), directory, closed: AtomicBool::new(false) }
	}
}

#[async_trait::async_trait]
impl Storage for ReopeningStorage {
	async fn commit(&self, writes: &[StorageWrite], context: &pi_chord::context::Context) -> Result<Seq, crate::errors::StorageRejected> {
		let current = self.current.lock().await.clone();
		let result = current.commit(writes, context).await;
		if !self.closed.load(Ordering::Relaxed) {
			current.close(context).await;
			let reopened = open_storage(&self.directory).await;
			*self.current.lock().await = Arc::new(reopened);
		}
		result
	}

	async fn mint_id(&self) -> u64 {
		self.current.lock().await.mint_id().await
	}

	async fn conversation(&self, id: ConversationId, context: &pi_chord::context::Context) -> Option<crate::types::ConversationRecord> {
		self.current.lock().await.conversation(id, context).await
	}

	async fn scan_conversations(
		&self,
		query: &crate::types::ConversationQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &pi_chord::context::Context,
	) -> Page<crate::types::ConversationRecord> {
		self.current.lock().await.scan_conversations(query, limit, cursor, context).await
	}

	async fn entry(&self, id: EntryId, context: &pi_chord::context::Context) -> Option<(crate::types::EntryRecord, Seq)> {
		self.current.lock().await.entry(id, context).await
	}

	async fn entry_in(
		&self,
		conversation_id: ConversationId,
		id: EntryId,
		context: &pi_chord::context::Context,
	) -> Option<(crate::types::EntryRecord, Seq)> {
		self.current.lock().await.entry_in(conversation_id, id, context).await
	}

	async fn find_latest_head_marker(
		&self,
		conversation_id: ConversationId,
		at_or_before_entry_id: Option<EntryId>,
		context: &pi_chord::context::Context,
	) -> Option<crate::types::EntryRecord> {
		self.current.lock().await.find_latest_head_marker(conversation_id, at_or_before_entry_id, context).await
	}

	async fn scan_entries(
		&self,
		query: &EntryQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &pi_chord::context::Context,
	) -> Page<crate::types::EntryRecord> {
		self.current.lock().await.scan_entries(query, limit, cursor, context).await
	}

	async fn task(&self, id: TaskId, context: &pi_chord::context::Context) -> Option<TaskRecord> {
		self.current.lock().await.task(id, context).await
	}

	async fn scan_tasks(
		&self,
		query: &crate::types::TaskQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &pi_chord::context::Context,
	) -> Page<TaskRecord> {
		self.current.lock().await.scan_tasks(query, limit, cursor, context).await
	}

	async fn submission(&self, id: SubmissionId, context: &pi_chord::context::Context) -> Option<crate::types::SubmissionRecord> {
		self.current.lock().await.submission(id, context).await
	}

	async fn scan_submissions(
		&self,
		query: &SubmissionQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &pi_chord::context::Context,
	) -> Page<crate::types::SubmissionRecord> {
		self.current.lock().await.scan_submissions(query, limit, cursor, context).await
	}

	async fn submission_by_request(
		&self,
		conversation_id: ConversationId,
		request_id: &str,
		context: &pi_chord::context::Context,
	) -> Option<crate::types::SubmissionRecord> {
		self.current.lock().await.submission_by_request(conversation_id, request_id, context).await
	}

	async fn find_document(
		&self,
		address: &DocumentAddress,
		at: DocumentPoint,
		context: &pi_chord::context::Context,
	) -> Option<crate::types::DocumentRecord> {
		self.current.lock().await.find_document(address, at, context).await
	}

	async fn document(&self, id: DocumentId, at: DocumentPoint, context: &pi_chord::context::Context) -> Option<StoredDocument> {
		self.current.lock().await.document(id, at, context).await
	}

	async fn scan_documents(
		&self,
		query: &DocumentQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &pi_chord::context::Context,
	) -> Page<crate::types::DocumentRecord> {
		self.current.lock().await.scan_documents(query, limit, cursor, context).await
	}

	async fn close(&self, context: &pi_chord::context::Context) {
		if self.closed.swap(true, Ordering::Relaxed) {
			return;
		}
		self.current.lock().await.close(context).await;
	}
}

#[tokio::test]
async fn storage_conformance() {
	run_conformance(|| {
		Box::pin(async {
			let directory = temp_directory();
			open_shared(&directory).await
		})
	})
	.await;
}

#[tokio::test]
async fn storage_conformance_across_reopen() {
	run_conformance(|| {
		Box::pin(async {
			let directory = temp_directory();
			let storage = ReopeningStorage::open(directory).await;
			Arc::new(storage) as Arc<dyn Storage>
		})
	})
	.await;
}

// ---------------------------------------------------------------------------
// Publication and recovery
// ---------------------------------------------------------------------------

#[tokio::test]
async fn opens_through_the_default_host() {
	let directory = temp_directory();
	let storage = open_shared(&directory).await;
	create_root(&storage).await;
	assert_eq!(
		storage.conversation(ROOT_CONVERSATION_ID, ctx()).await,
		Some(record(json!({ "id": ROOT_CONVERSATION_ID.0 })))
	);
}

#[tokio::test]
async fn publishes_every_commit_before_reclaiming_current_only_document_and_terminal_task_sidecars() {
	let directory = temp_directory();
	let storage = open_shared(&directory).await;
	create_root(&storage).await;
	let task_id = TaskId(storage.mint_id().await);
	let document_id = DocumentId(storage.mint_id().await);
	commit_ok(&storage, vec![task_write(task_id.0, "ready")]).await;
	commit_ok(&storage, vec![document_create_write(document_id, "test.document", json!({ "count": 0 }))]).await;
	commit_ok(&storage, vec![document_change_delta(document_id, 1, json!([]))]).await;
	commit_ok(&storage, vec![document_change_base(document_id, 1, json!({ "count": 1 }))]).await;
	assert_eq!(read_lines(&sidecar(&directory, "task", task_id.0)).len(), 1);
	assert_eq!(read_lines(&sidecar(&directory, "doc", document_id.0)).len(), 1);

	commit_ok(&storage, vec![document_retire_write(document_id)]).await;
	commit_ok(&storage, vec![terminal_task_write(task_id.0)]).await;

	assert_eq!(read_lines(&directory.join("main.jsonl")).len(), 7);
	assert!(!file_exists(&sidecar(&directory, "task", task_id.0)));
	assert!(!file_exists(&sidecar(&directory, "doc", document_id.0)));
	let marker_types: Vec<String> = read_lines(&directory.join("main.jsonl"))
		.iter()
		.map(|line| serde_json::from_str::<Value>(line).unwrap()["type"].as_str().unwrap().to_owned())
		.collect();
	assert_eq!(marker_types, vec!["commit"; 7]);
}

#[tokio::test]
async fn orders_multiple_live_task_replacements_in_one_sidecar_by_commit_ordinal() {
	let directory = temp_directory();
	let storage = open_shared(&directory).await;
	create_root(&storage).await;
	let task_id = TaskId(storage.mint_id().await);
	commit_ok(&storage, vec![task_write(task_id.0, "first"), task_write(task_id.0, "second")]).await;
	assert_eq!(read_lines(&sidecar(&directory, "task", task_id.0)).len(), 2);

	let reopened = open_shared(&directory).await;
	assert_eq!(reopened.task(task_id, ctx()).await, Some(pending_task(task_id.0, "second")));
}

#[tokio::test]
async fn removes_a_complete_plus_torn_unconfirmed_multi_record_sidecar_append() {
	let directory = temp_directory();
	let host = Arc::new(InstrumentedHost::new());
	let storage: Arc<dyn Storage> = Arc::new(open_storage_at(&directory, Some(host.clone()), false).await);
	create_root(&storage).await;
	let task_id = TaskId(storage.mint_id().await);
	fail(&host, FailureOperation::Append, 1, FailureMode::Short);
	let error = commit(
		&storage,
		vec![task_write(task_id.0, "first"), task_write(task_id.0, &format!("second-{}", "x".repeat(512)))],
	)
	.await
	.expect_err("poisoned commit");
	assert!(error.message.contains("poisoned"), "{}", error.message);
	let partial = std::fs::read(sidecar(&directory, "task", task_id.0)).unwrap();
	assert_ne!(*partial.last().unwrap(), 0x0a);
	assert_eq!(partial.iter().filter(|byte| **byte == 0x0a).count(), 1);

	let reopened = open_shared(&directory).await;
	assert_eq!(reopened.task(task_id, ctx()).await, None);
	assert_eq!(std::fs::metadata(sidecar(&directory, "task", task_id.0)).unwrap().len(), 0);
}

#[tokio::test]
async fn serializes_the_complete_candidate_before_io_and_leaves_preparation_failures_usable() {
	let directory = temp_directory();
	let host = Arc::new(InstrumentedHost::new());
	let storage: Arc<dyn Storage> = Arc::new(open_storage_at(&directory, Some(host.clone()), false).await);
	create_root(&storage).await;
	host.clear();
	// Preparation failure (D3): a conversation rewrite is rejected before any
	// byte reaches the host; upstream used an unserializable bigint.
	let error = commit(&storage, vec![json!({ "type": "conversation", "value": { "id": ROOT_CONVERSATION_ID.0 } })])
		.await
		.expect_err("conversation rewrite must reject");
	assert!(error.message.contains("already belongs to conversation"), "{}", error.message);
	assert_eq!(operations(&host), Vec::<String>::new());
	assert_eq!(storage.entry(EntryId(2), ctx()).await, None);
	assert_eq!(
		commit_ok(
			&storage,
			vec![json!({ "type": "entry", "value": { "id": 2, "conversationId": ROOT_CONVERSATION_ID.0, "kind": "good" } })]
		)
		.await,
		Seq(2)
	);
}

async fn poisoned_append_case(failure: Failure) {
	let directory = temp_directory();
	let host = Arc::new(InstrumentedHost::new());
	let storage: Arc<dyn Storage> = Arc::new(open_storage_at(&directory, Some(host.clone()), false).await);
	create_root(&storage).await;
	let first_id = DocumentId(storage.mint_id().await);
	let second_id = DocumentId(storage.mint_id().await);
	fail(&host, failure.operation, failure.call, failure.mode);
	let error = commit(
		&storage,
		vec![
			document_create_write(first_id, "first", json!({ "text": "α" })),
			document_create_write(second_id, "second", json!({ "text": "β" })),
		],
	)
	.await
	.expect_err("poisoned commit");
	assert!(error.message.contains("poisoned"), "{}", error.message);
	// Adapted from upstream rejection (D2): poisoned reads collapse to `None`.
	assert_eq!(storage.document(first_id, DocumentPoint::Current, ctx()).await, None);

	let reopened = open_shared(&directory).await;
	let marker_survived = failure.call == 3 && failure.mode == FailureMode::After;
	let first = reopened.document(first_id, DocumentPoint::Current, ctx()).await;
	let second = reopened.document(second_id, DocumentPoint::Current, ctx()).await;
	if marker_survived {
		assert_eq!(first.unwrap().value, json!({ "text": "α" }).as_object().unwrap().clone());
		assert_eq!(second.unwrap().value, json!({ "text": "β" }).as_object().unwrap().clone());
	} else {
		assert_eq!(first, None);
		assert_eq!(second, None);
	}
}

#[tokio::test]
async fn poisons_after_append_failures_and_recovers_only_confirmed_state() {
	for &(operation, call, mode) in &[
		(FailureOperation::Append, 1, FailureMode::Before),
		(FailureOperation::Append, 1, FailureMode::After),
		(FailureOperation::Append, 1, FailureMode::Short),
		(FailureOperation::Append, 2, FailureMode::Before),
		(FailureOperation::Append, 2, FailureMode::After),
		(FailureOperation::Append, 3, FailureMode::Before),
		(FailureOperation::Append, 3, FailureMode::Short),
		(FailureOperation::Append, 3, FailureMode::After),
	] {
		poisoned_append_case(Failure { operation, call, mode }).await;
	}
}

async fn poisoned_flush_case(failure: Failure) {
	let directory = temp_directory();
	let host = Arc::new(InstrumentedHost::new());
	let storage: Arc<dyn Storage> = Arc::new(open_storage_at(&directory, Some(host.clone()), true).await);
	create_root(&storage).await;
	let first_id = DocumentId(storage.mint_id().await);
	let second_id = DocumentId(storage.mint_id().await);
	fail(&host, failure.operation, failure.call, failure.mode);
	let error = commit(
		&storage,
		vec![
			document_create_write(first_id, "flush.first", json!({})),
			document_create_write(second_id, "flush.second", json!({})),
		],
	)
	.await
	.expect_err("poisoned commit");
	assert!(error.message.contains("poisoned"), "{}", error.message);
	let reopened = open_shared(&directory).await;
	assert_eq!(reopened.document(first_id, DocumentPoint::Current, ctx()).await, None);
	assert_eq!(reopened.document(second_id, DocumentPoint::Current, ctx()).await, None);
}

#[tokio::test]
async fn poisons_after_flush_failures_and_never_writes_a_marker() {
	for &(operation, call, mode) in &[
		(FailureOperation::Flush, 1, FailureMode::Before),
		(FailureOperation::Flush, 1, FailureMode::After),
		(FailureOperation::Flush, 2, FailureMode::Before),
		(FailureOperation::Flush, 2, FailureMode::After),
	] {
		poisoned_flush_case(Failure { operation, call, mode }).await;
	}
}

async fn ordering_case(fsync: bool) {
	let directory = temp_directory();
	let host = Arc::new(InstrumentedHost::new());
	let storage: Arc<dyn Storage> = Arc::new(open_storage_at(&directory, Some(host.clone()), fsync).await);
	create_root(&storage).await;
	let first_id = DocumentId(storage.mint_id().await);
	let second_id = DocumentId(storage.mint_id().await);
	host.clear();
	commit_ok(
		&storage,
		vec![
			document_create_write(second_id, "second", json!({})),
			document_create_write(first_id, "first", json!({})),
		],
	)
	.await;
	let flush = if fsync { |file: String| vec![format!("flush:{file}")] } else { |_: String| Vec::new() };
	let mut expected = vec![format!("append:doc-{}.jsonl", second_id.0), format!("append:doc-{}.jsonl", first_id.0)];
	for file in [format!("doc-{}.jsonl", second_id.0), format!("doc-{}.jsonl", first_id.0)] {
		expected.extend(flush(file));
	}
	expected.push("append:main.jsonl".into());
	assert_operations(&host, &expected);

	host.clear();
	commit_ok(&storage, vec![document_change_base(first_id, 1, json!({ "checkpoint": true }))]).await;
	let first_file = format!("doc-{}.jsonl", first_id.0);
	let mut expected = vec![format!("append:{first_file}")];
	expected.extend(flush(first_file.clone()));
	expected.push("append:main.jsonl".into());
	if fsync {
		expected.push("flush:main.jsonl".into());
	}
	expected.push(format!("write:{first_file}.reclaim"));
	if fsync {
		expected.push(format!("flush:{first_file}.reclaim"));
	}
	expected.push(format!("rename:{first_file}.reclaim->{first_file}"));
	assert_operations(&host, &expected);

	host.clear();
	commit_ok(
		&storage,
		vec![json!({ "type": "entry", "value": { "id": storage.mint_id().await, "conversationId": ROOT_CONVERSATION_ID.0, "kind": "main-only" } })],
	)
	.await;
	assert_operations(&host, &["append:main.jsonl".to_owned()]);

	let task_id = TaskId(storage.mint_id().await);
	commit_ok(&storage, vec![task_write(task_id.0, "ready")]).await;
	host.clear();
	commit_ok(&storage, vec![terminal_task_write(task_id.0)]).await;
	let mut expected = vec!["append:main.jsonl".to_owned()];
	if fsync {
		expected.push("flush:main.jsonl".into());
	}
	expected.push(format!("remove:task-{}.jsonl", task_id.0));
	assert_operations(&host, &expected);
}

fn assert_operations(host: &InstrumentedHost, expected: &[String]) {
	assert_eq!(operations(host), expected);
}

#[tokio::test]
async fn orders_publication_flushes_exactly_and_flushes_main_only_to_authorize_reclamation() {
	ordering_case(false).await;
	ordering_case(true).await;
}

async fn reclaim_recovery_case(failure: Failure) {
	let directory = temp_directory();
	let host = Arc::new(InstrumentedHost::new());
	let storage: Arc<dyn Storage> = Arc::new(open_storage_at(&directory, Some(host.clone()), false).await);
	create_root(&storage).await;
	let id = DocumentId(storage.mint_id().await);
	commit_ok(&storage, vec![document_create_write(id, "test.document", json!({ "count": 0 }))]).await;
	commit_ok(&storage, vec![document_change_delta(id, 1, json!([["s", ["count"], 1]]))]).await;

	fail(&host, failure.operation, failure.call, failure.mode);
	assert_eq!(
		commit_ok(&storage, vec![document_change_base(id, 1, json!({ "count": 2 }))]).await,
		Seq(4)
	);
	assert_eq!(
		storage.document(id, DocumentPoint::Current, ctx()).await.unwrap().value,
		json!({ "count": 2 }).as_object().unwrap().clone()
	);
	storage.close(ctx()).await;

	let reopened = open_shared(&directory).await;
	assert_eq!(
		reopened.document(id, DocumentPoint::Current, ctx()).await.unwrap().value,
		json!({ "count": 2 }).as_object().unwrap().clone()
	);
	assert_eq!(read_lines(&sidecar(&directory, "doc", id.0)).len(), 1);
	let strays: Vec<String> = std::fs::read_dir(&directory)
		.unwrap()
		.map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
		.filter(|name| name.ends_with(".reclaim"))
		.collect();
	assert_eq!(strays, Vec::<String>::new());
}

#[tokio::test]
async fn recovers_a_committed_base_across_reclaim_failures() {
	for &(operation, call, mode) in &[
		(FailureOperation::Write, 1, FailureMode::Before),
		(FailureOperation::Write, 1, FailureMode::Short),
		(FailureOperation::Write, 1, FailureMode::After),
		(FailureOperation::Rename, 1, FailureMode::Before),
		(FailureOperation::Rename, 1, FailureMode::After),
	] {
		reclaim_recovery_case(Failure { operation, call, mode }).await;
	}
}

async fn retirement_reclamation_case(mode: FailureMode) {
	let directory = temp_directory();
	let host = Arc::new(InstrumentedHost::new());
	let storage: Arc<dyn Storage> = Arc::new(open_storage_at(&directory, Some(host.clone()), false).await);
	create_root(&storage).await;
	let task_id = TaskId(storage.mint_id().await);
	let id = DocumentId(storage.mint_id().await);
	commit_ok(
		&storage,
		vec![
			task_write(task_id.0, "ready"),
			json!({
				"type": "document.create",
				"record": { "id": id.0, "kind": "task.document", "scope": { "kind": "task", "taskId": task_id.0 } },
				"content": { "kind": "base", "version": 1, "value": { "count": 1 } },
			}),
		],
	)
	.await;
	fail(&host, FailureOperation::Remove, 1, mode);
	assert_eq!(commit_ok(&storage, vec![document_retire_write(id)]).await, Seq(3));
	assert_eq!(storage.document(id, DocumentPoint::Current, ctx()).await, None);
	storage.close(ctx()).await;

	let reopened = open_shared(&directory).await;
	assert_eq!(reopened.document(id, DocumentPoint::Current, ctx()).await, None);
	assert_eq!(reopened.task(task_id, ctx()).await, Some(pending_task(task_id.0, "ready")));
	assert!(!file_exists(&sidecar(&directory, "doc", id.0)));
	let strays: Vec<String> = std::fs::read_dir(&directory)
		.unwrap()
		.map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
		.filter(|name| name.ends_with(".reclaim"))
		.collect();
	assert_eq!(strays, Vec::<String>::new());
}

#[tokio::test]
async fn recovers_document_retirement_reclamation_across_remove_failures() {
	for mode in [FailureMode::Before, FailureMode::After] {
		retirement_reclamation_case(mode).await;
	}
}

async fn deferred_reclamation_case(mode: FailureMode) {
	let directory = temp_directory();
	let host = Arc::new(InstrumentedHost::new());
	let storage: Arc<dyn Storage> = Arc::new(open_storage_at(&directory, Some(host.clone()), true).await);
	create_root(&storage).await;
	let id = DocumentId(storage.mint_id().await);
	commit_ok(&storage, vec![document_create_write(id, "test.document", json!({ "count": 0 }))]).await;
	fail(&host, FailureOperation::Flush, 2, mode);
	assert_eq!(
		commit_ok(&storage, vec![document_change_base(id, 1, json!({ "count": 2 }))]).await,
		Seq(3)
	);
	assert_eq!(
		operations(&host),
		vec![
			format!("append:doc-{}.jsonl", id.0),
			format!("flush:doc-{}.jsonl", id.0),
			"append:main.jsonl".into(),
			"flush:main.jsonl".into(),
		]
	);
	assert_eq!(
		storage.document(id, DocumentPoint::Current, ctx()).await.unwrap().value,
		json!({ "count": 2 }).as_object().unwrap().clone()
	);
	assert_eq!(read_lines(&sidecar(&directory, "doc", id.0)).len(), 2);
	storage.close(ctx()).await;

	let recovery_host = Arc::new(InstrumentedHost::new());
	let deferred: Arc<dyn Storage> = Arc::new(open_storage_at(&directory, Some(recovery_host.clone()), true).await);
	fail(&recovery_host, FailureOperation::Flush, 1, mode);
	assert_eq!(
		deferred.document(id, DocumentPoint::Current, ctx()).await.unwrap().value,
		json!({ "count": 2 }).as_object().unwrap().clone()
	);
	assert_operations(&recovery_host, &["flush:main.jsonl".to_owned()]);
	assert_eq!(read_lines(&sidecar(&directory, "doc", id.0)).len(), 2);
	deferred.close(ctx()).await;

	let reclaimed = open_storage_at(&directory, None, true).await;
	assert_eq!(
		reclaimed.document(id, DocumentPoint::Current, ctx()).await.unwrap().value,
		json!({ "count": 2 }).as_object().unwrap().clone()
	);
	assert_eq!(read_lines(&sidecar(&directory, "doc", id.0)).len(), 1);
}

/// Ignored per module ledger D5: deferred reclamation settles on the next
/// commit rather than proactively at `open()`.
#[tokio::test]
#[ignore = "D5: deferred reclamation settles on the next commit, not at open()"]
async fn defers_reclamation_after_authorizing_main_flush_failures() {
	for mode in [FailureMode::Before, FailureMode::After] {
		deferred_reclamation_case(mode).await;
	}
}

async fn reclaim_temp_flush_case(mode: FailureMode) {
	let directory = temp_directory();
	let host = Arc::new(InstrumentedHost::new());
	let storage: Arc<dyn Storage> = Arc::new(open_storage_at(&directory, Some(host.clone()), true).await);
	create_root(&storage).await;
	let id = DocumentId(storage.mint_id().await);
	commit_ok(&storage, vec![document_create_write(id, "test.document", json!({ "count": 0 }))]).await;
	fail(&host, FailureOperation::Flush, 3, mode);
	assert_eq!(
		commit_ok(&storage, vec![document_change_base(id, 1, json!({ "count": 2 }))]).await,
		Seq(3)
	);
	host.clear();
	commit_ok(&storage, vec![document_change_delta(id, 1, json!([["s", ["count"], 3]]))]).await;
	storage.close(ctx()).await;

	let reopened = open_storage_at(&directory, None, true).await;
	assert_eq!(
		reopened.document(id, DocumentPoint::Current, ctx()).await.unwrap().value,
		json!({ "count": 3 }).as_object().unwrap().clone()
	);
	assert_eq!(read_lines(&sidecar(&directory, "doc", id.0)).len(), 2);
}

#[tokio::test]
async fn keeps_a_committed_base_usable_after_reclaim_temp_flush_failures() {
	for mode in [FailureMode::Before, FailureMode::After] {
		reclaim_temp_flush_case(mode).await;
	}
}

async fn terminal_reclamation_case(mode: FailureMode) {
	let directory = temp_directory();
	let host = Arc::new(InstrumentedHost::new());
	let storage: Arc<dyn Storage> = Arc::new(open_storage_at(&directory, Some(host.clone()), false).await);
	create_root(&storage).await;
	let id = TaskId(storage.mint_id().await);
	commit_ok(&storage, vec![task_write(id.0, "ready")]).await;
	fail(&host, FailureOperation::Remove, 1, mode);
	assert_eq!(commit_ok(&storage, vec![terminal_task_write(id.0)]).await, Seq(3));
	assert_eq!(storage.task(id, ctx()).await, Some(terminal_task(id.0)));
	storage.close(ctx()).await;

	let reopened = open_shared(&directory).await;
	assert_eq!(reopened.task(id, ctx()).await, Some(terminal_task(id.0)));
	assert!(!file_exists(&sidecar(&directory, "task", id.0)));
	let strays: Vec<String> = std::fs::read_dir(&directory)
		.unwrap()
		.map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
		.filter(|name| name.ends_with(".reclaim"))
		.collect();
	assert_eq!(strays, Vec::<String>::new());
}

#[tokio::test]
async fn recovers_terminal_task_reclamation_across_remove_failures() {
	for mode in [FailureMode::Before, FailureMode::After] {
		terminal_reclamation_case(mode).await;
	}
}

#[tokio::test]
async fn appends_later_deltas_to_the_replacement_sidecar_after_a_current_only_base() {
	let directory = temp_directory();
	let storage = open_shared(&directory).await;
	create_root(&storage).await;
	let id = DocumentId(storage.mint_id().await);
	commit_ok(&storage, vec![document_create_write(id, "test.document", json!({ "count": 0 }))]).await;
	commit_ok(&storage, vec![document_change_base(id, 1, json!({ "count": 10 }))]).await;
	commit_ok(&storage, vec![document_change_delta(id, 1, json!([["s", ["count"], 11]]))]).await;
	assert_eq!(read_lines(&sidecar(&directory, "doc", id.0)).len(), 2);
	let reopened = open_shared(&directory).await;
	assert_eq!(
		reopened.document(id, DocumentPoint::Current, ctx()).await.unwrap().value,
		json!({ "count": 11 }).as_object().unwrap().clone()
	);
}

#[tokio::test]
async fn never_reclaims_rewindable_document_history_including_after_a_base_and_retirement() {
	let directory = temp_directory();
	let storage = open_shared(&directory).await;
	create_root(&storage).await;
	let id = DocumentId(storage.mint_id().await);
	let record_json = json!({
		"id": id.0,
		"kind": "rewindable",
		"scope": { "kind": "conversation", "conversationId": ROOT_CONVERSATION_ID.0 },
		"history": "rewindable",
		"fork": "asOf",
	});
	let created_at = commit_ok(
		&storage,
		vec![json!({
			"type": "document.create",
			"record": record_json,
			"content": { "kind": "base", "version": 1, "value": { "count": 0 } },
		})],
	)
	.await;
	let changed_at = commit_ok(&storage, vec![document_change_delta(id, 1, json!([["s", ["count"], 1]]))]).await;
	commit_ok(&storage, vec![document_change_base(id, 1, json!({ "count": 2 }))]).await;
	commit_ok(&storage, vec![document_retire_write(id)]).await;

	assert_eq!(read_lines(&sidecar(&directory, "doc", id.0)).len(), 3);
	let reopened = open_shared(&directory).await;
	assert_eq!(
		reopened.document(id, DocumentPoint::At(created_at), ctx()).await.unwrap().value,
		json!({ "count": 0 }).as_object().unwrap().clone()
	);
	assert_eq!(
		reopened.document(id, DocumentPoint::At(changed_at), ctx()).await.unwrap().value,
		json!({ "count": 1 }).as_object().unwrap().clone()
	);
	assert_eq!(reopened.document(id, DocumentPoint::Current, ctx()).await, None);
	assert_eq!(read_lines(&sidecar(&directory, "doc", id.0)).len(), 3);
}

#[tokio::test]
async fn reclaims_retired_task_session_and_latest_conversation_document_sidecars() {
	let directory = temp_directory();
	let storage = open_shared(&directory).await;
	create_root(&storage).await;
	let task_id = TaskId(storage.mint_id().await);
	let session_id = DocumentId(storage.mint_id().await);
	let latest_id = DocumentId(storage.mint_id().await);
	let task_document_id = DocumentId(storage.mint_id().await);
	let created_at = commit_ok(
		&storage,
		vec![
			task_write(task_id.0, "ready"),
			json!({
				"type": "document.create",
				"record": session_document(session_id.0, "session"),
				"content": { "kind": "base", "version": 1, "value": {} },
			}),
			json!({
				"type": "document.create",
				"record": {
					"id": latest_id.0,
					"kind": "latest",
					"scope": { "kind": "conversation", "conversationId": ROOT_CONVERSATION_ID.0 },
					"history": "latest",
					"fork": "current",
				},
				"content": { "kind": "base", "version": 1, "value": {} },
			}),
			json!({
				"type": "document.create",
				"record": { "id": task_document_id.0, "kind": "task", "scope": { "kind": "task", "taskId": task_id.0 } },
				"content": { "kind": "base", "version": 1, "value": {} },
			}),
		],
	)
	.await;
	let retired_at = commit_ok(
		&storage,
		vec![
			document_retire_write(session_id),
			document_retire_write(latest_id),
			document_retire_write(task_document_id),
			terminal_task_write(task_id.0),
		],
	)
	.await;
	for id in [session_id, latest_id, task_document_id] {
		assert!(!file_exists(&sidecar(&directory, "doc", id.0)));
	}
	assert!(!file_exists(&sidecar(&directory, "task", task_id.0)));

	let reopened = open_shared(&directory).await;
	assert_eq!(reopened.task(task_id, ctx()).await, Some(terminal_task(task_id.0)));
	assert_eq!(reopened.document(session_id, DocumentPoint::Current, ctx()).await, None);
	assert_eq!(reopened.document(latest_id, DocumentPoint::Current, ctx()).await, None);
	assert_eq!(reopened.document(task_document_id, DocumentPoint::Current, ctx()).await, None);
	let found = reopened
		.find_document(
			&record::<crate::types::DocumentAddress>(json!({ "kind": "session", "scope": { "kind": "session" } })),
			DocumentPoint::At(created_at),
			ctx(),
		)
		.await
		.unwrap();
	assert_eq!((found.id, found.created_at, found.retired_at), (session_id, created_at, Some(retired_at)));
	assert_eq!(
		reopened
			.find_document(
				&record::<crate::types::DocumentAddress>(json!({ "kind": "session", "scope": { "kind": "session" } })),
				DocumentPoint::At(retired_at),
				ctx(),
			)
			.await,
		None
	);
}

#[tokio::test]
async fn truncates_torn_utf8_tails_at_exact_byte_offsets_and_reuses_the_unconfirmed_sequence() {
	let directory = temp_directory();
	let storage = open_shared(&directory).await;
	create_root(&storage).await;
	let document_id = DocumentId(storage.mint_id().await);
	commit_ok(&storage, vec![document_create_write(document_id, "test.document", json!({ "text": "kept" }))]).await;
	let sidecar_path = sidecar(&directory, "doc", document_id.0);
	let main_path = directory.join("main.jsonl");
	let sidecar_size = std::fs::metadata(&sidecar_path).unwrap().len();
	let main_size = std::fs::metadata(&main_path).unwrap().len();
	let torn = b"{\"text\":\"\xe2\x82";
	append_bytes(&sidecar_path, &torn[..torn.len() - 1]);
	append_bytes(&main_path, &torn[..torn.len() - 1]);

	let reopened = open_shared(&directory).await;
	assert_eq!(std::fs::metadata(&sidecar_path).unwrap().len(), sidecar_size);
	assert_eq!(std::fs::metadata(&main_path).unwrap().len(), main_size);
	assert_eq!(
		commit_ok(&reopened, vec![document_change_delta(document_id, 1, json!([]))]).await,
		Seq(3)
	);
}

#[tokio::test]
async fn removes_complete_unconfirmed_sidecar_tails_without_resurrecting_terminal_tasks() {
	let directory = temp_directory();
	let storage = open_shared(&directory).await;
	create_root(&storage).await;
	let task_id = TaskId(storage.mint_id().await);
	commit_ok(&storage, vec![task_write(task_id.0, "ready")]).await;
	commit_ok(&storage, vec![terminal_task_write(task_id.0)]).await;
	let sidecar_path = sidecar(&directory, "task", task_id.0);
	assert!(!file_exists(&sidecar_path));
	let stale = json!({
		"format": 1,
		"type": "record",
		"seq": 4,
		"ordinal": 0,
		"payload": { "type": "task", "value": pending_task_json(task_id.0, "stale") },
	});
	append_bytes(&sidecar_path, format!("{stale}\n").as_bytes());

	let reopened = open_shared(&directory).await;
	assert!(!file_exists(&sidecar_path));
	assert_eq!(reopened.task(task_id, ctx()).await, Some(terminal_task(task_id.0)));
	assert_eq!(commit_ok(&reopened, vec![]).await, Seq(4));
}

fn expect_corruption(result: Result<JsonlStorage, JsonlError>, message: &str) {
	match result {
		Err(JsonlError::Corruption(corruption)) => {
			assert!(corruption.message.contains(message), "{}", corruption.message)
		}
		Err(other) => panic!("expected corruption error containing {message:?}, got {other}"),
		Ok(_) => panic!("expected corruption error containing {message:?}, but open succeeded"),
	}
}

#[tokio::test]
async fn fails_open_when_confirmed_sidecar_data_is_missing() {
	let directory = temp_directory();
	let storage: Arc<dyn Storage> = Arc::new(open_storage(&directory).await);
	create_root(&storage).await;
	let document_id = DocumentId(storage.mint_id().await);
	commit_ok(
		&storage,
		vec![document_create_write(document_id, "test.document", json!({}))],
	)
	.await;
	std::fs::write(sidecar(&directory, "doc", document_id.0), "").unwrap();
	let result = JsonlStorage::open(&directory, JsonlStorageOptions { fsync: false }).await;
	expect_corruption(result, "Missing confirmed sidecar record");
}

#[tokio::test]
async fn rejects_a_confirmed_record_after_an_unconfirmed_sidecar_record() {
	let directory = temp_directory();
	let storage = open_storage(&directory).await;
	let shared: Arc<dyn Storage> = Arc::new(storage);
	create_root(&shared).await;
	let document_id = DocumentId(shared.mint_id().await);
	commit_ok(&shared, vec![document_create_write(document_id, "test.document", json!({ "count": 0 }))]).await;
	commit_ok(&shared, vec![document_change_delta(document_id, 1, json!([["s", ["count"], 1]]))]).await;
	let path = sidecar(&directory, "doc", document_id.0);
	let lines = read_lines(&path);
	let (first, second) = (lines[0].clone(), lines[1].clone());
	let unconfirmed = json!({
		"format": 1,
		"type": "record",
		"seq": 2,
		"ordinal": 999,
		"payload": { "type": "document", "id": document_id.0, "content": { "kind": "delta", "version": 1, "ops": [] } },
	});
	std::fs::write(&path, format!("{first}\n{unconfirmed}\n{second}\n")).unwrap();
	let result = JsonlStorage::open(&directory, JsonlStorageOptions { fsync: false }).await;
	expect_corruption(result, "Confirmed record follows an unconfirmed tail");
}

#[tokio::test]
async fn rejects_non_increasing_main_commit_sequences() {
	let directory = temp_directory();
	let storage = open_storage(&directory).await;
	let shared: Arc<dyn Storage> = Arc::new(storage);
	create_root(&shared).await;
	let path = directory.join("main.jsonl");
	let contents = std::fs::read(&path).unwrap();
	append_bytes(&path, &contents);
	let result = JsonlStorage::open(&directory, JsonlStorageOptions { fsync: false }).await;
	expect_corruption(result, "Commit sequence does not strictly increase");
}

#[tokio::test]
async fn rejects_structurally_invalid_confirmed_document_content() {
	let directory = temp_directory();
	let storage = open_storage(&directory).await;
	let shared: Arc<dyn Storage> = Arc::new(storage);
	create_root(&shared).await;
	let document_id = DocumentId(shared.mint_id().await);
	commit_ok(&shared, vec![document_create_write(document_id, "test.document", json!({}))]).await;
	let path = sidecar(&directory, "doc", document_id.0);
	let mut record_value: Value = serde_json::from_str(read_lines(&path)[0].trim()).unwrap();
	record_value["payload"]["content"]
		.as_object_mut()
		.unwrap()
		.remove("value")
		.expect("content has a value");
	std::fs::write(&path, format!("{record_value}\n")).unwrap();
	let result = JsonlStorage::open(&directory, JsonlStorageOptions { fsync: false }).await;
	expect_corruption(result, "Invalid document content");
}

#[tokio::test]
async fn rejects_malformed_complete_main_and_sidecar_lines() {
	let main_directory = temp_directory();
	let main_storage = open_storage(&main_directory).await;
	create_root(&(Arc::new(main_storage) as Arc<dyn Storage>)).await;
	append_bytes(&main_directory.join("main.jsonl"), b"{bad}\n");
	let result = JsonlStorage::open(&main_directory, JsonlStorageOptions { fsync: false }).await;
	expect_corruption(result, "Malformed complete main.jsonl");

	let sidecar_directory = temp_directory();
	let sidecar_storage = open_storage(&sidecar_directory).await;
	create_root(&(Arc::new(sidecar_storage) as Arc<dyn Storage>)).await;
	std::fs::write(sidecar_directory.join("doc-99.jsonl"), "{bad}\n").unwrap();
	let result = JsonlStorage::open(&sidecar_directory, JsonlStorageOptions { fsync: false }).await;
	expect_corruption(result, "Malformed complete doc-99.jsonl");
}
