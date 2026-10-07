//! Port of `packages/durable/test/sqlite-storage.test.ts` and
//! `sqlite-facade.test.ts`, plus the shared conformance suite run fresh and
//! across reopen.
//!
//! Deviations (module ledger, see `mod.rs` D1/D2/D3):
//! - Facade cases that exist only to pin the Node async-adapter semantics have
//!   no portable counterpart and are deferred: "prepares each storage statement
//!   once" (rusqlite prepares per call; there is no statement cache to count),
//!   "awaits async transaction settlement" (`ControlledSettlementDatabase` has
//!   no counterpart — D2), "rejects a transaction handle used after its
//!   transaction settles" (the `&mut Connection` borrow cannot outlive the
//!   transaction), and "queues database calls made synchronously by a
//!   transaction" (Rust callbacks cannot re-enter the occupied connection).
//! - Read paths are infallible (trait D3), so upstream rejection cases
//!   (`missing a required base`, corrupt replay ops, reads after close) assert
//!   `None` / empty pages.
//! - "rolls SQL rows and sequence allocation back as one transaction" uses a
//!   conversation rewrite as the guaranteed preparation failure (upstream used
//!   a circular JSON structure, unrepresentable in `serde_json::Value`).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use serde_json::{Value, json};

use super::database::{NodeSqliteStorageOptions, SqliteDatabase, SqliteStorageError};
use super::storage::SqliteStorage;
use crate::storage::conformance::{ROOT_CONVERSATION_ID, at, commit, commit_ok, ctx, record, run_conformance};
use crate::types::{
	ConversationId, Cursor, DocumentAddress, DocumentId, DocumentPoint, DocumentQuery, EntryId, EntryQuery, Page, Seq,
	Storage, StorageWrite, StoredDocument, SubmissionId, SubmissionQuery, TaskId, TaskRecord,
};

static DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path() -> PathBuf {
	let directory = std::env::temp_dir().join(format!(
		"pi-durable-sqlite-{}-{}",
		std::process::id(),
		DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed)
	));
	let _ = std::fs::remove_dir_all(&directory);
	directory.join("storage.sqlite")
}

fn options() -> NodeSqliteStorageOptions {
	NodeSqliteStorageOptions::default()
}

async fn open_memory() -> SqliteStorage {
	SqliteStorage::open(SqliteDatabase::open(":memory:", &options()).unwrap()).await.unwrap()
}

async fn open_path(path: &Path) -> SqliteStorage {
	SqliteStorage::open_path(path, &options()).await.unwrap()
}

async fn open_shared_path(path: &Path) -> Arc<dyn Storage> {
	Arc::new(open_path(path).await)
}

async fn create_root(storage: &Arc<dyn Storage>) {
	commit_ok(storage, vec![json!({ "type": "conversation", "value": { "id": ROOT_CONVERSATION_ID.0 } })]).await;
}

fn entry_json(id: u64, conversation: u64, data: Option<Value>) -> Value {
	let mut value = json!({ "id": id, "conversationId": conversation, "kind": "message" });
	if let Some(data) = data {
		value["data"] = data;
	}
	value
}

fn entry_write(id: u64, conversation: u64, data: Option<Value>) -> Value {
	json!({ "type": "entry", "value": entry_json(id, conversation, data) })
}

fn pending_task_json(id: u64) -> Value {
	json!({
		"id": id,
		"conversationId": ROOT_CONVERSATION_ID.0,
		"kind": "test.task",
		"version": 1,
		"input": null,
		"state": { "status": "pending", "checkpoint": { "phase": "ready" } },
		"background": false,
		"abortRequested": false,
	})
}

fn raw_connection(path: &Path) -> rusqlite::Connection {
	rusqlite::Connection::open_with_flags(
		path,
		rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
	)
	.unwrap()
}

fn scalar(connection: &rusqlite::Connection, sql: &str) -> i64 {
	connection.query_row(sql, [], |row| row.get(0)).unwrap()
}

fn revision_count(path: &Path, document_id: DocumentId) -> i64 {
	let connection = raw_connection(path);
	connection
		.query_row("SELECT count(*) FROM document_revisions WHERE document_id = ?1", [document_id.0], |row| row.get(0))
		.unwrap()
}

// ---------------------------------------------------------------------------
// Conformance (fresh + across reopen)
// ---------------------------------------------------------------------------

/// Storage wrapper that closes and reopens the underlying SQLite file after
/// every commit (upstream `ReopeningStorage`).
struct ReopeningStorage {
	current: tokio::sync::Mutex<Arc<SqliteStorage>>,
	path: PathBuf,
	closed: AtomicBool,
}

impl ReopeningStorage {
	async fn open(path: PathBuf) -> Self {
		let current = open_path(&path).await;
		Self { current: tokio::sync::Mutex::new(Arc::new(current)), path, closed: AtomicBool::new(false) }
	}
}

#[async_trait::async_trait]
impl Storage for ReopeningStorage {
	async fn commit(&self, writes: &[StorageWrite], context: &pi_chord::context::Context) -> Result<Seq, crate::errors::StorageRejected> {
		let current = self.current.lock().await.clone();
		let result = current.commit(writes, context).await;
		if !self.closed.load(Ordering::Relaxed) {
			current.close(context).await;
			let reopened = open_path(&self.path).await;
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
			let storage = open_memory().await;
			Arc::new(storage) as Arc<dyn Storage>
		})
	})
	.await;
}

#[tokio::test]
async fn storage_conformance_across_reopen() {
	run_conformance(|| {
		Box::pin(async {
			let path = temp_path();
			let storage = ReopeningStorage::open(path).await;
			Arc::new(storage) as Arc<dyn Storage>
		})
	})
	.await;
}

// ---------------------------------------------------------------------------
// sqlite-storage.test.ts
// ---------------------------------------------------------------------------

#[tokio::test]
async fn persists_records_sequence_allocation_and_global_id_allocation_across_reopen() {
	let path = temp_path();
	let storage = open_shared_path(&path).await;
	create_root(&storage).await;
	let entry_id = storage.mint_id().await;
	assert_eq!(
		commit_ok(&storage, vec![entry_write(entry_id, ROOT_CONVERSATION_ID.0, None)]).await,
		Seq(2)
	);
	storage.close(ctx()).await;

	let reopened = open_shared_path(&path).await;
	let (stored, commit_seq) = reopened.entry(EntryId(entry_id), ctx()).await.unwrap();
	assert_eq!(stored, record(entry_json(entry_id, ROOT_CONVERSATION_ID.0, None)));
	assert_eq!(commit_seq, Seq(2));
	assert_eq!(reopened.mint_id().await, entry_id + 1);
	let task_id = reopened.mint_id().await;
	assert_eq!(
		commit_ok(&reopened, vec![json!({ "type": "task", "value": pending_task_json(task_id) })]).await,
		Seq(3)
	);
}

#[tokio::test]
async fn rejects_persisted_metadata_corruption_on_reopen() {
	let path = temp_path();
	let storage = open_path(&path).await;
	storage.close(ctx()).await;
	{
		let connection = rusqlite::Connection::open(&path).unwrap();
		connection.execute("DELETE FROM durable_metadata", []).unwrap();
	}
	let error = match SqliteStorage::open_path(&path, &options()).await {
		Err(error) => error,
		Ok(_) => panic!("metadata missing"),
	};
	assert!(error.to_string().contains("Durable SQLite metadata is missing"), "{error}");
}

#[tokio::test]
async fn rejects_a_document_whose_required_base_is_missing() {
	let path = temp_path();
	let storage = open_shared_path(&path).await;
	create_root(&storage).await;
	let id = storage.mint_id().await;
	commit_ok(
		&storage,
		vec![json!({
			"type": "document.create",
			"record": { "id": id, "kind": "corrupt", "scope": { "kind": "session" } },
			"content": { "kind": "base", "version": 1, "value": { "retained": true } },
		})],
	)
	.await;
	{
		let connection = rusqlite::Connection::open(&path).unwrap();
		connection.execute("DELETE FROM document_revisions WHERE document_id = ?1", [id]).unwrap();
	}
	// Adapted from upstream rejection (D3): corrupt reads collapse to `None`.
	assert_eq!(storage.document(DocumentId(id), DocumentPoint::Current, ctx()).await, None);
}

#[tokio::test]
async fn replays_detached_root_replacements_and_follow_up_edits() {
	let path = temp_path();
	let storage = open_shared_path(&path).await;
	create_root(&storage).await;
	let id = DocumentId(storage.mint_id().await);
	commit_ok(
		&storage,
		vec![json!({
			"type": "document.create",
			"record": { "id": id.0, "kind": "replay", "scope": { "kind": "session" } },
			"content": { "kind": "base", "version": 1, "value": { "nested": { "value": 1 }, "rows": [] } },
		})],
	)
	.await;
	commit_ok(
		&storage,
		vec![json!({
			"type": "document.change",
			"id": id.0,
			"content": { "kind": "delta", "version": 1, "ops": [["r", { "nested": { "value": 2 }, "rows": [{ "id": 1 }] }]] },
		})],
	)
	.await;
	commit_ok(
		&storage,
		vec![json!({
			"type": "document.change",
			"id": id.0,
			"content": {
				"kind": "delta",
				"version": 1,
				"ops": [["s", ["nested", "value"], 3], ["p", ["rows"], 1, 0, [{ "id": 2 }]], ["m", ["rows"], [1, 0]]],
			},
		})],
	)
	.await;
	commit_ok(
		&storage,
		vec![json!({
			"type": "document.change",
			"id": id.0,
			"content": { "kind": "delta", "version": 1, "ops": [["s", ["nested", "value"], 4]] },
		})],
	)
	.await;

	let expected = json!({ "nested": { "value": 4 }, "rows": [{ "id": 2 }, { "id": 1 }] });
	let first = storage.document(id, DocumentPoint::Current, ctx()).await.unwrap();
	assert_eq!(first.value, expected.as_object().unwrap().clone());
	// Returned values are detached: mutating them cannot affect stored state.
	let mut mutated = first.value.clone();
	mutated.insert("nested".into(), json!({ "value": 99 }));
	let rows = mutated.get_mut("rows").unwrap().as_array_mut().unwrap();
	rows[0].as_object_mut().unwrap().insert("id".into(), json!(99));
	assert_eq!(
		storage.document(id, DocumentPoint::Current, ctx()).await.unwrap().value,
		expected.as_object().unwrap().clone()
	);

	{
		let connection = rusqlite::Connection::open(&path).unwrap();
		connection
			.execute(
				"UPDATE document_revisions SET content = '[\"unknown\"]' WHERE document_id = ?1 AND seq = \
				 (SELECT max(seq) FROM document_revisions WHERE document_id = ?1)",
				[id.0],
			)
			.unwrap();
	}
	// Adapted from upstream rejection (D3): a corrupt replay op reads as `None`.
	assert_eq!(storage.document(id, DocumentPoint::Current, ctx()).await, None);
}

#[tokio::test]
async fn rolls_sql_rows_and_sequence_allocation_back_as_one_transaction() {
	let path = temp_path();
	let storage = open_shared_path(&path).await;
	create_root(&storage).await;
	// Guaranteed preparation failure (D3, ledger): conversation rewrite.
	let error = commit(&storage, vec![json!({ "type": "conversation", "value": { "id": ROOT_CONVERSATION_ID.0 } })])
		.await
		.expect_err("conversation rewrite must reject");
	assert!(
		error.message.contains(&format!("ID {} already belongs to conversation", ROOT_CONVERSATION_ID.0)),
		"{}",
		error.message
	);
	assert_eq!(storage.entry(EntryId(3), ctx()).await, None);
	let committed_id = storage.mint_id().await;
	assert_eq!(
		commit_ok(&storage, vec![entry_write(committed_id, ROOT_CONVERSATION_ID.0, None)]).await,
		Seq(2)
	);
	{
		let connection = raw_connection(&path);
		assert_eq!(scalar(&connection, "SELECT count(*) FROM entries"), 1);
		assert_eq!(
			scalar(&connection, "SELECT next_seq FROM durable_metadata WHERE singleton = 1"),
			3
		);
	}
}

#[tokio::test]
async fn reconstructs_recent_and_ancient_rewindable_points_after_reopen() {
	let path = temp_path();
	let storage = open_shared_path(&path).await;
	create_root(&storage).await;
	let id = DocumentId(storage.mint_id().await);
	let created_at = commit_ok(
		&storage,
		vec![json!({
			"type": "document.create",
			"record": {
				"id": id.0,
				"kind": "history",
				"scope": { "kind": "conversation", "conversationId": ROOT_CONVERSATION_ID.0 },
				"history": "rewindable",
				"fork": "asOf",
			},
			"content": { "kind": "base", "version": 1, "value": { "count": 0 } },
		})],
	)
	.await;
	let mut ancient_at = created_at;
	let mut recent_at = created_at;
	for count in 1..=40u64 {
		let content = if count == 20 {
			json!({ "kind": "base", "version": 1, "value": { "count": count } })
		} else {
			json!({ "kind": "delta", "version": 1, "ops": [["s", ["count"], count]] })
		};
		recent_at = commit_ok(&storage, vec![json!({ "type": "document.change", "id": id.0, "content": content })]).await;
		if count == 5 {
			ancient_at = recent_at;
		}
	}
	storage.close(ctx()).await;
	let reopened = open_shared_path(&path).await;
	assert_eq!(
		reopened.document(id, at(ancient_at), ctx()).await.unwrap().value,
		json!({ "count": 5 }).as_object().unwrap().clone()
	);
	assert_eq!(
		reopened.document(id, at(recent_at), ctx()).await.unwrap().value,
		json!({ "count": 40 }).as_object().unwrap().clone()
	);
}

#[tokio::test]
async fn uses_indexes_for_exact_addresses_scopes_entry_history_and_revision_tails() {
	let path = temp_path();
	{
		let storage = open_path(&path).await;
		storage.close(ctx()).await;
	}
	let connection = raw_connection(&path);
	let plans = [
		"EXPLAIN QUERY PLAN SELECT record FROM documents\n\t\t\t\t\tWHERE kind = 'kind' AND scope_kind = 'session' AND owner_id = 0 AND family = 0 AND key_value = ''\n\t\t\t\t\tAND retired_at IS NULL ORDER BY created_at DESC LIMIT 1",
		"EXPLAIN QUERY PLAN SELECT record FROM documents\n\t\t\t\t\tWHERE kind = 'kind' AND scope_kind = 'conversation' AND owner_id = 1 AND family = 0 AND key_value = ''\n\t\t\t\t\tAND created_at <= 10 AND (retired_at IS NULL OR retired_at > 10)\n\t\t\t\t\tORDER BY created_at DESC LIMIT 1",
		"EXPLAIN QUERY PLAN SELECT record FROM documents\n\t\t\t\t\tWHERE scope_kind = 'task' AND owner_id = 1 AND kind = 'kind' AND id > 0 ORDER BY id LIMIT 10",
		"EXPLAIN QUERY PLAN SELECT record FROM entries\n\t\t\t\t\tWHERE conversation_id = 1 AND id <= 10 ORDER BY id DESC LIMIT 10",
		"EXPLAIN QUERY PLAN SELECT record FROM entries\n\t\t\t\t\tWHERE conversation_id = 1 AND head IS NOT NULL AND id <= 10 ORDER BY id DESC LIMIT 1",
		"EXPLAIN QUERY PLAN SELECT record FROM tasks WHERE status = 'pending' AND id > 0 ORDER BY id LIMIT 10",
		"EXPLAIN QUERY PLAN SELECT seq, kind, version, content FROM document_revisions\n\t\t\t\t\tWHERE document_id = 1 AND kind = 'base' AND seq <= 10 ORDER BY seq DESC LIMIT 1",
		"EXPLAIN QUERY PLAN SELECT seq, kind, version, content FROM document_revisions\n\t\t\t\t\tWHERE document_id = 1 AND seq > 5 AND seq <= 10 ORDER BY seq",
	];
	let details: Vec<String> = plans
		.iter()
		.map(|sql| {
			let mut rows = Vec::new();
			let mut statement = connection.prepare(sql).unwrap();
			let mapped = statement.query_map([], |row| row.get::<_, String>(3)).unwrap();
			for row in mapped {
				rows.push(row.unwrap());
			}
			rows.join("\n")
		})
		.collect();
	assert!(details[0].contains("documents_by_address"), "{}", details[0]);
	assert!(details[1].contains("documents_by_address"), "{}", details[1]);
	assert!(details[2].contains("documents_by_scope_kind"), "{}", details[2]);
	assert!(details[3].contains("entries_by_conversation"), "{}", details[3]);
	assert!(details[4].contains("entry_heads_by_conversation"), "{}", details[4]);
	assert!(details[5].contains("tasks_by_status"), "{}", details[5]);
	assert!(details[6].contains("document_revisions_by_kind"), "{}", details[6]);
	assert!(details[7].contains("sqlite_autoindex_document_revisions_1"), "{}", details[7]);
	for detail in &details[0..3] {
		assert!(!detail.contains("SCAN documents"), "{detail}");
	}
	assert!(!details[7].contains("SCAN document_revisions"), "{}", details[7]);
}

#[tokio::test]
async fn reclaims_current_only_revisions_only_after_a_base_or_retirement() {
	let path = temp_path();
	let storage = open_shared_path(&path).await;
	create_root(&storage).await;
	let id = DocumentId(storage.mint_id().await);
	commit_ok(
		&storage,
		vec![json!({
			"type": "document.create",
			"record": { "id": id.0, "kind": "latest", "scope": { "kind": "session" } },
			"content": { "kind": "base", "version": 1, "value": { "count": 0 } },
		})],
	)
	.await;
	for count in 1..=10u64 {
		commit_ok(
			&storage,
			vec![json!({
				"type": "document.change",
				"id": id.0,
				"content": { "kind": "delta", "version": 1, "ops": [["s", ["count"], count]] },
			})],
		)
		.await;
	}
	assert_eq!(revision_count(&path, id), 11);
	{
		let connection = raw_connection(&path);
		let content: String = connection
			.query_row(
				"SELECT content FROM document_revisions WHERE document_id = ?1 AND kind = 'delta' ORDER BY seq DESC LIMIT 1",
				[id.0],
				|row| row.get(0),
			)
			.unwrap();
		assert_eq!(serde_json::from_str::<Value>(&content).unwrap(), json!([["s", ["count"], 10]]));
	}
	commit_ok(
		&storage,
		vec![json!({
			"type": "document.change",
			"id": id.0,
			"content": { "kind": "base", "version": 1, "value": { "count": 11 } },
		})],
	)
	.await;
	assert_eq!(revision_count(&path, id), 1);
	commit_ok(
		&storage,
		vec![json!({
			"type": "document.change",
			"id": id.0,
			"content": { "kind": "delta", "version": 1, "ops": [["r", { "count": 12 }]] },
		})],
	)
	.await;
	assert_eq!(revision_count(&path, id), 2);
	commit_ok(&storage, vec![json!({ "type": "document.retire", "id": id.0 })]).await;
	assert_eq!(revision_count(&path, id), 0);
}

#[tokio::test]
async fn auto_checkpoints_wal_frames_and_truncates_the_wal_on_close() {
	let path = temp_path();
	let wal_options = NodeSqliteStorageOptions { wal_auto_checkpoint_pages: Some(1), ..options() };
	let storage: Arc<dyn Storage> =
		Arc::new(SqliteStorage::open(SqliteDatabase::open(&path, &wal_options).unwrap()).await.unwrap());
	create_root(&storage).await;
	for index in 0..20u64 {
		commit_ok(
			&storage,
			vec![entry_write(storage.mint_id().await, ROOT_CONVERSATION_ID.0, Some(json!({
				"text": "x".repeat(32 * 1024),
				"index": index,
			})))],
		)
		.await;
	}
	let wal_path = PathBuf::from(format!("{}-wal", path.display()));
	let wal_size = std::fs::metadata(&wal_path).map(|metadata| metadata.len()).unwrap_or(0);
	assert!(wal_size < 512 * 1024, "{wal_size}");
	{
		let connection = raw_connection(&path);
		assert_eq!(scalar(&connection, "SELECT count(*) FROM entries"), 20);
	}
	storage.close(ctx()).await;
	let wal_size = std::fs::metadata(&wal_path).map(|metadata| metadata.len()).unwrap_or(0);
	assert_eq!(wal_size, 0);
}

#[tokio::test]
async fn reuses_pages_released_by_current_only_checkpoints() {
	let path = temp_path();
	let storage = open_shared_path(&path).await;
	create_root(&storage).await;
	let id = DocumentId(storage.mint_id().await);
	let large = "x".repeat(512 * 1024);
	commit_ok(
		&storage,
		vec![json!({
			"type": "document.create",
			"record": { "id": id.0, "kind": "reuse", "scope": { "kind": "session" } },
			"content": { "kind": "base", "version": 1, "value": { "text": large } },
		})],
	)
	.await;
	commit_ok(
		&storage,
		vec![json!({
			"type": "document.change",
			"id": id.0,
			"content": { "kind": "base", "version": 1, "value": { "text": "small" } },
		})],
	)
	.await;
	let (pages_after_delete, free_after_delete) = {
		let connection = raw_connection(&path);
		(scalar(&connection, "SELECT page_count FROM pragma_page_count()"), scalar(&connection, "SELECT freelist_count FROM pragma_freelist_count()"))
	};
	assert!(free_after_delete > 0);
	commit_ok(
		&storage,
		vec![json!({
			"type": "document.change",
			"id": id.0,
			"content": { "kind": "base", "version": 1, "value": { "text": large } },
		})],
	)
	.await;
	{
		let connection = raw_connection(&path);
		let pages_after_reuse = scalar(&connection, "SELECT page_count FROM pragma_page_count()");
		let free_after_reuse = scalar(&connection, "SELECT freelist_count FROM pragma_freelist_count()");
		assert!(pages_after_reuse <= pages_after_delete + 2, "{pages_after_reuse} <= {pages_after_delete}");
		assert!(free_after_reuse < free_after_delete, "{free_after_reuse} < {free_after_delete}");
	}
}

#[tokio::test]
async fn keeps_representative_row_and_document_storage_bounded() {
	let path = temp_path();
	let storage = open_shared_path(&path).await;
	create_root(&storage).await;
	for index in 0..100u64 {
		commit_ok(
			&storage,
			vec![entry_write(storage.mint_id().await, ROOT_CONVERSATION_ID.0, Some(json!({
				"index": index,
				"text": "x".repeat(1024),
			})))],
		)
		.await;
	}
	let document_id = storage.mint_id().await;
	commit_ok(
		&storage,
		vec![json!({
			"type": "document.create",
			"record": {
				"id": document_id,
				"kind": "size.history",
				"scope": { "kind": "conversation", "conversationId": ROOT_CONVERSATION_ID.0 },
				"history": "rewindable",
				"fork": "asOf",
			},
			"content": { "kind": "base", "version": 1, "value": { "count": 0 } },
		})],
	)
	.await;
	for count in 1..=100u64 {
		commit_ok(
			&storage,
			vec![json!({
				"type": "document.change",
				"id": document_id,
				"content": { "kind": "delta", "version": 1, "ops": [["s", ["count"], count]] },
			})],
		)
		.await;
	}
	storage.close(ctx()).await;
	assert!(std::fs::metadata(&path).unwrap().len() < 1024 * 1024);
}

// ---------------------------------------------------------------------------
// sqlite-facade.test.ts (portable subset; deferred cases in the module ledger)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn commits_work_done_through_the_transaction_handle_and_closes_idempotently() {
	let database = SqliteDatabase::open(":memory:", &options()).unwrap();
	database
		.transaction(|connection| {
			connection.execute_batch("CREATE TABLE async_probe (value INTEGER)")?;
			connection.execute("INSERT INTO async_probe (value) VALUES (?1)", [1])?;
			Ok(())
		})
		.await
		.unwrap();
	let value = database
		.transaction(|connection| {
			connection
				.query_row("SELECT value FROM async_probe", [], |row| row.get::<_, i64>(0))
				.map_err(SqliteStorageError::from)
		})
		.await
		.unwrap();
	assert_eq!(value, 1);
	database.close().await.unwrap();
	database.close().await.unwrap();
}

#[tokio::test]
async fn serializes_concurrent_transactions() {
	let database = Arc::new(SqliteDatabase::open(":memory:", &options()).unwrap());
	database.exec("CREATE TABLE transaction_queue (value INTEGER)").await.unwrap();
	let (started_sender, started_receiver) = tokio::sync::oneshot::channel::<()>();
	let (release_sender, release_receiver) = std::sync::mpsc::channel::<()>();
	let first_db = database.clone();
	let first_started = Arc::new(AtomicBool::new(false));
	let second_started = Arc::new(AtomicBool::new(false));

	let first_handle = tokio::spawn({
		let first_db = first_db.clone();
		let first_started = first_started.clone();
		async move {
			first_db
				.transaction(move |connection| {
					connection.execute("INSERT INTO transaction_queue (value) VALUES (1)", [])?;
					first_started.store(true, Ordering::SeqCst);
					let _ = started_sender.send(());
					release_receiver.recv().unwrap();
					Ok(())
				})
				.await
		}
	});
	tokio::time::timeout(std::time::Duration::from_secs(5), started_receiver)
		.await
		.expect("first transaction started")
		.expect("sender dropped");

	let second_db = database.clone();
	let second_flag = second_started.clone();
	let second = tokio::task::spawn(async move {
		second_db
			.transaction(move |connection| {
				second_flag.store(true, Ordering::SeqCst);
				connection.execute("INSERT INTO transaction_queue (value) VALUES (2)", [])?;
				Ok(())
			})
			.await
	});
	// The second transaction cannot start while the first holds the connection.
	tokio::time::sleep(std::time::Duration::from_millis(50)).await;
	assert!(!second_started.load(Ordering::SeqCst));

	release_sender.send(()).unwrap();
	first_handle.await.unwrap().unwrap();
	second.await.unwrap().unwrap();
	let values = database
		.transaction(|connection| {
			let mut statement = connection.prepare("SELECT value FROM transaction_queue ORDER BY value")?;
			let rows = statement
				.query_map([], |row| row.get::<_, i64>(0))?
				.collect::<Result<Vec<_>, rusqlite::Error>>()?;
			Ok::<_, SqliteStorageError>(rows)
		})
		.await
		.unwrap();
	assert_eq!(values, vec![1, 2]);
	database.close().await.unwrap();
}

#[tokio::test]
async fn queues_ordinary_operations_behind_an_active_transaction() {
	let database = Arc::new(SqliteDatabase::open(":memory:", &options()).unwrap());
	database.exec("CREATE TABLE operation_queue (value INTEGER)").await.unwrap();
	let (started_sender, started_receiver) = tokio::sync::oneshot::channel::<()>();
	let (release_sender, release_receiver) = std::sync::mpsc::channel::<()>();
	let write_started = Arc::new(AtomicBool::new(false));
	let read_started = Arc::new(AtomicBool::new(false));

	let pending_db = database.clone();
	let pending = tokio::spawn(async move {
		pending_db
			.transaction(move |connection| {
				connection.execute("INSERT INTO operation_queue (value) VALUES (1)", [])?;
				started_sender.send(()).unwrap();
				release_receiver.recv().unwrap();
				Ok(())
			})
			.await
	});
	tokio::time::timeout(std::time::Duration::from_secs(5), started_receiver)
		.await
		.expect("transaction started")
		.expect("sender dropped");

	let write_db = database.clone();
	let write_flag = write_started.clone();
	let write = tokio::task::spawn(async move {
		let result = write_db.exec("INSERT INTO operation_queue (value) VALUES (2)").await;
		write_flag.store(true, Ordering::SeqCst);
		result
	});
	let read_db = database.clone();
	let read_flag = read_started.clone();
	let read = tokio::task::spawn(async move {
		let result = read_db
			.transaction(|connection| {
				let mut statement = connection.prepare("SELECT value FROM operation_queue ORDER BY value")?;
				let rows = statement
					.query_map([], |row| row.get::<_, i64>(0))?
					.collect::<Result<Vec<_>, rusqlite::Error>>()?;
				Ok::<_, SqliteStorageError>(rows)
			})
			.await;
		read_flag.store(true, Ordering::SeqCst);
		result
	});
	tokio::time::sleep(std::time::Duration::from_millis(50)).await;
	assert!(!write_started.load(Ordering::SeqCst));
	assert!(!read_started.load(Ordering::SeqCst));

	release_sender.send(()).unwrap();
	pending.await.unwrap().unwrap();
	write.await.unwrap().unwrap();
	let values = read.await.unwrap().unwrap();
	assert_eq!(values, vec![1, 2]);
	database.close().await.unwrap();
}

#[tokio::test]
async fn lets_admitted_multi_query_reads_finish_before_storage_closes() {
	let storage: Arc<dyn Storage> = Arc::new(open_memory().await);
	let entry_id = EntryId(2);
	commit_ok(
		&storage,
		vec![
			json!({ "type": "conversation", "value": { "id": ROOT_CONVERSATION_ID.0 } }),
			json!({ "type": "entry", "value": { "id": entry_id.0, "conversationId": ROOT_CONVERSATION_ID.0, "kind": "probe" } }),
		],
	)
	.await;
	let scan_storage = storage.clone();
	let scan = tokio::spawn(async move {
		scan_storage
			.scan_entries(&record::<EntryQuery>(json!({ "conversationId": ROOT_CONVERSATION_ID.0 })), 10, None, ctx())
			.await
	});
	let entry_storage = storage.clone();
	let entry = tokio::spawn(async move { entry_storage.entry_in(ROOT_CONVERSATION_ID, entry_id, ctx()).await });
	let head_storage = storage.clone();
	let head = tokio::spawn(async move { head_storage.find_latest_head_marker(ROOT_CONVERSATION_ID, None, ctx()).await });
	// Let every admitted read pass its open check and queue its database work.
	for _ in 0..10 {
		tokio::task::yield_now().await;
	}
	// Close runs behind the admitted reads on the connection mutex.
	let close_storage = storage.clone();
	let closed = tokio::spawn(async move { close_storage.close(ctx()).await });
	// A repeated close is idempotent.
	storage.close(ctx()).await;

	let scan = scan.await.unwrap();
	assert_eq!(scan.items.iter().map(|item| item.id).collect::<Vec<_>>(), vec![entry_id]);
	assert_eq!(entry.await.unwrap().unwrap().0.kind, "probe");
	// Adapted from upstream `undefined` (D3): a missing head marker reads as `None`.
	assert_eq!(head.await.unwrap(), None);
	closed.await.unwrap();
	// Adapted from upstream rejection (D3): reads after close collapse.
	let after = storage
		.scan_entries(&record::<EntryQuery>(json!({ "conversationId": ROOT_CONVERSATION_ID.0 })), 10, None, ctx())
		.await;
	assert_eq!(after.items, Vec::new());
}

#[tokio::test]
async fn does_not_preserve_a_guaranteed_rejection_when_rollback_itself_fails() {
	let database = Arc::new(SqliteDatabase::open(":memory:", &options()).unwrap());
	database.exec("CREATE TABLE rollback_probe (value INTEGER)").await.unwrap();
	let error = database
		.transaction(|connection| -> Result<(), SqliteStorageError> {
			connection.execute("INSERT INTO rollback_probe (value) VALUES (1)", [])?;
			// Escape the managed transaction, so the trailing rollback must fail.
			connection.execute_batch("COMMIT")?;
			Err(SqliteStorageError("rejected after an escaped commit".into()))
		})
		.await
		.expect_err("settlement failure");
	assert!(error.to_string().contains("rollback"), "{error}");
	let value = database
		.transaction(|connection| {
			connection
				.query_row("SELECT value FROM rollback_probe", [], |row| row.get::<_, i64>(0))
				.map_err(SqliteStorageError::from)
		})
		.await
		.unwrap();
	assert_eq!(value, 1);
	database.close().await.unwrap();
}

#[tokio::test]
async fn reads_a_document_from_one_committed_state_while_a_commit_replaces_its_base() {
	let id = DocumentId(5);
	// Each yield count starts the commit at a different interleaving point.
	for yields in 0..16u32 {
		let storage: Arc<dyn Storage> = Arc::new(open_memory().await);
		commit_ok(
			&storage,
			vec![json!({
				"type": "document.create",
				"record": { "id": id.0, "kind": "replaced", "scope": { "kind": "session" } },
				"content": { "kind": "base", "version": 1, "value": { "value": 1 } },
			})],
		)
		.await;
		let read_storage = storage.clone();
		let read = tokio::spawn(async move { read_storage.document(id, DocumentPoint::Current, ctx()).await });
		for _ in 0..yields {
			tokio::task::yield_now().await;
		}
		let replace_storage = storage.clone();
		let replace = tokio::spawn(async move {
			commit_ok(
				&replace_storage,
				vec![json!({
					"type": "document.change",
					"id": id.0,
					"content": { "kind": "base", "version": 1, "value": { "value": 2 } },
				})],
			)
			.await
		});
		let (stored, _) = tokio::join!(read, replace);
		let value = json!(stored.unwrap().unwrap().value);
		assert!(value == json!({ "value": 1 }) || value == json!({ "value": 2 }), "{value}");
		storage.close(ctx()).await;
	}
}

