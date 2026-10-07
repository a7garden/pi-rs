//! Port of `packages/durable/src/storage/sqlite/storage.ts`.
//!
//! Blocking wrapper choice (module D1): every operation runs through
//! `tokio::task::spawn_blocking` around the mutex-guarded rusqlite connection;
//! multi-query reads execute inside one closure (or transaction) so they
//! observe a single committed state — the upstream `admitRead` drain counter is
//! unnecessary because `close()` takes the same mutex.

use std::collections::HashMap;
use std::sync::Arc;

use pi_chord::context::Context;
use pi_chord::delta::apply;
use rusqlite::{params_from_iter, Connection};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};

use crate::errors::StorageRejected;
use crate::storage::sqlite::database::{SqliteDatabase, SqliteStorageError};
use crate::storage::sqlite::migrations::apply_sqlite_migrations;
use crate::storage::memory::{content_version, cursor_id, is_alive_at, is_current_only, scope_kind};
use crate::types::{
	ConversationId, ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentContent, DocumentCreate,
	DocumentId, DocumentPoint, DocumentQuery, DocumentRecord, DocumentScope, EntryId, EntryQuery, EntryRecord, Page,
	Seq, Storage, StorageWrite, StoredDocument, SubmissionId, SubmissionQuery, SubmissionRecord, TaskId, TaskQuery,
	TaskRecord, TaskStateStatus, SubmissionStatus,
};

fn parse_json<T: DeserializeOwned>(record: &str) -> Option<T> {
	serde_json::from_str(record).ok()
}

fn encode_json<T: serde::Serialize>(value: &T) -> String {
	serde_json::to_string(value).expect("storage records serialize")
}

/// Some SQLite bindings replace lone UTF-16 surrogates. JSON encoding keeps
/// indexed identities lossless.
fn encode_indexed_string(value: &str) -> String {
	serde_json::to_string(value).expect("strings serialize")
}

fn cursor_value(after: u64) -> Cursor {
	let mut cursor = serde_json::Map::new();
	cursor.insert("after".into(), json!(after));
	cursor
}

fn page<T>(mut values: Vec<T>, id_of: impl Fn(&T) -> u64, limit: u64) -> Page<T> {
	if values.len() as u64 <= limit {
		return Page { items: values, next: None };
	}
	values.truncate(limit as usize);
	let after = id_of(values.last().unwrap());
	Page { items: values, next: Some(cursor_value(after)) }
}

struct ScopeColumns {
	scope_kind: &'static str,
	owner_id: u64,
}

fn scope_columns(scope: &DocumentScope) -> ScopeColumns {
	match scope {
		DocumentScope::Session => ScopeColumns { scope_kind: "session", owner_id: 0 },
		DocumentScope::Conversation { conversation_id } => {
			ScopeColumns { scope_kind: "conversation", owner_id: conversation_id.0 }
		}
		DocumentScope::Task { task_id } => ScopeColumns { scope_kind: "task", owner_id: task_id.0 },
	}
}

struct AddressParts {
	kind: String,
	scope_kind: &'static str,
	owner_id: u64,
	family: u64,
	key_value: String,
}

fn address_parts(address_kind: &str, scope: &DocumentScope, key: Option<&str>) -> AddressParts {
	let scope = scope_columns(scope);
	AddressParts {
		kind: encode_indexed_string(address_kind),
		scope_kind: scope.scope_kind,
		owner_id: scope.owner_id,
		family: if key.is_none() { 0 } else { 1 },
		key_value: encode_indexed_string(key.unwrap_or("")),
	}
}

fn address_key(kind: &str, scope: &DocumentScope, key: Option<&str>) -> String {
	let parts = address_parts(kind, scope, key);
	format!("{}|{}|{}|{}|{}", parts.kind, parts.scope_kind, parts.owner_id, parts.family, parts.key_value)
}

fn record_address_key(record: &DocumentRecord) -> String {
	address_key(&record.kind, &record.scope, record.key.as_deref())
}

fn action_address_key(record: &DocumentCreate) -> String {
	address_key(&record.kind, &record.scope, record.key.as_deref())
}

/// Fully validated mutation of one document in a commit.
#[derive(Debug, Default)]
struct DocumentAction {
	create: Option<DocumentCreate>,
	copy: Option<crate::types::DocumentCopySource>,
	content: Option<DocumentContent>,
	retire: bool,
}

/// Errors inside a commit transaction: storage rejections carry the upstream
/// message; SQL failures are stringified and re-wrapped as rejections (D3).
enum CommitError {
	Rejected(StorageRejected),
	Sql(String),
}

impl From<SqliteStorageError> for CommitError {
	fn from(error: SqliteStorageError) -> Self {
		CommitError::Sql(error.0)
	}
}

impl CommitError {
	fn into_sql(self) -> SqliteStorageError {
		match self {
			CommitError::Rejected(rejection) => SqliteStorageError(rejection.message),
			CommitError::Sql(message) => SqliteStorageError(message),
		}
	}
}

impl From<rusqlite::Error> for CommitError {
	fn from(error: rusqlite::Error) -> Self {
		CommitError::Sql(error.to_string())
	}
}

fn rejected(message: impl Into<String>) -> CommitError {
	CommitError::Rejected(StorageRejected::new(message))
}

fn optional_row<T>(
	connection: &Connection,
	sql: &str,
	params: impl rusqlite::Params,
	read: impl Fn(&rusqlite::Row) -> rusqlite::Result<T>,
) -> Result<Option<T>, SqliteStorageError> {
	connection
		.query_row(sql, params, read)
		.map(Some)
		.or_else(|error| match error {
			rusqlite::Error::QueryReturnedNoRows => Ok(None),
			error => Err(SqliteStorageError::from(error)),
		})
}

/// Portable SQLite implementation of the storage contract.
pub struct SqliteStorage {
	db: Arc<SqliteDatabase>,
	next_id: parking_lot::Mutex<u64>,
	closed: parking_lot::Mutex<bool>,
}

impl SqliteStorage {
	/// Initialize storage over an owned SQLite database facade.
	pub async fn open(db: SqliteDatabase) -> Result<Self, SqliteStorageError> {
		let db = Arc::new(db);
		let init_db = db.clone();
		let result: Result<Self, SqliteStorageError> = async {
			apply_sqlite_migrations(&db, super::migrations::sqlite_migrations()).await?;
			let metadata: Option<(String, u64)> = init_db
				.transaction(|connection| {
					optional_row(
						connection,
						"SELECT next_id, next_seq FROM durable_metadata WHERE singleton = 1",
						[],
						|row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64)),
					)
					.map_err(SqliteStorageError::from)
				})
				.await?;
			let Some((next_id, _)) = metadata else {
				return Err(SqliteStorageError("Durable SQLite metadata is missing".into()));
			};
			let next_id: u64 = next_id.parse().map_err(|_| SqliteStorageError("Durable SQLite metadata is missing".into()))?;
			Ok(Self { db: init_db, next_id: parking_lot::Mutex::new(next_id), closed: parking_lot::Mutex::new(false) })
		}
		.await;
		match result {
			Ok(storage) => Ok(storage),
			Err(error) => {
				// Preserve the initialization failure (upstream closes the db).
				let _ = db.close().await;
				Err(error)
			}
		}
	}

	/// Open and configure a file-backed storage database, then initialize
	/// storage over it (upstream `openNodeSqliteStorage`).
	pub async fn open_path(
		path: impl AsRef<std::path::Path>,
		options: &super::database::NodeSqliteStorageOptions,
	) -> Result<Self, SqliteStorageError> {
		let db = SqliteDatabase::open(path, options)?;
		Self::open(db).await
	}

	fn assert_open_commit(&self) -> Result<(), StorageRejected> {
		if *self.closed.lock() {
			return Err(StorageRejected::new("SqliteStorage is closed"));
		}
		Ok(())
	}

	fn prepare_document_actions(&self, writes: &[StorageWrite]) -> Result<Vec<(DocumentId, DocumentAction)>, StorageRejected> {
		let mut order: Vec<DocumentId> = Vec::new();
		let mut actions: HashMap<DocumentId, DocumentAction> = HashMap::new();
		macro_rules! action_for {
			($id:expr) => {{
				let id = $id;
				if !actions.contains_key(&id) {
					order.push(id);
					actions.insert(id, DocumentAction::default());
				}
				actions.get_mut(&id).expect("just inserted")
			}};
		}
		for write in writes {
			match write {
				StorageWrite::DocumentCreate { record, content } => {
					let action = action_for!(record.id);
					if action.create.is_some() || action.content.is_some() || action.copy.is_some() {
						return Err(StorageRejected::new(format!(
							"Document {} has more than one content command",
							record.id.0
						)));
					}
					action.create = Some(record.clone());
					action.content = Some(DocumentContent::Base { version: content.version, value: content.value.clone() });
				}
				StorageWrite::DocumentCopy { record, source } => {
					let action = action_for!(record.id);
					if action.create.is_some() || action.content.is_some() || action.copy.is_some() {
						return Err(StorageRejected::new(format!(
							"Document {} has more than one content command",
							record.id.0
						)));
					}
					action.create = Some(record.clone());
					action.copy = Some(*source);
				}
				StorageWrite::DocumentChange { id, content } => {
					let action = action_for!(*id);
					if action.content.is_some() || action.copy.is_some() {
						return Err(StorageRejected::new(format!(
							"Document {} has more than one content command",
							id.0
						)));
					}
					action.content = Some(content.clone());
				}
				StorageWrite::DocumentRetire { id } => {
					let action = action_for!(*id);
					if action.retire {
						return Err(StorageRejected::new(format!("Document {} is retired more than once", id.0)));
					}
					action.retire = true;
				}
				_ => {}
			}
		}
		Ok(order.into_iter().map(|id| (id, actions.remove(&id).unwrap())).collect())
	}

	fn candidate_next_id(&self, writes: &[StorageWrite]) -> u64 {
		let mut next_id = *self.next_id.lock();
		for write in writes {
			let id = match write {
				StorageWrite::Conversation { value } => value.id.0,
				StorageWrite::Entry { value } => value.id.0,
				StorageWrite::Task { value } => value.id.0,
				StorageWrite::Submission { value } => value.id.0,
				StorageWrite::DocumentCreate { record, .. } | StorageWrite::DocumentCopy { record, .. } => record.id.0,
				StorageWrite::DocumentChange { .. } | StorageWrite::DocumentRetire { .. } => continue,
			};
			next_id = next_id.max(id + 1);
		}
		next_id
	}

	fn write_id(write: &StorageWrite) -> Option<u64> {
		match write {
			StorageWrite::Conversation { value } => Some(value.id.0),
			StorageWrite::Entry { value } => Some(value.id.0),
			StorageWrite::Task { value } => Some(value.id.0),
			StorageWrite::Submission { value } => Some(value.id.0),
			StorageWrite::DocumentCreate { record, .. } | StorageWrite::DocumentCopy { record, .. } => Some(record.id.0),
			StorageWrite::DocumentChange { .. } | StorageWrite::DocumentRetire { .. } => None,
		}
	}

	/// Materialize one incarnation at the selected point.
	///
	/// `Err` carries the upstream corruption/history error strings; read-side
	/// callers collapse them to `None` (trait D3), the copy path wraps them.
	fn materialize_document(
		connection: &Connection,
		id: DocumentId,
		at: DocumentPoint,
	) -> Result<Option<StoredDocument>, String> {
		let record: Option<String> = optional_row(
			connection,
			"SELECT record FROM documents WHERE id = ?1",
			[id.0],
			|row| row.get(0),
		)
		.map_err(|error| error.to_string())?;
		let Some(record) = record else { return Ok(None) };
		let record: DocumentRecord = parse_json(&record).ok_or_else(|| format!("Document {} has an invalid record", id.0))?;
		if at != DocumentPoint::Current && is_current_only(&record) {
			return Err(format!("Document {} does not retain historical content", id.0));
		}
		if !is_alive_at(&record, at) {
			return Ok(None);
		}
		let upper: u64 = match at {
			DocumentPoint::Current => u64::MAX,
			DocumentPoint::At(seq) => seq.0,
		};
		let upper_bound: i64 = i64::try_from(upper).unwrap_or(i64::MAX);
		let base: Option<(u64, u64, String)> = optional_row(
			connection,
			"SELECT seq, version, content FROM document_revisions\n\t\t\t\tWHERE document_id = ?1 AND kind = 'base' AND seq <= ?2 ORDER BY seq DESC LIMIT 1",
			rusqlite::params![id.0, upper_bound],
			|row| Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64, row.get(2)?)),
		)
		.map_err(|error| error.to_string())?;
		let Some((base_seq, base_version, base_content)) = base else {
			return Err(format!("Document {} is missing a required base", id.0));
		};
		let mut value: Value = parse_json(&base_content).ok_or_else(|| format!("Document {} is missing a required base", id.0))?;
		let tail: Vec<(String, u64, String)> = {
			let mut statement = connection
				.prepare("SELECT seq, kind, version, content FROM document_revisions\n\t\t\t\tWHERE document_id = ?1 AND seq > ?2 AND seq <= ?3 ORDER BY seq")
				.map_err(|error| error.to_string())?;
			let rows = statement
				.query_map(rusqlite::params![id.0, base_seq as i64, upper_bound], |row| {
					Ok((row.get::<_, String>(1)?, row.get::<_, i64>(2)? as u64, row.get(3)?))
				})
				.map_err(|error| error.to_string())?;
			rows.collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())?
		};
		let mut deltas_since_base = 0u64;
		for (kind, version, content) in tail {
			if kind != "delta" || version != base_version {
				return Err(format!("Document {} crosses a stored version boundary without a base", id.0));
			}
			let ops: Vec<pi_chord::delta::Op> =
				parse_json(&content).ok_or_else(|| format!("Document {} has an invalid operation batch", id.0))?;
			apply(&mut value, &ops).map_err(|error| error.to_string())?;
			deltas_since_base += 1;
		}
		Ok(Some(StoredDocument {
			record,
			version: base_version,
			value: value.as_object().expect("document base is an object").clone(),
			deltas_since_base,
		}))
	}

	fn check_global_ids(connection: &Connection, writes: &[StorageWrite]) -> Result<(), CommitError> {
		let mut claimed: HashMap<u64, &'static str> = HashMap::new();
		for write in writes {
			let Some(id) = Self::write_id(write) else { continue };
			let table: &'static str = match write {
				StorageWrite::Conversation { .. } => "conversation",
				StorageWrite::Entry { .. } => "entry",
				StorageWrite::Task { .. } => "task",
				StorageWrite::Submission { .. } => "submission",
				_ => "document",
			};
			let existing: Option<String> = optional_row(
				connection,
				"SELECT record_type FROM record_ids WHERE id = ?1",
				[id],
				|row| row.get(0),
			)?;
			let earlier = claimed.get(&id).copied();
			if matches!(table, "conversation" | "entry" | "document") {
				if let Some(existing) = existing.as_deref() {
					return Err(rejected(format!("ID {id} already belongs to {existing}")));
				}
				if earlier.is_some() {
					return Err(rejected(format!("ID {id} is written more than once")));
				}
			} else {
				if let Some(existing) = existing.as_deref() {
					if existing != table {
						return Err(rejected(format!("ID {id} already belongs to {existing}")));
					}
				}
				if let Some(earlier) = earlier {
					if earlier != table {
						return Err(rejected(format!("ID {id} is written as two record types")));
					}
				}
			}
			claimed.insert(id, table);
		}
		Ok(())
	}

	fn current_document_id(connection: &Connection, parts: &AddressParts) -> Result<Option<DocumentId>, CommitError> {
		let id: Option<u64> = optional_row(
			connection,
			"SELECT id FROM documents\n\t\t\t\tWHERE kind = ?1 AND scope_kind = ?2 AND owner_id = ?3 AND family = ?4 AND key_value = ?5 AND retired_at IS NULL\n\t\t\t\tLIMIT 1",
			rusqlite::params![parts.kind, parts.scope_kind, parts.owner_id as i64, parts.family as i64, parts.key_value],
			|row| row.get(0),
		)?;
		Ok(id.map(DocumentId))
	}

	fn check_document_actions(
		connection: &Connection,
		actions: &[(DocumentId, DocumentAction)],
	) -> Result<(), CommitError> {
		let mut live_counts: HashMap<String, i64> = HashMap::new();
		for (id, action) in actions {
			if let Some(copy) = &action.copy {
				if actions.iter().any(|(other_id, _)| other_id == &copy.id) {
					return Err(rejected(format!("Document copy {} source is changed in the copy batch", id.0)));
				}
			}
			let existing: Option<DocumentRecord> = optional_row(
				connection,
				"SELECT record FROM documents WHERE id = ?1",
				[id.0],
				|row| row.get::<_, String>(0),
			)?
			.and_then(|record| parse_json(&record));
			if action.create.is_none() && existing.is_none() {
				return Err(rejected(format!("Unknown document: {}", id.0)));
			}
			if action.create.is_some() && existing.is_some() {
				return Err(rejected(format!("Document {} already exists", id.0)));
			}
			if existing.as_ref().map_or(false, |record| record.retired_at.is_some()) {
				return Err(rejected(format!("Document {} is retired", id.0)));
			}
			if let Some(DocumentContent::Delta { version, .. }) = &action.content {
				let previous: Option<u64> = optional_row(
					connection,
					"SELECT version FROM document_revisions WHERE document_id = ?1 ORDER BY seq DESC LIMIT 1",
					[id.0],
					|row| row.get::<_, i64>(0).map(|value| value as u64),
				)?;
				let Some(previous) = previous else {
					return Err(rejected(format!("Document {} delta has no base", id.0)));
				};
				if previous != *version {
					return Err(rejected(format!("Document {} version transition requires a base", id.0)));
				}
			}
			let (kind, scope, key) = match &action.create {
				Some(create) => (&create.kind, &create.scope, &create.key),
				None => {
					let existing = existing.as_ref().expect("existing or create");
					(&existing.kind, &existing.scope, &existing.key)
				}
			};
			let address = address_key(kind, scope, key.as_deref());
			let live = match live_counts.entry(address) {
				std::collections::hash_map::Entry::Occupied(occupied) => occupied.into_mut(),
				std::collections::hash_map::Entry::Vacant(vacant) => {
					let parts = address_parts(kind, scope, key.as_deref());
					vacant.insert(if Self::current_document_id(connection, &parts)?.is_some() { 1 } else { 0 })
				}
			};
			if action.retire && existing.is_some() {
				*live -= 1;
			}
			if action.create.is_some() && !action.retire {
				*live += 1;
			}
		}
		for live in live_counts.values() {
			if *live > 1 {
				return Err(rejected("Document address already has a current incarnation"));
			}
		}
		Ok(())
	}

	fn claim_id(connection: &Connection, id: u64, table: &str) -> Result<(), CommitError> {
		connection.execute(
			"INSERT OR IGNORE INTO record_ids (id, record_type) VALUES (?1, ?2)",
			rusqlite::params![id, table],
		)?;
		Ok(())
	}

	fn apply_table_write(connection: &Connection, write: &StorageWrite, seq: Seq) -> Result<(), CommitError> {
		match write {
			StorageWrite::Conversation { value } => {
				Self::claim_id(connection, value.id.0, "conversation")?;
				connection.execute(
					"INSERT INTO conversations (id, owner_conversation_id, owner_task_id, record) VALUES (?1, ?2, ?3, ?4)",
					rusqlite::params![
						value.id.0,
						value.owner.as_ref().map(|owner| owner.conversation_id.0),
						value.owner.as_ref().map(|owner| owner.task_id.0),
						encode_json(value),
					],
				)?;
			}
			StorageWrite::Entry { value } => {
				Self::claim_id(connection, value.id.0, "entry")?;
				connection.execute(
					"INSERT INTO entries (id, conversation_id, head, commit_seq, record) VALUES (?1, ?2, ?3, ?4, ?5)",
					rusqlite::params![
						value.id.0,
						value.conversation_id.0,
						value.head.map(|head| head.0),
						seq.0,
						encode_json(value),
					],
				)?;
			}
			StorageWrite::Task { value } => {
				Self::claim_id(connection, value.id.0, "task")?;
				connection.execute(
					"INSERT INTO tasks (id, conversation_id, kind, status, abort_requested, background, record)\n\t\t\t\t\t\tVALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)\n\t\t\t\t\t\tON CONFLICT(id) DO UPDATE SET conversation_id = excluded.conversation_id, kind = excluded.kind,\n\t\t\t\t\t\tstatus = excluded.status, abort_requested = excluded.abort_requested,\n\t\t\t\t\t\tbackground = excluded.background, record = excluded.record",
					rusqlite::params![
						value.id.0,
						value.conversation_id.0,
						encode_indexed_string(&value.kind),
						task_status_str(value.state.status),
						if value.abort_requested { 1 } else { 0 },
						if value.background { 1 } else { 0 },
						encode_json(value),
					],
				)?;
			}
			StorageWrite::Submission { value } => {
				Self::claim_id(connection, value.id.0, "submission")?;
				connection.execute(
					"INSERT INTO submissions (id, conversation_id, request_id, status, record) VALUES (?1, ?2, ?3, ?4, ?5)\n\t\t\t\t\t\tON CONFLICT(id) DO UPDATE SET conversation_id = excluded.conversation_id,\n\t\t\t\t\t\trequest_id = excluded.request_id, status = excluded.status, record = excluded.record",
					rusqlite::params![
						value.id.0,
						value.conversation_id.0,
						value.request_id.as_deref().map(encode_indexed_string),
						submission_status_str(value.status),
						encode_json(value),
					],
				)?;
			}
			_ => {}
		}
		Ok(())
	}

	fn apply_document_actions(
		connection: &Connection,
		actions: &[(DocumentId, DocumentAction)],
		seq: Seq,
	) -> Result<(), CommitError> {
		for (id, action) in actions {
			let mut content = action.content.clone();
			if let Some(copy) = &action.copy {
				let stored = Self::materialize_document(connection, copy.id, copy.at)
					.map_err(|error| {
						let _ = error;
						rejected(format!("Document copy {} was rejected", id.0))
					})?;
				let Some(stored) = stored else {
					return Err(rejected(format!("Document copy {} was rejected", id.0)));
				};
				let create = action.create.as_ref().expect("copy carries a record");
				if scope_kind(&stored.record.scope) != "conversation"
					|| scope_kind(&create.scope) != "conversation"
					|| stored.record.kind != create.kind
					|| stored.record.key != create.key
					|| stored.record.history != create.history
					|| stored.record.fork != create.fork
				{
					return Err(rejected(format!("Document copy {} was rejected", id.0)));
				}
				content = Some(DocumentContent::Base { version: stored.version, value: stored.value });
			}
			let mut record: DocumentRecord;
			if let Some(create) = &action.create {
				record = create.clone().into_record(seq, action.retire.then_some(seq));
				let parts = address_parts(&record.kind, &record.scope, record.key.as_deref());
				Self::claim_id(connection, id.0, "document")?;
				connection.execute(
					"INSERT INTO documents\n\t\t\t\t\t\t(id, kind, family, key_value, scope_kind, owner_id, created_at, retired_at, record)\n\t\t\t\t\t\tVALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
					rusqlite::params![
						id.0,
						parts.kind,
						parts.family as i64,
						parts.key_value,
						parts.scope_kind,
						parts.owner_id as i64,
						seq.0,
						action.retire.then_some(seq.0),
						encode_json(&record),
					],
				)?;
			} else {
				let row: String = connection
					.query_row("SELECT record FROM documents WHERE id = ?1", [id.0], |row| row.get(0))?;
				record = parse_json(&row).expect("stored record parses");
			}

			if let Some(content) = &content {
				if matches!(content, DocumentContent::Base { .. }) && is_current_only(&record) {
					connection.execute("DELETE FROM document_revisions WHERE document_id = ?1", [id.0])?;
				}
				let encoded_content = match content {
					DocumentContent::Base { value, .. } => encode_json(value),
					DocumentContent::Delta { ops, .. } => encode_json(ops),
				};
				connection.execute(
					"INSERT INTO document_revisions (document_id, seq, kind, version, content) VALUES (?1, ?2, ?3, ?4, ?5)",
					rusqlite::params![
						id.0,
						seq.0,
						match content {
							DocumentContent::Base { .. } => "base",
							DocumentContent::Delta { .. } => "delta",
						},
						content_version(content),
						encoded_content,
					],
				)?;
			}

			if action.retire {
				if action.create.is_none() {
					record.retired_at = Some(seq);
					connection.execute(
						"UPDATE documents SET retired_at = ?1, record = ?2 WHERE id = ?3",
						rusqlite::params![seq.0, encode_json(&record), id.0],
					)?;
				}
				if is_current_only(&record) {
					connection.execute("DELETE FROM document_revisions WHERE document_id = ?1", [id.0])?;
				}
			}
		}
		Ok(())
	}

	fn read_conversation(connection: &Connection, id: ConversationId) -> Result<Option<ConversationRecord>, SqliteStorageError> {
		let record: Option<String> =
			optional_row(connection, "SELECT record FROM conversations WHERE id = ?1", [id.0], |row| row.get(0))?;
		Ok(record.and_then(|record| parse_json(&record)))
	}

	async fn read_entries_page(&self, query: &EntryQuery, limit: u64, cursor: Option<&Cursor>) -> Page<EntryRecord> {
		let db = self.db.clone();
		let query = *query;
		let after = cursor_id(cursor);
		let min_entry_id = query.min_entry_id.map(|id| id.0);
		db.blocking(move |connection| {
			let mut conversation = Self::read_conversation_blocking(connection, query.conversation_id)?.ok_or_else(|| {
				SqliteStorageError(format!("Unknown conversation: {}", query.conversation_id.0))
			})?;
			let mut upper: Option<u64> = query.max_entry_id.map(|id| id.0);
			if let Some(after) = after {
				upper = Some(upper.unwrap_or(u64::MAX).min(after - 1));
			}
			let min_entry_id = query.min_entry_id.map(|id| id.0);
			let mut values: Vec<EntryRecord> = Vec::new();
			loop {
				let mut sql = String::from("SELECT record FROM entries WHERE conversation_id = ?1");
				let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(conversation.id.0)];
				if let Some(min_entry_id) = min_entry_id {
					sql.push_str(" AND id >= ?");
					params.push(Box::new(min_entry_id as i64));
				}
				if let Some(upper) = upper {
					sql.push_str(" AND id <= ?");
					params.push(Box::new(upper as i64));
				}
				sql.push_str(" ORDER BY id DESC LIMIT ?");
				params.push(Box::new(limit as i64 + 1 - values.len() as i64));
				let statement_sql = sql.clone();
				let rows: Vec<String> = {
					let mut statement = connection.prepare(&statement_sql)?;
					let mapped = statement.query_map(
						params_from_iter(params.iter().map(|param| param.as_ref())),
						|row| row.get::<_, String>(0),
					)?;
					mapped.collect::<Result<Vec<_>, _>>()?
				};
				for row in rows {
					values.push(parse_json(&row).ok_or_else(|| SqliteStorageError("invalid stored entry".into()))?);
				}
				let Some(parent) = &conversation.parent else { break };
				if values.len() as u64 > limit {
					break;
				}
				upper = Some(upper.unwrap_or(u64::MAX).min(parent.at.0));
				if min_entry_id.is_some_and(|min| upper.unwrap() < min) {
					break;
				}
				conversation = Self::read_conversation_blocking(connection, parent.conversation_id)?.ok_or_else(|| {
					SqliteStorageError(format!("Unknown conversation: {}", parent.conversation_id.0))
				})?;
			}
			Ok(page(values, |record| record.id.0, limit))
		})
		.await
		.unwrap_or_else(|_| Page::empty())
	}
}

impl SqliteStorage {
	fn read_conversation_blocking(connection: &Connection, id: ConversationId) -> Result<Option<ConversationRecord>, SqliteStorageError> {
		let record: Option<String> = optional_row(
			connection,
			"SELECT record FROM conversations WHERE id = ?1",
			[id.0],
			|row| row.get(0),
		)?;
		Ok(record.and_then(|record| parse_json(&record)))
	}
}

fn task_status_str(status: TaskStateStatus) -> &'static str {
	match status {
		TaskStateStatus::Pending => "pending",
		TaskStateStatus::Running => "running",
		TaskStateStatus::Waiting => "waiting",
		TaskStateStatus::Completing => "completing",
		TaskStateStatus::Terminal => "terminal",
	}
}

fn submission_status_str(status: SubmissionStatus) -> &'static str {
	match status {
		SubmissionStatus::Queued => "queued",
		SubmissionStatus::Placed => "placed",
		SubmissionStatus::Done => "done",
		SubmissionStatus::Unanswered => "unanswered",
	}
}

#[async_trait::async_trait]
impl Storage for SqliteStorage {
	async fn commit(&self, writes: &[StorageWrite], _context: &Context) -> Result<Seq, StorageRejected> {
		self.assert_open_commit()?;
		let document_actions = self.prepare_document_actions(writes)?;
		let candidate_next_id = self.candidate_next_id(writes);
		let writes = writes.to_vec();
		let db = self.db.clone();
		let seq: Seq = db
			.transaction(move |connection| {
				let (metadata_next_id, next_seq): (String, u64) = optional_row(
					connection,
					"SELECT next_id, next_seq FROM durable_metadata WHERE singleton = 1",
					[],
					|row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64)),
				)?
				.ok_or_else(|| SqliteStorageError("Durable SQLite metadata is missing".into()))?;
				let committed_seq = Seq(next_seq);
				Self::check_global_ids(connection, &writes).map_err(CommitError::into_sql)?;
				Self::check_document_actions(connection, &document_actions).map_err(CommitError::into_sql)?;
				for write in &writes {
					Self::apply_table_write(connection, write, committed_seq).map_err(CommitError::into_sql)?;
				}
				Self::apply_document_actions(connection, &document_actions, committed_seq)
					.map_err(CommitError::into_sql)?;
				let metadata_next_id: u64 = metadata_next_id.parse().unwrap_or(0);
				connection.execute(
					"UPDATE durable_metadata SET next_id = ?1, next_seq = ?2 WHERE singleton = 1",
					rusqlite::params![
						metadata_next_id.max(candidate_next_id).to_string(),
						committed_seq.0 + 1,
					],
				)?;
				Ok(committed_seq)
			})
			.await
			.map_err(|error| StorageRejected::new(error.0))?;
		{
			let mut next_id = self.next_id.lock();
			*next_id = (*next_id).max(candidate_next_id);
		}
		Ok(seq)
	}

	async fn mint_id(&self) -> u64 {
		let mut next_id = self.next_id.lock();
		let id = *next_id;
		*next_id += 1;
		id
	}

	async fn conversation(&self, id: ConversationId, _context: &Context) -> Option<ConversationRecord> {
		if *self.closed.lock() {
			return None;
		}
		let db = self.db.clone();
		db.blocking(move |connection| Self::read_conversation_blocking(connection, id))
			.await
			.ok()
			.flatten()
	}

	async fn scan_conversations(
		&self,
		query: &ConversationQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		_context: &Context,
	) -> Page<ConversationRecord> {
		if *self.closed.lock() {
			return Page::empty();
		}
		let after = cursor_id(cursor).map(|after| after as i64).unwrap_or(-1);
		let owner_conversation_id = query.owner_conversation_id.map(|id| id.0 as i64);
		let owner_task_id = query.owner_task_id.map(|id| id.0 as i64);
		let db = self.db.clone();
		let result: Result<Vec<ConversationRecord>, SqliteStorageError> = db
			.blocking(move |connection| {
				let mut sql = String::from("SELECT record FROM conversations WHERE id > ?1");
				let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(after)];
				if let Some(owner_conversation_id) = owner_conversation_id {
					sql.push_str(" AND owner_conversation_id = ?");
					params.push(Box::new(owner_conversation_id));
				}
				if let Some(owner_task_id) = owner_task_id {
					sql.push_str(" AND owner_task_id = ?");
					params.push(Box::new(owner_task_id));
				}
				sql.push_str(" ORDER BY id LIMIT ?");
				params.push(Box::new(limit as i64 + 1));
				let mut statement = connection.prepare(&sql)?;
				let rows = statement.query_map(params_from_iter(params.iter().map(|param| param.as_ref())), |row| {
					row.get::<_, String>(0)
				})?;
				let mut records = Vec::new();
				for row in rows {
					records.push(parse_json(&row?).ok_or_else(|| SqliteStorageError("invalid stored conversation".into()))?);
				}
				Ok(records)
			})
			.await;
		page(result.unwrap_or_default(), |record| record.id.0, limit)
	}

	async fn entry(&self, id: EntryId, _context: &Context) -> Option<(EntryRecord, Seq)> {
		if *self.closed.lock() {
			return None;
		}
		let db = self.db.clone();
		db.blocking(move |connection| {
			let row: Option<(String, i64)> = optional_row(
				connection,
				"SELECT record, commit_seq FROM entries WHERE id = ?1",
				[id.0],
				|row| Ok((row.get(0)?, row.get(1)?)),
			)?;
			let Some((record, commit_seq)) = row else { return Ok(None) };
			Ok(parse_json::<EntryRecord>(&record).map(|entry| (entry, Seq(commit_seq as u64))))
		})
		.await
		.ok()
		.flatten()
	}

	async fn entry_in(
		&self,
		conversation_id: ConversationId,
		id: EntryId,
		_context: &Context,
	) -> Option<(EntryRecord, Seq)> {
		if *self.closed.lock() {
			return None;
		}
		let db = self.db.clone();
		db.blocking(move |connection| {
			// Upstream validates conversation ancestry and throws
			// `Unknown conversation`; the fixed read surface collapses the
			// throw to `None`.
			let Some(mut conversation) = Self::read_conversation_blocking(connection, conversation_id)? else {
				return Ok(None);
			};
			let row: Option<(String, i64)> = optional_row(
				connection,
				"SELECT record, commit_seq FROM entries WHERE id = ?1",
				[id.0],
				|row| Ok((row.get(0)?, row.get(1)?)),
			)?;
			let Some((record, commit_seq)) = row else { return Ok(None) };
			let Some(entry) = parse_json::<EntryRecord>(&record) else { return Ok(None) };
			let mut upper_entry_id = u64::MAX;
			while conversation.id != entry.conversation_id {
				let Some(parent) = &conversation.parent else { return Ok(None) };
				upper_entry_id = upper_entry_id.min(parent.at.0);
				conversation = Self::read_conversation_blocking(connection, parent.conversation_id)?.ok_or_else(|| {
					SqliteStorageError(format!("Unknown conversation: {}", parent.conversation_id.0))
				})?;
			}
			if entry.id.0 > upper_entry_id {
				return Ok(None);
			}
			Ok(Some((entry, Seq(commit_seq as u64))))
		})
		.await
		.ok()
		.flatten()
	}

	async fn find_latest_head_marker(
		&self,
		conversation_id: ConversationId,
		at_or_before_entry_id: Option<EntryId>,
		_context: &Context,
	) -> Option<EntryRecord> {
		if *self.closed.lock() {
			return None;
		}
		let db = self.db.clone();
		db.blocking(move |connection| {
			let Some(mut conversation) = Self::read_conversation_blocking(connection, conversation_id)? else {
				return Ok(None);
			};
			let mut upper: Option<u64> = at_or_before_entry_id.map(|id| id.0);
			loop {
				let record: Option<String> = match upper {
					None => optional_row(
						connection,
						"SELECT record FROM entries WHERE conversation_id = ?1 AND head IS NOT NULL ORDER BY id DESC LIMIT 1",
						[conversation.id.0],
						|row| row.get(0),
					)?,
					Some(upper) => optional_row(
						connection,
						"SELECT record FROM entries WHERE conversation_id = ?1 AND head IS NOT NULL AND id <= ?2 ORDER BY id DESC LIMIT 1",
						rusqlite::params![conversation.id.0, upper as i64],
						|row| row.get(0),
					)?,
				};
				if let Some(record) = record {
					return Ok(parse_json::<EntryRecord>(&record));
				}
				let Some(parent) = conversation.parent.as_ref() else { return Ok(None) };
				upper = Some(upper.map_or(parent.at.0, |upper| upper.min(parent.at.0)));
				conversation = Self::read_conversation_blocking(connection, parent.conversation_id)?.ok_or_else(|| {
					SqliteStorageError(format!("Unknown conversation: {}", parent.conversation_id.0))
				})?;
			}
		})
		.await
		.ok()
		.flatten()
	}

	async fn scan_entries(
		&self,
		query: &EntryQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		_context: &Context,
	) -> Page<EntryRecord> {
		if *self.closed.lock() {
			return Page::empty();
		}
		self.read_entries_page(query, limit, cursor).await
	}

	async fn task(&self, id: TaskId, _context: &Context) -> Option<TaskRecord> {
		if *self.closed.lock() {
			return None;
		}
		let db = self.db.clone();
		db.blocking(move |connection| {
			let record: Option<String> = optional_row(
				connection,
				"SELECT record FROM tasks WHERE id = ?1",
				[id.0],
				|row| row.get(0),
			)?;
			Ok(record.and_then(|record| parse_json(&record)))
		})
		.await
		.ok()
		.flatten()
	}

	async fn scan_tasks(&self, query: &TaskQuery, limit: u64, cursor: Option<&Cursor>, _context: &Context) -> Page<TaskRecord> {
		if *self.closed.lock() {
			return Page::empty();
		}
		let after = cursor_id(cursor).map(|after| after as i64).unwrap_or(-1);
		let conversation_id = query.conversation_id.map(|id| id.0 as i64);
		let kind = query.kind.as_deref().map(encode_indexed_string);
		let status = query.status.map(|status| task_status_str(status).to_string());
		let abort_requested = query.abort_requested.map(|value| if value { 1i64 } else { 0 });
		let background = query.background.map(|value| if value { 1i64 } else { 0 });
		let db = self.db.clone();
		let records: Vec<TaskRecord> = db
			.blocking(move |connection| {
				let mut sql = String::from("SELECT record FROM tasks WHERE id > ?1");
				let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(after)];
				if let Some(conversation_id) = conversation_id {
					sql.push_str(" AND conversation_id = ?");
					params.push(Box::new(conversation_id));
				}
				if let Some(kind) = &kind {
					sql.push_str(" AND kind = ?");
					params.push(Box::new(kind.clone()));
				}
				if let Some(status) = &status {
					sql.push_str(" AND status = ?");
					params.push(Box::new(status.clone()));
				}
				if let Some(abort_requested) = abort_requested {
					sql.push_str(" AND abort_requested = ?");
					params.push(Box::new(abort_requested));
				}
				if let Some(background) = background {
					sql.push_str(" AND background = ?");
					params.push(Box::new(background));
				}
				sql.push_str(" ORDER BY id LIMIT ?");
				params.push(Box::new(limit as i64 + 1));
				let mut statement = connection.prepare(&sql)?;
				let rows = statement.query_map(params_from_iter(params.iter().map(|param| param.as_ref())), |row| {
					row.get::<_, String>(0)
				})?;
				let mut records = Vec::new();
				for row in rows {
					records.push(parse_json(&row?).ok_or_else(|| SqliteStorageError("invalid stored task".into()))?);
				}
				Ok(records)
			})
			.await
			.unwrap_or_default();
		page(records, |record| record.id.0, limit)
	}

	async fn submission(&self, id: SubmissionId, _context: &Context) -> Option<SubmissionRecord> {
		if *self.closed.lock() {
			return None;
		}
		let db = self.db.clone();
		db.blocking(move |connection| {
			let record: Option<String> = optional_row(
				connection,
				"SELECT record FROM submissions WHERE id = ?1",
				[id.0],
				|row| row.get(0),
			)?;
			Ok(record.and_then(|record| parse_json(&record)))
		})
		.await
		.ok()
		.flatten()
	}

	async fn scan_submissions(
		&self,
		query: &SubmissionQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		_context: &Context,
	) -> Page<SubmissionRecord> {
		if *self.closed.lock() {
			return Page::empty();
		}
		let after = cursor_id(cursor).map(|after| after as i64).unwrap_or(-1);
		let conversation_id = query.conversation_id.map(|id| id.0 as i64);
		let status = query.status.map(|status| submission_status_str(status).to_string());
		let db = self.db.clone();
		let records: Vec<SubmissionRecord> = db
			.blocking(move |connection| {
				let mut sql = String::from("SELECT record FROM submissions WHERE id > ?1");
				let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(after)];
				if let Some(conversation_id) = conversation_id {
					sql.push_str(" AND conversation_id = ?");
					params.push(Box::new(conversation_id));
				}
				if let Some(status) = &status {
					sql.push_str(" AND status = ?");
					params.push(Box::new(status.clone()));
				}
				sql.push_str(" ORDER BY id LIMIT ?");
				params.push(Box::new(limit as i64 + 1));
				let mut statement = connection.prepare(&sql)?;
				let rows = statement.query_map(params_from_iter(params.iter().map(|param| param.as_ref())), |row| {
					row.get::<_, String>(0)
				})?;
				let mut records = Vec::new();
				for row in rows {
					records.push(parse_json(&row?).ok_or_else(|| SqliteStorageError("invalid stored submission".into()))?);
				}
				Ok(records)
			})
			.await
			.unwrap_or_default();
		page(records, |record| record.id.0, limit)
	}

	async fn submission_by_request(
		&self,
		conversation_id: ConversationId,
		request_id: &str,
		_context: &Context,
	) -> Option<SubmissionRecord> {
		if *self.closed.lock() {
			return None;
		}
		let request_id = encode_indexed_string(request_id);
		let db = self.db.clone();
		db.blocking(move |connection| {
			let record: Option<String> = optional_row(
				connection,
				"SELECT record FROM submissions WHERE conversation_id = ?1 AND request_id = ?2",
				rusqlite::params![conversation_id.0, request_id],
				|row| row.get(0),
			)?;
			Ok(record.and_then(|record| parse_json(&record)))
		})
		.await
		.ok()
		.flatten()
	}

	async fn find_document(
		&self,
		address: &DocumentAddress,
		at: DocumentPoint,
		_context: &Context,
	) -> Option<DocumentRecord> {
		if *self.closed.lock() {
			return None;
		}
		let parts = address_parts(&address.kind, &address.scope, address.key.as_deref());
		let at = match at {
			DocumentPoint::Current => None,
			DocumentPoint::At(seq) => Some(seq.0 as i64),
		};
		let db = self.db.clone();
		db.blocking(move |connection| {
			let (sql, params): (&str, Vec<Box<dyn rusqlite::ToSql>>) = if at.is_none() {
				(
					"SELECT record FROM documents\n\t\t\t\t\tWHERE kind = ?1 AND scope_kind = ?2 AND owner_id = ?3 AND family = ?4 AND key_value = ?5\n\t\t\t\t\tAND retired_at IS NULL ORDER BY created_at DESC LIMIT 1",
					vec![
						Box::new(parts.kind.clone()),
						Box::new(parts.scope_kind.to_string()),
						Box::new(parts.owner_id as i64),
						Box::new(parts.family as i64),
						Box::new(parts.key_value.clone()),
					],
				)
			} else {
				let at = at.unwrap();
				(
					"SELECT record FROM documents\n\t\t\t\t\tWHERE kind = ?1 AND scope_kind = ?2 AND owner_id = ?3 AND family = ?4 AND key_value = ?5\n\t\t\t\t\tAND created_at <= ?6 AND (retired_at IS NULL OR retired_at > ?7)\n\t\t\t\t\tORDER BY created_at DESC LIMIT 1",
					vec![
						Box::new(parts.kind.clone()),
						Box::new(parts.scope_kind.to_string()),
						Box::new(parts.owner_id as i64),
						Box::new(parts.family as i64),
						Box::new(parts.key_value.clone()),
						Box::new(at),
						Box::new(at),
					],
				)
			};
			let record: Option<String> =
				optional_row(connection, sql, params_from_iter(params.iter().map(|param| param.as_ref())), |row| {
					row.get(0)
				})?;
			Ok(record.and_then(|record| parse_json(&record)))
		})
		.await
		.ok()
		.flatten()
	}

	async fn document(&self, id: DocumentId, at: DocumentPoint, _context: &Context) -> Option<StoredDocument> {
		if *self.closed.lock() {
			return None;
		}
		let db = self.db.clone();
		// The record and revision queries must observe one committed state.
		db.transaction(move |connection| {
			Self::materialize_document(connection, id, at).map_err(SqliteStorageError)
		})
		.await
		.ok()
		.flatten()
	}

	async fn scan_documents(
		&self,
		query: &DocumentQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		_context: &Context,
	) -> Page<DocumentRecord> {
		if *self.closed.lock() {
			return Page::empty();
		}
		let scope = scope_columns(&query.scope);
		let after = cursor_id(cursor).map(|after| after as i64).unwrap_or(-1);
		let kind = query.kind.as_deref().map(encode_indexed_string);
		let at = match query.at {
			DocumentPoint::Current => None,
			DocumentPoint::At(seq) => Some(seq.0 as i64),
		};
		let db = self.db.clone();
		let records: Vec<DocumentRecord> = db
			.blocking(move |connection| {
				let mut sql = String::from("SELECT record FROM documents WHERE scope_kind = ?1 AND owner_id = ?2 AND id > ?3");
				let mut params: Vec<Box<dyn rusqlite::ToSql>> =
					vec![Box::new(scope.scope_kind.to_string()), Box::new(scope.owner_id as i64), Box::new(after)];
				if let Some(kind) = &kind {
					sql.push_str(" AND kind = ?");
					params.push(Box::new(kind.clone()));
				}
				if let Some(at) = at {
					sql.push_str(" AND created_at <= ? AND (retired_at IS NULL OR retired_at > ?)");
					params.push(Box::new(at));
					params.push(Box::new(at));
				} else {
					sql.push_str(" AND retired_at IS NULL");
				}
				sql.push_str(" ORDER BY id LIMIT ?");
				params.push(Box::new(limit as i64 + 1));
				let mut statement = connection.prepare(&sql)?;
				let rows = statement.query_map(params_from_iter(params.iter().map(|param| param.as_ref())), |row| {
					row.get::<_, String>(0)
				})?;
				let mut records = Vec::new();
				for row in rows {
					records.push(parse_json(&row?).ok_or_else(|| SqliteStorageError("invalid stored document".into()))?);
				}
				Ok(records)
			})
			.await
			.unwrap_or_default();
		page(records, |record| record.id.0, limit)
	}

	async fn close(&self, _context: &Context) {
		if *self.closed.lock() {
			return;
		}
		*self.closed.lock() = true;
		let _ = self.db.close().await;
	}
}
