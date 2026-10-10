//! Port of `packages/durable/src/storage/sqlite/migrations.ts`.
//!
//! The schema statement list is byte-compatible with upstream, including the
//! `STRICT` table options and the `TEXT` `next_id` column (upstream keeps it
//! TEXT because `node:sqlite` rejects INTEGER results outside JavaScript's
//! safe integer range; rusqlite reads i64 natively but the wire shape and
//! stored values stay identical).

use super::database::{SqliteDatabase, SqliteStorageError};

/// One immutable step of the schema history.
#[derive(Debug, Clone)]
pub struct SqliteMigration {
    pub version: u64,
    pub statements: Vec<String>,
}

const INITIAL_SCHEMA: &[&str] = &[
	"CREATE TABLE durable_metadata (\n\t\tsingleton INTEGER PRIMARY KEY CHECK (singleton = 1),\n\t\tnext_id TEXT NOT NULL,\n\t\tnext_seq INTEGER NOT NULL\n\t) STRICT",
	"INSERT INTO durable_metadata (singleton, next_id, next_seq) VALUES (1, '2', 1)",
	"CREATE TABLE record_ids (\n\t\tid INTEGER PRIMARY KEY,\n\t\trecord_type TEXT NOT NULL CHECK (record_type IN ('conversation', 'entry', 'task', 'submission', 'document'))\n\t) STRICT",
	"CREATE TABLE conversations (\n\t\tid INTEGER PRIMARY KEY,\n\t\towner_conversation_id INTEGER,\n\t\towner_task_id INTEGER,\n\t\trecord TEXT NOT NULL CHECK (json_valid(record))\n\t) STRICT",
	"CREATE INDEX conversations_by_owner_conversation ON conversations (owner_conversation_id, id)",
	"CREATE INDEX conversations_by_owner_task ON conversations (owner_task_id, id)",
	"CREATE TABLE entries (\n\t\tid INTEGER PRIMARY KEY,\n\t\tconversation_id INTEGER NOT NULL,\n\t\thead INTEGER,\n\t\tcommit_seq INTEGER NOT NULL,\n\t\trecord TEXT NOT NULL CHECK (json_valid(record))\n\t) STRICT",
	"CREATE INDEX entries_by_conversation ON entries (conversation_id, id DESC)",
	"CREATE INDEX entry_heads_by_conversation ON entries (conversation_id, id DESC) WHERE head IS NOT NULL",
	"CREATE TABLE tasks (\n\t\tid INTEGER PRIMARY KEY,\n\t\tconversation_id INTEGER NOT NULL,\n\t\tkind TEXT NOT NULL,\n\t\tstatus TEXT NOT NULL CHECK (status IN ('pending', 'running', 'waiting', 'completing', 'terminal')),\n\t\tabort_requested INTEGER NOT NULL CHECK (abort_requested IN (0, 1)),\n\t\tbackground INTEGER NOT NULL CHECK (background IN (0, 1)),\n\t\trecord TEXT NOT NULL CHECK (json_valid(record))\n\t) STRICT",
	"CREATE INDEX tasks_by_status ON tasks (status, id)",
	"CREATE INDEX tasks_by_conversation ON tasks (conversation_id, id)",
	"CREATE INDEX tasks_by_kind ON tasks (kind, id)",
	"CREATE INDEX tasks_by_abort_requested ON tasks (abort_requested, id)",
	"CREATE INDEX tasks_by_background ON tasks (background, id)",
	"CREATE TABLE submissions (\n\t\tid INTEGER PRIMARY KEY,\n\t\tconversation_id INTEGER NOT NULL,\n\t\trequest_id TEXT,\n\t\tstatus TEXT NOT NULL CHECK (status IN ('queued', 'placed', 'done', 'unanswered')),\n\t\trecord TEXT NOT NULL CHECK (json_valid(record))\n\t) STRICT",
	"CREATE INDEX submissions_by_request ON submissions (conversation_id, request_id)",
	"CREATE INDEX submissions_by_conversation ON submissions (conversation_id, id)",
	"CREATE INDEX submissions_by_status ON submissions (status, id)",
	"CREATE TABLE documents (\n\t\tid INTEGER PRIMARY KEY,\n\t\tkind TEXT NOT NULL,\n\t\tfamily INTEGER NOT NULL CHECK (family IN (0, 1)),\n\t\tkey_value TEXT NOT NULL,\n\t\tscope_kind TEXT NOT NULL CHECK (scope_kind IN ('session', 'conversation', 'task')),\n\t\towner_id INTEGER NOT NULL,\n\t\tcreated_at INTEGER NOT NULL,\n\t\tretired_at INTEGER,\n\t\trecord TEXT NOT NULL CHECK (json_valid(record))\n\t) STRICT",
	"CREATE INDEX documents_by_address\n\t\tON documents (kind, scope_kind, owner_id, family, key_value, created_at DESC, retired_at)",
	"CREATE INDEX documents_by_scope ON documents (scope_kind, owner_id, id)",
	"CREATE INDEX documents_by_scope_kind ON documents (scope_kind, owner_id, kind, id)",
	"CREATE TABLE document_revisions (\n\t\tdocument_id INTEGER NOT NULL,\n\t\tseq INTEGER NOT NULL,\n\t\tkind TEXT NOT NULL CHECK (kind IN ('base', 'delta')),\n\t\tversion INTEGER NOT NULL,\n\t\tcontent TEXT NOT NULL CHECK (json_valid(content)),\n\t\tPRIMARY KEY (document_id, seq)\n\t) STRICT",
	"CREATE INDEX document_revisions_by_kind ON document_revisions (document_id, kind, seq DESC)",
];

/// Immutable, ordered schema history. Append new migrations after the initial
/// schema ships.
pub fn sqlite_migrations() -> &'static [SqliteMigration] {
    static MIGRATIONS: std::sync::LazyLock<Vec<SqliteMigration>> = std::sync::LazyLock::new(|| {
        vec![SqliteMigration {
            version: 1,
            statements: INITIAL_SCHEMA
                .iter()
                .map(|statement| (*statement).to_string())
                .collect(),
        }]
    });
    &MIGRATIONS
}

/// Version of the newest migration.
pub fn current_sqlite_schema_version() -> u64 {
    sqlite_migrations()
        .last()
        .map(|migration| migration.version)
        .unwrap_or(0)
}

/// Apply all pending schema migrations atomically.
pub async fn apply_sqlite_migrations(
    database: &SqliteDatabase,
    migrations: &[SqliteMigration],
) -> Result<(), SqliteStorageError> {
    for (index, migration) in migrations.iter().enumerate() {
        if migration.version != index as u64 + 1 {
            return Err(SqliteStorageError(
                "Durable SQLite migrations must have contiguous versions starting at 1".into(),
            ));
        }
    }
    let migrations: Vec<(u64, Vec<String>)> = migrations
        .iter()
        .map(|migration| (migration.version, migration.statements.clone()))
        .collect();
    database
		.transaction(move |connection| {
			connection.execute_batch(
				"CREATE TABLE IF NOT EXISTS durable_schema (\n\t\t\tsingleton INTEGER PRIMARY KEY CHECK (singleton = 1),\n\t\t\tversion INTEGER NOT NULL CHECK (version >= 0)\n\t\t) STRICT",
			)?;
			connection.execute("INSERT OR IGNORE INTO durable_schema (singleton, version) VALUES (1, 0)", [])?;
			let stored_version: Option<u64> = connection
				.query_row("SELECT version FROM durable_schema WHERE singleton = 1", [], |row| row.get(0))
				.map(Some)
				.or_else(|error| match error {
					rusqlite::Error::QueryReturnedNoRows => Ok(None),
					error => Err(error),
				})?;
			let Some(stored_version) = stored_version else {
				return Err(SqliteStorageError("Durable SQLite schema metadata is missing".into()));
			};
			let current_version = migrations.last().map(|(version, _)| *version).unwrap_or(0);
			if stored_version > current_version {
				return Err(SqliteStorageError(format!(
					"Durable SQLite schema version {} is newer than supported version {}",
					stored_version, current_version
				)));
			}
			for (version, statements) in &migrations {
				if *version <= stored_version {
					continue;
				}
				for statement in statements {
					connection.execute_batch(statement)?;
				}
				connection.execute("UPDATE durable_schema SET version = ?1 WHERE singleton = 1", [version])?;
			}
			Ok(())
		})
		.await
}
