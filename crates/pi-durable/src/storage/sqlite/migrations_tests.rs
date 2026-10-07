//! Port of `packages/durable/test/sqlite-migrations.test.ts`.
//!
//! Cases run against the portable [`SqliteDatabase`] facade (Node adapter
//! folded, module D1); schema introspection uses an independent read-only
//! rusqlite connection standing in for upstream's second `DatabaseSync`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use super::database::{NodeSqliteStorageOptions, SqliteDatabase};
use super::migrations::{
	SqliteMigration, apply_sqlite_migrations, current_sqlite_schema_version, sqlite_migrations,
};
use crate::storage::conformance::{ROOT_CONVERSATION_ID, commit_ok};
use crate::types::{EntryId, Storage};

static DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

fn database_path() -> PathBuf {
	let directory = std::env::temp_dir().join(format!(
		"pi-durable-migrations-{}-{}",
		std::process::id(),
		DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed)
	));
	let _ = std::fs::remove_dir_all(&directory);
	directory.join("storage.sqlite")
}

fn options() -> NodeSqliteStorageOptions {
	NodeSqliteStorageOptions::default()
}

fn raw_connection(path: &std::path::Path) -> rusqlite::Connection {
	rusqlite::Connection::open_with_flags(
		path,
		rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
	)
	.unwrap()
}

fn query_row(connection: &rusqlite::Connection, sql: &str) -> Value {
	let mut statement = connection.prepare(sql).unwrap();
	let names: Vec<String> = statement.column_names().iter().map(|name| name.to_string()).collect();
	let mut rows = statement.query([]).unwrap();
	let mut row = rows.next().unwrap().expect("row exists");
	let mut object = serde_json::Map::new();
	for (index, column) in names.iter().enumerate() {
		let name = column.clone();
		let value: Value = match row.get_ref(index).unwrap() {
			rusqlite::types::ValueRef::Null => Value::Null,
			rusqlite::types::ValueRef::Integer(value) => json!(value),
			rusqlite::types::ValueRef::Real(value) => json!(value),
			rusqlite::types::ValueRef::Text(text) => {
				json!(std::str::from_utf8(text).unwrap())
			}
			rusqlite::types::ValueRef::Blob(_) => panic!("unexpected blob"),
		};
		object.insert(name, value);
	}
	Value::Object(object)
}

fn scalar(connection: &rusqlite::Connection, sql: &str) -> i64 {
	connection.query_row(sql, [], |row| row.get(0)).unwrap()
}

#[tokio::test]
async fn creates_the_current_schema_and_can_be_applied_repeatedly() {
	let path = database_path();
	let database = SqliteDatabase::open(&path, &options()).unwrap();
	apply_sqlite_migrations(&database, sqlite_migrations()).await.unwrap();
	apply_sqlite_migrations(&database, sqlite_migrations()).await.unwrap();
	let connection = raw_connection(&path);
	assert_eq!(
		query_row(&connection, "SELECT version FROM durable_schema WHERE singleton = 1"),
		json!({ "version": current_sqlite_schema_version() })
	);
	assert_eq!(
		query_row(&connection, "SELECT next_id, next_seq FROM durable_metadata WHERE singleton = 1"),
		json!({ "next_id": "2", "next_seq": 1 })
	);
}

#[tokio::test]
async fn rejects_a_database_newer_than_the_portable_core() {
	let path = database_path();
	let database = SqliteDatabase::open(&path, &options()).unwrap();
	apply_sqlite_migrations(&database, sqlite_migrations()).await.unwrap();
	database
		.exec(format!(
			"UPDATE durable_schema SET version = {} WHERE singleton = 1",
			current_sqlite_schema_version() + 1
		))
		.await
		.unwrap();
	database.close().await.unwrap();

	let error = match super::storage::SqliteStorage::open_path(&path, &options()).await {
		Err(error) => error,
		Ok(_) => panic!("newer schema must reject"),
	};
	assert!(error.to_string().contains("is newer than supported version"), "{error}");
}

#[tokio::test]
async fn rolls_initial_bootstrap_and_every_pending_migration_back_together() {
	let path = database_path();
	let database = SqliteDatabase::open(&path, &options()).unwrap();
	let failed = vec![
		SqliteMigration {
			version: 1,
			statements: vec![
				"CREATE TABLE migration_first (value TEXT) STRICT".into(),
				"INSERT INTO migration_first (value) VALUES ('retained')".into(),
			],
		},
		SqliteMigration {
			version: 2,
			statements: vec!["CREATE TABLE migration_second (value TEXT) STRICT".into(), "THIS IS NOT SQL".into()],
		},
	];
	apply_sqlite_migrations(&database, &failed).await.expect_err("second migration fails");
	{
		let connection = raw_connection(&path);
		assert_eq!(
			scalar(
				&connection,
				"SELECT count(*) FROM sqlite_schema WHERE name IN ('durable_schema', 'migration_first', 'migration_second')"
			),
			0
		);
	}

	let successful = vec![
		failed[0].clone(),
		SqliteMigration {
			version: 2,
			statements: vec!["CREATE TABLE migration_second (value TEXT) STRICT".into()],
		},
	];
	apply_sqlite_migrations(&database, &successful).await.unwrap();
	let connection = raw_connection(&path);
	assert_eq!(
		query_row(&connection, "SELECT version FROM durable_schema WHERE singleton = 1"),
		json!({ "version": 2 })
	);
	assert_eq!(query_row(&connection, "SELECT value FROM migration_first"), json!({ "value": "retained" }));
}

#[tokio::test]
async fn rolls_a_failed_migration_back_and_preserves_stored_data_for_a_successful_retry() {
	let path = database_path();
	let storage = super::storage::SqliteStorage::open_path(&path, &options()).await.unwrap();
	let shared: std::sync::Arc<dyn Storage> = std::sync::Arc::new(storage);
	commit_ok(
		&shared,
		vec![
			json!({ "type": "conversation", "value": { "id": ROOT_CONVERSATION_ID.0 } }),
			json!({
				"type": "entry",
				"value": {
					"id": 2,
					"conversationId": ROOT_CONVERSATION_ID.0,
					"kind": "retained",
					"data": { "retained": true },
				},
			}),
		],
	)
	.await;
	shared.close(crate::storage::conformance::ctx()).await;

	let database = SqliteDatabase::open(&path, &options()).unwrap();
	let next_version = current_sqlite_schema_version() + 1;
	let mut failed_migrations = sqlite_migrations().to_vec();
	failed_migrations.push(SqliteMigration {
		version: next_version,
		statements: vec!["CREATE TABLE migration_probe (value TEXT) STRICT".into(), "THIS IS NOT SQL".into()],
	});
	apply_sqlite_migrations(&database, &failed_migrations).await.expect_err("probe migration fails");
	{
		let connection = raw_connection(&path);
		assert_eq!(
			query_row(&connection, "SELECT version FROM durable_schema WHERE singleton = 1"),
			json!({ "version": current_sqlite_schema_version() })
		);
		assert_eq!(
			scalar(&connection, "SELECT count(*) FROM sqlite_schema WHERE type = 'table' AND name = 'migration_probe'"),
			0
		);
	}

	let mut successful_migrations = sqlite_migrations().to_vec();
	successful_migrations.push(SqliteMigration {
		version: next_version,
		statements: vec!["CREATE TABLE migration_probe (value TEXT) STRICT".into()],
	});
	apply_sqlite_migrations(&database, &successful_migrations).await.unwrap();
	let connection = raw_connection(&path);
	assert_eq!(query_row(&connection, "SELECT version FROM durable_schema WHERE singleton = 1"), json!({ "version": next_version }));
	let stored = query_row(&connection, "SELECT record, commit_seq FROM entries WHERE id = 2");
	assert_eq!(
		serde_json::from_str::<Value>(stored["record"].as_str().unwrap()).unwrap(),
		json!({
			"id": 2,
			"conversationId": ROOT_CONVERSATION_ID.0,
			"kind": "retained",
			"data": { "retained": true },
		})
	);
	assert_eq!(stored["commit_seq"], json!(1));
	assert_eq!(
		query_row(&connection, "SELECT next_id, next_seq FROM durable_metadata WHERE singleton = 1"),
		json!({ "next_id": "3", "next_seq": 2 })
	);
	// Reference `EntryId` so the upstream import maps to the same nominal type.
	let _ = EntryId(2);
}
