//! Port of `packages/durable/src/storage/sqlite` (`database.ts`,
//! `migrations.ts`, `storage.ts`, `index.ts`, `node.ts`).
//!
//! Deviations (module ledger):
//! - D1 (node.ts fold): upstream defines an async `SqliteExecutor`/
//!   `SqliteDatabase` adapter interface plus a `node:sqlite` adapter with a
//!   serial operation queue; the Rust port has a single rusqlite-backed
//!   [`SqliteDatabase`] whose blocking calls run through
//!   `tokio::task::spawn_blocking` around a mutex-guarded connection. The
//!   upstream adapter trait has no second implementor and is omitted.
//! - D2: transaction settlement is synchronous-within-blocking, so the Node
//!   microtask-settlement semantics (`ControlledSettlementDatabase`) have no
//!   direct counterpart; serialization of concurrent transactions is provided
//!   by the connection mutex.
//! - D3: plain upstream `Error`s collapse into [`SqliteStorageError`] (open
//!   path) and [`StorageRejected`] (commit path) with upstream message strings.

mod database;
mod migrations;
mod storage;

#[cfg(test)]
mod migrations_tests;
#[cfg(test)]
mod storage_tests;

pub use database::{
	NodeSqliteStorageOptions, SqliteDatabase, SqliteStorageError, DEFAULT_BUSY_TIMEOUT_MS,
	DEFAULT_WAL_AUTO_CHECKPOINT_PAGES,
};
pub use migrations::{apply_sqlite_migrations, current_sqlite_schema_version, sqlite_migrations, SqliteMigration};
pub use storage::SqliteStorage;
