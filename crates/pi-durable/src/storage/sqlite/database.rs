//! Port of `packages/durable/src/storage/sqlite/database.ts` plus the
//! `node.ts` rusqlite adapter (folded per module D1).

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rusqlite::Connection;

/// Failure of a durable SQLite operation; the message keeps the upstream error
/// strings (module D3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SqliteStorageError(pub String);

impl std::fmt::Display for SqliteStorageError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str(&self.0)
	}
}

impl std::error::Error for SqliteStorageError {}

impl From<rusqlite::Error> for SqliteStorageError {
	fn from(error: rusqlite::Error) -> Self {
		Self(error.to_string())
	}
}

/// SQLite WAL auto-checkpoint threshold; 0 disables it.
pub const DEFAULT_WAL_AUTO_CHECKPOINT_PAGES: u32 = 1_000;
/// Time SQLite waits for a competing file lock.
pub const DEFAULT_BUSY_TIMEOUT_MS: u64 = 5_000;

/// Node SQLite connection settings for a durable storage file (upstream
/// `NodeSqliteStorageOptions`).
#[derive(Debug, Clone, Copy)]
pub struct NodeSqliteStorageOptions {
	pub wal_auto_checkpoint_pages: Option<u32>,
	pub busy_timeout_ms: Option<u64>,
}

impl Default for NodeSqliteStorageOptions {
	fn default() -> Self {
		Self { wal_auto_checkpoint_pages: None, busy_timeout_ms: None }
	}
}

/// Minimal rusqlite-backed database facade required by [`super::SqliteStorage`]
/// (upstream `SqliteDatabase` + `NodeSqliteDatabase`).
///
/// All operations run off the async runtime through `spawn_blocking`, serialized
/// by a mutex around the connection. `transaction` runs its callback under
/// `BEGIN IMMEDIATE`; unrelated operations queue behind it.
pub struct SqliteDatabase {
	connection: Arc<Mutex<Option<Connection>>>,
	closed: AtomicBool,
}

impl SqliteDatabase {
	/// Open and configure a database at `path` (`:memory:` supported).
	pub fn open(path: impl AsRef<Path>, options: &NodeSqliteStorageOptions) -> Result<Self, SqliteStorageError> {
		let path = path.as_ref();
		if path != Path::new(":memory:") {
			if let Some(parent) = path.parent() {
				std::fs::create_dir_all(parent)
					.map_err(|error| SqliteStorageError(format!("failed to create database directory: {error}")))?;
			}
		}
		let connection = Connection::open(path)?;
		let busy_timeout_ms = options.busy_timeout_ms.unwrap_or(DEFAULT_BUSY_TIMEOUT_MS);
		connection.busy_timeout(std::time::Duration::from_millis(busy_timeout_ms))?;
		connection.pragma_update(None, "journal_mode", "WAL")?;
		connection.pragma_update(None, "synchronous", "NORMAL")?;
		let checkpoint_pages = options.wal_auto_checkpoint_pages.unwrap_or(DEFAULT_WAL_AUTO_CHECKPOINT_PAGES);
		connection.pragma_update(None, "wal_autocheckpoint", checkpoint_pages)?;
		Ok(Self { connection: Arc::new(Mutex::new(Some(connection))), closed: AtomicBool::new(false) })
	}

	/// Alias of [`Self::open`] matching the upstream `openNodeSqliteDatabase`
	/// factory name.
	pub fn open_node(
		path: impl AsRef<Path>,
		options: &NodeSqliteStorageOptions,
	) -> Result<Self, SqliteStorageError> {
		Self::open(path, options)
	}

	pub(super) async fn blocking<T, F>(&self, operation: F) -> Result<T, SqliteStorageError>
	where
		T: Send + 'static,
		F: FnOnce(&mut Connection) -> Result<T, SqliteStorageError> + Send + 'static,
	{
		let connection = self.connection.clone();
		tokio::task::spawn_blocking(move || {
			let mut guard = connection.lock().expect("connection mutex");
			let Some(connection) = guard.as_mut() else {
				return Err(SqliteStorageError("SQLite database is closed".into()));
			};
			operation(connection)
		})
		.await
		.map_err(|error| SqliteStorageError(format!("SQLite blocking task failed: {error}")))?
	}

	/// Run SQL text without bindings; may contain several statements.
	pub async fn exec(&self, sql: impl Into<String>) -> Result<(), SqliteStorageError> {
		let sql = sql.into();
		self.blocking(move |connection| {
			connection.execute_batch(&sql)?;
			Ok(())
		})
		.await
	}

	/// Run one transaction (`BEGIN IMMEDIATE` … `COMMIT`/`ROLLBACK`). The
	/// callback must use the passed connection exclusively; it executes on a
	/// blocking thread while the connection mutex is held.
	pub async fn transaction<T, F>(&self, callback: F) -> Result<T, SqliteStorageError>
	where
		T: Send + 'static,
		F: FnOnce(&mut Connection) -> Result<T, SqliteStorageError> + Send + 'static,
	{
		self.blocking(move |connection| {
			connection.execute_batch("BEGIN IMMEDIATE")?;
			match callback(connection) {
				Ok(value) => {
					connection.execute_batch("COMMIT")?;
					Ok(value)
				}
				Err(error) => {
					if let Err(rollback) = connection.execute_batch("ROLLBACK") {
						return Err(SqliteStorageError(format!(
							"SQLite transaction failed and rollback failed: {error}; rollback: {rollback}"
						)));
					}
					Err(error)
				}
			}
		})
		.await
	}

	/// Close the database, checkpointing the WAL. Idempotent.
	pub async fn close(&self) -> Result<(), SqliteStorageError> {
		if self.closed.swap(true, Ordering::SeqCst) {
			return Ok(());
		}
		self.blocking(|connection| {
			connection.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
			Ok(())
		})
		.await?;
		self.connection.lock().expect("connection mutex").take();
		Ok(())
	}
}
