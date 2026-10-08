//! Port of `packages/coding-agent/src/core/models-store.ts` — the coding-agent
//! model-catalog stores consumed by the model runtime:
//! [`InMemoryCodingAgentModelsStore`] (session-local) and
//! [`FileModelsStore`] (locked JSON file under the agent dir, sharing the
//! file-locked backend with `auth-storage.ts` via
//! [`crate::core::auth_storage::FileAuthStorageBackend`]).
//!
//! Deviations (crate D60):
//! - Provider maps are `BTreeMap` (sorted key order; upstream relies on JS
//!   object insertion order), so rewritten files are key-sorted and persisted
//!   `ModelsStoreEntry` JSON uses the camelCase wire format via the pi-ai
//!   `serde` rename (upstream `lastModified`/`checkedAt`/`etag`).
//! - The single-flight reload promise becomes a `futures::Shared` boxed
//!   future raced per reader with `raceWithAbortSignal`; the reader-count
//!   cleanup (`finally` block) runs on lease drop, and the promise-settlement
//!   cleanup runs at the end of the reload future itself.
//! - `write`/`delete` fold the post-lock `updateReadState` into the lock
//!   closure (unobservable difference; upstream applies it after the lock
//!   returns).
//! - JSON errors carry `serde_json` messages instead of V8 `SyntaxError`
//!   text, and strict `serde` map parsing rejects non-object entry files that
//!   upstream would only trip over when reading a specific provider.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::future::{FutureExt, Shared};
use parking_lot::Mutex;

use pi_ai::auth::types::AuthOperationOptions;
use pi_ai::models_store::{ModelsStore, ModelsStoreEntry, ModelsStoreOperationOptions};
use pi_ai::utils::abort::{operation_signal, race_with_abort_signal, AbortController};
use pi_ai::utils::thrown::ThrownValue;

use crate::config::get_agent_dir;
use crate::core::auth_storage::{FileAuthStorageBackend, LockResult};
use crate::utils::paths::{get_file_revision, join, normalize_path, PathInputOptions};
use crate::utils::text::strip_bom;

/// `StoredModels` — provider catalogs keyed by provider id.
type StoredModels = BTreeMap<String, ModelsStoreEntry>;

/// Reload results are shared between concurrent readers, so the map itself
/// lives behind an `Arc` (upstream readers share one JS object).
type SharedModels = Arc<StoredModels>;

/// Shared single-flight reload handle (upstream `reload.promise`).
type ReloadFuture = Shared<futures_util::future::BoxFuture<'static, Result<SharedModels, ThrownValue>>>;

/// `parse(content)` — `JSON.parse(stripBom(content))`, empty for a missing
/// file. Parse failures reject the operation exactly like the upstream throw.
fn parse_models(content: Option<&str>) -> Result<StoredModels, ThrownValue> {
	let Some(content) = content else {
		return Ok(StoredModels::new());
	};
	serde_json::from_str(strip_bom(content)).map_err(|error| ThrownValue::message(error.to_string()))
}

/// `JSON.stringify(current, null, 2)` for the rewrite payload.
fn serialize_models(models: &StoredModels) -> Result<String, ThrownValue> {
	serde_json::to_string_pretty(models).map_err(|error| ThrownValue::message(error.to_string()))
}

/// Map [`ModelsStoreOperationOptions`] onto the backend's
/// [`AuthOperationOptions`] (structurally identical upstream).
fn auth_options(options: Option<&ModelsStoreOperationOptions>) -> Option<AuthOperationOptions> {
	options.map(|options| AuthOperationOptions { signal: options.signal.clone() })
}

/// `options?.signal?.throwIfAborted()`.
fn check_aborted(options: Option<&ModelsStoreOperationOptions>) -> Result<(), ThrownValue> {
	if let Some(signal) = options.and_then(|options| options.signal.as_ref()) {
		return signal.check().map_err(ThrownValue::error);
	}
	Ok(())
}

/// Session-local [`ModelsStore`] (`InMemoryCodingAgentModelsStore`). Entries
/// are cloned on the way in and out, matching upstream `structuredClone`.
#[derive(Default)]
pub struct InMemoryCodingAgentModelsStore {
	entries: Mutex<BTreeMap<String, ModelsStoreEntry>>,
}

impl InMemoryCodingAgentModelsStore {
	pub fn new() -> InMemoryCodingAgentModelsStore {
		InMemoryCodingAgentModelsStore::default()
	}
}

#[async_trait]
impl ModelsStore for InMemoryCodingAgentModelsStore {
	async fn read(
		&self,
		provider_id: &str,
		options: Option<ModelsStoreOperationOptions>,
	) -> Result<Option<ModelsStoreEntry>, ThrownValue> {
		check_aborted(options.as_ref())?;
		Ok(self.entries.lock().get(provider_id).cloned())
	}

	async fn write(
		&self,
		provider_id: &str,
		entry: ModelsStoreEntry,
		options: Option<ModelsStoreOperationOptions>,
	) -> Result<(), ThrownValue> {
		check_aborted(options.as_ref())?;
		self.entries.lock().insert(provider_id.to_string(), entry);
		Ok(())
	}

	async fn delete(&self, provider_id: &str, options: Option<ModelsStoreOperationOptions>) -> Result<(), ThrownValue> {
		check_aborted(options.as_ref())?;
		self.entries.lock().remove(provider_id);
		Ok(())
	}
}

/// One in-flight reload (upstream `ModelsFileReload`): its own controller so
/// the reload is aborted when the last reader leaves, the shared promise, and
/// the reader count.
struct ModelsFileReload {
	id: u64,
	controller: AbortController,
	promise: ReloadFuture,
	readers: usize,
}

/// Cached file state (upstream `ModelsFileReadState`): last-seen data, the
/// file revision it was read at, and the optional in-flight reload. Shared
/// between every [`FileModelsStore`] opened on the same path.
struct ModelsFileReadState {
	data: SharedModels,
	revision: Option<String>,
	reload: Option<ModelsFileReload>,
}

type ReadStateHandle = Arc<Mutex<ModelsFileReadState>>;

/// Upstream optimizes the common path with one module-level read state keyed
/// by path (`sharedModelsFileReadState`), never retaining more than one path.
static SHARED_MODELS_FILE_READ_STATE: Mutex<Option<(String, ReadStateHandle)>> = Mutex::new(None);

/// Reload generation counter, standing in for JS object identity checks.
static NEXT_RELOAD_ID: AtomicU64 = AtomicU64::new(1);

/// Decrements the reader count when a reader leaves the reload (upstream
/// `finally`), aborting and clearing the reload when the last reader goes.
/// Identity-checked like upstream: a slot holding a newer reload is left
/// alone.
struct ReloadLease {
	state: ReadStateHandle,
	reload_id: u64,
}

impl Drop for ReloadLease {
	fn drop(&mut self) {
		let mut state = self.state.lock();
		if state.reload.as_ref().is_some_and(|reload| reload.id == self.reload_id) {
			let reload = state.reload.as_mut().unwrap();
			reload.readers -= 1;
			if reload.readers == 0 {
				reload.controller.abort();
				state.reload = None;
			}
		}
	}
}

/// Locked JSON-backed storage for dynamically refreshed provider catalogs
/// (`FileModelsStore`). Concurrent instances on the same path share one read
/// state; the default path is `<agent dir>/models-store.json`.
pub struct FileModelsStore {
	storage: Arc<FileAuthStorageBackend>,
	path: String,
	read_state: ReadStateHandle,
}

impl FileModelsStore {
	pub fn new(path: Option<&str>) -> FileModelsStore {
		let default_path = join(&get_agent_dir(), "models-store.json");
		let path = normalize_path(path.unwrap_or(default_path.as_str()), &PathInputOptions::default());
		let storage = Arc::new(FileAuthStorageBackend::new(&path));
		let read_state = {
			let mut shared = SHARED_MODELS_FILE_READ_STATE.lock();
			match shared.as_ref() {
				Some((shared_path, read_state)) if *shared_path == path => Arc::clone(read_state),
				_ => {
					let read_state: ReadStateHandle = Arc::new(Mutex::new(ModelsFileReadState {
						data: SharedModels::new(StoredModels::new()),
						revision: None,
						reload: None,
					}));
					// Upstream only claims the shared slot for the first path.
					if shared.is_none() {
						*shared = Some((path.clone(), Arc::clone(&read_state)));
					}
					read_state
				}
			}
		};
		FileModelsStore { storage, path, read_state }
	}

	/// `reloadFromStorage` as a shareable future: one locked read + parse that
	/// updates the read state, then the upstream promise-settlement cleanup
	/// (`.then` with an identity check on the reload slot).
	fn reload_future(&self, reload_id: u64, controller: &AbortController) -> ReloadFuture {
		let storage = Arc::clone(&self.storage);
		let read_state = Arc::clone(&self.read_state);
		let path = self.path.clone();
		let signal = controller.signal().clone();
		async move {
			let closure_state = Arc::clone(&read_state);
			let outcome = storage
				.with_lock_async(
					move |content| {
						let data = SharedModels::new(parse_models(content.as_deref())?);
						let revision = get_file_revision(&path);
						let mut state = closure_state.lock();
						state.data = Arc::clone(&data);
						state.revision = revision;
						Ok(LockResult { result: Arc::clone(&data), next: None })
					},
					Some(AuthOperationOptions { signal: Some(signal) }),
				)
				.await;
			let mut state = read_state.lock();
			if state.reload.as_ref().is_some_and(|reload| reload.id == reload_id) {
				state.reload = None;
			}
			outcome
		}
		.boxed()
		.shared()
	}

	/// `readLatest`, projecting the freshest map through `project` (upstream
	/// callers index or clone straight out of the awaited object).
	async fn read_latest<T>(
		&self,
		options: Option<&ModelsStoreOperationOptions>,
		project: impl FnOnce(&StoredModels) -> T,
	) -> Result<T, ThrownValue> {
		check_aborted(options)?;
		let signal = operation_signal(options.and_then(|options| options.signal.as_ref()));

		let (promise, reload_id) = {
			let mut state = self.read_state.lock();
			let revision = get_file_revision(&self.path);
			if revision.is_some() && state.revision == revision {
				return Ok(project(&state.data));
			}
			if state.reload.is_none() {
				let controller = AbortController::new();
				let id = NEXT_RELOAD_ID.fetch_add(1, Ordering::Relaxed);
				let promise = self.reload_future(id, &controller);
				state.reload =
					Some(ModelsFileReload { id, controller, promise, readers: 0 });
			}
			let reload = state.reload.as_mut().unwrap();
			reload.readers += 1;
			(reload.promise.clone(), reload.id)
		};

		let lease = ReloadLease { state: Arc::clone(&self.read_state), reload_id };
		// Outer `Result`: the abort race; inner: the reload outcome itself.
		let data = race_with_abort_signal(promise, signal).await.map_err(ThrownValue::error)?;
		let projected = data.map(|data| project(&data));
		drop(lease);
		// Upstream re-checks the signal after the await (`read`).
		check_aborted(options)?;
		projected
	}
}

#[async_trait]
impl ModelsStore for FileModelsStore {
	async fn read(
		&self,
		provider_id: &str,
		options: Option<ModelsStoreOperationOptions>,
	) -> Result<Option<ModelsStoreEntry>, ThrownValue> {
		let entry = self
			.read_latest(options.as_ref(), |entries| entries.get(provider_id).cloned())
			.await?;
		check_aborted(options.as_ref())?;
		Ok(entry)
	}

	async fn write(
		&self,
		provider_id: &str,
		entry: ModelsStoreEntry,
		options: Option<ModelsStoreOperationOptions>,
	) -> Result<(), ThrownValue> {
		let read_state = Arc::clone(&self.read_state);
		let provider_id = provider_id.to_string();
		self.storage
			.with_lock_async(
				move |content| {
					let mut current = parse_models(content.as_deref())?;
					current.insert(provider_id, entry);
					let next = serialize_models(&current)?;
					let mut state = read_state.lock();
					state.data = SharedModels::new(current);
					state.revision = None;
					Ok(LockResult { result: (), next: Some(next) })
				},
				auth_options(options.as_ref()),
			)
			.await
	}

	async fn delete(&self, provider_id: &str, options: Option<ModelsStoreOperationOptions>) -> Result<(), ThrownValue> {
		let read_state = Arc::clone(&self.read_state);
		let provider_id = provider_id.to_string();
		self.storage
			.with_lock_async(
				move |content| {
					let mut current = parse_models(content.as_deref())?;
					current.remove(&provider_id);
					let next = serialize_models(&current)?;
					let mut state = read_state.lock();
					state.data = SharedModels::new(current);
					state.revision = None;
					Ok(LockResult { result: (), next: Some(next) })
				},
				auth_options(options.as_ref()),
			)
			.await
	}
}

#[cfg(test)]
mod models_store_wrapper_tests {
	use super::*;

	fn entry(etag: &str) -> ModelsStoreEntry {
		ModelsStoreEntry { etag: Some(etag.to_string()), ..Default::default() }
	}

	fn temp_dir(label: &str) -> String {
		let unique = std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.unwrap()
			.as_nanos();
		std::env::temp_dir()
			.join(format!("pi-models-store-{}-{}-{}", label, std::process::id(), unique))
			.to_string_lossy()
			.into_owned()
	}

	fn cleanup_dir(dir: &str) {
		let _ = std::fs::remove_dir_all(dir);
	}

	// `InMemoryCodingAgentModelsStore` --------------------------------------

	#[tokio::test]
	async fn in_memory_store_read_write_delete_round_trips_entries() {
		let store = InMemoryCodingAgentModelsStore::new();
		assert_eq!(store.read("p1", None).await.unwrap(), None);
		store.write("p1", entry("e1"), None).await.unwrap();
		assert_eq!(store.read("p1", None).await.unwrap(), Some(entry("e1")));
		store.write("p1", entry("e2"), None).await.unwrap();
		assert_eq!(
			store.read("p1", None).await.unwrap().unwrap().etag.as_deref(),
			Some("e2"),
			"write overwrites the previous entry"
		);
		store.delete("p1", None).await.unwrap();
		assert_eq!(store.read("p1", None).await.unwrap(), None);
		store.delete("p1", None).await.unwrap();
		assert_eq!(store.read("p1", None).await.unwrap(), None);
	}

	#[tokio::test]
	async fn in_memory_store_reads_return_independent_clones() {
		let store = InMemoryCodingAgentModelsStore::new();
		store.write("p1", entry("etag"), None).await.unwrap();
		let mut read_back = store.read("p1", None).await.unwrap().unwrap();
		read_back.etag = None;
		let again = store.read("p1", None).await.unwrap().unwrap();
		assert_eq!(again.etag.as_deref(), Some("etag"), "mutating a read entry leaves the store untouched");
	}

	#[tokio::test]
	async fn in_memory_store_operations_reject_a_pre_aborted_signal() {
		let store = InMemoryCodingAgentModelsStore::new();
		let controller = AbortController::new();
		controller.abort();
		let options = ModelsStoreOperationOptions { signal: Some(controller.signal().clone()) };
		assert!(store.read("p1", Some(options.clone())).await.is_err());
		assert!(store.write("p1", entry("e"), Some(options.clone())).await.is_err());
		assert!(store.delete("p1", Some(options)).await.is_err());
	}

	// `FileModelsStore` ------------------------------------------------------

	#[tokio::test]
	async fn file_store_round_trips_entries_and_persists_across_instances() {
		let temp = temp_dir("round-trip");
		std::fs::create_dir_all(&temp).unwrap();
		let path = format!("{temp}/models-store.json");
		let store = FileModelsStore::new(Some(&path));

		let mut first = entry("etag-1");
		first.last_modified = Some(111);
		first.checked_at = Some(222);
		store.write("p1", first.clone(), None).await.unwrap();
		store.write("p2", entry("etag-2"), None).await.unwrap();
		assert_eq!(store.read("p1", None).await.unwrap(), Some(first));

		// Persisted wire format: pretty-printed camelCase JSON, absent fields
		// for `None` optionals.
		let raw = std::fs::read_to_string(&path).unwrap();
		assert!(raw.contains("\"lastModified\": 111"), "camelCase fields on disk, got: {raw}");
		assert!(!raw.contains("last_modified"), "no snake_case leakage, got: {raw}");

		store.delete("p1", None).await.unwrap();
		assert_eq!(store.read("p1", None).await.unwrap(), None);

		// A fresh store over the same file sees the persisted state.
		let fresh = FileModelsStore::new(Some(&path));
		assert_eq!(fresh.read("p2", None).await.unwrap().unwrap().etag.as_deref(), Some("etag-2"));
		assert_eq!(fresh.read("p1", None).await.unwrap(), None);

		cleanup_dir(&temp);
	}

	#[tokio::test]
	async fn file_store_parses_files_with_a_utf8_bom() {
		let temp = temp_dir("bom");
		std::fs::create_dir_all(&temp).unwrap();
		let path = format!("{temp}/models-store.json");
		std::fs::write(&path, "\u{FEFF}{\"p1\":{\"models\":[],\"etag\":\"bom\"}}").unwrap();

		let store = FileModelsStore::new(Some(&path));
		let read = store.read("p1", None).await.unwrap().unwrap();
		assert_eq!(read.etag.as_deref(), Some("bom"));
		assert!(read.models.is_empty());

		cleanup_dir(&temp);
	}

	#[tokio::test]
	async fn file_store_surfaces_corrupt_json_as_an_error() {
		let temp = temp_dir("corrupt");
		std::fs::create_dir_all(&temp).unwrap();
		let path = format!("{temp}/models-store.json");
		std::fs::write(&path, "not json").unwrap();

		let store = FileModelsStore::new(Some(&path));
		assert!(store.read("p1", None).await.is_err());

		cleanup_dir(&temp);
	}

	#[tokio::test]
	async fn file_store_detects_external_file_modifications() {
		let temp = temp_dir("external");
		std::fs::create_dir_all(&temp).unwrap();
		let path = format!("{temp}/models-store.json");
		let store = FileModelsStore::new(Some(&path));

		store.write("p1", entry("first"), None).await.unwrap();
		assert_eq!(store.read("p1", None).await.unwrap().unwrap().etag.as_deref(), Some("first"));

		// An external writer replaces the file behind the store's back; the
		// revision cache must not serve the stale entry.
		std::fs::write(&path, "{\"p1\":{\"models\":[],\"etag\":\"external\"}}").unwrap();
		assert_eq!(
			store.read("p1", None).await.unwrap().unwrap().etag.as_deref(),
			Some("external"),
			"reads pick up external modifications"
		);

		cleanup_dir(&temp);
	}

	#[tokio::test]
	async fn file_store_instances_on_different_paths_keep_independent_state() {
		let temp = temp_dir("independent");
		std::fs::create_dir_all(&temp).unwrap();
		let path_a = format!("{temp}/a.json");
		let path_b = format!("{temp}/b.json");

		let store_a = FileModelsStore::new(Some(&path_a));
		let store_b = FileModelsStore::new(Some(&path_b));
		store_a.write("p1", entry("a"), None).await.unwrap();
		assert_eq!(store_b.read("p1", None).await.unwrap(), None, "different paths do not share state");
		assert_eq!(store_a.read("p1", None).await.unwrap().unwrap().etag.as_deref(), Some("a"));

		// A third instance on yet another path starts empty too.
		let store_c = FileModelsStore::new(Some(&format!("{temp}/c.json")));
		assert_eq!(store_c.read("p1", None).await.unwrap(), None);

		cleanup_dir(&temp);
	}

	#[tokio::test]
	async fn file_store_write_is_immediately_visible_without_a_revision() {
		let temp = temp_dir("write-visible");
		std::fs::create_dir_all(&temp).unwrap();
		let path = format!("{temp}/models-store.json");
		let store = FileModelsStore::new(Some(&path));

		store.write("p1", entry("v1"), None).await.unwrap();
		assert_eq!(store.read("p1", None).await.unwrap().unwrap().etag.as_deref(), Some("v1"));
		// Overwrite after the revision was cached; write clears it.
		store.write("p1", entry("v2"), None).await.unwrap();
		assert_eq!(store.read("p1", None).await.unwrap().unwrap().etag.as_deref(), Some("v2"));

		cleanup_dir(&temp);
	}
}
