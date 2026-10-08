//! Port of `packages/coding-agent/src/core/tools/file-mutation-queue.ts` —
//! serialize file mutation operations targeting the same file; operations for
//! different files still run in parallel.
//!
//! Upstream chains per-file promises plus a global registration queue; the
//! Rust port keys a registry on the same queue keys (realpath, falling back
//! to the resolved path for ENOENT/ENOTDIR) and serializes with a
//! `tokio::sync::Mutex` per key. Registry access is brief and synchronous,
//! which preserves the upstream registration ordering (D33).

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use parking_lot::Mutex;

struct QueueEntry {
	queue: Arc<tokio::sync::Mutex<()>>,
}

static REGISTRY: Mutex<Option<HashMap<String, QueueEntry>>> = Mutex::new(None);

fn is_missing_path_error(error: &std::io::Error) -> bool {
	matches!(error.kind(), std::io::ErrorKind::NotFound)
}

/// The mutation queue key: the real path when it resolves, else the resolved
/// path itself (`getMutationQueueKey`).
fn get_mutation_queue_key(file_path: &str) -> String {
	let resolved = crate::utils::paths::resolve_path_cwd(file_path);
	match std::fs::canonicalize(&resolved) {
		Ok(real) => real.to_string_lossy().into_owned(),
		Err(error) if is_missing_path_error(&error) => resolved,
		// Upstream rethrows other errors; the port folds them into the
		// resolved-path fallback so operations still serialize (D33).
		Err(_) => resolved,
	}
}

/// Serialize an async file mutation against other mutations of the same file
/// (`withFileMutationQueue`).
pub async fn with_file_mutation_queue<T, F, Fut>(file_path: &str, operation: F) -> T
where
	F: FnOnce() -> Fut,
	Fut: Future<Output = T>,
{
	let key = get_mutation_queue_key(file_path);
	let queue = {
		let mut guard = REGISTRY.lock();
		let registry = guard.get_or_insert_with(HashMap::new);
		let entry = registry
			.entry(key.clone())
			.or_insert_with(|| QueueEntry {
				queue: Arc::new(tokio::sync::Mutex::new(())),
			})
			.queue
			.clone();
		entry
	};

	// Hold the file's queue across the operation.
	let result = {
		let _guard = queue.lock().await;
		operation().await
	};

	// Drop the registry entry when this call is still the last one
	// (`fileMutationQueues.delete(key)` under the same condition).
	{
		let mut reg_guard = REGISTRY.lock();
		let registry = reg_guard.get_or_insert_with(HashMap::new);
		if registry.get(&key).is_some_and(|entry| Arc::ptr_eq(&entry.queue, &queue)) {
			registry.remove(&key);
		}
	}

	result
}
