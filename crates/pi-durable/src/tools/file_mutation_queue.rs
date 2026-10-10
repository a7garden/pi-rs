//! Port of `packages/durable/src/tools/file-mutation-queue.ts` — per-file mutation serialization.
//!
//! D1: upstream chains promises in a module-global `Map<string, Promise<void>>`; Rust uses a
//! global map of keyed [`tokio::sync::Mutex`] guards, which preserves the observed contract:
//! mutations of one file (same file-system id and canonical path) exclude each other, other
//! files never wait, and the entry is dropped when its holder finishes. Promise-chain FIFO
//! order is approximated by tokio's mutex wakeups; the upstream guarantee is exclusion, not
//! strict ordering ("concurrent calls run in the order their keys resolve").

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use parking_lot::Mutex;
use pi_chord::context::Context;
use tokio::sync::OwnedMutexGuard;

use crate::env::{get_or_throw, ExecutionEnv, FileErrorCode};

/// Tail of the mutation chain of each file, keyed by file-system id and canonical path.
static QUEUES: Mutex<Option<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> = Mutex::new(None);

async fn mutation_key(env: &Arc<dyn ExecutionEnv>, path: &str, context: &Context) -> String {
    let absolute_path = get_or_throw(env.absolute_path(path, context).await);
    format!(
        "{}\0{}",
        env.id(),
        canonical(env, &absolute_path, context).await
    )
}

/// The canonical path; for a file that does not exist yet, its canonical parent joined with its
/// name, so a `write` that creates a file and a later mutation of it share one key even under a
/// symlinked directory. (Boxed because the missing-file branch recurses.)
fn canonical<'a>(
    env: &'a Arc<dyn ExecutionEnv>,
    absolute_path: &'a str,
    context: &'a Context,
) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(canonical_inner(env, absolute_path, context))
}

async fn canonical_inner(
    env: &Arc<dyn ExecutionEnv>,
    absolute_path: &str,
    context: &Context,
) -> String {
    match env.canonical_path(absolute_path, context).await {
        Ok(value) => value,
        Err(error) => match error.code {
            FileErrorCode::NotSupported => absolute_path.to_string(),
            FileErrorCode::NotFound => {
                // The file system splits the path, so a name may contain characters that are separators elsewhere.
                let parent = get_or_throw(
                    env.join_path(&[absolute_path.to_string(), "..".to_string()], context)
                        .await,
                );
                if parent == absolute_path || !absolute_path.starts_with(&parent) {
                    return absolute_path.to_string();
                }
                let name = if parent.ends_with('/') || parent.ends_with('\\') {
                    absolute_path[parent.len()..].to_string()
                } else {
                    absolute_path[parent.len() + 1..].to_string()
                };
                get_or_throw(
                    env.join_path(&[canonical(env, &parent, context).await, name], context)
                        .await,
                )
            }
            _ => panic!("{}", error),
        },
    }
}

/// Serialize `edit` and `write` mutations of one file within this process: same file-system id
/// and canonical path, whichever environment object the call got. Other files, and other file
/// systems, never wait. Not a lock against `bash` or other processes.
pub async fn with_file_mutation_queue<T, F>(
    env: Arc<dyn ExecutionEnv>,
    path: &str,
    mutation: F,
    context: &Context,
) -> T
where
    F: Future<Output = T> + Send,
{
    let key = mutation_key(&env, path, context).await;
    // Take the slot, then release the map before awaiting, so no other call is blocked on the map.
    let mutex = {
        let mut queues = QUEUES.lock();
        let queues = queues.get_or_insert_with(HashMap::new);
        Arc::clone(queues.entry(key.clone()).or_default())
    };
    let _guard: OwnedMutexGuard<()> = Arc::clone(&mutex).lock_owned().await;
    let result = mutation.await;
    // Drop the entry when the last holder finishes, whichever holder that was.
    let mut queues = QUEUES.lock();
    if let Some(queues) = queues.as_mut() {
        if queues
            .get(&key)
            .is_some_and(|current| Arc::ptr_eq(current, &mutex))
        {
            queues.remove(&key);
        }
    }
    drop(_guard);
    result
}
