//! Port of `packages/durable/src/harness/util.ts`.
//!
//! Deviations: D1 — `Waiters` uses tokio oneshot channels per waiter keyed in
//! a `BTreeMap`; cancellation of a waiter's context removes it through the
//! chord abort listener (same settle-once semantics: `resolve`,
//! `reject_all`, or context cancellation).

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;

use pi_chord::context::{AbortReason, Context};

use crate::types::{Cursor, Page};

/// Pending waits by key. Each settles once: through `resolve`, `reject_all`,
/// or cancellation of its context.
#[derive(Clone)]
pub struct Waiters<K, T> {
    sets: Arc<parking_lot::Mutex<BTreeMap<K, Vec<WaiterSlot<T>>>>>,
}

type WaiterSlot<T> = (u64, tokio::sync::oneshot::Sender<Result<T, String>>);

impl<K: Ord + Clone + Send + Sync + 'static, T: Clone + Send + 'static> Default for Waiters<K, T> {
    fn default() -> Self {
        Self {
            sets: Arc::new(parking_lot::Mutex::new(BTreeMap::new())),
        }
    }
}

static NEXT_WAITER_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl<K: Ord + Clone + Send + Sync + 'static, T: Clone + Send + 'static> Waiters<K, T> {
    fn remove_waiter(sets: &parking_lot::Mutex<BTreeMap<K, Vec<WaiterSlot<T>>>>, key: &K, id: u64) {
        let mut sets = sets.lock();
        if let Some(set) = sets.get_mut(key) {
            set.retain(|(registered, _)| *registered != id);
            if set.is_empty() {
                sets.remove(key);
            }
        }
    }

    /// Register a waiter under `key`; rejects with the context's reason once
    /// it aborts.
    pub fn add(
        &self,
        key: K,
        context: &Context,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<T, String>> + Send + 'static>> {
        let signal = context.abort_signal();
        if let Some(signal) = signal.as_ref().filter(|signal| signal.aborted()) {
            let reason = signal.reason().unwrap_or(AbortReason::Aborted).to_string();
            return Box::pin(async move { Err(reason) });
        }
        let id = NEXT_WAITER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (sender, receiver) = tokio::sync::oneshot::channel::<Result<T, String>>();
        {
            let mut sets = self.sets.lock();
            sets.entry(key.clone()).or_default().push((id, sender));
        }
        let detach = signal.as_ref().map(|signal| {
            let sets = Arc::clone(&self.sets);
            signal.add_listener(move || Self::remove_waiter(&sets, &key, id))
        });
        Box::pin(async move {
            let reason = async {
                match &signal {
                    Some(signal) => signal.aborted_future().await,
                    None => std::future::pending::<AbortReason>().await,
                }
            };
            let result = tokio::select! {
                value = receiver => value.unwrap_or_else(|_| Err(match signal {
                    Some(signal) => abort_reason_text(&signal),
                    None => "waiter cancelled".to_string(),
                })),
                abort = reason => Err(abort.to_string()),
            };
            if let Some(detach) = detach {
                detach();
            }
            result
        })
    }

    /// Keys with pending waiters, in order.
    pub fn keys(&self) -> Vec<K> {
        self.sets.lock().keys().cloned().collect()
    }

    /// Resolve every waiter of `key` with `value` and drop the set.
    pub fn resolve(&self, key: &K, value: T) {
        let set = {
            let mut sets = self.sets.lock();
            sets.remove(key).unwrap_or_default()
        };
        for (_, sender) in set {
            let _ = sender.send(Ok(value.clone()));
        }
    }

    /// Reject every pending waiter.
    pub fn reject_all(&self, error: &str) {
        let sets = {
            let mut sets = self.sets.lock();
            std::mem::take(&mut *sets)
        };
        for set in sets.into_values() {
            for (_, sender) in set {
                let _ = sender.send(Err(error.to_string()));
            }
        }
    }
}

/// Surface an aborted context's reason as the waiter error.
fn reason_text(signal: &Option<pi_chord::context::AbortSignal>) -> String {
    signal
        .as_ref()
        .and_then(|signal| signal.reason())
        .unwrap_or(AbortReason::Aborted)
        .to_string()
}

impl<K, T> std::fmt::Debug for Waiters<K, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Waiters")
    }
}

/// Every item of a paginated scan, in page order (upstream `scanAll`).
pub async fn scan_all<T, F, Fut>(mut scan: F) -> Vec<T>
where
    F: FnMut(Option<Cursor>) -> Fut,
    Fut: Future<Output = Page<T>>,
{
    let mut items = Vec::new();
    let mut cursor: Option<Cursor> = None;
    loop {
        let page = scan(cursor).await;
        let next = page.next.clone();
        items.extend(page.items);
        cursor = next;
        if cursor.is_none() {
            return items;
        }
    }
}

/// Surface an aborted signal's reason as an error string.
pub fn abort_reason_text(signal: &pi_chord::context::AbortSignal) -> String {
    match signal.reason() {
        Some(reason) => reason.to_string(),
        None => "This operation was aborted".to_string(),
    }
}

/// `new Error("Harness is closed")`.
pub fn closed_error() -> String {
    "Harness is closed".to_string()
}

/// Flush pending scheduler microtasks: yields until queued jobs run.
pub async fn flush() {
    // Several yields: a queued tokio task may itself queue another.
    for _ in 0..16 {
        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
}
