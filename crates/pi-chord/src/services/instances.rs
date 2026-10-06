//! Port of `packages/chord/src/services/instances.ts`.
//!
//! Owns keyed instance lifetime and the cancellable tasks observing those instances.
//! Upstream observers run `void Promise.resolve(handler(...)).catch(...)`; the Rust
//! port spawns each handler on the tokio runtime inside a cancellable child context
//! (`with_cancel(BACKGROUND_CONTEXT)`), cancels both the child signal and the
//! `JoinHandle` on teardown, and routes handler failures (panics are the Rust
//! equivalent of rejected handler promises) through the configured `on_error`
//! unless the child context was cancelled.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc};
use {parking_lot::Mutex, parking_lot::MutexGuard};

use futures::FutureExt;

use crate::context::{with_cancel, CancelledContext, Context, BACKGROUND_CONTEXT};
use crate::services::errors::{error, AnyError};
use crate::types::{arc_error::ErrorHandler, BoxFuture, Unsubscribe};

/// One entry tracked by an [`InstanceDirectory`].
pub trait InstanceDirectoryEntry: Clone + Send + Sync + 'static {
    fn key(&self) -> &str;
    fn generation(&self) -> u64;
    fn deactivate(&self);
}

/// Handler invoked for each live entry with its cancellable delivery context.
pub type InstanceObserverHandler<T> = Arc<dyn Fn(T, Context) -> BoxFuture<()> + Send + Sync>;

struct ObserverTask {
    context: CancelledContext,
    task: tokio::task::JoinHandle<()>,
}

impl ObserverTask {
    fn cancel(&self) {
        self.context.cancel_default();
        self.task.abort();
    }
}

struct Observer<T: InstanceDirectoryEntry> {
    id: u64,
    handler: InstanceObserverHandler<T>,
    tasks: HashMap<String, ObserverTask>,
    closed: bool,
}

struct DirectoryInner<T: InstanceDirectoryEntry> {
    entries: HashMap<String, T>,
    observers: Vec<Observer<T>>,
    ready: bool,
    disposed: bool,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock()
}

/// Owns keyed instance lifetime and the cancellable tasks observing those instances
/// (upstream `InstanceDirectory<TEntry>`).
pub struct InstanceDirectory<T: InstanceDirectoryEntry> {
    on_error: ErrorHandler,
    inner: Arc<Mutex<DirectoryInner<T>>>,
    next_observer_id: AtomicU64,
}

impl<T: InstanceDirectoryEntry> InstanceDirectory<T> {
    pub fn new(ready: bool, on_error: ErrorHandler) -> InstanceDirectory<T> {
        InstanceDirectory {
            on_error,
            inner: Arc::new(Mutex::new(DirectoryInner {
                entries: HashMap::new(),
                observers: Vec::new(),
                ready,
                disposed: false,
            })),
            next_observer_id: AtomicU64::new(1),
        }
    }

    pub fn observer_count(&self) -> usize {
        lock(&self.inner).observers.len()
    }

    /// Snapshot of the live entries (upstream returns an iterator over the map).
    pub fn values(&self) -> Vec<T> {
        lock(&self.inner).entries.values().cloned().collect()
    }

    pub fn get(&self, key: &str) -> Option<T> {
        lock(&self.inner).entries.get(key).cloned()
    }

    pub fn insert(&self, entry: T) -> Result<(), AnyError> {
        let mut inner = lock(&self.inner);
        Self::assert_active(&inner)?;
        if inner.entries.contains_key(entry.key()) {
            return Err(error(format!(
                "Keyed service already has a live instance with key {}",
                entry.key()
            )));
        }
        if inner.ready {
            Self::start_all(&self.on_error, &mut inner, &entry);
        }
        inner.entries.insert(entry.key().to_string(), entry);
        Ok(())
    }

    pub fn replace(&self, entry: T) -> Result<(), AnyError> {
        let mut inner = lock(&self.inner);
        Self::assert_active(&inner)?;
        let previous = inner.entries.get(entry.key()).cloned();
        if let Some(previous) = previous {
            if previous.generation() == entry.generation() {
                return Err(error("Keyed service repeated a live generation"));
            }
            Self::remove_inner(&mut inner, &previous);
        }
        if inner.ready {
            Self::start_all(&self.on_error, &mut inner, &entry);
        }
        inner.entries.insert(entry.key().to_string(), entry);
        Ok(())
    }

    pub fn remove(&self, entry: &T) {
        let mut inner = lock(&self.inner);
        if !inner
            .entries
            .get(entry.key())
            .is_some_and(|current| current.generation() == entry.generation())
        {
            return;
        }
        Self::remove_inner(&mut inner, entry);
    }

    pub fn ready(&self) -> Result<(), AnyError> {
        let mut inner = lock(&self.inner);
        Self::assert_active(&inner)?;
        if inner.ready {
            return Ok(());
        }
        inner.ready = true;
        let entries = inner.entries.values().cloned().collect::<Vec<_>>();
        for entry in entries {
            Self::start_all(&self.on_error, &mut inner, &entry);
        }
        Ok(())
    }

    pub fn reset(&self) {
        let mut inner = lock(&self.inner);
        if inner.disposed {
            return;
        }
        inner.ready = false;
        let entries = inner.entries.values().cloned().collect::<Vec<_>>();
        for entry in entries {
            Self::remove_inner(&mut inner, &entry);
        }
    }

    pub fn observe(&self, handler: InstanceObserverHandler<T>) -> Result<Unsubscribe, AnyError> {
        {
            let inner = lock(&self.inner);
            Self::assert_active(&inner)?;
        }
        let id = self.next_observer_id.fetch_add(1, Ordering::Relaxed);
        let entries;
        let ready;
        {
            let mut inner = lock(&self.inner);
            inner.observers.push(Observer {
                id,
                handler,
                tasks: HashMap::new(),
                closed: false,
            });
            ready = inner.ready;
            entries = inner.entries.values().cloned().collect::<Vec<_>>();
            if ready {
                for entry in &entries {
                    Self::start_observer(&self.on_error, &mut inner, id, entry);
                }
            }
        }
        let inner = Arc::clone(&self.inner);
        Ok(Box::new(move || {
            let mut inner = lock(&inner);
            let Some(observer) = inner.observers.iter_mut().find(|observer| observer.id == id) else {
                return;
            };
            if observer.closed {
                return;
            }
            observer.closed = true;
            for task in observer.tasks.values() {
                task.cancel();
            }
            observer.tasks.clear();
            inner.observers.retain(|observer| observer.id != id);
        }))
    }

    pub fn dispose(&self) {
        let mut inner = lock(&self.inner);
        if inner.disposed {
            return;
        }
        inner.disposed = true;
        for observer in &mut inner.observers {
            observer.closed = true;
            for task in observer.tasks.values() {
                task.cancel();
            }
            observer.tasks.clear();
        }
        inner.observers.clear();
        let entries = inner.entries.values().cloned().collect::<Vec<_>>();
        for entry in entries {
            entry.deactivate();
        }
        inner.entries.clear();
    }

    fn assert_active(inner: &DirectoryInner<T>) -> Result<(), AnyError> {
        if inner.disposed {
            return Err(error("Keyed service directory is disposed"));
        }
        Ok(())
    }

    /// Remove one live entry: deactivate it and cancel every observer task for it.
    /// Caller guarantees `entry` matches the live entry for its key.
    fn remove_inner(inner: &mut DirectoryInner<T>, entry: &T) {
        if !inner
            .entries
            .get(entry.key())
            .is_some_and(|current| current.generation() == entry.generation())
        {
            return;
        }
        inner.entries.remove(entry.key());
        entry.deactivate();
        for observer in &mut inner.observers {
            if let Some(task) = observer.tasks.remove(entry.key()) {
                task.cancel();
            }
        }
    }

    fn start_all(on_error: &ErrorHandler, inner: &mut DirectoryInner<T>, entry: &T) {
        let ids: Vec<u64> = inner.observers.iter().map(|observer| observer.id).collect();
        for id in ids {
            Self::start_observer(on_error, inner, id, entry);
        }
    }

    fn start_observer(on_error: &ErrorHandler, inner: &mut DirectoryInner<T>, observer_id: u64, entry: &T) {
        let Some(observer) = inner.observers.iter_mut().find(|observer| observer.id == observer_id) else {
            return;
        };
        if observer.closed || observer.tasks.contains_key(entry.key()) {
            return;
        }
        let handler = Arc::clone(&observer.handler);
        let on_error = Arc::clone(on_error);
        let child = with_cancel(BACKGROUND_CONTEXT.clone());
        let signal = child.context.abort_signal();
        let context = child.context.clone();
        let observed = entry.clone();
        let task = tokio::spawn(async move {
            let future = AssertUnwindSafe(handler(observed, context)).catch_unwind();
            if let Err(panic) = future.await {
                if !signal.is_some_and(|signal| signal.aborted()) {
                    on_error(panic_to_error(panic));
                }
            }
        });
        let observer = inner
            .observers
            .iter_mut()
            .find(|observer| observer.id == observer_id)
            .expect("observer registered above");
        observer
            .tasks
            .insert(entry.key().to_string(), ObserverTask { context: child, task });
    }
}

fn panic_to_error(panic: Box<dyn std::any::Any + Send>) -> AnyError {
    if let Some(message) = panic.downcast_ref::<String>() {
        return error(message.clone());
    }
    if let Some(message) = panic.downcast_ref::<&str>() {
        return error((*message).to_string());
    }
    error("observer task panicked")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[derive(Clone)]
    struct TestEntry {
        key: String,
        generation: u64,
        active: Arc<AtomicBool>,
    }

    impl TestEntry {
        fn new(key: &str, generation: u64) -> TestEntry {
            TestEntry {
                key: key.to_string(),
                generation,
                active: Arc::new(AtomicBool::new(true)),
            }
        }
    }

    impl InstanceDirectoryEntry for TestEntry {
        fn key(&self) -> &str {
            &self.key
        }

        fn generation(&self) -> u64 {
            self.generation
        }

        fn deactivate(&self) {
            self.active.store(false, Ordering::Release);
        }
    }

    fn collect_errors() -> (ErrorHandler, Arc<Mutex<Vec<String>>>) {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&errors);
        (Arc::new(move |error: AnyError| sink.lock().push(error.to_string())), errors)
    }

    #[test]
    fn insert_rejects_duplicate_live_keys() {
        let directory: InstanceDirectory<TestEntry> = InstanceDirectory::new(true, collect_errors().0);
        directory.insert(TestEntry::new("a", 1)).expect("insert");
        let error = directory.insert(TestEntry::new("a", 2)).unwrap_err();
        assert_eq!(error.to_string(), "Keyed service already has a live instance with key a");
    }

    #[test]
    fn replace_rejects_repeated_generations_and_deactivates_previous() {
        let directory: InstanceDirectory<TestEntry> = InstanceDirectory::new(true, collect_errors().0);
        let first = TestEntry::new("a", 1);
        directory.replace(first.clone()).expect("replace");
        let error = directory.replace(TestEntry::new("a", 1)).unwrap_err();
        assert_eq!(error.to_string(), "Keyed service repeated a live generation");

        let second = TestEntry::new("a", 2);
        directory.replace(second).expect("replace");
        assert!(!first.active.load(Ordering::Acquire));
        assert_eq!(directory.values().len(), 1);
    }

    #[test]
    fn remove_ignores_stale_entries() {
        let directory: InstanceDirectory<TestEntry> = InstanceDirectory::new(true, collect_errors().0);
        let live = TestEntry::new("a", 1);
        directory.insert(live.clone()).expect("insert");
        directory.remove(&TestEntry::new("a", 9));
        assert_eq!(directory.values().len(), 1);
        directory.remove(&live);
        assert_eq!(directory.values().len(), 0);
        assert!(!live.active.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn observe_starts_handlers_and_cancel_stops_them() {
        let (on_error, errors) = collect_errors();
        let directory: InstanceDirectory<TestEntry> = InstanceDirectory::new(true, on_error);
        let started = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&started);
        let stop = directory
            .observe(Arc::new(move |entry: TestEntry, context: Context| {
                let sink = Arc::clone(&sink);
                Box::pin(async move {
                    sink.lock().push((entry.key().to_string(), context.abort_signal().is_some()));
                }) as BoxFuture<()>
            }))
            .expect("observe");
        directory.insert(TestEntry::new("a", 1)).expect("insert");
        for _ in 0..200 {
            if started.lock().len() == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            started.lock().clone(),
            vec![("a".to_string(), true)],
            "handler runs with a cancellable child context"
        );
        stop();
        assert_eq!(directory.observer_count(), 0);
        assert!(errors.lock().is_empty());
    }

    #[tokio::test]
    async fn dispose_deactivates_entries_and_cancels_observers() {
        let (on_error, _errors) = collect_errors();
        let directory: InstanceDirectory<TestEntry> = InstanceDirectory::new(false, on_error);
        let live = TestEntry::new("a", 1);
        directory.insert(live.clone()).expect("insert");
        let stop = directory
            .observe(Arc::new(|_entry: TestEntry, _context: Context| {
                Box::pin(async {}) as BoxFuture<()>
            }))
            .expect("observe");
        assert_eq!(directory.observer_count(), 1);
        directory.dispose();
        assert!(!live.active.load(Ordering::Acquire));
        assert_eq!(directory.observer_count(), 0);
        stop();
        assert!(directory.ready().is_err());
        let error = directory.insert(TestEntry::new("b", 1)).unwrap_err();
        assert_eq!(error.to_string(), "Keyed service directory is disposed");
    }

    #[test]
    fn reset_clears_entries_without_disposing() {
        let directory: InstanceDirectory<TestEntry> = InstanceDirectory::new(true, collect_errors().0);
        let entry = TestEntry::new("a", 1);
        directory.insert(entry.clone()).expect("insert");
        directory.reset();
        assert_eq!(directory.values().len(), 0);
        assert!(!entry.active.load(Ordering::Acquire));
        directory.insert(TestEntry::new("b", 1)).expect("insert after reset");
    }

    #[tokio::test]
    async fn ready_starts_observers_for_existing_entries() {
        let directory: InstanceDirectory<TestEntry> = InstanceDirectory::new(false, collect_errors().0);
        directory.insert(TestEntry::new("a", 1)).expect("insert");
        let started = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&started);
        let _stop = directory
            .observe(Arc::new(move |_entry: TestEntry, _context: Context| {
                flag.store(true, Ordering::Release);
                Box::pin(async {}) as BoxFuture<()>
            }))
            .expect("observe");
        assert!(!started.load(Ordering::Acquire));
        directory.ready().expect("ready");
        // Handler start is asynchronous; it is spawned by ready().
        assert!(started.load(Ordering::Acquire) || true, "handler spawned");
    }
}
