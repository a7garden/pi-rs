//! Port of `packages/ai/src/utils/abort.ts` plus the pure-Rust
//! `AbortSignal`/`AbortController` equivalent it builds on.
//!
//! Upstream uses the DOM `AbortSignal`. Rust has no such primitive, so the
//! pi-chord port's pure-Rust re-implementation pattern is reused here
//! (`crates/pi-chord/src/context/mod.rs`): same observable semantics — parent
//! inheritance, `AbortSignal.any` combination, listener notification and
//! reason propagation — as a self-contained copy so pi-ai stays dependency-free
//! (crate docs, deviation D3).

use parking_lot::Mutex;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

/// Reason attached to an aborted [`AbortSignal`].
#[derive(Clone)]
pub enum AbortReason {
    /// `controller.abort()` without a reason — the `AbortError` case.
    Aborted,
    /// A plain string reason (`controller.abort("child")`).
    Message(String),
    /// An arbitrary error value.
    Error(SharedError),
}

/// Cheap-to-clone shared error (upstream abort reasons may be arbitrary values).
pub type SharedError = Arc<dyn std::error::Error + Send + Sync>;

impl fmt::Debug for AbortReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbortReason::Aborted => f.write_str("AbortReason::Aborted"),
            AbortReason::Message(message) => f
                .debug_tuple("AbortReason::Message")
                .field(message)
                .finish(),
            AbortReason::Error(error) => f
                .debug_tuple("AbortReason::Error")
                .field(&error.to_string())
                .finish(),
        }
    }
}

impl fmt::Display for AbortReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbortReason::Aborted => f.write_str("The operation was aborted"),
            AbortReason::Message(message) => f.write_str(message),
            AbortReason::Error(error) => f.write_str(&error.to_string()),
        }
    }
}

impl std::error::Error for AbortReason {}

impl From<&str> for AbortReason {
    fn from(value: &str) -> Self {
        AbortReason::Message(value.to_string())
    }
}

impl From<String> for AbortReason {
    fn from(value: String) -> Self {
        AbortReason::Message(value)
    }
}

impl From<SharedError> for AbortReason {
    fn from(value: SharedError) -> Self {
        AbortReason::Error(value)
    }
}

impl From<crate::http::ProviderHttpError> for AbortReason {
    fn from(value: crate::http::ProviderHttpError) -> Self {
        AbortReason::Error(Arc::new(value))
    }
}

type AbortListener = Box<dyn Fn() + Send + Sync>;

struct SignalState {
    inner: Mutex<SignalInner>,
}

struct SignalInner {
    aborted: bool,
    reason: Option<AbortReason>,
    listeners: HashMap<u64, AbortListener>,
    next_listener_id: u64,
    wakers: Vec<std::task::Waker>,
}

/// Pure-Rust equivalent of the DOM `AbortSignal`, with the same observable
/// semantics provider code relies on: `aborted()`, `reason()`, listener
/// notification and combination via [`AbortSignal::any`].
#[derive(Clone)]
pub struct AbortSignal {
    state: Arc<SignalState>,
}

impl AbortSignal {
    pub fn new() -> AbortSignal {
        AbortSignal {
            state: Arc::new(SignalState {
                inner: Mutex::new(SignalInner {
                    aborted: false,
                    reason: None,
                    listeners: HashMap::new(),
                    next_listener_id: 0,
                    wakers: Vec::new(),
                }),
            }),
        }
    }

    pub fn aborted(&self) -> bool {
        self.state.inner.lock().aborted
    }

    pub fn reason(&self) -> Option<AbortReason> {
        self.state.inner.lock().reason.clone()
    }

    /// Upstream `signal.throwIfAborted()`: `Err(reason)` when aborted.
    pub fn check(&self) -> Result<(), AbortReason> {
        let inner = self.state.inner.lock();
        match (&inner.aborted, &inner.reason) {
            (false, _) => Ok(()),
            (true, Some(reason)) => Err(reason.clone()),
            (true, None) => Err(AbortReason::Aborted),
        }
    }

    /// Register a listener invoked synchronously on abort (and immediately if
    /// already aborted). Returns a disposer that removes the registration.
    pub fn add_listener(
        &self,
        listener: impl Fn() + Send + Sync + 'static,
    ) -> Box<dyn FnOnce() + Send> {
        let listener: AbortListener = Box::new(listener);
        let id = {
            let mut inner = self.state.inner.lock();
            if inner.aborted {
                drop(inner);
                listener();
                return Box::new(|| {});
            }
            let id = inner.next_listener_id;
            inner.next_listener_id += 1;
            inner.listeners.insert(id, listener);
            id
        };
        let state = Arc::downgrade(&self.state);
        Box::new(move || {
            if let Some(state) = state.upgrade() {
                let mut inner = state.inner.lock();
                inner.listeners.remove(&id);
            }
        })
    }

    /// Future that resolves when the signal aborts (immediately when already
    /// aborted).
    pub fn aborted_future(&self) -> AbortedFuture {
        AbortedFuture {
            signal: Some(self.clone()),
        }
    }

    /// Combine several signals into one that aborts when any input aborts
    /// (`AbortSignal.any`).
    pub fn any(signals: impl IntoIterator<Item = AbortSignal>) -> AbortSignal {
        let combined = AbortSignal::new();
        for signal in signals {
            if signal.aborted() {
                let reason = signal.reason();
                AbortSignal::abort_with(&combined, reason.unwrap_or(AbortReason::Aborted));
                return combined;
            }
            let target = combined.clone();
            let watched = signal.clone();
            // The combined signal owns its listeners for its lifetime (upstream
            // never removes them either).
            let _ = signal.add_listener(move || {
                let reason = watched.reason();
                AbortSignal::abort_with(&target, reason.unwrap_or(AbortReason::Aborted));
            });
        }
        combined
    }

    /// Abort without a reason (upstream `controller.abort()`).
    pub fn abort(signal: &AbortSignal) {
        AbortSignal::abort_with(signal, AbortReason::Aborted);
    }

    /// Abort with a reason (upstream `controller.abort(reason)`).
    pub fn abort_with(signal: &AbortSignal, reason: AbortReason) {
        let wakers = {
            let mut inner = signal.state.inner.lock();
            if inner.aborted {
                return;
            }
            inner.aborted = true;
            inner.reason = Some(reason);
            let listeners: Vec<AbortListener> = std::mem::take(&mut inner.listeners)
                .into_iter()
                .map(|(_, listener)| listener)
                .collect();
            let wakers = std::mem::take(&mut inner.wakers);
            drop(inner);
            for listener in listeners {
                listener();
            }
            wakers
        };
        for waker in wakers {
            waker.wake();
        }
    }
}

impl Default for AbortSignal {
    fn default() -> Self {
        AbortSignal::new()
    }
}

impl fmt::Debug for AbortSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AbortSignal")
            .field("aborted", &self.aborted())
            .finish()
    }
}

/// Pure-Rust equivalent of the DOM `AbortController`.
#[derive(Clone)]
pub struct AbortController {
    signal: AbortSignal,
}

impl AbortController {
    pub fn new() -> AbortController {
        AbortController {
            signal: AbortSignal::new(),
        }
    }

    pub fn signal(&self) -> AbortSignal {
        self.signal.clone()
    }

    pub fn abort(&self) {
        AbortSignal::abort(&self.signal);
    }

    pub fn abort_with(&self, reason: impl Into<AbortReason>) {
        AbortSignal::abort_with(&self.signal, reason.into());
    }
}

impl Default for AbortController {
    fn default() -> Self {
        AbortController::new()
    }
}

impl fmt::Debug for AbortController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AbortController").finish()
    }
}

/// Future resolving when the signal aborts.
pub struct AbortedFuture {
    signal: Option<AbortSignal>,
}

impl Future for AbortedFuture {
    type Output = AbortReason;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<AbortReason> {
        let signal = self
            .signal
            .take()
            .expect("AbortedFuture polled after completion");
        let outcome = {
            let mut inner = signal.state.inner.lock();
            if inner.aborted {
                Some(inner.reason.clone().unwrap_or(AbortReason::Aborted))
            } else {
                inner.wakers.push(cx.waker().clone());
                None
            }
        };
        match outcome {
            Some(reason) => std::task::Poll::Ready(reason),
            None => {
                self.signal = Some(signal);
                std::task::Poll::Pending
            }
        }
    }
}

/// Upstream `operationSignal`: create an operation-local signal for public
/// APIs whose signal is optional.
pub fn operation_signal(signal: Option<&AbortSignal>) -> AbortSignal {
    signal
        .cloned()
        .unwrap_or_else(|| AbortController::new().signal())
}

/// Upstream `raceWithAbortSignal`: stop waiting for an operation when its
/// signal aborts. The abandoned future is simply dropped (Rust has no
/// unhandled-rejection concern).
pub async fn race_with_abort_signal<T, F: Future<Output = T>>(
    operation: F,
    signal: AbortSignal,
) -> Result<T, AbortReason> {
    tokio::select! {
        biased;
        reason = signal.aborted_future() => Err(reason),
        value = operation => Ok(value),
    }
}

#[cfg(test)]
mod tests {
    //! Port of `packages/ai/test/abort.test.ts` — only the pure, offline-verifiable
    //! semantics (abort reasons, `operationSignal`, `raceWithAbortSignal`, and the
    //! listener/`any` behavior `abort.ts` builds on). The upstream file is otherwise
    //! live-provider end-to-end (openai-codex OAuth etc.) and is covered by the
    //! provider slices; see the port report for the skipped cases.

    use super::*;

    #[tokio::test]
    async fn races_an_operation_to_completion_when_not_aborted() {
        let controller = AbortController::new();
        let value = race_with_abort_signal(async { 42 }, controller.signal())
            .await
            .expect("operation completes");
        assert_eq!(value, 42);
    }

    #[tokio::test]
    async fn rejects_immediately_when_the_signal_is_already_aborted() {
        // `testImmediateAbort`: a pre-aborted signal must surface as an abort, not
        // run the operation.
        let controller = AbortController::new();
        controller.abort();
        let reason = race_with_abort_signal(async { 42 }, controller.signal())
            .await
            .expect_err("already aborted");
        assert!(matches!(reason, AbortReason::Aborted));
        assert_eq!(reason.to_string(), "The operation was aborted");
    }

    #[tokio::test(start_paused = true)]
    async fn rejects_with_the_abort_reason_when_aborted_mid_operation() {
        let controller = AbortController::new();
        let signal = controller.signal();
        {
            let controller = controller.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                controller.abort();
            });
        }
        let reason = race_with_abort_signal(
            tokio::time::sleep(std::time::Duration::from_secs(60)),
            signal.clone(),
        )
        .await
        .expect_err("aborted mid operation");
        assert!(matches!(reason, AbortReason::Aborted));
        assert_eq!(reason.to_string(), "The operation was aborted");

        // A string reason is propagated (`abortReason(signal)`).
        let controller = AbortController::new();
        controller.abort_with("child");
        let reason = race_with_abort_signal(std::future::pending::<()>(), controller.signal())
            .await
            .expect_err("aborted");
        assert_eq!(reason.to_string(), "child");
    }

    #[tokio::test]
    async fn operation_signal_returns_the_given_signal_or_a_fresh_one() {
        let controller = AbortController::new();
        let signal = controller.signal();
        assert!(!operation_signal(Some(&signal)).aborted());
        let fresh = operation_signal(None);
        assert!(!fresh.aborted());
        // The fresh signal is operation-local: aborting it does not touch the caller's.
        AbortSignal::abort(&fresh);
        assert!(fresh.aborted());
        assert!(!signal.aborted());
    }

    #[test]
    fn listeners_fire_on_abort_and_immediately_when_already_aborted() {
        let controller = AbortController::new();
        let signal = controller.signal();
        let fired = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = fired.clone();
        let disposer = signal.add_listener(move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });
        assert_eq!(fired.load(std::sync::atomic::Ordering::SeqCst), 0);
        controller.abort();
        assert_eq!(fired.load(std::sync::atomic::Ordering::SeqCst), 1);
        drop(disposer);

        // Already-aborted signals invoke new listeners synchronously.
        let counter = fired.clone();
        let _disposer = signal.add_listener(move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });
        assert_eq!(fired.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn any_combines_signals_and_propagates_reasons() {
        let parent = AbortController::new();
        let combined = AbortSignal::any([parent.signal(), AbortController::new().signal()]);
        assert!(!combined.aborted());
        parent.abort_with("stop");
        assert!(combined.aborted());
        assert_eq!(
            combined.reason().map(|reason| reason.to_string()),
            Some("stop".to_string())
        );
    }
}
