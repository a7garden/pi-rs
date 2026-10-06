//! Port of `packages/chord/src/context/index.ts`.
//!
//! Immutable invocation-scoped values passed explicitly through operations.
//!
//! Deviations from upstream (mechanical, recorded per PORT_MAP.md rule 3):
//! - `AbortSignal`/`AbortController` are re-implemented as pure-Rust equivalents
//!   (`AbortSignal`, `AbortController` below) with the same observable semantics:
//!   parent inheritance, `AbortSignal.any` combination, listener notification and
//!   reason propagation.
//! - Context values are type-erased (`Arc<dyn Any>`); `Context::value` clones the
//!   stored value out (values must be `Clone`), matching upstream's immutable
//!   value semantics.

use std::any::Any;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use {parking_lot::Mutex};
use std::task::{Context as TaskContext, Poll, Waker};

use crate::services::errors::AnyError;

/// An owned error handed across chord boundaries (upstream `Error`/`AggregateError`).
pub type BoxError = AnyError;

/// Reason attached to an aborted [`AbortSignal`].
#[derive(Clone)]
pub enum AbortReason {
    /// `controller.abort()` without a reason — the `AbortError` DOMException case.
    Aborted,
    /// A plain string reason (`controller.abort("child")`).
    Message(String),
    /// An arbitrary error value.
    Error(SharedError),
}

/// Cheap-to-clone shared error (upstream abort reasons may be arbitrary `Error`s).
pub type SharedError = Arc<dyn std::error::Error + Send + Sync>;

impl fmt::Debug for AbortReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbortReason::Aborted => f.write_str("AbortReason::Aborted"),
            AbortReason::Message(message) => f.debug_tuple("AbortReason::Message").field(message).finish(),
            AbortReason::Error(error) => f.debug_tuple("AbortReason::Error").field(&error.to_string()).finish(),
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

impl From<BoxError> for AbortReason {
    fn from(value: BoxError) -> Self {
        AbortReason::Error(Arc::from(value))
    }
}

type AbortListener = Box<dyn Fn() + Send + Sync>;

struct SignalState {
    inner: Mutex<SignalInner>,
}

struct SignalInner {
    aborted: bool,
    reason: Option<AbortReason>,
    listeners: Vec<(u64, AbortListener)>,
    next_listener_id: u64,
    wakers: Vec<Waker>,
}

/// Pure-Rust equivalent of the DOM `AbortSignal`, with the same observable
/// semantics chord relies on: `aborted()`, `reason()`, listener notification and
/// combination via [`AbortSignal::any`].
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
                    listeners: Vec::new(),
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

    /// Register a listener invoked synchronously on abort (and immediately if
    /// already aborted). Returns a disposer that removes the registration.
    pub fn add_listener(&self, listener: impl Fn() + Send + Sync + 'static) -> Box<dyn FnOnce() + Send> {
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
            inner.listeners.push((id, listener));
            id
        };
        let state = Arc::downgrade(&self.state);
        Box::new(move || {
            if let Some(state) = state.upgrade() {
                let mut inner = state.inner.lock();
                inner.listeners.retain(|(registered, _)| *registered != id);
            }
        })
    }

    /// Future that resolves when the signal aborts.
    pub fn aborted_future(&self) -> AbortedFuture {
        AbortedFuture {
            signal: self.clone(),
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
            let forward = combined.clone();
            let captured = signal.clone();
            let _listener = signal.add_listener(move || {
                let reason = captured.reason();
                AbortSignal::abort_with(&forward, reason.unwrap_or(AbortReason::Aborted));
            });
            // The forwarder is intentionally kept alive by the signals it listens to;
            // listeners die with their signal state.
        }
        combined
    }

    pub(crate) fn abort_with(&self, reason: AbortReason) {
        let listeners = {
            let mut inner = self.state.inner.lock();
            if inner.aborted {
                return;
            }
            inner.aborted = true;
            inner.reason = Some(reason);
            std::mem::take(&mut inner.listeners)
        };
        for waker in self.state.inner.lock().wakers.drain(..) {
            waker.wake();
        }
        for listener in listeners {
            listener.1();
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
        let inner = self.state.inner.lock();
        f.debug_struct("AbortSignal")
            .field("aborted", &inner.aborted)
            .field("reason", &inner.reason)
            .finish()
    }
}

/// Future resolving once the backing signal aborts.
pub struct AbortedFuture {
    signal: AbortSignal,
}

impl Future for AbortedFuture {
    type Output = AbortReason;

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        let mut inner = self.signal.state.inner.lock();
        if inner.aborted {
            return Poll::Ready(inner.reason.clone().unwrap_or(AbortReason::Aborted));
        }
        if !inner.wakers.iter().any(|waker| waker.will_wake(cx.waker())) {
            inner.wakers.push(cx.waker().clone());
        }
        Poll::Pending
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

    /// Abort with a reason; use [`AbortReason::Aborted`] for the no-reason case.
    pub fn abort(&self, reason: impl Into<AbortReason>) {
        self.signal.abort_with(reason.into());
    }

    pub fn abort_default(&self) {
        self.signal.abort_with(AbortReason::Aborted);
    }
}

impl Default for AbortController {
    fn default() -> Self {
        AbortController::new()
    }
}

/// Typed identity for one value carried by a [`Context`].
///
/// Keys compare by creation identity (upstream: unique symbol token).
pub struct ContextKey<T> {
    id: u64,
    description: Arc<str>,
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T> Clone for ContextKey<T> {
    fn clone(&self) -> Self {
        ContextKey {
            id: self.id,
            description: self.description.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> fmt::Debug for ContextKey<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextKey").field("description", &self.description).finish()
    }
}

static NEXT_CONTEXT_KEY_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub fn create_context_key<T>(description: &str) -> ContextKey<T> {
    ContextKey {
        id: NEXT_CONTEXT_KEY_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        description: Arc::from(description),
        _marker: std::marker::PhantomData,
    }
}

impl<T> ContextKey<T> {
    pub fn description(&self) -> &str {
        &self.description
    }

    fn id(&self) -> u64 {
        self.id
    }
}

struct ContextValueNode {
    parent: Context,
    key_id: u64,
    key_description: Arc<str>,
    value: Arc<dyn Any + Send + Sync>,
}

enum ContextNode {
    Empty { name: &'static str },
    Value(Box<ContextValueNode>),
}

/// Immutable invocation-scoped values passed explicitly through operations.
///
/// Cheap to clone; lookups walk the parent chain.
#[derive(Clone)]
pub struct Context {
    node: Arc<ContextNode>,
}

impl Context {
    fn empty(name: &'static str) -> Context {
        Context {
            node: Arc::new(ContextNode::Empty { name }),
        }
    }

    /// Read a value stored under `key`, cloning it out.
    pub fn value<T: Clone + Send + Sync + 'static>(&self, key: &ContextKey<T>) -> Option<T> {
        self.value_ref(key).cloned()
    }

    /// Borrow a value stored under `key` without cloning.
    pub fn value_ref<T: Send + Sync + 'static>(&self, key: &ContextKey<T>) -> Option<&T> {
        let mut node = self.node.as_ref();
        loop {
            match node {
                ContextNode::Empty { .. } => return None,
                ContextNode::Value(value_node) => {
                    if value_node.key_id == key.id() {
                        return value_node.value.downcast_ref::<T>();
                    }
                    node = value_node.parent.node.as_ref();
                }
            }
        }
    }

    /// The caller cancellation signal, if one was installed.
    pub fn abort_signal(&self) -> Option<AbortSignal> {
        let mut node = self.node.as_ref();
        loop {
            match node {
                ContextNode::Empty { .. } => return None,
                ContextNode::Value(value_node) => {
                    if value_node.key_id == ABORT_SIGNAL_CONTEXT_KEY_ID {
                        let stored = value_node.value.downcast_ref::<Option<AbortSignal>>();
                        return stored.and_then(|stored| stored.clone());
                    }
                    node = value_node.parent.node.as_ref();
                }
            }
        }
    }
}

impl fmt::Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.node.as_ref() {
            ContextNode::Empty { name } => f.write_str(name),
            ContextNode::Value(value_node) => {
                write!(f, "{}.WithValue({})", value_node.parent, value_node.key_description)
            }
        }
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Internal key used to carry the abort signal through the value chain (upstream
/// keeps a frozen `ContextKey<AbortSignal | undefined>` with a unique symbol).
const ABORT_SIGNAL_CONTEXT_KEY_ID: u64 = u64::MAX;

fn context_value<T: Send + Sync + 'static>(key: &ContextKey<T>, value: T, parent: Context) -> Context {
    Context {
        node: Arc::new(ContextNode::Value(Box::new(ContextValueNode {
            parent,
            key_id: key.id(),
            key_description: key.description.clone(),
            value: Arc::new(value),
        }))),
    }
}

pub static BACKGROUND_CONTEXT: LazyLock<Context> = LazyLock::new(|| Context::empty("[Context BACKGROUND_CONTEXT]"));
pub static TODO_CONTEXT: LazyLock<Context> = LazyLock::new(|| Context::empty("[Context TODO_CONTEXT]"));

/// Derive a context containing one additional or replaced value.
pub fn with_context_value<T: Send + Sync + 'static>(key: &ContextKey<T>, value: T, parent: Context) -> Context {
    context_value(key, value, parent)
}

/// Derive a context cancelled by either the parent signal or the supplied signal.
/// The parent context remains unchanged.
pub fn with_abort_signal(signal: AbortSignal, context: Context) -> Context {
    let combined = match context.abort_signal() {
        Some(parent) => AbortSignal::any([parent, signal]),
        None => signal,
    };
    with_abort_signal_value(Some(combined), context)
}

/// Derive a context retaining all values except caller cancellation. Intended for
/// mandatory cleanup only.
pub fn without_abort_signal(context: Context) -> Context {
    with_abort_signal_value(None, context)
}

fn with_abort_signal_value(signal: Option<AbortSignal>, parent: Context) -> Context {
    // Store under the reserved abort key. The key's type parameter is erased at
    // runtime; the value is `Option<AbortSignal>`.
    Context {
        node: Arc::new(ContextNode::Value(Box::new(ContextValueNode {
            parent,
            key_id: ABORT_SIGNAL_CONTEXT_KEY_ID,
            key_description: Arc::from("chord.abortSignal"),
            value: Arc::new(signal),
        }))),
    }
}

/// Cancellable child context handle (upstream `withCancel`).
pub struct CancelledContext {
    pub context: Context,
    controller: AbortController,
}

impl CancelledContext {
    /// Cancel the child context without touching the parent.
    pub fn cancel(&self, reason: impl Into<AbortReason>) {
        self.controller.abort(reason);
    }

    pub fn cancel_default(&self) {
        self.controller.abort_default();
    }
}

/// Derive an independently cancellable child context.
pub fn with_cancel(parent: Context) -> CancelledContext {
    let controller = AbortController::new();
    CancelledContext {
        context: with_abort_signal(controller.signal(), parent),
        controller,
    }
}

/// Error returned by [`await_with_context`] when the invocation is cancelled.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct AbortedError {
    message: String,
}

/// Upstream `abortError(signal)`: surface the abort reason as an error.
pub fn abort_error(signal: &AbortSignal) -> BoxError {
    match signal.reason() {
        Some(AbortReason::Error(error)) => Box::new(SharedErrorWrapper(error)),
        Some(AbortReason::Message(message)) => Box::new(AbortedError { message }),
        Some(AbortReason::Aborted) | None => Box::new(AbortedError {
            message: "The operation was aborted".to_string(),
        }),
    }
}

/// Adapter presenting a shared error as an owned `BoxError`.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct SharedErrorWrapper(SharedError);

/// Observe a future until it settles or the invocation is cancelled.
/// Cancellation rejects only this waiter; it does not cancel the underlying work.
pub async fn await_with_context<T, F>(work: F, context: &Context) -> Result<T, BoxError>
where
    F: Future<Output = T>,
{
    let signal = context.abort_signal();
    let Some(signal) = signal else {
        return Ok(work.await);
    };
    if signal.aborted() {
        return Err(abort_error(&signal));
    }
    let abort = signal.aborted_future();
    tokio::select! {
        value = work => Ok(value),
        _reason = abort => Err(abort_error(&signal)),
    }
}
