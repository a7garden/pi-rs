//! Port of `packages/durable/src/session/observation.ts`.
//!
//! Session-to-Chord observation of committed values:
//! - [`CommittedStateSource`] ports `CommittedStateSource`/
//!   `SessionSourceAttachment` on top of pi-chord's `ReplicatedStateSource`
//!   contract (`documentState()`, and the harness conversation views).
//! - [`CommittedWatch`] ports `CommittedWatch` on top of the erased
//!   [`crate::types::WatchHandle`]: a per-watch pump task delivers exact
//!   committed frames through the handle's pending-delivery boundary.
//! - [`retirement_operations`] / [`MAX_PENDING_WATCH_FRAMES`] /
//!   [`observed_operations`] port the retirement batch, the overflow bound,
//!   and the observer-version upgrade rule.
//!
//! Deviations (D1): the observed value is generic ([`ObservedValue`]); the
//! default [`ObservedDocumentValue`] retires on `None`, view values never
//! retire through frames (upstream relies on `null` for the default). D2: the
//! Rust listener is an async `Fn` returning `()`; a listener failure is
//! signaled by panicking, and the pump converts the panic into the
//! `ListenerError` watch end with the panic message (upstream awaits a
//! throwing listener and snapshots `error.message` into `WatchEnd`).

use std::collections::HashMap;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Weak;

use parking_lot::Mutex;
use pi_chord::context::{without_abort_signal, AbortSignal, Context};
use pi_chord::delta::Op;
use pi_chord::types::{
    ReplicatedStateSource, ReplicatedStateSourceAttachment, ReplicatedStateSourceFrame,
    ReplicatedStateSourceSnapshot, SourceFrameListener,
};
use serde_json::Value;

use crate::harness::view::ConversationView;
use crate::types::{JsonObject, WatchEnd, WatchHandle};

/// Observed document value: `Some(value)` while live, `None` for the retired
/// `null` revision (upstream `ObservedDocumentValue`).
pub type ObservedDocumentValue = Option<Arc<JsonObject>>;

/// Serialization and retirement marker of one observed value kind (D1).
pub trait ObservedValue: Clone + Send + Sync + 'static {
    /// The exact immutable revision delivered to Chord frames.
    fn to_frame_json(&self) -> Value;

    /// Whether this revision retires the source (upstream `null`).
    fn is_retired(&self) -> bool;
}

impl ObservedValue for ObservedDocumentValue {
    fn to_frame_json(&self) -> Value {
        match self {
            Some(object) => Value::Object(object.as_ref().clone()),
            None => Value::Null,
        }
    }

    fn is_retired(&self) -> bool {
        self.is_none()
    }
}

// `ConversationView: ObservedValue` is implemented by the harness (`view.rs`),
// which owns the type.

/// Maximum exact committed frames retained behind one unavailable watch
/// listener.
pub(crate) const MAX_PENDING_WATCH_FRAMES: usize = 100;

/// Canonical terminal update for a retired document incarnation
/// (upstream `RETIREMENT_OPERATIONS`).
pub fn retirement_operations() -> Vec<Op> {
    vec![Op::Replace(Value::Null)]
}

/// Serialized frame of one committed revision (upstream `WatchFrame`).
#[derive(Clone)]
struct WatchFrame<T> {
    value: T,
    ops: Vec<Op>,
    context: Context,
}

/// Asynchronous sole listener of one watch (upstream
/// `(value, ops, context) => Promise<void>`).
pub type WatchListener<T> =
    Arc<dyn Fn(T, Vec<Op>, Context) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Serialized exact-frame watch bound to one committed incarnation (upstream
/// `CommittedWatch`); the default retires on the `null` revision.
///
/// Frames are queued per commit and delivered one at a time by a pump task
/// through the shared [`WatchHandle`]; once the queue overflows, it collapses
/// into a single root replacement of the newest value.
#[derive(Clone)]
pub struct CommittedWatch<T = ObservedDocumentValue> {
    inner: Arc<WatchShared<T>>,
}

struct WatchShared<T> {
    handle: WatchHandle<T>,
    state: Mutex<WatchState<T>>,
    /// Wakes the pump when frames arrive, `start` runs, or the watch terminates.
    wake: tokio::sync::Notify,
}

struct WatchState<T> {
    pending: VecDeque<WatchFrame<T>>,
    started: bool,
    retired: bool,
    end: Option<WatchEnd>,
    detach: Option<Box<dyn FnOnce() + Send>>,
    cancellation: Option<(AbortSignal, Box<dyn FnOnce() + Send>)>,
    listener: Option<WatchListener<T>>,
}

impl<T: ObservedValue> CommittedWatch<T> {
    /// Create a watch observing `value` as of acquisition; `detach`
    /// unsubscribes it from the Session once it terminates.
    pub fn new(value: T, detach: Box<dyn FnOnce() + Send>) -> Self {
        let inner = Arc::new(WatchShared {
            handle: WatchHandle::new(value),
            state: Mutex::new(WatchState {
                pending: VecDeque::new(),
                started: false,
                retired: false,
                end: None,
                detach: Some(detach),
                cancellation: None,
                listener: None,
            }),
            wake: tokio::sync::Notify::new(),
        });
        tokio::spawn(pump(inner.clone()));
        Self { inner }
    }

    /// Acquisition revision before start; latest delivered immutable revision
    /// afterward.
    pub fn value(&self) -> T {
        self.inner.handle.value()
    }

    /// Install the sole asynchronous listener. Never invokes it inline.
    pub fn start(&self, listener: WatchListener<T>) -> Result<(), WatchError> {
        let mut state = self.inner.state.lock();
        if state.end.is_some() {
            return Err(WatchError("Watch is stopped".to_string()));
        }
        if state.started {
            return Err(WatchError("Watch is already started".to_string()));
        }
        state.started = true;
        state.listener = Some(listener.clone());
        drop(state);
        // The handle's listener erases the borrow; frames carry their own context.
        self.inner.handle.start(Arc::new(
            move |value: T, ops: Vec<Op>, context: &Context| {
                let listener = listener.clone();
                let context = context.clone();
                Box::pin(async move { listener(value, ops, context).await })
            },
        ));
        self.inner.wake.notify_waiters();
        Ok(())
    }

    /// Idempotently stop future callbacks and return this watch's terminal
    /// result.
    pub async fn stop(&self) -> WatchEnd {
        self.terminate(WatchEnd::Stopped);
        self.closed().await
    }

    /// Synchronous prefix of [`Self::stop`]: upstream `stop()` terminates
    /// synchronously and only awaits the closed promise; callers that must
    /// not yield (an invocation end that stops watches before aborting its
    /// signal) use this.
    pub fn stop_detached(&self) {
        self.terminate(WatchEnd::Stopped);
    }

    /// Settle when the watch terminates; an already-running callback remains
    /// caller-owned.
    pub async fn closed(&self) -> WatchEnd {
        self.inner.handle.closed().await
    }

    /// Observe one caller cancellation signal (upstream `observeCancellation`).
    pub fn observe_cancellation(&self, signal: &AbortSignal) -> Result<(), WatchError> {
        let mut state = self.inner.state.lock();
        if state.end.is_some() {
            return Ok(());
        }
        if state.cancellation.is_some() {
            return Err(WatchError(
                "Watch cancellation is already installed".to_string(),
            ));
        }
        let shared = Arc::downgrade(&self.inner);
        let listener = move || {
            let Some(shared) = shared.upgrade() else {
                return;
            };
            // Termination may run from an abort listener on any thread.
            tokio::spawn(async move {
                shared.terminate(WatchEnd::Cancelled);
            });
        };
        let disposer = signal.add_listener(listener);
        let already = signal.aborted();
        state.cancellation = Some((signal.clone(), disposer));
        drop(state);
        if already {
            self.terminate(WatchEnd::Cancelled);
        }
        Ok(())
    }

    /// Terminate with `cancelled` (upstream `cancel()`).
    pub fn cancel(&self) {
        self.terminate(WatchEnd::Cancelled);
    }

    /// Terminate with `session_closed` (upstream `closeSession()`).
    pub fn close_session(&self) {
        self.terminate(WatchEnd::SessionClosed);
    }

    /// Queue one committed frame with the upstream overflow rule.
    pub fn advance(&self, value: T, ops: Vec<Op>, context: Context) {
        let started = {
            let mut state = self.inner.state.lock();
            if state.end.is_some() || state.retired {
                return;
            }
            let retired = value.is_retired();
            if retired {
                state.retired = true;
            }
            if state.pending.len() >= MAX_PENDING_WATCH_FRAMES {
                state.pending.clear();
                let json = value.to_frame_json();
                state.pending.push_back(WatchFrame {
                    value,
                    ops: vec![Op::Replace(json)],
                    context,
                });
            } else {
                state.pending.push_back(WatchFrame {
                    value,
                    ops,
                    context,
                });
            }
            state.started
        };
        if started {
            self.inner.wake.notify_waiters();
        }
    }

    fn terminate(&self, end: WatchEnd) {
        self.inner.terminate(end);
    }
}

impl<T: ObservedValue> WatchShared<T> {
    fn terminate(&self, end: WatchEnd) {
        let settled = {
            let mut state = self.state.lock();
            if state.end.is_some() {
                Some(state.end.clone().expect("checked above"))
            } else {
                state.end = Some(end.clone());
                state.pending.clear();
                state.cancellation = None;
                None
            }
        };
        if settled.is_some() {
            return;
        }
        let detach = self.state.lock().detach.take();
        if let Some(detach) = detach {
            detach();
        }
        self.handle.terminate(end);
        self.wake.notify_waiters();
    }
}

/// Serialized pump loop: pop one queued frame, publish it through the handle's
/// pending-delivery boundary, and await the sole listener. A retired frame
/// ends the watch with `retired`; a failing listener ends it with
/// `listener_error`.
async fn pump<T: ObservedValue>(shared: Arc<WatchShared<T>>) {
    loop {
        // Register for wakeups before observing state so no push is lost.
        let notified = shared.wake.notified();
        let (started, end) = {
            let state = shared.state.lock();
            (state.started, state.end.clone())
        };
        if end.is_some() {
            return;
        }
        if !started {
            notified.await;
            continue;
        }
        let frame = shared.state.lock().pending.pop_front();
        let Some(frame) = frame else {
            notified.await;
            continue;
        };
        let Some(listener) = (|| {
            let state = shared.state.lock();
            if state.end.is_some() {
                return None;
            }
            state.listener.clone()
        })() else {
            return;
        };
        // Deliver through the handle so value()/termination bookkeeping stays
        // in one place.
        shared
            .handle
            .publish(frame.value.clone(), frame.ops.clone());
        let Some((value, ops)) = shared.handle.take_pending_delivery() else {
            continue;
        };
        let delivery_context = without_abort_signal(frame.context.clone());
        let invocation = listener(value.clone(), ops, delivery_context);
        let outcome = tokio::spawn(invocation).await;
        match outcome {
            Ok(()) => {}
            Err(join_error) => {
                let message = if join_error.is_panic() {
                    panic_message(join_error.into_panic())
                } else {
                    join_error.to_string()
                };
                shared.terminate(WatchEnd::ListenerError { error: message });
                return;
            }
        }
        if value.is_retired() {
            shared.terminate(WatchEnd::Retired);
            return;
        }
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "listener failed".to_string()
    }
}

/// Terminal result of a watch control operation; messages verbatim.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct WatchError(pub String);

// ─── CommittedStateSource ────────────────────────────────────────────────────

/// Session-to-Chord bridge owned one-to-one by one attached state; the default
/// value kind retires on the `null` revision (upstream `CommittedStateSource`).
pub struct CommittedStateSource<T = ObservedDocumentValue> {
    inner: Mutex<SourceState<T>>,
}

struct SourceState<T> {
    value: Option<T>,
    cursor: u64,
    retired: bool,
    closed: bool,
    release: Option<Box<dyn FnOnce() + Send>>,
    attachments: HashMap<usize, Arc<AttachmentInner<T>>>,
}

impl<T: ObservedValue> CommittedStateSource<T> {
    /// Create a source observing `value` as of acquisition; `release` runs at
    /// disposal. The second handle closes the source when dropped-call style
    /// detachment is preferred (upstream returns the instance alone).
    pub fn new(
        value: T,
        release: Box<dyn FnOnce() + Send>,
    ) -> (Arc<Self>, Box<dyn FnOnce() + Send>) {
        let source = Arc::new(Self {
            inner: Mutex::new(SourceState {
                value: Some(value),
                cursor: 0,
                retired: false,
                closed: false,
                release: Some(release),
                attachments: HashMap::new(),
            }),
        });
        let closer = Arc::downgrade(&source);
        let detach = move || {
            if let Some(source) = closer.upgrade() {
                source.close_session();
            }
        };
        (source, Box::new(detach))
    }

    /// Queue one committed revision for every attachment (upstream `advance`).
    pub fn advance(&self, value: T, ops: Vec<Op>, context: Context) {
        let deliveries = {
            let mut state = self.inner.lock();
            if state.closed || state.retired {
                return;
            }
            let retired = value.is_retired();
            state.value = Some(value);
            state.cursor += 1;
            if retired {
                state.retired = true;
            }
            let cursor = state.cursor;
            let current = state.value.as_ref().expect("just stored");
            let json = current.to_frame_json();
            state
                .attachments
                .values()
                .cloned()
                .map(|attachment| {
                    (
                        attachment,
                        cursor,
                        json.clone(),
                        ops.clone(),
                        context.clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        for (attachment, cursor, json, ops, context) in deliveries {
            attachment.publish(ReplicatedStateSourceFrame {
                cursor,
                value: json,
                ops,
                context,
            });
        }
    }

    /// The newest observed revision (upstream reads the private `#value`
    /// through attachment snapshots).
    pub fn snapshot_value(&self) -> Option<T> {
        let state = self.inner.lock();
        if state.closed {
            None
        } else {
            state.value.clone()
        }
    }

    /// Stop every attachment and release the Session subscription (upstream
    /// `closeSession`).
    pub fn close_session(&self) {
        let attachments = {
            let mut state = self.inner.lock();
            if state.closed {
                return;
            }
            let attachments = state.attachments.values().cloned().collect::<Vec<_>>();
            state.attachments.clear();
            attachments
        };
        for attachment in attachments {
            attachment.dispose();
        }
        self.finish_disposal();
    }

    fn finish_disposal(&self) {
        let mut state = self.inner.lock();
        if state.closed {
            return;
        }
        state.closed = true;
        state.value = None;
        if let Some(release) = state.release.take() {
            release();
        }
    }
}

impl<T: ObservedValue> ReplicatedStateSource for CommittedStateSource<T> {
    fn attach(self: Arc<Self>) -> Box<dyn ReplicatedStateSourceAttachment> {
        if self.inner.lock().closed {
            panic!("State source is closed");
        }
        let inner = Arc::new_cyclic(|self_ref| AttachmentInner {
            self_ref: self_ref.clone(),
            source: Arc::downgrade(&self),
            state: Mutex::new(AttachmentState {
                snapshot: {
                    let state = self.inner.lock();
                    ReplicatedStateSourceSnapshot {
                        value: state
                            .value
                            .as_ref()
                            .map(ObservedValue::to_frame_json)
                            .unwrap_or(Value::Null),
                        cursor: state.cursor,
                    }
                },
                frames: VecDeque::new(),
                listener: None,
                activated: false,
                disposed: false,
                scheduled: false,
                delivering: false,
            }),
        });
        self.inner.lock().attachments.insert(
            Arc::as_ptr(&inner) as *const AttachmentInner<T> as usize,
            inner.clone(),
        );
        // A newtype keeps the orphan rule intact (`Arc<_>` cannot implement the
        // foreign trait directly).
        Box::new(AttachmentHandle(inner))
    }
}

/// Orphan-safe attachment handle handed to Chord.
struct AttachmentHandle<T>(Arc<AttachmentInner<T>>);

impl<T: ObservedValue> ReplicatedStateSourceAttachment for AttachmentHandle<T> {
    fn snapshot(&self) -> ReplicatedStateSourceSnapshot {
        self.0.snapshot()
    }

    fn activate(&self, listener: SourceFrameListener) {
        self.0.activate(listener)
    }

    fn dispose(&self) {
        self.0.dispose()
    }
}

struct AttachmentState {
    snapshot: ReplicatedStateSourceSnapshot,
    frames: VecDeque<ReplicatedStateSourceFrame>,
    listener: Option<SourceFrameListener>,
    activated: bool,
    disposed: bool,
    scheduled: bool,
    delivering: bool,
}

struct AttachmentInner<T> {
    /// Set to this attachment's own `Weak` after construction.
    self_ref: Weak<AttachmentInner<T>>,
    source: Weak<CommittedStateSource<T>>,
    state: Mutex<AttachmentState>,
}

impl<T: ObservedValue> AttachmentInner<T> {
    fn publish(&self, frame: ReplicatedStateSourceFrame) {
        let should_schedule = {
            let mut state = self.state.lock();
            if state.disposed {
                return;
            }
            state.frames.push_back(frame);
            state.activated && !state.delivering && !state.scheduled
        };
        if !should_schedule {
            return;
        }
        {
            let mut state = self.state.lock();
            if state.disposed || state.scheduled {
                return;
            }
            state.scheduled = true;
        }
        let task = self.self_ref.clone();
        tokio::spawn(async move {
            let Some(inner) = task.upgrade() else {
                return;
            };
            inner.state.lock().scheduled = false;
            if inner.state.lock().disposed {
                return;
            }
            inner.drain();
        });
    }

    fn drain(&self) {
        {
            let mut state = self.state.lock();
            if state.listener.is_none() || state.delivering || state.disposed {
                return;
            }
            state.delivering = true;
        }
        while !self.state.lock().disposed {
            let frame = {
                let mut state = self.state.lock();
                state.frames.pop_front()
            };
            let Some(frame) = frame else {
                break;
            };
            let listener = self
                .state
                .lock()
                .listener
                .clone()
                .expect("activated listener");
            listener(frame);
        }
        self.state.lock().delivering = false;
    }
}

impl<T: ObservedValue> ReplicatedStateSourceAttachment for AttachmentInner<T> {
    fn snapshot(&self) -> ReplicatedStateSourceSnapshot {
        self.state.lock().snapshot.clone()
    }

    fn activate(&self, listener: SourceFrameListener) {
        {
            let mut state = self.state.lock();
            if state.activated {
                panic!("State attachment is already active");
            }
            if state.disposed {
                panic!("State attachment is disposed");
            }
            state.activated = true;
            state.listener = Some(listener);
        }
        self.drain();
    }

    fn dispose(&self) {
        {
            let mut state = self.state.lock();
            if state.disposed {
                return;
            }
            state.disposed = true;
            state.frames.clear();
            state.listener = None;
        }
        let Some(this) = self.self_ref.upgrade() else {
            return;
        };
        let Some(source) = self.source.upgrade() else {
            return;
        };
        let last = {
            let mut state = source.inner.lock();
            state
                .attachments
                .remove(&(Arc::as_ptr(&this) as *const AttachmentInner<T> as usize));
            state.attachments.is_empty()
        };
        if last {
            source.finish_disposal();
        }
    }
}

/// A conversation view's committed-state source (harness `view.ts` reuses the
/// session machinery with [`ConversationView`] revisions).
pub type ViewSource = Arc<CommittedStateSource<ConversationView>>;

/// Disposable read-only Chord state of one conversation view.
pub type DocumentStateView = Arc<pi_chord::AttachedReplicatedState>;

/// Attach a view source as a disposable read-only Chord state (upstream
/// `replicatedState(source)` in `view.ts`).
pub fn new_view(source: ViewSource) -> Result<DocumentStateView, String> {
    pi_chord::attach_replicated_state(
        source,
        pi_chord::types::ReplicatedStateSourceOptions { on_error: None },
    )
    .map_err(|error| error.to_string())
}

impl CommittedStateSource<ConversationView> {
    /// Associated-form entry point of [`new_view`].
    pub fn new_view(source: ViewSource) -> Result<DocumentStateView, String> {
        new_view(source)
    }
}

/// Erased document watch (upstream `DocumentWatch<T>` default).
pub type DocumentWatch = CommittedWatch<ObservedDocumentValue>;

/// Operations an observer applies for one committed change. An observer
/// hydrated under another definition version holds a differently shaped value,
/// so it receives the new value as a root replacement instead of operations for
/// that shape (upstream `observedOperations`).
pub(crate) fn observed_operations(
    observed_version: &mut u64,
    change: &crate::types::DocumentCommitChange,
) -> Vec<Op> {
    let crate::types::DocumentCommitChange::Document {
        version,
        value,
        ops,
        record: _,
        conversation_id: _,
    } = change
    else {
        return Vec::new();
    };
    let Some(value) = value else {
        return retirement_operations();
    };
    if *version == Some(*observed_version) {
        return ops.clone();
    }
    *observed_version = version.expect("retired changes carry no value");
    vec![Op::Replace(Value::Object(value.clone()))]
}

/// Derive the caller-cancellation message of an aborted acquisition
/// (upstream `cancellationError`): prefer the abort reason, else the DOM
/// `AbortError` text.
pub(crate) fn cancellation_message(signal: &AbortSignal) -> String {
    match signal.reason() {
        Some(reason) => reason.to_string(),
        None => "The operation was aborted".to_string(),
    }
}

/// Derive a committed frame's delivery context: retain values, drop caller
/// cancellation (upstream `withoutAbortSignal(context)`).
pub(crate) fn delivery_context(context: &Context) -> Context {
    without_abort_signal(context.clone())
}
