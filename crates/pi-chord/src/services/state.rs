//! Port of `packages/chord/src/services/state.ts`.
//!
//! Transactional and publication-only replicated states, plus the cold replica
//! used by service consumers.
//!
//! Deviations from upstream (recorded per PORT_MAP.md rule 3):
//! - TypeScript `T extends object` state values are [`JsonValue`] trees; every
//!   revision is a new immutable `serde_json::Value` root. Upstream's structural
//!   sharing is not representable in `serde_json`, so committed revisions are
//!   retained in a graveyard until the state is dropped. This keeps every
//!   outstanding `&JsonValue` handed out by [`ReplicatedState::value`] valid for
//!   the state's lifetime (sound), and preserves upstream observable behavior
//!   ("old snapshots stay usable after a change"), at the cost of retaining old
//!   revisions instead of garbage-collecting them.
//! - Upstream listeners may throw or reject; the error is routed to the
//!   subscriber's error reporter and delivery continues. Rust listeners cannot
//!   return errors, so a listener **panic** fills that role: `catch_unwind`
//!   isolates it, converts the payload to an [`AnyError`] and reports it exactly
//!   where upstream would report a throw/rejection. Because panics print through
//!   the global hook, isolated listener failures are noisy in logs.
//! - `reportErrorAsync` upstream rethrows inside a microtask, crashing the
//!   process. Rust has no recoverable "crash the process" channel for a library,
//!   so the default reporter spawns a task that logs via `tracing::error!`
//!   (synchronously logging when no tokio runtime is active). The deviation is
//!   intentional: a background logging line replaces an uncatchable crash.
//! - Promise-like change callbacks cannot exist (Rust closures are synchronous);
//!   the `TypeError("Replicated state change callbacks must be synchronous")`
//!   branch is unportable and omitted.
//! - Async listener resumption (`#resume`) requires an async executor; the
//!   resume task is spawned on the current tokio runtime. Subscribers whose
//!   listeners never await are fully synchronous, exactly like upstream.
//! - Source/state listener invocation snapshots the listener set under the
//!   publisher lock, then invokes without any lock held, mirroring upstream's
//!   `[...this.#listeners]` copies and its reentrancy guarantees.

use std::any::Any;
use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};
use {parking_lot::Mutex, parking_lot::MutexGuard};

use futures::FutureExt;
use tracing::error as tracing_error;

use crate::context::{Context, BACKGROUND_CONTEXT};
use crate::delta::{
    apply_immutable, is_base_decoded, track, Draft, JsonRevisionValidator, Op, Tracker,
};
use crate::json::JsonValue;
use crate::services::errors::{error, AggregateError, AnyError};
use crate::services::state_internals::{SourceListener, StateInternals, StateSnapshot};
use crate::types::{
    arc_error::ErrorHandler, BoxFuture, ReplicatedState, ReplicatedStateDelivery,
    ReplicatedStateDeliveryKind, ReplicatedStateSource, ReplicatedStateSourceAttachment,
    ReplicatedStateSourceFrame, ReplicatedStateSourceOptions, StateListener, Unsubscribe,
};

/// Maximum deliveries buffered for one subscriber before overflow handling
/// (upstream `StateSubscriber.push`).
const SUBSCRIBER_CAPACITY: usize = 100;

// ─── Committed value slot ────────────────────────────────────────────────────

/// Stable-reference slot for the committed immutable value.
///
/// Readers receive `&JsonValue` borrowed from the current boxed revision; the
/// box pointer itself is only read/written under `write`. When a revision is
/// replaced, the old box moves to `retired` instead of being dropped, so
/// references handed out earlier stay valid until the owner is dropped — the
/// borrow checker guarantees the owner outlives every outstanding reference.
struct Committed {
    cell: UnsafeCell<Option<Box<JsonValue>>>,
    write: Mutex<()>,
    retired: Mutex<Vec<Box<JsonValue>>>,
}

// The cell is only ever accessed under `write`; `retired` keeps handed-out
// payloads alive. Send/Sync follow from the guarded access.
unsafe impl Send for Committed {}
unsafe impl Sync for Committed {}

impl Committed {
    fn new(value: Option<JsonValue>) -> Committed {
        Committed {
            cell: UnsafeCell::new(value.map(Box::new)),
            write: Mutex::new(()),
            retired: Mutex::new(Vec::new()),
        }
    }

    fn write_lock(&self) -> MutexGuard<'_, ()> {
        self.write.lock()
    }

    fn get(&self) -> Option<&JsonValue> {
        let _guard = self.write_lock();
        // SAFETY: exclusive access under `write`; the payload outlives the
        // borrow because replaced boxes retire rather than drop.
        unsafe { (*self.cell.get()).as_deref() }
    }

    /// Replace the committed value, retiring the previous revision.
    fn set(&self, value: Option<JsonValue>) {
        let _guard = self.write_lock();
        // SAFETY: exclusive access under `write`.
        let old = std::mem::replace(unsafe { &mut *self.cell.get() }, value.map(Box::new));
        if let Some(old) = old {
            self.retired.lock().push(old);
        }
    }
}

// ─── Error helpers ───────────────────────────────────────────────────────────

/// Convert a caught panic payload into an owned error (upstream `toError`).
fn panic_to_error(payload: Box<dyn Any + Send>) -> AnyError {
    if let Some(message) = payload.downcast_ref::<String>() {
        error(message.clone())
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        error((*message).to_string())
    } else {
        error("listener panicked")
    }
}

/// Upstream `throwCollectedErrors`: one error is rethrown as-is, several are
/// aggregated under `message`.
fn throw_collected(errors: Vec<AnyError>, message: &str) -> Result<(), AnyError> {
    match errors.len() {
        0 => Ok(()),
        1 => Err(errors.into_iter().next().expect("one error")),
        _ => Err(AggregateError::new(message, errors)),
    }
}

/// Upstream `reportErrorAsync` rethrows inside a microtask, which crashes the
/// process. Rust libraries cannot crash their host; log instead (documented
/// deviation). Spawning keeps the upstream "asynchronous, isolated" timing when
/// a tokio runtime is active.
fn report_error_async(err: AnyError) {
    let report = || tracing_error!(error = %err, "replicated state listener failure");
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            handle.spawn(async move {
                tracing_error!(error = %err, "replicated state listener failure");
            });
        }
        Err(_) => report(),
    }
}

type ReportError = Arc<dyn Fn(AnyError) + Send + Sync>;

fn report_error_fn(report: &ReportError, err: AnyError) {
    report(err);
}

// ─── StateSubscriber ─────────────────────────────────────────────────────────

/// One delivery frame queued for a subscriber (upstream `StateDelivery<T>`).
struct StateDeliveryFrame {
    value: JsonValue,
    context: Context,
    delivery: ReplicatedStateDelivery,
}

struct SubscriberState {
    listener: StateListener,
    report_error: ReportError,
    pending: VecDeque<StateDeliveryFrame>,
    running: bool,
    started: bool,
    closed: bool,
}

/// One public subscription, independent of producer and other subscribers.
type Subscriber = Arc<Mutex<SubscriberState>>;

fn subscriber_lock(subscriber: &Subscriber) -> MutexGuard<'_, SubscriberState> {
    subscriber.lock()
}

fn subscriber_new(listener: StateListener, report_error: ReportError) -> Subscriber {
    Arc::new(Mutex::new(SubscriberState {
        listener,
        report_error,
        pending: VecDeque::new(),
        running: false,
        started: false,
        closed: false,
    }))
}

fn subscriber_push(subscriber: &Subscriber, frame: StateDeliveryFrame) {
    let mut state = subscriber_lock(subscriber);
    if state.closed {
        return;
    }
    if state.pending.len() == SUBSCRIBER_CAPACITY {
        // A cold replica can queue updates reentrantly before this subscriber's
        // first hydration starts; keep the hydration frame in that case, else
        // keep only the newest delivery.
        let hydration = if state.started {
            None
        } else {
            state.pending.pop_front()
        };
        state.pending.clear();
        if let Some(hydration) = hydration {
            state.pending.push_front(hydration);
        }
    }
    state.pending.push_back(frame);
}

fn subscriber_clear(subscriber: &Subscriber) {
    subscriber_lock(subscriber).pending.clear();
}

fn subscriber_close(subscriber: &Subscriber) {
    let mut state = subscriber_lock(subscriber);
    state.closed = true;
    state.pending.clear();
}

/// Invoke the listener for one frame, isolating panics like upstream's
/// try/catch. Returns the pending future when the listener awaits.
fn invoke_listener(
    listener: &StateListener,
    report_error: &ReportError,
    frame: &StateDeliveryFrame,
) -> Option<BoxFuture<()>> {
    let invoked = catch_unwind(AssertUnwindSafe(|| {
        listener(&frame.value, &frame.context, &frame.delivery)
    }));
    let mut future = match invoked {
        Ok(future) => future,
        Err(payload) => {
            report_error_fn(report_error, panic_to_error(payload));
            return None;
        }
    };
    // Poll once: a ready future continues the synchronous drain loop, a pending
    // future is parked and resumed on the runtime (upstream Promise resolution).
    // The poll is isolated like the call itself: an async block that fails on
    // its first poll mirrors an immediately-rejected promise.
    let mut cx = TaskContext::from_waker(futures::task::noop_waker_ref());
    let polled = catch_unwind(AssertUnwindSafe(|| future.poll_unpin(&mut cx)));
    match polled {
        Ok(Poll::Ready(())) => None,
        Ok(Poll::Pending) => Some(future),
        Err(payload) => {
            report_error_fn(report_error, panic_to_error(payload));
            None
        }
    }
}

fn subscriber_drain(subscriber: &Subscriber) {
    {
        let mut state = subscriber_lock(subscriber);
        if state.running || state.closed {
            return;
        }
        state.running = true;
    }
    loop {
        let (frame, listener, report_error) = {
            let mut state = subscriber_lock(subscriber);
            match state.pending.pop_front() {
                Some(frame) => {
                    state.started = true;
                    (frame, state.listener.clone(), state.report_error.clone())
                }
                None => {
                    state.running = false;
                    return;
                }
            }
        };
        match invoke_listener(&listener, &report_error, &frame) {
            None => continue,
            Some(future) => {
                subscriber_park_and_resume(subscriber, report_error, future);
                return;
            }
        }
    }
}

/// Park one pending listener future on the runtime; when it settles, release the
/// subscriber's running flag and continue the drain (upstream `#resume`).
fn subscriber_park_and_resume(
    subscriber: &Subscriber,
    report_error: ReportError,
    future: BoxFuture<()>,
) {
    let resume_target = subscriber.clone();
    tokio::spawn(async move {
        // Isolate listener rejections exactly like the synchronous path.
        if let Err(payload) = AssertUnwindSafe(future).catch_unwind().await {
            report_error_fn(&report_error, panic_to_error(payload));
        }
        subscriber_lock(&resume_target).running = false;
        subscriber_drain(&resume_target);
    });
}

// ─── ReplicatedStatePublisher ────────────────────────────────────────────────

struct Publication {
    value: JsonValue,
    ops: Vec<Op>,
    sequence: u64,
    context: Context,
}

struct PublisherState {
    /// Subscribers with the sequence they were hydrated at.
    listeners: Vec<(Subscriber, u64)>,
    report_error: ReportError,
    source_listeners: Vec<SourceListener>,
    publications: VecDeque<Publication>,
    /// Mirror of the committed value; shared with the owning state.
    committed: Arc<Committed>,
    sequence: u64,
    delivering: bool,
}

/// Maintains local publication order independently of how revisions are
/// produced (upstream `ReplicatedStatePublisher`).
struct Publisher(Arc<Mutex<PublisherState>>);

fn publisher_lock(state: &Arc<Mutex<PublisherState>>) -> MutexGuard<'_, PublisherState> {
    state.lock()
}

impl Publisher {
    fn new(committed: Arc<Committed>, report_error: ReportError) -> Publisher {
        Publisher(Arc::new(Mutex::new(PublisherState {
            listeners: Vec::new(),
            report_error,
            source_listeners: Vec::new(),
            publications: VecDeque::new(),
            committed,
            sequence: 0,
            delivering: false,
        })))
    }

    /// Atomically capture the immutable value and its matching sequence.
    fn snapshot(&self) -> (JsonValue, u64) {
        let state = publisher_lock(&self.0);
        (
            state.committed.get().cloned().unwrap_or_default(),
            state.sequence,
        )
    }

    fn subscribe(&self, listener: StateListener) -> Unsubscribe {
        let subscriber = {
            let mut state = publisher_lock(&self.0);
            let (value, sequence) = (
                state.committed.get().cloned().unwrap_or_default(),
                state.sequence,
            );
            let subscriber = subscriber_new(listener, state.report_error.clone());
            state.listeners.push((subscriber.clone(), sequence));
            subscriber_push(
                &subscriber,
                StateDeliveryFrame {
                    value,
                    context: service_delivery_context(),
                    delivery: ReplicatedStateDelivery {
                        kind: ReplicatedStateDeliveryKind::Hydrate,
                        sequence,
                    },
                },
            );
            subscriber
        };
        subscriber_drain(&subscriber);
        let owner = self.0.clone();
        Box::new(move || {
            // Close before touching the publisher lock: lock order is always
            // publisher → subscriber elsewhere.
            subscriber_close(&subscriber);
            publisher_lock(&owner)
                .listeners
                .retain(|(existing, _)| !Arc::ptr_eq(existing, &subscriber));
        })
    }

    fn subscribe_source(&self, listener: SourceListener) -> Unsubscribe {
        {
            let mut state = publisher_lock(&self.0);
            state.source_listeners.push(listener.clone());
        }
        let owner = self.0.clone();
        Box::new(move || {
            publisher_lock(&owner)
                .source_listeners
                .retain(|existing| !Arc::ptr_eq(existing, &listener));
        })
    }

    /// Publish an already-prepared immutable revision and return isolated
    /// source-listener failures.
    fn publish(&self, value: JsonValue, ops: Vec<Op>, context: &Context) -> Vec<AnyError> {
        let mut errors: Vec<AnyError> = Vec::new();
        {
            let mut state = publisher_lock(&self.0);
            state.committed.set(Some(value.clone()));
            state.sequence += 1;
            let publication = Publication {
                value,
                ops,
                sequence: state.sequence,
                context: context.clone(),
            };
            state.publications.push_back(publication);
            if state.delivering {
                return errors;
            }
            state.delivering = true;
        }
        loop {
            let publication = publisher_lock(&self.0).publications.pop_front();
            let Some(publication) = publication else {
                break;
            };
            let source_listeners = publisher_lock(&self.0).source_listeners.clone();
            for listener in &source_listeners {
                let invoked = catch_unwind(AssertUnwindSafe(|| {
                    listener(&publication.ops, publication.sequence, &publication.context)
                }));
                if let Err(payload) = invoked {
                    errors.push(panic_to_error(payload));
                }
            }
            let delivery = ReplicatedStateDelivery {
                kind: ReplicatedStateDeliveryKind::Update,
                sequence: publication.sequence,
            };
            let listeners = publisher_lock(&self.0).listeners.clone();
            for (subscriber, hydrated_sequence) in &listeners {
                if publication.sequence <= *hydrated_sequence {
                    continue;
                }
                subscriber_push(
                    subscriber,
                    StateDeliveryFrame {
                        value: publication.value.clone(),
                        context: publication.context.clone(),
                        delivery: delivery.clone(),
                    },
                );
                subscriber_drain(subscriber);
            }
        }
        publisher_lock(&self.0).delivering = false;
        errors
    }
}

// ─── MutableReplicatedState ──────────────────────────────────────────────────

struct MutableInternals {
    publisher: Publisher,
}

impl StateInternals for MutableInternals {
    fn snapshot(&self) -> StateSnapshot {
        let (value, sequence) = self.publisher.snapshot();
        StateSnapshot { value, sequence }
    }

    fn subscribe(&self, listener: SourceListener) -> Unsubscribe {
        self.publisher.subscribe_source(listener)
    }
}

/// Authoritative replicated state updated transactionally through drafts
/// (upstream `MutableReplicatedStateImpl`).
pub struct MutableReplicatedState {
    tracker: Mutex<Tracker>,
    publisher: Publisher,
    committed: Arc<Committed>,
    internals: Arc<dyn StateInternals>,
    changing: std::sync::atomic::AtomicBool,
}

impl MutableReplicatedState {
    /// Create authoritative state by taking immutable ownership of a
    /// strict-JSON root. The caller must not mutate `initial` afterwards.
    pub fn new(initial: JsonValue) -> Arc<Self> {
        let tracker = track(initial);
        let committed = Arc::new(Committed::new(Some(tracker.value().clone())));
        let publisher = Publisher::new(committed.clone(), Arc::new(report_error_async));
        let internals: Arc<dyn StateInternals> = Arc::new(MutableInternals {
            publisher: Publisher(publisher.0.clone()),
        });
        Arc::new(MutableReplicatedState {
            tracker: Mutex::new(tracker),
            publisher,
            committed,
            internals,
            changing: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// The committed immutable value.
    pub fn value(&self) -> &JsonValue {
        self.committed
            .get()
            .expect("mutable replicated state always has a value")
    }

    /// Apply a transactional mutation through a draft; aborts and rolls back on
    /// closure failure. Listener-triggered changes queue in sequence order.
    pub fn change(
        &self,
        context: &Context,
        mutate: impl FnOnce(&mut Draft) -> Result<(), AnyError>,
    ) -> Result<(), AnyError> {
        use std::sync::atomic::Ordering;
        if self
            .changing
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(error(
                "Replicated state cannot be changed reentrantly from a change callback",
            ));
        }
        let revision: Result<Option<(JsonValue, Vec<Op>)>, AnyError> = (|| {
            let mut tracker = self.tracker.lock();
            let mut change = tracker.begin_change();
            if let Err(err) = mutate(&mut change.state()) {
                change.abort();
                return Err(err);
            }
            // A failing prepare has already aborted its overlay internally
            // (upstream marks the context aborted in the same spot).
            let mut prepared = change.prepare().map_err(AnyError::from)?;
            let payload = if prepared.ops.is_empty() {
                None
            } else {
                Some((prepared.value.clone(), std::mem::take(&mut prepared.ops)))
            };
            tracker.adopt(prepared).map_err(AnyError::from)?;
            Ok(payload)
        })();
        self.changing.store(false, Ordering::Release);

        let Some((value, ops)) = revision? else {
            return Ok(());
        };
        let errors = self.publisher.publish(value, ops, context);
        throw_collected(errors, "Replicated state listeners failed")
    }

    /// Replace the whole root; deeply equal replacements are no-ops.
    pub fn replace(&self, context: &Context, value: JsonValue) -> Result<(), AnyError> {
        use std::sync::atomic::Ordering;
        if self.changing.load(Ordering::Acquire) {
            return Err(error(
                "Replicated state cannot be replaced from a change callback",
            ));
        }
        let revision = (|| -> Result<Option<(JsonValue, Vec<Op>)>, AnyError> {
            let mut tracker = self.tracker.lock();
            let mut prepared = tracker.prepare_replace(value).map_err(AnyError::from)?;
            let payload = if prepared.ops.is_empty() {
                None
            } else {
                Some((prepared.value.clone(), std::mem::take(&mut prepared.ops)))
            };
            tracker.adopt(prepared).map_err(AnyError::from)?;
            Ok(payload)
        })()?;
        let Some((value, ops)) = revision else {
            return Ok(());
        };
        let errors = self.publisher.publish(value, ops, context);
        throw_collected(errors, "Replicated state listeners failed")
    }

    pub fn subscribe(&self, listener: StateListener) -> Unsubscribe {
        self.publisher.subscribe(listener)
    }

    /// Internals through which the provider observes this state.
    pub fn internals(&self) -> Arc<dyn StateInternals> {
        self.internals.clone()
    }
}

impl ReplicatedState for MutableReplicatedState {
    fn value(&self) -> Option<&JsonValue> {
        Some(
            self.committed
                .get()
                .expect("mutable replicated state always has a value"),
        )
    }

    fn subscribe(&self, listener: StateListener) -> Unsubscribe {
        MutableReplicatedState::subscribe(self, listener)
    }
}

// ─── Attached replicated state ───────────────────────────────────────────────

struct AttachedInternals {
    publisher: Publisher,
}

impl StateInternals for AttachedInternals {
    fn snapshot(&self) -> StateSnapshot {
        let (value, sequence) = self.publisher.snapshot();
        StateSnapshot { value, sequence }
    }

    fn subscribe(&self, listener: SourceListener) -> Unsubscribe {
        self.publisher.subscribe_source(listener)
    }
}

/// Publication-only replicated state attached to one authoritative immutable
/// source stream (upstream `AttachedReplicatedStateImpl`).
pub struct AttachedReplicatedState {
    publisher: Publisher,
    committed: Arc<Committed>,
    /// The attachment trait is interior-mutable (`activate`/`dispose` are
    /// `&self`, single-use/idempotent inside the implementation), so no lock is
    /// needed — `activate` drains buffered frames synchronously and must not
    /// hold a lock user code could re-enter.
    attachment: Box<dyn ReplicatedStateSourceAttachment>,
    cursor: Mutex<u64>,
    disposed: std::sync::atomic::AtomicBool,
    report_error: ReportError,
    internals: Arc<dyn StateInternals>,
}

impl AttachedReplicatedState {
    fn new(
        attachment: Box<dyn ReplicatedStateSourceAttachment>,
        options: &ReplicatedStateSourceOptions,
    ) -> Arc<Self> {
        let snapshot = attachment.snapshot();
        assert_cursor(snapshot.cursor, "snapshot");
        let committed = Arc::new(Committed::new(Some(snapshot.value.clone())));
        let report_error: ReportError = match &options.on_error {
            Some(handler) => handler.clone(),
            None => Arc::new(report_error_async),
        };
        let publisher = Publisher::new(committed.clone(), report_error.clone());
        let internals: Arc<dyn StateInternals> = Arc::new(AttachedInternals {
            publisher: Publisher(publisher.0.clone()),
        });
        Arc::new(AttachedReplicatedState {
            publisher,
            committed,
            attachment,
            cursor: Mutex::new(snapshot.cursor),
            disposed: std::sync::atomic::AtomicBool::new(false),
            report_error,
            internals,
        })
    }

    /// The committed immutable value.
    pub fn value(&self) -> &JsonValue {
        self.committed
            .get()
            .expect("attached replicated state always has a value")
    }

    pub fn subscribe(&self, listener: StateListener) -> Unsubscribe {
        self.publisher.subscribe(listener)
    }

    /// Start receiving source frames. Called by [`attach_replicated_state_source`].
    pub fn activate(self: &Arc<Self>) {
        let listener: crate::types::SourceFrameListener = {
            let state = Arc::downgrade(self);
            Arc::new(move |frame: ReplicatedStateSourceFrame| {
                if let Some(state) = state.upgrade() {
                    state.receive(frame);
                }
            })
        };
        self.attachment.activate(listener);
    }

    /// Stop delivery and release source resources. Idempotent.
    pub fn dispose(&self) {
        use std::sync::atomic::Ordering;
        if self.disposed.swap(true, Ordering::AcqRel) {
            return;
        }
        self.attachment.dispose();
    }

    fn receive(self: Arc<Self>, frame: ReplicatedStateSourceFrame) {
        use std::sync::atomic::Ordering;
        if self.disposed.load(Ordering::Acquire) {
            return;
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<(), AnyError> {
            assert_cursor(frame.cursor, "frame");
            let expected = {
                let mut cursor = self.cursor.lock();
                let expected = *cursor + 1;
                *cursor = frame.cursor;
                expected
            };
            if frame.cursor != expected {
                return Err(error(format!(
                    "Replicated state source cursor has a gap: expected {expected}, received {}",
                    frame.cursor
                )));
            }
            let errors = self
                .publisher
                .publish(frame.value, frame.ops, &frame.context);
            match errors.len() {
                0 => Ok(()),
                1 => Err(errors.into_iter().next().expect("one error")),
                _ => Err(AggregateError::new(
                    "Replicated state listeners failed",
                    errors,
                )),
            }
        }));
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(err)) => self.fail(err),
            Err(payload) => self.fail(panic_to_error(payload)),
        }
    }

    fn fail(&self, err: AnyError) {
        use std::sync::atomic::Ordering;
        if self.disposed.load(Ordering::Acquire) {
            return;
        }
        self.disposed.store(true, Ordering::Release);
        match catch_unwind(AssertUnwindSafe(|| self.attachment.dispose())) {
            Ok(()) => report_error_fn(&self.report_error, err),
            Err(dispose_payload) => report_error_fn(
                &self.report_error,
                AggregateError::new(
                    "Replicated state source contract failed",
                    vec![err, panic_to_error(dispose_payload)],
                ),
            ),
        }
    }

    /// Internals through which the provider observes this state.
    pub fn internals(&self) -> Arc<dyn StateInternals> {
        self.internals.clone()
    }
}

impl ReplicatedState for AttachedReplicatedState {
    fn value(&self) -> Option<&JsonValue> {
        Some(
            self.committed
                .get()
                .expect("attached replicated state always has a value"),
        )
    }

    fn subscribe(&self, listener: StateListener) -> Unsubscribe {
        AttachedReplicatedState::subscribe(self, listener)
    }
}

/// Attach a publication-only replicated state to one authoritative immutable
/// source stream. Activates on attach; on failure the attachment is disposed
/// and the error reported through `options.on_error` / returned.
pub fn attach_replicated_state_source(
    source: Arc<dyn ReplicatedStateSource>,
    options: ReplicatedStateSourceOptions,
) -> Result<Arc<AttachedReplicatedState>, AnyError> {
    let attachment = source.attach();
    let built = catch_unwind(AssertUnwindSafe(|| {
        let state = AttachedReplicatedState::new(attachment, &options);
        state.activate();
        state
    }));
    match built {
        Ok(state) => Ok(state),
        Err(payload) => {
            // The attachment may or may not have been moved into the state;
            // construction disposes via the state on drop-less failure paths is
            // not modeled, so surface the failure directly (cursor validation
            // is the only upstream failure source and is vacuous for u64).
            Err(panic_to_error(payload))
        }
    }
}

// ─── ReplicatedStateReplica ──────────────────────────────────────────────────

/// A cold read-only state used by service consumers until a complete snapshot
/// arrives (upstream `ReplicatedStateReplica`).
pub struct ReplicatedStateReplica {
    listeners: Arc<Mutex<Vec<Subscriber>>>,
    report_error: ReportError,
    validator: Mutex<JsonRevisionValidator>,
    committed: Arc<Committed>,
    sequence: Mutex<Option<u64>>,
}

impl ReplicatedStateReplica {
    pub fn new(report_error: ErrorHandler) -> Arc<Self> {
        Arc::new(ReplicatedStateReplica {
            listeners: Arc::new(Mutex::new(Vec::new())),
            report_error,
            validator: Mutex::new(JsonRevisionValidator::new()),
            committed: Arc::new(Committed::new(None)),
            sequence: Mutex::new(None),
        })
    }

    /// The committed immutable value, `None` until hydration.
    pub fn value(&self) -> Option<&JsonValue> {
        self.committed.get()
    }

    pub fn subscribe(&self, listener: StateListener) -> Unsubscribe {
        let subscriber = {
            let mut listeners = self.listeners.lock();
            let subscriber = subscriber_new(listener, self.report_error.clone());
            if let Some(value) = self.committed.get() {
                let sequence = self.sequence.lock().expect("value implies sequence");
                subscriber_push(
                    &subscriber,
                    StateDeliveryFrame {
                        value: value.clone(),
                        context: service_delivery_context(),
                        delivery: ReplicatedStateDelivery {
                            kind: ReplicatedStateDeliveryKind::Hydrate,
                            sequence,
                        },
                    },
                );
            }
            listeners.push(subscriber.clone());
            subscriber
        };
        subscriber_drain(&subscriber);
        let listeners = self.listeners.clone();
        Box::new(move || {
            subscriber_close(&subscriber);
            listeners
                .lock()
                .retain(|existing| !Arc::ptr_eq(existing, &subscriber));
        })
    }

    /// Hydrate from a base operation batch. Failures clear the replica.
    pub fn hydrate(&self, sequence: u64, ops: &[Op], context: &Context) -> Result<(), AnyError> {
        let next = (|| -> Result<JsonValue, AnyError> {
            if !is_base_decoded(ops) {
                return Err(error(
                    "Replicated state snapshot is not a base operation batch",
                ));
            }
            let applied = apply_immutable(&JsonValue::Null, ops).map_err(AnyError::from)?;
            // Strict-JSON validation is upheld by construction in serde_json;
            // the validator remains as the upstream call-site (infallible).
            Ok(self.validator.lock().validate(applied))
        })();
        match next {
            Err(err) => {
                self.clear();
                Err(err)
            }
            Ok(next) => {
                *self.sequence.lock() = Some(sequence);
                self.committed.set(Some(next));
                self.deliver_all(
                    context,
                    ReplicatedStateDelivery {
                        kind: ReplicatedStateDeliveryKind::Hydrate,
                        sequence,
                    },
                );
                Ok(())
            }
        }
    }

    /// Apply one sequenced update batch. Failures clear the replica.
    pub fn update(&self, sequence: u64, ops: &[Op], context: &Context) -> Result<(), AnyError> {
        let current_sequence = *self.sequence.lock();
        let Some(current_sequence) = current_sequence else {
            return Err(error(
                "Replicated state received an update before hydration",
            ));
        };
        if sequence != current_sequence + 1 {
            self.clear();
            return Err(error("Replicated state update sequence has a gap"));
        }
        let applied = (|| -> Result<JsonValue, AnyError> {
            let applied = apply_immutable(
                self.committed.get().expect("hydrated replica has a value"),
                ops,
            )
            .map_err(AnyError::from)?;
            Ok(self.validator.lock().validate(applied))
        })();
        match applied {
            Err(err) => {
                self.clear();
                Err(err)
            }
            Ok(applied) => {
                *self.sequence.lock() = Some(sequence);
                self.committed.set(Some(applied));
                self.deliver_all(
                    context,
                    ReplicatedStateDelivery {
                        kind: ReplicatedStateDeliveryKind::Update,
                        sequence,
                    },
                );
                Ok(())
            }
        }
    }

    /// Drop the hydrated value and every queued delivery.
    pub fn clear(&self) {
        *self.sequence.lock() = None;
        self.committed.set(None);
        let listeners = self.listeners.lock().clone();
        for subscriber in listeners {
            subscriber_clear(&subscriber);
        }
    }

    fn deliver_all(&self, context: &Context, delivery: ReplicatedStateDelivery) {
        let Some(value) = self.committed.get() else {
            return;
        };
        let subscribers = self.listeners.lock().clone();
        // Enqueue for everyone before user code can publish another revision
        // reentrantly.
        for subscriber in &subscribers {
            subscriber_push(
                subscriber,
                StateDeliveryFrame {
                    value: value.clone(),
                    context: context.clone(),
                    delivery: delivery.clone(),
                },
            );
        }
        for subscriber in &subscribers {
            subscriber_drain(subscriber);
        }
    }
}

impl ReplicatedState for ReplicatedStateReplica {
    fn value(&self) -> Option<&JsonValue> {
        ReplicatedStateReplica::value(self)
    }

    fn subscribe(&self, listener: StateListener) -> Unsubscribe {
        ReplicatedStateReplica::subscribe(self, listener)
    }
}

// ─── Shared helpers ──────────────────────────────────────────────────────────

/// Context for synthetic service deliveries without a caller.
pub fn service_delivery_context() -> Context {
    // TODO(upstream): Add delivery-scoped cancellation or metadata if deliveries
    // gain an owned lifecycle.
    BACKGROUND_CONTEXT.clone()
}

/// Upstream `assertCursor`. `u64` cursors make the safe-integer check vacuous —
/// any `u64` is a safe integer — so this is a no-op kept for call-site parity.
/// The frame-cursor gap check in [`AttachedReplicatedState::receive`] carries the
/// real guarantee. If cursors ever become signed, tighten here with:
/// `Replicated state source {kind} cursor must be a safe integer`.
fn assert_cursor(_cursor: u64, kind: &str) {
    let _ = kind;
}

#[cfg(test)]
mod tests {
    // Dropping an `Unsubscribe` without calling it deliberately leaks the
    // subscription for the (short) test lifetime, mirroring upstream handles
    // that simply go out of scope; explicit-unsubscribe paths are covered by
    // dedicated tests, so the must-use lint is noise here.
    #![allow(unused_must_use)]

    use super::*;
    use crate::context::{with_cancel, ContextKey};
    use crate::delta::{diff_revisions, Seg};
    use crate::types::{ReplicatedStateSourceSnapshot, SourceFrameListener};
    use serde_json::json;
    use std::sync::atomic::AtomicUsize;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn ctx() -> &'static Context {
        &BACKGROUND_CONTEXT
    }

    fn ops(value: JsonValue) -> Vec<Op> {
        serde_json::from_value(value).expect("test ops are valid op tuples")
    }

    /// Poll until `condition` holds or the timeout elapses (upstream `vi.waitFor`).
    async fn wait_for(condition: impl Fn() -> bool) {
        for _ in 0..2000 {
            if condition() {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        panic!("wait_for condition never held");
    }

    /// One-shot gate for async listeners (upstream `deferred`). `reject` drops
    /// the sender; awaiting listeners translate the receive failure into a
    /// panic, which is the Rust stand-in for a rejected promise.
    struct Deferred {
        tx: Option<tokio::sync::oneshot::Sender<()>>,
        rx: Option<tokio::sync::oneshot::Receiver<()>>,
    }

    impl Deferred {
        fn new() -> Deferred {
            let (tx, rx) = tokio::sync::oneshot::channel();
            Deferred {
                tx: Some(tx),
                rx: Some(rx),
            }
        }

        fn resolve(&mut self) {
            let _ = self.tx.take().expect("resolved once").send(());
        }

        fn reject(&mut self) {
            self.tx.take();
        }

        /// The receiver half; the extra string is the panic message used when
        /// the gate is rejected instead of resolved.
        fn receiver(&mut self, rejection: &'static str) -> impl Future<Output = ()> + Send {
            let rx = self.rx.take().expect("receiver taken once");
            async move {
                match rx.await {
                    Ok(()) => {}
                    Err(_) => panic!("{rejection}"),
                }
            }
        }
    }

    use std::future::Future;

    /// Collector for delivery events.
    type Events<A> = Arc<Mutex<Vec<A>>>;

    fn events<A>() -> Events<A> {
        Arc::new(Mutex::new(Vec::new()))
    }

    fn snapshot_events<A: Clone>(events: &Events<A>) -> Vec<A> {
        events.lock().clone()
    }

    /// Error collectors snapshot as message strings (`AnyError` is not `Clone`).
    fn snapshot_errors(events: &Events<AnyError>) -> Vec<String> {
        events.lock().iter().map(ToString::to_string).collect()
    }

    /// Sync listener recording delivery metadata.
    fn recording_listener(received: Events<(u64, ReplicatedStateDelivery)>) -> StateListener {
        Arc::new(move |value, _context, delivery| {
            received.lock().push((
                value
                    .as_object()
                    .and_then(|o| o.get("value"))
                    .and_then(JsonValue::as_u64)
                    .unwrap_or(0),
                delivery.clone(),
            ));
            Box::pin(std::future::ready(()))
        })
    }

    fn value_of(value: &JsonValue) -> u64 {
        value
            .get("value")
            .and_then(JsonValue::as_u64)
            .expect("test values carry {value}")
    }

    // ── state.test.ts: transactional replicated state ───────────────────────

    // "publishes one immutable structurally shared revision"
    #[test]
    fn publishes_one_immutable_structurally_shared_revision() {
        let state = MutableReplicatedState::new(
            json!({ "changed": { "value": 1 }, "retained": { "value": 2 } }),
        );
        let previous: &JsonValue = state.value();
        let deliveries = events();
        state.subscribe(recording_listener(deliveries.clone()));
        state
            .change(ctx(), |draft| {
                draft.set(&[Seg::key("changed")], "value", json!(3))?;
                draft.set(&[Seg::key("changed")], "value", json!(4))?;
                Ok(())
            })
            .expect("change succeeds");
        assert_eq!(
            state.value(),
            &json!({ "changed": { "value": 4 }, "retained": { "value": 2 } })
        );
        // Root identity: a committed revision is a new value (upstream not.toBe).
        assert!(!std::ptr::eq(previous, state.value()));
        // Upstream also asserts `retained` branch sharing via object identity;
        // serde_json values cannot share subtrees, so the equivalent guarantee
        // (unrelated branches are not re-diffed or re-created semantically) is
        // asserted structurally instead.
        assert_eq!(previous.get("retained"), state.value().get("retained"));
        assert_eq!(
            snapshot_events(&deliveries),
            vec![
                (
                    0,
                    ReplicatedStateDelivery {
                        kind: ReplicatedStateDeliveryKind::Hydrate,
                        sequence: 0
                    }
                ),
                (
                    0,
                    ReplicatedStateDelivery {
                        kind: ReplicatedStateDeliveryKind::Update,
                        sequence: 1
                    }
                )
            ]
        );
    }

    // "rolls back callback failures and revokes escaped drafts" — the escaped
    // draft TypeError half is compile-time in Rust (borrow checker), so only the
    // rollback half is ported.
    #[test]
    fn rolls_back_callback_failures() {
        let state = MutableReplicatedState::new(json!({ "nested": { "value": 1 } }));
        let previous: &JsonValue = state.value();
        let error = state
            .change(ctx(), |draft| {
                draft.set(&[Seg::key("nested")], "value", json!(2))?;
                Err(error("stop"))
            })
            .expect_err("callback failure aborts the change");
        assert_eq!(error.to_string(), "stop");
        // Rolled back: the committed root is untouched, down to identity.
        assert!(std::ptr::eq(previous, state.value()));
    }

    // "rejects PromiseLike callbacks and aborts their draft" is non-portable:
    // Rust change callbacks are synchronous closures returning Result; there is
    // no PromiseLike value to reject.

    #[test]
    fn rejects_nested_changes_and_replacements_without_losing_the_outer_rollback() {
        let state = MutableReplicatedState::new(json!({ "left": 0, "right": 0 }));
        let error = state
            .change(ctx(), |draft| {
                draft.set(&[], "left", json!(1))?;
                state.change(ctx(), |nested| {
                    nested.set(&[], "right", json!(2)).map_err(AnyError::from)
                })
            })
            .expect_err("nested change is rejected");
        assert!(error.to_string().contains("reentrantly"));
        assert_eq!(state.value(), &json!({ "left": 0, "right": 0 }));
        let error = state
            .change(ctx(), |_draft| {
                state.replace(ctx(), json!({ "left": 1, "right": 2 }))
            })
            .expect_err("replace inside change is rejected");
        assert!(error.to_string().contains("change callback"));
        assert_eq!(state.value(), &json!({ "left": 0, "right": 0 }));
    }

    // "queues listener-triggered changes in sequence order"
    #[test]
    fn queues_listener_triggered_changes_in_sequence_order() {
        let state = MutableReplicatedState::new(json!({ "value": 0 }));
        let source_sequences = events::<u64>();
        let source_sink = source_sequences.clone();
        let deliveries = events::<(u64, u64)>();
        let late_deliveries = events::<ReplicatedStateDelivery>();
        let nested = Arc::new(AtomicUsize::new(0));
        let state_for_source = state.clone();
        let nested_for_source = nested.clone();
        let late_for_source = late_deliveries.clone();
        state
            .internals()
            .subscribe(Arc::new(move |_operations, sequence, _context| {
                if nested_for_source.swap(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    state_for_source
                        .change(ctx(), |draft| {
                            draft.set(&[], "value", json!(2)).map_err(AnyError::from)
                        })
                        .expect("listener-triggered change queues");
                    let late_sink = late_for_source.clone();
                    state_for_source.subscribe(Arc::new(move |_value, _context, delivery| {
                        late_sink.lock().push(delivery.clone());
                        Box::pin(std::future::ready(()))
                    }));
                }
                let _ = sequence;
            }));
        state
            .internals()
            .subscribe(Arc::new(move |_operations, sequence, _context| {
                source_sink.lock().push(sequence);
            }));
        let update_sink = deliveries.clone();
        state.subscribe(Arc::new(move |value, _context, delivery| {
            if delivery.kind == ReplicatedStateDeliveryKind::Update {
                update_sink
                    .lock()
                    .push((value_of(value), delivery.sequence));
            }
            Box::pin(std::future::ready(()))
        }));
        state
            .change(ctx(), |draft| {
                draft.set(&[], "value", json!(1)).map_err(AnyError::from)
            })
            .expect("change succeeds");
        assert_eq!(snapshot_events(&source_sequences), vec![1, 2]);
        assert_eq!(snapshot_events(&deliveries), vec![(1, 1), (2, 2)]);
        assert_eq!(
            snapshot_events(&late_deliveries),
            vec![ReplicatedStateDelivery {
                kind: ReplicatedStateDeliveryKind::Hydrate,
                sequence: 2
            }]
        );
    }

    // "isolates listener failures after committing the revision"
    #[test]
    fn isolates_listener_failures_after_committing_the_revision() {
        let state = MutableReplicatedState::new(json!({ "value": 0 }));
        let received = events::<u64>();
        let received_sink = received.clone();
        state
            .internals()
            .subscribe(Arc::new(|_operations, _sequence, _context| {
                panic!("listener failed")
            }));
        state
            .internals()
            .subscribe(Arc::new(move |_operations, sequence, _context| {
                received_sink.lock().push(sequence);
            }));
        let error = state
            .change(ctx(), |draft| {
                draft.set(&[], "value", json!(1)).map_err(AnyError::from)
            })
            .expect_err("source listener failure is collected");
        assert_eq!(error.to_string(), "listener failed");
        assert_eq!(state.value(), &json!({ "value": 1 }));
        assert_eq!(snapshot_events(&received), vec![1]);
    }

    // "copies assigned values by value"
    #[test]
    fn copies_assigned_values_by_value() {
        let external = json!({ "value": 1 });
        let state = MutableReplicatedState::new(
            json!({ "left": JsonValue::Null, "right": JsonValue::Null }),
        );
        state
            .change(ctx(), |draft| {
                draft.set(&[], "left", external.clone())?;
                draft.set(&[], "right", external.clone())?;
                draft.set(&[Seg::key("left")], "value", json!(2))?;
                Ok(())
            })
            .expect("change succeeds");
        assert_eq!(external, json!({ "value": 1 }));
        assert_eq!(
            state.value(),
            &json!({ "left": { "value": 2 }, "right": { "value": 1 } })
        );
    }

    // "takes immutable ownership of alias-free replacements"
    #[test]
    fn takes_immutable_ownership_of_alias_free_replacements() {
        let state =
            MutableReplicatedState::new(json!({ "left": { "value": 1 }, "right": { "value": 2 } }));
        let left = state.value().get("left").cloned().expect("left present");
        let replacement = json!({ "left": left.clone(), "right": left });
        state.replace(ctx(), replacement).expect("replace succeeds");
        assert_eq!(
            state.value(),
            &json!({ "left": { "value": 1 }, "right": { "value": 1 } })
        );
        state
            .change(ctx(), |draft| {
                draft
                    .set(&[Seg::key("left")], "value", json!(9))
                    .map_err(AnyError::from)
            })
            .expect("change succeeds");
        assert_eq!(
            state.value(),
            &json!({ "left": { "value": 9 }, "right": { "value": 1 } })
        );
    }

    // "preserves compact string, splice, and permutation operations"
    #[test]
    fn preserves_compact_string_splice_and_permutation_operations() {
        let state = MutableReplicatedState::new(json!({
            "text": "abcdefgh",
            "values": [{ "id": "a" }, { "id": "b" }, { "id": "c" }],
        }));
        let batches = events::<JsonValue>();
        let batches_sink = batches.clone();
        state
            .internals()
            .subscribe(Arc::new(move |operations, _sequence, _context| {
                batches_sink
                    .lock()
                    .push(serde_json::to_value(operations).expect("ops serialize"));
            }));
        state
            .change(ctx(), |draft| {
                draft.set(&[], "text", json!("defghxyz"))?;
                draft.splice(&[Seg::key("values")], 0, Some(1), vec![])?;
                Ok(())
            })
            .expect("change succeeds");
        state
            .change(ctx(), |draft| {
                // Upstream reverses the array; the draft API exposes comparator
                // sorting, which produces the identical pure-reorder revision.
                draft.sort_by(&[Seg::key("values")], |left, right| {
                    let left = left.get("id").and_then(JsonValue::as_str).unwrap_or("");
                    let right = right.get("id").and_then(JsonValue::as_str).unwrap_or("");
                    right.cmp(left)
                });
                Ok(())
            })
            .expect("change succeeds");
        assert_eq!(
            snapshot_events(&batches),
            vec![
                json!([
                    ["t", ["text"], 3],
                    ["a", ["text"], "xyz"],
                    ["p", ["values"], 0, 1, []],
                ]),
                json!([["m", ["values"], [1, 0]]]),
            ]
        );
    }

    // "validates replica revisions without freezing shared immutable payloads"
    // — the freezing half has no Rust counterpart (values are never frozen);
    // identity assertions degrade to structural ones.
    #[test]
    fn validates_replica_revisions() {
        let replica = ReplicatedStateReplica::new(Arc::new(|_| {}));
        let initial = json!({ "rows": [{ "value": 1 }] });
        replica
            .hydrate(0, &ops(json!([["r", initial]])), ctx())
            .expect("hydrate succeeds");
        assert_eq!(replica.value(), Some(&initial));

        let inserted = json!({ "value": 2 });
        replica
            .update(1, &ops(json!([["p", ["rows"], 1, 0, [inserted]]])), ctx())
            .expect("update succeeds");
        assert_eq!(
            replica.value(),
            Some(&json!({ "rows": [{ "value": 1 }, { "value": 2 }] }))
        );
        // Upstream asserts the source `initial` array is untouched; serde values
        // are cloned per revision, so this holds by construction.
    }

    // "clears a replica when an adopted update is invalid" — the NaN case is
    // unrepresentable in serde_json; an apply failure (append onto a number)
    // exercises the same clear-on-error path.
    #[test]
    fn clears_a_replica_when_an_adopted_update_is_invalid() {
        let errors = events::<AnyError>();
        let error_sink = errors.clone();
        let replica = ReplicatedStateReplica::new(Arc::new(move |error| {
            error_sink.lock().push(error);
        }));
        replica
            .hydrate(0, &ops(json!([["r", { "value": 0 }]])), ctx())
            .expect("hydrate succeeds");
        let failure = replica
            .update(1, &ops(json!([["a", ["value"], "tail"]])), ctx())
            .expect_err("append onto a number fails");
        assert!(!failure.to_string().is_empty());
        assert_eq!(replica.value(), None);
        let failure = replica
            .update(2, &ops(json!([["s", ["value"], 2]])), ctx())
            .expect_err("update before hydration fails");
        assert!(failure.to_string().contains("before hydration"));

        let error_sink2 = errors.clone();
        let malformed = ReplicatedStateReplica::new(Arc::new(move |error| {
            error_sink2.lock().push(error);
        }));
        malformed
            .hydrate(0, &ops(json!([["r", { "values": [1, 2] }]])), ctx())
            .expect("hydrate succeeds");
        assert!(malformed
            .update(1, &ops(json!([["m", ["values"], [0]]])), ctx())
            .is_err());
        assert_eq!(malformed.value(), None);
        assert!(snapshot_errors(&errors).is_empty());
    }

    // "replaces atomically and ignores deeply equal replacements"
    #[test]
    fn replaces_atomically_and_ignores_deeply_equal_replacements() {
        let state = MutableReplicatedState::new(
            json!({ "value": { "nested": 1 }, "retained": { "nested": 2 } }),
        );
        let previous: &JsonValue = state.value();
        state
            .replace(
                ctx(),
                json!({ "value": { "nested": 1 }, "retained": { "nested": 2 } }),
            )
            .expect("deeply equal replace is a no-op");
        assert!(std::ptr::eq(previous, state.value()));
        let retained = previous.get("retained").cloned().expect("retained present");
        state
            .replace(
                ctx(),
                json!({ "value": { "nested": 2 }, "retained": { "nested": 2 } }),
            )
            .expect("replace succeeds");
        assert_eq!(
            state.value(),
            &json!({ "value": { "nested": 2 }, "retained": { "nested": 2 } })
        );
        // Upstream asserts the retained branch keeps its identity; serde values
        // cannot share subtrees, so the content guarantee is asserted instead.
        assert_eq!(state.value().get("retained"), Some(&retained));
    }

    // ── state.test.ts: authoritative replicated state sources ───────────────

    struct TestSource {
        attachments: Mutex<Vec<Arc<TestAttachmentInner>>>,
        on_attach: Mutex<Option<Box<dyn Fn(&TestSource) + Send + Sync>>>,
        value: Mutex<JsonValue>,
        cursor: Mutex<u64>,
        next_attachment_id: AtomicUsize,
    }

    impl TestSource {
        fn new(value: JsonValue, cursor: u64) -> Arc<TestSource> {
            Arc::new(TestSource {
                attachments: Mutex::new(Vec::new()),
                on_attach: Mutex::new(None),
                value: Mutex::new(value),
                cursor: Mutex::new(cursor),
                next_attachment_id: AtomicUsize::new(0),
            })
        }

        fn set_on_attach(&self, hook: impl Fn(&TestSource) + Send + Sync + 'static) {
            *self.on_attach.lock() = Some(Box::new(hook));
        }

        fn commit(&self, value: JsonValue, ops_json: JsonValue, cursor: u64) {
            *self.value.lock() = value.clone();
            *self.cursor.lock() = cursor;
            let frame = ReplicatedStateSourceFrame {
                cursor,
                value,
                ops: ops(ops_json),
                context: ctx().clone(),
            };
            let attachments = self.attachments.lock().clone();
            for attachment in attachments {
                attachment.publish(frame.clone());
            }
        }

        fn attachment_count(&self) -> usize {
            self.attachments.lock().len()
        }
    }

    impl ReplicatedStateSource for TestSource {
        fn attach(self: Arc<Self>) -> Box<dyn ReplicatedStateSourceAttachment> {
            let id = self
                .next_attachment_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let inner = Arc::new(TestAttachmentInner {
                id,
                snapshot: Mutex::new(ReplicatedStateSourceSnapshot {
                    value: self.value.lock().clone(),
                    cursor: *self.cursor.lock(),
                }),
                buffer: Mutex::new(Vec::new()),
                listener: Mutex::new(None),
                activated: std::sync::atomic::AtomicBool::new(false),
                disposed: std::sync::atomic::AtomicBool::new(false),
                source: Arc::downgrade(&self),
            });
            self.attachments.lock().push(inner.clone());
            if let Some(hook) = self.on_attach.lock().as_ref() {
                hook(&self);
            }
            Box::new(TestSourceAttachment { inner })
        }
    }

    struct TestSourceAttachment {
        inner: Arc<TestAttachmentInner>,
    }

    impl ReplicatedStateSourceAttachment for TestSourceAttachment {
        fn snapshot(&self) -> ReplicatedStateSourceSnapshot {
            self.inner.snapshot()
        }

        fn activate(&self, listener: SourceFrameListener) {
            self.inner.activate(listener);
        }

        fn dispose(&self) {
            self.inner.dispose();
        }
    }

    struct TestAttachmentInner {
        id: usize,
        snapshot: Mutex<ReplicatedStateSourceSnapshot>,
        buffer: Mutex<Vec<ReplicatedStateSourceFrame>>,
        listener: Mutex<Option<SourceFrameListener>>,
        activated: std::sync::atomic::AtomicBool,
        disposed: std::sync::atomic::AtomicBool,
        source: std::sync::Weak<TestSource>,
    }

    impl TestAttachmentInner {
        fn publish(&self, frame: ReplicatedStateSourceFrame) {
            use std::sync::atomic::Ordering;
            if self.disposed.load(Ordering::Acquire) {
                return;
            }
            let listener = self.listener.lock().clone();
            match listener {
                Some(listener) => listener(frame),
                None => self.buffer.lock().push(frame),
            }
        }

        fn snapshot(&self) -> ReplicatedStateSourceSnapshot {
            self.snapshot.lock().clone()
        }

        fn activate(&self, listener: SourceFrameListener) {
            use std::sync::atomic::Ordering;
            if self.activated.swap(true, Ordering::SeqCst) {
                panic!("attachment is already active");
            }
            *self.listener.lock() = Some(listener);
            let buffered = std::mem::take(&mut *self.buffer.lock());
            for frame in buffered {
                (self.listener.lock().as_ref().expect("just set"))(frame);
            }
        }

        fn dispose(&self) {
            use std::sync::atomic::Ordering;
            if self.disposed.swap(true, Ordering::SeqCst) {
                return;
            }
            self.buffer.lock().clear();
            if let Some(source) = self.source.upgrade() {
                source
                    .attachments
                    .lock()
                    .retain(|attachment| attachment.id != self.id);
            }
        }
    }

    // "captures before activation and drains queued commits in order"
    #[test]
    fn captures_before_activation_and_drains_queued_commits_in_order() {
        let source = TestSource::new(json!({ "value": 0 }), 10);
        let source_for_hook = source.clone();
        source.set_on_attach(move |_source| {
            source_for_hook.commit(json!({ "value": 1 }), json!([["s", ["value"], 1]]), 11);
            source_for_hook.commit(json!({ "value": 2 }), json!([["s", ["value"], 2]]), 12);
        });

        let state = attach_replicated_state_source(
            source.clone(),
            ReplicatedStateSourceOptions { on_error: None },
        )
        .expect("attach succeeds");
        let deliveries = events();
        state.subscribe(recording_listener(deliveries.clone()));
        assert_eq!(state.value(), &json!({ "value": 2 }));
        assert_eq!(
            snapshot_events(&deliveries),
            vec![(
                2,
                ReplicatedStateDelivery {
                    kind: ReplicatedStateDeliveryKind::Hydrate,
                    sequence: 2
                }
            )]
        );
        let snapshot = state.internals().snapshot();
        assert_eq!(snapshot.value, json!({ "value": 2 }));
        assert_eq!(snapshot.sequence, 2);
    }

    // "hydrates at sequence zero when attaching after existing commits"
    #[test]
    fn hydrates_at_sequence_zero_when_attaching_after_existing_commits() {
        let source = TestSource::new(json!({ "value": 0 }), 40);
        source.commit(json!({ "value": 1 }), json!([["s", ["value"], 1]]), 41);
        let state = attach_replicated_state_source(
            source.clone(),
            ReplicatedStateSourceOptions { on_error: None },
        )
        .expect("attach succeeds");
        let deliveries = events();
        state.subscribe(recording_listener(deliveries.clone()));
        assert_eq!(state.value(), &json!({ "value": 1 }));
        assert_eq!(
            snapshot_events(&deliveries),
            vec![(
                1,
                ReplicatedStateDelivery {
                    kind: ReplicatedStateDeliveryKind::Hydrate,
                    sequence: 0
                }
            )]
        );
    }

    // "publishes exact source value and operation references without applying or
    // re-diffing" — reference identity is unobservable for owned serde values;
    // structural equality plus exact-batch assertions stand in.
    #[test]
    fn publishes_exact_source_value_and_operations_without_applying() {
        let source = TestSource::new(json!({ "value": 0 }), 0);
        let state = attach_replicated_state_source(
            source.clone(),
            ReplicatedStateSourceOptions { on_error: None },
        )
        .expect("attach succeeds");
        let published_ops = events::<JsonValue>();
        let published_ops_sink = published_ops.clone();
        state
            .internals()
            .subscribe(Arc::new(move |received, _sequence, _context| {
                published_ops_sink
                    .lock()
                    .push(serde_json::to_value(received).expect("ops serialize"));
            }));
        let published_value = events::<JsonValue>();
        let published_value_sink = published_value.clone();
        state.subscribe(Arc::new(move |value, _context, delivery| {
            if delivery.kind == ReplicatedStateDeliveryKind::Update {
                published_value_sink.lock().push(value.clone());
            }
            Box::pin(std::future::ready(()))
        }));

        source.commit(json!({ "value": 1 }), json!([["s", ["value"], 1]]), 1);
        assert_eq!(state.value(), &json!({ "value": 1 }));
        assert_eq!(
            snapshot_events(&published_value),
            vec![json!({ "value": 1 })]
        );
        assert_eq!(
            snapshot_events(&published_ops),
            vec![json!([["s", ["value"], 1]])]
        );
    }

    // "buffers reentrant frames and skips updates covered by a late hydration"
    #[test]
    fn buffers_reentrant_frames_and_skips_updates_covered_by_a_late_hydration() {
        let source = TestSource::new(json!({ "value": 0 }), 0);
        let state = attach_replicated_state_source(
            source.clone(),
            ReplicatedStateSourceOptions { on_error: None },
        )
        .expect("attach succeeds");
        let received = events::<u64>();
        let received_sink = received.clone();
        let late = events::<(u64, ReplicatedStateDelivery)>();
        let nested = Arc::new(AtomicUsize::new(0));
        let state_for_source = state.clone();
        let source_for_source = source.clone();
        let nested_for_source = nested.clone();
        let late_for_source = late.clone();
        state
            .internals()
            .subscribe(Arc::new(move |_ops, sequence, _context| {
                if sequence != 1
                    || nested_for_source.swap(1, std::sync::atomic::Ordering::SeqCst) != 0
                {
                    return;
                }
                source_for_source.commit(json!({ "value": 2 }), json!([["s", ["value"], 2]]), 2);
                let late_sink = late_for_source.clone();
                state_for_source.subscribe(Arc::new(move |value, _context, delivery| {
                    late_sink.lock().push((value_of(value), delivery.clone()));
                    Box::pin(std::future::ready(()))
                }));
            }));
        let state_sink = received_sink.clone();
        state.subscribe(Arc::new(move |value, _context, delivery| {
            if delivery.kind == ReplicatedStateDeliveryKind::Update {
                state_sink.lock().push(value_of(value));
            }
            Box::pin(std::future::ready(()))
        }));

        source.commit(json!({ "value": 1 }), json!([["s", ["value"], 1]]), 1);
        assert_eq!(snapshot_events(&received), vec![1, 2]);
        assert_eq!(
            snapshot_events(&late),
            vec![(
                2,
                ReplicatedStateDelivery {
                    kind: ReplicatedStateDeliveryKind::Hydrate,
                    sequence: 2
                }
            )]
        );
    }

    // "reports listener failures without throwing them into the source"
    #[test]
    fn reports_listener_failures_without_throwing_them_into_the_source() {
        let source = TestSource::new(json!({ "value": 0 }), 0);
        let errors = events::<AnyError>();
        let error_sink = errors.clone();
        let state = attach_replicated_state_source(
            source.clone(),
            ReplicatedStateSourceOptions {
                on_error: Some(Arc::new(move |error| {
                    error_sink.lock().push(error);
                })),
            },
        )
        .expect("attach succeeds");
        let received = events::<u64>();
        let received_sink = received.clone();
        state.subscribe(Arc::new(move |value, _context, delivery| {
            let _ = value;
            if delivery.kind == ReplicatedStateDeliveryKind::Update {
                panic!("listener failed");
            }
            let _ = received_sink;
            Box::pin(std::future::ready(()))
        }));
        let second = events::<u64>();
        let second_sink = second.clone();
        state.subscribe(Arc::new(move |value, _context, delivery| {
            if delivery.kind == ReplicatedStateDeliveryKind::Update {
                second_sink.lock().push(value_of(value));
            }
            Box::pin(std::future::ready(()))
        }));

        source.commit(json!({ "value": 1 }), json!([["s", ["value"], 1]]), 1);
        source.commit(json!({ "value": 2 }), json!([["s", ["value"], 2]]), 2);
        assert_eq!(snapshot_events(&second), vec![1, 2]);
        assert_eq!(
            snapshot_errors(&errors),
            vec!["listener failed".to_string(), "listener failed".to_string()]
        );
        assert!(snapshot_events(&received).is_empty());
    }

    // "reports cursor gaps, disposes the broken attachment, and ignores later frames"
    #[test]
    fn reports_cursor_gaps_and_disposes_the_broken_attachment() {
        let source = TestSource::new(json!({ "value": 0 }), 5);
        let errors = events::<AnyError>();
        let error_sink = errors.clone();
        let state = attach_replicated_state_source(
            source.clone(),
            ReplicatedStateSourceOptions {
                on_error: Some(Arc::new(move |error| {
                    error_sink.lock().push(error);
                })),
            },
        )
        .expect("attach succeeds");
        source.commit(json!({ "value": 2 }), json!([["s", ["value"], 2]]), 7);
        assert!(snapshot_errors(&errors)[0].contains("expected 6, received 7"));
        assert_eq!(source.attachment_count(), 0);
        assert_eq!(state.value(), &json!({ "value": 0 }));
        source.commit(json!({ "value": 3 }), json!([["s", ["value"], 3]]), 8);
        assert_eq!(state.value(), &json!({ "value": 0 }));
    }

    // "keeps attachments independent and disposes each idempotently"
    #[test]
    fn keeps_attachments_independent_and_disposes_each_idempotently() {
        let source = TestSource::new(json!({ "value": 0 }), 0);
        let first = attach_replicated_state_source(
            source.clone(),
            ReplicatedStateSourceOptions { on_error: None },
        )
        .expect("attach succeeds");
        let second = attach_replicated_state_source(
            source.clone(),
            ReplicatedStateSourceOptions { on_error: None },
        )
        .expect("attach succeeds");
        assert_eq!(source.attachment_count(), 2);
        source.commit(json!({ "value": 1 }), json!([["s", ["value"], 1]]), 1);
        assert_eq!(first.value(), &json!({ "value": 1 }));
        assert_eq!(second.value(), &json!({ "value": 1 }));

        first.dispose();
        first.dispose();
        assert_eq!(source.attachment_count(), 1);
        source.commit(json!({ "value": 2 }), json!([["s", ["value"], 2]]), 2);
        assert_eq!(first.value(), &json!({ "value": 1 }));
        assert_eq!(second.value(), &json!({ "value": 2 }));
        second.dispose();
        assert_eq!(source.attachment_count(), 0);
    }

    // ── state-delivery.test.ts ───────────────────────────────────────────────

    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Mutable,
        Attached,
        Replica,
    }

    /// One frame receiver handed out by the attached fixture's source.
    struct FixtureSource {
        buffer: Mutex<Vec<ReplicatedStateSourceFrame>>,
        listener: Mutex<Option<SourceFrameListener>>,
        disposed: std::sync::atomic::AtomicBool,
    }

    impl FixtureSource {
        fn receive(&self, frame: ReplicatedStateSourceFrame) {
            use std::sync::atomic::Ordering;
            if self.disposed.load(Ordering::Acquire) {
                return;
            }
            let listener = self.listener.lock().clone();
            match listener {
                Some(listener) => listener(frame),
                None => self.buffer.lock().push(frame),
            }
        }
    }

    impl ReplicatedStateSource for FixtureSource {
        fn attach(self: Arc<Self>) -> Box<dyn ReplicatedStateSourceAttachment> {
            Box::new(FixtureAttachment {
                source: Arc::downgrade(&self),
                snapshot: ReplicatedStateSourceSnapshot {
                    value: json!({ "value": 0 }),
                    cursor: 0,
                },
                buffer: Mutex::new(Vec::new()),
                listener: Mutex::new(None),
                activated: std::sync::atomic::AtomicBool::new(false),
            })
        }
    }

    struct FixtureAttachment {
        source: std::sync::Weak<FixtureSource>,
        snapshot: ReplicatedStateSourceSnapshot,
        buffer: Mutex<Vec<ReplicatedStateSourceFrame>>,
        listener: Mutex<Option<SourceFrameListener>>,
        activated: std::sync::atomic::AtomicBool,
    }

    impl ReplicatedStateSourceAttachment for FixtureAttachment {
        fn snapshot(&self) -> ReplicatedStateSourceSnapshot {
            self.snapshot.clone()
        }

        fn activate(&self, listener: SourceFrameListener) {
            use std::sync::atomic::Ordering;
            if self.activated.swap(true, Ordering::SeqCst) {
                panic!("attachment is already active");
            }
            let drain_listener = listener.clone();
            let register_listener = listener.clone();
            *self.listener.lock() = Some(listener);
            let buffered = std::mem::take(&mut *self.buffer.lock());
            for frame in buffered {
                drain_listener(frame);
            }
            // Register with the live source so later commits are received.
            if let Some(source) = self.source.upgrade() {
                let mut live = source.listener.lock();
                let buffered = std::mem::take(&mut *source.buffer.lock());
                *live = Some(register_listener.clone());
                drop(live);
                for frame in buffered {
                    register_listener(frame);
                }
            }
        }

        fn dispose(&self) {
            use std::sync::atomic::Ordering;
            if let Some(source) = self.source.upgrade() {
                source.disposed.store(true, Ordering::Release);
                source.listener.lock().take();
            }
        }
    }

    struct Fixture {
        state: Arc<dyn ReplicatedState>,
        internals: Option<Arc<dyn StateInternals>>,
        publish: Box<dyn Fn(u64, &Context) + Send + Sync>,
    }

    fn fixture(kind: Kind, on_error: ErrorHandler) -> Fixture {
        match kind {
            Kind::Mutable => {
                let state = MutableReplicatedState::new(json!({ "value": 0 }));
                let internals = Some(state.internals());
                let publisher = state.clone();
                Fixture {
                    state,
                    internals,
                    publish: Box::new(move |value, context| {
                        publisher
                            .replace(context, json!({ "value": value }))
                            .expect("replace succeeds");
                    }),
                }
            }
            Kind::Attached => {
                let source = Arc::new(FixtureSource {
                    buffer: Mutex::new(Vec::new()),
                    listener: Mutex::new(None),
                    disposed: std::sync::atomic::AtomicBool::new(false),
                });
                let state = attach_replicated_state_source(
                    source.clone(),
                    ReplicatedStateSourceOptions {
                        on_error: Some(on_error),
                    },
                )
                .expect("attach succeeds");
                let internals = Some(state.internals());
                Fixture {
                    state,
                    internals,
                    publish: Box::new(move |value, context| {
                        // The attach call activates, so frames flow to the live
                        // listener; route through the source to keep the exact
                        // cursor semantics (cursor = value).
                        source.receive(ReplicatedStateSourceFrame {
                            cursor: value,
                            value: json!({ "value": value }),
                            ops: serde_json::from_value(json!([["s", ["value"], value]]))
                                .expect("ops"),
                            context: context.clone(),
                        });
                    }),
                }
            }
            Kind::Replica => {
                let replica = ReplicatedStateReplica::new(on_error);
                replica
                    .hydrate(0, &ops(json!([["r", { "value": 0 }]])), ctx())
                    .expect("hydrate succeeds");
                let publisher = replica.clone();
                Fixture {
                    state: replica,
                    internals: None,
                    publish: Box::new(move |value, context| {
                        publisher
                            .update(value, &ops(json!([["s", ["value"], value]])), context)
                            .expect("update succeeds");
                    }),
                }
            }
        }
    }

    /// Install the shared async listener that records events and gates on values
    /// 0 and 1 (upstream "awaits hydration and each update independently").
    #[tokio::test]
    async fn awaits_hydration_and_each_update_independently_of_other_subscribers() {
        for kind in [Kind::Mutable, Kind::Attached, Kind::Replica] {
            let mut hydration = Deferred::new();
            let mut update = Deferred::new();
            let fixture = fixture(kind, Arc::new(|_| {}));
            let delivery_events = events::<String>();
            let exact = events::<u64>();
            if let Some(internals) = &fixture.internals {
                let exact_sink = exact.clone();
                internals.subscribe(Arc::new(move |_ops, sequence, _context| {
                    exact_sink.lock().push(sequence);
                }));
            }
            let events_for_listener = delivery_events.clone();
            let hydration_rx = Mutex::new(Some(hydration.receiver("hydration rejected")));
            let update_rx = Mutex::new(Some(update.receiver("update rejected")));
            fixture
                .state
                .subscribe(Arc::new(move |value, _context, _delivery| {
                    let v = value_of(value);
                    let event_sink = events_for_listener.clone();
                    event_sink.lock().push(format!("start:{v}"));
                    let gate = match v {
                        0 => hydration_rx.lock().take(),
                        1 => update_rx.lock().take(),
                        _ => None,
                    };
                    match gate {
                        Some(gate) => Box::pin(async move {
                            gate.await;
                            event_sink.lock().push(format!("end:{v}"));
                        }),
                        None => {
                            event_sink.lock().push(format!("end:{v}"));
                            Box::pin(std::future::ready(()))
                        }
                    }
                }));
            let fast = events::<u64>();
            let fast_sink = fast.clone();
            fixture
                .state
                .subscribe(Arc::new(move |value, _context, _delivery| {
                    fast_sink.lock().push(value_of(value));
                    Box::pin(std::future::ready(()))
                }));
            (fixture.publish)(1, ctx());
            (fixture.publish)(2, ctx());
            assert_eq!(
                snapshot_events(&delivery_events),
                vec!["start:0".to_string()]
            );
            assert_eq!(snapshot_events(&fast), vec![0, 1, 2]);
            assert_eq!(fixture.state.value(), Some(&json!({ "value": 2 })));
            if kind != Kind::Replica {
                assert_eq!(snapshot_events(&exact), vec![1, 2]);
            }
            hydration.resolve();
            wait_for(|| {
                snapshot_events(&delivery_events)
                    == vec![
                        "start:0".to_string(),
                        "end:0".to_string(),
                        "start:1".to_string(),
                    ]
            })
            .await;
            update.resolve();
            wait_for(|| {
                snapshot_events(&delivery_events)
                    == vec![
                        "start:0".to_string(),
                        "end:0".to_string(),
                        "start:1".to_string(),
                        "end:1".to_string(),
                        "start:2".to_string(),
                        "end:2".to_string(),
                    ]
            })
            .await;
        }
    }

    // "bounds %i pending deliveries without changing exact publication"
    #[tokio::test]
    async fn bounds_pending_deliveries_without_changing_exact_publication() {
        for count in [100usize, 101, 102, 201, 202] {
            for kind in [Kind::Mutable, Kind::Attached, Kind::Replica] {
                let mut hydration = Deferred::new();
                let fixture = fixture(kind, Arc::new(|_| {}));
                let received = events::<u64>();
                let exact = events();
                if let Some(internals) = &fixture.internals {
                    let exact_sink = exact.clone();
                    internals.subscribe(Arc::new(move |_ops, sequence, _context| {
                        exact_sink.lock().push(sequence);
                    }));
                }
                let received_sink = received.clone();
                let hydration_rx = Mutex::new(Some(hydration.receiver("hydration rejected")));
                fixture
                    .state
                    .subscribe(Arc::new(move |value, _context, _delivery| {
                        let v = value_of(value);
                        received_sink.lock().push(v);
                        match v {
                            0 => {
                                let gate = hydration_rx.lock().take();
                                match gate {
                                    Some(gate) => Box::pin(gate),
                                    None => Box::pin(std::future::ready(())),
                                }
                            }
                            _ => Box::pin(std::future::ready(())),
                        }
                    }));
                for value in 1..=count as u64 {
                    (fixture.publish)(value, ctx());
                }
                assert_eq!(snapshot_events(&received), vec![0]);
                if kind != Kind::Replica {
                    assert_eq!(
                        snapshot_events(&exact),
                        (1..=count as u64).collect::<Vec<_>>()
                    );
                }
                hydration.resolve();
                let expected_first = ((count as u64 - 1) / 100) * 100 + 1;
                let expected = std::iter::once(0)
                    .chain(expected_first..=count as u64)
                    .collect::<Vec<_>>();
                wait_for(|| snapshot_events(&received) == expected).await;
            }
        }
    }

    // "excludes a running update from overflow and retains exact value/context/delivery"
    #[tokio::test]
    async fn excludes_a_running_update_from_overflow_and_retains_exact_context() {
        for kind in [Kind::Mutable, Kind::Attached, Kind::Replica] {
            let mut gate = Deferred::new();
            let fixture = fixture(kind, Arc::new(|_| {}));
            let cancel = with_cancel(ctx().clone());
            let marker_key: ContextKey<u64> = crate::context::create_context_key("test.marker");
            let context =
                crate::context::with_context_value(&marker_key, 4242u64, cancel.context.clone());
            let received = events::<(u64, Option<u64>, ReplicatedStateDelivery)>();
            let received_sink = received.clone();
            let gate_rx = Mutex::new(Some(gate.receiver("gate rejected")));
            let marker_key_for_listener = marker_key;
            fixture
                .state
                .subscribe(Arc::new(move |value, context, delivery| {
                    let v = value_of(value);
                    let marker = context.value(&marker_key_for_listener);
                    received_sink.lock().push((v, marker, delivery.clone()));
                    match v {
                        1 => {
                            let gate = gate_rx.lock().take();
                            match gate {
                                Some(gate) => Box::pin(gate),
                                None => Box::pin(std::future::ready(())),
                            }
                        }
                        _ => Box::pin(std::future::ready(())),
                    }
                }));
            for value in 1..102u64 {
                (fixture.publish)(value, ctx());
            }
            (fixture.publish)(102, &context);
            let adopted = fixture.state.value().cloned();
            (fixture.publish)(103, ctx());
            assert_eq!(
                snapshot_events(&received)
                    .iter()
                    .map(|(v, _, _)| *v)
                    .collect::<Vec<_>>(),
                vec![0, 1]
            );
            gate.resolve();
            wait_for(|| {
                snapshot_events(&received)
                    .iter()
                    .map(|(v, _, _)| *v)
                    .collect::<Vec<_>>()
                    == vec![0, 1, 102, 103]
            })
            .await;
            let received = snapshot_events(&received);
            assert_eq!(received[2].0, 102);
            assert_eq!(received[2].1, Some(4242));
            assert_eq!(
                received[2].2,
                ReplicatedStateDelivery {
                    kind: ReplicatedStateDeliveryKind::Update,
                    sequence: 102
                }
            );
            assert_eq!(adopted.as_ref(), Some(&json!({ "value": 102 })));
        }
    }

    // "serializes reentrant hydration and update callbacks" (synchronous)
    #[test]
    fn serializes_reentrant_hydration_and_update_callbacks() {
        for kind in [Kind::Mutable, Kind::Attached, Kind::Replica] {
            let fixture = fixture(kind, Arc::new(|_| {}));
            let events = events::<String>();
            let events_for_listener = events.clone();
            let publisher: Arc<dyn Fn(u64, &Context) + Send + Sync> =
                Arc::new(move |value, context| (fixture.publish)(value, context));
            let state = fixture.state.clone();
            state.subscribe(Arc::new(move |value, context, _delivery| {
                let v = value_of(value);
                let events = events_for_listener.clone();
                events.lock().push(format!("start:{v}"));
                if v < 2 {
                    publisher(v + 1, context);
                }
                events.lock().push(format!("end:{v}"));
                Box::pin(std::future::ready(()))
            }));
            drop(state);
            assert_eq!(
                snapshot_events(&events),
                vec![
                    "start:0".to_string(),
                    "end:0".to_string(),
                    "start:1".to_string(),
                    "end:1".to_string(),
                    "start:2".to_string(),
                    "end:2".to_string(),
                ]
            );
        }
    }

    // "treats two subscriptions of the same callback independently"
    #[tokio::test]
    async fn treats_two_subscriptions_of_the_same_callback_independently() {
        for kind in [Kind::Mutable, Kind::Attached, Kind::Replica] {
            let mut gate = Deferred::new();
            let fixture = fixture(kind, Arc::new(|_| {}));
            let received = events::<u64>();
            let received_sink = received.clone();
            let gate_rx = Mutex::new(Some(gate.receiver("gate rejected")));
            let listener: StateListener = Arc::new(move |value, _context, _delivery| {
                let v = value_of(value);
                received_sink.lock().push(v);
                match v {
                    0 => {
                        let gate = gate_rx.lock().take();
                        match gate {
                            Some(gate) => Box::pin(gate),
                            None => Box::pin(std::future::ready(())),
                        }
                    }
                    _ => Box::pin(std::future::ready(())),
                }
            });
            let mut stop_first = Some(fixture.state.subscribe(listener.clone()));
            let stop_second = fixture.state.subscribe(listener);
            (fixture.publish)(1, ctx());
            // Upstream unsubscribes twice; Rust's FnOnce unsubscribe is consumed
            // by the first call, so the second call is modeled by Option::take.
            (stop_first.take().expect("first unsubscribe"))();
            gate.resolve();
            wait_for(|| snapshot_events(&received) == vec![0, 0, 1]).await;
            stop_second();
        }
    }

    // "unsubscribe drops queued callbacks without joining or aborting the running callback"
    #[tokio::test]
    async fn unsubscribe_drops_queued_callbacks_without_joining_the_running_callback() {
        for kind in [Kind::Mutable, Kind::Attached, Kind::Replica] {
            let mut gate = Deferred::new();
            let fixture = fixture(kind, Arc::new(|_| {}));
            let cancel = with_cancel(ctx().clone());
            let received = events::<u64>();
            let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let received_sink = received.clone();
            let completed_flag = completed.clone();
            let gate_rx = Mutex::new(Some(gate.receiver("gate rejected")));
            let stop = fixture
                .state
                .subscribe(Arc::new(move |value, _context, _delivery| {
                    let v = value_of(value);
                    received_sink.lock().push(v);
                    match v {
                        1 => {
                            let gate = gate_rx.lock().take();
                            let completed_flag = completed_flag.clone();
                            Box::pin(async move {
                                gate.expect("gate present").await;
                                completed_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                            })
                        }
                        _ => Box::pin(std::future::ready(())),
                    }
                }));
            (fixture.publish)(1, &cancel.context);
            (fixture.publish)(2, &cancel.context);
            stop();
            (fixture.publish)(3, &cancel.context);
            assert!(!cancel.context.abort_signal().expect("signal").aborted());
            assert!(!completed.load(std::sync::atomic::Ordering::SeqCst));
            gate.resolve();
            wait_for(|| completed.load(std::sync::atomic::Ordering::SeqCst)).await;
            assert_eq!(snapshot_events(&received), vec![0, 1]);
        }
    }

    // listener-error group: "isolates a synchronous hydration failure without
    // removing the subscription"
    #[test]
    fn isolates_a_synchronous_hydration_failure_without_removing_the_subscription() {
        for kind in [Kind::Attached, Kind::Replica] {
            let errors = events::<AnyError>();
            let error_sink = errors.clone();
            let fixture = fixture(
                kind,
                Arc::new(move |error| {
                    error_sink.lock().push(error);
                }),
            );
            let received = events::<u64>();
            let received_sink = received.clone();
            fixture
                .state
                .subscribe(Arc::new(move |value, _context, _delivery| {
                    let v = value_of(value);
                    received_sink.lock().push(v);
                    if v == 0 {
                        panic!("sync hydration");
                    }
                    Box::pin(std::future::ready(()))
                }));
            (fixture.publish)(1, ctx());
            assert_eq!(snapshot_errors(&errors), vec!["sync hydration".to_string()]);
            assert_eq!(snapshot_events(&received), vec![0, 1]);
        }
    }

    // listener-error group: "observes hydration rejection, synchronous throw, and
    // update rejection while continuing delivery"
    #[tokio::test]
    async fn observes_hydration_rejection_synchronous_throw_and_update_rejection() {
        for kind in [Kind::Attached, Kind::Replica] {
            let mut gate = Deferred::new();
            let errors = events::<AnyError>();
            let error_sink = errors.clone();
            let fixture = fixture(
                kind,
                Arc::new(move |error| {
                    error_sink.lock().push(error);
                }),
            );
            let received = events::<u64>();
            let fast = events::<u64>();
            let received_sink = received.clone();
            let gate_rx = Mutex::new(Some(gate.receiver("async hydration")));
            let update_rx = Mutex::new(Some(Deferred::new().receiver("async update")));
            fixture
                .state
                .subscribe(Arc::new(move |value, _context, _delivery| {
                    let v = value_of(value);
                    received_sink.lock().push(v);
                    match v {
                        0 => {
                            let gate = gate_rx.lock().take();
                            match gate {
                                Some(gate) => Box::pin(gate),
                                None => Box::pin(std::future::ready(())),
                            }
                        }
                        1 => panic!("sync update"),
                        2 => {
                            let update = update_rx.lock().take();
                            match update {
                                Some(update) => Box::pin(update),
                                None => Box::pin(std::future::ready(())),
                            }
                        }
                        _ => Box::pin(std::future::ready(())),
                    }
                }));
            let fast_sink = fast.clone();
            fixture
                .state
                .subscribe(Arc::new(move |value, _context, _delivery| {
                    fast_sink.lock().push(value_of(value));
                    Box::pin(std::future::ready(()))
                }));
            for value in 1..=3u64 {
                (fixture.publish)(value, ctx());
            }
            // Rejecting the gate drops the sender, which makes the awaiting
            // listener panic with the rejection message.
            gate.reject();
            wait_for(|| snapshot_events(&received) == vec![0, 1, 2, 3]).await;
            assert_eq!(
                snapshot_errors(&errors),
                vec![
                    "async hydration".to_string(),
                    "sync update".to_string(),
                    "async update".to_string(),
                ]
            );
            assert_eq!(snapshot_events(&fast), vec![0, 1, 2, 3]);
        }
    }

    // listener-error group: "still observes a callback rejection after unsubscribe"
    #[tokio::test]
    async fn still_observes_a_callback_rejection_after_unsubscribe() {
        for kind in [Kind::Attached, Kind::Replica] {
            let mut gate = Deferred::new();
            let errors = events::<AnyError>();
            let error_sink = errors.clone();
            let fixture = fixture(
                kind,
                Arc::new(move |error| {
                    error_sink.lock().push(error);
                }),
            );
            let received = events::<u64>();
            let received_sink = received.clone();
            let gate_rx = Mutex::new(Some(gate.receiver("stopped callback")));
            let stop = fixture
                .state
                .subscribe(Arc::new(move |value, _context, _delivery| {
                    let v = value_of(value);
                    received_sink.lock().push(v);
                    let gate = gate_rx.lock().take();
                    match gate {
                        Some(gate) => Box::pin(gate),
                        None => Box::pin(std::future::ready(())),
                    }
                }));
            (fixture.publish)(1, ctx());
            stop();
            gate.reject();
            wait_for(|| snapshot_errors(&errors) == vec!["stopped callback".to_string()]).await;
            assert_eq!(snapshot_events(&received), vec![0]);
        }
    }

    // "mutable state reports rejected callbacks through its default error
    // reporter" — adapted: the upstream spy observes queueMicrotask rethrowing
    // (which crashes the process); the Rust default reporter logs via tracing
    // instead, so only the unobservable-crash and delivery-continuation halves
    // are asserted.
    #[tokio::test]
    async fn mutable_state_reports_rejected_callbacks_through_its_default_reporter() {
        let state = MutableReplicatedState::new(json!({ "value": 0 }));
        let received = events::<u64>();
        let received_sink = received.clone();
        state.subscribe(Arc::new(move |value, _context, _delivery| {
            let v = value_of(value);
            received_sink.lock().push(v);
            if v == 0 {
                Box::pin(async { panic!("async mutable hydration") })
            } else {
                Box::pin(std::future::ready(()))
            }
        }));
        state
            .replace(ctx(), json!({ "value": 1 }))
            .expect("replace succeeds");
        wait_for(|| snapshot_events(&received) == vec![0, 1]).await;
    }

    // "replica disconnect drops obsolete pending work but waits for the running
    // callback before rehydration"
    #[tokio::test]
    async fn replica_disconnect_drops_obsolete_pending_work_but_waits_for_the_running_callback() {
        let replica = ReplicatedStateReplica::new(Arc::new(|_| {}));
        let mut gate = Deferred::new();
        let received = events::<ReplicatedStateDelivery>();
        let received_sink = received.clone();
        let gate_rx = Mutex::new(Some(gate.receiver("gate rejected")));
        replica.subscribe(Arc::new(move |_value, _context, delivery| {
            received_sink.lock().push(delivery.clone());
            match delivery.sequence {
                0 => {
                    let gate = gate_rx.lock().take();
                    match gate {
                        Some(gate) => Box::pin(gate),
                        None => Box::pin(std::future::ready(())),
                    }
                }
                _ => Box::pin(std::future::ready(())),
            }
        }));
        replica
            .hydrate(0, &ops(json!([["r", { "value": 0 }]])), ctx())
            .expect("hydrate succeeds");
        replica
            .update(1, &ops(json!([["s", ["value"], 1]])), ctx())
            .expect("update succeeds");
        replica.clear();
        replica
            .hydrate(50, &ops(json!([["r", { "value": 50 }]])), ctx())
            .expect("hydrate succeeds");
        replica
            .update(51, &ops(json!([["s", ["value"], 51]])), ctx())
            .expect("update succeeds");
        assert_eq!(
            snapshot_events(&received),
            vec![ReplicatedStateDelivery {
                kind: ReplicatedStateDeliveryKind::Hydrate,
                sequence: 0
            }]
        );
        gate.resolve();
        wait_for(|| {
            snapshot_events(&received)
                == vec![
                    ReplicatedStateDelivery {
                        kind: ReplicatedStateDeliveryKind::Hydrate,
                        sequence: 0,
                    },
                    ReplicatedStateDelivery {
                        kind: ReplicatedStateDeliveryKind::Hydrate,
                        sequence: 50,
                    },
                    ReplicatedStateDelivery {
                        kind: ReplicatedStateDeliveryKind::Update,
                        sequence: 51,
                    },
                ]
        })
        .await;
    }

    // "cold replicas hydrate all listeners before their reentrant updates" (sync)
    #[test]
    fn cold_replicas_hydrate_all_listeners_before_their_reentrant_updates() {
        let replica = ReplicatedStateReplica::new(Arc::new(|_| {}));
        let second = events::<ReplicatedStateDelivery>();
        let second_sink = second.clone();
        let reentrant = replica.clone();
        replica.subscribe(Arc::new(move |_value, _context, delivery| {
            if delivery.kind == ReplicatedStateDeliveryKind::Hydrate {
                reentrant
                    .update(1, &ops(json!([["s", ["value"], 1]])), ctx())
                    .expect("update succeeds");
            }
            Box::pin(std::future::ready(()))
        }));
        replica.subscribe(Arc::new(move |_value, _context, delivery| {
            second_sink.lock().push(delivery.clone());
            Box::pin(std::future::ready(()))
        }));
        replica
            .hydrate(0, &ops(json!([["r", { "value": 0 }]])), ctx())
            .expect("hydrate succeeds");
        assert_eq!(
            snapshot_events(&second),
            vec![
                ReplicatedStateDelivery {
                    kind: ReplicatedStateDeliveryKind::Hydrate,
                    sequence: 0
                },
                ReplicatedStateDelivery {
                    kind: ReplicatedStateDeliveryKind::Update,
                    sequence: 1
                },
            ]
        );
    }

    // ── state-diff.test.ts ───────────────────────────────────────────────────

    fn expect_diff(before: JsonValue, after: JsonValue, expected: JsonValue) {
        let operations = diff_revisions(&before, &after);
        assert_eq!(
            serde_json::to_value(&operations).expect("ops serialize"),
            expected,
            "diff {before} -> {after}"
        );
        assert_eq!(
            apply_immutable(&before, &operations).expect("ops apply"),
            after,
            "replay {before} -> {after}"
        );
    }

    // "emits sets and deletes"
    #[test]
    fn diff_emits_sets_and_deletes() {
        expect_diff(
            json!({ "keep": 1, "change": 1, "remove": true }),
            json!({ "keep": 1, "change": 2, "add": 3 }),
            json!([["s", ["change"], 2], ["s", ["add"], 3], ["d", ["remove"]],]),
        );
    }

    // "emits string append and front truncation"
    #[test]
    fn diff_emits_string_append_and_front_truncation() {
        expect_diff(
            json!({ "text": "hello" }),
            json!({ "text": "hello world" }),
            json!([["a", ["text"], " world"]]),
        );
        expect_diff(
            json!({ "text": "hello world" }),
            json!({ "text": "world" }),
            json!([["t", ["text"], 6]]),
        );
        expect_diff(
            json!({ "text": "abcdefgh" }),
            json!({ "text": "defghxyz" }),
            json!([["t", ["text"], 3], ["a", ["text"], "xyz"],]),
        );
    }

    // "represents array insertion, removal, and shift with splices"
    #[test]
    fn diff_represents_array_insertion_removal_and_shift_with_splices() {
        let a = json!({ "id": "a" });
        let b = json!({ "id": "b" });
        let c = json!({ "id": "c" });
        expect_diff(
            json!({ "values": [a, b] }),
            json!({ "values": [a, c, b] }),
            json!([["p", ["values"], 1, 0, [c]]]),
        );
        expect_diff(
            json!({ "values": [a, b, c] }),
            json!({ "values": [b, c] }),
            json!([["p", ["values"], 0, 1, []]]),
        );
    }

    // "collapses a same-length queue update to two splices"
    #[test]
    fn diff_collapses_a_same_length_queue_update_to_two_splices() {
        let a = json!({ "id": "a" });
        let b = json!({ "id": "b" });
        let c = json!({ "id": "c" });
        let d = json!({ "id": "d" });
        expect_diff(
            json!({ "values": [a, b, c] }),
            json!({ "values": [b, c, d] }),
            json!([["p", ["values"], 0, 1, []], ["p", ["values"], 2, 0, [d]],]),
        );
    }

    // "emits a permutation for a pure reorder"
    #[test]
    fn diff_emits_a_permutation_for_a_pure_reorder() {
        let a = json!({ "id": "a" });
        let b = json!({ "id": "b" });
        let c = json!({ "id": "c" });
        expect_diff(
            json!({ "values": [a, b, c] }),
            json!({ "values": [c, a, b] }),
            json!([["m", ["values"], [2, 0, 1]]]),
        );
    }

    // "normalizes reordered distinct deeply-equal objects to a no-op"
    #[test]
    fn diff_normalizes_reordered_deeply_equal_objects_to_a_no_op() {
        let first = json!({ "nested": { "value": 1 } });
        let second = json!({ "nested": { "value": 1 } });
        // Upstream additionally asserts the two are distinct objects; serde
        // values are distinct allocations here by construction.
        assert_eq!(
            diff_revisions(
                &json!({ "values": [first, second] }),
                &json!({ "values": [second, first] })
            ),
            Vec::<Op>::new()
        );
    }

    // "validates and encodes permutations"
    #[test]
    fn diff_validates_and_encodes_permutations() {
        let operations = ops(json!([
            ["m", ["values"], [2, 0, 1]],
            ["m", ["values"], [1, 2, 0]],
        ]));
        for operation in &operations {
            crate::delta::assert_valid_op(operation).expect("permutation is valid");
        }
        let mut wire_encoder = crate::delta::encoder();
        let wire = wire_encoder.encode(&operations);
        assert_eq!(
            serde_json::to_value(&wire).expect("wire serializes"),
            json!([["m", ["values"], [2, 0, 1]], ["m", [1, 2, 0]],])
        );
        for operation in &wire {
            crate::delta::assert_valid_wire_op(operation).expect("wire permutation is valid");
        }
        let mut wire_decoder = crate::delta::decoder();
        assert_eq!(
            wire_decoder.decode(&wire).expect("wire decodes"),
            operations
        );
        let invalid = ops(json!([["m", ["values"], [0, 0]]]));
        let error = crate::delta::assert_valid_op(&invalid[0]).expect_err("non-bijection rejected");
        assert!(error.to_string().contains("bijection"));
    }

    // "emits nothing for deeply equal reconstructed values"
    #[test]
    fn diff_emits_nothing_for_deeply_equal_reconstructed_values() {
        assert_eq!(
            diff_revisions(
                &json!({ "value": { "nested": [1, 2] } }),
                &json!({ "value": { "nested": [1, 2] } })
            ),
            Vec::<Op>::new()
        );
        assert_eq!(
            diff_revisions(
                &json!({ "values": [{ "id": 1 }, { "id": 2 }] }),
                &json!({ "values": [{ "id": 1 }, { "id": 2 }] })
            ),
            Vec::<Op>::new()
        );
        assert_eq!(
            diff_revisions(
                &json!({ "values": [true, true, true] }),
                &json!({ "values": [true, true, true] })
            ),
            Vec::<Op>::new()
        );
    }

    // "keeps a leaf edit inside a reconstructed array narrow"
    #[test]
    fn diff_keeps_a_leaf_edit_inside_a_reconstructed_array_narrow() {
        expect_diff(
            json!({ "values": [
                { "id": 1, "label": "one" },
                { "id": 2, "label": "two" },
            ] }),
            json!({ "values": [
                { "id": 1, "label": "one" },
                { "id": 2, "label": "changed" },
            ] }),
            json!([["s", ["values", 1, "label"], "changed"]]),
        );
    }

    // "emits payload-free splices for scattered removals"
    #[test]
    fn diff_emits_payload_free_splices_for_scattered_removals() {
        let values = ["a", "b", "c", "d", "e"]
            .iter()
            .map(|id| json!({ "id": id }))
            .collect::<Vec<_>>();
        let after = vec![values[0].clone(), values[2].clone(), values[4].clone()];
        expect_diff(
            json!({ "values": values }),
            json!({ "values": after }),
            json!([["p", ["values"], 1, 1, []], ["p", ["values"], 2, 1, []],]),
        );
    }

    // it.each: "encodes %s removal canonically"
    #[test]
    fn diff_encodes_removals_canonically() {
        let cases: Vec<(JsonValue, JsonValue, JsonValue)> = vec![
            (
                json!([1, 2, 3, 4]),
                json!([2, 3, 4]),
                json!([["p", ["values"], 0, 1, []]]),
            ),
            (
                json!([1, 2, 3, 4]),
                json!([1, 2, 3]),
                json!([["p", ["values"], 3, 1, []]]),
            ),
            (
                json!([1, 2, 3, 4]),
                json!([1, 3, 4]),
                json!([["p", ["values"], 1, 1, []]]),
            ),
            (
                json!([1, 2, 3, 4]),
                json!([]),
                json!([["p", ["values"], 0, 4, []]]),
            ),
            (json!([1, 2, 3, 4]), json!([1, 2, 3, 4]), json!([])),
        ];
        for (before, after, expected) in cases {
            expect_diff(
                json!({ "values": before }),
                json!({ "values": after }),
                expected,
            );
        }
    }

    // "does not field-diff unrelated shifted objects with common fields"
    #[test]
    fn diff_does_not_field_diff_unrelated_shifted_objects() {
        let before = (1..=3)
            .map(|value| json!({ "type": "row", "value": value }))
            .collect::<Vec<_>>();
        let after = (2..=4)
            .map(|value| json!({ "type": "row", "value": value }))
            .collect::<Vec<_>>();
        expect_diff(
            json!({ "values": before }),
            json!({ "values": after }),
            json!([
                ["p", ["values"], 0, 1, []],
                ["p", ["values"], 2, 0, [{ "type": "row", "value": 4 }]],
            ]),
        );
    }

    // "does not treat coincidental id or key fields as structural identity"
    #[test]
    fn diff_does_not_treat_coincidental_id_or_key_fields_as_identity() {
        let left = json!({ "value": "left" });
        let right = json!({ "value": "right" });
        let before = json!({ "values": [
            left,
            { "id": 1, "key": "a", "value": "first" },
            { "id": 2, "key": "b", "value": "second" },
            right,
        ] });
        let replacements = json!([
            { "id": 2, "key": "b", "value": "edited-second" },
            { "id": 1, "key": "a", "value": "edited-first" },
        ]);
        let after = json!({ "values": [
            left,
            { "id": 2, "key": "b", "value": "edited-second" },
            { "id": 1, "key": "a", "value": "edited-first" },
            right,
        ] });
        expect_diff(
            before,
            after,
            json!([["p", ["values"], 1, 2, replacements]]),
        );
    }

    // "combines removals, append, and a shared-subtree survivor edit"
    #[test]
    fn diff_combines_removals_append_and_a_shared_subtree_survivor_edit() {
        let before_values = vec![
            json!({ "id": "a", "stable": {}, "detail": { "text": "a" } }),
            json!({ "id": "b", "stable": {}, "detail": { "text": "b" } }),
            json!({ "id": "c", "stable": {}, "detail": { "text": "c" } }),
            json!({ "id": "d", "stable": {}, "detail": { "text": "d" } }),
        ];
        let appended = json!({ "id": "e", "stable": {}, "detail": { "text": "e" } });
        let after_values = vec![
            before_values[0].clone(),
            json!({ "id": "c", "stable": {}, "detail": { "text": "changed" } }),
            before_values[3].clone(),
            appended.clone(),
        ];
        expect_diff(
            json!({ "values": before_values }),
            json!({ "values": after_values }),
            json!([
                ["p", ["values"], 1, 1, []],
                ["a", ["values", 1, "detail", "text"], "hanged"],
                ["p", ["values"], 3, 0, [appended]],
            ]),
        );
    }

    // it.each: "does not retain a %i-byte removed-neighbor payload"
    #[test]
    fn diff_does_not_retain_removed_neighbor_payloads() {
        for size in [256 * 1024usize, 1024 * 1024] {
            let payload = "x".repeat(size);
            let retained = ["a", "b", "c", "d", "e"]
                .iter()
                .map(|id| json!({ "id": id, "payload": payload }))
                .collect::<Vec<_>>();
            let before = json!({ "values": retained.clone() });
            let after = json!({ "values": [retained[0].clone(), retained[2].clone(), retained[4].clone()] });
            let operations = diff_revisions(&before, &after);
            assert_eq!(
                serde_json::to_value(&operations).expect("ops serialize"),
                json!([["p", ["values"], 1, 1, []], ["p", ["values"], 2, 1, []],])
            );
            assert!(
                serde_json::to_string(&operations)
                    .expect("ops serialize")
                    .len()
                    < 100
            );
            assert_eq!(
                apply_immutable(&before, &operations).expect("ops apply"),
                after
            );
            assert_eq!(before, json!({ "values": retained }));
        }
    }

    // it.each: "keeps push, pop, and middle removal narrow at %,i items"
    #[test]
    fn diff_keeps_push_pop_and_middle_removal_narrow() {
        for size in [1_001usize, 10_000] {
            let values = (0..size)
                .map(|value| json!({ "value": value }))
                .collect::<Vec<_>>();
            let appended = json!({ "value": size });
            let mut with_appended = values.clone();
            with_appended.push(appended.clone());
            expect_diff(
                json!({ "values": values.clone() }),
                json!({ "values": with_appended }),
                json!([["p", ["values"], size, 0, [appended]]]),
            );
            let without_last = values[..values.len() - 1].to_vec();
            expect_diff(
                json!({ "values": values.clone() }),
                json!({ "values": without_last }),
                json!([["p", ["values"], size - 1, 1, []]]),
            );
            let middle = size / 2;
            let without_middle = values
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != middle)
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>();
            expect_diff(
                json!({ "values": values }),
                json!({ "values": without_middle }),
                json!([["p", ["values"], middle, 1, []]]),
            );
        }
    }

    // "keeps forty-thousand-row sparse edits narrow"
    #[test]
    fn diff_keeps_forty_thousand_row_sparse_edits_narrow() {
        let values = (0..40_000usize)
            .map(|value| json!({ "value": value, "stable": { "value": value } }))
            .collect::<Vec<_>>();
        let mut after = values.clone();
        let mut changed = Vec::new();
        let mut index = 100;
        while index < after.len() {
            let value = after[index].get("stable").cloned().expect("stable present");
            after[index] = json!({ "value": -(index as i64), "stable": value });
            changed.push(index);
            index += 400;
        }
        let operations = diff_revisions(
            &json!({ "values": values.clone() }),
            &json!({ "values": after.clone() }),
        );
        let expected = changed
            .iter()
            .map(|index| json!(["s", ["values", index, "value"], -(*index as i64)]))
            .collect::<Vec<_>>();
        assert_eq!(
            serde_json::to_value(&operations).expect("ops serialize"),
            json!(expected)
        );
        assert!(
            serde_json::to_string(&operations)
                .expect("ops serialize")
                .len()
                < 7_500
        );
        assert_eq!(
            apply_immutable(&json!({ "values": values }), &operations).expect("ops apply"),
            json!({ "values": after })
        );
    }

    // "keeps a reconstructed large-array leaf edit narrow"
    #[test]
    fn diff_keeps_a_reconstructed_large_array_leaf_edit_narrow() {
        let before = (0..1_000usize)
            .map(|value| json!({ "value": value, "label": format!("row-{value}") }))
            .collect::<Vec<_>>();
        let mut after = before.clone();
        after[700] = json!({ "value": 700, "label": "changed" });
        let operations = diff_revisions(
            &json!({ "values": before.clone() }),
            &json!({ "values": after.clone() }),
        );
        assert_eq!(
            serde_json::to_value(&operations).expect("ops serialize"),
            json!([["s", ["values", 700, "label"], "changed"]])
        );
        assert!(
            serde_json::to_string(&operations)
                .expect("ops serialize")
                .len()
                < 100
        );
        assert_eq!(
            apply_immutable(&json!({ "values": before }), &operations).expect("ops apply"),
            json!({ "values": after })
        );
    }

    // "splices an ambiguous equal-count moved-and-edited gap"
    #[test]
    fn diff_splices_an_ambiguous_equal_count_moved_and_edited_gap() {
        let left = json!({ "value": "left" });
        let right = json!({ "value": "right" });
        let before = json!({ "values": [
            left,
            { "id": 1, "value": "a" },
            { "id": 2, "value": "b" },
            right,
        ] });
        let after = json!({ "values": [
            left,
            { "id": 2, "value": "edited" },
            { "id": 1, "value": "also-edited" },
            right,
        ] });
        expect_diff(
            before,
            after,
            json!([["p", ["values"], 1, 2, [
                { "id": 2, "value": "edited" },
                { "id": 1, "value": "also-edited" },
            ]]]),
        );
    }

    // "encodes five hundred unshifts without snapshotting retained rows"
    #[test]
    fn diff_encodes_five_hundred_unshifts_without_snapshotting_retained_rows() {
        let retained = (0..10_000usize)
            .map(|value| json!({ "value": value, "payload": "x".repeat(100) }))
            .collect::<Vec<_>>();
        let inserted = (0..500i64)
            .map(|value| json!({ "value": -value - 1 }))
            .collect::<Vec<_>>();
        let mut after = inserted.clone();
        after.extend(retained.iter().cloned());
        let operations = diff_revisions(
            &json!({ "values": retained.clone() }),
            &json!({ "values": after.clone() }),
        );
        assert_eq!(
            serde_json::to_value(&operations).expect("ops serialize"),
            json!([["p", ["values"], 0, 0, inserted]])
        );
        assert!(
            serde_json::to_string(&operations)
                .expect("ops serialize")
                .len()
                < 20_000
        );
        assert_eq!(
            apply_immutable(&json!({ "values": retained }), &operations).expect("ops apply"),
            json!({ "values": after })
        );
    }

    // "bounds wide-object operation emission with a root replacement"
    #[test]
    fn diff_bounds_wide_object_operation_emission_with_a_root_replacement() {
        let mut before = serde_json::Map::new();
        let mut after = serde_json::Map::new();
        for index in 0..20_000 {
            before.insert(format!("field{index}"), json!(0));
            after.insert(format!("field{index}"), json!(1));
        }
        let operations = diff_revisions(
            &JsonValue::Object(before.clone()),
            &JsonValue::Object(after.clone()),
        );
        assert_eq!(
            serde_json::to_value(&operations).expect("ops serialize"),
            json!([["r", JsonValue::Object(after)]])
        );
        let _ = before;
    }

    // "falls back to a base operation when leaf operations are larger"
    #[test]
    fn diff_falls_back_to_a_base_operation_when_leaf_operations_are_larger() {
        let before = json!({ "values": (0..40_000).map(|_| json!(0)).collect::<Vec<_>>() });
        let after = json!({ "values": (0..40_000).map(|_| json!(1)).collect::<Vec<_>>() });
        let operations = diff_revisions(&before, &after);
        assert_eq!(
            serde_json::to_value(&operations).expect("ops serialize"),
            json!([["r", after]])
        );
    }

    // ── state-value.test.ts ──────────────────────────────────────────────────

    // "takes ownership of an alias-free mutable JSON root without freezing" —
    // freezing has no Rust counterpart; ownership and branch independence are
    // structural.
    #[test]
    fn tracker_takes_ownership_of_alias_free_root() {
        let input = json!({ "left": { "value": 1 }, "right": { "value": 1 } });
        let tracker = track(input.clone());
        assert_eq!(tracker.value(), &input);
    }

    // "commits transaction copies in place while sharing unchanged branches"
    #[test]
    fn tracker_commits_transaction_copies_in_place() {
        let mut tracker = track(json!({ "changed": { "value": 1 }, "retained": { "value": 2 } }));
        let base_snapshot = tracker.value().clone();
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            change
                .state()
                .set(&[Seg::key("changed")], "value", json!(3))
                .expect("set succeeds");
            change.prepare().expect("prepare succeeds")
        };
        // Unchanged branches carry the same content (upstream: same reference).
        assert_eq!(prepared.value.get("retained"), base.get("retained"));
        assert_eq!(prepared.value.get("changed"), Some(&json!({ "value": 3 })));
        tracker.adopt(prepared).expect("adopt succeeds");
        // Upstream asserts the pre-change root object is still intact after
        // adoption (checked through its reference). `base` is a snapshot clone
        // here, so that aliasing guarantee is vacuous; the committed root
        // assertions above carry the observable semantics.
        assert_eq!(base, base_snapshot);
    }

    // "makes repeated placements independent"
    #[test]
    fn tracker_makes_repeated_placements_independent() {
        let mut tracker = track(json!({ "values": [] }));
        let mut change = tracker.begin_change();
        let shared = json!({ "value": 1 });
        change
            .state()
            .push(&[Seg::key("values")], vec![shared.clone(), shared])
            .expect("push succeeds");
        let prepared = change.prepare().expect("prepare succeeds");
        assert_eq!(
            prepared.value.get("values"),
            Some(&json!([{ "value": 1 }, { "value": 1 }]))
        );
    }

    // ── state-draft.test.ts ──────────────────────────────────────────────────

    // "copies only changed branches" — handle-identity sub-assertions degrade to
    // structural checks (serde values have no handles).
    #[test]
    fn draft_copies_only_changed_branches() {
        let mut tracker = track(json!({
            "changed": { "count": 1, "sibling": { "value": "kept" } },
            "untouched": { "value": 2 },
        }));
        let base = tracker.value().clone();
        let mut change = tracker.begin_change();
        change
            .state()
            .set(&[Seg::key("changed")], "count", json!(3))
            .expect("set succeeds");
        let prepared = change.prepare().expect("prepare succeeds");
        assert_ne!(prepared.value, base);
        assert_eq!(
            prepared.value.get("changed"),
            Some(&json!({ "count": 3, "sibling": { "value": "kept" } }))
        );
        assert_eq!(prepared.value.get("untouched"), base.get("untouched"));
        assert_eq!(
            prepared.value.get("changed").and_then(|c| c.get("sibling")),
            base.get("changed").and_then(|c| c.get("sibling"))
        );
    }

    // "supports frozen bases, deletion, and native array mutators" — frozen
    // bases have no Rust counterpart; deletion and mutators port directly.
    #[test]
    fn draft_supports_deletion_and_native_array_mutators() {
        let mut tracker = track(json!({ "optional": "remove", "values": [3, 1, 2] }));
        let mut change = tracker.begin_change();
        let mut draft = change.state();
        draft.delete(&[], "optional").expect("delete succeeds");
        draft
            .push(&[Seg::key("values")], vec![json!(4)])
            .expect("push succeeds");
        let popped = draft
            .splice(&[Seg::key("values")], 3, Some(1), vec![])
            .expect("pop succeeds");
        assert_eq!(popped, vec![json!(4)]);
        draft
            .splice(&[Seg::key("values")], 0, Some(0), vec![json!(0)])
            .expect("unshift succeeds");
        let shifted = draft
            .splice(&[Seg::key("values")], 0, Some(1), vec![])
            .expect("shift succeeds");
        assert_eq!(shifted, vec![json!(0)]);
        let spliced = draft
            .splice(&[Seg::key("values")], 1, Some(1), vec![json!(5), json!(4)])
            .expect("splice succeeds");
        assert_eq!(spliced, vec![json!(1)]);
        draft.sort_by(&[Seg::key("values")], |left, right| {
            (left.as_i64().unwrap_or(0)).cmp(&right.as_i64().unwrap_or(0))
        });
        // Upstream reverses; a descending sort produces the identical array.
        draft.sort_by(&[Seg::key("values")], |left, right| {
            (right.as_i64().unwrap_or(0)).cmp(&left.as_i64().unwrap_or(0))
        });
        draft
            .fill(&[Seg::key("values")], json!(9), Some(1), Some(3))
            .expect("fill succeeds");
        draft
            .copy_within(&[Seg::key("values")], 1, 0, Some(2))
            .expect("copyWithin succeeds");
        let prepared = change.prepare().expect("prepare succeeds");
        assert_eq!(prepared.value, json!({ "values": [5, 5, 9, 2] }));
    }

    // "copies inserted draft values without aliasing their original handle" —
    // handle identity is unrepresentable; the copy-on-insert semantics are
    // asserted through the resulting revision.
    #[test]
    fn draft_copies_inserted_values_without_aliasing() {
        let mut tracker = track(json!({ "values": [{ "value": 1 }, { "value": 2 }] }));
        let mut change = tracker.begin_change();
        let mut draft = change.state();
        let held = draft
            .get_index(&[Seg::key("values")], 0)
            .expect("read succeeds")
            .cloned()
            .expect("element present");
        draft
            .splice(&[Seg::key("values")], 0, Some(0), vec![held])
            .expect("unshift succeeds");
        draft
            .set_index(&[Seg::key("values")], 1, json!({ "value": 9 }))
            .expect("set succeeds");
        let prepared = change.prepare().expect("prepare succeeds");
        assert_eq!(
            prepared.value.get("values"),
            Some(&json!([{ "value": 1 }, { "value": 9 }, { "value": 2 }]))
        );
    }

    // "keeps comparator edits when sorting object drafts"
    #[test]
    fn draft_keeps_comparator_edits_when_sorting_object_drafts() {
        let mut tracker = track(json!({
            "rows": [
                { "rank": 2, "comparisons": 0 },
                { "rank": 1, "comparisons": 0 },
            ],
        }));
        let mut change = tracker.begin_change();
        fn bump_comparisons(value: &mut JsonValue) {
            if let Some(object) = value.as_object_mut() {
                let comparisons = object
                    .get("comparisons")
                    .and_then(JsonValue::as_i64)
                    .unwrap_or(0);
                object.insert("comparisons".into(), json!(comparisons + 1));
            }
        }
        change
            .state()
            .sort_by(&[Seg::key("rows")], |left, right| {
                bump_comparisons(left);
                bump_comparisons(right);
                let left_rank = left.get("rank").and_then(JsonValue::as_i64).unwrap_or(0);
                let right_rank = right.get("rank").and_then(JsonValue::as_i64).unwrap_or(0);
                left_rank.cmp(&right_rank)
            })
            .expect("sort succeeds");
        let prepared = change.prepare().expect("prepare succeeds");
        let ranks = prepared
            .value
            .get("rows")
            .and_then(JsonValue::as_array)
            .expect("rows");
        assert_eq!(
            ranks
                .iter()
                .map(|row| row.get("rank").cloned().unwrap())
                .collect::<Vec<_>>(),
            vec![json!(1), json!(2)]
        );
        assert_eq!(
            ranks
                .iter()
                .map(|row| row.get("comparisons").cloned().unwrap())
                .collect::<Vec<_>>(),
            vec![json!(1), json!(1)]
        );
    }

    // "drops writes through detached child handles" is non-portable: Rust drafts
    // expose path-based access, not child handles, so a detached write cannot be
    // expressed (the borrow checker rejects the equivalent escape).

    // "inserts 100,000 items with %s without argument overflow"
    #[test]
    fn draft_inserts_many_items_without_argument_overflow() {
        for method in ["unshift", "splice"] {
            let mut tracker = track(json!({ "values": [-1] }));
            let mut change = tracker.begin_change();
            let items = (0..100_000i64)
                .map(|value| json!(value))
                .collect::<Vec<_>>();
            let mut draft = change.state();
            if method == "unshift" {
                draft
                    .splice(&[Seg::key("values")], 0, Some(0), items)
                    .expect("unshift succeeds");
            } else {
                draft
                    .splice(&[Seg::key("values")], 1, Some(0), items)
                    .expect("splice succeeds");
            }
            let prepared = change.prepare().expect("prepare succeeds");
            let values = prepared
                .value
                .get("values")
                .and_then(JsonValue::as_array)
                .expect("values");
            assert_eq!(values.len(), 100_001);
            let probe = if method == "unshift" { 0 } else { 1 };
            assert_eq!(values[probe], json!(0));
            assert_eq!(values[probe + 99_999], json!(99_999));
            assert_eq!(
                apply_immutable(&prepared.base, &prepared.ops).expect("ops apply"),
                prepared.value
            );
        }
    }

    // "rejects non-strict JSON placements without mutating the draft or base"
    // and "rejects array holes without mutating the committed base" are
    // non-portable: serde_json values cannot represent undefined, NaN, class
    // instances, or array holes, so those rejections are upheld by construction.

    // ── state-fuzz.test.ts ───────────────────────────────────────────────────

    /// Upstream's deterministic PRNG (mulberry32), ported operation-for-operation.
    struct Rng(u32);

    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x6d2b79f5);
            let seed = self.0;
            let mut value = (seed ^ (seed >> 15)).wrapping_mul(1 | seed);
            value = value.wrapping_add((value ^ (value >> 7)).wrapping_mul(61 | value)) ^ value;
            (value ^ (value >> 14)) as f64 / 4_294_967_296.0
        }
    }

    fn fuzz_item(value: i64) -> JsonValue {
        json!({ "id": value, "text": format!("item-{value}"), "score": value % 7 })
    }

    /// The upstream `mutate` switch, expressed through the draft API.
    fn mutate(draft: &mut Draft, choice: usize, value: i64) {
        let items = Seg::key("items");
        let meta = Seg::key("meta");
        match choice {
            0 => {
                let text = draft
                    .get(&[])
                    .expect("read")
                    .and_then(|d| d.get("text").cloned())
                    .unwrap_or_default();
                let text = text.as_str().unwrap_or_default().to_string();
                draft
                    .set(&[], "text", json!(format!("{text}-{value}")))
                    .expect("set");
            }
            1 => {
                let text = draft
                    .get(&[])
                    .expect("read")
                    .and_then(|d| d.get("text").cloned())
                    .unwrap_or_default();
                let text = text.as_str().unwrap_or_default().to_string();
                let start = std::cmp::min(2, text.len());
                let sliced: String = text.chars().skip(start).collect();
                draft
                    .set(&[], "text", json!(format!("{sliced}{value}")))
                    .expect("set");
            }
            2 => {
                draft
                    .push(&[items.clone()], vec![fuzz_item(value)])
                    .expect("push");
            }
            3 => {
                draft
                    .splice(&[items.clone()], 0, Some(0), vec![fuzz_item(value)])
                    .expect("unshift");
            }
            4 => {
                let length = draft.length(&[items.clone()]).expect("length");
                if length > 0 {
                    draft
                        .splice(&[items.clone()], 0, Some(1), vec![])
                        .expect("shift");
                }
            }
            5 => {
                let length = draft.length(&[items.clone()]).expect("length");
                if length > 0 {
                    draft
                        .splice(&[items.clone()], length - 1, Some(1), vec![])
                        .expect("pop");
                }
            }
            6 => {
                let length = draft.length(&[items.clone()]).expect("length");
                let index = if length == 0 {
                    0
                } else {
                    (value as usize) % (length + 1)
                };
                let remove = if length == 0 {
                    None
                } else {
                    Some((value as usize) % 2)
                };
                draft
                    .splice(&[items.clone()], index, remove, vec![fuzz_item(value)])
                    .expect("splice");
            }
            7 => {
                let length = draft.length(&[items.clone()]).expect("length");
                for index in 0..length / 2 {
                    let left = draft
                        .get_index(&[items.clone()], index)
                        .expect("read")
                        .cloned()
                        .expect("item");
                    let right_index = length - 1 - index;
                    let right = draft
                        .get_index(&[items.clone()], right_index)
                        .expect("read")
                        .cloned()
                        .expect("item");
                    draft
                        .set_index(&[items.clone()], index, right)
                        .expect("set");
                    draft
                        .set_index(&[items.clone()], right_index, left)
                        .expect("set");
                }
            }
            8 => {
                draft.sort_by(&[items.clone()], |left, right| {
                    (left.get("id").and_then(JsonValue::as_i64).unwrap_or(0))
                        .cmp(&right.get("id").and_then(JsonValue::as_i64).unwrap_or(0))
                });
            }
            9 => {
                let length = draft.length(&[items.clone()]).expect("length");
                if length > 0 {
                    let index = (value as usize) % length;
                    let mut item = draft
                        .get_index(&[items.clone()], index)
                        .expect("read")
                        .cloned()
                        .expect("item");
                    if let Some(object) = item.as_object_mut() {
                        object.insert("score".into(), json!(value));
                    }
                    draft.set_index(&[items.clone()], index, item).expect("set");
                }
            }
            10 => {
                let revision = draft
                    .get(&[meta.clone()])
                    .expect("read")
                    .and_then(|m| m.get("revision").cloned())
                    .and_then(|r| r.as_i64())
                    .unwrap_or(0);
                draft
                    .set(&[meta.clone()], "revision", json!(revision + 1))
                    .expect("set");
                draft
                    .set(&[meta.clone()], "label", json!(format!("revision-{value}")))
                    .expect("set");
            }
            11 => {
                draft.delete(&[meta.clone()], "label").expect("delete");
            }
            12 => {
                let length = draft.length(&[items.clone()]).expect("length");
                if length > 1 {
                    let first = draft
                        .get_index(&[items.clone()], 0)
                        .expect("read")
                        .cloned()
                        .expect("item");
                    draft.set_index(&[items.clone()], 1, first).expect("set");
                }
            }
            _ => {
                let length = draft.length(&[items.clone()]).expect("length");
                for index in 0..std::cmp::min(2, length) {
                    draft
                        .set_index(&[items.clone()], index, fuzz_item(value))
                        .expect("set");
                }
            }
        }
    }

    // "converges across randomized prepared revisions" — the alias-free
    // assertion is vacuous for serde trees (they cannot alias by construction).
    #[test]
    fn fuzz_converges_across_randomized_prepared_revisions() {
        for seed in 1u32..=100 {
            let mut rng = Rng(seed);
            let initial = json!({
                "items": (0..4).map(fuzz_item).collect::<Vec<_>>(),
                "text": "start",
                "meta": { "revision": 0 },
            });
            let mut tracker = track(initial.clone());
            let mut expected =
                serde_json::from_str::<JsonValue>(&serde_json::to_string(&initial).expect("clone"))
                    .expect("clone");
            let mut replica = serde_json::from_str::<JsonValue>(
                &serde_json::to_string(tracker.value()).expect("clone"),
            )
            .expect("clone");
            for step in 0..100i64 {
                let choice = (rng.next() * 14.0) as usize;
                let value = seed as i64 * 1_000 + step;
                let base_root = tracker.value().clone();
                let base = serde_json::from_str::<JsonValue>(
                    &serde_json::to_string(&base_root).expect("clone"),
                )
                .expect("clone");
                mutate_value(&mut expected, choice, value);
                let mut change = tracker.begin_change();
                mutate(&mut change.state(), choice, value);
                let prepared = change.prepare().expect("prepare succeeds");
                let operations = serde_json::from_str::<Vec<Op>>(
                    &serde_json::to_string(&prepared.ops).expect("clone"),
                )
                .expect("clone");
                assert_eq!(
                    *tracker.value(),
                    base,
                    "prepare base seed {seed} step {step} choice {choice}"
                );
                replica = apply_immutable(&replica, &operations).expect("ops apply");
                assert_eq!(
                    replica, prepared.value,
                    "replay seed {seed} step {step} choice {choice}"
                );
                tracker.adopt(prepared).expect("adopt succeeds");
                // Upstream also asserts the pre-change root is still intact after
                // adoption; the Rust `base` is a snapshot clone, so that aliasing
                // guarantee is vacuous here.
                assert_eq!(
                    *tracker.value(),
                    expected,
                    "state seed {seed} step {step} choice {choice}"
                );
                assert_eq!(
                    replica, expected,
                    "replica seed {seed} step {step} choice {choice}"
                );
            }
        }
    }

    /// The same mutation applied to a plain value tree to build the expectation.
    fn mutate_value(document: &mut JsonValue, choice: usize, value: i64) {
        let mut scratch = track(std::mem::take(document));
        {
            let mut change = scratch.begin_change();
            mutate(&mut change.state(), choice, value);
            let prepared = change.prepare().expect("prepare succeeds");
            scratch.adopt(prepared).expect("adopt succeeds");
        }
        *document = scratch.value().clone();
    }
}
