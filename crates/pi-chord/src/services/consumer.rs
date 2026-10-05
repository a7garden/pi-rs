//! Port of `packages/chord/src/services/consumer.ts`.
//!
//! Deviations from upstream (recorded per PORT_MAP.md rule 3):
//! - `ServiceFacade`/`MemberSlot` proxies become [`ServiceFacade`] implementing
//!   [`ServiceClientHandle`]; member access goes through `invoke`/`state` instead of
//!   JS property gets. The "Remote service method {id}.{member} requires a trailing
//!   Context" error is unportable: `Context` is an explicit parameter.
//! - Local-service rejection (`Service {id} is process-local`) is unreachable:
//!   `RemoteServices::use_service`/`observe` take bare service ids with no local flag.
//! - Returned state views cannot fail per call (`ReplicatedState` signatures are
//!   infallible); access failures surface at `state(member)` acquisition time and
//!   degrade to no-ops on a captured view.
//! - Upstream `Promise`s become spawned tokio tasks paired with a [`StartingTask`]
//!   completion cell so `ready`/`rebind`/`dispose` can await the same outcome,
//!   including error identity for `ready`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use futures::future::try_join_all;

use crate::context::{await_with_context, BACKGROUND_CONTEXT, Context};
use crate::delta::Op;
use crate::json::JsonValue;
use crate::services::errors::{
    error, remote_service_error, type_error, AggregateError, AnyError, RemoteServiceErrorCode,
};
use crate::services::instances::{InstanceDirectory, InstanceDirectoryEntry, InstanceObserverHandler};
use crate::services::state::{service_delivery_context, ReplicatedStateReplica};
use crate::types::{
    arc_error, BoxFuture, KeyedObservationHandler, RemoteServiceBindingOptions, RemoteServiceTransport,
    RemoteServices, ReplicatedState, ServiceCall, ServiceClientHandle, ServiceInstanceAddress,
    ServiceInstanceSnapshot, ServiceMemberSnapshot, ServiceMode, ServiceProviderUpdate,
    ServiceSubscription, ServiceSubscriptionSnapshot, StateListener, UpdateListener, Unsubscribe,
};

type AccessChecker = Arc<dyn Fn() -> Result<(), AnyError> + Send + Sync>;
type IsActive = Arc<dyn Fn() -> bool + Send + Sync>;
type MemberInvoke =
    Arc<dyn Fn(Vec<JsonValue>, Context) -> BoxFuture<Result<Option<JsonValue>, AnyError>> + Send + Sync>;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Error retained behind an `Arc` so one settled start outcome can be observed by
/// `ready`, `rebind`, `dispose` and the settle-time reporter (upstream: one Promise
/// with several consumers of the same rejection reason).
#[derive(Debug)]
struct SharedErrorBox(Arc<dyn std::error::Error + Send + Sync>);

impl std::fmt::Display for SharedErrorBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SharedErrorBox {}

/// Completion cell for one spawned async sequence (upstream: a `Promise<void>`).
struct StartingTask {
    result: Mutex<Option<Result<(), Arc<dyn std::error::Error + Send + Sync>>>>,
    done: tokio::sync::watch::Receiver<bool>,
}

/// Spawn `future` eagerly (upstream promises start eagerly) and return a handle any
/// number of consumers can await. `on_error_at_settle` reports rejections whose
/// settle-time conditions hold (upstream `.catch(...)` chains).
fn spawn_starting(
    future: BoxFuture<Result<(), AnyError>>,
    on_error_at_settle: Option<Box<dyn FnOnce(AnyError) + Send + Sync>>,
) -> Arc<StartingTask> {
    let (sender, receiver) = tokio::sync::watch::channel(false);
    let task = Arc::new(StartingTask {
        result: Mutex::new(None),
        done: receiver,
    });
    let driver = Arc::clone(&task);
    tokio::spawn(async move {
        let outcome = match future.await {
            Ok(()) => Ok(()),
            Err(failure) => {
                let shared: Arc<dyn std::error::Error + Send + Sync> = Arc::from(failure);
                if let Some(report) = on_error_at_settle {
                    report(Box::new(SharedErrorBox(Arc::clone(&shared))));
                }
                Err(shared)
            }
        };
        *lock(&driver.result) = Some(outcome);
        let _ = sender.send(true);
    });
    task
}

impl StartingTask {
    async fn wait(&self) -> Result<(), AnyError> {
        let mut done = self.done.clone();
        while !*done.borrow_and_update() {
            if done.changed().await.is_err() {
                break;
            }
        }
        match &*lock(&self.result) {
            Some(Ok(())) => Ok(()),
            Some(Err(failure)) => Err(Box::new(SharedErrorBox(Arc::clone(failure)))),
            None => Ok(()),
        }
    }
}

struct MemberKinds {
    described: Option<&'static str>,
    expected: Option<&'static str>,
}

/// One per-instance member slot: method invoker plus a [`ReplicatedStateReplica`]
/// for state members (upstream `MemberSlot`).
struct MemberSlot {
    service_id: String,
    member: String,
    invoke: MemberInvoke,
    state: Arc<ReplicatedStateReplica>,
    is_active: IsActive,
    assert_access: AccessChecker,
    kinds: Mutex<MemberKinds>,
}

impl MemberSlot {
    fn new(
        service_id: &str,
        member: &str,
        invoke: MemberInvoke,
        is_active: IsActive,
        assert_access: AccessChecker,
        report_error: arc_error::ErrorHandler,
    ) -> Arc<MemberSlot> {
        Arc::new(MemberSlot {
            service_id: service_id.to_string(),
            member: member.to_string(),
            invoke,
            state: ReplicatedStateReplica::new(report_error),
            is_active,
            assert_access,
            kinds: Mutex::new(MemberKinds {
                described: None,
                expected: None,
            }),
        })
    }

    fn set_description(&self, kind: &'static str) -> Result<(), AnyError> {
        let mut kinds = lock(&self.kinds);
        if let Some(existing) = kinds.described {
            if existing != kind {
                return Err(error(format!(
                    "Remote service member {}.{} changed kind",
                    self.service_id, self.member
                )));
            }
        }
        kinds.described = Some(kind);
        if let Some(expected) = kinds.expected {
            if expected != kind {
                return Err(remote_service_error(
                    RemoteServiceErrorCode::ServiceMemberMismatch,
                    format!(
                        "Remote service member {}.{} is {}, not {}",
                        self.service_id, self.member, kind, expected
                    ),
                ));
            }
        }
        Ok(())
    }

    fn expect_kind(&self, kind: &'static str) -> Result<(), AnyError> {
        let mut kinds = lock(&self.kinds);
        if let Some(expected) = kinds.expected {
            if expected != kind {
                return Err(remote_service_error(
                    RemoteServiceErrorCode::ServiceMemberMismatch,
                    format!(
                        "Remote service member {}.{} was used as two different kinds",
                        self.service_id, self.member
                    ),
                ));
            }
        }
        kinds.expected = Some(kind);
        if let Some(described) = kinds.described {
            if described != kind {
                return Err(remote_service_error(
                    RemoteServiceErrorCode::ServiceMemberMismatch,
                    format!(
                        "Remote service member {}.{} is {}, not {}",
                        self.service_id, self.member, described, kind
                    ),
                ));
            }
        }
        Ok(())
    }

    fn hydrate(&self, sequence: u64, ops: &[Op], context: &Context) -> Result<(), AnyError> {
        self.set_description("state")?;
        self.state.hydrate(sequence, ops, context)
    }

    fn update(&self, sequence: u64, ops: &[Op], context: &Context) -> Result<(), AnyError> {
        self.set_description("state")?;
        self.state.update(sequence, ops, context)
    }

    fn clear(&self) {
        self.state.clear();
    }

    fn invoke_method(
        &self,
        args: Vec<JsonValue>,
        context: &Context,
    ) -> BoxFuture<Result<Option<JsonValue>, AnyError>> {
        // Upstream performs these checks synchronously before returning the promise.
        let check = (|| -> Result<(), AnyError> {
            (self.assert_access)()?;
            self.expect_kind("method")?;
            if !(self.is_active)() {
                return Err(remote_service_error(
                    RemoteServiceErrorCode::ServiceStaleInstance,
                    format!("Remote service {} binding is closed", self.service_id),
                ));
            }
            Ok(())
        })();
        match check {
            Err(failure) => Box::pin(async move { Err(failure) }),
            Ok(()) => (self.invoke)(args, context.clone()),
        }
    }
}

/// Access-checked read view over one state member (upstream: the `.value`/`.subscribe`
/// surface of the member proxy, which asserts access on every use).
struct GuardedState {
    slot: Arc<MemberSlot>,
}

impl ReplicatedState for GuardedState {
    fn value(&self) -> Option<&JsonValue> {
        if (self.slot.assert_access)().is_err() || self.slot.expect_kind("state").is_err() {
            return None;
        }
        self.slot.state.value()
    }

    fn subscribe(&self, listener: StateListener) -> Unsubscribe {
        if (self.slot.assert_access)().is_err() || self.slot.expect_kind("state").is_err() {
            return Box::new(|| {});
        }
        self.slot.state.subscribe(listener)
    }
}

struct FacadeState {
    slots: HashMap<String, Arc<MemberSlot>>,
    descriptions: HashMap<String, &'static str>,
}

/// Per-instance remote facade (upstream `ServiceFacade`), exposed to consumers as a
/// [`ServiceClientHandle`].
pub struct ServiceFacade {
    service_id: String,
    address: Option<ServiceInstanceAddress>,
    transport: Arc<dyn RemoteServiceTransport>,
    report_error: arc_error::ErrorHandler,
    state: Mutex<FacadeState>,
    is_active: IsActive,
    assert_access: AccessChecker,
}

impl ServiceFacade {
    pub fn new(
        service_id: &str,
        address: Option<ServiceInstanceAddress>,
        transport: Arc<dyn RemoteServiceTransport>,
        report_error: arc_error::ErrorHandler,
        is_active: IsActive,
        assert_access: AccessChecker,
    ) -> Arc<ServiceFacade> {
        Arc::new(ServiceFacade {
            service_id: service_id.to_string(),
            address,
            transport,
            report_error,
            state: Mutex::new(FacadeState {
                slots: HashMap::new(),
                descriptions: HashMap::new(),
            }),
            is_active,
            assert_access,
        })
    }

    pub fn service_id(&self) -> &str {
        &self.service_id
    }

    pub fn install(&self, snapshot: &ServiceInstanceSnapshot, context: &Context) -> Result<(), AnyError> {
        if !same_address(snapshot.instance.as_ref(), self.address.as_ref()) {
            return Err(error("Remote service snapshot has the wrong address"));
        }
        let members = validate_members(&snapshot.members)?;
        {
            let state = lock(&self.state);
            for name in state.slots.keys() {
                if !members.contains_key(name) {
                    return Err(remote_service_error(
                        RemoteServiceErrorCode::ServiceMemberNotFound,
                        format!("Unknown remote service member {}.{}", self.service_id, name),
                    ));
                }
            }
        }
        {
            let mut state = lock(&self.state);
            state.descriptions.clear();
            for (name, kind) in &members {
                state.descriptions.insert(name.clone(), *kind);
            }
        }
        for member in &snapshot.members {
            match member {
                ServiceMemberSnapshot::State { name, sequence, ops } => {
                    let slot = self.slot(name)?;
                    slot.hydrate(*sequence, ops, context)?;
                }
                ServiceMemberSnapshot::Method { name } => {
                    let existing = lock(&self.state).slots.get(name).map(Arc::clone);
                    if let Some(slot) = existing {
                        slot.set_description("method")?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn update(
        &self,
        member: &str,
        sequence: u64,
        ops: &[Op],
        context: &Context,
    ) -> Result<(), AnyError> {
        {
            let state = lock(&self.state);
            if state.descriptions.get(member).copied() != Some("state") {
                return Err(error(format!(
                    "Remote service update targets non-state member {}.{}",
                    self.service_id, member
                )));
            }
        }
        let slot = self.slot(member)?;
        slot.update(sequence, ops, context)
    }

    pub fn clear(&self) {
        let slots: Vec<Arc<MemberSlot>> = lock(&self.state).slots.values().cloned().collect();
        for slot in slots {
            slot.clear();
        }
    }

    fn slot(&self, member: &str) -> Result<Arc<MemberSlot>, AnyError> {
        {
            let state = lock(&self.state);
            if let Some(slot) = state.slots.get(member) {
                return Ok(Arc::clone(slot));
            }
        }
        let transport = Arc::clone(&self.transport);
        let service_id = self.service_id.clone();
        let address = self.address.clone();
        let member_name = member.to_string();
        let invoke: MemberInvoke = Arc::new(move |args: Vec<JsonValue>, context: Context| {
            let call = ServiceCall {
                service_id: service_id.clone(),
                instance: address.clone(),
                member: member_name.clone(),
                args,
            };
            let transport = Arc::clone(&transport);
            Box::pin(async move { transport.invoke(&call, &context).await })
        });
        let slot = MemberSlot::new(
            &self.service_id,
            member,
            invoke,
            Arc::clone(&self.is_active),
            Arc::clone(&self.assert_access),
            Arc::clone(&self.report_error),
        );
        let kind = lock(&self.state).descriptions.get(member).copied();
        if let Some(kind) = kind {
            slot.set_description(kind)?;
        }
        lock(&self.state).slots.insert(member.to_string(), Arc::clone(&slot));
        Ok(slot)
    }
}

impl ServiceClientHandle for ServiceFacade {
    fn service_id(&self) -> &str {
        &self.service_id
    }

    fn invoke(
        &self,
        member: &str,
        args: Vec<JsonValue>,
        context: &Context,
    ) -> BoxFuture<Result<Option<JsonValue>, AnyError>> {
        match self.slot(member) {
            Ok(slot) => slot.invoke_method(args, context),
            Err(failure) => Box::pin(async move { Err(failure) }),
        }
    }

    fn state(&self, member: &str) -> Result<Option<Arc<dyn ReplicatedState>>, AnyError> {
        let slot = self.slot(member)?;
        (slot.assert_access)()?;
        slot.expect_kind("state")?;
        Ok(Some(Arc::new(GuardedState { slot })))
    }
}

/// Fencing wrapper handed to keyed observation handlers (upstream: the `ServiceSlot`
/// view with the observation-closed assertion).
struct ObservedServiceHandle {
    inner: Arc<ServiceFacade>,
    service_id: String,
    stopped: Arc<AtomicBool>,
    context: Context,
}

impl ObservedServiceHandle {
    fn assert_observation(&self) -> Result<(), AnyError> {
        let aborted = self
            .context
            .abort_signal()
            .is_some_and(|signal| signal.aborted());
        if self.stopped.load(Ordering::SeqCst) || aborted {
            return Err(remote_service_error(
                RemoteServiceErrorCode::ServiceStaleInstance,
                format!("Remote service {} observation is closed", self.service_id),
            ));
        }
        Ok(())
    }
}

impl ServiceClientHandle for ObservedServiceHandle {
    fn service_id(&self) -> &str {
        &self.service_id
    }

    fn invoke(
        &self,
        member: &str,
        args: Vec<JsonValue>,
        context: &Context,
    ) -> BoxFuture<Result<Option<JsonValue>, AnyError>> {
        match self.assert_observation() {
            Ok(()) => self.inner.invoke(member, args, context),
            Err(failure) => Box::pin(async move { Err(failure) }),
        }
    }

    fn state(&self, member: &str) -> Result<Option<Arc<dyn ReplicatedState>>, AnyError> {
        self.assert_observation()?;
        self.inner.state(member)
    }
}

/// One live keyed instance observed by the directory (upstream `KeyedInstance`).
#[derive(Clone)]
struct KeyedInstance {
    key: String,
    generation: u64,
    facade: Arc<ServiceFacade>,
    deactivate: Arc<dyn Fn() + Send + Sync>,
}

impl InstanceDirectoryEntry for KeyedInstance {
    fn key(&self) -> &str {
        &self.key
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn deactivate(&self) {
        (self.deactivate)();
    }
}

/// Keyed service binding (upstream `KeyedBinding<T>`).
struct KeyedBinding {
    service_id: String,
    transport: Arc<dyn RemoteServiceTransport>,
    report_error: arc_error::ErrorHandler,
    assert_access: AccessChecker,
    instances: Arc<InstanceDirectory<KeyedInstance>>,
    subscription: Mutex<Option<Box<dyn ServiceSubscription>>>,
    starting: Mutex<Option<Arc<StartingTask>>>,
    closed: Arc<AtomicBool>,
    bound: AtomicBool,
    revision: AtomicU64,
}

impl KeyedBinding {
    fn new(
        service_id: &str,
        transport: Arc<dyn RemoteServiceTransport>,
        report_error: arc_error::ErrorHandler,
        assert_access: AccessChecker,
        bound: bool,
    ) -> Arc<KeyedBinding> {
        let instances = Arc::new(InstanceDirectory::new(false, Arc::clone(&report_error)));
        Arc::new(KeyedBinding {
            service_id: service_id.to_string(),
            transport,
            report_error,
            assert_access,
            instances,
            subscription: Mutex::new(None),
            starting: Mutex::new(None),
            closed: Arc::new(AtomicBool::new(false)),
            bound: AtomicBool::new(bound),
            revision: AtomicU64::new(0),
        })
    }

    fn observe(me: &Arc<KeyedBinding>, handler: KeyedObservationHandler) -> Result<Unsubscribe, AnyError> {
        if me.closed.load(Ordering::SeqCst) {
            return Err(error("Remote keyed service binding is closed"));
        }
        let stopped = Arc::new(AtomicBool::new(false));
        let service_id = me.service_id.clone();
        let stopped_for_handler = Arc::clone(&stopped);
        let directory_handler: InstanceObserverHandler<KeyedInstance> =
            Arc::new(move |entry: KeyedInstance, context: Context| {
                let handle = Arc::new(ObservedServiceHandle {
                    inner: Arc::clone(&entry.facade),
                    service_id: service_id.clone(),
                    stopped: Arc::clone(&stopped_for_handler),
                    context: context.clone(),
                });
                let future = handler(handle, context);
                Box::pin(async move {
                    future.await;
                })
            });
        let directory_stop = me.instances.observe(directory_handler)?;
        if me.bound.load(Ordering::SeqCst) && lock(&me.starting).is_none() {
            let revision = me.revision.load(Ordering::SeqCst);
            let task = KeyedBinding::start(me, revision);
            *lock(&me.starting) = Some(task);
        }
        let stopped_for_stop = stopped;
        Ok(Box::new(move || {
            stopped_for_stop.store(true, Ordering::SeqCst);
            directory_stop();
        }))
    }

    async fn rebind(me: &Arc<KeyedBinding>, bound: bool, context: &Context) -> Result<(), AnyError> {
        if me.closed.load(Ordering::SeqCst) {
            return Ok(());
        }
        me.bound.store(bound, Ordering::SeqCst);
        let revision = me.revision.fetch_add(1, Ordering::SeqCst) + 1;
        KeyedBinding::reset(me, context, false).await;
        if me.closed.load(Ordering::SeqCst)
            || me.revision.load(Ordering::SeqCst) != revision
            || me.bound.load(Ordering::SeqCst) != bound
        {
            return Ok(());
        }
        if bound && me.instances.observer_count() > 0 {
            let task = KeyedBinding::start(me, revision);
            *lock(&me.starting) = Some(Arc::clone(&task));
            task.wait().await
        } else {
            Ok(())
        }
    }

    fn ready_task(&self) -> Option<Arc<StartingTask>> {
        lock(&self.starting).clone()
    }

    async fn close(&self, context: &Context) -> Result<(), AnyError> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        self.revision.fetch_add(1, Ordering::SeqCst);
        KeyedBinding::reset(self, context, true).await;
        self.instances.dispose();
        Ok(())
    }

    async fn reset(me: &KeyedBinding, context: &Context, wait_for_starting: bool) {
        me.instances.reset();
        let starting = lock(&me.starting).take();
        let subscription = lock(&me.subscription).take();
        if wait_for_starting {
            if let Some(task) = starting {
                let _ = task.wait().await;
            }
        }
        if let Some(subscription) = subscription {
            subscription.close(Some(context.clone())).await;
        }
    }

    fn start(me: &Arc<KeyedBinding>, revision: u64) -> Arc<StartingTask> {
        let transport = Arc::clone(&me.transport);
        let listener_binding = Arc::clone(me);
        let listener: UpdateListener = Arc::new(move |update: &ServiceProviderUpdate, context: &Context| {
            if listener_binding.revision.load(Ordering::SeqCst) != revision {
                return;
            }
            listener_binding.handle_update(update, context);
        });
        let on_error_at_settle = {
            let binding = Arc::clone(me);
            Box::new(move |failure: AnyError| {
                if !binding.closed.load(Ordering::SeqCst)
                    && binding.revision.load(Ordering::SeqCst) == revision
                    && binding.bound.load(Ordering::SeqCst)
                {
                    (binding.report_error)(failure);
                }
            }) as Box<dyn FnOnce(AnyError) + Send + Sync>
        };
        let binding = Arc::clone(me);
        let future: BoxFuture<Result<(), AnyError>> = Box::pin(async move {
            let subscription = transport
                .subscribe(&binding.service_id, ServiceMode::Keyed, listener, &BACKGROUND_CONTEXT)
                .await?;
            if binding.closed.load(Ordering::SeqCst)
                || !binding.bound.load(Ordering::SeqCst)
                || binding.revision.load(Ordering::SeqCst) != revision
            {
                subscription.close(None).await;
                return Ok(());
            }
            *lock(&binding.subscription) = Some(subscription);
            let snapshot = lock(&binding.subscription)
                .as_ref()
                .expect("subscription stored above")
                .snapshot()
                .clone();
            if snapshot.mode != ServiceMode::Keyed || snapshot.service_id != binding.service_id {
                return Err(error(format!(
                    "Remote service {} returned the wrong keyed snapshot",
                    binding.service_id
                )));
            }
            for instance_snapshot in &snapshot.instances {
                binding.spawn_instance(instance_snapshot, &service_delivery_context())?;
            }
            lock(&binding.subscription)
                .as_ref()
                .expect("subscription stored above")
                .activate();
            binding.instances.ready()?;
            Ok(())
        });
        spawn_starting(future, Some(on_error_at_settle))
    }

    fn handle_update(&self, update: &ServiceProviderUpdate, context: &Context) {
        if self.closed.load(Ordering::SeqCst) {
            return;
        }
        if let Err(failure) = self.apply_update(update, context) {
            (self.report_error)(failure);
        }
    }

    fn apply_update(&self, update: &ServiceProviderUpdate, context: &Context) -> Result<(), AnyError> {
        match update {
            ServiceProviderUpdate::Reset { snapshot } => {
                validate_reset_snapshot(snapshot, &self.service_id, ServiceMode::Keyed)?;
                let snapshots: HashMap<&str, &ServiceInstanceSnapshot> = snapshot
                    .instances
                    .iter()
                    .filter_map(|instance| {
                        instance
                            .instance
                            .as_ref()
                            .map(|address| (address.key.as_str(), instance))
                    })
                    .collect();
                for instance in self.instances.values() {
                    let retained = snapshots
                        .get(instance.key.as_str())
                        .and_then(|snapshot| snapshot.instance.as_ref())
                        .is_some_and(|address| address.generation == instance.generation);
                    if !retained {
                        self.instances.remove(&instance);
                    }
                }
                for instance_snapshot in &snapshot.instances {
                    let address = instance_snapshot
                        .instance
                        .as_ref()
                        .expect("validated by validate_reset_snapshot");
                    match self.instances.get(&address.key) {
                        None => self.spawn_instance(instance_snapshot, context)?,
                        Some(instance) => instance.facade.install(instance_snapshot, context)?,
                    }
                }
            }
            ServiceProviderUpdate::Unavailable | ServiceProviderUpdate::Replaced { .. } => {
                return Err(error("Keyed service received a singleton lifecycle update"));
            }
            ServiceProviderUpdate::Spawned { instance } => {
                self.spawn_instance(instance, context)?;
            }
            ServiceProviderUpdate::Closed { instance } => {
                if let Some(existing) = self.instances.get(&instance.key) {
                    if existing.generation == instance.generation {
                        self.instances.remove(&existing);
                    }
                }
            }
            ServiceProviderUpdate::State { instance: None, .. } => {
                return Err(error("Keyed state update has no instance address"));
            }
            ServiceProviderUpdate::State {
                instance: Some(address),
                member,
                sequence,
                ops,
            } => {
                if let Some(existing) = self.instances.get(&address.key) {
                    if existing.generation == address.generation {
                        existing.facade.update(member, *sequence, ops, context)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn spawn_instance(
        &self,
        snapshot: &ServiceInstanceSnapshot,
        context: &Context,
    ) -> Result<(), AnyError> {
        let Some(address) = &snapshot.instance else {
            return Err(error("Keyed service instance snapshot has no address"));
        };
        let active = Arc::new(AtomicBool::new(true));
        let is_active: IsActive = {
            let active = Arc::clone(&active);
            let closed = Arc::clone(&self.closed);
            Arc::new(move || active.load(Ordering::SeqCst) && !closed.load(Ordering::SeqCst))
        };
        let facade = ServiceFacade::new(
            &self.service_id,
            Some(address.clone()),
            Arc::clone(&self.transport),
            Arc::clone(&self.report_error),
            is_active,
            Arc::clone(&self.assert_access),
        );
        facade.install(snapshot, context)?;
        let deactivate = {
            let facade = Arc::clone(&facade);
            Arc::new(move || {
                active.store(false, Ordering::SeqCst);
                facade.clear();
            }) as Arc<dyn Fn() + Send + Sync>
        };
        self.instances.replace(KeyedInstance {
            key: address.key.clone(),
            generation: address.generation,
            facade,
            deactivate,
        })
    }
}

struct SingletonBinding {
    facade: Arc<ServiceFacade>,
    active: Arc<AtomicBool>,
    revision: AtomicU64,
    subscription: Mutex<Option<Box<dyn ServiceSubscription>>>,
    starting: Mutex<Option<Arc<StartingTask>>>,
}

#[derive(Default)]
struct BindingFlags {
    bound: AtomicBool,
    disposed: AtomicBool,
}

struct BindingInner {
    allowlist: HashSet<String>,
    modes: HashMap<String, ServiceMode>,
    singletons: HashMap<String, Arc<SingletonBinding>>,
    keyed: HashMap<String, Arc<KeyedBinding>>,
    readiness_revision: u64,
    binding_transition: Option<Arc<StartingTask>>,
}

/// Consumer binding over one remote service transport (upstream
/// `RemoteServiceBindingImpl`, exposed as `createRemoteServiceBinding`).
pub struct RemoteServiceBinding {
    flags: Arc<BindingFlags>,
    transport: Arc<dyn RemoteServiceTransport>,
    report_error: arc_error::ErrorHandler,
    assert_access: Option<AccessChecker>,
    inner: Arc<Mutex<BindingInner>>,
}

/// Coordinator-facing alias; sibling modules import the upstream impl name.
pub type RemoteServiceBindingImpl = RemoteServiceBinding;

impl RemoteServiceBinding {
    pub fn new(options: RemoteServiceBindingOptions) -> Result<RemoteServiceBinding, AnyError> {
        let mut seen = HashSet::new();
        for id in &options.services {
            if !seen.insert(id.as_str()) {
                return Err(type_error("Remote service binding has duplicate service IDs"));
            }
        }
        Ok(RemoteServiceBinding {
            flags: Arc::new(BindingFlags {
                bound: AtomicBool::new(options.bound),
                disposed: AtomicBool::new(false),
            }),
            transport: options.transport,
            report_error: options
                .on_error
                .unwrap_or_else(|| Arc::new(|_: AnyError| {})),
            assert_access: options.assert_access,
            inner: Arc::new(Mutex::new(BindingInner {
                allowlist: options.services.into_iter().collect(),
                modes: HashMap::new(),
                singletons: HashMap::new(),
                keyed: HashMap::new(),
                readiness_revision: 0,
                binding_transition: None,
            })),
        })
    }

    fn handle_access_checker(&self) -> AccessChecker {
        let flags = Arc::clone(&self.flags);
        let user = self.assert_access.clone();
        Arc::new(move || {
            if flags.disposed.load(Ordering::SeqCst) {
                return Err(error("Remote service binding is disposed"));
            }
            match &user {
                Some(check) => check(),
                None => Ok(()),
            }
        })
    }

    fn assert_available(&self, service_id: &str, mode: ServiceMode) -> Result<(), AnyError> {
        if self.flags.disposed.load(Ordering::SeqCst) {
            return Err(error("Remote service binding is disposed"));
        }
        if !lock(&self.inner).allowlist.contains(service_id) {
            return Err(remote_service_error(
                RemoteServiceErrorCode::ServiceNotAllowed,
                format!("Remote service {service_id} is not allowlisted"),
            ));
        }
        let mut inner = lock(&self.inner);
        match inner.modes.get(service_id) {
            Some(existing) if *existing != mode => {
                return Err(remote_service_error(
                    RemoteServiceErrorCode::ServiceModeMismatch,
                    format!(
                        "Remote service {service_id} is already used as {}",
                        existing.as_str()
                    ),
                ));
            }
            Some(_) => {}
            None => {
                inner.modes.insert(service_id.to_string(), mode);
            }
        }
        Ok(())
    }

    /// Reconnect or disconnect every acquired service (upstream `rebind`).
    pub async fn rebind(&self, bound: bool, context: &Context) -> Result<(), AnyError> {
        if self.flags.disposed.load(Ordering::SeqCst) {
            return Err(error("Remote service binding is disposed"));
        }
        self.flags.bound.store(bound, Ordering::SeqCst);
        let mut transitions: Vec<Arc<StartingTask>> = Vec::new();
        let keyed: Vec<Arc<KeyedBinding>> = {
            let mut inner = lock(&self.inner);
            inner.readiness_revision += 1;
            for (service_id, binding) in inner.singletons.iter() {
                let binding = Arc::clone(binding);
                let revision = binding.revision.fetch_add(1, Ordering::SeqCst) + 1;
                binding.facade.clear();
                let subscription = lock(&binding.subscription).take();
                let service_id = service_id.clone();
                let transport = Arc::clone(&self.transport);
                let flags = Arc::clone(&self.flags);
                let report_error = Arc::clone(&self.report_error);
                let context = context.clone();
                let binding_for_task = Arc::clone(&binding);
                let future: BoxFuture<Result<(), AnyError>> = Box::pin(async move {
                    if let Some(subscription) = subscription {
                        subscription.close(Some(context)).await;
                    }
                    if bound {
                        let task = start_singleton_task(
                            &transport,
                            &binding_for_task,
                            &service_id,
                            revision,
                            &flags,
                            &report_error,
                            None,
                        );
                        *lock(&binding_for_task.starting) = Some(Arc::clone(&task));
                        let _ = task.wait().await;
                    }
                    Ok(())
                });
                let task = spawn_starting(future, None);
                *lock(&binding.starting) = Some(Arc::clone(&task));
                transitions.push(task);
            }
            inner.keyed.values().cloned().collect()
        };
        for binding in keyed {
            let context = context.clone();
            let future: BoxFuture<Result<(), AnyError>> =
                Box::pin(async move { KeyedBinding::rebind(&binding, bound, &context).await });
            transitions.push(spawn_starting(future, None));
        }
        let completion = spawn_starting(
            Box::pin(async move {
                let mut errors: Vec<AnyError> = Vec::new();
                for task in transitions {
                    if let Err(failure) = task.wait().await {
                        errors.push(failure);
                    }
                }
                if errors.is_empty() {
                    Ok(())
                } else {
                    Err(AggregateError::new("Failed to rebind services", errors))
                }
            }),
            None,
        );
        lock(&self.inner).binding_transition = Some(Arc::clone(&completion));
        completion.wait().await
    }
}

/// Spawn one singleton subscription sequence (upstream `#startSingleton`); the
/// returned task is stored by the caller in `binding.starting`.
fn start_singleton_task(
    transport: &Arc<dyn RemoteServiceTransport>,
    binding: &Arc<SingletonBinding>,
    service_id: &str,
    revision: u64,
    flags: &Arc<BindingFlags>,
    report_error: &arc_error::ErrorHandler,
    on_error_at_settle: Option<Box<dyn FnOnce(AnyError) + Send + Sync>>,
) -> Arc<StartingTask> {
    let facade = Arc::clone(&binding.facade);
    let listener_active = Arc::clone(&binding.active);
    let listener_binding = Arc::clone(binding);
    let listener_report = Arc::clone(report_error);
    let listener_service_id = service_id.to_string();
    let listener: UpdateListener = Arc::new(move |update: &ServiceProviderUpdate, context: &Context| {
        if !listener_active.load(Ordering::SeqCst)
            || listener_binding.revision.load(Ordering::SeqCst) != revision
        {
            return;
        }
        if let Err(failure) =
            handle_singleton_update(&listener_binding, &listener_service_id, update, context)
        {
            listener_report(failure);
        }
    });
    let flags_for_task = Arc::clone(flags);
    let binding_for_task = Arc::clone(binding);
    let transport = Arc::clone(transport);
    let service_id = service_id.to_string();
    let future: BoxFuture<Result<(), AnyError>> = Box::pin(async move {
        let subscription = transport
            .subscribe(&service_id, ServiceMode::Singleton, listener, &BACKGROUND_CONTEXT)
            .await?;
        if !binding_for_task.active.load(Ordering::SeqCst)
            || flags_for_task.disposed.load(Ordering::SeqCst)
            || !flags_for_task.bound.load(Ordering::SeqCst)
            || binding_for_task.revision.load(Ordering::SeqCst) != revision
        {
            subscription.close(None).await;
            return Ok(());
        }
        *lock(&binding_for_task.subscription) = Some(subscription);
        let snapshot = lock(&binding_for_task.subscription)
            .as_ref()
            .expect("subscription stored above")
            .snapshot()
            .clone();
        if snapshot.mode != ServiceMode::Singleton
            || snapshot.service_id != service_id
            || snapshot.instances.len() != 1
        {
            return Err(error(format!(
                "Remote service {service_id} returned an invalid singleton snapshot"
            )));
        }
        let install_result = facade.install(&snapshot.instances[0], &service_delivery_context());
        install_result?;
        lock(&binding_for_task.subscription)
            .as_ref()
            .expect("subscription stored above")
            .activate();
        Ok(())
    });
    spawn_starting(future, on_error_at_settle)
}

fn handle_singleton_update(
    binding: &SingletonBinding,
    service_id: &str,
    update: &ServiceProviderUpdate,
    context: &Context,
) -> Result<(), AnyError> {
    match update {
        ServiceProviderUpdate::Reset { snapshot } => {
            validate_reset_snapshot(snapshot, service_id, ServiceMode::Singleton)?;
            match snapshot.instances.first() {
                None => binding.facade.clear(),
                Some(instance) => binding.facade.install(instance, context)?,
            }
        }
        ServiceProviderUpdate::Unavailable => binding.facade.clear(),
        ServiceProviderUpdate::Replaced { snapshot } => {
            if snapshot.instance.is_some() {
                return Err(error("Singleton replacement has an instance address"));
            }
            binding.facade.install(snapshot, context)?;
        }
        ServiceProviderUpdate::State {
            instance: None,
            member,
            sequence,
            ops,
        } => binding.facade.update(member, *sequence, ops, context)?,
        _ => {}
    }
    Ok(())
}

#[async_trait]
impl RemoteServices for RemoteServiceBinding {
    fn use_service(&self, service_id: &str) -> Result<Arc<dyn ServiceClientHandle>, AnyError> {
        self.assert_available(service_id, ServiceMode::Singleton)?;
        let mut inner = lock(&self.inner);
        if let Some(binding) = inner.singletons.get(service_id) {
            return Ok(Arc::clone(&binding.facade) as Arc<dyn ServiceClientHandle>);
        }
        let active = Arc::new(AtomicBool::new(true));
        let is_active: IsActive = {
            let active = Arc::clone(&active);
            let flags = Arc::clone(&self.flags);
            Arc::new(move || {
                active.load(Ordering::SeqCst)
                    && !flags.disposed.load(Ordering::SeqCst)
                    && flags.bound.load(Ordering::SeqCst)
            })
        };
        let facade = ServiceFacade::new(
            service_id,
            None,
            Arc::clone(&self.transport),
            Arc::clone(&self.report_error),
            is_active,
            self.handle_access_checker(),
        );
        let binding = Arc::new(SingletonBinding {
            facade: Arc::clone(&facade),
            active,
            revision: AtomicU64::new(0),
            subscription: Mutex::new(None),
            starting: Mutex::new(None),
        });
        inner.singletons.insert(service_id.to_string(), Arc::clone(&binding));
        inner.readiness_revision += 1;
        if self.flags.bound.load(Ordering::SeqCst) {
            let revision = binding.revision.load(Ordering::SeqCst);
            let on_error_at_settle = {
                let binding = Arc::clone(&binding);
                let flags = Arc::clone(&self.flags);
                let report_error = Arc::clone(&self.report_error);
                Box::new(move |failure: AnyError| {
                    if binding.active.load(Ordering::SeqCst)
                        && binding.revision.load(Ordering::SeqCst) == revision
                        && !flags.disposed.load(Ordering::SeqCst)
                        && flags.bound.load(Ordering::SeqCst)
                    {
                        report_error(failure);
                    }
                }) as Box<dyn FnOnce(AnyError) + Send + Sync>
            };
            let task = start_singleton_task(
                &self.transport,
                &binding,
                service_id,
                revision,
                &self.flags,
                &self.report_error,
                Some(on_error_at_settle),
            );
            *lock(&binding.starting) = Some(task);
        }
        Ok(facade as Arc<dyn ServiceClientHandle>)
    }

    fn observe(&self, service_id: &str, handler: KeyedObservationHandler) -> Result<Unsubscribe, AnyError> {
        self.assert_available(service_id, ServiceMode::Keyed)?;
        let binding = {
            let mut inner = lock(&self.inner);
            match inner.keyed.get(service_id) {
                Some(existing) => Arc::clone(existing),
                None => {
                    let binding = KeyedBinding::new(
                        service_id,
                        Arc::clone(&self.transport),
                        Arc::clone(&self.report_error),
                        self.handle_access_checker(),
                        self.flags.bound.load(Ordering::SeqCst),
                    );
                    inner.keyed.insert(service_id.to_string(), Arc::clone(&binding));
                    inner.readiness_revision += 1;
                    binding
                }
            }
        };
        let inner_stop = KeyedBinding::observe(&binding, handler)?;
        let binding_weak = Arc::downgrade(&binding);
        let inner_arc = Arc::clone(&self.inner);
        let report_error = Arc::clone(&self.report_error);
        let service_id = service_id.to_string();
        // `Unsubscribe` is `FnOnce`, so a re-entry guard is unnecessary (upstream's
        // `stopped` flag exists because JS closures can fire repeatedly).
        Ok(Box::new(move || {
            inner_stop();
            let Some(binding) = binding_weak.upgrade() else {
                return;
            };
            if binding.instances.observer_count() != 0 {
                return;
            }
            let mut inner = lock(&inner_arc);
            let is_current = inner
                .keyed
                .get(&service_id)
                .map(|current| Arc::ptr_eq(current, &binding))
                .unwrap_or(false);
            if !is_current {
                return;
            }
            inner.keyed.remove(&service_id);
            inner.readiness_revision += 1;
            drop(inner);
            let _ = tokio::spawn(async move {
                if let Err(failure) = binding.close(&BACKGROUND_CONTEXT).await {
                    report_error(failure);
                }
            });
        }))
    }

    async fn ready(&self, context: &Context) -> Result<(), AnyError> {
        loop {
            if self.flags.disposed.load(Ordering::SeqCst) {
                return Err(error("Remote service binding is disposed"));
            }
            let (revision, starts) = {
                let inner = lock(&self.inner);
                let mut starts: Vec<Arc<StartingTask>> = Vec::new();
                if let Some(transition) = &inner.binding_transition {
                    starts.push(Arc::clone(transition));
                }
                for binding in inner.singletons.values() {
                    if let Some(task) = lock(&binding.starting).clone() {
                        starts.push(task);
                    }
                }
                for binding in inner.keyed.values() {
                    if let Some(task) = binding.ready_task() {
                        starts.push(task);
                    }
                }
                (inner.readiness_revision, starts)
            };
            let waits = starts.iter().map(|task| task.wait());
            let _: Vec<()> = await_with_context(try_join_all(waits), context).await??;
            if self.flags.disposed.load(Ordering::SeqCst) {
                return Err(error("Remote service binding is disposed"));
            }
            if revision == lock(&self.inner).readiness_revision {
                return Ok(());
            }
        }
    }

    async fn dispose(&self, context: &Context) -> Result<(), AnyError> {
        if self.flags.disposed.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let (singleton_closes, keyed): (Vec<BoxFuture<()>>, Vec<Arc<KeyedBinding>>) = {
            let mut inner = lock(&self.inner);
            let mut closes: Vec<BoxFuture<()>> = Vec::new();
            let mut keyed = Vec::new();
            for binding in inner.singletons.values() {
                binding.active.store(false, Ordering::SeqCst);
                binding.facade.clear();
                if let Some(task) = lock(&binding.starting).clone() {
                    closes.push(Box::pin(async move {
                        let _ = task.wait().await;
                    }));
                }
                if let Some(subscription) = lock(&binding.subscription).take() {
                    let context = context.clone();
                    closes.push(Box::pin(async move {
                        subscription.close(Some(context)).await;
                    }));
                }
            }
            for binding in inner.keyed.values() {
                keyed.push(Arc::clone(binding));
            }
            inner.singletons.clear();
            inner.keyed.clear();
            (closes, keyed)
        };
        for close in singleton_closes {
            close.await;
        }
        let mut errors: Vec<AnyError> = Vec::new();
        for binding in keyed {
            if let Err(failure) = binding.close(context).await {
                errors.push(failure);
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(AggregateError::new("Failed to dispose services", errors))
        }
    }
}

fn validate_reset_snapshot(
    snapshot: &ServiceSubscriptionSnapshot,
    service_id: &str,
    mode: ServiceMode,
) -> Result<(), AnyError> {
    if snapshot.service_id != service_id
        || snapshot.mode != mode
        || (mode == ServiceMode::Singleton && snapshot.instances.len() > 1)
    {
        return Err(error("Remote service reset has the wrong service or mode"));
    }
    let mut keys = HashSet::new();
    for instance in &snapshot.instances {
        if mode == ServiceMode::Singleton {
            if instance.instance.is_some() {
                return Err(error("Remote service reset has an invalid instance address"));
            }
        } else if instance.instance.is_none() {
            return Err(error("Remote service reset has an invalid instance address"));
        }
        if let Some(address) = &instance.instance {
            if !keys.insert(address.key.as_str()) {
                return Err(error("Remote service reset repeats an instance key"));
            }
        }
        for member in &instance.members {
            if let ServiceMemberSnapshot::State { ops, .. } = member {
                let is_root_replacement = ops.len() == 1 && matches!(ops.first(), Some(Op::Replace(_)));
                if !is_root_replacement {
                    return Err(error("Remote service reset must contain full root replacements"));
                }
            }
        }
    }
    Ok(())
}

fn validate_members(members: &[ServiceMemberSnapshot]) -> Result<BTreeMap<String, &'static str>, AnyError> {
    let mut result = BTreeMap::new();
    for member in members {
        if member.name().is_empty() || result.contains_key(member.name()) {
            return Err(error("Remote service has invalid member descriptions"));
        }
        result.insert(member.name().to_string(), member.kind());
    }
    Ok(result)
}

fn same_address(left: Option<&ServiceInstanceAddress>, right: Option<&ServiceInstanceAddress>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => left.key == right.key && left.generation == right.generation,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    //! Port of the binding halves of upstream `test/services.test.ts` plus
    //! `test/service-delivery.test.ts` (provider halves live in `provider::tests`).
    //! Upstream helper transports are re-created here; loopback connects the
    //! provider to the binding in-process like upstream `helpers.ts`.

    use super::*;
    use crate::context::BACKGROUND_CONTEXT;
    use crate::services::loopback::create_loopback_service_transport;
    use crate::services::provider::tests::{object_with_state, update_kind as provider_update_kind, FakeState};
    use crate::services::provider::{ProviderDefinition, RemoteServiceProvider};
    use crate::types::{RemoteServiceTransport, RemoteServices, ServiceMemberSnapshot};
    use serde_json::json;
    use std::sync::Arc as StdArc;

    fn default_binding_options(
        services: &[&str],
        transport: Arc<dyn RemoteServiceTransport>,
    ) -> RemoteServiceBindingOptions {
        RemoteServiceBindingOptions {
            services: services.iter().map(|id| id.to_string()).collect(),
            transport,
            bound: true,
            on_error: None,
            assert_access: None,
        }
    }

    /// `Result::unwrap_err` for non-`Debug` success types.
    fn err_of<T>(result: Result<T, AnyError>) -> AnyError {
        match result {
            Err(error) => error,
            Ok(_) => panic!("expected an error"),
        }
    }

    fn error_sink() -> (crate::types::arc_error::ErrorHandler, Arc<Mutex<Vec<String>>>) {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&errors);
        (
            Arc::new(move |error: AnyError| sink.lock().unwrap().push(error.to_string())),
            errors,
        )
    }

    fn method_noop() -> crate::services::provider::tests::TestMethod {
        Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }))
    }

    /// Let lazily spawned start sequences run to completion on the current-thread
    /// test runtime (upstream starts these synchronously inside promises).
    async fn flush_tasks() {
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
    }

    async fn wait_for(label: &str, condition: impl Fn() -> bool) {
        for _ in 0..2000 {
            if condition() {
                return;
            }
            tokio::task::yield_now().await;
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        panic!("condition was not met: {label}");
    }

    async fn singleton_provider() -> (StdArc<RemoteServiceProvider>, Arc<FakeState>) {
        let provider = StdArc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new("test.models", ServiceMode::Singleton)])
                .expect("provider"),
        );
        let state = FakeState::new(json!({ "selected": null, "revision": 0 }));
        let received = Arc::new(Mutex::new(Vec::<(JsonValue, Context)>::new()));
        let sink = Arc::clone(&received);
        let revision = StdArc::new(AtomicU64::new(0));
        let method: crate::services::provider::tests::TestMethod = {
            let state = Arc::clone(&state);
            Box::new(move |mut args: Vec<JsonValue>, context: Context| {
                let model = if args.is_empty() { JsonValue::Null } else { args.remove(0) };
                sink.lock().unwrap().push((model.clone(), context.clone()));
                let next = revision.fetch_add(1, Ordering::SeqCst) + 1;
                state.replace_with(json!({ "selected": model, "revision": next }), &context);
                Box::pin(async move { Ok(None) })
            })
        };
        provider
            .provide("test.models", object_with_state(&state, "select", method))
            .expect("provide");
        (provider, state)
    }

    #[tokio::test]
    async fn echoes_payloads_through_the_loopback_transport() {
        // Port of "does not defensively clone method arguments or results": value
        // identity is unportable (Rust hands each side owned JSON trees), so the
        // round trip is asserted by content.
        let provider = StdArc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new("test.echo", ServiceMode::Singleton)])
                .expect("provider"),
        );
        let received = Arc::new(Mutex::new(Vec::<JsonValue>::new()));
        let sink = Arc::clone(&received);
        let method: crate::services::provider::tests::TestMethod =
            Box::new(move |mut args: Vec<JsonValue>, _context: Context| {
                let payload = args.remove(0);
                sink.lock().unwrap().push(payload.clone());
                Box::pin(async move { Ok(Some(json!({ "value": "response" }))) })
            });
        let state = FakeState::new(json!({}));
        provider
            .provide("test.echo", object_with_state(&state, "echo", method))
            .expect("provide");

        let binding = RemoteServiceBinding::new(default_binding_options(
            &["test.echo"],
            create_loopback_service_transport(StdArc::clone(&provider)),
        ))
        .expect("binding");
        let echo = binding.use_service("test.echo").expect("use");
        binding.ready(&BACKGROUND_CONTEXT).await.expect("ready");
        let response = echo
            .invoke("echo", vec![json!({ "value": "request" })], &BACKGROUND_CONTEXT)
            .await
            .expect("invoke");
        assert_eq!(response, Some(json!({ "value": "response" })));
        assert_eq!(*lock(&received), vec![json!({ "value": "request" })]);

        binding.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
        provider.dispose().expect("dispose");
    }

    #[tokio::test]
    async fn provides_and_consumes_one_singleton_with_replicated_state() {
        let (provider, _state) = singleton_provider().await;
        assert_eq!(
            serde_json::to_value(provider.catalogue()).expect("catalogue"),
            json!([{ "serviceId": "test.models", "mode": "singleton" }])
        );
        let initial = json!({ "selected": null, "revision": 0 });
        let (on_error, errors) = error_sink();
        let mut options = default_binding_options(
            &["test.models"],
            create_loopback_service_transport(StdArc::clone(&provider)),
        );
        options.on_error = Some(on_error);
        let namespace = RemoteServiceBinding::new(options).expect("binding");

        let first = namespace.use_service("test.models").expect("use");
        let second = namespace.use_service("test.models").expect("use");
        let first_erased: Arc<dyn ServiceClientHandle> = first.clone();
        let second_erased: Arc<dyn ServiceClientHandle> = second.clone();
        assert!(Arc::ptr_eq(&first_erased, &second_erased));
        let first_state = first.state("state").expect("state").expect("state member");
        assert!(first_state.value().is_none());
        namespace.ready(&BACKGROUND_CONTEXT).await.expect("ready");
        assert_eq!(first_state.value(), Some(&initial));

        let updates: Arc<Mutex<Vec<JsonValue>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&updates);
        let unsubscribe = second_state_subscribes(&second, move |value: &JsonValue| {
            sink.lock().unwrap().push(value.clone());
        });
        first
            .invoke(
                "select",
                vec![json!({ "provider": "test", "modelId": "one" })],
                &BACKGROUND_CONTEXT,
            )
            .await
            .expect("select");
        assert_eq!(first_state.value(), Some(&json!({
            "selected": { "provider": "test", "modelId": "one" },
            "revision": 1,
        })));
        assert_eq!(
            *lock(&updates),
            vec![initial, json!({ "selected": { "provider": "test", "modelId": "one" }, "revision": 1 })]
        );
        assert!(lock(&errors).is_empty());

        let late_namespace = RemoteServiceBinding::new(default_binding_options(
            &["test.models"],
            create_loopback_service_transport(StdArc::clone(&provider)),
        ))
        .expect("late binding");
        let late = late_namespace.use_service("test.models").expect("late use");
        late_namespace.ready(&BACKGROUND_CONTEXT).await.expect("late ready");
        let late_state = late.state("state").expect("late state").expect("state member");
        assert_eq!(late_state.value().expect("state hydrated")["revision"], json!(1));

        unsubscribe();
        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
        late_namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
        provider.dispose().expect("dispose");
    }

    fn second_state_subscribes(
        handle: &Arc<dyn ServiceClientHandle>,
        on_value: impl Fn(&JsonValue) + Send + Sync + 'static,
    ) -> Unsubscribe {
        let state = handle.state("state").expect("state").expect("state member");
        let on_value = StdArc::new(on_value);
        state.subscribe(Arc::new(move |value: &JsonValue, _context: &Context, _delivery: &crate::types::ReplicatedStateDelivery| {
            let on_value = StdArc::clone(&on_value);
            let value = value.clone();
            Box::pin(async move { on_value(&value) }) as BoxFuture<()>
        }))
    }

    #[tokio::test]
    async fn publishes_compact_tracked_operations_through_the_binding() {
        let provider = StdArc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new("test.timeline", ServiceMode::Singleton)])
                .expect("provider"),
        );
        let initial = json!({ "entries": [{ "id": "one" }], "retained": { "value": 1 } });
        let state = FakeState::new(initial.clone());
        provider
            .provide("test.timeline", object_with_state(&state, "extend", method_noop()))
            .expect("provide");
        let updates: Arc<Mutex<Vec<ServiceProviderUpdate>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&updates);
        let raw = provider
            .subscribe(
                "test.timeline",
                ServiceMode::Singleton,
                Arc::new(move |update: &ServiceProviderUpdate, _| sink.lock().unwrap().push(update.clone())),
            )
            .expect("subscribe");
        assert_eq!(
            serde_json::to_value(raw.snapshot()).expect("snapshot")["instances"][0]["members"],
            json!([
                { "name": "extend", "kind": "method" },
                { "name": "state", "kind": "state", "sequence": 0, "ops": [["r", initial]] },
            ])
        );
        raw.activate();

        let namespace = RemoteServiceBinding::new(default_binding_options(
            &["test.timeline"],
            create_loopback_service_transport(StdArc::clone(&provider)),
        ))
        .expect("binding");
        let timeline = namespace.use_service("test.timeline").expect("use");
        namespace.ready(&BACKGROUND_CONTEXT).await.expect("ready");
        let timeline_state = timeline.state("state").expect("state").expect("state member");
        assert_eq!(
            timeline_state.value(),
            Some(&json!({ "entries": [{ "id": "one" }], "retained": { "value": 1 } }))
        );
        state.publish(
            vec![Op::Splice {
                path: vec![crate::delta::Seg::Key("entries".into())],
                start: 1,
                remove: 0,
                items: vec![json!({ "id": "two" })],
            }],
            &BACKGROUND_CONTEXT,
        );
        assert_eq!(
            timeline_state.value(),
            Some(&json!({ "entries": [{ "id": "one" }, { "id": "two" }], "retained": { "value": 1 } }))
        );
        assert!(lock(&updates).iter().any(|update| {
            serde_json::to_value(update).expect("update")
                == json!({
                    "type": "state",
                    "member": "state",
                    "sequence": 1,
                    "ops": [["p", ["entries"], 1, 0, [{ "id": "two" }]]],
                })
        }));

        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
        futures::executor::block_on(raw.close(None));
        provider.dispose().expect("dispose");
    }

    #[tokio::test]
    async fn keeps_singleton_facades_stable_when_their_provider_is_replaced() {
        let provider = StdArc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new("test.models", ServiceMode::Singleton)])
                .expect("provider"),
        );
        let initial_state = FakeState::new(json!({ "selected": null, "revision": 1 }));
        provider
            .provide(
                "test.models",
                object_with_state(&initial_state, "select", method_noop()),
            )
            .expect("provide");
        let namespace = RemoteServiceBinding::new(default_binding_options(
            &["test.models"],
            create_loopback_service_transport(StdArc::clone(&provider)),
        ))
        .expect("binding");
        let models = namespace.use_service("test.models").expect("use");
        let state = models.state("state").expect("state").expect("state member");
        namespace.ready(&BACKGROUND_CONTEXT).await.expect("ready");
        assert_eq!(state.value().expect("state hydrated")["revision"], json!(1));

        let replacement_calls = Arc::new(Mutex::new(0usize));
        provider.withdraw("test.models").expect("withdraw");
        assert!(state.value().is_none());
        let error = err_of(models
            .invoke(
                "select",
                vec![json!({ "provider": "test", "modelId": "unavailable" })],
                &BACKGROUND_CONTEXT,
            )
            .await
            );
        assert_eq!(error.to_string(), "Remote service test.models has no provider");

        let calls_for_replacement = Arc::clone(&replacement_calls);
        let replacement_method: crate::services::provider::tests::TestMethod =
            Box::new(move |_args: Vec<JsonValue>, _context: Context| {
                *calls_for_replacement.lock().unwrap() += 1;
                Box::pin(async { Ok(None) })
            });
        let replacement_state = FakeState::new(json!({ "selected": null, "revision": 2 }));
        provider
            .replace(
                "test.models",
                object_with_state(&replacement_state, "select", replacement_method),
            )
            .expect("replace");

        let reused = namespace.use_service("test.models").expect("reuse");
        let models_erased: Arc<dyn ServiceClientHandle> = models.clone();
        let reused_erased: Arc<dyn ServiceClientHandle> = reused.clone();
        assert!(Arc::ptr_eq(&models_erased, &reused_erased));
        assert_eq!(state.value().expect("state hydrated")["revision"], json!(2));
        models
            .invoke(
                "select",
                vec![json!({ "provider": "test", "modelId": "replacement" })],
                &BACKGROUND_CONTEXT,
            )
            .await
            .expect("select hits the replacement");
        assert_eq!(*lock(&replacement_calls), 1);

        let mismatched: Arc<dyn crate::types::RemoteServiceImpl> =
            crate::services::provider::RemoteServiceObject::builder()
                .method("select", method_noop())
                .build();
        let error = err_of(provider.replace("test.models", mismatched));
        assert!(error.to_string().contains("replacement must preserve its member shape"));

        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
        provider.dispose().expect("dispose");
    }

    #[tokio::test]
    async fn clears_retained_facades_when_providers_and_bindings_are_disposed() {
        let provider = StdArc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new("test.models", ServiceMode::Singleton)])
                .expect("provider"),
        );
        let initial_state = FakeState::new(json!({ "selected": null, "revision": 1 }));
        provider
            .provide("test.models", object_with_state(&initial_state, "select", method_noop()))
            .expect("provide");
        let namespace = RemoteServiceBinding::new(default_binding_options(
            &["test.models"],
            create_loopback_service_transport(StdArc::clone(&provider)),
        ))
        .expect("binding");
        let models = namespace.use_service("test.models").expect("use");
        let state = models.state("state").expect("state").expect("state member");
        namespace.ready(&BACKGROUND_CONTEXT).await.expect("ready");
        assert_eq!(state.value().expect("state hydrated")["revision"], json!(1));

        provider.dispose().expect("dispose");
        assert!(state.value().is_none());
        let error = err_of(models
            .invoke("select", vec![json!({"provider": "test", "modelId": "one"})], &BACKGROUND_CONTEXT)
            .await
            );
        assert_eq!(error.to_string(), "Remote service provider is disposed");

        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
        let error = err_of(models.state("state"));
        assert_eq!(error.to_string(), "Remote service binding is disposed");
    }

    #[tokio::test]
    async fn applies_provider_disposal_buffered_while_subscriptions_are_starting() {
        let provider = StdArc::new(
            RemoteServiceProvider::new(vec![
                ProviderDefinition::new("test.models", ServiceMode::Singleton),
                ProviderDefinition::new("test.question-dialog", ServiceMode::Keyed),
            ])
            .expect("provider"),
        );
        let models_state = FakeState::new(json!({ "selected": null, "revision": 1 }));
        provider
            .provide("test.models", object_with_state(&models_state, "select", method_noop()))
            .expect("provide");
        let dialog_state = FakeState::new(json!({ "question": "Pending?" }));
        let _close = provider
            .spawn("test.question-dialog", "pending", object_with_state(&dialog_state, "submit", method_noop()))
            .expect("spawn");

        let (on_error, errors) = error_sink();
        let mut options = default_binding_options(
            &["test.models", "test.question-dialog"],
            create_loopback_service_transport(StdArc::clone(&provider)),
        );
        options.on_error = Some(on_error);
        let namespace = RemoteServiceBinding::new(options).expect("binding");
        let models = namespace.use_service("test.models").expect("use");
        let observed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&observed);
        // Gate the handler body: upstream leaves the observation callback queued as a
        // microtask and disposes before it ever runs; the never-notified gate parks
        // the spawned handler at its first await so dispose aborts it deterministically.
        let gate: StdArc<tokio::sync::Notify> = StdArc::new(tokio::sync::Notify::new());
        let handler_gate = StdArc::clone(&gate);
        let _observation = namespace
            .observe(
                "test.question-dialog",
                Arc::new(move |handle: Arc<dyn ServiceClientHandle>, _context: Context| {
                    let sink = Arc::clone(&sink);
                    let gate = StdArc::clone(&handler_gate);
                    Box::pin(async move {
                        gate.notified().await;
                        sink.lock().unwrap().push(handle.service_id().to_string());
                    }) as BoxFuture<()>
                }),
            )
            .expect("observe");

        // Upstream starts these subscriptions synchronously inside the promise; the
        // Rust tasks are lazy, so flush them before disposing to reproduce ordering.
        flush_tasks().await;
        provider.dispose().expect("dispose");
        namespace.ready(&BACKGROUND_CONTEXT).await.expect("ready");
        let models_state = models.state("state").expect("state").expect("state member");
        assert!(models_state.value().is_none());
        assert!(lock(&observed).is_empty());
        assert!(lock(&errors).is_empty(), "errors: {:?}", lock(&errors));

        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
    }

    #[tokio::test]
    async fn keeps_deferred_service_handles_inaccessible_until_host_activation() {
        let provider = StdArc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new("test.models", ServiceMode::Singleton)])
                .expect("provider"),
        );
        let initial_state = FakeState::new(json!({ "selected": null, "revision": 0 }));
        provider
            .provide("test.models", object_with_state(&initial_state, "select", method_noop()))
            .expect("provide");
        let mut options = default_binding_options(
            &["test.models"],
            create_loopback_service_transport(StdArc::clone(&provider)),
        );
        options.bound = false;
        let active = Arc::new(AtomicBool::new(false));
        let active_for_checker = Arc::clone(&active);
        options.assert_access = Some(Arc::new(move || {
            if active_for_checker.load(Ordering::SeqCst) {
                Ok(())
            } else {
                Err(error("Service handles are not active"))
            }
        }));
        let namespace = RemoteServiceBinding::new(options).expect("binding");
        let models = namespace.use_service("test.models").expect("use");

        let error = err_of(models.state("state"));
        assert_eq!(error.to_string(), "Service handles are not active");
        let error = err_of(models
            .invoke("select", vec![json!({ "provider": "test", "modelId": "one" })], &BACKGROUND_CONTEXT)
            .await
            );
        assert_eq!(error.to_string(), "Service handles are not active");

        namespace.rebind(true, &BACKGROUND_CONTEXT).await.expect("rebind");
        active.store(true, Ordering::SeqCst);
        namespace.ready(&BACKGROUND_CONTEXT).await.expect("ready");
        let state = models.state("state").expect("state").expect("state member");
        assert_eq!(state.value().expect("state hydrated")["revision"], json!(0));

        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
        provider.dispose().expect("dispose");
    }

    #[tokio::test]
    async fn rejects_namespace_readiness_when_initial_hydration_fails() {
        struct FailingTransport;
        #[async_trait]
        impl RemoteServiceTransport for FailingTransport {
            async fn invoke(&self, _call: &crate::types::ServiceCall, _context: &Context) -> Result<Option<JsonValue>, AnyError> {
                Err(error("unexpected invocation"))
            }
            async fn subscribe(
                &self,
                _service_id: &str,
                _mode: ServiceMode,
                _listener: UpdateListener,
                _context: &Context,
            ) -> Result<Box<dyn ServiceSubscription>, AnyError> {
                Err(error("initial hydration failed"))
            }
        }
        let (on_error, errors) = error_sink();
        let mut options = default_binding_options(&["test.models"], StdArc::new(FailingTransport));
        options.on_error = Some(on_error);
        let namespace = RemoteServiceBinding::new(options).expect("binding");
        let models = namespace.use_service("test.models").expect("use");

        let error = err_of(namespace.ready(&BACKGROUND_CONTEXT).await);
        assert_eq!(error.to_string(), "initial hydration failed");
        let state = models.state("state").expect("state").expect("state member");
        assert!(state.value().is_none());
        assert_eq!(*lock(&errors), vec!["initial hydration failed".to_string()]);
        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
    }

    #[tokio::test]
    async fn buffers_state_updates_that_race_subscription_hydration() {
        struct RacingTransport {
            provider: StdArc<RemoteServiceProvider>,
            state: Arc<FakeState>,
        }
        #[async_trait]
        impl RemoteServiceTransport for RacingTransport {
            async fn invoke(&self, call: &crate::types::ServiceCall, context: &Context) -> Result<Option<JsonValue>, AnyError> {
                self.provider.invoke(call, context).await
            }
            async fn subscribe(
                &self,
                service_id: &str,
                mode: ServiceMode,
                listener: UpdateListener,
                _context: &Context,
            ) -> Result<Box<dyn ServiceSubscription>, AnyError> {
                let subscription = self.provider.subscribe(service_id, mode, listener)?;
                self.state.replace_with(json!({ "selected": null, "revision": 1 }), &BACKGROUND_CONTEXT);
                Ok(subscription)
            }
        }
        let (provider, state) = singleton_provider().await;
        let namespace = RemoteServiceBinding::new(default_binding_options(
            &["test.models"],
            StdArc::new(RacingTransport { provider: StdArc::clone(&provider), state }),
        ))
        .expect("binding");
        let models = namespace.use_service("test.models").expect("use");
        let revisions: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&revisions);
        let state_view = models.state("state").expect("state").expect("state member");
        let unsubscribe = state_view.subscribe(Arc::new(move |value: &JsonValue, _context: &Context, _delivery: &crate::types::ReplicatedStateDelivery| {
            let sink = Arc::clone(&sink);
            let revision = value["revision"].as_i64();
            Box::pin(async move {
                if let Some(revision) = revision {
                    sink.lock().unwrap().push(revision);
                }
            }) as BoxFuture<()>
        }));
        namespace.ready(&BACKGROUND_CONTEXT).await.expect("ready");
        assert_eq!(*lock(&revisions), vec![0, 1]);
        assert_eq!(state_view.value().expect("state hydrated")["revision"], json!(1));
        unsubscribe();
        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
        provider.dispose().expect("dispose");
    }

    #[tokio::test]
    async fn clears_replicated_state_after_a_duplicate_operation_sequence() {
        assert_sequence_corruption_clears_state(0).await;
    }

    #[tokio::test]
    async fn clears_replicated_state_after_a_gap_operation_sequence() {
        assert_sequence_corruption_clears_state(2).await;
    }

    async fn assert_sequence_corruption_clears_state(sequence: u64) {
        struct ManualTransport {
            snapshot: ServiceSubscriptionSnapshot,
            send_update: Mutex<Option<UpdateListener>>,
        }
        #[async_trait]
        impl RemoteServiceTransport for ManualTransport {
            async fn invoke(&self, _call: &crate::types::ServiceCall, _context: &Context) -> Result<Option<JsonValue>, AnyError> {
                Err(error("unexpected invocation"))
            }
            async fn subscribe(
                &self,
                _service_id: &str,
                _mode: ServiceMode,
                listener: UpdateListener,
                _context: &Context,
            ) -> Result<Box<dyn ServiceSubscription>, AnyError> {
                *lock(&self.send_update) = Some(listener);
                Ok(Box::new(StaticSubscription {
                    snapshot: self.snapshot.clone(),
                }))
            }
        }
        struct StaticSubscription {
            snapshot: ServiceSubscriptionSnapshot,
        }
        #[async_trait]
        impl ServiceSubscription for StaticSubscription {
            fn snapshot(&self) -> &ServiceSubscriptionSnapshot {
                &self.snapshot
            }
            fn activate(&self) {}
            async fn close(&self, _context: Option<Context>) {}
        }

        let snapshot = ServiceSubscriptionSnapshot {
            service_id: "test.models".into(),
            mode: ServiceMode::Singleton,
            instances: vec![ServiceInstanceSnapshot {
                instance: None,
                members: vec![
                    ServiceMemberSnapshot::Method { name: "select".into() },
                    ServiceMemberSnapshot::State {
                        name: "state".into(),
                        sequence: 0,
                        ops: vec![Op::Replace(json!({ "selected": null, "revision": 0 }))],
                    },
                ],
            }],
        };
        let (on_error, errors) = error_sink();
        let transport = StdArc::new(ManualTransport {
            snapshot,
            send_update: Mutex::new(None),
        });
        let mut options = default_binding_options(&["test.models"], transport.clone());
        options.on_error = Some(on_error);
        let namespace = RemoteServiceBinding::new(options).expect("binding");
        let models = namespace.use_service("test.models").expect("use");
        namespace.ready(&BACKGROUND_CONTEXT).await.expect("ready");
        let state = models.state("state").expect("state").expect("state member");
        assert_eq!(state.value().expect("state hydrated")["revision"], json!(0));

        let listener = lock(&transport.send_update).clone().expect("listener registered");
        listener(
            &ServiceProviderUpdate::State {
                instance: None,
                member: "state".into(),
                sequence,
                ops: vec![Op::Replace(json!({ "selected": null, "revision": sequence }))],
            },
            &BACKGROUND_CONTEXT,
        );
        assert!(state.value().is_none());
        assert_eq!(lock(&errors).len(), 1);
        assert!(lock(&errors)[0].contains("sequence has a gap"), "errors: {:?}", lock(&errors));
        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
    }

    #[tokio::test]
    async fn hydrates_cold_replicas_and_replaces_them_across_rebinds() {
        let (provider, state) = singleton_provider().await;
        let mut options = default_binding_options(
            &["test.models"],
            create_loopback_service_transport(StdArc::clone(&provider)),
        );
        options.bound = false;
        let namespace = RemoteServiceBinding::new(options).expect("binding");
        let models = namespace.use_service("test.models").expect("use");
        let revisions: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&revisions);
        let state_view = models.state("state").expect("state").expect("state member");
        let unsubscribe = state_view.subscribe(Arc::new(move |value: &JsonValue, _context: &Context, _delivery: &crate::types::ReplicatedStateDelivery| {
            let sink = Arc::clone(&sink);
            let revision = value["revision"].as_i64();
            Box::pin(async move {
                if let Some(revision) = revision {
                    sink.lock().unwrap().push(revision);
                }
            }) as BoxFuture<()>
        }));
        assert!(state_view.value().is_none());
        assert!(lock(&revisions).is_empty());

        state.replace_with(json!({ "selected": null, "revision": 1 }), &BACKGROUND_CONTEXT);
        namespace.rebind(true, &BACKGROUND_CONTEXT).await.expect("rebind");
        assert_eq!(state_view.value().expect("state hydrated")["revision"], json!(1));
        assert_eq!(*lock(&revisions), vec![1]);
        state.replace_with(json!({ "selected": null, "revision": 2 }), &BACKGROUND_CONTEXT);
        assert_eq!(*lock(&revisions), vec![1, 2]);

        namespace.rebind(false, &BACKGROUND_CONTEXT).await.expect("unbind");
        assert!(state_view.value().is_none());
        state.replace_with(json!({ "selected": null, "revision": 3 }), &BACKGROUND_CONTEXT);
        assert_eq!(*lock(&revisions), vec![1, 2]);
        namespace.rebind(true, &BACKGROUND_CONTEXT).await.expect("rebind");
        assert_eq!(state_view.value().expect("state hydrated")["revision"], json!(3));
        assert_eq!(*lock(&revisions), vec![1, 2, 3]);

        unsubscribe();
        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
        provider.dispose().expect("dispose");
    }

    #[tokio::test]
    async fn hydrates_keyed_state_before_observe_handlers_and_fences_reused_keys() {
        let provider = StdArc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new(
                "test.question-dialog",
                ServiceMode::Keyed,
            )])
            .expect("provider"),
        );
        let transport = create_loopback_service_transport(StdArc::clone(&provider));
        let (on_error, errors) = error_sink();
        let mut options = default_binding_options(&["test.question-dialog"], transport);
        options.on_error = Some(on_error);
        let namespace = RemoteServiceBinding::new(options).expect("binding");

        struct Observed {
            question: Option<JsonValue>,
            handle: Arc<dyn ServiceClientHandle>,
            context: Context,
        }
        let observed: Arc<Mutex<Vec<Observed>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&observed);
        let stop = namespace
            .observe(
                "test.question-dialog",
                Arc::new(move |handle: Arc<dyn ServiceClientHandle>, context: Context| {
                    let sink = Arc::clone(&sink);
                    Box::pin(async move {
                        let question = handle
                            .state("request")
                            .expect("request state")
                            .expect("request member")
                            .value()
                            .cloned();
                        sink.lock().unwrap().push(Observed {
                            question,
                            handle,
                            context,
                        });
                    }) as BoxFuture<()>
                }),
            )
            .expect("observe");
        flush_tasks().await;
        assert!(lock(&errors).is_empty(), "errors: {:?}", lock(&errors));

        let first_request = FakeState::new(json!({ "question": "First?" }));
        let first_calls: Arc<Mutex<Vec<(JsonValue, Context)>> > = Arc::new(Mutex::new(Vec::new()));
        let calls_sink = Arc::clone(&first_calls);
        let first_submit: crate::services::provider::tests::TestMethod =
            Box::new(move |mut args: Vec<JsonValue>, context: Context| {
                let answer = if args.is_empty() { JsonValue::Null } else { args.remove(0) };
                calls_sink.lock().unwrap().push((answer, context.clone()));
                Box::pin(async { Ok(Some(json!({ "accepted": true }))) })
            });
        let close_first = provider
            .spawn(
                "test.question-dialog",
                "invocation-1",
                crate::services::provider::tests::object_with_named_state(
                    "request",
                    &first_request,
                    "submit",
                    first_submit,
                ),
            )
            .expect("spawn");
        wait_for("first observation", || lock(&observed).len() == 1).await;
        assert_eq!(lock(&observed)[0].question, Some(json!({ "question": "First?" })));

        let first_handle: Arc<dyn ServiceClientHandle> = StdArc::clone(&lock(&observed)[0].handle);
        first_request.replace_with(json!({ "question": "Updated?" }), &BACKGROUND_CONTEXT);
        let request_state = first_handle.state("request").expect("request").expect("member");
        assert_eq!(request_state.value(), Some(&json!({ "question": "Updated?" })));
        let response = first_handle
            .invoke("submit", vec![json!("yes")], &BACKGROUND_CONTEXT)
            .await
            .expect("submit");
        assert_eq!(response, Some(json!({ "accepted": true })));
        {
            let calls = lock(&first_calls);
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].0, json!("yes"));
            assert!(calls[0].1.abort_signal().is_none(), "caller context is forwarded");
        }

        close_first();
        wait_for(
            "first observation context aborted",
            || {
                lock(&observed)[0]
                    .context
                    .abort_signal()
                    .is_some_and(|signal| signal.aborted())
            },
        )
        .await;
        let error = err_of(first_handle.state("request"));
        assert!(error.to_string().contains("observation is closed"), "error: {error}");
        let error = err_of(first_handle
            .invoke("submit", vec![json!("late")], &BACKGROUND_CONTEXT)
            .await
            );
        assert!(error.to_string().contains("observation is closed"), "error: {error}");

        let second_request = FakeState::new(json!({ "question": "Again?" }));
        let close_second = provider
            .spawn(
                "test.question-dialog",
                "invocation-1",
                crate::services::provider::tests::object_with_named_state(
                    "request",
                    &second_request,
                    "submit",
                    method_noop(),
                ),
            )
            .expect("respawn");
        wait_for("second observation", || lock(&observed).len() == 2).await;
        assert_eq!(lock(&observed)[1].question, Some(json!({ "question": "Again?" })));
        assert!(!Arc::ptr_eq(
            &first_handle,
            &(StdArc::clone(&lock(&observed)[1].handle) as Arc<dyn ServiceClientHandle>)
        ));
        assert!(lock(&errors).is_empty(), "errors: {:?}", lock(&errors));

        let second_handle: Arc<dyn ServiceClientHandle> = StdArc::clone(&lock(&observed)[1].handle);
        stop();
        wait_for(
            "second observation context aborted",
            || {
                lock(&observed)[1]
                    .context
                    .abort_signal()
                    .is_some_and(|signal| signal.aborted())
            },
        )
        .await;
        let error = err_of(second_handle.state("request"));
        assert!(error.to_string().contains("observation is closed"), "error: {error}");
        let error = err_of(second_handle
            .invoke("submit", vec![json!("late")], &BACKGROUND_CONTEXT)
            .await
            );
        assert!(error.to_string().contains("observation is closed"), "error: {error}");
        close_second();

        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
        provider.dispose().expect("dispose");
    }

    #[tokio::test]
    async fn rebaselines_every_member_then_resumes_contiguous_deltas() {
        // Port of "rebaselines every member through wire codecs": upstream exercises
        // the state-codec round trip; here the serde wire shapes are round-tripped
        // through the landed wire parsers (state_codec is that module's own surface).
        struct WireTransport {
            provider: StdArc<RemoteServiceProvider>,
            before_activate: Arc<dyn Fn() + Send + Sync>,
            updates: Arc<Mutex<Vec<ServiceProviderUpdate>>>,
        }
        #[async_trait]
        impl RemoteServiceTransport for WireTransport {
            async fn invoke(&self, call: &crate::types::ServiceCall, context: &Context) -> Result<Option<JsonValue>, AnyError> {
                self.provider.invoke(call, context).await
            }
            async fn subscribe(
                &self,
                service_id: &str,
                mode: ServiceMode,
                listener: UpdateListener,
                _context: &Context,
            ) -> Result<Box<dyn ServiceSubscription>, AnyError> {
                let updates = Arc::clone(&self.updates);
                let subscription = self.provider.subscribe(
                    service_id,
                    mode,
                    Arc::new(move |update: &ServiceProviderUpdate, context: &Context| {
                        updates.lock().unwrap().push(update.clone());
                        listener(update, context);
                    }),
                )?;
                let encoded = serde_json::to_value(subscription.snapshot()).expect("snapshot serializes");
                let snapshot = crate::services::wire::parse_service_subscription_snapshot(&encoded)
                    .expect("snapshot parses");
                Ok(Box::new(WireSubscription {
                    inner: subscription,
                    snapshot,
                    before_activate: Arc::clone(&self.before_activate),
                }))
            }
        }
        struct WireSubscription {
            inner: Box<dyn ServiceSubscription>,
            snapshot: ServiceSubscriptionSnapshot,
            before_activate: Arc<dyn Fn() + Send + Sync>,
        }
        #[async_trait]
        impl ServiceSubscription for WireSubscription {
            fn snapshot(&self) -> &ServiceSubscriptionSnapshot {
                &self.snapshot
            }
            fn activate(&self) {
                (self.before_activate)();
                self.inner.activate();
            }
            async fn close(&self, _context: Option<Context>) {
                self.inner.close(None).await;
            }
        }

        let provider = StdArc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new("test.pair", ServiceMode::Singleton)])
                .expect("provider"),
        );
        let left = FakeState::new(json!({ "value": 0 }));
        let right = FakeState::new(json!({ "value": 0 }));
        let members: Arc<dyn crate::types::RemoteServiceImpl> =
            crate::services::provider::RemoteServiceObject::builder()
            .state("left", StdArc::clone(&left) as Arc<dyn crate::services::state_internals::StateInternals>)
            .state("right", StdArc::clone(&right) as Arc<dyn crate::services::state_internals::StateInternals>)
            .build();
        provider.provide("test.pair", members).expect("provide");

        let (on_error, errors) = error_sink();
        let updates: Arc<Mutex<Vec<ServiceProviderUpdate>>> = Arc::new(Mutex::new(Vec::new()));
        let before_activate: Arc<dyn Fn() + Send + Sync> = {
            let left = Arc::clone(&left);
            let right = Arc::clone(&right);
            Arc::new(move || {
                for value in 1..=101u64 {
                    left.replace_with(json!({ "value": value }), &BACKGROUND_CONTEXT);
                    right.replace_with(json!({ "value": value }), &BACKGROUND_CONTEXT);
                }
            })
        };
        let mut options = default_binding_options(
            &["test.pair"],
            StdArc::new(WireTransport {
                provider: StdArc::clone(&provider),
                before_activate: Arc::clone(&before_activate),
                updates: Arc::clone(&updates),
            }),
        );
        options.on_error = Some(on_error);
        let namespace = RemoteServiceBinding::new(options).expect("binding");
        let pair = namespace.use_service("test.pair").expect("use");
        namespace.ready(&BACKGROUND_CONTEXT).await.expect("ready");

        let kinds: Vec<String> = lock(&updates).iter().map(provider_update_kind).collect();
        assert_eq!(kinds, vec!["reset", "state"]);
        let left_state = pair.state("left").expect("left").expect("member");
        let right_state = pair.state("right").expect("right").expect("member");
        assert_eq!(left_state.value(), Some(&json!({ "value": 101 })));
        assert_eq!(right_state.value(), Some(&json!({ "value": 101 })));
        for value in 102..=104u64 {
            left.replace_with(json!({ "value": value }), &BACKGROUND_CONTEXT);
            right.replace_with(json!({ "value": value }), &BACKGROUND_CONTEXT);
        }
        assert_eq!(left_state.value(), Some(&json!({ "value": 104 })));
        assert_eq!(right_state.value(), Some(&json!({ "value": 104 })));
        assert!(lock(&errors).is_empty(), "errors: {:?}", lock(&errors));

        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
        provider.dispose().expect("dispose");
    }

    #[tokio::test]
    async fn an_overflow_reset_can_make_a_singleton_unavailable_before_a_later_replacement() {
        struct BeforeActivateTransport {
            provider: StdArc<RemoteServiceProvider>,
            before_activate: Arc<dyn Fn() + Send + Sync>,
            updates: Arc<Mutex<Vec<ServiceProviderUpdate>>>,
        }
        #[async_trait]
        impl RemoteServiceTransport for BeforeActivateTransport {
            async fn invoke(&self, call: &crate::types::ServiceCall, context: &Context) -> Result<Option<JsonValue>, AnyError> {
                self.provider.invoke(call, context).await
            }
            async fn subscribe(
                &self,
                service_id: &str,
                mode: ServiceMode,
                listener: UpdateListener,
                _context: &Context,
            ) -> Result<Box<dyn ServiceSubscription>, AnyError> {
                let updates = Arc::clone(&self.updates);
                let subscription = self.provider.subscribe(
                    service_id,
                    mode,
                    Arc::new(move |update: &ServiceProviderUpdate, context: &Context| {
                        updates.lock().unwrap().push(update.clone());
                        listener(update, context);
                    }),
                )?;
                Ok(Box::new(BeforeActivateSubscription {
                    inner: subscription,
                    before_activate: Arc::clone(&self.before_activate),
                }))
            }
        }
        struct BeforeActivateSubscription {
            inner: Box<dyn ServiceSubscription>,
            before_activate: Arc<dyn Fn() + Send + Sync>,
        }
        #[async_trait]
        impl ServiceSubscription for BeforeActivateSubscription {
            fn snapshot(&self) -> &ServiceSubscriptionSnapshot {
                self.inner.snapshot()
            }
            fn activate(&self) {
                (self.before_activate)();
                self.inner.activate();
            }
            async fn close(&self, _context: Option<Context>) {
                self.inner.close(None).await;
            }
        }

        let provider = StdArc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new("test.counter", ServiceMode::Singleton)])
                .expect("provider"),
        );
        let initial_state = FakeState::new(json!({ "value": 0 }));
        provider
            .provide("test.counter", object_with_state(&initial_state, "bump", method_noop()))
            .expect("provide");
        let (on_error, errors) = error_sink();
        let updates: Arc<Mutex<Vec<ServiceProviderUpdate>>> = Arc::new(Mutex::new(Vec::new()));
        let before_activate: Arc<dyn Fn() + Send + Sync> = {
            let provider = StdArc::clone(&provider);
            Arc::new(move || {
                for value in 1..=100u64 {
                    let state = FakeState::new(json!({ "value": value }));
                    provider
                        .replace("test.counter", object_with_state(&state, "bump", method_noop()))
                        .expect("replace");
                }
                provider.withdraw("test.counter").expect("withdraw");
            })
        };
        let mut options = default_binding_options(
            &["test.counter"],
            StdArc::new(BeforeActivateTransport {
                provider: StdArc::clone(&provider),
                before_activate: Arc::clone(&before_activate),
                updates: Arc::clone(&updates),
            }),
        );
        options.on_error = Some(on_error);
        let namespace = RemoteServiceBinding::new(options).expect("binding");
        let counter = namespace.use_service("test.counter").expect("use");
        namespace.ready(&BACKGROUND_CONTEXT).await.expect("ready");
        assert_eq!(
            lock(&updates)
                .iter()
                .map(|update| serde_json::to_value(update).expect("update"))
                .collect::<Vec<_>>(),
            vec![json!({
                "type": "reset",
                "snapshot": { "serviceId": "test.counter", "mode": "singleton", "instances": [] },
            })]
        );
        let counter_state = counter.state("state").expect("state").expect("member");
        assert!(counter_state.value().is_none());

        let replacement = FakeState::new(json!({ "value": 200 }));
        provider
            .replace("test.counter", object_with_state(&replacement, "bump", method_noop()))
            .expect("replace");
        replacement.replace_with(json!({ "value": 201 }), &BACKGROUND_CONTEXT);
        let reused = namespace.use_service("test.counter").expect("reuse");
        assert!(Arc::ptr_eq(
            &(counter as Arc<dyn ServiceClientHandle>),
            &(reused as Arc<dyn ServiceClientHandle>)
        ));
        assert_eq!(counter_state.value(), Some(&json!({ "value": 201 })));
        assert!(lock(&errors).is_empty(), "errors: {:?}", lock(&errors));

        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
        provider.dispose().expect("dispose");
    }

    #[tokio::test]
    // The storm's pending closes are best-effort: when the storm never fires the
    // still-boxed unsubscribes drop without being called (upstream relies on GC).
    #[allow(unused_must_use)]
    async fn keyed_resets_retain_live_generations_and_reconcile_closed_and_reused_keys() {
        struct RecordingTransport {
            inner: Arc<dyn RemoteServiceTransport>,
            updates: Arc<Mutex<Vec<ServiceProviderUpdate>>>,
        }
        #[async_trait]
        impl RemoteServiceTransport for RecordingTransport {
            async fn invoke(&self, call: &crate::types::ServiceCall, context: &Context) -> Result<Option<JsonValue>, AnyError> {
                self.inner.invoke(call, context).await
            }
            async fn subscribe(
                &self,
                service_id: &str,
                mode: ServiceMode,
                listener: UpdateListener,
                context: &Context,
            ) -> Result<Box<dyn ServiceSubscription>, AnyError> {
                let updates = Arc::clone(&self.updates);
                let inner_listener: UpdateListener =
                    Arc::new(move |update: &ServiceProviderUpdate, context: &Context| {
                        updates.lock().unwrap().push(update.clone());
                        listener(update, context);
                    });
                self.inner.subscribe(service_id, mode, inner_listener, context).await
            }
        }

        let provider = StdArc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new(
                "test.counter",
                ServiceMode::Keyed,
            )])
            .expect("provider"),
        );
        let retained = FakeState::new(json!({ "value": 0 }));
        let close_retained = provider
            .spawn("test.counter", "retained", object_with_state(&retained, "bump", method_noop()))
            .expect("spawn retained");
        let close_old = provider
            .spawn(
                "test.counter",
                "reused",
                object_with_state(&FakeState::new(json!({ "value": 0 })), "bump", method_noop()),
            )
            .expect("spawn reused");
        let close_removed = provider
            .spawn(
                "test.counter",
                "removed",
                object_with_state(&FakeState::new(json!({ "value": 0 })), "bump", method_noop()),
            )
            .expect("spawn removed");

        let (on_error, errors) = error_sink();
        let updates: Arc<Mutex<Vec<ServiceProviderUpdate>>> = Arc::new(Mutex::new(Vec::new()));
        let mut options = default_binding_options(
            &["test.counter"],
            StdArc::new(RecordingTransport {
                inner: create_loopback_service_transport(StdArc::clone(&provider)),
                updates: Arc::clone(&updates),
            }),
        );
        options.on_error = Some(on_error);
        let namespace = RemoteServiceBinding::new(options).expect("binding");

        struct Observed {
            handle: Arc<dyn ServiceClientHandle>,
            context: Context,
        }
        let observed: Arc<Mutex<Vec<Observed>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&observed);
        let _observation = namespace
            .observe(
                "test.counter",
                Arc::new(move |handle: Arc<dyn ServiceClientHandle>, context: Context| {
                    let sink = Arc::clone(&sink);
                    Box::pin(async move {
                        sink.lock().unwrap().push(Observed { handle, context });
                    }) as BoxFuture<()>
                }),
            )
            .expect("observe");
        namespace.ready(&BACKGROUND_CONTEXT).await.expect("ready");
        assert_eq!(lock(&observed).len(), 3);
        let replacement = FakeState::new(json!({ "value": 500 }));
        let pending_closes: Arc<Mutex<Vec<Unsubscribe>>> = Arc::new(Mutex::new(vec![close_old, close_removed]));
        let storm: StdArc<dyn Fn() + Send + Sync> = {
            let provider = StdArc::clone(&provider);
            let replacement = Arc::clone(&replacement);
            let retained = Arc::clone(&retained);
            let pending_closes = Arc::clone(&pending_closes);
            Arc::new(move || {
                for close in lock(&pending_closes).drain(..) {
                    close();
                }
                provider
                    .spawn(
                        "test.counter",
                        "reused",
                        object_with_state(&replacement, "bump", method_noop()),
                    )
                    .expect("respawn reused");
                retained.replace_with(json!({ "value": 102 }), &BACKGROUND_CONTEXT);
                for next in 501..=601u64 {
                    replacement.replace_with(json!({ "value": next }), &BACKGROUND_CONTEXT);
                }
            })
        };
        // Handler completion order across the three initial tasks is scheduler
        // defined (upstream relies on FIFO microtasks), so the storm is wired to
        // whichever observed facade carries the retained instance's state and the
        // lifecycle assertions below identify entries by state, not position.
        for entry in lock(&observed).iter() {
            let storm = Arc::clone(&storm);
            if let Ok(Some(state)) = entry.handle.state("state") {
                let _unsubscribe = state.subscribe(StdArc::new(move |value: &JsonValue, _context: &Context, _delivery: &crate::types::ReplicatedStateDelivery| {
                    let storm = Arc::clone(&storm);
                    let fire = value["value"] == json!(1);
                    Box::pin(async move {
                        if fire {
                            storm();
                        }
                    }) as BoxFuture<()>
                }));
            }
        }
        retained.replace_with(json!({ "value": 1 }), &BACKGROUND_CONTEXT);
        wait_for("storm observation", || lock(&observed).len() == 4).await;
        assert!(lock(&updates).iter().any(|update| matches!(update, ServiceProviderUpdate::Reset { .. })));
        let aborted = |entry: &Observed| {
            entry
                .context
                .abort_signal()
                .is_some_and(|signal| signal.aborted())
        };
        let stable_index = (0..3).find(|index| !aborted(&lock(&observed)[*index])).expect("one live initial entry");
        assert_eq!(2, (0..3).filter(|index| aborted(&lock(&observed)[*index])).count());
        let stable_handle: Arc<dyn ServiceClientHandle> = StdArc::clone(&lock(&observed)[stable_index].handle);
        let retained_state = stable_handle.state("state").expect("state").expect("member");
        assert_eq!(retained_state.value(), Some(&json!({ "value": 102 })));
        let replacement_handle: Arc<dyn ServiceClientHandle> = StdArc::clone(&lock(&observed)[3].handle);
        assert!(!aborted(&lock(&observed)[3]));
        let replacement_state = replacement_handle.state("state").expect("state").expect("member");
        assert_eq!(replacement_state.value(), Some(&json!({ "value": 601 })));
        replacement.replace_with(json!({ "value": 602 }), &BACKGROUND_CONTEXT);
        assert_eq!(replacement_state.value(), Some(&json!({ "value": 602 })));
        assert!(lock(&errors).is_empty(), "errors: {:?}", lock(&errors));

        let _ = close_retained;
        namespace.dispose(&BACKGROUND_CONTEXT).await.expect("dispose");
        provider.dispose().expect("dispose");
    }
}


