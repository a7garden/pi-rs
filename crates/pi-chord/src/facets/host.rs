//! Port of `packages/chord/src/facets/host.ts`.
//!
//! Private lifecycle and dependency kernel behind the atomic host entry point.
//!
//! Deviations from upstream (recorded per PORT_MAP.md rule 3):
//! - `ServiceSlot`/`ServiceView` JS proxies collapse into typed views created at
//!   access time (`services::handle::LocalServiceView`); remote member access
//!   goes through [`ServiceClientHandle`].
//! - "implementation must be an object" checks are vacuous: Rust service
//!   implementations are `Arc` values, always objects.
//! - `Facet.setup` is synchronous by construction (`Result`), so the upstream
//!   "setup must be synchronous" promise check cannot fire.

use std::any::Any;
use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::future::join_all;

use crate::context::{Context, BACKGROUND_CONTEXT};
use crate::json::JsonValue;
use crate::services::consumer::RemoteServiceBinding;
use crate::services::errors::{error, type_error, AggregateError, AnyError};
use crate::services::handle::ServiceSlot;
use crate::services::instances::{InstanceDirectory, InstanceDirectoryEntry};
use crate::services::loopback::create_loopback_service_transport;
use crate::services::provider::{
    validate_remote_service_implementation, ProviderDefinition, RemoteServiceProvider,
};
use crate::services::state::MutableReplicatedState;
use crate::types::{
    arc_error::ErrorHandler, BoxFuture, Facet, FacetOptions, KeyedObservationHandler,
    RemoteServiceBindingOptions, RemoteServiceImpl, RemoteServiceSource,
    RemoteServiceSourceOpenOptions, RemoteServices, Service, ServiceClientHandle, ServiceMode,
    Unsubscribe,
};

/// Owned cleanup callback (upstream `() => void | Promise<void>`); failures are
/// collected during disposal instead of thrown.
pub type Disposal = Arc<dyn Fn() -> BoxFuture<Result<(), AnyError>> + Send + Sync>;

/// Build an owned disposal from an [`Unsubscribe`] handle.
pub fn disposal_from_unsubscribe(unsubscribe: Unsubscribe) -> Disposal {
    let cell = Arc::new(Mutex::new(Some(unsubscribe)));
    Arc::new(move || {
        let pending = cell.lock().expect("disposal cell lock").take();
        Box::pin(async move {
            if let Some(unsubscribe) = pending {
                unsubscribe();
            }
            Ok(())
        }) as BoxFuture<Result<(), AnyError>>
    })
}

type AssertAccess = Arc<dyn Fn() -> Result<(), AnyError> + Send + Sync>;
type ObserveStart = Arc<dyn Fn() -> Result<Disposal, AnyError> + Send + Sync>;
type ClosedFlag = Arc<AtomicBool>;

static NEXT_REQUIREMENT_ID: AtomicU64 = AtomicU64::new(1);

fn next_requirement_id() -> u64 {
    NEXT_REQUIREMENT_ID.fetch_add(1, Ordering::SeqCst)
}

fn access_check(lifecycle: &Arc<FacetLifecycle>) -> AssertAccess {
    let lifecycle = lifecycle.clone();
    Arc::new(move || lifecycle.assert_service_access())
}

fn lock<T>(guard: std::sync::LockResult<T>, what: &str) -> T {
    match guard {
        Ok(value) => value,
        Err(poisoned) => {
            tracing::error!("poisoned {what}: {poisoned}");
            poisoned.into_inner()
        }
    }
}

// ─── Lifecycle ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LifecycleState {
    SettingUp,
    Prepared,
    Active,
    Disposing,
    Dead,
}

impl LifecycleState {
    fn as_str(self) -> &'static str {
        match self {
            LifecycleState::SettingUp => "setting_up",
            LifecycleState::Prepared => "prepared",
            LifecycleState::Active => "active",
            LifecycleState::Disposing => "disposing",
            LifecycleState::Dead => "dead",
        }
    }
}

struct FacetLifecycleInner {
    state: LifecycleState,
    service_access: bool,
    effects: Vec<Disposal>,
    observations: Vec<ObserveStart>,
    activation_callbacks: Vec<Disposal>,
}

/// Per-facet lifecycle: state machine, owned effects, observations and
/// activation callbacks (upstream `FacetLifecycle`).
struct FacetLifecycle {
    id: String,
    inner: Mutex<FacetLifecycleInner>,
}

impl FacetLifecycle {
    fn new(id: &str) -> FacetLifecycle {
        FacetLifecycle {
            id: id.to_string(),
            inner: Mutex::new(FacetLifecycleInner {
                state: LifecycleState::SettingUp,
                service_access: false,
                effects: Vec::new(),
                observations: Vec::new(),
                activation_callbacks: Vec::new(),
            }),
        }
    }

    fn assert_setting_up(&self, operation: &str) -> Result<(), AnyError> {
        let inner = lock(self.inner.lock(), "facet lifecycle");
        if inner.state != LifecycleState::SettingUp {
            return Err(error(format!(
                "Facet {} can {operation} only during setup",
                self.id
            )));
        }
        Ok(())
    }

    fn assert_running(&self, operation: &str) -> Result<(), AnyError> {
        let inner = lock(self.inner.lock(), "facet lifecycle");
        if inner.state != LifecycleState::SettingUp && inner.state != LifecycleState::Active {
            return Err(error(format!(
                "Facet {} cannot {operation} while {}",
                self.id,
                inner.state.as_str()
            )));
        }
        Ok(())
    }

    fn assert_active(&self, operation: &str) -> Result<(), AnyError> {
        let inner = lock(self.inner.lock(), "facet lifecycle");
        if inner.state != LifecycleState::Active {
            return Err(error(format!(
                "Facet {} can {operation} only while active",
                self.id
            )));
        }
        Ok(())
    }

    fn assert_service_access(&self) -> Result<(), AnyError> {
        let inner = lock(self.inner.lock(), "facet lifecycle");
        if !inner.service_access {
            return Err(error(format!(
                "Facet {} service handles cannot be used while {}",
                self.id,
                inner.state.as_str()
            )));
        }
        Ok(())
    }

    fn revoke(&self) {
        let mut inner = lock(self.inner.lock(), "facet lifecycle");
        inner.service_access = false;
    }

    fn own(&self, disposal: Disposal) -> Result<(), AnyError> {
        self.assert_running("own resources")?;
        lock(self.inner.lock(), "facet lifecycle")
            .effects
            .push(disposal);
        Ok(())
    }

    fn observe(&self, start: ObserveStart) -> Result<(), AnyError> {
        self.assert_setting_up("observe services")?;
        lock(self.inner.lock(), "facet lifecycle")
            .observations
            .push(start);
        Ok(())
    }

    fn on_activate(&self, callback: Disposal) -> Result<(), AnyError> {
        self.assert_setting_up("register activation callbacks")?;
        lock(self.inner.lock(), "facet lifecycle")
            .activation_callbacks
            .push(callback);
        Ok(())
    }

    fn prepared(&self) -> Result<(), AnyError> {
        self.assert_setting_up("finish setup")?;
        lock(self.inner.lock(), "facet lifecycle").state = LifecycleState::Prepared;
        Ok(())
    }

    async fn activate(&self) -> Result<(), AnyError> {
        {
            let mut inner = lock(self.inner.lock(), "facet lifecycle");
            if inner.state != LifecycleState::Prepared {
                return Err(error(format!("Facet {} is not prepared", self.id)));
            }
            inner.state = LifecycleState::Active;
            inner.service_access = true;
        }
        let starts = lock(self.inner.lock(), "facet lifecycle")
            .observations
            .clone();
        for start in starts {
            let stop = start()?;
            lock(self.inner.lock(), "facet lifecycle")
                .effects
                .push(stop);
        }
        let callbacks = lock(self.inner.lock(), "facet lifecycle")
            .activation_callbacks
            .clone();
        for callback in callbacks {
            callback().await?;
        }
        Ok(())
    }

    async fn dispose(&self) -> Result<(), AnyError> {
        {
            let mut inner = lock(self.inner.lock(), "facet lifecycle");
            if inner.state == LifecycleState::Dead {
                return Ok(());
            }
            inner.state = LifecycleState::Disposing;
        }
        let effects = std::mem::take(&mut lock(self.inner.lock(), "facet lifecycle").effects);
        let mut errors: Vec<AnyError> = Vec::new();
        for effect in effects.into_iter().rev() {
            if let Err(disposal_error) = effect().await {
                errors.push(disposal_error);
            }
        }
        {
            let mut inner = lock(self.inner.lock(), "facet lifecycle");
            inner.observations.clear();
            inner.activation_callbacks.clear();
            inner.service_access = false;
            inner.state = LifecycleState::Dead;
        }
        if errors.len() == 1 {
            return Err(errors.pop().expect("one error"));
        }
        if errors.len() > 1 {
            return Err(AggregateError::new(
                format!("Failed to dispose facet {}", self.id),
                errors,
            ));
        }
        Ok(())
    }
}

// ─── Local keyed service registry ────────────────────────────────────────────

/// Target handed to keyed observation callbacks: the raw implementation for
/// process-local keyed services, a client handle for remote ones.
enum KeyedTarget {
    Local(Arc<dyn Any + Send + Sync>),
    Client(Arc<dyn ServiceClientHandle>),
}

/// Source of live keyed instances a facet can observe.
trait KeyedServiceSource: Send + Sync {
    fn observe(
        &self,
        service_id: &str,
        handler: Arc<dyn Fn(KeyedTarget, Context) -> BoxFuture<()> + Send + Sync>,
    ) -> Result<Unsubscribe, AnyError>;
}

#[derive(Clone)]
struct RegistryEntry {
    key: String,
    generation: u64,
    service: Arc<dyn Any + Send + Sync>,
}

impl InstanceDirectoryEntry for RegistryEntry {
    fn key(&self) -> &str {
        &self.key
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn deactivate(&self) {}
}

struct LocalKeyedRegistration {
    generations: HashMap<String, u64>,
    directory: Arc<InstanceDirectory<RegistryEntry>>,
}

/// Host-owned registry for process-local keyed services (upstream
/// `LocalKeyedServiceRegistry`).
struct LocalKeyedServiceRegistry {
    registrations: Mutex<HashMap<String, LocalKeyedRegistration>>,
    disposed: AtomicBool,
}

impl LocalKeyedServiceRegistry {
    fn new(
        services: Vec<String>,
        on_error: ErrorHandler,
    ) -> Result<LocalKeyedServiceRegistry, AnyError> {
        let mut unique = HashSet::new();
        for service_id in &services {
            if !unique.insert(service_id.clone()) {
                return Err(type_error("Local keyed service registry has duplicate IDs"));
            }
        }
        let mut registrations = HashMap::new();
        for service_id in services {
            registrations.insert(
                service_id,
                LocalKeyedRegistration {
                    generations: HashMap::new(),
                    directory: Arc::new(InstanceDirectory::new(true, on_error.clone())),
                },
            );
        }
        Ok(LocalKeyedServiceRegistry {
            registrations: Mutex::new(registrations),
            disposed: AtomicBool::new(false),
        })
    }

    fn spawn(
        &self,
        service_id: &str,
        key: &str,
        implementation: Arc<dyn Any + Send + Sync>,
    ) -> Result<Unsubscribe, AnyError> {
        self.assert_active()?;
        if key.is_empty() {
            return Err(type_error("Local service instance key must not be empty"));
        }
        let (directory, generation) = {
            let mut registrations = lock(self.registrations.lock(), "keyed service registry");
            let registration = registrations.get_mut(service_id).ok_or_else(|| {
                error(format!(
                    "Local keyed service {service_id} is not registered"
                ))
            })?;
            if registration.directory.get(key).is_some() {
                return Err(error(format!(
                    "Local service {service_id} already has a live instance with key {key}"
                )));
            }
            let generation = registration.generations.get(key).copied().unwrap_or(0) + 1;
            registration.generations.insert(key.to_string(), generation);
            (registration.directory.clone(), generation)
        };
        let entry = RegistryEntry {
            key: key.to_string(),
            generation,
            service: implementation,
        };
        directory.insert(entry.clone())?;
        let closed = Arc::new(AtomicBool::new(false));
        let closed_for_close = closed.clone();
        Ok(Box::new(move || {
            if closed_for_close.swap(true, Ordering::SeqCst) {
                return;
            }
            directory.remove(&entry);
        }))
    }

    fn registration(
        &self,
        service_id: &str,
    ) -> Result<Arc<InstanceDirectory<RegistryEntry>>, AnyError> {
        lock(self.registrations.lock(), "keyed service registry")
            .get(service_id)
            .map(|registration| registration.directory.clone())
            .ok_or_else(|| {
                error(format!(
                    "Local keyed service {service_id} is not registered"
                ))
            })
    }

    fn assert_active(&self) -> Result<(), AnyError> {
        if self.disposed.load(Ordering::SeqCst) {
            return Err(error("Local keyed service registry is disposed"));
        }
        Ok(())
    }

    fn dispose(&self) {
        if self.disposed.swap(true, Ordering::SeqCst) {
            return;
        }
        let registrations = lock(self.registrations.lock(), "keyed service registry");
        for registration in registrations.values() {
            registration.directory.dispose();
        }
    }
}

impl KeyedServiceSource for LocalKeyedServiceRegistry {
    fn observe(
        &self,
        service_id: &str,
        handler: Arc<dyn Fn(KeyedTarget, Context) -> BoxFuture<()> + Send + Sync>,
    ) -> Result<Unsubscribe, AnyError> {
        self.assert_active()?;
        let directory = self.registration(service_id)?;
        let unsubscribe = directory.observe(Arc::new(move |entry, context| {
            handler(KeyedTarget::Local(entry.service.clone()), context)
        }))?;
        Ok(unsubscribe)
    }
}

/// Adapter exposing a [`RemoteServices`] consumer binding as a keyed source.
struct RemoteServicesKeyedSource(Arc<dyn RemoteServices>);

impl KeyedServiceSource for RemoteServicesKeyedSource {
    fn observe(
        &self,
        service_id: &str,
        handler: Arc<dyn Fn(KeyedTarget, Context) -> BoxFuture<()> + Send + Sync>,
    ) -> Result<Unsubscribe, AnyError> {
        let adapted: KeyedObservationHandler =
            Arc::new(move |client, context| handler(KeyedTarget::Client(client), context));
        let result = self.0.observe(service_id, adapted);
        result
    }
}

// ─── Host service slots ──────────────────────────────────────────────────────

/// Host-owned mutable singleton slots plus keyed observation sources (upstream
/// `HostServiceSlots`).
struct HostServiceSlots {
    singletons: Mutex<HashMap<String, Arc<ServiceSlot>>>,
    keyed_sources: Mutex<HashMap<String, Arc<dyn KeyedServiceSource>>>,
}

impl HostServiceSlots {
    fn new() -> HostServiceSlots {
        HostServiceSlots {
            singletons: Mutex::new(HashMap::new()),
            keyed_sources: Mutex::new(HashMap::new()),
        }
    }

    fn get_singleton(&self, service_id: &str, wrap_objects: bool) -> Arc<ServiceSlot> {
        let mut singletons = lock(self.singletons.lock(), "host service slots");
        singletons
            .entry(service_id.to_string())
            .or_insert_with(|| Arc::new(ServiceSlot::new(service_id, wrap_objects)))
            .clone()
    }

    fn has_singleton(&self, service_id: &str) -> bool {
        lock(self.singletons.lock(), "host service slots").contains_key(service_id)
    }

    /// Subscribe to a keyed service. The callback receives the raw target plus
    /// the shared closed flag; view construction (and the "observation is
    /// closed" staleness check) happens at access time in the caller.
    fn observe(
        &self,
        service_id: &str,
        on_instance: Arc<dyn Fn(KeyedTarget, ClosedFlag, Context) -> BoxFuture<()> + Send + Sync>,
    ) -> Result<Unsubscribe, AnyError> {
        let source = lock(self.keyed_sources.lock(), "host service slots")
            .get(service_id)
            .cloned()
            .ok_or_else(|| error(format!("Service {service_id} is disconnected")))?;
        let closed: ClosedFlag = Arc::new(AtomicBool::new(false));
        let closed_for_source = closed.clone();
        let stop = source.observe(
            service_id,
            Arc::new(move |target, context| {
                on_instance(target, closed_for_source.clone(), context)
            }),
        )?;
        Ok(Box::new(move || {
            closed.store(true, Ordering::SeqCst);
            stop();
        }))
    }

    fn bind_singleton(&self, service_id: &str, target: Arc<dyn Any + Send + Sync>) {
        if let Some(slot) = lock(self.singletons.lock(), "host service slots").get(service_id) {
            slot.bind(target);
        }
    }

    fn bind_keyed(&self, service_id: &str, source: Arc<dyn KeyedServiceSource>) {
        lock(self.keyed_sources.lock(), "host service slots")
            .insert(service_id.to_string(), source);
    }

    fn dispose(&self) {
        let mut singletons = lock(self.singletons.lock(), "host service slots");
        for slot in singletons.values() {
            slot.unbind();
        }
        singletons.clear();
        lock(self.keyed_sources.lock(), "host service slots").clear();
    }
}

// ─── Keyed views ─────────────────────────────────────────────────────────────

/// Typed view over one live local keyed instance. Access checks (lifecycle
/// revocation, closed observation, aborted delivery context) run at call time,
/// mirroring the upstream proxy member access.
#[derive(Clone)]
pub struct LocalServiceView<T> {
    service_id: Arc<str>,
    target: Arc<dyn Any + Send + Sync>,
    assert_access: AssertAccess,
    closed: ClosedFlag,
    context: Context,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Any + Send + Sync> LocalServiceView<T> {
    fn new(
        service_id: &str,
        target: Arc<dyn Any + Send + Sync>,
        assert_access: AssertAccess,
        closed: ClosedFlag,
        context: Context,
    ) -> LocalServiceView<T> {
        LocalServiceView {
            service_id: Arc::from(service_id),
            target,
            assert_access,
            closed,
            context,
            _marker: PhantomData,
        }
    }

    /// Run `f` against the live instance, enforcing access and staleness.
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> Result<R, AnyError> {
        (self.assert_access)()?;
        if self.closed.load(Ordering::SeqCst)
            || self
                .context
                .abort_signal()
                .is_some_and(|signal| signal.aborted())
        {
            return Err(error(format!(
                "Keyed service {} observation is closed",
                self.service_id
            )));
        }
        let target = self.target.clone().downcast::<T>().map_err(|_| {
            type_error(format!(
                "Service {} implementation type mismatch",
                self.service_id
            ))
        })?;
        Ok(f(&target))
    }
}

/// Client-handle view over one live remote keyed instance with the same
/// access/staleness guards as [`LocalServiceView`].
struct ObservationHandle {
    inner: Arc<dyn ServiceClientHandle>,
    assert_access: AssertAccess,
    closed: ClosedFlag,
    context: Context,
}

impl ObservationHandle {
    fn check(&self) -> Result<(), AnyError> {
        (self.assert_access)()?;
        if self.closed.load(Ordering::SeqCst)
            || self
                .context
                .abort_signal()
                .is_some_and(|signal| signal.aborted())
        {
            return Err(error(format!(
                "Keyed service {} observation is closed",
                self.inner.service_id()
            )));
        }
        Ok(())
    }
}

#[async_trait]
impl ServiceClientHandle for ObservationHandle {
    fn service_id(&self) -> &str {
        self.inner.service_id()
    }

    fn invoke(
        &self,
        member: &str,
        args: Vec<JsonValue>,
        context: &Context,
    ) -> BoxFuture<Result<Option<JsonValue>, AnyError>> {
        match self.check() {
            Ok(()) => self.inner.invoke(member, args, context),
            Err(invocation_error) => {
                let invocation_error: AnyError = invocation_error;
                Box::pin(async move { Err(invocation_error) })
            }
        }
    }

    fn state(
        &self,
        member: &str,
    ) -> Result<Option<Arc<dyn crate::types::ReplicatedState>>, AnyError> {
        self.check()?;
        self.inner.state(member)
    }
}

/// Stand-in for a client handle observed through a local keyed source; the
/// mode validation makes this unreachable, but the type stays total.
struct MismatchedClientHandle {
    service_id: String,
}

#[async_trait]
impl ServiceClientHandle for MismatchedClientHandle {
    fn service_id(&self) -> &str {
        &self.service_id
    }

    fn invoke(
        &self,
        _member: &str,
        _args: Vec<JsonValue>,
        _context: &Context,
    ) -> BoxFuture<Result<Option<JsonValue>, AnyError>> {
        let service_id = self.service_id.clone();
        Box::pin(async move {
            Err(type_error(format!(
                "Service {service_id} is not a remote service"
            )))
        })
    }

    fn state(
        &self,
        _member: &str,
    ) -> Result<Option<Arc<dyn crate::types::ReplicatedState>>, AnyError> {
        Err(type_error(format!(
            "Service {} is not a remote service",
            self.service_id
        )))
    }
}

/// Wrapper making a [`ServiceClientHandle`] storable in a [`ServiceSlot`].
#[derive(Clone)]
struct ClientTarget(Arc<dyn ServiceClientHandle>);

// ─── Staged service spawner ──────────────────────────────────────────────────

type ServiceInstanceInstaller =
    Arc<dyn Fn(&str, Arc<dyn Any + Send + Sync>) -> Result<Unsubscribe, AnyError> + Send + Sync>;
type SpawnValidator =
    Arc<dyn Fn(&str, &Arc<dyn Any + Send + Sync>) -> Result<(), AnyError> + Send + Sync>;

struct StagedServiceInstance {
    id: u64,
    key: String,
    implementation: Arc<dyn Any + Send + Sync>,
    release: Option<Unsubscribe>,
}

/// Deferred keyed-instance spawner installed once the backing registry or
/// provider connects (upstream `StagedServiceSpawner`).
struct StagedServiceSpawner {
    lifecycle: Arc<FacetLifecycle>,
    validate: SpawnValidator,
    instances: Mutex<Vec<StagedServiceInstance>>,
    installer: Mutex<Option<ServiceInstanceInstaller>>,
    next_id: AtomicU64,
}

impl StagedServiceSpawner {
    fn new(lifecycle: Arc<FacetLifecycle>, validate: SpawnValidator) -> Arc<StagedServiceSpawner> {
        Arc::new(StagedServiceSpawner {
            lifecycle,
            validate,
            instances: Mutex::new(Vec::new()),
            installer: Mutex::new(None),
            next_id: AtomicU64::new(0),
        })
    }

    fn connect(&self, installer: ServiceInstanceInstaller) -> Result<(), AnyError> {
        {
            let mut connected = lock(self.installer.lock(), "staged service spawner");
            if connected.is_some() {
                return Err(error("Facet service provider is already connected"));
            }
            *connected = Some(installer.clone());
        }
        let mut instances = lock(self.instances.lock(), "staged service spawner");
        for instance in instances.iter_mut() {
            if instance.release.is_some() {
                continue;
            }
            instance.release = Some(installer(&instance.key, instance.implementation.clone())?);
        }
        Ok(())
    }

    fn spawn(
        self: &Arc<Self>,
        key: &str,
        implementation: Arc<dyn Any + Send + Sync>,
    ) -> Result<Unsubscribe, AnyError> {
        self.lifecycle.assert_active("spawn service instances")?;
        (self.validate)(key, &implementation)?;
        let mut instances = lock(self.instances.lock(), "staged service spawner");
        if instances.iter().any(|instance| instance.key == key) {
            return Err(error(format!(
                "Facet service already has a live instance with key {key}"
            )));
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let installer = lock(self.installer.lock(), "staged service spawner").clone();
        let release = match &installer {
            Some(installer) => Some(installer(key, implementation.clone())?),
            None => None,
        };
        instances.push(StagedServiceInstance {
            id,
            key: key.to_string(),
            implementation,
            release,
        });
        let close: Arc<dyn Fn() + Send + Sync> = {
            let spawner = self.clone();
            Arc::new(move || {
                let mut instances = lock(spawner.instances.lock(), "staged service spawner");
                if let Some(position) = instances.iter().position(|instance| instance.id == id) {
                    let mut instance = instances.remove(position);
                    if let Some(release) = instance.release.take() {
                        release();
                    }
                }
            })
        };
        let owned: Disposal = {
            let close = close.clone();
            Arc::new(move || {
                let close = close.clone();
                Box::pin(async move {
                    close();
                    Ok(())
                })
            })
        };
        self.lifecycle.own(owned)?;
        Ok(Box::new(move || close()))
    }
}

// ─── Provisions ──────────────────────────────────────────────────────────────

/// Newtype carrying a remote implementation through the untyped spawner.
struct RemoteImplTarget(Arc<dyn RemoteServiceImpl>);

#[derive(Clone)]
struct SingletonProvision {
    service_id: String,
    local: bool,
    mode: ServiceMode,
    implementation: Arc<dyn Any + Send + Sync>,
    remote: Option<Arc<dyn RemoteServiceImpl>>,
}

#[derive(Clone)]
struct KeyedProvision {
    service_id: String,
    local: bool,
    spawner: Arc<StagedServiceSpawner>,
}

#[derive(Clone)]
enum FacetProvision {
    Singleton(SingletonProvision),
    Keyed(KeyedProvision),
}

impl FacetProvision {
    fn service_id(&self) -> &str {
        match self {
            FacetProvision::Singleton(provision) => &provision.service_id,
            FacetProvision::Keyed(provision) => &provision.service_id,
        }
    }

    fn mode(&self) -> ServiceMode {
        match self {
            FacetProvision::Singleton(provision) => provision.mode,
            FacetProvision::Keyed(_) => ServiceMode::Keyed,
        }
    }

    fn local(&self) -> bool {
        match self {
            FacetProvision::Singleton(provision) => provision.local,
            FacetProvision::Keyed(provision) => provision.local,
        }
    }

    fn install(&self, provider: &RemoteServiceProvider) -> Result<(), AnyError> {
        if let FacetProvision::Singleton(provision) = self {
            if !provision.local {
                let remote = provision
                    .remote
                    .clone()
                    .expect("remote singleton provision carries an implementation");
                return provider.provide(&provision.service_id, remote);
            }
        }
        Ok(())
    }

    fn validate_replacement(&self, provider: &RemoteServiceProvider) -> Result<(), AnyError> {
        if let FacetProvision::Singleton(provision) = self {
            if !provision.local {
                let remote = provision
                    .remote
                    .clone()
                    .expect("remote singleton provision carries an implementation");
                return provider.validate_replacement(&provision.service_id, remote);
            }
        }
        Ok(())
    }

    fn replace(&self, provider: &RemoteServiceProvider) -> Result<(), AnyError> {
        if let FacetProvision::Singleton(provision) = self {
            if !provision.local {
                let remote = provision
                    .remote
                    .clone()
                    .expect("remote singleton provision carries an implementation");
                return provider.replace(&provision.service_id, remote);
            }
        }
        Ok(())
    }

    fn connect_local(&self, registry: Arc<LocalKeyedServiceRegistry>) -> Result<(), AnyError> {
        if let FacetProvision::Keyed(provision) = self {
            if provision.local {
                let service_id = provision.service_id.clone();
                return provision.spawner.connect(Arc::new(
                    move |key: &str, implementation: Arc<dyn Any + Send + Sync>| {
                        registry.spawn(&service_id, key, implementation)
                    },
                ));
            }
        }
        Ok(())
    }

    fn connect_remote(&self, provider: Arc<RemoteServiceProvider>) -> Result<(), AnyError> {
        if let FacetProvision::Keyed(provision) = self {
            if !provision.local {
                let service_id = provision.service_id.clone();
                return provision.spawner.connect(Arc::new(
                    move |key: &str, implementation: Arc<dyn Any + Send + Sync>| {
                        let target =
                            implementation.downcast::<RemoteImplTarget>().map_err(|_| {
                                type_error(format!(
                                    "Service {service_id} implementation type mismatch"
                                ))
                            })?;
                        let remote = match &*target {
                            RemoteImplTarget(remote) => remote.clone(),
                        };
                        provider.spawn(&service_id, key, remote)
                    },
                ));
            }
        }
        Ok(())
    }
}

// ─── Facet shape and runtime ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
struct FacetServiceReference {
    service_id: String,
    mode: ServiceMode,
}

struct FacetRuntime {
    facet_id: String,
    requires: Vec<FacetServiceReference>,
    provides: Vec<FacetServiceReference>,
    lifecycle: Arc<FacetLifecycle>,
    provisions: Vec<FacetProvision>,
    singleton_views: HashMap<String, Arc<dyn Any + Send + Sync>>,
}

impl FacetRuntime {
    fn new(facet_id: &str) -> FacetRuntime {
        FacetRuntime {
            facet_id: facet_id.to_string(),
            requires: Vec::new(),
            provides: Vec::new(),
            lifecycle: Arc::new(FacetLifecycle::new(facet_id)),
            provisions: Vec::new(),
            singleton_views: HashMap::new(),
        }
    }

    fn shape(&self) -> FacetShape {
        FacetShape {
            facet_id: self.facet_id.clone(),
            requires: self.requires.clone(),
            provides: self.provides.clone(),
        }
    }
}

struct FacetShape {
    facet_id: String,
    requires: Vec<FacetServiceReference>,
    provides: Vec<FacetServiceReference>,
}

fn record_service_reference(
    target: &mut Vec<FacetServiceReference>,
    service_id: &str,
    mode: ServiceMode,
) {
    if target
        .iter()
        .any(|reference| reference.service_id == service_id && reference.mode == mode)
    {
        return;
    }
    target.push(FacetServiceReference {
        service_id: service_id.to_string(),
        mode,
    });
}

fn same_references(left: &[FacetServiceReference], right: &[FacetServiceReference]) -> bool {
    left.len() == right.len()
        && left.iter().all(|reference| {
            right.iter().any(|other| {
                other.service_id == reference.service_id && other.mode == reference.mode
            })
        })
}

fn same_facet_shape(left: &FacetShape, right: &FacetShape) -> bool {
    same_references(&left.requires, &right.requires)
        && same_references(&left.provides, &right.provides)
}

// ─── Facet environment ───────────────────────────────────────────────────────

/// Stable handle to one required process-local singleton service; the lifecycle
/// access check runs at every [`LocalRequirement::with`] call.
pub struct LocalRequirement<T> {
    slot: Arc<ServiceSlot>,
    access: AssertAccess,
    /// Identity token mirroring upstream proxy object identity (`env.use`
    /// returns the same view per facet, distinct views across facets).
    identity: u64,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Any + Send + Sync> LocalRequirement<T> {
    /// Run `f` against the live implementation, asserting the facet may access
    /// services right now.
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> Result<R, AnyError> {
        // Upstream asserts service access before resolving the slot binding.
        (self.access)()?;
        let view = self.slot.view::<T>(self.access.clone())?;
        view.with(f)
    }
}

/// Stable handle to one required remote singleton service.
pub struct RemoteRequirement {
    slot: Arc<ServiceSlot>,
    access: AssertAccess,
    /// Identity token mirroring upstream proxy object identity.
    identity: u64,
}

impl RemoteRequirement {
    /// Acquire the consumer client handle, asserting the facet may access
    /// services right now.
    pub fn client(&self) -> Result<Arc<dyn ServiceClientHandle>, AnyError> {
        // Upstream asserts service access before resolving the slot binding.
        (self.access)()?;
        let view = self.slot.view::<ClientTarget>(self.access.clone())?;
        view.with(|target| target.0.clone())
    }
}

impl<T> Clone for LocalRequirement<T> {
    fn clone(&self) -> Self {
        LocalRequirement {
            slot: self.slot.clone(),
            access: self.access.clone(),
            identity: self.identity,
            _marker: PhantomData,
        }
    }
}

impl Clone for RemoteRequirement {
    fn clone(&self) -> Self {
        RemoteRequirement {
            slot: self.slot.clone(),
            access: self.access.clone(),
            identity: self.identity,
        }
    }
}

/// Typed spawner for a process-local keyed service (upstream `ServiceSpawner<T>`
/// over a local provision).
pub struct LocalServiceSpawner<T> {
    inner: Arc<StagedServiceSpawner>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Any + Send + Sync> LocalServiceSpawner<T> {
    /// Spawn one live instance under `key`; returns the instance closer.
    pub fn spawn(&self, key: &str, implementation: Arc<T>) -> Result<Unsubscribe, AnyError> {
        self.inner
            .spawn(key, implementation as Arc<dyn Any + Send + Sync>)
    }
}

impl<T> Clone for LocalServiceSpawner<T> {
    fn clone(&self) -> Self {
        LocalServiceSpawner {
            inner: self.inner.clone(),
            _marker: PhantomData,
        }
    }
}

/// Spawner for a remotely exposable keyed service.
pub struct RemoteServiceSpawner {
    inner: Arc<StagedServiceSpawner>,
}

impl RemoteServiceSpawner {
    /// Spawn one live instance under `key`; returns the instance closer.
    pub fn spawn(
        &self,
        key: &str,
        implementation: Arc<dyn RemoteServiceImpl>,
    ) -> Result<Unsubscribe, AnyError> {
        self.inner.spawn(
            key,
            Arc::new(RemoteImplTarget(implementation)) as Arc<dyn Any + Send + Sync>,
        )
    }
}

impl Clone for RemoteServiceSpawner {
    fn clone(&self) -> Self {
        RemoteServiceSpawner {
            inner: self.inner.clone(),
        }
    }
}

/// Setup-time environment handed to every facet (upstream `FacetEnvironment`).
#[derive(Clone)]
pub struct FacetEnvironment {
    runtime: Arc<Mutex<FacetRuntime>>,
    slots: Arc<HostServiceSlots>,
}

impl FacetEnvironment {
    /// Declare a hard dependency on one process-local singleton service.
    pub fn use_local<T: Any + Send + Sync>(
        &self,
        service: &Service<T>,
    ) -> Result<LocalRequirement<T>, AnyError> {
        let mut runtime = lock(self.runtime.lock(), "facet runtime");
        runtime.lifecycle.assert_setting_up("acquire services")?;
        if !service.local() {
            return Err(type_error(format!(
                "Service {} is not process-local",
                service.id()
            )));
        }
        record_service_reference(&mut runtime.requires, service.id(), ServiceMode::Singleton);
        let service_id = service.id().to_string();
        if let Some(cached) = runtime.singleton_views.get(&service_id) {
            if let Some(requirement) = cached.downcast_ref::<LocalRequirement<T>>() {
                return Ok(requirement.clone());
            }
        }
        let requirement = LocalRequirement {
            slot: self.slots.get_singleton(&service_id, !service.local()),
            access: access_check(&runtime.lifecycle),
            identity: next_requirement_id(),
            _marker: PhantomData,
        };
        runtime
            .singleton_views
            .insert(service_id, Arc::new(requirement.clone()));
        Ok(requirement)
    }

    /// Declare a hard dependency on one remote singleton service.
    pub fn use_remote(&self, service_id: &str) -> Result<RemoteRequirement, AnyError> {
        let mut runtime = lock(self.runtime.lock(), "facet runtime");
        runtime.lifecycle.assert_setting_up("acquire services")?;
        record_service_reference(&mut runtime.requires, service_id, ServiceMode::Singleton);
        let service_key = service_id.to_string();
        if let Some(cached) = runtime.singleton_views.get(&service_key) {
            if let Some(requirement) = cached.downcast_ref::<RemoteRequirement>() {
                return Ok(requirement.clone());
            }
        }
        let requirement = RemoteRequirement {
            slot: self.slots.get_singleton(&service_key, true),
            access: access_check(&runtime.lifecycle),
            identity: next_requirement_id(),
        };
        runtime
            .singleton_views
            .insert(service_key, Arc::new(requirement.clone()));
        Ok(requirement)
    }

    /// Declare a hard dependency on a local keyed service and observe each live
    /// instance once this facet activates.
    pub fn observe_local<T: Any + Send + Sync>(
        &self,
        service: &Service<T>,
        handler: Arc<dyn Fn(LocalServiceView<T>, Context) -> BoxFuture<()> + Send + Sync>,
    ) -> Result<(), AnyError> {
        let mut runtime = lock(self.runtime.lock(), "facet runtime");
        runtime.lifecycle.assert_setting_up("observe services")?;
        record_service_reference(&mut runtime.requires, service.id(), ServiceMode::Keyed);
        let service_id: Arc<str> = Arc::from(service.id());
        let lifecycle = runtime.lifecycle.clone();
        let slots = self.slots.clone();
        let handler = handler.clone();
        runtime.lifecycle.observe(Arc::new(move || {
            let service_id = service_id.clone();
            let assert_access = access_check(&lifecycle);
            let slots = slots.clone();
            let handler = handler.clone();
            let view_service_id = service_id.clone();
            let unsubscribe = slots.observe(
                &service_id,
                Arc::new(move |target, closed, context| {
                    let target = match target {
                        KeyedTarget::Local(target) => target,
                        // Mode validation makes this arm unreachable; the view
                        // fails its downcast instead of panicking.
                        KeyedTarget::Client(_) => Arc::new(()) as Arc<dyn Any + Send + Sync>,
                    };
                    let view = LocalServiceView::<T>::new(
                        &view_service_id,
                        target,
                        assert_access.clone(),
                        closed,
                        context.clone(),
                    );
                    handler(view, context)
                }),
            )?;
            Ok(disposal_from_unsubscribe(unsubscribe))
        }))
    }

    /// Declare a hard dependency on a remote keyed service and observe each
    /// live instance once this facet activates.
    pub fn observe_remote(
        &self,
        service_id: &str,
        handler: Arc<dyn Fn(Arc<dyn ServiceClientHandle>, Context) -> BoxFuture<()> + Send + Sync>,
    ) -> Result<(), AnyError> {
        let mut runtime = lock(self.runtime.lock(), "facet runtime");
        runtime.lifecycle.assert_setting_up("observe services")?;
        record_service_reference(&mut runtime.requires, service_id, ServiceMode::Keyed);
        let service_id = service_id.to_string();
        let lifecycle = runtime.lifecycle.clone();
        let slots = self.slots.clone();
        let handler = handler.clone();
        runtime.lifecycle.observe(Arc::new(move || {
            let service_id = service_id.clone();
            let assert_access = access_check(&lifecycle);
            let slots = slots.clone();
            let handler = handler.clone();
            let view_service_id = service_id.clone();
            let unsubscribe = slots.observe(
                &service_id,
                Arc::new(move |target, closed, context| {
                    let handle: Arc<dyn ServiceClientHandle> = match target {
                        KeyedTarget::Client(client) => Arc::new(ObservationHandle {
                            inner: client,
                            assert_access: assert_access.clone(),
                            closed,
                            context: context.clone(),
                        }),
                        KeyedTarget::Local(_) => Arc::new(MismatchedClientHandle {
                            service_id: view_service_id.clone(),
                        }),
                    };
                    handler(handle, context)
                }),
            )?;
            Ok(disposal_from_unsubscribe(unsubscribe))
        }))
    }

    /// Declare and install this facet's process-local singleton implementation.
    pub fn provide_local<T: Any + Send + Sync>(
        &self,
        service: &Service<T>,
        implementation: Arc<T>,
    ) -> Result<(), AnyError> {
        let mut runtime = lock(self.runtime.lock(), "facet runtime");
        runtime.lifecycle.assert_setting_up("provide services")?;
        if !service.local() {
            return Err(type_error(format!(
                "Service {} is not process-local",
                service.id()
            )));
        }
        record_service_reference(&mut runtime.provides, service.id(), ServiceMode::Singleton);
        runtime
            .provisions
            .push(FacetProvision::Singleton(SingletonProvision {
                service_id: service.id().to_string(),
                local: true,
                mode: ServiceMode::Singleton,
                implementation: implementation as Arc<dyn Any + Send + Sync>,
                remote: None,
            }));
        Ok(())
    }

    /// Declare and install this facet's singleton remote implementation.
    pub fn provide_remote(
        &self,
        service_id: &str,
        mode: ServiceMode,
        implementation: Arc<dyn RemoteServiceImpl>,
    ) -> Result<(), AnyError> {
        let mut runtime = lock(self.runtime.lock(), "facet runtime");
        runtime.lifecycle.assert_setting_up("provide services")?;
        record_service_reference(&mut runtime.provides, service_id, mode);
        runtime
            .provisions
            .push(FacetProvision::Singleton(SingletonProvision {
                service_id: service_id.to_string(),
                local: false,
                mode,
                implementation: Arc::new(()),
                remote: Some(implementation),
            }));
        Ok(())
    }

    /// Declare ownership of a multi-instance process-local service.
    pub fn provide_many_local<T: Any + Send + Sync>(
        &self,
        service: &Service<T>,
    ) -> Result<LocalServiceSpawner<T>, AnyError> {
        let mut runtime = lock(self.runtime.lock(), "facet runtime");
        runtime
            .lifecycle
            .assert_setting_up("provide service instances")?;
        if !service.local() {
            return Err(type_error(format!(
                "Service {} is not process-local",
                service.id()
            )));
        }
        record_service_reference(&mut runtime.provides, service.id(), ServiceMode::Keyed);
        let service_id = service.id().to_string();
        let spawner = StagedServiceSpawner::new(
            runtime.lifecycle.clone(),
            Arc::new(
                move |key: &str, _implementation: &Arc<dyn Any + Send + Sync>| {
                    if key.is_empty() {
                        return Err(type_error("Facet service instance key must not be empty"));
                    }
                    Ok(())
                },
            ),
        );
        runtime
            .provisions
            .push(FacetProvision::Keyed(KeyedProvision {
                service_id,
                local: true,
                spawner: spawner.clone(),
            }));
        Ok(LocalServiceSpawner {
            inner: spawner,
            _marker: PhantomData,
        })
    }

    /// Declare ownership of a multi-instance remotely exposable service.
    pub fn provide_many_remote(
        &self,
        service_id: &str,
        mode: ServiceMode,
    ) -> Result<RemoteServiceSpawner, AnyError> {
        let mut runtime = lock(self.runtime.lock(), "facet runtime");
        runtime
            .lifecycle
            .assert_setting_up("provide service instances")?;
        record_service_reference(&mut runtime.provides, service_id, mode);
        let id = service_id.to_string();
        let spawner = StagedServiceSpawner::new(
            runtime.lifecycle.clone(),
            Arc::new(
                move |key: &str, implementation: &Arc<dyn Any + Send + Sync>| {
                    if key.is_empty() {
                        return Err(type_error("Facet service instance key must not be empty"));
                    }
                    let target = implementation
                        .downcast_ref::<RemoteImplTarget>()
                        .ok_or_else(|| {
                            type_error(format!("Service {id} implementation type mismatch"))
                        })?;
                    validate_remote_service_implementation(&id, target.0.clone())
                },
            ),
        );
        runtime
            .provisions
            .push(FacetProvision::Keyed(KeyedProvision {
                service_id: service_id.to_string(),
                local: false,
                spawner: spawner.clone(),
            }));
        Ok(RemoteServiceSpawner { inner: spawner })
    }

    /// Create initialized mutable replicated state owned by this facet.
    pub fn replicated_state(
        &self,
        initial: JsonValue,
    ) -> Result<Arc<MutableReplicatedState>, AnyError> {
        let runtime = lock(self.runtime.lock(), "facet runtime");
        runtime
            .lifecycle
            .assert_running("create replicated state")?;
        Ok(MutableReplicatedState::new(initial))
    }

    /// Give the facet ownership of a resource cleanup function.
    pub fn own(&self, disposal: Disposal) -> Result<(), AnyError> {
        let runtime = lock(self.runtime.lock(), "facet runtime");
        runtime.lifecycle.own(disposal)
    }

    /// Register asynchronous initialization after dependencies are bound and
    /// ready.
    pub fn on_activate(&self, callback: Disposal) -> Result<(), AnyError> {
        let runtime = lock(self.runtime.lock(), "facet runtime");
        runtime.lifecycle.on_activate(callback)
    }

    /// Register final facet teardown.
    pub fn on_deactivate(&self, callback: Disposal) -> Result<(), AnyError> {
        let runtime = lock(self.runtime.lock(), "facet runtime");
        runtime.lifecycle.own(callback)
    }
}

// ─── Validation ──────────────────────────────────────────────────────────────

fn validate_facets(
    records: &[Arc<Mutex<FacetRuntime>>],
    external_services: &HashMap<String, ExternalService>,
) -> Result<Vec<String>, AnyError> {
    let shapes: Vec<FacetShape> = records
        .iter()
        .map(|record| lock(record.lock(), "facet runtime").shape())
        .collect();
    let mut providers: HashMap<String, (Option<String>, Option<ServiceMode>)> = HashMap::new();
    for (service_id, external) in external_services {
        providers.insert(service_id.clone(), (None, Some(external.mode)));
    }
    for record in &shapes {
        for provision in &record.provides {
            if let Some(existing) = providers.get(&provision.service_id) {
                if let Some(existing_mode) = existing.1 {
                    if existing_mode != provision.mode {
                        return Err(error(format!(
                            "Service {} is provided as both singleton and keyed",
                            provision.service_id
                        )));
                    }
                }
                match &existing.0 {
                    None => {
                        return Err(error(format!(
                            "Service {} is provided by both the host and {}",
                            provision.service_id, record.facet_id
                        )));
                    }
                    Some(existing_facet) => {
                        return Err(error(format!(
                            "Service {} is provided by both {existing_facet} and {}",
                            provision.service_id, record.facet_id
                        )));
                    }
                }
            }
            providers.insert(
                provision.service_id.clone(),
                (Some(record.facet_id.clone()), Some(provision.mode)),
            );
        }
    }

    let mut dependencies: HashMap<String, HashSet<String>> = shapes
        .iter()
        .map(|record| (record.facet_id.clone(), HashSet::new()))
        .collect();
    let mut dependents: HashMap<String, HashSet<String>> = shapes
        .iter()
        .map(|record| (record.facet_id.clone(), HashSet::new()))
        .collect();
    for record in &shapes {
        for requirement in &record.requires {
            let provider = providers.get(&requirement.service_id).ok_or_else(|| {
                error(format!(
                    "Facet {} requires local/{}/{}, but no facet provides it",
                    record.facet_id,
                    requirement.service_id,
                    requirement.mode.as_str()
                ))
            })?;
            if let Some(provider_mode) = provider.1 {
                if provider_mode != requirement.mode {
                    let provider_facet = provider.0.as_deref().unwrap_or("the host");
                    return Err(error(format!(
                        "Facet {} requires {} as {}, but {provider_facet} provides it as {}",
                        record.facet_id,
                        requirement.service_id,
                        requirement.mode.as_str(),
                        provider_mode.as_str()
                    )));
                }
            }
            match &provider.0 {
                None => {}
                Some(self_id) if self_id == &record.facet_id => {}
                Some(provider_facet) => {
                    dependencies
                        .get_mut(&record.facet_id)
                        .expect("dependency set")
                        .insert(provider_facet.clone());
                    dependents
                        .get_mut(provider_facet)
                        .expect("dependent set")
                        .insert(record.facet_id.clone());
                }
            }
        }
    }
    topological_order(&shapes, &dependencies, &dependents)
}

fn topological_order(
    records: &[FacetShape],
    dependencies: &HashMap<String, HashSet<String>>,
    dependents: &HashMap<String, HashSet<String>>,
) -> Result<Vec<String>, AnyError> {
    let mut remaining: HashMap<&str, usize> = dependencies
        .iter()
        .map(|(id, values)| (id.as_str(), values.len()))
        .collect();
    let mut ready: VecDeque<&str> = records
        .iter()
        .map(|record| record.facet_id.as_str())
        .filter(|id| remaining.get(id).copied().unwrap_or(0) == 0)
        .collect();
    let mut order: Vec<String> = Vec::new();
    while let Some(id) = ready.pop_front() {
        order.push(id.to_string());
        if let Some(dependent_ids) = dependents.get(id) {
            for dependent in dependent_ids {
                let count = remaining
                    .get_mut(dependent.as_str())
                    .expect("tracked dependent");
                *count -= 1;
                if *count == 0 {
                    ready.push_back(dependent.as_str());
                }
            }
        }
    }
    if order.len() != records.len() {
        let cycle = records
            .iter()
            .map(|record| record.facet_id.as_str())
            .filter(|id| remaining.get(id).copied().unwrap_or(0) > 0)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(error(format!("Facet dependency cycle: {cycle}")));
    }
    Ok(order)
}

// ─── Generation phases ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerationPhase {
    Setup,
    Assembling,
    Connecting,
    Activating,
    Active,
    Reloading,
    Disposing,
    Dead,
}

impl GenerationPhase {
    fn as_str(self) -> &'static str {
        match self {
            GenerationPhase::Setup => "setup",
            GenerationPhase::Assembling => "assembling",
            GenerationPhase::Connecting => "connecting",
            GenerationPhase::Activating => "activating",
            GenerationPhase::Active => "active",
            GenerationPhase::Reloading => "reloading",
            GenerationPhase::Disposing => "disposing",
            GenerationPhase::Dead => "dead",
        }
    }
}

struct ExternalService {
    mode: ServiceMode,
    source_index: usize,
}

// ─── Kernel ──────────────────────────────────────────────────────────────────

/// Private lifecycle and dependency kernel behind the atomic host entry point.
pub struct FacetKernel {
    initial_facets: Vec<Facet>,
    service_sources: Vec<Arc<dyn RemoteServiceSource>>,
    on_error: ErrorHandler,
    facets: Vec<(String, Arc<Mutex<FacetRuntime>>)>,
    service_slots: Arc<HostServiceSlots>,
    source_bindings: Mutex<HashMap<usize, Arc<dyn RemoteServices>>>,
    activation_order: Vec<String>,
    provider: Option<Arc<RemoteServiceProvider>>,
    internal_services: Option<Arc<RemoteServiceBinding>>,
    local_keyed_services: Option<Arc<LocalKeyedServiceRegistry>>,
    phase: Arc<Mutex<GenerationPhase>>,
}

impl FacetKernel {
    pub fn new(options: FacetOptions) -> Result<FacetKernel, AnyError> {
        let mut unique_ids = HashSet::new();
        for facet in &options.facets {
            if facet.id().is_empty() {
                return Err(error("Facet ID must not be empty"));
            }
            if !unique_ids.insert(facet.id().to_string()) {
                return Err(error("Facet IDs must be unique within a generation"));
            }
        }
        Ok(FacetKernel {
            on_error: options.on_error.unwrap_or_else(|| Arc::new(|_| {})),
            initial_facets: options.facets,
            service_sources: options.service_sources,
            facets: Vec::new(),
            service_slots: Arc::new(HostServiceSlots::new()),
            source_bindings: Mutex::new(HashMap::new()),
            activation_order: Vec::new(),
            provider: None,
            internal_services: None,
            local_keyed_services: None,
            phase: Arc::new(Mutex::new(GenerationPhase::Setup)),
        })
    }

    /// The assembled remote service provider (upstream `get provider`).
    pub fn provider(&self) -> Result<Arc<RemoteServiceProvider>, AnyError> {
        self.provider
            .clone()
            .ok_or_else(|| error("Facet service provider is not assembled"))
    }

    fn phase(&self) -> GenerationPhase {
        *lock(self.phase.lock(), "facet generation phase")
    }

    fn set_phase(&self, phase: GenerationPhase) {
        *lock(self.phase.lock(), "facet generation phase") = phase;
    }

    fn assert_service_target_access(phase: &Arc<Mutex<GenerationPhase>>) -> Result<(), AnyError> {
        let current = *lock(phase.lock(), "facet generation phase");
        if !matches!(
            current,
            GenerationPhase::Activating
                | GenerationPhase::Active
                | GenerationPhase::Reloading
                | GenerationPhase::Disposing
        ) {
            return Err(error(format!(
                "Facet service targets cannot be used during {}",
                current.as_str()
            )));
        }
        Ok(())
    }

    fn facet(&self, facet_id: &str) -> Option<Arc<Mutex<FacetRuntime>>> {
        self.facets
            .iter()
            .find(|(id, _)| id == facet_id)
            .map(|(_, record)| record.clone())
    }

    fn insert_facet(&mut self, facet_id: &str, record: Arc<Mutex<FacetRuntime>>) {
        if let Some(slot) = self.facets.iter_mut().find(|(id, _)| id == facet_id) {
            slot.1 = record;
        } else {
            self.facets.push((facet_id.to_string(), record));
        }
    }

    fn take_facet(&mut self, facet_id: &str) -> Option<Arc<Mutex<FacetRuntime>>> {
        let position = self.facets.iter().position(|(id, _)| id == facet_id)?;
        Some(self.facets.remove(position).1)
    }

    fn create_facet_runtime(&mut self, facet_id: &str) -> Arc<Mutex<FacetRuntime>> {
        let record = Arc::new(Mutex::new(FacetRuntime::new(facet_id)));
        self.insert_facet(facet_id, record.clone());
        record
    }

    fn environment(&self, record: &Arc<Mutex<FacetRuntime>>) -> FacetEnvironment {
        FacetEnvironment {
            runtime: record.clone(),
            slots: self.service_slots.clone(),
        }
    }

    pub async fn activate(&mut self) -> Result<(), AnyError> {
        let initial_facets = self.initial_facets.clone();
        let setup: Result<(), AnyError> = async {
            for facet in &initial_facets {
                let record = self.create_facet_runtime(facet.id());
                let mut environment = self.environment(&record);
                facet.setup(&mut environment)?;
                lock(record.lock(), "facet runtime").lifecycle.prepared()?;
            }

            self.set_phase(GenerationPhase::Assembling);
            let external_services = self.resolve_external_services().await?;
            self.activation_order = validate_facets(&self.runtime_records(), &external_services)?;
            self.assemble_providers()?;
            self.bind_services(&external_services)?;

            self.set_phase(GenerationPhase::Connecting);
            let internal = self.internal_binding()?;
            let readiness: Vec<BoxFuture<Result<(), AnyError>>> = {
                let bindings: Vec<Arc<dyn RemoteServices>> =
                    lock(self.source_bindings.lock(), "source bindings")
                        .values()
                        .cloned()
                        .collect();
                bindings
                    .into_iter()
                    .map(|services| {
                        Box::pin(async move { services.ready(&BACKGROUND_CONTEXT).await })
                            as BoxFuture<_>
                    })
                    .chain(std::iter::once(Box::pin(async move {
                        internal.ready(&BACKGROUND_CONTEXT).await
                    }) as BoxFuture<_>))
                    .collect()
            };
            let results = join_all(readiness).await;
            for result in results {
                result?;
            }

            self.set_phase(GenerationPhase::Activating);
            let order = self.activation_order.clone();
            for id in order {
                let record = self
                    .facet(&id)
                    .ok_or_else(|| error(format!("Facet {id} is not registered")))?;
                let lifecycle = lock(record.lock(), "facet runtime").lifecycle.clone();
                lifecycle.activate().await?;
            }
            self.set_phase(GenerationPhase::Active);
            Ok(())
        }
        .await;
        if let Err(activation_error) = setup {
            let cleanup_errors = self.terminate(Vec::new()).await;
            if !cleanup_errors.is_empty() {
                let mut errors = Vec::with_capacity(cleanup_errors.len() + 1);
                errors.push(activation_error);
                errors.extend(cleanup_errors);
                return Err(AggregateError::new(
                    "Facet generation startup and cleanup failed",
                    errors,
                ));
            }
            return Err(activation_error);
        }
        Ok(())
    }

    pub async fn reload(&mut self, facets: Vec<Facet>) -> Result<(), AnyError> {
        if self.phase() != GenerationPhase::Active {
            return Err(error(format!(
                "Facet host cannot reload while {}",
                self.phase().as_str()
            )));
        }
        let mut unique_ids = HashSet::new();
        for facet in &facets {
            if facet.id().is_empty() {
                return Err(error("Facet ID must not be empty"));
            }
            if !unique_ids.insert(facet.id().to_string()) {
                return Err(error("Reloaded facet IDs must be unique"));
            }
        }
        for facet in &facets {
            if self.facet(facet.id()).is_none() {
                return Err(error(format!("Facet {} is not active", facet.id())));
            }
        }
        self.set_phase(GenerationPhase::Reloading);

        let mut staged: Vec<Arc<Mutex<FacetRuntime>>> = Vec::new();
        let mut candidates: Vec<Arc<Mutex<FacetRuntime>>> = Vec::new();
        let staged_setup: Result<(), AnyError> = async {
            for facet in &facets {
                let record = Arc::new(Mutex::new(FacetRuntime::new(facet.id())));
                staged.push(record.clone());
                let mut environment = self.environment(&record);
                facet.setup(&mut environment)?;
                lock(record.lock(), "facet runtime").lifecycle.prepared()?;
                let previous = self
                    .facet(facet.id())
                    .ok_or_else(|| error(format!("Facet {} is not active", facet.id())))?;
                let previous_shape = lock(previous.lock(), "facet runtime").shape();
                let record_shape = lock(record.lock(), "facet runtime").shape();
                if !same_facet_shape(&previous_shape, &record_shape) {
                    return Err(error(format!(
                        "Reloaded facet {} must preserve its service requirements and provisions",
                        facet.id()
                    )));
                }
                self.validate_replacement_provisions(
                    &lock(record.lock(), "facet runtime").provisions,
                )?;
                candidates.push(record);
            }
            Ok(())
        }
        .await;
        if let Err(setup_error) = staged_setup {
            staged.reverse();
            let cleanup_errors = dispose_facet_records(staged).await;
            if !cleanup_errors.is_empty() {
                let mut errors = Vec::new();
                errors.push(setup_error);
                errors.extend(cleanup_errors);
                errors.extend(self.abort(Vec::new()).await);
                return Err(AggregateError::new(
                    "Facet reload setup and cleanup failed",
                    errors,
                ));
            }
            self.set_phase(GenerationPhase::Active);
            return Err(setup_error);
        }

        let replacements: HashSet<String> = candidates
            .iter()
            .map(|record| lock(record.lock(), "facet runtime").facet_id.clone())
            .collect();
        let candidate_order: Vec<Arc<Mutex<FacetRuntime>>> = self
            .activation_order
            .iter()
            .filter_map(|id| {
                if replacements.contains(id) {
                    candidates
                        .iter()
                        .find(|record| lock(record.lock(), "facet runtime").facet_id == *id)
                        .cloned()
                } else {
                    None
                }
            })
            .collect();
        let activation: Result<(), AnyError> = async {
            for candidate in &candidate_order {
                let lifecycle = lock(candidate.lock(), "facet runtime").lifecycle.clone();
                lifecycle.activate().await?;
            }
            for candidate in &candidate_order {
                self.validate_replacement_provisions(
                    &lock(candidate.lock(), "facet runtime").provisions,
                )?;
            }
            Ok(())
        }
        .await;
        if let Err(activation_error) = activation {
            let mut reversed = candidate_order.clone();
            reversed.reverse();
            let cleanup_errors = dispose_facet_records(reversed).await;
            if !cleanup_errors.is_empty() {
                let mut errors = Vec::new();
                errors.push(activation_error);
                errors.extend(cleanup_errors);
                errors.extend(self.abort(Vec::new()).await);
                return Err(AggregateError::new(
                    "Facet reload activation and cleanup failed",
                    errors,
                ));
            }
            self.set_phase(GenerationPhase::Active);
            return Err(activation_error);
        }

        let mut previous: Vec<Arc<Mutex<FacetRuntime>>> = candidate_order
            .iter()
            .filter_map(|candidate| {
                let id = lock(candidate.lock(), "facet runtime").facet_id.clone();
                self.facet(&id)
            })
            .collect();
        for candidate in &candidate_order {
            let id = lock(candidate.lock(), "facet runtime").facet_id.clone();
            self.insert_facet(&id, candidate.clone());
        }
        let cutover: Result<(), AnyError> = async {
            for candidate in &candidate_order {
                let provisions = lock(candidate.lock(), "facet runtime").provisions.clone();
                for provision in provisions {
                    if let FacetProvision::Singleton(singleton) = &provision {
                        if singleton.local {
                            self.service_slots.bind_singleton(
                                &singleton.service_id,
                                singleton.implementation.clone(),
                            );
                        } else {
                            provision
                                .replace(self.provider.as_ref().expect("assembled provider"))?;
                        }
                    }
                }
            }
            previous.reverse();
            let retirement_errors = dispose_facet_records(previous.clone()).await;
            if retirement_errors.len() == 1 {
                return Err(retirement_errors.into_iter().next().expect("one error"));
            }
            if retirement_errors.len() > 1 {
                return Err(AggregateError::new(
                    "Failed to retire replaced facets",
                    retirement_errors,
                ));
            }
            let registry = self.local_keyed_registry()?;
            for candidate in &candidate_order {
                let provisions = lock(candidate.lock(), "facet runtime").provisions.clone();
                for provision in provisions {
                    match &provision {
                        FacetProvision::Keyed(keyed) => {
                            if keyed.local {
                                provision.connect_local(registry.clone())?;
                            } else {
                                provision.connect_remote(
                                    self.provider.as_ref().expect("assembled provider").clone(),
                                )?;
                            }
                        }
                        FacetProvision::Singleton(_) => {}
                    }
                }
            }
            Ok(())
        }
        .await;
        if let Err(cutover_error) = cutover {
            let abort_errors = self.abort(previous).await;
            let mut errors = Vec::new();
            errors.push(cutover_error);
            errors.extend(abort_errors);
            return Err(AggregateError::new(
                "Facet reload failed after cutover",
                errors,
            ));
        }
        self.set_phase(GenerationPhase::Active);
        Ok(())
    }

    pub async fn dispose(&mut self) -> Result<(), AnyError> {
        let phase = self.phase();
        if phase == GenerationPhase::Dead {
            return Ok(());
        }
        if phase != GenerationPhase::Active {
            return Err(error(format!(
                "Facet host cannot be disposed while {}",
                phase.as_str()
            )));
        }
        let errors = self.terminate(Vec::new()).await;
        if errors.len() == 1 {
            return Err(errors.into_iter().next().expect("one error"));
        }
        if errors.len() > 1 {
            return Err(AggregateError::new(
                "Failed to dispose facet generation",
                errors,
            ));
        }
        Ok(())
    }

    fn validate_replacement_provisions(
        &self,
        provisions: &[FacetProvision],
    ) -> Result<(), AnyError> {
        for provision in provisions {
            if let FacetProvision::Singleton(singleton) = provision {
                if singleton.local {
                    continue;
                }
                provision
                    .validate_replacement(self.provider.as_ref().expect("assembled provider"))?;
            }
        }
        Ok(())
    }

    fn runtime_records(&self) -> Vec<Arc<Mutex<FacetRuntime>>> {
        self.facets
            .iter()
            .map(|(_, record)| record.clone())
            .collect()
    }

    fn provisions(&self) -> Vec<FacetProvision> {
        self.facets
            .iter()
            .flat_map(|(_, record)| lock(record.lock(), "facet runtime").provisions.clone())
            .collect()
    }

    fn local_keyed_registry(&self) -> Result<Arc<LocalKeyedServiceRegistry>, AnyError> {
        self.local_keyed_services
            .clone()
            .ok_or_else(|| error("Facet keyed services are not assembled"))
    }

    fn internal_binding(&self) -> Result<Arc<RemoteServiceBinding>, AnyError> {
        self.internal_services
            .clone()
            .ok_or_else(|| error("Facet remote services are not assembled"))
    }

    async fn resolve_external_services(
        &mut self,
    ) -> Result<HashMap<String, ExternalService>, AnyError> {
        let mut catalogues: Vec<(usize, Vec<crate::types::ServiceCatalogueEntry>)> = Vec::new();
        for (index, source) in self.service_sources.iter().enumerate() {
            let entries = source.catalogue(&BACKGROUND_CONTEXT).await?;
            catalogues.push((index, entries));
        }
        let mut offered: HashMap<String, (ServiceMode, usize)> = HashMap::new();
        for (index, entries) in &catalogues {
            for entry in entries {
                if offered.contains_key(&entry.service_id) {
                    return Err(error(format!(
                        "Facet host service {} is offered by more than one source",
                        entry.service_id
                    )));
                }
                offered.insert(entry.service_id.clone(), (entry.mode, *index));
            }
        }

        let records = self.runtime_records();
        let local: HashSet<String> = records
            .iter()
            .flat_map(|record| lock(record.lock(), "facet runtime").provides.clone())
            .map(|reference| reference.service_id)
            .collect();
        let mut external: HashMap<String, ExternalService> = HashMap::new();
        for record in &records {
            let requires = lock(record.lock(), "facet runtime").requires.clone();
            for requirement in requires {
                if local.contains(&requirement.service_id)
                    || external.contains_key(&requirement.service_id)
                {
                    continue;
                }
                let mut resolved = offered.get(&requirement.service_id).copied();
                if resolved.is_none() {
                    let deferred: Vec<usize> = self
                        .service_sources
                        .iter()
                        .enumerate()
                        .filter(|(_, source)| source.accepts_unavailable_services())
                        .map(|(index, _)| index)
                        .collect();
                    if deferred.len() > 1 {
                        return Err(error(format!(
                            "Facet host service {} has more than one deferred source",
                            requirement.service_id
                        )));
                    }
                    if deferred.len() == 1 {
                        resolved = Some((requirement.mode, deferred[0]));
                    }
                }
                if let Some((mode, source_index)) = resolved {
                    external.insert(
                        requirement.service_id.clone(),
                        ExternalService { mode, source_index },
                    );
                }
            }
        }

        let mut service_ids_by_source: HashMap<usize, Vec<String>> = HashMap::new();
        for (service_id, external_service) in &external {
            service_ids_by_source
                .entry(external_service.source_index)
                .or_default()
                .push(service_id.clone());
        }
        for (source_index, service_ids) in service_ids_by_source {
            let source = self.service_sources[source_index].clone();
            let phase = self.phase.clone();
            let binding = source
                .open(RemoteServiceSourceOpenOptions {
                    services: service_ids,
                    assert_access: Arc::new(move || Self::assert_service_target_access(&phase)),
                    on_error: self.on_error.clone(),
                })
                .await?;
            lock(self.source_bindings.lock(), "source bindings")
                .insert(source_index, Arc::from(binding));
        }
        Ok(external)
    }

    fn assemble_providers(&mut self) -> Result<(), AnyError> {
        let provisions = self.provisions();
        let remote_provisions: Vec<&FacetProvision> = provisions
            .iter()
            .filter(|provision| !provision.local())
            .collect();
        let provider = Arc::new(RemoteServiceProvider::new(
            remote_provisions
                .iter()
                .map(|provision| ProviderDefinition {
                    service_id: provision.service_id().to_string(),
                    mode: provision.mode(),
                })
                .collect(),
        )?);
        let transport = create_loopback_service_transport(provider.clone());
        let phase = self.phase.clone();
        let internal_services = Arc::new(RemoteServiceBinding::new(RemoteServiceBindingOptions {
            services: remote_provisions
                .iter()
                .map(|provision| provision.service_id().to_string())
                .collect(),
            transport,
            bound: true,
            on_error: Some(self.on_error.clone()),
            assert_access: Some(Arc::new(move || Self::assert_service_target_access(&phase))),
        })?);
        let local_keyed_services = Arc::new(LocalKeyedServiceRegistry::new(
            provisions
                .iter()
                .filter(
                    |provision| matches!(provision, FacetProvision::Keyed(keyed) if keyed.local),
                )
                .map(|provision| provision.service_id().to_string())
                .collect(),
            self.on_error.clone(),
        )?);

        for provision in &provisions {
            match provision {
                FacetProvision::Singleton(_) => {
                    if !provision.local() {
                        provision.install(&provider)?;
                    }
                }
                FacetProvision::Keyed(_) => {
                    if provision.local() {
                        provision.connect_local(local_keyed_services.clone())?;
                    } else {
                        provision.connect_remote(provider.clone())?;
                    }
                }
            }
        }

        self.provider = Some(provider);
        self.internal_services = Some(internal_services);
        self.local_keyed_services = Some(local_keyed_services);
        Ok(())
    }

    fn bind_services(
        &self,
        external_services: &HashMap<String, ExternalService>,
    ) -> Result<(), AnyError> {
        for provision in self.provisions() {
            match &provision {
                FacetProvision::Singleton(singleton) => {
                    if !self.service_slots.has_singleton(&singleton.service_id) {
                        continue;
                    }
                    let target: Arc<dyn Any + Send + Sync> = if singleton.local {
                        singleton.implementation.clone()
                    } else {
                        let handle = self
                            .internal_binding()?
                            .use_service(&singleton.service_id)?;
                        Arc::new(ClientTarget(handle))
                    };
                    self.service_slots
                        .bind_singleton(&singleton.service_id, target);
                }
                FacetProvision::Keyed(keyed) => {
                    let source: Arc<dyn KeyedServiceSource> = if keyed.local {
                        self.local_keyed_registry()?
                    } else {
                        Arc::new(RemoteServicesKeyedSource(self.internal_binding()?))
                    };
                    self.service_slots.bind_keyed(&keyed.service_id, source);
                }
            }
        }
        for (service_id, external_service) in external_services {
            let services = lock(self.source_bindings.lock(), "source bindings")
                .get(&external_service.source_index)
                .cloned()
                .ok_or_else(|| error(format!("Service source for {service_id} is not open")))?;
            if external_service.mode == ServiceMode::Singleton {
                let handle = services.use_service(service_id)?;
                self.service_slots
                    .bind_singleton(service_id, Arc::new(ClientTarget(handle)));
            } else {
                self.service_slots
                    .bind_keyed(service_id, Arc::new(RemoteServicesKeyedSource(services)));
            }
        }
        Ok(())
    }

    async fn dispose_service_bindings(&mut self) -> Vec<AnyError> {
        let mut bindings: Vec<Arc<dyn RemoteServices>> =
            std::mem::take(&mut *lock(self.source_bindings.lock(), "source bindings"))
                .into_values()
                .collect();
        if let Some(internal) = self.internal_services.take() {
            bindings.push(internal);
        }
        let results: Vec<Result<(), AnyError>> = join_all(bindings.into_iter().map(|services| {
            Box::pin(async move { services.dispose(&BACKGROUND_CONTEXT).await })
                as BoxFuture<Result<(), AnyError>>
        }))
        .await;
        results
            .into_iter()
            .filter_map(|result| result.err())
            .collect()
    }

    async fn abort(&mut self, extra_records: Vec<Arc<Mutex<FacetRuntime>>>) -> Vec<AnyError> {
        for (_, record) in &self.facets {
            lock(record.lock(), "facet runtime").lifecycle.revoke();
        }
        for record in &extra_records {
            lock(record.lock(), "facet runtime").lifecycle.revoke();
        }
        self.terminate(extra_records).await
    }

    async fn terminate(&mut self, extra_records: Vec<Arc<Mutex<FacetRuntime>>>) -> Vec<AnyError> {
        self.set_phase(GenerationPhase::Disposing);
        let mut errors = self.dispose_lifecycles().await;
        let mut reversed = extra_records;
        reversed.reverse();
        errors.extend(dispose_facet_records(reversed).await);
        if let Some(registry) = self.local_keyed_services.take() {
            registry.dispose();
        }
        errors.extend(self.dispose_service_bindings().await);
        self.service_slots.dispose();
        if let Some(provider) = self.provider.take() {
            if let Err(dispose_error) = provider.dispose() {
                errors.push(dispose_error);
            }
        }
        self.set_phase(GenerationPhase::Dead);
        errors
    }

    async fn dispose_lifecycles(&mut self) -> Vec<AnyError> {
        let mut errors: Vec<AnyError> = Vec::new();
        let order: Vec<String> = if !self.activation_order.is_empty() {
            self.activation_order.iter().rev().cloned().collect()
        } else {
            self.facets.iter().rev().map(|(id, _)| id.clone()).collect()
        };
        for id in order {
            let Some(record) = self.take_facet(&id) else {
                continue;
            };
            let lifecycle = lock(record.lock(), "facet runtime").lifecycle.clone();
            if let Err(dispose_error) = lifecycle.dispose().await {
                errors.push(dispose_error);
            }
        }
        errors
    }
}

async fn dispose_facet_records(records: Vec<Arc<Mutex<FacetRuntime>>>) -> Vec<AnyError> {
    let mut errors: Vec<AnyError> = Vec::new();
    for record in records {
        let lifecycle = lock(record.lock(), "facet runtime").lifecycle.clone();
        if let Err(dispose_error) = lifecycle.dispose().await {
            errors.push(dispose_error);
        }
    }
    errors
}

#[cfg(test)]
mod tests {
    //! Port of upstream `test/facets.test.ts`.
    //!
    //! Non-portable cases and their reasons:
    //! - "terminates the host when keyed replacement publication fails" /
    //!   "terminates the host when keyed retirement publication fails": upstream
    //!   injects failure through a provider-subscription listener that throws;
    //!   the Rust `UpdateListener` returns `()`, so publication listeners cannot
    //!   fail and the reload-after-cutover error path is unreachable from tests.
    //! - asynchronous setup rejection ("Facet {id} setup must be synchronous"):
    //!   `Facet::setup` is synchronous by construction (`Result`), so the
    //!   promise-like check cannot fire.
    //! - JS proxy identity assertions (`Object.is` on service views) port to
    //!   `Arc::ptr_eq` on the underlying slot, and extracted member methods
    //!   (`const read = handle.read`) collapse into late-binding calls through
    //!   the same requirement handle.

    use super::*;
    use crate::api::{create_facet_host, create_remote_service_binding, define_service};
    use crate::services::state_internals::StateInternals;
    use crate::types::{
        FacetHost, RemoteServices, ServiceCall, ServiceCatalogueEntry, ServiceMember, StateListener,
    };
    use serde_json::json;
    use std::sync::LazyLock;
    use std::time::Duration;

    type Trace = Arc<Mutex<Vec<String>>>;

    // ─── Service identities (upstream `defineService` constants) ────────────

    struct SourceSvc;
    struct ProjectionSvc;
    struct KeyedValueSvc;
    struct WatchedSvc;
    struct LeftValueSvc;
    struct RightValueSvc;
    struct CombinedValueSvc;

    /// Local `HostValues` implementation (upstream `{ name, use }` object).
    #[derive(Clone)]
    struct HostValues {
        name: String,
        use_: String,
    }

    /// Local keyed `LocalKeyedValue` implementation (upstream `{ metadata, read }`).
    #[derive(Clone)]
    struct LocalKeyedValue {
        value: String,
    }

    impl LocalKeyedValue {
        fn read(&self) -> &str {
            &self.value
        }
    }

    static SOURCE: LazyLock<Service<SourceSvc>> = LazyLock::new(|| {
        define_service("test.experimental.source", false).expect("define service")
    });
    static PROJECTION: LazyLock<Service<ProjectionSvc>> = LazyLock::new(|| {
        define_service("test.experimental.projection", false).expect("define service")
    });
    static KEYED_VALUE: LazyLock<Service<KeyedValueSvc>> = LazyLock::new(|| {
        define_service("test.experimental.keyed-value", false).expect("define service")
    });
    static WATCHED: LazyLock<Service<WatchedSvc>> = LazyLock::new(|| {
        define_service("test.experimental.watched", false).expect("define service")
    });
    static HOST_VALUES: LazyLock<Service<HostValues>> = LazyLock::new(|| {
        define_service("test.experimental.host-values", true).expect("define service")
    });
    static LOCAL_KEYED_VALUE: LazyLock<Service<LocalKeyedValue>> = LazyLock::new(|| {
        define_service("test.experimental.local-keyed-value", true).expect("define service")
    });
    static LEFT_VALUE: LazyLock<Service<LeftValueSvc>> = LazyLock::new(|| {
        define_service("test.experimental.left-value", false).expect("define service")
    });
    static RIGHT_VALUE: LazyLock<Service<RightValueSvc>> = LazyLock::new(|| {
        define_service("test.experimental.right-value", false).expect("define service")
    });
    static COMBINED_VALUE: LazyLock<Service<CombinedValueSvc>> = LazyLock::new(|| {
        define_service("test.experimental.combined-value", false).expect("define service")
    });

    // ─── Helpers ─────────────────────────────────────────────────────────────

    fn trace() -> Trace {
        Arc::new(Mutex::new(Vec::new()))
    }

    fn push(trace: &Trace, entry: &str) {
        trace.lock().expect("trace lock").push(entry.to_string());
    }

    fn snapshot(trace: &Trace) -> Vec<String> {
        trace.lock().expect("trace lock").clone()
    }

    fn trace_disposal(trace: &Trace, entry: &str) -> Disposal {
        let trace = trace.clone();
        let entry = entry.to_string();
        Arc::new(move || {
            push(&trace, &entry);
            Box::pin(async { Ok(()) }) as BoxFuture<Result<(), AnyError>>
        })
    }

    /// Minimal [`RemoteServiceImpl`] standing in for upstream plain-object
    /// implementations (the provider's `RemoteServiceObject` builder is owned by
    /// the services slice; tests only need the trait).
    struct TestRemoteObject {
        methods: Vec<(&'static str, crate::types::RemoteMethod)>,
        states: Vec<(&'static str, Arc<dyn StateInternals>)>,
    }

    impl RemoteServiceImpl for TestRemoteObject {
        fn members(&self) -> Vec<(String, ServiceMember)> {
            let mut members: Vec<(String, ServiceMember)> = self
                .methods
                .iter()
                .map(|(name, method)| ((*name).to_string(), ServiceMember::Method(method.clone())))
                .collect();
            members.extend(self.states.iter().map(|(name, internals)| {
                ((*name).to_string(), ServiceMember::State(internals.clone()))
            }));
            members
        }
    }

    fn method_object(
        methods: Vec<(&'static str, crate::types::RemoteMethod)>,
    ) -> Arc<dyn RemoteServiceImpl> {
        Arc::new(TestRemoteObject {
            methods,
            states: Vec::new(),
        })
    }

    fn read_method(value: Arc<String>) -> crate::types::RemoteMethod {
        Arc::new(move |_args: Vec<JsonValue>, _context: Context| {
            let value = value.clone();
            Box::pin(async move { Ok(Some(json!(&*value))) }) as BoxFuture<_>
        })
    }

    fn read_impl(value: impl Into<String>) -> Arc<dyn RemoteServiceImpl> {
        method_object(vec![("read", read_method(Arc::new(value.into())))])
    }

    fn state_impl(
        name: &'static str,
        internals: Arc<dyn StateInternals>,
    ) -> Arc<dyn RemoteServiceImpl> {
        Arc::new(TestRemoteObject {
            methods: Vec::new(),
            states: vec![(name, internals)],
        })
    }

    fn invoke_read(
        client: &Arc<dyn ServiceClientHandle>,
        context: &Context,
    ) -> BoxFuture<Result<JsonValue, AnyError>> {
        let client = client.clone();
        let context = context.clone();
        Box::pin(async move {
            Ok(client
                .invoke("read", Vec::new(), &context)
                .await?
                .unwrap_or(json!(null)))
        })
    }

    /// Unwrap a service-handle access error without requiring the client type
    /// to be `Debug`.
    fn client_error(result: Result<Arc<dyn ServiceClientHandle>, AnyError>) -> AnyError {
        match result {
            Ok(_) => panic!("expected service access rejection"),
            Err(error) => error,
        }
    }

    async fn expect_host_failure(result: Result<Arc<dyn FacetHost>, AnyError>) -> AnyError {
        match result {
            Ok(_) => panic!("expected host creation failure"),
            Err(error) => error,
        }
    }

    async fn wait_for(condition: impl Fn() -> bool) {
        for _ in 0..1000 {
            if condition() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("timed out waiting for condition");
    }

    fn options(facets: Vec<Facet>) -> FacetOptions {
        FacetOptions {
            facets,
            service_sources: Vec::new(),
            on_error: None,
        }
    }

    // ─── facet host ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn discovers_setup_dependencies_before_connecting_stable_service_handles() {
        let trace = trace();
        let source_handle: Arc<Mutex<Option<RemoteRequirement>>> = Arc::new(Mutex::new(None));

        let projection_handle = source_handle.clone();
        let projection_trace = trace.clone();
        let projection = Facet::new("projection", move |env: &mut FacetEnvironment| {
            push(&projection_trace, "setup projection");
            let handle = env.use_remote(SOURCE.id())?;
            let setup_error = match handle.client() {
                Err(setup_error) => setup_error,
                Ok(_) => panic!("expected setup-time service access rejection"),
            };
            assert_eq!(
                setup_error.to_string(),
                "Facet projection service handles cannot be used while setting_up"
            );
            *projection_handle.lock().expect("source handle lock") = Some(handle.clone());
            let source = projection_handle.clone();
            let implementation = method_object(vec![(
                "read",
                Arc::new(move |_args: Vec<JsonValue>, context: Context| {
                    let source = source.clone();
                    Box::pin(async move {
                        let requirement = source
                            .lock()
                            .expect("source handle lock")
                            .clone()
                            .expect("source handle");
                        let client = requirement.client()?;
                        client.invoke("read", Vec::new(), &context).await
                    }) as BoxFuture<_>
                }),
            )]);
            env.provide_remote(PROJECTION.id(), ServiceMode::Singleton, implementation)?;
            env.on_activate(trace_disposal(&projection_trace, "activate projection"))?;
            env.on_deactivate(trace_disposal(&projection_trace, "dispose projection"))?;
            Ok(())
        });

        let source_trace = trace.clone();
        let source = Facet::new("source", move |env: &mut FacetEnvironment| {
            push(&source_trace, "setup source");
            env.provide_remote(SOURCE.id(), ServiceMode::Singleton, read_impl("value"))?;
            env.on_activate(trace_disposal(&source_trace, "activate source"))?;
            env.on_deactivate(trace_disposal(&source_trace, "dispose source"))?;
            Ok(())
        });

        let host = create_facet_host(options(vec![projection, source]))
            .await
            .expect("host activates");

        assert_eq!(
            snapshot(&trace),
            [
                "setup projection",
                "setup source",
                "activate source",
                "activate projection"
            ]
        );
        let requirement = source_handle
            .lock()
            .expect("source handle lock")
            .clone()
            .expect("source handle");
        let client = requirement.client().expect("client after activation");
        let value = client
            .invoke("read", Vec::new(), &BACKGROUND_CONTEXT)
            .await
            .expect("source read");
        assert_eq!(value, Some(json!("value")));

        let call = ServiceCall {
            service_id: PROJECTION.id().to_string(),
            instance: None,
            member: "read".to_string(),
            args: Vec::new(),
        };
        let value = host
            .services()
            .invoke(&call, &BACKGROUND_CONTEXT)
            .await
            .expect("projection read");
        assert_eq!(value, Some(json!("value")));

        host.dispose().await.expect("host disposes");
        let entries = snapshot(&trace);
        assert_eq!(
            &entries[entries.len() - 2..],
            ["dispose projection", "dispose source"]
        );
    }

    #[tokio::test]
    async fn connects_keyed_observations_only_when_the_observing_facet_activates() {
        let trace = trace();

        let observer = {
            let observer_trace = trace.clone();
            Facet::new("observer", move |env: &mut FacetEnvironment| {
                let handler_trace = observer_trace.clone();
                let handler: Arc<
                    dyn Fn(Arc<dyn ServiceClientHandle>, Context) -> BoxFuture<()> + Send + Sync,
                > = Arc::new(
                    move |client: Arc<dyn ServiceClientHandle>, context: Context| {
                        let observer_trace = handler_trace.clone();
                        Box::pin(async move {
                            if let Ok(Some(value)) =
                                client.invoke("read", Vec::new(), &context).await
                            {
                                push(
                                    &observer_trace,
                                    &format!("observe {}", value.as_str().unwrap_or_default()),
                                );
                            }
                        })
                    },
                );
                env.observe_remote(KEYED_VALUE.id(), handler)
                    .expect("observe");
                env.on_activate(trace_disposal(&observer_trace, "activate observer"))?;
                Ok(())
            })
        };

        let provider_trace = trace.clone();
        let provider = Facet::new("provider", move |env: &mut FacetEnvironment| {
            let spawner = env
                .provide_many_remote(KEYED_VALUE.id(), ServiceMode::Keyed)
                .expect("provide many");
            let activation_trace = provider_trace.clone();
            env.on_activate(Arc::new(move || {
                push(&activation_trace, "activate provider");
                if let Err(spawn_error) = spawner.spawn("one", read_impl("one")) {
                    return Box::pin(async move { Err(spawn_error) })
                        as BoxFuture<Result<(), AnyError>>;
                }
                Box::pin(async { Ok(()) }) as BoxFuture<Result<(), AnyError>>
            }))?;
            Ok(())
        });

        let mut kernel = FacetKernel::new(options(vec![observer, provider])).expect("kernel");
        kernel.activate().await.expect("activate");
        wait_for(|| snapshot(&trace).iter().any(|entry| entry == "observe one")).await;
        assert_eq!(
            snapshot(&trace),
            ["activate provider", "activate observer", "observe one"]
        );

        let provider = kernel.provider().expect("provider");
        let remote_services = create_remote_service_binding(RemoteServiceBindingOptions {
            services: vec![KEYED_VALUE.id().to_string()],
            transport: create_loopback_service_transport(provider),
            bound: true,
            on_error: None,
            assert_access: None,
        })
        .expect("binding");
        let remote_values: Arc<Mutex<Vec<JsonValue>>> = Arc::new(Mutex::new(Vec::new()));
        let remote_values_for_handler = remote_values.clone();
        let handler: KeyedObservationHandler = Arc::new(
            move |client: Arc<dyn ServiceClientHandle>, context: Context| {
                let remote_values = remote_values_for_handler.clone();
                Box::pin(async move {
                    if let Ok(Some(value)) = client.invoke("read", Vec::new(), &context).await {
                        remote_values
                            .lock()
                            .expect("remote values lock")
                            .push(value);
                    }
                })
            },
        );
        let _observation = remote_services
            .observe(KEYED_VALUE.id(), handler)
            .expect("binding observe");
        remote_services
            .ready(&BACKGROUND_CONTEXT)
            .await
            .expect("binding ready");
        wait_for(|| remote_values.lock().expect("remote values lock").len() == 1).await;
        assert_eq!(
            *remote_values.lock().expect("remote values lock"),
            vec![json!("one")]
        );

        remote_services
            .dispose(&BACKGROUND_CONTEXT)
            .await
            .expect("binding dispose");
        kernel.dispose().await.expect("kernel dispose");
    }

    #[tokio::test]
    async fn routes_remotely_exposable_keyed_services_through_the_host_provider() {
        let observed: Arc<Mutex<Vec<(Arc<dyn ServiceClientHandle>, Context)>>> =
            Arc::new(Mutex::new(Vec::new()));
        let consumer = {
            let observed_for_consumer = observed.clone();
            Facet::new(
                "remote-keyed-consumer",
                move |env: &mut FacetEnvironment| {
                    let handler_observed = observed_for_consumer.clone();
                    let handler: Arc<
                        dyn Fn(Arc<dyn ServiceClientHandle>, Context) -> BoxFuture<()>
                            + Send
                            + Sync,
                    > = Arc::new(
                        move |client: Arc<dyn ServiceClientHandle>, context: Context| {
                            let observed = handler_observed.clone();
                            Box::pin(async move {
                                observed
                                    .lock()
                                    .expect("observed lock")
                                    .push((client, context));
                            })
                        },
                    );
                    env.observe_remote(KEYED_VALUE.id(), handler)
                        .expect("observe");
                    Ok(())
                },
            )
        };

        let provider = |value: &'static str| {
            Facet::new(
                "remote-keyed-provider",
                move |env: &mut FacetEnvironment| {
                    let spawner = env
                        .provide_many_remote(KEYED_VALUE.id(), ServiceMode::Keyed)
                        .expect("provide many");
                    env.on_activate(Arc::new(move || {
                        if let Err(spawn_error) = spawner.spawn("current", read_impl(value)) {
                            return Box::pin(async move { Err(spawn_error) })
                                as BoxFuture<Result<(), AnyError>>;
                        }
                        Box::pin(async { Ok(()) }) as BoxFuture<Result<(), AnyError>>
                    }))?;
                    Ok(())
                },
            )
        };

        let host = create_facet_host(options(vec![consumer, provider("A")]))
            .await
            .expect("host activates");
        wait_for(|| observed.lock().expect("observed lock").len() == 1).await;
        let first = observed.lock().expect("observed lock")[0].clone();
        let value = invoke_read(&first.0, &first.1).await.expect("first read");
        assert_eq!(value, json!("A"));

        host.reload(vec![provider("B")]).await.expect("reload");
        wait_for(|| observed.lock().expect("observed lock").len() == 2).await;
        assert!(first
            .1
            .abort_signal()
            .is_some_and(|signal| signal.aborted()));
        let error = invoke_read(&first.0, &first.1)
            .await
            .expect_err("stale read");
        assert!(
            error
                .to_string()
                .contains("Keyed service test.experimental.keyed-value observation is closed"),
            "{error}"
        );
        let second = observed.lock().expect("observed lock")[1].clone();
        let value = invoke_read(&second.0, &second.1)
            .await
            .expect("second read");
        assert_eq!(value, json!("B"));

        host.dispose().await.expect("dispose");
        assert!(second
            .1
            .abort_signal()
            .is_some_and(|signal| signal.aborted()));
    }

    #[tokio::test]
    async fn keeps_unrestricted_local_keyed_services_process_local_across_provider_reloads() {
        let observed: Arc<Mutex<Vec<(LocalServiceView<LocalKeyedValue>, Context)>>> =
            Arc::new(Mutex::new(Vec::new()));
        let consumer = {
            let observed_for_consumer = observed.clone();
            Facet::new("local-keyed-consumer", move |env: &mut FacetEnvironment| {
                let handler_observed = observed_for_consumer.clone();
                let handler: Arc<
                    dyn Fn(LocalServiceView<LocalKeyedValue>, Context) -> BoxFuture<()>
                        + Send
                        + Sync,
                > = Arc::new(
                    move |view: LocalServiceView<LocalKeyedValue>, context: Context| {
                        let observed = handler_observed.clone();
                        Box::pin(async move {
                            observed
                                .lock()
                                .expect("observed lock")
                                .push((view, context));
                        })
                    },
                );
                env.observe_local(&*LOCAL_KEYED_VALUE, handler)
                    .expect("observe");
                Ok(())
            })
        };

        let provider = |value: &'static str| {
            Facet::new("local-keyed-provider", move |env: &mut FacetEnvironment| {
                let spawner = env
                    .provide_many_local(&*LOCAL_KEYED_VALUE)
                    .expect("provide many");
                env.on_activate(Arc::new(move || {
                    if let Err(spawn_error) = spawner.spawn(
                        "current",
                        Arc::new(LocalKeyedValue {
                            value: value.to_string(),
                        }),
                    ) {
                        return Box::pin(async move { Err(spawn_error) })
                            as BoxFuture<Result<(), AnyError>>;
                    }
                    Box::pin(async { Ok(()) }) as BoxFuture<Result<(), AnyError>>
                }))?;
                Ok(())
            })
        };

        let host = create_facet_host(options(vec![consumer, provider("A")]))
            .await
            .expect("host activates");
        wait_for(|| observed.lock().expect("observed lock").len() == 1).await;
        let first = observed.lock().expect("observed lock")[0].clone();
        assert_eq!(
            first.0.with(|service| service.read().to_string()).unwrap(),
            "A"
        );
        assert!(!host
            .services()
            .catalogue()
            .iter()
            .any(|entry| entry.service_id == LOCAL_KEYED_VALUE.id()));
        // Upstream asserts the provider's remotable check ("process-local"); the
        // Rust provider only sees IDs, so the local rejection surfaces as its
        // no-local-provider error instead.
        assert!(host
            .services()
            .use_singleton(LOCAL_KEYED_VALUE.id())
            .is_err());

        host.reload(vec![provider("B")]).await.expect("reload");
        wait_for(|| observed.lock().expect("observed lock").len() == 2).await;
        assert!(first
            .1
            .abort_signal()
            .is_some_and(|signal| signal.aborted()));
        let error = first
            .0
            .with(|service| service.read().to_string())
            .expect_err("stale local read");
        assert_eq!(
            error.to_string(),
            format!(
                "Keyed service {} observation is closed",
                LOCAL_KEYED_VALUE.id()
            )
        );
        let second = observed.lock().expect("observed lock")[1].clone();
        assert_eq!(
            second.0.with(|service| service.read().to_string()).unwrap(),
            "B"
        );

        host.dispose().await.expect("dispose");
        assert!(second
            .1
            .abort_signal()
            .is_some_and(|signal| signal.aborted()));
    }

    #[tokio::test]
    async fn keeps_remotely_exposable_local_state_replicas_stable_across_provider_reloads() {
        let sources: Arc<Mutex<Vec<Arc<MutableReplicatedState>>>> =
            Arc::new(Mutex::new(Vec::new()));
        let revisions: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let watched: Arc<Mutex<Option<RemoteRequirement>>> = Arc::new(Mutex::new(None));

        let watched_for_consumer = watched.clone();
        let revisions_for_consumer = revisions.clone();
        let consumer = Facet::new("state-consumer", move |env: &mut FacetEnvironment| {
            let requirement = env.use_remote(WATCHED.id())?;
            *watched_for_consumer.lock().expect("watched lock") = Some(requirement.clone());
            let revisions = revisions_for_consumer.clone();
            let watched = requirement.clone();
            let environment = env.clone();
            env.on_activate(Arc::new(move || {
                let watched = watched.clone();
                let revisions = revisions.clone();
                let environment = environment.clone();
                Box::pin(async move {
                    let client = watched.client()?;
                    let state = client.state("state")?.expect("watched state member");
                    let listener: StateListener = Arc::new(move |value, _context, _delivery| {
                        let revisions = revisions.clone();
                        let value = value.clone();
                        Box::pin(async move {
                            revisions
                                .lock()
                                .expect("revisions lock")
                                .push(value["value"].as_i64().unwrap_or_default());
                        })
                    });
                    let unsubscribe = state.subscribe(listener);
                    environment.own(disposal_from_unsubscribe(unsubscribe))?;
                    Ok(())
                })
            }))?;
            Ok(())
        });

        let provider = |value: i64, sources: Arc<Mutex<Vec<Arc<MutableReplicatedState>>>>| {
            Facet::new("state-provider", move |env: &mut FacetEnvironment| {
                let state = env.replicated_state(json!({ "value": value }))?;
                sources.lock().expect("sources lock").push(state.clone());
                let implementation = state_impl("state", state.internals());
                env.provide_remote(WATCHED.id(), ServiceMode::Singleton, implementation)?;
                Ok(())
            })
        };

        let host = create_facet_host(options(vec![consumer, provider(1, sources.clone())]))
            .await
            .expect("host activates");
        let retained_requirement = watched
            .lock()
            .expect("watched lock")
            .clone()
            .expect("watched");
        let retained_value = || async {
            let client = retained_requirement.client().expect("client");
            let state = client.state("state").expect("state").expect("state member");
            state.value().cloned().expect("hydrated value")
        };
        assert_eq!(retained_value().await, json!({ "value": 1 }));
        assert_eq!(*revisions.lock().expect("revisions lock"), [1]);

        host.reload(vec![provider(2, sources.clone())])
            .await
            .expect("reload");
        assert_eq!(retained_value().await, json!({ "value": 2 }));
        assert_eq!(*revisions.lock().expect("revisions lock"), [1, 2]);

        let retired_source = sources.lock().expect("sources lock")[0].clone();
        retired_source
            .replace(&BACKGROUND_CONTEXT, json!({ "value": 3 }))
            .expect("retired change");
        assert_eq!(retained_value().await, json!({ "value": 2 }));
        assert_eq!(*revisions.lock().expect("revisions lock"), [1, 2]);

        host.dispose().await.expect("dispose");
        let error = client_error(retained_requirement.client());
        assert_eq!(
            error.to_string(),
            "Facet state-consumer service handles cannot be used while dead"
        );
    }

    #[tokio::test]
    async fn scopes_singleton_service_views_to_each_facet_lifecycle() {
        let consumer_handles: Arc<Mutex<Vec<RemoteRequirement>>> = Arc::new(Mutex::new(Vec::new()));
        let cleanup_values: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let peer_handle: Arc<Mutex<Option<RemoteRequirement>>> = Arc::new(Mutex::new(None));

        let consumer = |generation: &'static str| {
            let consumer_handles = consumer_handles.clone();
            let cleanup_values = cleanup_values.clone();
            Facet::new("scoped-consumer", move |env: &mut FacetEnvironment| {
                let source = env.use_remote(SOURCE.id())?;
                let again = env.use_remote(SOURCE.id())?;
                // Upstream: `env.use(Source)` is the same view per facet.
                assert_eq!(source.identity, again.identity);
                consumer_handles
                    .lock()
                    .expect("consumer handles lock")
                    .push(source.clone());
                assert_eq!(
                    client_error(source.client()).to_string(),
                    "Facet scoped-consumer service handles cannot be used while setting_up"
                );
                let cleanup_source = source.clone();
                let cleanup_values = cleanup_values.clone();
                env.on_deactivate(Arc::new(move || {
                    let cleanup_source = cleanup_source.clone();
                    let cleanup_values = cleanup_values.clone();
                    Box::pin(async move {
                        let client = cleanup_source.client()?;
                        let value = client
                            .invoke("read", Vec::new(), &BACKGROUND_CONTEXT)
                            .await?
                            .unwrap_or(json!(null));
                        cleanup_values
                            .lock()
                            .expect("cleanup values lock")
                            .push(format!(
                                "{generation}:{}",
                                value.as_str().unwrap_or_default()
                            ));
                        Ok(())
                    })
                }))?;
                Ok(())
            })
        };

        let peer_handle_for_setup = peer_handle.clone();
        let peer = Facet::new("peer-consumer", move |env: &mut FacetEnvironment| {
            let handle = env.use_remote(SOURCE.id())?;
            *peer_handle_for_setup.lock().expect("peer handle lock") = Some(handle);
            Ok(())
        });

        let provider = Facet::new("scoped-provider", |env: &mut FacetEnvironment| {
            env.provide_remote(SOURCE.id(), ServiceMode::Singleton, read_impl("value"))?;
            Ok(())
        });

        let host = create_facet_host(options(vec![consumer("A"), peer, provider]))
            .await
            .expect("host activates");
        let handles = consumer_handles
            .lock()
            .expect("consumer handles lock")
            .clone();
        let peer_requirement = peer_handle
            .lock()
            .expect("peer handle lock")
            .clone()
            .unwrap();
        // Upstream: views are distinct objects across facets.
        assert_ne!(handles[0].identity, peer_requirement.identity);

        host.reload(vec![consumer("B")]).await.expect("reload");
        assert_eq!(
            *cleanup_values.lock().expect("cleanup values lock"),
            ["A:value"]
        );
        let new_handles = consumer_handles
            .lock()
            .expect("consumer handles lock")
            .clone();
        assert_ne!(new_handles[1].identity, handles[0].identity);
        assert_eq!(
            client_error(handles[0].client()).to_string(),
            "Facet scoped-consumer service handles cannot be used while dead"
        );
        let new_client = new_handles[1].client().expect("new client");
        let value = new_client
            .invoke("read", Vec::new(), &BACKGROUND_CONTEXT)
            .await
            .expect("new read");
        assert_eq!(value, Some(json!("value")));
        let peer_client = peer_requirement.client().expect("peer client");
        let value = peer_client
            .invoke("read", Vec::new(), &BACKGROUND_CONTEXT)
            .await
            .expect("peer read");
        assert_eq!(value, Some(json!("value")));

        host.dispose().await.expect("dispose");
        assert_eq!(
            *cleanup_values.lock().expect("cleanup values lock"),
            ["A:value", "B:value"]
        );
        assert_eq!(
            client_error(new_handles[1].client()).to_string(),
            "Facet scoped-consumer service handles cannot be used while dead"
        );
        assert_eq!(
            client_error(peer_requirement.client()).to_string(),
            "Facet peer-consumer service handles cannot be used while dead"
        );
    }

    #[tokio::test]
    async fn owns_resources_registered_during_activation() {
        let deliveries: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
        let watched: Arc<Mutex<Option<RemoteRequirement>>> = Arc::new(Mutex::new(None));

        let watched_for_consumer = watched.clone();
        let deliveries_for_consumer = deliveries.clone();
        let consumer = Facet::new("consumer", move |env: &mut FacetEnvironment| {
            let requirement = env.use_remote(WATCHED.id())?;
            *watched_for_consumer.lock().expect("watched lock") = Some(requirement.clone());
            let deliveries = deliveries_for_consumer.clone();
            let watched = requirement.clone();
            let environment = env.clone();
            env.on_activate(Arc::new(move || {
                let watched = watched.clone();
                let deliveries = deliveries.clone();
                let environment = environment.clone();
                Box::pin(async move {
                    let client = watched.client()?;
                    let state = client.state("state")?.expect("watched state member");
                    let listener: StateListener = Arc::new(move |_value, _context, _delivery| {
                        let deliveries = deliveries.clone();
                        let _ = _value;
                        Box::pin(async move {
                            *deliveries.lock().expect("deliveries lock") += 1;
                        })
                    });
                    let unsubscribe = state.subscribe(listener);
                    environment.own(disposal_from_unsubscribe(unsubscribe))?;
                    Ok(())
                })
            }))?;
            Ok(())
        });

        let state_cell: Arc<Mutex<Option<Arc<MutableReplicatedState>>>> =
            Arc::new(Mutex::new(None));
        let state_cell_for_provider = state_cell.clone();
        let provider = Facet::new("provider", move |env: &mut FacetEnvironment| {
            let state = env.replicated_state(json!({ "value": 0 }))?;
            *state_cell_for_provider.lock().expect("state cell lock") = Some(state.clone());
            env.provide_remote(
                WATCHED.id(),
                ServiceMode::Singleton,
                state_impl("state", state.internals()),
            )?;
            Ok(())
        });

        let host = create_facet_host(options(vec![consumer, provider]))
            .await
            .expect("host activates");
        assert_eq!(*deliveries.lock().expect("deliveries lock"), 1);
        let state = state_cell.lock().expect("state cell lock").clone().unwrap();
        state
            .replace(&BACKGROUND_CONTEXT, json!({ "value": 1 }))
            .expect("replace");
        assert_eq!(*deliveries.lock().expect("deliveries lock"), 2);

        host.dispose().await.expect("dispose");
        state
            .replace(&BACKGROUND_CONTEXT, json!({ "value": 2 }))
            .expect("replace after dispose");
        assert_eq!(*deliveries.lock().expect("deliveries lock"), 2);
    }

    #[tokio::test]
    async fn provides_arbitrary_host_services_through_the_facet_graph() {
        let consumer = Facet::new("host-service-consumer", |env: &mut FacetEnvironment| {
            let host_values = env.use_local(&*HOST_VALUES)?;
            assert_eq!(
                host_values
                    .with(|values: &HostValues| values.name.clone())
                    .unwrap_err()
                    .to_string(),
                "Facet host-service-consumer service handles cannot be used while setting_up"
            );
            let activation_values = host_values.clone();
            env.on_activate(Arc::new(move || {
                let activation_values = activation_values.clone();
                Box::pin(async move {
                    let name = activation_values
                        .with(|values: &HostValues| values.name.clone())
                        .expect("name after activation");
                    let use_ = activation_values
                        .with(|values: &HostValues| values.use_.clone())
                        .expect("use after activation");
                    assert_eq!(name, "session");
                    assert_eq!(use_, "host value");
                    Ok(())
                })
            }))?;
            Ok(())
        });
        let provider = Facet::new("host-service-provider", |env: &mut FacetEnvironment| {
            env.provide_local(
                &*HOST_VALUES,
                Arc::new(HostValues {
                    name: "session".to_string(),
                    use_: "host value".to_string(),
                }),
            )?;
            Ok(())
        });
        let host = create_facet_host(options(vec![consumer, provider]))
            .await
            .expect("host activates");
        // Upstream asserts "Service {id} is process-local"; the Rust provider only
        // sees IDs, so the local rejection surfaces as its no-local-provider error.
        assert!(host.services().use_singleton(HOST_VALUES.id()).is_err());
        host.dispose().await.expect("dispose");
    }

    #[tokio::test]
    async fn combines_connected_services_and_facet_provided_services_in_one_host() {
        let left_provider = Arc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition {
                service_id: LEFT_VALUE.id().to_string(),
                mode: ServiceMode::Singleton,
            }])
            .expect("left provider"),
        );
        left_provider
            .provide(LEFT_VALUE.id(), read_impl("left"))
            .expect("left provide");
        let right_provider = Arc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition {
                service_id: RIGHT_VALUE.id().to_string(),
                mode: ServiceMode::Singleton,
            }])
            .expect("right provider"),
        );
        right_provider
            .provide(RIGHT_VALUE.id(), read_impl("right"))
            .expect("right provide");

        let left_namespace = create_remote_service_binding(RemoteServiceBindingOptions {
            services: vec![LEFT_VALUE.id().to_string()],
            transport: create_loopback_service_transport(left_provider.clone()),
            bound: false,
            on_error: None,
            assert_access: None,
        })
        .expect("left binding");
        let right_namespace = create_remote_service_binding(RemoteServiceBindingOptions {
            services: vec![RIGHT_VALUE.id().to_string()],
            transport: create_loopback_service_transport(right_provider.clone()),
            bound: false,
            on_error: None,
            assert_access: None,
        })
        .expect("right binding");

        let left_source = Arc::new(TestServiceSource {
            accepts_unavailable: false,
            provider: left_provider.clone(),
            binding: left_namespace.clone(),
        });
        let right_source = Arc::new(TestServiceSource {
            accepts_unavailable: false,
            provider: right_provider.clone(),
            binding: right_namespace.clone(),
        });

        let facet = Facet::new("combined", |env: &mut FacetEnvironment| {
            let left = env.use_remote(LEFT_VALUE.id())?;
            let right = env.use_remote(RIGHT_VALUE.id())?;
            let implementation = method_object(vec![(
                "read",
                Arc::new(move |_args: Vec<JsonValue>, context: Context| {
                    let left = left.clone();
                    let right = right.clone();
                    Box::pin(async move {
                        let client = left.client()?;
                        let left_value = client
                            .invoke("read", Vec::new(), &context)
                            .await?
                            .unwrap_or(json!(null));
                        let client = right.client()?;
                        let right_value = client
                            .invoke("read", Vec::new(), &context)
                            .await?
                            .unwrap_or(json!(null));
                        Ok(Some(json!(format!(
                            "{} {}",
                            left_value.as_str().unwrap_or_default(),
                            right_value.as_str().unwrap_or_default()
                        ))))
                    }) as BoxFuture<_>
                }),
            )]);
            env.provide_remote(COMBINED_VALUE.id(), ServiceMode::Singleton, implementation)?;
            Ok(())
        });

        let host = create_facet_host(FacetOptions {
            facets: vec![facet],
            service_sources: vec![left_source, right_source],
            on_error: None,
        })
        .await
        .expect("host activates");
        let call = ServiceCall {
            service_id: COMBINED_VALUE.id().to_string(),
            instance: None,
            member: "read".to_string(),
            args: Vec::new(),
        };
        let value = host
            .services()
            .invoke(&call, &BACKGROUND_CONTEXT)
            .await
            .expect("combined read");
        assert_eq!(value, Some(json!("left right")));

        host.dispose().await.expect("dispose");
        left_namespace
            .dispose(&BACKGROUND_CONTEXT)
            .await
            .expect("left dispose");
        right_namespace
            .dispose(&BACKGROUND_CONTEXT)
            .await
            .expect("right dispose");
        left_provider.dispose().expect("left provider dispose");
        right_provider.dispose().expect("right provider dispose");
    }

    #[tokio::test]
    async fn rejects_a_service_offered_by_multiple_sources() {
        let consumer = Facet::new("duplicate-consumer", |env: &mut FacetEnvironment| {
            env.use_remote(LEFT_VALUE.id())?;
            Ok(())
        });
        let duplicate = Arc::new(DuplicateSource);
        let error = expect_host_failure(
            create_facet_host(FacetOptions {
                facets: vec![consumer],
                service_sources: vec![duplicate.clone(), duplicate],
                on_error: None,
            })
            .await,
        )
        .await;
        assert!(
            error.to_string().contains(&format!(
                "Facet host service {} is offered by more than one source",
                LEFT_VALUE.id()
            )),
            "{error}"
        );
    }

    #[tokio::test]
    async fn reopens_source_bindings_from_changed_catalogues_for_a_replacement_generation() {
        let left_provider = Arc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition {
                service_id: LEFT_VALUE.id().to_string(),
                mode: ServiceMode::Singleton,
            }])
            .expect("left provider"),
        );
        left_provider
            .provide(LEFT_VALUE.id(), read_impl("left"))
            .expect("left provide");
        let right_provider = Arc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition {
                service_id: RIGHT_VALUE.id().to_string(),
                mode: ServiceMode::Singleton,
            }])
            .expect("right provider"),
        );
        right_provider
            .provide(RIGHT_VALUE.id(), read_impl("right"))
            .expect("right provide");

        let state = Arc::new(SourceState {
            current: Mutex::new(left_provider.clone()),
            opened: AtomicU64::new(0),
            disposed: Arc::new(AtomicU64::new(0)),
        });
        let source = Arc::new(SwitchingSource {
            state: state.clone(),
        });

        let values: Arc<Mutex<Vec<JsonValue>>> = Arc::new(Mutex::new(Vec::new()));
        let left_consumer = {
            let values = values.clone();
            Facet::new("left-consumer", move |env: &mut FacetEnvironment| {
                let left = env.use_remote(LEFT_VALUE.id())?;
                let values = values.clone();
                env.on_activate(Arc::new(move || {
                    let left = left.clone();
                    let values = values.clone();
                    Box::pin(async move {
                        let client = left.client()?;
                        let value = client
                            .invoke("read", Vec::new(), &BACKGROUND_CONTEXT)
                            .await?
                            .unwrap_or(json!(null));
                        values.lock().expect("values lock").push(value);
                        Ok(())
                    })
                }))?;
                Ok(())
            })
        };
        let first = create_facet_host(FacetOptions {
            facets: vec![left_consumer],
            service_sources: vec![source.clone()],
            on_error: None,
        })
        .await
        .expect("first host");
        first.dispose().await.expect("first dispose");

        *state.current.lock().expect("current provider lock") = right_provider.clone();
        let right_consumer = {
            let values = values.clone();
            Facet::new("right-consumer", move |env: &mut FacetEnvironment| {
                let right = env.use_remote(RIGHT_VALUE.id())?;
                let values = values.clone();
                env.on_activate(Arc::new(move || {
                    let right = right.clone();
                    let values = values.clone();
                    Box::pin(async move {
                        let client = right.client()?;
                        let value = client
                            .invoke("read", Vec::new(), &BACKGROUND_CONTEXT)
                            .await?
                            .unwrap_or(json!(null));
                        values.lock().expect("values lock").push(value);
                        Ok(())
                    })
                }))?;
                Ok(())
            })
        };
        let second = create_facet_host(FacetOptions {
            facets: vec![right_consumer],
            service_sources: vec![source],
            on_error: None,
        })
        .await
        .expect("second host");
        second.dispose().await.expect("second dispose");

        assert_eq!(
            *values.lock().expect("values lock"),
            [json!("left"), json!("right")]
        );
        assert_eq!(state.opened.load(Ordering::SeqCst), 2);
        assert_eq!(state.disposed.load(Ordering::SeqCst), 2);
        left_provider.dispose().expect("left provider dispose");
        right_provider.dispose().expect("right provider dispose");
    }

    #[tokio::test]
    async fn rejects_missing_dependencies_and_cycles() {
        let activated: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let activated_for_setup = activated.clone();
        let missing = Facet::new("missing", move |env: &mut FacetEnvironment| {
            env.use_remote(SOURCE.id())?;
            let activated = activated_for_setup.clone();
            env.on_activate(Arc::new(move || {
                activated.store(true, Ordering::SeqCst);
                Box::pin(async { Ok(()) }) as BoxFuture<Result<(), AnyError>>
            }))?;
            Ok(())
        });
        let error = expect_host_failure(create_facet_host(options(vec![missing])).await).await;
        assert_eq!(
            error.to_string(),
            format!(
                "Facet missing requires local/{}/singleton, but no facet provides it",
                SOURCE.id()
            )
        );
        assert!(!activated.load(Ordering::SeqCst));

        let first = Facet::new("first", |env: &mut FacetEnvironment| {
            env.use_remote(PROJECTION.id())?;
            env.provide_remote(SOURCE.id(), ServiceMode::Singleton, read_impl("first"))?;
            Ok(())
        });
        let second = Facet::new("second", |env: &mut FacetEnvironment| {
            env.use_remote(SOURCE.id())?;
            env.provide_remote(PROJECTION.id(), ServiceMode::Singleton, read_impl("second"))?;
            Ok(())
        });
        let error =
            expect_host_failure(create_facet_host(options(vec![first, second])).await).await;
        assert_eq!(error.to_string(), "Facet dependency cycle: first, second");
    }

    // ─── Test doubles ────────────────────────────────────────────────────────

    /// Remote service source backed by an in-process provider and consumer
    /// binding; `ready` rebinding mirrors upstream's custom `ready()` wrappers.
    struct TestServiceSource {
        accepts_unavailable: bool,
        provider: Arc<RemoteServiceProvider>,
        binding: Arc<RemoteServiceBinding>,
    }

    #[async_trait]
    impl RemoteServiceSource for TestServiceSource {
        fn accepts_unavailable_services(&self) -> bool {
            self.accepts_unavailable
        }

        async fn catalogue(
            &self,
            _context: &Context,
        ) -> Result<Vec<ServiceCatalogueEntry>, AnyError> {
            Ok(self.provider.catalogue().to_vec())
        }

        async fn open(
            &self,
            _options: RemoteServiceSourceOpenOptions,
        ) -> Result<Box<dyn RemoteServices>, AnyError> {
            Ok(Box::new(RebindingSourceServices {
                binding: self.binding.clone(),
            }))
        }
    }

    struct RebindingSourceServices {
        binding: Arc<RemoteServiceBinding>,
    }

    #[async_trait]
    impl RemoteServices for RebindingSourceServices {
        fn use_service(&self, service_id: &str) -> Result<Arc<dyn ServiceClientHandle>, AnyError> {
            self.binding.use_service(service_id)
        }

        fn observe(
            &self,
            service_id: &str,
            handler: KeyedObservationHandler,
        ) -> Result<Unsubscribe, AnyError> {
            self.binding.observe(service_id, handler)
        }

        async fn ready(&self, context: &Context) -> Result<(), AnyError> {
            self.binding.rebind(true, context).await?;
            self.binding.ready(context).await
        }

        async fn dispose(&self, context: &Context) -> Result<(), AnyError> {
            self.binding.dispose(context).await
        }
    }

    struct DuplicateSource;

    #[async_trait]
    impl RemoteServiceSource for DuplicateSource {
        fn accepts_unavailable_services(&self) -> bool {
            false
        }

        async fn catalogue(
            &self,
            _context: &Context,
        ) -> Result<Vec<ServiceCatalogueEntry>, AnyError> {
            Ok(vec![ServiceCatalogueEntry {
                service_id: LEFT_VALUE.id().to_string(),
                mode: ServiceMode::Singleton,
            }])
        }

        async fn open(
            &self,
            _options: RemoteServiceSourceOpenOptions,
        ) -> Result<Box<dyn RemoteServices>, AnyError> {
            Err(error("Ambiguous sources must not open"))
        }
    }

    struct SourceState {
        current: Mutex<Arc<RemoteServiceProvider>>,
        opened: AtomicU64,
        disposed: Arc<AtomicU64>,
    }

    struct SwitchingSource {
        state: Arc<SourceState>,
    }

    #[async_trait]
    impl RemoteServiceSource for SwitchingSource {
        fn accepts_unavailable_services(&self) -> bool {
            false
        }

        async fn catalogue(
            &self,
            _context: &Context,
        ) -> Result<Vec<ServiceCatalogueEntry>, AnyError> {
            Ok(self
                .state
                .current
                .lock()
                .expect("current provider lock")
                .catalogue()
                .to_vec())
        }

        async fn open(
            &self,
            options: RemoteServiceSourceOpenOptions,
        ) -> Result<Box<dyn RemoteServices>, AnyError> {
            self.state.opened.fetch_add(1, Ordering::SeqCst);
            let provider = self
                .state
                .current
                .lock()
                .expect("current provider lock")
                .clone();
            let binding = Arc::new(RemoteServiceBinding::new(RemoteServiceBindingOptions {
                services: options.services,
                transport: create_loopback_service_transport(provider),
                bound: false,
                on_error: Some(options.on_error),
                assert_access: Some(options.assert_access),
            })?);
            Ok(Box::new(CountingSourceServices {
                binding,
                disposed: self.state.disposed.clone(),
            }))
        }
    }

    struct CountingSourceServices {
        binding: Arc<RemoteServiceBinding>,
        disposed: Arc<AtomicU64>,
    }

    #[async_trait]
    impl RemoteServices for CountingSourceServices {
        fn use_service(&self, service_id: &str) -> Result<Arc<dyn ServiceClientHandle>, AnyError> {
            self.binding.use_service(service_id)
        }

        fn observe(
            &self,
            service_id: &str,
            handler: KeyedObservationHandler,
        ) -> Result<Unsubscribe, AnyError> {
            self.binding.observe(service_id, handler)
        }

        async fn ready(&self, context: &Context) -> Result<(), AnyError> {
            self.binding.rebind(true, context).await?;
            self.binding.ready(context).await
        }

        async fn dispose(&self, context: &Context) -> Result<(), AnyError> {
            self.disposed.fetch_add(1, Ordering::SeqCst);
            self.binding.dispose(context).await
        }
    }
}
