//! Port of `packages/chord/src/types.ts`.
//!
//! Deviations from upstream (recorded per PORT_MAP.md rule 3):
//! - TypeScript compile-time contract checking (`RemoteServiceContract<T>`,
//!   `JsonRepresentation<T>`, `Draft<T>` mapped types) has no Rust counterpart:
//!   remote payloads are [`JsonValue`], so JSON-compatibility is enforced by
//!   construction instead of by conditional types.
//! - `Service<T>` keeps its type parameter as a phantom marker for local service
//!   registries; remote members are invoked by name with JSON values, so member
//!   surfaces are described at runtime via [`RemoteServiceImpl`] rather than
//!   through object property reflection.
//! - Interfaces returning `void | Promise<void>` map to async functions returning
//!   `Result<(), AnyError>`; upstream throws are `Err` values.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use crate::context::Context;
use crate::delta::Op;

use crate::services::errors::AnyError;
use crate::services::state_internals::StateInternals;

pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
pub type LocalBoxFuture<T> = Pin<Box<dyn Future<Output = T>>>;

/// Disposer returned by subscription APIs.
pub type Unsubscribe = Box<dyn FnOnce() + Send>;

pub use crate::context::{AbortController, AbortReason, AbortSignal};

/// Canonical strict-JSON tree shared across the workspace.
pub type JsonValue = crate::json::JsonValue;

// ─── Service identity ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceMode {
    #[serde(rename = "singleton")]
    Singleton,
    #[serde(rename = "keyed")]
    Keyed,
}

impl ServiceMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ServiceMode::Singleton => "singleton",
            ServiceMode::Keyed => "keyed",
        }
    }
}

/// Stable identity for one shared service contract.
///
/// Process-local services accept unrestricted Rust contracts and are never
/// published remotely; the type parameter only participates in local registries.
pub struct Service<T: ?Sized> {
    id: Arc<str>,
    local: bool,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ?Sized> Service<T> {
    pub(crate) fn new(id: impl Into<Arc<str>>, local: bool) -> Service<T> {
        Service {
            id: id.into(),
            local,
            _marker: PhantomData,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// Process-local services are never published remotely.
    pub fn local(&self) -> bool {
        self.local
    }
}

impl<T: ?Sized> Clone for Service<T> {
    fn clone(&self) -> Self {
        Service {
            id: self.id.clone(),
            local: self.local,
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized> PartialEq for Service<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.local == other.local
    }
}

impl<T: ?Sized> Eq for Service<T> {}

impl<T: ?Sized> std::fmt::Debug for Service<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Service")
            .field("id", &self.id)
            .field("local", &self.local)
            .finish()
    }
}

// ─── Replicated state ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicatedStateDelivery {
    pub kind: ReplicatedStateDeliveryKind,
    pub sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicatedStateDeliveryKind {
    #[serde(rename = "hydrate")]
    Hydrate,
    #[serde(rename = "update")]
    Update,
}

/// Listener for replicated state deliveries. Returned futures serialize the
/// subscriber's deliveries; values are immutable and may share unchanged
/// revision data.
pub type StateListener =
    Arc<dyn Fn(&JsonValue, &Context, &ReplicatedStateDelivery) -> BoxFuture<()> + Send + Sync>;

/// Read side of one replicated state value. `value` is `None` until hydration on
/// cold replicas; authoritative states always carry a value.
pub trait ReplicatedState: Send + Sync {
    fn value(&self) -> Option<&JsonValue>;
    fn subscribe(&self, listener: StateListener) -> Unsubscribe;
}

/// One immutable authoritative revision committed after an attachment snapshot.
#[derive(Clone)]
pub struct ReplicatedStateSourceFrame {
    /// Monotonic source cursor. The first frame after a snapshot must be
    /// `snapshot.cursor + 1`.
    pub cursor: u64,
    /// The exact immutable value produced by this commit.
    pub value: JsonValue,
    /// The exact immutable operation batch that produced `value` from the
    /// preceding source revision.
    pub ops: Vec<Op>,
    pub context: Context,
}

/// The fixed immutable snapshot captured at the atomic attachment boundary.
#[derive(Clone)]
pub struct ReplicatedStateSourceSnapshot {
    pub value: JsonValue,
    pub cursor: u64,
}

/// Listener installed by `activate`; receives every buffered and subsequent frame
/// in source commit order.
pub type SourceFrameListener = Arc<dyn Fn(ReplicatedStateSourceFrame) + Send + Sync>;

/// The attachment returned by [`ReplicatedStateSource::attach`].
pub trait ReplicatedStateSourceAttachment: Send + Sync {
    fn snapshot(&self) -> ReplicatedStateSourceSnapshot;
    /// Install the sole listener and synchronously drain every buffered frame.
    /// Single-use.
    fn activate(&self, listener: SourceFrameListener);
    /// Stop delivery and release source resources. Idempotent.
    fn dispose(&self);
}

/// An authoritative immutable revision source.
pub trait ReplicatedStateSource: Send + Sync {
    /// Synchronously and atomically capture one snapshot and register the
    /// returned attachment to buffer every later committed frame.
    fn attach(self: Arc<Self>) -> Box<dyn ReplicatedStateSourceAttachment>;
}

pub struct ReplicatedStateSourceOptions {
    /// Receives source-contract and publication-listener failures without
    /// throwing them into the source.
    pub on_error: Option<arc_error::ErrorHandler>,
}

pub mod arc_error {
    use super::AnyError;
    use std::sync::Arc;
    pub type ErrorHandler = Arc<dyn Fn(AnyError) + Send + Sync>;
}

// ─── Service catalogue and snapshots (wire types) ────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceCatalogueEntry {
    #[serde(rename = "serviceId")]
    pub service_id: String,
    pub mode: ServiceMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceInstanceAddress {
    pub key: String,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ServiceMemberSnapshot {
    #[serde(rename = "method")]
    Method { name: String },
    #[serde(rename = "state")]
    State {
        name: String,
        sequence: u64,
        ops: Vec<Op>,
    },
}

impl ServiceMemberSnapshot {
    pub fn name(&self) -> &str {
        match self {
            ServiceMemberSnapshot::Method { name } => name,
            ServiceMemberSnapshot::State { name, .. } => name,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            ServiceMemberSnapshot::Method { .. } => "method",
            ServiceMemberSnapshot::State { .. } => "state",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceInstanceSnapshot {
    #[serde(rename = "instance", skip_serializing_if = "Option::is_none")]
    pub instance: Option<ServiceInstanceAddress>,
    pub members: Vec<ServiceMemberSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceSubscriptionSnapshot {
    #[serde(rename = "serviceId")]
    pub service_id: String,
    pub mode: ServiceMode,
    pub instances: Vec<ServiceInstanceSnapshot>,
}

/// One provider-side publication. `State` carries delta operations; `Reset` is a
/// full subscription rebaseline after overflow; every state in it is a root
/// replacement at its new sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServiceProviderUpdate {
    #[serde(rename = "state")]
    State {
        #[serde(rename = "instance", skip_serializing_if = "Option::is_none")]
        instance: Option<ServiceInstanceAddress>,
        member: String,
        sequence: u64,
        ops: Vec<Op>,
    },
    #[serde(rename = "reset")]
    Reset {
        snapshot: ServiceSubscriptionSnapshot,
    },
    #[serde(rename = "unavailable")]
    Unavailable,
    #[serde(rename = "replaced")]
    Replaced { snapshot: ServiceInstanceSnapshot },
    #[serde(rename = "spawned")]
    Spawned { instance: ServiceInstanceSnapshot },
    #[serde(rename = "closed")]
    Closed { instance: ServiceInstanceAddress },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceCall {
    #[serde(rename = "serviceId")]
    pub service_id: String,
    #[serde(rename = "instance", skip_serializing_if = "Option::is_none")]
    pub instance: Option<ServiceInstanceAddress>,
    pub member: String,
    /// Borrowed immutable values. Chord validates but does not clone them.
    pub args: Vec<JsonValue>,
}

// ─── Remote service implementation surface ───────────────────────────────────

/// An invocable remote method: JSON args in, JSON result (or `None` for void).
pub type RemoteMethod = Arc<
    dyn Fn(Vec<JsonValue>, Context) -> BoxFuture<Result<Option<JsonValue>, AnyError>> + Send + Sync,
>;

/// One member of a remote service implementation.
pub enum ServiceMember {
    Method(RemoteMethod),
    State(Arc<dyn StateInternals>),
}

/// Runtime description of a remote service implementation.
///
/// Upstream reflects over plain object properties; Rust implementations either
/// build a [`crate::services::provider::RemoteServiceObject`] or implement this
/// trait directly.
pub trait RemoteServiceImpl: Send + Sync {
    /// Member table in deterministic (sorted by name) order; must be non-empty.
    fn members(&self) -> Vec<(String, ServiceMember)>;

    /// The concrete object handed back to process-local consumers. Defaults to
    /// `None` (the implementation itself is not directly shareable).
    fn as_local_object(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        None
    }
}

/// Listener for provider updates pushed through a subscription.
pub type UpdateListener = Arc<dyn Fn(&ServiceProviderUpdate, &Context) + Send + Sync>;

/// One active subscription to a remote service.
#[async_trait]
pub trait ServiceSubscription: Send + Sync {
    /// Atomic baseline. Later updates buffer until activation, with a full reset
    /// on pending delivery 101.
    fn snapshot(&self) -> &ServiceSubscriptionSnapshot;
    fn activate(&self);
    async fn close(&self, context: Option<Context>);
}

/// Pluggable wire boundary consumed by a remote service binding.
///
/// Implementations choose transport, framing, routing, and envelope encoding.
/// Values crossing this boundary must remain strict JSON. Chord does not clone
/// values or require a particular application wire protocol; adapters own
/// serialization and any isolation copies they require.
#[async_trait]
pub trait RemoteServiceTransport: Send + Sync {
    async fn invoke(
        &self,
        call: &ServiceCall,
        context: &Context,
    ) -> Result<Option<JsonValue>, AnyError>;
    async fn subscribe(
        &self,
        service_id: &str,
        mode: ServiceMode,
        listener: UpdateListener,
        context: &Context,
    ) -> Result<Box<dyn ServiceSubscription>, AnyError>;
}

#[derive(Clone)]
pub struct RemoteServiceBindingOptions {
    pub services: Vec<String>,
    pub transport: Arc<dyn RemoteServiceTransport>,
    pub bound: bool,
    pub on_error: Option<arc_error::ErrorHandler>,
    pub assert_access: Option<Arc<dyn Fn() -> Result<(), AnyError> + Send + Sync>>,
}

/// Consumer-side view of one remote service. Methods are invoked by name with
/// JSON arguments; state members expose [`ReplicatedState`].
#[async_trait]
pub trait ServiceClientHandle: Send + Sync {
    fn service_id(&self) -> &str;
    fn invoke(
        &self,
        member: &str,
        args: Vec<JsonValue>,
        context: &Context,
    ) -> BoxFuture<Result<Option<JsonValue>, AnyError>>;
    /// Read-only view over one replicated state member.
    fn state(&self, member: &str) -> Result<Option<Arc<dyn ReplicatedState>>, AnyError>;
}

/// The remote-services facade consumed by facets and bindings.
#[async_trait]
pub trait RemoteServices: Send + Sync {
    /// Acquire one singleton service handle.
    fn use_service(&self, service_id: &str) -> Result<Arc<dyn ServiceClientHandle>, AnyError>;
    /// Observe each live instance of a keyed service. Returns a disposer.
    fn observe(
        &self,
        service_id: &str,
        handler: KeyedObservationHandler,
    ) -> Result<Unsubscribe, AnyError>;
    /// Wait until every currently acquired service has installed its initial
    /// snapshot.
    async fn ready(&self, context: &Context) -> Result<(), AnyError>;
    async fn dispose(&self, context: &Context) -> Result<(), AnyError>;
}

/// Handler invoked for each live instance of a keyed service.
pub type KeyedObservationHandler =
    Arc<dyn Fn(Arc<dyn ServiceClientHandle>, Context) -> BoxFuture<()> + Send + Sync>;

// ─── Facets ──────────────────────────────────────────────────────────────────

/// One facet: a named unit of composition with a synchronous `setup`.
#[derive(Clone)]
pub struct Facet {
    id: Arc<str>,
    setup: Arc<
        dyn Fn(&mut crate::facets::host::FacetEnvironment) -> Result<(), AnyError> + Send + Sync,
    >,
}

impl Facet {
    pub fn new(
        id: impl Into<String>,
        setup: impl Fn(&mut crate::facets::host::FacetEnvironment) -> Result<(), AnyError>
            + Send
            + Sync
            + 'static,
    ) -> Facet {
        Facet {
            id: Arc::from(id.into().as_str()),
            setup: Arc::new(setup),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub(crate) fn setup(
        &self,
        env: &mut crate::facets::host::FacetEnvironment,
    ) -> Result<(), AnyError> {
        (self.setup)(env)
    }
}

impl std::fmt::Debug for Facet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Facet").field("id", &self.id).finish()
    }
}

/// An authoritative remote service source offered to the facet host.
#[async_trait]
pub trait RemoteServiceSource: Send + Sync {
    /// Whether this currently unavailable source may provisionally own absent
    /// requirements.
    fn accepts_unavailable_services(&self) -> bool;

    async fn catalogue(&self, context: &Context) -> Result<Vec<ServiceCatalogueEntry>, AnyError>;

    /// Open a binding for the given service ids.
    async fn open(
        &self,
        options: RemoteServiceSourceOpenOptions,
    ) -> Result<Box<dyn RemoteServices>, AnyError>;
}

pub struct RemoteServiceSourceOpenOptions {
    pub services: Vec<String>,
    pub assert_access: Arc<dyn Fn() -> Result<(), AnyError> + Send + Sync>,
    pub on_error: arc_error::ErrorHandler,
}

pub struct FacetOptions {
    pub facets: Vec<Facet>,
    pub service_sources: Vec<Arc<dyn RemoteServiceSource>>,
    pub on_error: Option<arc_error::ErrorHandler>,
}

/// An active host for one complete set of facets.
#[async_trait]
pub trait FacetHost: Send + Sync {
    fn services(&self) -> &crate::services::provider::RemoteServiceProvider;
    /// Shared handle to the assembled provider (upstream exposes the provider
    /// object itself; Rust callers need an `Arc` to build endpoints). Returns
    /// `None` when the host implementation cannot share it.
    fn provider_arc(&self) -> Option<Arc<crate::services::provider::RemoteServiceProvider>> {
        None
    }
    /// Activate and replace facets with matching IDs without disconnecting
    /// consumer service handles.
    async fn reload(&self, facets: Vec<Facet>) -> Result<(), AnyError>;
    async fn dispose(&self) -> Result<(), AnyError>;
}

/// One complete set of facets produced by a [`FacetLoader`].
pub struct LoadedFacets {
    pub facets: Vec<Facet>,
    pub dispose: Option<Box<dyn FnOnce() -> BoxFuture<Result<(), AnyError>> + Send>>,
}

impl LoadedFacets {
    pub fn new(facets: Vec<Facet>) -> LoadedFacets {
        LoadedFacets {
            facets,
            dispose: None,
        }
    }

    pub async fn dispose(&mut self) -> Result<(), AnyError> {
        match self.dispose.take() {
            Some(dispose) => dispose().await,
            None => Ok(()),
        }
    }
}

pub trait FacetLoader: Send + Sync {
    fn load(&self) -> BoxFuture<Result<LoadedFacets, AnyError>>;
}
