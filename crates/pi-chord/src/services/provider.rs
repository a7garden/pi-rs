//! Port of `packages/chord/src/services/provider.ts`.
//!
//! Deviations from upstream (recorded per PORT_MAP.md rule 3):
//! - `classifyRemoteServiceImplementation` reflected over object properties; the
//!   `TypeError`s "implementation must be an object", "must be a data property" and
//!   "is not remotely exposable" are unportable: Rust members arrive through
//!   [`RemoteServiceImpl::members`] with explicit kinds. Classification validates
//!   member presence ("has no members") and normalizes names to sorted-unique order.
//! - `ServiceProviderDefinition` carries no `local` flag, so the constructor
//!   rejection "Local service {id} cannot be published remotely" (and the
//!   provider-side `assertRemotable`) is unreachable and not ported.
//! - Subscriber listeners are infallible `UpdateListener`s; upstream collected
//!   synchronous listener throws. The Rust port surfaces listener failures as
//!   caught panics and aggregates them with the upstream messages.
//! - `ServiceSubscription::activate` is infallible in the trait signature, so the
//!   "Failed to activate remote service subscription" aggregate is logged instead
//!   of thrown.

use std::any::Any;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;

use crate::context::Context;
use crate::delta::Op;
use crate::json::JsonValue;
use crate::services::errors::{
    error, remote_service_error, type_error, AggregateError, AnyError, RemoteServiceErrorCode,
};
use crate::services::state::service_delivery_context;
use crate::services::state_internals::{SourceListener, StateInternals};
use crate::services::wire::{decode_service_control_call, ServiceControlCall};
use crate::types::{
    BoxFuture, RemoteServiceImpl, RemoteMethod, ServiceProviderUpdate, ServiceCall,
    ServiceCatalogueEntry, ServiceInstanceAddress, ServiceInstanceSnapshot, ServiceMember,
    ServiceMemberSnapshot, ServiceMode, ServiceSubscription, ServiceSubscriptionSnapshot,
    UpdateListener, Unsubscribe,
};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Publishes subscription payloads out of a remote service endpoint
/// (upstream `ServiceUpdatePublisher`).
pub type ServiceUpdatePublisher =
    Arc<dyn Fn(String, &ServiceProviderUpdate, &Context) -> BoxFuture<()> + Send + Sync>;

/// One catalogue definition handed to [`RemoteServiceProvider::new`]
/// (upstream `ServiceProviderDefinition`; there is no `local` flag).
#[derive(Debug, Clone)]
pub struct ProviderDefinition {
    pub service_id: String,
    pub mode: ServiceMode,
}

impl ProviderDefinition {
    pub fn new(service_id: impl Into<String>, mode: ServiceMode) -> ProviderDefinition {
        ProviderDefinition {
            service_id: service_id.into(),
            mode,
        }
    }
}

enum InstanceMember {
    Method(RemoteMethod),
    State(Arc<dyn StateInternals>),
}

struct ProviderInstance {
    address: Option<ServiceInstanceAddress>,
    /// Kept for parity with upstream `ProviderInstance.implementation`; the
    /// invocable surface lives in `members`.
    #[allow(dead_code)]
    implementation: Arc<dyn RemoteServiceImpl>,
    members: Vec<(String, InstanceMember)>,
    remove_member_listeners: Vec<Unsubscribe>,
    active: Arc<AtomicBool>,
}

struct SubscriberInner {
    buffer: Vec<(ServiceProviderUpdate, Context)>,
    snapshot_sequences: HashMap<String, u64>,
    active: bool,
    draining: bool,
    terminated: bool,
    closed: bool,
}

struct ProviderSubscriberState {
    listener: UpdateListener,
    inner: Mutex<SubscriberInner>,
}

struct ServiceRegistration {
    service_id: String,
    mode: ServiceMode,
    singleton: Option<ProviderInstance>,
    singleton_shape: Option<BTreeMap<String, &'static str>>,
    instances: HashMap<String, ProviderInstance>,
    generations: HashMap<String, u64>,
    subscribers: Vec<Arc<ProviderSubscriberState>>,
}

struct ProviderInner {
    registrations: HashMap<String, ServiceRegistration>,
    disposed: bool,
}

/// Hosts one provider for one remote consumer and owns that consumer's subscriptions.
pub struct RemoteServiceProvider {
    catalogue: Vec<ServiceCatalogueEntry>,
    inner: Arc<Mutex<ProviderInner>>,
}

impl RemoteServiceProvider {
    pub fn new(entries: Vec<ProviderDefinition>) -> Result<RemoteServiceProvider, AnyError> {
        let mut seen = HashSet::new();
        for entry in &entries {
            if !seen.insert(entry.service_id.as_str()) {
                return Err(type_error("Remote service catalogue contains duplicate IDs"));
            }
        }
        let catalogue = entries
            .iter()
            .map(|entry| ServiceCatalogueEntry {
                service_id: entry.service_id.clone(),
                mode: entry.mode,
            })
            .collect();
        let registrations = entries
            .into_iter()
            .map(|entry| {
                (
                    entry.service_id.clone(),
                    ServiceRegistration {
                        service_id: entry.service_id,
                        mode: entry.mode,
                        singleton: None,
                        singleton_shape: None,
                        instances: HashMap::new(),
                        generations: HashMap::new(),
                        subscribers: Vec::new(),
                    },
                )
            })
            .collect();
        Ok(RemoteServiceProvider {
            catalogue,
            inner: Arc::new(Mutex::new(ProviderInner {
                registrations,
                disposed: false,
            })),
        })
    }

    pub fn catalogue(&self) -> &[ServiceCatalogueEntry] {
        &self.catalogue
    }

    fn assert_active(&self) -> Result<(), AnyError> {
        if lock(&self.inner).disposed {
            return Err(error("Remote service provider is disposed"));
        }
        Ok(())
    }

    fn assert_allowed(&self, service_id: &str) -> Result<(), AnyError> {
        if !lock(&self.inner).registrations.contains_key(service_id) {
            return Err(remote_service_error(
                RemoteServiceErrorCode::ServiceNotAllowed,
                format!("Remote service {service_id} is not allowlisted"),
            ));
        }
        Ok(())
    }

    pub fn provide(
        &self,
        service_id: &str,
        implementation: Arc<dyn RemoteServiceImpl>,
    ) -> Result<(), AnyError> {
        self.assert_active()?;
        self.assert_allowed(service_id)?;
        let mut inner = lock(&self.inner);
        let registration = registration_mut(&mut inner, service_id, ServiceMode::Singleton)?;
        if registration.singleton.is_some() {
            return Err(remote_service_error(
                RemoteServiceErrorCode::ServiceModeMismatch,
                format!("Remote service {service_id} already has a provider"),
            ));
        }
        let members = classify(service_id, &implementation)?;
        let shape = service_member_shape(&members);
        assert_singleton_shape(registration, service_id, &shape)?;
        let instance = create_instance(&Arc::clone(&self.inner), registration, members, implementation, None);
        registration.singleton_shape = Some(shape);
        registration.singleton = Some(instance);
        Ok(())
    }

    /// Disconnect one singleton while preserving active subscriptions and remote facades.
    pub fn withdraw(&self, service_id: &str) -> Result<(), AnyError> {
        self.assert_active()?;
        self.assert_allowed(service_id)?;
        {
            let mut inner = lock(&self.inner);
            let registration = registration_mut(&mut inner, service_id, ServiceMode::Singleton)?;
            let Some(previous) = registration.singleton.take() else {
                return Ok(());
            };
            previous.active.store(false, Ordering::SeqCst);
            for remove in previous.remove_member_listeners {
                remove();
            }
        }
        emit(&self.inner, service_id, ServiceProviderUpdate::Unavailable, None)
    }

    /// Check a singleton replacement without changing the active provider.
    pub fn validate_replacement(
        &self,
        service_id: &str,
        implementation: Arc<dyn RemoteServiceImpl>,
    ) -> Result<(), AnyError> {
        self.assert_active()?;
        self.assert_allowed(service_id)?;
        let mut inner = lock(&self.inner);
        let registration = registration_mut(&mut inner, service_id, ServiceMode::Singleton)?;
        let members = classify(service_id, &implementation)?;
        let shape = service_member_shape(&members);
        assert_singleton_shape(registration, service_id, &shape)
    }

    /// Replace one singleton without making its stable remote facade unavailable.
    pub fn replace(
        &self,
        service_id: &str,
        implementation: Arc<dyn RemoteServiceImpl>,
    ) -> Result<(), AnyError> {
        self.assert_active()?;
        self.assert_allowed(service_id)?;
        let snapshot = {
            let mut inner = lock(&self.inner);
            let registration = registration_mut(&mut inner, service_id, ServiceMode::Singleton)?;
            let members = classify(service_id, &implementation)?;
            let shape = service_member_shape(&members);
            assert_singleton_shape(registration, service_id, &shape)?;
            let replacement =
                create_instance(&Arc::clone(&self.inner), registration, members, implementation, None);
            if let Some(previous) = registration.singleton.take() {
                previous.active.store(false, Ordering::SeqCst);
                for remove in previous.remove_member_listeners {
                    remove();
                }
            }
            registration.singleton_shape = Some(shape);
            let snapshot = snapshot_instance(&replacement);
            registration.singleton = Some(replacement);
            snapshot
        };
        emit(
            &self.inner,
            service_id,
            ServiceProviderUpdate::Replaced { snapshot },
            None,
        )
    }

    /// Hand back the process-local object registered for one singleton service.
    pub fn use_singleton(&self, service_id: &str) -> Result<Arc<dyn Any + Send + Sync>, AnyError> {
        self.assert_active()?;
        self.assert_allowed(service_id)?;
        let inner = lock(&self.inner);
        let object = inner
            .registrations
            .get(service_id)
            .filter(|registration| registration.mode == ServiceMode::Singleton)
            .and_then(|registration| registration.singleton.as_ref())
            .and_then(|singleton| singleton.implementation.as_local_object());
        match object {
            Some(object) => Ok(object),
            None => Err(remote_service_error(
                RemoteServiceErrorCode::ServiceNotFound,
                format!("Remote service {service_id} has no local provider"),
            )),
        }
    }

    pub fn spawn(
        &self,
        service_id: &str,
        key: &str,
        implementation: Arc<dyn RemoteServiceImpl>,
    ) -> Result<Unsubscribe, AnyError> {
        self.assert_active()?;
        self.assert_allowed(service_id)?;
        if key.is_empty() {
            return Err(type_error("Remote service instance key must not be empty"));
        }
        let inner_arc = Arc::clone(&self.inner);
        let (snapshot, active_flag) = {
            let mut inner = lock(&inner_arc);
            let registration = registration_mut(&mut inner, service_id, ServiceMode::Keyed)?;
            if registration.instances.contains_key(key) {
                return Err(remote_service_error(
                    RemoteServiceErrorCode::ServiceModeMismatch,
                    format!("Remote service {service_id} already has a live instance with key {key}"),
                ));
            }
            let generation = registration.generations.get(key).copied().unwrap_or(0) + 1;
            registration.generations.insert(key.to_string(), generation);
            let address = ServiceInstanceAddress {
                key: key.to_string(),
                generation,
            };
            let members = classify(service_id, &implementation)?;
            let instance =
                create_instance(&inner_arc, registration, members, implementation, Some(address.clone()));
            let snapshot = snapshot_instance(&instance);
            let active_flag = Arc::clone(&instance.active);
            registration.instances.insert(key.to_string(), instance);
            (snapshot, active_flag)
        };
        emit(
            &inner_arc,
            service_id,
            ServiceProviderUpdate::Spawned { instance: snapshot },
            None,
        )?;
        // `Unsubscribe` is `FnOnce`, so a re-entry guard is unnecessary (upstream's
        // `closed` flag exists because JS closures can fire repeatedly).
        let service_id = service_id.to_string();
        let key = key.to_string();
        Ok(Box::new(move || {
            let mut inner = lock(&inner_arc);
            let Some(registration) = inner.registrations.get_mut(&service_id) else {
                return;
            };
            let is_current = registration
                .instances
                .get(&key)
                .is_some_and(|current| Arc::ptr_eq(&current.active, &active_flag));
            if !is_current {
                return;
            }
            let instance = registration.instances.remove(&key).expect("checked above");
            instance.active.store(false, Ordering::SeqCst);
            for remove in instance.remove_member_listeners {
                remove();
            }
            let instance_address = instance.address.expect("keyed instances have addresses");
            drop(inner);
            // Upstream close() propagates emission failures to the caller; the
            // `Unsubscribe` signature is infallible, so they are dropped here.
            let _ = emit(
                &inner_arc,
                &service_id,
                ServiceProviderUpdate::Closed {
                    instance: instance_address,
                },
                None,
            );
        }))
    }

    pub async fn invoke(
        &self,
        call: &ServiceCall,
        context: &Context,
    ) -> Result<Option<JsonValue>, AnyError> {
        self.assert_active()?;
        self.assert_allowed(&call.service_id)?;
        let method = {
            let inner = lock(&self.inner);
            let Some(registration) = inner.registrations.get(&call.service_id) else {
                return Err(remote_service_error(
                    RemoteServiceErrorCode::ServiceNotFound,
                    format!("Unknown remote service {}", call.service_id),
                ));
            };
            let instance = resolve_instance(registration, call.instance.as_ref())?;
            let member = instance
                .members
                .iter()
                .find(|(name, _)| name == &call.member)
                .map(|(_, member)| member);
            match member {
                None => {
                    return Err(remote_service_error(
                        RemoteServiceErrorCode::ServiceMemberNotFound,
                        format!("Unknown remote service member {}.{}", call.service_id, call.member),
                    ))
                }
                Some(InstanceMember::Method(method)) => Arc::clone(method),
                Some(InstanceMember::State(_)) => {
                    return Err(remote_service_error(
                        RemoteServiceErrorCode::ServiceMemberMismatch,
                        format!(
                            "Remote service member {}.{} is not a method",
                            call.service_id, call.member
                        ),
                    ))
                }
            }
        };
        method(call.args.clone(), context.clone()).await
    }

    pub fn subscribe(
        &self,
        service_id: &str,
        mode: ServiceMode,
        listener: UpdateListener,
    ) -> Result<Box<dyn ServiceSubscription>, AnyError> {
        self.assert_active()?;
        self.assert_allowed(service_id)?;
        let subscriber = Arc::new(ProviderSubscriberState {
            listener,
            inner: Mutex::new(SubscriberInner {
                buffer: Vec::new(),
                snapshot_sequences: HashMap::new(),
                active: false,
                draining: false,
                terminated: false,
                closed: false,
            }),
        });
        let snapshot = {
            let mut inner = lock(&self.inner);
            let registration = registration_mut(&mut inner, service_id, mode)?;
            if registration.mode == ServiceMode::Singleton && registration.singleton.is_none() {
                return Err(remote_service_error(
                    RemoteServiceErrorCode::ServiceNotFound,
                    format!("Remote service {service_id} has no provider"),
                ));
            }
            registration.subscribers.push(Arc::clone(&subscriber));
            snapshot_registration(registration)
        };
        {
            let mut subscriber_inner = lock(&subscriber.inner);
            record_snapshot_sequences(&mut subscriber_inner.snapshot_sequences, &snapshot.instances);
        }
        Ok(Box::new(ProviderSubscription {
            snapshot,
            subscriber,
            inner: Arc::clone(&self.inner),
            service_id: service_id.to_string(),
        }))
    }

    pub fn dispose(&self) -> Result<(), AnyError> {
        let mut errors: Vec<AnyError> = Vec::new();
        let mut pending: Vec<(String, ServiceProviderUpdate)> = Vec::new();
        {
            let mut inner = lock(&self.inner);
            if inner.disposed {
                return Ok(());
            }
            inner.disposed = true;
            let ids: Vec<String> = inner.registrations.keys().cloned().collect();
            for id in ids {
                let registration = inner.registrations.get_mut(&id).expect("id from keys");
                if let Some(singleton) = registration.singleton.take() {
                    singleton.active.store(false, Ordering::SeqCst);
                    for remove in singleton.remove_member_listeners {
                        remove();
                    }
                    pending.push((id.clone(), ServiceProviderUpdate::Unavailable));
                }
                let keys: Vec<String> = registration.instances.keys().cloned().collect();
                for key in keys {
                    let instance = registration.instances.remove(&key).expect("key from keys");
                    instance.active.store(false, Ordering::SeqCst);
                    for remove in instance.remove_member_listeners {
                        remove();
                    }
                    let address = instance.address.clone().expect("keyed instances have addresses");
                    pending.push((id.clone(), ServiceProviderUpdate::Closed { instance: address }));
                }
                // Subscriber termination happens after the lifecycle emissions above,
                // mirroring upstream: a subscriber draining reentrantly still receives
                // its queued terminal update before being terminated.
            }
        }
        for (id, update) in pending {
            if let Err(error) = emit(&self.inner, &id, update, None) {
                errors.push(error);
            }
        }
        {
            let mut inner = lock(&self.inner);
            let ids: Vec<String> = inner.registrations.keys().cloned().collect();
            for id in ids {
                let registration = inner.registrations.get_mut(&id).expect("id from keys");
                for subscriber in registration.subscribers.iter() {
                    let mut subscriber_inner = lock(&subscriber.inner);
                    if subscriber_inner.active && !subscriber_inner.draining {
                        subscriber_inner.closed = true;
                        subscriber_inner.buffer.clear();
                    } else {
                        subscriber_inner.terminated = true;
                    }
                }
                registration.subscribers.clear();
            }
            inner.registrations.clear();
        }
        throw_collected_errors(errors, "Failed to dispose remote service provider")
    }
}

fn registration_mut<'a>(
    inner: &'a mut ProviderInner,
    service_id: &str,
    mode: ServiceMode,
) -> Result<&'a mut ServiceRegistration, AnyError> {
    let registration = inner.registrations.get_mut(service_id).ok_or_else(|| {
        remote_service_error(
            RemoteServiceErrorCode::ServiceNotFound,
            format!("Unknown remote service {service_id}"),
        )
    })?;
    if registration.mode != mode {
        return Err(remote_service_error(
            RemoteServiceErrorCode::ServiceModeMismatch,
            format!(
                "Remote service {service_id} is {}, not {}",
                registration.mode.as_str(),
                mode.as_str()
            ),
        ));
    }
    Ok(registration)
}

fn create_instance(
    inner: &Arc<Mutex<ProviderInner>>,
    registration: &ServiceRegistration,
    members: Vec<(String, InstanceMember)>,
    implementation: Arc<dyn RemoteServiceImpl>,
    address: Option<ServiceInstanceAddress>,
) -> ProviderInstance {
    let active = Arc::new(AtomicBool::new(true));
    let mut remove_member_listeners: Vec<Unsubscribe> = Vec::new();
    for (name, member) in &members {
        let InstanceMember::State(internals) = member else {
            continue;
        };
        let listener_active = Arc::clone(&active);
        let listener_inner = Arc::clone(inner);
        let listener_service_id = registration.service_id.clone();
        let listener_address = address.clone();
        let listener_member = name.clone();
        let listener: SourceListener = Arc::new(move |ops: &[Op], sequence: u64, context: &Context| {
            if !listener_active.load(Ordering::SeqCst) {
                return;
            }
            let update = ServiceProviderUpdate::State {
                instance: listener_address.clone(),
                member: listener_member.clone(),
                sequence,
                ops: ops.to_vec(),
            };
            // Emission failures cannot propagate through the infallible source
            // listener signature; upstream throws them into the state publisher.
            let _ = emit(&listener_inner, &listener_service_id, update, Some(context.clone()));
        });
        remove_member_listeners.push(internals.subscribe(listener));
    }
    ProviderInstance {
        address,
        implementation,
        members,
        remove_member_listeners,
        active,
    }
}

fn assert_singleton_shape(
    registration: &ServiceRegistration,
    service_id: &str,
    shape: &BTreeMap<String, &'static str>,
) -> Result<(), AnyError> {
    let Some(current) = &registration.singleton_shape else {
        return Ok(());
    };
    if same_service_member_shape(current, shape) {
        return Ok(());
    }
    Err(remote_service_error(
        RemoteServiceErrorCode::ServiceMemberMismatch,
        format!("Remote service {service_id} replacement must preserve its member shape"),
    ))
}

fn resolve_instance<'a>(
    registration: &'a ServiceRegistration,
    address: Option<&ServiceInstanceAddress>,
) -> Result<&'a ProviderInstance, AnyError> {
    if registration.mode == ServiceMode::Singleton {
        if address.is_some() {
            return Err(remote_service_error(
                RemoteServiceErrorCode::ServiceModeMismatch,
                format!("Remote service {} is singleton", registration.service_id),
            ));
        }
        return registration.singleton.as_ref().ok_or_else(|| {
            remote_service_error(
                RemoteServiceErrorCode::ServiceNotFound,
                format!("Remote service {} has no provider", registration.service_id),
            )
        });
    }
    let Some(address) = address else {
        return Err(remote_service_error(
            RemoteServiceErrorCode::ServiceModeMismatch,
            format!("Remote service {} is keyed", registration.service_id),
        ));
    };
    let Some(instance) = registration.instances.get(&address.key) else {
        return Err(remote_service_error(
            RemoteServiceErrorCode::ServiceInstanceNotFound,
            format!(
                "Remote service {} has no instance {}",
                registration.service_id, address.key
            ),
        ));
    };
    if instance.address.as_ref().map(|current| current.generation) != Some(address.generation) {
        return Err(remote_service_error(
            RemoteServiceErrorCode::ServiceStaleInstance,
            format!(
                "Remote service {} instance {} is stale",
                registration.service_id, address.key
            ),
        ));
    }
    Ok(instance)
}

fn snapshot_registration(registration: &ServiceRegistration) -> ServiceSubscriptionSnapshot {
    let instances = match registration.mode {
        ServiceMode::Singleton => match &registration.singleton {
            Some(singleton) => vec![snapshot_instance(singleton)],
            None => Vec::new(),
        },
        ServiceMode::Keyed => {
            let mut instances: Vec<&ProviderInstance> = registration.instances.values().collect();
            instances.sort_by(|left, right| {
                let left = left.address.as_ref().map(|address| address.key.as_str());
                let right = right.address.as_ref().map(|address| address.key.as_str());
                left.cmp(&right)
            });
            instances.into_iter().map(snapshot_instance).collect()
        }
    };
    ServiceSubscriptionSnapshot {
        service_id: registration.service_id.clone(),
        mode: registration.mode,
        instances,
    }
}

fn snapshot_instance(instance: &ProviderInstance) -> ServiceInstanceSnapshot {
    let members = instance
        .members
        .iter()
        .map(|(name, member)| match member {
            InstanceMember::Method(_) => ServiceMemberSnapshot::Method { name: name.clone() },
            InstanceMember::State(internals) => {
                let snapshot = internals.snapshot();
                ServiceMemberSnapshot::State {
                    name: name.clone(),
                    sequence: snapshot.sequence,
                    ops: vec![Op::Replace(snapshot.value)],
                }
            }
        })
        .collect();
    ServiceInstanceSnapshot {
        instance: instance.address.clone(),
        members,
    }
}

/// Queue one update for every subscriber, then drain them outside the provider lock.
fn emit(
    inner: &Arc<Mutex<ProviderInner>>,
    service_id: &str,
    update: ServiceProviderUpdate,
    context: Option<Context>,
) -> Result<(), AnyError> {
    let subscribers = {
        let mut guard = lock(inner);
        queue_update(&mut guard, service_id, update, context)
    };
    let errors = drain_all(&subscribers);
    throw_collected_errors(errors, &format!("Failed to publish remote service {service_id} update"))
}

/// Fill subscriber buffers (upstream `#emit` queueing phase). Returns the live
/// subscribers in delivery order; the caller drains them without the provider lock.
fn queue_update(
    inner: &mut ProviderInner,
    service_id: &str,
    update: ServiceProviderUpdate,
    context: Option<Context>,
) -> Vec<Arc<ProviderSubscriberState>> {
    let Some(registration) = inner.registrations.get_mut(service_id) else {
        return Vec::new();
    };
    if registration.subscribers.is_empty() {
        return Vec::new();
    }
    let delivery_context = context.unwrap_or_else(service_delivery_context);
    let mut live: Vec<Arc<ProviderSubscriberState>> = Vec::new();
    live.extend(registration.subscribers.iter().map(Arc::clone));
    // Queue for everyone before invoking user code, including reentrant publications.
    for subscriber in &live {
        let mut subscriber_inner = lock(&subscriber.inner);
        if subscriber_inner.closed
            || update_covered_by_snapshot(&mut subscriber_inner.snapshot_sequences, &update)
        {
            continue;
        }
        if subscriber_inner.buffer.len() == 100 {
            let snapshot = snapshot_registration(registration);
            subscriber_inner.buffer.clear();
            subscriber_inner.snapshot_sequences.clear();
            record_snapshot_sequences(&mut subscriber_inner.snapshot_sequences, &snapshot.instances);
            subscriber_inner
                .buffer
                .push((ServiceProviderUpdate::Reset { snapshot }, delivery_context.clone()));
        } else {
            subscriber_inner
                .buffer
                .push((update.clone(), delivery_context.clone()));
        }
    }
    live
}

fn drain_all(subscribers: &[Arc<ProviderSubscriberState>]) -> Vec<AnyError> {
    let mut errors = Vec::new();
    for subscriber in subscribers {
        errors.extend(drain_subscriber(subscriber));
    }
    errors
}

fn drain_subscriber(subscriber: &ProviderSubscriberState) -> Vec<AnyError> {
    let mut errors: Vec<AnyError> = Vec::new();
    let mut guard = lock(&subscriber.inner);
    if !guard.active || guard.closed || guard.draining {
        return errors;
    }
    guard.draining = true;
    while !guard.closed {
        if guard.buffer.is_empty() {
            break;
        }
        let entry = guard.buffer.remove(0);
        drop(guard);
        let listener = &subscriber.listener;
        let outcome = catch_unwind(AssertUnwindSafe(|| listener(&entry.0, &entry.1)));
        if let Err(panic) = outcome {
            errors.push(panic_to_error(panic));
        }
        guard = lock(&subscriber.inner);
    }
    guard.draining = false;
    if guard.terminated {
        guard.closed = true;
    }
    errors
}

fn record_snapshot_sequences(
    sequences: &mut HashMap<String, u64>,
    instances: &[ServiceInstanceSnapshot],
) {
    for instance in instances {
        for member in &instance.members {
            if let ServiceMemberSnapshot::State { name, sequence, .. } = member {
                sequences.insert(state_member_key(instance.instance.as_ref(), name), *sequence);
            }
        }
    }
}

fn update_covered_by_snapshot(
    sequences: &mut HashMap<String, u64>,
    update: &ServiceProviderUpdate,
) -> bool {
    match update {
        ServiceProviderUpdate::State {
            instance, member, sequence, ..
        } => {
            let key = state_member_key(instance.as_ref(), member);
            let Some(recorded) = sequences.get(&key) else {
                return false;
            };
            if *sequence <= *recorded {
                return true;
            }
            sequences.remove(&key);
            false
        }
        ServiceProviderUpdate::Reset { snapshot } => {
            sequences.clear();
            record_snapshot_sequences(sequences, &snapshot.instances);
            false
        }
        ServiceProviderUpdate::Replaced { snapshot } => {
            sequences.clear();
            record_snapshot_sequences(sequences, std::slice::from_ref(snapshot));
            false
        }
        ServiceProviderUpdate::Spawned { instance } => {
            record_snapshot_sequences(sequences, std::slice::from_ref(instance));
            false
        }
        ServiceProviderUpdate::Unavailable => {
            sequences.clear();
            false
        }
        ServiceProviderUpdate::Closed { instance } => {
            let encoded = serde_json::to_string(&serde_json::json!([instance.key, instance.generation]))
                .expect("address serializes");
            let prefix = format!("{},", encoded.trim_end_matches(']'));
            let stale: Vec<String> = sequences
                .keys()
                .filter(|key| key.starts_with(&prefix))
                .cloned()
                .collect();
            for key in stale {
                sequences.remove(&key);
            }
            false
        }
    }
}

fn state_member_key(instance: Option<&ServiceInstanceAddress>, member: &str) -> String {
    let value = match instance {
        None => serde_json::json!([member]),
        Some(address) => serde_json::json!([address.key, address.generation, member]),
    };
    serde_json::to_string(&value).expect("member key serializes")
}

struct ProviderSubscription {
    snapshot: ServiceSubscriptionSnapshot,
    subscriber: Arc<ProviderSubscriberState>,
    inner: Arc<Mutex<ProviderInner>>,
    service_id: String,
}

#[async_trait]
impl ServiceSubscription for ProviderSubscription {
    fn snapshot(&self) -> &ServiceSubscriptionSnapshot {
        &self.snapshot
    }

    fn activate(&self) {
        {
            let mut subscriber_inner = lock(&self.subscriber.inner);
            if subscriber_inner.closed || subscriber_inner.active {
                return;
            }
            subscriber_inner.active = true;
        }
        let errors = drain_subscriber(&self.subscriber);
        // Upstream throws "Failed to activate remote service subscription"; the
        // trait signature is infallible, so activation-time listener failures are
        // logged instead of propagated.
        if errors.len() == 1 {
            tracing::warn!("Failed to activate remote service subscription: {}", errors[0]);
        } else if !errors.is_empty() {
            tracing::warn!(
                "Failed to activate remote service subscription: {} failures",
                errors.len()
            );
        }
    }

    async fn close(&self, _context: Option<Context>) {
        {
            let mut subscriber_inner = lock(&self.subscriber.inner);
            if subscriber_inner.closed {
                return;
            }
            subscriber_inner.closed = true;
            subscriber_inner.buffer.clear();
        }
        let mut inner = lock(&self.inner);
        if let Some(registration) = inner.registrations.get_mut(&self.service_id) {
            let target = Arc::as_ptr(&self.subscriber);
            registration.subscribers.retain(|subscriber| Arc::as_ptr(subscriber) != target);
        }
    }
}

fn classify(
    service_id: &str,
    implementation: &Arc<dyn RemoteServiceImpl>,
) -> Result<Vec<(String, InstanceMember)>, AnyError> {
    let mut members = implementation.members();
    members.sort_by(|left, right| left.0.cmp(&right.0));
    members.dedup_by(|left, right| left.0 == right.0);
    if members.is_empty() {
        return Err(type_error(format!("Remote service {service_id} has no members")));
    }
    Ok(members
        .into_iter()
        .map(|(name, member)| {
            (
                name,
                match member {
                    ServiceMember::Method(method) => InstanceMember::Method(method),
                    ServiceMember::State(internals) => InstanceMember::State(internals),
                },
            )
        })
        .collect())
}

pub fn validate_remote_service_implementation(
    service_id: &str,
    implementation: Arc<dyn RemoteServiceImpl>,
) -> Result<(), AnyError> {
    classify(service_id, &implementation).map(|_| ())
}

/// Runtime-built remote service implementation (upstream: a plain object literal of
/// methods and replicated states, classified through property reflection).
pub struct RemoteServiceObject {
    members: BTreeMap<String, ServiceMember>,
    local_object: Option<Arc<dyn Any + Send + Sync>>,
}

impl RemoteServiceObject {
    pub fn builder() -> RemoteServiceObjectBuilder {
        RemoteServiceObjectBuilder {
            members: BTreeMap::new(),
            local_object: None,
        }
    }
}

impl RemoteServiceImpl for RemoteServiceObject {
    fn members(&self) -> Vec<(String, ServiceMember)> {
        self.members
            .iter()
            .map(|(name, member)| {
                (
                    name.clone(),
                    match member {
                        ServiceMember::Method(method) => ServiceMember::Method(Arc::clone(method)),
                        ServiceMember::State(internals) => ServiceMember::State(Arc::clone(internals)),
                    },
                )
            })
            .collect()
    }

    fn as_local_object(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        self.local_object.clone()
    }
}

/// Builder for [`RemoteServiceObject`]; members are kept in sorted order.
pub struct RemoteServiceObjectBuilder {
    members: BTreeMap<String, ServiceMember>,
    local_object: Option<Arc<dyn Any + Send + Sync>>,
}

impl RemoteServiceObjectBuilder {
    pub fn method(
        mut self,
        name: impl Into<String>,
        method: impl Fn(Vec<JsonValue>, Context) -> BoxFuture<Result<Option<JsonValue>, AnyError>>
        + Send
        + Sync
        + 'static,
    ) -> RemoteServiceObjectBuilder {
        self.members
            .insert(name.into(), ServiceMember::Method(Arc::new(method)));
        self
    }

    pub fn state(
        mut self,
        name: impl Into<String>,
        internals: Arc<dyn StateInternals>,
    ) -> RemoteServiceObjectBuilder {
        self.members.insert(name.into(), ServiceMember::State(internals));
        self
    }

    /// Registers the concrete object handed back by `use_singleton`
    /// (upstream: the implementation object itself).
    pub fn local_object(mut self, object: Arc<dyn Any + Send + Sync>) -> RemoteServiceObjectBuilder {
        self.local_object = Some(object);
        self
    }

    pub fn build(self) -> Arc<RemoteServiceObject> {
        Arc::new(RemoteServiceObject {
            members: self.members,
            local_object: self.local_object,
        })
    }
}

fn throw_collected_errors(errors: Vec<AnyError>, message: &str) -> Result<(), AnyError> {
    match errors.len() {
        0 => Ok(()),
        1 => Err(errors.into_iter().next().expect("one error")),
        _ => Err(AggregateError::new(message, errors)),
    }
}

fn service_member_shape(members: &[(String, InstanceMember)]) -> BTreeMap<String, &'static str> {
    members
        .iter()
        .map(|(name, member)| {
            (
                name.clone(),
                match member {
                    InstanceMember::Method(_) => "method",
                    InstanceMember::State(_) => "state",
                },
            )
        })
        .collect()
}

fn same_service_member_shape(
    left: &BTreeMap<String, &'static str>,
    right: &BTreeMap<String, &'static str>,
) -> bool {
    left == right
}

fn panic_to_error(panic: Box<dyn std::any::Any + Send>) -> AnyError {
    if let Some(message) = panic.downcast_ref::<String>() {
        return error(message.clone());
    }
    if let Some(message) = panic.downcast_ref::<&str>() {
        return error((*message).to_string());
    }
    error("listener panicked")
}

/// Hosts one provider for one remote consumer and owns that consumer's subscriptions
/// (upstream `createRemoteServiceEndpoint`).
pub struct RemoteServiceEndpoint {
    provider: Arc<RemoteServiceProvider>,
    subscriptions: Mutex<HashMap<String, Box<dyn ServiceSubscription>>>,
    disposed: AtomicBool,
}

pub fn create_remote_service_endpoint(provider: Arc<RemoteServiceProvider>) -> RemoteServiceEndpoint {
    RemoteServiceEndpoint {
        provider,
        subscriptions: Mutex::new(HashMap::new()),
        disposed: AtomicBool::new(false),
    }
}

impl RemoteServiceEndpoint {
    pub async fn invoke(
        &self,
        call: &ServiceCall,
        publish: &ServiceUpdatePublisher,
        context: &Context,
    ) -> Result<Option<JsonValue>, AnyError> {
        if self.disposed.load(Ordering::SeqCst) {
            return Err(error("Remote service endpoint is disposed"));
        }
        match decode_service_control_call(call) {
            Some(ServiceControlCall::Catalogue) => Ok(Some(
                serde_json::to_value(self.provider.catalogue())
                    .map_err(|serde_error| error(serde_error.to_string()))?,
            )),
            Some(ServiceControlCall::Subscribe {
                subscription_id,
                service_id,
                mode,
            }) => {
                {
                    let subscriptions = lock(&self.subscriptions);
                    if subscriptions.contains_key(&subscription_id) {
                        return Err(error("Service subscription ID is already active"));
                    }
                }
                let publisher = Arc::clone(publish);
                let publisher_subscription_id = subscription_id.clone();
                let listener: UpdateListener =
                    Arc::new(move |update: &ServiceProviderUpdate, context: &Context| {
                        let future = publisher(publisher_subscription_id.clone(), update, context);
                        // Upstream: `void Promise.resolve(publish(...)).catch(() => {})`.
                        let _ = tokio::spawn(async move {
                            future.await;
                        });
                    });
                let subscription = self.provider.subscribe(&service_id, mode, listener)?;
                let snapshot = serde_json::to_value(subscription.snapshot())
                    .map_err(|serde_error| error(serde_error.to_string()))?;
                {
                    let mut subscriptions = lock(&self.subscriptions);
                    subscriptions.insert(subscription_id.clone(), subscription);
                    subscriptions
                        .get(&subscription_id)
                        .expect("just inserted")
                        .activate();
                }
                Ok(Some(snapshot))
            }
            Some(ServiceControlCall::Unsubscribe { subscription_id }) => {
                let subscription = lock(&self.subscriptions).remove(&subscription_id);
                let Some(subscription) = subscription else {
                    return Err(error("Service subscription was not found"));
                };
                subscription.close(None).await;
                Ok(None)
            }
            None => self.provider.invoke(call, context).await,
        }
    }

    pub async fn dispose(&self) {
        if self.disposed.swap(true, Ordering::SeqCst) {
            return;
        }
        let subscriptions: Vec<Box<dyn ServiceSubscription>> = lock(&self.subscriptions)
            .drain()
            .map(|(_, value)| value)
            .collect();
        for subscription in subscriptions {
            subscription.close(None).await;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! Port of the provider halves of upstream `test/services.test.ts` and
    //! `test/service-delivery.test.ts` (binding halves live in `consumer::tests`).

    use super::*;
    use crate::context::BACKGROUND_CONTEXT;
    use serde_json::json;
    use std::sync::Arc as StdArc;

    pub(crate) fn lock_for_test<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        lock(mutex)
    }

    /// `Result::unwrap_err` for non-`Debug` success types.
    pub(crate) fn err_of<T>(result: Result<T, AnyError>) -> AnyError {
        match result {
            Err(error) => error,
            Ok(_) => panic!("expected an error"),
        }
    }

    /// Minimal `StateInternals` double with upstream publisher semantics: source
    /// listeners run per publication in queue order, reentrant publications defer.
    pub(crate) struct FakeState {
        inner: Mutex<FakeStateInner>,
    }

    struct FakeStateInner {
        value: JsonValue,
        sequence: u64,
        listeners: Vec<SourceListener>,
        publications: Vec<(Vec<Op>, u64, Context)>,
        delivering: bool,
    }

    impl FakeState {
        pub(crate) fn new(initial: JsonValue) -> Arc<FakeState> {
            Arc::new(FakeState {
                inner: Mutex::new(FakeStateInner {
                    value: initial,
                    sequence: 0,
                    listeners: Vec::new(),
                    publications: Vec::new(),
                    delivering: false,
                }),
            })
        }

        #[allow(dead_code)]
        pub(crate) fn value(&self) -> JsonValue {
            lock(&self.inner).value.clone()
        }

        #[allow(dead_code)]
        pub(crate) fn sequence(&self) -> u64 {
            lock(&self.inner).sequence
        }

        pub(crate) fn replace_with(&self, value: JsonValue, context: &Context) {
            self.publish(vec![Op::Replace(value)], context);
        }

        pub(crate) fn publish(&self, ops: Vec<Op>, context: &Context) {
            {
                let mut inner = lock(&self.inner);
                let next = crate::delta::apply_immutable(&inner.value, &ops).expect("fake ops valid");
                inner.value = next;
                let sequence = inner.sequence + 1;
                inner.sequence = sequence;
                inner.publications.push((ops, sequence, context.clone()));
                if inner.delivering {
                    return;
                }
                inner.delivering = true;
            }
            loop {
                let publication = {
                    let mut inner = lock(&self.inner);
                    if inner.publications.is_empty() {
                        inner.delivering = false;
                        None
                    } else {
                        Some(inner.publications.remove(0))
                    }
                };
                let Some((ops, sequence, context)) = publication else {
                    break;
                };
                let listeners = lock(&self.inner).listeners.clone();
                for listener in listeners {
                    listener(&ops, sequence, &context);
                }
            }
        }
    }

    impl StateInternals for FakeState {
        fn snapshot(&self) -> crate::services::state_internals::StateSnapshot {
            let inner = lock(&self.inner);
            crate::services::state_internals::StateSnapshot {
                value: inner.value.clone(),
                sequence: inner.sequence,
            }
        }

        fn subscribe(&self, listener: SourceListener) -> Unsubscribe {
            lock(&self.inner).listeners.push(listener);
            Box::new(|| {})
        }
    }

    /// One `RemoteServiceObject` with one state member and one method member.
    pub(crate) type TestMethod = Box<
        dyn Fn(Vec<JsonValue>, Context) -> BoxFuture<Result<Option<JsonValue>, AnyError>> + Send + Sync,
    >;

    pub(crate) fn object_with_state(
        state: &Arc<FakeState>,
        method_name: &str,
        method: TestMethod,
    ) -> Arc<dyn RemoteServiceImpl> {
        object_with_named_state("state", state, method_name, method)
    }

    pub(crate) fn object_with_named_state(
        state_name: &str,
        state: &Arc<FakeState>,
        method_name: &str,
        method: TestMethod,
    ) -> Arc<dyn RemoteServiceImpl> {
        let mut builder =
            RemoteServiceObject::builder().state(state_name, Arc::clone(state) as Arc<dyn StateInternals>);
        builder = builder.method(method_name, move |args: Vec<JsonValue>, context: Context| method(args, context));
        builder.build() as Arc<dyn RemoteServiceImpl>
    }

    pub(crate) fn recording_method(
        received: Arc<Mutex<Vec<JsonValue>>>,
        response: JsonValue,
    ) -> TestMethod {
        Box::new(move |mut args: Vec<JsonValue>, _context: Context| {
            let payload = if args.is_empty() { JsonValue::Null } else { args.remove(0) };
            received.lock().unwrap().push(payload.clone());
            let response = response.clone();
            Box::pin(async move { Ok(Some(response)) })
        })
    }

    fn snapshot_json(subscription: &dyn ServiceSubscription) -> JsonValue {
        serde_json::to_value(subscription.snapshot()).expect("snapshot serializes")
    }

    fn update_json(update: &ServiceProviderUpdate) -> JsonValue {
        serde_json::to_value(update).expect("update serializes")
    }

    #[test]
    fn constructor_rejects_duplicate_catalogue_ids() {
        let error = err_of(RemoteServiceProvider::new(vec![
            ProviderDefinition::new("test.dup", ServiceMode::Singleton),
            ProviderDefinition::new("test.dup", ServiceMode::Keyed),
        ])
        );
        assert_eq!(error.to_string(), "Remote service catalogue contains duplicate IDs");
    }

    #[test]
    fn validate_rejects_implementation_without_members() {
        struct Empty;
        impl RemoteServiceImpl for Empty {
            fn members(&self) -> Vec<(String, ServiceMember)> {
                Vec::new()
            }
        }
        let error = err_of(validate_remote_service_implementation("test.models", Arc::new(Empty)));
        assert_eq!(error.to_string(), "Remote service test.models has no members");
    }

    #[test]
    fn rejects_unallowlisted_services() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.models",
            ServiceMode::Singleton,
        )])
        .expect("provider");
        let error = err_of(provider.use_singleton("test.unknown"));
        assert_eq!(error.to_string(), "Remote service test.unknown is not allowlisted");
        let error = err_of(provider.subscribe("test.unknown", ServiceMode::Singleton, Arc::new(|_, _| {})));
        assert_eq!(error.to_string(), "Remote service test.unknown is not allowlisted");
    }

    #[test]
    fn reports_missing_singletons_with_upstream_messages() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.models",
            ServiceMode::Singleton,
        )])
        .expect("provider");
        let error = err_of(provider.use_singleton("test.models"));
        assert_eq!(error.to_string(), "Remote service test.models has no local provider");
        let error = err_of(provider
            .subscribe("test.models", ServiceMode::Singleton, Arc::new(|_, _| {}))
            );
        assert_eq!(error.to_string(), "Remote service test.models has no provider");
    }

    #[test]
    fn registration_mode_mismatches_use_upstream_messages() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.models",
            ServiceMode::Singleton,
        )])
        .expect("provider");
        let error = err_of(provider
            .subscribe("test.models", ServiceMode::Keyed, Arc::new(|_, _| {}))
            );
        assert_eq!(error.to_string(), "Remote service test.models is singleton, not keyed");
    }

    #[test]
    fn spawn_rejects_empty_and_duplicate_keys() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.keyed",
            ServiceMode::Keyed,
        )])
        .expect("provider");
        let state = FakeState::new(json!({ "question": "First?" }));
        let error = err_of(provider
            .spawn("test.keyed", "", object_with_state(&state, "submit", recording_method(Default::default(), json!(null))))
            );
        assert_eq!(error.to_string(), "Remote service instance key must not be empty");

        let method: TestMethod = Box::new(|_args: Vec<JsonValue>, _context: Context| {
            Box::pin(async { Ok(Some(json!({ "accepted": true }))) })
        });
        let _close = provider
            .spawn("test.keyed", "invocation-1", object_with_state(&state, "submit", method))
            .expect("spawn");
        let method: TestMethod = Box::new(|_args: Vec<JsonValue>, _context: Context| {
            Box::pin(async { Ok(None) })
        });
        let error = err_of(provider
            .spawn("test.keyed", "invocation-1", object_with_state(&state, "submit", method)));
        assert_eq!(
            error.to_string(),
            "Remote service test.keyed already has a live instance with key invocation-1"
        );
    }

    #[test]
    fn spawn_on_singleton_registration_reports_mode_mismatch() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.models",
            ServiceMode::Singleton,
        )])
        .expect("provider");
        let state = FakeState::new(json!({ "selected": null, "revision": 0 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        let error = err_of(provider
            .spawn("test.models", "wrong", object_with_state(&state, "select", method)));
        assert!(
            error.to_string().contains("singleton"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn invoke_reports_unknown_members_and_state_members() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.models",
            ServiceMode::Singleton,
        )])
        .expect("provider");
        let state = FakeState::new(json!({ "selected": null, "revision": 0 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide("test.models", object_with_state(&state, "select", method))
            .expect("provide");

        let error = err_of(provider
            .invoke(
                &ServiceCall {
                    service_id: "test.models".into(),
                    instance: None,
                    member: "missing".into(),
                    args: Vec::new(),
                },
                &BACKGROUND_CONTEXT,
            )
            .await
            );
        assert_eq!(error.to_string(), "Unknown remote service member test.models.missing");

        let error = err_of(provider
            .invoke(
                &ServiceCall {
                    service_id: "test.models".into(),
                    instance: None,
                    member: "state".into(),
                    args: Vec::new(),
                },
                &BACKGROUND_CONTEXT,
            )
            .await
            );
        assert_eq!(error.to_string(), "Remote service member test.models.state is not a method");

        let error = err_of(provider
            .invoke(
                &ServiceCall {
                    service_id: "test.unknown".into(),
                    instance: None,
                    member: "select".into(),
                    args: Vec::new(),
                },
                &BACKGROUND_CONTEXT,
            )
            .await
            );
        // Upstream also runs #assertAllowed before the registration lookup on invoke,
        // so an unknown service reports "not allowlisted"; the "Unknown remote
        // service {id}" registration error is defense-in-depth behind that check.
        assert_eq!(error.to_string(), "Remote service test.unknown is not allowlisted");
    }

    #[tokio::test]
    async fn invoke_dispatches_arguments_and_context_to_methods() {
        let provider = Arc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new("test.echo", ServiceMode::Singleton)])
                .expect("provider"),
        );
        let received = Arc::new(Mutex::new(Vec::new()));
        let method: TestMethod = {
            let received = Arc::clone(&received);
            Box::new(move |mut args: Vec<JsonValue>, context: Context| {
                let payload = args.remove(0);
                received.lock().unwrap().push(json!({
                    "payload": payload,
                    "context": context.to_string(),
                }));
                Box::pin(async move { Ok(Some(json!({ "value": "response" }))) })
            })
        };
        let state = FakeState::new(json!({}));
        provider
            .provide("test.echo", object_with_state(&state, "echo", method))
            .expect("provide");
        let result = provider
            .invoke(
                &ServiceCall {
                    service_id: "test.echo".into(),
                    instance: None,
                    member: "echo".into(),
                    args: vec![json!({ "value": "request" })],
                },
                &BACKGROUND_CONTEXT,
            )
            .await
            .expect("invoke");
        assert_eq!(result, Some(json!({ "value": "response" })));
        assert_eq!(
            lock_for_test(&received).clone(),
            vec![json!({
                "payload": { "value": "request" },
                "context": "[Context BACKGROUND_CONTEXT]",
            })]
        );
    }

    #[tokio::test]
    async fn invoke_resolves_instances_with_upstream_errors() {
        let provider = RemoteServiceProvider::new(vec![
            ProviderDefinition::new("test.models", ServiceMode::Singleton),
            ProviderDefinition::new("test.keyed", ServiceMode::Keyed),
        ])
        .expect("provider");
        let state = FakeState::new(json!({}));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide("test.models", object_with_state(&state, "select", method))
            .expect("provide");
        let keyed_method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        let _close = provider
            .spawn("test.keyed", "one", object_with_state(&state, "submit", keyed_method))
            .expect("spawn");

        let call = |service_id: &str, instance: Option<ServiceInstanceAddress>| ServiceCall {
            service_id: service_id.to_string(),
            instance,
            member: "select".into(),
            args: Vec::new(),
        };
        let error = err_of(provider
            .invoke(&call("test.models", Some(ServiceInstanceAddress { key: "one".into(), generation: 1 })), &BACKGROUND_CONTEXT)
            .await
            );
        assert_eq!(error.to_string(), "Remote service test.models is singleton");

        let error = err_of(provider.invoke(&call("test.keyed", None), &BACKGROUND_CONTEXT).await);
        assert_eq!(error.to_string(), "Remote service test.keyed is keyed");

        let error = err_of(provider
            .invoke(&call("test.keyed", Some(ServiceInstanceAddress { key: "nope".into(), generation: 1 })), &BACKGROUND_CONTEXT)
            .await
            );
        assert_eq!(error.to_string(), "Remote service test.keyed has no instance nope");

        let error = err_of(provider
            .invoke(
                &call(
                    "test.keyed",
                    Some(ServiceInstanceAddress { key: "one".into(), generation: 2 }),
                ),
                &BACKGROUND_CONTEXT,
            )
            .await
            );
        assert_eq!(error.to_string(), "Remote service test.keyed instance one is stale");
    }

    #[test]
    fn catalogue_lists_definitions_in_order() {
        let provider = RemoteServiceProvider::new(vec![
            ProviderDefinition::new("test.models", ServiceMode::Singleton),
            ProviderDefinition::new("test.question-dialog", ServiceMode::Keyed),
        ])
        .expect("provider");
        assert_eq!(
            serde_json::to_value(provider.catalogue()).expect("catalogue serializes"),
            json!([
                { "serviceId": "test.models", "mode": "singleton" },
                { "serviceId": "test.question-dialog", "mode": "keyed" },
            ])
        );
    }

    #[test]
    fn publishes_compact_tracked_operations_through_the_remote_provider() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.timeline",
            ServiceMode::Singleton,
        )])
        .expect("provider");
        let initial = json!({ "entries": [{ "id": "one" }], "retained": { "value": 1 } });
        let state = FakeState::new(initial.clone());
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide("test.timeline", object_with_state(&state, "extend", method))
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
            snapshot_json(raw.as_ref()),
            json!({
                "serviceId": "test.timeline",
                "mode": "singleton",
                "instances": [{
                    "members": [
                        { "name": "extend", "kind": "method" },
                        { "name": "state", "kind": "state", "sequence": 0, "ops": [["r", initial]] },
                    ],
                }],
            })
        );
        raw.activate();

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
            update_json(&lock_for_test(&updates)[0]),
            json!({
                "type": "state",
                "member": "state",
                "sequence": 1,
                "ops": [["p", ["entries"], 1, 0, [{ "id": "two" }]]],
            })
        );

        state.publish(
            vec![Op::Splice {
                path: vec![crate::delta::Seg::Key("entries".into())],
                start: 2,
                remove: 0,
                items: vec![json!({ "id": "three" })],
            }],
            &BACKGROUND_CONTEXT,
        );
        let late = provider
            .subscribe("test.timeline", ServiceMode::Singleton, Arc::new(|_, _| {}))
            .expect("late subscribe");
        assert_eq!(
            snapshot_json(late.as_ref()),
            json!({
                "serviceId": "test.timeline",
                "mode": "singleton",
                "instances": [{
                    "members": [
                        { "name": "extend", "kind": "method" },
                        {
                            "name": "state",
                            "kind": "state",
                            "sequence": 2,
                            "ops": [["r", { "entries": [{ "id": "one" }, { "id": "two" }, { "id": "three" }], "retained": { "value": 1 } }]],
                        },
                    ],
                }],
            })
        );
        futures::executor::block_on(late.close(None));
    }

    #[test]
    fn withdraw_and_replace_emit_lifecycle_updates_without_breaking_shape() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.models",
            ServiceMode::Singleton,
        )])
        .expect("provider");
        let state = FakeState::new(json!({ "selected": null, "revision": 1 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide("test.models", object_with_state(&state, "select", method))
            .expect("provide");
        let updates: Arc<Mutex<Vec<ServiceProviderUpdate>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&updates);
        let subscription = provider
            .subscribe(
                "test.models",
                ServiceMode::Singleton,
                Arc::new(move |update: &ServiceProviderUpdate, _| sink.lock().unwrap().push(update.clone())),
            )
            .expect("subscribe");
        subscription.activate();

        provider.withdraw("test.models").expect("withdraw");
        assert_eq!(update_json(&lock_for_test(&updates)[0]), json!({ "type": "unavailable" }));

        // Replacing after withdraw keeps requiring the original member shape.
        let mismatched: Arc<dyn RemoteServiceImpl> = RemoteServiceObject::builder()
            .method("select", |_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }))
            .build();
        let error = err_of(provider.replace("test.models", Arc::clone(&mismatched)));
        assert_eq!(
            error.to_string(),
            "Remote service test.models replacement must preserve its member shape"
        );
        let error = err_of(provider.validate_replacement("test.models", mismatched));
        assert_eq!(
            error.to_string(),
            "Remote service test.models replacement must preserve its member shape"
        );

        let replacement_state = FakeState::new(json!({ "selected": null, "revision": 2 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .replace(
                "test.models",
                object_with_state(&replacement_state, "select", method),
            )
            .expect("replace");
        let updates = lock_for_test(&updates);
        assert_eq!(update_json(&updates[1])["type"], json!("replaced"));
        assert_eq!(
            update_json(&updates[1])["snapshot"]["members"][1]["ops"][0][1],
            json!({ "selected": null, "revision": 2 })
        );
        drop(updates);
        provider.dispose().expect("dispose");
    }

    #[test]
    fn delivers_active_subscriber_updates_before_reporting_listener_failures() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.models",
            ServiceMode::Singleton,
        )])
        .expect("provider");
        let state = FakeState::new(json!({ "selected": null, "revision": 1 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide("test.models", object_with_state(&state, "select", method))
            .expect("provide");
        let failing = provider
            .subscribe(
                "test.models",
                ServiceMode::Singleton,
                Arc::new(|_, _| panic!("listener failed")),
            )
            .expect("subscribe");
        let delivered = Arc::new(Mutex::new(0usize));
        let counter = Arc::clone(&delivered);
        let succeeding = provider
            .subscribe(
                "test.models",
                ServiceMode::Singleton,
                Arc::new(move |_, _| *counter.lock().unwrap() += 1),
            )
            .expect("subscribe");
        failing.activate();
        succeeding.activate();

        let replacement_state = FakeState::new(json!({ "selected": null, "revision": 2 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        let error = err_of(provider.replace(
            "test.models",
            object_with_state(&replacement_state, "select", method),
        ));
        assert_eq!(error.to_string(), "listener failed");
        assert_eq!(*lock_for_test(&delivered), 1);

        futures::executor::block_on(failing.close(None));
        futures::executor::block_on(succeeding.close(None));
        provider.dispose().expect("dispose");
    }

    #[test]
    fn replays_every_buffered_update_before_reporting_listener_failures() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.models",
            ServiceMode::Singleton,
        )])
        .expect("provider");
        let state = FakeState::new(json!({ "selected": null, "revision": 0 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide("test.models", object_with_state(&state, "select", method))
            .expect("provide");
        let delivered = Arc::new(Mutex::new(0usize));
        let counter = Arc::clone(&delivered);
        let subscription = provider
            .subscribe(
                "test.models",
                ServiceMode::Singleton,
                Arc::new(move |_, _| {
                    let mut count = counter.lock().unwrap();
                    *count += 1;
                    if *count > 2 {
                        panic!("listener failed");
                    }
                }),
            )
            .expect("subscribe");
        state.replace_with(json!({ "selected": null, "revision": 1 }), &BACKGROUND_CONTEXT);
        state.replace_with(json!({ "selected": null, "revision": 2 }), &BACKGROUND_CONTEXT);

        // Upstream also throws "Failed to activate remote service subscription";
        // the Rust `activate` is infallible, so only the replay order is asserted.
        subscription.activate();
        assert_eq!(*lock_for_test(&delivered), 2);
        futures::executor::block_on(subscription.close(None));
        provider.dispose().expect("dispose");
    }

    #[test]
    fn does_not_replay_a_queued_revision_already_covered_by_a_new_snapshot() {
        let provider = Arc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new(
                "test.models",
                ServiceMode::Singleton,
            )])
            .expect("provider"),
        );
        let state = FakeState::new(json!({ "selected": null, "revision": 0 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide("test.models", object_with_state(&state, "select", method))
            .expect("provide");
        let late_updates: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
        let late_sink = Arc::clone(&late_updates);
        let late_holder: Arc<Mutex<Option<Box<dyn ServiceSubscription>>>> = Arc::new(Mutex::new(None));
        let late_holder_for_listener = Arc::clone(&late_holder);
        let provider_for_listener = Arc::clone(&provider);
        let state_for_listener = Arc::clone(&state);
        let first = provider
            .subscribe(
                "test.models",
                ServiceMode::Singleton,
                Arc::new(move |update: &ServiceProviderUpdate, _| {
                    let sequence = match update {
                        ServiceProviderUpdate::State { sequence, .. } => *sequence,
                        _ => return,
                    };
                    if sequence != 1 {
                        return;
                    }
                    state_for_listener.replace_with(json!({ "selected": null, "revision": 2 }), &BACKGROUND_CONTEXT);
                    let late_sink = Arc::clone(&late_sink);
                    let late = provider_for_listener
                        .subscribe(
                            "test.models",
                            ServiceMode::Singleton,
                            Arc::new(move |update: &ServiceProviderUpdate, _| {
                                if let ServiceProviderUpdate::State { sequence, .. } = update {
                                    late_sink.lock().unwrap().push(*sequence);
                                }
                            }),
                        )
                        .expect("late subscribe");
                    assert_eq!(
                        snapshot_json(late.as_ref())["instances"][0]["members"][1],
                        json!({
                            "name": "state",
                            "kind": "state",
                            "sequence": 2,
                            "ops": [["r", { "selected": null, "revision": 2 }]],
                        })
                    );
                    late.activate();
                    *lock_for_test(&late_holder_for_listener) = Some(late);
                }),
            )
            .expect("subscribe");
        first.activate();
        state.replace_with(json!({ "selected": null, "revision": 1 }), &BACKGROUND_CONTEXT);
        assert!(lock_for_test(&late_updates).is_empty());
        state.replace_with(json!({ "selected": null, "revision": 3 }), &BACKGROUND_CONTEXT);
        assert_eq!(*lock_for_test(&late_updates), vec![3]);
        if let Some(late) = lock_for_test(&late_holder).take() {
            futures::executor::block_on(late.close(None));
        }
        futures::executor::block_on(first.close(None));
        provider.dispose().expect("dispose");
    }

    #[test]
    fn appends_reentrant_updates_to_the_activation_fifo() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.counter",
            ServiceMode::Singleton,
        )])
        .expect("provider");
        let state = FakeState::new(json!({ "value": 0 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide("test.counter", object_with_state(&state, "bump", method))
            .expect("provide");
        let received: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&received);
        let state_for_listener = Arc::clone(&state);
        let subscription = provider
            .subscribe(
                "test.counter",
                ServiceMode::Singleton,
                Arc::new(move |update: &ServiceProviderUpdate, _| {
                    let sequence = match update {
                        ServiceProviderUpdate::State { sequence, .. } => *sequence,
                        _ => return,
                    };
                    sink.lock().unwrap().push(sequence);
                    if sequence == 1 {
                        state_for_listener.replace_with(json!({ "value": 3 }), &BACKGROUND_CONTEXT);
                    }
                }),
            )
            .expect("subscribe");
        state.replace_with(json!({ "value": 1 }), &BACKGROUND_CONTEXT);
        state.replace_with(json!({ "value": 2 }), &BACKGROUND_CONTEXT);
        subscription.activate();
        subscription.activate();
        assert_eq!(*lock_for_test(&received), vec![1, 2, 3]);
        provider.dispose().expect("dispose");
    }

    #[test]
    fn bounds_pending_updates_with_explicit_root_resets() {
        for count in [100u64, 101, 102, 201, 202] {
            let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
                "test.counter",
                ServiceMode::Singleton,
            )])
            .expect("provider");
            let state = FakeState::new(json!({ "value": 0 }));
            let method: TestMethod =
                Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
            provider
                .provide("test.counter", object_with_state(&state, "bump", method))
                .expect("provide");
            let updates: Arc<Mutex<Vec<(ServiceProviderUpdate, bool)>>> = Arc::new(Mutex::new(Vec::new()));
            let marked_context = crate::context::with_cancel(BACKGROUND_CONTEXT.clone()).context;
            let sink = Arc::clone(&updates);
            let subscription = provider
                .subscribe(
                    "test.counter",
                    ServiceMode::Singleton,
                    Arc::new(move |update: &ServiceProviderUpdate, context: &Context| {
                        sink.lock().unwrap().push((
                            update.clone(),
                            context.to_string().contains("WithValue"),
                        ));
                    }),
                )
                .expect("subscribe");
            for value in 1..=count {
                state.replace_with(json!({ "value": value }), &marked_context);
            }
            assert!(lock_for_test(&updates).is_empty());
            assert_eq!(
                snapshot_json(subscription.as_ref())["instances"][0]["members"][1]["sequence"],
                json!(0)
            );
            subscription.activate();
            let first = ((count - 1) / 100) * 100 + 1;
            let updates_guard = lock_for_test(&updates);
            assert_eq!(updates_guard.len() as u64, count - first + 1);
            if count > 100 {
                assert_eq!(
                    update_json(&updates_guard[0].0),
                    json!({
                        "type": "reset",
                        "snapshot": {
                            "serviceId": "test.counter",
                            "mode": "singleton",
                            "instances": [{
                                "members": [
                                    { "name": "bump", "kind": "method" },
                                    { "name": "state", "kind": "state", "sequence": first, "ops": [["r", { "value": first }]] },
                                ],
                            }],
                        },
                    })
                );
            }
            for (_, marked) in updates_guard.iter() {
                assert!(*marked, "every delivery carries the caller context");
            }
            drop(updates_guard);
            state.replace_with(json!({ "value": count + 1 }), &marked_context);
            let updates_guard = lock_for_test(&updates);
            let last = updates_guard.last().expect("final update");
            assert_eq!(update_json(&last.0)["type"], json!("state"));
            assert_eq!(update_json(&last.0)["sequence"], json!(count + 1));
            drop(updates_guard);
            futures::executor::block_on(subscription.close(None));
            provider.dispose().expect("dispose");
        }
    }

    #[test]
    fn does_not_compact_running_delivery_when_activation_overflows_reentrantly() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.counter",
            ServiceMode::Singleton,
        )])
        .expect("provider");
        let state = FakeState::new(json!({ "value": 0 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide("test.counter", object_with_state(&state, "bump", method))
            .expect("provide");
        let updates: Arc<Mutex<Vec<(String, u64)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&updates);
        let state_for_listener = Arc::clone(&state);
        let subscription = provider
            .subscribe(
                "test.counter",
                ServiceMode::Singleton,
                Arc::new(move |update: &ServiceProviderUpdate, _| {
                    match update {
                        ServiceProviderUpdate::State { sequence, .. } => {
                            sink.lock().unwrap().push(("state".into(), *sequence));
                            if *sequence == 1 {
                                for value in 3..=103 {
                                    state_for_listener.replace_with(json!({ "value": value }), &BACKGROUND_CONTEXT);
                                }
                            }
                        }
                        ServiceProviderUpdate::Reset { snapshot } => {
                            let sequence = snapshot.instances[0].members.iter().find_map(|member| match member {
                                crate::types::ServiceMemberSnapshot::State { sequence, .. } => Some(*sequence),
                                _ => None,
                            });
                            sink.lock().unwrap().push(("reset".into(), sequence.expect("state member")));
                        }
                        _ => {}
                    }
                }),
            )
            .expect("subscribe");
        state.replace_with(json!({ "value": 1 }), &BACKGROUND_CONTEXT);
        state.replace_with(json!({ "value": 2 }), &BACKGROUND_CONTEXT);
        subscription.activate();
        assert_eq!(
            *lock_for_test(&updates),
            vec![
                ("state".to_string(), 1),
                ("reset".to_string(), 102),
                ("state".to_string(), 103),
            ]
        );
        provider.dispose().expect("dispose");
    }

    #[test]
    fn suppresses_notifications_already_covered_by_an_overflow_snapshot() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.counter",
            ServiceMode::Singleton,
        )])
        .expect("provider");
        let state = FakeState::new(json!({ "value": 0 }));
        // Reentrant source listener mirrors the upstream getReplicatedStateInternals hook.
        let state_for_listener = Arc::clone(&state);
        let _source_listener = state.subscribe(Arc::new(move |_ops: &[Op], sequence: u64, _context: &Context| {
            if sequence == 101 {
                state_for_listener.replace_with(json!({ "value": 102 }), &BACKGROUND_CONTEXT);
                state_for_listener.replace_with(json!({ "value": 103 }), &BACKGROUND_CONTEXT);
            }
        }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide("test.counter", object_with_state(&state, "bump", method))
            .expect("provide");
        let updates: Arc<Mutex<Vec<(String, u64)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&updates);
        let subscription = provider
            .subscribe(
                "test.counter",
                ServiceMode::Singleton,
                Arc::new(move |update: &ServiceProviderUpdate, _| {
                    match update {
                        ServiceProviderUpdate::State { sequence, .. } => {
                            sink.lock().unwrap().push(("state".into(), *sequence));
                        }
                        ServiceProviderUpdate::Reset { snapshot } => {
                            let sequence = snapshot.instances[0].members.iter().find_map(|member| match member {
                                crate::types::ServiceMemberSnapshot::State { sequence, .. } => Some(*sequence),
                                _ => None,
                            });
                            sink.lock().unwrap().push(("reset".into(), sequence.expect("state member")));
                        }
                        _ => {}
                    }
                }),
            )
            .expect("subscribe");
        for value in 1..=101u64 {
            state.replace_with(json!({ "value": value }), &BACKGROUND_CONTEXT);
        }
        subscription.activate();
        assert_eq!(*lock_for_test(&updates), vec![("reset".to_string(), 103)]);
        state.replace_with(json!({ "value": 104 }), &BACKGROUND_CONTEXT);
        assert_eq!(
            *lock_for_test(&updates),
            vec![("reset".to_string(), 103), ("state".to_string(), 104)]
        );
        provider.dispose().expect("dispose");
    }

    #[test]
    fn preserves_lifecycle_ordering_across_subscribers_during_reentrant_publication() {
        let provider = Arc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new(
                "test.counter",
                ServiceMode::Singleton,
            )])
            .expect("provider"),
        );
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide(
                "test.counter",
                object_with_state(&FakeState::new(json!({ "value": 0 })), "bump", method),
            )
            .expect("provide");
        let first: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let second: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let provider_for_listener = Arc::clone(&provider);
        let first_sink = Arc::clone(&first);
        let first_update_count = Arc::new(Mutex::new(0usize));
        let counter = Arc::clone(&first_update_count);
        provider
            .subscribe(
                "test.counter",
                ServiceMode::Singleton,
                Arc::new(move |update: &ServiceProviderUpdate, _| {
                    first_sink.lock().unwrap().push(update_kind(update));
                    let mut count = counter.lock().unwrap();
                    *count += 1;
                    if *count == 1 {
                        let replacement_state = FakeState::new(json!({ "value": 2 }));
                        let method: TestMethod =
                            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
                        provider_for_listener
                            .replace("test.counter", object_with_state(&replacement_state, "bump", method))
                            .expect("reentrant replace");
                    }
                }),
            )
            .expect("subscribe")
            .activate();
        let second_sink = Arc::clone(&second);
        provider
            .subscribe(
                "test.counter",
                ServiceMode::Singleton,
                Arc::new(move |update: &ServiceProviderUpdate, _| {
                    second_sink.lock().unwrap().push(update_kind(update));
                }),
            )
            .expect("subscribe")
            .activate();
        let replacement_state = FakeState::new(json!({ "value": 1 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .replace("test.counter", object_with_state(&replacement_state, "bump", method))
            .expect("replace");
        assert_eq!(*lock_for_test(&first), *lock_for_test(&second));
        assert_eq!(lock_for_test(&second).len(), 2);
        assert_eq!(lock_for_test(&second)[0], "replaced");
        provider.dispose().expect("dispose");
    }

    pub(crate) fn update_kind(update: &ServiceProviderUpdate) -> String {
        match update {
            ServiceProviderUpdate::State { .. } => "state".into(),
            ServiceProviderUpdate::Reset { .. } => "reset".into(),
            ServiceProviderUpdate::Unavailable => "unavailable".into(),
            ServiceProviderUpdate::Replaced { .. } => "replaced".into(),
            ServiceProviderUpdate::Spawned { .. } => "spawned".into(),
            ServiceProviderUpdate::Closed { .. } => "closed".into(),
        }
    }

    #[test]
    fn close_during_a_callback_prevents_later_buffered_callbacks() {
        let provider = RemoteServiceProvider::new(vec![ProviderDefinition::new(
            "test.counter",
            ServiceMode::Singleton,
        )])
        .expect("provider");
        let state = FakeState::new(json!({ "value": 0 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide("test.counter", object_with_state(&state, "bump", method))
            .expect("provide");
        let received: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&received);
        // The listener closes through a shared OnceLock handle: activation borrows the
        // subscription without holding any lock the listener could re-acquire.
        let subscription: StdArc<std::sync::OnceLock<Box<dyn ServiceSubscription>>> = StdArc::new(std::sync::OnceLock::new());
        let listener_subscription = StdArc::clone(&subscription);
        let boxed = provider
            .subscribe(
                "test.counter",
                ServiceMode::Singleton,
                Arc::new(move |update: &ServiceProviderUpdate, _| {
                    sink.lock().unwrap().push(update_kind(update));
                    if let Some(subscription) = listener_subscription.get() {
                        futures::executor::block_on(subscription.close(None));
                    }
                }),
            )
            .expect("subscribe");
        assert!(subscription.set(boxed).is_ok(), "subscription stored once");
        state.replace_with(json!({ "value": 1 }), &BACKGROUND_CONTEXT);
        state.replace_with(json!({ "value": 2 }), &BACKGROUND_CONTEXT);
        subscription.get().expect("stored").activate();
        assert_eq!(lock_for_test(&received).len(), 1);
        futures::executor::block_on(subscription.get().expect("stored").close(None));
        provider.dispose().expect("dispose");
    }

    #[test]
    fn drains_terminal_updates_when_disposed_reentrantly() {
        let provider = Arc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new(
                "test.counter",
                ServiceMode::Singleton,
            )])
            .expect("provider"),
        );
        let state = FakeState::new(json!({ "value": 0 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide("test.counter", object_with_state(&state, "bump", method))
            .expect("provide");
        let received: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&received);
        let provider_for_listener = Arc::clone(&provider);
        let subscription = provider
            .subscribe(
                "test.counter",
                ServiceMode::Singleton,
                Arc::new(move |update: &ServiceProviderUpdate, _| {
                    sink.lock().unwrap().push(update_kind(update));
                    if matches!(update, ServiceProviderUpdate::State { .. }) {
                        provider_for_listener.dispose().expect("reentrant dispose");
                    }
                }),
            )
            .expect("subscribe");
        state.replace_with(json!({ "value": 1 }), &BACKGROUND_CONTEXT);
        subscription.activate();
        assert_eq!(
            *lock_for_test(&received),
            vec!["state".to_string(), "unavailable".to_string()]
        );
    }

    #[tokio::test]
    async fn endpoint_serves_catalogue_and_subscription_control_calls() {
        let provider = Arc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new(
                "test.models",
                ServiceMode::Singleton,
            )])
            .expect("provider"),
        );
        let state = FakeState::new(json!({ "selected": null, "revision": 0 }));
        let method: TestMethod =
            Box::new(|_args: Vec<JsonValue>, _context: Context| Box::pin(async { Ok(None) }));
        provider
            .provide("test.models", object_with_state(&state, "select", method))
            .expect("provide");
        let endpoint = create_remote_service_endpoint(Arc::clone(&provider));
        let published: Arc<Mutex<Vec<(String, JsonValue)>>> = Arc::new(Mutex::new(Vec::new()));
        let publish: ServiceUpdatePublisher = {
            let published = Arc::clone(&published);
            Arc::new(move |subscription_id: String, update: &ServiceProviderUpdate, _context: &Context| {
                let published = Arc::clone(&published);
                let update = update.clone();
                Box::pin(async move {
                    published
                        .lock()
                        .unwrap()
                        .push((subscription_id, serde_json::to_value(update).expect("update serializes")));
                })
            })
        };

        let catalogue = endpoint
            .invoke(
                &crate::services::wire::create_service_catalogue_call(),
                &publish,
                &BACKGROUND_CONTEXT,
            )
            .await
            .expect("catalogue");
        assert_eq!(
            catalogue,
            Some(json!([{ "serviceId": "test.models", "mode": "singleton" }]))
        );

        let snapshot = endpoint
            .invoke(
                &crate::services::wire::create_service_subscribe_call("sub-1", "test.models", ServiceMode::Singleton),
                &publish,
                &BACKGROUND_CONTEXT,
            )
            .await
            .expect("subscribe");
        assert_eq!(snapshot.expect("snapshot")["serviceId"], json!("test.models"));

        let error = err_of(endpoint
            .invoke(
                &crate::services::wire::create_service_subscribe_call("sub-1", "test.models", ServiceMode::Singleton),
                &publish,
                &BACKGROUND_CONTEXT,
            )
            .await
            );
        assert_eq!(error.to_string(), "Service subscription ID is already active");

        state.replace_with(json!({ "selected": null, "revision": 1 }), &BACKGROUND_CONTEXT);
        for _ in 0..200 {
            if !lock_for_test(&published).is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
        let published = lock_for_test(&published);
        assert_eq!(published.len(), 1);
        assert_eq!(published[0].0, "sub-1");
        assert_eq!(published[0].1["type"], json!("state"));
        drop(published);

        let result = endpoint
            .invoke(
                &crate::services::wire::create_service_unsubscribe_call("sub-1"),
                &publish,
                &BACKGROUND_CONTEXT,
            )
            .await
            .expect("unsubscribe");
        assert_eq!(result, None);
        let error = err_of(endpoint
            .invoke(
                &crate::services::wire::create_service_unsubscribe_call("sub-1"),
                &publish,
                &BACKGROUND_CONTEXT,
            )
            .await
            );
        assert_eq!(error.to_string(), "Service subscription was not found");

        endpoint.dispose().await;
        let error = err_of(endpoint
            .invoke(
                &crate::services::wire::create_service_catalogue_call(),
                &publish,
                &BACKGROUND_CONTEXT,
            )
            .await
            );
        assert_eq!(error.to_string(), "Remote service endpoint is disposed");
    }
}

