//! Port of `packages/client/src/client.ts`.
//!
//! Shared mutable client state lives in `Arc<ClientInner>` so request futures,
//! connection callbacks, and per-subscription delivery workers observe one
//! consistent view. The connection callbacks resolve the inner through a
//! `Weak` slot filled before `connect()` can run (the connection invokes no
//! callbacks before then).
//!
//! Deviations (mechanical, PORT_MAP.md rule 3):
//! - Upstream serializes async listener promises via `deliveryTail`; client
//!   listeners are synchronous `Fn(&ServiceProviderUpdate)` (the async case
//!   had no in-repo caller), and ordered delivery is provided by a
//!   per-subscription worker task instead.
//! - Upstream's static `Client.connect(options)` becomes the free async
//!   [`connect_client`] (Rust forbids an associated fn and a method sharing a
//!   name); the instance `connect()` keeps its name.
//! - Upstream runs the subscribe snapshot transform inside the request
//!   pipeline; the Rust port hydrates immediately after the request settles,
//!   which is observationally equivalent (the connection is single-state, and
//!   wire updates arriving before hydration are buffered either way).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use pi_chord::context::{abort_error, Context, BACKGROUND_CONTEXT};
use pi_chord::json::JsonValue;
use pi_chord::services::errors::{error, AnyError};
use pi_chord::{
    create_service_catalogue_call, create_service_state_decoder, create_service_subscribe_call,
    create_service_unsubscribe_call, parse_service_call, parse_service_catalogue,
    parse_wire_service_provider_update, parse_wire_service_subscription_snapshot,
    RemoteServiceTransport, ServiceCall, ServiceCatalogueEntry, ServiceMode, ServiceProviderUpdate,
    ServiceStateDecoder, UpdateListener,
};
use pi_protocol::{
    encode_client_message, is_server_id, FrameDecoderOptions, ProtocolValidationError,
};
use serde_json::json;

use crate::connection::{Connection, ConnectionCallbacks, ConnectionOptions, ServerHello};
use crate::errors::{to_error, ClientDisposedError, DisconnectedError, ServerError};
use crate::promise::{create_promise_resolvers, Promise};
use crate::types::{
    AttachmentChangeListener, ClientOptions, ConnectionChangeListener, ConnectionState,
    ConnectionStateChange, ListenerErrorHandler, RpcTarget, ServiceSubscription, SessionTarget,
    Unsubscribe,
};

type ServiceResult = Option<JsonValue>;

struct PendingRequest {
    resolve: Box<dyn FnOnce(ServiceResult) + Send>,
    reject: Box<dyn FnOnce(AnyError) + Send>,
    cleanup: Option<Box<dyn FnOnce() + Send>>,
}

struct ActiveServiceListener {
    subscription_id: String,
    target: RpcTarget,
    decoder: parking_lot::Mutex<ServiceStateDecoder>,
    /// Wire updates that arrived before the subscription snapshot hydrated them.
    queued_wire_updates: parking_lot::Mutex<Vec<JsonValue>>,
    /// Decoded updates buffered before activation.
    queued: parking_lot::Mutex<Vec<ServiceProviderUpdate>>,
    /// Ordered delivery channel; `None` once disposed.
    delivery: parking_lot::Mutex<Option<tokio::sync::mpsc::UnboundedSender<ServiceProviderUpdate>>>,
    hydrated: AtomicBool,
    ready: AtomicBool,
    disposed: AtomicBool,
}

impl ActiveServiceListener {
    fn start(&self) {
        if self.ready.swap(true, Ordering::SeqCst) {
            return;
        }
        let updates: Vec<ServiceProviderUpdate> = std::mem::take(&mut *self.queued.lock());
        if let Some(delivery) = self.delivery.lock().as_ref() {
            for update in updates {
                let _ = delivery.send(update);
            }
        }
    }
}

pub(crate) struct ClientInner {
    server_id: String,
    on_listener_error: Option<ListenerErrorHandler>,
    connection: Arc<Connection>,
    pending_requests: parking_lot::Mutex<HashMap<String, PendingRequest>>,
    connection_state_listeners: Arc<parking_lot::Mutex<Vec<ConnectionChangeListener>>>,
    attachment_listeners: Arc<parking_lot::Mutex<Vec<AttachmentChangeListener>>>,
    service_listeners: parking_lot::Mutex<HashMap<String, Arc<ActiveServiceListener>>>,
    request_sequence: AtomicU64,
    service_subscription_sequence: AtomicU64,
    hello: parking_lot::Mutex<Option<ServerHello>>,
    attachment: parking_lot::Mutex<Option<SessionTarget>>,
    disposed: AtomicBool,
    dispose_started: AtomicBool,
}

impl ClientInner {
    pub(crate) fn connected(&self) -> bool {
        self.connection.state() == ConnectionState::Connected
    }

    fn target_is_current(&self, target: &RpcTarget) -> bool {
        match target {
            RpcTarget::Server { server_id } => self
                .hello
                .lock()
                .as_ref()
                .map(|hello| hello.server_id == *server_id)
                .unwrap_or(false),
            RpcTarget::Session(session) => self
                .attachment
                .lock()
                .as_ref()
                .map(|current| *current == *session)
                .unwrap_or(false),
        }
    }

    fn report_listener_error(&self, listener_error: AnyError) {
        if let Some(on_listener_error) = &self.on_listener_error {
            let handler = Arc::clone(on_listener_error);
            // Diagnostics cannot affect protocol or transport state.
            let _ =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler(listener_error)));
        }
    }

    fn handle_message(self: &Arc<Self>, message: JsonValue) {
        match message.get("type").and_then(|value| value.as_str()) {
            Some("attachment") => self.handle_attachment_message(&message),
            Some("service_update") => self.handle_service_update_message(&message),
            _ => self.handle_response_message(message),
        }
    }

    fn handle_attachment_message(self: &Arc<Self>, message: &JsonValue) {
        let attachment = message.get("attachment");
        let is_null = attachment.map(|value| value.is_null()).unwrap_or(true);
        if !is_null {
            let belongs = attachment
                .and_then(|value| value.get("serverId"))
                .and_then(|value| value.as_str())
                .map(|server_id| server_id == self.server_id)
                .unwrap_or(false);
            if !belongs {
                self.connection.fail(Box::new(ProtocolValidationError::new(
                    "Attachment update belongs to another server",
                )));
                return;
            }
        }
        let parsed = if is_null {
            None
        } else {
            attachment.and_then(SessionTarget::from_json)
        };
        self.set_attachment(parsed);
    }

    fn handle_service_update_message(self: &Arc<Self>, message: &JsonValue) {
        let subscription_id = message
            .get("subscriptionId")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        let update = message.get("update").cloned().unwrap_or(JsonValue::Null);
        let active = {
            let listeners = self.service_listeners.lock();
            listeners.get(&subscription_id).cloned()
        };
        let Some(active) = active else { return };
        if !active.hydrated.load(Ordering::SeqCst) {
            active.queued_wire_updates.lock().push(update);
            return;
        }
        let decoded = parse_wire_service_provider_update(&update)
            .and_then(|wire| active.decoder.lock().decode_update(wire));
        let update = match decoded {
            Ok(update) => update,
            Err(decode_error) => {
                self.connection.fail(Box::new(ProtocolValidationError::new(
                    decode_error.to_string(),
                )));
                return;
            }
        };
        if active.ready.load(Ordering::SeqCst) {
            self.deliver_service_update(&active, update);
        } else {
            active.queued.lock().push(update);
        }
    }

    fn handle_response_message(self: &Arc<Self>, message: JsonValue) {
        let id = message
            .get("id")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        let pending = self.take_pending_request(&id);
        let Some(pending) = pending else {
            self.connection.fail(Box::new(ProtocolValidationError::new(
                "Response has no matching request",
            )));
            return;
        };
        let ok = message
            .get("ok")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        if !ok {
            let server_error = message
                .get("error")
                .and_then(ServerError::from_protocol)
                .unwrap_or(ServerError {
                    code: String::new(),
                    message: String::new(),
                });
            (pending.reject)(Box::new(server_error));
            return;
        }
        (pending.resolve)(message.get("result").cloned());
    }

    fn handle_connection_state_change(self: &Arc<Self>, change: &ConnectionStateChange) {
        if change.state == ConnectionState::Disconnected {
            *self.hello.lock() = None;
            self.set_attachment(None);
            let error = change.clone_error().unwrap_or_else(|| {
                Box::new(DisconnectedError::new("Client is disconnected")) as AnyError
            });
            self.reject_pending_requests(error);
            // Dropping the listeners drops the delivery senders; workers drain
            // and finish on their own.
            self.service_listeners.lock().clear();
        }
        for listener in self.connection_state_listeners.lock().iter() {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| listener(change)));
            if let Err(listener_error) = result {
                let message = listener_error
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_string())
                    .or_else(|| listener_error.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "listener panicked".to_string());
                self.report_listener_error(error(message));
            }
        }
    }

    fn take_pending_request(&self, id: &str) -> Option<PendingRequest> {
        let mut requests = self.pending_requests.lock();
        requests.remove(id).map(|mut request| {
            if let Some(cleanup) = request.cleanup.take() {
                cleanup();
            }
            request
        })
    }

    fn reject_pending_requests(&self, error: AnyError) {
        let requests: Vec<PendingRequest> = {
            let mut pending = self.pending_requests.lock();
            std::mem::take(&mut *pending).into_values().collect()
        };
        for mut request in requests {
            if let Some(cleanup) = request.cleanup.take() {
                cleanup();
            }
            (request.reject)(clone_any_error(&error));
        }
    }

    fn set_attachment(&self, attachment: Option<SessionTarget>) {
        {
            let mut current = self.attachment.lock();
            if *current == attachment {
                return;
            }
            *current = attachment.clone();
        }
        for listener in self.attachment_listeners.lock().iter() {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                listener(attachment.as_ref())
            }));
            if let Err(listener_error) = result {
                let message = listener_error
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_string())
                    .or_else(|| listener_error.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "listener panicked".to_string());
                self.report_listener_error(error(message));
            }
        }
    }

    fn deliver_service_update(
        &self,
        active: &Arc<ActiveServiceListener>,
        update: ServiceProviderUpdate,
    ) {
        if let Some(delivery) = active.delivery.lock().as_ref() {
            let _ = delivery.send(update);
        }
    }
}

/// Upstream `Client`: RPC requests, subscriptions, attachment tracking, and
/// connection lifecycle over one logical server.
#[derive(Clone)]
pub struct Client {
    inner: Arc<ClientInner>,
}

impl Client {
    pub fn new(options: ClientOptions) -> Result<Client, AnyError> {
        if !is_server_id(&options.server_id) {
            return Err(error("serverId must be a canonical lowercase UUIDv4"));
        }
        let inner_slot: Arc<OnceLock<Weak<ClientInner>>> = Arc::new(OnceLock::new());
        let callbacks = {
            let slot = Arc::clone(&inner_slot);
            let on_handshake = {
                let slot = Arc::clone(&slot);
                Arc::new(move |hello: &ServerHello| {
                    if let Some(inner) = slot.get().and_then(|weak| weak.upgrade()) {
                        *inner.hello.lock() = Some(hello.clone());
                    }
                }) as Arc<dyn Fn(&ServerHello) + Send + Sync>
            };
            let on_message = {
                let slot = Arc::clone(&slot);
                Arc::new(move |message: JsonValue| {
                    if let Some(inner) = slot.get().and_then(|weak| weak.upgrade()) {
                        inner.handle_message(message);
                    }
                }) as Arc<dyn Fn(JsonValue) + Send + Sync>
            };
            let on_state_change = {
                let slot = Arc::clone(&slot);
                Arc::new(move |change: ConnectionStateChange| {
                    if let Some(inner) = slot.get().and_then(|weak| weak.upgrade()) {
                        inner.handle_connection_state_change(&change);
                    }
                }) as Arc<dyn Fn(ConnectionStateChange) + Send + Sync>
            };
            ConnectionCallbacks {
                on_handshake,
                on_message,
                on_state_change,
            }
        };
        let connection_options = ConnectionOptions {
            transport_factory: Arc::clone(&options.transport_factory),
            server_id: options.server_id.clone(),
            max_frame_length: options.max_frame_length,
            callbacks,
        };
        let connection = Connection::new(connection_options)?;
        let inner = Arc::new(ClientInner {
            server_id: options.server_id,
            on_listener_error: options.on_listener_error,
            connection: Arc::new(connection),
            pending_requests: parking_lot::Mutex::new(HashMap::new()),
            connection_state_listeners: Arc::new(parking_lot::Mutex::new(Vec::new())),
            attachment_listeners: Arc::new(parking_lot::Mutex::new(Vec::new())),
            service_listeners: parking_lot::Mutex::new(HashMap::new()),
            request_sequence: AtomicU64::new(0),
            service_subscription_sequence: AtomicU64::new(0),
            hello: parking_lot::Mutex::new(None),
            attachment: parking_lot::Mutex::new(None),
            disposed: AtomicBool::new(false),
            dispose_started: AtomicBool::new(false),
        });
        let _ = inner_slot.set(Arc::downgrade(&inner));
        Ok(Client { inner })
    }

    pub fn disposed(&self) -> bool {
        self.inner.disposed.load(Ordering::SeqCst)
    }

    pub fn connection_state(&self) -> ConnectionState {
        self.inner.connection.state()
    }

    pub fn connected(&self) -> bool {
        self.inner.connection.state() == ConnectionState::Connected
    }

    pub fn server_id(&self) -> &str {
        &self.inner.server_id
    }

    pub fn hello(&self) -> Option<ServerHello> {
        self.inner.hello.lock().clone()
    }

    pub fn attachment(&self) -> Option<SessionTarget> {
        self.inner.attachment.lock().clone()
    }

    pub fn connect(&self) -> Promise<ServerHello> {
        if self.disposed() {
            let (resolvers, promise) = create_promise_resolvers();
            resolvers.reject(Box::new(ClientDisposedError));
            return promise;
        }
        *self.inner.hello.lock() = None;
        self.inner.connection.connect()
    }

    pub fn reconnect(&self) -> Promise<ServerHello> {
        self.connect()
    }

    pub fn disconnect(&self, reason: &str) {
        self.inner.connection.disconnect(reason);
    }

    pub fn on_connection_state_change(
        &self,
        listener: ConnectionChangeListener,
    ) -> Result<Unsubscribe, AnyError> {
        self.inner.assert_not_disposed()?;
        self.inner
            .connection_state_listeners
            .lock()
            .push(Arc::clone(&listener));
        let listeners = Arc::downgrade(&self.inner.connection_state_listeners);
        Ok(Box::new(move || {
            if let Some(listeners) = listeners.upgrade() {
                listeners
                    .lock()
                    .retain(|current| !Arc::ptr_eq(current, &listener));
            }
        }))
    }

    pub fn on_attachment_change(
        &self,
        listener: AttachmentChangeListener,
    ) -> Result<Unsubscribe, AnyError> {
        self.inner.assert_not_disposed()?;
        self.inner
            .attachment_listeners
            .lock()
            .push(Arc::clone(&listener));
        let listeners = Arc::downgrade(&self.inner.attachment_listeners);
        Ok(Box::new(move || {
            if let Some(listeners) = listeners.upgrade() {
                listeners
                    .lock()
                    .retain(|current| !Arc::ptr_eq(current, &listener));
            }
        }))
    }

    /// Invoke one low-level protocol call against an explicit routed target.
    pub fn request(
        &self,
        target: &RpcTarget,
        call: &ServiceCall,
        signal: Option<&pi_chord::context::AbortSignal>,
    ) -> Promise<ServiceResult> {
        self.inner.request_inner(target, call, signal)
    }

    pub async fn service_catalogue(
        &self,
        target: &RpcTarget,
        signal: Option<&pi_chord::context::AbortSignal>,
    ) -> Result<Vec<ServiceCatalogueEntry>, AnyError> {
        let inner = Arc::clone(&self.inner);
        let result = inner
            .request_inner(target, &create_service_catalogue_call(), signal)
            .await;
        match result {
            Ok(result) => {
                let value = result.unwrap_or(JsonValue::Null);
                match parse_service_catalogue(&value) {
                    Ok(catalogue) => Ok(catalogue),
                    Err(parse_error) => {
                        let validation_error =
                            ProtocolValidationError::new(parse_error.to_string());
                        inner.connection.fail(Box::new(ProtocolValidationError::new(
                            parse_error.to_string(),
                        )));
                        Err(Box::new(validation_error))
                    }
                }
            }
            Err(request_error) => Err(request_error),
        }
    }

    pub async fn subscribe_service(
        &self,
        target: &RpcTarget,
        service_id: &str,
        mode: ServiceMode,
        listener: Arc<dyn Fn(&ServiceProviderUpdate) + Send + Sync>,
        signal: Option<&pi_chord::context::AbortSignal>,
    ) -> Result<Arc<ServiceSubscription>, AnyError> {
        let inner = Arc::clone(&self.inner);
        let subscription_id = format!(
            "service-{}",
            inner
                .service_subscription_sequence
                .fetch_add(1, Ordering::SeqCst)
                + 1
        );
        let active = Arc::new(ActiveServiceListener {
            subscription_id: subscription_id.clone(),
            target: target.clone(),
            decoder: parking_lot::Mutex::new(create_service_state_decoder()),
            queued_wire_updates: parking_lot::Mutex::new(Vec::new()),
            queued: parking_lot::Mutex::new(Vec::new()),
            delivery: parking_lot::Mutex::new(None),
            hydrated: AtomicBool::new(false),
            ready: AtomicBool::new(false),
            disposed: AtomicBool::new(false),
        });
        inner
            .service_listeners
            .lock()
            .insert(subscription_id.clone(), Arc::clone(&active));

        // Ordered delivery worker (upstream `deliveryTail` promise chain).
        let (delivery, mut delivery_rx) =
            tokio::sync::mpsc::unbounded_channel::<ServiceProviderUpdate>();
        *active.delivery.lock() = Some(delivery);
        let worker_listener = Arc::clone(&listener);
        let weak_active = Arc::downgrade(&active);
        tokio::spawn(async move {
            while let Some(update) = delivery_rx.recv().await {
                let Some(active) = weak_active.upgrade() else {
                    break;
                };
                if active.disposed.load(Ordering::SeqCst) {
                    break;
                }
                worker_listener(&update);
            }
        });

        let call = create_service_subscribe_call(&subscription_id, service_id, mode);
        let result = inner.request_inner(target, &call, signal).await;
        let result = match result {
            Ok(result) => result,
            Err(request_error) => {
                remove_service_listener(&inner, &subscription_id, &active);
                return Err(request_error);
            }
        };
        if service_listener_missing(&inner, &subscription_id, &active) {
            return Err(Box::new(DisconnectedError::new("Client is disconnected")));
        }

        // Hydrate the snapshot (upstream runs this as the request transform).
        let hydration = (|| -> Result<pi_chord::types::ServiceSubscriptionSnapshot, AnyError> {
            let Some(value) = result else {
                return Err(error("Invalid service subscription snapshot"));
            };
            let wire = parse_wire_service_subscription_snapshot(&value)?;
            let snapshot = active.decoder.lock().decode_snapshot(wire)?;
            active.hydrated.store(true, Ordering::SeqCst);
            let wire_updates: Vec<JsonValue> =
                std::mem::take(&mut *active.queued_wire_updates.lock());
            let mut queued = active.queued.lock();
            for wire_update in wire_updates {
                let wire = parse_wire_service_provider_update(&wire_update)?;
                queued.push(active.decoder.lock().decode_update(wire)?);
            }
            Ok(snapshot)
        })();
        let snapshot = match hydration {
            Ok(snapshot) => snapshot,
            Err(hydration_error) => {
                remove_service_listener(&inner, &subscription_id, &active);
                let validation_error = ProtocolValidationError::new(hydration_error.to_string());
                inner.connection.fail(Box::new(ProtocolValidationError::new(
                    hydration_error.to_string(),
                )));
                return Err(Box::new(validation_error));
            }
        };

        let start_active = Arc::clone(&active);
        let dispose_active = Arc::clone(&active);
        let dispose_inner = Arc::downgrade(&inner);
        Ok(Arc::new(ServiceSubscription::new(
            subscription_id,
            target.clone(),
            snapshot,
            Box::new(move || start_active.start()),
            Box::new(move || {
                let active = Arc::clone(&dispose_active);
                let inner = dispose_inner.clone();
                Box::pin(async move {
                    if let Some(inner) = inner.upgrade() {
                        dispose_service_listener(&inner, &active).await;
                    }
                })
            }),
        )))
    }

    pub async fn dispose(&self) {
        let inner = &self.inner;
        if inner.dispose_started.swap(true, Ordering::SeqCst) {
            return;
        }
        inner.disposed.store(true, Ordering::SeqCst);
        inner.reject_pending_requests(Box::new(ClientDisposedError));
        inner.connection.disconnect("Client is disposed");
        *inner.hello.lock() = None;
        inner.set_attachment(None);
        inner.connection_state_listeners.lock().clear();
        inner.attachment_listeners.lock().clear();
        inner.service_listeners.lock().clear();
    }
}

/// Upstream `Client.connect(options)`: constructs a client and connects,
/// disposing on handshake failure.
pub async fn connect_client(options: ClientOptions) -> Result<Client, AnyError> {
    let client = Client::new(options)?;
    let connect = client.connect();
    match connect.await {
        Ok(_) => Ok(client),
        Err(error) => {
            client.dispose().await;
            Err(error)
        }
    }
}

/// Removes the listener only when it is still the registered one.
fn remove_service_listener(
    inner: &Arc<ClientInner>,
    subscription_id: &str,
    active: &Arc<ActiveServiceListener>,
) {
    let mut listeners = inner.service_listeners.lock();
    if listeners
        .get(subscription_id)
        .is_some_and(|current| Arc::ptr_eq(current, active))
    {
        listeners.remove(subscription_id);
    }
}

/// Upstream post-subscribe check: the registration vanished (disconnect or
/// disposal) between the response and activation.
fn service_listener_missing(
    inner: &Arc<ClientInner>,
    subscription_id: &str,
    active: &Arc<ActiveServiceListener>,
) -> bool {
    !inner
        .service_listeners
        .lock()
        .get(subscription_id)
        .is_some_and(|current| Arc::ptr_eq(current, active))
}

async fn dispose_service_listener(inner: &Arc<ClientInner>, active: &Arc<ActiveServiceListener>) {
    if active.disposed.swap(true, Ordering::SeqCst) {
        return;
    }
    {
        let mut listeners = inner.service_listeners.lock();
        if listeners
            .get(&active.subscription_id)
            .is_some_and(|current| Arc::ptr_eq(current, active))
        {
            listeners.remove(&active.subscription_id);
        }
    }
    if inner.connected() && inner.target_is_current(&active.target) {
        let call = create_service_unsubscribe_call(&active.subscription_id);
        let _ = inner.request_inner(&active.target, &call, None).await;
    }
    // Drop the sender so the worker drains in-flight updates and finishes.
    active.delivery.lock().take();
    active.queued_wire_updates.lock().clear();
    active.queued.lock().clear();
}

impl ClientInner {
    fn assert_not_disposed(&self) -> Result<(), AnyError> {
        if self.disposed.load(Ordering::SeqCst) {
            return Err(Box::new(ClientDisposedError));
        }
        Ok(())
    }

    fn request_inner(
        self: &Arc<Self>,
        target: &RpcTarget,
        call: &ServiceCall,
        signal: Option<&pi_chord::context::AbortSignal>,
    ) -> Promise<ServiceResult> {
        let (resolvers, promise) = create_promise_resolvers::<ServiceResult>();
        if self.disposed.load(Ordering::SeqCst) {
            resolvers.reject(Box::new(ClientDisposedError));
            return promise;
        }
        if self.connection.state() != ConnectionState::Connected {
            resolvers.reject(Box::new(DisconnectedError::new("Client is disconnected")));
            return promise;
        }
        if let Some(signal) = signal {
            if signal.aborted() {
                resolvers.reject(abort_error(signal));
                return promise;
            }
        }
        let id = format!(
            "request-{}",
            self.request_sequence.fetch_add(1, Ordering::SeqCst) + 1
        );
        let target_json = target.to_json();

        // Abort wiring: reject with the abort reason and send a cancel frame.
        let aborted = Arc::new(AtomicBool::new(false));
        let mut cleanup: Option<Box<dyn FnOnce() + Send>> = None;
        if let Some(signal) = signal {
            let resolvers = resolvers.clone();
            let listener_signal = signal.clone();
            let connection = Arc::clone(&self.connection);
            let request_id = id.clone();
            let target_json = target_json.clone();
            let max_frame_length = self.connection.max_frame_length();
            let aborted = Arc::clone(&aborted);
            let remove = signal.add_listener(move || {
                if aborted.swap(true, Ordering::SeqCst) {
                    return;
                }
                resolvers.reject(abort_error(&listener_signal));
                send_cancel(&connection, &request_id, &target_json, max_frame_length);
            });
            cleanup = Some(Box::new(remove));
        }

        self.pending_requests.lock().insert(
            id.clone(),
            PendingRequest {
                resolve: {
                    let resolvers = resolvers.clone();
                    Box::new(move |result| resolvers.resolve(result))
                },
                reject: {
                    let resolvers = resolvers.clone();
                    Box::new(move |reason| resolvers.reject(reason))
                },
                cleanup,
            },
        );

        let call_json = match serde_json::to_value(call) {
            Ok(call_json) => call_json,
            Err(serialize_error) => {
                self.reject_request(
                    &id,
                    to_error(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        serialize_error.to_string(),
                    ))),
                );
                return promise;
            }
        };
        let frame = match parse_service_call(&call_json) {
            Ok(parsed_call) => encode_client_message(
                &json!({ "type": "request", "id": id, "target": target_json, "call": parsed_call }),
                Some(&FrameDecoderOptions {
                    max_frame_length: Some(self.connection.max_frame_length()),
                }),
            ),
            Err(parse_error) => {
                self.reject_request(&id, parse_error);
                return promise;
            }
        };
        match frame {
            Ok(frame) => {
                if let Err(send_error) = self.connection.send(frame) {
                    self.reject_request(&id, send_error);
                    return promise;
                }
            }
            Err(encode_error) => {
                self.reject_request(&id, Box::new(encode_error));
                return promise;
            }
        }
        if aborted.load(Ordering::SeqCst) {
            send_cancel(
                &self.connection,
                &id,
                &target_json,
                self.connection.max_frame_length(),
            );
        }
        promise
    }

    fn reject_request(&self, id: &str, reason: AnyError) {
        if let Some(pending) = self.take_pending_request(id) {
            (pending.reject)(reason);
        }
    }
}

fn send_cancel(
    connection: &Arc<Connection>,
    id: &str,
    target: &JsonValue,
    max_frame_length: usize,
) {
    if connection.state() != ConnectionState::Connected {
        return;
    }
    let frame = encode_client_message(
        &json!({ "type": "cancel", "id": id, "target": target }),
        Some(&FrameDecoderOptions {
            max_frame_length: Some(max_frame_length),
        }),
    );
    match frame {
        Ok(frame) => {
            if let Err(send_error) = connection.send(frame) {
                connection.fail(send_error);
            }
        }
        Err(encode_error) => connection.fail(Box::new(encode_error)),
    }
}

/// Adapts a lazily resolved routed client target to a Chord service transport
/// (upstream `createClientServiceTransport`).
pub fn create_client_service_transport(
    client: &Client,
    get_target: impl Fn() -> Option<RpcTarget> + Send + Sync + 'static,
) -> Arc<dyn RemoteServiceTransport> {
    Arc::new(ClientServiceTransport {
        client: Arc::clone(&client.inner),
        get_target: Arc::new(get_target),
    })
}

struct ClientServiceTransport {
    client: Arc<ClientInner>,
    get_target: Arc<dyn Fn() -> Option<RpcTarget> + Send + Sync>,
}

impl ClientServiceTransport {
    fn target(&self) -> Result<RpcTarget, AnyError> {
        (self.get_target)().ok_or_else(|| error("Remote service target is unavailable"))
    }
}

#[async_trait::async_trait]
impl RemoteServiceTransport for ClientServiceTransport {
    async fn invoke(
        &self,
        call: &ServiceCall,
        context: &Context,
    ) -> Result<Option<JsonValue>, AnyError> {
        let target = self.target()?;
        self.client
            .request_inner(&target, call, context.abort_signal().as_ref())
            .await
    }

    async fn subscribe(
        &self,
        service_id: &str,
        mode: ServiceMode,
        listener: UpdateListener,
        context: &Context,
    ) -> Result<Box<dyn pi_chord::types::ServiceSubscription>, AnyError> {
        let target = self.target()?;
        let client = Client {
            inner: Arc::clone(&self.client),
        };
        let listener = Arc::clone(&listener);
        let subscription = client
            .subscribe_service(
                &target,
                service_id,
                mode,
                Arc::new(move |update| listener(update, &BACKGROUND_CONTEXT)),
                context.abort_signal().as_ref(),
            )
            .await?;
        Ok(Box::new(ChordSubscriptionAdapter(subscription)))
    }
}

/// Presents the client subscription as a chord `ServiceSubscription`.
struct ChordSubscriptionAdapter(Arc<ServiceSubscription>);

#[async_trait::async_trait]
impl pi_chord::types::ServiceSubscription for ChordSubscriptionAdapter {
    fn snapshot(&self) -> &pi_chord::types::ServiceSubscriptionSnapshot {
        &self.0.snapshot
    }

    fn activate(&self) {
        self.0.start();
    }

    async fn close(&self, _context: Option<Context>) {
        self.0.dispose().await;
    }
}

/// Deep clone for `AnyError` across the client error types (the connection
/// failure error is both rejected into pending requests and reported through
/// the state-change callback, mirroring upstream's shared error object).
pub(crate) fn clone_any_error(error: &AnyError) -> AnyError {
    use crate::errors::DisconnectedError;
    if let Some(disconnected) = error.downcast_ref::<DisconnectedError>() {
        return Box::new(DisconnectedError {
            message: disconnected.message.clone(),
            cause: disconnected.cause.as_ref().map(clone_any_error),
        });
    }
    if let Some(server_error) = error.downcast_ref::<ServerError>() {
        return Box::new(server_error.clone());
    }
    if let Some(validation_error) = error.downcast_ref::<ProtocolValidationError>() {
        return Box::new(ProtocolValidationError::new(validation_error.to_string()));
    }
    if let Some(disposed) = error.downcast_ref::<ClientDisposedError>() {
        return Box::new(*disposed);
    }
    if let Some(io_error) = error.downcast_ref::<std::io::Error>() {
        return Box::new(std::io::Error::new(io_error.kind(), io_error.to_string()));
    }
    Box::new(pi_chord::services::errors::ChordError {
        message: error.to_string(),
    })
}

#[cfg(test)]
mod tests {
    //! Port of `packages/client/test/client.test.ts`.

    use std::time::Duration;

    use pi_chord::context::{AbortController, AbortReason, SharedError};
    use pi_chord::services::errors::AnyError;
    use pi_protocol::{
        encode_cbor, encode_frame, encode_server_message, ProtocolValidationError, PROTOCOL_VERSION,
    };
    use serde_json::json;

    use super::*;
    use crate::test_support::{MemoryByteServer, DEFAULT_SERVER_ID};
    use crate::transport::{ByteTransport, ByteTransportFactory, ByteTransportHandlers};

    const SERVER_ID: &str = DEFAULT_SERVER_ID;

    fn server_target() -> RpcTarget {
        RpcTarget::Server {
            server_id: SERVER_ID.to_string(),
        }
    }

    async fn connect_test_client(server: &Arc<MemoryByteServer>) -> Client {
        connect_expect(server, SERVER_ID).await
    }

    async fn connect_test_client_as(
        server: &Arc<MemoryByteServer>,
        expected_server_id: &str,
    ) -> Result<Client, AnyError> {
        let server = Arc::clone(server);
        let factory: ByteTransportFactory = Arc::new(move |handlers| {
            let transport = server.connect(handlers);
            Box::pin(async move { Ok(transport) })
        });
        connect_client(ClientOptions {
            server_id: expected_server_id.to_string(),
            transport_factory: factory,
            max_frame_length: None,
            on_listener_error: None,
        })
        .await
    }

    async fn connect_expect(server: &Arc<MemoryByteServer>, expected_server_id: &str) -> Client {
        connect_test_client_as(server, expected_server_id)
            .await
            .expect("client connects")
    }

    async fn attach_test_client(client: &Client, server: &Arc<MemoryByteServer>, session_id: &str) {
        let expected = server.messages().len() + 1;
        let attaching = client.request(
            &server_target(),
            &ServiceCall {
                service_id: "pi.session-management".into(),
                instance: None,
                member: "attach".into(),
                args: vec![json!(session_id)],
            },
            None,
        );
        server.wait_for_messages(expected).await;
        let request = server.messages().last().cloned().expect("attach request");
        assert_eq!(
            request.get("type").and_then(|value| value.as_str()),
            Some("request")
        );
        server.send(&json!({
			"type": "attachment",
			"attachment": { "serverId": SERVER_ID, "sessionId": session_id, "attachmentId": format!("attachment-{session_id}") },
		}));
        server
            .send(&json!({ "type": "response", "id": request["id"], "ok": true, "result": null }));
        attaching.await.expect("attach resolves");
    }

    async fn wait_until(mut condition: impl FnMut() -> bool) {
        for _ in 0..400 {
            if condition() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("condition not reached");
    }

    #[test]
    fn requires_a_canonical_uuidv4_server_identity() {
        let factory: ByteTransportFactory =
            Arc::new(|_handlers| Box::pin(async { Err(error("rejected")) }));
        let result = Client::new(ClientOptions {
            server_id: "invalid-server".into(),
            transport_factory: factory,
            max_frame_length: None,
            on_listener_error: None,
        });
        let Err(result_error) = result else {
            panic!("expected serverId rejection")
        };
        assert!(result_error.to_string().contains("serverId"));
    }

    #[tokio::test]
    async fn connects_only_to_the_expected_logical_server() {
        let matching = MemoryByteServer::new();
        let client = connect_test_client(&matching).await;
        assert_eq!(client.hello().unwrap().server_id, SERVER_ID);
        client.dispose().await;

        let wrong = MemoryByteServer::with_server_id("00000000-0000-4000-8000-000000000002");
        let failure = connect_test_client_as(&wrong, SERVER_ID)
            .await
            .err()
            .expect("wrong server rejected");
        assert!(failure.downcast_ref::<ProtocolValidationError>().is_some());
    }

    #[tokio::test]
    async fn updates_attachment_state_from_out_of_band_server_routing() {
        let server = MemoryByteServer::new();
        let client = connect_test_client(&server).await;
        let changes: Arc<parking_lot::Mutex<Vec<Option<String>>>> = Arc::default();
        let recorded = Arc::clone(&changes);
        let _unsubscribe = client
            .on_attachment_change(Arc::new(move |attachment| {
                recorded
                    .lock()
                    .push(attachment.map(|target| target.session_id.clone()));
            }))
            .expect("listener registers");

        attach_test_client(&client, &server, "session-1").await;
        let attachment = client.attachment().expect("attached");
        assert_eq!(attachment.session_id, "session-1");
        assert_eq!(attachment.attachment_id, "attachment-session-1");
        let messages = server.messages();
        assert_eq!(
            messages[1].get("type").and_then(|value| value.as_str()),
            Some("request")
        );
        assert_eq!(messages[1]["target"]["serverId"], json!(SERVER_ID));
        assert_eq!(
            messages[1]["call"],
            json!({ "serviceId": "pi.session-management", "member": "attach", "args": ["session-1"] })
        );

        server.send(&json!({ "type": "attachment", "attachment": null }));
        assert!(client.attachment().is_none());
        assert_eq!(*changes.lock(), vec![Some("session-1".into()), None]);
        client.dispose().await;
    }

    #[tokio::test]
    async fn buffers_service_updates_until_the_subscription_snapshot_arrives() {
        let server = MemoryByteServer::new();
        let client = connect_test_client(&server).await;
        attach_test_client(&client, &server, "session-1").await;
        assert!(client.attachment().is_some());
        let transport = create_client_service_transport(&client, {
            let client = client.clone();
            move || client.attachment().map(RpcTarget::Session)
        });
        let updates: Arc<parking_lot::Mutex<Vec<ServiceProviderUpdate>>> = Arc::default();
        let recorded = Arc::clone(&updates);
        // Rust futures are lazy; drive the subscribe concurrently (upstream
        // starts the promise eagerly).
        let opening = tokio::spawn({
            let transport = Arc::clone(&transport);
            let recorded = Arc::clone(&recorded);
            async move {
                transport
                    .subscribe(
                        "pi.models",
                        pi_chord::ServiceMode::Singleton,
                        Arc::new(move |update, _context| recorded.lock().push(update.clone())),
                        &pi_chord::context::BACKGROUND_CONTEXT,
                    )
                    .await
                    .map(Arc::from)
                    .expect("subscription opens")
            }
        });

        server.wait_for_messages(3).await;
        let messages = server.messages();
        assert_eq!(messages[2]["call"]["serviceId"], json!("$chord.service"));
        assert_eq!(messages[2]["call"]["member"], json!("subscribe"));
        assert_eq!(
            messages[2]["call"]["args"],
            json!(["service-1", "pi.models", "singleton"])
        );

        server.send(&json!({
			"type": "service_update",
			"subscriptionId": "service-1",
			"update": { "type": "state", "member": "state", "sequence": 1, "ops": [["s", ["revision"], 1]] },
		}));
        tokio::task::yield_now().await;
        assert!(updates.lock().is_empty());

        server.send(&json!({
			"type": "response",
			"id": "request-2",
			"ok": true,
			"result": {
				"serviceId": "pi.models",
				"mode": "singleton",
				"instances": [{ "members": [{ "name": "state", "kind": "state", "sequence": 0, "ops": [["r", { "revision": 0 }]] }] }],
			},
		}));
        let subscription: Arc<dyn pi_chord::ServiceSubscription> = opening.await.unwrap();
        assert!(updates.lock().is_empty());
        server.send(&json!({
			"type": "service_update",
			"subscriptionId": "closed-subscription",
			"update": { "type": "state", "member": "state", "sequence": 99, "ops": [["s", 99, 99]] },
		}));
        tokio::task::yield_now().await;
        assert!(client.connected());
        let snapshot = subscription.snapshot();
        assert_eq!(snapshot.instances.len(), 1);
        assert_eq!(
            serde_json::to_value(&snapshot.instances[0].members).unwrap(),
            json!([{ "name": "state", "kind": "state", "sequence": 0, "ops": [["r", { "revision": 0 }]] }]),
        );
        subscription.activate();
        wait_until(|| {
            updates
                .lock()
                .iter()
                .filter(|update| matches!(update, ServiceProviderUpdate::State { .. }))
                .count()
                == 1
        })
        .await;
        server.send(&json!({
			"type": "service_update",
			"subscriptionId": "service-1",
			"update": { "type": "state", "member": "state", "sequence": 2, "ops": [["s", ["revision"], 2]] },
		}));
        server.send(&json!({
            "type": "service_update",
            "subscriptionId": "service-1",
            "update": {
                "type": "state",
                "member": "state",
                "sequence": 3,
                "ops": [["#", 0, ["revision"]], ["s", 0, 3]],
            },
        }));
        wait_until(|| updates.lock().len() == 3).await;
        let ServiceProviderUpdate::State { ops, .. } = &updates.lock()[2] else {
            panic!("state update")
        };
        assert_eq!(
            serde_json::to_value(ops).unwrap(),
            json!([["s", ["revision"], 3]])
        );

        let subscription: Arc<dyn pi_chord::ServiceSubscription> = Arc::from(subscription);
        let disposing_subscription = Arc::clone(&subscription);
        let disposing = tokio::spawn(async move {
            disposing_subscription
                .close(Some(pi_chord::context::BACKGROUND_CONTEXT.clone()))
                .await;
        });
        server.wait_for_messages(4).await;
        let messages = server.messages();
        assert_eq!(messages[3]["call"]["serviceId"], json!("$chord.service"));
        assert_eq!(messages[3]["call"]["member"], json!("unsubscribe"));
        assert_eq!(messages[3]["call"]["args"], json!(["service-1"]));
        server.send(&json!({ "type": "response", "id": "request-3", "ok": true }));
        disposing.await.unwrap();
        client.dispose().await;
    }

    #[tokio::test]
    async fn correlates_out_of_order_generic_service_responses() {
        let server = MemoryByteServer::new();
        let client = connect_test_client(&server).await;
        let first = client.request(
            &server_target(),
            &ServiceCall {
                service_id: "test".into(),
                instance: None,
                member: "first".into(),
                args: vec![],
            },
            None,
        );
        let second = client.request(
            &server_target(),
            &ServiceCall {
                service_id: "test".into(),
                instance: None,
                member: "second".into(),
                args: vec![],
            },
            None,
        );
        server.wait_for_messages(3).await;
        server.send(
            &json!({ "type": "response", "id": "request-2", "ok": true, "result": "second" }),
        );
        server
            .send(&json!({ "type": "response", "id": "request-1", "ok": true, "result": "first" }));
        assert_eq!(first.await.unwrap(), Some(json!("first")));
        assert_eq!(second.await.unwrap(), Some(json!("second")));
        client.dispose().await;
    }

    #[tokio::test]
    async fn exposes_bounded_server_errors() {
        let server = MemoryByteServer::new();
        let client = connect_test_client(&server).await;
        let pending = client.request(
            &server_target(),
            &ServiceCall {
                service_id: "test".into(),
                instance: None,
                member: "missing".into(),
                args: vec![],
            },
            None,
        );
        server.wait_for_messages(2).await;
        server.send(&json!({
            "type": "response",
            "id": "request-1",
            "ok": false,
            "error": { "code": "session_not_found", "message": "Unknown session" },
        }));
        let failure = pending.await.err().expect("request rejects");
        let Some(server_error) = failure.downcast_ref::<ServerError>() else {
            panic!("ServerError expected")
        };
        assert_eq!(server_error.code, "session_not_found");
        client.dispose().await;
    }

    #[tokio::test]
    async fn does_not_send_a_pre_aborted_untyped_rpc_request() {
        let server = MemoryByteServer::new();
        let client = connect_test_client(&server).await;
        let controller = AbortController::new();
        let reason: SharedError = Arc::new(std::io::Error::other("already cancelled"));
        controller.abort(AbortReason::Error(Arc::clone(&reason)));

        let pending = client.request(
            &server_target(),
            &ServiceCall {
                service_id: "test".into(),
                instance: None,
                member: "noop".into(),
                args: vec![],
            },
            Some(&controller.signal()),
        );
        let failure = pending.await.err().expect("aborted request rejects");
        assert_eq!(failure.to_string(), "already cancelled");
        assert_eq!(server.messages().len(), 1);
        client.dispose().await;
    }

    #[tokio::test]
    async fn cancels_one_untyped_rpc_request_without_disconnecting() {
        let server = MemoryByteServer::new();
        let client = connect_test_client(&server).await;
        let controller = AbortController::new();
        let reason: SharedError = Arc::new(std::io::Error::other("stop this request"));
        let pending = client.request(
            &server_target(),
            &ServiceCall {
                service_id: "test".into(),
                instance: None,
                member: "mutate".into(),
                args: vec![json!({ "value": 42 })],
            },
            Some(&controller.signal()),
        );
        server.wait_for_messages(2).await;
        let messages = server.messages();
        assert_eq!(messages[1]["type"], json!("request"));
        assert_eq!(messages[1]["id"], json!("request-1"));
        assert_eq!(messages[1]["target"], json!({ "serverId": SERVER_ID }));
        assert_eq!(
            messages[1]["call"],
            json!({ "serviceId": "test", "member": "mutate", "args": [{ "value": 42 }] })
        );

        controller.abort(AbortReason::Error(Arc::clone(&reason)));
        let failure = pending.await.err().expect("aborted request rejects");
        assert_eq!(failure.to_string(), "stop this request");
        server.wait_for_messages(3).await;
        let messages = server.messages();
        assert_eq!(
            messages[2],
            json!({ "type": "cancel", "id": "request-1", "target": { "serverId": SERVER_ID } })
        );
        server.send(&json!({
            "type": "response",
            "id": "request-1",
            "ok": false,
            "error": { "code": "cancelled", "message": "cancelled" },
        }));
        assert!(client.connected());
        client.dispose().await;
    }

    #[tokio::test]
    async fn rejects_pending_requests_after_disconnect_or_disposal() {
        let server = MemoryByteServer::new();
        let client = connect_test_client(&server).await;
        let pending = client.request(
            &server_target(),
            &ServiceCall {
                service_id: "test".into(),
                instance: None,
                member: "pending".into(),
                args: vec![],
            },
            None,
        );
        server.disconnect();
        let failure = pending.await.err().expect("pending rejects");
        assert!(failure.downcast_ref::<DisconnectedError>().is_some());
        client.dispose().await;
        let disposed = client.request(
            &server_target(),
            &ServiceCall {
                service_id: "test".into(),
                instance: None,
                member: "disposed".into(),
                args: vec![],
            },
            None,
        );
        let failure = disposed.await.err().expect("disposed request rejects");
        assert!(failure.downcast_ref::<ClientDisposedError>().is_some());
    }

    #[tokio::test]
    async fn rejects_server_data_delivered_before_the_client_hello_is_sent() {
        let close_count = Arc::new(AtomicU64::new(0));
        let send_count = Arc::new(AtomicU64::new(0));
        let transport_send = Arc::clone(&send_count);
        let transport_close = Arc::clone(&close_count);
        let factory: ByteTransportFactory = Arc::new(move |handlers| {
            let transport_send = Arc::clone(&transport_send);
            let transport_close = Arc::clone(&transport_close);
            handlers.on_data.clone()(
                encode_server_message(
                    &json!({ "type": "hello", "version": PROTOCOL_VERSION, "serverId": SERVER_ID }),
                    None,
                )
                .unwrap(),
            );
            Box::pin(async move {
                Ok(Box::new(CountingTransport {
                    on_send: transport_send,
                    on_close: transport_close,
                }) as Box<dyn ByteTransport>)
            })
        });
        let client = Client::new(ClientOptions {
            server_id: SERVER_ID.into(),
            transport_factory: factory,
            max_frame_length: None,
            on_listener_error: None,
        })
        .unwrap();
        let connect = client.connect();
        let failure = connect.await.err().expect("connect rejects");
        let Some(validation_error) = failure.downcast_ref::<ProtocolValidationError>() else {
            panic!("ProtocolValidationError expected")
        };
        assert_eq!(
            validation_error.to_string(),
            "Received server data before the client hello was sent"
        );
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
        assert_eq!(send_count.load(Ordering::SeqCst), 0);
        assert_eq!(close_count.load(Ordering::SeqCst), 1);
    }

    struct CountingTransport {
        on_send: Arc<AtomicU64>,
        on_close: Arc<AtomicU64>,
    }

    #[async_trait::async_trait]
    impl ByteTransport for CountingTransport {
        async fn send(&self, _chunk: Vec<u8>) -> Result<(), AnyError> {
            self.on_send.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn close(&self) {
            self.on_close.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn rejects_typed_handshake_errors_and_closes_the_transport() {
        let close_count = Arc::new(AtomicU64::new(0));
        let handlers_slot: Arc<parking_lot::Mutex<Option<ByteTransportHandlers>>> = Arc::default();
        let factory_handlers = Arc::clone(&handlers_slot);
        let transport_close = Arc::clone(&close_count);
        let factory: ByteTransportFactory = Arc::new(move |handlers| {
            let transport_close = Arc::clone(&transport_close);
            *factory_handlers.lock() = Some(handlers.clone());
            Box::pin(async move {
                Ok(Box::new(HandshakeErrorTransport {
                    handlers,
                    on_close: transport_close,
                }) as Box<dyn ByteTransport>)
            })
        });
        let client = Client::new(ClientOptions {
            server_id: SERVER_ID.into(),
            transport_factory: factory,
            max_frame_length: None,
            on_listener_error: None,
        })
        .unwrap();
        let failure = client.connect().await.err().expect("connect rejects");
        let Some(server_error) = failure.downcast_ref::<ServerError>() else {
            panic!("ServerError expected")
        };
        assert_eq!(server_error.code, "version");
        assert_eq!(server_error.to_string(), "Unsupported protocol version");
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
        assert_eq!(close_count.load(Ordering::SeqCst), 1);
    }

    struct HandshakeErrorTransport {
        handlers: ByteTransportHandlers,
        on_close: Arc<AtomicU64>,
    }

    #[async_trait::async_trait]
    impl ByteTransport for HandshakeErrorTransport {
        async fn send(&self, _chunk: Vec<u8>) -> Result<(), AnyError> {
            (self.handlers.on_data)(
                encode_server_message(
                    &json!({
                        "type": "hello_error",
                        "error": { "code": "version", "message": "Unsupported protocol version" },
                    }),
                    None,
                )
                .unwrap(),
            );
            Ok(())
        }

        fn close(&self) {
            self.on_close.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn rejects_pending_requests_and_reconnects_through_a_fresh_transport() {
        let first = MemoryByteServer::new();
        let second = MemoryByteServer::new();
        let connection = Arc::new(AtomicU64::new(0));
        let first_for_factory = Arc::clone(&first);
        let second_for_factory = Arc::clone(&second);
        let connection_for_factory = Arc::clone(&connection);
        let factory: ByteTransportFactory = Arc::new(move |handlers| {
            let index = connection_for_factory.fetch_add(1, Ordering::SeqCst);
            let server = if index == 0 {
                Arc::clone(&first_for_factory)
            } else {
                Arc::clone(&second_for_factory)
            };
            Box::pin(async move { Ok(server.connect(handlers)) })
        });
        let client = Client::new(ClientOptions {
            server_id: SERVER_ID.into(),
            transport_factory: factory,
            max_frame_length: None,
            on_listener_error: None,
        })
        .unwrap();
        let states: Arc<parking_lot::Mutex<Vec<String>>> = Arc::default();
        let recorded = Arc::clone(&states);
        let _unsubscribe = client
            .on_connection_state_change(Arc::new(move |change| {
                recorded.lock().push(change.state.as_str().to_string())
            }))
            .unwrap();
        client.connect().await.expect("first connect");
        attach_test_client(&client, &first, "session-1").await;
        let target = client.attachment().expect("attachment");
        let pending = client.request(
            &RpcTarget::Session(target),
            &ServiceCall {
                service_id: "test.session".into(),
                instance: None,
                member: "run".into(),
                args: vec![],
            },
            None,
        );
        first.wait_for_messages(3).await;
        let messages = first.messages();
        assert_eq!(messages[2]["call"]["serviceId"], json!("test.session"));
        assert_eq!(messages[2]["call"]["member"], json!("run"));
        first.disconnect();

        let failure = pending.await.err().expect("pending rejects");
        assert!(failure.downcast_ref::<DisconnectedError>().is_some());
        let hello = client.reconnect().await.expect("reconnect");
        assert_eq!(hello.server_id, SERVER_ID);
        assert_eq!(connection.load(Ordering::SeqCst), 2);
        assert!(client.connected());
        assert_eq!(second.messages().len(), 1);
        assert_eq!(
            *states.lock(),
            vec![
                "connecting",
                "connected",
                "disconnected",
                "connecting",
                "connected"
            ]
        );
        client.dispose().await;
    }

    #[tokio::test]
    async fn reports_transport_failures_without_leaving_requests_pending() {
        let server = MemoryByteServer::new();
        let client = connect_test_client(&server).await;
        let pending = client.request(
            &server_target(),
            &ServiceCall {
                service_id: "test".into(),
                instance: None,
                member: "pending".into(),
                args: vec![],
            },
            None,
        );
        server.wait_for_messages(2).await;
        server.error(Box::new(std::io::Error::other("read failed")));

        let failure = pending.await.err().expect("pending rejects");
        let Some(disconnected) = failure.downcast_ref::<DisconnectedError>() else {
            panic!("DisconnectedError expected")
        };
        assert_eq!(disconnected.message, "read failed");
        let cause = disconnected.cause.as_ref().expect("cause present");
        assert_eq!(cause.to_string(), "read failed");
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    }

    #[tokio::test]
    async fn disconnects_on_invalid_or_truncated_server_framing() {
        let invalid_server = MemoryByteServer::new();
        let invalid_client = connect_test_client(&invalid_server).await;
        let payload = encode_cbor(
            &pi_protocol::CborValue::from(
                &json!({ "type": "response", "id": "unknown", "ok": true, "result": 1 }),
            ),
            None,
        )
        .unwrap();
        invalid_server.send_raw(encode_frame(&payload).unwrap());
        assert_eq!(
            invalid_client.connection_state(),
            ConnectionState::Disconnected
        );

        let truncated_server = MemoryByteServer::new();
        let truncated_client = connect_test_client(&truncated_server).await;
        let pending = truncated_client.request(
            &server_target(),
            &ServiceCall {
                service_id: "test".into(),
                instance: None,
                member: "pending".into(),
                args: vec![],
            },
            None,
        );
        truncated_server.wait_for_messages(2).await;
        truncated_server.send_raw(vec![0, 0, 0, 2, 1]);
        truncated_server.disconnect();

        let failure = pending.await.err().expect("pending rejects");
        let Some(validation_error) = failure.downcast_ref::<ProtocolValidationError>() else {
            panic!("ProtocolValidationError expected")
        };
        assert!(validation_error
            .to_string()
            .to_lowercase()
            .contains("truncated"));
        assert_eq!(
            truncated_client.connection_state(),
            ConnectionState::Disconnected
        );
    }

    #[tokio::test]
    async fn disconnects_when_a_response_has_no_matching_request() {
        let server = MemoryByteServer::new();
        let client = connect_test_client(&server).await;
        server.send(
            &json!({ "type": "response", "id": "unknown-request", "ok": true, "result": [] }),
        );

        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
        wait_until(|| server.client_close_count() == 1).await;
    }
}
