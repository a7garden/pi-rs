//! Port of `packages/server/src/server.ts`.
//!
//! Upstream's evented per-connection promises become spawned tokio tasks with
//! a shared [`ConnectionState`] under a `parking_lot` lock; the handshake
//! timeout `NodeJS.Timeout` becomes an aborted task (cancellation races are
//! resolved under the state lock via `handshake_settled`). Deviations are
//! listed in the crate docs.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::types::into_any_error;
use pi_chord::context::{with_abort_signal, BACKGROUND_CONTEXT, TODO_CONTEXT};
use pi_chord::services::errors::RemoteServiceError;
use pi_chord::JsonValue;
use pi_chord::{
    decode_service_control_call, parse_service_call, parse_service_subscription_snapshot,
    AbortController, AbortReason, ChordError, ServiceControlCall,
};
use pi_protocol::{
    encode_server_message, is_server_id, is_supported_protocol_version, FrameDecoderOptions,
    ProtocolValidationError, DEFAULT_MAX_FRAME_LENGTH, PROTOCOL_VERSION,
};
use tokio::sync::watch;

use crate::connection::{
    is_terminal_connection, ActiveRequest, ByteConnection, ByteConnectionAcceptor,
    ByteConnectionHandler, ConnectionStage, ConnectionState,
};
use crate::errors::{ServerError, INTERNAL_SERVER_ERROR_MESSAGE};
use crate::listener::ServerListener;
use crate::session_router::{ClientId, SessionRouter};
use crate::types::{
    aggregate, any, AnyError, ConnectionCountCallback, ErrorCallback, RoutedServerPresentation,
    ServerHost, ServerOptions, ServicePublish,
};
use crate::wire;

pub(crate) const DEFAULT_HANDSHAKE_TIMEOUT_MS: u64 = 5_000;
pub(crate) const MAX_UINT32: usize = 0xffff_ffff;
pub(crate) const MAX_TIMER_DELAY_MS: u64 = 2_147_483_647;

/// One routed service call's publish bookkeeping (upstream closures over
/// `pendingUpdates`/`subscriptionReady`).
struct PublishTracker {
    pending: tokio::sync::Mutex<Vec<pi_chord::ServiceProviderUpdate>>,
    ready: AtomicBool,
}

/// Protocol error tuple (upstream `ProtocolError` shape).
type ProtocolErrorTuple = (String, String);

#[derive(Clone)]
pub struct Server {
    inner: Arc<ServerInner>,
}

impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server")
            .field("serverId", &self.inner.server_id)
            .finish()
    }
}

pub(crate) struct ServerInner {
    host: Arc<dyn ServerHost>,
    listeners: Vec<Arc<dyn ServerListener>>,
    server_id: String,
    max_frame_length: usize,
    handshake_timeout: Duration,
    on_connection_count_changed: Option<ConnectionCountCallback>,
    on_error: Option<ErrorCallback>,
    sessions: Arc<SessionRouter>,
    /// Shared with the router (upstream `this.closing` closure).
    closing: Arc<AtomicBool>,
    state: parking_lot::Mutex<ServerCoreState>,
    start_lock: parking_lot::Mutex<()>,
    start_inflight: parking_lot::Mutex<Option<watch::Sender<bool>>>,
    started: AtomicBool,
    close_lock: tokio::sync::Mutex<()>,
    close_result: parking_lot::Mutex<Option<Arc<Result<(), AnyError>>>>,
    closed_result: parking_lot::Mutex<Option<Arc<Result<(), AnyError>>>>,
    closed_notify: tokio::sync::Notify,
}

struct ServerCoreState {
    connections: HashMap<u64, Arc<Conn>>,
}

/// Per-connection server bookkeeping (upstream `ConnectionState` plus its
/// timeout handle and router identity).
pub(crate) struct Conn {
    pub(crate) id: ClientId,
    pub(crate) state: parking_lot::Mutex<ConnectionState>,
    pub(crate) handshake_timeout: parking_lot::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl Conn {
    fn cancel_handshake_timeout(&self) {
        if let Some(handle) = self.handshake_timeout.lock().take() {
            handle.abort();
        }
    }
}

impl Server {
    /// Upstream `new Server(host, options)`; construction-time option
    /// validation throws synchronously upstream, so this returns `Result`.
    pub fn new(host: Arc<dyn ServerHost>, options: ServerOptions) -> Result<Server, AnyError> {
        let (max_frame_length, handshake_timeout_ms) = resolve_options(&options)?;
        let closing = Arc::new(AtomicBool::new(false));
        let server_id = options.server_id.clone();
        let on_error = options.on_error.clone();
        let reporter: Arc<dyn Fn(AnyError) + Send + Sync> = {
            let on_error = on_error.clone();
            Arc::new(move |error: AnyError| {
                if let Some(on_error) = &on_error {
                    // Error observers cannot affect server state.
                    on_error(error);
                }
            })
        };
        let sessions = Arc::new(SessionRouter::new(
            host.clone(),
            server_id.clone(),
            closing.clone(),
            reporter,
        ));
        let inner = Arc::new(ServerInner {
            host,
            listeners: options.listeners,
            server_id,
            max_frame_length,
            handshake_timeout: Duration::from_millis(handshake_timeout_ms),
            on_connection_count_changed: options.on_connection_count_changed,
            on_error,
            sessions,
            closing,
            state: parking_lot::Mutex::new(ServerCoreState {
                connections: HashMap::new(),
            }),
            start_lock: parking_lot::Mutex::new(()),
            start_inflight: parking_lot::Mutex::new(None),
            started: AtomicBool::new(false),
            close_lock: tokio::sync::Mutex::new(()),
            close_result: parking_lot::Mutex::new(None),
            closed_result: parking_lot::Mutex::new(None),
            closed_notify: tokio::sync::Notify::new(),
        });
        // Install the attachment publisher (upstream routes it through
        // `sendMessage` on the connection state).
        let publisher = {
            let inner = Arc::clone(&inner);
            Arc::new(
                move |client: ClientId,
                      attachment: Option<JsonValue>,
                      _context: pi_chord::Context| {
                    let inner = Arc::clone(&inner);
                    Box::pin(async move {
                        let conn = inner.state.lock().connections.get(&client).cloned();
                        if let Some(conn) = conn {
                            inner
                                .send_message(&conn, wire::attachment(attachment))
                                .await;
                        }
                        Ok(())
                    }) as crate::types::MaybeFuture<Result<(), AnyError>>
                },
            ) as crate::session_router::PublishAttachment
        };
        inner.sessions.set_publish_attachment(publisher);
        Ok(Server { inner })
    }

    pub fn server_id(&self) -> &str {
        &self.inner.server_id
    }

    /// Upstream `start()`. Upstream runs `startInternal()` synchronously up to
    /// its first await, so the started/starting/closing decisions are made
    /// without holding a lock across startup (only the decision is serialized).
    pub async fn start(&self) -> Result<(), AnyError> {
        {
            let _guard = self.inner.start_lock.lock();
            if self.inner.started.load(Ordering::SeqCst) {
                return Err(error_message("Server is already started"));
            }
            if self.inner.start_inflight.lock().is_some() {
                return Err(error_message("Server is already starting"));
            }
            if self.inner.closing.load(Ordering::SeqCst) {
                return Err(error_message("Server is closing or closed"));
            }
            let (tx, _rx) = watch::channel(false);
            *self.inner.start_inflight.lock() = Some(tx);
        }
        let result = self.inner.start_internal().await;
        if let Some(tx) = self.inner.start_inflight.lock().take() {
            let _ = tx.send(true);
        }
        result
    }

    /// Upstream `close()`.
    pub async fn close(&self) -> Result<(), AnyError> {
        let _guard = self.inner.close_lock.lock().await;
        if let Some(existing) = &*self.inner.close_result.lock() {
            return (**existing).clone();
        }
        self.inner.closing.store(true, Ordering::SeqCst);
        let result = Arc::new(self.inner.close_internal().await);
        *self.inner.close_result.lock() = Some(result.clone());
        (*result).clone()
    }

    /// Upstream `closed` promise.
    pub async fn closed(&self) -> Result<(), AnyError> {
        loop {
            let notified = self.inner.closed_notify.notified();
            if let Some(result) = self.inner.closed_result.lock().clone() {
                return (*result).clone();
            }
            notified.await;
        }
    }

    /// Upstream `accept(connection)`; returns the connection handler the
    /// transport feeds events into.
    pub fn accept(&self, connection: Arc<dyn ByteConnection>) -> Arc<dyn ByteConnectionHandler> {
        self.inner.accept(connection)
    }

    /// Test determinism hook: clients parked joining the in-flight open.
    #[cfg(test)]
    pub(crate) fn debug_opening_joiners(&self, session_id: &str) -> usize {
        self.inner.sessions.opening_joiners(session_id)
    }

    /// Test hook: a `'static` acceptor bound to this server.
    #[cfg(test)]
    pub(crate) fn acceptor(&self) -> crate::connection::ByteConnectionAcceptor {
        let inner = Arc::clone(&self.inner);
        Arc::new(move |connection| inner.accept(connection))
    }
}

impl ServerInner {
    fn is_closing(&self) -> bool {
        self.closing.load(Ordering::SeqCst)
    }

    fn report_error(&self, error: AnyError) {
        if let Some(on_error) = &self.on_error {
            // Error observers cannot affect server state.
            on_error(error);
        }
    }

    fn notify_connection_count_changed(&self) {
        if let Some(callback) = &self.on_connection_count_changed {
            callback(self.state.lock().connections.len());
        }
    }

    pub(crate) fn accept(
        self: &Arc<Self>,
        connection: Arc<dyn ByteConnection>,
    ) -> Arc<dyn ByteConnectionHandler> {
        if self.is_closing() {
            let inner = Arc::clone(self);
            tokio::spawn(async move { inner.close_connection(&connection, None).await });
            return Arc::new(ClosedHandler {
                server: Arc::clone(self),
            });
        }
        let (client_id, _queue) = self.sessions.register_client();
        let conn = Arc::new(Conn {
            id: client_id,
            state: parking_lot::Mutex::new(ConnectionState::new(connection, self.max_frame_length)),
            handshake_timeout: parking_lot::Mutex::new(None),
        });
        self.state
            .lock()
            .connections
            .insert(client_id, Arc::clone(&conn));
        self.notify_connection_count_changed();
        // Handshake timeout (upstream setTimeout + unref).
        let timeout_server = Arc::clone(self);
        let timeout_conn = Arc::clone(&conn);
        let handle = tokio::spawn(async move {
            tokio::time::sleep(timeout_server.handshake_timeout).await;
            timeout_server.handshake_timeout_fired(&timeout_conn).await;
        });
        *conn.handshake_timeout.lock() = Some(handle);
        Arc::new(ConnHandler {
            server: Arc::clone(self),
            conn,
        })
    }

    async fn start_internal(self: &Arc<Self>) -> Result<(), AnyError> {
        let acceptor: ByteConnectionAcceptor = {
            let inner = Arc::clone(self);
            Arc::new(move |connection| inner.accept(connection))
        };
        let mut started: Vec<Arc<dyn ServerListener>> = Vec::new();
        let start_result: Result<(), AnyError> = async {
            for listener in &self.listeners {
                listener.start(acceptor.clone()).await?;
                started.push(Arc::clone(listener));
            }
            Ok(())
        }
        .await;
        match start_result {
            Ok(()) => {
                self.started.store(true, Ordering::SeqCst);
                Ok(())
            }
            Err(error) => {
                self.closing.store(true, Ordering::SeqCst);
                let mut cleanup_errors: Vec<AnyError> = Vec::new();
                let joins: Vec<_> = started
                    .iter()
                    .map(|listener| {
                        let listener = Arc::clone(listener);
                        tokio::spawn(async move { listener.close().await })
                    })
                    .collect();
                for join in joins {
                    if let Ok(Err(cleanup_error)) = join.await {
                        cleanup_errors.push(cleanup_error);
                    }
                }
                if let Err(cleanup_error) = self.close_server_state().await {
                    cleanup_errors.push(cleanup_error);
                }
                if !cleanup_errors.is_empty() {
                    let mut reasons = vec![error];
                    reasons.extend(cleanup_errors);
                    let failure: AnyError = aggregate("Server startup and cleanup failed", reasons);
                    self.settle_closed(Err(Arc::clone(&failure)));
                    return Err(failure);
                }
                self.settle_closed(Err(Arc::clone(&error)));
                Err(error)
            }
        }
    }

    async fn close_internal(self: &Arc<Self>) -> Result<(), AnyError> {
        // Await an in-flight startup, ignoring its result (upstream
        // `starting.catch(() => {})`).
        let starting = self.start_inflight.lock().as_ref().map(|tx| tx.subscribe());
        if let Some(mut rx) = starting {
            while !*rx.borrow_and_update() {
                if rx.changed().await.is_err() {
                    break;
                }
            }
        }
        let mut errors: Vec<AnyError> = Vec::new();
        let joins: Vec<_> = self
            .listeners
            .iter()
            .map(|listener| {
                let listener = Arc::clone(listener);
                tokio::spawn(async move { listener.close().await })
            })
            .collect();
        for join in joins {
            if let Ok(Err(error)) = join.await {
                errors.push(error);
            }
        }
        if let Err(error) = self.close_server_state().await {
            errors.push(error);
        }
        self.started.store(false, Ordering::SeqCst);
        if !errors.is_empty() {
            let failure = if errors.len() == 1 {
                errors.pop().expect("one error")
            } else {
                aggregate("Server shutdown failed", errors)
            };
            self.settle_closed(Err(Arc::clone(&failure)));
            return Err(failure);
        }
        self.settle_closed(Ok(()));
        Ok(())
    }

    async fn close_server_state(self: &Arc<Self>) -> Result<(), AnyError> {
        let connections: Vec<Arc<Conn>> = {
            let state = self.state.lock();
            let connections: Vec<Arc<Conn>> = state.connections.values().cloned().collect();
            for conn in &connections {
                conn.state.lock().stage = ConnectionStage::Closing;
                conn.cancel_handshake_timeout();
            }
            connections
        };
        for conn in &connections {
            let connection = conn.state.lock().connection.clone();
            self.close_connection(&connection, None).await;
        }
        for conn in &connections {
            self.disconnect(conn);
        }
        let cleanup = self.sessions.close(BACKGROUND_CONTEXT.clone()).await;
        self.state.lock().connections.clear();
        cleanup
    }

    /// Handshake timeout fire (upstream setTimeout callback).
    async fn handshake_timeout_fired(self: &Arc<Self>, conn: &Arc<Conn>) {
        {
            let mut state = conn.state.lock();
            if state.handshake_settled || state.disconnected || is_terminal_connection(&state) {
                return;
            }
            // Claim the closing stage so a completing handshake cannot race.
            state.stage = ConnectionStage::Closing;
        }
        self.fail_protocol_claimed(
            conn,
            (
                "invalid_request".to_string(),
                "Handshake timeout".to_string(),
            ),
        )
        .await;
    }

    async fn fail_protocol(self: &Arc<Self>, conn: &Arc<Conn>, error: ProtocolErrorTuple) {
        {
            let mut state = conn.state.lock();
            if state.disconnected
                || matches!(
                    state.stage,
                    ConnectionStage::Closing | ConnectionStage::Closed
                )
            {
                return;
            }
            state.stage = ConnectionStage::Closing;
        }
        self.fail_protocol_claimed(conn, error).await;
    }

    /// `failProtocol` body after the closing stage has been claimed.
    async fn fail_protocol_claimed(self: &Arc<Self>, conn: &Arc<Conn>, error: ProtocolErrorTuple) {
        conn.cancel_handshake_timeout();
        let message = wire::hello_error(&error.0, &error.1);
        let mut final_frame: Option<Vec<u8>> = None;
        match encode_server_message(&message, Some(&self.frame_options())) {
            Ok(frame) => final_frame = Some(frame),
            Err(encode_error) => self.report_error(any(encode_error)),
        }
        let connection = conn.state.lock().connection.clone();
        self.close_connection(&connection, final_frame).await;
        self.disconnect(conn);
    }

    fn frame_options(&self) -> FrameDecoderOptions {
        FrameDecoderOptions {
            max_frame_length: Some(self.max_frame_length),
        }
    }

    fn receive(self: &Arc<Self>, conn: &Arc<Conn>, chunk: &[u8]) {
        let messages: Vec<JsonValue> = {
            let mut state = conn.state.lock();
            if is_terminal_connection(&state) {
                return;
            }
            match state.decoder.push(chunk) {
                Ok(messages) => messages,
                Err(error) => {
                    drop(state);
                    let server = Arc::clone(self);
                    let conn = Arc::clone(conn);
                    tokio::spawn(async move {
                        let error = server.to_protocol_error(crate::types::any(error));
                        server.fail_protocol(&conn, error).await
                    });
                    return;
                }
            }
        };
        for message in messages {
            {
                let state = conn.state.lock();
                if is_terminal_connection(&state) {
                    return;
                }
            }
            self.dispatch_message(conn, message);
        }
    }

    fn dispatch_message(self: &Arc<Self>, conn: &Arc<Conn>, message: JsonValue) {
        let stage = conn.state.lock().stage;
        let message_type = message
            .as_object()
            .and_then(|o| o.get("type"))
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .to_string();
        if stage == ConnectionStage::AwaitingHello {
            if message_type != "hello" {
                let server = Arc::clone(self);
                let conn = Arc::clone(conn);
                tokio::spawn(async move {
                    server
                        .fail_protocol(
                            &conn,
                            (
                                "invalid_request".to_string(),
                                "The first client message must be hello".to_string(),
                            ),
                        )
                        .await
                });
                return;
            }
            conn.state.lock().stage = ConnectionStage::Handshaking;
            let handshake_server = Arc::clone(self);
            let handshake_conn = Arc::clone(conn);
            let (done_tx, done_rx) = tokio::sync::watch::channel(false);
            tokio::spawn(async move {
                handshake_server
                    .finish_handshake(&handshake_conn, message)
                    .await;
                let _ = done_tx.send(true);
            });
            conn.state.lock().handshake_done = Some(done_rx);
            return;
        }

        if message_type == "hello" {
            let server = Arc::clone(self);
            let conn = Arc::clone(conn);
            tokio::spawn(async move {
                server
                    .fail_protocol(
                        &conn,
                        (
                            "invalid_request".to_string(),
                            "hello may only be sent as the first message".to_string(),
                        ),
                    )
                    .await
            });
            return;
        }

        if stage == ConnectionStage::Ready {
            if message_type == "cancel" {
                self.handle_cancel(conn, &message);
            } else {
                let server = Arc::clone(self);
                let conn = Arc::clone(conn);
                tokio::spawn(async move { server.handle_request(&conn, message).await });
            }
            return;
        }
        if stage != ConnectionStage::Handshaking {
            return;
        }
        let handshake_done = conn.state.lock().handshake_done.clone();
        let Some(mut handshake_done) = handshake_done else {
            return;
        };
        let server = Arc::clone(self);
        let conn = Arc::clone(conn);
        tokio::spawn(async move {
            while !*handshake_done.borrow_and_update() {
                if handshake_done.changed().await.is_err() {
                    break;
                }
            }
            let ready = {
                let state = conn.state.lock();
                state.stage == ConnectionStage::Ready && !state.disconnected
            };
            if !ready {
                return;
            }
            if message
                .as_object()
                .and_then(|o| o.get("type"))
                .and_then(JsonValue::as_str)
                == Some("cancel")
            {
                server.handle_cancel(&conn, &message);
            } else {
                server.handle_request(&conn, message).await;
            }
        });
    }

    async fn finish_handshake(self: &Arc<Self>, conn: &Arc<Conn>, hello: JsonValue) {
        let version = wire::hello_version(&hello);
        if !version.map(is_supported_protocol_version).unwrap_or(false) {
            let version_text = match version {
                Some(version) => version.to_string(),
                None => format!("{:?}", version),
            };
            self.fail_protocol(
                conn,
                (
                    "version".to_string(),
                    format!(
                        "Unsupported protocol version {version_text}; expected {PROTOCOL_VERSION}"
                    ),
                ),
            )
            .await;
            return;
        }

        {
            let state = conn.state.lock();
            if self.is_closing()
                || state.disconnected
                || state.stage != ConnectionStage::Handshaking
                || state.connection.closed()
            {
                return;
            }
        }
        let presentation = Arc::new(ServerPresentation {
            server: Arc::clone(self),
            conn: Arc::clone(conn),
        });
        let services = match self
            .host
            .server_services()
            .attach_client(presentation, TODO_CONTEXT.clone())
            .await
        {
            Ok(services) => services,
            Err(error) => {
                self.fail_protocol(conn, self.to_protocol_error(error))
                    .await;
                return;
            }
        };
        let should_release = {
            let state = conn.state.lock();
            self.is_closing()
                || state.disconnected
                || state.stage != ConnectionStage::Handshaking
                || state.connection.closed()
        };
        if should_release {
            let _ = services.release(TODO_CONTEXT.clone()).await;
            return;
        }
        conn.state.lock().server_services = Some(Arc::clone(&services));
        let sent = self
            .send_message(conn, wire::server_hello(&self.server_id))
            .await;
        let mut state = conn.state.lock();
        state.handshake_settled = true;
        if sent && !state.disconnected && state.stage == ConnectionStage::Handshaking {
            state.stage = ConnectionStage::Ready;
            drop(state);
            conn.cancel_handshake_timeout();
        }
    }

    fn handle_cancel(self: &Arc<Self>, conn: &Arc<Conn>, envelope: &JsonValue) {
        let target = envelope
            .as_object()
            .and_then(|o| o.get("target"))
            .cloned()
            .unwrap_or(JsonValue::Null);
        if wire::target_server_id(&target) != self.server_id {
            return;
        }
        let id = wire::envelope_id(envelope);
        let state = conn.state.lock();
        if let Some(active) = state.active_requests.get(id) {
            if same_target(&active.target, &target) {
                active
                    .controller
                    .abort(AbortReason::Message("RPC request cancelled".to_string()));
            }
        }
    }

    async fn handle_request(self: &Arc<Self>, conn: &Arc<Conn>, envelope: JsonValue) {
        let Some(object) = envelope.as_object() else {
            return;
        };
        let object = object.clone();
        let id = object
            .get("id")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .to_string();
        let target = object.get("target").cloned().unwrap_or(JsonValue::Null);
        let duplicate = conn.state.lock().active_requests.contains_key(&id);
        if duplicate {
            self.send_message(
                conn,
                wire::response_error(&id, "invalid_request", "Request ID is already active"),
            )
            .await;
            return;
        }
        let call_json = object.get("call").cloned().unwrap_or(JsonValue::Null);
        let call = match parse_service_call(&call_json) {
            Ok(call) => call,
            Err(_) => {
                self.send_message(
                    conn,
                    wire::response_error(&id, "invalid_request", "Invalid service call"),
                )
                .await;
                return;
            }
        };
        let controller = AbortController::new();
        let active = ActiveRequest {
            controller: controller.clone(),
            target: target.clone(),
        };
        conn.state.lock().active_requests.insert(id.clone(), active);
        let context = with_abort_signal(controller.signal(), TODO_CONTEXT.clone());
        let control = decode_service_control_call(&call);
        let subscribing: Option<String> = match &control {
            Some(ServiceControlCall::Subscribe {
                subscription_id, ..
            }) => Some(subscription_id.clone()),
            _ => None,
        };
        let unknown_member_message =
            format!("Unknown service member {}.{}", call.service_id, call.member);
        let tracker = Arc::new(PublishTracker {
            pending: tokio::sync::Mutex::new(Vec::new()),
            ready: AtomicBool::new(subscribing.is_none()),
        });
        let publish: ServicePublish = {
            let server = Arc::clone(self);
            let conn = Arc::clone(conn);
            let tracker = Arc::clone(&tracker);
            let subscribing = subscribing.clone();
            Arc::new(move |subscription_id, update, _context| {
                let server = Arc::clone(&server);
                let conn = Arc::clone(&conn);
                let tracker = Arc::clone(&tracker);
                let subscribing = subscribing.clone();
                Box::pin(async move {
                    if let Some(expected) = &subscribing {
                        if &subscription_id == expected && !tracker.ready.load(Ordering::SeqCst) {
                            tracker.pending.lock().await.push(update);
                            return Ok(());
                        }
                    }
                    server
                        .send_service_update(&conn, &subscription_id, update)
                        .await
                })
            })
        };
        let responded = false;
        let installed_subscription_encoder = false;
        let is_session_target = wire::target_session(&target).is_some();
        let session_call = call.clone();
        let result: Result<Option<JsonValue>, AnyError> = async {
            if wire::target_server_id(&target) != self.server_id {
                return Err(any(ServerError::wrong_server()));
            }
            if let Some(subscription_id) = &subscribing {
                let exists = conn
                    .state
                    .lock()
                    .service_state_encoders
                    .read()
                    .contains_key(subscription_id);
                if exists {
                    return Err(any(ProtocolValidationError::new(format!(
                        "Duplicate service subscription {subscription_id}"
                    ))));
                }
            }
            if is_session_target {
                self.sessions
                    .execute_service_call(
                        session_call,
                        &target,
                        conn.id,
                        publish.clone(),
                        context.clone(),
                    )
                    .await
            } else {
                let services = conn.state.lock().server_services.clone();
                match services {
                    Some(services) => services.invoke_service(call, publish, context).await,
                    None => Err(any(ProtocolValidationError::new(unknown_member_message))),
                }
            }
        }
        .await;

        match result {
            Ok(mut result) => {
                if let Some(subscription_id) = &subscribing {
                    let encoded: Result<
                        Option<(JsonValue, pi_chord::ServiceStateEncoder)>,
                        AnyError,
                    > = (|| {
                        let Some(value) = result else {
                            return Err(any(ProtocolValidationError::new(
                                "Service subscription did not return a snapshot",
                            )));
                        };
                        let snapshot =
                            parse_service_subscription_snapshot(&value).map_err(into_any_error)?;
                        let mut state_encoder = pi_chord::create_service_state_encoder();
                        let wire_snapshot = state_encoder
                            .encode_snapshot(snapshot)
                            .map_err(into_any_error)?;
                        let value = serde_json::to_value(wire_snapshot).map_err(|error| {
                            any(ChordError {
                                message: error.to_string(),
                            })
                        })?;
                        Ok(Some((value, state_encoder)))
                    })();
                    match encoded {
                        Ok(Some((value, state_encoder))) => {
                            result = Some(value);
                            conn.state
                                .lock()
                                .service_state_encoders
                                .write()
                                .insert(subscription_id.clone(), state_encoder);
                        }
                        Ok(None) => unreachable!(),
                        Err(error) => {
                            self.finish_request_with_error(
                                conn,
                                &id,
                                &controller,
                                &subscribing,
                                installed_subscription_encoder,
                                responded,
                                error,
                            )
                            .await;
                            conn.state.lock().active_requests.remove(&id);
                            return;
                        }
                    }
                } else if let Some(ServiceControlCall::Unsubscribe { subscription_id }) = &control {
                    conn.state
                        .lock()
                        .service_state_encoders
                        .write()
                        .remove(subscription_id);
                }
                let message = match result {
                    None => wire::response_ok_void(&id),
                    Some(value) => wire::response_ok_result(&id, value),
                };
                self.send_message(conn, message).await;
                if let Some(subscription_id) = &subscribing {
                    loop {
                        let update = {
                            let mut pending = tracker.pending.lock().await;
                            if pending.is_empty() {
                                break;
                            }
                            pending.remove(0)
                        };
                        let _ = self
                            .send_service_update(conn, subscription_id, update)
                            .await;
                    }
                    tracker.ready.store(true, Ordering::SeqCst);
                }
            }
            Err(error) => {
                self.finish_request_with_error(
                    conn,
                    &id,
                    &controller,
                    &subscribing,
                    installed_subscription_encoder,
                    responded,
                    error,
                )
                .await;
            }
        }
        conn.state.lock().active_requests.remove(&id);
    }

    async fn finish_request_with_error(
        self: &Arc<Self>,
        conn: &Arc<Conn>,
        id: &str,
        controller: &AbortController,
        subscribing: &Option<String>,
        installed_subscription_encoder: bool,
        responded: bool,
        error: AnyError,
    ) {
        if let Some(subscription_id) = subscribing {
            if installed_subscription_encoder && !responded {
                conn.state
                    .lock()
                    .service_state_encoders
                    .write()
                    .remove(subscription_id);
            }
        }
        if responded {
            self.report_error(error);
            let connection = conn.state.lock().connection.clone();
            self.close_connection(&connection, None).await;
            self.disconnect(conn);
            return;
        }
        let (code, message) = if controller.signal().aborted() {
            ("cancelled".to_string(), "RPC request cancelled".to_string())
        } else {
            self.to_protocol_error(error)
        };
        self.send_message(conn, wire::response_error(id, &code, &message))
            .await;
    }

    fn transport_closed(self: &Arc<Self>, conn: &Arc<Conn>) {
        let decode_error: Option<pi_protocol::ProtocolValidationError> = {
            let mut state = conn.state.lock();
            if !state.disconnected && state.stage != ConnectionStage::Closing {
                state.decoder.end().err()
            } else {
                None
            }
        };
        if let Some(error) = decode_error {
            self.report_error(any(error));
        }
        self.disconnect(conn);
    }

    fn disconnect(self: &Arc<Self>, conn: &Arc<Conn>) {
        {
            let mut state = conn.state.lock();
            if state.disconnected {
                return;
            }
            state.disconnected = true;
            state.stage = ConnectionStage::Closed;
        }
        conn.cancel_handshake_timeout();
        {
            let mut state = conn.state.lock();
            for active in state.active_requests.values() {
                active
                    .controller
                    .abort(AbortReason::Message("Client disconnected".to_string()));
            }
            state.active_requests.clear();
            state.service_state_encoders.write().clear();
        }
        let removed = self.state.lock().connections.remove(&conn.id).is_some();
        if removed {
            self.notify_connection_count_changed();
        }
        let server_services = conn.state.lock().server_services.take();
        let sessions = Arc::clone(&self.sessions);
        let client = conn.id;
        let reporter = {
            let inner = Arc::clone(self);
            move |error: AnyError| inner.report_error(error)
        };
        tokio::spawn(async move {
            let disconnect_result = sessions.disconnect(client, TODO_CONTEXT.clone()).await;
            let release_result = async {
                match server_services {
                    Some(services) => services.release(TODO_CONTEXT.clone()).await,
                    None => Ok(()),
                }
            }
            .await;
            // Upstream allSettled: report each rejection.
            for result in [disconnect_result, release_result] {
                if let Err(error) = result {
                    reporter(error);
                }
            }
        });
    }

    async fn send_service_update(
        self: &Arc<Self>,
        conn: &Arc<Conn>,
        subscription_id: &str,
        update: pi_chord::ServiceProviderUpdate,
    ) -> Result<(), AnyError> {
        let encoded = {
            let state = conn.state.lock();
            let mut encoders = state.service_state_encoders.write();
            let Some(state_encoder) = encoders.get_mut(subscription_id) else {
                return Ok(());
            };
            state_encoder.encode_update(update).map_err(into_any_error)
        };
        // Upstream returns early when no encoder is installed for the
        // subscription.
        let Ok(encoded) = encoded else { return Ok(()) };
        let value = serde_json::to_value(encoded).map_err(|error| {
            any(ChordError {
                message: error.to_string(),
            })
        })?;
        self.send_message(conn, wire::service_update(subscription_id, value))
            .await;
        Ok(())
    }

    async fn send_message(self: &Arc<Self>, conn: &Arc<Conn>, message: JsonValue) -> bool {
        {
            let state = conn.state.lock();
            if state.disconnected || state.connection.closed() {
                return false;
            }
        }
        let frame = match encode_server_message(&message, Some(&self.frame_options())) {
            Ok(frame) => frame,
            Err(error) => {
                self.report_error(any(error));
                let connection = conn.state.lock().connection.clone();
                self.close_connection(&connection, None).await;
                self.disconnect(conn);
                return false;
            }
        };
        let connection = conn.state.lock().connection.clone();
        match connection.send(frame).await {
            Ok(()) => true,
            Err(error) => {
                self.report_error(error);
                let connection = conn.state.lock().connection.clone();
                self.close_connection(&connection, None).await;
                self.disconnect(conn);
                false
            }
        }
    }

    async fn close_connection(
        &self,
        connection: &Arc<dyn ByteConnection>,
        final_chunk: Option<Vec<u8>>,
    ) {
        if let Err(error) = connection.close(final_chunk).await {
            self.report_error(error);
        }
    }

    fn to_protocol_error(&self, error: AnyError) -> ProtocolErrorTuple {
        if let Some(server_error) = error.as_ref().downcast_ref::<ServerError>() {
            return (
                server_error.code.as_str().to_string(),
                server_error.message.clone(),
            );
        }
        if let Some(remote_error) = error.as_ref().downcast_ref::<RemoteServiceError>() {
            return (
                remote_error.code.as_str().to_string(),
                remote_error.message.clone(),
            );
        }
        if let Some(validation_error) = error.as_ref().downcast_ref::<ProtocolValidationError>() {
            return ("invalid_request".to_string(), validation_error.to_string());
        }
        self.report_error(error);
        (
            "internal_error".to_string(),
            INTERNAL_SERVER_ERROR_MESSAGE.to_string(),
        )
    }

    fn settle_closed(&self, result: Result<(), AnyError>) {
        let mut settled = self.closed_result.lock();
        if settled.is_some() {
            return;
        }
        *settled = Some(Arc::new(result));
        drop(settled);
        self.closed_notify.notify_waiters();
    }
}

fn error_message(message: &str) -> AnyError {
    Arc::new(ChordError {
        message: message.to_string(),
    })
}

fn same_target(left: &JsonValue, right: &JsonValue) -> bool {
    if wire::target_server_id(left) != wire::target_server_id(right) {
        return false;
    }
    match (wire::target_session(left), wire::target_session(right)) {
        (Some(left), Some(right)) => left == right,
        (None, None) => true,
        _ => false,
    }
}

/// Upstream `resolveOptions`.
fn resolve_options(options: &ServerOptions) -> Result<(usize, u64), AnyError> {
    if !is_server_id(&options.server_id) {
        return Err(into_any_error(pi_chord::services::errors::type_error(
            "serverId must be a canonical lowercase UUIDv4",
        )));
    }
    // Upstream also rejects a missing `listeners` array; in Rust the field is
    // required by construction.
    let max_frame_length = options.max_frame_length.unwrap_or(DEFAULT_MAX_FRAME_LENGTH);
    if max_frame_length == 0 || max_frame_length > MAX_UINT32 {
        return Err(into_any_error(pi_chord::services::errors::type_error(
            format!("Server maxFrameLength must be an integer between 1 and {MAX_UINT32}"),
        )));
    }
    let handshake_timeout_ms = options
        .handshake_timeout_ms
        .unwrap_or(DEFAULT_HANDSHAKE_TIMEOUT_MS);
    if handshake_timeout_ms == 0 || handshake_timeout_ms > MAX_TIMER_DELAY_MS {
        return Err(into_any_error(pi_chord::services::errors::type_error(
            format!(
                "Server handshakeTimeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}"
            ),
        )));
    }
    Ok((max_frame_length, handshake_timeout_ms))
}

/// Upstream `RoutedServerPresentation` implementation routed at one
/// connection.
struct ServerPresentation {
    server: Arc<ServerInner>,
    conn: Arc<Conn>,
}

#[async_trait::async_trait]
impl RoutedServerPresentation for ServerPresentation {
    async fn attach_session(
        &self,
        session_id: &str,
        context: pi_chord::Context,
    ) -> Result<(), AnyError> {
        self.server
            .sessions
            .attach_client(self.conn.id, session_id, context)
            .await
    }

    async fn detach_session(&self, context: pi_chord::Context) -> Result<(), AnyError> {
        self.server
            .sessions
            .detach_client(self.conn.id, context)
            .await
    }

    async fn prepare_session_removal(
        &self,
        session_id: &str,
        context: pi_chord::Context,
    ) -> Result<(), AnyError> {
        self.server
            .sessions
            .remove_session(session_id, context)
            .await
    }
}

/// Upstream `ByteConnectionHandler` implementation feeding one connection.
struct ConnHandler {
    server: Arc<ServerInner>,
    conn: Arc<Conn>,
}

impl ByteConnectionHandler for ConnHandler {
    fn on_data(&self, chunk: &[u8]) {
        self.server.receive(&self.conn, chunk);
    }

    fn on_close(&self) {
        self.server.transport_closed(&self.conn);
    }

    fn on_error(&self, error: AnyError) {
        self.server.report_error(error);
        let server = Arc::clone(&self.server);
        let conn = Arc::clone(&self.conn);
        tokio::spawn(async move {
            let connection = conn.state.lock().connection.clone();
            server.close_connection(&connection, None).await;
            server.disconnect(&conn);
        });
    }
}

/// Upstream no-op handler returned when a connection arrives while closing.
struct ClosedHandler {
    server: Arc<ServerInner>,
}

impl ByteConnectionHandler for ClosedHandler {
    fn on_data(&self, _chunk: &[u8]) {}
    fn on_close(&self) {}
    fn on_error(&self, error: AnyError) {
        self.server.report_error(error);
    }
}
