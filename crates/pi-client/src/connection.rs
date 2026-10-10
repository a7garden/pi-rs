//! Port of `packages/client/src/connection.ts`.
//!
//! The JS single-threaded lifecycle state machine becomes a
//! `parking_lot::Mutex`-guarded state enum; callbacks (`onHandshake`,
//! `onMessage`, `onStateChange`) are invoked outside the lock, preserving the
//! upstream re-entrancy checks by re-validating identity (`id` + state) after
//! each callback boundary.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use pi_chord::services::errors::{type_error, AnyError};
use pi_protocol::{
    encode_client_message, FrameDecoderOptions, ProtocolValidationError, ServerMessageDecoder,
    PROTOCOL_VERSION,
};
use serde_json::json;

use crate::errors::{to_disconnected_error, DisconnectedError, ServerError};
use crate::promise::{create_promise_resolvers, Promise, PromiseResolvers};
use crate::transport::{ByteTransport, ByteTransportFactory, ByteTransportHandlers};
use crate::types::{ConnectionState, ConnectionStateChange};

const MAX_UINT32: u64 = 0xffff_ffff;

/// Upstream `ServerHello` narrowed to the field the client consumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerHello {
    pub server_id: String,
}

impl ServerHello {
    fn from_message(message: &pi_chord::json::JsonValue) -> Option<ServerHello> {
        Some(ServerHello {
            server_id: message.get("serverId")?.as_str()?.to_string(),
        })
    }
}

type HandshakeResolvers = PromiseResolvers<ServerHello>;

enum Lifecycle {
    Disconnected,
    Connecting {
        id: u64,
        decoder: ServerMessageDecoder,
        handshake: HandshakeResolvers,
        transport: Option<Arc<dyn ByteTransport>>,
    },
    Connected {
        id: u64,
        decoder: ServerMessageDecoder,
        handshake: Option<HandshakeResolvers>,
        transport: Arc<dyn ByteTransport>,
    },
}

pub(crate) struct ConnectionCallbacks {
    pub on_handshake: Arc<dyn Fn(&ServerHello) + Send + Sync>,
    pub on_message: Arc<dyn Fn(pi_chord::json::JsonValue) + Send + Sync>,
    pub on_state_change: Arc<dyn Fn(ConnectionStateChange) + Send + Sync>,
}

pub(crate) struct ConnectionOptions {
    pub transport_factory: ByteTransportFactory,
    pub server_id: String,
    pub max_frame_length: Option<usize>,
    pub callbacks: ConnectionCallbacks,
}

/// Upstream `Connection`: byte-transport lifecycle, framing, and handshake.
pub struct Connection {
    options: ConnectionOptions,
    max_frame_length: usize,
    state: parking_lot::Mutex<Lifecycle>,
    sequence: AtomicU64,
}

impl Connection {
    pub(crate) fn new(options: ConnectionOptions) -> Result<Connection, AnyError> {
        let max_frame_length = options
            .max_frame_length
            .unwrap_or(pi_protocol::DEFAULT_MAX_FRAME_LENGTH);
        if max_frame_length == 0 || max_frame_length as u64 > MAX_UINT32 {
            return Err(type_error(format!(
                "Client maxFrameLength must be an integer between 1 and {MAX_UINT32}"
            )));
        }
        Ok(Connection {
            options,
            max_frame_length,
            state: parking_lot::Mutex::new(Lifecycle::Disconnected),
            sequence: AtomicU64::new(0),
        })
    }

    pub fn state(&self) -> ConnectionState {
        match &*self.state.lock() {
            Lifecycle::Disconnected => ConnectionState::Disconnected,
            Lifecycle::Connecting { .. } => ConnectionState::Connecting,
            Lifecycle::Connected { .. } => ConnectionState::Connected,
        }
    }

    pub fn max_frame_length(&self) -> usize {
        self.max_frame_length
    }

    pub fn connect(self: &Arc<Self>) -> Promise<ServerHello> {
        let (resolvers, promise) = create_promise_resolvers();
        {
            let state = self.state.lock();
            if !matches!(&*state, Lifecycle::Disconnected) {
                let state_name = match &*state {
                    Lifecycle::Connecting { .. } => "connecting",
                    _ => "connected",
                };
                drop(state);
                resolvers.reject(Box::new(DisconnectedError::new(format!(
                    "Client is already {state_name}"
                ))));
                return promise;
            }
        }
        let id = self.sequence.fetch_add(1, Ordering::SeqCst) + 1;
        let decoder = match ServerMessageDecoder::new(Some(&FrameDecoderOptions {
            max_frame_length: Some(self.max_frame_length),
        })) {
            Ok(decoder) => decoder,
            Err(decoder_error) => {
                resolvers.reject_with(decoder_error);
                return promise;
            }
        };
        *self.state.lock() = Lifecycle::Connecting {
            id,
            decoder,
            handshake: resolvers.clone(),
            transport: None,
        };
        (self.options.callbacks.on_state_change)(ConnectionStateChange {
            state: ConnectionState::Connecting,
            error: None,
        });
        let handlers = ByteTransportHandlers {
            on_data: {
                let connection = Arc::downgrade(self);
                Arc::new(move |chunk| {
                    if let Some(connection) = connection.upgrade() {
                        connection.handle_data(id, chunk);
                    }
                })
            },
            on_close: {
                let connection = Arc::downgrade(self);
                Arc::new(move || {
                    if let Some(connection) = connection.upgrade() {
                        if connection.is_current(id) {
                            connection.handle_close();
                        }
                    }
                })
            },
            on_error: {
                let connection = Arc::downgrade(self);
                Arc::new(move |transport_error| {
                    if let Some(connection) = connection.upgrade() {
                        if connection.is_current(id) {
                            connection
                                .fail_and_close(Box::new(to_disconnected_error(transport_error)));
                        }
                    }
                })
            },
        };
        let connection = Arc::clone(self);
        tokio::spawn(async move { connection.open_transport(id, handlers).await });
        promise
    }

    pub fn disconnect(&self, reason: &str) {
        if matches!(&*self.state.lock(), Lifecycle::Disconnected) {
            return;
        }
        self.fail_and_close(Box::new(DisconnectedError::new(reason.to_string())));
    }

    pub fn fail(&self, error: AnyError) {
        self.fail_and_close(error);
    }

    pub fn send(self: &Arc<Self>, frame: Vec<u8>) -> Result<(), AnyError> {
        let transport = {
            let state = self.state.lock();
            match &*state {
                Lifecycle::Connected { transport, .. } => Arc::clone(transport),
                _ => return Err(Box::new(DisconnectedError::new("Client is disconnected"))),
            }
        };
        let connection = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(send_error) = transport.send(frame).await {
                // Only a still-current connected transport failure fails the
                // connection (upstream compares lifecycle transport identity).
                let is_current = match &*connection.state.lock() {
                    Lifecycle::Connected {
                        transport: current, ..
                    } => Arc::ptr_eq(current, &transport),
                    _ => false,
                };
                if is_current {
                    connection.fail_and_close(Box::new(to_disconnected_error(send_error)));
                }
            }
        });
        Ok(())
    }

    async fn open_transport(self: &Arc<Self>, id: u64, handlers: ByteTransportHandlers) {
        let transport = match (self.options.transport_factory)(handlers).await {
            Ok(transport) => transport,
            Err(factory_error) => {
                self.fail(Box::new(to_disconnected_error(factory_error)));
                return;
            }
        };
        let transport: Arc<dyn ByteTransport> = Arc::from(transport);
        {
            let mut state = self.state.lock();
            match &mut *state {
                Lifecycle::Connecting {
                    id: current,
                    transport: slot,
                    ..
                } if *current == id => {
                    *slot = Some(Arc::clone(&transport));
                }
                _ => {
                    drop(state);
                    transport.close();
                    return;
                }
            }
        }
        let frame = match encode_client_message(
            &json!({ "type": "hello", "version": PROTOCOL_VERSION }),
            Some(&FrameDecoderOptions {
                max_frame_length: Some(self.max_frame_length),
            }),
        ) {
            Ok(frame) => frame,
            Err(encode_error) => {
                self.fail_and_close(Box::new(encode_error));
                return;
            }
        };
        if let Err(send_error) = transport.send(frame).await {
            if self.is_current(id) {
                self.fail_and_close(Box::new(to_disconnected_error(send_error)));
            }
        }
    }

    fn handle_data(self: &Arc<Self>, id: u64, chunk: Vec<u8>) {
        let decoded = {
            let mut state = self.state.lock();
            match &mut *state {
                Lifecycle::Disconnected => return,
                Lifecycle::Connecting {
                    id: current,
                    transport: None,
                    ..
                } if *current == id => {
                    drop(state);
                    self.fail_and_close(Box::new(ProtocolValidationError::new(
                        "Received server data before the client hello was sent",
                    )));
                    return;
                }
                Lifecycle::Connecting {
                    id: current,
                    decoder,
                    ..
                } if *current == id => decoder.push(&chunk),
                Lifecycle::Connected {
                    id: current,
                    decoder,
                    ..
                } if *current == id => decoder.push(&chunk),
                _ => return,
            }
        };
        let messages = match decoded {
            Ok(messages) => messages,
            Err(decode_error) => {
                self.fail_and_close(Box::new(decode_error));
                return;
            }
        };
        for message in messages {
            if matches!(&*self.state.lock(), Lifecycle::Disconnected) {
                return;
            }
            self.handle_message(message);
        }
    }

    fn handle_message(self: &Arc<Self>, message: pi_chord::json::JsonValue) {
        let action = match message.get("type").and_then(|value| value.as_str()) {
            Some("hello") => MessageAction::Hello,
            Some("hello_error") => MessageAction::HelloError,
            _ => MessageAction::Other,
        };

        // Handshake phase (upstream `state === "connecting"`).
        let handshake_hello = {
            let mut state = self.state.lock();
            if !matches!(&*state, Lifecycle::Connecting { .. }) {
                None
            } else {
                match action {
                    MessageAction::HelloError => {
                        let protocol_error: AnyError = message
                            .get("error")
                            .and_then(ServerError::from_protocol)
                            .map(|server_error| Box::new(server_error) as AnyError)
                            .unwrap_or_else(|| type_error("Invalid hello_error payload"));
                        drop(state);
                        self.fail_and_close(protocol_error);
                        return;
                    }
                    MessageAction::Other => {
                        drop(state);
                        self.fail_and_close(Box::new(ProtocolValidationError::new(
                            "Expected server hello as first message",
                        )));
                        return;
                    }
                    MessageAction::Hello => {
                        let reported_server_id = message
                            .get("serverId")
                            .and_then(|value| value.as_str())
                            .unwrap_or("")
                            .to_string();
                        if reported_server_id != self.options.server_id {
                            drop(state);
                            self.fail_and_close(Box::new(ProtocolValidationError::new(format!(
                                "Connected server {} does not match {}",
                                json!(reported_server_id),
                                json!(self.options.server_id)
                            ))));
                            return;
                        }
                        let Some(hello) = ServerHello::from_message(&message) else {
                            drop(state);
                            self.fail_and_close(Box::new(ProtocolValidationError::new(
                                "Expected server hello as first message",
                            )));
                            return;
                        };
                        let previous = std::mem::replace(&mut *state, Lifecycle::Disconnected);
                        let Lifecycle::Connecting {
                            id,
                            decoder,
                            handshake,
                            transport,
                        } = previous
                        else {
                            unreachable!("checked Connecting above");
                        };
                        let Some(transport) = transport else {
                            drop(state);
                            self.fail_and_close(Box::new(ProtocolValidationError::new(
                                "Received server hello before the client hello was sent",
                            )));
                            return;
                        };
                        *state = Lifecycle::Connected {
                            id,
                            decoder,
                            handshake: Some(handshake),
                            transport,
                        };
                        Some(hello)
                    }
                }
            }
        };
        if let Some(hello) = handshake_hello {
            // Upstream calls onHandshake while connected; the listener may fail
            // the connection, so liveness is re-checked afterwards.
            (self.options.callbacks.on_handshake)(&hello);
            let handshake = {
                let mut state = self.state.lock();
                match &mut *state {
                    Lifecycle::Connected { handshake, .. } => handshake.take(),
                    Lifecycle::Disconnected | Lifecycle::Connecting { .. } => return,
                }
            };
            (self.options.callbacks.on_state_change)(ConnectionStateChange {
                state: ConnectionState::Connected,
                error: None,
            });
            if !matches!(&*self.state.lock(), Lifecycle::Connected { .. }) {
                return;
            }
            if let Some(handshake) = handshake {
                handshake.resolve(hello);
            }
            return;
        }

        // Connected phase.
        let deliver = {
            let state = self.state.lock();
            if !matches!(&*state, Lifecycle::Connected { .. }) {
                false
            } else if matches!(action, MessageAction::Hello | MessageAction::HelloError) {
                drop(state);
                self.fail_and_close(Box::new(ProtocolValidationError::new(
                    "Unexpected handshake message",
                )));
                false
            } else {
                true
            }
        };
        if deliver {
            (self.options.callbacks.on_message)(message);
        }
    }

    fn handle_close(&self) {
        let error: AnyError = {
            let mut state = self.state.lock();
            let decoder = match &mut *state {
                Lifecycle::Disconnected => return,
                Lifecycle::Connecting { decoder, .. } | Lifecycle::Connected { decoder, .. } => {
                    decoder
                }
            };
            match decoder.end() {
                Ok(()) => Box::new(DisconnectedError::new("Byte transport closed")),
                Err(decoder_error) => Box::new(decoder_error),
            }
        };
        self.fail_connection(error);
    }

    fn fail_and_close(&self, error: AnyError) {
        let transport = {
            let state = self.state.lock();
            match &*state {
                Lifecycle::Disconnected => None,
                Lifecycle::Connecting { transport, .. } => transport.clone(),
                Lifecycle::Connected { transport, .. } => Some(Arc::clone(transport)),
            }
        };
        self.fail_connection(error);
        if let Some(transport) = transport {
            transport.close();
        }
    }

    fn fail_connection(&self, error: AnyError) {
        let handshake = {
            let mut state = self.state.lock();
            if matches!(&*state, Lifecycle::Disconnected) {
                return;
            }
            let previous = std::mem::replace(&mut *state, Lifecycle::Disconnected);
            match previous {
                Lifecycle::Disconnected => None,
                Lifecycle::Connecting { handshake, .. } => Some(handshake),
                Lifecycle::Connected { handshake, .. } => handshake,
            }
        };
        if let Some(handshake) = handshake {
            handshake.reject(crate::client::clone_any_error(&error));
        }
        (self.options.callbacks.on_state_change)(ConnectionStateChange {
            state: ConnectionState::Disconnected,
            error: Some(error),
        });
    }

    fn is_current(&self, id: u64) -> bool {
        match &*self.state.lock() {
            Lifecycle::Disconnected => false,
            Lifecycle::Connecting { id: current, .. }
            | Lifecycle::Connected { id: current, .. } => *current == id,
        }
    }
}

enum MessageAction {
    Hello,
    HelloError,
    Other,
}
