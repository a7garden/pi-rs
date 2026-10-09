//! Port of `experimental/radius-relay.ts` — the experimental server's
//! multiplexed, authenticated Radius host connection and the client-side
//! relay transport plus reconnect helper.
//!
//! Deviations (D150):
//! - Upstream drives the npm `undici` `WebSocket` (an event-emitter with
//!   `binaryType`/`bufferedAmount`). The Rust port defines the
//!   [`RadiusRelayWebSocket`] seam as a single-listener trait; the default
//!   factory is built on `tokio-tungstenite` (`Authorization` header +
//!   subprotocol selection verified after the handshake).
//! - The `OrderedWebSocketWriter` drain loop (polling `bufferedAmount` with a
//!   5 ms delay) keeps its upstream shape: sends are serialized and park on a
//!   5 ms poll while the socket's pending bytes exceed the drain threshold.
//! - Upstream's browser-API close-code restriction note does not apply;
//!   local protocol/transport errors still close with application codes
//!   4000/4001 so the gateway can distinguish them from RFC codes.
//! - `AbortSignal` cancellation of the host loop becomes the host's own
//!   closed flag + a `watch` channel; timers use tokio sleeps.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use pi_chord::services::errors::{error, AnyError};
use pi_client::transport::{ByteTransport, ByteTransportFactory, ByteTransportHandlers};
use pi_protocol::DEFAULT_MAX_FRAME_LENGTH;
use pi_server::connection::{ByteConnection, ByteConnectionHandler};
use pi_server::server::Server as RoutedServer;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::{AUTHORIZATION, SEC_WEBSOCKET_PROTOCOL};
use tokio_tungstenite::tungstenite::Message as WireMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::experimental::radius_auth::{RadiusRelayAuth, RadiusRelayAuthResolver, RadiusRelayResolveOptions};

pub const RADIUS_RELAY_HOST_SUBPROTOCOL: &str = "pi-session-relay.host.v1";
pub const RADIUS_RELAY_CLIENT_SUBPROTOCOL: &str = "pi-session-relay.client.v1";

const RELAY_DATA_HEADER_BYTES: usize = 18;
const RELAY_DATA_FRAME_VERSION: u8 = 1;
const RELAY_DATA_FRAME_TYPE: u8 = 1;
const MAX_PENDING_BYTES: usize = DEFAULT_MAX_FRAME_LENGTH * 4;
const DRAIN_THRESHOLD_BYTES: usize = 1024 * 1024;
const HOST_RETRY_INITIAL_MS: u64 = 1_000;
const HOST_RETRY_MAX_MS: u64 = 30_000;
const MISSING_AUTH_RETRY_MS: u64 = 30_000;
const CLIENT_RETRY_INITIAL_MS: u64 = 1_000;
const CLIENT_RETRY_MAX_MS: u64 = 30_000;
/// Upstream `LOCAL_PROTOCOL_ERROR_CLOSE_CODE`.
pub const LOCAL_PROTOCOL_ERROR_CLOSE_CODE: u16 = 4000;
/// Upstream `LOCAL_TRANSPORT_ERROR_CLOSE_CODE`.
pub const LOCAL_TRANSPORT_ERROR_CLOSE_CODE: u16 = 4001;

// ---------------------------------------------------------------------------
// WebSocket seam (upstream `RadiusRelayWebSocket`).
// ---------------------------------------------------------------------------

/// One inbound or outbound socket message (`data` values: strings are control
/// frames, binary chunks are relay data frames).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelaySocketMessage {
	Text(String),
	Binary(Vec<u8>),
}

impl RelaySocketMessage {
	fn byte_len(&self) -> usize {
		match self {
			RelaySocketMessage::Text(text) => text.len(),
			RelaySocketMessage::Binary(bytes) => bytes.len(),
		}
	}
}

/// Upstream's per-event listeners collapse into one listener object.
pub trait RadiusRelaySocketListener: Send + Sync {
	fn on_message(&self, message: RelaySocketMessage);
	fn on_close(&self, code: u16, reason: &str);
	fn on_error(&self, error: AnyError);
}

/// Upstream `RadiusRelayWebSocket` (undici event API → seam).
pub trait RadiusRelayWebSocket: Send + Sync {
	/// Upstream `readyState === OPEN`.
	fn is_open(&self) -> bool;
	/// The negotiated subprotocol.
	fn protocol(&self) -> String;
	/// Upstream `bufferedAmount`.
	fn buffered_amount(&self) -> usize;
	/// Upstream `send(data)`: enqueue without awaiting delivery.
	fn send(&self, message: RelaySocketMessage) -> Result<(), AnyError>;
	/// Upstream `close(code, reason)`.
	fn close(&self, code: u16, reason: &str);
	/// Replace the single event listener (upstream add/removeEventListener).
	fn set_listener(&self, listener: Option<Arc<dyn RadiusRelaySocketListener>>);
}

/// Upstream `RadiusRelayWebSocketFactory`.
pub type RadiusRelayWebSocketFactory = Arc<
	dyn Fn(RadiusRelayWebSocketRequest) -> futures_util::future::BoxFuture<'static, Result<Arc<dyn RadiusRelayWebSocket>, AnyError>>
		+ Send
		+ Sync,
>;

/// Factory argument (upstream's inline options object).
pub struct RadiusRelayWebSocketRequest {
	pub url: String,
	pub protocol: String,
	pub authorization: String,
}

enum SocketCommand {
	Send(RelaySocketMessage),
	Close { code: u16, reason: String },
}

struct TungsteniteSocketState {
	commands: tokio::sync::mpsc::UnboundedSender<SocketCommand>,
	pending_bytes: AtomicUsize,
	open: AtomicBool,
	protocol: String,
	listener: parking_lot::Mutex<Option<Arc<dyn RadiusRelaySocketListener>>>,
}

/// Default WebSocket factory implementation (`defaultWebSocketFactory` +
/// `openRadiusRelayWebSocket`): connect, verify the selected subprotocol,
/// and pump frames between the socket and its listener.
pub fn default_web_socket_factory() -> RadiusRelayWebSocketFactory {
	Arc::new(|request: RadiusRelayWebSocketRequest| {
		Box::pin(async move { open_radius_relay_websocket(&request.url, &request.protocol, &request.authorization).await })
	})
}

async fn open_radius_relay_websocket(
	url: &str,
	protocol: &str,
	authorization: &str,
) -> Result<Arc<dyn RadiusRelayWebSocket>, AnyError> {
	let mut request = url
		.into_client_request()
		.map_err(|handshake_error| error(format!("Radius relay WebSocket request failed: {handshake_error}")))?;
	request.headers_mut().insert(AUTHORIZATION, authorization.parse().expect("authorization header parses"));
	request
		.headers_mut()
		.insert(SEC_WEBSOCKET_PROTOCOL, protocol.parse().expect("subprotocol header parses"));
	let (stream, response) = tokio_tungstenite::connect_async(request)
		.await
		.map_err(|handshake_error| error(format!("Radius relay WebSocket handshake failed: {handshake_error}")))?;
	let selected = response
		.headers()
		.get(SEC_WEBSOCKET_PROTOCOL)
		.and_then(|value| value.to_str().ok())
		.unwrap_or_default()
		.to_string();
	if selected != protocol {
		return Err(error(format!(
			"Radius relay selected unexpected WebSocket protocol {selected:?}"
		)));
	}
	Ok(spawn_tungstenite_socket(stream, selected))
}

fn spawn_tungstenite_socket(stream: WebSocketStream<MaybeTlsStream<TcpStream>>, protocol: String) -> Arc<TungsteniteSocket> {
	let (commands_tx, commands_rx) = tokio::sync::mpsc::unbounded_channel::<SocketCommand>();
	let state = Arc::new(TungsteniteSocketState {
		commands: commands_tx,
		pending_bytes: AtomicUsize::new(0),
		open: AtomicBool::new(true),
		protocol,
		listener: parking_lot::Mutex::new(None),
	});
	tokio::spawn(pump(stream, Arc::clone(&state), commands_rx));
	Arc::new(TungsteniteSocket { state })
}

async fn pump(
	mut stream: WebSocketStream<MaybeTlsStream<TcpStream>>,
	state: Arc<TungsteniteSocketState>,
	mut commands: tokio::sync::mpsc::UnboundedReceiver<SocketCommand>,
) {
	let mut closing = false;
	loop {
		tokio::select! {
			command = commands.recv() => {
				let Some(command) = command else {
					break;
				};
				match command {
					SocketCommand::Send(message) => {
						let wire = match &message {
							RelaySocketMessage::Text(text) => WireMessage::Text(text.clone().into()),
							RelaySocketMessage::Binary(bytes) => WireMessage::Binary(bytes.clone().into()),
						};
						if let Err(send_error) = stream.send(wire).await {
							state.pending_bytes.fetch_sub(message.byte_len(), Ordering::SeqCst);
							socket_failed(&state, format!("Radius relay WebSocket send failed: {send_error}"));
							return;
						}
						state.pending_bytes.fetch_sub(message.byte_len(), Ordering::SeqCst);
						let _ = stream.flush().await;
					}
					SocketCommand::Close { code, reason } => {
						closing = true;
						if let Err(close_error) = stream.close(Some(tokio_tungstenite::tungstenite::protocol::frame::CloseFrame {
							code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::from(code),
							reason: reason.into(),
						})).await {
							socket_failed(&state, format!("Radius relay WebSocket close failed: {close_error}"));
							return;
						}
					}
				}
			}
			incoming = stream.next() => {
				match incoming {
					Some(Ok(WireMessage::Text(text))) => deliver(&state, RelaySocketMessage::Text(text.to_string())),
					Some(Ok(WireMessage::Binary(bytes))) => deliver(&state, RelaySocketMessage::Binary(bytes.to_vec())),
					Some(Ok(WireMessage::Close(frame))) => {
						let code = frame.as_ref().map(|frame| u16::from(frame.code)).unwrap_or(1005);
						let reason = frame.map(|frame| frame.reason.to_string()).unwrap_or_default();
						socket_closed(&state);
						listener(&state).on_close(code, &reason);
						return;
					}
					Some(Ok(WireMessage::Ping(_) | WireMessage::Pong(_) | WireMessage::Frame(_))) => {
						// Pings are answered automatically by tungstenite on
						// flush; raw frames are not a relay message.
						let _ = stream.flush().await;
					}
					Some(Err(stream_error)) => {
						socket_failed(&state, format!("Radius relay WebSocket failed: {stream_error}"));
						return;
					}
					None => {
						socket_closed(&state);
						listener(&state).on_close(1006, "Connection dropped without a close frame");
						return;
					}
				}
			}
		}
	}
	let _ = closing;
}

fn deliver(state: &Arc<TungsteniteSocketState>, message: RelaySocketMessage) {
	listener(state).on_message(message);
}

fn listener(state: &Arc<TungsteniteSocketState>) -> Arc<dyn RadiusRelaySocketListener> {
	state
		.listener
		.lock()
		.clone()
		.unwrap_or_else(|| Arc::new(SilentListener))
}

struct SilentListener;

impl RadiusRelaySocketListener for SilentListener {
	fn on_message(&self, _message: RelaySocketMessage) {}
	fn on_close(&self, _code: u16, _reason: &str) {}
	fn on_error(&self, _error: AnyError) {}
}

fn socket_closed(state: &Arc<TungsteniteSocketState>) {
	state.open.store(false, Ordering::SeqCst);
}

fn socket_failed(state: &Arc<TungsteniteSocketState>, message: String) {
	let was_open = state.open.swap(false, Ordering::SeqCst);
	if was_open {
		listener(state).on_error(error(message));
	}
}

struct TungsteniteSocket {
	state: Arc<TungsteniteSocketState>,
}

impl RadiusRelayWebSocket for TungsteniteSocket {
	fn is_open(&self) -> bool {
		self.state.open.load(Ordering::SeqCst)
	}

	fn protocol(&self) -> String {
		self.state.protocol.clone()
	}

	fn buffered_amount(&self) -> usize {
		self.state.pending_bytes.load(Ordering::SeqCst)
	}

	fn send(&self, message: RelaySocketMessage) -> Result<(), AnyError> {
		if !self.is_open() {
			return Err(error("Radius relay WebSocket is closed"));
		}
		self.state.pending_bytes.fetch_add(message.byte_len(), Ordering::SeqCst);
		self.state
			.commands
			.send(SocketCommand::Send(message))
			.map_err(|_| error("Radius relay WebSocket is closed"))
	}

	fn close(&self, code: u16, reason: &str) {
		if !self.is_open() {
			return;
		}
		self.state.open.store(false, Ordering::SeqCst);
		let _ = self
			.state
			.commands
			.send(SocketCommand::Close { code, reason: reason.to_string() });
	}

	fn set_listener(&self, listener: Option<Arc<dyn RadiusRelaySocketListener>>) {
		*self.state.listener.lock() = listener;
	}
}

// ---------------------------------------------------------------------------
// Relay data frames and control messages.
// ---------------------------------------------------------------------------

/// Upstream `CONNECTION_ID_PATTERN` (lowercase UUIDv4).
pub fn is_connection_id(value: &str) -> bool {
	let bytes = value.as_bytes();
	if bytes.len() != 36 {
		return false;
	}
	for (index, byte) in bytes.iter().enumerate() {
		match index {
			8 | 13 | 18 | 23 => {
				if *byte != b'-' {
					return false;
				}
			}
			14 => {
				if *byte != b'4' {
					return false;
				}
			}
			19 => {
				if !matches!(byte, b'8' | b'9' | b'a' | b'b') {
					return false;
				}
			}
			_ => {
				if !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase() {
					return false;
				}
			}
		}
	}
	true
}

/// Inbound host control messages (`HostInputControlMessage`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostInputControlMessage {
	Ping,
	Pong,
	ConnectionOpen { connection_id: String },
	ConnectionClose { connection_id: String, code: Option<u16> },
}

/// Outbound host control messages (`HostOutputControlMessage`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostOutputControlMessage {
	Pong,
	ConnectionClose { connection_id: String, code: Option<u16> },
}

impl HostOutputControlMessage {
	fn to_json(&self) -> serde_json::Value {
		match self {
			HostOutputControlMessage::Pong => serde_json::json!({ "version": 1, "type": "pong" }),
			HostOutputControlMessage::ConnectionClose { connection_id, code } => {
				serde_json::json!({ "version": 1, "type": "connection_close", "connection_id": connection_id, "code": code })
			}
		}
	}
}

/// `parseHostControlMessage`.
pub fn parse_host_control_message(value: &str) -> Result<HostInputControlMessage, AnyError> {
	let invalid = || error("Invalid Radius relay control message");
	let parsed: serde_json::Value = serde_json::from_str(value).map_err(|_| invalid())?;
	let object = parsed.as_object().ok_or_else(invalid)?;
	let version = object.get("version").and_then(|value| value.as_u64());
	if version != Some(1) {
		return Err(error("Unsupported Radius relay control version"));
	}
	match object.get("type").and_then(|value| value.as_str()) {
		Some("ping") => Ok(HostInputControlMessage::Ping),
		Some("pong") => Ok(HostInputControlMessage::Pong),
		Some(kind @ ("connection_open" | "connection_close")) => {
			let connection_id = object
				.get("connection_id")
				.and_then(|value| value.as_str())
				.ok_or_else(invalid)?;
			if !is_connection_id(connection_id) {
				return Err(invalid());
			}
			let code = match object.get("code") {
				None => None,
				Some(value) => {
					let code = value.as_u64().ok_or_else(invalid)?;
					if !(1000..=4999).contains(&code) {
						return Err(invalid());
					}
					Some(code as u16)
				}
			};
			match kind {
				"connection_open" => Ok(HostInputControlMessage::ConnectionOpen { connection_id: connection_id.to_string() }),
				_ => Ok(HostInputControlMessage::ConnectionClose { connection_id: connection_id.to_string(), code }),
			}
		}
		_ => Err(invalid()),
	}
}

/// `encodeRelayDataFrame`.
pub fn encode_relay_dataframe(connection_id: &str, payload: &[u8]) -> Result<Vec<u8>, AnyError> {
	if !is_connection_id(connection_id) {
		return Err(error("Invalid Radius relay connection ID"));
	}
	let mut frame = Vec::with_capacity(RELAY_DATA_HEADER_BYTES + payload.len());
	frame.push(RELAY_DATA_FRAME_VERSION);
	frame.push(RELAY_DATA_FRAME_TYPE);
	let hex = connection_id.replace('-', "");
	for index in 0..16 {
		frame.push(u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).expect("validated hex"));
	}
	frame.extend_from_slice(payload);
	Ok(frame)
}

/// Parsed relay data frame (`parseRelayDataFrame`).
pub struct RelayDataFrame {
	pub connection_id: String,
	pub payload: Vec<u8>,
}

/// `parseRelayDataFrame`; `None` for non-data frames.
pub fn parse_relay_dataframe(frame: &[u8]) -> Option<RelayDataFrame> {
	if frame.len() < RELAY_DATA_HEADER_BYTES {
		return None;
	}
	if frame[0] != RELAY_DATA_FRAME_VERSION || frame[1] != RELAY_DATA_FRAME_TYPE {
		return None;
	}
	let mut hex = String::with_capacity(32);
	for byte in &frame[2..RELAY_DATA_HEADER_BYTES] {
		hex.push_str(&format!("{byte:02x}"));
	}
	let connection_id = format!(
		"{}-{}-{}-{}-{}",
		&hex[0..8],
		&hex[8..12],
		&hex[12..16],
		&hex[16..20],
		&hex[20..32]
	);
	if !is_connection_id(&connection_id) {
		return None;
	}
	Some(RelayDataFrame { connection_id, payload: frame[RELAY_DATA_HEADER_BYTES..].to_vec() })
}

/// `relayWebSocketUrl`.
pub fn relay_web_socket_url(gateway: &str, server_id: &str) -> Result<String, AnyError> {
	let mut url = url::Url::parse(gateway).map_err(|parse_error| error(format!("Invalid Radius gateway URL: {parse_error}")))?;
	url.set_path(&format!("/v1/session-relays/{server_id}/connect"));
	match url.scheme() {
		"https" => url.set_scheme("wss").expect("wss is a valid scheme"),
		"http" => url.set_scheme("ws").expect("ws is a valid scheme"),
		other => return Err(error(format!("Unsupported Radius gateway protocol: {other}"))),
	}
	Ok(url.into())
}

// ---------------------------------------------------------------------------
// Ordered writer (upstream `OrderedWebSocketWriter`).
// ---------------------------------------------------------------------------

struct OrderedWebSocketWriter {
	socket: Arc<dyn RadiusRelayWebSocket>,
	/// Serializes sends (upstream `#tail` promise chain).
	tail: tokio::sync::Mutex<()>,
	pending_bytes: AtomicUsize,
	closed: AtomicBool,
}

impl OrderedWebSocketWriter {
	fn new(socket: Arc<dyn RadiusRelayWebSocket>) -> Arc<Self> {
		Arc::new(Self {
			socket,
			tail: tokio::sync::Mutex::new(()),
			pending_bytes: AtomicUsize::new(0),
			closed: AtomicBool::new(false),
		})
	}

	fn close(&self) {
		self.closed.store(true, Ordering::SeqCst);
	}

	async fn send(&self, message: RelaySocketMessage) -> Result<(), AnyError> {
		if self.closed.load(Ordering::SeqCst) {
			return Err(error("Radius relay WebSocket is closed"));
		}
		let byte_len = message.byte_len();
		if self.pending_bytes.load(Ordering::SeqCst) + byte_len > MAX_PENDING_BYTES {
			return Err(error("Radius relay exceeded its pending byte limit"));
		}
		self.pending_bytes.fetch_add(byte_len, Ordering::SeqCst);
		let result = async {
			let _guard = self.tail.lock().await;
			if self.closed.load(Ordering::SeqCst) || !self.socket.is_open() {
				return Err(error("Radius relay WebSocket is closed"));
			}
			self.socket.send(message)?;
			while self.socket.buffered_amount() > DRAIN_THRESHOLD_BYTES {
				if self.closed.load(Ordering::SeqCst) || !self.socket.is_open() {
					return Err(error("Radius relay WebSocket closed during write"));
				}
				tokio::time::sleep(Duration::from_millis(5)).await;
			}
			Ok(())
		}
		.await;
		self.pending_bytes.fetch_sub(byte_len, Ordering::SeqCst);
		result
	}
}

// ---------------------------------------------------------------------------
// Server-side byte connections (upstream `RelayServerByteConnection`).
// ---------------------------------------------------------------------------

struct ActiveConnection {
	connection: Arc<RelayServerByteConnection>,
	handler: Arc<dyn ByteConnectionHandler>,
}

/// The server-facing half of one relayed client connection.
struct RelayServerByteConnection {
	session: parking_lot::Mutex<Option<Arc<HostSession>>>,
	connection_id: String,
	closed: AtomicBool,
}

impl RelayServerByteConnection {
	fn mark_closed(&self) {
		self.closed.store(true, Ordering::SeqCst);
	}
}

/// pi-server's shared error type (`Arc<dyn Error + Send + Sync>`).
type SrvError = pi_server::types::AnyError;

fn srv_error(message: impl Into<String>) -> SrvError {
	Arc::new(pi_chord::services::errors::ChordError { message: message.into() })
}

#[async_trait::async_trait]
impl ByteConnection for RelayServerByteConnection {
	fn closed(&self) -> bool {
		self.closed.load(Ordering::SeqCst)
	}

	async fn send(&self, chunk: Vec<u8>) -> Result<(), SrvError> {
		if self.closed() {
			return Err(srv_error("Radius relay connection is closed"));
		}
		let session = self.session.lock().as_ref().map(Arc::downgrade);
		let Some(session) = session.and_then(|weak| weak.upgrade()) else {
			return Err(srv_error("Radius relay connection is closed"));
		};
		session.send_data(&self.connection_id, chunk).await.map_err(|failure| srv_error(failure.to_string()))
	}

	async fn close(&self, final_chunk: Option<Vec<u8>>) -> Result<(), SrvError> {
		if self.closed() {
			return Ok(());
		}
		self.closed.store(true, Ordering::SeqCst);
		let session = self.session.lock().as_ref().map(Arc::downgrade);
		let Some(session) = session.and_then(|weak| weak.upgrade()) else {
			return Ok(());
		};
		session
			.server_close_connection(&self.connection_id, final_chunk)
			.await
			.map_err(|failure| srv_error(failure.to_string()))
	}
}

// ---------------------------------------------------------------------------
// Host status and options.
// ---------------------------------------------------------------------------

/// Upstream `RadiusRelayHostStatus`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RadiusRelayHostStatus {
	NotAuthenticated,
	Connecting,
	Connected,
	Retrying { error: String },
}

impl RadiusRelayHostStatus {
	/// The upstream wire tag (the `status` discriminator).
	pub fn status_name(&self) -> &'static str {
		match self {
			RadiusRelayHostStatus::NotAuthenticated => "not_authenticated",
			RadiusRelayHostStatus::Connecting => "connecting",
			RadiusRelayHostStatus::Connected => "connected",
			RadiusRelayHostStatus::Retrying { .. } => "retrying",
		}
	}
}

/// Upstream `RadiusRelayHostOptions`.
pub struct RadiusRelayHostOptions {
	pub server_id: String,
	pub server: Arc<RoutedServer>,
	pub auth: Arc<RadiusRelayAuthResolver>,
	pub web_socket_factory: Option<RadiusRelayWebSocketFactory>,
	pub on_status: Option<Arc<dyn Fn(RadiusRelayHostStatus) + Send + Sync>>,
}

// ---------------------------------------------------------------------------
// Host loop.
// ---------------------------------------------------------------------------

/// Maintain the experimental server's multiplexed, authenticated Radius host
/// connection (upstream `RadiusRelayHost`).
pub struct RadiusRelayHost {
	options: RadiusRelayHostOptions,
	closed: AtomicBool,
	closed_tx: tokio::sync::watch::Sender<bool>,
	closed_rx: tokio::sync::watch::Receiver<bool>,
	/// The live serving session (replaced per connection attempt).
	session: parking_lot::Mutex<Option<Arc<HostSession>>>,
	loop_handle: parking_lot::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl RadiusRelayHost {
	pub fn new(options: RadiusRelayHostOptions) -> Arc<Self> {
		let (closed_tx, closed_rx) = tokio::sync::watch::channel(false);
		Arc::new(Self {
			options,
			closed: AtomicBool::new(false),
			closed_tx,
			closed_rx,
			session: parking_lot::Mutex::new(None),
			loop_handle: parking_lot::Mutex::new(None),
		})
	}

	/// `start()`: begin the (re)connection loop.
	pub fn start(self: &Arc<Self>) {
		let mut handle_slot = self.loop_handle.lock();
		if handle_slot.is_some() {
			return;
		}
		let host = Arc::clone(self);
		*handle_slot = Some(tokio::spawn(async move { host.run().await }));
	}

	/// `close()`: stop the loop, close the socket, and drop connections.
	pub async fn close(&self) {
		if !self.closed.swap(true, Ordering::SeqCst) {
			let _ = self.closed_tx.send(true);
			if let Some(session) = self.session.lock().take() {
				close_web_socket(&session.socket, 1000, "Pi server stopped");
				session.finish_ok();
			}
		}
		let handle = self.loop_handle.lock().take();
		if let Some(handle) = handle {
			let _ = handle.await;
		}
	}

	fn report_status(&self, status: RadiusRelayHostStatus) {
		if let Some(on_status) = &self.options.on_status {
			on_status(status);
		}
	}

	fn factory(&self) -> RadiusRelayWebSocketFactory {
		self.options.web_socket_factory.clone().unwrap_or_else(default_web_socket_factory)
	}

	async fn run(self: Arc<Self>) {
		let mut retry_ms = HOST_RETRY_INITIAL_MS;
		while !self.closed.load(Ordering::SeqCst) {
			let auth = match self
				.options
				.auth
				.resolve(&RadiusRelayResolveOptions { required: false, signal: None })
				.await
			{
				Ok(auth) => auth,
				// An aborted resolver (offline gate) parks like missing auth.
				Err(_) => None,
			};
			let Some(auth) = auth else {
				self.report_status(RadiusRelayHostStatus::NotAuthenticated);
				if self.delay(MISSING_AUTH_RETRY_MS).await.is_break() {
					break;
				}
				continue;
			};

			self.report_status(RadiusRelayHostStatus::Connecting);
			let opened = open_with_factory(
				&self.factory(),
				&relay_web_socket_url(&auth.gateway, &self.options.server_id).expect("gateway URL is valid"),
				RADIUS_RELAY_HOST_SUBPROTOCOL,
				&auth,
			)
			.await;
			let socket = match opened {
				Ok(socket) => socket,
				Err(open_error) => {
					if self.handle_attempt_failure(open_error, &mut retry_ms).await.is_break() {
						break;
					}
					continue;
				}
			};
			retry_ms = HOST_RETRY_INITIAL_MS;
			self.report_status(RadiusRelayHostStatus::Connected);
			let outcome = self.serve(socket).await;
			if self.closed.load(Ordering::SeqCst) {
				break;
			}
			match outcome {
				Ok(()) => {
					let failure = error("Radius relay host disconnected");
					if self.handle_attempt_failure(failure, &mut retry_ms).await.is_break() {
						break;
					}
				}
				Err(failure) => {
					if self.handle_attempt_failure(failure, &mut retry_ms).await.is_break() {
						break;
					}
				}
			}
		}
	}

	/// Upstream `delay(retryMs, signal)` inside the loop: cancellable by
	/// `close()`.
	async fn delay(&self, milliseconds: u64) -> std::ops::ControlFlow<()> {
		let mut closed = self.closed_rx.clone();
		tokio::select! {
			_ = tokio::time::sleep(Duration::from_millis(milliseconds)) => std::ops::ControlFlow::Continue(()),
			_ = closed.changed() => std::ops::ControlFlow::Break(()),
		}
	}

	async fn handle_attempt_failure(&self, failure: AnyError, retry_ms: &mut u64) -> std::ops::ControlFlow<()> {
		if self.closed.load(Ordering::SeqCst) {
			return std::ops::ControlFlow::Break(());
		}
		self.report_status(RadiusRelayHostStatus::Retrying { error: failure.to_string() });
		if self.delay(*retry_ms).await.is_break() {
			return std::ops::ControlFlow::Break(());
		}
		*retry_ms = (*retry_ms * 2).min(HOST_RETRY_MAX_MS);
		std::ops::ControlFlow::Continue(())
	}

	/// `#serve`: pump one connected socket until it closes.
	async fn serve(self: &Arc<Self>, socket: Arc<dyn RadiusRelayWebSocket>) -> Result<(), AnyError> {
		let session = Arc::new(HostSession::new(Arc::clone(self), Arc::clone(&socket)));
		*self.session.lock() = Some(Arc::clone(&session));
		socket.set_listener(Some(Arc::new(HostListener { session: Arc::downgrade(&session) })));
		session.serve().await
	}

	fn require_session(&self) -> Result<Arc<HostSession>, AnyError> {
		self.session
			.lock()
			.as_ref()
			.map(Arc::clone)
			.ok_or_else(|| error("Radius relay host is disconnected"))
	}

	async fn send_control(&self, message: HostOutputControlMessage) -> Result<(), AnyError> {
		self.require_session()?.writer.send(RelaySocketMessage::Text(message.to_json().to_string())).await
	}

	async fn send_data(&self, connection_id: &str, chunk: Vec<u8>) -> Result<(), AnyError> {
		let session = self.require_session()?;
		if !session.connections.lock().contains_key(connection_id) {
			return Err(error("Radius relay connection is closed"));
		}
		session
			.writer
			.send(RelaySocketMessage::Binary(encode_relay_dataframe(connection_id, &chunk)?))
			.await
	}

	async fn server_close_connection(&self, connection_id: &str, final_chunk: Option<Vec<u8>>) -> Result<(), AnyError> {
		{
			let session = self.require_session()?;
			if session.connections.lock().remove(connection_id).is_none() {
				return Ok(());
			}
		}
		if let Some(final_chunk) = final_chunk {
			self.require_session()?
				.writer
				.send(RelaySocketMessage::Binary(encode_relay_dataframe(connection_id, &final_chunk)?))
				.await?;
		}
		self.send_control(HostOutputControlMessage::ConnectionClose { connection_id: connection_id.to_string(), code: Some(1000) })
			.await
	}
}

/// Per-socket state (upstream `#serve` closure state).
struct HostSession {
	host: Arc<RadiusRelayHost>,
	socket: Arc<dyn RadiusRelayWebSocket>,
	writer: Arc<OrderedWebSocketWriter>,
	connections: parking_lot::Mutex<HashMap<String, ActiveConnection>>,
	/// Completes the serve future: `None` resolves it, `Some` rejects it
	/// (upstream `finish`).
	done: parking_lot::Mutex<Option<tokio::sync::oneshot::Sender<Option<AnyError>>>>,
}

impl HostSession {
	fn new(host: Arc<RadiusRelayHost>, socket: Arc<dyn RadiusRelayWebSocket>) -> Self {
		let writer = OrderedWebSocketWriter::new(Arc::clone(&socket));
		Self {
			host,
			socket,
			writer,
			connections: parking_lot::Mutex::new(HashMap::new()),
			done: parking_lot::Mutex::new(None),
		}
	}

	async fn serve(self: &Arc<Self>) -> Result<(), AnyError> {
		let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Option<AnyError>>();
		*self.done.lock() = Some(done_tx);
		let failure = done_rx.await.unwrap_or(None);
		// Idempotent terminal cleanup (the closer already ran it).
		let error_message = failure.as_ref().map(|failure| failure.to_string());
		self.done.lock().take();
		self.writer.close();
		self.drop_connections(error_message);
		match failure {
			Some(failure) => Err(failure),
			None => Ok(()),
		}
	}

	/// Terminal cleanup (upstream `finish`); `Some` rejects the serve future,
	/// `None` resolves it after forced teardown.
	fn finish(self: &Arc<Self>, failure: Option<AnyError>) {
		// `AnyError` is not `Clone`; connection drops carry the message.
		let error_message = failure.as_ref().map(|failure| failure.to_string());
		if let Some(done_tx) = self.done.lock().take() {
			let _ = done_tx.send(failure);
		}
		self.writer.close();
		self.drop_connections(error_message);
	}

	fn finish_ok(self: &Arc<Self>) {
		self.finish(None);
	}

	/// The connection-send closure (upstream `sendChunk`).
	async fn send_data(&self, connection_id: &str, chunk: Vec<u8>) -> Result<(), AnyError> {
		self.host.send_data(connection_id, chunk).await
	}

	/// The connection-close closure (upstream `closeConnection`).
	async fn server_close_connection(&self, connection_id: &str, final_chunk: Option<Vec<u8>>) -> Result<(), AnyError> {
		self.host.server_close_connection(connection_id, final_chunk).await
	}

	fn drop_connections(self: &Arc<Self>, failure_message: Option<String>) {
		let connections: Vec<ActiveConnection> = self.connections.lock().drain().map(|(_, active)| active).collect();
		for active in connections {
			active.connection.mark_closed();
			match &failure_message {
				Some(message) => active.handler.on_error(srv_error(message.clone())),
				None => active.handler.on_close(),
			}
		}
	}

	fn handle_message(self: &Arc<Self>, message: RelaySocketMessage) {
		match message {
			RelaySocketMessage::Text(text) => {
				let control = match parse_host_control_message(&text) {
					Ok(control) => control,
					Err(failure) => {
						close_web_socket(&self.socket, LOCAL_PROTOCOL_ERROR_CLOSE_CODE, "Radius relay protocol error");
						self.finish(Some(failure));
						return;
					}
				};
				match control {
					HostInputControlMessage::Ping => {
						let session = Arc::clone(self);
						tokio::spawn(async move {
							if session
								.writer
								.send(RelaySocketMessage::Text(
									HostOutputControlMessage::Pong.to_json().to_string(),
								))
								.await
								.is_err()
							{
								close_web_socket(
									&session.socket,
									LOCAL_TRANSPORT_ERROR_CLOSE_CODE,
									"Radius relay send failed",
								);
							}
						});
					}
					HostInputControlMessage::Pong => {}
					HostInputControlMessage::ConnectionOpen { connection_id } => self.open_connection(&connection_id),
					HostInputControlMessage::ConnectionClose { connection_id, .. } => self.remote_close_connection(&connection_id),
				}
			}
			RelaySocketMessage::Binary(bytes) => {
				let Some(frame) = parse_relay_dataframe(&bytes) else {
					close_web_socket(&self.socket, LOCAL_PROTOCOL_ERROR_CLOSE_CODE, "Radius relay protocol error");
					self.finish(Some(error("Invalid Radius relay data frame")));
					return;
				};
				let active = self.connections.lock().get(&frame.connection_id).map(|active| Arc::clone(&active.handler));
				match active {
					Some(handler) => handler.on_data(&frame.payload),
					None => {
						let session = Arc::clone(self);
						tokio::spawn(async move {
							let _ = session
								.writer
								.send(RelaySocketMessage::Text(
									HostOutputControlMessage::ConnectionClose {
										connection_id: frame.connection_id,
										code: Some(1000),
									}
									.to_json()
									.to_string(),
								))
								.await;
						});
					}
				}
			}
		}
	}

	fn open_connection(self: &Arc<Self>, connection_id: &str) {
		if self.connections.lock().contains_key(connection_id) {
			close_web_socket(&self.socket, LOCAL_PROTOCOL_ERROR_CLOSE_CODE, "Radius relay protocol error");
			self.finish(Some(error("Radius relay reused a connection ID")));
			return;
		}
		let connection = Arc::new(RelayServerByteConnection {
			session: parking_lot::Mutex::new(Some(Arc::clone(self))),
			connection_id: connection_id.to_string(),
			closed: AtomicBool::new(false),
		});
		let host = Arc::clone(&self.host);
		let handler = host.options.server.accept(connection.clone());
		if connection.closed() {
			let session = Arc::clone(self);
			let connection_id = connection_id.to_string();
			tokio::spawn(async move {
				let _ = session
					.writer
					.send(RelaySocketMessage::Text(
						HostOutputControlMessage::ConnectionClose { connection_id, code: Some(1012) }
							.to_json()
							.to_string(),
					))
					.await;
			});
		} else {
			self.connections.lock().insert(
				connection_id.to_string(),
				ActiveConnection { connection, handler },
			);
		}
	}

	fn remote_close_connection(self: &Arc<Self>, connection_id: &str) {
		let active = self.connections.lock().remove(connection_id);
		if let Some(active) = active {
			active.connection.mark_closed();
			active.handler.on_close();
		}
	}

	/// Rejections from connection callbacks close the socket like upstream's
	/// `onMessage` handler.
	fn fail(self: &Arc<Self>, failure: AnyError) {
		close_web_socket(&self.socket, LOCAL_PROTOCOL_ERROR_CLOSE_CODE, "Radius relay protocol error");
		self.finish(Some(failure));
	}
}

struct HostListener {
	session: Weak<HostSession>,
}

impl RadiusRelaySocketListener for HostListener {
	fn on_message(&self, message: RelaySocketMessage) {
		if let Some(session) = self.session.upgrade() {
			session.handle_message(message);
		}
	}

	fn on_close(&self, code: u16, reason: &str) {
		let Some(session) = self.session.upgrade() else {
			return;
		};
		if code == 1000 {
			session.finish_ok();
		} else {
			let detail = if reason.is_empty() { String::new() } else { format!(": {reason}") };
			session.finish(Some(error(format!("Radius relay host closed ({code}{detail})"))));
		}
	}

	fn on_error(&self, failure: AnyError) {
		if let Some(session) = self.session.upgrade() {
			session.finish(Some(failure));
		}
	}
}

fn close_web_socket(socket: &Arc<dyn RadiusRelayWebSocket>, code: u16, reason: &str) {
	// A WebSocket can reject close() while its opening handshake is still
	// pending; the seam swallows that.
	socket.close(code, reason);
}

async fn open_with_factory(
	factory: &RadiusRelayWebSocketFactory,
	url: &str,
	protocol: &str,
	auth: &RadiusRelayAuth,
) -> Result<Arc<dyn RadiusRelayWebSocket>, AnyError> {
	let socket = factory(RadiusRelayWebSocketRequest {
		url: url.to_string(),
		protocol: protocol.to_string(),
		authorization: format!("Bearer {}", auth.token),
	})
	.await?;
	if socket.protocol() != protocol {
		return Err(error(format!(
			"Radius relay selected unexpected WebSocket protocol {:?}",
			socket.protocol()
		)));
	}
	Ok(socket)
}

// ---------------------------------------------------------------------------
// Client transport (upstream `RadiusClientByteTransport` + factory).
// ---------------------------------------------------------------------------

/// Upstream `createRadiusClientTransportFactory`.
pub fn create_radius_client_transport_factory(options: RadiusClientTransportOptions) -> ByteTransportFactory {
	Arc::new(move |handlers: ByteTransportHandlers| {
		let options = options.clone();
		Box::pin(async move {
			let auth = options
				.auth
				.resolve(&RadiusRelayResolveOptions { required: true, signal: None })
				.await?
				.ok_or_else(|| error("Radius authentication is required"))?;
			let socket = open_with_factory(
				&options.web_socket_factory.clone().unwrap_or_else(default_web_socket_factory),
				&relay_web_socket_url(&auth.gateway, &options.server_id)?,
				RADIUS_RELAY_CLIENT_SUBPROTOCOL,
				&auth,
			)
			.await?;
			Ok(Box::new(RadiusClientByteTransport::new(socket, handlers)) as Box<dyn ByteTransport>)
		})
	})
}

/// Upstream factory options.
#[derive(Clone)]
pub struct RadiusClientTransportOptions {
	pub server_id: String,
	pub auth: Arc<RadiusRelayAuthResolver>,
	pub web_socket_factory: Option<RadiusRelayWebSocketFactory>,
}

/// The client-facing relay transport (`RadiusClientByteTransport`).
pub struct RadiusClientByteTransport {
	core: Arc<TransportCore>,
}

struct TransportCore {
	socket: Arc<dyn RadiusRelayWebSocket>,
	writer: Arc<OrderedWebSocketWriter>,
	handlers: ByteTransportHandlers,
	closed: AtomicBool,
}

impl TransportCore {
	fn mark_closed(self: &Arc<Self>) -> bool {
		if self.closed.swap(true, Ordering::SeqCst) {
			return false;
		}
		self.writer.close();
		true
	}

	fn fail(self: &Arc<Self>, failure: AnyError) {
		if !self.mark_closed() {
			return;
		}
		close_web_socket(&self.socket, LOCAL_TRANSPORT_ERROR_CLOSE_CODE, "Radius relay transport error");
		(self.handlers.on_error)(failure);
	}
}

struct TransportListener {
	core: std::sync::Weak<TransportCore>,
}

impl RadiusRelaySocketListener for TransportListener {
	fn on_message(&self, message: RelaySocketMessage) {
		let Some(core) = self.core.upgrade() else {
			return;
		};
		if core.closed.load(Ordering::SeqCst) {
			return;
		}
		match message {
			RelaySocketMessage::Binary(bytes) => (core.handlers.on_data)(bytes),
			RelaySocketMessage::Text(_) => core.fail(error("Radius relay client received a non-binary message")),
		}
	}

	fn on_close(&self, _code: u16, _reason: &str) {
		if let Some(core) = self.core.upgrade() {
			if core.mark_closed() {
				(core.handlers.on_close)();
			}
		}
	}

	fn on_error(&self, failure: AnyError) {
		if let Some(core) = self.core.upgrade() {
			core.fail(failure);
		}
	}
}

impl RadiusClientByteTransport {
	fn new(socket: Arc<dyn RadiusRelayWebSocket>, handlers: ByteTransportHandlers) -> Self {
		let core = Arc::new(TransportCore {
			writer: OrderedWebSocketWriter::new(Arc::clone(&socket)),
			socket,
			handlers,
			closed: AtomicBool::new(false),
		});
		let listener = TransportListener { core: Arc::downgrade(&core) };
		core.socket.set_listener(Some(Arc::new(listener)));
		Self { core }
	}
}

#[async_trait::async_trait]
impl ByteTransport for RadiusClientByteTransport {
	async fn send(&self, chunk: Vec<u8>) -> Result<(), AnyError> {
		if self.core.closed.load(Ordering::SeqCst) {
			return Err(error("Radius relay client is closed"));
		}
		self.core.writer.send(RelaySocketMessage::Binary(chunk)).await
	}

	fn close(&self) {
		if !self.core.mark_closed() {
			return;
		}
		self.core.socket.close(1000, "Pi client closed");
	}
}

// ---------------------------------------------------------------------------
// Client reconnect (upstream `RadiusClientReconnect`).
// ---------------------------------------------------------------------------

/// Reattach callback (upstream `reattach: (sessionId) => Promise<void>`).
pub type ReattachFn = Arc<dyn Fn(String) -> futures_util::future::BoxFuture<'static, Result<(), AnyError>> + Send + Sync>;

/// State shared between the reconnect helper and its client listeners
/// (upstream private fields).
struct ReconnectCore {
	client: pi_client::Client,
	reattach: ReattachFn,
	desired_session_id: parking_lot::Mutex<Option<String>>,
	/// The in-flight reconnect loop (upstream `#reconnecting`).
	reconnecting: parking_lot::Mutex<Option<tokio::task::JoinHandle<()>>>,
	disposed: AtomicBool,
}

impl ReconnectCore {
	fn start_reconnect(self: &Arc<Self>) {
		let mut reconnecting = self.reconnecting.lock();
		if reconnecting.is_some() {
			return;
		}
		let core = Arc::clone(self);
		*reconnecting = Some(tokio::spawn(async move {
			run_reconnect(core).await;
		}));
	}
}

/// Reconnect one established Radius client and restore its last selected
/// Session (upstream `RadiusClientReconnect`).
pub struct RadiusClientReconnect {
	core: Arc<ReconnectCore>,
	subscriptions: parking_lot::Mutex<Vec<pi_client::Unsubscribe>>,
}

impl RadiusClientReconnect {
	pub fn new(client: pi_client::Client, reattach: ReattachFn) -> Result<Self, AnyError> {
		let core = Arc::new(ReconnectCore {
			client,
			reattach,
			desired_session_id: parking_lot::Mutex::new(None),
			reconnecting: parking_lot::Mutex::new(None),
			disposed: AtomicBool::new(false),
		});
		*core.desired_session_id.lock() = core.client.attachment().map(|attachment| attachment.session_id);
		let mut subscriptions = Vec::new();
		{
			let core_for_attachment = Arc::clone(&core);
			let attachment_listener: pi_client::AttachmentChangeListener =
				Arc::new(move |attachment: Option<&pi_client::SessionTarget>| {
					let mut desired = core_for_attachment.desired_session_id.lock();
					match attachment {
						Some(attachment) => *desired = Some(attachment.session_id.clone()),
						None => {
							if core_for_attachment.client.connected() {
								*desired = None;
							}
						}
					}
				});
			subscriptions.push(core.client.on_attachment_change(attachment_listener)?);
		}
		{
			let core_for_state = Arc::clone(&core);
			let state_listener: pi_client::ConnectionChangeListener = Arc::new(move |change| {
				if change.state == pi_client::ConnectionState::Disconnected && !core_for_state.disposed.load(Ordering::SeqCst)
				{
					core_for_state.start_reconnect();
				}
			});
			subscriptions.push(core.client.on_connection_state_change(state_listener)?);
		}
		Ok(Self { core, subscriptions: parking_lot::Mutex::new(subscriptions) })
	}

	/// `dispose()`.
	pub async fn dispose(&self) {
		if self.core.disposed.swap(true, Ordering::SeqCst) {
			return;
		}
		for unsubscribe in self.subscriptions.lock().drain(..) {
			unsubscribe();
		}
		// Aborting the loop mirrors the upstream abort of `delay`; the loop
		// clears `#reconnecting` on its way out.
		let handle = self.core.reconnecting.lock().take();
		if let Some(handle) = handle {
			handle.abort();
			let _ = handle.await;
		}
		if self.core.client.connection_state() != pi_client::ConnectionState::Disconnected {
			self.core.client.disconnect("Radius reconnect stopped");
		}
	}
}

async fn run_reconnect(core: Arc<ReconnectCore>) {
	let mut retry_ms = CLIENT_RETRY_INITIAL_MS;
	while !core.disposed.load(Ordering::SeqCst) && !core.client.connected() {
		let result = async {
			tokio::time::sleep(Duration::from_millis(retry_ms)).await;
			core.client.reconnect().await?;
			let session_id = core.desired_session_id.lock().clone();
			if let Some(session_id) = session_id {
				(core.reattach)(session_id).await?;
			}
			Ok::<(), AnyError>(())
		}
		.await;
		match result {
			Ok(()) => break,
			Err(reconnect_error) => {
				if core.disposed.load(Ordering::SeqCst) {
					break;
				}
				if core.client.connected() {
					core.client.disconnect(&reconnect_error.to_string());
				}
				retry_ms = (retry_ms * 2).min(CLIENT_RETRY_MAX_MS);
			}
		}
	}
	*core.reconnecting.lock() = None;
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn connection_id_validation_matches_upstream_pattern() {
		assert!(is_connection_id("123e4567-e89b-42d3-a456-426614174000"));
		assert!(!is_connection_id("123e4567-e89b-12d3-a456-426614174000")); // not v4
		assert!(!is_connection_id("123E4567-e89b-42d3-a456-426614174000")); // uppercase
		assert!(!is_connection_id("123e4567e89b42d3a456426614174000"));
		assert!(!is_connection_id(""));
	}

	#[test]
	fn control_messages_round_trip() {
		let parsed = parse_host_control_message(r#"{"version":1,"type":"ping"}"#).unwrap();
		assert_eq!(parsed, HostInputControlMessage::Ping);
		let parsed = parse_host_control_message(
			r#"{"version":1,"type":"connection_open","connection_id":"123e4567-e89b-42d3-a456-426614174000"}"#,
		)
		.unwrap();
		assert_eq!(
			parsed,
			HostInputControlMessage::ConnectionOpen { connection_id: "123e4567-e89b-42d3-a456-426614174000".into() }
		);
		let parsed = parse_host_control_message(
			r#"{"version":1,"type":"connection_close","connection_id":"123e4567-e89b-42d3-a456-426614174000","code":1000}"#,
		)
		.unwrap();
		assert_eq!(
			parsed,
			HostInputControlMessage::ConnectionClose {
				connection_id: "123e4567-e89b-42d3-a456-426614174000".into(),
				code: Some(1000)
			}
		);
		assert!(parse_host_control_message(r#"{"version":2,"type":"ping"}"#).is_err());
		assert!(parse_host_control_message(r#"{"version":1,"type":"nope"}"#).is_err());
		assert!(parse_host_control_message(
			r#"{"version":1,"type":"connection_open","connection_id":"123e4567-e89b-12d3-a456-426614174000"}"#
		)
		.is_err());
		assert!(parse_host_control_message("not json").is_err());
		assert!(parse_host_control_message(
			r#"{"version":1,"type":"connection_close","connection_id":"123e4567-e89b-42d3-a456-426614174000","code":5000}"#
		)
		.is_err());
	}

	#[test]
	fn data_frames_round_trip() {
		let connection_id = "123e4567-e89b-42d3-a456-426614174000";
		let frame = encode_relay_dataframe(connection_id, b"payload").unwrap();
		assert_eq!(frame.len(), RELAY_DATA_HEADER_BYTES + 7);
		assert_eq!(frame[0], RELAY_DATA_FRAME_VERSION);
		assert_eq!(frame[1], RELAY_DATA_FRAME_TYPE);
		let parsed = parse_relay_dataframe(&frame).unwrap();
		assert_eq!(parsed.connection_id, connection_id);
		assert_eq!(parsed.payload, b"payload");
		// Short, wrong version, wrong type, and non-v4 IDs do not parse.
		assert!(parse_relay_dataframe(&frame[..8]).is_none());
		let mut wrong_version = frame.clone();
		wrong_version[0] = 2;
		assert!(parse_relay_dataframe(&wrong_version).is_none());
		let mut wrong_type = frame.clone();
		wrong_type[1] = 2;
		assert!(parse_relay_dataframe(&wrong_type).is_none());
		let mut wrong_id = frame;
		// Byte 6 of the connection id carries the v4 version nibble.
		wrong_id[8] = b'1';
		assert!(parse_relay_dataframe(&wrong_id).is_none());
		assert!(encode_relay_dataframe("not-an-id", b"").is_err());
	}

	#[test]
	fn relay_urls_upgrade_scheme() {
		assert_eq!(
			relay_web_socket_url("https://gateway.example", "sid").unwrap(),
			"wss://gateway.example/v1/session-relays/sid/connect"
		);
		assert_eq!(
			relay_web_socket_url("http://localhost:9000/", "sid").unwrap(),
			"ws://localhost:9000/v1/session-relays/sid/connect"
		);
		assert!(relay_web_socket_url("ftp://gateway.example", "sid").is_err());
	}
}
