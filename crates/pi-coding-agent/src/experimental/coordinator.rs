//! Port of `packages/coding-agent/src/experimental/coordinator.ts` — the
//! coordinator transport shim: a deliberately opaque JSON-line message
//! router over Unix sockets, plus its client endpoint and startup helper.
//!
//! Like upstream, this process never interprets Pi, session, worker, or
//! lifecycle payloads: control messages are routed by `type` string only and
//! application payloads stay opaque [`serde_json::Value`]s.
//!
//! # Deviations
//!
//! - **D139 — coordinator seams.**
//!   - Node `net.Socket` becomes [`tokio::net::UnixStream`]; the typebox
//!     wire schema becomes the serde enum [`CoordinatorMessage`] with
//!     identical wire names; the loose `ControlMessage` server side parses
//!     from `serde_json::Value` with the same `type` dispatch.
//!   - Upstream keeps module-global coordinator state (a per-process
//!     singleton guarded by `running`) with `process.once("SIGINT"/"SIGTERM")`
//!     hooks and `process.exit(0)`; the port keeps the state in a
//!     function-local [`CoordinatorServer`], races [`tokio::signal::ctrl_c`]
//!     (SIGINT; SIGTERM delivery to the spawned detached process is the
//!     spawner's concern here), performs the same socket cleanup, and
//!     returns. A second concurrent `run_coordinator_process` call in one
//!     process is still rejected ("Coordinator process is already running").
//!   - `emptyTimer.unref()` becomes an abortable tokio sleep task with the
//!     same grace windows and emptiness predicate.
//!   - Upstream's uncaught exception in a control handler crashes the whole
//!     coordinator process; the port surfaces the same error string through
//!     `run_coordinator_process` so the entrypoint exits non-zero, mirroring
//!     the crash. Reader-level framing errors destroy only the offending
//!     connection, as upstream.
//!   - `createConnection(endpoint)` accepts Unix-socket paths and
//!     `host:port` TCP endpoints; the public-connection pump keeps raw
//!     bytes bidirectional (`pipe` → `copy_bidirectional`).
//!   - `chmod 0o600` restriction applies on non-Windows only, as upstream
//!     (`restrictSocket`).

use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use super::process::{encode_control_line, spawn_internal_process, InternalProcessRole, LineReader};

/// Coordinator control-protocol version.
pub const COORDINATOR_PROTOCOL_VERSION: u64 = 3;
const COORDINATOR_START_TIMEOUT_MS: u64 = 10_000;
const COORDINATOR_RETRY_MS: u64 = 10;

const EMPTY_STARTUP_GRACE_MS: u64 = 30_000;
const EMPTY_SHUTDOWN_GRACE_MS: u64 = 250;

/// Messages the coordinator sends to a registered server or peer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CoordinatorMessage {
	ServerRegistered {
		#[serde(rename = "serverConnectionId")]
		server_connection_id: String,
		peers: Vec<String>,
	},
	ServerReplaced,
	PeerConnected {
		#[serde(rename = "peerId")]
		peer_id: String,
	},
	PeerDisconnected {
		#[serde(rename = "peerId")]
		peer_id: String,
	},
	Message {
		from: String,
		payload: serde_json::Value,
	},
}

/// Events surfaced by [`CoordinatorConnection::on_event`].
#[derive(Debug, Clone, PartialEq)]
pub enum CoordinatorConnectionEvent {
	PeerConnected { peer_id: String },
	PeerDisconnected { peer_id: String },
	Message { from: String, payload: serde_json::Value },
}

// ---------------------------------------------------------------------------
// Client endpoint (`CoordinatorConnection`).
// ---------------------------------------------------------------------------

type Listener = Arc<dyn Fn(&CoordinatorConnectionEvent) + Send + Sync>;

struct ClientState {
	writer: parking_lot::Mutex<Option<Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>>>,
	registered: bool,
	closed: bool,
	replaced_tx: tokio::sync::watch::Sender<bool>,
	peer_ids: BTreeSet<String>,
	listeners: BTreeMap<u64, Listener>,
	next_listener_id: u64,
	registered_tx: Option<oneshot::Sender<Result<(), String>>>,
	server_connection_id: String,
}

/// Options for [`CoordinatorConnection::new`] (`CoordinatorConnectionOptions`).
pub struct CoordinatorConnectionOptions {
	pub control_path: String,
	pub endpoint: String,
	pub server_connection_id: Option<String>,
}

/// The server-side endpoint of the coordinator's intentionally opaque
/// message router.
pub struct CoordinatorConnection {
	pub server_connection_id: String,
	control_path: String,
	endpoint: String,
	state: Arc<Mutex<ClientState>>,
	replaced_rx: tokio::sync::watch::Receiver<bool>,
}

/// Drops the listener registration (upstream's unsubscribe closure).
pub struct EventGuard {
	state: Arc<Mutex<ClientState>>,
	id: u64,
}

impl Drop for EventGuard {
	fn drop(&mut self) {
		self.state.lock().listeners.remove(&self.id);
	}
}

impl CoordinatorConnection {
	pub fn new(options: CoordinatorConnectionOptions) -> Self {
		let server_connection_id = options
			.server_connection_id
			.unwrap_or_else(super::session_catalog::new_session_id);
		let (replaced_tx, replaced_rx) = tokio::sync::watch::channel(false);
		Self {
			server_connection_id: server_connection_id.clone(),
			control_path: options.control_path,
			endpoint: options.endpoint,
			state: Arc::new(Mutex::new(ClientState {
				writer: parking_lot::Mutex::new(None),
				registered: false,
				closed: false,
				replaced_tx,
				peer_ids: BTreeSet::new(),
				listeners: BTreeMap::new(),
				next_listener_id: 0,
				registered_tx: None,
				server_connection_id,
			})),
			replaced_rx,
		}
	}

	pub fn control_path(&self) -> &str {
		&self.control_path
	}

	pub fn endpoint(&self) -> &str {
		&self.endpoint
	}

	pub fn was_replaced(&self) -> bool {
		*self.replaced_rx.borrow()
	}

	/// Resolves once the connection was replaced or dropped (upstream's
	/// `replaced` promise).
	pub async fn replaced(&self) {
		let mut rx = self.replaced_rx.clone();
		loop {
			if *rx.borrow_and_update() {
				return;
			}
			if rx.changed().await.is_err() {
				return;
			}
		}
	}

	/// Snapshot of the known peer ids (upstream exposes the live `peerIds`
	/// set; the port returns an owned copy).
	pub fn peer_ids(&self) -> BTreeSet<String> {
		self.state.lock().peer_ids.clone()
	}

	/// Register an event listener; the returned guard removes it on drop
	/// (upstream returns an unsubscribe closure).
	pub fn on_event(&self, listener: impl Fn(&CoordinatorConnectionEvent) + Send + Sync + 'static) -> EventGuard {
		let listener: Listener = Arc::new(listener);
		let mut state = self.state.lock();
		let id = state.next_listener_id;
		state.next_listener_id += 1;
		state.listeners.insert(id, listener);
		EventGuard { state: Arc::clone(&self.state), id }
	}

	/// Connect and register as the coordinator's current server.
	pub async fn connect(&self) -> Result<(), String> {
		if self.state.lock().writer.lock().is_some() {
			return Err("Coordinator server is already connected".into());
		}
		let socket = UnixStream::connect(&self.control_path)
			.await
			.map_err(|error| error.to_string())?;
		let (reader, writer) = socket.into_split();
		{
			let state = self.state.lock();
			let mut slot = state.writer.lock();
			if slot.is_some() {
				return Err("Coordinator server is already connected".into());
			}
			*slot = Some(Arc::new(tokio::sync::Mutex::new(writer)));
		}
		let (registered_tx, registered_rx) = oneshot::channel();
		self.state.lock().registered_tx = Some(registered_tx);
		spawn_client_reader(Arc::clone(&self.state), reader);
		let register = serde_json::json!({
			"type": "register_server",
			"protocol": COORDINATOR_PROTOCOL_VERSION,
			"serverConnectionId": self.server_connection_id,
			"endpoint": self.endpoint,
		});
		write_message(&self.state, &register).await?;
		match registered_rx.await {
			Ok(result) => result,
			// The oneshot was consumed elsewhere (a later close/reject) —
			// treat like the upstream unawaited close rejection.
			Err(_) => Err("Coordinator connection closed".into()),
		}
	}

	/// Send a payload to one peer by id (or `"server"`).
	pub async fn send(&self, peer_id: &str, payload: serde_json::Value) -> Result<(), String> {
		self.write(serde_json::json!({ "type": "send", "to": peer_id, "payload": payload }))
			.await
	}

	/// Send a payload to every peer (coordinator enforces server-only).
	pub async fn broadcast(&self, payload: serde_json::Value) -> Result<(), String> {
		self.write(serde_json::json!({ "type": "broadcast", "payload": payload }))
			.await
	}

	async fn write(&self, message: serde_json::Value) -> Result<(), String> {
		{
			let state = self.state.lock();
			if !state.registered || state.closed || state.writer.lock().is_none() {
				return Err("Coordinator server is not connected".into());
			}
		}
		write_message(&self.state, &message).await
	}

	/// Close the connection (`close()`).
	pub fn close(&self) {
		let mut state = self.state.lock();
		if state.closed {
			return;
		}
		state.closed = true;
		*state.writer.lock() = None;
		state.peer_ids.clear();
		if let Some(tx) = state.registered_tx.take() {
			let _ = tx.send(Err("Coordinator server closed".into()));
		}
	}
}

fn emit(state: &Arc<Mutex<ClientState>>, event: CoordinatorConnectionEvent) {
	let state = state.lock();
	for listener in state.listeners.values() {
		listener(&event);
	}
}

async fn write_message(
	state: &Arc<Mutex<ClientState>>,
	message: &serde_json::Value,
) -> Result<(), String> {
	let line = encode_control_line(message)?;
	// Clone the shared write half out of the parking-lot guard so no
	// non-Send guard is held across the await.
	let writer = {
		let state = state.lock();
		let slot = state.writer.lock();
		match slot.as_ref() {
			Some(writer) => Arc::clone(writer),
			None => return Err("Coordinator server is not connected".into()),
		}
	};
	let mut writer = writer.lock().await;
	writer
		.write_all(line.as_bytes())
		.await
		.map_err(|error| error.to_string())
}

fn spawn_client_reader(state: Arc<Mutex<ClientState>>, mut reader: tokio::net::unix::OwnedReadHalf) {
	tokio::spawn(async move {
		let mut line_reader = LineReader::new();
		let mut buffer = vec![0u8; 64 * 1024];
		loop {
			let read = match reader.read(&mut buffer).await {
				Ok(0) | Err(_) => break,
				Ok(read) => read,
			};
			let messages = match line_reader.push(&buffer[..read]) {
				Ok(messages) => messages,
				Err(_) => break,
			};
			let mut alive = true;
			for message in messages {
				if !handle_client_message(&state, message).await {
					alive = false;
					break;
				}
			}
			if !alive {
				break;
			}
		}
		disconnect_client(&state).await;
	});
}

async fn disconnect_client(state: &Arc<Mutex<ClientState>>) {
	*state.lock().writer.lock() = None;
	let (registered_tx, closed) = {
		let mut state = state.lock();
		(state.registered_tx.take(), state.closed)
	};
	if !closed {
		if let Some(tx) = registered_tx {
			let _ = tx.send(Err("Coordinator connection closed".into()));
		}
	}
	mark_replaced(state);
}

/// Destroy-or-continue handler for one decoded client message. `false`
/// means the connection must be destroyed.
async fn handle_client_message(state: &Arc<Mutex<ClientState>>, value: serde_json::Value) -> bool {
	let Ok(message) = serde_json::from_value::<CoordinatorMessage>(value) else {
		*state.lock().writer.lock() = None;
		return false;
	};
	match message {
		CoordinatorMessage::ServerRegistered {
			server_connection_id,
			peers,
		} => {
			let mut state = state.lock();
			if server_connection_id != state.server_connection_id {
				*state.writer.lock() = None;
				// Registration fails like upstream's socket destroy.
				let _ = state.replaced_tx.send(true);
				return false;
			}
			state.peer_ids.extend(peers);
			state.registered = true;
			if let Some(tx) = state.registered_tx.take() {
				let _ = tx.send(Ok(()));
			}
			true
		}
		CoordinatorMessage::ServerReplaced => {
			mark_replaced(state);
			true
		}
		CoordinatorMessage::PeerConnected { peer_id } => {
			state.lock().peer_ids.insert(peer_id.clone());
			emit(state, CoordinatorConnectionEvent::PeerConnected { peer_id });
			true
		}
		CoordinatorMessage::PeerDisconnected { peer_id } => {
			state.lock().peer_ids.remove(&peer_id);
			emit(state, CoordinatorConnectionEvent::PeerDisconnected { peer_id });
			true
		}
		CoordinatorMessage::Message { from, payload } => {
			emit(state, CoordinatorConnectionEvent::Message { from, payload });
			true
		}
	}
}

fn mark_replaced(state: &Arc<Mutex<ClientState>>) {
	let _ = state.lock().replaced_tx.send(true);
}

// ---------------------------------------------------------------------------
// Startup helper (`ensureCoordinator`).
// ---------------------------------------------------------------------------

/// A connected probe socket held open so the coordinator stays non-empty.
#[derive(Debug)]
pub struct CoordinatorStartupLease {
	socket: Option<UnixStream>,
}

impl CoordinatorStartupLease {
	/// Destroy the probe socket (upstream `lease.close()`).
	pub fn close(&mut self) {
		self.socket = None;
	}
}

/// Connect or map the upstream swallow list (`ENOENT`/`ECONNREFUSED`) to
/// `None`.
async fn try_connect(path: &str) -> Result<Option<UnixStream>, String> {
	match UnixStream::connect(path).await {
		Ok(socket) => Ok(Some(socket)),
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
		Err(error) if error.kind() == std::io::ErrorKind::ConnectionRefused => Ok(None),
		Err(error) => Err(error.to_string()),
	}
}

/// Ensure a coordinator process is listening on `control_path`, starting one
/// if necessary (upstream `ensureCoordinator`).
pub async fn ensure_coordinator(
	public_path: &str,
	control_path: &str,
) -> Result<CoordinatorStartupLease, String> {
	if let Some(socket) = try_connect(control_path).await? {
		return Ok(CoordinatorStartupLease { socket: Some(socket) });
	}
	let args = vec![public_path.to_string(), control_path.to_string()];
	let mut child = spawn_internal_process(InternalProcessRole::Coordinator, &args, &Default::default())
		.map_err(|error| error.to_string())?;
	let deadline = Instant::now() + Duration::from_millis(COORDINATOR_START_TIMEOUT_MS);
	loop {
		if let Some(socket) = try_connect(control_path).await? {
			return Ok(CoordinatorStartupLease { socket: Some(socket) });
		}
		if matches!(child.try_wait(), Ok(Some(_))) {
			return Err("Coordinator exited during startup".into());
		}
		if Instant::now() >= deadline {
			return Err("Timed out waiting for coordinator startup".into());
		}
		tokio::time::sleep(Duration::from_millis(COORDINATOR_RETRY_MS)).await;
	}
}

// ---------------------------------------------------------------------------
// Coordinator process (the transport shim).
// ---------------------------------------------------------------------------

type LineTx = mpsc::UnboundedSender<String>;

struct ServerPeerEntry {
	server_connection_id: String,
	endpoint: String,
	tx: LineTx,
}

#[derive(Default)]
struct ServerState {
	shutting_down: bool,
	peers: BTreeMap<String, LineTx>,
	control_tasks: BTreeMap<u64, JoinHandle<()>>,
	public_tasks: BTreeMap<u64, JoinHandle<()>>,
	current_server: Option<ServerPeerEntry>,
	next_connection_id: u64,
}

enum Fatal {
	Empty,
	Crash(String),
}

struct CoordinatorServer {
	#[allow(dead_code)] // kept for parity with the upstream module globals
	public_path: String,
	#[allow(dead_code)]
	control_path: String,
	state: tokio::sync::Mutex<ServerState>,
	timer: tokio::sync::Mutex<Option<JoinHandle<()>>>,
	fatal_tx: mpsc::UnboundedSender<Fatal>,
}

impl CoordinatorServer {
	fn new(public_path: String, control_path: String) -> (Arc<Self>, mpsc::UnboundedReceiver<Fatal>) {
		let (fatal_tx, fatal_rx) = mpsc::unbounded_channel();
		let server = Arc::new(Self {
			public_path,
			control_path,
			state: tokio::sync::Mutex::new(ServerState::default()),
			timer: tokio::sync::Mutex::new(None),
			fatal_tx,
		});
		(server, fatal_rx)
	}

	fn notify_peers(&self, state: &ServerState, message: &serde_json::Value) {
		for peer in state.peers.values() {
			write_routed_line_tx(peer, message);
		}
	}

	async fn check_empty(self: &Arc<Self>) {
		{
			let state = self.state.lock().await;
			if state.shutting_down
				|| state.current_server.is_some()
				|| !state.peers.is_empty()
				|| !state.public_tasks.is_empty()
				|| !state.control_tasks.is_empty()
			{
				drop(state);
				self.cancel_empty_shutdown().await;
				return;
			}
		}
		self.schedule_empty_shutdown(EMPTY_SHUTDOWN_GRACE_MS).await;
	}

	async fn schedule_empty_shutdown(self: &Arc<Self>, delay_ms: u64) {
		{
			let timer = self.timer.lock().await;
			let state = self.state.lock().await;
			if timer.is_some() || state.shutting_down {
				return;
			}
		}
		let server = Arc::clone(self);
		let handle = tokio::spawn(async move {
			tokio::time::sleep(Duration::from_millis(delay_ms)).await;
			let state = server.state.lock().await;
			if state.current_server.is_none()
				&& state.peers.is_empty()
				&& state.public_tasks.is_empty()
				&& state.control_tasks.is_empty()
			{
				let _ = server.fatal_tx.send(Fatal::Empty);
			}
		});
		*self.timer.lock().await = Some(handle);
	}

	async fn cancel_empty_shutdown(&self) {
		if let Some(handle) = self.timer.lock().await.take() {
			handle.abort();
		}
	}

	async fn accept_control_connection(self: &Arc<Self>, socket: UnixStream) {
		{
			let mut state = self.state.lock().await;
			if state.shutting_down {
				drop(state);
				return;
			}
			let id = state.next_connection_id;
			state.next_connection_id += 1;
			let server = Arc::clone(self);
			let handle = tokio::spawn(async move {
				server.run_control_connection(id, socket).await;
			});
			state.control_tasks.insert(id, handle);
		}
		self.cancel_empty_shutdown().await;
	}

	/// Per-connection control loop. Errors that upstream would throw inside
	/// the connection callback become fatal coordinator errors (D139).
	async fn run_control_connection(self: Arc<Self>, connection_id: u64, socket: UnixStream) {
		let (line_tx, mut line_rx) = mpsc::unbounded_channel::<String>();
		let (mut reader, mut writer) = socket.into_split();
		let writer_task = tokio::spawn(async move {
			while let Some(line) = line_rx.recv().await {
				if writer.write_all(line.as_bytes()).await.is_err() {
					break;
				}
			}
		});
		let mut role_server: Option<ServerPeerEntry> = None;
		let mut role_peer: Option<String> = None;
		let mut role_peer_tx: Option<LineTx> = None;
		let mut line_reader = LineReader::new();
		let mut buffer = vec![0u8; 64 * 1024];
		let mut outcome: Result<(), String> = Ok(());
		'read: loop {
			let read = match reader.read(&mut buffer).await {
				Ok(0) | Err(_) => break,
				Ok(read) => read,
			};
			let messages = match line_reader.push(&buffer[..read]) {
				Ok(messages) => messages,
				Err(_) => break, // destroy the connection on framing errors
			};
			for value in messages {
				let Some(message_type) = value.get("type").and_then(serde_json::Value::as_str).map(String::from)
				else {
					break 'read; // "Coordinator message must have a type"
				};
				let result = self
					.handle_control_message(
						&line_tx,
						&mut role_server,
						&mut role_peer,
						&mut role_peer_tx,
						&message_type,
						&value,
					)
					.await;
				if let Err(error) = result {
					outcome = Err(error);
					break 'read;
				}
			}
		}
		let was_server_id = role_server.as_ref().map(|server| server.server_connection_id.clone());
		let was_peer = role_peer.take();
		let was_peer_tx = role_peer_tx.take();
		if let Err(error) = outcome {
			let _ = self.fatal_tx.send(Fatal::Crash(error));
			writer_task.abort();
			return;
		}
		{
			let mut state = self.state.lock().await;
			state.control_tasks.remove(&connection_id);
			if let Some(server_connection_id) = &was_server_id {
				let replaced = state
					.current_server
					.as_ref()
					.map(|server| &server.server_connection_id == server_connection_id)
					.unwrap_or(false);
				if replaced {
					let entry = state.current_server.take().expect("checked above");
					let message = serde_json::json!({
						"type": "server_disconnected",
						"serverConnectionId": entry.server_connection_id,
					});
					self.notify_peers(&state, &message);
				}
			}
			if let (Some(peer_id), Some(peer_tx)) = (&was_peer, &was_peer_tx) {
				if state.peers.get(peer_id).map(|tx| tx.same_channel(peer_tx)).unwrap_or(false) {
					state.peers.remove(peer_id);
					if let Some(server) = state.current_server.as_ref() {
						write_routed_line(
							server,
							&serde_json::json!({ "type": "peer_disconnected", "peerId": peer_id }),
						);
					}
				}
			}
		}
		writer_task.abort();
		self.check_empty().await;
	}

	async fn handle_control_message(
		self: &Arc<Self>,
		line_tx: &LineTx,
		role_server: &mut Option<ServerPeerEntry>,
		role_peer: &mut Option<String>,
		role_peer_tx: &mut Option<LineTx>,
		message_type: &str,
		value: &serde_json::Value,
	) -> Result<(), String> {
		if role_server.is_none() && role_peer.is_none() {
			match message_type {
				"register_server" => {
					*role_server = Some(self.register_server(line_tx.clone(), value).await?);
					return Ok(());
				}
				"register_peer" => {
					*role_peer = Some(self.register_peer(line_tx.clone(), value).await?);
					*role_peer_tx = Some(line_tx.clone());
					return Ok(());
				}
				_ => return Err("Coordinator connection did not register a role".into()),
			}
		}
		if let Some(server) = role_server {
			let is_current = {
				let state = self.state.lock().await;
				state
					.current_server
					.as_ref()
					.map(|current| current.tx.same_channel(&server.tx))
					.unwrap_or(false)
			};
			if is_current {
				self.handle_routed_message("server", message_type, value).await?;
			}
			return Ok(());
		}
		let peer_id = role_peer.clone().expect("peer role set");
		self.handle_routed_message(&peer_id, message_type, value).await
	}

	async fn register_server(self: &Arc<Self>, line_tx: LineTx, value: &serde_json::Value) -> Result<ServerPeerEntry, String> {
		if value.get("protocol").and_then(serde_json::Value::as_u64) != Some(COORDINATOR_PROTOCOL_VERSION) {
			return Err("Unsupported coordinator protocol".into());
		}
		let server_connection_id = string_field(value, "serverConnectionId")
			.ok_or("Coordinator serverConnectionId must be a string")?;
		let endpoint = string_field(value, "endpoint").ok_or("Coordinator endpoint must be a string")?;
		let entry = ServerPeerEntry {
			server_connection_id: server_connection_id.clone(),
			endpoint: endpoint.clone(),
			tx: line_tx.clone(),
		};
		let mut state = self.state.lock().await;
		let previous = state.current_server.take();
		state.current_server = Some(ServerPeerEntry {
			server_connection_id: server_connection_id.clone(),
			endpoint,
			tx: line_tx.clone(),
		});
		drop(state);
		self.cancel_empty_shutdown().await;
		let state = self.state.lock().await;
		let peer_ids: Vec<String> = state.peers.keys().cloned().collect();
		send_line(
			&line_tx,
			&serde_json::json!({
				"type": "server_registered",
				"serverConnectionId": server_connection_id,
				"peers": peer_ids,
			}),
		);
		if let Some(previous) = previous {
			drop(state);
			self.close_public_connections().await;
			let state = self.state.lock().await;
			self.notify_peers(
				&state,
				&serde_json::json!({
					"type": "server_disconnected",
					"serverConnectionId": previous.server_connection_id,
				}),
			);
			write_routed_line(&previous, &serde_json::json!({ "type": "server_replaced" }));
			let state = self.state.lock().await;
			self.notify_peers(
				&state,
				&serde_json::json!({
					"type": "server_connected",
					"serverConnectionId": server_connection_id,
				}),
			);
			return Ok(entry);
		}
		self.notify_peers(
			&state,
			&serde_json::json!({
				"type": "server_connected",
				"serverConnectionId": server_connection_id,
			}),
		);
		Ok(entry)
	}

	async fn register_peer(self: &Arc<Self>, line_tx: LineTx, value: &serde_json::Value) -> Result<String, String> {
		if value.get("protocol").and_then(serde_json::Value::as_u64) != Some(COORDINATOR_PROTOCOL_VERSION) {
			return Err("Unsupported coordinator protocol".into());
		}
		let peer_id = string_field(value, "peerId").ok_or("Coordinator peerId must be a string")?;
		let mut state = self.state.lock().await;
		if peer_id == "server" || state.peers.contains_key(&peer_id) {
			return Err(format!("Coordinator peer is already connected: {peer_id}"));
		}
		state.peers.insert(peer_id.clone(), line_tx.clone());
		drop(state);
		self.cancel_empty_shutdown().await;
		let state = self.state.lock().await;
		let mut registered = serde_json::json!({ "type": "peer_registered", "peerId": peer_id });
		if let Some(server) = state.current_server.as_ref() {
			registered["serverConnectionId"] = serde_json::Value::String(server.server_connection_id.clone());
		}
		send_line(&line_tx, &registered);
		if let Some(server) = state.current_server.as_ref() {
			write_routed_line(server, &serde_json::json!({ "type": "peer_connected", "peerId": peer_id }));
		}
		Ok(peer_id)
	}

	async fn handle_routed_message(
		self: &Arc<Self>,
		from: &str,
		message_type: &str,
		value: &serde_json::Value,
	) -> Result<(), String> {
		match message_type {
			"send" => {
				let to = string_field(value, "to").filter(|to| !to.is_empty());
				let Some(to) = to else {
					return Err("Coordinator message target must be a string".into());
				};
				let line = encode_control_line(&serde_json::json!({
					"type": "message",
					"from": from,
					"payload": value.get("payload").cloned().unwrap_or(serde_json::Value::Null),
				}))?;
				let state = self.state.lock().await;
				let target = if to == "server" {
					state.current_server.as_ref().map(|server| server.tx.clone())
				} else {
					state.peers.get(&to).cloned()
				};
				drop(state);
				if let Some(target) = target {
					let _ = target.send(line);
				}
				Ok(())
			}
			"broadcast" => {
				if from != "server" {
					return Err("Only the current server may broadcast".into());
				}
				let state = self.state.lock().await;
				let targets: Vec<LineTx> = state.peers.values().cloned().collect();
				drop(state);
				let line = encode_control_line(&serde_json::json!({
					"type": "message",
					"from": from,
					"payload": value.get("payload").cloned().unwrap_or(serde_json::Value::Null),
				}))?;
				for target in targets {
					let _ = target.send(line.clone());
				}
				Ok(())
			}
			other => Err(format!("Unknown coordinator routing message: {other}")),
		}
	}

	async fn accept_public_connection(self: &Arc<Self>, client: UnixStream) {
		let endpoint = {
			let state = self.state.lock().await;
			if state.shutting_down {
				return; // destroy the client
			}
			match &state.current_server {
				Some(server) => server.endpoint.clone(),
				None => return, // destroy the client
			}
		};
		self.cancel_empty_shutdown().await;
		let mut state = self.state.lock().await;
		let id = state.next_connection_id;
		state.next_connection_id += 1;
		let server = Arc::clone(self);
		let handle = tokio::spawn(async move {
			let mut client = client;
			let mut upstream = match connect_endpoint(&endpoint).await {
				Ok(upstream) => upstream,
				Err(_) => {
					server.finalize_public(id).await;
					return;
				}
			};
			let _ = tokio::io::copy_bidirectional(&mut client, &mut upstream).await;
			server.finalize_public(id).await;
		});
		state.public_tasks.insert(id, handle);
	}

	async fn finalize_public(self: &Arc<Self>, id: u64) {
		self.state.lock().await.public_tasks.remove(&id);
		self.check_empty().await;
	}

	async fn close_public_connections(&self) {
		let tasks: Vec<JoinHandle<()>> =
			std::mem::take(&mut self.state.lock().await.public_tasks).into_values().collect();
		for handle in tasks {
			handle.abort();
		}
	}

	async fn shutdown(&self) {
		{
			let mut state = self.state.lock().await;
			if state.shutting_down {
				return;
			}
			state.shutting_down = true;
		}
		self.cancel_empty_shutdown().await;
		self.close_public_connections().await;
		let control_tasks: Vec<JoinHandle<()>> =
			std::mem::take(&mut self.state.lock().await.control_tasks).into_values().collect();
		for handle in control_tasks {
			handle.abort();
		}
	}
}

fn string_field(value: &serde_json::Value, key: &str) -> Option<String> {
	value.get(key).and_then(serde_json::Value::as_str).map(String::from)
}

fn send_line(tx: &LineTx, message: &serde_json::Value) {
	if let Ok(line) = encode_control_line(message) {
		let _ = tx.send(line);
	}
}

fn write_routed_line(target: &ServerPeerEntry, message: &serde_json::Value) {
	write_routed_line_tx(&target.tx, message);
}

fn write_routed_line_tx(target: &LineTx, message: &serde_json::Value) {
	if let Ok(line) = encode_control_line(message) {
		let _ = target.send(line);
	}
}

/// `createConnection(endpoint)`: Unix socket path or `host:port` TCP.
async fn connect_endpoint(endpoint: &str) -> std::io::Result<Box<dyn EndpointStream>> {
	if endpoint.starts_with('/') {
		let stream = UnixStream::connect(endpoint).await?;
		return Ok(Box::new(stream));
	}
	let Some((host, port)) = endpoint.rsplit_once(':') else {
		return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid endpoint"));
	};
	let port = port
		.parse::<u16>()
		.map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
	let stream = tokio::net::TcpStream::connect((host, port)).await?;
	Ok(Box::new(stream))
}

trait EndpointStream: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> EndpointStream for T {}

static RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn restrict_socket(path: &str) -> std::io::Result<()> {
	std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
}

async fn remove_stale_socket(path: &str) -> Result<(), String> {
	match std::fs::symlink_metadata(path) {
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
		Err(error) => return Err(error.to_string()),
		Ok(metadata) => {
			if !metadata.file_type().is_socket() {
				return Err(format!("Coordinator path is not a socket: {path}"));
			}
		}
	}
	let live = match UnixStream::connect(path).await {
		Ok(_) => true,
		Err(error)
			if error.kind() == std::io::ErrorKind::ConnectionRefused
				|| error.kind() == std::io::ErrorKind::NotFound =>
		{
			false
		}
		Err(error) => return Err(error.to_string()),
	};
	if live {
		return Err(format!("Coordinator socket is already active: {path}"));
	}
	cleanup_socket(path).await
}

async fn cleanup_socket(path: &str) -> Result<(), String> {
	match std::fs::remove_file(path) {
		Ok(()) => Ok(()),
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
		Err(error) => Err(error.to_string()),
	}
}

/// Entry point of the coordinator process (`runCoordinatorProcess`). Runs
/// until the socket set goes empty past the grace window or SIGINT arrives,
/// then cleans up its sockets.
pub async fn run_coordinator_process(args: &[String]) -> Result<(), String> {
	if RUNNING.swap(true, Ordering::SeqCst) {
		return Err("Coordinator process is already running".into());
	}
	let result = run_inner(args).await;
	RUNNING.store(false, Ordering::SeqCst);
	result
}

async fn run_inner(args: &[String]) -> Result<(), String> {
	let [requested_public_path, requested_control_path, ..] = args else {
		return Err("Coordinator requires public and control socket paths".into());
	};
	let public_path = requested_public_path.clone();
	let control_path = requested_control_path.clone();
	let (server, mut fatal_rx) = CoordinatorServer::new(public_path.clone(), control_path.clone());
	remove_stale_socket(&control_path).await?;
	remove_stale_socket(&public_path).await?;
	let control_listener = match tokio::net::UnixListener::bind(&control_path) {
		Ok(listener) => listener,
		Err(error) => {
			let _ = cleanup_socket(&control_path).await;
			let _ = cleanup_socket(&public_path).await;
			return Err(error.to_string());
		}
	};
	restrict_socket(&control_path).map_err(|error| error.to_string())?;
	let public_listener = match tokio::net::UnixListener::bind(&public_path) {
		Ok(listener) => listener,
		Err(error) => {
			drop(control_listener);
			let _ = cleanup_socket(&control_path).await;
			let _ = cleanup_socket(&public_path).await;
			return Err(error.to_string());
		}
	};
	restrict_socket(&public_path).map_err(|error| error.to_string())?;

	let accept_server = Arc::clone(&server);
	let control_listener_task = Arc::new(control_listener);
	let control_accept_listener = Arc::clone(&control_listener_task);
	let control_task = tokio::spawn(async move {
		while let Some(socket) = accept_next(&control_accept_listener).await {
			accept_server.accept_control_connection(socket).await;
		}
	});
	let accept_server = Arc::clone(&server);
	let public_listener_task = Arc::new(public_listener);
	let public_accept_listener = Arc::clone(&public_listener_task);
	let public_task = tokio::spawn(async move {
		while let Some(socket) = accept_next(&public_accept_listener).await {
			accept_server.accept_public_connection(socket).await;
		}
	});
	server.schedule_empty_shutdown(EMPTY_STARTUP_GRACE_MS).await;

	let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();
	let outcome = loop {
		tokio::select! {
			fatal = fatal_rx.recv() => match fatal {
				Some(Fatal::Empty) | None => break Ok(()),
				Some(Fatal::Crash(message)) => break Err(message),
			},
			_ = tokio::signal::ctrl_c() => break Ok(()),
			_ = async {
				match sigterm.as_mut() {
					Some(signal) => signal.recv().await,
					None => std::future::pending().await,
				}
			} => break Ok(()),
		}
	};
	drop(control_listener_task);
	drop(public_listener_task);
	control_task.abort();
	public_task.abort();
	server.shutdown().await;
	let _ = cleanup_socket(&public_path).await;
	let _ = cleanup_socket(&control_path).await;
	outcome
}

async fn accept_next(listener: &tokio::net::UnixListener) -> Option<UnixStream> {
	listener.accept().await.ok().map(|(socket, _)| socket)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn coordinator_message_wire_round_trip() {
		let message = CoordinatorMessage::ServerRegistered {
			server_connection_id: "srv-1".into(),
			peers: vec!["peer-a".into()],
		};
		let json = serde_json::to_value(&message).unwrap();
		assert_eq!(
			json,
			serde_json::json!({
				"type": "server_registered",
				"serverConnectionId": "srv-1",
				"peers": ["peer-a"],
			})
		);
		assert_eq!(serde_json::from_value::<CoordinatorMessage>(json).unwrap(), message);

		let replaced = serde_json::to_value(&CoordinatorMessage::ServerReplaced).unwrap();
		assert_eq!(replaced, serde_json::json!({ "type": "server_replaced" }));

		let routed = serde_json::from_value::<CoordinatorMessage>(serde_json::json!({
			"type": "message",
			"from": "peer-a",
			"payload": { "arbitrary": [1, 2, 3] },
		}))
		.unwrap();
		assert!(matches!(routed, CoordinatorMessage::Message { from, .. } if from == "peer-a"));

		// Schema check parity: an unknown type is rejected.
		assert!(serde_json::from_value::<CoordinatorMessage>(serde_json::json!({ "type": "nope" })).is_err());
	}

	#[tokio::test]
	async fn ensure_coordinator_fails_for_dead_child() {
		let _guard = crate::experimental::TEST_GUARD.lock();
		let dir = std::env::temp_dir().join(format!("coordinator-test-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let public = dir.join("missing-public.sock");
		let control = dir.join("missing-control.sock");
		let _ = std::fs::remove_file(&public);
		let _ = std::fs::remove_file(&control);
		// The spawned "coordinator" is this test binary without a coordinator
		// main, so it exits during startup like upstream's dead child.
		let error = ensure_coordinator(public.to_str().unwrap(), control.to_str().unwrap())
			.await
			.unwrap_err();
		assert!(error == "Coordinator exited during startup" || error == "Timed out waiting for coordinator startup");
		let _ = std::fs::remove_dir_all(&dir);
	}

	#[tokio::test]
	async fn coordinator_routes_messages_and_replaces_nothing_crashes_on_bad_broadcast() {
		let _guard = crate::experimental::TEST_GUARD.lock();
		let dir = std::env::temp_dir().join(format!("coordinator-live-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let public_path = dir.join("public.sock");
		let control_path = dir.join("control.sock");
		let _ = std::fs::remove_file(&public_path);
		let _ = std::fs::remove_file(&control_path);
		let public = public_path.to_str().unwrap().to_string();
		let control = control_path.to_str().unwrap().to_string();

		let args: Vec<String> = vec![public.clone(), control.clone()];
		let run = {
			let args = args.clone();
			tokio::spawn(async move { run_coordinator_process(&args).await })
		};

		let mut server_stream = None;
		for _ in 0..500 {
			if let Ok(stream) = UnixStream::connect(&control).await {
				server_stream = Some(stream);
				break;
			}
			tokio::time::sleep(Duration::from_millis(10)).await;
		}
		let server_stream = server_stream.expect("coordinator did not start");

		let connection = CoordinatorConnection::new(CoordinatorConnectionOptions {
			control_path: control.clone(),
			endpoint: public.clone(),
			server_connection_id: Some("srv-1".into()),
		});
		let events: Arc<std::sync::Mutex<Vec<CoordinatorConnectionEvent>>> = Default::default();
		let _guard = {
			let events = Arc::clone(&events);
			connection.on_event(move |event| events.lock().unwrap().push(event.clone()))
		};
		connection.connect().await.expect("server registration");
		assert!(connection.peer_ids().is_empty());

		// Register a raw peer over its own control connection.
		let (mut peer_reader, mut peer_writer) = server_stream.into_split();
		peer_writer
			.write_all(
				encode_control_line(&serde_json::json!({
					"type": "register_peer",
					"protocol": COORDINATOR_PROTOCOL_VERSION,
					"peerId": "peer-a",
				}))
				.unwrap()
				.as_bytes(),
			)
			.await
			.unwrap();
		let registration = read_line(&mut peer_reader).await;
		assert_eq!(registration["type"], "peer_registered");
		assert_eq!(registration["peerId"], "peer-a");
		assert_eq!(registration["serverConnectionId"], "srv-1");
		let deadline = Instant::now() + Duration::from_secs(5);
		while !connection.peer_ids().contains("peer-a") {
			assert!(Instant::now() < deadline, "timed out waiting for peer_connected");
			tokio::time::sleep(Duration::from_millis(10)).await;
		}
		assert_eq!(connection.peer_ids(), BTreeSet::from(["peer-a".to_string()]));

		// Server → peer send.
		connection
			.send("peer-a", serde_json::json!({ "hello": true }))
			.await
			.unwrap();
		let routed = read_line(&mut peer_reader).await;
		assert_eq!(routed["type"], "message");
		assert_eq!(routed["from"], "server");
		assert_eq!(routed["payload"], serde_json::json!({ "hello": true }));

		// Peer → server send.
		peer_writer
			.write_all(
				encode_control_line(&serde_json::json!({
					"type": "send",
					"to": "server",
					"payload": 42,
				}))
				.unwrap()
				.as_bytes(),
			)
			.await
			.unwrap();
		let deadline = Instant::now() + Duration::from_secs(5);
		loop {
			assert!(Instant::now() < deadline, "timed out waiting for routed message");
			if events
				.lock()
				.unwrap()
				.iter()
				.any(|event| matches!(event, CoordinatorConnectionEvent::Message { payload, .. } if *payload == serde_json::json!(42)))
			{
				break;
			}
			tokio::time::sleep(Duration::from_millis(10)).await;
		}

		// A peer broadcast is rejected (upstream: uncaught exception crash).
		peer_writer
			.write_all(
				encode_control_line(&serde_json::json!({ "type": "broadcast", "payload": null }))
					.unwrap()
					.as_bytes(),
			)
			.await
			.unwrap();
		let outcome = tokio::time::timeout(Duration::from_secs(5), run)
			.await
			.expect("coordinator did not crash")
			.expect("task joined");
		assert_eq!(outcome.unwrap_err(), "Only the current server may broadcast");

		connection.close();
		let _ = std::fs::remove_dir_all(&dir);
	}

	async fn read_line(reader: &mut tokio::net::unix::OwnedReadHalf) -> serde_json::Value {
		let mut buffer = Vec::new();
		let mut byte = [0u8; 1];
		loop {
			let read = reader.read(&mut byte).await.unwrap();
			assert!(read > 0, "connection closed before a full line");
			if byte[0] == b'\n' {
				break;
			}
			buffer.push(byte[0]);
		}
		serde_json::from_slice(&buffer).unwrap()
	}
}
