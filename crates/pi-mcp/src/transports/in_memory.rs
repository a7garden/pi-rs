//! Port of `packages/mcp/src/transports/in-memory.ts`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;

use crate::protocol::jsonrpc::{JsonRpcMessage, McpError};
use crate::transports::transport::{CloseListener, ErrorListener, McpTransport, MessageListener, TransportEvents, Unsubscribe};

/// Loopback transport for tests: `send` delivers a deep copy to the peer on the
/// next tick (upstream `structuredClone` + `queueMicrotask`).
pub struct InMemoryTransport {
	events: TransportEvents,
	state: Mutex<PeerState>,
	closed: AtomicBool,
	started: AtomicBool,
}

struct PeerState {
	peer: Option<std::sync::Weak<InMemoryTransport>>,
}

impl Default for InMemoryTransport {
	fn default() -> Self {
		Self::new()
	}
}

impl InMemoryTransport {
	pub fn new() -> InMemoryTransport {
		InMemoryTransport {
			events: TransportEvents::new(),
			state: Mutex::new(PeerState { peer: None }),
			closed: AtomicBool::new(false),
			started: AtomicBool::new(false),
		}
	}

	pub fn connect_peer(self: &std::sync::Arc<Self>, peer: std::sync::Arc<Self>) -> Result<(), McpError> {
		let mut state = self.lock();
		if state.peer.is_some() {
			return Err(McpError::other("In-memory MCP transport already has a peer"));
		}
		state.peer = Some(std::sync::Arc::downgrade(&peer));
		Ok(())
	}

	fn lock(&self) -> std::sync::MutexGuard<'_, PeerState> {
		self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
	}

	fn closed(&self) -> bool {
		self.closed.load(Ordering::SeqCst)
	}

	fn started(&self) -> bool {
		self.started.load(Ordering::SeqCst)
	}

	/// Exposed so tests can simulate transport-level failures.
	pub fn emit_error(&self, error: McpError) {
		self.events.emit_error(error);
	}

	fn deliver(&self, message: JsonRpcMessage) {
		if self.closed() {
			return;
		}
		self.events.emit_message(message);
	}
}

#[async_trait]
impl McpTransport for InMemoryTransport {
	async fn start(&self) -> Result<(), McpError> {
		if self.closed() {
			return Err(McpError::connection_closed("MCP connection closed"));
		}
		self.started.store(true, Ordering::SeqCst);
		Ok(())
	}

	async fn send(&self, message: JsonRpcMessage) -> Result<(), McpError> {
		if !self.started() || self.closed() {
			return Err(McpError::connection_closed("MCP connection closed"));
		}
		let peer = self.lock().peer.clone().and_then(|peer| peer.upgrade());
		match peer {
			Some(peer) if peer.started() && !peer.closed() => {
				// Upstream: structuredClone + queueMicrotask.
				tokio::spawn(async move { peer.deliver(message) });
				Ok(())
			}
			_ => Err(McpError::connection_closed("In-memory MCP peer is not connected")),
		}
	}

	async fn close(&self) -> Result<(), McpError> {
		if self.closed.swap(true, Ordering::SeqCst) {
			return Ok(());
		}
		self.events.emit_close();
		let peer = self.lock().peer.clone().and_then(|peer| peer.upgrade());
		if let Some(peer) = peer {
			peer.close().await?;
		}
		Ok(())
	}

	fn on_message(&self, listener: MessageListener) -> Unsubscribe {
		self.events.on_message(listener)
	}

	fn on_error(&self, listener: ErrorListener) -> Unsubscribe {
		self.events.on_error(listener)
	}

	fn on_close(&self, listener: CloseListener) -> Unsubscribe {
		self.events.on_close(listener)
	}
}

pub struct InMemoryTransportPair {
	pub client: std::sync::Arc<InMemoryTransport>,
	pub server: std::sync::Arc<InMemoryTransport>,
}

pub fn create_in_memory_transport_pair() -> InMemoryTransportPair {
	let client = std::sync::Arc::new(InMemoryTransport::new());
	let server = std::sync::Arc::new(InMemoryTransport::new());
	client.connect_peer(server.clone()).expect("fresh pair");
	server.connect_peer(client.clone()).expect("fresh pair");
	InMemoryTransportPair { client, server }
}
