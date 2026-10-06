//! Port of `packages/server/src/connection.ts`.
//!
//! Upstream `ConnectionState.handshakeTimeout: NodeJS.Timeout` becomes the
//! spawned timeout task handle tracked alongside the state (see
//! [`crate::server`]); `handshake?: Promise<void>` becomes a `JoinHandle`.

use std::collections::HashMap;
use std::sync::Arc;

use pi_chord::{AbortController, JsonValue, ServiceStateEncoder};
use pi_protocol::ClientMessageDecoder;

use crate::types::{AnyError, RoutedServerServiceAttachment};

/// An established, authorized ordered byte connection
/// (upstream `ByteConnection`).
#[async_trait::async_trait]
pub trait ByteConnection: Send + Sync {
	fn closed(&self) -> bool;
	async fn send(&self, chunk: Vec<u8>) -> Result<(), AnyError>;
	async fn close(&self, final_chunk: Option<Vec<u8>>) -> Result<(), AnyError>;
}

/// Upstream `ByteConnectionHandler` — synchronous callbacks; implementations
/// spawn async work internally (upstream `void`-prefixed promises).
pub trait ByteConnectionHandler: Send + Sync {
	fn on_data(&self, chunk: &[u8]);
	fn on_close(&self);
	fn on_error(&self, error: AnyError);
}

/// Upstream `ByteConnectionAcceptor`.
pub type ByteConnectionAcceptor = Arc<dyn Fn(Arc<dyn ByteConnection>) -> Arc<dyn ByteConnectionHandler> + Send + Sync>;

/// Upstream `ConnectionStage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionStage {
	AwaitingHello,
	Handshaking,
	Ready,
	Closing,
	Closed,
}

/// Upstream `activeRequests` entries.
pub struct ActiveRequest {
	pub controller: AbortController,
	pub target: JsonValue,
}

/// Upstream `ConnectionState`.
pub struct ConnectionState {
	pub connection: Arc<dyn ByteConnection>,
	pub decoder: ClientMessageDecoder,
	pub service_state_encoders: parking_lot::RwLock<HashMap<String, ServiceStateEncoder>>,
	pub stage: ConnectionStage,
	pub disconnected: bool,
	/// Set when the handshake future settles (replaces `clearTimeout` racing).
	pub handshake_settled: bool,
	/// Upstream `handshake?: Promise<void>` — completion signal of the
	/// spawned handshake task.
	pub handshake_done: Option<tokio::sync::watch::Receiver<bool>>,
	pub server_services: Option<Arc<dyn RoutedServerServiceAttachment>>,
	pub active_requests: HashMap<String, ActiveRequest>,
}

impl ConnectionState {
	pub fn new(connection: Arc<dyn ByteConnection>, max_frame_length: usize) -> ConnectionState {
		ConnectionState {
			connection,
			decoder: ClientMessageDecoder::new(Some(&pi_protocol::FrameDecoderOptions {
				max_frame_length: Some(max_frame_length),
			}))
			.expect("max frame length validated at server construction"),
			service_state_encoders: parking_lot::RwLock::new(HashMap::new()),
			stage: ConnectionStage::AwaitingHello,
			disconnected: false,
			handshake_settled: false,
			handshake_done: None,
			server_services: None,
			active_requests: HashMap::new(),
		}
	}
}

pub fn is_terminal_connection(state: &ConnectionState) -> bool {
	state.disconnected || matches!(state.stage, ConnectionStage::Closing | ConnectionStage::Closed)
}
