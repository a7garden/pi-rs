//! Port of `packages/server/src/transports/unix/types.ts`.

use std::sync::Arc;

use crate::types::ServerOptions;

/// Upstream `UnixListenerOptions`.
#[derive(Clone)]
pub struct UnixListenerOptions {
	pub path: String,
	/// Socket filesystem permissions. Defaults to owner read/write only (0o600).
	pub mode: Option<u32>,
	/// Maximum framed bytes queued per connection before a slow peer is
	/// disconnected.
	pub max_pending_bytes: Option<usize>,
	pub graceful_close_timeout_ms: Option<u64>,
	/// Used to derive and validate maxPendingBytes. Must match the server when
	/// customized.
	pub max_frame_length: Option<usize>,
	pub on_error: Option<crate::types::ErrorCallback>,
}

/// Upstream `UnixServerOptions extends Omit<ServerOptions, "listeners">`.
#[derive(Clone)]
pub struct UnixServerOptions {
	pub server_id: String,
	pub max_frame_length: Option<usize>,
	pub handshake_timeout_ms: Option<u64>,
	pub on_connection_count_changed: Option<crate::types::ConnectionCountCallback>,
	pub on_error: Option<crate::types::ErrorCallback>,
	// UnixListenerOptions fields:
	pub path: String,
	pub mode: Option<u32>,
	pub max_pending_bytes: Option<usize>,
	pub graceful_close_timeout_ms: Option<u64>,
}

impl UnixServerOptions {
	pub(crate) fn listener_options(&self) -> UnixListenerOptions {
		UnixListenerOptions {
			path: self.path.clone(),
			mode: self.mode,
			max_pending_bytes: self.max_pending_bytes,
			graceful_close_timeout_ms: self.graceful_close_timeout_ms,
			max_frame_length: self.max_frame_length,
			on_error: self.on_error.clone(),
		}
	}

	pub(crate) fn server_options(&self, listeners: Vec<Arc<dyn crate::listener::ServerListener>>) -> ServerOptions {
		ServerOptions {
			listeners,
			server_id: self.server_id.clone(),
			max_frame_length: self.max_frame_length,
			handshake_timeout_ms: self.handshake_timeout_ms,
			on_connection_count_changed: self.on_connection_count_changed.clone(),
			on_error: self.on_error.clone(),
		}
	}
}

