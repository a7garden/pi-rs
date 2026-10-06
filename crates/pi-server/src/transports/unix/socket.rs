//! Port of `packages/server/src/transports/unix/listener.ts`
//! (`UnixByteConnection`).
//!
//! The Node `Socket` event surface collapses into an internal [`RawSocket`]
//! seam: production uses tokio halves; the connection-level unit test drives a
//! controlled fake (upstream `ControlledSocket`), preserving the deterministic
//! "queue the final protocol error behind pending output" behavior.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::net::unix::OwnedWriteHalf;

use crate::connection::ByteConnection;
use crate::types::AnyError;

/// Internal `Socket` seam (see module docs).
#[async_trait::async_trait]
pub(crate) trait RawSocket: Send + Sync {
	fn destroyed(&self) -> bool;
	fn writable(&self) -> bool;
	async fn write(&self, chunk: &[u8]) -> std::io::Result<()>;
	/// Graceful FIN after any final bytes (upstream `socket.end(final?)`).
	async fn end(&self, final_chunk: Option<&[u8]>) -> std::io::Result<()>;
	fn destroy(&self);
}

/// Production [`RawSocket`] over a tokio `OwnedWriteHalf`.
pub(crate) struct TokioSocket {
	write_half: tokio::sync::Mutex<Option<OwnedWriteHalf>>,
	destroyed: AtomicBool,
}

impl TokioSocket {
	pub(crate) fn new(write_half: OwnedWriteHalf) -> Arc<TokioSocket> {
		Arc::new(TokioSocket {
			write_half: tokio::sync::Mutex::new(Some(write_half)),
			destroyed: AtomicBool::new(false),
		})
	}
}

#[async_trait::async_trait]
impl RawSocket for TokioSocket {
	fn destroyed(&self) -> bool {
		self.destroyed.load(Ordering::SeqCst)
	}

	fn writable(&self) -> bool {
		!self.destroyed.load(Ordering::SeqCst) && self.write_half.try_lock().map(|guard| guard.is_some()).unwrap_or(true)
	}

	async fn write(&self, chunk: &[u8]) -> std::io::Result<()> {
		let mut guard = self.write_half.lock().await;
		let Some(half) = guard.as_mut() else {
			return Err(std::io::Error::new(std::io::ErrorKind::NotConnected, "Unix connection is closed"));
		};
		half.write_all(chunk).await
	}

	async fn end(&self, final_chunk: Option<&[u8]>) -> std::io::Result<()> {
		let mut guard = self.write_half.lock().await;
		let Some(half) = guard.as_mut() else {
			return Err(std::io::Error::new(std::io::ErrorKind::NotConnected, "Unix connection is closed"));
		};
		if let Some(chunk) = final_chunk {
			half.write_all(chunk).await?;
		}
		half.shutdown().await
	}

	fn destroy(&self) {
		if self.destroyed.swap(true, Ordering::SeqCst) {
			return;
		}
		if let Ok(mut guard) = self.write_half.try_lock() {
			// Dropping the half closes the descriptor.
			*guard = None;
		}
	}
}

/// Upstream `UnixByteConnection` (exported for transport-level verification).
pub struct UnixByteConnection {
	socket: Arc<dyn RawSocket>,
	graceful_close_timeout: Duration,
	max_pending_bytes: usize,
	pending_bytes: AtomicUsize,
	closed_value: AtomicBool,
	closing: AtomicBool,
	close_started: AtomicBool,
	write_lock: Arc<tokio::sync::Mutex<()>>,
	closed_notify: tokio::sync::Notify,
}

impl UnixByteConnection {
	pub(crate) fn new(socket: Arc<dyn RawSocket>, graceful_close_timeout_ms: u64, max_pending_bytes: usize) -> UnixByteConnection {
		UnixByteConnection {
			socket,
			graceful_close_timeout: Duration::from_millis(graceful_close_timeout_ms),
			max_pending_bytes,
			pending_bytes: AtomicUsize::new(0),
			closed_value: AtomicBool::new(false),
			closing: AtomicBool::new(false),
			close_started: AtomicBool::new(false),
			write_lock: Arc::new(tokio::sync::Mutex::new(())),
			closed_notify: tokio::sync::Notify::new(),
		}
	}

	pub fn mark_closed(&self) {
		if self.closed_value.swap(true, Ordering::SeqCst) {
			return;
		}
		self.closing.store(true, Ordering::SeqCst);
		self.closed_notify.notify_waiters();
	}

	async fn write(&self, chunk: &[u8]) -> Result<(), AnyError> {
		if self.closed_value.load(Ordering::SeqCst)
			|| self.closing.load(Ordering::SeqCst)
			|| !self.socket.writable()
		{
			return Err(connection_closed_error());
		}
		self.socket.write(chunk).await.map_err(crate::types::io_error)
	}
}

fn connection_closed_error() -> AnyError {
	Arc::new(pi_chord::ChordError { message: "Unix connection is closed".to_string() })
}

#[async_trait::async_trait]
impl ByteConnection for UnixByteConnection {
	fn closed(&self) -> bool {
		self.closed_value.load(Ordering::SeqCst)
	}

	async fn send(&self, chunk: Vec<u8>) -> Result<(), AnyError> {
		if self.closed_value.load(Ordering::SeqCst) || self.closing.load(Ordering::SeqCst) {
			return Err(connection_closed_error());
		}
		let byte_length = chunk.len();
		// Reserve pending capacity (upstream maxPendingBytes accounting).
		loop {
			let current = self.pending_bytes.load(Ordering::SeqCst);
			if current + byte_length > self.max_pending_bytes {
				return Err(Arc::new(pi_chord::ChordError {
					message: "Unix connection exceeded its pending byte limit".to_string(),
				}));
			}
			if self
				.pending_bytes
				.compare_exchange(current, current + byte_length, Ordering::SeqCst, Ordering::SeqCst)
				.is_ok()
			{
				break;
			}
		}
		let result = async {
			let _guard = self.write_lock.lock().await;
			self.write(&chunk).await
		}
		.await;
		self.pending_bytes.fetch_sub(byte_length, Ordering::SeqCst);
		result
	}

	async fn close(&self, final_chunk: Option<Vec<u8>>) -> Result<(), AnyError> {
		if self.closed_value.load(Ordering::SeqCst) || self.socket.destroyed() {
			self.mark_closed();
			return Ok(());
		}
		if self.close_started.swap(true, Ordering::SeqCst) {
			// A second close awaits the same completion (upstream
			// `closePromise`).
			self.closed_notify.notified().await;
			return Ok(());
		}
		self.closing.store(true, Ordering::SeqCst);
		// Flush pending output, then finish gracefully with the final chunk.
		let flush_socket = Arc::clone(&self.socket);
		let write_lock = Arc::clone(&self.write_lock);
		tokio::spawn(async move {
			let _guard = write_lock.lock().await;
			if flush_socket.destroyed() {
				return;
			}
			let _ = flush_socket.end(final_chunk.as_deref()).await;
		});
		// Resolve on external close (peer EOF/destroy) or after the graceful
		// timeout destroys the socket.
		let notified = self.closed_notify.notified();
		tokio::select! {
			() = notified => {},
			() = tokio::time::sleep(self.graceful_close_timeout) => {
				self.socket.destroy();
			}
		}
		self.mark_closed();
		Ok(())
	}
}
