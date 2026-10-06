//! Port of `packages/server/test/unix-connection.test.ts`.
//!
//! Upstream drives a `ControlledSocket` Node fake whose `write` callback
//! completes manually; the Rust port drives the same scenario through the
//! crate-internal [`RawSocket`] seam.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use pi_chord::JsonValue;
use pi_protocol::{ServerMessageDecoder, encode_server_message};

use crate::connection::ByteConnection;
use crate::transports::unix::socket::{RawSocket, UnixByteConnection};

struct ControlledSocket {
	destroyed: AtomicBool,
	writable: AtomicBool,
	writable_length: AtomicUsize,
	ended: AtomicBool,
	final_chunk: parking_lot::Mutex<Option<Vec<u8>>>,
	write_callback: parking_lot::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

impl ControlledSocket {
	fn new() -> Arc<ControlledSocket> {
		Arc::new(ControlledSocket {
			destroyed: AtomicBool::new(false),
			writable: AtomicBool::new(true),
			writable_length: AtomicUsize::new(0),
			ended: AtomicBool::new(false),
			final_chunk: parking_lot::Mutex::new(None),
			write_callback: parking_lot::Mutex::new(None),
		})
	}

	fn complete_write(&self) {
		let callback = self.write_callback.lock().take().expect("No pending write");
		self.writable_length.store(0, Ordering::SeqCst);
		callback.send(()).expect("write waiter alive");
	}
}

#[async_trait]
impl RawSocket for ControlledSocket {
	fn destroyed(&self) -> bool {
		self.destroyed.load(Ordering::SeqCst)
	}

	fn writable(&self) -> bool {
		self.writable.load(Ordering::SeqCst)
	}

	async fn write(&self, _chunk: &[u8]) -> std::io::Result<()> {
		self.writable_length.store(1, Ordering::SeqCst);
		let (tx, rx) = tokio::sync::oneshot::channel();
		*self.write_callback.lock() = Some(tx);
		// Manual completion mirrors the upstream `writeCallback` fake.
		let _ = rx.await;
		Ok(())
	}

	async fn end(&self, final_chunk: Option<&[u8]>) -> std::io::Result<()> {
		self.ended.store(true, Ordering::SeqCst);
		if let Some(chunk) = final_chunk {
			*self.final_chunk.lock() = Some(chunk.to_vec());
		}
		Ok(())
	}

	fn destroy(&self) {
		if self.destroyed.swap(true, Ordering::SeqCst) {
			return;
		}
		self.writable.store(false, Ordering::SeqCst);
	}
}

#[tokio::test]
async fn queues_a_final_protocol_error_behind_pending_output_before_closing() {
	let socket = ControlledSocket::new();
	let connection = Arc::new(UnixByteConnection::new(socket.clone(), 1_000, 64 * 1024));
	let connection_for_send = Arc::clone(&connection);
	let pending_write = tokio::spawn(async move { connection_for_send.send(vec![1, 2, 3]).await });
	poll_writable_length(&socket, 1).await;
	let final_message: JsonValue = serde_json::json!({
		"type": "hello_error",
		"error": { "code": "invalid_request", "message": "Protocol violation" },
	});
	let final_frame = encode_server_message(&final_message, None).expect("frame");
	let connection_for_close = Arc::clone(&connection);
	let closing = tokio::spawn(async move { connection_for_close.close(Some(final_frame)).await });

	// The close must not end or destroy the socket while output is pending.
	assert!(!socket.ended.load(Ordering::SeqCst));
	assert!(!socket.destroyed.load(Ordering::SeqCst));

	socket.complete_write();
	let _ = pending_write.await.expect("pending write task");
	poll_until(|| socket.ended.load(Ordering::SeqCst)).await;
	let final_chunk = socket.final_chunk.lock().take().expect("final chunk");
	let messages = ServerMessageDecoder::new(None).expect("decoder").push(&final_chunk).expect("frames");
	assert_eq!(messages, vec![final_message]);

	connection.mark_closed();
	closing.await.expect("closing task").expect("close");
}

async fn poll_writable_length(socket: &ControlledSocket, expected: usize) {
	for _ in 0..800 {
		if socket.writable_length.load(Ordering::SeqCst) == expected {
			return;
		}
		tokio::time::sleep(std::time::Duration::from_millis(5)).await;
	}
	panic!("writableLength never reached {expected}");
}

async fn poll_until(condition: impl FnMut() -> bool) {
	crate::tests::poll_until(condition).await
}
