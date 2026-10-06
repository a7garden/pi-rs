//! Port of `packages/server/test/` (upstream vitest suites).
//!
//! Module layout mirrors the upstream test files 1:1; vitest `expect.poll`
//! becomes a bounded poll helper, and the in-memory `ByteConnection` /
//! `WireChannel` pair from the upstream tests becomes the loopback in
//! [`loopback`].

pub mod conformance_tests;
pub mod listener_tests;
pub mod protocol_tests;
pub mod server_tests;
pub mod unix_connection_tests;
pub mod unix_tests;

pub(crate) mod loopback {
	//! In-memory client/server wiring (upstream `connect()` in
	//! protocol.test.ts / conformance.test.ts).

	use std::sync::Arc;
	use std::sync::atomic::{AtomicBool, Ordering};

	use async_trait::async_trait;
	use crate::connection::{ByteConnection, ByteConnectionHandler};
	use crate::server::Server;
	use crate::testing::client::{ProtocolTestClient, WireChannel};
	use crate::types::AnyError;

	pub(crate) struct Loopback {
		pub client: ProtocolTestClient,
	}

	struct LoopbackConnection {
		client: Arc<std::sync::OnceLock<ProtocolTestClient>>,
		mark_closed_on_terminate: bool,
		closed: AtomicBool,
	}

	#[async_trait]
	impl ByteConnection for LoopbackConnection {
		fn closed(&self) -> bool {
			self.closed.load(Ordering::SeqCst)
		}

		async fn send(&self, chunk: Vec<u8>) -> Result<(), AnyError> {
			if let Some(client) = self.client.get() {
				client.receive(&chunk);
			}
			Ok(())
		}

		async fn close(&self, final_chunk: Option<Vec<u8>>) -> Result<(), AnyError> {
			if let (Some(client), Some(final_chunk)) = (self.client.get(), final_chunk) {
				client.receive(&final_chunk);
			}
			self.closed.store(true, Ordering::SeqCst);
			if self.mark_closed_on_terminate {
				if let Some(client) = self.client.get() {
					client.mark_closed();
				}
			}
			Ok(())
		}
	}

	struct LoopbackChannel {
		handler: Arc<std::sync::OnceLock<Arc<dyn ByteConnectionHandler>>>,
		client: Arc<std::sync::OnceLock<ProtocolTestClient>>,
	}

	#[async_trait]
	impl WireChannel for LoopbackChannel {
		async fn send(&self, chunk: Vec<u8>) -> Result<(), AnyError> {
			if let Some(handler) = self.handler.get() {
				handler.on_data(&chunk);
			}
			Ok(())
		}

		async fn send_fragmented(&self, chunk: Vec<u8>, split_at: usize) -> Result<(), AnyError> {
			self.send(chunk[..split_at].to_vec()).await?;
			self.send(chunk[split_at..].to_vec()).await
		}

		async fn close(&self) -> Result<(), AnyError> {
			if let Some(handler) = self.handler.get() {
				handler.on_close();
			}
			if let Some(client) = self.client.get() {
				client.mark_closed();
			}
			Ok(())
		}
	}

	/// Wire one server connection to a [`ProtocolTestClient`].
	pub(crate) fn connect(server: &Server) -> Loopback {
		let handler: Arc<std::sync::OnceLock<Arc<dyn ByteConnectionHandler>>> = Arc::default();
		let client_slot: Arc<std::sync::OnceLock<ProtocolTestClient>> = Arc::default();
		let connection = Arc::new(LoopbackConnection {
			client: Arc::clone(&client_slot),
			mark_closed_on_terminate: true,
			closed: AtomicBool::new(false),
		});
		let channel = Arc::new(LoopbackChannel { handler: Arc::clone(&handler), client: Arc::clone(&client_slot) });
		let client = ProtocolTestClient::new(channel);
		assert!(client_slot.set(client.clone()).is_ok(), "single connect per loopback");
		let accept = server.acceptor();
		assert!(handler.set(accept(connection)).is_ok(), "single accept per loopback");
		Loopback { client }
	}
}

/// Bounded condition polling (vitest `expect.poll`).
pub(crate) async fn poll_until(mut condition: impl FnMut() -> bool) {
	for _ in 0..800 {
		if condition() {
			return;
		}
		tokio::time::sleep(std::time::Duration::from_millis(5)).await;
	}
	panic!("poll condition was not met");
}

/// Message-shape assertion helpers over the protocol `JsonValue` envelopes.
pub(crate) mod msg {
	use pi_chord::JsonValue;
	pub(crate) fn kind(message: &JsonValue) -> &str {
		message.as_object().and_then(|o| o.get("type")).and_then(JsonValue::as_str).unwrap_or_default()
	}

	pub(crate) fn ok(message: &JsonValue) -> bool {
		message.as_object().and_then(|o| o.get("ok")).and_then(JsonValue::as_bool).unwrap_or(false)
	}

	pub(crate) fn error_code(message: &JsonValue) -> String {
		message
			.as_object()
			.and_then(|o| o.get("error"))
			.and_then(|e| e.as_object())
			.and_then(|e| e.get("code"))
			.and_then(JsonValue::as_str)
			.unwrap_or_default()
			.to_string()
	}

	pub(crate) fn error_message(message: &JsonValue) -> String {
		message
			.as_object()
			.and_then(|o| o.get("error"))
			.and_then(|e| e.as_object())
			.and_then(|e| e.get("message"))
			.and_then(JsonValue::as_str)
			.unwrap_or_default()
			.to_string()
	}

	pub(crate) fn id(message: &JsonValue) -> String {
		message.as_object().and_then(|o| o.get("id")).and_then(JsonValue::as_str).unwrap_or_default().to_string()
	}

	pub(crate) fn attachment(message: &JsonValue) -> Option<(String, String)> {
		message
			.as_object()
			.and_then(|o| o.get("attachment"))
			.and_then(|a| a.as_object())
			.map(|a| {
				(
					a.get("sessionId").and_then(JsonValue::as_str).unwrap_or_default().to_string(),
					a.get("attachmentId").and_then(JsonValue::as_str).unwrap_or_default().to_string(),
				)
			})
	}

	pub(crate) fn attachment_is_null(message: &JsonValue) -> bool {
		message.as_object().and_then(|o| o.get("attachment")).map(JsonValue::is_null).unwrap_or(false)
	}
}
