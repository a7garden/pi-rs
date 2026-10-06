//! Port of `packages/client/test/support.ts` (`MemoryByteServer`).
//!
//! In-memory byte transport that answers handshakes like the upstream test
//! double. Test-only (`cfg(test)`); shared by the client module tests.

#![cfg(test)]

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use pi_chord::json::JsonValue;
use pi_chord::services::errors::AnyError;
use pi_protocol::{encode_server_message, ClientMessageDecoder};
use parking_lot::Mutex;

use crate::transport::{ByteTransport, ByteTransportHandlers};

pub(crate) const DEFAULT_SERVER_ID: &str = "00000000-0000-4000-8000-000000000001";

pub(crate) struct MemoryByteServer {
	pub(crate) server_id: String,
	messages: Mutex<Vec<JsonValue>>,
	client_close_count: AtomicU32,
	handlers: Mutex<Option<ByteTransportHandlers>>,
	decoder: Mutex<ClientMessageDecoder>,
	notify: tokio::sync::Notify,
}

impl MemoryByteServer {
	pub(crate) fn new() -> Arc<MemoryByteServer> {
		Self::with_server_id(DEFAULT_SERVER_ID)
	}

	pub(crate) fn with_server_id(server_id: &str) -> Arc<MemoryByteServer> {
		Arc::new(MemoryByteServer {
			server_id: server_id.to_string(),
			messages: Mutex::new(Vec::new()),
			client_close_count: AtomicU32::new(0),
			handlers: Mutex::new(None),
			decoder: Mutex::new(ClientMessageDecoder::new(None).expect("default frame options are valid")),
			notify: tokio::sync::Notify::new(),
		})
	}

	pub(crate) fn messages(&self) -> Vec<JsonValue> {
		self.messages.lock().clone()
	}

	pub(crate) fn client_close_count(&self) -> u32 {
		self.client_close_count.load(Ordering::SeqCst)
	}

	pub(crate) fn connect(self: &Arc<Self>, handlers: ByteTransportHandlers) -> Box<dyn ByteTransport> {
		*self.handlers.lock() = Some(handlers);
		*self.decoder.lock() = ClientMessageDecoder::new(None).expect("default frame options are valid");
		Box::new(MemoryTransport { server: Arc::clone(self) })
	}

	pub(crate) async fn wait_for_messages(&self, count: usize) {
		loop {
			if self.messages.lock().len() >= count {
				return;
			}
			let notified = self.notify.notified();
			if self.messages.lock().len() >= count {
				return;
			}
			notified.await;
		}
	}

	pub(crate) fn send(&self, message: &JsonValue) {
		let handlers = self.handlers.lock().clone();
		let Some(handlers) = handlers else { panic!("No client connection") };
		let frame = encode_server_message(message, None).expect("test message encodes");
		(handlers.on_data)(frame);
	}

	pub(crate) fn send_raw(&self, chunk: Vec<u8>) {
		let handlers = self.handlers.lock().clone();
		let Some(handlers) = handlers else { panic!("No client connection") };
		(handlers.on_data)(chunk);
	}

	pub(crate) fn disconnect(&self) {
		let handlers = self.handlers.lock().take();
		if let Some(handlers) = handlers {
			(handlers.on_close)();
		}
	}

	pub(crate) fn error(&self, transport_error: AnyError) {
		let handlers = self.handlers.lock().take();
		if let Some(handlers) = handlers {
			(handlers.on_error)(transport_error);
		}
	}

	fn record_client_message(self: &Arc<Self>, message: JsonValue) {
		let is_hello = message.get("type").and_then(|value| value.as_str()) == Some("hello");
		self.messages.lock().push(message);
		self.notify.notify_waiters();
		if is_hello {
			self.send(&serde_json::json!({
				"type": "hello",
				"version": pi_protocol::PROTOCOL_VERSION,
				"serverId": self.server_id,
			}));
		}
	}
}

struct MemoryTransport {
	server: Arc<MemoryByteServer>,
}

#[async_trait::async_trait]
impl ByteTransport for MemoryTransport {
	async fn send(&self, chunk: Vec<u8>) -> Result<(), AnyError> {
		let messages = self
			.server
			.decoder
			.lock()
			.push(&chunk)
			.expect("test client messages decode");
		for message in messages {
			self.server.record_client_message(message);
		}
		Ok(())
	}

	fn close(&self) {
		let had_handlers = self.server.handlers.lock().is_some();
		self.server.handlers.lock().take();
		if had_handlers {
			self.server.client_close_count.fetch_add(1, Ordering::SeqCst);
		}
	}
}
