//! Port of `packages/server/src/testing/client.ts`.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use async_trait::async_trait;
use pi_chord::JsonValue;
use tokio::io::AsyncReadExt;
use pi_protocol::{PROTOCOL_VERSION, ServerMessageDecoder, encode_client_message};

use crate::testing::host::Deferred;
use crate::types::AnyError;

/// Upstream `WireChannel` — the raw transport the wire client writes to.
#[async_trait]
pub trait WireChannel: Send + Sync {
	async fn send(&self, chunk: Vec<u8>) -> Result<(), AnyError>;
	async fn send_fragmented(&self, chunk: Vec<u8>, split_at: usize) -> Result<(), AnyError>;
	async fn close(&self) -> Result<(), AnyError>;
}

struct Waiter {
	predicate: Box<dyn Fn(&JsonValue) -> bool + Send + Sync>,
	tx: tokio::sync::oneshot::Sender<JsonValue>,
}

pub(crate) struct ClientInner {
	messages: parking_lot::Mutex<Vec<JsonValue>>,
	waiters: parking_lot::Mutex<HashMap<u64, Waiter>>,
	next_waiter: AtomicU64,
	attachment: parking_lot::Mutex<Option<(String, String)>>,
	decoder: parking_lot::Mutex<ServerMessageDecoder>,
	closed_value: AtomicBool,
	closed_deferred: Deferred<()>,
	channel: Arc<dyn WireChannel>,
}

impl ClientInner {
	fn fail(&self) {
		self.waiters.lock().clear();
		// Dropping the senders rejects pending `next()` futures (upstream
		// rejects each waiter with the close/error reason).
	}
}

/// Upstream `ProtocolTestClient`.
#[derive(Clone)]
pub struct ProtocolTestClient {
	inner: Arc<ClientInner>,
	request_sequence: Arc<AtomicU64>,
}

impl ProtocolTestClient {
	pub fn new(channel: Arc<dyn WireChannel>) -> ProtocolTestClient {
		ProtocolTestClient {
			inner: Arc::new(ClientInner {
				messages: parking_lot::Mutex::new(Vec::new()),
				waiters: parking_lot::Mutex::new(HashMap::new()),
				next_waiter: AtomicU64::new(1),
				attachment: parking_lot::Mutex::new(None),
				decoder: parking_lot::Mutex::new(ServerMessageDecoder::new(None).expect("default frame options")),
				closed_value: AtomicBool::new(false),
				closed_deferred: Deferred::new(),
				channel,
			}),
			request_sequence: Arc::new(AtomicU64::new(0)),
		}
	}

	/// Interior-state handle for the reader task.
	pub(crate) fn shared(&self) -> Arc<ClientInner> {
		Arc::clone(&self.inner)
	}

	/// Reader-task entry points over shared state.
	pub(crate) fn receive_shared(inner: &Arc<ClientInner>, chunk: &[u8]) {
		let result: Result<Vec<JsonValue>, pi_protocol::ProtocolValidationError> = {
			let mut decoder = inner.decoder.lock();
			decoder.push(chunk)
		};
		match result {
			Ok(messages) => {
				for message in messages {
					if message.as_object().and_then(|o| o.get("type")).and_then(JsonValue::as_str) == Some("attachment") {
						let attachment = message
							.as_object()
							.and_then(|o| o.get("attachment"))
							.and_then(|a| a.as_object())
							.map(|a| {
								(
									a.get("sessionId").and_then(JsonValue::as_str).unwrap_or_default().to_string(),
									a.get("attachmentId").and_then(JsonValue::as_str).unwrap_or_default().to_string(),
								)
							});
						*inner.attachment.lock() = attachment;
					}
					inner.messages.lock().push(message.clone());
					let mut waiters = inner.waiters.lock();
					let matched: Vec<u64> = waiters
						.iter()
						.filter(|(_, waiter)| (waiter.predicate)(&message))
						.map(|(id, _)| *id)
						.collect();
					for id in matched {
						if let Some(waiter) = waiters.remove(&id) {
							let _ = waiter.tx.send(message.clone());
						}
					}
				}
			}
			Err(_) => inner.fail(),
		}
	}

	pub(crate) fn fail_shared(inner: &Arc<ClientInner>) {
		inner.fail();
	}

	pub(crate) fn mark_closed_shared(inner: &Arc<ClientInner>) {
		if inner.closed_value.swap(true, Ordering::SeqCst) {
			return;
		}
		inner.closed_deferred.resolve(());
		inner.fail();
	}

	pub fn closed(&self) -> bool {
		self.inner.closed_value.load(Ordering::SeqCst)
	}

	/// Snapshot of received messages (upstream `client.messages`).
	pub fn messages(&self) -> Vec<JsonValue> {
		self.inner.messages.lock().clone()
	}

	pub async fn hello(&self) -> Result<JsonValue, AnyError> {
		self.hello_with_version(PROTOCOL_VERSION).await
	}

	pub async fn hello_with_version(&self, version: i64) -> Result<JsonValue, AnyError> {
		let response = self.next(|message| {
			let kind = message.as_object().and_then(|o| o.get("type")).and_then(JsonValue::as_str);
			kind == Some("hello") || kind == Some("hello_error")
		});
		self.send_message(serde_json::json!({ "type": "hello", "version": version })).await?;
		response.await
	}

	pub async fn request_service(
		&self,
		target: JsonValue,
		call: JsonValue,
		id: impl Into<String>,
	) -> Result<JsonValue, AnyError> {
		let id = id.into();
		let response_id = id.clone();
		let response = self.next(move |message| {
			let Some(object) = message.as_object() else { return false };
			object.get("type").and_then(JsonValue::as_str) == Some("response")
				&& object.get("id").and_then(JsonValue::as_str) == Some(response_id.as_str())
		});
		self.send_message(serde_json::json!({
			"type": "request",
			"id": id,
			"target": target,
			"call": call,
		}))
		.await?;
		response.await
	}

	fn next_request_id(&self) -> String {
		format!("request-{}", self.request_sequence.fetch_add(1, Ordering::SeqCst) + 1)
	}

	pub async fn attach(&self, server_id: &str, session_id: &str) -> Result<JsonValue, AnyError> {
		let id = self.next_request_id();
		self.request_service(
			serde_json::json!({ "serverId": server_id }),
			serde_json::json!({ "serviceId": "pi.session-management", "member": "attach", "args": [session_id] }),
			id,
		)
		.await
	}

	pub async fn request_session_service(
		&self,
		server_id: &str,
		session_id: &str,
		call: JsonValue,
		id: Option<String>,
	) -> Result<JsonValue, AnyError> {
		let attachment = self.inner.attachment.lock().clone();
		let target = match attachment {
			Some((attached_session, attachment_id)) if attached_session == session_id => {
				serde_json::json!({ "serverId": server_id, "sessionId": session_id, "attachmentId": attachment_id })
			}
			_ => serde_json::json!({ "serverId": server_id, "sessionId": session_id, "attachmentId": "missing-attachment" }),
		};
		match id {
			Some(id) => self.request_service(target, call, id).await,
			None => {
				let id = self.next_request_id();
				self.request_service(target, call, id).await
			}
		}
	}

	pub async fn send_message(&self, message: JsonValue) -> Result<(), AnyError> {
		let frame = encode_client_message(&message, None).map_err(|error| crate::types::any(error))?;
		self.inner.channel.send(frame).await
	}

	pub async fn send_bytes(&self, chunk: Vec<u8>) -> Result<(), AnyError> {
		self.inner.channel.send(chunk).await
	}

	pub async fn send_fragmented_message(&self, message: JsonValue, split_at: usize) -> Result<(), AnyError> {
		let frame = encode_client_message(&message, None).map_err(|error| crate::types::any(error))?;
		self.inner.channel.send_fragmented(frame, split_at).await
	}

	pub fn next(
		&self,
		predicate: impl Fn(&JsonValue) -> bool + Send + Sync + 'static,
	) -> impl Future<Output = Result<JsonValue, AnyError>> {
		self.next_from(0, predicate)
	}

	pub fn next_from(
		&self,
		index: usize,
		predicate: impl Fn(&JsonValue) -> bool + Send + Sync + 'static,
	) -> impl Future<Output = Result<JsonValue, AnyError>> {
		let inner = Arc::clone(&self.inner);
		async move {
			{
				let messages = inner.messages.lock();
				if let Some(existing) = messages.iter().skip(index).find(|message| predicate(message)) {
					return Ok(existing.clone());
				}
			}
			if inner.closed_value.load(Ordering::SeqCst) {
				return Err(wire_error("Wire client is closed"));
			}
			let (tx, rx) = tokio::sync::oneshot::channel();
			let waiter_id = inner.next_waiter.fetch_add(1, Ordering::SeqCst);
			inner.waiters.lock().insert(waiter_id, Waiter { predicate: Box::new(predicate), tx });
			match rx.await {
				Ok(message) => Ok(message),
				Err(_) => Err(wire_error("Wire connection closed")),
			}
		}
	}

	pub async fn wait_for_close(&self) {
		self.inner.closed_deferred.promise().await;
	}

	pub async fn close(&self) -> Result<(), AnyError> {
		self.inner.channel.close().await
	}

	pub fn receive(&self, chunk: &[u8]) {
		Self::receive_shared(&self.inner, chunk);
	}

	pub fn mark_closed(&self) {
		Self::mark_closed_shared(&self.inner);
	}

	pub fn fail(&self) {
		self.inner.fail();
	}
}

fn wire_error(message: &str) -> AnyError {
	Arc::new(pi_chord::ChordError { message: message.to_string() })
}

/// Upstream `connectUnixTestClient`.
pub async fn connect_unix_test_client(path: &str) -> Result<ProtocolTestClient, AnyError> {
	let socket = tokio::net::UnixStream::connect(path).await.map_err(crate::types::io_error)?;
	let (mut read_half, write_half) = socket.into_split();
	let client = ProtocolTestClient::new(Arc::new(SocketChannel {
		write_half: tokio::sync::Mutex::new(write_half),
	}));
	let shared = client.shared();
	tokio::spawn(async move {
		let mut buffer = vec![0u8; 64 * 1024];
		loop {
			match read_half.read(&mut buffer).await {
				Ok(0) => break,
				Ok(read) => ProtocolTestClient::receive_shared(&shared, &buffer[..read]),
				Err(_) => ProtocolTestClient::fail_shared(&shared),
			}
		}
		ProtocolTestClient::mark_closed_shared(&shared);
	});
	Ok(client)
}

struct SocketChannel {
	write_half: tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>,
}

#[async_trait]
impl WireChannel for SocketChannel {
	async fn send(&self, chunk: Vec<u8>) -> Result<(), AnyError> {
		use tokio::io::AsyncWriteExt;
		self.write_half.lock().await.write_all(&chunk).await.map_err(crate::types::io_error)
	}

	async fn send_fragmented(&self, chunk: Vec<u8>, split_at: usize) -> Result<(), AnyError> {
		self.send(chunk[..split_at].to_vec()).await?;
		self.send(chunk[split_at..].to_vec()).await
	}

	async fn close(&self) -> Result<(), AnyError> {
		use tokio::io::AsyncWriteExt;
		let mut half = self.write_half.lock().await;
		half.shutdown().await.map_err(crate::types::io_error)
	}
}
