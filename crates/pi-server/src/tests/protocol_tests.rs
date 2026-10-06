//! Port of `packages/server/test/protocol.test.ts`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pi_chord::JsonValue;
use pi_protocol::{PROTOCOL_VERSION, encode_client_message, encode_frame};

use crate::connection::{ByteConnection, ByteConnectionHandler};
use crate::server::Server;
use crate::tests::loopback::Loopback;
use crate::tests::msg;
use crate::testing::TestServerHost;
use crate::types::{AnyError, ErrorCallback, ServerHost, ServerOptions};

const SERVER_ID: &str = "00000000-0000-4000-8000-000000000001";

fn test_server(on_error: Option<ErrorCallback>) -> Server {
	Server::new(
		TestServerHost::new() as Arc<dyn ServerHost>,
		ServerOptions {
			listeners: Vec::new(),
			server_id: SERVER_ID.to_string(),
			max_frame_length: None,
			handshake_timeout_ms: None,
			on_connection_count_changed: None,
			on_error,
		},
	)
	.expect("valid options")
}

fn connect() -> (Server, Loopback) {
	let server = test_server(None);
	let loopback = crate::tests::loopback::connect(&server);
	(server, loopback)
}

#[tokio::test]
async fn requires_hello_as_the_first_message() {
	let (server, loopback) = connect();
	let client = &loopback.client;
	let response = client.next(|message| msg::kind(message) == "hello_error");
	client
		.send_message(serde_json::json!({
			"type": "request",
			"id": "request-1",
			"target": { "serverId": SERVER_ID },
			"call": { "serviceId": "pi.session-directory", "member": "list", "args": [] },
		}))
		.await
		.expect("send");
	let message = response.await.expect("hello_error");
	assert_eq!(msg::kind(&message), "hello_error");
	assert_eq!(msg::error_code(&message), "invalid_request");
	client.wait_for_close().await;
	server.close().await.expect("close");
}

#[tokio::test]
async fn rejects_unsupported_protocol_versions() {
	let (server, loopback) = connect();
	let client = &loopback.client;
	let message = client.hello_with_version(PROTOCOL_VERSION + 1).await.expect("hello_error");
	assert_eq!(msg::kind(&message), "hello_error");
	assert_eq!(msg::error_code(&message), "version");
	client.wait_for_close().await;
	server.close().await.expect("close");
}

#[tokio::test]
async fn accepts_fragmented_hello_and_request_frames() {
	let (server, loopback) = connect();
	let client = &loopback.client;
	let hello = encode_client_message(
		&serde_json::json!({ "type": "hello", "version": PROTOCOL_VERSION }),
		None,
	)
	.expect("hello frame");
	let hello_response = client.next(|message| msg::kind(message) == "hello");
	client
		.send_fragmented_message(
			serde_json::json!({ "type": "hello", "version": PROTOCOL_VERSION }),
			hello.len() / 2,
		)
		.await
		.expect("fragmented hello");
	let hello_message = hello_response.await.expect("hello");
	assert_eq!(msg::kind(&hello_message), "hello");
	assert_eq!(
		hello_message.as_object().and_then(|o| o.get("serverId")).and_then(JsonValue::as_str),
		Some(SERVER_ID)
	);

	let response = client.next(|message| msg::kind(message) == "response");
	let request = serde_json::json!({
		"type": "request",
		"id": "request-1",
		"target": { "serverId": SERVER_ID },
		"call": { "serviceId": "pi.session-directory", "member": "list", "args": [] },
	});
	let frame = encode_client_message(&request, None).expect("request frame");
	client
		.send_fragmented_message(request, frame.len() / 2)
		.await
		.expect("fragmented request");
	let response_message = response.await.expect("response");
	assert_eq!(msg::ok(&response_message), false);
	assert_eq!(msg::error_code(&response_message), "internal_error");
	server.close().await.expect("close");
}

#[tokio::test]
async fn rejects_hostile_framed_input() {
	for (label, bytes) in [
		(
			"malformed CBOR",
			encode_frame(&[0xff]).expect("frame"),
		),
		(
			"schema-invalid CBOR",
			encode_frame(
				&pi_protocol::encode_cbor(
					&pi_protocol::CborValue::from(&serde_json::json!({
						"type": "hello", "version": 1, "extra": true
					})),
					None,
				)
				.expect("cbor"),
			)
			.expect("frame"),
		),
		// 0x01000001 exceeds DEFAULT_MAX_FRAME_LENGTH by one byte.
		("oversized frame", vec![1, 0, 0, 1]),
	] {
		let (server, loopback) = connect();
		let client = &loopback.client;
		let response = client.next(|message| msg::kind(message) == "hello_error");
		client.send_bytes(bytes).await.expect("send bytes");
		let message = response.await.unwrap_or_else(|_| panic!("hello_error for {label}"));
		assert_eq!(msg::error_code(&message), "invalid_request", "{label}");
		client.wait_for_close().await;
		server.close().await.expect("close");
	}
}

#[tokio::test]
async fn rejects_a_second_hello_after_completing_the_handshake() {
	let (server, loopback) = connect();
	let client = &loopback.client;
	client.hello().await.expect("hello");
	let response = client.next(|message| msg::kind(message) == "hello_error");
	client
		.send_message(serde_json::json!({ "type": "hello", "version": PROTOCOL_VERSION }))
		.await
		.expect("send");
	let message = response.await.expect("hello_error");
	assert_eq!(msg::error_code(&message), "invalid_request");
	assert!(
		msg::error_message(&message).contains("first message"),
		"message: {}",
		msg::error_message(&message)
	);
	client.wait_for_close().await;
	server.close().await.expect("close");
}

#[tokio::test]
async fn processes_a_hello_and_request_coalesced_in_one_byte_chunk() {
	let (server, loopback) = connect();
	let client = &loopback.client;
	let hello = encode_client_message(
		&serde_json::json!({ "type": "hello", "version": PROTOCOL_VERSION }),
		None,
	)
	.expect("hello frame");
	let request = encode_client_message(
		&serde_json::json!({
			"type": "request",
			"id": "request-1",
			"target": { "serverId": SERVER_ID },
			"call": { "serviceId": "pi.session-directory", "member": "list", "args": [] },
		}),
		None,
	)
	.expect("request frame");
	let mut wire = hello.clone();
	wire.extend_from_slice(&request);

	let hello_response = client.next(|message| msg::kind(message) == "hello");
	let request_response = client.next(|message| msg::kind(message) == "response");
	client.send_bytes(wire).await.expect("send");
	let hello_message = hello_response.await.expect("hello");
	assert_eq!(msg::kind(&hello_message), "hello");
	let response = request_response.await.expect("response");
	assert_eq!(msg::kind(&response), "response");
	assert_eq!(msg::id(&response), "request-1");
	assert_eq!(msg::ok(&response), false);
	assert_eq!(msg::error_code(&response), "internal_error");
	server.close().await.expect("close");
}

#[tokio::test]
async fn reports_a_truncated_final_frame_when_the_peer_closes() {
	let errors: Arc<parking_lot::Mutex<Vec<AnyError>>> = Arc::default();
	let errors_for_callback = Arc::clone(&errors);
	let server = test_server(Some(Arc::new(move |error| {
		errors_for_callback.lock().push(error);
	})));
	let handler: Arc<std::sync::OnceLock<Arc<dyn ByteConnectionHandler>>> = Arc::default();
	struct SilentConnection {
		closed: Arc<AtomicBool>,
	}
	#[async_trait::async_trait]
	impl ByteConnection for SilentConnection {
		fn closed(&self) -> bool {
			self.closed.load(Ordering::SeqCst)
		}
		async fn send(&self, _chunk: Vec<u8>) -> Result<(), AnyError> {
			Ok(())
		}
		async fn close(&self, _final_chunk: Option<Vec<u8>>) -> Result<(), AnyError> {
			self.closed.store(true, Ordering::SeqCst);
			Ok(())
		}
	}
	let closed = Arc::new(AtomicBool::new(false));
	let connection = Arc::new(SilentConnection { closed: Arc::clone(&closed) });
	assert!(handler.set(server.acceptor()(connection)).is_ok(), "single accept");
	let handler = handler.get().expect("handler");

	handler.on_data(&[0, 0, 0, 2, 1]);
	handler.on_close();

	assert!(!closed.load(Ordering::SeqCst));
	let recorded = errors.lock();
	assert_eq!(recorded.len(), 1, "expected one error");
	assert!(
		recorded[0].to_string().to_lowercase().contains("truncated"),
		"error: {}",
		recorded[0]
	);
	server.close().await.expect("close");
}
