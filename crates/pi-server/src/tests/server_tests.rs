//! Port of `packages/server/test/server.test.ts`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use pi_chord::JsonValue;
use pi_protocol::ServerMessageDecoder;

use crate::connection::ByteConnection;
use crate::listener::ServerListener;
use crate::server::Server;
use crate::testing::TestServerHost;
use crate::tests::poll_until;
use crate::transports::unix::create_unix_server;
use crate::transports::unix::types::UnixServerOptions;
use crate::types::AnyError;

fn host() -> Arc<TestServerHost> {
	TestServerHost::new()
}

fn server_options(
	listeners: Vec<Arc<dyn ServerListener>>,
	server_id: &str,
) -> crate::types::ServerOptions {
	crate::types::ServerOptions {
		listeners,
		server_id: server_id.to_string(),
		max_frame_length: None,
		handshake_timeout_ms: None,
		on_connection_count_changed: None,
		on_error: None,
	}
}

async fn make_socket_path() -> std::path::PathBuf {
	// Upstream tempdirs under /tmp; keep paths under SUN_LEN on macOS.
	let directory = std::path::PathBuf::from("/tmp").join(format!("pss-{}", &crate::session_router::uuid_v4()[..8]));
	tokio::fs::create_dir_all(&directory).await.expect("temp dir");
	directory.join("server.sock")
}

fn unix_options(path: &std::path::Path) -> UnixServerOptions {
	UnixServerOptions {
		server_id: "00000000-0000-4000-8000-000000000001".to_string(),
		max_frame_length: None,
		handshake_timeout_ms: None,
		on_connection_count_changed: None,
		on_error: None,
		path: path.to_string_lossy().to_string(),
		mode: None,
		max_pending_bytes: None,
		graceful_close_timeout_ms: None,
	}
}

#[test]
fn requires_explicit_listeners_and_a_canonical_uuidv4_server_identity() {
	let host = host();
	// Upstream also constructs `Server` with a missing `listeners` array; the
	// Rust field is required by construction (deviation D4).
	let error = Server::new(host.clone() as Arc<dyn crate::types::ServerHost>, server_options(Vec::new(), ""))
		.expect_err("empty serverId");
	assert!(error.to_string().contains("serverId"), "error: {error}");
	let error = Server::new(
		host,
		server_options(Vec::new(), "invalid-server"),
	)
	.expect_err("invalid serverId");
	assert!(error.to_string().contains("serverId"), "error: {error}");
}

#[tokio::test]
async fn rejects_concurrent_start_calls_without_leaking_the_unix_listener() {
	let path = make_socket_path().await;
	// Part 1 (upstream "already starting"): inject a listener whose start()
	// parks on a gate so the first start is deterministically in flight.
	struct GatedListener {
		in_start: AtomicBool,
		release: tokio::sync::Notify,
	}
	#[async_trait]
	impl ServerListener for GatedListener {
		async fn start(&self, _accept: crate::connection::ByteConnectionAcceptor) -> Result<(), AnyError> {
			self.in_start.store(true, Ordering::SeqCst);
			self.release.notified().await;
			Ok(())
		}
		async fn close(&self) -> Result<(), AnyError> {
			Ok(())
		}
	}
	let gate = Arc::new(GatedListener { in_start: AtomicBool::new(false), release: tokio::sync::Notify::new() });
	let core = Server::new(
		host() as Arc<dyn crate::types::ServerHost>,
		server_options(
			vec![gate.clone() as Arc<dyn ServerListener>],
			"00000000-0000-4000-8000-000000000001",
		),
	)
	.expect("valid options");
	let first_core = core.clone();
	let starting = tokio::spawn(async move { first_core.start().await });
	poll_until(|| gate.in_start.load(Ordering::SeqCst)).await;
	let error = core.start().await.expect_err("second start");
	assert!(error.to_string().contains("starting"), "error: {error}");
	gate.release.notify_waiters();
	starting.await.expect("first start").expect("first start result");
	core.close().await.expect("core close");

	// Part 2: the Unix listener is not leaked by a start/close cycle.
	let server = create_unix_server(host(), unix_options(&path)).expect("unix server");
	server.start().await.expect("start");
	server.close().await.expect("close");
	let stat = tokio::fs::symlink_metadata(&path).await;
	assert!(stat.is_err(), "socket must be removed");
	assert_eq!(stat.unwrap_err().kind(), std::io::ErrorKind::NotFound);
	let _ = std::fs::remove_dir_all(path.parent().expect("temp dir"));
}

#[tokio::test]
async fn handshake_timeout_closes_with_a_final_hello_error_frame() {
	struct TimedOutConnection {
		closed: AtomicBool,
		final_chunk: parking_lot::Mutex<Option<Vec<u8>>>,
		finished: tokio::sync::Notify,
	}
	#[async_trait]
	impl ByteConnection for TimedOutConnection {
		fn closed(&self) -> bool {
			self.closed.load(Ordering::SeqCst)
		}
		async fn send(&self, _chunk: Vec<u8>) -> Result<(), AnyError> {
			Err(Arc::new(pi_chord::ChordError {
				message: "handshake timeout must use the terminal close frame".to_string(),
			}))
		}
		async fn close(&self, final_chunk: Option<Vec<u8>>) -> Result<(), AnyError> {
			*self.final_chunk.lock() = final_chunk;
			self.closed.store(true, Ordering::SeqCst);
			self.finished.notify_waiters();
			Ok(())
		}
	}
	let core = Server::new(
		host() as Arc<dyn crate::types::ServerHost>,
		crate::types::ServerOptions {
			listeners: Vec::new(),
			server_id: "00000000-0000-4000-8000-000000000001".to_string(),
			max_frame_length: Some(1024),
			handshake_timeout_ms: Some(10),
			on_connection_count_changed: None,
			on_error: None,
		},
	)
	.expect("valid options");
	let connection = Arc::new(TimedOutConnection {
		closed: AtomicBool::new(false),
		final_chunk: parking_lot::Mutex::new(None),
		finished: tokio::sync::Notify::new(),
	});
	core.accept(Arc::clone(&connection) as Arc<dyn ByteConnection>);

	tokio::time::timeout(std::time::Duration::from_secs(5), async {
		loop {
			if connection.closed.load(Ordering::SeqCst) {
				break;
			}
			connection.finished.notified().await;
		}
	})
	.await
	.expect("timeout fires");
	assert!(connection.closed.load(Ordering::SeqCst));
	let final_chunk = connection.final_chunk.lock().take().expect("final chunk");
	let messages = ServerMessageDecoder::new(None)
		.expect("decoder")
		.push(&final_chunk)
		.expect("final chunk decodes");
	assert_eq!(messages.len(), 1);
	let message = &messages[0];
	assert_eq!(
		message.as_object().and_then(|o| o.get("type")).and_then(JsonValue::as_str),
		Some("hello_error")
	);
	assert_eq!(
		message
			.as_object()
			.and_then(|o| o.get("error"))
			.and_then(|e| e.as_object())
			.and_then(|e| e.get("code"))
			.and_then(JsonValue::as_str),
		Some("invalid_request")
	);
	core.close().await.expect("close");
}

#[test]
fn rejects_timeout_values_above_the_maximum_timer_delay() {
	let path = "/tmp/pi-server-timeout-test.sock";
	let mut options = unix_options(std::path::Path::new(path));
	options.handshake_timeout_ms = Some(2_147_483_648);
	let error = create_unix_server(host(), options).expect_err("handshakeTimeoutMs");
	assert!(error.to_string().contains("handshakeTimeoutMs"), "error: {error}");
	let mut options = unix_options(std::path::Path::new(path));
	options.graceful_close_timeout_ms = Some(2_147_483_648);
	let error = create_unix_server(host(), options).expect_err("gracefulCloseTimeoutMs");
	assert!(error.to_string().contains("gracefulCloseTimeoutMs"), "error: {error}");
}

#[tokio::test]
async fn rejects_pending_byte_limits_smaller_than_one_maximum_frame() {
	let path = make_socket_path().await;
	let mut options = unix_options(&path);
	options.max_frame_length = Some(128);
	options.max_pending_bytes = Some(131);
	let error = create_unix_server(host(), options).expect_err("maxPendingBytes");
	assert!(error.to_string().contains("maxPendingBytes"), "error: {error}");
	let _ = std::fs::remove_dir_all(path.parent().expect("temp dir"));
}

#[tokio::test]
async fn rejects_close_and_closed_when_listener_shutdown_fails() {
	struct FailingListener {
		failure: AnyError,
	}
	#[async_trait]
	impl ServerListener for FailingListener {
		async fn start(&self, _accept: crate::connection::ByteConnectionAcceptor) -> Result<(), AnyError> {
			Ok(())
		}
		async fn close(&self) -> Result<(), AnyError> {
			Err(Arc::clone(&self.failure))
		}
	}
	let failure: AnyError = Arc::new(pi_chord::ChordError { message: "listener close failed".to_string() });
	let core = Server::new(
		host() as Arc<dyn crate::types::ServerHost>,
		server_options(
			vec![Arc::new(FailingListener { failure: Arc::clone(&failure) }) as Arc<dyn ServerListener>],
			"00000000-0000-4000-8000-000000000001",
		),
	)
	.expect("valid options");
	core.start().await.expect("start");

	let close_error = core.close().await.expect_err("close must fail");
	assert!(Arc::ptr_eq(&close_error, &failure));
	let closed_error = core.closed().await.expect_err("closed must fail");
	assert!(Arc::ptr_eq(&closed_error, &failure));
}
