//! Port of `packages/server/test/listener.test.ts`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use std::sync::OnceLock;

use crate::connection::ByteConnectionAcceptor;
use crate::listener::ServerListener;
use crate::testing::{TestServer, TestServerOptions, create_test_server};
use crate::types::AnyError;

struct TestListener {
	accept: OnceLock<ByteConnectionAcceptor>,
	start_count: AtomicUsize,
	close_count: AtomicUsize,
	start_error: Option<AnyError>,
}

impl TestListener {
	fn new(start_error: Option<AnyError>) -> Arc<TestListener> {
		Arc::new(TestListener {
			accept: OnceLock::new(),
			start_count: AtomicUsize::new(0),
			close_count: AtomicUsize::new(0),
			start_error,
		})
	}
}

#[async_trait]
impl ServerListener for TestListener {
	async fn start(&self, accept: ByteConnectionAcceptor) -> Result<(), AnyError> {
		self.start_count.fetch_add(1, Ordering::SeqCst);
		let _ = self.accept.set(accept);
		if let Some(error) = &self.start_error {
			return Err(Arc::clone(error));
		}
		Ok(())
	}

	async fn close(&self) -> Result<(), AnyError> {
		self.close_count.fetch_add(1, Ordering::SeqCst);
		Ok(())
	}
}

fn failure(message: &str) -> AnyError {
	Arc::new(pi_chord::ChordError { message: message.to_string() })
}

#[tokio::test]
async fn starts_and_closes_every_configured_listener() {
	let first = TestListener::new(None);
	let second = TestListener::new(None);
	let TestServer { server, .. } = create_test_server(TestServerOptions {
		listeners: Some(vec![first.clone(), second.clone()]),
		..Default::default()
	})
	.expect("test server");

	server.start().await.expect("start");
	assert!(first.accept.get().is_some(), "first accept");
	assert!(second.accept.get().is_some(), "second accept");

	server.close().await.expect("close");
	assert_eq!(first.close_count.load(Ordering::SeqCst), 1);
	assert_eq!(second.close_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn closes_previously_started_listeners_when_startup_fails() {
	let first = TestListener::new(None);
	let listener_failure = failure("listener failed");
	let second = TestListener::new(Some(Arc::clone(&listener_failure)));
	let TestServer { server, .. } = create_test_server(TestServerOptions {
		listeners: Some(vec![first.clone(), second.clone()]),
		..Default::default()
	})
	.expect("test server");

	let error = server.start().await.expect_err("start must fail");
	assert!(Arc::ptr_eq(&error, &listener_failure));
	assert_eq!(first.close_count.load(Ordering::SeqCst), 1);
	assert_eq!(second.close_count.load(Ordering::SeqCst), 0);
}
