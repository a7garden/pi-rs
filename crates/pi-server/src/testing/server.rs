//! Port of `packages/server/src/testing/server.rs`.

use std::sync::Arc;

use crate::server::Server;
use crate::types::{AnyError, ErrorCallback, ServerHost, ServerOptions};

/// Upstream `TestServer`.
pub struct TestServer {
    pub server: Server,
    pub host: Arc<dyn ServerHost>,
}

/// Upstream `TestServerOptions` (defaults for the omitted fields).
pub struct TestServerOptions {
    pub host: Option<Arc<dyn ServerHost>>,
    pub server_id: Option<String>,
    pub max_frame_length: Option<usize>,
    pub handshake_timeout_ms: Option<u64>,
    pub on_error: Option<ErrorCallback>,
    pub listeners: Option<Vec<Arc<dyn crate::listener::ServerListener>>>,
}

impl Default for TestServerOptions {
    fn default() -> Self {
        TestServerOptions {
            host: None,
            server_id: None,
            max_frame_length: None,
            handshake_timeout_ms: None,
            on_error: None,
            listeners: None,
        }
    }
}

/// Create an unstarted Server with deterministic defaults for transport
/// conformance tests.
pub fn create_test_server(options: TestServerOptions) -> Result<TestServer, AnyError> {
    let host = options
        .host
        .clone()
        .unwrap_or_else(|| crate::testing::host::TestServerHost::new() as Arc<dyn ServerHost>);
    let server = Server::new(
        Arc::clone(&host),
        ServerOptions {
            listeners: options.listeners.unwrap_or_default(),
            server_id: options
                .server_id
                .unwrap_or_else(|| "00000000-0000-4000-8000-000000000001".to_string()),
            max_frame_length: options.max_frame_length,
            handshake_timeout_ms: options.handshake_timeout_ms,
            on_connection_count_changed: None,
            on_error: options.on_error,
        },
    )?;
    Ok(TestServer { server, host })
}
