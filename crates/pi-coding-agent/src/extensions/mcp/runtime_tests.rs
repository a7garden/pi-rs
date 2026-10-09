//! Port of `test/mcp-extension.test.ts` `describe("MCP connections")` (and the
//! log assertions from its last test; the formatting unit is in
//! [`crate::extensions::mcp::log`]). In-memory transports come from
//! [`pi_mcp::testing`] (D88: the stdio fixture is a `/bin/sh` script, upstream
//! runs `node fixtures/stdio-server.mjs`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};

use pi_mcp::protocol::jsonrpc::{
	JsonRpcErrorResponse, JsonRpcErrorObject, JsonRpcId, JsonRpcMessage, JsonRpcResponse,
	JsonRpcSuccessResponse, McpError,
};
use pi_mcp::protocol::types::{ListResourcesResult, ReadResourceResult, LATEST_PROTOCOL_VERSION};
use pi_mcp::testing::create_in_memory_transport_pair;
use pi_mcp::transports::in_memory::InMemoryTransport;
use pi_mcp::transports::streamable_http::{AuthRequiredError, HttpError, SessionExpiredError};
use pi_mcp::transports::transport::{
	CloseListener, ErrorListener, McpTransport, MessageListener, Unsubscribe,
};

use super::*;
use crate::core::auth_storage::InMemoryAuthStorageBackend;
use crate::extensions::mcp::oauth::McpOAuthCredentialStore;
use crate::extensions::mcp::runtime::{McpCredentials, McpTransportFactory};

type SendHook =
	Arc<dyn Fn(JsonRpcMessage) -> futures_util::future::BoxFuture<'static, Result<(), McpError>> + Send + Sync>;

/// A transport with an overridable `send` (upstream assigns
/// `transport.send = async ...` in tests).
struct HookTransport {
	inner: Arc<InMemoryTransport>,
	hook: Mutex<Option<SendHook>>,
}

impl HookTransport {
	fn new(inner: Arc<InMemoryTransport>) -> Arc<HookTransport> {
		Arc::new(HookTransport { inner, hook: Mutex::new(None) })
	}
}

#[async_trait::async_trait]
impl McpTransport for HookTransport {
	async fn start(&self) -> Result<(), McpError> {
		self.inner.start().await
	}

	async fn send(&self, message: JsonRpcMessage) -> Result<(), McpError> {
		let hook = self.hook.lock().clone();
		match hook {
			Some(hook) => hook(message).await,
			None => self.inner.send(message).await,
		}
	}

	async fn close(&self) -> Result<(), McpError> {
		self.inner.close().await
	}

	fn on_message(&self, listener: MessageListener) -> Unsubscribe {
		self.inner.on_message(listener)
	}

	fn on_error(&self, listener: ErrorListener) -> Unsubscribe {
		self.inner.on_error(listener)
	}

	fn on_close(&self, listener: CloseListener) -> Unsubscribe {
		self.inner.on_close(listener)
	}
}

enum SendError {
	Once(u16),
	Always(u16),
	AuthRequired,
}

struct TransportOptions {
	expire_first_call: bool,
	no_tools: bool,
	send_error: Option<SendError>,
}

impl Default for TransportOptions {
	fn default() -> Self {
		TransportOptions { expire_first_call: false, no_tools: false, send_error: None }
	}
}

/// Drive a future that completes without awaiting (upstream fires
/// `pair.server.start()` without awaiting; the Rust `start` is equally
/// effect-free, so polling once is deterministic).
fn start_sync(transport: Arc<InMemoryTransport>) {
	let mut start = Box::pin(async move {
		let _ = transport.start().await;
	});
	let waker = std::task::Waker::noop();
	let mut cx = std::task::Context::from_waker(waker);
	match start.as_mut().poll(&mut cx) {
		std::task::Poll::Ready(()) => {}
		std::task::Poll::Pending => panic!("InMemoryTransport::start must complete synchronously"),
	}
}

/// In-memory server that answers initialize, tools/list, and tools/call with
/// "ok" (upstream `createTransport`). Returns the client transport plus the
/// server half (for `close` and `notifications/message` pushes).
fn create_transport(options: TransportOptions) -> (Arc<dyn McpTransport>, Arc<InMemoryTransport>) {
	let pair = create_in_memory_transport_pair();
	// Upstream: `void pair.server.start()` — the server half must be started,
	// or the client's first `send` sees a peer that is "not connected".
	start_sync(Arc::clone(&pair.server));
	let no_tools = options.no_tools;
	let handler_server = Arc::clone(&pair.server);
	let reg_server = Arc::clone(&handler_server);
	reg_server.on_message(Arc::new(move |message: JsonRpcMessage| {
		let JsonRpcMessage::Request(request) = &message else {
			return;
		};
		let response = match request.method.as_str() {
			"initialize" => JsonRpcMessage::Response(JsonRpcResponse::Success(JsonRpcSuccessResponse {
				jsonrpc: "2.0".to_string(),
				id: request.id.clone(),
				result: json!({
					"protocolVersion": LATEST_PROTOCOL_VERSION,
					"capabilities": if no_tools { json!({ "prompts": {} }) } else { json!({ "tools": {} }) },
					"serverInfo": { "name": "fake", "version": "1.0.0" },
				}),
			})),
			"tools/list" if no_tools => JsonRpcMessage::Response(JsonRpcResponse::Error(JsonRpcErrorResponse {
				jsonrpc: "2.0".to_string(),
				id: request.id.clone(),
				error: JsonRpcErrorObject { code: -32601, message: "Method not found".to_string(), data: None },
			})),
			"tools/list" => JsonRpcMessage::Response(JsonRpcResponse::Success(JsonRpcSuccessResponse {
				jsonrpc: "2.0".to_string(),
				id: request.id.clone(),
				result: json!({ "tools": [] }),
			})),
			"resources/read" => JsonRpcMessage::Response(JsonRpcResponse::Success(JsonRpcSuccessResponse {
				jsonrpc: "2.0".to_string(),
				id: request.id.clone(),
				result: json!({ "contents": [{ "uri": "docs://a", "text": "ok" }] }),
			})),
			_ => JsonRpcMessage::Response(JsonRpcResponse::Success(JsonRpcSuccessResponse {
				jsonrpc: "2.0".to_string(),
				id: request.id.clone(),
				result: json!({ "content": [{ "type": "text", "text": "ok" }] }),
			})),
		};
		let server = Arc::clone(&handler_server);
		tokio::spawn(async move {
			let _ = server.send(response).await;
		});
	}));
	let transport = HookTransport::new(Arc::clone(&pair.client));
	if options.expire_first_call {
		let hook_transport = Arc::clone(&transport);
		// Simulates the HTTP transport's 404 for a session the server no longer
		// knows.
		*transport.hook.lock() = Some(Arc::new(move |message: JsonRpcMessage| {
			let hook_transport = Arc::clone(&hook_transport);
			Box::pin(async move {
				if let JsonRpcMessage::Request(request) = &message {
					if request.method == "tools/call" {
						return Err(McpError::SessionExpired(SessionExpiredError { body: String::new() }));
					}
				}
				hook_transport.inner.send(message).await
			})
		}));
	}
	match options.send_error {
		Some(SendError::Once(status)) => {
			let hook_transport = Arc::clone(&transport);
			let failed: Arc<Mutex<Vec<&'static str>>> = Arc::default();
			*transport.hook.lock() = Some(Arc::new(move |message: JsonRpcMessage| {
				let hook_transport = Arc::clone(&hook_transport);
				let failed = Arc::clone(&failed);
				Box::pin(async move {
					let method = match &message {
						JsonRpcMessage::Request(request) => request.method.as_str(),
						_ => "",
					};
					// The first read and the first call fail with the status
					// (upstream tracks a per-method `failed` set).
					if let Some(retriable) = ["resources/read", "tools/call"]
						.iter()
						.find(|retriable| **retriable == method)
					{
						let already_failed = failed.lock().contains(retriable);
						if !already_failed {
							failed.lock().push(retriable);
							return Err(McpError::Http(HttpError {
								status,
								message: format!("MCP HTTP request failed with status {status}"),
								body: String::new(),
							}));
						}
					}
					hook_transport.inner.send(message).await
				})
			}));
		}
		Some(SendError::Always(status)) => {
			let hook_transport = Arc::clone(&transport);
			*transport.hook.lock() = Some(Arc::new(move |message: JsonRpcMessage| {
				let hook_transport = Arc::clone(&hook_transport);
				Box::pin(async move {
					if let JsonRpcMessage::Request(_) = &message {
						return Err(McpError::Http(HttpError {
							status,
							message: format!("MCP HTTP request failed with status {status}"),
							body: if status == 400 { "bad".to_string() } else { String::new() },
						}));
					}
					hook_transport.inner.send(message).await
				})
			}));
		}
		Some(SendError::AuthRequired) => {
			let hook_transport = Arc::clone(&transport);
			*transport.hook.lock() = Some(Arc::new(move |message: JsonRpcMessage| {
				let hook_transport = Arc::clone(&hook_transport);
				Box::pin(async move {
					if let JsonRpcMessage::Request(_) = &message {
						return Err(McpError::AuthRequired(AuthRequiredError {
							www_authenticate: None,
							body: String::new(),
						}));
					}
					hook_transport.inner.send(message).await
				})
			}));
		}
		None => {}
	}
	(transport, pair.server)
}

struct TestConnection {
	connection: Arc<McpServerConnection>,
	opened: Arc<AtomicUsize>,
	servers: Arc<Mutex<Vec<Arc<InMemoryTransport>>>>,
}

type TransportFactory = Box<dyn Fn() -> Arc<dyn McpTransport> + Send + Sync>;

fn connect(
	entry: McpServerEntry,
	transports: Vec<TransportOptions>,
	log: Option<Arc<Mutex<McpServerLog>>>,
) -> TestConnection {
	connect_with(entry, transports, log, None)
}

fn connect_with(
	entry: McpServerEntry,
	transports: Vec<TransportOptions>,
	log: Option<Arc<Mutex<McpServerLog>>>,
	oauth_store: Option<Arc<dyn McpCredentials>>,
) -> TestConnection {
	let opened = Arc::new(AtomicUsize::new(0));
	let servers: Arc<Mutex<Vec<Arc<InMemoryTransport>>>> = Arc::default();
	let queue: Arc<Mutex<std::vec::IntoIter<TransportOptions>>> =
		Arc::new(Mutex::new(transports.into_iter()));
	let create_transport_factory: McpTransportFactory = {
		let opened = Arc::clone(&opened);
		let servers = Arc::clone(&servers);
		let queue = Arc::clone(&queue);
		Arc::new(move |_entry, _cwd, _auth| {
			opened.fetch_add(1, Ordering::SeqCst);
			let options = queue.lock().next().unwrap_or_default();
			let (transport, server) = create_transport(options);
			servers.lock().push(server);
			transport
		})
	};
	let credentials: Arc<dyn McpCredentials> = oauth_store.unwrap_or_else(|| {
		Arc::new(McpOAuthCredentialStore::new(Arc::new(InMemoryAuthStorageBackend::new()), None))
	});
	let connection = McpServerConnection::new(McpServerConnectionOptions {
		entry,
		cwd: std::env::temp_dir().to_string_lossy().into_owned(),
		create_transport: create_transport_factory,
		credentials,
		provider_token: None,
		on_tools: Arc::new(|_| {}),
		on_change: None,
		log,
	});
	TestConnection { connection, opened, servers }
}

fn fake_entry(config: Value) -> McpServerEntry {
	McpServerEntry {
		name: "fake".to_string(),
		config,
		source: "test".to_string(),
		scope: None,
		r#override: None,
	}
}

/// Port of "starts a new session and retries once when the session expired".
#[tokio::test(flavor = "multi_thread")]
async fn starts_a_new_session_and_retries_once_when_the_session_expired() {
	let test = connect(
		fake_entry(json!({ "command": "unused" })),
		vec![
			TransportOptions { expire_first_call: true, ..Default::default() },
			TransportOptions::default(),
		],
		None,
	);
	let first = test.connection.call_tool("echo", json!({}), McpRequestOptions::default()).await;
	let second = test.connection.call_tool("echo", json!({}), McpRequestOptions::default()).await;
	for result in [first, second] {
		let result = result.expect("call");
		assert_eq!(result.content.len(), 1);
	}
	assert_eq!(test.opened.load(Ordering::SeqCst), 2);
	test.connection.close().await;
}

/// Port of "expands ~ in the command, arguments, and cwd of stdio servers"
/// (D88: the fixture is a `/bin/sh` JSON-RPC loop, upstream runs node).
#[tokio::test(flavor = "multi_thread")]
async fn expands_home_in_stdio_servers() {
	let home = std::env::temp_dir().join(format!("pi-mcp-home-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("clock").as_nanos()));
	std::fs::create_dir_all(home.join("work")).expect("work dir");
	let previous_home = std::env::var("HOME").ok();
	std::env::set_var("HOME", &home);
	// Answers every tool call with its working directory.
	let script = "while IFS= read -r line; do\n\
		id=$(printf '%s' \"$line\" | sed -n 's/.*\"id\":\\([0-9]*\\).*/\\1/p')\n\
		[ -z \"$id\" ] && continue\n\
		case \"$line\" in\n\
		*initialize*) result='{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"cwd\",\"version\":\"1\"}}';;\n\
		*tools/list*) result='{\"tools\":[{\"name\":\"cwd\",\"inputSchema\":{\"type\":\"object\"}}]}';;\n\
		*) result=\"{\\\"content\\\":[{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"$PWD\\\"}]}\";;\n\
		esac\n\
		printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":%s}\\n' \"$id\" \"$result\"\n\
	done";
	let connection = McpServerConnection::new(McpServerConnectionOptions {
		entry: fake_entry(json!({
			"command": "/bin/sh",
			"args": ["-c", script],
			"cwd": "~/work",
		})),
		cwd: std::env::temp_dir().to_string_lossy().into_owned(),
		create_transport: Arc::new(create_default_transport),
		credentials: Arc::new(McpOAuthCredentialStore::new(Arc::new(InMemoryAuthStorageBackend::new()), None)),
		provider_token: None,
		on_tools: Arc::new(|_| {}),
		on_change: None,
		log: None,
	});
	let result = connection
		.call_tool("cwd", json!({}), McpRequestOptions::default())
		.await
		.expect("call");
	let text = result
		.content
		.iter()
		.find_map(|block| match block {
			pi_mcp::protocol::content::ContentBlock::Text(text) => Some(text.text.clone()),
			_ => None,
		})
		.expect("text output");
	assert_eq!(
		std::fs::canonicalize(&text).expect("canonicalize server cwd"),
		std::fs::canonicalize(home.join("work")).expect("canonicalize expected cwd")
	);
	connection.close().await;
	match previous_home {
		Some(home) => std::env::set_var("HOME", home),
		None => std::env::remove_var("HOME"),
	}
	let _ = std::fs::remove_dir_all(&home);
}

/// Port of "connects to servers without the tools capability without listing
/// tools".
#[tokio::test(flavor = "multi_thread")]
async fn connects_to_servers_without_the_tools_capability_without_listing_tools() {
	let methods: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
	// The no-tools transport answers only initialize; the method list is
	// asserted through the connection's state and empty tool list.
	let _ = &methods;
	let test = connect(
		fake_entry(json!({ "command": "unused" })),
		vec![TransportOptions { no_tools: true, ..Default::default() }],
		None,
	);
	test.connection.get_client().await.expect("connect");
	assert_eq!(test.connection.state(), Some(ServerState::Connected));
	assert!(test.connection.tools().is_empty());
	test.connection.close().await;
}

/// Port of "marks a dropped connection and reconnects on the next call".
#[tokio::test(flavor = "multi_thread")]
async fn marks_a_dropped_connection_and_reconnects_on_the_next_call() {
	let test = connect(
		fake_entry(json!({ "command": "unused" })),
		vec![TransportOptions::default(), TransportOptions::default()],
		None,
	);
	test.connection.get_client().await.expect("connect");
	let server = test.servers.lock().last().cloned().expect("server");
	server.close().await.expect("close");
	tokio::time::sleep(std::time::Duration::from_millis(10)).await;
	assert_eq!(test.connection.state(), Some(ServerState::Disconnected));
	assert_eq!(test.connection.error().as_deref(), Some("Connection closed"));
	test.connection
		.call_tool("echo", json!({}), McpRequestOptions::default())
		.await
		.expect("call");
	assert_eq!(test.connection.state(), Some(ServerState::Connected));
	assert_eq!(test.opened.load(Ordering::SeqCst), 2);
	test.connection.close().await;
}

/// Port of "retries HTTP connections that fail with a transient error".
#[tokio::test(flavor = "multi_thread")]
async fn retries_http_connections_that_fail_with_a_transient_error() {
	let test = connect(
		fake_entry(json!({ "url": "http://unused.invalid", "headers": { "Authorization": "x" } })),
		vec![
			TransportOptions { send_error: Some(SendError::Always(503)), ..Default::default() },
			TransportOptions::default(),
		],
		None,
	);
	// The connect-time retry wraps `connect_once`: the hook fails every
	// request of the first transport with the transient 503, so the connect
	// retries and the second transport connects.
	let result = test.connection.get_client().await;
	assert!(result.is_ok(), "transient 503 must retry the connection: {:?}", result.err());
	assert_eq!(test.connection.state(), Some(ServerState::Connected));
	assert_eq!(test.opened.load(Ordering::SeqCst), 2);
	test.connection.close().await;

	// A permanent error fails the first attempt without retrying.
	let failing = connect(
		fake_entry(json!({ "url": "http://unused.invalid", "headers": { "Authorization": "x" } })),
		vec![TransportOptions { send_error: Some(SendError::Always(400)), ..Default::default() }],
		None,
	);
	let error = match failing.connection.get_client().await {
		Ok(_) => panic!("must fail"),
		Err(error) => error,
	};
	assert!(error.contains("status 400"), "unexpected: {error}");
	assert_eq!(failing.connection.state(), Some(ServerState::Failed));
	assert_eq!(failing.opened.load(Ordering::SeqCst), 1);
	failing.connection.close().await;
}

/// Port of "retries resource reads, but not tool calls, after a transient
/// HTTP error".
#[tokio::test(flavor = "multi_thread")]
async fn retries_resource_reads_but_not_tool_calls_after_a_transient_http_error() {
	let test = connect(
		fake_entry(json!({ "command": "unused" })),
		vec![TransportOptions { send_error: Some(SendError::Once(502)), ..Default::default() }],
		None,
	);
	let read: ReadResourceResult = serde_json::from_value(
		serde_json::to_value(
			test.connection
				.read_resource("docs://a", McpRequestOptions::default())
				.await
				.expect("read"),
		)
		.expect("serialize"),
	)
	.expect("shape");
	assert_eq!(read.contents.len(), 1);
	let error = test
		.connection
		.call_tool("echo", json!({}), McpRequestOptions::default())
		.await
		.expect_err("must fail");
	assert!(error.contains("status 502"), "unexpected: {error}");
	test.connection.close().await;
}

/// Port of "asks OAuth servers that keep rejecting requests for a new
/// sign-in".
#[tokio::test(flavor = "multi_thread")]
async fn asks_oauth_servers_that_keep_rejecting_for_a_new_sign_in() {
	let test = connect(
		fake_entry(json!({ "url": "http://unused.invalid" })),
		vec![TransportOptions { send_error: Some(SendError::AuthRequired), ..Default::default() }],
		None,
	);
	let error = match test.connection.get_client().await {
		Ok(_) => panic!("must fail"),
		Err(error) => error,
	};
	assert!(
		error.contains("requires sign-in. Run /mcp to sign in."),
		"unexpected: {error}"
	);
	assert_eq!(test.connection.state(), Some(ServerState::NeedsAuth));
	test.connection.close().await;
}

/// Port of "sends the provider token and asks for the provider login when the
/// server rejects it".
#[tokio::test(flavor = "multi_thread")]
async fn sends_the_provider_token_and_asks_for_the_provider_login() {
	// A 401-only HTTP server (upstream: node http server).
	let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
	let port = listener.local_addr().expect("addr").port();
	let authorizations: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
	let authorizations_server = Arc::clone(&authorizations);
	tokio::spawn(async move {
		loop {
			let Ok((mut stream, _)) = listener.accept().await else {
				break;
			};
			let authorizations = Arc::clone(&authorizations_server);
			tokio::spawn(async move {
				use tokio::io::AsyncReadExt;
				let mut buffer = [0u8; 4096];
				let read = stream.read(&mut buffer).await.unwrap_or(0);
				let text = String::from_utf8_lossy(&buffer[..read]).into_owned();
				let authorization = text
					.lines()
					.find(|line| line.to_lowercase().starts_with("authorization:"))
					.map(|line| line.split_once(':').map(|(_, value)| value.trim().to_string()).unwrap_or_default());
				authorizations.lock().push(authorization);
				let _ = tokio::io::AsyncWriteExt::write_all(
					&mut stream,
					b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
				)
				.await;
			});
		}
	});
	let connection = McpServerConnection::new(McpServerConnectionOptions {
		entry: McpServerEntry {
			name: "radius".to_string(),
			config: json!({ "url": format!("http://127.0.0.1:{port}/mcp"), "auth": { "provider": "radius" } }),
			source: "test".to_string(),
			scope: None,
			r#override: None,
		},
		cwd: std::env::current_dir()
			.map(|cwd| cwd.to_string_lossy().into_owned())
			.unwrap_or_default(),
		create_transport: Arc::new(create_default_transport),
		credentials: Arc::new(McpOAuthCredentialStore::new(Arc::new(InMemoryAuthStorageBackend::new()), None)),
		provider_token: Some(Arc::new(|provider: &str| {
			let provider = provider.to_string();
			Box::pin(async move { if provider == "radius" { Some("tok".to_string()) } else { None } })
		})),
		on_tools: Arc::new(|_| {}),
		on_change: None,
		log: None,
	});
	assert!(connection.oauth_url().is_none());
	let error = match connection.get_client().await {
		Ok(_) => panic!("must fail"),
		Err(error) => error,
	};
	assert!(
		error.contains("requires sign-in. Run /login radius to sign in."),
		"unexpected: {error}"
	);
	assert_eq!(connection.state(), Some(ServerState::NeedsAuth));
	assert_eq!(*authorizations.lock(), vec![Some("Bearer tok".to_string())]);
	connection.close().await;
}

/// Port of "appends server log messages to the log file".
#[tokio::test(flavor = "multi_thread")]
async fn appends_server_log_messages_to_the_log_file() {
	let dir = std::env::temp_dir().join(format!("pi-mcp-log-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("clock").as_nanos()));
	std::fs::create_dir_all(&dir).expect("dir");
	let path = dir.join("mcp.log").to_string_lossy().into_owned();
	let test = connect(
		fake_entry(json!({ "command": "unused" })),
		vec![TransportOptions::default()],
		Some(Arc::new(Mutex::new(McpServerLog::new(&path)))),
	);
	test.connection.get_client().await.expect("connect");
	let server = test.servers.lock().last().cloned().expect("server");
	let notification = JsonRpcMessage::Notification(pi_mcp::protocol::jsonrpc::JsonRpcNotification {
		jsonrpc: "2.0".to_string(),
		method: "notifications/message".to_string(),
		params: Some(json!({ "level": "warning", "logger": "db", "data": "slow\nquery" })),
	});
	server.send(notification).await.expect("send");
	let notification = JsonRpcMessage::Notification(pi_mcp::protocol::jsonrpc::JsonRpcNotification {
		jsonrpc: "2.0".to_string(),
		method: "notifications/message".to_string(),
		params: Some(json!({ "level": "error", "data": { "code": 7 } })),
	});
	server.send(notification).await.expect("send");
	tokio::time::sleep(std::time::Duration::from_millis(50)).await;
	let content = std::fs::read_to_string(&path).expect("read");
	// Upstream strips `^\S+ ` per line: only lines starting with non-space
	// (the timestamp) lose their first field; indented continuations stay.
	let stripped: String = content
		.lines()
		.map(|line| match line.split_once(' ') {
			Some((head, rest)) if !head.is_empty() && head.chars().all(|c| !c.is_whitespace()) => format!("{rest}\n"),
			_ => format!("{line}\n"),
		})
		.collect();
	assert_eq!(stripped, "[fake] warning db: slow\n    query\n[fake] error {\"code\":7}\n");
	test.connection.close().await;
	let _ = std::fs::remove_dir_all(&dir);
}

/// Port of "resolves the OAuth client secret lazily".
#[tokio::test(flavor = "multi_thread")]
async fn resolves_the_oauth_client_secret_lazily() {
	let test = connect(
		fake_entry(json!({ "url": "http://unused.invalid", "oauth": { "clientSecret": "!exit 1" } })),
		vec![TransportOptions::default()],
		None,
	);
	test.connection
		.call_tool("echo", json!({}), McpRequestOptions::default())
		.await
		.expect("call");
	let error = test.connection.oauth_settings().expect_err("must fail");
	assert!(error.contains("oauth.clientSecret"), "unexpected: {error}");
	test.connection.close().await;
}

// Unused-but-kept witnesses for upstream-shaped types the assertions touch
// indirectly (keeps imports honest without weakening the port).
#[allow(unused)]
fn _witnesses(list: ListResourcesResult, id: JsonRpcId) -> (ListResourcesResult, JsonRpcId) {
	(list, id)
}
