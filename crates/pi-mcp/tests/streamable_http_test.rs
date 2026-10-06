//! Port of `test/streamable-http.test.ts`.

use std::sync::{Arc};
use {parking_lot::Mutex as StdMutex};

use bytes::Bytes;
use futures_util::future::BoxFuture;
use futures_util::StreamExt;

use pi_mcp::auth_provider::{
	AuthProvider, HttpRequest, HttpResponse, HttpHeaders, UnauthorizedContext,
};
use pi_mcp::client::{McpClient, McpClientOptions, McpRequestOptions};
use pi_mcp::protocol::types::LATEST_PROTOCOL_VERSION;
use pi_mcp::transports::streamable_http::{
	consume_sse_stream, ConsumeSseOptions, StreamableHttpReconnectOptions,
	StreamableHttpTransport, StreamableHttpTransportOptions,
};
use pi_mcp::McpError;

mod helpers;
use helpers::{body_from, listen, open_body, read_body, TestHttpServer};

#[derive(Clone)]
struct RecordedRequest {
	method: String,
	headers: HttpHeaders,
	message: Option<serde_json::Value>,
}

type Recorded = Arc<StdMutex<Vec<RecordedRequest>>>;

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
	headers
		.iter()
		.find(|(key, _)| key.eq_ignore_ascii_case(name))
		.map(|(_, value)| value.as_str())
}

fn json_response(value: &serde_json::Value) -> HttpResponse {
	json_response_with(Vec::new(), value)
}

fn json_response_with(headers: HttpHeaders, value: &serde_json::Value) -> HttpResponse {
	let mut headers = headers;
	headers.push(("content-type".to_string(), "application/json".to_string()));
	HttpResponse { status: 200, headers, body: body_from(value.to_string()) }
}

fn sse_response(headers: HttpHeaders, body: pi_mcp::auth_provider::ByteStream) -> HttpResponse {
	let mut headers = headers;
	headers.push(("content-type".to_string(), "text/event-stream".to_string()));
	HttpResponse { status: 200, headers, body }
}

fn tool_result(id: &serde_json::Value, text: &str) -> serde_json::Value {
	serde_json::json!({
		"jsonrpc": "2.0",
		"id": id,
		"result": { "content": [{ "type": "text", "text": text }] },
	})
}

async fn protocol_handler(
	request: HttpRequest,
	requests: Recorded,
	body: Option<serde_json::Value>,
) -> HttpResponse {
	if request.method == "GET" {
		requests
			.lock()
			
			.push(RecordedRequest { method: "GET".to_string(), headers: request.headers.clone(), message: None });
		return HttpResponse { status: 405, headers: Vec::new(), body: body_from(Bytes::new()) };
	}
	if request.method == "DELETE" {
		requests.lock().push(RecordedRequest {
			method: "DELETE".to_string(),
			headers: request.headers.clone(),
			message: None,
		});
		return HttpResponse { status: 200, headers: Vec::new(), body: body_from(Bytes::new()) };
	}
	let message = match body {
		Some(body) => body,
		None => match serde_json::from_str(&read_body(&request).await) {
			Ok(value) => value,
			Err(_) => serde_json::Value::Null,
		},
	};
	requests
		.lock()
		
		.push(RecordedRequest {
			method: request.method.clone(),
			headers: request.headers.clone(),
			message: Some(message.clone()),
		});
	let Some(id) = message.get("id").cloned() else {
		return HttpResponse { status: 202, headers: Vec::new(), body: body_from(Bytes::new()) };
	};
	let method = message.get("method").and_then(serde_json::Value::as_str).unwrap_or_default();
	if method == "initialize" {
		return HttpResponse {
			status: 200,
			headers: vec![
				("content-type".to_string(), "application/json".to_string()),
				("mcp-session-id".to_string(), "session-1".to_string()),
			],
			body: body_from(
				serde_json::json!({
					"jsonrpc": "2.0",
					"id": id,
					"result": {
						"protocolVersion": LATEST_PROTOCOL_VERSION,
						"capabilities": { "tools": {} },
						"serverInfo": { "name": "http-fixture", "version": "1.0.0" },
					},
				})
				.to_string(),
			),
		};
	}
	if method == "tools/list" {
		return json_response(&serde_json::json!({
			"jsonrpc": "2.0",
			"id": id,
			"result": { "tools": [{ "name": "echo", "inputSchema": { "type": "object" } }] },
		}));
	}
	sse_response(
		Vec::new(),
		body_from(format!("id: tool-result\ndata: {}\n\n", tool_result(&id, "hello"))),
	)
}

fn recorded(requests: &Recorded) -> Vec<RecordedRequest> {
	requests.lock().clone()
}

fn transport_options(url: &str) -> StreamableHttpTransportOptions {
	StreamableHttpTransportOptions::new(url::Url::parse(url).expect("valid url"))
}

// Port of streamable-http.test.ts "handles JSON and SSE responses with session and protocol headers".
#[tokio::test]
async fn handles_json_and_sse_responses_with_session_and_protocol_headers() {
	let requests: Recorded = Arc::new(StdMutex::new(Vec::new()));
	let recorded_for_handler = Arc::clone(&requests);
	let server = listen(Arc::new(move |request| {
		let requests = Arc::clone(&recorded_for_handler);
		Box::pin(async move { protocol_handler(request, requests, None).await })
	}))
	.await;
	let url = format!("{}/mcp", server.origin);
	let transport = StreamableHttpTransport::new(transport_options(&url));
	let client = McpClient::new(McpClientOptions::new("http-test", "1.0.0"));
	client.connect(transport.clone()).await.expect("connect");
	assert_eq!(transport.session_id().as_deref(), Some("session-1"));
	let tools = client.list_tools(&Default::default()).await.expect("list tools");
	assert_eq!(
		serde_json::to_value(&tools).unwrap(),
		serde_json::json!([{ "name": "echo", "inputSchema": { "type": "object" } }])
	);
	let mut args = serde_json::Map::new();
	args.insert("text".to_string(), serde_json::json!("hello"));
	let result = client.call_tool("echo", Some(args), &Default::default()).await.expect("call");
	assert_eq!(
		serde_json::to_value(&result).unwrap(),
		serde_json::json!({ "content": [{ "type": "text", "text": "hello" }] })
	);
	client.close().await.expect("close");

	let requests = recorded(&requests);
	let list_request = requests
		.iter()
		.find(|entry| entry.message.as_ref().map(|m| m.get("method").and_then(serde_json::Value::as_str)) == Some(Some("tools/list")))
		.expect("tools/list recorded");
	assert_eq!(header(&list_request.headers, "mcp-session-id"), Some("session-1"));
	assert_eq!(header(&list_request.headers, "mcp-protocol-version"), Some(LATEST_PROTOCOL_VERSION));
	assert!(requests.iter().any(|entry| entry.method == "GET"));
	assert!(requests.iter().any(|entry| entry.method == "DELETE"));
	server.close().await;
}

// Port of streamable-http.test.ts "classifies authentication failures".
#[tokio::test]
async fn classifies_authentication_failures() {
	let server = listen(Arc::new(|request| {
		Box::pin(async move {
			let _ = read_body(&request).await;
			HttpResponse {
				status: 401,
				headers: vec![(
					"www-authenticate".to_string(),
					"Bearer resource_metadata=\"https://example.com/meta\"".to_string(),
				)],
				body: body_from("login required"),
			}
		})
	}))
	.await;
	let url = format!("{}/mcp", server.origin);
	let transport = StreamableHttpTransport::new(transport_options(&url));
	let client = McpClient::new(McpClientOptions::new("http-test", "1.0.0"));
	let error = client.connect(transport).await.expect_err("401");
	match &error {
		McpError::AuthRequired(auth) => {
			assert_eq!(auth.body, "login required");
			assert_eq!(
				auth.www_authenticate.as_deref(),
				Some("Bearer resource_metadata=\"https://example.com/meta\"")
			);
			assert_eq!(auth.to_string(), "MCP server requires authentication");
		}
		other => panic!("expected auth-required error, got {other:?}"),
	}
	server.close().await;
}

// Port of streamable-http.test.ts "fails only the request whose SSE stream breaks".
#[tokio::test]
async fn fails_only_the_request_whose_sse_stream_breaks() {
	let requests: Recorded = Arc::new(StdMutex::new(Vec::new()));
	let recorded_for_handler = Arc::clone(&requests);
	let release = Arc::new(tokio::sync::Notify::new());
	let release_for_handler = Arc::clone(&release);
	let server = listen(Arc::new(move |request| {
		let requests = Arc::clone(&recorded_for_handler);
		let release = Arc::clone(&release_for_handler);
		Box::pin(async move {
			if request.method != "POST" {
				return protocol_handler(request, requests, None).await;
			}
			let message: serde_json::Value =
				serde_json::from_str(&read_body(&request).await).unwrap_or(serde_json::Value::Null);
			let name = message
				.pointer("/params/name")
				.and_then(serde_json::Value::as_str)
				.unwrap_or_default()
				.to_string();
			if name == "broken" {
				return sse_response(Vec::new(), body_from("data: not json\n\n"));
			}
			if name == "slow" {
				release.notified().await;
			}
			protocol_handler(request, requests, Some(message)).await
		})
	}))
	.await;
	let url = format!("{}/mcp", server.origin);
	let transport = {
		let mut options = transport_options(&url);
		options.open_get_stream = Some(false);
		StreamableHttpTransport::new(options)
		};
	let client = McpClient::new(McpClientOptions::new("http-test", "1.0.0"));
	let errors: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
	let error_sink = Arc::clone(&errors);
	client.on_error(Arc::new(move |error: &McpError| {
		error_sink.lock().push(error.to_string());
	}));
	client.connect(transport).await.expect("connect");

	let slow_options = McpRequestOptions::default();
	let slow = client.call_tool("slow", None, &slow_options);
	let error = client
		.call_tool("broken", None, &Default::default())
		.await
		.expect_err("broken stream");
	assert!(error.to_string().starts_with("MCP response stream failed"), "{error}");
	release.notify_one();
	let result = slow.await.expect("slow call");
	assert_eq!(
		serde_json::to_value(&result).unwrap(),
		serde_json::json!({ "content": [{ "type": "text", "text": "hello" }] })
	);
	assert_eq!(errors.lock().len(), 1);
	client.close().await.expect("close");
	server.close().await;
}

// Port of streamable-http.test.ts
// "opens the GET stream after initialization and sends Last-Event-ID only when resuming".
#[tokio::test]
async fn opens_the_get_stream_after_initialization_and_sends_last_event_id_only_when_resuming() {
	let requests: Recorded = Arc::new(StdMutex::new(Vec::new()));
	let order: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
	let recorded_for_handler = Arc::clone(&requests);
	let order_for_handler = Arc::clone(&order);
	let server = listen(Arc::new(move |request| {
		let requests = Arc::clone(&recorded_for_handler);
		let order = Arc::clone(&order_for_handler);
		Box::pin(async move {
			if request.method == "POST" {
				let message: serde_json::Value =
					serde_json::from_str(&read_body(&request).await).unwrap_or(serde_json::Value::Null);
				order
					.lock()
					
					.push(
						message
							.get("method")
							.and_then(serde_json::Value::as_str)
							.unwrap_or_default()
							.to_string(),
					);
				return protocol_handler(request, requests, Some(message)).await;
			}
			order
				.lock()
				
				.push(request.method.clone());
			protocol_handler(request, requests, None).await
		})
	}))
	.await;
	let url = format!("{}/mcp", server.origin);
	let transport = StreamableHttpTransport::new(transport_options(&url));
	let client = McpClient::new(McpClientOptions::new("http-test", "1.0.0"));
	client.connect(transport).await.expect("connect");
	let _ = client.call_tool("echo", None, &Default::default()).await.expect("call");
	let _ = client.list_tools(&Default::default()).await.expect("list");
	client.close().await.expect("close");
	let order = order.lock().clone();
	let get_position = order.iter().position(|entry| entry == "GET").expect("GET observed");
	let initialized_position =
		order.iter().position(|entry| entry == "notifications/initialized").expect("initialized observed");
	assert!(get_position > initialized_position);
	assert!(recorded(&requests)
		.iter()
		.all(|entry| header(&entry.headers, "last-event-id").is_none()));
	server.close().await;
}

// Port of streamable-http.test.ts "resumes a response stream the server closed before answering".
#[tokio::test]
async fn resumes_a_response_stream_the_server_closed_before_answering() {
	let requests: Recorded = Arc::new(StdMutex::new(Vec::new()));
	let recorded_for_handler = Arc::clone(&requests);
	let resume_headers: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
	let resume_sink = Arc::clone(&resume_headers);
	let server = listen(Arc::new(move |request| {
		let requests = Arc::clone(&recorded_for_handler);
		let resume_sink = Arc::clone(&resume_sink);
		Box::pin(async move {
			if request.method == "GET" {
				if let Some(last_event_id) = header(&request.headers, "last-event-id") {
					resume_sink
						.lock()
						
						.push(last_event_id.to_string());
					return sse_response(
						Vec::new(),
						body_from(format!(
							"id: 2\ndata: {}\n\n",
							tool_result(&serde_json::json!(2), "resumed")
						)),
					);
				}
				return protocol_handler(request, requests, None).await;
			}
			if request.method != "POST" {
				return protocol_handler(request, requests, None).await;
			}
			let message: serde_json::Value =
				serde_json::from_str(&read_body(&request).await).unwrap_or(serde_json::Value::Null);
			if message.get("method").and_then(serde_json::Value::as_str) != Some("tools/call") {
				return protocol_handler(request, requests, Some(message)).await;
			}
			// Priming event (ID, no data) and a retry hint, then the server drops the stream.
			sse_response(Vec::new(), body_from("id: 1\nretry: 5\ndata:\n\n"))
		})
	}))
	.await;
	let url = format!("{}/mcp", server.origin);
	let transport = {
		let mut options = transport_options(&url);
		options.open_get_stream = Some(false);
		StreamableHttpTransport::new(options)
		};
	let client = McpClient::new(McpClientOptions::new("http-test", "1.0.0"));
	let errors: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
	let error_sink = Arc::clone(&errors);
	client.on_error(Arc::new(move |error: &McpError| {
		error_sink.lock().push(error.to_string());
	}));
	client.connect(transport).await.expect("connect");
	let result = client.call_tool("echo", None, &Default::default()).await.expect("resumed call");
	assert_eq!(
		serde_json::to_value(&result).unwrap(),
		serde_json::json!({ "content": [{ "type": "text", "text": "resumed" }] })
	);
	assert_eq!(
		resume_headers.lock().clone(),
		vec!["1".to_string()]
	);
	assert!(errors.lock().is_empty());
	client.close().await.expect("close");
	server.close().await;
}

// Port of streamable-http.test.ts "fails a request whose response stream ends without an answer".
#[tokio::test]
async fn fails_a_request_whose_response_stream_ends_without_an_answer() {
	let requests: Recorded = Arc::new(StdMutex::new(Vec::new()));
	let recorded_for_handler = Arc::clone(&requests);
	let server = listen(Arc::new(move |request| {
		let requests = Arc::clone(&recorded_for_handler);
		Box::pin(async move {
			if request.method != "POST" {
				return protocol_handler(request, requests, None).await;
			}
			let message: serde_json::Value =
				serde_json::from_str(&read_body(&request).await).unwrap_or(serde_json::Value::Null);
			if message.get("method").and_then(serde_json::Value::as_str) != Some("tools/call") {
				return protocol_handler(request, requests, Some(message)).await;
			}
			sse_response(Vec::new(), body_from(": nothing here\n\n"))
		})
	}))
	.await;
	let url = format!("{}/mcp", server.origin);
	let transport = {
		let mut options = transport_options(&url);
		options.open_get_stream = Some(false);
		StreamableHttpTransport::new(options)
		};
	let client = McpClient::new(McpClientOptions::new("http-test", "1.0.0"));
	client.connect(transport).await.expect("connect");
	let error = {
		let options = McpRequestOptions::default().timeout_ms(5_000);
		client.call_tool("echo", None, &options)
			.await
			.expect_err("stream ended")
	};
	assert_eq!(
		error.to_string(),
		"MCP response stream failed: stream ended without a response"
	);
	client.close().await.expect("close");
	server.close().await;
}

// Port of streamable-http.test.ts "reconnects the GET stream after it drops".
#[tokio::test]
async fn reconnects_the_get_stream_after_it_drops() {
	let requests: Recorded = Arc::new(StdMutex::new(Vec::new()));
	let recorded_for_handler = Arc::clone(&requests);
	let gets = Arc::new(StdMutex::new(0usize));
	let gets_for_handler = Arc::clone(&gets);
	let notification = serde_json::json!({ "jsonrpc": "2.0", "method": "notifications/tools/list_changed" });
	let notification_for_handler = notification.clone();
	let server = listen(Arc::new(move |request| {
		let requests = Arc::clone(&recorded_for_handler);
		let gets = Arc::clone(&gets_for_handler);
		let notification = notification_for_handler.clone();
		Box::pin(async move {
			if request.method != "GET" {
				return protocol_handler(request, requests, None).await;
			}
			requests
				.lock()
				
				.push(RecordedRequest {
					method: "GET".to_string(),
					headers: request.headers.clone(),
					message: None,
				});
			let count = {
				let mut gets = gets.lock();
				*gets += 1;
				*gets
			};
			if count == 1 {
				return sse_response(
					Vec::new(),
					body_from(format!("id: g1\ndata: {notification}\n\n")),
				);
			}
			let (tx, body) = open_body();
			let _ = tx
				.send(Ok(Bytes::from(format!("id: g2\ndata: {notification}\n\n"))))
				.await;
			// The stream stays open until the sender drops at test end.
			tokio::spawn(async move {
				tokio::time::sleep(std::time::Duration::from_secs(30)).await;
				drop(tx);
			});
			sse_response(Vec::new(), body)
		})
	}))
	.await;
	let url = format!("{}/mcp", server.origin);
	let transport = {
	let mut options = transport_options(&url);
	options.reconnect = Some(StreamableHttpReconnectOptions { initial_delay_ms: Some(1), ..Default::default() });
	StreamableHttpTransport::new(options)
};
	let client = McpClient::new(McpClientOptions::new("http-test", "1.0.0"));
	let changes = Arc::new(StdMutex::new(0usize));
	let (second_change_tx, second_change_rx) = tokio::sync::oneshot::channel::<()>();
	let second_change_tx = Arc::new(StdMutex::new(Some(second_change_tx)));
	let changes_sink = Arc::clone(&changes);
	let second_change_flag = Arc::clone(&second_change_tx);
	client.on_notification(
		"notifications/tools/list_changed",
		Arc::new(move |_params| {
			let mut changes = changes_sink.lock();
			*changes += 1;
			if *changes == 2 {
				if let Some(tx) = second_change_flag.lock().take() {
					let _ = tx.send(());
				}
			}
		}),
	);
	client.connect(transport).await.expect("connect");
	let _ = tokio::time::timeout(std::time::Duration::from_secs(10), second_change_rx).await;
	assert_eq!(*changes.lock(), 2);
	let requests = recorded(&requests);
	let last_event_ids: Vec<Option<&str>> = requests
		.iter()
		.filter(|entry| entry.method == "GET")
		.map(|entry| header(&entry.headers, "last-event-id"))
		.collect();
	assert_eq!(last_event_ids, vec![None, Some("g1")]);
	client.close().await.expect("close");
	server.close().await;
}

// Port of streamable-http.test.ts "rejects a request the server accepts without a response".
#[tokio::test]
async fn rejects_a_request_the_server_accepts_without_a_response() {
	let requests: Recorded = Arc::new(StdMutex::new(Vec::new()));
	let recorded_for_handler = Arc::clone(&requests);
	let server = listen(Arc::new(move |request| {
		let requests = Arc::clone(&recorded_for_handler);
		Box::pin(async move {
			if request.method != "POST" {
				return protocol_handler(request, requests, None).await;
			}
			let message: serde_json::Value =
				serde_json::from_str(&read_body(&request).await).unwrap_or(serde_json::Value::Null);
			if message.get("method").and_then(serde_json::Value::as_str) != Some("tools/call") {
				return protocol_handler(request, requests, Some(message)).await;
			}
			HttpResponse { status: 202, headers: Vec::new(), body: body_from(Bytes::new()) }
		})
	}))
	.await;
	let url = format!("{}/mcp", server.origin);
	let transport = {
		let mut options = transport_options(&url);
		options.open_get_stream = Some(false);
		StreamableHttpTransport::new(options)
		};
	let client = McpClient::new(McpClientOptions::new("http-test", "1.0.0"));
	client.connect(transport).await.expect("connect");
	let error = client.call_tool("echo", None, &Default::default()).await.expect_err("202 without reply");
	assert!(error.to_string().contains("without a response"), "{error}");
	client.close().await.expect("close");
	server.close().await;
}

// Port of streamable-http.test.ts "includes the response body in HTTP errors".
#[tokio::test]
async fn includes_the_response_body_in_http_errors() {
	let server = listen(Arc::new(|request| {
		Box::pin(async move {
			let _ = read_body(&request).await;
			HttpResponse { status: 400, headers: Vec::new(), body: body_from("Invalid Accept header") }
		})
	}))
	.await;
	let url = format!("{}/mcp", server.origin);
	let transport = StreamableHttpTransport::new(transport_options(&url));
	let client = McpClient::new(McpClientOptions::new("http-test", "1.0.0"));
	let error = client.connect(transport).await.expect_err("400");
	assert_eq!(
		error.to_string(),
		"MCP HTTP request failed with status 400: Invalid Accept header"
	);
	server.close().await;
}

// Port of streamable-http.test.ts
// "hands 401 and insufficient-scope 403 responses to the auth provider with the rejected token".
#[tokio::test]
async fn hands_401_and_insufficient_scope_403_responses_to_the_auth_provider() {
	let requests: Recorded = Arc::new(StdMutex::new(Vec::new()));
	let recorded_for_handler = Arc::clone(&requests);
	let token = Arc::new(StdMutex::new("old".to_string()));
	let token_for_handler = Arc::clone(&token);
	let server = listen(Arc::new(move |request| {
		let requests = Arc::clone(&recorded_for_handler);
		let token = Arc::clone(&token_for_handler);
		Box::pin(async move {
			if request.method != "POST" {
				return protocol_handler(request, requests, None).await;
			}
			let message: serde_json::Value =
				serde_json::from_str(&read_body(&request).await).unwrap_or(serde_json::Value::Null);
			let authorization = header(&request.headers, "authorization").unwrap_or_default();
			if authorization == "Bearer old" {
				return HttpResponse {
					status: 401,
					headers: vec![("www-authenticate".to_string(), "Bearer".to_string())],
					body: body_from("Unauthorized"),
				};
			}
			if message.get("method").and_then(serde_json::Value::as_str) == Some("tools/call")
				&& authorization == "Bearer new"
			{
				return HttpResponse {
					status: 403,
					headers: vec![(
						"www-authenticate".to_string(),
						"Bearer error=\"insufficient_scope\", scope=\"admin\"".to_string(),
					)],
					body: body_from(Bytes::new()),
				};
			}
			protocol_handler(request, requests, Some(message)).await
		})
	}))
	.await;
	*token.lock() = "old".to_string();
	let seen: Arc<StdMutex<Vec<(u16, Option<String>)>>> = Arc::new(StdMutex::new(Vec::new()));
	let seen_sink = Arc::clone(&seen);
	let token_for_provider = Arc::clone(&token);
	let provider: Arc<dyn AuthProvider> = Arc::new(RecordingAuthProvider {
		token_for_provider,
		seen_sink,
	});
	let url = format!("{}/mcp", server.origin);
	let transport = {
		let mut options = transport_options(&url);
		options.open_get_stream = Some(false);
		options.auth_provider = Some(provider);
		StreamableHttpTransport::new(options)
		};
	let client = McpClient::new(McpClientOptions::new("http-test", "1.0.0"));
	client.connect(transport).await.expect("connect");
	let result = client.call_tool("echo", None, &Default::default()).await.expect("call");
	assert_eq!(
		serde_json::to_value(&result).unwrap(),
		serde_json::json!({ "content": [{ "type": "text", "text": "hello" }] })
	);
	assert_eq!(
		seen.lock().clone(),
		vec![(401u16, Some("old".to_string())), (403u16, Some("new".to_string()))]
	);
	client.close().await.expect("close");
	server.close().await;
}

#[derive(Clone)]
struct RecordingAuthProvider {
	token_for_provider: Arc<StdMutex<String>>,
	seen_sink: Arc<StdMutex<Vec<(u16, Option<String>)>>>,
}

#[async_trait::async_trait]
impl AuthProvider for RecordingAuthProvider {
	async fn token(&self) -> Result<Option<String>, McpError> {
		Ok(Some(self.token_for_provider.lock().clone()))
	}

	fn handles_unauthorized(&self) -> bool {
		true
	}

	async fn on_unauthorized(&self, context: UnauthorizedContext) -> Result<(), McpError> {
		self.seen_sink
			.lock()
			
			.push((context.response.status, context.token.clone()));
		if context.response.status == 401 {
			*self.token_for_provider.lock() = "new".to_string();
		} else {
			*self.token_for_provider.lock() = "admin".to_string();
		}
		Ok(())
	}
}

// Port of streamable-http.test.ts "calls fetch without a receiver" (#10188). The
// Cloudflare receiver semantics have no Rust equivalent; the injected-fetch and
// `context.fetch` assertions still apply.
#[tokio::test]
async fn uses_the_injected_fetch_and_the_challenge_context_fetch() {
	let requests: Recorded = Arc::new(StdMutex::new(Vec::new()));
	let recorded_for_handler = Arc::clone(&requests);
	let server = listen(Arc::new(move |request| {
		let requests = Arc::clone(&recorded_for_handler);
		Box::pin(async move {
			if request.method == "POST" && header(&request.headers, "authorization").is_none() {
				let _ = read_body(&request).await;
				return HttpResponse {
					status: 401,
					headers: vec![("www-authenticate".to_string(), "Bearer".to_string())],
					body: body_from(Bytes::new()),
				};
			}
			protocol_handler(request, requests, None).await
		})
	}))
	.await;
	let url = format!("{}/mcp", server.origin);
	let connect_with = |fetch: Option<pi_mcp::auth_provider::McpFetch>| {
		let url = url.clone();
		async move {
			let token = Arc::new(StdMutex::new(None::<String>));
			let token_for_provider = Arc::clone(&token);
			let provider: Arc<dyn AuthProvider> = Arc::new(ContextFetchProvider { token_for_provider });
			let transport = {
				let mut options = transport_options(&url);
				options.fetch = fetch;
				options.open_get_stream = Some(false);
				options.auth_provider = Some(provider);
				StreamableHttpTransport::new(options)
				};
			let client = McpClient::new(McpClientOptions::new("http-test", "1.0.0"));
			client.connect(transport).await.expect("connect");
			let tools = client.list_tools(&Default::default()).await.expect("list tools");
			assert_eq!(
				serde_json::to_value(&tools).unwrap(),
				serde_json::json!([{ "name": "echo", "inputSchema": { "type": "object" } }])
			);
			client.close().await.expect("close");
		}
	};

	let default_fetch = pi_mcp::auth_provider::default_fetch();
	let injected: pi_mcp::auth_provider::McpFetch = {
		let default_fetch = default_fetch.clone();
		Arc::new(move |request| {
			let default_fetch = Arc::clone(&default_fetch);
			Box::pin(async move { default_fetch(request).await })
		})
	};
	connect_with(Some(injected)).await;
	connect_with(None).await;
	server.close().await;
}

struct ContextFetchProvider {
	token_for_provider: Arc<StdMutex<Option<String>>>,
}

#[async_trait::async_trait]
impl AuthProvider for ContextFetchProvider {
	async fn token(&self) -> Result<Option<String>, McpError> {
		Ok(self.token_for_provider.lock().clone())
	}

	fn handles_unauthorized(&self) -> bool {
		true
	}

	async fn on_unauthorized(&self, context: UnauthorizedContext) -> Result<(), McpError> {
		// Exercise `context.fetch` exactly like upstream.
		let response = (context.fetch)(HttpRequest {
			method: "GET".to_string(),
			url: context.server_url.clone(),
			headers: Vec::new(),
			body: None,
		})
		.await
		.expect("context fetch");
		drop(response);
		*self.token_for_provider.lock() = Some("token".to_string());
		Ok(())
	}
}

// Port of streamable-http.test.ts "classifies an expired established session".
#[tokio::test]
async fn classifies_an_expired_established_session() {
	let requests: Recorded = Arc::new(StdMutex::new(Vec::new()));
	let recorded_for_handler = Arc::clone(&requests);
	let posts = Arc::new(StdMutex::new(0usize));
	let posts_for_handler = Arc::clone(&posts);
	let server = listen(Arc::new(move |request| {
		let requests = Arc::clone(&recorded_for_handler);
		let posts = Arc::clone(&posts_for_handler);
		Box::pin(async move {
			if request.method == "POST" {
				let expired = {
					let mut count = posts.lock();
					*count += 1;
					*count >= 3
				};
				if expired {
					let _ = read_body(&request).await;
					return HttpResponse { status: 404, headers: Vec::new(), body: body_from("gone") };
				}
			}
			protocol_handler(request, requests, None).await
		})
	}))
	.await;
	let url = format!("{}/mcp", server.origin);
	let transport = {
		let mut options = transport_options(&url);
		options.open_get_stream = Some(false);
		StreamableHttpTransport::new(options)
		};
	let client = McpClient::new(McpClientOptions::new("http-test", "1.0.0"));
	client.connect(transport).await.expect("connect");
	let error = client.list_tools(&Default::default()).await.expect_err("expired");
	assert!(matches!(error, McpError::SessionExpired(_)), "{error:?}");
	assert_eq!(error.to_string(), "MCP session expired");
	client.close().await.expect("close");
	server.close().await;
}

// Ports of streamable-http.test.ts consumeSseStream cases.

// "parses chunked CRLF events, comments, IDs, and multiline data".
#[tokio::test]
async fn parses_chunked_crlf_events_comments_ids_and_multiline_data() {
	let chunks = [
		": keepalive\r\nid: 7\r\ndata: {\"one\":\r\n",
		"data: 1}\r\n\r\n",
	];
	let stream: pi_mcp::auth_provider::ByteStream = Box::pin(futures_util::stream::iter(
		chunks.into_iter().map(|chunk| Ok::<Bytes, McpError>(Bytes::from(chunk))),
	));
	let events: Arc<StdMutex<Vec<(Option<String>, String, Option<String>)>>> =
		Arc::new(StdMutex::new(Vec::new()));
	let sink = Arc::clone(&events);
	consume_sse_stream(
		stream,
		ConsumeSseOptions {
			max_event_bytes: None,
			on_event: Box::new(move |event| {
				sink
					.lock()
					
					.push((event.event, event.data, event.id));
			}),
			on_id: None,
			on_retry: None,
		},
	)
	.await
	.expect("parse");
	assert_eq!(
		events.lock().clone(),
		vec![(None, "{\"one\":\n1}".to_string(), Some("7".to_string()))]
	);
}

// "rejects events whose data lines exceed the limit without a blank line".
#[tokio::test]
async fn rejects_events_whose_data_lines_exceed_the_limit_without_a_blank_line() {
	let sent = Arc::new(StdMutex::new(0usize));
	let sent_for_stream = Arc::clone(&sent);
	// Pull-based stream like upstream's ReadableStream `pull`.
	let stream: pi_mcp::auth_provider::ByteStream = Box::pin(futures_util::stream::unfold(
		(sent_for_stream, 0usize),
		|(sent, count)| async move {
			{
				let mut sent = sent.lock();
				*sent += 1;
			}
			if count > 1000 {
				None
			} else {
				Some((Ok::<Bytes, McpError>(Bytes::from("data: xxxxxxxxxxxxxxxx\n")), (sent, count + 1)))
			}
		},
	));
	let error = consume_sse_stream(
		stream,
		ConsumeSseOptions {
			max_event_bytes: Some(256),
			on_event: Box::new(|_| {}),
			on_id: None,
			on_retry: None,
		},
	)
	.await
	.expect_err("over-limit");
	assert_eq!(error.to_string(), "MCP SSE event exceeds 256 bytes");
	assert!(*sent.lock() < 100);
}
