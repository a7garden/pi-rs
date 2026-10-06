//! Port of `packages/mcp/src/transports/streamable-http.ts`.

use std::sync::{Arc};
use {parking_lot::Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::StreamExt;
use thiserror::Error;
use url::Url;

use crate::auth_provider::{
	default_fetch, discard, header_find, header_set, AuthProvider, ByteStream, HttpRequest,
	HttpResponse, HttpHeaders, McpFetch, UnauthorizedContext, UnauthorizedResponse,
};
use crate::protocol::jsonrpc::{
	parse_json_rpc_message, JsonRpcErrorResponse, JsonRpcErrorObject, JsonRpcId, JsonRpcMessage,
	JsonRpcResponse, McpError, JSON_RPC_ERROR_CODES_INTERNAL_ERROR, JSONRPC_VERSION,
};
use crate::transports::transport::{
	CloseListener, ErrorListener, McpTransport, MessageListener, TransportEvents, Unsubscribe,
	DEFAULT_MAX_MESSAGE_BYTES,
};

const MAX_ERROR_BODY_BYTES: usize = 8 * 1024;
const ERROR_MESSAGE_BODY_CHARS: usize = 500;
const DEFAULT_RECONNECT_INITIAL_DELAY_MS: u64 = 1_000;
const DEFAULT_RECONNECT_MAX_DELAY_MS: u64 = 30_000;
const DEFAULT_RECONNECT_MAX_RETRIES: u32 = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
	pub event: Option<String>,
	pub data: String,
	pub id: Option<String>,
}

/// Upstream `McpHttpError`.
#[derive(Debug, Clone, Error)]
#[error("{message}")]
pub struct HttpError {
	pub status: u16,
	pub message: String,
	pub body: String,
}

/// Upstream `McpAuthRequiredError`.
#[derive(Debug, Clone, Error)]
#[error("MCP server requires authentication")]
pub struct AuthRequiredError {
	pub www_authenticate: Option<String>,
	pub body: String,
}

/// Upstream `McpSessionExpiredError`.
#[derive(Debug, Clone, Error)]
#[error("MCP session expired")]
pub struct SessionExpiredError {
	pub body: String,
}

pub struct ConsumeSseOptions<'a> {
	pub max_event_bytes: Option<usize>,
	pub on_event: Box<dyn FnMut(SseEvent) + Send + 'a>,
	/// Called for every `id` field, including events without data (for example resumption priming events).
	pub on_id: Option<Box<dyn FnMut(&str) + Send + 'a>>,
	/// Called for every valid `retry` field, in milliseconds.
	pub on_retry: Option<Box<dyn FnMut(u64) + Send + 'a>>,
}

/// Port of upstream `consumeSseStream`: parse an SSE byte stream into events.
/// Deviation: upstream decodes incrementally with `TextDecoder`; here bytes are
/// split on `\n` first (a newline byte never occurs inside a multi-byte UTF-8
/// sequence) and each line is decoded as UTF-8 lossily, which is equivalent for
/// well-formed streams.
pub async fn consume_sse_stream(
	stream: ByteStream,
	options: ConsumeSseOptions<'_>,
) -> Result<(), McpError> {
	let ConsumeSseOptions { max_event_bytes, mut on_event, mut on_id, mut on_retry } = options;
	let max_event_bytes = max_event_bytes.unwrap_or(DEFAULT_MAX_MESSAGE_BYTES);
	let mut body = stream;

	let mut buffered: Vec<u8> = Vec::new();
	let mut event_name: Option<String> = None;
	let mut event_id: Option<String> = None;
	let mut data_lines: Vec<String> = Vec::new();
	// Bytes of the pending event's data, including the "\n" joins, so events streamed as many
	// short `data:` lines without a terminating blank line cannot grow without bound.
	let mut data_bytes = 0usize;
	while let Some(chunk) = body.next().await {
		let chunk: Bytes = chunk?;
		buffered.extend_from_slice(&chunk);
		while let Some(newline) = buffered.iter().position(|b| *b == b'\n') {
			let line: Vec<u8> = buffered.drain(..newline).collect();
			buffered.drain(..1); // the newline
			process_line(
				&line,
				max_event_bytes,
				&mut on_event,
				&mut on_id,
				&mut on_retry,
				&mut event_name,
				&mut event_id,
				&mut data_lines,
				&mut data_bytes,
			)?;
		}
		if buffered.len() > max_event_bytes {
			return Err(McpError::other(format!("MCP SSE event exceeds {max_event_bytes} bytes")));
		}
	}
	// End of stream: process any trailing line, then dispatch a final pending event
	// (upstream flushes the `TextDecoder` and calls `dispatch()`).
	if !buffered.is_empty() {
		let tail = std::mem::take(&mut buffered);
		process_line(
			&tail,
			max_event_bytes,
			&mut on_event,
			&mut on_id,
			&mut on_retry,
			&mut event_name,
			&mut event_id,
			&mut data_lines,
			&mut data_bytes,
		)?;
	}
	dispatch(
		&mut on_event,
		&mut event_name,
		&mut event_id,
		&mut data_lines,
		&mut data_bytes,
	);
	Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_line<'a>(
	raw_line: &[u8],
	max_event_bytes: usize,
	on_event: &mut (dyn FnMut(SseEvent) + Send),
	on_id: &mut Option<Box<dyn FnMut(&str) + Send + 'a>>,
	on_retry: &mut Option<Box<dyn FnMut(u64) + Send + 'a>>,
	event_name: &mut Option<String>,
	event_id: &mut Option<String>,
	data_lines: &mut Vec<String>,
	data_bytes: &mut usize,
) -> Result<(), McpError> {
	let line: &[u8] = if raw_line.last() == Some(&b'\r') { &raw_line[..raw_line.len() - 1] } else { raw_line };
	if line.is_empty() {
		dispatch(on_event, event_name, event_id, data_lines, data_bytes);
		return Ok(());
	}
	if line[0] == b':' {
		return Ok(());
	}
	let colon = line.iter().position(|b| *b == b':');
	let (field, mut value) = match colon {
		Some(index) => (&line[..index], &line[index + 1..]),
		None => (line, &line[..0]),
	};
	if value.first() == Some(&b' ') {
		value = &value[1..];
	}
	if field == b"data" {
		*data_bytes += value.len() + usize::from(!data_lines.is_empty());
		if *data_bytes > max_event_bytes {
			return Err(McpError::other(format!("MCP SSE event exceeds {max_event_bytes} bytes")));
		}
		data_lines.push(String::from_utf8_lossy(value).into_owned());
	} else if field == b"event" {
		*event_name = Some(String::from_utf8_lossy(value).into_owned());
	} else if field == b"id" {
		let text = String::from_utf8_lossy(value).into_owned();
		if !text.contains('\0') {
			*event_id = Some(text.clone());
			if let Some(on_id) = on_id.as_deref_mut() {
				on_id(&text);
			}
		}
	} else if field == b"retry" {
		let text = String::from_utf8_lossy(value);
		if !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()) {
			if let Some(on_retry) = on_retry.as_deref_mut() {
				on_retry(text.parse::<u64>().unwrap_or(u64::MAX));
			}
		}
	}
	Ok(())
}

fn dispatch(
	on_event: &mut (dyn FnMut(SseEvent) + Send),
	event_name: &mut Option<String>,
	event_id: &mut Option<String>,
	data_lines: &mut Vec<String>,
	data_bytes: &mut usize,
) {
	if data_lines.is_empty() {
		*event_name = None;
		*event_id = None;
		return;
	}
	let data = data_lines.join("\n");
	on_event(SseEvent {
		event: event_name.take(),
		data,
		id: event_id.take(),
	});
	*data_lines = Vec::new();
	*data_bytes = 0;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StreamableHttpReconnectOptions {
	/// Delay before the first reconnection attempt, unless the server sent a `retry` field. Default: 1000.
	pub initial_delay_ms: Option<u64>,
	/// Upper bound for the exponential backoff. Default: 30000.
	pub max_delay_ms: Option<u64>,
	/// Consecutive failed attempts before giving up on a stream. Default: 5.
	pub max_retries: Option<u32>,
}

#[derive(Clone)]
pub struct StreamableHttpTransportOptions {
	pub url: Url,
	pub headers: Option<HttpHeaders>,
	pub fetch: Option<McpFetch>,
	/// Open the server-to-client GET stream after initialization. Default: true.
	pub open_get_stream: Option<bool>,
	pub max_message_bytes: Option<usize>,
	pub auth_provider: Option<Arc<dyn AuthProvider>>,
	/// Reconnection of dropped SSE streams (the GET stream, and response streams that carry event IDs).
	pub reconnect: Option<StreamableHttpReconnectOptions>,
}

impl StreamableHttpTransportOptions {
	/// Options for `url` with every optional field defaulted (upstream object literal).
	pub fn new(url: Url) -> StreamableHttpTransportOptions {
		StreamableHttpTransportOptions {
			url,
			headers: None,
			fetch: None,
			open_get_stream: None,
			max_message_bytes: None,
			auth_provider: None,
			reconnect: None,
		}
	}
}

struct HttpState {
	started: bool,
	closed: bool,
	session_id: Option<String>,
	protocol_version: Option<String>,
	get_stream_started: bool,
}

/// Upstream `StreamCursor`.
#[derive(Default, Clone)]
struct StreamCursor {
	last_event_id: Option<String>,
	retry_ms: Option<u64>,
	/// Whether the stream delivered any event since it was (re)opened.
	received: bool,
}

/// Upstream `StreamableHttpTransport`. Construct through [`StreamableHttpTransport::new`],
/// which returns an `Arc` (the transport spawns stream tasks referencing itself).
pub struct StreamableHttpTransport {
	pub url: Url,
	pub options: StreamableHttpTransportOptions,
	fetch: McpFetch,
	events: TransportEvents,
	state: Mutex<HttpState>,
	closed_notify: Arc<tokio::sync::Notify>,
	me: Mutex<std::sync::Weak<StreamableHttpTransport>>,
}

impl StreamableHttpTransport {
	pub fn new(options: StreamableHttpTransportOptions) -> Arc<StreamableHttpTransport> {
		let url = options.url.clone();
		let fetch = options.fetch.clone().unwrap_or_else(default_fetch);
		let transport = Arc::new(StreamableHttpTransport {
			url,
			fetch,
			events: TransportEvents::new(),
			state: Mutex::new(HttpState {
				started: false,
				closed: false,
				session_id: None,
				protocol_version: None,
				get_stream_started: false,
			}),
			closed_notify: Arc::new(tokio::sync::Notify::new()),
			me: Mutex::new(std::sync::Weak::new()),
			options,
		});
		*transport.me.lock() = Arc::downgrade(&transport);
		transport
	}

	pub fn session_id(&self) -> Option<String> {
		self.lock().session_id.clone()
	}

	fn self_arc(&self) -> Option<Arc<StreamableHttpTransport>> {
		self.me
			.lock()
			
			.upgrade()
	}

	fn lock(&self) -> parking_lot::MutexGuard<'_, HttpState> {
		self.state.lock()
	}

	fn closed(&self) -> bool {
		self.lock().closed
	}

	/// Resolves when `close` runs (upstream `AbortController` signal).
	async fn closed_wait(&self) {
		if self.closed() {
			return;
		}
		let mut notified = std::pin::pin!(self.closed_notify.notified());
		// Register interest before re-checking, so a concurrent `close` cannot slip through.
		notified.as_mut().enable();
		if self.closed() {
			return;
		}
		notified.await;
	}

	/// Resolves `false` when the transport closed while waiting (upstream `sleep`).
	async fn sleep(&self, ms: u64) -> bool {
		tokio::select! {
			() = tokio::time::sleep(Duration::from_millis(ms)) => true,
			() = self.closed_wait() => false,
		}
	}

	fn capture_session(&self, response: &HttpResponse) {
		if let Some(session_id) = header_find(&response.headers, "mcp-session-id") {
			self.lock().session_id = Some(session_id.to_string());
		}
	}

	async fn check_response(&self, response: HttpResponse) -> Result<HttpResponse, McpError> {
		if (200..300).contains(&response.status) {
			return Ok(response);
		}
		let body = read_body_text(response.body).await;
		if response.status == 401 {
			return Err(McpError::AuthRequired(AuthRequiredError {
				www_authenticate: header_find(&response.headers, "www-authenticate").map(str::to_string),
				body,
			}));
		}
		if response.status == 404 && self.lock().session_id.is_some() {
			return Err(McpError::SessionExpired(SessionExpiredError { body }));
		}
		Err(McpError::Http(HttpError {
			status: response.status,
			message: describe_http_failure(response.status, &body),
			body,
		}))
	}

	/// Headers sent with every request: configured headers, extras, session id,
	/// protocol version, and bearer token (upstream `headers()`).
	async fn headers(&self, extra: &[(&str, String)]) -> Result<(HttpHeaders, Option<String>), McpError> {
		let mut headers = self.options.headers.clone().unwrap_or_default();
		for (name, value) in extra {
			header_set(&mut headers, name, value.clone());
		}
		{
			let state = self.lock();
			if let Some(session_id) = &state.session_id {
				header_set(&mut headers, "Mcp-Session-Id", session_id.clone());
			}
			if let Some(protocol_version) = &state.protocol_version {
				header_set(&mut headers, "MCP-Protocol-Version", protocol_version.clone());
			}
		}
		let mut token = None;
		if let Some(provider) = &self.options.auth_provider {
			token = provider.token().await?;
		}
		if let Some(token) = &token {
			header_set(&mut headers, "Authorization", format!("Bearer {token}"));
		}
		Ok((headers, token))
	}

	/// 401, or 403 with an `insufficient_scope` bearer challenge (step-up authorization).
	fn needs_authorization(response: &HttpResponse) -> bool {
		if response.status == 401 {
			return true;
		}
		if response.status != 403 {
			return false;
		}
		let header = header_find(&response.headers, "www-authenticate").unwrap_or_default();
		insufficient_scope_challenge(header)
	}

	/// Fetch with auth headers. A 401 (or a 403 asking for more scope) is handed to the auth provider
	/// once, and the request is retried with whatever credentials it left behind.
	async fn authorized_fetch(
		&self,
		method: &str,
		extra: Vec<(&'static str, String)>,
		body: Option<Vec<u8>>,
	) -> Result<HttpResponse, McpError> {
		let mut attempt = 0u32;
		loop {
			let (request_headers, token) = self.headers(&extra).await?;
			let request = HttpRequest {
				method: method.to_string(),
				url: self.url.clone(),
				headers: request_headers,
				body: body.clone(),
			};
			let response = tokio::select! {
				result = (self.fetch)(request) => result?,
				() = self.closed_wait() => {
					// Upstream aborts the socket mid-request on close.
					return Err(McpError::connection_closed("MCP connection closed"));
				}
			};
			let provider = self.options.auth_provider.as_ref();
			let handles = provider.map(|p| p.handles_unauthorized()).unwrap_or(false);
			if attempt > 0 || !Self::needs_authorization(&response) || !handles {
				return Ok(response);
			}
			let provider = provider.expect("checked above");
			provider
				.on_unauthorized(UnauthorizedContext {
					response: UnauthorizedResponse {
						status: response.status,
						headers: response.headers.clone(),
					},
					server_url: self.url.clone(),
					fetch: self.fetch.clone(),
					token: token.clone(),
				})
				.await?;
			discard(response);
			attempt += 1;
		}
	}

	/// Open a GET SSE stream. Resolves to `None` when the server answers 405 (no GET stream).
	async fn open_sse_stream(
		&self,
		last_event_id: Option<&str>,
	) -> Result<Option<ByteStream>, McpError> {
		let mut extra: Vec<(&'static str, String)> = vec![("accept", "text/event-stream".to_string())];
		if let Some(last_event_id) = last_event_id {
			extra.push(("last-event-id", last_event_id.to_string()));
		}
		let response = self.authorized_fetch("GET", extra, None).await?;
		if response.status == 405 {
			discard(response);
			return Ok(None);
		}
		let response = self.check_response(response).await?;
		self.capture_session(&response);
		let content_type = response.content_type();
		let status = response.status;
		if content_type.as_deref() != Some("text/event-stream") {
			discard(response);
			return Err(McpError::Http(HttpError {
				status,
				message: format!(
					"Unsupported MCP GET response content type: {}",
					content_type.unwrap_or_else(|| "missing".to_string())
				),
				body: String::new(),
			}));
		}
		Ok(Some(response.into_body_stream()))
	}

	async fn consume_sse(
		&self,
		stream: ByteStream,
		cursor: &mut StreamCursor,
		mut on_message: Option<&mut (dyn FnMut(JsonRpcMessage) + Send)>,
	) -> Result<(), McpError> {
		let max_event_bytes = self.options.max_message_bytes.unwrap_or(DEFAULT_MAX_MESSAGE_BYTES);
		// The three SSE callbacks share the cursor (upstream mutates one object).
		let shared = Arc::new(Mutex::new(cursor.clone()));
		let event_cursor = shared.clone();
		let id_cursor = shared.clone();
		let retry_cursor = shared.clone();
		let result = consume_sse_stream(
			stream,
			ConsumeSseOptions {
				max_event_bytes: Some(max_event_bytes),
				on_event: Box::new(move |event| {
					lock_cursor(&event_cursor).received = true;
					// Events without data prime resumption; other event types are not JSON-RPC.
					if event.data.trim().is_empty()
						|| event.event.as_deref().map(|e| e != "message").unwrap_or(false)
					{
						return;
					}
					let parsed = serde_json::from_str::<serde_json::Value>(&event.data)
						.map_err(|e| McpError::other(e.to_string()))
						.and_then(parse_json_rpc_message);
					match parsed {
						Ok(message) => {
							if let Some(on_message) = on_message.as_deref_mut() {
								on_message(message.clone());
							}
							self.events.emit_message(message);
						}
						Err(error) => self.events.emit_error(error),
					}
				}),
				on_id: Some(Box::new(move |id| {
					lock_cursor(&id_cursor).last_event_id = Some(id.to_string());
				})),
				on_retry: Some(Box::new(move |delay_ms| {
					lock_cursor(&retry_cursor).retry_ms = Some(delay_ms);
				})),
			},
		)
		.await;
		*cursor = lock_cursor(&shared).clone();
		result
	}

	/// Network failures and transient statuses are retried; auth, session, and protocol errors are not.
	/// Upstream `fetch` reports network failures, including a connection dropped mid-body, as
	/// `TypeError`, which maps to [`McpError::Network`].
	fn is_retryable(&self, error: &McpError) -> bool {
		match error {
			McpError::Http(error) => is_transient_status(error.status),
			McpError::Network(_) => true,
			_ => false,
		}
	}

	fn reconnect_delay(&self, attempt: u32, server_delay_ms: Option<u64>) -> u64 {
		if let Some(server_delay_ms) = server_delay_ms {
			return server_delay_ms;
		}
		let initial = self
			.options
			.reconnect
			.and_then(|r| r.initial_delay_ms)
			.unwrap_or(DEFAULT_RECONNECT_INITIAL_DELAY_MS);
		initial
			.saturating_mul(2u64.saturating_pow(attempt))
			.min(self.max_delay())
	}

	fn max_delay(&self) -> u64 {
		self.options
			.reconnect
			.and_then(|r| r.max_delay_ms)
			.unwrap_or(DEFAULT_RECONNECT_MAX_DELAY_MS)
	}

	fn max_retries(&self) -> u32 {
		self.options
			.reconnect
			.and_then(|r| r.max_retries)
			.unwrap_or(DEFAULT_RECONNECT_MAX_RETRIES)
	}

	fn start_get_stream(self: &Arc<Self>) {
		{
			let mut state = self.lock();
			if self.options.open_get_stream == Some(false) || state.get_stream_started || state.closed {
				return;
			}
			state.get_stream_started = true;
		}
		let me = self.clone();
		tokio::spawn(async move {
			me.run_get_stream().await;
		});
	}

	/// Keep the server-to-client stream open, reconnecting with backoff when it drops.
	async fn run_get_stream(self: &Arc<Self>) {
		let mut cursor = StreamCursor::default();
		let mut attempt = 0u32;
		while !self.closed() {
			let mut failure: Option<McpError> = None;
			match self.open_sse_stream(cursor.last_event_id.as_deref()).await {
				Ok(None) => return, // The server does not offer a GET stream.
				Ok(Some(stream)) => {
					let opened_at = std::time::Instant::now();
					match self.consume_sse(stream, &mut cursor, None).await {
						Ok(()) => {
							// A stream that stayed up for a while counts as healthy, even if it was idle.
							if cursor.received || opened_at.elapsed() > Duration::from_millis(self.max_delay()) {
								attempt = 0;
							}
						}
						Err(error) => failure = Some(error),
					}
				}
				Err(error) => failure = Some(error),
			}
			if self.closed() {
				return;
			}
			if let Some(error) = &failure {
				if !self.is_retryable(error) {
					self.events.emit_error(error.clone());
					return;
				}
			}
			cursor.received = false;
			if attempt >= self.max_retries() {
				self.events.emit_error(McpError::other(
					"MCP server-to-client stream dropped and could not be reopened",
				));
				return;
			}
			let delay = self.reconnect_delay(attempt, cursor.retry_ms);
			attempt += 1;
			if !self.sleep(delay).await {
				return;
			}
		}
	}

	/// Read the SSE stream answering one request. When the stream ends or breaks before the response
	/// arrives and the server assigned event IDs, resume it with GET and `Last-Event-ID`, as the
	/// server may close response streams at will. Otherwise only this request fails.
	async fn consume_response_stream(self: &Arc<Self>, body: ByteStream, request_id: JsonRpcId) {
		let mut cursor = StreamCursor::default();
		let mut answered = false;
		let mut stream: Option<ByteStream> = Some(body);
		let mut failure: Option<McpError> = None;
		let mut attempt = 0u32;
		loop {
			if let Some(body) = stream.take() {
				let mut saw_answer = |message: JsonRpcMessage| {
					if let JsonRpcMessage::Response(response) = &message {
						if response.id() == &request_id {
							answered = true;
						}
					}
				};
				match self.consume_sse(body, &mut cursor, Some(&mut saw_answer)).await {
					Ok(()) => failure = None,
					Err(error) => failure = Some(error),
				}
			}
			if answered || self.closed() {
				return;
			}
			if let Some(error) = &failure {
				if !self.is_retryable(error) {
					break;
				}
			}
			if cursor.last_event_id.is_none() || attempt >= self.max_retries() {
				break;
			}
			if cursor.received {
				attempt = 0;
			}
			cursor.received = false;
			let delay = self.reconnect_delay(attempt, cursor.retry_ms);
			attempt += 1;
			if !self.sleep(delay).await {
				return;
			}
			match self.open_sse_stream(cursor.last_event_id.as_deref()).await {
				Ok(new_stream) => stream = new_stream,
				Err(error) => {
					let retryable = self.is_retryable(&error);
					failure = Some(error);
					if !retryable {
						break;
					}
					stream = None;
				}
			}
		}
		if self.closed() {
			return;
		}
		let reason = match &failure {
			None => "stream ended without a response".to_string(),
			Some(error) => error.to_string(),
		};
		self.events
			.emit_message(JsonRpcMessage::Response(JsonRpcResponse::Error(JsonRpcErrorResponse {
				jsonrpc: JSONRPC_VERSION.to_string(),
				id: request_id,
				error: JsonRpcErrorObject {
					code: JSON_RPC_ERROR_CODES_INTERNAL_ERROR,
					message: format!("MCP response stream failed: {reason}"),
					data: None,
				},
			})));
	}
}

fn lock_cursor(cell: &Mutex<StreamCursor>) -> parking_lot::MutexGuard<'_, StreamCursor> {
	cell.lock()
}

/// `/?error="?insufficient_scope"?/`-style bearer challenge check
/// (upstream regex `/(?:^|[\s,])error="?insufficient_scope"?/i`).
fn insufficient_scope_challenge(header: &str) -> bool {
	const KEY: &str = "error=";
	let mut rest = header;
	loop {
		let Some(index) = rest.to_ascii_lowercase().find(KEY) else {
			return false;
		};
		let (before, after) = rest.split_at(index);
		if before.is_empty()
			|| before.ends_with([' ', ',', '\t', '\n', '\r', '\x0b', '\x0c'])
		{
			let value = &after[KEY.len()..];
			let value = value.strip_prefix('"').unwrap_or(value);
			if value.len() >= "insufficient_scope".len()
				&& value
					.get(.."insufficient_scope".len())
					.map(|prefix| prefix.eq_ignore_ascii_case("insufficient_scope"))
					.unwrap_or(false)
			{
				return true;
			}
		}
		rest = &rest[index + KEY.len()..];
	}
}

fn is_transient_status(status: u16) -> bool {
	status == 408 || status == 429 || status >= 500
}

fn describe_http_failure(status: u16, body: &str) -> String {
	let text = body.trim();
	let snippet = if text.chars().count() > ERROR_MESSAGE_BODY_CHARS {
		let truncated: String = text.chars().take(ERROR_MESSAGE_BODY_CHARS - 3).collect();
		format!("{truncated}...")
	} else {
		text.to_string()
	};
	if snippet.is_empty() {
		format!("MCP HTTP request failed with status {status}")
	} else {
		format!("MCP HTTP request failed with status {status}: {snippet}")
	}
}

impl HttpResponse {
	/// Take ownership of the body stream.
	fn into_body_stream(self) -> ByteStream {
		self.body
	}
}

#[async_trait]
impl McpTransport for StreamableHttpTransport {
	async fn start(&self) -> Result<(), McpError> {
		let mut state = self.lock();
		if state.started {
			return Err(McpError::other("MCP Streamable HTTP transport already started"));
		}
		if state.closed {
			return Err(McpError::connection_closed("MCP connection closed"));
		}
		state.started = true;
		Ok(())
	}

	fn set_protocol_version(&self, version: &str) {
		self.lock().protocol_version = Some(version.to_string());
	}

	async fn send(&self, message: JsonRpcMessage) -> Result<(), McpError> {
		{
			let state = self.lock();
			if !state.started || state.closed {
				return Err(McpError::connection_closed("MCP connection closed"));
			}
		}
		let payload = serde_json::to_vec(&message).map_err(|e| McpError::other(e.to_string()))?;
		let response = self
			.authorized_fetch(
				"POST",
				vec![
					("accept", "application/json, text/event-stream".to_string()),
					("content-type", "application/json".to_string()),
				],
				Some(payload),
			)
			.await?;
		let response = self.check_response(response).await?;
		self.capture_session(&response);

		let method = match &message {
			JsonRpcMessage::Request(request) => Some(request.method.clone()),
			JsonRpcMessage::Notification(notification) => Some(notification.method.clone()),
			JsonRpcMessage::Response(_) => None,
		};
		if !is_request(&message) {
			// Notifications and responses are acknowledged with 202 and carry no reply; ignore any body.
			discard(response);
			// The server-to-client stream may only open once the session is initialized.
			if method.as_deref() == Some("notifications/initialized") {
				if let Some(me) = self.self_arc() {
					me.start_get_stream();
				}
			}
			return Ok(());
		}
		let method = method.unwrap_or_default();
		if response.status == 202 || response.status == 204 {
			let status = response.status;
			discard(response);
			return Err(McpError::Http(HttpError {
				status,
				message: format!("MCP server accepted request {method} without a response"),
				body: String::new(),
			}));
		}
		let content_type = response.content_type();
		if content_type.as_deref() == Some("application/json") {
			let body: serde_json::Value = response.json().await?;
			let items: Vec<serde_json::Value> = match body {
				serde_json::Value::Array(items) => items,
				other => vec![other],
			};
			for item in items {
				let parsed = parse_json_rpc_message(item)?;
				self.events.emit_message(parsed);
			}
			return Ok(());
		}
		if content_type.as_deref() == Some("text/event-stream") {
			let me = self.self_arc().expect("transport Arc is alive during send");
			let request_id = match &message {
				JsonRpcMessage::Request(request) => request.id.clone(),
				_ => unreachable!("checked above"),
			};
			let body = response.into_body_stream();
			tokio::spawn(async move {
				me.consume_response_stream(body, request_id).await;
			});
			return Ok(());
		}
		let status = response.status;
		discard(response);
		Err(McpError::Http(HttpError {
			status,
			message: format!(
				"Unsupported MCP response content type: {}",
				content_type.unwrap_or_else(|| "missing".to_string())
			),
			body: String::new(),
		}))
	}

	async fn close(&self) -> Result<(), McpError> {
		{
			let mut state = self.lock();
			if state.closed {
				return Ok(());
			}
			state.closed = true;
		}
		self.closed_notify.notify_waiters();
		let (started, has_session) = {
			let state = self.lock();
			(state.started, state.session_id.is_some())
		};
		if started && has_session {
			let _ = tokio::time::timeout(Duration::from_secs(1), async {
				// Resolving auth headers failed; the session will expire on the server.
				let Ok((headers, _)) = self.headers(&[]).await else {
					return;
				};
				let request = HttpRequest {
					method: "DELETE".to_string(),
					url: self.url.clone(),
					headers,
					body: None,
				};
				if let Ok(response) = (self.fetch)(request).await {
					discard(response);
				}
			})
			.await;
		}
		self.events.emit_close();
		Ok(())
	}

	fn on_message(&self, listener: MessageListener) -> Unsubscribe {
		self.events.on_message(listener)
	}

	fn on_error(&self, listener: ErrorListener) -> Unsubscribe {
		self.events.on_error(listener)
	}

	fn on_close(&self, listener: CloseListener) -> Unsubscribe {
		self.events.on_close(listener)
	}
}

fn is_request(message: &JsonRpcMessage) -> bool {
	matches!(message, JsonRpcMessage::Request(_))
}

/// Upstream reads the whole error body then slices it to `MAX_ERROR_BODY_BYTES`;
/// here the read stops early (same resulting string for realistic bodies).
async fn read_body_text(body: ByteStream) -> String {
	let mut body = body;
	let mut out: Vec<u8> = Vec::new();
	while out.len() < MAX_ERROR_BODY_BYTES {
		match body.next().await {
			Some(Ok(chunk)) => {
				let take = (MAX_ERROR_BODY_BYTES - out.len()).min(chunk.len());
				out.extend_from_slice(&chunk[..take]);
			}
			Some(Err(_)) | None => break,
		}
	}
	String::from_utf8_lossy(&out).into_owned()
}
