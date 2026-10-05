//! Port of `packages/mcp/src/oauth/callback.ts`.
//!
//! Upstream builds on `node:http`; here a minimal HTTP/1.1 GET server runs on
//! `tokio::TcpListener` (GET-with-query is all an OAuth redirect delivers).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use url::Url;

use crate::protocol::jsonrpc::McpError;

#[derive(Debug, Clone, PartialEq)]
pub struct OAuthCallback {
	pub code: String,
	pub state: String,
	pub iss: Option<String>,
}

/// Outcome shown on the browser page after the redirect.
#[derive(Debug, Clone, PartialEq)]
pub enum OAuthCallbackPage {
	Ok,
	Failure { message: String, details: Option<String> },
}

impl OAuthCallbackPage {
	fn plain_text(&self) -> String {
		match self {
			OAuthCallbackPage::Ok => "Authorization complete. You may close this window.".to_string(),
			OAuthCallbackPage::Failure { message, details } => match details {
				Some(details) => format!("{message}\n\n{details}"),
				None => message.clone(),
			},
		}
	}
}

pub type RenderPage = Arc<dyn Fn(&OAuthCallbackPage) -> String + Send + Sync>;

#[derive(Clone)]
pub struct OAuthCallbackServerOptions {
	/// Address to listen on. Default: `127.0.0.1`.
	pub host: Option<String>,
	/// Host name in `redirectUrl`, for example `localhost` for a client registered with it while
	/// listening on `127.0.0.1`. Default: `host`.
	pub redirect_host: Option<String>,
	pub port: Option<u16>,
	pub path: Option<String>,
	/// More paths that receive the callback, for example a server-specific path of a redirect URI.
	pub extra_paths: Vec<String>,
	pub timeout_ms: Option<u64>,
	/// Render the browser page as HTML. Default: a plain-text message.
	pub render_page: Option<RenderPage>,
}

impl Default for OAuthCallbackServerOptions {
	fn default() -> Self {
		OAuthCallbackServerOptions {
			host: None,
			redirect_host: None,
			port: None,
			path: None,
			extra_paths: Vec::new(),
			timeout_ms: None,
			render_page: None,
		}
	}
}

struct PendingCallback {
	tx: tokio::sync::oneshot::Sender<Result<OAuthCallback, McpError>>,
	/// With `path`, a response on another path fails, so a server-specific redirect URI
	/// can tell authorization servers apart (RFC 9700 section 4.4.2.2).
	path: Option<String>,
}

struct CallbackState {
	pending: std::sync::Mutex<HashMap<String, PendingCallback>>,
	paths: Vec<String>,
	render_page: Option<RenderPage>,
	timeout_ms: u64,
	shutdown: Arc<tokio::sync::Notify>,
}

/// Upstream `OAuthCallbackServer`.
pub struct OAuthCallbackServer {
	pub redirect_url: String,
	state: Arc<CallbackState>,
	listener: Arc<TcpListener>,
}

impl OAuthCallbackServer {
	/// Upstream `OAuthCallbackServer.listen`. Returns an `Arc` so waiters can run
	/// on separate tasks (upstream shares the instance through the event loop).
	pub async fn listen(options: OAuthCallbackServerOptions) -> Result<Arc<OAuthCallbackServer>, McpError> {
		let host = options.host.clone().unwrap_or_else(|| "127.0.0.1".to_string());
		let redirect_host = options.redirect_host.clone().unwrap_or_else(|| host.clone());
		let path = options.path.clone().unwrap_or_else(|| "/callback".to_string());
		let listener = Arc::new(
			TcpListener::bind((host.as_str(), options.port.unwrap_or(0)))
				.await
				.map_err(|e| McpError::other(e.to_string()))?,
		);
		let port = listener
			.local_addr()
			.map_err(|e| McpError::other(e.to_string()))?
			.port();
		let host_in_url = if redirect_host.contains(':') {
			format!("[{redirect_host}]")
		} else {
			redirect_host
		};
		let redirect_url = format!("http://{host_in_url}:{port}{path}");
		let state = Arc::new(CallbackState {
			pending: std::sync::Mutex::new(HashMap::new()),
			paths: [vec![path], options.extra_paths].concat(),
			render_page: options.render_page,
			timeout_ms: options.timeout_ms.unwrap_or(5 * 60_000),
			shutdown: Arc::new(tokio::sync::Notify::new()),
		});
		let server = Arc::new(OAuthCallbackServer {
			redirect_url,
			state: state.clone(),
			listener: Arc::clone(&listener),
		});
		tokio::spawn(async move {
			loop {
				let next = tokio::select! {
					accepted = listener.accept() => accepted,
					() = state.shutdown.notified() => break,
				};
				let Ok((stream, _)) = next else {
					break;
				};
				let state = state.clone();
				tokio::spawn(async move {
					handle_connection(stream, state).await;
				});
			}
		});
		Ok(server)
	}

	/// Wait for the authorization response with `state`. With `path`, a response on another
	/// path fails, so a server-specific redirect URI can tell authorization servers apart
	/// (RFC 9700 section 4.4.2.2).
	pub async fn wait_for_callback(
		&self,
		state: &str,
		path: Option<&str>,
	) -> Result<OAuthCallback, McpError> {
		let (tx, rx) = tokio::sync::oneshot::channel();
		{
			let mut pending = self.state.pending.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
			if pending.contains_key(state) {
				return Err(McpError::other("OAuth state is already pending"));
			}
			pending.insert(
				state.to_string(),
				PendingCallback { tx, path: path.map(str::to_string) },
			);
		}
		// Upstream arms a per-waiter timeout.
		let pending_state = Arc::clone(&self.state);
		let state_key = state.to_string();
		let timeout = Duration::from_millis(self.state.timeout_ms);
		tokio::spawn(async move {
			tokio::time::sleep(timeout).await;
			let mut pending = pending_state
				.pending
				.lock()
				.unwrap_or_else(std::sync::PoisonError::into_inner);
			if let Some(waiter) = pending.remove(&state_key) {
				let _ = waiter.tx.send(Err(McpError::other("OAuth callback timed out")));
			}
		});
		match rx.await {
			Ok(result) => result,
			// The server closed without answering this waiter.
			Err(_) => Err(McpError::other("OAuth callback server closed")),
		}
	}

	pub async fn close(self: Arc<Self>) -> Result<(), McpError> {
		{
			let mut pending = self.state.pending.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
			for (_, waiter) in pending.drain() {
				let _ = waiter.tx.send(Err(McpError::other("OAuth callback server closed")));
			}
		}
		self.state.shutdown.notify_waiters();
		// Dropping the listener ends the accept loop once the task observes shutdown.
		drop(Arc::clone(&self.listener));
		Ok(())
	}
}

async fn handle_connection(mut stream: tokio::net::TcpStream, state: Arc<CallbackState>) {
	// Read the request head; OAuth callbacks are GETs with a small query.
	let mut buffer = Vec::new();
	let mut chunk = [0u8; 4096];
	let request_head = loop {
		match stream.read(&mut chunk).await {
			Ok(0) | Err(_) => return,
			Ok(n) => {
				buffer.extend_from_slice(&chunk[..n]);
				if let Some(position) = find_subsequence(&buffer, b"\r\n\r\n") {
					break String::from_utf8_lossy(&buffer[..position]).into_owned();
				}
				if buffer.len() > 64 * 1024 {
					return;
				}
			}
		}
	};
	let request_line = request_head.lines().next().unwrap_or_default();
	let mut parts = request_line.split_whitespace();
	let _method = parts.next().unwrap_or_default();
	let target = parts.next().unwrap_or("/");
	let (status, content_type, body) = handle_request(state, target);
	let status_text = match status {
		200 => "OK",
		400 => "Bad Request",
		404 => "Not Found",
		_ => "Error",
	};
	let response = format!(
		"HTTP/1.1 {status} {status_text}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
		body.len()
	);
	let _ = stream.write_all(response.as_bytes()).await;
	let _ = stream.shutdown().await;
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
	haystack.windows(needle.len()).position(|window| window == needle)
}

fn handle_request(state: Arc<CallbackState>, target: &str) -> (u16, &'static str, String) {
	// Upstream parses `request.url` against the redirect URL; only path + query matter here.
	let base = Url::parse("http://callback.local").expect("static base");
	let url = match base.join(target) {
		Ok(url) => url,
		Err(_) => {
			return reply(
				&state,
				400,
				&OAuthCallbackPage::Failure {
					message: "Invalid or expired OAuth state".to_string(),
					details: None,
				},
			)
		}
	};
	if !state.paths.iter().any(|path| path == url.path()) {
		return reply(
			&state,
			404,
			&OAuthCallbackPage::Failure { message: "Not found".to_string(), details: None },
		);
	}
	let query_state =
		url.query_pairs().find(|(name, _)| name == "state").map(|(_, value)| value.into_owned());
	let pending = query_state.as_ref().and_then(|state_key| {
		state
			.pending
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner)
			.remove(state_key)
	});
	let (Some(query_state), Some(pending)) = (query_state, pending) else {
		return reply(
			&state,
			400,
			&OAuthCallbackPage::Failure {
				message: "Invalid or expired OAuth state".to_string(),
				details: None,
			},
		);
	};
	if let Some(expected_path) = &pending.path {
		if url.path() != expected_path {
			let _ = pending.tx.send(Err(McpError::other(
				"The authorization response arrived on another redirect URI",
			)));
			return reply(
				&state,
				400,
				&OAuthCallbackPage::Failure {
					message: "Unexpected redirect URI".to_string(),
					details: None,
				},
			);
		}
	}
	if let Some(error) = query_param(&url, "error") {
		let description = query_param(&url, "error_description").unwrap_or(error);
		let _ = pending.tx.send(Err(McpError::other(description.clone())));
		return reply(
			&state,
			200,
			&OAuthCallbackPage::Failure {
				message: "Authorization failed. You may close this window.".to_string(),
				details: Some(description),
			},
		);
	}
	let Some(code) = query_param(&url, "code") else {
		let _ = pending.tx.send(Err(McpError::other(
			"OAuth callback did not include an authorization code",
		)));
		return reply(
			&state,
			400,
			&OAuthCallbackPage::Failure {
				message: "Missing authorization code".to_string(),
				details: None,
			},
		);
	};
	let iss = query_param(&url, "iss");
	let _ = pending.tx.send(Ok(OAuthCallback { code, state: query_state, iss }));
	reply(&state, 200, &OAuthCallbackPage::Ok)
}

fn query_param(url: &Url, name: &str) -> Option<String> {
	url.query_pairs().find(|(key, _)| key == name).map(|(_, value)| value.into_owned())
}

fn reply(state: &CallbackState, status: u16, page: &OAuthCallbackPage) -> (u16, &'static str, String) {
	match &state.render_page {
		Some(render) => (status, "text/html; charset=utf-8", render(page)),
		None => (status, "text/plain; charset=utf-8", page.plain_text()),
	}
}
