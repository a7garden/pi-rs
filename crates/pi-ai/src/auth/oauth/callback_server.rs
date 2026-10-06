//! Port of `packages/ai/src/auth/oauth/callback-server.ts` — the loopback
//! OAuth redirect handler shared by the browser sign-in flows.
//!
//! Upstream uses `node:http`; the Rust port is a minimal `tokio` TCP listener
//! speaking just enough HTTP/1.1 for the browser redirect GET (the pi-mcp
//! crate's callback-server approach). Request routing, status codes, page
//! bodies, and wait/cancel/close semantics match upstream.
//!
//! # Deviations
//!
//! - D27: responses carry `Connection: close` and each connection is closed
//!   after one response, so the upstream spare-connection problem that
//!   motivated `closeAllConnections()` cannot occur; `close()` drops the
//!   listener immediately. Node's runtime `server.on("error")` plumbing has no
//!   equivalent — accept-loop failures terminate the loop, and the wait is
//!   already finished by abort/timeout/close in every reachable case.
//! - D28: `start_oauth_callback_server` returns a typed
//!   [`CallbackServerError`] instead of a Node error carrying `code`; callers
//!   (OpenAI ChatGPT login) match `AddrInUse` by variant, preserving the
//!   EADDRINUSE fallback behavior.

use std::future::Future;
use std::sync::{Arc};
use {parking_lot::Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::auth::types::{AuthPrompt, AuthPromptKind, ProviderAuthInteraction};
use crate::utils::abort::{AbortController, AbortSignal};
use crate::utils::oauth_page::{oauth_error_html, oauth_success_html};
use crate::utils::thrown::ThrownValue;

/// Exchange callback executed before the browser page is sent
/// (upstream `complete: (code) => Promise<T>`).
pub type CompleteFn<T> = Arc<
	dyn Fn(String) -> futures_util::future::BoxFuture<'static, Result<T, ThrownValue>> + Send + Sync,
>;

/// Upstream `OAuthCallbackServerOptions<T>`.
pub struct OAuthCallbackServerOptions<T> {
	/// Provider name used on the browser page, for example `OpenAI`.
	pub provider_name: String,
	/// Address to listen on.
	pub host: String,
	/// Port to listen on; `0` picks a free port.
	pub port: u16,
	pub path: String,
	/// Host in `redirectUri` when it differs from `host`, for example `localhost`.
	pub redirect_host: Option<String>,
	/// Expected `state` parameter. `None` when the provider does not send one.
	pub state: Option<String>,
	/// Finishes the sign-in with the received code before the browser page is
	/// sent, so the page can show exchange failures. Pass
	/// `Arc::new(|code| Box::pin(async move { Ok(code) }))` to exchange later.
	pub complete: CompleteFn<T>,
	pub signal: Option<AbortSignal>,
	pub timeout_ms: Option<u64>,
}

/// Failure modes of [`start_oauth_callback_server`] (upstream Node errors).
#[derive(Debug)]
pub enum CallbackServerError {
	/// `signal.aborted` at start time: `"Login cancelled"`.
	LoginCancelled,
	/// The requested port is taken (Node `EADDRINUSE`).
	AddrInUse { host: String, port: u16 },
	/// Any other listen failure, message preserved.
	Listen(std::io::Error),
	/// `server.address()` was not a TCP address.
	DidNotBindTcp,
}

impl std::fmt::Display for CallbackServerError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			CallbackServerError::LoginCancelled => f.write_str("Login cancelled"),
			CallbackServerError::AddrInUse { host, port } => {
				write!(f, "listen EADDRINUSE: address already in use {host}:{port}")
			}
			CallbackServerError::Listen(error) => write!(f, "listen {error}"),
			CallbackServerError::DidNotBindTcp => f.write_str("OAuth callback server did not bind to TCP"),
		}
	}
}

impl std::error::Error for CallbackServerError {
	fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
		match self {
			CallbackServerError::Listen(error) => Some(error),
			_ => None,
		}
	}
}

impl From<CallbackServerError> for ThrownValue {
	fn from(error: CallbackServerError) -> ThrownValue {
		match error {
			CallbackServerError::Listen(source) => ThrownValue::error(CallbackServerError::Listen(source)),
			other => ThrownValue::message(other.to_string()),
		}
	}
}

/// Terminal state of the wait promise (upstream resolve/reject).
enum WaitOutcome<T> {
	/// Resolved with the result of `complete`, or `None` after `cancel()`.
	Completed(Option<T>),
	/// Rejected with an error.
	Failed(ThrownValue),
}

struct ServerState<T> {
	/// A callback has been claimed by `complete` (upstream `claimed`).
	claimed: bool,
	settled: bool,
	outcome: Option<WaitOutcome<T>>,
}

struct ServerInner<T> {
	state: Mutex<ServerState<T>>,
	notify: tokio::sync::Notify,
	listener: Mutex<Option<Arc<tokio::net::TcpListener>>>,
	accept_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
	/// Aborts the timeout task when the wait settles first.
	timeout_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
	/// Removes the signal's abort listener.
	remove_abort_listener: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl<T: Clone + Send + Sync + 'static> ServerInner<T> {
	/// Upstream `finish` — idempotent; wakes every waiter.
	fn finish(&self, outcome: WaitOutcome<T>) {
		{
			let mut state = self.state.lock();
			if state.settled {
				return;
			}
			state.settled = true;
			state.outcome = Some(outcome);
		}
		if let Some(task) = self.timeout_task.lock().take() {
			task.abort();
		}
		if let Some(remove) = self.remove_abort_listener.lock().take() {
			remove();
		}
		self.notify.notify_waiters();
	}

	/// `claimed || settled` probe for the 409 page.
	fn claimed_or_settled(&self) -> bool {
		let state = self.state.lock();
		state.claimed || state.settled
	}

	/// Atomic check-and-set of `claimed` (upstream does the same within one
	/// event-loop turn).
	fn claim(&self) -> bool {
		let mut state = self.state.lock();
		if state.claimed || state.settled {
			return false;
		}
		state.claimed = true;
		true
	}
}

fn outcome_result<T: Clone>(outcome: &WaitOutcome<T>) -> Result<Option<T>, ThrownValue> {
	match outcome {
		WaitOutcome::Completed(value) => Ok(value.clone()),
		WaitOutcome::Failed(error) => Err(error.clone()),
	}
}

/// Loopback callback server handle (upstream `OAuthCallbackServer<T>`).
pub struct OAuthCallbackServer<T>
where
	T: Clone + Send + Sync + 'static,
{
	inner: Arc<ServerInner<T>>,
	redirect_uri: String,
}

impl<T: Clone + Send + Sync + 'static> Clone for OAuthCallbackServer<T> {
	fn clone(&self) -> Self {
		OAuthCallbackServer { inner: self.inner.clone(), redirect_uri: self.redirect_uri.clone() }
	}
}

impl<T: Clone + Send + Sync + 'static> OAuthCallbackServer<T> {
	/// The redirect URI handed to the provider.
	pub fn redirect_uri(&self) -> &str {
		&self.redirect_uri
	}

	/// Resolves with the result of `complete`, or `None` after `cancel()`.
	/// Rejects when the provider redirects with an error, `complete` fails, the
	/// signal aborts, or the timeout elapses. Any number of callers may observe
	/// the same outcome (upstream shared `waitPromise`).
	pub async fn wait(&self) -> Result<Option<T>, ThrownValue> {
		loop {
			let notified = self.inner.notify.notified();
			{
				let state = self.inner.state.lock();
				if let Some(outcome) = &state.outcome {
					break outcome_result(outcome);
				}
			}
			notified.await;
		}
	}

	/// Stop waiting for the browser unless a callback is already being
	/// completed (upstream `cancel()`).
	pub fn cancel(&self) {
		if !self.inner.state.lock().claimed {
			self.inner.finish(WaitOutcome::Completed(None));
		}
	}

	/// Stop the server; a pending wait settles with
	/// `"OAuth callback server closed"` (upstream `close()`).
	pub fn close(&self) {
		self.inner.finish(WaitOutcome::Failed(ThrownValue::message("OAuth callback server closed")));
		// Drop the stored listener clone (closing the socket) and abort the
		// accept loop, which drops its own clone.
		*self.inner.listener.lock() = None;
		if let Some(task) = self.inner.accept_task.lock().take() {
			task.abort();
		}
	}
}

fn status_text(status: u16) -> &'static str {
	match status {
		200 => "OK",
		400 => "Bad Request",
		404 => "Not Found",
		409 => "Conflict",
		502 => "Bad Gateway",
		_ => "",
	}
}

/// Upstream chatgpt sendHtml — Content-Type only (no cache-control).
pub(crate) async fn send_html_page(stream: &mut tokio::net::TcpStream, status: u16, html: &str) {
	let response = format!(
		"HTTP/1.1 {} {}\r\nContent-Type: text/html; charset=utf-8\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
		status,
		crate::auth::oauth::http_status_text(status),
		html.len(),
		html
	);
	let _ = stream.write_all(response.as_bytes()).await;
	let _ = stream.flush().await;
	let _ = stream.shutdown().await;
	let _ = stream.read(&mut [0u8; 1]).await; // drain until the peer closes
}

/// Node `Response.statusText` equivalents for the statuses these flows hit
/// (delegated to the oauth module root helper).
pub(crate) fn http_status_text_for(status: u16) -> &'static str {
	crate::auth::oauth::http_status_text(status)
}

/// Upstream `sendPage` — HTML with the pinned headers.
async fn send_page(stream: &mut tokio::net::TcpStream, status: u16, html: &str) {
	let response = format!(
		"HTTP/1.1 {} {}\r\ncontent-type: text/html; charset=utf-8\r\ncache-control: no-store\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
		status,
		status_text(status),
		html.len(),
		html
	);
	let _ = stream.write_all(response.as_bytes()).await;
	let _ = stream.flush().await;
	let _ = stream.shutdown().await;
	let _ = stream.read(&mut [0u8; 1]).await; // drain until the peer closes
}

/// Read one HTTP request head (request line + headers) with a size cap.
pub(crate) async fn read_request_head(stream: &mut tokio::net::TcpStream) -> Option<String> {
	let mut buffer: Vec<u8> = Vec::with_capacity(1024);
	let mut chunk = [0u8; 1024];
	loop {
		if let Some(position) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
			return Some(String::from_utf8_lossy(&buffer[..position]).into_owned());
		}
		if buffer.len() > 16 * 1024 {
			return None;
		}
		let read = stream.read(&mut chunk).await.ok()?;
		if read == 0 {
			return None;
		}
		buffer.extend_from_slice(&chunk[..read]);
	}
}

/// Request-routing data shared with connection handlers.
struct SharedRequestOptions<T> {
	provider_name: String,
	path: String,
	state: Option<String>,
	complete: CompleteFn<T>,
}

/// Upstream `startOAuthCallbackServer<T>`.
pub async fn start_oauth_callback_server<T>(
	options: OAuthCallbackServerOptions<T>,
) -> Result<OAuthCallbackServer<T>, CallbackServerError>
where
	T: Clone + Send + Sync + 'static,
{
	if options.signal.as_ref().is_some_and(|signal| signal.aborted()) {
		return Err(CallbackServerError::LoginCancelled);
	}

	let listener = tokio::net::TcpListener::bind((options.host.as_str(), options.port))
		.await
		.map_err(|error| match error.kind() {
			std::io::ErrorKind::AddrInUse => CallbackServerError::AddrInUse {
				host: options.host.clone(),
				port: options.port,
			},
			_ => CallbackServerError::Listen(error),
		})?;
	let port = listener.local_addr().map_err(|_| CallbackServerError::DidNotBindTcp)?.port();
	let listener = Arc::new(listener);

	let inner = Arc::new(ServerInner {
		state: Mutex::new(ServerState { claimed: false, settled: false, outcome: None }),
		notify: tokio::sync::Notify::new(),
		listener: Mutex::new(Some(listener.clone())),
		accept_task: Mutex::new(None),
		timeout_task: Mutex::new(None),
		remove_abort_listener: Mutex::new(None),
	});

	// const redirectHost = options.redirectHost ?? options.host;
	let redirect_host = options.redirect_host.clone().unwrap_or_else(|| options.host.clone());
	let server = OAuthCallbackServer {
		inner: inner.clone(),
		redirect_uri: format!(
			"http://{}:{port}{}",
			if redirect_host.contains(':') { format!("[{redirect_host}]") } else { redirect_host.clone() },
			options.path
		),
	};

	// Accept loop: each connection is handled on its own task and closed after
	// one response (deviation D27).
	let handler_options = Arc::new(SharedRequestOptions {
		provider_name: options.provider_name.clone(),
		path: options.path.clone(),
		state: options.state.clone(),
		complete: options.complete.clone(),
	});
	let accept_inner = inner.clone();
	let accept_task = tokio::spawn(async move {
		loop {
			let (stream, _) = match listener.accept().await {
				Ok(accepted) => accepted,
				Err(_) => break,
			};
			let handler_options = handler_options.clone();
			let handler_inner = accept_inner.clone();
			tokio::spawn(async move {
				handle_connection(stream, handler_options, handler_inner).await;
			});
		}
	});
	*inner.accept_task.lock() = Some(accept_task);

	// signal?.addEventListener("abort", onAbort, { once: true })
	if let Some(signal) = options.signal.clone() {
		let abort_inner = inner.clone();
		let remove = signal.add_listener(move || {
			abort_inner.finish(WaitOutcome::Failed(ThrownValue::message("Login cancelled")));
		});
		*inner.remove_abort_listener.lock() = Some(remove);
	}

	// timer = setTimeout(() => finish({ error: ... }), options.timeoutMs)
	if let Some(timeout_ms) = options.timeout_ms {
		let timeout_inner = inner.clone();
		let provider_name = options.provider_name.clone();
		let task = tokio::spawn(async move {
			tokio::time::sleep(std::time::Duration::from_millis(timeout_ms)).await;
			timeout_inner.finish(WaitOutcome::Failed(ThrownValue::message(format!(
				"{provider_name} sign-in timed out"
			))));
		});
		*inner.timeout_task.lock() = Some(task);
	}

	Ok(server)
}

/// One connection = at most one request, then the socket closes (D27).
/// Mirrors the upstream `createServer` handler branch for branch.
async fn handle_connection<T>(
	mut stream: tokio::net::TcpStream,
	options: Arc<SharedRequestOptions<T>>,
	inner: Arc<ServerInner<T>>,
) where
	T: Clone + Send + Sync + 'static,
{
	let Some(head) = read_request_head(&mut stream).await else {
		return;
	};
	let Some(request_line) = head.lines().next() else {
		return;
	};
	let mut parts = request_line.split_whitespace();
	let method = parts.next().unwrap_or("");
	let target = parts.next().unwrap_or("/");

	// const url = new URL(request.url ?? "/", "http://localhost");
	let Ok(url) = url::Url::parse(&format!("http://localhost{target}")) else {
		send_page(&mut stream, 404, &oauth_error_html("Callback route not found.", None)).await;
		return;
	};
	let query: std::collections::BTreeMap<String, String> = url
		.query_pairs()
		.map(|(key, value)| (key.into_owned(), value.into_owned()))
		.collect();

	if method != "GET" || url.path() != options.path {
		send_page(&mut stream, 404, &oauth_error_html("Callback route not found.", None)).await;
		return;
	}
	if let Some(expected_state) = &options.state {
		if query.get("state").map(String::as_str) != Some(expected_state.as_str()) {
			send_page(&mut stream, 400, &oauth_error_html("State mismatch.", None)).await;
			return;
		}
	}
	if inner.claimed_or_settled() {
		send_page(&mut stream, 409, &oauth_error_html("This sign-in has already been handled.", None)).await;
		return;
	}
	if let Some(error) = query.get("error") {
		let description = query.get("error_description").unwrap_or(error);
		send_page(
			&mut stream,
			400,
			&oauth_error_html(&format!("{} authorization failed.", options.provider_name), Some(description)),
		)
		.await;
		inner.finish(WaitOutcome::Failed(ThrownValue::message(format!(
			"{} authorization failed: {description}",
			options.provider_name
		))));
		return;
	}
	let Some(code) = query.get("code") else {
		send_page(&mut stream, 400, &oauth_error_html("Missing authorization code.", None)).await;
		return;
	};
	if !inner.claim() {
		send_page(&mut stream, 409, &oauth_error_html("This sign-in has already been handled.", None)).await;
		return;
	}
	match (options.complete)(code.clone()).await {
		Ok(value) => {
			send_page(
				&mut stream,
				200,
				&oauth_success_html(&format!("Signed in to {}. You may now close this page.", options.provider_name)),
			)
			.await;
			inner.finish(WaitOutcome::Completed(Some(value)));
		}
		Err(error) => {
			let failure = crate::utils::diagnostics::format_thrown_value(&error);
			send_page(
				&mut stream,
				502,
				&oauth_error_html(&format!("{} sign-in failed.", options.provider_name), Some(&failure)),
			)
			.await;
			inner.finish(WaitOutcome::Failed(error));
		}
	}
}

/// How the manual prompt resolves (upstream
/// `{ type: "callback"; value: T } | { type: "manual"; input: string }`).
pub enum CallbackOrManual<T> {
	Callback(T),
	Manual(String),
}

/// Wait for the browser callback, or for the user to paste the code or
/// redirect URL when the browser cannot reach the loopback server (for
/// example over SSH). Without a callback server only the manual prompt is
/// used (upstream `waitForCallbackOrManualInput`).
pub async fn wait_for_callback_or_manual_input<T>(
	interaction: &ProviderAuthInteraction,
	callback: Option<&OAuthCallbackServer<T>>,
	prompt: (&str, &str),
) -> Result<CallbackOrManual<T>, ThrownValue>
where
	T: Clone + Send + Sync + 'static,
{
	let manual_abort = AbortController::new();
	let manual_signal = manual_abort.signal();
	let prompt_future = (interaction.prompt)(AuthPrompt {
		signal: Some(manual_signal.clone()),
		kind: AuthPromptKind::ManualCode {
			message: prompt.0.to_string(),
			placeholder: Some(prompt.1.to_string()),
		},
	});

	// The manual prompt runs to settlement on its own task; on settlement (ok
	// or error) it cancels the callback wait, like the upstream `.then` /
	// `.catch` chain.
	let manual_task = {
		let callback = callback.cloned();
		tokio::spawn(async move {
			let result = prompt_future.await;
			if let Some(callback) = callback {
				callback.cancel();
			}
			result
		})
	};

	// const value = await callback?.wait();
	let wait_outcome: Option<Result<Option<T>, ThrownValue>> = match callback {
		Some(callback) => Some(callback.wait().await),
		None => None,
	};

	let result = match wait_outcome {
		// Upstream: a rejected wait propagates before manualError is checked.
		Some(Err(error)) => {
			manual_task.abort();
			Err(error)
		}
		Some(Ok(Some(value))) => {
			// if (manualError) throw manualError — visible only when the manual
			// prompt already settled; a still-pending prompt is discarded (the
			// upstream finally aborts it).
			if manual_task.is_finished() {
				match manual_task.await {
					Ok(Err(manual_error)) => Err(manual_error),
					Ok(Ok(_)) => Ok(CallbackOrManual::Callback(value)),
					Err(_) => Err(ThrownValue::message("Login cancelled")),
				}
			} else {
				manual_task.abort();
				Ok(CallbackOrManual::Callback(value))
			}
		}
		Some(Ok(None)) | None => {
			// The prompt stays live while the user types (upstream awaits the
			// manual promise before its finally aborts the signal).
			match manual_task.await {
				Err(join_error) if join_error.is_cancelled() => Err(ThrownValue::message("Login cancelled")),
				Err(_) => Err(ThrownValue::message("Login cancelled")),
				Ok(Err(manual_error)) => Err(manual_error),
				Ok(Ok(input)) => Ok(CallbackOrManual::Manual(input)),
			}
		}
	};
	// finally { manualAbort.abort(); } — the prompt UI is dismissed once login
	// settles.
	manual_abort.abort();
	result
}

/// Convenience for flows that ignore the manual/callback distinction.
pub fn boxed_complete<T, F, Fut>(complete: F) -> CompleteFn<T>
where
	T: Clone + Send + Sync + 'static,
	F: Fn(String) -> Fut + Send + Sync + 'static,
	Fut: Future<Output = Result<T, ThrownValue>> + Send + 'static,
{
	Arc::new(move |code| Box::pin(complete(code)))
}
