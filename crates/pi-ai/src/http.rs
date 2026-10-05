//! HTTP transport seam for pi-ai.
//!
//! Upstream providers call `globalThis.fetch` (type alias `FetchFunction`) and
//! consume `Response.body` as a `ReadableStream`. Rust has no ambient fetch,
//! so the same seam is an explicit injectable function type: every request
//! path accepts `options.fetch` and falls back to
//! [`default_fetch`], a reqwest (rustls) adapter. Tests inject recording
//! fakes — the port of `test/fetch-option.test.ts` asserts that the custom
//! implementation is used and the default one is not.
//!
//! This module is the crate's single point where HTTP actually happens; the
//! core stays pure Rust (crate docs, deviation D4).

use std::future::{Future, pending};
use std::pin::Pin;
use std::sync::{Arc, LazyLock};

/// A chunk of request or response body.
pub enum HttpBody {
	Empty,
	/// Fully buffered bytes (requests, and responses when the adapter
	/// buffers).
	Bytes(Vec<u8>),
	/// Incrementally delivered bytes (streaming responses, SSE).
	Stream(BodyStream),
}

impl std::fmt::Debug for HttpBody {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			HttpBody::Empty => f.write_str("HttpBody::Empty"),
			HttpBody::Bytes(bytes) => f.debug_tuple("HttpBody::Bytes").field(&bytes.len()).finish(),
			HttpBody::Stream(_) => f.write_str("HttpBody::Stream"),
		}
	}
}

impl HttpBody {
	pub fn bytes(bytes: impl Into<Vec<u8>>) -> HttpBody {
		HttpBody::Bytes(bytes.into())
	}

	pub fn empty() -> HttpBody {
		HttpBody::Empty
	}

	/// Stream of byte chunks; errors carry [`ProviderHttpError`].
	pub fn stream(inner: impl futures_core::Stream<Item = Result<Vec<u8>, ProviderHttpError>> + Send + 'static) -> HttpBody {
		HttpBody::Stream(Box::pin(inner))
	}

	/// Collect the body into bytes, draining streams chunk by chunk.
	pub async fn collect(self) -> Result<Vec<u8>, ProviderHttpError> {
		match self {
			HttpBody::Empty => Ok(Vec::new()),
			HttpBody::Bytes(bytes) => Ok(bytes),
			HttpBody::Stream(stream) => {
				let mut out = Vec::new();
				futures_util::pin_mut!(stream);
				while let Some(chunk) = futures_util::StreamExt::next(&mut stream).await {
					out.extend_from_slice(&chunk?);
				}
				Ok(out)
			}
		}
	}
}

/// Boxed, sendable byte-chunk stream (upstream `ReadableStream<Uint8Array>`).
pub type BodyStream = Pin<Box<dyn futures_core::Stream<Item = Result<Vec<u8>, ProviderHttpError>> + Send>>;

/// Boxed sendable future.
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Error thrown by a fetch implementation. Carries the optional HTTP status,
/// response headers and raw body when the failure came from a completed
/// response, mirroring the SDK error objects that
/// [`crate::utils::error_body`] normalizes and
/// [`crate::utils::provider_retry`] classifies upstream (crate docs,
/// deviation D15).
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ProviderHttpError {
	pub message: String,
	/// HTTP status code when the failure came from a completed response.
	pub status: Option<u16>,
	/// Response headers when the failure came from a completed response.
	pub headers: Vec<(String, String)>,
	/// Raw response body text when the failure came from a completed response.
	pub body: Option<String>,
	#[source]
	pub source: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

impl ProviderHttpError {
	pub fn new(message: impl Into<String>) -> ProviderHttpError {
		ProviderHttpError {
			message: message.into(),
			status: None,
			headers: Vec::new(),
			body: None,
			source: None,
		}
	}

	pub fn with_source(
		message: impl Into<String>,
		source: Arc<dyn std::error::Error + Send + Sync>,
	) -> ProviderHttpError {
		ProviderHttpError {
			message: message.into(),
			status: None,
			headers: Vec::new(),
			body: None,
			source: Some(source),
		}
	}

	/// Error built from a completed (non-2xx) response (upstream SDK errors
	/// carrying `status` plus the raw body).
	pub fn from_response(message: impl Into<String>, status: u16, body: Option<String>) -> ProviderHttpError {
		ProviderHttpError {
			message: message.into(),
			status: Some(status),
			headers: Vec::new(),
			body,
			source: None,
		}
	}

	/// Attach response headers (case-insensitively looked up by
	/// [`ProviderHttpError::header`]).
	pub fn with_headers(mut self, headers: Vec<(String, String)>) -> ProviderHttpError {
		self.headers = headers;
		self
	}

	/// Case-insensitive single header lookup (upstream `Headers.get`).
	pub fn header(&self, name: &str) -> Option<&str> {
		self.headers
			.iter()
			.find(|(key, _)| key.eq_ignore_ascii_case(name))
			.map(|(_, value)| value.as_str())
	}
}

/// Outgoing HTTP request (upstream `fetch(url, init)`).
#[derive(Debug)]
pub struct HttpRequest {
	pub method: String,
	pub url: String,
	/// Header name/value pairs, insertion-ordered.
	pub headers: Vec<(String, String)>,
	pub body: HttpBody,
	/// HTTP request timeout in milliseconds (`options.timeoutMs`).
	pub timeout_ms: Option<u64>,
	/// Cooperative cancellation; adapters check it around connection and read
	/// steps.
	pub signal: Option<crate::utils::abort::AbortSignal>,
}

impl HttpRequest {
	pub fn new(method: impl Into<String>, url: impl Into<String>) -> HttpRequest {
		HttpRequest {
			method: method.into(),
			url: url.into(),
			headers: Vec::new(),
			body: HttpBody::Empty,
			timeout_ms: None,
			signal: None,
		}
	}

	pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> HttpRequest {
		self.headers.push((name.into(), value.into()));
		self
	}

	pub fn json_body(mut self, value: &serde_json::Value) -> HttpRequest {
		let bytes = serde_json::to_vec(value).expect("request body serializes");
		self.body = HttpBody::Bytes(bytes);
		self
	}
}

/// Incoming HTTP response (upstream `Response`).
#[derive(Debug)]
pub struct HttpResponse {
	pub status: u16,
	/// Header name/value pairs as received.
	pub headers: Vec<(String, String)>,
	pub body: HttpBody,
}

impl HttpResponse {
	/// Case-insensitive header lookup returning the first match.
	pub fn header(&self, name: &str) -> Option<&str> {
		self.headers
			.iter()
			.find(|(key, _)| key.eq_ignore_ascii_case(name))
			.map(|(_, value)| value.as_str())
	}

	/// Case-insensitive header lookup collecting every value.
	pub fn header_all(&self, name: &str) -> Vec<&str> {
		self.headers
			.iter()
			.filter(|(key, _)| key.eq_ignore_ascii_case(name))
			.map(|(_, value)| value.as_str())
			.collect()
	}

	/// Whether the status is 2xx (upstream `response.ok`).
	pub fn ok(&self) -> bool {
		(200..300).contains(&self.status)
	}
}

/// Upstream `FetchFunction`: sends a request, resolves to a response, and may
/// reject with [`ProviderHttpError`]. Cheap to clone.
#[derive(Clone)]
pub struct FetchFn(Arc<dyn Fn(HttpRequest) -> BoxFuture<Result<HttpResponse, ProviderHttpError>> + Send + Sync>);

impl FetchFn {
	pub fn new<F>(f: F) -> FetchFn
	where
		F: Fn(HttpRequest) -> BoxFuture<Result<HttpResponse, ProviderHttpError>> + Send + Sync + 'static,
	{
		FetchFn(Arc::new(f))
	}

	pub async fn send(&self, request: HttpRequest) -> Result<HttpResponse, ProviderHttpError> {
		(self.0)(request).await
	}
}

impl std::fmt::Debug for FetchFn {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str("FetchFn")
	}
}

impl Default for FetchFn {
	fn default() -> Self {
		default_fetch()
	}
}

/// The process-wide default adapter (`globalThis.fetch` equivalent).
pub fn default_fetch() -> FetchFn {
	static DEFAULT: LazyLock<FetchFn> = LazyLock::new(new_reqwest_fetch);
	DEFAULT.clone()
}

/// Build a fresh reqwest-backed adapter. Used by [`default_fetch`]; exposed so
/// callers can own dedicated clients.
pub fn new_reqwest_fetch() -> FetchFn {
	let client = reqwest::Client::builder()
		.build()
		.expect("reqwest client builds");
	FetchFn::new(move |request| {
		let client = client.clone();
		Box::pin(async move { send_with_reqwest(&client, request).await })
	})
}

async fn send_with_reqwest(
	client: &reqwest::Client,
	request: HttpRequest,
) -> Result<HttpResponse, ProviderHttpError> {
	let method = reqwest::Method::from_bytes(request.method.as_bytes())
		.map_err(|error| ProviderHttpError::with_source(format!("invalid HTTP method {}", request.method), Arc::new(error)))?;
	let mut builder = client.request(method, &request.url);
	for (name, value) in &request.headers {
		builder = builder.header(name, value);
	}
	if let Some(timeout_ms) = request.timeout_ms {
		builder = builder.timeout(std::time::Duration::from_millis(timeout_ms));
	}
	let body = request.body.collect().await?;
	builder = match body.len() {
		0 => builder,
		_ => builder.body(reqwest::Body::from(body)),
	};

	let send = builder.send();
	let response = tokio::select! {
		biased;
		_ = optional_abort(&request.signal), if request.signal.as_ref().is_some_and(|s| s.aborted()) => {
			return Err(ProviderHttpError::new("The operation was aborted"));
		}
		response = send => response.map_err(|error| ProviderHttpError::with_source("request failed".to_string(), Arc::new(error)))?,
	};

	let status = response.status().as_u16();
	let headers: Vec<(String, String)> = response
		.headers()
		.iter()
		.map(|(name, value)| (name.as_str().to_string(), value.to_str().unwrap_or_default().to_string()))
		.collect();
	let bytes_stream = response.bytes_stream();
	let body = HttpBody::stream(futures_util::StreamExt::map(bytes_stream, |chunk| {
		chunk
			.map(|bytes| bytes.to_vec())
			.map_err(|error| ProviderHttpError::with_source("response body read failed".to_string(), Arc::new(error)))
	}));

	Ok(HttpResponse { status, headers, body })
}

async fn optional_abort(signal: &Option<crate::utils::abort::AbortSignal>) {
	match signal {
		Some(signal) => {
			let _ = signal.aborted_future().await;
		}
		None => pending::<()>().await,
	}
}
