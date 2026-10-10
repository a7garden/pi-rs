//! Port of `packages/mcp/src/auth-provider.ts`.
//!
//! Also hosts the crate's minimal HTTP primitives (`HttpRequest`, `HttpResponse`,
//! `McpFetch`): upstream piggybacks on the platform `fetch`/`Response` types, which
//! do not exist in pure Rust. The default [`default_fetch`] implementation is backed
//! by `reqwest`; tests and applications may inject their own [`McpFetch`].

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use url::Url;

use crate::protocol::jsonrpc::McpError;

/// An ordered, case-insensitive header list (upstream `Headers`).
pub type HttpHeaders = Vec<(String, String)>;

/// Case-insensitive header lookup (upstream `Headers.get`).
pub fn header_find<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

/// Case-insensitive header set: replaces any existing values, like `Headers.set`.
pub fn header_set(headers: &mut HttpHeaders, name: &str, value: impl Into<String>) {
    headers.retain(|(key, _)| !key.eq_ignore_ascii_case(name));
    headers.push((name.to_string(), value.into()));
}

/// Header `content-type` up to the first `;`, trimmed and lowercased.
pub fn content_type_of(headers: &[(String, String)]) -> Option<String> {
    header_find(headers, "content-type")
        .and_then(|value| value.split(';').next())
        .map(|value| value.trim().to_ascii_lowercase())
}

/// A request body chunk stream (upstream `ReadableStream<Uint8Array>`).
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, McpError>> + Send>>;

/// An outbound HTTP request (upstream builds these inline on `fetch`, passing
/// `signal` in the init when the caller wants abortable requests).
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: String,
    pub url: Url,
    pub headers: HttpHeaders,
    pub body: Option<Vec<u8>>,
    /// Aborts the request (upstream `init.signal`). The default fetch races it against
    /// the response; custom fetch implementations decide how (or whether) to honor it.
    pub signal: Option<crate::client::AbortSignal>,
}

/// An inbound HTTP response; the body is a byte stream that may be consumed once.
pub struct HttpResponse {
    pub status: u16,
    pub headers: HttpHeaders,
    pub body: ByteStream,
}

impl std::fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .finish_non_exhaustive()
    }
}

impl HttpResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        header_find(&self.headers, name)
    }

    pub fn content_type(&self) -> Option<String> {
        content_type_of(&self.headers)
    }

    /// Upstream `response.text()`; read errors surface as [`McpError::Network`].
    pub async fn text(self) -> Result<String, McpError> {
        let bytes = self.bytes().await?;
        String::from_utf8(bytes).map_err(|e| McpError::Network(e.to_string()))
    }

    /// Read the whole body.
    pub async fn bytes(self) -> Result<Vec<u8>, McpError> {
        let mut out = Vec::new();
        let mut body = self.body;
        while let Some(chunk) = body.next().await {
            out.extend_from_slice(&chunk?);
        }
        Ok(out)
    }

    /// Upstream `response.json()`.
    pub async fn json(self) -> Result<serde_json::Value, McpError> {
        let text = self.text().await?;
        serde_json::from_str(&text).map_err(|e| McpError::Network(e.to_string()))
    }
}

/// Drop the response body without reading it (upstream `discard`).
pub fn discard(response: HttpResponse) {
    drop(response);
}

pub type FetchFuture = Pin<Box<dyn Future<Output = Result<HttpResponse, McpError>> + Send>>;

/// Upstream `McpFetch`: `(input, init) => Promise<Response>`.
pub type McpFetch = Arc<dyn Fn(HttpRequest) -> FetchFuture + Send + Sync>;

/// The default `fetch` implementation, backed by `reqwest`.
pub fn default_fetch() -> McpFetch {
    let client = reqwest::Client::new();
    Arc::new(move |request: HttpRequest| {
        let client = client.clone();
        Box::pin(async move {
            let method = reqwest::Method::from_bytes(request.method.as_bytes())
                .map_err(|e| McpError::Network(e.to_string()))?;
            let mut builder = client.request(method, request.url.as_str());
            for (name, value) in &request.headers {
                builder = builder.header(name.as_str(), value.as_str());
            }
            if let Some(body) = request.body {
                builder = builder.body(body);
            }
            let send = builder.send();
            // An already-aborted signal wins the race before the request is polled, so
            // nothing is sent (upstream fetch rejects an aborted init immediately).
            let response = match request.signal {
                Some(signal) => tokio::select! {
                    () = signal.cancelled() => {
                        return Err(McpError::abort("This operation was aborted"));
                    }
                    response = send => response.map_err(reqwest_error)?,
                },
                None => send.await.map_err(reqwest_error)?,
            };
            let status = response.status().as_u16();
            let headers: HttpHeaders = response
                .headers()
                .iter()
                .map(|(name, value)| {
                    (
                        name.as_str().to_string(),
                        value.to_str().unwrap_or_default().to_string(),
                    )
                })
                .collect();
            let body = response
                .bytes_stream()
                .map(|chunk| chunk.map_err(reqwest_error))
                .boxed();
            Ok(HttpResponse {
                status,
                headers,
                body,
            })
        })
    })
}

/// Upstream reports fetch failures (including a connection dropped mid-body) as
/// `TypeError`; they map to the always-retryable [`McpError::Network`].
pub(crate) fn reqwest_error(error: reqwest::Error) -> McpError {
    McpError::Network(error.to_string())
}

/// The 401 response — or a 403 response whose challenge reports `insufficient_scope` —
/// as handed to [`AuthProvider::on_unauthorized`].
#[derive(Debug, Clone)]
pub struct UnauthorizedResponse {
    pub status: u16,
    pub headers: HttpHeaders,
}

impl UnauthorizedResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        header_find(&self.headers, name)
    }
}

/// Upstream `UnauthorizedContext`.
#[derive(Clone)]
pub struct UnauthorizedContext {
    pub response: UnauthorizedResponse,
    pub server_url: Url,
    pub fetch: McpFetch,
    /// Access token the rejected request carried, if any. A different current token means another request already refreshed it.
    pub token: Option<String>,
}

impl std::fmt::Debug for UnauthorizedContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnauthorizedContext")
            .field("response", &self.response)
            .field("server_url", &self.server_url)
            .field("token", &self.token)
            .finish_non_exhaustive()
    }
}

/// Supplies bearer tokens to an MCP HTTP transport and may refresh them after a 401 response.
#[async_trait]
pub trait AuthProvider: Send + Sync {
    /// Token for the next request. May refresh it over the network first, so the transport calls it only
    /// before requests whose answer it waits for. Closing a session reuses the token of the last request.
    async fn token(&self) -> Result<Option<String>, McpError>;

    /// Whether [`AuthProvider::on_unauthorized`] is implemented. Mirrors upstream's
    /// optional `onUnauthorized?` method: when absent, a 401/403 challenge response
    /// is returned to the caller as-is instead of being retried.
    fn handles_unauthorized(&self) -> bool {
        false
    }

    /// Handle an authorization challenge; the request is retried afterwards.
    async fn on_unauthorized(&self, _context: UnauthorizedContext) -> Result<(), McpError> {
        Ok(())
    }
}
