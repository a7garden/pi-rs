//! Port of `@earendil-works/pi-mcp` (upstream: packages/mcp, v1.1.0).
//!
//! A standalone Model Context Protocol **client** (upstream ships client-only):
//! JSON-RPC 2.0 framing, MCP protocol types, tool-result content conversion, a
//! transport abstraction with stdio / Streamable HTTP / in-memory implementations,
//! and an OAuth 2.1 client flow (PKCE, discovery, dynamic registration, local
//! callback server).
//!
//! # Public API
//!
//! Mirrors upstream `src/index.ts`; the `./oauth` and `./testing` entries are the
//! [`oauth`] and [`testing`] modules.
//!
//! - Client: [`McpClient`], [`McpClientOptions`], [`RequestOptions`]
//! - JSON-RPC: [`JsonRpcMessage`], [`JsonRpcRequest`], [`JsonRpcNotification`],
//!   [`JsonRpcResponse`], [`parse_json_rpc_message`], [`JSON_RPC_ERROR_CODES_INTERNAL_ERROR`], ...
//! - Protocol types: [`Tool`], [`Resource`], [`ResourceTemplate`], [`CallToolResult`],
//!   [`ContentBlock`], [`to_llm_content`], [`LATEST_PROTOCOL_VERSION`], [`SUPPORTED_PROTOCOL_VERSIONS`], ...
//! - Transports: [`McpTransport`] trait, [`StdioTransport`], [`StreamableHttpTransport`],
//!   [`InMemoryTransport`] / [`create_in_memory_transport_pair`] (also under [`testing`]),
//!   [`consume_sse_stream`]
//! - Auth: [`AuthProvider`], [`McpFetch`], [`OAuthClientProvider`], [`authorize_mcp`],
//!   [`McpOAuthProvider`], [`OAuthCallbackServer`], ...
//!
//! # Port deviations
//!
//! - **Error classes → one enum.** Upstream error classes (`McpError`,
//!   `McpConnectionClosedError`, `McpTimeoutError`, `McpAbortError`, `McpHttpError`,
//!   `McpAuthRequiredError`, `McpSessionExpiredError`, the `oauth/errors.ts` classes)
//!   map to variants of [`McpError`] (payload structs keep the upstream names and
//!   `Display` strings). All fallible APIs are `Result`-based.
//! - **Fetch abstraction.** Upstream uses the platform `fetch`/`Response` types;
//!   Rust gets neutral [`HttpRequest`]/[`HttpResponse`] values behind [`McpFetch`],
//!   with a `reqwest`-backed default. Abortable requests carry the crate-local
//!   signal on `HttpRequest.signal`; the default fetch races it against the
//!   response and reports aborts as [`McpError::Abort`] (upstream `AbortError`),
//!   while custom fetch implementations decide how to honor it. The
//!   Cloudflare-receiver concern of upstream test #10188 ("calls fetch without a
//!   receiver") does not exist here; the test keeps the custom-fetch +
//!   `context.fetch` assertions only.
//! - **AbortSignal/AbortController.** Platform signal objects become the crate-local
//!   [`AbortSignal`]/[`AbortController`] (watch-based); pi-runtime can adapt its own.
//! - **Optional provider/handler methods → trait defaults.** Upstream probes
//!   `provider.method?`; Rust traits expose defaults, with presence probes where
//!   behavior differs (`AuthProvider::handles_unauthorized`).
//! - **Flow/provider entry points take `Arc<dyn OAuthClientProvider>`** so the
//!   optional client-authentication hook routes through the shared trait object.
//! - **Stdio.** `cross-spawn` → `tokio::process`; the child runs in its own process
//!   group (Unix) and tree-kill signals the group; Windows falls back to the direct
//!   child (upstream shells out to `taskkill /T /F`). The host-exit safety net uses
//!   `atexit(3)` (Unix) instead of `process.once("exit")`. The `stdio` test fixture
//!   is a Rust bin (`mcp-stdio-fixture`) spawned via `CARGO_BIN_EXE_*` instead of
//!   `node fixtures/stdio-server.mjs`; upstream does have a stubborn fixture, so the
//!   `mcp-stubborn-fixture` bin mirrors it.
//! - **OAuth callback server.** `node:http` → a minimal tokio HTTP/1.1 GET server.
//! - **Test timers.** vitest fake timers → `tokio` paused clocks where determinism
//!   is needed; otherwise real short timeouts.
//! - **JSON fidelity.** Wire types are serde structs; `params`/`result` stay
//!   `serde_json::Value` passthrough, unknown fields are preserved with
//!   `#[serde(flatten)]` where upstream spreads objects, and `serde_json` uses
//!   `preserve_order`. Null-valued *unknown* OAuth metadata fields are dropped
//!   where upstream's `compact` only drops `undefined` (JSON has no `undefined`).
//! - **Structural validation.** OAuth metadata validation ports the upstream
//!   hand-rolled checks (required/optional strings, URL scheme checks, JS `Number`
//!   coercion for `expires_in`).
//! - **SSE decoding.** Upstream decodes incrementally with `TextDecoder`; lines are
//!   split on `\n` at the byte level first (equivalent for well-formed UTF-8, then
//!   lossy per-line decode).
//! - **Listener panics.** Upstream isolates listener exceptions into error
//!   callbacks; Rust listener panics propagate (panics are bugs).

pub mod auth_provider;
pub mod client;
pub mod oauth;
pub mod protocol;
pub mod testing;
pub mod transports;

// Mirrors upstream `src/index.ts` re-exports.
pub use auth_provider::{AuthProvider, McpFetch, UnauthorizedContext};
pub use client::{
    AbortController, AbortSignal, ClientState, McpClient, McpClientOptions, McpRequestOptions,
    RequestContext, Roots, RootsProvider,
};
pub use protocol::content::{
    to_llm_content, AudioContent, BlobResourceContents, CallToolResult, ContentAnnotations,
    ContentBlock, EmbeddedResourceContent, ImageContent, LlmContent, ResourceLinkContent,
    TextContent, TextResourceContents,
};
pub use protocol::jsonrpc::{
    is_json_rpc_notification, is_json_rpc_request, is_json_rpc_response, parse_json_rpc_message,
    JsonRpcErrorObject, JsonRpcErrorResponse, JsonRpcId, JsonRpcMessage, JsonRpcNotification,
    JsonRpcRequest, JsonRpcResponse, JsonRpcSuccessResponse, McpError,
    JSON_RPC_ERROR_CODES_INTERNAL_ERROR, JSON_RPC_ERROR_CODES_INVALID_PARAMS,
    JSON_RPC_ERROR_CODES_INVALID_REQUEST, JSON_RPC_ERROR_CODES_METHOD_NOT_FOUND,
    JSON_RPC_ERROR_CODES_PARSE_ERROR,
};
pub use protocol::types::{
    CancelledNotification, ClientCapabilities, Implementation, InitializeParams, InitializeResult,
    ListResourceTemplatesResult, ListResourcesResult, ListToolsResult, ProgressNotification,
    ReadResourceResult, Resource, ResourceTemplate, Root, ServerCapabilities, Tool,
    ToolAnnotations, ToolExecution, LATEST_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS,
};
pub use transports::stdio::{StdioMode, StdioTransport, StdioTransportOptions};
pub use transports::streamable_http::{
    consume_sse_stream, AuthRequiredError, ConsumeSseOptions, HttpError, SessionExpiredError,
    SseEvent, StreamableHttpReconnectOptions, StreamableHttpTransport,
    StreamableHttpTransportOptions,
};
pub use transports::transport::{
    CloseListener, ErrorListener, McpTransport, MessageListener, TransportEvents, Unsubscribe,
};
pub use transports::{in_memory::create_in_memory_transport_pair, in_memory::InMemoryTransport};

/// Upstream `JSON_RPC_ERROR_CODES` object, flattened into constants.
pub mod json_rpc_error_codes {
    pub use crate::protocol::jsonrpc::{
        JSON_RPC_ERROR_CODES_INTERNAL_ERROR as INTERNAL_ERROR,
        JSON_RPC_ERROR_CODES_INVALID_PARAMS as INVALID_PARAMS,
        JSON_RPC_ERROR_CODES_INVALID_REQUEST as INVALID_REQUEST,
        JSON_RPC_ERROR_CODES_METHOD_NOT_FOUND as METHOD_NOT_FOUND,
        JSON_RPC_ERROR_CODES_PARSE_ERROR as PARSE_ERROR,
    };
}
