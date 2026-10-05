//! Port of `packages/mcp/src/oauth/errors.ts`.
//!
//! Upstream defines one error class per file section; here they are small structs
//! wrapped as variants of the crate-wide [`McpError`](crate::protocol::jsonrpc::McpError),
//! with `Display` matching the upstream messages.

use thiserror::Error;

/// Upstream `OAuthError`: an OAuth protocol error code with description and URI.
#[derive(Debug, Error, Clone)]
#[error("{message}")]
pub struct OAuthError {
	pub code: String,
	pub message: String,
	pub error_uri: Option<String>,
}

impl OAuthError {
	/// Upstream `new OAuthError(code, message, errorUri)`: a missing message falls
	/// back to the code.
	pub fn new(code: impl Into<String>, message: impl Into<String>, error_uri: Option<String>) -> OAuthError {
		let code = code.into();
		let message = message.into();
		OAuthError {
			message: if message.is_empty() { code.clone() } else { message },
			code,
			error_uri,
		}
	}
}

/// Upstream `OAuthIssuerMismatchError`.
#[derive(Debug, Error, Clone)]
#[error("OAuth issuer mismatch: expected {}, received {}", quote(&expected), match received { None => "none".to_string(), Some(received) => quote(&received) })]
pub struct OAuthIssuerMismatchError {
	pub expected: String,
	/// `None` when an authorization response lacks the `iss` parameter its server promised (RFC 9207).
	pub received: Option<String>,
}

fn quote(value: &str) -> String {
	format!("{value:?}")
}

/// Upstream `OAuthInsecureEndpointError`.
#[derive(Debug, Error, Clone)]
#[error("Refusing to send OAuth credentials to non-HTTPS endpoint {endpoint}")]
pub struct OAuthInsecureEndpointError {
	pub endpoint: String,
}

/// Upstream `OAuthRegistrationError`.
#[derive(Debug, Error, Clone)]
#[error("OAuth dynamic client registration failed with status {status}: {body}")]
pub struct OAuthRegistrationError {
	pub status: u16,
	pub body: String,
}

/// Upstream `McpOAuthAuthorizationRequiredError`.
#[derive(Debug, Error, Clone)]
#[error("MCP OAuth authorization requires user interaction")]
pub struct McpOAuthAuthorizationRequiredError;
