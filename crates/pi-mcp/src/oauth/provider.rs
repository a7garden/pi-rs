//! Port of `packages/mcp/src/oauth/provider.ts`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use url::Url;

use crate::oauth::flow::InvalidationKind;
use crate::oauth::flow::OAuthClientMetadataDocument;
use crate::oauth::flow::OAuthClientProvider;
use crate::oauth::types::{OAuthClientInformationMixed, OAuthClientMetadata, OAuthDiscoveryState, OAuthTokens};
use crate::protocol::jsonrpc::McpError;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpOAuthState {
	pub server_url: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub client_information: Option<OAuthClientInformationMixed>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tokens: Option<OAuthTokens>,
	/// When the access token expires, in milliseconds since the epoch, from `expires_in` at the time it was saved.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tokens_expire_at: Option<u128>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub code_verifier: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub oauth_state: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub discovery: Option<OAuthDiscoveryState>,
}

#[async_trait::async_trait]
pub trait McpOAuthStateStore: Send + Sync {
	async fn load(&self) -> Result<Option<McpOAuthState>, McpError>;
	async fn save(&self, state: &McpOAuthState) -> Result<(), McpError>;
}

pub struct McpOAuthProviderOptions {
	pub server_url: Url,
	pub redirect_url: String,
	/// `redirect_uris` defaults to `[redirect_url]`; see upstream's `Omit<..., "redirect_uris">`.
	pub client_metadata: OAuthClientMetadata,
	/// See [`OAuthClientProvider::client_metadata_document`].
	pub client_metadata_document: Option<Arc<
		dyn Fn(Option<&crate::oauth::types::AuthorizationServerMetadata>) -> Option<OAuthClientMetadataDocument>
			+ Send
			+ Sync,
	>>,
	pub client_id: Option<String>,
	pub client_secret: Option<String>,
	pub store: Option<Arc<dyn McpOAuthStateStore>>,
	pub on_redirect:
		Arc<dyn Fn(Url) -> futures_util::future::BoxFuture<'static, Result<(), McpError>> + Send + Sync>,
}

/// Upstream `MemoryOAuthStateStore`.
#[derive(Default)]
pub struct MemoryOAuthStateStore {
	value: parking_lot::Mutex<Option<McpOAuthState>>,
}

impl MemoryOAuthStateStore {
	pub fn new() -> MemoryOAuthStateStore {
		MemoryOAuthStateStore::default()
	}
}

#[async_trait::async_trait]
impl McpOAuthStateStore for MemoryOAuthStateStore {
	async fn load(&self) -> Result<Option<McpOAuthState>, McpError> {
		Ok(self
			.value
			.lock()
			
			.clone())
	}

	async fn save(&self, state: &McpOAuthState) -> Result<(), McpError> {
		*self
			.value
			.lock()
			 = Some(state.clone());
		Ok(())
	}
}

/// Default stateful provider for one exact MCP server URL. Applications inject durable storage if needed.
pub struct McpOAuthProvider {
	redirect_url: String,
	client_metadata: OAuthClientMetadata,
	client_metadata_document: Option<Arc<
		dyn Fn(Option<&crate::oauth::types::AuthorizationServerMetadata>) -> Option<OAuthClientMetadataDocument>
			+ Send
			+ Sync,
	>>,
	server_url: String,
	configured_client: Option<OAuthClientInformationMixed>,
	store: Arc<dyn McpOAuthStateStore>,
	on_redirect:
		Arc<dyn Fn(Url) -> futures_util::future::BoxFuture<'static, Result<(), McpError>> + Send + Sync>,
	/// Serializes load-modify-save cycles (upstream chains writes on a promise).
	write_lock: tokio::sync::Mutex<()>,
}

impl McpOAuthProvider {
	pub fn new(options: McpOAuthProviderOptions) -> McpOAuthProvider {
		let server_url = options.server_url.to_string();
		let redirect_url = options.redirect_url.clone();
		let mut client_metadata = options.client_metadata;
		if client_metadata.redirect_uris.is_empty() {
			client_metadata.redirect_uris = vec![redirect_url.clone()];
		}
		if client_metadata.grant_types.is_none() {
			client_metadata.grant_types =
				Some(vec!["authorization_code".to_string(), "refresh_token".to_string()]);
		}
		if client_metadata.response_types.is_none() {
			client_metadata.response_types = Some(vec!["code".to_string()]);
		}
		if client_metadata.token_endpoint_auth_method.is_none() {
			client_metadata.token_endpoint_auth_method = Some(
				if options.client_secret.is_some() { "client_secret_post" } else { "none" }.to_string(),
			);
		}
		let configured_client = options.client_id.clone().map(|client_id| {
			OAuthClientInformationMixed::Plain(crate::oauth::types::OAuthClientInformation {
				client_id,
				client_secret: options.client_secret.clone(),
				client_id_issued_at: None,
				client_secret_expires_at: None,
			})
		});
		McpOAuthProvider {
			server_url,
			redirect_url,
			client_metadata,
			client_metadata_document: options.client_metadata_document,
			configured_client,
			store: options.store.unwrap_or_else(|| Arc::new(MemoryOAuthStateStore::new())),
			on_redirect: options.on_redirect,
			write_lock: tokio::sync::Mutex::new(()),
		}
	}

	async fn load(&self) -> Result<McpOAuthState, McpError> {
		// Upstream awaits queued writes before every read; the lock gives the same
		// serialization (reads included).
		let _guard = self.write_lock.lock().await;
		let state = self.store.load().await?;
		Ok(self.own(state))
	}

	async fn update(
		&self,
		update: impl FnOnce(McpOAuthState) -> McpOAuthState,
	) -> Result<(), McpError> {
		let _guard = self.write_lock.lock().await;
		let state = self.own(self.store.load().await?);
		self.store.save(&update(state)).await
	}

	/// Stored state for another server URL is ignored so credentials never leak across servers.
	fn own(&self, state: Option<McpOAuthState>) -> McpOAuthState {
		match state {
			Some(state) if state.server_url == self.server_url => state,
			_ => McpOAuthState { server_url: self.server_url.clone(), ..Default::default() },
		}
	}
}

#[async_trait::async_trait]
impl OAuthClientProvider for McpOAuthProvider {
	fn redirect_url(&self) -> String {
		self.redirect_url.clone()
	}

	fn client_metadata(&self) -> &OAuthClientMetadata {
		&self.client_metadata
	}

	async fn client_metadata_document(
		&self,
		metadata: Option<&crate::oauth::types::AuthorizationServerMetadata>,
	) -> Option<OAuthClientMetadataDocument> {
		let hook = self.client_metadata_document.as_ref()?;
		hook(metadata)
	}

	async fn state(&self) -> Result<Option<String>, McpError> {
		let existing = self.load().await?.oauth_state;
		if let Some(existing) = existing {
			return Ok(Some(existing));
		}
		let mut bytes = [0u8; 32];
		getrandom::getrandom(&mut bytes).map_err(|e| McpError::other(e.to_string()))?;
		let state = hex_encode(&bytes);
		self.update(|mut value| {
			value.oauth_state = Some(state.clone());
			value
		})
		.await?;
		Ok(Some(state))
	}

	async fn client_information(&self) -> Result<Option<OAuthClientInformationMixed>, McpError> {
		if let Some(configured) = &self.configured_client {
			return Ok(Some(configured.clone()));
		}
		Ok(self.load().await?.client_information)
	}

	async fn save_client_information(
		&self,
		information: &OAuthClientInformationMixed,
	) -> Result<(), McpError> {
		if self.configured_client.is_some() {
			return Ok(());
		}
		self.update(|mut value| {
			value.client_information = Some(information.clone());
			value
		})
		.await
	}

	async fn tokens(&self) -> Result<Option<OAuthTokens>, McpError> {
		Ok(self.load().await?.tokens)
	}

	async fn save_tokens(&self, tokens: &OAuthTokens) -> Result<(), McpError> {
		let expires_at = tokens
			.expires_in
			.map(|expires_in| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("epoch") + std::time::Duration::from_secs_f64(expires_in));
		self.update(|mut value| {
			value.tokens = Some(tokens.clone());
			match expires_at {
				None => value.tokens_expire_at = None,
				Some(expires_at) => value.tokens_expire_at = Some(expires_at.as_millis()),
			}
			value
		})
		.await
	}

	async fn redirect_to_authorization(&self, url: Url) -> Result<(), McpError> {
		(self.on_redirect)(url).await
	}

	async fn save_code_verifier(&self, verifier: &str) -> Result<(), McpError> {
		let verifier = verifier.to_string();
		self.update(|mut value| {
			value.code_verifier = Some(verifier.clone());
			value
		})
		.await
	}

	async fn code_verifier(&self) -> Result<String, McpError> {
		let verifier = self.load().await?.code_verifier;
		match verifier {
			Some(verifier) if !verifier.is_empty() => Ok(verifier),
			_ => Err(McpError::other("No OAuth PKCE code verifier is stored")),
		}
	}

	async fn invalidate_credentials(&self, kind: InvalidationKind) -> Result<(), McpError> {
		self.update(|mut value| {
			if matches!(kind, InvalidationKind::All | InvalidationKind::Client) {
				value.client_information = None;
			}
			if matches!(kind, InvalidationKind::All | InvalidationKind::Tokens) {
				value.tokens = None;
				value.tokens_expire_at = None;
			}
			if matches!(kind, InvalidationKind::All | InvalidationKind::Verifier) {
				value.code_verifier = None;
			}
			if matches!(kind, InvalidationKind::All | InvalidationKind::Discovery) {
				value.discovery = None;
			}
			if matches!(kind, InvalidationKind::All) {
				value.oauth_state = None;
			}
			value
		})
		.await
	}

	async fn save_discovery_state(&self, discovery: &OAuthDiscoveryState) -> Result<(), McpError> {
		let discovery = discovery.clone();
		self.update(|mut value| {
			value.discovery = Some(discovery.clone());
			value
		})
		.await
	}

	async fn discovery_state(&self) -> Result<Option<OAuthDiscoveryState>, McpError> {
		Ok(self.load().await?.discovery)
	}
}

fn hex_encode(bytes: &[u8]) -> String {
	let mut out = String::with_capacity(bytes.len() * 2);
	for byte in bytes {
		out.push_str(&format!("{byte:02x}"));
	}
	out
}
