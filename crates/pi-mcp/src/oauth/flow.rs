//! Port of `packages/mcp/src/oauth/flow.ts`.
//!
//! Adapted from modelcontextprotocol/typescript-sdk v1.29.0 src/client/auth.ts.
//! Modified to remove SDK/Zod dependencies and use sha2 + getrandom for PKCE
//! (upstream comment). Optional provider methods become trait defaults; absence
//! is observable where upstream checks `provider.method?`. Deviation: flow
//! entry points take `Arc<dyn OAuthClientProvider>` so the client-authentication
//! hook can be routed through the shared provider trait object.

use std::sync::Arc;

use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use base64::Engine;
use futures_util::future::{BoxFuture, Shared};
use futures_util::FutureExt;
use sha2::{Digest, Sha256};
use url::Url;

use crate::auth_provider::{
	default_fetch, AuthProvider, HttpRequest, McpFetch, UnauthorizedContext,
};
use crate::oauth::discovery::{
	discover_authorization_server_metadata, discover_oauth_server_info, parse_www_authenticate,
	select_resource, DiscoverAuthorizationServerOptions, DiscoverServerInfoOptions,
};
use crate::oauth::errors::{
	McpOAuthAuthorizationRequiredError, OAuthError, OAuthInsecureEndpointError,
	OAuthIssuerMismatchError, OAuthRegistrationError,
};
use crate::oauth::types::{
	parse_client_information, parse_oauth_tokens, AuthorizationServerMetadata, OAuthClientInformation,
	OAuthClientInformationFull, OAuthClientInformationMixed, OAuthClientMetadata, OAuthDiscoveryState,
	OAuthServerInfo, OAuthTokens,
};
use crate::protocol::jsonrpc::{is_object, McpError};

/// Upstream `AddClientAuthentication` params type (`URLSearchParams`).
pub type FormParams = Vec<(String, String)>;

/// Upstream `AddClientAuthentication`: custom client authentication hook. Returns
/// `Ok(false)` when the hook is absent/no-op (standard client authentication is
/// applied instead), `Ok(true)` after applying its own.
pub type AddClientAuthentication = Arc<
	dyn for<'a> Fn(
			&'a mut Vec<(String, String)>,
			&'a mut FormParams,
			&'a Url,
			Option<&'a AuthorizationServerMetadata>,
		) -> BoxFuture<'a, Result<bool, McpError>>
		+ Send
		+ Sync,
>;

/// A Client ID Metadata Document: an https URL used as `client_id`, and a redirect URI it lists.
#[derive(Debug, Clone, PartialEq)]
pub struct OAuthClientMetadataDocument {
	pub url: String,
	pub redirect_url: String,
}

/// Upstream `OAuthClientProvider`. Methods that are optional upstream have no-op or
/// fallible defaults; `client_information`, `tokens`, `save_tokens`,
/// `redirect_to_authorization`, `save_code_verifier` and `code_verifier` are required.
#[async_trait::async_trait]
pub trait OAuthClientProvider: Send + Sync {
	fn redirect_url(&self) -> String;
	fn client_metadata(&self) -> &OAuthClientMetadata;

	/// Upstream `clientMetadataDocument?`: identify as a Client ID Metadata Document
	/// instead of registering dynamically. `metadata` is `None` when the authorization
	/// server has none; check `client_id_metadata_document_supported`.
	async fn client_metadata_document(
		&self,
		_metadata: Option<&AuthorizationServerMetadata>,
	) -> Option<OAuthClientMetadataDocument> {
		None
	}

	/// Upstream `state?()`.
	async fn state(&self) -> Result<Option<String>, McpError> {
		Ok(None)
	}

	async fn client_information(&self) -> Result<Option<OAuthClientInformationMixed>, McpError>;

	/// Upstream `saveClientInformation?`; the default reports "cannot be persisted",
	/// matching the flow's error when the method is missing.
	async fn save_client_information(
		&self,
		_information: &OAuthClientInformationMixed,
	) -> Result<(), McpError> {
		Err(McpError::other("OAuth client information cannot be persisted"))
	}

	async fn tokens(&self) -> Result<Option<OAuthTokens>, McpError>;
	async fn save_tokens(&self, tokens: &OAuthTokens) -> Result<(), McpError>;
	async fn redirect_to_authorization(&self, url: Url) -> Result<(), McpError>;
	async fn save_code_verifier(&self, verifier: &str) -> Result<(), McpError>;
	async fn code_verifier(&self) -> Result<String, McpError>;

	/// Upstream `addClientAuthentication?`. The default (absent) applies the standard
	/// client authentication; an override applies its own and returns `Ok(true)`.
	async fn add_client_authentication(
		&self,
		_headers: &mut Vec<(String, String)>,
		_params: &mut FormParams,
		_url: &Url,
		_metadata: Option<&AuthorizationServerMetadata>,
	) -> Result<bool, McpError> {
		Ok(false)
	}

	/// Upstream `invalidateCredentials?`.
	async fn invalidate_credentials(&self, _kind: InvalidationKind) -> Result<(), McpError> {
		Ok(())
	}

	/// Upstream `saveDiscoveryState?`.
	async fn save_discovery_state(&self, _state: &OAuthDiscoveryState) -> Result<(), McpError> {
		Ok(())
	}

	/// Upstream `discoveryState?`.
	async fn discovery_state(&self) -> Result<Option<OAuthDiscoveryState>, McpError> {
		Ok(None)
	}
}

/// Upstream `invalidateCredentials` kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidationKind {
	All,
	Client,
	Tokens,
	Verifier,
	Discovery,
}

#[derive(Clone)]
pub struct OAuthFlowOptions {
	pub server_url: Url,
	pub authorization_code: Option<String>,
	/// `iss` parameter of the authorization response that delivered `authorizationCode` (RFC 9207).
	pub iss: Option<String>,
	pub scope: Option<String>,
	pub resource_metadata_url: Option<Url>,
	/// Authorization server metadata document to use instead of discovery, for servers that advertise a
	/// wrong authorization server or none. It is trusted as configured. Must use https, except on loopback.
	pub authorization_server_metadata_url: Option<Url>,
	pub fetch: Option<McpFetch>,
	pub skip_issuer_validation: bool,
	/// Go straight to the authorization redirect instead of refreshing stored tokens, for example when the
	/// server asks for scopes the current grant lacks (a refresh keeps the old scope).
	pub skip_refresh: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OAuthFlowResult {
	Authorized,
	Redirect,
}

#[derive(Clone, Default)]
pub struct TokenRequestOptions {
	pub metadata: Option<AuthorizationServerMetadata>,
	pub client_information: Option<OAuthClientInformationMixed>,
	pub resource: Option<String>,
	pub add_client_authentication: Option<AddClientAuthentication>,
	pub fetch: Option<McpFetch>,
}

fn loopback(hostname: &str) -> bool {
	matches!(hostname, "localhost" | "127.0.0.1" | "[::1]" | "::1")
}

fn secure_endpoint(value: &Url) -> Result<Url, McpError> {
	if value.scheme() != "https" && !loopback(value.host_str().unwrap_or_default()) {
		return Err(McpError::InsecureEndpoint(OAuthInsecureEndpointError {
			endpoint: value.to_string(),
		}));
	}
	Ok(value.clone())
}

fn parse_url(value: &str) -> Result<Url, McpError> {
	Url::parse(value).map_err(|e| McpError::other(e.to_string()))
}

fn select_client_auth_method(
	information: &OAuthClientInformationMixed,
	supported: &[String],
) -> &'static str {
	let hinted = information.token_endpoint_auth_method().filter(|hint| {
		matches!(*hint, "client_secret_basic" | "client_secret_post" | "none")
			&& (supported.is_empty() || supported.iter().any(|s| s == hint))
	});
	if let Some(hint) = hinted {
		return match hint {
			"client_secret_post" => "client_secret_post",
			"none" => "none",
			_ => "client_secret_basic",
		};
	}
	let has_secret = information.client_secret().is_some();
	if supported.is_empty() {
		return if has_secret { "client_secret_basic" } else { "none" };
	}
	if has_secret && supported.iter().any(|s| s == "client_secret_basic") {
		return "client_secret_basic";
	}
	if has_secret && supported.iter().any(|s| s == "client_secret_post") {
		return "client_secret_post";
	}
	if supported.iter().any(|s| s == "none") {
		return "none";
	}
	if has_secret { "client_secret_post" } else { "none" }
}

fn apply_client_authentication(
	method: &str,
	information: &OAuthClientInformationMixed,
	headers: &mut Vec<(String, String)>,
	params: &mut FormParams,
) -> Result<(), McpError> {
	if method == "client_secret_basic" {
		let Some(client_secret) = information.client_secret() else {
			return Err(McpError::other("client_secret_basic requires a client secret"));
		};
		let credentials = format!("{}:{}", information.client_id(), client_secret);
		headers.retain(|(name, _)| !name.eq_ignore_ascii_case("Authorization"));
		headers.push(("Authorization".to_string(), format!("Basic {}", STANDARD.encode(credentials.as_bytes()))));
	} else {
		params.push(("client_id".to_string(), information.client_id().to_string()));
		if method == "client_secret_post" {
			if let Some(client_secret) = information.client_secret() {
				params.push(("client_secret".to_string(), client_secret.to_string()));
			}
		}
	}
	Ok(())
}

/// Upstream `pkce()`: 32 random bytes, base64url verifier, S256 challenge.
async fn pkce() -> Result<(String, String), McpError> {
	let mut bytes = [0u8; 32];
	getrandom::getrandom(&mut bytes).map_err(|e| McpError::other(e.to_string()))?;
	let verifier = URL_SAFE_NO_PAD.encode(bytes);
	let digest = Sha256::digest(verifier.as_bytes());
	let challenge = URL_SAFE_NO_PAD.encode(digest);
	Ok((verifier, challenge))
}

pub struct StartAuthorization {
	pub authorization_url: Url,
	pub code_verifier: String,
}

pub struct StartAuthorizationOptions<'a> {
	pub metadata: Option<&'a AuthorizationServerMetadata>,
	pub client_information: &'a OAuthClientInformationMixed,
	pub redirect_url: String,
	pub scope: Option<&'a str>,
	pub state: Option<&'a str>,
	pub resource: Option<&'a str>,
}

/// Upstream `startAuthorization`.
pub async fn start_authorization(
	authorization_server_url: &Url,
	options: StartAuthorizationOptions<'_>,
) -> Result<StartAuthorization, McpError> {
	let metadata = options.metadata;
	if let Some(metadata) = metadata {
		if !metadata.response_types_supported.iter().any(|t| t == "code") {
			return Err(McpError::other("Authorization server does not support authorization codes"));
		}
		if metadata
			.code_challenge_methods_supported
			.as_ref()
			.map(|methods| !methods.iter().any(|m| m == "S256"))
			.unwrap_or(false)
		{
			return Err(McpError::other("Authorization server does not support PKCE S256"));
		}
	}
	let mut url = match metadata {
		Some(metadata) => parse_url(&metadata.authorization_endpoint)?,
		None => authorization_server_url.join("/authorize").map_err(|e| McpError::other(e.to_string()))?,
	};
	let (verifier, challenge) = pkce().await?;
	{
		let mut query = url.query_pairs_mut();
		query.append_pair("response_type", "code");
		query.append_pair("client_id", options.client_information.client_id());
		query.append_pair("code_challenge", &challenge);
		query.append_pair("code_challenge_method", "S256");
		query.append_pair("redirect_uri", options.redirect_url.as_str());
		if let Some(state) = options.state {
			query.append_pair("state", state);
		}
		if let Some(scope) = options.scope {
			query.append_pair("scope", scope);
		}
		if options
			.scope
			.map(|scope| scope.split_whitespace().any(|part| part == "offline_access"))
			.unwrap_or(false)
		{
			query.append_pair("prompt", "consent");
		}
		if let Some(resource) = options.resource {
			query.append_pair("resource", resource);
		}
	}
	Ok(StartAuthorization { authorization_url: url, code_verifier: verifier })
}

fn form_urlencoded(params: &[(String, String)]) -> String {
	let mut serializer = url::form_urlencoded::Serializer::new(String::new());
	for (name, value) in params {
		serializer.append_pair(name, value);
	}
	serializer.finish()
}

async fn token_request(
	authorization_server_url: &Url,
	options: &TokenRequestOptions,
	mut params: FormParams,
) -> Result<OAuthTokens, McpError> {
	let url = match &options.metadata {
		Some(metadata) => secure_endpoint(&parse_url(&metadata.token_endpoint)?)?,
		None => secure_endpoint(
			&authorization_server_url
				.join("/token")
				.map_err(|e| McpError::other(e.to_string()))?,
		)?,
	};
	let mut headers: Vec<(String, String)> = vec![
		("Accept".to_string(), "application/json".to_string()),
		("content-type".to_string(), "application/x-www-form-urlencoded".to_string()),
	];
	if let Some(resource) = &options.resource {
		params.push(("resource".to_string(), resource.clone()));
	}
	let Some(client_information) = &options.client_information else {
		return Err(McpError::other("OAuth client information is required"));
	};
	let mut applied_custom = false;
	if let Some(add) = &options.add_client_authentication {
		applied_custom = add(&mut headers, &mut params, &url, options.metadata.as_ref()).await?;
	}
	if !applied_custom {
		let supported = options
			.metadata
			.as_ref()
			.and_then(|metadata| metadata.token_endpoint_auth_methods_supported.clone())
			.unwrap_or_default();
		apply_client_authentication(
			select_client_auth_method(client_information, &supported),
			client_information,
			&mut headers,
			&mut params,
		)?;
	}
	let fetch = options.fetch.clone().unwrap_or_else(default_fetch);
	let request = HttpRequest {
		method: "POST".to_string(),
		url: url.clone(),
		headers,
		body: Some(form_urlencoded(&params).into_bytes()),
	};
	let response = fetch(request).await?;
	let status = response.status;
	let text = response.text().await?;
	let value: serde_json::Value = serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
	// Servers may report OAuth errors with any status, so check the body before the status.
	if is_object(&value) {
		if let Some(error) = value.get("error").and_then(serde_json::Value::as_str) {
			let description = value
				.get("error_description")
				.and_then(serde_json::Value::as_str)
				.map(str::to_string)
				.unwrap_or_else(|| error.to_string());
			let error_uri = value
				.get("error_uri")
				.and_then(serde_json::Value::as_str)
				.map(str::to_string);
			return Err(McpError::OAuth(OAuthError::new(error, description, error_uri)));
		}
	}
	if !(200..300).contains(&status) {
		return Err(McpError::OAuth(OAuthError::new(
			"server_error",
			format!("HTTP {status}: {text}"),
			None,
		)));
	}
	parse_oauth_tokens(value)
}

/// Upstream `registerClient`.
pub async fn register_client(
	authorization_server_url: &Url,
	options: RegisterClientOptions<'_>,
) -> Result<OAuthClientInformationFull, McpError> {
	let endpoint = match options.metadata {
		Some(metadata) => match &metadata.registration_endpoint {
			Some(endpoint) => Some(endpoint.clone()),
			None => {
				return Err(McpError::other(
					"Authorization server does not support dynamic client registration",
				))
			}
		},
		None => None,
	};
	let url = match endpoint {
		Some(endpoint) => parse_url(&endpoint)?,
		None => authorization_server_url
			.join("/register")
			.map_err(|e| McpError::other(e.to_string()))?,
	};
	let mut body = serde_json::to_value(options.client_metadata).map_err(|e| McpError::other(e.to_string()))?;
	if let Some(scope) = options.scope {
		body["scope"] = serde_json::json!(scope);
	}
	let fetch = options.fetch.clone().unwrap_or_else(default_fetch);
	let request = HttpRequest {
		method: "POST".to_string(),
		url,
		headers: vec![
			("Accept".to_string(), "application/json".to_string()),
			("content-type".to_string(), "application/json".to_string()),
		],
		body: Some(serde_json::to_vec(&body).map_err(|e| McpError::other(e.to_string()))?),
	};
	let response = fetch(request).await?;
	let status = response.status;
	let text = response.text().await?;
	if !(200..300).contains(&status) {
		return Err(McpError::Registration(OAuthRegistrationError { status, body: text }));
	}
	let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| McpError::other(e.to_string()))?;
	parse_client_information(value)
}

pub struct RegisterClientOptions<'a> {
	pub metadata: Option<&'a AuthorizationServerMetadata>,
	pub client_metadata: &'a OAuthClientMetadata,
	pub scope: Option<&'a str>,
	pub fetch: Option<McpFetch>,
}

pub struct ExchangeOptions {
	pub token: TokenRequestOptions,
	pub code: String,
	pub code_verifier: String,
	pub redirect_url: String,
}

/// Upstream `exchangeAuthorizationCode`.
pub async fn exchange_authorization_code(
	authorization_server_url: &Url,
	options: ExchangeOptions,
) -> Result<OAuthTokens, McpError> {
	let mut params = FormParams::new();
	params.push(("grant_type".to_string(), "authorization_code".to_string()));
	params.push(("code".to_string(), options.code));
	params.push(("code_verifier".to_string(), options.code_verifier));
	params.push(("redirect_uri".to_string(), options.redirect_url));
	token_request(authorization_server_url, &options.token, params).await
}

pub struct RefreshOptions {
	pub token: TokenRequestOptions,
	pub refresh_token: String,
}

/// Upstream `refreshAuthorization`: a response without `refresh_token` keeps the
/// exchanged one (`{ refresh_token: options.refreshToken, ...tokens }`).
pub async fn refresh_authorization(
	authorization_server_url: &Url,
	options: RefreshOptions,
) -> Result<OAuthTokens, McpError> {
	let mut params = FormParams::new();
	params.push(("grant_type".to_string(), "refresh_token".to_string()));
	params.push(("refresh_token".to_string(), options.refresh_token.clone()));
	let mut tokens = token_request(authorization_server_url, &options.token, params).await?;
	if tokens.refresh_token.is_none() {
		tokens.refresh_token = Some(options.refresh_token);
	}
	Ok(tokens)
}

fn with_scope(tokens: OAuthTokens, scope: Option<&str>) -> OAuthTokens {
	if tokens.scope.is_none() {
		if let Some(scope) = scope {
			let mut tokens = tokens;
			tokens.scope = Some(scope.to_string());
			return tokens;
		}
	}
	tokens
}

/// Scopes for a step-up authorization: the challenged scopes plus the ones granted so far, since a
/// challenge may list only the missing scopes and a token with just those would lose access the old
/// one had (SEP-2350). Without challenged scopes, `None` lets the flow pick its default.
pub fn step_up_scope(granted: Option<&str>, challenged: Option<&str>) -> Option<String> {
	let challenged = challenged?;
	let mut scopes: Vec<String> = Vec::new();
	for source in [granted, Some(challenged)] {
		if let Some(scope) = source {
			for part in scope.split_whitespace().filter(|part| !part.is_empty()) {
				if !scopes.iter().any(|existing| existing == part) {
					scopes.push(part.to_string());
				}
			}
		}
	}
	if scopes.is_empty() {
		None
	} else {
		Some(scopes.join(" "))
	}
}

/// Route the provider's optional `addClientAuthentication` trait method through the
/// `TokenRequestOptions` hook (upstream passes `provider.addClientAuthentication`,
/// possibly `undefined`).
fn provider_client_auth(provider: &Arc<dyn OAuthClientProvider>) -> AddClientAuthentication {
	let provider = Arc::clone(provider);
	Arc::new(
		move |headers: &mut Vec<(String, String)>,
		      params: &mut FormParams,
		      url: &Url,
		      metadata: Option<&AuthorizationServerMetadata>| {
			let provider = Arc::clone(&provider);
			Box::pin(async move {
				provider.add_client_authentication(headers, params, url, metadata).await
			}) as BoxFuture<'_, Result<bool, McpError>>
		},
	)
}

async fn run_flow(
	provider: &Arc<dyn OAuthClientProvider>,
	options: &OAuthFlowOptions,
) -> Result<OAuthFlowResult, McpError> {
	let metadata_url = match &options.authorization_server_metadata_url {
		Some(url) => Some(secure_endpoint(url)?),
		None => None,
	};
	// With a configured metadata URL, discovery is not cached, so changing the URL applies at once.
	let cached = if metadata_url.is_none() { provider.discovery_state().await? } else { None };
	let discovered = match cached.filter(|cached| !cached.authorization_server_url.is_empty()) {
		Some(cached) => {
			let authorization_server_metadata = match cached.authorization_server_metadata {
				Some(metadata) => Some(metadata),
				None => {
					discover_authorization_server_metadata(
						parse_url(&cached.authorization_server_url)?,
						DiscoverAuthorizationServerOptions {
							fetch: options.fetch.as_ref(),
							protocol_version: None,
							skip_issuer_validation: options.skip_issuer_validation,
						},
					)
					.await?
				}
			};
			OAuthServerInfo {
				authorization_server_url: cached.authorization_server_url,
				authorization_server_metadata,
				resource_metadata: cached.resource_metadata,
			}
		}
		None => {
			discover_oauth_server_info(
				options.server_url.clone(),
				DiscoverServerInfoOptions {
					resource_metadata_url: options.resource_metadata_url.clone(),
					authorization_server_metadata_url: metadata_url.clone(),
					fetch: options.fetch.as_ref(),
					skip_issuer_validation: options.skip_issuer_validation,
				},
			)
			.await?
		}
	};
	if metadata_url.is_none() {
		provider
			.save_discovery_state(&OAuthDiscoveryState {
				authorization_server_url: discovered.authorization_server_url.clone(),
				authorization_server_metadata: discovered.authorization_server_metadata.clone(),
				resource_metadata: discovered.resource_metadata.clone(),
				resource_metadata_url: options.resource_metadata_url.as_ref().map(Url::to_string),
			})
			.await?;
	}
	let metadata = discovered.authorization_server_metadata.as_ref();
	let resource = select_resource(&options.server_url, discovered.resource_metadata.as_ref())?;
	// `||`, not `??`: an empty scope (for example from `scopes_supported: []`) falls through to the next source.
	let scope = options
		.scope
		.clone()
		.filter(|scope| !scope.is_empty())
		.or_else(|| {
			discovered
				.resource_metadata
				.as_ref()
				.and_then(|resource| resource.scopes_supported.as_ref())
				.map(|scopes| scopes.join(" "))
		})
		.filter(|scope| !scope.is_empty())
		.or_else(|| provider.client_metadata().scope.clone())
		.filter(|scope| !scope.is_empty());
	let stored = provider.client_information().await?;
	let client_document = match &stored {
		None => provider.client_metadata_document(metadata).await,
		Some(_) => None,
	};
	if let Some(document) = &client_document {
		let url = parse_url(&document.url)?;
		if url.scheme() != "https" || url.path() == "/" {
			return Err(McpError::other("Invalid OAuth client metadata URL"));
		}
	}
	let client = match (&stored, &client_document) {
		(Some(stored), _) => Some(stored.clone()),
		(None, Some(document)) => Some(OAuthClientInformationMixed::Plain(OAuthClientInformation {
			client_id: document.url.clone(),
			client_secret: None,
			client_id_issued_at: None,
			client_secret_expires_at: None,
		})),
		(None, None) => None,
	};
	let client = match client {
		Some(client) => client,
		None => {
			if options.authorization_code.is_some() {
				return Err(McpError::other("OAuth client information is missing during code exchange"));
			}
			let registered = register_client(
				&parse_url(&discovered.authorization_server_url)?,
				RegisterClientOptions {
					metadata,
					client_metadata: provider.client_metadata(),
					scope: scope.as_deref(),
					fetch: options.fetch.clone(),
				},
			)
			.await?;
			let registered = OAuthClientInformationMixed::Full(registered);
			provider.save_client_information(&registered).await?;
			registered
		}
	};
	// The document's redirect URI may differ from the provider's, for example by a server-specific path.
	let redirect_url = client_document
		.as_ref()
		.map(|document| document.redirect_url.clone())
		.unwrap_or_else(|| provider.redirect_url());
	let token_options = TokenRequestOptions {
		metadata: metadata.cloned(),
		client_information: Some(client),
		resource,
		add_client_authentication: Some(provider_client_auth(provider)),
		fetch: options.fetch.clone(),
	};
	if let Some(authorization_code) = &options.authorization_code {
		// RFC 9207: never send a code from another authorization server to this one.
		if let Some(metadata) = metadata {
			if options.iss.is_some() || metadata.authorization_response_iss_parameter_supported == Some(true) {
				if options.iss.as_deref() != Some(metadata.issuer.as_str()) {
					return Err(McpError::IssuerMismatch(OAuthIssuerMismatchError {
						expected: metadata.issuer.clone(),
						received: options.iss.clone(),
					}));
				}
			}
		}
		let code_verifier = provider.code_verifier().await?;
		let tokens = exchange_authorization_code(
			&parse_url(&discovered.authorization_server_url)?,
			ExchangeOptions {
				token: token_options,
				code: authorization_code.clone(),
				code_verifier,
				redirect_url,
			},
		)
		.await?;
		// A response without `scope` grants the requested scope (RFC 6749 §5.1). Recorded so a step-up can
		// keep it. Callers pass the options of the authorization request, so `scope` is what was requested.
		provider.save_tokens(&with_scope(tokens, scope.as_deref())).await?;
		return Ok(OAuthFlowResult::Authorized);
	}
	let existing = if options.skip_refresh { None } else { provider.tokens().await? };
	if let Some(existing) = existing.filter(|existing| existing.refresh_token.is_some()) {
		let result = refresh_authorization(
			&parse_url(&discovered.authorization_server_url)?,
			RefreshOptions {
				token: token_options.clone(),
				refresh_token: existing.refresh_token.clone().expect("filtered"),
			},
		)
		.await;
		match result {
			Ok(tokens) => {
				// A refresh without `scope` keeps the scope of the grant (RFC 6749 §6).
				provider
					.save_tokens(&with_scope(tokens, existing.scope.as_deref()))
					.await?;
				return Ok(OAuthFlowResult::Authorized);
			}
			Err(McpError::InsecureEndpoint(_)) => return Err(result.unwrap_err()),
			Err(McpError::OAuth(oauth_error)) if oauth_error.code != "server_error" => {
				return Err(McpError::OAuth(oauth_error));
			}
			Err(_) => {}
		}
	}
	let state = provider.state().await?;
	let authorization = start_authorization(
		&parse_url(&discovered.authorization_server_url)?,
		StartAuthorizationOptions {
			metadata,
			client_information: token_options.client_information.as_ref().expect("client set"),
			redirect_url,
			scope: scope.as_deref(),
			state: state.as_deref(),
			resource: token_options.resource.as_deref(),
		},
	)
	.await?;
	provider.save_code_verifier(&authorization.code_verifier).await?;
	provider.redirect_to_authorization(authorization.authorization_url).await?;
	Ok(OAuthFlowResult::Redirect)
}

/// Upstream `authorizeMcp`: retries with invalidated credentials on stale-client
/// and stale-grant errors.
pub async fn authorize_mcp(
	provider: Arc<dyn OAuthClientProvider>,
	options: OAuthFlowOptions,
) -> Result<OAuthFlowResult, McpError> {
	match run_flow(&provider, &options).await {
		Ok(result) => Ok(result),
		Err(McpError::OAuth(error))
			if matches!(error.code.as_str(), "invalid_client" | "unauthorized_client") =>
		{
			provider.invalidate_credentials(InvalidationKind::All).await?;
			run_flow(&provider, &options).await
		}
		Err(McpError::OAuth(error)) if error.code == "invalid_grant" => {
			provider.invalidate_credentials(InvalidationKind::Tokens).await?;
			run_flow(&provider, &options).await
		}
		Err(error) => Err(error),
	}
}

/// Auth provider for `StreamableHttpTransport`. After a 401 it refreshes the tokens, or throws
/// `McpOAuthAuthorizationRequiredError` when the user has to authorize (again). Concurrent 401s share
/// one refresh, and a request whose token was already replaced is just retried: with rotating refresh
/// tokens, a second refresh with the old refresh token would fail and discard the new grant.
pub fn adapt_oauth_provider<P>(provider: P) -> Arc<dyn AuthProvider>
where
	P: OAuthClientProvider + 'static,
{
	adapt_oauth_provider_arc(Arc::new(provider))
}

pub fn adapt_oauth_provider_arc(provider: Arc<dyn OAuthClientProvider>) -> Arc<dyn AuthProvider> {
	Arc::new(OAuthProviderAdapter {
		provider,
		in_flight: Arc::new(tokio::sync::Mutex::new(None)),
	})
}

type RefreshFuture = Shared<BoxFuture<'static, Result<(), McpError>>>;

struct OAuthProviderAdapter {
	provider: Arc<dyn OAuthClientProvider>,
	in_flight: Arc<tokio::sync::Mutex<Option<RefreshFuture>>>,
}

#[async_trait::async_trait]
impl AuthProvider for OAuthProviderAdapter {
	async fn token(&self) -> Result<Option<String>, McpError> {
		Ok(self.provider.tokens().await?.map(|tokens| tokens.access_token))
	}

	fn handles_unauthorized(&self) -> bool {
		true
	}

	async fn on_unauthorized(&self, context: UnauthorizedContext) -> Result<(), McpError> {
		let challenge = parse_www_authenticate(context.response.header("www-authenticate"));
		let insufficient_scope = challenge.error.as_deref() == Some("insufficient_scope");
		if !insufficient_scope && self.in_flight.lock().await.is_none() && context.token.is_some() {
			let current = self.provider.tokens().await?.map(|tokens| tokens.access_token);
			if current.is_some() && current != context.token {
				// Another request already refreshed the credentials.
				return Ok(());
			}
		}
		let future: RefreshFuture = {
			let mut guard = self.in_flight.lock().await;
			match guard.clone() {
				Some(existing) => existing,
				None => {
					let provider = Arc::clone(&self.provider);
					let in_flight_for_future = Arc::clone(&self.in_flight);
					let challenge_scope = challenge.scope.clone();
					let resource_metadata_url = challenge.resource_metadata_url.clone();
					let server_url = context.server_url.clone();
					let fetch = context.fetch.clone();
					let future: BoxFuture<'static, Result<(), McpError>> = Box::pin(async move {
						// Upstream resolves `provider.tokens()` inside the shared promise
						// chain, i.e. after the slot is claimed.
						let granted = if insufficient_scope { provider.tokens().await? } else { None };
						let result = authorize_mcp(
							Arc::clone(&provider),
							OAuthFlowOptions {
								server_url,
								authorization_code: None,
								iss: None,
								scope: if insufficient_scope {
									step_up_scope(
										granted.as_ref().and_then(|tokens| tokens.scope.as_deref()),
										challenge_scope.as_deref(),
									)
								} else {
									challenge_scope.clone()
								},
								resource_metadata_url,
								authorization_server_metadata_url: None,
								fetch: Some(fetch),
								skip_issuer_validation: false,
								skip_refresh: insufficient_scope,
							},
						)
						.await;
						// Upstream `.finally`: the shared slot clears when the refresh settles.
						*in_flight_for_future.lock().await = None;
						match result {
							Ok(OAuthFlowResult::Authorized) => Ok(()),
							Ok(OAuthFlowResult::Redirect) => Err(McpError::OAuthAuthorizationRequired(
								McpOAuthAuthorizationRequiredError,
							)),
							Err(error) => Err(error),
						}
					});
					let shared = future.shared();
					*guard = Some(shared.clone());
					shared
				}
			}
		};
		Ok(future.await?)
	}
}
