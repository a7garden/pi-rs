//! Port of `packages/mcp/src/oauth/index.ts` (upstream package entry `./oauth`).

pub mod callback;
pub mod discovery;
pub mod errors;
pub mod flow;
pub mod provider;
pub mod types;

// Mirrors upstream `src/oauth/index.ts` re-exports.
pub use callback::{
	OAuthCallback, OAuthCallbackPage, OAuthCallbackServer, OAuthCallbackServerOptions, RenderPage,
};
pub use discovery::{
	build_authorization_server_discovery_urls, discover_authorization_server_metadata,
	discover_oauth_server_info, discover_protected_resource_metadata, parse_www_authenticate,
	resource_url_from_server_url, select_resource, DiscoverAuthorizationServerOptions,
	DiscoverProtectedResourceOptions, DiscoverServerInfoOptions,
};
pub use errors::{
	McpOAuthAuthorizationRequiredError, OAuthError, OAuthInsecureEndpointError,
	OAuthIssuerMismatchError, OAuthRegistrationError,
};
pub use flow::{
	adapt_oauth_provider, adapt_oauth_provider_arc, authorize_mcp, exchange_authorization_code,
	refresh_authorization, register_client, start_authorization, step_up_scope,
	AddClientAuthentication, ExchangeOptions, FormParams, InvalidationKind,
	OAuthClientMetadataDocument, OAuthClientProvider, OAuthFlowOptions, OAuthFlowResult,
	RefreshOptions, RegisterClientOptions, StartAuthorization, StartAuthorizationOptions,
	TokenRequestOptions,
};
pub use provider::{
	MemoryOAuthStateStore, McpOAuthProvider, McpOAuthProviderOptions, McpOAuthState, McpOAuthStateStore,
};
pub use types::{
	parse_authorization_server_metadata, parse_client_information, parse_oauth_tokens,
	parse_protected_resource_metadata, AuthorizationServerMetadata, OAuthChallenge,
	OAuthClientInformation, OAuthClientInformationFull, OAuthClientInformationMixed,
	OAuthClientMetadata, OAuthDiscoveryState, OAuthProtectedResourceMetadata, OAuthServerInfo,
	OAuthTokens,
};
