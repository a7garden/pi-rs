//! Port of `test/oauth.test.ts`.

use std::sync::{Arc, Mutex as StdMutex};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use sha2::{Digest, Sha256};
use url::Url;

use pi_mcp::auth_provider::{AuthProvider, HttpResponse, UnauthorizedContext, UnauthorizedResponse};
use pi_mcp::client::{McpClient, McpClientOptions};
use pi_mcp::oauth::{
	adapt_oauth_provider, adapt_oauth_provider_arc, authorize_mcp, discover_authorization_server_metadata,
	InvalidationKind, MemoryOAuthStateStore, McpOAuthProvider, McpOAuthProviderOptions,
	McpOAuthStateStore, OAuthCallbackPage, OAuthCallbackServer, OAuthCallbackServerOptions,
	OAuthClientInformation, OAuthClientInformationMixed, OAuthClientMetadata, OAuthClientProvider,
	OAuthDiscoveryState, OAuthFlowOptions, OAuthFlowResult, OAuthTokens,
};
use pi_mcp::transports::streamable_http::StreamableHttpTransportOptions;
use bytes::Bytes;
use pi_mcp::McpError;

mod helpers;
use helpers::{body_from, listen, read_body, TestHttpServer};

fn client_information(client_id: &str) -> OAuthClientInformationMixed {
	OAuthClientInformationMixed::Plain(OAuthClientInformation {
		client_id: client_id.to_string(),
		client_secret: None,
		client_id_issued_at: None,
		client_secret_expires_at: None,
	})
}

fn flow_options(server_url: Url) -> OAuthFlowOptions {
	OAuthFlowOptions {
		server_url,
		authorization_code: None,
		iss: None,
		scope: None,
		resource_metadata_url: None,
		authorization_server_metadata_url: None,
		fetch: None,
		skip_issuer_validation: false,
		skip_refresh: false,
	}
}

struct TestOAuthProvider {
	redirect_url: String,
	client_metadata: OAuthClientMetadata,
	client: StdMutex<Option<OAuthClientInformationMixed>>,
	token_set: StdMutex<Option<OAuthTokens>>,
	verifier: StdMutex<Option<String>>,
	discovery: StdMutex<Option<OAuthDiscoveryState>>,
	authorization_url: StdMutex<Option<Url>>,
}

impl TestOAuthProvider {
	fn new(redirect_url: &str) -> TestOAuthProvider {
		TestOAuthProvider {
			redirect_url: redirect_url.to_string(),
			client_metadata: OAuthClientMetadata {
				redirect_uris: vec![redirect_url.to_string()],
				client_name: Some("pi-mcp-test".to_string()),
				grant_types: Some(vec!["authorization_code".to_string(), "refresh_token".to_string()]),
				response_types: Some(vec!["code".to_string()]),
				token_endpoint_auth_method: Some("none".to_string()),
				..Default::default()
			},
			client: StdMutex::new(None),
			token_set: StdMutex::new(None),
			verifier: StdMutex::new(None),
			discovery: StdMutex::new(None),
			authorization_url: StdMutex::new(None),
		}
	}

	fn authorization_url(&self) -> Url {
		self.authorization_url
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner)
			.clone()
			.expect("authorization url set")
	}

	fn set_tokens_access(&self, access_token: &str) {
		let mut tokens = self.token_set.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
		if let Some(tokens) = tokens.as_mut() {
			tokens.access_token = access_token.to_string();
		}
	}

	fn tokens_snapshot(&self) -> Option<OAuthTokens> {
		self.token_set.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
	}
}

#[async_trait::async_trait]
impl OAuthClientProvider for TestOAuthProvider {
	fn redirect_url(&self) -> String {
		self.redirect_url.clone()
	}

	fn client_metadata(&self) -> &OAuthClientMetadata {
		&self.client_metadata
	}

	async fn state(&self) -> Result<Option<String>, McpError> {
		Ok(Some("expected-state".to_string()))
	}

	async fn client_information(&self) -> Result<Option<OAuthClientInformationMixed>, McpError> {
		Ok(self.client.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone())
	}

	async fn save_client_information(
		&self,
		information: &OAuthClientInformationMixed,
	) -> Result<(), McpError> {
		*self.client.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(information.clone());
		Ok(())
	}

	async fn tokens(&self) -> Result<Option<OAuthTokens>, McpError> {
		Ok(self.token_set.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone())
	}

	async fn save_tokens(&self, tokens: &OAuthTokens) -> Result<(), McpError> {
		*self.token_set.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(tokens.clone());
		Ok(())
	}

	async fn redirect_to_authorization(&self, url: Url) -> Result<(), McpError> {
		*self.authorization_url.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(url);
		Ok(())
	}

	async fn save_code_verifier(&self, verifier: &str) -> Result<(), McpError> {
		*self.verifier.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(verifier.to_string());
		Ok(())
	}

	async fn code_verifier(&self) -> Result<String, McpError> {
		self.verifier
			.lock()
			.unwrap_or_else(std::sync::PoisonError::into_inner)
			.clone()
			.ok_or_else(|| McpError::other("Missing code verifier"))
	}

	async fn invalidate_credentials(&self, kind: InvalidationKind) -> Result<(), McpError> {
		if matches!(kind, InvalidationKind::All | InvalidationKind::Client) {
			*self.client.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
		}
		if matches!(kind, InvalidationKind::All | InvalidationKind::Tokens) {
			*self.token_set.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
		}
		if matches!(kind, InvalidationKind::All | InvalidationKind::Verifier) {
			*self.verifier.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
		}
		if matches!(kind, InvalidationKind::All | InvalidationKind::Discovery) {
			*self.discovery.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
		}
		Ok(())
	}

	async fn save_discovery_state(&self, state: &OAuthDiscoveryState) -> Result<(), McpError> {
		*self.discovery.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(state.clone());
		Ok(())
	}

	async fn discovery_state(&self) -> Result<Option<OAuthDiscoveryState>, McpError> {
		Ok(self.discovery.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone())
	}
}

fn query_of(url: &Url, name: &str) -> Option<String> {
	url.query_pairs().find(|(key, _)| key == name).map(|(_, value)| value.into_owned())
}

async fn connect_oauth(
	server: &TestHttpServer,
	headers: Vec<(String, String)>,
	provider: Arc<dyn AuthProvider>,
) -> Result<McpClient, McpError> {
	let url = format!("{}/mcp", server.origin);
	let mut options = StreamableHttpTransportOptions::new(url::Url::parse(&url).expect("url"));
	options.headers = Some(headers);
	options.auth_provider = Some(provider);
	options.open_get_stream = Some(false);
	let transport = pi_mcp::transports::streamable_http::StreamableHttpTransport::new(options);
	let client = McpClient::new(McpClientOptions::new("oauth-test", "1.0.0"));
	client.connect(transport).await?;
	Ok(client)
}

// Port of oauth.test.ts "discovers, registers, authorizes with PKCE, and refreshes on 401".
#[tokio::test]
async fn discovers_registers_authorizes_with_pkce_and_refreshes_on_401() {
	let expected_challenge: Arc<StdMutex<Option<String>>> = Arc::new(StdMutex::new(None));
	let refreshes = Arc::new(StdMutex::new(0usize));
	let challenge_sink = Arc::clone(&expected_challenge);
	let refresh_sink = Arc::clone(&refreshes);
	let server = listen(Arc::new(move |request| {
		let challenge_sink = Arc::clone(&challenge_sink);
		let refresh_sink = Arc::clone(&refresh_sink);
		Box::pin(async move {
			let path = request.url.path().to_string();
			let origin = format!("{}://{}", request.url.scheme(), request.url.host_str().unwrap_or_default())
				+ &request
					.url
					.port()
					.map(|port| format!(":{port}"))
					.unwrap_or_default();
			match path.as_str() {
				"/.well-known/oauth-protected-resource/mcp" => {
					return HttpResponse {
						status: 200,
						headers: vec![("content-type".to_string(), "application/json".to_string())],
						body: body_from(
							serde_json::json!({
								"resource": format!("{origin}/mcp"),
								"authorization_servers": [origin],
								"scopes_supported": ["org:read"],
							})
							.to_string(),
						),
					};
				}
				"/.well-known/oauth-authorization-server" => {
					return HttpResponse {
						status: 200,
						headers: vec![("content-type".to_string(), "application/json".to_string())],
						body: body_from(
							serde_json::json!({
								"issuer": origin,
								"authorization_endpoint": format!("{origin}/authorize"),
								"token_endpoint": format!("{origin}/token"),
								"registration_endpoint": format!("{origin}/register"),
								"response_types_supported": ["code"],
								"grant_types_supported": ["authorization_code", "refresh_token"],
								"token_endpoint_auth_methods_supported": ["none"],
								"code_challenge_methods_supported": ["S256"],
							})
							.to_string(),
						),
					};
				}
				"/register" => {
					let metadata: serde_json::Value =
						serde_json::from_str(&read_body(&request).await).unwrap_or(serde_json::Value::Null);
					// Empty and null optional fields count as absent (#10266).
					let mut reply = metadata;
					reply["client_id"] = serde_json::json!("test-client");
					reply["client_secret"] = serde_json::json!("");
					return HttpResponse {
						status: 201,
						headers: vec![("content-type".to_string(), "application/json".to_string())],
						body: body_from(reply.to_string()),
					};
				}
				"/authorize" => {
					let challenge = query_of(&request.url, "code_challenge").unwrap_or_default();
					*challenge_sink.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(challenge);
					let redirect = Url::parse(&query_of(&request.url, "redirect_uri").unwrap_or_default())
						.expect("redirect uri");
					let mut redirect = redirect;
					redirect.query_pairs_mut().append_pair("code", "test-code");
					redirect
						.query_pairs_mut()
						.append_pair("state", &query_of(&request.url, "state").unwrap_or_default());
					return HttpResponse {
						status: 302,
						headers: vec![("location".to_string(), redirect.to_string())],
						body: body_from(Bytes::new()),
					};
				}
				"/token" => {
					let params: Vec<(String, String)> = url::form_urlencoded::parse(request.body.as_deref().unwrap_or_default())
						.into_owned()
						.collect();
					let param = |name: &str| {
						params
							.iter()
							.find(|(key, _)| key == name)
							.map(|(_, value)| value.clone())
							.unwrap_or_default()
					};
					if param("grant_type") == "refresh_token" {
						*refresh_sink.lock().unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
						return HttpResponse {
							status: 200,
							headers: vec![("content-type".to_string(), "application/json".to_string())],
							body: body_from(
								serde_json::json!({
									"access_token": "refreshed-token",
									"token_type": "Bearer",
									"refresh_token": "",
									"expires_in": serde_json::Value::Null,
								})
								.to_string(),
							),
						};
					}
					let verifier = param("code_verifier");
					let challenge: String = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
					if param("code") != "test-code"
						|| Some(&challenge)
							!= challenge_sink
								.lock()
								.unwrap_or_else(std::sync::PoisonError::into_inner)
								.as_ref()
					{
						return HttpResponse {
							status: 400,
							headers: vec![("content-type".to_string(), "application/json".to_string())],
							body: body_from(serde_json::json!({ "error": "invalid_grant" }).to_string()),
						};
					}
					return HttpResponse {
						status: 200,
						headers: vec![("content-type".to_string(), "application/json".to_string())],
						body: body_from(
							serde_json::json!({
								"access_token": "first-token",
								"refresh_token": "refresh-token",
								"token_type": "Bearer",
								"scope": "",
							})
							.to_string(),
						),
					};
				}
				_ => {}
			}
			if path != "/mcp" {
				return HttpResponse { status: 404, headers: Vec::new(), body: body_from(Bytes::new()) };
			}
			if request.method == "GET" {
				return HttpResponse { status: 405, headers: Vec::new(), body: body_from(Bytes::new()) };
			}
			if request.method == "DELETE" {
				return HttpResponse { status: 200, headers: Vec::new(), body: body_from(Bytes::new()) };
			}
			let token = request
				.headers
				.iter()
				.find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
				.map(|(_, value)| value.clone())
				.unwrap_or_default();
			if token != "Bearer first-token" && token != "Bearer refreshed-token" {
				let _ = read_body(&request).await;
				return HttpResponse {
					status: 401,
					headers: vec![(
						"www-authenticate".to_string(),
						// An empty scope falls through to the resource metadata's scopes_supported.
						format!(
							"Bearer resource_metadata=\"{origin}/.well-known/oauth-protected-resource/mcp\", scope=\"\""
						),
					)],
					body: body_from("Unauthorized"),
				};
			}
			let message: serde_json::Value =
				serde_json::from_str(&read_body(&request).await).unwrap_or(serde_json::Value::Null);
			let Some(id) = message.get("id").cloned() else {
				return HttpResponse { status: 202, headers: Vec::new(), body: body_from(Bytes::new()) };
			};
			let result = if message.get("method").and_then(serde_json::Value::as_str) == Some("initialize") {
				serde_json::json!({
					"protocolVersion": "2025-06-18",
					"capabilities": { "tools": {} },
					"serverInfo": { "name": "oauth-test", "version": "1.0.0" },
				})
			} else {
				serde_json::json!({ "tools": [{ "name": "issues", "inputSchema": { "type": "object" } }] })
			};
			HttpResponse {
				status: 200,
				headers: vec![("content-type".to_string(), "application/json".to_string())],
				body: body_from(serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string()),
			}
		})
	}))
	.await;

	let callback = OAuthCallbackServer::listen(OAuthCallbackServerOptions::default())
		.await
		.expect("callback server");
	let provider = Arc::new(TestOAuthProvider::new(&callback.redirect_url));
	let first_error = connect_oauth(
		&server,
		vec![("Authorization".to_string(), "Bearer caller-supplied-stale-token".to_string())],
		adapt_oauth_provider_arc(provider.clone()),
	)
	.await
	.err()
	.expect("requires user interaction");
	assert!(
		matches!(first_error, McpError::OAuthAuthorizationRequired(_)),
		"{first_error:?}"
	);
	let authorization_url = provider.authorization_url();
	assert_eq!(query_of(&authorization_url, "scope").as_deref(), Some("org:read"));
	assert_eq!(query_of(&authorization_url, "resource").as_deref(), Some(format!("{}/mcp", server.origin).as_str()));

	// Simulate the browser round trip: fetch the authorization URL (reqwest follows
	// the 302 to the callback URL, delivering the code and state), after registering
	// the callback waiter exactly like upstream.
	let authorization_fetch = {
		let fetch = pi_mcp::auth_provider::default_fetch();
		let url = authorization_url.clone();
		tokio::spawn(async move {
			fetch(pi_mcp::auth_provider::HttpRequest {
				method: "GET".to_string(),
				url,
				headers: Vec::new(),
				body: None,
			})
			.await
		})
	};
	let callback_value = callback
		.wait_for_callback("expected-state", None)
		.await
		.expect("callback delivered");
	let _ = authorization_fetch.await.expect("authorization fetch join");
	assert_eq!(callback_value.code, "test-code");

	let result = authorize_mcp(
		provider.clone(),
		OAuthFlowOptions {
			authorization_code: Some(callback_value.code),
			..flow_options(Url::parse(&format!("{}/mcp", server.origin)).expect("url"))
		},
	)
	.await
	.expect("code exchange");
	assert_eq!(result, OAuthFlowResult::Authorized);

	let client = connect_oauth(
		&server,
		vec![("Authorization".to_string(), "Bearer caller-supplied-stale-token".to_string())],
		adapt_oauth_provider_arc(provider.clone()),
	)
	.await
	.expect("connect after authorization");
	let tools = client.list_tools(&Default::default()).await.expect("list tools");
	assert_eq!(
		serde_json::to_value(&tools).unwrap(),
		serde_json::json!([{ "name": "issues", "inputSchema": { "type": "object" } }])
	);
	client.close().await.expect("close");

	provider.set_tokens_access("stale-token");
	let refreshed_client = connect_oauth(
		&server,
		vec![("Authorization".to_string(), "Bearer caller-supplied-stale-token".to_string())],
		adapt_oauth_provider_arc(provider.clone()),
	)
	.await
	.expect("connect after refresh");
	// Neither token response names a scope, so the grant has the requested scope.
	assert_eq!(
		provider.tokens_snapshot(),
		Some(OAuthTokens {
			access_token: "refreshed-token".to_string(),
			token_type: "Bearer".to_string(),
			scope: Some("org:read".to_string()),
			refresh_token: Some("refresh-token".to_string()),
			expires_in: None,
			id_token: None,
		})
	);
	assert_eq!(*refreshes.lock().unwrap_or_else(std::sync::PoisonError::into_inner), 1);
	refreshed_client.close().await.expect("close");
	callback.close().await.expect("close callback server");
	server.close().await;
}

// Port of oauth.test.ts "shares one refresh between concurrent 401s when refresh tokens rotate".
#[tokio::test]
async fn shares_one_refresh_between_concurrent_401s_when_refresh_tokens_rotate() {
	let grants: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
	let grants_for_handler = Arc::clone(&grants);
	let server = listen(Arc::new(move |request| {
		let grants = Arc::clone(&grants_for_handler);
		Box::pin(async move {
			let path = request.url.path().to_string();
			let origin = format!("{}://{}", request.url.scheme(), request.url.host_str().unwrap_or_default())
				+ &request
					.url
					.port()
					.map(|port| format!(":{port}"))
					.unwrap_or_default();
			if path == "/.well-known/oauth-protected-resource/mcp" {
				// Invalid resource metadata falls back to the server origin instead of failing discovery.
				return HttpResponse {
					status: 200,
					headers: vec![("content-type".to_string(), "application/json".to_string())],
					body: body_from(
						serde_json::json!({
							"resource": format!("{origin}/mcp"),
							"authorization_servers": ["not a url"],
						})
						.to_string(),
					),
				};
			}
			if path == "/.well-known/oauth-authorization-server" {
				return HttpResponse {
					status: 200,
					headers: vec![("content-type".to_string(), "application/json".to_string())],
					body: body_from(
						serde_json::json!({
							// Issuer without the trailing slash that URL parsing adds to the fallback server URL.
							"issuer": origin,
							"authorization_endpoint": format!("{origin}/authorize"),
							"token_endpoint": format!("{origin}/token"),
							"response_types_supported": ["code"],
						})
						.to_string(),
					),
				};
			}
			if path == "/token" {
				let params: Vec<(String, String)> =
					url::form_urlencoded::parse(request.body.as_deref().unwrap_or_default())
						.into_owned()
						.collect();
				let refresh_token = params
					.iter()
					.find(|(key, _)| key == "refresh_token")
					.map(|(_, value)| value.clone())
					.unwrap_or_default();
				grants
					.lock()
					.unwrap_or_else(std::sync::PoisonError::into_inner)
					.push(refresh_token.clone());
				if refresh_token != "r1" {
					return HttpResponse {
						status: 400,
						headers: vec![("content-type".to_string(), "application/json".to_string())],
						body: body_from(serde_json::json!({ "error": "invalid_grant" }).to_string()),
					};
				}
				tokio::time::sleep(std::time::Duration::from_millis(20)).await;
				return HttpResponse {
					status: 200,
					headers: vec![("content-type".to_string(), "application/json".to_string())],
					body: body_from(
						serde_json::json!({
							"access_token": "a2",
							"refresh_token": "r2",
							"token_type": "Bearer",
							"expires_in": 3600,
						})
						.to_string(),
					),
				};
			}
			HttpResponse { status: 404, headers: Vec::new(), body: body_from(Bytes::new()) }
		})
	}))
	.await;
	let store = Arc::new(MemoryOAuthStateStore::new());
	let provider = McpOAuthProvider::new(McpOAuthProviderOptions {
		server_url: Url::parse(&format!("{}/mcp", server.origin)).expect("url"),
		redirect_url: "http://127.0.0.1/callback".to_string(),
		client_metadata: OAuthClientMetadata {
			client_name: Some("test".to_string()),
			..Default::default()
		},
		client_metadata_document: None,
		client_id: Some("client".to_string()),
		client_secret: None,
		store: Some(store.clone()),
		on_redirect: Arc::new(|_| Box::pin(async { Ok(()) })),
	});
	provider
		.save_tokens(&OAuthTokens {
			access_token: "a1".to_string(),
			refresh_token: Some("r1".to_string()),
			token_type: "Bearer".to_string(),
			expires_in: None,
			scope: None,
			id_token: None,
		})
		.await
		.expect("save tokens");
	let auth = adapt_oauth_provider(provider);
	let unauthorized = || UnauthorizedContext {
		response: UnauthorizedResponse {
			status: 401,
			headers: vec![("www-authenticate".to_string(), "Bearer".to_string())],
		},
		server_url: Url::parse(&format!("{}/mcp", server.origin)).expect("url"),
		fetch: pi_mcp::auth_provider::default_fetch(),
		token: Some("a1".to_string()),
	};
	let (first, second) = tokio::join!(
		auth.on_unauthorized(unauthorized()),
		auth.on_unauthorized(unauthorized())
	);
	first.expect("first refresh");
	second.expect("second refresh");
	// A late 401 for a request that still carried the old token must not refresh again.
	auth.on_unauthorized(unauthorized()).await.expect("late 401");
	assert_eq!(
		grants.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone(),
		vec!["r1".to_string()]
	);
	assert_eq!(auth.token().await.expect("token").as_deref(), Some("a2"));
	let state = store.load().await.expect("load").expect("state saved");
	assert_eq!(
		state.tokens.as_ref().and_then(|tokens| tokens.refresh_token.as_deref()),
		Some("r2")
	);
	let expires_at = state.tokens_expire_at.expect("expiry recorded");
	assert!(
		expires_at as u128
			> std::time::SystemTime::now()
				.duration_since(std::time::UNIX_EPOCH)
				.expect("epoch")
				.as_millis()
				+ 3_500_000
	);
	server.close().await;
}

// Port of oauth.test.ts "asks for authorization instead of refreshing when the server needs more scope".
#[tokio::test]
async fn asks_for_authorization_instead_of_refreshing_when_the_server_needs_more_scope() {
	let provider = Arc::new(TestOAuthProvider::new("http://127.0.0.1/callback"));
	*provider.client.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
		Some(client_information("client"));
	*provider.token_set.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(OAuthTokens {
		access_token: "a1".to_string(),
		refresh_token: Some("r1".to_string()),
		token_type: "Bearer".to_string(),
		scope: Some("repo read:org".to_string()),
		expires_in: None,
		id_token: None,
	});
	let server = listen(Arc::new(|request| {
		Box::pin(async move {
			let path = request.url.path().to_string();
			let origin = format!("{}://{}", request.url.scheme(), request.url.host_str().unwrap_or_default())
				+ &request
					.url
					.port()
					.map(|port| format!(":{port}"))
					.unwrap_or_default();
			if path == "/.well-known/oauth-authorization-server" {
				return HttpResponse {
					status: 200,
					headers: vec![("content-type".to_string(), "application/json".to_string())],
					body: body_from(
						serde_json::json!({
							"issuer": origin,
							"authorization_endpoint": format!("{origin}/authorize"),
							"token_endpoint": format!("{origin}/token"),
							"response_types_supported": ["code"],
						})
						.to_string(),
					),
				};
			}
			HttpResponse {
				status: if path == "/token" { 500 } else { 404 },
				headers: Vec::new(),
				body: body_from(Bytes::new()),
			}
		})
	}))
	.await;
	let auth = adapt_oauth_provider_arc(provider.clone());
	let error = auth
		.on_unauthorized(UnauthorizedContext {
			response: UnauthorizedResponse {
				status: 403,
				headers: vec![(
					"www-authenticate".to_string(),
					"Bearer error=\"insufficient_scope\", scope=\"repo admin\"".to_string(),
				)],
			},
			server_url: Url::parse(&format!("{}/mcp", server.origin)).expect("url"),
			fetch: pi_mcp::auth_provider::default_fetch(),
			token: Some("a1".to_string()),
		})
		.await
		.expect_err("requires interaction");
	assert!(matches!(error, McpError::OAuthAuthorizationRequired(_)));
	// The challenge may list only the missing scopes; the new grant keeps the old ones too.
	let authorization_url = provider.authorization_url();
	assert_eq!(
		query_of(&authorization_url, "scope").as_deref(),
		Some("repo read:org admin")
	);
	// The working grant is kept until the user authorizes the new scope.
	assert_eq!(
		provider.tokens_snapshot().and_then(|tokens| Some(tokens.access_token)),
		Some("a1".to_string())
	);
	server.close().await;
}

// Port of oauth.test.ts "binds persisted credentials to the exact MCP server URL".
#[tokio::test]
async fn binds_persisted_credentials_to_the_exact_mcp_server_url() {
	let store = Arc::new(MemoryOAuthStateStore::new());
	let first = McpOAuthProvider::new(McpOAuthProviderOptions {
		server_url: Url::parse("https://one.example/mcp").expect("url"),
		redirect_url: "http://127.0.0.1/callback".to_string(),
		client_metadata: OAuthClientMetadata {
			client_name: Some("test".to_string()),
			..Default::default()
		},
		client_metadata_document: None,
		client_id: None,
		client_secret: None,
		store: Some(store.clone()),
		on_redirect: Arc::new(|_| Box::pin(async { Ok(()) })),
	});
	first
		.save_tokens(&OAuthTokens {
			access_token: "secret".to_string(),
			refresh_token: None,
			token_type: "Bearer".to_string(),
			expires_in: None,
			scope: None,
			id_token: None,
		})
		.await
		.expect("save");
	assert_eq!(
		first.tokens().await.expect("tokens").and_then(|tokens| Some(tokens.access_token)),
		Some("secret".to_string())
	);

	let second = McpOAuthProvider::new(McpOAuthProviderOptions {
		server_url: Url::parse("https://two.example/mcp").expect("url"),
		redirect_url: "http://127.0.0.1/callback".to_string(),
		client_metadata: OAuthClientMetadata {
			client_name: Some("test".to_string()),
			..Default::default()
		},
		client_metadata_document: None,
		client_id: None,
		client_secret: None,
		store: Some(store.clone()),
		on_redirect: Arc::new(|_| Box::pin(async { Ok(()) })),
	});
	assert!(second.tokens().await.expect("tokens").is_none());
}

// Port of oauth.test.ts "rejects authorization metadata whose issuer does not match discovery".
#[tokio::test]
async fn rejects_authorization_metadata_whose_issuer_does_not_match_discovery() {
	let server = listen(Arc::new(|request| {
		Box::pin(async move {
			let path = request.url.path().to_string();
			let origin = format!("{}://{}", request.url.scheme(), request.url.host_str().unwrap_or_default())
				+ &request
					.url
					.port()
					.map(|port| format!(":{port}"))
					.unwrap_or_default();
			if path == "/.well-known/oauth-authorization-server" {
				return HttpResponse {
					status: 200,
					headers: vec![("content-type".to_string(), "application/json".to_string())],
					body: body_from(
						serde_json::json!({
							"issuer": "https://attacker.example",
							"authorization_endpoint": format!("{origin}/authorize"),
							"token_endpoint": format!("{origin}/token"),
							"response_types_supported": ["code"],
						})
						.to_string(),
					),
				};
			}
			HttpResponse { status: 404, headers: Vec::new(), body: body_from(Bytes::new()) }
		})
	}))
	.await;
	let metadata = discover_authorization_server_metadata(
		Url::parse(&server.origin).expect("url"),
		Default::default(),
	)
	.await;
	match metadata {
		Err(McpError::IssuerMismatch(_)) => {}
		other => panic!("expected issuer mismatch, got {other:?}"),
	}
	server.close().await;
}

// Port of oauth.test.ts "uses a configured authorization server metadata document as is" (#10172).
#[tokio::test]
async fn uses_a_configured_authorization_server_metadata_document_as_is() {
	let server = listen(Arc::new(|request| {
		Box::pin(async move {
			let path = request.url.path().to_string();
			let origin = format!("{}://{}", request.url.scheme(), request.url.host_str().unwrap_or_default())
				+ &request
					.url
					.port()
					.map(|port| format!(":{port}"))
					.unwrap_or_default();
			let response = if path == "/.well-known/oauth-protected-resource/mcp" {
				// Names the MCP server itself, which serves no authorization server metadata.
				Some((
					200,
					serde_json::json!({
						"resource": format!("{origin}/mcp"),
						"authorization_servers": [origin],
					}),
				))
			} else if path == "/idp/metadata.json" {
				Some((
					200,
					serde_json::json!({
						// Not derivable from the document URL; a configured document is not checked.
						"issuer": "https://idp.example",
						"authorization_endpoint": format!("{origin}/idp/authorize"),
						"token_endpoint": format!("{origin}/idp/token"),
						"response_types_supported": ["code"],
					}),
				))
			} else {
				None
			};
			match response {
				Some((status, body)) => HttpResponse {
					status,
					headers: vec![("content-type".to_string(), "application/json".to_string())],
					body: body_from(body.to_string()),
				},
				None => HttpResponse { status: 404, headers: Vec::new(), body: body_from(Bytes::new()) },
			}
		})
	}))
	.await;
	let provider = Arc::new(TestOAuthProvider::new("http://127.0.0.1/callback"));
	*provider.client.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
		Some(client_information("client"));
	let options = OAuthFlowOptions {
		authorization_server_metadata_url: Some(Url::parse(&format!("{}/idp/metadata.json", server.origin)).expect("url")),
		..flow_options(Url::parse(&format!("{}/mcp", server.origin)).expect("url"))
	};
	let result = authorize_mcp(provider.clone(), options.clone()).await.expect("redirect");
	assert_eq!(result, OAuthFlowResult::Redirect);
	let authorization_url = provider.authorization_url();
	assert_eq!(
		format!("{}{}", authorization_url.origin().unicode_serialization(), authorization_url.path()),
		format!("{}/idp/authorize", server.origin)
	);
	assert_eq!(
		query_of(&authorization_url, "resource").as_deref(),
		Some(format!("{}/mcp", server.origin).as_str())
	);

	let insecure = OAuthFlowOptions {
		authorization_server_metadata_url: Some(Url::parse("http://idp.example/metadata.json").expect("url")),
		..options
	};
	let error = authorize_mcp(provider, insecure).await.expect_err("insecure");
	assert!(matches!(error, McpError::InsecureEndpoint(_)), "{error:?}");
	server.close().await;
}

// Port of oauth.test.ts
// "exchanges a code only when its iss parameter names the authorization server".
#[tokio::test]
async fn exchanges_a_code_only_when_its_iss_parameter_names_the_authorization_server() {
	let codes: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
	let codes_for_handler = Arc::clone(&codes);
	let server = listen(Arc::new(move |request| {
		let codes = Arc::clone(&codes_for_handler);
		Box::pin(async move {
			let params: Vec<(String, String)> =
				url::form_urlencoded::parse(request.body.as_deref().unwrap_or_default())
					.into_owned()
					.collect();
			codes
				.lock()
				.unwrap_or_else(std::sync::PoisonError::into_inner)
				.push(
					params
						.iter()
						.find(|(key, _)| key == "code")
						.map(|(_, value)| value.clone())
						.unwrap_or_default(),
				);
			HttpResponse {
				status: 200,
				headers: vec![("content-type".to_string(), "application/json".to_string())],
				body: body_from(
					serde_json::json!({ "access_token": "token", "token_type": "Bearer" }).to_string(),
				),
			}
		})
	}))
	.await;
	let exchange = |code: String, iss: Option<String>, iss_parameter_supported: bool| {
		let server_origin = server.origin.clone();
		async move {
			let provider = Arc::new(TestOAuthProvider::new("http://127.0.0.1/callback"));
			*provider.client.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
				Some(client_information("client"));
			*provider.verifier.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
				Some("verifier".to_string());
			*provider.discovery.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
				Some(OAuthDiscoveryState {
					authorization_server_url: server_origin.clone(),
					authorization_server_metadata: Some(
						pi_mcp::oauth::parse_authorization_server_metadata(serde_json::json!({
							"issuer": server_origin,
							"authorization_endpoint": format!("{server_origin}/authorize"),
							"token_endpoint": format!("{server_origin}/token"),
							"response_types_supported": ["code"],
							"authorization_response_iss_parameter_supported": iss_parameter_supported,
						}))
						.expect("metadata"),
					),
					resource_metadata: None,
					resource_metadata_url: None,
				});
			authorize_mcp(
				provider,
				OAuthFlowOptions {
					authorization_code: Some(code.to_string()),
					iss,
					..flow_options(Url::parse(&format!("{server_origin}/mcp")).expect("url"))
					},
			)
			.await
		}
	};
	let error = exchange("other".to_string(), Some("https://attacker.example".to_string()), false)
		.await
		.err()
		.expect("mismatched iss");
	assert!(matches!(error, McpError::IssuerMismatch(_)), "{error:?}");
	let error = exchange("missing".to_string(), None, true).await.err().expect("missing iss");
	assert!(matches!(error, McpError::IssuerMismatch(_)), "{error:?}");
	let result = exchange("matching".to_string(), Some(server.origin.clone()), true).await.expect("matching");
	assert_eq!(result, OAuthFlowResult::Authorized);
	// Servers that do not promise the parameter may omit it.
	let result = exchange("omitted".to_string(), None, false).await.expect("omitted");
	assert_eq!(result, OAuthFlowResult::Authorized);
	assert_eq!(
		codes.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone(),
		vec!["matching".to_string(), "omitted".to_string()]
	);
	server.close().await;
}

// Port of oauth.test.ts "renders plain text by default".
#[tokio::test]
async fn callback_renders_plain_text_by_default() {
	let callback = OAuthCallbackServer::listen(OAuthCallbackServerOptions::default())
		.await
		.expect("listen");
	let fetch = pi_mcp::auth_provider::default_fetch();
	let waiter = Arc::clone(&callback);
	let wait_task = tokio::spawn(async move { waiter.wait_for_callback("s1", None).await });
	// Give the spawned waiter a beat to register its state.
	tokio::time::sleep(std::time::Duration::from_millis(50)).await;
	let url = Url::parse(&format!("{}?code=abc&state=s1", callback.redirect_url)).expect("url");
	let response = fetch(pi_mcp::auth_provider::HttpRequest {
		method: "GET".to_string(),
		url,
		headers: Vec::new(),
		body: None,
	})
	.await
	.expect("callback");
	assert_eq!(response.status, 200);
	assert_eq!(
		response.header("content-type"),
		Some("text/plain; charset=utf-8")
	);
	let body = response.text().await.expect("body");
	assert_eq!(body, "Authorization complete. You may close this window.");
	let callback_value = wait_task.await.expect("join").expect("callback");
	assert_eq!(callback_value.code, "abc");
	callback.close().await.expect("close");
}

// Port of oauth.test.ts "rejects a response on another path than the expected one" (#10302).
#[tokio::test]
async fn callback_rejects_a_response_on_another_path_than_the_expected_one() {
	let callback = OAuthCallbackServer::listen(OAuthCallbackServerOptions {
		extra_paths: vec!["/callback/server-id".to_string()],
		..Default::default()
	})
	.await
	.expect("listen");
	let fetch = pi_mcp::auth_provider::default_fetch();
	let origin = Url::parse(&callback.redirect_url)
		.expect("url")
		.origin()
		.unicode_serialization();
	let waiter = Arc::clone(&callback);
	let mixed_up =
		tokio::spawn(async move { waiter.wait_for_callback("s1", Some("/callback/server-id")).await });
	// Give the spawned waiter a beat to register its state.
	tokio::time::sleep(std::time::Duration::from_millis(50)).await;
	let wrong_url = Url::parse(&format!("{origin}/callback?code=abc&state=s1")).expect("url");
	let wrong = fetch(pi_mcp::auth_provider::HttpRequest {
		method: "GET".to_string(),
		url: wrong_url,
		headers: Vec::new(),
		body: None,
	})
	.await
	.expect("wrong path hit");
	assert_eq!(wrong.status, 400);
	let error = mixed_up
		.await
		.expect("join")
		.expect_err("path mismatch");
	assert!(error.to_string().contains("arrived on another redirect URI"), "{error}");

	let waiter = Arc::clone(&callback);
	let pending =
		tokio::spawn(async move { waiter.wait_for_callback("s2", Some("/callback/server-id")).await });
	// Give the spawned waiter a beat to register its state.
	tokio::time::sleep(std::time::Duration::from_millis(50)).await;
	let right_url = Url::parse(&format!("{origin}/callback/server-id?code=abc&state=s2")).expect("url");
	let right = fetch(pi_mcp::auth_provider::HttpRequest {
		method: "GET".to_string(),
		url: right_url,
		headers: Vec::new(),
		body: None,
	})
	.await
	.expect("right path hit");
	assert_eq!(right.status, 200);
	assert_eq!(pending.await.expect("join").expect("callback").code, "abc");
	callback.close().await.expect("close");
}

// Port of oauth.test.ts "renders pages through renderPage".
#[tokio::test]
async fn callback_renders_pages_through_render_page() {
	let pages: Arc<StdMutex<Vec<OAuthCallbackPage>>> = Arc::new(StdMutex::new(Vec::new()));
	let pages_sink = Arc::clone(&pages);
	let callback = OAuthCallbackServer::listen(OAuthCallbackServerOptions {
		render_page: Some(Arc::new(move |page| {
			pages_sink
				.lock()
				.unwrap_or_else(std::sync::PoisonError::into_inner)
				.push(page.clone());
			match page {
				OAuthCallbackPage::Ok => "<p>ok</p>".to_string(),
				OAuthCallbackPage::Failure { message, .. } => format!("<p>{message}</p>"),
			}
		})),
		..Default::default()
	})
	.await
	.expect("listen");
	let fetch = pi_mcp::auth_provider::default_fetch();
	let waiter = Arc::clone(&callback);
	let denied = tokio::spawn(async move { waiter.wait_for_callback("s1", None).await });
	// Give the spawned waiter a beat to register its state.
	tokio::time::sleep(std::time::Duration::from_millis(50)).await;
	let failure_url = Url::parse(&format!(
		"{}?error=access_denied&error_description=Denied&state=s1",
		callback.redirect_url
	))
	.expect("url");
	let failure = fetch(pi_mcp::auth_provider::HttpRequest {
		method: "GET".to_string(),
		url: failure_url,
		headers: Vec::new(),
		body: None,
	})
	.await
	.expect("failure hit");
	assert_eq!(failure.header("content-type"), Some("text/html; charset=utf-8"));
	let error = denied.await.expect("join").expect_err("denied");
	assert_eq!(error.to_string(), "Denied");
	assert_eq!(
		pages.lock().unwrap_or_else(std::sync::PoisonError::into_inner).last(),
		Some(&OAuthCallbackPage::Failure {
			message: "Authorization failed. You may close this window.".to_string(),
			details: Some("Denied".to_string()),
		})
	);

	let waiter = Arc::clone(&callback);
	let pending = tokio::spawn(async move { waiter.wait_for_callback("s2", None).await });
	// Give the spawned waiter a beat to register its state.
	tokio::time::sleep(std::time::Duration::from_millis(50)).await;
	let success_url = Url::parse(&format!("{}?code=abc&state=s2", callback.redirect_url)).expect("url");
	let success = fetch(pi_mcp::auth_provider::HttpRequest {
		method: "GET".to_string(),
		url: success_url,
		headers: Vec::new(),
		body: None,
	})
	.await
	.expect("success hit");
	let body = success.text().await.expect("body");
	assert_eq!(body, "<p>ok</p>");
	let callback_value = pending.await.expect("join").expect("callback");
	assert_eq!(callback_value.code, "abc");
	callback.close().await.expect("close");
}

