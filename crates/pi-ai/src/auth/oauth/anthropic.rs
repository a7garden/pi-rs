//! Port of `packages/ai/src/auth/oauth/anthropic.ts` — the Anthropic OAuth
//! flow (Claude Pro/Max).
//!
//! Upstream reads `PI_OAUTH_CALLBACK_HOST` at module load (`const`); the Rust
//! port keeps that with a `LazyLock` static (evaluated on first login).
//! `formatErrorDetails` surfaces `name: message`; Node-specific `code` /
//! `errno` / `stack` fields do not exist on Rust errors, so only the message
//! and cause chain are rendered (deviation D30).

use std::sync::LazyLock;

use crate::auth::oauth::callback_server::{
	boxed_complete, start_oauth_callback_server, wait_for_callback_or_manual_input, CallbackOrManual,
	OAuthCallbackServerOptions,
};
use crate::auth::oauth::pkce::generate_pkce;
use crate::auth::types::{
	ModelAuth, OAuthAuth, OAuthCredential, OAuthCredentials, OAuthLoginFn, OAuthRefreshFn, OAuthToAuthFn,
	ProviderAuthInteraction,
};
use crate::http::{default_fetch, FetchFn, HttpBody, HttpRequest};
use crate::utils::abort::AbortSignal;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::thrown::ThrownValue;

/// Upstream `atob("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl")`.
const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const CALLBACK_PORT: u16 = 53692;
const CALLBACK_PATH: &str = "/callback";
const REDIRECT_URI: &str = "http://localhost:53692/callback";
const COPY_CODE_REDIRECT_URI: &str = "https://platform.claude.com/oauth/code/callback";
const ANTHROPIC_BROWSER_LOGIN_METHOD: &str = "browser";
const ANTHROPIC_COPY_CODE_LOGIN_METHOD: &str = "copy_code";
const SCOPES: &str =
	"org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";

fn callback_host() -> &'static str {
	static CALLBACK_HOST: LazyLock<String> = LazyLock::new(|| {
		get_provider_env_value("PI_OAUTH_CALLBACK_HOST", None).unwrap_or_else(|| "127.0.0.1".to_string())
	});
	CALLBACK_HOST.as_str()
}

/// Upstream `parseAuthorizationInput` → `{ code?, state? }`.
fn parse_authorization_input(input: &str) -> (Option<String>, Option<String>) {
	let value = input.trim();
	if value.is_empty() {
		return (None, None);
	}

	if let Ok(url) = url::Url::parse(value) {
		let query: std::collections::BTreeMap<String, String> = url
			.query_pairs()
			.map(|(key, item)| (key.into_owned(), item.into_owned()))
			.collect();
		return (query.get("code").cloned(), query.get("state").cloned());
	}

	if value.contains('#') {
		let mut parts = value.splitn(2, '#');
		let code = parts.next().map(str::to_string);
		let state = parts.next().map(str::to_string);
		return (code, state);
	}

	if value.contains("code=") {
		let params: std::collections::BTreeMap<String, String> = url::form_urlencoded::parse(value.as_bytes())
			.map(|(key, item)| (key.into_owned(), item.into_owned()))
			.collect();
		return (params.get("code").cloned(), params.get("state").cloned());
	}

	(Some(value.to_string()), None)
}

/// Upstream `formatErrorDetails`. Rust errors carry no `code`/`errno`/`stack`
/// (deviation D30); the message is rendered.
fn format_error_details(error: &ThrownValue) -> String {
	let name = error.error_name().unwrap_or_else(|| "Error".to_string());
	format!("{name}: {}", crate::utils::diagnostics::format_thrown_value(error))
}

/// Upstream `postJson` — POST with JSON body, 30s timeout, response text.
async fn post_json(url: &str, body: serde_json::Value, signal: &AbortSignal) -> Result<String, ThrownValue> {
	let mut request = HttpRequest::new("POST", url);
	request.headers.push(("Content-Type".to_string(), "application/json".to_string()));
	request.headers.push(("Accept".to_string(), "application/json".to_string()));
	request.body = HttpBody::bytes(serde_json::to_vec(&body).expect("request body serializes"));
	request.signal = Some(crate::auth::oauth::timeout_signal(signal, 30_000));

	let response = crate::auth::oauth::oauth_fetch()
		.send(request)
		.await
		.map_err(|error| ThrownValue::message(error.message.clone()))?;
	let status = response.status;
	let body_bytes = response
		.body
		.collect()
		.await
		.map_err(|error| ThrownValue::message(error.message.clone()))?;
	let response_body = String::from_utf8_lossy(&body_bytes).into_owned();

	if !(200..300).contains(&status) {
		return Err(ThrownValue::message(format!(
			"HTTP request failed. status={status}; url={url}; body={response_body}"
		)));
	}

	Ok(response_body)
}

struct TokenData {
	refresh_token: String,
	access_token: String,
	expires_in: f64,
}

/// Upstream `exchangeAuthorizationCode` request + parse (error strings
/// interpolate the actual redirect URI, like upstream's parameter).
async fn exchange_token_response(
	body: serde_json::Value,
	redirect_uri: &str,
	signal: &AbortSignal,
) -> Result<TokenData, ThrownValue> {
	let response_body = post_json(TOKEN_URL, body, signal).await.map_err(|error| {
		ThrownValue::message(format!(
			"Token exchange request failed. url={TOKEN_URL}; redirect_uri={redirect_uri}; response_type=authorization_code; details={}",
			format_error_details(&error)
		))
	})?;

	let data: serde_json::Value = serde_json::from_str(&response_body).map_err(|error| {
		ThrownValue::message(format!(
			"Token exchange returned invalid JSON. url={TOKEN_URL}; body={response_body}; details=Error: {error}"
		))
	})?;

	Ok(TokenData {
		access_token: data.get("access_token").and_then(serde_json::Value::as_str).unwrap_or_default().to_string(),
		refresh_token: data.get("refresh_token").and_then(serde_json::Value::as_str).unwrap_or_default().to_string(),
		expires_in: data.get("expires_in").and_then(serde_json::Value::as_f64).unwrap_or(0.0),
	})
}

/// Upstream `expires` arithmetic: `Date.now() + expires_in * 1000 - 5 * 60 * 1000`.
fn expires_with_skew(expires_in_seconds: f64) -> i64 {
	crate::utils::diagnostics::now_ms() + (expires_in_seconds * 1000.0) as i64 - 5 * 60 * 1000
}

fn oauth_credential(refresh: &str, access: &str, expires: i64) -> OAuthCredential {
	OAuthCredential {
		credentials: OAuthCredentials {
			refresh: refresh.to_string(),
			access: access.to_string(),
			expires,
			extra: Default::default(),
		},
	}
}

async fn exchange_authorization_code(
	code: &str,
	state: &str,
	verifier: &str,
	redirect_uri: &str,
	signal: &AbortSignal,
) -> Result<OAuthCredential, ThrownValue> {
	let data = exchange_token_response(
		serde_json::json!({
			"grant_type": "authorization_code",
			"client_id": CLIENT_ID,
			"code": code,
			"state": state,
			"redirect_uri": redirect_uri,
			"code_verifier": verifier,
		}),
		redirect_uri,
		signal,
	)
	.await?;

	Ok(oauth_credential(&data.refresh_token, &data.access_token, expires_with_skew(data.expires_in)))
}

async fn login_anthropic(interaction: &ProviderAuthInteraction) -> Result<OAuthCredential, ThrownValue> {
	let (verifier, challenge) = generate_pkce().await?;
	let callback = start_oauth_callback_server(OAuthCallbackServerOptions {
		provider_name: "Anthropic".to_string(),
		host: callback_host().to_string(),
		port: CALLBACK_PORT,
		path: CALLBACK_PATH.to_string(),
		redirect_host: Some("localhost".to_string()),
		state: Some(verifier.clone()),
		complete: boxed_complete(|code| async move { Ok::<String, ThrownValue>(code) }),
		signal: Some(interaction.signal.clone()),
		timeout_ms: None,
	})
	.await
	.ok();

	let result = async {
		let authorize_params = crate::auth::oauth::form_urlencode(&[
			("code", "true"),
			("client_id", CLIENT_ID),
			("response_type", "code"),
			("redirect_uri", REDIRECT_URI),
			("scope", SCOPES),
			("code_challenge", challenge.as_str()),
			("code_challenge_method", "S256"),
			("state", verifier.as_str()),
		]);
		(interaction.notify)(crate::auth::types::AuthEvent::AuthUrl {
			url: format!("{AUTHORIZE_URL}?{authorize_params}"),
			instructions: Some(
				"Complete login in your browser. If the browser is on another machine, paste the final redirect URL here."
					.to_string(),
			),
		});

		let result = wait_for_callback_or_manual_input(
			interaction,
			callback.as_ref(),
			(
				"Complete login in your browser, or paste the authorization code / redirect URL here:",
				REDIRECT_URI,
			),
		)
		.await?;
		let (code, state) = match result {
			CallbackOrManual::Callback(value) => (Some(value), verifier.clone()),
			CallbackOrManual::Manual(input) => {
				let (parsed_code, parsed_state) = parse_authorization_input(&input);
				if parsed_state.as_deref().is_some_and(|state| state != verifier) {
					return Err(ThrownValue::message("OAuth state mismatch"));
				}
				(parsed_code, parsed_state.unwrap_or_else(|| verifier.clone()))
			}
		};

		let Some(code) = code else {
			return Err(ThrownValue::message("Missing authorization code"));
		};
		(interaction.notify)(crate::auth::types::AuthEvent::Progress {
			message: "Exchanging authorization code for tokens...".to_string(),
		});
		exchange_authorization_code(&code, &state, &verifier, REDIRECT_URI, &interaction.signal).await
	}
	.await;

	if let Some(callback) = callback {
		callback.close();
	}
	result
}

async fn login_anthropic_copy_code(interaction: &ProviderAuthInteraction) -> Result<OAuthCredential, ThrownValue> {
	let (verifier, challenge) = generate_pkce().await?;
	let authorize_params = crate::auth::oauth::form_urlencode(&[
		("code", "true"),
		("client_id", CLIENT_ID),
		("response_type", "code"),
		("redirect_uri", COPY_CODE_REDIRECT_URI),
		("scope", SCOPES),
		("code_challenge", challenge.as_str()),
		("code_challenge_method", "S256"),
		("state", verifier.as_str()),
	]);
	(interaction.notify)(crate::auth::types::AuthEvent::AuthUrl {
		url: format!("{AUTHORIZE_URL}?{authorize_params}"),
		instructions: Some(
			"Complete login in your browser, then copy the code Anthropic shows and paste it here.".to_string(),
		),
	});

	let input = (interaction.prompt)(crate::auth::types::AuthPrompt {
		signal: Some(interaction.signal.clone()),
		kind: crate::auth::types::AuthPromptKind::ManualCode {
			message: "Paste the code Anthropic shows after you sign in:".to_string(),
			placeholder: Some("code#state".to_string()),
		},
	})
	.await?;
	let (code, state) = parse_authorization_input(&input);
	if state.as_deref().is_some_and(|state| state != verifier) {
		return Err(ThrownValue::message("OAuth state mismatch"));
	}
	let Some(code) = code else {
		return Err(ThrownValue::message("Missing authorization code"));
	};
	(interaction.notify)(crate::auth::types::AuthEvent::Progress {
		message: "Exchanging authorization code for tokens...".to_string(),
	});
	exchange_authorization_code(
		&code,
		state.as_deref().unwrap_or(&verifier),
		&verifier,
		COPY_CODE_REDIRECT_URI,
		&interaction.signal,
	)
	.await
}

/// Refresh Anthropic OAuth token.
async fn refresh_anthropic_token(refresh_token: &str, signal: &AbortSignal) -> Result<OAuthCredential, ThrownValue> {
	let response_body = post_json(
		TOKEN_URL,
		serde_json::json!({
			"grant_type": "refresh_token",
			"client_id": CLIENT_ID,
			"refresh_token": refresh_token,
		}),
		signal,
	)
	.await
	.map_err(|error| {
		ThrownValue::message(format!(
			"Anthropic token refresh request failed. url={TOKEN_URL}; details={}",
			format_error_details(&error)
		))
	})?;

	let data: serde_json::Value = serde_json::from_str(&response_body).map_err(|error| {
		ThrownValue::message(format!(
			"Anthropic token refresh returned invalid JSON. url={TOKEN_URL}; body={response_body}; details=Error: {error}"
		))
	})?;

	Ok(oauth_credential(
		data.get("refresh_token").and_then(serde_json::Value::as_str).unwrap_or_default(),
		data.get("access_token").and_then(serde_json::Value::as_str).unwrap_or_default(),
		expires_with_skew(data.get("expires_in").and_then(serde_json::Value::as_f64).unwrap_or(0.0)),
	))
}

/// Upstream `export const anthropicOAuth` — a constructor because `OAuthAuth`
/// holds boxed callbacks (crate convention).
pub fn anthropic_oauth() -> OAuthAuth {
	let login: OAuthLoginFn = Box::new(|interaction, _options| {
		Box::pin(async move {
			let method = (interaction.prompt)(crate::auth::types::AuthPrompt {
				signal: Some(interaction.signal.clone()),
				kind: crate::auth::types::AuthPromptKind::Select {
					message: "Select Anthropic login method:".to_string(),
					options: vec![
						crate::auth::types::AuthSelectOption {
							id: ANTHROPIC_BROWSER_LOGIN_METHOD.to_string(),
							label: "Browser login (default)".to_string(),
							description: None,
						},
						crate::auth::types::AuthSelectOption {
							id: ANTHROPIC_COPY_CODE_LOGIN_METHOD.to_string(),
							label: "Copy code login (headless)".to_string(),
							description: None,
						},
					],
				},
			})
			.await?;

			if method == ANTHROPIC_COPY_CODE_LOGIN_METHOD {
				return login_anthropic_copy_code(&interaction).await;
			}
			if method != ANTHROPIC_BROWSER_LOGIN_METHOD {
				return Err(ThrownValue::message(format!("Unknown Anthropic login method: {method}")));
			}

			login_anthropic(&interaction).await
		})
	});

	let refresh: OAuthRefreshFn = Box::new(|credential, signal| {
		Box::pin(async move { refresh_anthropic_token(&credential.credentials.refresh, &signal).await })
	});

	let to_auth: OAuthToAuthFn = Box::new(|credential| {
		Box::pin(async move {
			Ok(ModelAuth { api_key: Some(credential.credentials.access.clone()), headers: None, base_url: None })
		})
	});

	OAuthAuth {
		name: "Anthropic (Claude Pro/Max)".to_string(),
		is_subscription: true,
		login_label: None,
		login,
		refresh,
		to_auth,
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::auth::oauth::OAuthTestExt;
	use crate::auth::types::{AuthEvent, AuthPromptKind};
	use crate::http::{HttpResponse, ProviderHttpError};
	use crate::utils::abort::AbortController;
	use std::sync::{Arc, Mutex};

	fn never_aborted_signal() -> AbortSignal {
		AbortController::new().signal()
	}

	fn json_response(body: serde_json::Value, status: u16) -> HttpResponse {
		HttpResponse {
			status,
			headers: vec![("content-type".to_string(), "application/json".to_string())],
			body: HttpBody::bytes(serde_json::to_vec(&body).expect("json")),
		}
	}

	async fn body_json(body: HttpBody) -> serde_json::Value {
		let bytes = body.collect().await.expect("body");
		serde_json::from_slice(&bytes).expect("json body")
	}

	fn stub_fetch(
		routes: impl Fn(
				HttpRequest,
			) -> futures_util::future::BoxFuture<'static, Result<HttpResponse, ProviderHttpError>>
			+ Send
			+ Sync
			+ 'static,
	) -> FetchFn {
		FetchFn::new(move |request| routes(request))
	}

	fn interaction(
		signal: AbortSignal,
		prompt: impl Fn(
				crate::auth::types::AuthPrompt,
			) -> futures_util::future::BoxFuture<'static, Result<String, ThrownValue>>
			+ Send
			+ Sync
			+ 'static,
		notify: impl Fn(AuthEvent) + Send + Sync + 'static,
	) -> ProviderAuthInteraction {
		ProviderAuthInteraction { signal, prompt: Box::new(prompt), notify: Box::new(notify) }
	}

	fn auth_query(auth_url: &str, key: &str) -> String {
		url::Url::parse(auth_url)
			.expect("auth url")
			.query_pairs()
			.find(|(name, _)| name == key)
			.map(|(_, value)| value.into_owned())
			.unwrap_or_else(|| panic!("missing {key} in {auth_url}"))
	}

	// Port of test/anthropic-oauth.test.ts "keeps the localhost redirect_uri for
	// manual callback login".
	#[tokio::test]
	async fn keeps_the_localhost_redirect_uri_for_manual_callback_login() {
		let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
		let seen: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
		let seen_for_fetch = seen.clone();
		crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |mut request| {
				let seen_for_fetch = seen_for_fetch.clone();
				Box::pin(async move {
				
			assert_eq!(request.url, TOKEN_URL);
			assert_eq!(request.method, "POST");
			let body = body_json(std::mem::replace(&mut request.body, HttpBody::Empty)).await;
			assert_eq!(body["grant_type"], "authorization_code");
			assert_eq!(body["code"], "manual-code");
			assert_eq!(body["redirect_uri"], "http://localhost:53692/callback");
			seen_for_fetch.lock().unwrap().push(body);
			Ok(json_response(
				serde_json::json!({
					"access_token": "access-token",
					"refresh_token": "refresh-token",
					"expires_in": 3600,
				}),
				200,
			))
			})
		})));

		let auth_url: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
		let auth_url_for_notify = auth_url.clone();
		let auth_url_for_prompt = auth_url.clone();
		let credentials = anthropic_oauth()
			.login(
				interaction(
					never_aborted_signal(),
					move |prompt| {
					let auth_url_for_prompt = auth_url_for_prompt.clone();
					Box::pin(async move { match &prompt.kind {
						AuthPromptKind::Select { .. } => Ok("browser".to_string()),
						AuthPromptKind::ManualCode { .. } => {
							let current = auth_url_for_prompt.lock().unwrap().clone();
							Ok(format!(
								"{}?code=manual-code&state={}",
								auth_query(&current, "redirect_uri"),
								auth_query(&current, "state")
							))
						}
						_ => panic!("Unexpected prompt"),
									} }) as futures_util::future::BoxFuture<'static, Result<String, ThrownValue>>
				},
					move |event| {
						if let AuthEvent::AuthUrl { url, .. } = event {
							*auth_url_for_notify.lock().unwrap() = url;
						}
					},
				),
				None,
			)
			.await
			.expect("credentials");

		assert_eq!(credentials.credentials.access, "access-token");
		assert_eq!(credentials.credentials.refresh, "refresh-token");
		assert_eq!(seen.lock().unwrap().len(), 1);
		crate::auth::oauth::set_oauth_fetch(None);
	}

	// Port of "cancels when Anthropic login method selection is cancelled".
	#[tokio::test]
	async fn cancels_when_login_method_selection_is_cancelled() {
		let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
		crate::auth::oauth::set_oauth_fetch(None);
		let error = anthropic_oauth()
			.login(
				interaction(
					never_aborted_signal(),
					|_prompt| {
					Box::pin(async { Err(ThrownValue::message("Login cancelled")) })
						as futures_util::future::BoxFuture<'static, Result<String, ThrownValue>>
				},
					|_event| {},
				),
				None,
			)
			.await
			.expect_err("cancelled");
		assert_eq!(crate::models::thrown_message(&error), "Login cancelled");
	}

	// Port of "omits scope from refresh token requests".
	#[tokio::test]
	async fn omits_scope_from_refresh_token_requests() {
		let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
		let seen: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
		let seen_for_fetch = seen.clone();
		crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |mut request| {
				let seen_for_fetch = seen_for_fetch.clone();
				Box::pin(async move {
				let seen_for_fetch = seen_for_fetch.clone();
				
			assert_eq!(request.url, TOKEN_URL);
			assert_eq!(request.method, "POST");
			let body = body_json(std::mem::replace(&mut request.body, HttpBody::Empty)).await;
			assert_eq!(body["grant_type"], "refresh_token");
			assert!(body["client_id"].as_str().is_some_and(|id| !id.is_empty()));
			assert_eq!(body["refresh_token"], "refresh-token");
			assert!(body.get("scope").is_none(), "scope must be omitted");
			seen_for_fetch.lock().unwrap().push(body);
			Ok(json_response(
				serde_json::json!({
					"access_token": "new-access-token",
					"refresh_token": "new-refresh-token",
					"expires_in": 3600,
				}),
				200,
			))
			})
		})));

		let credentials = anthropic_oauth()
			.refresh(oauth_credential("refresh-token", "old-access-token", 0), never_aborted_signal())
			.await
			.expect("credentials");

		assert_eq!(credentials.credentials.access, "new-access-token");
		assert_eq!(credentials.credentials.refresh, "new-refresh-token");
		assert_eq!(seen.lock().unwrap().len(), 1);
		crate::auth::oauth::set_oauth_fetch(None);
	}

	// Port of "offers browser login first and uses the selected Anthropic copy
	// code flow" (prompt order + redirect_uri + code#state parsing).
	#[tokio::test]
	async fn offers_browser_login_first_and_uses_copy_code_flow() {
		let _guard = crate::auth::oauth::test_fetch_guard().lock().await;

		let select_prompts: Arc<Mutex<Vec<crate::auth::types::AuthPrompt>>> = Arc::new(Mutex::new(Vec::new()));
		let select_for_prompt = select_prompts.clone();
		let auth_url: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
		let auth_url_for_notify = auth_url.clone();
		let auth_url_for_prompt = auth_url.clone();
		let auth_url_for_fetch = auth_url.clone();
		crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |mut request| {
				let auth_url_for_fetch = auth_url_for_fetch.clone();
				Box::pin(async move {
				
			assert_eq!(request.url, TOKEN_URL);
			let body = body_json(std::mem::replace(&mut request.body, HttpBody::Empty)).await;
			assert_eq!(body["grant_type"], "authorization_code");
			assert_eq!(body["code"], "copied-code");
			assert_eq!(
				body["state"],
				auth_query(&auth_url_for_fetch.lock().unwrap(), "state")
			);
			assert_eq!(body["redirect_uri"], "https://platform.claude.com/oauth/code/callback");
			Ok(json_response(
				serde_json::json!({
					"access_token": "access-token",
					"refresh_token": "refresh-token",
					"expires_in": 3600,
				}),
				200,
			))
			})
		})));

		let credentials = anthropic_oauth()
			.login(
				interaction(
					never_aborted_signal(),
					move |prompt| {
					let select_for_prompt = select_for_prompt.clone();
					let auth_url_for_prompt = auth_url_for_prompt.clone();
					Box::pin(async move { match &prompt.kind {
						AuthPromptKind::Select { message, options } => {
							select_for_prompt.lock().unwrap().push(crate::auth::types::AuthPrompt {
								signal: prompt.signal.clone(),
								kind: AuthPromptKind::Select { message: message.clone(), options: options.clone() },
							});
							Ok("copy_code".to_string())
						}
						AuthPromptKind::ManualCode { .. } => {
							let state = auth_query(&auth_url_for_prompt.lock().unwrap(), "state");
							Ok(format!("copied-code#{state}"))
						}
						_ => panic!("Unexpected prompt"),
									} }) as futures_util::future::BoxFuture<'static, Result<String, ThrownValue>>
				},
					move |event| {
						if let AuthEvent::AuthUrl { url, .. } = event {
							*auth_url_for_notify.lock().unwrap() = url;
						}
					},
				),
				None,
			)
			.await
			.expect("credentials");

		assert_eq!(credentials.credentials.access, "access-token");
		assert_eq!(
			auth_query(&auth_url.lock().unwrap(), "redirect_uri"),
			"https://platform.claude.com/oauth/code/callback"
		);
		{
			let prompts = select_prompts.lock().unwrap();
			assert_eq!(prompts.len(), 1);
			match &prompts[0].kind {
				AuthPromptKind::Select { message, options } => {
					assert_eq!(message, "Select Anthropic login method:");
					assert_eq!(options.len(), 2);
					assert_eq!(options[0].id, "browser");
					assert_eq!(options[0].label, "Browser login (default)");
					assert_eq!(options[1].id, "copy_code");
					assert_eq!(options[1].label, "Copy code login (headless)");
				}
				_ => panic!("expected select prompt"),
			}
		}
		crate::auth::oauth::set_oauth_fetch(None);
	}

	// Port of "anthropicOAuth.login resolves through the manual_code prompt and
	// aborts it after settling".
	#[tokio::test]
	async fn aborts_the_manual_prompt_after_login_settles() {
		let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
		crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |request| Box::pin(async move {
			assert!(request.url.contains("/oauth/token"));
			Ok(json_response(
				serde_json::json!({ "access_token": "access", "refresh_token": "refresh", "expires_in": 3600 }),
				200,
			))
		}))));

		let saw_auth_url = Arc::new(Mutex::new(false));
		let saw_manual = Arc::new(Mutex::new(false));
		let manual_signal: Arc<Mutex<Option<AbortSignal>>> = Arc::new(Mutex::new(None));
		let (saw_auth_for_notify, saw_manual_for_prompt, signal_for_prompt) =
			(saw_auth_url.clone(), saw_manual.clone(), manual_signal.clone());

		let credential = anthropic_oauth()
			.login(
				interaction(
					never_aborted_signal(),
					move |prompt| {
					let saw_manual_for_prompt = saw_manual_for_prompt.clone();
					let signal_for_prompt = signal_for_prompt.clone();
					Box::pin(async move { match &prompt.kind {
						AuthPromptKind::Select { .. } => Ok("browser".to_string()),
						AuthPromptKind::ManualCode { .. } => {
							*saw_manual_for_prompt.lock().unwrap() = true;
							*signal_for_prompt.lock().unwrap() = prompt.signal.clone();
							Ok("the-code".to_string())
						}
						_ => panic!("Unexpected prompt"),
									} }) as futures_util::future::BoxFuture<'static, Result<String, ThrownValue>>
				},
					move |event| {
						if let AuthEvent::AuthUrl { .. } = event {
							*saw_auth_for_notify.lock().unwrap() = true;
						}
					},
				),
				None,
			)
			.await
			.expect("credential");

		assert_eq!(credential.credentials.access, "access");
		assert!(*saw_auth_url.lock().unwrap());
		assert!(*saw_manual.lock().unwrap());
		// the prompt's signal is aborted once login settles, so UIs can dismiss it
		assert!(manual_signal.lock().unwrap().as_ref().expect("manual signal").aborted());
		crate::auth::oauth::set_oauth_fetch(None);
	}

	// Port of "completes login through the browser callback and shows the
	// sign-in page" — the callback server is real loopback HTTP.
	#[tokio::test]
	async fn completes_login_through_the_browser_callback() {
		let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
		let exchanged: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
		let exchanged_for_fetch = exchanged.clone();
		crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |mut request| {
				let exchanged_for_fetch = exchanged_for_fetch.clone();
				Box::pin(async move {
				
			assert_eq!(request.url, TOKEN_URL);
			let body = body_json(std::mem::replace(&mut request.body, HttpBody::Empty)).await;
			*exchanged_for_fetch.lock().unwrap() = Some(body["code"].as_str().expect("code").to_string());
			Ok(json_response(
				serde_json::json!({ "access_token": "access", "refresh_token": "refresh", "expires_in": 3600 }),
				200,
			))
			})
		})));

		// The browser hits the real loopback server when auth_url fires.
		let (page_tx, page_rx) = tokio::sync::oneshot::channel::<u16>();
		let page_tx = Arc::new(std::sync::Mutex::new(Some(page_tx)));

		let credential = anthropic_oauth()
			.login(
				interaction(
					never_aborted_signal(),
					|prompt| {
					Box::pin(async move { match &prompt.kind {
						AuthPromptKind::Select { .. } => Ok("browser".to_string()),
						AuthPromptKind::ManualCode { .. } => {
							// pending until the flow aborts its signal after settling
							let signal = prompt.signal.clone().expect("manual prompt signal");
							let _ = signal.aborted_future().await;
							Err(ThrownValue::message("prompt aborted"))
						}
						_ => panic!("Unexpected prompt"),
										} }) as futures_util::future::BoxFuture<'static, Result<String, ThrownValue>>
				},
					move |event| {
						if let AuthEvent::AuthUrl { url, .. } = event {
							let state = auth_query(&url, "state");
							let page_tx = page_tx.clone();
							tokio::spawn(async move {
								let request =
									HttpRequest::new("GET", format!("http://127.0.0.1:53692/callback?code=browser-code&state={state}"));
								if let Ok(response) = default_fetch().send(request).await {
									// oneshot permits exactly one send; take the
									// sender out so a repeat browser hit is a no-op.
									if let Some(sender) = page_tx.lock().unwrap().take() {
										let _ = sender.send(response.status);
									}
								}
							});
						}
					},
				),
				None,
			)
			.await
			.expect("credential");

		assert_eq!(credential.credentials.access, "access");
		assert_eq!(exchanged.lock().unwrap().as_deref(), Some("browser-code"));
		let status = tokio::time::timeout(std::time::Duration::from_secs(5), page_rx)
			.await
			.expect("page response in time")
			.expect("page sent");
		assert_eq!(status, 200);
		crate::auth::oauth::set_oauth_fetch(None);
	}
}
