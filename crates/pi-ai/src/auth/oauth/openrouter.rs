//! Port of `packages/ai/src/auth/oauth/openrouter.ts` — the OpenRouter OAuth
//! PKCE flow.
//!
//! OpenRouter exchanges an authorization code for a permanent API key; the
//! credential keeps `refresh: ""` and `expires: Number.MAX_SAFE_INTEGER` so
//! the standard scheduler never refreshes it (`refresh` returns the
//! credential unchanged).

use crate::auth::oauth::callback_server::{
	start_oauth_callback_server, wait_for_callback_or_manual_input, CallbackOrManual, OAuthCallbackServerOptions,
};
use crate::auth::oauth::pkce::generate_pkce;
use crate::auth::types::{
	ModelAuth, OAuthAuth, OAuthCredential, OAuthCredentials, OAuthLoginFn, OAuthRefreshFn, OAuthToAuthFn,
	ProviderAuthInteraction,
};
use crate::http::{HttpBody, HttpRequest};
use crate::utils::abort::AbortSignal;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::thrown::ThrownValue;

const AUTHORIZE_URL: &str = "https://openrouter.ai/auth";
const TOKEN_URL: &str = "https://openrouter.ai/api/v1/auth/keys";
const LOGIN_TIMEOUT_MS: u64 = 5 * 60 * 1000;
const TOKEN_EXCHANGE_TIMEOUT_MS: u64 = 30_000;

fn callback_host() -> String {
	get_provider_env_value("PI_OAUTH_CALLBACK_HOST", None).unwrap_or_else(|| "127.0.0.1".to_string())
}

/// Random path segment (upstream `crypto.randomUUID()`).
fn random_uuid() -> Result<String, ThrownValue> {
	let mut bytes = [0u8; 16];
	getrandom::getrandom(&mut bytes).map_err(|_| ThrownValue::message("Failed to generate random bytes"))?;
	bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
	bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 10
	let hex = hex::encode(bytes);
	Ok(format!(
		"{}-{}-{}-{}-{}",
		&hex[0..8],
		&hex[8..12],
		&hex[12..16],
		&hex[16..20],
		&hex[20..32]
	))
}

/// Upstream `parseAuthorizationInput` — a bare code or a redirect URL.
fn parse_authorization_input(input: &str) -> Option<String> {
	let value = input.trim();
	if value.is_empty() {
		return None;
	}

	if let Ok(url) = url::Url::parse(value) {
		return url
			.query_pairs()
			.find(|(key, _)| key == "code")
			.map(|(_, value)| value.into_owned());
	}

	if value.contains("code=") {
		return url::form_urlencoded::parse(value.as_bytes())
			.find(|(key, _)| key == "code")
			.map(|(_, value)| value.into_owned());
	}

	Some(value.to_string())
}

/// Upstream `errorDetail` — first present of `error_description`, `message`,
/// `error`, `error.message`.
fn error_detail(body: &serde_json::Value) -> Option<String> {
	for key in ["error_description", "message", "error"] {
		if let Some(value) = body.get(key).and_then(serde_json::Value::as_str) {
			return Some(value.to_string());
		}
	}
	if let Some(message) = body
		.get("error")
		.filter(|error| error.is_object() && !error.is_array())
		.and_then(|error| error.get("message"))
		.and_then(serde_json::Value::as_str)
	{
		return Some(message.to_string());
	}
	None
}

async fn exchange_authorization_code(
	code: &str,
	verifier: &str,
	signal: &AbortSignal,
) -> Result<OAuthCredential, ThrownValue> {
	if signal.aborted() {
		return Err(ThrownValue::message("Login cancelled"));
	}
	let timeout = crate::auth::oauth::timeout_signal(signal, TOKEN_EXCHANGE_TIMEOUT_MS);
	// The timeout abort carries the dedicated message, mirrored below.
	let mut request = HttpRequest::new("POST", TOKEN_URL);
	request.headers.push(("accept".to_string(), "application/json".to_string()));
	request
		.headers
		.push(("content-type".to_string(), "application/json".to_string()));
	request.body = HttpBody::bytes(
		serde_json::json!({
			"code": code,
			"code_verifier": verifier,
			"code_challenge_method": "S256",
		})
		.to_string()
		.into_bytes(),
	);
	request.signal = Some(timeout.clone());

	let exchange = async {
		let response = crate::auth::oauth::oauth_fetch().send(request).await;
		let mut response = match response {
			Ok(response) => response,
			Err(error) => {
				if signal.aborted() {
					return Err(ThrownValue::message("Login cancelled"));
				}
				if timeout.aborted() && !signal.aborted() {
					return Err(ThrownValue::message("OpenRouter OAuth token exchange timed out"));
				}
				return Err(ThrownValue::message(error.message.clone()));
			}
		};
		let status = response.status;
		let body = std::mem::replace(&mut response.body, HttpBody::Empty);
		let text = match body.collect().await {
			Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
			Err(error) => {
				if signal.aborted() {
					return Err(ThrownValue::message("Login cancelled"));
				}
				if timeout.aborted() && !signal.aborted() {
					return Err(ThrownValue::message("OpenRouter OAuth token exchange timed out"));
				}
				return Err(ThrownValue::message(error.message));
			}
		};
		let parsed: serde_json::Value = serde_json::from_str(&text)
			.map_err(|_| ThrownValue::message("OpenRouter OAuth returned invalid JSON"))?;
		if !(200..300).contains(&status) {
			let detail = error_detail(&parsed);
			return Err(ThrownValue::message(format!(
				"OpenRouter OAuth key exchange failed (HTTP {status}){}",
				detail.map(|detail| format!(": {detail}")).unwrap_or_default()
			)));
		}
		match parsed.get("key").and_then(serde_json::Value::as_str) {
			Some(key) if !key.is_empty() => Ok(oauth_credential(key.to_string())),
			_ => Err(ThrownValue::message("OpenRouter OAuth response carries no \"key\"")),
		}
	}
	.await;

	exchange
}

async fn login_openrouter(interaction: &ProviderAuthInteraction) -> Result<OAuthCredential, ThrownValue> {
	let (verifier, challenge) = generate_pkce().await?;
	// OpenRouter sends no `state`; the random path keeps stray requests from
	// completing the sign-in.
	let callback = start_oauth_callback_server(OAuthCallbackServerOptions {
		provider_name: "OpenRouter".to_string(),
		host: callback_host(),
		port: 0,
		path: format!("/oauth/callback/{}", random_uuid()?),
		redirect_host: None,
		state: None,
		complete: {
			let signal = interaction.signal.clone();
			let verifier = verifier.clone();
			std::sync::Arc::new(move |code| {
				let signal = signal.clone();
				let verifier = verifier.clone();
				Box::pin(async move { exchange_authorization_code(&code, &verifier, &signal).await })
			})
		},
		signal: Some(interaction.signal.clone()),
		timeout_ms: Some(LOGIN_TIMEOUT_MS),
	})
	.await?;

	let result = async {
		let mut authorize_url = url::Url::parse(AUTHORIZE_URL).map_err(|_| ThrownValue::message("invalid authorize URL"))?;
		{
			let mut query = authorize_url.query_pairs_mut();
			query.append_pair("callback_url", callback.redirect_uri());
			query.append_pair("code_challenge", &challenge);
			query.append_pair("code_challenge_method", "S256");
		}

		(interaction.notify)(crate::auth::types::AuthEvent::Progress {
			message: format!("Listening for OpenRouter OAuth callback on {}", callback.redirect_uri()),
		});
		(interaction.notify)(crate::auth::types::AuthEvent::AuthUrl {
			url: authorize_url.to_string(),
			instructions: Some(
				"Complete sign-in in your browser. If the browser is on another machine, paste the final redirect URL here."
					.to_string(),
			),
		});

		let result = wait_for_callback_or_manual_input(
			interaction,
			Some(&callback),
			(
				"Complete sign-in in your browser, or paste the authorization code / redirect URL here:",
				callback.redirect_uri(),
			),
		)
		.await?;
		match result {
			CallbackOrManual::Callback(credential) => Ok(credential),
			CallbackOrManual::Manual(input) => {
				let Some(code) = parse_authorization_input(&input) else {
					return Err(ThrownValue::message("Missing authorization code"));
				};
				(interaction.notify)(crate::auth::types::AuthEvent::Progress {
					message: "Exchanging authorization code for an API key...".to_string(),
				});
				exchange_authorization_code(&code, &verifier, &interaction.signal).await
			}
		}
	}
	.await;

	callback.close();
	result
}

/// Upstream `export const openRouterOAuth`.
pub fn openrouter_oauth() -> OAuthAuth {
	let login: OAuthLoginFn = Box::new(|interaction, _options| Box::pin(async move { login_openrouter(&interaction).await }));

	let refresh: OAuthRefreshFn = Box::new(|credential, _signal| Box::pin(async move { Ok(credential) }));

	let to_auth: OAuthToAuthFn = Box::new(|credential| {
		Box::pin(async move {
			Ok(ModelAuth { api_key: Some(credential.credentials.access.clone()), headers: None, base_url: None })
		})
	});

	OAuthAuth {
		name: "OpenRouter OAuth".to_string(),
		is_subscription: false,
		login_label: Some("Sign in with OpenRouter".to_string()),
		login,
		refresh,
		to_auth,
	}
}

fn oauth_credential(access: String) -> OAuthCredential {
	OAuthCredential {
		credentials: OAuthCredentials {
			refresh: String::new(),
			access,
			// Number.MAX_SAFE_INTEGER
			expires: 9_007_199_254_740_991,
			extra: Default::default(),
		},
	}
}
