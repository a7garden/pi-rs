//! Port of `packages/ai/src/auth/oauth/radius.ts` — the Radius gateway OAuth
//! flow.
//!
//! Radius is a pi-messages gateway. OAuth client APIs live on the configured
//! gateway; only the interactive browser authorization endpoint is discovered.
//!
//! # Deviations
//!
//! - D33: `normalizeRadiusGatewayUrl` (upstream `providers/radius-config.ts`,
//!   sibling module `providers/radius_config.rs`) is mirrored locally in
//!   [`normalize_gateway_url`] — an isolated two-line helper; final
//!   integration can point callers at the sibling's function.

use crate::auth::oauth::callback_server::{
	boxed_complete, start_oauth_callback_server, OAuthCallbackServerOptions,
};
use crate::auth::oauth::device_code::{poll_oauth_device_code_flow, OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult};
use crate::auth::oauth::pkce::generate_pkce;
use crate::auth::types::{
	ModelAuth, OAuthAuth, OAuthCredential, OAuthCredentials, OAuthLoginFn, OAuthRefreshFn, OAuthToAuthFn,
	ProviderAuthInteraction,
};
use crate::http::{HttpBody, HttpRequest, HttpResponse};
use crate::utils::abort::AbortSignal;
use crate::utils::thrown::ThrownValue;

const CALLBACK_HOST: &str = "127.0.0.1";
const CALLBACK_PORT: u16 = 1456;
const CALLBACK_PATH: &str = "/oauth/callback";
const REDIRECT_URI: &str = "http://127.0.0.1:1456/oauth/callback";
const TOKEN_EXPIRY_SKEW_MS: i64 = 60_000;
const LOGIN_METHOD_BROWSER: &str = "browser";
const LOGIN_METHOD_DEVICE_CODE: &str = "device-code";
const OAUTH_CLIENT_ID: &str = "pi-gateway";
const OAUTH_SCOPE: &str = "gateway offline_access";
const OAUTH_DEVICE_CODE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";

/// Upstream `normalizeRadiusGatewayUrl` (deviation D33).
pub(crate) fn normalize_gateway_url(value: &str) -> String {
	let with_scheme = if value.starts_with("http://") || value.starts_with("https://") {
		value.to_string()
	} else {
		format!("https://{value}")
	};
	with_scheme.trim_end_matches('/').to_string()
}

struct RadiusOAuthDiscovery {
	authorization_endpoint: String,
}

struct DeviceAuthorizationResponse {
	device_code: String,
	user_code: String,
	verification_uri: String,
	expires_in: f64,
	interval: Option<f64>,
}

async fn response_text(response: &mut HttpResponse) -> String {
	let body = std::mem::replace(&mut response.body, HttpBody::Empty);
	match body.collect().await {
		Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
		Err(_) => String::new(),
	}
}

async fn gateway_fetch(request: HttpRequest, signal: &AbortSignal) -> Result<HttpResponse, ThrownValue> {
	let mut request = request;
	request.signal = Some(signal.clone());
	match crate::auth::oauth::oauth_fetch().send(request).await {
		Ok(response) => Ok(response),
		Err(error) => {
			if signal.aborted() {
				return Err(ThrownValue::message("Login cancelled"));
			}
			Err(ThrownValue::message(error.message.clone()))
		}
	}
}

async fn load_radius_oauth_discovery(gateway: &str, signal: &AbortSignal) -> Result<RadiusOAuthDiscovery, ThrownValue> {
	let url = url::Url::parse(&format!("{gateway}/v1/oauth")).map_err(|_| ThrownValue::message("invalid gateway URL"))?;
	let response = gateway_fetch(
		HttpRequest::new("GET", url.as_str()).header("accept", "application/json"),
		signal,
	)
	.await?;

	let status = response.status;
	if !(200..300).contains(&status) {
		let mut response = response;
		let text = response_text(&mut response).await;
		return Err(ThrownValue::message(format!(
			"Could not load Radius OAuth config from {gateway}: {status} {text}"
		)));
	}

	let bytes = response.body.collect().await.map_err(|error| ThrownValue::message(error.message))?;
	let discovery: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| ThrownValue::message(error.to_string()))?;
	let authorization_endpoint = discovery
		.get("authorizationEndpoint")
		.and_then(serde_json::Value::as_str)
		.unwrap_or_default();
	if authorization_endpoint.is_empty() {
		return Err(ThrownValue::message(format!("Invalid Radius OAuth config from {gateway}")));
	}
	Ok(RadiusOAuthDiscovery { authorization_endpoint: authorization_endpoint.to_string() })
}

/// Upstream `OAuthResponseError` — the composed message plus the `error` code
/// (kept separately for the device-flow polling switch).
#[derive(Debug)]
struct OAuthResponseError {
	message: String,
	oauth_error: Option<String>,
}

impl OAuthResponseError {
	fn into_thrown_value(self) -> ThrownValue {
		ThrownValue::message(self.message)
	}
}

async fn read_oauth_response_error(mut response: HttpResponse, message: &str) -> OAuthResponseError {
	let text = response_text(&mut response).await;
	let mut oauth_error: Option<String> = None;
	let mut description: Option<String> = None;

	if !text.is_empty() {
		match serde_json::from_str::<serde_json::Value>(&text) {
			Ok(data) => {
				oauth_error = data.get("error").and_then(serde_json::Value::as_str).map(str::to_string);
				description = data
					.get("error_description")
					.and_then(serde_json::Value::as_str)
					.map(str::to_string);
			}
			Err(_) => description = Some(text),
		}
	}

	let detail = match (&oauth_error, &description) {
		(Some(oauth_error), Some(description)) => format!("{oauth_error}: {description}"),
		(Some(oauth_error), None) => oauth_error.clone(),
		(None, Some(description)) => description.clone(),
		(None, None) => response.status.to_string(),
	};
	OAuthResponseError { message: format!("{message}: {detail}"), oauth_error }
}

async fn request_oauth_token(
	gateway: &str,
	body: Vec<(&str, &str)>,
	signal: &AbortSignal,
) -> Result<OAuthCredential, ThrownValue> {
	let url = url::Url::parse(&format!("{gateway}/v1/oauth/token")).map_err(|_| ThrownValue::message("invalid gateway URL"))?;
	let mut request = HttpRequest::new("POST", url.as_str());
	request.headers.push(("accept".to_string(), "application/json".to_string()));
	request
		.headers
		.push(("content-type".to_string(), "application/x-www-form-urlencoded".to_string()));
	request.body = HttpBody::bytes(crate::auth::oauth::form_urlencode(&body).into_bytes());
	let response = gateway_fetch(request, signal).await?;

	if !response.ok() {
		return Err(read_oauth_response_error(response, "Radius OAuth token request failed").await.into_thrown_value());
	}

	let bytes = response.body.collect().await.map_err(|error| ThrownValue::message(error.message))?;
	let data: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| ThrownValue::message(error.to_string()))?;
	let access_token = data.get("access_token").and_then(serde_json::Value::as_str).unwrap_or_default();
	let refresh_token = data.get("refresh_token").and_then(serde_json::Value::as_str).unwrap_or_default();
	let expires_in = data.get("expires_in").and_then(serde_json::Value::as_f64).unwrap_or(0.0);

	let mut extra = std::collections::BTreeMap::new();
	if let Some(scope) = data.get("scope").and_then(serde_json::Value::as_str) {
		extra.insert("scope".to_string(), serde_json::Value::String(scope.to_string()));
	}
	Ok(OAuthCredential {
		credentials: OAuthCredentials {
			access: access_token.to_string(),
			refresh: refresh_token.to_string(),
			expires: crate::utils::diagnostics::now_ms() + (expires_in * 1000.0) as i64 - TOKEN_EXPIRY_SKEW_MS,
			extra,
		},
	})
}

async fn login_with_browser(
	gateway: &str,
	authorization_endpoint: &str,
	interaction: &ProviderAuthInteraction,
) -> Result<OAuthCredential, ThrownValue> {
	let (verifier, challenge) = generate_pkce().await?;
	let state = random_uuid()?;
	let authorize_url = {
		let mut url = url::Url::parse(authorization_endpoint).map_err(|_| ThrownValue::message("invalid authorization endpoint"))?;
		{
			let mut query = url.query_pairs_mut();
			query.append_pair("response_type", "code");
			query.append_pair("client_id", OAUTH_CLIENT_ID);
			query.append_pair("redirect_uri", REDIRECT_URI);
			query.append_pair("scope", OAUTH_SCOPE);
			query.append_pair("code_challenge", &challenge);
			query.append_pair("code_challenge_method", "S256");
			query.append_pair("handoff", "url");
			query.append_pair("state", &state);
		}
		url.to_string()
	};

	let gateway_for_complete = gateway.to_string();
	let verifier_for_complete = verifier.clone();
	let signal_for_complete = interaction.signal.clone();
	let callback = start_oauth_callback_server(OAuthCallbackServerOptions {
		provider_name: "Radius".to_string(),
		host: CALLBACK_HOST.to_string(),
		port: CALLBACK_PORT,
		path: CALLBACK_PATH.to_string(),
		redirect_host: None,
		state: Some(state),
		complete: std::sync::Arc::new(move |code| {
			let gateway = gateway_for_complete.clone();
			let verifier = verifier_for_complete.clone();
			let signal = signal_for_complete.clone();
			Box::pin(async move {
				request_oauth_token(
					&gateway,
					vec![
						("grant_type", "authorization_code"),
						("client_id", OAUTH_CLIENT_ID),
						("redirect_uri", REDIRECT_URI),
						("code", code.as_str()),
						("code_verifier", verifier.as_str()),
					],
					&signal,
				)
				.await
			})
		}),
		signal: Some(interaction.signal.clone()),
		timeout_ms: None,
	})
	.await?;
	(interaction.notify)(crate::auth::types::AuthEvent::Progress {
		message: format!("Listening for OAuth callback on {REDIRECT_URI}"),
	});
	(interaction.notify)(crate::auth::types::AuthEvent::AuthUrl {
		url: authorize_url,
		instructions: Some("Continue in your browser.".to_string()),
	});

	let result = async {
		let credential = callback.wait().await?;
		match credential {
			Some(credential) => Ok(credential),
			None => Err(ThrownValue::message("OAuth callback did not complete.")),
		}
	}
	.await;

	callback.close();
	result
}

/// Random UUID (upstream `crypto.randomUUID()`).
fn random_uuid() -> Result<String, ThrownValue> {
	let mut bytes = [0u8; 16];
	getrandom::getrandom(&mut bytes).map_err(|_| ThrownValue::message("Failed to generate random bytes"))?;
	bytes[6] = (bytes[6] & 0x0f) | 0x40;
	bytes[8] = (bytes[8] & 0x3f) | 0x80;
	let hex = hex::encode(bytes);
	Ok(format!("{}-{}-{}-{}-{}", &hex[0..8], &hex[8..12], &hex[12..16], &hex[16..20], &hex[20..32]))
}

async fn request_device_authorization(gateway: &str, signal: &AbortSignal) -> Result<DeviceAuthorizationResponse, ThrownValue> {
	let url = url::Url::parse(&format!("{gateway}/v1/oauth/device")).map_err(|_| ThrownValue::message("invalid gateway URL"))?;
	let mut request = HttpRequest::new("POST", url.as_str());
	request.headers.push(("accept".to_string(), "application/json".to_string()));
	request
		.headers
		.push(("content-type".to_string(), "application/x-www-form-urlencoded".to_string()));
	request.body = HttpBody::bytes(
		crate::auth::oauth::form_urlencode(&[("client_id", OAUTH_CLIENT_ID), ("scope", OAUTH_SCOPE)]).into_bytes(),
	);
	let response = gateway_fetch(request, signal).await?;

	if !response.ok() {
		return Err(
			read_oauth_response_error(response, "Radius OAuth device authorization failed")
				.await
				.into_thrown_value(),
		);
	}

	let bytes = response.body.collect().await.map_err(|error| ThrownValue::message(error.message))?;
	let data: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| ThrownValue::message(error.to_string()))?;
	let device_code = data.get("device_code").and_then(serde_json::Value::as_str).unwrap_or_default();
	let user_code = data.get("user_code").and_then(serde_json::Value::as_str).unwrap_or_default();
	let verification_uri = data.get("verification_uri").and_then(serde_json::Value::as_str).unwrap_or_default();
	let expires_in = data.get("expires_in").and_then(serde_json::Value::as_f64);
	if device_code.is_empty() || user_code.is_empty() || verification_uri.is_empty() || expires_in.is_none() {
		return Err(ThrownValue::message(
			"Radius OAuth device authorization response is missing required fields",
		));
	}

	Ok(DeviceAuthorizationResponse {
		device_code: device_code.to_string(),
		user_code: user_code.to_string(),
		verification_uri: verification_uri.to_string(),
		expires_in: expires_in.unwrap(),
		interval: data.get("interval").and_then(serde_json::Value::as_f64),
	})
}

async fn login_with_device_code(gateway: &str, interaction: &ProviderAuthInteraction) -> Result<OAuthCredential, ThrownValue> {
	let device = request_device_authorization(gateway, &interaction.signal).await?;
	(interaction.notify)(crate::auth::types::AuthEvent::DeviceCode {
		user_code: device.user_code.clone(),
		verification_uri: device.verification_uri.clone(),
		interval_seconds: device.interval.map(|interval| interval as u64),
		expires_in_seconds: Some(device.expires_in as u64),
	});

	let gateway_for_poll = gateway.to_string();
	let device_code = device.device_code.clone();
	let signal = interaction.signal.clone();
	poll_oauth_device_code_flow(
		OAuthDeviceCodePollOptions {
			interval_seconds: device.interval,
			expires_in_seconds: Some(device.expires_in),
			wait_before_first_poll: false,
			signal: interaction.signal.clone(),
		},
		move || {
			let gateway = gateway_for_poll.clone();
			let device_code = device_code.clone();
			let signal = signal.clone();
			async move {
				match request_oauth_token(
					&gateway,
					vec![
						("grant_type", OAUTH_DEVICE_CODE_GRANT_TYPE),
						("client_id", OAUTH_CLIENT_ID),
						("device_code", device_code.as_str()),
					],
					&signal,
				)
				.await
				{
					Ok(credentials) => Ok(OAuthDeviceCodePollResult::Complete { value: credentials }),
					Err(error) => {
						// Only OAuthResponseError-style messages carry the
						// `oauth_error` code; recover the code from the message.
						let message = crate::utils::diagnostics::format_thrown_value(&error);
						if let Some(oauth_error) = oauth_error_from_message(&message) {
							match oauth_error.as_str() {
								"authorization_pending" => return Ok(OAuthDeviceCodePollResult::Pending),
								"slow_down" => return Ok(OAuthDeviceCodePollResult::SlowDown { interval_seconds: None }),
								"expired_token" => {
									return Ok(OAuthDeviceCodePollResult::Failed {
										message: "Device authorization expired.".to_string(),
									});
								}
								"access_denied" => {
									return Ok(OAuthDeviceCodePollResult::Failed {
										message: "Device authorization was denied.".to_string(),
									});
								}
								_ => return Err(error),
							}
						}
						Err(error)
					}
				}
			}
		},
	)
	.await
}

/// Extract the `oauth_error` code the upstream `OAuthResponseError` would have
/// carried ("Radius OAuth token request failed: <code>[: <description>]"). Kept
/// in one place because Rust models the error as a message.
fn oauth_error_from_message(message: &str) -> Option<String> {
	let rest = message.strip_prefix("Radius OAuth token request failed: ")?;
	let code = rest.split(": ").next()?;
	if code.is_empty() || code.chars().all(|c| c.is_ascii_digit()) {
		None
	} else {
		Some(code.to_string())
	}
}

/// Upstream `RadiusOAuthOptions` + `createRadiusOAuth`.
pub struct RadiusOAuthOptions {
	pub name: String,
	pub gateway: String,
}

pub fn create_radius_oauth(options: RadiusOAuthOptions) -> OAuthAuth {
	let gateway = normalize_gateway_url(&options.gateway);
	let name = options.name.clone();

	let login: OAuthLoginFn = {
		let gateway = gateway.clone();
		Box::new(move |interaction, _options| {
		let gateway = gateway.clone();
		let name = name.clone();
		Box::pin(async move {
			let login_method = (interaction.prompt)(crate::auth::types::AuthPrompt {
				signal: Some(interaction.signal.clone()),
				kind: crate::auth::types::AuthPromptKind::Select {
					message: format!("Sign in to {name}:"),
					options: vec![
						crate::auth::types::AuthSelectOption {
							id: LOGIN_METHOD_BROWSER.to_string(),
							label: "Sign in with browser (recommended)".to_string(),
							description: None,
						},
						crate::auth::types::AuthSelectOption {
							id: LOGIN_METHOD_DEVICE_CODE.to_string(),
							label: "Sign in with device code (when signing in from another device)".to_string(),
							description: None,
						},
					],
				},
			})
			.await?;

			if login_method == LOGIN_METHOD_DEVICE_CODE {
				return login_with_device_code(&gateway, &interaction).await;
			}
			if login_method == LOGIN_METHOD_BROWSER {
				let discovery = load_radius_oauth_discovery(&gateway, &interaction.signal).await?;
				return login_with_browser(&gateway, &discovery.authorization_endpoint, &interaction).await;
			}
			Err(ThrownValue::message(format!("Unknown {name} sign-in method: {login_method}")))
		})
	})
};

	let refresh: OAuthRefreshFn = {
		let gateway = gateway.clone();
		Box::new(move |credential, signal| {
		let gateway = gateway.clone();
		Box::pin(async move {
			request_oauth_token(
				&gateway,
				vec![
					("grant_type", "refresh_token"),
					("client_id", OAUTH_CLIENT_ID),
				("refresh_token", credential.credentials.refresh.as_str()),
				],
				&signal,
			)
			.await
		})
	})
};

	let to_auth: OAuthToAuthFn = Box::new(|credential| {
		Box::pin(async move {
			Ok(ModelAuth { api_key: Some(credential.credentials.access.clone()), headers: None, base_url: None })
		})
	});

	OAuthAuth {
		name: options.name,
		is_subscription: false,
		login_label: None,
		login,
		refresh,
		to_auth,
	}
}
