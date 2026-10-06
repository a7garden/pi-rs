//! Port of `packages/ai/src/auth/oauth/openai-chatgpt.ts` — Sign in with
//! ChatGPT for the OpenAI Responses API.
//!
//! This public-client flow uses no client secret and sends the resulting user
//! access token directly to api.openai.com.
//!
//! Every login registers a new user-owned client (`dynamic_agent_client`);
//! OpenAI returns the issued client id in the callback URL and it is stored
//! on the credential (`clientId`, `scopes` — round-tripped through
//! [`OAuthCredentials::extra`]).
//!
//! Unlike the other flows this server is not the shared callback-server: the
//! issued `client_id` rides on the callback and the error pages differ, so a
//! dedicated loopback server is assembled from the same request-reading
//! primitives (upstream keeps a bespoke `startCallbackServer` too). Binding
//! the fixed port (1455) failing with `EADDRINUSE` is mapped to a friendly
//! error, matching upstream.

use crate::auth::oauth::callback_server::{read_request_head, send_html_page};
use crate::auth::oauth::pkce::{base64url_encode, generate_pkce};
use crate::auth::types::{
	LoginOptions, ModelAuth, OAuthAuth, OAuthCredential, OAuthCredentials, OAuthLoginFn, OAuthRefreshFn,
	OAuthToAuthFn, ProviderAuthInteraction,
};
use crate::http::{HttpBody, HttpRequest};
use crate::utils::abort::{AbortController, AbortSignal};
use crate::utils::oauth_page::{oauth_error_html, oauth_success_html};
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::thrown::ThrownValue;

/// Every login registers a new client with this ID; OpenAI returns the issued
/// client ID in the callback.
const DYNAMIC_CLIENT_ID: &str = "dynamic_agent_client";
const AGENT_NAME_HINT: &str = "Pi";
const UUID_PATTERN: &str = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$";
const AUTHORIZE_URL: &str = "https://auth.openai.com/api/accounts/authorize";
const TOKEN_URL: &str = "https://auth.openai.com/api/accounts/oauth/token";
const RESOURCE: &str = "https://api.openai.com/v1";
const CALLBACK_PORT: u16 = 1455;
const CALLBACK_PATH: &str = "/auth/callback";
const REDIRECT_URI: &str = "http://127.0.0.1:1455/auth/callback";
const DIRECT_TOKEN_SCOPE: &str = "chatgpt.tokens.use.direct";
const SCOPE: &str = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";
/// Refresh this long before the real expiry so a request never starts with a
/// token about to expire.
const EXPIRY_MARGIN_MS: i64 = 3 * 60 * 1000;

fn callback_host() -> &'static str {
	static CALLBACK_HOST: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
		get_provider_env_value("PI_OAUTH_CALLBACK_HOST", None).unwrap_or_else(|| "127.0.0.1".to_string())
	});
	CALLBACK_HOST.as_str()
}

/// Upstream `AuthorizationResult`.
#[derive(Debug, Clone)]
struct AuthorizationResult {
	code: String,
	client_id: String,
}

struct TokenResponse {
	access_token: Option<String>,
	refresh_token: Option<String>,
	expires_in: Option<f64>,
	id_token: Option<String>,
	scope: Option<String>,
}

/// Upstream `randomValue` — 32 random bytes, base64url.
fn random_value() -> Result<String, ThrownValue> {
	let mut bytes = [0u8; 32];
	getrandom::getrandom(&mut bytes).map_err(|_| ThrownValue::message("Failed to generate random bytes"))?;
	Ok(base64url_encode(&bytes))
}

fn is_uuid(value: &str) -> bool {
	regex::Regex::new(UUID_PATTERN).expect("static uuid pattern").is_match(value)
}

fn query_pairs(url: &url::Url) -> std::collections::BTreeMap<String, String> {
	url.query_pairs().map(|(key, value)| (key.into_owned(), value.into_owned())).collect()
}

/// Upstream `authorizationResultFromCallback`.
fn authorization_result_from_callback(url: &url::Url, expected_state: &str) -> Result<AuthorizationResult, ThrownValue> {
	let params = query_pairs(url);
	let Some(code) = params.get("code").filter(|code| !code.is_empty()) else {
		return Err(ThrownValue::message("Missing authorization code"));
	};
	let Some(state) = params.get("state").filter(|state| !state.is_empty()) else {
		return Err(ThrownValue::message("Missing OAuth state"));
	};
	if state != expected_state {
		return Err(ThrownValue::message("OAuth state mismatch"));
	}
	let client_id = params.get("client_id").map(|id| id.trim().to_string()).unwrap_or_default();
	if client_id.is_empty() {
		return Err(ThrownValue::message(
			"OpenAI OAuth registration callback did not contain an issued client ID",
		));
	}
	Ok(AuthorizationResult { code: code.clone(), client_id })
}

/// Upstream `authorizationResultFromManualInput`.
fn authorization_result_from_manual_input(
	input: &str,
	expected_state: &str,
) -> Result<AuthorizationResult, ThrownValue> {
	let parsed = url::Url::parse(input.trim())
		.map_err(|_| ThrownValue::message("Paste the full callback URL from the browser"))?;
	let expected = url::Url::parse(REDIRECT_URI).expect("static redirect uri");
	let same_origin = parsed.origin() == expected.origin() && parsed.path() == expected.path();
	if !same_origin {
		return Err(ThrownValue::message(format!("The pasted callback URL must start with {REDIRECT_URI}")));
	}
	let params = query_pairs(&parsed);
	if let Some(error) = params.get("error").filter(|error| !error.is_empty()) {
		return Err(ThrownValue::message(format!("ChatGPT authorization failed: {error}")));
	}
	authorization_result_from_callback(&parsed, expected_state)
}

/// Upstream `startCallbackServer` — loopback server on the fixed port whose
/// result promise carries the parsed authorization result.
async fn start_callback_server(
	expected_state: String,
) -> Result<CallbackServer, ThrownValue> {
	let listener = match tokio::net::TcpListener::bind((callback_host(), CALLBACK_PORT)).await {
		Ok(listener) => std::sync::Arc::new(listener),
		Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => {
			return Err(ThrownValue::message(format!(
				"Port {CALLBACK_PORT} is in use, probably by an unfinished login in another pi session or by the Codex CLI. Cancel that login and try again."
			)));
		}
		Err(error) => return Err(ThrownValue::message(format!("listen {error}"))),
	};

	let (result_tx, result_rx) =
		tokio::sync::oneshot::channel::<Result<AuthorizationResult, ThrownValue>>();
	let result_tx = std::sync::Arc::new(std::sync::Mutex::new(Some(result_tx)));
	let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
	let stop_tx = std::sync::Arc::new(std::sync::Mutex::new(Some(stop_tx)));

	let accept_task = {
		let listener = listener.clone();
		let result_tx = result_tx.clone();
		let state_for_handler = expected_state.clone();
		tokio::spawn(async move {
			let mut stop_rx = stop_rx;
			loop {
				let accepted = tokio::select! {
					biased;
					_ = &mut stop_rx => break,
					accepted = listener.accept() => accepted,
				};
				let Ok((mut stream, _)) = accepted else { break };
				let state_for_connection = state_for_handler.clone();
				let result_tx = result_tx.clone();
				tokio::spawn(async move {
					let Some(head) = read_request_head(&mut stream).await else { return };
					let Some(request_line) = head.lines().next() else { return };
					let target = request_line.split_whitespace().nth(1).unwrap_or("/");
					let Ok(url) = url::Url::parse(&format!("http://localhost{target}")) else { return };
					if url.path() != CALLBACK_PATH {
						send_html_page(&mut stream, 404, &oauth_error_html("Callback route not found.", None)).await;
						return;
					}
					let params = query_pairs(&url);
					if let Some(error) = params.get("error").filter(|error| !error.is_empty()) {
						send_html_page(
							&mut stream,
							400,
							&oauth_error_html("ChatGPT was not connected.", Some(&format!("Error: {error}"))),
						)
						.await;
						if let Some(sender) = result_tx.lock().unwrap().take() {
							let _ = sender.send(Err(ThrownValue::message(format!(
								"ChatGPT authorization failed: {error}"
							))));
						}
						return;
					}
					match authorization_result_from_callback(&url, &state_for_connection) {
						Ok(result) => {
							send_html_page(
								&mut stream,
								200,
								&oauth_success_html(
									"ChatGPT authentication completed. You can close this window.",
								),
							)
							.await;
							if let Some(sender) = result_tx.lock().unwrap().take() {
								let _ = sender.send(Ok(result));
							}
						}
						Err(error) => {
							let message = crate::utils::diagnostics::format_thrown_value(&error);
							send_html_page(&mut stream, 400, &oauth_error_html(&message, None)).await;
						}
					}
				});
			}
		})
	};

	Ok(CallbackServer { accept_task, result_rx, stop_tx })
}

/// Upstream `CallbackServer` — `result` promise plus `server.close()`.
struct CallbackServer {
	accept_task: tokio::task::JoinHandle<()>,
	result_rx: tokio::sync::oneshot::Receiver<Result<AuthorizationResult, ThrownValue>>,
	/// Closing the server stops accepting (and drops the listener).
	stop_tx: std::sync::Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
}

impl CallbackServer {
	fn close(self) {
		// Dropping the listener stops accepting; in-flight handler tasks end
		// after their single response (deviation D27 — no spare connections).
		if let Some(stop) = self.stop_tx.lock().unwrap().take() {
			let _ = stop.send(());
		}
		self.accept_task.abort();
	}
}

async fn request_token(body: Vec<(&str, &str)>, signal: &AbortSignal) -> Result<TokenResponse, ThrownValue> {
	let mut request = HttpRequest::new("POST", TOKEN_URL);
	request.headers.push(("accept".to_string(), "application/json".to_string()));
	request
		.headers
		.push(("content-type".to_string(), "application/x-www-form-urlencoded".to_string()));
	request.body = HttpBody::bytes(crate::auth::oauth::form_urlencode(&body).into_bytes());
	request.signal = Some(signal.clone());

	let mut response = crate::auth::oauth::oauth_fetch()
		.send(request)
		.await
		.map_err(|error| ThrownValue::message(error.message.clone()))?;
	let status = response.status;
	let body = std::mem::replace(&mut response.body, HttpBody::Empty);
	let text = match body.collect().await {
		Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
		Err(error) => return Err(ThrownValue::message(error.message)),
	};
	if !(200..300).contains(&status) {
		return Err(ThrownValue::message(format!(
			"OpenAI OAuth token request failed ({status}): {}",
			if text.is_empty() { crate::auth::oauth::http_status_text(status).to_string() } else { text }
		)));
	}
	let data: serde_json::Value = serde_json::from_str(&text)
		.map_err(|_| ThrownValue::message("OpenAI OAuth token response must be an object"))?;
	if !data.is_object() {
		return Err(ThrownValue::message("OpenAI OAuth token response must be an object"));
	}
	Ok(TokenResponse {
		access_token: data.get("access_token").and_then(serde_json::Value::as_str).map(str::to_string),
		refresh_token: data.get("refresh_token").and_then(serde_json::Value::as_str).map(str::to_string),
		expires_in: data.get("expires_in").and_then(serde_json::Value::as_f64),
		id_token: data.get("id_token").and_then(serde_json::Value::as_str).map(str::to_string),
		scope: data.get("scope").and_then(serde_json::Value::as_str).map(str::to_string),
	})
}

fn require_token_string(value: Option<&str>, field: &str) -> Result<String, ThrownValue> {
	match value {
		Some(value) if !value.trim().is_empty() => Ok(value.to_string()),
		_ => Err(ThrownValue::message(format!("OpenAI OAuth token response has invalid {field}"))),
	}
}

fn credential_from_token_response(token: TokenResponse, client_id: String) -> Result<OAuthCredential, ThrownValue> {
	let access = require_token_string(token.access_token.as_deref(), "access_token")?;
	let refresh = require_token_string(token.refresh_token.as_deref(), "refresh_token")?;
	let scope = require_token_string(token.scope.as_deref(), "scope")?;
	let expires_in = match token.expires_in {
		Some(expires_in) if expires_in.is_finite() && expires_in > 0.0 => expires_in,
		_ => return Err(ThrownValue::message("OpenAI OAuth token response has invalid expires_in")),
	};
	let scopes: Vec<String> = scope
		.trim()
		.split_whitespace()
		.filter(|part| !part.is_empty())
		.map(str::to_string)
		.collect();
	if !scopes.iter().any(|scope| scope == DIRECT_TOKEN_SCOPE) {
		return Err(ThrownValue::message(format!(
			"OpenAI OAuth grant did not include {DIRECT_TOKEN_SCOPE}"
		)));
	}
	let mut extra = std::collections::BTreeMap::new();
	extra.insert("clientId".to_string(), serde_json::Value::String(client_id));
	extra.insert(
		"scopes".to_string(),
		serde_json::Value::Array(scopes.into_iter().map(serde_json::Value::String).collect()),
	);
	Ok(OAuthCredential {
		credentials: OAuthCredentials {
			refresh,
			access,
			expires: crate::utils::diagnostics::now_ms() + (expires_in * 1000.0) as i64 - EXPIRY_MARGIN_MS,
			extra,
		},
	})
}

async fn exchange_authorization_code(
	code: &str,
	verifier: &str,
	client_id: &str,
	signal: &AbortSignal,
) -> Result<OAuthCredential, ThrownValue> {
	let token = request_token(
		vec![
			("grant_type", "authorization_code"),
			("client_id", client_id),
			("code", code),
			("code_verifier", verifier),
			("redirect_uri", REDIRECT_URI),
			("resource", RESOURCE),
		],
		signal,
	)
	.await?;
	// Pi does not use the ID token to identify the user or read profile data.
	// Keep the presence check as part of the token-response contract.
	if token.id_token.as_deref().is_none_or(|id_token| id_token.trim().is_empty()) {
		return Err(ThrownValue::message("OpenAI OAuth token response did not contain an ID token"));
	}
	credential_from_token_response(token, client_id.to_string())
}

async fn refresh_access_token(credential: OAuthCredential, signal: &AbortSignal) -> Result<OAuthCredential, ThrownValue> {
	let client_id = credential
		.credentials
		.extra
		.get("clientId")
		.and_then(serde_json::Value::as_str)
		.unwrap_or_default()
		.to_string();
	if client_id.trim().is_empty() {
		return Err(ThrownValue::message(
			"Stored OpenAI OAuth credential does not contain an issued client ID; reconnect ChatGPT",
		));
	}
	let token = request_token(
		vec![
			("grant_type", "refresh_token"),
			("client_id", client_id.as_str()),
			("refresh_token", credential.credentials.refresh.as_str()),
			("resource", RESOURCE),
		],
		signal,
	)
	.await?;
	credential_from_token_response(token, client_id)
}

/// OpenAI identifies each installation ("agent host") by a stable URI such as
/// `urn:uuid:<uuid>`.
fn agent_host_id(device_id: Option<&str>) -> Result<String, ThrownValue> {
	match device_id.filter(|id| is_uuid(id)) {
		Some(device_id) => Ok(format!("urn:uuid:{}", device_id.to_lowercase())),
		None => Err(ThrownValue::message(
			"Sign in with ChatGPT requires a device ID (UUID) for this installation",
		)),
	}
}

async fn login_openai_chatgpt(
	interaction: &ProviderAuthInteraction,
	options: Option<LoginOptions>,
) -> Result<OAuthCredential, ThrownValue> {
	let device_id = options.as_ref().and_then(|options| options.get_device_id.as_ref()).map(|get| get());
	let host_id = agent_host_id(device_id.as_deref())?;
	let (verifier, challenge) = generate_pkce().await?;
	let state = random_value()?;
	let nonce = random_value()?;
	// Without this server, the browser's callback would reach whatever else
	// holds the port (another pending login or the Codex CLI), which rejects it
	// as a state mismatch. Fail with a clear error instead.
	let callback = start_callback_server(state.clone()).await?;

	let authorization_params = crate::auth::oauth::form_urlencode(&[
		("client_id", DYNAMIC_CLIENT_ID),
		("agent_name_hint", AGENT_NAME_HINT),
		("ext_agent_host_id", host_id.as_str()),
		("response_type", "code"),
		("redirect_uri", REDIRECT_URI),
		("resource", RESOURCE),
		("scope", SCOPE),
		("state", state.as_str()),
		("code_challenge", challenge.as_str()),
		("code_challenge_method", "S256"),
		("nonce", nonce.as_str()),
	]);
	(interaction.notify)(crate::auth::types::AuthEvent::AuthUrl {
		url: format!("{AUTHORIZE_URL}?{authorization_params}"),
		instructions: Some(
			"Complete sign-in in your browser. If the callback does not complete, paste the final redirect URL here."
				.to_string(),
		),
	});

	// Promise.race([callback.result, manualCode]) — the manual prompt runs
	// alongside the browser callback with its own abort signal.
	let manual_abort = AbortController::new();
	let manual_task = {
		let prompt = (interaction.prompt)(crate::auth::types::AuthPrompt {
			signal: Some(AbortSignal::any([manual_abort.signal(), interaction.signal.clone()])),
			kind: crate::auth::types::AuthPromptKind::ManualCode {
				message: "Complete login in your browser, or paste the final redirect URL here:".to_string(),
				placeholder: Some(REDIRECT_URI.to_string()),
			},
		});
		let state_for_manual = state.clone();
		tokio::spawn(async move {
			let input = prompt.await?;
			authorization_result_from_manual_input(&input, &state_for_manual)
		})
	};

	let CallbackServer { accept_task, result_rx, stop_tx } = callback;
	let mut result_rx = result_rx;
	let result = async {
		let raced = tokio::select! {
			biased;
			race = manual_task => match race {
				Ok(result) => result,
				Err(join_error) => Err(ThrownValue::message(join_error.to_string())),
			},
			result = &mut result_rx => match result {
				Ok(result) => result,
				Err(_) => Err(ThrownValue::message("ChatGPT login was cancelled")),
			},
		};
		let result = match raced {
			Ok(result) => result,
			Err(error) => {
				// catch (error) { if (interaction.signal.aborted) throw new
				// Error("Login cancelled"); throw error; }
				if interaction.signal.aborted() {
					return Err(ThrownValue::message("Login cancelled"));
				}
				return Err(error);
			}
		};
		(interaction.notify)(crate::auth::types::AuthEvent::Progress {
			message: "Exchanging authorization code for tokens...".to_string(),
		});
		match exchange_authorization_code(&result.code, &verifier, &result.client_id, &interaction.signal).await {
			Ok(credential) => Ok(credential),
			Err(error) => {
				if interaction.signal.aborted() {
					Err(ThrownValue::message("Login cancelled"))
				} else {
					Err(error)
				}
			}
		}
	}
	.await;

	// finally { manualAbort.abort(); callback.server.close(); }
	manual_abort.abort();
	CallbackServer { accept_task, result_rx, stop_tx }.close();
	result
}

/// Upstream `export const openaiChatGPTOAuth`.
pub fn openai_chatgpt_oauth() -> OAuthAuth {
	let login: OAuthLoginFn = Box::new(|interaction, options| {
		Box::pin(async move { login_openai_chatgpt(&interaction, options).await })
	});

	let refresh: OAuthRefreshFn = Box::new(|credential, signal| Box::pin(async move { refresh_access_token(credential, &signal).await }));

	let to_auth: OAuthToAuthFn = Box::new(|credential| {
		Box::pin(async move {
			Ok(ModelAuth { api_key: Some(credential.credentials.access.clone()), headers: None, base_url: None })
		})
	});

	OAuthAuth {
		name: "OpenAI (ChatGPT subscription)".to_string(),
		is_subscription: true,
		login_label: Some("Sign in with ChatGPT".to_string()),
		login,
		refresh,
		to_auth,
	}
}
