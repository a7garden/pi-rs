//! Port of `packages/ai/src/auth/oauth/openai-codex.ts` — the OpenAI Codex
//! (ChatGPT OAuth) flow.
//!
//! The browser flow's fixed callback port (1455) is shared with the Codex
//! CLI; when the bind fails the server start fails and login falls back to
//! the pasted redirect URL (upstream `.catch(() => undefined)`).
//! `createState` uses `getrandom` (upstream `node:crypto.randomBytes`); the
//! lazy Node-crypto import dance is Rust-static (deviation D31).

use crate::auth::oauth::callback_server::{
    boxed_complete, start_oauth_callback_server, wait_for_callback_or_manual_input,
    CallbackOrManual, OAuthCallbackServerOptions,
};
use crate::auth::oauth::device_code::{
    poll_oauth_device_code_flow, OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult,
};
use crate::auth::oauth::pkce::generate_pkce;
use crate::auth::types::{
    LoginOptions, ModelAuth, OAuthAuth, OAuthCredential, OAuthCredentials, OAuthLoginFn,
    OAuthRefreshFn, OAuthToAuthFn, ProviderAuthInteraction,
};
use crate::http::{HttpBody, HttpRequest};
use crate::utils::abort::AbortSignal;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::thrown::ThrownValue;
use base64::Engine as _;

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const AUTH_BASE_URL: &str = "https://auth.openai.com";
const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const DEVICE_USER_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
const DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
const DEVICE_VERIFICATION_URI: &str = "https://auth.openai.com/codex/device";
const DEVICE_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
const DEVICE_CODE_TIMEOUT_SECONDS: f64 = 15.0 * 60.0;
const OPENAI_CODEX_BROWSER_LOGIN_METHOD: &str = "browser";
const OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD: &str = "device_code";
const SCOPE: &str = "openid profile email offline_access";
/// `https://api.openai.com/auth` — the JWT claim path carrying the account id.
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";

fn callback_host() -> String {
    get_provider_env_value("PI_OAUTH_CALLBACK_HOST", None)
        .unwrap_or_else(|| "127.0.0.1".to_string())
}

struct DeviceAuthInfo {
    device_auth_id: String,
    user_code: String,
    interval_seconds: f64,
}

struct DeviceTokenSuccess {
    authorization_code: String,
    code_verifier: String,
}

struct OAuthToken {
    access: String,
    refresh: String,
    expires: i64,
}

fn status_ok(status: u16) -> bool {
    (200..300).contains(&status)
}

/// Upstream `createState` — 16 random bytes as hex.
fn create_state() -> Result<String, ThrownValue> {
    let mut bytes = [0u8; 16];
    getrandom::getrandom(&mut bytes)
        .map_err(|_| ThrownValue::message("Failed to generate random bytes"))?;
    Ok(hex::encode(bytes))
}

/// Upstream `parseAuthorizationInput`.
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
        let params: std::collections::BTreeMap<String, String> =
            url::form_urlencoded::parse(value.as_bytes())
                .map(|(key, item)| (key.into_owned(), item.into_owned()))
                .collect();
        return (params.get("code").cloned(), params.get("state").cloned());
    }

    (Some(value.to_string()), None)
}

/// Upstream `decodeJwt` — base64 (standard alphabet, unpadded input) payload
/// parse; signature is not verified, like upstream.
fn decode_jwt(token: &str) -> Option<serde_json::Value> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let payload = parts.get(1)?;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .ok()?;
    serde_json::from_slice(&decoded).ok()
}

/// Upstream `fetchWithLoginCancellation` — fetch failures become
/// `"Login cancelled"` when the signal aborted.
async fn fetch_with_login_cancellation(
    request: HttpRequest,
    signal: &AbortSignal,
) -> Result<crate::http::HttpResponse, ThrownValue> {
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

async fn response_text(response: &mut crate::http::HttpResponse) -> String {
    let body = std::mem::replace(&mut response.body, crate::http::HttpBody::Empty);
    match body.collect().await {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(_) => String::new(),
    }
}

/// Node `Response.statusText` equivalents for the statuses these flows hit.
pub(crate) fn http_status_text(status: u16) -> &'static str {
    match status {
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        408 => "Request Timeout",
        409 => "Conflict",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "",
    }
}

async fn read_token_response(
    response: crate::http::HttpResponse,
    operation: &str,
) -> Result<OAuthToken, ThrownValue> {
    let status = response.status;
    if !status_ok(status) {
        let mut response = response;
        let text = response_text(&mut response).await;
        return Err(ThrownValue::message(format!(
            "OpenAI Codex token {operation} failed ({status}): {}",
            if text.is_empty() {
                http_status_text(status).to_string()
            } else {
                text
            }
        )));
    }

    let mut response = response;
    let text = response_text(&mut response).await;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|_| {
        ThrownValue::message(format!(
            "OpenAI Codex token {operation} response missing fields: {text}"
        ))
    })?;
    let access_token = json
        .get("access_token")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let refresh_token = json
        .get("refresh_token")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let expires_in = json.get("expires_in").and_then(serde_json::Value::as_f64);
    if access_token.is_empty() || refresh_token.is_empty() || expires_in.is_none() {
        return Err(ThrownValue::message(format!(
            "OpenAI Codex token {operation} response missing fields: {text}"
        )));
    }

    Ok(OAuthToken {
        access: access_token.to_string(),
        refresh: refresh_token.to_string(),
        expires: crate::utils::diagnostics::now_ms() + (expires_in.unwrap() * 1000.0) as i64,
    })
}

async fn exchange_authorization_code(
    code: &str,
    verifier: &str,
    redirect_uri: &str,
    signal: &AbortSignal,
) -> Result<OAuthToken, ThrownValue> {
    let mut request = HttpRequest::new("POST", TOKEN_URL);
    request.headers.push((
        "Content-Type".to_string(),
        "application/x-www-form-urlencoded".to_string(),
    ));
    request.body = HttpBody::bytes(
        crate::auth::oauth::form_urlencode(&[
            ("grant_type", "authorization_code"),
            ("client_id", CLIENT_ID),
            ("code", code),
            ("code_verifier", verifier),
            ("redirect_uri", redirect_uri),
        ])
        .into_bytes(),
    );
    request.signal = Some(signal.clone());

    let response = fetch_with_login_cancellation(request, signal).await?;
    read_token_response(response, "exchange").await
}

async fn refresh_access_token(
    refresh_token: &str,
    signal: &AbortSignal,
) -> Result<OAuthToken, ThrownValue> {
    let mut request = HttpRequest::new("POST", TOKEN_URL);
    request.headers.push((
        "Content-Type".to_string(),
        "application/x-www-form-urlencoded".to_string(),
    ));
    request.body = HttpBody::bytes(
        crate::auth::oauth::form_urlencode(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", CLIENT_ID),
        ])
        .into_bytes(),
    );
    request.signal = Some(signal.clone());

    let response = match crate::auth::oauth::oauth_fetch().send(request).await {
        Ok(response) => response,
        Err(error) => {
            return Err(ThrownValue::message(format!(
                "OpenAI Codex token refresh error: {}",
                error.message
            )));
        }
    };

    read_token_response(response, "refresh").await
}

async fn start_openai_codex_device_auth(
    signal: &AbortSignal,
) -> Result<DeviceAuthInfo, ThrownValue> {
    let mut request = HttpRequest::new("POST", DEVICE_USER_CODE_URL);
    request
        .headers
        .push(("Content-Type".to_string(), "application/json".to_string()));
    request.body = HttpBody::bytes(
        serde_json::json!({ "client_id": CLIENT_ID })
            .to_string()
            .into_bytes(),
    );
    request.signal = Some(signal.clone());

    let mut response = fetch_with_login_cancellation(request, signal).await?;
    let status = response.status;
    if !status_ok(status) {
        if status == 404 {
            return Err(ThrownValue::message(
				"OpenAI Codex device code login is not enabled for this server. Use browser login or verify the server URL.",
			));
        }
        let response_body = response_text(&mut response).await;
        return Err(ThrownValue::message(format!(
            "OpenAI Codex device code request failed with status {status}{}",
            if response_body.is_empty() {
                String::new()
            } else {
                format!(": {response_body}")
            }
        )));
    }

    let text = response_text(&mut response).await;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|_| {
        ThrownValue::message(format!("Invalid OpenAI Codex device code response: {text}"))
    })?;
    let device_auth_id = json
        .get("device_auth_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let user_code = json
        .get("user_code")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let interval = json.get("interval");
    let interval_seconds = match interval {
        Some(serde_json::Value::String(text)) => text.trim().parse::<f64>().ok(),
        Some(value) => value.as_f64(),
        None => None,
    };
    if device_auth_id.is_empty()
        || user_code.is_empty()
        || interval_seconds.is_none()
        || !interval_seconds.unwrap().is_finite()
        || interval_seconds.unwrap() < 0.0
    {
        return Err(ThrownValue::message(format!(
            "Invalid OpenAI Codex device code response: {text}"
        )));
    }

    Ok(DeviceAuthInfo {
        device_auth_id: device_auth_id.to_string(),
        user_code: user_code.to_string(),
        interval_seconds: interval_seconds.unwrap(),
    })
}

async fn poll_openai_codex_device_auth(
    device: &DeviceAuthInfo,
    signal: &AbortSignal,
) -> Result<DeviceTokenSuccess, ThrownValue> {
    let device_auth_id = device.device_auth_id.clone();
    let user_code = device.user_code.clone();
    poll_oauth_device_code_flow(
        OAuthDeviceCodePollOptions {
            interval_seconds: Some(device.interval_seconds),
            expires_in_seconds: Some(DEVICE_CODE_TIMEOUT_SECONDS),
            wait_before_first_poll: false,
            signal: signal.clone(),
        },
        move || {
            let device_auth_id = device_auth_id.clone();
            let user_code = user_code.clone();
            let signal = signal.clone();
            async move {
                let mut request = HttpRequest::new("POST", DEVICE_TOKEN_URL);
                request
                    .headers
                    .push(("Content-Type".to_string(), "application/json".to_string()));
                request.body = HttpBody::bytes(
                    serde_json::json!({
                        "device_auth_id": device_auth_id,
                        "user_code": user_code,
                    })
                    .to_string()
                    .into_bytes(),
                );
                request.signal = Some(signal.clone());

                let mut response = fetch_with_login_cancellation(request, &signal).await?;
                let ok = status_ok(response.status);
                if ok {
                    let text = response_text(&mut response).await;
                    let json: serde_json::Value = serde_json::from_str(&text).map_err(|_| {
                        ThrownValue::message(format!(
                            "Invalid OpenAI Codex device auth token response: {text}"
                        ))
                    })?;
                    let authorization_code = json
                        .get("authorization_code")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default();
                    let code_verifier = json
                        .get("code_verifier")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default();
                    if authorization_code.is_empty() || code_verifier.is_empty() {
                        return Ok(OAuthDeviceCodePollResult::<DeviceTokenSuccess>::Failed {
                            message: format!(
                                "Invalid OpenAI Codex device auth token response: {text}"
                            ),
                        });
                    }
                    return Ok(OAuthDeviceCodePollResult::Complete {
                        value: DeviceTokenSuccess {
                            authorization_code: authorization_code.to_string(),
                            code_verifier: code_verifier.to_string(),
                        },
                    });
                }

                if response.status == 403 || response.status == 404 {
                    return Ok(OAuthDeviceCodePollResult::Pending);
                }

                let response_body = response_text(&mut response).await;
                let error_code = serde_json::from_str::<serde_json::Value>(&response_body)
                    .ok()
                    .and_then(|json| json.get("error").cloned())
                    .map(|error| match error {
                        serde_json::Value::Object(object) => object
                            .get("code")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null),
                        other => other,
                    })
                    .and_then(|code| code.as_str().map(str::to_string));

                if error_code.as_deref() == Some("deviceauth_authorization_pending") {
                    return Ok(OAuthDeviceCodePollResult::Pending);
                }
                if error_code.as_deref() == Some("slow_down") {
                    return Ok(OAuthDeviceCodePollResult::SlowDown {
                        interval_seconds: None,
                    });
                }

                Ok(OAuthDeviceCodePollResult::Failed {
                    message: format!(
                        "OpenAI Codex device auth failed with status {}{}",
                        response.status,
                        if response_body.is_empty() {
                            String::new()
                        } else {
                            format!(": {response_body}")
                        }
                    ),
                })
            }
        },
    )
    .await
}

/// Upstream `createAuthorizationFlow` — PKCE + state + authorize URL.
async fn create_authorization_flow(
    agent_name: Option<String>,
) -> Result<(String, String, String), ThrownValue> {
    let (verifier, challenge) = generate_pkce().await?;
    let state = create_state()?;

    let mut url = url::Url::parse(AUTHORIZE_URL)
        .map_err(|_| ThrownValue::message("invalid authorize URL"))?;
    {
        let mut query = url.query_pairs_mut();
        query.append_pair("response_type", "code");
        query.append_pair("client_id", CLIENT_ID);
        query.append_pair("redirect_uri", REDIRECT_URI);
        query.append_pair("scope", SCOPE);
        query.append_pair("code_challenge", &challenge);
        query.append_pair("code_challenge_method", "S256");
        query.append_pair("state", &state);
        query.append_pair("id_token_add_organizations", "true");
        query.append_pair("codex_cli_simplified_flow", "true");
        query.append_pair("originator", agent_name.as_deref().unwrap_or("pi"));
    }

    Ok((verifier, state, url.to_string()))
}

/// Upstream `getAccountId` — `chatgpt_account_id` from the access token JWT.
fn get_account_id(access_token: &str) -> Option<String> {
    let payload = decode_jwt(access_token)?;
    let auth = payload.get(JWT_CLAIM_PATH)?;
    let account_id = auth.get("chatgpt_account_id")?.as_str()?;
    if account_id.is_empty() {
        None
    } else {
        Some(account_id.to_string())
    }
}

/// Upstream `credentialsFromToken`.
fn credentials_from_token(token: OAuthToken) -> Result<OAuthCredential, ThrownValue> {
    let account_id = get_account_id(&token.access)
        .ok_or_else(|| ThrownValue::message("Failed to extract accountId from token"))?;

    let mut extra = std::collections::BTreeMap::new();
    extra.insert(
        "accountId".to_string(),
        serde_json::Value::String(account_id),
    );
    Ok(OAuthCredential {
        credentials: OAuthCredentials {
            refresh: token.refresh,
            access: token.access,
            expires: token.expires,
            extra,
        },
    })
}

async fn exchange_authorization_code_for_credentials(
    code: &str,
    verifier: &str,
    redirect_uri: &str,
    signal: &AbortSignal,
) -> Result<OAuthCredential, ThrownValue> {
    credentials_from_token(exchange_authorization_code(code, verifier, redirect_uri, signal).await?)
}

async fn login_openai_codex_device_code(
    interaction: &ProviderAuthInteraction,
) -> Result<OAuthCredential, ThrownValue> {
    let device = start_openai_codex_device_auth(&interaction.signal).await?;
    (interaction.notify)(crate::auth::types::AuthEvent::DeviceCode {
        user_code: device.user_code.clone(),
        verification_uri: DEVICE_VERIFICATION_URI.to_string(),
        interval_seconds: Some(device.interval_seconds as u64),
        expires_in_seconds: Some(DEVICE_CODE_TIMEOUT_SECONDS as u64),
    });
    let code = poll_openai_codex_device_auth(&device, &interaction.signal).await?;
    exchange_authorization_code_for_credentials(
        &code.authorization_code,
        &code.code_verifier,
        DEVICE_REDIRECT_URI,
        &interaction.signal,
    )
    .await
}

async fn login_openai_codex(
    interaction: &ProviderAuthInteraction,
    options: Option<LoginOptions>,
) -> Result<OAuthCredential, ThrownValue> {
    let (verifier, state, url) =
        create_authorization_flow(options.and_then(|options| options.agent_name)).await?;
    // Port 1455 is shared with the Codex CLI; when it is taken, fall back to the
    // pasted redirect URL.
    let callback = start_oauth_callback_server(OAuthCallbackServerOptions {
        provider_name: "OpenAI".to_string(),
        host: callback_host(),
        port: 1455,
        path: "/auth/callback".to_string(),
        redirect_host: None,
        state: Some(state.clone()),
        complete: boxed_complete(|code| async move { Ok::<String, ThrownValue>(code) }),
        signal: Some(interaction.signal.clone()),
        timeout_ms: None,
    })
    .await
    .ok();

    (interaction.notify)(crate::auth::types::AuthEvent::AuthUrl {
        url,
        instructions: Some("A browser window should open. Complete login to finish.".to_string()),
    });

    let result = async {
        let result = wait_for_callback_or_manual_input(
			interaction,
			callback.as_ref(),
			(
				"Complete login in your browser, or paste the authorization code / redirect URL here:",
				REDIRECT_URI,
			),
		)
		.await?;
        let code = match result {
            CallbackOrManual::Callback(value) => Some(value),
            CallbackOrManual::Manual(input) => {
                let (parsed_code, parsed_state) = parse_authorization_input(&input);
                if parsed_state
                    .as_deref()
                    .is_some_and(|candidate| candidate != state)
                {
                    return Err(ThrownValue::message("State mismatch"));
                }
                parsed_code
            }
        };

        let Some(code) = code else {
            return Err(ThrownValue::message("Missing authorization code"));
        };
        exchange_authorization_code_for_credentials(
            &code,
            &verifier,
            REDIRECT_URI,
            &interaction.signal,
        )
        .await
    }
    .await;

    if let Some(callback) = callback {
        callback.close();
    }
    result
}

/// Refresh OpenAI Codex OAuth token.
async fn refresh_openai_codex_token(
    refresh_token: &str,
    signal: &AbortSignal,
) -> Result<OAuthCredential, ThrownValue> {
    credentials_from_token(refresh_access_token(refresh_token, signal).await?)
}

/// Upstream `export const openaiCodexOAuth`.
pub fn openai_codex_oauth() -> OAuthAuth {
    let login: OAuthLoginFn = Box::new(|interaction, options| {
        Box::pin(async move {
            let method = (interaction.prompt)(crate::auth::types::AuthPrompt {
                signal: Some(interaction.signal.clone()),
                kind: crate::auth::types::AuthPromptKind::Select {
                    message: "Select OpenAI Codex login method:".to_string(),
                    options: vec![
                        crate::auth::types::AuthSelectOption {
                            id: OPENAI_CODEX_BROWSER_LOGIN_METHOD.to_string(),
                            label: "Browser login (default)".to_string(),
                            description: None,
                        },
                        crate::auth::types::AuthSelectOption {
                            id: OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD.to_string(),
                            label: "Device code login (headless)".to_string(),
                            description: None,
                        },
                    ],
                },
            })
            .await?;

            if method == OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD {
                return login_openai_codex_device_code(&interaction).await;
            }
            if method != OPENAI_CODEX_BROWSER_LOGIN_METHOD {
                return Err(ThrownValue::message(format!(
                    "Unknown OpenAI Codex login method: {method}"
                )));
            }

            login_openai_codex(&interaction, options).await
        })
    });

    let refresh: OAuthRefreshFn = Box::new(|credential, signal| {
        Box::pin(async move {
            refresh_openai_codex_token(&credential.credentials.refresh, &signal).await
        })
    });

    let to_auth: OAuthToAuthFn = Box::new(|credential| {
        Box::pin(async move {
            Ok(ModelAuth {
                api_key: Some(credential.credentials.access.clone()),
                headers: None,
                base_url: None,
            })
        })
    });

    OAuthAuth {
        name: "OpenAI (ChatGPT Plus/Pro)".to_string(),
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

    // Port of test/openai-codex-oauth.test.ts "uses the app's agent name as the
    // browser login originator" (v1.1.0, #10433): the propagated LoginOptions
    // name replaces pi's own in the authorize URL.
    #[tokio::test]
    async fn uses_the_app_agent_name_as_the_browser_login_originator() {
        let (_, _, url) = create_authorization_flow(Some("my-app".to_string()))
            .await
            .expect("flow");
        let query: Vec<(String, String)> = url::Url::parse(&url)
            .expect("url")
            .query_pairs()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        assert!(
            query.contains(&("originator".to_string(), "my-app".to_string())),
            "originator not overridden: {query:?}"
        );

        let (_, _, default_url) = create_authorization_flow(None).await.expect("flow");
        let default_query: Vec<(String, String)> = url::Url::parse(&default_url)
            .expect("url")
            .query_pairs()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        assert!(
            default_query.contains(&("originator".to_string(), "pi".to_string())),
            "default originator missing: {default_query:?}"
        );
    }
    use crate::http::{FetchFn, HttpResponse, ProviderHttpError};
    use crate::utils::abort::AbortController;
    use base64::Engine as _;
    use parking_lot::Mutex;
    use std::sync::Arc;

    type Shared<T> = Arc<Mutex<T>>;

    fn shared<T>(value: T) -> Shared<T> {
        Arc::new(Mutex::new(value))
    }

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

    fn stub_fetch(
        routes: impl Fn(
                HttpRequest,
            )
                -> futures_util::future::BoxFuture<'static, Result<HttpResponse, ProviderHttpError>>
            + Send
            + Sync
            + 'static,
    ) -> FetchFn {
        FetchFn::new(move |request| routes(request))
    }

    async fn body_json(body: HttpBody) -> serde_json::Value {
        let bytes = body.collect().await.expect("body");
        serde_json::from_slice(&bytes).expect("json body")
    }

    fn interaction(
        signal: AbortSignal,
        prompt: impl Fn(crate::auth::types::AuthPrompt) -> Result<String, ThrownValue>
            + Send
            + Sync
            + 'static,
        notify: impl Fn(AuthEvent) + Send + Sync + 'static,
    ) -> ProviderAuthInteraction {
        ProviderAuthInteraction {
            signal,
            prompt: Box::new(move |auth_prompt| {
                let result = prompt(auth_prompt);
                Box::pin(async move { result })
            }),
            notify: Box::new(notify),
        }
    }

    /// Access token carrying `chatgpt_account_id` (upstream `createAccessToken`).
    fn create_access_token(account_id: &str) -> String {
        let header = base64::engine::general_purpose::STANDARD
            .encode(serde_json::json!({ "alg": "none" }).to_string());
        let payload = base64::engine::general_purpose::STANDARD.encode(
            serde_json::json!({ JWT_CLAIM_PATH: { "chatgpt_account_id": account_id } }).to_string(),
        );
        format!("{header}.{payload}.signature")
    }

    fn device_auth_pending_response() -> HttpResponse {
        json_response(
            serde_json::json!({
                "error": {
                    "message": "Device authorization is pending. Please try again.",
                    "type": "invalid_request_error",
                    "param": null,
                    "code": "deviceauth_authorization_pending",
                }
            }),
            403,
        )
    }

    async fn login_device_code_for_test(
        signal: AbortSignal,
        notify: impl Fn(AuthEvent) + Send + Sync + 'static,
    ) -> Result<OAuthCredential, ThrownValue> {
        openai_codex_oauth()
            .login(
                interaction(
                    signal,
                    |prompt| match &prompt.kind {
                        AuthPromptKind::Select { .. } => Ok("device_code".to_string()),
                        _ => panic!("Unexpected prompt"),
                    },
                    notify,
                ),
                None,
            )
            .await
    }

    // Port of test/openai-codex-oauth.test.ts "logs in with the OpenAI Codex
    // device code flow".
    #[tokio::test(start_paused = true)]
    async fn logs_in_with_the_device_code_flow() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        let access_token = create_access_token("account-123");
        let poll_times: Shared<Vec<i64>> = shared(Vec::new());
        let poll_responses: Shared<Vec<HttpResponse>> = shared(vec![
            device_auth_pending_response(),
            json_response(
                serde_json::json!({
                    "authorization_code": "oauth-code",
                    "code_challenge": "device-code-challenge",
                    "code_verifier": "device-code-verifier",
                }),
                200,
            ),
        ]);

        let poll_times_for_fetch = poll_times.clone();
        let poll_responses_for_fetch = poll_responses.clone();
        let access_token_for_fetch = access_token.clone();
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |mut request| {
            let url = request.url.clone();
            let poll_times_for_fetch = poll_times_for_fetch.clone();
            let poll_responses_for_fetch = poll_responses_for_fetch.clone();
            let access_token_for_fetch = access_token_for_fetch.clone();
            Box::pin(async move {
                if url == DEVICE_USER_CODE_URL {
                    assert_eq!(request.method, "POST");
                    assert!(
                        request
                            .headers
                            .iter()
                            .any(|(name, value)| name == "Content-Type"
                                && value == "application/json")
                    );
                    let body =
                        body_json(std::mem::replace(&mut request.body, HttpBody::Empty)).await;
                    assert_eq!(body, serde_json::json!({ "client_id": CLIENT_ID }));
                    return Ok(json_response(
                        serde_json::json!({
                            "device_auth_id": "device-auth-id",
                            "user_code": "ABCD-1234",
                            "interval": "5",
                        }),
                        200,
                    ));
                }
                if url == DEVICE_TOKEN_URL {
                    poll_times_for_fetch
                        .lock()
                        .push(crate::utils::diagnostics::now_ms());
                    assert_eq!(request.method, "POST");
                    let body =
                        body_json(std::mem::replace(&mut request.body, HttpBody::Empty)).await;
                    assert_eq!(
                        body,
                        serde_json::json!({ "device_auth_id": "device-auth-id", "user_code": "ABCD-1234" })
                    );
                    return Ok(poll_responses_for_fetch.lock().remove(0));
                }
                if url == TOKEN_URL {
                    assert_eq!(request.method, "POST");
                    let body = std::mem::replace(&mut request.body, HttpBody::Empty)
                        .collect()
                        .await
                        .expect("body");
                    let params =
                        crate::auth::oauth::parse_form(&String::from_utf8(body).expect("utf8"));
                    assert_eq!(
                        params.get("grant_type").map(String::as_str),
                        Some("authorization_code")
                    );
                    assert_eq!(params.get("client_id").map(String::as_str), Some(CLIENT_ID));
                    assert_eq!(params.get("code").map(String::as_str), Some("oauth-code"));
                    assert_eq!(
                        params.get("redirect_uri").map(String::as_str),
                        Some("https://auth.openai.com/deviceauth/callback")
                    );
                    assert_eq!(
                        params.get("code_verifier").map(String::as_str),
                        Some("device-code-verifier")
                    );
                    return Ok(json_response(
                        serde_json::json!({
                            "access_token": access_token_for_fetch,
                            "refresh_token": "refresh-token",
                            "expires_in": 3600,
                        }),
                        200,
                    ));
                }
                Err(ProviderHttpError::new(format!(
                    "Unexpected fetch URL: {url}"
                )))
            })
        })));

        let device_infos: Shared<Vec<(String, String, Option<u64>, Option<u64>)>> =
            shared(Vec::new());
        let device_infos_for_notify = device_infos.clone();
        let credential = tokio::spawn(login_device_code_for_test(
            never_aborted_signal(),
            move |event| {
                if let AuthEvent::DeviceCode {
                    user_code,
                    verification_uri,
                    interval_seconds,
                    expires_in_seconds,
                } = event
                {
                    device_infos_for_notify.lock().push((
                        user_code,
                        verification_uri,
                        interval_seconds,
                        expires_in_seconds,
                    ));
                }
            },
        ));

        // First poll happens immediately (interval "5" from the string field).
        tokio::time::advance(std::time::Duration::from_millis(0)).await;
        {
            let infos = device_infos.lock();
            assert_eq!(infos.len(), 1);
            assert_eq!(infos[0].0, "ABCD-1234");
            assert_eq!(infos[0].1, "https://auth.openai.com/codex/device");
            assert_eq!(infos[0].2, Some(5));
            assert_eq!(infos[0].3, Some(900));
        }
        assert_eq!(poll_times.lock().len(), 1);

        tokio::time::advance(std::time::Duration::from_millis(4999)).await;
        assert_eq!(poll_times.lock().len(), 1);

        tokio::time::advance(std::time::Duration::from_millis(1)).await;
        let credential = credential.await.expect("task").expect("credentials");
        assert_eq!(credential.credentials.access, access_token);
        assert_eq!(credential.credentials.refresh, "refresh-token");
        assert_eq!(
            credential
                .credentials
                .extra
                .get("accountId")
                .and_then(serde_json::Value::as_str),
            Some("account-123")
        );
        assert_eq!(poll_times.lock().len(), 2);
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "offers browser login first and uses the selected OpenAI Codex
    // device code flow".
    #[tokio::test]
    async fn offers_browser_login_first_and_uses_the_selected_device_code_flow() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        let access_token = create_access_token("account-456");
        let select_prompts: Shared<Vec<crate::auth::types::AuthPrompt>> = shared(Vec::new());
        let device_infos: Shared<Vec<(String, String, Option<u64>, Option<u64>)>> =
            shared(Vec::new());

        let access_token_for_fetch = access_token.clone();
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |mut request| {
            let url = request.url.clone();
            let access_token_for_fetch = access_token_for_fetch.clone();
            Box::pin(async move {
                if url == DEVICE_USER_CODE_URL {
                    let body =
                        body_json(std::mem::replace(&mut request.body, HttpBody::Empty)).await;
                    assert_eq!(body, serde_json::json!({ "client_id": CLIENT_ID }));
                    return Ok(json_response(
                        serde_json::json!({
                            "device_auth_id": "device-auth-id",
                            "user_code": "WXYZ-7890",
                            "interval": "5",
                        }),
                        200,
                    ));
                }
                if url == DEVICE_TOKEN_URL {
                    return Ok(json_response(
                        serde_json::json!({
                            "authorization_code": "oauth-code",
                            "code_challenge": "device-code-challenge",
                            "code_verifier": "device-code-verifier",
                        }),
                        200,
                    ));
                }
                if url == TOKEN_URL {
                    return Ok(json_response(
                        serde_json::json!({
                            "access_token": access_token_for_fetch,
                            "refresh_token": "refresh-token",
                            "expires_in": 3600,
                        }),
                        200,
                    ));
                }
                Err(ProviderHttpError::new(format!(
                    "Unexpected fetch URL: {url}"
                )))
            })
        })));

        let select_for_prompt = select_prompts.clone();
        let device_infos_for_notify = device_infos.clone();
        let credential = openai_codex_oauth()
            .login(
                interaction(
                    never_aborted_signal(),
                    move |prompt| match &prompt.kind {
                        AuthPromptKind::Select { message, options } => {
                            select_for_prompt
                                .lock()
                                .push(crate::auth::types::AuthPrompt {
                                    signal: prompt.signal.clone(),
                                    kind: AuthPromptKind::Select {
                                        message: message.clone(),
                                        options: options.clone(),
                                    },
                                });
                            Ok("device_code".to_string())
                        }
                        _ => panic!("Text prompt should not be used"),
                    },
                    move |event| match event {
                        AuthEvent::AuthUrl { .. } => panic!("Browser login should not start"),
                        AuthEvent::DeviceCode {
                            user_code,
                            verification_uri,
                            interval_seconds,
                            expires_in_seconds,
                        } => {
                            device_infos_for_notify.lock().push((
                                user_code,
                                verification_uri,
                                interval_seconds,
                                expires_in_seconds,
                            ));
                        }
                        _ => {}
                    },
                ),
                None,
            )
            .await
            .expect("credentials");

        assert_eq!(credential.credentials.access, access_token);
        assert_eq!(
            credential
                .credentials
                .extra
                .get("accountId")
                .and_then(serde_json::Value::as_str),
            Some("account-456")
        );
        {
            let prompts = select_prompts.lock();
            assert_eq!(prompts.len(), 1);
            match &prompts[0].kind {
                AuthPromptKind::Select { message, options } => {
                    assert_eq!(message, "Select OpenAI Codex login method:");
                    assert_eq!(options.len(), 2);
                    assert_eq!(options[0].id, "browser");
                    assert_eq!(options[0].label, "Browser login (default)");
                    assert_eq!(options[1].id, "device_code");
                    assert_eq!(options[1].label, "Device code login (headless)");
                }
                _ => panic!("expected select prompt"),
            }
        }
        {
            let infos = device_infos.lock();
            assert_eq!(infos.len(), 1);
            assert_eq!(infos[0].0, "WXYZ-7890");
            assert_eq!(infos[0].1, "https://auth.openai.com/codex/device");
            assert_eq!(infos[0].2, Some(5));
            assert_eq!(infos[0].3, Some(900));
        }
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "cancels when OpenAI Codex login method selection is cancelled".
    #[tokio::test]
    async fn cancels_when_login_method_selection_is_cancelled() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        crate::auth::oauth::set_oauth_fetch(None);
        let error = openai_codex_oauth()
            .login(
                interaction(
                    never_aborted_signal(),
                    |_prompt| Err(ThrownValue::message("Login cancelled")),
                    |_event| {},
                ),
                None,
            )
            .await
            .expect_err("cancelled");
        assert_eq!(crate::models::thrown_message(&error), "Login cancelled");
    }

    // Port of "cancels the OpenAI Codex device code flow while waiting".
    #[tokio::test(start_paused = true)]
    async fn cancels_the_device_code_flow_while_waiting() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        let poll_times: Shared<Vec<i64>> = shared(Vec::new());
        let poll_times_for_fetch = poll_times.clone();
        let controller = AbortController::new();
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |mut request| {
            let url = request.url.clone();
            let poll_times_for_fetch = poll_times_for_fetch.clone();
            Box::pin(async move {
                if url == DEVICE_USER_CODE_URL {
                    let body =
                        body_json(std::mem::replace(&mut request.body, HttpBody::Empty)).await;
                    assert_eq!(body, serde_json::json!({ "client_id": CLIENT_ID }));
                    return Ok(json_response(
                        serde_json::json!({
                            "device_auth_id": "device-auth-id",
                            "user_code": "ABCD-1234",
                            "interval": "5",
                        }),
                        200,
                    ));
                }
                if url == DEVICE_TOKEN_URL {
                    poll_times_for_fetch
                        .lock()
                        .push(crate::utils::diagnostics::now_ms());
                    return Ok(device_auth_pending_response());
                }
                Err(ProviderHttpError::new(format!(
                    "Unexpected fetch URL: {url}"
                )))
            })
        })));

        let credentials =
            tokio::spawn(login_device_code_for_test(controller.signal(), |_event| {}));
        tokio::time::advance(std::time::Duration::from_millis(0)).await;
        assert_eq!(poll_times.lock().len(), 1);

        controller.abort();
        tokio::time::advance(std::time::Duration::from_millis(1)).await;
        let error = credentials.await.expect("task").expect_err("cancelled");
        assert_eq!(crate::models::thrown_message(&error), "Login cancelled");
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "times out the OpenAI Codex device code flow after 15 minutes".
    #[tokio::test(start_paused = true)]
    async fn times_out_the_device_code_flow_after_fifteen_minutes() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        let poll_times: Shared<Vec<i64>> = shared(Vec::new());
        let poll_times_for_fetch = poll_times.clone();
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |mut request| {
            let url = request.url.clone();
            let poll_times_for_fetch = poll_times_for_fetch.clone();
            Box::pin(async move {
                if url == DEVICE_USER_CODE_URL {
                    let body =
                        body_json(std::mem::replace(&mut request.body, HttpBody::Empty)).await;
                    assert_eq!(body, serde_json::json!({ "client_id": CLIENT_ID }));
                    return Ok(json_response(
                        serde_json::json!({
                            "device_auth_id": "device-auth-id",
                            "user_code": "ABCD-1234",
                            "interval": "60",
                        }),
                        200,
                    ));
                }
                if url == DEVICE_TOKEN_URL {
                    poll_times_for_fetch
                        .lock()
                        .push(crate::utils::diagnostics::now_ms());
                    return Ok(device_auth_pending_response());
                }
                Err(ProviderHttpError::new(format!(
                    "Unexpected fetch URL: {url}"
                )))
            })
        })));

        let credentials = tokio::spawn(login_device_code_for_test(
            never_aborted_signal(),
            |_event| {},
        ));
        tokio::time::advance(std::time::Duration::from_millis(0)).await;
        assert_eq!(poll_times.lock().len(), 1);

        tokio::time::advance(std::time::Duration::from_secs(15 * 60)).await;
        let error = credentials.await.expect("task").expect_err("timeout");
        assert_eq!(
            crate::models::thrown_message(&error),
            "Device flow timed out"
        );
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "treats OpenAI Codex device auth 403 and 404 responses as pending".
    #[tokio::test(start_paused = true)]
    async fn treats_403_and_404_responses_as_pending() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        let access_token = create_access_token("account-403-404");
        let poll_times: Shared<Vec<i64>> = shared(Vec::new());
        let poll_responses: Shared<Vec<HttpResponse>> = shared(vec![
            json_response(
                serde_json::json!({ "error": "access_denied", "error_description": "denied" }),
                403,
            ),
            text_response_404(),
            json_response(
                serde_json::json!({
                    "authorization_code": "oauth-code",
                    "code_challenge": "device-code-challenge",
                    "code_verifier": "device-code-verifier",
                }),
                200,
            ),
        ]);

        let poll_times_for_fetch = poll_times.clone();
        let poll_responses_for_fetch = poll_responses.clone();
        let access_token_for_fetch = access_token.clone();
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |request| {
            let url = request.url.clone();
            let poll_times_for_fetch = poll_times_for_fetch.clone();
            let poll_responses_for_fetch = poll_responses_for_fetch.clone();
            let access_token_for_fetch = access_token_for_fetch.clone();
            Box::pin(async move {
                if url == DEVICE_USER_CODE_URL {
                    return Ok(json_response(
                        serde_json::json!({
                            "device_auth_id": "device-auth-id",
                            "user_code": "ABCD-1234",
                            "interval": "1",
                        }),
                        200,
                    ));
                }
                if url == DEVICE_TOKEN_URL {
                    poll_times_for_fetch
                        .lock()
                        .push(crate::utils::diagnostics::now_ms());
                    return Ok(poll_responses_for_fetch.lock().remove(0));
                }
                if url == TOKEN_URL {
                    return Ok(json_response(
                        serde_json::json!({
                            "access_token": access_token_for_fetch,
                            "refresh_token": "refresh-token",
                            "expires_in": 3600,
                        }),
                        200,
                    ));
                }
                Err(ProviderHttpError::new(format!(
                    "Unexpected fetch URL: {url}"
                )))
            })
        })));

        let credentials = tokio::spawn(login_device_code_for_test(
            never_aborted_signal(),
            |_event| {},
        ));
        tokio::time::advance(std::time::Duration::from_millis(0)).await;
        tokio::time::advance(std::time::Duration::from_secs(1)).await;
        tokio::time::advance(std::time::Duration::from_secs(1)).await;

        let credential = credentials.await.expect("task").expect("credentials");
        assert_eq!(credential.credentials.access, access_token);
        assert_eq!(
            credential
                .credentials
                .extra
                .get("accountId")
                .and_then(serde_json::Value::as_str),
            Some("account-403-404")
        );
        assert_eq!(poll_times.lock().len(), 3);
        crate::auth::oauth::set_oauth_fetch(None);
    }

    fn text_response_404() -> HttpResponse {
        HttpResponse {
            status: 404,
            headers: vec![("content-type".to_string(), "text/plain".to_string())],
            body: HttpBody::bytes(b"not ready".to_vec()),
        }
    }

    // Port of "includes the response body in OpenAI Codex device auth poll
    // failures".
    #[tokio::test]
    async fn includes_the_response_body_in_poll_failures() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |request| {
            let url = request.url.clone();
            Box::pin(async move {
                if url == DEVICE_USER_CODE_URL {
                    return Ok(json_response(
                        serde_json::json!({
                            "device_auth_id": "device-auth-id",
                            "user_code": "ABCD-1234",
                            "interval": "5",
                        }),
                        200,
                    ));
                }
                if url == DEVICE_TOKEN_URL {
                    return Ok(json_response(
                        serde_json::json!({ "error": "server_error", "error_description": "try again later" }),
                        500,
                    ));
                }
                Err(ProviderHttpError::new(format!(
                    "Unexpected fetch URL: {url}"
                )))
            })
        })));

        let error = login_device_code_for_test(never_aborted_signal(), |_event| {})
            .await
            .expect_err("failed");
        assert_eq!(
			crate::models::thrown_message(&error),
			"OpenAI Codex device auth failed with status 500: {\"error\":\"server_error\",\"error_description\":\"try again later\"}"
		);
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "does not write token refresh failures to stderr" — Rust has no
    // console.error path; the assertion reduces to the rejection message.
    #[tokio::test]
    async fn surfaces_refresh_failure_body_without_logging() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |_request| {
            Box::pin(async {
                Ok(json_response(
                    serde_json::json!({
                        "error": {
                            "message": "Could not validate your token. Please try signing in again.",
                            "type": "invalid_request_error",
                        }
                    }),
                    401,
                ))
            })
        })));

        let error = openai_codex_oauth()
            .refresh(
                OAuthCredential {
                    credentials: OAuthCredentials {
                        refresh: "invalid-refresh-token".to_string(),
                        access: "invalid-access-token".to_string(),
                        expires: 0,
                        extra: Default::default(),
                    },
                },
                never_aborted_signal(),
            )
            .await
            .expect_err("failed");
        let message = crate::models::thrown_message(&error);
        assert!(
            message.starts_with("OpenAI Codex token refresh failed (401)"),
            "got: {message}"
        );
        assert!(
            message.contains("Could not validate your token"),
            "got: {message}"
        );
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "falls back to the pasted redirect URL when the fixed callback
    // port is taken".
    #[tokio::test]
    async fn falls_back_to_the_pasted_redirect_url_when_the_fixed_port_is_taken() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        // Occupy port 1455 unless it already is.
        let blocker = tokio::net::TcpListener::bind(("127.0.0.1", 1455))
            .await
            .expect("block port 1455 for the fallback test");

        let access_token = create_access_token("acct");
        let exchange_seen: Shared<Option<std::collections::BTreeMap<String, String>>> =
            shared(None);
        let exchange_for_fetch = exchange_seen.clone();
        let access_for_fetch = access_token.clone();
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |mut request| {
            let url = request.url.clone();
            let exchange_for_fetch = exchange_for_fetch.clone();
            let access_for_fetch = access_for_fetch.clone();
            Box::pin(async move {
                assert_eq!(url, TOKEN_URL);
                let body = std::mem::replace(&mut request.body, HttpBody::Empty)
                    .collect()
                    .await
                    .expect("body");
                let exchange =
                    crate::auth::oauth::parse_form(&String::from_utf8(body).expect("utf8"));
                *exchange_for_fetch.lock() = Some(exchange);
                Ok(json_response(
                    serde_json::json!({
                        "access_token": access_for_fetch,
                        "refresh_token": "refresh",
                        "expires_in": 3600,
                    }),
                    200,
                ))
            })
        })));

        let auth_url: Shared<String> = shared(String::new());
        let auth_url_for_notify = auth_url.clone();
        let auth_url_for_prompt = auth_url.clone();
        let credential = openai_codex_oauth()
			.login(
				interaction(
					never_aborted_signal(),
					move |prompt| match &prompt.kind {
						AuthPromptKind::Select { .. } => Ok("browser".to_string()),
						AuthPromptKind::ManualCode { .. } => {
							let state = url::Url::parse(&auth_url_for_prompt.lock())
								.expect("auth url")
								.query_pairs()
								.find(|(key, _)| key == "state")
								.map(|(_, value)| value.into_owned())
								.expect("state");
							Ok(format!("http://localhost:1455/auth/callback?code=pasted-code&state={state}"))
						}
						_ => panic!("Unexpected prompt"),
					},
					move |event| {
						if let AuthEvent::AuthUrl { url, .. } = event {
							*auth_url_for_notify.lock() = url;
						}
					},
				),
				None,
			)
			.await
			.expect("credential");

        assert_eq!(credential.credentials.access, access_token);
        assert_eq!(
            credential
                .credentials
                .extra
                .get("accountId")
                .and_then(serde_json::Value::as_str),
            Some("acct")
        );
        {
            let exchange = exchange_seen.lock().clone().expect("exchange captured");
            assert_eq!(
                exchange.get("code").map(String::as_str),
                Some("pasted-code")
            );
            assert_eq!(
                exchange.get("redirect_uri").map(String::as_str),
                Some("http://localhost:1455/auth/callback")
            );
        }
        crate::auth::oauth::set_oauth_fetch(None);
        drop(blocker);
    }
}
