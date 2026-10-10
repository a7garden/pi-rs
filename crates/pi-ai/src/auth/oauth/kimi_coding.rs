//! Port of `packages/ai/src/auth/oauth/kimi-coding.ts` — the Kimi Code
//! (subscription) OAuth flow.
//!
//! RFC 8628 device authorization grant against https://auth.kimi.com with
//! JSON responses. The access token authenticates requests to
//! https://api.kimi.com/coding as an `Authorization: Bearer` header. Token
//! refresh retries 429/5xx up to three times with 1s/2s/4s backoff;
//! 401/403/`invalid_grant` fail fast (the stored credential is dead).

use crate::auth::oauth::device_code::{
    poll_oauth_device_code_flow, OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult,
};
use crate::auth::types::{
    ModelAuth, OAuthAuth, OAuthCredential, OAuthCredentials, OAuthLoginFn, OAuthRefreshFn,
    OAuthToAuthFn, ProviderAuthInteraction,
};
use crate::http::{HttpBody, HttpRequest, HttpResponse};
use crate::types::ProviderHeaders;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::thrown::ThrownValue;

const CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
const DEFAULT_OAUTH_HOST: &str = "https://auth.kimi.com";
const DEVICE_CODE_TIMEOUT_SECONDS: f64 = 15.0 * 60.0;
const DEFAULT_POLL_INTERVAL_SECONDS: f64 = 5.0;
const REQUEST_TIMEOUT_MS: u64 = 30 * 1000;
const REFRESH_MAX_RETRIES: u32 = 3;

fn status_ok(status: u16) -> bool {
    (200..300).contains(&status)
}

struct DeviceAuthorization {
    device_code: String,
    user_code: String,
    verification_uri: String,
    verification_uri_complete: String,
    interval_seconds: f64,
    expires_in_seconds: f64,
}

struct TokenResponse {
    access: String,
    refresh: String,
    /// Unix timestamp in milliseconds.
    expires: i64,
}

fn get_oauth_host() -> String {
    let override_host = get_provider_env_value("KIMI_CODE_OAUTH_HOST", None)
        .or_else(|| get_provider_env_value("KIMI_OAUTH_HOST", None));
    let host = override_host.unwrap_or_else(|| DEFAULT_OAUTH_HOST.to_string());
    let trimmed = host.trim_end_matches('/');
    trimmed.to_string()
}

fn request_signal(signal: &crate::utils::abort::AbortSignal) -> crate::utils::abort::AbortSignal {
    crate::auth::oauth::timeout_signal(signal, REQUEST_TIMEOUT_MS)
}

async fn response_text(response: &mut HttpResponse) -> String {
    let body = std::mem::replace(&mut response.body, HttpBody::Empty);
    match body.collect().await {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(_) => String::new(),
    }
}

/// Upstream `readJson` — `null` on any parse failure or non-object.
async fn read_json(response: &mut HttpResponse) -> Option<serde_json::Value> {
    let text = response_text(response).await;
    serde_json::from_str(&text)
        .ok()
        .filter(|json: &serde_json::Value| json.is_object())
}

/// The verification URI is opened in the user's browser; only http(s) URLs are
/// trusted (upstream `trustedHttpUrl`).
fn trusted_http_url(value: &str) -> bool {
    match url::Url::parse(value) {
        Ok(url) => url.scheme() == "https" || url.scheme() == "http",
        Err(_) => false,
    }
}

async fn form_post(
    url: &str,
    fields: Vec<(&str, &str)>,
    signal: &crate::utils::abort::AbortSignal,
) -> Result<HttpResponse, ThrownValue> {
    let mut request = HttpRequest::new("POST", url);
    request.headers.push((
        "Content-Type".to_string(),
        "application/x-www-form-urlencoded".to_string(),
    ));
    request
        .headers
        .push(("Accept".to_string(), "application/json".to_string()));
    request.body = HttpBody::bytes(crate::auth::oauth::form_urlencode(&fields).into_bytes());
    request.signal = Some(request_signal(signal));
    crate::auth::oauth::oauth_fetch()
        .send(request)
        .await
        .map_err(|error| ThrownValue::message(error.message.clone()))
}

async fn start_device_authorization(
    oauth_host: &str,
    signal: &crate::utils::abort::AbortSignal,
) -> Result<DeviceAuthorization, ThrownValue> {
    let mut response = form_post(
        &format!("{oauth_host}/api/oauth/device_authorization"),
        vec![("client_id", CLIENT_ID)],
        signal,
    )
    .await?;
    let status = response.status;
    if !status_ok(status) {
        let text = response_text(&mut response).await;
        return Err(ThrownValue::message(format!(
            "Kimi Code device authorization failed with status {status}{}",
            if text.is_empty() {
                String::new()
            } else {
                format!(": {text}")
            }
        )));
    }

    let mut response = response;
    let json = read_json(&mut response).await;
    let get = |key: &str| {
        json.as_ref()
            .and_then(|json| json.get(key))
            .and_then(serde_json::Value::as_str)
    };
    let device_code = get("device_code");
    let user_code = get("user_code");
    let verification_uri = get("verification_uri");
    let verification_uri_complete = get("verification_uri_complete");
    let (
        Some(device_code),
        Some(user_code),
        Some(verification_uri),
        Some(verification_uri_complete),
    ) = (
        device_code,
        user_code,
        verification_uri,
        verification_uri_complete,
    )
    else {
        return Err(ThrownValue::message(format!(
            "Invalid Kimi Code device authorization response: {}",
            json.map(|json| json.to_string())
                .unwrap_or_else(|| "null".to_string())
        )));
    };
    if !trusted_http_url(verification_uri_complete) || !trusted_http_url(verification_uri) {
        return Err(ThrownValue::message(format!(
            "Invalid Kimi Code device authorization response: {}",
            json.map(|json| json.to_string())
                .unwrap_or_else(|| "null".to_string())
        )));
    }

    let interval = json
        .as_ref()
        .and_then(|json| json.get("interval"))
        .and_then(serde_json::Value::as_f64);
    let expires_in = json
        .as_ref()
        .and_then(|json| json.get("expires_in"))
        .and_then(serde_json::Value::as_f64);
    Ok(DeviceAuthorization {
        device_code: device_code.to_string(),
        user_code: user_code.to_string(),
        verification_uri: verification_uri.to_string(),
        verification_uri_complete: verification_uri_complete.to_string(),
        interval_seconds: match interval {
            Some(interval) if interval.is_finite() && interval > 0.0 => interval,
            _ => DEFAULT_POLL_INTERVAL_SECONDS,
        },
        expires_in_seconds: match expires_in {
            Some(expires_in) if expires_in.is_finite() && expires_in > 0.0 => expires_in,
            _ => DEVICE_CODE_TIMEOUT_SECONDS,
        },
    })
}

fn parse_token_response(
    json: Option<&serde_json::Value>,
    operation: &str,
) -> Result<TokenResponse, ThrownValue> {
    let access_token = json
        .and_then(|json| json.get("access_token"))
        .and_then(serde_json::Value::as_str);
    let refresh_token = json
        .and_then(|json| json.get("refresh_token"))
        .and_then(serde_json::Value::as_str);
    let expires_in = json
        .and_then(|json| json.get("expires_in"))
        .and_then(serde_json::Value::as_f64);
    match (access_token, refresh_token, expires_in) {
        (Some(access), Some(refresh), Some(expires_in))
            if !access.is_empty()
                && !refresh.is_empty()
                && expires_in.is_finite()
                && expires_in > 0.0 =>
        {
            Ok(TokenResponse {
                access: access.to_string(),
                refresh: refresh.to_string(),
                expires: crate::utils::diagnostics::now_ms() + (expires_in * 1000.0) as i64,
            })
        }
        _ => Err(ThrownValue::message(format!(
            "Kimi Code token {operation} response missing fields: {}",
            json.map(|json| json.to_string())
                .unwrap_or_else(|| "null".to_string())
        ))),
    }
}

async fn poll_for_token(
    oauth_host: String,
    device: &DeviceAuthorization,
    signal: &crate::utils::abort::AbortSignal,
) -> Result<TokenResponse, ThrownValue> {
    let device_code = device.device_code.clone();
    let expires_in_seconds = device.expires_in_seconds;
    let interval_seconds = device.interval_seconds;
    poll_oauth_device_code_flow(
        OAuthDeviceCodePollOptions {
            interval_seconds: Some(interval_seconds),
            expires_in_seconds: Some(expires_in_seconds),
            wait_before_first_poll: true,
            signal: signal.clone(),
        },
        move || {
            let oauth_host = oauth_host.clone();
            let device_code = device_code.clone();
            let signal = signal.clone();
            async move {
                let response = form_post(
                    &format!("{oauth_host}/api/oauth/token"),
                    vec![
                        ("client_id", CLIENT_ID),
                        ("device_code", device_code.as_str()),
                        ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ],
                    &signal,
                )
                .await?;

                if response.status >= 500 {
                    let mut response = response;
                    let text = response_text(&mut response).await;
                    return Ok(OAuthDeviceCodePollResult::<TokenResponse>::Failed {
                        message: format!(
                            "Kimi Code device token request failed with status {}{}",
                            response.status,
                            if text.is_empty() {
                                String::new()
                            } else {
                                format!(": {text}")
                            }
                        ),
                    });
                }

                let mut response = response;
                let json = read_json(&mut response).await;
                if response.ok()
                    && json
                        .as_ref()
                        .and_then(|json| json.get("access_token"))
                        .and_then(serde_json::Value::as_str)
                        .is_some()
                {
                    return match parse_token_response(json.as_ref(), "poll") {
                        Ok(token) => Ok(OAuthDeviceCodePollResult::Complete { value: token }),
                        Err(error) => Ok(OAuthDeviceCodePollResult::Failed {
                            message: crate::utils::diagnostics::format_thrown_value(&error),
                        }),
                    };
                }

                let error = json.as_ref().and_then(|json| json.get("error"));
                let error_str = error.and_then(serde_json::Value::as_str);
                let description = match error.and_then(serde_json::Value::as_str) {
                    Some(_) => json
                        .as_ref()
                        .and_then(|json| json.get("error_description"))
                        .and_then(serde_json::Value::as_str)
                        .map(|description| format!(": {description}"))
                        .unwrap_or_default(),
                    None => String::new(),
                };
                match error_str {
                    Some("authorization_pending") => Ok(OAuthDeviceCodePollResult::Pending),
                    Some("slow_down") => Ok(OAuthDeviceCodePollResult::SlowDown {
                        interval_seconds: json
                            .as_ref()
                            .and_then(|json| json.get("interval"))
                            .and_then(serde_json::Value::as_f64)
                            .filter(|interval: &f64| *interval > 0.0),
                    }),
                    Some("expired_token") => Ok(OAuthDeviceCodePollResult::Failed {
                        message: "Kimi Code device authorization expired. Please restart login."
                            .to_string(),
                    }),
                    Some("access_denied") => Ok(OAuthDeviceCodePollResult::Failed {
                        message: "Kimi Code login was denied.".to_string(),
                    }),
                    _ => Ok(OAuthDeviceCodePollResult::Failed {
                        message: format!(
                            "Kimi Code device token request failed (status {}){}",
                            response.status,
                            match error_str {
                                Some(_) => format!(": {}{description}", error_str.unwrap()),
                                None => String::new(),
                            }
                        ),
                    }),
                }
            }
        },
    )
    .await
}

fn is_retryable_refresh_failure(status: u16) -> bool {
    status == 429 || status >= 500
}

async fn refresh_token(
    oauth_host: &str,
    refresh_token_value: &str,
    signal: &crate::utils::abort::AbortSignal,
) -> Result<TokenResponse, ThrownValue> {
    let mut last_error: Option<ThrownValue> = None;
    for attempt in 0..=REFRESH_MAX_RETRIES {
        if attempt > 0 {
            crate::utils::sleep::sleep(1000 * 2u64.pow(attempt - 1), signal)
                .await
                .map_err(|_| ThrownValue::message("Kimi Code token refresh aborted"))?;
        }
        if signal.aborted() {
            return Err(ThrownValue::message("Kimi Code token refresh aborted"));
        }

        let response = form_post(
            &format!("{oauth_host}/api/oauth/token"),
            vec![
                ("client_id", CLIENT_ID),
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token_value),
            ],
            signal,
        )
        .await;

        let response = match response {
            Ok(response) => response,
            Err(error) => {
                last_error = Some(error);
                continue;
            }
        };

        let mut response = response;
        let json = read_json(&mut response).await;
        if response.ok() {
            return parse_token_response(json.as_ref(), "refresh");
        }

        // Unauthorized: the stored credential is dead; Models clears it and
        // prompts re-login.
        if response.status == 401
            || response.status == 403
            || json
                .as_ref()
                .and_then(|json| json.get("error"))
                .and_then(serde_json::Value::as_str)
                == Some("invalid_grant")
        {
            let description = json
                .as_ref()
                .and_then(|json| json.get("error_description"))
                .and_then(serde_json::Value::as_str)
                .map(|description| format!(": {description}"))
                .unwrap_or_default();
            return Err(ThrownValue::message(format!(
                "Kimi Code token refresh unauthorized (status {}){description}",
                response.status
            )));
        }

        if is_retryable_refresh_failure(response.status) && attempt < REFRESH_MAX_RETRIES {
            last_error = Some(ThrownValue::message(format!(
                "Kimi Code token refresh failed with status {}",
                response.status
            )));
            continue;
        }

        let text = json
            .map(|json| json.to_string())
            .unwrap_or_else(|| "null".to_string());
        return Err(ThrownValue::message(format!(
            "Kimi Code token refresh failed with status {}{}",
            response.status,
            if text.is_empty() {
                String::new()
            } else {
                format!(": {text}")
            }
        )));
    }

    Err(last_error.unwrap_or_else(|| ThrownValue::message("Kimi Code token refresh failed")))
}

fn credential_from_token(token: TokenResponse) -> OAuthCredential {
    OAuthCredential {
        credentials: OAuthCredentials {
            refresh: token.refresh,
            access: token.access,
            expires: token.expires,
            extra: Default::default(),
        },
    }
}

async fn login_kimi_coding(
    interaction: &ProviderAuthInteraction,
) -> Result<OAuthCredential, ThrownValue> {
    let oauth_host = get_oauth_host();
    let device = start_device_authorization(&oauth_host, &interaction.signal).await?;
    (interaction.notify)(crate::auth::types::AuthEvent::DeviceCode {
        user_code: device.user_code.clone(),
        verification_uri: device.verification_uri_complete.clone(),
        interval_seconds: Some(device.interval_seconds as u64),
        expires_in_seconds: Some(device.expires_in_seconds as u64),
    });
    let token = poll_for_token(oauth_host, &device, &interaction.signal).await?;
    Ok(credential_from_token(token))
}

/// Upstream `export const kimiCodingOAuth`.
pub fn kimi_coding_oauth() -> OAuthAuth {
    let login: OAuthLoginFn = Box::new(|interaction, _options| {
        Box::pin(async move { login_kimi_coding(&interaction).await })
    });

    let refresh: OAuthRefreshFn = Box::new(|credential, signal| {
        Box::pin(async move {
            let token =
                refresh_token(&get_oauth_host(), &credential.credentials.refresh, &signal).await?;
            Ok(credential_from_token(token))
        })
    });

    let to_auth: OAuthToAuthFn = Box::new(|credential| {
        let access = credential.credentials.access.clone();
        Box::pin(async move {
            let mut headers = ProviderHeaders::new();
            headers.insert(
                "Authorization".to_string(),
                Some(format!("Bearer {access}")),
            );
            Ok(ModelAuth {
                api_key: None,
                headers: Some(headers),
                base_url: None,
            })
        })
    });

    OAuthAuth {
        name: "Kimi Code (subscription)".to_string(),
        is_subscription: true,
        login_label: Some("Sign in with Kimi Code".to_string()),
        login,
        refresh,
        to_auth,
    }
}
