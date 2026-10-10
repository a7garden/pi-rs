//! Port of `packages/ai/src/auth/oauth/meta.ts` — the Meta Model API OAuth
//! flow (RFC 8628 device authorization grant against https://auth.meta.com).
//!
//! Meta splits identity from API access: the identity token (stored as
//! `refresh`) is exchanged for a Model API key (stored as `access`) via the
//! Muse Code key-mint endpoint; minted keys live about a day and are re-minted
//! by the standard refresh path. The identity token itself is not renewable —
//! a 401/403 from mint means the session is dead.

use crate::auth::oauth::device_code::{
    poll_oauth_device_code_flow, OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult,
};
use crate::auth::types::{
    ModelAuth, OAuthAuth, OAuthCredential, OAuthCredentials, OAuthLoginFn, OAuthRefreshFn,
    OAuthToAuthFn, ProviderAuthInteraction,
};
use crate::http::{HttpBody, HttpRequest};
use crate::utils::abort::AbortSignal;
use crate::utils::thrown::ThrownValue;

/// Muse Code CLI client id.
const CLIENT_ID: &str = "1031625952748946";
const AUTH_HOST: &str = "https://auth.meta.com";
const DEVICE_AUTHORIZATION_URL: &str = "https://auth.meta.com/oidc/device/authorization/";
const DEVICE_TOKEN_URL: &str = "https://auth.meta.com/oidc/device/token/";
const API_KEY_MINT_URL: &str = "https://api.meta.ai/muse-code/key";
const API_KEY_LIFETIME_MS: i64 = 24 * 60 * 60 * 1000;
const REQUEST_TIMEOUT_MS: u64 = 30 * 1000;

fn request_signal(signal: &AbortSignal) -> AbortSignal {
    crate::auth::oauth::timeout_signal(signal, REQUEST_TIMEOUT_MS)
}

async fn response_text(response: &mut crate::http::HttpResponse) -> String {
    let body = std::mem::replace(&mut response.body, crate::http::HttpBody::Empty);
    match body.collect().await {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(_) => String::new(),
    }
}

/// Upstream `readJson` — `null` on any parse failure or non-object.
async fn read_json(response: &mut crate::http::HttpResponse) -> Option<serde_json::Value> {
    let text = response_text(response).await;
    serde_json::from_str(&text)
        .ok()
        .filter(|json: &serde_json::Value| json.is_object())
}

/// Upstream `errorDetail`.
fn error_detail(json: Option<&serde_json::Value>) -> String {
    for key in ["error_description", "detail", "message", "error"] {
        if let Some(value) = json
            .and_then(|json| json.get(key))
            .and_then(serde_json::Value::as_str)
        {
            if !value.trim().is_empty() {
                return format!(": {}", value.trim());
            }
        }
    }
    String::new()
}

/// The verification URI is opened in the user's browser; only http(s) URLs are
/// trusted (upstream `trustedHttpUrl`).
fn trusted_http_url(value: Option<&serde_json::Value>) -> Option<String> {
    let value = value?.as_str()?;
    if value.is_empty() {
        return None;
    }
    let url = url::Url::parse(value).ok()?;
    if url.scheme() != "https" && url.scheme() != "http" {
        return None;
    }
    Some(url.to_string())
}

fn positive_number(value: Option<&serde_json::Value>) -> Option<f64> {
    match value {
        Some(serde_json::Value::Number(number)) => {
            let value = number.as_f64()?;
            if value.is_finite() && value > 0.0 {
                Some(value)
            } else {
                None
            }
        }
        _ => None,
    }
}

struct DeviceAuthorization {
    device_code: String,
    user_code: String,
    verification_uri: String,
    interval_seconds: Option<f64>,
    expires_in_seconds: Option<f64>,
}

async fn form_post(
    url: &str,
    fields: Vec<(&str, &str)>,
    signal: &AbortSignal,
) -> Result<crate::http::HttpResponse, ThrownValue> {
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
    signal: &AbortSignal,
) -> Result<DeviceAuthorization, ThrownValue> {
    let response = form_post(
        DEVICE_AUTHORIZATION_URL,
        vec![("client_id", CLIENT_ID)],
        signal,
    )
    .await?;
    let mut response = response;
    let json = read_json(&mut response).await;
    if !response.ok() {
        return Err(ThrownValue::message(format!(
            "Meta device authorization failed with status {}{}",
            response.status,
            error_detail(json.as_ref())
        )));
    }
    let device_code = json
        .as_ref()
        .and_then(|json| json.get("device_code"))
        .and_then(serde_json::Value::as_str);
    let user_code = json
        .as_ref()
        .and_then(|json| json.get("user_code"))
        .and_then(serde_json::Value::as_str);
    let verification_uri = trusted_http_url(
        json.as_ref()
            .and_then(|json| json.get("verification_uri_complete"))
            .or_else(|| json.as_ref().and_then(|json| json.get("verification_uri"))),
    );
    let (Some(device_code), Some(user_code), Some(verification_uri)) =
        (device_code, user_code, verification_uri)
    else {
        return Err(ThrownValue::message(format!(
            "Invalid Meta device authorization response: {}",
            json.map(|json| json.to_string())
                .unwrap_or_else(|| "null".to_string())
        )));
    };
    Ok(DeviceAuthorization {
        device_code: device_code.to_string(),
        user_code: user_code.to_string(),
        verification_uri,
        interval_seconds: positive_number(json.as_ref().and_then(|json| json.get("interval"))),
        expires_in_seconds: positive_number(json.as_ref().and_then(|json| json.get("expires_in"))),
    })
}

async fn poll_for_identity_token(
    device: &DeviceAuthorization,
    signal: &AbortSignal,
) -> Result<String, ThrownValue> {
    let device_code = device.device_code.clone();
    poll_oauth_device_code_flow(
        OAuthDeviceCodePollOptions {
            interval_seconds: device.interval_seconds,
            expires_in_seconds: device.expires_in_seconds,
            wait_before_first_poll: true,
            signal: signal.clone(),
        },
        move || {
            let device_code = device_code.clone();
            let signal = signal.clone();
            async move {
                let response = form_post(
                    DEVICE_TOKEN_URL,
                    vec![
                        ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                        ("device_code", device_code.as_str()),
                        ("client_id", CLIENT_ID),
                    ],
                    &signal,
                )
                .await?;
                let mut response = response;
                let json = read_json(&mut response).await;
                if response.ok() {
                    if let Some(access_token) = json
                        .as_ref()
                        .and_then(|json| json.get("access_token"))
                        .and_then(serde_json::Value::as_str)
                        .filter(|token| !token.is_empty())
                    {
                        return Ok(OAuthDeviceCodePollResult::<String>::Complete {
                            value: access_token.to_string(),
                        });
                    }
                }
                match json
                    .as_ref()
                    .and_then(|json| json.get("error"))
                    .and_then(serde_json::Value::as_str)
                {
                    Some("authorization_pending") => Ok(OAuthDeviceCodePollResult::Pending),
                    Some("slow_down") => Ok(OAuthDeviceCodePollResult::SlowDown {
                        interval_seconds: positive_number(
                            json.as_ref().and_then(|json| json.get("interval")),
                        ),
                    }),
                    Some("access_denied") => Ok(OAuthDeviceCodePollResult::Failed {
                        message: "Meta login was denied.".to_string(),
                    }),
                    Some("expired_token") => Ok(OAuthDeviceCodePollResult::Failed {
                        message: "Meta device authorization expired. Please restart login."
                            .to_string(),
                    }),
                    _ => Ok(OAuthDeviceCodePollResult::Failed {
                        message: format!(
                            "Meta device token request failed with status {}{}",
                            response.status,
                            error_detail(json.as_ref())
                        ),
                    }),
                }
            }
        },
    )
    .await
}

/// Exchange an identity token for a Model API key. Keys are valid for about a
/// day.
async fn mint_api_key(
    identity_token: &str,
    signal: &AbortSignal,
) -> Result<OAuthCredential, ThrownValue> {
    let mut request = HttpRequest::new("POST", API_KEY_MINT_URL);
    request
        .headers
        .push(("Accept".to_string(), "application/json".to_string()));
    request.headers.push((
        "Authorization".to_string(),
        format!("Bearer {identity_token}"),
    ));
    request
        .headers
        .push(("Content-Type".to_string(), "application/json".to_string()));
    request
        .headers
        .push(("x-api-version".to_string(), "1.0.0".to_string()));
    request.body = HttpBody::bytes(b"{}".to_vec());
    request.signal = Some(request_signal(signal));

    let response = crate::auth::oauth::oauth_fetch()
        .send(request)
        .await
        .map_err(|error| {
            if signal.aborted() {
                ThrownValue::message("Login cancelled")
            } else {
                ThrownValue::message(error.message.clone())
            }
        })?;
    let mut response = response;
    let json = read_json(&mut response).await;
    if response.status == 401 || response.status == 403 {
        // Identity token is not renewable (see file header); only a fresh
        // device flow helps.
        return Err(ThrownValue::message(format!(
            "Meta session expired (status {}). Run `/login meta` to sign in again.{}",
            response.status,
            error_detail(json.as_ref())
        )));
    }
    if !response.ok() {
        return Err(ThrownValue::message(format!(
            "Meta API key mint failed with status {}{}",
            response.status,
            error_detail(json.as_ref())
        )));
    }
    match json
        .as_ref()
        .and_then(|json| json.get("api_key"))
        .and_then(serde_json::Value::as_str)
    {
        Some(api_key) if !api_key.is_empty() => Ok(OAuthCredential {
            credentials: OAuthCredentials {
                refresh: identity_token.to_string(),
                access: api_key.to_string(),
                expires: crate::utils::diagnostics::now_ms() + API_KEY_LIFETIME_MS,
                extra: Default::default(),
            },
        }),
        _ => {
            let action_url =
                trusted_http_url(json.as_ref().and_then(|json| json.get("action_url")));
            Err(ThrownValue::message(format!(
                "Meta did not issue an API key.{}",
                action_url
                    .map(|url| format!(" Complete setup at {url}"))
                    .unwrap_or_default()
            )))
        }
    }
}

async fn login_meta(interaction: &ProviderAuthInteraction) -> Result<OAuthCredential, ThrownValue> {
    let result = async {
        let device = start_device_authorization(&interaction.signal).await?;
        (interaction.notify)(crate::auth::types::AuthEvent::DeviceCode {
            user_code: device.user_code.clone(),
            verification_uri: device.verification_uri.clone(),
            interval_seconds: device.interval_seconds.map(|seconds| seconds as u64),
            expires_in_seconds: device.expires_in_seconds.map(|seconds| seconds as u64),
        });
        let identity_token = poll_for_identity_token(&device, &interaction.signal).await?;
        (interaction.notify)(crate::auth::types::AuthEvent::Progress {
            message: "Enabling Meta Model API access...".to_string(),
        });
        mint_api_key(&identity_token, &interaction.signal).await
    }
    .await;
    // An in-flight fetch rejects with a DOMException on abort; the login UI
    // matches on this message.
    if interaction.signal.aborted() {
        return Err(ThrownValue::message("Login cancelled"));
    }
    result
}

/// Upstream `export const metaOAuth`.
pub fn meta_oauth() -> OAuthAuth {
    let login: OAuthLoginFn =
        Box::new(|interaction, _options| Box::pin(async move { login_meta(&interaction).await }));

    let refresh: OAuthRefreshFn = Box::new(|credential, signal| {
        Box::pin(async move { mint_api_key(&credential.credentials.refresh, &signal).await })
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
        name: "Meta (Muse subscription)".to_string(),
        is_subscription: true,
        login_label: Some("Sign in with Meta".to_string()),
        login,
        refresh,
        to_auth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::oauth::OAuthTestExt;
    use crate::auth::types::AuthEvent;
    use crate::http::{FetchFn, HttpResponse, ProviderHttpError};
    use crate::utils::abort::AbortController;
    use parking_lot::Mutex;
    use std::sync::Arc;

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

    fn interaction(notify: impl Fn(AuthEvent) + Send + Sync + 'static) -> ProviderAuthInteraction {
        ProviderAuthInteraction {
            signal: never_aborted_signal(),
            prompt: Box::new(|_prompt| {
                Box::pin(async { Err(ThrownValue::message("Meta login should not prompt")) })
            }),
            notify: Box::new(notify),
        }
    }

    // Port of test/meta-oauth.test.ts "logs in with the device flow and mints a
    // Model API key".
    #[tokio::test(start_paused = true)]
    async fn logs_in_with_the_device_flow_and_mints_a_model_api_key() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        let poll_responses: Arc<Mutex<Vec<HttpResponse>>> = Arc::new(Mutex::new(vec![
            json_response(serde_json::json!({ "error": "authorization_pending" }), 400),
            json_response(
                serde_json::json!({ "access_token": "identity-token", "token_type": "Bearer" }),
                200,
            ),
        ]));
        let poll_for_fetch = poll_responses.clone();
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |mut request| {
            let url = request.url.clone();
            let poll_for_fetch = poll_for_fetch.clone();
            Box::pin(async move {
                if url == DEVICE_AUTHORIZATION_URL {
                    assert_eq!(request.method, "POST");
                    let body = std::mem::replace(&mut request.body, HttpBody::Empty)
                        .collect()
                        .await
                        .expect("body");
                    let form =
                        crate::auth::oauth::parse_form(&String::from_utf8(body).expect("utf8"));
                    assert_eq!(form.get("client_id").map(String::as_str), Some(CLIENT_ID));
                    return Ok(json_response(
                        serde_json::json!({
                            "device_code": "device-code-123",
                            "user_code": "ABCD-1234",
                            "verification_uri": "https://auth.meta.com/oauth/device/",
                            "verification_uri_complete": "https://auth.meta.com/oauth/device/?code=ABCD-1234",
                            "interval": 5,
                            "expires_in": 600,
                        }),
                        200,
                    ));
                }
                if url == DEVICE_TOKEN_URL {
                    let body = std::mem::replace(&mut request.body, HttpBody::Empty)
                        .collect()
                        .await
                        .expect("body");
                    let form =
                        crate::auth::oauth::parse_form(&String::from_utf8(body).expect("utf8"));
                    assert_eq!(
                        form.get("grant_type").map(String::as_str),
                        Some("urn:ietf:params:oauth:grant-type:device_code")
                    );
                    assert_eq!(form.get("client_id").map(String::as_str), Some(CLIENT_ID));
                    assert_eq!(
                        form.get("device_code").map(String::as_str),
                        Some("device-code-123")
                    );
                    return Ok(poll_for_fetch.lock().remove(0));
                }
                if url == API_KEY_MINT_URL {
                    assert_eq!(request.method, "POST");
                    assert!(request
                        .headers
                        .iter()
                        .any(|(name, value)| name == "Authorization"
                            && value == "Bearer identity-token"));
                    return Ok(json_response(
                        serde_json::json!({ "api_key": "LLM|minted-key" }),
                        200,
                    ));
                }
                Err(ProviderHttpError::new(format!(
                    "Unexpected fetch URL: {url}"
                )))
            })
        })));

        let events: Arc<Mutex<Vec<AuthEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let events_for_notify = events.clone();
        let credential = tokio::spawn(meta_oauth().login(
            interaction(move |event| events_for_notify.lock().push(event)),
            None,
        ));

        // The first poll happens after the 5s wait-before-first-poll.
        tokio::time::advance(std::time::Duration::from_millis(0)).await;
        {
            let events = events.lock();
            assert!(
                matches!(events.first(), Some(AuthEvent::DeviceCode { user_code, verification_uri, interval_seconds, expires_in_seconds })
				if user_code == "ABCD-1234"
					&& verification_uri == "https://auth.meta.com/oauth/device/?code=ABCD-1234"
					&& *interval_seconds == Some(5)
					&& *expires_in_seconds == Some(600))
            );
        }

        tokio::time::advance(std::time::Duration::from_secs(10)).await;
        let credential = credential.await.expect("task").expect("credentials");
        assert_eq!(credential.credentials.refresh, "identity-token");
        assert_eq!(credential.credentials.access, "LLM|minted-key");
        assert_eq!(
            credential.credentials.expires,
            crate::utils::diagnostics::now_ms() + DAY_MS()
        );
        crate::auth::oauth::set_oauth_fetch(None);
    }

    fn DAY_MS() -> i64 {
        API_KEY_LIFETIME_MS
    }

    // Port of "re-mints the API key from the stored identity token on refresh".
    #[tokio::test]
    async fn re_mints_the_api_key_from_the_stored_identity_token_on_refresh() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |mut request| {
            let url = request.url.clone();
            Box::pin(async move {
                assert_eq!(url, API_KEY_MINT_URL);
                assert!(request.headers.iter().any(
                    |(name, value)| name == "Authorization" && value == "Bearer identity-token"
                ));
                Ok(json_response(
                    serde_json::json!({ "api_key": "LLM|fresh-key" }),
                    200,
                ))
            })
        })));

        let credential = meta_oauth()
            .refresh(
                OAuthCredential {
                    credentials: OAuthCredentials {
                        refresh: "identity-token".to_string(),
                        access: "LLM|old-key".to_string(),
                        expires: 1,
                        extra: Default::default(),
                    },
                },
                never_aborted_signal(),
            )
            .await
            .expect("credentials");
        assert_eq!(credential.credentials.refresh, "identity-token");
        assert_eq!(credential.credentials.access, "LLM|fresh-key");
        assert_eq!(
            credential.credentials.expires,
            crate::utils::diagnostics::now_ms() + DAY_MS()
        );
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "reports the setup URL when Meta issues no key".
    #[tokio::test]
    async fn reports_the_setup_url_when_meta_issues_no_key() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |_request| {
            Box::pin(async {
                Ok(json_response(
                    serde_json::json!({ "require_payment": true, "action_url": "https://dev.meta.ai/billing" }),
                    200,
                ))
            })
        })));

        let error = meta_oauth()
            .refresh(
                OAuthCredential {
                    credentials: OAuthCredentials {
                        refresh: "identity-token".to_string(),
                        access: String::new(),
                        expires: 1,
                        extra: Default::default(),
                    },
                },
                never_aborted_signal(),
            )
            .await
            .expect_err("no key");
        assert_eq!(
            crate::models::thrown_message(&error),
            "Meta did not issue an API key. Complete setup at https://dev.meta.ai/billing"
        );
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "uses the minted key as the request api key".
    #[tokio::test]
    async fn uses_the_minted_key_as_the_request_api_key() {
        let auth = meta_oauth()
            .to_auth(&OAuthCredential {
                credentials: OAuthCredentials {
                    refresh: "identity-token".to_string(),
                    access: "LLM|key".to_string(),
                    expires: 1,
                    extra: Default::default(),
                },
            })
            .await
            .expect("auth");
        assert_eq!(auth.api_key.as_deref(), Some("LLM|key"));
    }
}
