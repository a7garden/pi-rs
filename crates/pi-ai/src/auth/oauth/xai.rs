//! Port of `packages/ai/src/auth/oauth/xai.ts` — the xAI OAuth device-code
//! flow.
//!
//! The verification URI is forced to https so a malicious response cannot
//! make `open` launch something else. xAI may omit `refresh_token` on refresh
//! when the token was not rotated; the previous refresh token is preserved.

use crate::auth::oauth::device_code::{
    poll_oauth_device_code_flow, OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult,
};
use crate::auth::types::{
    ModelAuth, OAuthAuth, OAuthCredential, OAuthCredentials, OAuthLoginFn, OAuthRefreshFn,
    OAuthToAuthFn, ProviderAuthInteraction,
};
use crate::http::{HttpBody, HttpRequest, HttpResponse};
use crate::utils::abort::AbortSignal;
use crate::utils::thrown::ThrownValue;

const XAI_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const XAI_SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
const XAI_DEVICE_CODE_URL: &str = "https://auth.x.ai/oauth2/device/code";
const XAI_TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
/// Refresh slightly before the reported expiry to avoid using a token that
/// dies mid-request.
const REFRESH_SKEW_MS: i64 = 5 * 60 * 1000;
const DEFAULT_TOKEN_LIFETIME_SECONDS: f64 = 3600.0;

struct XaiDeviceCode {
    device_code: String,
    user_code: String,
    verification_uri: String,
    verification_uri_complete: Option<String>,
    interval_seconds: Option<f64>,
    expires_in_seconds: f64,
}

fn required_string(body: &serde_json::Value, field: &str) -> Result<String, ThrownValue> {
    match body.get(field).and_then(serde_json::Value::as_str) {
        Some(value) if !value.is_empty() => Ok(value.to_string()),
        _ => Err(ThrownValue::message(format!(
            "Invalid xAI OAuth response field: {field}"
        ))),
    }
}

fn positive_number(body: &serde_json::Value, field: &str) -> Result<f64, ThrownValue> {
    match body.get(field).and_then(serde_json::Value::as_f64) {
        Some(value) if value.is_finite() && value > 0.0 => Ok(value),
        _ => Err(ThrownValue::message(format!(
            "Invalid xAI OAuth response field: {field}"
        ))),
    }
}

/// The verification URI is opened in the user's browser; force it to be an
/// https URL (upstream `validateVerificationUri`).
fn validate_verification_uri(raw: &str) -> Result<String, ThrownValue> {
    let url = url::Url::parse(raw)
        .map_err(|_| ThrownValue::message("Untrusted verification URI in xAI OAuth response"))?;
    if url.scheme() != "https" {
        return Err(ThrownValue::message(
            "Untrusted verification URI in xAI OAuth response",
        ));
    }
    Ok(url.to_string())
}

async fn post_form(
    url: &str,
    fields: Vec<(&str, &str)>,
    signal: &AbortSignal,
) -> Result<HttpResponse, ThrownValue> {
    let mut request = HttpRequest::new("POST", url);
    request
        .headers
        .push(("Accept".to_string(), "application/json".to_string()));
    request.headers.push((
        "Content-Type".to_string(),
        "application/x-www-form-urlencoded".to_string(),
    ));
    request.body = HttpBody::bytes(crate::auth::oauth::form_urlencode(&fields).into_bytes());
    request.signal = Some(signal.clone());

    let response = match crate::auth::oauth::oauth_fetch().send(request).await {
        Ok(response) => response,
        Err(error) => {
            if signal.aborted() {
                return Err(ThrownValue::message("Login cancelled"));
            }
            return Err(ThrownValue::message(error.message.clone()));
        }
    };

    let bytes = response.body.collect().await.map_err(|error| {
        if signal.aborted() {
            ThrownValue::message("Login cancelled")
        } else {
            ThrownValue::message(error.message)
        }
    })?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let body: serde_json::Value = match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(body) if body.is_object() && !body.is_array() => body,
        _ => {
            if signal.aborted() {
                return Err(ThrownValue::message("Login cancelled"));
            }
            return Err(ThrownValue::message(format!(
                "xAI OAuth returned invalid JSON (HTTP {})",
                response.status
            )));
        }
    };
    // Pair the status with the parsed body for callers.
    let _ = text;
    Ok(HttpResponse {
        status: response.status,
        headers: response.headers,
        body: HttpBody::bytes(body.to_string().into_bytes()),
    })
}

/// Upstream `requestFailure` — `error: description` detail joining.
fn request_failure(action: &str, body: &serde_json::Value, status: u16) -> ThrownValue {
    let error = body.get("error").and_then(serde_json::Value::as_str);
    let description = body
        .get("error_description")
        .and_then(serde_json::Value::as_str);
    let detail = [error, description]
        .into_iter()
        .flatten()
        .collect::<Vec<&str>>()
        .join(": ");
    ThrownValue::message(format!(
        "xAI OAuth {action} failed (HTTP {status}){}",
        if detail.is_empty() {
            String::new()
        } else {
            format!(": {detail}")
        }
    ))
}

fn parse_device_code(body: &serde_json::Value) -> Result<XaiDeviceCode, ThrownValue> {
    // RFC 8628 allows interval 0 (no minimum wait); fall back to the poller's
    // default instead of failing on non-positive or malformed values.
    let interval_seconds = match body.get("interval").and_then(serde_json::Value::as_f64) {
        Some(interval) if interval.is_finite() && interval > 0.0 => Some(interval),
        _ => None,
    };
    let verification_uri_complete = match body
        .get("verification_uri_complete")
        .and_then(serde_json::Value::as_str)
    {
        Some(raw) if !raw.is_empty() => Some(validate_verification_uri(raw)?),
        _ => None,
    };
    Ok(XaiDeviceCode {
        device_code: required_string(body, "device_code")?,
        user_code: required_string(body, "user_code")?,
        verification_uri: validate_verification_uri(&required_string(body, "verification_uri")?)?,
        verification_uri_complete,
        interval_seconds,
        expires_in_seconds: positive_number(body, "expires_in")?,
    })
}

fn credentials_from_token_response(
    body: &serde_json::Value,
    previous_refresh_token: Option<&str>,
) -> Result<OAuthCredential, ThrownValue> {
    let access = required_string(body, "access_token")?;
    // xAI may omit refresh_token on refresh when the token is not rotated.
    let refresh = if body.get("refresh_token").is_none() && previous_refresh_token.is_some() {
        previous_refresh_token
            .expect("previous refresh token")
            .to_string()
    } else {
        required_string(body, "refresh_token")?
    };
    let expires_in_seconds = match body.get("expires_in") {
        None => DEFAULT_TOKEN_LIFETIME_SECONDS,
        Some(_) => positive_number(body, "expires_in")?,
    };
    Ok(OAuthCredential {
        credentials: OAuthCredentials {
            refresh,
            access,
            expires: crate::utils::diagnostics::now_ms() + (expires_in_seconds * 1000.0) as i64
                - REFRESH_SKEW_MS,
            extra: Default::default(),
        },
    })
}

async fn request_device_code(signal: &AbortSignal) -> Result<XaiDeviceCode, ThrownValue> {
    let response = post_form(
        XAI_DEVICE_CODE_URL,
        vec![
            ("client_id", XAI_CLIENT_ID),
            ("scope", XAI_SCOPE),
            ("referrer", "pi"),
        ],
        signal,
    )
    .await?;
    let body = parse_response_body(response).await?;
    if !body.0 {
        return Err(request_failure("device authorization", &body.1, body.2));
    }
    parse_device_code(&body.1)
}

/// Extract (ok, body, status) from a `post_form` response.
async fn parse_response_body(
    response: HttpResponse,
) -> Result<(bool, serde_json::Value, u16), ThrownValue> {
    let status = response.status;
    let ok = response.ok();
    let bytes = response
        .body
        .collect()
        .await
        .map_err(|error| ThrownValue::message(error.message))?;
    let body: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|_| ThrownValue::message("xAI OAuth returned invalid JSON".to_string()))?;
    Ok((ok, body, status))
}

async fn poll_for_tokens(
    device: &XaiDeviceCode,
    signal: &AbortSignal,
) -> Result<OAuthCredential, ThrownValue> {
    let device_code = device.device_code.clone();
    poll_oauth_device_code_flow(
        OAuthDeviceCodePollOptions {
            interval_seconds: device.interval_seconds,
            expires_in_seconds: Some(device.expires_in_seconds),
            wait_before_first_poll: true,
            signal: signal.clone(),
        },
        move || {
            let device_code = device_code.clone();
            let signal = signal.clone();
            async move {
                let response = post_form(
                    XAI_TOKEN_URL,
                    vec![
                        ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                        ("client_id", XAI_CLIENT_ID),
                        ("device_code", device_code.as_str()),
                    ],
                    &signal,
                )
                .await?;
                let (ok, body, status) = parse_response_body(response).await?;

                if ok {
                    return Ok(OAuthDeviceCodePollResult::Complete {
                        value: credentials_from_token_response(&body, None)?,
                    });
                }

                match body.get("error").and_then(serde_json::Value::as_str) {
                    Some("authorization_pending") => return Ok(OAuthDeviceCodePollResult::Pending),
                    Some("slow_down") => {
                        return Ok(OAuthDeviceCodePollResult::SlowDown {
                            interval_seconds: body
                                .get("interval")
                                .and_then(serde_json::Value::as_f64),
                        });
                    }
                    Some("access_denied") | Some("authorization_denied") => {
                        return Ok(OAuthDeviceCodePollResult::Failed {
                            message: "xAI device authorization was denied".to_string(),
                        });
                    }
                    Some("expired_token") => {
                        return Ok(OAuthDeviceCodePollResult::Failed {
                            message: "xAI device code expired".to_string(),
                        });
                    }
                    _ => {}
                }
                Ok(OAuthDeviceCodePollResult::Failed {
                    message: crate::utils::diagnostics::format_thrown_value(&request_failure(
                        "device token polling",
                        &body,
                        status,
                    )),
                })
            }
        },
    )
    .await
}

async fn login_xai(interaction: &ProviderAuthInteraction) -> Result<OAuthCredential, ThrownValue> {
    let device = request_device_code(&interaction.signal).await?;
    (interaction.notify)(crate::auth::types::AuthEvent::DeviceCode {
        user_code: device.user_code.clone(),
        verification_uri: device
            .verification_uri_complete
            .clone()
            .unwrap_or_else(|| device.verification_uri.clone()),
        interval_seconds: device.interval_seconds.map(|seconds| seconds as u64),
        expires_in_seconds: Some(device.expires_in_seconds as u64),
    });
    poll_for_tokens(&device, &interaction.signal).await
}

async fn refresh_xai_token(
    refresh_token: &str,
    signal: &AbortSignal,
) -> Result<OAuthCredential, ThrownValue> {
    let response = post_form(
        XAI_TOKEN_URL,
        vec![
            ("grant_type", "refresh_token"),
            ("client_id", XAI_CLIENT_ID),
            ("refresh_token", refresh_token),
        ],
        signal,
    )
    .await?;
    let (ok, body, status) = parse_response_body(response).await?;
    if !ok {
        return Err(request_failure("token refresh", &body, status));
    }
    credentials_from_token_response(&body, Some(refresh_token))
}

/// Upstream `export const xaiOAuth`.
pub fn xai_oauth() -> OAuthAuth {
    let login: OAuthLoginFn =
        Box::new(|interaction, _options| Box::pin(async move { login_xai(&interaction).await }));

    let refresh: OAuthRefreshFn = Box::new(|credential, signal| {
        Box::pin(async move { refresh_xai_token(&credential.credentials.refresh, &signal).await })
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
        name: "xAI (Grok/X subscription)".to_string(),
        is_subscription: true,
        login_label: Some("Sign in with SuperGrok or X Premium".to_string()),
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
    use crate::http::{FetchFn, ProviderHttpError};
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

    async fn body_form(body: HttpBody) -> std::collections::BTreeMap<String, String> {
        let bytes = body.collect().await.expect("body");
        crate::auth::oauth::parse_form(&String::from_utf8(bytes).expect("utf8"))
    }

    /// Applies response-field overrides; `null` removes the key (JSON has no
    /// `undefined`, which is what upstream tests model with it).
    fn with_overrides(
        mut base: serde_json::Value,
        overrides: serde_json::Value,
    ) -> serde_json::Value {
        if let (Some(base), Some(overrides)) = (base.as_object_mut(), overrides.as_object()) {
            for (key, value) in overrides {
                if value.is_null() {
                    base.remove(key);
                } else {
                    base.insert(key.clone(), value.clone());
                }
            }
        }
        base
    }

    fn device_code_response(overrides: serde_json::Value) -> serde_json::Value {
        with_overrides(
            serde_json::json!({
                "device_code": "device-code",
                "user_code": "ABCD-1234",
                "verification_uri": "https://accounts.x.ai/oauth2/device",
                "expires_in": 900,
                "interval": 5,
            }),
            overrides,
        )
    }

    fn token_response(overrides: serde_json::Value) -> serde_json::Value {
        with_overrides(
            serde_json::json!({
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 21_600,
                "token_type": "Bearer",
            }),
            overrides,
        )
    }

    async fn login_xai_for_test(
        signal: AbortSignal,
        on_device_code: impl FnMut(AuthEvent) + Send + 'static,
    ) -> Result<OAuthCredential, ThrownValue> {
        let on_device_code = parking_lot::Mutex::new(on_device_code);
        xai_oauth()
            .login(
                ProviderAuthInteraction {
                    signal,
                    prompt: Box::new(|_prompt| {
                        Box::pin(async { Err(ThrownValue::message("Unexpected prompt")) })
                    }),
                    notify: Box::new(move |event| (on_device_code.lock())(event)),
                },
                None,
            )
            .await
    }

    async fn refresh_xai_for_test(refresh_token: &str) -> Result<OAuthCredential, ThrownValue> {
        xai_oauth()
            .refresh(
                OAuthCredential {
                    credentials: OAuthCredentials {
                        refresh: refresh_token.to_string(),
                        access: "old-access".to_string(),
                        expires: 0,
                        extra: Default::default(),
                    },
                },
                never_aborted_signal(),
            )
            .await
    }

    // Port of test/xai-oauth.test.ts "uses the device grant, delays polling, and
    // handles pending and slow_down".
    #[tokio::test(start_paused = true)]
    async fn uses_the_device_grant_and_handles_pending_and_slow_down() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        let poll_times: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let replies: Arc<Mutex<Vec<HttpResponse>>> = Arc::new(Mutex::new(vec![
            json_response(serde_json::json!({ "error": "authorization_pending" }), 400),
            json_response(
                serde_json::json!({ "error": "slow_down", "interval": 10 }),
                400,
            ),
            json_response(token_response(serde_json::Value::Null), 200),
        ]));
        let poll_times_for_fetch = poll_times.clone();
        let replies_for_fetch = replies.clone();
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |mut request| {
            let url = request.url.clone();
            let poll_times_for_fetch = poll_times_for_fetch.clone();
            let replies_for_fetch = replies_for_fetch.clone();
            Box::pin(async move {
                if url == XAI_DEVICE_CODE_URL {
                    let form =
                        body_form(std::mem::replace(&mut request.body, HttpBody::Empty)).await;
                    assert_eq!(
                        form.get("client_id").map(String::as_str),
                        Some(XAI_CLIENT_ID)
                    );
                    assert_eq!(
                        form.get("scope").map(String::as_str),
                        Some("openid profile email offline_access grok-cli:access api:access")
                    );
                    assert_eq!(form.get("referrer").map(String::as_str), Some("pi"));
                    return Ok(json_response(
                        device_code_response(serde_json::Value::Null),
                        200,
                    ));
                }
                if url == XAI_TOKEN_URL {
                    poll_times_for_fetch
                        .lock()
                        .push(crate::utils::diagnostics::now_ms());
                    let form =
                        body_form(std::mem::replace(&mut request.body, HttpBody::Empty)).await;
                    assert_eq!(
                        form.get("grant_type").map(String::as_str),
                        Some("urn:ietf:params:oauth:grant-type:device_code")
                    );
                    assert_eq!(
                        form.get("client_id").map(String::as_str),
                        Some(XAI_CLIENT_ID)
                    );
                    assert_eq!(
                        form.get("device_code").map(String::as_str),
                        Some("device-code")
                    );
                    return Ok(replies_for_fetch.lock().remove(0));
                }
                Err(ProviderHttpError::new(format!("Unexpected request: {url}")))
            })
        })));

        let credential = tokio::spawn(login_xai_for_test(never_aborted_signal(), |event| {
            if let AuthEvent::DeviceCode {
                user_code,
                verification_uri,
                interval_seconds,
                expires_in_seconds,
            } = event
            {
                assert_eq!(user_code, "ABCD-1234");
                assert_eq!(verification_uri, "https://accounts.x.ai/oauth2/device");
                assert_eq!(interval_seconds, Some(5));
                assert_eq!(expires_in_seconds, Some(900));
            }
        }));

        // waitBeforeFirstPoll: nothing before 5s.
        tokio::time::advance(std::time::Duration::from_millis(0)).await;
        assert_eq!(poll_times.lock().len(), 0);
        tokio::time::advance(std::time::Duration::from_secs(5)).await;
        // Let the woken login task run its poll before asserting.
        tokio::task::yield_now().await;
        assert_eq!(poll_times.lock().len(), 1);
        // slow_down raised the interval to 10 seconds.
        tokio::time::advance(std::time::Duration::from_secs(5)).await;
        tokio::task::yield_now().await;
        assert_eq!(poll_times.lock().len(), 2);
        tokio::time::advance(std::time::Duration::from_secs(10)).await;

        let credential = credential.await.expect("task").expect("credentials");
        assert_eq!(credential.credentials.access, "access-token");
        assert_eq!(credential.credentials.refresh, "refresh-token");
        assert_eq!(poll_times.lock().len(), 3);
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "falls back to the default poll interval when the response reports
    // interval 0".
    #[tokio::test(start_paused = true)]
    async fn falls_back_to_the_default_poll_interval_on_zero() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        let poll_times: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let poll_times_for_fetch = poll_times.clone();
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |request| {
            let url = request.url.clone();
            let poll_times_for_fetch = poll_times_for_fetch.clone();
            Box::pin(async move {
                if url == XAI_DEVICE_CODE_URL {
                    return Ok(json_response(
                        device_code_response(serde_json::json!({ "interval": 0 })),
                        200,
                    ));
                }
                poll_times_for_fetch
                    .lock()
                    .push(crate::utils::diagnostics::now_ms());
                Ok(json_response(token_response(serde_json::Value::Null), 200))
            })
        })));

        let credential = tokio::spawn(login_xai_for_test(never_aborted_signal(), |_event| {}));
        // RFC 8628 default interval is 5 seconds when the server does not require
        // a wait.
        tokio::time::advance(std::time::Duration::from_secs(5)).await;
        credential.await.expect("task").expect("credentials");
        assert_eq!(poll_times.lock().len(), 1);
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "prefers verification_uri_complete when the server provides it".
    #[tokio::test(start_paused = true)]
    async fn prefers_verification_uri_complete() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |request| {
            let url = request.url.clone();
            Box::pin(async move {
                if url == XAI_DEVICE_CODE_URL {
                    return Ok(json_response(
                        device_code_response(serde_json::json!({
                            "verification_uri_complete": "https://accounts.x.ai/oauth2/device?user_code=ABCD-1234"
                        })),
                        200,
                    ));
                }
                Ok(json_response(token_response(serde_json::Value::Null), 200))
            })
        })));

        let seen: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let seen_for_notify = seen.clone();
        let credential = tokio::spawn(login_xai_for_test(never_aborted_signal(), move |event| {
            if let AuthEvent::DeviceCode {
                verification_uri, ..
            } = event
            {
                *seen_for_notify.lock() = Some(verification_uri);
            }
        }));
        tokio::time::advance(std::time::Duration::from_secs(5)).await;
        credential.await.expect("task").expect("credentials");
        assert_eq!(
            seen.lock().as_deref(),
            Some("https://accounts.x.ai/oauth2/device?user_code=ABCD-1234")
        );
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "rejects a non-https verification_uri_complete" and the
    // verification_uri table case.
    #[tokio::test]
    async fn rejects_non_https_verification_uris() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        for raw in [
            "$(id>/tmp/pwned)",
            "http://accounts.x.ai/oauth2/device",
            "file:///etc/passwd",
            "not a url",
        ] {
            crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |_request| {
                let raw = raw.to_string();
                Box::pin(async move {
                    Ok(json_response(
                        device_code_response(serde_json::json!({ "verification_uri": raw })),
                        200,
                    ))
                })
            })));
            let error = login_xai_for_test(never_aborted_signal(), |_event| {})
                .await
                .expect_err("untrusted uri");
            assert!(
                crate::models::thrown_message(&error).contains("Untrusted verification URI"),
                "got: {}",
                crate::models::thrown_message(&error)
            );
        }
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "fails when device authorization is denied" (both error spellings).
    #[tokio::test(start_paused = true)]
    async fn fails_when_device_authorization_is_denied() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        for error_code in ["access_denied", "authorization_denied"] {
            let request_count = Arc::new(Mutex::new(0));
            let request_count_for_fetch = request_count.clone();
            let error_code_for_fetch = error_code.to_string();
            crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |request| {
                let url = request.url.clone();
                let count = request_count_for_fetch.clone();
                let error_code = error_code_for_fetch.clone();
                Box::pin(async move {
                    let mut current = count.lock();
                    *current += 1;
                    if *current == 1 {
                        drop(current);
                        return Ok(json_response(
                            device_code_response(serde_json::json!({ "interval": 1 })),
                            200,
                        ));
                    }
                    drop(current);
                    if url != XAI_TOKEN_URL {
                        return Err(ProviderHttpError::new(format!("Unexpected request: {url}")));
                    }
                    Ok(json_response(
                        serde_json::json!({ "error": error_code }),
                        400,
                    ))
                })
            })));

            let credential = tokio::spawn(login_xai_for_test(never_aborted_signal(), |_event| {}));
            tokio::time::advance(std::time::Duration::from_millis(0)).await;
            tokio::time::advance(std::time::Duration::from_secs(1)).await;
            let error = credential.await.expect("task").expect_err("denied");
            assert_eq!(
                crate::models::thrown_message(&error),
                "xAI device authorization was denied"
            );
            crate::auth::oauth::set_oauth_fetch(None);
        }
    }

    // Port of "cancels while waiting for the first token poll".
    #[tokio::test(start_paused = true)]
    async fn cancels_while_waiting_for_the_first_token_poll() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        let controller = AbortController::new();
        let fetch_count = Arc::new(Mutex::new(0));
        let fetch_count_for_stub = fetch_count.clone();
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |request| {
            let url = request.url.clone();
            let count = fetch_count_for_stub.clone();
            Box::pin(async move {
                if url == XAI_DEVICE_CODE_URL {
                    *count.lock() += 1;
                    return Ok(json_response(
                        device_code_response(serde_json::Value::Null),
                        200,
                    ));
                }
                Err(ProviderHttpError::new(format!("Unexpected request: {url}")))
            })
        })));

        let signal = controller.signal();
        let controller_for_abort = controller.clone();
        let credential = tokio::spawn(login_xai_for_test(signal.clone(), move |_event| {
            controller_for_abort.abort();
        }));

        tokio::time::advance(std::time::Duration::from_millis(0)).await;
        tokio::time::advance(std::time::Duration::from_secs(5)).await;
        let error = credential.await.expect("task").expect_err("cancelled");
        assert_eq!(crate::models::thrown_message(&error), "Login cancelled");
        assert_eq!(*fetch_count.lock(), 1);
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "refreshes tokens and preserves an unrotated refresh token".
    #[tokio::test]
    async fn refreshes_tokens_and_preserves_an_unrotated_refresh_token() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        let request_count = Arc::new(Mutex::new(0));
        let request_count_for_stub = request_count.clone();
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |mut request| {
            let url = request.url.clone();
            let count = request_count_for_stub.clone();
            Box::pin(async move {
                assert_eq!(url, XAI_TOKEN_URL);
                let form = body_form(std::mem::replace(&mut request.body, HttpBody::Empty)).await;
                assert_eq!(
                    form.get("grant_type").map(String::as_str),
                    Some("refresh_token")
                );
                assert_eq!(
                    form.get("client_id").map(String::as_str),
                    Some(XAI_CLIENT_ID)
                );
                let mut current = count.lock();
                *current += 1;
                if *current == 1 {
                    assert_eq!(
                        form.get("refresh_token").map(String::as_str),
                        Some("old-refresh")
                    );
                    drop(current);
                    return Ok(json_response(
                        token_response(serde_json::json!({
                            "access_token": "new-access",
                            "refresh_token": "new-refresh",
                        })),
                        200,
                    ));
                }
                assert_eq!(
                    form.get("refresh_token").map(String::as_str),
                    Some("keep-refresh")
                );
                drop(current);
                Ok(json_response(
                    token_response(serde_json::json!({
                        "access_token": "newer-access",
                        "refresh_token": serde_json::Value::Null,
                    })),
                    200,
                ))
            })
        })));

        let rotated = refresh_xai_for_test("old-refresh").await.expect("rotated");
        let preserved = refresh_xai_for_test("keep-refresh")
            .await
            .expect("preserved");
        assert_eq!(rotated.credentials.refresh, "new-refresh");
        assert_eq!(rotated.credentials.access, "new-access");
        assert_eq!(preserved.credentials.refresh, "keep-refresh");
        assert_eq!(preserved.credentials.access, "newer-access");
        assert_eq!(xai_oauth().name, "xAI (Grok/X subscription)");
        let auth = xai_oauth().to_auth(&preserved).await.expect("auth");
        assert_eq!(auth.api_key.as_deref(), Some("newer-access"));
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "assumes a one-hour lifetime when expires_in is missing".
    #[tokio::test]
    async fn assumes_a_one_hour_lifetime_when_expires_in_is_missing() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |_request| {
            Box::pin(async {
                let mut body = token_response(serde_json::Value::Null);
                if let Some(object) = body.as_object_mut() {
                    object.remove("expires_in");
                }
                Ok(json_response(body, 200))
            })
        })));

        let credentials = refresh_xai_for_test("old-refresh")
            .await
            .expect("credentials");
        assert_eq!(
            credentials.credentials.expires,
            crate::utils::diagnostics::now_ms() + 3_600_000 - 300_000
        );
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "rejects token responses with missing fields".
    #[tokio::test]
    async fn rejects_token_responses_with_missing_fields() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |_request| {
            Box::pin(async {
                let mut body = token_response(serde_json::Value::Null);
                if let Some(object) = body.as_object_mut() {
                    object.remove("access_token");
                }
                Ok(json_response(body, 200))
            })
        })));

        let error = refresh_xai_for_test("old-refresh")
            .await
            .expect_err("invalid");
        assert_eq!(
            crate::models::thrown_message(&error),
            "Invalid xAI OAuth response field: access_token"
        );
        crate::auth::oauth::set_oauth_fetch(None);
    }

    // Port of "surfaces the upstream error code and description on refresh
    // failure".
    #[tokio::test]
    async fn surfaces_the_upstream_error_code_and_description_on_refresh_failure() {
        let _guard = crate::auth::oauth::test_fetch_guard().lock().await;
        crate::auth::oauth::set_oauth_fetch(Some(stub_fetch(move |_request| {
            Box::pin(async {
                Ok(json_response(
                    serde_json::json!({ "error": "invalid_grant", "error_description": "refresh token revoked" }),
                    400,
                ))
            })
        })));

        let error = refresh_xai_for_test("old-refresh")
            .await
            .expect_err("failed");
        assert_eq!(
            crate::models::thrown_message(&error),
            "xAI OAuth token refresh failed (HTTP 400): invalid_grant: refresh token revoked"
        );
        crate::auth::oauth::set_oauth_fetch(None);
    }
}
