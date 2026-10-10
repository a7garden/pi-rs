//! Port of `packages/ai/src/auth/oauth/github-copilot.ts` — the GitHub
//! Copilot OAuth flow.
//!
//! Device flow against github.com (or a GHE domain), a Copilot internal token
//! exchange, catalog fetch with 429 retry-after handling, and best-effort
//! policy enablement for unconfigured account models.
//!
//! # Deviations
//!
//! - D32: `GITHUB_COPILOT_MODELS` (the generated Copilot catalog, sibling
//!   module `providers/github_copilot_models.rs`) is consulted through
//!   [`is_known_copilot_model`], which carries the same 34 model ids derived
//!   from `providers/data/github-copilot.json`. Final integration should swap
//!   the body to read the sibling catalog when it lands (isolated function —
//!   the only coupling point).

use crate::auth::oauth::device_code::{
    poll_oauth_device_code_flow, OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult,
};
use crate::auth::types::{
    ModelAuth, OAuthAuth, OAuthCredential, OAuthCredentials, OAuthLoginFn, OAuthRefreshFn,
    OAuthToAuthFn, ProviderAuthInteraction,
};
use crate::http::{HttpBody, HttpRequest, HttpResponse};
use crate::types::ProviderHeaders;
use crate::utils::abort::AbortSignal;
use crate::utils::sleep::sleep;
use crate::utils::thrown::ThrownValue;

/// Upstream `atob("SXYxLmI1MDdhMDhjODdlY2ZlOTg=")`.
const CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";

const COPILOT_HEADERS: &[(&str, &str)] = &[
    ("User-Agent", "GitHubCopilotChat/0.35.0"),
    ("Editor-Version", "vscode/1.107.0"),
    ("Editor-Plugin-Version", "copilot-chat/0.35.0"),
    ("Copilot-Integration-Id", "vscode-chat"),
];
const COPILOT_API_VERSION: &str = "2026-06-01";

/// Final-integration coupling point (deviation D32): swap to the sibling
/// `GITHUB_COPILOT_MODELS` catalog check (upstream filters against the
/// generated catalog); the ids come from the sibling generated wrapper.
fn is_known_copilot_model(id: &str) -> bool {
    crate::providers::github_copilot_models::github_copilot_models()
        .iter()
        .any(|model| model.base.id == id)
}

fn copilot_headers() -> Vec<(String, String)> {
    COPILOT_HEADERS
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect()
}

fn normalize_domain(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    let candidate = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("https://{trimmed}")
    };
    url::Url::parse(&candidate)
        .ok()?
        .host_str()
        .map(str::to_string)
}

fn get_urls(domain: &str) -> (String, String, String) {
    (
        format!("https://{domain}/login/device/code"),
        format!("https://{domain}/login/oauth/access_token"),
        format!("https://api.{domain}/copilot_internal/v2/token"),
    )
}

/// Parse the proxy-ep from a Copilot token and convert to API base URL
/// (`tid=...;exp=...;proxy-ep=proxy.individual.githubcopilot.com;...`).
fn get_base_url_from_token(token: &str) -> Option<String> {
    let start = token.find("proxy-ep=")? + "proxy-ep=".len();
    let rest = &token[start..];
    let end = rest.find(';').unwrap_or(rest.len());
    let proxy_host = &rest[..end];
    let api_host = proxy_host.strip_prefix("proxy.").unwrap_or(proxy_host);
    Some(format!("https://{api_host}"))
}

fn get_github_copilot_base_url(token: Option<&str>, enterprise_domain: Option<&str>) -> String {
    // If we have a token, extract the base URL from proxy-ep
    if let Some(token) = token {
        if let Some(url) = get_base_url_from_token(token) {
            return url;
        }
    }
    // Fallback for enterprise or if token parsing fails
    if let Some(enterprise_domain) = enterprise_domain {
        return format!("https://copilot-api.{enterprise_domain}");
    }
    "https://api.individual.githubcopilot.com".to_string()
}

struct AccountModel {
    id: String,
    picker_enabled: bool,
    policy_state: Option<String>,
}

/// Upstream `parseGitHubCopilotModelCatalog` → `(availableModelIds,
/// policyModelIds)`.
fn parse_github_copilot_model_catalog(
    raw: &serde_json::Value,
    allow_policy_fallback: bool,
) -> Result<(Vec<String>, Vec<String>), ThrownValue> {
    let data = raw.get("data").and_then(serde_json::Value::as_array);
    let Some(data) = data else {
        return Err(ThrownValue::message("Invalid Copilot models response"));
    };

    let mut account_models: Vec<AccountModel> = Vec::new();
    for raw_item in data {
        let Some(id) = raw_item.get("id").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let supports = raw_item
            .get("capabilities")
            .and_then(|capabilities| capabilities.get("supports"))
            .and_then(|supports| supports.get("tool_calls"));
        if supports == Some(&serde_json::Value::Bool(false)) {
            continue;
        }
        account_models.push(AccountModel {
            id: id.to_string(),
            picker_enabled: raw_item
                .get("model_picker_enabled")
                .and_then(serde_json::Value::as_bool)
                == Some(true),
            policy_state: raw_item
                .get("policy")
                .and_then(|policy| policy.get("state"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
        });
    }
    let picker_model_ids: Vec<String> = account_models
        .iter()
        .filter(|model| model.picker_enabled && model.policy_state.as_deref() != Some("disabled"))
        .map(|model| model.id.clone())
        .collect();
    let use_policy_fallback = allow_policy_fallback && picker_model_ids.is_empty();
    let available_model_ids: Vec<String> = if !picker_model_ids.is_empty() || !allow_policy_fallback
    {
        picker_model_ids
    } else {
        account_models
            .iter()
            .filter(|model| model.policy_state.as_deref() == Some("enabled"))
            .map(|model| model.id.clone())
            .collect()
    };
    let policy_model_ids: Vec<String> = account_models
        .iter()
        .filter(|model| {
            model.policy_state.as_deref() == Some("unconfigured")
                && is_known_copilot_model(&model.id)
                && (model.picker_enabled || use_policy_fallback)
        })
        .map(|model| model.id.clone())
        .collect();
    Ok((available_model_ids, policy_model_ids))
}

struct RetryPolicy {
    max_retries: u32,
    max_elapsed_ms: u64,
}

async fn response_text(response: &mut HttpResponse) -> String {
    let body = std::mem::replace(&mut response.body, HttpBody::Empty);
    match body.collect().await {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(_) => String::new(),
    }
}

/// Upstream `fetchWithRateLimitRetry` — shared 5s per-request timeout,
/// exponential backoff honoring `retry-after`. Takes a request factory because
/// retries re-send the same init upstream.
async fn fetch_with_rate_limit_retry(
    make_request: impl Fn() -> HttpRequest,
    signal: &AbortSignal,
    retry_policy: RetryPolicy,
) -> Result<HttpResponse, ThrownValue> {
    let budget = if retry_policy.max_retries > 0 && retry_policy.max_elapsed_ms > 0 {
        Some(crate::auth::oauth::timeout_signal(
            &crate::utils::abort::AbortController::new().signal(),
            retry_policy.max_elapsed_ms,
        ))
    } else {
        None
    };
    let request_signal = match &budget {
        Some(budget) => AbortSignal::any([signal.clone(), budget.clone()]),
        None => signal.clone(),
    };
    let retry_deadline: Option<i64> = budget
        .as_ref()
        .map(|_| crate::utils::diagnostics::now_ms() + retry_policy.max_elapsed_ms as i64);

    let mut retry: u32 = 0;
    loop {
        let mut attempt_request = make_request();
        attempt_request.signal = Some(crate::auth::oauth::timeout_signal(&request_signal, 5000));
        let response = crate::auth::oauth::oauth_fetch()
            .send(attempt_request)
            .await
            .map_err(|error| {
                if signal.aborted() {
                    ThrownValue::message("Login cancelled")
                } else {
                    ThrownValue::message(error.message.clone())
                }
            })?;

        if response.status != 429 || retry == retry_policy.max_retries {
            return Ok(response);
        }

        let retry_after = response.header("retry-after").map(str::to_string);
        let mut delay_ms = 500u64 * 2u64.pow(retry);
        if let Some(retry_after) = retry_after {
            let seconds = retry_after.trim().parse::<f64>().ok();
            delay_ms = match seconds {
                Some(seconds) => (seconds * 1000.0) as u64,
                // HTTP-date form (upstream `Date.parse`); unparseable → return the
                // response like upstream's `Number.isFinite` guard.
                None => return Ok(response),
            };
        }
        if let Some(retry_deadline) = retry_deadline {
            if delay_ms as i64 >= retry_deadline - crate::utils::diagnostics::now_ms() {
                return Ok(response);
            }
        }
        sleep(delay_ms, &request_signal)
            .await
            .map_err(|_| ThrownValue::message("Login cancelled"))?;
        retry += 1;
    }
}

async fn fetch_json(url: &str, request: HttpRequest) -> Result<serde_json::Value, ThrownValue> {
    let mut response = crate::auth::oauth::oauth_fetch()
        .send(request)
        .await
        .map_err(|error| ThrownValue::message(error.message.clone()))?;
    let status = response.status;
    if !(200..300).contains(&status) {
        let text = response_text(&mut response).await;
        return Err(ThrownValue::message(format!(
            "{status} {}: {text}",
            crate::auth::oauth::http_status_text(status)
        )));
    }
    let bytes = response
        .body
        .collect()
        .await
        .map_err(|error| ThrownValue::message(error.message))?;
    serde_json::from_slice(&bytes).map_err(|error| ThrownValue::message(error.to_string()))
}

fn form_request(method: &str, url: &str, body: Vec<(&str, &str)>) -> HttpRequest {
    let mut request = HttpRequest::new(method, url);
    request
        .headers
        .push(("Accept".to_string(), "application/json".to_string()));
    request.headers.push((
        "Content-Type".to_string(),
        "application/x-www-form-urlencoded".to_string(),
    ));
    request.headers.push((
        "User-Agent".to_string(),
        "GitHubCopilotChat/0.35.0".to_string(),
    ));
    request.body = HttpBody::bytes(crate::auth::oauth::form_urlencode(&body).into_bytes());
    request
}

struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    interval: Option<f64>,
    expires_in: f64,
}

async fn start_device_flow(
    domain: &str,
    signal: &AbortSignal,
) -> Result<DeviceCodeResponse, ThrownValue> {
    let (device_code_url, _, _) = get_urls(domain);
    let mut request = form_request(
        "POST",
        &device_code_url,
        vec![("client_id", CLIENT_ID), ("scope", "read:user")],
    );
    request.signal = Some(signal.clone());
    let data = fetch_json(&device_code_url, request).await?;

    if !data.is_object() {
        return Err(ThrownValue::message("Invalid device code response"));
    }
    let device_code = data.get("device_code").and_then(serde_json::Value::as_str);
    let user_code = data.get("user_code").and_then(serde_json::Value::as_str);
    let verification_uri = data
        .get("verification_uri")
        .and_then(serde_json::Value::as_str);
    let interval = data.get("interval").and_then(serde_json::Value::as_f64);
    let expires_in = data.get("expires_in").and_then(serde_json::Value::as_f64);
    let (Some(device_code), Some(user_code), Some(verification_uri), Some(expires_in)) =
        (device_code, user_code, verification_uri, expires_in)
    else {
        return Err(ThrownValue::message("Invalid device code response fields"));
    };
    if interval.is_some_and(|value| !value.is_finite()) {
        return Err(ThrownValue::message("Invalid device code response fields"));
    }

    // The verification URI is opened in the user's browser and to prevent `open`
    // from opening an executable or similar, we force it to be a URL.
    let parsed_uri = url::Url::parse(verification_uri)
        .map_err(|_| ThrownValue::message("Untrusted verification_uri in device code response"))?;
    if parsed_uri.scheme() != "https" && parsed_uri.scheme() != "http" {
        return Err(ThrownValue::message(
            "Untrusted verification_uri in device code response",
        ));
    }

    Ok(DeviceCodeResponse {
        device_code: device_code.to_string(),
        user_code: user_code.to_string(),
        verification_uri: parsed_uri.to_string(),
        interval,
        expires_in,
    })
}

async fn poll_for_github_access_token(
    domain: &str,
    device: &DeviceCodeResponse,
    signal: &AbortSignal,
) -> Result<String, ThrownValue> {
    let (_, access_token_url, _) = get_urls(domain);
    let device_code = device.device_code.clone();
    let access_token_url = access_token_url.clone();
    poll_oauth_device_code_flow(
        OAuthDeviceCodePollOptions {
            interval_seconds: device.interval,
            expires_in_seconds: Some(device.expires_in),
            wait_before_first_poll: true,
            signal: signal.clone(),
        },
        move || {
            let device_code = device_code.clone();
            let access_token_url = access_token_url.clone();
            let signal = signal.clone();
            async move {
                let mut request = form_request(
                    "POST",
                    &access_token_url,
                    vec![
                        ("client_id", CLIENT_ID),
                        ("device_code", device_code.as_str()),
                        ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ],
                );
                request.signal = Some(signal.clone());
                let raw = fetch_json(&access_token_url, request).await?;

                if let Some(access_token) =
                    raw.get("access_token").and_then(serde_json::Value::as_str)
                {
                    return Ok(OAuthDeviceCodePollResult::<String>::Complete {
                        value: access_token.to_string(),
                    });
                }

                let error = raw.get("error").and_then(serde_json::Value::as_str);
                if let Some(error) = error {
                    let description = raw
                        .get("error_description")
                        .and_then(serde_json::Value::as_str);
                    match error {
                        "authorization_pending" => return Ok(OAuthDeviceCodePollResult::Pending),
                        "slow_down" => {
                            return Ok(OAuthDeviceCodePollResult::SlowDown {
                                interval_seconds: raw
                                    .get("interval")
                                    .and_then(serde_json::Value::as_f64),
                            });
                        }
                        _ => {}
                    }
                    let description_suffix = description
                        .map(|description| format!(": {description}"))
                        .unwrap_or_default();
                    return Ok(OAuthDeviceCodePollResult::Failed {
                        message: format!("Device flow failed: {error}{description_suffix}"),
                    });
                }

                Ok(OAuthDeviceCodePollResult::Failed {
                    message: "Invalid device token response".to_string(),
                })
            }
        },
    )
    .await
}

fn credential_with(
    refresh: String,
    access: String,
    expires: i64,
    enterprise_domain: Option<&str>,
    available_model_ids: Option<Vec<String>>,
) -> OAuthCredential {
    let mut extra = std::collections::BTreeMap::new();
    if let Some(enterprise_domain) = enterprise_domain {
        extra.insert(
            "enterpriseUrl".to_string(),
            serde_json::Value::String(enterprise_domain.to_string()),
        );
    }
    if let Some(available_model_ids) = available_model_ids {
        extra.insert(
            "availableModelIds".to_string(),
            serde_json::Value::Array(
                available_model_ids
                    .into_iter()
                    .map(serde_json::Value::String)
                    .collect(),
            ),
        );
    }
    OAuthCredential {
        credentials: OAuthCredentials {
            refresh,
            access,
            expires,
            extra,
        },
    }
}

async fn refresh_github_copilot_access_token(
    refresh_token: &str,
    enterprise_domain: Option<&str>,
    signal: &AbortSignal,
) -> Result<OAuthCredential, ThrownValue> {
    let domain = enterprise_domain.unwrap_or("github.com");
    let (_, _, copilot_token_url) = get_urls(domain);

    let mut request = HttpRequest::new("GET", &copilot_token_url);
    request
        .headers
        .push(("Accept".to_string(), "application/json".to_string()));
    request.headers.push((
        "Authorization".to_string(),
        format!("Bearer {refresh_token}"),
    ));
    for (name, value) in copilot_headers() {
        request.headers.push((name, value));
    }
    request.signal = Some(signal.clone());

    let response = crate::auth::oauth::oauth_fetch()
        .send(request)
        .await
        .map_err(|error| ThrownValue::message(error.message.clone()))?;
    let status = response.status;
    if !(200..300).contains(&status) {
        let mut response = response;
        let text = response_text(&mut response).await;
        return Err(ThrownValue::message(format!(
            "{status} {}: {text}",
            crate::auth::oauth::http_status_text(status)
        )));
    }
    let bytes = response
        .body
        .collect()
        .await
        .map_err(|error| ThrownValue::message(error.message))?;
    let raw: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|_| ThrownValue::message("Invalid Copilot token response"))?;

    if !raw.is_object() {
        return Err(ThrownValue::message("Invalid Copilot token response"));
    }

    let token = raw.get("token").and_then(serde_json::Value::as_str);
    let expires_at = raw.get("expires_at").and_then(serde_json::Value::as_f64);
    let (Some(token), Some(expires_at)) = (token, expires_at) else {
        return Err(ThrownValue::message(
            "Invalid Copilot token response fields",
        ));
    };

    Ok(credential_with(
        refresh_token.to_string(),
        token.to_string(),
        (expires_at * 1000.0) as i64 - 5 * 60 * 1000,
        enterprise_domain,
        None,
    ))
}

async fn fetch_github_copilot_models(
    copilot_token: &str,
    enterprise_domain: Option<&str>,
    signal: &AbortSignal,
    retry_policy: RetryPolicy,
) -> Result<(Vec<String>, Vec<String>), ThrownValue> {
    let base_url = get_github_copilot_base_url(Some(copilot_token), enterprise_domain);
    // Some Individual accounts return false for every picker flag despite
    // explicit enabled policies. Limit the fallback to that endpoint so other
    // account types keep strict picker semantics.
    let allow_policy_fallback = base_url == "https://api.individual.githubcopilot.com";
    let make_request = || {
        let mut request = HttpRequest::new("GET", &format!("{base_url}/models"));
        request
            .headers
            .push(("Accept".to_string(), "application/json".to_string()));
        request.headers.push((
            "Authorization".to_string(),
            format!("Bearer {copilot_token}"),
        ));
        for (name, value) in copilot_headers() {
            request.headers.push((name, value));
        }
        request.headers.push((
            "X-GitHub-Api-Version".to_string(),
            COPILOT_API_VERSION.to_string(),
        ));
        request
    };

    let mut response = fetch_with_rate_limit_retry(make_request, signal, retry_policy).await?;
    let status = response.status;
    if !(200..300).contains(&status) {
        let text = response_text(&mut response).await;
        return Err(ThrownValue::message(format!(
            "{status} {}: {text}",
            crate::auth::oauth::http_status_text(status)
        )));
    }
    let bytes = response
        .body
        .collect()
        .await
        .map_err(|error| ThrownValue::message(error.message))?;
    let raw: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| ThrownValue::message(error.to_string()))?;
    parse_github_copilot_model_catalog(&raw, allow_policy_fallback)
}

/// Enable a model for the user's GitHub Copilot account. This is required for
/// some models (like Claude, Grok) before they can be used.
async fn enable_github_copilot_model(
    token: &str,
    model_id: &str,
    enterprise_domain: Option<&str>,
    signal: &AbortSignal,
) -> Result<bool, ThrownValue> {
    let base_url = get_github_copilot_base_url(Some(token), enterprise_domain);
    let url = format!("{base_url}/models/{model_id}/policy");

    let make_request = || {
        let mut request = HttpRequest::new("POST", &url);
        request
            .headers
            .push(("Content-Type".to_string(), "application/json".to_string()));
        request
            .headers
            .push(("Authorization".to_string(), format!("Bearer {token}")));
        for (name, value) in copilot_headers() {
            request.headers.push((name, value));
        }
        request
            .headers
            .push(("openai-intent".to_string(), "chat-policy".to_string()));
        request
            .headers
            .push(("x-interaction-type".to_string(), "chat-policy".to_string()));
        request.body = HttpBody::bytes(
            serde_json::json!({ "state": "enabled" })
                .to_string()
                .into_bytes(),
        );
        request
    };

    let response = match fetch_with_rate_limit_retry(
        make_request,
        signal,
        RetryPolicy {
            max_retries: 2,
            max_elapsed_ms: 5000,
        },
    )
    .await
    {
        Ok(response) => response,
        Err(error) => {
            if signal.aborted() {
                return Err(error);
            }
            return Ok(false);
        }
    };
    if response.status == 429 {
        let mut response = response;
        let text = response_text(&mut response).await;
        return Err(ThrownValue::message(format!(
            "{} {}: {text}",
            response.status,
            crate::auth::oauth::http_status_text(response.status)
        )));
    }
    Ok(response.ok())
}

/// Enable the requested GitHub Copilot models and return the successful IDs.
/// Policy updates are best effort; exhausted rate limiting stops the batch.
async fn enable_github_copilot_models(
    token: &str,
    model_ids: &[String],
    enterprise_domain: Option<&str>,
    signal: &AbortSignal,
) -> Result<Vec<String>, ThrownValue> {
    let mut enabled_model_ids = Vec::new();
    for model_id in model_ids {
        match enable_github_copilot_model(token, model_id, enterprise_domain, signal).await {
            Ok(true) => enabled_model_ids.push(model_id.clone()),
            Ok(false) => {}
            Err(error) => {
                if signal.aborted() {
                    return Err(error);
                }
                break;
            }
        }
    }
    Ok(enabled_model_ids)
}

fn copilot_enterprise_domain(credential: &OAuthCredential) -> Option<String> {
    let enterprise_url = credential
        .credentials
        .extra
        .get("enterpriseUrl")
        .and_then(serde_json::Value::as_str)?;
    if enterprise_url.is_empty() {
        return None;
    }
    normalize_domain(enterprise_url)
}

async fn login_github_copilot(
    interaction: &ProviderAuthInteraction,
) -> Result<OAuthCredential, ThrownValue> {
    let input = (interaction.prompt)(crate::auth::types::AuthPrompt {
        signal: Some(interaction.signal.clone()),
        kind: crate::auth::types::AuthPromptKind::Text {
            message: "GitHub Enterprise URL/domain (blank for github.com)".to_string(),
            placeholder: Some("company.ghe.com".to_string()),
        },
    })
    .await?;
    if interaction.signal.aborted() {
        return Err(ThrownValue::message("Login cancelled"));
    }

    let trimmed = input.trim();
    let enterprise_domain = normalize_domain(&input);
    if !trimmed.is_empty() && enterprise_domain.is_none() {
        return Err(ThrownValue::message("Invalid GitHub Enterprise URL/domain"));
    }
    let enterprise_domain = enterprise_domain.clone();
    let domain = enterprise_domain
        .clone()
        .unwrap_or_else(|| "github.com".to_string());

    let device = start_device_flow(&domain, &interaction.signal).await?;
    (interaction.notify)(crate::auth::types::AuthEvent::DeviceCode {
        user_code: device.user_code.clone(),
        verification_uri: device.verification_uri.clone(),
        interval_seconds: device.interval.map(|interval| interval as u64),
        expires_in_seconds: Some(device.expires_in as u64),
    });

    let github_access_token =
        poll_for_github_access_token(&domain, &device, &interaction.signal).await?;
    let credentials = refresh_github_copilot_access_token(
        &github_access_token,
        enterprise_domain.as_deref(),
        &interaction.signal,
    )
    .await?;
    let models = fetch_github_copilot_models(
        &credentials.credentials.access,
        enterprise_domain.as_deref(),
        &interaction.signal,
        RetryPolicy {
            max_retries: 2,
            max_elapsed_ms: 5000,
        },
    )
    .await?;
    let mut enabled_model_ids: Vec<String> = Vec::new();
    if !models.1.is_empty() {
        (interaction.notify)(crate::auth::types::AuthEvent::Progress {
            message: "Enabling models...".to_string(),
        });
        enabled_model_ids = enable_github_copilot_models(
            &credentials.credentials.access,
            &models.1,
            enterprise_domain.as_deref(),
            &interaction.signal,
        )
        .await?;
    }
    let mut available = models.0;
    available.extend(enabled_model_ids);
    // [...new Set([...models.availableModelIds, ...enabledModelIds])]
    let mut seen = std::collections::BTreeSet::new();
    available.retain(|id| seen.insert(id.clone()));
    Ok(credential_with(
        credentials.credentials.refresh.clone(),
        credentials.credentials.access.clone(),
        credentials.credentials.expires,
        enterprise_domain.as_deref(),
        Some(available),
    ))
}

async fn refresh_github_copilot_token(
    refresh_token: &str,
    enterprise_domain: Option<&str>,
    signal: &AbortSignal,
) -> Result<OAuthCredential, ThrownValue> {
    let credentials =
        refresh_github_copilot_access_token(refresh_token, enterprise_domain, signal).await?;
    let (available_model_ids, _) = fetch_github_copilot_models(
        &credentials.credentials.access,
        enterprise_domain,
        signal,
        RetryPolicy {
            max_retries: 0,
            max_elapsed_ms: 0,
        },
    )
    .await?;
    Ok(credential_with(
        credentials.credentials.refresh.clone(),
        credentials.credentials.access.clone(),
        credentials.credentials.expires,
        enterprise_domain,
        Some(available_model_ids),
    ))
}

/// Upstream `export const githubCopilotOAuth`.
pub fn github_copilot_oauth() -> OAuthAuth {
    let login: OAuthLoginFn = Box::new(|interaction, _options| {
        Box::pin(async move { login_github_copilot(&interaction).await })
    });

    let refresh: OAuthRefreshFn = Box::new(|credential, signal| {
        Box::pin(async move {
            let enterprise_domain = copilot_enterprise_domain(&credential);
            refresh_github_copilot_token(
                &credential.credentials.refresh,
                enterprise_domain.as_deref(),
                &signal,
            )
            .await
        })
    });

    let to_auth: OAuthToAuthFn = Box::new(|credential| {
        Box::pin(async move {
            let enterprise_domain = copilot_enterprise_domain(credential);
            // Derive the credential-specific proxy endpoint for each request.
            Ok(ModelAuth {
                api_key: Some(credential.credentials.access.clone()),
                headers: None,
                base_url: Some(get_github_copilot_base_url(
                    Some(&credential.credentials.access),
                    enterprise_domain.as_deref(),
                )),
            })
        })
    });

    OAuthAuth {
        name: "GitHub Copilot".to_string(),
        is_subscription: true,
        login_label: None,
        login,
        refresh,
        to_auth,
    }
}
