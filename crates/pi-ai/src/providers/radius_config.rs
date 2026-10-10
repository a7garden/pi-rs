//! Port of `packages/ai/src/providers/radius-config.ts` — the Radius gateway
//! config catalog helpers.
//!
//! # Deviations (per-file ledger)
//!
//! - **D15 — `RadiusOAuthCredential.gatewayConfig` is read from the OAuth
//!   credential's extra-keys map** ([`crate::auth::types::OAuthCredentials::
//!   extra`], upstream's `[key: string]: unknown` index signature) instead of
//!   a named field, because the frozen `auth` types carry extra keys
//!   untyped.
//! - **D16 — sanitization runs on raw `serde_json::Value`s** like upstream's
//!   unknown-tolerant validators; `RadiusGatewayModel` round-trips through
//!   JSON when promoting gateway models to [`crate::types::Model`].
//! - **D17 — `loadRadiusGatewayConfig` takes an explicit `FetchFn`** (plus
//!   the optional signal) instead of `globalThis.fetch`, matching the crate's
//!   [`crate::http::FetchFn`] seam (crate D4); the provider threads the crate
//!   default adapter.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

use crate::auth::types::OAuthCredential;
use crate::http::{FetchFn, HttpRequest};
use crate::types::{Model, ModelCost, ModelInput, ThinkingLevelMap};

/// Upstream `DEFAULT_RADIUS_GATEWAY`.
pub const DEFAULT_RADIUS_GATEWAY: &str = "https://radius.pi.dev";

/// One gateway-advertised model (`RadiusGatewayModel`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RadiusGatewayModel {
    pub id: String,
    pub name: String,
    pub reasoning: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    pub input: Vec<ModelInput>,
    pub cost: ModelCost,
    pub context_window: u64,
    pub max_tokens: u64,
}

/// The gateway's advertised catalog (`RadiusGatewayConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RadiusGatewayConfig {
    pub base_url: String,
    pub models: Vec<RadiusGatewayModel>,
}

fn is_radius_gateway_model(value: &Value) -> bool {
    let Some(model) = value.as_object() else {
        return false;
    };
    model.get("id").map(Value::is_string).unwrap_or(false)
        && model.get("name").map(Value::is_string).unwrap_or(false)
        && model
            .get("reasoning")
            .map(Value::is_boolean)
            .unwrap_or(false)
        && model.get("input").map(Value::is_array).unwrap_or(false)
        && model
            .get("cost")
            .map(|cost| cost.is_object())
            .unwrap_or(false)
        && model
            .get("contextWindow")
            .map(Value::is_number)
            .unwrap_or(false)
        && model
            .get("maxTokens")
            .map(Value::is_number)
            .unwrap_or(false)
}

fn sanitize_radius_gateway_config(config: Option<&Value>) -> Option<RadiusGatewayConfig> {
    let config = config?;
    let base_url = config.get("baseUrl").and_then(Value::as_str)?;
    let models = config.get("models").and_then(Value::as_array)?;
    Some(RadiusGatewayConfig {
        base_url: base_url.to_string(),
        models: models
            .iter()
            .filter(|model| is_radius_gateway_model(model))
            .filter_map(|model| serde_json::from_value(model.clone()).ok())
            .collect(),
    })
}

/// `normalizeRadiusGatewayUrl`.
pub fn normalize_radius_gateway_url(value: &str) -> String {
    let with_scheme = if value.starts_with("http://") || value.starts_with("https://") {
        value.to_string()
    } else {
        format!("https://{value}")
    };
    with_scheme.trim_end_matches('/').to_string()
}

/// `getRadiusCredentialConfig` (deviation D15).
pub fn get_radius_credential_config(
    credential: Option<&OAuthCredential>,
) -> Option<RadiusGatewayConfig> {
    let raw = credential.and_then(|credential| credential.credentials.extra.get("gatewayConfig"));
    sanitize_radius_gateway_config(raw)
}

/// `getRadiusModelsFromConfig`.
pub fn get_radius_models_from_config(
    provider_id: &str,
    config: &RadiusGatewayConfig,
) -> Vec<Model> {
    config
        .models
        .iter()
        .map(|model| {
            let mut value = serde_json::to_value(model).expect("gateway model serializes");
            let object = value.as_object_mut().expect("gateway model is an object");
            object.insert("api".to_string(), Value::String("pi-messages".to_string()));
            object.insert(
                "provider".to_string(),
                Value::String(provider_id.to_string()),
            );
            object.insert(
                "baseUrl".to_string(),
                Value::String(config.base_url.clone()),
            );
            serde_json::from_value(value).expect("gateway model converts to a chat model")
        })
        .collect()
}

/// `getRadiusModels`.
pub fn get_radius_models(provider_id: &str, credential: Option<&OAuthCredential>) -> Vec<Model> {
    match get_radius_credential_config(credential) {
        Some(config) => get_radius_models_from_config(provider_id, &config),
        None => Vec::new(),
    }
}

fn truncate_http_body(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.chars().count() > 512 {
        let truncated: String = trimmed.chars().take(512).collect();
        format!("{truncated}…")
    } else {
        trimmed.to_string()
    }
}

/// `loadRadiusGatewayConfig` (deviation D17: explicit fetch seam).
pub async fn load_radius_gateway_config(
    gateway: &str,
    api_key: Option<&str>,
    signal: Option<crate::utils::abort::AbortSignal>,
    fetch: FetchFn,
) -> Result<RadiusGatewayConfig, String> {
    let mut headers: Vec<(String, String)> =
        vec![("accept".to_string(), "application/json".to_string())];
    if let Some(api_key) = api_key {
        headers.push(("authorization".to_string(), format!("Bearer {api_key}")));
    }
    // `new URL("/v1/config", gateway)`: the absolute path replaces the
    // gateway URL's path entirely.
    let mut url = Url::parse(gateway).map_err(|error| error.to_string())?;
    url.set_path("/v1/config");
    let mut request = HttpRequest::new("GET", url.as_str());
    request.headers = headers;
    request.signal = signal;
    let response = fetch.send(request).await.map_err(|error| error.message)?;
    let body = response
        .body
        .collect()
        .await
        .map_err(|error| error.message)?;
    let body = String::from_utf8_lossy(&body).into_owned();
    if !(200..300).contains(&response.status) {
        return Err(format!(
            "Could not load Radius config from {gateway}: {}: {}",
            response.status,
            truncate_http_body(&body)
        ));
    }
    let parsed: Value = serde_json::from_str(&body).map_err(|error| error.to_string())?;
    let config = sanitize_radius_gateway_config(Some(&parsed));
    config.ok_or_else(|| format!("Invalid Radius config from {gateway}"))
}
