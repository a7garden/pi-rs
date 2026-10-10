//! Port of `packages/mcp/src/oauth/types.ts`.
//!
//! Adapted from modelcontextprotocol/typescript-sdk v1.29.0. Modified to use
//! dependency-free structural validation (upstream dropped Zod the same way):
//! metadata documents deserialize into serde structs that preserve unknown fields
//! via `#[serde(flatten)]`, matching upstream's spread semantics.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use url::Url;

use crate::protocol::jsonrpc::{is_object, McpError};

fn object(value: Value, name: &str) -> Result<Map<String, Value>, McpError> {
    if !is_object(&value) {
        return Err(McpError::other(format!("Invalid {name}")));
    }
    Ok(value.as_object().cloned().expect("checked"))
}

fn required_string(input: &Map<String, Value>, key: &str, name: &str) -> Result<String, McpError> {
    match input.get(key).and_then(Value::as_str) {
        Some(value) if !value.is_empty() => Ok(value.to_string()),
        _ => Err(McpError::other(format!("Invalid {name}"))),
    }
}

/// Treats `null` and `""` as absent: servers send them for fields they have no value
/// for, like `scope: ""`.
fn absent(input: &Map<String, Value>, key: &str) -> bool {
    match input.get(key) {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) => s.is_empty(),
        Some(_) => false,
    }
}

fn optional_string(
    input: &Map<String, Value>,
    key: &str,
    name: &str,
) -> Result<Option<String>, McpError> {
    if absent(input, key) {
        return Ok(None);
    }
    Ok(Some(required_string(input, key, name)?))
}

fn optional_strings(
    input: &Map<String, Value>,
    key: &str,
    name: &str,
) -> Result<Option<Vec<String>>, McpError> {
    if matches!(input.get(key), None | Some(Value::Null)) {
        return Ok(None);
    }
    match input.get(key).and_then(Value::as_array) {
        Some(items) if items.iter().all(|item| item.is_string()) => Ok(Some(
            items
                .iter()
                .map(|item| item.as_str().expect("checked").to_string())
                .collect(),
        )),
        _ => Err(McpError::other(format!("Invalid {name}"))),
    }
}

/// Upstream `safeUrl`: `URL.canParse` plus a dangerous-scheme check. Note upstream
/// reserves `URL` parse failures as `TypeError` (network) only inside discovery
/// flows; here parse problems surface as plain validation errors like upstream's
/// `throw new Error("Invalid ...")`.
fn safe_url(value: &str, name: &str) -> Result<String, McpError> {
    let parsed = Url::parse(value).map_err(|_| McpError::other(format!("Invalid {name}")))?;
    if matches!(parsed.scheme(), "javascript" | "data" | "vbscript") {
        return Err(McpError::other(format!("Invalid {name}")));
    }
    Ok(value.to_string())
}

fn optional_url(
    input: &Map<String, Value>,
    key: &str,
    name: &str,
) -> Result<Option<String>, McpError> {
    if absent(input, key) {
        return Ok(None);
    }
    Ok(Some(safe_url(
        input.get(key).and_then(Value::as_str).unwrap_or_default(),
        name,
    )?))
}

/// Drops `null` values the way upstream's `compact` drops `undefined`, so optional
/// fields are absent rather than present-but-null.
fn compact(map: Map<String, Value>) -> Map<String, Value> {
    map.into_iter()
        .filter(|(_, value)| !value.is_null())
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OAuthProtectedResourceMetadata {
    pub resource: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorization_servers: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scopes_supported: Option<Vec<String>>,
    /// Upstream index signature `[key: string]: unknown`.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthorizationServerMetadata {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registration_endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scopes_supported: Option<Vec<String>>,
    pub response_types_supported: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grant_types_supported: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_endpoint_auth_methods_supported: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_challenge_methods_supported: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_id_metadata_document_supported: Option<bool>,
    /// Whether authorization responses carry an `iss` parameter (RFC 9207).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorization_response_iss_parameter_supported: Option<bool>,
    /// Upstream index signature `[key: string]: unknown`.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OAuthTokens {
    pub access_token: String,
    pub token_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_in: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct OAuthClientMetadata {
    pub redirect_uris: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_endpoint_auth_method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grant_types: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_types: Option<Vec<String>>,
    /// OpenID Connect client type, `native` or `web`. Dynamic client registration derives it from
    /// `redirect_uris` when absent (MCP SEP-837).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub application_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logo_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contacts: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tos_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jwks_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jwks: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub software_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub software_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub software_statement: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OAuthClientInformation {
    pub client_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_id_issued_at: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_secret_expires_at: Option<f64>,
}

/// Upstream `OAuthClientInformationFull = OAuthClientInformation & OAuthClientMetadata`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OAuthClientInformationFull {
    #[serde(flatten)]
    pub information: OAuthClientInformation,
    #[serde(flatten)]
    pub metadata: OAuthClientMetadata,
    /// Registration responses carry arbitrary metadata fields; preserved like
    /// upstream's spread semantics.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Upstream `OAuthClientInformationMixed = OAuthClientInformation | OAuthClientInformationFull`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OAuthClientInformationMixed {
    Full(OAuthClientInformationFull),
    Plain(OAuthClientInformation),
}

impl OAuthClientInformationMixed {
    pub fn client_id(&self) -> &str {
        match self {
            OAuthClientInformationMixed::Full(full) => &full.information.client_id,
            OAuthClientInformationMixed::Plain(plain) => &plain.client_id,
        }
    }

    pub fn client_secret(&self) -> Option<&str> {
        match self {
            OAuthClientInformationMixed::Full(full) => full.information.client_secret.as_deref(),
            OAuthClientInformationMixed::Plain(plain) => plain.client_secret.as_deref(),
        }
    }

    /// Upstream `"token_endpoint_auth_method" in information`.
    pub fn token_endpoint_auth_method(&self) -> Option<&str> {
        match self {
            OAuthClientInformationMixed::Full(full) => {
                full.metadata.token_endpoint_auth_method.as_deref()
            }
            OAuthClientInformationMixed::Plain(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OAuthDiscoveryState {
    pub authorization_server_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorization_server_metadata: Option<AuthorizationServerMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_metadata: Option<OAuthProtectedResourceMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_metadata_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OAuthServerInfo {
    pub authorization_server_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorization_server_metadata: Option<AuthorizationServerMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_metadata: Option<OAuthProtectedResourceMetadata>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct OAuthChallenge {
    pub resource_metadata_url: Option<Url>,
    pub scope: Option<String>,
    pub error: Option<String>,
    pub error_description: Option<String>,
}

/// Upstream `parseProtectedResourceMetadata`.
pub fn parse_protected_resource_metadata(
    value: Value,
) -> Result<OAuthProtectedResourceMetadata, McpError> {
    let input = object(value, "OAuth protected resource metadata")?;
    let mut result = input.clone();
    result.insert(
        "resource".to_string(),
        json_str(safe_url(
            required_string(
                &input,
                "resource",
                "OAuth protected resource metadata resource",
            )?
            .as_str(),
            "OAuth protected resource metadata resource",
        )?),
    );
    if let Some(servers) =
        optional_strings(&input, "authorization_servers", "authorization_servers")?
    {
        let mut checked = Vec::new();
        for url in servers {
            checked.push(safe_url(&url, "authorization server URL")?);
        }
        result.insert(
            "authorization_servers".to_string(),
            serde_json::json!(checked),
        );
    } else {
        result.remove("authorization_servers");
    }
    match optional_strings(&input, "scopes_supported", "scopes_supported")? {
        Some(scopes) => {
            result.insert("scopes_supported".to_string(), serde_json::json!(scopes));
        }
        None => {
            result.remove("scopes_supported");
        }
    }
    serde_json::from_value(Value::Object(compact(result)))
        .map_err(|_| McpError::other("Invalid OAuth protected resource metadata"))
}

fn json_str(value: String) -> Value {
    Value::String(value)
}

/// Upstream `parseAuthorizationServerMetadata`.
pub fn parse_authorization_server_metadata(
    value: Value,
) -> Result<AuthorizationServerMetadata, McpError> {
    let input = object(value, "authorization server metadata")?;
    let response_types = optional_strings(
        &input,
        "response_types_supported",
        "response_types_supported",
    )?
    .ok_or_else(|| McpError::other("Invalid response_types_supported"))?;
    let mut result = input.clone();
    result.insert(
        "issuer".to_string(),
        json_str(safe_url(
            required_string(&input, "issuer", "authorization server issuer")?.as_str(),
            "authorization server issuer",
        )?),
    );
    result.insert(
        "authorization_endpoint".to_string(),
        json_str(safe_url(
            required_string(&input, "authorization_endpoint", "authorization endpoint")?.as_str(),
            "authorization endpoint",
        )?),
    );
    result.insert(
        "token_endpoint".to_string(),
        json_str(safe_url(
            required_string(&input, "token_endpoint", "token endpoint")?.as_str(),
            "token endpoint",
        )?),
    );
    match optional_url(&input, "registration_endpoint", "registration endpoint")? {
        Some(url) => {
            result.insert("registration_endpoint".to_string(), json_str(url));
        }
        None => {
            result.remove("registration_endpoint");
        }
    }
    match optional_strings(&input, "scopes_supported", "scopes_supported")? {
        Some(scopes) => {
            result.insert("scopes_supported".to_string(), serde_json::json!(scopes));
        }
        None => {
            result.remove("scopes_supported");
        }
    }
    result.insert(
        "response_types_supported".to_string(),
        serde_json::json!(response_types),
    );
    for key in [
        "grant_types_supported",
        "token_endpoint_auth_methods_supported",
        "code_challenge_methods_supported",
    ] {
        match optional_strings(&input, key, key)? {
            Some(values) => {
                result.insert(key.to_string(), serde_json::json!(values));
            }
            None => {
                result.remove(key);
            }
        }
    }
    for key in [
        "client_id_metadata_document_supported",
        "authorization_response_iss_parameter_supported",
    ] {
        match input.get(key) {
            Some(Value::Bool(value)) => {
                result.insert(key.to_string(), Value::Bool(*value));
            }
            _ => {
                result.remove(key);
            }
        }
    }
    serde_json::from_value(Value::Object(compact(result)))
        .map_err(|_| McpError::other("Invalid authorization server metadata"))
}

/// Upstream `parseOAuthTokens`.
pub fn parse_oauth_tokens(value: Value) -> Result<OAuthTokens, McpError> {
    let input = object(value, "OAuth token response")?;
    // `Number(null)` is 0, which would mark the token as expired at once. Upstream
    // applies JS `Number(...)` coercion; `NaN`/`Infinity` fail validation.
    let expires = if absent(&input, "expires_in") {
        None
    } else {
        let coerced = match input.get("expires_in") {
            Some(Value::Number(n)) => n.as_f64(),
            Some(Value::Bool(b)) => Some(if *b { 1.0 } else { 0.0 }),
            Some(Value::Array(items)) if items.is_empty() => Some(0.0),
            Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
            _ => None,
        };
        if !coerced.map(|expires| expires.is_finite()).unwrap_or(false) {
            return Err(McpError::other("Invalid expires_in"));
        }
        coerced
    };
    let tokens = OAuthTokens {
        access_token: required_string(&input, "access_token", "access_token")?,
        token_type: required_string(&input, "token_type", "token_type")?,
        expires_in: expires,
        scope: optional_string(&input, "scope", "scope")?,
        refresh_token: optional_string(&input, "refresh_token", "refresh_token")?,
        id_token: optional_string(&input, "id_token", "id_token")?,
    };
    Ok(tokens)
}

/// Upstream `parseClientInformation`.
pub fn parse_client_information(value: Value) -> Result<OAuthClientInformationFull, McpError> {
    let input = object(value, "OAuth client registration response")?;
    // Structural validation first (upstream validates while spreading).
    required_string(&input, "client_id", "client_id")?;
    optional_string(&input, "client_secret", "client_secret")?;
    let mut normalized = input;
    // Upstream defaults missing `redirect_uris` to `[]`.
    if !normalized.contains_key("redirect_uris") {
        normalized.insert("redirect_uris".to_string(), serde_json::json!([]));
    }
    serde_json::from_value(Value::Object(normalized))
        .map_err(|_| McpError::other("Invalid OAuth client registration response"))
}
