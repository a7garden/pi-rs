//! Port of `packages/coding-agent/src/core/mcp-servers.ts` — MCP server
//! configuration and the servers extensions register with
//! `pi.registerMcpServer()`.
//!
//! The core only validates and stores registrations. The MCP extension (built
//! in, or another extension that handles `mcp_servers_change`) connects them
//! next to the servers from `mcp.json` ([`crate::extensions::mcp`]).
//!
//! # Port deviations
//!
//! - **D120.** Upstream validates loosely-typed JSON and returns the same
//!   object with exposure aliases resolved (a spread copy that keeps unknown
//!   fields). The port validates from [`serde_json::Value`] to keep alias
//!   resolution and error strings exact, then produces the typed structs
//!   below; unknown extra fields are dropped, and the shared base fields are
//!   repeated per variant (Rust has no interface inheritance).
//! - **D121.** `toolPatternRegExp` compiles a JS `RegExp`; the port has no
//!   regex engine, so [`tool_pattern_matches`] implements the compiled pattern
//!   directly: split on `*`, every other character literal, full match. The
//!   upstream `.` also excludes line terminators, which the wildcard here
//!   matches; unobservable for real tool names.
//! - **D122.** JS `overrides[toolName]` also sees `Object.prototype` names
//!   (`"toString"`, …); the typed map lookup is own-property only, so a tool
//!   named like an inherited property falls through to the patterns.
//! - **D123.** `timeout` is [`serde_json::Number`] so integer seconds keep
//!   their wire form (JS numbers do not distinguish `60` from `60.0`).
//! - **D124.** `structuredClone` → [`Clone`] in [`McpServerRegistry::list`];
//!   the registry here (replace-on-register, ownership-checked unregister,
//!   change listener) replaces the minimal in-crate placeholder in
//!   [`crate::core::extensions::types`].

use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

/// Upstream `McpExposure`:
///
/// - `codemode`: tools are callable from codemode scripts but neither declared
///   to the model nor listed in the codemode description, which lists only the
///   server's namespace. Scripts find them with `searchTools()`.
///   `codemode-deferred` is accepted as an alias.
/// - `deferred`: not declared to the model until the `tool_search` tool loads
///   them; the model then calls them directly. Does not need codemode.
/// - `direct`: tools are declared to the model like any other tool (and
///   callable from codemode).
/// - `hidden`: tools are registered but unreachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpExposure {
    Codemode,
    Deferred,
    Direct,
    Hidden,
}

impl McpExposure {
    pub fn as_str(&self) -> &'static str {
        match self {
            McpExposure::Codemode => "codemode",
            McpExposure::Deferred => "deferred",
            McpExposure::Direct => "direct",
            McpExposure::Hidden => "hidden",
        }
    }

    pub fn parse(value: &str) -> Option<McpExposure> {
        match value {
            "codemode" => Some(McpExposure::Codemode),
            "deferred" => Some(McpExposure::Deferred),
            "direct" => Some(McpExposure::Direct),
            "hidden" => Some(McpExposure::Hidden),
            _ => None,
        }
    }
}

const MCP_EXPOSURES: [&str; 4] = ["codemode", "deferred", "direct", "hidden"];

/// Older exposure names, accepted in configs and replaced by their current
/// name when validated (upstream `MCP_EXPOSURE_ALIASES`).
fn resolve_exposure_alias(value: &str) -> &str {
    if value == "codemode-deferred" {
        "codemode"
    } else {
        value
    }
}

/// Upstream `McpServerConfig`'s inline `type` values: `"stdio"`, `"http"`, or
/// the legacy-but-accepted `"streamable-http"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum McpServerType {
    Stdio,
    Http,
    StreamableHttp,
}

/// Upstream `McpStdioServerConfig`. The shared `McpServerConfigBase` fields
/// (`exposure`, `description`, `toolExposure`, `enabled`, `timeout`) are
/// repeated on both config variants (D120).
///
/// - `exposure`: default `codemode`.
/// - `description`: what the server offers, in a sentence.
/// - `tool_exposure`: exposure of single tools, overriding `exposure`. Keys
///   are tool names as the server offers them, or patterns where `*` matches
///   any characters. An exact name wins over patterns; among patterns the
///   first match in insertion order wins.
/// - `enabled`: set to false to keep the entry without connecting. Default
///   true.
/// - `timeout`: per-request timeout in seconds. Progress notifications from
///   the server reset it. Default 60.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpStdioServerConfig {
    // -- McpServerConfigBase (D120) ----------------------------------------
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure: Option<McpExposure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_exposure: Option<IndexMap<String, McpExposure>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<serde_json::Number>,
    // ----------------------------------------------------------------------
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub r#type: Option<McpServerType>,
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<String>>,
    /// Values may reference environment variables (`${NAME}`) or commands
    /// (`!cmd`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<IndexMap<String, String>>,
    /// Relative paths resolve against the session working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
}

/// Upstream `McpOAuthConfig` — OAuth client settings for servers that do not
/// support dynamic client registration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpOAuthConfig {
    /// Pre-registered client id. Without it, pi registers a client with the
    /// authorization server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    /// May reference environment variables (`${NAME}`) or commands (`!cmd`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    /// Port of the loopback callback server, for clients registered with a
    /// fixed redirect URI. Without `callbackUrl`, the redirect URI is
    /// `http://127.0.0.1:<port>/callback`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub callback_port: Option<u16>,
    /// Redirect URI registered for `clientId`, for example
    /// `http://localhost:8080/oauth/callback`. It must be an `http` URI on
    /// `localhost`, `127.0.0.1`, or `[::1]`. Without a port, the callback
    /// server listens on `callbackPort` or a free port, which is added to the
    /// URI (RFC 8252).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub callback_url: Option<String>,
    /// Scopes to request, separated by spaces. Default: the scopes the server
    /// advertises.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// `client_name` sent with dynamic client registration, for servers that
    /// only accept known clients. Default: `pi`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_name: Option<String>,
    /// How pi identifies itself without `clientId`. `dcr` (default): dynamic
    /// client registration. `cimd`: pi's Client ID Metadata Document on
    /// pi.dev, for authorization servers that allow pi by that URL. The server
    /// must support it for public clients, and the callback must use the
    /// default path `/callback`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_registration: Option<McpClientRegistration>,
    /// Authorization server metadata document (RFC 8414 or OpenID Connect
    /// discovery) to use instead of discovery through the server, for servers
    /// that advertise a wrong authorization server or none. The document is
    /// trusted as configured. Must use https, except on loopback hosts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_server_metadata_url: Option<String>,
}

/// Upstream `clientRegistration?: "dcr" | "cimd"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpClientRegistration {
    Dcr,
    Cimd,
}

const LOOPBACK_HOSTS: [&str; 3] = ["localhost", "127.0.0.1", "[::1]"];

/// Whether a parsed URL is an `http` URL on a loopback host without query or
/// fragment (the body of upstream `isLoopbackRedirectUri`; JS
/// `url.search === ""` also holds for a bare `?`, hence `is_empty`).
fn url_is_loopback_redirect(url: &Url) -> bool {
    url.scheme() == "http"
        && LOOPBACK_HOSTS.contains(&url.host_str().unwrap_or_default())
        && url.query().is_none_or(str::is_empty)
        && url.fragment().is_none_or(str::is_empty)
}

/// Whether a redirect URI can be served by pi's loopback callback server
/// (`isLoopbackRedirectUri`).
pub fn is_loopback_redirect_uri(value: &str) -> bool {
    // `URL.canParse` → `Url::parse` (D85 in extensions/mcp).
    Url::parse(value)
        .map(|url| url_is_loopback_redirect(&url))
        .unwrap_or(false)
}

/// Upstream `McpHttpServerConfig` (shared `McpServerConfigBase` fields as on
/// [`McpStdioServerConfig`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpHttpServerConfig {
    // -- McpServerConfigBase (D120) ----------------------------------------
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure: Option<McpExposure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_exposure: Option<IndexMap<String, McpExposure>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<serde_json::Number>,
    // ----------------------------------------------------------------------
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub r#type: Option<McpServerType>,
    pub url: String,
    /// Values may reference environment variables (`${NAME}`) or commands
    /// (`!cmd`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oauth: Option<McpOAuthConfig>,
    /// Send the token of a pi provider (`/login <provider>`) instead of using
    /// OAuth. Not allowed in project `mcp.json` files, and requires https
    /// except on loopback hosts, since it sends the credential to `url`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<McpAuthConfig>,
}

/// Upstream `auth?: { provider: string }` on `McpHttpServerConfig`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpAuthConfig {
    pub provider: String,
}

/// Upstream `McpServerConfig = McpStdioServerConfig | McpHttpServerConfig`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpServerConfig {
    Stdio(McpStdioServerConfig),
    Http(McpHttpServerConfig),
}

/// Namespace of a server's tools: `mcp__<server>` with `-` replaced by `_`,
/// like the tool names (`mcpNamespace`).
pub fn mcp_namespace(server: &str) -> String {
    format!("mcp__{}", server.replace('-', "_"))
}

/// Whether every entry of the value is a string (`isStringRecord`).
fn is_string_record(value: &Value) -> bool {
    value
        .as_object()
        .is_some_and(|object| object.values().all(|entry| entry.is_string()))
}

/// Upstream `validateOAuth`: checks the loosely-typed `oauth` entry
/// (`None` = absent) and returns the error message, if any.
fn validate_oauth(oauth: Option<&Value>) -> Option<String> {
    let oauth = oauth?;
    if !oauth.is_object() {
        return Some("oauth must be an object".to_string());
    }
    if let Some(client_id) = oauth.get("clientId") {
        if client_id.as_str().is_none() {
            return Some("oauth.clientId must be a string".to_string());
        }
    }
    if let Some(client_secret) = oauth.get("clientSecret") {
        if client_secret.as_str().is_none() {
            return Some("oauth.clientSecret must be a string".to_string());
        }
    }
    let callback_port = match oauth.get("callbackPort") {
        None => None,
        Some(raw) => match raw.as_f64() {
            Some(port) if port.fract() == 0.0 && (1.0..=65535.0).contains(&port) => {
                Some(port as u16)
            }
            _ => return Some("oauth.callbackPort must be a port number".to_string()),
        },
    };
    if let Some(raw) = oauth.get("callbackUrl") {
        let parsed = raw.as_str().and_then(|text| Url::parse(text).ok());
        if !parsed.as_ref().is_some_and(url_is_loopback_redirect) {
            return Some(
				"oauth.callbackUrl must be an http URI on localhost, 127.0.0.1, or [::1] without query or fragment"
					.to_string(),
			);
        }
        // JS `url.port` is "" for a default or absent port; `url.port()` is
        // `None` then, so the comparison only runs for an explicit port.
        if let (Some(url), Some(port)) = (parsed, callback_port) {
            if url.port().is_some_and(|url_port| url_port != port) {
                return Some(
                    "oauth.callbackUrl and oauth.callbackPort name different ports".to_string(),
                );
            }
        }
    }
    if let Some(scope) = oauth.get("scope") {
        if scope.as_str().is_none() {
            return Some("oauth.scope must be a string".to_string());
        }
    }
    if let Some(client_name) = oauth.get("clientName") {
        if !client_name
            .as_str()
            .is_some_and(|text| !text.trim().is_empty())
        {
            return Some("oauth.clientName must be a non-empty string".to_string());
        }
    }
    if let Some(registration) = oauth.get("clientRegistration") {
        if registration.as_str() != Some("dcr") {
            if registration.as_str() != Some("cimd") {
                return Some("oauth.clientRegistration must be \"dcr\" or \"cimd\"".to_string());
            }
            if oauth.get("clientId").is_some() || oauth.get("clientName").is_some() {
                return Some(
					"oauth.clientRegistration \"cimd\" cannot be combined with oauth.clientId or oauth.clientName"
						.to_string(),
				);
            }
            let callback = oauth
                .get("callbackUrl")
                .filter(|raw| raw.is_string())
                .and_then(|raw| raw.as_str())
                .and_then(|text| Url::parse(text).ok());
            if let Some(callback) = callback {
                if callback.host_str() == Some("[::1]") || callback.path() != "/callback" {
                    return Some(
						"oauth.clientRegistration \"cimd\" requires oauth.callbackUrl on localhost or 127.0.0.1 with path /callback"
							.to_string(),
					);
                }
            }
        }
    }
    if let Some(metadata_url) = oauth.get("authServerMetadataUrl") {
        let parsed = metadata_url.as_str().and_then(|text| Url::parse(text).ok());
        let valid = parsed.is_some_and(|url| {
            url.scheme() == "https"
                || (url.scheme() == "http"
                    && LOOPBACK_HOSTS.contains(&url.host_str().unwrap_or_default()))
        });
        if !valid {
            return Some(
				"oauth.authServerMetadataUrl must be an https URL, or http on localhost, 127.0.0.1, or [::1]"
					.to_string(),
			);
        }
    }
    None
}

/// The exposure a raw config entry stands for, with aliases resolved; non-
/// strings and unknown names are `None` (upstream resolves aliases and then
/// checks `isExposure`).
fn exposure_of(raw: &Value) -> Option<McpExposure> {
    raw.as_str()
        .map(resolve_exposure_alias)
        .and_then(McpExposure::parse)
}

/// Whether a tool name matches any of the entries, each an exact name or a
/// pattern (upstream `createToolNameMatcher`; the returned closure is a
/// matcher struct here, D121).
pub fn create_tool_name_matcher(entries: &[String]) -> ToolNameMatcher {
    ToolNameMatcher {
        names: entries
            .iter()
            .filter(|entry| !entry.contains('*'))
            .cloned()
            .collect(),
        patterns: entries
            .iter()
            .filter(|entry| entry.contains('*'))
            .cloned()
            .collect(),
    }
}

/// Matcher over exact tool names and `*` patterns; see
/// [`create_tool_name_matcher`].
#[derive(Debug, Clone, Default)]
pub struct ToolNameMatcher {
    names: std::collections::HashSet<String>,
    patterns: Vec<String>,
}

impl ToolNameMatcher {
    /// Whether `name` matches any entry.
    pub fn matches(&self, name: &str) -> bool {
        self.names.contains(name)
            || self
                .patterns
                .iter()
                .any(|pattern| tool_pattern_matches(pattern, name))
    }
}

/// MCP resource tools, which reach every server with resources
/// (upstream `LIST_MCP_RESOURCES_TOOL` and siblings).
pub const LIST_MCP_RESOURCES_TOOL: &str = "list_mcp_resources";
pub const LIST_MCP_RESOURCE_TEMPLATES_TOOL: &str = "list_mcp_resource_templates";
pub const READ_MCP_RESOURCE_TOOL: &str = "read_mcp_resource";

/// Whether a tool comes from MCP: a server tool (`mcp__<server>__<tool>`) or a
/// resource tool (`isMcpToolName`).
pub fn is_mcp_tool_name(name: &str) -> bool {
    name.starts_with("mcp__")
        || matches!(
            name,
            LIST_MCP_RESOURCES_TOOL | LIST_MCP_RESOURCE_TEMPLATES_TOOL | READ_MCP_RESOURCE_TOOL
        )
}

/// Upstream `toolPatternRegExp` without a regex engine (D121): `*` matches any
/// characters, every other character is literal, and the pattern must match
/// the whole tool name.
fn tool_pattern_matches(pattern: &str, tool_name: &str) -> bool {
    let segments: Vec<&str> = pattern.split('*').collect();
    let Some((first, rest)) = segments.split_first() else {
        return true;
    };
    let Some(&last) = rest.last() else {
        // No `*`: the anchored pattern is an exact literal match.
        return tool_name == *first;
    };
    let Some(after_first) = tool_name.strip_prefix(first) else {
        return false;
    };
    let Some(before_last) = after_first.strip_suffix(last) else {
        return false;
    };
    // The segments between the first and last must occur in order; taking the
    // earliest occurrence of each is complete for this kind of pattern.
    let mut cursor = 0usize;
    for segment in &rest[..rest.len() - 1] {
        match before_last[cursor..].find(segment) {
            Some(found) => cursor += found + segment.len(),
            None => return false,
        }
    }
    true
}

/// Exposure of one tool of a server: its `toolExposure` entry, else the
/// server's `exposure` (`getMcpToolExposure`).
pub fn get_mcp_tool_exposure(config: &McpServerConfig, tool_name: &str) -> McpExposure {
    let (exposure, tool_exposure) = match config {
        McpServerConfig::Stdio(config) => (&config.exposure, &config.tool_exposure),
        McpServerConfig::Http(config) => (&config.exposure, &config.tool_exposure),
    };
    if let Some(overrides) = tool_exposure {
        // An exact name wins over patterns (own-property lookup, D122).
        if let Some(exact) = overrides.get(tool_name) {
            return *exact;
        }
        // Among patterns, the first match in insertion order wins.
        for (pattern, override_exposure) in overrides {
            if pattern.contains('*') && tool_pattern_matches(pattern, tool_name) {
                return *override_exposure;
            }
        }
    }
    exposure.unwrap_or(McpExposure::Codemode)
}

/// Validate one server entry of the `mcpServers` shape. Returns the config
/// with exposure aliases resolved, or the upstream error message
/// (`validateMcpServerConfig`).
pub fn validate_mcp_server_config(name: &str, raw: &Value) -> Result<McpServerConfig, String> {
    // `/^[A-Za-z0-9_-]+$/`
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(format!(
            "invalid server name \"{name}\" (use letters, digits, \"_\" and \"-\")"
        ));
    }
    let Some(value) = raw.as_object() else {
        return Err(format!("server \"{name}\" must be an object"));
    };
    let exposures = MCP_EXPOSURES
        .iter()
        .map(|exposure| format!("\"{exposure}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let exposure = match value.get("exposure") {
        None => None,
        Some(raw) => match exposure_of(raw) {
            Some(exposure) => Some(exposure),
            None => {
                return Err(format!(
                    "server \"{name}\": exposure must be one of {exposures}"
                ))
            }
        },
    };
    let tool_exposure = match value.get("toolExposure") {
        None => None,
        Some(raw) => {
            let Some(map) = raw.as_object() else {
                return Err(format!(
                    "server \"{name}\": toolExposure must map tool names to exposures"
                ));
            };
            let mut resolved = IndexMap::with_capacity(map.len());
            for (tool, entry) in map {
                let Some(exposure) = exposure_of(entry) else {
                    return Err(format!(
                        "server \"{name}\": toolExposure \"{tool}\" must be one of {exposures}"
                    ));
                };
                resolved.insert(tool.clone(), exposure);
            }
            Some(resolved)
        }
    };
    let enabled = match value.get("enabled") {
        None => None,
        Some(raw) => match raw.as_bool() {
            Some(enabled) => Some(enabled),
            None => return Err(format!("server \"{name}\": enabled must be a boolean")),
        },
    };
    let description = match value.get("description") {
        None => None,
        Some(raw) => match raw.as_str() {
            Some(description) => Some(description.to_string()),
            None => return Err(format!("server \"{name}\": description must be a string")),
        },
    };
    let timeout = match value.get("timeout") {
        None => None,
        Some(raw) => match raw.as_number() {
            Some(number) if number.as_f64().is_some_and(|seconds| seconds > 0.0) => {
                Some(number.clone())
            }
            _ => {
                return Err(format!(
                    "server \"{name}\": timeout must be a positive number of seconds"
                ))
            }
        },
    };
    if value.get("type").and_then(Value::as_str) == Some("sse") {
        return Err(format!(
            "server \"{name}\": legacy SSE transport is not supported; use the streamable HTTP URL"
        ));
    }
    let raw_type = value.get("type").and_then(Value::as_str);
    if let Some(url_text) = value.get("url").and_then(Value::as_str) {
        if matches!(raw_type, None | Some("http") | Some("streamable-http")) {
            let url = match Url::parse(url_text) {
                Ok(url) if matches!(url.scheme(), "http" | "https") => url,
                _ => {
                    return Err(format!(
                        "server \"{name}\": url must be an http or https URL"
                    ))
                }
            };
            if let Some(headers) = value.get("headers") {
                if !is_string_record(headers) {
                    return Err(format!(
                        "server \"{name}\": headers must map names to strings"
                    ));
                }
            }
            if let Some(oauth_error) = validate_oauth(value.get("oauth")) {
                return Err(format!("server \"{name}\": {oauth_error}"));
            }
            let auth = match value.get("auth") {
                None => None,
                Some(raw) => {
                    let provider = raw
                        .as_object()
                        .and_then(|auth| auth.get("provider"))
                        .and_then(Value::as_str);
                    match provider {
                        Some(provider) if !provider.is_empty() => {
                            if url.scheme() != "https"
                                && !LOOPBACK_HOSTS.contains(&url.host_str().unwrap_or_default())
                            {
                                return Err(format!(
									"server \"{name}\": auth requires an https URL, or http on localhost, 127.0.0.1, or [::1]"
								));
                            }
                            Some(McpAuthConfig {
                                provider: provider.to_string(),
                            })
                        }
                        _ => {
                            return Err(format!(
                                "server \"{name}\": auth.provider must be a provider name"
                            ))
                        }
                    }
                }
            };
            return Ok(McpServerConfig::Http(McpHttpServerConfig {
                exposure,
                description,
                tool_exposure,
                enabled,
                timeout,
                r#type: raw_type.map(|text| {
                    if text == "streamable-http" {
                        McpServerType::StreamableHttp
                    } else {
                        McpServerType::Http
                    }
                }),
                url: url_text.to_string(),
                headers: value.get("headers").map(|raw| {
                    raw.as_object()
                        .expect("validated string record")
                        .iter()
                        .map(|(header, entry)| {
                            (
                                header.clone(),
                                entry.as_str().expect("validated string").to_string(),
                            )
                        })
                        .collect()
                }),
                oauth: value.get("oauth").map(|raw| McpOAuthConfig {
                    client_id: raw
                        .get("clientId")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    client_secret: raw
                        .get("clientSecret")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    callback_port: raw
                        .get("callbackPort")
                        .and_then(Value::as_f64)
                        .map(|port| port as u16),
                    callback_url: raw
                        .get("callbackUrl")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    scope: raw.get("scope").and_then(Value::as_str).map(str::to_string),
                    client_name: raw
                        .get("clientName")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    client_registration: raw
                        .get("clientRegistration")
                        .and_then(Value::as_str)
                        .and_then(|text| match text {
                            "dcr" => Some(McpClientRegistration::Dcr),
                            "cimd" => Some(McpClientRegistration::Cimd),
                            _ => None,
                        }),
                    auth_server_metadata_url: raw
                        .get("authServerMetadataUrl")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                }),
                auth,
            }));
        }
    }
    if let Some(command) = value.get("command").and_then(Value::as_str) {
        if matches!(raw_type, None | Some("stdio")) {
            if let Some(args) = value.get("args") {
                let valid = args
                    .as_array()
                    .is_some_and(|args| args.iter().all(|arg| arg.is_string()));
                if !valid {
                    return Err(format!(
                        "server \"{name}\": args must be an array of strings"
                    ));
                }
            }
            if let Some(env) = value.get("env") {
                if !is_string_record(env) {
                    return Err(format!("server \"{name}\": env must map names to strings"));
                }
            }
            if let Some(cwd) = value.get("cwd") {
                if cwd.as_str().is_none() {
                    return Err(format!("server \"{name}\": cwd must be a string"));
                }
            }
            return Ok(McpServerConfig::Stdio(McpStdioServerConfig {
                exposure,
                description,
                tool_exposure,
                enabled,
                timeout,
                r#type: raw_type.map(|_| McpServerType::Stdio),
                command: command.to_string(),
                args: value.get("args").map(|raw| {
                    raw.as_array()
                        .expect("validated array")
                        .iter()
                        .map(|arg| arg.as_str().expect("validated string").to_string())
                        .collect()
                }),
                env: value.get("env").map(|raw| {
                    raw.as_object()
                        .expect("validated string record")
                        .iter()
                        .map(|(name, entry)| {
                            (
                                name.clone(),
                                entry.as_str().expect("validated string").to_string(),
                            )
                        })
                        .collect()
                }),
                cwd: value.get("cwd").and_then(Value::as_str).map(str::to_string),
            }));
        }
    }
    Err(format!(
        "server \"{name}\" needs either \"command\" (stdio) or \"url\" (streamable HTTP)"
    ))
}

/// A server an extension registered with `pi.registerMcpServer()` (upstream
/// `RegisteredMcpServer`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RegisteredMcpServer {
    pub name: String,
    pub config: McpServerConfig,
    /// Path of the extension that registered the server.
    pub extension_path: String,
}

/// Listener invoked after every registry change (upstream `changeListener`).
pub type McpChangeListener = Arc<dyn Fn() + Send + Sync>;

/// Servers registered by the extensions of one runtime (upstream
/// `McpServerRegistry`). The map keeps registration order, like the JS `Map`.
#[derive(Default)]
pub struct McpServerRegistry {
    servers: Mutex<IndexMap<String, RegisteredMcpServer>>,
    change_listener: Mutex<Option<McpChangeListener>>,
}

impl std::fmt::Debug for McpServerRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpServerRegistry")
            .field("servers", &self.servers.lock().len())
            .finish()
    }
}

impl McpServerRegistry {
    /// Register or replace a server. The caller checks ownership. Like the JS
    /// `Map`, replacing keeps the original position.
    pub fn register(&self, server: RegisteredMcpServer) {
        self.servers.lock().insert(server.name.clone(), server);
        self.notify_changed();
    }

    /// Remove a server registered by `extension_path`. Servers of other
    /// extensions are left alone.
    pub fn unregister(&self, name: &str, extension_path: &str) {
        let owned = self
            .servers
            .lock()
            .get(name)
            .is_some_and(|server| server.extension_path == extension_path);
        if !owned {
            return;
        }
        self.servers.lock().shift_remove(name);
        self.notify_changed();
    }

    pub fn get(&self, name: &str) -> Option<RegisteredMcpServer> {
        self.servers.lock().get(name).cloned()
    }

    /// Copies of the registered servers, in registration order.
    pub fn list(&self) -> Vec<RegisteredMcpServer> {
        self.servers.lock().values().cloned().collect()
    }

    /// Called after every change. The runner sets it when it binds, to emit
    /// `mcp_servers_change`.
    pub fn set_change_listener(&self, listener: Option<McpChangeListener>) {
        *self.change_listener.lock() = listener;
    }

    fn notify_changed(&self) {
        if let Some(listener) = self.change_listener.lock().clone() {
            listener();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- tool name matching (regression #5109 semantics) -----

    fn matcher(entries: &[&str]) -> ToolNameMatcher {
        create_tool_name_matcher(
            &entries
                .iter()
                .map(|entry| entry.to_string())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn tool_name_matcher_matches_exact_names_and_patterns() {
        let m = matcher(&["*_tool", "ask_*", "re*"]);
        assert!(m.matches("dynamic_tool"));
        assert!(m.matches("ask_question"));
        assert!(m.matches("read"));
        assert!(!m.matches("write"));
        // Exact names win without pattern machinery.
        assert!(matcher(&["read"]).matches("read"));
        assert!(!matcher(&["read"]).matches("readx"));
    }

    #[test]
    fn mcp_tool_names_cover_server_and_resource_tools() {
        assert!(is_mcp_tool_name("mcp__radius__search"));
        assert!(is_mcp_tool_name(LIST_MCP_RESOURCES_TOOL));
        assert!(is_mcp_tool_name(LIST_MCP_RESOURCE_TEMPLATES_TOOL));
        assert!(is_mcp_tool_name(READ_MCP_RESOURCE_TOOL));
        assert!(!is_mcp_tool_name("read"));
    }

    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn validate(name: &str, raw: Value) -> Result<McpServerConfig, String> {
        validate_mcp_server_config(name, &raw)
    }

    // -- name validation ------------------------------------------------------

    #[test]
    fn accepts_valid_server_names() {
        for name in ["a", "Z", "0", "_", "-", "dev-radius", "my_server_2"] {
            assert!(validate(name, json!({ "command": "c" })).is_ok(), "{name}");
        }
    }

    #[test]
    fn rejects_invalid_server_names() {
        for name in [
            "",
            "my server",
            "dot.name",
            "slash/name",
            "unicode-名前",
            "plus+",
        ] {
            assert_eq!(
                validate(name, json!({ "command": "c" })),
                Err(format!(
                    "invalid server name \"{name}\" (use letters, digits, \"_\" and \"-\")"
                ))
            );
        }
    }

    // -- exposure aliases ------------------------------------------------------

    #[test]
    fn resolves_exposure_aliases() {
        let config = validate(
            "x",
            json!({ "command": "c", "exposure": "codemode-deferred" }),
        )
        .expect("valid");
        let McpServerConfig::Stdio(config) = config else {
            panic!("stdio");
        };
        assert_eq!(config.exposure, Some(McpExposure::Codemode));

        let config = validate(
            "x",
            json!({
                "command": "c",
                "exposure": "codemode-deferred",
                "toolExposure": { "a": "codemode-deferred", "b": "hidden" }
            }),
        )
        .expect("valid");
        let McpServerConfig::Stdio(config) = config else {
            panic!("stdio");
        };
        assert_eq!(
            config.tool_exposure.as_ref().expect("map").get("a"),
            Some(&McpExposure::Codemode)
        );
        assert_eq!(
            config.tool_exposure.as_ref().expect("map").get("b"),
            Some(&McpExposure::Hidden)
        );

        // The resolved config serializes with current names only.
        let written = serde_json::to_value(&config).expect("serialize");
        assert_eq!(written["exposure"], "codemode");
        assert_eq!(written["toolExposure"]["a"], "codemode");
    }

    // -- validate error strings -------------------------------------------------

    #[test]
    fn rejects_non_object_entries() {
        for raw in [json!(5), json!("c"), json!([]), json!(null), json!(true)] {
            assert_eq!(
                validate("x", raw),
                Err("server \"x\" must be an object".to_string())
            );
        }
    }

    #[test]
    fn rejects_bad_exposure() {
        assert_eq!(
			validate("x", json!({ "command": "c", "exposure": "loud" })),
			Err("server \"x\": exposure must be one of \"codemode\", \"deferred\", \"direct\", \"hidden\"".to_string())
		);
        // Non-strings fail the same way (upstream `typeof` check).
        assert_eq!(
			validate("x", json!({ "command": "c", "exposure": 5 })),
			Err("server \"x\": exposure must be one of \"codemode\", \"deferred\", \"direct\", \"hidden\"".to_string())
		);
    }

    #[test]
    fn rejects_bad_tool_exposure() {
        assert_eq!(
            validate("x", json!({ "command": "c", "toolExposure": "all" })),
            Err("server \"x\": toolExposure must map tool names to exposures".to_string())
        );
        assert_eq!(
			validate("x", json!({ "command": "c", "toolExposure": { "read": "direct", "write": "loud" } })),
			Err("server \"x\": toolExposure \"write\" must be one of \"codemode\", \"deferred\", \"direct\", \"hidden\"".to_string())
		);
    }

    #[test]
    fn rejects_bad_enabled_description_timeout() {
        assert_eq!(
            validate("x", json!({ "command": "c", "enabled": "yes" })),
            Err("server \"x\": enabled must be a boolean".to_string())
        );
        assert_eq!(
            validate("x", json!({ "command": "c", "description": 5 })),
            Err("server \"x\": description must be a string".to_string())
        );
        for timeout in [json!(0), json!(-1), json!(-0.5)] {
            assert_eq!(
                validate("x", json!({ "command": "c", "timeout": timeout })),
                Err("server \"x\": timeout must be a positive number of seconds".to_string())
            );
        }
        assert_eq!(
            validate("x", json!({ "command": "c", "timeout": "60" })),
            Err("server \"x\": timeout must be a positive number of seconds".to_string())
        );
    }

    #[test]
    fn rejects_legacy_sse_before_other_url_checks() {
        assert_eq!(
            validate(
                "x",
                json!({ "type": "sse", "url": "https://example.com/sse" })
            ),
            Err(
                "server \"x\": legacy SSE transport is not supported; use the streamable HTTP URL"
                    .to_string()
            )
        );
        // `sse` wins even when the URL would fail its own check.
        assert_eq!(
            validate("x", json!({ "type": "sse", "url": "not a url" })),
            Err(
                "server \"x\": legacy SSE transport is not supported; use the streamable HTTP URL"
                    .to_string()
            )
        );
    }

    #[test]
    fn rejects_bad_http_configs() {
        assert_eq!(
            validate("x", json!({ "url": "not a url" })),
            Err("server \"x\": url must be an http or https URL".to_string())
        );
        assert_eq!(
            validate("x", json!({ "url": "ftp://example.com/mcp" })),
            Err("server \"x\": url must be an http or https URL".to_string())
        );
        assert_eq!(
            validate(
                "x",
                json!({ "url": "https://example.com/mcp", "headers": { "A": 1 } })
            ),
            Err("server \"x\": headers must map names to strings".to_string())
        );
    }

    #[test]
    fn rejects_bad_auth() {
        assert_eq!(
            validate("x", json!({ "url": "https://example.com/mcp", "auth": {} })),
            Err("server \"x\": auth.provider must be a provider name".to_string())
        );
        assert_eq!(
            validate(
                "x",
                json!({ "url": "https://example.com/mcp", "auth": { "provider": "" } })
            ),
            Err("server \"x\": auth.provider must be a provider name".to_string())
        );
        assert_eq!(
            validate(
                "x",
                json!({ "url": "https://example.com/mcp", "auth": "openai" })
            ),
            Err("server \"x\": auth.provider must be a provider name".to_string())
        );
        assert_eq!(
			validate("x", json!({ "url": "http://example.com/mcp", "auth": { "provider": "openai" } })),
			Err("server \"x\": auth requires an https URL, or http on localhost, 127.0.0.1, or [::1]".to_string())
		);
        // Loopback http is allowed.
        assert!(validate(
            "x",
            json!({ "url": "http://localhost:3000/mcp", "auth": { "provider": "openai" } })
        )
        .is_ok());
    }

    #[test]
    fn rejects_bad_stdio_configs() {
        assert_eq!(
            validate("x", json!({ "command": "c", "args": "all" })),
            Err("server \"x\": args must be an array of strings".to_string())
        );
        assert_eq!(
            validate("x", json!({ "command": "c", "args": ["ok", 5] })),
            Err("server \"x\": args must be an array of strings".to_string())
        );
        assert_eq!(
            validate("x", json!({ "command": "c", "env": { "A": 1 } })),
            Err("server \"x\": env must map names to strings".to_string())
        );
        assert_eq!(
            validate("x", json!({ "command": "c", "cwd": 5 })),
            Err("server \"x\": cwd must be a string".to_string())
        );
    }

    #[test]
    fn rejects_entries_without_transport() {
        assert_eq!(
            validate("x", json!({})),
            Err(
                "server \"x\" needs either \"command\" (stdio) or \"url\" (streamable HTTP)"
                    .to_string()
            )
        );
        // A `type` that does not admit the present field falls through.
        assert_eq!(
            validate(
                "x",
                json!({ "type": "stdio", "url": "https://example.com/mcp" })
            ),
            Err(
                "server \"x\" needs either \"command\" (stdio) or \"url\" (streamable HTTP)"
                    .to_string()
            )
        );
        assert_eq!(
            validate("x", json!({ "type": "http", "command": "c" })),
            Err(
                "server \"x\" needs either \"command\" (stdio) or \"url\" (streamable HTTP)"
                    .to_string()
            )
        );
        // Non-string `url` is not an http candidate.
        assert_eq!(
            validate("x", json!({ "url": 5 })),
            Err(
                "server \"x\" needs either \"command\" (stdio) or \"url\" (streamable HTTP)"
                    .to_string()
            )
        );
    }

    // -- oauth validation --------------------------------------------------------

    #[test]
    fn rejects_bad_oauth() {
        let url = json!({ "url": "https://example.com/mcp" });
        let cases: Vec<(Value, &str)> = vec![
			(json!({ "oauth": null }), "oauth must be an object"),
			(json!({ "oauth": "x" }), "oauth must be an object"),
			(json!({ "oauth": { "clientId": 5 } }), "oauth.clientId must be a string"),
			(json!({ "oauth": { "clientSecret": 5 } }), "oauth.clientSecret must be a string"),
			(json!({ "oauth": { "callbackPort": 0 } }), "oauth.callbackPort must be a port number"),
			(json!({ "oauth": { "callbackPort": 65536 } }), "oauth.callbackPort must be a port number"),
			(json!({ "oauth": { "callbackPort": 1.5 } }), "oauth.callbackPort must be a port number"),
			(json!({ "oauth": { "callbackPort": "8080" } }), "oauth.callbackPort must be a port number"),
			(
				json!({ "oauth": { "callbackUrl": "https://localhost/callback" } }),
				"oauth.callbackUrl must be an http URI on localhost, 127.0.0.1, or [::1] without query or fragment",
			),
			(
				json!({ "oauth": { "callbackUrl": "http://example.com/callback" } }),
				"oauth.callbackUrl must be an http URI on localhost, 127.0.0.1, or [::1] without query or fragment",
			),
			(
				json!({ "oauth": { "callbackUrl": "http://localhost/callback?code=1" } }),
				"oauth.callbackUrl must be an http URI on localhost, 127.0.0.1, or [::1] without query or fragment",
			),
			(
				json!({ "oauth": { "callbackUrl": "http://localhost/callback#frag" } }),
				"oauth.callbackUrl must be an http URI on localhost, 127.0.0.1, or [::1] without query or fragment",
			),
			(
				json!({ "oauth": { "callbackPort": 8080, "callbackUrl": "http://localhost:9000/callback" } }),
				"oauth.callbackUrl and oauth.callbackPort name different ports",
			),
			(json!({ "oauth": { "scope": 5 } }), "oauth.scope must be a string"),
			(json!({ "oauth": { "clientName": "" } }), "oauth.clientName must be a non-empty string"),
			(json!({ "oauth": { "clientName": "   " } }), "oauth.clientName must be a non-empty string"),
			(json!({ "oauth": { "clientName": 5 } }), "oauth.clientName must be a non-empty string"),
			(
				json!({ "oauth": { "clientRegistration": "wat" } }),
				"oauth.clientRegistration must be \"dcr\" or \"cimd\"",
			),
			(
				json!({ "oauth": { "clientRegistration": 5 } }),
				"oauth.clientRegistration must be \"dcr\" or \"cimd\"",
			),
			(
				json!({ "oauth": { "clientRegistration": "cimd", "clientId": "pi" } }),
				"oauth.clientRegistration \"cimd\" cannot be combined with oauth.clientId or oauth.clientName",
			),
			(
				json!({ "oauth": { "clientRegistration": "cimd", "clientName": "pi" } }),
				"oauth.clientRegistration \"cimd\" cannot be combined with oauth.clientId or oauth.clientName",
			),
			(
				json!({ "oauth": { "clientRegistration": "cimd", "callbackUrl": "http://[::1]:9000/callback" } }),
				"oauth.clientRegistration \"cimd\" requires oauth.callbackUrl on localhost or 127.0.0.1 with path /callback",
			),
			(
				json!({ "oauth": { "clientRegistration": "cimd", "callbackUrl": "http://localhost:9000/other" } }),
				"oauth.clientRegistration \"cimd\" requires oauth.callbackUrl on localhost or 127.0.0.1 with path /callback",
			),
			(
				json!({ "oauth": { "authServerMetadataUrl": "http://example.com/.well-known/oauth-authorization-server" } }),
				"oauth.authServerMetadataUrl must be an https URL, or http on localhost, 127.0.0.1, or [::1]",
			),
			(
				json!({ "oauth": { "authServerMetadataUrl": "not a url" } }),
				"oauth.authServerMetadataUrl must be an https URL, or http on localhost, 127.0.0.1, or [::1]",
			),
			(
				json!({ "oauth": { "authServerMetadataUrl": 5 } }),
				"oauth.authServerMetadataUrl must be an https URL, or http on localhost, 127.0.0.1, or [::1]",
			),
		];
        for (extra, expected) in cases {
            let mut raw = url.clone();
            raw.as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("object").clone());
            assert_eq!(
                validate("x", raw),
                Err(format!("server \"x\": {expected}")),
                "{extra}"
            );
        }
    }

    #[test]
    fn accepts_valid_oauth() {
        let config = validate(
			"x",
			json!({
				"url": "https://example.com/mcp",
				"oauth": {
					"clientId": "pi",
					"clientSecret": "!secret-cmd",
					"callbackPort": 8080,
					"callbackUrl": "http://localhost:8080/callback",
					"scope": "read write",
					"clientName": "pi",
					"clientRegistration": "dcr",
					"authServerMetadataUrl": "https://example.com/.well-known/oauth-authorization-server"
				}
			}),
		)
		.expect("valid");
        let McpServerConfig::Http(config) = config else {
            panic!("http");
        };
        let oauth = config.oauth.expect("oauth");
        assert_eq!(oauth.client_id.as_deref(), Some("pi"));
        assert_eq!(oauth.callback_port, Some(8080));
        assert_eq!(oauth.client_registration, Some(McpClientRegistration::Dcr));

        // `cimd` with the default callback path is fine.
        assert!(
			validate(
				"x",
				json!({
					"url": "https://example.com/mcp",
					"oauth": { "clientRegistration": "cimd", "callbackUrl": "http://localhost:9000/callback" }
				})
			)
			.is_ok()
		);
        // Matching ports are fine.
        assert!(validate(
            "x",
            json!({
                "url": "https://example.com/mcp",
                "oauth": { "callbackPort": 9000, "callbackUrl": "http://localhost:9000/callback" }
            })
        )
        .is_ok());
    }

    // -- successful validation -----------------------------------------------------

    #[test]
    fn validates_stdio_config() {
        let config = validate(
            "dev-radius",
            json!({
                "command": "npx",
                "args": ["-y", "radius", "serve"],
                "env": { "A": "1", "B": "x=y" },
                "cwd": "/tmp",
                "exposure": "direct",
                "description": "Radius",
                "toolExposure": { "one": "hidden" },
                "enabled": false,
                "timeout": 120
            }),
        )
        .expect("valid");
        let McpServerConfig::Stdio(config) = &config else {
            panic!("stdio");
        };
        assert_eq!(config.command, "npx");
        assert_eq!(
            config.args.as_ref().expect("args"),
            &["-y".to_string(), "radius".to_string(), "serve".to_string()]
        );
        assert_eq!(
            config.env.as_ref().expect("env").get("B"),
            Some(&"x=y".to_string())
        );
        assert_eq!(config.cwd.as_deref(), Some("/tmp"));
        assert_eq!(config.exposure, Some(McpExposure::Direct));
        assert_eq!(config.enabled, Some(false));
        assert_eq!(config.timeout.as_ref().and_then(|t| t.as_i64()), Some(120));
        // Absent `type` stays absent on the wire.
        let written = serde_json::to_value(config).expect("serialize");
        assert!(written.get("type").is_none());
        assert!(written.get("exposure").is_some());

        // Explicit `type: "stdio"` round-trips.
        let config = validate("x", json!({ "type": "stdio", "command": "c" })).expect("valid");
        let McpServerConfig::Stdio(config) = config else {
            panic!("stdio");
        };
        assert_eq!(config.r#type, Some(McpServerType::Stdio));
    }

    #[test]
    fn validates_http_config() {
        let config = validate(
            "docs",
            json!({
                "type": "streamable-http",
                "url": "http://localhost:3000/mcp",
                "headers": { "X-Team": "core" },
                "exposure": "codemode-deferred",
                "timeout": 0.5
            }),
        )
        .expect("valid");
        let McpServerConfig::Http(config) = &config else {
            panic!("http");
        };
        assert_eq!(config.url, "http://localhost:3000/mcp");
        assert_eq!(config.r#type, Some(McpServerType::StreamableHttp));
        assert_eq!(
            config.headers.as_ref().expect("headers").get("X-Team"),
            Some(&"core".to_string())
        );
        assert_eq!(config.exposure, Some(McpExposure::Codemode));
        assert_eq!(config.timeout.as_ref().and_then(|t| t.as_f64()), Some(0.5));

        // Minimal http config; absent fields stay absent.
        let config = validate("x", json!({ "url": "https://example.com/mcp" })).expect("valid");
        let McpServerConfig::Http(config) = &config else {
            panic!("http");
        };
        assert_eq!(config.r#type, None);
        assert_eq!(config.headers, None);
        assert_eq!(config.oauth, None);
        let written = serde_json::to_value(config).expect("serialize");
        assert!(written.get("type").is_none());

        // An http config with `auth` and header/oauth extras.
        let config = validate(
            "x",
            json!({
                "url": "https://example.com/mcp",
                "headers": {},
                "auth": { "provider": "openai" }
            }),
        )
        .expect("valid");
        let McpServerConfig::Http(config) = config else {
            panic!("http");
        };
        assert_eq!(
            config.auth,
            Some(McpAuthConfig {
                provider: "openai".to_string()
            })
        );

        // Round-trip through serde keeps the typed config.
        let parsed: McpServerConfig =
            serde_json::from_value(serde_json::to_value(&config).expect("serialize"))
                .expect("parse");
        assert_eq!(parsed, McpServerConfig::Http(config));
    }

    // -- tool exposure ------------------------------------------------------------

    #[test]
    fn exact_tool_name_beats_patterns() {
        let config = validate(
            "x",
            json!({
                "command": "c",
                "toolExposure": { "a*": "hidden", "ab": "direct" }
            }),
        )
        .expect("valid");
        assert_eq!(get_mcp_tool_exposure(&config, "ab"), McpExposure::Direct);
        assert_eq!(get_mcp_tool_exposure(&config, "ax"), McpExposure::Hidden);
    }

    #[test]
    fn first_pattern_in_insertion_order_wins() {
        let config = validate(
            "x",
            json!({
                "command": "c",
                "toolExposure": { "a*": "hidden", "a*x": "direct" }
            }),
        )
        .expect("valid");
        // "a*" was inserted first, so it wins even though "a*x" is more specific.
        assert_eq!(get_mcp_tool_exposure(&config, "ax"), McpExposure::Hidden);
        // Insertion order, not alphabetical order: "z*" sorts after "a*" but was
        // inserted first.
        let config = validate(
            "x",
            json!({
                "command": "c",
                "toolExposure": { "z*": "direct", "a*": "hidden" }
            }),
        )
        .expect("valid");
        assert_eq!(get_mcp_tool_exposure(&config, "ax"), McpExposure::Hidden);
        assert_eq!(get_mcp_tool_exposure(&config, "zx"), McpExposure::Direct);
    }

    #[test]
    fn tool_exposure_falls_back_to_server_exposure_then_codemode() {
        let config = validate("x", json!({ "command": "c", "exposure": "hidden" })).expect("valid");
        assert_eq!(get_mcp_tool_exposure(&config, "any"), McpExposure::Hidden);
        let config = validate("x", json!({ "command": "c" })).expect("valid");
        assert_eq!(get_mcp_tool_exposure(&config, "any"), McpExposure::Codemode);
        // Patterns without `*` only match exactly.
        let config = validate(
            "x",
            json!({ "command": "c", "toolExposure": { "ab": "direct" } }),
        )
        .expect("valid");
        assert_eq!(get_mcp_tool_exposure(&config, "ab"), McpExposure::Direct);
        assert_eq!(get_mcp_tool_exposure(&config, "abc"), McpExposure::Codemode);
    }

    #[test]
    fn tool_patterns_use_literal_characters_and_star() {
        let config = validate(
            "x",
            json!({ "command": "c", "toolExposure": { "a.b*": "direct" } }),
        )
        .expect("valid");
        assert_eq!(
            get_mcp_tool_exposure(&config, "a.b-last"),
            McpExposure::Direct
        );
        // `.` is literal, not "any character".
        assert_eq!(
            get_mcp_tool_exposure(&config, "axb-last"),
            McpExposure::Codemode
        );

        let config = validate(
            "x",
            json!({ "command": "c", "toolExposure": { "c+(d)*e": "direct" } }),
        )
        .expect("valid");
        assert_eq!(
            get_mcp_tool_exposure(&config, "c+(d)-1-e"),
            McpExposure::Direct
        );
        assert_eq!(
            get_mcp_tool_exposure(&config, "cdd-1-e"),
            McpExposure::Codemode
        );

        let config = validate(
            "x",
            json!({ "command": "c", "toolExposure": { "*": "hidden" } }),
        )
        .expect("valid");
        assert_eq!(get_mcp_tool_exposure(&config, ""), McpExposure::Hidden);
        assert_eq!(
            get_mcp_tool_exposure(&config, "anything"),
            McpExposure::Hidden
        );
    }

    // -- namespace ------------------------------------------------------------------

    #[test]
    fn converts_server_name_to_namespace() {
        assert_eq!(mcp_namespace("radius"), "mcp__radius");
        assert_eq!(mcp_namespace("dev-radius"), "mcp__dev_radius");
        assert_eq!(mcp_namespace("a-b-c"), "mcp__a_b_c");
    }

    // -- loopback redirect URIs --------------------------------------------------------

    #[test]
    fn loopback_redirect_uris() {
        assert!(is_loopback_redirect_uri("http://localhost/callback"));
        assert!(is_loopback_redirect_uri("http://127.0.0.1:8080/callback"));
        assert!(is_loopback_redirect_uri("http://[::1]/callback"));
        // JS `url.search === ""` also holds for a bare `?` (and `#`).
        assert!(is_loopback_redirect_uri("http://localhost/callback?"));
        assert!(is_loopback_redirect_uri("http://localhost/callback#"));
        assert!(!is_loopback_redirect_uri("https://localhost/callback"));
        assert!(!is_loopback_redirect_uri("http://example.com/callback"));
        assert!(!is_loopback_redirect_uri(
            "http://localhost/callback?code=1"
        ));
        assert!(!is_loopback_redirect_uri("http://localhost/callback#frag"));
        assert!(!is_loopback_redirect_uri("not a url"));
        assert!(!is_loopback_redirect_uri(""));
    }

    // -- registry -------------------------------------------------------------------

    #[test]
    fn registry_registers_replaces_and_lists_in_order() {
        let registry = McpServerRegistry::default();
        let server = |name: &str, command: &str| RegisteredMcpServer {
            name: name.to_string(),
            config: validate(name, json!({ "command": command })).expect("valid"),
            extension_path: format!("/{name}.js"),
        };
        registry.register(server("a", "a-cmd"));
        registry.register(server("b", "b-cmd"));
        registry.register(server("c", "c-cmd"));
        // Replace keeps the original position, like the JS `Map`.
        registry.register(server("b", "b-cmd-2"));
        let listed_names = registry.list();
        let names: Vec<&str> = listed_names
            .iter()
            .map(|server| server.name.as_str())
            .collect();
        assert_eq!(names, ["a", "b", "c"]);
        let listed = registry.list();
        let McpServerConfig::Stdio(config) = &listed[1].config else {
            panic!("stdio");
        };
        assert_eq!(config.command, "b-cmd-2");
        assert_eq!(registry.get("c").expect("c").extension_path, "/c.js");
        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn registry_unregisters_only_owned_servers() {
        let registry = McpServerRegistry::default();
        let server = |name: &str, extension: &str| RegisteredMcpServer {
            name: name.to_string(),
            config: validate(name, json!({ "command": "c" })).expect("valid"),
            extension_path: extension.to_string(),
        };
        registry.register(server("a", "/one.js"));
        // Wrong extension path: left alone.
        registry.unregister("a", "/two.js");
        assert!(registry.get("a").is_some());
        // Unknown name: left alone.
        registry.unregister("missing", "/one.js");
        // Owning path removes it.
        registry.unregister("a", "/one.js");
        assert!(registry.get("a").is_none());
        assert!(registry.list().is_empty());
    }

    #[test]
    fn registry_notifies_change_listener() {
        let registry = McpServerRegistry::default();
        let changes = Arc::new(AtomicUsize::new(0));
        registry.set_change_listener(Some({
            let changes = Arc::clone(&changes);
            Arc::new(move || {
                changes.fetch_add(1, Ordering::SeqCst);
            })
        }));
        let server = RegisteredMcpServer {
            name: "a".to_string(),
            config: validate("a", json!({ "command": "c" })).expect("valid"),
            extension_path: "/one.js".to_string(),
        };
        registry.register(server.clone());
        registry.register(server);
        assert_eq!(changes.load(Ordering::SeqCst), 2);
        registry.unregister("a", "/two.js");
        assert_eq!(changes.load(Ordering::SeqCst), 2);
        registry.unregister("a", "/one.js");
        assert_eq!(changes.load(Ordering::SeqCst), 3);
        // Detaching the listener stops the notifications.
        registry.set_change_listener(None);
        registry.register(RegisteredMcpServer {
            name: "a".to_string(),
            config: validate("a", json!({ "command": "c" })).expect("valid"),
            extension_path: "/one.js".to_string(),
        });
        assert_eq!(changes.load(Ordering::SeqCst), 3);
    }
}
