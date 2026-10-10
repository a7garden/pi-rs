//! Port of `packages/coding-agent/src/extensions/mcp/config.ts` and the
//! config/validation half of `core/mcp-servers.ts` (the registry half lives in
//! [`crate::core::extensions::types::McpServerRegistry`]).
//!
//! Server configs are carried as raw JSON (`McpServerConfig` is a newtype
//! over `serde_json::Value` per core extensions D30); validation mirrors the
//! upstream hand-rolled checks 1:1 and returns the config with exposure
//! aliases (`codemode-deferred` → `codemode`) resolved.
//!
//! # Port deviations
//!
//! - **D82.** `JSON.stringify(parsed, null, indent)` → a local pretty printer
//!   over `serde_json` (with `preserve_order`, so JS object key order is
//!   kept); string escaping differs in cosmetic cases only.
//! - **D85.** JS `URL.canParse`/`new URL` → the `url` crate; upstream's
//!   `[::1]` hostname spelling is matched literally (the `url` crate
//!   normalizes it to `[::1]` in `host_str`).

use std::path::Path;

use serde_json::{Map, Value};

use crate::config::CONFIG_DIR_NAME;

/// Upstream `McpExposure`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum McpExposure {
    #[serde(rename = "codemode")]
    Codemode,
    #[serde(rename = "deferred")]
    Deferred,
    #[serde(rename = "direct")]
    Direct,
    #[serde(rename = "hidden")]
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

/// Older exposure names accepted in configs (upstream `MCP_EXPOSURE_ALIASES`).
fn resolve_exposure_alias(value: &str) -> &str {
    if value == "codemode-deferred" {
        "codemode"
    } else {
        value
    }
}

fn is_exposure(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|text| MCP_EXPOSURES.contains(&text))
}

/// Namespace of a server's tools: `mcp__<server>` with `-` replaced by `_`
/// (`mcpNamespace`).
pub fn mcp_namespace(server: &str) -> String {
    format!("mcp__{}", server.replace('-', "_"))
}

const LOOPBACK_HOSTS: [&str; 3] = ["localhost", "127.0.0.1", "[::1]"];

/// Whether a redirect URI can be served by pi's loopback callback server
/// (`isLoopbackRedirectUri`).
pub fn is_loopback_redirect_uri(value: &str) -> bool {
    let Ok(url) = url::Url::parse(value) else {
        return false;
    };
    url.scheme() == "http"
        && LOOPBACK_HOSTS.contains(&url.host_str().unwrap_or_default())
        && url.query().is_none()
        && url.fragment().is_none()
}

fn is_record(value: &Value) -> bool {
    value.is_object()
}

fn is_string_record(value: &Value) -> bool {
    value
        .as_object()
        .is_some_and(|object| object.values().all(|entry| entry.is_string()))
}

/// The exposure an alias stands for; other values are returned unchanged.
fn resolve_alias_value(value: &Value) -> Value {
    match value.as_str() {
        Some(text) => Value::String(resolve_exposure_alias(text).to_string()),
        None => value.clone(),
    }
}

/// A copy of the server entry with exposure aliases replaced by their current
/// names (`resolveExposureAliases`).
fn resolve_exposure_aliases(value: &Map<String, Value>) -> Map<String, Value> {
    let mut resolved = value.clone();
    if let Some(exposure) = value.get("exposure") {
        resolved.insert("exposure".to_string(), resolve_alias_value(exposure));
    }
    if let Some(tool_exposure) = value.get("toolExposure").filter(|value| is_record(value)) {
        resolved.insert(
            "toolExposure".to_string(),
            Value::Object(
                tool_exposure
                    .as_object()
                    .expect("checked record")
                    .iter()
                    .map(|(tool, entry)| (tool.clone(), resolve_alias_value(entry)))
                    .collect(),
            ),
        );
    }
    resolved
}

/// Exposure of one tool of a server: its `toolExposure` entry, else the
/// server's `exposure` (`getMcpToolExposure`). Patterns use `*` as a wildcard
/// for any characters; an exact name wins, then the first matching pattern in
/// object order.
pub fn get_mcp_tool_exposure(config: &Value, tool_name: &str) -> McpExposure {
    let default_exposure = || {
        config
            .get("exposure")
            .and_then(|value| value.as_str())
            .and_then(McpExposure::parse)
            .unwrap_or(McpExposure::Codemode)
    };
    let Some(overrides) = config
        .get("toolExposure")
        .and_then(|value| value.as_object())
    else {
        return default_exposure();
    };
    if let Some(exact) = overrides.get(tool_name) {
        if let Some(exposure) = exact.as_str().and_then(McpExposure::parse) {
            return exposure;
        }
    }
    for (pattern, exposure) in overrides {
        if pattern.contains('*') && tool_pattern_matches(pattern, tool_name) {
            if let Some(exposure) = exposure.as_str().and_then(McpExposure::parse) {
                return exposure;
            }
        }
    }
    default_exposure()
}

/// `toolPatternRegExp` without a regex engine: `*` matches any characters,
/// every other character is literal, and the pattern must match the whole tool
/// name.
fn tool_pattern_matches(pattern: &str, tool_name: &str) -> bool {
    let segments: Vec<&str> = pattern.split('*').collect();
    let mut position = 0usize;
    for (index, segment) in segments.iter().enumerate() {
        let last = index == segments.len() - 1;
        if index == 0 {
            // Anchored `^`: the first segment must be a prefix.
            if !tool_name.starts_with(*segment) {
                return false;
            }
            position = segment.len();
            continue;
        }
        if last {
            // `.*<segment>$`: the segment may start anywhere at or after
            // `position`; an empty segment matches the rest unconditionally.
            return segment.is_empty()
                || (tool_name.len() >= position + segment.len()
                    && tool_name[position..].ends_with(segment));
        }
        if segment.is_empty() {
            continue;
        }
        match tool_name[position..].find(segment) {
            Some(found) => position += found + segment.len(),
            None => return false,
        }
    }
    true
}

/// Validate one server entry of the `mcpServers` shape. Returns a copy of the
/// config with exposure aliases resolved, or the upstream error message
/// (`validateMcpServerConfig`).
pub fn validate_mcp_server_config(name: &str, raw: &Value) -> Result<Value, String> {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(format!(
            "invalid server name \"{name}\" (use letters, digits, \"_\" and \"-\")"
        ));
    }
    if !is_record(raw) {
        return Err(format!("server \"{name}\" must be an object"));
    }
    let value = resolve_exposure_aliases(raw.as_object().expect("checked record"));
    let exposures = MCP_EXPOSURES
        .iter()
        .map(|value| format!("\"{value}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let exposure = value.get("exposure");
    if exposure.is_some() && !is_exposure(exposure.expect("checked")) {
        return Err(format!(
            "server \"{name}\": exposure must be one of {exposures}"
        ));
    }
    if let Some(tool_exposure) = value.get("toolExposure") {
        if !is_record(tool_exposure) {
            return Err(format!(
                "server \"{name}\": toolExposure must map tool names to exposures"
            ));
        }
        for (tool, entry) in tool_exposure.as_object().expect("checked record") {
            if !is_exposure(entry) {
                return Err(format!(
                    "server \"{name}\": toolExposure \"{tool}\" must be one of {exposures}"
                ));
            }
        }
    }
    if let Some(enabled) = value.get("enabled") {
        if !enabled.is_boolean() {
            return Err(format!("server \"{name}\": enabled must be a boolean"));
        }
    }
    if let Some(description) = value.get("description") {
        if !description.is_string() {
            return Err(format!("server \"{name}\": description must be a string"));
        }
    }
    if let Some(timeout) = value.get("timeout") {
        let valid = timeout.as_f64().is_some_and(|timeout| timeout > 0.0);
        if !valid {
            return Err(format!(
                "server \"{name}\": timeout must be a positive number of seconds"
            ));
        }
    }
    if value.get("type").and_then(|value| value.as_str()) == Some("sse") {
        return Err(format!(
            "server \"{name}\": legacy SSE transport is not supported; use the streamable HTTP URL"
        ));
    }
    let type_matches = matches!(
        value.get("type").and_then(|value| value.as_str()),
        None | Some("http") | Some("streamable-http")
    );
    if let Some(url_text) = value.get("url").and_then(|value| value.as_str()) {
        if !type_matches {
            return http_or_stdio_fallback(name, &Value::Object(value.clone()));
        }
        let parsed = url::Url::parse(url_text).ok();
        let valid = parsed
            .as_ref()
            .is_some_and(|url| url.scheme() == "http" || url.scheme() == "https");
        if !valid {
            return Err(format!(
                "server \"{name}\": url must be an http or https URL"
            ));
        }
        if let Some(headers) = value.get("headers") {
            if !is_string_record(headers) {
                return Err(format!(
                    "server \"{name}\": headers must map names to strings"
                ));
            }
        }
        if let Some(oauth) = value.get("oauth") {
            if let Some(message) = validate_oauth(oauth) {
                return Err(format!("server \"{name}\": {message}"));
            }
        }
        if let Some(auth) = value.get("auth") {
            let provider_ok = auth
                .as_object()
                .and_then(|auth| auth.get("provider"))
                .and_then(|provider| provider.as_str())
                .is_some_and(|provider| !provider.is_empty());
            if !provider_ok {
                return Err(format!(
                    "server \"{name}\": auth.provider must be a provider name"
                ));
            }
            let url = parsed.expect("validated above");
            let secure = url.scheme() == "https"
                || LOOPBACK_HOSTS.contains(&url.host_str().unwrap_or_default());
            if !secure {
                return Err(format!(
					"server \"{name}\": auth requires an https URL, or http on localhost, 127.0.0.1, or [::1]"
				));
            }
        }
        return Ok(Value::Object(value));
    }
    let command_matches = matches!(
        value.get("type").and_then(|value| value.as_str()),
        None | Some("stdio")
    );
    if let Some(command) = value.get("command").and_then(|value| value.as_str()) {
        let _ = command;
        if !command_matches {
            return Err(format!(
                "server \"{name}\" needs either \"command\" (stdio) or \"url\" (streamable HTTP)"
            ));
        }
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
            if !cwd.is_string() {
                return Err(format!("server \"{name}\": cwd must be a string"));
            }
        }
        return Ok(Value::Object(value));
    }
    // A non-matching `type` with `command` or `url` falls through to the
    // generic error, like upstream.
    http_or_stdio_fallback(name, &Value::Object(value))
}

fn http_or_stdio_fallback(name: &str, _value: &Value) -> Result<Value, String> {
    Err(format!(
        "server \"{name}\" needs either \"command\" (stdio) or \"url\" (streamable HTTP)"
    ))
}

/// Upstream `validateOAuth`: checks the OAuth client settings of an HTTP
/// server; returns the upstream error message.
fn validate_oauth(value: &Value) -> Option<String> {
    if !is_record(value) {
        return Some("oauth must be an object".to_string());
    }
    let oauth = value.as_object().expect("checked record");
    if let Some(client_id) = oauth.get("clientId") {
        if !client_id.is_string() {
            return Some("oauth.clientId must be a string".to_string());
        }
    }
    if let Some(client_secret) = oauth.get("clientSecret") {
        if !client_secret.is_string() {
            return Some("oauth.clientSecret must be a string".to_string());
        }
    }
    let port = oauth.get("callbackPort");
    if let Some(port) = port {
        let valid = port
            .as_i64()
            .is_some_and(|port| (1..=65535).contains(&port));
        if !valid {
            return Some("oauth.callbackPort must be a port number".to_string());
        }
    }
    if let Some(callback_url) = oauth.get("callbackUrl") {
        let valid = callback_url.as_str().is_some_and(is_loopback_redirect_uri);
        if !valid {
            return Some(
				"oauth.callbackUrl must be an http URI on localhost, 127.0.0.1, or [::1] without query or fragment"
					.to_string(),
			);
        }
        let url_port = url::Url::parse(callback_url.as_str().expect("checked string"))
            .ok()
            .and_then(|url| url.port());
        if let (Some(url_port), Some(port)) = (
            url_port,
            port.and_then(|port| port.as_u64()).map(|port| port as u16),
        ) {
            if url_port != port {
                return Some(
                    "oauth.callbackUrl and oauth.callbackPort name different ports".to_string(),
                );
            }
        }
    }
    if let Some(scope) = oauth.get("scope") {
        if !scope.is_string() {
            return Some("oauth.scope must be a string".to_string());
        }
    }
    if let Some(client_name) = oauth.get("clientName") {
        let valid = client_name
            .as_str()
            .is_some_and(|name| !name.trim().is_empty());
        if !valid {
            return Some("oauth.clientName must be a non-empty string".to_string());
        }
    }
    if let Some(registration) = oauth
        .get("clientRegistration")
        .and_then(|value| value.as_str())
    {
        if registration != "dcr" {
            if registration != "cimd" {
                return Some("oauth.clientRegistration must be \"dcr\" or \"cimd\"".to_string());
            }
            if oauth.get("clientId").is_some() || oauth.get("clientName").is_some() {
                return Some(
					"oauth.clientRegistration \"cimd\" cannot be combined with oauth.clientId or oauth.clientName"
						.to_string(),
				);
            }
            if let Some(callback) = oauth.get("callbackUrl").and_then(|value| value.as_str()) {
                if let Ok(url) = url::Url::parse(callback) {
                    let hostname = url.host_str().unwrap_or_default();
                    if hostname == "[::1]" || url.path() != "/callback" {
                        return Some(
							"oauth.clientRegistration \"cimd\" requires oauth.callbackUrl on localhost or 127.0.0.1 with path /callback"
								.to_string(),
						);
                    }
                }
            }
        }
    }
    if let Some(metadata_url) = oauth.get("authServerMetadataUrl") {
        let parsed = metadata_url
            .as_str()
            .and_then(|value| url::Url::parse(value).ok());
        let valid = parsed.as_ref().is_some_and(|url| {
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

/// Upstream `McpServerEntry.scope`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpServerScope {
    Global,
    Project,
    Extension,
}

impl McpServerScope {
    pub fn as_str(&self) -> &'static str {
        match self {
            McpServerScope::Global => "global",
            McpServerScope::Project => "project",
            McpServerScope::Extension => "extension",
        }
    }
}

/// Upstream `McpServerEntry`.
#[derive(Debug, Clone, PartialEq)]
pub struct McpServerEntry {
    pub name: String,
    pub config: Value,
    /// Config file that defined the entry, or the path of the extension that
    /// registered it.
    pub source: String,
    /// Scope of the defining file.
    pub scope: Option<McpServerScope>,
    /// Project `mcp.json` with an override of this global server's `enabled`,
    /// `exposure`, or `toolExposure`.
    pub r#override: Option<String>,
}

/// Upstream `LoadedMcpConfig`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoadedMcpConfig {
    pub servers: Vec<McpServerEntry>,
    /// Activate the codemode tool when `codemode` servers connect. Default: true.
    pub auto_enable_codemode: Option<bool>,
    pub errors: Vec<String>,
    /// The project `mcp.json` when the project is trusted, where `/mcp` saves
    /// project overrides.
    pub project_config: Option<String>,
}

const OVERRIDE_KEYS: [&str; 3] = ["enabled", "exposure", "toolExposure"];

/// Whether an entry overrides a server defined elsewhere instead of defining
/// one (`isOverride`).
fn is_override(value: &Map<String, Value>) -> bool {
    value.get("command").is_none() && value.get("url").is_none() && value.get("type").is_none()
}

#[derive(Default)]
struct McpConfigState {
    servers: Vec<(String, McpServerEntry)>,
    auto_enable_codemode: Option<bool>,
    errors: Vec<String>,
}

impl McpConfigState {
    fn get(&self, name: &str) -> Option<&McpServerEntry> {
        self.servers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, entry)| entry)
    }

    fn insert(&mut self, name: &str, entry: McpServerEntry) {
        if let Some(slot) = self.servers.iter_mut().find(|(key, _)| key == name) {
            slot.1 = entry;
        } else {
            self.servers.push((name.to_string(), entry));
        }
    }
}

fn read_config_file(path: &str, scope: McpServerScope, state: &mut McpConfigState) {
    if !Path::new(path).exists() {
        return;
    }
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            state.errors.push(format!("{path}: {error}"));
            return;
        }
    };
    let parsed: Value = match serde_json::from_str(&text) {
        Ok(parsed) => parsed,
        Err(error) => {
            state.errors.push(format!("{path}: {error}"));
            return;
        }
    };
    if !is_record(&parsed)
        || (parsed.get("mcpServers").is_some()
            && !is_record(parsed.get("mcpServers").expect("checked")))
    {
        state.errors.push(format!(
            "{path}: expected an object with an \"mcpServers\" object"
        ));
        return;
    }
    match parsed.get("autoEnableCodemode") {
        Some(value) if value.is_boolean() => {
            state.auto_enable_codemode = value.as_bool();
        }
        Some(_) => state
            .errors
            .push(format!("{path}: autoEnableCodemode must be a boolean")),
        None => {}
    }
    let Some(servers) = parsed.get("mcpServers").and_then(|value| value.as_object()) else {
        return;
    };
    for (name, value) in servers {
        if scope == McpServerScope::Project
            && is_record(value)
            && is_override(value.as_object().expect("checked"))
        {
            let object = value.as_object().expect("checked record");
            let base = state.get(name).cloned();
            let extra: Vec<&String> = object
                .keys()
                .filter(|key| !OVERRIDE_KEYS.contains(&key.as_str()))
                .collect();
            let Some(base) = base else {
                state.errors.push(format!(
					"{path}: server \"{name}\" needs \"command\" or \"url\", or a global server to override"
				));
                continue;
            };
            if !extra.is_empty() {
                state.errors.push(format!(
                    "{path}: server \"{name}\": an override can only set {}",
                    OVERRIDE_KEYS.join(", ")
                ));
                continue;
            }
            let mut merged = base.config.clone();
            if let Some(merged_object) = merged.as_object_mut() {
                for (key, entry) in object {
                    merged_object.insert(key.clone(), entry.clone());
                }
            }
            match validate_mcp_server_config(name, &merged) {
                Ok(config) => state.insert(
                    name,
                    McpServerEntry {
                        config,
                        r#override: Some(path.to_string()),
                        ..base
                    },
                ),
                Err(message) => state.errors.push(format!("{path}: {message}")),
            }
            continue;
        }
        let config = match validate_mcp_server_config(name, value) {
            Ok(config) => config,
            Err(message) => {
                state.errors.push(format!("{path}: {message}"));
                continue;
            }
        };
        // Names that differ only in `-` and `_` would share a namespace.
        if let Some((clash, _)) = state
            .servers
            .iter()
            .find(|(other, _)| other != name && mcp_namespace(other) == mcp_namespace(name))
            .map(|(key, entry)| (key.clone(), entry.clone()))
        {
            state.errors.push(format!(
                "{path}: server \"{name}\" conflicts with \"{clash}\""
            ));
            continue;
        }
        if scope == McpServerScope::Project
            && config.get("url").is_some()
            && config.get("auth").is_some()
        {
            state.errors.push(format!(
                "{path}: server \"{name}\": auth is only allowed in the global mcp.json"
            ));
            continue;
        }
        state.insert(
            name,
            McpServerEntry {
                name: name.clone(),
                config,
                source: path.to_string(),
                scope: Some(scope),
                r#override: None,
            },
        );
    }
}

/// Load global and (when trusted) project MCP configuration. Disabled servers
/// are included with `enabled: false`, so they can be enabled again
/// (`loadMcpConfig`).
pub fn load_mcp_config(agent_dir: &str, cwd: &str, project_trusted: bool) -> LoadedMcpConfig {
    let mut state = McpConfigState::default();
    read_config_file(
        &join_path(agent_dir, "mcp.json"),
        McpServerScope::Global,
        &mut state,
    );
    let project_config = if project_trusted {
        Some(join_path(&join_path(cwd, CONFIG_DIR_NAME), "mcp.json"))
    } else {
        None
    };
    if let Some(project_config) = &project_config {
        read_config_file(project_config, McpServerScope::Project, &mut state);
    }
    LoadedMcpConfig {
        servers: state.servers.into_iter().map(|(_, entry)| entry).collect(),
        auto_enable_codemode: state.auto_enable_codemode,
        errors: state.errors,
        project_config,
    }
}

/// Settings `/mcp` changes (`McpServerConfigPatch`). `enabled: true` and
/// `exposure: "codemode"` are the defaults and remove the key.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct McpServerConfigPatch {
    pub enabled: Option<bool>,
    pub exposure: Option<McpExposure>,
}

/// Change one server's settings in the `mcp.json` that defines or overrides it
/// (`updateMcpServerConfig`). With `override`, a missing entry is added as an
/// override. Overrides keep default values, since they replace the global
/// server's. Other content is kept; the file is rewritten with its
/// indentation.
pub fn update_mcp_server_config(
    path: &str,
    name: &str,
    patch: McpServerConfigPatch,
    r#override: bool,
) -> Result<(), String> {
    edit_mcp_servers(path, |servers, parsed| {
        let existing = servers
            .as_ref()
            .and_then(|servers| servers.get(name))
            .cloned();
        let mut server = match existing {
            Some(server) => server,
            None => {
                if !r#override {
                    return Err(format!("{path} does not define MCP server \"{name}\""));
                }
                let server = Map::new();
                let mut next_servers = servers.cloned().unwrap_or_default();
                next_servers.insert(name.to_string(), Value::Object(server.clone()));
                parsed.insert("mcpServers".to_string(), Value::Object(next_servers));
                Value::Object(server)
            }
        };
        let object = server.as_object_mut().expect("record");
        let keep_defaults = is_override(object);
        if let Some(enabled) = patch.enabled {
            if enabled && !keep_defaults {
                // JS `delete` keeps the order of the remaining keys.
                object.shift_remove("enabled");
            } else {
                object.insert("enabled".to_string(), Value::Bool(enabled));
            }
        }
        if let Some(exposure) = patch.exposure {
            if exposure == McpExposure::Codemode && !keep_defaults {
                object.shift_remove("exposure");
            } else {
                object.insert(
                    "exposure".to_string(),
                    Value::String(exposure.as_str().to_string()),
                );
            }
        }
        // Upstream mutates the entry inside `parsed.mcpServers` in place; the
        // port clones the entry, so it stores the mutated value back.
        if let Some(servers) = parsed
            .get_mut("mcpServers")
            .and_then(|value| value.as_object_mut())
        {
            servers.insert(name.to_string(), server);
        }
        Ok(true)
    })
    .map(|_| ())
}

/// Add a server to an `mcp.json`, creating the file when missing. An existing
/// entry with the same name is replaced. Returns true when an entry was
/// replaced (`addMcpServerConfig`).
pub fn add_mcp_server_config(path: &str, name: &str, config: &Value) -> Result<bool, String> {
    let mut replaced = false;
    edit_mcp_servers(path, |servers, parsed| {
        let mut target = servers.cloned().unwrap_or_default();
        replaced = target.contains_key(name);
        target.insert(name.to_string(), config.clone());
        parsed.insert("mcpServers".to_string(), Value::Object(target));
        Ok(true)
    })?;
    Ok(replaced)
}

/// Remove a server from an `mcp.json`. Returns false when the file does not
/// define it (`removeMcpServerConfig`).
pub fn remove_mcp_server_config(path: &str, name: &str) -> Result<bool, String> {
    if !Path::new(path).exists() {
        return Ok(false);
    }
    let mut removed = false;
    edit_mcp_servers(path, |servers, parsed| {
        let Some(servers) = servers else {
            return Ok(false);
        };
        if !servers.contains_key(name) {
            return Ok(false);
        }
        if let Some(servers) = parsed
            .get_mut("mcpServers")
            .and_then(|value| value.as_object_mut())
        {
            // JS `delete` keeps the order of the remaining keys; serde_json's
            // `remove` swaps the last key into the freed slot.
            servers.shift_remove(name);
        }
        removed = true;
        Ok(true)
    })?;
    Ok(removed)
}

/// Read an `mcp.json` (an empty config when missing), let `edit` change its
/// `mcpServers`, and write it back with its indentation when `edit` returns
/// true. Other content is kept (`editMcpServers`).
fn edit_mcp_servers(
    path: &str,
    edit: impl FnOnce(Option<&Map<String, Value>>, &mut Map<String, Value>) -> Result<bool, String>,
) -> Result<(), String> {
    let text = std::fs::read_to_string(path).ok();
    let parsed: Value = match text.as_deref() {
        None => Value::Object(Map::new()),
        Some(text) => serde_json::from_str(text).map_err(|error| error.to_string())?,
    };
    if !is_record(&parsed)
        || (parsed.get("mcpServers").is_some()
            && !is_record(parsed.get("mcpServers").expect("checked")))
    {
        return Err(format!(
            "{path}: expected an object with an \"mcpServers\" object"
        ));
    }
    let mut parsed_object = parsed.as_object().expect("checked record").clone();
    let servers = parsed_object
        .get("mcpServers")
        .and_then(|value| value.as_object())
        .cloned();
    if !edit(servers.as_ref(), &mut parsed_object)? {
        return Ok(());
    }
    let indent = text.as_deref().and_then(detect_indent).unwrap_or("  ");
    if let Some(parent) = Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let rendered = render_json(&Value::Object(parsed_object), indent, 0);
    std::fs::write(path, format!("{rendered}\n")).map_err(|error| error.to_string())
}

/// Upstream `const indent = text.match(/^([ \t]+)\S/m)?.[1] || "  "`: the
/// leading whitespace of the first indented line.
fn detect_indent(text: &str) -> Option<&str> {
    for line in text.lines() {
        let trimmed = line.trim_start_matches([' ', '\t']);
        if trimmed.is_empty() || trimmed.len() == line.len() {
            continue;
        }
        return Some(&line[..line.len() - trimmed.len()]);
    }
    None
}

/// `JSON.stringify(value, null, indent)`: objects and arrays print one entry
/// per line with the given indent (and `[]`/`{}` when empty).
fn render_json(value: &Value, indent: &str, depth: usize) -> String {
    let pad = indent.repeat(depth);
    let pad_inner = indent.repeat(depth + 1);
    match value {
        Value::Object(object) if !object.is_empty() => {
            let entries = object
                .iter()
                .map(|(key, entry)| {
                    format!(
                        "{pad_inner}{}: {}",
                        serde_json::to_string(key).expect("key serialization is infallible"),
                        render_json(entry, indent, depth + 1)
                    )
                })
                .collect::<Vec<_>>()
                .join(",\n");
            format!("{{\n{entries}\n{pad}}}")
        }
        Value::Array(items) if !items.is_empty() => {
            let entries = items
                .iter()
                .map(|item| format!("{pad_inner}{}", render_json(item, indent, depth + 1)))
                .collect::<Vec<_>>()
                .join(",\n");
            format!("[\n{entries}\n{pad}]")
        }
        other => serde_json::to_string(other).expect("value serialization is infallible"),
    }
}

fn join_path(base: &str, suffix: &str) -> String {
    Path::new(base).join(suffix).to_string_lossy().into_owned()
}

#[cfg(test)]
mod config_tests {
    //! Port of `test/mcp-extension.test.ts` `describe("MCP config")`.

    use super::*;

    fn setup(global: Value, project: Value) -> (String, String) {
        let root = temp_dir("pi-mcp-config-");
        let agent_dir = format!("{root}/agent");
        let cwd = format!("{root}/project");
        std::fs::create_dir_all(&agent_dir).expect("agent dir");
        std::fs::create_dir_all(format!("{cwd}/.pi")).expect("project dir");
        std::fs::write(
            join_path(&agent_dir, "mcp.json"),
            serde_json::to_string(&global).expect("serialize"),
        )
        .expect("write global");
        std::fs::write(
            join_path(&join_path(&cwd, CONFIG_DIR_NAME), "mcp.json"),
            serde_json::to_string(&project).expect("serialize"),
        )
        .expect("write project");
        (agent_dir, cwd)
    }

    fn temp_dir(prefix: &str) -> String {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst) as u64;
        let dir = std::env::temp_dir()
            .join(format!("{prefix}{}-{id}", std::process::id()))
            .to_string_lossy()
            .into_owned();
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// Port of "merges global and trusted project servers and validates entries".
    #[test]
    fn merges_global_and_trusted_project_servers_and_validates_entries() {
        let json = |value: Value| value;
        let (agent_dir, cwd) = setup(
            json(serde_json::json!({
                "mcpServers": {
                    "shared": { "command": "global-cmd" },
                    "remote": { "url": "https://example.com/mcp", "headers": { "Authorization": "Bearer ${TOKEN}" } },
                    "off": { "command": "x", "enabled": false },
                    "bad": { "args": ["no command"] },
                    "legacy": { "type": "sse", "url": "https://example.com/sse" },
                    "badUrl": { "url": "example.com/mcp" },
                    "bad name": { "command": "x" }
                }
            })),
            json(
                serde_json::json!({ "mcpServers": { "shared": { "command": "project-cmd", "exposure": "direct" } } }),
            ),
        );

        let trusted = load_mcp_config(&agent_dir, &cwd, true);
        let summary: Vec<(String, &str)> = trusted
            .servers
            .iter()
            .map(|server| (server.name.clone(), server.scope.expect("scope").as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                ("shared".to_string(), "project"),
                ("remote".to_string(), "global"),
                ("off".to_string(), "global"),
            ]
        );
        assert_eq!(
            trusted.servers[0].config,
            serde_json::json!({ "command": "project-cmd", "exposure": "direct" })
        );
        assert_eq!(trusted.errors.len(), 4);
        assert!(trusted.errors[0].contains("server \"bad\" needs either \"command\""));
        assert!(trusted.errors[1].contains("legacy SSE transport is not supported"));
        assert!(trusted.errors[2].contains("server \"badUrl\": url must be an http or https URL"));
        assert!(trusted.errors[3].contains("invalid server name \"bad name\""));

        let untrusted = load_mcp_config(&agent_dir, &cwd, false);
        let shared = untrusted
            .servers
            .iter()
            .find(|server| server.name == "shared")
            .expect("shared");
        assert_eq!(
            shared.config,
            serde_json::json!({ "command": "global-cmd" })
        );
    }

    /// Port of #10277 "lets project entries override enabled and exposure of
    /// global servers".
    #[test]
    fn lets_project_entries_override_enabled_and_exposure_of_global_servers() {
        let (agent_dir, cwd) = setup(
            serde_json::json!({ "mcpServers": { "tools": { "command": "x", "env": { "TOKEN": "secret" } } } }),
            serde_json::json!({ "mcpServers": { "tools": { "enabled": false, "args": ["y"] }, "missing": { "enabled": false } } }),
        );
        let project = join_path(&join_path(&cwd, CONFIG_DIR_NAME), "mcp.json");
        let loaded = load_mcp_config(&agent_dir, &cwd, true);
        assert_eq!(loaded.servers.len(), 1);
        assert_eq!(loaded.servers[0].name, "tools");
        assert_eq!(loaded.servers[0].r#override, None);
        assert_eq!(
            loaded.servers[0].config,
            serde_json::json!({ "command": "x", "env": { "TOKEN": "secret" } })
        );
        assert_eq!(loaded.errors.len(), 2);
        assert!(loaded.errors[0].contains(
            "server \"tools\": an override can only set enabled, exposure, toolExposure"
        ));
        assert!(loaded.errors[1].contains(
            "server \"missing\" needs \"command\" or \"url\", or a global server to override"
        ));

        std::fs::write(
            &project,
            serde_json::to_string(
                &serde_json::json!({ "mcpServers": { "tools": { "enabled": false } } }),
            )
            .expect("serialize"),
        )
        .expect("write project");
        let loaded = load_mcp_config(&agent_dir, &cwd, true);
        assert_eq!(
            loaded.servers[0].r#override.as_deref(),
            Some(project.as_str())
        );
        assert_eq!(
            loaded.servers[0].config,
            serde_json::json!({ "command": "x", "env": { "TOKEN": "secret" }, "enabled": false })
        );

        // Overrides keep `enabled: true`, since it replaces the global value.
        update_mcp_server_config(
            &project,
            "tools",
            McpServerConfigPatch {
                enabled: Some(true),
                exposure: None,
            },
            false,
        )
        .expect("update");
        let project_config: Value =
            serde_json::from_str(&std::fs::read_to_string(&project).expect("read")).expect("parse");
        assert_eq!(
            project_config
                .pointer("/mcpServers/tools")
                .expect("tools entry"),
            &serde_json::json!({ "enabled": true })
        );
    }

    /// Port of the #10239 regression "rejects server names that differ only in
    /// - and _".
    #[test]
    fn rejects_server_names_that_differ_only_in_dash_and_underscore() {
        let (agent_dir, cwd) = setup(
            serde_json::json!({ "mcpServers": { "work-files": { "command": "a" }, "work_files": { "command": "b" } } }),
            serde_json::json!({}),
        );
        let loaded = load_mcp_config(&agent_dir, &cwd, false);
        let names: Vec<&str> = loaded
            .servers
            .iter()
            .map(|server| server.name.as_str())
            .collect();
        assert_eq!(names, vec!["work-files"]);
        assert_eq!(loaded.errors.len(), 1);
        assert!(loaded.errors[0].contains("server \"work_files\" conflicts with \"work-files\""));
    }

    /// Port of "validates exposure and reads autoEnableCodemode with project
    /// precedence".
    #[test]
    fn validates_exposure_and_reads_auto_enable_codemode_with_project_precedence() {
        let (agent_dir, cwd) = setup(
            serde_json::json!({
                "autoEnableCodemode": false,
                "mcpServers": {
                    "later": { "command": "x", "exposure": "deferred" },
                    "scripts": { "command": "x", "exposure": "codemode-deferred", "toolExposure": { "a": "codemode-deferred" } },
                    "off": { "command": "x", "exposure": "hidden" },
                    "wrong": { "command": "x", "exposure": "model-only" },
                    "described": { "command": "x", "description": "Docs search" },
                    "badDescription": { "command": "x", "description": 1 }
                }
            }),
            serde_json::json!({ "autoEnableCodemode": "yes", "mcpServers": {} }),
        );

        let untrusted = load_mcp_config(&agent_dir, &cwd, false);
        assert_eq!(untrusted.auto_enable_codemode, Some(false));
        let exposures: Vec<(&str, Option<&str>)> = untrusted
            .servers
            .iter()
            .map(|server| {
                (
                    server.name.as_str(),
                    server.config.get("exposure").and_then(|v| v.as_str()),
                )
            })
            .collect();
        assert_eq!(
            exposures,
            vec![
                ("later", Some("deferred")),
                ("scripts", Some("codemode")),
                ("off", Some("hidden")),
                ("described", None),
            ]
        );
        let scripts = untrusted
            .servers
            .iter()
            .find(|s| s.name == "scripts")
            .expect("scripts");
        assert_eq!(
            scripts.config.get("toolExposure"),
            Some(&serde_json::json!({ "a": "codemode" }))
        );
        let described = untrusted
            .servers
            .iter()
            .find(|s| s.name == "described")
            .expect("described");
        assert_eq!(
            described.config.get("description"),
            Some(&serde_json::json!("Docs search"))
        );
        assert_eq!(untrusted.errors.len(), 2);
        assert!(untrusted.errors[0].contains("server \"wrong\": exposure must be one of"));
        assert!(
            untrusted.errors[1].contains("server \"badDescription\": description must be a string")
        );

        let trusted = load_mcp_config(&agent_dir, &cwd, true);
        assert_eq!(trusted.auto_enable_codemode, Some(false));
        assert!(trusted
            .errors
            .iter()
            .any(|error| error.contains("autoEnableCodemode must be a boolean")));
    }

    /// Port of "validates the OAuth callback URL, scope, and client name".
    #[test]
    fn validates_the_oauth_callback_url_scope_and_client_name() {
        let (agent_dir, cwd) = setup(
            serde_json::json!({
                "mcpServers": {
                    "ok": { "url": "https://a.example/mcp", "oauth": { "callbackUrl": "http://localhost:8080/callback", "scope": "a b" } },
                    "ipv6": { "url": "https://a.example/mcp", "oauth": { "callbackUrl": "http://[::1]/cb", "callbackPort": 9000 } },
                    "same": { "url": "https://a.example/mcp", "oauth": { "callbackUrl": "http://127.0.0.1:2/cb", "callbackPort": 2 } },
                    "remote": { "url": "https://a.example/mcp", "oauth": { "callbackUrl": "https://example.com/callback" } },
                    "both": { "url": "https://a.example/mcp", "oauth": { "callbackUrl": "http://127.0.0.1:1/cb", "callbackPort": 2 } },
                    "scope": { "url": "https://a.example/mcp", "oauth": { "scope": ["a"] } },
                    "named": { "url": "https://a.example/mcp", "oauth": { "clientName": "Claude Code" } },
                    "unnamed": { "url": "https://a.example/mcp", "oauth": { "clientName": " " } },
                    "metadata": { "url": "https://a.example/mcp", "oauth": { "authServerMetadataUrl": "https://idp.example/m" } },
                    "plainMetadata": { "url": "https://a.example/mcp", "oauth": { "authServerMetadataUrl": "http://idp.example/m" } },
                    "cimd": { "url": "https://a.example/mcp", "oauth": { "clientRegistration": "cimd", "callbackUrl": "http://localhost/callback" } },
                    "badRegistration": { "url": "https://a.example/mcp", "oauth": { "clientRegistration": "auto" } },
                    "cimdClient": { "url": "https://a.example/mcp", "oauth": { "clientRegistration": "cimd", "clientId": "x" } },
                    "cimdPath": { "url": "https://a.example/mcp", "oauth": { "clientRegistration": "cimd", "callbackUrl": "http://127.0.0.1/cb" } }
                }
            }),
            serde_json::json!({}),
        );
        let loaded = load_mcp_config(&agent_dir, &cwd, false);
        let names: Vec<&str> = loaded
            .servers
            .iter()
            .map(|server| server.name.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["ok", "ipv6", "same", "named", "metadata", "cimd"]
        );
        let expected = [
            "server \"remote\": oauth.callbackUrl must be an http URI on localhost",
            "server \"both\": oauth.callbackUrl and oauth.callbackPort name different ports",
            "server \"scope\": oauth.scope must be a string",
            "server \"unnamed\": oauth.clientName must be a non-empty string",
            "server \"plainMetadata\": oauth.authServerMetadataUrl must be an https URL",
            "server \"badRegistration\": oauth.clientRegistration must be \"dcr\" or \"cimd\"",
            "server \"cimdClient\": oauth.clientRegistration \"cimd\" cannot be combined",
            "server \"cimdPath\": oauth.clientRegistration \"cimd\" requires oauth.callbackUrl",
        ];
        assert_eq!(loaded.errors.len(), expected.len());
        for (error, expected) in loaded.errors.iter().zip(expected) {
            assert!(
                error.contains(expected),
                "error `{error}` lacks `{expected}`"
            );
        }
    }

    /// Port of "resolves per-tool exposure from exact names, then patterns in
    /// order".
    #[test]
    fn resolves_per_tool_exposure_from_exact_names_then_patterns_in_order() {
        let (agent_dir, cwd) = setup(
            serde_json::json!({
                "mcpServers": {
                    "gh": {
                        "command": "x",
                        "exposure": "deferred",
                        "toolExposure": { "get_*": "codemode", "get_me": "direct", "*delete*": "hidden", "get_file.*": "direct" }
                    },
                    "bad": { "command": "x", "toolExposure": { "a": "visible" } }
                }
            }),
            serde_json::json!({}),
        );
        let loaded = load_mcp_config(&agent_dir, &cwd, false);
        assert_eq!(loaded.errors.len(), 1);
        assert!(loaded.errors[0].contains("server \"bad\": toolExposure \"a\" must be one of"));
        let config = &loaded.servers[0].config;
        assert_eq!(get_mcp_tool_exposure(config, "get_me"), McpExposure::Direct);
        assert_eq!(
            get_mcp_tool_exposure(config, "get_issue"),
            McpExposure::Codemode
        );
        assert_eq!(
            get_mcp_tool_exposure(config, "get_delete_hint"),
            McpExposure::Codemode
        );
        assert_eq!(
            get_mcp_tool_exposure(config, "delete_repo"),
            McpExposure::Hidden
        );
        assert_eq!(
            get_mcp_tool_exposure(config, "list_issues"),
            McpExposure::Deferred
        );
        // Only `*` is special.
        let plain =
            serde_json::json!({ "command": "x", "toolExposure": { "get_file.*": "direct" } });
        assert_eq!(
            get_mcp_tool_exposure(&plain, "get_file_x"),
            McpExposure::Codemode
        );
    }

    /// Port of "validates provider auth and accepts it only in the global
    /// mcp.json".
    #[test]
    fn validates_provider_auth_and_accepts_it_only_in_the_global_mcp_json() {
        let (agent_dir, cwd) = setup(
            serde_json::json!({
                "mcpServers": {
                    "radius": { "url": "https://radius.example/mcp", "auth": { "provider": "radius" } },
                    "local": { "url": "http://localhost:8788/mcp", "auth": { "provider": "radius-dev" } },
                    "plain": { "url": "http://radius.example/mcp", "auth": { "provider": "radius" } },
                    "empty": { "url": "https://radius.example/mcp", "auth": { "provider": "" } }
                }
            }),
            serde_json::json!({ "mcpServers": { "radius": { "url": "https://evil.example/mcp", "auth": { "provider": "radius" } } } }),
        );
        let loaded = load_mcp_config(&agent_dir, &cwd, true);
        let summary: Vec<(String, &str)> = loaded
            .servers
            .iter()
            .map(|server| (server.name.clone(), server.scope.expect("scope").as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                ("radius".to_string(), "global"),
                ("local".to_string(), "global")
            ]
        );
        assert_eq!(
            loaded.servers[0].config.get("url"),
            Some(&serde_json::json!("https://radius.example/mcp"))
        );
        assert_eq!(loaded.errors.len(), 3);
        assert!(loaded.errors[0].contains("server \"plain\": auth requires an https URL"));
        assert!(
            loaded.errors[1].contains("server \"empty\": auth.provider must be a provider name")
        );
        assert!(loaded.errors[2]
            .contains("server \"radius\": auth is only allowed in the global mcp.json"));
    }

    /// Port of `updateMcpServerConfig` behavior exercised by #10277 and the
    /// command tests: overrides keep default values, defaults are removed.
    #[test]
    fn update_removes_default_keys_and_keeps_other_content() {
        let root = temp_dir("pi-mcp-config-update-");
        let path = join_path(&root, "mcp.json");
        std::fs::write(
			&path,
			"{\n  \"other\": 1,\n  \"mcpServers\": {\n    \"docs\": { \"url\": \"https://a/mcp\", \"exposure\": \"direct\", \"enabled\": false }\n  }\n}\n",
		)
		.expect("write");
        update_mcp_server_config(
            &path,
            "docs",
            McpServerConfigPatch {
                enabled: Some(true),
                exposure: Some(McpExposure::Codemode),
            },
            false,
        )
        .expect("update");
        let updated: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parse");
        assert_eq!(
            updated.pointer("/mcpServers/docs").expect("docs"),
            &serde_json::json!({ "url": "https://a/mcp" })
        );
        assert_eq!(updated.get("other"), Some(&serde_json::json!(1)));
    }
}
