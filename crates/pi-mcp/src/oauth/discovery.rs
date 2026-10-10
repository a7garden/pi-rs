//! Port of `packages/mcp/src/oauth/discovery.ts`.
//!
//! Adapted from modelcontextprotocol/typescript-sdk v1.29.0 src/client/auth.ts.
//! Modified to remove Zod/CORS shims and enforce authorization-server issuer
//! validation (upstream comment).

use url::Url;

use crate::auth_provider::{default_fetch, discard, HttpRequest, McpFetch};
use crate::oauth::errors::OAuthIssuerMismatchError;
use crate::oauth::types::{
    parse_authorization_server_metadata, parse_protected_resource_metadata,
    AuthorizationServerMetadata, OAuthChallenge, OAuthProtectedResourceMetadata, OAuthServerInfo,
};
use crate::protocol::jsonrpc::McpError;
use crate::protocol::types::LATEST_PROTOCOL_VERSION;

pub type DiscoveryResult<T> = Result<T, McpError>;

fn discard_response(response: Option<crate::auth_provider::HttpResponse>) {
    if let Some(response) = response {
        discard(response);
    }
}

/// 4xx and 502 mean "not here", so discovery tries the next candidate URL.
fn is_discovery_miss(status: u16) -> bool {
    (400..500).contains(&status) || status == 502
}

/// Path suffix for `/.well-known/<kind><path>`; empty for the root path.
fn path_suffix(pathname: &str) -> &str {
    if pathname.ends_with('/') {
        &pathname[..pathname.len() - 1]
    } else {
        pathname
    }
}

/// One `name="value"` / `name=value` field of a challenge header (upstream `field`):
/// the first boundary-preceded `name=` occurrence wins; an empty value counts as absent.
fn field(header: &str, name: &str) -> Option<String> {
    let key = format!("{name}=");
    let lower = header.to_ascii_lowercase();
    let mut rest: &str = &lower;
    let mut offset = 0usize;
    loop {
        let Some(index) = rest.find(&key) else {
            return None;
        };
        let before = &rest[..index];
        let boundary =
            before.is_empty() || before.ends_with([' ', ',', '\t', '\n', '\r', '\x0b', '\x0c']);
        let absolute = offset + index + key.len();
        if boundary {
            let value = &header[absolute..];
            let extracted = if let Some(quoted) = value.strip_prefix('"') {
                match quoted.find('"') {
                    Some(end) => &quoted[..end],
                    None => quoted,
                }
            } else {
                match value.find([',', ' ', '\t', '\n', '\r']) {
                    Some(end) => &value[..end],
                    None => value,
                }
            };
            // An empty value (`scope=""`) carries no information, so it counts as absent.
            return if extracted.is_empty() {
                None
            } else {
                Some(extracted.to_string())
            };
        }
        rest = &rest[index + key.len()..];
        offset = absolute;
    }
}

/// Upstream `parseWwwAuthenticate`.
pub fn parse_www_authenticate(header: Option<&str>) -> OAuthChallenge {
    let Some(header) = header else {
        return OAuthChallenge::default();
    };
    let scheme = header
        .trim_start()
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if scheme != "bearer" && scheme != "dpop" {
        return OAuthChallenge::default();
    }
    let mut challenge = OAuthChallenge {
        resource_metadata_url: None,
        scope: field(header, "scope"),
        error: field(header, "error"),
        error_description: field(header, "error_description"),
    };
    if let Some(resource_metadata) = field(header, "resource_metadata") {
        // Upstream ignores URLs that fail to parse.
        challenge.resource_metadata_url = Url::parse(&resource_metadata).ok();
    }
    challenge
}

async fn fetch_metadata(
    url: Url,
    fetch: &McpFetch,
    protocol_version: &str,
    signal: Option<crate::client::AbortSignal>,
) -> Result<crate::auth_provider::HttpResponse, McpError> {
    let request = HttpRequest {
        method: "GET".to_string(),
        url,
        headers: vec![
            ("Accept".to_string(), "application/json".to_string()),
            (
                "MCP-Protocol-Version".to_string(),
                protocol_version.to_string(),
            ),
        ],
        body: None,
        signal,
    };
    fetch(request).await
}

/// Upstream `discoverProtectedResourceMetadata`.
pub async fn discover_protected_resource_metadata(
    server_url: Url,
    options: DiscoverProtectedResourceOptions<'_>,
) -> DiscoveryResult<OAuthProtectedResourceMetadata> {
    let fetch = options.fetch.cloned().unwrap_or_else(default_fetch);
    let version = options.protocol_version.unwrap_or(LATEST_PROTOCOL_VERSION);
    let default_url = || {
        server_url
            .join(&format!(
                "/.well-known/oauth-protected-resource{}",
                path_suffix(server_url.path())
            ))
            .map_err(|e| McpError::other(e.to_string()))
    };
    let has_explicit_url = options.resource_metadata_url.is_some();
    let mut url = match options.resource_metadata_url.clone() {
        Some(url) => url,
        None => default_url()?,
    };
    let mut response = fetch_metadata(url.clone(), &fetch, version, options.signal.clone()).await?;
    if !has_explicit_url && server_url.path() != "/" && is_discovery_miss(response.status) {
        discard(response);
        url = server_url
            .join("/.well-known/oauth-protected-resource")
            .map_err(|e| McpError::other(e.to_string()))?;
        response = fetch_metadata(url, &fetch, version, options.signal.clone()).await?;
    }
    if !(200..300).contains(&response.status) {
        let status = response.status;
        discard(response);
        return Err(McpError::other(format!(
            "HTTP {status} loading OAuth protected resource metadata"
        )));
    }
    let body = response.json().await.map_err(|error| match error {
        // Upstream propagates JSON parse failures as-is (caught by `discoverOAuthServerInfo`).
        McpError::Network(message) => McpError::other(message),
        other => other,
    })?;
    parse_protected_resource_metadata(body)
}

#[derive(Default)]
pub struct DiscoverProtectedResourceOptions<'a> {
    pub resource_metadata_url: Option<Url>,
    pub protocol_version: Option<&'a str>,
    pub fetch: Option<&'a McpFetch>,
    pub signal: Option<crate::client::AbortSignal>,
}

/// Upstream `buildAuthorizationServerDiscoveryUrls`.
pub fn build_authorization_server_discovery_urls(
    authorization_server_url: &Url,
) -> Vec<(Url, &'static str)> {
    let path = path_suffix(authorization_server_url.path());
    let join = |suffix: &str| {
        authorization_server_url
            .join(suffix)
            .expect("static well-known path")
    };
    let mut urls = vec![
        (
            join(&format!("/.well-known/oauth-authorization-server{path}")),
            "oauth",
        ),
        (
            join(&format!("/.well-known/openid-configuration{path}")),
            "oidc",
        ),
    ];
    if !path.is_empty() {
        // Upstream: `${path}/.well-known/openid-configuration` against the origin.
        urls.push((
            join(&format!("/{path}/.well-known/openid-configuration")),
            "oidc",
        ));
    }
    urls
}

#[derive(Default)]
pub struct DiscoverAuthorizationServerOptions<'a> {
    pub fetch: Option<&'a McpFetch>,
    pub protocol_version: Option<&'a str>,
    pub skip_issuer_validation: bool,
    pub signal: Option<crate::client::AbortSignal>,
}

/// Upstream `discoverAuthorizationServerMetadata`. Returns `Ok(None)` only when the
/// candidate list is exhausted without a definitive answer (unreachable upstream).
pub async fn discover_authorization_server_metadata(
    authorization_server_url: Url,
    options: DiscoverAuthorizationServerOptions<'_>,
) -> DiscoveryResult<Option<AuthorizationServerMetadata>> {
    let fetch = options.fetch.cloned().unwrap_or_else(default_fetch);
    for (url, _kind) in build_authorization_server_discovery_urls(&authorization_server_url) {
        let response = fetch_metadata(
            url.clone(),
            &fetch,
            options.protocol_version.unwrap_or(LATEST_PROTOCOL_VERSION),
            options.signal.clone(),
        )
        .await?;
        if !(200..300).contains(&response.status) {
            let status = response.status;
            discard(response);
            if is_discovery_miss(status) {
                continue;
            }
            return Err(McpError::other(format!(
                "HTTP {status} loading authorization server metadata from {url}"
            )));
        }
        let body = response.json().await.map_err(|error| match error {
            McpError::Network(message) => McpError::other(message),
            other => other,
        })?;
        let metadata = parse_authorization_server_metadata(body)?;
        if !options.skip_issuer_validation {
            // URL parsing adds a trailing slash to bare origins, so compare without one on either side.
            let expected = trim_slash(authorization_server_url.as_str());
            if trim_slash(&metadata.issuer) != expected {
                return Err(McpError::IssuerMismatch(OAuthIssuerMismatchError {
                    expected: authorization_server_url.to_string(),
                    received: Some(metadata.issuer.clone()),
                }));
            }
        }
        return Ok(Some(metadata));
    }
    Ok(None)
}

fn trim_slash(value: &str) -> &str {
    value.strip_suffix('/').unwrap_or(value)
}

#[derive(Default)]
pub struct DiscoverServerInfoOptions<'a> {
    pub resource_metadata_url: Option<Url>,
    /// Metadata document to use instead of discovery. It is trusted as configured, so its issuer is not checked.
    pub authorization_server_metadata_url: Option<Url>,
    pub fetch: Option<&'a McpFetch>,
    pub skip_issuer_validation: bool,
    pub signal: Option<crate::client::AbortSignal>,
}

/// Upstream `discoverOAuthServerInfo`.
pub async fn discover_oauth_server_info(
    server_url: Url,
    options: DiscoverServerInfoOptions<'_>,
) -> DiscoveryResult<OAuthServerInfo> {
    let mut resource_metadata: Option<OAuthProtectedResourceMetadata> = None;
    let discovery = discover_protected_resource_metadata(
        server_url.clone(),
        DiscoverProtectedResourceOptions {
            resource_metadata_url: options.resource_metadata_url.clone(),
            fetch: options.fetch,
            protocol_version: None,
            signal: options.signal.clone(),
        },
    )
    .await;
    match discovery {
        Ok(metadata) => resource_metadata = Some(metadata),
        // Upstream rethrows `TypeError` (network failures) and swallows other errors.
        Err(McpError::Network(_)) => return Err(discovery.unwrap_err()),
        Err(_) => {}
    }
    if let Some(url) = &options.authorization_server_metadata_url {
        let fetch = options.fetch.cloned().unwrap_or_else(default_fetch);
        let response = fetch_metadata(
            url.clone(),
            &fetch,
            LATEST_PROTOCOL_VERSION,
            options.signal.clone(),
        )
        .await?;
        if !(200..300).contains(&response.status) {
            let status = response.status;
            discard_response(Some(response));
            return Err(McpError::other(format!(
                "HTTP {status} loading authorization server metadata from {url}"
            )));
        }
        let body = response.json().await.map_err(|error| match error {
            McpError::Network(message) => McpError::other(message),
            other => other,
        })?;
        let metadata = parse_authorization_server_metadata(body)?;
        return Ok(OAuthServerInfo {
            authorization_server_url: metadata.issuer.clone(),
            authorization_server_metadata: Some(metadata),
            resource_metadata,
        });
    }
    let authorization_server_url = resource_metadata
        .as_ref()
        .and_then(|metadata| metadata.authorization_servers.as_ref())
        .and_then(|servers| servers.first())
        .cloned()
        .unwrap_or_else(|| {
            // `String(new URL("/", serverUrl))`.
            server_url
                .join("/")
                .map(|root| root.to_string())
                .unwrap_or_default()
        });
    let parsed =
        Url::parse(&authorization_server_url).map_err(|e| McpError::other(e.to_string()))?;
    let authorization_server_metadata = discover_authorization_server_metadata(
        parsed,
        DiscoverAuthorizationServerOptions {
            fetch: options.fetch,
            protocol_version: None,
            skip_issuer_validation: options.skip_issuer_validation,
            signal: options.signal.clone(),
        },
    )
    .await?;
    Ok(OAuthServerInfo {
        authorization_server_url,
        authorization_server_metadata,
        resource_metadata,
    })
}

/// Upstream `resourceUrlFromServerUrl`.
pub fn resource_url_from_server_url(value: &Url) -> Url {
    let mut url = value.clone();
    url.set_fragment(None);
    url
}

/// Upstream `selectResource`.
pub fn select_resource(
    server_url: &Url,
    metadata: Option<&OAuthProtectedResourceMetadata>,
) -> DiscoveryResult<Option<String>> {
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    let requested = resource_url_from_server_url(server_url);
    let configured = Url::parse(&metadata.resource).map_err(|e| McpError::other(e.to_string()))?;
    if requested.origin().unicode_serialization() != configured.origin().unicode_serialization() {
        return Err(McpError::other(format!(
            "Protected resource {} does not match MCP server {}",
            metadata.resource, requested
        )));
    }
    let requested_path = ensure_trailing_slash(requested.path());
    let configured_path = ensure_trailing_slash(configured.path());
    if !requested_path.starts_with(&configured_path) {
        return Err(McpError::other(format!(
            "Protected resource {} does not match MCP server {}",
            metadata.resource, requested
        )));
    }
    Ok(Some(metadata.resource.clone()))
}

fn ensure_trailing_slash(path: &str) -> String {
    if path.ends_with('/') {
        path.to_string()
    } else {
        format!("{path}/")
    }
}
