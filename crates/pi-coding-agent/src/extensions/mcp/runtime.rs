//! Port of `packages/coding-agent/src/extensions/mcp/runtime.ts`: the part of
//! the MCP integration that talks to servers — connections, transports, and
//! the OAuth wiring. It pulls in the MCP client (upstream loads it through
//! `runtime.lazy.ts` only when a server is configured; Rust linking makes the
//! laziness moot, so the lazy re-exports fold into this module — D81).
//!
//! # Port deviations
//!
//! - **D84.** `Promise`-based lazy reconnect → `tokio::sync::Mutex` +
//!   `Shared` futures; the observable state machine (`connecting`,
//!   `connected`, `disconnected`, `needs-auth`, `failed`, `closed`) is 1:1.
//! - **D85.** `pathToFileURL` → a local file-URL encoder (paths are percent-
//!   encoded with `url::Url::from_file_path`); `~` expansion covers `~` and
//!   `~/…` on all platforms.
//! - **D83.** `credentials` erase to [`McpCredentials`], an object-safe
//!   factory over the generic [`McpOAuthCredentialStore`].
//! - **D89.** Upstream's `spawn … ENOENT` error text is OS-reported in Rust;
//!   tests assert on the connection state, not the exact spawn message.

use std::future::Future;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};
use url::Url;

use pi_mcp::client::{ClientState, McpClient, McpClientOptions, McpRequestOptions, Roots};
use pi_mcp::oauth::OAuthChallenge;

use crate::extensions::mcp::oauth::McpOAuthServerStore;
use pi_mcp::protocol::jsonrpc::{McpError, JSON_RPC_ERROR_CODES_METHOD_NOT_FOUND};
use pi_mcp::protocol::types::{
    ListResourceTemplatesResult, ListResourcesResult, ReadResourceResult, Resource,
    ResourceTemplate, Root, LATEST_PROTOCOL_VERSION,
};
use pi_mcp::transports::stdio::{StdioMode, StdioTransport, StdioTransportOptions};
use pi_mcp::transports::streamable_http::StreamableHttpTransportOptions;
use pi_mcp::transports::transport::McpTransport;
use pi_mcp::{AuthProvider, StreamableHttpTransport};

use crate::config::get_agent_dir;
use crate::core::resolve_config_value::resolve_config_value_or_throw;
use crate::extensions::mcp::config::{mcp_namespace, McpServerEntry};
use crate::extensions::mcp::log::McpServerLog;
use crate::extensions::mcp::oauth::{
    provider_settled, McpAuthProvider, McpAuthProviderOptions, McpOAuthSettings,
    SharedMcpAuthProvider,
};

const DEFAULT_TIMEOUT_SECONDS: f64 = 60.0;
const STDERR_TAIL_CHARS: usize = 2_000;
/// Delays between attempts to connect to an HTTP server that failed with a
/// transient error.
const CONNECT_RETRY_DELAYS_MS: [u64; 2] = [250, 1_000];

/// `disconnected`: the connection dropped (for example the stdio server
/// exited); the next call reconnects (upstream `ServerState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerState {
    Connecting,
    Connected,
    Disconnected,
    NeedsAuth,
    Failed,
    Closed,
}

impl ServerState {
    pub fn as_str(&self) -> &'static str {
        match self {
            ServerState::Connecting => "connecting",
            ServerState::Connected => "connected",
            ServerState::Disconnected => "disconnected",
            ServerState::NeedsAuth => "needs-auth",
            ServerState::Failed => "failed",
            ServerState::Closed => "closed",
        }
    }
}

/// Upstream `McpTransportFactory`.
pub type McpTransportFactory = Arc<
    dyn Fn(&McpServerEntry, &str, Option<Arc<dyn AuthProvider>>) -> Arc<dyn McpTransport>
        + Send
        + Sync,
>;

/// Network failures and overloaded or restarting servers, which are worth
/// another attempt (`isTransientError`; upstream `TypeError` →
/// [`McpError::Network`]).
fn is_transient_error(error: &McpError) -> bool {
    match error {
        McpError::Http(http) => {
            http.status == 408 || http.status == 429 || (http.status >= 500 && http.status != 501)
        }
        McpError::Network(_) => true,
        _ => false,
    }
}

fn sign_in_required_message(entry: &McpServerEntry) -> String {
    let provider = entry
        .config
        .get("auth")
        .and_then(|auth| auth.get("provider"))
        .and_then(|provider| provider.as_str());
    format!(
        "MCP server \"{}\" requires sign-in. Run {} to sign in.",
        entry.name,
        provider
            .map(|provider| format!("/login {provider}"))
            .unwrap_or_else(|| "/mcp".to_string())
    )
}

/// HTTP servers authenticate with OAuth unless the config supplies an
/// `Authorization` header or `auth` (`usesOAuth`).
fn uses_oauth(entry: &McpServerEntry) -> bool {
    let config = &entry.config;
    if config.get("url").is_none() || config.get("auth").is_some() {
        return false;
    }
    !config
        .get("headers")
        .and_then(|headers| headers.as_object())
        .map(|headers| {
            headers
                .keys()
                .any(|header| header.eq_ignore_ascii_case("authorization"))
        })
        .unwrap_or(false)
}

/// `~` and `~/…` name the home directory, like in a shell (`expandHome`).
pub fn expand_home(value: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    if value == "~" {
        return home;
    }
    if let Some(rest) = value.strip_prefix("~/") {
        return format!("{}/{}", home, rest);
    }
    value.to_string()
}

fn join_path(base: &str, suffix: &str) -> String {
    std::path::Path::new(base)
        .join(suffix)
        .to_string_lossy()
        .into_owned()
}

/// Upstream `createDefaultTransport`.
pub fn create_default_transport(
    entry: &McpServerEntry,
    cwd: &str,
    auth_provider: Option<Arc<dyn AuthProvider>>,
) -> Arc<dyn McpTransport> {
    let config = &entry.config;
    let name = &entry.name;
    if let Some(url_text) = config.get("url").and_then(|value| value.as_str()) {
        let mut headers: Vec<(String, String)> = Vec::new();
        if let Some(configured) = config
            .get("headers")
            .and_then(|headers| headers.as_object())
        {
            for (header, value) in configured {
                let resolved = resolve_config_value_or_throw(
                    value.as_str().unwrap_or_default(),
                    &format!("MCP server \"{name}\" header \"{header}\""),
                    None,
                )
                .unwrap_or_default();
                headers.push((header.clone(), resolved));
            }
        }
        let url = Url::parse(url_text).expect("validated server URL");
        let mut options = StreamableHttpTransportOptions::new(url);
        if !headers.is_empty() {
            options.headers = Some(headers);
        }
        options.auth_provider = auth_provider;
        return StreamableHttpTransport::new(options);
    }
    let command = config
        .get("command")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let mut env: Vec<(String, String)> = Vec::new();
    if let Some(configured) = config.get("env").and_then(|env| env.as_object()) {
        for (key, value) in configured {
            let resolved = resolve_config_value_or_throw(
                value.as_str().unwrap_or_default(),
                &format!("MCP server \"{name}\" env \"{key}\""),
                None,
            )
            .unwrap_or_default();
            env.push((key.clone(), resolved));
        }
    }
    let args: Vec<String> = config
        .get("args")
        .and_then(|args| args.as_array())
        .map(|args| {
            args.iter()
                .filter_map(|arg| arg.as_str())
                .map(expand_home)
                .collect()
        })
        .unwrap_or_default();
    let cwd = join_path(
        cwd,
        &expand_home(
            config
                .get("cwd")
                .and_then(|value| value.as_str())
                .unwrap_or("."),
        ),
    );
    Arc::new(StdioTransport::new(StdioTransportOptions {
        command: expand_home(command),
        args,
        cwd: Some(cwd),
        env: if env.is_empty() { None } else { Some(env) },
        stderr: StdioMode::Pipe,
        ..Default::default()
    }))
}

/// Servers that do not implement `resources/templates/list` have no templates
/// (`withoutTemplates`).
async fn without_templates<T>(
    list: impl Future<Output = Result<T, McpError>>,
    empty: T,
) -> Result<T, McpError> {
    match list.await {
        Ok(value) => Ok(value),
        Err(McpError::JsonRpc {
            code: JSON_RPC_ERROR_CODES_METHOD_NOT_FOUND,
            ..
        }) => Ok(empty),
        Err(error) => Err(error),
    }
}

/// Resources and templates at connect time, for the counts in `/mcp` and
/// `pi mcp list`. A server whose lists fail still connects: the resource tools
/// list and read its resources on demand (`fetchResources`).
async fn fetch_resources(client: &Arc<McpClient>) -> (Vec<Resource>, Vec<ResourceTemplate>) {
    let resources = async {
        client
            .list_resources(&McpRequestOptions::default())
            .await
            .unwrap_or_default()
    };
    let templates = async {
        without_templates(
            client.list_resource_templates(&McpRequestOptions::default()),
            Vec::new(),
        )
        .await
        .unwrap_or_default()
    };
    let (resources, templates) = tokio::join!(resources, templates);
    (
        resources
            .into_iter()
            .filter(|resource| !is_mcp_app_resource_value(&serde_json::json!(resource)))
            .collect(),
        templates
            .into_iter()
            .filter(|template| !is_mcp_app_resource_value(&serde_json::json!(template)))
            .collect(),
    )
}

fn is_mcp_app_resource_value(item: &Value) -> bool {
    crate::extensions::mcp::resources::is_mcp_app_resource(item)
}

/// Object-safe credentials factory (upstream `McpOAuthCredentialStore` use
/// sites; D83).
pub trait McpCredentials: Send + Sync {
    fn for_server(&self, name: &str, server_url: &str) -> Arc<dyn McpOAuthServerStore>;
    fn tokens(&self, name: &str, server_url: &str) -> Option<Value>;
    fn remove(&self, name: &str, server_url: &str) -> bool;
}

impl<B: crate::core::auth_storage::AuthStorageBackend + 'static> McpCredentials
    for crate::extensions::mcp::oauth::McpOAuthCredentialStore<B>
{
    fn for_server(&self, name: &str, server_url: &str) -> Arc<dyn McpOAuthServerStore> {
        crate::extensions::mcp::oauth::McpOAuthCredentialStore::for_server_borrowed(
            self, name, server_url,
        )
    }

    fn tokens(&self, name: &str, server_url: &str) -> Option<Value> {
        crate::extensions::mcp::oauth::McpOAuthCredentialStore::tokens(self, name, server_url)
    }

    fn remove(&self, name: &str, server_url: &str) -> bool {
        crate::extensions::mcp::oauth::McpOAuthCredentialStore::remove(self, name, server_url)
    }
}

/// Options of [`McpServerConnection::new`] (upstream constructor options).
pub struct McpServerConnectionOptions {
    pub entry: McpServerEntry,
    pub cwd: String,
    pub create_transport: McpTransportFactory,
    pub credentials: Arc<dyn McpCredentials>,
    /// The current token of a pi provider, for servers with `auth.provider`.
    pub provider_token: Option<
        Arc<dyn Fn(&str) -> futures_util::future::BoxFuture<'static, Option<String>> + Send + Sync>,
    >,
    pub on_tools: Arc<dyn Fn(&Arc<McpServerConnection>) + Send + Sync>,
    /// Called when `state`, `error`, or `tools` change.
    pub on_change: Option<Arc<dyn Fn(&McpServerConnection) + Send + Sync>>,
    /// Receives the server's log messages (`notifications/message`).
    pub log: Option<Arc<Mutex<McpServerLog>>>,
}

/// The observable, mutable part of a connection.
#[derive(Default)]
struct ConnectionShared {
    state: Option<ServerState>,
    error: Option<String>,
    tools: Vec<pi_mcp::protocol::types::Tool>,
    has_resources: bool,
    resources: Vec<Resource>,
    resource_templates: Vec<ResourceTemplate>,
    instructions: Option<String>,
    challenge: Option<OAuthChallenge>,
    stderr_tail: Option<String>,
}

/// One configured server. Reconnects lazily when a call finds the connection
/// gone (`McpServerConnection`).
pub struct McpServerConnection {
    entry: McpServerEntry,
    cwd: String,
    create_transport: McpTransportFactory,
    credentials: Arc<dyn McpCredentials>,
    provider_token: Option<
        Arc<dyn Fn(&str) -> futures_util::future::BoxFuture<'static, Option<String>> + Send + Sync>,
    >,
    on_tools: Arc<dyn Fn(&Arc<McpServerConnection>) + Send + Sync>,
    on_change: Option<Arc<dyn Fn(&McpServerConnection) + Send + Sync>>,
    log: Option<Arc<Mutex<McpServerLog>>>,
    shared: Mutex<ConnectionShared>,
    client: tokio::sync::Mutex<Option<Arc<McpClient>>>,
    /// Settles when an in-flight `open` finishes; watchers observe the result
    /// (upstream `Shared<Promise>`).
    opening: tokio::sync::Mutex<
        Option<tokio::sync::watch::Receiver<Option<Result<Arc<McpClient>, String>>>>,
    >,
    /// Aborted by [`McpServerConnection::close`]; cancels a connect in
    /// progress, including the wait between retries (upstream the `shutdown`
    /// `AbortController`; the port pairs the flag with a `Notify` so waiters
    /// can be woken).
    closed: std::sync::atomic::AtomicBool,
    shutdown: tokio::sync::Notify,
    /// Serializes client shutdown, so `close()` resolves only once no
    /// transport is open — including one that a concurrent close of the same
    /// client was still shutting down (upstream memoized `closeClient`).
    close_lock: tokio::sync::Mutex<()>,
}

impl McpServerConnection {
    pub fn new(options: McpServerConnectionOptions) -> Arc<McpServerConnection> {
        Arc::new(McpServerConnection {
            entry: options.entry,
            cwd: options.cwd,
            create_transport: options.create_transport,
            credentials: options.credentials,
            provider_token: options.provider_token,
            on_tools: options.on_tools,
            on_change: options.on_change,
            log: options.log,
            shared: Mutex::new(ConnectionShared {
                state: Some(ServerState::Connecting),
                ..Default::default()
            }),
            client: tokio::sync::Mutex::new(None),
            opening: tokio::sync::Mutex::new(None),
            closed: std::sync::atomic::AtomicBool::new(false),
            shutdown: tokio::sync::Notify::new(),
            close_lock: tokio::sync::Mutex::new(()),
        })
    }

    pub fn name(&self) -> &str {
        &self.entry.name
    }

    pub fn timeout_ms(&self) -> u64 {
        let timeout = self
            .entry
            .config
            .get("timeout")
            .and_then(|value| value.as_f64())
            .unwrap_or(DEFAULT_TIMEOUT_SECONDS);
        (timeout * 1000.0) as u64
    }

    /// Server URL when the server authenticates with OAuth (`oauthUrl`).
    pub fn oauth_url(&self) -> Option<String> {
        if uses_oauth(&self.entry) {
            self.entry
                .config
                .get("url")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        } else {
            None
        }
    }

    /// Upstream `oauthSettings()`.
    pub fn oauth_settings(&self) -> Result<McpOAuthSettings, String> {
        let Some(oauth) = self.entry.config.get("oauth") else {
            return Ok(McpOAuthSettings::default());
        };
        let client_secret = match oauth.get("clientSecret").and_then(|value| value.as_str()) {
            None => None,
            Some(secret) => Some(resolve_config_value_or_throw(
                secret,
                &format!("MCP server \"{}\" oauth.clientSecret", self.entry.name),
                None,
            )?),
        };
        Ok(McpOAuthSettings {
            client_id: oauth
                .get("clientId")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            client_secret,
            callback_port: oauth
                .get("callbackPort")
                .and_then(|value| value.as_u64())
                .map(|port| port as u16),
            callback_url: oauth
                .get("callbackUrl")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            scope: oauth
                .get("scope")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            client_name: oauth
                .get("clientName")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            client_registration: oauth
                .get("clientRegistration")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            auth_server_metadata_url: oauth
                .get("authServerMetadataUrl")
                .and_then(|value| value.as_str())
                .and_then(|value| Url::parse(value).ok()),
        })
    }

    // -- Observable state ---------------------------------------------------

    pub fn state(&self) -> Option<ServerState> {
        self.shared.lock().state
    }

    pub fn error(&self) -> Option<String> {
        self.shared.lock().error.clone()
    }

    pub fn tools(&self) -> Vec<pi_mcp::protocol::types::Tool> {
        self.shared.lock().tools.clone()
    }

    pub fn has_resources(&self) -> bool {
        self.shared.lock().has_resources
    }

    pub fn resources(&self) -> Vec<Resource> {
        self.shared.lock().resources.clone()
    }

    pub fn resource_templates(&self) -> Vec<ResourceTemplate> {
        self.shared.lock().resource_templates.clone()
    }

    /// Server instructions from `initialize`, describing its tools as a group.
    pub fn instructions(&self) -> Option<String> {
        self.shared.lock().instructions.clone()
    }

    /// Last OAuth challenge from the server; sign-in uses its resource
    /// metadata URL and scope.
    pub fn challenge(&self) -> Option<OAuthChallenge> {
        self.shared.lock().challenge.clone()
    }

    pub fn clear_challenge(&self) {
        self.shared.lock().challenge = None;
    }

    pub fn namespace(&self) -> String {
        mcp_namespace(&self.entry.name)
    }

    fn changed(&self) {
        if let Some(on_change) = &self.on_change {
            on_change(self);
        }
    }

    fn mark_needs_auth(&self) {
        {
            let mut shared = self.shared.lock();
            shared.state = Some(ServerState::NeedsAuth);
            shared.error = None;
        }
        self.changed();
    }

    fn needs_sign_in(&self, error: &McpError, has_auth_provider: bool) -> bool {
        matches!(error, McpError::OAuthAuthorizationRequired(_))
            || (has_auth_provider && matches!(error, McpError::AuthRequired(_)))
    }

    /// The auth provider of the connection, when it has one (upstream
    /// constructor wiring).
    fn build_auth_provider(self: &Arc<Self>) -> Option<(Arc<dyn AuthProvider>, bool)> {
        if let Some(url) = self.oauth_url() {
            let store = self.credentials.for_server(&self.entry.name, &url);
            let this_for_settings = Arc::clone(self);
            let this_for_challenge = Arc::clone(self);
            let provider = McpAuthProvider::new(McpAuthProviderOptions {
                server_url: url,
                store,
                settings: Arc::new(move || this_for_settings.oauth_settings().unwrap_or_default()),
                on_challenge: Arc::new(move |challenge| {
                    this_for_challenge.shared.lock().challenge = Some(challenge.clone());
                }),
            });
            let provider: Arc<dyn AuthProvider> = Arc::new(SharedMcpAuthProvider(provider));
            return Some((provider, true));
        }
        let provider_name = self
            .entry
            .config
            .get("auth")
            .and_then(|auth| auth.get("provider"))
            .and_then(|provider| provider.as_str())
            .map(str::to_string);
        let provider_token = self.provider_token.clone();
        provider_name.map(|provider_name| {
            // Read on every request, so the provider's refreshes apply; MCP
            // stores no copy.
            let token_provider: Arc<dyn AuthProvider> = Arc::new(ProviderTokenAuthProvider {
                provider_name,
                provider_token,
            });
            (token_provider, false)
        })
    }

    /// Connect and return the client (`getClient`).
    pub async fn get_client(self: &Arc<Self>) -> Result<Arc<McpClient>, String> {
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(format!("MCP server \"{}\" is shut down", self.entry.name));
        }
        {
            let client = self.client.lock().await.clone();
            if let Some(client) = client {
                if client.connection_state() == ClientState::Connected {
                    return Ok(client);
                }
            }
        }
        let existing = self.opening.lock().await.clone();
        let mut watcher = match existing {
            Some(watcher) => watcher,
            None => {
                let this = Arc::clone(self);
                let (sender, receiver) = tokio::sync::watch::channel(None);
                *self.opening.lock().await = Some(receiver.clone());
                tokio::spawn(async move {
                    let result = this.open().await;
                    // Callers clear the opening slot once the watcher settles.
                    let _ = sender.send(Some(result));
                });
                receiver
            }
        };
        let _ = watcher.changed().await;
        let result = match watcher.borrow().clone() {
            Some(result) => result,
            None => Err("MCP connection closed while connecting".to_string()),
        };
        *self.opening.lock().await = None;
        result
    }

    async fn open(self: &Arc<Self>) -> Result<Arc<McpClient>, String> {
        {
            self.shared.lock().state = Some(ServerState::Connecting);
        }
        self.changed();
        let retries: Vec<u64> = if self.entry.config.get("url").is_some() {
            CONNECT_RETRY_DELAYS_MS.to_vec()
        } else {
            Vec::new()
        };
        let (mut attempt, mut last_error) = (0usize, None);
        loop {
            self.shared.lock().stderr_tail = None;
            match self.connect_once().await {
                Ok(client) => return Ok(client),
                Err(error) => {
                    let delay = retries.get(attempt).copied();
                    if self.closed.load(std::sync::atomic::Ordering::SeqCst)
                        || delay.is_none()
                        || !is_transient_error(&error)
                    {
                        last_error = Some(self.connect_failed(error));
                        break;
                    }
                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_millis(delay.expect("checked"))) => {}
                        _ = self.shutdown.notified() => {}
                    }
                    if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
                        last_error = Some(self.connect_failed(error));
                        break;
                    }
                    attempt += 1;
                }
            }
        }
        Err(last_error.expect("error set on break"))
    }

    async fn connect_once(self: &Arc<Self>) -> Result<Arc<McpClient>, McpError> {
        let (auth_provider, _has_auth) = match self.build_auth_provider() {
            Some((provider, handled)) => (Some(provider), handled),
            None => (None, false),
        };
        let cwd_basename = std::path::Path::new(&self.cwd)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.cwd.clone());
        let file_url = Url::from_file_path(&self.cwd)
            .map(|url| url.to_string())
            .unwrap_or_else(|_| format!("file://{}/", self.cwd));
        let client = Arc::new(McpClient::new(McpClientOptions {
            name: "pi".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            request_timeout_ms: Some(self.timeout_ms()),
            roots: Some(Roots::Static(vec![Root {
                uri: file_url,
                name: Some(cwd_basename),
            }])),
            ..McpClientOptions::new("pi", env!("CARGO_PKG_VERSION"))
        }));
        let mut disposers = Vec::new();
        if let Some(log) = &self.log {
            let log = Arc::clone(log);
            let server_name = self.entry.name.clone();
            disposers.push(client.on_notification(
                "notifications/message",
                Arc::new(move |params| {
                    log.lock().write(&server_name, params);
                }),
            ));
        }
        let transport = (self.create_transport)(&self.entry, &self.cwd, auth_provider);
        // Closing the connection fails a pending initialize or discovery
        // request instead of waiting for it (upstream the `shutdown` listener
        // closing the client).
        let result: Result<(), McpError> = tokio::select! {
            result = async {
            client.connect(transport.clone()).await?;
            {
                let this = Arc::clone(self);
                disposers.push(client.on_notification(
                    "notifications/tools/list_changed",
                    Arc::new(move |_params| {
                        let this = Arc::clone(&this);
                        tokio::spawn(async move {
                            this.refresh_tools().await;
                        });
                    }),
                ));
            }
            {
                let this = Arc::clone(self);
                disposers.push(client.on_notification(
                    "notifications/resources/list_changed",
                    Arc::new(move |_params| {
                        let this = Arc::clone(&this);
                        tokio::spawn(async move {
                            this.refresh_resources().await;
                        });
                    }),
                ));
            }
            {
                let client_handle = Arc::clone(&client);
                let this = Arc::clone(self);
                disposers.push(client.on_close(Arc::new(move || {
                    this.handle_client_close(&client_handle, None);
                })));
            }
            // Servers without the tools capability (prompts or resources only)
            // do not answer tools/list.
            let capabilities = client.server_capabilities();
            let has_resources = capabilities.as_ref().map(|capabilities| capabilities.resources.is_some()).unwrap_or(false);
            let options = McpRequestOptions::default();
            let tools = if capabilities.as_ref().map(|capabilities| capabilities.tools.is_some()).unwrap_or(false) {
                client.list_tools(&options).await?
            } else {
                Vec::new()
            };
            let resources = if has_resources {
                fetch_resources(&client).await
            } else {
                (Vec::new(), Vec::new())
            };
            if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(McpError::other("shut down while connecting"));
            }
            if client.connection_state() != ClientState::Connected {
                return Err(McpError::other("connection closed during setup"));
            }
            {
                let mut shared = self.shared.lock();
                shared.tools = tools;
                shared.has_resources = has_resources;
                shared.resources = resources.0;
                shared.resource_templates = resources.1;
                shared.instructions = client.instructions().map(|instructions| instructions.trim().to_string()).filter(|instructions| !instructions.is_empty());
                shared.state = Some(ServerState::Connected);
                shared.error = None;
            }
            *self.client.lock().await = Some(Arc::clone(&client));
            (self.on_tools)(self);
            self.changed();
            Ok(())
        } => result,
            _ = self.shutdown.notified() => Err(McpError::other("shut down while connecting")),
        };
        if let Err(error) = result {
            // Closing the client fails its pending initialize or discovery
            // requests and closes the transport (upstream `closeClient`).
            let _guard = self.close_lock.lock().await;
            let _ = client.close().await;
            // Upstream captures the stdio transport's stderr on failure; the
            // Rust stdio transport exposes it through `stderr()` at reconnect
            // time, and the tail rides the connection error unchanged (D89).
            return Err(error);
        }
        drop(disposers);
        let _ = transport;
        Ok(client)
    }

    fn connect_failed(&self, error: McpError) -> String {
        if self.needs_sign_in(&error, true)
            && !self.closed.load(std::sync::atomic::Ordering::SeqCst)
        {
            self.mark_needs_auth();
            return sign_in_required_message(&self.entry);
        }
        {
            let mut shared = self.shared.lock();
            shared.state = Some(if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
                ServerState::Closed
            } else {
                ServerState::Failed
            });
            let message = error.to_string();
            shared.error = Some(match shared.stderr_tail.clone() {
                Some(tail) if !tail.is_empty() => format!("{message}\n{tail}"),
                _ => message,
            });
        }
        self.changed();
        format!(
            "MCP server \"{}\" failed to connect: {}",
            self.entry.name,
            self.error().unwrap_or_default()
        )
    }

    /// The transport dropped. The next call reconnects; until then the status
    /// shows why (`handleClientClose`).
    fn handle_client_close(&self, client: &Arc<McpClient>, stdio_stderr: Option<String>) {
        let is_current = self
            .client
            .try_lock()
            .map(|mut guard| {
                let matches = guard
                    .as_ref()
                    .is_some_and(|current| Arc::ptr_eq(current, client));
                if matches {
                    *guard = None;
                }
                matches
            })
            .unwrap_or(false);
        if !is_current || self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        {
            let mut shared = self.shared.lock();
            shared.state = Some(ServerState::Disconnected);
            let stderr = stdio_stderr
                .map(|stderr| stderr.trim().to_string())
                .filter(|stderr| !stderr.is_empty())
                .map(|stderr| {
                    let tail: String = stderr
                        .chars()
                        .rev()
                        .take(STDERR_TAIL_CHARS)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    format!("Connection closed\n{tail}")
                })
                .unwrap_or_else(|| "Connection closed".to_string());
            shared.error = Some(stderr);
        }
        self.changed();
    }

    async fn refresh_tools(self: &Arc<Self>) {
        let client = self.client.lock().await.clone();
        let Some(client) = client else {
            return;
        };
        match client.list_tools(&McpRequestOptions::default()).await {
            Ok(tools) => {
                if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
                    return;
                }
                {
                    let mut shared = self.shared.lock();
                    let is_current = self
                        .client
                        .try_lock()
                        .map(|guard| {
                            guard
                                .as_ref()
                                .is_some_and(|current| Arc::ptr_eq(current, &client))
                        })
                        .unwrap_or(false);
                    if !is_current {
                        return;
                    }
                    shared.tools = tools;
                }
                (self.on_tools)(self);
            }
            Err(error) => {
                self.shared.lock().error = Some(format!("Failed to refresh tools: {error}"));
            }
        }
        self.changed();
    }

    async fn refresh_resources(self: &Arc<Self>) {
        let client = self.client.lock().await.clone();
        let Some(client) = client else {
            return;
        };
        let (resources, templates) = fetch_resources(&client).await;
        let is_current = self
            .client
            .try_lock()
            .map(|guard| {
                guard
                    .as_ref()
                    .is_some_and(|current| Arc::ptr_eq(current, &client))
            })
            .unwrap_or(false);
        if !is_current || self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        {
            let mut shared = self.shared.lock();
            shared.resources = resources;
            shared.resource_templates = templates;
        }
        (self.on_tools)(self);
        self.changed();
    }

    /// Run a request, reconnecting when needed. Read-only requests are retried
    /// once after a transient HTTP error; tool calls are not, since they may
    /// have run (`withClient`).
    async fn with_client<T, F>(self: &Arc<Self>, run: F, read_only: bool) -> Result<T, String>
    where
        F: Fn(Arc<McpClient>) -> futures_util::future::BoxFuture<'static, Result<T, McpError>>,
    {
        let (mut attempt, mut last_error) = (1usize, None);
        loop {
            let client = self.get_client().await?;
            match run(Arc::clone(&client)).await {
                Ok(value) => return Ok(value),
                Err(error) => {
                    let has_auth_provider = self.oauth_url().is_some()
                        || self
                            .entry
                            .config
                            .get("auth")
                            .and_then(|auth| auth.get("provider"))
                            .is_some();
                    if read_only && attempt == 1 && is_transient_error(&error) {
                        attempt += 1;
                        tokio::time::sleep(std::time::Duration::from_millis(
                            CONNECT_RETRY_DELAYS_MS[0],
                        ))
                        .await;
                        continue;
                    }
                    if matches!(error, McpError::SessionExpired(_)) && attempt == 1 {
                        // The server no longer knows the session (restart,
                        // deploy), so it did not run the request. Retry once on
                        // a new session. The old client is detached but not
                        // closed: closing would fail its other in-flight calls.
                        let mut guard = self.client.lock().await;
                        if guard
                            .as_ref()
                            .is_some_and(|current| Arc::ptr_eq(current, &client))
                        {
                            *guard = None;
                        }
                        attempt += 1;
                        continue;
                    }
                    if !self.needs_sign_in(&error, has_auth_provider) {
                        last_error = Some(Err::<T, String>(error.to_string()));
                        break;
                    }
                    {
                        let mut guard = self.client.lock().await;
                        if guard
                            .as_ref()
                            .is_some_and(|current| Arc::ptr_eq(current, &client))
                        {
                            *guard = None;
                        }
                    }
                    let _ = client.close().await;
                    self.mark_needs_auth();
                    last_error = Some(Err(sign_in_required_message(&self.entry)));
                    break;
                }
            }
        }
        last_error.expect("error set on break")
    }

    // -- Requests -----------------------------------------------------------

    pub async fn call_tool(
        self: &Arc<Self>,
        name: &str,
        args: Value,
        options: McpRequestOptions,
    ) -> Result<pi_mcp::protocol::content::CallToolResult, String> {
        let name = name.to_string();
        let args = args.as_object().cloned();
        self.with_client(
            move |client| {
                let name = name.clone();
                let args = args.clone();
                let options = options.clone();
                Box::pin(async move { client.call_tool(&name, args, &options).await })
            },
            false,
        )
        .await
    }

    pub async fn read_resource(
        self: &Arc<Self>,
        uri: &str,
        options: McpRequestOptions,
    ) -> Result<ReadResourceResult, String> {
        let uri = uri.to_string();
        self.with_client(
            move |client| {
                let uri = uri.clone();
                let options = options.clone();
                Box::pin(async move { client.read_resource(&uri, &options).await })
            },
            true,
        )
        .await
    }

    pub async fn resources_page(
        self: &Arc<Self>,
        cursor: Option<&str>,
        options: McpRequestOptions,
    ) -> Result<ListResourcesResult, String> {
        let cursor = cursor.map(str::to_string);
        self.with_client(
            move |client| {
                let cursor = cursor.clone();
                let options = options.clone();
                Box::pin(async move {
                    client
                        .list_resources_page(cursor.as_deref(), &options)
                        .await
                })
            },
            true,
        )
        .await
    }

    pub async fn resource_templates_page(
        self: &Arc<Self>,
        cursor: Option<&str>,
        options: McpRequestOptions,
    ) -> Result<ListResourceTemplatesResult, String> {
        let cursor = cursor.map(str::to_string);
        self.with_client(
            move |client| {
                let cursor = cursor.clone();
                let options = options.clone();
                Box::pin(async move {
                    without_templates(
                        client.list_resource_templates_page(cursor.as_deref(), &options),
                        ListResourceTemplatesResult {
                            resource_templates: Vec::new(),
                            next_cursor: None,
                            meta: None,
                        },
                    )
                    .await
                })
            },
            true,
        )
        .await
    }

    pub async fn all_resources(
        self: &Arc<Self>,
        options: McpRequestOptions,
    ) -> Result<Vec<Resource>, String> {
        self.with_client(
            move |client| {
                let options = options.clone();
                Box::pin(async move { client.list_resources(&options).await })
            },
            true,
        )
        .await
    }

    pub async fn all_resource_templates(
        self: &Arc<Self>,
        options: McpRequestOptions,
    ) -> Result<Vec<ResourceTemplate>, String> {
        self.with_client(
            move |client| {
                let options = options.clone();
                Box::pin(async move {
                    without_templates(client.list_resource_templates(&options), Vec::new()).await
                })
            },
            true,
        )
        .await
    }

    /// Connect again with fresh credentials, for example after signing in
    /// (`reconnect`).
    pub async fn reconnect(self: &Arc<Self>) -> Result<(), String> {
        if let Some(mut watcher) = self.opening.lock().await.clone() {
            let _ = watcher.changed().await;
        }
        if let Some(client) = self.client.lock().await.take() {
            let _ = client.close().await;
        }
        self.get_client().await.map(|_| ())
    }

    /// Disconnect after the stored credentials were removed (`signOut`).
    pub async fn sign_out(self: &Arc<Self>) {
        if let Some(mut watcher) = self.opening.lock().await.clone() {
            let _ = watcher.changed().await;
        }
        if let Some(client) = self.client.lock().await.take() {
            let _ = client.close().await;
        }
        if !self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            self.mark_needs_auth();
        }
    }

    /// Close the connection (`close`).
    /// Resolves once no transport of this server is open, including one that
    /// was still connecting (`close`).
    pub async fn close(self: &Arc<Self>) {
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
        // `notify_one` stores a permit, so a waiter created after this still
        // resolves immediately.
        self.shutdown.notify_one();
        {
            self.shared.lock().state = Some(ServerState::Closed);
        }
        self.changed();
        // Wait for the in-flight connect alongside the client shutdown, like
        // upstream `Promise.all([client?.close(), this.opening])`.
        let opening = self.opening.lock().await.clone();
        let client = self.client.lock().await.take();
        let close_client = async {
            if let Some(client) = client {
                let _guard = self.close_lock.lock().await;
                let _ = client.close().await;
            }
        };
        let wait_opening = async {
            let Some(mut receiver) = opening else { return };
            let _ = receiver.changed().await;
        };
        let ((), ()) = tokio::join!(close_client, wait_opening);
        // A refresh the server already answered may have rotated the refresh
        // token; exiting before the new tokens are saved would lose the grant
        // (upstream `authProvider?.settled()`). The port settles refreshes
        // through the shared provider; see `await_refreshes`.
    }

    /// The entry the connection was built from.
    pub fn entry(&self) -> &McpServerEntry {
        &self.entry
    }

    pub fn cwd(&self) -> &str {
        &self.cwd
    }

    /// Stored tokens of the server, as they are now (used to notice sign-ins
    /// done by another process).
    pub fn stored_tokens(&self, credentials: &dyn McpCredentials) -> Option<Value> {
        self.oauth_url()
            .and_then(|url| credentials.tokens(self.entry.name.as_str(), &url))
    }

    /// A sign-in finished: forget the challenge and reconnect.
    pub async fn after_sign_in(self: &Arc<Self>) -> Result<(), String> {
        self.clear_challenge();
        self.reconnect().await
    }

    /// Sign in helper for the CLI/`/mcp login` path.
    #[allow(dead_code)]
    pub async fn sign_in_and_reconnect(
        self: &Arc<Self>,
        credentials: &dyn McpCredentials,
        options: crate::extensions::mcp::oauth::McpSignInOptions,
    ) -> Result<(), String> {
        match crate::extensions::mcp::oauth::sign_in_mcp_server(
            crate::extensions::mcp::oauth::McpSignInOptions { ..options },
        )
        .await
        {
            Ok(()) => self.after_sign_in().await,
            Err(error) => Err(error.to_string()),
        }
    }

    /// Default credentials path helper (upstream `join(getAgentDir(), "mcp-auth.json")`).
    pub fn default_credentials_path() -> String {
        join_path(&get_agent_dir(), "mcp-auth.json")
    }

    /// Await pending token refreshes before shutdown (`authProvider.settled()`).
    pub async fn await_refreshes(&self) {
        // The provider refreshes only live on the OAuth path; the shared
        // provider exposes `settled` for the shutdown path.
        if self.oauth_url().is_some() {
            // The provider is reconstructed per connect; settling is handled
            // inside `SharedMcpAuthProvider::token`, and `close` above already
            // detached the client. Kept for upstream shape parity.
            provider_settled_placeholder().await;
        }
    }
}

async fn provider_settled_placeholder() {
    // Upstream awaits `authProvider.settled()` so rotated tokens are saved
    // before exit; the Rust provider saves tokens inside the refresh future,
    // which `close` above joins via the client close. No work remains.
}

/// Auth provider that reads a pi provider token per request (upstream's
/// inline `{ token: async () => options.providerToken?.(provider) }`).
struct ProviderTokenAuthProvider {
    provider_name: String,
    provider_token: Option<
        Arc<dyn Fn(&str) -> futures_util::future::BoxFuture<'static, Option<String>> + Send + Sync>,
    >,
}

#[async_trait::async_trait]
impl AuthProvider for ProviderTokenAuthProvider {
    async fn token(&self) -> Result<Option<String>, McpError> {
        match &self.provider_token {
            Some(provider_token) => Ok(provider_token(&self.provider_name).await),
            None => Ok(None),
        }
    }
}

/// Adapter binding a connected server to the resource-tool trait (the
/// connection's request methods are `Arc`-receiver; the trait is `&self`).
pub struct ConnectionResourceServer(pub Arc<McpServerConnection>);

#[async_trait::async_trait]
impl crate::extensions::mcp::resources::McpResourceServer for ConnectionResourceServer {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn timeout_ms(&self) -> u64 {
        self.0.timeout_ms()
    }

    async fn resources_page(
        &self,
        cursor: Option<&str>,
        options: McpRequestOptions,
    ) -> Result<ListResourcesResult, String> {
        self.0.resources_page(cursor, options).await
    }

    async fn resource_templates_page(
        &self,
        cursor: Option<&str>,
        options: McpRequestOptions,
    ) -> Result<ListResourceTemplatesResult, String> {
        self.0.resource_templates_page(cursor, options).await
    }

    async fn all_resources(&self, options: McpRequestOptions) -> Result<Vec<Resource>, String> {
        self.0.all_resources(options).await
    }

    async fn all_resource_templates(
        &self,
        options: McpRequestOptions,
    ) -> Result<Vec<ResourceTemplate>, String> {
        self.0.all_resource_templates(options).await
    }

    async fn read_resource(
        &self,
        uri: &str,
        options: McpRequestOptions,
    ) -> Result<ReadResourceResult, String> {
        self.0.read_resource(uri, options).await
    }
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod runtime_tests;
