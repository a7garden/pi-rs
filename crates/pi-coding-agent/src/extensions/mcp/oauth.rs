//! Port of `packages/coding-agent/src/extensions/mcp/oauth.ts`: OAuth sign-in
//! for remote MCP servers, built on [`pi_mcp::oauth`] (the provider, flow,
//! discovery, and callback server live there; this module is the
//! coding-agent-specific glue: the token store in `mcp-auth.json`, per-server
//! provider wiring, and refresh handling).
//!
//! # Port deviations
//!
//! - **D83.** The upstream `AuthStorageBackend` is not object-safe (D70a), so
//!   [`McpOAuthCredentialStore`] is generic over the backend; the
//!   connection/runtime layers erase the per-server stores behind
//!   `Arc<dyn McpOAuthServerStore>`.
//! - **D84.** Upstream's `proper-lockfile` refresh lock becomes a
//!   create-exclusive lock file in the lock directory with mtime-based stale
//!   detection (same 20s stale / 25s wait / 100ms retry constants). A lock its
//!   holder stopped renewing (the process was killed) is taken over the same
//!   way.
//! - **D85.** `URL` → the `url` crate; `URL.canParse` → `Url::parse`; the
//!   `[::1]` loopback hostname is matched literally.
//! - **D86.** `sha256(...).toString("base64url")` and the hex state use local
//!   encoders (no `base64` dependency; `pi-mcp` uses the same approach).

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::future::BoxFuture;
use futures_util::FutureExt;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use url::Url;

use pi_mcp::oauth::{
	authorize_mcp, parse_www_authenticate, step_up_scope, McpOAuthAuthorizationRequiredError, McpOAuthProvider,
	McpOAuthProviderOptions, McpOAuthState, McpOAuthStateStore, OAuthCallback, OAuthCallbackPage,
	OAuthCallbackServer, OAuthCallbackServerOptions, OAuthChallenge, OAuthClientInformationMixed,
	OAuthClientProvider,
	OAuthClientMetadata, OAuthClientMetadataDocument, OAuthFlowOptions, OAuthFlowResult,
};
use pi_mcp::protocol::jsonrpc::McpError;
use pi_mcp::AuthProvider;

use crate::config::{get_agent_dir, APP_NAME};
use crate::core::auth_storage::{AuthStorageBackend, FileAuthStorageBackend, LockResult};
use crate::extensions::mcp::config::mcp_namespace;

pub const CALLBACK_HOST: &str = "127.0.0.1";
pub const CALLBACK_PATH: &str = "/callback";
/// Where pi.dev serves pi's Client ID Metadata Documents: `client.json` and
/// `<callback ID>/client.json`.
pub const CLIENT_METADATA_BASE_URL: &str = "https://pi.dev/oauth";
/// Redirect URI for refreshes when none is stored. Refreshing never redirects
/// the user.
pub const FALLBACK_REDIRECT_URL: &str = "http://127.0.0.1/callback";
/// Access tokens this close to expiry are refreshed before they are sent.
pub const REFRESH_SKEW_MS: u128 = 30_000;
/// Bounds each request of a refresh, so it cannot hold the refresh lock or
/// delay shutdown for long.
pub const REFRESH_REQUEST_TIMEOUT_MS: u64 = 15_000;
/// A refresh lock that its holder stopped renewing (the process was killed) is
/// taken over after this.
pub const REFRESH_LOCK_STALE_MS: u64 = 20_000;
/// How long to wait for another process's refresh: longer than a stale lock
/// lives.
pub const REFRESH_LOCK_WAIT_MS: u64 = 25_000;
pub const REFRESH_LOCK_RETRY_MS: u64 = 100;

/// Upstream `McpOAuthSettings`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct McpOAuthSettings {
	pub client_id: Option<String>,
	/// Already resolved.
	pub client_secret: Option<String>,
	pub callback_port: Option<u16>,
	/// Loopback redirect URI; see `McpOAuthConfig.callbackUrl`.
	pub callback_url: Option<String>,
	/// Scopes to request, separated by spaces.
	pub scope: Option<String>,
	/// `client_name` for dynamic client registration. Default: `APP_NAME`.
	pub client_name: Option<String>,
	/// See `McpOAuthConfig.clientRegistration`.
	pub client_registration: Option<String>,
	/// See `McpOAuthConfig.authServerMetadataUrl`.
	pub auth_server_metadata_url: Option<Url>,
}

/// Where the loopback callback server listens and the redirect URI it serves
/// (upstream `CallbackSettings`).
struct CallbackSettings {
	host: String,
	redirect_host: String,
	port: Option<u16>,
	path: String,
	/// The exact redirect URI, when the port is fixed.
	fixed_redirect_url: Option<String>,
}

fn callback_settings(settings: &McpOAuthSettings) -> Result<CallbackSettings, McpError> {
	let base = settings
		.callback_url
		.clone()
		.unwrap_or_else(|| format!("http://{CALLBACK_HOST}{CALLBACK_PATH}"));
	let mut url = Url::parse(&base).map_err(|error| McpError::other(error.to_string()))?;
	let address = url
		.host_str()
		.unwrap_or_default()
		.trim_start_matches('[')
		.trim_end_matches(']')
		.to_string();
	let port = url.port().or(settings.callback_port);
	let mut fixed_redirect_url: Option<String> = None;
	// A configured URI with a port is sent exactly as written, since servers
	// compare it as a string.
	if url.port().is_some() {
		fixed_redirect_url = Some(settings.callback_url.clone().expect("checked above"));
	} else if let Some(port) = port {
		url.set_port(Some(port)).map_err(|_| McpError::other("cannot set the callback port"))?;
		fixed_redirect_url = Some(url.to_string());
	}
	Ok(CallbackSettings {
		// `localhost` is served on 127.0.0.1; browsers fall back to it when ::1 refuses.
		redirect_host: address.clone(),
		host: if address == "localhost" { CALLBACK_HOST.to_string() } else { address },
		port,
		path: url.path().to_string(),
		fixed_redirect_url,
	})
}

/// Scopes of both lists, each once (`mergeScopes`).
pub fn merge_scopes(scopes: &[Option<&str>]) -> Option<String> {
	let mut merged: Vec<String> = Vec::new();
	for scope in scopes.iter().flatten() {
		for part in scope.split_whitespace() {
			if !merged.iter().any(|known| known == part) {
				merged.push(part.to_string());
			}
		}
	}
	if merged.is_empty() {
		None
	} else {
		Some(merged.join(" "))
	}
}

type StoredStates = Map<String, Value>;

fn parse_states(content: Option<&str>) -> StoredStates {
	let Some(content) = content else {
		return Map::new();
	};
	if content.trim().is_empty() {
		return Map::new();
	}
	match serde_json::from_str::<Value>(content) {
		Ok(Value::Object(states)) => states,
		_ => Map::new(),
	}
}

fn serialize_states(states: &StoredStates) -> String {
	format!("{}\n", serde_json::to_string_pretty(&Value::Object(states.clone())).expect("states serialization is infallible"))
}

/// Keys of a server's state: by name and URL, so servers sharing a URL keep
/// separate accounts, and the legacy key by URL alone, written by older
/// versions (`storeKeys`).
fn store_keys(name: &str, server_url: &str) -> (String, String) {
	let legacy_key = Url::parse(server_url)
		.map(|url| url.to_string())
		.unwrap_or_else(|_| server_url.to_string());
	(format!("{}|{}", mcp_namespace(name), legacy_key), legacy_key)
}

/// Upstream `McpOAuthServerStore`: a per-server state store plus the refresh
/// lock. The closure form keeps the trait object-safe (the Rust backend lock
/// is not object-safe — D83).
pub trait McpOAuthServerStore: McpOAuthStateStore {
	/// Run `f` while no other process refreshes the server's tokens.
	fn with_refresh_lock(
		&self,
		f: BoxFuture<'static, Result<(), McpError>>,
	) -> BoxFuture<'static, Result<(), McpError>>;
}

impl<B: AuthStorageBackend + 'static> McpOAuthCredentialStore<B> {
	/// `forServer` through a shared reference (the generic store is not
	/// object-safe, D83 — callers erase behind [`McpOAuthServerStore`]).
	pub fn for_server_borrowed(&self, name: &str, server_url: &str) -> Arc<dyn McpOAuthServerStore> {
		let (key, legacy_key) = store_keys(name, server_url);
		Arc::new(PerServerStore {
			backend: Arc::clone(&self.backend),
			lock_dir: self.lock_dir.clone(),
			key,
			legacy_key,
		})
	}
}

/// Per-server OAuth state (client registration, tokens, pending PKCE
/// verifier) in `mcp-auth.json` (`McpOAuthCredentialStore`).
pub struct McpOAuthCredentialStore<B: AuthStorageBackend = FileAuthStorageBackend> {
	backend: Arc<B>,
	/// Directory for the refresh lock files. Without one, refreshes are only
	/// serialized in this process.
	lock_dir: Option<String>,
}

impl McpOAuthCredentialStore<FileAuthStorageBackend> {
	/// Upstream `new McpOAuthCredentialStore()`: `mcp-auth.json` in the agent
	/// directory, refresh locks beside it.
	pub fn open_default() -> McpOAuthCredentialStore<FileAuthStorageBackend> {
		McpOAuthCredentialStore::new(
			Arc::new(FileAuthStorageBackend::new(&join_path(&get_agent_dir(), "mcp-auth.json"))),
			Some(get_agent_dir()),
		)
	}
}

impl<B: AuthStorageBackend + 'static> McpOAuthCredentialStore<B> {
	/// With an explicit backend, `lock_dir` defaults to the agent directory
	/// (upstream: `backend ? lockDir : getAgentDir()`).
	pub fn new(backend: Arc<B>, lock_dir: Option<String>) -> McpOAuthCredentialStore<B> {
		// Upstream: `backend ? lockDir : getAgentDir()` — an explicit backend
		// with no lock directory refreshes without a cross-process lock.
		McpOAuthCredentialStore { backend, lock_dir }
	}

	/// The per-server store. The first server to load legacy state takes it
	/// over; others with the same URL sign in again (`forServer`).
	pub fn for_server(self: &Arc<Self>, name: &str, server_url: &str) -> Arc<dyn McpOAuthServerStore> {
		let (key, legacy_key) = store_keys(name, server_url);
		Arc::new(PerServerStore {
			backend: Arc::clone(&self.backend),
			lock_dir: self.lock_dir.clone(),
			key,
			legacy_key,
		})
	}

	/// The stored tokens of a server, for noticing sign-ins done by another
	/// process. Does not take over legacy state (`tokens`).
	pub fn tokens(&self, name: &str, server_url: &str) -> Option<Value> {
		let (key, legacy_key) = store_keys(name, server_url);
		let states = self.read_states();
		states
			.get(&key)
			.or_else(|| states.get(&legacy_key))
			.and_then(|state| state.get("tokens").cloned())
	}

	/// Returns whether credentials were stored for the server. Removes legacy
	/// state the server would take over (`remove`).
	pub fn remove(&self, name: &str, server_url: &str) -> bool {
		let (key, legacy_key) = store_keys(name, server_url);
		self.write_states(|states| {
			let stored = if states.contains_key(&key) {
				Some(key.clone())
			} else if states.contains_key(&legacy_key) {
				Some(legacy_key.clone())
			} else {
				None
			};
			match stored {
				None => (false, false),
				Some(stored) => {
					states.remove(&stored);
					(true, true)
				}
			}
		})
	}

	fn read_states(&self) -> StoredStates {
		self.backend
			.with_lock(|current| LockResult { result: parse_states(current.as_deref()), next: None })
			.unwrap_or_default()
	}

	fn write_states(&self, update: impl FnOnce(&mut StoredStates) -> (bool, bool) + Send) -> bool {
		self.backend
			.with_lock(|current| {
				let mut states = parse_states(current.as_deref());
				let (changed, result) = update(&mut states);
				LockResult {
					result,
					next: if changed { Some(serialize_states(&states)) } else { None },
				}
			})
			.unwrap_or(false)
	}

	fn lock_dir(&self) -> Option<&str> {
		self.lock_dir.as_deref()
	}
}

/// Upstream `forServer(...)`'s object literal, as a trait object.
struct PerServerStore<B: AuthStorageBackend> {
	backend: Arc<B>,
	lock_dir: Option<String>,
	key: String,
	legacy_key: String,
}

#[async_trait::async_trait]
impl<B: AuthStorageBackend + 'static> McpOAuthStateStore for PerServerStore<B> {
		async fn load(&self) -> Result<Option<McpOAuthState>, McpError> {
		let outer = Arc::clone(&self.backend);
		let key = self.key.clone();
		let legacy_key = self.legacy_key.clone();
		let value = tokio::task::spawn_blocking(move || {
			outer.with_lock(|current| {
				let mut states = parse_states(current.as_deref());
				if states.contains_key(&key) {
					return LockResult { result: states.get(&key).cloned(), next: None };
				}
				let Some(legacy) = states.remove(&legacy_key) else {
					return LockResult { result: None, next: None };
				};
				states.insert(key.clone(), legacy.clone());
				LockResult { result: Some(legacy), next: Some(serialize_states(&states)) }
			})
		})
		.await
		.map_err(|error| McpError::other(error.to_string()))?
		.map_err(|error: pi_ai::utils::thrown::ThrownValue| McpError::other(match error {
			pi_ai::utils::thrown::ThrownValue::Text(message) => message,
			pi_ai::utils::thrown::ThrownValue::Error(error) => error.to_string(),
		}))?;
		let Some(value) = value else {
			return Ok(None);
		};
		serde_json::from_value::<McpOAuthState>(value).map(Some).map_err(|error| McpError::other(error.to_string()))
	}

	async fn save(&self, state: &McpOAuthState) -> Result<(), McpError> {
		let outer = Arc::clone(&self.backend);
		let key = self.key.clone();
		let value = serde_json::to_value(state).map_err(|error| McpError::other(error.to_string()))?;
		let saved = tokio::task::spawn_blocking(move || {
			outer.with_lock(|current| {
				let mut states = parse_states(current.as_deref());
				states.insert(key.clone(), value.clone());
				LockResult { result: true, next: Some(serialize_states(&states)) }
			})
			.unwrap_or(false)
		})
		.await
		.map_err(|error| McpError::other(error.to_string()))?;
		if saved {
			Ok(())
		} else {
			Err(McpError::other("MCP OAuth state could not be saved"))
		}
	}
}

impl<B: AuthStorageBackend + 'static> McpOAuthServerStore for PerServerStore<B> {
	fn with_refresh_lock(
		&self,
		f: BoxFuture<'static, Result<(), McpError>>,
	) -> BoxFuture<'static, Result<(), McpError>> {
		let lock_dir = self.lock_dir.clone();
		let key = self.key.clone();
		Box::pin(async move {
			let Some(lock_dir) = lock_dir else {
				return f.await;
			};
			let hash = hex_encode(&Sha256::digest(key.as_bytes())[..16]);
			let path = format!("{lock_dir}/mcp-auth-refresh-{hash}");
			let _guard = acquire_refresh_lock(&path).await?;
			f.await
		})
	}
}

/// Upstream `proper-lockfile` stand-in (D84): an exclusive lock file, retried
/// every [`REFRESH_LOCK_RETRY_MS`] for [`REFRESH_LOCK_WAIT_MS`], with a lock
/// older than [`REFRESH_LOCK_STALE_MS`] taken over.
async fn acquire_refresh_lock(path: &str) -> Result<RefreshLockGuard, McpError> {
	let deadline = std::time::Instant::now() + Duration::from_millis(REFRESH_LOCK_WAIT_MS);
	loop {
		match std::fs::OpenOptions::new().write(true).create_new(true).open(path) {
			Ok(_) => return Ok(RefreshLockGuard { path: path.to_string() }),
			Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
				let stale = std::fs::metadata(path)
					.and_then(|metadata| metadata.modified())
					.ok()
					.and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
					.is_some_and(|age| {
						let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
						now.saturating_sub(age).as_millis() > REFRESH_LOCK_STALE_MS as u128
					});
				if stale {
					// A lost lock at worst lets two refreshes overlap (upstream
					// `onCompromised: () => {}`).
					let _ = std::fs::remove_file(path);
				}
			}
			Err(error) => {
				if let Some(parent) = std::path::Path::new(path).parent() {
					let _ = std::fs::create_dir_all(parent);
					// Permissions 0o700 like upstream `mkdirSync(..., { mode: 0o700 })`.
					let _ = std::fs::set_permissions(parent, std::os::unix::fs::PermissionsExt::from_mode(0o700));
				}
				if std::fs::OpenOptions::new().write(true).create_new(true).open(path).is_ok() {
					return Ok(RefreshLockGuard { path: path.to_string() });
				}
				return Err(McpError::other(error.to_string()));
			}
		}
		if std::time::Instant::now() >= deadline {
			return Err(McpError::other("timed out waiting for the MCP token refresh lock"));
		}
		tokio::time::sleep(Duration::from_millis(REFRESH_LOCK_RETRY_MS)).await;
	}
}

struct RefreshLockGuard {
	path: String,
}

impl Drop for RefreshLockGuard {
	fn drop(&mut self) {
		// When the process exits, the lock file it holds is removed with it
		// (upstream: proper-lockfile removes locks on exit; stale detection
		// covers a killed process).
		let _ = std::fs::remove_file(&self.path);
	}
}

fn registered_redirect_urls(client: Option<&OAuthClientInformationMixed>) -> Vec<String> {
	match client {
		Some(OAuthClientInformationMixed::Full(full)) => full.metadata.redirect_uris.clone(),
		_ => Vec::new(),
	}
}

fn hex_encode(bytes: &[u8]) -> String {
	bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// 12 characters identifying an MCP server URL in callback paths, computed
/// like Codex does (`callbackId`).
pub fn callback_id(server_url: &str) -> String {
	let mut url = Url::parse(server_url).expect("server URL");
	url.set_fragment(None);
	base64_url_no_pad(&Sha256::digest(url.to_string().as_bytes())[..9])
}

fn base64_url_no_pad(bytes: &[u8]) -> String {
	const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
	let mut output = String::new();
	for chunk in bytes.chunks(3) {
		let buffer = [
			chunk[0],
			chunk.get(1).copied().unwrap_or(0),
			chunk.get(2).copied().unwrap_or(0),
		];
		let value = (u32::from(buffer[0]) << 16) | (u32::from(buffer[1]) << 8) | u32::from(buffer[2]);
		output.push(ALPHABET[(value >> 18) as usize & 63] as char);
		output.push(ALPHABET[(value >> 12) as usize & 63] as char);
		if chunk.len() > 1 {
			output.push(ALPHABET[(value >> 6) as usize & 63] as char);
		}
		if chunk.len() > 2 {
			output.push(ALPHABET[value as usize & 63] as char);
		}
	}
	output
}

/// pi's Client ID Metadata Document, for `clientRegistration: "cimd"`, chosen
/// like Codex chooses its own (`clientMetadataDocument`). The configuration
/// ensures the default callback path. Without the `iss` parameter in
/// authorization responses (RFC 9207), the redirect URI and the document are
/// specific to the MCP server, so a response cannot be mixed up with one from
/// another authorization server (RFC 9700 section 4.4.2.2).
pub fn client_metadata_document(
	server_url: &str,
	redirect_url: &str,
	metadata: Option<&pi_mcp::oauth::AuthorizationServerMetadata>,
) -> Result<OAuthClientMetadataDocument, McpError> {
	let supported = metadata
		.is_some_and(|metadata| metadata.client_id_metadata_document_supported == Some(true))
		&& metadata
			.and_then(|metadata| metadata.token_endpoint_auth_methods_supported.as_ref())
			.is_some_and(|methods| methods.iter().any(|method| method == "none"));
	if !supported {
		return Err(McpError::other(
			"The authorization server does not support Client ID Metadata Documents for public clients; remove oauth.clientRegistration \"cimd\"",
		));
	}
	if metadata.and_then(|metadata| metadata.authorization_response_iss_parameter_supported) == Some(true) {
		return Ok(OAuthClientMetadataDocument {
			url: format!("{CLIENT_METADATA_BASE_URL}/client.json"),
			redirect_url: redirect_url.to_string(),
		});
	}
	let id = callback_id(server_url);
	let mut redirect = Url::parse(redirect_url).map_err(|error| McpError::other(error.to_string()))?;
	redirect.set_path(&format!("{CALLBACK_PATH}/{id}"));
	Ok(OAuthClientMetadataDocument {
		url: format!("{CLIENT_METADATA_BASE_URL}/{id}/client.json"),
		redirect_url: redirect.to_string(),
	})
}

fn create_provider(
	server_url: &str,
	store: Arc<dyn McpOAuthStateStore>,
	settings: &McpOAuthSettings,
	redirect_url: &str,
	on_redirect: Arc<dyn Fn(Url) -> BoxFuture<'static, Result<(), McpError>> + Send + Sync>,
) -> McpOAuthProvider {
	let server_url_for_document = server_url.to_string();
	let redirect_url_for_document = redirect_url.to_string();
	let is_cimd = settings.client_registration.as_deref() == Some("cimd");
	McpOAuthProvider::new(McpOAuthProviderOptions {
		server_url: Url::parse(server_url).expect("server URL"),
		redirect_url: redirect_url.to_string(),
		client_metadata: OAuthClientMetadata {
			client_name: Some(settings.client_name.clone().unwrap_or_else(|| APP_NAME.to_string())),
			..Default::default()
		},
		client_metadata_document: if is_cimd {
			Some(Arc::new(move |metadata| {
				client_metadata_document(&server_url_for_document, &redirect_url_for_document, metadata)
					.map(Some)
			}))
		} else {
			None
		},
		client_id: settings.client_id.clone(),
		client_secret: settings.client_secret.clone(),
		store: Some(store),
		on_redirect,
	})
}

/// Bounds every request of a refresh (`AbortSignal.timeout(...)` upstream).
fn with_request_timeout(fetch: pi_mcp::McpFetch, timeout_ms: u64) -> pi_mcp::McpFetch {
	Arc::new(move |request| {
		let fetch = Arc::clone(&fetch);
		Box::pin(async move {
			let future = fetch(request);
			tokio::time::timeout(Duration::from_millis(timeout_ms), future)
				.await
				.map_err(|_| McpError::other("MCP token refresh request timed out"))?
		})
	})
}

/// Auth provider for MCP connections (`createMcpAuthProvider`): sends the
/// stored access token and refreshes it when it is about to expire or after a
/// 401. Errors with `McpOAuthAuthorizationRequiredError` when the user has to
/// sign in, including when the server asks for more scope
/// (`insufficient_scope`). `on_challenge` receives the server's
/// `WWW-Authenticate` challenge so sign-in can use its resource metadata URL
/// and scope. `settings` is only called when a refresh is needed, so a secret
/// that fails to resolve fails the refresh instead of the whole connection
/// setup.
///
/// Many servers rotate refresh tokens, so two refreshes with the same refresh
/// token lose the grant. Requests in this process share one refresh, and other
/// processes are kept out by the store's refresh lock, held from reading the
/// tokens to saving new ones. Tokens that changed meanwhile (another process
/// refreshed them, or the user signed in) are used without refreshing.
pub struct McpAuthProvider {
	server_url: String,
	store: Arc<dyn McpOAuthServerStore>,
	settings: Arc<dyn Fn() -> McpOAuthSettings + Send + Sync>,
	on_challenge: Arc<dyn Fn(&OAuthChallenge) + Send + Sync>,
	refreshing: tokio::sync::Mutex<Option<futures_util::future::Shared<BoxFuture<'static, Result<(), McpError>>>>>,
}

pub struct McpAuthProviderOptions {
	pub server_url: String,
	pub store: Arc<dyn McpOAuthServerStore>,
	pub settings: Arc<dyn Fn() -> McpOAuthSettings + Send + Sync>,
	pub on_challenge: Arc<dyn Fn(&OAuthChallenge) + Send + Sync>,
}

impl McpAuthProvider {
	pub fn new(options: McpAuthProviderOptions) -> Arc<McpAuthProvider> {
		Arc::new(McpAuthProvider {
			server_url: options.server_url,
			store: options.store,
			settings: options.settings,
			on_challenge: options.on_challenge,
			refreshing: tokio::sync::Mutex::new(None),
		})
	}

	/// Replace `stale_token`, the access token that expired or was rejected
	/// (upstream `refresh`).
	fn refresh(
		self: &Arc<Self>,
		stale_token: Option<String>,
		challenge: Option<OAuthChallenge>,
	) -> BoxFuture<'static, Result<(), McpError>> {
		let provider = Arc::clone(self);
		Box::pin(async move {
			let existing = provider.refreshing.lock().await.clone();
			if let Some(existing) = existing {
				return existing.await;
			}
			let store = Arc::clone(&provider.store);
			let inner_provider = Arc::clone(&provider);
			let challenge = challenge.clone();
			let future: BoxFuture<'static, Result<(), McpError>> = Box::pin(async move {
				store
					.with_refresh_lock(Box::pin(async move {
						let state = inner_provider.store.load().await?;
						let state = match state {
							Some(state) => state,
							None => return Ok(()),
						};
						let access_token = state.tokens.as_ref().map(|tokens| tokens.access_token.clone());
						if access_token.as_deref() != stale_token.as_deref().filter(|token| !token.is_empty()) {
							return Ok(());
						}
						let refresh_token = state
							.tokens
							.as_ref()
							.and_then(|tokens| tokens.refresh_token.clone())
							.ok_or(McpError::OAuthAuthorizationRequired(McpOAuthAuthorizationRequiredError))?;
						let settings = (inner_provider.settings)();
						let callback = callback_settings(&settings)?;
						let client = state.client_information.as_ref();
						let redirect_url = callback
							.fixed_redirect_url
							.clone()
							.or_else(|| registered_redirect_urls(client).first().cloned())
							.unwrap_or_else(|| FALLBACK_REDIRECT_URL.to_string());
						let provider = Arc::new(create_provider(
							&inner_provider.server_url,
							Arc::clone(&inner_provider.store) as Arc<dyn McpOAuthStateStore>,
							&settings,
							&redirect_url,
							Arc::new(|_| Box::pin(async { Ok(()) })),
						));
						// Refreshes the tokens, or reports that a new sign-in is needed.
						let result = authorize_mcp(
							provider,
							OAuthFlowOptions {
								server_url: Url::parse(&inner_provider.server_url).expect("server URL"),
								resource_metadata_url: challenge
									.as_ref()
									.and_then(|challenge| challenge.resource_metadata_url.clone()),
								authorization_server_metadata_url: settings.auth_server_metadata_url.clone(),
								scope: challenge.as_ref().and_then(|challenge| challenge.scope.clone()),
								fetch: Some(with_request_timeout(
									pi_mcp::auth_provider::default_fetch(),
									REFRESH_REQUEST_TIMEOUT_MS,
								)),
								skip_issuer_validation: false,
								skip_refresh: false,
								authorization_code: None,
								iss: None,
							},
						)
						.await?;
						if result == OAuthFlowResult::Redirect {
							return Err(McpError::OAuthAuthorizationRequired(McpOAuthAuthorizationRequiredError));
						}
						Ok(())
					}))
					.await?;
					Ok(())
			});
			let shared: futures_util::future::Shared<BoxFuture<'static, Result<(), McpError>>> =
				future.shared();
			*provider.refreshing.lock().await = Some(shared.clone());
			let result = shared.await;
			*provider.refreshing.lock().await = None;
			result
		})
	}

	/// Resolves when no refresh is running, so shutdown does not drop rotated
	/// tokens before they are saved (`settled`).
	pub async fn settled(&self) {
		let existing = self.refreshing.lock().await.clone();
		if let Some(existing) = existing {
			let _ = existing.await;
		}
	}

}

/// `Arc<dyn AuthProvider>` stand-in: the trait's `&self` methods need the
/// single-flight refresh to recover the owning `Arc`, so the connection layers
/// wrap their provider in this wrapper (D83).
pub struct SharedMcpAuthProvider(pub Arc<McpAuthProvider>);

#[async_trait::async_trait]
impl AuthProvider for SharedMcpAuthProvider {
	async fn token(&self) -> Result<Option<String>, McpError> {
		let provider = &self.0;
		if let Some(refreshing) = provider.refreshing.lock().await.clone() {
			let _ = refreshing.await;
		}
		let state = provider.store.load().await?;
		let Some(state) = state else {
			return Ok(None);
		};
		let Some(tokens) = &state.tokens else {
			return Ok(None);
		};
		let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
		let expired = state
			.tokens_expire_at
			.map(|expires_at| expires_at.saturating_sub(REFRESH_SKEW_MS) <= now)
			.unwrap_or(false);
		if !expired || tokens.refresh_token.is_none() {
			return Ok(Some(tokens.access_token.clone()));
		}
		// Failures fall through: the request goes out with the old token and a
		// 401 decides what happens.
		let _ = provider.refresh(Some(tokens.access_token.clone()), None).await;
		let state = provider.store.load().await?;
		Ok(state.and_then(|state| state.tokens.map(|tokens| tokens.access_token)))
	}

	fn handles_unauthorized(&self) -> bool {
		true
	}

	async fn on_unauthorized(&self, context: pi_mcp::UnauthorizedContext) -> Result<(), McpError> {
		let provider = &self.0;
		let challenge = parse_www_authenticate(context.response.header("www-authenticate"));
		(provider.on_challenge)(&challenge);
		// A refresh keeps the granted scope, so more scope needs a new sign-in.
		if challenge.error.as_deref() == Some("insufficient_scope") {
			return Err(McpError::OAuthAuthorizationRequired(McpOAuthAuthorizationRequiredError));
		}
		provider.refresh(context.token, Some(challenge)).await
	}
}

/// Resolves when no refresh is running; see [`McpAuthProvider::settled`].
pub async fn provider_settled(provider: &Arc<McpAuthProvider>) {
	provider.settled().await
}

/// Upstream `McpSignInPrompt`.
#[derive(Clone)]
pub struct McpSignInPrompt {
	/// Show the authorization URL to the user and open it in a browser.
	pub show_authorization_url: Arc<dyn Fn(Url) + Send + Sync>,
	/// Ask for the redirect URL from the browser address bar, for when the
	/// browser cannot reach the loopback callback (for example over SSH).
	/// Resolving to `None` cancels the sign-in.
	pub prompt_for_redirect_url: Arc<
		dyn Fn(pi_mcp::client::AbortSignal) -> BoxFuture<'static, Option<String>> + Send + Sync,
	>,
}

type AuthorizationResponse = OAuthCallback;

fn response_from_redirect_url(
	input: &str,
	state: &str,
	redirect_url: &Url,
) -> Result<AuthorizationResponse, McpError> {
	let url = Url::parse(input.trim()).map_err(|_| McpError::other("Expected the full redirect URL from the browser address bar"))?;
	// A server-specific redirect URI tells authorization servers apart, so it
	// must match exactly.
	if url.origin().ascii_serialization() != redirect_url.origin().ascii_serialization()
		|| url.path() != redirect_url.path()
	{
		return Err(McpError::other("The redirect URL does not match this sign-in's redirect URI"));
	}
	let error = query_param(&url, "error");
	if let Some(error) = error {
		let description = query_param(&url, "error_description").unwrap_or(error.clone());
		return Err(McpError::other(description));
	}
	if query_param(&url, "state").as_deref() != Some(state) {
		return Err(McpError::other("The redirect URL belongs to a different sign-in"));
	}
	let Some(code) = query_param(&url, "code") else {
		return Err(McpError::other("The redirect URL does not contain an authorization code"));
	};
	Ok(OAuthCallback { code, state: state.to_string(), iss: query_param(&url, "iss") })
}

fn query_param(url: &Url, key: &str) -> Option<String> {
	url.query_pairs().find(|(name, _)| name == key).map(|(_, value)| value.into_owned())
}

/// Wait for the browser callback or a pasted redirect URL, whichever comes
/// first (`waitForAuthorizationResponse`).
async fn wait_for_authorization_response(
	callback: &Arc<OAuthCallbackServer>,
	state: &str,
	redirect_url: &Url,
	prompt: &McpSignInPrompt,
) -> Result<AuthorizationResponse, McpError> {
	let controller = pi_mcp::client::AbortController::new();
	let from_browser = callback.wait_for_callback(state, Some(redirect_url.path()));
	let from_user = {
		let prompt = prompt.clone();
		let state = state.to_string();
		let redirect_url = redirect_url.clone();
		let signal = controller.signal();
		async move {
			let Some(input) = (prompt.prompt_for_redirect_url)(signal).await else {
				return Err(McpError::other(SIGN_IN_CANCELLED_MESSAGE));
			};
			response_from_redirect_url(&input, &state, &redirect_url)
		}
	};
	let result = tokio::select! {
		result = from_browser => result,
		result = from_user => result,
	};
	// The losing side rejects once the prompt is aborted or the callback
	// server closes.
	controller.abort(Some("authorization response arrived".to_string()));
	result
}

/// Listen on `port`, or on a free port when it is taken and not `required`
/// (`listenForCallback`).
async fn listen_for_callback(
	settings: &CallbackSettings,
	extra_paths: Vec<String>,
	port: Option<u16>,
	required: bool,
) -> Result<Arc<OAuthCallbackServer>, McpError> {
	let options = |port: Option<u16>| OAuthCallbackServerOptions {
		host: Some(settings.host.clone()),
		redirect_host: Some(settings.redirect_host.clone()),
		port,
		path: Some(settings.path.clone()),
		extra_paths: extra_paths.clone(),
		timeout_ms: None,
		render_page: None,
	};
	match OAuthCallbackServer::listen(options(port)).await {
		Ok(server) => Ok(server),
		Err(error) => {
			if required || port.is_none() {
				Err(error)
			} else {
				Ok(OAuthCallbackServer::listen(options(None)).await?)
			}
		}
	}
}

pub const SIGN_IN_CANCELLED_MESSAGE: &str = "Sign-in cancelled";

/// Sign in to an MCP server. Uses the stored refresh token when possible;
/// otherwise runs the browser authorization code flow. Tokens are saved to
/// `store` (`signInMcpServer`).
pub async fn sign_in_mcp_server(options: McpSignInOptions) -> Result<(), McpError> {
	let McpSignInOptions { server_url, store, settings, challenge, prompt } = options;
	let stored = store.load().await?;
	let step_up = challenge
		.as_ref()
		.and_then(|challenge| challenge.error.as_deref())
		== Some("insufficient_scope");
	let callback_options = callback_settings(&settings)?;
	// Reuse the port of the registered redirect URI so the registered client
	// stays valid.
	let registered = registered_redirect_urls(stored.as_ref().and_then(|state| state.client_information.as_ref()));
	let preferred_port = callback_options.port.or_else(|| {
		registered
			.first()
			.and_then(|uri| Url::parse(uri).ok())
			.and_then(|url| url.port())
	});
	let cimd = settings.client_registration.as_deref() == Some("cimd");
	let callback = listen_for_callback(
		&callback_options,
		// The redirect URI of a server-specific Client ID Metadata Document.
		if cimd { vec![format!("{CALLBACK_PATH}/{}", callback_id(&server_url))] } else { vec![] },
		preferred_port,
		callback_options.port.is_some(),
	)
	.await?;
	let result = sign_in_with_callback(SignInWithCallback {
		server_url: &server_url,
		store: &store,
		settings: &settings,
		challenge: challenge.as_ref(),
		prompt: &prompt,
		stored,
		step_up,
		callback: &callback,
		fixed_redirect_url: callback_options.fixed_redirect_url.clone(),
	})
	.await;
	callback.close().await?;
	result
}

pub struct McpSignInOptions {
	pub server_url: String,
	pub store: Arc<dyn McpOAuthStateStore>,
	pub settings: McpOAuthSettings,
	pub challenge: Option<OAuthChallenge>,
	pub prompt: McpSignInPrompt,
}

struct SignInWithCallback<'a> {
	server_url: &'a str,
	store: &'a Arc<dyn McpOAuthStateStore>,
	settings: &'a McpOAuthSettings,
	challenge: Option<&'a OAuthChallenge>,
	prompt: &'a McpSignInPrompt,
	stored: Option<McpOAuthState>,
	step_up: bool,
	callback: &'a Arc<OAuthCallbackServer>,
	fixed_redirect_url: Option<String>,
}

async fn sign_in_with_callback(options: SignInWithCallback<'_>) -> Result<(), McpError> {
	let SignInWithCallback { server_url, store, settings, challenge, prompt, stored, step_up, callback, fixed_redirect_url } =
		options;
	let redirect_url = fixed_redirect_url.unwrap_or_else(|| callback.redirect_url.clone());
	// The granted scope, as it was when the sign-in started (a step-up request
	// adds the challenged scope on top of it).
	let granted_scope = stored
		.as_ref()
		.and_then(|state| state.tokens.as_ref())
		.and_then(|tokens| tokens.scope.clone());
	if let Some(stored) = stored {
		let mut next = stored;
		// Every sign-in gets a fresh `state` parameter.
		next.oauth_state = None;
		// A registered client cannot use another redirect URI, and its tokens
		// belong to it. A Client ID Metadata Document is not stored, so with
		// one, a stored client was registered before and is replaced.
		let registered = registered_redirect_urls(next.client_information.as_ref());
		let keep_client = settings.client_id.is_some()
			|| if settings.client_registration.as_deref() == Some("cimd") {
				next.client_information.is_none()
			} else {
				registered.iter().any(|uri| uri == &redirect_url)
			};
		if !keep_client {
			next.client_information = None;
			next.tokens = None;
			next.tokens_expire_at = None;
		}
		store.save(&next).await?;
	}

	let authorization_url: Arc<parking_lot::Mutex<Option<Url>>> = Arc::new(parking_lot::Mutex::new(None));
	let authorization_url_sink = Arc::clone(&authorization_url);
	let provider: Arc<dyn OAuthClientProvider> = Arc::new(create_provider(
		server_url,
		Arc::clone(store),
		settings,
		&redirect_url,
		Arc::new(move |url| {
			*authorization_url_sink.lock() = Some(url.clone());
			Box::pin(async { Ok(()) })
		}),
	));
	// A server asking for more scope gets it on top of the configured scope
	// and, since the challenge may list only the missing scopes, on top of the
	// scope granted so far.
	let scope = merge_scopes(&[
		settings.scope.as_deref(),
		(if step_up {
			step_up_scope(granted_scope.as_deref(), challenge.and_then(|challenge| challenge.scope.as_deref()))
		} else {
			challenge.and_then(|challenge| challenge.scope.clone())
		})
		.as_deref(),
	]);
	let flow = OAuthFlowOptions {
		server_url: Url::parse(server_url).expect("server URL"),
		resource_metadata_url: challenge.and_then(|challenge| challenge.resource_metadata_url.clone()),
		authorization_server_metadata_url: settings.auth_server_metadata_url.clone(),
		scope,
		// A refresh keeps the granted scope; a server asking for more needs the
		// browser flow.
		skip_issuer_validation: false,
		skip_refresh: step_up,
		authorization_code: None,
		iss: None,
		fetch: None,
	};
	if authorize_mcp(Arc::clone(&provider), flow.clone()).await? == OAuthFlowResult::Authorized {
		return Ok(());
	}
	let authorization_url_value = authorization_url
		.lock()
		.clone()
		.ok_or_else(|| McpError::other("OAuth flow did not produce an authorization URL"))?;
	let state = provider
		.state()
		.await?
		.ok_or_else(|| McpError::other("OAuth flow did not produce a state parameter"))?;
	// The flow picks the redirect URI, which may be specific to the MCP server.
	let authorization_redirect_url = query_param(&authorization_url_value, "redirect_uri")
		.and_then(|value| Url::parse(&value).ok())
		.unwrap_or_else(|| Url::parse(&redirect_url).expect("redirect URL"));
	(prompt.show_authorization_url)(authorization_url_value.clone());
	let response = wait_for_authorization_response(callback, &state, &authorization_redirect_url, prompt).await?;
	let flow = OAuthFlowOptions { authorization_code: Some(response.code), iss: response.iss, ..flow };
	authorize_mcp(provider, flow).await?;
	Ok(())
}

fn join_path(base: &str, suffix: &str) -> String {
	std::path::Path::new(base).join(suffix).to_string_lossy().into_owned()
}

/// Upstream `McpSignInCancelledError` (message marker; the store/UI layers
/// match on [`SIGN_IN_CANCELLED_MESSAGE`]).
pub fn is_sign_in_cancelled(error: &McpError) -> bool {
	matches!(error, McpError::Other(message) if message == SIGN_IN_CANCELLED_MESSAGE)
}

/// Render the callback page like upstream `oauthSuccessHtml`/`oauthErrorHtml`
/// without importing the TUI-adjacent HTML helpers: keep the plain-text
/// defaults of [`pi_mcp::oauth::OAuthCallbackPage`] (D87).
pub fn render_callback_page(page: &OAuthCallbackPage) -> String {
	match page {
		OAuthCallbackPage::Ok => {
			"Signed in to the MCP server. You may now close this page.".to_string()
		}
		OAuthCallbackPage::Failure { message, details } => match details {
			Some(details) => format!("{message}\n\n{details}"),
			None => message.clone(),
		},
	}
}

#[cfg(test)]
#[path = "oauth_tests.rs"]
mod oauth_tests;
