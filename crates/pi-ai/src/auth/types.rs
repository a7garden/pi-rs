//! Port of `packages/ai/src/auth/types.ts`.
//!
//! Auth surfaces are callback-heavy upstream; the Rust port expresses them as
//! trait objects with boxed futures (`AuthContext`, `CredentialStore`) and
//! struct-style prompt/event values. Everything here is the pinned interface
//! used by `auth/resolve.rs`, `models.rs` and the provider factories.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::types::{ProviderEnv, ProviderHeaders};
use crate::utils::abort::AbortSignal;
use crate::utils::thrown::ThrownValue;

/// Boxed sendable future (auth callbacks).
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Request auth for a single model request (`ModelAuth`). Values that cannot
/// be expressed as `apiKey`, `headers`, or `baseUrl` are provider config, not
/// auth.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelAuth {
	pub api_key: Option<String>,
	pub headers: Option<ProviderHeaders>,
	pub base_url: Option<String>,
}

/// Stored api-key credential (`ApiKeyCredential`). `env` holds
/// provider-scoped environment/config values such as Cloudflare
/// account/gateway ids.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ApiKeyCredential {
	pub key: Option<String>,
	pub env: Option<ProviderEnv>,
}

/// OAuth token data (`OAuthCredentials`). Unknown extra keys upstream are
/// preserved verbatim.
#[derive(Debug, Clone, PartialEq)]
pub struct OAuthCredentials {
	pub refresh: String,
	pub access: String,
	/// Unix timestamp in milliseconds.
	pub expires: i64,
	/// Upstream index signature `[key: string]: unknown`, preserved on
	/// round-trip.
	pub extra: BTreeMap<String, serde_json::Value>,
}

/// Stored canonical OAuth credential (`OAuthCredential`, `type: "oauth"`).
#[derive(Debug, Clone, PartialEq)]
pub struct OAuthCredential {
	pub credentials: OAuthCredentials,
}

/// One type-tagged credential per provider — the shape of today's `auth.json`
/// (`Credential`).
#[derive(Debug, Clone, PartialEq)]
pub enum Credential {
	ApiKey(ApiKeyCredential),
	OAuth(OAuthCredential),
}

impl Credential {
	pub fn type_name(&self) -> &'static str {
		match self {
			Credential::ApiKey(_) => "api_key",
			Credential::OAuth(_) => "oauth",
		}
	}
}

/// Wire (de)serialization of `Credential` matching upstream's discriminated
/// union on `type`.
impl serde::Serialize for Credential {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		use serde::ser::SerializeMap;
		let mut map = serializer.serialize_map(None)?;
		match self {
			Credential::ApiKey(credential) => {
				map.serialize_entry("type", "api_key")?;
				if let Some(key) = &credential.key {
					map.serialize_entry("key", key)?;
				}
				if let Some(env) = &credential.env {
					map.serialize_entry("env", env)?;
				}
			}
			Credential::OAuth(credential) => {
				map.serialize_entry("type", "oauth")?;
				map.serialize_entry("refresh", &credential.credentials.refresh)?;
				map.serialize_entry("access", &credential.credentials.access)?;
				map.serialize_entry("expires", &credential.credentials.expires)?;
				for (key, value) in &credential.credentials.extra {
					map.serialize_entry(key, value)?;
				}
			}
		}
		map.end()
	}
}

impl<'de> serde::Deserialize<'de> for Credential {
	fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
		let value = serde_json::Value::deserialize(deserializer).map_err(serde::de::Error::custom)?;
		let kind = value.get("type").and_then(serde_json::Value::as_str);
		match kind {
			Some("api_key") => {
				let key = value.get("key").and_then(serde_json::Value::as_str).map(str::to_string);
				let env = value
					.get("env")
					.filter(|v| v.is_object())
					.map(|v| serde_json::from_value::<ProviderEnv>(v.clone()).map_err(serde::de::Error::custom))
					.transpose()?;
				Ok(Credential::ApiKey(ApiKeyCredential { key, env }))
			}
			Some("oauth") => {
				let get_str = |name: &str| -> Result<String, String> {
					value
						.get(name)
						.and_then(serde_json::Value::as_str)
						.map(str::to_string)
						.ok_or_else(|| format!("missing field `{name}`"))
				};
				let refresh = get_str("refresh").map_err(serde::de::Error::custom)?;
				let access = get_str("access").map_err(serde::de::Error::custom)?;
				let expires = value
					.get("expires")
					.and_then(serde_json::Value::as_i64)
					.ok_or_else(|| serde::de::Error::missing_field("expires"))?;
				let mut extra = BTreeMap::new();
				if let Some(object) = value.as_object() {
					for (key, item) in object {
						if !matches!(key.as_str(), "type" | "refresh" | "access" | "expires") {
							extra.insert(key.clone(), item.clone());
						}
					}
				}
				Ok(Credential::OAuth(OAuthCredential {
					credentials: OAuthCredentials { refresh, access, expires, extra },
				}))
			}
			_ => Err(serde::de::Error::custom("credential must have type \"api_key\" or \"oauth\"")),
		}
	}
}

/// Non-secret credential metadata (`CredentialInfo`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialInfo {
	pub provider_id: String,
	#[serde(rename = "type")]
	pub credential_type: String,
}

/// Optional cancellation for public auth and credential operations
/// (`AuthOperationOptions`).
#[derive(Debug, Clone, Default)]
pub struct AuthOperationOptions {
	pub signal: Option<AbortSignal>,
}

/// App-owned credential storage (`CredentialStore`), keyed by provider id,
/// one credential per provider. `modify` is the only write path, so every
/// mutation is a serialized read-modify-write.
#[async_trait::async_trait]
pub trait CredentialStore: Send + Sync {
	/// Read the stored credential, possibly expired. `None` for missing
	/// entries; errors only on storage failure.
	async fn read(
		&self,
		provider_id: &str,
		options: Option<AuthOperationOptions>,
	) -> Result<Option<Credential>, ThrownValue>;

	/// List stored credential metadata without resolving or exposing secrets.
	async fn list(&self, options: Option<AuthOperationOptions>) -> Result<Vec<CredentialInfo>, ThrownValue>;

	/// Serialized write — the only write path. `modify_fn` sees the current
	/// credential; return the new credential, or `None` to leave the entry
	/// unchanged. Mutual exclusion per provider id.
	async fn modify(
		&self,
		provider_id: &str,
		modify_fn: CredentialModifyFn,
		options: Option<AuthOperationOptions>,
	) -> Result<Option<Credential>, ThrownValue>;

	/// Remove a credential (logout). Serializes against `modify`.
	async fn delete(&self, provider_id: &str, options: Option<AuthOperationOptions>) -> Result<(), ThrownValue>;
}

/// Closure passed to [`CredentialStore::modify`]: receives the current
/// credential, resolves to the next one (`None` leaves the entry unchanged).
pub type CredentialModifyFn = Box<
	dyn for<'a> FnOnce(Option<&'a Credential>) -> BoxFuture<'a, Result<Option<Credential>, ThrownValue>> + Send,
>;

/// Environment access for auth resolution (`AuthContext`). Injectable for
/// tests and host apps.
pub trait AuthContext: Send + Sync {
	/// Read an environment variable.
	fn env<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Option<String>>;
	/// Check whether a file exists. Supports a leading `~`.
	fn file_exists<'a>(&'a self, path: &'a str) -> BoxFuture<'a, bool>;
}

/// Result of resolving auth for a model (`AuthResult`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AuthResult {
	pub auth: ModelAuth,
	/// Provider-scoped environment/config values resolved from credentials and
	/// ambient context.
	pub env: Option<ProviderEnv>,
	/// Human-readable label for status UI: `"ANTHROPIC_API_KEY"`, `"OAuth"`,
	/// `"~/.aws/credentials"`.
	pub source: Option<String>,
}

/// Availability check result (`AuthCheck`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthCheck {
	pub source: Option<String>,
	pub check_type: AuthType,
}

/// Auth method type (`AuthType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AuthType {
	#[serde(rename = "api_key")]
	ApiKey,
	#[serde(rename = "oauth")]
	OAuth,
}

/// Prompt shown to the user during login (`AuthPrompt`). `signal` lets a flow
/// cancel a pending prompt when an out-of-band event resolves the step.
#[derive(Debug, Clone)]
pub struct AuthPrompt {
	pub signal: Option<AbortSignal>,
	pub kind: AuthPromptKind,
}

/// Prompt payload variants (`AuthPrompt` union).
#[derive(Debug, Clone)]
pub enum AuthPromptKind {
	Text { message: String, placeholder: Option<String> },
	Secret { message: String, placeholder: Option<String> },
	Select { message: String, options: Vec<AuthSelectOption> },
	ManualCode { message: String, placeholder: Option<String> },
}

/// One `select` prompt option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthSelectOption {
	pub id: String,
	pub label: String,
	pub description: Option<String>,
}

/// Link rendered with an `info` event (`AuthInfoLink`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthInfoLink {
	pub url: String,
	pub label: Option<String>,
}

/// Login flow notifications (`AuthEvent`).
#[derive(Debug, Clone)]
pub enum AuthEvent {
	Info { message: String, links: Vec<AuthInfoLink> },
	AuthUrl { url: String, instructions: Option<String> },
	DeviceCode {
		user_code: String,
		verification_uri: String,
		interval_seconds: Option<u64>,
		expires_in_seconds: Option<u64>,
	},
	Progress { message: String },
}

/// Login interaction callbacks (`AuthInteraction`).
pub struct AuthInteraction {
	pub signal: Option<AbortSignal>,
	/// Returns the entered/selected string (`select` returns the option id).
	pub prompt: Box<dyn Fn(AuthPrompt) -> BoxFuture<'static, Result<String, ThrownValue>> + Send + Sync>,
	/// Fire-and-forget notification.
	pub notify: Box<dyn Fn(AuthEvent) + Send + Sync>,
}

/// Api-key auth method (`ApiKeyAuth`).
pub struct ApiKeyAuth {
	/// Display name, e.g. "Anthropic API key".
	pub name: String,
	/// Interactive setup. `None` = ambient-only.
	pub login: Option<ApiLoginFn>,
	/// Optional side-effect-free availability check.
	pub check: Option<ApiKeyCheckFn>,
	/// Resolve auth from the stored credential and/or ambient sources,
	/// merging per field. `None` = not configured.
	pub resolve: ApiKeyResolveFn,
}

/// `login(interaction) -> ApiKeyCredential` callback.
pub type ApiLoginFn = Box<dyn Fn(ProviderAuthInteraction) -> BoxFuture<'static, Result<ApiKeyCredential, ThrownValue>> + Send + Sync>;

/// Input bag shared by `check`/`resolve`.
pub struct ApiKeyAuthInput<'a> {
	pub ctx: Arc<dyn AuthContext>,
	pub credential: Option<&'a ApiKeyCredential>,
	pub signal: AbortSignal,
}

/// `check(input) -> AuthCheck | undefined` callback.
pub type ApiKeyCheckFn =
	Box<dyn for<'a> Fn(ApiKeyAuthInput<'a>) -> BoxFuture<'a, Result<Option<AuthCheck>, ThrownValue>> + Send + Sync>;

/// `resolve(input) -> AuthResult | undefined` callback.
pub type ApiKeyResolveFn =
	Box<dyn for<'a> Fn(ApiKeyAuthInput<'a>) -> BoxFuture<'a, Result<Option<AuthResult>, ThrownValue>> + Send + Sync>;

/// App-supplied context for `Models.login` (`LoginOptions`).
#[derive(Default, Clone)]
pub struct LoginOptions {
	/// Returns the stable ID of this app installation.
	pub get_device_id: Option<Arc<dyn Fn() -> String + Send + Sync>>,
}

/// OAuth auth method (`OAuthAuth`). The `refresh`/`to_auth` split lets
/// `Models` own the locked refresh pattern.
pub struct OAuthAuth {
	/// Display name, e.g. "Anthropic (Claude Pro/Max)".
	pub name: String,
	/// Whether access through this auth method is backed by a subscription.
	pub is_subscription: bool,
	/// Selector label for the OAuth login option.
	pub login_label: Option<String>,
	pub login: OAuthLoginFn,
	/// Exchange the refresh token. Network call; errors on failure.
	pub refresh: OAuthRefreshFn,
	/// Side-effect-free derivation of request auth from a valid credential.
	pub to_auth: OAuthToAuthFn,
}

/// `login(interaction, options) -> OAuthCredential` callback.
pub type OAuthLoginFn = Box<
	dyn Fn(ProviderAuthInteraction, Option<LoginOptions>) -> BoxFuture<'static, Result<OAuthCredential, ThrownValue>>
		+ Send
		+ Sync,
>;

/// `refresh(credential, signal) -> OAuthCredential` callback.
pub type OAuthRefreshFn =
	Box<dyn Fn(OAuthCredential, AbortSignal) -> BoxFuture<'static, Result<OAuthCredential, ThrownValue>> + Send + Sync>;

/// `toAuth(credential) -> ModelAuth` callback.
pub type OAuthToAuthFn =
	Box<dyn for<'a> Fn(&'a OAuthCredential) -> BoxFuture<'a, Result<ModelAuth, ThrownValue>> + Send + Sync>;

/// Normalized interaction passed to provider login implementations
/// (`ProviderAuthInteraction`): the interaction with a guaranteed signal.
pub struct ProviderAuthInteraction {
	pub signal: AbortSignal,
	pub prompt: Box<dyn Fn(AuthPrompt) -> BoxFuture<'static, Result<String, ThrownValue>> + Send + Sync>,
	pub notify: Box<dyn Fn(AuthEvent) + Send + Sync>,
}

impl ProviderAuthInteraction {
	/// Narrow an [`AuthInteraction`] to the normalized shape, defaulting the
	/// signal like upstream's spread.
	pub fn from_interaction(interaction: AuthInteraction, signal: AbortSignal) -> ProviderAuthInteraction {
		ProviderAuthInteraction { signal, prompt: interaction.prompt, notify: interaction.notify }
	}
}

/// Provider auth methods (`ProviderAuth`). At least one of `api_key`/`oauth`
/// must be present. `oauth` is `Arc`-shared so refresh/login closures stay
/// `'static` and cloneable into concurrently running tasks (integrator
/// decision, deviation D14).
#[derive(Default)]
pub struct ProviderAuth {
	pub api_key: Option<ApiKeyAuth>,
	pub oauth: Option<Arc<OAuthAuth>>,
}
