//! Port of `packages/ai/src/auth/resolve.ts` (auth resolution shared by all
//! operations in a `Models` collection).
//!
//! A stored credential owns the provider: ambient/env is consulted only when
//! nothing is stored. No silent env fallback after a failed refresh or for a
//! credential type without a matching handler.
//!
//! Deviations:
//! - D19: the OAuth handle is `Arc<OAuthAuth>`. `refresh_stored_oauth_credential`
//!   captures it into the credential store's `'static` `modify` closure while
//!   `to_auth` still uses it afterwards; `OAuthAuth` is not `Clone`, and
//!   `ProviderAuth.oauth` stores the shared `Arc` (deviation D14 in
//!   `auth/types.rs`), so callers clone the handle out.
//! - D20: upstream bounds the refresh with `AbortSignal.timeout(15s)`. The
//!   Rust port spawns a watcher task that aborts a dedicated timeout signal
//!   after 15s; if a refresh implementation ignores the signal the awaited
//!   future is still bounded because it is dropped with the caller on abort.
//! - Abort reasons from `race_with_abort_signal` surface as
//!   `ThrownValue::Error(AbortReason)`; upstream `ModelsError` causes are
//!   folded into the message via `ModelsError::with_cause` (crate convention).

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::auth::types::{
	ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthContext, AuthOperationOptions, AuthResult, BoxFuture,
	Credential, CredentialModifyFn, CredentialStore, OAuthAuth, OAuthCredential,
};
use crate::types::ProviderEnv;
use crate::utils::abort::{
	operation_signal, race_with_abort_signal, AbortController, AbortReason, AbortSignal,
};

/// Upstream re-export: `export { ModelsError, type ModelsErrorCode } from
/// "../utils/models-error.ts"`. Also used throughout this module.
pub use crate::utils::models_error::{ModelsError, ModelsErrorCode};
use crate::utils::thrown::ThrownValue;

/// Overrides applied on top of stored credentials when resolving auth
/// (`AuthResolutionOverrides`).
#[derive(Debug, Clone, Default)]
pub struct AuthResolutionOverrides {
	pub api_key: Option<String>,
	pub env: Option<ProviderEnv>,
	/// Require this much remaining OAuth-token validity in milliseconds;
	/// defaults to five minutes (upstream `minOAuthValidityMs`).
	pub min_oauth_validity_ms: Option<i64>,
	pub signal: Option<AbortSignal>,
}

/// Structural `{ id, auth }` provider view passed to
/// [`resolve_provider_auth`] (upstream's inline parameter object).
pub struct AuthResolutionProvider<'a> {
	pub id: &'a str,
	pub api_key: Option<&'a ApiKeyAuth>,
	pub oauth: Option<Arc<OAuthAuth>>,
}

const DEFAULT_OAUTH_MINIMUM_VALIDITY_MS: i64 = 5 * 60 * 1000;
const DEFAULT_OAUTH_REFRESH_TIMEOUT_MS: u64 = 15_000;
const TIMEOUT_ABORT_MESSAGE: &str = "The operation was aborted due to timeout";

/// Auth resolution entry point (`resolveProviderAuth`).
pub async fn resolve_provider_auth(
	provider: AuthResolutionProvider<'_>,
	credentials: Arc<dyn CredentialStore>,
	auth_context: Arc<dyn AuthContext>,
	overrides: Option<AuthResolutionOverrides>,
) -> Result<Option<AuthResult>, ThrownValue> {
	let signal = operation_signal(overrides.as_ref().and_then(|overrides| overrides.signal.as_ref()));
	match race_with_abort_signal(
		resolve_provider_auth_with_signal(
			provider,
			credentials,
			auth_context,
			overrides.as_ref(),
			signal.clone(),
		),
		signal,
	)
	.await
	{
		Ok(result) => result,
		Err(reason) => Err(ThrownValue::error(reason)),
	}
}

async fn resolve_provider_auth_with_signal(
	provider: AuthResolutionProvider<'_>,
	credentials: Arc<dyn CredentialStore>,
	auth_context: Arc<dyn AuthContext>,
	overrides: Option<&AuthResolutionOverrides>,
	signal: AbortSignal,
) -> Result<Option<AuthResult>, ThrownValue> {
	signal.check().map_err(ThrownValue::error)?;
	let overrides_env = overrides.and_then(|overrides| overrides.env.clone());
	let request_auth_context: Arc<dyn AuthContext> = match &overrides_env {
		Some(env) => Arc::new(OverlayEnvAuthContext { base: auth_context, values: env.clone() }),
		None => auth_context,
	};

	// Upstream: `overrides?.apiKey !== undefined && provider.auth.apiKey` — an
	// explicitly empty key still overrides.
	if let Some(api_key) = provider.api_key {
		if let Some(override_key) = overrides.and_then(|overrides| overrides.api_key.clone()) {
			return resolve_api_key(
				request_auth_context,
				api_key,
				provider.id,
				Some(ApiKeyCredential { key: Some(override_key), env: overrides_env }),
				&signal,
			)
			.await;
		}
	}

	let stored = read_credential(&credentials, provider.id, &signal).await?;
	if let Some(stored) = stored {
		match stored {
			Credential::OAuth(oauth_credential) => {
				match provider.oauth.as_ref() {
					Some(oauth) => {
						return resolve_stored_oauth(
							&credentials,
							provider.id,
							oauth.clone(),
							oauth_credential,
							&signal,
							overrides.and_then(|overrides| overrides.min_oauth_validity_ms),
						)
						.await;
					}
					// Credential type without a matching handler.
					None => return Ok(None),
				}
			}
			Credential::ApiKey(api_key_credential) => {
				match provider.api_key {
					Some(api_key) => {
						let credential = match &overrides_env {
							Some(env) => {
								// Upstream `{ ...stored, env: { ...stored.env, ...overrides.env } }`:
								// override values win.
								let mut merged = api_key_credential.env.clone().unwrap_or_default();
								merged.extend(env.iter().map(|(name, value)| (name.clone(), value.clone())));
								ApiKeyCredential { key: api_key_credential.key.clone(), env: Some(merged) }
							}
							None => api_key_credential,
						};
						return resolve_api_key(
							request_auth_context,
							api_key,
							provider.id,
							Some(credential),
							&signal,
						)
						.await;
					}
					// Credential type without a matching handler.
					None => return Ok(None),
				}
			}
		}
	}

	// Ambient (env vars, AWS profiles, ADC files).
	match provider.api_key {
		Some(api_key) => resolve_api_key(request_auth_context, api_key, provider.id, None, &signal).await,
		None => Ok(None),
	}
}

/// Upstream `overlayEnvAuthContext`: `env[name] || base.env(name)`,
/// `fileExists` delegates.
struct OverlayEnvAuthContext {
	base: Arc<dyn AuthContext>,
	values: ProviderEnv,
}

impl AuthContext for OverlayEnvAuthContext {
	fn env<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Option<String>> {
		Box::pin(async move {
			match self.values.get(name) {
				Some(value) if !value.is_empty() => Some(value.clone()),
				_ => self.base.env(name).await,
			}
		})
	}

	fn file_exists<'a>(&'a self, path: &'a str) -> BoxFuture<'a, bool> {
		self.base.file_exists(path)
	}
}

/// Refresh a stored OAuth credential under the credential-store lock and
/// persist the result before the lock is released
/// (`refreshStoredOAuthCredential`). `needs_refresh` is re-checked under the
/// lock, so concurrent callers and processes refresh only once.
///
/// `signal` cancels only the wait for the lock. Once a refresh starts, the
/// provider may already have rotated the refresh token, so the refresh and
/// its persistence ignore `signal` and are bounded only by a timeout.
pub async fn refresh_stored_oauth_credential(
	credentials: Arc<dyn CredentialStore>,
	provider_id: &str,
	oauth: Arc<OAuthAuth>,
	needs_refresh: impl Fn(&OAuthCredential) -> bool + Send + Sync + 'static,
	signal: AbortSignal,
) -> Result<Option<OAuthCredential>, ThrownValue> {
	let lock_wait = AbortController::new();
	// Upstream `cancelLockWait`: aborting `signal` aborts the lock wait.
	let listener = {
		let watched = signal.clone();
		let lock_wait = lock_wait.clone();
		signal.add_listener(move || {
			let reason = watched.reason().unwrap_or(AbortReason::Aborted);
			lock_wait.abort_with(reason);
		})
	};
	if signal.aborted() {
		let reason = signal.reason().unwrap_or(AbortReason::Aborted);
		lock_wait.abort_with(reason);
	}

	// `removeEventListener` mapping: the disposer is consumed inside the modify
	// closure (upstream's first statement) and again after, like `finally`.
	let disposer: Arc<Mutex<Option<Box<dyn FnOnce() + Send>>>> = Arc::new(Mutex::new(Some(listener)));
	let consume_disposer = |disposer: &Arc<Mutex<Option<Box<dyn FnOnce() + Send>>>>| {
		if let Some(dispose) = disposer.lock().expect("abort listener lock").take() {
			dispose();
		}
	};

	let modify_fn: CredentialModifyFn = {
		let signal = signal.clone();
		let oauth = oauth.clone();
		let provider_id = provider_id.to_string();
		let disposer = disposer.clone();
		Box::new(move |current: Option<&Credential>| {
			let signal = signal.clone();
			let oauth = oauth.clone();
			let provider_id = provider_id.clone();
			let disposer = disposer.clone();
			Box::pin(async move {
				if let Some(dispose) = disposer.lock().expect("abort listener lock").take() {
					dispose();
				}
				signal.check().map_err(ThrownValue::error)?;
				let Some(Credential::OAuth(current)) = current else {
					return Ok(None); // logged out meanwhile (or non-OAuth entry)
				};
				if !needs_refresh(current) {
					return Ok(None); // another process/request refreshed
				}
				match refresh_with_timeout(&oauth, current).await {
					Ok(next) => Ok(Some(Credential::OAuth(next))),
					Err(error) => Err(ThrownValue::error(ModelsError::with_cause(
						ModelsErrorCode::OAuth,
						format!("OAuth refresh failed for {provider_id}"),
						Some(&error),
					))),
				}
			})
		})
	};

	let post = match credentials
		.modify(provider_id, modify_fn, Some(AuthOperationOptions { signal: Some(lock_wait.signal()) }))
		.await
	{
		Ok(post) => post,
		Err(error) => {
			consume_disposer(&disposer);
			// Upstream: `if (error instanceof ModelsError) throw error;`
			if matches!(&error, ThrownValue::Error(inner) if inner.is::<ModelsError>()) {
				return Err(error);
			}
			signal.check().map_err(ThrownValue::error)?;
			return Err(ThrownValue::error(ModelsError::with_cause(
				ModelsErrorCode::Auth,
				format!("Credential store modify failed for {provider_id}"),
				Some(&error),
			)));
		}
	};
	consume_disposer(&disposer);
	Ok(match post {
		Some(Credential::OAuth(credential)) => Some(credential),
		_ => None,
	})
}

/// OAuth resolution with double-checked locking: tokens with less than five
/// minutes remaining are refreshed through
/// [`refresh_stored_oauth_credential`] (`resolveStoredOAuth`).
async fn resolve_stored_oauth(
	credentials: &Arc<dyn CredentialStore>,
	provider_id: &str,
	oauth: Arc<OAuthAuth>,
	stored: OAuthCredential,
	signal: &AbortSignal,
	min_oauth_validity_ms: Option<i64>,
) -> Result<Option<AuthResult>, ThrownValue> {
	let minimum_validity_ms = DEFAULT_OAUTH_MINIMUM_VALIDITY_MS.max(min_oauth_validity_ms.unwrap_or(0));
	let expires_soon =
		move |credential: &OAuthCredential| unix_now_ms() + minimum_validity_ms >= credential.credentials.expires;
	let mut credential = stored;

	if expires_soon(&credential) {
		// Optimistic check said expired; the authoritative check runs under the lock.
		let Some(post) =
			refresh_stored_oauth_credential(credentials.clone(), provider_id, oauth.clone(), expires_soon, signal.clone())
				.await?
		else {
			return Ok(None); // logged out meanwhile
		};
		credential = post;
		// The normal five-minute window triggers a refresh but does not impose a
		// provider contract. Explicit callers (such as bearer-token export) do
		// require the requested minimum after the refresh.
		if min_oauth_validity_ms.is_some() && expires_soon(&credential) {
			return Err(ThrownValue::error(ModelsError::new(
				ModelsErrorCode::OAuth,
				format!("OAuth refresh returned a token that expires too soon for {provider_id}"),
			)));
		}
	}

	match (oauth.to_auth)(&credential).await {
		Ok(auth) => Ok(Some(AuthResult { auth, env: None, source: Some("OAuth".to_string()) })),
		Err(error) => Err(ThrownValue::error(ModelsError::with_cause(
			ModelsErrorCode::OAuth,
			format!("OAuth auth derivation failed for {provider_id}"),
			Some(&error),
		))),
	}
}

async fn resolve_api_key(
	auth_context: Arc<dyn AuthContext>,
	api_key: &ApiKeyAuth,
	provider_id: &str,
	credential: Option<ApiKeyCredential>,
	signal: &AbortSignal,
) -> Result<Option<AuthResult>, ThrownValue> {
	let input = ApiKeyAuthInput {
		ctx: auth_context,
		credential: credential.as_ref(),
		signal: signal.clone(),
	};
	match (api_key.resolve)(input).await {
		Ok(result) => Ok(result),
		Err(error) => Err(ThrownValue::error(ModelsError::with_cause(
			ModelsErrorCode::Auth,
			format!("API key auth failed for provider {provider_id}"),
			Some(&error),
		))),
	}
}

async fn read_credential(
	credentials: &Arc<dyn CredentialStore>,
	provider_id: &str,
	signal: &AbortSignal,
) -> Result<Option<Credential>, ThrownValue> {
	match credentials
		.read(provider_id, Some(AuthOperationOptions { signal: Some(signal.clone()) }))
		.await
	{
		Ok(credential) => Ok(credential),
		Err(error) => Err(ThrownValue::error(ModelsError::with_cause(
			ModelsErrorCode::Auth,
			format!("Credential store read failed for {provider_id}"),
			Some(&error),
		))),
	}
}

/// Upstream `AbortSignal.timeout(DEFAULT_OAUTH_REFRESH_TIMEOUT_MS)` passed to
/// `oauth.refresh`: the provider refresh runs against a signal that aborts
/// after 15 seconds (deviation D20).
async fn refresh_with_timeout(oauth: &OAuthAuth, current: &OAuthCredential) -> Result<OAuthCredential, ThrownValue> {
	let timeout_controller = AbortController::new();
	let timeout_signal = timeout_controller.signal();
	let watcher_signal = timeout_signal.clone();
	let watcher = tokio::spawn(async move {
		tokio::time::sleep(std::time::Duration::from_millis(DEFAULT_OAUTH_REFRESH_TIMEOUT_MS)).await;
		AbortSignal::abort_with(&watcher_signal, AbortReason::from(TIMEOUT_ABORT_MESSAGE));
	});
	let result = (oauth.refresh)(current.clone(), timeout_signal).await;
	watcher.abort();
	result
}

fn unix_now_ms() -> i64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map(|duration| duration.as_millis() as i64)
		.unwrap_or(0)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::auth::credential_store::InMemoryCredentialStore;
	use crate::auth::helpers::env_api_key_auth;
	use crate::auth::types::{ApiKeyCredential, ModelAuth, OAuthCredentials};
	use crate::utils::abort::AbortController;
	use std::collections::BTreeMap;
	use std::sync::atomic::{AtomicUsize, Ordering};

	struct MapContext {
		vars: BTreeMap<String, String>,
	}

	impl AuthContext for MapContext {
		fn env<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Option<String>> {
			Box::pin(async move { self.vars.get(name).cloned() })
		}

		fn file_exists<'a>(&'a self, path: &'a str) -> BoxFuture<'a, bool> {
			Box::pin(async move { path.is_empty() })
		}
	}

	fn ctx(vars: &[(&str, &str)]) -> Arc<dyn AuthContext> {
		Arc::new(MapContext {
			vars: vars.iter().map(|(name, value)| (name.to_string(), value.to_string())).collect(),
		})
	}

	fn env_map(entries: &[(&str, &str)]) -> ProviderEnv {
		entries.iter().map(|(name, value)| (name.to_string(), value.to_string())).collect()
	}

	fn api_credential(key: &str) -> Credential {
		Credential::ApiKey(ApiKeyCredential { key: Some(key.to_string()), env: None })
	}

	fn oauth_credential(expires_in_ms: i64) -> Credential {
		Credential::OAuth(OAuthCredential {
			credentials: OAuthCredentials {
				refresh: "r".to_string(),
				access: "a".to_string(),
				expires: unix_now_ms() + expires_in_ms,
				extra: BTreeMap::new(),
			},
		})
	}

	/// OAuth fixture: `refresh` bumps the expiry to `refresh_expiry_ms` from
	/// now and counts calls; `to_auth` surfaces the access token.
	fn oauth_auth(refresh_expiry_ms: i64, refresh_calls: Arc<AtomicUsize>) -> Arc<OAuthAuth> {
		Arc::new(OAuthAuth {
			name: "Test OAuth".to_string(),
			is_subscription: false,
			login_label: None,
			login: Box::new(|_, _| Box::pin(async { Err(ThrownValue::message("unused")) })),
			refresh: Box::new(move |mut credential, _signal| {
				let refresh_calls = refresh_calls.clone();
				Box::pin(async move {
					refresh_calls.fetch_add(1, Ordering::SeqCst);
					credential.credentials.expires = unix_now_ms() + refresh_expiry_ms;
					Ok(credential)
				})
			}),
			to_auth: Box::new(|credential| {
				Box::pin(async move {
					Ok(ModelAuth { api_key: Some(credential.credentials.access.clone()), ..Default::default() })
				})
			}),
		})
	}

	fn failing_oauth() -> Arc<OAuthAuth> {
		Arc::new(OAuthAuth {
			name: "Failing OAuth".to_string(),
			is_subscription: false,
			login_label: None,
			login: Box::new(|_, _| Box::pin(async { Err(ThrownValue::message("unused")) })),
			refresh: Box::new(|_credential, _signal| {
				Box::pin(async { Err(ThrownValue::message("refresh exploded")) })
			}),
			to_auth: Box::new(|_credential| {
				Box::pin(async { Err(ThrownValue::message("derive exploded")) })
			}),
		})
	}

	fn provider<'a>(
		id: &'a str,
		api_key: Option<&'a ApiKeyAuth>,
		oauth: Option<Arc<OAuthAuth>>,
	) -> AuthResolutionProvider<'a> {
		AuthResolutionProvider { id, api_key, oauth }
	}

	async fn store_with(credential: Credential) -> Arc<InMemoryCredentialStore> {
		let store = Arc::new(InMemoryCredentialStore::new());
		store
			.modify("p", Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })), None)
			.await
			.unwrap();
		store
	}

	fn assert_models_error(error: ThrownValue, code: ModelsErrorCode, message_prefix: &str) -> ModelsError {
		let ThrownValue::Error(inner) = &error else {
			panic!("expected error value, got {error:?}");
		};
		let models_error = inner.downcast_ref::<ModelsError>().unwrap_or_else(|| panic!("expected ModelsError, got {error:?}"));
		assert_eq!(models_error.code, code);
		assert!(models_error.message.starts_with(message_prefix), "message: {}", models_error.message);
		models_error.clone()
	}

	#[tokio::test]
	async fn override_api_key_wins_and_carries_override_env() {
		let auth = env_api_key_auth("K", &["TEST_RESOLVE_A"]);
		let store = store_with(api_credential("stored")).await;
		let result = resolve_provider_auth(
			provider("p", Some(&auth), Some(oauth_auth(600_000, Arc::new(AtomicUsize::new(0))))),
			store,
			ctx(&[]),
			Some(AuthResolutionOverrides {
				api_key: Some("override".to_string()),
				env: Some(env_map(&[("ov", "1")])),
				..Default::default()
			}),
		)
		.await
		.unwrap()
		.unwrap();
		assert_eq!(result.auth.api_key.as_deref(), Some("override"));
		assert_eq!(result.source.as_deref(), Some("stored credential"));
		assert_eq!(result.env.as_ref().unwrap().get("ov").map(String::as_str), Some("1"));
	}

	#[tokio::test]
	async fn stored_credential_wins_over_ambient_and_merges_override_env() {
		let auth = env_api_key_auth("K", &["TEST_RESOLVE_A"]);
		let stored = Credential::ApiKey(ApiKeyCredential {
			key: Some("stored".to_string()),
			env: Some(env_map(&[("a", "1")])),
		});
		let store = store_with(stored).await;
		let result = resolve_provider_auth(
			provider("p", Some(&auth), None),
			store,
			ctx(&[("TEST_RESOLVE_A", "ambient")]),
			Some(AuthResolutionOverrides { env: Some(env_map(&[("b", "2")])), ..Default::default() }),
		)
		.await
		.unwrap()
		.unwrap();
		assert_eq!(result.auth.api_key.as_deref(), Some("stored"));
		assert_eq!(result.env, Some(env_map(&[("a", "1"), ("b", "2")])));
		assert_eq!(result.source.as_deref(), Some("stored credential"));
	}

	#[tokio::test]
	async fn ambient_env_resolution_labels_source() {
		let auth = env_api_key_auth("K", &["TEST_RESOLVE_MISSING", "TEST_RESOLVE_B"]);
		let result = resolve_provider_auth(
			provider("p", Some(&auth), None),
			Arc::new(InMemoryCredentialStore::new()),
			ctx(&[("TEST_RESOLVE_B", "b-value")]),
			None,
		)
		.await
		.unwrap()
		.unwrap();
		assert_eq!(result.auth.api_key.as_deref(), Some("b-value"));
		assert_eq!(result.source.as_deref(), Some("TEST_RESOLVE_B"));
		assert_eq!(result.env, None);
	}

	#[tokio::test]
	async fn ambient_resolution_with_no_sources_returns_none() {
		let auth = env_api_key_auth("K", &["TEST_RESOLVE_MISSING"]);
		let result = resolve_provider_auth(
			provider("p", Some(&auth), None),
			Arc::new(InMemoryCredentialStore::new()),
			ctx(&[]),
			None,
		)
		.await
		.unwrap();
		assert_eq!(result, None);
	}

	#[tokio::test]
	async fn stored_credential_type_without_handler_returns_none() {
		// Stored OAuth credential, provider only supports api keys.
		let auth = env_api_key_auth("K", &[]);
		let result = resolve_provider_auth(
			provider("p", Some(&auth), None),
			store_with(oauth_credential(600_000)).await,
			ctx(&[]),
			None,
		)
		.await
		.unwrap();
		assert_eq!(result, None);

		// Stored api-key credential, provider only supports OAuth.
		let oauth = oauth_auth(600_000, Arc::new(AtomicUsize::new(0)));
		let result = resolve_provider_auth(
			provider("p", None, Some(oauth)),
			store_with(api_credential("k")).await,
			ctx(&[]),
			None,
		)
		.await
		.unwrap();
		assert_eq!(result, None);
	}

	#[tokio::test]
	async fn valid_oauth_token_skips_refresh() {
		let refresh_calls = Arc::new(AtomicUsize::new(0));
		let oauth = oauth_auth(600_000, refresh_calls.clone());
		let result = resolve_provider_auth(
			provider("p", None, Some(oauth)),
			store_with(oauth_credential(600_000)).await,
			ctx(&[]),
			None,
		)
		.await
		.unwrap()
		.unwrap();
		assert_eq!(result.auth.api_key.as_deref(), Some("a"));
		assert_eq!(result.source.as_deref(), Some("OAuth"));
		assert_eq!(result.env, None);
		assert_eq!(refresh_calls.load(Ordering::SeqCst), 0);
	}

	#[tokio::test]
	async fn expired_oauth_token_refreshes_once_under_lock() {
		let refresh_calls = Arc::new(AtomicUsize::new(0));
		let oauth = oauth_auth(600_000, refresh_calls.clone());
		let store = store_with(oauth_credential(-1)).await;
		for _ in 0..2 {
			let result = resolve_provider_auth(provider("p", None, Some(oauth.clone())), store.clone(), ctx(&[]), None)
				.await
				.unwrap()
				.unwrap();
			assert_eq!(result.auth.api_key.as_deref(), Some("a"));
			assert_eq!(result.source.as_deref(), Some("OAuth"));
		}
		// The refreshed token is persisted, so the second call skips the refresh
		// (double-checked locking).
		assert_eq!(refresh_calls.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn oauth_refresh_failure_wraps_in_models_error() {
		let error = resolve_provider_auth(
			provider("p", None, Some(failing_oauth())),
			store_with(oauth_credential(-1)).await,
			ctx(&[]),
			None,
		)
		.await
		.unwrap_err();
		let models_error = assert_models_error(error, ModelsErrorCode::OAuth, "OAuth refresh failed for p");
		assert!(models_error.message.contains("refresh exploded"));
	}

	#[tokio::test]
	async fn oauth_to_auth_failure_wraps_in_models_error() {
		// `to_auth` fails, but the token is valid so no refresh runs.
		let error = resolve_provider_auth(
			provider("p", None, Some(failing_oauth())),
			store_with(oauth_credential(600_000)).await,
			ctx(&[]),
			None,
		)
		.await
		.unwrap_err();
		assert_models_error(error, ModelsErrorCode::OAuth, "OAuth auth derivation failed for p");
	}

	#[tokio::test]
	async fn min_validity_enforced_after_refresh() {
		let refresh_calls = Arc::new(AtomicUsize::new(0));
		// Refresh returns a token valid for 7 minutes; the caller requires 10.
		let oauth = oauth_auth(7 * 60 * 1000, refresh_calls.clone());
		let error = resolve_provider_auth(
			provider("p", None, Some(oauth)),
			store_with(oauth_credential(6 * 60 * 1000)).await,
			ctx(&[]),
			Some(AuthResolutionOverrides { min_oauth_validity_ms: Some(10 * 60 * 1000), ..Default::default() }),
		)
		.await
		.unwrap_err();
		assert_models_error(
			error,
			ModelsErrorCode::OAuth,
			"OAuth refresh returned a token that expires too soon for p",
		);
		assert_eq!(refresh_calls.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn refresh_stored_returns_none_without_oauth_credential() {
		let signal = AbortController::new().signal();
		// Missing entry.
		let result = refresh_stored_oauth_credential(
			Arc::new(InMemoryCredentialStore::new()),
			"p",
			oauth_auth(600_000, Arc::new(AtomicUsize::new(0))),
			|_| true,
			signal.clone(),
		)
		.await
		.unwrap();
		assert_eq!(result, None);
		// Non-OAuth entry (logged out meanwhile).
		let result = refresh_stored_oauth_credential(
			store_with(api_credential("k")).await,
			"p",
			oauth_auth(600_000, Arc::new(AtomicUsize::new(0))),
			|_| true,
			signal,
		)
		.await
		.unwrap();
		assert_eq!(result, None);
	}

	#[tokio::test]
	async fn refresh_stored_respects_needs_refresh() {
		let refresh_calls = Arc::new(AtomicUsize::new(0));
		let result = refresh_stored_oauth_credential(
			store_with(oauth_credential(600_000)).await,
			"p",
			oauth_auth(600_000, refresh_calls.clone()),
			|_| false,
			AbortController::new().signal(),
		)
		.await
		.unwrap();
		// Upstream: the modify fn returns undefined (entry unchanged), the
		// store resolves `next ?? current`, and the still-valid OAuth
		// credential flows back out unrefreshed.
		let expected = match oauth_credential(600_000) {
			Credential::OAuth(credential) => Some(credential),
			other => panic!("expected oauth credential, got {other:?}"),
		};
		assert_eq!(result, expected);
		assert_eq!(refresh_calls.load(Ordering::SeqCst), 0);
	}

	#[tokio::test]
	async fn credential_store_read_failure_wraps_in_models_error() {
		struct FailingStore;
		#[async_trait::async_trait]
		impl CredentialStore for FailingStore {
			async fn read(
				&self,
				_provider_id: &str,
				_options: Option<AuthOperationOptions>,
			) -> Result<Option<Credential>, ThrownValue> {
				Err(ThrownValue::message("io"))
			}
			async fn list(
				&self,
				_options: Option<AuthOperationOptions>,
			) -> Result<Vec<crate::auth::types::CredentialInfo>, ThrownValue> {
				Err(ThrownValue::message("io"))
			}
			async fn modify(
				&self,
				_provider_id: &str,
				_modify_fn: CredentialModifyFn,
				_options: Option<AuthOperationOptions>,
			) -> Result<Option<Credential>, ThrownValue> {
				Err(ThrownValue::message("io"))
			}
			async fn delete(
				&self,
				_provider_id: &str,
				_options: Option<AuthOperationOptions>,
			) -> Result<(), ThrownValue> {
				Err(ThrownValue::message("io"))
			}
		}

		let auth = env_api_key_auth("K", &[]);
		let error = resolve_provider_auth(
			provider("p", Some(&auth), None),
			Arc::new(FailingStore),
			ctx(&[]),
			None,
		)
		.await
		.unwrap_err();
		let models_error = assert_models_error(error, ModelsErrorCode::Auth, "Credential store read failed for p");
		assert!(models_error.message.contains("io"));
	}

	#[tokio::test]
	async fn api_key_resolve_failure_wraps_in_models_error() {
		let failing = ApiKeyAuth {
			name: "Failing".to_string(),
			login: None,
			check: None,
			resolve: Box::new(|_input| Box::pin(async { Err(ThrownValue::message("nope")) })),
		};
		let error = resolve_provider_auth(
			provider("p", Some(&failing), None),
			Arc::new(InMemoryCredentialStore::new()),
			ctx(&[]),
			None,
		)
		.await
		.unwrap_err();
		assert_models_error(error, ModelsErrorCode::Auth, "API key auth failed for provider p");
	}

	#[tokio::test]
	async fn aborted_signal_fails_resolution() {
		let auth = env_api_key_auth("K", &["TEST_RESOLVE_A"]);
		let controller = AbortController::new();
		controller.abort();
		let result = resolve_provider_auth(
			provider("p", Some(&auth), None),
			Arc::new(InMemoryCredentialStore::new()),
			ctx(&[("TEST_RESOLVE_A", "v")]),
			Some(AuthResolutionOverrides { signal: Some(controller.signal()), ..Default::default() }),
		)
		.await;
		assert!(result.is_err());
	}

	#[tokio::test]
	async fn override_env_takes_precedence_over_context_env() {
		let auth = env_api_key_auth("K", &["TEST_RESOLVE_A"]);
		let result = resolve_provider_auth(
			provider("p", Some(&auth), None),
			Arc::new(InMemoryCredentialStore::new()),
			ctx(&[("TEST_RESOLVE_A", "ctx-value")]),
			Some(AuthResolutionOverrides {
				env: Some(env_map(&[("TEST_RESOLVE_A", "override-value")])),
				..Default::default()
			}),
		)
		.await
		.unwrap()
		.unwrap();
		assert_eq!(result.auth.api_key.as_deref(), Some("override-value"));
		assert_eq!(result.source.as_deref(), Some("TEST_RESOLVE_A"));
	}

	#[tokio::test]
	async fn override_env_empty_value_falls_through_to_context() {
		let auth = env_api_key_auth("K", &["TEST_RESOLVE_A"]);
		let result = resolve_provider_auth(
			provider("p", Some(&auth), None),
			Arc::new(InMemoryCredentialStore::new()),
			ctx(&[("TEST_RESOLVE_A", "ctx-value")]),
			Some(AuthResolutionOverrides {
				env: Some(env_map(&[("TEST_RESOLVE_A", "")])),
				..Default::default()
			}),
		)
		.await
		.unwrap()
		.unwrap();
		assert_eq!(result.auth.api_key.as_deref(), Some("ctx-value"));
	}
}
