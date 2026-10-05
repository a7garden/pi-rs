//! Port of `packages/ai/src/auth/helpers.ts`.
//!
//! Standard api-key auth and the lazy OAuth wrapper. Upstream wraps a
//! dynamic `import()` of provider-specific OAuth flows; the Rust port is
//! statically linked, so [`lazy_oauth`] memoizes the loader's result and runs
//! it on first `login`/`refresh`/`to_auth` call instead.
//!
//! Deviations:
//! - D18: upstream memoizes a failed lazy load forever (`promise ??=
//!   load()`); the Rust loader is infallible (`Fn() -> OAuthAuth`), so a
//!   panicking loader is retried on the next call instead of caching a
//!   permanent failure.

use std::sync::{Arc, OnceLock};

use crate::auth::types::{
	ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, ApiKeyResolveFn, ApiLoginFn, AuthPrompt, AuthPromptKind,
	AuthResult, ModelAuth, OAuthAuth, OAuthLoginFn, OAuthRefreshFn, OAuthToAuthFn, ProviderAuthInteraction,
};
use crate::utils::thrown::ThrownValue;

/// Standard api-key auth (`envApiKeyAuth`): a stored credential key wins,
/// otherwise the first set env var resolves. Includes a `login` that prompts
/// for the key. Providers with non-standard resolution (provider env,
/// ambient files, IAM) write their own `ApiKeyAuth`.
pub fn env_api_key_auth(name: impl Into<String>, env_vars: &[&str]) -> ApiKeyAuth {
	let name = name.into();
	let env_vars: Arc<[String]> = env_vars.iter().map(|env_var| env_var.to_string()).collect();

	let login: ApiLoginFn = {
		let name = name.clone();
		Box::new(move |interaction: ProviderAuthInteraction| {
			let name = name.clone();
			Box::pin(async move {
				interaction.signal.check().map_err(ThrownValue::error)?;
				let key = (interaction.prompt)(AuthPrompt {
					signal: Some(interaction.signal.clone()),
					kind: AuthPromptKind::Secret { message: format!("Enter {name}"), placeholder: None },
				})
				.await?;
				interaction.signal.check().map_err(ThrownValue::error)?;
				Ok(ApiKeyCredential { key: Some(key), env: None })
			})
		})
	};

	let resolve: ApiKeyResolveFn = Box::new(move |input: ApiKeyAuthInput<'_>| {
		let env_vars = env_vars.clone();
		Box::pin(async move {
			input.signal.check().map_err(ThrownValue::error)?;
			// Upstream `if (credential?.key)` — an empty key is falsy and falls
			// through to the env vars.
			if let Some(key) =
				input.credential.and_then(|credential| credential.key.as_deref()).filter(|key| !key.is_empty())
			{
				return Ok(Some(AuthResult {
					auth: ModelAuth { api_key: Some(key.to_string()), ..Default::default() },
					env: input.credential.and_then(|credential| credential.env.clone()),
					source: Some("stored credential".to_string()),
				}));
			}
			for env_var in env_vars.iter() {
				let value = input.ctx.env(env_var).await;
				input.signal.check().map_err(ThrownValue::error)?;
				// Upstream `if (value)` — empty strings are skipped.
				if let Some(value) = value.filter(|value| !value.is_empty()) {
					return Ok(Some(AuthResult {
						auth: ModelAuth { api_key: Some(value), ..Default::default() },
						env: None,
						source: Some(env_var.clone()),
					}));
				}
			}
			Ok(None)
		})
	});

	ApiKeyAuth { name, login: Some(login), check: None, resolve }
}

/// Input for [`lazy_oauth`] (upstream's object parameter).
pub struct LazyOAuthInput<F> {
	pub name: String,
	/// Whether access through this auth method is backed by a subscription.
	pub is_subscription: bool,
	/// Selector label for the OAuth login option.
	pub login_label: Option<String>,
	/// Builds the underlying `OAuthAuth`; runs once on first use.
	pub load: F,
}

/// Wraps a lazily-constructed `OAuthAuth` (`lazyOAuth`) so provider
/// definitions can advertise OAuth without paying for the implementation.
/// The flow loads on first `login`/`refresh`/`to_auth` call and is memoized.
pub fn lazy_oauth<F>(input: LazyOAuthInput<F>) -> OAuthAuth
where
	F: Fn() -> OAuthAuth + Send + Sync + 'static,
{
	let loader = Arc::new(input.load);
	let cell = Arc::new(OnceLock::<Arc<OAuthAuth>>::new());
	let make_loaded = || {
		let cell = cell.clone();
		let loader = loader.clone();
		// Memoize the loaded implementation behind an Arc so the closures can
		// hand out 'static handles (upstream caches the resolved module).
		move || cell.get_or_init(move || Arc::new(loader())).clone()
	};

	let login: OAuthLoginFn = {
		let loaded = make_loaded();
		Box::new(move |interaction, options| {
			let loaded = loaded.clone();
			Box::pin(async move { (loaded().login)(interaction, options).await })
		})
	};
	let refresh: OAuthRefreshFn = {
		let loaded = make_loaded();
		Box::new(move |credential, signal| {
			let loaded = loaded.clone();
			Box::pin(async move { (loaded().refresh)(credential, signal).await })
		})
	};
	let to_auth: OAuthToAuthFn = {
		let loaded = make_loaded();
		Box::new(move |credential| {
			let loaded = loaded.clone();
			Box::pin(async move { (loaded().to_auth)(credential).await })
		})
	};

	OAuthAuth {
		name: input.name,
		is_subscription: input.is_subscription,
		login_label: input.login_label,
		login,
		refresh,
		to_auth,
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::auth::types::{AuthContext, BoxFuture, OAuthCredential, OAuthCredentials};
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

	fn input<'a>(
		ctx: Arc<dyn AuthContext>,
		credential: Option<&'a ApiKeyCredential>,
	) -> ApiKeyAuthInput<'a> {
		ApiKeyAuthInput { ctx, credential, signal: AbortController::new().signal() }
	}

	#[tokio::test]
	async fn resolve_prefers_stored_credential() {
		let auth = env_api_key_auth("Test key", &["TEST_ENV_A"]);
		let mut env = BTreeMap::new();
		env.insert("account".to_string(), "acc-1".to_string());
		let credential = ApiKeyCredential { key: Some("stored-key".to_string()), env: Some(env) };
		let result = (auth.resolve)(input(Arc::new(MapContext { vars: BTreeMap::new() }), Some(&credential)))
			.await
			.unwrap()
			.unwrap();
		assert_eq!(result.auth.api_key.as_deref(), Some("stored-key"));
		assert_eq!(result.env.as_ref().unwrap().get("account").map(String::as_str), Some("acc-1"));
		assert_eq!(result.source.as_deref(), Some("stored credential"));
	}

	#[tokio::test]
	async fn empty_stored_key_falls_through_to_env() {
		let auth = env_api_key_auth("Test key", &["TEST_ENV_A", "TEST_ENV_B"]);
		let mut vars = BTreeMap::new();
		// Empty values are skipped (upstream truthiness).
		vars.insert("TEST_ENV_A".to_string(), "".to_string());
		vars.insert("TEST_ENV_B".to_string(), "b-value".to_string());
		let credential = ApiKeyCredential { key: Some("".to_string()), env: None };
		let result = (auth.resolve)(input(Arc::new(MapContext { vars }), Some(&credential)))
			.await
			.unwrap()
			.unwrap();
		assert_eq!(result.auth.api_key.as_deref(), Some("b-value"));
		assert_eq!(result.source.as_deref(), Some("TEST_ENV_B"));
		assert_eq!(result.env, None);
	}

	#[tokio::test]
	async fn resolve_returns_none_without_sources() {
		let auth = env_api_key_auth("Test key", &["TEST_ENV_MISSING"]);
		let result =
			(auth.resolve)(input(Arc::new(MapContext { vars: BTreeMap::new() }), None)).await.unwrap();
		assert_eq!(result, None);
	}

	#[tokio::test]
	async fn login_prompts_for_secret_key() {
		let auth = env_api_key_auth("Test key", &[]);
		let interaction = ProviderAuthInteraction {
			signal: AbortController::new().signal(),
			prompt: Box::new(|prompt: AuthPrompt| {
				Box::pin(async move {
					match prompt.kind {
						AuthPromptKind::Secret { message, placeholder } => {
							assert_eq!(message, "Enter Test key");
							assert_eq!(placeholder, None);
							Ok("sk-123".to_string())
						}
						_ => panic!("expected secret prompt"),
					}
				})
			}),
			notify: Box::new(|_| {}),
		};
		let credential = (auth.login.unwrap())(interaction).await.unwrap();
		assert_eq!(credential.key.as_deref(), Some("sk-123"));
	}

	#[tokio::test]
	async fn login_rejects_when_aborted() {
		let auth = env_api_key_auth("Test key", &[]);
		let controller = AbortController::new();
		controller.abort();
		let interaction = ProviderAuthInteraction {
			signal: controller.signal(),
			prompt: Box::new(|_| Box::pin(async { Ok(String::new()) })),
			notify: Box::new(|_| {}),
		};
		assert!((auth.login.unwrap())(interaction).await.is_err());
	}

	fn test_oauth(load_calls: Arc<AtomicUsize>) -> OAuthAuth {
		OAuthAuth {
			name: "inner".to_string(),
			is_subscription: false,
			login_label: None,
			login: Box::new(|_, _| Box::pin(async { Err(ThrownValue::message("unused")) })),
			refresh: Box::new(move |mut credential, _signal| {
				let load_calls = load_calls.clone();
				Box::pin(async move {
					load_calls.fetch_add(1, Ordering::SeqCst);
					credential.credentials.expires += 60_000;
					Ok(credential)
				})
			}),
			to_auth: Box::new(|credential| {
				Box::pin(async move {
					Ok(ModelAuth { api_key: Some(credential.credentials.access.clone()), ..Default::default() })
				})
			}),
		}
	}

	fn test_oauth_credential() -> OAuthCredential {
		OAuthCredential {
			credentials: OAuthCredentials {
				refresh: "r".to_string(),
				access: "a".to_string(),
				expires: 0,
				extra: BTreeMap::new(),
			},
		}
	}

	#[tokio::test]
	async fn lazy_oauth_loads_once_on_first_use() {
		let load_calls = Arc::new(AtomicUsize::new(0));
		let lazy = lazy_oauth(LazyOAuthInput {
			name: "Lazy".to_string(),
			is_subscription: true,
			login_label: Some("Use Lazy".to_string()),
			load: {
				let load_calls = load_calls.clone();
				move || test_oauth(load_calls.clone())
			},
		});
		// Metadata is advertised without loading the implementation.
		assert_eq!(lazy.name, "Lazy");
		assert!(lazy.is_subscription);
		assert_eq!(lazy.login_label.as_deref(), Some("Use Lazy"));
		assert_eq!(load_calls.load(Ordering::SeqCst), 0);

		let credential = test_oauth_credential();
		let refreshed = (lazy.refresh)(credential.clone(), AbortController::new().signal()).await.unwrap();
		assert_eq!(refreshed.credentials.expires, 60_000);
		assert_eq!(load_calls.load(Ordering::SeqCst), 1);

		let auth = (lazy.to_auth)(&refreshed).await.unwrap();
		assert_eq!(auth.api_key.as_deref(), Some("a"));
		// Memoized: still the single load.
		assert_eq!(load_calls.load(Ordering::SeqCst), 1);
	}
}
