//! Port of `packages/ai/src/providers/anthropic.ts`.
//!
//! Deviations:
//! - D15: upstream `oauth: lazyOAuth(...)` is omitted — `auth/oauth/load.ts`
//!   (the Anthropic OAuth flow) is out of scope for this wave; the provider
//!   advertises api-key auth only.
//! - D16: the `ANTHROPIC_*_ENV` name constants (upstream `env-api-keys.ts`,
//!   out of scope) are defined locally as `pub const`s with upstream names
//!   and values.
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`crate::models::create_provider`] returns `Result`, unwrapped with
//!   `expect` because the configuration here is static and valid.

use std::sync::Arc;

use futures_util::future::BoxFuture;

use crate::auth::types::{
	ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthPrompt, AuthPromptKind, AuthResult, ModelAuth, ProviderAuth,
	ProviderAuthInteraction,
};
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};
use crate::types::ProviderEnv;
use crate::utils::thrown::ThrownValue;

/// Upstream `ANTHROPIC_AUTH_TOKEN_ENV`.
pub const ANTHROPIC_AUTH_TOKEN_ENV: &str = "ANTHROPIC_AUTH_TOKEN";
/// Upstream `ANTHROPIC_OAUTH_TOKEN_ENV`.
pub const ANTHROPIC_OAUTH_TOKEN_ENV: &str = "ANTHROPIC_OAUTH_TOKEN";
/// Upstream `ANTHROPIC_API_KEY_ENV`.
pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
/// Upstream `ANTHROPIC_FEDERATION_RULE_ID_ENV`.
pub const ANTHROPIC_FEDERATION_RULE_ID_ENV: &str = "ANTHROPIC_FEDERATION_RULE_ID";
/// Upstream `ANTHROPIC_ORGANIZATION_ID_ENV`.
pub const ANTHROPIC_ORGANIZATION_ID_ENV: &str = "ANTHROPIC_ORGANIZATION_ID";
/// Upstream `ANTHROPIC_SERVICE_ACCOUNT_ID_ENV`.
pub const ANTHROPIC_SERVICE_ACCOUNT_ID_ENV: &str = "ANTHROPIC_SERVICE_ACCOUNT_ID";
/// Upstream `ANTHROPIC_IDENTITY_TOKEN_FILE_ENV`.
pub const ANTHROPIC_IDENTITY_TOKEN_FILE_ENV: &str = "ANTHROPIC_IDENTITY_TOKEN_FILE";
/// Upstream `ANTHROPIC_WORKSPACE_ID_ENV`.
pub const ANTHROPIC_WORKSPACE_ID_ENV: &str = "ANTHROPIC_WORKSPACE_ID";

/// Upstream `signal.throwIfAborted()` (`Err(reason)` when aborted).
fn check_aborted(signal: &crate::utils::abort::AbortSignal) -> Result<(), ThrownValue> {
	signal.check().map_err(ThrownValue::error)
}

/// The `login` of [`anthropic_api_key_auth`].
fn login_anthropic(interaction: ProviderAuthInteraction) -> BoxFuture<'static, Result<ApiKeyCredential, ThrownValue>> {
	Box::pin(async move {
		check_aborted(&interaction.signal)?;
		let key = (interaction.prompt)(AuthPrompt {
			signal: Some(interaction.signal.clone()),
			kind: AuthPromptKind::Secret { message: "Enter Anthropic API key".to_string(), placeholder: None },
		})
		.await?;
		check_aborted(&interaction.signal)?;
		Ok(ApiKeyCredential { key: Some(key), env: None })
	})
}

/// The `resolve` entry of [`anthropic_api_key_auth`].
fn resolve_anthropic_entry<'a>(
	input: ApiKeyAuthInput<'a>,
) -> BoxFuture<'a, Result<Option<AuthResult>, ThrownValue>> {
	Box::pin(async move { resolve_anthropic(&input).await })
}

/// Upstream `anthropicApiKeyAuth()`.
fn anthropic_api_key_auth() -> ApiKeyAuth {
	ApiKeyAuth {
		name: "Anthropic API key".to_string(),
		login: Some(Box::new(login_anthropic)),
		check: None,
		resolve: Box::new(resolve_anthropic_entry),
	}
}

/// The `resolve` body of [`anthropic_api_key_auth`], kept as a free function
/// so the lifetime-generic closure type stays nameable.
async fn resolve_anthropic(input: &ApiKeyAuthInput<'_>) -> Result<Option<AuthResult>, ThrownValue> {
	let ApiKeyAuthInput { ctx, credential, signal } = input;
	check_aborted(signal)?;
	if let Some(key) = credential.as_ref().and_then(|credential| credential.key.as_deref()) {
		return Ok(Some(AuthResult {
			auth: ModelAuth { api_key: Some(key.to_string()), headers: None, base_url: None },
			env: credential.as_ref().and_then(|credential| credential.env.clone()),
			source: Some("stored credential".to_string()),
		}));
	}

	let auth_token = ctx.env(ANTHROPIC_AUTH_TOKEN_ENV).await;
	check_aborted(signal)?;
	if let Some(auth_token) = auth_token {
		return Ok(Some(AuthResult {
			auth: ModelAuth {
				headers: Some(
					[("Authorization".to_string(), Some(format!("Bearer {auth_token}")))].into_iter().collect(),
				),
				..ModelAuth::default()
			},
			env: None,
			source: Some(ANTHROPIC_AUTH_TOKEN_ENV.to_string()),
		}));
	}

	for env_var in [ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV] {
		let api_key = ctx.env(env_var).await;
		check_aborted(signal)?;
		if let Some(api_key) = api_key {
			return Ok(Some(AuthResult {
				auth: ModelAuth { api_key: Some(api_key), ..ModelAuth::default() },
				env: None,
				source: Some(env_var.to_string()),
			}));
		}
	}

	// Workload identity federation: the Anthropic SDK exchanges the identity
	// token for a short-lived access token and refreshes it itself. Last in
	// line so keys and ANTHROPIC_AUTH_TOKEN keep winning, as in the SDK. The
	// ids are provider config rather than auth, so they travel in `env`.
	let mut federation = ProviderEnv::new();
	for env_var in [ANTHROPIC_FEDERATION_RULE_ID_ENV, ANTHROPIC_ORGANIZATION_ID_ENV, ANTHROPIC_IDENTITY_TOKEN_FILE_ENV] {
		let value = ctx.env(env_var).await;
		check_aborted(signal)?;
		let Some(value) = value else { return Ok(None) };
		federation.insert(env_var.to_string(), value);
	}
	for env_var in [ANTHROPIC_SERVICE_ACCOUNT_ID_ENV, ANTHROPIC_WORKSPACE_ID_ENV] {
		let value = ctx.env(env_var).await;
		check_aborted(signal)?;
		if let Some(value) = value {
			federation.insert(env_var.to_string(), value);
		}
	}
	Ok(Some(AuthResult {
		auth: ModelAuth::default(),
		env: Some(federation),
		source: Some("workload identity federation".to_string()),
	}))
}

/// Upstream `anthropicProvider()`. The OAuth auth method is omitted (D15).
pub fn anthropic_provider() -> Arc<dyn Provider> {
	create_provider(CreateProviderOptions {
		id: "anthropic".to_string(),
		name: Some("Anthropic".to_string()),
		base_url: Some("https://api.anthropic.com".to_string()),
		headers: None,
		auth: ProviderAuth { api_key: Some(anthropic_api_key_auth()), oauth: None },
		models: super::anthropic_models::anthropic_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::Single(crate::api::anthropic_messages::anthropic_messages_api())),
		images: Default::default(),
		classifiers: Default::default(),
	})
	.expect("static anthropic provider configuration is valid")
}
