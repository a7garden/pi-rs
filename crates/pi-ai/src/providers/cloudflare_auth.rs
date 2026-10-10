//! Port of `packages/ai/src/providers/cloudflare-auth.ts`.
//!
//! Deviations (per-file ledger): none beyond the crate-wide conventions (the
//! `check` slot is `None` because upstream cloudflare auth declares no check;
//! `signal.throwIfAborted()` maps to [`check_aborted`]).

use std::sync::Arc;

use crate::auth::types::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, ApiKeyResolveFn, AuthPrompt, AuthPromptKind,
    AuthResult, BoxFuture, ModelAuth, ProviderAuthInteraction,
};
use crate::types::ProviderEnv;
use crate::utils::abort::AbortSignal;
use crate::utils::thrown::ThrownValue;

const CLOUDFLARE_API_KEY: &str = "CLOUDFLARE_API_KEY";
const CLOUDFLARE_ACCOUNT_ID: &str = "CLOUDFLARE_ACCOUNT_ID";
const CLOUDFLARE_GATEWAY_ID: &str = "CLOUDFLARE_GATEWAY_ID";

/// Upstream `type CloudflareAuthKind = "workers-ai" | "ai-gateway"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CloudflareAuthKind {
    WorkersAi,
    AiGateway,
}

/// Upstream `signal.throwIfAborted()` (`Err(reason)` when aborted).
fn check_aborted(signal: &AbortSignal) -> Result<(), ThrownValue> {
    signal.check().map_err(ThrownValue::error)
}

/// Per-field merge of a stored credential and the ambient context
/// (`resolveValue`): prefer the credential value, fall back to ambient env. A
/// credential carrying only the API key must still pick up the account /
/// gateway id from the environment.
async fn resolve_value(
    name: &str,
    ctx: &Arc<dyn crate::auth::types::AuthContext>,
    credential: Option<&ApiKeyCredential>,
    signal: &AbortSignal,
) -> Result<Option<String>, ThrownValue> {
    let from_credential = credential.and_then(|credential| {
        if name == CLOUDFLARE_API_KEY {
            credential.key.clone()
        } else {
            credential
                .env
                .as_ref()
                .and_then(|env| env.get(name).cloned())
        }
    });
    if from_credential.is_some() {
        return Ok(from_credential);
    }
    check_aborted(signal)?;
    let value = ctx.env(name).await;
    check_aborted(signal)?;
    Ok(value)
}

/// One resolved Cloudflare environment (`resolveCloudflareEnv` return shape).
struct ResolvedCloudflareEnv {
    api_key: String,
    env: ProviderEnv,
    source: &'static str,
}

async fn resolve_cloudflare_env(
    kind: CloudflareAuthKind,
    ctx: &Arc<dyn crate::auth::types::AuthContext>,
    credential: Option<&ApiKeyCredential>,
    signal: &AbortSignal,
) -> Result<Option<ResolvedCloudflareEnv>, ThrownValue> {
    let api_key = resolve_value(CLOUDFLARE_API_KEY, ctx, credential, signal).await?;
    let account_id = resolve_value(CLOUDFLARE_ACCOUNT_ID, ctx, credential, signal).await?;
    let gateway_id = match kind {
        CloudflareAuthKind::AiGateway => {
            resolve_value(CLOUDFLARE_GATEWAY_ID, ctx, credential, signal).await?
        }
        CloudflareAuthKind::WorkersAi => None,
    };

    // Upstream truthiness: empty strings do not count as configured.
    let truthy = |value: Option<String>| value.filter(|value| !value.is_empty());
    let (Some(api_key), Some(account_id)) = (truthy(api_key), truthy(account_id)) else {
        return Ok(None);
    };
    let gateway_id = truthy(gateway_id);
    if kind == CloudflareAuthKind::AiGateway && gateway_id.is_none() {
        return Ok(None);
    }

    let mut env = ProviderEnv::new();
    env.insert(CLOUDFLARE_ACCOUNT_ID.to_string(), account_id);
    if let Some(gateway_id) = gateway_id {
        env.insert(CLOUDFLARE_GATEWAY_ID.to_string(), gateway_id);
    }
    Ok(Some(ResolvedCloudflareEnv {
        api_key,
        env,
        source: if credential.is_some() {
            "stored credential"
        } else {
            CLOUDFLARE_API_KEY
        },
    }))
}

/// Sequential login prompts (`interaction.prompt` upstream).
async fn prompt(
    interaction: &ProviderAuthInteraction,
    kind: AuthPromptKind,
) -> Result<String, ThrownValue> {
    (interaction.prompt)(AuthPrompt {
        signal: Some(interaction.signal.clone()),
        kind,
    })
    .await
}

/// The `login` of [`cloudflare_workers_ai_auth`].
async fn login_workers_ai(
    interaction: ProviderAuthInteraction,
) -> Result<ApiKeyCredential, ThrownValue> {
    let key = prompt(
        &interaction,
        AuthPromptKind::Secret {
            message: "Enter Cloudflare API key".to_string(),
            placeholder: None,
        },
    )
    .await?;
    let account_id = prompt(
        &interaction,
        AuthPromptKind::Text {
            message: "Enter Cloudflare account ID".to_string(),
            placeholder: None,
        },
    )
    .await?;
    Ok(ApiKeyCredential {
        key: Some(key),
        env: Some(
            [(CLOUDFLARE_ACCOUNT_ID.to_string(), account_id)]
                .into_iter()
                .collect(),
        ),
    })
}

/// The `login` of [`cloudflare_ai_gateway_auth`].
async fn login_ai_gateway(
    interaction: ProviderAuthInteraction,
) -> Result<ApiKeyCredential, ThrownValue> {
    let key = prompt(
        &interaction,
        AuthPromptKind::Secret {
            message: "Enter Cloudflare API key".to_string(),
            placeholder: None,
        },
    )
    .await?;
    let account_id = prompt(
        &interaction,
        AuthPromptKind::Text {
            message: "Enter Cloudflare account ID".to_string(),
            placeholder: None,
        },
    )
    .await?;
    let gateway_id = prompt(
        &interaction,
        AuthPromptKind::Text {
            message: "Enter Cloudflare AI Gateway ID".to_string(),
            placeholder: None,
        },
    )
    .await?;
    Ok(ApiKeyCredential {
        key: Some(key),
        env: Some(
            [
                (CLOUDFLARE_ACCOUNT_ID.to_string(), account_id),
                (CLOUDFLARE_GATEWAY_ID.to_string(), gateway_id),
            ]
            .into_iter()
            .collect(),
        ),
    })
}

/// The `resolve` body of [`cloudflare_workers_ai_auth`], kept as a free
/// function so the lifetime-generic closure type stays nameable.
async fn resolve_workers_ai(
    input: &ApiKeyAuthInput<'_>,
) -> Result<Option<AuthResult>, ThrownValue> {
    let resolved = resolve_cloudflare_env(
        CloudflareAuthKind::WorkersAi,
        &input.ctx,
        input.credential,
        &input.signal,
    )
    .await?;
    Ok(resolved.map(|resolved| AuthResult {
        auth: ModelAuth {
            api_key: Some(resolved.api_key),
            ..ModelAuth::default()
        },
        env: Some(resolved.env),
        source: Some(resolved.source.to_string()),
    }))
}

/// The `resolve` body of [`cloudflare_ai_gateway_auth`].
async fn resolve_ai_gateway(
    input: &ApiKeyAuthInput<'_>,
) -> Result<Option<AuthResult>, ThrownValue> {
    let resolved = resolve_cloudflare_env(
        CloudflareAuthKind::AiGateway,
        &input.ctx,
        input.credential,
        &input.signal,
    )
    .await?;
    Ok(resolved.map(|resolved| AuthResult {
        auth: ModelAuth {
            headers: Some(
                [
                    (
                        "cf-aig-authorization".to_string(),
                        Some(format!("Bearer {}", resolved.api_key)),
                    ),
                    ("Authorization".to_string(), None),
                    ("x-api-key".to_string(), None),
                ]
                .into_iter()
                .collect(),
            ),
            ..ModelAuth::default()
        },
        env: Some(resolved.env),
        source: Some(resolved.source.to_string()),
    }))
}

fn resolve_workers_ai_entry<'a>(
    input: ApiKeyAuthInput<'a>,
) -> BoxFuture<'a, Result<Option<AuthResult>, ThrownValue>> {
    Box::pin(async move { resolve_workers_ai(&input).await })
}

fn resolve_ai_gateway_entry<'a>(
    input: ApiKeyAuthInput<'a>,
) -> BoxFuture<'a, Result<Option<AuthResult>, ThrownValue>> {
    Box::pin(async move { resolve_ai_gateway(&input).await })
}

fn login_workers_ai_entry(
    interaction: ProviderAuthInteraction,
) -> BoxFuture<'static, Result<ApiKeyCredential, ThrownValue>> {
    Box::pin(login_workers_ai(interaction))
}

fn login_ai_gateway_entry(
    interaction: ProviderAuthInteraction,
) -> BoxFuture<'static, Result<ApiKeyCredential, ThrownValue>> {
    Box::pin(login_ai_gateway(interaction))
}

/// Upstream `cloudflareWorkersAIAuth()`.
pub fn cloudflare_workers_ai_auth() -> ApiKeyAuth {
    ApiKeyAuth {
        name: "Cloudflare API key".to_string(),
        login: Some(Box::new(login_workers_ai_entry)),
        check: None,
        resolve: Box::new(resolve_workers_ai_entry) as ApiKeyResolveFn,
    }
}

/// Upstream `cloudflareAIGatewayAuth()`.
pub fn cloudflare_ai_gateway_auth() -> ApiKeyAuth {
    ApiKeyAuth {
        name: "Cloudflare API key".to_string(),
        login: Some(Box::new(login_ai_gateway_entry)),
        check: None,
        resolve: Box::new(resolve_ai_gateway_entry) as ApiKeyResolveFn,
    }
}

#[cfg(test)]
mod tests {
    // Port of the Cloudflare cases of test/providers.test.ts
    // (`getAuth` resolution through the models collection).
    use std::collections::BTreeMap;

    use super::*;
    use crate::auth::types::AuthContext;
    use crate::models::{create_models, CreateModelsOptions, Models, MutableModels};
    use crate::providers::{
        cloudflare_ai_gateway::cloudflare_ai_gateway_provider,
        cloudflare_workers_ai::cloudflare_workers_ai_provider,
    };

    struct MapContext {
        vars: BTreeMap<String, String>,
    }

    impl AuthContext for MapContext {
        fn env<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Option<String>> {
            Box::pin(async move { self.vars.get(name).cloned() })
        }

        fn file_exists<'a>(&'a self, path: &'a str) -> BoxFuture<'a, bool> {
            let _ = path;
            Box::pin(async move { false })
        }
    }

    fn auth_context(vars: &[(&str, &str)]) -> Arc<dyn AuthContext> {
        Arc::new(MapContext {
            vars: vars
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
        })
    }

    fn env_entries<'a>(env: &'a ProviderEnv, keys: &[&'a str]) -> Vec<(&'a str, Option<&'a str>)> {
        keys.iter()
            .map(|key| (*key, env.get(*key).map(String::as_str)))
            .collect()
    }

    #[tokio::test]
    async fn requires_workers_ai_account_config_and_returns_scoped_env() {
        let missing_account = create_models(Some(CreateModelsOptions {
            auth_context: Some(auth_context(&[("CLOUDFLARE_API_KEY", "cf-key")])),
            ..Default::default()
        }));
        missing_account.set_provider(cloudflare_workers_ai_provider());
        let model = missing_account
            .get_models(Some("cloudflare-workers-ai"))
            .into_iter()
            .next()
            .expect("workers-ai model");
        let result = missing_account
            .get_auth_by_provider(model.provider(), None)
            .await
            .expect("auth resolves");
        assert!(result.is_none());

        let configured = create_models(Some(CreateModelsOptions {
            auth_context: Some(auth_context(&[
                ("CLOUDFLARE_API_KEY", "cf-key"),
                ("CLOUDFLARE_ACCOUNT_ID", "account-id"),
            ])),
            ..Default::default()
        }));
        configured.set_provider(cloudflare_workers_ai_provider());
        let result = configured
            .get_auth_by_provider(model.provider(), None)
            .await
            .expect("auth resolves")
            .expect("configured");
        assert_eq!(
            result.auth,
            ModelAuth {
                api_key: Some("cf-key".to_string()),
                ..ModelAuth::default()
            }
        );
        assert_eq!(
            env_entries(
                result.env.as_ref().expect("env"),
                &["CLOUDFLARE_ACCOUNT_ID"]
            ),
            vec![("CLOUDFLARE_ACCOUNT_ID", Some("account-id"))]
        );
    }

    #[tokio::test]
    async fn requires_ai_gateway_account_and_gateway_config_and_returns_scoped_env_headers() {
        let missing_gateway = create_models(Some(CreateModelsOptions {
            auth_context: Some(auth_context(&[
                ("CLOUDFLARE_API_KEY", "cf-key"),
                ("CLOUDFLARE_ACCOUNT_ID", "account-id"),
            ])),
            ..Default::default()
        }));
        missing_gateway.set_provider(cloudflare_ai_gateway_provider());
        let model = missing_gateway
            .get_models(Some("cloudflare-ai-gateway"))
            .into_iter()
            .next()
            .expect("gateway model");
        let result = missing_gateway
            .get_auth_by_provider(model.provider(), None)
            .await
            .expect("auth resolves");
        assert!(result.is_none());

        let configured = create_models(Some(CreateModelsOptions {
            auth_context: Some(auth_context(&[
                ("CLOUDFLARE_API_KEY", "cf-key"),
                ("CLOUDFLARE_ACCOUNT_ID", "account-id"),
                ("CLOUDFLARE_GATEWAY_ID", "gateway-id"),
            ])),
            ..Default::default()
        }));
        configured.set_provider(cloudflare_ai_gateway_provider());
        let result = configured
            .get_auth_by_provider(model.provider(), None)
            .await
            .expect("auth resolves")
            .expect("configured");
        let headers = result.auth.headers.as_ref().expect("headers");
        assert_eq!(
            headers
                .get("cf-aig-authorization")
                .and_then(|value| value.as_deref()),
            Some("Bearer cf-key")
        );
        assert_eq!(headers.get("Authorization"), Some(&None));
        assert_eq!(headers.get("x-api-key"), Some(&None));
        assert_eq!(
            env_entries(
                result.env.as_ref().expect("env"),
                &["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_GATEWAY_ID"]
            ),
            vec![
                ("CLOUDFLARE_ACCOUNT_ID", Some("account-id")),
                ("CLOUDFLARE_GATEWAY_ID", Some("gateway-id")),
            ]
        );
    }
}
