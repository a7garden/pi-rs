//! `cli/credential-print.ts` — resolve one configured provider credential.

use pi_ai::models::Models as _;

use super::args::Args;
use super::auth_command::{
    get_auth_credential, validate_auth_command_args, AuthCommandError, AuthCommandKind,
};
use crate::core::model_runtime::ModelRuntime;

pub const DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS: u64 = 30 * 60_000;

type CredentialPrintKind = AuthCommandKind;

/// Provider selection entry (upstream `{ id, model? }`).
struct ProviderSelection {
    id: String,
    model: Option<pi_ai::types::Model>,
}

/// Resolve one configured provider credential
/// (`resolveCredentialForPrint`).
///
/// This intentionally calls `ModelRuntime.getAuth()`, which refreshes and
/// persists OAuth credentials with less than five minutes remaining through
/// the normal request-auth path.
pub async fn resolve_credential_for_print(
    args: &Args,
    model_runtime: &ModelRuntime,
    kind: CredentialPrintKind,
    min_expiry_ms: Option<u64>,
    signal: Option<pi_ai::utils::abort::AbortSignal>,
) -> Result<String, AuthCommandError> {
    let (cli_provider, cli_model) = validate_auth_command_args(args, kind)?;

    let credential_types = model_runtime
        .list_credentials(None)
        .await
        .map_err(|_| AuthCommandError("Failed to list credentials".to_string()))?;
    let credential_types: std::collections::HashMap<String, String> = credential_types
        .into_iter()
        .map(|credential| (credential.provider_id, credential.credential_type))
        .collect();

    let mut providers: Vec<ProviderSelection> = Vec::new();
    if let Some(cli_provider) = &cli_provider {
        let provider = model_runtime.get_provider(cli_provider);
        let Some(provider) = provider else {
            return Err(AuthCommandError(format!(
				"Unknown provider \"{cli_provider}\". Use --list-models to see available providers."
			)));
        };
        if let Some(cli_model) = &cli_model {
            let resolved = crate::core::model_resolver::resolve_cli_model(
                crate::core::model_resolver::ResolveCliModelOptions {
                    cli_provider: Some(provider.id()),
                    cli_model: Some(cli_model),
                    cli_thinking: None,
                    model_runtime,
                },
            );
            match resolved.model {
                Some(model) if resolved.error.is_none() => {
                    providers.push(ProviderSelection {
                        id: provider.id().to_string(),
                        model: Some(model),
                    });
                }
                _ => {
                    return Err(AuthCommandError(resolved.error.unwrap_or_else(|| {
                        "Unable to resolve the requested provider/model".to_string()
                    })));
                }
            }
        } else {
            providers.push(ProviderSelection {
                id: provider.id().to_string(),
                model: None,
            });
        }
    } else {
        let Some(cli_model) = &cli_model else {
            return Err(AuthCommandError(
                "Credential printing requires --provider <provider> or --model <model>".to_string(),
            ));
        };
        let mut matched: Vec<ProviderSelection> = Vec::new();
        for provider in model_runtime.get_providers() {
            if !credential_types.contains_key(provider.id()) {
                continue;
            }
            let resolved = crate::core::model_resolver::resolve_cli_model(
                crate::core::model_resolver::ResolveCliModelOptions {
                    cli_provider: Some(provider.id()),
                    cli_model: Some(cli_model),
                    cli_thinking: None,
                    model_runtime,
                },
            );
            let matches = resolved.model.is_some()
                && resolved.error.is_none()
                && !resolved
                    .warning
                    .as_deref()
                    .is_some_and(|w| w.contains("Using custom model id"));
            if matches {
                matched.push(ProviderSelection {
                    id: provider.id().to_string(),
                    model: resolved.model,
                });
            }
        }
        providers = matched;
        if providers.is_empty() {
            return Err(AuthCommandError(format!(
                "Model \"{cli_model}\" not found. Use --list-models to see available models."
            )));
        }
    }

    let mut credentials: Vec<(String, String)> = Vec::new();
    for provider in &providers {
        let credential_type = credential_types.get(&provider.id).map(|s| s.as_str());
        if kind == AuthCommandKind::ApiKey && credential_type == Some("oauth") {
            continue;
        }
        if kind == AuthCommandKind::BearerToken && credential_type != Some("oauth") {
            continue;
        }
        let min_oauth_validity_ms = if kind == AuthCommandKind::BearerToken {
            Some((min_expiry_ms.unwrap_or(DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS)) as i64)
        } else {
            None
        };
        let overrides = crate::core::model_runtime::ModelRuntimeAuthOverrides {
            api_key: None,
            env: None,
            min_oauth_validity_ms,
            signal: signal.clone(),
        };
        let auth = match &provider.model {
            Some(model) => {
                model_runtime
                    .get_auth_for_model(
                        &pi_ai::types::AnyModel::Chat(model.clone()),
                        Some(overrides),
                    )
                    .await
            }
            None => {
                model_runtime
                    .get_auth_for_provider(&provider.id, Some(overrides))
                    .await
            }
        };
        if let Ok(Some(auth)) = auth {
            if let Some(value) = get_auth_credential(Some(&auth)) {
                credentials.push((provider.id.clone(), value));
            }
        }
    }

    if credentials.len() == 1 {
        return Ok(credentials.remove(0).1);
    }
    if credentials.is_empty() {
        let provider_id = providers.first().map(|p| p.id.clone());
        let credential_type = provider_id
            .as_ref()
            .and_then(|id| credential_types.get(id).map(|s| s.as_str()));
        if let (Some(provider_id), true) = (&cli_provider, kind == AuthCommandKind::ApiKey) {
            if credential_type == Some("oauth") {
                return Err(AuthCommandError(format!(
                    "Provider \"{provider_id}\" is configured with OAuth, not an API key"
                )));
            }
        }
        if let (Some(provider_id), true) = (&cli_provider, kind == AuthCommandKind::BearerToken) {
            if credential_type != Some("oauth") {
                return Err(AuthCommandError(format!(
                    "Provider \"{provider_id}\" is not configured with an OAuth bearer token"
                )));
            }
        }
        return Err(AuthCommandError(format!(
            "No usable {} is configured",
            if kind == AuthCommandKind::ApiKey {
                "API key"
            } else {
                "OAuth bearer token"
            }
        )));
    }
    let ids = credentials
        .iter()
        .map(|(id, _)| id.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    Err(AuthCommandError(format!(
        "Multiple configured providers matched ({ids}). Specify --provider."
    )))
}
