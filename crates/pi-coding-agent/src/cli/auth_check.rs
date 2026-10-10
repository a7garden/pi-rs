//! `cli/auth-check.ts` — provider readiness checks for `pi auth check`.

use std::sync::Arc;

use pi_agent::types::AgentThinkingLevel;
use pi_ai::auth::types::AuthType;
use pi_ai::auth::types::CredentialStore as _;
use pi_ai::models::Models as _;

use super::args::Args;
use super::auth_command::{validate_auth_command_args, AuthCommandError, AuthCommandKind};
use crate::core::auth_storage::{AuthStorage, ReadOnlyAuthStorage};
use crate::core::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::core::models_store_wrapper::InMemoryCodingAgentModelsStore;

/// `AuthCheckStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthCheckStatus {
    Ready,
    NotReady,
    Invalid,
}

/// `AuthCheckReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthCheckReason {
    ProviderNotFound,
    CredentialsNotConfigured,
    CredentialNotAvailable,
    InvalidState,
}

/// `AuthCheckResult`.
#[derive(Debug, Clone)]
pub struct AuthCheckResult {
    pub status: AuthCheckStatus,
    pub provider: String,
    pub reason: Option<AuthCheckReason>,
    pub auth_type: Option<AuthType>,
}

impl AuthCheckResult {
    pub fn new(
        status: AuthCheckStatus,
        provider: impl Into<String>,
        reason: Option<AuthCheckReason>,
    ) -> Self {
        AuthCheckResult {
            status,
            provider: provider.into(),
            reason,
            auth_type: None,
        }
    }
}

/// `checkProviderAuth`.
pub async fn check_provider_auth(
    args: &Args,
    model_runtime: &ModelRuntime,
    refresh: bool,
) -> Result<AuthCheckResult, AuthCommandError> {
    let (cli_provider, cli_model) = validate_auth_command_args(args, AuthCommandKind::Check)?;
    let mut provider = cli_provider.clone();
    if let Some(cli_model) = &cli_model {
        let resolved = crate::core::model_resolver::resolve_cli_model(
            crate::core::model_resolver::ResolveCliModelOptions {
                cli_provider: cli_provider.as_deref(),
                cli_model: Some(cli_model),
                cli_thinking: None::<AgentThinkingLevel>,
                model_runtime,
            },
        );
        if resolved.error.is_some() || resolved.model.is_none() {
            return Err(AuthCommandError(resolved.error.unwrap_or_else(|| {
                format!("Unable to resolve model \"{cli_model}\"")
            })));
        }
        provider = Some(resolved.model.unwrap().provider().to_string());
    }
    let Some(provider) = provider else {
        return Err(AuthCommandError(
            "Unable to resolve an auth provider".to_string(),
        ));
    };
    if model_runtime.get_error().is_some() {
        return Ok(AuthCheckResult::new(
            AuthCheckStatus::Invalid,
            provider,
            Some(AuthCheckReason::InvalidState),
        ));
    }
    if model_runtime.get_provider(&provider).is_none() {
        return Ok(AuthCheckResult::new(
            AuthCheckStatus::NotReady,
            provider,
            Some(AuthCheckReason::ProviderNotFound),
        ));
    }
    match model_runtime.check_auth(&provider, None).await {
        Ok(None) => Ok(AuthCheckResult::new(
            AuthCheckStatus::NotReady,
            provider,
            Some(AuthCheckReason::CredentialsNotConfigured),
        )),
        Ok(Some(auth)) => {
            if refresh {
                let has_auth = model_runtime.get_auth_for_provider(&provider, None).await;
                match has_auth {
                    Ok(None) => {
                        return Ok(AuthCheckResult::new(
                            AuthCheckStatus::NotReady,
                            provider,
                            Some(AuthCheckReason::CredentialsNotConfigured),
                        ));
                    }
                    Err(_) => {
                        return Ok(AuthCheckResult::new(
                            AuthCheckStatus::Invalid,
                            provider,
                            Some(AuthCheckReason::InvalidState),
                        ));
                    }
                    Ok(Some(_)) => {}
                }
            }
            Ok(AuthCheckResult {
                status: AuthCheckStatus::Ready,
                provider,
                reason: None,
                auth_type: Some(auth.check_type),
            })
        }
        Err(_) => Ok(AuthCheckResult::new(
            AuthCheckStatus::Invalid,
            provider,
            Some(AuthCheckReason::InvalidState),
        )),
    }
}

/// `getProviderCredential`.
pub async fn get_provider_credential(
    provider_id: &str,
    model_runtime: &ModelRuntime,
    credentials: &AuthStorage,
    refresh: bool,
) -> Option<String> {
    let credential = credentials.read(provider_id, None).await.ok().flatten();
    if !refresh {
        if let Some(pi_ai::auth::types::Credential::OAuth(oauth)) = credential {
            return Some(oauth.credentials.access);
        }
    }
    let auth = model_runtime
        .get_auth_for_provider(provider_id, None)
        .await
        .ok()
        .flatten();
    super::auth_command::get_auth_credential(auth.as_ref())
}

/// `createAuthCheckModelRuntime`.
pub async fn create_auth_check_model_runtime(
    credentials: Arc<dyn pi_ai::auth::types::CredentialStore>,
) -> Result<Arc<ModelRuntime>, pi_ai::utils::thrown::ThrownValue> {
    ModelRuntime::create(CreateModelRuntimeOptions {
        credentials: Some(credentials),
        models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::new())),
        allow_model_network: false,
        refresh_on_create: false,
        ..Default::default()
    })
    .await
}

/// `ReadOnlyAuthStorage` bound to a path (upstream constructor).
pub fn read_only_auth_storage(auth_path: &str) -> ReadOnlyAuthStorage {
    ReadOnlyAuthStorage::new(auth_path)
}

/// Convenience: in-memory `AuthStorage` (upstream `AuthStorage.inMemory`).
pub fn in_memory_auth_storage(data: crate::core::auth_storage::AuthStorageData) -> AuthStorage {
    AuthStorage::in_memory(data)
}

#[cfg(test)]
mod tests {
    //! Runtime-backed cases from `test/auth-check.test.ts`. The OAuth
    //! refresh-injection cases are deferred: swapping a provider's refresh
    //! closure is a `pi-ai` runtime capability the port does not expose.
    use super::*;
    use crate::cli::args::parse_args;
    use pi_ai::auth::types::{ApiKeyCredential, Credential, OAuthCredential, OAuthCredentials};

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn storage_with(entries: &[(&str, Credential)]) -> AuthStorage {
        let mut data = crate::core::auth_storage::AuthStorageData::new();
        for (provider, credential) in entries {
            data.insert(provider.to_string(), credential.clone());
        }
        AuthStorage::in_memory(data)
    }

    fn api_key(key: &str) -> Credential {
        Credential::ApiKey(ApiKeyCredential {
            key: Some(key.to_string()),
            env: None,
        })
    }

    async fn runtime_with(credentials: Arc<AuthStorage>) -> Arc<ModelRuntime> {
        create_auth_check_model_runtime(credentials).await.unwrap()
    }

    #[tokio::test]
    async fn reports_a_configured_provider_as_ready() {
        let runtime =
            runtime_with(Arc::new(storage_with(&[("openai", api_key("test-key"))]))).await;
        let parsed = parse_args(&args(&["--provider", "openai"]));
        let result = check_provider_auth(&parsed, &runtime, false).await.unwrap();
        assert_eq!(result.status, AuthCheckStatus::Ready);
        assert_eq!(result.provider, "openai");
        assert_eq!(result.auth_type, Some(AuthType::ApiKey));
    }

    #[tokio::test]
    async fn reports_an_unknown_provider_as_not_ready() {
        let runtime = runtime_with(Arc::new(storage_with(&[]))).await;
        let parsed = parse_args(&args(&["--provider", "not-installed"]));
        let result = check_provider_auth(&parsed, &runtime, false).await.unwrap();
        assert_eq!(result.status, AuthCheckStatus::NotReady);
        assert_eq!(result.provider, "not-installed");
        assert_eq!(result.reason, Some(AuthCheckReason::ProviderNotFound));
    }

    #[tokio::test]
    async fn reports_unconfigured_provider_as_not_ready() {
        let runtime = runtime_with(Arc::new(storage_with(&[]))).await;
        let parsed = parse_args(&args(&["--provider", "openai"]));
        let result = check_provider_auth(&parsed, &runtime, false).await.unwrap();
        assert_eq!(result.status, AuthCheckStatus::NotReady);
        assert_eq!(
            result.reason,
            Some(AuthCheckReason::CredentialsNotConfigured)
        );
    }

    #[tokio::test]
    async fn does_not_treat_an_unresolved_stored_environment_reference_as_configured() {
        let temp = std::env::temp_dir().join(format!("pi-auth-check-{}.json", std::process::id()));
        std::fs::write(
            &temp,
            r#"{"openai": {"type": "api_key", "key": "$MISSING_AUTH_CHECK_KEY"}}"#,
        )
        .unwrap();
        let runtime = create_auth_check_model_runtime(Arc::new(ReadOnlyAuthStorage::new(
            &temp.to_string_lossy(),
        )))
        .await
        .unwrap();
        let parsed = parse_args(&args(&["--provider", "openai"]));
        let result = check_provider_auth(&parsed, &runtime, false).await.unwrap();
        assert_eq!(result.status, AuthCheckStatus::NotReady);
        assert_eq!(
            result.reason,
            Some(AuthCheckReason::CredentialsNotConfigured)
        );
        let _ = std::fs::remove_file(&temp);
    }

    #[tokio::test]
    async fn reports_malformed_auth_state_as_invalid() {
        let temp =
            std::env::temp_dir().join(format!("pi-auth-check-bad-{}.json", std::process::id()));
        std::fs::write(&temp, "{invalid-json").unwrap();
        let runtime = create_auth_check_model_runtime(Arc::new(ReadOnlyAuthStorage::new(
            &temp.to_string_lossy(),
        )))
        .await
        .unwrap();
        let parsed = parse_args(&args(&["--provider", "openai"]));
        let result = check_provider_auth(&parsed, &runtime, false).await.unwrap();
        assert_eq!(result.status, AuthCheckStatus::Invalid);
        assert_eq!(result.reason, Some(AuthCheckReason::InvalidState));
        let _ = std::fs::remove_file(&temp);
    }

    #[tokio::test]
    async fn does_not_create_an_auth_file_or_its_parent_directory() {
        let temp = std::env::temp_dir().join(format!("pi-auth-check-none-{}", std::process::id()));
        let auth_path = temp.join("agent").join("auth.json");
        let runtime = create_auth_check_model_runtime(Arc::new(ReadOnlyAuthStorage::new(
            &auth_path.to_string_lossy(),
        )))
        .await
        .unwrap();
        let parsed = parse_args(&args(&["--provider", "openai"]));
        let result = check_provider_auth(&parsed, &runtime, false).await.unwrap();
        assert_eq!(result.status, AuthCheckStatus::NotReady);
        assert_eq!(
            result.reason,
            Some(AuthCheckReason::CredentialsNotConfigured)
        );
        assert!(!auth_path.exists());
        assert!(!auth_path.parent().unwrap().exists());
    }

    #[tokio::test]
    async fn creates_an_auth_check_runtime_without_catalog_storage() {
        let runtime = runtime_with(Arc::new(storage_with(&[]))).await;
        assert!(ModelRuntime::get_provider(runtime.as_ref(), "openai").is_some());
    }

    #[tokio::test]
    async fn get_provider_credential_reads_without_refreshing() {
        let credentials = Arc::new(storage_with(&[("openai", api_key("test-key"))]));
        let runtime = runtime_with(credentials.clone()).await;
        let value = get_provider_credential("openai", &runtime, &credentials, false).await;
        assert_eq!(value.as_deref(), Some("test-key"));
    }

    #[tokio::test]
    async fn get_provider_credential_returns_stored_oauth_access_without_refresh() {
        let oauth = Credential::OAuth(OAuthCredential {
            credentials: OAuthCredentials {
                refresh: "refresh-token".to_string(),
                access: "old-token".to_string(),
                expires: 0,
                extra: Default::default(),
            },
        });
        let credentials = Arc::new(storage_with(&[("openai-codex", oauth)]));
        let runtime = runtime_with(credentials.clone()).await;
        let value = get_provider_credential("openai-codex", &runtime, &credentials, false).await;
        assert_eq!(value.as_deref(), Some("old-token"));
    }
}
