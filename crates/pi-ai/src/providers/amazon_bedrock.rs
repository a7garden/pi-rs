//! Port of `packages/ai/src/providers/amazon-bedrock.ts`.
//!
//! Deviations (per-file ledger):
//! - D15: upstream factories return a non-failing `Provider`; Rust
//!   [`crate::models::create_provider`] returns `Result`, unwrapped with
//!   `expect` because the configuration here is static and valid.
//! - D16: upstream imports the API implementation from
//!   `bedrock-converse-stream.lazy.ts`; Rust links statically, so the factory
//!   calls [`crate::api::bedrock_converse_stream::bedrock_converse_stream_api`]
//!   directly (crate docs, deviation D10).

use std::sync::Arc;

use futures_util::future::BoxFuture;

use crate::auth::types::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthEvent, AuthInfoLink, AuthPrompt,
    AuthPromptKind, AuthResult, AuthSelectOption, ModelAuth, ProviderAuth, ProviderAuthInteraction,
};
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};
use crate::types::ProviderEnv;
use crate::utils::thrown::ThrownValue;

/// Upstream `signal.throwIfAborted()` (`Err(reason)` when aborted).
fn check_aborted(signal: &crate::utils::abort::AbortSignal) -> Result<(), ThrownValue> {
    signal.check().map_err(ThrownValue::error)
}

/// The `login` of [`bedrock_auth`].
fn login_bedrock(
    interaction: ProviderAuthInteraction,
) -> BoxFuture<'static, Result<ApiKeyCredential, ThrownValue>> {
    Box::pin(async move {
        check_aborted(&interaction.signal)?;
        let method = (interaction.prompt)(AuthPrompt {
            signal: Some(interaction.signal.clone()),
            kind: AuthPromptKind::Select {
                message: "Select Amazon Bedrock authentication method:".to_string(),
                options: vec![
                    AuthSelectOption {
                        id: "bearer-token".to_string(),
                        label: "Bearer token".to_string(),
                        description: None,
                    },
                    AuthSelectOption {
                        id: "aws-profile".to_string(),
                        label: "AWS profile".to_string(),
                        description: None,
                    },
                    AuthSelectOption {
                        id: "credential-chain".to_string(),
                        label: "Existing AWS credential chain".to_string(),
                        description: None,
                    },
                ],
            },
        })
        .await?;
        check_aborted(&interaction.signal)?;
        if method == "bearer-token" {
            let key = (interaction.prompt)(AuthPrompt {
                signal: Some(interaction.signal.clone()),
                kind: AuthPromptKind::Secret {
                    message: "Enter Amazon Bedrock bearer token".to_string(),
                    placeholder: None,
                },
            })
            .await?;
            return Ok(ApiKeyCredential {
                key: Some(key),
                env: None,
            });
        }
        (interaction.notify)(AuthEvent::Info {
            message:
                "Amazon Bedrock supports AWS profiles, IAM credentials, and role-based credentials."
                    .to_string(),
            links: vec![AuthInfoLink {
                label: Some("AWS credential provider chain".to_string()),
                url:
                    "https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html"
                        .to_string(),
            }],
        });
        if method == "aws-profile" {
            let profile = (interaction.prompt)(AuthPrompt {
                signal: Some(interaction.signal.clone()),
                kind: AuthPromptKind::Text {
                    message: "Enter AWS profile name".to_string(),
                    placeholder: None,
                },
            })
            .await?;
            let mut env = ProviderEnv::new();
            env.insert("AWS_PROFILE".to_string(), profile);
            return Ok(ApiKeyCredential {
                key: None,
                env: Some(env),
            });
        }
        if method != "credential-chain" {
            return Err(ThrownValue::message(format!(
                "Unknown Amazon Bedrock auth method: {method}"
            )));
        }
        (interaction.prompt)(AuthPrompt {
            signal: Some(interaction.signal.clone()),
            kind: AuthPromptKind::Text {
                message: "Configure AWS credentials, then press Enter to continue".to_string(),
                placeholder: None,
            },
        })
        .await?;
        Ok(ApiKeyCredential {
            key: None,
            env: None,
        })
    })
}

/// The `resolve` entry of [`bedrock_auth`].
fn resolve_bedrock_entry<'a>(
    input: ApiKeyAuthInput<'a>,
) -> BoxFuture<'a, Result<Option<AuthResult>, ThrownValue>> {
    Box::pin(async move { resolve_bedrock(&input).await })
}

/// The `resolve` body of [`bedrock_auth`], kept as a free function so the
/// lifetime-generic closure type stays nameable.
///
/// Bedrock accepts a bearer token or the AWS SDK's default credential chain.
/// The login flow can store a token/profile choice; resolve also detects
/// ambient AWS credentials without copying them into pi's credential store.
async fn resolve_bedrock(input: &ApiKeyAuthInput<'_>) -> Result<Option<AuthResult>, ThrownValue> {
    let ApiKeyAuthInput {
        ctx,
        credential,
        signal,
    } = input;
    let env = async |name: &str| -> Result<Option<String>, ThrownValue> {
        check_aborted(signal)?;
        let value = ctx.env(name).await;
        check_aborted(signal)?;
        Ok(value)
    };
    if let Some(key) = credential
        .as_ref()
        .and_then(|credential| credential.key.as_deref())
    {
        return Ok(Some(AuthResult {
            auth: ModelAuth {
                api_key: Some(key.to_string()),
                headers: None,
                base_url: None,
            },
            env: credential
                .as_ref()
                .and_then(|credential| credential.env.clone()),
            source: Some("stored credential".to_string()),
        }));
    }
    if env("AWS_BEARER_TOKEN_BEDROCK")
        .await?
        .is_some_and(|value| !value.is_empty())
    {
        return Ok(Some(AuthResult {
            auth: ModelAuth::default(),
            env: None,
            source: Some("AWS_BEARER_TOKEN_BEDROCK".to_string()),
        }));
    }
    let scoped_profile = credential
        .as_ref()
        .and_then(|credential| credential.env.as_ref())
        .and_then(|env| env.get("AWS_PROFILE"));
    if let Some(profile) = scoped_profile {
        let _ = env("AWS_PROFILE").await?;
        return Ok(Some(AuthResult {
            auth: ModelAuth::default(),
            env: credential
                .as_ref()
                .and_then(|credential| credential.env.clone()),
            source: Some(
                if !profile.is_empty() {
                    "stored credential"
                } else {
                    "AWS_PROFILE"
                }
                .to_string(),
            ),
        }));
    }
    if let Some(profile) = env("AWS_PROFILE").await? {
        if !profile.is_empty() {
            return Ok(Some(AuthResult {
                auth: ModelAuth::default(),
                env: credential
                    .as_ref()
                    .and_then(|credential| credential.env.clone()),
                source: Some("AWS_PROFILE".to_string()),
            }));
        }
    }
    let access_key_id = env("AWS_ACCESS_KEY_ID").await?;
    let secret_access_key = env("AWS_SECRET_ACCESS_KEY").await?;
    if access_key_id.is_some_and(|value| !value.is_empty())
        && secret_access_key.is_some_and(|value| !value.is_empty())
    {
        return Ok(Some(AuthResult {
            auth: ModelAuth::default(),
            env: None,
            source: Some("AWS access keys".to_string()),
        }));
    }
    if env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
        .await?
        .is_some_and(|value| !value.is_empty())
    {
        return Ok(Some(AuthResult {
            auth: ModelAuth::default(),
            env: None,
            source: Some("ECS task role".to_string()),
        }));
    }
    if env("AWS_CONTAINER_CREDENTIALS_FULL_URI")
        .await?
        .is_some_and(|value| !value.is_empty())
    {
        return Ok(Some(AuthResult {
            auth: ModelAuth::default(),
            env: None,
            source: Some("ECS task role".to_string()),
        }));
    }
    if env("AWS_WEB_IDENTITY_TOKEN_FILE")
        .await?
        .is_some_and(|value| !value.is_empty())
    {
        return Ok(Some(AuthResult {
            auth: ModelAuth::default(),
            env: None,
            source: Some("web identity token".to_string()),
        }));
    }
    Ok(None)
}

/// Upstream `bedrockAuth`.
fn bedrock_auth() -> ApiKeyAuth {
    ApiKeyAuth {
        name: "AWS credentials or bearer token".to_string(),
        login: Some(Box::new(login_bedrock)),
        check: None,
        resolve: Box::new(resolve_bedrock_entry),
    }
}

/// Upstream `amazonBedrockProvider()`.
pub fn amazon_bedrock_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: "amazon-bedrock".to_string(),
        name: Some("Amazon Bedrock".to_string()),
        base_url: None,
        headers: None,
        auth: ProviderAuth {
            api_key: Some(bedrock_auth()),
            oauth: None,
        },
        models: super::amazon_bedrock_models::amazon_bedrock_any_models(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
        api: Some(ApiDispatch::Single(
            crate::api::bedrock_converse_stream::bedrock_converse_stream_api(),
        )),
        images: Default::default(),
        classifiers: Default::default(),
    })
    .expect("static amazon-bedrock provider configuration is valid")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::types::{AuthContext, BoxFuture};
    use crate::models::{Models, MutableModels};
    use std::collections::BTreeMap;
    use parking_lot::Mutex;

    /// `fakeAuthContext` from `test/helpers.ts`: env lookups from a map.
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
            vars: vars
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
        })
    }

    fn never_aborted_signal() -> crate::utils::abort::AbortSignal {
        crate::utils::abort::AbortController::new().signal()
    }

    /// Port of `test/providers.test.ts`
    /// "runs provider-owned Bedrock bearer token and AWS profile login flows".
    #[tokio::test]
    async fn runs_provider_owned_bearer_token_and_aws_profile_login_flows() {
        let provider = amazon_bedrock_provider();
        let auth = provider.auth().api_key.as_ref().expect("api key auth");

        // Bearer token flow.
        let answers = Arc::new(Mutex::new(vec![
            "bearer-token".to_string(),
            "bedrock-token".to_string(),
        ]));
        let signal = never_aborted_signal();
        let prompt_answers = answers.clone();
        let credential = (auth.login.as_ref().expect("login present"))(ProviderAuthInteraction {
            signal: signal.clone(),
            prompt: Box::new(move |_prompt| {
                let answers = prompt_answers.clone();
                Box::pin(async move { Ok(answers.lock().remove(0)) })
            }),
            notify: Box::new(|_event| {}),
        })
        .await
        .expect("login succeeds");
        assert_eq!(credential.key.as_deref(), Some("bedrock-token"));
        assert!(credential.env.is_none());

        // AWS profile flow.
        let answers = Arc::new(Mutex::new(vec![
            "aws-profile".to_string(),
            "work".to_string(),
        ]));
        let events = Arc::new(Mutex::new(Vec::<AuthEvent>::new()));
        let event_sink = events.clone();
        let credential = (auth.login.as_ref().expect("login present"))(ProviderAuthInteraction {
            signal: signal.clone(),
            prompt: Box::new(move |_prompt| {
                let answers = answers.clone();
                Box::pin(async move { Ok(answers.lock().remove(0)) })
            }),
            notify: Box::new(move |event| event_sink.lock().push(event)),
        })
        .await
        .expect("login succeeds");
        let env = credential
            .env
            .clone()
            .expect("profile credential carries env");
        assert_eq!(env.get("AWS_PROFILE").map(String::as_str), Some("work"));
        assert!(credential.key.is_none());
        let events = events.lock();
        assert_eq!(events.len(), 1);
        match &events[0] {
            AuthEvent::Info { links, .. } => {
                assert_eq!(links.len(), 1);
                assert_eq!(
                    links[0].label.as_deref(),
                    Some("AWS credential provider chain")
                );
            }
            other => panic!("expected info event, got {other:?}"),
        }

        // Resolving the stored profile credential keeps the scoped env.
        let resolution = (auth.resolve)(ApiKeyAuthInput {
            ctx: ctx(&[]),
            credential: Some(&credential),
            signal: signal.clone(),
        })
        .await
        .expect("resolve succeeds")
        .expect("credential resolves");
        assert_eq!(resolution.auth, ModelAuth::default());
        assert_eq!(
            resolution
                .env
                .as_ref()
                .and_then(|env| env.get("AWS_PROFILE").cloned()),
            Some("work".to_string())
        );
        assert_eq!(resolution.source.as_deref(), Some("stored credential"));
    }

    /// Port of `test/providers.test.ts`
    /// "reports bedrock as configured from ambient AWS credentials without an api key".
    #[tokio::test]
    async fn reports_bedrock_as_configured_from_ambient_aws_credentials() {
        let models = crate::models::create_models(Some(crate::models::CreateModelsOptions {
            auth_context: Some(ctx(&[("AWS_PROFILE", "dev")])),
            ..Default::default()
        }));
        models.set_provider(amazon_bedrock_provider());
        let provider_models =
            crate::models::Models::get_models(models.as_ref(), Some("amazon-bedrock"));
        assert!(!provider_models.is_empty());

        let result = models
            .get_auth_by_provider("amazon-bedrock", None)
            .await
            .expect("get_auth succeeds")
            .expect("ambient AWS_PROFILE resolves");
        assert_eq!(result.auth, ModelAuth::default());
        assert_eq!(result.source.as_deref(), Some("AWS_PROFILE"));

        let unconfigured = crate::models::create_models(Some(crate::models::CreateModelsOptions {
            auth_context: Some(ctx(&[])),
            ..Default::default()
        }));
        unconfigured.set_provider(amazon_bedrock_provider());
        let result = unconfigured
            .get_auth_by_provider("amazon-bedrock", None)
            .await
            .expect("get_auth succeeds");
        assert!(result.is_none());
    }

    /// The resolve chain's ambient detections (upstream `resolve` branches not
    /// covered by the providers.test.ts cases above, exercised through the
    /// same entry point).
    #[tokio::test]
    async fn resolves_ambient_bedrock_credential_sources() {
        let provider = amazon_bedrock_provider();
        let auth = provider.auth().api_key.as_ref().expect("api key auth");
        let signal = never_aborted_signal();

        let stored = ApiKeyCredential {
            key: Some("bearer".to_string()),
            env: None,
        };
        let result = (auth.resolve)(ApiKeyAuthInput {
            ctx: ctx(&[]),
            credential: Some(&stored),
            signal: signal.clone(),
        })
        .await
        .expect("resolve")
        .expect("resolved");
        assert_eq!(result.auth.api_key.as_deref(), Some("bearer"));
        assert_eq!(result.source.as_deref(), Some("stored credential"));

        // Ambient bearer token.
        let result = (auth.resolve)(ApiKeyAuthInput {
            ctx: ctx(&[("AWS_BEARER_TOKEN_BEDROCK", "env-bearer")]),
            credential: None,
            signal: signal.clone(),
        })
        .await
        .expect("resolve")
        .expect("resolved");
        assert_eq!(result.source.as_deref(), Some("AWS_BEARER_TOKEN_BEDROCK"));

        // Ambient IAM keys.
        let result = (auth.resolve)(ApiKeyAuthInput {
            ctx: ctx(&[
                ("AWS_ACCESS_KEY_ID", "AKIA"),
                ("AWS_SECRET_ACCESS_KEY", "secret"),
            ]),
            credential: None,
            signal: signal.clone(),
        })
        .await
        .expect("resolve")
        .expect("resolved");
        assert_eq!(result.source.as_deref(), Some("AWS access keys"));

        // ECS / web-identity ambient sources.
        for (var, expected) in [
            ("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "ECS task role"),
            ("AWS_CONTAINER_CREDENTIALS_FULL_URI", "ECS task role"),
            ("AWS_WEB_IDENTITY_TOKEN_FILE", "web identity token"),
        ] {
            let result = (auth.resolve)(ApiKeyAuthInput {
                ctx: ctx(&[(var, "1")]),
                credential: None,
                signal: signal.clone(),
            })
            .await
            .expect("resolve")
            .expect("resolved");
            assert_eq!(result.source.as_deref(), Some(expected));
        }

        // Unknown login method rejects with the upstream message.
        let error = (auth.login.as_ref().expect("login present"))(ProviderAuthInteraction {
            signal: signal.clone(),
            prompt: Box::new(|_prompt| Box::pin(async { Ok("nonsense".to_string()) })),
            notify: Box::new(|_event| {}),
        })
        .await
        .expect_err("unknown method rejects");
        let crate::utils::thrown::ThrownValue::Text(message) = error else {
            panic!("expected a text throw, got {error:?}");
        };
        assert_eq!(message, "Unknown Amazon Bedrock auth method: nonsense");
    }
}
