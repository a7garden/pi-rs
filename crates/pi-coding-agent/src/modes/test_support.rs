//! Shared test doubles for the mode suites (upstream `test/utilities.ts` +
//! `test/model-runtime-test-utils.ts`). `#[cfg(test)]`-only.

use std::sync::Arc;

use parking_lot::Mutex;

use pi_agent::types::StreamFn;
use pi_ai::auth::credential_store::InMemoryCredentialStore;
use pi_ai::auth::types::{ApiKeyCredential, Credential, CredentialStore};
use std::future::Future;

use pi_ai::models::{Models, RefreshModelsContext};
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream, BaseModelFields, Model,
    ModelCost, ModelCostRates, ModelInput, StreamDoneReason, StreamErrorReason, TranscriptContext,
    Usage, UsageCost,
};
use pi_ai::utils::thrown::ThrownValue;

use crate::core::extensions::types::{ExtensionFactory, InlineExtension};
use crate::core::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::core::resource_loader::{
    DefaultResourceLoader, DefaultResourceLoaderOptions, ResourceLoaderReloadOptions,
};

pub const PROVIDER_ID: &str = "test-provider";
pub const MODEL_ID: &str = "test-model";

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

pub fn zero_cost() -> UsageCost {
    UsageCost {
        input: 0.0,
        output: 0.0,
        cache_read: 0.0,
        cache_write: 0.0,
        total: 0.0,
    }
}

pub fn usage(total_tokens: u64) -> Usage {
    Usage {
        input: total_tokens,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens,
        cost: zero_cost(),
    }
}

/// `createAssistantMessage`: a completed text assistant message on the shared
/// test model.
pub fn assistant_message(text: impl Into<String>) -> AssistantMessage {
    AssistantMessage {
        content: vec![pi_ai::types::AssistantContent::Text(
            pi_ai::types::TextContent::new(text.into()),
        )],
        api: "openai-completions".to_string(),
        provider: PROVIDER_ID.to_string(),
        model: MODEL_ID.to_string(),
        response_model: None,
        response_id: None,
        thinking_level: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: usage(0),
        stop_reason: pi_ai::types::StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        duration_ms: None,
        timestamp: now_ms(),
    }
}

/// Assistant message carrying an error stop reason.
pub fn error_assistant_message(error_message: &str) -> AssistantMessage {
    AssistantMessage {
        stop_reason: pi_ai::types::StopReason::Error,
        error_message: Some(error_message.to_string()),
        ..assistant_message("")
    }
}

pub fn text_model(context_window: u64) -> Model {
    Model {
        base: BaseModelFields {
            id: MODEL_ID.to_string(),
            name: "Test Model".to_string(),
            api: "openai-completions".to_string(),
            provider: PROVIDER_ID.to_string(),
            base_url: "https://example.test/v1".to_string(),
            input: vec![ModelInput::Text],
            input_limits: None,
            cost: ModelCost {
                rates: ModelCostRates::default(),
                tiers: None,
            },
            headers: None,
        },
        model_type: None,
        reasoning: false,
        thinking_level_map: None,
        prompt_cache: None,
        context_window,
        max_tokens: 4_096,
        sampling_params: None,
        sampling_params_by_thinking_level: None,
        compat: None,
    }
}

/// Shared provider auth double (`resolve` mirrors the stored API key).
pub fn test_provider_auth() -> &'static pi_ai::auth::types::ProviderAuth {
    static AUTH: std::sync::LazyLock<pi_ai::auth::types::ProviderAuth> =
        std::sync::LazyLock::new(|| pi_ai::auth::types::ProviderAuth {
            api_key: Some(pi_ai::auth::types::ApiKeyAuth {
                name: "Test API key".to_string(),
                login: None,
                check: None,
                resolve: Box::new(|input| {
                    let credential = input.credential.cloned();
                    Box::pin(async move {
                        Ok(credential.map(|credential| pi_ai::auth::types::AuthResult {
                            auth: pi_ai::auth::types::ModelAuth {
                                api_key: credential.key.clone(),
                                headers: None,
                                base_url: None,
                            },
                            env: None,
                            source: Some("test".to_string()),
                        }))
                    })
                }),
            }),
            oauth: None,
        });
    &AUTH
}

fn unused_stream(model: &Model) -> AssistantMessageEventStream {
    pi_ai::api::lazy::lazy_stream(model, async { Err(ThrownValue::message("unused")) })
}

/// Test provider serving the given model.
pub struct TestProvider {
    pub model: Model,
}

impl pi_ai::models::Provider for TestProvider {
    fn id(&self) -> &str {
        PROVIDER_ID
    }

    fn name(&self) -> &str {
        PROVIDER_ID
    }

    fn auth(&self) -> &pi_ai::auth::types::ProviderAuth {
        test_provider_auth()
    }

    fn get_models(&self) -> Vec<Model> {
        vec![self.model.clone()]
    }

    fn refresh_models(
        &self,
        _context: RefreshModelsContext,
    ) -> Option<std::pin::Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
        None
    }

    fn stream(
        &self,
        model: &Model,
        _context: &TranscriptContext,
        _options: &pi_ai::types::StreamOptions,
    ) -> AssistantMessageEventStream {
        unused_stream(model)
    }

    fn stream_simple(
        &self,
        model: &Model,
        _context: &TranscriptContext,
        _options: &pi_ai::types::SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        unused_stream(model)
    }
}

/// In-memory model runtime with the test provider registered; `with_auth`
/// seeds a stored API key (`createInMemoryModelRegistry` + `authStorage`
/// setup upstream).
pub async fn test_model_runtime(model: Model, with_auth: bool) -> Arc<ModelRuntime> {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
    if with_auth {
        credentials
            .modify(
                PROVIDER_ID,
                Box::new(|_| {
                    Box::pin(async move {
                        Ok(Some(Credential::ApiKey(ApiKeyCredential {
                            key: Some("test-key".to_string()),
                            env: None,
                        })))
                    })
                }),
                None,
            )
            .await
            .unwrap();
    }
    let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
        credentials: Some(credentials),
        models_path: Some(None),
        refresh_on_create: true,
        ..Default::default()
    })
    .await
    .unwrap();
    runtime
        .register_native_provider(Arc::new(TestProvider { model }))
        .unwrap();
    Models::refresh(
        &*runtime,
        Some(pi_ai::models::ModelsRefreshOptions {
            allow_network: Some(false),
            providers: Some(vec![PROVIDER_ID.to_string()]),
            ..Default::default()
        }),
    )
    .await;
    runtime
}

/// Resource loader with inline extension factories and every other source
/// disabled (`createTestResourceLoader`).
pub async fn loader_with_factories(
    cwd: &str,
    factories: Vec<ExtensionFactory>,
) -> Arc<Mutex<DefaultResourceLoader>> {
    let cwd = cwd.to_string();
    tokio::task::spawn_blocking(move || {
        let mut loader = DefaultResourceLoader::new(DefaultResourceLoaderOptions {
            agent_dir: cwd.clone(),
            cwd,
            no_skills: true,
            no_prompt_templates: true,
            no_themes: true,
            no_context_files: true,
            extension_factories: factories
                .into_iter()
                .map(InlineExtension::Factory)
                .collect(),
            ..Default::default()
        });
        loader.reload(ResourceLoaderReloadOptions::default());
        Arc::new(Mutex::new(loader))
    })
    .await
    .unwrap()
}

pub fn temp_dir(tag: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "pi-rs-modes-{tag}-{}-{}",
        now_ms(),
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().to_string()
}

/// `streamFn` emitting start immediately and the completed message after
/// `delay_ms` (the semantics-suite `MockAssistantStream` timing).
pub fn delayed_stream_fn(delay_ms: u64, response: AssistantMessage) -> StreamFn {
    Arc::new(move |_model, _context, _options| {
        let start = assistant_message("");
        let done = response.clone();
        Box::pin(async move {
            let stream = AssistantMessageEventStream::new();
            stream.push(AssistantMessageEvent::Start { partial: start });
            if delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
            stream.push(AssistantMessageEvent::Done {
                reason: StreamDoneReason::Stop,
                message: done,
            });
            Ok(stream)
        })
    })
}

/// Stream function that always fails with the given message (provider-failure
/// cases).
pub fn failing_stream_fn(error_message: &'static str) -> StreamFn {
    Arc::new(move |_model, _context, _options| {
        Box::pin(async move {
            let stream = AssistantMessageEventStream::new();
            stream.push(AssistantMessageEvent::Start {
                partial: assistant_message(""),
            });
            stream.push(AssistantMessageEvent::Error {
                reason: StreamErrorReason::Error,
                error: error_assistant_message(error_message),
            });
            Ok(stream)
        })
    })
}
