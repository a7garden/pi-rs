//! Port of `packages/ai/src/compat.ts` — the temporary compatibility
//! entrypoint preserving the old global pi-ai API surface: api-dispatch
//! `stream()`/`complete()` with env API key injection, the api-registry,
//! generated catalog reads (`get_model`/`get_models`/`get_providers`),
//! per-API stream wrappers, and image generation.
//!
//! # Final-integration wiring points
//!
//! - `builtin_api_streams()` lists the builtin API implementations in
//!   upstream `BUILTIN_APIS` order. Only the APIs landed this wave are wired;
//!   the remaining seven entries land at final integration with the same
//!   one-line pattern (`crate::api::<module>::<api>_api()`).
//! - `builtin_models()` mirrors upstream `providers/all.ts`'s
//!   `builtinModels()` over [`crate::providers::builtin_provider_factories`]
//!   (which final integration extends as sibling factories land), so this
//!   module needs no edits when the provider list grows.
//!
//! # Deviations
//!
//! - D42: upstream registers the builtin APIs at module load; Rust globals
//!   initialize lazily, so `register_built_in_api_providers` runs on first
//!   registry access (same observable orderings — user overrides registered
//!   before first access win, `reset` restores builtins beneath them).

pub mod extension_oauth_types;

use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock};

use crate::models::{
    Models, ModelsApiStreamOptions, ModelsSimpleStreamOptions, MutableModels, Provider,
    ProviderStreams,
};
use crate::providers::faux::{FauxCore, RegisterFauxProviderOptions};
use crate::types::{AssistantMessage, Context, Model, SimpleStreamOptions, StreamOptions};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::thrown::ThrownValue;
use crate::utils::transcript::normalize_context;

/// The ambient auth marker resolved providers use for non-key credentials
/// (upstream `AMBIENT_AUTH_MARKER`).
const AMBIENT_AUTH_MARKER: &str = "<authenticated>";

// ==== api registry (upstream ApiProvider registry) ====

/// Upstream `ApiStreamFunction`.
pub type ApiStreamFunction = Arc<
    dyn Fn(&Model, &crate::types::TranscriptContext, &StreamOptions) -> AssistantMessageEventStream
        + Send
        + Sync,
>;

/// Upstream `ApiStreamSimpleFunction`.
pub type ApiStreamSimpleFunction = Arc<
    dyn Fn(
            &Model,
            &crate::types::TranscriptContext,
            &SimpleStreamOptions,
        ) -> AssistantMessageEventStream
        + Send
        + Sync,
>;

/// Upstream `ApiProviderInternal` — the registry value.
#[derive(Clone)]
pub struct ApiProviderInternal {
    pub api: String,
    pub stream: ApiStreamFunction,
    pub stream_simple: ApiStreamSimpleFunction,
}

struct RegisteredApiProvider {
    provider: ApiProviderInternal,
    source_id: Option<String>,
}

fn api_registry() -> &'static RwLock<BTreeMap<String, RegisteredApiProvider>> {
    static REGISTRY: LazyLock<RwLock<BTreeMap<String, RegisteredApiProvider>>> =
        LazyLock::new(|| {
            let registry = RwLock::new(BTreeMap::new());
            // registerBuiltInApiProviders() module-load side effect (D42).
            register_built_in_api_providers(&registry);
            registry
        });
    &REGISTRY
}

/// The builtin API implementations in upstream `BUILTIN_APIS` order. The
/// entries commented out land at final integration as their sibling modules
/// land (one-line pattern: `crate::api::<module>::<api>_api()`).
fn builtin_api_streams() -> Vec<(&'static str, Arc<dyn ProviderStreams>)> {
    vec![
        (
            "anthropic-messages",
            crate::api::anthropic_messages::anthropic_messages_api(),
        ),
        (
            "openai-completions",
            crate::api::openai_completions::openai_completions_api(),
        ),
        (
            "openai-responses",
            crate::api::openai_responses::openai_responses_api(),
        ),
        (
            "openai-codex-responses",
            crate::api::openai_codex_responses::openai_codex_responses_api(),
        ),
        (
            "azure-openai-responses",
            crate::api::azure_openai_responses::azure_openai_responses_api(),
        ),
        (
            "google-generative-ai",
            crate::api::google_generative_ai::google_generative_ai_api(),
        ),
        (
            "google-vertex",
            crate::api::google_vertex::google_vertex_api(),
        ),
        (
            "mistral-conversations",
            crate::api::mistral_conversations::mistral_conversations_api(),
        ),
        (
            "bedrock-converse-stream",
            crate::api::bedrock_converse_stream::bedrock_converse_stream_api(),
        ),
        ("pi-messages", crate::api::pi_messages::pi_messages_api()),
    ]
}

/// Wrap an API implementation's `stream` with the api guard
/// (upstream `wrapStream`).
fn wrap_stream(api: &'static str, streams: Arc<dyn ProviderStreams>) -> ApiStreamFunction {
    Arc::new(move |model, context, options| {
        if model.api() != api {
            return mismatched_stream(model, api);
        }
        streams.stream(model, context, options)
    })
}

fn wrap_stream_simple(
    api: &'static str,
    streams: Arc<dyn ProviderStreams>,
) -> ApiStreamSimpleFunction {
    Arc::new(move |model, context, options| {
        if model.api() != api {
            return mismatched_stream_simple(model, api);
        }
        streams.stream_simple(model, context, options)
    })
}

fn mismatched_stream(model: &Model, api: &str) -> AssistantMessageEventStream {
    let outer = AssistantMessageEventStream::new();
    let message = crate::types::AssistantMessage {
        content: Vec::new(),
        api: model.api().to_string(),
        provider: model.provider().to_string(),
        model: model.id().to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        usage: Default::default(),
        stop_reason: crate::types::StopReason::Error,
        deferred: None,
        error_message: Some(format!("Mismatched api: {} expected {}", model.api(), api)),
        raw_stop_reason: None,
        end_turn: None,
        duration_ms: None,
        timestamp: crate::utils::diagnostics::now_ms(),
    };
    outer.push(crate::types::AssistantMessageEvent::Start {
        partial: message.clone(),
    });
    outer.push(crate::types::AssistantMessageEvent::Error {
        reason: crate::types::StreamErrorReason::Error,
        error: message.clone(),
    });
    outer.end(Some(message));
    outer
}

fn mismatched_stream_simple(model: &Model, api: &str) -> AssistantMessageEventStream {
    mismatched_stream(model, api)
}

/// Upstream `registerApiProvider` — replaces any existing entry for the api.
pub fn register_api_provider(provider: ApiProviderInternal, source_id: Option<String>) {
    api_registry().write().insert(
        provider.api.clone(),
        RegisteredApiProvider {
            provider,
            source_id,
        },
    );
}

/// Upstream `getApiProvider`.
pub fn get_api_provider(api: &str) -> Option<ApiProviderInternal> {
    api_registry()
        .read()
        .get(api)
        .map(|entry| entry.provider.clone())
}

/// Upstream `getApiProviders`.
pub fn get_api_providers() -> Vec<ApiProviderInternal> {
    api_registry()
        .read()
        .values()
        .map(|entry| entry.provider.clone())
        .collect()
}

/// Upstream `unregisterApiProviders(sourceId)`.
pub fn unregister_api_providers(source_id: &str) {
    api_registry()
        .write()
        .retain(|_, entry| entry.source_id.as_deref() != Some(source_id));
}

fn clear_api_providers() {
    api_registry().write().clear();
}

/// Upstream `registerFauxProvider` — registers a faux API + returns the
/// registration handle.
pub fn register_faux_provider(
    options: RegisterFauxProviderOptions,
) -> FauxProviderRegistrationCompat {
    let core = FauxCore::new(options);
    let source_id = format!("faux-provider-{}", random_compat_suffix());
    register_faux_core(&core, &source_id);
    FauxProviderRegistrationCompat {
        api: core.api().to_string(),
        core,
        source_id,
    }
}

/// Upstream `Math.random().toString(36).slice(2, 10)` suffix.
fn random_compat_suffix() -> String {
    let mut bytes = [0u8; 8];
    let _ = getrandom::getrandom(&mut bytes);
    let mut value = u64::from_le_bytes(bytes);
    let mut suffix = String::new();
    while suffix.len() < 8 {
        let digit = (value % 36) as u32;
        value /= 36;
        let ch = if digit < 10 {
            char::from(b'0' + digit as u8)
        } else {
            char::from(b'a' + (digit - 10) as u8)
        };
        suffix.push(ch);
    }
    suffix
}

/// The compat-facing registration handle (upstream
/// `FauxProviderRegistration`): the scripted core plus `unregister()`.
pub struct FauxProviderRegistrationCompat {
    pub api: String,
    pub core: FauxCore,
    source_id: String,
}

impl FauxProviderRegistrationCompat {
    pub fn unregister(self) {
        unregister_api_providers(&self.source_id);
    }
}

fn register_faux_core(core: &FauxCore, source_id: &str) {
    let api = core.api().to_string();
    let core_for_stream = core.clone();
    let core_for_simple = core.clone();
    register_api_provider(
        ApiProviderInternal {
            api: api.clone(),
            stream: Arc::new(move |model, context, options| {
                core_for_stream.stream(
                    model,
                    context,
                    Some(&SimpleStreamOptions {
                        base: options.clone(),
                        ..Default::default()
                    }),
                )
            }),
            stream_simple: Arc::new(move |model, context, options| {
                core_for_simple.stream(model, context, Some(options))
            }),
        },
        Some(source_id.to_string()),
    );
}

/// Snapshots of the builtin registry entries, used to detect whether a model's
/// api still resolves to the builtin implementation
/// (upstream `builtinApiProviderInstances`).
fn builtin_api_provider_instances() -> &'static BTreeMap<String, ApiProviderInternal> {
    static INSTANCES: LazyLock<BTreeMap<String, ApiProviderInternal>> = LazyLock::new(|| {
        let mut instances = BTreeMap::new();
        for (api, streams) in builtin_api_streams() {
            instances.insert(
                api.to_string(),
                ApiProviderInternal {
                    api: api.to_string(),
                    stream: wrap_stream(api, streams.clone()),
                    stream_simple: wrap_stream_simple(api, streams),
                },
            );
        }
        instances
    });
    &INSTANCES
}

/// Upstream `registerBuiltInApiProviders` — registers builtins without
/// clobbering existing entries.
fn register_built_in_api_providers(registry: &RwLock<BTreeMap<String, RegisteredApiProvider>>) {
    for (api, streams) in builtin_api_streams() {
        let mut guard = registry.write();
        if !guard.contains_key(api) {
            let provider = ApiProviderInternal {
                api: api.to_string(),
                stream: wrap_stream(api, streams.clone()),
                stream_simple: wrap_stream_simple(api, streams),
            };
            guard.insert(
                api.to_string(),
                RegisteredApiProvider {
                    provider,
                    source_id: None,
                },
            );
        }
    }
}

/// Upstream `resetApiProviders` — clears everything and restores builtins.
pub fn reset_api_providers() {
    clear_api_providers();
    register_built_in_api_providers(api_registry());
}

// ==== builtin provider catalog reads (upstream providers/all.ts surface) ====

/// Upstream `builtinModels()` — a `Models` collection over every builtin
/// provider factory. Final integration extends
/// [`crate::providers::builtin_provider_factories`]; this lazy stays correct
/// per process (factories are static).
pub fn builtin_models() -> Arc<crate::models::ModelsImpl> {
    static MODELS: LazyLock<Arc<crate::models::ModelsImpl>> = LazyLock::new(|| {
        let models = crate::models::create_models(None);
        for (id, factory) in crate::providers::builtin_provider_factories() {
            let _ = id;
            models.set_provider(factory());
        }
        models
    });
    MODELS.clone()
}

/// Upstream `getBuiltinModel` (compat re-exports it as `getModel`).
pub fn get_builtin_model(provider: &str, id: &str) -> Option<Model> {
    builtin_models().get_model(provider, id)
}

/// Upstream `getBuiltinModels` (compat re-exports it as `getModels`).
pub fn get_builtin_models(provider: Option<&str>) -> Vec<Model> {
    builtin_models().get_models(provider)
}

/// Upstream `getBuiltinProviders` (compat re-exports it as `getProviders`).
pub fn get_builtin_providers() -> Vec<Arc<dyn Provider>> {
    builtin_models().get_providers()
}

/// Deprecated alias for [`get_builtin_model`] (upstream `getModel`).
pub fn get_model(provider: &str, id: &str) -> Option<Model> {
    get_builtin_model(provider, id)
}

/// Deprecated alias for [`get_builtin_models`] (upstream `getModels`).
pub fn get_models(provider: Option<&str>) -> Vec<Model> {
    get_builtin_models(provider)
}

/// Deprecated alias for [`get_builtin_providers`] (upstream `getProviders`).
pub fn get_providers() -> Vec<Arc<dyn Provider>> {
    get_builtin_providers()
}

// ==== env api key injection + dispatch ====

fn has_explicit_api_key(api_key: Option<&String>) -> bool {
    api_key.is_some_and(|key| !key.trim().is_empty())
}

/// Upstream `withEnvApiKey` — fills `options.apiKey` from the provider's env
/// when the caller did not pass one.
fn with_env_api_key(model: &Model, options: &StreamOptions) -> StreamOptions {
    if has_explicit_api_key(options.api_key.as_ref()) {
        return options.clone();
    }
    let api_key = get_env_api_key(model.provider(), options.env.as_ref());
    match api_key {
        Some(api_key) if api_key != AMBIENT_AUTH_MARKER => {
            let mut options = options.clone();
            options.api_key = Some(api_key);
            options
        }
        _ => options.clone(),
    }
}

fn with_env_api_key_simple(model: &Model, options: &SimpleStreamOptions) -> SimpleStreamOptions {
    let merged = with_env_api_key(model, &options.base);
    let mut options = options.clone();
    options.base = merged;
    options
}

/// Cloudflare providers fall back to auth-resolved streaming until a key or
/// gateway header exists (upstream `hasResolvedCloudflareAuth`).
fn has_resolved_cloudflare_auth(options: Option<&StreamOptions>) -> bool {
    let Some(options) = options else { return false };
    has_explicit_api_key(options.api_key.as_ref())
        || options
            .headers
            .as_ref()
            .and_then(|headers| headers.get("cf-aig-authorization"))
            .is_some_and(|value| value.is_some())
}

/// Whether the api's registry entry is still the builtin one (entries
/// registered with a `source_id` are user/test overrides; builtins carry
/// `None`).
fn is_builtin_api_provider(api: &str) -> bool {
    fn entry_source_is_none(api: &str) -> Option<bool> {
        let registry = api_registry();
        let guard = registry.read();
        guard.get(api).map(|entry| entry.source_id.is_none())
    }
    entry_source_is_none(api).unwrap_or(false)
}

/// Upstream `getBuiltinProviderForModel` — the builtin provider only applies
/// when the api registry entry is the builtin one and the provider's catalog
/// lists the model's api.
fn get_builtin_provider_for_model(model: &Model) -> Option<Arc<dyn Provider>> {
    if !is_builtin_api_provider(model.api()) {
        return None;
    }
    if !builtin_api_provider_instances().contains_key(model.api()) {
        return None;
    }
    let provider = builtin_models().get_provider(model.provider())?;
    let lists_model = provider
        .get_models()
        .iter()
        .any(|candidate| candidate.api() == model.api());
    lists_model.then_some(provider)
}

/// Upstream `resolveApiProvider`.
fn resolve_api_provider(api: &str) -> Result<ApiProviderInternal, ThrownValue> {
    get_api_provider(api)
        .ok_or_else(|| ThrownValue::message(format!("No API provider registered for api: {api}")))
}

/// Upstream `stream` — the legacy global API dispatch.
pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<StreamOptions>,
) -> AssistantMessageEventStream {
    let transcript = normalize_context(context);
    let options = options.unwrap_or_default();
    if let Some(builtin_provider) = get_builtin_provider_for_model(model) {
        if model.provider().starts_with("cloudflare-")
            && !has_resolved_cloudflare_auth(Some(&options))
        {
            return builtin_models().stream(
                model,
                context,
                Some(ModelsApiStreamOptions {
                    base: options,
                    ..Default::default()
                }),
            );
        }
        let options = with_env_api_key(model, &options);
        return builtin_provider.stream(model, &transcript, &options);
    }
    match resolve_api_provider(model.api()) {
        Ok(provider) => (provider.stream)(model, &transcript, &with_env_api_key(model, &options)),
        Err(error) => {
            let outer = AssistantMessageEventStream::new();
            let message = crate::types::AssistantMessage {
                content: Vec::new(),
                api: model.api().to_string(),
                provider: model.provider().to_string(),
                model: model.id().to_string(),
                response_model: None,
                response_id: None,
                provider_thinking_level: None,
                thinking_level: None,
                diagnostics: None,
                usage: Default::default(),
                stop_reason: crate::types::StopReason::Error,
                deferred: None,
                error_message: Some(crate::utils::diagnostics::format_thrown_value(&error)),
                raw_stop_reason: None,
                end_turn: None,
                duration_ms: None,
                timestamp: crate::utils::diagnostics::now_ms(),
            };
            outer.push(crate::types::AssistantMessageEvent::Start {
                partial: message.clone(),
            });
            outer.push(crate::types::AssistantMessageEvent::Error {
                reason: crate::types::StreamErrorReason::Error,
                error: message.clone(),
            });
            outer.end(Some(message));
            outer
        }
    }
}

/// Upstream `complete`.
pub async fn complete(
    model: &Model,
    context: &Context,
    options: Option<StreamOptions>,
) -> AssistantMessage {
    stream(model, context, options).result().await
}

/// Upstream `streamSimple`.
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let transcript = normalize_context(context);
    let options = options.unwrap_or_default();
    if let Some(builtin_provider) = get_builtin_provider_for_model(model) {
        if model.provider().starts_with("cloudflare-")
            && !has_resolved_cloudflare_auth(Some(&options.base))
        {
            return builtin_models().stream_simple(
                model,
                context,
                Some(ModelsSimpleStreamOptions {
                    base: options,
                    ..Default::default()
                }),
            );
        }
        let options = with_env_api_key_simple(model, &options);
        return builtin_provider.stream_simple(model, &transcript, &options);
    }
    match resolve_api_provider(model.api()) {
        Ok(provider) => (provider.stream_simple)(
            model,
            &transcript,
            &with_env_api_key_simple(model, &options),
        ),
        Err(error) => {
            let outer = AssistantMessageEventStream::new();
            let message = crate::types::AssistantMessage {
                content: Vec::new(),
                api: model.api().to_string(),
                provider: model.provider().to_string(),
                model: model.id().to_string(),
                response_model: None,
                response_id: None,
                provider_thinking_level: None,
                thinking_level: None,
                diagnostics: None,
                usage: Default::default(),
                stop_reason: crate::types::StopReason::Error,
                deferred: None,
                error_message: Some(crate::utils::diagnostics::format_thrown_value(&error)),
                raw_stop_reason: None,
                end_turn: None,
                duration_ms: None,
                timestamp: crate::utils::diagnostics::now_ms(),
            };
            outer.push(crate::types::AssistantMessageEvent::Start {
                partial: message.clone(),
            });
            outer.push(crate::types::AssistantMessageEvent::Error {
                reason: crate::types::StreamErrorReason::Error,
                error: message.clone(),
            });
            outer.end(Some(message));
            outer
        }
    }
}

/// Upstream `completeSimple`.
pub async fn complete_simple(
    model: &Model,
    context: &Context,
    options: Option<SimpleStreamOptions>,
) -> AssistantMessage {
    stream_simple(model, context, options).result().await
}

// Deprecated per-API wrappers, re-exported for the compat surface
// (upstream `export * from "./legacy-api-aliases.ts"`).
pub use crate::legacy_api_aliases::{
    stream_anthropic as compat_stream_anthropic,
    stream_openai_completions as compat_stream_openai_completions,
    stream_openai_responses as compat_stream_openai_responses,
    stream_simple_anthropic as compat_stream_simple_anthropic,
    stream_simple_openai_completions as compat_stream_simple_openai_completions,
    stream_simple_openai_responses as compat_stream_simple_openai_responses,
};

// Re-exports matching upstream `export * from` list (in-scope modules):
// env-api-keys, image-models, images, images-api-registry, index, legacy
// aliases.
pub use crate::env_api_keys::{
    find_env_keys, get_env_api_key, ANTHROPIC_API_KEY_ENV, ANTHROPIC_AUTH_TOKEN_ENV,
    ANTHROPIC_FEDERATION_RULE_ID_ENV, ANTHROPIC_IDENTITY_TOKEN_FILE_ENV, ANTHROPIC_OAUTH_TOKEN_ENV,
    ANTHROPIC_ORGANIZATION_ID_ENV, ANTHROPIC_SERVICE_ACCOUNT_ID_ENV, ANTHROPIC_WORKSPACE_ID_ENV,
};
pub use crate::image_models::{get_image_model, get_image_models, get_image_providers};
pub use crate::images::generate_images;
pub use crate::images_api_registry::{
    get_images_api_provider, register_images_api_provider, ImagesApiFunction, ImagesApiProvider,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn model(api: &str) -> Model {
        serde_json::from_value(serde_json::json!({
            "id": "test-model",
            "name": "Test Model",
            "api": api,
            "provider": "custom-openai",
            "baseUrl": "https://example.invalid/v1",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 128_000,
            "maxTokens": 4_096,
        }))
        .expect("model")
    }

    fn context() -> Context {
        Context {
            system_prompt: None,
            messages: vec![crate::types::Message::User(crate::types::UserMessage {
                content: crate::types::UserContentField::Text("hi".to_string()),
                timestamp: crate::utils::diagnostics::now_ms(),
            })],
            tools: None,
        }
    }

    // Port of test/compat-env.test.ts "dispatches unknown providers through the
    // legacy API registry" — a custom api-key flows through untouched.
    #[tokio::test]
    async fn dispatches_unknown_providers_through_the_legacy_api_registry() {
        let _guard = test_registry_guard().lock().await;
        reset_api_providers();
        let captured: Arc<parking_lot::Mutex<Option<Option<String>>>> =
            Arc::new(parking_lot::Mutex::new(None));
        let captured_for_stream = captured.clone();
        let provider = ApiProviderInternal {
            api: "openai-responses".to_string(),
            stream: Arc::new(move |_model, context, options| {
                *captured_for_stream.lock() = Some(options.base.api_key.clone());
                ok_stream(context)
            }),
            stream_simple: Arc::new(move |_model, _context, _options| {
                ok_stream(&crate::utils::transcript::normalize_context(
                    &crate::types::Context {
                        system_prompt: None,
                        messages: Vec::new(),
                        tools: None,
                    },
                ))
            }),
        };
        register_api_provider(provider, None);

        let message = complete(
            &model("openai-responses"),
            &context(),
            Some(StreamOptions {
                base: crate::types::ProviderRequestOptions {
                    api_key: Some("request-key".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .await;
        assert_eq!(message.stop_reason, crate::types::StopReason::Stop);
        assert_eq!(
            captured.lock().clone().flatten().as_deref(),
            Some("request-key")
        );
        reset_api_providers();
    }

    fn ok_stream(_context: &crate::types::TranscriptContext) -> AssistantMessageEventStream {
        let outer = AssistantMessageEventStream::new();
        let message = crate::types::AssistantMessage {
            content: vec![crate::types::AssistantContent::Text(
                crate::types::TextContent::new("ok"),
            )],
            api: "openai-responses".to_string(),
            provider: "custom-openai".to_string(),
            model: "test-model".to_string(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            thinking_level: None,
            diagnostics: None,
            usage: Default::default(),
            stop_reason: crate::types::StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            duration_ms: None,
            timestamp: crate::utils::diagnostics::now_ms(),
        };
        outer.push(crate::types::AssistantMessageEvent::Start {
            partial: message.clone(),
        });
        outer.push(crate::types::AssistantMessageEvent::Done {
            reason: crate::types::StreamDoneReason::Stop,
            message: message.clone(),
        });
        outer.end(Some(message));
        outer
    }

    /// Serializes registry-mutating tests.
    pub(crate) fn test_registry_guard() -> &'static tokio::sync::Mutex<()> {
        static GUARD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
        &GUARD
    }

    // Port of the "No API provider registered" error path from upstream
    // resolveApiProvider.
    #[tokio::test]
    async fn errors_for_unregistered_apis() {
        let _guard = test_registry_guard().lock().await;
        reset_api_providers();
        let result = stream(&model("never-registered"), &context(), None)
            .result()
            .await;
        assert_eq!(result.stop_reason, crate::types::StopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("No API provider registered for api: never-registered")
        );
        reset_api_providers();
    }
}
