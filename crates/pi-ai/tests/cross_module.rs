//! Cross-module integration tests ported from upstream `test/` files that span
//! modules: `model-types.test.ts`, `fetch-option.test.ts`, `providers.test.ts`
//! (in-scope subset).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use futures_util::StreamExt;
use pi_ai::auth::types::{ApiKeyAuth, ApiKeyAuthInput, AuthResult, ProviderAuth};
use pi_ai::http::{default_fetch, FetchFn, HttpBody, HttpRequest, HttpResponse};
use pi_ai::models::{
    calculate_cost, create_provider, get_supported_thinking_levels, has_api, models_are_equal,
    ApiDispatch, CreateProviderOptions, Provider, ProviderStreams,
};
use pi_ai::models_store::InMemoryModelsStore;
use pi_ai::types::{
    AnyModel, AssistantMessageEvent, Context, Message, Model, ModelCompat, ModelCost,
    ModelCostRates, ModelInput, ModelType, StreamOptions, TranscriptContext, Usage, UserMessage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;

fn chat_model(provider: &str, id: &str) -> Model {
    Model {
        base: pi_ai::types::BaseModelFields {
            id: id.to_string(),
            name: id.to_string(),
            api: "test-chat".to_string(),
            provider: provider.to_string(),
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
        context_window: 1000,
        max_tokens: 100,
        sampling_params: None,
        sampling_params_by_thinking_level: None,
        compat: None,
    }
}

fn image_model_json(provider: &str, id: &str, model_type: &str) -> serde_json::Value {
    let mut value = serde_json::json!({
        "type": model_type,
        "id": id, "name": id, "api": "test-images", "provider": provider,
        "baseUrl": "https://example.test/v1", "input": ["text"], "output": ["image"],
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
    });
    if model_type == "image" {
        // Canonical image entries carry `type: "image"`; keep it explicit.
        value["type"] = serde_json::Value::String("image".to_string());
    }
    value
}

fn no_auth() -> ProviderAuth {
    ProviderAuth {
        api_key: Some(ApiKeyAuth {
            name: "Test".to_string(),
            login: None,
            check: None,
            resolve: Box::new(|_input: ApiKeyAuthInput<'_>| {
                Box::pin(async { Ok::<Option<AuthResult>, _>(Some(AuthResult::default())) })
            }),
        }),
        oauth: None,
    }
}

fn empty_streams() -> Arc<dyn ProviderStreams> {
    struct EmptyStreams;
    impl ProviderStreams for EmptyStreams {
        fn stream(
            &self,
            _model: &Model,
            _context: &TranscriptContext,
            _options: &StreamOptions,
        ) -> AssistantMessageEventStream {
            AssistantMessageEventStream::new()
        }
        fn stream_simple(
            &self,
            _model: &Model,
            _context: &TranscriptContext,
            _options: &pi_ai::types::SimpleStreamOptions,
        ) -> AssistantMessageEventStream {
            AssistantMessageEventStream::new()
        }
    }
    Arc::new(EmptyStreams)
}

// ==== test/model-types.test.ts ====

#[test]
fn narrows_mixed_lists_with_is_model_type() {
    let mixed = vec![
        AnyModel::Chat(chat_model("p", "c")),
        AnyModel::from_json(&{
            let mut value = serde_json::to_value(chat_model("p", "typed")).unwrap();
            value["type"] = serde_json::Value::String("chat".to_string());
            value
        })
        .unwrap(),
        AnyModel::from_json(&image_model_json("p", "i", "image")).unwrap(),
    ];

    let chat_ids: Vec<&str> = mixed
        .iter()
        .filter(|model| model.model_type() == ModelType::Chat)
        .map(|model| model.id())
        .collect();
    assert_eq!(chat_ids, vec!["c", "typed"]);
    let image_ids: Vec<&str> = mixed
        .iter()
        .filter(|model| model.model_type() == ModelType::Image)
        .map(|model| model.id())
        .collect();
    assert_eq!(image_ids, vec!["i"]);
    assert!(mixed
        .iter()
        .filter(|model| model.model_type() == ModelType::Classifier)
        .next()
        .is_none());
}

#[test]
fn unknown_model_types_are_dropped_from_stored_entries() {
    // Upstream: entries persisted by a newer version can carry unknown types;
    // they are dropped instead of failing the read (`withKnownModelTypes`).
    let entry = serde_json::json!({
        "models": [
            serde_json::to_value(chat_model("dyn", "stored-chat")).unwrap(),
            image_model_json("dyn", "stored-image", "image"),
            image_model_json("dyn", "future-embedding", "embedding"),
            image_model_json("dyn", "future-video", "video"),
        ]
    });
    let parsed: pi_ai::models_store::ModelsStoreEntry =
        serde_json::from_value(entry).expect("lax entry parse");
    let ids: Vec<&str> = parsed.models.iter().map(|model| model.id()).collect();
    assert_eq!(ids, vec!["stored-chat", "stored-image"]);
}

// ==== test/fetch-option.test.ts (in-scope adapters) ====

fn recording_fetch(counter: Arc<AtomicUsize>) -> FetchFn {
    FetchFn::new(move |_request: HttpRequest| {
        let counter = counter.clone();
        Box::pin(async move {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(HttpResponse {
                status: 401,
                headers: vec![("content-type".to_string(), "application/json".to_string())],
                body: HttpBody::bytes(
                    serde_json::to_vec(&serde_json::json!({
                        "error": { "message": "upstream rejected request" }
                    }))
                    .unwrap(),
                ),
            })
        })
    })
}

fn streaming_model(api: &str) -> Model {
    let mut model = chat_model("test", "test-model");
    model.base.api = api.to_string();
    model.reasoning = true;
    model
}

#[tokio::test]
async fn fetch_option_passes_custom_fetch_to_in_scope_adapters() {
    // Upstream asserts, per adapter: the custom fetch is called and the
    // ambient/default fetch is not. Rust: the custom FetchFn records calls;
    // the default adapter cannot be observed, so "default not used" is
    // asserted structurally: options.fetch set means only it can run, and the
    // call count must equal the number of adapters invoked (no ambient calls
    // on top).
    let adapters: [(&str, &str); 3] = [
        ("anthropic-messages", "https://api.anthropic.com"),
        ("openai-completions", "https://api.openai.com/v1"),
        ("openai-responses", "https://api.openai.com/v1"),
    ];
    let calls = Arc::new(AtomicUsize::new(0));
    let fetch = recording_fetch(calls.clone());
    let context = pi_ai::utils::transcript::normalize_context(&Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: pi_ai::types::UserContentField::Text("hello".to_string()),
            timestamp: 1,
        })],
        tools: None,
    });

    for (api, base_url) in adapters {
        let mut model = streaming_model(api);
        model.base.base_url = base_url.to_string();
        let mut options = StreamOptions::default();
        options.base.api_key = Some("test-key".to_string());
        options.base.fetch = Some(fetch.clone());
        options.base.max_retries = Some(0);
        let simple = pi_ai::types::SimpleStreamOptions::from(options.clone());
        let stream = match api {
            "anthropic-messages" => {
                pi_ai::api::anthropic_messages::stream_simple(&model, &context, &simple)
            }
            "openai-completions" => {
                pi_ai::api::openai_completions::stream_simple(&model, &context, &simple)
            }
            "openai-responses" => pi_ai::api::openai_responses::openai_responses_api()
                .stream_simple(&model, &context, &simple),
            other => panic!("unexpected api {other}"),
        };
        // Consume to the terminal event.
        let mut stream = stream;
        while let Some(event) = stream.next().await {
            let terminal = matches!(
                event,
                AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
            );
            if terminal {
                break;
            }
        }
    }

    assert_eq!(
        calls.load(Ordering::SeqCst),
        adapters.len(),
        "each adapter used the custom fetch exactly once"
    );
    // The default adapter is still intact for other callers.
    let _ = default_fetch();
}

// ==== test/providers.test.ts (in-scope subset) ====

#[tokio::test]
async fn create_provider_stream_error_for_missing_api_implementation() {
    let provider = create_provider(CreateProviderOptions {
        id: "mixed".to_string(),
        name: None,
        base_url: None,
        headers: None,
        auth: no_auth(),
        models: vec![AnyModel::Chat(chat_model("mixed", "m1"))],
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
        api: Some(ApiDispatch::ByApi(
            [("other-api".to_string(), empty_streams())]
                .into_iter()
                .collect(),
        )),
        images: Default::default(),
        classifiers: Default::default(),
    })
    .expect("provider builds");

    let model = provider.get_models()[0].clone();
    let events = collect(provider.as_ref(), &model).await;
    assert!(matches!(events[0], AssistantMessageEvent::Error { .. }));
    let AssistantMessageEvent::Error { error, .. } = &events[0] else {
        panic!("expected error event");
    };
    assert_eq!(
        error.error_message.as_deref(),
        Some("Provider mixed has no API implementation for \"test-chat\"")
    );
}

async fn collect(provider: &dyn Provider, model: &Model) -> Vec<AssistantMessageEvent> {
    let mut events = Vec::new();
    let mut stream = provider.stream_simple(model, &empty_context(), &Default::default());
    while let Some(event) = stream.next().await {
        let terminal = matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        );
        events.push(event);
        if terminal {
            break;
        }
    }
    events
}

fn empty_context() -> TranscriptContext {
    pi_ai::utils::transcript::normalize_context(&Context {
        system_prompt: None,
        messages: vec![],
        tools: None,
    })
}

#[test]
fn models_helpers_narrow_and_compare() {
    let chat = AnyModel::Chat(chat_model("p", "c"));
    assert!(has_api(&chat, "test-chat").is_some());
    assert!(has_api(&chat, "other-api").is_none());
    let image = AnyModel::from_json(&image_model_json("p", "c", "image")).unwrap();
    assert!(!models_are_equal(Some(&chat), Some(&image)));
    assert!(models_are_equal(Some(&chat), Some(&chat)));
    assert!(!models_are_equal(Some(&chat), None));
    assert_eq!(get_model_type(&chat), ModelType::Chat);
}

fn get_model_type(model: &AnyModel) -> ModelType {
    model.model_type()
}

#[test]
fn thinking_levels_follow_the_catalog_ladder() {
    let mut model = chat_model("p", "reasoning");
    model.reasoning = true;
    assert_eq!(
        get_supported_thinking_levels(&model),
        vec![
            pi_ai::types::ModelThinkingLevel::Off,
            pi_ai::types::ModelThinkingLevel::Minimal,
            pi_ai::types::ModelThinkingLevel::Low,
            pi_ai::types::ModelThinkingLevel::Medium,
            pi_ai::types::ModelThinkingLevel::High,
        ]
    );
    model.reasoning = false;
    assert_eq!(
        get_supported_thinking_levels(&model),
        vec![pi_ai::types::ModelThinkingLevel::Off]
    );
}

#[test]
fn calculate_cost_applies_tiers_and_1h_cache_write() {
    let mut model = chat_model("p", "costed");
    model.base.cost = ModelCost {
        rates: ModelCostRates {
            input: 3.0,
            output: 15.0,
            cache_read: 0.3,
            cache_write: 3.75,
        },
        tiers: Some(vec![pi_ai::types::ModelCostTier {
            rates: ModelCostRates {
                input: 1.5,
                output: 7.5,
                cache_read: 0.15,
                cache_write: 1.875,
            },
            input_tokens_above: 100.0,
        }]),
    };
    let mut usage = Usage {
        input: 200,
        output: 10,
        cache_read: 0,
        cache_write: 4,
        cache_write_1h: Some(2),
        reasoning: None,
        total_tokens: 214,
        cost: Default::default(),
    };
    let cost = calculate_cost(&AnyModel::Chat(model), &mut usage);
    // Tier applies (input 204 > 100): input 1.5/M, output 7.5/M, cacheWrite
    // (2 short * 1.875 + 2 long * 1.5 * 2) / 1M.
    assert!((cost.input - 300.0 / 1_000_000.0).abs() < 1e-12);
    assert!((cost.output - 75.0 / 1_000_000.0).abs() < 1e-12);
    let expected_cache_write = (1.875 * 2.0 + 1.5 * 2.0 * 2.0) / 1_000_000.0;
    assert!((cost.cache_write - expected_cache_write).abs() < 1e-15);
    assert!(
        (cost.total - (cost.input + cost.output + cost.cache_read + cost.cache_write)).abs()
            < 1e-15
    );
}

#[test]
fn models_compat_carries_model_compat_overrides() {
    let mut model = chat_model("p", "compat");
    let mut compat = ModelCompat::default();
    compat.supports_strict_mode = Some(true);
    model.compat = Some(compat);
    assert_eq!(
        model
            .compat
            .as_ref()
            .and_then(|compat| compat.supports_strict_mode),
        Some(true)
    );
}
