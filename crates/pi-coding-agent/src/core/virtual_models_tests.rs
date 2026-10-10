//! Port of the portable suites of upstream
//! `test/virtual-models.test.ts`. The `getBranchSelection` describe block
//! ports 1:1 (its `runtime.getModel` catalog lookup becomes a static
//! catalog); the `ModelRuntime virtual models` and
//! `createAgentSession with virtual models` describe blocks need the
//! `ModelRuntime`/`AgentSession` runtime and defer with their ports. The
//! catalog-entry, unrouted-stream and catalog-filtering behavior those
//! suites assert about *this* module is covered by the extra tests below
//! (they exercise [`create_virtual_model`] and [`with_virtual_models`]
//! directly).

use std::cell::RefCell;
use std::sync::Arc;

use pi_ai::providers::faux::{
    faux_assistant_message, faux_provider, FauxMessageOptions, FauxModelDefinition,
    FauxProviderHandle, FauxResponseStep, RegisterFauxProviderOptions,
};
use pi_ai::types::{
    AnyModel, AssistantMessage, Context, Message, Model, ModelInput, ModelThinkingLevel,
    StopReason, StreamOptions, UserContentField, UserMessage,
};
use pi_ai::utils::thrown::ThrownValue;
use pi_ai::utils::transcript::normalize_context;

use super::{
    create_virtual_model, find_latest_response, get_branch_selection, get_virtual_model_state,
    is_virtual_model, with_virtual_models, BranchSelection, ModelRouteFn, VirtualModelDefinition,
    VIRTUAL_MODEL_STATE_ENTRY,
};
use crate::core::session_manager::SessionManager;

fn unreachable_route() -> ModelRouteFn {
    Arc::new(|_request| {
        Box::pin(async { Err(ThrownValue::message("route not expected in this test")) })
    })
}

/// The `createRuntime` definition: provider "router", id "auto", thinking
/// levels `["low", "high"]`.
fn auto_definition() -> VirtualModelDefinition {
    VirtualModelDefinition {
        provider: "router".to_string(),
        id: "auto".to_string(),
        name: "Auto".to_string(),
        thinking_levels: Some(vec![ModelThinkingLevel::Low, ModelThinkingLevel::High]),
        context_window: None,
        max_tokens: None,
        input: None,
        route: unreachable_route(),
    }
}

fn faux_models() -> (FauxProviderHandle, Model, Model) {
    let handle = faux_provider(RegisterFauxProviderOptions {
        models: vec![
            FauxModelDefinition {
                id: "small".to_string(),
                context_window: Some(1000),
                max_tokens: Some(100),
                input: Some(vec![ModelInput::Text]),
                ..Default::default()
            },
            FauxModelDefinition {
                id: "large".to_string(),
                context_window: Some(50_000),
                max_tokens: Some(5000),
                input: Some(vec![ModelInput::Text, ModelInput::Image]),
                reasoning: Some(true),
                ..Default::default()
            },
        ],
        ..Default::default()
    });
    let small = handle.get_model("small").expect("small model");
    let large = handle.get_model("large").expect("large model");
    (handle, small, large)
}

fn assistant_from(model: &Model, text: &str) -> AssistantMessage {
    let mut message = faux_assistant_message(text, FauxMessageOptions::default());
    message.api = model.api().to_string();
    message.provider = model.provider().to_string();
    message.model = model.id().to_string();
    message
}

/// The catalog behind `getBranchSelection`'s `getModel` lookup, recording
/// lookups like the upstream test does.
struct Catalog {
    small: Model,
    large: Model,
    virtual_auto: Model,
}

impl Catalog {
    fn new(small: Model, large: Model, virtual_auto: Model) -> Catalog {
        Catalog {
            small,
            large,
            virtual_auto,
        }
    }

    /// Upstream `select(...)`: build a branch, resolve its selection, and
    /// report the catalog lookups it cost.
    fn select(
        &self,
        build: impl FnOnce(&mut SessionManager),
    ) -> (Option<BranchSelection>, Vec<String>) {
        let mut session_manager =
            SessionManager::in_memory(None, None, None).expect("in-memory session");
        build(&mut session_manager);
        let lookups = RefCell::new(Vec::new());
        let selection =
            get_branch_selection(&session_manager.get_branch(), &mut |provider, model_id| {
                lookups.borrow_mut().push(format!("{provider}/{model_id}"));
                self.lookup(provider, model_id)
            });
        (selection, lookups.into_inner())
    }

    fn lookup(&self, provider: &str, model_id: &str) -> Option<Model> {
        match (provider, model_id) {
            ("faux", "small") => Some(self.small.clone()),
            ("faux", "large") => Some(self.large.clone()),
            ("router", "auto") => Some(self.virtual_auto.clone()),
            _ => None,
        }
    }
}

// #10198: the selection must not cost one catalog lookup per assistant message.
#[test]
fn looks_up_only_the_last_model_change() {
    let (_faux, small, large) = faux_models();
    let virtual_auto = create_virtual_model(&auto_definition());
    let catalog = Catalog::new(small.clone(), large.clone(), virtual_auto.clone());

    let physical = catalog.select(|session| {
        session
            .append_model_change(small.provider(), small.id())
            .unwrap();
        for _ in 0..100 {
            session
                .append_message(assistant_from(&large, "ok"))
                .unwrap();
        }
    });
    assert_eq!(
        physical.0,
        Some(BranchSelection {
            provider: "faux".to_string(),
            model_id: "large".to_string()
        })
    );
    assert_eq!(physical.1, ["faux/small"]);

    let routed = catalog.select(|session| {
        session
            .append_model_change(small.provider(), small.id())
            .unwrap();
        session
            .append_message(assistant_from(&small, "ok"))
            .unwrap();
        session
            .append_model_change(virtual_auto.provider(), virtual_auto.id())
            .unwrap();
        for _ in 0..100 {
            session
                .append_message(assistant_from(&large, "ok"))
                .unwrap();
        }
    });
    assert_eq!(
        routed.0,
        Some(BranchSelection {
            provider: "router".to_string(),
            model_id: "auto".to_string()
        })
    );
    assert_eq!(routed.1, ["router/auto"]);
}

#[test]
fn uses_the_last_model_change_without_responses_after_it() {
    let (_faux, small, large) = faux_models();
    let virtual_auto = create_virtual_model(&auto_definition());
    let catalog = Catalog::new(small.clone(), large, virtual_auto.clone());

    let (selection, lookups) = catalog.select(|session| {
        session
            .append_model_change(virtual_auto.provider(), virtual_auto.id())
            .unwrap();
        session
            .append_message(assistant_from(&small, "ok"))
            .unwrap();
        session
            .append_model_change(small.provider(), small.id())
            .unwrap();
    });
    assert_eq!(
        selection,
        Some(BranchSelection {
            provider: "faux".to_string(),
            model_id: "small".to_string()
        })
    );
    assert!(lookups.is_empty());
}

#[test]
fn uses_the_latest_physical_response_without_a_model_change() {
    let (_faux, small, large) = faux_models();
    let virtual_auto = create_virtual_model(&auto_definition());
    let catalog = Catalog::new(small, large, virtual_auto.clone());

    let (selection, lookups) = catalog.select(|session| {
        session
            .append_message(assistant_from(&catalog.small, "ok"))
            .unwrap();
        session
            .append_message(assistant_from(&catalog.large, "ok"))
            .unwrap();
        // Failed routing leaves the virtual model on its message.
        let mut failed = assistant_from(&virtual_auto, "");
        failed.stop_reason = StopReason::Error;
        session.append_message(failed).unwrap();
    });
    assert_eq!(
        selection,
        Some(BranchSelection {
            provider: "faux".to_string(),
            model_id: "large".to_string()
        })
    );
    assert!(lookups.is_empty());
}

/// Catalog-entry behavior the upstream runtime suite asserts before routing
/// (`lists a virtual model and routes it...`): shape, defaults, and the
/// thinking-level map.
#[test]
fn create_virtual_model_lists_the_selected_thinking_levels() {
    let model = create_virtual_model(&auto_definition());
    assert_eq!(model.provider(), "router");
    assert_eq!(model.id(), "auto");
    assert!(is_virtual_model(&model));
    assert_eq!(model.context_window, 0);
    assert_eq!(model.max_tokens, 0);
    assert_eq!(model.base.input, [ModelInput::Text, ModelInput::Image]);
    assert!(model.reasoning);
    assert_eq!(
        pi_ai::get_supported_thinking_levels(&model),
        [ModelThinkingLevel::Low, ModelThinkingLevel::High]
    );
    let map = model.thinking_level_map.expect("thinking level map");
    assert_eq!(map.get(&ModelThinkingLevel::Off), Some(&None));
    assert_eq!(
        map.get(&ModelThinkingLevel::Low).and_then(Option::as_deref),
        Some("low")
    );
    assert_eq!(
        map.get(&ModelThinkingLevel::High)
            .and_then(Option::as_deref),
        Some("high")
    );
    assert_eq!(map.get(&ModelThinkingLevel::Medium), Some(&None));
    assert_eq!(map.get(&ModelThinkingLevel::Max), Some(&None));

    // Defaults: `thinkingLevels` defaults to `["off"]`, input to text+images,
    // unknown limits to 0.
    let plain = create_virtual_model(&VirtualModelDefinition {
        provider: "router".to_string(),
        id: "plain".to_string(),
        name: "Plain".to_string(),
        thinking_levels: None,
        context_window: None,
        max_tokens: None,
        input: None,
        route: unreachable_route(),
    });
    assert!(!plain.reasoning);
    assert_eq!(
        pi_ai::get_supported_thinking_levels(&plain),
        [ModelThinkingLevel::Off]
    );
    assert_eq!(plain.base.input, [ModelInput::Text, ModelInput::Image]);
    assert_eq!(plain.context_window, 0);

    let sized = create_virtual_model(&VirtualModelDefinition {
        provider: "router".to_string(),
        id: "sized".to_string(),
        name: "Sized".to_string(),
        thinking_levels: Some(vec![ModelThinkingLevel::Off]),
        context_window: Some(4096),
        max_tokens: Some(512),
        input: Some(vec![ModelInput::Text]),
        route: unreachable_route(),
    });
    assert_eq!(sized.context_window, 4096);
    assert_eq!(sized.max_tokens, 512);
    assert_eq!(sized.base.input, [ModelInput::Text]);
}

/// Direct coverage of `findLatestResponse` (the branch-selection suite only
/// exercises the failed case): aborted requests are skipped too.
#[test]
fn find_latest_response_skips_failed_and_aborted_requests() {
    let (_faux, small, large) = faux_models();
    assert!(find_latest_response(&[]).is_none());

    let mut aborted = assistant_from(&small, "");
    aborted.stop_reason = StopReason::Aborted;
    let latest = assistant_from(&large, "ok");
    let messages = vec![
        Message::User(UserMessage {
            content: UserContentField::Text("hi".to_string()),
            timestamp: 1,
        }),
        Message::Assistant(assistant_from(&small, "answer")),
        Message::Assistant(aborted),
        Message::Assistant(latest.clone()),
    ];
    assert_eq!(find_latest_response(&messages), Some(&latest));
}

/// Router state lives on custom entries; the latest matching one wins.
#[test]
fn get_virtual_model_state_reads_the_latest_matching_entry() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory session");
    assert_eq!(
        get_virtual_model_state(&session.get_branch(), "router", "auto"),
        None
    );

    session
        .append_custom_entry(
            "other",
            Some(serde_json::json!({ "provider": "router", "modelId": "auto", "state": 1 })),
        )
        .unwrap();
    session
		.append_custom_entry(
			VIRTUAL_MODEL_STATE_ENTRY,
			Some(serde_json::json!({ "provider": "router", "modelId": "auto", "state": { "mode": "auto" } })),
		)
		.unwrap();
    session
        .append_custom_entry(
            VIRTUAL_MODEL_STATE_ENTRY,
            Some(serde_json::json!({ "provider": "router", "modelId": "auto", "state": "second" })),
        )
        .unwrap();

    let branch = session.get_branch();
    assert_eq!(
        get_virtual_model_state(&branch, "router", "auto"),
        Some(serde_json::json!("second"))
    );
    assert_eq!(get_virtual_model_state(&branch, "faux", "small"), None);
}

/// Upstream `fails unrouted stream calls on virtual models`, through the
/// keyless provider `withVirtualModels` builds without a base provider.
#[tokio::test]
async fn with_virtual_models_without_provider_fails_unrouted_streams() {
    let virtual_auto = create_virtual_model(&auto_definition());
    let provider = with_virtual_models("router", None, vec![virtual_auto.clone()]);
    assert_eq!(provider.id(), "router");
    assert_eq!(provider.name(), "router");
    assert_eq!(provider.get_models(), vec![virtual_auto.clone()]);
    assert_eq!(
        provider.get_all_models(),
        vec![AnyModel::Chat(virtual_auto.clone())]
    );

    let context = normalize_context(&Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserContentField::Text("hi".to_string()),
            timestamp: 1,
        })],
        tools: None,
    });
    let message = provider
        .stream(&virtual_auto, &context, &StreamOptions::default())
        .into_result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    let error_message = message.error_message.expect("error message");
    assert!(
        error_message.contains("must be routed before streaming"),
        "{error_message}"
    );
    assert!(error_message.contains("router/auto"), "{error_message}");
}

/// Upstream `lists several virtual models under a provider with physical
/// models`, minus the runtime registration: same-id physical chat models are
/// hidden, virtual models are appended and survive filtering, and only
/// virtual requests fail unrouted.
#[tokio::test]
async fn with_virtual_models_hides_same_id_physical_models_and_keeps_virtuals_in_filters() {
    let handle = faux_provider(RegisterFauxProviderOptions {
        models: vec![
            FauxModelDefinition {
                id: "small".to_string(),
                context_window: Some(1000),
                max_tokens: Some(100),
                input: Some(vec![ModelInput::Text]),
                ..Default::default()
            },
            FauxModelDefinition {
                id: "large".to_string(),
                context_window: Some(50_000),
                max_tokens: Some(5000),
                input: Some(vec![ModelInput::Text, ModelInput::Image]),
                reasoning: Some(true),
                ..Default::default()
            },
            // A physical chat model the virtual "auto" hides (a catalog
            // refresh can add it after registration).
            FauxModelDefinition {
                id: "auto".to_string(),
                ..Default::default()
            },
        ],
        ..Default::default()
    });
    let mut auto = auto_definition();
    auto.provider = "faux".to_string();
    let mut fast = auto.clone();
    fast.id = "fast".to_string();
    fast.name = "Fast".to_string();
    let virtual_auto = create_virtual_model(&auto);
    let virtual_fast = create_virtual_model(&fast);
    let provider = with_virtual_models(
        "faux",
        Some(handle.provider.clone()),
        vec![virtual_auto.clone(), virtual_fast],
    );

    let ids: Vec<String> = provider
        .get_models()
        .iter()
        .map(|model| model.id().to_string())
        .collect();
    assert_eq!(ids, ["small", "large", "auto", "fast"]);
    let all_ids: Vec<String> = provider
        .get_all_models()
        .iter()
        .map(|model| model.id().to_string())
        .collect();
    assert_eq!(all_ids, ["small", "large", "auto", "fast"]);

    // Virtual models survive availability filtering; the physical same-id
    // model stays hidden.
    let filtered = provider
        .filter_models(&provider.get_models(), None)
        .expect("filter_models");
    let filtered_ids: Vec<String> = filtered
        .iter()
        .map(|model| model.id().to_string())
        .collect();
    assert_eq!(filtered_ids, ["small", "large", "auto", "fast"]);

    // The faux provider defines no `filterAllModels` policy, so the wrapped
    // provider stays without one (upstream `filterAllModels && ...`).
    assert!(provider
        .filter_all_models(&provider.get_all_models(), None)
        .is_none());

    let context = normalize_context(&Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserContentField::Text("hi".to_string()),
            timestamp: 1,
        })],
        tools: None,
    });

    // Physical requests stream through to the provider...
    handle.set_responses(vec![FauxResponseStep::Message(assistant_from(
        &handle.get_model("small").unwrap(),
        "hello",
    ))]);
    let small = provider
        .get_models()
        .into_iter()
        .find(|model| model.id() == "small")
        .expect("small");
    let message = provider
        .stream(&small, &context, &StreamOptions::default())
        .into_result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.model, "small");

    // ...while virtual requests fail unrouted.
    let message = provider
        .stream(&virtual_auto, &context, &StreamOptions::default())
        .into_result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    let error_message = message.error_message.expect("error message");
    assert!(
        error_message.contains("must be routed before streaming"),
        "{error_message}"
    );
}
