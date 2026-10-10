//! Port of `packages/ai/src/providers/openai.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`crate::models::create_provider`] returns `Result`, unwrapped with
//!   `expect` because the configuration here is static and valid.
//! - The upstream `openai-decisions.lazy.ts` dynamic import becomes a direct
//!   reference (static linking, crate deviation D10): the Rust binary has no
//!   import step to defer.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::{Credential, ProviderAuth};
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};
use crate::types::{AnyModel, ClassifierModel};

/// Upstream `openaiProvider()`.
pub fn openai_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: "openai".to_string(),
        name: Some("OpenAI".to_string()),
        base_url: Some("https://api.openai.com/v1".to_string()),
        headers: None,
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("OpenAI API key", &["OPENAI_API_KEY"])),
            oauth: Some(Arc::new(crate::auth::helpers::lazy_oauth(
                crate::auth::helpers::LazyOAuthInput {
                    name: "OpenAI (ChatGPT subscription)".to_string(),
                    is_subscription: true,
                    login_label: Some("Sign in with ChatGPT".to_string()),
                    load: crate::auth::oauth::load::load_openai_chatgpt_oauth,
                },
            ))),
        },
        models: {
            let mut models: Vec<AnyModel> = super::openai_models::openai_models()
                .into_iter()
                .map(AnyModel::Chat)
                .collect();
            models.extend(
                super::openai_models::openai_classifier_models()
                    .into_iter()
                    .map(AnyModel::Classifier),
            );
            models
        },
        fetch_models: None,
        filter_models: None,
        // Sign in with ChatGPT tokens only reach the Responses API; the
        // Decisions API rejects them, so classifier models are hidden while
        // the provider uses OAuth credentials.
        filter_all_models: Some(Box::new(
            |models: &[AnyModel], credential: Option<&Credential>| match credential {
                Some(Credential::OAuth(_)) => models
                    .iter()
                    .filter(|model| !matches!(model, AnyModel::Classifier(_)))
                    .cloned()
                    .collect(),
                _ => models.to_vec(),
            },
        )),
        api: Some(ApiDispatch::Single(
            crate::api::openai_responses::openai_responses_api(),
        )),
        images: Default::default(),
        classifiers: [(
            crate::types::classifier_api_ids::OPENAI_DECISIONS.to_string(),
            Arc::new(OpenAiDecisionsClassifier) as Arc<dyn crate::models::ProviderClassifier>,
        )]
        .into_iter()
        .collect(),
    })
    .expect("static openai provider configuration is valid")
}

/// Upstream `openai-decisions.lazy.ts` (`openAIDecisionsApi()`): defers to the
/// `openai-decisions` classifier implementation.
struct OpenAiDecisionsClassifier;

impl crate::models::ProviderClassifier for OpenAiDecisionsClassifier {
    fn classify<'a>(
        &'a self,
        model: &'a ClassifierModel,
        context: &'a crate::types::ClassifierContext,
        options: &'a crate::types::ClassifierOptions,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = crate::types::ClassifierResult> + Send + 'a>,
    > {
        Box::pin(crate::api::openai_decisions::classify(
            model,
            context,
            Some(options),
        ))
    }
}

#[cfg(test)]
mod tests {
    // Ports of the v1.1.0 test/classifier-models.test.ts additions that need
    // the OpenAI provider wiring: GPT-6 Luna routing through the Decisions API
    // and classifier availability by credential type.
    use super::*;
    use crate::http::{FetchFn, HttpBody, HttpRequest, HttpResponse};
    use crate::types::{
        ClassifierAnswer, ClassifierContext, ClassifierOptions, ClassifierQuestion,
        ClassifierStopReason, JsonObject, ModelInput, ProviderRequestOptions,
    };
    use parking_lot::Mutex;

    #[test]
    fn ships_the_gpt_6_luna_classifier_entry() {
        let models = crate::providers::openai_models::openai_classifier_models();
        let luna = models
            .iter()
            .find(|model| model.base.id == "gpt-6-luna")
            .expect("missing OpenAI Decisions model");
        assert_eq!(luna.base.api, "openai-decisions");
        assert!(luna.base.input.contains(&ModelInput::Image));
        assert_eq!(luna.context_window, 922_000);
        // The chat entry with the same id stays separate.
        let chat = crate::providers::openai_models::openai_models();
        assert_eq!(
            chat.iter()
                .find(|model| model.base.id == "gpt-6-luna")
                .expect("chat luna")
                .base
                .api,
            "openai-responses"
        );
    }

    #[test]
    fn hides_classifier_models_for_oauth_credentials_only() {
        let provider = openai_provider();
        let all = provider.get_all_models();
        let oauth = Credential::OAuth(crate::auth::types::OAuthCredential {
            credentials: crate::auth::types::OAuthCredentials {
                refresh: "refresh".to_string(),
                access: "access".to_string(),
                expires: i64::MAX,
                extra: Default::default(),
            },
        });
        let visible_oauth = provider
            .filter_all_models(&all, Some(&oauth))
            .expect("filter is configured");
        assert!(visible_oauth
            .iter()
            .all(|model| !matches!(model, AnyModel::Classifier(_))));
        // Chat models stay available with ChatGPT OAuth.
        assert!(visible_oauth
            .iter()
            .any(|model| matches!(model, AnyModel::Chat(model) if model.id() == "gpt-6-luna")));

        let visible_without = provider
            .filter_all_models(&all, None)
            .expect("filter is configured");
        assert!(visible_without
            .iter()
            .any(|model| matches!(model, AnyModel::Classifier(_))));
    }

    #[tokio::test]
    async fn routes_the_luna_classifier_through_the_decisions_api() {
        let provider = openai_provider();
        let all = provider.get_all_models();
        let Some(AnyModel::Classifier(luna)) = all
            .iter()
            .find(|model| matches!(model, AnyModel::Classifier(m) if m.base.id == "gpt-6-luna"))
        else {
            panic!("missing OpenAI Decisions model");
        };
        let luna = luna.clone();

        let recording: Arc<Mutex<Vec<String>>> = Default::default();
        let recording_for_fetch = recording.clone();
        let fetch = FetchFn::new(move |request: HttpRequest| {
            recording_for_fetch.lock().push(request.url.clone());
            let body = serde_json::to_vec(&serde_json::json!({
                "answers": [{ "type": "predicate", "name": "approved", "probability": 0.8 }],
            }))
            .expect("response serializes");
            Box::pin(async move {
                Ok(HttpResponse {
                    status: 200,
                    headers: Vec::new(),
                    body: HttpBody::Bytes(body),
                })
            })
        });
        let state: JsonObject = serde_json::json!({ "text": "ok" })
            .as_object()
            .expect("object")
            .clone();
        let mut context = ClassifierContext {
            state,
            images: Some(vec![crate::types::ImageContent::new(
                "aW1hZ2U=",
                "image/png",
            )]),
            questions: Default::default(),
        };
        context.questions.insert(
			"approved".to_string(),
			serde_json::from_value::<ClassifierQuestion>(serde_json::json!({
				"type": "bool", "instructions": "Approve?", "criteria": { "true": "Yes", "false": "No" }
			}))
			.expect("question"),
		);
        let options = ClassifierOptions {
            base: ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(fetch),
                ..Default::default()
            },
            temperature: None,
        };

        let operation = provider
            .classify(&luna, &context, &options)
            .expect("openai-decisions classifier is wired for gpt-6-luna");
        let result = operation.await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        assert_eq!(
            result.answers.get("approved"),
            Some(&ClassifierAnswer::Bool { probability: 0.8 })
        );
        assert_eq!(
            recording.lock().as_slice(),
            ["https://api.openai.com/v1/decisions".to_string()]
        );
    }
}
