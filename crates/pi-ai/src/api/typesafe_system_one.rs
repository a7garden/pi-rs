//! Port of `packages/ai/src/api/typesafe-system-one.ts`.
//!
//! TypeSafe's native System One protocol. OpenRouter serves the same protocol,
//! so both providers use this API with different base URLs.
//!
//! Deviations: shared-client deviations D24–D26 apply (error carrier, per
//! attempt timeout signals, hook model view).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::{Map, Value};

use crate::api::classifier_shared::is_record;
use crate::api::system_one_shared::{
    classify_system_one, SystemOneTransport, SystemOneWireRequest,
};
use crate::http::ProviderHttpError;
use crate::models::ProviderClassifier;
use crate::types::{
    ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierResult, JsonObject,
};

/// The transport shared by TypeSafe and OpenRouter (`transport` upstream).
pub struct TypesafeSystemOneTransport;

static TRANSPORT: TypesafeSystemOneTransport = TypesafeSystemOneTransport;

impl SystemOneTransport for TypesafeSystemOneTransport {
    fn api(&self) -> &'static str {
        "typesafe-system-one"
    }

    fn label(&self) -> &'static str {
        "System One API"
    }

    fn url(&self, model: &ClassifierModel) -> String {
        let base = model.base.base_url.trim_end_matches('/');
        format!("{base}/systemone")
    }

    fn payload(&self, model: &ClassifierModel, request: &SystemOneWireRequest) -> Value {
        // `{ model: model.id, ...request }`.
        let mut payload = Map::new();
        payload.insert("model".to_string(), Value::String(model.id().to_string()));
        payload.insert("state".to_string(), Value::Object(request.state.clone()));
        payload.insert(
            "questions".to_string(),
            Value::Object(request.questions.clone()),
        );
        Value::Object(payload)
    }

    fn output(&self, body: &Value) -> Result<JsonObject, ProviderHttpError> {
        if !is_record(body) {
            return Err(ProviderHttpError::new(
                "System One API returned an unexpected response",
            ));
        }
        Ok(body.as_object().expect("checked record above").clone())
    }
}

/// TypeSafe System One classifier API (`classify` upstream).
pub struct TypesafeSystemOneApi;

impl ProviderClassifier for TypesafeSystemOneApi {
    fn classify<'a>(
        &'a self,
        model: &'a ClassifierModel,
        context: &'a ClassifierContext,
        options: &'a ClassifierOptions,
    ) -> Pin<Box<dyn Future<Output = ClassifierResult> + Send + 'a>> {
        Box::pin(classify_system_one(
            &TRANSPORT,
            model,
            context,
            Some(options),
        ))
    }
}

/// TypeSafe System One classification with public `bool` values mapped to
/// wire-level `noul`.
pub fn typesafe_system_one_api() -> Arc<dyn ProviderClassifier> {
    Arc::new(TypesafeSystemOneApi)
}

#[cfg(test)]
pub(crate) mod tests {
    // Port of test/typesafe-system-one.test.ts. `pub(crate)` so the shared
    // client tests in `system_one_shared.rs` can reuse the fixtures.
    use parking_lot::Mutex;

    use super::*;
    use crate::http::{FetchFn, HttpBody, HttpRequest, HttpResponse};
    use crate::types::{
        BaseModelFields, ClassifierAnswer, ClassifierStopReason, ClassifierTypeTag, ModelCost,
        ModelCostRates, ModelInput,
    };
    use crate::utils::abort::AbortSignal;

    pub fn make_model() -> ClassifierModel {
        ClassifierModel {
            base: BaseModelFields {
                id: "jev-latest".to_string(),
                name: "Jev".to_string(),
                api: "typesafe-system-one".to_string(),
                provider: "typesafe".to_string(),
                base_url: "https://api.typesafe.ai/v1/".to_string(),
                input: vec![ModelInput::Text],
                input_limits: None,
                cost: ModelCost {
                    rates: ModelCostRates {
                        input: 0.0,
                        output: 0.0,
                        cache_read: 0.0,
                        cache_write: 0.0,
                    },
                    tiers: None,
                },
                headers: None,
            },
            model_type: ClassifierTypeTag::Classifier,
            context_window: 64_000,
        }
    }

    pub fn make_context() -> ClassifierContext {
        ClassifierContext {
            state: serde_json::json!({ "text": "The deployment succeeded, thank you." })
                .as_object()
                .expect("object")
                .clone(),
            images: None,
            questions: [
                (
                    "category",
                    serde_json::json!({
                        "type": "choice",
                        "instructions": "Classify the message",
                        "criteria": { "success": "Successful", "failure": "Failed" }
                    }),
                ),
                (
                    "satisfaction",
                    serde_json::json!({
                        "type": "score",
                        "instructions": "Score satisfaction",
                        "criteria": ["low", "neutral", "high"]
                    }),
                ),
                (
                    "approved",
                    serde_json::json!({
                        "type": "bool",
                        "instructions": "Does the user approve?",
                        "criteria": { "true": "Approval", "false": "No approval" }
                    }),
                ),
            ]
            .into_iter()
            .map(|(id, question)| {
                (
                    id.to_string(),
                    serde_json::from_value(question).expect("question parses"),
                )
            })
            .collect(),
        }
    }

    fn wire_answers() -> Value {
        serde_json::json!({
            "category": {
                "type": "choice",
                "choice": "success",
                "probabilities": { "success": 0.9, "failure": 0.1 },
                "confidence": 0.8
            },
            "satisfaction": { "type": "score", "score": 2, "confidence": 0.7 },
            "approved": { "type": "noul", "noul": 0.95 }
        })
    }

    #[derive(Debug, Clone)]
    struct RecordedRequest {
        url: String,
        headers: Vec<(String, String)>,
        body: Value,
        signal: Option<AbortSignal>,
    }

    type Recording = Arc<Mutex<Vec<RecordedRequest>>>;

    fn record(request: &HttpRequest) -> RecordedRequest {
        RecordedRequest {
            url: request.url.clone(),
            headers: request.headers.clone(),
            body: match &request.body {
                HttpBody::Bytes(bytes) => {
                    serde_json::from_slice(bytes).expect("recorded body is JSON")
                }
                other => panic!("unexpected request body {other:?}"),
            },
            signal: request.signal.clone(),
        }
    }

    fn json_response(body: &Value) -> HttpResponse {
        HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: HttpBody::bytes(serde_json::to_vec(body).expect("response serializes")),
        }
    }

    /// Fetch fake that records requests and always answers with `body`.
    fn recording_json_fetch(recording: Recording, body: Value) -> FetchFn {
        FetchFn::new(move |request| {
            recording.lock().push(record(&request));
            let response = json_response(&body);
            Box::pin(async move { Ok(response) })
        })
    }

    fn base_options() -> ClassifierOptions {
        ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                ..Default::default()
            },
            temperature: None,
        }
    }

    fn assert_choice_success(answer: &ClassifierAnswer) {
        match answer {
            ClassifierAnswer::Choice { choice, .. } => assert_eq!(choice, "success"),
            other => panic!("expected choice answer, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_public_bool_questions_and_answers_to_typesafe_noul_values() {
        let model = make_model();
        let context = make_context();
        let recording: Recording = Default::default();
        let fetch = recording_json_fetch(
            recording.clone(),
            serde_json::json!({ "answers": wire_answers() }),
        );

        let options = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(fetch),
                ..Default::default()
            },
            temperature: Some(1.5),
        };
        let result = classify_system_one(&TRANSPORT, &model, &context, Some(&options)).await;

        let requests = recording.lock();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.url, "https://api.typesafe.ai/v1/systemone");
        assert_eq!(
            request.body.get("model").and_then(Value::as_str),
            Some("jev-latest")
        );
        assert_eq!(
            request
                .body
                .pointer("/questions/category/type")
                .and_then(Value::as_str),
            Some("choice")
        );
        assert_eq!(
            request
                .body
                .pointer("/questions/satisfaction/type")
                .and_then(Value::as_str),
            Some("score")
        );
        assert_eq!(
            request
                .body
                .pointer("/questions/approved/type")
                .and_then(Value::as_str),
            Some("noul")
        );
        // System One has no temperature field; the option is ignored.
        assert!(request.body.get("temperature").is_none());
        assert_eq!(
            request
                .headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
                .map(|(_, value)| value.as_str()),
            Some("Bearer secret")
        );

        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        assert_eq!(
            result.answers.get("approved"),
            Some(&ClassifierAnswer::Bool { probability: 0.95 })
        );
        assert_choice_success(result.answers.get("category").expect("category answer"));
        assert_eq!(
            result.answers.get("satisfaction"),
            Some(&ClassifierAnswer::Score {
                score: 2.0,
                confidence: 0.7
            })
        );
        assert!(result.usage.is_none());

        // Priced from the model catalog like chat usage.
        let mut priced_model = make_model();
        priced_model.base.cost.rates.input = 0.042;
        let priced_fetch = recording_json_fetch(
            Default::default(),
            serde_json::json!({ "answers": wire_answers(), "usage": { "input_tokens": 308, "output_tokens": 23 } }),
        );
        let priced_options = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(priced_fetch),
                ..Default::default()
            },
            temperature: None,
        };
        let priced =
            classify_system_one(&TRANSPORT, &priced_model, &context, Some(&priced_options)).await;
        let usage = priced.usage.expect("usage present");
        assert_eq!(usage.input, 308);
        assert_eq!(usage.output, 23);
        assert_eq!(usage.total_tokens, 331);
        assert!(
            (usage.cost.total - 0.000012936).abs() < 1e-12,
            "unexpected cost {}",
            usage.cost.total
        );
    }

    #[tokio::test]
    async fn posts_openrouter_system_one_requests_to_its_typesafe_compatible_endpoint() {
        let context = make_context();
        let recording: Recording = Default::default();
        let fetch = recording_json_fetch(
            recording.clone(),
            serde_json::json!({
                "id": "gen-dec-1",
                "provider": "TypeSafe",
                "answers": wire_answers(),
                "usage": { "input_tokens": 308, "output_tokens": 23, "cost": 0.000012936 }
            }),
        );
        let mut open_router_model = make_model();
        open_router_model.base.id = "typesafe/jev-1.13".to_string();
        open_router_model.base.provider = "openrouter".to_string();
        open_router_model.base.base_url = "https://openrouter.ai/api/v1".to_string();
        open_router_model.base.cost.rates.input = 0.042;

        let options = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(fetch),
                ..Default::default()
            },
            temperature: None,
        };
        let result =
            classify_system_one(&TRANSPORT, &open_router_model, &context, Some(&options)).await;

        let requests = recording.lock();
        assert_eq!(requests[0].url, "https://openrouter.ai/api/v1/systemone");
        assert_eq!(
            requests[0].body.get("model").and_then(Value::as_str),
            Some("typesafe/jev-1.13")
        );
        assert_eq!(
            requests[0].body.get("state"),
            Some(&context.state.clone().into())
        );
        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        assert_eq!(
            result.answers.get("approved"),
            Some(&ClassifierAnswer::Bool { probability: 0.95 })
        );
        // Priced from the catalog like chat usage; matches OpenRouter's reported cost.
        let usage = result.usage.expect("usage present");
        assert!(
            (usage.cost.total - 0.000012936).abs() < 1e-12,
            "unexpected cost {}",
            usage.cost.total
        );
    }

    #[tokio::test]
    async fn rejects_models_for_other_classifier_apis() {
        let mut model = make_model();
        model.base.api = "cloudflare-workers-ai-system-one".to_string();
        let recording: Recording = Default::default();
        let fetch = recording_json_fetch(
            recording.clone(),
            serde_json::json!({ "answers": wire_answers() }),
        );
        let options = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(fetch),
                ..Default::default()
            },
            temperature: None,
        };

        let result = classify_system_one(&TRANSPORT, &model, &make_context(), Some(&options)).await;

        assert!(recording.lock().is_empty());
        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        let error_message = result.error_message.expect("error message");
        assert!(
            error_message.contains("Unsupported classifier API: cloudflare-workers-ai-system-one"),
            "unexpected message {error_message}"
        );
    }

    #[tokio::test]
    async fn rejects_image_input_before_sending() {
        let mut model = make_model();
        model.base.input = vec![
            crate::types::ModelInput::Text,
            crate::types::ModelInput::Image,
        ];
        let recording: Recording = Default::default();
        let fetch = recording_json_fetch(
            recording.clone(),
            serde_json::json!({ "answers": wire_answers() }),
        );
        let options = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(fetch),
                ..Default::default()
            },
            temperature: None,
        };
        let mut context = make_context();
        context.images = Some(vec![crate::types::ImageContent::new(
            "aW1hZ2U=",
            "image/png",
        )]);

        let result = classify_system_one(&TRANSPORT, &model, &context, Some(&options)).await;

        assert!(recording.lock().is_empty());
        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("System One API does not support image input"),
            "unexpected message {:?}",
            result.error_message
        );
    }

    #[tokio::test]
    async fn merges_headers_case_insensitively_and_supports_null_suppression() {
        let recording: Recording = Default::default();
        let fetch = recording_json_fetch(
            recording.clone(),
            serde_json::json!({ "answers": wire_answers() }),
        );
        let mut model_with_headers = make_model();
        model_with_headers.base.headers = Some(
            [
                ("authorization".to_string(), "Bearer model".to_string()),
                ("X-Source".to_string(), "model".to_string()),
            ]
            .into_iter()
            .collect(),
        );

        let with_request_headers = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(fetch.clone()),
                headers: Some(
                    [
                        (
                            "Authorization".to_string(),
                            Some("Bearer request".to_string()),
                        ),
                        ("x-source".to_string(), Some("request".to_string())),
                    ]
                    .into_iter()
                    .collect(),
                ),
                ..Default::default()
            },
            temperature: None,
        };
        classify_system_one(
            &TRANSPORT,
            &model_with_headers,
            &make_context(),
            Some(&with_request_headers),
        )
        .await;

        let with_null_header = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(fetch),
                headers: Some([("Authorization".to_string(), None)].into_iter().collect()),
                ..Default::default()
            },
            temperature: None,
        };
        classify_system_one(
            &TRANSPORT,
            &model_with_headers,
            &make_context(),
            Some(&with_null_header),
        )
        .await;

        let requests = recording.lock();
        assert_eq!(requests.len(), 2);
        let first = &requests[0].headers;
        assert_eq!(
            first
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
                .map(|(_, value)| value.as_str()),
            Some("Bearer request")
        );
        assert_eq!(
            first
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("x-source"))
                .map(|(_, value)| value.as_str()),
            Some("request")
        );
        assert_eq!(
            first
                .iter()
                .filter(|(name, _)| name.eq_ignore_ascii_case("authorization"))
                .count(),
            1
        );
        let second = &requests[1].headers;
        assert!(
            !second
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("authorization")),
            "authorization must be suppressed: {second:?}"
        );
    }

    #[tokio::test]
    async fn preserves_prototype_sensitive_question_ids_in_answers() {
        let model = make_model();
        let prototype_context: ClassifierContext = serde_json::from_str(
			r#"{"state":{},"questions":{"__proto__":{"type":"bool","instructions":"Is this true?","criteria":{"true":"Yes","false":"No"}}}}"#,
		)
		.expect("context parses");
        let fetch = recording_json_fetch(
            Default::default(),
            serde_json::from_str(r#"{"answers":{"__proto__":{"type":"noul","noul":0.75}}}"#)
                .expect("answers parse"),
        );
        let options = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(fetch),
                ..Default::default()
            },
            temperature: None,
        };

        let result =
            classify_system_one(&TRANSPORT, &model, &prototype_context, Some(&options)).await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        let keys: Vec<&String> = result.answers.keys().collect();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0], "__proto__");
        assert_eq!(
            result.answers.get("__proto__"),
            Some(&ClassifierAnswer::Bool { probability: 0.75 })
        );
        // Serialized answers keep the id as a normal key.
        let serialized = serde_json::to_value(&result.answers).expect("answers serialize");
        assert_eq!(
            serialized.get("__proto__"),
            Some(&serde_json::json!({ "type": "bool", "probability": 0.75 }))
        );
    }

    #[tokio::test]
    async fn reports_request_timeouts_separately_from_caller_cancellation() {
        let model = make_model();
        let hang_until_aborted: FetchFn = FetchFn::new(|request| {
            Box::pin(async move {
                let Some(signal) = request.signal else {
                    return Err(ProviderHttpError::new("missing request signal"));
                };
                let _ = signal.aborted_future().await;
                Err(ProviderHttpError::new("The operation was aborted"))
            })
        });
        let options = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(hang_until_aborted),
                timeout_ms: Some(5),
                max_retries: Some(0),
                ..Default::default()
            },
            temperature: None,
        };

        let result = classify_system_one(&TRANSPORT, &model, &make_context(), Some(&options)).await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("Request timed out after 5ms")
        );
    }

    #[tokio::test]
    async fn creates_a_fresh_timeout_for_every_retry_attempt() {
        let model = make_model();
        let signals: Arc<Mutex<Vec<Option<AbortSignal>>>> = Default::default();
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let attempts_for_fetch = attempts.clone();
        let signals_for_fetch = signals.clone();
        let fetch = FetchFn::new(move |request| {
            let attempt = attempts_for_fetch.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            signals_for_fetch.lock().push(request.signal.clone());
            let response = if attempt == 1 {
                HttpResponse {
                    status: 500,
                    headers: vec![("retry-after-ms".to_string(), "0".to_string())],
                    body: HttpBody::bytes(b"retry".to_vec()),
                }
            } else {
                json_response(&serde_json::json!({ "answers": wire_answers() }))
            };
            Box::pin(async move { Ok(response) })
        });
        let options = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(fetch),
                timeout_ms: Some(1000),
                max_retries: Some(1),
                ..Default::default()
            },
            temperature: None,
        };

        let result = classify_system_one(&TRANSPORT, &model, &make_context(), Some(&options)).await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
        let signals = signals.lock();
        assert_eq!(signals.len(), 2);
        // A fresh per-attempt timeout signal backs each request (upstream asserts
        // `signals[0] !== signals[1]`; signal identity is not observable in Rust,
        // so the fresh-guard property is enforced by construction in D25 and the
        // retry is verified behaviorally here).
        assert!(signals[0].is_some());
        assert!(signals[1].is_some());
        assert_eq!(
            result.answers.get("approved"),
            Some(&ClassifierAnswer::Bool { probability: 0.95 })
        );
    }

    #[tokio::test]
    async fn returns_malformed_responses_as_classifier_errors() {
        let model = make_model();
        let fetch = recording_json_fetch(
            Default::default(),
            serde_json::json!({ "answers": {}, "usage": { "input_tokens": 10, "output_tokens": 2 } }),
        );
        let options = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(fetch),
                ..Default::default()
            },
            temperature: None,
        };

        let result = classify_system_one(&TRANSPORT, &model, &make_context(), Some(&options)).await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        assert!(result.answers.is_empty());
        let error_message = result.error_message.expect("error message");
        // Upstream asserts "did not return an answer for category" (first
        // question in JS insertion order); the Rust questions map is a BTreeMap,
        // so the first iterated missing id is `approved`. The upstream property —
        // the error names the first unanswered question — is preserved.
        assert!(
            error_message.contains("did not return an answer for approved"),
            "unexpected message {error_message}"
        );
        // The request was billed, so its usage is kept.
        let usage = result.usage.expect("usage kept");
        assert_eq!((usage.input, usage.output), (10, 2));
    }

    #[tokio::test]
    async fn ignores_malformed_usage() {
        let model = make_model();
        let context = make_context();
        let malformed_tokens_fetch = recording_json_fetch(
            Default::default(),
            serde_json::json!({ "answers": wire_answers(), "usage": { "input_tokens": "many", "output_tokens": 3 } }),
        );
        let with_tokens = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(malformed_tokens_fetch),
                ..Default::default()
            },
            temperature: None,
        };
        let result = classify_system_one(&TRANSPORT, &model, &context, Some(&with_tokens)).await;

        let cost_only_fetch = recording_json_fetch(
            Default::default(),
            serde_json::json!({ "answers": wire_answers(), "usage": { "cost": 0.1 } }),
        );
        let without_tokens = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(cost_only_fetch),
                ..Default::default()
            },
            temperature: None,
        };
        let without =
            classify_system_one(&TRANSPORT, &model, &context, Some(&without_tokens)).await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        let usage = result.usage.expect("usage present");
        assert_eq!((usage.input, usage.output, usage.total_tokens), (0, 3, 3));
        assert_eq!(without.stop_reason, ClassifierStopReason::Stop);
        assert!(without.usage.is_none());
    }

    #[tokio::test]
    async fn typesafe_system_one_api_dispatches_through_the_classifier_trait() {
        // The exported entry point serves the same transport.
        let api = typesafe_system_one_api();
        let mut model = make_model();
        model.base.base_url = "https://openrouter.ai/api/v1".to_string();
        let fetch = recording_json_fetch(
            Default::default(),
            serde_json::json!({ "answers": wire_answers() }),
        );
        let options = ClassifierOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(fetch),
                ..Default::default()
            },
            temperature: None,
        };

        let result = api.classify(&model, &make_context(), &options).await;
        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        assert_eq!(
            result.answers.get("approved"),
            Some(&ClassifierAnswer::Bool { probability: 0.95 })
        );
    }
}
