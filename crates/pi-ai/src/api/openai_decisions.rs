//! Port of `packages/ai/src/api/openai-decisions.ts` — classification through
//! OpenAI's Decisions API (`POST /v1/decisions` with `{ model, input,
//! questions }`).
//!
//! https://developers.openai.com/api/docs/guides/decisions
//!
//! The state is sent as JSON text. With images, the input becomes one user
//! message with the state as `input_text` followed by `input_image` data
//! URLs. Questions map to Decisions types: `choice` to `choice`, `score` to
//! `score`, and `bool` to `predicate`. Predicates have no criteria field, so
//! the meanings of true and false are appended to the instructions.
//!
//! Only OpenAI API keys work: Sign in with ChatGPT tokens are rejected on
//! this route.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::http::ProviderHttpError;
use crate::types::{
    ClassifierAnswer, ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierQuestion,
    ClassifierResult, ClassifierStopReason,
};
use crate::utils::diagnostics::now_ms;
use crate::utils::error_body::{format_provider_error, normalize_provider_error};

use super::classifier_shared::{
    is_record, parse_classifier_usage, post_classifier_request, required_number,
};

const LABEL: &str = "OpenAI Decisions";

/// The endpoint accepts at most this many image parts per request.
const MAX_IMAGES: usize = 128;

/// `predicateInstructions`: predicate questions have no criteria field, so
/// the meanings of true and false are appended to the instructions.
fn predicate_instructions(
    question: &crate::types::ClassifierBoolCriteria,
    instructions: &str,
) -> String {
    let true_label = question.true_label.as_str();
    let false_label = question.false_label.as_str();
    let mut meanings = Vec::new();
    if !true_label.is_empty() {
        meanings.push(format!("True means: {true_label}"));
    }
    if !false_label.is_empty() {
        meanings.push(format!("False means: {false_label}"));
    }
    if meanings.is_empty() {
        instructions.to_string()
    } else {
        format!("{instructions}\n\n{}", meanings.join("\n"))
    }
}

/// `wireQuestion`: maps a public question to the Decisions wire shape.
fn wire_question(name: &str, question: &ClassifierQuestion) -> Value {
    match question {
        ClassifierQuestion::Choice {
            instructions,
            criteria,
        } => {
            let choices: Vec<Value> = criteria
                .iter()
                .map(|(value, description)| {
                    if description.is_empty() {
                        json!({ "value": value })
                    } else {
                        json!({ "value": value, "description": description })
                    }
                })
                .collect();
            json!({ "type": "choice", "name": name, "instructions": instructions, "choices": choices })
        }
        ClassifierQuestion::Score {
            instructions,
            criteria,
        } => {
            let levels: Vec<Value> = criteria
                .iter()
                .map(|label| json!({ "label": label }))
                .collect();
            json!({ "type": "score", "name": name, "instructions": instructions, "levels": levels })
        }
        ClassifierQuestion::Bool {
            instructions,
            criteria,
        } => {
            json!({
                "type": "predicate",
                "name": name,
                "instructions": predicate_instructions(criteria, instructions),
            })
        }
    }
}

/// `wireInput`: the state as JSON text; with images, one user message with the
/// state followed by the images as data URLs.
fn wire_input(context: &ClassifierContext) -> Result<Value, ProviderHttpError> {
    let state = serde_json::to_string(&context.state).map_err(|error| {
        ProviderHttpError::with_source(
            "state does not serialize".to_string(),
            std::sync::Arc::new(error),
        )
    })?;
    let images = context.images.as_deref().unwrap_or_default();
    if images.is_empty() {
        return Ok(Value::String(state));
    }
    if images.len() > MAX_IMAGES {
        return Err(ProviderHttpError::new(format!(
            "{LABEL} accepts at most {MAX_IMAGES} images, got {}",
            images.len()
        )));
    }
    let mut content = vec![json!({ "type": "input_text", "text": state })];
    for image in images {
        content.push(json!({
            "type": "input_image",
            "image_url": format!("data:{};base64,{}", image.mime_type, image.data),
        }));
    }
    Ok(json!([{ "role": "user", "content": content }]))
}

/// `choiceProbabilities`.
fn choice_probabilities(
    value: Option<&Value>,
    id: &str,
) -> Result<BTreeMap<String, f64>, ProviderHttpError> {
    let Some(array) = value.and_then(Value::as_array) else {
        return Err(ProviderHttpError::new(format!(
            "{LABEL} returned invalid probabilities for {id}"
        )));
    };
    let mut probabilities = BTreeMap::new();
    for entry in array {
        let Some(object) = entry.as_object() else {
            return Err(ProviderHttpError::new(format!(
                "{LABEL} returned invalid probabilities for {id}"
            )));
        };
        let Some(value) = object.get("value").and_then(Value::as_str) else {
            return Err(ProviderHttpError::new(format!(
                "{LABEL} returned invalid probabilities for {id}"
            )));
        };
        let probability = required_number(
            LABEL,
            object.get("probability"),
            &format!("probability for {id}.{value}"),
        )?;
        probabilities.insert(value.to_string(), probability);
    }
    Ok(probabilities)
}

/// `parseAnswer`: one Decisions answer, matched against its question.
fn parse_answer(
    id: &str,
    question: &ClassifierQuestion,
    answer: &Value,
) -> Result<ClassifierAnswer, ProviderHttpError> {
    let answer_type = answer.get("type").and_then(Value::as_str);
    if answer_type == Some("refusal") {
        return Err(ProviderHttpError::new(format!(
            "{LABEL} refused to answer {id}"
        )));
    }
    match question {
        ClassifierQuestion::Choice { .. } => {
            let choice = answer.get("choice").and_then(Value::as_str);
            if answer_type != Some("choice") || choice.is_none() {
                return Err(ProviderHttpError::new(format!(
                    "{LABEL} did not return a choice answer for {id}"
                )));
            }
            Ok(ClassifierAnswer::Choice {
                choice: choice.expect("checked above").to_string(),
                probabilities: choice_probabilities(answer.get("probabilities"), id)?,
                confidence: required_number(
                    LABEL,
                    answer.get("confidence"),
                    &format!("confidence for {id}"),
                )?,
            })
        }
        ClassifierQuestion::Score { .. } => {
            if answer_type != Some("score") {
                return Err(ProviderHttpError::new(format!(
                    "{LABEL} did not return a score answer for {id}"
                )));
            }
            Ok(ClassifierAnswer::Score {
                score: required_number(LABEL, answer.get("score"), &format!("score for {id}"))?,
                confidence: required_number(
                    LABEL,
                    answer.get("confidence"),
                    &format!("confidence for {id}"),
                )?,
            })
        }
        ClassifierQuestion::Bool { .. } => {
            if answer_type != Some("predicate") {
                return Err(ProviderHttpError::new(format!(
                    "{LABEL} did not return a predicate answer for {id}"
                )));
            }
            Ok(ClassifierAnswer::Bool {
                probability: required_number(
                    LABEL,
                    answer.get("probability"),
                    &format!("probability for {id}"),
                )?,
            })
        }
    }
}

/// `parseAnswers`: answers come back by name, in any order.
fn parse_answers(
    value: Option<&Value>,
    context: &ClassifierContext,
) -> Result<BTreeMap<String, ClassifierAnswer>, ProviderHttpError> {
    let Some(array) = value.and_then(Value::as_array) else {
        return Err(ProviderHttpError::new(format!(
            "{LABEL} returned an unexpected response"
        )));
    };
    let mut by_name = BTreeMap::new();
    for answer in array {
        if let Some(name) = answer.get("name").and_then(Value::as_str) {
            by_name.insert(name.to_string(), answer);
        }
    }
    let mut answers = BTreeMap::new();
    for (id, question) in &context.questions {
        let Some(answer) = by_name.get(id) else {
            return Err(ProviderHttpError::new(format!(
                "{LABEL} did not return an answer for {id}"
            )));
        };
        answers.insert(id.clone(), parse_answer(id, question, answer)?);
    }
    Ok(answers)
}

/// Cloudflare in front of api.openai.com answers 504 with an HTML page when a
/// request runs longer than about five seconds. Large inputs, currently above
/// roughly 600K tokens, hit this limit, and retrying the same input runs into
/// it again, so 504 is not retried.
const NO_RETRY_STATUSES: &[u16] = &[504];

fn error_message(error: &ProviderHttpError) -> String {
    if error.status == Some(504) {
        return format!(
			"{LABEL} error (504): the request timed out at the gateway. Very large inputs (above roughly 600K tokens) currently exceed its time limit."
		);
    }
    format_provider_error(
        &normalize_provider_error(error),
        Some(&format!("{LABEL} error")),
    )
}

/// Classifies with OpenAI's Decisions API (`classify`).
pub async fn classify(
    model: &ClassifierModel,
    context: &ClassifierContext,
    options: Option<&ClassifierOptions>,
) -> ClassifierResult {
    let mut output = ClassifierResult {
        api: model.api().to_string(),
        provider: model.provider().to_string(),
        model: model.id().to_string(),
        answers: BTreeMap::new(),
        usage: None,
        stop_reason: ClassifierStopReason::Stop,
        error_message: None,
        timestamp: now_ms(),
    };

    let result = classify_inner(model, context, options, &mut output).await;
    if let Err(error) = result {
        let caller_aborted = options
            .and_then(|options| options.base.signal.as_ref())
            .map(|signal| signal.aborted())
            .unwrap_or(false);
        output.answers = BTreeMap::new();
        output.stop_reason = if caller_aborted {
            ClassifierStopReason::Aborted
        } else {
            ClassifierStopReason::Error
        };
        output.error_message = Some(error_message(&error));
    }
    output
}

async fn classify_inner(
    model: &ClassifierModel,
    context: &ClassifierContext,
    options: Option<&ClassifierOptions>,
    output: &mut ClassifierResult,
) -> Result<(), ProviderHttpError> {
    if model.api() != "openai-decisions" {
        return Err(ProviderHttpError::new(format!(
            "Unsupported classifier API: {}",
            model.api()
        )));
    }
    let url = format!("{}/decisions", model.base.base_url.trim_end_matches('/'));
    let input = wire_input(context)?;
    let questions: Vec<Value> = context
        .questions
        .iter()
        .map(|(id, question)| wire_question(id, question))
        .collect();
    let body = post_classifier_request(
        LABEL,
        url,
        model,
        json!({ "model": model.id(), "input": input, "questions": questions }),
        options,
        Some(NO_RETRY_STATUSES),
    )
    .await?;
    if !is_record(&body) {
        return Err(ProviderHttpError::new(format!(
            "{LABEL} returned an unexpected response"
        )));
    }
    // Set before parsing answers: a request with malformed or refused answers
    // was still billed.
    if let Some(usage) = parse_classifier_usage(body.get("usage"), model) {
        output.usage = Some(usage);
    }
    output.answers = parse_answers(body.get("answers"), context)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // Port of test/openai-decisions.test.ts (fixture data and cases 1:1).
    use super::*;
    use crate::http::{FetchFn, HttpBody, HttpRequest, HttpResponse};
    use crate::types::{
        BaseModelFields, ClassifierBoolCriteria, ClassifierTypeTag, ImageContent, JsonObject,
        ModelCost, ModelCostRates, ModelCostTier, ModelInput, ProviderRequestOptions,
    };
    use parking_lot::Mutex;
    use std::sync::Arc;

    fn make_model() -> ClassifierModel {
        ClassifierModel {
            base: BaseModelFields {
                id: "gpt-6-luna".to_string(),
                name: "GPT-6 Luna".to_string(),
                api: "openai-decisions".to_string(),
                provider: "openai".to_string(),
                base_url: "https://api.openai.com/v1".to_string(),
                input: vec![ModelInput::Text, ModelInput::Image],
                input_limits: None,
                cost: ModelCost {
                    rates: ModelCostRates {
                        input: 0.1,
                        output: 0.0,
                        cache_read: 0.0,
                        cache_write: 0.0,
                    },
                    tiers: Some(vec![ModelCostTier {
                        rates: ModelCostRates {
                            input: 0.2,
                            output: 0.0,
                            cache_read: 0.0,
                            cache_write: 0.0,
                        },
                        input_tokens_above: 272_000.0,
                    }]),
                },
                headers: None,
            },
            model_type: ClassifierTypeTag::Classifier,
            context_window: 922_000,
        }
    }

    fn make_context() -> ClassifierContext {
        let mut questions = BTreeMap::new();
        questions.insert(
            "category".to_string(),
            ClassifierQuestion::Choice {
                instructions: "Classify the message".to_string(),
                criteria: [
                    ("success".to_string(), "Successful".to_string()),
                    ("failure".to_string(), String::new()),
                ]
                .into_iter()
                .collect(),
            },
        );
        questions.insert(
            "satisfaction".to_string(),
            ClassifierQuestion::Score {
                instructions: "Score satisfaction".to_string(),
                criteria: vec!["low".to_string(), "neutral".to_string(), "high".to_string()],
            },
        );
        questions.insert(
            "approved".to_string(),
            ClassifierQuestion::Bool {
                instructions: "Does the user approve?".to_string(),
                criteria: ClassifierBoolCriteria {
                    true_label: "Approval".to_string(),
                    false_label: "No approval".to_string(),
                },
            },
        );
        let state: JsonObject =
            serde_json::from_value(json!({ "text": "The deployment succeeded, thank you." }))
                .expect("state parses");
        ClassifierContext {
            state,
            images: None,
            questions,
        }
    }

    /// Response shape from the API reference and live `gpt-6-luna` requests.
    fn wire_answers() -> Value {
        json!([
            {
                "type": "choice",
                "name": "category",
                "choice": "success",
                "probabilities": [
                    { "value": "success", "probability": 0.9 },
                    { "value": "failure", "probability": 0.1 },
                ],
                "confidence": 0.8,
            },
            {
                "type": "score",
                "name": "satisfaction",
                "score": 1.8,
                "probabilities": [
                    { "value": 0, "label": "low", "probability": 0.05 },
                    { "value": 1, "label": "neutral", "probability": 0.1 },
                    { "value": 2, "label": "high", "probability": 0.85 },
                ],
                "confidence": 0.7,
            },
            { "type": "predicate", "name": "approved", "probability": 0.95 },
        ])
    }

    fn wire_usage() -> Value {
        json!({
            "input_tokens": 164,
            "input_tokens_details": { "cached_tokens": 0, "cache_write_tokens": 0 },
            "output_tokens": 0,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": 164,
        })
    }

    fn image() -> ImageContent {
        ImageContent::new("aW1hZ2U=", "image/png")
    }

    fn json_response(status: u16, body: &Value) -> HttpResponse {
        HttpResponse {
            status,
            headers: Vec::new(),
            body: HttpBody::bytes(serde_json::to_vec(body).expect("response serializes")),
        }
    }

    fn raw_response(status: u16, headers: Vec<(String, String)>, body: &[u8]) -> HttpResponse {
        HttpResponse {
            status,
            headers,
            body: HttpBody::bytes(body.to_vec()),
        }
    }

    #[derive(Clone)]
    struct Recorded {
        url: String,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    }

    fn record(request: &HttpRequest) -> Recorded {
        Recorded {
            url: request.url.clone(),
            headers: request.headers.clone(),
            body: match &request.body {
                HttpBody::Bytes(bytes) => bytes.clone(),
                other => panic!("unexpected request body: {other:?}"),
            },
        }
    }

    type Recording = Arc<Mutex<Vec<Recorded>>>;

    fn queue_fetch(recording: Recording, responses: Vec<HttpResponse>) -> FetchFn {
        let responses = Arc::new(Mutex::new(responses));
        FetchFn::new(move |request| {
            recording.lock().push(record(&request));
            let response = responses.lock().pop().expect("queued response");
            Box::pin(async move { Ok(response) })
        })
    }

    fn options_with_fetch(fetch: FetchFn, max_retries: Option<u32>) -> ClassifierOptions {
        ClassifierOptions {
            base: ProviderRequestOptions {
                api_key: Some("secret".to_string()),
                fetch: Some(fetch),
                max_retries,
                ..Default::default()
            },
            temperature: None,
        }
    }

    #[tokio::test]
    async fn maps_questions_to_decisions_types_and_answers_back_by_name() {
        let recording: Recording = Default::default();
        let reversed: Vec<Value> = wire_answers()
            .as_array()
            .expect("array")
            .iter()
            .rev()
            .cloned()
            .collect();
        let fetch = queue_fetch(
            recording.clone(),
            vec![json_response(
                200,
                &json!({ "model": "gpt-6-luna", "answers": reversed, "usage": wire_usage() }),
            )],
        );
        let model = make_model();
        let context = make_context();
        let result = classify(&model, &context, Some(&options_with_fetch(fetch, None))).await;

        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        assert_eq!(
            result.answers.get("category"),
            Some(&ClassifierAnswer::Choice {
                choice: "success".to_string(),
                probabilities: [("success".to_string(), 0.9), ("failure".to_string(), 0.1)]
                    .into_iter()
                    .collect(),
                confidence: 0.8,
            })
        );
        assert_eq!(
            result.answers.get("satisfaction"),
            Some(&ClassifierAnswer::Score {
                score: 1.8,
                confidence: 0.7
            })
        );
        assert_eq!(
            result.answers.get("approved"),
            Some(&ClassifierAnswer::Bool { probability: 0.95 })
        );

        assert_eq!(recording.lock().len(), 1);
        let recorded = &recording.lock()[0];
        assert_eq!(recorded.url, "https://api.openai.com/v1/decisions");
        let authorization = recorded
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
            .map(|(_, value)| value.clone())
            .expect("authorization header");
        assert_eq!(authorization, "Bearer secret");
        let sent: Value = serde_json::from_slice(&recorded.body).expect("json");
        assert_eq!(sent["model"], "gpt-6-luna");
        // The state goes out as JSON text; the choice criteria order follows
        // the sorted map (Rust BTreeMap deviation) and empty descriptions are
        // omitted.
        assert_eq!(
            sent["input"],
            serde_json::to_string(&context.state)
                .expect("state")
                .as_str()
        );
        let questions = sent["questions"].as_array().expect("questions");
        let category = questions
            .iter()
            .find(|question| question.get("name").and_then(Value::as_str) == Some("category"))
            .expect("category question");
        let satisfaction = questions
            .iter()
            .find(|question| question.get("name").and_then(Value::as_str) == Some("satisfaction"))
            .expect("satisfaction question");
        let approved = questions
            .iter()
            .find(|question| question.get("name").and_then(Value::as_str) == Some("approved"))
            .expect("approved question");
        assert_eq!(
            category["choices"],
            json!([{ "value": "failure" }, { "value": "success", "description": "Successful" }])
        );
        assert_eq!(
            satisfaction["levels"],
            json!([{ "label": "low" }, { "label": "neutral" }, { "label": "high" }])
        );
        assert_eq!(
            approved["instructions"],
            "Does the user approve?\n\nTrue means: Approval\nFalse means: No approval"
        );
        let usage = result.usage.expect("usage");
        assert_eq!(
            (
                usage.input,
                usage.output,
                usage.cache_read,
                usage.total_tokens
            ),
            (164, 0, 0, 164)
        );
        assert!((usage.cost.total - 0.0000164).abs() < 1e-12);
    }

    #[tokio::test]
    async fn prices_long_context_requests_at_the_long_context_input_rate() {
        let recording: Recording = Default::default();
        let fetch = queue_fetch(
            recording,
            vec![json_response(
                200,
                &json!({ "answers": wire_answers(), "usage": { "input_tokens": 300000, "output_tokens": 0 } }),
            )],
        );
        let model = make_model();
        let context = make_context();
        let result = classify(&model, &context, Some(&options_with_fetch(fetch, None))).await;
        let usage = result.usage.expect("usage");
        assert!((usage.cost.total - 0.06).abs() < 1e-12);
    }

    #[tokio::test]
    async fn sends_images_after_the_state_in_one_user_message() {
        let recording: Recording = Default::default();
        let fetch = queue_fetch(
            recording.clone(),
            vec![json_response(200, &json!({ "answers": wire_answers() }))],
        );
        let model = make_model();
        let mut context = make_context();
        context.images = Some(vec![image(), ImageContent::new("aW1hZ2U=", "image/jpeg")]);
        let result = classify(&model, &context, Some(&options_with_fetch(fetch, None))).await;
        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        let sent: Value = serde_json::from_slice(&recording.lock()[0].body).expect("json");
        assert_eq!(
            sent["input"],
            json!([
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": serde_json::to_string(&context.state).expect("state") },
                        { "type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U=" },
                        { "type": "input_image", "image_url": "data:image/jpeg;base64,aW1hZ2U=" },
                    ],
                },
            ])
        );
    }

    #[tokio::test]
    async fn rejects_more_than_max_images_before_sending() {
        let recording: Recording = Default::default();
        let fetch = queue_fetch(
            recording.clone(),
            vec![json_response(200, &json!({ "answers": wire_answers() }))],
        );
        let model = make_model();
        let mut context = make_context();
        context.images = Some(vec![image(); MAX_IMAGES + 1]);
        let result = classify(&model, &context, Some(&options_with_fetch(fetch, None))).await;
        assert!(recording.lock().is_empty());
        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        let message = result.error_message.expect("error message");
        assert!(message.contains(&format!(
            "at most {MAX_IMAGES} images, got {}",
            MAX_IMAGES + 1
        )));
    }

    #[tokio::test]
    async fn fails_the_result_when_a_question_is_refused_and_keeps_the_billed_usage() {
        let recording: Recording = Default::default();
        let answers = json!({
            "answers": [
                wire_answers()[0].clone(),
                wire_answers()[1].clone(),
                { "type": "refusal", "name": "approved" },
            ],
            "usage": wire_usage(),
        });
        let fetch = queue_fetch(recording, vec![json_response(200, &answers)]);
        let model = make_model();
        let context = make_context();
        let result = classify(&model, &context, Some(&options_with_fetch(fetch, None))).await;
        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        assert!(result.answers.is_empty());
        assert_eq!(
            result.error_message.as_deref(),
            Some("OpenAI Decisions refused to answer approved")
        );
        assert_eq!(result.usage.as_ref().map(|usage| usage.input), Some(164));
    }

    #[tokio::test]
    async fn returns_missing_and_mistyped_answers_as_classifier_errors() {
        let missing = json!({ "answers": [wire_answers()[0].clone(), wire_answers()[1].clone()] });
        let mistyped = json!({ "answers": [
			wire_answers()[0].clone(),
			wire_answers()[1].clone(),
			{ "type": "score", "name": "approved" },
		] });
        let model = make_model();
        let context = make_context();

        let recording: Recording = Default::default();
        let fetch = queue_fetch(recording, vec![json_response(200, &missing)]);
        let missing = classify(&model, &context, Some(&options_with_fetch(fetch, None))).await;
        let recording: Recording = Default::default();
        let fetch = queue_fetch(recording, vec![json_response(200, &mistyped)]);
        let mistyped = classify(&model, &context, Some(&options_with_fetch(fetch, None))).await;

        assert_eq!(missing.stop_reason, ClassifierStopReason::Error);
        assert!(missing
            .error_message
            .expect("message")
            .contains("did not return an answer for approved"));
        assert_eq!(mistyped.stop_reason, ClassifierStopReason::Error);
        assert!(mistyped
            .error_message
            .expect("message")
            .contains("did not return a predicate answer for approved"));
    }

    #[tokio::test]
    async fn preserves_unusual_question_ids_in_answers() {
        let recording: Recording = Default::default();
        let fetch = queue_fetch(
            recording,
            vec![json_response(
                200,
                &json!({ "answers": [{ "type": "predicate", "name": "__proto__", "probability": 0.75 }] }),
            )],
        );
        let model = make_model();
        let context: ClassifierContext = serde_json::from_value(json!({
			"state": {},
			"questions": {
				"__proto__": { "type": "bool", "instructions": "Is this true?", "criteria": { "true": "Yes", "false": "No" } }
			}
		}))
		.expect("context parses");
        let result = classify(&model, &context, Some(&options_with_fetch(fetch, None))).await;
        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
        assert_eq!(
            result.answers.get("__proto__"),
            Some(&ClassifierAnswer::Bool { probability: 0.75 })
        );
    }

    #[tokio::test]
    async fn does_not_retry_gateway_timeouts_and_explains_them_instead_of_returning_the_html_page()
    {
        let recording: Recording = Default::default();
        // Default retries: the same input would time out again.
        let fetch = queue_fetch(
            recording.clone(),
            vec![raw_response(
                504,
                vec![("retry-after-ms".to_string(), "0".to_string())],
                b"<!DOCTYPE html><html>Gateway time-out</html>",
            )],
        );
        let model = make_model();
        let context = make_context();
        let result = classify(&model, &context, Some(&options_with_fetch(fetch, None))).await;
        assert_eq!(recording.lock().len(), 1);
        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        let message = result.error_message.expect("message");
        assert!(
            message.contains("OpenAI Decisions error (504): the request timed out at the gateway")
        );
        assert!(!message.contains("<html>"));
    }

    #[tokio::test]
    async fn still_retries_other_server_errors() {
        let recording: Recording = Default::default();
        let fetch = queue_fetch(
            recording.clone(),
            vec![
                json_response(200, &json!({ "answers": wire_answers() })),
                raw_response(
                    503,
                    vec![("retry-after-ms".to_string(), "0".to_string())],
                    b"busy",
                ),
            ],
        );
        let model = make_model();
        let context = make_context();
        let result = classify(&model, &context, Some(&options_with_fetch(fetch, None))).await;
        assert_eq!(recording.lock().len(), 2);
        assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
    }

    #[tokio::test]
    async fn includes_the_api_error_body_for_other_http_failures() {
        let recording: Recording = Default::default();
        let fetch = queue_fetch(
            recording,
            vec![json_response(
                400,
                &json!({ "error": { "message": "Decision input exceeds the token limit.", "type": "invalid_request_error" } }),
            )],
        );
        let model = make_model();
        let context = make_context();
        let result = classify(&model, &context, Some(&options_with_fetch(fetch, Some(0)))).await;
        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        let message = result.error_message.expect("message");
        assert!(message.contains("OpenAI Decisions error (400)"));
        assert!(message.contains("Decision input exceeds the token limit."));
    }

    #[tokio::test]
    async fn rejects_models_for_other_classifier_apis_and_missing_api_keys() {
        let model = make_model();
        let context = make_context();
        let mut other = make_model();
        other.base.api = "typesafe-system-one".to_string();

        let recording: Recording = Default::default();
        let fetch = queue_fetch(
            recording.clone(),
            vec![json_response(200, &json!({ "answers": wire_answers() }))],
        );
        let other_api = classify(&other, &context, Some(&options_with_fetch(fetch, None))).await;
        let recording: Recording = Default::default();
        let fetch = queue_fetch(
            recording.clone(),
            vec![json_response(200, &json!({ "answers": wire_answers() }))],
        );
        let without_key = ClassifierOptions {
            base: ProviderRequestOptions {
                fetch: Some(fetch),
                ..Default::default()
            },
            temperature: None,
        };
        let no_key = classify(&model, &context, Some(&without_key)).await;
        assert!(recording.lock().is_empty());
        assert!(other_api
            .error_message
            .expect("message")
            .contains("Unsupported classifier API: typesafe-system-one"));
        assert!(no_key
            .error_message
            .expect("message")
            .contains("No API key for provider: openai"));
    }
}
