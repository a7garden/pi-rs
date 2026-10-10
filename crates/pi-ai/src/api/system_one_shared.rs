//! Port of `packages/ai/src/api/system-one-shared.ts` — the shared client for
//! services serving System One classifier models (TypeSafe, OpenRouter).
//! The HTTP plumbing (bearer auth, per-attempt timeouts, retries, usage
//! parsing) lives in [`super::classifier_shared`] (upstream
//! `classifier-shared.ts`); its D24–D26 deviations apply here too.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::http::ProviderHttpError;
use crate::types::{
    ClassifierAnswer, ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierQuestion,
    ClassifierResult, ClassifierStopReason, JsonObject,
};
use crate::utils::diagnostics::now_ms;
use crate::utils::error_body::{format_provider_error, normalize_provider_error};

use super::classifier_shared::{
    classifier_model_view, is_record, parse_classifier_usage, post_classifier_request,
    required_number,
};

/// TypeSafe System One request body without the transport-specific envelope
/// (`SystemOneWireRequest`).
#[derive(Debug, Clone, PartialEq)]
pub struct SystemOneWireRequest {
    pub state: JsonObject,
    pub questions: JsonObject,
}

/// Differences between services that serve System One models
/// (`SystemOneTransport`).
pub trait SystemOneTransport: Send + Sync {
    /// Classifier API implemented by this transport.
    fn api(&self) -> &'static str;
    /// Service name used in error messages.
    fn label(&self) -> &'static str;
    /// Absolute request URL.
    fn url(&self, model: &ClassifierModel) -> String;
    /// Wraps the System One request in the service's request envelope.
    fn payload(&self, model: &ClassifierModel, request: &SystemOneWireRequest) -> Value;
    /// Extracts the System One output (`{ answers, usage }`) from the service's
    /// response envelope.
    fn output(&self, body: &Value) -> Result<JsonObject, ProviderHttpError>;
}

fn probabilities(
    label: &str,
    value: Option<&Value>,
    id: &str,
) -> Result<BTreeMap<String, f64>, ProviderHttpError> {
    let Some(map) = value.and_then(Value::as_object) else {
        return Err(ProviderHttpError::new(format!(
            "{label} returned invalid probabilities for {id}"
        )));
    };
    map.iter()
        .map(|(key, probability)| {
            required_number(
                label,
                Some(probability),
                &format!("probability for {id}.{key}"),
            )
            .map(|number| (key.clone(), number))
        })
        .collect()
}

fn parse_answers(
    label: &str,
    value: Option<&Value>,
    context: &ClassifierContext,
) -> Result<BTreeMap<String, ClassifierAnswer>, ProviderHttpError> {
    let Some(value) = value else {
        return Err(ProviderHttpError::new(format!(
            "{label} returned an unexpected response"
        )));
    };
    let Some(map) = value.as_object() else {
        return Err(ProviderHttpError::new(format!(
            "{label} returned an unexpected response"
        )));
    };
    let mut answers = BTreeMap::new();
    for (id, question) in &context.questions {
        let answer_type = map
            .get(id)
            .and_then(|answer| answer.get("type"))
            .and_then(Value::as_str);
        let answer = map.get(id).and_then(Value::as_object);
        let Some(answer) = answer else {
            return Err(ProviderHttpError::new(format!(
                "{label} did not return an answer for {id}"
            )));
        };
        match question {
            ClassifierQuestion::Choice { .. } => {
                let choice = answer.get("choice").and_then(Value::as_str);
                if answer_type != Some("choice") || choice.is_none() {
                    return Err(ProviderHttpError::new(format!(
                        "{label} did not return a choice answer for {id}"
                    )));
                }
                answers.insert(
                    id.clone(),
                    ClassifierAnswer::Choice {
                        choice: choice.expect("checked above").to_string(),
                        probabilities: probabilities(label, answer.get("probabilities"), id)?,
                        confidence: required_number(
                            label,
                            answer.get("confidence"),
                            &format!("confidence for {id}"),
                        )?,
                    },
                );
            }
            ClassifierQuestion::Score { .. } => {
                if answer_type != Some("score") {
                    return Err(ProviderHttpError::new(format!(
                        "{label} did not return a score answer for {id}"
                    )));
                }
                answers.insert(
                    id.clone(),
                    ClassifierAnswer::Score {
                        score: required_number(
                            label,
                            answer.get("score"),
                            &format!("score for {id}"),
                        )?,
                        confidence: required_number(
                            label,
                            answer.get("confidence"),
                            &format!("confidence for {id}"),
                        )?,
                    },
                );
            }
            ClassifierQuestion::Bool { .. } => {
                if answer_type != Some("noul") {
                    return Err(ProviderHttpError::new(format!(
                        "{label} did not return a bool answer for {id}"
                    )));
                }
                answers.insert(
                    id.clone(),
                    ClassifierAnswer::Bool {
                        probability: required_number(
                            label,
                            answer.get("noul"),
                            &format!("probability for {id}"),
                        )?,
                    },
                );
            }
        }
    }
    Ok(answers)
}

/// Usage from System One's `{ input_tokens, output_tokens }`, priced from the
/// model catalog like chat usage. A missing or malformed usage object leaves
/// the result without usage instead of failing it.

/// Maps public `bool` questions to TypeSafe's wire-level `noul` type.
fn wire_request(context: &ClassifierContext) -> SystemOneWireRequest {
    SystemOneWireRequest {
        state: context.state.clone(),
        questions: context
            .questions
            .iter()
            .map(|(id, question)| {
                let mut value = serde_json::to_value(question).expect("question serializes");
                if let Value::Object(object) = &mut value {
                    if matches!(question, ClassifierQuestion::Bool { .. }) {
                        object.insert("type".to_string(), Value::String("noul".to_string()));
                    }
                }
                (id.clone(), value)
            })
            .collect(),
    }
}

/// Runs one System One classification over the given transport
/// (`classifySystemOne`). Failures become a `ClassifierResult` with
/// `stopReason: "error"`/`"aborted"` — never a panic or `Err`.
pub async fn classify_system_one(
    transport: &dyn SystemOneTransport,
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

    if let Err(error) =
        classify_system_one_inner(transport, model, context, options, &mut output).await
    {
        let caller_aborted = options
            .and_then(|options| options.base.signal.as_ref())
            .map(|signal| signal.aborted())
            .unwrap_or(false);
        output.stop_reason = if caller_aborted {
            ClassifierStopReason::Aborted
        } else {
            ClassifierStopReason::Error
        };
        let prefix = format!("{} error", transport.label());
        output.error_message = Some(format_provider_error(
            &normalize_provider_error(&error),
            Some(&prefix),
        ));
    }
    output
}

async fn classify_system_one_inner(
    transport: &dyn SystemOneTransport,
    model: &ClassifierModel,
    context: &ClassifierContext,
    options: Option<&ClassifierOptions>,
    output: &mut ClassifierResult,
) -> Result<(), ProviderHttpError> {
    if model.api() != transport.api() {
        return Err(ProviderHttpError::new(format!(
            "Unsupported classifier API: {}",
            model.api()
        )));
    }
    if context
        .images
        .as_ref()
        .is_some_and(|images| !images.is_empty())
    {
        return Err(ProviderHttpError::new(format!(
            "{} does not support image input",
            transport.label()
        )));
    }
    let body = post_classifier_request(
        transport.label(),
        transport.url(model),
        model,
        transport.payload(model, &wire_request(context)),
        options,
        None,
    )
    .await?;
    let result = transport.output(&body)?;
    // Set before parsing answers: a request with malformed answers was still billed.
    if let Some(usage) = parse_classifier_usage(result.get("usage"), model) {
        output.usage = Some(usage);
    }
    output.answers = parse_answers(transport.label(), result.get("answers"), context)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // The shared-client behavior is exercised through the TypeSafe transport
    // tests in `typesafe_system_one.rs` (the upstream suite lives there too).
    use super::*;

    #[test]
    fn maps_bool_questions_to_wire_noul() {
        let context = crate::api::typesafe_system_one::tests::make_context();
        let wire = wire_request(&context);
        assert_eq!(
            wire.questions
                .get("approved")
                .and_then(|question| question.get("type"))
                .and_then(Value::as_str),
            Some("noul")
        );
        assert_eq!(
            wire.questions
                .get("category")
                .and_then(|question| question.get("type"))
                .and_then(Value::as_str),
            Some("choice")
        );
        assert_eq!(
            wire.questions
                .get("satisfaction")
                .and_then(|question| question.get("type"))
                .and_then(Value::as_str),
            Some("score")
        );
    }

    #[test]
    fn parses_usage_from_token_counts() {
        let model = crate::api::typesafe_system_one::tests::make_model();
        let usage = parse_classifier_usage(
            Some(&serde_json::json!({ "input_tokens": 308, "output_tokens": 23 })),
            &model,
        )
        .expect("usage present");
        assert_eq!(usage.input, 308);
        assert_eq!(usage.output, 23);
        assert_eq!(usage.total_tokens, 331);
        // Unpriced catalog → zero cost.
        assert_eq!(usage.cost.total, 0.0);

        assert!(
            parse_classifier_usage(Some(&serde_json::json!({ "cost": 0.1 })), &model).is_none()
        );
        assert!(parse_classifier_usage(None, &model).is_none());
    }
}
