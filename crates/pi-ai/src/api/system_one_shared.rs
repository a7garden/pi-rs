//! Port of `packages/ai/src/api/system-one-shared.ts` — the shared HTTP client
//! for services serving System One classifier models (TypeSafe, OpenRouter).
//!
//! Deviations:
//! - D24: upstream throws `SystemOneHttpError` objects carrying
//!   `status`/`headers`/`body`; Rust folds that shape into
//!   [`crate::http::ProviderHttpError`], the crate's single provider error
//!   carrier whose completed-response fields (`status`, `headers`, `body`)
//!   feed [`crate::utils::provider_retry`] and
//!   [`crate::utils::error_body`].
//! - D25: upstream builds per-attempt `AbortSignal.timeout` signals; Rust has
//!   no such primitive, so each attempt starts a timer that aborts a
//!   pure-Rust [`crate::utils::abort::AbortSignal`], combined with the caller
//!   signal via `AbortSignal::any`. When the timer fires and the caller signal
//!   has not, the failure is converted to the upstream `"Request timed out
//!   after Nms"` error.
//! - D26: the shared `onPayload`/`onResponse` hooks are typed with the chat
//!   [`Model`] in this crate (foundation `PayloadHook`/`ResponseHook`); the
//!   classifier model is passed as a chat-`Model` view (same id/provider/api/
//!   baseUrl/headers/cost) because [`crate::types::ClassifierModel`] cannot be
//!   converted losslessly. Hooks reading only model identity — the upstream
//!   contract for these hooks — observe correct values.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::http::{FetchFn, HttpBody, HttpRequest, HttpResponse, ProviderHttpError};
use crate::models::calculate_cost;
use crate::types::{
	AnyModel, ClassifierAnswer, ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierQuestion,
	ClassifierResult, ClassifierStopReason, JsonObject, Model, ProviderHeaders, ProviderResponse, Usage, UsageCost,
};
use crate::utils::abort::{AbortController, AbortSignal};
use crate::utils::diagnostics::now_ms;
use crate::utils::error_body::{format_provider_error, normalize_provider_error};
use crate::utils::headers::{headers_to_record, provider_headers_to_record};
use crate::utils::provider_retry::{retry_provider_request, ProviderRetryOptions};

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

fn http_error(label: &str, status: u16, headers: Vec<(String, String)>, body: String) -> ProviderHttpError {
	ProviderHttpError {
		message: format!("{label} returned {status}"),
		status: Some(status),
		headers,
		body: Some(body),
		source: None,
	}
}

fn timeout_error(timeout_ms: u64) -> ProviderHttpError {
	ProviderHttpError::new(format!("Request timed out after {timeout_ms}ms"))
}

pub fn is_record(value: &Value) -> bool {
	value.is_object()
}

fn required_number(label: &str, value: Option<&Value>, field: &str) -> Result<f64, ProviderHttpError> {
	match value.and_then(Value::as_f64).filter(|number| number.is_finite()) {
		Some(number) => Ok(number),
		None => Err(ProviderHttpError::new(format!("{label} returned an invalid {field}"))),
	}
}

fn probabilities(label: &str, value: Option<&Value>, id: &str) -> Result<BTreeMap<String, f64>, ProviderHttpError> {
	let Some(map) = value.and_then(Value::as_object) else {
		return Err(ProviderHttpError::new(format!("{label} returned invalid probabilities for {id}")));
	};
	map.iter()
		.map(|(key, probability)| {
			required_number(label, Some(probability), &format!("probability for {id}.{key}"))
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
		return Err(ProviderHttpError::new(format!("{label} returned an unexpected response")));
	};
	let Some(map) = value.as_object() else {
		return Err(ProviderHttpError::new(format!("{label} returned an unexpected response")));
	};
	let mut answers = BTreeMap::new();
	for (id, question) in &context.questions {
		let answer_type = map.get(id).and_then(|answer| answer.get("type")).and_then(Value::as_str);
		let answer = map.get(id).and_then(Value::as_object);
		let Some(answer) = answer else {
			return Err(ProviderHttpError::new(format!("{label} did not return an answer for {id}")));
		};
		match question {
			ClassifierQuestion::Choice { .. } => {
				let choice = answer.get("choice").and_then(Value::as_str);
				if answer_type != Some("choice") || choice.is_none() {
					return Err(ProviderHttpError::new(format!("{label} did not return a choice answer for {id}")));
				}
				answers.insert(
					id.clone(),
					ClassifierAnswer::Choice {
						choice: choice.expect("checked above").to_string(),
						probabilities: probabilities(label, answer.get("probabilities"), id)?,
						confidence: required_number(label, answer.get("confidence"), &format!("confidence for {id}"))?,
					},
				);
			}
			ClassifierQuestion::Score { .. } => {
				if answer_type != Some("score") {
					return Err(ProviderHttpError::new(format!("{label} did not return a score answer for {id}")));
				}
				answers.insert(
					id.clone(),
					ClassifierAnswer::Score {
						score: required_number(label, answer.get("score"), &format!("score for {id}"))?,
						confidence: required_number(label, answer.get("confidence"), &format!("confidence for {id}"))?,
					},
				);
			}
			ClassifierQuestion::Bool { .. } => {
				if answer_type != Some("noul") {
					return Err(ProviderHttpError::new(format!("{label} did not return a bool answer for {id}")));
				}
				answers.insert(
					id.clone(),
					ClassifierAnswer::Bool {
						probability: required_number(label, answer.get("noul"), &format!("probability for {id}"))?,
					},
				);
			}
		}
	}
	Ok(answers)
}

fn token_count(value: Option<&Value>) -> u64 {
	value
		.and_then(Value::as_f64)
		.filter(|number| number.is_finite() && *number > 0.0)
		.map(|number| number as u64)
		.unwrap_or(0)
}

/// Usage from System One's `{ input_tokens, output_tokens }`, priced from the
/// model catalog like chat usage. A missing or malformed usage object leaves
/// the result without usage instead of failing it.
fn parse_usage(value: Option<&Value>, model: &ClassifierModel) -> Option<Usage> {
	let object = value.and_then(Value::as_object)?;
	if !object.contains_key("input_tokens") && !object.contains_key("output_tokens") {
		return None;
	}
	let input = token_count(object.get("input_tokens"));
	let output = token_count(object.get("output_tokens"));
	let mut usage = Usage {
		input,
		output,
		cache_read: 0,
		cache_write: 0,
		cache_write_1h: None,
		reasoning: None,
		total_tokens: input + output,
		cost: UsageCost::default(),
	};
	calculate_cost(&AnyModel::Classifier(model.clone()), &mut usage);
	Some(usage)
}

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

fn request_headers(
	model: &ClassifierModel,
	api_key: &str,
	options_headers: Option<&ProviderHeaders>,
) -> Vec<(String, String)> {
	let mut base = ProviderHeaders::new();
	base.insert("authorization".to_string(), Some(format!("Bearer {api_key}")));
	base.insert("content-type".to_string(), Some("application/json".to_string()));
	let model_headers: Option<ProviderHeaders> = model.base.headers.as_ref().map(|headers| {
		headers.iter().map(|(name, value)| (name.clone(), Some(value.clone()))).collect()
	});
	provider_headers_to_record(&[Some(&base), model_headers.as_ref(), options_headers])
		.unwrap_or_default()
		.into_iter()
		.collect()
}

/// Chat-`Model` view of the classifier model for the shared request hooks
/// (deviation D26).
fn classifier_model_view(model: &ClassifierModel) -> Model {
	Model {
		base: model.base.clone(),
		model_type: None,
		reasoning: false,
		thinking_level_map: None,
		prompt_cache: None,
		context_window: 0,
		max_tokens: 0,
		sampling_params: None,
		sampling_params_by_thinking_level: None,
		compat: None,
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

	if let Err(error) = classify_system_one_inner(transport, model, context, options, &mut output).await {
		let caller_aborted = options
			.and_then(|options| options.base.signal.as_ref())
			.map(|signal| signal.aborted())
			.unwrap_or(false);
		output.stop_reason = if caller_aborted { ClassifierStopReason::Aborted } else { ClassifierStopReason::Error };
		let prefix = format!("{} error", transport.label());
		output.error_message = Some(format_provider_error(&normalize_provider_error(&error), Some(&prefix)));
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
		return Err(ProviderHttpError::new(format!("Unsupported classifier API: {}", model.api())));
	}
	// Upstream: `if (!options?.apiKey) throw new Error(...)`.
	let Some(options) = options else {
		return Err(ProviderHttpError::new(format!("No API key for provider: {}", model.provider())));
	};
	let Some(api_key) = options.base.api_key.clone() else {
		return Err(ProviderHttpError::new(format!("No API key for provider: {}", model.provider())));
	};
	let mut payload = transport.payload(model, &wire_request(context));
	if let Some(hook) = &options.base.on_payload {
		// (deviation D26) hook receives a chat-Model view of the classifier.
		if let Some(transformed) = hook(payload.clone(), classifier_model_view(model)).await {
			payload = transformed;
		}
	}
	let request_fetch: FetchFn = options.base.fetch.clone().unwrap_or_default();

	let caller_signal = options.base.signal.clone();
	let attempt = || async {
		// Fresh timeout per attempt (deviation D25).
		let timeout_signal: Option<AbortSignal> = options.base.timeout_ms.map(|timeout_ms| {
			let controller = AbortController::new();
			let signal = controller.signal();
			tokio::spawn(async move {
				tokio::time::sleep(Duration::from_millis(timeout_ms)).await;
				controller.abort();
			});
			signal
		});
		let request_signal = match (&caller_signal, &timeout_signal) {
			(Some(caller), Some(timeout)) => Some(AbortSignal::any([caller.clone(), timeout.clone()])),
			(Some(caller), None) => Some(caller.clone()),
			(None, Some(timeout)) => Some(timeout.clone()),
			(None, None) => None,
		};
		let timed_out = || -> Option<ProviderHttpError> {
			let fired = timeout_signal.as_ref().map(|signal| signal.aborted()).unwrap_or(false);
			let caller_aborted = caller_signal.as_ref().map(|signal| signal.aborted()).unwrap_or(false);
			if fired && !caller_aborted {
				Some(timeout_error(options.base.timeout_ms.expect("timeout configured")))
			} else {
				None
			}
		};

		let mut request = HttpRequest::new("POST", transport.url(model));
		for (name, value) in request_headers(model, &api_key, options.base.headers.as_ref()) {
			request = request.header(name, value);
		}
		let body = serde_json::to_vec(&payload).expect("request body serializes");
		request.body = HttpBody::Bytes(body);
		request.signal = request_signal;

		let HttpResponse { status, headers, body } = match request_fetch.send(request).await {
			Ok(response) => response,
			Err(error) => return Err(timed_out().unwrap_or(error)),
		};
		if !(200..300).contains(&status) {
			let body = match body.collect().await {
				Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
				Err(error) => return Err(timed_out().unwrap_or(error)),
			};
			return Err(timed_out().unwrap_or_else(|| http_error(transport.label(), status, headers, body)));
		}
		let bytes = match body.collect().await {
			Ok(bytes) => bytes,
			Err(error) => return Err(timed_out().unwrap_or(error)),
		};
		let parsed: Value = match serde_json::from_slice(&bytes) {
			Ok(parsed) => parsed,
			Err(error) => {
				return Err(ProviderHttpError::with_source("invalid JSON response".to_string(), Arc::new(error)))
			}
		};
		Ok((status, headers, parsed))
	};
	let retry_options = ProviderRetryOptions {
		max_retries: Some(options.base.max_retries.unwrap_or(2)),
		max_retry_delay_ms: options.base.max_retry_delay_ms,
		signal: caller_signal.clone(),
	};
	let (status, headers, parsed) = retry_provider_request(attempt, retry_options).await?;

	if let Some(hook) = &options.base.on_response {
		// (deviation D26) hook receives a chat-Model view of the classifier.
		hook(ProviderResponse { status, headers: headers_to_record(&headers) }, classifier_model_view(model)).await;
	}

	let result = transport.output(&parsed)?;
	// Set before parsing answers: a request with malformed answers was still billed.
	if let Some(usage) = parse_usage(result.get("usage"), model) {
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
			wire.questions.get("approved").and_then(|question| question.get("type")).and_then(Value::as_str),
			Some("noul")
		);
		assert_eq!(
			wire.questions.get("category").and_then(|question| question.get("type")).and_then(Value::as_str),
			Some("choice")
		);
		assert_eq!(
			wire.questions.get("satisfaction").and_then(|question| question.get("type")).and_then(Value::as_str),
			Some("score")
		);
	}

	#[test]
	fn parses_usage_from_token_counts() {
		let model = crate::api::typesafe_system_one::tests::make_model();
		let usage = parse_usage(Some(&serde_json::json!({ "input_tokens": 308, "output_tokens": 23 })), &model)
			.expect("usage present");
		assert_eq!(usage.input, 308);
		assert_eq!(usage.output, 23);
		assert_eq!(usage.total_tokens, 331);
		// Unpriced catalog → zero cost.
		assert_eq!(usage.cost.total, 0.0);

		assert!(parse_usage(Some(&serde_json::json!({ "cost": 0.1 })), &model).is_none());
		assert!(parse_usage(None, &model).is_none());
	}
}
