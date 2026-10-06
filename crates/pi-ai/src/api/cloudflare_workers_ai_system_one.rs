//! Port of `packages/ai/src/api/cloudflare-workers-ai-system-one.ts`.
//!
//! System One models on the Workers AI REST endpoint:
//! `POST /accounts/{account}/ai/run` with `{ model, input }`. The REST API
//! wraps the model output in Cloudflare's API envelope. Third-party models
//! such as `typesafe/jev` add a run record:
//! `{ success, result: { state: "Completed", result: { answers, usage } } }`.
//! https://developers.cloudflare.com/ai/models/typesafe/jev/
//! Cloudflare-hosted models such as `@cf/cloudflare/clef` return the output
//! directly: `{ success, result: { model, answers, usage } }`.
//! https://developers.cloudflare.com/workers-ai/models/clef/
//!
//! Deviations (per-file ledger):
//! - **D15** — shared System One client deviations D24–D26 apply (upstream
//!   `SystemOneHttpError` folded into [`crate::http::ProviderHttpError`],
//!   per-attempt timeout signals, hook receives a chat-`Model` view).
//! - **D16** — upstream's `it` case asserting equality against
//!   `getBuiltinClassifierModel` (exported by `providers/all.ts`) checks the
//!   same embedded catalog through
//!   [`crate::providers::cloudflare_workers_ai_models::
//!   cloudflare_workers_ai_classifier_models`] instead;
//!   `providers/all.rs` registration is final integration's job.
//! - **D17** — upstream stringifies the run state with JS `String(...)` in
//!   the "run did not complete" message; the local [`js_string`] helper
//!   reproduces that coercion for the JSON values Rust can hold.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::{Map, Value};

use crate::api::system_one_shared::{
	classify_system_one, is_record, SystemOneTransport, SystemOneWireRequest,
};
use crate::http::ProviderHttpError;
use crate::models::ProviderClassifier;
use crate::types::{ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierResult, JsonObject};

const LABEL: &str = "Cloudflare Workers AI";

/// JS `String(value)` for the JSON values the wire can carry (deviation D17).
fn js_string(value: &Value) -> String {
	match value {
		Value::String(string) => string.clone(),
		Value::Bool(bool) => bool.to_string(),
		Value::Number(number) => number.to_string(),
		Value::Null => "null".to_string(),
		Value::Array(_) | Value::Object(_) => "[object Object]".to_string(),
	}
}

fn cloudflare_error_message(errors: Option<&Value>) -> String {
	if let Some(errors) = errors.and_then(Value::as_array) {
		let messages: Vec<&str> = errors
			.iter()
			.filter_map(|error| {
				if is_record(error) {
					error.get("message").and_then(Value::as_str)
				} else {
					None
				}
			})
			.collect();
		if !messages.is_empty() {
			return format!("{LABEL} error: {}", messages.join("; "));
		}
	}
	format!("{LABEL} request failed")
}

fn unexpected_response() -> ProviderHttpError {
	ProviderHttpError::new(format!("{LABEL} returned an unexpected response"))
}

/// The transport shared by the Workers AI System One models (`transport`
/// upstream).
pub struct CloudflareWorkersAiSystemOneTransport;

static TRANSPORT: CloudflareWorkersAiSystemOneTransport = CloudflareWorkersAiSystemOneTransport;

impl SystemOneTransport for CloudflareWorkersAiSystemOneTransport {
	fn api(&self) -> &'static str {
		"cloudflare-workers-ai-system-one"
	}

	fn label(&self) -> &'static str {
		LABEL
	}

	fn url(&self, model: &ClassifierModel) -> String {
		// `new URL("run", `${model.baseUrl.replace(/\/+$/u, "")}/`)`.
		format!("{}/run", model.base.base_url.trim_end_matches('/'))
	}

	fn payload(&self, model: &ClassifierModel, request: &SystemOneWireRequest) -> Value {
		Value::Object(Map::from_iter([
			("model".to_string(), Value::String(model.id().to_string())),
			(
				"input".to_string(),
				Value::Object(Map::from_iter([
					("state".to_string(), Value::Object(request.state.clone())),
					("questions".to_string(), Value::Object(request.questions.clone())),
				])),
			),
		]))
	}

	fn output(&self, body: &Value) -> Result<JsonObject, ProviderHttpError> {
		if !is_record(body) {
			return Err(unexpected_response());
		}
		if body.get("success") == Some(&Value::Bool(false)) {
			return Err(ProviderHttpError::new(cloudflare_error_message(body.get("errors"))));
		}
		let result = body.get("result").ok_or_else(unexpected_response)?;
		if !is_record(result) {
			return Err(unexpected_response());
		}
		let result = result.as_object().expect("checked record above");
		if result.contains_key("answers") {
			return Ok(result.clone());
		}
		if result.get("state").map(js_string).as_deref() != Some("Completed") {
			let state = result.get("state").map(js_string).unwrap_or_else(|| "undefined".to_string());
			return Err(ProviderHttpError::new(format!("{LABEL} run did not complete (state: {state})")));
		}
		let inner = result.get("result").ok_or_else(unexpected_response)?;
		if !is_record(inner) {
			return Err(unexpected_response());
		}
		Ok(inner.as_object().expect("checked record above").clone())
	}
}

/// Cloudflare Workers AI System One classification with public `bool` values
/// mapped to wire-level `noul` (`classify` upstream).
pub struct CloudflareWorkersAiSystemOneApi;

impl ProviderClassifier for CloudflareWorkersAiSystemOneApi {
	fn classify<'a>(
		&'a self,
		model: &'a ClassifierModel,
		context: &'a ClassifierContext,
		options: &'a ClassifierOptions,
	) -> Pin<Box<dyn Future<Output = ClassifierResult> + Send + 'a>> {
		Box::pin(classify_system_one(&TRANSPORT, model, context, Some(options)))
	}
}

/// The Workers AI System One classifier API (`cloudflareWorkersAISystemOneApi`
/// upstream).
pub fn cloudflare_workers_ai_system_one_api() -> Arc<dyn ProviderClassifier> {
	Arc::new(CloudflareWorkersAiSystemOneApi)
}

#[cfg(test)]
mod tests {
	// Port of test/cloudflare-workers-ai-system-one.test.ts. Runs the full
	// stack — provider factory, Cloudflare env resolution and the models
	// collection — like upstream.
	use parking_lot::Mutex;

	use super::*;
	use crate::http::{FetchFn, HttpBody, HttpRequest, HttpResponse};
	use crate::models::{create_models, Models, ModelsClassifierOptions, MutableModels};
	use crate::types::ModelType;
	use crate::providers::cloudflare_workers_ai::cloudflare_workers_ai_provider;
	use crate::types::{
		ClassifierAnswer, ClassifierQuestion, ClassifierStopReason, ClassifierTypeTag, ProviderEnv,
	};

	fn classifier_question(question: Value) -> ClassifierQuestion {
		serde_json::from_value(question).expect("question parses")
	}

	fn context() -> ClassifierContext {
		let state = serde_json::json!({ "message": "Help! My payouts have been failing for 3 days." });
		ClassifierContext {
			state: state.as_object().expect("object").clone(),
			questions: [
				(
					"is_urgent",
					classifier_question(serde_json::json!({
						"type": "bool",
						"instructions": "Does this convey urgency?",
						"criteria": { "true": "Explicitly time-sensitive", "false": "No urgency expressed" },
					})),
				),
				(
					"department",
					classifier_question(serde_json::json!({
						"type": "choice",
						"instructions": "Which team should handle this?",
						"criteria": { "billing": "Payments", "technical": "Bugs" },
					})),
				),
			]
			.into_iter()
			.map(|(id, question)| (id.to_string(), question))
			.collect(),
		}
	}

	// Model output from https://developers.cloudflare.com/ai/models/typesafe/jev/
	fn jev_output() -> Value {
		serde_json::json!({
			"model": "jev-1.13.0",
			"answers": {
				"is_urgent": { "type": "noul", "noul": 0.95 },
				"department": {
					"type": "choice",
					"choice": "billing",
					"confidence": 0.8,
					"probabilities": { "billing": 0.87, "technical": 0.13 },
				},
			},
			"usage": { "input_tokens": 426, "output_tokens": 73 },
		})
	}

	// REST envelope observed from the live /ai/run endpoint.
	fn rest_response(state: &str, result: Value) -> Value {
		serde_json::json!({
			"result": { "state": state, "result": result, "gatewayMetadata": { "keySource": "Unified" } },
			"success": true,
			"errors": [],
			"messages": [],
		})
	}

	// Cloudflare-hosted output observed from a live /ai/run call, question ids renamed to
	// match `context`. The envelope carries the output directly, without a run record.
	fn clef_output() -> Value {
		serde_json::json!({
			"model": "clef",
			"answers": {
				"is_urgent": { "type": "noul", "noul": 0.9912 },
				"department": {
					"type": "choice",
					"choice": "technical",
					"probabilities": { "billing": 0.1632, "technical": 0.8368 },
					"confidence": 0.4538,
				},
			},
			"usage": { "input_tokens": 222, "output_tokens": 0 },
		})
	}

	fn direct_response(result: Value) -> Value {
		serde_json::json!({ "result": result, "success": true, "errors": [], "messages": [] })
	}

	fn setup() -> (Arc<crate::models::ModelsImpl>, ClassifierModel) {
		let models = create_models(None);
		models.set_provider(cloudflare_workers_ai_provider());
		let jev = models
			.get_model_of_type(ModelType::Classifier, "cloudflare-workers-ai", "typesafe/jev")
			.expect("missing Cloudflare Jev model");
		let jev = match jev {
			crate::types::AnyModel::Classifier(model) => model,
			other => panic!("expected classifier model, got {other:?}"),
		};
		(models, jev)
	}

	fn recording_fetch(response: Value, requests: Arc<Mutex<Vec<HttpRequest>>>) -> FetchFn {
		FetchFn::new(move |request: HttpRequest| {
			requests.lock().push(request);
			let prepared = HttpResponse {
				status: 200,
				headers: vec![("content-type".to_string(), "application/json".to_string())],
				body: HttpBody::Bytes(response.to_string().into_bytes()),
			};
			Box::pin(async move { Ok(prepared) })
		})
	}

	fn classify_options(auth: &AuthFixture, fetch: FetchFn) -> ModelsClassifierOptions {
		let mut options = ModelsClassifierOptions::default();
		options.base.base.api_key = Some(auth.api_key.clone());
		options.base.base.env = Some(auth.env.clone());
		options.base.base.fetch = Some(fetch);
		options
	}

	struct AuthFixture {
		api_key: String,
		env: ProviderEnv,
	}

	fn auth() -> AuthFixture {
		AuthFixture {
			api_key: "cf-key".to_string(),
			env: [("CLOUDFLARE_ACCOUNT_ID".to_string(), "account-id".to_string())]
				.into_iter()
				.collect(),
		}
	}

	#[test]
	fn exposes_jev_only_through_classifier_catalog_accessors() {
		let (models, jev) = setup();
		let builtin = crate::providers::cloudflare_workers_ai_models::cloudflare_workers_ai_classifier_models()
			.into_iter()
			.find(|model| model.id() == "typesafe/jev")
			.expect("builtin Jev classifier");
		assert_eq!(jev, builtin);
		assert_eq!(jev.model_type, ClassifierTypeTag::Classifier);
		assert_eq!(jev.base.api, "cloudflare-workers-ai-system-one");
		assert!(models.get_model("cloudflare-workers-ai", "typesafe/jev").is_none());
	}

	#[tokio::test]
	async fn runs_jev_through_the_account_scoped_ai_run_endpoint() {
		let (models, jev) = setup();
		let requests: Arc<Mutex<Vec<HttpRequest>>> = Arc::new(Mutex::new(Vec::new()));
		let fetch = recording_fetch(rest_response("Completed", jev_output()), requests.clone());

		let result = models.classify(&jev, &context(), Some(classify_options(&auth(), fetch))).await;

		let captured = requests.lock();
		assert_eq!(captured.len(), 1);
		assert_eq!(
			captured[0].url,
			"https://api.cloudflare.com/client/v4/accounts/account-id/ai/run"
		);
		let authorization = captured[0]
			.headers
			.iter()
			.find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
			.map(|(_, value)| value.as_str());
		assert_eq!(authorization, Some("Bearer cf-key"));
		let payload: Value = match &captured[0].body {
			HttpBody::Bytes(bytes) => serde_json::from_slice(bytes).expect("payload parses"),
			_ => panic!("request body missing"),
		};
		assert_eq!(payload["model"], "typesafe/jev");
		let state = serde_json::json!({ "message": "Help! My payouts have been failing for 3 days." });
		assert_eq!(payload["input"]["state"], state);
		assert_eq!(payload["input"]["questions"]["is_urgent"]["type"], "noul");
		assert_eq!(payload["input"]["questions"]["department"]["type"], "choice");

		assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
		assert_eq!(
			result.answers.get("is_urgent"),
			Some(&ClassifierAnswer::Bool { probability: 0.95 })
		);
		match result.answers.get("department") {
			Some(ClassifierAnswer::Choice { choice, confidence, .. }) => {
				assert_eq!(choice, "billing");
				assert_eq!(*confidence, 0.8);
			}
			other => panic!("expected choice answer, got {other:?}"),
		}
		let usage = result.usage.as_ref().expect("usage present");
		assert_eq!(usage.input, 426);
		assert_eq!(usage.output, 73);
		assert_eq!(usage.total_tokens, 499);
	}

	#[tokio::test]
	async fn runs_clef_models_through_ai_run_and_parses_direct_output() {
		// Upstream `it.each` over ["@cf/cloudflare/clef", 0.24] and
		// ["@cf/cloudflare/clef-flash", 0.09].
		for (id, input_price) in [("@cf/cloudflare/clef", 0.24), ("@cf/cloudflare/clef-flash", 0.09)] {
			let (models, _jev) = setup();
			let clef = models
				.get_model_of_type(ModelType::Classifier, "cloudflare-workers-ai", id)
				.unwrap_or_else(|| panic!("missing Cloudflare {id} model"));
			let clef = match clef {
				crate::types::AnyModel::Classifier(model) => model,
				other => panic!("expected classifier model, got {other:?}"),
			};
			let requests: Arc<Mutex<Vec<HttpRequest>>> = Arc::new(Mutex::new(Vec::new()));
			let fetch = recording_fetch(direct_response(clef_output()), requests.clone());

			let result = models.classify(&clef, &context(), Some(classify_options(&auth(), fetch))).await;

			let captured = requests.lock();
			assert_eq!(captured.len(), 1);
			assert_eq!(
				captured[0].url,
				"https://api.cloudflare.com/client/v4/accounts/account-id/ai/run"
			);
			let payload: Value = match &captured[0].body {
				HttpBody::Bytes(bytes) => serde_json::from_slice(bytes).expect("payload parses"),
				_ => panic!("request body missing"),
			};
			assert_eq!(payload["model"], id);
			let state = serde_json::json!({ "message": "Help! My payouts have been failing for 3 days." });
			assert_eq!(payload["input"]["state"], state);
			assert_eq!(payload["input"]["questions"]["is_urgent"]["type"], "noul");

			assert_eq!(result.stop_reason, ClassifierStopReason::Stop);
			assert_eq!(
				result.answers.get("is_urgent"),
				Some(&ClassifierAnswer::Bool { probability: 0.9912 })
			);
			match result.answers.get("department") {
				Some(ClassifierAnswer::Choice { choice, confidence, .. }) => {
					assert_eq!(choice, "technical");
					assert_eq!(*confidence, 0.4538);
				}
				other => panic!("expected choice answer, got {other:?}"),
			}
			let usage = result.usage.as_ref().expect("usage present");
			assert_eq!(usage.input, 222);
			assert_eq!(usage.output, 0);
			assert_eq!(usage.total_tokens, 222);
			let expected_cost = (222.0 * input_price) / 1_000_000.0;
			assert!((usage.cost.input - expected_cost).abs() < 1e-12, "cost {}", usage.cost.input);
		}
	}

	#[tokio::test]
	async fn reports_runs_that_did_not_complete() {
		let (models, jev) = setup();
		let fetch = recording_fetch(rest_response("Queued", Value::Null), Arc::new(Mutex::new(Vec::new())));

		let result = models.classify(&jev, &context(), Some(classify_options(&auth(), fetch))).await;

		assert_eq!(result.stop_reason, ClassifierStopReason::Error);
		let message = result.error_message.as_deref().expect("error message present");
		assert!(message.contains("run did not complete (state: Queued)"), "unexpected: {message}");
	}

	#[tokio::test]
	async fn reports_cloudflare_envelope_errors() {
		let (models, jev) = setup();
		let response = serde_json::json!({
			"success": false,
			"errors": [{ "code": 5007, "message": "No such model" }],
			"result": null,
		});
		let fetch = recording_fetch(response, Arc::new(Mutex::new(Vec::new())));

		let result = models.classify(&jev, &context(), Some(classify_options(&auth(), fetch))).await;

		assert_eq!(result.stop_reason, ClassifierStopReason::Error);
		let message = result.error_message.as_deref().expect("error message present");
		assert!(message.contains("Cloudflare Workers AI error: No such model"), "unexpected: {message}");
	}
}
