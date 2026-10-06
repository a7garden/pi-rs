//! Azure wrapper of the shared OpenAI Chat Completions API.
//!
//! Upstream has NO `api/azure-openai-completions.ts`: `providers/azure.ts`
//! wraps the shared `openai-completions` API with `azureStreams`, which
//! resolves the Azure endpoint (`resolveAzureModel`) and rewrites the request
//! model to the deployment name (`withDeploymentName`) inside `lazyStream`
//! (`openAICompletionsApi` composition from `openai-completions.lazy.ts`).
//! This module hosts that wiring; the completions implementation itself is
//! reused from [`crate::api::openai_completions`] (never duplicated).
//!
//! Deviations:
//! - **D15**: upstream passes `AzureEndpointOptions` (extends StreamOptions)
//!   through the wrapper; per-API options cannot ride `StreamOptions`
//!   (types.rs frozen; deviation follows the crate D2 pattern), so the
//!   wrapper resolves endpoints from the scoped/process environment and
//!   `model.baseUrl` exactly as upstream does when no explicit options are
//!   given. Final integration should add the nested `StreamOptions` group
//!   (reported in the wave reexports).
//! - **D16**: upstream `lazyStream` defers the dynamic import; Rust links
//!   statically (crate D10) and keeps the stream-error semantics for endpoint
//!   resolution failures.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::api::azure_openai_config::{resolve_azure_base_url, resolve_deployment_name, AzureEndpointOptions};
use crate::api::lazy::lazy_stream;
use crate::models::ProviderStreams;
use crate::types::{
	DeferredCancelOptions, DeferredFetchOptions, DeferredHandle, Model, SimpleStreamOptions, StreamOptions,
	TranscriptContext,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::thrown::ThrownValue;

/// Upstream `resolveAzureModel`: Azure models ship without a base URL; one
/// resource per user, resolved per request.
pub fn resolve_azure_model(
	model: &Model,
	options: Option<&AzureEndpointOptions>,
	env: Option<&crate::types::ProviderEnv>,
) -> Result<Model, String> {
	let mut resolved = model.clone();
	resolved.base.base_url = resolve_azure_base_url(model, options, env)?;
	Ok(resolved)
}

/// Upstream `withDeploymentName` for full stream options: send the deployment
/// name as the request's model, keeping `model.id` as the catalog id. The
/// rewrite runs before the caller's `onPayload`, which sees the deployment
/// name and may still override the final payload.
pub fn with_deployment_name(model: &Model, options: StreamOptions, endpoint: &AzureEndpointOptions, env: Option<&crate::types::ProviderEnv>) -> StreamOptions {
	let deployment_name = resolve_deployment_name(model, Some(endpoint), env);
	if deployment_name == model.id() {
		return options;
	}
	let mut options = options;
	let caller_hook = options.base.on_payload.take();
	options.base.on_payload = Some(Arc::new(move |payload: Value, payload_model: Model| {
		let caller_hook = caller_hook.clone();
		let deployment_name = deployment_name.clone();
		Box::pin(async move {
			// `{ ...(payload), model: deploymentName }`
			let params = match payload {
				Value::Object(mut map) => {
					map.insert("model".to_string(), json!(deployment_name));
					Value::Object(map)
				}
				other => other,
			};
			// `(await options?.onPayload?.(params, payloadModel)) ?? params`
			match caller_hook {
				Some(hook) => hook(params.clone(), payload_model).await.or(Some(params)),
				None => Some(params),
			}
		})
	}));
	options
}

/// Upstream `withDeploymentName` for simple stream options (same rewrite on
/// the nested request options).
pub fn with_deployment_name_simple(
	model: &Model,
	options: SimpleStreamOptions,
	endpoint: &AzureEndpointOptions,
	env: Option<&crate::types::ProviderEnv>,
) -> SimpleStreamOptions {
	let deployment_name = resolve_deployment_name(model, Some(endpoint), env);
	if deployment_name == model.id() {
		return options;
	}
	let mut options = options;
	let caller_hook = options.base.base.on_payload.take();
	options.base.base.on_payload = Some(Arc::new(move |payload: Value, payload_model: Model| {
		let caller_hook = caller_hook.clone();
		let deployment_name = deployment_name.clone();
		Box::pin(async move {
			let params = match payload {
				Value::Object(mut map) => {
					map.insert("model".to_string(), json!(deployment_name));
					Value::Object(map)
				}
				other => other,
			};
			match caller_hook {
				Some(hook) => hook(params.clone(), payload_model).await.or(Some(params)),
				None => Some(params),
			}
		})
	}));
	options
}

/// Upstream `azureStreams`: resolve the Azure endpoint and deployment before
/// dispatch, inside `lazyStream` so an unconfigured endpoint errors on the
/// stream instead of throwing out of `stream()`.
pub fn azure_streams(streams: Arc<dyn ProviderStreams>) -> Arc<dyn ProviderStreams> {
	Arc::new(AzureStreams { inner: streams })
}

/// Upstream composition `azureStreams(openAICompletionsApi())` — the
/// `openai-completions` entry of the Azure provider api map.
pub fn azure_openai_completions_api() -> Arc<dyn ProviderStreams> {
	azure_streams(crate::api::openai_completions::openai_completions_api())
}

/// The wrapped API object (`azureStreams(...)`).
struct AzureStreams {
	inner: Arc<dyn ProviderStreams>,
}

impl ProviderStreams for AzureStreams {
	fn stream(&self, model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
		lazy_stream(model, {
			let model = model.clone();
			let context = context.clone();
			let options = options.clone();
			let inner = self.inner.clone();
			async move {
				let env = options.base.env.clone();
				let resolved = resolve_azure_model(&model, None, env.as_ref()).map_err(ThrownValue::message)?;
				let options = with_deployment_name(&model, options, &AzureEndpointOptions::default(), env.as_ref());
				Ok(inner.stream(&resolved, &context, &options))
			}
		})
	}

	fn stream_simple(
		&self,
		model: &Model,
		context: &TranscriptContext,
		options: &SimpleStreamOptions,
	) -> AssistantMessageEventStream {
		lazy_stream(model, {
			let model = model.clone();
			let context = context.clone();
			let options = options.clone();
			let inner = self.inner.clone();
			async move {
				let env = options.base.base.env.clone();
				let resolved = resolve_azure_model(&model, None, env.as_ref()).map_err(ThrownValue::message)?;
				let options = with_deployment_name_simple(&model, options, &AzureEndpointOptions::default(), env.as_ref());
				Ok(inner.stream_simple(&resolved, &context, &options))
			}
		})
	}

	fn has_fetch_deferred(&self) -> bool {
		self.inner.has_fetch_deferred()
	}

	fn has_cancel_deferred(&self) -> bool {
		self.inner.has_cancel_deferred()
	}

	fn fetch_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: &DeferredFetchOptions,
	) -> Option<AssistantMessageEventStream> {
		self.inner.fetch_deferred(model, handle, options)
	}

	fn cancel_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: &DeferredCancelOptions,
	) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		self.inner.cancel_deferred(model, handle, options)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::models::Provider;
	use crate::types::{
		AssistantContent, Context, Message, StopReason, ThinkingLevel, UserContentField, UserMessage,
	};
	use crate::utils::transcript::normalize_context;
	use std::collections::BTreeMap;

	/// Upstream `azureProvider()` under test.
	fn azure() -> Arc<dyn Provider> {
		crate::providers::azure::azure_provider()
	}

	/// `getModel("azure", id)`.
	fn catalog_model(id: &str) -> Model {
		crate::providers::azure_models::azure_models()
			.into_iter()
			.find(|model| model.id() == id)
			.unwrap_or_else(|| panic!("{id} in azure catalog"))
	}

	fn deepseek_model() -> Model {
		catalog_model("deepseek-v4-pro")
	}

	fn context() -> crate::types::TranscriptContext {
		normalize_context(&Context {
			system_prompt: Some("sys".to_string()),
			messages: vec![Message::User(UserMessage {
				content: UserContentField::Text("hi".to_string()),
				timestamp: 1,
			})],
			tools: None,
		})
	}

	/// Process-environment guard (shared with the azure config/responses test
	/// modules so parallel tests cannot race env mutation).
	fn env_lock() -> std::sync::MutexGuard<'static, ()> {
		crate::api::azure_openai_config::AzureEndpointOptions::test_env_lock()
	}

	fn clear_relevant_env() {
		std::env::remove_var("AZURE_OPENAI_BASE_URL");
		std::env::remove_var("AZURE_OPENAI_RESOURCE_NAME");
		std::env::remove_var("AZURE_OPENAI_DEPLOYMENT_NAME_MAP");
		std::env::remove_var("PI_CACHE_RETENTION");
	}

	/// Recording FetchFn fake returning one chat-completions chunk with usage
	/// and `[DONE]` (upstream `FakeOpenAI.chat.completions.create`).
	fn completions_fetch(recorded: Arc<std::sync::Mutex<Vec<crate::http::HttpRequest>>>) -> crate::http::FetchFn {
		let mut body = String::new();
		body.push_str(&format!(
			"data: {}\n\n",
			json!({
				"choices": [{ "delta": {}, "finish_reason": "stop" }],
				"usage": {
					"prompt_tokens": 1,
					"completion_tokens": 1,
					"prompt_tokens_details": { "cached_tokens": 0 },
				},
			})
		));
		body.push_str("data: [DONE]\n\n");
		let body = body.into_bytes();
		crate::http::FetchFn::new(move |request: crate::http::HttpRequest| {
			recorded.lock().expect("lock").push(request);
			let response = crate::http::HttpResponse {
				status: 200,
				headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
				body: crate::http::HttpBody::Bytes(body.clone()),
			};
			Box::pin(async move { Ok(response) })
		})
	}

	/// Recording FetchFn fake answering Responses-style SSE (for api-map
	/// dispatch assertions on responses-routed models).
	fn responses_fetch(recorded: Arc<std::sync::Mutex<Vec<crate::http::HttpRequest>>>) -> crate::http::FetchFn {
		let mut body = String::new();
		body.push_str(&format!(
			"data: {}\n\n",
			json!({ "type": "response.created", "sequence_number": 0, "response": { "id": "resp_test" } })
		));
		body.push_str(&format!(
			"data: {}\n\n",
			json!({ "type": "response.completed", "sequence_number": 1, "response": { "id": "resp_test", "status": "completed" } })
		));
		let body = body.into_bytes();
		crate::http::FetchFn::new(move |request: crate::http::HttpRequest| {
			recorded.lock().expect("lock").push(request);
			let response = crate::http::HttpResponse {
				status: 200,
				headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
				body: crate::http::HttpBody::Bytes(body.clone()),
			};
			Box::pin(async move { Ok(response) })
		})
	}

	fn captured_body(request: &crate::http::HttpRequest) -> Value {
		match &request.body {
			crate::http::HttpBody::Bytes(bytes) => {
				serde_json::from_str(&String::from_utf8_lossy(bytes)).expect("request body parses")
			}
			_ => Value::Null,
		}
	}

	/// The last captured chat-completions payload (`mockState.lastParams`).
	fn last_params(recorded: &Arc<std::sync::Mutex<Vec<crate::http::HttpRequest>>>) -> Value {
		let requests = recorded.lock().expect("lock");
		let last = requests.last().expect("request captured");
		captured_body(last)
	}

	#[tokio::test]
	async fn turns_thinking_on_with_reasoning_effort_instead_of_thinking_field() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::SimpleStreamOptions::default();
		options.base.base.api_key = Some("test-key".to_string());
		options.base.base.fetch = Some(completions_fetch(recorded.clone()));
		options.reasoning = Some(ThinkingLevel::High);

		let result = azure()
			.stream_simple(&deepseek_model(), &context(), &options)
			.into_result()
			.await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		let params = last_params(&recorded);
		assert_eq!(params["reasoning_effort"], "high");
		assert!(params.get("thinking").is_none(), "thinking: {}", params["thinking"]);
	}

	#[tokio::test]
	async fn clamps_thinking_levels_the_deployment_does_not_accept() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::SimpleStreamOptions::default();
		options.base.base.api_key = Some("test-key".to_string());
		options.base.base.fetch = Some(completions_fetch(recorded.clone()));
		options.reasoning = Some(ThinkingLevel::Max);

		let result = azure()
			.stream_simple(&deepseek_model(), &context(), &options)
			.into_result()
			.await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		assert_eq!(last_params(&recorded)["reasoning_effort"], "high");
	}

	#[tokio::test]
	async fn sends_no_reasoning_effort_when_no_thinking_level_is_requested() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::SimpleStreamOptions::default();
		options.base.base.api_key = Some("test-key".to_string());
		options.base.base.fetch = Some(completions_fetch(recorded.clone()));

		let result = azure()
			.stream_simple(&deepseek_model(), &context(), &options)
			.into_result()
			.await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		let params = last_params(&recorded);
		assert!(params.get("reasoning_effort").is_none(), "reasoning_effort: {}", params["reasoning_effort"]);
		assert!(params.get("thinking").is_none(), "thinking: {}", params["thinking"]);
	}

	#[tokio::test]
	async fn omits_prompt_cache_parameters_when_long_retention_comes_from_env() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		std::env::set_var("PI_CACHE_RETENTION", "long");
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::StreamOptions::default();
		options.base.api_key = Some("test-key".to_string());
		options.base.fetch = Some(completions_fetch(recorded.clone()));
		options.session_id = Some("session-env".to_string());

		let result = azure().stream(&deepseek_model(), &context(), &options).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		let params = last_params(&recorded);
		assert!(params.get("prompt_cache_key").is_none(), "prompt_cache_key: {}", params["prompt_cache_key"]);
		assert!(
			params.get("prompt_cache_retention").is_none(),
			"prompt_cache_retention: {}",
			params["prompt_cache_retention"]
		);
	}

	#[tokio::test]
	async fn sends_the_system_prompt_under_the_system_role() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::StreamOptions::default();
		options.base.api_key = Some("test-key".to_string());
		options.base.fetch = Some(completions_fetch(recorded.clone()));
		options.openai_completions.reasoning_effort = Some(ThinkingLevel::Low);

		let result = azure().stream(&deepseek_model(), &context(), &options).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		let params = last_params(&recorded);
		assert_eq!(params["messages"][0]["role"], "system");
		assert_eq!(params["messages"][0]["content"], "sys");
	}

	#[tokio::test]
	async fn keeps_mid_conversation_system_messages_in_place() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::StreamOptions::default();
		options.base.api_key = Some("test-key".to_string());
		options.base.fetch = Some(completions_fetch(recorded.clone()));

		let resumed = normalize_context(&Context {
			system_prompt: Some("first".to_string()),
			messages: vec![
				Message::User(UserMessage { content: UserContentField::Text("hi".to_string()), timestamp: 1 }),
				Message::System(crate::types::SystemMessage {
					content: crate::types::SystemContent::Text("second".to_string()),
					sections: None,
					tools_added: None,
					tools_removed: None,
					timestamp: 2,
				}),
				Message::User(UserMessage { content: UserContentField::Text("again".to_string()), timestamp: 2 }),
			],
			tools: None,
		});

		let result = azure().stream(&deepseek_model(), &resumed, &options).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		let params = last_params(&recorded);
		let roles: Vec<&str> = params["messages"]
			.as_array()
			.expect("messages")
			.iter()
			.map(|message| message["role"].as_str().expect("role"))
			.collect();
		assert_eq!(roles, vec!["system", "user", "system", "user"]);
	}

	#[tokio::test]
	async fn omits_prompt_cache_parameters_even_when_long_retention_is_requested() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::StreamOptions::default();
		options.base.api_key = Some("test-key".to_string());
		options.base.fetch = Some(completions_fetch(recorded.clone()));
		options.cache_retention = Some(crate::types::CacheRetention::Long);
		options.session_id = Some("session-1".to_string());

		let result = azure().stream(&deepseek_model(), &context(), &options).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		let params = last_params(&recorded);
		assert!(params.get("prompt_cache_key").is_none(), "prompt_cache_key: {}", params["prompt_cache_key"]);
		assert!(
			params.get("prompt_cache_retention").is_none(),
			"prompt_cache_retention: {}",
			params["prompt_cache_retention"]
		);
	}

	#[tokio::test]
	async fn replays_reasoning_content_on_assistant_turns_so_the_cached_prefix_is_unchanged() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::StreamOptions::default();
		options.base.api_key = Some("test-key".to_string());
		options.base.fetch = Some(completions_fetch(recorded.clone()));

		let assistant = crate::types::AssistantMessage {
			content: vec![
				AssistantContent::Thinking(crate::types::ThinkingContent {
					thinking: "internal reasoning".to_string(),
					thinking_signature: Some("reasoning_content".to_string()),
					redacted: None,
				}),
				AssistantContent::Text(crate::types::TextContent { text: "answer".to_string(), text_signature: None }),
			],
			api: crate::types::api_ids::OPENAI_COMPLETIONS.to_string(),
			provider: "azure".to_string(),
			model: "deepseek-v4-pro".to_string(),
			response_model: None,
			response_id: None,
			provider_thinking_level: None,
			thinking_level: None,
			diagnostics: None,
			usage: crate::types::Usage::default(),
			stop_reason: StopReason::Stop,
			deferred: None,
			error_message: None,
			raw_stop_reason: None,
			end_turn: None,
			timestamp: 1,
		};
		let resumed = normalize_context(&Context {
			system_prompt: Some("sys".to_string()),
			messages: vec![
				Message::User(UserMessage { content: UserContentField::Text("first".to_string()), timestamp: 1 }),
				Message::Assistant(assistant),
				Message::User(UserMessage { content: UserContentField::Text("second".to_string()), timestamp: 2 }),
			],
			tools: None,
		});

		let result = azure().stream(&deepseek_model(), &resumed, &options).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		let params = last_params(&recorded);
		let assistant_message = params["messages"]
			.as_array()
			.expect("messages")
			.iter()
			.find(|message| message["role"] == "assistant")
			.expect("assistant message");
		assert_eq!(assistant_message["reasoning_content"], "internal reasoning");
	}

	// ---- azure Chat Completions endpoint resolution -------------------------

	#[tokio::test]
	async fn normalizes_the_azure_endpoint_the_completions_client_is_built_with() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::StreamOptions::default();
		options.base.api_key = Some("test-key".to_string());
		options.base.fetch = Some(completions_fetch(recorded.clone()));

		let result = azure().stream(&deepseek_model(), &context(), &options).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		let requests = recorded.lock().expect("lock");
		// Upstream asserts the client's `baseURL`; the port asserts the wire
		// URL the client builds from it.
		assert_eq!(
			requests.last().expect("request").url,
			"https://my-resource.services.ai.azure.com/openai/v1/chat/completions"
		);
	}

	#[tokio::test]
	async fn surfaces_an_unconfigured_endpoint_as_an_error_event() {
		let _guard = env_lock();
		clear_relevant_env();
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::StreamOptions::default();
		options.base.api_key = Some("test-key".to_string());
		options.base.fetch = Some(completions_fetch(recorded));

		let result = azure().stream(&deepseek_model(), &context(), &options).into_result().await;

		assert_eq!(result.stop_reason, StopReason::Error);
		let message = result.error_message.expect("error message");
		assert!(message.contains("Azure OpenAI base URL is required"), "got: {message}");
	}

	#[tokio::test]
	async fn sends_the_model_id_as_the_request_model() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::StreamOptions::default();
		options.base.api_key = Some("test-key".to_string());
		options.base.fetch = Some(completions_fetch(recorded.clone()));

		let result = azure().stream(&deepseek_model(), &context(), &options).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		assert_eq!(last_params(&recorded)["model"], "deepseek-v4-pro");
		assert_eq!(result.model, "deepseek-v4-pro");
	}

	#[tokio::test]
	async fn sends_the_mapped_deployment_name_while_keeping_the_catalog_id_on_the_message() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		std::env::set_var("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "deepseek-v4-pro=my-deepseek");
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::SimpleStreamOptions::default();
		options.base.base.api_key = Some("test-key".to_string());
		options.base.base.fetch = Some(completions_fetch(recorded.clone()));

		let result = azure()
			.stream_simple(&deepseek_model(), &context(), &options)
			.into_result()
			.await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		assert_eq!(last_params(&recorded)["model"], "my-deepseek");
		assert_eq!(result.model, "deepseek-v4-pro");
	}

	#[tokio::test]
	async fn passes_the_deployment_name_through_a_callers_on_payload() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		std::env::set_var("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "deepseek-v4-pro=my-deepseek");

		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let seen_model = Arc::new(std::sync::Mutex::new(None::<String>));
		let hook_seen = seen_model.clone();
		let mut options = crate::types::StreamOptions::default();
		options.base.api_key = Some("test-key".to_string());
		options.base.fetch = Some(completions_fetch(recorded.clone()));
		options.base.on_payload = Some(Arc::new(move |payload: Value, _model: Model| {
			let hook_seen = hook_seen.clone();
			Box::pin(async move {
				*hook_seen.lock().expect("lock") = payload.get("model").and_then(Value::as_str).map(str::to_string);
				let mut modified = payload;
				if let Value::Object(map) = &mut modified {
					map.insert("temperature".to_string(), json!(0.1));
				}
				Some(modified)
			})
		}));

		let result = azure().stream(&deepseek_model(), &context(), &options).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		assert_eq!(seen_model.lock().expect("lock").as_deref(), Some("my-deepseek"));
		let params = last_params(&recorded);
		assert_eq!(params["model"], "my-deepseek");
		assert_eq!(params["temperature"], 0.1);
	}

	// ---- azure api map -------------------------------------------------------

	#[tokio::test]
	async fn still_routes_responses_models_to_the_responses_api() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::StreamOptions::default();
		options.base.api_key = Some("test-key".to_string());
		options.base.fetch = Some(responses_fetch(recorded.clone()));

		let result = azure()
			.stream(&catalog_model("gpt-4o-mini"), &context(), &options)
			.into_result()
			.await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		// Upstream asserts `dispatchedTo === "responses"`.
		assert!(
			recorded.lock().expect("lock").last().expect("request").url.contains("/responses"),
			"expected a responses dispatch"
		);
	}

	#[tokio::test]
	async fn routes_openai_completions_models_to_chat_completions() {
		let _guard = env_lock();
		clear_relevant_env();
		std::env::set_var("AZURE_OPENAI_BASE_URL", "https://my-resource.services.ai.azure.com");
		let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = crate::types::StreamOptions::default();
		options.base.api_key = Some("test-key".to_string());
		options.base.fetch = Some(completions_fetch(recorded.clone()));

		let result = azure().stream(&deepseek_model(), &context(), &options).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		// Upstream asserts `dispatchedTo === "chat.completions"`.
		assert!(
			recorded.lock().expect("lock").last().expect("request").url.contains("/chat/completions"),
			"expected a chat completions dispatch"
		);
	}

	// Provider-level regression for the azure.rs wiring itself (upstream has no
	// providers.test.ts azure cases): the factory registers the api map, the
	// auth method, and the model catalog.
	#[test]
	fn azure_provider_wires_api_map_auth_and_models() {
		let provider = azure();
		assert_eq!(provider.id(), "azure");
		assert_eq!(provider.name(), "Azure");
		assert!(
			provider.auth().api_key.is_some(),
			"envApiKeyAuth(\"Azure OpenAI API key\", [\"AZURE_OPENAI_API_KEY\"])"
		);
		let models = provider.get_models();
		assert_eq!(models.len(), 45, "44 responses + 1 completions catalog models");
		let apis: BTreeMap<String, usize> = models
			.iter()
			.fold(BTreeMap::new(), |mut acc, model| {
				*acc.entry(model.api().to_string()).or_default() += 1;
				acc
			});
		assert_eq!(apis.get("azure-openai-responses"), Some(&44));
		assert_eq!(apis.get("openai-completions"), Some(&1));
	}
}
