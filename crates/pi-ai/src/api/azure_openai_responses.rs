//! Port of `packages/ai/src/api/azure-openai-responses.ts` (plus its
//! `azure-openai-responses.lazy.ts` twin, deviation crate D10) — the Azure
//! OpenAI Responses stream adapter.
//!
//! Upstream home: this module hosts the `azure-openai-responses` API whose
//! upstream file builds an `AzureOpenAI` SDK client; the request shape below
//! mirrors what that client sends (`POST {baseUrl}/responses` with the
//! `api-key` auth header and the `api-version` default query parameter —
//! openai-node `AzureOpenAI.authHeaders`/`defaultQuery`).
//!
//! Deviations:
//! - **D15**: upstream `AzureOpenAIResponsesOptions extends
//!   AzureEndpointOptions extends StreamOptions`; per-API options cannot ride
//!   `StreamOptions` (types.rs frozen; deviation follows the crate D2
//!   pattern). The struct carries only the upstream extra fields; public
//!   `stream`/`stream_simple` entries build [`AzureOpenAIResponsesOptions::default`]
//!   and delegate to the `*_with_options` variants. Final integration should
//!   add the nested `StreamOptions` group (reported in the wave reexports).
//! - **D16**: the AzureOpenAI SDK client is hand-rolled over
//!   [`crate::http::FetchFn`] (same seam as `openai_responses.rs` D17): the
//!   `api-key` header replaces the Bearer header, and `?api-version=<version>`
//!   is appended to the request URL. The SSE decoder is the shared
//!   `openai_responses.rs` decoder (the upstream SDK transport is shared).
//! - **D17**: synchronous setup failures (`streamSimple` key check) surface as
//!   the stream `error` event per the crate's no-throw-across-stream rule.
//! - **D18**: the upstream catch block deletes transient streaming scratch
//!   fields (`index`/`partialJson`/`customInput`) from content blocks; Rust
//!   content blocks never carry them.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::api::azure_openai_config::{
	resolve_azure_config, resolve_deployment_name, AzureEndpointOptions, AzureResolvedConfig,
};
use crate::api::constrained_sampling::create_grammar_tool_input_properties;
use crate::api::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::api::openai_responses_shared::{
	convert_responses_messages, convert_responses_tools, process_responses_stream, ConvertResponsesMessagesOptions,
	ConvertResponsesToolsOptions, OpenAIResponsesStreamOptions,
};
use crate::api::simple_options::resolve_sampling_params;
use crate::http::{FetchFn, HttpBody, HttpRequest, HttpResponse, ProviderHttpError};
use crate::models::{clamp_thinking_level, ProviderStreams};
use crate::types::{
	AssistantMessage, AssistantMessageEvent, Model, ModelCompat, ModelThinkingLevel, ProviderResponse,
	SimpleStreamOptions, StopReason, StreamDoneReason, StreamErrorReason, StreamOptions, ThinkingLevel,
	ToolChoice, TranscriptContext, Usage,
};
use crate::utils::abort::{race_with_abort_signal, AbortReason};
use crate::utils::diagnostics::now_ms;
use crate::utils::error_body::{format_provider_error, truncate_error_text, NormalizedProviderError, MAX_PROVIDER_ERROR_BODY_CHARS};
use crate::utils::event_stream::{done_event, error_event, AssistantMessageEventStream};
use crate::utils::headers::headers_to_record;
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_retry::{retry_provider_request, ProviderRetryOptions};
use crate::utils::transcript::{get_declared_tools, resolve_transcript, resolve_transcript_tools};

/// Upstream `AZURE_TOOL_CALL_PROVIDERS`.
const AZURE_TOOL_CALL_PROVIDERS: [&str; 4] = ["openai", "openai-codex", "opencode", "azure"];
// OpenAI Responses rejects max_output_tokens below 16 (issue #6265).
const OPENAI_RESPONSES_MIN_OUTPUT_TOKENS: u64 = 16;

/// Azure OpenAI Responses-specific options (upstream
/// `AzureOpenAIResponsesOptions`; deviation D15).
#[derive(Debug, Clone, Default)]
pub struct AzureOpenAIResponsesOptions {
	/// Endpoint options (upstream inherits these from
	/// `AzureEndpointOptions`).
	pub azure_api_version: Option<String>,
	/// Endpoint options (upstream inherits these from
	/// `AzureEndpointOptions`).
	pub azure_resource_name: Option<String>,
	/// Endpoint options (upstream inherits these from
	/// `AzureEndpointOptions`).
	pub azure_base_url: Option<String>,
	/// Endpoint options (upstream inherits these from
	/// `AzureEndpointOptions`).
	pub azure_deployment_name: Option<String>,
	/// Reasoning effort (`"minimal" | ... | "max"`).
	pub reasoning_effort: Option<ThinkingLevel>,
	/// OpenAI Responses tool choice (sent verbatim; string or object).
	pub tool_choice: Option<Value>,
	/// Reasoning summary verbosity; `Some(None)` is upstream `null`.
	pub reasoning_summary: Option<Option<crate::types::ReasoningSummary>>,
}

impl AzureOpenAIResponsesOptions {
	/// The `AzureEndpointOptions` view of these options (upstream
	/// inheritance).
	pub fn endpoint_options(&self) -> AzureEndpointOptions {
		AzureEndpointOptions::from_parts(
			self.azure_api_version.clone(),
			self.azure_resource_name.clone(),
			self.azure_base_url.clone(),
			self.azure_deployment_name.clone(),
		)
	}
}

/// `Required<AzureResponsesCompat>`-style compat resolution: the flags the
/// Azure Responses request assembly reads, with their upstream defaults.
#[derive(Debug, Clone)]
struct AzureResponsesCompat {
	supports_mid_convo_system_messages: bool,
	supports_strict_mode: bool,
	supports_openai_grammar_tools: bool,
	supports_additional_tools: bool,
	supports_tool_search: bool,
}

fn get_compat(model: &Model) -> AzureResponsesCompat {
	let compat = model.compat.as_ref();
	let flag = |get: fn(&ModelCompat) -> Option<bool>| compat.map(|compat| get(compat)).flatten();
	AzureResponsesCompat {
		supports_mid_convo_system_messages: flag(|c| c.supports_mid_convo_system_messages).unwrap_or(false),
		supports_strict_mode: flag(|c| c.supports_strict_mode).unwrap_or(true),
		supports_openai_grammar_tools: flag(|c| c.supports_openai_grammar_tools).unwrap_or(false),
		supports_additional_tools: flag(|c| c.supports_additional_tools).unwrap_or(false),
		supports_tool_search: flag(|c| c.supports_tool_search).unwrap_or(false),
	}
}

fn azure_tool_call_providers() -> BTreeSet<String> {
	AZURE_TOOL_CALL_PROVIDERS.iter().map(|provider| provider.to_string()).collect()
}

/// The SDK's `{baseURL}/responses` join: the path is inserted before any
/// existing query string and the `api-version` default query parameter is
/// appended (`new URL(path, baseURL)` + `defaultQuery`).
fn build_responses_url(base_url: &str, api_version: &str) -> Result<String, String> {
	let mut url = url::Url::parse(base_url).map_err(|_| format!("Invalid Azure OpenAI base URL: {base_url}"))?;
	let base_path = url.path().trim_end_matches('/').to_string();
	url.set_path(&format!("{base_path}/responses"));
	url.query_pairs_mut().append_pair("api-version", api_version);
	Ok(url.to_string())
}

/// `formatAzureOpenAIError`: `formatProviderError(normalizeProviderError(error),
/// "Azure OpenAI API error")`.
fn format_azure_openai_error(error: &AzureResponsesRequestError) -> String {
	let (message, status, raw_body) = match error {
		AzureResponsesRequestError::Http { message, status, body, .. } => (message.clone(), *status, body.clone()),
		AzureResponsesRequestError::Abort => ("Request was aborted.".to_string(), None, None),
		AzureResponsesRequestError::Message(message) => (message.clone(), None, None),
	};
	let body = raw_body
		.as_deref()
		.map(str::trim)
		.filter(|trimmed| !trimmed.is_empty())
		.map(|trimmed| truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS));
	let message_carries_body = match &body {
		None => true,
		Some(body) => message.contains(body),
	};
	let normalized = NormalizedProviderError { status, body, message, message_carries_body };
	format_provider_error(&normalized, Some("Azure OpenAI API error"))
}

/// The exported API implementation (`azureOpenAIResponsesApi` — the lazy twin
/// collapses to a static constructor, crate D10).
pub fn azure_openai_responses_api() -> Arc<dyn ProviderStreams> {
	Arc::new(AzureOpenAIResponsesApi)
}

/// The API implementation object (`export const stream`/`streamSimple`).
struct AzureOpenAIResponsesApi;

impl ProviderStreams for AzureOpenAIResponsesApi {
	fn stream(&self, model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
		stream(model, context, options)
	}

	fn stream_simple(
		&self,
		model: &Model,
		context: &TranscriptContext,
		options: &SimpleStreamOptions,
	) -> AssistantMessageEventStream {
		stream_simple(model, context, options)
	}
}

/// The `stream` export of `azure-openai-responses.ts` (per-API options cannot
/// ride `StreamOptions`; deviation D15 — this entry uses the options group
/// default).
pub fn stream(model: &Model, context: &TranscriptContext, base: &StreamOptions) -> AssistantMessageEventStream {
	stream_with_options(model, context, base, &AzureOpenAIResponsesOptions::default())
}

/// The `stream` export with the upstream `AzureOpenAIResponsesOptions`.
pub fn stream_with_options(
	model: &Model,
	context: &TranscriptContext,
	base: &StreamOptions,
	options: &AzureOpenAIResponsesOptions,
) -> AssistantMessageEventStream {
	let stream = AssistantMessageEventStream::new();
	let compat = get_compat(model);
	let normalized_context = resolve_transcript(context, Some(compat.supports_mid_convo_system_messages));

	let model = model.clone();
	let base = base.clone();
	let options = options.clone();
	let handle = stream.clone();
	tokio::spawn(async move {
		let mut output = AssistantMessage {
			content: Vec::new(),
			api: model.api().to_string(),
			provider: model.provider().to_string(),
			model: model.id().to_string(),
			response_model: None,
			response_id: None,
			provider_thinking_level: None,
			thinking_level: None,
			diagnostics: None,
			usage: Usage::default(),
			stop_reason: StopReason::Pending,
			deferred: None,
			error_message: None,
			raw_stop_reason: None,
			end_turn: None,
			timestamp: now_ms(),
		};

		let result = run_azure_responses(&model, &normalized_context, &base, &options, &handle, &mut output).await;
		match result {
			Ok(()) => {
				let reason = match output.stop_reason {
					StopReason::Stop => StreamDoneReason::Stop,
					StopReason::Length => StreamDoneReason::Length,
					StopReason::ToolUse => StreamDoneReason::ToolUse,
					StopReason::Deferred => StreamDoneReason::Deferred,
					// Guarded by run(): error/aborted/pending never reach here.
					_ => StreamDoneReason::Stop,
				};
				handle.push(done_event(output, reason));
			}
			Err(message) => {
				let aborted = base.base.signal.as_ref().map(|signal| signal.aborted()).unwrap_or(false);
				output.stop_reason = if aborted { StopReason::Aborted } else { StopReason::Error };
				output.error_message = Some(message);
				handle.push(error_event(output, if aborted { StreamErrorReason::Aborted } else { StreamErrorReason::Error }));
			}
		}
	});
	stream
}

/// The `streamSimple` export of `azure-openai-responses.ts` (deviation D15).
pub fn stream_simple(
	model: &Model,
	context: &TranscriptContext,
	options: &SimpleStreamOptions,
) -> AssistantMessageEventStream {
	stream_simple_with_options(model, context, options, &AzureOpenAIResponsesOptions::default())
}

/// The `streamSimple` export with the upstream option group (the endpoint
/// options resolve from the environment/model like upstream, whose
/// `SimpleStreamOptions` cannot carry them either).
pub fn stream_simple_with_options(
	model: &Model,
	context: &TranscriptContext,
	options: &SimpleStreamOptions,
	azure: &AzureOpenAIResponsesOptions,
) -> AssistantMessageEventStream {
	// Upstream throws synchronously before a stream exists; Rust surfaces the
	// failure as the stream error event (D17).
	let has_api_key = options.base.base.api_key.as_deref().is_some_and(|key| !key.is_empty());
	if !has_api_key {
		let stream = AssistantMessageEventStream::new();
		stream.push(error_event(
			AssistantMessage {
				content: Vec::new(),
				api: model.api().to_string(),
				provider: model.provider().to_string(),
				model: model.id().to_string(),
				response_model: None,
				response_id: None,
				provider_thinking_level: None,
				thinking_level: None,
				diagnostics: None,
				usage: Usage::default(),
				stop_reason: StopReason::Error,
				deferred: None,
				error_message: Some(format!("No API key for provider: {}", model.provider())),
				raw_stop_reason: None,
				end_turn: None,
				timestamp: now_ms(),
			},
			StreamErrorReason::Error,
		));
		return stream;
	}

	let base = crate::api::simple_options::build_base_options(model, context, options, options.base.base.api_key.as_deref());
	let mut azure_options = azure.clone();
	azure_options.tool_choice = options.tool_choice.map(|choice| match choice {
		ToolChoice::Auto => json!("auto"),
		ToolChoice::None => json!("none"),
	});
	let clamped_reasoning =
		options.reasoning.map(|reasoning| clamp_thinking_level(model, ModelThinkingLevel::from(reasoning)));
	// `clampedReasoning === "off" ? undefined : clampedReasoning`.
	azure_options.reasoning_effort = clamped_reasoning.and_then(|level| match level {
		ModelThinkingLevel::Off => None,
		ModelThinkingLevel::Minimal => Some(ThinkingLevel::Minimal),
		ModelThinkingLevel::Low => Some(ThinkingLevel::Low),
		ModelThinkingLevel::Medium => Some(ThinkingLevel::Medium),
		ModelThinkingLevel::High => Some(ThinkingLevel::High),
		ModelThinkingLevel::Xhigh => Some(ThinkingLevel::Xhigh),
		ModelThinkingLevel::Max => Some(ThinkingLevel::Max),
	});

	stream_with_options(model, context, &base, &azure_options)
}

/// Request-level error mirroring the SDK error objects (see
/// `openai_responses.rs`). Implements the
/// [`crate::utils::provider_retry::ProviderRequestError`] protocol.
enum AzureResponsesRequestError {
	Http { message: String, status: Option<u16>, headers: Vec<(String, String)>, body: Option<String> },
	Abort,
	Message(String),
}

impl crate::utils::provider_retry::ProviderRequestError for AzureResponsesRequestError {
	fn status(&self) -> Option<u16> {
		match self {
			AzureResponsesRequestError::Http { status, .. } => *status,
			_ => None,
		}
	}

	fn header(&self, name: &str) -> Option<String> {
		match self {
			AzureResponsesRequestError::Http { headers, .. } => headers
				.iter()
				.find(|(key, _)| key.eq_ignore_ascii_case(name))
				.map(|(_, value)| value.clone()),
			_ => None,
		}
	}

	fn message(&self) -> &str {
		match self {
			AzureResponsesRequestError::Http { message, .. } => message,
			AzureResponsesRequestError::Abort => "Request was aborted.",
			AzureResponsesRequestError::Message(message) => message,
		}
	}

	fn from_abort() -> Self {
		AzureResponsesRequestError::Abort
	}
}

/// Default request headers plus caller overrides (`createClient`
/// `defaultHeaders`): pi user agent, model headers, caller headers last.
/// Caller `None` values suppress a default header.
fn build_request_headers(
	model: &Model,
	options_headers: Option<&BTreeMap<String, Option<String>>>,
) -> Vec<(String, String)> {
	let mut headers: Vec<(String, String)> = Vec::new();
	headers.push(("User-Agent".to_string(), get_pi_user_agent()));
	let set_header = |headers: &mut Vec<(String, String)>, name: &str, value: String| {
		if let Some(entry) = headers.iter_mut().find(|(key, _)| key.eq_ignore_ascii_case(name)) {
			entry.1 = value;
		} else {
			headers.push((name.to_string(), value));
		}
	};
	if let Some(model_headers) = &model.base.headers {
		for (key, value) in model_headers {
			set_header(&mut headers, key, value.clone());
		}
	}
	if let Some(options_headers) = options_headers {
		for (key, value) in options_headers {
			match value {
				Some(value) => set_header(&mut headers, key, value.clone()),
				None => headers.retain(|(name, _)| !name.eq_ignore_ascii_case(key)),
			}
		}
	}
	headers
}

/// Build the Responses create payload (`buildParams`). `deployment_name` is
/// sent as the request model, keeping the catalog id on the message. The
/// grammar-tool map reaches `convert_responses_messages` in
/// [`run_azure_responses`] (upstream threads it through the messages
/// conversion the same way).
fn build_params(
	model: &Model,
	messages: Vec<Value>,
	request_tools: &[crate::types::Tool],
	base: &StreamOptions,
	options: &AzureOpenAIResponsesOptions,
	deployment_name: &str,
	compat: &AzureResponsesCompat,
) -> Result<Value, String> {
	let mut params = Map::new();
	params.insert("model".to_string(), json!(deployment_name));
	params.insert("input".to_string(), Value::Array(messages));
	params.insert("stream".to_string(), json!(true));
	if let Some(key) = clamp_openai_prompt_cache_key(base.session_id.as_deref()) {
		params.insert("prompt_cache_key".to_string(), json!(key));
	}
	params.insert("store".to_string(), json!(false));

	// `if (options?.maxTokens)` — falsy (0) is skipped upstream.
	if let Some(max_tokens) = base.max_tokens.filter(|max_tokens| *max_tokens > 0) {
		params.insert(
			"max_output_tokens".to_string(),
			json!(std::cmp::max(max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)),
		);
	}

	if let Some(temperature) = base.temperature {
		params.insert("temperature".to_string(), json!(temperature));
	}

	if !request_tools.is_empty() {
		let tool_options = ConvertResponsesToolsOptions {
			strict: None,
			supports_strict_mode: Some(compat.supports_strict_mode),
			supports_openai_grammar_tools: Some(compat.supports_openai_grammar_tools),
			tool_search_result: false,
		};
		params.insert("tools".to_string(), Value::Array(convert_responses_tools(request_tools, Some(&tool_options))?));
	}
	if let Some(tool_choice) = &options.tool_choice {
		params.insert("tool_choice".to_string(), tool_choice.clone());
	}

	let reasoning_summary = options.reasoning_summary;
	// `options?.reasoningEffort ?? (options?.reasoningSummary ? "medium" : undefined)`
	// — an explicit `null` summary is falsy upstream and does not imply a
	// default effort.
	let reasoning_effort = options
		.reasoning_effort
		.or_else(|| reasoning_summary.filter(|summary| matches!(summary, Some(_))).map(|_| ThinkingLevel::Medium));

	if model.reasoning {
		if let Some(effort_level) = reasoning_effort {
			// `options?.reasoningEffort ? (model.thinkingLevelMap?.[effort] ?? effort) : reasoningEffort`
			let effort = if options.reasoning_effort.is_some() {
				let map_value = model
					.thinking_level_map
					.as_ref()
					.and_then(|map| map.get(&ModelThinkingLevel::from(effort_level)))
					.cloned()
					.flatten();
				map_value
					.map(Value::String)
					.unwrap_or_else(|| serde_json::to_value(effort_level).expect("effort serializes"))
			} else {
				serde_json::to_value(effort_level).expect("effort serializes")
			};
			let summary = match reasoning_summary {
				Some(Some(summary)) => json!(summary),
				_ => json!("auto"),
			};
			params.insert("reasoning".to_string(), json!({ "effort": effort, "summary": summary }));
			params.insert("include".to_string(), json!(["reasoning.encrypted_content"]));
		} else if model
			.thinking_level_map
			.as_ref()
			.and_then(|map| map.get(&ModelThinkingLevel::Off))
			.cloned()
			!= Some(None)
		{
			// `model.thinkingLevelMap?.off ?? "none"`
			let effort = model
				.thinking_level_map
				.as_ref()
				.and_then(|map| map.get(&ModelThinkingLevel::Off))
				.cloned()
				.flatten()
				.unwrap_or_else(|| "none".to_string());
			params.insert("reasoning".to_string(), json!({ "effort": effort }));
		}
	}

	// Last so model and request sampling parameters override named request fields.
	let effort_for_sampling = reasoning_effort
		.map(ModelThinkingLevel::from)
		.unwrap_or(ModelThinkingLevel::Off);
	if let Some(sampling_params) =
		resolve_sampling_params(model, effort_for_sampling, base.sampling_params.as_ref())
	{
		for (key, value) in sampling_params {
			params.insert(key, value);
		}
	}

	Ok(Value::Object(params))
}

/// The request/response/stream pipeline between stream construction and the
/// terminal event. Returns `Err` with the formatted error message on failure.
async fn run_azure_responses(
	model: &Model,
	normalized_context: &TranscriptContext,
	base: &StreamOptions,
	options: &AzureOpenAIResponsesOptions,
	stream: &AssistantMessageEventStream,
	output: &mut AssistantMessage,
) -> Result<(), String> {
	let compat = get_compat(model);
	let endpoint_options = options.endpoint_options();

	// `options?.apiKey` — the explicit key only (upstream does not consult
	// headers here).
	let api_key = base
		.base
		.api_key
		.as_deref()
		.filter(|key| !key.is_empty())
		.ok_or_else(|| format!("No API key for provider: {}", model.provider()))?;

	let deployment_name = resolve_deployment_name(model, Some(&endpoint_options), base.base.env.as_ref());

	let declared_tools = get_declared_tools(normalized_context.messages());
	let grammar_tool_input_properties =
		create_grammar_tool_input_properties(&declared_tools, compat.supports_openai_grammar_tools)?;

	let messages = convert_responses_messages(
		model,
		normalized_context,
		&azure_tool_call_providers(),
		Some(&ConvertResponsesMessagesOptions {
			include_system_prompt: None,
			grammar_tool_input_properties: Some(grammar_tool_input_properties.clone()),
			supports_mid_convo_system_messages: compat.supports_mid_convo_system_messages,
			supports_additional_tools: compat.supports_additional_tools,
			supports_tool_search: compat.supports_tool_search,
			tool_options: Some(ConvertResponsesToolsOptions {
				strict: None,
				supports_strict_mode: Some(compat.supports_strict_mode),
				supports_openai_grammar_tools: Some(compat.supports_openai_grammar_tools),
				tool_search_result: false,
			}),
		}),
	)?;

	let transcript_tools = resolve_transcript_tools(
		normalized_context.messages(),
		compat.supports_additional_tools || compat.supports_tool_search,
	);

	let mut params = build_params(
		model,
		messages,
		&transcript_tools.request_tools,
		base,
		options,
		&deployment_name,
		&compat,
	)?;
	if let Some(on_payload) = &base.base.on_payload {
		if let Some(next) = on_payload(params.clone(), model.clone()).await {
			params = next;
		}
	}

	// Build the HTTP request (D16): the AzureOpenAI SDK's
	// `client.responses.create` — `{baseURL}/responses` plus the `api-version`
	// default query parameter and the `api-key` auth header. The path joins
	// before any proxy query string and `api-version` is appended, matching
	// the SDK's URL builder.
	let AzureResolvedConfig { base_url, api_version } =
		resolve_azure_config(model, Some(&endpoint_options), base.base.env.as_ref())?;
	let url = build_responses_url(&base_url, &api_version)?;
	let mut request = HttpRequest::new("POST", &url);
	request.headers.push(("api-key".to_string(), api_key.to_string()));
	request.headers.push(("Content-Type".to_string(), "application/json".to_string()));
	request.headers.push(("Accept".to_string(), "application/json".to_string()));
	for (key, value) in build_request_headers(model, base.base.headers.as_ref()) {
		if key.eq_ignore_ascii_case("api-key") || key.eq_ignore_ascii_case("content-type") {
			// Caller values override the SDK defaults above.
			request.headers.retain(|(name, _)| !name.eq_ignore_ascii_case(&key));
		}
		request.headers.push((key, value));
	}

	// Plain request data captured by the retry factory (a `HttpRequest` can
	// carry a non-`Sync` stream body, so it cannot be captured by reference).
	let request_template = (
		request.method.clone(),
		request.url.clone(),
		request.headers.clone(),
		serde_json::to_vec(&params).expect("request body serializes"),
		base.base.timeout_ms,
		base.base.signal.clone(),
	);

	let fetch: FetchFn = base.base.fetch.clone().unwrap_or_default();
	let max_retries = base.base.max_retries;
	let max_retry_delay_ms = base.base.max_retry_delay_ms;
	let signal_for_retry = base.base.signal.clone();

	let response: HttpResponse = match race_with_abort_signal(
		retry_provider_request(
			|| {
				let fetch = fetch.clone();
				let request = HttpRequest {
					method: request_template.0.clone(),
					url: request_template.1.clone(),
					headers: request_template.2.clone(),
					body: HttpBody::Bytes(request_template.3.clone()),
					timeout_ms: request_template.4,
					signal: request_template.5.clone(),
				};
				async move {
					match fetch.send(request).await {
						Ok(response) if !response.ok() => {
							let body = response.body.collect().await.unwrap_or_default();
							let body = String::from_utf8_lossy(&body).to_string();
							Err(AzureResponsesRequestError::Http {
								message: response.status.to_string(),
								status: Some(response.status),
								headers: response.headers.clone(),
								body: Some(body),
							})
						}
						Ok(response) => Ok(response),
						Err(error) => Err(AzureResponsesRequestError::Http {
							message: error.message,
							status: error.status,
							headers: error.headers,
							body: error.body,
						}),
					}
				}
			},
			ProviderRetryOptions { max_retries, max_retry_delay_ms, signal: signal_for_retry },
		),
		base.base.signal.clone().unwrap_or_default(),
	)
	.await
	{
		Ok(result) => result.map_err(|error| format_azure_openai_error(&error))?,
		Err(reason) => {
			let error = match reason {
				AbortReason::Aborted => AzureResponsesRequestError::Abort,
				AbortReason::Message(message) => AzureResponsesRequestError::Message(message),
				AbortReason::Error(error) => AzureResponsesRequestError::Message(error.to_string()),
			};
			return Err(format_azure_openai_error(&error));
		}
	};

	if let Some(on_response) = &base.base.on_response {
		on_response(
			ProviderResponse { status: response.status, headers: headers_to_record(&response.headers) },
			model.clone(),
		)
		.await;
	}

	let mut event_stream = Box::pin(decode_sse(response.body));

	stream.push(AssistantMessageEvent::Start { partial: output.clone() });

	let stream_options = OpenAIResponsesStreamOptions {
		on_provider_stream_event: base.on_provider_stream_event.clone(),
		service_tier: None,
		grammar_tool_input_properties,
		resolve_service_tier: None,
		apply_service_tier_pricing: None,
	};

	process_responses_stream(&mut event_stream, output, stream, model, Some(&stream_options)).await?;

	if base.base.signal.as_ref().map(|signal| signal.aborted()).unwrap_or(false) {
		return Err("Request was aborted".to_string());
	}
	if output.stop_reason == StopReason::Pending {
		return Err("Azure OpenAI Responses stream ended without a stop reason".to_string());
	}
	if output.stop_reason == StopReason::Aborted || output.stop_reason == StopReason::Error {
		return Err(output.error_message.clone().unwrap_or_else(|| "An unknown error occurred".to_string()));
	}
	Ok(())
}

// =============================================================================
// SSE decoding (hand-rolled, upstream uses the SDK's decoder; same decoder as
// `openai_responses.rs`)
// =============================================================================

/// Incremental `text/event-stream` decoder producing the JSON payload of each
/// event. `data:` lines are joined with `\n`; blank lines dispatch; unknown
/// fields and comments are ignored.
#[derive(Default)]
struct SseDecoder {
	buffer: String,
	data_lines: Vec<String>,
}

impl SseDecoder {
	fn feed(&mut self, chunk: &str) -> Vec<Value> {
		self.buffer.push_str(chunk);
		let mut events = Vec::new();
		while let Some(break_at) = self.buffer.find('\n') {
			let line: String = self.buffer.drain(..break_at + 1).collect();
			let line = line.trim_end_matches(['\n', '\r']);
			if let Some(event) = self.handle_line(line) {
				events.push(event);
			}
		}
		events
	}

	/// Dispatch any trailing event at end of stream.
	fn finish(&mut self) -> Vec<Value> {
		let line = self.buffer.clone();
		self.buffer.clear();
		let mut events = self.handle_line(&line).into_iter().collect::<Vec<_>>();
		events.extend(self.flush().into_iter());
		events
	}

	fn handle_line(&mut self, line: &str) -> Option<Value> {
		if line.is_empty() {
			return self.flush();
		}
		if let Some(data) = line.strip_prefix("data:") {
			self.data_lines.push(data.strip_prefix(' ').unwrap_or(data).to_string());
		}
		// `event:`, `id:`, `retry:`, comments (`:`) — the Responses API encodes
		// the event type inside the data payload, so the SSE event name is
		// redundant and ignored, as in the SDK decoder.
		None
	}

	fn flush(&mut self) -> Option<Value> {
		if self.data_lines.is_empty() {
			return None;
		}
		let data = self.data_lines.join("\n");
		self.data_lines.clear();
		serde_json::from_str(&data).ok()
	}
}

/// Decode an SSE body into a stream of JSON event payloads.
fn decode_sse(body: HttpBody) -> futures_core::stream::BoxStream<'static, Result<Value, ProviderHttpError>> {
	use futures_util::StreamExt;
	let chunks: futures_core::stream::BoxStream<'static, Result<Vec<u8>, ProviderHttpError>> = match body {
		HttpBody::Empty => Box::pin(futures_util::stream::empty()),
		HttpBody::Bytes(bytes) => Box::pin(futures_util::stream::iter(vec![Ok(bytes)])),
		HttpBody::Stream(stream) => stream,
	};
	Box::pin(futures_util::stream::unfold(
		(chunks, SseDecoder::default(), std::collections::VecDeque::new(), false),
		|(mut chunks, mut decoder, mut pending, mut finished)| async move {
			loop {
				if let Some(event) = pending.pop_front() {
					return Some((Ok(event), (chunks, decoder, pending, finished)));
				}
				if finished {
					return None;
				}
				match chunks.next().await {
					Some(Ok(bytes)) => {
						let text = String::from_utf8_lossy(&bytes).into_owned();
						pending.extend(decoder.feed(&text));
					}
					Some(Err(error)) => {
						finished = true;
						return Some((Err(error), (chunks, decoder, pending, finished)));
					}
					None => {
						pending.extend(decoder.finish());
						finished = true;
					}
				}
			}
		},
	))
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::{
		api_ids, provider_ids, BaseModelFields, Context, Message, ModelCost, ModelCostRates, ModelInput,
		UserContentField, UserMessage,
	};
	use crate::utils::transcript::normalize_context;

	// ---- helpers (upstream test scaffolding) --------------------------------

	/// `getModel("azure", id)`: the catalog entry for the model id.
	fn catalog_model(id: &str) -> Model {
		crate::providers::azure_models::azure_models()
			.into_iter()
			.find(|model| model.id() == id)
			.unwrap_or_else(|| panic!("{id} in azure catalog"))
	}

	fn user_message(text: &str, timestamp: i64) -> Message {
		Message::User(UserMessage { content: UserContentField::Text(text.to_string()), timestamp })
	}

	fn user_context(text: &str) -> Context {
		Context { system_prompt: None, messages: vec![user_message(text, 1)], tools: None }
	}

	fn normalize(context: &Context) -> TranscriptContext {
		normalize_context(context)
	}

	/// Process-environment guard (shared with the azure config/completions
	/// test modules so parallel tests cannot race env mutation).
	fn env_lock() -> std::sync::MutexGuard<'static, ()> {
		AzureEndpointOptions::test_env_lock()
	}

	const AZURE_ENV_VARS: [&str; 4] =
		["AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_RESOURCE_NAME", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_API_KEY"];

	fn clear_azure_env() {
		for name in AZURE_ENV_VARS {
			std::env::remove_var(name);
		}
	}

	fn set_env(name: &str, value: &str) {
		std::env::set_var(name, value);
	}

	/// Recording FetchFn fake returning constructed Responses SSE events
	/// (upstream `AzureOpenAI` mock's `responses.create` + `withResponse`).
	fn responses_fetch(sse: String, requests: Arc<std::sync::Mutex<Vec<HttpRequest>>>) -> FetchFn {
		let body = Arc::new(sse.into_bytes());
		FetchFn::new(move |request: HttpRequest| {
			requests.lock().expect("lock").push(request);
			let response = HttpResponse {
				status: 200,
				headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
				body: HttpBody::Bytes((*body).clone()),
			};
			Box::pin(async move { Ok(response) })
		})
	}

	fn sse_body(events: &[Value]) -> String {
		events.iter().map(|event| format!("data: {event}\n\n")).collect()
	}

	fn completion_events(id: &str) -> Vec<Value> {
		vec![
			json!({ "type": "response.created", "sequence_number": 0, "response": { "id": id } }),
			json!({ "type": "response.completed", "sequence_number": 1, "response": { "id": id, "status": "completed" } }),
		]
	}

	fn base_options(fetch: FetchFn) -> StreamOptions {
		let mut options = StreamOptions::default();
		options.base.api_key = Some("test-api-key".to_string());
		options.base.fetch = Some(fetch);
		options
	}

	fn header<'a>(request: &'a HttpRequest, name: &str) -> &'a str {
		request
			.headers
			.iter()
			.find(|(key, _)| key.eq_ignore_ascii_case(name))
			.map(|(_, value)| value.as_str())
			.unwrap_or_default()
	}

	/// `captureClientBaseUrl`: stream with the given env base URL and return
	/// the captured request URL. Upstream asserts the SDK client's `baseURL`;
	/// the port asserts the wire request the SDK would have built
	/// (`{baseURL}/responses?api-version=v1`).
	async fn capture_request_url(env_base_url: &str) -> String {
		let _guard = env_lock();
		clear_azure_env();
		set_env("AZURE_OPENAI_BASE_URL", env_base_url);

		let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
		let model = catalog_model("gpt-4o-mini");
		let options = base_options(responses_fetch(sse_body(&completion_events("resp_test")), requests.clone()));
		let result = stream(&model, &normalize(&user_context("hello")), &options).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let requests = requests.lock().expect("lock");
		assert_eq!(requests.len(), 1);
		requests[0].url.clone()
	}

	/// `captureClientHeaders`: stream with an explicit `azureBaseUrl` option
	/// (and optional caller headers); return the captured request headers.
	async fn capture_request_headers(headers: Option<crate::types::ProviderHeaders>) -> Vec<(String, String)> {
		let _guard = env_lock();
		clear_azure_env();
		let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
		let model = catalog_model("gpt-4o-mini");
		let mut options = base_options(responses_fetch(sse_body(&completion_events("resp_test")), requests.clone()));
		options.base.headers = headers;
		let azure = AzureOpenAIResponsesOptions {
			azure_base_url: Some("https://my-resource.openai.azure.com".to_string()),
			..Default::default()
		};
		let result = stream_with_options(&model, &normalize(&user_context("hello")), &options, &azure)
			.into_result()
			.await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let headers = requests.lock().expect("lock")[0].headers.clone();
		headers
	}

	// ---- azure base URL normalization (azure-openai-base-url.test.ts) -------

	#[tokio::test]
	async fn normalizes_cognitive_services_root_endpoints_to_openai_v1() {
		let url = capture_request_url("https://marc-quicktests-resource.cognitiveservices.azure.com").await;
		assert_eq!(url, "https://marc-quicktests-resource.cognitiveservices.azure.com/openai/v1/responses?api-version=v1");
	}

	#[tokio::test]
	async fn normalizes_microsoft_foundry_root_endpoints_to_openai_v1() {
		let url = capture_request_url("https://marc-quicktests-resource.ai.azure.com").await;
		assert_eq!(url, "https://marc-quicktests-resource.ai.azure.com/openai/v1/responses?api-version=v1");
	}

	#[tokio::test]
	async fn normalizes_azure_openai_root_endpoints_to_openai_v1() {
		let url = capture_request_url("https://my-resource.openai.azure.com").await;
		assert_eq!(url, "https://my-resource.openai.azure.com/openai/v1/responses?api-version=v1");
	}

	#[tokio::test]
	async fn normalizes_openai_to_openai_v1() {
		let url = capture_request_url("https://my-resource.cognitiveservices.azure.com/openai").await;
		assert_eq!(url, "https://my-resource.cognitiveservices.azure.com/openai/v1/responses?api-version=v1");
	}

	#[tokio::test]
	async fn preserves_openai_v1_endpoints() {
		let url = capture_request_url("https://my-resource.cognitiveservices.azure.com/openai/v1").await;
		assert_eq!(url, "https://my-resource.cognitiveservices.azure.com/openai/v1/responses?api-version=v1");
	}

	#[tokio::test]
	async fn normalizes_openai_v1_responses_to_openai_v1() {
		let url = capture_request_url("https://my-resource.services.ai.azure.com/openai/v1/responses").await;
		assert_eq!(url, "https://my-resource.services.ai.azure.com/openai/v1/responses?api-version=v1");
	}

	#[tokio::test]
	async fn preserves_explicit_non_azure_proxy_paths() {
		let url = capture_request_url("https://my-proxy.example.com/v1").await;
		assert_eq!(url, "https://my-proxy.example.com/v1/responses?api-version=v1");
	}

	#[tokio::test]
	async fn strips_query_params_when_normalizing_azure_host_urls() {
		let url = capture_request_url("https://my-resource.openai.azure.com/openai?api-version=2024-12-01").await;
		assert_eq!(url, "https://my-resource.openai.azure.com/openai/v1/responses?api-version=v1");
	}

	#[tokio::test]
	async fn preserves_query_params_on_non_azure_proxy_urls() {
		let url = capture_request_url("https://my-proxy.example.com/v1?custom=true").await;
		// The SDK joins `{baseURL}/responses` before the existing query and
		// appends `api-version` as a default query parameter.
		assert_eq!(url, "https://my-proxy.example.com/v1/responses?custom=true&api-version=v1");
	}

	#[tokio::test]
	async fn errors_on_invalid_urls() {
		let _guard = env_lock();
		clear_azure_env();
		set_env("AZURE_OPENAI_BASE_URL", "not-a-url");
		let model = catalog_model("gpt-4o-mini");
		let options = base_options(responses_fetch(sse_body(&completion_events("resp_test")), Default::default()));
		let result = stream(&model, &normalize(&user_context("hello")), &options).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Error);
		let message = result.error_message.expect("error message");
		assert!(message.contains("Invalid Azure OpenAI base URL"), "got: {message}");
		assert!(message.contains("not-a-url"), "got: {message}");
	}

	#[tokio::test]
	async fn clamps_prompt_cache_key_to_openai_64_character_limit() {
		let _guard = env_lock();
		clear_azure_env();
		let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
		let model = catalog_model("gpt-4o-mini");
		let mut options = base_options(responses_fetch(sse_body(&completion_events("resp_test")), requests.clone()));
		options.session_id = Some("x".repeat(67));
		let azure = AzureOpenAIResponsesOptions {
			azure_base_url: Some("https://my-resource.openai.azure.com".to_string()),
			..Default::default()
		};
		let result = stream_with_options(&model, &normalize(&user_context("hello")), &options, &azure)
			.into_result()
			.await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let request = &requests.lock().expect("lock")[0];
		let body = match &request.body {
			HttpBody::Bytes(bytes) => String::from_utf8_lossy(bytes).to_string(),
			_ => String::new(),
		};
		let params: Value = serde_json::from_str(&body).expect("request body parses");
		assert_eq!(params["prompt_cache_key"], "x".repeat(64));
	}

	#[tokio::test]
	async fn disables_server_side_response_storage() {
		let _guard = env_lock();
		clear_azure_env();
		let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
		let model = catalog_model("gpt-4o-mini");
		let options = base_options(responses_fetch(sse_body(&completion_events("resp_test")), requests.clone()));
		let azure = AzureOpenAIResponsesOptions {
			azure_base_url: Some("https://my-resource.openai.azure.com".to_string()),
			..Default::default()
		};
		let result = stream_with_options(&model, &normalize(&user_context("hello")), &options, &azure)
			.into_result()
			.await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let request = &requests.lock().expect("lock")[0];
		let body = match &request.body {
			HttpBody::Bytes(bytes) => String::from_utf8_lossy(bytes).to_string(),
			_ => String::new(),
		};
		let params: Value = serde_json::from_str(&body).expect("request body parses");
		assert_eq!(params["store"], false);
	}

	#[tokio::test]
	async fn honors_supports_strict_mode_false() {
		let _guard = env_lock();
		clear_azure_env();
		let mut model = catalog_model("gpt-4o-mini");
		let mut compat = model.compat.clone().unwrap_or_default();
		compat.supports_strict_mode = Some(false);
		model.compat = Some(compat);

		let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
		let mut options = base_options(responses_fetch(sse_body(&completion_events("resp_test")), requests.clone()));
		options.base.api_key = Some("test-api-key".to_string());
		let azure = AzureOpenAIResponsesOptions {
			azure_base_url: Some("https://my-resource.openai.azure.com".to_string()),
			..Default::default()
		};
		let context = Context {
			system_prompt: None,
			messages: vec![user_message("hello", 1)],
			tools: Some(vec![crate::types::Tool {
				name: "preferred".to_string(),
				description: "Preferred constrained tool".to_string(),
				parameters: crate::utils::typebox_helpers::object_schema(
					[("value".to_string(), json!({ "type": "string" }))].into_iter().collect(),
					&["value"],
				),
				constrained_sampling: Some(crate::types::ConstrainedSamplingSetting::Config(
					crate::types::ConstrainedSamplingConfig::JsonSchema {
						strict: crate::types::StrictSchemaMode::Prefer,
					},
				)),
			}]),
		};
		let result = stream_with_options(&model, &normalize(&context), &options, &azure)
			.into_result()
			.await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		let request = &requests.lock().expect("lock")[0];
		let body = match &request.body {
			HttpBody::Bytes(bytes) => String::from_utf8_lossy(bytes).to_string(),
			_ => String::new(),
		};
		let params: Value = serde_json::from_str(&body).expect("request body parses");
		let tool = &params["tools"][0];
		assert!(tool.get("strict").is_none(), "tool carries strict: {tool}");
	}

	#[tokio::test]
	async fn builds_correct_default_url_from_resource_name_env() {
		let _guard = env_lock();
		clear_azure_env();
		set_env("AZURE_OPENAI_RESOURCE_NAME", "my-resource");
		let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
		let model = catalog_model("gpt-4o-mini");
		let options = base_options(responses_fetch(sse_body(&completion_events("resp_test")), requests.clone()));
		let result = stream(&model, &normalize(&user_context("hello")), &options).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		assert_eq!(
			requests.lock().expect("lock")[0].url,
			"https://my-resource.openai.azure.com/openai/v1/responses?api-version=v1"
		);
	}

	// ---- azure provider stream events (azure-openai-base-url.test.ts) -------

	#[tokio::test]
	async fn forwards_parsed_events_in_order_before_normalizing_the_response() {
		let _guard = env_lock();
		clear_azure_env();
		let events = vec![
			json!({ "type": "response.created", "sequence_number": 0, "response": { "id": "resp_azure" } }),
			json!({ "type": "response.completed", "sequence_number": 1, "response": { "id": "resp_azure", "status": "completed" } }),
		];
		let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
		let model = catalog_model("gpt-4o-mini");
		let mut options = base_options(responses_fetch(sse_body(&events), requests));
		let azure = AzureOpenAIResponsesOptions {
			azure_base_url: Some("https://my-resource.openai.azure.com".to_string()),
			..Default::default()
		};

		let received = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
		let event_models = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
		let hook_events = received.clone();
		let hook_models = event_models.clone();
		options.on_provider_stream_event = Some(Arc::new(move |event: Value, event_model: Model| {
			let hook_events = hook_events.clone();
			let hook_models = hook_models.clone();
			Box::pin(async move {
				hook_events.lock().expect("lock").push(event);
				hook_models
					.lock()
					.expect("lock")
					.push((event_model.provider().to_string(), event_model.id().to_string()));
			})
		}));

		let result = stream_with_options(&model, &normalize(&user_context("hello")), &options, &azure)
			.into_result()
			.await;

		// Upstream additionally asserts object identity (`toBe`) of each
		// forwarded event; Rust hooks receive parsed clones, so equality is
		// the observable equivalent.
		assert_eq!(*received.lock().expect("lock"), events);
		assert_eq!(*event_models.lock().expect("lock"), vec![("azure".to_string(), "gpt-4o-mini".to_string()); 2]);
		assert_eq!(result.stop_reason, StopReason::Stop);
		assert_eq!(result.response_id.as_deref(), Some("resp_azure"));
	}

	// ---- azure user agent (azure-openai-base-url.test.ts) -------------------

	#[tokio::test]
	async fn uses_pi_user_agent_by_default() {
		let headers = capture_request_headers(None).await;
		let agent = header_value(&headers, "User-Agent");
		assert_eq!(agent, Some(get_pi_user_agent().as_str()));
	}

	#[tokio::test]
	async fn lets_explicit_headers_override_the_default_user_agent() {
		let headers = capture_request_headers(Some(
			[("User-Agent".to_string(), Some("custom-agent".to_string()))].into_iter().collect(),
		))
		.await;
		assert_eq!(header_value(&headers, "User-Agent"), Some("custom-agent"));
	}

	fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
		headers
			.iter()
			.find(|(key, _)| key.eq_ignore_ascii_case(name))
			.map(|(_, value)| value.as_str())
	}

	// ---- azure tool choice (azure-openai-tool-choice.test.ts) ---------------

	fn tool_choice_model() -> Model {
		Model {
			base: BaseModelFields {
				id: "test-deployment".to_string(),
				name: "Test Deployment".to_string(),
				api: api_ids::AZURE_OPENAI_RESPONSES.to_string(),
				provider: provider_ids::AZURE.to_string(),
				base_url: "http://127.0.0.1:9/openai/v1".to_string(),
				input: vec![ModelInput::Text],
				input_limits: None,
				cost: ModelCost { rates: ModelCostRates { input: 0.0, output: 0.0, cache_read: 0.0, cache_write: 0.0 }, tiers: None },
				headers: None,
			},
			model_type: None,
			reasoning: false,
			thinking_level_map: None,
			prompt_cache: None,
			context_window: 10_000,
			max_tokens: 1_000,
			sampling_params: None,
			sampling_params_by_thinking_level: None,
			compat: None,
		}
	}

	fn read_tool_context() -> Context {
		Context {
			system_prompt: None,
			messages: vec![user_message("Summarize this", 1)],
			tools: Some(vec![crate::types::Tool {
				name: "read".to_string(),
				description: "Read a file".to_string(),
				parameters: crate::utils::typebox_helpers::object_schema(
					[("path".to_string(), json!({ "type": "string" }))].into_iter().collect(),
					&["path"],
				),
				constrained_sampling: None,
			}]),
		}
	}

	#[tokio::test]
	async fn forwards_provider_specific_tool_choice_while_preserving_tool_definitions() {
		let _guard = env_lock();
		clear_azure_env();
		let captured = Arc::new(std::sync::Mutex::new(None::<Value>));
		let hook_capture = captured.clone();
		let mut options = base_options(responses_fetch(sse_body(&completion_events("resp_test")), Default::default()));
		options.base.on_payload = Some(Arc::new(move |payload: Value, _model: Model| {
			let hook_capture = hook_capture.clone();
			Box::pin(async move {
				// Upstream throws from the hook to end the request after the
				// capture; the Rust hook keeps the payload and lets the fake
				// fetch terminate the stream.
				*hook_capture.lock().expect("lock") = Some(payload);
				None
			})
		}));

		let model = tool_choice_model();
		stream_with_options(
			&model,
			&normalize(&read_tool_context()),
			&options,
			&AzureOpenAIResponsesOptions { tool_choice: Some(json!("required")), ..Default::default() },
		)
		.into_result()
		.await;

		let payload = captured.lock().expect("lock").clone().expect("payload captured");
		assert_eq!(payload["tool_choice"], "required");
		assert_eq!(payload["tools"].as_array().expect("tools").len(), 1);
	}

	#[tokio::test]
	async fn forwards_provider_neutral_tool_choice_from_simple_options() {
		let _guard = env_lock();
		clear_azure_env();
		let captured = Arc::new(std::sync::Mutex::new(None::<Value>));
		let hook_capture = captured.clone();
		let mut options = crate::types::SimpleStreamOptions::default();
		options.base.base.api_key = Some("test-key".to_string());
		options.base.base.fetch = Some(responses_fetch(sse_body(&completion_events("resp_test")), Default::default()));
		options.base.base.on_payload = Some(Arc::new(move |payload: Value, _model: Model| {
			let hook_capture = hook_capture.clone();
			Box::pin(async move {
				*hook_capture.lock().expect("lock") = Some(payload);
				None
			})
		}));
		options.tool_choice = Some(ToolChoice::None);

		let model = tool_choice_model();
		stream_simple(&model, &normalize(&read_tool_context()), &options).into_result().await;

		let payload = captured.lock().expect("lock").clone().expect("payload captured");
		assert_eq!(payload["tool_choice"], "none");
		assert_eq!(payload["tools"].as_array().expect("tools").len(), 1);
	}

	// ---- reasoning replay (azure-openai-responses-reasoning-replay.test.ts) -

	fn replay_model() -> Model {
		Model {
			base: BaseModelFields {
				id: "gpt-5-mini".to_string(),
				name: "GPT-5 Mini".to_string(),
				api: api_ids::AZURE_OPENAI_RESPONSES.to_string(),
				provider: provider_ids::AZURE.to_string(),
				base_url: "https://example.invalid".to_string(),
				input: vec![ModelInput::Text],
				input_limits: None,
				cost: ModelCost { rates: ModelCostRates { input: 0.0, output: 0.0, cache_read: 0.0, cache_write: 0.0 }, tiers: None },
				headers: None,
			},
			model_type: None,
			reasoning: true,
			thinking_level_map: None,
			prompt_cache: None,
			context_window: 400_000,
			max_tokens: 128_000,
			sampling_params: None,
			sampling_params_by_thinking_level: None,
			compat: None,
		}
	}

	fn replay_output(model: &Model) -> AssistantMessage {
		AssistantMessage {
			content: Vec::new(),
			api: model.api().to_string(),
			provider: model.provider().to_string(),
			model: model.id().to_string(),
			response_model: None,
			response_id: None,
			provider_thinking_level: None,
			thinking_level: None,
			diagnostics: None,
			usage: Usage::default(),
			stop_reason: StopReason::Pending,
			deferred: None,
			error_message: None,
			raw_stop_reason: None,
			end_turn: None,
			timestamp: now_ms(),
		}
	}

	fn get_replayed_reasoning(model: &Model, assistant: AssistantMessage) -> Value {
		let context = normalize_context(&Context {
			system_prompt: None,
			messages: vec![
				user_message("first", now_ms() - 1),
				Message::Assistant(assistant),
				user_message("follow-up", now_ms()),
			],
			tools: None,
		});
		let input = convert_responses_messages(model, &context, &BTreeSet::from(["azure".to_string()]), None)
			.expect("input converts");
		input
			.into_iter()
			.find(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
			.expect("reasoning item replayed")
	}

	async fn run_replay(done_item: Value, completed_item: Value) -> AssistantMessage {
		let model = replay_model();
		let mut output = replay_output(&model);
		let events = vec![
			Ok(json!({
				"type": "response.output_item.added",
				"output_index": 0,
				"sequence_number": 0,
				"item": { "type": "reasoning", "id": done_item["id"], "summary": [] }
			})),
			Ok(json!({
				"type": "response.output_item.done",
				"output_index": 0,
				"sequence_number": 1,
				"item": done_item
			})),
			Ok(json!({
				"type": "response.completed",
				"sequence_number": 2,
				"response": { "id": "resp_test", "status": "completed", "output": [completed_item] }
			})),
		];
		let mut event_stream = futures_util::stream::iter(events);
		process_responses_stream(&mut event_stream, &mut output, &AssistantMessageEventStream::new(), &model, None)
			.await
			.expect("stream processes");
		output
	}

	#[tokio::test]
	async fn preserves_existing_encrypted_content_from_output_item_done() {
		let done_item = json!({
			"type": "reasoning",
			"id": "rs_done",
			"summary": [],
			"encrypted_content": "from-output-item-done"
		});
		let completed_item = {
			let mut item = done_item.clone();
			item["encrypted_content"] = json!("from-response-completed");
			item
		};

		let output = run_replay(done_item, completed_item).await;
		let reasoning = get_replayed_reasoning(&replay_model(), output);
		assert_eq!(reasoning["type"], "reasoning");
		assert_eq!(reasoning["id"], "rs_done");
		assert_eq!(reasoning["encrypted_content"], "from-output-item-done");
	}

	#[tokio::test]
	async fn fills_encrypted_content_when_output_item_done_omitted_it() {
		let done_item = json!({
			"type": "reasoning",
			"id": "rs_missing",
			"summary": []
		});
		let completed_item = {
			let mut item = done_item.clone();
			item["encrypted_content"] = json!("from-response-completed");
			item
		};

		let output = run_replay(done_item, completed_item).await;
		let reasoning = get_replayed_reasoning(&replay_model(), output);
		assert_eq!(reasoning["type"], "reasoning");
		assert_eq!(reasoning["id"], "rs_missing");
		assert_eq!(reasoning["encrypted_content"], "from-response-completed");
	}
}
