//! Port of `packages/ai/src/api/google-vertex.ts` — the `google-vertex` API
//! adapter over Vertex AI.
//!
//! Upstream drives the `@google/genai` SDK with `vertexai: true` (API key
//! express mode, or project/location with ADC). The Rust port hand-rolls the
//! equivalent requests over the [`crate::http::FetchFn`] seam:
//!
//! - API-key (express) mode:
//!   `POST https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:streamGenerateContent?alt=sse`
//!   with `x-goog-api-key`.
//! - ADC mode:
//!   `POST https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:streamGenerateContent?alt=sse`
//!   with `Authorization: Bearer <token>`.
//! - Custom `model.baseUrl` (`httpOptions.baseUrl` +
//!   `ResourceScope.COLLECTION`): the SDK skips the project/location path
//!   prefix, so `{baseUrl}/{v1}/publishers/google/models/...`; when the
//!   baseUrl already contains an API-version segment the version is
//!   elided (`httpOptions.apiVersion: ""`).
//!
//! ADC credentials come from `GOOGLE_APPLICATION_CREDENTIALS` (or the gcloud
//! well-known file): service-account JSON is exchanged via an RS256 JWT at
//! `https://oauth2.googleapis.com/token`; authorized-user JSON (gcloud
//! `application-default login`) uses the refresh-token grant — the same two
//! token requests `google-auth-library` issues.
//!
//! # Deviations (file ledger)
//!
//! - **D15** — per-API options cannot ride `StreamOptions` (types.rs frozen;
//!   deviation follows crate D2 pattern): [`GoogleVertexOptions`] lives here
//!   with upstream-exact fields, `stream`/`stream_simple` build its default
//!   and delegate to [`stream_with_options`]/[`stream_simple_with_options`];
//!   final integration should add the nested `StreamOptions` group (reported
//!   in reexports). The `ProviderStreams` entry uses the group `Default`.
//! - **D16** — upstream rejects custom fetches
//!   (`Custom fetch is not supported by the Google Vertex adapter`) because
//!   the SDK owns HTTP; over the explicit [`FetchFn`] seam an injected fetch
//!   IS the transport (crate D4), so `options.base.fetch` is consumed like
//!   every other adapter and the guard is unreachable (the fetch-option port
//!   scopes those upstream cases out).
//! - **D17** — the SDK's `x-goog-api-client` library-telemetry header is not
//!   reproduced; `User-Agent` (pi), `Content-Type` and the auth headers are.
//! - **D18** — non-2xx responses become
//!   [`crate::http::ProviderHttpError`] with status/headers/body; the error
//!   message is minimal (`status` + raw body) rather than the SDK's
//!   `ApiError` string.
//! - **D19** — when ADC resolves to no readable credentials file the adapter
//!   fails fast with google-auth-library's message
//!   (`Could not load the default credentials. …`) instead of probing the GCE
//!   metadata server.
//! - **D20** — `toolCallCounter` is a process-global atomic; ids synthesize
//!   as `{name}_{Date.now()}_{counter}` exactly as upstream.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::api::google_shared::{
	GoogleApiThinkingLevel, GoogleContent, GoogleGenerateContentParameters, GoogleSseStream, GoogleToolChoice,
	convert_messages, convert_tools, get_disabled_google_thinking_config, get_google_vertex_budget,
	google_retry_options, is_thinking_part, map_stop_reason, resolve_google_function_calling_mode,
	resolve_google_thinking_level, retain_thought_signature, retry_google_request,
	supports_google_strict_tool_sampling, to_google_sdk_thinking_level, to_google_thinking_level,
	uses_google_thinking_level,
};
use crate::http::{FetchFn, HttpBody, HttpRequest, HttpResponse, ProviderHttpError};
use crate::models::{calculate_cost, ProviderStreams};
use crate::types::{
	AnyModel, AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream, Model,
	ModelThinkingLevel, ProviderEnv, SimpleStreamOptions, StopReason, StreamDoneReason, StreamErrorReason,
	StreamOptions, TextContent, ThinkingContent, ThinkingLevel, Tool, ToolCall, ToolChoice, TranscriptContext, Usage,
};
use crate::utils::diagnostics::now_ms;
use crate::utils::error_body::{format_provider_error, normalize_provider_error};
use crate::utils::event_stream::{done_event, error_event};
use crate::utils::headers::provider_headers_to_record;
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::text::get_system_message_text;
use crate::utils::transcript::{collapse_system_messages, get_current_tools, get_initial_system_message};

/// Upstream `GoogleVertexOptions`.
#[derive(Clone, Default)]
pub struct GoogleVertexOptions {
	/// `"auto" | "none" | "any"`.
	pub tool_choice: Option<GoogleToolChoice>,
	/// Thinking request; `-1` budget means dynamic, `0` disables.
	pub thinking: Option<GoogleThinkingOptions>,
	pub project: Option<String>,
	pub location: Option<String>,
}

/// Upstream `GoogleVertexOptions.thinking`.
#[derive(Clone, Default)]
pub struct GoogleThinkingOptions {
	pub enabled: bool,
	pub budget_tokens: Option<i64>,
	pub level: Option<GoogleApiThinkingLevel>,
}

const API_VERSION: &str = "v1";
const GCP_VERTEX_CREDENTIALS_MARKER: &str = "gcp-vertex-credentials";

/// gcloud ADC well-known file (upstream `VERTEX_ADC_PATH`, providers side).
const VERTEX_ADC_PATH: &str = "~/.config/gcloud/application_default_credentials.json";

/// Upstream `TOKEN_URL` of google-auth-library.
const OAUTH_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";

/// Upstream `google-auth-library` ADC failure message.
const COULD_NOT_LOAD_DEFAULT_CREDENTIALS: &str = "Could not load the default credentials. Browse to https://cloud.google.com/docs/authentication/getting-started for more information.";

/// Counter for generating unique tool call IDs (upstream module-level
/// `toolCallCounter`).
static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Upstream `stream` export for the google-vertex API (options group
/// defaulted; deviation D15).
pub fn stream(model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
	stream_with_options(model, context, options, &GoogleVertexOptions::default())
}

/// [`stream`] with the per-API options group (deviation D15).
pub fn stream_with_options(
	model: &Model,
	context: &TranscriptContext,
	base: &StreamOptions,
	vertex: &GoogleVertexOptions,
) -> AssistantMessageEventStream {
	let stream = AssistantMessageEventStream::new();
	let normalized_context = collapse_system_messages(context);

	let model = model.clone();
	let base = base.clone();
	let vertex = vertex.clone();
	let inner_stream = stream.clone();
	tokio::spawn(async move {
		run_stream(model, normalized_context, base, vertex, inner_stream).await;
	});

	stream
}

#[allow(clippy::too_many_lines)]
async fn run_stream(
	model: Model,
	normalized_context: TranscriptContext,
	options: StreamOptions,
	vertex: GoogleVertexOptions,
	stream: AssistantMessageEventStream,
) {
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
		usage: Default::default(),
		stop_reason: StopReason::Pending,
		deferred: None,
		error_message: None,
		raw_stop_reason: None,
		end_turn: None,
		timestamp: now_ms(),
	};

	let result: Result<(), String> = async {
		// Create the client using either a Vertex API key, if provided, or ADC
		// with project and location.
		let api_key = resolve_api_key(&options);
		let project = if api_key.is_none() { Some(resolve_project(&vertex, &options)?) } else { None };
		let location = if api_key.is_none() { Some(resolve_location(&vertex, &options)?) } else { None };

		let params = build_params(&model, &normalized_context, &options, &vertex)?;
		let mut params_value = serde_json::to_value(&params).map_err(|error| error.to_string())?;
		if let Some(hook) = &options.base.on_payload {
			if let Some(next) = hook(params_value.clone(), model.clone()).await {
				params_value = next;
			}
		}

		let url = build_url(&model, project.as_deref(), location.as_deref());
		let headers = build_headers(&model, options.base.headers.as_ref());
		let body_bytes = serde_json::to_vec(&params_value).map_err(|error| error.to_string())?;
		let timeout_ms = options.base.timeout_ms;
		let signal = options.base.signal.clone();
		let env = options.base.env.clone();

		let fetch: FetchFn = options.base.fetch.clone().unwrap_or_default();
		let response = retry_google_request(
			|| {
				let mut request = HttpRequest::new("POST", url.clone());
				request.headers = headers.clone();
				request.body = HttpBody::Bytes(body_bytes.clone());
				request.timeout_ms = timeout_ms;
				request.signal = signal.clone();
				let fetch = fetch.clone();
				let env = env.clone();
				let api_key = api_key.clone();
				let abort = signal.clone();
				async move {
					let request = match api_key {
						Some(api_key) => request
							.header("content-type", "application/json")
							.header("x-goog-api-key", api_key),
						None => {
							let (token, quota_project) =
								resolve_adc_access_token(env.as_ref(), &fetch, abort.as_ref()).await?;
							let mut request = request
								.header("content-type", "application/json")
								.header("authorization", format!("Bearer {token}"));
							if let Some(quota_project) = quota_project {
								request = request.header("x-goog-user-project", quota_project);
							}
							request
						}
					};
					match fetch.send(request).await {
						Ok(response) if response.ok() => Ok(response),
						Ok(response) => Err(response_to_error(response).await),
						Err(error) => Err(error),
					}
				}
			},
			google_retry_options(&options),
		)
		.await
		.map_err(|error| error.message)?;

		stream.push(AssistantMessageEvent::Start { partial: output.clone() });

		let body = match response.body {
			HttpBody::Stream(body_stream) => body_stream,
			other => Box::pin(futures_util::stream::once(async move { other.collect().await })),
		};
		let mut chunks = GoogleSseStream::new(body, options.base.signal.clone());

		// The live block: its index in `output.content` and kind (upstream
		// keeps a `currentBlock` reference).
		enum BlockKind {
			Text,
			Thinking,
		}
		let mut current_block: Option<(usize, BlockKind)> = None;
		let block_index = |blocks: &mut Vec<AssistantContent>| (blocks.len() - 1) as u32;

		let chunks_result: Result<(), String> = 'chunks: {
			while let Some(chunk) = futures_util::StreamExt::next(&mut chunks).await {
				let chunk = match chunk {
					Ok(chunk) => chunk,
					Err(message) => break 'chunks Err(message),
				};

				if let Some(hook) = &options.on_provider_stream_event {
					hook(chunk.clone(), model.clone()).await;
				}
				let chunk = match crate::api::google_shared::parse_chunk(&chunk) {
					Ok(chunk) => chunk,
					Err(message) => break 'chunks Err(message),
				};

				// Vertex uses the same @google/genai GenerateContentResponse type as
				// Gemini. responseId is documented there as an output-only
				// identifier for each response.
				if output.response_id.as_deref().map(str::is_empty).unwrap_or(true) {
					if let Some(response_id) = chunk.response_id.filter(|id| !id.is_empty()) {
						output.response_id = Some(response_id);
					}
				}

				let candidate = chunk.candidates.as_ref().and_then(|candidates| candidates.first());
				if let Some(parts) = candidate.and_then(|candidate| candidate.content.as_ref()).and_then(|content| content.parts.as_ref()) {
					for part in parts {
						if let Some(text) = &part.text {
							let is_thinking = is_thinking_part(part);
							let needs_new_block = match &current_block {
								None => true,
								Some((_, kind)) => {
									if is_thinking {
										!matches!(kind, BlockKind::Thinking)
									} else {
										!matches!(kind, BlockKind::Text)
									}
								}
							};
							if needs_new_block {
								if let Some((index, kind)) = current_block.take() {
									match kind {
										BlockKind::Text => stream.push(AssistantMessageEvent::TextEnd {
											content_index: index as u32,
											content: block_text(&output.content, index),
											partial: output.clone(),
										}),
										BlockKind::Thinking => stream.push(AssistantMessageEvent::ThinkingEnd {
											content_index: index as u32,
											content: block_thinking(&output.content, index),
											partial: output.clone(),
										}),
									}
								}
								if is_thinking {
									output.content.push(AssistantContent::Thinking(ThinkingContent {
										thinking: String::new(),
										thinking_signature: None,
										redacted: None,
									}));
									let index = block_index(&mut output.content);
									current_block = Some((index as usize, BlockKind::Thinking));
									stream.push(AssistantMessageEvent::ThinkingStart {
										content_index: index,
										partial: output.clone(),
									});
								} else {
									output.content.push(AssistantContent::Text(TextContent {
										text: String::new(),
										text_signature: None,
									}));
									let index = block_index(&mut output.content);
									current_block = Some((index as usize, BlockKind::Text));
									stream.push(AssistantMessageEvent::TextStart {
										content_index: index,
										partial: output.clone(),
									});
								}
							}
							let Some((index, kind)) = &current_block else {
								unreachable!("block just opened");
							};
							let index = *index;
							match kind {
								BlockKind::Thinking => {
									if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(index) {
										block.thinking += text.as_str();
										block.thinking_signature = retain_thought_signature(
											block.thinking_signature.take(),
											part.thought_signature.as_deref(),
										);
									}
									stream.push(AssistantMessageEvent::ThinkingDelta {
										content_index: index as u32,
										delta: text.clone(),
										partial: output.clone(),
									});
								}
								BlockKind::Text => {
									if let Some(AssistantContent::Text(block)) = output.content.get_mut(index) {
										block.text += text.as_str();
										block.text_signature = retain_thought_signature(
											block.text_signature.take(),
											part.thought_signature.as_deref(),
										);
									}
									stream.push(AssistantMessageEvent::TextDelta {
										content_index: index as u32,
										delta: text.clone(),
										partial: output.clone(),
									});
								}
							}
						}

						if let Some(function_call) = &part.function_call {
							if let Some((index, kind)) = current_block.take() {
								match kind {
									BlockKind::Text => stream.push(AssistantMessageEvent::TextEnd {
										content_index: index as u32,
										content: block_text(&output.content, index),
										partial: output.clone(),
									}),
									BlockKind::Thinking => stream.push(AssistantMessageEvent::ThinkingEnd {
										content_index: index as u32,
										content: block_thinking(&output.content, index),
										partial: output.clone(),
									}),
								}
							}

							let provided_id = function_call.id.clone();
							let needs_new_id = provided_id.as_deref().map(str::is_empty).unwrap_or(true)
								|| output.content.iter().any(|block| match block {
									AssistantContent::ToolCall(existing) => {
										Some(existing.id.as_str()) == provided_id.as_deref()
									}
									_ => false,
								});
							let tool_call_id = if needs_new_id {
								format!(
									"{}_{}_{}",
									function_call.name.clone().unwrap_or_default(),
									now_ms(),
									TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed) + 1,
								)
							} else {
								provided_id.expect("checked above")
							};

							let tool_call = ToolCall {
								id: tool_call_id,
								name: function_call.name.clone().unwrap_or_default(),
								arguments: function_call.args.clone().unwrap_or_default(),
								thought_signature: part.thought_signature.clone().filter(|sig| !sig.is_empty()),
								namespace: None,
							};

							output.content.push(AssistantContent::ToolCall(tool_call.clone()));
							let content_index = block_index(&mut output.content);
							stream.push(AssistantMessageEvent::ToolcallStart {
								content_index,
								partial: output.clone(),
							});
							stream.push(AssistantMessageEvent::ToolcallDelta {
								content_index,
								delta: serde_json::to_string(&tool_call.arguments)
									.map_err(|error| error.to_string())?,
								partial: output.clone(),
							});
							stream.push(AssistantMessageEvent::ToolcallEnd {
								content_index,
								tool_call,
								partial: output.clone(),
							});
						}
					}
				}

				if let Some(finish_reason) = candidate.and_then(|candidate| candidate.finish_reason.as_ref()) {
					output.raw_stop_reason = Some(finish_reason.clone());
					output.stop_reason = match map_stop_reason(&crate::api::google_shared::FinishReason::parse(finish_reason))
					{
						Ok(reason) => reason,
						Err(message) => break 'chunks Err(message),
					};
					if output.content.iter().any(|block| matches!(block, AssistantContent::ToolCall(_)))
						&& output.stop_reason == StopReason::Stop
					{
						output.stop_reason = StopReason::ToolUse;
					}
				}

				if let Some(usage) = &chunk.usage_metadata {
					let cached = usage.cached_content_token_count.unwrap_or(0);
					let thoughts = usage.thoughts_token_count.unwrap_or(0);
					output.usage = Usage {
						input: usage.prompt_token_count.unwrap_or(0).saturating_sub(cached),
						output: usage.candidates_token_count.unwrap_or(0).saturating_add(thoughts),
						cache_read: cached,
						cache_write: 0,
						cache_write_1h: None,
						reasoning: Some(thoughts),
						total_tokens: usage.total_token_count.unwrap_or(0),
						cost: Default::default(),
					};
					calculate_cost(&AnyModel::Chat(model.clone()), &mut output.usage);
				}
			}
			Ok(())
		};
		chunks_result?;

		if let Some((index, kind)) = current_block {
			match kind {
				BlockKind::Text => stream.push(AssistantMessageEvent::TextEnd {
					content_index: index as u32,
					content: block_text(&output.content, index),
					partial: output.clone(),
				}),
				BlockKind::Thinking => stream.push(AssistantMessageEvent::ThinkingEnd {
					content_index: index as u32,
					content: block_thinking(&output.content, index),
					partial: output.clone(),
				}),
			}
		}

		if options.base.signal.as_ref().is_some_and(|signal| signal.aborted()) {
			return Err("Request was aborted".to_string());
		}

		if output.stop_reason == StopReason::Pending {
			return Err("Google Vertex stream ended without a finish reason".to_string());
		}
		if output.stop_reason == StopReason::Aborted || output.stop_reason == StopReason::Error {
			let error_message = match &output.raw_stop_reason {
				Some(raw) => format!("Provider stopped with: {raw}"),
				None => "An unknown error occurred".to_string(),
			};
			return Err(error_message);
		}

		let reason = match output.stop_reason {
			StopReason::Stop => StreamDoneReason::Stop,
			StopReason::Length => StreamDoneReason::Length,
			StopReason::ToolUse => StreamDoneReason::ToolUse,
			_ => StreamDoneReason::Stop,
		};
		stream.push(done_event(output.clone(), reason));
		stream.end(Some(output.clone()));
		Ok(())
	}
	.await;

	if let Err(message) = result {
		let aborted = options.base.signal.as_ref().is_some_and(|signal| signal.aborted());
		output.stop_reason = if aborted { StopReason::Aborted } else { StopReason::Error };
		output.error_message = Some(format_provider_error(&normalize_provider_error(&ProviderHttpError::new(message)), None));
		let reason = if aborted { StreamErrorReason::Aborted } else { StreamErrorReason::Error };
		stream.push(error_event(output.clone(), reason));
		stream.end(None);
	}
}

fn block_text(content: &[AssistantContent], index: usize) -> String {
	match content.get(index) {
		Some(AssistantContent::Text(block)) => block.text.clone(),
		_ => String::new(),
	}
}

fn block_thinking(content: &[AssistantContent], index: usize) -> String {
	match content.get(index) {
		Some(AssistantContent::Thinking(block)) => block.thinking.clone(),
		_ => String::new(),
	}
}

/// Converts a non-2xx response into the retry error (upstream SDK `ApiError`;
/// deviation D18).
async fn response_to_error(response: HttpResponse) -> ProviderHttpError {
	let status = response.status;
	let headers = response.headers.clone();
	let body = response.body.collect().await.unwrap_or_default();
	let body_text = String::from_utf8_lossy(&body).to_string();
	let message = if body_text.is_empty() {
		format!("Request failed with status {status}")
	} else {
		format!("{body_text} (status {status})")
	};
	ProviderHttpError { message, status: Some(status), headers, body: Some(body_text), source: None }
}

/// Upstream `streamSimple` export.
pub fn stream_simple(model: &Model, context: &TranscriptContext, options: &SimpleStreamOptions) -> AssistantMessageEventStream {
	match derive_vertex_options(model, options) {
		Ok(vertex) => {
			let base = crate::api::simple_options::build_base_options(model, context, options, None);
			stream_with_options(model, context, &base, &vertex)
		}
		Err(message) => error_stream(model, &message),
	}
}

/// [`stream_simple`] with the per-API options group prebuilt (deviation D15).
pub fn stream_simple_with_options(
	model: &Model,
	context: &TranscriptContext,
	base: &SimpleStreamOptions,
	vertex: &GoogleVertexOptions,
) -> AssistantMessageEventStream {
	let base_options = crate::api::simple_options::build_base_options(model, context, base, None);
	stream_with_options(model, context, &base_options, vertex)
}

/// The `{ ...base, toolChoice, thinking }` derivation of upstream
/// `streamSimple`.
fn derive_vertex_options(model: &Model, options: &SimpleStreamOptions) -> Result<GoogleVertexOptions, String> {
	let tool_choice = options.tool_choice.map(|tool_choice| match tool_choice {
		ToolChoice::Auto => GoogleToolChoice::Auto,
		ToolChoice::None => GoogleToolChoice::None,
	});

	let disabled = || {
		Ok(GoogleVertexOptions {
			tool_choice,
			thinking: Some(GoogleThinkingOptions { enabled: false, ..Default::default() }),
			project: None,
			location: None,
		})
	};

	let Some(reasoning) = options.reasoning else {
		return disabled();
	};

	let clamped_reasoning = crate::models::clamp_thinking_level(model, ModelThinkingLevel::from(reasoning));
	if clamped_reasoning == ModelThinkingLevel::Off {
		return disabled();
	}
	let resolved_level = resolve_google_thinking_level(model, thinking_level_of(clamped_reasoning))?;

	if uses_google_thinking_level(model) {
		return Ok(GoogleVertexOptions {
			tool_choice,
			thinking: Some(GoogleThinkingOptions {
				enabled: true,
				level: Some(to_google_thinking_level(resolved_level)),
				budget_tokens: None,
			}),
			project: None,
			location: None,
		});
	}

	Ok(GoogleVertexOptions {
		tool_choice,
		thinking: Some(GoogleThinkingOptions {
			enabled: true,
			budget_tokens: Some(get_google_vertex_budget(
				model,
				resolved_level,
				options.thinking_budgets.as_ref(),
			)),
			level: None,
		}),
		project: None,
		location: None,
	})
}

fn thinking_level_of(level: ModelThinkingLevel) -> ThinkingLevel {
	match level {
		ModelThinkingLevel::Minimal => ThinkingLevel::Minimal,
		ModelThinkingLevel::Low => ThinkingLevel::Low,
		ModelThinkingLevel::Medium => ThinkingLevel::Medium,
		ModelThinkingLevel::High => ThinkingLevel::High,
		ModelThinkingLevel::Xhigh => ThinkingLevel::Xhigh,
		ModelThinkingLevel::Max => ThinkingLevel::Max,
		ModelThinkingLevel::Off => ThinkingLevel::Low,
	}
}

/// Synchronous setup failure surfaced as a terminated error stream (crate D1).
fn error_stream(model: &Model, message: &str) -> AssistantMessageEventStream {
	let stream = AssistantMessageEventStream::new();
	let output = AssistantMessage {
		content: Vec::new(),
		api: model.api().to_string(),
		provider: model.provider().to_string(),
		model: model.id().to_string(),
		response_model: None,
		response_id: None,
		provider_thinking_level: None,
		thinking_level: None,
		diagnostics: None,
		usage: Default::default(),
		stop_reason: StopReason::Error,
		deferred: None,
		error_message: Some(message.to_string()),
		raw_stop_reason: None,
		end_turn: None,
		timestamp: now_ms(),
	};
	stream.push(error_event(output, StreamErrorReason::Error));
	stream.end(None);
	stream
}

// ---------------------------------------------------------------------------
// Client/URL/auth resolution
// ---------------------------------------------------------------------------

/// Upstream `resolveApiKey`.
fn resolve_api_key(options: &StreamOptions) -> Option<String> {
	let api_key = options.base.api_key.as_deref().map(str::trim);
	match api_key {
		None => None,
		Some(api_key) if api_key.is_empty() || api_key == GCP_VERTEX_CREDENTIALS_MARKER || is_placeholder_api_key(api_key) => {
			None
		}
		Some(api_key) => Some(api_key.to_string()),
	}
}

fn is_placeholder_api_key(api_key: &str) -> bool {
	let bytes = api_key.as_bytes();
	bytes.len() >= 2 && bytes[0] == b'<' && bytes[bytes.len() - 1] == b'>'
}

/// Upstream `resolveProject`.
fn resolve_project(vertex: &GoogleVertexOptions, options: &StreamOptions) -> Result<String, String> {
	let project = vertex
		.project
		.clone()
		.filter(|project| !project.is_empty())
		.or_else(|| get_provider_env_value("GOOGLE_CLOUD_PROJECT", options.base.env.as_ref()))
		.or_else(|| get_provider_env_value("GCLOUD_PROJECT", options.base.env.as_ref()));
	match project {
		Some(project) => Ok(project),
		None => Err(
			"Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
				.to_string(),
		),
	}
}

/// Upstream `resolveLocation`.
fn resolve_location(vertex: &GoogleVertexOptions, options: &StreamOptions) -> Result<String, String> {
	let location = vertex
		.location
		.clone()
		.filter(|location| !location.is_empty())
		.or_else(|| get_provider_env_value("GOOGLE_CLOUD_LOCATION", options.base.env.as_ref()));
	match location {
		Some(location) => Ok(location),
		None => Err("Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.".to_string()),
	}
}

/// Upstream `resolveCustomBaseUrl`.
fn resolve_custom_base_url(base_url: &str) -> Option<String> {
	let trimmed = base_url.trim();
	if trimmed.is_empty() || trimmed.contains("{location}") {
		return None;
	}
	Some(trimmed.to_string())
}

/// Upstream `baseUrlIncludesApiVersion`.
fn base_url_includes_api_version(base_url: &str) -> bool {
	match url::Url::parse(base_url) {
		Ok(parsed) => parsed.path().split('/').any(|part| is_api_version_segment(part)),
		Err(_) => {
			// `/(?:^|\/)v\d+(?:beta\d*)?(?:\/|$)/`
			(0..base_url.len()).any(|index| matches_api_version_at(base_url, index))
		}
	}
}

fn matches_api_version_at(base_url: &str, index: usize) -> bool {
	let bytes = base_url.as_bytes();
	if index != 0 && bytes[index - 1] != b'/' {
		return false;
	}
	let Some(rest) = base_url[index..].strip_prefix('v') else {
		return false;
	};
	let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
	if digits.is_empty() {
		return false;
	}
	let after = &rest[digits.len()..];
	let after = after.strip_prefix("beta").unwrap_or(after);
	let beta_digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
	let after = &after[beta_digits.len()..];
	after.is_empty() || after.starts_with('/')
}

fn is_api_version_segment(part: &str) -> bool {
	let Some(rest) = part.strip_prefix('v') else { return false };
	let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
	if digits.is_empty() {
		return false;
	}
	let after = &rest[digits.len()..];
	let after = match after.strip_prefix("beta") {
		Some(after_beta) => {
			let beta_digits: String = after_beta.chars().take_while(|c| c.is_ascii_digit()).collect();
			&after_beta[beta_digits.len()..]
		}
		None => after,
	};
	after.is_empty()
}

/// The SDK's observable request URL (`constructUrl` + `tModel`): the path is
/// always `publishers/google/models/{model}:streamGenerateContent?alt=sse`;
/// the project/location prefix is dropped for custom baseUrls
/// (`ResourceScope.COLLECTION`) and API-key express mode uses
/// `https://aiplatform.googleapis.com`.
fn build_url(model: &Model, project: Option<&str>, location: Option<&str>) -> String {
	let id = model.id();
	let path = format!("publishers/google/models/{id}:streamGenerateContent?alt=sse");
	match resolve_custom_base_url(&model.base.base_url) {
		Some(base) => {
			let base = base.trim_end_matches('/');
			if base_url_includes_api_version(base) {
				format!("{base}/{path}")
			} else {
				format!("{base}/{API_VERSION}/{path}")
			}
		}
		None => match (project, location) {
			(Some(project), Some(location)) => format!(
				"https://{location}-aiplatform.googleapis.com/{API_VERSION}/projects/{project}/locations/{location}/{path}"
			),
			_ => format!("https://aiplatform.googleapis.com/{API_VERSION}/{path}"),
		},
	}
}

fn build_headers(model: &Model, options_headers: Option<&crate::types::ProviderHeaders>) -> Vec<(String, String)> {
	let mut source = crate::types::ProviderHeaders::new();
	source.insert("User-Agent".to_string(), Some(get_pi_user_agent()));
	for (name, value) in model.base.headers.iter().flatten() {
		source.insert(name.clone(), Some(value.clone()));
	}
	if let Some(headers) = options_headers {
		for (name, value) in headers {
			source.insert(name.clone(), value.clone());
		}
	}
	let merged = provider_headers_to_record(&[Some(&source)]).unwrap_or_default();
	merged.into_iter().collect()
}

// ---------------------------------------------------------------------------
// ADC access tokens
// ---------------------------------------------------------------------------

/// The ADC access token plus the optional `x-goog-user-project` quota id.
type AdcToken = (String, Option<String>);

/// Resolve an ADC access token from `GOOGLE_APPLICATION_CREDENTIALS` (or the
/// gcloud well-known file). Service-account JSON is exchanged via an RS256
/// JWT; authorized-user JSON via the refresh-token grant — the same requests
/// google-auth-library issues (deviation D19: no metadata-server probe).
async fn resolve_adc_access_token(
	env: Option<&ProviderEnv>,
	fetch: &FetchFn,
	signal: Option<&crate::utils::abort::AbortSignal>,
) -> Result<AdcToken, ProviderHttpError> {
	if let Some(signal) = signal {
		if signal.aborted() {
			return Err(ProviderHttpError::new("Request aborted"));
		}
	}
	let path = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env)
		.unwrap_or_else(|| VERTEX_ADC_PATH.to_string());
	let path = expand_tilde(&path);
	let contents = match tokio::fs::read_to_string(&path).await {
		Ok(contents) => contents,
		Err(_) => return Err(ProviderHttpError::new(COULD_NOT_LOAD_DEFAULT_CREDENTIALS)),
	};
	let credentials: serde_json::Value = match serde_json::from_str(&contents) {
		Ok(credentials) => credentials,
		Err(_) => return Err(ProviderHttpError::new(COULD_NOT_LOAD_DEFAULT_CREDENTIALS)),
	};

	let quota_project = credentials
		.get("quota_project_id")
		.and_then(serde_json::Value::as_str)
		.map(str::to_string);

	if let (Some(private_key), Some(client_email)) = (
		credentials.get("private_key").and_then(serde_json::Value::as_str),
		credentials.get("client_email").and_then(serde_json::Value::as_str),
	) {
		let iat = (now_ms() as f64 / 1000.0).floor() as i64;
		let assertion = sign_service_account_jwt(private_key, client_email, iat)
			.map_err(|message| ProviderHttpError::new(message))?;
		let form = format!(
			"grant_type={GRANT_TYPE_JWT}&assertion={assertion}"
		);
		let response = post_token_form(fetch, form, signal).await?;
		let token = response
			.get("access_token")
			.and_then(serde_json::Value::as_str)
			.ok_or_else(|| ProviderHttpError::new("Token response does not contain an access token"))?;
		return Ok((token.to_string(), quota_project));
	}

	if let (Some(refresh_token), Some(client_id), Some(client_secret)) = (
		credentials.get("refresh_token").and_then(serde_json::Value::as_str),
		credentials.get("client_id").and_then(serde_json::Value::as_str),
		credentials.get("client_secret").and_then(serde_json::Value::as_str),
	) {
		let form = url::form_urlencoded::Serializer::new(String::new())
			.append_pair("grant_type", "refresh_token")
			.append_pair("refresh_token", refresh_token)
			.append_pair("client_id", client_id)
			.append_pair("client_secret", client_secret)
			.finish();
		let response = post_token_form(fetch, form, signal).await?;
		let token = response
			.get("access_token")
			.and_then(serde_json::Value::as_str)
			.ok_or_else(|| ProviderHttpError::new("Token response does not contain an access token"))?;
		return Ok((token.to_string(), quota_project));
	}

	Err(ProviderHttpError::new(COULD_NOT_LOAD_DEFAULT_CREDENTIALS))
}

const GRANT_TYPE_JWT: &str = "urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer";

async fn post_token_form(
	fetch: &FetchFn,
	form: String,
	signal: Option<&crate::utils::abort::AbortSignal>,
) -> Result<serde_json::Value, ProviderHttpError> {
	let mut request = HttpRequest::new("POST", OAUTH_TOKEN_URL);
	request = request
		.header("content-type", "application/x-www-form-urlencoded");
	request.body = HttpBody::Bytes(form.into_bytes());
	request.signal = signal.cloned();
	let response = fetch.send(request).await?;
	if !response.ok() {
		return Err(response_to_error(response).await);
	}
	let body = response.body.collect().await?;
	serde_json::from_slice(&body)
		.map_err(|error| ProviderHttpError::new(format!("Failed to parse token response: {error}")))
}

/// Sign an RS256 JWT for the service-account token exchange. The `rsa`
/// dependency is pulled without its `sha2` feature (no
/// `AssociatedObjectIdentifier` impl), so the PKCS#1 v1.5 `DigestInfo` prefix
/// is built here and signed via `Pkcs1v15Sign::new_unprefixed` — the same
/// RS256 signature bytes `SigningKey::<Sha256>::new` would produce.
fn sign_service_account_jwt(private_key_pem: &str, client_email: &str, iat: i64) -> Result<String, String> {
	use rsa::pkcs1::DecodeRsaPrivateKey;
	use rsa::pkcs8::DecodePrivateKey;
	use sha2::{Digest, Sha256};

	/// `DigestInfo ::= SEQUENCE { algid SEQUENCE { OID 2.16.840.1.101.3.4.2.1,
	/// NULL }, OCTET STRING }` prefix for SHA-256.
	const SHA256_DIGEST_INFO_PREFIX: [u8; 19] = [
		0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0x04,
		0x20,
	];

	let key = match rsa::RsaPrivateKey::from_pkcs8_pem(private_key_pem) {
		Ok(key) => key,
		Err(_) => rsa::RsaPrivateKey::from_pkcs1_pem(private_key_pem)
			.map_err(|error| format!("Failed to load service account private key: {error}"))?,
	};

	let header = base64url_json(&serde_json::json!({ "alg": "RS256", "typ": "JWT" }));
	let claims = base64url_json(&serde_json::json!({
		"iss": client_email,
		"scope": "https://www.googleapis.com/auth/cloud-platform",
		"aud": OAUTH_TOKEN_URL,
		"iat": iat,
		"exp": iat + 3600,
	}));
	let signing_input = format!("{header}.{claims}");
	let digest = Sha256::digest(signing_input.as_bytes());
	let mut digest_info = SHA256_DIGEST_INFO_PREFIX.to_vec();
	digest_info.extend_from_slice(&digest);
	let signature = key
		.sign(rsa::pkcs1v15::Pkcs1v15Sign::new_unprefixed(), &digest_info)
		.map_err(|error| format!("Failed to sign service-account JWT: {error}"))?;
	Ok(format!("{signing_input}.{}", base64url(&signature)))
}

fn base64url_json(value: &serde_json::Value) -> String {
	base64url(&serde_json::to_vec(value).expect("JWT part serializes"))
}

fn base64url(bytes: &[u8]) -> String {
	use base64::Engine as _;
	base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// Upstream: `resolved.startsWith("~")` → `os.homedir() + resolved.slice(1)`.
/// With `HOME` unset the path is used as-is.
fn expand_tilde(path: &str) -> String {
	if let Some(rest) = path.strip_prefix('~') {
		if let Some(home) = std::env::var_os("HOME") {
			return format!("{}{rest}", home.to_string_lossy());
		}
	}
	path.to_string()
}

/// Upstream `buildParams`.
fn build_params(
	model: &Model,
	context: &TranscriptContext,
	options: &StreamOptions,
	vertex: &GoogleVertexOptions,
) -> Result<GoogleGenerateContentParameters, String> {
	let contents: Vec<GoogleContent> = convert_messages(model, context);
	let initial_system_message = get_initial_system_message(context.messages());
	let current_tools: Vec<Tool> = get_current_tools(context.messages());

	let supports_strict_mode = supports_google_strict_tool_sampling(model.id());
	let function_calling_mode = if !current_tools.is_empty() {
		resolve_google_function_calling_mode(&current_tools, vertex.tool_choice, supports_strict_mode)?
	} else {
		None
	};
	let system_instruction = initial_system_message.map(get_system_message_text).unwrap_or_default();

	let mut config = crate::api::google_shared::GoogleConfig::default();
	if let Some(temperature) = options.temperature {
		config.temperature = Some(temperature);
	}
	if let Some(max_tokens) = options.max_tokens {
		config.max_output_tokens = Some(max_tokens);
	}
	if !system_instruction.is_empty() {
		config.system_instruction = Some(sanitize_surrogates(&system_instruction));
	}
	if !current_tools.is_empty() {
		config.tools = convert_tools(&current_tools, false, supports_strict_mode)?;
	}
	if let Some(mode) = function_calling_mode {
		config.tool_config = Some(crate::api::google_shared::GoogleToolConfig {
			function_calling_config: crate::api::google_shared::GoogleFunctionCallingConfig { mode },
		});
	}

	if let Some(thinking) = &vertex.thinking {
		if thinking.enabled && model.reasoning {
			let mut thinking_config = crate::api::google_shared::GoogleThinkingConfig {
				include_thoughts: Some(true),
				..Default::default()
			};
			if let Some(level) = thinking.level {
				thinking_config.thinking_level = Some(to_google_sdk_thinking_level(level));
			} else if let Some(budget_tokens) = thinking.budget_tokens {
				thinking_config.thinking_budget = Some(budget_tokens);
			}
			config.thinking_config = Some(thinking_config);
		} else if model.reasoning && !thinking.enabled {
			config.thinking_config = Some(get_disabled_google_thinking_config(model));
		}
	}

	if let Some(signal) = &options.base.signal {
		if signal.aborted() {
			return Err("Request aborted".to_string());
		}
		// `config.abortSignal` is the transport signal here.
	}

	Ok(GoogleGenerateContentParameters { model: model.id().to_string(), contents, config })
}

// ---------------------------------------------------------------------------
// ProviderStreams export
// ---------------------------------------------------------------------------

/// The google-vertex API implementation (`googleVertexApi`). No deferred
/// methods are exported upstream, so both presence probes stay `false`.
pub struct GoogleVertexApi;

impl ProviderStreams for GoogleVertexApi {
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

/// Upstream `googleVertexApi` factory.
pub fn google_vertex_api() -> Arc<dyn ProviderStreams> {
	Arc::new(GoogleVertexApi)
}

// ---------------------------------------------------------------------------
// Tests (ports of test/google-vertex-api-key-resolution.test.ts,
// test/google-raw-stop-reason.test.ts vertex cases, and
// test/google-thinking-level-map.test.ts vertex cases)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::{Context, ThinkingBudgets, UserContentField, UserMessage};
	use serde_json::{json, Value};
	use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
	use std::sync::{Arc, LazyLock};
	use {parking_lot::Mutex};

	fn catalog_model(id: &str) -> Model {
		crate::providers::google_vertex_models::google_vertex_models()
			.into_iter()
			.find(|model| model.id() == id)
			.unwrap_or_else(|| panic!("catalog model google-vertex/{id}"))
	}

	fn with_base_url(model: &Model, base_url: &str) -> Model {
		let mut model = model.clone();
		model.base.base_url = base_url.to_string();
		model
	}

	fn with_level_map(model: &Model, thinking_level_map: Value) -> Model {
		let mut model = model.clone();
		model.thinking_level_map = Some(serde_json::from_value(thinking_level_map).expect("thinkingLevelMap parses"));
		model
	}

	fn hello_context() -> TranscriptContext {
		crate::utils::transcript::normalize_context(&Context {
			system_prompt: None,
			messages: vec![crate::types::Message::User(UserMessage {
				content: UserContentField::Text("hello".to_string()),
				timestamp: 0,
			})],
			tools: None,
		})
	}

	fn vertex_options(project: Option<&str>, location: Option<&str>) -> GoogleVertexOptions {
		GoogleVertexOptions {
			project: project.map(str::to_string),
			location: location.map(str::to_string),
			..Default::default()
		}
	}

	fn stream_with_adc(
		model: &Model,
		options: StreamOptions,
		vertex: &GoogleVertexOptions,
	) -> AssistantMessageEventStream {
		stream_with_options(model, &hello_context(), &options, vertex)
	}

	// -- ADC fixture ----------------------------------------------------------

	/// RSA-2048 PEM generated once per process (service-account fixtures need a
	/// real RS256 key; rsa 0.9 has no bundled RNG, so one is provided from the
	/// platform entropy source).
	struct OsEntropy;

	impl rsa::rand_core::RngCore for OsEntropy {
		fn next_u32(&mut self) -> u32 {
			let mut bytes = [0u8; 4];
			getrandom::getrandom(&mut bytes).expect("platform entropy");
			u32::from_le_bytes(bytes)
		}

		fn next_u64(&mut self) -> u64 {
			let mut bytes = [0u8; 8];
			getrandom::getrandom(&mut bytes).expect("platform entropy");
			u64::from_le_bytes(bytes)
		}

		fn fill_bytes(&mut self, dest: &mut [u8]) {
			getrandom::getrandom(dest).expect("platform entropy");
		}

		fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rsa::rand_core::Error> {
			getrandom::getrandom(dest)
				.map_err(|error| rsa::rand_core::Error::new(error))
		}
	}

	impl rsa::rand_core::CryptoRng for OsEntropy {}

	static SERVICE_ACCOUNT_PEM: LazyLock<String> = LazyLock::new(|| {
		use rsa::pkcs8::EncodePrivateKey;
		let mut rng = OsEntropy;
		let key = rsa::RsaPrivateKey::new(&mut rng, 2048).expect("key generation");
		key.to_pkcs8_pem(rsa::pkcs8::LineEnding::LF).expect("pem").to_string()
	});

	fn service_account_pem() -> &'static str {
		&SERVICE_ACCOUNT_PEM
	}

	fn service_account_file(tag: &str) -> String {
		static COUNTER: AtomicU32 = AtomicU32::new(0);
		let path = std::env::temp_dir().join(format!(
			"pi-ai-vertex-test-{}-{}-{}.json",
			std::process::id(),
			tag,
			COUNTER.fetch_add(1, AtomicOrdering::Relaxed),
		));
		std::fs::write(
			&path,
			serde_json::json!({
				"type": "service_account",
				"private_key": service_account_pem(),
				"client_email": "sa@test-project.iam.gserviceaccount.com",
			})
			.to_string(),
		)
		.expect("fixture written");
		path.to_string_lossy().to_string()
	}

	fn adc_env(path: &str) -> ProviderEnv {
		[("GOOGLE_APPLICATION_CREDENTIALS".to_string(), path.to_string())].into_iter().collect()
	}

	/// Recording FetchFn fake: answers the OAuth token endpoint and returns an
	/// SSE body for model requests.
	fn vertex_fetch(
		sse_chunks: &[Value],
		recorded: Arc<Mutex<Vec<Value>>>,
	) -> FetchFn {
		let body = format!(
			"{}",
			sse_chunks
				.iter()
				.map(|chunk| format!("data: {chunk}\n\n"))
				.collect::<String>()
		);
		let body = Arc::new(body.into_bytes());
		FetchFn::new(move |request: HttpRequest| {
			let request_body = match &request.body {
				HttpBody::Bytes(bytes) => String::from_utf8_lossy(bytes).to_string(),
				_ => String::new(),
			};
			recorded.lock().push(json!({
				"url": request.url,
				"headers": request.headers.iter().cloned().collect::<Vec<_>>(),
				"body": serde_json::from_str::<Value>(&request_body).ok(),
			}));
			if request.url.contains("oauth2.googleapis.com/token") {
				let response = HttpResponse {
					status: 200,
					headers: vec![("content-type".to_string(), "application/json".to_string())],
					body: HttpBody::Bytes(
						serde_json::json!({ "access_token": "test-access-token", "expires_in": 3599 })
							.to_string()
							.into_bytes(),
					),
				};
				return Box::pin(async move { Ok(response) });
			}
			let response = HttpResponse {
				status: 200,
				headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
				body: HttpBody::Bytes((*body).clone()),
			};
			Box::pin(async move { Ok(response) })
		})
	}

	fn ok_chunk() -> Value {
		json!({
			"responseId": "vertex-response-id",
			"candidates": [{ "content": { "parts": [{ "text": "ok" }] }, "finishReason": "STOP" }],
			"usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2 },
		})
	}

	fn header_value<'a>(request: &'a Value, name: &str) -> Option<&'a str> {
		request
			.get("headers")
			.and_then(Value::as_array)
			.and_then(|headers| {
				headers.iter().find_map(|entry| {
					entry.as_array().and_then(|pair| {
						if pair.first().and_then(Value::as_str) == Some(name) {
							pair.get(1).and_then(Value::as_str)
						} else {
							None
						}
					})
				})
			})
	}

	async fn run_vertex_stream(
		model: &Model,
		mut options: StreamOptions,
		vertex: &GoogleVertexOptions,
		api_key: Option<&str>,
		env: Option<ProviderEnv>,
	) -> (AssistantMessage, Arc<Mutex<Vec<Value>>>) {
		options.base.api_key = api_key.map(str::to_string);
		options.base.env = env;
		let recorded = Arc::new(Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(vertex_fetch(&[ok_chunk()], recorded.clone()));
		let result = stream_with_adc(model, options, vertex).into_result().await;
		(result, recorded)
	}

	// -- google-vertex api key resolution -------------------------------------

	#[tokio::test]
	async fn falls_back_to_adc_when_options_api_key_is_a_placeholder_marker() {
		let model = catalog_model("gemini-3-flash-preview");
		let credentials = service_account_file("placeholder");
		let options = StreamOptions::default();
		let vertex = vertex_options(Some("test-project"), Some("us-central1"));
		let (result, recorded) =
			run_vertex_stream(&model, options, &vertex, Some("<authenticated>"), Some(adc_env(&credentials))).await;

		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let requests = recorded.lock().clone();
		assert_eq!(requests.len(), 2, "token exchange + model request: {requests:?}");
		let token_request = &requests[0];
		assert!(token_request["url"].as_str().expect("url").contains("oauth2.googleapis.com/token"));
		let model_request = &requests[1];
		let url = model_request["url"].as_str().expect("url");
		assert!(url.starts_with(
			"https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
		));
		assert_eq!(header_value(model_request, "authorization"), Some("Bearer test-access-token"));
		assert!(header_value(model_request, "x-goog-api-key").is_none());
	}

	#[tokio::test]
	async fn falls_back_to_adc_when_options_api_key_is_the_gcp_marker() {
		let model = catalog_model("gemini-3-flash-preview");
		let credentials = service_account_file("gcp-marker");
		let options = StreamOptions::default();
		let vertex = vertex_options(Some("test-project"), Some("us-central1"));
		let (result, recorded) = run_vertex_stream(
			&model,
			options,
			&vertex,
			Some(GCP_VERTEX_CREDENTIALS_MARKER),
			Some(adc_env(&credentials)),
		)
		.await;

		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let requests = recorded.lock().clone();
		assert_eq!(requests.len(), 2);
		let model_request = &requests[1];
		assert!(model_request["url"]
			.as_str()
			.expect("url")
			.contains("projects/test-project/locations/us-central1"));
		assert_eq!(header_value(model_request, "authorization"), Some("Bearer test-access-token"));
		assert!(header_value(model_request, "x-goog-api-key").is_none());
	}

	#[tokio::test]
	async fn falls_back_to_adc_when_env_api_key_is_a_placeholder_marker() {
		let model = catalog_model("gemini-3-flash-preview");
		let credentials = service_account_file("env-placeholder");
		let mut env = adc_env(&credentials);
		env.insert("GOOGLE_CLOUD_API_KEY".to_string(), "<authenticated>".to_string());
		let options = StreamOptions::default();
		let vertex = vertex_options(Some("test-project"), Some("us-central1"));
		let (result, recorded) = run_vertex_stream(&model, options, &vertex, None, Some(env)).await;

		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let requests = recorded.lock().clone();
		assert_eq!(requests.len(), 2);
		let model_request = &requests[1];
		assert!(model_request["url"]
			.as_str()
			.expect("url")
			.contains("projects/test-project/locations/us-central1"));
		assert!(header_value(model_request, "x-goog-api-key").is_none());
	}

	#[tokio::test]
	async fn still_uses_the_api_key_client_for_real_api_keys() {
		let model = catalog_model("gemini-3-flash-preview");
		let options = StreamOptions::default();
		let vertex = vertex_options(None, None);
		let (result, recorded) = run_vertex_stream(
			&model,
			options,
			&vertex,
			Some("AIzaSyExampleRealisticLookingApiKey123456"),
			None,
		)
		.await;

		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let requests = recorded.lock().clone();
		assert_eq!(requests.len(), 1, "no token exchange for API-key clients");
		let request = &requests[0];
		let url = request["url"].as_str().expect("url");
		assert!(url.starts_with(
			"https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
		));
		assert_eq!(header_value(request, "x-goog-api-key"), Some("AIzaSyExampleRealisticLookingApiKey123456"));
		assert!(header_value(request, "authorization").is_none());
	}

	#[tokio::test]
	async fn does_not_forward_generated_vertex_base_url_placeholders() {
		let model = catalog_model("gemini-3-flash-preview");
		assert_eq!(model.base.base_url, "https://{location}-aiplatform.googleapis.com");
		let credentials = service_account_file("placeholder-base-url");
		let options = StreamOptions::default();
		let vertex = vertex_options(Some("test-project"), Some("us-central1"));
		let (result, recorded) =
			run_vertex_stream(&model, options, &vertex, None, Some(adc_env(&credentials))).await;

		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let requests = recorded.lock().clone();
		let model_request = &requests[1];
		let url = model_request["url"].as_str().expect("url");
		assert!(url.starts_with("https://us-central1-aiplatform.googleapis.com/v1/"));
		assert_eq!(header_value(model_request, "User-Agent"), Some(get_pi_user_agent().as_str()));
	}

	#[tokio::test]
	async fn lets_explicit_headers_override_the_default_user_agent() {
		let model = catalog_model("gemini-3-flash-preview");
		let credentials = service_account_file("ua-override");
		let mut options = StreamOptions::default();
		options.base.headers = Some(
			[("User-Agent".to_string(), Some("custom-agent".to_string()))].into_iter().collect(),
		);
		let vertex = vertex_options(Some("test-project"), Some("us-central1"));
		let (result, recorded) =
			run_vertex_stream(&model, options, &vertex, None, Some(adc_env(&credentials))).await;

		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let requests = recorded.lock().clone();
		let model_request = &requests[1];
		assert_eq!(header_value(model_request, "User-Agent"), Some("custom-agent"));
	}

	#[tokio::test]
	async fn forwards_custom_base_url_to_the_adc_client() {
		let model = with_base_url(&catalog_model("gemini-3-flash-preview"), "https://proxy.example.com");
		let credentials = service_account_file("custom-base-adc");
		let options = StreamOptions::default();
		let vertex = vertex_options(Some("test-project"), Some("us-central1"));
		let (result, recorded) =
			run_vertex_stream(&model, options, &vertex, None, Some(adc_env(&credentials))).await;

		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let requests = recorded.lock().clone();
		assert_eq!(requests.len(), 2);
		let model_request = &requests[1];
		let url = model_request["url"].as_str().expect("url");
		// ResourceScope.COLLECTION: no project/location prefix on custom baseUrls.
		assert_eq!(
			url,
			"https://proxy.example.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
		);
		assert_eq!(header_value(model_request, "authorization"), Some("Bearer test-access-token"));
	}

	#[tokio::test]
	async fn forwards_custom_base_url_to_the_api_key_client() {
		let model = with_base_url(&catalog_model("gemini-3-flash-preview"), "https://proxy.example.com");
		let options = StreamOptions::default();
		let vertex = vertex_options(None, None);
		let (result, recorded) = run_vertex_stream(
			&model,
			options,
			&vertex,
			Some("AIzaSyExampleRealisticLookingApiKey123456"),
			None,
		)
		.await;

		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let requests = recorded.lock().clone();
		assert_eq!(requests.len(), 1);
		let request = &requests[0];
		assert_eq!(
			request["url"].as_str().expect("url"),
			"https://proxy.example.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
		);
		assert_eq!(header_value(request, "x-goog-api-key"), Some("AIzaSyExampleRealisticLookingApiKey123456"));
	}

	#[tokio::test]
	async fn does_not_append_api_version_when_custom_base_url_already_includes_one() {
		let model = with_base_url(
			&catalog_model("gemini-3-flash-preview"),
			"https://proxy.example.com/v1/projects/test-project/locations/global",
		);
		let credentials = service_account_file("base-with-version");
		let options = StreamOptions::default();
		let vertex = vertex_options(Some("test-project"), Some("us-central1"));
		let (result, recorded) =
			run_vertex_stream(&model, options, &vertex, None, Some(adc_env(&credentials))).await;

		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);
		let requests = recorded.lock().clone();
		let model_request = &requests[1];
		assert_eq!(
			model_request["url"].as_str().expect("url"),
			"https://proxy.example.com/v1/projects/test-project/locations/global/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
		);
	}

	// -- google-raw-stop-reason (vertex adapter cases) ------------------------

	#[tokio::test]
	async fn preserves_raw_gemini_finish_reasons_for_vertex_errors() {
		let model = catalog_model("gemini-3-flash-preview");
		let credentials = service_account_file("safety");
		let mut options = StreamOptions::default();
		options.base.env = Some(adc_env(&credentials));
		let vertex = vertex_options(Some("test-project"), Some("us-central1"));
		let recorded = Arc::new(Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(vertex_fetch(
			&[json!({
				"candidates": [{ "finishReason": "SAFETY" }],
				"usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 0, "totalTokenCount": 1 },
			})],
			recorded,
		));

		let message = stream_with_adc(&model, options, &vertex).into_result().await;

		assert_eq!(message.stop_reason, StopReason::Error);
		assert_eq!(message.raw_stop_reason.as_deref(), Some("SAFETY"));
		assert_eq!(message.error_message.as_deref(), Some("Provider stopped with: SAFETY"));
	}

	#[tokio::test]
	async fn preserves_max_tokens_with_a_tool_call_as_length_for_vertex() {
		let model = catalog_model("gemini-3-flash-preview");
		let credentials = service_account_file("max-tokens");
		let mut options = StreamOptions::default();
		options.base.env = Some(adc_env(&credentials));
		let vertex = vertex_options(Some("test-project"), Some("us-central1"));
		let recorded = Arc::new(Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(vertex_fetch(
			&[json!({
				"candidates": [{
					"content": { "parts": [{ "functionCall": { "id": "call-1", "name": "echo", "args": { "value": "truncated" } } }] },
					"finishReason": "MAX_TOKENS",
				}],
			})],
			recorded,
		));

		let message = stream_with_adc(&model, options, &vertex).into_result().await;

		assert_eq!(message.stop_reason, StopReason::Length);
		assert_eq!(message.raw_stop_reason.as_deref(), Some("MAX_TOKENS"));
		assert!(message.content.iter().any(|block| matches!(block, AssistantContent::ToolCall(_))));
	}

	#[tokio::test]
	async fn maps_stop_with_a_tool_call_to_tool_use_for_vertex() {
		let model = catalog_model("gemini-3-flash-preview");
		let credentials = service_account_file("stop-tooluse");
		let mut options = StreamOptions::default();
		options.base.env = Some(adc_env(&credentials));
		let vertex = vertex_options(Some("test-project"), Some("us-central1"));
		let recorded = Arc::new(Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(vertex_fetch(
			&[json!({
				"candidates": [{
					"content": { "parts": [{ "functionCall": { "id": "call-1", "name": "echo", "args": { "value": "truncated" } } }] },
					"finishReason": "STOP",
				}],
			})],
			recorded,
		));

		let message = stream_with_adc(&model, options, &vertex).into_result().await;

		assert_eq!(message.stop_reason, StopReason::ToolUse);
		assert_eq!(message.raw_stop_reason.as_deref(), Some("STOP"));
		assert!(message.content.iter().any(|block| matches!(block, AssistantContent::ToolCall(_))));
	}

	#[tokio::test]
	async fn forwards_each_sdk_chunk_in_order_before_normalizing_it_for_vertex() {
		let model = catalog_model("gemini-3-flash-preview");
		let credentials = service_account_file("forwarding");
		let chunks = vec![
			json!({
				"responseId": "resp_vertex",
				"candidates": [{ "content": { "parts": [{ "text": "hello" }] } }],
			}),
			json!({
				"candidates": [{ "finishReason": "STOP" }],
				"usageMetadata": { "promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3 },
			}),
		];
		let received = Arc::new(Mutex::new(Vec::<Value>::new()));
		let event_models = Arc::new(Mutex::new(Vec::<(String, String)>::new()));

		let mut options = StreamOptions::default();
		options.base.env = Some(adc_env(&credentials));
		let hook_received = received.clone();
		let hook_models = event_models.clone();
		options.on_provider_stream_event = Some(Arc::new(move |chunk: Value, event_model: Model| {
			let hook_received = hook_received.clone();
			let hook_models = hook_models.clone();
			Box::pin(async move {
				hook_received.lock().push(chunk);
				hook_models
					.lock()
					.push((event_model.provider().to_string(), event_model.id().to_string()));
			})
		}));
		let recorded = Arc::new(Mutex::new(Vec::<Value>::new()));
		options.base.fetch = Some(vertex_fetch(&chunks, recorded));
		let vertex = vertex_options(Some("test-project"), Some("us-central1"));

		let result = stream_with_adc(&model, options, &vertex).into_result().await;

		// Drop the token-exchange request; only stream chunks were hooked.
		assert_eq!(*received.lock(), chunks);
		let models = event_models.lock().clone();
		assert!(
			models.iter().all(|(provider, id)| provider == "google-vertex" && id == model.id()),
			"unexpected event models: {models:?}"
		);
		assert_eq!(result.stop_reason, StopReason::Stop);
		assert_eq!(result.response_id.as_deref(), Some("resp_vertex"));
		assert_eq!(
			serde_json::to_value(&result.content).expect("content"),
			json!([{ "type": "text", "text": "hello" }])
		);
	}

	// -- google-thinking-level-map (vertex adapter cases) ---------------------

	async fn capture_vertex_payload(
		model: &Model,
		reasoning: Option<ThinkingLevel>,
		thinking_budgets: Option<ThinkingBudgets>,
	) -> Value {
		let credentials = service_account_file("thinking-map");
		let mut options = SimpleStreamOptions::default();
		options.base.base.api_key = Some("test".to_string());
		options.reasoning = reasoning;
		options.thinking_budgets = thinking_budgets;
		options.base.base.env = Some(adc_env(&credentials));
		let recorded = Arc::new(Mutex::new(Vec::<Value>::new()));
		options.base.base.fetch = Some(vertex_fetch(&[ok_chunk()], recorded.clone()));

		let derived = derive_vertex_options(model, &options).expect("derivation");
		let vertex = GoogleVertexOptions {
			project: Some("test-project".to_string()),
			location: Some("us-central1".to_string()),
			..derived
		};
		let base = crate::api::simple_options::build_base_options(model, &hello_context(), &options, None);
		let result = stream_with_options(model, &hello_context(), &base, &vertex).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Stop, "error: {:?}", result.error_message);

		let requests = recorded.lock().clone();
		let model_request = requests.iter().find(|request| !request["url"].as_str().expect("url").contains("oauth2")).expect("model request");
		model_request.get("body").cloned().expect("payload captured")
	}

	#[tokio::test]
	async fn uses_the_lowest_supported_vertex_level_when_reasoning_is_omitted() {
		let model = with_level_map(
			&catalog_model("gemini-3.8-flash"),
			json!({ "off": null, "minimal": null, "low": "low", "medium": "medium", "high": "high", "xhigh": null, "max": null }),
		);
		let payload = capture_vertex_payload(&model, None, None).await;

		assert_eq!(
			payload.pointer("/config/thinkingConfig").expect("thinkingConfig"),
			&json!({ "thinkingLevel": "LOW" })
		);
	}

	#[tokio::test]
	async fn preserves_native_medium_effort_for_vertex_gemini_3_1_pro() {
		let model = with_level_map(
			&catalog_model("gemini-3.1-pro-preview"),
			json!({ "off": null, "minimal": null, "low": "low", "medium": "medium", "high": "high", "xhigh": null, "max": null }),
		);
		let payload = capture_vertex_payload(&model, Some(ThinkingLevel::Medium), None).await;

		assert_eq!(
			payload.pointer("/config/thinkingConfig").expect("thinkingConfig"),
			&json!({ "includeThoughts": true, "thinkingLevel": "MEDIUM" })
		);
	}

	#[tokio::test]
	async fn disables_vertex_gemini_2_5_thinking_when_reasoning_is_omitted() {
		let model = with_level_map(&catalog_model("gemini-2.5-flash"), json!({}));
		let payload = capture_vertex_payload(&model, None, None).await;

		assert_eq!(
			payload.pointer("/config/thinkingConfig").expect("thinkingConfig"),
			&json!({ "thinkingBudget": 0 })
		);
	}

	#[tokio::test]
	async fn maps_google_vertex_extended_levels() {
		let model = with_level_map(&catalog_model("gemini-3.7-flash"), json!({ "xhigh": "high" }));
		let payload = capture_vertex_payload(&model, Some(ThinkingLevel::Xhigh), None).await;

		assert_eq!(
			payload.pointer("/config/thinkingConfig").expect("thinkingConfig"),
			&json!({ "includeThoughts": true, "thinkingLevel": "HIGH" })
		);
	}

	#[tokio::test]
	async fn uses_mapped_google_vertex_levels_for_token_budgets() {
		let model = with_level_map(&catalog_model("gemini-2.5-flash"), json!({ "max": "high" }));
		let budgets = serde_json::from_value::<ThinkingBudgets>(json!({ "high": 4321 })).expect("budgets");
		let payload = capture_vertex_payload(&model, Some(ThinkingLevel::Max), Some(budgets)).await;

		let thinking_config = payload.pointer("/config/thinkingConfig").expect("thinkingConfig");
		// Upstream `toMatchObject` (subset): `includeThoughts` rides along.
		assert_eq!(thinking_config.get("thinkingBudget"), Some(&json!(4321)));
	}

	// -- setup failures -------------------------------------------------------

	#[tokio::test]
	async fn reports_missing_project_and_location_with_upstream_strings() {
		let model = catalog_model("gemini-3-flash-preview");
		let options = StreamOptions::default();
		let vertex = vertex_options(None, None);
		let result = stream_with_adc(&model, options, &vertex).into_result().await;
		assert_eq!(
			result.error_message.as_deref(),
			Some("Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options.")
		);

		let options = StreamOptions::default();
		let vertex = vertex_options(Some("test-project"), None);
		let result = stream_with_adc(&model, options, &vertex).into_result().await;
		assert_eq!(
			result.error_message.as_deref(),
			Some("Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.")
		);
	}

	#[tokio::test]
	async fn reports_missing_adc_credentials_with_the_google_auth_message() {
		let model = catalog_model("gemini-3-flash-preview");
		let mut options = StreamOptions::default();
		// A path that never exists keeps the test off the ambient gcloud file.
		options.base.env = Some(
			[("GOOGLE_APPLICATION_CREDENTIALS".to_string(), "/nonexistent/pi-ai-vertex-creds.json".to_string())]
				.into_iter()
				.collect(),
		);
		let vertex = vertex_options(Some("test-project"), Some("us-central1"));
		let result = stream_with_adc(&model, options, &vertex).into_result().await;
		assert_eq!(
			result.error_message.as_deref(),
			Some(COULD_NOT_LOAD_DEFAULT_CREDENTIALS),
		);
	}

	#[tokio::test]
	async fn rejects_an_aborted_signal_before_sending() {
		let model = catalog_model("gemini-3-flash-preview");
		let mut options = StreamOptions::default();
		let controller = crate::utils::abort::AbortController::new();
		controller.abort();
		options.base.signal = Some(controller.signal());
		let vertex = vertex_options(Some("test-project"), Some("us-central1"));
		let result = stream_with_adc(&model, options, &vertex).into_result().await;
		assert_eq!(result.stop_reason, StopReason::Aborted);
		assert_eq!(result.error_message.as_deref(), Some("Request aborted"));
	}
}

