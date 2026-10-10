//! Port of `packages/ai/src/api/google-generative-ai.ts` — the
//! `google-generative-ai` API adapter over the Gemini API.
//!
//! Upstream drives the `@google/genai` SDK
//! (`client.models.generateContentStream`); the Rust port hand-rolls the
//! equivalent HTTP request over the [`crate::http::FetchFn`] seam:
//! `POST {baseUrl}/{apiVersion}/models/{model}:streamGenerateContent?alt=sse`
//! (SDK defaults `https://generativelanguage.googleapis.com` + `v1beta`,
//! collapsing to `{model.baseUrl}/models/...` when the model carries a
//! baseUrl, exactly like the SDK's `httpOptions.apiVersion: ""`), with
//! `x-goog-api-key` auth and the merged pi/model/caller headers. SSE payloads
//! are decoded by [`crate::api::google_shared::GoogleSseStream`], mirroring
//! the SDK's `processStreamResponse`.
//!
//! # Deviations (file ledger)
//!
//! - **D15** — per-API options cannot ride `StreamOptions` (types.rs frozen;
//!   deviation follows crate D2 pattern): [`GoogleOptions`] lives here with
//!   upstream-exact fields, `stream`/`stream_simple` build its default and
//!   delegate to [`stream_with_options`]/[`stream_simple_with_options`];
//!   final integration should add the nested `StreamOptions` group (reported
//!   in reexports). The `ProviderStreams` entry uses the group `Default`.
//! - **D16** — upstream rejects custom fetches
//!   (`Custom fetch is not supported by the Google Generative AI adapter`)
//!   because the SDK owns HTTP; over the explicit [`FetchFn`] seam an
//!   injected fetch IS the transport (crate D4), so `options.base.fetch` is
//!   consumed like every other adapter and the guard is unreachable (the
//!   fetch-option port scopes those upstream cases out).
//! - **D17** — the SDK's `x-goog-api-client` library-telemetry header is not
//!   reproduced; `User-Agent` (pi), `Content-Type` and `x-goog-api-key` are.
//! - **D18** — non-2xx responses become
//!   [`crate::http::ProviderHttpError`] with status/headers/body; the error
//!   message is minimal (`status` + raw body) rather than the SDK's
//!   `ApiError` string. In-stream `data:` chunks stay verbatim.
//! - **D19** — per-chunk `String::from_utf8_lossy` where upstream streams
//!   through a `TextDecoder` (SSE payloads are ASCII in practice).
//! - **D20** — `toolCallCounter` is a process-global atomic; ids synthesize
//!   as `{name}_{Date.now()}_{counter}` exactly as upstream.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::api::google_shared::{
    convert_messages, convert_tools, get_disabled_google_thinking_config,
    get_google_generative_ai_budget, google_retry_options, is_thinking_part, map_stop_reason,
    resolve_google_function_calling_mode, resolve_google_thinking_level, retain_thought_signature,
    retry_google_request, supports_google_strict_tool_sampling, to_google_sdk_thinking_level,
    to_google_thinking_level, uses_google_thinking_level, GoogleApiThinkingLevel, GoogleContent,
    GoogleGenerateContentParameters, GoogleSseStream, GoogleToolChoice,
};
use crate::http::{FetchFn, HttpBody, HttpRequest, HttpResponse, ProviderHttpError};
use crate::models::{calculate_cost, ProviderStreams};
use crate::types::{
    AnyModel, AssistantContent, AssistantMessage, AssistantMessageEvent,
    AssistantMessageEventStream, Model, ModelThinkingLevel, SimpleStreamOptions, StopReason,
    StreamDoneReason, StreamErrorReason, StreamOptions, TextContent, ThinkingContent,
    ThinkingLevel, Tool, ToolCall, ToolChoice, TranscriptContext, Usage,
};
use crate::utils::diagnostics::now_ms;
use crate::utils::error_body::{format_provider_error, normalize_provider_error};
use crate::utils::event_stream::{done_event, error_event};
use crate::utils::headers::provider_headers_to_record;
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::text::get_system_message_text;
use crate::utils::transcript::{
    collapse_system_messages, get_current_tools, get_initial_system_message,
};

/// Upstream `GoogleOptions`.
#[derive(Clone, Default)]
pub struct GoogleOptions {
    /// `"auto" | "none" | "any"`.
    pub tool_choice: Option<GoogleToolChoice>,
    /// Thinking request; `-1` budget means dynamic, `0` disables.
    pub thinking: Option<GoogleThinkingOptions>,
}

/// Upstream `GoogleOptions.thinking`.
#[derive(Clone, Default)]
pub struct GoogleThinkingOptions {
    pub enabled: bool,
    pub budget_tokens: Option<i64>,
    pub level: Option<GoogleApiThinkingLevel>,
}

/// SDK default endpoint pieces (`@google/genai` Gemini API defaults).
const DEFAULT_GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com";
const DEFAULT_GEMINI_API_VERSION: &str = "v1beta";

/// Counter for generating unique tool call IDs (upstream module-level
/// `toolCallCounter`).
static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Upstream `stream` export for the google-generative-ai API (options group
/// defaulted; deviation D15).
pub fn stream(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
) -> AssistantMessageEventStream {
    stream_with_options(model, context, options, &GoogleOptions::default())
}

/// [`stream`] with the per-API options group (deviation D15).
pub fn stream_with_options(
    model: &Model,
    context: &TranscriptContext,
    base: &StreamOptions,
    google: &GoogleOptions,
) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    let normalized_context = collapse_system_messages(context);

    let model = model.clone();
    let base = base.clone();
    let google = google.clone();
    let inner_stream = stream.clone();
    tokio::spawn(async move {
        run_stream(model, normalized_context, base, google, inner_stream).await;
    });

    stream
}

#[allow(clippy::too_many_lines)]
async fn run_stream(
    model: Model,
    normalized_context: TranscriptContext,
    options: StreamOptions,
    google: GoogleOptions,
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
        duration_ms: None,
    };

    let result: Result<(), String> = async {
        let api_key = options.base.api_key.clone().filter(|key| !key.is_empty());
        let Some(api_key) = api_key else {
            return Err(format!("No API key for provider: {}", model.provider()));
        };

        let params = build_params(&model, &normalized_context, &options, &google)?;
        let mut params_value = serde_json::to_value(&params).map_err(|error| error.to_string())?;
        if let Some(hook) = &options.base.on_payload {
            if let Some(next) = hook(params_value.clone(), model.clone()).await {
                params_value = next;
            }
        }

        let url = build_url(&model);
        let headers = build_headers(&model, options.base.headers.as_ref());
        let mut request = HttpRequest::new("POST", url);
        request.headers = headers;
        request.body =
            HttpBody::Bytes(serde_json::to_vec(&params_value).map_err(|error| error.to_string())?);
        request.timeout_ms = options.base.timeout_ms;
        request.signal = options.base.signal.clone();
        request = request
            .header("content-type", "application/json")
            .header("x-goog-api-key", api_key);

        let fetch: FetchFn = options.base.fetch.clone().unwrap_or_default();
        let response = retry_google_request(
            move || {
                let request = clone_request(&request);
                let fetch = fetch.clone();
                async move {
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

        stream.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });

        let body = match response.body {
            HttpBody::Stream(body_stream) => body_stream,
            other => Box::pin(futures_util::stream::once(
                async move { other.collect().await },
            )),
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

                // @google/genai documents GenerateContentResponse.responseId as an
                // output-only field used to identify each response. Keep the first
                // non-empty one from the stream.
                if output
                    .response_id
                    .as_deref()
                    .map(str::is_empty)
                    .unwrap_or(true)
                {
                    if let Some(response_id) = chunk.response_id.filter(|id| !id.is_empty()) {
                        output.response_id = Some(response_id);
                    }
                }

                let candidate = chunk
                    .candidates
                    .as_ref()
                    .and_then(|candidates| candidates.first());
                if let Some(parts) = candidate
                    .and_then(|candidate| candidate.content.as_ref())
                    .and_then(|content| content.parts.as_ref())
                {
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
                                        BlockKind::Text => {
                                            stream.push(AssistantMessageEvent::TextEnd {
                                                content_index: index as u32,
                                                content: block_text(&output.content, index),
                                                partial: output.clone(),
                                            })
                                        }
                                        BlockKind::Thinking => {
                                            stream.push(AssistantMessageEvent::ThinkingEnd {
                                                content_index: index as u32,
                                                content: block_thinking(&output.content, index),
                                                partial: output.clone(),
                                            })
                                        }
                                    }
                                }
                                if is_thinking {
                                    output.content.push(AssistantContent::Thinking(
                                        ThinkingContent {
                                            thinking: String::new(),
                                            thinking_signature: None,
                                            redacted: None,
                                        },
                                    ));
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
                                    if let Some(AssistantContent::Thinking(block)) =
                                        output.content.get_mut(index)
                                    {
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
                                    if let Some(AssistantContent::Text(block)) =
                                        output.content.get_mut(index)
                                    {
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
                                    BlockKind::Text => {
                                        stream.push(AssistantMessageEvent::TextEnd {
                                            content_index: index as u32,
                                            content: block_text(&output.content, index),
                                            partial: output.clone(),
                                        })
                                    }
                                    BlockKind::Thinking => {
                                        stream.push(AssistantMessageEvent::ThinkingEnd {
                                            content_index: index as u32,
                                            content: block_thinking(&output.content, index),
                                            partial: output.clone(),
                                        })
                                    }
                                }
                            }

                            // Generate unique ID if not provided or if it's a duplicate
                            let provided_id = function_call.id.clone();
                            let needs_new_id =
                                provided_id.as_deref().map(str::is_empty).unwrap_or(true)
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
                                thought_signature: part
                                    .thought_signature
                                    .clone()
                                    .filter(|sig| !sig.is_empty()),
                                namespace: None,
                            };

                            output
                                .content
                                .push(AssistantContent::ToolCall(tool_call.clone()));
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

                if let Some(finish_reason) =
                    candidate.and_then(|candidate| candidate.finish_reason.as_ref())
                {
                    output.raw_stop_reason = Some(finish_reason.clone());
                    output.stop_reason = match map_stop_reason(
                        &crate::api::google_shared::FinishReason::parse(finish_reason),
                    ) {
                        Ok(reason) => reason,
                        Err(message) => break 'chunks Err(message),
                    };
                    if output
                        .content
                        .iter()
                        .any(|block| matches!(block, AssistantContent::ToolCall(_)))
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
                        output: usage
                            .candidates_token_count
                            .unwrap_or(0)
                            .saturating_add(thoughts),
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

        if options
            .base
            .signal
            .as_ref()
            .is_some_and(|signal| signal.aborted())
        {
            return Err("Request was aborted".to_string());
        }

        if output.stop_reason == StopReason::Pending {
            return Err("Google stream ended without a finish reason".to_string());
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
        let aborted = options
            .base
            .signal
            .as_ref()
            .is_some_and(|signal| signal.aborted());
        output.stop_reason = if aborted {
            StopReason::Aborted
        } else {
            StopReason::Error
        };
        // formatProviderError(normalizeProviderError(error)): plain error
        // messages pass through unchanged.
        output.error_message = Some(format_provider_error(
            &normalize_provider_error(&ProviderHttpError::new(message)),
            None,
        ));
        let reason = if aborted {
            StreamErrorReason::Aborted
        } else {
            StreamErrorReason::Error
        };
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

fn clone_request(request: &HttpRequest) -> HttpRequest {
    let mut cloned = HttpRequest::new(request.method.clone(), request.url.clone());
    cloned.headers = request.headers.clone();
    cloned.body = match &request.body {
        HttpBody::Bytes(bytes) => HttpBody::Bytes(bytes.clone()),
        HttpBody::Empty => HttpBody::Empty,
        HttpBody::Stream(_) => HttpBody::Empty,
    };
    cloned.timeout_ms = request.timeout_ms;
    cloned.signal = request.signal.clone();
    cloned
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
    ProviderHttpError {
        message,
        status: Some(status),
        headers,
        body: Some(body_text),
        source: None,
    }
}

/// Upstream `streamSimple` export.
pub fn stream_simple(
    model: &Model,
    context: &TranscriptContext,
    options: &SimpleStreamOptions,
) -> AssistantMessageEventStream {
    let api_key = options.base.base.api_key.clone();
    if api_key.as_deref().unwrap_or("").is_empty() {
        return error_stream(
            model,
            &format!("No API key for provider: {}", model.provider()),
        );
    }

    match derive_google_options(model, options) {
        Ok(google) => {
            let base = crate::api::simple_options::build_base_options(
                model,
                context,
                options,
                api_key.as_deref(),
            );
            stream_with_options(model, context, &base, &google)
        }
        Err(message) => error_stream(model, &message),
    }
}

/// [`stream_simple`] with the per-API options group prebuilt (deviation D15).
pub fn stream_simple_with_options(
    model: &Model,
    context: &TranscriptContext,
    base: &SimpleStreamOptions,
    google: &GoogleOptions,
) -> AssistantMessageEventStream {
    let api_key = base.base.base.api_key.clone();
    if api_key.as_deref().unwrap_or("").is_empty() {
        return error_stream(
            model,
            &format!("No API key for provider: {}", model.provider()),
        );
    }
    let base_options =
        crate::api::simple_options::build_base_options(model, context, base, api_key.as_deref());
    stream_with_options(model, context, &base_options, google)
}

/// The `{ ...base, toolChoice, thinking }` derivation of upstream
/// `streamSimple`.
fn derive_google_options(
    model: &Model,
    options: &SimpleStreamOptions,
) -> Result<GoogleOptions, String> {
    let tool_choice = options.tool_choice.map(|tool_choice| match tool_choice {
        ToolChoice::Auto => GoogleToolChoice::Auto,
        ToolChoice::None => GoogleToolChoice::None,
    });

    let disabled = || {
        Ok(GoogleOptions {
            tool_choice,
            thinking: Some(GoogleThinkingOptions {
                enabled: false,
                ..Default::default()
            }),
        })
    };

    let Some(reasoning) = options.reasoning else {
        return disabled();
    };

    let clamped_reasoning =
        crate::models::clamp_thinking_level(model, ModelThinkingLevel::from(reasoning));
    if clamped_reasoning == ModelThinkingLevel::Off {
        return disabled();
    }
    let resolved_level =
        resolve_google_thinking_level(model, thinking_level_of(clamped_reasoning))?;

    if uses_google_thinking_level(model) {
        return Ok(GoogleOptions {
            tool_choice,
            thinking: Some(GoogleThinkingOptions {
                enabled: true,
                level: Some(to_google_thinking_level(resolved_level)),
                budget_tokens: None,
            }),
        });
    }

    Ok(GoogleOptions {
        tool_choice,
        thinking: Some(GoogleThinkingOptions {
            enabled: true,
            budget_tokens: Some(get_google_generative_ai_budget(
                model,
                resolved_level,
                options.thinking_budgets.as_ref(),
            )),
            level: None,
        }),
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
        duration_ms: None,
    };
    stream.push(error_event(output, StreamErrorReason::Error));
    stream.end(None);
    stream
}

/// Upstream `createClient` + request assembly: URL and merged headers.
fn build_url(model: &Model) -> String {
    let (base, api_version) = if model.base.base_url.trim().is_empty() {
        (
            DEFAULT_GEMINI_BASE_URL.to_string(),
            DEFAULT_GEMINI_API_VERSION,
        )
    } else {
        // baseUrl already includes version path, don't append.
        (model.base.base_url.trim_end_matches('/').to_string(), "")
    };
    let id = model.id();
    if api_version.is_empty() {
        format!("{base}/models/{id}:streamGenerateContent?alt=sse")
    } else {
        format!("{base}/{api_version}/models/{id}:streamGenerateContent?alt=sse")
    }
}

fn build_headers(
    model: &Model,
    options_headers: Option<&crate::types::ProviderHeaders>,
) -> Vec<(String, String)> {
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

/// Upstream `buildParams`.
fn build_params(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    google: &GoogleOptions,
) -> Result<GoogleGenerateContentParameters, String> {
    let contents: Vec<GoogleContent> = convert_messages(model, context);
    let initial_system_message = get_initial_system_message(context.messages());
    let current_tools: Vec<Tool> = get_current_tools(context.messages());

    let supports_strict_mode = supports_google_strict_tool_sampling(model.id());
    let function_calling_mode = if !current_tools.is_empty() {
        resolve_google_function_calling_mode(
            &current_tools,
            google.tool_choice,
            supports_strict_mode,
        )?
    } else {
        None
    };
    let system_instruction = initial_system_message
        .map(get_system_message_text)
        .unwrap_or_default();

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
            function_calling_config: crate::api::google_shared::GoogleFunctionCallingConfig {
                mode,
            },
        });
    }

    if let Some(thinking) = &google.thinking {
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

    Ok(GoogleGenerateContentParameters {
        model: model.id().to_string(),
        contents,
        config,
    })
}

// ---------------------------------------------------------------------------
// ProviderStreams export
// ---------------------------------------------------------------------------

/// The google-generative-ai API implementation (`googleGenerativeAIApi`).
/// No deferred methods are exported upstream, so both presence probes stay
/// `false`.
pub struct GoogleGenerativeAiApi;

impl ProviderStreams for GoogleGenerativeAiApi {
    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream {
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

/// Upstream `googleGenerativeAIApi` factory.
pub fn google_generative_ai_api() -> Arc<dyn ProviderStreams> {
    Arc::new(GoogleGenerativeAiApi)
}

// ---------------------------------------------------------------------------
// Tests (ports of test/google-thinking-level-map.test.ts google cases,
// test/google-raw-stop-reason.test.ts google cases, and the google cases of
// test/google-thinking-disable.test.ts — the latter are live E2E and gated
// upstream on GEMINI_API_KEY, so they are #[ignore] here)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Context, ThinkingBudgets, UserContentField, UserMessage};
    use parking_lot::Mutex;
    use serde_json::{json, Value};
    use std::sync::Arc;

    fn catalog_model(provider: &str, id: &str) -> Model {
        crate::providers::google_models::google_models()
            .into_iter()
            .find(|model| model.provider() == provider && model.id() == id)
            .unwrap_or_else(|| panic!("catalog model {provider}/{id}"))
    }

    fn test_model(id: &str) -> Model {
        let mut model = catalog_model("google", "gemini-2.5-flash");
        model.base.id = id.to_string();
        model.base.provider = "test-google".to_string();
        model.base.base_url = "https://example.invalid/v1beta".to_string();
        model
    }

    fn with_level_map(model: &Model, thinking_level_map: Value) -> Model {
        let mut model = model.clone();
        model.thinking_level_map =
            Some(serde_json::from_value(thinking_level_map).expect("thinkingLevelMap parses"));
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

    /// Recording FetchFn fake returning a `data:`-line SSE body.
    fn sse_fetch(body: String, recorded: Arc<Mutex<Vec<Value>>>) -> FetchFn {
        let body = Arc::new(body.into_bytes());
        FetchFn::new(move |request: HttpRequest| {
            let request_body = match &request.body {
                HttpBody::Bytes(bytes) => String::from_utf8_lossy(bytes).to_string(),
                _ => String::new(),
            };
            if let Ok(parsed) = serde_json::from_str::<Value>(&request_body) {
                recorded.lock().push(parsed);
            }
            let response = HttpResponse {
                status: 200,
                headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
                body: HttpBody::Bytes((*body).clone()),
            };
            Box::pin(async move { Ok(response) })
        })
    }

    fn sse_body(chunks: &[Value]) -> String {
        chunks
            .iter()
            .map(|chunk| format!("data: {chunk}\n\n"))
            .collect()
    }

    fn stop_chunk() -> Value {
        json!({
            "candidates": [{ "finishReason": "STOP" }],
            "usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2 },
        })
    }

    async fn capture_google_payload(
        model: &Model,
        reasoning: Option<ThinkingLevel>,
        thinking_budgets: Option<ThinkingBudgets>,
    ) -> Value {
        let recorded = Arc::new(Mutex::new(Vec::<Value>::new()));
        let mut options = SimpleStreamOptions::default();
        options.base.base.api_key = Some("test".to_string());
        options.reasoning = reasoning;
        options.thinking_budgets = thinking_budgets;
        options.base.base.fetch = Some(sse_fetch(sse_body(&[stop_chunk()]), recorded.clone()));

        let result = stream_simple(model, &hello_context(), &options)
            .into_result()
            .await;
        assert_eq!(
            result.stop_reason,
            StopReason::Stop,
            "error: {:?}",
            result.error_message
        );

        let recorded = recorded.lock();
        let payload = recorded.first().expect("payload captured");
        payload.clone()
    }

    // -- google-thinking-level-map (google adapter cases) ---------------------

    #[tokio::test]
    async fn uses_the_lowest_supported_google_level_when_reasoning_is_omitted() {
        let model = with_level_map(
            &test_model("gemini-3.8-flash"),
            json!({ "off": null, "minimal": null, "low": "low", "medium": "medium", "high": "high", "xhigh": null, "max": null }),
        );
        let payload = capture_google_payload(&model, None, None).await;

        assert_eq!(
            payload
                .pointer("/config/thinkingConfig")
                .expect("thinkingConfig"),
            &json!({ "thinkingLevel": "LOW" })
        );
    }

    #[tokio::test]
    async fn preserves_native_medium_effort_for_gemini_3_1_pro() {
        let model = with_level_map(
            &test_model("gemini-3.1-pro-preview"),
            json!({ "off": null, "minimal": null, "low": "low", "medium": "medium", "high": "high", "xhigh": null, "max": null }),
        );
        let payload = capture_google_payload(&model, Some(ThinkingLevel::Medium), None).await;

        assert_eq!(
            payload
                .pointer("/config/thinkingConfig")
                .expect("thinkingConfig"),
            &json!({ "includeThoughts": true, "thinkingLevel": "MEDIUM" })
        );
    }

    #[tokio::test]
    async fn disables_gemini_2_5_thinking_when_reasoning_is_omitted() {
        let model = with_level_map(&test_model("gemini-2.5-flash"), json!({}));
        let payload = capture_google_payload(&model, None, None).await;

        assert_eq!(
            payload
                .pointer("/config/thinkingConfig")
                .expect("thinkingConfig"),
            &json!({ "thinkingBudget": 0 })
        );
    }

    #[tokio::test]
    async fn maps_google_generative_ai_xhigh_to_a_supported_level() {
        for reasoning in [ThinkingLevel::Xhigh, ThinkingLevel::Max] {
            let model = with_level_map(
                &test_model("gemini-3.7-flash"),
                json!({ "xhigh": "high", "max": "high" }),
            );
            let payload = capture_google_payload(&model, Some(reasoning), None).await;

            assert_eq!(
                payload
                    .pointer("/config/thinkingConfig")
                    .expect("thinkingConfig"),
                &json!({ "includeThoughts": true, "thinkingLevel": "HIGH" })
            );
        }
    }

    #[tokio::test]
    async fn honors_uppercase_provider_values_for_standard_levels() {
        let model = with_level_map(&test_model("gemini-3.7-flash"), json!({ "high": "LOW" }));
        let payload = capture_google_payload(&model, Some(ThinkingLevel::High), None).await;

        let thinking_config = payload
            .pointer("/config/thinkingConfig")
            .expect("thinkingConfig");
        // Upstream `toMatchObject` (subset): `includeThoughts` rides along.
        assert_eq!(thinking_config.get("thinkingLevel"), Some(&json!("LOW")));
    }

    #[tokio::test]
    async fn uses_mapped_levels_for_token_budgets() {
        let model = with_level_map(&test_model("gemini-2.5-flash"), json!({ "xhigh": "high" }));
        let budgets = serde_json::from_value(json!({ "high": 1234 })).expect("budgets");
        let payload =
            capture_google_payload(&model, Some(ThinkingLevel::Xhigh), Some(budgets)).await;

        let thinking_config = payload
            .pointer("/config/thinkingConfig")
            .expect("thinkingConfig");
        // Upstream `toMatchObject` (subset): `includeThoughts` rides along.
        assert_eq!(thinking_config.get("thinkingBudget"), Some(&json!(1234)));
    }

    // -- google-raw-stop-reason (google adapter cases) ------------------------

    #[tokio::test]
    async fn preserves_raw_gemini_finish_reasons_for_google_generative_ai_errors() {
        let model = catalog_model("google", "gemini-2.5-flash");
        let recorded = Arc::new(Mutex::new(Vec::<Value>::new()));
        let mut options = StreamOptions::default();
        options.base.api_key = Some("test-api-key".to_string());
        options.base.fetch = Some(sse_fetch(
            sse_body(&[json!({
                "responseId": "google-response-id",
                "candidates": [{ "finishReason": "MALFORMED_FUNCTION_CALL" }],
                "usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 0, "totalTokenCount": 1 },
            })]),
            recorded,
        ));

        let message = stream(&model, &hello_context(), &options)
            .into_result()
            .await;

        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(
            message.raw_stop_reason.as_deref(),
            Some("MALFORMED_FUNCTION_CALL")
        );
        assert_eq!(
            message.error_message.as_deref(),
            Some("Provider stopped with: MALFORMED_FUNCTION_CALL")
        );
    }

    #[tokio::test]
    async fn preserves_max_tokens_with_a_tool_call_as_length() {
        let model = catalog_model("google", "gemini-2.5-flash");
        let recorded = Arc::new(Mutex::new(Vec::<Value>::new()));
        let mut options = StreamOptions::default();
        options.base.api_key = Some("test-api-key".to_string());
        options.base.fetch = Some(sse_fetch(
            sse_body(&[json!({
                "candidates": [{
                    "content": { "parts": [{ "functionCall": { "id": "call-1", "name": "echo", "args": { "value": "truncated" } } }] },
                    "finishReason": "MAX_TOKENS",
                }],
            })]),
            recorded,
        ));

        let message = stream(&model, &hello_context(), &options)
            .into_result()
            .await;

        assert_eq!(message.stop_reason, StopReason::Length);
        assert_eq!(message.raw_stop_reason.as_deref(), Some("MAX_TOKENS"));
        assert!(message
            .content
            .iter()
            .any(|block| matches!(block, AssistantContent::ToolCall(_))));
    }

    #[tokio::test]
    async fn maps_stop_with_a_tool_call_to_tool_use() {
        let model = catalog_model("google", "gemini-2.5-flash");
        let recorded = Arc::new(Mutex::new(Vec::<Value>::new()));
        let mut options = StreamOptions::default();
        options.base.api_key = Some("test-api-key".to_string());
        options.base.fetch = Some(sse_fetch(
            sse_body(&[json!({
                "candidates": [{
                    "content": { "parts": [{ "functionCall": { "id": "call-1", "name": "echo", "args": { "value": "truncated" } } }] },
                    "finishReason": "STOP",
                }],
            })]),
            recorded,
        ));

        let message = stream(&model, &hello_context(), &options)
            .into_result()
            .await;

        assert_eq!(message.stop_reason, StopReason::ToolUse);
        assert_eq!(message.raw_stop_reason.as_deref(), Some("STOP"));
        assert!(message
            .content
            .iter()
            .any(|block| matches!(block, AssistantContent::ToolCall(_))));
    }

    #[tokio::test]
    async fn forwards_each_sdk_chunk_in_order_before_normalizing_it() {
        let model = catalog_model("google", "gemini-2.5-flash");
        let chunks = vec![
            json!({
                "responseId": "resp_google",
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
        options.base.api_key = Some("test-api-key".to_string());
        let hook_received = received.clone();
        let hook_models = event_models.clone();
        options.on_provider_stream_event =
            Some(Arc::new(move |chunk: Value, event_model: Model| {
                let hook_received = hook_received.clone();
                let hook_models = hook_models.clone();
                Box::pin(async move {
                    hook_received.lock().push(chunk);
                    hook_models.lock().push((
                        event_model.provider().to_string(),
                        event_model.id().to_string(),
                    ));
                })
            }));
        let recorded = Arc::new(Mutex::new(Vec::<Value>::new()));
        options.base.fetch = Some(sse_fetch(sse_body(&chunks), recorded));

        let result = stream(&model, &hello_context(), &options)
            .into_result()
            .await;

        // Upstream asserts chunk identity (`toBe`); Rust adapters re-serialize
        // the parsed chunks, so structural equality is the observable.
        assert_eq!(*received.lock(), chunks);
        let models = event_models.lock().clone();
        assert!(
            models
                .iter()
                .all(|(provider, id)| provider == "google" && id == model.id()),
            "unexpected event models: {models:?}"
        );
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.response_id.as_deref(), Some("resp_google"));
        assert_eq!(
            serde_json::to_value(&result.content).expect("content"),
            json!([{ "type": "text", "text": "hello" }])
        );
    }

    // -- Google Generative AI user agent -------------------------------------

    /// Captures the outgoing HTTP headers (upstream inspects the SDK client's
    /// `httpOptions.headers`; the Rust seam observes the actual request).
    async fn capture_request_headers(
        headers: Option<crate::types::ProviderHeaders>,
    ) -> Vec<(String, String)> {
        let model = catalog_model("google", "gemini-2.5-flash");
        let captured = Arc::new(Mutex::new(Vec::<Vec<(String, String)>>::new()));
        let sink = Arc::new(Mutex::new(Vec::<Value>::new()));
        let recording = sse_fetch(sse_body(&[stop_chunk()]), sink);
        let wrapped = {
            let captured = captured.clone();
            FetchFn::new(move |request: HttpRequest| {
                captured.lock().push(request.headers.clone());
                let recording = recording.clone();
                Box::pin(async move { recording.send(request).await })
            })
        };
        let mut options = StreamOptions::default();
        options.base.api_key = Some("test-api-key".to_string());
        options.base.headers = headers;
        options.base.fetch = Some(wrapped);

        let result = stream(&model, &hello_context(), &options)
            .into_result()
            .await;
        assert_eq!(
            result.stop_reason,
            StopReason::Stop,
            "error: {:?}",
            result.error_message
        );
        let headers = captured.lock().first().cloned().expect("request captured");
        headers
    }

    #[tokio::test]
    async fn uses_pis_user_agent_by_default() {
        let headers = capture_request_headers(None).await;
        let user_agent = headers
            .iter()
            .find(|(name, _)| name == "User-Agent")
            .map(|(_, value)| value.clone())
            .expect("User-Agent header");
        assert_eq!(user_agent, get_pi_user_agent());
    }

    #[tokio::test]
    async fn lets_explicit_headers_override_the_default_user_agent() {
        let headers = capture_request_headers(Some(
            [("User-Agent".to_string(), Some("custom-agent".to_string()))]
                .into_iter()
                .collect(),
        ))
        .await;
        let user_agent = headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("User-Agent"))
            .map(|(_, value)| value.clone())
            .expect("User-Agent header");
        assert_eq!(user_agent, "custom-agent");
    }

    // -- google-thinking-disable (live E2E; upstream gates on GEMINI_API_KEY) --

    async fn expect_thinking_disabled_e2e(provider: &str, id: &str) {
        let model = catalog_model(provider, id);
        let mut options = SimpleStreamOptions::default();
        options.base.base.api_key =
            Some(std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY set for live E2E"));
        options.base.max_tokens = Some(160);
        options.base.temperature = Some(0.0);

        let stream = stream_simple(&model, &hello_context(), &options);
        let mut events = stream.clone();
        let mut thinking_event_count = 0usize;
        while let Some(event) = futures_util::StreamExt::next(&mut events).await {
            match event {
                AssistantMessageEvent::ThinkingStart { .. }
                | AssistantMessageEvent::ThinkingEnd { .. } => {
                    thinking_event_count += 1;
                }
                AssistantMessageEvent::ThinkingDelta { .. } => thinking_event_count += 1,
                AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => break,
                _ => {}
            }
        }

        let response = stream.into_result().await;
        assert_eq!(
            response.stop_reason,
            StopReason::Stop,
            "{}",
            response.error_message.unwrap_or_default()
        );
        assert_eq!(thinking_event_count, 0);
        assert!(!response
            .content
            .iter()
            .any(|block| matches!(block, AssistantContent::Thinking(_))));
        let text: String = response
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        let pongs = text.to_lowercase().matches("pong").count();
        assert!(pongs >= 35, "expected at least 35 pongs, got {pongs}");
    }

    #[tokio::test]
    #[ignore = "live E2E — upstream gates on GEMINI_API_KEY"]
    async fn disables_thinking_for_gemini_2_5_live() {
        expect_thinking_disabled_e2e("google", "gemini-2.5-flash").await;
    }

    #[tokio::test]
    #[ignore = "live E2E — upstream gates on GEMINI_API_KEY"]
    async fn disables_thinking_for_gemini_3_x_live() {
        expect_thinking_disabled_e2e("google", "gemini-3-flash-preview").await;
    }

    #[tokio::test]
    #[ignore = "live E2E — upstream gates on GEMINI_API_KEY"]
    async fn does_not_error_when_thinking_is_off_for_gemini_3_1_pro_live() {
        expect_thinking_disabled_e2e("google", "gemini-3.1-pro-preview").await;
    }
}
