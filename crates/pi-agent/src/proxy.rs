//! Port of `packages/agent/src/proxy.ts` — proxy stream function for apps
//! that route LLM calls through a server. The server manages auth and
//! proxies requests to LLM providers; delta events arrive with the `partial`
//! field stripped and the client reconstructs the partial message.

use std::collections::HashMap;

use pi_ai::http::{FetchFn, HttpBody, HttpRequest, ProviderHttpError};
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, Model, StopReason, StreamDoneReason,
    StreamErrorReason, ToolCall, TranscriptContext, Usage,
};
use pi_ai::utils::abort::AbortSignal;
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_ai::utils::json_parse::parse_streaming_json;
use serde::Deserialize;
use serde_json::Value;

/// Proxy event types — server sends these with the `partial` field stripped
/// to reduce bandwidth (`ProxyAssistantMessageEvent`).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ProxyAssistantMessageEvent {
    #[serde(rename = "start")]
    Start,
    #[serde(rename = "text_start")]
    TextStart {
        #[serde(rename = "contentIndex")]
        content_index: u32,
    },
    #[serde(rename = "text_delta")]
    TextDelta {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "text_end")]
    TextEnd {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        #[serde(rename = "contentSignature")]
        content_signature: Option<String>,
    },
    #[serde(rename = "thinking_start")]
    ThinkingStart {
        #[serde(rename = "contentIndex")]
        content_index: u32,
    },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "thinking_end")]
    ThinkingEnd {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        #[serde(rename = "thinkingSignature")]
        thinking_signature: Option<String>,
    },
    #[serde(rename = "toolcall_start")]
    ToolcallStart {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
    },
    #[serde(rename = "toolcall_delta")]
    ToolcallDelta {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "toolcall_end")]
    ToolcallEnd {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        #[serde(rename = "toolCall")]
        tool_call: ToolCall,
    },
    #[serde(rename = "done")]
    Done {
        reason: StreamDoneReason,
        usage: Usage,
        #[serde(rename = "providerThinkingLevel")]
        provider_thinking_level: Option<String>,
    },
    #[serde(rename = "error")]
    Error {
        reason: StreamErrorReason,
        #[serde(rename = "errorMessage")]
        error_message: Option<String>,
        usage: Usage,
        #[serde(rename = "providerThinkingLevel")]
        provider_thinking_level: Option<String>,
    },
}

/// Stream function that proxies through a server instead of calling LLM
/// providers directly (`ProxyStreamOptions`). The server strips the
/// `partial` field from delta events to reduce bandwidth; we reconstruct
/// the partial message client-side.
///
/// Use this as the `streamFn` option when creating an [`crate::agent::Agent`]
/// that needs to go through a proxy.
#[derive(Clone)]
pub struct ProxyStreamOptions {
    /// Auth token for the proxy server.
    pub auth_token: String,
    /// Proxy server URL (e.g. "https://genai.example.com").
    pub proxy_url: String,
    /// Local abort signal for the proxy request.
    pub signal: Option<AbortSignal>,
    // ==== ProxySerializableStreamOptions subset ====
    pub temperature: Option<f64>,
    pub sampling_params: Option<serde_json::Map<String, Value>>,
    pub max_tokens: Option<u64>,
    pub reasoning: Option<pi_ai::types::ThinkingLevel>,
    pub cache_retention: Option<pi_ai::types::CacheRetention>,
    pub session_id: Option<String>,
    pub headers: Option<pi_ai::types::ProviderHeaders>,
    pub metadata: Option<Value>,
    pub transport: Option<pi_ai::types::Transport>,
    pub thinking_budgets: Option<pi_ai::types::ThinkingBudgets>,
    pub max_retry_delay_ms: Option<u64>,
    /// HTTP seam (crate docs D5); `None` uses the crate default adapter.
    pub fetch: Option<FetchFn>,
}

impl Default for ProxyStreamOptions {
    fn default() -> ProxyStreamOptions {
        ProxyStreamOptions {
            auth_token: String::new(),
            proxy_url: String::new(),
            signal: None,
            temperature: None,
            sampling_params: None,
            max_tokens: None,
            reasoning: None,
            cache_retention: None,
            session_id: None,
            headers: None,
            metadata: None,
            transport: None,
            thinking_budgets: None,
            max_retry_delay_ms: None,
            fetch: None,
        }
    }
}

/// `buildProxyRequestOptions`: only defined values are serialized (upstream
/// object spread drops `undefined` keys; crate docs D5).
fn build_proxy_request_options(options: &ProxyStreamOptions) -> Value {
    fn to_json<T: serde::Serialize>(value: &Option<T>) -> Option<Value> {
        match value {
            Some(value) => serde_json::to_value(value).ok(),
            None => None,
        }
    }
    let mut object = serde_json::Map::new();
    let mut insert = |key: &str, value: Option<Value>| {
        if let Some(value) = value {
            object.insert(key.to_string(), value);
        }
    };
    insert("temperature", options.temperature.map(|value| value.into()));
    insert(
        "samplingParams",
        options.sampling_params.clone().map(Value::Object),
    );
    insert("maxTokens", options.max_tokens.map(|value| value.into()));
    insert("reasoning", to_json(&options.reasoning));
    insert("cacheRetention", to_json(&options.cache_retention));
    insert("sessionId", options.session_id.clone().map(Value::String));
    insert("headers", to_json(&options.headers));
    insert("metadata", options.metadata.clone());
    insert("transport", to_json(&options.transport));
    insert("thinkingBudgets", to_json(&options.thinking_budgets));
    insert(
        "maxRetryDelayMs",
        options.max_retry_delay_ms.map(|value| value.into()),
    );
    Value::Object(object)
}

/// Upstream `streamProxy`. Returns the same concrete event stream the
/// provider surface uses; completes on `done`/`error`.
pub fn stream_proxy(
    model: &Model,
    context: &TranscriptContext,
    options: ProxyStreamOptions,
) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    let producer = stream.clone();
    let model = model.clone();
    let messages: Vec<pi_ai::types::Message> = context.messages().to_vec();
    let options = options.clone();
    tokio::spawn(async move {
        run_proxy_stream(producer, model, messages, options).await;
    });
    stream
}

async fn run_proxy_stream(
    stream: AssistantMessageEventStream,
    model: Model,
    messages: Vec<pi_ai::types::Message>,
    options: ProxyStreamOptions,
) {
    // Initialize the partial message that we'll build up from events
    let mut partial = AssistantMessage {
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
        timestamp: crate::types::now_ms(),
        duration_ms: None,
    };
    // Side map replacing upstream's `partialJson` field on the tool call
    // (crate docs D4).
    let mut partial_json: HashMap<u32, String> = HashMap::new();

    let fetch = options.fetch.clone().unwrap_or_default();

    let outcome = async {
        let mut request = HttpRequest::new("POST", format!("{}/api/stream", options.proxy_url))
            .header("Authorization", format!("Bearer {}", options.auth_token))
            .header("Content-Type", "application/json");
        request.body = HttpBody::bytes(
			serde_json::to_vec(&serde_json::json!({
				"model": serde_json::to_value(&model).expect("model serializes"),
				"context": { "messages": serde_json::to_value(&messages).expect("context serializes") },
				"options": build_proxy_request_options(&options),
			}))
			.expect("proxy request body serializes"),
		);
        request.signal = options.signal.clone();

        let response = fetch
            .send(request)
            .await
            .map_err(|error: ProviderHttpError| error.message)?;

        if !response.ok() {
            let mut error_message = format!("Proxy error: {} {}", response.status, "");
            if let Ok(body) = response.body.collect().await {
                if let Ok(value) = serde_json::from_slice::<Value>(&body) {
                    if let Some(error) = value.get("error").and_then(|error| error.as_str()) {
                        error_message = format!("Proxy error: {error}");
                    }
                }
            }
            return Err(error_message);
        }

        let mut saw_terminal_event = false;
        let mut buffer = String::new();
        let process_line = |line: &str,
                            partial: &mut AssistantMessage,
                            partial_json: &mut HashMap<u32, String>,
                            saw_terminal_event: &mut bool|
         -> Result<Option<AssistantMessageEvent>, String> {
            let Some(data) = line.strip_prefix("data: ") else {
                return Ok(None);
            };
            let data = data.trim();
            if data.is_empty() {
                return Ok(None);
            }
            let proxy_event: ProxyAssistantMessageEvent = serde_json::from_str(data)
                .map_err(|error| format!("Failed to parse proxy event: {error}"))?;
            let event = process_proxy_event(proxy_event, partial, partial_json)?;
            Ok(event.inspect(|event| {
                if matches!(
                    event,
                    AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
                ) {
                    *saw_terminal_event = true;
                }
            }))
        };

        let mut body = response.body;
        while let Some(chunk) = read_chunk(&mut body).await? {
            if options
                .signal
                .as_ref()
                .is_some_and(|signal| signal.aborted())
            {
                return Err("Request aborted by user".into());
            }
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            let mut lines = buffer.split('\n').map(str::to_string).collect::<Vec<_>>();
            buffer = lines.pop().unwrap_or_default();
            for line in &lines {
                if let Some(event) = process_line(
                    line,
                    &mut partial,
                    &mut partial_json,
                    &mut saw_terminal_event,
                )? {
                    stream.push(event);
                }
            }
        }

        if options
            .signal
            .as_ref()
            .is_some_and(|signal| signal.aborted())
        {
            return Err("Request aborted by user".into());
        }

        // The final event may not be newline-terminated; flush and process
        // whatever is left in the buffer.
        if !buffer.is_empty() {
            if let Some(event) = process_line(
                &buffer,
                &mut partial,
                &mut partial_json,
                &mut saw_terminal_event,
            )? {
                stream.push(event);
            }
        }

        if !saw_terminal_event {
            // A clean EOF without a done/error event means the server dropped
            // the response mid-stream. Surface it as an error instead of
            // leaving consumers waiting on a result that never arrives.
            partial.stop_reason = StopReason::Error;
            partial.error_message =
                Some("Connection closed by proxy server before the response completed".into());
            return Err("__eof_error__".into());
        }

        stream.end(None);
        Ok(())
    }
    .await;

    let error_message = match outcome {
        Ok(()) => return,
        Err(message) => message,
    };
    let reason = if options
        .signal
        .as_ref()
        .is_some_and(|signal| signal.aborted())
    {
        StreamErrorReason::Aborted
    } else {
        StreamErrorReason::Error
    };
    if error_message == "__eof_error__" {
        partial.stop_reason = StopReason::Error;
        stream.push(AssistantMessageEvent::Error {
            reason: StreamErrorReason::Error,
            error: partial,
        });
        stream.end(None);
        return;
    }
    partial.stop_reason = match reason {
        StreamErrorReason::Aborted => StopReason::Aborted,
        StreamErrorReason::Error => StopReason::Error,
    };
    partial.error_message = Some(error_message.clone());
    stream.push(AssistantMessageEvent::Error {
        reason,
        error: partial,
    });
    stream.end(None);
}

async fn read_chunk(body: &mut HttpBody) -> Result<Option<Vec<u8>>, String> {
    match body {
        HttpBody::Empty => Ok(None),
        HttpBody::Bytes(bytes) => {
            let bytes = std::mem::take(bytes);
            *body = HttpBody::Empty;
            Ok(Some(bytes))
        }
        HttpBody::Stream(inner) => {
            use futures_util::StreamExt;
            match StreamExt::next(inner).await {
                Some(Ok(chunk)) => Ok(Some(chunk)),
                Some(Err(error)) => Err(error.message),
                None => Ok(None),
            }
        }
    }
}

/// Process a proxy event and update the partial message.
///
/// The `__eof_error__` sentinel from the EOF path (above) reuses the catch
/// branch, matching upstream's throw-inside-async-IIFE control flow.
fn process_proxy_event(
    proxy_event: ProxyAssistantMessageEvent,
    partial: &mut AssistantMessage,
    partial_json: &mut HashMap<u32, String>,
) -> Result<Option<AssistantMessageEvent>, String> {
    match proxy_event {
        ProxyAssistantMessageEvent::Start => Ok(Some(AssistantMessageEvent::Start {
            partial: partial.clone(),
        })),

        ProxyAssistantMessageEvent::TextStart { content_index } => {
            set_content(
                partial,
                content_index,
                pi_ai::types::AssistantContent::Text(pi_ai::types::TextContent::new("")),
            );
            Ok(Some(AssistantMessageEvent::TextStart {
                content_index,
                partial: partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::TextDelta {
            content_index,
            delta,
        } => match content_at(partial, content_index) {
            Some(pi_ai::types::AssistantContent::Text(text)) => {
                text.text += &delta;
                Ok(Some(AssistantMessageEvent::TextDelta {
                    content_index,
                    delta,
                    partial: partial.clone(),
                }))
            }
            _ => Err("Received text_delta for non-text content".into()),
        },

        ProxyAssistantMessageEvent::TextEnd {
            content_index,
            content_signature,
        } => match content_at(partial, content_index) {
            Some(pi_ai::types::AssistantContent::Text(text)) => {
                text.text_signature = content_signature;
                Ok(Some(AssistantMessageEvent::TextEnd {
                    content_index,
                    content: text.text.clone(),
                    partial: partial.clone(),
                }))
            }
            _ => Err("Received text_end for non-text content".into()),
        },

        ProxyAssistantMessageEvent::ThinkingStart { content_index } => {
            set_content(
                partial,
                content_index,
                pi_ai::types::AssistantContent::Thinking(pi_ai::types::ThinkingContent::new("")),
            );
            Ok(Some(AssistantMessageEvent::ThinkingStart {
                content_index,
                partial: partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::ThinkingDelta {
            content_index,
            delta,
        } => match content_at(partial, content_index) {
            Some(pi_ai::types::AssistantContent::Thinking(thinking)) => {
                thinking.thinking += &delta;
                Ok(Some(AssistantMessageEvent::ThinkingDelta {
                    content_index,
                    delta,
                    partial: partial.clone(),
                }))
            }
            _ => Err("Received thinking_delta for non-thinking content".into()),
        },

        ProxyAssistantMessageEvent::ThinkingEnd {
            content_index,
            thinking_signature,
        } => match content_at(partial, content_index) {
            Some(pi_ai::types::AssistantContent::Thinking(thinking)) => {
                thinking.thinking_signature = thinking_signature;
                Ok(Some(AssistantMessageEvent::ThinkingEnd {
                    content_index,
                    content: thinking.thinking.clone(),
                    partial: partial.clone(),
                }))
            }
            _ => Err("Received thinking_end for non-thinking content".into()),
        },

        ProxyAssistantMessageEvent::ToolcallStart {
            content_index,
            id,
            tool_name,
        } => {
            set_content(
                partial,
                content_index,
                pi_ai::types::AssistantContent::ToolCall(ToolCall {
                    id,
                    name: tool_name,
                    arguments: serde_json::Map::new(),
                    thought_signature: None,
                    namespace: None,
                }),
            );
            Ok(Some(AssistantMessageEvent::ToolcallStart {
                content_index,
                partial: partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::ToolcallDelta {
            content_index,
            delta,
        } => match content_at(partial, content_index) {
            Some(pi_ai::types::AssistantContent::ToolCall(tool_call)) => {
                let entry = partial_json.entry(content_index).or_default();
                entry.push_str(&delta);
                let parsed = parse_streaming_json(Some(entry.as_str()));
                tool_call.arguments = parsed.as_object().cloned().unwrap_or_default();
                Ok(Some(AssistantMessageEvent::ToolcallDelta {
                    content_index,
                    delta,
                    partial: partial.clone(),
                }))
            }
            _ => Err("Received toolcall_delta for non-toolCall content".into()),
        },

        ProxyAssistantMessageEvent::ToolcallEnd {
            content_index,
            tool_call: incoming,
        } => {
            match content_at(partial, content_index) {
                Some(pi_ai::types::AssistantContent::ToolCall(tool_call)) => {
                    // Object.assign(content, proxyEvent.toolCall): the server's
                    // fields win, including metadata sent only here.
                    *tool_call = incoming.clone();
                    Ok(Some(AssistantMessageEvent::ToolcallEnd {
                        content_index,
                        tool_call: incoming,
                        partial: partial.clone(),
                    }))
                }
                _ => Ok(None),
            }
        }

        ProxyAssistantMessageEvent::Done {
            reason,
            usage,
            provider_thinking_level,
        } => {
            partial.stop_reason = reason.into();
            partial.usage = usage;
            if let Some(level) = provider_thinking_level {
                partial.provider_thinking_level = Some(level);
            }
            Ok(Some(AssistantMessageEvent::Done {
                reason,
                message: partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::Error {
            reason,
            error_message,
            usage,
            provider_thinking_level,
        } => {
            partial.stop_reason = match reason {
                StreamErrorReason::Aborted => StopReason::Aborted,
                StreamErrorReason::Error => StopReason::Error,
            };
            partial.error_message = error_message;
            partial.usage = usage;
            if let Some(level) = provider_thinking_level {
                partial.provider_thinking_level = Some(level);
            }
            Ok(Some(AssistantMessageEvent::Error {
                reason,
                error: partial.clone(),
            }))
        }
    }
}

fn content_at(
    partial: &mut AssistantMessage,
    index: u32,
) -> Option<&mut pi_ai::types::AssistantContent> {
    partial.content.get_mut(index as usize)
}

fn set_content(
    partial: &mut AssistantMessage,
    index: u32,
    content: pi_ai::types::AssistantContent,
) {
    let index = index as usize;
    while partial.content.len() <= index {
        // Placeholder blocks are overwritten before any consumer observes
        // them; the loop only pads gaps in server indices.
        partial.content.push(pi_ai::types::AssistantContent::Text(
            pi_ai::types::TextContent::new(""),
        ));
    }
    partial.content[index] = content;
}
