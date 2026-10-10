//! Shared test fixtures ported from `packages/agent/test/*` helpers and the
//! upstream `MockAssistantStream`.

#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use futures_util::StreamExt;
use parking_lot::Mutex;
use pi_agent::{
    AgentEventStream, AgentLoopConfig, AgentMessage, AgentTool, AgentToolExecute, AgentToolResult,
    BoxFuture, ConvertToLlm, StreamFn,
};
use pi_ai::types::{
    AssistantContent, AssistantMessage, Message, Model, ModelCost, ModelCostRates, ModelInput,
    StopReason, ToolCall, Usage, UserContent, UserContentField, UserMessage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use serde_json::{json, Value};

/// `Type.Object({...})` equivalent (pi-ai D6): a JSON Schema object with the
/// given `name: {"type": type}` properties, all required.
pub fn object_schema(properties: &[(&str, &str)]) -> Value {
    let props: serde_json::Map<String, Value> = properties
        .iter()
        .map(|(name, kind)| (name.to_string(), json!({ "type": kind })))
        .collect();
    let required: Vec<Value> = properties.iter().map(|(name, _)| json!(name)).collect();
    json!({ "type": "object", "properties": props, "required": required })
}

/// The empty object schema (`Type.Object({})`).
pub fn empty_object_schema() -> Value {
    object_schema(&[])
}

pub fn create_usage() -> Usage {
    Usage::default()
}

pub fn create_model() -> Model {
    Model {
        base: pi_ai::types::BaseModelFields {
            id: "mock".into(),
            name: "mock".into(),
            api: "openai-responses".into(),
            provider: "openai".into(),
            base_url: "https://example.invalid".into(),
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
        model_type: None,
        reasoning: false,
        thinking_level_map: None,
        prompt_cache: None,
        context_window: 8192,
        max_tokens: 2048,
        sampling_params: None,
        sampling_params_by_thinking_level: None,
        compat: None,
    }
}

pub fn create_assistant_message(
    content: Vec<AssistantContent>,
    stop_reason: StopReason,
) -> AssistantMessage {
    AssistantMessage {
        content,
        api: "openai-responses".into(),
        provider: "openai".into(),
        model: "mock".into(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        usage: create_usage(),
        stop_reason,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        duration_ms: None,
        timestamp: pi_agent::now_ms(),
    }
}

pub fn text_block(text: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(pi_ai::types::TextContent::new(text))
}

pub fn create_assistant_message_text(text: impl Into<String>) -> AssistantMessage {
    create_assistant_message(vec![text_block(text)], StopReason::Stop)
}

pub fn create_user_message(text: impl Into<String>) -> UserMessage {
    UserMessage {
        content: UserContentField::Text(text.into()),
        timestamp: pi_agent::now_ms(),
    }
}

pub fn create_tool_call(
    id: impl Into<String>,
    name: impl Into<String>,
    arguments: Value,
) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: arguments.as_object().cloned().unwrap_or_default(),
        thought_signature: None,
        namespace: None,
    }
}

pub fn tool_use_message(tool_calls: Vec<ToolCall>) -> AssistantMessage {
    create_assistant_message(
        tool_calls
            .into_iter()
            .map(AssistantContent::ToolCall)
            .collect(),
        StopReason::ToolUse,
    )
}

/// Assistant message with tool calls and an explicit stop reason (used by the
/// length-truncation test).
pub fn tool_calls_message(tool_calls: Vec<ToolCall>, stop_reason: StopReason) -> AssistantMessage {
    create_assistant_message(
        tool_calls
            .into_iter()
            .map(AssistantContent::ToolCall)
            .collect(),
        stop_reason,
    )
}

/// Upstream `identityConverter`: pass through standard LLM messages.
pub fn identity_converter() -> ConvertToLlm {
    Arc::new(|messages| Box::pin(async move { pi_agent::default_convert_to_llm(&messages) }))
}

/// `AgentLoopConfig { model, convertToLlm: identityConverter }`.
pub fn loop_config() -> AgentLoopConfig {
    AgentLoopConfig {
        model: create_model(),
        convert_to_llm: identity_converter(),
        ..Default::default()
    }
}

pub struct MockAssistantStream {
    pub stream: AssistantMessageEventStream,
}

impl MockAssistantStream {
    pub fn new() -> MockAssistantStream {
        MockAssistantStream {
            stream: AssistantMessageEventStream::new(),
        }
    }

    /// Push a `done` event with the given message.
    pub fn push_done(&self, message: AssistantMessage, reason: StopReason) {
        self.stream.push(pi_ai::types::AssistantMessageEvent::Done {
            reason: match reason {
                StopReason::Stop => pi_ai::types::StreamDoneReason::Stop,
                StopReason::Length => pi_ai::types::StreamDoneReason::Length,
                StopReason::ToolUse => pi_ai::types::StreamDoneReason::ToolUse,
                other => panic!("not a done reason: {other:?}"),
            },
            message,
        });
    }

    /// Push an `error` event carrying the given message.
    pub fn push_error(&self, error: AssistantMessage, reason: pi_ai::types::StreamErrorReason) {
        self.stream
            .push(pi_ai::types::AssistantMessageEvent::Error { reason, error });
    }
}

impl Default for MockAssistantStream {
    fn default() -> Self {
        MockAssistantStream::new()
    }
}

/// Wrap a synchronous `StreamFn` body (upstream mock stream functions push
/// eagerly and return the stream; crate docs D3).
pub fn stream_fn(
    f: impl Fn(
            Model,
            pi_ai::types::TranscriptContext,
            Option<pi_ai::types::SimpleStreamOptions>,
        ) -> AssistantMessageEventStream
        + Send
        + Sync
        + 'static,
) -> StreamFn {
    Arc::new(move |model, context, options| {
        let stream = f(model, context, options);
        Box::pin(std::future::ready(Ok(stream)))
    })
}

/// A stream fn that fails like upstream's `() => { throw ... }` (crate docs
/// D2).
pub fn failing_stream_fn(message: impl Into<String>) -> StreamFn {
    let message = message.into();
    Arc::new(move |_model, _context, _options| {
        let message = message.clone();
        Box::pin(std::future::ready(Err(message)))
    })
}

/// Simple synchronous tool execute wrapper.
pub fn tool_execute(
    f: impl Fn(String, serde_json::Map<String, Value>) -> Result<AgentToolResult, String>
        + Send
        + Sync
        + 'static,
) -> AgentToolExecute {
    Arc::new(move |tool_call_id, params, _signal, _on_update| {
        let outcome = f(tool_call_id, params);
        Box::pin(std::future::ready(outcome))
    })
}

/// An async tool execute wrapper with access to the abort signal.
pub fn tool_execute_async(
    f: impl Fn(
            String,
            serde_json::Map<String, Value>,
            Option<pi_ai::utils::abort::AbortSignal>,
        ) -> BoxFuture<'static, Result<AgentToolResult, String>>
        + Send
        + Sync
        + 'static,
) -> AgentToolExecute {
    Arc::new(move |tool_call_id, params, signal, _on_update| f(tool_call_id, params, signal))
}

/// `createTool(name)`: executes successfully, echoing its name.
pub fn create_tool(name: &str) -> AgentTool {
    let owned = name.to_string();
    AgentTool::new(
        name,
        format!("{name} tool"),
        empty_object_schema(),
        name,
        tool_execute(move |_id, _params| {
            Ok(AgentToolResult::new(
                vec![UserContent::Text(pi_ai::types::TextContent::new(
                    owned.clone(),
                ))],
                json!({}),
            ))
        }),
    )
}

/// An `echo` tool whose schema is `{ value: string }`; records executed
/// values into `executed`.
pub fn echo_tool(executed: Arc<Mutex<Vec<Value>>>) -> AgentTool {
    AgentTool::new(
        "echo",
        "Echo tool",
        object_schema(&[("value", "string")]),
        "Echo",
        tool_execute(move |_id, params| {
            executed
                .lock()
                .push(params.get("value").cloned().unwrap_or(Value::Null));
            Ok(AgentToolResult::new(
                vec![UserContent::Text(pi_ai::types::TextContent::new(format!(
                    "echoed: {}",
                    params
                        .get("value")
                        .and_then(|value| value.as_str())
                        .unwrap_or("")
                )))],
                json!({ "value": params.get("value").cloned().unwrap_or(Value::Null) }),
            ))
        }),
    )
}

/// Drain the event stream, then resolve the run's new messages.
pub async fn run_to_completion(
    stream: &AgentEventStream,
) -> (Vec<pi_agent::AgentEvent>, Vec<AgentMessage>) {
    let mut rx = stream.clone();
    let mut events = Vec::new();
    while let Some(event) = rx.next().await {
        events.push(event);
    }
    let messages = stream.result().await;
    (events, messages)
}

pub fn event_types(events: &[pi_agent::AgentEvent]) -> Vec<&'static str> {
    events.iter().map(|event| event.event_type()).collect()
}

/// Roles of the run's new messages.
pub fn message_roles(messages: &[AgentMessage]) -> Vec<String> {
    messages
        .iter()
        .map(|message| message.role().to_string())
        .collect()
}

/// Collect `message_start` (role: tag) markers for sequence assertions.
pub fn message_start_markers(events: &[pi_agent::AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            pi_agent::AgentEvent::MessageStart { message } => match message {
                AgentMessage::Llm(Message::ToolResult(tool_result)) => {
                    Some(format!("tool:{}", tool_result.tool_call_id))
                }
                AgentMessage::Llm(Message::User(user)) => match &user.content {
                    UserContentField::Text(text) => Some(text.clone()),
                    UserContentField::Blocks(blocks) => {
                        blocks.iter().find_map(|block| match block {
                            UserContent::Text(text) => Some(text.text.clone()),
                            _ => None,
                        })
                    }
                },
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// Roles of `message_end` events (LLM messages only).
pub fn message_end_roles(events: &[pi_agent::AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            pi_agent::AgentEvent::MessageEnd { message } => Some(message.role().to_string()),
            _ => None,
        })
        .collect()
}

pub fn tool_execution_end_ids(events: &[pi_agent::AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            pi_agent::AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                Some(tool_call_id.clone())
            }
            _ => None,
        })
        .collect()
}

pub fn user_texts(context: &pi_ai::types::TranscriptContext) -> Vec<String> {
    context
        .messages()
        .iter()
        .filter_map(|message| match message {
            Message::User(user) => match &user.content {
                UserContentField::Text(text) => Some(text.clone()),
                UserContentField::Blocks(blocks) => blocks.iter().find_map(|block| match block {
                    UserContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                }),
            },
            _ => None,
        })
        .collect()
}

/// Monotonic call counter shared between a mock stream fn and the test.
pub fn call_counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

pub fn calls(counter: &Arc<AtomicUsize>) -> usize {
    counter.load(Ordering::SeqCst)
}
