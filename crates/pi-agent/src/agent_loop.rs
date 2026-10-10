//! Port of `packages/agent/src/agent-loop.ts` — the multi-turn agent loop
//! that works with [`AgentMessage`] throughout and transforms to
//! [`Message`]s only at the LLM call boundary.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures_util::StreamExt;
use parking_lot::Mutex;
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, Context, Message, ModelThinkingLevel, StopReason,
    SystemContent, SystemMessage, Tool, ToolCall, ToolResultMessage, UserContent,
};
use pi_ai::utils::abort::AbortSignal;
use pi_ai::utils::event_stream::EventStream;
use pi_ai::utils::transcript::{
    get_current_tools, get_tool_state_changes, normalize_context, to_tool_declaration,
};
use pi_ai::utils::validation::validate_tool_arguments;
use serde_json::Value;

use crate::types::{
    after_tool_call_context, before_tool_call_context, now_ms, AfterToolCallHook, AgentContext,
    AgentError, AgentEvent, AgentEventSink, AgentLoopConfig, AgentMessage, AgentTool,
    AgentToolCall, AgentToolCallOutcome, AgentToolResult, AgentToolUpdateCallback,
    BeforeToolCallHook, BoxFuture, PrepareNextTurnContext, StreamFn,
};

/// `(event) => Promise<void> | void` sink of the low-level loop.
pub type ToolUpdateSink = Arc<dyn Fn(AgentToolResult) -> BoxFuture<'static, ()> + Send + Sync>;

/// Event stream returned by [`agent_loop`] / [`agent_loop_continue`]:
/// completes on `agent_end` with the run's new messages.
pub type AgentEventStream = EventStream<AgentEvent, Vec<AgentMessage>>;

/// Start an agent loop with new prompt messages. The prompts are added to
/// the context and events are emitted for them.
pub fn agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<AbortSignal>,
    stream_fn: Option<StreamFn>,
) -> AgentEventStream {
    let stream = create_agent_stream();
    let producer = stream.clone();
    tokio::spawn(async move {
        let sink: AgentEventSink = {
            let producer = producer.clone();
            Arc::new(move |event| {
                let producer = producer.clone();
                Box::pin(async move {
                    producer.push(event);
                })
            })
        };
        let messages = run_agent_loop(prompts, context, config, sink, signal, stream_fn).await;
        match messages {
            Ok(messages) => producer.end(Some(messages)),
            // Upstream: the loop task rejects and the stream's result promise
            // never settles (crate docs D2).
            Err(_) => producer.end(None),
        }
    });
    stream
}

/// Continue an agent loop from the current context without adding a new
/// message. Used for retries — context already has a user message or tool
/// results.
///
/// **Important:** the last message in context must convert to a `user` or
/// `toolResult` message via `convertToLlm`. If it doesn't, the LLM provider
/// will reject the request. This cannot be validated here since
/// `convertToLlm` is only called once per turn.
pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<AbortSignal>,
    stream_fn: Option<StreamFn>,
) -> Result<AgentEventStream, AgentError> {
    validate_continue(&context)?;
    let stream = create_agent_stream();
    let producer = stream.clone();
    tokio::spawn(async move {
        let sink: AgentEventSink = {
            let producer = producer.clone();
            Arc::new(move |event| {
                let producer = producer.clone();
                Box::pin(async move {
                    producer.push(event);
                })
            })
        };
        let messages = run_agent_loop_continue(context, config, sink, signal, stream_fn).await;
        match messages {
            Ok(messages) => producer.end(Some(messages)),
            Err(_) => producer.end(None),
        }
    });
    Ok(stream)
}

fn validate_continue(context: &AgentContext) -> Result<(), AgentError> {
    if context.messages.is_empty() {
        return Err(AgentError("Cannot continue: no messages in context".into()));
    }
    if context.messages.last().expect("nonempty").role() == "assistant" {
        return Err(AgentError(
            "Cannot continue from message role: assistant".into(),
        ));
    }
    Ok(())
}

/// Upstream `runAgentLoop`.
pub async fn run_agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Option<AbortSignal>,
    stream_fn: Option<StreamFn>,
) -> Result<Vec<AgentMessage>, AgentError> {
    let stream_fn = stream_fn.unwrap_or_else(crate::stream_fn::get_default_stream_fn);
    let initial_messages = declare_tool_changes(&context, prompts);
    let new_messages: Vec<AgentMessage> = initial_messages.clone();
    let mut context_messages = context.messages.clone();
    context_messages.extend(initial_messages.iter().cloned());
    let current_context = AgentContext {
        messages: context_messages,
        tools: context.tools.clone(),
    };

    emit(AgentEvent::AgentStart).await;
    emit(AgentEvent::TurnStart).await;
    for message in &initial_messages {
        emit(AgentEvent::MessageStart {
            message: message.clone(),
        })
        .await;
        emit(AgentEvent::MessageEnd {
            message: message.clone(),
        })
        .await;
    }

    run_loop(
        current_context,
        new_messages,
        config,
        signal,
        emit,
        stream_fn,
    )
    .await
}

/// Upstream `runAgentLoopContinue`.
pub async fn run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Option<AbortSignal>,
    stream_fn: Option<StreamFn>,
) -> Result<Vec<AgentMessage>, AgentError> {
    validate_continue(&context)?;
    let stream_fn = stream_fn.unwrap_or_else(crate::stream_fn::get_default_stream_fn);
    let new_messages: Vec<AgentMessage> = Vec::new();
    let current_context = AgentContext {
        messages: context.messages.clone(),
        tools: context.tools.clone(),
    };

    emit(AgentEvent::AgentStart).await;
    emit(AgentEvent::TurnStart).await;

    run_loop(
        current_context,
        new_messages,
        config,
        signal,
        emit,
        stream_fn,
    )
    .await
}

fn create_agent_stream() -> AgentEventStream {
    EventStream::new(
        |event| matches!(event, AgentEvent::AgentEnd { .. }),
        |event| match event {
            AgentEvent::AgentEnd { messages } => messages.clone(),
            _ => Vec::new(),
        },
    )
}

/// Main loop logic shared by [`run_agent_loop`] and
/// [`run_agent_loop_continue`].
async fn run_loop(
    mut current_context: AgentContext,
    mut new_messages: Vec<AgentMessage>,
    mut config: AgentLoopConfig,
    signal: Option<AbortSignal>,
    emit: AgentEventSink,
    stream_fn: StreamFn,
) -> Result<Vec<AgentMessage>, AgentError> {
    let mut last_completed_turn: Option<PrepareNextTurnContext> = None;
    let mut explicit_continuation = false;
    // Check for steering messages at start (user may have typed while waiting)
    let mut pending_messages: Vec<AgentMessage> = poll_steering(&config).await;

    // Outer loop: continues when queued follow-up messages arrive after agent would stop
    loop {
        let mut has_more_tool_calls = true;

        // Inner loop: process tool calls and steering messages
        while has_more_tool_calls || !pending_messages.is_empty() {
            let mut prepared_messages: Vec<AgentMessage> = Vec::new();
            if let Some(last_turn) = last_completed_turn.take() {
                if let Some(hook) = &config.prepare_next_turn {
                    if let Some(next_turn_snapshot) = hook(last_turn).await {
                        if let Some(context) = next_turn_snapshot.context {
                            current_context = context;
                        }
                        prepared_messages = next_turn_snapshot.messages;
                        if let Some(model) = next_turn_snapshot.model {
                            config.model = model;
                        }
                        if let Some(thinking_level) = next_turn_snapshot.thinking_level {
                            config.simple.reasoning = thinking_level.to_reasoning();
                        }
                    }
                }
                // Preparation can be long-running (for example, compaction). Pick up steering
                // queued while it ran. Only poll again if the earlier poll returned nothing;
                // otherwise one-at-a-time mode would deliver two messages in this turn.
                if pending_messages.is_empty() {
                    pending_messages = poll_steering(&config).await;
                }
                emit(AgentEvent::TurnStart).await;
            }

            // Process prepared and queued messages before the next assistant response.
            let mut to_append = prepared_messages;
            to_append.extend(pending_messages.drain(..));
            for message in declare_tool_changes(&current_context, to_append) {
                emit(AgentEvent::MessageStart {
                    message: message.clone(),
                })
                .await;
                emit(AgentEvent::MessageEnd {
                    message: message.clone(),
                })
                .await;
                current_context.messages.push(message.clone());
                new_messages.push(message);
            }

            if let Some(hook) = &config.prepare_request {
                let request = crate::types::PrepareRequestContext {
                    context: current_context.clone(),
                    model: config.model.clone(),
                    thinking_level: crate::types::AgentThinkingLevel::from_reasoning(
                        config.simple.reasoning,
                    ),
                };
                if let Some(request_update) = hook(request, signal.clone()).await {
                    if let Some(context) = request_update.context {
                        current_context = context;
                    }
                    if let Some(model) = request_update.model {
                        config.model = model;
                    }
                    if let Some(thinking_level) = request_update.thinking_level {
                        config.simple.reasoning = thinking_level.to_reasoning();
                    }
                }
            }

            // Stream assistant response
            let message = stream_assistant_response(
                &mut current_context,
                &config,
                signal.clone(),
                &emit,
                &stream_fn,
            )
            .await?;
            new_messages.push(AgentMessage::assistant(message.clone()));

            if message.stop_reason == StopReason::Error
                || message.stop_reason == StopReason::Aborted
            {
                last_completed_turn = Some(PrepareNextTurnContext {
                    message: message.clone(),
                    tool_results: Vec::new(),
                    context: current_context.clone(),
                    new_messages: new_messages.clone(),
                });
                if let Some(hook) = &config.finish_turn {
                    hook(
                        last_completed_turn.clone().expect("just set"),
                        signal.clone(),
                    )
                    .await;
                }
                emit(AgentEvent::TurnEnd {
                    message: AgentMessage::assistant(message.clone()),
                    tool_results: Vec::new(),
                })
                .await;
                emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await;
                return Ok(new_messages);
            }

            // Check for tool calls
            let tool_calls: Vec<ToolCall> = message
                .content
                .iter()
                .filter_map(|content| match content {
                    pi_ai::types::AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
                    _ => None,
                })
                .collect();

            let mut tool_results: Vec<ToolResultMessage> = Vec::new();
            has_more_tool_calls = false;
            if !tool_calls.is_empty() {
                // A "length" stop means the output was cut off by the token limit, so
                // every tool call in the message may carry truncated arguments. Fail
                // them all instead of executing potentially borked calls.
                let executed_tool_batch = if message.stop_reason == StopReason::Length {
                    fail_tool_calls_from_truncated_message(&tool_calls, &emit).await
                } else {
                    execute_tool_calls(&current_context, &message, &config, signal.clone(), &emit)
                        .await
                };
                tool_results = executed_tool_batch.messages;
                has_more_tool_calls = !executed_tool_batch.terminate;

                for result in &tool_results {
                    current_context
                        .messages
                        .push(AgentMessage::tool_result(result.clone()));
                    new_messages.push(AgentMessage::tool_result(result.clone()));
                }
            }

            last_completed_turn = Some(PrepareNextTurnContext {
                message: message.clone(),
                tool_results: tool_results.clone(),
                context: current_context.clone(),
                new_messages: new_messages.clone(),
            });
            let decision = match &config.finish_turn {
                Some(hook) => {
                    hook(
                        last_completed_turn.clone().expect("just set"),
                        signal.clone(),
                    )
                    .await
                }
                None => None,
            };
            emit(AgentEvent::TurnEnd {
                message: AgentMessage::assistant(message.clone()),
                tool_results,
            })
            .await;

            if decision == Some(crate::types::AgentTurnDecision::End) {
                emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await;
                return Ok(new_messages);
            }

            explicit_continuation = decision == Some(crate::types::AgentTurnDecision::Continue);
            pending_messages = poll_steering(&config).await;
            if has_more_tool_calls || !pending_messages.is_empty() {
                explicit_continuation = false;
            }
        }

        // Agent would stop here. Check for follow-up messages.
        let follow_up_messages = match &config.get_follow_up_messages {
            Some(hook) => hook().await,
            None => Vec::new(),
        };
        if !follow_up_messages.is_empty() {
            // Set as pending so inner loop processes them
            explicit_continuation = false;
            pending_messages = follow_up_messages;
            continue;
        }

        // No natural request was selected, so fulfill the continuation decision with one context-only turn.
        if explicit_continuation {
            explicit_continuation = false;
            continue;
        }

        // No more messages, exit
        break;
    }

    emit(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    })
    .await;
    Ok(new_messages)
}

async fn poll_steering(config: &AgentLoopConfig) -> Vec<AgentMessage> {
    match &config.get_steering_messages {
        Some(hook) => hook().await,
        None => Vec::new(),
    }
}

const NO_CHANGES_TOOLS: Option<Vec<Tool>> = None;

/// Declare tool loadout changes to the model.
///
/// `context.tools` is what the runtime can execute; the transcript's system
/// messages declare what the model may call. Before each request the
/// difference becomes `toolsAdded` and `toolsRemoved` on a system message.
/// When a pending system message exists, its tool fields are treated as
/// intent and replaced with the delta between the committed transcript and
/// the executable set, so replay always yields exactly `context.tools`.
/// Otherwise a new system message is inserted before the first non-system
/// pending message.
fn declare_tool_changes(
    context: &AgentContext,
    pending_messages: Vec<AgentMessage>,
) -> Vec<AgentMessage> {
    let mut system_index: Option<usize> = None;
    for index in (0..pending_messages.len()).rev() {
        if pending_messages[index].role() == "system" {
            system_index = Some(index);
            break;
        }
    }
    let pending: Option<SystemMessage> =
        system_index.and_then(|index| match &pending_messages[index] {
            AgentMessage::Llm(Message::System(system)) => Some(system.clone()),
            _ => None,
        });
    let baseline: Vec<AgentMessage> = match (&pending, system_index) {
        (Some(pending_system), Some(index)) => pending_messages
            .iter()
            .enumerate()
            .map(|(position, message)| {
                if position == index {
                    AgentMessage::system(with_tool_changes(pending_system, &[], &[]))
                } else {
                    message.clone()
                }
            })
            .collect(),
        _ => pending_messages.clone(),
    };
    let desired_tools: Vec<Tool> = context
        .tools
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|tool| to_tool_declaration(&tool.tool))
        .collect();
    let mut committed: Vec<Message> = context
        .messages
        .iter()
        .filter_map(|message| message.as_llm().cloned())
        .collect();
    committed.extend(
        baseline
            .iter()
            .filter_map(|message| message.as_llm().cloned()),
    );
    let changes = get_tool_state_changes(&get_current_tools(&committed), &desired_tools);
    let unchanged = changes.tools_added.is_empty() && changes.tools_removed.is_empty();

    if let Some(pending_system) = &pending {
        // Keep the caller's message object when it already declares no tool changes.
        let pending_empty = pending_system
            .tools_added
            .as_ref()
            .is_none_or(|tools| tools.is_empty())
            && pending_system
                .tools_removed
                .as_ref()
                .is_none_or(|tools| tools.is_empty());
        if unchanged && pending_empty {
            return pending_messages;
        }
        let index = system_index.expect("pending implies index");
        let mut result = baseline;
        result[index] = AgentMessage::system(with_tool_changes(
            pending_system,
            &changes.tools_added,
            &changes.tools_removed,
        ));
        return result;
    }
    if unchanged {
        return pending_messages;
    }
    let update = with_tool_changes(
        &SystemMessage {
            content: SystemContent::Text(String::new()),
            sections: None,
            tools_added: NO_CHANGES_TOOLS,
            tools_removed: None,
            timestamp: now_ms(),
        },
        &changes.tools_added,
        &changes.tools_removed,
    );
    let insert_index = pending_messages
        .iter()
        .position(|message| message.role() != "system")
        .unwrap_or(pending_messages.len());
    let mut result = Vec::with_capacity(pending_messages.len() + 1);
    result.extend_from_slice(&pending_messages[..insert_index]);
    result.push(AgentMessage::system(update));
    result.extend_from_slice(&pending_messages[insert_index..]);
    result
}

/// Copy a system message with its tool fields replaced by `changes`; empty
/// lists omit the field.
fn with_tool_changes(
    message: &SystemMessage,
    tools_added: &[Tool],
    tools_removed: &[pi_ai::types::ToolReference],
) -> SystemMessage {
    SystemMessage {
        content: message.content.clone(),
        sections: message.sections.clone(),
        tools_added: if tools_added.is_empty() {
            None
        } else {
            Some(tools_added.to_vec())
        },
        tools_removed: if tools_removed.is_empty() {
            None
        } else {
            Some(tools_removed.to_vec())
        },
        timestamp: message.timestamp,
    }
}

/// Stream an assistant response from the LLM. This is where
/// `AgentMessage[]` gets transformed to `Message[]` for the LLM.
async fn stream_assistant_response(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    signal: Option<AbortSignal>,
    emit: &AgentEventSink,
    stream_fn: &StreamFn,
) -> Result<AssistantMessage, AgentError> {
    // Apply context transform if configured (AgentMessage[] → AgentMessage[])
    let mut messages = context.messages.clone();
    if let Some(hook) = &config.transform_context {
        messages = hook(messages, signal.clone()).await;
    }

    // Convert to LLM-compatible messages (AgentMessage[] → Message[])
    let llm_messages = (config.convert_to_llm)(messages).await;

    let llm_context = normalize_context(&Context {
        system_prompt: None,
        messages: llm_messages,
        tools: None,
    });

    // Resolve API key (important for expiring tokens). `"" || config.apiKey`
    // upstream: an empty resolved key falls through.
    let resolved_api_key = match &config.get_api_key {
        Some(hook) => hook(config.model.provider())
            .await
            .filter(|key| !key.is_empty()),
        None => None,
    }
    .or_else(|| config.simple.base.api_key.clone());

    let mut options = config.simple.clone();
    options.base.api_key = resolved_api_key;
    options.base.signal = signal.clone();

    let response = (stream_fn)(config.model.clone(), llm_context, Some(options))
        .await
        .map_err(AgentError)?;
    // Record the requested level, whichever stream function answered.
    let requested_level: ModelThinkingLevel = match config.simple.reasoning {
        Some(level) => level.into(),
        None => ModelThinkingLevel::Off,
    };

    let mut partial_message: Option<AssistantMessage> = None;
    let mut added_partial = false;

    let mut rx = response.clone();
    while let Some(event) = rx.next().await {
        match event {
            AssistantMessageEvent::Start { partial } => {
                partial_message = Some(partial.clone());
                context
                    .messages
                    .push(AgentMessage::assistant(partial.clone()));
                added_partial = true;
                emit(AgentEvent::MessageStart {
                    message: AgentMessage::assistant(partial.clone()),
                })
                .await;
            }
            AssistantMessageEvent::TextStart { ref partial, .. }
            | AssistantMessageEvent::TextDelta { ref partial, .. }
            | AssistantMessageEvent::TextEnd { ref partial, .. }
            | AssistantMessageEvent::ThinkingStart { ref partial, .. }
            | AssistantMessageEvent::ThinkingDelta { ref partial, .. }
            | AssistantMessageEvent::ThinkingEnd { ref partial, .. }
            | AssistantMessageEvent::ToolcallStart { ref partial, .. }
            | AssistantMessageEvent::ToolcallDelta { ref partial, .. }
            | AssistantMessageEvent::ToolcallEnd { ref partial, .. } => {
                if partial_message.is_some() {
                    partial_message = Some(partial.clone());
                    let last = context.messages.len() - 1;
                    context.messages[last] = AgentMessage::assistant(partial.clone());
                    emit(AgentEvent::MessageUpdate {
                        message: AgentMessage::assistant(partial.clone()),
                        assistant_message_event: event,
                    })
                    .await;
                }
            }
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => {
                let mut final_message = response.result().await;
                final_message.thinking_level = Some(requested_level);
                finish_assistant_message(context, added_partial, final_message.clone(), emit).await;
                return Ok(final_message);
            }
        }
    }

    // Stream ended without a completing event: same tail as upstream.
    let mut final_message = response.result().await;
    final_message.thinking_level = Some(requested_level);
    if !added_partial {
        emit(AgentEvent::MessageStart {
            message: AgentMessage::assistant(final_message.clone()),
        })
        .await;
    }
    emit(AgentEvent::MessageEnd {
        message: AgentMessage::assistant(final_message.clone()),
    })
    .await;
    Ok(final_message)
}

async fn finish_assistant_message(
    context: &mut AgentContext,
    added_partial: bool,
    final_message: AssistantMessage,
    emit: &AgentEventSink,
) {
    if added_partial {
        let last = context.messages.len() - 1;
        context.messages[last] = AgentMessage::assistant(final_message.clone());
    } else {
        context
            .messages
            .push(AgentMessage::assistant(final_message.clone()));
        emit(AgentEvent::MessageStart {
            message: AgentMessage::assistant(final_message.clone()),
        })
        .await;
    }
    emit(AgentEvent::MessageEnd {
        message: AgentMessage::assistant(final_message.clone()),
    })
    .await;
}

struct ExecutedToolCallBatch {
    messages: Vec<ToolResultMessage>,
    terminate: bool,
}

/// Fail all tool calls from an assistant message that was truncated by the
/// output token limit. Streamed tool-call arguments are finalized with a
/// best-effort JSON salvage parser, so a truncated message can yield tool
/// calls whose arguments parse and validate but are silently incomplete.
/// None of them are safe to execute; report each as an error so the model
/// can re-issue them.
async fn fail_tool_calls_from_truncated_message(
    tool_calls: &[ToolCall],
    emit: &AgentEventSink,
) -> ExecutedToolCallBatch {
    let mut messages: Vec<ToolResultMessage> = Vec::new();
    for tool_call in tool_calls {
        emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: Value::Object(tool_call.arguments.clone()),
        })
        .await;
        let finalized = AgentToolCallOutcome {
			tool_call: tool_call.clone(),
			result: create_error_tool_result(format!(
				"Tool call \"{}\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
				tool_call.name
			)),
			is_error: true,
			duration_ms: None,
		};
        emit_tool_execution_end(&finalized, emit).await;
        let tool_result_message = create_tool_result_message(&finalized);
        emit_tool_result_message(&tool_result_message, emit).await;
        messages.push(tool_result_message);
    }
    ExecutedToolCallBatch {
        messages,
        terminate: false,
    }
}

/// Execute tool calls from an assistant message.
async fn execute_tool_calls(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    config: &AgentLoopConfig,
    signal: Option<AbortSignal>,
    emit: &AgentEventSink,
) -> ExecutedToolCallBatch {
    let tool_calls: Vec<ToolCall> = assistant_message
        .content
        .iter()
        .filter_map(|content| match content {
            pi_ai::types::AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .collect();
    let tools = current_context.tools.as_deref().unwrap_or(&[]);
    let has_sequential_tool_call = tool_calls.iter().any(|tool_call| {
        tools
            .iter()
            .find(|tool| tool.name() == tool_call.name)
            .and_then(|tool| tool.execution_mode)
            == Some(crate::types::ToolExecutionMode::Sequential)
    });
    if config.tool_execution == Some(crate::types::ToolExecutionMode::Sequential)
        || has_sequential_tool_call
    {
        execute_tool_calls_sequential(
            current_context,
            assistant_message,
            &tool_calls,
            config,
            signal,
            emit,
        )
        .await
    } else {
        execute_tool_calls_parallel(
            current_context,
            assistant_message,
            &tool_calls,
            config,
            signal,
            emit,
        )
        .await
    }
}

async fn execute_tool_calls_sequential(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    signal: Option<AbortSignal>,
    emit: &AgentEventSink,
) -> ExecutedToolCallBatch {
    let mut finalized_calls: Vec<AgentToolCallOutcome> = Vec::new();
    let mut messages: Vec<ToolResultMessage> = Vec::new();

    for tool_call in tool_calls {
        emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: Value::Object(tool_call.arguments.clone()),
        })
        .await;

        let preparation = prepare_tool_call(
            current_context,
            assistant_message,
            tool_call,
            config,
            signal.as_ref(),
            None,
        )
        .await;
        let finalized = match preparation {
            PreparedToolCall::Immediate { result, is_error } => AgentToolCallOutcome {
                tool_call: tool_call.clone(),
                result,
                is_error,
                duration_ms: None,
            },
            PreparedToolCall::Prepared { tool, args } => {
                let prepared = PreparedCall {
                    tool_call: tool_call.clone(),
                    tool,
                    args,
                };
                let executed = execute_prepared_tool_call(
                    &prepared,
                    signal.as_ref(),
                    &emit_tool_execution_update(tool_call, emit),
                )
                .await;
                finalize_executed_tool_call(
                    current_context,
                    assistant_message,
                    &prepared,
                    executed,
                    config,
                    signal.as_ref(),
                )
                .await
            }
        };

        emit_tool_execution_end(&finalized, emit).await;
        let tool_result_message = create_tool_result_message(&finalized);
        emit_tool_result_message(&tool_result_message, emit).await;
        finalized_calls.push(finalized);
        messages.push(tool_result_message);

        if signal.as_ref().is_some_and(|signal| signal.aborted()) {
            break;
        }
    }

    ExecutedToolCallBatch {
        messages,
        terminate: should_terminate_tool_batch(&finalized_calls),
    }
}

async fn execute_tool_calls_parallel(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    signal: Option<AbortSignal>,
    emit: &AgentEventSink,
) -> ExecutedToolCallBatch {
    enum FinalizedToolCallEntry {
        Ready(AgentToolCallOutcome),
        Pending {
            tool_call: ToolCall,
            tool: AgentTool,
            args: Value,
        },
    }

    let mut finalized_calls: Vec<FinalizedToolCallEntry> = Vec::new();

    for tool_call in tool_calls {
        emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: Value::Object(tool_call.arguments.clone()),
        })
        .await;

        let preparation = prepare_tool_call(
            current_context,
            assistant_message,
            tool_call,
            config,
            signal.as_ref(),
            None,
        )
        .await;
        match preparation {
            PreparedToolCall::Immediate { result, is_error } => {
                let finalized = AgentToolCallOutcome {
                    tool_call: tool_call.clone(),
                    result,
                    is_error,
                    duration_ms: None,
                };
                emit_tool_execution_end(&finalized, emit).await;
                finalized_calls.push(FinalizedToolCallEntry::Ready(finalized));
                if signal.as_ref().is_some_and(|signal| signal.aborted()) {
                    break;
                }
            }
            PreparedToolCall::Prepared { tool, args } => {
                finalized_calls.push(FinalizedToolCallEntry::Pending {
                    tool_call: tool_call.clone(),
                    tool,
                    args,
                });
                if signal.as_ref().is_some_and(|signal| signal.aborted()) {
                    break;
                }
            }
        }
    }

    // Prepare sequential; execute allowed tools concurrently. Events emitted
    // inside each branch appear in completion order; the tool-result
    // messages below are emitted later in assistant source order.
    let shared_context = Arc::new(current_context.clone());
    let shared_message = Arc::new(assistant_message.clone());
    let shared_config = Arc::new(config.clone());
    let futures = finalized_calls.into_iter().map(|entry| {
        let shared_context = shared_context.clone();
        let shared_message = shared_message.clone();
        let shared_config = shared_config.clone();
        let signal = signal.clone();
        let emit = emit.clone();
        async move {
            match entry {
                FinalizedToolCallEntry::Ready(finalized) => finalized,
                FinalizedToolCallEntry::Pending {
                    tool_call,
                    tool,
                    args,
                } => {
                    if signal.as_ref().is_some_and(|signal| signal.aborted()) {
                        let finalized = AgentToolCallOutcome {
                            tool_call: tool_call.clone(),
                            result: create_error_tool_result("Operation aborted"),
                            is_error: true,
                            duration_ms: None,
                        };
                        emit_tool_execution_end(&finalized, &emit).await;
                        return finalized;
                    }
                    let prepared = PreparedCall {
                        tool_call: tool_call.clone(),
                        tool,
                        args,
                    };
                    let executed = execute_prepared_tool_call(
                        &prepared,
                        signal.as_ref(),
                        &emit_tool_execution_update(&tool_call, &emit),
                    )
                    .await;
                    let finalized = finalize_executed_tool_call(
                        &shared_context,
                        shared_message.as_ref(),
                        &prepared,
                        executed,
                        shared_config.as_ref(),
                        signal.as_ref(),
                    )
                    .await;
                    emit_tool_execution_end(&finalized, &emit).await;
                    finalized
                }
            }
        }
    });
    let ordered_finalized_calls = futures_util::future::join_all(futures).await;

    let mut messages: Vec<ToolResultMessage> = Vec::new();
    for finalized in &ordered_finalized_calls {
        let tool_result_message = create_tool_result_message(finalized);
        emit_tool_result_message(&tool_result_message, emit).await;
        messages.push(tool_result_message);
    }

    ExecutedToolCallBatch {
        messages,
        terminate: should_terminate_tool_batch(&ordered_finalized_calls),
    }
}

fn should_terminate_tool_batch(finalized_calls: &[AgentToolCallOutcome]) -> bool {
    !finalized_calls.is_empty()
        && finalized_calls
            .iter()
            .all(|finalized| finalized.result.terminate == Some(true))
}

fn prepare_tool_call_arguments(tool: &AgentTool, tool_call: &AgentToolCall) -> AgentToolCall {
    let Some(prepare) = &tool.prepare_arguments else {
        return tool_call.clone();
    };
    let raw = Value::Object(tool_call.arguments.clone());
    let prepared_arguments = prepare(raw.clone());
    if prepared_arguments == raw {
        return tool_call.clone();
    }
    let mut updated = tool_call.clone();
    updated.arguments = prepared_arguments.as_object().cloned().unwrap_or_default();
    updated
}

enum PreparedToolCall {
    Prepared {
        tool: AgentTool,
        args: Value,
    },
    Immediate {
        result: AgentToolResult,
        is_error: bool,
    },
}

/// Internal shape after `PreparedToolCall` resolution (tool call + tool +
/// validated args together).
struct PreparedCall {
    tool_call: AgentToolCall,
    tool: AgentTool,
    args: Value,
}

/// The `beforeToolCall`/`afterToolCall` hook pair of
/// [`AgentLoopConfig`]. Split access keeps the loop code readable.
struct ToolCallHooks<'a> {
    before_tool_call: &'a Option<BeforeToolCallHook>,
    after_tool_call: &'a Option<AfterToolCallHook>,
}

impl<'a> From<&'a AgentLoopConfig> for ToolCallHooks<'a> {
    fn from(config: &'a AgentLoopConfig) -> ToolCallHooks<'a> {
        ToolCallHooks {
            before_tool_call: &config.before_tool_call,
            after_tool_call: &config.after_tool_call,
        }
    }
}

impl<'a> From<&'a crate::agent_loop::RunToolCallOptions> for ToolCallHooks<'a> {
    fn from(options: &'a RunToolCallOptions) -> ToolCallHooks<'a> {
        ToolCallHooks {
            before_tool_call: &options.before_tool_call,
            after_tool_call: &options.after_tool_call,
        }
    }
}

async fn prepare_tool_call(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_call: &AgentToolCall,
    hooks: impl Into<ToolCallHooks<'_>>,
    signal: Option<&AbortSignal>,
    forced_tools: Option<&[AgentTool]>,
) -> PreparedToolCall {
    let hooks = hooks.into();
    let tools: &[AgentTool] = match forced_tools {
        Some(tools) => tools,
        None => current_context.tools.as_deref().unwrap_or(&[]),
    };
    let Some(tool) = tools
        .iter()
        .find(|tool| tool.name() == tool_call.name)
        .cloned()
    else {
        return PreparedToolCall::Immediate {
            result: create_error_tool_result(format!("Tool {} not found", tool_call.name)),
            is_error: true,
        };
    };

    let outcome: Result<PreparedToolCall, String> = async {
        let prepared_tool_call = prepare_tool_call_arguments(&tool, tool_call);
        let validated_args =
            validate_tool_arguments(&tool.tool, &prepared_tool_call).map_err(|error| error.0)?;
        let mut validated_args = Value::Object(validated_args);
        if let Some(hook) = hooks.before_tool_call {
            let before_result = hook(
                before_tool_call_context(
                    assistant_message.clone(),
                    tool_call.clone(),
                    validated_args.clone(),
                    current_context.clone(),
                ),
                signal.cloned(),
            )
            .await?;
            if signal.is_some_and(|signal| signal.aborted()) {
                return Ok(PreparedToolCall::Immediate {
                    result: create_error_tool_result("Operation aborted"),
                    is_error: true,
                });
            }
            if let Some(before_result) = before_result {
                if before_result.block == Some(true) {
                    let mut result = create_error_tool_result(
                        before_result
                            .reason
                            .unwrap_or_else(|| "Tool execution was blocked".into()),
                    );
                    if before_result.terminate == Some(true) {
                        result.terminate = Some(true);
                    }
                    return Ok(PreparedToolCall::Immediate {
                        result,
                        is_error: true,
                    });
                }
                // Crate docs D8: replaced validated args (upstream in-place mutation).
                if let Some(args) = before_result.args {
                    validated_args = args;
                }
            }
        }
        if signal.is_some_and(|signal| signal.aborted()) {
            return Ok(PreparedToolCall::Immediate {
                result: create_error_tool_result("Operation aborted"),
                is_error: true,
            });
        }
        Ok(PreparedToolCall::Prepared {
            tool,
            args: validated_args,
        })
    }
    .await;
    match outcome {
        Ok(prepared) => prepared,
        Err(message) => PreparedToolCall::Immediate {
            result: create_error_tool_result(message),
            is_error: true,
        },
    }
}

fn emit_tool_execution_update(tool_call: &AgentToolCall, emit: &AgentEventSink) -> ToolUpdateSink {
    let tool_call = tool_call.clone();
    let emit = emit.clone();
    Arc::new(move |partial_result| {
        let tool_call = tool_call.clone();
        let emit = emit.clone();
        Box::pin(async move {
            emit(AgentEvent::ToolExecutionUpdate {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                args: Value::Object(tool_call.arguments.clone()),
                partial_result,
            })
            .await;
        })
    })
}

/// Options for [`run_tool_call`].
pub struct RunToolCallOptions {
    /// Tools the call resolves against.
    pub tools: Vec<AgentTool>,
    /// Passed to the hooks as the message that issued the call.
    pub assistant_message: AssistantMessage,
    /// Passed to the hooks as the current agent context.
    pub context: AgentContext,
    pub signal: Option<AbortSignal>,
    pub on_update: Option<ToolUpdateSink>,
    pub before_tool_call: Option<BeforeToolCallHook>,
    pub after_tool_call: Option<AfterToolCallHook>,
}

/// Run one tool call through the same steps as a model-issued call:
/// argument preparation, schema validation, `beforeToolCall`, execution,
/// and `afterToolCall`. Emits no events and adds no messages. Tools that
/// call other tools use this so the hooks (for example permission checks)
/// apply to those calls too.
///
/// Never fails for tool failures: unknown tools, validation errors, blocked
/// calls, and thrown errors come back as `isError: true`.
pub async fn run_tool_call(
    tool_call: AgentToolCall,
    options: RunToolCallOptions,
) -> AgentToolCallOutcome {
    let preparation = prepare_tool_call(
        &options.context,
        &options.assistant_message,
        &tool_call,
        &options,
        options.signal.as_ref(),
        Some(&options.tools),
    )
    .await;
    let prepared = match preparation {
        PreparedToolCall::Immediate { result, is_error } => {
            return AgentToolCallOutcome {
                tool_call,
                result,
                is_error,
                duration_ms: None,
            };
        }
        PreparedToolCall::Prepared { tool, args } => PreparedCall {
            tool_call,
            tool,
            args,
        },
    };
    let executed = execute_prepared_tool_call(
        &prepared,
        options.signal.as_ref(),
        &options
            .on_update
            .clone()
            .unwrap_or_else(|| Arc::new(|_| Box::pin(async {}))),
    )
    .await;
    finalize_executed_tool_call(
        &options.context,
        &options.assistant_message,
        &prepared,
        executed,
        &options,
        options.signal.as_ref(),
    )
    .await
}

async fn execute_prepared_tool_call(
    prepared: &PreparedCall,
    signal: Option<&AbortSignal>,
    on_update: &ToolUpdateSink,
) -> ExecutedToolCallOutcome {
    let update_events: Arc<Mutex<Vec<BoxFuture<'static, ()>>>> = Arc::new(Mutex::new(Vec::new()));
    let accepting_updates = Arc::new(AtomicBool::new(true));
    let started_at = std::time::Instant::now();
    let elapsed = || u64::try_from(started_at.elapsed().as_millis()).unwrap_or(0);

    let update_callback: AgentToolUpdateCallback = {
        let update_events = update_events.clone();
        let accepting_updates = accepting_updates.clone();
        let on_update = on_update.clone();
        Arc::new(move |partial_result| {
            if !accepting_updates.load(Ordering::SeqCst) {
                return;
            }
            let future = on_update(partial_result);
            update_events.lock().push(future);
        })
    };

    let outcome = (prepared.tool.execute)(
        prepared.tool_call.id.clone(),
        prepared.args.as_object().cloned().unwrap_or_default(),
        signal.cloned(),
        Some(update_callback),
    )
    .await;
    // Upstream captures `durationMs` as soon as `execute()` settles, before
    // the queued update events are drained.
    let duration_ms = elapsed();
    accepting_updates.store(false, Ordering::SeqCst);
    let pending_updates: Vec<BoxFuture<'static, ()>> = std::mem::take(&mut *update_events.lock());
    for update in pending_updates {
        update.await;
    }
    match outcome {
        Ok(result) => ExecutedToolCallOutcome {
            is_error: result.is_error == Some(true),
            result,
            duration_ms,
        },
        Err(message) => ExecutedToolCallOutcome {
            result: create_error_tool_result(message),
            is_error: true,
            duration_ms,
        },
    }
}

struct ExecutedToolCallOutcome {
    result: AgentToolResult,
    is_error: bool,
    duration_ms: u64,
}

async fn finalize_executed_tool_call(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    prepared: &PreparedCall,
    executed: ExecutedToolCallOutcome,
    hooks: impl Into<ToolCallHooks<'_>>,
    signal: Option<&AbortSignal>,
) -> AgentToolCallOutcome {
    let hooks = hooks.into();
    let mut result = executed.result;
    let mut is_error = executed.is_error;

    if let Some(hook) = hooks.after_tool_call {
        let after = hook(
            after_tool_call_context(
                assistant_message.clone(),
                prepared.tool_call.clone(),
                prepared.args.clone(),
                result.clone(),
                is_error,
                current_context.clone(),
            ),
            signal.cloned(),
        )
        .await;
        match after {
            Ok(after_result) => {
                if let Some(after_result) = after_result {
                    // Structured content not replaced along with the content may no
                    // longer match it.
                    let structured_content = after_result.structured_content.or_else(|| {
                        if after_result.content.is_some() {
                            None
                        } else {
                            result.structured_content.clone()
                        }
                    });
                    if let Some(content) = after_result.content {
                        result.content = content;
                    }
                    if let Some(details) = after_result.details {
                        result.details = details;
                    }
                    if let Some(usage) = after_result.usage {
                        result.usage = Some(usage);
                    }
                    if let Some(terminate) = after_result.terminate {
                        result.terminate = Some(terminate);
                    }
                    // `undefined` deletes the field upstream; `None` here.
                    result.structured_content = structured_content;
                    if let Some(after_is_error) = after_result.is_error {
                        is_error = after_is_error;
                    }
                }
            }
            Err(message) => {
                result = create_error_tool_result(message);
                is_error = true;
            }
        }
    }

    AgentToolCallOutcome {
        tool_call: prepared.tool_call.clone(),
        result,
        is_error,
        duration_ms: Some(executed.duration_ms),
    }
}

fn create_error_tool_result(message: impl Into<String>) -> AgentToolResult {
    AgentToolResult::new(
        vec![UserContent::Text(pi_ai::types::TextContent::new(
            message.into(),
        ))],
        Value::Object(serde_json::Map::new()),
    )
}

async fn emit_tool_execution_end(finalized: &AgentToolCallOutcome, emit: &AgentEventSink) {
    emit(AgentEvent::ToolExecutionEnd {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        result: finalized.result.clone(),
        is_error: finalized.is_error,
        duration_ms: finalized.duration_ms,
    })
    .await;
}

fn create_tool_result_message(finalized: &AgentToolCallOutcome) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        // Untyped tools (JS extensions) can return results without content;
        // normalize so null never enters session history or provider payloads.
        content: finalized.result.content.clone(),
        details: if finalized.result.details.is_null() {
            None
        } else {
            Some(finalized.result.details.clone())
        },
        usage: finalized.result.usage.clone(),
        nested_calls: None,
        is_error: finalized.is_error,
        duration_ms: finalized.duration_ms,
        timestamp: now_ms(),
    }
}

async fn emit_tool_result_message(tool_result_message: &ToolResultMessage, emit: &AgentEventSink) {
    emit(AgentEvent::MessageStart {
        message: AgentMessage::tool_result(tool_result_message.clone()),
    })
    .await;
    emit(AgentEvent::MessageEnd {
        message: AgentMessage::tool_result(tool_result_message.clone()),
    })
    .await;
}
