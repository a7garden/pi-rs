//! Port of `packages/agent/test/agent-loop.test.ts`.

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use pi_agent::{
    after_tool_call, agent_loop, agent_loop_continue, before_tool_call, finish_turn,
    get_follow_up_messages, get_steering_messages, prepare_next_turn, prepare_request,
    run_agent_loop, run_tool_call, transform_context, AfterToolCallResult, AgentContext,
    AgentEvent, AgentLoopTurnUpdate, AgentMessage, AgentTool, AgentToolResult,
    BeforeToolCallResult, RunToolCallOptions,
};
use pi_ai::types::{
    Message, StopReason, StreamErrorReason, SystemContent, SystemMessage, Usage, UserContent,
    UserContentField,
};
use serde_json::{json, Value};

use common::*;

// ==== helpers local to this file ====

fn agent_context(messages: Vec<AgentMessage>, tools: Vec<pi_agent::AgentTool>) -> AgentContext {
    AgentContext {
        messages,
        tools: Some(tools),
    }
}

fn empty_context() -> AgentContext {
    AgentContext {
        messages: Vec::new(),
        tools: Some(Vec::new()),
    }
}

fn system_message(content: impl Into<String>, timestamp: i64) -> SystemMessage {
    SystemMessage {
        content: SystemContent::Text(content.into()),
        sections: None,
        tools_added: None,
        tools_removed: None,
        timestamp,
    }
}

/// Stream fn whose responses are scripted per provider call.
fn scripted_stream_fn(
    responses: Arc<Mutex<Vec<(pi_ai::types::AssistantMessage, StopReason)>>>,
) -> pi_agent::StreamFn {
    let counter = call_counter();
    stream_fn(move |_model, _context, _options| {
        let index = counter.fetch_add(1, Ordering::SeqCst);
        let responses = responses.lock();
        let (message, reason) = if index < responses.len() {
            responses[index].clone()
        } else {
            responses.last().expect("scripted response").clone()
        };
        let mock = MockAssistantStream::new();
        mock.push_done(message, reason);
        mock.stream
    })
}

// ==== default stream function compatibility ====

#[tokio::test]
async fn uses_the_configured_default_when_a_legacy_caller_omits_stream_fn() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_hook = calls.clone();
    pi_agent::set_default_stream_fn(Some(stream_fn(move |_model, _context, _options| {
        calls_for_hook.fetch_add(1, Ordering::SeqCst);
        let mock = MockAssistantStream::new();
        mock.push_done(create_assistant_message_text("fallback"), StopReason::Stop);
        mock.stream
    })));

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("Hello"))],
        empty_context(),
        loop_config(),
        None,
        None,
    );

    let (_events, _messages) = run_to_completion(&stream).await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    pi_agent::set_default_stream_fn(None);
}

// ==== agentLoop with AgentMessage ====

#[tokio::test]
async fn emits_events_with_agent_message_types() {
    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("Hello"))],
        empty_context(),
        loop_config(),
        None,
        Some(stream_fn(move |_model, _context, _options| {
            let mock = MockAssistantStream::new();
            mock.push_done(create_assistant_message_text("Hi there!"), StopReason::Stop);
            mock.stream
        })),
    );

    let (events, messages) = run_to_completion(&stream).await;

    // Should have user message and assistant message
    assert_eq!(messages.len(), 2);
    assert_eq!(message_roles(&messages), vec!["user", "assistant"]);

    // Verify event sequence
    let types = event_types(&events);
    for expected in [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ] {
        assert!(types.contains(&expected), "missing {expected}");
    }
}

#[tokio::test]
async fn builds_provider_context_exclusively_from_transcript_messages() {
    let initial_system = AgentMessage::system(system_message("Transcript prompt", 1));
    let expected_first_message = Message::System(system_message("Transcript prompt", 1));
    let _ = &initial_system;
    let stream_fn = stream_fn(move |_model, provider_context, _options| {
        // The provider receives a transcript: no top-level prompt or tool fields.
        assert_eq!(
            provider_context.messages().first(),
            Some(&expected_first_message)
        );
        let mock = MockAssistantStream::new();
        mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
        mock.stream
    });

    let stream = agent_loop(
        vec![
            initial_system,
            AgentMessage::user(create_user_message("Hello")),
        ],
        empty_context(),
        loop_config(),
        None,
        Some(stream_fn),
    );

    run_to_completion(&stream).await;
}

#[tokio::test]
async fn handles_custom_message_types_via_convert_to_llm() {
    let notification = AgentMessage::Custom(json!({
        "role": "notification",
        "text": "This is a notification",
        "timestamp": pi_agent::now_ms(),
    }));

    let converted: Arc<Mutex<Vec<Message>>> = Arc::new(Mutex::new(Vec::new()));
    let converted_for_hook = converted.clone();
    let mut config = loop_config();
    config.convert_to_llm = Arc::new(move |messages| {
        let converted_for_hook = converted_for_hook.clone();
        Box::pin(async move {
            // Filter out notifications, convert rest
            let filtered: Vec<Message> = messages
                .iter()
                .filter(|message| message.role() != "notification")
                .filter(|message| matches!(message.role(), "user" | "assistant" | "toolResult"))
                .filter_map(|message| message.as_llm().cloned())
                .collect();
            *converted_for_hook.lock() = filtered.clone();
            filtered
        })
    });

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("Hello"))],
        agent_context(vec![notification], vec![]),
        config,
        None,
        Some(stream_fn(move |_model, _context, _options| {
            let mock = MockAssistantStream::new();
            mock.push_done(create_assistant_message_text("Response"), StopReason::Stop);
            mock.stream
        })),
    );

    run_to_completion(&stream).await;

    // The notification should have been filtered out in convertToLlm
    let converted = converted.lock();
    assert_eq!(converted.len(), 1); // Only user message
    assert_eq!(converted[0].role(), "user");
}

#[tokio::test]
async fn applies_transform_context_before_convert_to_llm() {
    let transformed_len: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let converted_len: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));

    let mut config = loop_config();
    {
        let transformed_len = transformed_len.clone();
        config.transform_context = Some(transform_context(move |messages, _signal| {
            let transformed_len = transformed_len.clone();
            Box::pin(async move {
                // Keep only last 2 messages (prune old ones)
                let pruned = messages[messages.len() - 2..].to_vec();
                *transformed_len.lock() = pruned.len();
                pruned
            })
        }));
    }
    {
        let converted_len = converted_len.clone();
        config.convert_to_llm = Arc::new(move |messages| {
            let converted_len = converted_len.clone();
            Box::pin(async move {
                let filtered: Vec<Message> = messages
                    .iter()
                    .filter(|message| matches!(message.role(), "user" | "assistant" | "toolResult"))
                    .filter_map(|message| message.as_llm().cloned())
                    .collect();
                *converted_len.lock() = filtered.len();
                filtered
            })
        });
    }

    let context = agent_context(
        vec![
            AgentMessage::user(create_user_message("old message 1")),
            AgentMessage::assistant(create_assistant_message_text("old response 1")),
            AgentMessage::user(create_user_message("old message 2")),
            AgentMessage::assistant(create_assistant_message_text("old response 2")),
        ],
        vec![],
    );

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("new message"))],
        context,
        config,
        None,
        Some(stream_fn(move |_model, _context, _options| {
            let mock = MockAssistantStream::new();
            mock.push_done(create_assistant_message_text("Response"), StopReason::Stop);
            mock.stream
        })),
    );

    run_to_completion(&stream).await;

    // transformContext should have been called first, keeping only last 2
    assert_eq!(*transformed_len.lock(), 2);
    // Then convertToLlm receives the pruned messages
    assert_eq!(*converted_len.lock(), 2);
}

#[tokio::test]
async fn handles_tool_calls_and_results() {
    let executed: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let tool_usage = Usage {
        input: 1,
        output: 2,
        cache_read: 3,
        cache_write: 4,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 10,
        cost: pi_ai::types::UsageCost {
            input: 0.1,
            output: 0.2,
            cache_read: 0.3,
            cache_write: 0.4,
            total: 1.0,
        },
    };
    let patched_tool_usage = Usage {
        input: 5,
        output: 6,
        cache_read: 7,
        cache_write: 8,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 26,
        cost: pi_ai::types::UsageCost {
            input: 0.5,
            output: 0.6,
            cache_read: 0.7,
            cache_write: 0.8,
            total: 2.6,
        },
    };
    let observed_tool_usage: Arc<Mutex<Option<Usage>>> = Arc::new(Mutex::new(None));

    let executed_for_tool = executed.clone();
    let mut tool = echo_tool(executed.clone());
    tool.execute = {
        let tool_usage = tool_usage.clone();
        tool_execute(move |_id, params| {
            executed_for_tool
                .lock()
                .push(params.get("value").cloned().unwrap_or(Value::Null));
            Ok(AgentToolResult {
                content: vec![UserContent::Text(pi_ai::types::TextContent::new(format!(
                    "echoed: {}",
                    params
                        .get("value")
                        .and_then(|value| value.as_str())
                        .unwrap_or("")
                )))],
                details: json!({ "value": params.get("value").cloned() }),
                structured_content: None,
                usage: Some(tool_usage.clone()),
                is_error: None,
                terminate: None,
            })
        })
    };

    let mut config = loop_config();
    {
        let observed = observed_tool_usage.clone();
        let patched = patched_tool_usage.clone();
        config.after_tool_call = Some(after_tool_call(move |context, _signal| {
            let observed = observed.clone();
            let patched = patched.clone();
            Box::pin(async move {
                *observed.lock() = context.result.usage.clone();
                Ok(Some(AfterToolCallResult {
                    usage: Some(patched.clone()),
                    ..Default::default()
                }))
            })
        }));
    }

    let responses = Arc::new(Mutex::new(vec![
        (
            tool_use_message(vec![create_tool_call(
                "tool-1",
                "echo",
                json!({ "value": "hello" }),
            )]),
            StopReason::ToolUse,
        ),
        (create_assistant_message_text("done"), StopReason::Stop),
    ]));

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("echo something"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(scripted_stream_fn(responses)),
    );

    let (events, messages) = run_to_completion(&stream).await;

    // Tool should have been executed
    assert_eq!(*executed.lock(), vec![json!("hello")]);

    // Should have tool execution events
    let tool_end = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionEnd { is_error, .. } => Some(*is_error),
        _ => None,
    });
    assert_eq!(tool_end, Some(false));
    assert_eq!(*observed_tool_usage.lock(), Some(tool_usage));
    let tool_result = messages
        .iter()
        .find(|message| message.role() == "toolResult");
    let usage = match tool_result.expect("tool result") {
        AgentMessage::Llm(Message::ToolResult(tool_result)) => tool_result.usage.clone(),
        _ => unreachable!(),
    };
    assert_eq!(usage, Some(patched_tool_usage));
}

#[tokio::test]
async fn does_not_execute_tool_calls_from_a_length_truncated_assistant_message() {
    let executed: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let tool = echo_tool(executed.clone());

    let responses = Arc::new(Mutex::new(vec![
        (
            tool_calls_message(
                vec![create_tool_call(
                    "tool-1",
                    "echo",
                    json!({ "value": "hel" }),
                )],
                StopReason::Length,
            ),
            StopReason::Length,
        ),
        (create_assistant_message_text("done"), StopReason::Stop),
    ]));

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("echo something"))],
        agent_context(vec![], vec![tool]),
        loop_config(),
        None,
        Some(scripted_stream_fn(responses)),
    );
    let (events, messages) = run_to_completion(&stream).await;

    // The tool must never execute with potentially truncated arguments.
    assert_eq!(*executed.lock(), Vec::<Value>::new());

    let tool_end = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionEnd {
            result, is_error, ..
        } => Some((result.clone(), *is_error)),
        _ => None,
    });
    let (result, is_error) = tool_end.expect("tool end event");
    assert!(is_error);
    let text = result
        .content
        .iter()
        .filter_map(|block| match block {
            UserContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .next()
        .unwrap_or_default();
    assert!(
        text.contains("output token limit"),
        "unexpected text: {text}"
    );

    // The loop continues so the model can re-issue the tool call.
    assert_eq!(
        message_roles(&messages).last().map(String::as_str),
        Some("assistant")
    );
}

// #10549
#[tokio::test]
async fn records_how_long_execute_took_on_the_tool_result_excluding_hooks() {
    fn sleep_echo_tool() -> pi_agent::AgentTool {
        pi_agent::AgentTool::new(
            "echo",
            "Echo tool",
            object_schema(&[("value", "string")]),
            "Echo",
            tool_execute(|_id, params| {
                std::thread::sleep(std::time::Duration::from_millis(30));
                let value = params.get("value").cloned().unwrap_or(Value::Null);
                Ok(AgentToolResult::new(
                    vec![UserContent::Text(pi_ai::types::TextContent::new(
                        value.as_str().unwrap_or_default().to_string(),
                    ))],
                    json!({ "value": value }),
                ))
            }),
        )
    }

    let tool = sleep_echo_tool();

    let mut config = loop_config();
    config.before_tool_call = Some(before_tool_call(|context, _signal| {
        Box::pin(async move {
            // Hook time must not count into `durationMs`.
            std::thread::sleep(std::time::Duration::from_millis(100));
            if context.tool_call.id == "blocked" {
                Ok(Some(BeforeToolCallResult {
                    block: Some(true),
                    reason: Some("no".to_string()),
                    ..Default::default()
                }))
            } else {
                Ok(None)
            }
        })
    }));

    let responses = Arc::new(Mutex::new(vec![
        (
            tool_use_message(vec![
                create_tool_call("ran", "echo", json!({ "value": "a" })),
                create_tool_call("blocked", "echo", json!({ "value": "b" })),
            ]),
            StopReason::ToolUse,
        ),
        (create_assistant_message_text("done"), StopReason::Stop),
    ]));

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("go"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(scripted_stream_fn(responses)),
    );
    let (_events, messages) = run_to_completion(&stream).await;

    let results: Vec<pi_ai::types::ToolResultMessage> = messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Llm(Message::ToolResult(tool_result)) => Some(tool_result.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 2);
    let ran = &results[0];
    let blocked = &results[1];

    let duration = ran.duration_ms.expect("executed tool carries durationMs");
    assert!(
        duration >= 25,
        "durationMs {duration} should cover the 30ms execute"
    );
    assert!(
        duration < 100,
        "durationMs {duration} must exclude the 100ms hook"
    );

    assert!(blocked.is_error);
    assert_eq!(blocked.duration_ms, None, "blocked calls did not run");
}

#[tokio::test]
async fn executes_mutated_before_tool_call_args_without_revalidation() {
    let executed: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let tool = echo_tool(executed.clone());

    let mut config = loop_config();
    config.before_tool_call = Some(before_tool_call(|_context, _signal| {
        Box::pin(async move {
            Ok(Some(BeforeToolCallResult {
                args: Some(json!({ "value": 123 })),
                ..Default::default()
            }))
        })
    }));

    let responses = Arc::new(Mutex::new(vec![
        (
            tool_use_message(vec![create_tool_call(
                "tool-1",
                "echo",
                json!({ "value": "hello" }),
            )]),
            StopReason::ToolUse,
        ),
        (create_assistant_message_text("done"), StopReason::Stop),
    ]));

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("echo something"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(scripted_stream_fn(responses)),
    );
    run_to_completion(&stream).await;

    assert_eq!(*executed.lock(), vec![json!(123)]);
}

#[tokio::test]
async fn prepares_tool_arguments_for_validation() {
    let executed: Arc<Mutex<Vec<Vec<Value>>>> = Arc::new(Mutex::new(Vec::new()));
    let executed_for_tool = executed.clone();
    // toolSchema = Type.Object({ edits: Type.Array(Type.Object({ oldText, newText })) })
    let edit_schema = json!({
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": { "oldText": { "type": "string" }, "newText": { "type": "string" } },
                    "required": ["oldText", "newText"]
                }
            }
        },
        "required": ["edits"]
    });
    let mut tool = AgentTool::new(
        "edit",
        "Edit tool",
        edit_schema,
        "Edit",
        tool_execute(move |_id, params| {
            let executed = executed_for_tool.clone();
            let edits = params.get("edits").cloned().unwrap_or(Value::Null);
            executed
                .lock()
                .push(edits.as_array().cloned().unwrap_or_default());
            let count = edits.as_array().map(|edits| edits.len()).unwrap_or(0);
            Ok(AgentToolResult::new(
                vec![UserContent::Text(pi_ai::types::TextContent::new(format!(
                    "edited {count}"
                )))],
                json!({ "count": count }),
            ))
        }),
    );
    tool.prepare_arguments = Some(Arc::new(|args: Value| {
        let Some(input) = args.as_object() else {
            return args;
        };
        let (Some(old), Some(new)) = (
            input.get("oldText").and_then(|value| value.as_str()),
            input.get("newText").and_then(|value| value.as_str()),
        ) else {
            return args;
        };
        let mut edits: Vec<Value> = input
            .get("edits")
            .and_then(|edits| edits.as_array())
            .cloned()
            .unwrap_or_default();
        edits.push(json!({ "oldText": old, "newText": new }));
        json!({ "edits": edits })
    }));

    let responses = Arc::new(Mutex::new(vec![
        (
            tool_use_message(vec![create_tool_call(
                "tool-1",
                "edit",
                json!({ "oldText": "before", "newText": "after" }),
            )]),
            StopReason::ToolUse,
        ),
        (create_assistant_message_text("done"), StopReason::Stop),
    ]));

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("edit something"))],
        agent_context(vec![], vec![tool]),
        loop_config(),
        None,
        Some(scripted_stream_fn(responses)),
    );
    run_to_completion(&stream).await;

    assert_eq!(
        *executed.lock(),
        vec![vec![json!({ "oldText": "before", "newText": "after" })]]
    );
}

/// Shared harness for the parallel/sequential execution-order tests: two
/// tool calls where the first blocks until released.
struct OrderingHarness {
    executed: Arc<Mutex<Vec<String>>>,
    first_resolved: Arc<AtomicBool>,
    parallel_observed: Arc<AtomicBool>,
    release: Arc<tokio::sync::Notify>,
}

impl OrderingHarness {
    fn new() -> OrderingHarness {
        OrderingHarness {
            executed: Arc::new(Mutex::new(Vec::new())),
            first_resolved: Arc::new(AtomicBool::new(false)),
            parallel_observed: Arc::new(AtomicBool::new(false)),
            release: Arc::new(tokio::sync::Notify::new()),
        }
    }

    fn tool(
        &self,
        name: &str,
        execution_mode: Option<pi_agent::ToolExecutionMode>,
    ) -> pi_agent::AgentTool {
        let tool_name = name.to_string();
        let label = name.to_string();
        let executed = self.executed.clone();
        let first_resolved = self.first_resolved.clone();
        let parallel_observed = self.parallel_observed.clone();
        let release = self.release.clone();
        let mut tool = AgentTool::new(
            name,
            format!("{name} tool"),
            object_schema(&[("value", "string")]),
            name,
            tool_execute_async(move |_id, params, _signal| {
                let _ = (&tool_name, &label);
                let executed = executed.clone();
                let first_resolved = first_resolved.clone();
                let parallel_observed = parallel_observed.clone();
                let release = release.clone();
                Box::pin(async move {
                    let value = params
                        .get("value")
                        .and_then(|value| value.as_str())
                        .unwrap_or("")
                        .to_string();
                    if value == "first" {
                        release.notified().await;
                        first_resolved.store(true, Ordering::SeqCst);
                    }
                    if value == "second" && !first_resolved.load(Ordering::SeqCst) {
                        parallel_observed.store(true, Ordering::SeqCst);
                    }
                    executed.lock().push(format!("{value}"));
                    Ok(AgentToolResult::new(
                        vec![UserContent::Text(pi_ai::types::TextContent::new(format!(
                            "echoed: {value}"
                        )))],
                        json!({ "value": value }),
                    ))
                })
            }),
        );
        tool.execution_mode = execution_mode;
        tool
    }

    /// Two tool calls ("first" then "second"); the release fires 20ms after
    /// the first provider response.
    fn stream_fn(&self) -> pi_agent::StreamFn {
        let counter = call_counter();
        let release = self.release.clone();
        stream_fn(move |_model, _context, _options| {
            let index = counter.fetch_add(1, Ordering::SeqCst);
            let mock = MockAssistantStream::new();
            if index == 0 {
                mock.push_done(
                    tool_use_message(vec![
                        create_tool_call("tool-1", "echo", json!({ "value": "first" })),
                        create_tool_call("tool-2", "echo", json!({ "value": "second" })),
                    ]),
                    StopReason::ToolUse,
                );
                let release = release.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    release.notify_one();
                });
            } else {
                mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
            }
            mock.stream
        })
    }
}

#[tokio::test]
async fn emits_tool_execution_end_in_completion_order_but_persists_tool_results_in_source_order() {
    let harness = OrderingHarness::new();
    let tool = harness.tool("echo", None);
    let mut config = loop_config();
    config.tool_execution = Some(pi_agent::ToolExecutionMode::Parallel);

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("echo both"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(harness.stream_fn()),
    );

    let (events, _messages) = run_to_completion(&stream).await;

    let tool_execution_end_ids = tool_execution_end_ids(&events);
    let tool_result_ids: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd { message } => match message {
                AgentMessage::Llm(Message::ToolResult(tool_result)) => {
                    Some(tool_result.tool_call_id.clone())
                }
                _ => None,
            },
            _ => None,
        })
        .collect();
    let turn_tool_result_ids: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TurnEnd { tool_results, .. } => Some(
                tool_results
                    .iter()
                    .map(|result| result.tool_call_id.clone())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .flatten()
        .collect();

    assert!(harness.parallel_observed.load(Ordering::SeqCst));
    assert_eq!(tool_execution_end_ids, vec!["tool-2", "tool-1"]);
    assert_eq!(tool_result_ids, vec!["tool-1", "tool-2"]);
    assert_eq!(turn_tool_result_ids, vec!["tool-1", "tool-2"]);
}

#[tokio::test]
async fn injects_queued_messages_after_all_tool_calls_complete() {
    let executed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let executed_for_tool = executed.clone();
    let mut tool = echo_tool(Arc::new(Mutex::new(Vec::new())));
    tool.execute = tool_execute(move |_id, params| {
        let value = params
            .get("value")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        executed_for_tool.lock().push(value);
        Ok(AgentToolResult::new(
            vec![UserContent::Text(pi_ai::types::TextContent::new(format!(
                "ok:{}",
                params
                    .get("value")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
            )))],
            json!({ "value": params.get("value").cloned().unwrap_or(Value::Null) }),
        ))
    });

    let queued_delivered = Arc::new(AtomicBool::new(false));
    let mut config = loop_config();
    config.tool_execution = Some(pi_agent::ToolExecutionMode::Sequential);
    {
        let executed = executed.clone();
        let queued_delivered = queued_delivered.clone();
        config.get_steering_messages = Some(get_steering_messages(move || {
            let executed = executed.clone();
            let queued_delivered = queued_delivered.clone();
            Box::pin(async move {
                // Return steering message after tool execution has started.
                if executed.lock().len() >= 1 && !queued_delivered.swap(true, Ordering::SeqCst) {
                    return vec![AgentMessage::user(create_user_message("interrupt"))];
                }
                Vec::new()
            })
        }));
    }

    let saw_interrupt: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    let saw_interrupt_for_stream = saw_interrupt.clone();
    let counter = call_counter();
    let stream_fn_impl = stream_fn(move |_model, ctx, _options| {
        // Check if interrupt message is in context on second call
        if calls(&counter) == 1 && user_texts(&ctx).iter().any(|text| text == "interrupt") {
            saw_interrupt_for_stream.store(true, Ordering::SeqCst);
        }
        let mock = MockAssistantStream::new();
        if calls(&counter) == 0 {
            mock.push_done(
                tool_use_message(vec![
                    create_tool_call("tool-1", "echo", json!({ "value": "first" })),
                    create_tool_call("tool-2", "echo", json!({ "value": "second" })),
                ]),
                StopReason::ToolUse,
            );
        } else {
            mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
        }
        counter.fetch_add(1, Ordering::SeqCst);
        mock.stream
    });

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("start"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(stream_fn_impl),
    );

    let (events, _messages) = run_to_completion(&stream).await;

    // Both tools should execute before steering is injected
    assert_eq!(
        *executed.lock(),
        vec!["first".to_string(), "second".to_string()]
    );

    let tool_ends: Vec<bool> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { is_error, .. } => Some(*is_error),
            _ => None,
        })
        .collect();
    assert_eq!(tool_ends, vec![false, false]);

    // Queued message should appear in events after both tool result messages
    let event_sequence = message_start_markers(&events);
    let interrupt_index = event_sequence
        .iter()
        .position(|marker| marker == "interrupt")
        .expect("interrupt");
    let tool_1_index = event_sequence
        .iter()
        .position(|marker| marker == "tool:tool-1")
        .expect("tool-1");
    let tool_2_index = event_sequence
        .iter()
        .position(|marker| marker == "tool:tool-2")
        .expect("tool-2");
    assert!(tool_1_index < interrupt_index);
    assert!(tool_2_index < interrupt_index);

    // Interrupt message should be in context when second LLM call is made
    assert!(saw_interrupt.load(Ordering::SeqCst));
}

#[tokio::test]
async fn forces_sequential_execution_when_a_tool_has_sequential_mode_with_default_parallel_config()
{
    let harness = OrderingHarness::new();
    let tool = harness.tool("slow", Some(pi_agent::ToolExecutionMode::Sequential));

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("run both"))],
        agent_context(vec![], vec![tool]),
        // config is parallel (default), but tool forces sequential
        loop_config(),
        None,
        Some(harness.stream_fn()),
    );

    let (events, _messages) = run_to_completion(&stream).await;

    // With sequential execution, second tool should NOT start before first finishes
    assert!(!harness.parallel_observed.load(Ordering::SeqCst));

    let tool_result_ids: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd { message } => match message {
                AgentMessage::Llm(Message::ToolResult(tool_result)) => {
                    Some(tool_result.tool_call_id.clone())
                }
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(tool_result_ids, vec!["tool-1", "tool-2"]);
}

#[tokio::test]
async fn forces_sequential_execution_when_one_of_multiple_tools_has_sequential_mode() {
    let execution_order: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let release = Arc::new(tokio::sync::Notify::new());

    let mut slow =
        OrderingHarness::new().tool("slow", Some(pi_agent::ToolExecutionMode::Sequential));
    slow.execute = {
        let execution_order = execution_order.clone();
        let release = release.clone();
        tool_execute_async(move |_id, params, _signal| {
            let execution_order = execution_order.clone();
            let release = release.clone();
            Box::pin(async move {
                let value = params
                    .get("value")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_string();
                execution_order.lock().push(format!("slow:{value}"));
                if value == "a" {
                    release.notified().await;
                }
                Ok(AgentToolResult::new(
                    vec![UserContent::Text(pi_ai::types::TextContent::new(format!(
                        "slow: {value}"
                    )))],
                    json!({ "value": value }),
                ))
            })
        })
    };
    let fast = {
        let execution_order = execution_order.clone();
        let mut tool = AgentTool::new(
            "fast",
            "Fast tool",
            object_schema(&[("value", "string")]),
            "Fast",
            tool_execute(move |_id, params| {
                let value = params
                    .get("value")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_string();
                execution_order.lock().push(format!("fast:{value}"));
                Ok(AgentToolResult::new(
                    vec![UserContent::Text(pi_ai::types::TextContent::new(format!(
                        "fast: {value}"
                    )))],
                    json!({ "value": value }),
                ))
            }),
        );
        // no executionMode = defaults to parallel
        tool.execution_mode = None;
        tool
    };

    let counter = call_counter();
    let release_for_stream = release.clone();
    let stream_fn_impl = stream_fn(move |_model, _context, _options| {
        let mock = MockAssistantStream::new();
        if calls(&counter) == 0 {
            mock.push_done(
                tool_use_message(vec![
                    create_tool_call("tool-1", "slow", json!({ "value": "a" })),
                    create_tool_call("tool-2", "fast", json!({ "value": "b" })),
                ]),
                StopReason::ToolUse,
            );
            let release = release_for_stream.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                release.notify_one();
            });
        } else {
            mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
        }
        counter.fetch_add(1, Ordering::SeqCst);
        mock.stream
    });

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("run both"))],
        agent_context(vec![], vec![slow, fast]),
        // parallel by default, but slowTool forces sequential
        loop_config(),
        None,
        Some(stream_fn_impl),
    );

    run_to_completion(&stream).await;

    // Fast tool should NOT run before slow tool finishes
    assert_eq!(execution_order.lock()[0], "slow:a");
    assert!(execution_order.lock().iter().any(|entry| entry == "fast:b"));
}

#[tokio::test]
async fn allows_parallel_execution_when_all_tools_have_parallel_mode() {
    let harness = OrderingHarness::new();
    // Overwrite the released notify with an already-set one so "first"
    // completes only after the explicit release below.
    let tool = harness.tool("echo", Some(pi_agent::ToolExecutionMode::Parallel));

    let counter = call_counter();
    let release = harness.release.clone();
    let stream_fn_impl = stream_fn(move |_model, _context, _options| {
        let mock = MockAssistantStream::new();
        if calls(&counter) == 0 {
            mock.push_done(
                tool_use_message(vec![
                    create_tool_call("tool-1", "echo", json!({ "value": "first" })),
                    create_tool_call("tool-2", "echo", json!({ "value": "second" })),
                ]),
                StopReason::ToolUse,
            );
            let release = release.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                release.notify_one();
            });
        } else {
            mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
        }
        counter.fetch_add(1, Ordering::SeqCst);
        mock.stream
    });

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("echo both"))],
        agent_context(vec![], vec![tool]),
        loop_config(),
        None,
        Some(stream_fn_impl),
    );

    run_to_completion(&stream).await;

    // With executionMode=parallel, second tool should start before first finishes
    assert!(harness.parallel_observed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn runs_finish_turn_after_tool_result_messages_and_before_turn_end() {
    let mut tool = OrderingHarness::new().tool("echo", None);
    tool.execute = tool_execute(|_id, params| {
        Ok(AgentToolResult {
            content: vec![UserContent::Text(pi_ai::types::TextContent::new(
                params
                    .get("value")
                    .and_then(|value| value.as_str())
                    .unwrap_or(""),
            ))],
            details: json!({ "value": params.get("value").cloned().unwrap_or(Value::Null) }),
            structured_content: None,
            usage: None,
            is_error: None,
            terminate: Some(true),
        })
    });

    let ordering: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let mut config = loop_config();
    {
        let ordering = ordering.clone();
        config.finish_turn = Some(finish_turn(move |turn, _signal| {
            let ordering = ordering.clone();
            Box::pin(async move {
                ordering.lock().push("finishTurn".into());
                assert_eq!(turn.tool_results.len(), 1);
                assert_eq!(
                    turn.context.messages.last().map(|message| message.role()),
                    Some("toolResult")
                );
                None
            })
        }));
    }

    let ordering_for_emit = ordering.clone();
    run_agent_loop(
        vec![AgentMessage::user(create_user_message("echo"))],
        agent_context(vec![], vec![tool]),
        config,
        Arc::new(move |event| {
            let ordering = ordering_for_emit.clone();
            Box::pin(async move {
                match &event {
                    AgentEvent::MessageEnd { message } => {
                        ordering
                            .lock()
                            .push(format!("message_end:{}", message.role()));
                    }
                    AgentEvent::TurnEnd { .. } => ordering.lock().push("turn_end".into()),
                    _ => {}
                }
            })
        }),
        None,
        Some(stream_fn(move |_model, _context, _options| {
            let mock = MockAssistantStream::new();
            mock.push_done(
                tool_use_message(vec![create_tool_call(
                    "tool-1",
                    "echo",
                    json!({ "value": "hello" }),
                )]),
                StopReason::ToolUse,
            );
            mock.stream
        })),
    )
    .await
    .expect("run completes");

    let ordering = ordering.lock();
    let tail = &ordering[ordering.len() - 3..];
    assert_eq!(
        tail,
        &[
            "message_end:toolResult".to_string(),
            "finishTurn".to_string(),
            "turn_end".to_string()
        ]
    );
}

#[tokio::test]
async fn runs_finish_turn_for_error_and_aborted_assistants_before_turn_end() {
    for reason in [StopReason::Error, StopReason::Aborted] {
        let ordering: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let steering_polls = Arc::new(AtomicUsize::new(0));
        let follow_up_polls = Arc::new(AtomicUsize::new(0));

        let mut config = loop_config();
        {
            let ordering = ordering.clone();
            config.finish_turn = Some(finish_turn(move |turn, _signal| {
                let ordering = ordering.clone();
                Box::pin(async move {
                    assert_eq!(turn.message.stop_reason, reason);
                    ordering.lock().push("finishTurn".into());
                    Some(pi_agent::AgentTurnDecision::Continue)
                })
            }));
            let steering_polls = steering_polls.clone();
            config.get_steering_messages = Some(get_steering_messages(move || {
                steering_polls.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Vec::new() })
            }));
            let follow_up_polls = follow_up_polls.clone();
            config.get_follow_up_messages = Some(get_follow_up_messages(move || {
                follow_up_polls.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { vec![AgentMessage::user(create_user_message("queued"))] })
            }));
        }

        let ordering_for_emit = ordering.clone();
        let provider_calls_for_stream = provider_calls.clone();
        let stream_fn_impl = stream_fn(move |_model, _context, _options| {
            provider_calls_for_stream.fetch_add(1, Ordering::SeqCst);
            let mock = MockAssistantStream::new();
            let mut error_message = create_assistant_message(Vec::new(), reason);
            error_message.error_message = Some(
                match reason {
                    StopReason::Error => "error",
                    StopReason::Aborted => "aborted",
                    _ => unreachable!(),
                }
                .to_string(),
            );
            mock.push_error(
                error_message,
                match reason {
                    StopReason::Error => StreamErrorReason::Error,
                    StopReason::Aborted => StreamErrorReason::Aborted,
                    _ => unreachable!(),
                },
            );
            mock.stream
        });

        run_agent_loop(
            vec![AgentMessage::user(create_user_message("run"))],
            empty_context(),
            config,
            Arc::new(move |event| {
                let ordering = ordering_for_emit.clone();
                Box::pin(async move {
                    if matches!(event, AgentEvent::TurnEnd { .. }) {
                        ordering.lock().push("turn_end".into());
                    }
                })
            }),
            None,
            Some(stream_fn_impl),
        )
        .await
        .expect("run completes");

        assert_eq!(
            *ordering.lock(),
            vec!["finishTurn".to_string(), "turn_end".to_string()]
        );
        assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
        assert_eq!(steering_polls.load(Ordering::SeqCst), 1);
        assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn action_end_skips_queue_polling_and_next_turn_preparation() {
    let tool = create_tool("noop");
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let steering_polls = Arc::new(AtomicUsize::new(0));
    let follow_up_polls = Arc::new(AtomicUsize::new(0));
    let prepare_next_turn_calls = Arc::new(AtomicUsize::new(0));

    let mut config = loop_config();
    config.finish_turn = Some(finish_turn(|_context, _signal| {
        Box::pin(async { Some(pi_agent::AgentTurnDecision::End) })
    }));
    {
        let prepare_next_turn_calls = prepare_next_turn_calls.clone();
        config.prepare_next_turn = Some(prepare_next_turn(move |_context| {
            prepare_next_turn_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { None })
        }));
        let steering_polls = steering_polls.clone();
        config.get_steering_messages = Some(get_steering_messages(move || {
            steering_polls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Vec::new() })
        }));
        let follow_up_polls = follow_up_polls.clone();
        config.get_follow_up_messages = Some(get_follow_up_messages(move || {
            follow_up_polls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { vec![AgentMessage::user(create_user_message("queued"))] })
        }));
    }

    let provider_calls_for_stream = provider_calls.clone();
    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("run"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(stream_fn(move |_model, _context, _options| {
            provider_calls_for_stream.fetch_add(1, Ordering::SeqCst);
            let mock = MockAssistantStream::new();
            mock.push_done(
                tool_use_message(vec![create_tool_call("tool-1", "noop", json!({}))]),
                StopReason::ToolUse,
            );
            mock.stream
        })),
    );
    run_to_completion(&stream).await;

    assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
    assert_eq!(steering_polls.load(Ordering::SeqCst), 1);
    assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
    assert_eq!(prepare_next_turn_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn makes_exactly_one_context_only_request_when_no_natural_request_satisfies_continuation() {
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let finish_calls = Arc::new(AtomicUsize::new(0));

    let mut config = loop_config();
    {
        let finish_calls = finish_calls.clone();
        config.finish_turn = Some(finish_turn(move |_context, _signal| {
            let finish_calls = finish_calls.clone();
            Box::pin(async move {
                let calls = finish_calls.fetch_add(1, Ordering::SeqCst) + 1;
                (calls == 1).then_some(pi_agent::AgentTurnDecision::Continue)
            })
        }));
    }

    let provider_calls_for_stream = provider_calls.clone();
    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("run"))],
        empty_context(),
        config,
        None,
        Some(stream_fn(move |_model, _context, _options| {
            let response_index = provider_calls_for_stream.fetch_add(1, Ordering::SeqCst) + 1;
            let mock = MockAssistantStream::new();
            mock.push_done(
                create_assistant_message_text(format!("response {response_index}")),
                StopReason::Stop,
            );
            mock.stream
        })),
    );
    run_to_completion(&stream).await;

    assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
    assert_eq!(finish_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn lets_a_natural_tool_result_request_satisfy_continuation() {
    let tool = create_tool("noop");
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let finish_calls = Arc::new(AtomicUsize::new(0));

    let mut config = loop_config();
    {
        let finish_calls = finish_calls.clone();
        config.finish_turn = Some(finish_turn(move |_context, _signal| {
            let finish_calls = finish_calls.clone();
            Box::pin(async move {
                let calls = finish_calls.fetch_add(1, Ordering::SeqCst) + 1;
                (calls == 1).then_some(pi_agent::AgentTurnDecision::Continue)
            })
        }));
    }

    let provider_calls_for_stream = provider_calls.clone();
    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("run"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(stream_fn(move |_model, _context, _options| {
            let response_index = provider_calls_for_stream.fetch_add(1, Ordering::SeqCst) + 1;
            let mock = MockAssistantStream::new();
            if response_index == 1 {
                mock.push_done(
                    tool_use_message(vec![create_tool_call("tool-1", "noop", json!({}))]),
                    StopReason::ToolUse,
                );
            } else {
                mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
            }
            mock.stream
        })),
    );
    run_to_completion(&stream).await;

    assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
    assert_eq!(finish_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn lets_a_natural_steering_or_follow_up_request_satisfy_continuation() {
    for queue_kind in ["steering", "follow-up"] {
        let queued_message = AgentMessage::user(create_user_message(queue_kind));
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let finish_calls = Arc::new(AtomicUsize::new(0));
        let steering_polls = Arc::new(AtomicUsize::new(0));
        let follow_up_delivered = Arc::new(AtomicBool::new(false));
        let second_request_users: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        let mut config = loop_config();
        {
            let finish_calls = finish_calls.clone();
            config.finish_turn = Some(finish_turn(move |_context, _signal| {
                let finish_calls = finish_calls.clone();
                Box::pin(async move {
                    let calls = finish_calls.fetch_add(1, Ordering::SeqCst) + 1;
                    (calls == 1).then_some(pi_agent::AgentTurnDecision::Continue)
                })
            }));
            let steering_polls = steering_polls.clone();
            let steering_queued = queued_message.clone();
            config.get_steering_messages = Some(get_steering_messages(move || {
                let steering_polls = steering_polls.clone();
                let steering_queued = steering_queued.clone();
                Box::pin(async move {
                    let polls = steering_polls.fetch_add(1, Ordering::SeqCst) + 1;
                    if queue_kind == "steering" && polls == 2 {
                        return vec![steering_queued];
                    }
                    Vec::new()
                })
            }));
            let follow_up_delivered = follow_up_delivered.clone();
            let follow_up_queued = queued_message.clone();
            config.get_follow_up_messages = Some(get_follow_up_messages(move || {
                let follow_up_delivered = follow_up_delivered.clone();
                let follow_up_queued = follow_up_queued.clone();
                Box::pin(async move {
                    if queue_kind != "follow-up" || follow_up_delivered.swap(true, Ordering::SeqCst)
                    {
                        return Vec::new();
                    }
                    vec![follow_up_queued]
                })
            }));
        }

        let provider_calls_for_stream = provider_calls.clone();
        let second_request_users_for_stream = second_request_users.clone();
        let stream = agent_loop(
            vec![AgentMessage::user(create_user_message("run"))],
            empty_context(),
            config,
            None,
            Some(stream_fn(move |_model, context, _options| {
                let provider_calls = provider_calls_for_stream.clone();
                let second_request_users = second_request_users_for_stream.clone();
                {
                    let call = provider_calls.fetch_add(1, Ordering::SeqCst) + 1;
                    if call == 2 {
                        second_request_users.lock().extend(user_texts(&context));
                    }
                    let mock = MockAssistantStream::new();
                    mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
                    mock.stream
                }
            })),
        );
        run_to_completion(&stream).await;

        assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
        assert_eq!(finish_calls.load(Ordering::SeqCst), 2);
        assert!(second_request_users
            .lock()
            .iter()
            .any(|text| text == queue_kind));
    }
}

#[tokio::test]
async fn prepares_the_initial_request_after_pending_messages_and_can_replace_request_state() {
    let mut replacement_model = create_model();
    replacement_model.base.id = "replacement".into();
    replacement_model.base.name = "replacement".into();
    let canonical_message = AgentMessage::user(create_user_message("canonical projection"));
    let steering_message = AgentMessage::user(create_user_message("steering"));
    let completed_messages: Arc<Mutex<Vec<AgentMessage>>> = Arc::new(Mutex::new(Vec::new()));
    let prepare_calls = Arc::new(AtomicUsize::new(0));

    let mut config = loop_config();
    {
        let completed = completed_messages.clone();
        let steering = steering_message.clone();
        config.get_steering_messages = Some(get_steering_messages(move || {
            let completed = completed.clone();
            let steering = steering.clone();
            Box::pin(async move {
                if completed.lock().iter().any(|message| message == &steering) {
                    return Vec::new();
                }
                completed.lock().push(steering.clone());
                vec![steering]
            })
        }));
        let prepare_calls = prepare_calls.clone();
        let steering = steering_message.clone();
        let canonical = canonical_message.clone();
        let replacement = replacement_model.clone();
        config.prepare_request = Some(prepare_request(move |request, _signal| {
            let prepare_calls = prepare_calls.clone();
            let completed = completed_messages.clone();
            let steering = steering.clone();
            let canonical = canonical.clone();
            let replacement = replacement.clone();
            Box::pin(async move {
                prepare_calls.fetch_add(1, Ordering::SeqCst);
                assert!(*completed.lock() == vec![steering.clone()]);
                assert!(request
                    .context
                    .messages
                    .iter()
                    .any(|message| message == &steering));
                Some(pi_agent::AgentRequestUpdate {
                    context: Some(AgentContext {
                        messages: vec![canonical],
                        tools: Some(Vec::new()),
                    }),
                    model: Some(replacement),
                    thinking_level: Some(pi_agent::AgentThinkingLevel::High),
                })
            })
        }));
    }

    let replacement_for_stream = replacement_model.clone();
    let canonical_for_stream = canonical_message.clone();
    run_agent_loop(
        vec![AgentMessage::user(create_user_message("prompt"))],
        empty_context(),
        config,
        Arc::new(|_event| Box::pin(async {})),
        None,
        Some(stream_fn(move |model, context, options| {
            assert_eq!(&model, &replacement_for_stream);
            assert_eq!(
                context.messages().to_vec(),
                vec![Message::User(user_message_from_agent(
                    &canonical_for_stream
                ))]
            );
            assert_eq!(
                options.as_ref().and_then(|options| options.reasoning),
                Some(pi_ai::types::ThinkingLevel::High)
            );
            let mock = MockAssistantStream::new();
            mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
            mock.stream
        })),
    )
    .await
    .expect("run completes");

    assert_eq!(prepare_calls.load(Ordering::SeqCst), 1);
}

fn user_message_from_agent(message: &AgentMessage) -> pi_ai::types::UserMessage {
    match message {
        AgentMessage::Llm(Message::User(user)) => user.clone(),
        _ => panic!("expected user message"),
    }
}

#[tokio::test]
async fn does_not_poll_steering_after_prepare_request() {
    let queued: Arc<Mutex<Vec<AgentMessage>>> = Arc::new(Mutex::new(Vec::new()));
    let late_steering = AgentMessage::user(create_user_message("late steering"));
    let request_included_steering: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));
    let request_preparations = Arc::new(AtomicUsize::new(0));
    let steering_polls = Arc::new(AtomicUsize::new(0));

    let mut config = loop_config();
    {
        let queued_for_poll = queued.clone();
        let steering_polls_for_hook = steering_polls.clone();
        config.get_steering_messages = Some(get_steering_messages(move || {
            let queued_for_poll = queued_for_poll.clone();
            let steering_polls = steering_polls_for_hook.clone();
            Box::pin(async move {
                steering_polls.fetch_add(1, Ordering::SeqCst);
                std::mem::take(&mut *queued_for_poll.lock())
            })
        }));
        let request_preparations = request_preparations.clone();
        let queued_for_hook = queued.clone();
        let late = late_steering.clone();
        config.prepare_request = Some(prepare_request(move |_request, _signal| {
            let request_preparations = request_preparations.clone();
            let queued_for_hook = queued_for_hook.clone();
            let late = late.clone();
            Box::pin(async move {
                let call = request_preparations.fetch_add(1, Ordering::SeqCst) + 1;
                if call == 1 {
                    queued_for_hook.lock().push(late.clone());
                }
                None
            })
        }));
    }

    let request_included_for_stream = request_included_steering.clone();
    let late_for_stream = late_steering.clone();
    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("run"))],
        empty_context(),
        config,
        None,
        Some(stream_fn(move |_model, context, _options| {
            request_included_for_stream
                .lock()
                .push(context.messages().iter().any(|message| {
                    *message == Message::User(user_message_from_agent(&late_for_stream))
                }));
            let mock = MockAssistantStream::new();
            mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
            mock.stream
        })),
    );
    run_to_completion(&stream).await;

    assert_eq!(*request_included_steering.lock(), vec![false, true]);
    assert_eq!(request_preparations.load(Ordering::SeqCst), 2);
    // Startup, post-turn delivery, then the final natural-stop check.
    assert_eq!(steering_polls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn uses_prepare_next_turn_snapshot_before_continuing() {
    let tool = echo_tool(Arc::new(Mutex::new(Vec::new())));
    let converted_second_turn_has_update = Arc::new(AtomicBool::new(false));
    let prepare_calls = Arc::new(AtomicUsize::new(0));
    let prepared = Arc::new(AtomicBool::new(false));

    let mut config = loop_config();
    {
        let prepare_calls = prepare_calls.clone();
        let prepared = prepared.clone();
        config.prepare_next_turn = Some(prepare_next_turn(move |current_context| {
            let prepare_calls = prepare_calls.clone();
            let prepared = prepared.clone();
            Box::pin(async move {
                prepare_calls.fetch_add(1, Ordering::SeqCst);
                if prepared.load(Ordering::SeqCst) {
                    return None;
                }
                prepared.store(true, Ordering::SeqCst);
                Some(AgentLoopTurnUpdate {
                    context: Some(AgentContext {
                        messages: current_context.context.messages.clone(),
                        tools: current_context.context.tools.clone(),
                    }),
                    messages: vec![AgentMessage::system(system_message("updated guidance", 1))],
                    ..Default::default()
                })
            })
        }));
    }

    let llm_calls = Arc::new(AtomicUsize::new(0));
    let converted_flag = converted_second_turn_has_update.clone();
    let llm_calls_for_stream = llm_calls.clone();
    let stream_fn_impl = stream_fn(move |_model, ctx, _options| {
        let call = llm_calls_for_stream.fetch_add(1, Ordering::SeqCst) + 1;
        if call == 2 && ctx.messages().iter().any(|message| {
			matches!(message, Message::System(system) if matches!(&system.content, SystemContent::Text(text) if text == "updated guidance"))
		}) {
			converted_flag.store(true, Ordering::SeqCst);
		}
        let mock = MockAssistantStream::new();
        if call == 1 {
            mock.push_done(
                tool_use_message(vec![create_tool_call(
                    "tool-1",
                    "echo",
                    json!({ "value": "hello" }),
                )]),
                StopReason::ToolUse,
            );
        } else {
            mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
        }
        mock.stream
    });

    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("echo something"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(stream_fn_impl),
    );

    run_to_completion(&stream).await;

    assert_eq!(llm_calls.load(Ordering::SeqCst), 2);
    assert_eq!(prepare_calls.load(Ordering::SeqCst), 1);
    assert!(converted_second_turn_has_update.load(Ordering::SeqCst));
}

#[tokio::test]
async fn picks_up_steering_queued_during_prepare_next_turn_before_the_next_request() {
    let tool = create_tool("noop");
    let queued: Arc<Mutex<Vec<AgentMessage>>> = Arc::new(Mutex::new(Vec::new()));
    let late_steering = AgentMessage::user(create_user_message("late steering"));
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let second_request_included_steering = Arc::new(AtomicBool::new(false));

    let mut config = loop_config();
    {
        let queued_for_hook = queued.clone();
        let late_for_hook = late_steering.clone();
        config.prepare_next_turn = Some(prepare_next_turn(move |_context| {
            let queued_for_hook = queued_for_hook.clone();
            let late_for_hook = late_for_hook.clone();
            Box::pin(async move {
                queued_for_hook.lock().push(late_for_hook.clone());
                None
            })
        }));
        let queued_for_poll = queued.clone();
        config.get_steering_messages = Some(get_steering_messages(move || {
            let queued_for_poll = queued_for_poll.clone();
            Box::pin(async move { std::mem::take(&mut *queued_for_poll.lock()) })
        }));
    }

    let provider_calls_for_stream = provider_calls.clone();
    let second_flag = second_request_included_steering.clone();
    let late_for_stream = late_steering.clone();
    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("run"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(stream_fn(move |_model, context, _options| {
            let provider_calls = provider_calls_for_stream.clone();
            let second_flag = second_flag.clone();
            let late = late_for_stream.clone();
            {
                let call = provider_calls.fetch_add(1, Ordering::SeqCst) + 1;
                let late_message = Message::User(user_message_from_agent(&late));
                if call == 2
                    && context
                        .messages()
                        .iter()
                        .any(|message| *message == late_message)
                {
                    second_flag.store(true, Ordering::SeqCst);
                }
                let mock = MockAssistantStream::new();
                if call == 1 {
                    mock.push_done(
                        tool_use_message(vec![create_tool_call("tool-1", "noop", json!({}))]),
                        StopReason::ToolUse,
                    );
                } else {
                    mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
                }
                mock.stream
            }
        })),
    );
    run_to_completion(&stream).await;

    assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
    assert!(second_request_included_steering.load(Ordering::SeqCst));
}

#[tokio::test]
async fn action_end_receives_finalized_turn_context_and_stops_before_queue_polling() {
    let executed: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let tool = echo_tool(executed.clone());

    let steering_polls = Arc::new(AtomicUsize::new(0));
    let follow_up_polls = Arc::new(AtomicUsize::new(0));
    let callback_tool_result_ids: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let callback_context_roles: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let mut config = loop_config();
    {
        let callback_tool_result_ids = callback_tool_result_ids.clone();
        let callback_context_roles = callback_context_roles.clone();
        config.finish_turn = Some(finish_turn(move |turn, _signal| {
            let callback_tool_result_ids = callback_tool_result_ids.clone();
            let callback_context_roles = callback_context_roles.clone();
            Box::pin(async move {
                assert!(turn.message.stop_reason != StopReason::Aborted || true);
                let _ = &turn.message; // assistant message asserted via stop_reason below
                *callback_tool_result_ids.lock() = turn
                    .tool_results
                    .iter()
                    .map(|result| result.tool_call_id.clone())
                    .collect();
                *callback_context_roles.lock() = turn
                    .context
                    .messages
                    .iter()
                    .map(|message| message.role().to_string())
                    .collect();
                Some(pi_agent::AgentTurnDecision::End)
            })
        }));
        let steering_polls = steering_polls.clone();
        config.get_steering_messages = Some(get_steering_messages(move || {
            steering_polls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Vec::new() })
        }));
        let follow_up_polls = follow_up_polls.clone();
        config.get_follow_up_messages = Some(get_follow_up_messages(move || {
            follow_up_polls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                vec![AgentMessage::user(create_user_message(
                    "follow up should stay queued",
                ))]
            })
        }));
    }

    let llm_calls = Arc::new(AtomicUsize::new(0));
    let llm_calls_for_stream = llm_calls.clone();
    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("echo something"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(stream_fn(move |_model, _context, _options| {
            let call = llm_calls_for_stream.fetch_add(1, Ordering::SeqCst) + 1;
            let mock = MockAssistantStream::new();
            if call == 1 {
                mock.push_done(
                    tool_use_message(vec![create_tool_call(
                        "tool-1",
                        "echo",
                        json!({ "value": "hello" }),
                    )]),
                    StopReason::ToolUse,
                );
            } else {
                mock.push_done(
                    create_assistant_message_text("should not run"),
                    StopReason::Stop,
                );
            }
            mock.stream
        })),
    );

    let (events, messages) = run_to_completion(&stream).await;

    assert_eq!(llm_calls.load(Ordering::SeqCst), 1);
    assert_eq!(*executed.lock(), vec![json!("hello")]);
    assert_eq!(steering_polls.load(Ordering::SeqCst), 1);
    assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
    assert_eq!(*callback_tool_result_ids.lock(), vec!["tool-1".to_string()]);
    assert_eq!(
        *callback_context_roles.lock(),
        vec!["system", "user", "assistant", "toolResult"]
    );
    // The context declares no tools, so the loop announces the loadout with a system message.
    assert_eq!(
        message_roles(&messages),
        vec!["system", "user", "assistant", "toolResult"]
    );
    assert_eq!(
        event_types(&events),
        vec![
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "tool_execution_start",
            "tool_execution_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );
}

#[tokio::test]
async fn stops_after_a_tool_batch_when_every_tool_result_sets_terminate() {
    let mut tool = echo_tool(Arc::new(Mutex::new(Vec::new())));
    tool.execute = tool_execute(|_id, params| {
        Ok(AgentToolResult {
            content: vec![UserContent::Text(pi_ai::types::TextContent::new(format!(
                "echoed: {}",
                params
                    .get("value")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
            )))],
            details: json!({ "value": params.get("value").cloned().unwrap_or(Value::Null) }),
            structured_content: None,
            usage: None,
            is_error: None,
            terminate: Some(true),
        })
    });

    let llm_calls = Arc::new(AtomicUsize::new(0));
    let llm_calls_for_stream = llm_calls.clone();
    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("echo something"))],
        agent_context(vec![], vec![tool]),
        loop_config(),
        None,
        Some(stream_fn(move |_model, _context, _options| {
            llm_calls_for_stream.fetch_add(1, Ordering::SeqCst);
            let mock = MockAssistantStream::new();
            mock.push_done(
                tool_use_message(vec![create_tool_call(
                    "tool-1",
                    "echo",
                    json!({ "value": "hello" }),
                )]),
                StopReason::ToolUse,
            );
            mock.stream
        })),
    );

    let (events, messages) = run_to_completion(&stream).await;

    assert_eq!(llm_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        message_roles(&messages),
        vec!["system", "user", "assistant", "toolResult"]
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AgentEvent::TurnEnd { .. }))
            .count(),
        1
    );
}

#[tokio::test]
async fn stops_after_a_blocked_tool_call_when_before_tool_call_sets_terminate() {
    let executed = Arc::new(AtomicBool::new(false));
    let mut tool = echo_tool(Arc::new(Mutex::new(Vec::new())));
    {
        let executed = executed.clone();
        tool.execute = tool_execute(move |_id, _params| {
            executed.store(true, Ordering::SeqCst);
            Ok(AgentToolResult::new(
                vec![UserContent::Text(pi_ai::types::TextContent::new(
                    "should not execute",
                ))],
                json!({ "value": "unexpected" }),
            ))
        });
    }

    let mut config = loop_config();
    config.before_tool_call = Some(before_tool_call(|_context, _signal| {
        Box::pin(async move {
            Ok(Some(BeforeToolCallResult {
                block: Some(true),
                reason: Some("Blocked by policy".into()),
                terminate: Some(true),
                args: None,
            }))
        })
    }));

    let llm_calls = Arc::new(AtomicUsize::new(0));
    let llm_calls_for_stream = llm_calls.clone();
    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("echo something"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(stream_fn(move |_model, _context, _options| {
            let call = llm_calls_for_stream.fetch_add(1, Ordering::SeqCst) + 1;
            let mock = MockAssistantStream::new();
            if call == 1 {
                mock.push_done(
                    tool_use_message(vec![create_tool_call(
                        "tool-1",
                        "echo",
                        json!({ "value": "hello" }),
                    )]),
                    StopReason::ToolUse,
                );
            } else {
                mock.push_done(
                    create_assistant_message_text("should not run"),
                    StopReason::Stop,
                );
            }
            mock.stream
        })),
    );

    run_to_completion(&stream).await;

    assert!(!executed.load(Ordering::SeqCst));
    assert_eq!(llm_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn continues_after_a_mixed_batch_with_one_terminating_blocked_call() {
    let executed: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let tool = echo_tool(executed.clone());

    let mut config = loop_config();
    config.tool_execution = Some(pi_agent::ToolExecutionMode::Parallel);
    config.before_tool_call = Some(before_tool_call(|context, _signal| {
        Box::pin(async move {
            let value = context.args.get("value").and_then(|value| value.as_str());
            if value == Some("first") {
                Ok(Some(BeforeToolCallResult {
                    block: Some(true),
                    reason: Some("Blocked first".into()),
                    terminate: Some(true),
                    args: None,
                }))
            } else {
                Ok(None)
            }
        })
    }));

    let llm_calls = Arc::new(AtomicUsize::new(0));
    let llm_calls_for_stream = llm_calls.clone();
    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("echo both"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(stream_fn(move |_model, _context, _options| {
            let call = llm_calls_for_stream.fetch_add(1, Ordering::SeqCst) + 1;
            let mock = MockAssistantStream::new();
            if call == 1 {
                mock.push_done(
                    tool_use_message(vec![
                        create_tool_call("tool-1", "echo", json!({ "value": "first" })),
                        create_tool_call("tool-2", "echo", json!({ "value": "second" })),
                    ]),
                    StopReason::ToolUse,
                );
            } else {
                mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
            }
            mock.stream
        })),
    );

    run_to_completion(&stream).await;

    assert_eq!(*executed.lock(), vec![json!("second")]);
    assert_eq!(llm_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn continues_after_parallel_tool_calls_when_not_all_tool_results_terminate() {
    let mut tool = echo_tool(Arc::new(Mutex::new(Vec::new())));
    tool.execute = tool_execute(|_id, params| {
        let value = params
            .get("value")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        Ok(AgentToolResult {
            content: vec![UserContent::Text(pi_ai::types::TextContent::new(format!(
                "echoed: {value}"
            )))],
            details: json!({ "value": value }),
            structured_content: None,
            usage: None,
            is_error: None,
            terminate: (value == "first").then_some(true),
        })
    });

    let mut config = loop_config();
    config.tool_execution = Some(pi_agent::ToolExecutionMode::Parallel);

    let call_counter = Arc::new(AtomicUsize::new(0));
    let counter_for_stream = call_counter.clone();
    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("echo both"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(stream_fn(move |_model, _context, _options| {
            let call = counter_for_stream.fetch_add(1, Ordering::SeqCst);
            let mock = MockAssistantStream::new();
            if call == 0 {
                mock.push_done(
                    tool_use_message(vec![
                        create_tool_call("tool-1", "echo", json!({ "value": "first" })),
                        create_tool_call("tool-2", "echo", json!({ "value": "second" })),
                    ]),
                    StopReason::ToolUse,
                );
            } else {
                mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
            }
            mock.stream
        })),
    );

    run_to_completion(&stream).await;

    assert_eq!(call_counter.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn allows_after_tool_call_to_mark_a_tool_batch_as_terminating() {
    let tool = echo_tool(Arc::new(Mutex::new(Vec::new())));

    let mut config = loop_config();
    config.after_tool_call = Some(after_tool_call(|_context, _signal| {
        Box::pin(async {
            Ok(Some(AfterToolCallResult {
                terminate: Some(true),
                ..Default::default()
            }))
        })
    }));

    let llm_calls = Arc::new(AtomicUsize::new(0));
    let llm_calls_for_stream = llm_calls.clone();
    let stream = agent_loop(
        vec![AgentMessage::user(create_user_message("echo something"))],
        agent_context(vec![], vec![tool]),
        config,
        None,
        Some(stream_fn(move |_model, _context, _options| {
            llm_calls_for_stream.fetch_add(1, Ordering::SeqCst);
            let mock = MockAssistantStream::new();
            mock.push_done(
                tool_use_message(vec![create_tool_call(
                    "tool-1",
                    "echo",
                    json!({ "value": "hello" }),
                )]),
                StopReason::ToolUse,
            );
            mock.stream
        })),
    );

    run_to_completion(&stream).await;

    assert_eq!(llm_calls.load(Ordering::SeqCst), 1);
}

// ==== agentLoopContinue with AgentMessage ====

#[tokio::test]
async fn continue_throws_when_context_has_no_messages() {
    let result = agent_loop_continue(
        empty_context(),
        loop_config(),
        None,
        Some(unexpected_stream_fn()),
    );
    match result {
        Ok(_) => panic!("expected error"),
        Err(error) => assert_eq!(error.0, "Cannot continue: no messages in context"),
    }
}

fn unexpected_stream_fn() -> pi_agent::StreamFn {
    stream_fn(|_model, _context, _options| panic!("Unexpected stream call"))
}

#[tokio::test]
async fn continues_from_existing_context_without_emitting_user_message_events() {
    let stream = agent_loop_continue(
        agent_context(
            vec![AgentMessage::user(create_user_message("Hello"))],
            vec![],
        ),
        loop_config(),
        None,
        Some(stream_fn(move |_model, _context, _options| {
            let mock = MockAssistantStream::new();
            mock.push_done(create_assistant_message_text("Response"), StopReason::Stop);
            mock.stream
        })),
    )
    .expect("continue succeeds");

    let (events, messages) = run_to_completion(&stream).await;

    // Should only return the new assistant message (not the existing user message)
    assert_eq!(messages.len(), 1);
    assert_eq!(message_roles(&messages), vec!["assistant"]);

    // Should NOT have user message events (that's the key difference from agentLoop)
    let message_end_events = message_end_roles(&events);
    assert_eq!(message_end_events, vec!["assistant"]);
}

#[tokio::test]
async fn allows_custom_message_types_as_last_message_caller_responsibility() {
    // Custom message that will be converted to user message by convertToLlm
    let custom_message = AgentMessage::Custom(json!({
        "role": "custom",
        "text": "Hook content",
        "timestamp": pi_agent::now_ms(),
    }));

    let mut config = loop_config();
    config.convert_to_llm = Arc::new(|messages| {
        Box::pin(async move {
            // Convert custom to user message
            messages
                .iter()
                .map(|message| match message.role() {
                    "custom" => AgentMessage::user(pi_ai::types::UserMessage {
                        content: UserContentField::Text(
                            message.as_llm().map(|_| String::new()).unwrap_or_else(
                                || match message {
                                    AgentMessage::Custom(value) => value
                                        .get("text")
                                        .and_then(|text| text.as_str())
                                        .unwrap_or("")
                                        .to_string(),
                                    _ => String::new(),
                                },
                            ),
                        ),
                        timestamp: message.timestamp(),
                    }),
                    _ => message.clone(),
                })
                .filter(|message| matches!(message.role(), "user" | "assistant" | "toolResult"))
                .filter_map(|message| message.as_llm().cloned())
                .collect()
        })
    });

    let stream = agent_loop_continue(
        agent_context(vec![custom_message], vec![]),
        config,
        None,
        Some(stream_fn(move |_model, _context, _options| {
            let mock = MockAssistantStream::new();
            mock.push_done(
                create_assistant_message_text("Response to custom message"),
                StopReason::Stop,
            );
            mock.stream
        })),
    )
    .expect("custom last message is caller responsibility");

    let (_events, messages) = run_to_completion(&stream).await;
    assert_eq!(messages.len(), 1);
    assert_eq!(message_roles(&messages), vec!["assistant"]);
}

// ==== runToolCall ====

fn echo_tool_for_run() -> pi_agent::AgentTool {
    let mut tool = echo_tool_with_updates(Arc::new(Mutex::new(Vec::new())));
    tool.output_schema = Some(object_schema(&[("value", "string")]));
    tool
}

fn failing_tool() -> pi_agent::AgentTool {
    AgentTool::new(
        "failing",
        "Returns an error result",
        empty_object_schema(),
        "Failing",
        tool_execute(|_id, _params| {
            Ok(AgentToolResult {
                content: vec![UserContent::Text(pi_ai::types::TextContent::new("bad"))],
                details: json!({ "partial": true }),
                structured_content: None,
                usage: None,
                is_error: Some(true),
                terminate: None,
            })
        }),
    )
}

#[tokio::test]
async fn validates_runs_the_hooks_and_reports_failures_as_error_outcomes() {
    let hook_calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let updates: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let assistant_message = create_assistant_message(Vec::new(), StopReason::Stop);
    let call = |id: &str, name: &str, args: Value| create_tool_call(id, name, args);

    let tools = vec![echo_tool_with_updates(updates.clone()), failing_tool()];
    let before = {
        let hook_calls = hook_calls.clone();
        before_hook(hook_calls)
    };
    let after = {
        let hook_calls = hook_calls.clone();
        after_hook(hook_calls)
    };

    let options_for = || RunToolCallOptions {
        tools: tools.clone(),
        assistant_message: assistant_message.clone(),
        context: AgentContext {
            messages: Vec::new(),
            tools: None,
        },
        signal: None,
        on_update: None,
        before_tool_call: before.clone(),
        after_tool_call: after.clone(),
    };

    let a = run_tool_call(call("a", "echo", json!({ "value": "a" })), options_for()).await;
    assert_eq!(a.tool_call.id, "a");
    assert_eq!(a.result.structured_content, Some(json!({ "value": "a" })));
    assert!(!a.is_error);

    let b = run_tool_call(
        call("b", "echo", json!({ "value": { "nested": true } })),
        options_for(),
    )
    .await;
    assert!(b.is_error);

    let c = run_tool_call(
        call("c", "echo", json!({ "value": "blocked" })),
        options_for(),
    )
    .await;
    let c_text = c
        .result
        .content
        .iter()
        .filter_map(|block| match block {
            UserContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .next()
        .unwrap_or_default();
    assert_eq!(c_text, "nope");
    assert!(c.is_error);

    let d = run_tool_call(call("d", "missing", json!({})), options_for()).await;
    let d_text = d
        .result
        .content
        .iter()
        .filter_map(|block| match block {
            UserContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .next()
        .unwrap_or_default();
    assert_eq!(d_text, "Tool missing not found");
    assert!(d.is_error);

    // Error results keep their details.
    let e = run_tool_call(call("e", "failing", json!({})), options_for()).await;
    assert_eq!(e.result.details, json!({ "partial": true }));
    assert!(e.is_error);

    assert_eq!(
        *updates.lock(),
        vec![json!({ "content": [{ "type": "text", "text": "partial" }], "details": {} })]
    );
    // Validation failures and unknown tools never reach the hooks; blocked calls skip afterToolCall.
    assert_eq!(
        *hook_calls.lock(),
        vec![
            "before a".to_string(),
            "after a".to_string(),
            "before c".to_string(),
            "before e".to_string(),
            "after e".to_string(),
        ]
    );
}

/// Echo tool emitting one `partial` update (used by the runToolCall tests).
fn echo_tool_with_updates(updates: Arc<Mutex<Vec<Value>>>) -> pi_agent::AgentTool {
    AgentTool::new(
        "echo",
        "Echo tool",
        object_schema(&[("value", "string")]),
        "Echo",
        Arc::new(move |_tool_call_id, params, _signal, on_update| {
            let updates = updates.clone();
            let value = params
                .get("value")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string();
            Box::pin(async move {
                if let Some(on_update) = &on_update {
                    let partial = AgentToolResult::new(
                        vec![UserContent::Text(pi_ai::types::TextContent::new("partial"))],
                        json!({}),
                    );
                    updates.lock().push(serde_json::to_value(&partial).unwrap());
                    on_update(partial);
                }
                Ok(AgentToolResult {
                    content: vec![UserContent::Text(pi_ai::types::TextContent::new(
                        value.clone(),
                    ))],
                    details: json!({}),
                    structured_content: Some(json!({ "value": value })),
                    usage: None,
                    is_error: None,
                    terminate: None,
                })
            })
        }),
    )
}

fn before_hook(
    hook_calls: Arc<Mutex<Vec<String>>>,
) -> Option<
    Arc<
        dyn Fn(
                pi_agent::BeforeToolCallContext,
                Option<pi_ai::utils::abort::AbortSignal>,
            )
                -> pi_agent::BoxFuture<'static, Result<Option<BeforeToolCallResult>, String>>
            + Send
            + Sync,
    >,
> {
    Some(Arc::new(move |context, _signal| {
        let hook_calls = hook_calls.clone();
        Box::pin(async move {
            hook_calls
                .lock()
                .push(format!("before {}", context.tool_call.id));
            let blocked =
                context.args.get("value").and_then(|value| value.as_str()) == Some("blocked");
            Ok(blocked.then(|| BeforeToolCallResult {
                block: Some(true),
                reason: Some("nope".into()),
                ..Default::default()
            }))
        })
    }))
}

fn after_hook(
    hook_calls: Arc<Mutex<Vec<String>>>,
) -> Option<
    Arc<
        dyn Fn(
                pi_agent::AfterToolCallContext,
                Option<pi_ai::utils::abort::AbortSignal>,
            )
                -> pi_agent::BoxFuture<'static, Result<Option<AfterToolCallResult>, String>>
            + Send
            + Sync,
    >,
> {
    Some(Arc::new(move |context, _signal| {
        let hook_calls = hook_calls.clone();
        Box::pin(async move {
            hook_calls
                .lock()
                .push(format!("after {}", context.tool_call.id));
            Ok(None)
        })
    }))
}

#[tokio::test]
async fn lets_after_tool_call_replace_structured_content_and_drops_it_when_only_content_is_replaced(
) {
    let redacted = vec![UserContent::Text(pi_ai::types::TextContent::new(
        "redacted",
    ))];
    let after_results = vec![
        AfterToolCallResult {
            content: Some(redacted.clone()),
            ..Default::default()
        },
        AfterToolCallResult {
            structured_content: Some(json!({ "value": "replaced" })),
            ..Default::default()
        },
        AfterToolCallResult {
            content: Some(redacted),
            structured_content: Some(json!({ "value": "both" })),
            ..Default::default()
        },
        AfterToolCallResult {
            details: Some(json!({ "note": "kept" })),
            ..Default::default()
        },
    ];

    let assistant_message = create_assistant_message(Vec::new(), StopReason::Stop);
    let mut seen = Vec::new();
    for after_result in after_results {
        let outcome = run_tool_call(
            create_tool_call("x", "echo", json!({ "value": "original" })),
            RunToolCallOptions {
                tools: vec![echo_tool_for_run()],
                assistant_message: assistant_message.clone(),
                context: AgentContext {
                    messages: Vec::new(),
                    tools: None,
                },
                signal: None,
                on_update: None,
                before_tool_call: None,
                after_tool_call: Some(after_tool_call(move |_context, _signal| {
                    let after_result = after_result.clone();
                    Box::pin(async move { Ok(Some(after_result)) })
                })),
            },
        )
        .await;
        seen.push(outcome.result.structured_content);
    }
    assert_eq!(
        seen,
        vec![
            None,
            Some(json!({ "value": "replaced" })),
            Some(json!({ "value": "both" })),
            Some(json!({ "value": "original" })),
        ]
    );
}
