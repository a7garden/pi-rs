//! Port of `packages/agent/test/agent.test.ts`.

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use futures_util::StreamExt;
use parking_lot::Mutex;
use pi_agent::{
    finish_turn, prepare_next_turn_with_context, Agent, AgentEvent, AgentInitialState,
    AgentMessage, AgentOptions, AgentTool, AgentToolResult,
};
use pi_ai::types::{
    AssistantMessageEvent, Message, StopReason, StreamErrorReason, SystemContent, SystemMessage,
    ToolReference, UserContent,
};
use pi_ai::utils::transcript::get_current_system_message;
use serde_json::{json, Value};

use common::*;

fn empty_initial_state() -> AgentInitialState {
    AgentInitialState::default()
}

fn tool_with_name(name: &str) -> AgentTool {
    create_tool(name)
}

fn unused_stream_fn() -> pi_agent::StreamFn {
    stream_fn(|_model, _context, _options| panic!("Unexpected stream call"))
}

/// Stream fn producing one immediate `done` with the given text.
fn done_stream_fn(text: &'static str) -> pi_agent::StreamFn {
    stream_fn(move |_model, _context, _options| {
        let mock = MockAssistantStream::new();
        mock.push_done(create_assistant_message_text(text), StopReason::Stop);
        mock.stream
    })
}

/// Stream fn that streams `start` then polls the request signal, ending with
/// an `aborted` error event once aborted.
fn abort_aware_stream_fn() -> pi_agent::StreamFn {
    stream_fn(move |_model, _context, options| {
        let mock = MockAssistantStream::new();
        let signal = options.and_then(|options| options.base.signal.clone());
        mock.stream.push(AssistantMessageEvent::Start {
            partial: create_assistant_message_text(""),
        });
        let producer = mock.stream.clone();
        tokio::spawn(async move {
            loop {
                if signal.as_ref().is_some_and(|signal| signal.aborted()) {
                    producer.push(AssistantMessageEvent::Error {
                        reason: StreamErrorReason::Aborted,
                        error: create_assistant_message_text("Aborted"),
                    });
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        });
        mock.stream
    })
}

fn notify_pair() -> (Arc<tokio::sync::Notify>, Arc<tokio::sync::Notify>) {
    (
        Arc::new(tokio::sync::Notify::new()),
        Arc::new(tokio::sync::Notify::new()),
    )
}

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

    let agent = Arc::new(Agent::new(AgentOptions::default()));
    agent
        .clone()
        .prompt("Hello")
        .await
        .expect("prompt succeeds");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    pi_agent::set_default_stream_fn(None);
}

#[tokio::test]
async fn creates_an_agent_instance_with_default_state() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        ..Default::default()
    });

    assert_eq!(agent.model().base.id, "unknown");
    assert_eq!(agent.thinking_level(), pi_agent::AgentThinkingLevel::Off);
    assert!(agent.tools().is_empty());
    assert!(agent.messages().is_empty());
    assert!(!agent.is_streaming());
    assert_eq!(agent.streaming_message(), None);
    assert!(agent.pending_tool_calls().is_empty());
    assert_eq!(agent.error_message(), None);
}

#[tokio::test]
async fn creates_an_agent_instance_with_custom_initial_state() {
    let custom_model = pi_ai::compat::get_model("openai", "gpt-4o-mini").expect("catalog model");
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        initial_state: AgentInitialState {
            system_prompt: Some("You are a helpful assistant.".into()),
            model: Some(custom_model.clone()),
            thinking_level: Some(pi_agent::AgentThinkingLevel::Low),
            ..empty_initial_state()
        },
        ..Default::default()
    });

    assert_eq!(agent.messages().len(), 1);
    match &agent.messages()[0] {
        AgentMessage::Llm(Message::System(system)) => {
            assert!(
                matches!(&system.content, SystemContent::Text(text) if text == "You are a helpful assistant.")
            );
            assert_eq!(system.timestamp, 0);
        }
        other => panic!("expected system message, got {other:?}"),
    }
    assert_eq!(agent.model(), custom_model);
    assert_eq!(agent.thinking_level(), pi_agent::AgentThinkingLevel::Low);
}

#[tokio::test]
async fn converts_initial_prompt_and_tools_into_transcript_state() {
    let tool = AgentTool::new(
        "echo",
        "Echo input",
        empty_object_schema(),
        "Echo",
        tool_execute(|_id, _params| {
            Ok(AgentToolResult::new(
                vec![UserContent::Text(pi_ai::types::TextContent::new("echo"))],
                json!({}),
            ))
        }),
    );
    let agent = Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some("You are helpful.".into()),
            tools: vec![tool],
            ..empty_initial_state()
        },
        stream_fn: Some(unused_stream_fn()),
        ..Default::default()
    });

    let initial = &agent.messages()[0];
    let AgentMessage::Llm(Message::System(system)) = initial else {
        panic!("expected system message")
    };
    assert!(matches!(&system.content, SystemContent::Text(text) if text == "You are helpful."));
    let tools_added = system.tools_added.as_ref().expect("toolsAdded");
    assert_eq!(
        tools_added
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        vec!["echo"]
    );
}

#[tokio::test]
async fn declares_tool_loadout_changes_to_the_model_before_the_next_request() {
    let first = tool_with_name("first");
    let second = tool_with_name("second");
    let requests: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let requests_for_stream = requests.clone();
    let agent = Arc::new(Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some("You are helpful.".into()),
            tools: vec![first],
            ..empty_initial_state()
        },
        stream_fn: Some(stream_fn(move |_model, context, _options| {
            let markers = context
                .messages()
                .iter()
                .filter_map(|message| match message {
                    Message::System(system) => {
                        let added = system
                            .tools_added
                            .as_ref()
                            .map(|tools| {
                                tools
                                    .iter()
                                    .map(|tool| tool.name.clone())
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        let removed = system
                            .tools_removed
                            .as_ref()
                            .map(|tools| {
                                tools
                                    .iter()
                                    .map(|tool| tool.name.clone())
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        Some(format!("+{}-{}", added.join(","), removed.join(",")))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            requests_for_stream.lock().push(markers);
            let mock = MockAssistantStream::new();
            mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
            mock.stream
        })),
        ..Default::default()
    }));

    agent.clone().prompt("one").await.expect("prompt");
    agent.set_tools(vec![second]);
    agent.clone().prompt("two").await.expect("prompt");
    agent.clone().prompt("three").await.expect("prompt");

    // Each request announces the cumulative transcript delta; the marker
    // strings concatenate `+added` and `-removed` per system message.
    let requests = requests.lock();
    assert_eq!(requests.len(), 3);
    assert!(requests[0].iter().all(|marker| marker.contains("+first")));
    assert!(requests[1].iter().any(|marker| marker.contains("+second")));
    assert!(requests[1].iter().any(|marker| marker.contains("-first")));
    assert!(requests[2].iter().any(|marker| marker.contains("+second")));
    assert!(requests[2].iter().any(|marker| marker.contains("-first")));

    let update = agent.messages().iter().find_map(|message| match message {
        AgentMessage::Llm(Message::System(system)) if system.tools_removed.is_some() => {
            Some(system.clone())
        }
        _ => None,
    });
    let update = update.expect("tool update message");
    assert_eq!(
        update.tools_removed,
        Some(vec![ToolReference {
            name: "first".into()
        }])
    );
    let initial = &agent.messages()[0];
    let AgentMessage::Llm(Message::System(system)) = initial else {
        panic!("expected system message")
    };
    // Declarations never carry the executable closure.
    assert!(system
        .tools_added
        .as_ref()
        .map(|tools| !tools.is_empty())
        .unwrap_or(false));
}

#[tokio::test]
async fn merges_tool_changes_into_a_pending_system_message() {
    let tool = tool_with_name("echo");
    let saw_two_system: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    let flag = saw_two_system.clone();
    let agent = Arc::new(Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some("You are helpful.".into()),
            ..empty_initial_state()
        },
        stream_fn: Some(stream_fn(move |_model, context, _options| {
            if context
                .messages()
                .iter()
                .filter(|message| matches!(message, Message::System(_)))
                .count()
                == 2
            {
                flag.store(true, Ordering::SeqCst);
            }
            let mock = MockAssistantStream::new();
            mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
            mock.stream
        })),
        ..Default::default()
    }));

    agent.set_tools(vec![tool]);
    agent
        .clone()
        .prompt(vec![
            AgentMessage::system(SystemMessage {
                content: SystemContent::Text(String::new()),
                sections: Some(
                    [("skills".to_string(), Some("<skills>x</skills>".to_string()))]
                        .into_iter()
                        .collect(),
                ),
                tools_added: None,
                tools_removed: None,
                timestamp: 1,
            }),
            AgentMessage::user(pi_ai::types::UserMessage {
                content: UserContentField::Text("hi".into()),
                timestamp: 2,
            }),
        ])
        .await
        .expect("prompt");

    assert!(saw_two_system.load(Ordering::SeqCst));
    let AgentMessage::Llm(Message::System(merged)) = &agent.messages()[1] else {
        panic!("expected system message at index 1");
    };
    assert!(matches!(&merged.content, SystemContent::Text(text) if text.is_empty()));
    assert_eq!(
        merged
            .sections
            .as_ref()
            .map(|sections| sections.get("skills").cloned().flatten()),
        Some(Some("<skills>x</skills>".to_string()))
    );
    assert_eq!(merged.timestamp, 1);
    let tools_added = merged.tools_added.as_ref().expect("toolsAdded merged in");
    assert_eq!(
        tools_added
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        vec!["echo"]
    );
}

use pi_ai::types::UserContentField;

#[tokio::test]
async fn rewrites_pending_tool_declarations_to_match_the_executable_set() {
    let agent = Arc::new(Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some("You are helpful.".into()),
            tools: vec![tool_with_name("first")],
            ..empty_initial_state()
        },
        stream_fn: Some(done_stream_fn("done")),
        ..Default::default()
    }));

    // The pending message claims to add `second` and remove `first`, but the executable
    // set still has `first` and lacks `second`: the executable set wins.
    agent
        .clone()
        .prompt(vec![
            AgentMessage::system(SystemMessage {
                content: SystemContent::Text(String::new()),
                sections: Some(
                    [("note".to_string(), Some("<note>x</notes>".to_string()))]
                        .into_iter()
                        .collect(),
                ),
                tools_added: Some(vec![pi_ai::utils::transcript::to_tool_declaration(
                    &tool_with_name("second").tool,
                )]),
                tools_removed: Some(vec![ToolReference {
                    name: "first".into(),
                }]),
                timestamp: 1,
            }),
            AgentMessage::user(pi_ai::types::UserMessage {
                content: UserContentField::Text("hi".into()),
                timestamp: 2,
            }),
        ])
        .await
        .expect("prompt");

    let AgentMessage::Llm(Message::System(rewritten)) = &agent.messages()[1] else {
        panic!("expected system message at index 1");
    };
    assert!(rewritten.tools_added.is_none());
    assert!(rewritten.tools_removed.is_none());
    assert_eq!(rewritten.timestamp, 1);
    let messages: Vec<Message> = agent
        .messages()
        .iter()
        .filter_map(|message| message.as_llm().cloned())
        .collect();
    let current = get_current_system_message(&messages).expect("current system message");
    assert_eq!(
        current.tools_added.as_ref().map(|tools| tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<_>>()),
        Some(vec!["first".to_string()])
    );
}

#[tokio::test]
async fn restores_the_transcript_baseline_when_reset() {
    let tool = tool_with_name("echo");
    let agent = Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some("You are helpful.".into()),
            tools: vec![tool],
            messages: vec![AgentMessage::user(pi_ai::types::UserMessage {
                content: UserContentField::Text("old".into()),
                timestamp: 1,
            })],
            ..empty_initial_state()
        },
        stream_fn: Some(unused_stream_fn()),
        ..Default::default()
    });

    agent.reset().expect("reset while idle");

    assert_eq!(agent.messages().len(), 1);
    let AgentMessage::Llm(Message::System(system)) = &agent.messages()[0] else {
        panic!("expected system message");
    };
    assert!(matches!(&system.content, SystemContent::Text(text) if text == "You are helpful."));
    let tools_added = system.tools_added.as_ref().expect("toolsAdded");
    assert_eq!(
        tools_added
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        vec!["echo"]
    );
}

#[tokio::test]
async fn subscribes_to_events_without_initial_or_mutation_events() {
    let agent = Arc::new(Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        ..Default::default()
    }));

    let event_count = Arc::new(AtomicUsize::new(0));
    let counter = event_count.clone();
    let unsubscribe = agent.subscribe(move |_event, _signal| {
        let counter = counter.clone();
        Box::pin(async move {
            counter.fetch_add(1, Ordering::SeqCst);
        })
    });

    // No initial event on subscribe
    assert_eq!(event_count.load(Ordering::SeqCst), 0);

    // State mutators don't emit events
    agent.set_thinking_level(pi_agent::AgentThinkingLevel::Low);
    assert_eq!(event_count.load(Ordering::SeqCst), 0);
    assert_eq!(agent.thinking_level(), pi_agent::AgentThinkingLevel::Low);

    // Unsubscribe should work
    unsubscribe();
    agent.set_thinking_level(pi_agent::AgentThinkingLevel::High);
    assert_eq!(event_count.load(Ordering::SeqCst), 0); // Should not increase
}

#[tokio::test]
async fn emits_full_lifecycle_events_for_thrown_run_failures() {
    let agent = Arc::new(Agent::new(AgentOptions {
        stream_fn: Some(failing_stream_fn("provider exploded")),
        ..Default::default()
    }));
    let events: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    {
        let events = events.clone();
        agent.subscribe(move |event, _signal| {
            let events = events.clone();
            Box::pin(async move {
                events.lock().push(event.event_type());
            })
        });
    }

    agent
        .clone()
        .prompt("hello")
        .await
        .expect("failure is swallowed into the lifecycle");

    assert_eq!(
        *events.lock(),
        vec![
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );
    let last = agent.messages().last().cloned().expect("last message");
    assert_eq!(last.role(), "assistant");
    let AgentMessage::Llm(Message::Assistant(assistant)) = last else {
        panic!("expected assistant")
    };
    assert_eq!(assistant.stop_reason, StopReason::Error);
    assert_eq!(
        assistant.error_message.as_deref(),
        Some("provider exploded")
    );
    assert_eq!(agent.error_message().as_deref(), Some("provider exploded"));
}

#[tokio::test]
async fn awaits_async_subscribers_before_prompt_resolves() {
    let (barrier, _release) = notify_pair();
    let agent = Arc::new(Agent::new(AgentOptions {
        stream_fn: Some(done_stream_fn("ok")),
        ..Default::default()
    }));

    let listener_finished = Arc::new(AtomicBool::new(false));
    {
        let listener_finished = listener_finished.clone();
        let barrier = barrier.clone();
        agent.subscribe(move |event, _signal| {
            let listener_finished = listener_finished.clone();
            let barrier = barrier.clone();
            Box::pin(async move {
                if matches!(event, AgentEvent::AgentEnd { .. }) {
                    barrier.notified().await;
                    listener_finished.store(true, Ordering::SeqCst);
                }
            })
        });
    }

    let prompt_resolved = Arc::new(AtomicBool::new(false));
    {
        let prompt_resolved = prompt_resolved.clone();
        let agent = Arc::downgrade(&agent);
        tokio::spawn(async move {
            let Some(agent) = agent.upgrade() else { return };
            let _ = agent.prompt("hello").await;
            prompt_resolved.store(true, Ordering::SeqCst);
        });
    }

    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    assert!(!prompt_resolved.load(Ordering::SeqCst));
    assert!(!listener_finished.load(Ordering::SeqCst));
    assert!(agent.is_streaming());

    barrier.notify_one();
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    assert!(listener_finished.load(Ordering::SeqCst));
    assert!(prompt_resolved.load(Ordering::SeqCst));
    assert!(!agent.is_streaming());
}

#[tokio::test]
async fn wait_for_idle_waits_for_async_subscribers() {
    let barrier = Arc::new(tokio::sync::Notify::new());
    let agent = Arc::new(Agent::new(AgentOptions {
        stream_fn: Some(done_stream_fn("ok")),
        ..Default::default()
    }));

    {
        let barrier = barrier.clone();
        agent.subscribe(move |event, _signal| {
            let barrier = barrier.clone();
            Box::pin(async move {
                if let AgentEvent::MessageEnd { message } = &event {
                    if message.role() == "assistant" {
                        barrier.notified().await;
                    }
                }
            })
        });
    }

    let agent_weak = Arc::downgrade(&agent);
    let agent_for_prompt = agent.clone();
    let prompt_task = tokio::spawn(async move {
        agent_for_prompt.prompt("hello").await.ok();
    });
    let idle_resolved = Arc::new(AtomicBool::new(false));
    {
        let idle_resolved = idle_resolved.clone();
        let agent_weak = agent_weak.clone();
        tokio::spawn(async move {
            let Some(agent) = agent_weak.upgrade() else {
                return;
            };
            agent.wait_for_idle().await;
            idle_resolved.store(true, Ordering::SeqCst);
        });
    }

    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    assert!(!idle_resolved.load(Ordering::SeqCst));
    assert!(agent.is_streaming());

    barrier.notify_one();
    prompt_task.await.ok();
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    assert!(idle_resolved.load(Ordering::SeqCst));
    assert!(!agent.is_streaming());
}

#[tokio::test]
async fn passes_the_active_abort_signal_to_subscribers() {
    let received: Arc<Mutex<Option<pi_ai::utils::abort::AbortSignal>>> = Arc::new(Mutex::new(None));
    let received_for_listener = received.clone();
    let agent = Arc::new(Agent::new(AgentOptions {
        stream_fn: Some(abort_aware_stream_fn()),
        ..Default::default()
    }));

    {
        let received = received_for_listener.clone();
        agent.subscribe(move |event, signal| {
            let received = received.clone();
            Box::pin(async move {
                if matches!(event, AgentEvent::AgentStart) {
                    *received.lock() = Some(signal);
                }
            })
        });
    }

    let agent_for_prompt = agent.clone();
    let prompt_task = tokio::spawn(async move {
        agent_for_prompt.prompt("hello").await.ok();
    });
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    let signal = received.lock().clone().expect("signal received");
    assert!(!signal.aborted());

    agent.abort();
    prompt_task.await.ok();

    assert!(signal.aborted());
}

#[tokio::test]
async fn ignores_tool_updates_after_the_tool_execution_settles() {
    let delayed_update: Arc<Mutex<Option<pi_agent::AgentToolUpdateCallback>>> =
        Arc::new(Mutex::new(None));
    let events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));

    let delayed_for_tool = delayed_update.clone();
    let mut tool = AgentTool::new(
        "delayed_tool",
        "Captures progress callbacks",
        empty_object_schema(),
        "Delayed Tool",
        Arc::new(move |_id, _params, _signal, on_update| {
            let delayed_for_tool = delayed_for_tool.clone();
            Box::pin(async move {
                if let Some(on_update) = on_update {
                    *delayed_for_tool.lock() = Some(on_update.clone());
                    on_update(AgentToolResult::new(
                        vec![UserContent::Text(pi_ai::types::TextContent::new("running"))],
                        json!({ "status": "running" }),
                    ));
                }
                Ok(AgentToolResult {
                    content: vec![UserContent::Text(pi_ai::types::TextContent::new("ok"))],
                    details: json!({ "status": "done" }),
                    structured_content: None,
                    usage: None,
                    is_error: None,
                    terminate: Some(true),
                })
            })
        }),
    );
    tool.execution_mode = Some(pi_agent::ToolExecutionMode::Sequential);

    let agent = Arc::new(Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            tools: vec![tool],
            ..empty_initial_state()
        },
        stream_fn: Some(stream_fn(move |_model, _context, _options| {
            let mock = MockAssistantStream::new();
            mock.push_done(
                tool_use_message(vec![create_tool_call("call-1", "delayed_tool", json!({}))]),
                StopReason::ToolUse,
            );
            mock.stream
        })),
        ..Default::default()
    }));
    {
        let events = events.clone();
        agent.subscribe(move |event, _signal| {
            let events = events.clone();
            Box::pin(async move {
                events.lock().push(event);
            })
        });
    }

    agent.clone().prompt("run tool").await.expect("prompt");
    let event_count_after_prompt = events.lock().len();

    if let Some(on_update) = delayed_update.lock().clone() {
        on_update(AgentToolResult::new(
            vec![UserContent::Text(pi_ai::types::TextContent::new("late"))],
            json!({ "status": "late" }),
        ));
    }
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    assert_eq!(
        events
            .lock()
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolExecutionUpdate { .. }))
            .count(),
        1
    );
    assert_eq!(events.lock().len(), event_count_after_prompt);
}

#[tokio::test]
async fn ignores_a_settled_parallel_tool_update_while_another_tool_is_still_running() {
    let slow_started = Arc::new(tokio::sync::Notify::new());
    let settled_tool_ended = Arc::new(tokio::sync::Notify::new());
    let release_slow = Arc::new(tokio::sync::Notify::new());
    let settled_tool_update: Arc<Mutex<Option<pi_agent::AgentToolUpdateCallback>>> =
        Arc::new(Mutex::new(None));
    let events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));

    let settled_for_tool = settled_tool_update.clone();
    let settled_tool = AgentTool::new(
        "settled_tool",
        "Captures progress callbacks",
        empty_object_schema(),
        "Settled Tool",
        Arc::new(move |_id, _params, _signal, on_update| {
            let settled_for_tool = settled_for_tool.clone();
            Box::pin(async move {
                if let Some(on_update) = on_update {
                    *settled_for_tool.lock() = Some(on_update.clone());
                }
                Ok(AgentToolResult {
                    content: vec![UserContent::Text(pi_ai::types::TextContent::new("done"))],
                    details: json!({ "status": "done" }),
                    structured_content: None,
                    usage: None,
                    is_error: None,
                    terminate: Some(true),
                })
            })
        }),
    );
    let slow_started_for_tool = slow_started.clone();
    let release_slow_for_tool = release_slow.clone();
    let slow_tool = AgentTool::new(
        "slow_tool",
        "Keeps the agent run active",
        empty_object_schema(),
        "Slow Tool",
        Arc::new(move |_id, _params, _signal, _on_update| {
            let slow_started_for_tool = slow_started_for_tool.clone();
            let release_slow_for_tool = release_slow_for_tool.clone();
            Box::pin(async move {
                slow_started_for_tool.notify_one();
                release_slow_for_tool.notified().await;
                Ok(AgentToolResult {
                    content: vec![UserContent::Text(pi_ai::types::TextContent::new("done"))],
                    details: json!({ "status": "done" }),
                    structured_content: None,
                    usage: None,
                    is_error: None,
                    terminate: Some(true),
                })
            })
        }),
    );

    let agent = Arc::new(Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            tools: vec![settled_tool, slow_tool],
            ..empty_initial_state()
        },
        stream_fn: Some(stream_fn(move |_model, _context, _options| {
            let mock = MockAssistantStream::new();
            mock.push_done(
                tool_use_message(vec![
                    create_tool_call("call-1", "settled_tool", json!({})),
                    create_tool_call("call-2", "slow_tool", json!({})),
                ]),
                StopReason::ToolUse,
            );
            mock.stream
        })),
        ..Default::default()
    }));
    {
        let events = events.clone();
        let settled_tool_ended = settled_tool_ended.clone();
        agent.subscribe(move |event, _signal| {
            let events = events.clone();
            let settled_tool_ended = settled_tool_ended.clone();
            Box::pin(async move {
                if let AgentEvent::ToolExecutionEnd { tool_call_id, .. } = &event {
                    if tool_call_id == "call-1" {
                        settled_tool_ended.notify_one();
                    }
                }
                events.lock().push(event);
            })
        });
    }

    let agent_for_prompt = agent.clone();
    let prompt_task = tokio::spawn(async move {
        agent_for_prompt.prompt("run tools").await.expect("prompt");
    });

    slow_started.notified().await;
    settled_tool_ended.notified().await;
    let event_count_before_late_update = events.lock().len();

    if let Some(on_update) = settled_tool_update.lock().clone() {
        on_update(AgentToolResult::new(
            vec![UserContent::Text(pi_ai::types::TextContent::new("late"))],
            json!({ "status": "late" }),
        ));
    }
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    assert_eq!(events.lock().len(), event_count_before_late_update);

    release_slow.notify_one();
    prompt_task.await.expect("prompt task");

    assert_eq!(
        events
            .lock()
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolExecutionUpdate { .. }))
            .count(),
        0
    );
}

#[tokio::test]
async fn updates_state_with_mutators() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        ..Default::default()
    });

    // setModel
    let new_model = pi_ai::compat::get_model("google", "gemini-2.5-flash").expect("catalog model");
    agent.set_model(new_model.clone());
    assert_eq!(agent.model(), new_model);

    // setThinkingLevel
    agent.set_thinking_level(pi_agent::AgentThinkingLevel::High);
    assert_eq!(agent.thinking_level(), pi_agent::AgentThinkingLevel::High);

    // setTools (stored by copy upstream; owned here)
    agent.set_tools(vec![{
        let mut tool = create_tool("test");
        tool.tool.description = "test tool".into();
        tool
    }]);
    assert_eq!(agent.tools().len(), 1);

    // replaceMessages
    agent.set_messages(vec![AgentMessage::user(pi_ai::types::UserMessage {
        content: UserContentField::Text("Hello".into()),
        timestamp: pi_agent::now_ms(),
    })]);
    assert_eq!(agent.messages().len(), 1);

    // appendMessage
    agent
        .lock_state()
        .messages
        .push(AgentMessage::assistant(create_assistant_message_text("Hi")));
    assert_eq!(agent.messages().len(), 2);
    assert_eq!(agent.messages()[1].role(), "assistant");

    // clearMessages
    agent.set_messages(vec![]);
    assert!(agent.messages().is_empty());
}

#[tokio::test]
async fn supports_steering_message_queue() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        ..Default::default()
    });

    let message = AgentMessage::user(pi_ai::types::UserMessage {
        content: UserContentField::Text("Steering message".into()),
        timestamp: pi_agent::now_ms(),
    });
    agent.steer(message.clone());

    // The message is queued but not yet in state.messages
    assert!(!agent.messages().contains(&message));
}

#[tokio::test]
async fn supports_follow_up_message_queue() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        ..Default::default()
    });

    let message = AgentMessage::user(pi_ai::types::UserMessage {
        content: UserContentField::Text("Follow-up message".into()),
        timestamp: pi_agent::now_ms(),
    });
    agent.follow_up(message.clone());

    // The message is queued but not yet in state.messages
    assert!(!agent.messages().contains(&message));
}

#[tokio::test]
async fn handles_abort_controller_when_idle() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        ..Default::default()
    });

    // Should not throw even if nothing is running
    agent.abort();
}

#[tokio::test]
async fn rejects_reset_while_processing_without_corrupting_the_transcript() {
    let stream_started = Arc::new(tokio::sync::Notify::new());
    let release_response = Arc::new(tokio::sync::Notify::new());

    let stream_started_for_stream = stream_started.clone();
    let release_for_stream = release_response.clone();
    let agent = Arc::new(Agent::new(AgentOptions {
        stream_fn: Some(stream_fn(move |_model, _context, _options| {
            let mock = MockAssistantStream::new();
            let producer = mock.stream.clone();
            let stream_started = stream_started_for_stream.clone();
            let release = release_for_stream.clone();
            producer.push(AssistantMessageEvent::Start {
                partial: create_assistant_message_text(""),
            });
            tokio::spawn(async move {
                stream_started.notify_one();
                release.notified().await;
                producer.push(AssistantMessageEvent::Done {
                    reason: pi_ai::types::StreamDoneReason::Stop,
                    message: create_assistant_message_text("Done"),
                });
            });
            mock.stream
        })),
        ..Default::default()
    }));

    let agent_for_prompt = agent.clone();
    let prompt_task = tokio::spawn(async move { agent_for_prompt.prompt("Hello").await });
    stream_started.notified().await;

    assert!(agent.is_streaming());
    assert_eq!(
        agent
            .messages()
            .iter()
            .map(|message| message.role())
            .collect::<Vec<_>>(),
        vec!["user"]
    );
    let error = agent.reset().expect_err("reset while processing");
    assert_eq!(
        error.0,
        "Agent is already processing. Wait for completion before resetting."
    );
    assert!(agent.is_streaming());
    assert_eq!(
        agent
            .messages()
            .iter()
            .map(|message| message.role())
            .collect::<Vec<_>>(),
        vec!["user"]
    );

    release_response.notify_one();
    prompt_task
        .await
        .expect("prompt task")
        .expect("prompt succeeds");

    assert!(!agent.is_streaming());
    assert_eq!(
        agent
            .messages()
            .iter()
            .map(|message| message.role())
            .collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
}

#[tokio::test]
async fn throws_when_prompt_called_while_streaming() {
    let agent = Arc::new(Agent::new(AgentOptions {
        stream_fn: Some(abort_aware_stream_fn()),
        ..Default::default()
    }));

    let agent_for_prompt = agent.clone();
    let first_prompt = tokio::spawn(async move { agent_for_prompt.prompt("First message").await });

    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    assert!(agent.is_streaming());

    let error = agent
        .clone()
        .prompt("Second message")
        .await
        .expect_err("second prompt rejects");
    assert_eq!(
		error.0,
		"Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
	);

    agent.abort();
    let inner = first_prompt.await.expect("join prompt task");
    assert!(inner.is_ok(), "aborted runs still settle like upstream");
}

#[tokio::test]
async fn throws_when_continue_called_while_streaming() {
    let agent = Arc::new(Agent::new(AgentOptions {
        stream_fn: Some(abort_aware_stream_fn()),
        ..Default::default()
    }));

    let agent_for_prompt = agent.clone();
    let first_prompt = tokio::spawn(async move { agent_for_prompt.prompt("First message").await });
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    assert!(agent.is_streaming());

    let error = agent
        .clone()
        .continue_()
        .await
        .expect_err("continue rejects");
    assert_eq!(
        error.0,
        "Agent is already processing. Wait for completion before continuing."
    );

    agent.abort();
    let inner = first_prompt.await.expect("join prompt task");
    assert!(inner.is_ok(), "aborted runs still settle like upstream");
}

#[tokio::test]
async fn continue_processes_queued_follow_up_messages_after_an_assistant_turn() {
    let agent = Arc::new(Agent::new(AgentOptions {
        stream_fn: Some(done_stream_fn("Processed")),
        ..Default::default()
    }));

    agent.set_messages(vec![
        AgentMessage::user(pi_ai::types::UserMessage {
            content: UserContentField::Blocks(vec![UserContent::Text(
                pi_ai::types::TextContent::new("Initial"),
            )]),
            timestamp: pi_agent::now_ms() - 10,
        }),
        AgentMessage::assistant(create_assistant_message_text("Initial response")),
    ]);

    agent.follow_up(AgentMessage::user(pi_ai::types::UserMessage {
        content: UserContentField::Blocks(vec![UserContent::Text(pi_ai::types::TextContent::new(
            "Queued follow-up",
        ))]),
        timestamp: pi_agent::now_ms(),
    }));

    agent.clone().continue_().await.expect("continue succeeds");

    let has_queued_follow_up = agent.messages().iter().any(|message| match message {
        AgentMessage::Llm(Message::User(user)) => match &user.content {
            UserContentField::Text(text) => text == "Queued follow-up",
            UserContentField::Blocks(blocks) => blocks.iter().any(
                |block| matches!(block, UserContent::Text(text) if text.text == "Queued follow-up"),
            ),
        },
        _ => false,
    });
    assert!(has_queued_follow_up);
    assert_eq!(
        agent.messages().last().map(|message| message.role()),
        Some("assistant")
    );
}

#[tokio::test]
async fn continue_keeps_steering_semantics_for_assistant_tail_fallback() {
    for (mode, expected_requests) in [
        (pi_agent::QueueMode::OneAtATime, 2usize),
        (pi_agent::QueueMode::All, 1),
    ] {
        let requests: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let requests_for_stream = requests.clone();
        let agent = Arc::new(Agent::new(AgentOptions {
            steering_mode: Some(mode),
            stream_fn: Some(stream_fn(move |_model, context, _options| {
                requests_for_stream.lock().push(user_texts(&context));
                let mock = MockAssistantStream::new();
                mock.push_done(create_assistant_message_text("Processed"), StopReason::Stop);
                mock.stream
            })),
            ..Default::default()
        }));
        agent.set_messages(vec![
            AgentMessage::user(create_user_message("Initial")),
            AgentMessage::assistant(create_assistant_message_text("Initial response")),
        ]);
        agent.steer(AgentMessage::user(create_user_message("Steering 1")));
        agent.steer(AgentMessage::user(create_user_message("Steering 2")));

        agent.clone().continue_().await.expect("continue succeeds");

        let requests = requests.lock();
        assert_eq!(requests.len(), expected_requests);
        assert!(requests[0].contains(&"Steering 1".to_string()));
        if mode == pi_agent::QueueMode::OneAtATime {
            assert!(!requests[0].contains(&"Steering 2".to_string()));
            assert!(requests[1].contains(&"Steering 2".to_string()));
        } else {
            assert!(requests[0].contains(&"Steering 2".to_string()));
        }
    }
}

#[tokio::test]
async fn keeps_legacy_prepare_next_turn_signal_callback_behavior() {
    let tool = create_tool("noop");
    let request_count = Arc::new(AtomicUsize::new(0));
    let saw_abort_signal = Arc::new(AtomicBool::new(false));

    let saw_for_hook = saw_abort_signal.clone();
    let request_count_for_stream = request_count.clone();
    let agent = Arc::new(Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            tools: vec![tool],
            ..empty_initial_state()
        },
        prepare_next_turn: Some(prepare_next_turn_with_context(move |context, signal| {
            let _ = context;
            let saw_for_hook = saw_for_hook.clone();
            Box::pin(async move {
                // Signal-only variant: the hook receives the active abort signal.
                saw_for_hook.store(signal.is_some(), Ordering::SeqCst);
                None
            })
        })),
        stream_fn: Some(stream_fn(move |_model, _context, _options| {
            let mock = MockAssistantStream::new();
            let count = request_count_for_stream.fetch_add(1, Ordering::SeqCst) + 1;
            if count == 1 {
                mock.push_done(
                    tool_use_message(vec![create_tool_call("tool-1", "noop", json!({}))]),
                    StopReason::ToolUse,
                );
            } else {
                mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
            }
            mock.stream
        })),
        ..Default::default()
    }));

    agent.clone().prompt("start").await.expect("prompt");

    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    assert!(saw_abort_signal.load(Ordering::SeqCst));
}

#[tokio::test]
async fn forwards_finish_turn_through_agent_options_with_the_active_abort_signal() {
    let tool = create_tool("noop");
    let request_count = Arc::new(AtomicUsize::new(0));
    let saw_abort_signal = Arc::new(AtomicBool::new(false));
    let callback_context_roles: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let saw_for_hook = saw_abort_signal.clone();
    let roles_for_hook = callback_context_roles.clone();
    let request_count_for_stream = request_count.clone();
    let agent = Arc::new(Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            tools: vec![tool],
            ..empty_initial_state()
        },
        finish_turn: Some(finish_turn(move |context, signal| {
            let saw_for_hook = saw_for_hook.clone();
            let roles_for_hook = roles_for_hook.clone();
            Box::pin(async move {
                saw_for_hook.store(signal.is_some(), Ordering::SeqCst);
                *roles_for_hook.lock() = context
                    .context
                    .messages
                    .iter()
                    .map(|message| message.role().to_string())
                    .collect();
                Some(pi_agent::AgentTurnDecision::End)
            })
        })),
        stream_fn: Some(stream_fn(move |_model, _context, _options| {
            let mock = MockAssistantStream::new();
            let count = request_count_for_stream.fetch_add(1, Ordering::SeqCst) + 1;
            if count == 1 {
                mock.push_done(
                    tool_use_message(vec![create_tool_call("tool-1", "noop", json!({}))]),
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
        ..Default::default()
    }));

    agent.clone().prompt("start").await.expect("prompt");

    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    assert!(saw_abort_signal.load(Ordering::SeqCst));
    assert_eq!(
        *callback_context_roles.lock(),
        vec!["system", "user", "assistant", "toolResult"]
    );
}

#[tokio::test]
async fn rejects_a_queued_continuation_from_empty_or_system_only_context() {
    for messages in [
        vec![],
        vec![AgentMessage::system(SystemMessage {
            content: SystemContent::Text("system only".into()),
            sections: None,
            tools_added: None,
            tools_removed: None,
            timestamp: 1,
        })],
    ] {
        let agent = Arc::new(Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                messages,
                ..empty_initial_state()
            },
            stream_fn: Some(unused_stream_fn()),
            ..Default::default()
        }));
        let steering = AgentMessage::user(create_user_message("steering"));
        let follow_up = AgentMessage::user(create_user_message("follow-up"));
        agent.steer(steering.clone());
        agent.follow_up(follow_up.clone());

        let error = agent
            .clone()
            .continue_()
            .await
            .expect_err("continue rejects");
        assert_eq!(error.0, "No messages to continue from");
        assert_eq!(agent.peek_queued_messages(), vec![steering.clone()]);
        agent.clear_steering_queue();
        assert_eq!(agent.peek_queued_messages(), vec![follow_up]);
    }
}

#[tokio::test]
async fn defers_follow_up_input_on_the_first_continuation_request() {
    for messages in [
        vec![AgentMessage::user(create_user_message("existing user"))],
        vec![
            AgentMessage::user(create_user_message("existing user")),
            AgentMessage::assistant(tool_use_message(vec![create_tool_call(
                "call-1",
                "noop",
                json!({}),
            )])),
            AgentMessage::tool_result(pi_ai::types::ToolResultMessage {
                tool_call_id: "call-1".into(),
                tool_name: "noop".into(),
                content: vec![UserContent::Text(pi_ai::types::TextContent::new("done"))],
                details: None,
                usage: None,
                nested_calls: None,
                is_error: false,
                duration_ms: None,
                timestamp: 1,
            }),
        ],
    ] {
        let requests: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let requests_for_stream = requests.clone();
        let agent = Arc::new(Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                messages,
                ..empty_initial_state()
            },
            stream_fn: Some(stream_fn(move |_model, context, _options| {
                requests_for_stream.lock().push(user_texts(&context));
                let mock = MockAssistantStream::new();
                mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
                mock.stream
            })),
            ..Default::default()
        }));
        agent.follow_up(AgentMessage::user(create_user_message("follow-up")));

        agent.clone().continue_().await.expect("continue succeeds");

        let requests = requests.lock();
        assert_eq!(requests.len(), 2);
        assert!(!requests[0].contains(&"follow-up".to_string()));
        assert!(requests[1].contains(&"follow-up".to_string()));
    }
}

#[tokio::test]
async fn polls_steering_at_continuation_startup() {
    for (mode, expected_requests) in [
        (pi_agent::QueueMode::OneAtATime, 2usize),
        (pi_agent::QueueMode::All, 1),
    ] {
        let requests: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let requests_for_stream = requests.clone();
        let agent = Arc::new(Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                messages: vec![AgentMessage::user(create_user_message("existing"))],
                ..empty_initial_state()
            },
            steering_mode: Some(mode),
            stream_fn: Some(stream_fn(move |_model, context, _options| {
                requests_for_stream.lock().push(user_texts(&context));
                let mock = MockAssistantStream::new();
                mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
                mock.stream
            })),
            ..Default::default()
        }));
        agent.steer(AgentMessage::user(create_user_message("first")));
        agent.steer(AgentMessage::user(create_user_message("second")));

        agent.clone().continue_().await.expect("continue succeeds");

        let requests = requests.lock();
        assert_eq!(requests.len(), expected_requests);
        assert!(requests[0].contains(&"first".to_string()));
        if mode == pi_agent::QueueMode::OneAtATime {
            assert!(!requests[0].contains(&"second".to_string()));
            assert!(requests[1].contains(&"second".to_string()));
        } else {
            assert!(requests[0].contains(&"second".to_string()));
        }
    }
}

#[tokio::test]
async fn keeps_steering_ahead_of_follow_up_from_a_non_assistant_continuation_tail() {
    let requests: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let requests_for_stream = requests.clone();
    let agent = Arc::new(Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            messages: vec![AgentMessage::user(create_user_message("existing"))],
            ..empty_initial_state()
        },
        stream_fn: Some(stream_fn(move |_model, context, _options| {
            requests_for_stream.lock().push(user_texts(&context));
            let mock = MockAssistantStream::new();
            mock.push_done(create_assistant_message_text("done"), StopReason::Stop);
            mock.stream
        })),
        ..Default::default()
    }));
    agent.steer(AgentMessage::user(create_user_message("steering")));
    agent.follow_up(AgentMessage::user(create_user_message("follow-up")));

    agent.clone().continue_().await.expect("continue succeeds");

    let requests = requests.lock();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].contains(&"steering".to_string()));
    assert!(!requests[0].contains(&"follow-up".to_string()));
    assert!(requests[1].contains(&"follow-up".to_string()));
}

#[tokio::test]
async fn keeps_queues_on_error_and_aborted_responses_even_when_finish_turn_continues() {
    for stop_reason in [StopReason::Error, StopReason::Aborted] {
        let queued_during_response = AgentMessage::user(create_user_message("steering"));
        let follow_up = AgentMessage::user(create_user_message("follow-up"));
        let queued_for_listener = queued_during_response.clone();
        let agent = Arc::new(Agent::new(AgentOptions {
            finish_turn: Some(finish_turn(|_context, _signal| {
                Box::pin(async { Some(pi_agent::AgentTurnDecision::Continue) })
            })),
            stream_fn: Some(stream_fn(move |_model, _context, _options| {
                let mock = MockAssistantStream::new();
                let mut error = create_assistant_message_text(stop_reason_text(stop_reason));
                error.stop_reason = stop_reason;
                error.error_message = Some(stop_reason_text(stop_reason).to_string());
                mock.push_error(
                    error,
                    match stop_reason {
                        StopReason::Aborted => StreamErrorReason::Aborted,
                        _ => StreamErrorReason::Error,
                    },
                );
                mock.stream
            })),
            ..Default::default()
        }));
        agent.follow_up(follow_up.clone());
        {
            let queued_for_listener = queued_for_listener.clone();
            let agent_weak = Arc::downgrade(&agent);
            agent.subscribe(move |event, _signal| {
                let queued_for_listener = queued_for_listener.clone();
                let agent_weak = agent_weak.clone();
                Box::pin(async move {
                    if let AgentEvent::MessageEnd { message } = &event {
                        if message.role() == "assistant" {
                            if let Some(agent) = agent_weak.upgrade() {
                                agent.steer(queued_for_listener.clone());
                            }
                        }
                    }
                })
            });
        }

        agent.clone().prompt("start").await.expect("prompt");

        assert_eq!(
            agent.peek_queued_messages(),
            vec![queued_during_response.clone()]
        );
        agent.clear_steering_queue();
        assert_eq!(agent.peek_queued_messages(), vec![follow_up.clone()]);
    }
}

fn stop_reason_text(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Error => "error",
        StopReason::Aborted => "aborted",
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn keeps_queues_when_finish_turn_ends_the_run() {
    let queued_during_response = AgentMessage::user(create_user_message("steering"));
    let follow_up = AgentMessage::user(create_user_message("follow-up"));
    let queued_for_listener = queued_during_response.clone();
    let agent = Arc::new(Agent::new(AgentOptions {
        finish_turn: Some(finish_turn(|_context, _signal| {
            Box::pin(async { Some(pi_agent::AgentTurnDecision::End) })
        })),
        stream_fn: Some(done_stream_fn("done")),
        ..Default::default()
    }));
    agent.follow_up(follow_up.clone());
    {
        let queued_for_listener = queued_for_listener.clone();
        let agent_weak = Arc::downgrade(&agent);
        agent.subscribe(move |event, _signal| {
            let queued_for_listener = queued_for_listener.clone();
            let agent_weak = agent_weak.clone();
            Box::pin(async move {
                if let AgentEvent::MessageEnd { message } = &event {
                    if message.role() == "assistant" {
                        if let Some(agent) = agent_weak.upgrade() {
                            agent.steer(queued_for_listener.clone());
                        }
                    }
                }
            })
        });
    }

    agent.clone().prompt("start").await.expect("prompt");

    assert_eq!(
        agent.peek_queued_messages(),
        vec![queued_during_response.clone()]
    );
    agent.clear_steering_queue();
    assert_eq!(agent.peek_queued_messages(), vec![follow_up.clone()]);
}

#[tokio::test]
async fn previews_the_next_selected_queued_messages_without_consuming_them() {
    let agent = Agent::new(AgentOptions {
        steering_mode: Some(pi_agent::QueueMode::OneAtATime),
        follow_up_mode: Some(pi_agent::QueueMode::All),
        stream_fn: Some(stream_fn(|_model, _context, _options| {
            MockAssistantStream::new().stream
        })),
        ..Default::default()
    });
    let first = AgentMessage::user(create_user_message("first steering"));
    let second = AgentMessage::user(create_user_message("second steering"));
    let follow_up = AgentMessage::user(create_user_message("follow-up"));
    agent.steer(first.clone());
    agent.steer(second.clone());
    agent.follow_up(follow_up.clone());

    assert_eq!(agent.peek_queued_messages(), vec![first.clone()]);
    assert_eq!(agent.peek_queued_messages(), vec![first.clone()]);
    agent.clear_steering_queue();
    assert_eq!(agent.peek_queued_messages(), vec![follow_up.clone()]);
}

#[tokio::test]
async fn forwards_provider_stream_event_observers_through_agent_options() {
    let provider_events: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let provider_events_for_hook = provider_events.clone();
    let agent = Arc::new(Agent::new(AgentOptions {
        on_provider_stream_event: Some(Arc::new(move |data, _model| {
            let provider_events_for_hook = provider_events_for_hook.clone();
            Box::pin(async move {
                provider_events_for_hook.lock().push(data);
            })
        })),
        stream_fn: Some(stream_fn(move |model, _context, options| {
            let mock = MockAssistantStream::new();
            let producer = mock.stream.clone();
            let hook = options.and_then(|options| options.base.on_provider_stream_event.clone());
            let model = model.clone();
            tokio::spawn(async move {
                if let Some(hook) = hook {
                    hook(json!({ "request_cost": 0.01 }), model).await;
                }
                producer.push(AssistantMessageEvent::Done {
                    reason: pi_ai::types::StreamDoneReason::Stop,
                    message: create_assistant_message_text("ok"),
                });
            });
            mock.stream
        })),
        ..Default::default()
    }));

    agent.clone().prompt("hello").await.expect("prompt");

    assert_eq!(
        *provider_events.lock(),
        vec![json!({ "request_cost": 0.01 })]
    );
}

#[tokio::test]
async fn forwards_session_id_to_stream_function_options() {
    let received_session_id: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let received_for_stream = received_session_id.clone();
    let agent = Arc::new(Agent::new(AgentOptions {
        session_id: Some("session-abc".into()),
        stream_fn: Some(stream_fn(move |_model, _context, options| {
            *received_for_stream.lock() =
                options.and_then(|options| options.base.session_id.clone());
            let mock = MockAssistantStream::new();
            mock.push_done(create_assistant_message_text("ok"), StopReason::Stop);
            mock.stream
        })),
        ..Default::default()
    }));

    agent.clone().prompt("hello").await.expect("prompt");
    assert_eq!(received_session_id.lock().as_deref(), Some("session-abc"));

    // Test setter
    agent.set_session_id(Some("session-def".into()));
    assert_eq!(agent.session_id().as_deref(), Some("session-def"));

    agent.clone().prompt("hello again").await.expect("prompt");
    assert_eq!(received_session_id.lock().as_deref(), Some("session-def"));
}

// Silence unused import lint for the hook constructor re-export set.
#[allow(unused)]
fn _assert_types(_: Option<after_tool_call_marker>, _: Option<finish_turn_marker>) {}
type after_tool_call_marker = ();
type finish_turn_marker = ();
