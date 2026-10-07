use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use pi_ai::models::MutableModels;
use pi_ai::providers::faux::{
    faux_assistant_message, faux_text, faux_thinking, faux_tool_call, faux_provider, FauxContentBlock,
    FauxDeferredOptions, FauxMessageOptions, FauxResponseStep, FauxTokenSize, RegisterFauxProviderOptions,
};
use pi_ai::types::{AssistantContent, AssistantMessage, Message, StopReason, UserContentField};
use serde_json::{json, Value};
use tokio::task::JoinHandle;

use super::support::context;
use crate::harness::registry::create_registry;
use crate::harness::events::{watch_events, AgentEvent, AgentEventStream, MessageChange, ToolOutputChange};
use crate::harness::harness::{open_harness, Conversation, Harness};
use crate::harness::types::{
    AgentChange, ConversationCreateOptions, ConversationId, ConversationStreamOptions, Extension,
    HarnessOptions, HarnessSettings, InputSubmissionDraft, ModelRef, ProgressPolicyOverrides,
    RetryPolicyOverrides, SetOrClear, Submission, SubmissionDraft, SubmissionStatus, ToolDiagnostic,
    ToolDiagnosticSeverity, ToolExecutionMode, ToolExecutionResult, ToolOutputLimits, ToolRegistration,
    WhenBusy,
};
use crate::storage::memory::MemoryStorage;
use crate::types::{CommitChange, DocumentCommitChange, EntryDraft};

struct Fixture {
    harness: Arc<Harness>,
    root: Arc<Conversation>,
    faux: pi_ai::providers::faux::FauxProviderHandle,
}

async fn fixture(
    responses: Vec<FauxResponseStep>,
    provider_options: RegisterFauxProviderOptions,
    settings: HarnessSettings,
    extensions: Vec<Extension>,
) -> Fixture {
    let faux = faux_provider(provider_options);
    faux.set_responses(responses);
    let models = pi_ai::create_models(None);
    models.set_provider(faux.provider.clone());
    let registry = Arc::new(create_registry());
    for extension in extensions {
        crate::harness::types::Registry::install(registry.as_ref(), extension);
    }
    let harness = open_harness(
        Arc::new(MemoryStorage::new()),
        HarnessOptions {
            models,
            registry: registry.clone(),
            settings: Some(settings),
            ..HarnessOptions::default()
        },
        &context(),
    )
    .await
    .expect("open Faux harness");
    let root = Arc::new(
        harness
            .root(&context(), Some(Default::default()))
            .await
            .expect("create root conversation"),
    );
    root.configure(
        AgentChange {
            model: Some(SetOrClear::Set(ModelRef { provider: "faux".into(), model_id: "faux-1".into() })),
            ..Default::default()
        },
        &context(),
    )
    .await
    .expect("select Faux model");
    Fixture { harness, root, faux }
}

fn default_fixture(responses: Vec<FauxResponseStep>, extensions: Vec<Extension>) -> impl Future<Output = Fixture> {
    fixture(responses, RegisterFauxProviderOptions::default(), HarnessSettings::default(), extensions)
}

struct Recorder {
    batches: Arc<Mutex<Vec<Vec<AgentEvent>>>>,
    join: Mutex<Option<JoinHandle<()>>>,
}

impl Recorder {
    async fn attach(harness: &Arc<Harness>, conversation_id: ConversationId) -> Self {
        let stream = watch_events(harness, conversation_id, &context()).await.expect("attach event stream");
        Self::from_stream(stream)
    }

    fn from_stream(mut stream: AgentEventStream) -> Self {
        let batches = Arc::new(Mutex::new(Vec::new()));
        let delivered = batches.clone();
        let join = tokio::spawn(async move {
            while let Some(result) = stream.next().await {
                match result {
                    Ok(batch) => delivered.lock().push(batch),
                    Err(_) => break,
                }
            }
        });
        Self { batches, join: Mutex::new(Some(join)) }
    }

    fn batches(&self) -> Vec<Vec<AgentEvent>> {
        self.batches.lock().clone()
    }

    fn events(&self) -> Vec<AgentEvent> {
        self.batches().into_iter().flatten().collect()
    }

    async fn wait_until(&self, predicate: impl Fn(&[AgentEvent]) -> bool) {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let events = self.events();
                if predicate(&events) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("expected lifecycle event was not delivered");
    }

    async fn close(&self, harness: &Arc<Harness>) {
        harness.close(&context()).await.expect("close harness");
        let join = self.join.lock().take().expect("event listener task");
        join.await.expect("event listener");
    }
}

fn event_name(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::Snapshot { .. } => "snapshot",
        AgentEvent::RunStart { .. } => "run_start",
        AgentEvent::RunEnd { .. } => "run_end",
        AgentEvent::TurnStart => "turn_start",
        AgentEvent::TurnEnd => "turn_end",
        AgentEvent::MessageStart { .. } => "message_start",
        AgentEvent::MessageUpdate { .. } => "message_update",
        AgentEvent::MessageEnd { .. } => "message_end",
        AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
        AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
        AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
        AgentEvent::InboxUpdate { .. } => "inbox_update",
        AgentEvent::Submission { .. } => "submission",
        AgentEvent::AutoRetryStart { .. } => "auto_retry_start",
        AgentEvent::AutoRetryEnd { .. } => "auto_retry_end",
        AgentEvent::DeferredPoll { .. } => "deferred_poll",
        AgentEvent::EntryAppended { .. } => "entry_appended",
        AgentEvent::AgentChanged { .. } => "agent_changed",
        AgentEvent::UsageChanged { .. } => "usage_changed",
        AgentEvent::TaskFailed { .. } => "task_failed",
        AgentEvent::CompactionStart { .. } => "compaction_start",
        AgentEvent::CompactionEnd { .. } => "compaction_end",
    }
}

fn input(text: &str, when_busy: Option<WhenBusy>) -> SubmissionDraft {
    SubmissionDraft::Input(InputSubmissionDraft {
        request_id: None,
        content: UserContentField::Text(text.to_owned()),
        when_busy,
    })
}

fn text_of(message: &AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

fn streamed_text(events: &[AgentEvent]) -> String {
    let mut text = String::new();
    for event in events {
        if let AgentEvent::MessageStart { message: Message::Assistant(message) } = event {
            text = message.content.first().and_then(|block| match block {
                AssistantContent::Text(block) => Some(block.text.clone()),
                _ => None,
            }).unwrap_or_default();
        }
        if let AgentEvent::MessageUpdate { changes, .. } = event {
            for change in changes {
                match change {
                    MessageChange::TextStart { content_index: 0, block } => {
                        text = block.get("text").and_then(Value::as_str).unwrap_or_default().to_owned();
                    }
                    MessageChange::TextDelta { content_index: 0, delta } => text.push_str(delta),
                    _ => {}
                }
            }
        }
    }
    text
}

fn apply_changes(message: &AssistantMessage, changes: &[MessageChange]) -> AssistantMessage {
    let mut next = serde_json::to_value(message).expect("assistant message serializes");
    for change in changes {
        match change {
            MessageChange::Message { message } => {
                next = serde_json::to_value(message).expect("replacement message serializes");
            }
            MessageChange::Block { content_index, block } => {
                next["content"].as_array_mut().expect("content array")[*content_index as usize] = block.clone();
            }
            MessageChange::TextStart { content_index, block }
            | MessageChange::ThinkingStart { content_index, block }
            | MessageChange::ToolcallStart { content_index, block } => {
                next["content"].as_array_mut().expect("content array").insert(*content_index as usize, block.clone());
            }
            MessageChange::TextDelta { content_index, delta } => {
                append_json_string(&mut next["content"][*content_index as usize]["text"], delta);
            }
            MessageChange::ThinkingDelta { content_index, delta } => {
                append_json_string(&mut next["content"][*content_index as usize]["thinking"], delta);
            }
            MessageChange::ToolcallDelta { content_index, path, delta } => {
                let mut target = &mut next["content"][*content_index as usize]["arguments"];
                for segment in &path[..path.len() - 1] {
                    target = path_segment_mut(target, segment);
                }
                let leaf = path_segment_mut(target, path.last().expect("tool argument path"));
                append_json_string(leaf, delta);
            }
        }
    }
    serde_json::from_value(next).expect("updated assistant message deserializes")
}

fn append_json_string(target: &mut Value, delta: &str) {
    let prior = target.as_str().unwrap_or_default();
    *target = Value::String(format!("{prior}{delta}"));
}

fn path_segment_mut<'a>(target: &'a mut Value, segment: &Value) -> &'a mut Value {
    match segment {
        Value::String(key) => target.as_object_mut().expect("tool arguments object").get_mut(key).expect("tool argument key"),
        Value::Number(index) => &mut target.as_array_mut().expect("tool arguments array")[index.as_u64().expect("nonnegative path index") as usize],
        _ => panic!("tool argument path segment is a string or index: {segment:?}"),
    }
}

fn collect_partials(harness: &Harness) -> Arc<Mutex<Vec<AssistantMessage>>> {
    let partials = Arc::new(Mutex::new(Vec::new()));
    let recorded = partials.clone();
    let _subscription = harness.session.subscribe_commits(Arc::new(move |publication, _context| {
        for change in &publication.changes {
            let CommitChange::Document(DocumentCommitChange::Document {
                record,
                value: Some(value),
                ..
            }) = change else {
                continue;
            };
            if record.kind != "pi.live" {
                continue;
            }
            let Ok(live) = serde_json::from_value::<crate::harness::live::LiveState>(Value::Object(value.clone())) else {
                continue;
            };
            let Some(message) = live.generation.and_then(|generation| generation.message) else {
                continue;
            };
            if let Ok(message) = serde_json::from_value::<AssistantMessage>(message) {
                recorded.lock().push(message);
            }
        }
    }));
    partials
}
fn collect_tool_outputs(harness: &Harness) -> Arc<Mutex<Vec<String>>> {
    let outputs = Arc::new(Mutex::new(Vec::new()));
    let recorded = outputs.clone();
    let _subscription = harness.session.subscribe_commits(Arc::new(move |publication, _context| {
        for change in &publication.changes {
            let CommitChange::Document(DocumentCommitChange::Document {
                record,
                value: Some(value),
                ..
            }) = change else {
                continue;
            };
            if record.kind != "pi.live" {
                continue;
            }
            let Ok(live) = serde_json::from_value::<crate::harness::live::LiveState>(Value::Object(value.clone())) else {
                continue;
            };
            let Some(output) = live.tools.as_ref().and_then(|tools| tools.first()).and_then(|tool| tool.output.as_ref()) else {
                continue;
            };
            let mut recorded = recorded.lock();
            if recorded.last() != Some(output) {
                recorded.push(output.clone());
            }
        }
    }));
    outputs
}

fn tool_extension(name: &str, execute: crate::harness::types::ToolExecute) -> Extension {
    Extension::new("events-tools").with_tools(vec![Arc::new(ToolRegistration {
        tool: pi_ai::Tool {
            name: name.into(),
            description: format!("{name} tool"),
            parameters: json!({"type":"object","properties":{"n":{"type":"number"}}}),
            constrained_sampling: None,
        },
        replay: None,
        execution_mode: None,
        prepare_arguments: None,
        output_limits: None,
        execute,
    })])
}

fn fixed_tool_call(name: &str, call_id: &str) -> FauxResponseStep {
    FauxResponseStep::Message(faux_assistant_message(
        FauxContentBlock::from(faux_tool_call(name, json!({"n":1}).as_object().unwrap().clone(), Some(call_id))),
        FauxMessageOptions { stop_reason: Some(StopReason::ToolUse), ..Default::default() },
    ))
}

fn held_response(release: tokio::sync::oneshot::Receiver<()>, message: AssistantMessage) -> FauxResponseStep {
    let release = Arc::new(Mutex::new(Some(release)));
    FauxResponseStep::Factory(Arc::new(move |_transcript, _options, _state, _model| {
        let release = release.lock().take().expect("held Faux response requested once");
        let message = message.clone();
        Box::pin(async move {
            let _ = release.await;
            Ok(message)
        })
    }))
}

async fn live_state(fixture: &Fixture) -> crate::harness::live::LiveState {
    let value = fixture
        .harness
        .session
        .snapshot(&crate::harness::live::live_doc(), Some(fixture.root.id.0), None, &context())
        .await
        .expect("read live document")
        .expect("live document exists");
    serde_json::from_value(Value::Object((*value).clone())).expect("decode live state")
}

#[tokio::test]
async fn faux_run_streams_lifecycle_order_and_reconstructs_committed_text_deltas() {
    let answer = "streamed answer text ".repeat(20);
    let response = faux_assistant_message(answer.clone(), FauxMessageOptions::default());
    let fixture = fixture(
        vec![FauxResponseStep::Message(response)],
        RegisterFauxProviderOptions {
            tokens_per_second: Some(400.0),
            token_size: Some(FauxTokenSize { min: Some(1), max: Some(1) }),
            ..Default::default()
        },
        HarnessSettings::default(),
        vec![],
    )
    .await;
    let partials = collect_partials(&fixture.harness);
    let recorder = Recorder::attach(&fixture.harness, fixture.root.id).await;
    let submission = fixture.root.submit(input("hi", None), &context()).await.expect("submit input");
    assert_eq!(submission.wait(&context()).await.expect("wait for run").status, SubmissionStatus::Done);
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::RunEnd { .. }))).await;
    recorder.close(&fixture.harness).await;

    let events = recorder.events();
    let run = events.iter().filter(|event| !matches!(event, AgentEvent::Snapshot { .. })).collect::<Vec<_>>();
    let names = run.iter().map(|event| event_name(event)).collect::<Vec<_>>();
    let compacted = names.iter().enumerate().filter_map(|(index, name)| {
        if index == 0 || names[index - 1] != *name { Some(*name) } else { None }
    }).collect::<Vec<_>>();
    assert_eq!(compacted, [
        "message_start", "message_end", "submission", "run_start", "turn_start",
        "message_start", "message_update", "message_end", "turn_end", "run_end",
        "submission", "usage_changed",
    ]);
    assert!(matches!(run.iter().find(|event| matches!(event, AgentEvent::RunStart { .. })), Some(AgentEvent::RunStart { inputs }) if inputs == &[submission.id]));

    let partials = partials.lock().clone();
    let rebuilt = events.iter().enumerate().filter_map(|(index, event)| {
        (matches!(event, AgentEvent::MessageStart { message: Message::Assistant(_) })
            || matches!(event, AgentEvent::MessageUpdate { .. }))
        .then(|| streamed_text(&events[..=index]))
    }).collect::<Vec<_>>();
    assert_eq!(rebuilt, partials.iter().map(text_of).collect::<Vec<_>>());
    assert!(partials.len() > 1, "Faux token pacing must commit more than one partial");
    let final_message = events.iter().filter_map(|event| match event {
        AgentEvent::MessageEnd { entry } if entry.kind == "pi.assistant" => entry.model.as_ref(),
        _ => None,
    }).flatten().find_map(|message| match message {
        Message::Assistant(message) => Some(message),
        _ => None,
    }).expect("assistant message end carries its converted entry");
    assert_eq!(text_of(final_message), answer);
}

#[tokio::test]
async fn faux_stream_rebuilds_thinking_text_and_tool_argument_partials() {
    let response = faux_assistant_message(
        vec![
            FauxContentBlock::from(faux_thinking("thinking about it ".repeat(10))),
            FauxContentBlock::from(faux_text("some text ".repeat(10))),
            FauxContentBlock::from(faux_tool_call(
                "missing",
                json!({"path":"a/long/path/".repeat(10),"note":"x".repeat(60)}).as_object().unwrap().clone(),
                Some("call-1"),
            )),
        ],
        FauxMessageOptions { stop_reason: Some(StopReason::ToolUse), ..Default::default() },
    );
    let fixture = fixture(
        vec![FauxResponseStep::Message(response), FauxResponseStep::Message(faux_assistant_message("done", FauxMessageOptions::default()))],
        RegisterFauxProviderOptions {
            tokens_per_second: Some(150.0),
            token_size: Some(FauxTokenSize { min: Some(1), max: Some(1) }),
            ..Default::default()
        },
        HarnessSettings::default(),
        vec![],
    )
    .await;
    let partials = collect_partials(&fixture.harness);
    let recorder = Recorder::attach(&fixture.harness, fixture.root.id).await;
    let submission = fixture.root.submit(input("go", None), &context()).await.expect("submit input");
    submission.wait(&context()).await.expect("wait for tool-call round");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::RunEnd { .. }))).await;
    recorder.close(&fixture.harness).await;

    let all = recorder.events();
    let mut current = None;
    let mut rebuilt = Vec::new();
    for event in &all {
        match event {
            AgentEvent::MessageEnd { entry } if entry.kind == "pi.assistant" => break,
            AgentEvent::MessageStart { message: Message::Assistant(message) } => current = Some(message.clone()),
            AgentEvent::MessageUpdate { changes, .. } => {
                current = Some(apply_changes(current.as_ref().expect("message start precedes deltas"), changes));
            }
            _ => continue,
        }
        rebuilt.push(current.clone().expect("current assistant partial"));
    }
    let partials = partials.lock().clone();
    assert!(partials.len() > 2);
    assert_eq!(rebuilt.iter().map(|message| &message.content).collect::<Vec<_>>(), partials.iter().map(|message| &message.content).collect::<Vec<_>>());
    let changes = all.iter().filter_map(|event| match event {
        AgentEvent::MessageUpdate { changes, .. } => Some(changes.as_slice()),
        _ => None,
    }).flatten().collect::<Vec<_>>();
    assert!(changes.iter().any(|change| matches!(change, MessageChange::ThinkingDelta { .. })));
    assert!(changes.iter().any(|change| matches!(change, MessageChange::TextDelta { .. })));
}

#[tokio::test]
async fn tool_output_details_diagnostics_and_result_events_are_paired_in_order() {
    let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
    let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
    let release = Arc::new(Mutex::new(Some(release_rx)));
    let started = Arc::new(Mutex::new(Some(started_tx)));
    let execute = {
        let release = release.clone();
        let started = started.clone();
        Arc::new(move |_args: Value, api: Arc<dyn crate::harness::types::ToolExecutionApi>, tool_context: pi_chord::context::Context| {
            let release = release.clone();
            let started = started.clone();
            Box::pin(async move {
                api.output(b"one\n", None);
                api.details(json!({"phase":"first"}), &tool_context).await.expect("publish initial details");
                api.diagnostic(ToolDiagnostic {
                    severity: ToolDiagnosticSeverity::Warn,
                    message: "truncated detail".into(),
                    code: Some("output".into()),
                });
                tokio::time::sleep(Duration::from_millis(2)).await;
                api.output(b"two\n", None);
                api.details(json!({"phase":"done"}), &tool_context).await.expect("publish final details");
                let _ = started.lock().take().expect("tool starts once").send(());
                let release = release.lock().take().expect("tool completes once");
                let _ = release.await;
                ToolExecutionResult::default()
            }) as std::pin::Pin<Box<dyn Future<Output = ToolExecutionResult> + Send>>
        })
    };
    let fixture = default_fixture(
        vec![fixed_tool_call("print", "c1"), FauxResponseStep::Message(faux_assistant_message("done", FauxMessageOptions::default()))],
        vec![tool_extension("print", execute)],
    )
    .await;
    let recorder = Recorder::attach(&fixture.harness, fixture.root.id).await;
    let submission = fixture.root.submit(input("go", None), &context()).await.expect("submit input");
    tokio::time::timeout(Duration::from_secs(5), started_rx).await.expect("tool started").expect("start signal");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::ToolExecutionUpdate {
        output: Some(ToolOutputChange::Delta { append: Some(append), .. }), ..
    } if append.ends_with("two\n")))).await;
    release_tx.send(()).expect("release tool");
    assert_eq!(submission.wait(&context()).await.expect("wait for tool run").status, SubmissionStatus::Done);
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::RunEnd { .. }))).await;
    recorder.close(&fixture.harness).await;

    let events = recorder.events();
    let tool_events = events.iter().filter(|event| matches!(event,
        AgentEvent::ToolExecutionStart { .. } | AgentEvent::ToolExecutionUpdate { .. } | AgentEvent::ToolExecutionEnd { .. }
    )).collect::<Vec<_>>();
    assert!(matches!(tool_events.first(), Some(AgentEvent::ToolExecutionStart { tool_call_id, tool_name, args })
        if tool_call_id == "c1" && tool_name == "print" && args.get("n") == Some(&json!(1))));
    assert!(tool_events.iter().any(|event| matches!(event, AgentEvent::ToolExecutionUpdate {
        tool_call_id, output: Some(output), ..
    } if tool_call_id == "c1" && matches!(output, ToolOutputChange::Delta { append: Some(append), .. } if append.ends_with("two\n")))));
    assert!(tool_events.iter().any(|event| matches!(event, AgentEvent::ToolExecutionUpdate {
        tool_call_id, details: Some(Some(details)), ..
    } if tool_call_id == "c1" && details == &json!({"phase":"done"}))));
    assert!(tool_events.iter().any(|event| matches!(event, AgentEvent::ToolExecutionUpdate {
        tool_call_id, diagnostics: Some(diagnostics), ..
    } if tool_call_id == "c1" && diagnostics.iter().any(|diagnostic| diagnostic.code.as_deref() == Some("output")))));
    let end_index = events.iter().position(|event| matches!(event, AgentEvent::ToolExecutionEnd { tool_call_id, entry: Some(_), .. } if tool_call_id == "c1")).expect("tool result end event");
    assert!(matches!(&events[end_index], AgentEvent::ToolExecutionEnd { entry: Some(entry), .. } if entry.kind == "pi.tool-result"));
    let result = match &events[end_index] {
        AgentEvent::ToolExecutionEnd { entry: Some(entry), .. } => entry.model.as_ref()
            .expect("tool result entry has a model message")
            .iter()
            .find_map(|message| match message {
                Message::ToolResult(result) => Some(result),
                _ => None,
            })
            .expect("tool result entry contains its result message"),
        _ => panic!("tool execution end carries its result entry"),
    };
    assert_eq!(result.tool_call_id, "c1");
    assert!(!result.is_error);
    assert_eq!(result.details.as_ref(), Some(&json!({"phase":"done"})));
    assert!(result.content.iter().any(|content| matches!(content, pi_ai::types::UserContent::Text(text) if text.text == "one\ntwo\n")));
    assert!(matches!(events.get(end_index + 1), Some(AgentEvent::MessageStart { message: Message::ToolResult(result) }) if result.tool_call_id == "c1"));
    assert!(matches!(events.get(end_index + 2), Some(AgentEvent::MessageEnd { entry }) if entry.kind == "pi.tool-result"));
    assert_eq!(events.iter().filter(|event| matches!(event, AgentEvent::TurnStart)).count(), 2);
    assert_eq!(events.iter().filter(|event| matches!(event, AgentEvent::TurnEnd)).count(), 2);
}

#[tokio::test]
async fn tool_tail_output_events_rebuild_the_retained_window() {
    let execute = Arc::new(|_args: Value, api: Arc<dyn crate::harness::types::ToolExecutionApi>, _context: pi_chord::context::Context| {
        Box::pin(async move {
            for line in 0..6 {
                api.output(format!("line {line}\n").as_bytes(), None);
                tokio::time::sleep(Duration::from_millis(3)).await;
            }
            ToolExecutionResult::default()
        }) as std::pin::Pin<Box<dyn Future<Output = ToolExecutionResult> + Send>>
    });
    let tool = ToolRegistration {
        tool: pi_ai::Tool {
            name: "tail".into(), description: "tail output".into(),
            parameters: json!({"type":"object"}), constrained_sampling: None,
        },
        replay: None,
        execution_mode: Some(ToolExecutionMode::Sequential),
        prepare_arguments: None,
        output_limits: Some(ToolOutputLimits { max_bytes: None, max_lines: Some(3), retain: Some(crate::harness::types::OutputRetain::Tail) }),
        execute,
    };
    let extension = Extension::new("tail-tool").with_tools(vec![Arc::new(tool)]);
    let call = FauxResponseStep::Message(faux_assistant_message(
        FauxContentBlock::from(faux_tool_call("tail", serde_json::Map::new(), Some("tail-1"))),
        FauxMessageOptions { stop_reason: Some(StopReason::ToolUse), ..Default::default() },
    ));
    let fixture = fixture(
        vec![call, FauxResponseStep::Message(faux_assistant_message("done", FauxMessageOptions::default()))],
        RegisterFauxProviderOptions::default(),
        HarnessSettings {
            progress: Some(ProgressPolicyOverrides { partial_interval_ms: Some(1), output_interval_ms: Some(1) }),
            ..Default::default()
        },
        vec![extension],
    ).await;
    let outputs = collect_tool_outputs(&fixture.harness);
    let recorder = Recorder::attach(&fixture.harness, fixture.root.id).await;
    let submission = fixture.root.submit(input("go", None), &context()).await.expect("submit input");
    assert_eq!(submission.wait(&context()).await.expect("wait for bounded tool").status, SubmissionStatus::Done);
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::RunEnd { .. }))).await;
    recorder.close(&fixture.harness).await;
    let committed = outputs.lock().clone();
    let mut streamed = Vec::<String>::new();
    let mut current = String::new();
    for event in recorder.events() {
        if let AgentEvent::ToolExecutionUpdate { output: Some(output), .. } = event {
            match output {
                ToolOutputChange::Set { set } => current = set,
                ToolOutputChange::Delta { trim_start, append } => {
                    if let Some(trim_start) = trim_start {
                        let byte = current.char_indices().nth(trim_start as usize).map(|(byte, _)| byte).unwrap_or(current.len());
                        current.drain(..byte);
                    }
                    if let Some(append) = append { current.push_str(&append); }
                }
            }
            if streamed.last() != Some(&current) { streamed.push(current.clone()); }
        }
    }
    assert_eq!(streamed, committed);
    assert!(streamed.iter().any(|output| output == "line 3\nline 4\nline 5\n"));
    assert!(recorder.events().iter().any(|event| matches!(event, AgentEvent::ToolExecutionUpdate { output: Some(ToolOutputChange::Delta { trim_start: Some(count), .. }), .. } if *count > 0)));
}

#[tokio::test]
async fn queued_follow_up_updates_inbox_and_a_retry_completes_both_submissions() {
    let (release_tx, release_rx) = tokio::sync::oneshot::channel();
    let fixture = fixture(
        vec![
            held_response(release_rx, faux_assistant_message("first", FauxMessageOptions::default())),
            FauxResponseStep::Message(faux_assistant_message("", FauxMessageOptions { stop_reason: Some(StopReason::Error), error_message: Some("503 Service Unavailable".into()), ..Default::default() })),
            FauxResponseStep::Message(faux_assistant_message("second", FauxMessageOptions::default())),
        ],
        RegisterFauxProviderOptions::default(),
        HarnessSettings {
            retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(1), base_delay_ms: Some(1), ..Default::default() }),
            ..Default::default()
        },
        vec![],
    ).await;
    let recorder = Recorder::attach(&fixture.harness, fixture.root.id).await;
    let first = fixture.root.submit(input("a", None), &context()).await.expect("submit first input");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::RunStart { .. }))).await;
    let follow_up = fixture.root.submit(input("f", Some(WhenBusy::FollowUp)), &context()).await.expect("queue follow-up");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::InboxUpdate { items } if items.iter().any(|item| item.id == follow_up.id && item.mode == crate::harness::events::InboxMode::FollowUp)))).await;
    release_tx.send(()).expect("release first response");
    assert_eq!(first.wait(&context()).await.expect("wait for first submission").status, SubmissionStatus::Done);
    assert_eq!(follow_up.wait(&context()).await.expect("wait for follow-up").status, SubmissionStatus::Done);
    recorder.wait_until(|events| events.iter().filter(|event| matches!(event, AgentEvent::RunStart { .. })).count() == 2
        && events.iter().any(|event| matches!(event, AgentEvent::AutoRetryEnd { .. }))).await;
    recorder.close(&fixture.harness).await;

    let events = recorder.events();
    let inbox_updates = events.iter().filter_map(|event| match event {
        AgentEvent::InboxUpdate { items } => Some(items),
        _ => None,
    }).collect::<Vec<_>>();
    assert!(inbox_updates.iter().any(|items| items.len() == 1 && items[0].id == follow_up.id && items[0].mode == crate::harness::events::InboxMode::FollowUp));
    assert!(inbox_updates.iter().any(|items| items.is_empty()));
    assert!(events.iter().any(|event| matches!(event, AgentEvent::AutoRetryStart { error_message, .. } if error_message.contains("503 Service Unavailable"))));
    assert_eq!(events.iter().filter(|event| matches!(event, AgentEvent::AutoRetryEnd { .. })).count(), 1);
    let statuses = events.iter().filter_map(|event| match event {
        AgentEvent::Submission { record } => Some(record.status),
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(statuses, [SubmissionStatus::Placed, SubmissionStatus::Queued, SubmissionStatus::Done, SubmissionStatus::Placed, SubmissionStatus::Done]);
    assert_eq!(events.iter().filter(|event| matches!(event, AgentEvent::RunStart { .. })).count(), 2);
}

#[tokio::test]
async fn run_end_and_queued_follow_up_start_share_one_boundary_batch() {
    let (release_tx, release_rx) = tokio::sync::oneshot::channel();
    let fixture = fixture(
        vec![
            held_response(release_rx, faux_assistant_message("first", FauxMessageOptions::default())),
            FauxResponseStep::Message(faux_assistant_message("second", FauxMessageOptions::default())),
        ],
        RegisterFauxProviderOptions::default(),
        HarnessSettings::default(),
        vec![],
    )
    .await;
    let recorder = Recorder::attach(&fixture.harness, fixture.root.id).await;
    let input_submission = fixture.root.submit(input("a", None), &context()).await.expect("submit first");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::RunStart { .. }))).await;
    let follow_up = fixture.root.submit(input("f", Some(WhenBusy::FollowUp)), &context()).await.expect("queue second");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::InboxUpdate { items } if items.iter().any(|item| item.id == follow_up.id)))).await;
    release_tx.send(()).expect("release first response");
    assert_eq!(follow_up.wait(&context()).await.expect("wait for follow-up").status, SubmissionStatus::Done);
    recorder.wait_until(|events| events.iter().filter(|event| matches!(event, AgentEvent::RunStart { .. })).count() == 2).await;
    recorder.close(&fixture.harness).await;
    let boundary = recorder.batches().into_iter().find(|batch| batch.iter().any(|event| matches!(event, AgentEvent::RunEnd { .. }))).expect("run boundary batch");
    assert_eq!(boundary.iter().map(event_name).collect::<Vec<_>>(), [
        "message_start", "message_end", "message_start", "message_end", "turn_end",
        "run_end", "submission", "submission", "inbox_update", "usage_changed", "run_start", "turn_start",
    ]);
    let ids = boundary.iter().filter_map(|event| match event {
        AgentEvent::Submission { record } => Some(record.id),
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(ids, [input_submission.id, follow_up.id]);
}

#[tokio::test]
async fn unavailable_and_aborted_tool_calls_end_with_their_result_entries() {
    let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
    let started = Arc::new(Mutex::new(Some(started_tx)));
    let execute = {
        let started = started.clone();
        Arc::new(move |_args: Value, _api: Arc<dyn crate::harness::types::ToolExecutionApi>, call_context: pi_chord::context::Context| {
            let started = started.clone();
            Box::pin(async move {
                let signal = call_context.abort_signal().expect("tool invocation abort signal");
                let (abort_tx, abort_rx) = tokio::sync::oneshot::channel();
                let abort_tx = Arc::new(Mutex::new(Some(abort_tx)));
                let _detach = signal.add_listener(move || {
                    if let Some(abort_tx) = abort_tx.lock().take() { let _ = abort_tx.send(()); }
                });
                let _ = started.lock().take().expect("tool starts once").send(());
                let _ = abort_rx.await;
                ToolExecutionResult::default()
            }) as std::pin::Pin<Box<dyn Future<Output = ToolExecutionResult> + Send>>
        })
    };
    let response = faux_assistant_message(
        vec![
            FauxContentBlock::from(faux_tool_call("ghost", serde_json::Map::new(), Some("c1"))),
            FauxContentBlock::from(faux_tool_call("wait", serde_json::Map::new(), Some("c2"))),
        ],
        FauxMessageOptions { stop_reason: Some(StopReason::ToolUse), ..Default::default() },
    );
    let fixture = fixture(
        vec![FauxResponseStep::Message(response)],
        RegisterFauxProviderOptions::default(),
        HarnessSettings::default(),
        vec![tool_extension("wait", execute)],
    ).await;
    let recorder = Recorder::attach(&fixture.harness, fixture.root.id).await;
    let submission = fixture.root.submit(input("go", None), &context()).await.expect("submit tool calls");
    started_rx.await.expect("wait tool start");
    let live = live_state(&fixture).await;
    fixture.harness.abort_task(live.run.expect("live run").task_id, &context()).await.expect("abort generation task");
    submission.wait(&context()).await.expect("wait for aborted tool round");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::RunEnd { .. }))).await;
    recorder.close(&fixture.harness).await;
    let events = recorder.events();
    let round = events.iter().map(|event| match event {
        AgentEvent::MessageEnd { entry } => format!("end:{}", entry.kind),
        AgentEvent::MessageStart { message } => format!("start:{}", match message { Message::ToolResult(_) => "toolResult", Message::Assistant(_) => "assistant", Message::User(_) => "user", _ => "other" }),
        event => event_name(event).to_owned(),
    }).collect::<Vec<_>>();
    let ghost_end = events.iter().position(|event| matches!(event, AgentEvent::ToolExecutionEnd { tool_call_id, .. } if tool_call_id == "c1")).expect("unavailable call end");
    assert_eq!(&round[ghost_end - 1..ghost_end + 3], ["end:pi.assistant", "tool_execution_end", "start:toolResult", "end:pi.tool-result"]);
    let tool_lifecycle = events.iter().filter_map(|event| match event {
        AgentEvent::ToolExecutionStart { tool_call_id, .. } => Some(("start", tool_call_id.as_str(), false)),
        AgentEvent::ToolExecutionEnd { tool_call_id, entry, .. } => Some(("end", tool_call_id.as_str(), entry.is_some())),
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(tool_lifecycle, [("end", "c1", true), ("start", "c2", false), ("end", "c2", true)]);
}

#[tokio::test]
async fn steering_reset_and_other_conversation_commits_stay_with_the_root_stream() {
    let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
    let release = Arc::new(Mutex::new(Some(release_rx)));
    let execute = {
        let release = release.clone();
        Arc::new(move |_args: Value, _api: Arc<dyn crate::harness::types::ToolExecutionApi>, _context: pi_chord::context::Context| {
            let release = release.clone();
            Box::pin(async move {
                let release = release.lock().take().expect("held tool starts once");
                let _ = release.await;
                ToolExecutionResult::default()
            }) as std::pin::Pin<Box<dyn Future<Output = ToolExecutionResult> + Send>>
        })
    };
    let fixture = default_fixture(
        vec![fixed_tool_call("hold", "c1"), FauxResponseStep::Message(faux_assistant_message("done", FauxMessageOptions::default()))],
        vec![tool_extension("hold", execute)],
    ).await;
    let other = fixture.harness.create_conversation(ConversationCreateOptions {
        ownership: crate::types::ConversationOwnership::Ownerless,
        agent: None,
        init: None,
    }, &context()).await.expect("create unrelated conversation");
    let recorder = Recorder::attach(&fixture.harness, fixture.root.id).await;
    let first = fixture.root.submit(input("a", None), &context()).await.expect("submit input");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::ToolExecutionStart { .. }))).await;
    let steering = fixture.root.submit(input("steer", Some(WhenBusy::Steer)), &context()).await.expect("submit steering input");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::Submission { record } if record.id == steering.id))).await;
    let batch_count = recorder.batches().len();
    let other_id = other.id;
    other.commit(move |tx| tx.append_entry(other_id, EntryDraft { kind: Some("note".into()), ..Default::default() }).map_err(|error| error.to_string()), &context()).await.expect("commit unrelated entry");
    assert_eq!(recorder.batches().len(), batch_count, "other conversations must not publish to this stream");
    release_tx.send(()).expect("release held tool");
    first.wait(&context()).await.expect("wait for run");
    fixture.root.reset(None, &context()).await.expect("reset root conversation");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::EntryAppended { entry } if entry.kind == "pi.reset"))).await;
    recorder.close(&fixture.harness).await;
    let events = recorder.events();
    assert_eq!(events.iter().filter(|event| matches!(event, AgentEvent::RunStart { .. })).count(), 1);
    assert_eq!(events.iter().filter(|event| matches!(event, AgentEvent::RunEnd { .. })).count(), 1);
    assert_eq!(recorder.batches().last().expect("reset batch").iter().map(event_name).collect::<Vec<_>>(), ["entry_appended", "submission"]);
}

#[tokio::test]
async fn cancelling_event_attachment_while_waiting_for_the_session_line_returns_the_reason() {
    let fixture = default_fixture(vec![], vec![]).await;
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel::<()>();
    let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
    let blocking_harness = fixture.harness.clone();
    let blocking = tokio::spawn(async move {
        blocking_harness.session.commit_async(move |_tx| {
            Box::pin(async move {
                let _ = entered_tx.send(());
                let _ = release_rx.await;
                Ok(())
            })
        }, &context()).await
    });
    tokio::time::timeout(Duration::from_secs(5), entered_rx).await.expect("commit acquired the Session line").expect("entered signal");
    let controller = pi_chord::context::AbortController::new();
    let cancelled_context = pi_chord::context::with_abort_signal(controller.signal(), context());
    let harness = fixture.harness.clone();
    let conversation_id = fixture.root.id;
    let attaching = tokio::spawn(async move { watch_events(&harness, conversation_id, &cancelled_context).await });
    controller.abort("cancelled");
    release_tx.send(()).expect("release blocked commit");
    blocking.await.expect("commit task").expect("blocking commit");
    let error = match attaching.await.expect("attachment task") {
        Ok(_) => panic!("expected cancelled attachment"),
        Err(error) => error,
    };
    assert!(error.contains("cancelled"));
    fixture.harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn deferred_faux_responses_emit_polls_before_the_answer() {
    let fixture = fixture(
        vec![FauxResponseStep::Message(faux_assistant_message("deferred answer", FauxMessageOptions::default()))],
        RegisterFauxProviderOptions {
            deferred: Some(FauxDeferredOptions { pending_fetches: Some(1), poll_after_ms: Some(1) }),
            ..Default::default()
        },
        HarnessSettings {
            stream: Some(ConversationStreamOptions { deferred: Some(json!(true)), ..Default::default() }),
            ..Default::default()
        },
        vec![],
    ).await;
    let recorder = Recorder::attach(&fixture.harness, fixture.root.id).await;
    let submission = fixture.root.submit(input("hi", None), &context()).await.expect("submit deferred request");
    assert_eq!(submission.wait(&context()).await.expect("wait for deferred answer").status, SubmissionStatus::Done);
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::RunEnd { .. }))).await;
    recorder.close(&fixture.harness).await;
    let events = recorder.events();
    assert!(events.iter().any(|event| matches!(event, AgentEvent::DeferredPoll { poll_at } if *poll_at > 0)));
    assert!(events.iter().any(|event| matches!(event, AgentEvent::MessageEnd { entry } if entry.kind == "pi.assistant")));
    assert_eq!(fixture.faux.state().deferred_fetch_count(), 2);
}

#[tokio::test]
async fn failed_durable_task_emits_its_kind_and_error() {
    let phase: crate::tasks::PhaseHandler = Arc::new(|_record, _runtime, _context| {
        Box::pin(async { Err("scripted task failure".to_owned()) }) as crate::tasks::BoxFuture<Result<(), String>>
    });
    let abort: crate::tasks::PhaseHandler = Arc::new(|_record, _runtime, _context| {
        Box::pin(async { Ok(()) }) as crate::tasks::BoxFuture<Result<(), String>>
    });
    let task = crate::tasks::define_task(crate::tasks::TaskDefinition {
        name: "event.task-failure".into(),
        version: 1,
        initial: Arc::new(|_| json!({"phase":"run"})),
        phases: Arc::new(BTreeMap::from([("run".to_owned(), phase)])),
        abort,
        migrate: None,
        hooks: None,
    });
    let any_task = crate::harness::types::AnyTask { definition: task.definition };
    let fixture = default_fixture(
        vec![],
        vec![Extension::new("event-task-failure").with_tasks(vec![any_task.clone()])],
    )
    .await;
    let recorder = Recorder::attach(&fixture.harness, fixture.root.id).await;
    let task_id = fixture.root.commit(move |tx| {
        tx.create_task(
            &any_task,
            Value::Null,
            crate::types::TaskOptions {
                ownership: crate::types::TaskOwnership::Conversation,
                conversation_id: None,
                background: None,
            },
        )
        .map_err(|error| error.to_string())
    }, &context()).await.expect("create failing task");
    fixture.harness.wait_for_task(task_id, &context()).await.expect("wait for failed task outcome");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::TaskFailed { task_id: failed, .. } if *failed == task_id))).await;
    recorder.close(&fixture.harness).await;
    assert!(recorder.events().iter().any(|event| matches!(event, AgentEvent::TaskFailed { task_id: failed, kind, message }
        if *failed == task_id && kind == "event.task-failure" && message.contains("scripted task failure"))));
}

#[tokio::test]
async fn manual_compaction_emits_start_and_end_around_its_faux_summary() {
    let fixture = fixture(
        vec![
            FauxResponseStep::Message(faux_assistant_message("A sufficiently detailed answer to make a summary candidate.", FauxMessageOptions::default())),
            FauxResponseStep::Message(faux_assistant_message("A second answer to create earlier history.", FauxMessageOptions::default())),
            FauxResponseStep::Message(faux_assistant_message("A concise durable summary.", FauxMessageOptions::default())),
        ],
        RegisterFauxProviderOptions::default(),
        HarnessSettings {
            compaction: Some(crate::harness::types::CompactionPolicyOverrides { keep_recent_tokens: Some(1), ..Default::default() }),
            ..Default::default()
        },
        vec![],
    ).await;
    let recorder = Recorder::attach(&fixture.harness, fixture.root.id).await;
    let submission = fixture.root.submit(input("Please summarize the conversation.", None), &context()).await.expect("submit conversation");
    assert_eq!(submission.wait(&context()).await.expect("wait for first answer").status, SubmissionStatus::Done);
    let second = fixture.root.submit(input("Continue the conversation.", None), &context()).await.expect("submit second exchange");
    assert_eq!(second.wait(&context()).await.expect("wait for second answer").status, SubmissionStatus::Done);
    let task_id = fixture.root.compact(None, &context()).await.expect("start manual compaction");
    fixture.harness.wait_for_task(task_id, &context()).await.expect("wait for compaction");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::CompactionEnd { task_id: ended, .. } if *ended == task_id))).await;
    recorder.close(&fixture.harness).await;
    let events = recorder.events();
    assert!(events.iter().any(|event| matches!(event, AgentEvent::CompactionStart { task_id: started, reason: crate::harness::types::CompactionReason::Manual, blocking: false } if *started == task_id)));
    assert!(events.iter().any(|event| matches!(event, AgentEvent::CompactionEnd { task_id: ended, reason: crate::harness::types::CompactionReason::Manual } if *ended == task_id)));
    assert!(events.iter().any(|event| matches!(event, AgentEvent::MessageEnd { entry } if entry.kind == "pi.compaction")));
}

#[tokio::test]
async fn abort_after_a_committed_generation_partial_ends_that_assistant_entry() {
    let fixture = fixture(
        vec![FauxResponseStep::Message(faux_assistant_message("x".repeat(400), FauxMessageOptions::default()))],
        RegisterFauxProviderOptions {
            tokens_per_second: Some(100.0),
            token_size: Some(FauxTokenSize { min: Some(1), max: Some(1) }),
            ..Default::default()
        },
        HarnessSettings {
            progress: Some(ProgressPolicyOverrides { partial_interval_ms: Some(1), output_interval_ms: Some(1) }),
            ..Default::default()
        },
        vec![],
    ).await;
    let recorder = Recorder::attach(&fixture.harness, fixture.root.id).await;
    let submission = fixture.root.submit(input("hi", None), &context()).await.expect("submit long response");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::MessageStart { message: Message::Assistant(_) }))).await;
    let task_id = live_state(&fixture).await.run.expect("generation run").task_id;
    fixture.harness.abort_task(task_id, &context()).await.expect("abort generation");
    let _ = submission.wait(&context()).await.expect("wait for aborted generation");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::RunEnd { .. }))).await;
    recorder.close(&fixture.harness).await;
    let events = recorder.events();
    let starts = events.iter().filter(|event| matches!(event, AgentEvent::MessageStart { message: Message::Assistant(_) })).count();
    let ends = events.iter().filter(|event| matches!(event, AgentEvent::MessageEnd { entry } if entry.kind == "pi.assistant")).count();
    assert_eq!(starts, 1);
    assert_eq!(ends, 1, "the committed partial must terminate with its converted entry on abort");
}

async fn set_live(fixture: &Fixture, edit: impl FnOnce(&mut crate::harness::live::LiveState) + Send + 'static) {
    let id = fixture.root.id;
    fixture.root.commit(move |tx| {
        let draft = tx.doc(&crate::harness::live::live_doc(), Some(id.0), None, None).map_err(|error| error.to_string())?;
        let mut live = crate::harness::live::read_live(&draft);
        edit(&mut live);
        crate::harness::live::store_live(&draft, &live)
    }, &context()).await.expect("commit live state");
}

#[tokio::test]
async fn committed_progress_only_updates_deferred_polls_and_retired_documents_translate_distinctly() {
    let fixture = default_fixture(vec![], vec![]).await;
    let recorder = Recorder::attach(&fixture.harness, fixture.root.id).await;
    let partial = faux_assistant_message("partial", FauxMessageOptions::default());
    set_live(&fixture, move |live| {
        live.generation = Some(crate::harness::live::GenerationStatus {
            attempt: 1,
            message: Some(serde_json::to_value(partial).expect("partial serializes")),
            ..Default::default()
        });
    }).await;
    set_live(&fixture, |live| {
        live.generation.as_mut().unwrap().message.as_mut().unwrap()["usage"]["input"] = json!(42);
    }).await;
    set_live(&fixture, |live| {
        live.generation = Some(crate::harness::live::GenerationStatus {
            attempt: 1,
            deferred: Some(crate::harness::live::DeferredPoll { poll_at: 1 }),
            ..Default::default()
        });
    }).await;
    set_live(&fixture, |live| {
        live.generation.as_mut().unwrap().deferred.as_mut().unwrap().poll_at = 2;
    }).await;
    set_live(&fixture, |live| {
        live.tools = Some(vec![crate::harness::live::ToolSlot {
            call_id: "c1".into(), name: "t".into(), status: crate::harness::live::SlotStatus::Running,
            details: Some(json!({"n":1})), diagnostics: Some(vec![]), ..Default::default()
        }]);
    }).await;
    set_live(&fixture, |live| {
        live.tools.as_mut().unwrap()[0].details = None;
        live.tools.as_mut().unwrap()[0].diagnostics = None;
    }).await;
    set_live(&fixture, |live| live.tools.as_mut().unwrap()[0].status = crate::harness::live::SlotStatus::Done).await;
    let root_id = fixture.root.id;
    fixture.root.commit(move |tx| tx.retire_doc(&crate::harness::usage::usage_doc(), Some(root_id.0), None).map_err(|error| error.to_string()), &context()).await.expect("retire usage doc");
    fixture.root.commit(move |tx| tx.retire_doc(&crate::harness::agent::agent_doc(), Some(root_id.0), None).map_err(|error| error.to_string()), &context()).await.expect("retire agent doc");
    recorder.wait_until(|events| events.iter().any(|event| matches!(event, AgentEvent::AgentChanged { .. }))).await;
    recorder.close(&fixture.harness).await;
    let batches = recorder.batches().into_iter().filter(|batch| !batch.iter().all(|event| matches!(event, AgentEvent::Snapshot { .. }))).collect::<Vec<_>>();
    let visible = batches.iter().map(|batch| batch.iter().filter(|event| !matches!(event, AgentEvent::TaskFailed { .. })).collect::<Vec<_>>()).collect::<Vec<_>>();
    assert!(matches!(visible[0].as_slice(), [AgentEvent::MessageStart { message: Message::Assistant(message) }] if text_of(message) == "partial"));
    assert!(matches!(visible[1].as_slice(), [AgentEvent::MessageUpdate { changes, usage }] if changes.is_empty() && usage.input == 42));
    assert!(matches!(visible[2].as_slice(), [AgentEvent::DeferredPoll { poll_at: 1 }]));
    assert!(matches!(visible[3].as_slice(), [AgentEvent::DeferredPoll { poll_at: 2 }]));
    assert!(matches!(visible[4].as_slice(), [AgentEvent::ToolExecutionStart { tool_call_id, tool_name, args }] if tool_call_id == "c1" && tool_name == "t" && args.is_empty()));
    assert!(matches!(visible[5].as_slice(), [AgentEvent::ToolExecutionUpdate { tool_call_id, details, diagnostics, .. }] if tool_call_id == "c1" && details.is_none() && diagnostics.as_ref().is_some_and(Vec::is_empty)));
    assert!(matches!(visible[6].as_slice(), [AgentEvent::ToolExecutionEnd { tool_call_id, entry: None, .. }] if tool_call_id == "c1"));
    assert!(matches!(visible[7].as_slice(), [AgentEvent::UsageChanged { usage }] if usage == &crate::harness::usage::UsageState::default()));
    assert!(matches!(visible[8].as_slice(), [AgentEvent::AgentChanged { agent }] if agent == &crate::harness::types::AgentState::default()));
}

#[tokio::test]
async fn overflow_snapshot_keeps_a_partial_message_for_following_text_deltas() {
    let fixture = default_fixture(vec![], vec![]).await;
    let mut stream = watch_events(&fixture.harness, fixture.root.id, &context()).await.expect("attach event stream");
    let initial = stream.next().await.expect("initial event batch").expect("initial snapshot");
    assert!(matches!(initial.as_slice(), [AgentEvent::Snapshot { .. }]));
    set_live(&fixture, move |live| {
        live.generation = Some(crate::harness::live::GenerationStatus {
            attempt: 1,
            message: Some(serde_json::to_value(faux_assistant_message("hel", FauxMessageOptions::default())).expect("partial serializes")),
            ..Default::default()
        });
    }).await;
    for _ in 0..101 {
        let id = fixture.root.id;
        fixture.root.commit(move |tx| {
            tx.append_entry(id, EntryDraft { kind: Some("note".into()), ..Default::default() })
                .map_err(|error| error.to_string())
        }, &context()).await.expect("append overflow entry");
    }
    let replacement = stream.next().await.expect("overflow replacement batch").expect("snapshot result");
    assert_eq!(replacement.len(), 1);
    let AgentEvent::Snapshot { snapshot } = &replacement[0] else {
        panic!("overflow must replace pending deltas with a snapshot");
    };
    let generation = snapshot.generation.as_ref().expect("snapshot holds generation");
    let partial = generation.message.as_ref().expect("snapshot holds partial message").clone();

    set_live(&fixture, |live| {
        live.generation.as_mut().unwrap().message.as_mut().unwrap()["content"][0]["text"] = json!("hello");
    }).await;
    let update_batch = loop {
        let batch = stream.next().await.expect("post-overflow update").expect("message update batch");
        if matches!(batch.as_slice(), [AgentEvent::MessageUpdate { .. }]) {
            break batch;
        }
        assert!(batch.iter().all(|event| matches!(event, AgentEvent::EntryAppended { entry } if entry.kind == "note")), "unexpected event after replacement snapshot: {batch:?}");
    };
    let AgentEvent::MessageUpdate { changes, .. } = &update_batch[0] else {
        panic!("expected a delta after the replacement snapshot, got {update_batch:?}");
    };
    assert!(matches!(changes.as_slice(), [MessageChange::TextDelta { content_index: 0, delta }] if delta == "lo"));
    let rebuilt = apply_changes(&partial, changes);
    assert!(matches!(rebuilt.content.as_slice(), [AssistantContent::Text(text)] if text.text == "hello"));
    fixture.harness.close(&context()).await.expect("close harness");
}
