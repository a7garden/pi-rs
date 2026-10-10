use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;
use pi_ai::types::{
    AssistantContent, AssistantMessage, Message, StopReason, SystemContent, SystemMessage,
    TextContent, ToolCall, ToolResultMessage, UserContent, UserContentField, UserMessage,
};
use pi_chord::context::Context;
use serde_json::{json, Value};

use super::support::{context, open_default, root};
use crate::harness::harness::Conversation;
use crate::harness::types::{AnyTask, ConversationCreateOptions, Extension, Registry};
use crate::storage::memory::MemoryStorage;
use crate::tasks::{
    commit_change, define_task, BoxFuture, NextTaskState, PhaseHandler, TaskDefinition, TaskRuntime,
};
use crate::types::{
    ConversationId, ConversationOwnership, EntryDraft, EntryHead, EntryId, EntryRecord, TaskId,
    TaskOptions, TaskOutcomeStatus, TaskOwnership, TaskRecord, TaskStateStatus,
};

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: UserContentField::Text(text.to_owned()),
        timestamp: 0,
    })
}

fn assistant(text: &str, stop_reason: StopReason) -> Message {
    Message::Assistant(AssistantMessage {
        content: vec![AssistantContent::Text(TextContent::new(text))],
        api: "test".into(),
        provider: "test".into(),
        model: "test".into(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        usage: Default::default(),
        stop_reason,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        duration_ms: None,
        timestamp: 0,
    })
}

fn assistant_with_calls(text: &str, calls: &[(&str, &str)]) -> Message {
    let mut message = match assistant(text, StopReason::ToolUse) {
        Message::Assistant(message) => message,
        _ => unreachable!(),
    };
    message.content.extend(calls.iter().map(|(id, name)| {
        AssistantContent::ToolCall(ToolCall {
            id: (*id).to_owned(),
            name: (*name).to_owned(),
            arguments: Default::default(),
            thought_signature: None,
            namespace: None,
        })
    }));
    Message::Assistant(message)
}

fn system(text: &str) -> Message {
    Message::System(SystemMessage {
        content: SystemContent::Text(text.to_owned()),
        sections: None,
        tools_added: None,
        tools_removed: None,
        timestamp: 0,
    })
}

fn tool_result(id: &str, name: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: id.to_owned(),
        tool_name: name.to_owned(),
        content: vec![UserContent::Text(TextContent::new(format!("result {id}")))],
        details: None,
        usage: None,
        nested_calls: None,
        is_error: false,
        duration_ms: None,
        timestamp: 0,
    })
}

async fn append(conversation: &Conversation, draft: EntryDraft) -> EntryRecord {
    let conversation_id = conversation.id;
    conversation
        .commit(
            move |tx| {
                tx.append_entry(conversation_id, draft)
                    .map_err(|error| error.to_string())
            },
            &context(),
        )
        .await
        .expect("append entry")
}

async fn append_model(conversation: &Conversation, kind: &str, model: Vec<Message>) -> EntryRecord {
    append(
        conversation,
        EntryDraft {
            kind: Some(kind.to_owned()),
            model: Some(model),
            ..Default::default()
        },
    )
    .await
}

#[tokio::test]
async fn context_excludes_deferred_assistant_messages_but_retains_the_raw_entry() {
    let (_storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    let question = append_model(&conversation, "message", vec![user("question")]).await;
    let deferred = append_model(
        &conversation,
        "message",
        vec![assistant("pending", StopReason::Deferred)],
    )
    .await;
    let answer = append_model(
        &conversation,
        "message",
        vec![assistant("ready", StopReason::Stop)],
    )
    .await;

    let view = conversation
        .context(&context(), None)
        .await
        .expect("derive context");
    assert!(view.head.is_none());
    assert_eq!(
        view.entries
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>(),
        [question.id, deferred.id, answer.id]
    );
    assert_eq!(view.messages.len(), 2);
    assert!(
        matches!(&view.messages[0], Message::User(message) if message.content == UserContentField::Text("question".into()))
    );
    assert!(matches!(&view.messages[1], Message::Assistant(message)
		if message.stop_reason == StopReason::Stop
			&& message.content.iter().any(|content| matches!(content, AssistantContent::Text(text) if text.text == "ready"))));
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn context_summary_head_keeps_its_target_and_following_entries() {
    let (_storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    append_model(&conversation, "message", vec![user("old")]).await;
    append(
        &conversation,
        EntryDraft {
            kind: Some("reset".into()),
            head: Some(EntryHead::SelfMarker(crate::types::SelfMarker)),
            model: Some(vec![user("fresh start")]),
            ..Default::default()
        },
    )
    .await;
    let kept = append_model(
        &conversation,
        "message",
        vec![assistant("after reset", StopReason::Stop)],
    )
    .await;
    let summary = append(
        &conversation,
        EntryDraft {
            kind: Some("summary".into()),
            head: Some(EntryHead::Entry(kept.id)),
            model: Some(vec![user("summary")]),
            ..Default::default()
        },
    )
    .await;
    let tail = append_model(&conversation, "message", vec![user("next")]).await;

    let view = conversation
        .context(&context(), None)
        .await
        .expect("derive summarized context");
    assert_eq!(view.head.as_ref().map(|entry| entry.id), Some(summary.id));
    assert_eq!(
        view.entries
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>(),
        [summary.id, kept.id, tail.id]
    );
    assert_eq!(view.messages.len(), 3);
    assert!(
        matches!(&view.messages[0], Message::User(message) if message.content == UserContentField::Text("summary".into()))
    );
    assert!(matches!(&view.messages[1], Message::Assistant(message)
		if message.content.iter().any(|content| matches!(content, AssistantContent::Text(text) if text.text == "after reset"))));
    assert!(
        matches!(&view.messages[2], Message::User(message) if message.content == UserContentField::Text("next".into()))
    );
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn context_keeps_system_messages_at_their_transcript_positions() {
    let (_storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    let before = append_model(&conversation, "pi.system", vec![system("You help.")]).await;
    let first_user = append_model(&conversation, "pi.user", vec![user("first")]).await;
    let middle = append_model(&conversation, "pi.system", vec![system("cwd: /repo")]).await;
    let second_user = append_model(&conversation, "pi.user", vec![user("second")]).await;

    let view = conversation
        .context(&context(), None)
        .await
        .expect("derive context");
    assert_eq!(
        view.entries
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>(),
        [before.id, first_user.id, middle.id, second_user.id]
    );
    assert_eq!(view.messages.len(), 4);
    assert!(
        matches!(&view.messages[0], Message::System(message) if matches!(&message.content, SystemContent::Text(text) if text == "You help."))
    );
    assert!(
        matches!(&view.messages[1], Message::User(message) if message.content == UserContentField::Text("first".into()))
    );
    assert!(
        matches!(&view.messages[2], Message::System(message) if matches!(&message.content, SystemContent::Text(text) if text == "cwd: /repo"))
    );
    assert!(
        matches!(&view.messages[3], Message::User(message) if message.content == UserContentField::Text("second".into()))
    );
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn fork_context_hides_later_entries_synthesizes_missing_results_and_drops_cut_results() {
    let (_storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    let input = append_model(&conversation, "message", vec![user("go")]).await;
    let call = append_model(
        &conversation,
        "message",
        vec![assistant_with_calls(
            "calling",
            &[("x", "tool-x"), ("y", "tool-y")],
        )],
    )
    .await;
    append_model(&conversation, "message", vec![tool_result("x", "tool-x")]).await;
    let second_result =
        append_model(&conversation, "message", vec![tool_result("y", "tool-y")]).await;

    let fork = conversation
        .fork(
            call.id,
            ConversationCreateOptions {
                ownership: ConversationOwnership::Ownerless,
                agent: None,
                init: None,
            },
            &context(),
        )
        .await
        .expect("fork at tool call");
    let fork_view = fork
        .context(&context(), None)
        .await
        .expect("read fork context");
    assert_eq!(
        fork_view
            .entries
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>(),
        [input.id, call.id]
    );
    assert_eq!(fork_view.messages.len(), 4);
    assert!(
        matches!(&fork_view.messages[0], Message::User(message) if message.content == UserContentField::Text("go".into()))
    );
    assert!(matches!(&fork_view.messages[1], Message::Assistant(message)
		if message.content.iter().any(|content| matches!(content, AssistantContent::Text(text) if text.text == "calling"))));
    for (message, (expected_id, expected_name)) in fork_view.messages[2..]
        .iter()
        .zip([("x", "tool-x"), ("y", "tool-y")])
    {
        let Message::ToolResult(result) = message else {
            panic!("missing tool result was not synthesized")
        };
        assert_eq!(result.tool_call_id, expected_id);
        assert_eq!(result.tool_name, expected_name);
        assert!(result.is_error);
        assert_eq!(result.details, Some(json!({"reason": "missing_result"})));
        assert_eq!(
            result.content,
            [UserContent::Text(TextContent::new(
                "Tool result unavailable: history ends before this call completed."
            ))]
        );
    }

    let reset = append(
        &conversation,
        EntryDraft {
            kind: Some("reset".into()),
            head: Some(EntryHead::Entry(second_result.id)),
            ..Default::default()
        },
    )
    .await;
    let parent_view = conversation
        .context(&context(), None)
        .await
        .expect("read parent context after cut");
    assert!(parent_view.messages.is_empty());
    assert_eq!(
        parent_view
            .entries
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>(),
        [reset.id, second_result.id]
    );
    harness.close(&context()).await.expect("close harness");
}

#[derive(Debug, PartialEq, Eq)]
struct ContextRead {
    entries: Vec<EntryId>,
    messages: Vec<String>,
}

fn user_texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .map(|message| match message {
            Message::User(message) => match &message.content {
                UserContentField::Text(text) => text.clone(),
                _ => panic!("expected text user message"),
            },
            _ => panic!("expected only user messages"),
        })
        .collect()
}

fn context_read(view: crate::session::ContextView) -> ContextRead {
    ContextRead {
        entries: view.entries.iter().map(|entry| entry.id).collect(),
        messages: user_texts(&view.messages),
    }
}

#[tokio::test]
async fn task_context_reads_can_be_cut_off_at_each_visible_entry() {
    let (_storage, registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    let first = append_model(&conversation, "message", vec![user("one")]).await;
    let second = append_model(&conversation, "message", vec![user("two")]).await;
    let observed = Arc::new(Mutex::new(Vec::<ContextRead>::new()));
    let conversation_id = conversation.id;
    let phase: PhaseHandler = {
        let observed = observed.clone();
        Arc::new(
            move |_record: TaskRecord, runtime: Arc<dyn TaskRuntime>, ctx: Context| {
                let observed = observed.clone();
                Box::pin(async move {
                    let full = runtime.context_view(conversation_id, &ctx, None).await?;
                    let first_id = full
                        .entries
                        .first()
                        .ok_or_else(|| "first context entry missing".to_owned())?
                        .id;
                    let second_id = full
                        .entries
                        .get(1)
                        .ok_or_else(|| "second context entry missing".to_owned())?
                        .id;
                    let through_first = runtime
                        .context_view(conversation_id, &ctx, Some(first_id))
                        .await?;
                    let through_second = runtime
                        .context_view(conversation_id, &ctx, Some(second_id))
                        .await?;
                    observed.lock().extend([
                        context_read(full),
                        context_read(through_first),
                        context_read(through_second),
                    ]);
                    runtime
                        .commit(
                            commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                            &ctx,
                        )
                        .await?;
                    Ok(())
                }) as BoxFuture<Result<(), String>>
            },
        )
    };
    let abort: PhaseHandler = Arc::new(|_record, runtime, ctx| {
        Box::pin(async move {
            runtime
                .commit(
                    commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("stopped")))),
                    &ctx,
                )
                .await?;
            Ok(())
        })
    });
    let reader = define_task(TaskDefinition {
        name: "test.runtime-context-reader".into(),
        version: 1,
        initial: Arc::new(|_| json!({"phase": "read"})),
        phases: Arc::new(BTreeMap::from([("read".into(), phase)])),
        abort,
        migrate: None,
        hooks: None,
    });
    registry.install(
        Extension::new("runtime-context-tests").with_tasks(vec![AnyTask {
            definition: reader.definition.clone(),
        }]),
    );
    let task_definition = reader.definition;
    let task_id = conversation
        .commit(
            move |tx| {
                tx.create_task(
                    &AnyTask {
                        definition: task_definition,
                    },
                    Value::Null,
                    TaskOptions {
                        ownership: TaskOwnership::Conversation,
                        conversation_id: None,
                        background: None,
                    },
                )
                .map_err(|error| error.to_string())
            },
            &context(),
        )
        .await
        .expect("create context reader");

    harness.resume();
    let receipt = harness
        .wait_for_task(task_id, &context())
        .await
        .expect("task settles");
    assert_eq!(receipt.state.status, TaskStateStatus::Terminal);
    assert_eq!(
        receipt.state.outcome.as_ref().expect("task outcome").status,
        TaskOutcomeStatus::Completed
    );
    assert_eq!(
        *observed.lock(),
        [
            ContextRead {
                entries: vec![first.id, second.id],
                messages: vec!["one".into(), "two".into()]
            },
            ContextRead {
                entries: vec![first.id],
                messages: vec!["one".into()]
            },
            ContextRead {
                entries: vec![first.id, second.id],
                messages: vec!["one".into(), "two".into()]
            },
        ]
    );
    harness.close(&context()).await.expect("close harness");
}

// ─── v1.1.0: range reuse, retention, and system-leading ───────────────────

fn system_with_sections(sections: &[(&str, &str)]) -> Message {
    Message::System(SystemMessage {
        content: SystemContent::Text(String::new()),
        sections: Some(
            sections
                .iter()
                .map(|(key, value)| (key.to_string(), Some(value.to_string())))
                .collect(),
        ),
        tools_added: None,
        tools_removed: None,
        timestamp: 0,
    })
}

fn describe(message: &Message) -> String {
    match message {
        Message::System(system) => match &system.content {
            SystemContent::Text(text)
                if system
                    .sections
                    .as_ref()
                    .is_none_or(|sections| sections.is_empty()) =>
            {
                format!("system:{text}")
            }
            _ => {
                let first = system
                    .sections
                    .iter()
                    .flatten()
                    .next()
                    .map(|(key, _)| key.clone())
                    .unwrap_or_default();
                format!("system:{first}")
            }
        },
        Message::User(user) => format!("user:{}", message_text_user(user)),
        Message::Assistant(assistant) => format!(
            "assistant:{}",
            assistant
                .content
                .iter()
                .filter_map(|block| match block {
                    AssistantContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        ),
        Message::ToolResult(result) => format!("result:{}", result.tool_call_id),
    }
}

fn message_text_user(user: &UserMessage) -> String {
    match &user.content {
        UserContentField::Text(text) => text.clone(),
        UserContentField::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                UserContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

#[tokio::test]
async fn context_leads_with_a_system_message_that_only_user_messages_precede() {
    let (_storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    append_model(&conversation, "message", vec![user("first")]).await;
    append_model(&conversation, "message", vec![user("steered")]).await;
    append_model(
        &conversation,
        "pi.system",
        vec![system_with_sections(&[("preamble", "You help.")])],
    )
    .await;
    append_model(
        &conversation,
        "message",
        vec![assistant("answer", StopReason::Stop)],
    )
    .await;
    append_model(&conversation, "message", vec![user("next")]).await;
    append_model(
        &conversation,
        "pi.system",
        vec![system_with_sections(&[("cwd", "/repo")])],
    )
    .await;
    let view = conversation
        .context(&context(), None)
        .await
        .expect("derive context");
    let described: Vec<String> = view.messages.iter().map(describe).collect();
    assert_eq!(
        described,
        [
            "system:preamble",
            "user:first",
            "user:steered",
            "assistant:answer",
            "user:next",
            "system:cwd"
        ]
    );
    // Stored order and contributions stay as committed.
    let contributed: Vec<String> = view.contributions.iter().flatten().map(describe).collect();
    assert_eq!(
        &contributed[..3],
        ["user:first", "user:steered", "system:preamble"]
    );

    // After a reset, the baseline written after the handoff leads too.
    append(
        &conversation,
        EntryDraft {
            kind: Some("reset".into()),
            head: Some(EntryHead::SelfMarker(crate::types::SelfMarker)),
            model: Some(vec![user("handoff")]),
            ..Default::default()
        },
    )
    .await;
    append_model(
        &conversation,
        "pi.system",
        vec![system_with_sections(&[("preamble", "Baseline.")])],
    )
    .await;
    let view = conversation
        .context(&context(), None)
        .await
        .expect("derive context after reset");
    let described: Vec<String> = view.messages.iter().map(describe).collect();
    assert_eq!(described, ["system:preamble", "user:handoff"]);
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn context_as_of_an_earlier_entry_matches_a_fork_at_that_entry() {
    let (_storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    let first = append_model(&conversation, "message", vec![user("first")]).await;
    let call = append_model(
        &conversation,
        "message",
        vec![assistant_with_calls(
            "calling",
            &[("x", "tool-x"), ("y", "tool-y")],
        )],
    )
    .await;
    let result = append_model(&conversation, "message", vec![tool_result("x", "tool-x")]).await;
    append_model(&conversation, "message", vec![tool_result("y", "tool-y")]).await;
    let edit = append(
        &conversation,
        EntryDraft {
            kind: Some("edit".into()),
            edits: Some(vec![crate::types::ContextEdit {
                target: first.id,
                action: crate::types::ContextEditAction::Replace,
                messages: Some(vec![user("first v2")]),
            }]),
            ..Default::default()
        },
    )
    .await;
    let reset = append(
        &conversation,
        EntryDraft {
            kind: Some("reset".into()),
            head: Some(EntryHead::SelfMarker(crate::types::SelfMarker)),
            model: Some(vec![user("fresh start")]),
            ..Default::default()
        },
    )
    .await;
    let tail = append_model(
        &conversation,
        "message",
        vec![assistant("after reset", StopReason::Stop)],
    )
    .await;

    for at in [first.id, call.id, result.id, edit.id, reset.id, tail.id] {
        let fork = conversation
            .fork(
                at,
                ConversationCreateOptions {
                    ownership: ConversationOwnership::Ownerless,
                    agent: None,
                    init: None,
                },
                &context(),
            )
            .await
            .expect("fork at entry");
        let from_fork = fork.context(&context(), None).await.expect("fork context");
        assert_eq!(
            conversation
                .context(&context(), Some(at))
                .await
                .expect("context at entry"),
            from_fork
        );
    }
    // Stepped back before the edit and the reset: neither applies, and a cut call gets synthesized results.
    let stepped = conversation
        .context(&context(), Some(call.id))
        .await
        .expect("context at call");
    assert_eq!(
        stepped.messages.iter().map(describe).collect::<Vec<_>>(),
        [
            "user:first",
            "assistant:calling",
            "result:x:error",
            "result:y:error"
        ]
    );
    let full = conversation
        .context(&context(), None)
        .await
        .expect("full context");
    assert_eq!(
        conversation
            .context(&context(), Some(tail.id))
            .await
            .expect("context at tail"),
        full
    );
    let other = conversation
        .fork(
            first.id,
            ConversationCreateOptions {
                ownership: ConversationOwnership::Ownerless,
                agent: None,
                init: None,
            },
            &context(),
        )
        .await
        .expect("fork at first");
    let error = other
        .context(&context(), Some(tail.id))
        .await
        .expect_err("cut entry is not visible");
    assert!(error.contains("is not visible"), "{error}");
    harness.close(&context()).await.expect("close harness");
}

/// A storage wrapper over [`MemoryStorage`] that counts scanned entry rows
/// and can hold the next range scan (bounds probes read one row; range scans
/// read more). Upstream test uses a `Proxy` over `MemoryStorage`.
struct CountingStorage {
    inner: Arc<MemoryStorage>,
    rows: std::sync::atomic::AtomicU64,
    hold: parking_lot::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

impl CountingStorage {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(MemoryStorage::new()),
            rows: std::sync::atomic::AtomicU64::new(0),
            hold: parking_lot::Mutex::new(None),
        })
    }

    fn rows(&self) -> u64 {
        self.rows.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn hold_next_scan(&self, hold: tokio::sync::oneshot::Receiver<()>) {
        *self.hold.lock() = Some(hold);
    }
}

fn rows_of(storage: &CountingStorage) -> u64 {
    storage.rows()
}

#[async_trait::async_trait]
impl crate::types::Storage for CountingStorage {
    async fn commit(
        &self,
        writes: &[crate::types::StorageWrite],
        context: &Context,
    ) -> Result<crate::types::Seq, crate::errors::StorageRejected> {
        self.inner.commit(writes, context).await
    }

    async fn mint_id(&self) -> u64 {
        self.inner.mint_id().await
    }

    async fn conversation(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Option<crate::types::ConversationRecord> {
        self.inner.conversation(id, context).await
    }

    async fn scan_conversations(
        &self,
        query: &crate::types::ConversationQuery,
        limit: u64,
        cursor: Option<&crate::types::Cursor>,
        context: &Context,
    ) -> crate::types::Page<crate::types::ConversationRecord> {
        self.inner
            .scan_conversations(query, limit, cursor, context)
            .await
    }

    async fn entry(
        &self,
        id: EntryId,
        context: &Context,
    ) -> Option<(EntryRecord, crate::types::Seq)> {
        self.inner.entry(id, context).await
    }

    async fn entry_in(
        &self,
        conversation_id: ConversationId,
        id: EntryId,
        context: &Context,
    ) -> Option<(EntryRecord, crate::types::Seq)> {
        self.inner.entry_in(conversation_id, id, context).await
    }

    async fn find_latest_head_marker(
        &self,
        conversation_id: ConversationId,
        at_or_before_entry_id: Option<EntryId>,
        context: &Context,
    ) -> Option<EntryRecord> {
        self.inner
            .find_latest_head_marker(conversation_id, at_or_before_entry_id, context)
            .await
    }

    async fn scan_entries(
        &self,
        query: &crate::types::EntryQuery,
        limit: u64,
        cursor: Option<&crate::types::Cursor>,
        context: &Context,
    ) -> crate::types::Page<EntryRecord> {
        // Only range scans wait for the hold; bounds probes read one row.
        let held = if limit > 1 {
            self.hold.lock().take()
        } else {
            None
        };
        if let Some(held) = held {
            let _ = held.await;
        }
        let page = self.inner.scan_entries(query, limit, cursor, context).await;
        self.rows
            .fetch_add(page.items.len() as u64, std::sync::atomic::Ordering::SeqCst);
        page
    }

    async fn task(&self, id: TaskId, context: &Context) -> Option<TaskRecord> {
        self.inner.task(id, context).await
    }

    async fn scan_tasks(
        &self,
        query: &crate::types::TaskQuery,
        limit: u64,
        cursor: Option<&crate::types::Cursor>,
        context: &Context,
    ) -> crate::types::Page<TaskRecord> {
        self.inner.scan_tasks(query, limit, cursor, context).await
    }

    async fn submission(
        &self,
        id: crate::types::SubmissionId,
        context: &Context,
    ) -> Option<crate::types::SubmissionRecord> {
        self.inner.submission(id, context).await
    }

    async fn scan_submissions(
        &self,
        query: &crate::types::SubmissionQuery,
        limit: u64,
        cursor: Option<&crate::types::Cursor>,
        context: &Context,
    ) -> crate::types::Page<crate::types::SubmissionRecord> {
        self.inner
            .scan_submissions(query, limit, cursor, context)
            .await
    }

    async fn submission_by_request(
        &self,
        conversation_id: ConversationId,
        request_id: &str,
        context: &Context,
    ) -> Option<crate::types::SubmissionRecord> {
        self.inner
            .submission_by_request(conversation_id, request_id, context)
            .await
    }

    async fn find_document(
        &self,
        address: &crate::types::DocumentAddress,
        at: crate::types::DocumentPoint,
        context: &Context,
    ) -> Option<crate::types::DocumentRecord> {
        self.inner.find_document(address, at, context).await
    }

    async fn document(
        &self,
        id: crate::types::DocumentId,
        at: crate::types::DocumentPoint,
        context: &Context,
    ) -> Option<crate::types::StoredDocument> {
        self.inner.document(id, at, context).await
    }

    async fn scan_documents(
        &self,
        query: &crate::types::DocumentQuery,
        limit: u64,
        cursor: Option<&crate::types::Cursor>,
        context: &Context,
    ) -> crate::types::Page<crate::types::DocumentRecord> {
        self.inner
            .scan_documents(query, limit, cursor, context)
            .await
    }

    async fn close(&self, context: &Context) {
        self.inner.close(context).await;
    }
}

/// Open a harness over a [`CountingStorage`] with a 21-entry root transcript
/// (upstream `countingSetup`).
async fn counting_setup(
    settings: Option<crate::harness::types::HarnessSettings>,
    now: Option<crate::harness::types::NowFn>,
) -> (
    std::sync::Arc<crate::harness::registry::RegistryImpl>,
    Conversation,
    std::sync::Arc<CountingStorage>,
    EntryRecord,
    std::sync::Arc<crate::harness::harness::Harness>,
) {
    let storage = CountingStorage::new();
    let registry = super::support::registry();
    let options = crate::harness::types::HarnessOptions {
        registry: registry.clone(),
        settings,
        now,
        ..Default::default()
    };
    let harness = super::support::open(storage.clone(), options).await;
    let conversation = root(&harness).await;
    let first = append_model(&conversation, "message", vec![user("first")]).await;
    for index in 0..20 {
        append_model(
            &conversation,
            "message",
            vec![assistant(&format!("old {index}"), StopReason::Stop)],
        )
        .await;
    }
    (registry, conversation, storage, first, harness)
}

fn run_phase(
    f: impl Fn(TaskRecord, Arc<dyn TaskRuntime>, Context) -> BoxFuture<Result<(), String>>
        + Send
        + Sync
        + 'static,
) -> PhaseHandler {
    Arc::new(f)
}

fn simple_task(name: &str, run: PhaseHandler) -> AnyTask {
    AnyTask {
        definition: Arc::new(TaskDefinition {
            name: name.to_string(),
            version: 1,
            initial: Arc::new(|_| json!({"phase": "run"})),
            phases: Arc::new(BTreeMap::from([("run".to_string(), run)])),
            abort: run_phase(|_, runtime, ctx| {
                Box::pin(async move {
                    runtime
                        .commit(
                            commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("aborted")))),
                            &ctx,
                        )
                        .await?;
                    Ok(())
                })
            }),
            migrate: None,
            hooks: None,
        }),
    }
}

/// Read the conversation context through the task runtime and report how
/// many rows the read scanned (upstream `read` inside the Reads task).
async fn counted_read(
    runtime: &Arc<dyn TaskRuntime>,
    conversation_id: ConversationId,
    storage: &Arc<CountingStorage>,
    at: Option<EntryId>,
    ctx: &Context,
) -> (crate::session::ContextView, u64) {
    let before = storage.rows();
    let view = runtime
        .context_view(conversation_id, ctx, at)
        .await
        .expect("task context read");
    (view, storage.rows() - before)
}

#[tokio::test]
async fn task_context_read_extends_with_only_newer_entries() {
    let (registry, conversation, storage, first, harness) = counting_setup(None, None).await;
    let conversation_id = conversation.id;
    let seen = Arc::new(parking_lot::Mutex::new(Vec::<u64>::new()));
    let seen_in = seen.clone();
    let conversation_view = conversation.clone();
    let task = simple_task(
        "test.context-reads",
        run_phase(move |_, runtime, ctx| {
            let conversation = conversation_view.clone();
            let conversation_id = conversation_id;
            let storage = storage.clone();
            let first = first.id;
            let seen = seen_in.clone();
            Box::pin(async move {
                let write = |runtime: Arc<dyn TaskRuntime>, draft: EntryDraft, ctx: Context| {
                    Box::pin(async move {
                        runtime
                            .commit(
                                commit_change(move |tx, _| {
                                    tx.append_entry(conversation_id, draft)
                                        .map_err(|error| error.to_string())?;
                                    Ok(None)
                                }),
                                &ctx,
                            )
                            .await?;
                        Ok::<(), String>(())
                    })
                        as std::pin::Pin<
                            Box<dyn std::future::Future<Output = Result<(), String>> + Send>,
                        >
                };
                let (initial, _initial_rows) =
                    counted_read(&runtime, conversation_id, &storage, None, &ctx).await;
                assert_eq!(initial, conversation.context(&ctx, None).await.unwrap());
                // Three new entries: the bounds probe reads one row, the extension the three new ones.
                write(
                    runtime.clone(),
                    EntryDraft {
                        kind: Some("message".into()),
                        model: Some(vec![user("new")]),
                        ..Default::default()
                    },
                    ctx.clone(),
                )
                .await?;
                write(
                    runtime.clone(),
                    EntryDraft {
                        kind: Some("note".into()),
                        data: Some(json!({"text": "display only"})),
                        ..Default::default()
                    },
                    ctx.clone(),
                )
                .await?;
                write(
                    runtime.clone(),
                    EntryDraft {
                        kind: Some("message".into()),
                        model: Some(vec![assistant("answer", StopReason::Stop)]),
                        ..Default::default()
                    },
                    ctx.clone(),
                )
                .await?;
                let (extended, extended_rows) =
                    counted_read(&runtime, conversation_id, &storage, None, &ctx).await;
                assert_eq!(extended_rows, 4);
                assert_eq!(extended, conversation.context(&ctx, None).await.unwrap());

                // A newer edit of an older entry applies to the extended range.
                write(
                    runtime.clone(),
                    EntryDraft {
                        kind: Some("edit".into()),
                        edits: Some(vec![crate::types::ContextEdit {
                            target: first,
                            action: crate::types::ContextEditAction::Omit,
                            messages: None,
                        }]),
                        ..Default::default()
                    },
                    ctx.clone(),
                )
                .await?;
                let (edited, edited_rows) =
                    counted_read(&runtime, conversation_id, &storage, None, &ctx).await;
                assert_eq!(edited_rows, 2);
                assert_eq!(edited, conversation.context(&ctx, None).await.unwrap());
                assert!(!edited.messages.iter().any(|message| matches!(message, Message::User(user) if message_text_user(&user) == "first")));

                // An earlier cutoff reuses the range.
                let cutoff_at = extended.entries.last().unwrap().id;
                let (cutoff, cutoff_rows) =
                    counted_read(&runtime, conversation_id, &storage, Some(cutoff_at), &ctx).await;
                assert_eq!(cutoff_rows, 0);
                assert_eq!(cutoff, extended);

                // A new head marker changes the range: read it whole.
                write(
                    runtime.clone(),
                    EntryDraft {
                        kind: Some("reset".into()),
                        head: Some(EntryHead::SelfMarker(crate::types::SelfMarker)),
                        model: Some(vec![user("fresh")]),
                        ..Default::default()
                    },
                    ctx.clone(),
                )
                .await?;
                let (reset, _reset_rows) =
                    counted_read(&runtime, conversation_id, &storage, None, &ctx).await;
                assert_eq!(reset, conversation.context(&ctx, None).await.unwrap());
                assert_eq!(
                    reset.messages.iter().map(describe).collect::<Vec<_>>(),
                    ["user:fresh"]
                );
                seen.lock().push(extended_rows + edited_rows + cutoff_rows);
                runtime
                    .commit(
                        commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                        &ctx,
                    )
                    .await?;
                Ok(())
            })
        }),
    );
    registry.install(Extension::new("context-cache-tests").with_tasks(vec![task.clone()]));
    let task_id = conversation
        .commit(
            move |tx| {
                tx.create_task(
                    &task,
                    Value::Null,
                    TaskOptions {
                        ownership: TaskOwnership::Conversation,
                        conversation_id: None,
                        background: None,
                    },
                )
                .map_err(|error| error.to_string())
            },
            &context(),
        )
        .await
        .expect("create context-reads task");
    harness.resume();
    let receipt = harness
        .wait_for_task(task_id, &context())
        .await
        .expect("task settles");
    assert_eq!(
        receipt.state.outcome.as_ref().expect("outcome").status,
        TaskOutcomeStatus::Completed
    );
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn keeps_a_conversation_context_read_across_tasks_and_for_the_retention_period_once_idle() {
    let clock = Arc::new(parking_lot::Mutex::new(1_000i64));
    let clock_now = clock.clone();
    let settings = crate::harness::types::HarnessSettings {
        context_retention_ms: Some(600_000),
        ..Default::default()
    };
    let (registry, conversation, storage, _first, harness) =
        counting_setup(Some(settings), Some(Arc::new(move || *clock_now.lock()))).await;
    let conversation_id = conversation.id;
    let rows = Arc::new(parking_lot::Mutex::new(BTreeMap::<String, u64>::new()));

    // Child: writes one entry, then reads.
    let child_storage = storage.clone();
    let child_rows = rows.clone();
    let child = simple_task(
        "test.context-child",
        run_phase(move |_, runtime, ctx| {
            let storage = child_storage.clone();
            let rows = child_rows.clone();
            let conversation_id = conversation_id;
            Box::pin(async move {
                runtime
                    .commit(
                        commit_change(move |tx, _| {
                            tx.append_entry(
                                conversation_id,
                                EntryDraft {
                                    kind: Some("message".into()),
                                    model: Some(vec![user("from child")]),
                                    ..Default::default()
                                },
                            )
                            .map_err(|error| error.to_string())?;
                            Ok(None)
                        }),
                        &ctx,
                    )
                    .await?;
                let before = storage.rows();
                runtime.context_view(conversation_id, &ctx, None).await?;
                rows.lock().insert("child".into(), storage.rows() - before);
                runtime
                    .commit(
                        commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                        &ctx,
                    )
                    .await?;
                Ok(())
            })
        }),
    );

    // Parent: reads, then waits for the child, then reads again.
    let parent_start_storage = storage.clone();
    let parent_start_rows = rows.clone();
    let parent_after_storage = storage.clone();
    let parent_after_rows = rows.clone();
    let parent = {
        let child = child.clone();
        AnyTask {
            definition: Arc::new(TaskDefinition {
                name: "test.context-parent".into(),
                version: 1,
                initial: Arc::new(|_| json!({"phase": "start"})),
                phases: Arc::new(BTreeMap::from([
                    (
                        "start".to_string(),
                        run_phase(move |record, runtime, ctx| {
                            let storage = parent_start_storage.clone();
                            let rows = parent_start_rows.clone();
                            let conversation_id = conversation_id;
                            let child = child.clone();
                            let parent_task_id = record.id;
                            Box::pin(async move {
                                let before = storage.rows();
                                runtime.context_view(conversation_id, &ctx, None).await?;
                                rows.lock().insert("parent".into(), storage.rows() - before);
                                runtime
                                    .commit(
                                        commit_change(move |tx, _| {
                                            let child_id = tx
                                                .create_task(
                                                    &child,
                                                    Value::Null,
                                                    TaskOptions {
                                                        ownership: TaskOwnership::Task {
                                                            task_id: parent_task_id,
                                                        },
                                                        conversation_id: Some(conversation_id),
                                                        background: None,
                                                    },
                                                )
                                                .map_err(|error| error.to_string())?;
                                            Ok(Some(NextTaskState::Waiting {
                                                checkpoint: json!({"phase": "after"}),
                                                on: vec![child_id],
                                                policy: crate::types::JoinPolicy::AllSettled,
                                            }))
                                        }),
                                        &ctx,
                                    )
                                    .await?;
                                Ok(())
                            })
                        }),
                    ),
                    (
                        "after".to_string(),
                        run_phase(move |_, runtime, ctx| {
                            let storage = parent_after_storage.clone();
                            let rows = parent_after_rows.clone();
                            let conversation_id = conversation_id;
                            Box::pin(async move {
                                let before = storage.rows();
                                runtime.context_view(conversation_id, &ctx, None).await?;
                                rows.lock().insert("after".into(), storage.rows() - before);
                                runtime
                                    .commit(
                                        commit_change(|_, _| {
                                            Ok(Some(NextTaskState::completed(Value::Null)))
                                        }),
                                        &ctx,
                                    )
                                    .await?;
                                Ok(())
                            })
                        }),
                    ),
                ])),
                abort: run_phase(|_, runtime, ctx| {
                    Box::pin(async move {
                        runtime
                            .commit(
                                commit_change(|_, _| {
                                    Ok(Some(NextTaskState::aborted_with("aborted")))
                                }),
                                &ctx,
                            )
                            .await?;
                        Ok(())
                    })
                }),
                migrate: None,
                hooks: None,
            }),
        }
    };

    // Probe: reads its conversation's context once (upstream `probeTask`).
    let probe_storage = storage.clone();
    let probe_rows = rows.clone();
    let probe = simple_task(
        "test.context-probe",
        run_phase(move |task, runtime, ctx| {
            let storage = probe_storage.clone();
            let rows = probe_rows.clone();
            let conversation_id = conversation_id;
            Box::pin(async move {
                let name = task.input.as_str().unwrap_or("probe").to_string();
                let before = storage.rows();
                runtime.context_view(conversation_id, &ctx, None).await?;
                rows.lock().insert(name, storage.rows() - before);
                runtime
                    .commit(
                        commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                        &ctx,
                    )
                    .await?;
                Ok(())
            })
        }),
    );

    registry.install(Extension::new("context-retention-tests").with_tasks(vec![
        child.clone(),
        parent.clone(),
        probe.clone(),
    ]));
    let parent_id = conversation
        .commit(
            {
                let parent = parent.clone();
                move |tx| {
                    tx.create_task(
                        &parent,
                        Value::Null,
                        TaskOptions {
                            ownership: TaskOwnership::Conversation,
                            conversation_id: None,
                            background: None,
                        },
                    )
                    .map_err(|error| error.to_string())
                }
            },
            &context(),
        )
        .await
        .expect("create parent");
    harness.resume();
    let receipt = harness
        .wait_for_task(parent_id, &context())
        .await
        .expect("parent settles");
    assert_eq!(
        receipt.state.outcome.as_ref().expect("outcome").status,
        TaskOutcomeStatus::Completed
    );
    conversation
        .wait_for_idle(&context())
        .await
        .expect("conversation goes idle");

    let run_probe = |name: &'static str,
                     probe: AnyTask,
                     conversation: Conversation,
                     harness: Arc<crate::harness::harness::Harness>| {
        Box::pin(async move {
            let id = conversation
                .commit(
                    move |tx| {
                        tx.create_task(
                            &probe,
                            json!(name),
                            TaskOptions {
                                ownership: TaskOwnership::Conversation,
                                conversation_id: None,
                                background: None,
                            },
                        )
                        .map_err(|error| error.to_string())
                    },
                    &context(),
                )
                .await
                .expect("create probe");
            let receipt = harness
                .wait_for_task(id, &context())
                .await
                .expect("probe settles");
            assert_eq!(
                receipt.state.outcome.as_ref().expect("outcome").status,
                TaskOutcomeStatus::Completed
            );
            conversation
                .wait_for_idle(&context())
                .await
                .expect("conversation goes idle");
        })
    };
    run_probe("idle", probe.clone(), conversation.clone(), harness.clone()).await;
    *clock.lock() += 600_000;
    run_probe(
        "expired",
        probe.clone(),
        conversation.clone(),
        harness.clone(),
    )
    .await;
    // Bounds probes read one row each. The child scans its own new entry; the waiting parent's later invocation and
    // the probe within the retention period scan nothing new. After it, the probe reads the whole transcript again.
    let rows = rows.lock().clone();
    assert_eq!(
        rows,
        BTreeMap::from([
            ("parent".into(), 22),
            ("child".into(), 2),
            ("after".into(), 1),
            ("idle".into(), 1),
            ("expired".into(), 23)
        ])
    );
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn drops_a_conversation_context_read_once_idle_with_zero_retention() {
    let settings = crate::harness::types::HarnessSettings {
        context_retention_ms: Some(0),
        ..Default::default()
    };
    let (registry, conversation, storage, _first, harness) =
        counting_setup(Some(settings), None).await;
    let conversation_id = conversation.id;
    let rows = Arc::new(parking_lot::Mutex::new(BTreeMap::<String, u64>::new()));
    let probe_storage = storage.clone();
    let probe_rows = rows.clone();
    let probe = simple_task(
        "test.context-probe",
        run_phase(move |task, runtime, ctx| {
            let storage = probe_storage.clone();
            let rows = probe_rows.clone();
            let conversation_id = conversation_id;
            Box::pin(async move {
                let name = task.input.as_str().unwrap_or("probe").to_string();
                let before = storage.rows();
                runtime.context_view(conversation_id, &ctx, None).await?;
                rows.lock().insert(name, storage.rows() - before);
                runtime
                    .commit(
                        commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                        &ctx,
                    )
                    .await?;
                Ok(())
            })
        }),
    );
    registry
        .install(Extension::new("context-zero-retention-tests").with_tasks(vec![probe.clone()]));
    harness.resume();
    for name in ["first", "second"] {
        let id = conversation
            .commit(
                {
                    let probe = probe.clone();
                    move |tx| {
                        tx.create_task(
                            &probe,
                            json!(name),
                            TaskOptions {
                                ownership: TaskOwnership::Conversation,
                                conversation_id: None,
                                background: None,
                            },
                        )
                        .map_err(|error| error.to_string())
                    }
                },
                &context(),
            )
            .await
            .expect("create probe");
        harness
            .wait_for_task(id, &context())
            .await
            .expect("probe settles");
        conversation
            .wait_for_idle(&context())
            .await
            .expect("conversation goes idle");
    }
    assert_eq!(
        *rows.lock(),
        BTreeMap::from([("first".into(), 22), ("second".into(), 22)])
    );
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn drops_an_idle_conversation_context_read_after_the_retention_period() {
    let settings = crate::harness::types::HarnessSettings {
        context_retention_ms: Some(300),
        ..Default::default()
    };
    let (registry, conversation, storage, _first, harness) =
        counting_setup(Some(settings), None).await;
    let conversation_id = conversation.id;
    let rows = Arc::new(parking_lot::Mutex::new(BTreeMap::<String, u64>::new()));
    let probe_storage = storage.clone();
    let probe_rows = rows.clone();
    let probe = simple_task(
        "test.context-probe",
        run_phase(move |task, runtime, ctx| {
            let storage = probe_storage.clone();
            let rows = probe_rows.clone();
            let conversation_id = conversation_id;
            Box::pin(async move {
                let name = task.input.as_str().unwrap_or("probe").to_string();
                let before = storage.rows();
                runtime.context_view(conversation_id, &ctx, None).await?;
                rows.lock().insert(name, storage.rows() - before);
                runtime
                    .commit(
                        commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                        &ctx,
                    )
                    .await?;
                Ok(())
            })
        }),
    );
    registry.install(Extension::new("context-expiry-tests").with_tasks(vec![probe.clone()]));
    harness.resume();
    let run_probe = |name: &'static str, probe: AnyTask| {
        let conversation = conversation.clone();
        let harness = harness.clone();
        Box::pin(async move {
            let id = conversation
                .commit(
                    move |tx| {
                        tx.create_task(
                            &probe,
                            json!(name),
                            TaskOptions {
                                ownership: TaskOwnership::Conversation,
                                conversation_id: None,
                                background: None,
                            },
                        )
                        .map_err(|error| error.to_string())
                    },
                    &context(),
                )
                .await
                .expect("create probe");
            harness
                .wait_for_task(id, &context())
                .await
                .expect("probe settles");
            conversation
                .wait_for_idle(&context())
                .await
                .expect("conversation goes idle");
        })
    };
    run_probe("first", probe.clone()).await;
    // The expiry task fires while nothing else runs; the next read scans whole again.
    tokio::time::sleep(std::time::Duration::from_millis(600)).await;
    run_probe("second", probe.clone()).await;
    assert_eq!(
        *rows.lock(),
        BTreeMap::from([("first".into(), 22), ("second".into(), 22)])
    );
    harness.close(&context()).await.expect("close harness");
}

/// A read that finishes after its task ended keeps no range (upstream
/// `does not keep a read that finishes after its task ended`).
#[tokio::test]
async fn does_not_keep_a_read_that_finishes_after_its_task_ended() {
    let settings = crate::harness::types::HarnessSettings {
        context_retention_ms: Some(0),
        ..Default::default()
    };
    let (registry, conversation, storage, _first, harness) =
        counting_setup(Some(settings), None).await;
    let conversation_id = conversation.id;
    let (scan_release_tx, scan_release) = tokio::sync::oneshot::channel::<()>();
    let probe_started = Arc::new(tokio::sync::Notify::new());
    // Park the next range scan until the probe task is running again.
    storage.hold_next_scan(scan_release);
    let late_handle: Arc<parking_lot::Mutex<Option<tokio::task::JoinHandle<()>>>> = Arc::default();

    let late_storage = storage.clone();
    let late_handle_in = late_handle.clone();
    let late = simple_task(
        "test.context-late",
        run_phase(move |_, runtime, ctx| {
            let storage = late_storage.clone();
            let late_handle = late_handle_in.clone();
            let conversation_id = conversation_id;
            Box::pin(async move {
                // The read starts but its scan is parked; the task settles first.
                let read = {
                    let runtime = runtime.clone();
                    let ctx = ctx.clone();
                    tokio::spawn(async move {
                        let _ = runtime.context_view(conversation_id, &ctx, None).await;
                    })
                };
                *late_handle.lock() = Some(read);
                runtime
                    .commit(
                        commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                        &ctx,
                    )
                    .await?;
                Ok(())
            })
        }),
    );
    registry.install(Extension::new("context-late-tests").with_tasks(vec![
        late.clone(),
        simple_task(
            "test.noop",
            run_phase(|_, runtime, ctx| {
                Box::pin(async move {
                    runtime
                        .commit(
                            commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                            &ctx,
                        )
                        .await?;
                    Ok(())
                })
            }),
        ),
    ]));
    harness.resume();
    let create = |task: AnyTask, input: Value| {
        let conversation = conversation.clone();
        Box::pin(async move {
            conversation
                .commit(
                    move |tx| {
                        tx.create_task(
                            &task,
                            input,
                            TaskOptions {
                                ownership: TaskOwnership::Conversation,
                                conversation_id: None,
                                background: None,
                            },
                        )
                        .map_err(|error| error.to_string())
                    },
                    &context(),
                )
                .await
                .expect("create task")
        })
    };
    let late_id = create(late.clone(), Value::Null).await;
    harness
        .wait_for_task(late_id, &context())
        .await
        .expect("late task settles");
    let _ = scan_release_tx.send(());
    // The late read finishes before the probe starts; its range is not kept.
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let handle = late_handle.lock().take();
            if let Some(handle) = handle {
                let _ = handle.await;
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("late read finishes");
    let rows_before = storage.rows();
    let probe_started_in = probe_started.clone();
    let probe_id = create(
        simple_task(
            "test.probe",
            run_phase(move |_, runtime, ctx| {
                let probe_started = probe_started_in.clone();
                Box::pin(async move {
                    probe_started.notify_one();
                    runtime
                        .commit(
                            commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                            &ctx,
                        )
                        .await?;
                    Ok(())
                })
            }),
        ),
        Value::Null,
    )
    .await;
    probe_started.notified().await;
    harness
        .wait_for_task(probe_id, &context())
        .await
        .expect("probe settles");
    // The late read kept nothing (retention 0 and the conversation was busy again when it landed): the probe read whole.
    assert_eq!(storage.rows() - rows_before, 22);
    harness.close(&context()).await.expect("close harness");
}

/// Extended reads derive the same views as whole reads under a
/// deterministic pseudo-random transcript (upstream LCG test).
#[tokio::test]
async fn derives_the_same_view_from_extended_reads_as_from_whole_reads() {
    let (registry, conversation, _storage, _first, harness) = counting_setup(None, None).await;
    let conversation_id = conversation.id;
    let seed = Arc::new(parking_lot::Mutex::new(7u32));
    let conversation_view = conversation.clone();
    let task = simple_task(
        "test.context-steps",
        run_phase(move |_, runtime, ctx| {
            let conversation = conversation_view.clone();
            let conversation_id = conversation_id;
            let seed = seed.clone();
            Box::pin(async move {
                let random = |seed: &parking_lot::Mutex<u32>, n: u32| {
                    let mut state = seed.lock();
                    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((*state as u64) * (n as u64) >> 32) as u32
                };
                let written: Arc<parking_lot::Mutex<Vec<EntryRecord>>> = Arc::default();
                let calls: Arc<parking_lot::Mutex<Vec<String>>> = Arc::default();
                let next = Arc::new(parking_lot::Mutex::new(0u32));
                let mut step = 0;
                while step < 60 {
                    step += 1;
                    // Append 1..=3 entries of mixed kinds.
                    let drafts = {
                        let next_count = {
                            let mut next = next.lock();
                            let value = *next;
                            *next = value + 1;
                            value
                        };
                        let _ = next_count;
                        let mut drafts: Vec<EntryDraft> = Vec::new();
                        let added = 1 + random(&seed, 3);
                        for _ in 0..added {
                            let pick = random(&seed, 14);
                            let mut next_value = next.lock();
                            let draft = if pick < 3 {
                                let name = format!("u{}", *next_value);
                                *next_value += 1;
                                EntryDraft {
                                    kind: Some("message".into()),
                                    model: Some(vec![user(&name)]),
                                    ..Default::default()
                                }
                            } else if pick < 5 {
                                let ids: Vec<String> = (0..random(&seed, 3))
                                    .map(|_| {
                                        let id = format!("c{}", *next_value);
                                        *next_value += 1;
                                        id
                                    })
                                    .collect();
                                calls.lock().extend(ids.iter().cloned());
                                let stop = match random(&seed, 9) {
                                    0 => Some(StopReason::Aborted),
                                    1 => Some(StopReason::Error),
                                    2 => Some(StopReason::Deferred),
                                    _ => None,
                                };
                                let mut message = match assistant(
                                    &format!("a{}", *next_value),
                                    StopReason::ToolUse,
                                ) {
                                    Message::Assistant(message) => message,
                                    _ => unreachable!(),
                                };
                                *next_value += 1;
                                message.stop_reason = stop.unwrap_or(StopReason::Stop);
                                message.content.extend(ids.iter().map(|id| {
                                    AssistantContent::ToolCall(ToolCall {
                                        id: id.clone(),
                                        name: "tool".into(),
                                        arguments: Default::default(),
                                        thought_signature: None,
                                        namespace: None,
                                    })
                                }));
                                EntryDraft {
                                    kind: Some("message".into()),
                                    model: Some(vec![Message::Assistant(message)]),
                                    ..Default::default()
                                }
                            } else if pick < 8 {
                                let calls_len = calls.lock().len();
                                let matched = calls_len > 0 && random(&seed, 5) > 0;
                                let id = if matched {
                                    let at = random(&seed, calls_len as u32) as usize;
                                    calls.lock().remove(at)
                                } else {
                                    format!("stray{}", {
                                        let v = *next_value;
                                        *next_value += 1;
                                        v
                                    })
                                };
                                EntryDraft {
                                    kind: Some("message".into()),
                                    model: Some(vec![tool_result(&id, "tool")]),
                                    ..Default::default()
                                }
                            } else if pick < 9 {
                                let name = format!("s{}", {
                                    let v = *next_value;
                                    *next_value += 1;
                                    v
                                });
                                EntryDraft {
                                    kind: Some("message".into()),
                                    model: Some(vec![system(&name)]),
                                    ..Default::default()
                                }
                            } else if pick < 10 {
                                let value = {
                                    let v = *next_value;
                                    *next_value += 1;
                                    v
                                };
                                EntryDraft {
                                    kind: Some("note".into()),
                                    data: Some(json!({"n": value})),
                                    ..Default::default()
                                }
                            } else {
                                let written_len = written.lock().len();
                                let target = if written_len > 0 {
                                    written
                                        .lock()
                                        .get(random(&seed, written_len as u32) as usize)
                                        .map(|entry| entry.id)
                                } else {
                                    None
                                };
                                if pick < 12 {
                                    match target {
                                        Some(target) => {
                                            let omit = random(&seed, 2) == 0;
                                            let (action, messages) = if omit {
                                                (crate::types::ContextEditAction::Omit, None)
                                            } else {
                                                let name = format!("r{}", *next_value);
                                                *next_value += 1;
                                                (
                                                    crate::types::ContextEditAction::Replace,
                                                    Some(vec![user(&name)]),
                                                )
                                            };
                                            EntryDraft {
                                                kind: Some("edit".into()),
                                                edits: Some(vec![crate::types::ContextEdit {
                                                    target,
                                                    action,
                                                    messages,
                                                }]),
                                                ..Default::default()
                                            }
                                        }
                                        None => EntryDraft {
                                            kind: Some("message".into()),
                                            model: Some(vec![user("fallback")]),
                                            ..Default::default()
                                        },
                                    }
                                } else if pick < 13 {
                                    match target {
                                        Some(target) => {
                                            let name = format!("h{}", *next_value);
                                            *next_value += 1;
                                            EntryDraft {
                                                kind: Some("summary".into()),
                                                head: Some(EntryHead::Entry(target)),
                                                model: Some(vec![user(&name)]),
                                                ..Default::default()
                                            }
                                        }
                                        None => EntryDraft {
                                            kind: Some("message".into()),
                                            model: Some(vec![user("fallback")]),
                                            ..Default::default()
                                        },
                                    }
                                } else {
                                    let name = format!("u{}", *next_value);
                                    *next_value += 1;
                                    EntryDraft {
                                        kind: Some("message".into()),
                                        model: Some(vec![user(&name)]),
                                        ..Default::default()
                                    }
                                }
                            };
                            drafts.push(draft);
                        }
                        drafts
                    };
                    runtime
                        .commit(
                            {
                                let written = written.clone();
                                commit_change(move |tx, _| {
                                    for draft in drafts {
                                        let record = tx
                                            .append_entry(conversation_id, draft)
                                            .map_err(|error| error.to_string())?;
                                        written.lock().push(record);
                                    }
                                    Ok(None)
                                })
                            },
                            &ctx,
                        )
                        .await?;
                    assert_eq!(
                        runtime.context_view(conversation_id, &ctx, None).await?,
                        conversation.context(&ctx, None).await.unwrap()
                    );
                }
                runtime
                    .commit(
                        commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                        &ctx,
                    )
                    .await?;
                Ok(())
            })
        }),
    );
    registry.install(Extension::new("context-steps-tests").with_tasks(vec![task.clone()]));
    let id = conversation
        .commit(
            move |tx| {
                tx.create_task(
                    &task,
                    Value::Null,
                    TaskOptions {
                        ownership: TaskOwnership::Conversation,
                        conversation_id: None,
                        background: None,
                    },
                )
                .map_err(|error| error.to_string())
            },
            &context(),
        )
        .await
        .expect("create steps task");
    harness.resume();
    let receipt = harness
        .wait_for_task(id, &context())
        .await
        .expect("steps task settles");
    assert_eq!(
        receipt.state.outcome.as_ref().expect("outcome").status,
        TaskOutcomeStatus::Completed
    );
    harness.close(&context()).await.expect("close harness");
}

/// A returned view owns its arrays; mutating it never changes the kept range
/// or later reads (upstream `keeps returned views and kept ranges apart`;
/// freeze assertions are D3).
#[tokio::test]
async fn keeps_returned_views_and_kept_ranges_apart() {
    let (registry, conversation, _storage, _first, harness) = counting_setup(None, None).await;
    let conversation_id = conversation.id;
    let conversation_view = conversation.clone();
    let task = simple_task(
        "test.context-mutate",
        run_phase(move |_, runtime, ctx| {
            let conversation = conversation_view.clone();
            let conversation_id = conversation_id;
            Box::pin(async move {
                let mut view = runtime.context_view(conversation_id, &ctx, None).await?;
                view.messages.clear();
                view.entries.pop();
                assert_eq!(
                    runtime.context_view(conversation_id, &ctx, None).await?,
                    conversation.context(&ctx, None).await.unwrap()
                );
                runtime
                    .commit(
                        commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                        &ctx,
                    )
                    .await?;
                Ok(())
            })
        }),
    );
    registry.install(Extension::new("context-mutate-tests").with_tasks(vec![task.clone()]));
    let id = conversation
        .commit(
            move |tx| {
                tx.create_task(
                    &task,
                    Value::Null,
                    TaskOptions {
                        ownership: TaskOwnership::Conversation,
                        conversation_id: None,
                        background: None,
                    },
                )
                .map_err(|error| error.to_string())
            },
            &context(),
        )
        .await
        .expect("create mutate task");
    harness.resume();
    let receipt = harness
        .wait_for_task(id, &context())
        .await
        .expect("mutate task settles");
    assert_eq!(
        receipt.state.outcome.as_ref().expect("outcome").status,
        TaskOutcomeStatus::Completed
    );
    harness.close(&context()).await.expect("close harness");
}
