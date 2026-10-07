use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;
use pi_ai::types::{
	AssistantContent, AssistantMessage, Message, StopReason, SystemContent, SystemMessage, TextContent,
	ToolCall, ToolResultMessage, UserContent, UserContentField, UserMessage,
};
use pi_chord::context::Context;
use serde_json::{Value, json};

use super::support::{context, open_default, root};
use crate::harness::harness::Conversation;
use crate::harness::types::{AnyTask, ConversationCreateOptions, Extension, Registry};
use crate::tasks::{BoxFuture, NextTaskState, PhaseHandler, TaskDefinition, TaskRuntime, commit_change, define_task};
use crate::types::{
	ConversationOwnership, EntryDraft, EntryHead, EntryId, EntryRecord, TaskOptions, TaskOutcomeStatus,
	TaskOwnership, TaskRecord, TaskStateStatus,
};

fn user(text: &str) -> Message {
	Message::User(UserMessage { content: UserContentField::Text(text.to_owned()), timestamp: 0 })
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
		timestamp: 0,
	})
}

async fn append(conversation: &Conversation, draft: EntryDraft) -> EntryRecord {
	let conversation_id = conversation.id;
	conversation
		.commit(
			move |tx| tx.append_entry(conversation_id, draft).map_err(|error| error.to_string()),
			&context(),
		)
		.await
		.expect("append entry")
}

async fn append_model(conversation: &Conversation, kind: &str, model: Vec<Message>) -> EntryRecord {
	append(conversation, EntryDraft { kind: Some(kind.to_owned()), model: Some(model), ..Default::default() }).await
}

#[tokio::test]
async fn context_excludes_deferred_assistant_messages_but_retains_the_raw_entry() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let question = append_model(&conversation, "message", vec![user("question")]).await;
	let deferred = append_model(&conversation, "message", vec![assistant("pending", StopReason::Deferred)]).await;
	let answer = append_model(&conversation, "message", vec![assistant("ready", StopReason::Stop)]).await;

	let view = conversation.context(&context()).await.expect("derive context");
	assert!(view.head.is_none());
	assert_eq!(view.entries.iter().map(|entry| entry.id).collect::<Vec<_>>(), [question.id, deferred.id, answer.id]);
	assert_eq!(view.messages.len(), 2);
	assert!(matches!(&view.messages[0], Message::User(message) if message.content == UserContentField::Text("question".into())));
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
	let kept = append_model(&conversation, "message", vec![assistant("after reset", StopReason::Stop)]).await;
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

	let view = conversation.context(&context()).await.expect("derive summarized context");
	assert_eq!(view.head.as_ref().map(|entry| entry.id), Some(summary.id));
	assert_eq!(view.entries.iter().map(|entry| entry.id).collect::<Vec<_>>(), [summary.id, kept.id, tail.id]);
	assert_eq!(view.messages.len(), 3);
	assert!(matches!(&view.messages[0], Message::User(message) if message.content == UserContentField::Text("summary".into())));
	assert!(matches!(&view.messages[1], Message::Assistant(message)
		if message.content.iter().any(|content| matches!(content, AssistantContent::Text(text) if text.text == "after reset"))));
	assert!(matches!(&view.messages[2], Message::User(message) if message.content == UserContentField::Text("next".into())));
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

	let view = conversation.context(&context()).await.expect("derive context");
	assert_eq!(view.entries.iter().map(|entry| entry.id).collect::<Vec<_>>(), [before.id, first_user.id, middle.id, second_user.id]);
	assert_eq!(view.messages.len(), 4);
	assert!(matches!(&view.messages[0], Message::System(message) if matches!(&message.content, SystemContent::Text(text) if text == "You help.")));
	assert!(matches!(&view.messages[1], Message::User(message) if message.content == UserContentField::Text("first".into())));
	assert!(matches!(&view.messages[2], Message::System(message) if matches!(&message.content, SystemContent::Text(text) if text == "cwd: /repo")));
	assert!(matches!(&view.messages[3], Message::User(message) if message.content == UserContentField::Text("second".into())));
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn fork_context_hides_later_entries_synthesizes_missing_results_and_drops_cut_results() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let input = append_model(&conversation, "message", vec![user("go")]).await;
	let call = append_model(&conversation, "message", vec![assistant_with_calls("calling", &[("x", "tool-x"), ("y", "tool-y")])]).await;
	append_model(&conversation, "message", vec![tool_result("x", "tool-x")]).await;
	let second_result = append_model(&conversation, "message", vec![tool_result("y", "tool-y")]).await;

	let fork = conversation
		.fork(
			call.id,
			ConversationCreateOptions { ownership: ConversationOwnership::Ownerless, agent: None, init: None },
			&context(),
		)
		.await
		.expect("fork at tool call");
	let fork_view = fork.context(&context()).await.expect("read fork context");
	assert_eq!(fork_view.entries.iter().map(|entry| entry.id).collect::<Vec<_>>(), [input.id, call.id]);
	assert_eq!(fork_view.messages.len(), 4);
	assert!(matches!(&fork_view.messages[0], Message::User(message) if message.content == UserContentField::Text("go".into())));
	assert!(matches!(&fork_view.messages[1], Message::Assistant(message)
		if message.content.iter().any(|content| matches!(content, AssistantContent::Text(text) if text.text == "calling"))));
	for (message, (expected_id, expected_name)) in fork_view.messages[2..].iter().zip([("x", "tool-x"), ("y", "tool-y")]) {
		let Message::ToolResult(result) = message else { panic!("missing tool result was not synthesized") };
		assert_eq!(result.tool_call_id, expected_id);
		assert_eq!(result.tool_name, expected_name);
		assert!(result.is_error);
		assert_eq!(result.details, Some(json!({"reason": "missing_result"})));
		assert_eq!(result.content, [UserContent::Text(TextContent::new("Tool result unavailable: history ends before this call completed."))]);
	}

	let reset = append(
		&conversation,
		EntryDraft { kind: Some("reset".into()), head: Some(EntryHead::Entry(second_result.id)), ..Default::default() },
	)
	.await;
	let parent_view = conversation.context(&context()).await.expect("read parent context after cut");
	assert!(parent_view.messages.is_empty());
	assert_eq!(parent_view.entries.iter().map(|entry| entry.id).collect::<Vec<_>>(), [reset.id, second_result.id]);
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
		Arc::new(move |_record: TaskRecord, runtime: Arc<dyn TaskRuntime>, ctx: Context| {
			let observed = observed.clone();
			Box::pin(async move {
				let full = runtime.context_view(conversation_id, &ctx, None).await?;
				let first_id = full.entries.first().ok_or_else(|| "first context entry missing".to_owned())?.id;
				let second_id = full.entries.get(1).ok_or_else(|| "second context entry missing".to_owned())?.id;
				let through_first = runtime.context_view(conversation_id, &ctx, Some(first_id)).await?;
				let through_second = runtime.context_view(conversation_id, &ctx, Some(second_id)).await?;
				observed.lock().extend([
					context_read(full),
					context_read(through_first),
					context_read(through_second),
				]);
				runtime
					.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))), &ctx)
					.await?;
				Ok(())
			}) as BoxFuture<Result<(), String>>
		})
	};
	let abort: PhaseHandler = Arc::new(|_record, runtime, ctx| {
		Box::pin(async move {
			runtime
				.commit(commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("stopped")))), &ctx)
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
	registry.install(Extension::new("runtime-context-tests").with_tasks(vec![AnyTask {
		definition: reader.definition.clone(),
	}]));
	let task_definition = reader.definition;
	let task_id = conversation
		.commit(
			move |tx| {
				tx.create_task(
					&AnyTask { definition: task_definition },
					Value::Null,
					TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
				)
				.map_err(|error| error.to_string())
			},
			&context(),
		)
		.await
		.expect("create context reader");

	harness.resume();
	let receipt = harness.wait_for_task(task_id, &context()).await.expect("task settles");
	assert_eq!(receipt.state.status, TaskStateStatus::Terminal);
	assert_eq!(receipt.state.outcome.as_ref().expect("task outcome").status, TaskOutcomeStatus::Completed);
	assert_eq!(
		*observed.lock(),
		[
			ContextRead { entries: vec![first.id, second.id], messages: vec!["one".into(), "two".into()] },
			ContextRead { entries: vec![first.id], messages: vec!["one".into()] },
			ContextRead { entries: vec![first.id, second.id], messages: vec!["one".into(), "two".into()] },
		]
	);
	harness.close(&context()).await.expect("close harness");
}
