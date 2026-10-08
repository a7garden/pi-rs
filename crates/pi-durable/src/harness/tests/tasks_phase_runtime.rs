use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use pi_ai::types::{ModelThinkingLevel, Tool};
use pi_chord::context::{AbortController, Context, with_abort_signal};
use serde_json::{Value, json};
use tokio::sync::{Notify, oneshot};

use super::support::{context, open, open_default, root};
use crate::documents::{DocDefinition, DocToken, ScopeKind};
use crate::harness::harness::{Conversation, Harness};
use crate::harness::registry::RegistryImpl;
use crate::harness::types::{
	AgentChange, AnyTask, EntryDraft, Extension, HarnessOptions, HookRegistration, Registry,
	SetOrClear, ToolRegistration,
};
use crate::session::observation::DocumentWatch;
use crate::session::support::ControlledStorage;
use crate::tasks::{BoxFuture, HookHandler, HookInvokeFn, NextTaskState, PhaseHandler, Task, TaskDefinition, TaskRuntime, commit_change, define_task};
use crate::types::{
	ConversationOwnership, DocumentFork, DocumentHistory, EntryId, JsonObject, StorageWrite, TaskId, TaskOptions,
	TaskOutcomeStatus, TaskOwnership, TaskStateStatus, WatchEnd,
};

fn object(value: Value) -> JsonObject {
	value.as_object().expect("document initial value is an object").clone()
}

fn handler(
	f: impl Fn(crate::types::TaskRecord, Arc<dyn TaskRuntime>, Context) -> BoxFuture<Result<(), String>> + Send + Sync + 'static,
) -> PhaseHandler {
	Arc::new(f)
}

fn task(name: &str, initial: Value, phases: impl IntoIterator<Item = (String, PhaseHandler)>, abort: PhaseHandler) -> Task {
	define_task(TaskDefinition {
		name: name.to_string(),
		version: 1,
		initial: Arc::new(move |_| initial.clone()),
		phases: Arc::new(phases.into_iter().collect::<BTreeMap<_, _>>()),
		abort,
		migrate: None,
		hooks: None,
	})
}

fn abort_completed(reason: &'static str) -> PhaseHandler {
	handler(move |_, runtime, ctx| {
		Box::pin(async move {
			runtime.commit(commit_change(move |_, _| Ok(Some(NextTaskState::aborted_with(reason)))), &ctx).await?;
			Ok(())
		})
	})
}

fn register(registry: &Arc<RegistryImpl>, tasks: Vec<Task>) {
	registry.install(Extension::new("tasks-phase-runtime-tests").with_tasks(
		tasks.into_iter().map(|task| AnyTask { definition: task.definition }).collect(),
	));
}

fn task_options() -> TaskOptions {
	TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None }
}

async fn start(conversation: &Conversation, task: Task) -> TaskId {
	let definition = task.definition.clone();
	conversation
		.commit(
			move |tx| tx.create_task(&AnyTask { definition }, Value::Null, task_options()).map_err(|error| error.to_string()),
			&context(),
		)
		.await
		.expect("create task")
}

fn user(text: &str) -> pi_ai::Message {
	pi_ai::Message::User(pi_ai::types::UserMessage {
		content: pi_ai::types::UserContentField::Text(text.to_string()),
		timestamp: 0,
	})
}

fn test_tool(name: &str) -> Arc<ToolRegistration> {
	Arc::new(crate::harness::api::define_tool(ToolRegistration {
		tool: Tool {
			name: name.to_string(),
			description: name.to_string(),
			parameters: json!({"type":"object","properties":{}}),
			constrained_sampling: None,
		},
		replay: None,
		execution_mode: None,
		prepare_arguments: None,
		output_limits: None,
		execute: Arc::new(|_, _, _| Box::pin(async { crate::harness::api::ToolExecutionResult::default() })),
	}))
}

#[tokio::test]
async fn task_keeps_a_committed_terminal_outcome_when_the_handler_throws_afterwards() {
	let late_commit = Arc::new(Mutex::new(None::<String>));
	let done = task(
		"test.phase-terminal-throw",
		json!({"phase":"run"}),
		[("run".into(), {
			let late_commit = late_commit.clone();
			handler(move |_, runtime, ctx| {
				let late_commit = late_commit.clone();
				Box::pin(async move {
					runtime
						.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(json!("ok"))))), &ctx)
						.await?;
					let error = runtime
						.commit(commit_change(|_, _| Ok(None)), &ctx)
						.await
						.expect_err("a commit after terminal progress must reject");
					*late_commit.lock() = Some(error);
					Err("after terminal".into())
				})
			})
		})],
		abort_completed("stop"),
	);
	let (_storage, registry, harness) = open_default().await;
	register(&registry, vec![done.clone()]);
	let conversation = root(&harness).await;
	let id = start(&conversation, done).await;
	harness.resume();
	let receipt = harness.wait_for_task(id, &context()).await.expect("terminal receipt");
	assert_eq!(receipt.state.outcome.as_ref().expect("outcome").status, TaskOutcomeStatus::Completed);
	assert_eq!(receipt.state.outcome.as_ref().and_then(|outcome| outcome.result.clone()), Some(json!("ok")));
	let late = late_commit.lock().clone().expect("late commit was attempted");
	assert!(late.contains(&format!("Task {id}")), "{late}");
	assert!(late.contains("terminal") || late.contains("completing"), "{late}");
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn task_compares_checkpoints_by_value_including_arrays() {
	let collect = task(
		"test.phase-collect-value",
		json!({"phase":"collect","items": []}),
		[("collect".into(), handler(|record, runtime, ctx| Box::pin(async move {
			let items = record.state.checkpoint.as_ref().and_then(|checkpoint| checkpoint["items"].as_array()).expect("items checkpoint");
			let next = if items.len() < 2 {
				let mut next = items.clone();
				next.push(json!(format!("item{}", items.len())));
				next
			} else {
				items.clone()
			};
			runtime.commit(
				commit_change(move |_, _| Ok(Some(NextTaskState::Running { checkpoint: json!({"phase":"collect","items":next}) }))),
				&ctx,
			).await?;
			Ok(())
		})))],
		abort_completed("stop"),
	);
	let (_storage, registry, harness) = open_default().await;
	register(&registry, vec![collect.clone()]);
	let conversation = root(&harness).await;
	let id = start(&conversation, collect).await;
	harness.resume();
	let receipt = harness.wait_for_task(id, &context()).await.expect("fault receipt");
	let outcome = receipt.state.outcome.as_ref().expect("outcome");
	assert_eq!(outcome.status, TaskOutcomeStatus::Faulted);
	assert_eq!(outcome.error.as_ref().expect("fault").message, "Task test.phase-collect-value phase collect returned without durable progress");
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn task_commits_entries_and_outcomes_atomically_retains_memos_while_live_and_retires_documents() {
	let storage = Arc::new(ControlledStorage::new());
	let registry = super::support::registry();
	let harness = open(storage.clone(), HarnessOptions { registry: registry.clone(), ..Default::default() }).await;
	let progress = DocToken::new(DocDefinition::singleton("test.phase-progress", 1, ScopeKind::Task, || object(json!({"lines":[]}))));
	let child = task(
		"test.phase-child",
		json!({"phase":"run"}),
		[("run".into(), handler(|_, runtime, ctx| Box::pin(async move {
			runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))), &ctx).await?;
			Ok(())
		})))],
		abort_completed("stop"),
	);
	let child_definition = child.definition.clone();
	let child_id = Arc::new(Mutex::new(None::<TaskId>));
	let progress_seen = Arc::new(Mutex::new(None::<Value>));
	let writer = task(
		"test.phase-writer",
		json!({"phase":"write"}),
		[
			("write".into(), {
				let progress = progress.clone();
				let child_definition = child_definition.clone();
				let child_id = child_id.clone();
				handler(move |record, runtime, ctx| {
					let progress = progress.clone();
					let child_definition = child_definition.clone();
					let child_id = child_id.clone();
					Box::pin(async move {
						assert_eq!(runtime.memo("choice", &ctx).await?, None);
						let (first, second) = tokio::join!(
							runtime.memo_put("choice", json!("a"), &ctx),
							runtime.memo_put("choice", json!("b"), &ctx),
						);
						assert_eq!((first?, second?), (json!("a"), json!("a")));
						assert_eq!(runtime.memo("choice", &ctx).await?, Some(json!("a")));
						assert_eq!(runtime.memo("toString", &ctx).await?, None);
						assert_eq!(runtime.memo_put("toString", json!("own"), &ctx).await?, json!("own"));
						runtime
							.commit(
								commit_change(move |tx, current| {
									let draft = tx.doc(&progress, Some(current.id.0), None, None).map_err(|error| error.to_string())?;
									draft.set(&[pi_chord::delta::Seg::key("lines")], json!(["wrote"])).map_err(|error| error.to_string())?;
									let child = tx.create_task(&AnyTask { definition: child_definition }, Value::Null, task_options()).map_err(|error| error.to_string())?;
									*child_id.lock() = Some(child);
									Ok(Some(NextTaskState::Running { checkpoint: json!({"phase":"answer"}) }))
								}),
								&ctx,
							)
							.await?;
						let _ = record;
						Ok(())
					})
				})
			}),
			("answer".into(), {
				let progress = progress.clone();
				let progress_seen = progress_seen.clone();
				handler(move |record, runtime, ctx| {
					let progress = progress.clone();
					let progress_seen = progress_seen.clone();
					Box::pin(async move {
						let memos = Value::Object(record.memos.clone().expect("live task memos"));
						assert_eq!(memos, json!({"choice":"a","toString":"own"}));
						runtime
							.commit(commit_change(move |tx, current| {
								let draft = tx.doc(&progress, Some(current.id.0), None, None).map_err(|error| error.to_string())?;
								*progress_seen.lock() = draft.value().ok().and_then(|value| value.get("lines").cloned());
								Ok(None)
							}), &ctx)
							.await?;
						runtime
							.commit(commit_change(|tx, current| {
								let entry = tx.append_entry(current.conversation_id, EntryDraft {
									kind: Some("answer".into()),
									model: Some(vec![user("done")]),
									..Default::default()
								}).map_err(|error| error.to_string())?;
								Ok(Some(NextTaskState::completed(json!({"entryId": entry.id.0}))))
							}), &ctx)
							.await?;
						Ok(())
					})
				})
			}),
		],
		abort_completed("stop"),
	);
	register(&registry, vec![writer.clone(), child.clone()]);
	let root_conversation = root(&harness).await;
	let conversation = harness
		.create_conversation(crate::harness::types::ConversationCreateOptions {
			ownership: ConversationOwnership::Ownerless,
			agent: None,
			init: None,
		}, &context())
		.await
		.expect("create task conversation");
	let id = start(&conversation, writer).await;
	harness.resume();
	let receipt = harness.wait_for_task(id, &context()).await.expect("writer receipt");
	assert_eq!(receipt.state.checkpoint, None);
	assert_eq!(receipt.memos, None);
	assert_eq!(*progress_seen.lock(), Some(json!(["wrote"])));
	let outcome = receipt.state.outcome.as_ref().expect("writer outcome");
	assert_eq!(outcome.status, TaskOutcomeStatus::Completed);
	let entry_id = EntryId(outcome.result.as_ref().and_then(|result| result["entryId"].as_u64()).expect("answer entry id"));
	let entry = harness.session.read_on_line_tx(move |tx| tx.entry(entry_id).map_err(|error| error.to_string()), &context()).await.expect("read answer").expect("answer entry");
	assert_eq!(entry.kind, "answer");
	assert_eq!(entry.conversation_id, conversation.id);
	assert!(harness.session.snapshot(&progress, Some(id.0), None, &context()).await.expect("read retired task document").is_none());
	assert_eq!(harness.get_task(id, &context()).await.expect("get task").expect("task remains"), receipt.record);
	let child = child_id.lock().expect("child task created");
	let child_receipt = harness.wait_for_task(child, &context()).await.expect("child receipt");
	assert_eq!(child_receipt.conversation_id, conversation.id);
	assert_ne!(root_conversation.id, conversation.id);

	let commits = storage.commits();
	let answer_commit = commits.iter().find(|batch| batch.iter().any(|write| matches!(write, StorageWrite::Entry { value } if value.id == entry.id))).expect("answer was committed");
	// The completed outcome lands atomically with the answer entry; whether it
	// holds as Completing (child still live) or settles Terminal in the same
	// commit depends on the child-vs-writer scheduling race, so only the
	// outcome is pinned here.
	assert!(answer_commit.iter().any(|write| matches!(write, StorageWrite::Task { value } if value.id == id && value.state.outcome.as_ref().is_some_and(|outcome| outcome.status == TaskOutcomeStatus::Completed))));
	let progress_document_id = commits.iter().flatten().find_map(|write| match write {
		StorageWrite::DocumentCreate { record, .. } if record.kind == "test.phase-progress" => Some(record.id),
		_ => None,
	}).expect("progress document was created");
	assert!(commits.iter().any(|batch| {
		batch.iter().any(|write| matches!(write, StorageWrite::DocumentRetire { id: retired } if *retired == progress_document_id))
			&& batch.iter().any(|write| matches!(write, StorageWrite::Task { value } if value.id == id && value.state.status == TaskStateStatus::Terminal))
	}), "task document retirement must share terminal settlement");
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn task_keeps_one_registry_snapshot_per_phase_and_refreshes_at_phase_boundaries() {
	let seen = Arc::new(Mutex::new(Vec::<(String, Vec<String>, bool)>::new()));
	let (entered_tx, entered_rx) = oneshot::channel();
	let (release_tx, release_rx) = oneshot::channel();
	let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
	let release_rx = Arc::new(Mutex::new(Some(release_rx)));
	let snapshots = task(
		"test.phase-registry-snapshot",
		json!({"phase":"a"}),
		[
			("a".into(), {
				let seen = seen.clone();
				let entered_tx = entered_tx.clone();
				let release_rx = release_rx.clone();
				handler(move |_, runtime, ctx| {
					let seen = seen.clone();
					let entered_tx = entered_tx.clone();
					let release_rx = release_rx.clone();
					Box::pin(async move {
						let before = runtime.registry();
						let _ = entered_tx.lock().take().expect("phase starts once").send(());
						let release_rx = release_rx.lock().take().expect("phase release");
						release_rx.await.map_err(|error| error.to_string())?;
						let tools = runtime.registry().tools().into_iter().map(|(_, tool)| tool.tool.name.clone()).collect();
						seen.lock().push(("a".into(), tools, Arc::ptr_eq(&before, &runtime.registry())));
						runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::Running { checkpoint: json!({"phase":"b"}) }))), &ctx).await?;
						Ok(())
					})
				})
			}),
			("b".into(), {
				let seen = seen.clone();
				handler(move |_, runtime, ctx| {
					let seen = seen.clone();
					Box::pin(async move {
						let snapshot = runtime.registry();
						let tools = snapshot.tools().into_iter().map(|(_, tool)| tool.tool.name.clone()).collect();
						seen.lock().push(("b".into(), tools, Arc::ptr_eq(&snapshot, &runtime.registry())));
						runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))), &ctx).await?;
						Ok(())
					})
				})
			}),
		],
		abort_completed("stop"),
	);
	let (_storage, registry, harness) = open_default().await;
	register(&registry, vec![snapshots.clone()]);
	let conversation = root(&harness).await;
	let id = start(&conversation, snapshots).await;
	harness.resume();
	entered_rx.await.expect("phase entered");
	registry.install(Extension::new("phase-late-tool").with_tools(vec![test_tool("late")]));
	release_tx.send(()).expect("release phase");
	harness.wait_for_task(id, &context()).await.expect("task completes");
	assert_eq!(*seen.lock(), vec![
		("a".into(), Vec::<String>::new(), true),
		("b".into(), vec!["late".into()], true),
	]);
	harness.close(&context()).await.expect("close harness");
}

#[derive(Clone)]
struct PingHandler(Arc<dyn Fn() + Send + Sync>);

async fn ping(runtime: &Arc<dyn TaskRuntime>, calls: &Arc<Mutex<Vec<String>>>) -> Result<Vec<String>, String> {
	calls.lock().clear();
	let invoke: HookInvokeFn = Arc::new(|handler| {
		let callback = (**handler).as_any().downcast_ref::<PingHandler>().map(|handler| handler.0.clone());
		Box::pin(async move {
			if let Some(callback) = callback {
				callback();
			}
			Ok(())
		})
	});
	runtime.hooks().each("ping", invoke).await?;
	Ok(calls.lock().clone())
}

fn ping_extension(name: &str, task_name: &str, label: &str, calls: Arc<Mutex<Vec<String>>>) -> Extension {
	let label = label.to_string();
	Extension::new(name).with_hooks(vec![HookRegistration {
		task: task_name.to_string(),
		handlers: Arc::new(PingHandler(Arc::new(move || calls.lock().push(label.clone())))),
	}])
}

#[tokio::test]
async fn task_resolves_agent_lazily_once_per_phase_and_refreshes_registry_and_settings_next_phase() {
	let task_name = "test.phase-agent-snapshot";
	let calls = Arc::new(Mutex::new(Vec::new()));
	let seen = Arc::new(Mutex::new(Vec::<(String, Vec<String>, ModelThinkingLevel, Option<bool>)>::new()));
	let (waiting_before_tx, waiting_before_rx) = oneshot::channel();
	let (before_use_tx, before_use_rx) = oneshot::channel();
	let (waiting_after_tx, waiting_after_rx) = oneshot::channel();
	let (after_use_tx, after_use_rx) = oneshot::channel();
	let waiting_before_tx = Arc::new(Mutex::new(Some(waiting_before_tx)));
	let before_use_rx = Arc::new(Mutex::new(Some(before_use_rx)));
	let waiting_after_tx = Arc::new(Mutex::new(Some(waiting_after_tx)));
	let after_use_rx = Arc::new(Mutex::new(Some(after_use_rx)));
	let phases = task(
		task_name,
		json!({"phase":"a"}),
		[
			("a".into(), {
				let calls = calls.clone();
				let seen = seen.clone();
				let waiting_before_tx = waiting_before_tx.clone();
				let before_use_rx = before_use_rx.clone();
				let waiting_after_tx = waiting_after_tx.clone();
				let after_use_rx = after_use_rx.clone();
				handler(move |_, runtime, ctx| {
					let calls = calls.clone();
					let seen = seen.clone();
					let waiting_before_tx = waiting_before_tx.clone();
					let before_use_rx = before_use_rx.clone();
					let waiting_after_tx = waiting_after_tx.clone();
					let after_use_rx = after_use_rx.clone();
					Box::pin(async move {
						let _ = waiting_before_tx.lock().take().expect("before-use signal").send(());
						let before_use_rx = before_use_rx.lock().take().expect("first gate");
						before_use_rx.await.map_err(|error| error.to_string())?;
						let first = runtime.agent(&ctx).await?;
						let first_pings = ping(&runtime, &calls).await?;
						seen.lock().push(("a".into(), first_pings, first.thinking_level.clone(), None));
						let _ = waiting_after_tx.lock().take().expect("after-use signal").send(());
						let after_use_rx = after_use_rx.lock().take().expect("second gate");
						after_use_rx.await.map_err(|error| error.to_string())?;
						let second = runtime.agent(&ctx).await?;
						let second_pings = ping(&runtime, &calls).await?;
						let same = first.thinking_level == second.thinking_level
							&& first.extensions.len() == second.extensions.len()
							&& first.extensions.iter().zip(&second.extensions).all(|(before, after)| Arc::ptr_eq(before, after));
						seen.lock().push(("a".into(), second_pings, second.thinking_level.clone(), Some(same)));
						runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::Running { checkpoint: json!({"phase":"b"}) }))), &ctx).await?;
						Ok(())
					})
				})
			}),
			("b".into(), {
				let calls = calls.clone();
				let seen = seen.clone();
				handler(move |_, runtime, ctx| {
					let calls = calls.clone();
					let seen = seen.clone();
					Box::pin(async move {
						let agent = runtime.agent(&ctx).await?;
						let pings = ping(&runtime, &calls).await?;
						seen.lock().push(("b".into(), pings, agent.thinking_level, None));
						runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))), &ctx).await?;
						Ok(())
					})
				})
			}),
		],
		abort_completed("stop"),
	);
	let (_storage, registry, harness) = open_default().await;
	registry.install(ping_extension("phase-before", task_name, "before", calls.clone()));
	let conversation = root(&harness).await;
	register(&registry, vec![phases.clone()]);
	let id = start(&conversation, phases).await;
	harness.resume();
	waiting_before_rx.await.expect("phase waits before first use");
	conversation.configure(AgentChange { thinking_level: Some(SetOrClear::Set(ModelThinkingLevel::Low)), ..Default::default() }, &context())
		.await.expect("configure before first agent use");
	before_use_tx.send(()).expect("allow first use");
	waiting_after_rx.await.expect("first agent use completed");
	registry.install(ping_extension("phase-late", task_name, "late", calls.clone()));
	conversation.configure(AgentChange { thinking_level: Some(SetOrClear::Set(ModelThinkingLevel::High)), ..Default::default() }, &context())
		.await.expect("configure after first use");
	after_use_tx.send(()).expect("allow next phase");
	harness.wait_for_task(id, &context()).await.expect("task completes");
	assert_eq!(*seen.lock(), vec![
		("a".into(), vec!["before".into()], ModelThinkingLevel::Low, None),
		("a".into(), vec!["before".into()], ModelThinkingLevel::Low, Some(true)),
		("b".into(), vec!["before".into(), "late".into()], ModelThinkingLevel::High, None),
	]);
	harness.close(&context()).await.expect("close harness");
}

// Deferred with the `awaitWithContext(agent(), context)` parity work (see
// local/pi-durable-remaining.md): hangs/fails until the runtime.agent
// caller-cancellation seam lands.
#[tokio::test]
#[ignore = "agent caller-cancellation parity deferred (awaitWithContext seam)"]
async fn task_agent_rejects_a_pre_cancelled_caller_without_poisoning_phase_resolution() {
	let cancelled = Arc::new(Mutex::new(None::<String>));
	let resolved = Arc::new(Mutex::new(None::<ModelThinkingLevel>));
	let agent = task(
		"test.phase-agent-cancelled-caller",
		json!({"phase":"run"}),
		[("run".into(), {
			let cancelled = cancelled.clone();
			let resolved = resolved.clone();
			handler(move |_, runtime, ctx| {
				let cancelled = cancelled.clone();
				let resolved = resolved.clone();
				Box::pin(async move {
					let controller = AbortController::new();
					controller.abort("caller gone");
					let caller = with_abort_signal(controller.signal(), ctx.clone());
					let error = runtime.agent(&caller).await.err();
					*cancelled.lock() = error;
					let thinking = runtime.agent(&ctx).await?.thinking_level;
					*resolved.lock() = Some(thinking);
					runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))), &ctx).await?;
					Ok(())
				})
			})
		})],
		abort_completed("stop"),
	);
	let (_storage, registry, harness) = open_default().await;
	register(&registry, vec![agent.clone()]);
	let conversation = root(&harness).await;
	let id = start(&conversation, agent).await;
	harness.resume();
	harness.wait_for_task(id, &context()).await.expect("task completes");
	assert!(cancelled.lock().as_deref().is_some_and(|error| error.contains("caller gone")));
	assert_eq!(*resolved.lock(), Some(ModelThinkingLevel::Off));
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
#[ignore = "agent caller-cancellation parity deferred (awaitWithContext seam)"]
async fn task_surfaces_agent_resolution_failure_when_its_only_caller_is_cancelled() {
	let storage = Arc::new(ControlledStorage::new());
	let registry = super::support::registry();
	let harness = open(storage.clone(), HarnessOptions { registry: registry.clone(), ..Default::default() }).await;
	let observed = Arc::new(Mutex::new(None::<(Option<String>, Option<String>)>));
	let agent = task(
		"test.phase-agent-resolution-failure",
		json!({"phase":"run"}),
		[("run".into(), {
			let storage = storage.clone();
			let observed = observed.clone();
			handler(move |_, runtime, ctx| {
				let storage = storage.clone();
				let observed = observed.clone();
				Box::pin(async move {
					let mut gate = storage.hold_find_document();
					let controller = AbortController::new();
					let caller = with_abort_signal(controller.signal(), ctx.clone());
					let first_runtime = runtime.clone();
					let first = tokio::spawn(async move { first_runtime.agent(&caller).await });
					gate.entered().await;
					controller.abort("caller gone");
					gate.release();
					let first_error = first.await.map_err(|error| error.to_string())?.err();
					let second_error = runtime.agent(&ctx).await.err();
					*observed.lock() = Some((first_error, second_error));
					runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))), &ctx).await?;
					Ok(())
				})
			})
		})],
		abort_completed("stop"),
	);
	register(&registry, vec![agent.clone()]);
	let conversation = root(&harness).await;
	let id = start(&conversation, agent).await;
	harness.session.unload_documents().await;
	harness.resume();
	harness.wait_for_task(id, &context()).await.expect("task completes");
	let (first, second) = observed.lock().clone().expect("agent resolution results");
	assert!(first.as_deref().is_some_and(|error| error.contains("caller gone")), "first result: {first:?}");
	assert!(second.as_deref().is_some_and(|error| error.contains("caller gone")), "later waiter: {second:?}");
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn runtime_operations_reject_after_invocation_ends_and_its_watches_stop() {
	let notes = DocToken::new(DocDefinition::singleton("test.phase-runtime-notes", 1, ScopeKind::Session, || object(json!({"text":""}))));
	let absent_doc = DocToken::new(DocDefinition::singleton("test.phase-runtime-absent", 1, ScopeKind::Session, || object(json!({"text":""}))));
	let captured = Arc::new(Mutex::new(None::<Arc<dyn TaskRuntime>>));
	let absent = Arc::new(Mutex::new(None::<bool>));
	let watch_slot = Arc::new(Mutex::new(None::<DocumentWatch>));
	let deliveries = Arc::new(Mutex::new(Vec::<String>::new()));
	let handler_context = Arc::new(Mutex::new(None::<Context>));
	let watcher = task(
		"test.phase-runtime-watcher",
		json!({"phase":"run"}),
		[("run".into(), {
			let notes = notes.clone();
			let absent_doc = absent_doc.clone();
			let captured = captured.clone();
			let absent = absent.clone();
			let watch_slot = watch_slot.clone();
			let deliveries = deliveries.clone();
			let handler_context = handler_context.clone();
			handler(move |_, runtime, ctx| {
				let notes = notes.clone();
				let absent_doc = absent_doc.clone();
				let captured = captured.clone();
				let absent = absent.clone();
				let watch_slot = watch_slot.clone();
				let deliveries = deliveries.clone();
				let handler_context = handler_context.clone();
				Box::pin(async move {
					*captured.lock() = Some(runtime.clone());
					*handler_context.lock() = Some(ctx.clone());
					*absent.lock() = Some(runtime.watch_doc(&absent_doc, runtime.conversation_id(), &ctx).await?.is_none());
					let watch = runtime.watch_doc(&notes, runtime.conversation_id(), &ctx).await?.expect("notes document exists");
					let delivered = deliveries.clone();
					watch.start(Arc::new(move |value, _, _| {
						let delivered = delivered.clone();
						Box::pin(async move {
							let text = value.as_ref().and_then(|value| value.get("text")).and_then(Value::as_str).unwrap_or("retired");
							delivered.lock().push(text.to_string());
						})
					})).map_err(|error| error.to_string())?;
					*watch_slot.lock() = Some(watch);
					runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))), &ctx).await?;
					Ok(())
				})
			})
		})],
		abort_completed("stop"),
	);
	let (_storage, registry, harness) = open_default().await;
	register(&registry, vec![watcher.clone()]);
	let conversation = root(&harness).await;
	conversation.commit({
		let notes = notes.clone();
		move |tx| {
			let draft = tx.doc(&notes, None, None, None).map_err(|error| error.to_string())?;
			draft.set(&[pi_chord::delta::Seg::key("text")], json!("hello")).map_err(|error| error.to_string())
		}
	}, &context()).await.expect("create notes");
	let id = start(&conversation, watcher).await;
	harness.resume();
	let receipt = harness.wait_for_task(id, &context()).await.expect("task completes");
	assert_eq!(*absent.lock(), Some(true));
	let watch = watch_slot.lock().clone().expect("watch acquired");
	assert_eq!(watch.closed().await, WatchEnd::Stopped);
	conversation.commit({
		let notes = notes.clone();
		move |tx| {
			let draft = tx.doc(&notes, None, None, None).map_err(|error| error.to_string())?;
			draft.set(&[pi_chord::delta::Seg::key("text")], json!("after")).map_err(|error| error.to_string())
		}
	}, &context()).await.expect("update notes after invocation");
	assert!(deliveries.lock().is_empty());
	let runtime = captured.lock().clone().expect("captured runtime");
	let err = runtime.commit(commit_change(|_, _| Ok(None)), &context()).await.expect_err("ended commit");
	assert!(err.contains("invocation has ended"), "{err}");
	assert!(runtime.memo("x", &context()).await.expect_err("ended memo read").contains("invocation has ended"));
	assert!(runtime.memo_put("x", json!(1), &context()).await.expect_err("ended memo write").contains("invocation has ended"));
	assert!(runtime.sleep(0, &context()).await.expect_err("ended sleep").contains("invocation has ended"));
	assert!(runtime.watch_doc(&notes, runtime.conversation_id(), &context()).await.err().expect("ended watch").contains("invocation has ended"));
	let handler_context = handler_context.lock().clone().expect("handler context");
	assert!(runtime.agent(&handler_context).await.expect_err("ended agent").contains("invocation has ended"));
	let now_error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runtime.now())).err().map(panic_message);
	let report_error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runtime.report("late".into()))).err().map(panic_message);
	assert!(now_error.as_deref().is_some_and(|error| error.contains("invocation has ended")), "{now_error:?}");
	assert!(report_error.as_deref().is_some_and(|error| error.contains("invocation has ended")), "{report_error:?}");
	assert_eq!(receipt.state.status, TaskStateStatus::Terminal);
	harness.close(&context()).await.expect("close harness");
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
	if let Some(message) = payload.downcast_ref::<String>() {
		message.clone()
	} else if let Some(message) = payload.downcast_ref::<&str>() {
		(*message).to_string()
	} else {
		"non-string panic".into()
	}
}

#[tokio::test]
async fn runtime_reads_documents_and_context_forwards_clock_and_reports() {
	let notes = DocToken::new(DocDefinition::conversation(
		"test.phase-runtime-history",
		1,
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		|| object(json!({"text":""})),
	));
	let captured = Arc::new(Mutex::new(None::<Arc<dyn TaskRuntime>>));
	let seen = Arc::new(Mutex::new(None::<(Value, Value, Vec<pi_ai::Message>, Vec<pi_ai::Message>, u64)>));
	let reader = task(
		"test.phase-runtime-reader",
		json!({"phase":"run"}),
		[("run".into(), {
			let notes = notes.clone();
			let captured = captured.clone();
			let seen = seen.clone();
			handler(move |_, runtime, ctx| {
				let notes = notes.clone();
				let captured = captured.clone();
				let seen = seen.clone();
				Box::pin(async move {
					*captured.lock() = Some(runtime.clone());
					let full = runtime.context_view(runtime.conversation_id(), &ctx, None).await?;
					let first = full.entries.first().expect("first history entry").id;
					let second = full.entries.get(1).expect("second history entry").id;
					let current = runtime.snapshot(&notes, runtime.conversation_id(), &ctx).await?.expect("current notes");
					let as_of = runtime.snapshot_as_of(&notes, runtime.conversation_id(), first, &ctx).await?.expect("notes as of first entry");
					let through_first = runtime.context_view(runtime.conversation_id(), &ctx, Some(first)).await?;
					let through_second = runtime.context_view(runtime.conversation_id(), &ctx, Some(second)).await?;
					*seen.lock() = Some((
						Value::Object(current),
						Value::Object(as_of),
						through_first.messages,
						through_second.messages,
						runtime.now(),
					));
					runtime.report("reported".into());
					runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))), &ctx).await?;
					Ok(())
				})
			})
		})],
		abort_completed("stop"),
	);
	let reports = Arc::new(Mutex::new(Vec::<String>::new()));
	let registry = super::support::registry();
	let report_sink = reports.clone();
	let harness = open(
		Arc::new(crate::storage::memory::MemoryStorage::new()),
		HarnessOptions {
			registry: registry.clone(),
			now: Some(Arc::new(|| 1234)),
			on_report: Some(Arc::new(move |error| report_sink.lock().push(error))),
			..Default::default()
		},
	).await;
	register(&registry, vec![reader.clone()]);
	let conversation = root(&harness).await;
	for text in ["one", "two"] {
		let notes = notes.clone();
		let id = conversation.id;
		let value = text.to_string();
		conversation.commit(move |tx| {
			let document = tx.doc(&notes, Some(id.0), None, None).map_err(|error| error.to_string())?;
			document.set(&[pi_chord::delta::Seg::key("text")], json!(value)).map_err(|error| error.to_string())?;
			tx.append_entry(id, EntryDraft { kind: Some("message".into()), model: Some(vec![user(&value)]), ..Default::default() })
				.map(|_| ())
				.map_err(|error| error.to_string())
		}, &context()).await.expect("append history and update document");
	}
	let id = start(&conversation, reader).await;
	harness.resume();
	harness.wait_for_task(id, &context()).await.expect("task completes");
	assert_eq!(*seen.lock(), Some((json!({"text":"two"}), json!({"text":"one"}), vec![user("one")], vec![user("one"), user("two")], 1234)));
	assert_eq!(*reports.lock(), ["reported"]);
	let runtime = captured.lock().clone().expect("captured runtime");
	assert!(runtime.snapshot(&notes, conversation.id, &context()).await.expect_err("ended snapshot").contains("invocation has ended"));
	assert!(runtime.context_view(conversation.id, &context(), None).await.expect_err("ended context").contains("invocation has ended"));
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn runtime_commits_queue_before_the_step_and_reject_after_it() {
	let storage = Arc::new(ControlledStorage::new());
	let harness_slot = Arc::new(Mutex::new(None::<Arc<Harness>>));
	let gate_slot = Arc::new(Mutex::new(None::<crate::session::support::Gate>));
	let runtime_slot = Arc::new(Mutex::new(None::<Arc<dyn TaskRuntime>>));
	let before_result = Arc::new(Mutex::new(None::<Result<(), String>>));
	let before_done = Arc::new(Notify::new());
	let (blocked_tx, blocked_rx) = oneshot::channel();
	let (queued_tx, queued_rx) = oneshot::channel();
	let (commit_started_tx, commit_started_rx) = oneshot::channel();
	let blocked_tx = Arc::new(Mutex::new(Some(blocked_tx)));
	let queued_tx = Arc::new(Mutex::new(Some(queued_tx)));
	let commit_started_tx = Arc::new(Mutex::new(Some(commit_started_tx)));
	let commit_started_rx = Arc::new(Mutex::new(Some(commit_started_rx)));
	let task = task(
		"test.phase-runtime-order",
		json!({"phase":"run"}),
		[("run".into(), {
			let storage = storage.clone();
			let harness_slot = harness_slot.clone();
			let gate_slot = gate_slot.clone();
			let runtime_slot = runtime_slot.clone();
			let before_result = before_result.clone();
			let before_done = before_done.clone();
			handler(move |_, runtime, ctx| {
				let storage = storage.clone();
				let harness_slot = harness_slot.clone();
				let gate_slot = gate_slot.clone();
				let runtime_slot = runtime_slot.clone();
				let before_result = before_result.clone();
				let before_done = before_done.clone();
				let blocked_tx = blocked_tx.clone();
				let queued_tx = queued_tx.clone();
				let commit_started_tx = commit_started_tx.clone();
				let commit_started_rx = commit_started_rx.clone();
				Box::pin(async move {
					*runtime_slot.lock() = Some(runtime.clone());
					let mut gate = storage.hold_commits();
					let harness = harness_slot.lock().clone().expect("harness installed");
					let conversation_id = runtime.conversation_id();
					tokio::spawn(async move {
						harness.session.commit(move |tx| tx.append_entry(conversation_id, EntryDraft { kind: Some("blocker".into()), ..Default::default() }).map(|_| ()).map_err(|error| error.to_string()), &context()).await
					});
					gate.entered().await;
					*gate_slot.lock() = Some(gate);
					let _ = blocked_tx.lock().take().expect("blocker signal").send(());
					let runtime_before = runtime.clone();
					let before_ctx = ctx.clone();
					let before_result_copy = before_result.clone();
					let before_done_copy = before_done.clone();
					let commit_started_tx = commit_started_tx.clone();
					tokio::spawn(async move {
						let commit = runtime_before.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(json!("before"))))), &before_ctx);
						let _ = commit_started_tx.lock().take().expect("commit-started signal").send(());
						let result = commit.await.map(|_| ()).map_err(|error| error.to_string());
						*before_result_copy.lock() = Some(result);
						before_done_copy.notify_one();
					});
					let commit_started_rx = commit_started_rx.lock().take().expect("commit-started receiver");
					commit_started_rx.await.map_err(|error| error.to_string())?;
					let _ = queued_tx.lock().take().expect("runtime commit queued").send(());
					Ok(())
				})
			})
		})],
		abort_completed("stop"),
	);
	let registry = super::support::registry();
	let harness = open(storage.clone(), HarnessOptions { registry: registry.clone(), ..Default::default() }).await;
	register(&registry, vec![task.clone()]);
	*harness_slot.lock() = Some(harness.clone());
	let conversation = root(&harness).await;
	let id = start(&conversation, task).await;
	harness.resume();
	blocked_rx.await.expect("blocker entered storage");
	queued_rx.await.expect("runtime commit started before handler returned");
	gate_slot.lock().take().expect("held storage gate").release();
	before_done.notified().await;
	assert_eq!(*before_result.lock(), Some(Ok(())));
	let receipt = harness.wait_for_task(id, &context()).await.expect("task terminal receipt");
	let runtime = runtime_slot.lock().clone().expect("captured runtime");
	let after = runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(json!("after"))))), &context())
		.await.expect_err("commit after the step rejects");
	assert!(after.contains("invocation has ended"), "{after}");
	assert_eq!(receipt.state.outcome.as_ref().and_then(|outcome| outcome.result.clone()), Some(json!("before")));
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn runtime_rejects_a_deferred_watch_acquisition_after_invocation_ends() {
	let storage = Arc::new(ControlledStorage::new());
	let notes = DocToken::new(DocDefinition::singleton("test.phase-late-watch", 1, ScopeKind::Session, || object(json!({"text":""}))));
	let watching = Arc::new(Mutex::new(None::<Result<(), String>>));
	let (started_tx, started_rx) = oneshot::channel();
	let (release_tx, release_rx) = oneshot::channel();
	let (finished_tx, finished_rx) = oneshot::channel();
	let started_tx = Arc::new(Mutex::new(Some(started_tx)));
	let release_rx = Arc::new(Mutex::new(Some(release_rx)));
	let finished_tx = Arc::new(Mutex::new(Some(finished_tx)));
	let late = task(
		"test.phase-late-watch",
		json!({"phase":"run"}),
		[("run".into(), {
			let notes = notes.clone();
			let watching = watching.clone();
			let started_tx = started_tx.clone();
			let release_rx = release_rx.clone();
			let finished_tx = finished_tx.clone();
			handler(move |_, runtime, _| {
				let notes = notes.clone();
				let watching = watching.clone();
				let started_tx = started_tx.clone();
				let release_rx = release_rx.clone();
				let finished_tx = finished_tx.clone();
				Box::pin(async move {
					let runtime = runtime.clone();
					tokio::spawn(async move {
						let _ = started_tx.lock().take().expect("watch-started signal").send(());
						let release_rx = release_rx.lock().take().expect("watch release");
						let _ = release_rx.await;
						let result = match runtime.watch_doc(&notes, runtime.conversation_id(), &context()).await {
							Ok(Some(watch)) => { watch.stop().await; Ok(()) }
							Ok(None) => Ok(()),
							Err(error) => Err(error),
						};
						*watching.lock() = Some(result);
						let _ = finished_tx.lock().take().expect("watch-finished signal").send(());
					});
					Ok(())
				})
			})
		})],
		abort_completed("stop"),
	);
	let registry = super::support::registry();
	let harness = open(storage, HarnessOptions { registry: registry.clone(), ..Default::default() }).await;
	register(&registry, vec![late.clone()]);
	let conversation = root(&harness).await;
	conversation.commit({
		let notes = notes.clone();
		move |tx| {
			let draft = tx.doc(&notes, None, None, None).map_err(|error| error.to_string())?;
			draft.set(&[pi_chord::delta::Seg::key("text")], json!("x")).map_err(|error| error.to_string())
		}
	}, &context()).await.expect("create watched document");
	let id = start(&conversation, late).await;
	harness.resume();
	started_rx.await.expect("watch waits for invocation to end");
	let receipt = harness.wait_for_task(id, &context()).await.expect("no-progress fault receipt");
	release_tx.send(()).expect("release deferred watch");
	finished_rx.await.expect("deferred watch finishes");
	assert_eq!(*watching.lock(), Some(Err(format!("Task {id} invocation has ended"))));
	assert_eq!(receipt.state.outcome.as_ref().expect("outcome").status, TaskOutcomeStatus::Faulted);
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn runtime_sleep_rechecks_the_harness_clock_after_each_timer() {
	let clock = Arc::new(std::sync::atomic::AtomicI64::new(1_000));
	let waiting = Arc::new(Notify::new());
	let sleeper = task(
		"test.phase-clock-sleeper",
		json!({"phase":"run"}),
		[("run".into(), {
			let clock_waiting = waiting.clone();
			handler(move |_, runtime, ctx| {
				let clock_waiting = clock_waiting.clone();
				Box::pin(async move {
					runtime.sleep(900, &ctx).await?;
					clock_waiting.notify_one();
					runtime.sleep(1_005, &ctx).await?;
					runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))), &ctx).await?;
					Ok(())
				})
			})
		})],
		abort_completed("stop"),
	);
	let registry = super::support::registry();
	let now = clock.clone();
	let harness = open(
		Arc::new(crate::storage::memory::MemoryStorage::new()),
		HarnessOptions { registry: registry.clone(), now: Some(Arc::new(move || now.load(std::sync::atomic::Ordering::SeqCst))), ..Default::default() },
	).await;
	register(&registry, vec![sleeper.clone()]);
	let conversation = root(&harness).await;
	let id = start(&conversation, sleeper).await;
	harness.resume();
	waiting.notified().await;
	assert!(tokio::time::timeout(Duration::from_millis(30), harness.wait_for_task(id, &context())).await.is_err(), "task must remain asleep while the clock stands still");
	clock.store(1_005, std::sync::atomic::Ordering::SeqCst);
	let receipt = harness.wait_for_task(id, &context()).await.expect("sleep deadline reached");
	assert_eq!(receipt.state.outcome.as_ref().expect("outcome").status, TaskOutcomeStatus::Completed);
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn runtime_sleep_rejects_when_its_invocation_or_own_context_is_cancelled() {
	let results = Arc::new(Mutex::new(Vec::<String>::new()));
	let signalled_sleeping = Arc::new(Notify::new());
	let signalled = task(
		"test.phase-sleep-signalled",
		json!({"phase":"run"}),
		[("run".into(), {
			let results = results.clone();
			let signalled_sleeping = signalled_sleeping.clone();
			handler(move |_, runtime, ctx| {
				let results = results.clone();
				let signalled_sleeping = signalled_sleeping.clone();
				Box::pin(async move {
					signalled_sleeping.notify_one();
					let error = runtime.sleep(61_000, &ctx).await.expect_err("abort signal interrupts sleep");
					results.lock().push(format!("signalled:{error}"));
					Err(error)
				})
			})
		})],
		abort_completed("test"),
	);
	let cancelled = task(
		"test.phase-sleep-cancelled",
		json!({"phase":"run"}),
		[("run".into(), {
			let results = results.clone();
			handler(move |_, runtime, ctx| {
				let results = results.clone();
				Box::pin(async move {
					let controller = AbortController::new();
					let own_context = with_abort_signal(controller.signal(), ctx.clone());
					let sleeping = runtime.sleep(61_000, &own_context);
					controller.abort("stop sleeping");
					let error = sleeping.await.expect_err("cancelled sleep context rejects");
					results.lock().push(format!("cancelled:{error}"));
					runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))), &ctx).await?;
					Ok(())
				})
			})
		})],
		abort_completed("stop"),
	);
	let registry = super::support::registry();
	let harness = open(
		Arc::new(crate::storage::memory::MemoryStorage::new()),
		HarnessOptions { registry: registry.clone(), now: Some(Arc::new(|| 1_000)), ..Default::default() },
	).await;
	register(&registry, vec![signalled.clone(), cancelled.clone()]);
	let conversation = root(&harness).await;
	let signalled_id = start(&conversation, signalled).await;
	let cancelled_id = start(&conversation, cancelled).await;
	harness.resume();
	let cancelled_receipt = harness.wait_for_task(cancelled_id, &context()).await.expect("cancelled-context task completes");
	assert_eq!(cancelled_receipt.state.outcome.as_ref().expect("cancelled task outcome").status, TaskOutcomeStatus::Completed);
	signalled_sleeping.notified().await;
	assert_eq!(harness.abort_task(signalled_id, &context()).await.expect("signal sleeping task"), "marked");
	let signalled_receipt = harness.wait_for_task(signalled_id, &context()).await.expect("abort outcome");
	assert_eq!(signalled_receipt.state.outcome.as_ref().expect("signalled outcome").status, TaskOutcomeStatus::Aborted);
	assert_eq!(signalled_receipt.state.outcome.as_ref().and_then(|outcome| outcome.reason.as_deref()), Some("test"));
	let results = results.lock().clone();
	assert_eq!(results, vec![
		"cancelled:stop sleeping".to_string(),
		format!("signalled:Task {signalled_id} invocation has ended"),
	]);
	harness.close(&context()).await.expect("close harness");
}
