use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::harness::tests::support::{context, open, open_default, registry, root};
use crate::harness::types::{AgentChange, AnyTask, ConversationCreateOptions, ConversationEntryQuery, Extension, RootConversationOptions, SetOrClear, TaskInspectionState, Registry};
use crate::types::{ConversationOwnership, ConversationQuery, EntryDraft, EntryHead, JoinPolicy, SelfMarker, Storage, TaskOptions, TaskOutcome, TaskOutcomeError, TaskOutcomeStatus, TaskOwnership, ROOT_CONVERSATION_ID};
use std::collections::BTreeMap;
use crate::tasks::{NextTaskState, PhaseHandler, TaskDefinition, TaskMigration};
fn finishing_task(name: &str) -> AnyTask {
	let phase: PhaseHandler = Arc::new(|_, runtime, context| {
		Box::pin(async move {
			runtime
				.commit(
					Box::new(|_, _| Ok(Some(NextTaskState::completed(serde_json::json!({"finished": true}))))),
					&context,
				)
				.await?;
			Ok(())
		})
	});
	AnyTask {
		definition: Arc::new(TaskDefinition {
			name: name.into(),
			version: 1,
			initial: Arc::new(|_| serde_json::json!({"phase": "run"})),
			phases: Arc::new(BTreeMap::from([("run".into(), phase.clone())])),
			abort: phase,
			migrate: None,
			hooks: None,
		}),
	}
}

fn failing_task(name: &str) -> AnyTask {
	let phase: PhaseHandler = Arc::new(|_, runtime, context| {
		Box::pin(async move {
			runtime
				.commit(
					Box::new(|_, _| {
						Ok(Some(NextTaskState::Terminal {
							outcome: TaskOutcome::failed(TaskOutcomeError { message: "declined".into(), detail: None }),
						}))
					}),
					&context,
				)
				.await?;
			Ok(())
		})
	});
	AnyTask {
		definition: Arc::new(TaskDefinition {
			name: name.into(),
			version: 1,
			initial: Arc::new(|_| serde_json::json!({"phase": "run"})),
			phases: Arc::new(BTreeMap::from([("run".into(), phase.clone())])),
			abort: phase,
			migrate: None,
			hooks: None,
		}),
	}
}

fn held_until_aborted_task(name: &str, started: tokio::sync::mpsc::UnboundedSender<crate::types::TaskId>) -> AnyTask {
	let phase: PhaseHandler = Arc::new(move |record, runtime, _context| {
		let started = started.clone();
		Box::pin(async move {
			let _ = started.send(record.id);
			runtime.signal().aborted_future().await;
			Ok(())
		})
	});
	let abort: PhaseHandler = Arc::new(|_, runtime, context| {
		Box::pin(async move {
			runtime
				.commit(Box::new(|_, _| Ok(Some(NextTaskState::aborted_with("test abort")))), &context)
				.await?;
			Ok(())
		})
	});
	AnyTask {
		definition: Arc::new(TaskDefinition {
			name: name.into(),
			version: 1,
			initial: Arc::new(|_| serde_json::json!({"phase": "run"})),
			phases: Arc::new(BTreeMap::from([("run".into(), phase)])),
			abort,
			migrate: None,
			hooks: None,
		}),
	}
}

fn all_settled_parent(
	name: &str,
	children: Vec<AnyTask>,
	policy: JoinPolicy,
	observed: Arc<parking_lot::Mutex<Vec<TaskOutcomeStatus>>>,
) -> AnyTask {
	let child_ids = Arc::new(parking_lot::Mutex::new(Vec::<crate::types::TaskId>::new()));
	let run_children = child_ids.clone();
	let definitions = children;
	let run: PhaseHandler = Arc::new(move |_, runtime, context| {
		let children = run_children.clone();
		let definitions = definitions.clone();
		Box::pin(async move {
			runtime.commit(Box::new(move |tx, owner| {
				let mut created = Vec::with_capacity(definitions.len());
				for child in definitions {
					created.push(tx.create_task(&child, serde_json::Value::Null, TaskOptions {
						ownership: TaskOwnership::Task { task_id: owner.id },
						conversation_id: None,
						background: None,
					}).map_err(|error| error.to_string())?);
				}
				*children.lock() = created.clone();
				Ok(Some(NextTaskState::Waiting {
					checkpoint: serde_json::json!({"phase": "resume"}), on: created, policy,
				}))
			}), &context).await?;
			Ok(())
		})
	});
	let wait_children = child_ids;
	let resume: PhaseHandler = Arc::new(move |_, runtime, context| {
		let children = wait_children.lock().clone();
		let observed = observed.clone();
		Box::pin(async move {
			let outcomes = runtime.outcomes(&children, &context).await?;
			*observed.lock() = outcomes.iter().map(|outcome| outcome.status).collect();
			runtime.commit(Box::new(|_, _| Ok(Some(NextTaskState::completed(serde_json::json!({"joined": true}))))), &context).await?;
			Ok(())
		})
	});
	AnyTask { definition: Arc::new(TaskDefinition {
		name: name.into(), version: 1,
		initial: Arc::new(|_| serde_json::json!({"phase": "run"})),
		phases: Arc::new(BTreeMap::from([("run".into(), run), ("resume".into(), resume)])),
		abort: Arc::new(|_, _, _| Box::pin(async { Ok(()) })), migrate: None, hooks: None,
	}) }
}

#[tokio::test]
async fn root_creation_is_lazy_and_root_handle_is_stable() {
	let (_storage, _registry, harness) = open_default().await;
	assert!(harness.conversation(ROOT_CONVERSATION_ID, &context()).await.expect("root lookup").is_none());
	let first = root(&harness).await;
	assert_eq!(first.id, ROOT_CONVERSATION_ID);
	assert!(harness.session.snapshot(&crate::harness::live::live_doc(), Some(first.id.0), None, &context()).await.expect("live doc").expect("live document").is_empty());
	let inbox = harness.session.snapshot(&crate::harness::inbox::inbox_doc(), Some(first.id.0), None, &context()).await.expect("inbox doc").expect("inbox document");
	assert!(inbox.get("items").and_then(serde_json::Value::as_array).is_some_and(Vec::is_empty));
	let provider = harness.session.snapshot(&crate::harness::provider::provider_doc(), Some(first.id.0), None, &context()).await.expect("provider doc").expect("provider document");
	assert!(provider.get("sessionId").and_then(serde_json::Value::as_str).is_some_and(|id| !id.is_empty()));
	let usage = harness.session.snapshot(&crate::harness::usage::usage_doc(), Some(first.id.0), None, &context()).await.expect("usage doc").expect("usage document");
	assert!(usage.get("models").and_then(serde_json::Value::as_object).is_some_and(serde_json::Map::is_empty));
	assert!(usage.get("tools").and_then(serde_json::Value::as_object).is_some_and(serde_json::Map::is_empty));
	assert!(harness.session.snapshot(&crate::harness::agent::agent_doc(), Some(first.id.0), None, &context()).await.expect("agent doc").expect("agent document").is_empty());
	let second = harness.root(&context(), Some(RootConversationOptions::default())).await.expect("root again");
	assert_eq!(second.id, first.id);
	let agent = first.agent(&context()).await.expect("root agent");
	assert!(agent.model.is_none());
	assert!(agent.tools.is_empty());
	assert!(agent.cwd.is_none());
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn independent_conversations_and_forks_keep_their_own_visible_history() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	conversation.configure(AgentChange { cwd: Some(SetOrClear::Set("/as-of".into())), ..Default::default() }, &context()).await.expect("configure before fork point");
	let id = conversation.id;
	let (first, second) = conversation.commit(move |tx| {
		let first = tx.append_entry(id, EntryDraft { kind: Some("first".into()), ..Default::default() }).map_err(|e| e.to_string())?;
		let second = tx.append_entry(id, EntryDraft { kind: Some("second".into()), ..Default::default() }).map_err(|e| e.to_string())?;
		Ok((first.id, second.id))
	}, &context()).await.expect("append entries");
	conversation.configure(AgentChange { cwd: Some(SetOrClear::Set("/later".into())), ..Default::default() }, &context()).await.expect("configure after fork point");
	let fork = conversation.fork(first, ConversationCreateOptions { ownership: ConversationOwnership::Ownerless, agent: None, init: None }, &context()).await.expect("fork at first entry");
	let fork_id = fork.id;
	let child_entry = fork.commit(move |tx| tx.append_entry(fork_id, EntryDraft { kind: Some("child".into()), ..Default::default() }).map(|entry| entry.id).map_err(|e| e.to_string()), &context()).await.expect("append child history");
	let grandchild = fork.fork(child_entry, ConversationCreateOptions { ownership: ConversationOwnership::Ownerless, agent: None, init: None }, &context()).await.expect("fork a fork");
	let unrelated = harness.create_conversation(ConversationCreateOptions { ownership: ConversationOwnership::Ownerless, agent: None, init: None }, &context()).await.expect("independent conversation");
	assert_eq!(conversation.agent(&context()).await.expect("parent agent").cwd.as_deref(), Some("/later"));
	let as_of = harness.session.snapshot_as_of(&crate::harness::agent::agent_doc(), conversation.id, None, first, &context()).await.expect("agent as of fork entry").expect("agent state at fork entry");
	assert_eq!(as_of.get("cwd").and_then(serde_json::Value::as_str), Some("/as-of"));
	assert_eq!(fork.agent(&context()).await.expect("fork agent").cwd.as_deref(), Some("/as-of"));
	let root_provider = harness.session.snapshot(&crate::harness::provider::provider_doc(), Some(conversation.id.0), None, &context()).await.expect("root provider").expect("root provider doc");
	let fork_provider = harness.session.snapshot(&crate::harness::provider::provider_doc(), Some(fork.id.0), None, &context()).await.expect("fork provider").expect("fork provider doc");
	let unrelated_provider = harness.session.snapshot(&crate::harness::provider::provider_doc(), Some(unrelated.id.0), None, &context()).await.expect("independent provider").expect("independent provider doc");
	let root_session = root_provider.get("sessionId").and_then(serde_json::Value::as_str).expect("root session id");
	let fork_session = fork_provider.get("sessionId").and_then(serde_json::Value::as_str).expect("fork session id");
	let unrelated_session = unrelated_provider.get("sessionId").and_then(serde_json::Value::as_str).expect("independent session id");
	assert_ne!(root_session, fork_session); assert_ne!(root_session, unrelated_session); assert_ne!(fork_session, unrelated_session);
	assert!(unrelated.fork(first, ConversationCreateOptions { ownership: ConversationOwnership::Ownerless, agent: None, init: None }, &context()).await.expect_err("foreign entry is invisible").contains("not visible"));
	let parent = conversation.entries(ConversationEntryQuery::default(), 10, None, &context()).await.expect("parent entries");
	assert_eq!(parent.items.iter().map(|entry| entry.id).collect::<Vec<_>>(), vec![second, first]);
	let inherited = fork.entries(ConversationEntryQuery::default(), 10, None, &context()).await.expect("fork entries");
	let deep = grandchild.entries(ConversationEntryQuery::default(), 10, None, &context()).await.expect("grandchild entries");
	assert_eq!(deep.items.iter().map(|entry| entry.id).collect::<Vec<_>>(), vec![child_entry, first]);
	assert_eq!(inherited.items.iter().map(|entry| entry.id).collect::<Vec<_>>(), vec![first]);
	assert!(unrelated.entries(ConversationEntryQuery::default(), 10, None, &context()).await.expect("empty entries").items.is_empty());
	assert!(harness.conversation(fork.id, &context()).await.expect("lookup fork").is_some());
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn failed_conversation_init_rolls_back_creation() {
	let (storage, _registry, harness) = open_default().await;
	let options = ConversationCreateOptions { ownership: ConversationOwnership::Ownerless, agent: None, init: Some(Arc::new(|_, _| Err("init failed".to_string()))) };
	let error = harness.create_conversation(options, &context()).await.err().expect("init failure");
	assert!(error.contains("init failed"));
	let page = storage.scan_conversations(&ConversationQuery::default(), 10, None, &context()).await;
	assert!(page.items.is_empty());
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn close_terminates_existing_view_watches_and_rejects_new_watch_acquisition() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let watch = conversation.watch(&context()).await.expect("acquire watch");
	harness.close(&context()).await.expect("close harness");
	assert_eq!(watch.closed().await, crate::types::WatchEnd::SessionClosed);
	let error = conversation.watch(&context()).await.err().expect("closed session rejects a new watch");
	assert!(harness.inspect(&context()).await.expect_err("inspect after close").to_lowercase().contains("closed"));
	assert!(error.to_lowercase().contains("closed"));
}

#[tokio::test]
async fn closed_harness_conversation_lookup_returns_an_error_not_a_panic() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	harness.close(&context()).await.expect("close harness");
	let id = conversation.id;
	let result = tokio::spawn(async move { harness.conversation(id, &context()).await }).await.expect("closed lookup must not panic");
	assert!(result.err().expect("closed harness lookup").contains("closed"));
}

#[tokio::test]
async fn conversation_configuration_replaces_fields_and_explicitly_clears_them() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	conversation
		.configure(
			crate::harness::types::AgentChange {
				cwd: Some(crate::harness::types::SetOrClear::Set("/repo".into())),
				instructions: Some(crate::harness::types::SetOrClear::Set("Be concise.".into())),
				..Default::default()
			},
			&context(),
		)
		.await
		.expect("configure agent");
	let configured = conversation.agent(&context()).await.expect("resolved agent");
	assert_eq!(configured.cwd.as_deref(), Some("/repo"));
	assert_eq!(configured.instructions.as_deref(), Some("Be concise."));
	conversation
		.configure(
			AgentChange { cwd: Some(SetOrClear::Set("/workspace".into())), ..Default::default() },
			&context(),
		)
		.await
		.expect("change cwd without changing other fields");
	assert_eq!(conversation.agent(&context()).await.expect("agent after partial change").instructions.as_deref(), Some("Be concise."));
	conversation
		.configure(
			crate::harness::types::AgentChange {
				cwd: Some(crate::harness::types::SetOrClear::Set("/worktree".into())),
				instructions: Some(crate::harness::types::SetOrClear::Clear),
				..Default::default()
			},
			&context(),
		)
		.await
		.expect("replace and clear");
	let changed = conversation.agent(&context()).await.expect("resolved changed agent");
	assert_eq!(changed.cwd.as_deref(), Some("/worktree"));
	assert!(changed.instructions.is_none());
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn plain_session_conversation_resolves_default_agent_without_writing_until_configured() {
	let (_storage, _registry, harness) = open_default().await;
	let id = harness
		.session
		.commit(
			|tx| tx.create_conversation(ConversationOwnership::Ownerless).map(|record| record.id).map_err(|error| error.to_string()),
			&context(),
		)
		.await
		.expect("create conversation through Session");
	let conversation = harness.conversation(id, &context()).await.expect("lookup raw conversation").expect("conversation exists");
	assert!(harness.session.snapshot(&crate::harness::agent::agent_doc(), Some(id.0), None, &context()).await
		.expect("agent before resolution").is_none());
	let agent = conversation.agent(&context()).await.expect("default agent");
	assert!(agent.model.is_none());
	assert!(agent.tools.is_empty());
	assert!(agent.cwd.is_none());
	assert!(harness.session.snapshot(&crate::harness::agent::agent_doc(), Some(id.0), None, &context()).await
		.expect("agent after resolution").is_none());
	conversation
		.configure(AgentChange { cwd: Some(SetOrClear::Set("/plain".into())), ..Default::default() }, &context())
		.await
		.expect("configure absent agent");
	let stored = harness.session.snapshot(&crate::harness::agent::agent_doc(), Some(id.0), None, &context())
		.await.expect("configured agent").expect("agent created by configure");
	assert_eq!(stored.get("cwd").and_then(serde_json::Value::as_str), Some("/plain"));
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn task_owned_work_records_its_owner_and_conversation() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let owner_definition = AnyTask {
		definition: Arc::new(TaskDefinition {
			name: "test.owner".into(),
			version: 1,
			initial: Arc::new(|_| serde_json::json!({"phase": "run"})),
			phases: Arc::new(BTreeMap::new()),
			abort: Arc::new(|_, _, _| Box::pin(async { Ok(()) })),
			migrate: None,
			hooks: None,
		}),
	};
	let child_definition = AnyTask {
		definition: Arc::new(TaskDefinition {
			name: "test.child".into(),
			version: 1,
			initial: Arc::new(|_| serde_json::json!({"phase": "run"})),
			phases: Arc::new(BTreeMap::new()),
			abort: Arc::new(|_, _, _| Box::pin(async { Ok(()) })),
			migrate: None,
			hooks: None,
		}),
	};
	let conversation_id = conversation.id;
	let (owner, child) = conversation
		.commit(
			move |tx| {
				let owner = tx
					.create_task(
						&owner_definition,
						serde_json::Value::Null,
						TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
					)
					.map_err(|error| error.to_string())?;
				let child = tx
					.create_task(
						&child_definition,
						serde_json::Value::Null,
						TaskOptions {
							ownership: TaskOwnership::Task { task_id: owner },
							conversation_id: None,
							background: None,
						},
					)
					.map_err(|error| error.to_string())?;
				Ok((owner, child))
			},
			&context(),
		)
		.await
		.expect("create owned task pair");
	let owner_record = harness.get_task(owner, &context()).await.expect("read owner").expect("owner exists");
	let child_record = harness.get_task(child, &context()).await.expect("read child").expect("child exists");
	assert_eq!(owner_record.conversation_id, conversation_id);
	assert_eq!(child_record.conversation_id, conversation_id);
	assert_eq!(child_record.owner, Some(owner));
	assert!(harness.close(&context()).await.is_ok());
}
#[tokio::test]
async fn inspect_derives_a_missing_definition_as_blocked_without_running_it() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let handler: PhaseHandler = Arc::new(|_, _, _| Box::pin(async { Ok(()) }));
	let task = AnyTask {
		definition: Arc::new(TaskDefinition {
			name: "test.unregistered".into(), version: 1,
			initial: Arc::new(|_| serde_json::json!({"phase": "run"})),
			phases: Arc::new(BTreeMap::from([("run".into(), handler.clone())])),
			abort: handler, migrate: None, hooks: None,
		}),
	};
	let id = conversation
		.commit(
			move |tx| tx.create_task(
				&task,
				serde_json::Value::Null,
				TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
			).map_err(|error| error.to_string()),
			&context(),
		)
		.await
		.expect("create pending task");
	let inspection = harness.inspect(&context()).await.expect("inspect pending task");
	assert_eq!(inspection.tasks.len(), 1);
	assert_eq!(inspection.tasks[0].record.id, id);
	assert!(matches!(inspection.tasks[0].state, TaskInspectionState::Blocked { reason: crate::harness::types::BlockedReason::MissingTask, .. }));
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn inspect_reports_a_paused_empty_scheduler() {
	let (_storage, _registry, harness) = open_default().await;
	let inspection = harness.inspect(&context()).await.expect("inspect");
	assert_eq!(inspection.scheduling, crate::harness::types::SchedulingState::Paused);
	assert!(inspection.tasks.is_empty());
	assert!(inspection.submissions.is_empty());
	harness.resume();
	assert_eq!(
		harness.inspect(&context()).await.expect("inspect resumed scheduler").scheduling,
		crate::harness::types::SchedulingState::Running
	);
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn conversation_view_watch_publishes_committed_entry_updates() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let state = conversation.view_state(&context()).await.expect("view state");
	let other_state = conversation.view_state(&context()).await.expect("independent view state");
	assert!(!Arc::ptr_eq(&state, &other_state));
	let watch = conversation.watch(&context()).await.expect("view watch");
	assert!(watch.value().entries.is_empty());
	let (send, receive) = tokio::sync::oneshot::channel();
	let send = Arc::new(parking_lot::Mutex::new(Some(send)));
	watch.start(Arc::new(move |view, ops, _| {
		let send = send.lock().take();
		Box::pin(async move { if let Some(send) = send { let _ = send.send((view.entries.len(), ops.len())); } })
	})).expect("start view watch");
	let conversation_id = conversation.id;
	conversation.commit(move |tx| {
		tx.append_entry(conversation_id, EntryDraft { kind: Some("note-a".into()), ..Default::default() }).map_err(|error| error.to_string())?;
		tx.append_entry(conversation_id, EntryDraft { kind: Some("note-b".into()), ..Default::default() }).map(|_| ()).map_err(|error| error.to_string())
	}, &context()).await.expect("append watched entries");
	let (entry_count, operation_count) = receive.await.expect("watch frame");
	assert_eq!(entry_count, 2);
	assert!(operation_count > 0);
	assert!(matches!(watch.stop().await, crate::types::WatchEnd::Stopped));
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn registered_task_runs_and_persists_its_terminal_outcome() {
	let (_storage, registry, harness) = open_default().await;
	let task = finishing_task("test.finish");
	registry.install(Extension::new("test-tasks").with_tasks(vec![task.clone()]));
	let conversation = root(&harness).await;
	let task_id = conversation
		.commit(
			move |tx| {
				tx.create_task(
					&task,
					serde_json::Value::Null,
					TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
				)
				.map_err(|error| error.to_string())
			},
			&context(),
		)
		.await
		.expect("create task");
	let paused = harness.inspect(&context()).await.expect("inspect ready task");
	assert!(matches!(paused.tasks[0].state, TaskInspectionState::Ready { migrates: false }));
	harness.resume();
	let settled = harness.wait_for_task(task_id, &context()).await.expect("wait for task");
	assert_eq!(settled.record.state.status, crate::types::TaskStateStatus::Terminal);
	assert_eq!(settled.record.state.outcome.expect("terminal outcome").status, crate::types::TaskOutcomeStatus::Completed);
	assert!(harness.inspect(&context()).await.expect("inspect after completion").tasks.is_empty());
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn conversation_abort_cascades_to_live_owned_task_subtrees() {
	let (_storage, registry, harness) = open_default().await;
	let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
	let owner_task = held_until_aborted_task("test.held-owner", started_tx.clone());
	let child_task = held_until_aborted_task("test.held-child", started_tx);
	registry.install(Extension::new("test-owned-work").with_tasks(vec![owner_task.clone(), child_task.clone()]));
	let conversation = root(&harness).await;
	let conversation_id = conversation.id;
	let (owner_id, child_id) = conversation
		.commit(
			move |tx| {
				let owner_id = tx
					.create_task(
						&owner_task,
						serde_json::Value::Null,
						TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
					)
					.map_err(|error| error.to_string())?;
				let child_id = tx
					.create_task(
						&child_task,
						serde_json::Value::Null,
						TaskOptions { ownership: TaskOwnership::Task { task_id: owner_id }, conversation_id: None, background: None },
					)
					.map_err(|error| error.to_string())?;
				Ok((owner_id, child_id))
			},
			&context(),
		)
		.await
		.expect("create owned tasks");
	harness.resume();
	let mut started = vec![started_rx.recv().await.expect("owner started"), started_rx.recv().await.expect("child started")];
	started.sort();
	let mut expected = vec![owner_id, child_id];
	expected.sort();
	assert_eq!(started, expected);
	conversation.abort(&context(), None).await.expect("abort conversation");
	conversation.wait_for_idle(&context()).await.expect("owned tasks drain");
	for id in [owner_id, child_id] {
		let task = harness.get_task(id, &context()).await.expect("read aborted task").expect("task exists");
		assert_eq!(task.conversation_id, conversation_id);
		assert_eq!(task.state.status, crate::types::TaskStateStatus::Terminal);
		assert_eq!(task.state.outcome.expect("abort outcome").status, crate::types::TaskOutcomeStatus::Aborted);
	}
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn background_task_does_not_block_idle_and_can_be_aborted_directly() {
	let (_storage, registry, harness) = open_default().await;
	let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
	let task = held_until_aborted_task("test.background", started_tx);
	registry.install(Extension::new("background-work").with_tasks(vec![task.clone()]));
	let conversation = root(&harness).await;
	let task_id = conversation
		.commit(
			move |tx| {
				tx.create_task(
					&task,
					serde_json::Value::Null,
					TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: Some(true) },
				)
				.map_err(|error| error.to_string())
			},
			&context(),
		)
		.await
		.expect("create background task");
	harness.resume();
	assert_eq!(started_rx.recv().await, Some(task_id));
	conversation.wait_for_idle(&context()).await.expect("background task is outside idle wait");
	let live = harness.inspect(&context()).await.expect("inspect running background task");
	assert!(matches!(live.tasks[0].state, TaskInspectionState::Running));
	assert_eq!(
		harness.get_task(task_id, &context()).await.expect("read running background task").expect("task exists").state.status,
		crate::types::TaskStateStatus::Running
	);
	harness.abort_task(task_id, &context()).await.expect("abort background task");
	let settled = harness.wait_for_task(task_id, &context()).await.expect("wait after abort");
	assert_eq!(settled.record.state.outcome.expect("aborted outcome").status, crate::types::TaskOutcomeStatus::Aborted);
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn structured_all_settled_wait_resumes_with_terminal_outcomes_in_child_order() {
	let (_storage, registry, harness) = open_default().await;
	let failed = failing_task("test.join-failed-child");
	let child = finishing_task("test.join-child");
	let observed = Arc::new(parking_lot::Mutex::new(Vec::new()));
	let parent = all_settled_parent("test.join-parent", vec![failed.clone(), child.clone()], JoinPolicy::AllSettled, observed.clone());
	registry.install(Extension::new("test-join").with_tasks(vec![failed, child, parent.clone()]));
	let conversation = root(&harness).await;
	let parent_id = conversation.commit(move |tx| tx.create_task(
		&parent, serde_json::Value::Null,
		TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
	).map_err(|error| error.to_string()), &context()).await.expect("create waiter");
	harness.resume();
	let settled = harness.wait_for_task(parent_id, &context()).await.expect("wait for all-settled parent");
	assert_eq!(settled.record.state.outcome.expect("parent outcome").status, TaskOutcomeStatus::Completed);
	assert_eq!(*observed.lock(), vec![TaskOutcomeStatus::Failed, TaskOutcomeStatus::Completed]);
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn unbound_session_task_creation_requires_an_explicit_conversation() {
	let (_storage, _registry, harness) = open_default().await;
	let task = finishing_task("test.unbound");
	let result = harness
		.session
		.commit(
			move |tx| {
				tx.create_task(
					&task,
					serde_json::Value::Null,
					TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
				)
				.map_err(|error| error.to_string())
			},
			&context(),
		)
		.await;
	assert!(format!("{:?}", result.expect_err("a conversation-owned task needs a bound conversation")).contains("conversation"));
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn a_failing_view_listener_does_not_interrupt_another_watch() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let broken = conversation.watch(&context()).await.expect("failing observer");
	let good = conversation.watch(&context()).await.expect("healthy observer");
	broken
		.start(Arc::new(|_, _, _| Box::pin(async { panic!("observer failed") })))
		.expect("start failing observer");
	let (send, receive) = tokio::sync::oneshot::channel();
	let send = Arc::new(parking_lot::Mutex::new(Some(send)));
	good.start(Arc::new(move |view, _, _| {
		let send = send.lock().take();
		Box::pin(async move {
			if let Some(send) = send {
				let _ = send.send(view.entries.len());
			}
		})
	}))
	.expect("start healthy observer");
	let conversation_id = conversation.id;
	conversation
		.commit(
			move |tx| {
				tx.append_entry(conversation_id, EntryDraft { kind: Some("observed".into()), ..Default::default() })
					.map(|_| ())
					.map_err(|error| error.to_string())
			},
			&context(),
		)
		.await
		.expect("commit view update");
	assert_eq!(receive.await.expect("healthy observer frame"), 1);
	assert!(matches!(broken.closed().await, crate::types::WatchEnd::ListenerError { .. }));
	assert!(matches!(good.stop().await, crate::types::WatchEnd::Stopped));
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn conversation_created_hook_runs_for_raw_transaction_creations() {
	let registry = registry();
	let calls = Arc::new(AtomicUsize::new(0));
	let hook_calls = calls.clone();
	let harness = open(
		Arc::new(crate::storage::memory::MemoryStorage::new()),
		crate::harness::types::HarnessOptions {
			registry,
			conversation_created: Some(Arc::new(move |_, _| {
				hook_calls.fetch_add(1, Ordering::SeqCst);
				Ok(())
			})),
			..Default::default()
		},
	)
	.await;
	let _root = root(&harness).await;
	harness
		.session
		.commit(
			|tx| tx.create_conversation(ConversationOwnership::Ownerless).map(|_| ()).map_err(|error| error.to_string()),
			&context(),
		)
		.await
		.expect("create raw transaction conversation");
	assert_eq!(
		calls.load(Ordering::SeqCst),
		2,
		"the creation hook must run for every conversation-creating commit"
	);
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn registry_definition_installed_after_open_is_used_on_resume() {
	let (_storage, registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let task = finishing_task("test.late-install");
	let registered_task = task.clone();
	let task_id = conversation
		.commit(
			move |tx| {
				tx.create_task(
					&task,
					serde_json::Value::Null,
					TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
				)
				.map_err(|error| error.to_string())
			},
			&context(),
		)
		.await
		.expect("create task before registering definition");
	let blocked = harness.inspect(&context()).await.expect("inspect missing definition");
	assert!(matches!(blocked.tasks[0].state, TaskInspectionState::Blocked { reason: crate::harness::types::BlockedReason::MissingTask, .. }));
	registry.install(Extension::new("late-task").with_tasks(vec![registered_task]));
	let ready = harness.inspect(&context()).await.expect("inspect newly registered task");
	assert!(matches!(ready.tasks[0].state, TaskInspectionState::Ready { migrates: false }));
	harness.resume();
	let settled = harness.wait_for_task(task_id, &context()).await.expect("run new task definition");
	assert_eq!(settled.record.state.outcome.expect("terminal outcome").status, TaskOutcomeStatus::Completed);
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn view_watch_unmounts_a_retired_document_and_mounts_its_recreation() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let watch = conversation.watch(&context()).await.expect("watch conversation view");
	assert!(watch.value().docs.contains_key("pi.live"));
	let (send, mut receive) = tokio::sync::mpsc::unbounded_channel();
	watch
		.start(Arc::new(move |view, ops, _| {
			let _ = send.send((view.docs.contains_key("pi.live"), ops.len()));
			Box::pin(async {})
		}))
		.expect("start view watch");
	let id = conversation.id;
	conversation
		.commit(
			move |tx| tx.retire_doc(&crate::harness::live::live_doc(), Some(id.0), None).map_err(|error| error.to_string()),
			&context(),
		)
		.await
		.expect("retire mounted live document");
	let (present, operations) = receive.recv().await.expect("retirement view frame");
	assert!(!present);
	assert!(operations > 0);
	conversation
		.commit(
			move |tx| tx.doc(&crate::harness::live::live_doc(), Some(id.0), None, None).map(|_| ()).map_err(|error| error.to_string()),
			&context(),
		)
		.await
		.expect("recreate live document");
	let (present, operations) = receive.recv().await.expect("recreation view frame");
	assert!(present);
	assert!(operations > 0);
	assert!(matches!(watch.stop().await, crate::types::WatchEnd::Stopped));
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn conversation_view_head_marker_cuts_older_entries() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let watch = conversation.watch(&context()).await.expect("watch view");
	let (send, receive) = tokio::sync::oneshot::channel();
	let send = Arc::new(parking_lot::Mutex::new(Some(send)));
	watch
		.start(Arc::new(move |view, _, _| {
			let kinds = view.entries.into_iter().map(|entry| entry.kind).collect::<Vec<_>>();
			let send = send.lock().take();
			Box::pin(async move { if let Some(send) = send { let _ = send.send(kinds); } })
		}))
		.expect("start watch");
	let id = conversation.id;
	conversation
		.commit(
			move |tx| {
				tx.append_entry(id, EntryDraft { kind: Some("old".into()), ..Default::default() })
					.map_err(|error| error.to_string())?;
				tx.append_entry(
					id,
					EntryDraft { kind: Some("head".into()), head: Some(EntryHead::SelfMarker(SelfMarker)), ..Default::default() },
				)
				.map_err(|error| error.to_string())?;
				tx.append_entry(id, EntryDraft { kind: Some("new".into()), ..Default::default() })
					.map(|_| ())
					.map_err(|error| error.to_string())
			},
			&context(),
		)
		.await
		.expect("append headed history");
	assert_eq!(receive.await.expect("headed view frame"), vec!["head", "new"]);
	assert!(matches!(watch.stop().await, crate::types::WatchEnd::Stopped));
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn structured_fail_fast_wait_fails_parent_after_a_child_fails() {
	let (_storage, registry, harness) = open_default().await;
	let failed = failing_task("test.join-failed-child");
	let sibling = finishing_task("test.join-success-child");
	let observed = Arc::new(parking_lot::Mutex::new(Vec::new()));
	let parent = all_settled_parent(
		"test.fail-fast-parent",
		vec![failed.clone(), sibling.clone()],
		JoinPolicy::FailFast,
		observed.clone(),
	);
	registry.install(Extension::new("test-fail-fast").with_tasks(vec![failed, sibling, parent.clone()]));
	let conversation = root(&harness).await;
	let parent_id = conversation
		.commit(
			move |tx| {
				tx.create_task(
					&parent,
					serde_json::Value::Null,
					TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
				)
				.map_err(|error| error.to_string())
			},
			&context(),
		)
		.await
		.expect("create fail-fast parent");
	harness.resume();
	let settled = harness.wait_for_task(parent_id, &context()).await.expect("wait for fail-fast parent");
	assert_eq!(settled.record.state.outcome.expect("failure outcome").status, TaskOutcomeStatus::Failed);
	assert!(observed.lock().is_empty(), "fail-fast parent does not run its resume phase");
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn stalled_view_watch_collapses_more_than_one_hundred_pending_frames_to_latest_view() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let watch = conversation.watch(&context()).await.expect("watch view");
	let (send, mut receive) = tokio::sync::mpsc::unbounded_channel();
	let callback_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
	let release = Arc::new(tokio::sync::Notify::new());
	let release_in_callback = release.clone();
	watch.start(Arc::new(move |view, ops, _| {
			let send = send.clone();
			let callback_count = callback_count.clone();
			let release = release_in_callback.clone();
			Box::pin(async move {
				let first = callback_count.fetch_add(1, Ordering::SeqCst) == 0;
				let _ = send.send((view.entries.len(), ops.len()));
				if first {
					release.notified().await;
				}
			})
		}))
		.expect("start stalled watcher");
	let conversation_id = conversation.id;
	conversation
		.commit(
			move |tx| tx.append_entry(conversation_id, EntryDraft { kind: Some("initial".into()), ..Default::default() })
				.map(|_| ())
				.map_err(|error| error.to_string()),
			&context(),
		)
		.await
		.expect("first entry");
	assert_eq!(receive.recv().await.expect("initial frame").0, 1);
	for index in 0..101 {
		conversation
			.commit(
				move |tx| tx.append_entry(conversation_id, EntryDraft { kind: Some(format!("entry-{index}")), ..Default::default() })
					.map(|_| ())
					.map_err(|error| error.to_string()),
				&context(),
			)
			.await
			.expect("append pending view frame");
	}
	release.notify_one();
	let (latest_count, operation_count) = receive.recv().await.expect("collapsed latest frame");
	assert_eq!(latest_count, 102);
	assert!(operation_count > 0);
	assert!(receive.try_recv().is_err(), "overflow coalesces queued revisions to one current view");
	assert!(matches!(watch.stop().await, crate::types::WatchEnd::Stopped));
	assert!(harness.close(&context()).await.is_ok());
}

#[tokio::test]
async fn conversation_view_watch_skips_commits_that_touch_no_state() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let watch = conversation.watch(&context()).await.expect("watch view");
	let (send, mut receive) = tokio::sync::mpsc::unbounded_channel();
	watch
		.start(Arc::new(move |_, _, _| {
			let _ = send.send(());
			Box::pin(async {})
		}))
		.expect("start watch");
	conversation.commit(|_| Ok(()), &context()).await.expect("empty commit");
	assert!(tokio::time::timeout(std::time::Duration::from_millis(10), receive.recv()).await.is_err());
	assert!(matches!(watch.stop().await, crate::types::WatchEnd::Stopped));
	assert!(harness.close(&context()).await.is_ok());
}


#[tokio::test]
async fn failed_owner_cascades_but_successful_owner_holds_its_child_until_drained() {
	let (_storage, registry, harness) = open_default().await;
	let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
	let failed_owner = failing_task("test.failed-owner");
	let completed_owner = finishing_task("test.completed-owner");
	let child = held_until_aborted_task("test.owner-child", started_tx);
	registry.install(Extension::new("test-owner-outcomes").with_tasks(vec![failed_owner.clone(), completed_owner.clone(), child.clone()]));
	let conversation = root(&harness).await;
	let (failed_owner_id, failed_child_id, completed_owner_id, completed_child_id) = conversation
		.commit(
			move |tx| {
				let failed_owner_id = tx.create_task(
					&failed_owner,
					serde_json::Value::Null,
					TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
				).map_err(|error| error.to_string())?;
				let failed_child_id = tx.create_task(
					&child,
					serde_json::Value::Null,
					TaskOptions { ownership: TaskOwnership::Task { task_id: failed_owner_id }, conversation_id: None, background: None },
				).map_err(|error| error.to_string())?;
				let completed_owner_id = tx.create_task(
					&completed_owner,
					serde_json::Value::Null,
					TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
				).map_err(|error| error.to_string())?;
				let completed_child_id = tx.create_task(
					&child,
					serde_json::Value::Null,
					TaskOptions { ownership: TaskOwnership::Task { task_id: completed_owner_id }, conversation_id: None, background: None },
				).map_err(|error| error.to_string())?;
				Ok((failed_owner_id, failed_child_id, completed_owner_id, completed_child_id))
			},
			&context(),
		)
		.await
		.expect("create owner-child pairs");
	harness.resume();
	let mut children_started = vec![started_rx.recv().await.expect("first child started"), started_rx.recv().await.expect("second child started")];
	children_started.sort();
	let mut expected_children = vec![failed_child_id, completed_child_id];
	expected_children.sort();
	assert_eq!(children_started, expected_children);
	let failed = harness.wait_for_task(failed_owner_id, &context()).await.expect("failed owner settles");
	assert_eq!(failed.record.state.outcome.expect("owner failure").status, TaskOutcomeStatus::Failed);
	let failed_child = harness.get_task(failed_child_id, &context()).await.expect("read failed owner's child").expect("child exists");
	assert_eq!(failed_child.state.outcome.expect("cascaded abort").status, TaskOutcomeStatus::Aborted);
	let completed_owner = harness.get_task(completed_owner_id, &context()).await.expect("read completing owner").expect("owner exists");
	assert_eq!(completed_owner.state.status, crate::types::TaskStateStatus::Completing);
	assert_eq!(
		harness.get_task(completed_child_id, &context()).await.expect("read live child").expect("child exists").state.status,
		crate::types::TaskStateStatus::Running
	);
	conversation.abort(&context(), None).await.expect("abort remaining child");
	conversation.wait_for_idle(&context()).await.expect("drain successful owner's child");
	assert_eq!(
		harness.get_task(completed_owner_id, &context()).await.expect("read settled owner").expect("owner exists")
			.state.outcome.expect("success outcome").status,
		TaskOutcomeStatus::Completed
	);
	assert!(harness.close(&context()).await.is_ok());
}


#[tokio::test]
async fn inspection_reports_needed_migration_without_running_it_until_resume() {
	let (_storage, registry, harness) = open_default().await;
	let name = "test.versioned";
	let executable = finishing_task(name);
	let migrations = Arc::new(AtomicUsize::new(0));
	let migration_count = migrations.clone();
	let current = AnyTask {
		definition: Arc::new(TaskDefinition {
			name: name.into(),
			version: 2,
			initial: executable.definition.initial.clone(),
			phases: executable.definition.phases.clone(),
			abort: executable.definition.abort.clone(),
			migrate: Some(Arc::new(move |input, checkpoint, _| {
				migration_count.fetch_add(1, Ordering::SeqCst);
				Ok(TaskMigration { input: input.clone(), checkpoint: checkpoint.clone() })
			})),
			hooks: None,
		}),
	};
	let stored = AnyTask {
		definition: Arc::new(TaskDefinition {
			name: name.into(),
			version: 1,
			initial: executable.definition.initial.clone(),
			phases: executable.definition.phases.clone(),
			abort: executable.definition.abort.clone(),
			migrate: None,
			hooks: None,
		}),
	};
	registry.install(Extension::new("versioned-task").with_tasks(vec![current]));
	let conversation = root(&harness).await;
	let task_id = conversation
		.commit(
			move |tx| {
				tx.create_task(
					&stored,
					serde_json::Value::Null,
					TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
				)
				.map_err(|error| error.to_string())
			},
			&context(),
		)
		.await
		.expect("create v1 record");
	let inspection = harness.inspect(&context()).await.expect("inspect old task");
	assert!(matches!(inspection.tasks[0].state, TaskInspectionState::Ready { migrates: true }));
	assert_eq!(migrations.load(Ordering::SeqCst), 0);
	harness.resume();
	let settled = harness.wait_for_task(task_id, &context()).await.expect("run migrated task");
	assert_eq!(settled.record.state.outcome.expect("completion").status, TaskOutcomeStatus::Completed);
	assert_eq!(migrations.load(Ordering::SeqCst), 1);
	assert!(harness.close(&context()).await.is_ok());
}
