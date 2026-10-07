use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;
use pi_chord::context::Context;
use serde_json::{Value, json};

use super::support::{context, open, open_default, root};
use crate::harness::types::{AnyTask, Extension, Registry};
use crate::tasks::{
	BoxFuture, NextTaskState, PhaseHandler, Task, TaskDefinition, TaskRuntime, commit_change, define_task,
};
use crate::types::{JoinPolicy, TaskOptions, TaskOutcomeStatus, TaskOwnership, TaskStateStatus};

fn task(name: &str, phases: impl IntoIterator<Item = (String, PhaseHandler)>, abort: PhaseHandler) -> Task {
	let phases = phases.into_iter().collect::<BTreeMap<_, _>>();
	define_task(TaskDefinition {
		name: name.to_string(),
		version: 1,
		initial: Arc::new(|_| json!({"phase":"run"})),
		phases: Arc::new(phases),
		abort,
		migrate: None,
		hooks: None,
	})
}

fn handler(f: impl Fn(crate::types::TaskRecord, Arc<dyn TaskRuntime>, Context) -> BoxFuture<Result<(), String>> + Send + Sync + 'static) -> PhaseHandler {
	Arc::new(f)
}

fn register(registry: &Arc<crate::harness::registry::RegistryImpl>, tasks: Vec<Task>) {
	registry.install(Extension::new("task-runtime-tests").with_tasks(
		tasks.into_iter().map(|task| AnyTask { definition: task.definition }).collect(),
	));
}

async fn create(root: &crate::harness::harness::Conversation, task: Task) -> crate::types::TaskId {
	root.commit(move |tx| {
		tx.create_task(
			&AnyTask { definition: task.definition },
			Value::Null,
			TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
		)
		.map_err(|error| error.to_string())
	}, &context())
	.await
	.expect("create task")
}

fn abort_completed(reason: &'static str) -> PhaseHandler {
	handler(move |_record, runtime, ctx| Box::pin(async move {
		runtime.commit(commit_change(move |_, _| Ok(Some(NextTaskState::aborted_with(reason)))), &ctx).await?;
		Ok(())
	}))
}

#[tokio::test]
async fn task_runs_each_checkpoint_phase_and_persists_typed_result() {
	let seen = Arc::new(Mutex::new(Vec::new()));
	let phases = {
		let seen = seen.clone();
		vec![
			("count".into(), handler(move |record, runtime, ctx| {
				let seen = seen.clone();
				Box::pin(async move {
					let n = record.state.checkpoint.as_ref().and_then(|c| c["n"].as_u64()).unwrap_or(0);
					seen.lock().push(n);
					let next = if n == 2 {
						NextTaskState::completed(json!(3))
					} else {
						NextTaskState::Running { checkpoint: json!({"phase":"count","n":n+1}) }
					};
					runtime.commit(commit_change(move |_, _| Ok(Some(next))), &ctx).await?;
					Ok(())
				})
			})),
		]
	};
	let counter = define_task(TaskDefinition {
		name: "test.runtime-counter".into(), version: 1,
		initial: Arc::new(|_| json!({"phase":"count","n":0})), phases: Arc::new(phases.into_iter().collect()),
		abort: abort_completed("aborted"), migrate: None, hooks: None,
	});
	let (_storage, registry, harness) = open_default().await;
	register(&registry, vec![counter.clone()]);
	let conversation = root(&harness).await;
	let id = create(&conversation, counter).await;
	harness.resume();
	let receipt = harness.wait_for_task(id, &context()).await.expect("task settles");
	assert_eq!(*seen.lock(), [0, 1, 2]);
	assert_eq!(receipt.state.status, TaskStateStatus::Terminal);
	let outcome = receipt.state.outcome.as_ref().expect("outcome");
	assert_eq!(outcome.status, TaskOutcomeStatus::Completed);
	assert_eq!(outcome.result, Some(json!(3)));
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn phase_without_durable_progress_faults_instead_of_being_reported_as_success() {
	let idle = task("test.runtime-idle", [("run".into(), handler(|_, _, _| Box::pin(async { Ok(()) })))], abort_completed("stop"));
	let (_storage, registry, harness) = open_default().await;
	register(&registry, vec![idle.clone()]);
	let conversation = root(&harness).await;
	let id = create(&conversation, idle).await;
	harness.resume();
	let receipt = harness.wait_for_task(id, &context()).await.expect("fault receipt");
	let outcome = receipt.state.outcome.as_ref().expect("outcome");
	assert_eq!(outcome.status, TaskOutcomeStatus::Faulted);
	assert!(outcome.error.as_ref().unwrap().message.contains("returned without durable progress"));
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn task_waits_for_all_settled_dependencies_including_faulted_tasks() {
	let (_storage, registry, harness) = open_default().await;
	let gate = Arc::new(tokio::sync::Notify::new());
	let dependency = task("test.runtime-dependency", [("run".into(), {
		let gate = gate.clone();
		handler(move |_, runtime, ctx| { let gate = gate.clone(); Box::pin(async move {
			gate.notified().await;
			runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))), &ctx).await?;
			Ok(())
		}) })
	})], abort_completed("stop"));
	let failing = task("test.runtime-failing-dependency", [("run".into(), handler(|_, _, _| Box::pin(async { Err("expected failure".into()) })))], abort_completed("stop"));
	let ids = Arc::new(Mutex::new(Vec::new()));
	let waiter = task("test.runtime-waiter", [
		("wait".into(), { let ids = ids.clone(); handler(move |_, runtime, ctx| {
			let on = ids.lock().clone();
			Box::pin(async move {
				runtime.commit(commit_change(move |_, _| Ok(Some(NextTaskState::Waiting {
					checkpoint: json!({"phase":"resume"}), on, policy: JoinPolicy::AllSettled,
				}))), &ctx).await?;
				Ok(())
			})
		}) }),
		("resume".into(), handler(|_, runtime, ctx| Box::pin(async move {
			runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(json!("resumed"))))), &ctx).await?;
			Ok(())
		}))),
	], abort_completed("stop"));
	register(&registry, vec![dependency.clone(), failing.clone(), waiter.clone()]);
	let conversation = root(&harness).await;
	let dep = create(&conversation, dependency).await;
	let fail = create(&conversation, failing).await;
	*ids.lock() = vec![dep, fail];
	let waiting = create(&conversation, waiter).await;
	harness.resume();
	let first_waiting = loop {
		let record = harness.get_task(waiting, &context()).await.expect("read").expect("exists");
		if record.state.status == TaskStateStatus::Waiting { break record; }
		tokio::task::yield_now().await;
	};
	assert_eq!(first_waiting.state.on, Some(vec![dep, fail]));
	gate.notify_one();
	let receipt = harness.wait_for_task(waiting, &context()).await.expect("waiter settles");
	assert_eq!(receipt.state.outcome.as_ref().unwrap().result, Some(json!("resumed")));
	assert_eq!(harness.get_task(fail, &context()).await.unwrap().unwrap().state.outcome.unwrap().status, TaskOutcomeStatus::Faulted);
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn interrupted_running_task_reopens_pending_with_its_checkpoint_and_resumes() {
	let (storage, registry, harness) = open_default().await;
	let reached = Arc::new(tokio::sync::Notify::new());
	let work = task("test.runtime-restart", [("run".into(), {
		let reached = reached.clone();
		handler(move |record, runtime, ctx| {
			let reached = reached.clone();
			Box::pin(async move {
				if record.state.checkpoint.as_ref().and_then(|v| v["phase"].as_str()) == Some("run") {
					runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::Running {
						checkpoint: json!({"phase":"resume"}),
					}))), &ctx).await?;
					reached.notify_one();
					runtime.signal().aborted_future().await;
					return Ok(());
				}
				runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(json!("done"))))), &ctx).await?;
				Ok(())
			})
		})
	})], abort_completed("stop"));
	register(&registry, vec![work.clone()]);
	let conversation = root(&harness).await;
	let id = create(&conversation, work.clone()).await;
	harness.resume();
	reached.notified().await;
	harness.close(&context()).await.expect("close interrupted harness");
	let reopened = open(storage, crate::harness::types::HarnessOptions { registry: registry.clone(), ..Default::default() }).await;
	let pending = reopened.get_task(id, &context()).await.expect("read").expect("task survives");
	assert_eq!(pending.state.status, TaskStateStatus::Pending);
	assert_eq!(pending.state.checkpoint, Some(json!({"phase":"resume"})));
	reopened.resume();
	let receipt = reopened.wait_for_task(id, &context()).await.expect("recovered receipt");
	assert_eq!(receipt.state.outcome.as_ref().unwrap().result, Some(json!("done")));
	reopened.close(&context()).await.expect("close reopened harness");
}

#[tokio::test]
async fn newer_task_definition_migrates_the_checkpoint_before_resuming() {
	let (storage, registry, harness) = open_default().await;
	let old = define_task(TaskDefinition {
		name: "test.runtime-migration".into(),
		version: 1,
		initial: Arc::new(|_| json!({"phase":"run","old":true})),
		phases: Arc::new(BTreeMap::new()),
		abort: abort_completed("stop"),
		migrate: None,
		hooks: None,
	});
	register(&registry, vec![old.clone()]);
	let conversation = root(&harness).await;
	let id = create(&conversation, old).await;
	harness.close(&context()).await.expect("close v1 harness");

	let upgraded = define_task(TaskDefinition {
		name: "test.runtime-migration".into(),
		version: 2,
		initial: Arc::new(|_| json!({"phase":"run"})),
		phases: Arc::new(BTreeMap::from([(
			"run".into(),
			handler(|record, runtime, ctx| Box::pin(async move {
				assert_eq!(record.version, 2);
				assert_eq!(record.state.checkpoint, Some(json!({"phase":"run","migrated":true})));
				runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(json!("v2"))))), &ctx).await?;
				Ok(())
			})),
		)])),
		abort: abort_completed("stop"),
		migrate: Some(Arc::new(|input, checkpoint, from_version| {
			assert_eq!(from_version, 1);
			let mut migrated = checkpoint.clone();
			migrated["migrated"] = json!(true);
			Ok(crate::tasks::TaskMigration { input: input.clone(), checkpoint: migrated })
		})),
		hooks: None,
	});
	register(&registry, vec![upgraded]);
	let reopened = open(storage, crate::harness::types::HarnessOptions {
		registry: registry.clone(),
		..Default::default()
	}).await;
	reopened.resume();
	let receipt = reopened.wait_for_task(id, &context()).await.expect("migrated task settles");
	assert_eq!(receipt.version, 2);
	assert_eq!(receipt.state.outcome.as_ref().unwrap().result, Some(json!("v2")));
	reopened.close(&context()).await.expect("close v2 harness");
}

#[tokio::test]
async fn durable_abort_mark_signals_run_and_settles_through_abort_handler() {
	let (_storage, registry, harness) = open_default().await;
	let started = Arc::new(tokio::sync::Notify::new());
	let abortable = task("test.runtime-abort", [("run".into(), {
		let started = started.clone();
		handler(move |_, runtime, _| {
			let started = started.clone();
			Box::pin(async move {
				started.notify_one();
				runtime.signal().aborted_future().await;
				Ok(())
			})
		})
	})], abort_completed("durable-mark"));
	register(&registry, vec![abortable.clone()]);
	let conversation = root(&harness).await;
	let id = create(&conversation, abortable).await;
	harness.resume();
	started.notified().await;
	assert_eq!(harness.abort_task(id, &context()).await.expect("mark task"), "marked");
	let receipt = harness.wait_for_task(id, &context()).await.expect("abort settles");
	assert!(receipt.abort_requested);
	assert_eq!(receipt.state.status, TaskStateStatus::Terminal);
	let outcome = receipt.state.outcome.as_ref().expect("aborted outcome");
	assert_eq!(outcome.status, TaskOutcomeStatus::Aborted);
	assert_eq!(outcome.reason.as_deref(), Some("durable-mark"));
	harness.close(&context()).await.expect("close");
}

