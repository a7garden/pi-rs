use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use parking_lot::Mutex;
use pi_chord::context::{AbortController, Context, with_abort_signal};
use serde_json::{Value, json};

use super::support::{context, open, root};
use crate::harness::harness::{Conversation, Harness};
use crate::harness::registry::{create_registry, RegistryImpl};
use crate::harness::types::{AnyTask, Extension, HarnessOptions, Registry};
use crate::session::support::ControlledStorage;
use crate::tasks::{BoxFuture, NextTaskState, PhaseHandler, Task, TaskDefinition, TaskRuntime, commit_change, define_task};
use crate::types::{JoinPolicy, Storage, TaskId, TaskOptions, TaskOutcomeStatus, TaskOwnership, TaskRecord, TaskStateStatus};

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

fn handler(
    f: impl Fn(TaskRecord, Arc<dyn TaskRuntime>, Context) -> BoxFuture<Result<(), String>> + Send + Sync + 'static,
) -> PhaseHandler {
    Arc::new(f)
}

fn task(name: &str, phases: impl IntoIterator<Item = (String, PhaseHandler)>, abort: PhaseHandler) -> Task {
    let phases = phases.into_iter().collect::<BTreeMap<_, _>>();
    define_task(TaskDefinition {
        name: name.to_string(),
        version: 1,
        initial: Arc::new(|_| json!({"phase": "run"})),
        phases: Arc::new(phases),
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
    registry.install(Extension::new("task-abort-tests").with_tasks(
        tasks.into_iter().map(|task| AnyTask { definition: task.definition }).collect(),
    ));
}

async fn open_with_tasks(tasks: Vec<Task>, storage: Arc<dyn Storage>) -> (Arc<Harness>, Conversation) {
    let registry = Arc::new(create_registry());
    let harness = open(storage, HarnessOptions { registry: registry.clone(), ..HarnessOptions::default() }).await;
    register(&registry, tasks);
    let conversation = root(&harness).await;
    (harness, conversation)
}

async fn open_memory(tasks: Vec<Task>) -> (Arc<Harness>, Conversation) {
    let storage: Arc<dyn Storage> = Arc::new(crate::storage::memory::MemoryStorage::new());
    open_with_tasks(tasks, storage).await
}

async fn open_controlled(tasks: Vec<Task>, storage: &Arc<ControlledStorage>) -> (Arc<Harness>, Conversation) {
    let storage: Arc<dyn Storage> = storage.clone();
    open_with_tasks(tasks, storage).await
}

async fn create(conversation: &Conversation, task: Task) -> TaskId {
    conversation
        .commit(
            move |tx| {
                tx.create_task(
                    &AnyTask { definition: task.definition },
                    Value::Null,
                    TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None },
                )
                .map_err(|error| error.to_string())
            },
            &context(),
        )
        .await
        .expect("create task")
}

async fn wait_for_status(harness: &Harness, id: TaskId, status: TaskStateStatus) -> TaskRecord {
    tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            if let Some(record) = harness.get_task(id, &context()).await.expect("read task") {
                if record.state.status == status {
                    return record;
                }
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("task reaches expected status")
}

#[tokio::test]
async fn mark_during_a_progressing_phase_prevents_the_next_phase() {
    let reached = Arc::new(tokio::sync::Notify::new());
    let proceed = Arc::new(tokio::sync::Notify::new());
    let phases = Arc::new(Mutex::new(Vec::new()));
    let one = {
        let reached = reached.clone();
        let proceed = proceed.clone();
        let phases = phases.clone();
        handler(move |_, runtime, ctx| {
            let reached = reached.clone();
            let proceed = proceed.clone();
            let phases = phases.clone();
            Box::pin(async move {
                phases.lock().push("one");
                runtime
                    .commit(commit_change(|_, _| Ok(Some(NextTaskState::Running { checkpoint: json!({"phase": "two"}) }))), &ctx)
                    .await?;
                reached.notify_one();
                proceed.notified().await;
                Ok(())
            })
        })
    };
    let two = {
        let phases = phases.clone();
        handler(move |_, _, _| {
            let phases = phases.clone();
            Box::pin(async move {
                phases.lock().push("two");
                Ok(())
            })
        })
    };
    let abort = {
        let phases = phases.clone();
        handler(move |_, runtime, ctx| {
            let phases = phases.clone();
            Box::pin(async move {
                phases.lock().push("abort");
                runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("boundary")))), &ctx).await?;
                Ok(())
            })
        })
    };
    let two_phase = define_task(TaskDefinition {
        name: "test.abort-boundary".into(),
        version: 1,
        initial: Arc::new(|_| json!({"phase": "one"})),
        phases: Arc::new(BTreeMap::from([("one".into(), one), ("two".into(), two)])),
        abort,
        migrate: None,
        hooks: None,
    });
    let (harness, conversation) = open_memory(vec![two_phase.clone()]).await;
    let id = create(&conversation, two_phase).await;
    harness.resume();
    reached.notified().await;

    let harness_for_abort = harness.clone();
    let aborting = tokio::spawn(async move { harness_for_abort.abort_task(id, &context()).await });
    tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            if harness.get_task(id, &context()).await.expect("read marked task").expect("task exists").abort_requested {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("abort mark is durable before releasing the phase");
    proceed.notify_one();
    assert_eq!(aborting.await.expect("abort caller joins").expect("abort mark"), "marked");
    let receipt = tokio::time::timeout(TEST_TIMEOUT, harness.wait_for_task(id, &context()))
        .await
        .expect("abort outcome settles")
        .expect("abort settles");
    assert_eq!(receipt.state.outcome.as_ref().expect("outcome").status, TaskOutcomeStatus::Aborted);
    assert_eq!(receipt.state.outcome.as_ref().unwrap().reason.as_deref(), Some("boundary"));
    assert_eq!(*phases.lock(), ["one", "abort"]);
    harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn aborting_waiting_tasks_bypasses_dependencies_and_faults_nonterminal_handlers() {
    let dependency_gate = Arc::new(tokio::sync::Notify::new());
    let dependency_started = Arc::new(tokio::sync::Notify::new());
    let dependency = {
        let gate = dependency_gate.clone();
        let started = dependency_started.clone();
        task(
            "test.abort-wait-dependency",
            [(
                "run".into(),
                handler(move |_, runtime, ctx| {
                    let gate = gate.clone();
                    let started = started.clone();
                    Box::pin(async move {
                        started.notify_one();
                        gate.notified().await;
                        runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))), &ctx).await?;
                        Ok(())
                    })
                }),
            )],
            abort_completed("stop"),
        )
    };
    let dependency_id = Arc::new(Mutex::new(None));
    let make_waiter = |name: &str, abort: PhaseHandler| {
        let dependency_id = dependency_id.clone();
        task(
            name,
            [(
                "run".into(),
                handler(move |_, runtime, ctx| {
                    let dependency = dependency_id.lock().expect("dependency id set before resume");
                    Box::pin(async move {
                        runtime
                            .commit(commit_change(move |_, _| {
                                Ok(Some(NextTaskState::Waiting {
                                    checkpoint: json!({"phase": "run"}),
                                    on: vec![dependency],
                                    policy: JoinPolicy::AllSettled,
                                }))
                            }), &ctx)
                            .await?;
                        Ok(())
                    })
                }),
            )],
            abort,
        )
    };
    let lazy = make_waiter(
        "test.abort-wait-lazy-handler",
        handler(|_, _, _| Box::pin(async { Ok(()) })),
    );
    let throwing = make_waiter(
        "test.abort-wait-throwing-handler",
        handler(|_, _, _| Box::pin(async { Err("abort failed".to_owned()) })),
    );
    let (harness, conversation) = open_memory(vec![dependency.clone(), lazy.clone(), throwing.clone()]).await;
    let dependency_task = create(&conversation, dependency).await;
    *dependency_id.lock() = Some(dependency_task);
    let lazy_id = create(&conversation, lazy).await;
    let throwing_id = create(&conversation, throwing).await;
    harness.resume();
    dependency_started.notified().await;
    wait_for_status(&harness, lazy_id, TaskStateStatus::Waiting).await;
    wait_for_status(&harness, throwing_id, TaskStateStatus::Waiting).await;

    assert_eq!(harness.abort_task(lazy_id, &context()).await.expect("abort waiting task"), "marked");
    assert_eq!(harness.abort_task(throwing_id, &context()).await.expect("abort waiting task"), "marked");
    let lazy_receipt = harness.wait_for_task(lazy_id, &context()).await.expect("lazy handler settles");
    assert_eq!(lazy_receipt.state.outcome.as_ref().expect("lazy outcome").status, TaskOutcomeStatus::Faulted);
    assert_eq!(
		lazy_receipt.state.outcome.as_ref().unwrap().error.as_ref().unwrap().message,
        format!("Abort handler of task {lazy_id} returned without a terminal outcome"),
    );
    let throwing_receipt = harness.wait_for_task(throwing_id, &context()).await.expect("throwing handler settles");
    assert_eq!(throwing_receipt.state.outcome.as_ref().expect("throwing outcome").status, TaskOutcomeStatus::Faulted);
	assert_eq!(throwing_receipt.state.outcome.as_ref().unwrap().error.as_ref().unwrap().message, "abort failed");
    assert_eq!(
        harness.get_task(dependency_task, &context()).await.expect("read dependency").expect("dependency exists").state.status,
        TaskStateStatus::Running,
    );

    dependency_gate.notify_one();
    let dependency_receipt = harness.wait_for_task(dependency_task, &context()).await.expect("dependency settles");
	assert_eq!(dependency_receipt.state.outcome.as_ref().expect("dependency outcome").status, TaskOutcomeStatus::Completed);
    harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn abort_signals_and_joins_the_run_before_returning_and_can_be_repeated() {
    let started = Arc::new(tokio::sync::Notify::new());
    let run_ended = Arc::new(AtomicBool::new(false));
    let aborts = Arc::new(AtomicUsize::new(0));
    let abortable = {
        let started = started.clone();
        let run_ended = run_ended.clone();
        task(
            "test.abort-joins-run",
            [(
                "run".into(),
                handler(move |_, runtime, _| {
                    let started = started.clone();
                    let run_ended = run_ended.clone();
                    Box::pin(async move {
                        started.notify_one();
                        runtime.signal().aborted_future().await;
                        run_ended.store(true, Ordering::SeqCst);
                        Ok(())
                    })
                }),
            )],
            {
                let aborts = aborts.clone();
                handler(move |_, runtime, ctx| {
                    let aborts = aborts.clone();
                    Box::pin(async move {
                        aborts.fetch_add(1, Ordering::SeqCst);
                        runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("test")))), &ctx).await?;
                        Ok(())
                    })
                })
            },
        )
    };
    let (harness, conversation) = open_memory(vec![abortable.clone()]).await;
    let id = create(&conversation, abortable).await;
    harness.resume();
    started.notified().await;

    assert_eq!(harness.abort_task(id, &context()).await.expect("mark abort"), "marked");
    assert!(run_ended.load(Ordering::SeqCst), "abort_task returns only after the run invocation ends");
    assert_eq!(harness.abort_task(id, &context()).await.expect("repeat abort"), "marked");
    let receipt = harness.wait_for_task(id, &context()).await.expect("abort outcome");
    assert_eq!(receipt.state.outcome.as_ref().expect("outcome").status, TaskOutcomeStatus::Aborted);
	assert_eq!(receipt.state.outcome.as_ref().unwrap().reason.as_deref(), Some("test"));
    assert_eq!(aborts.load(Ordering::SeqCst), 1);
    harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn cancelled_abort_caller_leaves_mark_and_repeated_abort_does_not_signal_handler() {
    let run_started = Arc::new(tokio::sync::Notify::new());
    let run_release = Arc::new(tokio::sync::Notify::new());
    let handler_started = Arc::new(tokio::sync::Notify::new());
    let handler_release = Arc::new(tokio::sync::Notify::new());
    let aborts = Arc::new(AtomicUsize::new(0));
    let handler_signalled = Arc::new(AtomicBool::new(true));
    let work = {
        let run_started = run_started.clone();
        let run_release = run_release.clone();
        let handler_started = handler_started.clone();
        let handler_release = handler_release.clone();
        let aborts = aborts.clone();
        let handler_signalled = handler_signalled.clone();
        task(
            "test.abort-caller-cancelled",
            [(
                "run".into(),
                handler(move |_, _, _| {
                    let run_started = run_started.clone();
                    let run_release = run_release.clone();
                    Box::pin(async move {
                        run_started.notify_one();
                        run_release.notified().await;
                        Ok(())
                    })
                }),
            )],
            handler(move |_, runtime, ctx| {
                let handler_started = handler_started.clone();
                let handler_release = handler_release.clone();
                let aborts = aborts.clone();
                let handler_signalled = handler_signalled.clone();
                Box::pin(async move {
                    aborts.fetch_add(1, Ordering::SeqCst);
                    handler_started.notify_one();
                    handler_release.notified().await;
                    handler_signalled.store(runtime.signal().aborted(), Ordering::SeqCst);
                    runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("once")))), &ctx).await?;
                    Ok(())
                })
            }),
        )
    };
    let (harness, conversation) = open_memory(vec![work.clone()]).await;
    let id = create(&conversation, work).await;
    harness.resume();
    run_started.notified().await;
    let controller = AbortController::new();
    let cancelled_context = with_abort_signal(controller.signal(), context());
    let harness_for_cancelled = harness.clone();
    let cancelled = tokio::spawn(async move { harness_for_cancelled.abort_task(id, &cancelled_context).await });
    tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            if harness.get_task(id, &context()).await.expect("read abort mark").expect("task exists").abort_requested {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("abort mark persists before caller cancellation");
    controller.abort("caller gave up");
    let error = tokio::time::timeout(TEST_TIMEOUT, cancelled)
        .await
        .expect("caller cancellation releases the abort wait")
        .expect("cancelled abort task")
        .expect_err("caller cancellation rejects");
    assert!(error.contains("caller gave up"), "the caller sees its cancellation reason: {error}");
    run_release.notify_one();
    tokio::time::timeout(TEST_TIMEOUT, handler_started.notified()).await.expect("abort handler starts");
    assert_eq!(
        tokio::time::timeout(TEST_TIMEOUT, harness.abort_task(id, &context()))
            .await
            .expect("a repeated abort does not wait for the abort handler")
            .expect("repeat while handler runs"),
        "marked",
    );
    handler_release.notify_one();
    let receipt = harness.wait_for_task(id, &context()).await.expect("abort handler settles");
    assert_eq!(receipt.state.outcome.as_ref().expect("outcome").status, TaskOutcomeStatus::Aborted);
	assert_eq!(receipt.state.outcome.as_ref().unwrap().reason.as_deref(), Some("once"));
    assert!(!handler_signalled.load(Ordering::SeqCst), "a repeated abort does not signal the abort invocation");
    assert_eq!(aborts.load(Ordering::SeqCst), 1);
    harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn abort_handler_throwing_after_commit_keeps_its_terminal_outcome() {
    let started = Arc::new(tokio::sync::Notify::new());
    let work = {
        let started = started.clone();
        task(
            "test.abort-terminal-then-throw",
            [(
                "run".into(),
                handler(move |_, runtime, _| {
                    let started = started.clone();
                    Box::pin(async move {
                        started.notify_one();
                        runtime.signal().aborted_future().await;
                        Ok(())
                    })
                }),
            )],
            handler(|_, runtime, ctx| {
                Box::pin(async move {
                    runtime.commit(commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("done")))), &ctx).await?;
                    Err("after terminal".to_owned())
                })
            }),
        )
    };
    let (harness, conversation) = open_memory(vec![work.clone()]).await;
    let id = create(&conversation, work).await;
    harness.resume();
    started.notified().await;
    assert_eq!(harness.abort_task(id, &context()).await.expect("mark abort"), "marked");

    let receipt = harness.wait_for_task(id, &context()).await.expect("terminal outcome survives late handler error");
    assert_eq!(receipt.state.outcome.as_ref().expect("outcome").status, TaskOutcomeStatus::Aborted);
	assert_eq!(receipt.state.outcome.as_ref().unwrap().reason.as_deref(), Some("done"));
    harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn abort_during_reservation_settlement_prevents_phase_one_from_starting() {
    let ran = Arc::new(AtomicBool::new(false));
    let work = {
        let ran = ran.clone();
        task(
            "test.abort-reservation",
            [(
                "run".into(),
                handler(move |_, runtime, _| {
                    let ran = ran.clone();
                    Box::pin(async move {
                        ran.store(true, Ordering::SeqCst);
                        runtime.signal().aborted_future().await;
                        Ok(())
                    })
                }),
            )],
            abort_completed("test"),
        )
    };
    let storage = Arc::new(ControlledStorage::new());
    let (harness, conversation) = open_controlled(vec![work.clone()], &storage).await;
    let id = create(&conversation, work).await;
    let mut held = storage.hold_commits();
    harness.resume();
    held.entered().await;

	let abort_context = context();
	let mut aborting = Box::pin(harness.abort_task(id, &abort_context));
    tokio::select! {
        biased;
        result = &mut aborting => panic!("abort mark bypassed the held reservation: {result:?}"),
        _ = tokio::task::yield_now() => {}
    }
    held.release();
    assert_eq!(aborting.await.expect("abort mark"), "marked");
    let receipt = harness.wait_for_task(id, &context()).await.expect("task settles");
    assert_eq!(receipt.state.outcome.as_ref().expect("outcome").status, TaskOutcomeStatus::Aborted);
	assert_eq!(receipt.state.outcome.as_ref().unwrap().reason.as_deref(), Some("test"));
    assert!(!ran.load(Ordering::SeqCst), "the reserved task's first phase never starts");
    harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn abort_mark_wins_over_a_fault_that_races_its_commit() {
    let phase_started = Arc::new(tokio::sync::Notify::new());
    let fail_phase = Arc::new(tokio::sync::Notify::new());
    let phase_returning_error = Arc::new(tokio::sync::Notify::new());
    let work = {
        let phase_started = phase_started.clone();
        let fail_phase = fail_phase.clone();
        let phase_returning_error = phase_returning_error.clone();
        task(
            "test.abort-fault-race",
            [(
                "run".into(),
                handler(move |_, _, _| {
                    let phase_started = phase_started.clone();
                    let fail_phase = fail_phase.clone();
                    let phase_returning_error = phase_returning_error.clone();
                    Box::pin(async move {
                        phase_started.notify_one();
                        fail_phase.notified().await;
                        phase_returning_error.notify_one();
                        Err("would fault".to_owned())
                    })
                }),
            )],
            abort_completed("test"),
        )
    };
    let storage = Arc::new(ControlledStorage::new());
    let (harness, conversation) = open_controlled(vec![work.clone()], &storage).await;
    let id = create(&conversation, work).await;
    harness.resume();
    phase_started.notified().await;

    let mut held = storage.hold_commits();
	let abort_context = context();
	let mut aborting = Box::pin(harness.abort_task(id, &abort_context));
    tokio::select! {
        biased;
        result = &mut aborting => panic!("abort mark did not wait on its held storage commit: {result:?}"),
        _ = held.entered() => {}
    }
    fail_phase.notify_one();
    phase_returning_error.notified().await;
	crate::session::support::flush().await;
    held.release();

    assert_eq!(aborting.await.expect("abort mark and run join"), "marked");
    let receipt = harness.wait_for_task(id, &context()).await.expect("racing task settles");
    assert_eq!(receipt.state.outcome.as_ref().expect("outcome").status, TaskOutcomeStatus::Aborted);
    assert_eq!(receipt.state.outcome.as_ref().unwrap().reason.as_deref(), Some("test"));
    harness.close(&context()).await.expect("close");
}
