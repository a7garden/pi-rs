use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use pi_chord::context::{with_abort_signal, AbortController};
use serde_json::{json, Value};
use tokio::sync::Notify;

use super::support::{context, open, open_default, registry, root};
use crate::harness::harness::{Conversation, Harness};
use crate::harness::registry::RegistryImpl;
use crate::harness::types::{
    AnyTask, ConversationCreateOptions, Extension, HarnessOptions, Registry, ReportHook,
};
use crate::session::support::{flush, ControlledStorage};
use crate::tasks::{
    commit_change, BoxFuture, NextTaskState, PhaseHandler, TaskDefinition, TaskRuntime,
};
use crate::types::{
    ConversationOwnership, ScanOrder, TaskId, TaskOptions, TaskOutcomeStatus, TaskOwnership,
    TaskQuery, TaskStateStatus,
};

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

fn phase(
    f: impl Fn(
            crate::types::TaskRecord,
            Arc<dyn TaskRuntime>,
            pi_chord::context::Context,
        ) -> BoxFuture<Result<(), String>>
        + Send
        + Sync
        + 'static,
) -> PhaseHandler {
    Arc::new(f)
}

fn finish_phase() -> PhaseHandler {
    phase(|_, runtime, context| {
        Box::pin(async move {
            runtime
                .commit(
                    Box::new(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                    &context,
                )
                .await?;
            Ok(())
        })
    })
}

fn task(name: &str, run: PhaseHandler) -> AnyTask {
    AnyTask {
        definition: Arc::new(TaskDefinition {
            name: name.to_string(),
            version: 1,
            initial: Arc::new(|_| serde_json::json!({"phase": "run"})),
            phases: Arc::new(BTreeMap::from([("run".to_string(), run)])),
            abort: finish_phase(),
            migrate: None,
            hooks: None,
        }),
    }
}

fn finishing_task(name: &str) -> AnyTask {
    task(name, finish_phase())
}

fn notify_then_finish(name: &str, started: Arc<tokio::sync::Notify>) -> AnyTask {
    let finish = finish_phase();
    task(
        name,
        phase(move |record, runtime, context| {
            let started = started.clone();
            let finish = finish.clone();
            Box::pin(async move {
                started.notify_one();
                finish(record, runtime, context).await
            })
        }),
    )
}

fn install(registry: &RegistryImpl, extension: &str, tasks: Vec<AnyTask>) {
    registry.install(Extension::new(extension).with_tasks(tasks));
}

fn report_hook(reports: Arc<Mutex<Vec<String>>>) -> ReportHook {
    Arc::new(move |error| reports.lock().push(error))
}

async fn open_controlled(
    storage: &Arc<ControlledStorage>,
    registry: &Arc<RegistryImpl>,
    on_report: Option<ReportHook>,
) -> Arc<Harness> {
    let storage: Arc<dyn crate::types::Storage> = storage.clone();
    open(
        storage,
        HarnessOptions {
            registry: registry.clone(),
            on_report,
            ..HarnessOptions::default()
        },
    )
    .await
}

async fn start(root: &Conversation, task: AnyTask, background: bool) -> TaskId {
    root.commit(
        move |tx| {
            tx.create_task(
                &task,
                Value::Null,
                TaskOptions {
                    ownership: TaskOwnership::Conversation,
                    conversation_id: None,
                    background: background.then_some(true),
                },
            )
            .map_err(|error| error.to_string())
        },
        &context(),
    )
    .await
    .expect("create task")
}

#[tokio::test]
async fn rejected_reservation_retries_after_a_later_registry_wakeup() {
    let storage = Arc::new(ControlledStorage::new());
    let registry = registry();
    let reports = Arc::new(Mutex::new(Vec::new()));
    let started = Arc::new(tokio::sync::Notify::new());
    let once = notify_then_finish("test.scheduling-reservation-retry", started.clone());
    install(&registry, "task-scheduling-reservation", vec![once.clone()]);
    let harness = open_controlled(&storage, &registry, Some(report_hook(reports.clone()))).await;
    let root = root(&harness).await;
    let id = start(&root, once, false).await;

    storage.fail_next_commit("busy");
    harness.resume();
    tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            if !reports.lock().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("rejected reservation is reported");
    assert!(
        reports.lock()[0].contains("busy"),
        "reservation rejection is surfaced to the report hook"
    );
    assert_eq!(
        harness
            .get_task(id, &context())
            .await
            .expect("read pending task")
            .expect("task exists")
            .state
            .status,
        TaskStateStatus::Pending,
        "a rejected reservation leaves the task pending"
    );

    // Publishing any registry change is the next scheduler wakeup.
    install(
        &registry,
        "task-scheduling-reservation-wake",
        vec![finishing_task("test.scheduling-wake")],
    );
    tokio::time::timeout(TEST_TIMEOUT, started.notified())
        .await
        .expect("registry publication retries the reservation");
    let settled = harness
        .wait_for_task(id, &context())
        .await
        .expect("wait for retried task");
    assert_eq!(
        settled.record.state.outcome.expect("completion").status,
        TaskOutcomeStatus::Completed
    );
    assert_eq!(
        reports.lock().len(),
        1,
        "the rejected reservation is reported once"
    );
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn wakeup_during_rejected_reservation_is_retained() {
    let storage = Arc::new(ControlledStorage::new());
    let registry = registry();
    let first_started = Arc::new(tokio::sync::Notify::new());
    let late_started = Arc::new(tokio::sync::Notify::new());
    let first = notify_then_finish("test.scheduling-first", first_started.clone());
    let late = notify_then_finish("test.scheduling-late", late_started.clone());
    install(&registry, "task-scheduling-first", vec![first.clone()]);
    let harness = open_controlled(&storage, &registry, None).await;
    let root = root(&harness).await;
    let first_id = start(&root, first, false).await;
    let late_id = start(&root, late.clone(), false).await;

    let mut held = storage.hold_commits();
    storage.fail_next_commit("busy");
    harness.resume();
    tokio::time::timeout(TEST_TIMEOUT, held.entered())
        .await
        .expect("reservation commit reaches held storage");
    // The late task is blocked on its missing definition. Its registration wakes the
    // scheduler while the first reservation is still in storage.
    install(&registry, "task-scheduling-late", vec![late]);
    held.release();

    tokio::time::timeout(TEST_TIMEOUT, first_started.notified())
        .await
        .expect("first task is reserved after the failed pass");
    tokio::time::timeout(TEST_TIMEOUT, late_started.notified())
        .await
        .expect("wake during storage failure is not lost");
    for id in [first_id, late_id] {
        let settled = harness
            .wait_for_task(id, &context())
            .await
            .expect("wait for task");
        assert_eq!(
            settled.record.state.outcome.expect("completion").status,
            TaskOutcomeStatus::Completed
        );
    }
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn rejected_fault_write_retries_the_task() {
    let storage = Arc::new(ControlledStorage::new());
    let registry = registry();
    let reports = Arc::new(Mutex::new(Vec::new()));
    let runs = Arc::new(AtomicUsize::new(0));
    let throws = {
        let runs = runs.clone();
        let storage = storage.clone();
        task(
            "test.scheduling-rejected-fault",
            phase(move |_, _, _| {
                let run = runs.fetch_add(1, Ordering::SeqCst);
                if run == 0 {
                    storage.fail_next_commit("busy");
                }
                Box::pin(async { Err("boom".to_string()) })
            }),
        )
    };
    install(&registry, "task-scheduling-fault", vec![throws.clone()]);
    let harness = open_controlled(&storage, &registry, Some(report_hook(reports.clone()))).await;
    let root = root(&harness).await;
    let id = start(&root, throws, false).await;
    harness.resume();
    let settled = harness
        .wait_for_task(id, &context())
        .await
        .expect("task faults after retry");
    let outcome = settled.record.state.outcome.expect("fault outcome");
    assert_eq!(outcome.status, TaskOutcomeStatus::Faulted);
    assert_eq!(outcome.error.expect("fault message").message, "boom");
    assert_eq!(
        runs.load(Ordering::SeqCst),
        2,
        "the rejected fault write causes a fresh invocation"
    );
    assert_eq!(reports.lock().len(), 1);
    assert!(
        reports.lock()[0].contains("busy"),
        "the rejected fault write is reported"
    );
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn idle_waits_count_blocked_foreground_work_and_ignore_background_tasks() {
    let registry = registry();
    let (started_tx, mut started_rx) =
        tokio::sync::mpsc::unbounded_channel::<(TaskId, tokio::sync::oneshot::Sender<()>)>();
    let gated = task(
        "test.scheduling-gated",
        phase(move |record, runtime, context| {
            let started = started_tx.clone();
            Box::pin(async move {
                let (release, blocked) = tokio::sync::oneshot::channel();
                started
                    .send((record.id, release))
                    .map_err(|error| error.to_string())?;
                blocked.await.map_err(|error| error.to_string())?;
                runtime
                    .commit(
                        Box::new(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                        &context,
                    )
                    .await?;
                Ok(())
            })
        }),
    );
    install(&registry, "task-scheduling-idle", vec![gated.clone()]);
    let harness = open_controlled(&Arc::new(ControlledStorage::new()), &registry, None).await;
    let root = root(&harness).await;
    harness
        .wait_for_idle(&context())
        .await
        .expect("initial harness is idle");
    let other = harness
        .create_conversation(
            ConversationCreateOptions {
                ownership: ConversationOwnership::Ownerless,
                agent: None,
                init: None,
            },
            &context(),
        )
        .await
        .expect("create ownerless conversation");
    let foreground = start(&root, gated.clone(), false).await;
    let background = start(&root, gated.clone(), true).await;
    let elsewhere = start(&other, gated, false).await;

    let cancelled = AbortController::new();
    let cancelled_context = with_abort_signal(cancelled.signal(), context());
    let harness_for_cancelled_wait = harness.clone();
    let cancelled_wait = tokio::spawn(async move {
        harness_for_cancelled_wait
            .wait_for_idle(&cancelled_context)
            .await
    });
    cancelled.abort("stop waiting");
    let cancelled_error = cancelled_wait
        .await
        .expect("cancelled wait task")
        .expect_err("cancelling a wait rejects only that wait");
    assert!(cancelled_error.contains("stop waiting"));

    let already_cancelled = AbortController::new();
    already_cancelled.abort("already cancelled");
    let already_cancelled_context = with_abort_signal(already_cancelled.signal(), context());
    let already_cancelled_error = root
        .wait_for_idle(&already_cancelled_context)
        .await
        .expect_err("an already-cancelled idle wait rejects");
    assert!(already_cancelled_error.contains("already cancelled"));

    harness.resume();
    let mut releases = BTreeMap::new();
    for _ in 0..3 {
        let (id, release) = tokio::time::timeout(TEST_TIMEOUT, started_rx.recv())
            .await
            .expect("gated task starts")
            .expect("start signal is sent");
        releases.insert(id, release);
    }
    let idle_context = context();
    let root_idle = root.wait_for_idle(&idle_context);
    let harness_idle = harness.wait_for_idle(&idle_context);
    tokio::pin!(root_idle);
    tokio::pin!(harness_idle);
    releases
        .remove(&foreground)
        .expect("foreground gate")
        .send(())
        .expect("release foreground task");
    root_idle
        .await
        .expect("root becomes idle while background and other-conversation work remain");
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut harness_idle)
            .await
            .is_err(),
        "harness idleness includes foreground work in another ownerless conversation"
    );
    releases
        .remove(&elsewhere)
        .expect("other conversation gate")
        .send(())
        .expect("release other-conversation task");
    harness_idle
        .await
        .expect("harness is idle after ownerless foreground work settles");
    assert_eq!(
        harness
            .get_task(background, &context())
            .await
            .expect("read background task")
            .expect("task exists")
            .state
            .status,
        TaskStateStatus::Running,
        "background tasks do not prevent idle"
    );
    releases
        .remove(&background)
        .expect("background gate")
        .send(())
        .expect("release background task");
    harness
        .wait_for_task(background, &context())
        .await
        .expect("wait for background task");
    harness.close(&context()).await.expect("close harness");
    assert!(harness
        .wait_for_idle(&context())
        .await
        .expect_err("closed harness idle wait rejects")
        .contains("closed"));
    assert!(root
        .wait_for_idle(&context())
        .await
        .expect_err("closed conversation idle wait rejects")
        .contains("closed"));
}

#[tokio::test]
async fn unknown_tasks_reject_and_terminal_tasks_are_reported() {
    let registry = registry();
    let done = finishing_task("test.scheduling-quick");
    install(&registry, "task-scheduling-unknown", vec![done.clone()]);
    let harness = open_controlled(&Arc::new(ControlledStorage::new()), &registry, None).await;
    let root = root(&harness).await;
    let unknown = TaskId(999_999);
    assert!(harness
        .get_task(unknown, &context())
        .await
        .expect("read unknown task")
        .is_none());
    assert!(harness
        .wait_for_task(unknown, &context())
        .await
        .expect_err("waiting for an unknown task rejects")
        .contains("does not exist"));
    assert!(harness
        .abort_task(unknown, &context())
        .await
        .expect_err("aborting an unknown task rejects")
        .contains("does not exist"));

    let id = start(&root, done, false).await;
    harness.resume();
    let terminal = harness
        .wait_for_task(id, &context())
        .await
        .expect("wait for completed task");
    assert_eq!(terminal.record.state.status, TaskStateStatus::Terminal);
    assert_eq!(
        harness
            .abort_task(id, &context())
            .await
            .expect("abort terminal task"),
        "terminal"
    );
    harness.close(&context()).await.expect("close harness");
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| harness.resume())).is_err());
}

#[tokio::test]
async fn task_and_idle_waits_reject_on_cancellation_or_close_while_queued() {
    let storage = Arc::new(ControlledStorage::new());
    let registry = registry();
    let (started_tx, mut started_rx) =
        tokio::sync::mpsc::unbounded_channel::<tokio::sync::oneshot::Sender<()>>();
    let blocking = task(
        "test.scheduling-wait-close",
        phase(move |_, runtime, context| {
            let started = started_tx.clone();
            Box::pin(async move {
                let (release, blocked) = tokio::sync::oneshot::channel();
                started.send(release).map_err(|error| error.to_string())?;
                blocked.await.map_err(|error| error.to_string())?;
                runtime
                    .commit(
                        Box::new(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                        &context,
                    )
                    .await?;
                Ok(())
            })
        }),
    );
    install(
        &registry,
        "task-scheduling-wait-close",
        vec![blocking.clone()],
    );
    let harness = open_controlled(&storage, &registry, None).await;
    let root = root(&harness).await;
    let id = start(&root, blocking, false).await;

    let mut held = storage.hold_commits();
    let root_for_blocker = root.clone();
    let blocker = tokio::spawn(async move {
        let conversation_id = root_for_blocker.id;
        root_for_blocker
            .commit(
                move |tx| {
                    tx.append_entry(
                        conversation_id,
                        crate::types::EntryDraft {
                            kind: Some("blocker".into()),
                            ..Default::default()
                        },
                    )
                    .map(|_| ())
                    .map_err(|error| error.to_string())
                },
                &context(),
            )
            .await
    });
    tokio::time::timeout(TEST_TIMEOUT, held.entered())
        .await
        .expect("blocker holds the Session line");
    let controller = AbortController::new();
    let cancelled_context = with_abort_signal(controller.signal(), context());
    let harness_for_cancelled_wait = harness.clone();
    let cancelled_wait = tokio::spawn(async move {
        harness_for_cancelled_wait
            .wait_for_task(id, &cancelled_context)
            .await
    });
    controller.abort("wait cancelled");
    held.release();
    blocker
        .await
        .expect("blocker task")
        .expect("blocker commit");
    let cancelled_error = cancelled_wait
        .await
        .expect("cancelled wait task")
        .expect_err("queued task wait rejects on cancellation");
    assert!(cancelled_error.contains("wait cancelled"));

    let release = tokio::time::timeout(TEST_TIMEOUT, started_rx.recv())
        .await
        .expect("task starts after the held commit")
        .expect("task start signal");
    let harness_for_pending = harness.clone();
    let pending =
        tokio::spawn(async move { harness_for_pending.wait_for_task(id, &context()).await });
    let harness_for_idle = harness.clone();
    let idle = tokio::spawn(async move { harness_for_idle.wait_for_idle(&context()).await });
    let root_for_idle = root.clone();
    let conversation_idle =
        tokio::spawn(async move { root_for_idle.wait_for_idle(&context()).await });
    flush().await;

    let mut held_again = storage.hold_commits();
    let root_for_blocker = root.clone();
    let blocker_again = tokio::spawn(async move {
        let conversation_id = root_for_blocker.id;
        root_for_blocker
            .commit(
                move |tx| {
                    tx.append_entry(
                        conversation_id,
                        crate::types::EntryDraft {
                            kind: Some("blocker".into()),
                            ..Default::default()
                        },
                    )
                    .map(|_| ())
                    .map_err(|error| error.to_string())
                },
                &context(),
            )
            .await
    });
    tokio::time::timeout(TEST_TIMEOUT, held_again.entered())
        .await
        .expect("second blocker holds the Session line");
    let harness_for_closed_wait = harness.clone();
    let closed_while_queued =
        tokio::spawn(async move { harness_for_closed_wait.wait_for_task(id, &context()).await });
    flush().await;
    let harness_for_close = harness.clone();
    let closing = tokio::spawn(async move { harness_for_close.close(&context()).await });

    held_again.release();
    blocker_again
        .await
        .expect("second blocker task")
        .expect("second blocker commit");
    let closed_error = closed_while_queued
        .await
        .expect("queued closed wait")
        .expect_err("wait queued during close rejects");
    assert!(closed_error.contains("closed"));
    assert!(pending
        .await
        .expect("pending wait")
        .expect_err("pending task wait rejects on close")
        .contains("closed"));
    assert!(idle
        .await
        .expect("harness idle wait")
        .expect_err("harness idle wait rejects on close")
        .contains("closed"));
    assert!(conversation_idle
        .await
        .expect("conversation idle wait")
        .expect_err("conversation idle wait rejects on close")
        .contains("closed"));
    release
        .send(())
        .expect("release blocked task for close to join");
    closing.await.expect("close task").expect("close harness");
}

/// Tasks page newest first by ID and older ones stay behind the cursor
/// (upstream #10546).
#[tokio::test]
async fn pages_the_newest_tasks_first_without_scanning_older_ones() {
    let idle = finishing_task("test.idle");
    let (_storage, registry, harness) = open_default().await;
    install(&registry, "idle", vec![idle]);
    let root = root(&harness).await;
    let mut ids: Vec<TaskId> = Vec::new();
    for _ in 0..5 {
        ids.push(start(&root, finishing_task("test.idle-queued"), false).await);
    }
    let first = harness
        .session
        .commit(
            move |tx| {
                tx.scan_tasks(
                    &TaskQuery {
                        order: Some(ScanOrder::Descending),
                        ..Default::default()
                    },
                    2,
                    None,
                )
                .map_err(|error| error.to_string())
            },
            &context(),
        )
        .await
        .expect("scan tasks descending");
    assert_eq!(
        first
            .items
            .iter()
            .map(|record| record.id)
            .collect::<Vec<_>>(),
        vec![ids[4], ids[3]]
    );
    let second = harness
        .session
        .commit(
            move |tx| {
                tx.scan_tasks(&TaskQuery::default(), 2, first.next.as_ref())
                    .map_err(|error| error.to_string())
            },
            &context(),
        )
        .await
        .expect("scan tasks next page");
    assert_eq!(
        second
            .items
            .iter()
            .map(|record| record.id)
            .collect::<Vec<_>>(),
        vec![ids[2], ids[1]]
    );
    harness.close(&context()).await.expect("close harness");
}

/// `startedAt` is stamped at the first run and kept through a wait;
/// `endedAt` is stamped when the task settles (upstream #10549).
#[tokio::test]
async fn stamps_started_at_at_the_first_run_and_ended_at_when_terminal() {
    let clock = Arc::new(std::sync::atomic::AtomicI64::new(1_000));
    let now = {
        let clock = clock.clone();
        Arc::new(move || clock.load(std::sync::atomic::Ordering::SeqCst))
            as crate::harness::types::NowFn
    };
    let gate = Arc::new(Notify::new());
    let held_started = Arc::new(Notify::new());
    let held = {
        let gate = gate.clone();
        let held_started = held_started.clone();
        AnyTask {
            definition: Arc::new(TaskDefinition {
                name: "test.timed-held".into(),
                version: 1,
                initial: Arc::new(|_| json!({"phase": "hold"})),
                phases: Arc::new(BTreeMap::from([(
                    "hold".to_string(),
                    phase(move |_, runtime, ctx| {
                        let gate = gate.clone();
                        let held_started = held_started.clone();
                        Box::pin(async move {
                            held_started.notify_one();
                            gate.notified().await;
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
                )])),
                abort: phase(|_, runtime, ctx| {
                    Box::pin(async move {
                        runtime
                            .commit(
                                commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("test")))),
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
    let on: Arc<Mutex<Vec<TaskId>>> = Arc::new(Mutex::new(Vec::new()));
    let waiter = {
        let on = on.clone();
        AnyTask {
            definition: Arc::new(TaskDefinition {
                name: "test.timed-waiter".into(),
                version: 1,
                initial: Arc::new(|_| json!({"phase": "wait"})),
                phases: Arc::new(BTreeMap::from([
                    (
                        "wait".to_string(),
                        phase(move |_, runtime, ctx| {
                            let on = on.lock().clone();
                            Box::pin(async move {
                                runtime
                                    .commit(
                                        commit_change(move |_, _| {
                                            Ok(Some(NextTaskState::Waiting {
                                                checkpoint: json!({"phase": "resume"}),
                                                on,
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
                    ("resume".to_string(), finish_phase()),
                ])),
                abort: phase(|_, runtime, ctx| {
                    Box::pin(async move {
                        runtime
                            .commit(
                                commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("test")))),
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
    let registry = registry();
    let storage: Arc<dyn crate::types::Storage> =
        Arc::new(crate::storage::memory::MemoryStorage::new());
    let harness = open(
        storage,
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            now: Some(now.clone()),
            ..Default::default()
        },
    )
    .await;
    install(&registry, "timed", vec![waiter.clone(), held.clone()]);
    let root = root(&harness).await;
    let held_id = start(&root, held, false).await;
    *on.lock() = vec![held_id];
    let waiter_id = root
        .commit(
            move |tx| {
                tx.create_task(
                    &waiter,
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
        .expect("create waiter");
    let times = |harness: Arc<Harness>, held_id: TaskId, waiter_id: TaskId| {
        Box::pin(async move {
            let held_record = harness
                .get_task(held_id, &context())
                .await
                .expect("read held")
                .expect("held exists");
            let waiter_record = harness
                .get_task(waiter_id, &context())
                .await
                .expect("read waiter")
                .expect("waiter exists");
            [
                (held_record.started_at, held_record.ended_at),
                (waiter_record.started_at, waiter_record.ended_at),
            ]
        })
    };
    assert_eq!(
        times(harness.clone(), held_id, waiter_id).await,
        [(None, None), (None, None)]
    );

    // First runs stamp startedAt at 2_000.
    clock.store(2_000, std::sync::atomic::Ordering::SeqCst);
    harness.resume();
    tokio::time::timeout(TEST_TIMEOUT, held_started.notified())
        .await
        .expect("held started");
    tokio::time::timeout(TEST_TIMEOUT, async {
        while harness
            .get_task(waiter_id, &context())
            .await
            .expect("read waiter")
            .map(|record| record.state.status)
            != Some(TaskStateStatus::Waiting)
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("waiter reaches waiting");
    assert_eq!(
        times(harness.clone(), held_id, waiter_id).await,
        [(Some(2_000), None), (Some(2_000), None)]
    );

    // The waiter runs again at 5_000; its start stays the first run.
    clock.store(5_000, std::sync::atomic::Ordering::SeqCst);
    gate.notify_waiters();
    let settled_waiter = harness
        .wait_for_task(waiter_id, &context())
        .await
        .expect("waiter settles");
    let settled_held = harness
        .get_task(held_id, &context())
        .await
        .expect("read held")
        .expect("held exists");
    assert_eq!(
        (settled_held.started_at, settled_held.ended_at),
        (Some(2_000), Some(5_000))
    );
    assert_eq!(
        (settled_waiter.started_at, settled_waiter.ended_at),
        (Some(2_000), Some(5_000))
    );
    harness.close(&context()).await.expect("close harness");
}
