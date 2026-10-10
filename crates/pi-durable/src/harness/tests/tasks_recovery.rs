use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use pi_chord::context::Context;
use serde_json::{json, Map, Value};
use tokio::sync::Notify;

use super::support::{context, open, root};
use crate::documents::{DocDefinition, DocToken, ScopeKind};
use crate::harness::harness::{open_harness, Conversation, Harness};
use crate::harness::registry::{create_registry, RegistryImpl};
use crate::harness::types::{
    AnyTask, Extension, Registry, RegistryReader, RegistrySnapshot, ReportHook, Unsubscribe,
};
use crate::session::support::ControlledStorage;
use crate::tasks::{
    commit_change, define_task, BoxFuture, NextTaskState, PhaseHandler, Task, TaskDefinition,
    TaskMigration, TaskRuntime,
};
use crate::types::{
    EntryDraft, Storage, TaskId, TaskOptions, TaskOutcomeStatus, TaskOwnership, TaskRecord,
    TaskStateStatus,
};

const WAIT: Duration = Duration::from_secs(5);

type Migration = Arc<dyn Fn(&Value, &Value, u64) -> Result<TaskMigration, String> + Send + Sync>;
type Log = Arc<Mutex<Vec<String>>>;

fn handler(
    f: impl Fn(TaskRecord, Arc<dyn TaskRuntime>, Context) -> BoxFuture<Result<(), String>>
        + Send
        + Sync
        + 'static,
) -> PhaseHandler {
    Arc::new(f)
}

fn make_task(
    name: &str,
    version: u64,
    initial: Value,
    phases: impl IntoIterator<Item = (String, PhaseHandler)>,
    abort: PhaseHandler,
    migrate: Option<Migration>,
) -> Task {
    define_task(TaskDefinition {
        name: name.into(),
        version,
        initial: Arc::new(move |_| initial.clone()),
        phases: Arc::new(phases.into_iter().collect::<BTreeMap<_, _>>()),
        abort,
        migrate,
        hooks: None,
    })
}

fn any(task: &Task) -> AnyTask {
    AnyTask::shared(task.definition.clone())
}

fn register(registry: &Arc<RegistryImpl>, extension: &str, tasks: impl IntoIterator<Item = Task>) {
    registry.install(
        Extension::new(extension).with_tasks(tasks.into_iter().map(|task| any(&task)).collect()),
    );
}

fn completed(result: Value) -> PhaseHandler {
    handler(move |_record, runtime, ctx| {
        let result = result.clone();
        Box::pin(async move {
            runtime
                .commit(
                    commit_change(move |_, _| Ok(Some(NextTaskState::completed(result)))),
                    &ctx,
                )
                .await?;
            Ok(())
        })
    })
}

fn aborted(reason: impl Into<String> + Clone + Send + Sync + 'static) -> PhaseHandler {
    let reason = reason.into();
    handler(move |_record, runtime, ctx| {
        let reason = reason.clone();
        Box::pin(async move {
            runtime
                .commit(
                    commit_change(move |_, _| Ok(Some(NextTaskState::aborted_with(reason)))),
                    &ctx,
                )
                .await?;
            Ok(())
        })
    })
}

fn versioned(version: u64, result: &'static str, migrate: Option<Migration>) -> Task {
    make_task(
        "test.recovery-versioned",
        version,
        json!({"phase":"run"}),
        [("run".into(), completed(json!(result)))],
        aborted(result),
        migrate,
    )
}

async fn create(harness: &Arc<Harness>, task: &Task, input: Value) -> TaskId {
    let conversation = root(harness).await;
    let executable = any(task);
    conversation
        .commit(
            move |tx| {
                tx.create_task(
                    &executable,
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
        .expect("create recovery task")
}

async fn wait_record(
    harness: &Arc<Harness>,
    id: TaskId,
    mut predicate: impl FnMut(&TaskRecord) -> bool,
) -> TaskRecord {
    tokio::time::timeout(WAIT, async {
        loop {
            let record = harness
                .get_task(id, &context())
                .await
                .expect("read task")
                .expect("task exists");
            if predicate(&record) {
                return record;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("task did not reach the expected state")
}

async fn wait_log(log: &Log, length: usize) {
    tokio::time::timeout(WAIT, async {
        while log.lock().len() < length {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("task handler did not reach the expected gate");
}

async fn ticks() {
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
}

fn object(value: Value) -> Map<String, Value> {
    value.as_object().expect("JSON object").clone()
}

async fn sqlite(path: &Path) -> Arc<dyn Storage> {
    Arc::new(
        crate::storage::sqlite::SqliteStorage::open_path(
            path,
            &crate::storage::sqlite::NodeSqliteStorageOptions::default(),
        )
        .await
        .expect("open sqlite storage"),
    )
}

fn transfer_task(
    calls: Arc<Mutex<(u64, HashMap<String, u64>)>>,
    interrupt_first: Arc<std::sync::atomic::AtomicBool>,
    applied: Arc<Notify>,
) -> Task {
    let prepare = handler(|record, runtime, ctx| {
        Box::pin(async move {
            let requested = record.input["amount"].clone();
            runtime.memo_put("requested", requested, &ctx).await?;
            let id = record.id;
            runtime
                .commit(
                    commit_change(move |_, _| {
                        Ok(Some(NextTaskState::Running {
                            checkpoint: json!({"phase":"apply","key":format!("transfer-{id}")}),
                        }))
                    }),
                    &ctx,
                )
                .await?;
            Ok(())
        })
    });
    let apply = {
        let calls = calls.clone();
        handler(move |record, runtime, ctx| {
            let calls = calls.clone();
            let interrupt_first = interrupt_first.clone();
            let applied = applied.clone();
            Box::pin(async move {
                let key = record
                    .state
                    .checkpoint
                    .as_ref()
                    .and_then(|value| value["key"].as_str())
                    .expect("apply key")
                    .to_string();
                let amount = record.input["amount"].as_u64().expect("transfer amount");
                let receipt = {
                    let mut service = calls.lock();
                    service.0 += 1;
                    *service.1.entry(key.clone()).or_insert(amount * 10)
                };
                if interrupt_first.swap(false, std::sync::atomic::Ordering::SeqCst) {
                    applied.notify_one();
                    runtime.signal().aborted_future().await;
                    return Ok(());
                }
                runtime
                    .commit(
                        commit_change(move |_, _| {
                            Ok(Some(NextTaskState::completed(json!({"receipt":receipt}))))
                        }),
                        &ctx,
                    )
                    .await?;
                Ok(())
            })
        })
    };
    make_task(
        "test.recovery-transfer",
        1,
        json!({"phase":"prepare"}),
        [("prepare".into(), prepare), ("apply".into(), apply)],
        aborted("stop"),
        None,
    )
}

#[tokio::test]
async fn intent_effect_and_outcome_survive_close_and_reopen() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("session.sqlite");
    let calls = Arc::new(Mutex::new((0, HashMap::new())));
    let interrupt_first = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let applied = Arc::new(Notify::new());
    let transfer = transfer_task(calls.clone(), interrupt_first.clone(), applied.clone());
    let registry = Arc::new(create_registry());
    register(&registry, "recovery-transfer", [transfer.clone()]);

    let first = open(
        sqlite(&path).await,
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let id = create(&first, &transfer, json!({"amount":7})).await;
    first.resume();
    tokio::time::timeout(WAIT, applied.notified())
        .await
        .expect("effect completed before interruption");
    first.close(&context()).await.expect("close first harness");

    let second = open(
        sqlite(&path).await,
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let pending = second
        .get_task(id, &context())
        .await
        .expect("read recovered task")
        .expect("task survives");
    assert!(!pending.abort_requested);
    assert_eq!(
        pending
            .memos
            .as_ref()
            .and_then(|memos| memos.get("requested")),
        Some(&json!(7))
    );
    assert_eq!(pending.state.status, TaskStateStatus::Pending);
    assert_eq!(
        pending.state.checkpoint,
        Some(json!({"phase":"apply","key":format!("transfer-{id}")}))
    );
    assert_eq!(calls.lock().0, 1);
    second.resume();
    let receipt = second
        .wait_for_task(id, &context())
        .await
        .expect("complete recovered transfer");
    assert_eq!(
        receipt
            .state
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.result.as_ref()),
        Some(&json!({"receipt":70}))
    );
    assert_eq!(calls.lock().0, 2);
    assert_eq!(
        calls.lock().1.len(),
        1,
        "idempotency key admits one transfer effect"
    );
    second
        .close(&context())
        .await
        .expect("close second harness");

    let third = open(
        sqlite(&path).await,
        crate::harness::types::HarnessOptions {
            registry,
            ..Default::default()
        },
    )
    .await;
    third.resume();
    assert_eq!(
        third
            .wait_for_task(id, &context())
            .await
            .expect("read durable receipt"),
        receipt
    );
    assert_eq!(calls.lock().0, 2);
    third.close(&context()).await.expect("close third harness");
}

#[tokio::test]
async fn abort_recovery_resumes_at_each_close_stage() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("session.sqlite");
    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let run_release = Arc::new(Notify::new());
    let abort_reached = Arc::new(Notify::new());
    let block_abort = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let run = {
        let log = log.clone();
        let run_release = run_release.clone();
        handler(move |_record, _runtime, _ctx| {
            let log = log.clone();
            let run_release = run_release.clone();
            Box::pin(async move {
                log.lock().push("run".into());
                run_release.notified().await;
                Ok(())
            })
        })
    };
    let abort = {
        let log = log.clone();
        let abort_reached = abort_reached.clone();
        let block_abort = block_abort.clone();
        handler(move |_record, runtime, ctx| {
            let log = log.clone();
            let abort_reached = abort_reached.clone();
            let block_abort = block_abort.clone();
            Box::pin(async move {
                log.lock().push("abort".into());
                if block_abort.load(std::sync::atomic::Ordering::SeqCst) {
                    abort_reached.notify_one();
                    runtime.signal().aborted_future().await;
                    return Ok(());
                }
                runtime
                    .commit(
                        commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("stop")))),
                        &ctx,
                    )
                    .await?;
                Ok(())
            })
        })
    };
    let abortable = make_task(
        "test.recovery-abort",
        1,
        json!({"phase":"run"}),
        [("run".into(), run)],
        abort,
        None,
    );
    let registry = Arc::new(create_registry());
    register(&registry, "recovery-abort", [abortable.clone()]);

    let mut harness = open(
        sqlite(&path).await,
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let id = create(&harness, &abortable, Value::Null).await;
    harness.resume();
    wait_log(&log, 1).await;
    let old = harness.clone();
    let aborting = tokio::spawn(async move { old.abort_task(id, &context()).await });
    wait_record(&harness, id, |record| record.abort_requested).await;
    let closing = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.close(&context()).await })
    };
    run_release.notify_one();
    closing
        .await
        .expect("close task")
        .expect("close first harness");
    assert_eq!(
        aborting.await.expect("abort call").expect("abort mark"),
        "marked"
    );
    assert_eq!(*log.lock(), ["run"]);

    harness = open(
        sqlite(&path).await,
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let pending = harness
        .get_task(id, &context())
        .await
        .expect("read abort-marked task")
        .expect("task exists");
    assert!(pending.abort_requested);
    assert_eq!(pending.state.status, TaskStateStatus::Pending);
    harness.resume();
    tokio::time::timeout(WAIT, abort_reached.notified())
        .await
        .expect("abort invocation started");
    harness
        .close(&context())
        .await
        .expect("close active abort handler");
    assert_eq!(*log.lock(), ["run", "abort"]);

    block_abort.store(false, std::sync::atomic::Ordering::SeqCst);
    harness = open(
        sqlite(&path).await,
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    harness.resume();
    let receipt = harness
        .wait_for_task(id, &context())
        .await
        .expect("fresh abort invocation settles");
    assert_eq!(
        receipt.state.outcome.as_ref().map(|outcome| outcome.status),
        Some(TaskOutcomeStatus::Aborted)
    );
    assert_eq!(
        receipt
            .state
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.reason.as_deref()),
        Some("stop")
    );
    harness
        .close(&context())
        .await
        .expect("close recovered abort");
    assert_eq!(*log.lock(), ["run", "abort", "abort"]);

    harness = open(
        sqlite(&path).await,
        crate::harness::types::HarnessOptions {
            registry,
            ..Default::default()
        },
    )
    .await;
    harness.resume();
    ticks().await;
    assert_eq!(
        harness
            .abort_task(id, &context())
            .await
            .expect("abort terminal task"),
        "terminal"
    );
    assert_eq!(*log.lock(), ["run", "abort", "abort"]);
    harness
        .close(&context())
        .await
        .expect("close terminal harness");
}

fn crash_task(log: Log, block_abort: bool) -> Task {
    let run = {
        let log = log.clone();
        handler(move |_record, _runtime, _ctx| {
            let log = log.clone();
            Box::pin(async move {
                log.lock().push("run".into());
                std::future::pending::<()>().await;
                Ok(())
            })
        })
    };
    let abort = handler(move |_record, runtime, ctx| {
        let log = log.clone();
        Box::pin(async move {
            log.lock().push("abort".into());
            if block_abort {
                std::future::pending::<()>().await;
            }
            runtime
                .commit(
                    commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("recovered")))),
                    &ctx,
                )
                .await?;
            Ok(())
        })
    });
    make_task(
        "test.recovery-crash",
        1,
        json!({"phase":"run"}),
        [("run".into(), run)],
        abort,
        None,
    )
}

async fn crashed_run(
    log: Log,
) -> (
    Arc<ControlledStorage>,
    Arc<Harness>,
    Arc<RegistryImpl>,
    TaskId,
) {
    let storage = Arc::new(ControlledStorage::new());
    let registry = Arc::new(create_registry());
    let task = crash_task(log.clone(), true);
    register(&registry, "recovery-crash", [task.clone()]);
    let harness = open(
        storage.clone(),
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let id = create(&harness, &task, Value::Null).await;
    harness.resume();
    wait_log(&log, 1).await;
    (storage, harness, registry, id)
}

async fn recover_controlled(
    storage: Arc<ControlledStorage>,
    registry: Arc<RegistryImpl>,
    log: Log,
    id: TaskId,
) -> Arc<Harness> {
    storage.crash();
    let task = crash_task(log, false);
    register(&registry, "recovery-crash", [task]);
    open(
        storage,
        crate::harness::types::HarnessOptions {
            registry,
            ..Default::default()
        },
    )
    .await
}

#[tokio::test]
async fn crash_during_abort_mark_commit_resumes_the_unmarked_run() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let (storage, _harness, registry, id) = crashed_run(log.clone()).await;
    let mut held = storage.hold_commits();
    let old = _harness.clone();
    let _aborting = tokio::spawn(async move { old.abort_task(id, &context()).await });
    tokio::time::timeout(WAIT, held.entered())
        .await
        .expect("abort mark commit reached storage");
    let recovered = recover_controlled(storage, registry, log.clone(), id).await;
    let record = recovered
        .get_task(id, &context())
        .await
        .expect("read recovered task")
        .expect("task exists");
    assert!(!record.abort_requested);
    assert_eq!(record.state.status, TaskStateStatus::Pending);
    recovered.resume();
    wait_log(&log, 2).await;
    assert_eq!(*log.lock(), ["run", "run"]);
}

#[tokio::test]
async fn crash_after_abort_mark_before_run_join_dispatches_only_abort() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let (storage, harness, registry, id) = crashed_run(log.clone()).await;
    let old = harness.clone();
    let joined = tokio::spawn(async move { old.abort_task(id, &context()).await });
    let marked = wait_record(&harness, id, |record| record.abort_requested).await;
    assert_eq!(marked.state.status, TaskStateStatus::Running);
    ticks().await;
    assert!(
        !joined.is_finished(),
        "abort waits for the run invocation to join"
    );
    let recovered = recover_controlled(storage, registry, log.clone(), id).await;
    let record = recovered
        .get_task(id, &context())
        .await
        .expect("read recovered mark")
        .expect("task exists");
    assert!(record.abort_requested);
    assert_eq!(record.state.status, TaskStateStatus::Pending);
    recovered.resume();
    let receipt = recovered
        .wait_for_task(id, &context())
        .await
        .expect("recovered abort settles");
    assert_eq!(
        receipt.state.outcome.as_ref().map(|outcome| outcome.status),
        Some(TaskOutcomeStatus::Aborted)
    );
    assert_eq!(
        receipt
            .state
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.reason.as_deref()),
        Some("recovered")
    );
    assert_eq!(*log.lock(), ["run", "abort"]);
    recovered
        .close(&context())
        .await
        .expect("close recovered harness");
}

#[tokio::test]
async fn crash_during_abort_handler_starts_a_fresh_abort_invocation() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let storage = Arc::new(ControlledStorage::new());
    let registry = Arc::new(create_registry());
    let task = crash_task(log.clone(), true);
    register(&registry, "recovery-crash", [task.clone()]);
    let harness = open(
        storage.clone(),
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let id = create(&harness, &task, Value::Null).await;
    assert_eq!(
        harness
            .abort_task(id, &context())
            .await
            .expect("mark before resume"),
        "marked"
    );
    harness.resume();
    wait_log(&log, 1).await;
    assert_eq!(
        harness
            .get_task(id, &context())
            .await
            .unwrap()
            .unwrap()
            .state
            .status,
        TaskStateStatus::Running
    );
    let recovered = recover_controlled(storage, registry, log.clone(), id).await;
    recovered.resume();
    let receipt = recovered
        .wait_for_task(id, &context())
        .await
        .expect("recovered abort handler settles");
    assert_eq!(
        receipt.state.outcome.as_ref().map(|outcome| outcome.status),
        Some(TaskOutcomeStatus::Aborted)
    );
    assert_eq!(*log.lock(), ["abort", "abort"]);
    recovered
        .close(&context())
        .await
        .expect("close recovered harness");
}

#[tokio::test]
async fn crash_during_abort_outcome_commit_runs_the_abort_handler_again() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let storage = Arc::new(ControlledStorage::new());
    let registry = Arc::new(create_registry());
    let reached = Arc::new(Notify::new());
    let proceed = Arc::new(Notify::new());
    let first_abort = {
        let log = log.clone();
        let reached = reached.clone();
        let proceed = proceed.clone();
        handler(move |_record, runtime, ctx| {
            let log = log.clone();
            let reached = reached.clone();
            let proceed = proceed.clone();
            Box::pin(async move {
                log.lock().push("abort".into());
                reached.notify_one();
                proceed.notified().await;
                runtime
                    .commit(
                        commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("lost")))),
                        &ctx,
                    )
                    .await?;
                Ok(())
            })
        })
    };
    let task = make_task(
        "test.recovery-crash",
        1,
        json!({"phase":"run"}),
        [("run".into(), completed(Value::Null))],
        first_abort,
        None,
    );
    register(&registry, "recovery-crash", [task.clone()]);
    let harness = open(
        storage.clone(),
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let id = create(&harness, &task, Value::Null).await;
    assert_eq!(
        harness.abort_task(id, &context()).await.expect("mark task"),
        "marked"
    );
    harness.resume();
    tokio::time::timeout(WAIT, reached.notified())
        .await
        .expect("abort invocation reached gate");
    let mut held = storage.hold_commits();
    proceed.notify_one();
    tokio::time::timeout(WAIT, held.entered())
        .await
        .expect("abort outcome commit reached storage");
    let recovered = recover_controlled(storage, registry, log.clone(), id).await;
    recovered.resume();
    let receipt = recovered
        .wait_for_task(id, &context())
        .await
        .expect("replayed abort outcome");
    assert_eq!(
        receipt.state.outcome.as_ref().map(|outcome| outcome.status),
        Some(TaskOutcomeStatus::Aborted)
    );
    assert_eq!(
        receipt
            .state
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.reason.as_deref()),
        Some("recovered")
    );
    assert_eq!(*log.lock(), ["abort", "abort"]);
    recovered
        .close(&context())
        .await
        .expect("close recovered harness");
}

#[tokio::test]
async fn crash_after_terminal_outcome_does_not_run_the_task_again() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let storage = Arc::new(ControlledStorage::new());
    let registry = Arc::new(create_registry());
    let task = crash_task(log.clone(), false);
    register(&registry, "recovery-crash", [task.clone()]);
    let harness = open(
        storage.clone(),
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let id = create(&harness, &task, Value::Null).await;
    assert_eq!(
        harness.abort_task(id, &context()).await.expect("mark task"),
        "marked"
    );
    harness.resume();
    harness
        .wait_for_task(id, &context())
        .await
        .expect("first abort settles");
    let recovered = recover_controlled(storage, registry, log.clone(), id).await;
    recovered.resume();
    ticks().await;
    let record = recovered
        .get_task(id, &context())
        .await
        .expect("read terminal task")
        .expect("task exists");
    assert_eq!(record.state.status, TaskStateStatus::Terminal);
    assert_eq!(
        record.state.outcome.as_ref().map(|outcome| outcome.status),
        Some(TaskOutcomeStatus::Aborted)
    );
    assert_eq!(*log.lock(), ["abort"]);
    recovered
        .close(&context())
        .await
        .expect("close recovered harness");
}

#[tokio::test]
async fn crash_during_reservation_commit_leaves_the_task_pending() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let storage = Arc::new(ControlledStorage::new());
    let registry = Arc::new(create_registry());
    let task = crash_task(log.clone(), false);
    register(&registry, "recovery-crash", [task.clone()]);
    let harness = open(
        storage.clone(),
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let id = create(&harness, &task, Value::Null).await;
    let mut held = storage.hold_commits();
    harness.resume();
    tokio::time::timeout(WAIT, held.entered())
        .await
        .expect("reservation commit reached storage");
    let recovered = recover_controlled(storage, registry, log.clone(), id).await;
    let record = recovered
        .get_task(id, &context())
        .await
        .expect("read pending task")
        .expect("task exists");
    assert_eq!(record.state.status, TaskStateStatus::Pending);
    assert!(
        log.lock().is_empty(),
        "reservation crash dispatches no invocation"
    );
    assert_eq!(
        recovered
            .abort_task(id, &context())
            .await
            .expect("mark pending task"),
        "marked"
    );
    recovered.resume();
    recovered
        .wait_for_task(id, &context())
        .await
        .expect("abort pending task");
    assert_eq!(*log.lock(), ["abort"]);
    recovered
        .close(&context())
        .await
        .expect("close recovered harness");
}

#[tokio::test]
async fn missing_definition_stays_live_for_idle_wait_until_registered() {
    let registry = Arc::new(create_registry());
    let task = versioned(1, "v1", None);
    let harness = open(
        Arc::new(crate::storage::memory::MemoryStorage::new()),
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let id = create(&harness, &task, Value::Null).await;
    harness.resume();
    ticks().await;
    assert_eq!(
        harness
            .get_task(id, &context())
            .await
            .unwrap()
            .unwrap()
            .state
            .status,
        TaskStateStatus::Pending
    );
    let idle = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.wait_for_idle(&context()).await })
    };
    ticks().await;
    assert!(
        !idle.is_finished(),
        "blocked task remains part of the idle wait"
    );
    register(&registry, "recovery-versioned", [task]);
    idle.await
        .expect("idle waiter")
        .expect("wait for recovered task");
    assert_eq!(
        harness
            .get_task(id, &context())
            .await
            .unwrap()
            .unwrap()
            .state
            .outcome
            .unwrap()
            .result,
        Some(json!("v1"))
    );
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn newer_stored_definition_waits_for_a_fitting_registration() {
    let registry = Arc::new(create_registry());
    let old = versioned(1, "old", None);
    register(&registry, "recovery-versioned", [old]);
    let harness = open(
        Arc::new(crate::storage::memory::MemoryStorage::new()),
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let id = create(&harness, &versioned(2, "new", None), Value::Null).await;
    harness.resume();
    ticks().await;
    assert_eq!(
        harness
            .get_task(id, &context())
            .await
            .unwrap()
            .unwrap()
            .state
            .status,
        TaskStateStatus::Pending
    );
    register(&registry, "recovery-versioned", [versioned(2, "new", None)]);
    let receipt = harness
        .wait_for_task(id, &context())
        .await
        .expect("newer task definition settles");
    assert_eq!(
        receipt
            .state
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.result.as_ref()),
        Some(&json!("new"))
    );
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn failed_migration_is_not_retried_until_the_definition_changes() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("session.sqlite");
    let old = versioned(1, "v1", None);
    let first = open(sqlite(&path).await, Default::default()).await;
    let id = create(&first, &old, Value::Null).await;
    first.close(&context()).await.expect("close v1 harness");

    let failures = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let broken_failures = failures.clone();
    let broken: Migration = Arc::new(move |_, _, _| {
        broken_failures.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Err("cannot migrate".into())
    });
    let registry = Arc::new(create_registry());
    register(
        &registry,
        "recovery-versioned",
        [versioned(2, "v2", Some(broken))],
    );
    let reports = Arc::new(Mutex::new(Vec::<String>::new()));
    let report_slot = reports.clone();
    let on_report: ReportHook = Arc::new(move |message| report_slot.lock().push(message));
    let harness = open(
        sqlite(&path).await,
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            on_report: Some(on_report),
            ..Default::default()
        },
    )
    .await;
    harness.resume();
    tokio::time::timeout(WAIT, async {
        while reports.lock().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("failed migration is reported");
    registry.install(Extension::new("unrelated-recovery-extension"));
    ticks().await;
    assert_eq!(failures.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(reports.lock().len(), 1);
    assert!(reports.lock()[0].contains("cannot migrate"));
    let unchanged = harness.get_task(id, &context()).await.unwrap().unwrap();
    assert_eq!(unchanged.version, 1);
    assert_eq!(unchanged.state.status, TaskStateStatus::Pending);
    assert_eq!(unchanged.state.checkpoint, Some(json!({"phase":"run"})));

    let migrated_from = Arc::new(Mutex::new(Vec::new()));
    let observed_from = migrated_from.clone();
    let migration: Migration = Arc::new(move |input, checkpoint, from| {
        observed_from.lock().push(from);
        Ok(TaskMigration {
            input: input.clone(),
            checkpoint: checkpoint.clone(),
        })
    });
    register(
        &registry,
        "recovery-versioned",
        [versioned(2, "v2", Some(migration))],
    );
    let receipt = harness
        .wait_for_task(id, &context())
        .await
        .expect("replacement migration settles");
    assert_eq!(receipt.version, 2);
    assert_eq!(
        receipt
            .state
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.result.as_ref()),
        Some(&json!("v2"))
    );
    assert_eq!(*migrated_from.lock(), [1]);
    harness
        .close(&context())
        .await
        .expect("close migrated harness");
}

#[tokio::test]
async fn older_record_without_migration_remains_blocked() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("session.sqlite");
    let first = open(sqlite(&path).await, Default::default()).await;
    let id = create(&first, &versioned(1, "v1", None), Value::Null).await;
    first.close(&context()).await.expect("close v1 harness");
    let registry = Arc::new(create_registry());
    register(&registry, "recovery-versioned", [versioned(2, "v2", None)]);
    let reports = Arc::new(Mutex::new(Vec::<String>::new()));
    let report_slot = reports.clone();
    let harness = open(
        sqlite(&path).await,
        crate::harness::types::HarnessOptions {
            registry,
            on_report: Some(Arc::new(move |message| report_slot.lock().push(message))),
            ..Default::default()
        },
    )
    .await;
    harness.resume();
    tokio::time::timeout(WAIT, async {
        while reports.lock().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("missing migration is reported");
    assert!(reports.lock()[0].contains("has no migration from 1"));
    let record = harness.get_task(id, &context()).await.unwrap().unwrap();
    assert_eq!(record.version, 1);
    assert_eq!(record.state.status, TaskStateStatus::Pending);
    harness
        .close(&context())
        .await
        .expect("close blocked harness");
}

#[tokio::test]
async fn aborting_a_missing_task_orphans_it_and_retires_task_documents() {
    let token: DocToken =
        DocDefinition::singleton("test.recovery-scratch", 1, ScopeKind::Task, || {
            object(json!({"n":0}))
        })
        .into();
    let task = versioned(1, "x", None);
    let harness = open(
        Arc::new(crate::storage::memory::MemoryStorage::new()),
        Default::default(),
    )
    .await;
    let executable = any(&task);
    let token_for_create = token.clone();
    let root = root(&harness).await;
    let id = root
        .commit(
            move |tx| {
                let id = tx
                    .create_task(
                        &executable,
                        Value::Null,
                        TaskOptions {
                            ownership: TaskOwnership::Conversation,
                            conversation_id: None,
                            background: None,
                        },
                    )
                    .map_err(|error| error.to_string())?;
                tx.doc(&token_for_create, Some(id.0), None, None)
                    .map_err(|error| error.to_string())?
                    .set(&[pi_chord::delta::Seg::key("n")], json!(1))
                    .map_err(|error| error.to_string())?;
                Ok(id)
            },
            &context(),
        )
        .await
        .expect("create orphan task and document");
    assert_eq!(
        harness
            .abort_task(id, &context())
            .await
            .expect("abort missing task"),
        "marked"
    );
    let record = harness.get_task(id, &context()).await.unwrap().unwrap();
    assert_eq!(record.state.status, TaskStateStatus::Terminal);
    assert_eq!(
        record.state.outcome.as_ref().map(|outcome| outcome.status),
        Some(TaskOutcomeStatus::Orphaned)
    );
    assert_eq!(
        record
            .state
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.reason.as_deref()),
        Some("missing_task")
    );
    assert!(harness
        .session
        .snapshot(&token, Some(id.0), None, &context())
        .await
        .expect("read retired doc")
        .is_none());
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn marked_active_task_with_removed_definition_is_orphaned_without_abort_handler() {
    let reached = Arc::new(Notify::new());
    let run = {
        let reached = reached.clone();
        handler(move |_record, runtime, _ctx| {
            let reached = reached.clone();
            Box::pin(async move {
                reached.notify_one();
                runtime.signal().aborted_future().await;
                Ok(())
            })
        })
    };
    let panic_abort = handler(|_record, _runtime, _ctx| {
        Box::pin(async { Err("removed task abort must not run".into()) })
    });
    let task = make_task(
        "test.recovery-vanishing",
        1,
        json!({"phase":"run"}),
        [("run".into(), run)],
        panic_abort,
        None,
    );
    let registry = Arc::new(create_registry());
    let extension = Extension::new("recovery-vanishing").with_tasks(vec![any(&task)]);
    registry.install(extension.clone());
    let harness = open(
        Arc::new(crate::storage::memory::MemoryStorage::new()),
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let id = create(&harness, &task, Value::Null).await;
    harness.resume();
    tokio::time::timeout(WAIT, reached.notified())
        .await
        .expect("run phase starts");
    registry.uninstall(&extension);
    assert_eq!(
        harness
            .abort_task(id, &context())
            .await
            .expect("mark disappearing task"),
        "marked"
    );
    let receipt = harness
        .wait_for_task(id, &context())
        .await
        .expect("removed task becomes orphaned");
    assert_eq!(
        receipt.state.outcome.as_ref().map(|outcome| outcome.status),
        Some(TaskOutcomeStatus::Orphaned)
    );
    assert_eq!(
        receipt
            .state
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.reason.as_deref()),
        Some("missing_task")
    );
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn reopened_abort_mark_without_definition_orphans_when_scheduling_resumes() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("session.sqlite");
    let task = versioned(1, "x", None);
    let registry = Arc::new(create_registry());
    register(&registry, "recovery-versioned", [task.clone()]);
    let first = open(
        sqlite(&path).await,
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let id = create(&first, &task, Value::Null).await;
    assert_eq!(
        first
            .abort_task(id, &context())
            .await
            .expect("mark before resume"),
        "marked"
    );
    first
        .close(&context())
        .await
        .expect("close marked task harness");
    let reopened = open(sqlite(&path).await, Default::default()).await;
    let pending = reopened.get_task(id, &context()).await.unwrap().unwrap();
    assert!(pending.abort_requested);
    assert_eq!(pending.state.status, TaskStateStatus::Pending);
    reopened.resume();
    reopened
        .wait_for_idle(&context())
        .await
        .expect("orphan settles idle scope");
    let record = reopened.get_task(id, &context()).await.unwrap().unwrap();
    assert_eq!(record.state.status, TaskStateStatus::Terminal);
    assert_eq!(
        record.state.outcome.as_ref().map(|outcome| outcome.status),
        Some(TaskOutcomeStatus::Orphaned)
    );
    assert_eq!(
        record
            .state
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.reason.as_deref()),
        Some("missing_task")
    );
    reopened
        .close(&context())
        .await
        .expect("close orphan harness");
}

#[derive(Default)]
struct HandoverGates {
    a: Option<Arc<Notify>>,
    b: Option<Arc<Notify>>,
    on_a_end: Option<Arc<dyn Fn() + Send + Sync>>,
}

fn handover_task(
    label: &'static str,
    version: u64,
    log: Log,
    gates: HandoverGates,
    migrate: Option<Migration>,
) -> Task {
    let advance = |phase: &'static str,
                   next: Value,
                   gate: Option<Arc<Notify>>,
                   on_end: Option<Arc<dyn Fn() + Send + Sync>>| {
        let log = log.clone();
        handler(move |_record, runtime, ctx| {
            let log = log.clone();
            let gate = gate.clone();
            let on_end = on_end.clone();
            let next = next.clone();
            Box::pin(async move {
                log.lock().push(format!("{label}:{phase} start"));
                if let Some(gate) = gate {
                    gate.notified().await;
                }
                runtime
                    .commit(
                        commit_change(move |_, _| {
                            Ok(Some(NextTaskState::Running { checkpoint: next }))
                        }),
                        &ctx,
                    )
                    .await?;
                ticks().await;
                if let Some(on_end) = on_end {
                    on_end();
                }
                log.lock().push(format!("{label}:{phase} end"));
                Ok(())
            })
        })
    };
    let a_gate = gates.a.clone();
    let b_gate = gates.b.clone();
    let on_a_end = gates.on_a_end.clone();
    let phases = [
        (
            "a".into(),
            advance("a", json!({"phase":"b"}), a_gate, on_a_end),
        ),
        ("b".into(), advance("b", json!({"phase":"c"}), b_gate, None)),
        (
            "c".into(),
            handler({
                let log = log.clone();
                move |_record, runtime, ctx| {
                    let log = log.clone();
                    Box::pin(async move {
                        log.lock().push(format!("{label}:c"));
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
                }
            }),
        ),
    ];
    let abort = {
        let log = log.clone();
        handler(move |_record, runtime, ctx| {
            let log = log.clone();
            Box::pin(async move {
                log.lock().push(format!("{label}:abort"));
                runtime
                    .commit(
                        commit_change(move |_, _| Ok(Some(NextTaskState::aborted_with(label)))),
                        &ctx,
                    )
                    .await?;
                Ok(())
            })
        })
    };
    make_task(
        "test.recovery-handover",
        version,
        json!({"phase":"a"}),
        phases,
        abort,
        migrate,
    )
}

async fn start_handover(
    log: Log,
    gates: HandoverGates,
    storage: Arc<dyn Storage>,
    on_report: Option<ReportHook>,
) -> (Arc<Harness>, Arc<RegistryImpl>, Task, TaskId) {
    let registry = Arc::new(create_registry());
    let old = handover_task("old", 1, log.clone(), gates, None);
    register(&registry, "recovery-handover", [old.clone()]);
    let harness = open(
        storage,
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            on_report,
            ..Default::default()
        },
    )
    .await;
    let id = create(&harness, &old, Value::Null).await;
    harness.resume();
    wait_log(&log, 1).await;
    (harness, registry, old, id)
}

#[tokio::test]
async fn same_version_handover_occurs_at_phase_boundary_without_overlap() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let gate = Arc::new(Notify::new());
    let (harness, registry, _old, id) = start_handover(
        log.clone(),
        HandoverGates {
            a: Some(gate.clone()),
            ..Default::default()
        },
        Arc::new(crate::storage::memory::MemoryStorage::new()),
        None,
    )
    .await;
    register(
        &registry,
        "recovery-handover",
        [handover_task(
            "new",
            1,
            log.clone(),
            HandoverGates::default(),
            None,
        )],
    );
    gate.notify_one();
    harness
        .wait_for_task(id, &context())
        .await
        .expect("handover settles");
    assert_eq!(
        *log.lock(),
        [
            "old:a start",
            "old:a end",
            "new:b start",
            "new:b end",
            "new:c"
        ]
    );
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn first_writer_memos_survive_a_definition_handover() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let gate = Arc::new(Notify::new());
    let reached = Arc::new(Notify::new());
    let memo_task = |label: &'static str, log: Log, gate: Arc<Notify>, reached: Arc<Notify>| {
        let a = {
            let gate = gate.clone();
            let reached = reached.clone();
            handler(move |_record, runtime, ctx| {
                let gate = gate.clone();
                let reached = reached.clone();
                Box::pin(async move {
                    runtime.memo_put("picked", json!(label), &ctx).await?;
                    reached.notify_one();
                    gate.notified().await;
                    runtime
                        .commit(
                            commit_change(|_, _| {
                                Ok(Some(NextTaskState::Running {
                                    checkpoint: json!({"phase":"b"}),
                                }))
                            }),
                            &ctx,
                        )
                        .await?;
                    Ok(())
                })
            })
        };
        let b = {
            let log = log.clone();
            handler(move |_record, runtime, ctx| {
                let log = log.clone();
                Box::pin(async move {
                    let winner = runtime.memo_put("picked", json!(label), &ctx).await?;
                    log.lock().push(format!(
                        "{label}:b {}",
                        winner.as_str().expect("memo value")
                    ));
                    runtime
                        .commit(
                            commit_change(move |_, _| Ok(Some(NextTaskState::completed(winner)))),
                            &ctx,
                        )
                        .await?;
                    Ok(())
                })
            })
        };
        make_task(
            "test.recovery-handover-memo",
            1,
            json!({"phase":"a"}),
            [("a".into(), a), ("b".into(), b)],
            aborted("stop"),
            None,
        )
    };
    let registry = Arc::new(create_registry());
    let old = memo_task("old", log.clone(), gate.clone(), reached.clone());
    register(&registry, "recovery-handover-memo", [old.clone()]);
    let harness = open(
        Arc::new(crate::storage::memory::MemoryStorage::new()),
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let id = create(&harness, &old, Value::Null).await;
    harness.resume();
    tokio::time::timeout(WAIT, reached.notified())
        .await
        .expect("first definition memoized value");
    register(
        &registry,
        "recovery-handover-memo",
        [memo_task("new", log.clone(), gate.clone(), reached.clone())],
    );
    gate.notify_one();
    let receipt = harness
        .wait_for_task(id, &context())
        .await
        .expect("memo handover settles");
    assert_eq!(
        receipt
            .state
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.result.as_ref()),
        Some(&json!("old"))
    );
    assert_eq!(*log.lock(), ["new:b old"]);
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn handover_to_newer_version_migrates_checkpoint_at_the_boundary() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let gate = Arc::new(Notify::new());
    let (harness, registry, _old, id) = start_handover(
        log.clone(),
        HandoverGates {
            a: Some(gate.clone()),
            ..Default::default()
        },
        Arc::new(crate::storage::memory::MemoryStorage::new()),
        None,
    )
    .await;
    let migration: Migration = Arc::new(|input, _checkpoint, _from| {
        Ok(TaskMigration {
            input: input.clone(),
            checkpoint: json!({"phase":"c"}),
        })
    });
    register(
        &registry,
        "recovery-handover",
        [handover_task(
            "v2",
            2,
            log.clone(),
            HandoverGates::default(),
            Some(migration),
        )],
    );
    gate.notify_one();
    let receipt = harness
        .wait_for_task(id, &context())
        .await
        .expect("v2 task settles");
    assert_eq!(receipt.version, 2);
    assert_eq!(*log.lock(), ["old:a start", "old:a end", "v2:c"]);
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn failed_handover_migration_keeps_the_task_pending_at_its_checkpoint() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let gate = Arc::new(Notify::new());
    let reports = Arc::new(Mutex::new(Vec::<String>::new()));
    let report_slot = reports.clone();
    let on_report: ReportHook = Arc::new(move |message| report_slot.lock().push(message));
    let (harness, registry, _old, id) = start_handover(
        log.clone(),
        HandoverGates {
            a: Some(gate.clone()),
            ..Default::default()
        },
        Arc::new(crate::storage::memory::MemoryStorage::new()),
        Some(on_report),
    )
    .await;
    let broken: Migration = Arc::new(|_, _, _| Err("broken migration".into()));
    register(
        &registry,
        "recovery-handover",
        [handover_task(
            "broken",
            2,
            log.clone(),
            HandoverGates::default(),
            Some(broken),
        )],
    );
    gate.notify_one();
    wait_record(&harness, id, |record| {
        record.state.status == TaskStateStatus::Pending && record.version == 1
    })
    .await;
    tokio::time::timeout(WAIT, async {
        while reports.lock().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("failed handover migration is reported");
    ticks().await;
    assert_eq!(reports.lock().len(), 1);
    assert!(reports.lock()[0].contains("broken migration"));
    let record = harness.get_task(id, &context()).await.unwrap().unwrap();
    assert_eq!(record.state.checkpoint, Some(json!({"phase":"b"})));
    assert_eq!(*log.lock(), ["old:a start", "old:a end"]);
    harness
        .close(&context())
        .await
        .expect("close blocked handover");
}

#[tokio::test]
async fn old_definition_continues_when_replacement_is_missing_or_incompatible() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let gate_a = Arc::new(Notify::new());
    let gate_b = Arc::new(Notify::new());
    let reports = Arc::new(Mutex::new(Vec::<String>::new()));
    let report_slot = reports.clone();
    let on_report: ReportHook = Arc::new(move |message| report_slot.lock().push(message));
    let (harness, registry, _old, id) = start_handover(
        log.clone(),
        HandoverGates {
            a: Some(gate_a.clone()),
            b: Some(gate_b.clone()),
            ..Default::default()
        },
        Arc::new(crate::storage::memory::MemoryStorage::new()),
        Some(on_report),
    )
    .await;
    registry.uninstall(&Extension::new("recovery-handover"));
    gate_a.notify_one();
    wait_log(&log, 3).await;
    register(
        &registry,
        "recovery-handover",
        [handover_task(
            "incompatible",
            2,
            log.clone(),
            HandoverGates::default(),
            None,
        )],
    );
    gate_b.notify_one();
    harness
        .wait_for_task(id, &context())
        .await
        .expect("old definition completes task");
    tokio::time::timeout(WAIT, async {
        while reports.lock().len() < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("missing and incompatible replacements are reported");
    let causes = reports
        .lock()
        .iter()
        .filter_map(|message| {
            if message.contains("missing_task") {
                Some("missing_task")
            } else if message.contains("incompatible_task") {
                Some("incompatible_task")
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(causes, ["missing_task", "incompatible_task"]);
    assert_eq!(
        *log.lock(),
        [
            "old:a start",
            "old:a end",
            "old:b start",
            "old:b end",
            "old:c"
        ]
    );
    harness.close(&context()).await.expect("close harness");
}
#[tokio::test]
async fn old_runtime_commit_queued_behind_handover_is_rejected() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let gate = Arc::new(Notify::new());
    let storage = Arc::new(ControlledStorage::new());
    let held_slot = Arc::new(Mutex::new(None));
    let runtime_slot = Arc::new(Mutex::new(None::<Arc<dyn TaskRuntime>>));
    let blocker_done = Arc::new(Notify::new());
    let registry = Arc::new(create_registry());
    let blocker_storage = storage.clone();
    let blocker_root_slot = Arc::new(Mutex::new(None::<Conversation>));
    let old = {
        let gate = gate.clone();
        let held_slot = held_slot.clone();
        let runtime_slot = runtime_slot.clone();
        let blocker_done = blocker_done.clone();
        let blocker_root_slot = blocker_root_slot.clone();
        let blocker_storage = blocker_storage.clone();
        let phase_a = handler(move |_record, runtime, ctx| {
            let gate = gate.clone();
            let held_slot = held_slot.clone();
            let runtime_slot = runtime_slot.clone();
            let blocker_done = blocker_done.clone();
            let blocker_root_slot = blocker_root_slot.clone();
            let blocker_storage = blocker_storage.clone();
            Box::pin(async move {
                *runtime_slot.lock() = Some(runtime.clone());
                gate.notified().await;
                runtime
                    .commit(
                        commit_change(|_, _| {
                            Ok(Some(NextTaskState::Running {
                                checkpoint: json!({"phase":"b"}),
                            }))
                        }),
                        &ctx,
                    )
                    .await?;
                *held_slot.lock() = Some(blocker_storage.hold_commits());
                let root = blocker_root_slot
                    .lock()
                    .clone()
                    .expect("blocker root installed");
                tokio::spawn(async move {
                    root.commit(
                        move |tx| {
                            tx.append_entry(
                                root.id,
                                EntryDraft {
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
                    .expect("commit queue blocker");
                    blocker_done.notify_one();
                });
                Ok(())
            })
        });
        make_task(
            "test.recovery-handover",
            1,
            json!({"phase":"a"}),
            [
                ("a".into(), phase_a),
                ("b".into(), handler(|_, _, _| Box::pin(async { Ok(()) }))),
                ("c".into(), handler(|_, _, _| Box::pin(async { Ok(()) }))),
            ],
            aborted("old"),
            None,
        )
    };
    register(&registry, "recovery-handover", [old.clone()]);
    let harness = open(
        storage.clone(),
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let conversation = root(&harness).await;
    *blocker_root_slot.lock() = Some(conversation.clone());
    let id = create(&harness, &old, Value::Null).await;
    harness.resume();
    tokio::time::timeout(WAIT, async {
        while runtime_slot.lock().is_none() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("old invocation started");
    wait_record(&harness, id, |record| {
        record.state.status == TaskStateStatus::Running
    })
    .await;
    register(
        &registry,
        "recovery-handover",
        [handover_task(
            "new",
            1,
            log.clone(),
            HandoverGates::default(),
            None,
        )],
    );
    gate.notify_one();
    tokio::time::timeout(WAIT, async {
        while held_slot.lock().is_none() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("blocker held next commit");
    let mut held = held_slot.lock().take().expect("held storage gate");
    tokio::time::timeout(WAIT, held.entered())
        .await
        .expect("blocker commit entered storage");
    ticks().await;
    let old_runtime = runtime_slot.lock().clone().expect("old runtime");
    let late = tokio::spawn(async move {
        old_runtime
            .commit(
                commit_change(|_, _| Ok(Some(NextTaskState::completed(json!("late"))))),
                &context(),
            )
            .await
    });
    ticks().await;
    held.release();
    assert!(late
        .await
        .expect("late runtime commit")
        .expect_err("ended invocation rejects late commit")
        .contains("invocation has ended"));
    tokio::time::timeout(WAIT, blocker_done.notified())
        .await
        .expect("unrelated blocker commit completed");
    harness
        .wait_for_task(id, &context())
        .await
        .expect("replacement settles");
    assert_eq!(*log.lock(), ["new:b start", "new:b end", "new:c"]);
    harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn abort_mark_queued_behind_handover_is_preserved_for_new_definition() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let gate = Arc::new(Notify::new());
    let storage = Arc::new(ControlledStorage::new());
    let held_slot = Arc::new(Mutex::new(None));
    let storage_for_end = storage.clone();
    let held_for_end = held_slot.clone();
    let on_a_end: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
        let mut slot = held_for_end.lock();
        if slot.is_none() {
            *slot = Some(storage_for_end.hold_commits());
        }
    });
    let (harness, registry, _old, id) = start_handover(
        log.clone(),
        HandoverGates {
            a: Some(gate.clone()),
            on_a_end: Some(on_a_end),
            ..Default::default()
        },
        storage.clone(),
        None,
    )
    .await;
    register(
        &registry,
        "recovery-handover",
        [handover_task(
            "new",
            1,
            log.clone(),
            HandoverGates::default(),
            None,
        )],
    );
    gate.notify_one();
    tokio::time::timeout(WAIT, async {
        while held_slot.lock().is_none() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("handover commit gate installed");
    let mut held = held_slot.lock().take().expect("handover gate");
    tokio::time::timeout(WAIT, held.entered())
        .await
        .expect("handover commit entered storage");
    let aborting_harness = harness.clone();
    let aborting = tokio::spawn(async move { aborting_harness.abort_task(id, &context()).await });
    ticks().await;
    held.release();
    assert_eq!(
        aborting.await.expect("abort call").expect("abort mark"),
        "marked"
    );
    let receipt = harness
        .wait_for_task(id, &context())
        .await
        .expect("new definition handles abort");
    assert_eq!(
        receipt.state.outcome.as_ref().map(|outcome| outcome.status),
        Some(TaskOutcomeStatus::Aborted)
    );
    assert_eq!(
        receipt
            .state
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.reason.as_deref()),
        Some("new")
    );
    let states = storage
        .commits()
        .into_iter()
        .flat_map(|batch| batch.into_iter())
        .filter_map(|write| match write {
            crate::types::StorageWrite::Task { value } if value.id == id => {
                let status = match value.state.status {
                    TaskStateStatus::Pending => "pending",
                    TaskStateStatus::Running => "running",
                    TaskStateStatus::Waiting => "waiting",
                    TaskStateStatus::Completing => "completing",
                    TaskStateStatus::Terminal => "terminal",
                };
                Some(format!(
                    "{status}{}",
                    if value.abort_requested { "+mark" } else { "" }
                ))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        states,
        [
            "pending",
            "running",
            "running",
            "pending",
            "pending+mark",
            "running+mark",
            "terminal+mark"
        ]
    );
    assert_eq!(*log.lock(), ["old:a start", "old:a end", "new:abort"]);
    harness.close(&context()).await.expect("close harness");
}

struct CountingReader {
    inner: Arc<RegistryImpl>,
    subscriptions: Arc<std::sync::atomic::AtomicUsize>,
}

impl RegistryReader for CountingReader {
    fn snapshot(&self) -> Arc<dyn RegistrySnapshot> {
        self.inner.snapshot()
    }

    fn subscribe(&self, listener: crate::harness::types::RegistryListener) -> Unsubscribe {
        self.subscriptions
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let unsubscribe = self.inner.subscribe(listener);
        let subscriptions = self.subscriptions.clone();
        Box::new(move || {
            unsubscribe();
            subscriptions.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        })
    }
}

#[tokio::test]
async fn failed_open_releases_registry_subscription_and_closes_its_session() {
    let storage = Arc::new(ControlledStorage::new());
    let registry = Arc::new(create_registry());
    let log = Arc::new(Mutex::new(Vec::new()));
    let run = {
        let log = log.clone();
        handler(move |_record, runtime, _ctx| {
            let log = log.clone();
            Box::pin(async move {
                log.lock().push("run".into());
                runtime.signal().aborted_future().await;
                Ok(())
            })
        })
    };
    let task = make_task(
        "test.recovery-open-failure",
        1,
        json!({"phase":"run"}),
        [("run".into(), run)],
        aborted("stop"),
        None,
    );
    register(&registry, "recovery-open-failure", [task.clone()]);
    let first = open(
        storage.clone(),
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            ..Default::default()
        },
    )
    .await;
    let _id = create(&first, &task, Value::Null).await;
    first.resume();
    wait_log(&log, 1).await;

    let subscriptions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let reader: Arc<dyn RegistryReader> = Arc::new(CountingReader {
        inner: registry,
        subscriptions: subscriptions.clone(),
    });
    storage.fail_next_commit("disk full");
    let result = open_harness(
        storage.clone(),
        crate::harness::types::HarnessOptions {
            registry: reader,
            ..Default::default()
        },
        &context(),
    )
    .await;
    let error = match result {
        Ok(harness) => {
            harness
                .close(&context())
                .await
                .expect("close unexpected harness");
            panic!("open should fail during running-task reconciliation")
        }
        Err(error) => error,
    };
    assert!(error.contains("disk full"));
    assert_eq!(subscriptions.load(std::sync::atomic::Ordering::SeqCst), 0);
    // Upstream: the failed harness's close closed the shared storage, so its
    // reads reject afterwards (`storage.task` throws); the port's controlled
    // storage blinds them instead.
    assert!(storage.is_closed());
    first
        .close(&context())
        .await
        .expect("close original harness");
}

/// `startedAt` survives close and reopen; `endedAt` is stamped when the task
/// settles (upstream #10549, adapted to the Rust clock control).
#[tokio::test]
async fn keeps_started_at_across_close_and_reopen_and_stamps_ended_at_when_it_settles() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("session.sqlite");
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let started = Arc::new(Notify::new());
    let calls_for_handler = calls.clone();
    let started_for_handler = started.clone();
    let transfer = make_task(
        "test.timed-transfer",
        1,
        json!({"phase": "run"}),
        [
            (
                "run".into(),
                handler(move |_record, runtime, ctx| {
                    let calls = calls_for_handler.clone();
                    let started = started_for_handler.clone();
                    Box::pin(async move {
                        calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        started.notify_one();
                        runtime
                            .commit(
                                commit_change(|_, _| {
                                    Ok(Some(NextTaskState::Running {
                                        checkpoint: json!({"phase": "finish"}),
                                    }))
                                }),
                                &ctx,
                            )
                            .await?;
                        Ok(())
                    })
                }),
            ),
            ("finish".into(), completed(json!("transferred"))),
        ],
        aborted("transfer aborted"),
        None,
    );
    let clock = Arc::new(std::sync::atomic::AtomicI64::new(2_000));
    let now = {
        let clock = clock.clone();
        Arc::new(move || clock.load(std::sync::atomic::Ordering::SeqCst))
            as crate::harness::types::NowFn
    };
    let registry = Arc::new(create_registry());
    register(&registry, "timed-transfer", [transfer.clone()]);

    let first = open(
        sqlite(&path).await,
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            now: Some(now.clone()),
            ..Default::default()
        },
    )
    .await;
    let id = create(&first, &transfer, json!({"amount": 7})).await;
    first.resume();
    tokio::time::timeout(WAIT, started.notified())
        .await
        .expect("first run started");
    // The task is running at 3_000 when the harness closes.
    clock.store(3_000, std::sync::atomic::Ordering::SeqCst);
    first.close(&context()).await.expect("close first harness");

    // Reopened at 9_000: `startedAt` is the first run's time and the task reruns.
    clock.store(9_000, std::sync::atomic::Ordering::SeqCst);
    let second = open(
        sqlite(&path).await,
        crate::harness::types::HarnessOptions {
            registry: registry.clone(),
            now: Some(now.clone()),
            ..Default::default()
        },
    )
    .await;
    let reopened = second
        .get_task(id, &context())
        .await
        .expect("read task")
        .expect("task exists");
    assert_eq!(reopened.state.status, TaskStateStatus::Pending);
    assert_eq!(reopened.started_at, Some(2_000));
    assert_eq!(reopened.ended_at, None);
    second.resume();
    let receipt = second
        .wait_for_task(id, &context())
        .await
        .expect("task settles");
    assert_eq!(receipt.started_at, Some(2_000));
    assert_eq!(receipt.ended_at, Some(9_000));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    second
        .close(&context())
        .await
        .expect("close second harness");
}
