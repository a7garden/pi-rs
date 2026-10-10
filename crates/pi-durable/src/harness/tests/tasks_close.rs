use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use parking_lot::Mutex;
use pi_chord::context::{AbortSignal, Context};
use serde_json::{json, Value};
use tokio::task::JoinHandle;

use super::support::{context, root};
use crate::documents::{DocDefinition, DocToken, ScopeKind};
use crate::harness::harness::{open_harness, Conversation, Harness};
use crate::harness::registry::{create_registry, RegistryImpl};
use crate::harness::types::{
    AnyTask, Extension, HarnessOptions, Registry, RegistryListener, RegistryReader,
    RegistrySnapshot, Unsubscribe,
};
use crate::session::support::{flush, ControlledStorage};
use crate::tasks::{
    commit_change, define_task, BoxFuture, NextTaskState, PhaseHandler, Task, TaskDefinition,
    TaskRuntime,
};
use crate::types::{
    EntryDraft, Storage, StorageWrite, TaskId, TaskOptions, TaskOwnership, TaskRecord,
    TaskStateStatus, WatchEnd,
};

fn handler(
    f: impl Fn(TaskRecord, Arc<dyn TaskRuntime>, Context) -> BoxFuture<Result<(), String>>
        + Send
        + Sync
        + 'static,
) -> PhaseHandler {
    Arc::new(f)
}

fn task(
    name: &str,
    initial: Value,
    phases: impl IntoIterator<Item = (String, PhaseHandler)>,
    abort: PhaseHandler,
) -> Task {
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

fn add_task(registry: &Arc<RegistryImpl>, task: &Task) {
    registry.install(Extension::new("task-close-tests").with_tasks(vec![AnyTask {
        definition: task.definition.clone(),
    }]));
}

async fn open(storage: Arc<ControlledStorage>, registry: Arc<dyn RegistryReader>) -> Arc<Harness> {
    let storage: Arc<dyn Storage> = storage;
    open_harness(
        storage,
        HarnessOptions {
            registry,
            ..HarnessOptions::default()
        },
        &context(),
    )
    .await
    .expect("open harness")
}

async fn open_task(storage: Arc<ControlledStorage>, task: &Task) -> (Arc<Harness>, Conversation) {
    let registry = Arc::new(create_registry());
    add_task(&registry, task);
    let harness = open(storage, registry).await;
    let conversation = root(&harness).await;
    (harness, conversation)
}

async fn create(root: &Conversation, task: Task) -> TaskId {
    root.commit(
        move |tx| {
            tx.create_task(
                &AnyTask {
                    definition: task.definition.clone(),
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
    .expect("create task")
}

async fn start_close(harness: Arc<Harness>) -> JoinHandle<Result<(), String>> {
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let closing = tokio::spawn(async move {
        let _ = started_tx.send(());
        let context = context();
        harness.close(&context).await
    });
    started_rx.await.expect("close task started");
    // The close future runs synchronously through admission sealing and the
    // scheduler's close listener before it suspends while joining invocations.
    tokio::task::yield_now().await;
    closing
}

async fn mark_durably(
    harness: Arc<Harness>,
    id: TaskId,
) -> JoinHandle<Result<&'static str, String>> {
    let aborting_harness = harness.clone();
    let aborting = tokio::spawn(async move {
        let context = context();
        aborting_harness.abort_task(id, &context).await
    });
    loop {
        let record = harness
            .get_task(id, &context())
            .await
            .expect("read task")
            .expect("task exists");
        if record.abort_requested {
            return aborting;
        }
        tokio::task::yield_now().await;
    }
}

fn task_records(storage: &ControlledStorage, id: TaskId) -> Vec<TaskRecord> {
    storage
        .commits()
        .into_iter()
        .flatten()
        .filter_map(|write| match write {
            StorageWrite::Task { value } if value.id == id => Some(value),
            _ => None,
        })
        .collect()
}

fn append_blocker(root: Conversation) -> JoinHandle<Result<(), String>> {
    tokio::spawn(async move {
        let id = root.id;
        root.commit(
            move |tx| {
                tx.append_entry(
                    id,
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
    })
}

fn blocker_is_admitted(storage: &ControlledStorage) -> bool {
    storage
        .commits()
        .into_iter()
        .flatten()
        .any(|write| matches!(write, StorageWrite::Entry { value } if value.kind == "blocker"))
}

async fn wait_for_blocker(storage: &ControlledStorage) {
    while !blocker_is_admitted(storage) {
        tokio::task::yield_now().await;
    }
}

#[tokio::test]
async fn close_preserves_running_checkpoint_without_outcome_or_next_phase() {
    let reached = Arc::new(tokio::sync::Notify::new());
    let proceed = Arc::new(tokio::sync::Notify::new());
    let phases = Arc::new(Mutex::new(Vec::<String>::new()));
    let abort_ran = Arc::new(AtomicBool::new(false));
    let first_reached = reached.clone();
    let first_proceed = proceed.clone();
    let first_phases = phases.clone();
    let second_phases = phases.clone();
    let abort_ran_in = abort_ran.clone();
    let two = task(
        "test.close-two-phases",
        json!({"phase":"one"}),
        [
            (
                "one".into(),
                handler(move |_, runtime, ctx| {
                    let reached = first_reached.clone();
                    let proceed = first_proceed.clone();
                    let phases = first_phases.clone();
                    Box::pin(async move {
                        phases.lock().push("one".into());
                        runtime
                            .commit(
                                commit_change(|_, _| {
                                    Ok(Some(NextTaskState::Running {
                                        checkpoint: json!({"phase":"two"}),
                                    }))
                                }),
                                &ctx,
                            )
                            .await?;
                        reached.notify_one();
                        proceed.notified().await;
                        Ok(())
                    })
                }),
            ),
            (
                "two".into(),
                handler(move |_, _, _| {
                    let phases = second_phases.clone();
                    Box::pin(async move {
                        phases.lock().push("two".into());
                        Ok(())
                    })
                }),
            ),
        ],
        handler(move |_, runtime, ctx| {
            abort_ran_in.store(true, Ordering::SeqCst);
            Box::pin(async move {
                runtime
                    .commit(
                        commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("close")))),
                        &ctx,
                    )
                    .await?;
                Ok(())
            })
        }),
    );
    let storage = Arc::new(ControlledStorage::new());
    let (harness, conversation) = open_task(storage.clone(), &two).await;
    let id = create(&conversation, two).await;
    harness.resume();
    reached.notified().await;
    let aborting = mark_durably(harness.clone(), id).await;
    flush().await;
    let before_close = harness
        .get_task(id, &context())
        .await
        .expect("read task")
        .expect("task exists");
    let commits_before_close = storage.commits().len();
    let closing = start_close(harness.clone()).await;
    proceed.notify_one();
    closing
        .await
        .expect("close task joins")
        .expect("close succeeds");
    assert_eq!(
        aborting
            .await
            .expect("abort task joins")
            .expect("abort mark"),
        "marked"
    );
    assert_eq!(storage.commits().len(), commits_before_close);
    assert_eq!(*phases.lock(), ["one"]);
    assert!(!abort_ran.load(Ordering::SeqCst));
    assert!(before_close.abort_requested);
    assert_eq!(before_close.state.status, TaskStateStatus::Running);
    assert_eq!(before_close.state.checkpoint, Some(json!({"phase":"two"})));
    assert!(before_close.state.outcome.is_none());
    let error = conversation
        .commit(|_| Ok(()), &context())
        .await
        .expect_err("close seals conversation commits");
    assert!(error.to_lowercase().contains("closed"), "{error}");
}

#[tokio::test]
async fn close_seals_admission_before_signalling_and_stops_watches_before_joining() {
    let notes = DocToken::new(DocDefinition::singleton(
        "test.close-notes",
        1,
        ScopeKind::Conversation,
        || serde_json::Map::from_iter([("text".to_owned(), Value::String(String::new()))]),
    ));
    let reached = Arc::new(tokio::sync::Notify::new());
    let root_slot = Arc::new(Mutex::new(None::<Conversation>));
    let listener_result = Arc::new(Mutex::new(
        None::<tokio::sync::oneshot::Sender<Result<(), String>>>,
    ));
    let watch_end = Arc::new(Mutex::new(None::<WatchEnd>));
    let notes_for_task = notes.clone();
    let reached_for_task = reached.clone();
    let root_for_task = root_slot.clone();
    let listener_for_task = listener_result.clone();
    let end_for_task = watch_end.clone();
    let stubborn = task(
        "test.close-stubborn-watch",
        json!({"phase":"run"}),
        [(
            "run".into(),
            handler(move |_, runtime, ctx| {
                let notes = notes_for_task.clone();
                let reached = reached_for_task.clone();
                let root_slot = root_for_task.clone();
                let listener_result = listener_for_task.clone();
                let watch_end = end_for_task.clone();
                Box::pin(async move {
                    let watch = runtime
                        .watch_doc(&notes, runtime.conversation_id(), &ctx)
                        .await?
                        .expect("notes document exists");
                    let signal = runtime.signal();
                    let _detach = signal.add_listener(move || {
                        let Some(sender) = listener_result.lock().take() else {
                            return;
                        };
                        let conversation = root_slot
                            .lock()
                            .as_ref()
                            .expect("root is installed")
                            .clone();
                        tokio::spawn(async move {
                            let result = conversation.commit(|_| Ok(()), &context()).await;
                            let _ = sender.send(result);
                        });
                    });
                    reached.notify_one();
                    *watch_end.lock() = Some(watch.closed().await);
                    Ok(())
                })
            }),
        )],
        abort_completed("test"),
    );
    let storage = Arc::new(ControlledStorage::new());
    let (harness, conversation) = open_task(storage.clone(), &stubborn).await;
    *root_slot.lock() = Some(conversation.clone());
    let root_id = conversation.id;
    let notes_for_commit = notes.clone();
    conversation
        .commit(
            move |tx| {
                tx.doc(&notes_for_commit, Some(root_id.0), None, None)
                    .map_err(|error| error.to_string())?
                    .set(&[pi_chord::delta::Seg::key("text")], json!("x"))
                    .map_err(|error| error.to_string())
            },
            &context(),
        )
        .await
        .expect("create watched notes");
    let (listener_tx, listener_rx) = tokio::sync::oneshot::channel();
    *listener_result.lock() = Some(listener_tx);
    let id = create(&conversation, stubborn).await;
    harness.resume();
    reached.notified().await;
    let closing = start_close(harness.clone()).await;
    closing
        .await
        .expect("close task joins")
        .expect("close succeeds");
    let listener_error = listener_rx
        .await
        .expect("abort listener attempted a commit")
        .expect_err("closed admission rejects listener commit");
    assert!(
        listener_error.to_lowercase().contains("closed"),
        "{listener_error}"
    );
    assert_eq!(*watch_end.lock(), Some(WatchEnd::SessionClosed));
    let task = task_records(&storage, id)
        .pop()
        .expect("task was durably reserved");
    assert_eq!(task.state.status, TaskStateStatus::Running);
}

#[tokio::test]
async fn close_suppresses_fault_when_the_failed_step_is_queued() {
    let storage = Arc::new(ControlledStorage::new());
    let gate_slot = Arc::new(Mutex::new(None));
    let gate_created = Arc::new(tokio::sync::Notify::new());
    let root_slot = Arc::new(Mutex::new(None::<Conversation>));
    let blocker_slot = Arc::new(Mutex::new(None::<JoinHandle<Result<(), String>>>));
    let task_storage = storage.clone();
    let gate_for_task = gate_slot.clone();
    let created_for_task = gate_created.clone();
    let root_for_task = root_slot.clone();
    let blocker_for_task = blocker_slot.clone();
    let throws = task(
        "test.close-fault-queued",
        json!({"phase":"run"}),
        [(
            "run".into(),
            handler(move |_, _, _| {
                let storage = task_storage.clone();
                let gate_slot = gate_for_task.clone();
                let gate_created = created_for_task.clone();
                let root_slot = root_for_task.clone();
                let blocker_slot = blocker_for_task.clone();
                Box::pin(async move {
                    *gate_slot.lock() = Some(storage.hold_commits());
                    gate_created.notify_one();
                    let root = root_slot
                        .lock()
                        .as_ref()
                        .expect("root is installed")
                        .clone();
                    *blocker_slot.lock() = Some(append_blocker(root));
                    wait_for_blocker(&storage).await;
                    Err("would fault".into())
                })
            }),
        )],
        abort_completed("test"),
    );
    let (harness, conversation) = open_task(storage.clone(), &throws).await;
    *root_slot.lock() = Some(conversation.clone());
    let id = create(&conversation, throws).await;
    harness.resume();
    gate_created.notified().await;
    let mut gate = loop {
        if let Some(gate) = gate_slot.lock().take() {
            break gate;
        }
        tokio::task::yield_now().await;
    };
    gate.entered().await;
    flush().await;
    let closing = start_close(harness).await;
    gate.release();
    closing
        .await
        .expect("close task joins")
        .expect("close succeeds");
    let blocker = blocker_slot.lock().take();
    if let Some(blocker) = blocker {
        blocker
            .await
            .expect("blocker commit task")
            .expect("blocker commit succeeds");
    }
    let statuses = task_records(&storage, id)
        .into_iter()
        .map(|record| record.state.status)
        .collect::<Vec<_>>();
    assert_eq!(
        statuses,
        [TaskStateStatus::Pending, TaskStateStatus::Running]
    );
}

#[tokio::test]
async fn close_joins_abort_reservation_without_starting_abort_handler() {
    let abort_ran = Arc::new(AtomicBool::new(false));
    let abort_ran_in = abort_ran.clone();
    let marked = task(
        "test.close-abort-reservation",
        json!({"phase":"run"}),
        [("run".into(), handler(|_, _, _| Box::pin(async { Ok(()) })))],
        handler(move |_, _, _| {
            abort_ran_in.store(true, Ordering::SeqCst);
            Box::pin(async { Ok(()) })
        }),
    );
    let storage = Arc::new(ControlledStorage::new());
    let (harness, conversation) = open_task(storage.clone(), &marked).await;
    let id = create(&conversation, marked).await;
    assert_eq!(
        harness
            .abort_task(id, &context())
            .await
            .expect("mark abort"),
        "marked"
    );
    flush().await;
    let mut gate = storage.hold_commits();
    harness.resume();
    gate.entered().await;
    let closing = start_close(harness.clone()).await;
    gate.release();
    closing
        .await
        .expect("close task joins")
        .expect("close succeeds");
    assert!(!abort_ran.load(Ordering::SeqCst));
    let last = storage.commits().pop().expect("reservation commit");
    let task_writes = last
        .into_iter()
        .filter_map(|write| match write {
            StorageWrite::Task { value } if value.id == id => Some(value),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(task_writes.len(), 1);
    assert!(task_writes[0].abort_requested);
    assert_eq!(task_writes[0].state.status, TaskStateStatus::Running);
}

#[tokio::test]
async fn close_rejects_runtime_commit_queued_before_admission_was_sealed() {
    let storage = Arc::new(ControlledStorage::new());
    let gate_slot = Arc::new(Mutex::new(None));
    let gate_created = Arc::new(tokio::sync::Notify::new());
    let proceed = Arc::new(tokio::sync::Notify::new());
    let root_slot = Arc::new(Mutex::new(None::<Conversation>));
    let blocker_slot = Arc::new(Mutex::new(None::<JoinHandle<Result<(), String>>>));
    let queued_slot = Arc::new(Mutex::new(None::<JoinHandle<Result<(), String>>>));
    let task_storage = storage.clone();
    let gate_for_task = gate_slot.clone();
    let created_for_task = gate_created.clone();
    let proceed_for_task = proceed.clone();
    let root_for_task = root_slot.clone();
    let blocker_for_task = blocker_slot.clone();
    let queued_for_task = queued_slot.clone();
    let queued = task(
        "test.close-queued-runtime-commit",
        json!({"phase":"run"}),
        [(
            "run".into(),
            handler(move |_, runtime, _ctx| {
                let storage = task_storage.clone();
                let gate_slot = gate_for_task.clone();
                let gate_created = created_for_task.clone();
                let proceed = proceed_for_task.clone();
                let root_slot = root_for_task.clone();
                let blocker_slot = blocker_for_task.clone();
                let queued_slot = queued_for_task.clone();
                Box::pin(async move {
                    *gate_slot.lock() = Some(storage.hold_commits());
                    gate_created.notify_one();
                    let root = root_slot
                        .lock()
                        .as_ref()
                        .expect("root is installed")
                        .clone();
                    *blocker_slot.lock() = Some(append_blocker(root));
                    wait_for_blocker(&storage).await;
                    let runtime_for_commit = runtime.clone();
                    // Upstream queues the runtime commit with the base test
                    // context, so the rejection comes from the scheduler's
                    // closing check, not the invocation signal.
                    let queued_context = context();
                    *queued_slot.lock() = Some(tokio::spawn(async move {
                        runtime_for_commit
                            .commit(
                                commit_change(|_, _| {
                                    Ok(Some(NextTaskState::completed(Value::Null)))
                                }),
                                &queued_context,
                            )
                            .await
                            .map(|_| ())
                    }));
                    proceed.notified().await;
                    Ok(())
                })
            }),
        )],
        abort_completed("test"),
    );
    let (harness, conversation) = open_task(storage.clone(), &queued).await;
    *root_slot.lock() = Some(conversation.clone());
    let id = create(&conversation, queued).await;
    harness.resume();
    gate_created.notified().await;
    let mut gate = loop {
        if let Some(gate) = gate_slot.lock().take() {
            break gate;
        }
        tokio::task::yield_now().await;
    };
    gate.entered().await;
    flush().await;
    let closing = start_close(harness).await;
    gate.release();
    let queued_commit = queued_slot
        .lock()
        .take()
        .expect("runtime commit was queued");
    let queued_result = queued_commit.await.expect("queued runtime commit task");
    let error = queued_result.expect_err("close rejects the queued runtime commit");
    assert!(error.to_lowercase().contains("closed"), "{error}");
    proceed.notify_one();
    closing
        .await
        .expect("close task joins")
        .expect("close succeeds");
    let blocker = blocker_slot.lock().take();
    if let Some(blocker) = blocker {
        blocker
            .await
            .expect("blocker commit task")
            .expect("blocker commit succeeds");
    }
    assert_eq!(task_records(&storage, id).len(), 2);
}

struct CloseOnSnapshotRegistry {
    inner: Arc<RegistryImpl>,
    armed: AtomicBool,
    harness: Mutex<Option<Arc<Harness>>>,
    signal: Mutex<Option<AbortSignal>>,
    close_started: tokio::sync::Notify,
    closing: Mutex<Option<std::thread::JoinHandle<Result<(), String>>>>,
}

impl CloseOnSnapshotRegistry {
    fn new(inner: Arc<RegistryImpl>) -> Self {
        Self {
            inner,
            armed: AtomicBool::new(false),
            harness: Mutex::new(None),
            signal: Mutex::new(None),
            close_started: tokio::sync::Notify::new(),
            closing: Mutex::new(None),
        }
    }
}

impl RegistryReader for CloseOnSnapshotRegistry {
    fn snapshot(&self) -> Arc<dyn RegistrySnapshot> {
        if self.armed.swap(false, Ordering::SeqCst) {
            let harness = self
                .harness
                .lock()
                .as_ref()
                .expect("harness is installed")
                .clone();
            // Upstream's `void (async () => close())` executes the close's
            // synchronous prefix (through admission sealing and the seal
            // signal) inline; tokio::spawn would queue the whole future on
            // this worker's unstealable LIFO slot and starve while the spin
            // below waits for that signal. Run close on its own thread.
            let runtime_handle = tokio::runtime::Handle::current();
            let closing = std::thread::spawn(move || {
                let context = context();
                runtime_handle.block_on(harness.close(&context))
            });
            *self.closing.lock() = Some(closing);
            self.close_started.notify_one();
            let signal = self
                .signal
                .lock()
                .clone()
                .expect("phase signal is installed");
            // This snapshot is taken synchronously inside the scheduler's line
            // callback. Let close seal and signal the invocation before allowing
            // the callback to return and the run loop to dispatch another phase.
            while !signal.aborted() {
                std::thread::yield_now();
            }
        }
        self.inner.snapshot()
    }

    fn subscribe(&self, listener: RegistryListener) -> Unsubscribe {
        self.inner.subscribe(listener)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn close_during_progress_snapshot_prevents_the_next_phase() {
    let inner = Arc::new(create_registry());
    let reader = Arc::new(CloseOnSnapshotRegistry::new(inner.clone()));
    let phases = Arc::new(Mutex::new(Vec::<String>::new()));
    let reader_for_task = reader.clone();
    let first_phases = phases.clone();
    let second_phases = phases.clone();
    let two = task(
        "test.close-in-progress-snapshot",
        json!({"phase":"one"}),
        [
            (
                "one".into(),
                handler(move |_, runtime, ctx| {
                    let reader = reader_for_task.clone();
                    let phases = first_phases.clone();
                    Box::pin(async move {
                        phases.lock().push("one".into());
                        *reader.signal.lock() = Some(runtime.signal());
                        runtime
                            .commit(
                                commit_change(|_, _| {
                                    Ok(Some(NextTaskState::Running {
                                        checkpoint: json!({"phase":"two"}),
                                    }))
                                }),
                                &ctx,
                            )
                            .await?;
                        reader.armed.store(true, Ordering::SeqCst);
                        Ok(())
                    })
                }),
            ),
            (
                "two".into(),
                handler(move |_, _, _| {
                    let phases = second_phases.clone();
                    Box::pin(async move {
                        phases.lock().push("two".into());
                        Ok(())
                    })
                }),
            ),
        ],
        abort_completed("test"),
    );
    add_task(&inner, &two);
    let storage = Arc::new(ControlledStorage::new());
    let registry: Arc<dyn RegistryReader> = reader.clone();
    let harness = open(storage, registry).await;
    *reader.harness.lock() = Some(harness.clone());
    let conversation = root(&harness).await;
    let _id = create(&conversation, two).await;
    harness.resume();
    reader.close_started.notified().await;
    let closing = loop {
        if let Some(closing) = reader.closing.lock().take() {
            break closing;
        }
        tokio::task::yield_now().await;
    };
    closing
        .join()
        .expect("close task joins")
        .expect("close succeeds");
    assert_eq!(*phases.lock(), ["one"]);
}

#[tokio::test]
async fn close_joins_a_reservation_that_settles_without_starting_its_handler() {
    let ran = Arc::new(AtomicBool::new(false));
    let ran_in = ran.clone();
    let never = task(
        "test.close-reservation",
        json!({"phase":"run"}),
        [(
            "run".into(),
            handler(move |_, _, _| {
                let ran = ran_in.clone();
                Box::pin(async move {
                    ran.store(true, Ordering::SeqCst);
                    Ok(())
                })
            }),
        )],
        abort_completed("test"),
    );
    let storage = Arc::new(ControlledStorage::new());
    let (harness, conversation) = open_task(storage.clone(), &never).await;
    let id = create(&conversation, never).await;
    flush().await;
    let mut gate = storage.hold_commits();
    harness.resume();
    gate.entered().await;
    let closing = start_close(harness).await;
    gate.release();
    closing
        .await
        .expect("close task joins")
        .expect("close succeeds");
    assert!(!ran.load(Ordering::SeqCst));
    let last = storage.commits().pop().expect("reservation commit");
    let task_writes = last
        .into_iter()
        .filter_map(|write| match write {
            StorageWrite::Task { value } if value.id == id => Some(value),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(task_writes.len(), 1);
    assert_eq!(task_writes[0].state.status, TaskStateStatus::Running);
}
