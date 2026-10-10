//! Port of `packages/durable/test/session-watches.test.ts` (`watchDoc`
//! watches: exact committed frames, overflow collapse, retirement, context
//! propagation).
//!
//! D-deviations: identity assertions (`toBe`) collapse to structural equality;
//! listener failures are signaled by panicking (observation D2), so the
//! `listener_error` end carries the panic message.

use std::sync::Arc;

use parking_lot::Mutex;
use pi_chord::context::{
    create_context_key, with_abort_signal, with_cancel, with_context_value, AbortController,
    AbortReason, Context, BACKGROUND_CONTEXT,
};
use pi_chord::delta::Op;
use serde_json::{json, Value};

use crate::documents::{DocDefinition, DocToken, ScopeKind};
use crate::session::support::{context, document_changes, flush, open_test_session};
use crate::types::{JsonObject, WatchEnd};

fn obj(value: Value) -> JsonObject {
    value.as_object().expect("object literal").clone()
}

fn state_doc() -> DocToken {
    DocDefinition::singleton("watch.state", 1, ScopeKind::Session, || {
        obj(json!({ "value": 0, "items": ["a", "b"], "retained": { "label": "stable" } }))
    })
    .into()
}

/// JSON of one observed watch value.
fn value_json(value: &Option<Arc<JsonObject>>) -> Option<Value> {
    value
        .as_ref()
        .map(|object| Value::Object(object.as_ref().clone()))
}

/// Recorded delivery of one watch listener.
type Delivery = (Option<Value>, Vec<Op>);

type Record = Arc<Mutex<Vec<Delivery>>>;

fn record_listener(
    record: &Record,
) -> crate::session::observation::WatchListener<crate::session::observation::ObservedDocumentValue>
{
    let record = record.clone();
    Arc::new(move |value, ops, _context: Context| {
        let record = record.clone();
        Box::pin(async move {
            record.lock().push((value_json(&value), ops));
        })
    })
}

async fn create_state() -> crate::session::support::TestSession {
    let harness = open_test_session();
    let token = state_doc();
    harness
        .session
        .commit(
            move |tx| {
                tx.doc(&token, None, None, None)
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            },
            &context(),
        )
        .await
        .expect("commit");
    harness
}

async fn bump(session: &crate::session::session::Session, token: &DocToken, value: i64) {
    let token = token.clone();
    session
        .commit(
            move |tx| -> Result<(), String> {
                let draft = tx
                    .doc(&token, None, None, None)
                    .map_err(|e| e.to_string())?;
                draft
                    .set(&[pi_chord::delta::Seg::key("value")], json!(value))
                    .map_err(|e| e.to_string())
            },
            &context(),
        )
        .await
        .expect("commit");
}

/// Upstream `never creates an absent document`.
#[tokio::test]
async fn never_creates_an_absent_document() {
    let harness = open_test_session();
    let token = state_doc();
    let commits = harness.storage.commits().len();
    assert!(harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .is_none());
    assert_eq!(harness.storage.commits().len(), commits);
    assert_eq!(harness.storage.mint_count(), 0);
}

/// Upstream `keeps the acquisition revision until start and delivers exact
/// committed frames`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keeps_the_acquisition_revision_until_start_and_delivers_exact_committed_frames() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let initial = watch.value();
    bump(&harness.session, &token, 1).await;
    flush().await;
    let first_published = document_changes(
        &harness
            .publications
            .lock()
            .last()
            .expect("publication")
            .clone(),
    )[0]
    .clone();
    bump(&harness.session, &token, 2).await;
    flush().await;
    let second_published = document_changes(
        &harness
            .publications
            .lock()
            .last()
            .expect("publication")
            .clone(),
    )[0]
    .clone();
    assert_eq!(value_json(&watch.value()), value_json(&initial));

    let deliveries: Record = Arc::default();
    {
        let deliveries = deliveries.clone();
        watch
            .start(Arc::new(move |value, ops, _context: Context| {
                let deliveries = deliveries.clone();
                Box::pin(async move {
                    deliveries.lock().push((value_json(&value), ops));
                })
            }))
            .expect("start");
    }
    assert!(deliveries.lock().is_empty());
    flush().await;
    let guard = deliveries.lock();
    let values: Vec<Option<Value>> = guard.iter().map(|(value, _)| value.clone()).collect();
    let expected = [
        Some(Value::Object(
            document_value(&first_published).expect("live"),
        )),
        Some(Value::Object(
            document_value(&second_published).expect("live"),
        )),
    ];
    assert_eq!(values, expected);
    assert_eq!(&guard[0].1, &document_ops(&first_published));
    assert_eq!(&guard[1].1, &document_ops(&second_published));
    assert_eq!(value_json(&initial).unwrap()["value"], json!(0));
    drop(guard);
    watch.stop().await;
}

fn document_ops(change: &crate::types::DocumentCommitChange) -> Vec<Op> {
    match change {
        crate::types::DocumentCommitChange::Document { ops, .. } => ops.clone(),
        crate::types::DocumentCommitChange::DocumentCopy { .. } => {
            panic!("expected a document change")
        }
    }
}

fn document_value(change: &crate::types::DocumentCommitChange) -> Option<JsonObject> {
    match change {
        crate::types::DocumentCommitChange::Document { value, .. } => value.clone(),
        crate::types::DocumentCommitChange::DocumentCopy { .. } => {
            panic!("expected a document change")
        }
    }
}

/// Upstream `serializes callbacks and buffers exact frames committed while one
/// is in flight`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serializes_callbacks_and_buffers_exact_frames_committed_while_one_is_in_flight() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let (entered_tx, entered_receiver) = tokio::sync::oneshot::channel::<()>();
    let release = Arc::new(tokio::sync::Notify::new());
    let values: Arc<Mutex<Vec<i64>>> = Arc::default();
    let active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let max_active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let entered = Mutex::new(Some(entered_tx));
    {
        let values = values.clone();
        let active = active.clone();
        let max_active = max_active.clone();
        let release = release.clone();
        let entered_sender = entered;
        watch
            .start(Arc::new(move |value, _ops, _context: Context| {
                let values = values.clone();
                let active = active.clone();
                let max_active = max_active.clone();
                let release = release.clone();
                let entered_sender = entered_sender.lock().take();
                Box::pin(async move {
                    let now = active.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    max_active.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
                    values.lock().push(
                        value
                            .as_ref()
                            .map(|v| v["value"].as_i64().unwrap_or(-1))
                            .unwrap_or(-1),
                    );
                    if values.lock().len() == 1 {
                        if let Some(sender) = entered_sender {
                            let _ = sender.send(());
                        }
                        release.notified().await;
                    }
                    active.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                })
            }))
            .expect("start");
    }
    bump(&harness.session, &token, 1).await;
    tokio::time::timeout(std::time::Duration::from_secs(5), entered_receiver)
        .await
        .expect("entered")
        .expect("channel");
    for value in 2..=20 {
        bump(&harness.session, &token, value).await;
    }
    assert_eq!(*values.lock(), vec![1]);
    release.notify_waiters();
    flush().await;
    assert_eq!(max_active.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(*values.lock(), (1..=20).collect::<Vec<i64>>());
    watch.stop().await;
}

/// Upstream `allows a listener to initiate a later Session commit`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn allows_a_listener_to_initiate_a_later_session_commit() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let (completed_tx, completed_rx) = tokio::sync::oneshot::channel::<()>();
    let values: Arc<Mutex<Vec<i64>>> = Arc::default();
    {
        let values = values.clone();
        let token = token.clone();
        let completed = Arc::new(Mutex::new(Some(completed_tx)));
        // `Session::commit`'s future is !Send, so the listener signals a
        // spawned helper task to run the follow-up commit.
        let (requests_tx, requests_rx) = std::sync::mpsc::channel::<i64>();
        let session = harness.session.clone();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("helper runtime");
            for value in requests_rx {
                runtime.block_on(bump(&session, &token, value));
            }
        });
        watch
            .start(Arc::new(move |value, _ops, _context: Context| {
                let values = values.clone();
                let requests_tx = requests_tx.clone();
                let completed = completed.clone();
                Box::pin(async move {
                    let current = value
                        .as_ref()
                        .map(|v| v["value"].as_i64().unwrap_or(-1))
                        .unwrap_or(-1);
                    values.lock().push(current);
                    if current == 1 {
                        let _ = requests_tx.send(2);
                    } else if let Some(completed) = completed.lock().take() {
                        let _ = completed.send(());
                    }
                })
            }))
            .expect("start");
    }
    bump(&harness.session, &token, 1).await;
    tokio::time::timeout(std::time::Duration::from_secs(5), completed_rx)
        .await
        .expect("completed")
        .expect("channel");
    assert_eq!(*values.lock(), vec![1, 2]);
    watch.stop().await;
}

/// Upstream `collapses 101 pending commits to one root replacement`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn collapses_101_pending_commits_to_one_root_replacement() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    for value in 1..=101 {
        bump(&harness.session, &token, value).await;
    }
    let deliveries: Record = Arc::default();
    {
        let deliveries = deliveries.clone();
        watch.start(record_listener(&deliveries)).expect("start");
    }
    flush().await;
    let expected_value = value_json(&watch.value());
    let guard = deliveries.lock();
    assert_eq!(
        *guard,
        vec![(
            expected_value.clone(),
            vec![Op::Replace(expected_value.unwrap())]
        )]
    );
    drop(guard);
    watch.stop().await;
}

/// Upstream `never folds the in-flight frame into an overflow reset`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn never_folds_the_in_flight_frame_into_an_overflow_reset() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let entered = tokio::sync::oneshot::channel::<()>();
    let release = Arc::new(tokio::sync::Notify::new());
    let deliveries: Record = Arc::default();
    {
        let deliveries = deliveries.clone();
        let release = release.clone();
        let entered = Mutex::new(Some(entered.0));
        watch
            .start(Arc::new(move |value, ops, _context: Context| {
                let deliveries = deliveries.clone();
                let release = release.clone();
                let mut entered = entered.lock().take();
                Box::pin(async move {
                    deliveries.lock().push((value_json(&value), ops));
                    if deliveries.lock().len() == 1 {
                        if let Some(sender) = entered.take() {
                            let _ = sender.send(());
                        }
                        release.notified().await;
                    }
                })
            }))
            .expect("start");
    }
    bump(&harness.session, &token, 1).await;
    // Wait until the first delivery is in flight.
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    for value in 2..=102 {
        bump(&harness.session, &token, value).await;
    }
    release.notify_waiters();
    flush().await;
    let guard = deliveries.lock();
    assert_eq!(
        guard[0].0,
        Some(json!({ "value": 1, "items": ["a", "b"], "retained": { "label": "stable" } }))
    );
    let newest = value_json(&watch.value());
    assert_eq!(
        guard[1],
        (newest.clone(), vec![Op::Replace(newest.unwrap())])
    );
    assert_eq!(guard.len(), 2);
    drop(guard);
    watch.stop().await;
}

/// Upstream `folds retirement into an overflow reset and then closes`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn folds_retirement_into_an_overflow_reset_and_then_closes() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    for value in 1..=100 {
        bump(&harness.session, &token, value).await;
    }
    harness
        .session
        .commit(
            {
                let token = token.clone();
                move |tx| tx.retire_doc(&token, None, None).map_err(|e| e.to_string())
            },
            &context(),
        )
        .await
        .expect("retire");
    let deliveries: Record = Arc::default();
    {
        let deliveries = deliveries.clone();
        watch.start(record_listener(&deliveries)).expect("start");
    }
    assert_eq!(watch.closed().await, WatchEnd::Retired);
    assert_eq!(
        *deliveries.lock(),
        vec![(None, vec![Op::Replace(Value::Null)])]
    );
}

/// Upstream `delivers replayable structural no-op commits instead of
/// suppressing them`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delivers_replayable_structural_no_op_commits_instead_of_suppressing_them() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let initial = watch.value();
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, None, None, None)
                        .map_err(|e| e.to_string())?;
                    let first = draft
                        .shift(&[pi_chord::delta::Seg::key("items")])
                        .map_err(|e| e.to_string())?;
                    draft
                        .unshift(
                            &[pi_chord::delta::Seg::key("items")],
                            vec![first.expect("shifted")],
                        )
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    let newest = harness
        .session
        .snapshot(&token, None, None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(
        Value::Object((*newest).clone()),
        value_json(&initial).unwrap()
    );
    let deliveries: Record = Arc::default();
    {
        let deliveries = deliveries.clone();
        watch.start(record_listener(&deliveries)).expect("start");
    }
    flush().await;
    let guard = deliveries.lock();
    assert_eq!(guard.len(), 1);
    assert!(!guard[0].1.is_empty());
    assert_eq!(guard[0].0, Some(Value::Object((*newest).clone())));
    drop(guard);
    watch.stop().await;
}

/// Upstream `preserves commit Context values without inheriting producer
/// cancellation`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn preserves_commit_context_values_without_inheriting_producer_cancellation() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let key = create_context_key::<String>("watch-test");
    let controller = AbortController::new();
    let commit_context = with_context_value(
        &key,
        "newest-commit".to_string(),
        with_abort_signal(controller.signal(), BACKGROUND_CONTEXT.clone()),
    );
    let entered = tokio::sync::oneshot::channel::<()>();
    let release = Arc::new(tokio::sync::Notify::new());
    let delivery_context: Arc<Mutex<Option<Context>>> = Arc::default();
    {
        let delivery_context = delivery_context.clone();
        let release = release.clone();
        let entered = Mutex::new(Some(entered.0));
        watch
            .start(Arc::new(move |_value, _ops, delivered: Context| {
                let delivery_context = delivery_context.clone();
                let release = release.clone();
                let mut entered = entered.lock().take();
                Box::pin(async move {
                    *delivery_context.lock() = Some(delivered);
                    if let Some(sender) = entered.take() {
                        let _ = sender.send(());
                    }
                    release.notified().await;
                })
            }))
            .expect("start");
    }
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, None, None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[pi_chord::delta::Seg::key("value")], json!(1))
                        .map_err(|e| e.to_string())
                },
                &commit_context,
            )
            .await
            .expect("commit");
    }
    tokio::time::timeout(std::time::Duration::from_secs(5), entered.1)
        .await
        .expect("entered")
        .expect("channel");
    {
        let guard = delivery_context.lock();
        let delivered = guard.as_ref().expect("delivered");
        assert_eq!(delivered.value(&key), Some("newest-commit".to_string()));
        assert!(delivered.abort_signal().is_none());
    }
    controller.abort(AbortReason::Message("caller finished".into()));
    let stopped = watch.stop().await;
    assert_eq!(stopped, WatchEnd::Stopped);
    assert!(delivery_context
        .lock()
        .as_ref()
        .expect("delivered")
        .abort_signal()
        .is_none());
    release.notify_waiters();
}

/// Upstream `keeps earlier immutable revisions stable`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keeps_earlier_immutable_revisions_stable() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let initial = watch.value();
    let delivered: Arc<Mutex<Option<Option<Value>>>> = Arc::default();
    {
        let delivered = delivered.clone();
        watch
            .start(Arc::new(move |value, _ops, _context: Context| {
                let delivered = delivered.clone();
                Box::pin(async move {
                    *delivered.lock() = Some(value_json(&value));
                })
            }))
            .expect("start");
    }
    bump(&harness.session, &token, 1).await;
    flush().await;
    let guard = delivered.lock();
    assert_eq!(guard.as_ref().unwrap(), &value_json(&watch.value()));
    assert_ne!(guard.as_ref().unwrap(), &value_json(&initial));
    assert_eq!(
        guard.as_ref().unwrap().as_ref().unwrap()["retained"],
        value_json(&initial).unwrap()["retained"]
    );
    assert_eq!(value_json(&initial).unwrap()["value"], json!(0));
    drop(guard);
    watch.stop().await;
}

/// Upstream `delivers retirement and does not follow recreation`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delivers_retirement_and_does_not_follow_recreation() {
    let harness = create_state().await;
    let token = state_doc();
    let old_watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let values: Arc<Mutex<Vec<Option<Value>>>> = Arc::default();
    {
        let values = values.clone();
        old_watch
            .start(Arc::new(move |value, _ops, _context: Context| {
                let values = values.clone();
                Box::pin(async move {
                    values.lock().push(value_json(&value));
                })
            }))
            .expect("start");
    }
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    tx.retire_doc(&token, None, None)
                        .map_err(|e| e.to_string())?;
                    let draft = tx
                        .doc(&token, None, None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[pi_chord::delta::Seg::key("value")], json!(10))
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    assert_eq!(old_watch.closed().await, WatchEnd::Retired);
    assert_eq!(*values.lock(), vec![None]);
    assert_eq!(value_json(&old_watch.value()), None);

    let replacement = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    assert_eq!(
        value_json(&replacement.value()).unwrap()["value"],
        json!(10)
    );
    bump(&harness.session, &token, 11).await;
    flush().await;
    assert_eq!(value_json(&old_watch.value()), None);
    replacement.stop().await;
}

/// Upstream `Session close discards retirement buffered before start`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_close_discards_retirement_buffered_before_start() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let baseline = watch.value();
    harness
        .session
        .commit(
            {
                let token = token.clone();
                move |tx| tx.retire_doc(&token, None, None).map_err(|e| e.to_string())
            },
            &context(),
        )
        .await
        .expect("retire");
    harness.session.close(&context()).await.expect("close");
    assert_eq!(watch.closed().await, WatchEnd::SessionClosed);
    assert_eq!(value_json(&watch.value()), value_json(&baseline));
    let error = {
        let listener: crate::session::observation::WatchListener<
            crate::session::observation::ObservedDocumentValue,
        > = Arc::new(|_value, _ops, _context: Context| Box::pin(async {}));
        watch.start(listener).expect_err("stopped")
    };
    assert_eq!(error.0, "Watch is stopped");
}

/// Upstream `Session close discards retirement behind an in-flight callback`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_close_discards_retirement_behind_an_in_flight_callback() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let entered = tokio::sync::oneshot::channel::<()>();
    let release = Arc::new(tokio::sync::Notify::new());
    let values: Arc<Mutex<Vec<Option<i64>>>> = Arc::default();
    {
        let values = values.clone();
        let release = release.clone();
        let entered = Mutex::new(Some(entered.0));
        watch
            .start(Arc::new(move |value, _ops, _context: Context| {
                let values = values.clone();
                let release = release.clone();
                let mut entered = entered.lock().take();
                Box::pin(async move {
                    values
                        .lock()
                        .push(value.as_ref().map(|v| v["value"].as_i64().unwrap_or(-1)));
                    if let Some(sender) = entered.take() {
                        let _ = sender.send(());
                    }
                    release.notified().await;
                })
            }))
            .expect("start");
    }
    bump(&harness.session, &token, 1).await;
    tokio::time::timeout(std::time::Duration::from_secs(5), entered.1)
        .await
        .expect("entered")
        .expect("channel");
    harness
        .session
        .commit(
            {
                let token = token.clone();
                move |tx| tx.retire_doc(&token, None, None).map_err(|e| e.to_string())
            },
            &context(),
        )
        .await
        .expect("retire");
    harness.session.close(&context()).await.expect("close");
    assert_eq!(watch.closed().await, WatchEnd::SessionClosed);
    release.notify_waiters();
    flush().await;
    assert_eq!(*values.lock(), vec![Some(1)]);
}

/// Upstream `supports idempotent stop and rejects repeated or late start`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn supports_idempotent_stop_and_rejects_repeated_or_late_start() {
    let harness = create_state().await;
    let token = state_doc();
    let started = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    {
        let listener: crate::session::observation::WatchListener<
            crate::session::observation::ObservedDocumentValue,
        > = Arc::new(|_value, _ops, _context: Context| Box::pin(async {}));
        started.start(listener).expect("start");
        let listener: crate::session::observation::WatchListener<
            crate::session::observation::ObservedDocumentValue,
        > = Arc::new(|_value, _ops, _context: Context| Box::pin(async {}));
        let error = started.start(listener).expect_err("already started");
        assert_eq!(error.0, "Watch is already started");
    }
    let first = started.stop().await;
    let second = started.stop().await;
    assert_eq!(second, first);
    assert_eq!(first, WatchEnd::Stopped);

    let stopped = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    stopped.stop().await;
    let listener: crate::session::observation::WatchListener<
        crate::session::observation::ObservedDocumentValue,
    > = Arc::new(|_value, _ops, _context: Context| Box::pin(async {}));
    let error = stopped.start(listener).expect_err("stopped");
    assert_eq!(error.0, "Watch is stopped");
}

/// Upstream `cancels acquisition without leaking a registered watch`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancels_acquisition_without_leaking_a_registered_watch() {
    let harness = create_state().await;
    let token = state_doc();
    harness.session.unload_documents().await;
    let mut gate = harness.storage.hold_find_document();
    let child = with_cancel(context());
    let acquire_context = child.context.clone();
    let acquire_session = harness.session.clone();
    let acquire_token = token.clone();
    let mut acquisition = tokio::spawn(async move {
        acquire_session
            .watch_doc(&acquire_token, None, None, &acquire_context)
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), gate.entered())
        .await
        .expect("watch acquisition reaches storage");
    child.cancel(AbortReason::Message("cancel acquisition".into()));
    gate.release();
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), &mut acquisition)
        .await
        .expect("watch acquisition settles")
        .expect("acquisition task");
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("acquisition should be cancelled"),
    };
    assert_eq!(error.to_string(), "cancel acquisition");
    harness.session.close(&context()).await.expect("close");
}

/// Upstream `cancels future delivery without aborting an in-flight callback`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancels_future_delivery_without_aborting_an_in_flight_callback() {
    let harness = create_state().await;
    let token = state_doc();
    let child = with_cancel(context());
    let watch = harness
        .session
        .watch_doc(&token, None, None, &child.context)
        .await
        .unwrap()
        .expect("watch");
    let entered = tokio::sync::oneshot::channel::<()>();
    let release = Arc::new(tokio::sync::Notify::new());
    let delivery_context: Arc<Mutex<Option<Context>>> = Arc::default();
    {
        let delivery_context = delivery_context.clone();
        let release = release.clone();
        let entered = Mutex::new(Some(entered.0));
        watch
            .start(Arc::new(move |_value, _ops, delivered: Context| {
                let delivery_context = delivery_context.clone();
                let release = release.clone();
                let mut entered = entered.lock().take();
                Box::pin(async move {
                    *delivery_context.lock() = Some(delivered);
                    if let Some(sender) = entered.take() {
                        let _ = sender.send(());
                    }
                    release.notified().await;
                })
            }))
            .expect("start");
    }
    bump(&harness.session, &token, 1).await;
    tokio::time::timeout(std::time::Duration::from_secs(5), entered.1)
        .await
        .expect("entered")
        .expect("channel");
    child.cancel_default();
    assert_eq!(watch.closed().await, WatchEnd::Cancelled);
    assert!(delivery_context
        .lock()
        .as_ref()
        .expect("delivered")
        .abort_signal()
        .is_none());
    release.notify_waiters();
}

/// Upstream `Session close stops future delivery without joining an in-flight callback`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_close_stops_future_delivery_without_joining_an_in_flight_callback() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let entered = tokio::sync::oneshot::channel::<()>();
    let release = Arc::new(tokio::sync::Notify::new());
    let delivery_context: Arc<Mutex<Option<Context>>> = Arc::default();
    {
        let delivery_context = delivery_context.clone();
        let release = release.clone();
        let entered = Mutex::new(Some(entered.0));
        watch
            .start(Arc::new(move |_value, _ops, delivered: Context| {
                let delivery_context = delivery_context.clone();
                let release = release.clone();
                let mut entered = entered.lock().take();
                Box::pin(async move {
                    *delivery_context.lock() = Some(delivered);
                    if let Some(sender) = entered.take() {
                        let _ = sender.send(());
                    }
                    release.notified().await;
                })
            }))
            .expect("start");
    }
    bump(&harness.session, &token, 1).await;
    tokio::time::timeout(std::time::Duration::from_secs(5), entered.1)
        .await
        .expect("entered")
        .expect("channel");
    let close_context = context();
    let close_session = harness.session.clone();
    let mut close = tokio::spawn(async move { close_session.close(&close_context).await });
    assert!(delivery_context
        .lock()
        .as_ref()
        .expect("delivered")
        .abort_signal()
        .is_none());
    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(5), watch.closed())
            .await
            .expect("watch closes"),
        WatchEnd::SessionClosed
    );
    tokio::time::timeout(std::time::Duration::from_secs(5), &mut close)
        .await
        .expect("close settles")
        .expect("close task")
        .expect("close");
    release.notify_waiters();
}

/// Upstream `settles listener failure on only the affected watch`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn settles_listener_failure_on_only_the_affected_watch() {
    let harness = create_state().await;
    let token = state_doc();
    let failed = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let healthy = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    failed
        .start(Arc::new(|_value, _ops, _context: Context| {
            Box::pin(async {
                panic!("listener failed");
            })
        }))
        .expect("start");
    let healthy_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    {
        let healthy_calls = healthy_calls.clone();
        healthy
            .start(Arc::new(move |_value, _ops, _context: Context| {
                let healthy_calls = healthy_calls.clone();
                Box::pin(async move {
                    healthy_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                })
            }))
            .expect("start");
    }
    bump(&harness.session, &token, 1).await;
    assert_eq!(
        failed.closed().await,
        WatchEnd::ListenerError {
            error: "listener failed".into()
        }
    );
    flush().await;
    assert_eq!(healthy_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    healthy.stop().await;
}

/// Upstream `hydrates migration without writing and observes the later exact edit`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hydrates_migration_without_writing_and_observes_the_later_exact_edit() {
    let old: DocToken = DocDefinition::singleton("watch.migration", 1, ScopeKind::Session, || {
        obj(json!({ "value": 3 }))
    })
    .into();
    let current: DocToken =
        DocDefinition::singleton("watch.migration", 2, ScopeKind::Session, || {
            obj(json!({ "value": 0, "migrated": false }))
        })
        .with_migrate(|value, _from| obj(json!({ "value": value["value"], "migrated": true })))
        .into();
    let harness = open_test_session();
    harness
        .session
        .commit(
            {
                let old = old.clone();
                move |tx| {
                    tx.doc(&old, None, None, None)
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                }
            },
            &context(),
        )
        .await
        .expect("commit");
    harness.session.unload_documents().await;
    let commits = harness.storage.commits().len();
    let watch = harness
        .session
        .watch_doc(&current, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    assert_eq!(
        value_json(&watch.value()),
        Some(json!({ "value": 3, "migrated": true }))
    );
    assert_eq!(harness.storage.commits().len(), commits);
    let values: Arc<Mutex<Vec<i64>>> = Arc::default();
    {
        let values = values.clone();
        watch
            .start(Arc::new(move |value, _ops, _context: Context| {
                let values = values.clone();
                Box::pin(async move {
                    values.lock().push(
                        value
                            .as_ref()
                            .map(|v| v["value"].as_i64().unwrap_or(-1))
                            .unwrap_or(-1),
                    );
                })
            }))
            .expect("start");
    }
    harness
        .session
        .commit(
            {
                let current = current.clone();
                move |tx| {
                    tx.doc(&current, None, None, None)
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                }
            },
            &context(),
        )
        .await
        .expect("commit");
    flush().await;
    assert_eq!(*values.lock(), Vec::<i64>::new());
    bump(&harness.session, &current, 4).await;
    flush().await;
    assert_eq!(*values.lock(), vec![4]);
    watch.stop().await;
}

/// Upstream `can replay every delivered exact operation batch from the
/// acquisition revision`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn can_replay_every_delivered_exact_operation_batch_from_the_acquisition_revision() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let replica: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(value_json(&watch.value())));
    {
        let replica = replica.clone();
        watch
            .start(Arc::new(move |value, ops, _context: Context| {
                let replica = replica.clone();
                Box::pin(async move {
                    let mut guard = replica.lock();
                    for op in &ops {
                        if let Some(current) = guard.take() {
                            *guard = Some(
                                pi_chord::delta::apply_immutable(
                                    &current,
                                    std::slice::from_ref(op),
                                )
                                .expect("replay"),
                            );
                        }
                    }
                    assert_eq!(*guard, value_json(&value));
                })
            }))
            .expect("start");
    }
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, None, None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[pi_chord::delta::Seg::key("value")], json!(3))
                        .map_err(|e| e.to_string())?;
                    draft
                        .splice(&[pi_chord::delta::Seg::key("items")], 0, Some(1), vec![])
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    flush().await;
    assert_eq!(*replica.lock(), value_json(&watch.value()));
    watch.stop().await;
}

/// Upstream `continues after the tracker cache unloads`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn continues_after_the_tracker_cache_unloads() {
    let harness = create_state().await;
    let token = state_doc();
    let watch = harness
        .session
        .watch_doc(&token, None, None, &context())
        .await
        .unwrap()
        .expect("watch");
    let baseline = watch.value();
    let reads = harness.storage.document_read_count();
    harness.session.unload_documents().await;
    let reloaded = harness
        .session
        .snapshot(&token, None, None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(*reloaded, **baseline.as_ref().expect("live"));
    assert_eq!(harness.storage.document_read_count() > reads, true);
    {
        let listener: crate::session::observation::WatchListener<
            crate::session::observation::ObservedDocumentValue,
        > = Arc::new(|_value, _ops, _context: Context| Box::pin(async {}));
        watch.start(listener).expect("start");
    }
    bump(&harness.session, &token, 7).await;
    flush().await;
    assert_eq!(value_json(&watch.value()).unwrap()["value"], json!(7));
    watch.stop().await;
}
