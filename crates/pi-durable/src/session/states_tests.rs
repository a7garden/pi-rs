//! Port of `packages/durable/test/session-states.test.ts` (`documentState`:
//! disposable Chord read-only states over one committed incarnation).
//!
//! D-deviations:
//! - JS reference-identity assertions (`toBe` on values and operation arrays)
//!   collapse to structural equality; exact shared `Arc` values are not
//!   observable through the chord `JsonValue` surface.
//! - `exposes trusted shared immutable values without freezing` drops the
//!   `Object.isFrozen` probes (unrepresentable) and asserts the shared
//!   committed value instead.

use std::sync::Arc;

use parking_lot::Mutex;
use pi_chord::context::Context;
use serde_json::{json, Value};

use crate::documents::{DocDefinition, DocToken, ScopeKind};
use crate::session::support::{
    context, create_conversation, document_changes, flush, open_test_session,
};
use crate::types::{ConversationId, DocumentFork, DocumentHistory, JsonObject};

fn obj(value: Value) -> JsonObject {
    value.as_object().expect("object literal").clone()
}

fn state_doc() -> DocToken {
    DocDefinition::singleton("state.state", 1, ScopeKind::Session, || {
        obj(json!({ "value": 0, "retained": { "label": "stable" } }))
    })
    .into()
}

fn family_doc() -> DocToken {
    DocDefinition::family("state.family", 1, ScopeKind::Session, |seed| {
        obj(json!({ "value": seed.as_str().expect("string seed").len() }))
    })
    .into()
}

/// Recorded `(value, delivery sequence)` of one state subscriber.
type Recorded = Arc<Mutex<Vec<(Value, u64)>>>;

fn subscribe_values(
    state: &Arc<pi_chord::AttachedReplicatedState>,
    record: &Recorded,
) -> pi_chord::types::Unsubscribe {
    let record = record.clone();
    state.subscribe(Arc::new(
        move |value: &Value, _context: &Context, delivery| {
            let record = record.clone();
            let value = value.clone();
            let sequence = delivery.sequence;
            Box::pin(async move {
                record.lock().push((value, sequence));
            })
        },
    ))
}

async fn set_value(session: &crate::session::session::Session, token: &DocToken, value: i64) {
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
    flush().await;
    harness
}

async fn state_value(state: &Arc<pi_chord::AttachedReplicatedState>) -> Value {
    state.value().clone()
}

/// Upstream `never creates an absent document`.
#[tokio::test]
async fn never_creates_an_absent_document() {
    let harness = open_test_session();
    let state = state_doc();
    let family = family_doc();
    let commits = harness.storage.commits().len();
    assert!(harness
        .session
        .document_state(&state, None, None, &context())
        .await
        .unwrap()
        .is_none());
    assert!(harness
        .session
        .document_state(&family, None, Some("missing"), &context())
        .await
        .unwrap()
        .is_none());
    assert_eq!(harness.storage.commits().len(), commits);
    assert_eq!(harness.storage.mint_count(), 0);
}

/// Upstream `returns an immediately hydrated read-only state with contiguous
/// Chord deliveries`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn returns_an_immediately_hydrated_read_only_state_with_contiguous_chord_deliveries() {
    let harness = create_state().await;
    let token = state_doc();
    let baseline = harness
        .session
        .snapshot(&token, None, None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    let state = harness
        .session
        .document_state(&token, None, None, &context())
        .await
        .unwrap()
        .expect("state");
    let deliveries: Recorded = Arc::default();
    let unsubscribe = subscribe_values(&state, &deliveries);
    assert_eq!(
        state_value(&state).await,
        Value::Object((*baseline).clone())
    );

    set_value(&harness.session, &token, 1).await;
    set_value(&harness.session, &token, 2).await;
    flush().await;

    let guard = deliveries.lock();
    assert_eq!(
        guard
            .iter()
            .map(|(_, sequence)| *sequence)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        guard
            .iter()
            .map(|(value, _)| value["value"].clone())
            .collect::<Vec<_>>(),
        vec![json!(0), json!(1), json!(2)]
    );
    drop(guard);
    let snapshot = harness
        .session
        .snapshot(&token, None, None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(
        state_value(&state).await,
        Value::Object((*snapshot).clone())
    );
    unsubscribe();
    state.dispose();
}

/// Upstream `creates independent disposable states for one incarnation`
/// (state-object identity becomes separate subscriptions — D).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn creates_independent_disposable_states_for_one_incarnation() {
    let harness = create_state().await;
    let token = state_doc();
    let first = harness
        .session
        .document_state(&token, None, None, &context())
        .await
        .unwrap()
        .expect("state");
    let second = harness
        .session
        .document_state(&token, None, None, &context())
        .await
        .unwrap()
        .expect("state");
    assert!(!Arc::ptr_eq(&first, &second));
    set_value(&harness.session, &token, 1).await;
    flush().await;
    assert_eq!(state_value(&first).await["value"], json!(1));
    assert_eq!(state_value(&second).await["value"], json!(1));

    first.dispose();
    set_value(&harness.session, &token, 2).await;
    flush().await;
    assert_eq!(state_value(&first).await["value"], json!(1));
    assert_eq!(state_value(&second).await["value"], json!(2));
    second.dispose();
}

/// Upstream `shares exact committed value and operation references with
/// Chord` — identity becomes structural equality; the internals subscription
/// carries the exact published operation batch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shares_exact_committed_value_and_operation_references_with_chord() {
    let harness = create_state().await;
    let token = state_doc();
    let state = harness
        .session
        .document_state(&token, None, None, &context())
        .await
        .unwrap()
        .expect("state");
    let received_ops: Arc<Mutex<Vec<Vec<pi_chord::delta::Op>>>> = Arc::default();
    {
        let received_ops = received_ops.clone();
        state.internals().subscribe(Arc::new(
            move |ops: &[pi_chord::delta::Op], _sequence: u64, _context: &Context| {
                received_ops.lock().push(ops.to_vec());
            },
        ));
    }
    set_value(&harness.session, &token, 4).await;
    flush().await;
    let publication = harness
        .publications
        .lock()
        .last()
        .expect("publication")
        .clone();
    let published = document_changes(&publication)[0].clone();
    let published_ops = match &published {
        crate::types::DocumentCommitChange::Document { ops, .. } => ops.clone(),
        crate::types::DocumentCommitChange::DocumentCopy { .. } => {
            panic!("expected document change")
        }
    };
    assert_eq!(state.value().clone()["value"], json!(4));
    assert_eq!(
        *received_ops.lock().last().expect("ops delivered"),
        published_ops
    );
    let snapshot = harness
        .session
        .snapshot(&token, None, None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(state.value().clone()["retained"], snapshot["retained"]);
    state.dispose();
}

/// Upstream `captures a late baseline without redelivering an already covered commit`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn captures_a_late_baseline_without_redelivering_an_already_covered_commit() {
    let harness = create_state().await;
    let token = state_doc();
    set_value(&harness.session, &token, 1).await;
    let state = harness
        .session
        .document_state(&token, None, None, &context())
        .await
        .unwrap()
        .expect("state");
    let deliveries: Recorded = Arc::default();
    {
        let deliveries = deliveries.clone();
        state.subscribe(Arc::new(
            move |_value: &Value, _context: &Context, delivery| {
                let deliveries = deliveries.clone();
                let sequence = delivery.sequence;
                Box::pin(async move {
                    deliveries.lock().push((Value::Null, sequence));
                })
            },
        ));
    }
    flush().await;
    assert_eq!(state_value(&state).await["value"], json!(1));
    assert_eq!(
        deliveries
            .lock()
            .iter()
            .map(|(_, sequence)| *sequence)
            .collect::<Vec<_>>(),
        vec![0]
    );
    state.dispose();
}

/// Upstream `publishes null retirement and never follows a replacement incarnation`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn publishes_null_retirement_and_never_follows_a_replacement_incarnation() {
    let harness = create_state().await;
    let token = state_doc();
    let old_state = harness
        .session
        .document_state(&token, None, None, &context())
        .await
        .unwrap()
        .expect("state");
    let retirement_ops: Arc<Mutex<Vec<Vec<pi_chord::delta::Op>>>> = Arc::default();
    {
        let retirement_ops = retirement_ops.clone();
        old_state.internals().subscribe(Arc::new(
            move |ops: &[pi_chord::delta::Op], _sequence: u64, _context: &Context| {
                retirement_ops.lock().push(ops.to_vec());
            },
        ));
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
    flush().await;
    assert_eq!(old_state.value().clone(), Value::Null);
    assert_eq!(
        *retirement_ops.lock().last().expect("retirement delivered"),
        vec![pi_chord::delta::Op::Replace(Value::Null)]
    );

    let replacement = harness
        .session
        .document_state(&token, None, None, &context())
        .await
        .unwrap()
        .expect("replacement state");
    assert_eq!(replacement.value().clone()["value"], json!(10));
    set_value(&harness.session, &token, 11).await;
    flush().await;
    assert_eq!(old_state.value().clone(), Value::Null);
    assert_eq!(replacement.value().clone()["value"], json!(11));
    old_state.dispose();
    replacement.dispose();
}

/// Upstream `cold-loads a definition-free fork copy`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_loads_a_definition_free_fork_copy() {
    let copied: DocToken = DocDefinition::conversation(
        "state.copied",
        1,
        DocumentHistory::Latest,
        DocumentFork::Current,
        || obj(json!({ "value": 0 })),
    )
    .into();
    let harness = open_test_session();
    let parent_id = create_conversation(&harness.session).await;
    let entry = {
        let copied = copied.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<_, String> {
                    let created = tx
                        .append_entry(
                            parent_id,
                            crate::types::EntryDraft {
                                kind: Some("point".into()),
                                ..Default::default()
                            },
                        )
                        .map_err(|e| e.to_string())?;
                    let draft = tx
                        .doc(&copied, Some(parent_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[pi_chord::delta::Seg::key("value")], json!(7))
                        .map_err(|e| e.to_string())?;
                    Ok(created)
                },
                &context(),
            )
            .await
            .expect("commit")
    };
    let child_id = harness
        .session
        .commit(
            move |tx| {
                tx.fork_conversation(
                    parent_id,
                    entry.id,
                    crate::types::ConversationOwnership::Ownerless,
                )
                .map(|record| record.id)
                .map_err(|e| e.to_string())
            },
            &context(),
        )
        .await
        .expect("fork");
    let reads = harness.storage.document_read_count();
    let state = harness
        .session
        .document_state(&copied, Some(child_id.0), None, &context())
        .await
        .unwrap()
        .expect("state");
    assert_eq!(state.value().clone(), json!({ "value": 7 }));
    assert!(harness.storage.document_read_count() > reads);
    state.dispose();
}

/// Upstream `hydrates a migrated tracker without writing and skips an equal
/// version-base update`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hydrates_a_migrated_tracker_without_writing_and_skips_an_equal_version_base_update() {
    let old: DocToken = DocDefinition::singleton("state.migration", 1, ScopeKind::Session, || {
        obj(json!({ "value": 3 }))
    })
    .into();
    let current: DocToken =
        DocDefinition::singleton("state.migration", 2, ScopeKind::Session, || {
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
    let state = harness
        .session
        .document_state(&current, None, None, &context())
        .await
        .unwrap()
        .expect("state");
    assert_eq!(
        state.value().clone(),
        json!({ "value": 3, "migrated": true })
    );
    assert_eq!(harness.storage.commits().len(), commits);
    let baseline = state_value(&state).await;

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
    assert_eq!(harness.storage.commits().len(), commits + 1);
    assert_eq!(state_value(&state).await, baseline);
    set_value(&harness.session, &current, 4).await;
    flush().await;
    assert_eq!(
        state_value(&state).await,
        json!({ "value": 4, "migrated": true })
    );
    state.dispose();
}

/// Upstream `continues from exact committed values after the tracker cache unloads`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn continues_from_exact_committed_values_after_the_tracker_cache_unloads() {
    let harness = create_state().await;
    let token = state_doc();
    let state = harness
        .session
        .document_state(&token, None, None, &context())
        .await
        .unwrap()
        .expect("state");
    let baseline = state_value(&state).await;
    let reads = harness.storage.document_read_count();
    harness.session.unload_documents().await;
    let reloaded = harness
        .session
        .snapshot(&token, None, None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(Value::Object((*reloaded).clone()), baseline);
    assert!(harness.storage.document_read_count() > reads);
    set_value(&harness.session, &token, 6).await;
    flush().await;
    assert_eq!(state_value(&state).await["value"], json!(6));
    state.dispose();
}

/// Upstream `exposes trusted shared immutable values without freezing` — the
/// freeze probes are unrepresentable (D); the shared committed value carries
/// the case.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exposes_trusted_shared_immutable_values() {
    let harness = create_state().await;
    let token = state_doc();
    let snapshot = harness
        .session
        .snapshot(&token, None, None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    let state = harness
        .session
        .document_state(&token, None, None, &context())
        .await
        .unwrap()
        .expect("state");
    assert_eq!(state.value().clone(), Value::Object((*snapshot).clone()));
    assert_eq!(
        state.value().clone()["retained"],
        json!({ "label": "stable" })
    );
    state.dispose();
}

// Keep the unused import honest for the ConversationId re-export.
#[allow(dead_code)]
fn unused(conversation_id: ConversationId) -> ConversationId {
    conversation_id
}
