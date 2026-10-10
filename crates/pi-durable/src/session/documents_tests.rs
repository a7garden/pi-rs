//! Port of `packages/durable/test/session-documents.test.ts` (transaction
//! documents: bases, deltas, retirement, migration rejection, settlement).
//!
//! D-deviations:
//! - JS reference-identity assertions (`toBe` on values/subtrees) collapse to
//!   structural equality; per-placement copying and shared-subtree identity
//!   are unobservable over `serde_json::Value` clones.
//! - Pending asynchronous acquisitions are unrepresentable: `Tx::doc` is
//!   synchronous (module D1), so `rejects a callback that succeeds with a
//!   pending acquisition`, `does not initialize or mint for an absent
//!   acquisition that finishes after settlement`, and `rejects with the
//!   callback error when it fails with a pending acquisition` are deferred.
//! - Strict-JSON rejections are unrepresentable (module header): `rejects
//!   non-JSON initializer values and draft placements` is deferred.
//! - `poisons the Session after an uncertain Storage failure` adapts to
//!   session D4/D5: `Storage::commit` rejection carries the guarantee that no
//!   batch effect committed, so the Session is not poisoned and remains
//!   usable; the error still surfaces and nothing publishes.
//! - `rejects Tx use after the callback settles` adapts to the borrowed-`Tx`
//!   shape: the escrowable object is the `DocDraft` handle (session D2), so
//!   the settled-draft rejections stand in for the settled-`Tx` ones.
//! - Context identity of a publication listener is asserted by delivery
//!   occurrence; `Context` has no public identity comparison.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use pi_chord::delta::Seg;
use serde_json::{json, Value};

use crate::documents::{DocDefinition, DocToken, ScopeKind};
use crate::session::support::{
    context, create_conversation, document_changes, flush, open_test_session,
};
use crate::types::Storage as _;
use crate::types::{
    CommitChange, ConversationId, DocumentFork, DocumentHistory, DocumentScope, JsonObject,
    StorageWrite, TableCommitChange,
};

fn obj(value: Value) -> JsonObject {
    value.as_object().expect("object literal").clone()
}

/// Destructure the `Document` variant of a document commit change.
fn as_document(
    change: &crate::types::DocumentCommitChange,
) -> (
    &crate::types::DocumentRecord,
    Option<ConversationId>,
    Option<u64>,
    &Option<JsonObject>,
    &Vec<pi_chord::delta::Op>,
) {
    match change {
        crate::types::DocumentCommitChange::Document {
            record,
            conversation_id,
            version,
            value,
            ops,
        } => (record, *conversation_id, *version, value, ops),
        crate::types::DocumentCommitChange::DocumentCopy { .. } => {
            panic!("expected a document change")
        }
    }
}

fn document_writes(writes: &[StorageWrite]) -> Vec<&StorageWrite> {
    writes
        .iter()
        .filter(|write| {
            matches!(
                write,
                StorageWrite::DocumentCreate { .. }
                    | StorageWrite::DocumentChange { .. }
                    | StorageWrite::DocumentRetire { .. }
            )
        })
        .collect()
}

fn live_definition() -> DocDefinition {
    DocDefinition::conversation(
        "test.live",
        1,
        DocumentHistory::Latest,
        DocumentFork::Initial,
        || obj(json!({ "items": [], "nested": { "count": 0 }, "other": { "label": "x" } })),
    )
}

fn live_doc() -> DocToken {
    live_definition().into()
}

fn rewindable_live_doc() -> DocToken {
    DocDefinition::conversation(
        "test.live",
        1,
        DocumentHistory::Rewindable,
        DocumentFork::AsOf,
        || obj(json!({ "items": [], "nested": { "count": 0 }, "other": { "label": "x" } })),
    )
    .into()
}

fn counter_doc() -> DocToken {
    DocDefinition::singleton("test.counter", 1, ScopeKind::Session, || {
        obj(json!({ "count": 0 }))
    })
    .into()
}

fn member_doc() -> DocToken {
    DocDefinition::family("test.member", 1, ScopeKind::Session, |seed| {
        obj(json!({ "seed": seed, "hits": 0 }))
    })
    .into()
}

async fn set_message(
    session: &crate::session::session::Session,
    token: &DocToken,
    conversation_id: ConversationId,
    message: &str,
) {
    let token = token.clone();
    let message = message.to_string();
    session
        .commit(
            move |tx| -> Result<(), String> {
                let draft = tx
                    .doc(&token, Some(conversation_id.0), None, None)
                    .map_err(|e| e.to_string())?;
                draft
                    .set(&[Seg::key("message")], json!(message))
                    .map_err(|e| e.to_string())?;
                Ok(())
            },
            &context(),
        )
        .await
        .expect("commit");
}

/// Upstream `setupLive`.
async fn setup_live() -> (
    crate::session::support::TestSession,
    ConversationId,
    DocToken,
) {
    let harness = open_test_session();
    let token = live_doc();
    let conversation_id = create_conversation(&harness.session).await;
    set_message_keyed(&harness.session, &token, conversation_id, &["a", "b"]).await;
    (harness, conversation_id, token)
}

async fn set_message_keyed(
    session: &crate::session::session::Session,
    token: &DocToken,
    conversation_id: ConversationId,
    items: &[&str],
) {
    let token = token.clone();
    let items: Vec<Value> = items.iter().map(|item| json!(item)).collect();
    session
        .commit(
            move |tx| -> Result<(), String> {
                let draft = tx
                    .doc(&token, Some(conversation_id.0), None, None)
                    .map_err(|e| e.to_string())?;
                draft
                    .push(&[Seg::key("items")], items)
                    .map_err(|e| e.to_string())?;
                Ok(())
            },
            &context(),
        )
        .await
        .expect("commit");
}

/// Upstream `creates an initial base on first access and adopts it after Storage success`.
#[tokio::test]
async fn creates_an_initial_base_on_first_access_and_adopts_it_after_storage_success() {
    let (harness, conversation_id, token) = {
        let harness = open_test_session();
        let token = live_doc();
        let conversation_id = create_conversation(&harness.session).await;
        set_message(&harness.session, &token, conversation_id, "hello").await;
        (harness, conversation_id, token)
    };
    let writes = harness.storage.commits().pop().expect("writes");
    assert_eq!(writes.len(), 1);
    match &writes[0] {
        StorageWrite::DocumentCreate { record, content } => {
            assert_eq!(record.kind, "test.live");
            assert_eq!(
                record.scope,
                DocumentScope::Conversation { conversation_id }
            );
            assert_eq!(record.history, Some(DocumentHistory::Latest));
            assert_eq!(record.fork, Some(DocumentFork::Initial));
            assert_eq!(content.version, 1);
            assert_eq!(content.value["message"], json!("hello"));
            assert_eq!(content.value["items"], json!([]));
            assert_eq!(content.value["nested"], json!({ "count": 0 }));
        }
        other => panic!("expected document.create, got {other:?}"),
    }
    let snapshot = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(
        *snapshot,
        obj(
            json!({ "message": "hello", "items": [], "nested": { "count": 0 }, "other": { "label": "x" } })
        )
    );
    flush().await;
    let publication = harness
        .publications
        .lock()
        .last()
        .expect("publication")
        .clone();
    let published = as_document(document_changes(&publication)[0]);
    assert_eq!(published.0.created_at, publication.seq);
    assert_eq!(published.3, &Some((*snapshot).clone()));
    assert_eq!(published.1, Some(conversation_id));
    let admitted = harness.storage.admitted_commits().pop().expect("admitted");
    for write in &admitted {
        if let StorageWrite::DocumentCreate { content, .. } = write {
            assert_eq!(&content.value, published.3.as_ref().unwrap());
        }
    }
}

/// Upstream `never creates on snapshot and returns undefined when absent`.
#[tokio::test]
async fn never_creates_on_snapshot_and_returns_undefined_when_absent() {
    let harness = open_test_session();
    let conversation_id = create_conversation(&harness.session).await;
    let before = harness.storage.commits().len();
    let token = live_doc();
    let counter = counter_doc();
    let member = member_doc();
    assert!(harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .unwrap()
        .is_none());
    assert!(harness
        .session
        .snapshot(&counter, None, None, &context())
        .await
        .unwrap()
        .is_none());
    assert!(harness
        .session
        .snapshot(&member, None, Some("k"), &context())
        .await
        .unwrap()
        .is_none());
    assert_eq!(harness.storage.commits().len(), before);
    assert_eq!(harness.storage.mint_count(), 1);
}

/// Upstream `returns shared immutable snapshots and keeps prior revisions
/// stable` (identity assertions become structural equality — D).
#[tokio::test]
async fn returns_shared_immutable_snapshots_and_keeps_prior_revisions_stable() {
    let (harness, conversation_id, token) = setup_live().await;
    let first = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    let second_before = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(*second_before, *first);
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[Seg::key("nested"), Seg::key("count")], json!(1))
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    let second = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_ne!(*second, *first);
    assert_eq!(first["nested"]["count"], json!(0));
    assert_eq!(second["nested"]["count"], json!(1));
}

/// Upstream `adopts by pointer swap and shares operation payloads with the
/// published revision` (pointer identity becomes op equality — D).
#[tokio::test]
async fn adopts_by_pointer_swap_and_shares_operation_payloads_with_the_published_revision() {
    let (harness, conversation_id, token) = setup_live().await;
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[Seg::key("other")], json!({ "label": "y" }))
                        .map_err(|e| e.to_string())?;
                    draft
                        .push(&[Seg::key("items")], vec![json!("c")])
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    flush().await;
    let snapshot = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    let publication = harness
        .publications
        .lock()
        .last()
        .expect("publication")
        .clone();
    let published = as_document(document_changes(&publication)[0]);
    assert_eq!(published.3, &Some((*snapshot).clone()));
    let admitted = harness.storage.admitted_commits().pop().expect("admitted");
    for write in &admitted {
        if let StorageWrite::DocumentChange { content, .. } = write {
            match content {
                crate::types::DocumentContent::Delta { ops, .. } => {
                    assert_eq!(ops, published.4);
                    let set = ops
						.iter()
						.find(|op| matches!(op, pi_chord::delta::Op::Set { path, value: _, .. } if path == &[Seg::key("other")]));
                    assert!(set.is_some(), "expected set on other, ops {ops:?}");
                }
                crate::types::DocumentContent::Base { .. } => panic!("expected delta"),
            }
        }
    }
}

/// Upstream `copies assigned values per placement` (per-placement copying is
/// inherent to `Value` clones; only the outcome is asserted — D).
#[tokio::test]
async fn copies_assigned_values_per_placement() {
    let (harness, conversation_id, token) = setup_live().await;
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    let shared = json!({ "label": "shared" });
                    draft
                        .set(&[Seg::key("other")], shared.clone())
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[Seg::key("copy")], shared)
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    let snapshot = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(snapshot["other"], json!({ "label": "shared" }));
    assert_eq!(snapshot["copy"], json!({ "label": "shared" }));
}

/// Upstream `suppresses writes and publications for empty batches`.
#[tokio::test]
async fn suppresses_writes_and_publications_for_empty_batches() {
    let (harness, conversation_id, token) = setup_live().await;
    flush().await;
    let commits = harness.storage.commits().len();
    let published = harness.publications.lock().len();
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[Seg::key("nested"), Seg::key("count")], json!(0))
                        .map_err(|e| e.to_string())?;
                    draft
                        .push(&[Seg::key("items")], vec![json!("z")])
                        .map_err(|e| e.to_string())?;
                    draft.pop(&[Seg::key("items")]).map_err(|e| e.to_string())?;
                    Ok(())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    flush().await;
    assert_eq!(harness.storage.commits().len(), commits);
    assert_eq!(harness.publications.lock().len(), published);
}

/// Upstream `writes and publishes replayable nonempty structural no-ops`.
#[tokio::test]
async fn writes_and_publishes_replayable_nonempty_structural_no_ops() {
    let (harness, conversation_id, token) = setup_live().await;
    let before = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    let first = draft
                        .shift(&[Seg::key("items")])
                        .map_err(|e| e.to_string())?;
                    draft
                        .unshift(&[Seg::key("items")], vec![first.expect("shifted item")])
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    flush().await;
    let writes = harness.storage.commits().pop().expect("writes");
    assert!(matches!(
        &writes[0],
        StorageWrite::DocumentChange {
            content: crate::types::DocumentContent::Delta { .. },
            ..
        }
    ));
    let after = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(*after, *before);
    let publication = harness
        .publications
        .lock()
        .last()
        .expect("publication")
        .clone();
    assert!(!as_document(document_changes(&publication)[0]).4.is_empty());
}

/// Upstream `revokes escaped drafts when the callback settles`.
#[tokio::test]
async fn revokes_escaped_drafts_when_the_callback_settles() {
    let (harness, conversation_id, token) = setup_live().await;
    let escaped: Arc<Mutex<Option<crate::session::transaction::DocDraft>>> = Arc::default();
    {
        let escaped = escaped.clone();
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[Seg::key("message")], json!("inside"))
                        .map_err(|e| e.to_string())?;
                    *escaped.lock() = Some(draft.clone());
                    Ok(())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    let escaped = escaped.lock().take().expect("escaped draft");
    assert_eq!(
        escaped.get(&[Seg::key("message")]).unwrap_err().to_string(),
        "Cannot use a settled overlay"
    );
    assert_eq!(
        escaped
            .set(&[Seg::key("message")], json!("outside"))
            .unwrap_err()
            .to_string(),
        "Cannot use a settled overlay"
    );
    assert_eq!(
        escaped
            .length(&[Seg::key("items")])
            .unwrap_err()
            .to_string(),
        "Cannot use a settled overlay"
    );
    assert_eq!(
        escaped
            .push(&[Seg::key("items")], vec![json!("outside")])
            .unwrap_err()
            .to_string(),
        "Cannot use a settled overlay"
    );
    let snapshot = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(snapshot["message"], json!("inside"));

    let returned: crate::session::transaction::DocDraft = harness
        .session
        .commit(
            {
                let token = token.clone();
                move |tx| {
                    tx.doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())
                }
            },
            &context(),
        )
        .await
        .expect("commit");
    assert_eq!(
        returned
            .get(&[Seg::key("message")])
            .unwrap_err()
            .to_string(),
        "Cannot use a settled overlay"
    );
}

/// Upstream `aborts every change when the callback fails`.
#[tokio::test]
async fn aborts_every_change_when_the_callback_fails() {
    let (harness, conversation_id, token) = setup_live().await;
    let counter = counter_doc();
    let counter_for_tx = counter.clone();
    let before = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    let commits = harness.storage.commits().len();
    let error = {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[Seg::key("message")], json!("lost"))
                        .map_err(|e| e.to_string())?;
                    let counter_draft = tx
                        .doc(&counter_for_tx, None, None, None)
                        .map_err(|e| e.to_string())?;
                    counter_draft
                        .set(&[Seg::key("count")], json!(5))
                        .map_err(|e| e.to_string())?;
                    Err("callback failed".to_string())
                },
                &context(),
            )
            .await
            .expect_err("callback failed")
    };
    assert_eq!(error.to_string(), "callback failed");
    assert_eq!(harness.storage.commits().len(), commits);
    assert_eq!(
        *harness
            .session
            .snapshot(&token, Some(conversation_id.0), None, &context())
            .await
            .expect("snapshot")
            .expect("created"),
        *before
    );
    assert!(harness
        .session
        .snapshot(&counter, None, None, &context())
        .await
        .unwrap()
        .is_none());
    set_message(&harness.session, &token, conversation_id, "kept").await;
    let snapshot = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(snapshot["message"], json!("kept"));
}

/// Upstream `memoizes concurrent duplicate acquisition and initializes once`
/// (handle identity is unobservable over cloned drafts; init count and single
/// create carry the case — D).
#[tokio::test]
async fn memoizes_duplicate_acquisition_and_initializes_once() {
    let init_count = Arc::new(AtomicUsize::new(0));
    let harness = open_test_session();
    let conversation_id = create_conversation(&harness.session).await;
    let token: DocToken = {
        let init_count = init_count.clone();
        DocDefinition::conversation(
            "test.live2",
            1,
            DocumentHistory::Latest,
            DocumentFork::Initial,
            move || {
                init_count.fetch_add(1, Ordering::SeqCst);
                obj(json!({ "items": [], "nested": { "count": 0 }, "other": { "label": "x" } }))
            },
        )
        .into()
    };
    let mints = harness.storage.mint_count();
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let first = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    let second = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    second
                        .set(&[Seg::key("message")], json!("once"))
                        .map_err(|e| e.to_string())?;
                    drop(first);
                    Ok(())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    assert_eq!(init_count.load(Ordering::SeqCst), 1);
    assert_eq!(harness.storage.mint_count(), mints + 1);
    let writes = harness.storage.commits().pop().expect("writes");
    assert_eq!(
        writes
            .iter()
            .filter(|write| matches!(write, StorageWrite::DocumentCreate { .. }))
            .count(),
        1
    );
}

/// Upstream `uses the first family seed and ignores seeds for existing members`.
#[tokio::test]
async fn uses_the_first_family_seed_and_ignores_seeds_for_existing_members() {
    let harness = open_test_session();
    let member = member_doc();
    {
        let member = member.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let first = tx
                        .doc(&member, None, Some("k"), Some(json!("first")))
                        .map_err(|e| e.to_string())?;
                    let second = tx
                        .doc(&member, None, Some("k"), Some(json!("second")))
                        .map_err(|e| e.to_string())?;
                    second
                        .set(&[Seg::key("hits")], json!(1))
                        .map_err(|e| e.to_string())?;
                    drop(first);
                    Ok(())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    {
        let member = member.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let third = tx
                        .doc(&member, None, Some("k"), Some(json!("third")))
                        .map_err(|e| e.to_string())?;
                    third
                        .set(&[Seg::key("hits")], json!(2))
                        .map_err(|e| e.to_string())?;
                    let other = tx
                        .doc(&member, None, Some("other"), Some(json!("fourth")))
                        .map_err(|e| e.to_string())?;
                    other
                        .set(&[Seg::key("hits")], json!(1))
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    let k = harness
        .session
        .snapshot(&member, None, Some("k"), &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(*k, obj(json!({ "seed": "first", "hits": 2 })));
    let other = harness
        .session
        .snapshot(&member, None, Some("other"), &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(*other, obj(json!({ "seed": "fourth", "hits": 1 })));
}

/// Upstream `rejects Tx use after the callback settles` — the escrowable
/// object in Rust is the `DocDraft` handle (D); settled rejections cover the
/// post-settle use case.
#[tokio::test]
async fn rejects_use_after_the_callback_settles() {
    let (harness, conversation_id, token) = setup_live().await;
    let escaped: Arc<Mutex<Option<crate::session::transaction::DocDraft>>> = Arc::default();
    {
        let escaped = escaped.clone();
        let token = token.clone();
        harness
            .session
            .commit_with(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    *escaped.lock() = Some(draft);
                    Ok(())
                },
                &context(),
                Default::default(),
            )
            .await
            .expect("commit");
    }
    let escaped = escaped.lock().take().expect("escaped draft");
    assert!(escaped.get(&[Seg::key("message")]).is_err());
    assert!(escaped.set(&[Seg::key("message")], json!("x")).is_err());
}

/// Upstream `rejects tokens whose semantics or version disagree with the
/// stored incarnation`.
#[tokio::test]
async fn rejects_tokens_whose_semantics_or_version_disagree_with_the_stored_incarnation() {
    let (harness, conversation_id, _token) = setup_live().await;
    let error = harness
        .session
        .snapshot(
            &rewindable_live_doc(),
            Some(conversation_id.0),
            None,
            &context(),
        )
        .await
        .expect_err("semantics rejected");
    assert!(
        error
            .to_string()
            .contains("does not match the supplied definition semantics"),
        "{error}"
    );
    let rewindable = rewindable_live_doc();
    let error = {
        harness
            .session
            .commit(
                move |tx| {
                    tx.doc(&rewindable, Some(conversation_id.0), None, None)
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect_err("semantics rejected")
    };
    assert!(
        error
            .to_string()
            .contains("does not match the supplied definition semantics"),
        "{error}"
    );
    let v2: DocToken = DocDefinition::conversation(
        "test.live",
        2,
        DocumentHistory::Latest,
        DocumentFork::Initial,
        || obj(json!({ "items": [], "nested": { "count": 0 }, "other": { "label": "x" } })),
    )
    .into();
    let error = {
        harness
            .session
            .commit(
                move |tx| {
                    tx.doc(&v2, Some(conversation_id.0), None, None)
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect_err("version rejected")
    };
    assert!(
        error
            .to_string()
            .contains("requires migration from version 1"),
        "{error}"
    );
}

/// Upstream `rolls back prepared documents when batch assembly fails`.
#[tokio::test]
async fn rolls_back_prepared_documents_when_batch_assembly_fails() {
    let (harness, conversation_id, token) = setup_live().await;
    let counter = counter_doc();
    {
        let counter = counter.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&counter, None, None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[Seg::key("count")], json!(1))
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    let live_before = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    let counter_before = harness
        .session
        .snapshot(&counter, None, None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    let commits = harness.storage.commits().len();
    let error = {
        let token = token.clone();
        let counter = counter.clone();
        harness
            .session
            .commit_with(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[Seg::key("message")], json!("lost"))
                        .map_err(|e| e.to_string())?;
                    let counter_draft = tx
                        .doc(&counter, None, None, None)
                        .map_err(|e| e.to_string())?;
                    counter_draft
                        .set(&[Seg::key("count")], json!(2))
                        .map_err(|e| e.to_string())?;
                    // Replacing a missing task fails during assembly.
                    tx.set_task(crate::types::TaskRecord {
                        id: crate::ids::id_from_number(999),
                        conversation_id,
                        kind: "missing".into(),
                        version: 1,
                        input: Value::Null,
                        owner: None,
                        background: false,
                        abort_requested: false,
                        state: crate::types::TaskState {
                            status: crate::types::TaskStateStatus::Pending,
                            checkpoint: Some(json!({ "phase": "start" })),
                            on: None,
                            policy: None,
                            outcome: None,
                        },
                        started_at: None,
                        ended_at: None,
                        memos: None,
                    })
                    .map_err(|e| e.to_string())
                },
                &context(),
                Default::default(),
            )
            .await
            .expect_err("assembly failed")
    };
    assert!(
        error.to_string().contains("Task 999 does not exist"),
        "{error}"
    );
    assert_eq!(harness.storage.commits().len(), commits);
    assert_eq!(
        *harness
            .session
            .snapshot(&token, Some(conversation_id.0), None, &context())
            .await
            .expect("snapshot")
            .expect("created"),
        *live_before
    );
    assert_eq!(
        *harness
            .session
            .snapshot(&counter, None, None, &context())
            .await
            .expect("snapshot")
            .expect("created"),
        *counter_before
    );
    set_message(&harness.session, &token, conversation_id, "next").await;
    {
        let counter = counter.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&counter, None, None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[Seg::key("count")], json!(3))
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    let counter_after = harness
        .session
        .snapshot(&counter, None, None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(counter_after["count"], json!(3));
}

/// Upstream `poisons the Session after an uncertain Storage failure and
/// publishes nothing` — adapted to session D4/D5: a `StorageRejected` failure
/// is certain (no effect committed), so the Session is NOT poisoned.
#[tokio::test]
async fn keeps_the_session_usable_after_a_storage_rejection_and_publishes_nothing() {
    let (harness, conversation_id, token) = setup_live().await;
    flush().await;
    let before = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    let published = harness.publications.lock().len();
    harness.storage.fail_next_commit("disk vanished");
    let error = {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[Seg::key("message")], json!("uncertain"))
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect_err("storage rejected")
    };
    assert_eq!(error.to_string(), "disk vanished");
    flush().await;
    assert_eq!(harness.publications.lock().len(), published);
    assert!(before.get("message").is_none());
    // D4/D5: the rejection was certain, so the session remains usable.
    let snapshot = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("session remains usable");
    assert!(snapshot.is_some());
    harness
        .session
        .commit(|_| Ok(()), &context())
        .await
        .expect("session remains usable");
    harness.session.close(&context()).await.expect("close");
}

/// Upstream `keeps the previous revision unchanged through Storage settlement`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keeps_the_previous_revision_unchanged_through_storage_settlement() {
    let (harness, conversation_id, token) = setup_live().await;
    let before = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    let copy = before.clone();
    let mut gate = harness.storage.hold_commits();
    let ctx = context();
    let mut commit = {
        let session = harness.session.clone();
        let token = token.clone();
        let commit_context = ctx.clone();
        tokio::spawn(async move {
            session
                .commit(
                    move |tx| -> Result<(), String> {
                        let draft = tx
                            .doc(&token, Some(conversation_id.0), None, None)
                            .map_err(|e| e.to_string())?;
                        draft
                            .push(&[Seg::key("items")], vec![json!("c")])
                            .map(|_| ())
                            .map_err(|e| e.to_string())?;
                        draft
                            .set(&[Seg::key("nested"), Seg::key("count")], json!(9))
                            .map_err(|e| e.to_string())
                    },
                    &commit_context,
                )
                .await
        })
    };
    tokio::select! {
        biased;
        result = &mut commit => panic!("held commit unexpectedly settled: {result:?}"),
        _ = gate.entered() => {}
    }
    let settled = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &ctx)
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(*settled, *before);
    assert_eq!(*before, *copy);
    gate.release();
    commit.await.expect("commit task").expect("commit");
    let after = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &ctx)
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(after["items"], json!(["a", "b", "c"]));
    assert_eq!(*before, *copy);
}

/// Upstream `retires documents and creates a new incarnation at the same address`.
#[tokio::test]
async fn retires_documents_and_creates_a_new_incarnation_at_the_same_address() {
    let (harness, conversation_id, token) = setup_live().await;
    flush().await;
    let old_id = as_document(
        document_changes(
            &harness
                .publications
                .lock()
                .last()
                .expect("publication")
                .clone(),
        )[0],
    )
    .0
    .id;
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[Seg::key("message")], json!("final"))
                        .map_err(|e| e.to_string())?;
                    tx.retire_doc(&token, Some(conversation_id.0), None)
                        .map_err(|e| e.to_string())?;
                    let replacement = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    replacement
                        .set(&[Seg::key("message")], json!("new"))
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    let writes = harness.storage.commits().pop().expect("writes");
    assert_eq!(writes.len(), 3);
    assert!(writes.iter().any(|write| matches!(
		write,
		StorageWrite::DocumentChange { id, content: crate::types::DocumentContent::Delta { .. } } if *id == old_id
	)));
    assert!(writes
        .iter()
        .any(|write| matches!(write, StorageWrite::DocumentRetire { id } if *id == old_id)));
    assert!(writes
		.iter()
		.any(|write| matches!(write, StorageWrite::DocumentCreate { record, .. } if record.kind == "test.live")));
    flush().await;
    let publication = harness
        .publications
        .lock()
        .last()
        .expect("publication")
        .clone();
    let changes = document_changes(&publication);
    assert_eq!(changes.len(), 2);
    let (retired, created) = (as_document(changes[0]), as_document(changes[1]));
    assert_eq!(retired.0.id, old_id);
    assert_eq!(retired.3, &None);
    assert!(retired.4.is_empty());
    assert_eq!(retired.0.retired_at, Some(publication.seq));
    assert_ne!(created.0.id, old_id);
    assert_eq!(created.0.created_at, publication.seq);
    assert_eq!(created.4, &Vec::new());
    assert_eq!(created.3.as_ref().unwrap()["message"], json!("new"));
    let snapshot = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(snapshot["message"], json!("new"));

    harness
        .session
        .commit(
            {
                let token = token.clone();
                move |tx| {
                    tx.retire_doc(&token, Some(conversation_id.0), None)
                        .map_err(|e| e.to_string())
                }
            },
            &context(),
        )
        .await
        .expect("retire");
    assert!(harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .unwrap()
        .is_none());
    harness.session.unload_documents().await;
    assert!(harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .unwrap()
        .is_none());
    // Retiring an absent address is a no-op.
    let commits = harness.storage.commits().len();
    harness
        .session
        .commit(
            {
                let token = token.clone();
                move |tx| {
                    tx.retire_doc(&token, Some(conversation_id.0), None)
                        .map_err(|e| e.to_string())
                }
            },
            &context(),
        )
        .await
        .expect("retire");
    assert_eq!(harness.storage.commits().len(), commits);
}

/// Upstream `retires without acquisition and recreates both existing and
/// absent addresses`.
#[tokio::test]
async fn retires_without_acquisition_and_recreates_both_existing_and_absent_addresses() {
    let (harness, conversation_id, token) = setup_live().await;
    flush().await;
    let old_id = as_document(
        document_changes(
            &harness
                .publications
                .lock()
                .last()
                .expect("publication")
                .clone(),
        )[0],
    )
    .0
    .id;
    harness.session.unload_documents().await;
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    tx.retire_doc(&token, Some(conversation_id.0), None)
                        .map_err(|e| e.to_string())?;
                    let replacement = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    replacement
                        .set(&[Seg::key("message")], json!("replacement"))
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    let existing_writes = harness.storage.commits().pop().expect("writes");
    assert_eq!(existing_writes.len(), 2);
    assert!(existing_writes
        .iter()
        .any(|write| matches!(write, StorageWrite::DocumentRetire { id } if *id == old_id)));
    assert!(existing_writes
		.iter()
		.any(|write| matches!(write, StorageWrite::DocumentCreate { record, .. } if record.kind == "test.live")));

    let member = member_doc();
    {
        let member = member.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    tx.retire_doc(&member, None, Some("absent"))
                        .map_err(|e| e.to_string())?;
                    let replacement = tx
                        .doc(&member, None, Some("absent"), Some(json!("seed")))
                        .map_err(|e| e.to_string())?;
                    replacement
                        .set(&[Seg::key("hits")], json!(1))
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    let absent_writes = harness.storage.commits().pop().expect("writes");
    assert_eq!(
        absent_writes
            .iter()
            .filter(|write| matches!(write, StorageWrite::DocumentRetire { .. }))
            .count(),
        0
    );
    assert_eq!(
        absent_writes
            .iter()
            .filter(|write| matches!(write, StorageWrite::DocumentCreate { .. }))
            .count(),
        1
    );
    let snapshot = harness
        .session
        .snapshot(&member, None, Some("absent"), &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(*snapshot, obj(json!({ "seed": "seed", "hits": 1 })));
}

/// Upstream `retires the existing incarnation when retirement races a pending
/// acquisition` — `Tx::doc` is synchronous (D), so the deterministic
/// acquire-retire ordering replaces the promise race.
#[tokio::test]
async fn retires_the_existing_incarnation_when_retirement_follows_acquisition() {
    let (harness, conversation_id, token) = setup_live().await;
    flush().await;
    let old_id = as_document(
        document_changes(
            &harness
                .publications
                .lock()
                .last()
                .expect("publication")
                .clone(),
        )[0],
    )
    .0
    .id;
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let acquired = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    tx.retire_doc(&token, Some(conversation_id.0), None)
                        .map_err(|e| e.to_string())?;
                    acquired
                        .set(&[Seg::key("message")], json!("final"))
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    let first_writes = harness.storage.commits().pop().expect("writes");
    assert_eq!(first_writes.len(), 2);
    assert!(first_writes.iter().any(|write| matches!(
		write,
		StorageWrite::DocumentChange { id, content: crate::types::DocumentContent::Delta { .. } } if *id == old_id
	)));
    assert!(first_writes
        .iter()
        .any(|write| matches!(write, StorageWrite::DocumentRetire { id } if *id == old_id)));
    assert!(harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .unwrap()
        .is_none());

    set_message(&harness.session, &token, conversation_id, "second").await;
    flush().await;
    let second_id = as_document(
        document_changes(
            &harness
                .publications
                .lock()
                .last()
                .expect("publication")
                .clone(),
        )[0],
    )
    .0
    .id;
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let old = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    tx.retire_doc(&token, Some(conversation_id.0), None)
                        .map_err(|e| e.to_string())?;
                    let fresh = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    fresh
                        .set(&[Seg::key("message")], json!("third"))
                        .map_err(|e| e.to_string())?;
                    drop(old);
                    Ok(())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    let writes = harness.storage.commits().pop().expect("writes");
    assert_eq!(writes.len(), 2);
    assert!(writes
        .iter()
        .any(|write| matches!(write, StorageWrite::DocumentRetire { id } if *id == second_id)));
    assert!(writes
		.iter()
		.any(|write| matches!(write, StorageWrite::DocumentCreate { record, .. } if record.kind == "test.live")));
    let snapshot = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(snapshot["message"], json!("third"));
}

/// Upstream `reloads an unloaded document from Storage`.
#[tokio::test]
async fn reloads_an_unloaded_document_from_storage() {
    let (harness, conversation_id, token) = setup_live().await;
    let loaded = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    harness.session.unload_documents().await;
    let reloaded = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(*reloaded, *loaded);
    {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<(), String> {
                    let draft = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .push(&[Seg::key("items")], vec![json!("c")])
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                },
                &context(),
            )
            .await
            .expect("commit");
    }
    harness.session.unload_documents().await;
    let snapshot = harness
        .session
        .snapshot(&token, Some(conversation_id.0), None, &context())
        .await
        .expect("snapshot")
        .expect("created");
    assert_eq!(snapshot["items"], json!(["a", "b", "c"]));
}

/// Upstream `delivers complete publications synchronously after adoption`.
#[tokio::test]
async fn delivers_complete_publications_synchronously_after_adoption() {
    let (harness, conversation_id, token) = setup_live().await;
    let published = harness.publications.lock().len();
    let delivered = Arc::new(AtomicUsize::new(0));
    let unsubscribe = {
        let delivered = delivered.clone();
        harness
            .session
            .subscribe_commits(Arc::new(move |_publication, _context| {
                delivered.fetch_add(1, Ordering::SeqCst);
            }))
    };
    let result: &str = {
        let token = token.clone();
        harness
            .session
            .commit(
                move |tx| -> Result<&'static str, String> {
                    let draft = tx
                        .doc(&token, Some(conversation_id.0), None, None)
                        .map_err(|e| e.to_string())?;
                    draft
                        .set(&[Seg::key("message")], json!("m"))
                        .map_err(|e| e.to_string())?;
                    Ok("done")
                },
                &context(),
            )
            .await
            .expect("commit")
    };
    assert_eq!(result, "done");
    assert_eq!(delivered.load(Ordering::SeqCst), 1);
    assert_eq!(harness.publications.lock().len(), published + 1);
    unsubscribe.call();
}

/// Upstream `publishes close synchronously and supports unsubscription`.
#[tokio::test]
async fn publishes_close_synchronously_and_supports_unsubscription() {
    let harness = open_test_session();
    let calls: Arc<Mutex<Vec<&'static str>>> = Arc::default();
    harness.session.subscribe_close({
        let calls = calls.clone();
        Arc::new(move || calls.lock().push("active"))
    });
    let unsubscribe = {
        let calls = calls.clone();
        harness
            .session
            .subscribe_close(Arc::new(move || calls.lock().push("removed")))
    };
    unsubscribe.call();
    harness.session.close(&context()).await.expect("close");
    assert_eq!(*calls.lock(), vec!["active"]);
}

/// Upstream `settles admitted commits before close and rejects later admission`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn settles_admitted_commits_before_close_and_rejects_later_admission() {
    let (harness, conversation_id, token) = setup_live().await;
    let ctx = context();
    let timeout = std::time::Duration::from_secs(5);
    let mut gate = harness.storage.hold_commits();
    let mut commit = {
        let session = harness.session.clone();
        let token = token.clone();
        let commit_context = ctx.clone();
        tokio::spawn(async move {
            session
                .commit(
                    move |tx| -> Result<(), String> {
                        let draft = tx
                            .doc(&token, Some(conversation_id.0), None, None)
                            .map_err(|e| e.to_string())?;
                        draft
                            .set(&[Seg::key("message")], json!("admitted"))
                            .map_err(|e| e.to_string())
                    },
                    &commit_context,
                )
                .await
        })
    };
    tokio::time::timeout(timeout, async {
        tokio::select! {
            biased;
            result = &mut commit => panic!("held commit unexpectedly settled: {result:?}"),
            _ = gate.entered() => {}
        }
    })
    .await
    .expect("commit reaches held storage");

    let queued_session = harness.session.clone();
    let queued_context = ctx.clone();
    let queued_token = token.clone();
    let (queued_tx, queued_rx) = tokio::sync::oneshot::channel();
    let mut queued = tokio::spawn(async move {
        let mut future = Box::pin(queued_session.commit(
            move |tx| -> Result<(), String> {
                let draft = tx
                    .doc(&queued_token, Some(conversation_id.0), None, None)
                    .map_err(|e| e.to_string())?;
                draft
                    .set(&[Seg::key("message")], json!("queued"))
                    .map_err(|e| e.to_string())
            },
            &queued_context,
        ));
        let mut queued_tx = Some(queued_tx);
        std::future::poll_fn(|cx| match std::future::Future::poll(future.as_mut(), cx) {
            std::task::Poll::Pending => {
                if let Some(tx) = queued_tx.take() {
                    let _ = tx.send(());
                }
                std::task::Poll::Ready(())
            }
            std::task::Poll::Ready(result) => {
                panic!("queued commit settled before the held commit: {result:?}")
            }
        })
        .await;
        future.await
    });
    tokio::time::timeout(timeout, queued_rx)
        .await
        .expect("queued commit is admitted")
        .expect("queued signal");

    let member = member_doc();
    let snapshot_session = harness.session.clone();
    let snapshot_context = ctx.clone();
    let (snapshot_tx, snapshot_rx) = tokio::sync::oneshot::channel();
    let mut admitted_snapshot = tokio::spawn(async move {
        let mut future =
            Box::pin(snapshot_session.snapshot(&member, None, Some("absent"), &snapshot_context));
        let mut snapshot_tx = Some(snapshot_tx);
        std::future::poll_fn(|cx| match std::future::Future::poll(future.as_mut(), cx) {
            std::task::Poll::Pending => {
                if let Some(tx) = snapshot_tx.take() {
                    let _ = tx.send(());
                }
                std::task::Poll::Ready(())
            }
            std::task::Poll::Ready(result) => {
                panic!("snapshot settled before the held commit: {result:?}")
            }
        })
        .await;
        future.await
    });
    tokio::time::timeout(timeout, snapshot_rx)
        .await
        .expect("snapshot is admitted")
        .expect("snapshot signal");

    let close_session = harness.session.clone();
    let close_context = ctx.clone();
    let (close_tx, close_rx) = tokio::sync::oneshot::channel();
    let mut close = tokio::spawn(async move {
        let mut future = Box::pin(close_session.close(&close_context));
        let mut close_tx = Some(close_tx);
        std::future::poll_fn(|cx| match std::future::Future::poll(future.as_mut(), cx) {
            std::task::Poll::Pending => {
                if let Some(tx) = close_tx.take() {
                    let _ = tx.send(());
                }
                std::task::Poll::Ready(())
            }
            std::task::Poll::Ready(result) => {
                panic!("close settled while admitted work was held: {result:?}")
            }
        })
        .await;
        future.await
    });
    tokio::time::timeout(timeout, close_rx)
        .await
        .expect("close seals admission")
        .expect("close signal");

    let error = tokio::time::timeout(timeout, harness.session.commit(|_| Ok(()), &ctx))
        .await
        .expect("closed commit rejects")
        .expect_err("closed");
    assert_eq!(error.to_string(), "Session is closed");
    let error = tokio::time::timeout(
        timeout,
        harness
            .session
            .snapshot(&token, Some(conversation_id.0), None, &ctx),
    )
    .await
    .expect("closed snapshot rejects")
    .expect_err("closed");
    assert_eq!(error.to_string(), "Session is closed");

    gate.release();
    tokio::time::timeout(timeout, &mut commit)
        .await
        .expect("admitted commit settles")
        .expect("commit task")
        .expect("admitted commit");
    tokio::time::timeout(timeout, &mut queued)
        .await
        .expect("queued commit settles")
        .expect("queued task")
        .expect("queued commit");
    assert!(tokio::time::timeout(timeout, &mut admitted_snapshot)
        .await
        .expect("admitted snapshot settles")
        .expect("snapshot task")
        .expect("snapshot")
        .is_none());
    tokio::time::timeout(timeout, &mut close)
        .await
        .expect("close settles")
        .expect("close task")
        .expect("close");
    assert!(harness.storage.is_closed(), "storage close reached");
}

// Keep the table-change helper referenced even where unused (publication
// table assertions live in the tables suite).
#[allow(dead_code)]
fn unused(change: &CommitChange) -> Option<&TableCommitChange> {
    match change {
        CommitChange::Table(table) => Some(table),
        CommitChange::Document(_) => None,
    }
}
