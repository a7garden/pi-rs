//! Port of `packages/durable/test/session-checkpoints-migrations.test.ts`
//! (checkpoint selection, migrations, historical snapshots, tracker cache
//! across definition versions).
//!
//! D-deviations: identity assertions (`toBe`) collapse to structural
//! equality; predicate throwing is signaled by panicking (the Rust
//! `checkpoint_when` cannot return an error), surfaced as the panic message.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use pi_chord::context::Context;
use pi_chord::delta::Seg;
use serde_json::{json, Value};

use crate::documents::{DocDefinition, DocToken, ScopeKind};
use crate::session::support::{context, create_conversation, document_changes, flush, open_test_session};
use crate::types::Storage as _;
use crate::types::{
	ConversationId, DocumentContent, DocumentFork, DocumentHistory, DocumentScope, EntryDraft, EntryId, JsonObject,
	StorageWrite,
};

fn obj(value: Value) -> JsonObject {
	value.as_object().expect("object literal").clone()
}

fn document_writes(writes: &[StorageWrite]) -> Vec<&StorageWrite> {
	writes
		.iter()
		.filter(|write| {
			matches!(
				write,
				StorageWrite::DocumentCreate { .. } | StorageWrite::DocumentChange { .. } | StorageWrite::DocumentRetire { .. }
			)
		})
		.collect()
}

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
		crate::types::DocumentCommitChange::Document { record, conversation_id, version, value, ops } => {
			(record, *conversation_id, *version, value, ops)
		}
		crate::types::DocumentCommitChange::DocumentCopy { .. } => panic!("expected a document change"),
	}
}

fn doc(kind: &str, version: u64, initial: JsonObject) -> DocToken {
	DocDefinition::singleton(kind, version, ScopeKind::Session, move || initial.clone()).into()
}

async fn set_count(session: &crate::session::session::Session, token: &DocToken, count: i64) {
	let token = token.clone();
	session
		.commit(
			move |tx| -> Result<(), String> {
				let draft = tx.doc(&token, None, None, None).map_err(|e| e.to_string())?;
				draft.set(&[Seg::key("count")], json!(count)).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
}

async fn snapshot(session: &crate::session::session::Session, token: &DocToken) -> Option<JsonObject> {
	session
		.snapshot(token, None, None, &context())
		.await
		.expect("snapshot")
		.map(|value| (*value).clone())
}

/// Upstream `calls checkpointWhen with its definition as the receiver`
/// (`this.initial()` collapses to the captured initial value).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn calls_checkpoint_when_against_the_definition_initial() {
	let token: DocToken = DocDefinition::singleton("checkpoint.receiver", 1, ScopeKind::Session, || {
		obj(json!({ "count": 0 }))
	})
	.with_checkpoint_when(|value, _ops, _info| value["count"] == json!(2))
	.into();
	let harness = open_test_session();
	for count in 0..=2 {
		set_count(&harness.session, &token, count).await;
	}
	let kinds: Vec<String> = harness
		.storage
		.commits()
		.iter()
		.flat_map(|writes| writes.iter())
		.filter_map(|write| match write {
			StorageWrite::DocumentChange { content, .. } => Some(
				match content {
					DocumentContent::Base { .. } => "base",
					DocumentContent::Delta { .. } => "delta",
				}
				.to_string(),
			),
			_ => None,
		})
		.collect();
	assert_eq!(kinds, vec!["delta".to_string(), "base".to_string()]);
}

/// Upstream `selects bases only for nonempty ordinary batches and passes the
/// exact prepared revision and ops` (identity assertions become equality — D).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn selects_bases_only_for_nonempty_ordinary_batches() {
	let base_calls: Arc<Mutex<Vec<(Value, Vec<pi_chord::delta::Op>)>>> = Arc::default();
	let delta_calls: Arc<Mutex<Vec<(Value, Vec<pi_chord::delta::Op>)>>> = Arc::default();
	let base: DocToken = {
		let calls = base_calls.clone();
		DocDefinition::singleton("checkpoint.base", 1, ScopeKind::Session, || obj(json!({ "items": [] })))
			.with_checkpoint_when(move |value, ops, _info| {
				calls.lock().push((Value::Object(value.clone()), ops.to_vec()));
				true
			})
			.into()
	};
	let delta: DocToken = {
		let calls = delta_calls.clone();
		DocDefinition::singleton("checkpoint.delta", 1, ScopeKind::Session, || obj(json!({ "count": 0 })))
			.with_checkpoint_when(move |value, ops, _info| {
				calls.lock().push((Value::Object(value.clone()), ops.to_vec()));
				false
			})
			.into()
	};
	let default = doc("checkpoint.default", 1, obj(json!({ "count": 0 })));
	let harness = open_test_session();

	{
		let base = base.clone();
		let delta = delta.clone();
		let default = default.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					tx.doc(&base, None, None, None).map(|_| ()).map_err(|e| e.to_string())?;
					tx.doc(&delta, None, None, None).map(|_| ()).map_err(|e| e.to_string())?;
					tx.doc(&default, None, None, None).map(|_| ()).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	assert!(base_calls.lock().is_empty());
	assert!(delta_calls.lock().is_empty());
	let committed = harness.storage.commits().pop().expect("writes");
	let writes = document_writes(&committed);
	for write in writes {
		match write {
			StorageWrite::DocumentCreate { content, .. } => assert_eq!(content.version, 1),
			other => panic!("expected create, got {other:?}"),
		}
	}

	{
		let base = base.clone();
		let delta = delta.clone();
		let default = default.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let base_draft = tx.doc(&base, None, None, None).map_err(|e| e.to_string())?;
					base_draft.push(&[Seg::key("items")], vec![json!("x")]).map(|_| ()).map_err(|e| e.to_string())?;
					let delta_draft = tx.doc(&delta, None, None, None).map_err(|e| e.to_string())?;
					let count = delta_draft.get(&[Seg::key("count")]).map_err(|e| e.to_string())?;
					delta_draft
						.set(&[Seg::key("count")], json!(count.unwrap_or(json!(0)).as_i64().unwrap_or(0) + 1))
						.map_err(|e| e.to_string())?;
					let default_draft = tx.doc(&default, None, None, None).map_err(|e| e.to_string())?;
					let count = default_draft.get(&[Seg::key("count")]).map_err(|e| e.to_string())?;
					default_draft
						.set(&[Seg::key("count")], json!(count.unwrap_or(json!(0)).as_i64().unwrap_or(0) + 1))
						.map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	flush().await;
	let admitted = harness.storage.admitted_commits().pop().expect("admitted");
	let change_kinds: Vec<String> = admitted
		.iter()
		.filter_map(|write| match write {
			StorageWrite::DocumentChange { content, .. } => Some(
				match content {
					DocumentContent::Base { .. } => "base",
					DocumentContent::Delta { .. } => "delta",
				}
				.to_string(),
			),
			_ => None,
		})
		.collect();
	assert_eq!(change_kinds, vec!["base".to_string(), "delta".to_string(), "delta".to_string()]);
	assert_eq!(base_calls.lock().len(), 1);
	assert_eq!(delta_calls.lock().len(), 1);
	let base_snapshot = snapshot(&harness.session, &base).await.expect("created");
	let delta_snapshot = snapshot(&harness.session, &delta).await.expect("created");
	let publication = harness.publications.lock().last().expect("publication").clone();
	let changes = document_changes(&publication);
	assert_eq!(changes.len(), 3);
	let published_ops: Vec<Vec<pi_chord::delta::Op>> =
		changes.iter().map(|change| as_document(change).4.clone()).collect();
	assert_eq!(base_calls.lock()[0].0, Value::Object(base_snapshot.clone()));
	assert_eq!(base_calls.lock()[0].1, published_ops.iter().find(|ops| !ops.is_empty()).cloned().unwrap_or_default());
	assert_eq!(delta_calls.lock()[0].0, Value::Object(delta_snapshot.clone()));
}

/// Upstream `passes the stored delta count since the newest base, including
/// after unload and version bases`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn passes_the_stored_delta_count_since_the_newest_base() {
	let seen: Arc<Mutex<Vec<u64>>> = Arc::default();
	let v1: DocToken = {
		let seen = seen.clone();
		DocDefinition::singleton("checkpoint.deltas-since-base", 1, ScopeKind::Session, || {
			obj(json!({ "count": 0 }))
		})
		.with_checkpoint_when(move |_value, _ops, info| {
			seen.lock().push(info.deltas_since_base);
			info.deltas_since_base >= 2
		})
		.into()
	};
	let v2: DocToken = {
		let seen = seen.clone();
		DocDefinition::singleton("checkpoint.deltas-since-base", 2, ScopeKind::Session, || {
			obj(json!({ "count": 0 }))
		})
		.with_migrate(|value, _from| obj(json!({ "count": value["count"] })))
		.with_checkpoint_when(move |_value, _ops, info| {
			seen.lock().push(info.deltas_since_base);
			false
		})
		.into()
	};
	let harness = open_test_session();
	harness
		.session
		.commit(
			{
				let v1 = v1.clone();
				move |tx| tx.doc(&v1, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	for _ in 0..3 {
		set_count(&harness.session, &v1, 1).await;
	}
	harness.session.unload_documents().await;
	set_count(&harness.session, &v1, 1).await;
	assert_eq!(*seen.lock(), vec![0, 1, 2, 0]);
	let kinds: Vec<String> = harness
		.storage
		.commits()
		.iter()
		.rev()
		.take(4)
		.rev()
		.map(|writes| match &writes[0] {
			StorageWrite::DocumentChange { content, .. } => match content {
				DocumentContent::Base { .. } => "base".to_string(),
				DocumentContent::Delta { .. } => "delta".to_string(),
			},
			other => format!("{other:?}"),
		})
		.collect();
	assert_eq!(kinds, vec!["delta", "delta", "base", "delta"]);

	// A required version base resets the count without calling the predicate.
	let before = seen.lock().len();
	set_count(&harness.session, &v2, 1).await;
	set_count(&harness.session, &v2, 1).await;
	assert_eq!(*seen.lock(), vec![0, 1, 2, 0, 0]);
	assert_eq!(seen.lock().len() - before, 1);
	let writes = harness.storage.commits();
	let last = &writes[writes.len() - 1];
	match &last[0] {
		StorageWrite::DocumentChange { content: DocumentContent::Base { version, .. }, .. } => {
			assert_eq!(*version, 2)
		}
		other => panic!("expected version base, got {other:?}"),
	}
}

/// Upstream `skips the predicate for empty batches but calls it for nonempty
/// structural no-ops`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn skips_the_predicate_for_empty_batches_but_calls_it_for_nonempty_structural_no_ops() {
	let calls = Arc::new(AtomicUsize::new(0));
	let token: DocToken = {
		let calls = calls.clone();
		DocDefinition::singleton("checkpoint.no-op", 1, ScopeKind::Session, || obj(json!({ "items": ["a", "b"] })))
			.with_checkpoint_when(move |_value, _ops, _info| {
				calls.fetch_add(1, Ordering::SeqCst);
				false
			})
			.into()
	};
	let harness = open_test_session();
	harness
		.session
		.commit(
			{
				let token = token.clone();
				move |tx| tx.doc(&token, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	let commits = harness.storage.commits().len();

	{
		let token = token.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let draft = tx.doc(&token, None, None, None).map_err(|e| e.to_string())?;
					draft.push(&[Seg::key("items")], vec![json!("x")]).map(|_| ()).map_err(|e| e.to_string())?;
					draft.pop(&[Seg::key("items")]).map(|_| ()).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	assert_eq!(calls.load(Ordering::SeqCst), 0);
	assert_eq!(harness.storage.commits().len(), commits);

	{
		let token = token.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let draft = tx.doc(&token, None, None, None).map_err(|e| e.to_string())?;
					let first = draft.shift(&[Seg::key("items")]).map_err(|e| e.to_string())?;
					draft.unshift(&[Seg::key("items")], vec![first.expect("shifted")])
						.map(|_| ())
						.map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	assert_eq!(calls.load(Ordering::SeqCst), 1);
	let writes = harness.storage.commits().pop().expect("writes");
	match &writes[0] {
		StorageWrite::DocumentChange { content: DocumentContent::Delta { .. }, .. } => {}
		other => panic!("expected delta, got {other:?}"),
	}
}

/// Upstream `rolls back every prepared document when a checkpoint predicate throws`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolls_back_every_prepared_document_when_a_checkpoint_predicate_throws() {
	let throw_checkpoint = Arc::new(AtomicBool::new(true));
	let first_calls = Arc::new(AtomicUsize::new(0));
	let first: DocToken = {
		let first_calls = first_calls.clone();
		DocDefinition::singleton("checkpoint.rollback.first", 1, ScopeKind::Session, || obj(json!({ "count": 0 })))
			.with_checkpoint_when(move |_value, _ops, _info| {
				first_calls.fetch_add(1, Ordering::SeqCst);
				false
			})
			.into()
	};
	let second: DocToken = {
		let throw_checkpoint = throw_checkpoint.clone();
		DocDefinition::singleton("checkpoint.rollback.second", 1, ScopeKind::Session, || obj(json!({ "count": 0 })))
			.with_checkpoint_when(move |_value, _ops, _info| {
				if throw_checkpoint.load(Ordering::SeqCst) {
					panic!("checkpoint failed");
				}
				false
			})
			.into()
	};
	let harness = open_test_session();
	{
		let first = first.clone();
		let second = second.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					tx.doc(&first, None, None, None).map(|_| ()).map_err(|e| e.to_string())?;
					tx.doc(&second, None, None, None).map(|_| ()).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	flush().await;
	let first_before = snapshot(&harness.session, &first).await.expect("created");
	let second_before = snapshot(&harness.session, &second).await.expect("created");
	let commits = harness.storage.commits().len();
	let published = harness.publications.lock().len();

	let error = {
		let first = first.clone();
		let second = second.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let first_draft = tx.doc(&first, None, None, None).map_err(|e| e.to_string())?;
					first_draft.set(&[Seg::key("count")], json!(1)).map_err(|e| e.to_string())?;
					let second_draft = tx.doc(&second, None, None, None).map_err(|e| e.to_string())?;
					second_draft.set(&[Seg::key("count")], json!(2)).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect_err("checkpoint failed")
	};
	assert_eq!(error.to_string(), "checkpoint failed");
	flush().await;
	assert_eq!(first_calls.load(Ordering::SeqCst), 1);
	assert_eq!(harness.storage.commits().len(), commits);
	assert_eq!(harness.publications.lock().len(), published);
	assert_eq!(snapshot(&harness.session, &first).await, Some(first_before));
	assert_eq!(snapshot(&harness.session, &second).await, Some(second_before));

	throw_checkpoint.store(false, Ordering::SeqCst);
	{
		let first = first.clone();
		let second = second.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let first_draft = tx.doc(&first, None, None, None).map_err(|e| e.to_string())?;
					first_draft.set(&[Seg::key("count")], json!(3)).map_err(|e| e.to_string())?;
					let second_draft = tx.doc(&second, None, None, None).map_err(|e| e.to_string())?;
					second_draft.set(&[Seg::key("count")], json!(4)).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	assert_eq!(snapshot(&harness.session, &first).await, Some(obj(json!({ "count": 3 }))));
	assert_eq!(snapshot(&harness.session, &second).await, Some(obj(json!({ "count": 4 }))));
}

/// Upstream `persists repeated false decisions as deltas and replays the complete tail`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn persists_repeated_false_decisions_as_deltas_and_replays_the_complete_tail() {
	let calls = Arc::new(AtomicUsize::new(0));
	let token: DocToken = {
		let calls = calls.clone();
		DocDefinition::singleton("checkpoint.tail", 1, ScopeKind::Session, || obj(json!({ "values": [] })))
			.with_checkpoint_when(move |_value, _ops, _info| {
				calls.fetch_add(1, Ordering::SeqCst);
				false
			})
			.into()
	};
	let harness = open_test_session();
	harness
		.session
		.commit(
			{
				let token = token.clone();
				move |tx| tx.doc(&token, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	for value in 1..=8 {
		let token = token.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let draft = tx.doc(&token, None, None, None).map_err(|e| e.to_string())?;
					draft.push(&[Seg::key("values")], vec![json!(value)]).map(|_| ()).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	let all_commits = harness.storage.commits();
	let changes: Vec<&StorageWrite> = all_commits
		.iter()
		.flat_map(|writes| writes.iter())
		.filter(|write| matches!(write, StorageWrite::DocumentChange { .. }))
		.collect();
	assert_eq!(changes.len(), 8);
	for write in changes {
		match write {
			StorageWrite::DocumentChange { content: DocumentContent::Delta { .. }, .. } => {}
			other => panic!("expected delta, got {other:?}"),
		}
	}
	assert_eq!(calls.load(Ordering::SeqCst), 8);
	harness.session.unload_documents().await;
	assert_eq!(
		snapshot(&harness.session, &token).await,
		Some(obj(json!({ "values": [1, 2, 3, 4, 5, 6, 7, 8] })))
	);
}

/// Upstream `keeps a prepared root replacement as a delta when the predicate is false`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keeps_a_prepared_root_replacement_as_a_delta_when_the_predicate_is_false() {
	let mut initial = serde_json::Map::new();
	for index in 0..4_100 {
		initial.insert(format!("field{index}"), json!(0));
	}
	let token: DocToken = DocDefinition::singleton("checkpoint.root-replacement", 1, ScopeKind::Session, move || {
		initial.clone()
	})
	.with_checkpoint_when(|_value, _ops, _info| false)
	.into();
	let harness = open_test_session();
	harness
		.session
		.commit(
			{
				let token = token.clone();
				move |tx| tx.doc(&token, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	{
		let token = token.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let draft = tx.doc(&token, None, None, None).map_err(|e| e.to_string())?;
					for index in 0..4_100 {
						draft.set(&[Seg::key(format!("field{index}"))], json!(1)).map_err(|e| e.to_string())?;
					}
					Ok(())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	let writes = harness.storage.commits().pop().expect("writes");
	match &writes[0] {
		StorageWrite::DocumentChange { content: DocumentContent::Delta { ops, .. }, .. } => {
			assert_eq!(ops.len(), 1);
			assert!(matches!(ops[0], pi_chord::delta::Op::Replace(_)));
		}
		other => panic!("expected root replacement delta, got {other:?}"),
	}
	harness.session.unload_documents().await;
	let snapshot = snapshot(&harness.session, &token).await.expect("created");
	assert_eq!(snapshot["field4099"], json!(1));
}

/// Upstream `uses ordinary checkpoint selection before retirement`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn uses_ordinary_checkpoint_selection_before_retirement() {
	let calls = Arc::new(AtomicUsize::new(0));
	let token: DocToken = {
		let calls = calls.clone();
		DocDefinition::singleton("checkpoint.retire", 1, ScopeKind::Session, || obj(json!({ "count": 0 })))
			.with_checkpoint_when(move |_value, _ops, _info| {
				calls.fetch_add(1, Ordering::SeqCst);
				true
			})
			.into()
	};
	let harness = open_test_session();
	harness
		.session
		.commit(
			{
				let token = token.clone();
				move |tx| tx.doc(&token, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	{
		let token = token.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let draft = tx.doc(&token, None, None, None).map_err(|e| e.to_string())?;
					draft.set(&[Seg::key("count")], json!(1)).map_err(|e| e.to_string())?;
					tx.retire_doc(&token, None, None).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	assert_eq!(calls.load(Ordering::SeqCst), 1);
	let committed = harness.storage.commits().pop().expect("writes");
	let writes = document_writes(&committed);
	assert_eq!(writes.len(), 2);
	assert!(matches!(
		writes[0],
		StorageWrite::DocumentChange { content: DocumentContent::Base { .. }, .. }
	));
	assert!(matches!(writes[1], StorageWrite::DocumentRetire { .. }));
}

// ─── Migrations ──────────────────────────────────────────────────────────────

/// Upstream `migrates read-only once per cold load, copies the callback
/// result, and writes nothing` (the mutation probes are unrepresentable over
/// immutable snapshots; the copy semantics carry the case — D).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn migrates_read_only_once_per_cold_load() {
	let old = doc("migration.read-only", 1, obj(json!({ "count": 2 })));
	let calls = Arc::new(AtomicUsize::new(0));
	let current: DocToken = {
		let calls = calls.clone();
		DocDefinition::singleton("migration.read-only", 3, ScopeKind::Session, || {
			obj(json!({ "count": 0, "labels": [] }))
		})
		.with_migrate(move |value, from_version| {
			assert_eq!(from_version, 1);
			calls.fetch_add(1, Ordering::SeqCst);
			obj(json!({ "count": value["count"], "labels": ["migrated"] }))
		})
		.into()
	};
	let harness = open_test_session();
	harness
		.session
		.commit(
			{
				let old = old.clone();
				move |tx| tx.doc(&old, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	harness.session.unload_documents().await;
	let commits = harness.storage.commits().len();

	let first = snapshot(&harness.session, &current).await.expect("migrated");
	let second_same = snapshot(&harness.session, &current).await.expect("migrated");
	assert_eq!(first, second_same);
	assert_eq!(calls.load(Ordering::SeqCst), 1);
	assert_eq!(harness.storage.commits().len(), commits);
	assert_eq!(first, obj(json!({ "count": 2, "labels": ["migrated"] })));

	harness.session.unload_documents().await;
	let second = snapshot(&harness.session, &current).await.expect("migrated");
	assert_eq!(second, first);
	assert_eq!(calls.load(Ordering::SeqCst), 2);
	assert_eq!(harness.storage.commits().len(), commits);
}

/// Upstream `writes the required base on the first successful transaction,
/// then writes deltas`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn writes_the_required_base_on_the_first_successful_transaction() {
	let old = doc("migration.transition", 1, obj(json!({ "count": 4 })));
	let checkpoints = Arc::new(AtomicUsize::new(0));
	let current: DocToken = {
		let checkpoints = checkpoints.clone();
		DocDefinition::singleton("migration.transition", 3, ScopeKind::Session, || obj(json!({ "count": 0 })))
			.with_migrate(|value, from_version| {
				let count = value["count"].as_i64().unwrap_or(0);
				obj(json!({ "count": count + from_version as i64 - 1 }))
			})
			.with_checkpoint_when(move |_value, _ops, _info| {
				checkpoints.fetch_add(1, Ordering::SeqCst);
				false
			})
			.into()
	};
	let harness = open_test_session();
	harness
		.session
		.commit(
			{
				let old = old.clone();
				move |tx| tx.doc(&old, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	harness.session.unload_documents().await;
	// An observer of the older shape.
	let watch = harness.session.watch_doc(&old, None, None, &context()).await.unwrap().expect("watch");
	let frames: Arc<Mutex<Vec<(Option<Value>, Vec<pi_chord::delta::Op>)>>> = Arc::default();
	{
		let frames = frames.clone();
		watch
			.start(Arc::new(move |value, ops, _context: Context| {
				let frames = frames.clone();
				let value = value.as_ref().map(|object| Value::Object(object.as_ref().clone()));
				Box::pin(async move {
					frames.lock().push((value, ops));
				})
			}))
			.expect("start");
	}
	let current_snapshot = snapshot(&harness.session, &current).await;
	assert_eq!(current_snapshot, Some(obj(json!({ "count": 4 }))));
	// An observer of the new shape.
	let current_watch = harness.session.watch_doc(&current, None, None, &context()).await.unwrap().expect("watch");
	let current_frames: Arc<Mutex<Vec<Option<Value>>>> = Arc::default();
	{
		let current_frames = current_frames.clone();
		current_watch
			.start(Arc::new(move |value, _ops, _context: Context| {
				let current_frames = current_frames.clone();
				let value = value.as_ref().map(|object| Value::Object(object.as_ref().clone()));
				Box::pin(async move {
					current_frames.lock().push(value);
				})
			}))
			.expect("start");
	}

	harness
		.session
		.commit(
			{
				let current = current.clone();
				move |tx| tx.doc(&current, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	flush().await;
	{
		let writes = harness.storage.commits().pop().expect("writes");
		match &writes[0] {
			StorageWrite::DocumentChange {
				content: DocumentContent::Base { version, value },
				..
			} => {
				assert_eq!(*version, 3);
				assert_eq!(value, &obj(json!({ "count": 4 })));
			}
			other => panic!("expected migration base, got {other:?}"),
		}
	}
	assert_eq!(checkpoints.load(Ordering::SeqCst), 0);
	let publication = harness.publications.lock().last().expect("publication").clone();
	let changes = document_changes(&publication);
	assert_eq!(changes.len(), 1);
	let (_, _, version, value, ops) = as_document(changes[0]);
	assert_eq!(version, Some(3));
	assert_eq!(value, &Some(obj(json!({ "count": 4 }))));
	assert!(ops.is_empty());
	let guard = frames.lock();
	assert_eq!(
		*guard,
		vec![(
			Some(json!({ "count": 4 })),
			vec![pi_chord::delta::Op::Replace(json!({ "count": 4 }))]
		)]
	);
	drop(guard);
	assert_eq!(*current_frames.lock(), Vec::<Option<Value>>::new());

	{
		let snapshot = current_snapshot.clone().expect("snapshot");
		let current_watch_value = value_json_of(&current_watch);
		assert_eq!(current_watch_value, Some(Value::Object(snapshot)));
		current_watch.stop().await;
		watch.stop().await;
	}

	set_count(&harness.session, &current, 7).await;
	{
		let writes = harness.storage.commits().pop().expect("writes");
		match &writes[0] {
			StorageWrite::DocumentChange {
				content: DocumentContent::Delta { version, .. },
				..
			} => assert_eq!(*version, 3),
			other => panic!("expected delta, got {other:?}"),
		}
	}
	assert_eq!(checkpoints.load(Ordering::SeqCst), 1);
	harness.session.unload_documents().await;
	assert_eq!(snapshot(&harness.session, &current).await, Some(obj(json!({ "count": 7 }))));
}

fn value_json_of(watch: &crate::session::observation::DocumentWatch) -> Option<Value> {
	watch.value().as_ref().map(|object| Value::Object(object.as_ref().clone()))
}

use parking_lot::Mutex;

/// Upstream `rolls migration and edits back with the callback, then coalesces
/// later edits into one base`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolls_migration_and_edits_back_with_the_callback() {
	let old = doc("migration.rollback", 1, obj(json!({ "count": 1 })));
	let migrations = Arc::new(AtomicUsize::new(0));
	let current: DocToken = {
		let migrations = migrations.clone();
		DocDefinition::singleton("migration.rollback", 2, ScopeKind::Session, || {
			obj(json!({ "count": 0, "migrated": false }))
		})
		.with_migrate(move |value, _from| {
			migrations.fetch_add(1, Ordering::SeqCst);
			obj(json!({ "count": value["count"], "migrated": true }))
		})
		.into()
	};
	let harness = open_test_session();
	harness
		.session
		.commit(
			{
				let old = old.clone();
				move |tx| tx.doc(&old, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	harness.session.unload_documents().await;
	let commits = harness.storage.commits().len();

	let error = {
		let current = current.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let draft = tx.doc(&current, None, None, None).map_err(|e| e.to_string())?;
					draft.set(&[Seg::key("count")], json!(8)).map_err(|e| e.to_string())?;
					Err("rollback".to_string())
				},
				&context(),
			)
			.await
			.expect_err("rollback")
	};
	assert_eq!(error.to_string(), "rollback");
	assert_eq!(harness.storage.commits().len(), commits);
	assert_eq!(snapshot(&harness.session, &current).await, Some(obj(json!({ "count": 1, "migrated": true }))));
	assert_eq!(migrations.load(Ordering::SeqCst), 1);

	{
		let current = current.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let draft = tx.doc(&current, None, None, None).map_err(|e| e.to_string())?;
					draft.set(&[Seg::key("count")], json!(9)).map_err(|e| e.to_string())?;
					draft.set(&[Seg::key("migrated")], json!(false)).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	{
		let committed = harness.storage.commits().pop().expect("writes");
	let writes = document_writes(&committed);
		assert_eq!(writes.len(), 1);
		match writes[0] {
			StorageWrite::DocumentChange {
				content: DocumentContent::Base { version, value },
				..
			} => {
				assert_eq!(*version, 2);
				assert_eq!(*value, obj(json!({ "count": 9, "migrated": false })));
			}
			other => panic!("expected migration base, got {other:?}"),
		}
	}
	flush().await;
	let publication = harness.publications.lock().last().expect("publication").clone();
	let published = as_document(document_changes(&publication)[0]).3.clone().expect("value");
	let admitted = harness.storage.admitted_commits().pop().expect("admitted");
	match &admitted[0] {
		StorageWrite::DocumentChange {
			content: DocumentContent::Base { .. },
			..
		} => {}
		other => panic!("expected migration base, got {other:?}"),
	}
	let latest = snapshot(&harness.session, &current).await.expect("current");
	assert_eq!(published, latest);
	assert_eq!(as_document(document_changes(&publication)[0]).4.len() > 0, true);
	assert_eq!(migrations.load(Ordering::SeqCst), 1);
}

/// Upstream `strict-checks migration results before tracker ownership and
/// remains usable after rejection` — the non-JSON migration result is
/// unrepresentable over `serde_json::Value`; the port asserts the rejection
/// surface through a version-incompatible definition instead (D), keeping the
/// remainder of the case.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn strict_checks_migration_results_and_remains_usable_after_rejection() {
	let old = doc("migration.invalid", 1, obj(json!({ "count": 1 })));
	let incompatible: DocToken =
		DocDefinition::singleton("migration.invalid", 2, ScopeKind::Session, || obj(json!({}))).into();
	let other = doc("migration.invalid.other", 1, obj(json!({ "ok": true })));
	let harness = open_test_session();
	harness
		.session
		.commit(
			{
				let old = old.clone();
				move |tx| tx.doc(&old, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	harness.session.unload_documents().await;
	let commits = harness.storage.commits().len();
	let error = harness
		.session
		.snapshot(&incompatible, None, None, &context())
		.await
		.expect_err("incompatible");
	assert!(
		error.to_string().contains("newer version 2 than 1"),
		"{error}"
	);
	let error = {
		let incompatible = incompatible.clone();
		harness
			.session
			.commit(
				move |tx| tx.doc(&incompatible, None, None, None).map(|_| ()).map_err(|e| e.to_string()),
				&context(),
			)
			.await
			.expect_err("incompatible")
	};
	assert!(error.to_string().contains("newer version 2 than 1"), "{error}");
	assert_eq!(harness.storage.commits().len(), commits);
	harness
		.session
		.commit(
			{
				let other = other.clone();
				move |tx| tx.doc(&other, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("other commit");
	assert_eq!(snapshot(&harness.session, &other).await, Some(obj(json!({ "ok": true }))));
}

/// Upstream `rejects newer stored versions and older versions without
/// migration for snapshots and transactions`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rejects_newer_stored_versions_and_older_versions_without_migration() {
	let v2 = doc("migration.compatibility", 2, obj(json!({ "count": 2 })));
	let v1 = doc("migration.compatibility", 1, obj(json!({ "count": 1 })));
	let v3_without_migration = doc("migration.compatibility", 3, obj(json!({ "count": 3 })));
	let harness = open_test_session();
	harness
		.session
		.commit(
			{
				let v2 = v2.clone();
				move |tx| tx.doc(&v2, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	harness.session.unload_documents().await;
	let commits = harness.storage.commits().len();

	let error = harness.session.snapshot(&v1, None, None, &context()).await.expect_err("newer stored version");
	assert!(error.to_string().contains("newer version 2 than 1"), "{error}");
	let error = {
		let v1 = v1.clone();
		harness
			.session
			.commit(move |tx| tx.doc(&v1, None, None, None).map(|_| ()).map_err(|e| e.to_string()), &context())
			.await
			.expect_err("newer stored version")
	};
	assert!(error.to_string().contains("newer version 2 than 1"), "{error}");
	let error = harness
		.session
		.snapshot(&v3_without_migration, None, None, &context())
		.await
		.expect_err("requires migration");
	assert!(error.to_string().contains("requires migration from version 2"), "{error}");
	let error = {
		let v3 = v3_without_migration.clone();
		harness
			.session
			.commit(move |tx| tx.doc(&v3, None, None, None).map(|_| ()).map_err(|e| e.to_string()), &context())
			.await
			.expect_err("requires migration")
	};
	assert!(error.to_string().contains("requires migration from version 2"), "{error}");
	assert_eq!(harness.storage.commits().len(), commits);
}

/// Upstream `persists a required migration base before retirement without
/// consulting the checkpoint predicate`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn persists_a_required_migration_base_before_retirement() {
	let old = doc("migration.retire", 1, obj(json!({ "count": 1 })));
	let current: DocToken = DocDefinition::singleton("migration.retire", 2, ScopeKind::Session, || {
		obj(json!({ "count": 0 }))
	})
	.with_migrate(|value, _from| obj(json!({ "count": value["count"] })))
	.with_checkpoint_when(|_value, _ops, _info| panic!("must not run"))
	.into();
	let harness = open_test_session();
	harness
		.session
		.commit(
			{
				let old = old.clone();
				move |tx| tx.doc(&old, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	harness.session.unload_documents().await;
	{
		let current = current.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					tx.doc(&current, None, None, None).map(|_| ()).map_err(|e| e.to_string())?;
					tx.retire_doc(&current, None, None).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	let committed = harness.storage.commits().pop().expect("writes");
	let writes = document_writes(&committed);
	assert_eq!(writes.len(), 2);
	match writes[0] {
		StorageWrite::DocumentChange {
			content: DocumentContent::Base { version, value },
			..
		} => {
			assert_eq!(*version, 2);
			assert_eq!(*value, obj(json!({ "count": 1 })));
		}
		other => panic!("expected migration base, got {other:?}"),
	}
	assert!(matches!(writes[1], StorageWrite::DocumentRetire { .. }));
}

/// Upstream `leaves unaccessed older documents and unavailable definitions untouched`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn leaves_unaccessed_older_documents_and_unavailable_definitions_untouched() {
	let first_v1 = doc("migration.lazy.first", 1, obj(json!({ "count": 1 })));
	let second_v1 = doc("migration.lazy.second", 1, obj(json!({ "count": 2 })));
	let second_migrations = Arc::new(AtomicUsize::new(0));
	let first_v2: DocToken = DocDefinition::singleton("migration.lazy.first", 2, ScopeKind::Session, || {
		obj(json!({ "count": 0 }))
	})
	.with_migrate(|value, _from| obj(json!({ "count": value["count"] })))
	.into();
	let _second_v2: DocToken = {
		let second_migrations = second_migrations.clone();
		DocDefinition::singleton("migration.lazy.second", 2, ScopeKind::Session, || obj(json!({ "count": 0 })))
			.with_migrate(move |value, _from| {
				second_migrations.fetch_add(1, Ordering::SeqCst);
				obj(json!({ "count": value["count"] }))
			})
			.into()
	};
	let harness = open_test_session();
	{
		let first_v1 = first_v1.clone();
		let second_v1 = second_v1.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					tx.doc(&first_v1, None, None, None).map(|_| ()).map_err(|e| e.to_string())?;
					tx.doc(&second_v1, None, None, None).map(|_| ()).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	harness.session.unload_documents().await;
	let commits = harness.storage.commits().len();
	assert_eq!(snapshot(&harness.session, &first_v2).await, Some(obj(json!({ "count": 1 }))));
	assert_eq!(harness.storage.commits().len(), commits);
	assert_eq!(second_migrations.load(Ordering::SeqCst), 0);
	let second_record = harness
		.storage
		.find_document(
			&crate::types::DocumentAddress {
				kind: "migration.lazy.second".into(),
				scope: DocumentScope::Session,
				key: None,
			},
			crate::types::DocumentPoint::Current,
			&context(),
		)
		.await
		.expect("second record");
	let stored = harness
		.storage
		.document(second_record.id, crate::types::DocumentPoint::Current, &context())
		.await
		.expect("stored");
	assert_eq!(stored.version, 1);
}

// ─── Historical snapshots ────────────────────────────────────────────────────

/// Upstream `migrates current and historical rewindable values independently
/// and follows fork ancestry`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn migrates_current_and_historical_rewindable_values_independently() {
	let v1: DocToken = DocDefinition::conversation(
		"history.migration",
		1,
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		|| obj(json!({ "count": 0 })),
	)
	.into();
	let family: DocToken = DocDefinition::conversation_family(
		"history.family",
		1,
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		|seed| obj(json!({ "seed": seed, "count": 0 })),
	)
	.into();
	let migrations: Arc<Mutex<Vec<u64>>> = Arc::default();
	let v3: DocToken = {
		let migrations = migrations.clone();
		DocDefinition::conversation(
			"history.migration",
			3,
			DocumentHistory::Rewindable,
			DocumentFork::AsOf,
			|| obj(json!({ "count": 0, "version": 3 })),
		)
		.with_migrate(move |value, from_version| {
			migrations.lock().push(from_version);
			obj(json!({ "count": value["count"], "version": 3 }))
		})
		.into()
	};
	let harness = open_test_session();
	let conversation_id = create_conversation(&harness.session).await;
	let (first_entry, second_entry) = {
		let v1 = v1.clone();
		let family = family.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let first = tx
						.append_entry(conversation_id, EntryDraft { kind: Some("first".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					let v1_draft = tx.doc(&v1, Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
					v1_draft.set(&[Seg::key("count")], json!(1)).map_err(|e| e.to_string())?;
					let family_draft =
						tx.doc(&family, Some(conversation_id.0), Some("member"), Some(json!("seed")))
							.map_err(|e| e.to_string())?;
					family_draft.set(&[Seg::key("count")], json!(1)).map_err(|e| e.to_string())?;
					let second = tx
						.append_entry(conversation_id, EntryDraft { kind: Some("second".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					let v1_draft = tx.doc(&v1, Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
					v1_draft.set(&[Seg::key("count")], json!(2)).map_err(|e| e.to_string())?;
					Ok((first.id, second.id))
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	harness.session.unload_documents().await;
	let commits = harness.storage.commits().len();
	assert_eq!(
		snapshot(&harness.session, &v3).await,
		Some(obj(json!({ "count": 2, "version": 3 })))
	);
	assert_eq!(*migrations.lock(), vec![1]);
	assert_eq!(harness.storage.commits().len(), commits);

	let third_entry = {
		let v3 = v3.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let third = tx
						.append_entry(conversation_id, EntryDraft { kind: Some("third".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					tx.doc(&v3, Some(conversation_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())?;
					Ok(third.id)
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	{
		let writes = harness.storage.commits().pop().expect("writes");
		assert!(writes.iter().any(|write| matches!(
			write,
			StorageWrite::DocumentChange { content: DocumentContent::Base { version: 3, .. }, .. }
		)));
	}

	async fn snapshot_as_of(
		session: &crate::session::session::Session,
		token: &DocToken,
		conversation_id: ConversationId,
		key: Option<&str>,
		at: EntryId,
	) -> Option<JsonObject> {
		session
			.snapshot_as_of(token, conversation_id, key, at, &context())
			.await
			.expect("as-of")
	}

	assert_eq!(
		snapshot_as_of(&harness.session, &v3, conversation_id, None, first_entry).await,
		Some(obj(json!({ "count": 1, "version": 3 })))
	);
	assert_eq!(
		snapshot_as_of(&harness.session, &v3, conversation_id, None, second_entry).await,
		Some(obj(json!({ "count": 2, "version": 3 })))
	);
	assert_eq!(
		snapshot_as_of(&harness.session, &v3, conversation_id, None, third_entry).await,
		Some(obj(json!({ "count": 2, "version": 3 })))
	);
	assert_eq!(*migrations.lock(), vec![1, 1, 1]);
	assert_eq!(
		snapshot_as_of(&harness.session, &family, conversation_id, Some("member"), first_entry).await,
		Some(obj(json!({ "seed": "seed", "count": 1 })))
	);

	let child_id = harness
		.session
		.commit(
			move |tx| {
				tx.fork_conversation(conversation_id, second_entry, crate::types::ConversationOwnership::Ownerless)
					.map(|record| record.id)
					.map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("fork");
	assert_eq!(
		snapshot_as_of(&harness.session, &v3, child_id, None, first_entry).await,
		Some(obj(json!({ "count": 1, "version": 3 })))
	);
	assert_eq!(
		snapshot_as_of(&harness.session, &v3, child_id, None, second_entry).await,
		Some(obj(json!({ "count": 2, "version": 3 })))
	);
	assert_eq!(
		snapshot_as_of(&harness.session, &family, child_id, Some("member"), first_entry).await,
		Some(obj(json!({ "seed": "seed", "count": 1 })))
	);
	let error = harness
		.session
		.snapshot_as_of(&v3, child_id, None, third_entry, &context())
		.await
		.expect_err("invisible entry");
	assert!(error.to_string().contains(&format!("Entry {third_entry} is not visible")), "{error}");
}

/// Upstream `selects the incarnation alive at the entry commit across
/// retirement and recreation`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn selects_the_incarnation_alive_at_the_entry_commit_across_retirement_and_recreation() {
	let token: DocToken = DocDefinition::conversation(
		"history.incarnation",
		1,
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		|| obj(json!({ "value": "initial" })),
	)
	.into();
	let harness = open_test_session();
	let conversation_id = create_conversation(&harness.session).await;
	let before_creation = harness
		.session
		.commit(
			move |tx| {
				tx.append_entry(conversation_id, EntryDraft { kind: Some("before".into()), ..Default::default() })
					.map(|record| record.id)
					.map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	let mut entries = Vec::new();
	for (kind, value) in [("create", Some("old")), ("retire", None), ("recreate", Some("new"))] {
		let entry = harness
			.session
			.commit(
				{
					let token = token.clone();
					move |tx| -> Result<_, String> {
						let entry = tx
							.append_entry(conversation_id, EntryDraft { kind: Some(kind.into()), ..Default::default() })
							.map_err(|e| e.to_string())?;
						match value {
							Some(value) => {
								let draft = tx.doc(&token, Some(conversation_id.0), None, None)
									.map_err(|e| e.to_string())?;
								draft.set(&[Seg::key("value")], json!(value)).map_err(|e| e.to_string())?;
							}
							None => tx.retire_doc(&token, Some(conversation_id.0), None)
								.map_err(|e| e.to_string())?,
						}
						Ok(entry.id)
					}
				},
				&context(),
			)
			.await
			.expect("commit");
		entries.push(entry);
	}
	let (created_at, retired_at, recreated_at) = (entries[0], entries[1], entries[2]);

	async fn as_of(session: &crate::session::session::Session, token: &DocToken, conversation_id: ConversationId, at: EntryId) -> Option<JsonObject> {
		session
			.snapshot_as_of(token, conversation_id, None, at, &context())
			.await
			.expect("as-of")
	}
	assert_eq!(as_of(&harness.session, &token, conversation_id, before_creation).await, None);
	assert_eq!(
		as_of(&harness.session, &token, conversation_id, created_at).await,
		Some(obj(json!({ "value": "old" })))
	);
	assert_eq!(as_of(&harness.session, &token, conversation_id, retired_at).await, None);
	assert_eq!(
		as_of(&harness.session, &token, conversation_id, recreated_at).await,
		Some(obj(json!({ "value": "new" })))
	);
	harness.session.close(&context()).await.expect("close");
	let error = harness
		.session
		.snapshot_as_of(&token, conversation_id, None, recreated_at, &context())
		.await
		.expect_err("closed");
	assert_eq!(error.to_string(), "Session is closed");
}

// ─── Tracker cache across definition versions ────────────────────────────────

fn versioned_v1() -> DocToken {
	doc("cache.versioned", 1, obj(json!({ "name": "first" })))
}

fn versioned_v2() -> DocToken {
	DocDefinition::singleton("cache.versioned", 2, ScopeKind::Session, || obj(json!({ "names": [] })))
		.with_migrate(|value, _from| obj(json!({ "names": [value["name"]] })))
		.into()
}

/// Upstream `migrates a document cached by an older token without unloading`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn migrates_a_document_cached_by_an_older_token_without_unloading() {
	let harness = open_test_session();
	let v1 = versioned_v1();
	let v2 = versioned_v2();
	harness
		.session
		.commit(
			{
				let v1 = v1.clone();
				move |tx| tx.doc(&v1, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	assert_eq!(snapshot(&harness.session, &v1).await, Some(obj(json!({ "name": "first" }))));

	// Reloaded extension code accesses the still-cached document with a newer token.
	assert_eq!(snapshot(&harness.session, &v2).await, Some(obj(json!({ "names": ["first"] }))));
	{
		let v2 = v2.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let draft = tx.doc(&v2, None, None, None).map_err(|e| e.to_string())?;
					let mut names = draft
						.get(&[Seg::key("names")])
						.map_err(|e| e.to_string())?
						.unwrap_or(json!([]));
					names.as_array_mut().expect("names array").push(json!("second"));
					draft.set(&[Seg::key("names")], names).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	{
		let committed = harness.storage.commits().pop().expect("writes");
	let writes = document_writes(&committed);
		match writes[0] {
			StorageWrite::DocumentChange {
				content: DocumentContent::Base { version, value },
				..
			} => {
				assert_eq!(*version, 2);
				assert_eq!(*value, obj(json!({ "names": ["first", "second"] })));
			}
			other => panic!("expected version base, got {other:?}"),
		}
	}
	let error = harness.session.snapshot(&v1, None, None, &context()).await.expect_err("older token");
	assert!(error.to_string().contains("newer version 2"), "{error}");
}

/// Upstream `sends observers of an older shape a root replacement after a
/// newer token writes`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sends_observers_of_an_older_shape_a_root_replacement_after_a_newer_token_writes() {
	let harness = open_test_session();
	let v1 = versioned_v1();
	let v2 = versioned_v2();
	harness
		.session
		.commit(
			{
				let v1 = v1.clone();
				move |tx| tx.doc(&v1, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	let state = harness
		.session
		.document_state(&v1, None, None, &context())
		.await
		.unwrap()
		.expect("state");
	let watch = harness.session.watch_doc(&v1, None, None, &context()).await.unwrap().expect("watch");
	let frames: Arc<Mutex<Vec<Vec<pi_chord::delta::Op>>>> = Arc::default();
	{
		let frames = frames.clone();
		watch
			.start(Arc::new(move |_value, ops, _context: Context| {
				let frames = frames.clone();
				Box::pin(async move {
					frames.lock().push(ops);
				})
			}))
			.expect("start");
	}
	{
		let v2 = v2.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let draft = tx.doc(&v2, None, None, None).map_err(|e| e.to_string())?;
					let mut names = draft
						.get(&[Seg::key("names")])
						.map_err(|e| e.to_string())?
						.unwrap_or(json!([]));
					names.as_array_mut().expect("names array").push(json!("second"));
					draft.set(&[Seg::key("names")], names).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	tokio::time::timeout(std::time::Duration::from_secs(5), async {
		while frames.lock().is_empty() {
			flush().await;
		}
	})
	.await
	.expect("delivered");
	assert_eq!(state.value().clone(), json!({ "names": ["first", "second"] }));
	assert_eq!(value_json_of(&watch), Some(json!({ "names": ["first", "second"] })));
	assert_eq!(
		*frames.lock(),
		vec![vec![pi_chord::delta::Op::Replace(json!({ "names": ["first", "second"] }))]]
	);
	state.dispose();
	watch.stop().await;
}

/// Upstream `serves an older token from Storage after a newer token migrated
/// only in memory`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serves_an_older_token_from_storage_after_a_newer_token_migrated_only_in_memory() {
	let harness = open_test_session();
	let v1 = versioned_v1();
	let v2 = versioned_v2();
	harness
		.session
		.commit(
			{
				let v1 = v1.clone();
				move |tx| tx.doc(&v1, None, None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	harness.session.unload_documents().await;
	assert_eq!(snapshot(&harness.session, &v2).await, Some(obj(json!({ "names": ["first"] }))));
	assert_eq!(snapshot(&harness.session, &v1).await, Some(obj(json!({ "name": "first" }))));
	{
		let v1 = v1.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let draft = tx.doc(&v1, None, None, None).map_err(|e| e.to_string())?;
					draft.set(&[Seg::key("name")], json!("renamed")).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	assert_eq!(snapshot(&harness.session, &v2).await, Some(obj(json!({ "names": ["renamed"] }))));
}
