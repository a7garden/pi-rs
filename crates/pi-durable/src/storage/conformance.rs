//! Port of `packages/durable/src/testing/storage-conformance.ts` — the shared
//! backend conformance suite, run against every storage backend in this crate
//! (`MemoryStorage`, `JsonlStorage` fresh and across reopen, `SqliteStorage`
//! fresh and across reopen).
//!
//! Deviations (module ledger):
//! - D1: upstream freezes prepared writes; Rust detachment is by clone, so the
//!   "frozen write throws on mutation" expectation degrades to
//!   "mutating a returned value cannot affect stored state".
//! - D2: the fixed `Storage` trait surface is infallible for reads, so cases
//!   that upstream assert as rejections (`Unknown conversation`, historical
//!   reads of current-only documents, reads/mints after close, ID-space
//!   exhaustion) assert `None` / continuation instead; each adaptation is
//!   marked inline.
//! - D3: "keeps indexed string identities lossless" cannot construct lone
//!   UTF-16 surrogates in Rust `String`s; distinct multi-byte sentinels are
//!   used with identical assertions.

#![cfg(test)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use pi_chord::context::{Context, BACKGROUND_CONTEXT};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};

use crate::errors::StorageRejected;
pub(crate) use crate::types::{
	ConversationId, Cursor, DocumentAddress, DocumentCreate, DocumentId, DocumentPoint, DocumentQuery, EntryId,
	EntryRecord, Page, Seq, Storage, SubmissionId, SubmissionRecord, SubmissionQuery, TaskId, TaskQuery, TaskRecord,
	TaskStateStatus, ROOT_CONVERSATION_ID,
};

pub(crate) fn ctx() -> &'static Context {
	&BACKGROUND_CONTEXT
}

pub(crate) fn from_json<T: DeserializeOwned>(value: Value) -> T {
	serde_json::from_value(value).expect("test record deserializes")
}

pub(crate) fn record<T: DeserializeOwned>(value: Value) -> T {
	from_json(value)
}

pub(crate) fn conversation_json(id: u64) -> Value {
	json!({ "id": id })
}

pub(crate) fn conversation(id: u64) -> crate::types::ConversationRecord {
	record(conversation_json(id))
}

pub(crate) fn conversation_with_parent(id: u64, parent_conversation: u64, at: u64) -> Value {
	json!({ "id": id, "parent": { "conversationId": parent_conversation, "at": at } })
}

pub(crate) fn conversation_with_owner(id: u64, owner_conversation: u64, owner_task: u64) -> Value {
	json!({ "id": id, "owner": { "conversationId": owner_conversation, "taskId": owner_task } })
}

pub(crate) fn pending_task_json(id: u64, conversation: u64, phase: &str) -> Value {
	json!({
		"id": id,
		"conversationId": conversation,
		"kind": "test.task",
		"version": 1,
		"input": { "value": id },
		"state": { "status": "pending", "checkpoint": { "phase": phase } },
		"background": false,
		"abortRequested": false,
	})
}

pub(crate) fn pending_task(id: u64, conversation: u64) -> TaskRecord {
	record(pending_task_json(id, conversation, "ready"))
}

pub(crate) fn entry_json(id: u64, conversation: u64, kind: &str, extra: Value) -> Value {
	let mut object = json!({ "id": id, "conversationId": conversation, "kind": kind });
	if let (Value::Object(base), Value::Object(extra)) = (&mut object, extra) {
		for (key, value) in extra {
			base.insert(key, value);
		}
	}
	object
}

pub(crate) fn entry(id: u64, conversation: u64) -> EntryRecord {
	record(entry_json(id, conversation, "message", json!({})))
}

pub(crate) async fn commit(storage: &Arc<dyn Storage>, writes: Vec<Value>) -> Result<Seq, StorageRejected> {
	let writes: Vec<crate::types::StorageWrite> =
		writes.into_iter().map(|write| from_json(write)).collect();
	storage.commit(&writes, ctx()).await
}

pub(crate) async fn commit_ok(storage: &Arc<dyn Storage>, writes: Vec<Value>) -> Seq {
	commit(storage, writes).await.expect("commit succeeds")
}

pub(crate) fn ids<T>(page: &Page<T>, id_of: impl Fn(&T) -> u64) -> Vec<u64> {
	page.items.iter().map(id_of).collect()
}

pub(crate) fn at(seq: Seq) -> DocumentPoint {
	DocumentPoint::At(seq)
}

pub(crate) fn session_document(id: u64, kind: &str) -> Value {
	json!({ "id": id, "kind": kind, "scope": { "kind": "session" } })
}

async fn create_root(storage: &Arc<dyn Storage>) -> ConversationId {
	commit_ok(storage, vec![json!({ "type": "conversation", "value": conversation_json(ROOT_CONVERSATION_ID.0) })]).await;
	ROOT_CONVERSATION_ID
}

type StorageFuture = Pin<Box<dyn Future<Output = Arc<dyn Storage>> + Send>>;

/// Run every conformance case against storages produced by `factory`; each
/// case receives a fresh storage.
pub(super) async fn run_conformance(factory: impl Fn() -> StorageFuture + Send + Sync) {
	run_case(&factory, "reserves ID 1 for the immutable root conversation", case_reserves_root_id).await;
	run_case(&factory, "commits mixed table writes atomically and rolls all of them back on failure", case_atomic_rollback).await;
	run_case(&factory, "detaches retained writes and every returned record", case_detaches_records).await;
	run_case(&factory, "detaches prototype-like JSON keys without changing object prototypes", case_proto_keys).await;
	run_case(&factory, "indexes entries committed out of ID order", case_out_of_order_entries).await;
	run_case(&factory, "continues an entry cursor below its last item after a newer commit", case_entry_cursor).await;
	run_case(&factory, "paginates conversations by opaque cursor in ascending ID order", case_conversation_cursor).await;
	run_case(&factory, "filters and pages conversations by durable owner edges", case_owner_edges).await;
	run_case(&factory, "scans deep fork history newest-first through every ancestor cap", case_deep_fork_history).await;
	run_case(&factory, "replaces complete task records and pages filtered task scans", case_task_replace_scan).await;
	run_case(&factory, "stores owners and scans waiting and completing tasks by status", case_waiting_completing).await;
	run_case(&factory, "indexes request IDs per conversation and replaces complete submission records", case_request_ids).await;
	run_case(&factory, "stores passive write submissions without input-only lifecycle states", case_passive_writes).await;
	run_case(&factory, "reconstructs rewindable documents and preserves half-open incarnations", case_rewindable_documents).await;
	run_case(&factory, "streams long document tails across root replacement deltas", case_long_tails).await;
	run_case(&factory, "copies stored document bases independently and rejects ambiguous sources", case_document_copies).await;
	run_case(&factory, "uses bases for version transitions and rejects historical reads of current-only documents", case_version_transitions).await;
	run_case(&factory, "indexes logical addresses and exact-scope scans independently", case_address_indexing).await;
	run_case(&factory, "keeps document lifecycle failures atomic and gives create-plus-retire an empty lifetime", case_lifecycle_atomic).await;
	run_case(&factory, "rolls back record tables and secondary indexes when a document command fails", case_document_rollback).await;
	run_case(&factory, "keeps indexed string identities lossless", case_indexed_identities).await;
	run_case(&factory, "keeps one global record ID namespace and rejects exhausted ID minting", case_global_namespace).await;
	run_case(&factory, "rejects every operation after close", case_rejects_after_close).await;
}

async fn run_case(
	factory: &(impl Fn() -> StorageFuture + Send + Sync),
	name: &str,
	case: fn(Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>>,
) {
	let storage = factory().await;
	let result = tokio::task::spawn({
		let name = name.to_string();
		async move { case(storage).await }
	})
	.await;
	if let Err(panic) = result {
		std::panic::resume_unwind(panic.into_panic());
	}
}

fn case_reserves_root_id(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		assert_eq!(storage.mint_id().await, 2);
		let root = create_root(&storage).await;
		assert_eq!(root, ROOT_CONVERSATION_ID);
		assert_eq!(storage.conversation(ROOT_CONVERSATION_ID, ctx()).await, Some(conversation(ROOT_CONVERSATION_ID.0)));
		let error = commit(&storage, vec![json!({ "type": "conversation", "value": conversation_json(ROOT_CONVERSATION_ID.0) })])
			.await
			.expect_err("duplicate root is rejected");
		assert!(
			error.message.contains(&format!("ID {} already belongs to conversation", ROOT_CONVERSATION_ID.0)),
			"unexpected message: {}",
			error.message
		);
	})
}

fn case_atomic_rollback(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let entry_id = EntryId(storage.mint_id().await);
		let task_id = TaskId(storage.mint_id().await);
		let submission_id = SubmissionId(storage.mint_id().await);
		let task = pending_task(task_id.0, root.0);
		let input = record::<SubmissionRecord>(json!({
			"id": submission_id.0,
			"conversationId": root.0,
			"requestId": "request-1",
			"type": "input",
			"status": "placed",
			"entry": entry_id.0,
		}));
		let initial_seq = commit_ok(
			&storage,
			vec![
				json!({ "type": "entry", "value": entry_json(entry_id.0, root.0, "user", json!({ "data": { "text": "hello" } })) }),
				json!({ "type": "task", "value": pending_task_json(task_id.0, root.0, "ready") }),
				json!({ "type": "submission", "value": {
					"id": submission_id.0, "conversationId": root.0, "requestId": "request-1",
					"type": "input", "status": "placed", "entry": entry_id.0,
				}}),
			],
		)
		.await;

		assert_eq!(
			storage.entry(entry_id, ctx()).await,
			Some((record(entry_json(entry_id.0, root.0, "user", json!({ "data": { "text": "hello" } }))), initial_seq))
		);
		assert_eq!(storage.task(task_id, ctx()).await, Some(task.clone()));
		assert_eq!(storage.submission(submission_id, ctx()).await, Some(input.clone()));

		let transient_entry_id = EntryId(storage.mint_id().await);
		let error = commit(
			&storage,
			vec![
				json!({ "type": "task", "value": {
					"id": task_id.0, "conversationId": root.0, "kind": "test.task", "version": 1,
					"input": { "value": task_id.0 },
					"state": { "status": "running", "checkpoint": { "phase": "effect" } },
					"background": false, "abortRequested": false,
				}}),
				json!({ "type": "submission", "value": {
					"id": submission_id.0, "conversationId": root.0, "requestId": "request-1",
					"type": "input", "status": "done", "entry": entry_id.0, "answer": transient_entry_id.0,
				}}),
				json!({ "type": "entry", "value": entry_json(transient_entry_id.0, root.0, "assistant", json!({})) }),
				json!({ "type": "conversation", "value": conversation_json(root.0) }),
			],
		)
		.await
		.expect_err("conversation rewrite must reject");
		assert!(error.message.contains(&format!("ID {} already belongs to conversation", root.0)), "{}", error.message);

		assert_eq!(storage.task(task_id, ctx()).await, Some(task));
		assert_eq!(storage.submission(submission_id, ctx()).await, Some(input));
		assert_eq!(storage.entry(transient_entry_id, ctx()).await, None);
		let after_rollback_seq = commit_ok(
			&storage,
			vec![json!({ "type": "entry", "value": entry_json(storage.mint_id().await, root.0, "after-rollback", json!({})) })],
		)
		.await;
		assert!(after_rollback_seq > initial_seq);
	})
}

fn case_detaches_records(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let entry_id = EntryId(storage.mint_id().await);
		let task_id = TaskId(storage.mint_id().await);
		let submission_id = SubmissionId(storage.mint_id().await);
		let stored_entry = record::<EntryRecord>(entry_json(entry_id.0, root.0, "note", json!({ "data": { "nested": [1, 2] } })));
		let stored_task = record::<TaskRecord>(json!({
			"id": task_id.0, "conversationId": root.0, "kind": "test.task", "version": 1,
			"input": { "value": task_id.0 },
			"state": { "status": "pending", "checkpoint": { "phase": "ready", "nested": { "count": 1 } } },
			"background": false, "abortRequested": false,
		}));
		let stored_input = record::<SubmissionRecord>(json!({
			"id": submission_id.0, "conversationId": root.0, "type": "input",
			"status": "unanswered", "reason": "failed", "detail": { "codes": ["initial"] },
		}));
		commit_ok(
			&storage,
			vec![
				json!({ "type": "entry", "value": entry_json(entry_id.0, root.0, "note", json!({ "data": { "nested": [1, 2] } })) }),
				json!({ "type": "task", "value": {
					"id": task_id.0, "conversationId": root.0, "kind": "test.task", "version": 1,
					"input": { "value": task_id.0 },
					"state": { "status": "pending", "checkpoint": { "phase": "ready", "nested": { "count": 1 } } },
					"background": false, "abortRequested": false,
				}}),
				json!({ "type": "submission", "value": {
					"id": submission_id.0, "conversationId": root.0, "type": "input",
					"status": "unanswered", "reason": "failed", "detail": { "codes": ["initial"] },
				}}),
			],
		)
		.await;

		// Reads return detached values (Rust clones), so caller-side mutation
		// of returned records cannot observe or change stored state (D1).
		let read_entry = storage.entry(entry_id, ctx()).await.unwrap().0;
		assert_eq!(read_entry, stored_entry);
		let read_task = storage.task(task_id, ctx()).await.unwrap();
		assert_eq!(read_task.state, stored_task.state);
		let read_input = storage.submission(submission_id, ctx()).await.unwrap();
		assert_eq!(read_input, stored_input);

		assert_eq!(storage.entry(entry_id, ctx()).await, Some((stored_entry.clone(), commit_seq_of(&storage, entry_id).await)));
		assert_eq!(storage.task(task_id, ctx()).await, Some(stored_task));
		assert_eq!(storage.submission(submission_id, ctx()).await, Some(stored_input));
	})
}

async fn commit_seq_of(storage: &Arc<dyn Storage>, entry_id: EntryId) -> Seq {
	storage.entry(entry_id, ctx()).await.unwrap().1
}

fn case_proto_keys(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let entry_id = EntryId(storage.mint_id().await);
		let data = json!({
			"__proto__": { "polluted": false },
			"constructor": { "label": "stored" },
			"toString": "value",
		});
		commit_ok(&storage, vec![json!({ "type": "entry", "value": entry_json(entry_id.0, root.0, "note", json!({ "data": data })) })])
			.await;
		let read = storage.entry(entry_id, ctx()).await.unwrap().0;
		assert_eq!(read.data, Some(data));
	})
}

fn case_out_of_order_entries(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		commit_ok(
			&storage,
			vec![
				json!({ "type": "entry", "value": entry_json(30, root.0, "message", json!({})) }),
				json!({ "type": "entry", "value": entry_json(10, root.0, "message", json!({})) }),
				json!({ "type": "entry", "value": entry_json(20, root.0, "marker", json!({ "head": 10 })) }),
			],
		)
		.await;
		let page = storage.scan_entries(&record(json!({ "conversationId": root.0 })), 10, None, ctx()).await;
		assert_eq!(ids(&page, |entry| entry.id.0), vec![30, 20, 10]);
		assert_eq!(storage.find_latest_head_marker(root, None, ctx()).await.unwrap().id.0, 20);
	})
}

fn case_entry_cursor(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let oldest = storage.mint_id().await;
		let middle = storage.mint_id().await;
		let newest = storage.mint_id().await;
		commit_ok(
			&storage,
			vec![
				json!({ "type": "entry", "value": entry_json(oldest, root.0, "message", json!({})) }),
				json!({ "type": "entry", "value": entry_json(middle, root.0, "message", json!({})) }),
				json!({ "type": "entry", "value": entry_json(newest, root.0, "message", json!({})) }),
			],
		)
		.await;
		let first = storage.scan_entries(&record(json!({ "conversationId": root.0 })), 2, None, ctx()).await;
		assert_eq!(ids(&first, |entry| entry.id.0), vec![newest, middle]);
		let appended = storage.mint_id().await;
		commit_ok(&storage, vec![json!({ "type": "entry", "value": entry_json(appended, root.0, "message", json!({})) })]).await;
		let second = storage.scan_entries(&record(json!({ "conversationId": root.0 })), 2, first.next.as_ref(), ctx()).await;
		assert_eq!(ids(&second, |entry| entry.id.0), vec![oldest]);
		assert!(second.next.is_none());
	})
}

fn case_conversation_cursor(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let second = storage.mint_id().await;
		let third = storage.mint_id().await;
		commit_ok(
			&storage,
			vec![
				json!({ "type": "conversation", "value": conversation_json(third) }),
				json!({ "type": "conversation", "value": conversation_json(second) }),
			],
		)
		.await;
		let first = storage.scan_conversations(&Default::default(), 2, None, ctx()).await;
		assert_eq!(ids(&first, |record| record.id.0), vec![root.0, second]);
		assert!(first.next.is_some());
		// Cursors are opaque JSON round-tripped through serialization.
		let round_tripped: Cursor = serde_json::from_value(serde_json::to_value(first.next.as_ref().unwrap()).unwrap()).unwrap();
		let second_page = storage.scan_conversations(&Default::default(), 2, Some(&round_tripped), ctx()).await;
		assert_eq!(ids(&second_page, |record| record.id.0), vec![third]);
		assert!(second_page.next.is_none());
	})
}

fn case_owner_edges(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let other_owner = storage.mint_id().await;
		let first_task = storage.mint_id().await;
		let second_task = storage.mint_id().await;
		let first = storage.mint_id().await;
		let second = storage.mint_id().await;
		let third = storage.mint_id().await;
		commit_ok(
			&storage,
			vec![
				json!({ "type": "conversation", "value": conversation_json(other_owner) }),
				json!({ "type": "conversation", "value": conversation_with_owner(first, root.0, first_task) }),
				json!({ "type": "conversation", "value": conversation_with_owner(second, root.0, second_task) }),
				json!({ "type": "conversation", "value": conversation_with_owner(third, other_owner, first_task) }),
			],
		)
		.await;
		let first_page = storage.scan_conversations(&record(json!({ "ownerConversationId": root.0 })), 1, None, ctx()).await;
		assert_eq!(ids(&first_page, |record| record.id.0), vec![first]);
		assert!(first_page.next.is_some());
		let second_page = storage
			.scan_conversations(&record(json!({ "ownerConversationId": root.0 })), 1, first_page.next.as_ref(), ctx())
			.await;
		assert_eq!(ids(&second_page, |record| record.id.0), vec![second]);
		assert!(second_page.next.is_none());
		assert_eq!(
			ids(&storage.scan_conversations(&record(json!({ "ownerTaskId": first_task })), 10, None, ctx()).await, |record| record.id.0),
			vec![first, third]
		);
		assert_eq!(
			ids(
				&storage
					.scan_conversations(&record(json!({ "ownerConversationId": root.0, "ownerTaskId": first_task })), 10, None, ctx())
					.await,
				|record| record.id.0
			),
			vec![first]
		);
	})
}

fn case_deep_fork_history(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let root_first = storage.mint_id().await;
		let root_fork_point = storage.mint_id().await;
		let root_excluded_same_commit = storage.mint_id().await;
		let root_entries_seq = commit_ok(
			&storage,
			vec![
				json!({ "type": "entry", "value": entry_json(root_first, root.0, "message", json!({})) }),
				json!({ "type": "entry", "value": entry_json(root_fork_point, root.0, "marker", json!({ "head": root_first })) }),
				json!({ "type": "entry", "value": entry_json(root_excluded_same_commit, root.0, "message", json!({})) }),
			],
		)
		.await;
		let child = storage.mint_id().await;
		commit_ok(&storage, vec![json!({ "type": "conversation", "value": conversation_with_parent(child, root.0, root_fork_point) })]).await;
		let child_fork_point = storage.mint_id().await;
		let child_excluded = storage.mint_id().await;
		commit_ok(
			&storage,
			vec![
				json!({ "type": "entry", "value": entry_json(child_fork_point, child, "note", json!({})) }),
				json!({ "type": "entry", "value": entry_json(child_excluded, child, "message", json!({})) }),
			],
		)
		.await;
		let root_excluded_later = storage.mint_id().await;
		commit_ok(&storage, vec![json!({ "type": "entry", "value": entry_json(root_excluded_later, root.0, "message", json!({})) })]).await;
		let grandchild = storage.mint_id().await;
		commit_ok(&storage, vec![json!({ "type": "conversation", "value": conversation_with_parent(grandchild, child, child_fork_point) })]).await;
		let grandchild_head = storage.mint_id().await;
		let grandchild_tail = storage.mint_id().await;
		let grandchild_entries_seq = commit_ok(
			&storage,
			vec![
				json!({ "type": "entry", "value": entry_json(grandchild_head, grandchild, "marker", json!({ "head": grandchild_head })) }),
				json!({ "type": "entry", "value": entry_json(grandchild_tail, grandchild, "message", json!({})) }),
			],
		)
		.await;
		let child_excluded_later = storage.mint_id().await;
		commit_ok(&storage, vec![json!({ "type": "entry", "value": entry_json(child_excluded_later, child, "message", json!({})) })]).await;

		let query = json!({ "conversationId": grandchild });
		let first = storage.scan_entries(&record(query.clone()), 2, None, ctx()).await;
		assert_eq!(ids(&first, |entry| entry.id.0), vec![grandchild_tail, grandchild_head]);
		let second = storage.scan_entries(&record(query.clone()), 2, first.next.as_ref(), ctx()).await;
		assert_eq!(ids(&second, |entry| entry.id.0), vec![child_fork_point, root_fork_point]);
		let third = storage.scan_entries(&record(query.clone()), 2, second.next.as_ref(), ctx()).await;
		assert_eq!(ids(&third, |entry| entry.id.0), vec![root_first]);
		assert!(third.next.is_none());

		let current_marker = storage.find_latest_head_marker(ConversationId(grandchild), None, ctx()).await.unwrap();
		assert_eq!(current_marker.id.0, grandchild_head);
		assert_eq!(current_marker.head, Some(EntryId(grandchild_head)));
		let historical_marker = storage.find_latest_head_marker(ConversationId(grandchild), Some(EntryId(child_fork_point)), ctx()).await.unwrap();
		assert_eq!(historical_marker.id.0, root_fork_point);
		assert_eq!(historical_marker.head, Some(EntryId(root_first)));
		assert!(storage.find_latest_head_marker(ConversationId(grandchild), Some(EntryId(root_first)), ctx()).await.is_none());

		let active_first = storage
			.scan_entries(&record(json!({ "conversationId": grandchild, "minEntryId": current_marker.head })), 1, None, ctx())
			.await;
		assert_eq!(ids(&active_first, |entry| entry.id.0), vec![grandchild_tail]);
		assert!(active_first.next.is_some());
		let active_second = storage
			.scan_entries(&record(json!({ "conversationId": grandchild, "minEntryId": current_marker.head })), 1, active_first.next.as_ref(), ctx())
			.await;
		assert_eq!(ids(&active_second, |entry| entry.id.0), vec![grandchild_head]);
		assert!(active_second.next.is_none());

		assert_eq!(
			ids(
				&storage
					.scan_entries(
						&record(json!({ "conversationId": grandchild, "minEntryId": historical_marker.head, "maxEntryId": child_fork_point })),
						10,
						None,
						ctx(),
					)
					.await,
				|entry| entry.id.0
			),
			vec![child_fork_point, root_fork_point, root_first]
		);

		assert_eq!(storage.entry(EntryId(root_first), ctx()).await, Some((entry(root_first, root.0), root_entries_seq)));
		assert_eq!(storage.entry(EntryId(root_fork_point), ctx()).await.unwrap().1, root_entries_seq);
		assert_eq!(storage.entry(EntryId(grandchild_head), ctx()).await.unwrap().1, grandchild_entries_seq);
		assert_eq!(storage.entry(EntryId(grandchild_tail), ctx()).await.unwrap().1, grandchild_entries_seq);
		assert_eq!(storage.entry(EntryId(999_999), ctx()).await, None);

		assert_eq!(storage.entry_in(ConversationId(grandchild), EntryId(root_first), ctx()).await, Some((entry(root_first, root.0), root_entries_seq)));
		assert_eq!(storage.entry_in(ConversationId(grandchild), EntryId(child_fork_point), ctx()).await.unwrap().0.conversation_id.0, child);
		assert_eq!(storage.entry_in(ConversationId(grandchild), EntryId(grandchild_tail), ctx()).await.unwrap().1, grandchild_entries_seq);
		// Adapted from upstream rejections (D2): invisible / unknown-ancestor
		// lookups collapse to `None`.
		assert_eq!(storage.entry_in(ConversationId(grandchild), EntryId(root_excluded_same_commit), ctx()).await, None);
		assert_eq!(storage.entry_in(ConversationId(grandchild), EntryId(root_excluded_later), ctx()).await, None);
		assert_eq!(storage.entry_in(ConversationId(grandchild), EntryId(child_excluded), ctx()).await, None);
		assert_eq!(storage.entry_in(ConversationId(grandchild), EntryId(child_excluded_later), ctx()).await, None);
		assert_eq!(storage.entry_in(root, EntryId(grandchild_head), ctx()).await, None);
		assert_eq!(storage.entry_in(ConversationId(grandchild), EntryId(999_999), ctx()).await, None);
		assert_eq!(storage.entry_in(ConversationId(999_999), EntryId(root_first), ctx()).await, None);
		assert!(storage.scan_entries(&record(json!({ "conversationId": 999_999 })), 10, None, ctx()).await.items.is_empty());
	})
}

fn case_task_replace_scan(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let first = storage.mint_id().await;
		let second = storage.mint_id().await;
		let third = storage.mint_id().await;
		let mut first_task = pending_task_json(first, root.0, "ready");
		first_task["memos"] = json!({ "winner": "first" });
		let mut second_task = pending_task_json(second, root.0, "ready");
		second_task["background"] = json!(true);
		let mut third_task = pending_task_json(third, root.0, "ready");
		third_task["abortRequested"] = json!(true);
		commit_ok(
			&storage,
			vec![
				json!({ "type": "task", "value": first_task }),
				json!({ "type": "task", "value": second_task }),
				json!({ "type": "task", "value": third_task }),
			],
		)
		.await;

		let running = json!({
			"id": first, "conversationId": root.0, "kind": "test.task", "version": 1,
			"input": { "value": first },
			"state": { "status": "running", "checkpoint": { "phase": "effect", "attempt": 1 } },
			"background": false, "abortRequested": true,
			"memos": { "winner": "first" },
		});
		commit_ok(&storage, vec![json!({ "type": "task", "value": running.clone() })]).await;
		assert_eq!(storage.task(TaskId(first), ctx()).await, Some(record(running.clone())));
		let terminal = json!({
			"id": first, "conversationId": root.0, "kind": "test.task", "version": 1,
			"input": { "value": first },
			"state": { "status": "terminal", "outcome": { "status": "completed", "result": { "entryId": 99 } } },
			"background": false, "abortRequested": true,
		});
		commit_ok(&storage, vec![json!({ "type": "task", "value": terminal.clone() })]).await;
		assert_eq!(storage.task(TaskId(first), ctx()).await, Some(record(terminal.clone())));

		let pending_page = storage.scan_tasks(&record(json!({ "status": "pending" })), 1, None, ctx()).await;
		assert_eq!(ids(&pending_page, |task| task.id.0), vec![second]);
		assert!(pending_page.next.is_some());
		assert_eq!(
			ids(&storage.scan_tasks(&record(json!({ "status": "pending" })), 1, pending_page.next.as_ref(), ctx()).await, |task| task.id.0),
			vec![third]
		);
		assert_eq!(
			ids(&storage.scan_tasks(&record(json!({ "status": "terminal", "abortRequested": true })), 10, None, ctx()).await, |task| task.id.0),
			vec![first]
		);
		assert_eq!(
			ids(&storage.scan_tasks(&record(json!({ "background": true })), 10, None, ctx()).await, |task| task.id.0),
			vec![second]
		);
	})
}

fn case_waiting_completing(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let owner = storage.mint_id().await;
		let waiting = storage.mint_id().await;
		let completing = storage.mint_id().await;
		let waiting_task = json!({
			"id": waiting, "conversationId": root.0, "kind": "test.task", "version": 1,
			"input": { "value": waiting },
			"state": { "status": "waiting", "checkpoint": { "phase": "next" }, "on": [owner], "policy": "allSettled" },
			"background": false, "abortRequested": false,
			"memos": { "kept": true },
		});
		let completing_task = json!({
			"id": completing, "conversationId": root.0, "kind": "test.task", "version": 1,
			"input": { "value": completing },
			"state": { "status": "completing", "outcome": { "status": "failed", "error": { "message": "held" } } },
			"background": false, "abortRequested": false,
		});
		commit_ok(
			&storage,
			vec![
				json!({ "type": "task", "value": pending_task_json(owner, root.0, "ready") }),
				json!({ "type": "task", "value": waiting_task.clone() }),
				json!({ "type": "task", "value": completing_task.clone() }),
			],
		)
		.await;
		assert_eq!(storage.task(TaskId(waiting), ctx()).await, Some(record(waiting_task.clone())));
		assert_eq!(storage.task(TaskId(completing), ctx()).await, Some(record(completing_task.clone())));
		let scan = |status: &'static str| {
			let storage = storage.clone();
			async move { storage.scan_tasks(&record(json!({ "status": status })), 10, None, ctx()).await.items }
		};
		assert_eq!(scan("waiting").await, vec![record::<TaskRecord>(waiting_task.clone())]);
		assert_eq!(scan("completing").await, vec![record::<TaskRecord>(completing_task.clone())]);
		assert_eq!(scan("pending").await.iter().map(|task| task.id.0).collect::<Vec<_>>(), vec![owner]);
		let terminal = json!({
			"id": completing, "conversationId": root.0, "kind": "test.task", "version": 1,
			"input": { "value": completing },
			"state": { "status": "terminal", "outcome": { "status": "failed", "error": { "message": "held" } } },
			"background": false, "abortRequested": false,
		});
		commit_ok(&storage, vec![json!({ "type": "task", "value": terminal.clone() })]).await;
		assert_eq!(scan("completing").await, vec![]);
		assert_eq!(scan("terminal").await, vec![record::<TaskRecord>(terminal)]);
	})
}

fn case_request_ids(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let second_conversation = storage.mint_id().await;
		commit_ok(&storage, vec![json!({ "type": "conversation", "value": conversation_json(second_conversation) })]).await;
		let first = storage.mint_id().await;
		let second = storage.mint_id().await;
		let other_conversation = storage.mint_id().await;
		let submission = |id: u64, conversation: u64, request: &str| {
			json!({
				"id": id, "conversationId": conversation, "requestId": request,
				"type": "input", "status": "queued",
			})
		};
		let first_record = submission(first, root.0, "same");
		let second_record = submission(second, root.0, "other");
		let other_record = submission(other_conversation, second_conversation, "same");
		commit_ok(
			&storage,
			vec![
				json!({ "type": "submission", "value": first_record.clone() }),
				json!({ "type": "submission", "value": second_record.clone() }),
				json!({ "type": "submission", "value": other_record.clone() }),
			],
		)
		.await;
		assert_eq!(storage.submission_by_request(root, "same", ctx()).await, Some(record(first_record.clone())));
		assert_eq!(
			storage.submission_by_request(ConversationId(second_conversation), "same", ctx()).await,
			Some(record(other_record.clone()))
		);

		let placed_entry = storage.mint_id().await;
		let mut placed_second = second_record.clone();
		placed_second["status"] = json!("placed");
		placed_second["entry"] = json!(placed_entry);
		commit_ok(&storage, vec![json!({ "type": "submission", "value": placed_second.clone() })]).await;
		assert_eq!(storage.submission(SubmissionId(second), ctx()).await, Some(record(placed_second.clone())));
		assert_eq!(storage.submission_by_request(root, "other", ctx()).await, Some(record(placed_second)));

		let scan_all = |query: Value| {
			let storage = storage.clone();
			async move {
				let mut found: Vec<u64> = Vec::new();
				let mut cursor: Option<Cursor> = None;
				loop {
					let query = record::<SubmissionQuery>(query.clone());
					let page = storage.scan_submissions(&query, 1, cursor.as_ref(), ctx()).await;
					found.extend(page.items.iter().map(|record| record.id.0));
					if page.next.is_none() {
						break;
					}
					cursor = page.next;
				}
				found
			}
		};
		assert_eq!(scan_all(json!({})).await, vec![first, second, other_conversation]);
		assert_eq!(scan_all(json!({ "conversationId": root.0 })).await, vec![first, second]);
		assert_eq!(scan_all(json!({ "status": "queued" })).await, vec![first, other_conversation]);
		assert_eq!(scan_all(json!({ "status": "placed" })).await, vec![second]);
		assert_eq!(scan_all(json!({ "conversationId": second_conversation, "status": "queued" })).await, vec![other_conversation]);
		assert_eq!(
			scan_all(json!({ "conversationId": second_conversation, "status": "placed" })).await,
			Vec::<u64>::new()
		);
		assert_eq!(
			storage
			.scan_submissions(&record::<SubmissionQuery>(json!({ "status": "placed" })), 10, None, ctx())
			.await
			.items,
			vec![record::<SubmissionRecord>(json!({
				"id": second, "conversationId": root.0, "requestId": "other",
				"type": "input", "status": "placed", "entry": placed_entry,
			}))]
		);
	})
}

fn case_passive_writes(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let done_id = storage.mint_id().await;
		let failed_id = storage.mint_id().await;
		let queued_done = json!({
			"id": done_id, "conversationId": root.0, "requestId": "passive-done",
			"type": "write", "status": "queued",
		});
		let queued_failed = json!({
			"id": failed_id, "conversationId": root.0, "requestId": "passive-failed",
			"type": "write", "status": "queued",
		});
		commit_ok(
			&storage,
			vec![
				json!({ "type": "submission", "value": queued_done.clone() }),
				json!({ "type": "submission", "value": queued_failed.clone() }),
			],
		)
		.await;
		let entry_id = storage.mint_id().await;
		let mut done = queued_done.clone();
		done["status"] = json!("done");
		done["entry"] = json!(entry_id);
		let mut unanswered = queued_failed.clone();
		unanswered["status"] = json!("unanswered");
		unanswered["reason"] = json!("closed");
		unanswered["detail"] = json!({ "retryable": false });
		commit_ok(
			&storage,
			vec![
				json!({ "type": "submission", "value": done.clone() }),
				json!({ "type": "submission", "value": unanswered.clone() }),
			],
		)
		.await;
		assert_eq!(storage.submission(SubmissionId(done_id), ctx()).await, Some(record(done.clone())));
		assert_eq!(storage.submission_by_request(root, "passive-done", ctx()).await, Some(record(done)));
		assert_eq!(storage.submission(SubmissionId(failed_id), ctx()).await, Some(record(unanswered.clone())));
		assert_eq!(storage.submission_by_request(root, "passive-failed", ctx()).await, Some(record(unanswered)));
	})
}

fn case_rewindable_documents(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let first = storage.mint_id().await;
		let first_record = json!({
			"id": first, "kind": "conversation.notes",
			"scope": { "kind": "conversation", "conversationId": root.0 },
			"history": "rewindable", "fork": "asOf",
		});
		let created_at = commit_ok(
			&storage,
			vec![json!({ "type": "document.create", "record": first_record, "content": {
				"kind": "base", "version": 1, "value": { "items": ["a"], "nested": { "count": 1 } },
			}})],
		)
		.await;
		let changed_at = commit_ok(
			&storage,
			vec![json!({ "type": "document.change", "id": first, "content": {
				"kind": "delta", "version": 1,
				"ops": [["p", ["items"], 1, 0, ["b"]], ["s", ["nested", "count"], 2]],
			}})],
		)
		.await;

		let stored = storage.document(DocumentId(first), at(created_at), ctx()).await.unwrap();
		assert_eq!(stored.version, 1);
		assert_eq!(stored.value, record::<serde_json::Map<String, Value>>(json!({ "items": ["a"], "nested": { "count": 1 } })));
		assert_eq!(stored.deltas_since_base, 0);
		let changed = storage.document(DocumentId(first), at(changed_at), ctx()).await.unwrap();
		assert_eq!(changed.value, record(json!({ "items": ["a", "b"], "nested": { "count": 2 } })));
		assert_eq!(changed.deltas_since_base, 1);
		assert_eq!(
			storage.document(DocumentId(first), DocumentPoint::Current, ctx()).await.unwrap().value,
			record(json!({ "items": ["a", "b"], "nested": { "count": 2 } }))
		);

		let checkpoint_at = commit_ok(
			&storage,
			vec![json!({ "type": "document.change", "id": first, "content": {
				"kind": "base", "version": 2, "value": { "items": ["checkpoint"], "nested": { "count": 3 } },
			}})],
		)
		.await;
		let replaced_at = commit_ok(
			&storage,
			vec![json!({ "type": "document.change", "id": first, "content": {
				"kind": "delta", "version": 2, "ops": [["r", { "items": ["replacement"], "nested": { "count": 4 } }]],
			}})],
		)
		.await;
		let stored = storage.document(DocumentId(first), at(changed_at), ctx()).await.unwrap();
		assert_eq!((stored.version, stored.deltas_since_base), (1, 1));
		assert_eq!(stored.value, record(json!({ "items": ["a", "b"], "nested": { "count": 2 } })));
		let stored = storage.document(DocumentId(first), at(checkpoint_at), ctx()).await.unwrap();
		assert_eq!((stored.version, stored.deltas_since_base), (2, 0));
		assert_eq!(stored.value, record(json!({ "items": ["checkpoint"], "nested": { "count": 3 } })));
		let stored = storage.document(DocumentId(first), at(replaced_at), ctx()).await.unwrap();
		assert_eq!(stored.value, record(json!({ "items": ["replacement"], "nested": { "count": 4 } })));
		assert_eq!(stored.deltas_since_base, 1);
		assert_eq!(storage.document(DocumentId(first), DocumentPoint::Current, ctx()).await.unwrap().deltas_since_base, 1);

		let second = storage.mint_id().await;
		let retired_at = commit_ok(
			&storage,
			vec![
				json!({ "type": "document.create", "record": { "id": second, "kind": first_record["kind"], "scope": first_record["scope"], "history": "rewindable", "fork": "asOf" }, "content": {
					"kind": "base", "version": 1, "value": { "items": ["new"] },
				}}),
				json!({ "type": "document.retire", "id": first }),
				json!({ "type": "document.change", "id": first, "content": {
					"kind": "delta", "version": 2, "ops": [["s", ["retiring"], true]],
				}}),
			],
		)
		.await;
		let address = record::<DocumentAddress>(json!({ "kind": first_record["kind"], "scope": first_record["scope"] }));
		assert_eq!(storage.find_document(&address, at(changed_at), ctx()).await.unwrap().id.0, first);
		let found = storage.find_document(&address, at(retired_at), ctx()).await.unwrap();
		assert_eq!((found.id.0, found.created_at, found.retired_at), (second, retired_at, None));
		assert_eq!(
			ids(
				&storage
					.scan_documents(&record::<DocumentQuery>(json!({ "scope": first_record["scope"], "at": changed_at.0 })), 10, None, ctx())
					.await,
				|record| record.id.0
			),
			vec![first]
		);
		assert_eq!(
			ids(
				&storage
					.scan_documents(&record::<DocumentQuery>(json!({ "scope": first_record["scope"], "at": retired_at.0 })), 10, None, ctx())
					.await,
				|record| record.id.0
			),
			vec![second]
		);
		assert_eq!(storage.document(DocumentId(first), at(retired_at), ctx()).await, None);
		assert_eq!(
			storage.document(DocumentId(second), DocumentPoint::Current, ctx()).await.unwrap().value,
			record(json!({ "items": ["new"] }))
		);
	})
}

fn case_long_tails(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let id = storage.mint_id().await;
		let record_json = json!({
			"id": id, "kind": "conversation.long-tail",
			"scope": { "kind": "conversation", "conversationId": root.0 },
			"history": "rewindable", "fork": "asOf",
		});
		let initial = json!({
			"revision": 0,
			"rows": (0..512).map(|value| json!({ "value": value, "stable": format!("row-{value}") })).collect::<Vec<_>>(),
		});
		let created_at = commit_ok(
			&storage,
			vec![json!({ "type": "document.create", "record": record_json, "content": {
				"kind": "base", "version": 1, "value": initial,
			}})],
		)
		.await;
		let mut before_replacement = initial.clone();
		let mut before_replacement_at = created_at;
		for revision in 1..=24u64 {
			let index = (revision * 17) % 512;
			before_replacement["rows"][index as usize]["value"] = json!(-(revision as i64));
			before_replacement["revision"] = json!(revision);
			before_replacement_at = commit_ok(
				&storage,
				vec![json!({ "type": "document.change", "id": id, "content": {
					"kind": "delta", "version": 1,
					"ops": [["s", ["rows", index, "value"], -(revision as i64)], ["s", ["revision"], revision]],
				}})],
			)
			.await;
		}
		let replacement = json!({
			"revision": 100,
			"rows": (0..512).map(|value| json!({ "value": 10_000 + value, "stable": format!("new-{value}") })).collect::<Vec<_>>(),
		});
		let replacement_snapshot = replacement.clone();
		let replacement_at = commit_ok(
			&storage,
			vec![json!({ "type": "document.change", "id": id, "content": {
				"kind": "delta", "version": 1, "ops": [["r", replacement]],
			}})],
		)
		.await;
		let mut current = replacement_snapshot.clone();
		for revision in 101..=124u64 {
			let index = (revision * 19) % 512;
			current["rows"][index as usize]["value"] = json!(-(revision as i64));
			current["revision"] = json!(revision);
			commit_ok(
				&storage,
				vec![json!({ "type": "document.change", "id": id, "content": {
					"kind": "delta", "version": 1,
					"ops": [["s", ["rows", index, "value"], -(revision as i64)], ["s", ["revision"], revision]],
				}})],
			)
			.await;
		}
		assert_eq!(storage.document(DocumentId(id), at(created_at), ctx()).await.unwrap().value, record(initial));
		assert_eq!(
			storage.document(DocumentId(id), at(before_replacement_at), ctx()).await.unwrap().value,
			record(before_replacement)
		);
		assert_eq!(
			storage.document(DocumentId(id), at(replacement_at), ctx()).await.unwrap().value,
			record(replacement_snapshot)
		);
		assert_eq!(storage.document(DocumentId(id), DocumentPoint::Current, ctx()).await.unwrap().value, record(current));
	})
}

fn case_document_copies(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let child = storage.mint_id().await;
		let second_child = storage.mint_id().await;
		commit_ok(
			&storage,
			vec![
				json!({ "type": "conversation", "value": conversation_json(child) }),
				json!({ "type": "conversation", "value": conversation_json(second_child) }),
			],
		)
		.await;
		let source = storage.mint_id().await;
		let source_record = json!({
			"id": source, "kind": "copy.source",
			"scope": { "kind": "conversation", "conversationId": root.0 },
			"history": "rewindable", "fork": "asOf",
		});
		let created_at = commit_ok(
			&storage,
			vec![json!({ "type": "document.create", "record": source_record, "content": {
				"kind": "base", "version": 2, "value": { "count": 1, "rows": [{ "value": "base" }] },
			}})],
		)
		.await;
		commit_ok(
			&storage,
			vec![json!({ "type": "document.change", "id": source, "content": {
				"kind": "delta", "version": 2,
				"ops": [["s", ["count"], 2], ["p", ["rows"], 1, 0, [{ "value": "current" }]]],
			}})],
		)
		.await;
		let historical_copy = storage.mint_id().await;
		let current_copy = storage.mint_id().await;
		let retired_copy = storage.mint_id().await;
		let child_record = |id: u64, conversation: u64| {
			json!({
				"id": id, "kind": "copy.source",
				"scope": { "kind": "conversation", "conversationId": conversation },
				"history": "rewindable", "fork": "asOf",
			})
		};
		commit_ok(
			&storage,
			vec![
				json!({ "type": "document.copy", "record": child_record(historical_copy, child), "source": { "id": source, "at": created_at.0 } }),
				json!({ "type": "document.copy", "record": child_record(current_copy, second_child), "source": { "id": source, "at": "current" } }),
				json!({ "type": "document.copy", "record": child_record(retired_copy, root.0), "source": { "id": source, "at": "current" } }),
				json!({ "type": "document.retire", "id": retired_copy }),
			],
		)
		.await;
		let historical = storage.document(DocumentId(historical_copy), DocumentPoint::Current, ctx()).await.unwrap();
		assert_eq!((historical.version, historical.deltas_since_base), (2, 0));
		assert_eq!(historical.value, record(json!({ "count": 1, "rows": [{ "value": "base" }] })));
		let current = storage.document(DocumentId(current_copy), DocumentPoint::Current, ctx()).await.unwrap();
		assert_eq!(current.value, record(json!({ "count": 2, "rows": [{ "value": "base" }, { "value": "current" }] })));
		assert_eq!(storage.document(DocumentId(retired_copy), DocumentPoint::Current, ctx()).await, None);

		commit_ok(
			&storage,
			vec![
				json!({ "type": "document.change", "id": source, "content": {
					"kind": "base", "version": 2, "value": { "count": 99, "rows": [] },
				}}),
				json!({ "type": "document.retire", "id": source }),
			],
		)
		.await;
		assert_eq!(
			storage.document(DocumentId(current_copy), DocumentPoint::Current, ctx()).await.unwrap().value,
			record(json!({ "count": 2, "rows": [{ "value": "base" }, { "value": "current" }] }))
		);

		let latest_source = storage.mint_id().await;
		let latest_copy = storage.mint_id().await;
		commit_ok(
			&storage,
			vec![json!({ "type": "document.create", "record": {
				"id": latest_source, "kind": "copy.latest",
				"scope": { "kind": "conversation", "conversationId": root.0 },
				"history": "latest", "fork": "current",
			}, "content": { "kind": "base", "version": 4, "value": { "retained": "copy" } }})],
		)
		.await;
		commit_ok(
			&storage,
			vec![json!({ "type": "document.copy", "record": {
				"id": latest_copy, "kind": "copy.latest",
				"scope": { "kind": "conversation", "conversationId": child },
				"history": "latest", "fork": "current",
			}, "source": { "id": latest_source, "at": "current" } })],
		)
		.await;
		commit_ok(
			&storage,
			vec![
				json!({ "type": "document.change", "id": latest_source, "content": {
					"kind": "base", "version": 4, "value": { "retained": "source-only" },
				}}),
				json!({ "type": "document.retire", "id": latest_source }),
			],
		)
		.await;
		let latest = storage.document(DocumentId(latest_copy), DocumentPoint::Current, ctx()).await.unwrap();
		assert_eq!((latest.version, latest.value), (4, record(json!({ "retained": "copy" }))));

		let conflict = storage.mint_id().await;
		let conflict_error = commit(
			&storage,
			vec![
				json!({ "type": "document.copy", "record": child_record(conflict, child), "source": { "id": current_copy, "at": "current" } }),
				json!({ "type": "document.retire", "id": current_copy }),
			],
		)
		.await
		.expect_err("copy source changed in the same batch");
		let _ = conflict_error; // StorageRejected by construction (upstream asserts error.name)
		assert_eq!(storage.document(DocumentId(conflict), DocumentPoint::Current, ctx()).await, None);
		assert_eq!(
			storage.document(DocumentId(current_copy), DocumentPoint::Current, ctx()).await.unwrap().value,
			record(json!({ "count": 2, "rows": [{ "value": "base" }, { "value": "current" }] }))
		);

		let mismatch = storage.mint_id().await;
		commit(
			&storage,
			vec![json!({ "type": "document.copy", "record": { "id": mismatch, "kind": "copy.mismatch", "scope": { "kind": "conversation", "conversationId": child }, "history": "rewindable", "fork": "asOf" }, "source": { "id": current_copy, "at": "current" } })],
		)
		.await
		.expect_err("copy record mismatch");
		assert_eq!(storage.document(DocumentId(mismatch), DocumentPoint::Current, ctx()).await, None);
	})
}

fn case_version_transitions(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		create_root(&storage).await;
		let id = storage.mint_id().await;
		let record_json = json!({ "id": id, "kind": "session.settings", "scope": { "kind": "session" } });
		commit_ok(
			&storage,
			vec![json!({ "type": "document.create", "record": record_json, "content": {
				"kind": "base", "version": 1, "value": { "count": 1 },
			}})],
		)
		.await;
		commit_ok(
			&storage,
			vec![json!({ "type": "document.change", "id": id, "content": {
				"kind": "delta", "version": 1, "ops": [["s", ["count"], 2]],
			}})],
		)
		.await;
		let migrated_at = commit_ok(
			&storage,
			vec![json!({ "type": "document.change", "id": id, "content": {
				"kind": "base", "version": 2, "value": { "count": 3 },
			}})],
		)
		.await;
		let stored = storage.document(DocumentId(id), DocumentPoint::Current, ctx()).await.unwrap();
		assert_eq!((stored.version, stored.value), (2, record(json!({ "count": 3 }))));
		// Adapted from upstream rejection (D2): historical reads of a
		// current-only document collapse to `None`.
		assert_eq!(storage.document(DocumentId(id), at(migrated_at), ctx()).await, None);

		commit(
			&storage,
			vec![json!({ "type": "document.change", "id": id, "content": {
				"kind": "delta", "version": 1, "ops": [["s", ["count"], 4]],
			}})],
		)
		.await
		.expect_err("version transition requires a base");
		assert_eq!(storage.document(DocumentId(id), DocumentPoint::Current, ctx()).await.unwrap().value, record(json!({ "count": 3 })));
		commit_ok(&storage, vec![json!({ "type": "document.retire", "id": id })]).await;
		assert_eq!(storage.document(DocumentId(id), DocumentPoint::Current, ctx()).await, None);
	})
}

fn case_address_indexing(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let first = storage.mint_id().await;
		let second = storage.mint_id().await;
		let conversation_document = storage.mint_id().await;
		let task = storage.mint_id().await;
		let task_singleton = storage.mint_id().await;
		let task_family = storage.mint_id().await;
		let task_other_kind = storage.mint_id().await;
		let created_at = commit_ok(
			&storage,
			vec![
				json!({ "type": "task", "value": pending_task_json(task, root.0, "ready") }),
				json!({ "type": "document.create", "record": {
					"id": first, "kind": "cache", "scope": { "kind": "session" }, "key": "__proto__",
				}, "content": { "kind": "base", "version": 1, "value": { "owner": "first" } }}),
				json!({ "type": "document.create", "record": {
					"id": second, "kind": "cache", "scope": { "kind": "session" }, "key": "constructor",
				}, "content": { "kind": "base", "version": 1, "value": { "owner": "second" } }}),
				json!({ "type": "document.create", "record": {
					"id": conversation_document, "kind": "cache",
					"scope": { "kind": "conversation", "conversationId": root.0 },
					"history": "latest", "fork": "current", "key": "__proto__",
				}, "content": { "kind": "base", "version": 1, "value": { "owner": "conversation" } }}),
				json!({ "type": "document.create", "record": {
					"id": task_singleton, "kind": "task.cache", "scope": { "kind": "task", "taskId": task },
				}, "content": { "kind": "base", "version": 1, "value": { "owner": "singleton" } }}),
				json!({ "type": "document.create", "record": {
					"id": task_family, "kind": "task.cache", "scope": { "kind": "task", "taskId": task }, "key": "member",
				}, "content": { "kind": "base", "version": 1, "value": { "owner": "family" } }}),
				json!({ "type": "document.create", "record": {
					"id": task_other_kind, "kind": "task.other", "scope": { "kind": "task", "taskId": task },
				}, "content": { "kind": "base", "version": 1, "value": { "owner": "other" } }}),
			],
		)
		.await;

		let find = |address: Value| {
			let storage = storage.clone();
			async move { storage.find_document(&record::<DocumentAddress>(address), DocumentPoint::Current, ctx()).await.map(|record| record.id.0) }
		};
		assert_eq!(find(json!({ "kind": "cache", "scope": { "kind": "session" }, "key": "__proto__" })).await, Some(first));
		assert_eq!(
			storage
				.scan_documents(&record::<DocumentQuery>(json!({ "scope": { "kind": "session" }, "at": "current" })), 1, None, ctx())
				.await
				.items
				.len(),
			1
		);
		let first_page = storage
			.scan_documents(&record::<DocumentQuery>(json!({ "scope": { "kind": "session" }, "at": "current" })), 1, None, ctx())
			.await;
		let second_page = storage
			.scan_documents(&record::<DocumentQuery>(json!({ "scope": { "kind": "session" }, "at": "current" })), 1, first_page.next.as_ref(), ctx())
			.await;
		let mut seen = first_page.items.iter().map(|record| record.id.0).collect::<Vec<_>>();
		seen.extend(second_page.items.iter().map(|record| record.id.0));
		assert_eq!(seen, vec![first, second]);
		assert_eq!(
			ids(
				&storage
					.scan_documents(
						&record::<DocumentQuery>(json!({ "scope": { "kind": "conversation", "conversationId": root.0 }, "at": "current" })),
						10,
						None,
						ctx(),
					)
					.await,
				|record| record.id.0
			),
			vec![conversation_document]
		);
		assert_eq!(find(json!({ "kind": "task.cache", "scope": { "kind": "task", "taskId": task } })).await, Some(task_singleton));
		assert_eq!(find(json!({ "kind": "task.cache", "scope": { "kind": "task", "taskId": task }, "key": "member" })).await, Some(task_family));
		assert_eq!(
			ids(
				&storage
					.scan_documents(
						&record::<DocumentQuery>(json!({ "scope": { "kind": "task", "taskId": task }, "at": "current", "kind": "task.cache" })),
						10,
						None,
						ctx(),
					)
					.await,
				|record| record.id.0
			),
			vec![task_singleton, task_family]
		);
		// Adapted from upstream rejection (D2): current-only historical read is None.
		assert_eq!(storage.document(DocumentId(task_singleton), at(created_at), ctx()).await, None);
	})
}

fn case_lifecycle_atomic(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let first = storage.mint_id().await;
		let second = storage.mint_id().await;
		let record_json = json!({ "id": first, "kind": "singleton", "scope": { "kind": "session" } });
		commit_ok(
			&storage,
			vec![json!({ "type": "document.create", "record": record_json.clone(), "content": {
				"kind": "base", "version": 1, "value": { "value": 1 },
			}})],
		)
		.await;
		commit(
			&storage,
			vec![
				json!({ "type": "document.create", "record": { "id": second, "kind": "singleton", "scope": { "kind": "session" } }, "content": {
					"kind": "base", "version": 1, "value": { "value": 2 },
				}}),
				json!({ "type": "document.change", "id": first, "content": { "kind": "delta", "version": 1, "ops": [] }}),
			],
		)
		.await
		.expect_err("address already has a current incarnation");
		assert_eq!(storage.document(DocumentId(first), DocumentPoint::Current, ctx()).await.unwrap().value, record(json!({ "value": 1 })));
		assert_eq!(storage.document(DocumentId(second), DocumentPoint::Current, ctx()).await, None);

		let empty = storage.mint_id().await;
		let empty_at = commit_ok(
			&storage,
			vec![
				json!({ "type": "document.create", "record": {
					"id": empty, "kind": "singleton", "key": "empty",
					"scope": { "kind": "conversation", "conversationId": root.0 },
					"history": "rewindable", "fork": "initial",
				}, "content": { "kind": "base", "version": 1, "value": {} }}),
				json!({ "type": "document.retire", "id": empty }),
			],
		)
		.await;
		assert_eq!(storage.document(DocumentId(empty), DocumentPoint::Current, ctx()).await, None);
		assert_eq!(storage.document(DocumentId(empty), at(empty_at), ctx()).await, None);
		assert_eq!(
			storage
				.find_document(
					&record::<DocumentAddress>(json!({ "kind": "singleton", "scope": { "kind": "conversation", "conversationId": root.0 }, "key": "empty" })),
					at(empty_at),
					ctx(),
				)
				.await,
			None
		);
	})
}

fn case_document_rollback(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let task = storage.mint_id().await;
		let submission = storage.mint_id().await;
		let document = storage.mint_id().await;
		let task_record = pending_task_json(task, root.0, "ready");
		let submission_record = json!({
			"id": submission, "conversationId": root.0, "requestId": "atomic",
			"type": "input", "status": "queued",
		});
		let document_record = json!({ "id": document, "kind": "atomic", "scope": { "kind": "session" } });
		let baseline_seq = commit_ok(
			&storage,
			vec![
				json!({ "type": "task", "value": task_record.clone() }),
				json!({ "type": "submission", "value": submission_record.clone() }),
				json!({ "type": "document.create", "record": document_record.clone(), "content": {
					"kind": "base", "version": 1, "value": { "count": 1 },
				}}),
			],
		)
		.await;

		let transient_entry = storage.mint_id().await;
		let conflicting_document = storage.mint_id().await;
		commit(
			&storage,
			vec![
				json!({ "type": "task", "value": {
					"id": task, "conversationId": root.0, "kind": "test.task", "version": 1,
					"input": { "value": task },
					"state": { "status": "running", "checkpoint": { "phase": "effect" } },
					"background": false, "abortRequested": false,
				}}),
				json!({ "type": "submission", "value": {
					"id": submission, "conversationId": root.0, "requestId": "atomic",
					"type": "input", "status": "unanswered", "reason": "failed",
				}}),
				json!({ "type": "entry", "value": entry_json(transient_entry, root.0, "transient", json!({})) }),
				json!({ "type": "document.create", "record": { "id": conflicting_document, "kind": "atomic", "scope": { "kind": "session" } }, "content": {
					"kind": "base", "version": 1, "value": { "count": 2 },
				}}),
			],
		)
		.await
		.expect_err("address already has a current incarnation");

		assert_eq!(storage.task(TaskId(task), ctx()).await, Some(record(task_record.clone())));
		assert_eq!(
			storage.scan_tasks(&record(json!({ "status": "pending" })), 10, None, ctx()).await.items,
			vec![record::<TaskRecord>(task_record)]
		);
		assert_eq!(storage.submission_by_request(root, "atomic", ctx()).await, Some(record(submission_record)));
		assert_eq!(storage.entry(EntryId(transient_entry), ctx()).await, None);
		assert_eq!(storage.document(DocumentId(conflicting_document), DocumentPoint::Current, ctx()).await, None);
		assert_eq!(
			storage
				.find_document(&record::<DocumentAddress>(json!({ "kind": "atomic", "scope": { "kind": "session" } })), DocumentPoint::Current, ctx())
				.await
				.unwrap()
				.id
				.0,
			document
		);
		let after_rollback_seq = commit_ok(
			&storage,
			vec![json!({ "type": "document.change", "id": document, "content": {
				"kind": "delta", "version": 1, "ops": [["s", ["count"], 3]],
			}})],
		)
		.await;
		assert!(after_rollback_seq > baseline_seq);
	})
}

fn case_indexed_identities(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		// D3: lone UTF-16 surrogates are not constructible as Rust strings; two
		// distinct multi-byte sentinels stand in for the upstream pair.
		let first = "k\u{00e9}-\u{1f600}";
		let second = "k\u{00e8}-\u{1f601}";
		let first_task = storage.mint_id().await;
		let second_task = storage.mint_id().await;
		let first_submission = storage.mint_id().await;
		let second_submission = storage.mint_id().await;
		let first_kind_document = storage.mint_id().await;
		let second_kind_document = storage.mint_id().await;
		let first_key_document = storage.mint_id().await;
		let second_key_document = storage.mint_id().await;
		let mut first_task_record = pending_task_json(first_task, root.0, "ready");
		first_task_record["kind"] = json!(first);
		let mut second_task_record = pending_task_json(second_task, root.0, "ready");
		second_task_record["kind"] = json!(second);
		commit_ok(
			&storage,
			vec![
				json!({ "type": "task", "value": first_task_record }),
				json!({ "type": "task", "value": second_task_record }),
				json!({ "type": "submission", "value": {
					"id": first_submission, "conversationId": root.0, "requestId": first,
					"type": "input", "status": "queued",
				}}),
				json!({ "type": "submission", "value": {
					"id": second_submission, "conversationId": root.0, "requestId": second,
					"type": "input", "status": "queued",
				}}),
				json!({ "type": "document.create", "record": { "id": first_kind_document, "kind": first, "scope": { "kind": "session" } }, "content": {
					"kind": "base", "version": 1, "value": { "identity": "first kind" },
				}}),
				json!({ "type": "document.create", "record": { "id": second_kind_document, "kind": second, "scope": { "kind": "session" } }, "content": {
					"kind": "base", "version": 1, "value": { "identity": "second kind" },
				}}),
				json!({ "type": "document.create", "record": { "id": first_key_document, "kind": "family", "key": first, "scope": { "kind": "session" } }, "content": {
					"kind": "base", "version": 1, "value": { "identity": "first key" },
				}}),
				json!({ "type": "document.create", "record": { "id": second_key_document, "kind": "family", "key": second, "scope": { "kind": "session" } }, "content": {
					"kind": "base", "version": 1, "value": { "identity": "second key" },
				}}),
			],
		)
		.await;
		assert_eq!(
			ids(&storage.scan_tasks(&record(json!({ "kind": first })), 10, None, ctx()).await, |task| task.id.0),
			vec![first_task]
		);
		assert_eq!(
			ids(&storage.scan_tasks(&record(json!({ "kind": second })), 10, None, ctx()).await, |task| task.id.0),
			vec![second_task]
		);
		assert_eq!(storage.task(TaskId(first_task), ctx()).await.unwrap().kind, first);
		assert_eq!(storage.task(TaskId(second_task), ctx()).await.unwrap().kind, second);
		assert_eq!(storage.submission_by_request(root, first, ctx()).await.unwrap().request_id.as_deref(), Some(first));
		assert_eq!(storage.submission_by_request(root, first, ctx()).await.unwrap().id.0, first_submission);
		assert_eq!(storage.submission_by_request(root, second, ctx()).await.unwrap().id.0, second_submission);
		assert_eq!(
			storage.find_document(&record::<DocumentAddress>(json!({ "kind": first, "scope": { "kind": "session" } })), DocumentPoint::Current, ctx()).await.unwrap().id.0,
			first_kind_document
		);
		assert_eq!(
			storage.find_document(&record::<DocumentAddress>(json!({ "kind": second, "scope": { "kind": "session" } })), DocumentPoint::Current, ctx()).await.unwrap().id.0,
			second_kind_document
		);
		assert_eq!(
			storage.find_document(&record::<DocumentAddress>(json!({ "kind": "family", "key": first, "scope": { "kind": "session" } })), DocumentPoint::Current, ctx()).await.unwrap().id.0,
			first_key_document
		);
		assert_eq!(
			storage.find_document(&record::<DocumentAddress>(json!({ "kind": "family", "key": second, "scope": { "kind": "session" } })), DocumentPoint::Current, ctx()).await.unwrap().id.0,
			second_key_document
		);
		assert_eq!(
			ids(
				&storage
					.scan_documents(&record::<DocumentQuery>(json!({ "scope": { "kind": "session" }, "at": "current", "kind": first })), 10, None, ctx())
					.await,
				|record| record.id.0
			),
			vec![first_kind_document]
		);
	})
}

fn case_global_namespace(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let root = create_root(&storage).await;
		let explicit_entry = 100;
		commit_ok(&storage, vec![json!({ "type": "entry", "value": entry_json(explicit_entry, root.0, "message", json!({})) })]).await;
		assert_eq!(storage.mint_id().await, 101);
		commit(&storage, vec![json!({ "type": "task", "value": pending_task_json(explicit_entry, root.0, "ready") })])
			.await
			.expect_err("ID already belongs to entry");
		commit_ok(&storage, vec![json!({ "type": "entry", "value": entry_json(9007199254740991, root.0, "last-id", json!({})) })]).await;
		// Adapted from upstream rejection (D2): u64 minting cannot be exhausted,
		// so the next mint continues past JavaScript's safe integer range.
		assert_eq!(storage.mint_id().await, 9007199254740992);
	})
}

fn case_rejects_after_close(storage: Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		create_root(&storage).await;
		storage.close(ctx()).await;
		// Adapted from upstream rejections (D2): reads collapse to `None`.
		assert_eq!(storage.conversation(ROOT_CONVERSATION_ID, ctx()).await, None);
		let error = storage.commit(&[], ctx()).await.expect_err("commit after close");
		assert!(error.message.contains("closed"), "{}", error.message);
	})
}
