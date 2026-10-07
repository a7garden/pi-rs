//! Port of `packages/durable/test/session-tables.test.ts` (transaction table
//! reads/writes, ownership validation, task documents).
//!
//! D-deviations:
//! - `expect(...).rejects` assertions inside one transaction become local
//!   checks of the `TxError` returned by the synchronous `Tx` methods.
//! - The strict-JSON `Number.NaN` rejection in `takes ownership of table
//!   JSON` is unrepresentable over `serde_json::Value`; that assertion is
//!   dropped (module header: strict-JSON rejections are unrepresentable).

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::documents::{DocDefinition, DocDefinition as Doc, DocToken, ScopeKind};
use crate::errors::ReadAfterWrite;
use crate::harness::types::AnyTask;
use crate::session::support::{context, create_conversation, document_changes, flush, open_test_session};
use crate::session::transaction::TxError;
use crate::tasks::{define_task, PhaseHandler, TaskDefinition};
use crate::types::Storage as _;
use crate::types::{
	ConversationId, ConversationOwnership, EntryDraft, EntryId, JsonObject, Page, TaskId, TaskOptions,
	TaskOwnership, TaskRecord, TaskState, TaskStateStatus,
};

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
		crate::types::DocumentCommitChange::Document { record, conversation_id, version, value, ops } => {
			(record, *conversation_id, *version, value, ops)
		}
		crate::types::DocumentCommitChange::DocumentCopy { .. } => panic!("expected a document change"),
	}
}

fn obj(value: Value) -> JsonObject {
	value.as_object().expect("object literal").clone()
}

/// Upstream `WorkTask`.
fn work_task() -> AnyTask {
	let phases: BTreeMap<String, PhaseHandler> = BTreeMap::new();
	AnyTask::new(TaskDefinition {
		name: "test.work".to_string(),
		version: 1,
		initial: Arc::new(|_| json!({ "phase": "start" })),
		phases: Arc::new(phases),
		abort: Arc::new(|_record, _runtime, _context| Box::pin(async { Ok(()) })),
		migrate: None,
		hooks: None,
	})
}

/// Upstream `ProgressDoc` (task-scope).
fn progress_doc() -> DocToken {
	Doc::singleton("test.progress", 1, ScopeKind::Task, || obj(json!({ "lines": [] }))).into()
}

/// Upstream `StepDoc` (task-scope family).
fn step_doc() -> DocToken {
	Doc::family("test.step", 1, ScopeKind::Task, |_seed| obj(json!({ "lines": [] }))).into()
}

/// Upstream `terminal(task)`.
fn terminal(task: &TaskRecord) -> TaskRecord {
	let mut task = task.clone();
	task.state = TaskState {
		status: TaskStateStatus::Terminal,
		checkpoint: None,
		on: None,
		policy: None,
		outcome: Some(crate::types::TaskOutcome::completed(json!({ "ok": true }))),
	};
	task
}

fn pending_state() -> TaskState {
	TaskState {
		status: TaskStateStatus::Pending,
		checkpoint: Some(json!({ "phase": "start" })),
		on: None,
		policy: None,
		outcome: None,
	}
}

async fn create_task(
	session: &crate::session::session::Session,
	conversation_id: ConversationId,
) -> TaskId {
	session
		.commit(
			move |tx| {
				tx.create_task(
					&work_task(),
					json!({ "path": "a" }),
					TaskOptions {
						ownership: TaskOwnership::Conversation,
						conversation_id: Some(conversation_id),
						background: None,
					},
				)
				.map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("task created")
}

/// Upstream `allows table reads only before the first table write`.
#[tokio::test]
async fn allows_table_reads_only_before_the_first_table_write() {
	let harness = open_test_session();
	let conversation_id = create_conversation(&harness.session).await;
	let notes = DocDefinition::conversation(
		"test.notes",
		1,
		crate::types::DocumentHistory::Rewindable,
		crate::types::DocumentFork::AsOf,
		|| obj(json!({ "text": "" })),
	);
	let token: DocToken = notes.into();
	let token_for_tx = token.clone();
	harness
		.session
		.commit(
			move |tx| -> Result<(), String> {
				let record = tx.conversation(conversation_id).map_err(|e| e.to_string())?;
				assert_eq!(record, Some(crate::types::ConversationRecord {
					id: conversation_id,
					parent: None,
					owner: None,
				}));
				let scanned = tx
					.scan_conversations(&Default::default(), 1, None)
					.map_err(|e| e.to_string())?;
				assert_eq!(scanned.items, vec![crate::types::ConversationRecord {
					id: conversation_id,
					parent: None,
					owner: None,
				}]);
				let tasks = tx
					.scan_tasks(&crate::types::TaskQuery { conversation_id: Some(conversation_id), ..Default::default() }, 10, None)
					.map_err(|e| e.to_string())?;
				assert_eq!(tasks, Page { items: vec![], next: None });
				let entries = tx
					.scan_entries(&crate::types::EntryQuery { conversation_id, min_entry_id: None, max_entry_id: None }, 10, None)
					.map_err(|e| e.to_string())?;
				assert_eq!(entries, Page { items: vec![], next: None });

				tx.append_entry(conversation_id, EntryDraft { kind: Some("note".into()), ..Default::default() })
					.map_err(|e| e.to_string())?;

				let error = tx.conversation(conversation_id).expect_err("read after write");
				assert!(matches!(error, TxError::ReadAfterWrite(_)), "{error}");
				let error = tx.task(TaskId(1)).expect_err("read after write");
				assert_eq!(
					error.to_string(),
					"Tx.task() cannot read tables after the first table write"
				);
				assert!(matches!(
					error,
					TxError::ReadAfterWrite(ReadAfterWrite { .. })
				));
				let error = tx.entry(EntryId(1)).expect_err("read after write");
				assert!(matches!(error, TxError::ReadAfterWrite(_)), "{error}");
				let error = tx
					.scan_conversations(&Default::default(), 10, None)
					.expect_err("read after write");
				assert!(matches!(error, TxError::ReadAfterWrite(_)), "{error}");
				let error = tx
					.scan_entries(&crate::types::EntryQuery { conversation_id, min_entry_id: None, max_entry_id: None }, 10, None)
					.expect_err("read after write");
				assert!(matches!(error, TxError::ReadAfterWrite(_)), "{error}");

				// Document access remains available after table writes.
				let draft = tx.doc(&token_for_tx, Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
				draft.set(&[pi_chord::delta::Seg::key("text")], json!("after write"))
					.map_err(|e| e.to_string())?;
				Ok(())
			},
			&context(),
		)
		.await
		.expect("commit");
	let snapshot = harness
		.session
		.snapshot(&token, Some(conversation_id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("created");
	assert_eq!(*snapshot, obj(json!({ "text": "after write" })));
}

/// Upstream `passes caller-selected limits and cursors through table scans`.
#[tokio::test]
async fn passes_caller_selected_limits_and_cursors_through_table_scans() {
	let harness = open_test_session();
	let mut ids = Vec::new();
	for _ in 0..3 {
		ids.push(create_conversation(&harness.session).await);
	}
	harness
		.session
		.commit(
			move |tx| -> Result<(), String> {
				let first = tx.scan_conversations(&Default::default(), 2, None).map_err(|e| e.to_string())?;
				assert_eq!(
					first.items.iter().map(|record| record.id).collect::<Vec<_>>(),
					ids[..2]
				);
				let cursor = first.next.clone().expect("continuation cursor");
				let second = tx
					.scan_conversations(&Default::default(), 2, Some(&cursor))
					.map_err(|e| e.to_string())?;
				assert_eq!(
					second.items.iter().map(|record| record.id).collect::<Vec<_>>(),
					ids[2..]
				);
				assert!(second.next.is_none());
				Ok(())
			},
			&context(),
		)
		.await
		.expect("commit");
}

/// Upstream `treats synchronous setTask as the first table write`.
#[tokio::test]
async fn treats_synchronous_set_task_as_the_first_table_write() {
	let harness = open_test_session();
	let conversation_id = create_conversation(&harness.session).await;
	let task_id = create_task(&harness.session, conversation_id).await;
	harness
		.session
		.commit_with(
			move |tx| -> Result<(), String> {
				let task = tx.task(task_id).map_err(|e| e.to_string())?.expect("task");
				tx.set_task(task).map_err(|e| e.to_string())?;
				let error = tx.task(task_id).expect_err("read after write");
				assert!(matches!(error, TxError::ReadAfterWrite(_)), "{error}");
				Ok(())
			},
			&context(),
			Default::default(),
		)
		.await
		.expect("commit");
}

/// Upstream `creates conversations, entries, and tasks with minted IDs`.
#[tokio::test]
async fn creates_conversations_entries_and_tasks_with_minted_ids() {
	let harness = open_test_session();
	let created = harness
		.session
		.commit(
			move |tx| -> Result<_, String> {
				let conversation =
					tx.create_conversation(ConversationOwnership::Ownerless).map_err(|e| e.to_string())?;
				let first = tx
					.append_entry(
						conversation.id,
						EntryDraft { kind: Some("note".into()), data: Some(json!("one")), ..Default::default() },
					)
					.map_err(|e| e.to_string())?;
				let headed = tx
					.append_entry(
						conversation.id,
						EntryDraft {
							kind: Some("summary".into()),
							head: Some(crate::types::EntryHead::SelfMarker(crate::types::SelfMarker)),
							..Default::default()
						},
					)
					.map_err(|e| e.to_string())?;
				let task = tx
					.create_task(
						&work_task(),
						json!({ "path": "x" }),
						TaskOptions {
							ownership: TaskOwnership::Conversation,
							conversation_id: Some(conversation.id),
							background: Some(true),
						},
					)
					.map_err(|e| e.to_string())?;
				Ok((conversation, first, headed, task))
			},
			&context(),
		)
		.await
		.expect("commit");
	let (conversation, first, headed, task) = created;
	let ids = [conversation.id.0, first.id.0, headed.id.0, task.0];
	assert_eq!(std::collections::HashSet::<_>::from_iter(ids).len(), 4);
	assert_eq!(headed.head, Some(headed.id));
	assert_eq!(first.conversation_id, conversation.id);
	assert_eq!(first.kind, "note");
	assert_eq!(first.data, Some(json!("one")));

	let stored = harness.storage.entry(headed.id, &context()).await.expect("headed stored").0;
	assert_eq!(stored, headed);
	let stored_task = harness.storage.task(task, &context()).await.expect("task stored");
	assert_eq!(stored_task.kind, "test.work");
	assert_eq!(stored_task.version, 1);
	assert_eq!(stored_task.input, json!({ "path": "x" }));
	assert!(stored_task.background);
	assert!(!stored_task.abort_requested);
	assert_eq!(stored_task.state, pending_state());

	flush().await;
	let publication = harness.publications.lock().last().expect("publication").clone();
	assert_eq!(publication.changes.len(), 4);
	let admitted = harness.storage.admitted_commits().pop().expect("admitted");
	for change in &publication.changes {
		let serialized = serde_json::to_string(change).unwrap();
		assert!(
			admitted.iter().any(|write| serde_json::to_string(write).unwrap() == serialized),
			"change {serialized} not admitted"
		);
	}
	let types: Vec<String> = publication
		.changes
		.iter()
		.map(|change| match change {
			crate::types::CommitChange::Table(table) => match table {
				crate::types::TableCommitChange::Conversation { .. } => "conversation".into(),
				crate::types::TableCommitChange::Entry { .. } => "entry".into(),
				crate::types::TableCommitChange::Task { .. } => "task".into(),
				crate::types::TableCommitChange::Submission { .. } => "submission".into(),
			},
			crate::types::CommitChange::Document(_) => "document".into(),
		})
		.collect();
	for expected in ["conversation", "entry", "entry", "task"] {
		assert!(types.iter().any(|kind| kind == expected), "missing {expected} in {types:?}");
	}
	for change in &publication.changes {
		if let crate::types::CommitChange::Table(crate::types::TableCommitChange::Conversation { value }) = change {
			assert_eq!(value, &conversation);
		}
		if let crate::types::CommitChange::Table(crate::types::TableCommitChange::Entry { value }) = change {
			assert!(value == &first || value == &headed);
		}
	}

	let error = harness
		.session
		.commit(
			|tx| {
				tx.create_task(
					&work_task(),
					json!({ "path": "x" }),
					TaskOptions {
						ownership: TaskOwnership::Conversation,
						conversation_id: None,
						background: None,
					},
				)
				.map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect_err("conversationId required");
	assert!(error.to_string().contains("requires options.conversationId"), "{error}");
	let error = harness
		.session
		.commit(
			|tx| {
				tx.append_entry(ConversationId(12345), EntryDraft { kind: Some("note".into()), ..Default::default() })
					.map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect_err("missing conversation");
	assert!(error.to_string().contains("Conversation 12345 does not exist"), "{error}");
}

/// Upstream `creates a conversation with an explicitly staged task owner`.
#[tokio::test]
async fn creates_a_conversation_with_an_explicitly_staged_task_owner() {
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;
	let (supervisor_id, child) = harness
		.session
		.commit(
			move |tx| -> Result<_, String> {
				let supervisor_id = tx
					.create_task(
						&work_task(),
						json!({ "path": "background" }),
						TaskOptions {
							ownership: TaskOwnership::Conversation,
							conversation_id: Some(parent_id),
							background: Some(true),
						},
					)
					.map_err(|e| e.to_string())?;
				let child = tx
					.create_conversation(ConversationOwnership::Task { task_id: supervisor_id })
					.map_err(|e| e.to_string())?;
				Ok((supervisor_id, child))
			},
			&context(),
		)
		.await
		.expect("commit");

	assert_eq!(
		child.owner,
		Some(crate::types::ConversationOwner { conversation_id: parent_id, task_id: supervisor_id })
	);
	assert_eq!(harness.storage.conversation(child.id, &context()).await, Some(child.clone()));

	let error = harness
		.session
		.commit_with(
			move |tx| -> Result<(), String> {
				let mut supervisor = tx.task(supervisor_id).map_err(|e| e.to_string())?.expect("task");
				supervisor.conversation_id = child.id;
				tx.set_task(supervisor).map_err(|e| e.to_string())
			},
			&context(),
			Default::default(),
		)
		.await
		.expect_err("conversation change rejected");
	assert!(error.to_string().contains("cannot change conversations"), "{error}");
	assert_eq!(
		harness.storage.task(supervisor_id, &context()).await.expect("task").conversation_id,
		parent_id
	);
}

/// Upstream `rejects missing, terminal, and abort-marked conversation owners atomically`.
#[tokio::test]
async fn rejects_missing_terminal_and_abort_marked_conversation_owners_atomically() {
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;

	// A staged task moved to another conversation rejects atomically.
	let moved_task_id: Arc<parking_lot::Mutex<Option<TaskId>>> = Arc::default();
	let moved_for_tx = moved_task_id.clone();
	let move_error = harness
		.session
		.commit_with(
			move |tx| -> Result<(), String> {
				let task_id = tx
					.create_task(
						&work_task(),
						json!({ "path": "move" }),
						TaskOptions {
							ownership: TaskOwnership::Conversation,
							conversation_id: Some(parent_id),
							background: None,
						},
					)
					.map_err(|e| e.to_string())?;
				*moved_for_tx.lock() = Some(task_id);
				let mut record = tx.staged_tasks().into_iter().find(|record| record.id == task_id).expect("staged");
				record.conversation_id = ConversationId(998);
				tx.set_task(record).map_err(|e| e.to_string())
			},
			&context(),
			Default::default(),
		)
		.await
		.expect_err("move rejected");
	assert!(move_error.to_string().contains("cannot change conversations"), "{move_error}");
	let moved_task_id = moved_task_id.lock().expect("staged task id captured");
	assert!(harness.storage.task(moved_task_id, &context()).await.is_none());

	let missing_task_id = TaskId(999);
	let error = harness
		.session
		.commit(
			move |tx| {
				tx.create_conversation(ConversationOwnership::Task { task_id: missing_task_id })
					.map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect_err("missing owner rejected");
	assert!(error.to_string().contains("Conversation owner task 999 does not exist"), "{error}");

	// Abort-marked owner.
	let abort_child = harness
		.session
		.commit_with(
			move |tx| -> Result<ConversationId, String> {
				let supervisor_id = tx
					.create_task(
						&work_task(),
						json!({ "path": "aborting" }),
						TaskOptions {
							ownership: TaskOwnership::Conversation,
							conversation_id: Some(parent_id),
							background: None,
						},
					)
					.map_err(|e| e.to_string())?;
				let child = tx
					.create_conversation(ConversationOwnership::Task { task_id: supervisor_id })
					.map_err(|e| e.to_string())?;
				let mut record = tx.staged_tasks().into_iter().find(|record| record.id == supervisor_id).expect("task");
				record.abort_requested = true;
				if let Err(error) = tx.set_task(record) {
					return Err(error.to_string());
				}
				Ok(child.id)
			},
			&context(),
			Default::default(),
		)
		.await
		.expect_err("abort-marked owner rejected");
	assert!(abort_child.to_string().contains("is abort-marked"), "{abort_child}");

	// Terminal owner staged in the same transaction.
	let terminal_error = harness
		.session
		.commit_with(
			move |tx| -> Result<(), String> {
				let supervisor_id = tx
					.create_task(
						&work_task(),
						json!({ "path": "terminal" }),
						TaskOptions {
							ownership: TaskOwnership::Conversation,
							conversation_id: Some(parent_id),
							background: None,
						},
					)
					.map_err(|e| e.to_string())?;
				tx.create_conversation(ConversationOwnership::Task { task_id: supervisor_id })
					.map_err(|e| e.to_string())?;
				let mut record = tx.staged_tasks().into_iter().find(|record| record.id == supervisor_id).expect("task");
				record.state = TaskState {
					status: TaskStateStatus::Terminal,
					checkpoint: None,
					on: None,
					policy: None,
					outcome: Some(crate::types::TaskOutcome::completed(json!({ "ok": true }))),
				};
				tx.set_task(record).map_err(|e| e.to_string())
			},
			&context(),
			Default::default(),
		)
		.await
		.expect_err("terminal owner rejected");
	assert!(terminal_error.to_string().contains("is terminal"), "{terminal_error}");

	// Committed terminal owner.
	let terminal_owner_id = create_task(&harness.session, parent_id).await;
	harness
		.session
		.commit_with(
			move |tx| -> Result<(), String> {
				let task = tx.task(terminal_owner_id).map_err(|e| e.to_string())?.expect("task");
				tx.set_task(terminal(&task)).map_err(|e| e.to_string())
			},
			&context(),
			Default::default(),
		)
		.await
		.expect("terminal settlement");
	let error = harness
		.session
		.commit(
			move |tx| {
				tx.create_conversation(ConversationOwnership::Task { task_id: terminal_owner_id })
					.map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect_err("terminal owner rejected");
	assert!(error.to_string().contains("is terminal"), "{error}");
}

/// Upstream `takes ownership of table JSON and rejects non-strict values`
/// (the `NaN` rejection is dropped — unrepresentable over `serde_json::Value`).
#[tokio::test]
async fn takes_ownership_of_table_json() {
	let harness = open_test_session();
	let conversation_id = create_conversation(&harness.session).await;
	let payload = Arc::new(parking_lot::Mutex::new(json!({ "nested": { "value": 1 } })));
	let payload_for_tx = payload.clone();
	let entry = harness
		.session
		.commit(
			move |tx| -> Result<_, String> {
				let created = tx
					.append_entry(
						conversation_id,
						EntryDraft {
							kind: Some("data".into()),
							data: Some(payload_for_tx.lock().clone()),
							..Default::default()
						},
					)
					.map_err(|e| e.to_string())?;
				payload_for_tx.lock()["nested"]["value"] = json!(2);
				Ok(created)
			},
			&context(),
		)
		.await
		.expect("commit");
	let stored = harness.storage.entry(entry.id, &context()).await.expect("stored").0;
	assert_eq!(stored.data, Some(json!({ "nested": { "value": 1 } })));

	let omitted = harness
		.session
		.commit(
			move |tx| {
				tx.append_entry(
					conversation_id,
					EntryDraft { kind: Some("omitted".into()), data: None, ..Default::default() },
				)
				.map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("commit");
	assert!(omitted.data.is_none());
	assert!(harness.storage.entry(omitted.id, &context()).await.expect("stored").0.data.is_none());
}

/// Upstream `replaces task records completely`.
#[tokio::test]
async fn replaces_task_records_completely() {
	let harness = open_test_session();
	let conversation_id = create_conversation(&harness.session).await;
	let task_id = create_task(&harness.session, conversation_id).await;
	harness
		.session
		.commit_with(
			move |tx| -> Result<(), String> {
				let mut task = tx.task(task_id).map_err(|e| e.to_string())?.expect("task");
				task.state = TaskState {
					status: TaskStateStatus::Running,
					checkpoint: Some(json!({ "phase": "next", "step": 2 })),
					on: None,
					policy: None,
					outcome: None,
				};
				task.memos = Some(obj(json!({ "choice": "b" })));
				tx.set_task(task).map_err(|e| e.to_string())
			},
			&context(),
			Default::default(),
		)
		.await
		.expect("commit");
	let stored = harness.storage.task(task_id, &context()).await.expect("task");
	assert_eq!(stored.state.checkpoint, Some(json!({ "phase": "next", "step": 2 })));
	assert_eq!(stored.state.status, TaskStateStatus::Running);
	assert_eq!(stored.memos, Some(obj(json!({ "choice": "b" }))));
}

/// Upstream `creates a task and then its document in one transaction without ReadAfterWrite`.
#[tokio::test]
async fn creates_a_task_and_then_its_document_in_one_transaction_without_read_after_write() {
	let harness = open_test_session();
	let conversation_id = create_conversation(&harness.session).await;
	let progress = progress_doc();
	let step = step_doc();
	let progress_for_tx = progress.clone();
	let step_for_tx = step.clone();
	let task_id = harness
		.session
		.commit(
			move |tx| -> Result<TaskId, String> {
				let task_id = tx
					.create_task(
						&work_task(),
						json!({ "path": "a" }),
						TaskOptions {
							ownership: TaskOwnership::Conversation,
							conversation_id: Some(conversation_id),
							background: None,
						},
					)
					.map_err(|e| e.to_string())?;
				let progress_draft = tx.doc(&progress_for_tx, Some(task_id.0), None, None).map_err(|e| e.to_string())?;
				progress_draft.push(&[pi_chord::delta::Seg::key("lines")], vec![json!("started")])
					.map_err(|e| e.to_string())?;
				let step_draft =
					tx.doc(&step_for_tx, Some(task_id.0), Some("one"), Some(Value::Null)).map_err(|e| e.to_string())?;
				step_draft.push(&[pi_chord::delta::Seg::key("lines")], vec![json!("step")])
					.map_err(|e| e.to_string())?;
				Ok(task_id)
			},
			&context(),
		)
		.await
		.expect("commit");
	let snapshot = harness
		.session
		.snapshot(&progress, Some(task_id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("created");
	assert_eq!(*snapshot, obj(json!({ "lines": ["created"] })));
	flush().await;
	let publication = harness.publications.lock().last().expect("publication").clone();
	let documents = document_changes(&publication);
	assert!(publication.changes.iter().any(|change| matches!(
		change,
		crate::types::CommitChange::Table(crate::types::TableCommitChange::Task { value })
			if value.id == task_id
	)));
	assert_eq!(documents.len(), 2);
	for document in documents {
		assert_eq!(as_document(document).1, Some(conversation_id));
	}

	harness
		.session
		.commit(
			move |tx| -> Result<(), String> {
				tx.create_conversation(ConversationOwnership::Ownerless).map_err(|e| e.to_string())?;
				let draft = tx.doc(&progress, Some(task_id.0), None, None).map_err(|e| e.to_string())?;
				draft.push(&[pi_chord::delta::Seg::key("lines")], vec![json!("committed task")])
					.map_err(|e| e.to_string())?;
				Ok(())
			},
			&context(),
		)
		.await
		.expect("commit");
	flush().await;
	let publication = harness.publications.lock().last().expect("publication").clone();
	assert_eq!(as_document(document_changes(&publication)[0]).1, Some(conversation_id));
}

/// Upstream `rejects task documents after a terminal candidate`.
#[tokio::test]
async fn rejects_task_documents_after_a_terminal_candidate() {
	let harness = open_test_session();
	let conversation_id = create_conversation(&harness.session).await;
	let progress = progress_doc();
	let step = step_doc();
	let task_id = {
		let progress = progress.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<TaskId, String> {
					let task_id = tx
						.create_task(
							&work_task(),
							json!({ "path": "a" }),
							TaskOptions {
								ownership: TaskOwnership::Conversation,
								conversation_id: Some(conversation_id),
								background: None,
							},
						)
						.map_err(|e| e.to_string())?;
					let draft = tx.doc(&progress, Some(task_id.0), None, None).map_err(|e| e.to_string())?;
					draft.push(&[pi_chord::delta::Seg::key("lines")], vec![json!("started")])
						.map_err(|e| e.to_string())?;
					Ok(task_id)
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	let progress_for_tx = progress.clone();
	harness
		.session
		.commit_with(
			move |tx| -> Result<(), String> {
				let task = tx.task(task_id).map_err(|e| e.to_string())?.expect("task");
				let progress_draft = tx.doc(&progress_for_tx, Some(task_id.0), None, None).map_err(|e| e.to_string())?;
				tx.set_task(terminal(&task)).map_err(|e| e.to_string())?;
				let error = match tx.doc(&progress_for_tx, Some(task_id.0), None, None) {
					Err(error) => error,
					Ok(_) => panic!("terminal task doc accepted"),
				};
				assert_eq!(error.to_string(), format!("Task {task_id} is terminal"));
				let error = match tx.doc(&step, Some(task_id.0), Some("late"), Some(Value::Null)) {
					Err(error) => error,
					Ok(_) => panic!("terminal task doc accepted"),
				};
				assert_eq!(error.to_string(), format!("Task {task_id} is terminal"));
				let error = tx.set_task(task).expect_err("terminal candidate");
				assert!(error.to_string().contains("terminal candidate"), "{error}");
				progress_draft.push(&[pi_chord::delta::Seg::key("lines")], vec![json!("final")])
					.map_err(|e| e.to_string())?;
				Ok(())
			},
			&context(),
			Default::default(),
		)
		.await
		.expect("commit");
	let error = harness
		.session
		.commit(
			move |tx| tx.doc(&progress, Some(task_id.0), None, None).map(|_| ()).map_err(|e| e.to_string()),
			&context(),
		)
		.await
		.expect_err("terminal task doc");
	assert_eq!(error.to_string(), format!("Task {task_id} is terminal"));
}

/// Upstream `retires task documents at terminal settlement, including
/// documents created in the same transaction`.
#[tokio::test]
async fn retires_task_documents_at_terminal_settlement() {
	let harness = open_test_session();
	let conversation_id = create_conversation(&harness.session).await;
	let progress = progress_doc();
	let step = step_doc();
	let task_id = {
		let progress = progress.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<TaskId, String> {
					let task_id = tx
						.create_task(
							&work_task(),
							json!({ "path": "a" }),
							TaskOptions {
								ownership: TaskOwnership::Conversation,
								conversation_id: Some(conversation_id),
								background: None,
							},
						)
						.map_err(|e| e.to_string())?;
					let draft = tx.doc(&progress, Some(task_id.0), None, None).map_err(|e| e.to_string())?;
					draft.push(&[pi_chord::delta::Seg::key("lines")], vec![json!("started")])
						.map_err(|e| e.to_string())?;
					Ok(task_id)
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	harness
		.session
		.commit(
			{
				let step = step.clone();
				move |tx| -> Result<(), String> {
					let draft = tx.doc(&step, Some(task_id.0), Some("committed"), Some(Value::Null))
						.map_err(|e| e.to_string())?;
					draft.push(&[pi_chord::delta::Seg::key("lines")], vec![json!("x")])
						.map_err(|e| e.to_string())?;
					Ok(())
				}
			},
			&context(),
		)
		.await
		.expect("commit");
	flush().await;
	let published = harness.publications.lock().len();
	harness
		.session
		.commit_with(
			{
				let step = step.clone();
				move |tx| -> Result<(), String> {
					let task = tx.task(task_id).map_err(|e| e.to_string())?.expect("task");
					let draft = tx.doc(&step, Some(task_id.0), Some("new"), Some(Value::Null))
						.map_err(|e| e.to_string())?;
					draft.push(&[pi_chord::delta::Seg::key("lines")], vec![json!("created then retired")])
						.map_err(|e| e.to_string())?;
					tx.set_task(terminal(&task)).map_err(|e| e.to_string())
				}
			},
			&context(),
			Default::default(),
		)
		.await
		.expect("commit");
	let writes = harness.storage.commits().pop().expect("writes");
	assert_eq!(writes.len(), 5);
	assert_eq!(
		writes.iter().filter(|write| matches!(write, crate::types::StorageWrite::Task { .. })).count(),
		1
	);
	let creations: Vec<_> = writes
		.iter()
		.filter_map(|write| match write {
			crate::types::StorageWrite::DocumentCreate { record, .. } => Some(record),
			_ => None,
		})
		.collect();
	let retirements: Vec<_> = writes
		.iter()
		.filter_map(|write| match write {
			crate::types::StorageWrite::DocumentRetire { id } => Some(id),
			_ => None,
		})
		.collect();
	assert_eq!(creations.len(), 1);
	assert_eq!(retirements.len(), 3);
	assert!(retirements.contains(&&creations[0].id));
	flush().await;
	{
		let publications = harness.publications.lock();
		assert_eq!(publications.len(), published + 1);
		let publication = publications.last().expect("publication");
		let task_changes = publication
			.changes
			.iter()
			.filter(|change| matches!(change, crate::types::CommitChange::Table(crate::types::TableCommitChange::Task { .. })))
			.count();
		assert_eq!(task_changes, 1);
		let documents = document_changes(publication);
		assert_eq!(documents.len(), 3);
		for document in documents {
			assert_eq!(as_document(document).3, &None);
			assert!(as_document(document).4.is_empty());
			assert_eq!(as_document(document).1, Some(conversation_id));
		}
	}
	assert!(harness.session.snapshot(&progress, Some(task_id.0), None, &context()).await.unwrap().is_none());
	assert!(harness
		.session
		.snapshot(&step, Some(task_id.0), Some("committed"), &context())
		.await
		.unwrap()
		.is_none());
	let alive = harness
		.storage
		.scan_documents(
			&crate::types::DocumentQuery { scope: crate::types::DocumentScope::Task { task_id }, at: crate::types::DocumentPoint::Current, kind: None },
			10,
			None,
			&context(),
		)
		.await;
	assert_eq!(alive.items, vec![]);
	let error = harness
		.session
		.commit_with(
			move |tx| -> Result<(), String> {
				let task = tx.task(task_id).map_err(|e| e.to_string())?.expect("task");
				tx.set_task(terminal(&task)).map_err(|e| e.to_string())
			},
			&context(),
			Default::default(),
		)
		.await
		.expect_err("already terminal");
	assert_eq!(error.to_string(), format!("Task {task_id} is already terminal"));
}

/// Upstream `validates document owners`.
#[tokio::test]
async fn validates_document_owners() {
	let harness = open_test_session();
	let progress = progress_doc();
	let error = harness
		.session
		.commit(
			move |tx| tx.doc(&progress, Some(4242), None, None).map(|_| ()).map_err(|e| e.to_string()),
			&context(),
		)
		.await
		.expect_err("missing task");
	assert!(error.to_string().contains("Task 4242 does not exist"), "{error}");
	let notes: DocToken = DocDefinition::conversation(
		"test.notes",
		1,
		crate::types::DocumentHistory::Rewindable,
		crate::types::DocumentFork::AsOf,
		|| obj(json!({ "text": "" })),
	)
	.into();
	let error = harness
		.session
		.commit(
			move |tx| tx.doc(&notes, Some(4242), None, None).map(|_| ()).map_err(|e| e.to_string()),
			&context(),
		)
		.await
		.expect_err("missing conversation");
	assert!(error.to_string().contains("Conversation 4242 does not exist"), "{error}");
}

// `define_task` is re-exported for parity with upstream; the tests build the
// erased definition directly.
#[allow(unused_imports)]
use define_task as _define_task_unused;
