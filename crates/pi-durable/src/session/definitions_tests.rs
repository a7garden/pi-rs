//! Port of `packages/durable/test/session-definitions.test.ts` (document
//! definition construction, addressing, and typed access).
//!
//! D-deviations:
//! - The TS test is mostly compile-time (`expectTypeOf`); the Rust surface has
//!   one erased `(token, owner: Option<u64>, key: Option<&str>, seed)` shape
//!   (session D5), so the overload typing cases become runtime smoke calls and
//!   the `@ts-expect-error` cases become rejection assertions where the
//!   runtime still rejects (missing owner). Session documents silently ignore
//!   a supplied owner (unrepresentable rejection), and version `1.5` is
//!   unrepresentable over `u64`.

use std::sync::Arc;

use serde_json::json;

use crate::documents::{DocDefinition, DocToken, ScopeKind, validate_definition};
use crate::session::support::{context, create_conversation, open_test_session};
use crate::session::transaction::Tx;
use crate::types::{DocumentFork, DocumentHistory, JsonObject, ROOT_CONVERSATION_ID, TaskId};

fn obj(value: serde_json::Value) -> JsonObject {
	value.as_object().expect("object literal").clone()
}

fn initial() -> JsonObject {
	obj(json!({ "value": 0 }))
}

fn session_doc() -> DocToken {
	DocDefinition::singleton("t.session", 1, ScopeKind::Session, initial).into()
}

fn latest_doc() -> DocToken {
	DocDefinition::conversation("t.latest", 1, DocumentHistory::Latest, DocumentFork::Current, initial).into()
}

fn rewindable_doc() -> DocToken {
	DocDefinition::conversation("t.rewindable", 1, DocumentHistory::Rewindable, DocumentFork::AsOf, initial)
		.into()
}

fn task_doc() -> DocToken {
	DocDefinition::singleton("t.task", 1, ScopeKind::Task, initial).into()
}

fn session_family() -> DocToken {
	DocDefinition::family("t.session-family", 1, ScopeKind::Session, |seed| {
		obj(json!({ "value": seed }))
	})
	.into()
}

fn conversation_family() -> DocToken {
	DocDefinition::conversation_family(
		"t.conversation-family",
		1,
		DocumentHistory::Rewindable,
		DocumentFork::Initial,
		|seed| obj(json!({ "value": seed })),
	)
	.into()
}

fn task_family() -> DocToken {
	DocDefinition::family("t.task-family", 1, ScopeKind::Task, |seed| obj(json!({ "value": seed }))).into()
}

/// Upstream `validates persisted version semantics`.
#[test]
fn validates_persisted_version_semantics() {
	let error = validate_definition(&DocDefinition::singleton("k", 0, ScopeKind::Session, initial))
		.expect_err("version 0 rejected");
	assert!(error.to_string().contains("positive integer"), "{error}");
	// `version: 1.5` is unrepresentable over `u64` (D).
	let empty = DocDefinition::singleton("", 1, ScopeKind::Session, initial);
	assert!(validate_definition(&empty).is_ok());
	assert_eq!(empty.kind, "");
	assert_eq!(session_doc().definition().kind, "t.session");
}

/// Upstream `types every owner, key, and seed overload` — the erased Rust
/// surface keeps one runtime shape, so this exercises every addressing form
/// end to end instead of at compile time. Upstream only type-checks the body
/// (`expectTypeOf`) without running it; the runtime owner validation needs a
/// real task, so the check creates one first (the fabricated `TaskId` of an
/// earlier draft was rejected as nonexistent).
#[tokio::test]
async fn addresses_every_owner_key_and_seed_overload() {
	use crate::harness::types::AnyTask;
	use crate::tasks::{PhaseHandler, TaskDefinition};
	use crate::types::{TaskOptions, TaskOwnership};
	let session = crate::session::create_session(Arc::new(crate::storage::memory::MemoryStorage::new()));
	let conversation_id = create_conversation(&session).await;

	fn work_task() -> AnyTask {
		let phases: std::collections::BTreeMap<String, PhaseHandler> = std::collections::BTreeMap::new();
		AnyTask::new(TaskDefinition {
			name: "test.overloads".to_string(),
			version: 1,
			initial: Arc::new(|_| json!({ "phase": "start" })),
			phases: Arc::new(phases),
			abort: Arc::new(|_record, _runtime, _context| Box::pin(async { Ok(()) })),
			migrate: None,
			hooks: None,
		})
	}

	let check = move |tx: &mut Tx<'_>| -> Result<(), String> {
		let task_id = tx
			.create_task(
				&work_task(),
				json!({}),
				TaskOptions {
					ownership: TaskOwnership::Conversation,
					conversation_id: Some(conversation_id),
					background: None,
				},
			)
			.map_err(|e| e.to_string())?;
		tx.doc(&session_doc(), None, None, None).map(|_| ()).map_err(|e| e.to_string())?;
		tx.doc(&latest_doc(), Some(conversation_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())?;
		tx.doc(&rewindable_doc(), Some(conversation_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())?;
		tx.doc(&task_doc(), Some(task_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())?;
		tx.doc(&session_family(), None, Some("k"), Some(json!(1))).map(|_| ()).map_err(|e| e.to_string())?;
		tx.doc(&conversation_family(), Some(conversation_id.0), Some("k"), Some(json!(1)))
			.map(|_| ()).map_err(|e| e.to_string())?;
		tx.doc(&task_family(), Some(task_id.0), Some("k"), Some(json!(1))).map_err(|e| e.to_string())?;
		tx.retire_doc(&session_doc(), None, None).map_err(|e| e.to_string())?;
		tx.retire_doc(&latest_doc(), Some(conversation_id.0), None).map_err(|e| e.to_string())?;
		tx.retire_doc(&task_doc(), Some(task_id.0), None).map_err(|e| e.to_string())?;
		tx.retire_doc(&session_family(), None, Some("k")).map_err(|e| e.to_string())?;
		tx.retire_doc(&conversation_family(), Some(conversation_id.0), Some("k")).map_err(|e| e.to_string())?;
		tx.retire_doc(&task_family(), Some(task_id.0), Some("k")).map_err(|e| e.to_string())?;
		Ok(())
	};
	session.commit(check, &context()).await.expect("check runs");
}

/// Conversation documents without an owner are rejected at runtime (the
/// upstream `@ts-expect-error` case that the runtime can still express).
#[tokio::test]
async fn rejects_missing_owner_for_conversation_documents() {
	let harness = open_test_session();
	let error = harness
		.session
		.commit(
			|tx| tx.doc(&latest_doc(), None, None, None).map(|_| ()).map_err(|e| e.to_string()),
			&context(),
		)
		.await
		.expect_err("missing owner rejected");
	assert_eq!(error.to_string(), "Document t.latest requires a conversation ID");
	harness.session.close(&context()).await.expect("close");
}

/// Upstream `types historical reads, states, watches, and typed entries` —
/// smoke-runs the historical/state/watch read surface (the typed-entry
/// overloads are covered by the tables and entries suites).
#[tokio::test]
async fn exercises_historical_state_and_watch_reads() {
	let harness = open_test_session();
	let conversation_id = create_conversation(&harness.session).await;
	let task_id = TaskId(1);
	let at = crate::types::EntryId(1);

	let session = &harness.session;
	// Absent reads across every shape; `snapshot_as_of` requires a
	// conversation document.
	assert!(session.snapshot(&rewindable_doc(), Some(conversation_id.0), None, &context()).await.unwrap().is_none());
	assert!(session
		.snapshot_as_of(&rewindable_doc(), conversation_id, None, at, &context())
		.await
		.is_err());
	assert!(session.snapshot(&session_doc(), None, None, &context()).await.unwrap().is_none());
	assert!(session.snapshot(&task_doc(), Some(task_id.0), None, &context()).await.unwrap().is_none());
	assert!(session.snapshot(&session_family(), None, Some("k"), &context()).await.unwrap().is_none());
	assert!(session
		.snapshot(&conversation_family(), Some(ROOT_CONVERSATION_ID.0), Some("k"), &context())
		.await
		.unwrap()
		.is_none());
	assert!(session
		.snapshot(&task_family(), Some(task_id.0), Some("k"), &context())
		.await
		.unwrap()
		.is_none());
	assert!(session.document_state(&session_doc(), None, None, &context()).await.unwrap().is_none());
	assert!(session.watch_doc(&session_doc(), None, None, &context()).await.unwrap().is_none());
	harness.session.close(&context()).await.expect("close");
}
