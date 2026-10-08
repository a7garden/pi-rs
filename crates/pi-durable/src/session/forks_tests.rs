//! Port of `packages/durable/test/session-forks.test.ts` (conversation
//! document fork selection and copies).
//!
//! D-deviations: identity assertions collapse to structural equality where
//! the erased `Value` clones make pointer identity unobservable; otherwise the
//! cases port directly over the synchronous `Tx` surface.

use std::sync::Arc;

use pi_chord::delta::Seg;
use serde_json::{json, Value};

use crate::documents::{DocDefinition, DocToken, ScopeKind};
use crate::session::support::{context, create_conversation, document_changes, document_copy_changes, flush, open_test_session};
use crate::types::Storage as _;
use crate::types::{
	ConversationId, DocumentFork, DocumentHistory, DocumentScope, EntryDraft, EntryId, JsonObject, StorageWrite,
};

fn obj(value: Value) -> JsonObject {
	value.as_object().expect("object literal").clone()
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

fn as_copy(change: &crate::types::DocumentCommitChange) -> (&crate::types::DocumentRecord, ConversationId) {
	match change {
		crate::types::DocumentCommitChange::DocumentCopy { record, conversation_id, .. } => (record, *conversation_id),
		crate::types::DocumentCommitChange::Document { .. } => panic!("expected a document copy change"),
	}
}

fn conversation_doc(kind: &str, history: DocumentHistory, fork: DocumentFork, initial: JsonObject) -> DocToken {
	DocDefinition::conversation(kind, 1, history, fork, move || initial.clone()).into()
}

fn conversation_family_doc(
	kind: &str,
	history: DocumentHistory,
	fork: DocumentFork,
) -> DocToken {
	DocDefinition::conversation_family(kind, 1, history, fork, |seed| obj(json!({ "value": seed }))).into()
}

fn document_creates(writes: &[StorageWrite]) -> Vec<&StorageWrite> {
	writes.iter().filter(|write| matches!(write, StorageWrite::DocumentCreate { .. })).collect()
}

fn document_copies(writes: &[StorageWrite]) -> Vec<&StorageWrite> {
	writes.iter().filter(|write| matches!(write, StorageWrite::DocumentCopy { .. })).collect()
}

fn document_retires(writes: &[StorageWrite]) -> Vec<&StorageWrite> {
	writes.iter().filter(|write| matches!(write, StorageWrite::DocumentRetire { .. })).collect()
}

async fn fork(session: &crate::session::session::Session, parent_id: ConversationId, at: EntryId) -> crate::types::ConversationRecord {
	session
		.commit(
			move |tx| tx.fork_conversation(parent_id, at, crate::types::ConversationOwnership::Ownerless)
				.map_err(|e| e.to_string()),
			&context(),
		)
		.await
		.expect("fork")
}

async fn append(session: &crate::session::session::Session, conversation_id: ConversationId, kind: &str) -> crate::types::EntryRecord {
	let kind = kind.to_string();
	session
		.commit(
			move |tx| {
				tx.append_entry(conversation_id, EntryDraft { kind: Some(kind), ..Default::default() })
					.map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("append")
}

/// Upstream `copies as-of and current singleton and family bases while
/// leaving initial documents absent`.
#[tokio::test]
async fn copies_as_of_and_current_bases_while_leaving_initial_documents_absent() {
	let as_of = conversation_doc(
		"fork.policies.as-of",
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		obj(json!({ "value": "as-initial" })),
	);
	let current = conversation_doc(
		"fork.policies.current",
		DocumentHistory::Latest,
		DocumentFork::Current,
		obj(json!({ "value": "current-initial" })),
	);
	let initial = conversation_doc(
		"fork.policies.initial",
		DocumentHistory::Latest,
		DocumentFork::Initial,
		obj(json!({ "value": "fresh" })),
	);
	let as_of_family = conversation_family_doc(
		"fork.policies.as-of-family",
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
	);
	let current_family = conversation_family_doc(
		"fork.policies.current-family",
		DocumentHistory::Latest,
		DocumentFork::Current,
	);
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;
	// Upstream bundles the fork-point entry and the six at-fork writes in one
	// commit, and the four after-fork writes in a second commit.
	let set_in = |tx: &crate::session::Tx,
	             token: &DocToken,
	             conversation_id: ConversationId,
	             key: Option<&str>,
	             value: &str|
	 -> Result<(), String> {
		// Upstream passes a placeholder seed for family members
		// (`tx.doc(AsOfFamily, parentId, "a", "unused")`); non-family
		// definitions ignore it.
		let draft = tx
			.doc(token, Some(conversation_id.0), key, Some(json!("unused")))
			.map_err(|e| e.to_string())?;
		draft.set(&[Seg::key("value")], json!(value)).map_err(|e| e.to_string())
	};
	let fork_at = {
		let (as_of, current, initial) = (as_of.clone(), current.clone(), initial.clone());
		let (as_of_family, current_family) = (as_of_family.clone(), current_family.clone());
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let fork_at = tx
						.append_entry(parent_id, EntryDraft { kind: Some("fork-point".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					set_in(&tx, &as_of, parent_id, None, "as-at-fork")?;
					set_in(&tx, &current, parent_id, None, "current-at-fork")?;
					set_in(&tx, &initial, parent_id, None, "parent-only")?;
					set_in(&tx, &as_of_family, parent_id, Some("a"), "family-as-a")?;
					set_in(&tx, &as_of_family, parent_id, Some("b"), "family-as-b")?;
					set_in(&tx, &current_family, parent_id, Some("a"), "family-current-a")?;
					Ok(fork_at.id)
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	{
		let (as_of, current) = (as_of.clone(), current.clone());
		let (as_of_family, current_family) = (as_of_family.clone(), current_family.clone());
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					set_in(&tx, &as_of, parent_id, None, "as-after-fork")?;
					set_in(&tx, &current, parent_id, None, "current-when-copied")?;
					set_in(&tx, &as_of_family, parent_id, Some("a"), "family-as-after")?;
					set_in(&tx, &current_family, parent_id, Some("a"), "family-current-when-copied")
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	flush().await;
	let document_reads = harness.storage.document_read_count();
	// Child-side writes stay separate commits (upstream `tx.doc` awaits).
	let set = |token: &DocToken, conversation_id: ConversationId, key: Option<&str>, value: &str| {
		let token = token.clone();
		let key = key.map(str::to_string);
		let value = value.to_string();
		let session = harness.session.clone();
		async move {
			session
				.commit(
					move |tx| -> Result<(), String> {
						let draft = tx.doc(&token, Some(conversation_id.0), key.as_deref(), None)
							.map_err(|e| e.to_string())?;
						draft.set(&[Seg::key("value")], json!(value)).map_err(|e| e.to_string())
					},
					&context(),
				)
				.await
				.expect("commit");
		}
	};

	let child = fork(&harness.session, parent_id, fork_at).await;
	flush().await;
	assert_eq!(harness.storage.document_read_count(), document_reads);

	async fn snapshot(session: &crate::session::session::Session, token: &DocToken, conversation_id: ConversationId, key: Option<&str>) -> Option<JsonObject> {
		session
			.snapshot(token, Some(conversation_id.0), key, &context())
			.await
			.expect("snapshot")
			.map(|value| (*value).clone())
	}
	assert_eq!(snapshot(&harness.session, &as_of, child.id, None).await, Some(obj(json!({ "value": "as-at-fork" }))));
	assert_eq!(
		snapshot(&harness.session, &current, child.id, None).await,
		Some(obj(json!({ "value": "current-when-copied" })))
	);
	assert_eq!(snapshot(&harness.session, &initial, child.id, None).await, None);
	assert_eq!(
		snapshot(&harness.session, &as_of_family, child.id, Some("a")).await,
		Some(obj(json!({ "value": "family-as-a" })))
	);
	assert_eq!(
		snapshot(&harness.session, &as_of_family, child.id, Some("b")).await,
		Some(obj(json!({ "value": "family-as-b" })))
	);
	assert_eq!(
		snapshot(&harness.session, &current_family, child.id, Some("a")).await,
		Some(obj(json!({ "value": "family-current-when-copied" })))
	);

	let admitted = harness.storage.admitted_commits().pop().expect("admitted");
	let copies = document_copies(&admitted);
	assert_eq!(copies.len(), 5);
	for copy in &copies {
		match copy {
			StorageWrite::DocumentCopy { record, .. } => {
				assert_eq!(record.scope, DocumentScope::Conversation { conversation_id: child.id });
			}
			other => panic!("expected copy, got {other:?}"),
		}
	}
	let publication = harness.publications.lock().last().expect("publication").clone();
	let copied = document_copy_changes(&publication);
	assert_eq!(copied.len(), 5);
	for change in &copied {
		let (record, conversation_id) = as_copy(change);
		assert_eq!(record.created_at, publication.seq);
		assert_eq!(conversation_id, child.id);
	}
	let parent_record = harness
		.storage
		.find_document(
			&crate::types::DocumentAddress {
				kind: "fork.policies.as-of".into(),
				scope: DocumentScope::Conversation { conversation_id: parent_id },
				key: None,
			},
			crate::types::DocumentPoint::Current,
			&context(),
		)
		.await
		.expect("parent record");
	let child_record = harness
		.storage
		.find_document(
			&crate::types::DocumentAddress {
				kind: "fork.policies.as-of".into(),
				scope: DocumentScope::Conversation { conversation_id: child.id },
				key: None,
			},
			crate::types::DocumentPoint::Current,
			&context(),
		)
		.await
		.expect("child record");
	assert_ne!(child_record.id, parent_record.id);

	set(&as_of, child.id, None, "child-independent").await;
	assert_eq!(
		snapshot(&harness.session, &as_of, parent_id, None).await,
		Some(obj(json!({ "value": "as-after-fork" })))
	);
	assert_eq!(
		snapshot(&harness.session, &as_of, child.id, None).await,
		Some(obj(json!({ "value": "child-independent" })))
	);
	set(&initial, child.id, None, "child-created").await;
	assert_eq!(
		snapshot(&harness.session, &initial, child.id, None).await,
		Some(obj(json!({ "value": "child-created" })))
	);
}

/// Upstream `uses final document state from the fork entry commit while
/// excluding later same-commit entries`.
#[tokio::test]
async fn uses_final_document_state_from_the_fork_entry_commit() {
	let doc = conversation_doc(
		"fork.same-commit",
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		obj(json!({ "value": "initial" })),
	);
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;
	let (fork_at, excluded) = {
		let doc = doc.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let fork_at = tx
						.append_entry(parent_id, EntryDraft { kind: Some("included".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					let draft = tx.doc(&doc, Some(parent_id.0), None, None).map_err(|e| e.to_string())?;
					draft.set(&[Seg::key("value")], json!("final-state-of-commit")).map_err(|e| e.to_string())?;
					let excluded = tx
						.append_entry(parent_id, EntryDraft { kind: Some("excluded".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					Ok((fork_at.id, excluded.id))
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	let child = fork(&harness.session, parent_id, fork_at).await;
	let snapshot = harness
		.session
		.snapshot(&doc, Some(child.id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("copied");
	assert_eq!(*snapshot, obj(json!({ "value": "final-state-of-commit" })));
	let visible = harness
		.session
		.commit(
			move |tx| {
				tx.scan_entries(
					&crate::types::EntryQuery { conversation_id: child.id, min_entry_id: None, max_entry_id: None },
					10,
					None,
				)
				.map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("scan");
	let ids: Vec<EntryId> = visible.items.iter().map(|entry| entry.id).collect();
	assert!(ids.contains(&fork_at));
	assert!(!ids.contains(&excluded));
}

/// Upstream `selects the entry-owning ancestor for as-of copies and the
/// immediate parent for current copies`.
#[tokio::test]
async fn selects_the_entry_owning_ancestor_for_as_of_copies() {
	let as_of = conversation_doc(
		"fork.ancestry.as-of",
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		obj(json!({ "value": "initial" })),
	);
	let current = conversation_doc(
		"fork.ancestry.current",
		DocumentHistory::Latest,
		DocumentFork::Current,
		obj(json!({ "value": "initial" })),
	);
	let harness = open_test_session();
	let root_id = create_conversation(&harness.session).await;
	let inherited = {
		let as_of = as_of.clone();
		let current = current.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let inherited = tx
						.append_entry(root_id, EntryDraft { kind: Some("root".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					let as_of_draft = tx.doc(&as_of, Some(root_id.0), None, None).map_err(|e| e.to_string())?;
					as_of_draft.set(&[Seg::key("value")], json!("root-at-entry")).map_err(|e| e.to_string())?;
					let current_draft = tx.doc(&current, Some(root_id.0), None, None).map_err(|e| e.to_string())?;
					current_draft.set(&[Seg::key("value")], json!("root-current")).map_err(|e| e.to_string())?;
					Ok(inherited.id)
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	let parent = fork(&harness.session, root_id, inherited).await;
	let parent_entry = {
		let as_of = as_of.clone();
		let current = current.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let entry = tx
						.append_entry(parent.id, EntryDraft { kind: Some("parent".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					let as_of_draft = tx.doc(&as_of, Some(parent.id.0), None, None).map_err(|e| e.to_string())?;
					as_of_draft.set(&[Seg::key("value")], json!("parent-at-own-entry")).map_err(|e| e.to_string())?;
					let current_draft = tx.doc(&current, Some(parent.id.0), None, None).map_err(|e| e.to_string())?;
					current_draft.set(&[Seg::key("value")], json!("parent-current")).map_err(|e| e.to_string())?;
					Ok(entry.id)
				},
				&context(),
			)
			.await
			.expect("commit")
	};

	let inherited_fork = fork(&harness.session, parent.id, inherited).await;
	let value = harness
		.session
		.snapshot(&as_of, Some(inherited_fork.id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("copied");
	assert_eq!(*value, obj(json!({ "value": "root-at-entry" })));
	let value = harness
		.session
		.snapshot(&current, Some(inherited_fork.id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("copied");
	assert_eq!(*value, obj(json!({ "value": "parent-current" })));

	let own_entry_fork = fork(&harness.session, parent.id, parent_entry).await;
	let value = harness
		.session
		.snapshot(&as_of, Some(own_entry_fork.id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("copied");
	assert_eq!(*value, obj(json!({ "value": "parent-at-own-entry" })));
	let value = harness
		.session
		.snapshot(&current, Some(own_entry_fork.id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("copied");
	assert_eq!(*value, obj(json!({ "value": "parent-current" })));
}

/// Upstream `copies the stored value and version without consulting migration
/// definitions or migrated caches`.
#[tokio::test]
async fn copies_the_stored_value_and_version_without_consulting_migrations() {
	let v1 = conversation_doc(
		"fork.stored-version",
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		obj(json!({ "count": 1 })),
	);
	let migrations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
	let v3: DocToken = {
		let migrations = migrations.clone();
		DocDefinition::conversation(
			"fork.stored-version",
			3,
			DocumentHistory::Rewindable,
			DocumentFork::AsOf,
			|| obj(json!({ "count": 0, "migrated": false })),
		)
		.with_migrate(move |value, from_version| {
			assert_eq!(from_version, 1);
			migrations.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
			obj(json!({ "count": value["count"], "migrated": true }))
		})
		.into()
	};
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;
	let fork_at = {
		let v1 = v1.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let point = tx
						.append_entry(parent_id, EntryDraft { kind: Some("point".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					tx.doc(&v1, Some(parent_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())?;
					Ok(point.id)
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	harness.session.unload_documents().await;
	let migrated = harness
		.session
		.snapshot(&v3, Some(parent_id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("migrated");
	assert_eq!(*migrated, obj(json!({ "count": 1, "migrated": true })));
	assert_eq!(migrations.load(std::sync::atomic::Ordering::SeqCst), 1);

	let child = fork(&harness.session, parent_id, fork_at).await;
	assert_eq!(migrations.load(std::sync::atomic::Ordering::SeqCst), 1);
	let child_record = harness
		.storage
		.find_document(
			&crate::types::DocumentAddress {
				kind: "fork.stored-version".into(),
				scope: DocumentScope::Conversation { conversation_id: child.id },
				key: None,
			},
			crate::types::DocumentPoint::Current,
			&context(),
		)
		.await
		.expect("child record");
	let stored = harness
		.storage
		.document(child_record.id, crate::types::DocumentPoint::Current, &context())
		.await
		.expect("stored");
	assert_eq!(stored.version, 1);
	assert_eq!(stored.value, obj(json!({ "count": 1 })));
	let migrated = harness
		.session
		.snapshot(&v3, Some(child.id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("migrated");
	assert_eq!(*migrated, obj(json!({ "count": 1, "migrated": true })));
	assert_eq!(migrations.load(std::sync::atomic::Ordering::SeqCst), 2);
}

/// Upstream `coalesces a typed migration and override into the copied creation base`.
#[tokio::test]
async fn coalesces_a_typed_migration_and_override_into_the_copied_creation_base() {
	let v1 = conversation_doc(
		"fork.override",
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		obj(json!({ "count": 2 })),
	);
	let checkpoints = Arc::new(std::sync::atomic::AtomicUsize::new(0));
	let v2: DocToken = {
		let checkpoints = checkpoints.clone();
		DocDefinition::conversation(
			"fork.override",
			2,
			DocumentHistory::Rewindable,
			DocumentFork::AsOf,
			|| obj(json!({ "count": 0, "migrated": false })),
		)
		.with_migrate(|value, _from| obj(json!({ "count": value["count"], "migrated": true })))
		.with_checkpoint_when(move |_value, _ops, _info| {
			checkpoints.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
			false
		})
		.into()
	};
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;
	let fork_at = {
		let v1 = v1.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let point = tx
						.append_entry(parent_id, EntryDraft { kind: Some("point".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					tx.doc(&v1, Some(parent_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())?;
					Ok(point.id)
				},
				&context(),
			)
			.await
			.expect("commit")
	};

	let document_reads = harness.storage.document_read_count();
	let child = {
		let v2 = v2.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let created = tx
						.fork_conversation(parent_id, fork_at, crate::types::ConversationOwnership::Ownerless)
						.map_err(|e| e.to_string())?;
					let draft = tx.doc(&v2, Some(created.id.0), None, None).map_err(|e| e.to_string())?;
					draft.set(&[Seg::key("count")], json!(9)).map_err(|e| e.to_string())?;
					Ok(created)
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	flush().await;
	assert_eq!(harness.storage.document_read_count(), document_reads + 1);
	let admitted = harness.storage.admitted_commits().pop().expect("admitted");
	let creates = document_creates(&admitted);
	assert_eq!(creates.len(), 1);
	match creates[0] {
		StorageWrite::DocumentCreate { content, .. } => {
			assert_eq!(content.version, 2);
			assert_eq!(content.value, obj(json!({ "count": 9, "migrated": true })));
		}
		other => panic!("expected create, got {other:?}"),
	}
	assert_eq!(checkpoints.load(std::sync::atomic::Ordering::SeqCst), 0);
	let publication = harness.publications.lock().last().expect("publication").clone();
	let changes = document_changes(&publication);
	assert_eq!(changes.len(), 1);
	assert_eq!(as_document(changes[0]).2, Some(2));
	let snapshot = harness
		.session
		.snapshot(&v2, Some(child.id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("copied");
	assert_eq!(*snapshot, obj(json!({ "count": 9, "migrated": true })));
	let parent_record = harness
		.storage
		.find_document(
			&crate::types::DocumentAddress {
				kind: "fork.override".into(),
				scope: DocumentScope::Conversation { conversation_id: parent_id },
				key: None,
			},
			crate::types::DocumentPoint::Current,
			&context(),
		)
		.await
		.expect("parent record");
	let stored = harness
		.storage
		.document(parent_record.id, crate::types::DocumentPoint::Current, &context())
		.await
		.expect("stored");
	assert_eq!(stored.version, 1);
}

/// Upstream `copies the incarnation alive at the fork point across retirement
/// and recreation`.
#[tokio::test]
async fn copies_the_incarnation_alive_at_the_fork_point_across_retirement_and_recreation() {
	let doc = conversation_doc(
		"fork.incarnations",
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		obj(json!({ "value": "initial" })),
	);
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;
	// Upstream bundles each entry append with its write (or retire) in the
	// same commit, so the entry's commit carries that document state.
	let mut entries = Vec::new();
	for (kind, value) in [("old", Some("old")), ("retired", None), ("new", Some("new"))] {
		let doc = doc.clone();
		let entry = harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let entry = tx
						.append_entry(parent_id, EntryDraft { kind: Some(kind.into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					match value {
						Some(value) => {
							let draft = tx.doc(&doc, Some(parent_id.0), None, None).map_err(|e| e.to_string())?;
							draft.set(&[Seg::key("value")], json!(value)).map_err(|e| e.to_string())?;
						}
						None => tx.retire_doc(&doc, Some(parent_id.0), None).map_err(|e| e.to_string())?,
					}
					Ok(entry.id)
				},
				&context(),
			)
			.await
			.expect("commit");
		entries.push(entry);
	}
	let (old_at, retired_at, new_at) = (entries[0], entries[1], entries[2]);
	let old_child = fork(&harness.session, parent_id, old_at).await;
	let empty_child = fork(&harness.session, parent_id, retired_at).await;
	let new_child = fork(&harness.session, parent_id, new_at).await;
	async fn snapshot(session: &crate::session::session::Session, token: &DocToken, conversation_id: ConversationId) -> Option<JsonObject> {
		session
			.snapshot(token, Some(conversation_id.0), None, &context())
			.await
			.expect("snapshot")
			.map(|value| (*value).clone())
	}
	assert_eq!(snapshot(&harness.session, &doc, old_child.id).await, Some(obj(json!({ "value": "old" }))));
	assert_eq!(snapshot(&harness.session, &doc, empty_child.id).await, None);
	assert_eq!(snapshot(&harness.session, &doc, new_child.id).await, Some(obj(json!({ "value": "new" }))));
}

/// Upstream `rejects invisible fork points before admission and remains usable`.
#[tokio::test]
async fn rejects_invisible_fork_points_before_admission_and_remains_usable() {
	let harness = open_test_session();
	let root_id = create_conversation(&harness.session).await;
	let visible = append(&harness.session, root_id, "visible").await.id;
	let hidden = append(&harness.session, root_id, "hidden").await.id;
	let parent = fork(&harness.session, root_id, visible).await;
	flush().await;
	let commits = harness.storage.commits().len();
	let published = harness.publications.lock().len();
	let error = harness
		.session
		.commit(
			move |tx| {
				tx.fork_conversation(parent.id, hidden, crate::types::ConversationOwnership::Ownerless)
					.map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect_err("invisible fork point");
	assert!(error.to_string().contains(&format!("Entry {hidden} is not visible")), "{error}");
	flush().await;
	assert_eq!(harness.storage.commits().len(), commits);
	assert_eq!(harness.publications.lock().len(), published);
	let independent = harness
		.session
		.commit(
			|tx| tx.create_conversation(crate::types::ConversationOwnership::Ownerless).map_err(|e| e.to_string()),
			&context(),
		)
		.await
		.expect("independent commit");
	assert_eq!(harness.storage.conversation(independent.id, &context()).await, Some(independent));
}

/// Upstream `rejects duplicate as-of and current selections for one child
/// address before admission`.
#[tokio::test]
async fn rejects_duplicate_as_of_and_current_selections_for_one_child_address() {
	let as_of = conversation_doc(
		"fork.duplicate-policy",
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		obj(json!({ "value": "as-of" })),
	);
	let current = conversation_doc(
		"fork.duplicate-policy",
		DocumentHistory::Latest,
		DocumentFork::Current,
		obj(json!({ "value": "current" })),
	);
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;
	// Upstream creates the as-of document in the same commit as the "old"
	// entry, so the as-of pass at that entry finds an incarnation.
	let old_at = {
		let as_of = as_of.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let old_at = tx
						.append_entry(parent_id, EntryDraft { kind: Some("old".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					tx.doc(&as_of, Some(parent_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())?;
					Ok(old_at.id)
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
				let as_of = as_of.clone();
				move |tx| tx.retire_doc(&as_of, Some(parent_id.0), None).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("retire");
	harness
		.session
		.commit(
			{
				let current = current.clone();
				move |tx| tx.doc(&current, Some(parent_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect("create current");
	let commits = harness.storage.commits().len();
	let error = harness
		.session
		.commit(
			move |tx| {
				tx.fork_conversation(parent_id, old_at, crate::types::ConversationOwnership::Ownerless)
					.map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect_err("duplicate selection");
	assert!(error.to_string().contains("Fork selects multiple source documents"), "{error}");
	assert_eq!(harness.storage.commits().len(), commits);
	let independent = harness
		.session
		.commit(
			|tx| tx.create_conversation(crate::types::ConversationOwnership::Ownerless).map_err(|e| e.to_string()),
			&context(),
		)
		.await
		.expect("independent commit");
	assert_eq!(harness.storage.conversation(independent.id, &context()).await, Some(independent));
}

/// Upstream `rejects current and as-of source writes in the fork transaction`.
#[tokio::test]
async fn rejects_current_and_as_of_source_writes_in_the_fork_transaction() {
	let current = conversation_doc(
		"fork.same-transaction-current",
		DocumentHistory::Latest,
		DocumentFork::Current,
		obj(json!({ "value": "committed" })),
	);
	let as_of = conversation_doc(
		"fork.same-transaction-as-of",
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		obj(json!({ "value": "committed" })),
	);
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;
	let fork_at = {
		let current = current.clone();
		let as_of = as_of.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let point = tx
						.append_entry(parent_id, EntryDraft { kind: Some("point".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					tx.doc(&current, Some(parent_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())?;
					tx.doc(&as_of, Some(parent_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())?;
					Ok(point.id)
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	let commits = harness.storage.commits().len();
	for write_before in [true, false] {
		let current = current.clone();
		let error = harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					if write_before {
						let draft = tx.doc(&current, Some(parent_id.0), None, None).map_err(|e| e.to_string())?;
						draft.set(&[Seg::key("value")], json!("before-fork")).map_err(|e| e.to_string())?;
					}
					tx.fork_conversation(parent_id, fork_at, crate::types::ConversationOwnership::Ownerless)
						.map_err(|e| e.to_string())?;
					if !write_before {
						let draft = tx.doc(&current, Some(parent_id.0), None, None).map_err(|e| e.to_string())?;
						draft.set(&[Seg::key("value")], json!("after-fork")).map_err(|e| e.to_string())?;
					}
					Ok(())
				},
				&context(),
			)
			.await
			.expect_err("source write rejected");
		assert!(error.to_string().contains("Cannot change fork source document"), "{error}");
	}
	{
		let as_of = as_of.clone();
		let error = harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let draft = tx.doc(&as_of, Some(parent_id.0), None, None).map_err(|e| e.to_string())?;
					draft.set(&[Seg::key("value")], json!("as-of-write")).map_err(|e| e.to_string())?;
					tx.fork_conversation(parent_id, fork_at, crate::types::ConversationOwnership::Ownerless)
						.map_err(|e| e.to_string())?;
					Ok(())
				},
				&context(),
			)
			.await
			.expect_err("as-of source write rejected");
		assert!(error.to_string().contains("Cannot change fork source document"), "{error}");
	}
	assert_eq!(harness.storage.commits().len(), commits);
	let value = harness
		.session
		.snapshot(&current, Some(parent_id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("committed");
	assert_eq!(*value, obj(json!({ "value": "committed" })));
	let child = fork(&harness.session, parent_id, fork_at).await;
	let value = harness
		.session
		.snapshot(&current, Some(child.id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("copied");
	assert_eq!(*value, obj(json!({ "value": "committed" })));
}

/// Upstream `rolls every copied base back when later pre-admission assembly fails`.
#[tokio::test]
async fn rolls_every_copied_base_back_when_later_pre_admission_assembly_fails() {
	let copied = conversation_doc(
		"fork.rollback.copied",
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		obj(json!({ "value": "copied" })),
	);
	let fail_checkpoint = Arc::new(std::sync::atomic::AtomicBool::new(true));
	let failure: DocToken = {
		let fail_checkpoint = fail_checkpoint.clone();
		DocDefinition::singleton("fork.rollback.failure", 1, ScopeKind::Session, || obj(json!({ "count": 0 })))
			.with_checkpoint_when(move |_value, _ops, _info| {
				if fail_checkpoint.load(std::sync::atomic::Ordering::SeqCst) {
					panic!("checkpoint failed");
				}
				false
			})
			.into()
	};
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;
	let fork_at = {
		let copied = copied.clone();
		let failure = failure.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let point = tx
						.append_entry(parent_id, EntryDraft { kind: Some("point".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					tx.doc(&copied, Some(parent_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())?;
					tx.doc(&failure, None, None, None).map(|_| ()).map_err(|e| e.to_string())?;
					Ok(point.id)
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	flush().await;
	let commits = harness.storage.commits().len();
	let published = harness.publications.lock().len();
	let child_id: Arc<parking_lot::Mutex<Option<ConversationId>>> = Arc::default();
	let child_for_tx = child_id.clone();
	let failure_for_tx = failure.clone();
	let error = harness
		.session
		.commit(
			move |tx| -> Result<(), String> {
				let created = tx
					.fork_conversation(parent_id, fork_at, crate::types::ConversationOwnership::Ownerless)
					.map_err(|e| e.to_string())?;
				*child_for_tx.lock() = Some(created.id);
				let draft = tx.doc(&failure_for_tx, None, None, None).map_err(|e| e.to_string())?;
				draft.set(&[Seg::key("count")], json!(1)).map_err(|e| e.to_string())
			},
			&context(),
		)
		.await
		.expect_err("checkpoint failed");
	assert_eq!(error.to_string(), "checkpoint failed");
	flush().await;
	assert_eq!(harness.storage.commits().len(), commits);
	assert_eq!(harness.publications.lock().len(), published);
	assert!(harness.storage.conversation(child_id.lock().expect("child id"), &context()).await.is_none());
	fail_checkpoint.store(false, std::sync::atomic::Ordering::SeqCst);
	{
		let failure = failure.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					let draft = tx.doc(&failure, None, None, None).map_err(|e| e.to_string())?;
					draft.set(&[Seg::key("count")], json!(2)).map_err(|e| e.to_string())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	let value = harness
		.session
		.snapshot(&failure, None, None, &context())
		.await
		.expect("snapshot")
		.expect("created");
	assert_eq!(*value, obj(json!({ "count": 2 })));
	let value = harness
		.session
		.snapshot(&copied, Some(parent_id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("created");
	assert_eq!(*value, obj(json!({ "value": "copied" })));
}

/// Upstream `rolls back a guaranteed Storage rejection without poisoning the Session`.
#[tokio::test]
async fn rolls_back_a_guaranteed_storage_rejection_without_poisoning_the_session() {
	let doc = conversation_doc(
		"fork.storage-rejected",
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		obj(json!({ "value": "source" })),
	);
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;
	let fork_at = {
		let doc = doc.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let point = tx
						.append_entry(parent_id, EntryDraft { kind: Some("point".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					tx.doc(&doc, Some(parent_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())?;
					Ok(point.id)
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	let rejected_child_id: Arc<parking_lot::Mutex<Option<ConversationId>>> = Arc::default();
	let rejected_for_tx = rejected_child_id.clone();
	harness.storage.fail_next_commit("copy rejected");
	let error = harness
		.session
		.commit(
			move |tx| -> Result<(), String> {
				let created = tx
					.fork_conversation(parent_id, fork_at, crate::types::ConversationOwnership::Ownerless)
					.map_err(|e| e.to_string())?;
				*rejected_for_tx.lock() = Some(created.id);
				Ok(())
			},
			&context(),
		)
		.await
		.expect_err("copy rejected");
	assert_eq!(error.to_string(), "copy rejected");
	assert!(harness
		.storage
		.conversation(rejected_child_id.lock().expect("child id"), &context())
		.await
		.is_none());
	let next = harness
		.session
		.commit(
			|tx| tx.create_conversation(crate::types::ConversationOwnership::Ownerless).map_err(|e| e.to_string()),
			&context(),
		)
		.await
		.expect("next commit");
	assert_eq!(harness.storage.conversation(next.id, &context()).await, Some(next));
}

/// Upstream `retires a copied document and can recreate the address in the
/// fork transaction`.
#[tokio::test]
async fn retires_a_copied_document_and_can_recreate_the_address_in_the_fork_transaction() {
	let doc = conversation_doc(
		"fork.retire-recreate",
		DocumentHistory::Rewindable,
		DocumentFork::AsOf,
		obj(json!({ "value": "fresh" })),
	);
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;
	// Upstream appends the fork point and creates the document value in one
	// commit, so the doc is the incarnation alive at the fork point.
	let fork_at = {
		let doc = doc.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let fork_at = tx
						.append_entry(parent_id, EntryDraft { kind: Some("point".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					let draft = tx.doc(&doc, Some(parent_id.0), None, None).map_err(|e| e.to_string())?;
					draft.set(&[Seg::key("value")], json!("copied")).map_err(|e| e.to_string())?;
					Ok(fork_at.id)
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	let child = {
		let doc = doc.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let created = tx
						.fork_conversation(parent_id, fork_at, crate::types::ConversationOwnership::Ownerless)
						.map_err(|e| e.to_string())?;
					tx.retire_doc(&doc, Some(created.id.0), None).map_err(|e| e.to_string())?;
					let draft = tx.doc(&doc, Some(created.id.0), None, None).map_err(|e| e.to_string())?;
					draft.set(&[Seg::key("value")], json!("replacement")).map_err(|e| e.to_string())?;
					Ok(created)
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	let writes = harness.storage.commits().pop().expect("writes");
	assert_eq!(document_copies(&writes).len(), 1);
	assert_eq!(document_creates(&writes).len(), 1);
	assert_eq!(document_retires(&writes).len(), 1);
	let value = harness
		.session
		.snapshot(&doc, Some(child.id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("recreated");
	assert_eq!(*value, obj(json!({ "value": "replacement" })));
}

/// Upstream `copies only conversation documents, leaving Session and task
/// documents in their original scopes`.
#[tokio::test]
async fn copies_only_conversation_documents() {
	let copied = conversation_doc(
		"fork.scope.conversation",
		DocumentHistory::Latest,
		DocumentFork::Current,
		obj(json!({ "value": "conversation" })),
	);
	let session_only: DocToken =
		DocDefinition::singleton("fork.scope.session", 1, ScopeKind::Session, || obj(json!({ "value": "session" })))
			.into();
	let task_only: DocToken =
		DocDefinition::singleton("fork.scope.task", 1, ScopeKind::Task, || obj(json!({ "value": "task" }))).into();
	let work = crate::harness::types::AnyTask::new(crate::tasks::TaskDefinition {
		name: "fork.scope.work".into(),
		version: 1,
		initial: Arc::new(|_| json!({ "phase": "start" })),
		phases: Arc::new(Default::default()),
		abort: Arc::new(|_record, _runtime, _context| Box::pin(async { Ok(()) })),
		migrate: None,
		hooks: None,
	});
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;
	let (fork_at, task_id) = {
		let copied = copied.clone();
		let session_only = session_only.clone();
		let task_only = task_only.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<_, String> {
					let point = tx
						.append_entry(parent_id, EntryDraft { kind: Some("point".into()), ..Default::default() })
						.map_err(|e| e.to_string())?;
					let task_id = tx
						.create_task(
							&work,
							Value::Null,
							crate::types::TaskOptions {
								ownership: crate::types::TaskOwnership::Conversation,
								conversation_id: Some(parent_id),
								background: None,
							},
						)
						.map_err(|e| e.to_string())?;
					tx.doc(&copied, Some(parent_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())?;
					tx.doc(&session_only, None, None, None).map(|_| ()).map_err(|e| e.to_string())?;
					tx.doc(&task_only, Some(task_id.0), None, None).map(|_| ()).map_err(|e| e.to_string())?;
					Ok((point.id, task_id))
				},
				&context(),
			)
			.await
			.expect("commit")
	};
	let child = fork(&harness.session, parent_id, fork_at).await;
	flush().await;
	let writes = harness.storage.commits().pop().expect("writes");
	let copies = document_copies(&writes);
	let kinds: Vec<String> = copies
		.iter()
		.map(|write| match write {
			StorageWrite::DocumentCopy { record, .. } => record.kind.clone(),
			other => panic!("expected copy, got {other:?}"),
		})
		.collect();
	assert_eq!(kinds, vec!["fork.scope.conversation".to_string()]);
	let publication = harness.publications.lock().last().expect("publication").clone();
	let copy_kinds: Vec<String> = document_copy_changes(&publication)
		.iter()
		.map(|change| as_copy(change).0.kind.clone())
		.collect();
	assert_eq!(copy_kinds, vec!["fork.scope.conversation".to_string()]);
	let value = harness
		.session
		.snapshot(&copied, Some(child.id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("copied");
	assert_eq!(*value, obj(json!({ "value": "conversation" })));
	let value = harness
		.session
		.snapshot(&session_only, None, None, &context())
		.await
		.expect("snapshot")
		.expect("created");
	assert_eq!(*value, obj(json!({ "value": "session" })));
	let value = harness
		.session
		.snapshot(&task_only, Some(task_id.0), None, &context())
		.await
		.expect("snapshot")
		.expect("created");
	assert_eq!(*value, obj(json!({ "value": "task" })));
}

/// Upstream `copies every family member across storage scan pages`.
#[tokio::test]
async fn copies_every_family_member_across_storage_scan_pages() {
	let family: DocToken = DocDefinition::conversation_family(
		"fork.pagination",
		1,
		DocumentHistory::Latest,
		DocumentFork::Current,
		|seed| obj(json!({ "value": seed })),
	)
	.into();
	let harness = open_test_session();
	let parent_id = create_conversation(&harness.session).await;
	let fork_at = append(&harness.session, parent_id, "point").await.id;
	{
		let family = family.clone();
		harness
			.session
			.commit(
				move |tx| -> Result<(), String> {
					for index in 0..260 {
						tx.doc(&family, Some(parent_id.0), Some(&format!("member-{index}")), Some(json!(index)))
							.map(|_| ())
							.map_err(|e| e.to_string())?;
					}
					Ok(())
				},
				&context(),
			)
			.await
			.expect("commit");
	}
	let child = fork(&harness.session, parent_id, fork_at).await;
	assert_eq!(document_copies(&harness.storage.commits().pop().expect("writes")).len(), 260);
	for index in [0usize, 259] {
		let value = harness
			.session
			.snapshot(&family, Some(child.id.0), Some(&format!("member-{index}")), &context())
			.await
			.expect("snapshot")
			.expect("copied");
		assert_eq!(*value, obj(json!({ "value": index })));
	}
}

