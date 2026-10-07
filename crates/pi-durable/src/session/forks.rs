//! Port of `packages/durable/src/session/forks.ts`.
//!
//! Selects the persisted conversation documents one fork copies into the child
//! scope: rewindable `asOf` incarnations from the fork entry's commit and
//! `current` incarnations from the parent's present. Error strings verbatim.
//!
//! Deviations: D1 — the `ForkDocumentCopy` list is a plain `Vec` (upstream
//! array). D2 — storage access runs through the session's synchronous
//! [`StorageBridge`] (session module D1) instead of the async `Storage` trait;
//! selection order and minted-ID usage are unchanged.

use pi_chord::context::Context;

use crate::documents::address_id;
use crate::session::bridge::StorageBridge;
use crate::types::{
	Cursor, ConversationId, DocumentCopySource, DocumentCreate, DocumentFork, DocumentHistory, DocumentId,
	DocumentPoint, DocumentQuery, DocumentRecord, DocumentScope, EntryId, Page,
};

const SCAN_PAGE_SIZE: u64 = 256;

/// Fork initialization policy of the scanned source (upstream `ForkPolicy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForkPolicy {
	AsOf,
	Current,
}

/// One definition-free document copy to create with a forked conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct ForkDocumentCopy {
	pub record: DocumentCreate,
	pub source: DocumentCopySource,
}

/// Select every persisted conversation document copied by one fork
/// (upstream `prepareForkDocumentCopies`).
pub(crate) fn prepare_fork_document_copies(
	bridge: &StorageBridge,
	parent_conversation_id: ConversationId,
	at: EntryId,
	child_conversation_id: ConversationId,
	context: &Context,
) -> Result<Vec<ForkDocumentCopy>, String> {
	let Some((entry, commit_seq)) = bridge.entry_in(parent_conversation_id, at, context) else {
		return Err(format!(
			"Entry {at} is not visible from conversation {parent_conversation_id}"
		));
	};

	let mut copies = Vec::new();
	let mut copied_addresses = std::collections::HashSet::new();
	collect_copies(
		bridge,
		entry.conversation_id,
		DocumentPoint::At(commit_seq),
		ForkPolicy::AsOf,
		child_conversation_id,
		&mut copies,
		&mut copied_addresses,
		context,
	)?;
	collect_copies(
		bridge,
		parent_conversation_id,
		DocumentPoint::Current,
		ForkPolicy::Current,
		child_conversation_id,
		&mut copies,
		&mut copied_addresses,
		context,
	)?;
	Ok(copies)
}

#[allow(clippy::too_many_arguments)]
fn collect_copies(
	bridge: &StorageBridge,
	scope: ConversationId,
	at: DocumentPoint,
	policy: ForkPolicy,
	child_conversation_id: ConversationId,
	copies: &mut Vec<ForkDocumentCopy>,
	copied_addresses: &mut std::collections::HashSet<String>,
	context: &Context,
) -> Result<(), String> {
	let mut cursor: Option<Cursor> = None;
	loop {
		let query = DocumentQuery {
			scope: DocumentScope::Conversation { conversation_id: scope },
			at,
			kind: None,
		};
		let page: Page<DocumentRecord> = bridge.scan_documents(&query, SCAN_PAGE_SIZE, cursor.as_ref(), context);
		for source in &page.items {
			let DocumentScope::Conversation { .. } = source.scope else {
				continue;
			};
			let Some(source_fork) = source.fork else {
				continue;
			};
			let policy_matches = match (source_fork, policy) {
				(DocumentFork::AsOf, ForkPolicy::AsOf) => true,
				(DocumentFork::Current, ForkPolicy::Current) => true,
				_ => false,
			};
			if !policy_matches {
				continue;
			}
			let id = DocumentId::from(bridge.mint_id());
			let history = match source.history {
				Some(DocumentHistory::Latest) => DocumentHistory::Latest,
				_ => DocumentHistory::Rewindable,
			};
			let record = DocumentCreate {
				id,
				kind: source.kind.clone(),
				key: source.key.clone(),
				scope: DocumentScope::Conversation { conversation_id: child_conversation_id },
				history: Some(history),
				fork: Some(source_fork),
			};
			let copy_address = address_id_of_create(&record);
			if copied_addresses.contains(&copy_address) {
				let member = match &record.key {
					Some(key) => format!("{key}/{}", record.kind),
					None => record.kind.clone(),
				};
				return Err(format!("Fork selects multiple source documents for {member}"));
			}
			copied_addresses.insert(copy_address);
			copies.push(ForkDocumentCopy {
				record,
				source: DocumentCopySource { id: source.id, at },
			});
		}
		cursor = page.next;
		if cursor.is_none() {
			return Ok(());
		}
	}
}

/// Stable string identity of one child copy address (upstream `addressId(record)`
/// over the create record's shared address fields).
fn address_id_of_create(record: &DocumentCreate) -> String {
	address_id(&crate::types::DocumentAddress {
		kind: record.kind.clone(),
		scope: record.scope.clone(),
		key: record.key.clone(),
	})
}
