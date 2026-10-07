//! Port of `packages/durable/src/documents.ts`.
//!
//! Upstream's overloaded `defineDoc`/`defineDocFamily` + branded tokens become
//! one erased [`DocDefinition`] plus builder constructors ([`DocToken`]);
//! scope/history/fork semantics, validation, address resolution, and
//! materialization keep the upstream messages verbatim (D1: the TS overloads
//! collapse to explicit constructor parameters instead of runtime argument
//! parsing — `resolve_address` takes the owner and key directly).

use std::sync::Arc;

use pi_chord::delta::Op;
use serde_json::Value;

use crate::types::{
	CheckpointInfo, DocumentAddress, DocumentCreate, DocumentFork, DocumentHistory, DocumentId,
	DocumentRecord, DocumentScope, JsonObject, StoredDocument,
};

/// Which scope a document definition binds to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopeKind {
	Session,
	Conversation,
	Task,
}

impl ScopeKind {
	/// Upstream `scope` discriminator string.
	pub fn as_str(&self) -> &'static str {
		match self {
			ScopeKind::Session => "session",
			ScopeKind::Conversation => "conversation",
			ScopeKind::Task => "task",
		}
	}
}

/// Erased definition shape used by the Session (upstream `AnyDocDefinition`).
#[derive(Clone)]
pub struct DocDefinition {
	/// Stable persisted kind; part of the public protocol.
	pub kind: String,
	/// Positive integer version of the stored value shape.
	pub version: u64,
	/// Ownership scope.
	pub scope: ScopeKind,
	/// History retention (conversation documents only).
	pub history: Option<DocumentHistory>,
	/// Fork initialization policy (conversation documents only).
	pub fork: Option<DocumentFork>,
	/// True for keyed document families.
	pub family: bool,
	/// `initial(seed?)`: builds the value of a new incarnation; singleton documents ignore the seed.
	pub initial: Arc<dyn Fn(Option<&Value>) -> JsonObject + Send + Sync>,
	/// `migrate(value, fromVersion)`: converts a stored older-version value.
	pub migrate: Option<Arc<dyn Fn(JsonObject, u64) -> JsonObject + Send + Sync>>,
	/// Return true to store this ordinary change as a complete base instead of a delta.
	pub checkpoint_when: Option<Arc<dyn Fn(&JsonObject, &[Op], CheckpointInfo) -> bool + Send + Sync>>,
}

impl std::fmt::Debug for DocDefinition {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("DocDefinition")
			.field("kind", &self.kind)
			.field("version", &self.version)
			.field("scope", &self.scope.as_str())
			.field("history", &self.history)
			.field("fork", &self.fork)
			.field("family", &self.family)
			.finish_non_exhaustive()
	}
}

impl DocDefinition {
	/// Singleton definition in any scope; `initial` takes no seed.
	pub fn singleton(
		kind: impl Into<String>,
		version: u64,
		scope: ScopeKind,
		initial: impl Fn() -> JsonObject + Send + Sync + 'static,
	) -> Self {
		Self {
			kind: kind.into(),
			version,
			scope,
			history: None,
			fork: None,
			family: false,
			initial: Arc::new(move |_| initial()),
			migrate: None,
			checkpoint_when: None,
		}
	}

	/// Conversation singleton with explicit history/fork semantics.
	pub fn conversation(
		kind: impl Into<String>,
		version: u64,
		history: DocumentHistory,
		fork: DocumentFork,
		initial: impl Fn() -> JsonObject + Send + Sync + 'static,
	) -> Self {
		let mut def = Self::singleton(kind, version, ScopeKind::Conversation, initial);
		def.history = Some(history);
		def.fork = Some(fork);
		def
	}

	/// Keyed family definition in any scope; `initial` receives the family seed.
	pub fn family(
		kind: impl Into<String>,
		version: u64,
		scope: ScopeKind,
		initial: impl Fn(&Value) -> JsonObject + Send + Sync + 'static,
	) -> Self {
		Self {
			kind: kind.into(),
			version,
			scope,
			history: None,
			fork: None,
			family: true,
			initial: Arc::new(move |seed| initial(seed.expect("family initial requires a seed"))),
			migrate: None,
			checkpoint_when: None,
		}
	}

	/// Conversation family with explicit history/fork semantics.
	pub fn conversation_family(
		kind: impl Into<String>,
		version: u64,
		history: DocumentHistory,
		fork: DocumentFork,
		initial: impl Fn(&Value) -> JsonObject + Send + Sync + 'static,
	) -> Self {
		let mut def = Self::family(kind, version, ScopeKind::Conversation, initial);
		def.history = Some(history);
		def.fork = Some(fork);
		def
	}

	/// Attach a migration step (builder).
	pub fn with_migrate(
		mut self,
		migrate: impl Fn(JsonObject, u64) -> JsonObject + Send + Sync + 'static,
	) -> Self {
		self.migrate = Some(Arc::new(migrate));
		self
	}

	/// Attach a checkpoint predicate (builder).
	pub fn with_checkpoint_when(
		mut self,
		checkpoint_when: impl Fn(&JsonObject, &[Op], CheckpointInfo) -> bool + Send + Sync + 'static,
	) -> Self {
		self.checkpoint_when = Some(Arc::new(checkpoint_when));
		self
	}

	fn history_fork(&self) -> Option<(DocumentHistory, DocumentFork)> {
		Some((self.history?, self.fork?))
	}
}

/// Typed singleton or family token passed explicitly to typed access (upstream `DocToken` /
/// `DocFamilyToken` erased).
#[derive(Clone, Debug)]
pub struct DocToken(pub Arc<DocDefinition>);

impl DocToken {
	pub fn new(definition: DocDefinition) -> Self {
		Self(Arc::new(definition))
	}

	pub fn definition(&self) -> &DocDefinition {
		&self.0
	}
}

impl From<DocDefinition> for DocToken {
	fn from(definition: DocDefinition) -> Self {
		Self::new(definition)
	}
}

/// Error raised by typed document access; messages match upstream verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocumentAccessError {
	/// Upstream `TypeError`.
	Type(String),
	/// Upstream `Error`.
	Other(String),
}

impl std::fmt::Display for DocumentAccessError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			DocumentAccessError::Type(message) | DocumentAccessError::Other(message) => f.write_str(message),
		}
	}
}

impl std::error::Error for DocumentAccessError {}

/// Validate definition invariants (upstream `validateDefinition`).
pub fn validate_definition(definition: &DocDefinition) -> Result<(), DocumentAccessError> {
	if definition.version < 1 {
		return Err(DocumentAccessError::Type(format!(
			"Document {} version must be a positive integer",
			definition.kind
		)));
	}
	Ok(())
}

/// Logical address plus its string identity.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAddress {
	pub address: DocumentAddress,
	/// Stable string identity ([`address_id`]).
	pub id: String,
}

/// Resolve the logical address of one document access (upstream `resolveAddress`; the TS
/// variadic argument scan collapses to explicit owner/key parameters).
pub fn resolve_address(
	definition: &DocDefinition,
	owner: Option<u64>,
	key: Option<&str>,
) -> Result<ResolvedAddress, DocumentAccessError> {
	let scope = match definition.scope {
		ScopeKind::Session => DocumentScope::Session,
		ScopeKind::Conversation => DocumentScope::Conversation {
			conversation_id: owner
				.map(crate::types::ConversationId::from)
				.ok_or_else(|| type_error_owner(definition))?,
		},
		ScopeKind::Task => DocumentScope::Task {
			task_id: owner
				.map(crate::types::TaskId::from)
				.ok_or_else(|| type_error_owner(definition))?,
		},
	};
	let key = key.map(str::to_string);
	let address = DocumentAddress { kind: definition.kind.clone(), scope, key };
	let id = address_id(&address);
	Ok(ResolvedAddress { address, id })
}

fn type_error_owner(definition: &DocDefinition) -> DocumentAccessError {
	DocumentAccessError::Type(format!(
		"Document {} requires a {} ID",
		definition.kind,
		definition.scope.as_str()
	))
}

/// Stable string identity of one logical address (upstream `addressId`).
pub fn address_id(address: &DocumentAddress) -> String {
	let owner = match &address.scope {
		DocumentScope::Session => Value::Null,
		DocumentScope::Conversation { conversation_id } => Value::from(conversation_id.0),
		DocumentScope::Task { task_id } => Value::from(task_id.0),
	};
	let array = serde_json::json!([
		address.kind,
		match &address.scope {
			DocumentScope::Session => "session",
			DocumentScope::Conversation { .. } => "conversation",
			DocumentScope::Task { .. } => "task",
		},
		owner,
		address.key.clone().map(Value::String).unwrap_or(Value::Null),
	]);
	array.to_string()
}

/// Build the storage create record for a new incarnation at an address
/// (upstream `documentCreate`).
pub fn document_create(
	definition: &DocDefinition,
	address: &DocumentAddress,
	id: DocumentId,
) -> DocumentCreate {
	let scope = address.scope.clone();
	match &scope {
		DocumentScope::Conversation { .. } => DocumentCreate {
			id,
			kind: address.kind.clone(),
			key: address.key.clone(),
			scope,
			history: definition.history,
			fork: definition.fork,
		},
		DocumentScope::Session | DocumentScope::Task { .. } => DocumentCreate {
			id,
			kind: address.kind.clone(),
			key: address.key.clone(),
			scope,
			history: None,
			fork: None,
		},
	}
}

/// Either a stored create record or a full record; shared shape of the typed-access checks.
#[derive(Clone, Copy)]
pub enum DocumentRecordRef<'a> {
	Create(&'a DocumentCreate),
	Record(&'a DocumentRecord),
}

impl DocumentRecordRef<'_> {
	fn id(&self) -> DocumentId {
		match self {
			DocumentRecordRef::Create(record) => record.id,
			DocumentRecordRef::Record(record) => record.id,
		}
	}
	fn kind(&self) -> &str {
		match self {
			DocumentRecordRef::Create(record) => &record.kind,
			DocumentRecordRef::Record(record) => &record.kind,
		}
	}
	fn scope(&self) -> &DocumentScope {
		match self {
			DocumentRecordRef::Create(record) => &record.scope,
			DocumentRecordRef::Record(record) => &record.scope,
		}
	}
	fn history_fork(&self) -> Option<(DocumentHistory, DocumentFork)> {
		match self {
			DocumentRecordRef::Create(record) => Some((record.history?, record.fork?)),
			DocumentRecordRef::Record(record) => Some((record.history?, record.fork?)),
		}
	}
}

/// Reject typed access whose token disagrees with the persisted scope, history, or fork
/// semantics (upstream `checkRecordScope`).
pub fn check_record_scope(
	definition: &DocDefinition,
	record: DocumentRecordRef<'_>,
) -> Result<(), DocumentAccessError> {
	let scope_matches = match (definition.scope, record.scope()) {
		(ScopeKind::Session, DocumentScope::Session) => true,
		(ScopeKind::Conversation, DocumentScope::Conversation { .. }) => true,
		(ScopeKind::Task, DocumentScope::Task { .. }) => true,
		_ => false,
	};
	let semantics_match = match (definition.scope, record.scope()) {
		(ScopeKind::Conversation, DocumentScope::Conversation { .. }) => {
			record.history_fork() == definition.history_fork()
		}
		_ => true,
	};
	if !scope_matches || !semantics_match {
		return Err(DocumentAccessError::Type(format!(
			"Document {} ({}) does not match the supplied definition semantics",
			record.id(),
			record.kind()
		)));
	}
	Ok(())
}

/// Reject typed access to a stored version the supplied definition cannot use
/// (upstream `checkRecordVersion`).
pub fn check_record_version(
	definition: &DocDefinition,
	record: DocumentRecordRef<'_>,
	version: u64,
) -> Result<(), DocumentAccessError> {
	if version > definition.version {
		return Err(DocumentAccessError::Other(format!(
			"Document {} ({}) has newer version {} than {}",
			record.id(),
			record.kind(),
			version,
			definition.version
		)));
	}
	if version < definition.version && definition.migrate.is_none() {
		return Err(DocumentAccessError::Other(format!(
			"Document {} ({}) requires migration from version {}",
			record.id(),
			record.kind(),
			version
		)));
	}
	Ok(())
}

/// Validate and materialize a detached stored value for typed access.
pub fn materialize_document(
	definition: &DocDefinition,
	stored: &StoredDocument,
) -> Result<JsonObject, DocumentAccessError> {
	materialize_document_value(definition, DocumentRecordRef::Record(&stored.record), stored.version, &stored.value)
}

/// Validate and materialize one detached value before its first persisted incarnation.
pub fn materialize_document_value(
	definition: &DocDefinition,
	record: DocumentRecordRef<'_>,
	version: u64,
	value: &JsonObject,
) -> Result<JsonObject, DocumentAccessError> {
	check_record_scope(definition, record)?;
	check_record_version(definition, record, version)?;
	if version == definition.version {
		return Ok(value.clone());
	}
	let migrate = definition.migrate.as_ref().expect("checked above");
	Ok(migrate(value.clone(), version))
}

impl From<crate::types::DocumentScope> for ScopeKind {
	fn from(scope: crate::types::DocumentScope) -> Self {
		match scope {
			crate::types::DocumentScope::Session => ScopeKind::Session,
			crate::types::DocumentScope::Conversation { .. } => ScopeKind::Conversation,
			crate::types::DocumentScope::Task { .. } => ScopeKind::Task,
		}
	}
}

impl DocDefinition {
	/// Scope kind derived from this definition.
	pub fn scope_kind(&self) -> ScopeKind {
		self.scope
	}
}
