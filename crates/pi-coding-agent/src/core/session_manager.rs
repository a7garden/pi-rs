//! Port of `packages/coding-agent/src/core/session-manager.ts` — manages
//! conversation sessions as append-only trees stored in JSONL files.
//!
//! Each session entry has an id and parentId forming a tree structure. The
//! "leaf" pointer tracks the current position. Appending creates a child of
//! the current leaf. Branching moves the leaf to an earlier entry, allowing
//! new branches without modifying history.
//!
//! Use [`build_session_context`] (or [`SessionManager::build_session_context`])
//! to get the resolved message list for the LLM, which handles compaction
//! summaries and follows the path from root to current leaf.
//!
//! Wire notes (crate docs, deviation D1): entry/header types are serde
//! structs in upstream interface field order; v1 files (entries without
//! `id`/`parentId`, compactions with `firstKeptEntryIndex`) deserialize with
//! defaults and are repaired by [`migrate_session_entries`], mirroring the
//! upstream mutate-in-place migrations.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use indexmap::{IndexMap, IndexSet};
use pi_agent::AgentMessage;
use pi_ai::types::{AssistantContent, Message, SystemMessage, Usage, UserContent};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{get_agent_dir, get_sessions_dir, APP_NAME};
use crate::core::messages::{
	create_branch_summary_message, create_compaction_summary_message, create_custom_message, custom_message_to_agent,
	empty_user_content,
};
use crate::utils::paths::{join, normalize_path, resolve_path_cwd, PathInputOptions};
use crate::utils::time::{now_ms, parse_timestamp_ms, to_iso_timestamp};

pub const CURRENT_SESSION_VERSION: u32 = 3;

const SESSION_READ_BUFFER_SIZE: usize = 1024 * 1024;
const SESSION_HEADER_READ_BUFFER_SIZE: usize = 4096;
/// Bound synchronous header discovery while allowing large cwd and custom
/// metadata fields.
const MAX_SESSION_HEADER_SCAN_BYTES: usize = 1024 * 1024;

const MAX_CONCURRENT_SESSION_INFO_LOADS: usize = 10;
const MAX_CONCURRENT_SESSION_DISCOVERY_LOADS: usize = 64;
const CURRENT_SESSION_LIST_PUBLISH_INTERVAL: usize = 10;
const ALL_SESSION_LIST_PUBLISH_INTERVAL: usize = 100;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// Literal `"session"` discriminant of [`SessionHeader`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionHeaderType {
	#[serde(rename = "session")]
	Session,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionHeader {
	#[serde(rename = "type")]
	pub header_type: SessionHeaderType,
	/// v1 sessions don't have this.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub version: Option<u32>,
	pub id: String,
	pub timestamp: String,
	pub cwd: String,
	#[serde(rename = "parentSession", default, skip_serializing_if = "Option::is_none")]
	pub parent_session: Option<String>,
}

impl SessionHeader {
	pub fn new(id: impl Into<String>, timestamp: impl Into<String>, cwd: impl Into<String>) -> SessionHeader {
		SessionHeader {
			header_type: SessionHeaderType::Session,
			version: Some(CURRENT_SESSION_VERSION),
			id: id.into(),
			timestamp: timestamp.into(),
			cwd: cwd.into(),
			parent_session: None,
		}
	}
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct NewSessionOptions {
	pub id: Option<String>,
	pub parent_session: Option<String>,
}

/// Shared base fields of every session entry (`SessionEntryBase`). `id` and
/// `parent_id` default on read so v1 files parse; migration fills them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionEntryBase {
	#[serde(default)]
	pub id: String,
	#[serde(default)]
	pub parent_id: Option<String>,
	pub timestamp: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionMessageEntry {
	#[serde(flatten)]
	pub base: SessionEntryBase,
	pub message: AgentMessage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinkingLevelChangeEntry {
	#[serde(flatten)]
	pub base: SessionEntryBase,
	#[serde(rename = "thinkingLevel")]
	pub thinking_level: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelChangeEntry {
	#[serde(flatten)]
	pub base: SessionEntryBase,
	pub provider: String,
	#[serde(rename = "modelId")]
	pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageEntry {
	#[serde(flatten)]
	pub base: SessionEntryBase,
	/// Arbitrary usage category, such as "cache_warm".
	pub kind: String,
	pub provider: String,
	pub model: String,
	pub usage: Usage,
	/// Optional human-readable qualifier for usage notices.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionEntry {
	#[serde(flatten)]
	pub base: SessionEntryBase,
	pub summary: String,
	pub first_kept_entry_id: String,
	pub tokens_before: u64,
	/// Extension-specific data (e.g., ArtifactIndex, version markers for
	/// structured compaction).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub details: Option<Value>,
	/// Usage from the LLM call(s) that generated this summary, if available.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub usage: Option<Usage>,
	/// True if generated by an extension, undefined/false if pi-generated
	/// (backward compatible).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub from_hook: Option<bool>,
	/// Complete prompt and tool state at this compaction boundary.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub system_message: Option<SystemMessage>,
	/// v1 field: index into the entry list; replaced by `firstKeptEntryId`
	/// during migration and deleted.
	#[serde(rename = "firstKeptEntryIndex", default, skip_serializing_if = "Option::is_none")]
	pub first_kept_entry_index: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryEntry {
	#[serde(flatten)]
	pub base: SessionEntryBase,
	pub from_id: String,
	pub summary: String,
	/// Extension-specific data (not sent to LLM).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub details: Option<Value>,
	/// Usage from the LLM call that generated this summary, if available.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub usage: Option<Usage>,
	/// True if generated by an extension, false if pi-generated.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub from_hook: Option<bool>,
}

/// Custom entry for extensions to store extension-specific data in the
/// session (`CustomEntry`). Does NOT participate in LLM context; for
/// injecting content into context, see [`CustomMessageEntry`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomEntry {
	#[serde(flatten)]
	pub base: SessionEntryBase,
	#[serde(rename = "customType")]
	pub custom_type: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub data: Option<Value>,
}

/// Label entry for user-defined bookmarks/markers on entries (`LabelEntry`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LabelEntry {
	#[serde(flatten)]
	pub base: SessionEntryBase,
	pub target_id: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub label: Option<String>,
}

/// Session metadata entry, e.g. user-defined display name
/// (`SessionInfoEntry`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfoEntry {
	#[serde(flatten)]
	pub base: SessionEntryBase,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub name: Option<String>,
}

/// Custom message entry for extensions to inject messages into LLM context
/// (`CustomMessageEntry`). Unlike [`CustomEntry`], this DOES participate in
/// LLM context; `display` controls TUI rendering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessageEntry {
	#[serde(flatten)]
	pub base: SessionEntryBase,
	#[serde(rename = "customType")]
	pub custom_type: String,
	#[serde(default = "empty_user_content")]
	pub content: pi_ai::types::UserContentField,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub details: Option<Value>,
	#[serde(default)]
	pub display: bool,
}

/// Content an append-only context edit may replace without changing message
/// metadata (`ContextEditableContent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContextEditableContent {
	Text(String),
	Blocks(Vec<UserContent>),
	AssistantBlocks(Vec<AssistantContent>),
}

/// `replacement: { content } | null` — `Null` omits the target from model
/// context; `Content` replaces only its content (`ContextEditEntry`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContextEditReplacement {
	Content { content: ContextEditableContent },
	Null,
}

/// Append-only change to one earlier entry's contribution to model context
/// (`ContextEditEntry`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextEditEntry {
	#[serde(flatten)]
	pub base: SessionEntryBase,
	pub target_id: String,
	pub replacement: ContextEditReplacement,
}

/// Session entry — has id/parentId for tree structure (`SessionEntry`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SessionEntry {
	#[serde(rename = "message")]
	Message(SessionMessageEntry),
	#[serde(rename = "thinking_level_change")]
	ThinkingLevelChange(ThinkingLevelChangeEntry),
	#[serde(rename = "model_change")]
	ModelChange(ModelChangeEntry),
	#[serde(rename = "usage")]
	Usage(UsageEntry),
	#[serde(rename = "compaction")]
	Compaction(CompactionEntry),
	#[serde(rename = "branch_summary")]
	BranchSummary(BranchSummaryEntry),
	#[serde(rename = "custom")]
	Custom(CustomEntry),
	#[serde(rename = "custom_message")]
	CustomMessage(CustomMessageEntry),
	#[serde(rename = "context_edit")]
	ContextEdit(ContextEditEntry),
	#[serde(rename = "label")]
	Label(LabelEntry),
	#[serde(rename = "session_info")]
	SessionInfo(SessionInfoEntry),
}

macro_rules! for_each_entry {
	($entry:expr, $payload:ident => $body:expr) => {
		match $entry {
			SessionEntry::Message($payload) => $body,
			SessionEntry::ThinkingLevelChange($payload) => $body,
			SessionEntry::ModelChange($payload) => $body,
			SessionEntry::Usage($payload) => $body,
			SessionEntry::Compaction($payload) => $body,
			SessionEntry::BranchSummary($payload) => $body,
			SessionEntry::Custom($payload) => $body,
			SessionEntry::CustomMessage($payload) => $body,
			SessionEntry::ContextEdit($payload) => $body,
			SessionEntry::Label($payload) => $body,
			SessionEntry::SessionInfo($payload) => $body,
		}
	};
}

impl SessionEntry {
	pub fn base(&self) -> &SessionEntryBase {
		for_each_entry!(self, payload => &payload.base)
	}

	pub fn base_mut(&mut self) -> &mut SessionEntryBase {
		for_each_entry!(self, payload => &mut payload.base)
	}

	pub fn id(&self) -> &str {
		&self.base().id
	}

	pub fn parent_id(&self) -> Option<&str> {
		self.base().parent_id.as_deref()
	}

	pub fn timestamp(&self) -> &str {
		&self.base().timestamp
	}

	/// Upstream `entry.type` discriminant.
	pub fn type_name(&self) -> &'static str {
		match self {
			SessionEntry::Message(_) => "message",
			SessionEntry::ThinkingLevelChange(_) => "thinking_level_change",
			SessionEntry::ModelChange(_) => "model_change",
			SessionEntry::Usage(_) => "usage",
			SessionEntry::Compaction(_) => "compaction",
			SessionEntry::BranchSummary(_) => "branch_summary",
			SessionEntry::Custom(_) => "custom",
			SessionEntry::CustomMessage(_) => "custom_message",
			SessionEntry::ContextEdit(_) => "context_edit",
			SessionEntry::Label(_) => "label",
			SessionEntry::SessionInfo(_) => "session_info",
		}
	}
}

/// Raw file entry (includes header) (`FileEntry`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FileEntry {
	Header(SessionHeader),
	Entry(SessionEntry),
}

impl FileEntry {
	pub fn as_session_header(&self) -> Option<&SessionHeader> {
		match self {
			FileEntry::Header(header) => Some(header),
			FileEntry::Entry(_) => None,
		}
	}

	pub fn as_entry(&self) -> Option<&SessionEntry> {
		match self {
			FileEntry::Header(_) => None,
			FileEntry::Entry(entry) => Some(entry),
		}
	}
}

impl From<SessionEntry> for FileEntry {
	fn from(entry: SessionEntry) -> Self {
		FileEntry::Entry(entry)
	}
}

/// Tree node for getTree() — defensive copy of session structure
/// (`SessionTreeNode`).
#[derive(Debug, Clone, PartialEq)]
pub struct SessionTreeNode {
	pub entry: SessionEntry,
	pub children: Vec<SessionTreeNode>,
	/// Resolved label for this entry, if any.
	pub label: Option<String>,
	/// Timestamp of the latest label change for this entry, if any.
	pub label_timestamp: Option<String>,
}

/// Model reference carried by projections (upstream `{ provider, modelId }`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionModelRef {
	pub provider: String,
	pub model_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedSessionEntry {
	/// Raw append-only entry that owns this projected contribution.
	pub source_entry: SessionEntry,
	/// Model-visible messages after context edits. Empty for state-only
	/// entries and omissions.
	pub messages: Vec<AgentMessage>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionProjection {
	pub entries: Vec<ProjectedSessionEntry>,
	pub messages: Vec<AgentMessage>,
	pub thinking_level: String,
	pub model: Option<SessionModelRef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionContext {
	pub messages: Vec<AgentMessage>,
	pub thinking_level: String,
	pub model: Option<SessionModelRef>,
}

/// One discovered session file (`SessionInfo`). Upstream `created`/`modified`
/// are `Date`s; the port carries unix epoch milliseconds (crate docs D2).
#[derive(Debug, Clone, PartialEq)]
pub struct SessionInfo {
	pub path: String,
	pub id: String,
	/// Working directory where the session was started. Empty string for old
	/// sessions.
	pub cwd: String,
	/// User-defined display name from session_info entries.
	pub name: Option<String>,
	/// Path to the parent session (if this session was forked).
	pub parent_session_path: Option<String>,
	pub created: i64,
	pub modified: i64,
	pub message_count: u64,
	pub first_message: String,
	pub all_messages_text: String,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum SessionManagerError {
	#[error(
		"Session id must be non-empty, contain only alphanumeric characters, '-', '_', and '.', and start and end with an alphanumeric character"
	)]
	InvalidSessionId,
	#[error("Session file is not a valid {app_name} session: {path}")]
	NotASession { app_name: String, path: String },
	#[error("Entry {0} not found")]
	EntryNotFound(String),
	#[error("Entry {0} is not on the active branch")]
	NotOnActiveBranch(String),
	#[error("Entry {0} does not contribute editable model content")]
	NotEditableContent(String),
	#[error("Cannot fork: source session file is empty or invalid: {0}")]
	ForkSourceInvalid(String),
	#[error("Cannot fork: source session has no header: {0}")]
	ForkSourceNoHeader(String),
	#[error("Session header exceeds {0}-byte scan limit: {1}")]
	HeaderScanLimit(usize, String),
	#[error(transparent)]
	Io(#[from] std::io::Error),
}

/// Failure of the async session listing APIs (upstream `AbortError`
/// rejections).
#[derive(Debug, thiserror::Error)]
pub enum SessionListError {
	#[error(transparent)]
	Aborted(pi_ai::utils::abort::AbortReason),
	#[error(transparent)]
	Io(#[from] std::io::Error),
}

impl From<pi_ai::utils::abort::AbortReason> for SessionListError {
	fn from(reason: pi_ai::utils::abort::AbortReason) -> Self {
		SessionListError::Aborted(reason)
	}
}

// ---------------------------------------------------------------------------
// Ids
// ---------------------------------------------------------------------------

fn create_session_id() -> String {
	// uuidv7 clock errors are unreachable for wall-clock timestamps.
	pi_ai::utils::uuid::uuidv7_now().expect("uuidv7 clock")
}

pub fn assert_valid_session_id(id: &str) -> Result<(), SessionManagerError> {
	let valid = !id.is_empty()
		&& id.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
		&& id.chars().next_back().is_some_and(|c| c.is_ascii_alphanumeric())
		&& id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'));
	if valid {
		Ok(())
	} else {
		Err(SessionManagerError::InvalidSessionId)
	}
}

/// `crypto.randomUUID()` equivalent (v4).
fn random_uuid() -> String {
	let mut bytes = [0u8; 16];
	getrandom::getrandom(&mut bytes).expect("os randomness");
	bytes[6] = (bytes[6] & 0x0f) | 0x40;
	bytes[8] = (bytes[8] & 0x3f) | 0x80;
	let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
	format!(
		"{}-{}-{}-{}-{}",
		&hex[0..8],
		&hex[8..12],
		&hex[12..16],
		&hex[16..20],
		&hex[20..32]
	)
}

/// Generate a unique short ID (8 hex chars, collision-checked)
/// (`generateId`).
fn generate_id(is_taken: impl Fn(&str) -> bool) -> String {
	for _ in 0..100 {
		let id = random_uuid()[..8].to_string();
		if !is_taken(&id) {
			return id;
		}
	}
	// Fallback to full UUID if somehow we have collisions
	random_uuid()
}

fn generate_id_for_map(map: &IndexMap<String, SessionEntry>) -> String {
	generate_id(|id| map.contains_key(id))
}

// ---------------------------------------------------------------------------
// Migrations
// ---------------------------------------------------------------------------

/// Migrate v1 → v2: add id/parentId tree structure. Mutates in place
/// (`migrateV1ToV2`).
fn migrate_v1_to_v2(entries: &mut [FileEntry]) {
	let mut ids: HashSet<String> = HashSet::new();
	let mut prev_id: Option<String> = None;

	for entry in entries.iter_mut() {
		match entry {
			FileEntry::Header(header) => {
				header.version = Some(2);
			}
			FileEntry::Entry(session_entry) => {
				let id = generate_id(|id| ids.contains(id));
				session_entry.base_mut().parent_id = prev_id.clone();
				session_entry.base_mut().id = id.clone();
				prev_id = Some(id.clone());
				ids.insert(id);
			}
		}
	}

	// Convert firstKeptEntryIndex to firstKeptEntryId for compaction. Only
	// reads entries earlier in the file, whose ids are already assigned.
	for index in 0..entries.len() {
		let target_index = match &entries[index] {
			FileEntry::Entry(SessionEntry::Compaction(compaction)) => compaction.first_kept_entry_index,
			_ => None,
		};
		let Some(target_index) = target_index else { continue };
		if let FileEntry::Entry(SessionEntry::Compaction(compaction)) = &mut entries[index] {
			compaction.first_kept_entry_index = None;
		}
		let Some(FileEntry::Entry(target)) = entries.get(target_index as usize) else { continue };
		let target_id = target.id().to_string();
		if let FileEntry::Entry(SessionEntry::Compaction(compaction)) = &mut entries[index] {
			compaction.first_kept_entry_id = target_id;
		}
	}
}

/// Migrate v2 → v3: rename hookMessage role to custom. Mutates in place
/// (`migrateV2ToV3`).
fn migrate_v2_to_v3(entries: &mut [FileEntry]) {
	for entry in entries.iter_mut() {
		match entry {
			FileEntry::Header(header) => {
				header.version = Some(3);
			}
			FileEntry::Entry(SessionEntry::Message(message_entry)) => {
				if let AgentMessage::Custom(value) = &mut message_entry.message {
					if value.get("role").and_then(|role| role.as_str()) == Some("hookMessage") {
						if let Some(role) = value.get_mut("role") {
							*role = Value::String("custom".to_string());
						}
					}
				}
			}
			_ => {}
		}
	}
}

/// Run all necessary migrations to bring entries to current version. Mutates
/// entries in place. Returns true if any migration was applied
/// (`migrateToCurrentVersion`).
fn migrate_to_current_version(entries: &mut [FileEntry]) -> bool {
	let version = entries
		.iter()
		.find_map(|entry| entry.as_session_header())
		.and_then(|header| header.version)
		.unwrap_or(1);

	if version >= CURRENT_SESSION_VERSION {
		return false;
	}

	if version < 2 {
		migrate_v1_to_v2(entries);
	}
	if version < 3 {
		migrate_v2_to_v3(entries);
	}

	true
}

/// Exported for testing (`migrateSessionEntries`).
pub fn migrate_session_entries(entries: &mut [FileEntry]) {
	migrate_to_current_version(entries);
}

/// Exported for compaction.test.ts (`parseSessionEntries`).
pub fn parse_session_entries(content: &str) -> Vec<FileEntry> {
	let mut entries = Vec::new();
	for line in content.trim().split('\n') {
		if line.trim().is_empty() {
			continue;
		}
		if let Ok(entry) = serde_json::from_str::<FileEntry>(line) {
			entries.push(entry);
		}
		// Skip malformed lines
	}
	entries
}

pub fn get_latest_compaction_entry(entries: &[SessionEntry]) -> Option<&CompactionEntry> {
	entries.iter().rev().find_map(|entry| match entry {
		SessionEntry::Compaction(compaction) => Some(compaction),
		_ => None,
	})
}

// ---------------------------------------------------------------------------
// Tree path / context building
// ---------------------------------------------------------------------------

/// Leaf selector for the free-function context builders (upstream JS
/// distinguishes `undefined` from `null`; crate docs D5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leaf<'a> {
	/// JS `undefined` — resolve to the last entry in the file.
	Undefined,
	/// JS `null` — the path is empty.
	Null,
	/// Walk from this entry id; unknown ids fall back to the last entry.
	Id(&'a str),
}

impl<'a> From<Option<&'a str>> for Leaf<'a> {
	fn from(value: Option<&'a str>) -> Self {
		match value {
			None => Leaf::Undefined,
			Some(id) => Leaf::Id(id),
		}
	}
}

type EntryIndex<'a> = HashMap<&'a str, &'a SessionEntry>;

fn build_entry_index<'a>(
	entries: &'a [SessionEntry],
	by_id: Option<&'a IndexMap<String, SessionEntry>>,
) -> EntryIndex<'a> {
	if let Some(by_id) = by_id {
		return by_id.iter().map(|(id, entry)| (id.as_str(), entry)).collect();
	}
	entries.iter().map(|entry| (entry.id(), entry)).collect()
}

fn build_session_path<'a>(
	entries: &'a [SessionEntry],
	leaf_id: Leaf<'_>,
	by_id: Option<&'a IndexMap<String, SessionEntry>>,
) -> Vec<&'a SessionEntry> {
	let index = build_entry_index(entries, by_id);
	let mut leaf: Option<&SessionEntry> = None;
	if let Leaf::Id(id) = leaf_id {
		leaf = index.get(id).copied();
	}
	if leaf.is_none() {
		if let Leaf::Null = leaf_id {
			return Vec::new();
		}
		leaf = entries.last();
	}
	let Some(mut current) = leaf else { return Vec::new() };

	let mut path = Vec::new();
	loop {
		path.push(current);
		match current.parent_id().and_then(|parent_id| index.get(parent_id).copied()) {
			Some(parent) => current = parent,
			None => break,
		}
	}
	path.reverse();
	path
}

fn get_session_context_settings(path: &[&SessionEntry]) -> (String, Option<SessionModelRef>) {
	let mut thinking_level = "off".to_string();
	let mut model: Option<SessionModelRef> = None;

	for entry in path {
		match entry {
			SessionEntry::ThinkingLevelChange(change) => thinking_level = change.thinking_level.clone(),
			SessionEntry::ModelChange(change) => {
				model = Some(SessionModelRef {
					provider: change.provider.clone(),
					model_id: change.model_id.clone(),
				});
			}
			SessionEntry::Message(message_entry) => {
				if let Some((provider, model_id)) = assistant_provider_model(&message_entry.message) {
					model = Some(SessionModelRef { provider, model_id });
				}
			}
			_ => {}
		}
	}

	(thinking_level, model)
}

/// `entry.message.role === "assistant"` provider/model extraction, including
/// opaque (partially typed) messages.
fn assistant_provider_model(message: &AgentMessage) -> Option<(String, String)> {
	match message {
		AgentMessage::Llm(Message::Assistant(assistant)) => {
			Some((assistant.provider.clone(), assistant.model.clone()))
		}
		AgentMessage::Custom(value) => {
			if value.get("role").and_then(|role| role.as_str()) == Some("assistant") {
				let provider = value.get("provider").and_then(|value| value.as_str())?;
				let model = value.get("model").and_then(|value| value.as_str())?;
				Some((provider.to_string(), model.to_string()))
			} else {
				None
			}
		}
		_ => None,
	}
}

/// Project one selected session entry into LLM/runtime messages
/// (`sessionEntryToContextMessages`). Plain custom entries are display/state
/// entries and do not participate in context.
pub fn session_entry_to_context_messages(entry: &SessionEntry) -> Vec<AgentMessage> {
	match entry {
		SessionEntry::Message(message_entry) => {
			let mut message = message_entry.message.clone();
			// Session files are parsed without validation; old versions,
			// forks, or hand-edited files can contain messages with null or
			// missing content. Typed messages are already normalized at the
			// pi-ai deserialization boundary (pi-ai D13); opaque messages are
			// repaired here like upstream does.
			if let AgentMessage::Custom(value) = &mut message {
				if value.get("role").and_then(|role| role.as_str()).is_some() {
					let content_missing_or_null =
						value.get("content").map(|content| content.is_null()).unwrap_or(true);
					if content_missing_or_null {
						let role = value.get("role").and_then(|role| role.as_str()).unwrap_or("");
						let replacement = match role {
							"system" => Some(Value::String(String::new())),
							"user" | "assistant" | "toolResult" => Some(Value::Array(Vec::new())),
							_ => None,
						};
						if let Some(replacement) = replacement {
							value["content"] = replacement;
						}
					}
				}
			}
			vec![message]
		}
		SessionEntry::CustomMessage(custom_message_entry) => {
			vec![custom_message_to_agent(&create_custom_message(
				&custom_message_entry.custom_type,
				custom_message_entry.content.clone(),
				custom_message_entry.display,
				custom_message_entry.details.clone(),
				&custom_message_entry.base.timestamp,
			))]
		}
		SessionEntry::BranchSummary(branch_summary_entry) => {
			if !branch_summary_entry.summary.is_empty() {
				vec![custom_message_to_agent(&create_branch_summary_message(
					&branch_summary_entry.summary,
					&branch_summary_entry.from_id,
					&branch_summary_entry.base.timestamp,
				))]
			} else {
				Vec::new()
			}
		}
		SessionEntry::Compaction(compaction_entry) => {
			let summary = custom_message_to_agent(&create_compaction_summary_message(
				&compaction_entry.summary,
				compaction_entry.tokens_before,
				&compaction_entry.base.timestamp,
			));
			match &compaction_entry.system_message {
				Some(system_message) => vec![AgentMessage::system(system_message.clone()), summary],
				None => vec![summary],
			}
		}
		_ => Vec::new(),
	}
}

/// Build the active, compaction-aware session entry list
/// (`buildContextEntries`).
///
/// This follows the current leaf path. If the path contains compaction
/// entries, the latest compaction is represented by the compaction entry
/// itself, followed by the kept entries starting at firstKeptEntryId and all
/// entries after the compaction entry. Older summarized entries are omitted.
pub fn build_context_entries(
	entries: &[SessionEntry],
	leaf_id: Leaf<'_>,
	by_id: Option<&IndexMap<String, SessionEntry>>,
) -> Vec<SessionEntry> {
	let path = build_session_path(entries, leaf_id, by_id);
	let mut compaction: Option<&CompactionEntry> = None;

	for entry in &path {
		if let SessionEntry::Compaction(compaction_entry) = entry {
			compaction = Some(compaction_entry);
		}
	}

	let Some(compaction) = compaction else { return path.into_iter().cloned().collect() };

	let compaction_idx = path.iter().position(|entry| entry.id() == compaction.base.id);
	let Some(compaction_idx) = compaction_idx else { return path.into_iter().cloned().collect() };

	let mut context_entries: Vec<SessionEntry> = vec![SessionEntry::Compaction(compaction.clone())];
	let mut found_first_kept = false;
	for entry in &path[..compaction_idx] {
		if entry.id() == compaction.first_kept_entry_id {
			found_first_kept = true;
		}
		let is_system_message = matches!(
			entry,
			SessionEntry::Message(message_entry) if message_entry.message.role() == "system"
		);
		if found_first_kept && !is_system_message {
			context_entries.push((*entry).clone());
		}
	}
	context_entries.extend(path[compaction_idx + 1..].iter().map(|entry| (*entry).clone()));
	context_entries
}

/// Serialize replacement content; assistant/toolResult string replacements
/// wrap into a single text block (upstream inline logic).
fn content_value_for_role(content: &ContextEditableContent, role: Option<&str>) -> Value {
	if let (Some(role), ContextEditableContent::Text(text)) = (role, content) {
		if role == "assistant" || role == "toolResult" {
			return serde_json::json!([{ "type": "text", "text": text }]);
		}
	}
	match content {
		ContextEditableContent::Text(text) => Value::String(text.clone()),
		ContextEditableContent::Blocks(blocks) => serde_json::to_value(blocks).unwrap_or(Value::Null),
		ContextEditableContent::AssistantBlocks(blocks) => serde_json::to_value(blocks).unwrap_or(Value::Null),
	}
}

/// Build the session context from entries using tree traversal
/// (`projectContextEntry`). If leafId is provided, walks from that entry to
/// root. Handles compaction and branch summaries along the path.
fn project_context_entry(entry: &SessionEntry, edit: Option<&ContextEditEntry>) -> Vec<AgentMessage> {
	let messages = session_entry_to_context_messages(entry);
	let Some(edit) = edit else { return messages };
	match &edit.replacement {
		ContextEditReplacement::Null => Vec::new(),
		ContextEditReplacement::Content { content } => messages
			.into_iter()
			.map(|message| {
				let role = message.role().to_string();
				if !matches!(role.as_str(), "user" | "assistant" | "toolResult" | "custom") {
					return message;
				}
				let replaced = content_value_for_role(content, Some(&role));
				let mut value = match serde_json::to_value(&message) {
					Ok(value) => value,
					Err(_) => return message,
				};
				value["content"] = replaced;
				serde_json::from_value::<AgentMessage>(value).unwrap_or(message)
			})
			.collect(),
	}
}

/// Build provenance-preserving, compaction-aware model context
/// (`buildSessionProjection`).
pub fn build_session_projection(
	entries: &[SessionEntry],
	leaf_id: Leaf<'_>,
	by_id: Option<&IndexMap<String, SessionEntry>>,
) -> SessionProjection {
	let path = build_session_path(entries, leaf_id, by_id);
	let (thinking_level, model) = get_session_context_settings(&path);
	let context_entries = build_context_entries(entries, leaf_id, by_id);
	let mut edits: HashMap<String, ContextEditEntry> = HashMap::new();
	for entry in &context_entries {
		if let SessionEntry::ContextEdit(context_edit) = entry {
			edits.insert(context_edit.target_id.clone(), context_edit.clone());
		}
	}
	let projected_entries: Vec<ProjectedSessionEntry> = context_entries
		.into_iter()
		.enumerate()
		.map(|(index, source_entry)| {
			let edit = edits.get(source_entry.id());
			// buildContextEntries() may retain an older compaction entry
			// because its raw ID lies inside the newest retained range. Only
			// the newest compaction at index zero contributes a checkpoint and
			// summary.
			let messages = if index > 0 && matches!(source_entry, SessionEntry::Compaction(_)) {
				Vec::new()
			} else {
				project_context_entry(&source_entry, edit)
			};
			ProjectedSessionEntry { source_entry, messages }
		})
		.collect();
	let messages = projected_entries.iter().flat_map(|entry| entry.messages.iter().cloned()).collect();
	SessionProjection { entries: projected_entries, messages, thinking_level, model }
}

/// Build the finalized model context from the canonical session projection
/// (`buildSessionContext`).
pub fn build_session_context(
	entries: &[SessionEntry],
	leaf_id: Leaf<'_>,
	by_id: Option<&IndexMap<String, SessionEntry>>,
) -> SessionContext {
	let SessionProjection { messages, thinking_level, model, .. } =
		build_session_projection(entries, leaf_id, by_id);
	SessionContext { messages, thinking_level, model }
}

// ---------------------------------------------------------------------------
// Session directories
// ---------------------------------------------------------------------------

/// Compute the default session directory for a cwd. Encodes cwd into a safe
/// directory name under ~/.pi/agent/sessions/ (`getDefaultSessionDirPath`).
fn get_default_session_dir_path(cwd: &str, agent_dir: &str) -> String {
	let resolved_cwd = resolve_path_cwd(cwd);
	let resolved_agent_dir = resolve_path_cwd(agent_dir);
	let trimmed = resolved_cwd
		.strip_prefix(['/', '\\'])
		.unwrap_or(&resolved_cwd)
		.replace(['/', '\\', ':'], "-");
	let safe_path = format!("--{trimmed}--");
	join(&resolved_agent_dir, &join("sessions", &safe_path))
}

/// Resolve and create the default session directory (`getDefaultSessionDir`).
pub fn get_default_session_dir(cwd: &str) -> Result<String, std::io::Error> {
	get_default_session_dir_with_agent_dir(cwd, &get_agent_dir())
}

pub(crate) fn get_default_session_dir_with_agent_dir(cwd: &str, agent_dir: &str) -> Result<String, std::io::Error> {
	let session_dir = get_default_session_dir_path(cwd, agent_dir);
	if !std::path::Path::new(&session_dir).exists() {
		std::fs::create_dir_all(&session_dir)?;
	}
	Ok(session_dir)
}

// ---------------------------------------------------------------------------
// File loading / header discovery
// ---------------------------------------------------------------------------

fn parse_session_entry_line(line: &str) -> Option<FileEntry> {
	if line.trim().is_empty() {
		return None;
	}
	serde_json::from_str::<FileEntry>(line).ok()
	// Skip malformed lines
}

/// Load all parsed entries from a session JSONL file
/// (`loadEntriesFromFile`). Repairs a missing trailing newline after a valid
/// session file; files without a valid leading session header are returned
/// unmodified as `[]`.
pub fn load_entries_from_file(file_path: &str) -> Result<Vec<FileEntry>, std::io::Error> {
	let resolved_file_path = normalize_path(file_path, &PathInputOptions::default());
	let path = std::path::Path::new(&resolved_file_path);
	if !path.exists() {
		return Ok(Vec::new());
	}

	let mut file = std::fs::File::open(path)?;
	let mut entries: Vec<FileEntry> = Vec::new();
	let mut pending: Vec<u8> = Vec::new();
	let mut buffer = vec![0u8; SESSION_READ_BUFFER_SIZE];
	loop {
		let bytes_read = file.read(&mut buffer)?;
		if bytes_read == 0 {
			break;
		}
		pending.extend_from_slice(&buffer[..bytes_read]);
		while let Some(newline_index) = find_newline(&pending) {
			let line = String::from_utf8_lossy(&pending[..newline_index]).into_owned();
			pending.drain(..=newline_index);
			if let Some(entry) = parse_session_entry_line(&line) {
				entries.push(entry);
			}
		}
	}

	// Flush the decoder: the final (unterminated) line.
	let final_line = String::from_utf8_lossy(&pending).into_owned();
	pending.clear();
	if let Some(final_entry) = parse_session_entry_line(&final_line) {
		entries.push(final_entry);
	}

	// Validate session header before repairing the file.
	if entries.is_empty() {
		return Ok(entries);
	}
	if entries[0].as_session_header().is_none() {
		return Ok(Vec::new());
	}

	if !final_line.is_empty() {
		let mut append = std::fs::OpenOptions::new().append(true).open(path)?;
		append.write_all(b"\n")?;
	}
	Ok(entries)
}

fn find_newline(bytes: &[u8]) -> Option<usize> {
	bytes.iter().position(|byte| *byte == b'\n')
}

/// Inspect a physical line while searching for the first parsed session
/// entry (`parseSessionHeaderCandidate`). Blank and malformed lines are
/// skipped to match loadEntriesFromFile(). Returns `Err(())` to keep
/// scanning, `Ok(None)` for a parsed non-header entry, or `Ok(header)`.
fn parse_session_header_candidate(line: &str) -> Result<Option<SessionHeader>, ()> {
	if line.trim().is_empty() {
		return Err(());
	}
	let Some(entry) = parse_session_entry_line(line) else { return Err(()) };
	match entry.as_session_header() {
		Some(header) => Ok(Some(header.clone())),
		None => Ok(None),
	}
}

/// Read just the first entry if it is a session header
/// (`readSessionHeader`). Bounded by [`MAX_SESSION_HEADER_SCAN_BYTES`].
fn read_session_header(file_path: &str) -> Result<Option<SessionHeader>, SessionManagerError> {
	let mut file = std::fs::File::open(file_path)?;
	let mut line_chunks: Vec<u8> = Vec::new();
	let mut scanned_bytes = 0usize;
	let mut buffer = vec![0u8; SESSION_HEADER_READ_BUFFER_SIZE];

	while scanned_bytes < MAX_SESSION_HEADER_SCAN_BYTES {
		let read_length = buffer.len().min(MAX_SESSION_HEADER_SCAN_BYTES - scanned_bytes);
		let bytes_read = file.read(&mut buffer[..read_length])?;
		if bytes_read == 0 {
			let line = String::from_utf8_lossy(&line_chunks).into_owned();
			// At EOF, blank/malformed final chunks read as "no header".
			return Ok(parse_session_header_candidate(&line).ok().flatten());
		}
		scanned_bytes += bytes_read;

		line_chunks.extend_from_slice(&buffer[..bytes_read]);
		while let Some(newline_index) = find_newline(&line_chunks) {
			let line: String = String::from_utf8_lossy(&line_chunks[..newline_index]).into_owned();
			line_chunks.drain(..=newline_index);
			match parse_session_header_candidate(&line) {
				Ok(Some(header)) => return Ok(Some(header)),
				Ok(None) => return Ok(None),
				Err(()) => {}
			}
		}
	}

	// Probe for EOF so a final header without a newline is allowed when it
	// ends exactly at the scan limit. Any additional byte exceeds the bounded
	// scan.
	let mut probe = [0u8; 1];
	if file.read(&mut probe)? == 0 {
		let line = String::from_utf8_lossy(&line_chunks).into_owned();
		return Ok(parse_session_header_candidate(&line).ok().flatten());
	}
	Err(SessionManagerError::HeaderScanLimit(MAX_SESSION_HEADER_SCAN_BYTES, file_path.to_string()))
}

/// Discovery wrapper: unreadable or oversized files are not sessions
/// (`readSessionHeaderForDiscovery`).
fn read_session_header_for_discovery(file_path: &str) -> Option<SessionHeader> {
	read_session_header(file_path).ok().flatten()
}

fn get_session_header_cwd(header: &SessionHeader) -> &str {
	&header.cwd
}

fn session_cwd_matches(cwd: Option<&str>, resolved_cwd: &str) -> bool {
	match cwd {
		Some(cwd) if !cwd.is_empty() => resolve_path_cwd(cwd) == resolved_cwd,
		_ => false,
	}
}

/// Find the most recently modified valid session file in a directory,
/// optionally filtered by the session's cwd (`findMostRecentSession`).
pub fn find_most_recent_session(session_dir: &str, cwd: Option<&str>) -> Option<String> {
	let resolved_session_dir = normalize_path(session_dir, &PathInputOptions::default());
	let resolved_cwd = cwd.map(resolve_path_cwd);
	let Ok(dir_entries) = std::fs::read_dir(&resolved_session_dir) else {
		// Directory access races make recent-session discovery unavailable.
		return None;
	};
	let mut files: Vec<(String, i64)> = Vec::new();
	for entry in dir_entries.flatten() {
		let file_name = entry.file_name();
		let file_name = file_name.to_string_lossy();
		if !file_name.ends_with(".jsonl") {
			continue;
		}
		let path = join(&resolved_session_dir, &file_name);
		let Ok(metadata) = std::fs::metadata(&path) else { continue };
		let mtime = modified_time_ms(&metadata);
		files.push((path, mtime));
	}
	files.sort_by(|a, b| b.1.cmp(&a.1));

	for (path, _) in files {
		let Some(header) = read_session_header_for_discovery(&path) else { continue };
		let header_cwd = get_session_header_cwd(&header);
		let matches_cwd = match &resolved_cwd {
			None => true,
			Some(resolved_cwd) => session_cwd_matches(Some(header_cwd), resolved_cwd),
		};
		if matches_cwd {
			return Some(path);
		}
	}
	None
}

fn modified_time_ms(metadata: &std::fs::Metadata) -> i64 {
	metadata
		.modified()
		.ok()
		.and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
		.map(|duration| duration.as_millis() as i64)
		.unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Session info + listing
// ---------------------------------------------------------------------------

/// `isMessageWithContent` — role is a string and the message carries content.
fn is_message_with_content(message: &AgentMessage) -> bool {
	match message {
		AgentMessage::Llm(_) => true,
		AgentMessage::Custom(value) => {
			value.get("role").and_then(|role| role.as_str()).is_some() && value.get("content").is_some()
		}
	}
}

/// `extractTextContent` — string content as-is; text blocks joined with " ".
fn extract_text_content(message: &AgentMessage) -> String {
	match message {
		AgentMessage::Llm(Message::User(user)) => match &user.content {
			pi_ai::types::UserContentField::Text(text) => text.clone(),
			pi_ai::types::UserContentField::Blocks(blocks) => blocks
				.iter()
				.filter_map(|block| match block {
					UserContent::Text(text) => Some(text.text.clone()),
					_ => None,
				})
				.collect::<Vec<_>>()
				.join(" "),
		},
		AgentMessage::Llm(Message::Assistant(assistant)) => assistant
			.content
			.iter()
			.filter_map(|block| match block {
				AssistantContent::Text(text) => Some(text.text.clone()),
				_ => None,
			})
			.collect::<Vec<_>>()
			.join(" "),
		AgentMessage::Llm(_) => String::new(),
		AgentMessage::Custom(value) => {
			let content = value.get("content").cloned().unwrap_or(Value::Null);
			match content {
				Value::String(text) => text,
				Value::Array(blocks) => blocks
					.iter()
					.filter(|block| block.get("type").and_then(|value| value.as_str()) == Some("text"))
					.filter_map(|block| block.get("text").and_then(|value| value.as_str()))
					.map(String::from)
					.collect::<Vec<_>>()
					.join(" "),
				_ => String::new(),
			}
		}
	}
}

/// `getMessageActivityTime` — message timestamp if numeric, else the entry
/// timestamp.
fn get_message_activity_time(entry: &SessionMessageEntry) -> Option<i64> {
	if !is_message_with_content(&entry.message) {
		return None;
	}
	let role = entry.message.role();
	if role != "user" && role != "assistant" {
		return None;
	}

	if let Some(timestamp) = match &entry.message {
		AgentMessage::Llm(Message::User(user)) => Some(user.timestamp),
		AgentMessage::Llm(Message::Assistant(assistant)) => Some(assistant.timestamp),
		AgentMessage::Custom(value) => value.get("timestamp").and_then(|value| value.as_i64()),
		_ => None,
	} {
		return Some(timestamp);
	}

	parse_timestamp_ms(&entry.base.timestamp)
}

/// Build [`SessionInfo`] by reading one session file (`buildSessionInfo`).
/// Synchronous; callers offload to blocking threads.
fn build_session_info(file_path: &str, file_stats: Option<std::fs::Metadata>) -> Option<SessionInfo> {
	let stats = file_stats.or_else(|| std::fs::metadata(file_path).ok())?;
	let mut header: Option<SessionHeader> = None;
	let mut message_count = 0u64;
	let mut first_message = String::new();
	let mut all_messages: Vec<String> = Vec::new();
	let mut name: Option<String> = None;
	let mut last_activity_time: Option<i64> = None;

	let content = std::fs::read(file_path).ok()?;
	for line in iter_lines(&content) {
		let Some(entry) = parse_session_entry_line(&line) else { continue };

		if header.is_none() {
			header = Some(entry.as_session_header()?.clone());
			continue;
		}

		let Some(session_entry) = entry.as_entry() else { continue };
		match session_entry {
			// Extract session name (use latest, including explicit clears)
			SessionEntry::SessionInfo(info) => {
				name = info.name.as_deref().map(str::trim).filter(|name| !name.is_empty()).map(String::from);
			}
			SessionEntry::Message(message_entry) => {
				message_count += 1;

				if let Some(activity_time) = get_message_activity_time(message_entry) {
					last_activity_time = Some(last_activity_time.unwrap_or(0).max(activity_time));
				}

				if !is_message_with_content(&message_entry.message) {
					continue;
				}
				let role = message_entry.message.role();
				if role != "user" && role != "assistant" {
					continue;
				}

				let text_content = extract_text_content(&message_entry.message);
				if text_content.is_empty() {
					continue;
				}

				all_messages.push(text_content.clone());
				if first_message.is_empty() && role == "user" {
					first_message = text_content;
				}
			}
			_ => {}
		}
	}

	let header = header?;
	let modified_time = modified_time_ms(&stats);
	let header_time = parse_timestamp_ms(&header.timestamp);
	let modified = match last_activity_time {
		Some(activity) if activity > 0 => activity,
		_ => header_time.unwrap_or(modified_time),
	};

	Some(SessionInfo {
		path: file_path.to_string(),
		id: header.id.clone(),
		cwd: header.cwd.clone(),
		name,
		parent_session_path: header.parent_session.clone(),
		created: header_time.unwrap_or(0),
		modified,
		message_count,
		first_message: if first_message.is_empty() { "(no messages)".to_string() } else { first_message },
		all_messages_text: all_messages.join(" "),
	})
}

/// Split on `\n` (readline equivalent; `\r` stays in the line, matching
/// loadEntriesFromFile semantics).
fn iter_lines(content: &[u8]) -> Vec<String> {
	let mut lines = Vec::new();
	let mut start = 0;
	while start <= content.len() {
		let end = content[start..]
			.iter()
			.position(|byte| *byte == b'\n')
			.map(|index| start + index)
			.unwrap_or(content.len());
		// Invalid UTF-8 is lossy-converted like the upstream StringDecoder.
		lines.push(String::from_utf8_lossy(&content[start..end]).into_owned());
		if end == content.len() {
			break;
		}
		start = end + 1;
	}
	lines
}

/// Progress callback for session listing (`SessionListProgress`).
pub type SessionListProgress = Arc<dyn Fn(usize, usize, Option<&[SessionInfo]>) + Send + Sync>;

fn sort_session_infos(mut sessions: Vec<SessionInfo>) -> Vec<SessionInfo> {
	sessions.sort_by(|a, b| b.modified.cmp(&a.modified));
	sessions
}

type OnSessionLoaded = Arc<dyn Fn(Option<&SessionInfo>, usize) + Send + Sync>;

/// Load session infos for files with bounded concurrency
/// (`buildSessionInfosWithConcurrency` + `mapWithConcurrency`).
async fn build_session_infos_with_concurrency(
	files: Vec<String>,
	on_loaded: OnSessionLoaded,
	signal: Option<pi_ai::utils::abort::AbortSignal>,
) -> Result<Vec<Option<SessionInfo>>, SessionListError> {
	let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_SESSION_INFO_LOADS));
	let mut handles = Vec::with_capacity(files.len());
	for (index, path) in files.into_iter().enumerate() {
		let semaphore = semaphore.clone();
		let signal = signal.clone();
		let on_loaded = on_loaded.clone();
		let task: tokio::task::JoinHandle<Result<Option<SessionInfo>, pi_ai::utils::abort::AbortReason>> =
			tokio::spawn(async move {
				// Bounded concurrency: acquire before doing work.
				let _permit = semaphore.acquire_owned().await;
				if let Some(signal) = &signal {
					signal.check()?;
				}
				let info = tokio::task::spawn_blocking(move || build_session_info(&path, None))
					.await
					.map_err(|_| {
						pi_ai::utils::abort::AbortReason::Message("session info task panicked".to_string())
					})?;
				// Upstream reads each file through a signal-bound stream: an
				// abort during a load fails that load with AbortError even if
				// the read had finished.
				if let Some(signal) = &signal {
					signal.check()?;
				}
				on_loaded(info.as_ref(), index);
				Ok(info)
			});
		handles.push(task);
	}

	let mut results = Vec::with_capacity(handles.len());
	for handle in handles {
		results.push(handle.await.map_err(|_| {
			SessionListError::Aborted(pi_ai::utils::abort::AbortReason::Message(
				"session info task panicked".to_string(),
			))
		})??);
	}
	Ok(results)
}

/// List sessions in one directory (`listSessionsFromDir`).
async fn list_sessions_from_dir(
	dir: &str,
	on_progress: Option<&SessionListProgress>,
	signal: Option<pi_ai::utils::abort::AbortSignal>,
) -> Result<Vec<SessionInfo>, SessionListError> {
	if let Some(signal) = &signal {
		signal.check()?;
	}
	if !std::path::Path::new(dir).exists() {
		return Ok(Vec::new());
	}

	let dir_entries = std::fs::read_dir(dir)?;
	let mut files: Vec<String> = dir_entries
		.flatten()
		.map(|entry| entry.file_name().to_string_lossy().into_owned())
		.filter(|file| file.ends_with(".jsonl"))
		.collect();
	// Upstream sorts names descending (b.localeCompare(a)), then joins.
	files.sort_by(|a, b| b.cmp(a));
	let files: Vec<String> = files.iter().map(|file| join(dir, file)).collect();
	let total = files.len();

	let loaded = Arc::new(AtomicUsize::new(0));
	let partial_sessions: Arc<Mutex<Vec<SessionInfo>>> = Arc::new(Mutex::new(Vec::new()));
	let progress = on_progress.cloned();
	let results = build_session_infos_with_concurrency(
		files,
		{
			let loaded = loaded.clone();
			let partial_sessions = partial_sessions.clone();
			Arc::new(move |info, _| {
				let loaded = loaded.fetch_add(1, Ordering::SeqCst) + 1;
				if let Some(info) = info {
					partial_sessions.lock().expect("partial sessions lock").push(info.clone());
				}
				let Some(progress) = &progress else { return };
				let publish_partial =
					loaded == 1 || loaded % CURRENT_SESSION_LIST_PUBLISH_INTERVAL == 0 || loaded == total;
				let snapshot: Option<Vec<SessionInfo>> = if publish_partial {
					Some(sort_session_infos(partial_sessions.lock().expect("partial sessions lock").clone()))
				} else {
					None
				};
				progress(loaded, total, snapshot.as_deref());
			})
		},
		signal,
	)
	.await?;
	Ok(results.into_iter().flatten().collect())
}

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------

/// Manages conversation sessions as append-only trees stored in JSONL files
/// (`SessionManager`).
pub struct SessionManager {
	session_id: String,
	session_file: Option<String>,
	session_dir: String,
	cwd: String,
	persist: bool,
	flushed: bool,
	file_entries: Vec<FileEntry>,
	by_id: IndexMap<String, SessionEntry>,
	labels_by_id: IndexMap<String, String>,
	label_timestamps_by_id: IndexMap<String, String>,
	leaf_id: Option<String>,
}

fn entry_base(id: String, parent_id: Option<String>) -> SessionEntryBase {
	SessionEntryBase { id, parent_id, timestamp: to_iso_timestamp(now_ms()) }
}

fn entry_base_with_timestamp(
	id: String,
	parent_id: Option<String>,
	timestamp: String,
) -> SessionEntryBase {
	SessionEntryBase { id, parent_id, timestamp }
}

impl SessionManager {
	fn new(
		cwd: &str,
		session_dir: &str,
		session_file: Option<String>,
		persist: bool,
		new_session_options: Option<&NewSessionOptions>,
		preloaded_file_entries: Option<Vec<FileEntry>>,
	) -> Result<Self, SessionManagerError> {
		let cwd = resolve_path_cwd(cwd);
		let session_dir = normalize_path(session_dir, &PathInputOptions::default());
		if persist && !session_dir.is_empty() && !std::path::Path::new(&session_dir).exists() {
			std::fs::create_dir_all(&session_dir)?;
		}

		let mut manager = SessionManager {
			session_id: String::new(),
			session_file: None,
			session_dir,
			cwd,
			persist,
			flushed: false,
			file_entries: Vec::new(),
			by_id: IndexMap::new(),
			labels_by_id: IndexMap::new(),
			label_timestamps_by_id: IndexMap::new(),
			leaf_id: None,
		};

		if let Some(session_file) = session_file {
			manager.set_session_file_internal(session_file, preloaded_file_entries)?;
		} else if preloaded_file_entries.as_ref().is_some_and(|entries| !entries.is_empty()) {
			manager.load_entries(preloaded_file_entries.expect("checked non-empty"), new_session_options)?;
		} else {
			manager.new_session(new_session_options)?;
		}
		Ok(manager)
	}

	/// Switch to a different session file (used for resume and branching).
	pub fn set_session_file(&mut self, session_file: &str) -> Result<(), SessionManagerError> {
		self.set_session_file_internal(session_file.to_string(), None)
	}

	fn set_session_file_internal(
		&mut self,
		session_file: String,
		preloaded_file_entries: Option<Vec<FileEntry>>,
	) -> Result<(), SessionManagerError> {
		let resolved = resolve_path_cwd(&session_file);
		self.session_file = Some(resolved.clone());
		if std::path::Path::new(&resolved).exists() {
			let entries = match preloaded_file_entries {
				Some(entries) => entries,
				None => load_entries_from_file(&resolved)?,
			};

			// If file was empty, initialize it with a valid session header. If
			// it was non-empty but did not parse as a pi session, fail without
			// modifying it.
			if entries.is_empty() {
				if std::fs::metadata(&resolved)?.len() > 0 {
					return Err(SessionManagerError::NotASession {
						app_name: APP_NAME.to_string(),
						path: resolved,
					});
				}
				self.new_session(None)?;
				self.session_file = Some(resolved);
				self.rewrite_file()?;
				self.flushed = true;
				return Ok(());
			}

			self.load_entries(entries, None)?;
			self.flushed = true;
		} else {
			self.new_session(None)?;
			self.session_file = Some(resolved); // preserve explicit path from --session flag
		}
		Ok(())
	}

	/// Start a new session (`newSession`).
	pub fn new_session(&mut self, options: Option<&NewSessionOptions>) -> Result<(), SessionManagerError> {
		if let Some(id) = options.and_then(|options| options.id.as_deref()) {
			assert_valid_session_id(id)?;
		}
		self.session_id = options.and_then(|options| options.id.clone()).unwrap_or_else(create_session_id);
		let timestamp = to_iso_timestamp(now_ms());
		let header = SessionHeader {
			header_type: SessionHeaderType::Session,
			version: Some(CURRENT_SESSION_VERSION),
			id: self.session_id.clone(),
			timestamp: timestamp.clone(),
			cwd: self.cwd.clone(),
			parent_session: options.and_then(|options| options.parent_session.clone()),
		};
		self.file_entries = vec![FileEntry::Header(header)];
		self.by_id.clear();
		self.labels_by_id.clear();
		self.label_timestamps_by_id.clear();
		self.leaf_id = None;
		self.flushed = false;

		if self.persist {
			let file_timestamp = timestamp.replace([':', '.'], "-");
			self.session_file = Some(join(
				&self.get_session_dir(),
				&format!("{file_timestamp}_{}.jsonl", self.session_id),
			));
		}
		Ok(())
	}

	fn load_entries(
		&mut self,
		entries: Vec<FileEntry>,
		options: Option<&NewSessionOptions>,
	) -> Result<(), SessionManagerError> {
		let header = entries.iter().find_map(|entry| entry.as_session_header().cloned());

		if let Some(header) = header {
			self.file_entries = entries;
			self.session_id = header.id;

			if migrate_to_current_version(&mut self.file_entries) {
				self.rewrite_file()?;
			}
		} else {
			self.new_session(options)?;
			self.file_entries.extend(entries);
		}

		self.build_index();
		Ok(())
	}

	fn build_index(&mut self) {
		self.by_id.clear();
		self.labels_by_id.clear();
		self.label_timestamps_by_id.clear();
		self.leaf_id = None;
		for entry in &self.file_entries {
			let Some(session_entry) = entry.as_entry() else { continue };
			self.by_id.insert(session_entry.id().to_string(), session_entry.clone());
			self.leaf_id = Some(session_entry.id().to_string());
			if let SessionEntry::Label(label_entry) = session_entry {
				if label_entry.label.as_deref().is_some_and(|label| !label.is_empty()) {
					self.labels_by_id
						.insert(label_entry.target_id.clone(), label_entry.label.clone().unwrap_or_default());
					self.label_timestamps_by_id
						.insert(label_entry.target_id.clone(), label_entry.base.timestamp.clone());
				} else {
					self.labels_by_id.shift_remove(&label_entry.target_id);
					self.label_timestamps_by_id.shift_remove(&label_entry.target_id);
				}
			}
		}
	}

	fn rewrite_file(&self) -> Result<(), SessionManagerError> {
		if !self.persist {
			return Ok(());
		}
		let Some(session_file) = &self.session_file else { return Ok(()) };
		let mut file = std::fs::File::create(session_file)?;
		for entry in &self.file_entries {
			writeln!(file, "{}", serde_json::to_string(entry).expect("session entry serialization"))?;
		}
		Ok(())
	}

	pub fn is_persisted(&self) -> bool {
		self.persist
	}

	pub fn get_cwd(&self) -> &str {
		&self.cwd
	}

	pub fn get_session_dir(&self) -> String {
		self.session_dir.clone()
	}

	pub fn uses_default_session_dir(&self) -> bool {
		self.session_dir == get_default_session_dir_path(&self.cwd, &get_agent_dir())
	}

	pub fn get_session_id(&self) -> &str {
		&self.session_id
	}

	pub fn get_session_file(&self) -> Option<&str> {
		self.session_file.as_deref()
	}

	/// A new session file is created only once the session contains a user or
	/// assistant message. Setup entries alone (model, thinking level, system
	/// prompt) stay in memory so opening and closing pi without chatting
	/// leaves no file behind. Starting at the user message (not the first
	/// assistant reply) keeps the prompt on disk if the first turn never
	/// completes (#10000).
	fn has_conversation(&self) -> bool {
		self.file_entries.iter().any(|entry| {
			matches!(
				entry.as_entry(),
				Some(SessionEntry::Message(message_entry))
					if matches!(message_entry.message.role(), "user" | "assistant")
			)
		})
	}

	fn persist_entry(&mut self, entry: &SessionEntry) -> Result<(), SessionManagerError> {
		if !self.persist {
			return Ok(());
		}
		let Some(session_file) = &self.session_file else { return Ok(()) };

		if !self.flushed {
			if !self.has_conversation() {
				return Ok(());
			}
			let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(session_file)?;
			for entry in &self.file_entries {
				writeln!(file, "{}", serde_json::to_string(entry).expect("session entry serialization"))?;
			}
			self.flushed = true;
		} else {
			let mut file = std::fs::OpenOptions::new().append(true).open(session_file)?;
			writeln!(file, "{}", serde_json::to_string(entry).expect("session entry serialization"))?;
		}
		Ok(())
	}

	fn append_entry(&mut self, entry: SessionEntry) -> Result<(), SessionManagerError> {
		let id = entry.id().to_string();
		self.file_entries.push(FileEntry::Entry(entry.clone()));
		self.by_id.insert(id, entry.clone());
		self.leaf_id = Some(entry.id().to_string());
		self.persist_entry(&entry)?;
		Ok(())
	}

	/// Append a message as child of current leaf, then advance leaf. Returns
	/// entry id. Does not allow writing CompactionSummaryMessage and
	/// BranchSummaryMessage directly: these are top-level entries appended via
	/// [`SessionManager::append_compaction`] and
	/// [`SessionManager::branch_with_summary`] (`appendMessage`).
	pub fn append_message(&mut self, message: impl Into<AgentMessage>) -> Result<String, SessionManagerError> {
		let entry = SessionEntry::Message(SessionMessageEntry {
			base: entry_base(generate_id_for_map(&self.by_id), self.leaf_id.clone()),
			message: message.into(),
		});
		let id = entry.id().to_string();
		self.append_entry(entry)?;
		Ok(id)
	}

	/// Append a thinking level change as child of current leaf, then advance
	/// leaf. Returns entry id.
	pub fn append_thinking_level_change(&mut self, thinking_level: &str) -> Result<String, SessionManagerError> {
		let entry = SessionEntry::ThinkingLevelChange(ThinkingLevelChangeEntry {
			base: entry_base(generate_id_for_map(&self.by_id), self.leaf_id.clone()),
			thinking_level: thinking_level.to_string(),
		});
		let id = entry.id().to_string();
		self.append_entry(entry)?;
		Ok(id)
	}

	/// Append a model change as child of current leaf, then advance leaf.
	/// Returns entry id.
	pub fn append_model_change(&mut self, provider: &str, model_id: &str) -> Result<String, SessionManagerError> {
		let entry = SessionEntry::ModelChange(ModelChangeEntry {
			base: entry_base(generate_id_for_map(&self.by_id), self.leaf_id.clone()),
			provider: provider.to_string(),
			model_id: model_id.to_string(),
		});
		let id = entry.id().to_string();
		self.append_entry(entry)?;
		Ok(id)
	}

	/// Append model-attributed usage that does not participate in LLM
	/// context. Returns the appended entry (`appendUsage`).
	pub fn append_usage(
		&mut self,
		kind: &str,
		provider: &str,
		model: &str,
		usage: Usage,
		note: Option<String>,
	) -> Result<UsageEntry, SessionManagerError> {
		let entry = SessionEntry::Usage(UsageEntry {
			base: entry_base(generate_id_for_map(&self.by_id), self.leaf_id.clone()),
			kind: kind.to_string(),
			provider: provider.to_string(),
			model: model.to_string(),
			usage,
			note: note.filter(|note| !note.is_empty()),
		});
		let SessionEntry::Usage(usage_entry) = entry.clone() else { unreachable!() };
		self.append_entry(entry)?;
		Ok(usage_entry)
	}

	/// Append a compaction summary as child of current leaf, then advance
	/// leaf. Returns entry id (`appendCompaction`).
	pub fn append_compaction(
		&mut self,
		summary: &str,
		first_kept_entry_id: Option<&str>,
		tokens_before: u64,
		details: Option<Value>,
		from_hook: Option<bool>,
		usage: Option<Usage>,
	) -> Result<String, SessionManagerError> {
		let timestamp = to_iso_timestamp(now_ms());
		let projection = self.build_session_projection();
		let llm_messages: Vec<Message> =
			projection.messages.iter().filter_map(|message| message.as_llm().cloned()).collect();
		let system_message =
			pi_ai::utils::transcript::get_current_system_message(&llm_messages).map(|mut system_message| {
				system_message.timestamp = parse_timestamp_ms(&timestamp).unwrap_or(0);
				system_message
			});
		let id = generate_id_for_map(&self.by_id);
		let entry = SessionEntry::Compaction(CompactionEntry {
			base: entry_base(id.clone(), self.leaf_id.clone()),
			summary: summary.to_string(),
			first_kept_entry_id: first_kept_entry_id.map(String::from).unwrap_or_else(|| id.clone()),
			tokens_before,
			details,
			usage,
			from_hook,
			system_message,
			first_kept_entry_index: None,
		});
		self.append_entry(entry)?;
		Ok(id)
	}

	/// Append a custom entry (for extensions) as child of current leaf, then
	/// advance leaf. Returns entry id.
	pub fn append_custom_entry(
		&mut self,
		custom_type: &str,
		data: Option<Value>,
	) -> Result<String, SessionManagerError> {
		let entry = SessionEntry::Custom(CustomEntry {
			base: entry_base(generate_id_for_map(&self.by_id), self.leaf_id.clone()),
			custom_type: custom_type.to_string(),
			data,
		});
		let id = entry.id().to_string();
		self.append_entry(entry)?;
		Ok(id)
	}

	/// Append a session info entry (e.g., display name). Returns entry id.
	pub fn append_session_info(&mut self, name: &str) -> Result<String, SessionManagerError> {
		let sanitized_name = collapse_newlines(name);
		let entry = SessionEntry::SessionInfo(SessionInfoEntry {
			base: entry_base(generate_id_for_map(&self.by_id), self.leaf_id.clone()),
			name: Some(sanitized_name),
		});
		let id = entry.id().to_string();
		self.append_entry(entry)?;
		Ok(id)
	}

	/// Get the current session name from the latest session_info entry, if
	/// any (`getSessionName`).
	pub fn get_session_name(&self) -> Option<String> {
		// Walk entries in reverse to find the latest session_info entry.
		// Empty names explicitly clear the session title. Reads fileEntries
		// directly: the footer calls this on every frame.
		for entry in self.file_entries.iter().rev() {
			if let Some(SessionEntry::SessionInfo(info)) = entry.as_entry() {
				return info
					.name
					.as_deref()
					.map(str::trim)
					.filter(|name| !name.is_empty())
					.map(String::from);
			}
		}
		None
	}

	/// Append a custom message entry (for extensions) that participates in
	/// LLM context (`appendCustomMessageEntry`).
	///
	/// - `custom_type`: extension identifier for filtering on reload
	/// - `content`: message content (string or text/image block array)
	/// - `display`: whether to show in TUI (true = styled display, false =
	///   hidden)
	/// - `details`: optional extension-specific metadata (not sent to LLM)
	pub fn append_custom_message_entry(
		&mut self,
		custom_type: &str,
		content: pi_ai::types::UserContentField,
		display: bool,
		details: Option<Value>,
	) -> Result<String, SessionManagerError> {
		let entry = SessionEntry::CustomMessage(CustomMessageEntry {
			base: entry_base(generate_id_for_map(&self.by_id), self.leaf_id.clone()),
			custom_type: custom_type.to_string(),
			content,
			details,
			display,
		});
		let id = entry.id().to_string();
		self.append_entry(entry)?;
		Ok(id)
	}

	/// Append a branch-local edit to an earlier model-visible entry
	/// (`appendContextEdit`).
	pub fn append_context_edit(
		&mut self,
		target_id: &str,
		replacement: Option<ContextEditableContent>,
	) -> Result<String, SessionManagerError> {
		let target = self
			.by_id
			.get(target_id)
			.cloned()
			.ok_or_else(|| SessionManagerError::EntryNotFound(target_id.to_string()))?;
		if !self.get_branch().iter().any(|entry| entry.id() == target_id) {
			return Err(SessionManagerError::NotOnActiveBranch(target_id.to_string()));
		}
		let target_role = match &target {
			SessionEntry::CustomMessage(_) => "custom".to_string(),
			SessionEntry::Message(message_entry) => message_entry.message.role().to_string(),
			_ => return Err(SessionManagerError::NotEditableContent(target_id.to_string())),
		};
		let editable = match &target {
			SessionEntry::CustomMessage(_) => true,
			SessionEntry::Message(message_entry) => {
				matches!(message_entry.message.role(), "user" | "assistant" | "toolResult")
			}
			_ => false,
		};
		if !editable {
			return Err(SessionManagerError::NotEditableContent(target_id.to_string()));
		}
		let normalized_replacement = match replacement {
			Some(ContextEditableContent::Text(text))
				if target_role == "assistant" || target_role == "toolResult" =>
			{
				Some(ContextEditableContent::AssistantBlocks(vec![AssistantContent::Text(
					pi_ai::types::TextContent::new(text),
				)]))
			}
			other => other,
		};
		let entry = SessionEntry::ContextEdit(ContextEditEntry {
			base: entry_base(generate_id_for_map(&self.by_id), self.leaf_id.clone()),
			target_id: target_id.to_string(),
			replacement: normalized_replacement
				.map(|content| ContextEditReplacement::Content { content })
				.unwrap_or(ContextEditReplacement::Null),
		});
		let id = entry.id().to_string();
		self.append_entry(entry)?;
		Ok(id)
	}

	// =========================================================================
	// Tree Traversal
	// =========================================================================

	pub fn get_leaf_id(&self) -> Option<&str> {
		self.leaf_id.as_deref()
	}

	pub fn get_leaf_entry(&self) -> Option<&SessionEntry> {
		self.leaf_id.as_deref().and_then(|id| self.by_id.get(id))
	}

	pub fn get_entry(&self, id: &str) -> Option<&SessionEntry> {
		self.by_id.get(id)
	}

	/// Get all direct children of an entry (`getChildren`).
	pub fn get_children(&self, parent_id: &str) -> Vec<SessionEntry> {
		self.by_id
			.values()
			.filter(|entry| entry.parent_id() == Some(parent_id))
			.cloned()
			.collect()
	}

	/// Get the label for an entry, if any (`getLabel`).
	pub fn get_label(&self, id: &str) -> Option<String> {
		self.labels_by_id.get(id).cloned()
	}

	/// Set or clear a label on an entry. Labels are user-defined markers for
	/// bookmarking/navigation. Pass `None` or an empty string to clear the
	/// label (`appendLabelChange`).
	pub fn append_label_change(
		&mut self,
		target_id: &str,
		label: Option<String>,
	) -> Result<String, SessionManagerError> {
		if !self.by_id.contains_key(target_id) {
			return Err(SessionManagerError::EntryNotFound(target_id.to_string()));
		}
		let entry = SessionEntry::Label(LabelEntry {
			base: entry_base(generate_id_for_map(&self.by_id), self.leaf_id.clone()),
			target_id: target_id.to_string(),
			label: label.clone(),
		});
		let id = entry.id().to_string();
		let timestamp = entry.base().timestamp.clone();
		if label.as_deref().is_some_and(|label| !label.is_empty()) {
			self.labels_by_id.insert(target_id.to_string(), label.expect("checked non-empty"));
			self.label_timestamps_by_id.insert(target_id.to_string(), timestamp);
		} else {
			self.labels_by_id.shift_remove(target_id);
			self.label_timestamps_by_id.shift_remove(target_id);
		}
		self.append_entry(entry)?;
		Ok(id)
	}

	/// Walk from entry to root, returning all entries in path order
	/// (`getBranch`). Includes all entry types (messages, compaction, model
	/// changes, etc.). Use [`SessionManager::build_session_context`] to get
	/// the resolved messages for the LLM.
	pub fn get_branch(&self) -> Vec<SessionEntry> {
		self.get_branch_from_option(self.leaf_id.clone())
	}

	/// `getBranch(fromId)`.
	pub fn get_branch_from(&self, from_id: &str) -> Vec<SessionEntry> {
		self.get_branch_from_option(Some(from_id.to_string()))
	}

	fn get_branch_from_option(&self, start_id: Option<String>) -> Vec<SessionEntry> {
		let mut path: Vec<SessionEntry> = Vec::new();
		let mut current = start_id.and_then(|id| self.by_id.get(&id).cloned());
		while let Some(entry) = current {
			current = entry.parent_id().and_then(|parent_id| self.by_id.get(parent_id).cloned());
			path.push(entry);
		}
		path.reverse();
		path
	}

	/// Build the active, compaction-aware entry list for context/rendering.
	/// Uses tree traversal from current leaf (`buildContextEntries`).
	pub fn build_context_entries(&self) -> Vec<SessionEntry> {
		build_context_entries(&self.get_entries(), self.leaf_selector(), Some(&self.by_id))
	}

	/// Build the session context (what gets sent to the LLM). Uses tree
	/// traversal from current leaf (`buildSessionProjection`).
	pub fn build_session_projection(&self) -> SessionProjection {
		build_session_projection(&self.get_entries(), self.leaf_selector(), Some(&self.by_id))
	}

	pub fn build_session_context(&self) -> SessionContext {
		let SessionProjection { messages, thinking_level, model, .. } = self.build_session_projection();
		SessionContext { messages, thinking_level, model }
	}

	fn leaf_selector(&self) -> Leaf<'_> {
		match &self.leaf_id {
			Some(id) => Leaf::Id(id),
			None => Leaf::Null,
		}
	}

	/// Get session header (`getHeader`).
	pub fn get_header(&self) -> Option<SessionHeader> {
		self.file_entries.iter().find_map(|entry| entry.as_session_header().cloned())
	}

	/// Number of session entries (excludes header), without copying them like
	/// [`SessionManager::get_entries`] (`getEntryCount`).
	pub fn get_entry_count(&self) -> usize {
		self.by_id.len()
	}

	/// Get all session entries (excludes header) (`getEntries`). The session
	/// is append-only: use appendXXX() to add entries, branch() to change the
	/// leaf pointer. Entries cannot be modified or deleted.
	pub fn get_entries(&self) -> Vec<SessionEntry> {
		self.file_entries.iter().filter_map(|entry| entry.as_entry().cloned()).collect()
	}

	/// Get the session as a tree structure (`getTree`). Returns a defensive
	/// copy of all entries. A well-formed session has exactly one root (first
	/// entry with parentId === null). Orphaned entries (broken parent chain)
	/// are also returned as roots.
	pub fn get_tree(&self) -> Vec<SessionTreeNode> {
		let entries = self.get_entries();
		let mut node_map: HashMap<String, SessionTreeNode> = HashMap::new();
		let mut children_of: HashMap<String, Vec<String>> = HashMap::new();
		let mut root_ids: Vec<String> = Vec::new();

		// Create nodes with resolved labels
		for entry in &entries {
			let label = self.labels_by_id.get(entry.id()).cloned();
			let label_timestamp = self.label_timestamps_by_id.get(entry.id()).cloned();
			node_map.insert(
				entry.id().to_string(),
				SessionTreeNode {
					entry: entry.clone(),
					children: Vec::new(),
					label,
					label_timestamp,
				},
			);
		}

		// Build tree
		for entry in &entries {
			let id = entry.id();
			match entry.parent_id() {
				None => root_ids.push(id.to_string()),
				Some(parent_id) if parent_id == id => root_ids.push(id.to_string()),
				Some(parent_id) => match node_map.contains_key(parent_id) {
					true => children_of.entry(parent_id.to_string()).or_default().push(id.to_string()),
					// Orphan — treat as root
					false => root_ids.push(id.to_string()),
				},
			}
		}

		// Sort children by timestamp (oldest first, newest at bottom). Use an
		// iterative approach to avoid stack overflow on deep trees.
		for children in children_of.values_mut() {
			children.sort_by_key(|id| {
				node_map
					.get(id)
					.and_then(|node| parse_timestamp_ms(node.entry.timestamp()))
					.unwrap_or(0)
			});
		}

		// Iterative post-order assembly (cycle-safe).
		enum Frame {
			Enter(String),
			Complete(String),
		}
		let mut completed: HashMap<String, SessionTreeNode> = HashMap::new();
		let mut stack: Vec<Frame> = root_ids.iter().rev().map(|id| Frame::Enter(id.clone())).collect();
		while let Some(frame) = stack.pop() {
			match frame {
				Frame::Enter(id) => {
					// Complete goes on first so the children (pushed after)
					// pop and finish before the parent completes.
					stack.push(Frame::Complete(id.clone()));
					if let Some(children) = children_of.get(&id) {
						for child in children.iter().rev() {
							stack.push(Frame::Enter(child.clone()));
						}
					}
				}
				Frame::Complete(id) => {
					if completed.contains_key(&id) {
						continue; // cycle guard
					}
					let Some(mut node) = node_map.remove(&id) else { continue };
					let children = children_of.remove(&id).unwrap_or_default();
					node.children =
						children.iter().filter_map(|child| completed.remove(child)).collect();
					completed.insert(id, node);
				}
			}
		}
		root_ids.iter().filter_map(|id| completed.remove(id)).collect()
	}

	// =========================================================================
	// Branching
	// =========================================================================

	/// Start a new branch from an earlier entry (`branch`). Moves the leaf
	/// pointer to the specified entry. The next appendXXX() call will create a
	/// child of that entry, forming a new branch. Existing entries are not
	/// modified or deleted.
	pub fn branch(&mut self, branch_from_id: &str) -> Result<(), SessionManagerError> {
		if !self.by_id.contains_key(branch_from_id) {
			return Err(SessionManagerError::EntryNotFound(branch_from_id.to_string()));
		}
		self.leaf_id = Some(branch_from_id.to_string());
		Ok(())
	}

	/// Reset the leaf pointer to null (before any entries) (`resetLeaf`). The
	/// next appendXXX() call will create a new root entry (parentId = null).
	/// Use this when navigating to re-edit the first user message.
	pub fn reset_leaf(&mut self) {
		self.leaf_id = None;
	}

	/// Start a new branch with a summary of the abandoned path
	/// (`branchWithSummary`). Same as [`SessionManager::branch`], but also
	/// appends a branch_summary entry that captures context from the
	/// abandoned conversation path.
	pub fn branch_with_summary(
		&mut self,
		branch_from_id: Option<&str>,
		summary: &str,
		details: Option<Value>,
		from_hook: Option<bool>,
		usage: Option<Usage>,
	) -> Result<String, SessionManagerError> {
		if let Some(branch_from_id) = branch_from_id {
			if !self.by_id.contains_key(branch_from_id) {
				return Err(SessionManagerError::EntryNotFound(branch_from_id.to_string()));
			}
		}
		let from_id = self.leaf_id.clone().unwrap_or_else(|| "root".to_string());
		self.leaf_id = branch_from_id.map(String::from);
		let entry = SessionEntry::BranchSummary(BranchSummaryEntry {
			base: entry_base(generate_id_for_map(&self.by_id), self.leaf_id.clone()),
			from_id,
			summary: summary.to_string(),
			details,
			usage,
			from_hook,
		});
		let id = entry.id().to_string();
		self.append_entry(entry)?;
		Ok(id)
	}

	/// Create a new session file containing only the path from root to the
	/// specified leaf (`createBranchedSession`). Useful for extracting a
	/// single conversation path from a branched session. Returns the new
	/// session file path, or `None` if not persisting.
	pub fn create_branched_session(&mut self, leaf_id: &str) -> Result<Option<String>, SessionManagerError> {
		let previous_session_file = self.session_file.clone();
		let path = self.get_branch_from(leaf_id);
		if path.is_empty() {
			return Err(SessionManagerError::EntryNotFound(leaf_id.to_string()));
		}

		// Filter out LabelEntry from path — we'll recreate them from the
		// resolved map. Because labels are real tree entries, later entries
		// can be children of labels; removing labels requires re-chaining the
		// retained path to avoid orphaned subtrees.
		let mut path_without_labels: Vec<SessionEntry> = Vec::new();
		let mut replacement_by_label_id: HashMap<String, String> = HashMap::new();
		let mut pending_label_ids: Vec<String> = Vec::new();
		let mut path_parent_id: Option<String> = None;
		for entry in path {
			if let SessionEntry::Label(_) = &entry {
				pending_label_ids.push(entry.id().to_string());
				continue;
			}
			for label_id in pending_label_ids.drain(..) {
				replacement_by_label_id.insert(label_id, entry.id().to_string());
			}
			let mut entry = entry;
			if let SessionEntry::Compaction(compaction) = &mut entry {
				compaction.base.parent_id = path_parent_id.clone();
				compaction.first_kept_entry_id = if compaction.first_kept_entry_id == compaction.base.id {
					compaction.base.id.clone()
				} else {
					replacement_by_label_id
						.get(&compaction.first_kept_entry_id)
						.cloned()
						.unwrap_or_else(|| compaction.first_kept_entry_id.clone())
				};
			} else {
				entry.base_mut().parent_id = path_parent_id.clone();
			}
			path_parent_id = Some(entry.id().to_string());
			path_without_labels.push(entry);
		}

		let new_session_id = create_session_id();
		let timestamp = to_iso_timestamp(now_ms());
		let file_timestamp = timestamp.replace([':', '.'], "-");
		let new_session_file =
			join(&self.get_session_dir(), &format!("{file_timestamp}_{new_session_id}.jsonl"));

		let header = SessionHeader {
			header_type: SessionHeaderType::Session,
			version: Some(CURRENT_SESSION_VERSION),
			id: new_session_id.clone(),
			timestamp,
			cwd: self.cwd.clone(),
			parent_session: if self.persist { previous_session_file } else { None },
		};

		// Collect labels for entries in the path
		let path_entry_ids: IndexSet<String> =
			path_without_labels.iter().map(|entry| entry.id().to_string()).collect();
		let labels_to_write: Vec<(String, String, String)> = self
			.labels_by_id
			.iter()
			.filter(|(target_id, _)| path_entry_ids.contains(*target_id))
			.map(|(target_id, label)| {
				(target_id.clone(), label.clone(), self.label_timestamps_by_id[target_id].clone())
			})
			.collect();

		if self.persist {
			// Build label entries
			let mut parent_id = path_without_labels.last().map(|entry| entry.id().to_string());
			let mut label_entries: Vec<SessionEntry> = Vec::new();
			for (target_id, label, label_timestamp) in labels_to_write {
				let mut taken = path_entry_ids.clone();
				taken.extend(label_entries.iter().map(|entry| entry.id().to_string()));
				let label_entry = SessionEntry::Label(LabelEntry {
					base: entry_base_with_timestamp(
						generate_id(|id| taken.contains(id)),
						parent_id.clone(),
						label_timestamp,
					),
					target_id,
					label: Some(label),
				});
				parent_id = Some(label_entry.id().to_string());
				label_entries.push(label_entry);
			}

			self.file_entries = std::iter::once(FileEntry::Header(header))
				.chain(path_without_labels.into_iter().map(FileEntry::Entry))
				.chain(label_entries.into_iter().map(FileEntry::Entry))
				.collect();
			self.session_id = new_session_id;
			self.session_file = Some(new_session_file.clone());
			self.build_index();

			// Use the same rule as persist_entry(): write now if the branched
			// path already has a conversation, otherwise let persist_entry()
			// create the file later.
			if self.has_conversation() {
				self.rewrite_file()?;
				self.flushed = true;
			} else {
				self.flushed = false;
			}

			return Ok(Some(new_session_file));
		}

		// In-memory mode: replace current session with the path + labels
		let mut parent_id = path_without_labels.last().map(|entry| entry.id().to_string());
		let mut label_entries: Vec<SessionEntry> = Vec::new();
		for (target_id, label, label_timestamp) in labels_to_write {
			let mut taken: IndexSet<String> = path_entry_ids.clone();
			taken.extend(label_entries.iter().map(|entry| entry.id().to_string()));
			let label_entry = SessionEntry::Label(LabelEntry {
				base: entry_base_with_timestamp(
					generate_id(|id| taken.contains(id)),
					parent_id.clone(),
					label_timestamp,
				),
				target_id,
				label: Some(label),
			});
			parent_id = Some(label_entry.id().to_string());
			label_entries.push(label_entry);
		}
		self.file_entries = std::iter::once(FileEntry::Header(header))
			.chain(path_without_labels.into_iter().map(FileEntry::Entry))
			.chain(label_entries.into_iter().map(FileEntry::Entry))
			.collect();
		self.session_id = new_session_id;
		self.build_index();
		Ok(None)
	}

	// =========================================================================
	// Constructors (upstream statics)
	// =========================================================================

	/// Create a new session (`SessionManager.create`).
	///
	/// - `cwd`: working directory (stored in session header)
	/// - `session_dir`: optional session directory; if omitted, uses default
	///   (~/.pi/agent/sessions/&lt;encoded-cwd&gt;/)
	pub fn create(
		cwd: &str,
		session_dir: Option<&str>,
		options: Option<NewSessionOptions>,
	) -> Result<Self, SessionManagerError> {
		let dir = match session_dir {
			Some(dir) => normalize_path(dir, &PathInputOptions::default()),
			None => get_default_session_dir(cwd)?,
		};
		Self::new(cwd, &dir, None, true, options.as_ref(), None)
	}

	/// Open a specific session file (`SessionManager.open`).
	///
	/// - `path`: path to session file
	/// - `session_dir`: optional session directory for /new or /branch; if
	///   omitted, derives from file's parent
	/// - `cwd_override`: optional cwd override instead of the session header
	///   cwd
	pub fn open(
		path: &str,
		session_dir: Option<&str>,
		cwd_override: Option<&str>,
	) -> Result<Self, SessionManagerError> {
		let resolved_path = resolve_path_cwd(path);
		let mut header: Option<SessionHeader> = None;
		let mut preloaded_file_entries: Option<Vec<FileEntry>> = None;
		if cwd_override.is_none() && std::path::Path::new(&resolved_path).exists() {
			match read_session_header(&resolved_path) {
				Ok(found) => header = found,
				// The bounded scan is only a discovery optimization. A full
				// load remains authoritative for legacy files with very large
				// headers or prefixes.
				Err(SessionManagerError::HeaderScanLimit(..)) => {
					preloaded_file_entries = Some(load_entries_from_file(&resolved_path)?);
					header = preloaded_file_entries
						.as_ref()
						.and_then(|entries| entries.first())
						.and_then(|first| first.as_session_header().cloned());
				}
				Err(error) => return Err(error),
			}
		}
		let cwd = cwd_override
			.map(String::from)
			.or_else(|| header.as_ref().map(|header| header.cwd.clone()))
			.unwrap_or_else(current_dir_string);
		// If no sessionDir provided, derive from file's parent directory
		let dir = match session_dir {
			Some(dir) => normalize_path(dir, &PathInputOptions::default()),
			None => std::path::Path::new(&resolved_path)
				.parent()
				.map(|parent| parent.to_string_lossy().into_owned())
				.unwrap_or_else(|| ".".to_string()),
		};
		Self::new(&cwd, &dir, Some(resolved_path), true, None, preloaded_file_entries)
	}

	/// Continue the most recent session, or create new if none
	/// (`SessionManager.continueRecent`).
	pub fn continue_recent(cwd: &str, session_dir: Option<&str>) -> Result<Self, SessionManagerError> {
		let dir = match session_dir {
			Some(dir) => normalize_path(dir, &PathInputOptions::default()),
			None => get_default_session_dir(cwd)?,
		};
		let filter_cwd = session_dir.is_some() && dir != get_default_session_dir_path(cwd, &get_agent_dir());
		let most_recent = find_most_recent_session(&dir, if filter_cwd { Some(cwd) } else { None });
		if let Some(most_recent) = most_recent {
			return Self::new(cwd, &dir, Some(most_recent), true, None, None);
		}
		Self::new(cwd, &dir, None, true, None, None)
	}

	/// Create an in-memory session (no file persistence), optionally from
	/// entries held outside the filesystem (`SessionManager.inMemory`).
	pub fn in_memory(
		cwd: Option<&str>,
		options: Option<NewSessionOptions>,
		entries: Option<Vec<FileEntry>>,
	) -> Result<Self, SessionManagerError> {
		Self::new(
			&cwd.map(String::from).unwrap_or_else(current_dir_string),
			"",
			None,
			false,
			options.as_ref(),
			entries,
		)
	}

	/// Fork a session from another project directory into the current project
	/// (`SessionManager.forkFrom`). Creates a new session in the target cwd
	/// with the full history from the source session.
	pub fn fork_from(
		source_path: &str,
		target_cwd: &str,
		session_dir: Option<&str>,
		options: Option<NewSessionOptions>,
	) -> Result<Self, SessionManagerError> {
		let resolved_source_path = resolve_path_cwd(source_path);
		let resolved_target_cwd = resolve_path_cwd(target_cwd);
		let source_entries = load_entries_from_file(&resolved_source_path)?;
		if source_entries.is_empty() {
			return Err(SessionManagerError::ForkSourceInvalid(resolved_source_path));
		}

		let source_header = source_entries.iter().find_map(|entry| entry.as_session_header().cloned());
		let Some(_) = source_header else {
			return Err(SessionManagerError::ForkSourceNoHeader(resolved_source_path));
		};

		let dir = match session_dir {
			Some(dir) => normalize_path(dir, &PathInputOptions::default()),
			None => get_default_session_dir(&resolved_target_cwd)?,
		};
		if !std::path::Path::new(&dir).exists() {
			std::fs::create_dir_all(&dir)?;
		}

		// Create new session file with new ID but forked content
		if let Some(id) = options.as_ref().and_then(|options| options.id.as_deref()) {
			assert_valid_session_id(id)?;
		}
		let new_session_id = options.and_then(|options| options.id).unwrap_or_else(create_session_id);
		let timestamp = to_iso_timestamp(now_ms());
		let file_timestamp = timestamp.replace([':', '.'], "-");
		let new_session_file = join(&dir, &format!("{file_timestamp}_{new_session_id}.jsonl"));

		// Write new header pointing to source as parent, with updated cwd
		let new_header = SessionHeader {
			header_type: SessionHeaderType::Session,
			version: Some(CURRENT_SESSION_VERSION),
			id: new_session_id,
			timestamp,
			cwd: resolved_target_cwd.clone(),
			parent_session: Some(resolved_source_path),
		};
		{
			let mut file =
				std::fs::OpenOptions::new().write(true).create_new(true).open(&new_session_file)?;
			writeln!(file, "{}", serde_json::to_string(&new_header).expect("header serialization"))?;
		}

		// Copy all non-header entries from source
		{
			let mut file = std::fs::OpenOptions::new().append(true).open(&new_session_file)?;
			for entry in &source_entries {
				if let Some(session_entry) = entry.as_entry() {
					writeln!(
						file,
						"{}",
						serde_json::to_string(session_entry).expect("session entry serialization")
					)?;
				}
			}
		}

		Self::new(&resolved_target_cwd, &dir, Some(new_session_file), true, None, None)
	}

	/// Find an exact session ID without loading transcript bodies
	/// (`SessionManager.findById`).
	pub fn find_by_id(
		cwd: &str,
		id: &str,
		session_dir: Option<&str>,
	) -> Result<Option<String>, SessionManagerError> {
		let dir = match session_dir {
			Some(dir) => normalize_path(dir, &PathInputOptions::default()),
			None => get_default_session_dir(cwd)?,
		};
		let filter_cwd = session_dir.is_some() && dir != get_default_session_dir_path(cwd, &get_agent_dir());
		let resolved_cwd = resolve_path_cwd(cwd);

		let Ok(dir_entries) = std::fs::read_dir(&dir) else {
			// Exact session discovery is best-effort, matching list().
			return Ok(None);
		};
		for entry in dir_entries.flatten() {
			let file_name = entry.file_name().to_string_lossy().into_owned();
			if !file_name.ends_with(".jsonl") {
				continue;
			}
			let path = join(&dir, &file_name);
			let Some(header) = read_session_header_for_discovery(&path) else { continue };
			if header.id != id {
				continue;
			}
			if filter_cwd && !session_cwd_matches(Some(&header.cwd), &resolved_cwd) {
				continue;
			}
			return Ok(Some(path));
		}
		Ok(None)
	}

	/// List all sessions for a directory (`SessionManager.list`).
	///
	/// - `cwd`: working directory (used to compute default session directory)
	/// - `session_dir`: optional session directory; if omitted, uses default
	/// - `on_progress`: optional callback for progress updates (loaded, total,
	///   partial sessions)
	pub async fn list(
		cwd: &str,
		session_dir: Option<&str>,
		on_progress: Option<SessionListProgress>,
		signal: Option<pi_ai::utils::abort::AbortSignal>,
	) -> Result<Vec<SessionInfo>, SessionListError> {
		let dir = match session_dir {
			Some(dir) => normalize_path(dir, &PathInputOptions::default()),
			None => get_default_session_dir(cwd)?,
		};
		let filter_cwd = session_dir.is_some() && dir != get_default_session_dir_path(cwd, &get_agent_dir());
		let resolved_cwd = resolve_path_cwd(cwd);
		let include_session =
			|session: &SessionInfo| !filter_cwd || session_cwd_matches(Some(&session.cwd), &resolved_cwd);
		let progress: Option<SessionListProgress> = on_progress.map(|on_progress| {
			// SessionListProgress requires 'static: capture owned state.
			let resolved_cwd = resolved_cwd.clone();
			Arc::new(move |loaded: usize, total: usize, partial_sessions: Option<&[SessionInfo]>| {
				if let Some(partial_sessions) = partial_sessions {
					let filtered: Vec<SessionInfo> = partial_sessions
						.iter()
						.filter(|session| !filter_cwd || session_cwd_matches(Some(&session.cwd), &resolved_cwd))
						.cloned()
						.collect();
					on_progress(loaded, total, Some(&filtered));
				} else {
					on_progress(loaded, total, None);
				}
			}) as SessionListProgress
		});
		let sessions = list_sessions_from_dir(&dir, progress.as_ref(), signal).await?;
		Ok(sort_session_infos(sessions.into_iter().filter(include_session).collect()))
	}

	/// List all sessions across all project directories
	/// (`SessionManager.listAll` — no-directory overload).
	pub async fn list_all(
		on_progress: Option<SessionListProgress>,
		signal: Option<pi_ai::utils::abort::AbortSignal>,
	) -> Result<Vec<SessionInfo>, SessionListError> {
		// Upstream catches everything, rethrows only an aborted signal, and
		// otherwise returns [].
		match list_all_impl(get_sessions_dir(), on_progress, signal.clone()).await {
			Ok(sessions) => Ok(sessions),
			Err(SessionListError::Aborted(reason)) => Err(SessionListError::Aborted(reason)),
			Err(_) => {
				if let Some(signal) = &signal {
					signal.check()?;
				}
				Ok(Vec::new())
			}
		}
	}

	/// `SessionManager.listAll(sessionDir, ...)` — list all sessions in one
	/// flat directory.
	pub async fn list_all_in_dir(
		session_dir: &str,
		on_progress: Option<SessionListProgress>,
		signal: Option<pi_ai::utils::abort::AbortSignal>,
	) -> Result<Vec<SessionInfo>, SessionListError> {
		let sessions = list_sessions_from_dir(session_dir, on_progress.as_ref(), signal).await?;
		Ok(sort_session_infos(sessions))
	}
}

fn list_all_impl(
	sessions_dir: String,
	on_progress: Option<SessionListProgress>,
	signal: Option<pi_ai::utils::abort::AbortSignal>,
) -> impl Future<Output = Result<Vec<SessionInfo>, SessionListError>> {
	async move {
		if let Some(signal) = &signal {
			signal.check()?;
		}
		if !std::path::Path::new(&sessions_dir).exists() {
			return Ok(Vec::new());
		}
		let entries = std::fs::read_dir(&sessions_dir)?;
		let dirs: Vec<String> = entries
			.flatten()
			.filter(|entry| {
				let Ok(file_type) = entry.file_type() else { return false };
				file_type.is_dir() || file_type.is_symlink()
			})
			.map(|entry| join(&sessions_dir, &entry.file_name().to_string_lossy()))
			.collect();

		let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_SESSION_DISCOVERY_LOADS));
		let mut dir_handles = Vec::with_capacity(dirs.len());
		for dir in dirs {
			let semaphore = semaphore.clone();
			dir_handles.push(tokio::spawn(async move {
				let _permit = semaphore.acquire_owned().await;
				let Ok(entries) = std::fs::read_dir(&dir) else { return Vec::new() };
				entries
					.flatten()
					.map(|entry| entry.file_name().to_string_lossy().into_owned())
					.filter(|file| file.ends_with(".jsonl"))
					.map(|file| join(&dir, &file))
					.collect::<Vec<_>>()
			}));
		}
		let mut all_files: Vec<String> = Vec::new();
		for handle in dir_handles {
			all_files.extend(handle.await.unwrap_or_default());
		}

		let mut candidates: Vec<(String, i64)> = Vec::with_capacity(all_files.len());
		for path in all_files {
			let path_for_stat = path.clone();
			let mtime = tokio::task::spawn_blocking(move || {
				std::fs::metadata(&path_for_stat).ok().map(|metadata| modified_time_ms(&metadata))
			})
			.await
			.unwrap_or(None)
			.unwrap_or(-1);
			candidates.push((path, mtime));
		}
		candidates.sort_by(|a, b| {
			b.1.cmp(&a.1).then_with(|| {
				let base_a = std::path::Path::new(&a.0).file_name().unwrap_or_default();
				let base_b = std::path::Path::new(&b.0).file_name().unwrap_or_default();
				base_b.cmp(base_a)
			})
		});

		let total_files = candidates.len();
		let loaded = Arc::new(AtomicUsize::new(0));
		let first_candidate_loaded = Arc::new(AtomicUsize::new(0));
		let partial_sessions: Arc<Mutex<Vec<SessionInfo>>> = Arc::new(Mutex::new(Vec::new()));
		let paths: Vec<String> = candidates.into_iter().map(|(path, _)| path).collect();
		let results = build_session_infos_with_concurrency(
			paths,
			{
				let loaded = loaded.clone();
				let first_candidate_loaded = first_candidate_loaded.clone();
				let partial_sessions = partial_sessions.clone();
				Arc::new(move |info, index| {
					let loaded = loaded.fetch_add(1, Ordering::SeqCst) + 1;
					if index == 0 {
						first_candidate_loaded.store(1, Ordering::SeqCst);
					}
					if let Some(info) = info {
						partial_sessions.lock().expect("partial sessions lock").push(info.clone());
					}
					let Some(progress) = &on_progress else { return };
					let publish_partial = first_candidate_loaded.load(Ordering::SeqCst) > 0
						&& (index == 0
							|| loaded % ALL_SESSION_LIST_PUBLISH_INTERVAL == 0
							|| loaded == total_files);
					let snapshot: Option<Vec<SessionInfo>> = if publish_partial {
						Some(sort_session_infos(
							partial_sessions.lock().expect("partial sessions lock").clone(),
						))
					} else {
						None
					};
					progress(loaded, total_files, snapshot.as_deref());
				})
			},
			signal,
		)
		.await?;
		Ok(sort_session_infos(results.into_iter().flatten().collect()))
	}
}

fn current_dir_string() -> String {
	std::env::current_dir()
		.map(|dir| dir.to_string_lossy().into_owned())
		.unwrap_or_else(|_| ".".to_string())
}

/// `name.replace(/[\r\n]+/g, " ").trim()` without a regex dependency.
fn collapse_newlines(name: &str) -> String {
	let mut sanitized = String::with_capacity(name.len());
	let mut in_newline_run = false;
	for c in name.chars() {
		if c == '\r' || c == '\n' {
			if !in_newline_run {
				sanitized.push(' ');
				in_newline_run = true;
			}
		} else {
			sanitized.push(c);
			in_newline_run = false;
		}
	}
	sanitized.trim().to_string()
}

#[cfg(test)]
#[path = "session_manager_tests.rs"]
mod session_manager_tests;
