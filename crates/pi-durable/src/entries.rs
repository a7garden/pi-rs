//! Port of `packages/durable/src/entries.ts`.
//!
//! Upstream `defineEntry<D>(kind)` returns a kind token with an `is` narrowing
//! guard; Rust ports the token as [`Entry`] with `is()` plus a data accessor.
//! The built-in kinds keep their exact upstream strings (part of the protocol).
//! (D1: upstream's type-level `D` payload parameter has no runtime counterpart;
//! typed data access goes through [`Entry::data`].)

use serde_json::Value;
use std::sync::LazyLock as Lazy;

use crate::types::EntryRecord;

/// Typed entry kind with a narrowing guard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
	pub kind: String,
}

impl Entry {
	/// Define a typed entry kind whose `is` guard narrows by `EntryRecord.kind`.
	pub fn define(kind: impl Into<String>) -> Self {
		let kind = kind.into();
		assert!(!kind.is_empty(), "Entry kind must be a non-empty string");
		Self { kind }
	}

	/// True when `entry` is present and carries this kind.
	pub fn is(&self, entry: Option<&EntryRecord>) -> bool {
		entry.is_some_and(|entry| entry.kind == self.kind)
	}

	/// The entry's typed `data` payload when `is(entry)` holds.
	pub fn data<'a>(&self, entry: Option<&'a EntryRecord>) -> Option<&'a Value> {
		if self.is(entry) {
			entry.and_then(|entry| entry.data.as_ref())
		} else {
			None
		}
	}
}

/// Define a typed entry kind whose `is` guard narrows by `EntryRecord.kind`.
pub fn define_entry(kind: impl Into<String>) -> Entry {
	Entry::define(kind)
}

/// User input: `model` is `[UserMessage]`. Written by submissions.
pub static USER_ENTRY: Lazy<Entry> = Lazy::new(|| Entry::define("pi.user"));
/// Provider result with any stop reason: `model` is `[AssistantMessage]`. Written by generation.
pub static ASSISTANT_ENTRY: Lazy<Entry> = Lazy::new(|| Entry::define("pi.assistant"));
/// Positional prompt and tool change: `model` is `[SystemMessage]` with empty `content`.
pub static SYSTEM_ENTRY: Lazy<Entry> = Lazy::new(|| Entry::define("pi.system"));
/// Tool result: `model` is `[ToolResultMessage]`, whose content ends with the rendered diagnostics
/// block; `data` holds the structured diagnostics, possibly none. Written by tool tasks, and by
/// generation for calls it did not offer.
pub static TOOL_RESULT_ENTRY: Lazy<Entry> = Lazy::new(|| Entry::define("pi.tool-result"));
/// Start of a new context: always `head: "self"`, with `model` absent for a plain reset or
/// `[UserMessage]` carrying the handoff text. Written by `Conversation.reset()` and the `handoff`
/// tool control.
pub static RESET_ENTRY: Lazy<Entry> = Lazy::new(|| Entry::define("pi.reset"));
/// Compaction summary: `model` is `[UserMessage]` with the wrapped summary, `head` the first kept
/// entry. Written by compaction tasks, directly or through a write submission.
pub static COMPACTION_ENTRY: Lazy<Entry> = Lazy::new(|| Entry::define("pi.compaction"));
