//! Port of `packages/durable/src/harness/prompt.ts`.
//!
//! Deviations: D1 — `renderSections` reports errors through a
//! `report(String)` callback (erased thrown value, matching harness types
//! D5). D2 — maps are `serde_json::Map<String, Value>` (ordered like JS objects
//! with string keys under `preserve_order`). D3 — `plan_system_entries`
//! returns erased `EntryDraft`s carrying a single `SystemMessage` in
//! `model` (upstream `TypedEntryDraft<never>`).

use std::collections::BTreeMap;

use indexmap::IndexMap;

use serde_json::Value;
use pi_ai::types::{Message, SystemMessage, Tool, ToolReference};
use pi_ai::utils::transcript::{declarations_equal, get_current_tools, to_tool_declaration};
use pi_chord::context::Context;

use crate::entries::{define_entry, Entry};
use crate::harness::types::{ContextView, PromptInput, PromptSection};
use crate::types::{ContextEdit, ContextEditAction, EntryDraft, EntryId};

/// Sections in effect after replaying system messages in order.
pub fn replay_sections(messages: &[Message]) -> serde_json::Map<String, Value> {
	let mut shown = serde_json::Map::new();
	for message in messages {
		let Message::System(system) = message else { continue };
		for (key, value) in system.sections.iter().flatten() {
			match value {
				None => {
					shown.shift_remove(key);
				}
				Some(value) => {
					if let Some(existing) = shown.get_mut(key) {
						*existing = Value::String(value.clone());
					} else {
						shown.insert(key.clone(), Value::String(value.clone()));
					}
				}
			}
		}
	}
	shown
}

/// Render the agent's sections in order. `Ok(None)` omits a section; tagged
/// text is wrapped as `<key>\n...\n</key>`. A section that fails keeps its
/// shown text, if any, and is reported; errors after `context` is aborted
/// propagate.
pub async fn render_sections(
	sections: &[PromptSection],
	input: PromptInput,
	shown: &serde_json::Map<String, Value>,
	report: &dyn Fn(String),
	context: Context,
) -> Result<serde_json::Map<String, Value>, String> {
	let mut desired = serde_json::Map::new();
	for section in sections {
		match (section.render)(input.clone(), context.clone()).await {
			Err(error) => {
				if context.abort_signal().map(|s| s.aborted()).unwrap_or(false) {
					return Err(error);
				}
				report(error);
				if let Some(kept) = shown.get(&section.key).and_then(|value| value.as_str()) {
					desired.insert(section.key.clone(), Value::String(kept.to_string()));
				}
			}
			Ok(None) => {}
			Ok(Some(text)) => {
				let text = if section.tag == Some(false) {
					text
				} else {
					format!("<{}>\n{}\n</{}>", section.key, text, section.key)
				};
				desired.insert(section.key.clone(), Value::String(text));
			}
		}
	}
	Ok(desired)
}

/// Tool changes carried by a planned `pi.system` entry.
#[derive(Debug, Clone, Default)]
struct ToolChanges {
	tools_removed: Vec<ToolReference>,
	tools_added: Vec<Tool>,
}

/// The `pi.system` entry kind.
static SYSTEM_ENTRY: std::sync::LazyLock<Entry> = std::sync::LazyLock::new(|| define_entry("pi.system"));

/// Plan the `pi.system` entries that make the replayed sections and tools of
/// `view` equal `desired` and `tools` in values and order.
///
/// - A head marker with no later `pi.system` entry in context: one complete
///   baseline that omits every retained earlier `pi.system` entry, written
///   even when it restates the replayed values.
/// - Otherwise, when a minimal section patch would leave a different order:
///   remove every shown section, then re-add every desired section in order.
/// - Otherwise the minimal patch of changed values and `None` removals, or
///   nothing.
///
/// Tool changes ride on the last planned entry, or on one entry of their
/// own.
pub fn plan_system_entries(
	view: &ContextView,
	desired: &serde_json::Map<String, Value>,
	tools: &[Tool],
	timestamp: i64,
) -> Vec<EntryDraft> {
	let system: &Entry = &SYSTEM_ENTRY;
	let head = view.head.as_ref();
	if let Some(head) = head.filter(|head| {
		!view.entries.iter().any(|entry| system.is(Some(entry)) && entry.id > head.id)
	}) {
			let mut edits = Vec::new();
			for entry in &view.entries {
				if system.is(Some(entry)) {
					edits.push(ContextEdit { target: entry.id, action: ContextEditAction::Omit, messages: None });
				}
			}
			let baseline = system_entry(
				system,
				Some(
					desired
						.iter()
						.map(|(k, v)| (k.clone(), v.as_str().map(|s| s.to_string())))
						.collect(),
				),
				Some(ToolChanges { tools_removed: Vec::new(), tools_added: tools.iter().map(to_tool_declaration).collect() }),
				timestamp,
			);
			let mut baseline = baseline;
			if !edits.is_empty() {
				baseline.edits = Some(edits);
			}
			return vec![baseline];
	}
	let sections = plan_sections(&replay_sections(&view.messages), desired);
	let changes = plan_tools(&get_current_tools(&view.messages), tools);
	if changes.tools_removed.is_empty() && changes.tools_added.is_empty() {
		return sections
			.into_iter()
			.map(|patch| system_entry(system, Some(patch), None, timestamp))
			.collect();
	}
	if sections.is_empty() {
		return vec![system_entry(system, None, Some(changes), timestamp)];
	}
	let count = sections.len();
	sections
		.into_iter()
		.enumerate()
		.map(|(index, patch)| {
			let changes = if index == count - 1 { Some(changes.clone()) } else { None };
			system_entry(system, Some(patch), changes, timestamp)
		})
		.collect()
}

/// Tool changes from `offered` to `desired`. A changed declaration is
/// removed and re-added. Replay keeps retained tools in place and appends
/// additions; when that would not yield the desired order, every offered
/// tool is removed and every desired tool re-added in order.
fn plan_tools(offered: &[Tool], desired: &[Tool]) -> ToolChanges {
	let wanted: BTreeMap<&str, &Tool> = desired.iter().map(|tool| (tool.name.as_str(), tool)).collect();
	let kept: Vec<&Tool> = offered
		.iter()
		.filter(|tool| wanted.get(tool.name.as_str()).is_some_and(|next| declarations_equal(tool, next)))
		.collect();
	let kept_names: std::collections::BTreeSet<&str> = kept.iter().map(|tool| tool.name.as_str()).collect();
	let added: Vec<&Tool> = desired.iter().filter(|tool| !kept_names.contains(tool.name.as_str())).collect();
	let replayed: Vec<&str> = kept.iter().map(|t| t.name.as_str()).chain(added.iter().map(|t| t.name.as_str())).collect();
	if replayed.iter().enumerate().any(|(index, name)| desired[index].name.as_str() != *name) {
		return ToolChanges {
			tools_removed: offered.iter().map(|tool| ToolReference { name: tool.name.clone() }).collect(),
			tools_added: desired.iter().map(to_tool_declaration).collect(),
		};
	}
	ToolChanges {
		tools_removed: offered
			.iter()
			.filter(|tool| !kept_names.contains(tool.name.as_str()))
			.map(|tool| ToolReference { name: tool.name.clone() })
			.collect(),
		tools_added: added.iter().map(|tool| to_tool_declaration(tool)).collect(),
	}
}

/// Section patches: none, the minimal patch, or a remove-all/re-add-all pair
/// when the order would differ.
fn plan_sections(
	shown: &serde_json::Map<String, Value>,
	desired: &serde_json::Map<String, Value>,
) -> Vec<IndexMap<String, Option<String>>> {
	let patched_order: Vec<&String> = shown
		.keys()
		.filter(|key| desired.contains_key(*key))
		.chain(desired.keys().filter(|key| !shown.contains_key(*key)))
		.collect();
	let desired_order: Vec<&String> = desired.keys().collect();
	if patched_order.iter().enumerate().any(|(index, key)| *key != desired_order[index]) {
		let mut remove_all: IndexMap<String, Option<String>> = IndexMap::new();
		for key in shown.keys() {
			remove_all.insert(key.clone(), None);
		}
		let add_all: IndexMap<String, Option<String>> = desired
			.iter()
			.map(|(k, v)| (k.clone(), v.as_str().map(|s| s.to_string())))
			.collect();
		return vec![remove_all, add_all];
	}

	let mut patch: IndexMap<String, Option<String>> = IndexMap::new();
	for (key, value) in shown {
		let next = desired.get(key).and_then(|value| value.as_str());
		if next != value.as_str() {
			patch.insert(key.clone(), next.map(|s| s.to_string()));
		}
	}
	for (key, value) in desired {
		if !shown.contains_key(key) {
			patch.insert(key.clone(), value.as_str().map(|s| s.to_string()));
		}
	}
	if patch.is_empty() {
		Vec::new()
	} else {
		vec![patch]
	}
}

/// An erased `pi.system` entry draft carrying one system message.
fn system_entry(
	system: &Entry,
	sections: Option<IndexMap<String, Option<String>>>,
	tools: Option<ToolChanges>,
	timestamp: i64,
) -> EntryDraft {
	let message = SystemMessage {
		content: pi_ai::types::SystemContent::Text(String::new()),
		sections,
		tools_removed: tools.as_ref().filter(|c| !c.tools_removed.is_empty()).map(|c| c.tools_removed.clone()),
		tools_added: tools.filter(|c| !c.tools_added.is_empty()).map(|c| c.tools_added.clone()),
		timestamp,
	};
	let mut draft = EntryDraft { kind: Some(system.kind.clone()), ..Default::default() };
	draft.model = Some(vec![Message::System(message)]);
	draft
}

// Unused EntryId import guard: upstream signature keeps `at: EntryId`
// plumbing in callers.
#[allow(unused)]
fn _entry_id_marker(_: EntryId) {}
