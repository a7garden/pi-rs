//! Port of `packages/durable/src/harness/agent.ts`.
//!
//! Deviations: D1 — `AgentState`/`AgentChange` are typed serde structs (see
//! `harness/types.rs`), so `applyChange` matches variants instead of
//! JS object surgery; `Object.assign` copies become struct clones.
//! D2 — wrappers that fail are reported as error text (types D5).

use std::collections::{BTreeSet, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use indexmap::IndexMap;

use serde_json::{json, Value};

use crate::documents::{DocDefinition, DocToken};
use crate::types::{DocumentFork, DocumentHistory};
use crate::harness::types::{
	Agent, AgentChange, AgentState, CompactionPolicy, CompactionPolicyOverrides, Extension,
	ExtensionsChange, HarnessSettings, ModelRef, ProgressPolicy, ProgressPolicyOverrides,
	RegistrySnapshot, RetryPolicyOverrides, ConversationRetryPolicy, SectionRender, Settings,
	SetOrClear, ToolRegistration, ToolsChange, ToolsSelection, ExtensionSelection, PromptSection,
	ConversationId, ConversationRecord,
};
use crate::session::Tx;
use crate::tasks::BoxFuture;

use pi_chord::context::Context;

/// Default durable generation attempt retries.
pub const DEFAULT_RETRY_POLICY: ConversationRetryPolicy = ConversationRetryPolicy {
	enabled: true,
	max_retries: 3,
	base_delay_ms: 2000,
	max_agent_delay_ms: Some(60000),
};

/// Default automatic compaction thresholds.
pub const DEFAULT_COMPACTION_POLICY: CompactionPolicy = CompactionPolicy {
	enabled: true,
	reserve_tokens: 16384,
	keep_recent_tokens: 20000,
	background_tokens: 32768,
};

/// Default running-progress commit cadence.
pub const DEFAULT_PROGRESS_POLICY: ProgressPolicy = ProgressPolicy {
	partial_interval_ms: 100,
	output_interval_ms: 100,
};

/// The reserved section key of the agent's `instructions`.
pub const INSTRUCTIONS_KEY: &str = "instructions";

fn agent_definition() -> DocDefinition {
	// Built-in agent document; rewindable so forks start from the agent at
	// their fork entry.
	DocDefinition::conversation("pi.agent", 1, DocumentHistory::Rewindable, DocumentFork::AsOf, serde_json::Map::new)
		.with_checkpoint_when(|_, _, _| true)
}

static AGENT_DOC: std::sync::LazyLock<DocToken> =
	std::sync::LazyLock::new(|| DocToken::new(agent_definition()));

/// The conversation agent-state document.
pub fn agent_doc() -> DocToken {
	AGENT_DOC.clone()
}

/// Resolve the host settings: every field over its built-in default, object
/// fields merged (see `harness::agent::resolve_settings`).
pub fn resolve_settings(settings: Option<&HarnessSettings>) -> Settings {
	let defaults = HarnessSettings::default();
	let settings = settings.unwrap_or(&defaults);
	Settings {
		extensions: settings.extensions.clone(),
		stream: settings.stream.clone().unwrap_or_default(),
		retry: merge_retry(&DEFAULT_RETRY_POLICY, settings.retry),
		compaction: merge_compaction(&DEFAULT_COMPACTION_POLICY, settings.compaction),
		progress: merge_progress(&DEFAULT_PROGRESS_POLICY, settings.progress),
		tool_execution: settings.tool_execution.unwrap_or(crate::harness::api::ToolExecutionMode::Parallel),
		steering_mode: settings.steering_mode.unwrap_or(crate::harness::api::QueueMode::OneAtATime),
		follow_up_mode: settings.follow_up_mode.unwrap_or(crate::harness::api::QueueMode::OneAtATime),
	}
}

fn merge_retry(base: &ConversationRetryPolicy, over: Option<RetryPolicyOverrides>) -> ConversationRetryPolicy {
	let Some(over) = over else { return base.clone() };
	ConversationRetryPolicy {
		enabled: over.enabled.unwrap_or(base.enabled),
		max_retries: over.max_retries.unwrap_or(base.max_retries),
		base_delay_ms: over.base_delay_ms.unwrap_or(base.base_delay_ms),
		max_agent_delay_ms: over.max_agent_delay_ms.or(base.max_agent_delay_ms),
	}
}

fn merge_compaction(base: &CompactionPolicy, over: Option<CompactionPolicyOverrides>) -> CompactionPolicy {
	let Some(over) = over else { return base.clone() };
	CompactionPolicy {
		enabled: over.enabled.unwrap_or(base.enabled),
		reserve_tokens: over.reserve_tokens.unwrap_or(base.reserve_tokens),
		keep_recent_tokens: over.keep_recent_tokens.unwrap_or(base.keep_recent_tokens),
		background_tokens: over.background_tokens.unwrap_or(base.background_tokens),
	}
}

fn merge_progress(base: &ProgressPolicy, over: Option<ProgressPolicyOverrides>) -> ProgressPolicy {
	let Some(over) = over else { return base.clone() };
	ProgressPolicy {
		partial_interval_ms: over.partial_interval_ms.unwrap_or(base.partial_interval_ms),
		output_interval_ms: over.output_interval_ms.unwrap_or(base.output_interval_ms),
	}
}

/// Apply one change to `pi.agent`: a given field replaces the stored one,
/// `Clear` removes it, `None` changes nothing.
pub fn configure(
	tx: &Tx<'_>,
	conversation_id: ConversationId,
	change: &AgentChange,
) -> Result<(), String> {
	let draft = tx.doc(&agent_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
	let mut state: AgentState =
		serde_json::from_value(Value::Object(draft.value().unwrap_or_default())).unwrap_or_default();
	apply_change(&mut state, change);
	store_agent_state(&draft, &state)
}

/// Per-field write of the erased agent state (drafts reject whole-root
/// replacement; per-key sets keep upstream's delta granularity).
fn store_agent_state(draft: &crate::session::DocDraft, state: &AgentState) -> Result<(), String> {
	use pi_chord::delta::Seg;
	let value = serde_json::to_value(state).expect("AgentState serializes");
	let obj = value.as_object().expect("AgentState is an object");
	for key in ["model", "thinkingLevel", "extensions", "tools", "instructions", "cwd"] {
		match obj.get(key) {
			Some(value) => draft.set(&[Seg::key(key)], value.clone())?,
			None => {
				if draft.has(&[], key)? {
					draft.delete(&[Seg::key(key)])?;
				}
			}
		}
	}
	Ok(())
}

/// `addTools` of a tool round: a list gets each name it lacks appended,
/// `{ remove }` loses the names, and unset tools already offer every tool,
/// so nothing is written.
pub fn add_tools(
	tx: &Tx<'_>,
	conversation_id: ConversationId,
	added: &[String],
) -> Result<(), String> {
	let draft = tx.doc(&agent_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
	let mut state: AgentState =
		serde_json::from_value(Value::Object(draft.value().unwrap_or_default())).unwrap_or_default();
	let Some(tools) = &mut state.tools else { return Ok(()) };
	match tools {
		ToolsSelection::List(names) => {
			for name in added {
				if !names.contains(name) {
					names.push(name.clone());
				}
			}
		}
		ToolsSelection::Remove { remove } => {
			if remove.iter().any(|name| added.contains(name)) {
				remove.retain(|name| !added.contains(name));
			}
		}
	}
	store_agent_state(&draft, &state)?;
	Ok(())
}

fn apply_change(state: &mut AgentState, change: &AgentChange) {
	set(&mut state.model, &change.model);
	set(&mut state.thinking_level, &change.thinking_level);
	if let Some(extensions) = &change.extensions {
		match extensions {
			SetOrClear::Clear => state.extensions = None,
			SetOrClear::Set(ExtensionsChange::List(list)) => {
				state.extensions = Some(ExtensionSelection::List(names_of_ext(list)));
			}
			SetOrClear::Set(ExtensionsChange::Edit { add, remove }) => {
				state.extensions = Some(ExtensionSelection::Edit {
					add: (!add.is_empty()).then(|| names_of_ext(add)),
					remove: (!remove.is_empty()).then(|| names_of_ext(remove)),
				});
			}
		}
	}
	if let Some(tools) = &change.tools {
		match tools {
			SetOrClear::Clear => state.tools = None,
			SetOrClear::Set(ToolsChange::List(list)) => {
				state.tools = Some(ToolsSelection::List(list.iter().map(|t| t.name().to_string()).collect()));
			}
			SetOrClear::Set(ToolsChange::Remove(list)) => {
				state.tools = Some(ToolsSelection::Remove {
					remove: list.iter().map(|t| t.name().to_string()).collect(),
				});
			}
		}
	}
	set(&mut state.instructions, &change.instructions);
	set(&mut state.cwd, &change.cwd);
}

fn set<T: Clone>(target: &mut Option<T>, change: &Option<SetOrClear<T>>) {
	match change {
		None => {}
		Some(SetOrClear::Clear) => *target = None,
		Some(SetOrClear::Set(value)) => *target = Some(value.clone()),
	}
}

fn names_of_ext(items: &[Extension]) -> Vec<String> {
	items.iter().map(|item| item.name.clone()).collect()
}

/// Built-in part of every Harness commit that creates or forks a
/// conversation, for `pi.agent`: a fork keeps its `asOf` copy; a new
/// task-owned conversation copies the stored agent of its owner task's
/// conversation; a new ownerless one starts empty.
pub fn create_agent(tx: &Tx<'_>, conversation: &ConversationRecord) -> Result<(), String> {
	if conversation.parent.is_some() {
		return Ok(());
	}
	// Upstream creates the conversation's own agent document first, then
	// returns for ownerless conversations: every non-fork conversation has a
	// (possibly empty) `pi.agent` document.
	let draft = tx
		.doc(&agent_doc(), Some(conversation.id.0), None, None)
		.map_err(|e| e.to_string())?;
	let owner = conversation.owner.as_ref().map(|owner| owner.conversation_id);
	let Some(owner_id) = owner else { return Ok(()) };
	let owner_draft = tx.doc(&agent_doc(), Some(owner_id.0), None, None).map_err(|e| e.to_string())?;
	let owner_state = owner_draft.value().unwrap_or_default();
	// A task-owned conversation copies the owner's whole agent state; write
	// it field by field.
	let owner_agent: AgentState = serde_json::from_value(Value::Object(owner_state)).unwrap_or_default();
	store_agent_state(&draft, &owner_agent)
}

/// Handlers of the selected extensions' hooks for a task name, in extension
/// order.
pub fn agent_hooks(agent: &Agent, task_name: &str) -> Vec<Arc<dyn crate::tasks::HookHandler>> {
	let mut handlers = Vec::new();
	for extension in &agent.extensions {
		for hook in &extension.hooks {
			if hook.task == task_name {
				handlers.push(hook.handlers.clone());
			}
		}
	}
	handlers
}

/// Resolve an agent from its stored state (absent: every field unset), a
/// registry snapshot, and resolved settings. A wrapper that fails or renames
/// drops its target and is reported; a wrapper without a target does
/// nothing.
pub fn resolve_agent(
	state: Option<&AgentState>,
	snapshot: &dyn RegistrySnapshot,
	settings: &Settings,
	report: &dyn Fn(String),
) -> Agent {
	let extensions = select_extensions(state.and_then(|s| s.extensions.as_ref()), snapshot, settings);

	let mut composed: IndexMap<String, Arc<ToolRegistration>> = IndexMap::new();
	for extension in &extensions {
		for tool in &extension.tools {
			if let Some(existing) = composed.get_mut(&tool.tool.name) {
				*existing = tool.clone();
			} else {
				composed.insert(tool.tool.name.clone(), tool.clone());
			}
		}
	}
	let mut sections: IndexMap<String, PromptSection> = IndexMap::new();
	for extension in &extensions {
		for section in &extension.sections {
			if let Some(existing) = sections.get_mut(&section.key) {
				*existing = section.clone();
			} else {
				sections.insert(section.key.clone(), section.clone());
			}
		}
	}
	for extension in &extensions {
		for wrap in &extension.wraps {
			match wrap {
				crate::harness::types::Wrap::Tool(tool_wrap) => {
					apply_wrap(&mut composed, &tool_wrap.tool, |tool| (tool_wrap.wrap)(tool.clone()), |t| t.name().to_string(), report);
				}
				crate::harness::types::Wrap::Section(section_wrap) => {
					apply_wrap(
						&mut sections,
						&section_wrap.section,
						|section| (section_wrap.wrap)(section),
						|s| s.key.clone(),
						report,
					);
				}
			}
		}
	}

	let filter = state.and_then(|s| s.tools.as_ref());
	let tools: Vec<Arc<ToolRegistration>> = match filter {
		None => composed.into_values().collect(),
		Some(ToolsSelection::List(names)) => {
			let mut tools = Vec::new();
			let mut seen = HashSet::new();
			for name in names {
				if seen.insert(name) {
					if let Some(tool) = composed.get(name) {
						tools.push(tool.clone());
					}
				}
			}
			tools
		}
		Some(ToolsSelection::Remove { remove }) => {
			let removed: BTreeSet<&String> = remove.iter().collect();
			composed.into_values().filter(|tool| !removed.contains(&tool.tool.name)).collect()
		}
	};

	let instructions = state.and_then(|s| s.instructions.clone());
	let mut agent_sections: Vec<PromptSection> = sections.into_values().collect();
	if let Some(instructions) = &instructions {
		let text = instructions.clone();
		let render: SectionRender =
			Arc::new(move |_input: crate::harness::types::PromptInput, _context: Context| {
				let text = text.clone();
				let future: BoxFuture<Result<Option<String>, String>> = Box::pin(async move { Ok(Some(text)) });
				future
			});
		agent_sections.push(PromptSection { key: INSTRUCTIONS_KEY.to_string(), render, tag: None });
	}

	Agent {
		model: state.and_then(|s| s.model.clone()),
		thinking_level: state
			.and_then(|s| s.thinking_level)
			.unwrap_or(pi_ai::types::ModelThinkingLevel::Off),
		extensions,
		tools,
		sections: agent_sections,
		instructions,
		cwd: state.and_then(|s| s.cwd.clone()),
	}
}

/// Selected installed extensions: the stored list, or the default selection
/// edited by `{ add, remove }`.
fn select_extensions(
	stored: Option<&ExtensionSelection>,
	snapshot: &dyn RegistrySnapshot,
	settings: &Settings,
) -> Vec<Arc<Extension>> {
	let selected: Vec<String> = match stored {
		Some(ExtensionSelection::List(names)) => names.clone(),
		stored => {
			let base: Vec<String> = match &settings.extensions {
				Some(extensions) => extensions.iter().map(|e| e.name.clone()).collect(),
				None => snapshot.installed().iter().map(|e| e.name.clone()).collect(),
			};
			let removed: BTreeSet<&String> = match stored {
				Some(ExtensionSelection::Edit { remove: Some(remove), .. }) => remove.iter().collect(),
				_ => BTreeSet::new(),
			};
			let add: Vec<String> = match stored {
				Some(ExtensionSelection::Edit { add: Some(add), .. }) => add.clone(),
				_ => Vec::new(),
			};
			base.into_iter().chain(add).filter(|name| !removed.contains(name)).collect()
		}
	};
	let mut extensions = Vec::new();
	let mut seen = HashSet::new();
	for name in &selected {
		if seen.insert(name) {
			if let Some(extension) = snapshot.extension(name) {
				extensions.push(extension);
			}
		}
	}
	extensions
}

fn apply_wrap<T: Clone>(
	items: &mut IndexMap<String, T>,
	target: &str,
	wrap: impl Fn(T) -> T,
	name_of: impl Fn(&T) -> String,
	report: &dyn Fn(String),
) {
	let Some(item) = items.get(target).cloned() else { return };
	let wrapped = catch_unwind(AssertUnwindSafe(|| {
		let wrapped = wrap(item);
		let wrapped_name = name_of(&wrapped);
		if wrapped_name == target {
			Ok(wrapped)
		} else {
			Err(format!("Wrapper renamed {target} to {wrapped_name}"))
		}
	}));
	match wrapped {
		Ok(Ok(wrapped)) => {
			if let Some(current) = items.get_mut(target) {
				*current = wrapped;
			}
		}
		Ok(Err(error)) => {
			items.shift_remove(target);
			report(error);
		}
		Err(payload) => {
			let message = payload
				.downcast_ref::<String>()
				.cloned()
				.or_else(|| payload.downcast_ref::<&str>().map(|message| (*message).to_owned()))
				.unwrap_or_else(|| "Wrapper panicked".to_string());
			items.shift_remove(target);
			report(message);
		}
	}
}
