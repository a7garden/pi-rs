//! Port of `packages/durable/src/harness/registry.ts`.
//!
//! Deviations: D1 — erased `Tool` generic; extensions hold
//! `Arc<ToolRegistration>`s. D2 — listeners are `Arc<dyn Fn()>` and
//! `subscribe` returns the erased [`Unsubscribe`] disposer.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use indexmap::IndexMap;

use crate::harness::agent::INSTRUCTIONS_KEY;
use crate::harness::compaction::compaction_task;
use crate::harness::generation::generation_task;
use crate::harness::tool::tool_task;
use crate::harness::types::{
	AnyTask, Extension, PromptSection, Registry, RegistryListener, RegistryReader, RegistrySnapshot,
	ToolRegistration, Unsubscribe,
};
const SECTION_KEY_PATTERN: fn(&str) -> bool = |key: &str| {
	let mut chars = key.chars();
	match chars.next() {
		Some(first) if first.is_ascii_lowercase() => {
			first.is_ascii_lowercase()
				&& chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
		}
		_ => false,
	}
};

/// Built-in task definitions every registry holds; they are not an extension
/// and cannot be removed or replaced.
pub fn builtin_tasks() -> Vec<AnyTask> {
	vec![generation_task(), tool_task(), compaction_task()]
}

/// Immutable published registry state.
pub struct RegistryState {
	extensions: Vec<Arc<Extension>>,
	by_name: BTreeMap<String, Arc<Extension>>,
	tasks: IndexMap<String, AnyTask>,
}

impl RegistryState {
	fn new(extensions: Vec<Arc<Extension>>) -> Result<Self, String> {
		let mut by_name = BTreeMap::new();
		for extension in &extensions {
			by_name.insert(extension.name.clone(), extension.clone());
		}
		let mut tasks = IndexMap::new();
		for task in builtin_tasks() {
			tasks.insert(task.definition.name.clone(), task);
		}
		for extension in &extensions {
			for task in &extension.tasks {
				let name = task.definition.name.clone();
				if tasks.contains_key(&name) {
					return Err(format!(
						"Task {name} of extension {} is already installed",
						extension.name
					));
				}
				tasks.insert(name, task.clone());
			}
		}
		Ok(Self { extensions, by_name, tasks })
	}
}

impl RegistrySnapshot for RegistryState {
	fn installed(&self) -> Vec<Arc<Extension>> {
		self.extensions.clone()
	}

	fn extension(&self, name: &str) -> Option<Arc<Extension>> {
		self.by_name.get(name).cloned()
	}

	fn tools(&self) -> Vec<(Arc<Extension>, Arc<ToolRegistration>)> {
		self.extensions
			.iter()
			.flat_map(|extension| extension.tools.iter().map(move |tool| (extension.clone(), tool.clone())))
			.collect()
	}

	fn sections(&self) -> Vec<(Arc<Extension>, PromptSection)> {
		self.extensions
			.iter()
			.flat_map(|extension| extension.sections.iter().map(move |section| (extension.clone(), section.clone())))
			.collect()
	}

	fn tasks(&self) -> Vec<AnyTask> {
		self.tasks.values().cloned().collect()
	}

	fn task(&self, name: &str) -> Option<AnyTask> {
		self.tasks.get(name).cloned()
	}
}

/// Application-owned registry of extensions.
pub struct RegistryImpl {
	current: parking_lot::Mutex<Arc<RegistryState>>,
	listeners: Arc<parking_lot::Mutex<Vec<(u64, RegistryListener)>>>,
	next_listener: std::sync::atomic::AtomicU64,
}

impl RegistryReader for RegistryImpl {
	fn snapshot(&self) -> Arc<dyn RegistrySnapshot> {
		self.current.lock().clone()
	}

	fn subscribe(&self, listener: RegistryListener) -> Unsubscribe {
		let id = self
			.next_listener
			.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
		self.listeners.lock().push((id, listener));
		let listeners = self.listeners.clone();
		Box::new(move || {
			listeners.lock().retain(|(registered, _)| *registered != id);
		})
	}
}

impl Registry for RegistryImpl {
	fn install(&self, extension: Extension) {
		if let Err(error) = validate_extension(&extension) {
			panic!("{error}");
		}
		let extension = Arc::new(extension);
		let current = self.current.lock().installed();
		let next: Vec<Arc<Extension>> = match current.iter().position(|installed| installed.name == extension.name) {
			None => {
				let mut next = current;
				next.push(extension);
				next
			}
			Some(index) => current
				.into_iter()
				.enumerate()
				.map(|(at, installed)| if at == index { extension.clone() } else { installed })
				.collect(),
		};
		self.publish(next);
	}

	fn uninstall(&self, extension: &Extension) {
		let current = self.current.lock().installed();
		if !current.iter().any(|installed| installed.name == extension.name) {
			return;
		}
		let next: Vec<Arc<Extension>> = current
			.into_iter()
			.filter(|installed| installed.name != extension.name)
			.collect();
		self.publish(next);
	}
}

impl RegistryImpl {
	pub fn new() -> Self {
		Self {
			current: parking_lot::Mutex::new(Arc::new(
				RegistryState::new(Vec::new()).expect("empty registry state"),
			)),
			listeners: Arc::new(parking_lot::Mutex::new(Vec::new())),
			next_listener: std::sync::atomic::AtomicU64::new(0),
		}
	}

	/// Build and validate the next state, which fails on a task name
	/// collision, then publish it synchronously.
	fn publish(&self, extensions: Vec<Arc<Extension>>) {
		let next = match RegistryState::new(extensions) {
			Ok(next) => next,
			Err(error) => {
				// Upstream throws synchronously from install(); the erased
				// surface already validated in install, so a collision here
				// panics with the same message.
				panic!("{error}");
			}
		};
		*self.current.lock() = Arc::new(next);
		for (_, listener) in self.listeners.lock().clone() {
			listener();
		}
	}
}

/// Unique tool names and section keys within one extension; valid,
/// unreserved section keys.
fn validate_extension(extension: &Extension) -> Result<(), String> {
	let mut tools = BTreeSet::new();
	for tool in &extension.tools {
		if !tools.insert(tool.tool.name.clone()) {
			return Err(format!(
				"Extension {} has two tools named {}",
				extension.name, tool.tool.name
			));
		}
	}
	let mut sections = BTreeSet::new();
	for section in &extension.sections {
		if !SECTION_KEY_PATTERN(&section.key) {
			return Err(format!(
				"Section key {} must match /^[a-z][a-z0-9_-]*$/",
				serde_json::to_string(&section.key).unwrap_or_default()
			));
		}
		if section.key == INSTRUCTIONS_KEY {
			return Err(format!(
				"Section key {} is reserved for the agent's instructions",
				section.key
			));
		}
		if !sections.insert(section.key.clone()) {
			return Err(format!(
				"Extension {} has two sections with key {}",
				extension.name, section.key
			));
		}
	}
	Ok(())
}

/// Create an application-owned registry holding only the built-in tasks.
pub fn create_registry() -> RegistryImpl {
	RegistryImpl::new()
}
