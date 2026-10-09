//! Port of `experimental/durable/prompt.ts` — pi's system prompt as one
//! extension: the sections of `buildSystemPromptSections()` for the request's
//! tools and the conversation's directory.
//!
//! Deviations:
//! - **D171 — the per-request `WeakMap<PromptInput, sections>` build cache
//!   collapses to per-render builds.** `PromptInput` has no weakly
//!   referenceable identity here; each section render rebuilds the section
//!   map from the cached per-directory resources, which is the expensive
//!   part (context files and skills load once per directory, like upstream).
//! - `loadProjectContextFiles({cwd, agentDir})` maps to
//!   [`crate::core::resource_loader::load_project_context_files`]'s
//!   positional form; the tool prompt contributions read the crate's
//!   `*_TOOL_SYSTEM_PROMPT_SNIPPET`/`*_GUIDELINES` constants.

use std::collections::HashMap;
use std::sync::Arc;

use pi_chord::context::Context;
use pi_durable::harness::types::{Extension, PromptInput, SectionRender};
use pi_durable::tasks::BoxFuture;

use crate::config::get_agent_dir;
use crate::core::extensions::types::{BuildSystemPromptOptions, ContextFileEntry};
use crate::core::resource_loader::load_project_context_files;
use crate::core::settings_manager::SettingsManager;
use crate::core::skills::{load_skills, Skill};
use crate::core::system_prompt::{build_system_prompt_sections, SystemPromptSections};
use crate::core::tools::bash::{BASH_TOOL_SYSTEM_PROMPT_GUIDELINES, BASH_TOOL_SYSTEM_PROMPT_SNIPPET};
use crate::core::tools::edit::{EDIT_TOOL_SYSTEM_PROMPT_GUIDELINES, EDIT_TOOL_SYSTEM_PROMPT_SNIPPET};
use crate::core::tools::read::{READ_TOOL_SYSTEM_PROMPT_GUIDELINES, READ_TOOL_SYSTEM_PROMPT_SNIPPET};
use crate::core::tools::write::{WRITE_TOOL_SYSTEM_PROMPT_GUIDELINES, WRITE_TOOL_SYSTEM_PROMPT_SNIPPET};

type SharedSettings = Arc<parking_lot::Mutex<SettingsManager>>;

/// pi's section order; `buildSystemPromptSections()` omits the ones without
/// content (upstream `KEYS`).
const KEYS: [&str; 7] = ["preamble", "tools", "rules", "docs", "project_context", "skills", "cwd"];

fn contribution(name: &str) -> Option<(&'static str, Vec<String>)> {
	fn guidelines(lines: &[&str]) -> Vec<String> {
		lines.iter().map(|line| line.to_string()).collect()
	}
	match name {
		"read" => Some((READ_TOOL_SYSTEM_PROMPT_SNIPPET, guidelines(&READ_TOOL_SYSTEM_PROMPT_GUIDELINES))),
		"bash" => Some((BASH_TOOL_SYSTEM_PROMPT_SNIPPET, guidelines(BASH_TOOL_SYSTEM_PROMPT_GUIDELINES))),
		"edit" => Some((EDIT_TOOL_SYSTEM_PROMPT_SNIPPET, guidelines(&EDIT_TOOL_SYSTEM_PROMPT_GUIDELINES))),
		"write" => Some((WRITE_TOOL_SYSTEM_PROMPT_SNIPPET, guidelines(&WRITE_TOOL_SYSTEM_PROMPT_GUIDELINES))),
		_ => None,
	}
}

/// Context files and skills per directory, loaded once (upstream `resources`).
struct Resources {
	context_files: Vec<ContextFileEntry>,
	skills: Vec<Skill>,
}

/// pi's system prompt as one extension (upstream `createPiPrompt`).
pub fn create_pi_prompt(settings: SharedSettings, fallback_cwd: String) -> Extension {
	let resources: Arc<parking_lot::Mutex<HashMap<String, Arc<Resources>>>> = Arc::default();
	let load: Arc<dyn Fn(&str) -> Arc<Resources> + Send + Sync> = {
		let resources = Arc::clone(&resources);
		let settings = Arc::clone(&settings);
		Arc::new(move |cwd: &str| {
			if let Some(found) = resources.lock().get(cwd) {
				return Arc::clone(found);
			}
			let agent_dir = get_agent_dir();
			let skill_paths = settings.lock().get_skill_paths();
			let found = Arc::new(Resources {
				context_files: load_project_context_files(cwd, &agent_dir)
					.into_iter()
					.map(|file| ContextFileEntry { path: file.path, content: file.content })
					.collect(),
				skills: load_skills(crate::core::skills::LoadSkillsOptions {
					cwd,
					agent_dir: Some(agent_dir.as_str()),
					skill_paths: skill_paths.as_slice(),
					include_defaults: true,
				})
				.skills,
			});
			resources.lock().insert(cwd.to_string(), Arc::clone(&found));
			found
		})
	};
	// Build one request's sections from the request's tools and directory.
	let build: Arc<dyn Fn(&PromptInput) -> SystemPromptSections + Send + Sync> = Arc::new(move |input: &PromptInput| {
			let cwd = input
				.env
				.as_ref()
				.map(|env| env.cwd().to_string())
				.or_else(|| input.agent.cwd.clone())
				.unwrap_or_else(|| fallback_cwd.clone());
			let selected_tools: Vec<String> = input.agent.tools.iter().map(|tool| tool.tool.name.clone()).collect();
			let mut snippets = std::collections::BTreeMap::new();
			let mut guidelines: std::collections::BTreeMap<String, Vec<String>> = std::collections::BTreeMap::new();
			for name in &selected_tools {
				let Some((snippet, tool_guidelines)) = contribution(name) else {
					continue;
				};
				snippets.insert(name.clone(), snippet.to_string());
				guidelines.insert(name.clone(), tool_guidelines);
			}
			let found = load(&cwd);
			build_system_prompt_sections(&BuildSystemPromptOptions {
				cwd,
				selected_tools: Some(selected_tools),
				tool_snippets: Some(snippets),
				tool_guidelines: Some(guidelines),
				context_files: Some(found.context_files.clone()),
				skills: Some(found.skills.clone()),
				..Default::default()
			})
			.unwrap_or_default()
		});
	let sections = KEYS
		.iter()
		.map(|key| {
			let build = Arc::clone(&build);
			let section_key = key.to_string();
			let render: SectionRender = Arc::new(move |input: PromptInput, _context: Context| {
				let built = build(&input);
				let key = section_key.clone();
				Box::pin(async move { Ok(built.get(&key).cloned()) }) as BoxFuture<Result<Option<String>, String>>
			});
			// The built sections carry their own tags.
			pi_durable::harness::define::section(*key, render, Some(false))
		})
		.collect();
	Extension::new("pi-prompt").with_sections(sections)
}
