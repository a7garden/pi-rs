//! Port of `packages/coding-agent/src/core/system-prompt.ts` — system prompt
//! construction from ordered, independently replaceable sections.
//!
//! Deviations:
//! - **D59 — custom section map ordering is sorted, not insertion order.**
//!   Upstream `sections` is a JS object (insertion order); here it is a
//!   [`BTreeMap`] (D30), so user-contributed sections render sorted by name.
//!   Built-in section order (`preamble`, `addendum`, `project_context`,
//!   `skills`, `cwd`) is preserved.
//!
//! Upstream throws on invalid section names; the `*_sections` entry point
//! returns `Err` with the same message and the convenience wrappers panic.

use std::collections::{BTreeMap, BTreeSet};

use indexmap::IndexMap;

use crate::core::extensions::types::{
    BuildSystemPromptOptions, ContextFileEntry, NormalizedBuildSystemPromptOptions,
};
use crate::core::skills::{format_skills_for_prompt, FileReadTool};
use pi_ai::types::{SystemContent, SystemMessage};
use pi_ai::utils::text::get_system_message_text;

/// Ordered system prompt sections, keyed by name. `preamble` is untagged text;
/// every other section is wrapped in a tag of the same name so the model can
/// match later updates to it. These become `SystemMessage.sections` in the
/// transcript (upstream `SystemPromptSections`).
pub type SystemPromptSections = IndexMap<String, String>;

/// The complete prompt state for an input (upstream `buildSystemPromptState`
/// return shape). A forced prompt is opaque and lives in `content` with no
/// sections; otherwise `content` is empty and the structured sections carry
/// the prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemPromptState {
    pub content: String,
    pub sections: Option<SystemPromptSections>,
}

/// Normalize prompt input into the mutable, collection-complete shape exposed
/// to extensions (upstream `normalizeBuildSystemPromptOptions`).
pub fn normalize_build_system_prompt_options(
    input: &BuildSystemPromptOptions,
) -> NormalizedBuildSystemPromptOptions {
    NormalizedBuildSystemPromptOptions::from(input.clone())
}

fn is_valid_section_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.is_ascii_lowercase() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

fn render_project_context(context_files: &[ContextFileEntry]) -> String {
    let mut parts = vec!["Project-specific instructions and guidelines:".to_string()];
    for file in context_files {
        parts.push(format!(
            "<project_instructions path=\"{}\">\n{}\n</project_instructions>",
            file.path, file.content
        ));
    }
    parts.join("\n\n")
}

fn build_rules(
    selected_tools: &[String],
    tool_guidelines: &BTreeMap<String, Vec<String>>,
    prompt_guidelines: &[String],
) -> String {
    let mut rules: Vec<String> = Vec::new();
    let mut seen = BTreeSet::new();
    let mut add_rule = |rule: &str| {
        let normalized = rule.trim();
        if normalized.is_empty() || !seen.insert(normalized.to_string()) {
            return;
        }
        rules.push(normalized.to_string());
    };

    let has_bash = selected_tools.iter().any(|t| t == "bash");
    let has_power_shell = selected_tools.iter().any(|t| t == "powershell");
    let has_grep = selected_tools.iter().any(|t| t == "grep");
    let has_find = selected_tools.iter().any(|t| t == "find");
    let has_ls = selected_tools.iter().any(|t| t == "ls");

    if (has_bash || has_power_shell) && !has_grep && !has_find && !has_ls {
        if has_bash && has_power_shell {
            add_rule("Use bash or PowerShell for file operations like listing, searching, and finding files");
        } else if has_power_shell {
            add_rule(
                "Use PowerShell for file operations like listing, searching, and finding files",
            );
        } else {
            add_rule("Use bash for file operations like ls, rg, find");
        }
    }

    for name in selected_tools {
        for rule in tool_guidelines.get(name).into_iter().flatten() {
            add_rule(rule);
        }
    }
    for rule in prompt_guidelines {
        add_rule(rule);
    }
    add_rule("Be concise in your responses");
    add_rule("Show file paths clearly when working with files");
    rules
        .iter()
        .map(|rule| format!("- {rule}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build the ordered, independently replaceable sections of the structured
/// system prompt (upstream `buildSystemPromptSections`).
pub fn build_system_prompt_sections(
    input: &BuildSystemPromptOptions,
) -> Result<SystemPromptSections, String> {
    let options = normalize_build_system_prompt_options(input);
    build_system_prompt_sections_normalized(&options)
}

fn build_system_prompt_sections_normalized(
    options: &NormalizedBuildSystemPromptOptions,
) -> Result<SystemPromptSections, String> {
    for name in options.sections.keys() {
        if !is_valid_section_name(name) || name == "preamble" {
            return Err(format!("Invalid system prompt section name: {name}"));
        }
    }

    fn set_section(sections: &mut Vec<(String, String)>, name: &str, content: String) {
        if let Some(entry) = sections.iter_mut().find(|(existing, _)| existing == name) {
            entry.1 = content;
        } else {
            sections.push((name.to_string(), content));
        }
    }

    let mut prompt_sections: Vec<(String, String)> = Vec::new();
    if let Some(custom_prompt) = &options.custom_prompt {
        if !custom_prompt.is_empty() {
            set_section(&mut prompt_sections, "preamble", custom_prompt.clone());
        }
    }
    if !prompt_sections.iter().any(|(name, _)| name == "preamble") {
        set_section(&mut prompt_sections, "preamble",
			"You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.".to_string(),
		);
        let declared_tools: Vec<&String> = options
            .selected_tools
            .iter()
            .filter(|name| !options.hidden_tools.contains(*name))
            .collect();
        let visible_tools: Vec<&String> = declared_tools
            .iter()
            .copied()
            .filter(|name| options.tool_snippets.contains_key(*name))
            .collect();
        let tools = if visible_tools.is_empty() {
            "(none)".to_string()
        } else {
            visible_tools
                .iter()
                .map(|name| format!("- {name}: {}", options.tool_snippets[*name]))
                .collect::<Vec<_>>()
                .join("\n")
        };
        set_section(&mut prompt_sections, "tools",
			format!(
				"{tools}\n\nIn addition to the tools above, you may have access to other custom tools depending on the project."
			),
		);
        let declared_tool_names: Vec<String> =
            declared_tools.iter().map(|name| (*name).clone()).collect();
        set_section(
            &mut prompt_sections,
            "rules",
            build_rules(
                &declared_tool_names,
                &options.tool_guidelines,
                &options.prompt_guidelines,
            ),
        );
        set_section(&mut prompt_sections, "docs",
			format!(
				"Pi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):\n\
				 - Main documentation: {readme}\n\
				 - Additional docs: {docs}\n\
				 - Examples: {examples} (extensions, custom tools, SDK)\n\
				 - When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory\n\
				 - When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md), environment variables (docs/environment-variables.md), MCP servers (docs/mcp.md), codemode scripts and non-LLM models such as classifiers and image models (docs/codemode.md)\n\
				 - When working on pi topics, read the docs and examples, and follow .md cross-references before implementing\n\
				 - Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)",
				readme = crate::config::get_readme_path(),
				docs = crate::config::get_docs_path(),
				examples = crate::config::get_examples_path(),
			),
		);
    }

    if !options.append_system_prompt.is_empty() {
        set_section(
            &mut prompt_sections,
            "addendum",
            options.append_system_prompt.clone(),
        );
    }
    let context_files = options.context_files.clone().unwrap_or_default();
    if !context_files.is_empty() {
        set_section(
            &mut prompt_sections,
            "project_context",
            render_project_context(&context_files),
        );
    }
    let skills = options.skills.clone().unwrap_or_default();
    // A hidden reader is still reachable through another tool, so skills stay but the hint names no tool.
    let readers = ["read", "bash"];
    let is_declared = |name: &str| {
        options.selected_tools.iter().any(|tool| tool == name)
            && !options.hidden_tools.iter().any(|tool| tool == name)
    };
    let skill_file_read_tool = readers
        .iter()
        .find(|tool| is_declared(tool))
        .map(|tool| match *tool {
            "read" => FileReadTool::Read,
            _ => FileReadTool::Bash,
        })
        .or_else(|| {
            readers
                .iter()
                .any(|tool| options.selected_tools.iter().any(|name| name == tool))
                .then_some(FileReadTool::Indirect)
        });
    if let Some(file_read_tool) = skill_file_read_tool {
        if !skills.is_empty() {
            let skills_prompt = format_skills_for_prompt(&skills, file_read_tool)
                .trim()
                .to_string();
            if !skills_prompt.is_empty() {
                set_section(&mut prompt_sections, "skills", skills_prompt);
            }
        }
    }
    set_section(&mut prompt_sections, "cwd", options.cwd.replace('\\', "/"));
    for (name, content) in &options.sections {
        if !content.is_empty() {
            set_section(&mut prompt_sections, name, content.clone());
        }
    }

    let mut sections: SystemPromptSections = IndexMap::new();
    sections.insert("preamble".to_string(), prompt_sections[0].1.clone());
    for (name, content) in &prompt_sections {
        if name == "preamble" {
            continue;
        }
        sections.insert(name.clone(), format!("<{name}>\n{content}\n</{name}>"));
    }
    Ok(sections)
}

/// The complete prompt state for `input` (upstream `buildSystemPromptState`).
pub fn build_system_prompt_state(input: &BuildSystemPromptOptions) -> SystemPromptState {
    if let Some(forced) = &input.force_system_prompt {
        return SystemPromptState {
            content: forced.clone(),
            sections: None,
        };
    }
    let sections = build_system_prompt_sections(input).unwrap_or_else(|error| panic!("{error}"));
    SystemPromptState {
        content: String::new(),
        sections: Some(sections),
    }
}

/// Build the system prompt text, rendered exactly as the transcript's system
/// message replays it (upstream `buildSystemPrompt`).
pub fn build_system_prompt(input: &BuildSystemPromptOptions) -> String {
    let state = build_system_prompt_state(input);
    let sections = state.sections.map(|sections| {
        sections
            .into_iter()
            .map(|(name, text)| (name, Some(text)))
            .collect::<IndexMap<_, _>>()
    });
    get_system_message_text(&SystemMessage {
        content: SystemContent::Text(state.content),
        sections,
        tools_added: None,
        tools_removed: None,
        timestamp: 0,
    })
}

/// Render the prompt from already-normalized options (the shape the runner
/// seam, D44, hands to the renderer).
pub fn build_system_prompt_normalized(options: &NormalizedBuildSystemPromptOptions) -> String {
    let input = BuildSystemPromptOptions {
        custom_prompt: options.custom_prompt.clone(),
        force_system_prompt: options.force_system_prompt.clone(),
        selected_tools: Some(options.selected_tools.clone()),
        hidden_tools: Some(options.hidden_tools.clone()),
        tool_snippets: Some(options.tool_snippets.clone()),
        tool_guidelines: Some(options.tool_guidelines.clone()),
        prompt_guidelines: Some(options.prompt_guidelines.clone()),
        append_system_prompt: Some(options.append_system_prompt.clone()),
        sections: Some(options.sections.clone()),
        cwd: options.cwd.clone(),
        context_files: options.context_files.clone(),
        skills: options.skills.clone(),
    };
    build_system_prompt(&input)
}

/// Diff the sections the model currently has (replayed from the transcript, so
/// never null) against the desired ones. Returns a `SystemMessage.sections`
/// patch, or `None` when nothing changed (upstream `diffSystemPromptSections`).
pub fn diff_system_prompt_sections(
    previous: &IndexMap<String, Option<String>>,
    current: &SystemPromptSections,
) -> Option<IndexMap<String, Option<String>>> {
    let mut patch: IndexMap<String, Option<String>> = IndexMap::new();
    for (name, text) in current {
        if previous.get(name).map(|value| value.as_deref()) != Some(Some(text.as_str())) {
            patch.insert(name.clone(), Some(text.clone()));
        }
    }
    for name in previous.keys() {
        if !current.contains_key(name) {
            patch.insert(name.clone(), None);
        }
    }
    if patch.is_empty() {
        None
    } else {
        Some(patch)
    }
}

#[cfg(test)]
#[path = "system_prompt_tests.rs"]
mod system_prompt_tests;
