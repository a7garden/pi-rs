//! Port of `upstream/pi/packages/coding-agent/test/system-prompt.test.ts`
//! (1:1), plus the two pure-function cases from
//! `test/system-prompt-updates.test.ts` ("diffs sections into a patch",
//! "keeps the preamble untagged and replaces it like any section"). The
//! remaining updates cases build a full AgentSession and port with that
//! module.

use indexmap::IndexMap;

use super::{build_system_prompt, build_system_prompt_sections, build_system_prompt_state, diff_system_prompt_sections, SystemPromptSections};
use crate::core::extensions::types::BuildSystemPromptOptions;
use crate::core::skills::Skill;
use crate::core::source_info::create_synthetic_source_info;

fn test_skill() -> Skill {
	Skill {
		name: "test-skill".to_string(),
		description: "A test skill.".to_string(),
		file_path: "/skills/test-skill/SKILL.md".to_string(),
		base_dir: "/skills/test-skill".to_string(),
		source_info: create_synthetic_source_info("/skills/test-skill/SKILL.md", "test", None, None, None),
		disable_model_invocation: false,
	}
}

fn options(cwd: &str) -> BuildSystemPromptOptions {
	BuildSystemPromptOptions { cwd: cwd.to_string(), ..Default::default() }
}

fn to_optional_map(sections: &SystemPromptSections) -> IndexMap<String, Option<String>> {
	sections
		.iter()
		.map(|(name, text)| (name.clone(), Some(text.clone())))
		.collect()
}

// -- empty tools -----------------------------------------------------------

#[test]
fn shows_none_for_empty_tools_list() {
	let prompt = build_system_prompt(&BuildSystemPromptOptions {
		selected_tools: Some(vec![]),
		context_files: Some(vec![]),
		skills: Some(vec![]),
		..options(std::env::current_dir().unwrap().to_str().unwrap())
	});

	assert!(prompt.contains("<tools>\n(none)\n"));
}

#[test]
fn shows_file_paths_guideline_even_with_no_tools() {
	let prompt = build_system_prompt(&BuildSystemPromptOptions {
		selected_tools: Some(vec![]),
		context_files: Some(vec![]),
		skills: Some(vec![]),
		..options(std::env::current_dir().unwrap().to_str().unwrap())
	});

	assert!(prompt.contains("Show file paths clearly"));
}

// -- prompt structure ------------------------------------------------------

#[test]
fn keeps_the_default_and_custom_prompt_prefixes_exact() {
	let default_prompt = build_system_prompt(&options("/tmp"));
	let custom = build_system_prompt(&BuildSystemPromptOptions {
		custom_prompt: Some("You are Exact.".to_string()),
		..options("/tmp")
	});

	assert!(default_prompt.starts_with("You are an expert coding assistant operating inside pi"));
	assert!(custom.starts_with("You are Exact.\n\n<cwd>"));
}

#[test]
fn preserves_an_exact_forced_prompt_without_sections() {
	assert_eq!(build_system_prompt(&BuildSystemPromptOptions { force_system_prompt: Some("exact".to_string()), ..options("/tmp") }), "exact");
}

#[test]
fn maps_appended_instructions_and_project_context_to_stable_sections() {
	let prompt = build_system_prompt(&BuildSystemPromptOptions {
		custom_prompt: Some("You are Exact.".to_string()),
		append_system_prompt: Some("Additional instructions.".to_string()),
		context_files: Some(vec![crate::core::extensions::types::ContextFileEntry {
			path: "/tmp/AGENTS.md".to_string(),
			content: "Project instructions.".to_string(),
		}]),
		selected_tools: Some(vec![]),
		skills: Some(vec![]),
		..options("/tmp")
	});

	assert!(prompt.contains("<addendum>\nAdditional instructions.\n</addendum>"));
	assert!(prompt.contains(
		"<project_context>\nProject-specific instructions and guidelines:\n\n<project_instructions path=\"/tmp/AGENTS.md\">"
	));
	assert!(prompt.contains("<cwd>\n/tmp\n</cwd>"));
}

// -- default tools ---------------------------------------------------------

#[test]
fn includes_all_default_tools_when_snippets_are_provided() {
	let prompt = build_system_prompt(&BuildSystemPromptOptions {
		tool_snippets: Some(
			[
				("read", "Read file contents"),
				("bash", "Execute bash commands"),
				("edit", "Make surgical edits"),
				("write", "Create or overwrite files"),
			]
			.iter()
			.map(|(k, v)| (k.to_string(), v.to_string()))
			.collect(),
		),
		context_files: Some(vec![]),
		skills: Some(vec![]),
		..options(std::env::current_dir().unwrap().to_str().unwrap())
	});

	assert!(prompt.contains("- read:"));
	assert!(prompt.contains("- bash:"));
	assert!(prompt.contains("- edit:"));
	assert!(prompt.contains("- write:"));
}

#[test]
fn uses_shell_specific_guidance_for_powershell() {
	let prompt = build_system_prompt(&BuildSystemPromptOptions {
		selected_tools: Some(vec!["powershell".to_string()]),
		context_files: Some(vec![]),
		skills: Some(vec![]),
		..options(std::env::current_dir().unwrap().to_str().unwrap())
	});

	assert!(prompt.contains("Use PowerShell for file operations"));
}

#[test]
fn uses_shell_specific_guidance_for_bash_and_powershell() {
	let prompt = build_system_prompt(&BuildSystemPromptOptions {
		selected_tools: Some(vec!["bash".to_string(), "powershell".to_string()]),
		context_files: Some(vec![]),
		skills: Some(vec![]),
		..options(std::env::current_dir().unwrap().to_str().unwrap())
	});

	assert!(prompt.contains("Use bash or PowerShell for file operations"));
}

#[test]
fn instructs_models_to_resolve_pi_docs_and_examples_under_absolute_base_paths() {
	let prompt = build_system_prompt(&options(std::env::current_dir().unwrap().to_str().unwrap()));

	assert!(prompt.contains(
		"- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory"
	));
	assert!(prompt.contains("environment variables (docs/environment-variables.md), MCP servers (docs/mcp.md)"));
}

// -- custom tool snippets --------------------------------------------------

#[test]
fn includes_custom_tools_in_available_tools_section_when_prompt_snippet_is_provided() {
	let prompt = build_system_prompt(&BuildSystemPromptOptions {
		selected_tools: Some(vec!["read".to_string(), "dynamic_tool".to_string()]),
		tool_snippets: Some([("dynamic_tool", "Run dynamic test behavior")].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()),
		context_files: Some(vec![]),
		skills: Some(vec![]),
		..options(std::env::current_dir().unwrap().to_str().unwrap())
	});

	assert!(prompt.contains("- dynamic_tool: Run dynamic test behavior"));
}

#[test]
fn omits_custom_tools_from_available_tools_section_when_prompt_snippet_is_not_provided() {
	let prompt = build_system_prompt(&BuildSystemPromptOptions {
		selected_tools: Some(vec!["read".to_string(), "dynamic_tool".to_string()]),
		context_files: Some(vec![]),
		skills: Some(vec![]),
		..options(std::env::current_dir().unwrap().to_str().unwrap())
	});

	assert!(!prompt.contains("dynamic_tool"));
}

// -- prompt guidelines -----------------------------------------------------

#[test]
fn appends_prompt_guidelines_to_default_guidelines() {
	let prompt = build_system_prompt(&BuildSystemPromptOptions {
		selected_tools: Some(vec!["read".to_string(), "dynamic_tool".to_string()]),
		prompt_guidelines: Some(vec!["Use dynamic_tool for project summaries.".to_string()]),
		context_files: Some(vec![]),
		skills: Some(vec![]),
		..options(std::env::current_dir().unwrap().to_str().unwrap())
	});

	assert!(prompt.contains("- Use dynamic_tool for project summaries."));
}

#[test]
fn deduplicates_and_trims_prompt_guidelines() {
	let prompt = build_system_prompt(&BuildSystemPromptOptions {
		selected_tools: Some(vec!["read".to_string(), "dynamic_tool".to_string()]),
		prompt_guidelines: Some(vec![
			"Use dynamic_tool for summaries.".to_string(),
			"  Use dynamic_tool for summaries.  ".to_string(),
			"   ".to_string(),
		]),
		context_files: Some(vec![]),
		skills: Some(vec![]),
		..options(std::env::current_dir().unwrap().to_str().unwrap())
	});

	assert_eq!(prompt.matches("- Use dynamic_tool for summaries.").count(), 1);
}

// -- skills ----------------------------------------------------------------

#[test]
fn includes_skills_with_only_bash_in_the_default_prompt() {
	let prompt = build_system_prompt(&BuildSystemPromptOptions {
		selected_tools: Some(vec!["bash".to_string()]),
		context_files: Some(vec![]),
		skills: Some(vec![test_skill()]),
		..options(std::env::current_dir().unwrap().to_str().unwrap())
	});

	assert!(prompt.contains("<skills>"));
	assert!(prompt.contains("<available_skills>"));
	assert!(prompt.contains("<name>test-skill</name>"));
	assert!(prompt.contains("Use bash to load a skill's file"));
}

#[test]
fn includes_skills_with_only_bash_in_the_custom_prompt() {
	let prompt = build_system_prompt(&BuildSystemPromptOptions {
		custom_prompt: Some("Custom system prompt".to_string()),
		selected_tools: Some(vec!["bash".to_string()]),
		context_files: Some(vec![]),
		skills: Some(vec![test_skill()]),
		..options(std::env::current_dir().unwrap().to_str().unwrap())
	});

	assert!(prompt.contains("<skills>"));
	assert!(prompt.contains("<available_skills>"));
	assert!(prompt.contains("<name>test-skill</name>"));
	assert!(prompt.contains("Use bash to load a skill's file"));
}

#[test]
fn omits_skills_without_read_or_bash() {
	let prompt = build_system_prompt(&BuildSystemPromptOptions {
		selected_tools: Some(vec!["write".to_string()]),
		context_files: Some(vec![]),
		skills: Some(vec![test_skill()]),
		..options(std::env::current_dir().unwrap().to_str().unwrap())
	});

	assert!(!prompt.contains("<available_skills>"));
}

// -- system-prompt-updates.test.ts (pure-function cases) --------------------

#[test]
fn diffs_sections_into_a_patch() {
	let previous = build_system_prompt_sections(&BuildSystemPromptOptions {
		sections: Some([("plan_mode", "Plan only.")].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()),
		..options("/tmp")
	})
	.unwrap();
	let current = build_system_prompt_sections(&BuildSystemPromptOptions {
		sections: Some([("plan_mode", "Implementation allowed.")].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()),
		..options("/tmp")
	})
	.unwrap();
	assert_eq!(
		diff_system_prompt_sections(&to_optional_map(&previous), &current).unwrap(),
		[("plan_mode".to_string(), Some("<plan_mode>\nImplementation allowed.\n</plan_mode>".to_string()))]
			.into_iter()
			.collect::<IndexMap<_, _>>()
	);
	assert!(diff_system_prompt_sections(&to_optional_map(&previous), &previous).is_none());
	assert_eq!(
		diff_system_prompt_sections(&to_optional_map(&previous), &build_system_prompt_sections(&options("/tmp")).unwrap()).unwrap(),
		[("plan_mode".to_string(), None)].into_iter().collect::<IndexMap<_, _>>()
	);
}

#[test]
fn keeps_the_preamble_untagged_and_replaces_it_like_any_section() {
	let previous = build_system_prompt_sections(&BuildSystemPromptOptions {
		custom_prompt: Some("You are A.".to_string()),
		..options("/tmp")
	})
	.unwrap();
	let current = build_system_prompt_sections(&BuildSystemPromptOptions {
		custom_prompt: Some("You are B.".to_string()),
		..options("/tmp")
	})
	.unwrap();
	assert_eq!(previous["preamble"], "You are A.");
	assert_eq!(
		diff_system_prompt_sections(&to_optional_map(&previous), &current).unwrap(),
		[("preamble".to_string(), Some("You are B.".to_string()))].into_iter().collect::<IndexMap<_, _>>()
	);

	assert_eq!(
		build_system_prompt_state(&BuildSystemPromptOptions { force_system_prompt: Some("Exact prompt.".to_string()), ..options("/tmp") }),
		super::SystemPromptState { content: "Exact prompt.".to_string(), sections: None }
	);
	assert_eq!(
		build_system_prompt_state(&options("/tmp")),
		super::SystemPromptState { content: String::new(), sections: Some(build_system_prompt_sections(&options("/tmp")).unwrap()) }
	);
	let error = build_system_prompt_sections(&BuildSystemPromptOptions {
		sections: Some([("preamble", "x")].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()),
		..options("/tmp")
	})
	.unwrap_err();
	assert!(error.contains("Invalid system prompt section name"), "{error}");
}

