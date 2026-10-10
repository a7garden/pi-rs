//! Port of `packages/coding-agent/test/skills.test.ts`.
//!
//! Upstream fixtures are copied 1:1 to `src/core/test_fixtures/`.

use std::collections::HashMap;

use super::{
    format_skills_for_prompt, load_skills, load_skills_from_dir, FileReadTool,
    LoadSkillsFromDirOptions, LoadSkillsOptions, Skill,
};
use crate::core::diagnostics::ResourceDiagnostic;
use crate::core::source_info::{create_synthetic_source_info, SourceScope};
use crate::utils::paths::{home_dir, join};

fn fixtures_dir() -> String {
    format!(
        "{}/src/core/test_fixtures/skills",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn collision_fixtures_dir() -> String {
    format!(
        "{}/src/core/test_fixtures/skills-collision",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn create_test_skill(
    name: &str,
    description: &str,
    file_path: &str,
    base_dir: &str,
    disable_model_invocation: bool,
    source: &str,
) -> Skill {
    Skill {
        name: name.to_string(),
        description: description.to_string(),
        file_path: file_path.to_string(),
        base_dir: base_dir.to_string(),
        source_info: create_synthetic_source_info(file_path, source, None, None, None),
        disable_model_invocation,
    }
}

fn find_diagnostic(diagnostics: &[ResourceDiagnostic], needle: &str) -> bool {
    diagnostics.iter().any(|d| d.message.contains(needle))
}

#[test]
fn load_dir_should_load_a_valid_skill() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "valid-skill"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].name, "valid-skill");
    assert_eq!(
        result.skills[0].description,
        "A valid skill for testing purposes."
    );
    assert_eq!(result.skills[0].source_info.source, "test");
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn load_dir_should_allow_names_that_dont_match_parent_directory() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "name-mismatch"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].name, "different-name");
    assert!(!find_diagnostic(
        &result.diagnostics,
        "does not match parent directory"
    ));
}

#[test]
fn load_dir_should_warn_when_name_contains_invalid_characters() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "invalid-name-chars"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert!(find_diagnostic(&result.diagnostics, "invalid characters"));
}

#[test]
fn load_dir_should_warn_when_name_exceeds_64_characters() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "long-name"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert!(find_diagnostic(
        &result.diagnostics,
        "exceeds 64 characters"
    ));
}

#[test]
fn load_dir_should_warn_and_skip_skill_when_description_is_missing() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "missing-description"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 0);
    assert!(find_diagnostic(
        &result.diagnostics,
        "description is required"
    ));
}

#[test]
fn load_dir_should_ignore_unknown_frontmatter_fields() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "unknown-field"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn load_dir_should_load_nested_skills_recursively() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "nested"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].name, "child-skill");
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn load_dir_should_prefer_a_directorys_root_skill_md_over_nested_skill_md_files() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "root-skill-preferred"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].name, "root-skill-preferred");
    assert_eq!(result.skills[0].description, "Root skill should win.");
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn load_dir_should_skip_files_without_frontmatter() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "no-frontmatter"),
        source: "test",
    });

    // no-frontmatter has no description, so it should be skipped
    assert_eq!(result.skills.len(), 0);
    assert!(find_diagnostic(
        &result.diagnostics,
        "description is required"
    ));
}

#[test]
fn load_dir_should_warn_and_skip_skill_when_yaml_frontmatter_is_invalid() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "invalid-yaml"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 0);
    assert!(find_diagnostic(&result.diagnostics, "at line"));
}

#[test]
fn load_dir_should_preserve_multiline_descriptions_from_yaml() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "multiline-description"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert!(result.skills[0].description.contains('\n'));
    assert!(result.skills[0]
        .description
        .contains("This is a multiline description."));
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn load_dir_should_warn_when_name_contains_consecutive_hyphens() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "consecutive-hyphens"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert!(find_diagnostic(&result.diagnostics, "consecutive hyphens"));
}

#[test]
fn load_dir_should_load_all_skills_from_fixture_directory() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &fixtures_dir(),
        source: "test",
    });

    // Should load all skills that have descriptions (even with warnings)
    // valid-skill, name-mismatch, invalid-name-chars, long-name, unknown-field,
    // nested/child-skill, consecutive-hyphens
    // NOT: missing-description, no-frontmatter (both missing descriptions)
    assert!(result.skills.len() >= 6);
}

#[test]
fn load_dir_should_return_empty_for_non_existent_directory() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: "/non/existent/path",
        source: "test",
    });

    assert_eq!(result.skills.len(), 0);
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn load_dir_should_use_parent_directory_name_when_name_not_in_frontmatter() {
    // The no-frontmatter fixture has no name in frontmatter, so it should use
    // "no-frontmatter". But it also has no description, so it won't load.
    // Let's test with a valid skill that relies on directory name
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "valid-skill"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].name, "valid-skill");
}

#[test]
fn load_dir_should_parse_disable_model_invocation_frontmatter_field() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "disable-model-invocation"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].name, "disable-model-invocation");
    assert!(result.skills[0].disable_model_invocation);
    // Should not warn about unknown field
    assert!(!find_diagnostic(
        &result.diagnostics,
        "unknown frontmatter field"
    ));
}

#[test]
fn load_dir_should_default_disable_model_invocation_to_false_when_not_specified() {
    let result = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&fixtures_dir(), "valid-skill"),
        source: "test",
    });

    assert_eq!(result.skills.len(), 1);
    assert!(!result.skills[0].disable_model_invocation);
}

// formatSkillsForPrompt

#[test]
fn prompt_should_return_empty_string_for_no_skills() {
    assert_eq!(format_skills_for_prompt(&[], FileReadTool::Read), "");
}

#[test]
fn prompt_should_format_skills_as_xml() {
    let skills = [create_test_skill(
        "test-skill",
        "A test skill.",
        "/path/to/skill/SKILL.md",
        "/path/to/skill",
        false,
        "test",
    )];

    let result = format_skills_for_prompt(&skills, FileReadTool::Read);

    assert!(result.contains("<available_skills>"));
    assert!(result.contains("</available_skills>"));
    assert!(result.contains("<skill>"));
    assert!(result.contains("<name>test-skill</name>"));
    assert!(result.contains("<description>A test skill.</description>"));
    assert!(result.contains("<location>/path/to/skill/SKILL.md</location>"));
}

#[test]
fn prompt_should_include_intro_text_before_xml() {
    let skills = [create_test_skill(
        "test-skill",
        "A test skill.",
        "/path/to/skill/SKILL.md",
        "/path/to/skill",
        false,
        "test",
    )];

    let result = format_skills_for_prompt(&skills, FileReadTool::Read);
    let xml_start = result.find("<available_skills>").unwrap();
    let intro_text = &result[..xml_start];

    assert!(intro_text.contains("The following skills provide specialized instructions"));
    assert!(intro_text.contains("Use the read tool to load a skill's file"));
}

#[test]
fn prompt_should_escape_xml_special_characters() {
    let skills = [create_test_skill(
        "test-skill",
        "A skill with <special> & \"characters\".",
        "/path/to/skill/SKILL.md",
        "/path/to/skill",
        false,
        "test",
    )];

    let result = format_skills_for_prompt(&skills, FileReadTool::Read);

    assert!(result.contains("&lt;special&gt;"));
    assert!(result.contains("&amp;"));
    assert!(result.contains("&quot;characters&quot;"));
}

#[test]
fn prompt_should_format_multiple_skills() {
    let skills = [
        create_test_skill(
            "skill-one",
            "First skill.",
            "/path/one/SKILL.md",
            "/path/one",
            false,
            "test",
        ),
        create_test_skill(
            "skill-two",
            "Second skill.",
            "/path/two/SKILL.md",
            "/path/two",
            false,
            "test",
        ),
    ];

    let result = format_skills_for_prompt(&skills, FileReadTool::Read);

    assert!(result.contains("<name>skill-one</name>"));
    assert!(result.contains("<name>skill-two</name>"));
    assert_eq!(result.matches("<skill>").count(), 2);
}

#[test]
fn prompt_should_exclude_skills_with_disable_model_invocation_from_prompt() {
    let skills = [
        create_test_skill(
            "visible-skill",
            "A visible skill.",
            "/path/visible/SKILL.md",
            "/path/visible",
            false,
            "test",
        ),
        create_test_skill(
            "hidden-skill",
            "A hidden skill.",
            "/path/hidden/SKILL.md",
            "/path/hidden",
            true,
            "test",
        ),
    ];

    let result = format_skills_for_prompt(&skills, FileReadTool::Read);

    assert!(result.contains("<name>visible-skill</name>"));
    assert!(!result.contains("<name>hidden-skill</name>"));
    assert_eq!(result.matches("<skill>").count(), 1);
}

#[test]
fn prompt_should_return_empty_string_when_all_skills_have_disable_model_invocation() {
    let skills = [create_test_skill(
        "hidden-skill",
        "A hidden skill.",
        "/path/hidden/SKILL.md",
        "/path/hidden",
        true,
        "test",
    )];

    assert_eq!(format_skills_for_prompt(&skills, FileReadTool::Read), "");
}

// loadSkills with options

#[test]
fn options_should_load_from_explicit_skill_paths() {
    let empty_agent_dir = format!(
        "{}/src/core/test_fixtures/empty-agent",
        env!("CARGO_MANIFEST_DIR")
    );
    let empty_cwd = format!(
        "{}/src/core/test_fixtures/empty-cwd",
        env!("CARGO_MANIFEST_DIR")
    );

    let result = load_skills(LoadSkillsOptions {
        agent_dir: Some(&empty_agent_dir),
        cwd: &empty_cwd,
        skill_paths: &[join(&fixtures_dir(), "valid-skill")],
        include_defaults: true,
    });
    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].source_info.scope, SourceScope::Temporary);
    assert_eq!(result.diagnostics.len(), 0);
}

#[test]
fn options_should_warn_when_skill_path_does_not_exist() {
    let empty_agent_dir = format!(
        "{}/src/core/test_fixtures/empty-agent",
        env!("CARGO_MANIFEST_DIR")
    );
    let empty_cwd = format!(
        "{}/src/core/test_fixtures/empty-cwd",
        env!("CARGO_MANIFEST_DIR")
    );

    let result = load_skills(LoadSkillsOptions {
        agent_dir: Some(&empty_agent_dir),
        cwd: &empty_cwd,
        skill_paths: &["/non/existent/path".to_string()],
        include_defaults: true,
    });
    assert_eq!(result.skills.len(), 0);
    assert!(find_diagnostic(&result.diagnostics, "does not exist"));
}

#[test]
fn options_should_expand_tilde_in_skill_paths() {
    let empty_agent_dir = format!(
        "{}/src/core/test_fixtures/empty-agent",
        env!("CARGO_MANIFEST_DIR")
    );
    let empty_cwd = format!(
        "{}/src/core/test_fixtures/empty-cwd",
        env!("CARGO_MANIFEST_DIR")
    );
    let home_skills_dir = join(&home_dir(), ".pi/agent/skills");
    let with_tilde = load_skills(LoadSkillsOptions {
        agent_dir: Some(&empty_agent_dir),
        cwd: &empty_cwd,
        skill_paths: &["~/.pi/agent/skills".to_string()],
        include_defaults: true,
    });
    let without_tilde = load_skills(LoadSkillsOptions {
        agent_dir: Some(&empty_agent_dir),
        cwd: &empty_cwd,
        skill_paths: &[home_skills_dir],
        include_defaults: true,
    });
    assert_eq!(with_tilde.skills.len(), without_tilde.skills.len());
}

// collision handling

#[test]
fn collisions_should_detect_name_collisions_and_keep_first_skill() {
    // Load from first directory
    let first = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&collision_fixtures_dir(), "first"),
        source: "first",
    });

    let second = load_skills_from_dir(LoadSkillsFromDirOptions {
        dir: &join(&collision_fixtures_dir(), "second"),
        source: "second",
    });

    // Simulate the collision behavior from loadSkills()
    let mut skill_map: HashMap<String, &Skill> = HashMap::new();
    let mut collision_warnings: Vec<(String, String)> = Vec::new();

    for skill in &first.skills {
        skill_map.insert(skill.name.clone(), skill);
    }

    for skill in &second.skills {
        match skill_map.get(&skill.name) {
            Some(existing) => collision_warnings.push((
                skill.file_path.clone(),
                format!(
                    "name collision: \"{}\" already loaded from {}",
                    skill.name, existing.file_path
                ),
            )),
            None => {
                skill_map.insert(skill.name.clone(), skill);
            }
        }
    }

    assert_eq!(skill_map.len(), 1);
    assert_eq!(
        skill_map.get("calendar").unwrap().source_info.source,
        "first"
    );
    assert_eq!(collision_warnings.len(), 1);
    assert!(collision_warnings[0].1.contains("name collision"));
}
