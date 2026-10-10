//! Port of `packages/coding-agent/src/core/skills.ts` — Agent Skills
//! discovery, validation, and system-prompt formatting.
//!
//! Deviations:
//! - D24 — upstream's npm `ignore` matcher is replaced by the `ignore`
//!   crate's `gitignore` builder with identical gitignore semantics; ignore
//!   rules are accumulated per directory (parent first) and the matcher is
//!   rebuilt for each directory level, matching upstream's shared-matcher
//!   ordering.
//! - D26 — validation error messages use Unicode scalar counts where
//!   upstream uses JS `String.length` (UTF-16 units), and I/O failures
//!   surface `std::io::Error` display strings instead of Node error
//!   messages.
//!
//! `resolvePath(input, base, { trim: true })` (options form) is expressed by
//! trimming before calling [`utils::paths::resolve_path`]; behavior is
//! identical.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use indexmap::IndexMap;

use crate::config::{get_agent_dir, CONFIG_DIR_NAME};
use crate::core::diagnostics::{
    DiagnosticType, ResourceCollision, ResourceDiagnostic, ResourceType,
};
use crate::core::source_info::{create_synthetic_source_info, SourceInfo, SourceScope};
use crate::utils::frontmatter::parse_frontmatter;
use crate::utils::paths::{canonicalize_path, join, resolve_path, resolve_path_cwd};

/// Max name length per spec
const MAX_NAME_LENGTH: usize = 64;

/// Max description length per spec
const MAX_DESCRIPTION_LENGTH: usize = 1024;

const IGNORE_FILE_NAMES: [&str; 3] = [".gitignore", ".ignore", ".fdignore"];

/// Test-facing read-tool selection for [`format_skills_for_prompt`]
/// (upstream the `"read" | "bash" | "indirect"` union).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FileReadTool {
    #[default]
    Read,
    Bash,
    /// Names no tool: the reader is reachable only through another tool.
    Indirect,
}

/// A discovered skill (upstream `Skill`).
#[derive(Debug, Clone, PartialEq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub file_path: String,
    pub base_dir: String,
    pub source_info: SourceInfo,
    pub disable_model_invocation: bool,
}

/// Result of loading skills from one or more locations (upstream
/// `LoadSkillsResult`).
#[derive(Debug, Clone, Default)]
pub struct LoadSkillsResult {
    pub skills: Vec<Skill>,
    pub diagnostics: Vec<ResourceDiagnostic>,
}

/// Options for [`load_skills_from_dir`] (upstream `LoadSkillsFromDirOptions`).
pub struct LoadSkillsFromDirOptions<'a> {
    /// Directory to scan for skills
    pub dir: &'a str,
    /// Source identifier for these skills
    pub source: &'a str,
}

/// Options for [`load_skills`] (upstream `LoadSkillsOptions`).
pub struct LoadSkillsOptions<'a> {
    /// Working directory for project-local skills.
    pub cwd: &'a str,
    /// Agent config directory for global skills.
    pub agent_dir: Option<&'a str>,
    /// Explicit skill paths (files or directories)
    pub skill_paths: &'a [String],
    /// Include default skills directories.
    pub include_defaults: bool,
}

fn to_posix_path(p: &str) -> String {
    p.replace(std::path::MAIN_SEPARATOR, "/")
}

fn prefix_ignore_pattern(line: &str, prefix: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('#') && !trimmed.starts_with("\\#") {
        return None;
    }

    let mut pattern = line;
    let mut negated = false;

    if let Some(rest) = pattern.strip_prefix('!') {
        negated = true;
        pattern = rest;
    } else if let Some(rest) = pattern.strip_prefix("\\!") {
        pattern = rest;
    }

    pattern = pattern.strip_prefix('/').unwrap_or(pattern);

    let prefixed = if prefix.is_empty() {
        pattern.to_string()
    } else {
        format!("{prefix}{pattern}")
    };
    Some(if negated {
        format!("!{prefixed}")
    } else {
        prefixed
    })
}

/// Accumulated gitignore rules rooted at the scan root (see deviation D24).
#[derive(Debug, Clone)]
struct IgnoreMatcher {
    root: String,
    lines: Vec<String>,
}

impl IgnoreMatcher {
    fn new(root: &str) -> Self {
        IgnoreMatcher {
            root: root.to_string(),
            lines: Vec::new(),
        }
    }

    /// Read the ignore files of `dir` and append their (prefixed) patterns.
    fn add_dir_rules(&mut self, dir: &str) {
        let relative_dir = path_relative(&self.root, dir);
        let prefix = if relative_dir.is_empty() {
            String::new()
        } else {
            format!("{}/", to_posix_path(&relative_dir))
        };

        for filename in IGNORE_FILE_NAMES {
            let ignore_path = join(dir, filename);
            if !Path::new(&ignore_path).exists() {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&ignore_path) else {
                continue;
            };
            for line in content.split(|c| c == '\n' || c == '\r') {
                if let Some(pattern) = prefix_ignore_pattern(line, &prefix) {
                    self.lines.push(pattern);
                }
            }
        }
    }

    /// Build a matcher over the rules accumulated so far. Infallible:
    /// malformed lines are dropped (upstream swallows `ig.add` failures).
    fn build(&self) -> Gitignore {
        let mut builder = GitignoreBuilder::new(&self.root);
        for line in &self.lines {
            let _ = builder.add_line(Some(PathBuf::from(&self.root)), line);
        }
        match builder.build() {
            Ok(gitignore) => gitignore,
            Err(_) => GitignoreBuilder::new(&self.root)
                .build()
                .unwrap_or_else(|_| Gitignore::empty()),
        }
    }

    /// Whether `rel` (posix relative to the root) is ignored.
    fn ignores(&self, gitignore: &Gitignore, rel: &str, is_dir: bool) -> bool {
        matches!(
            gitignore.matched(Path::new(rel), is_dir),
            ignore::Match::Ignore(_)
        )
    }
}

/// `path.relative(root, dir)`-style lexical relative path ("" if equal).
fn path_relative(root: &str, dir: &str) -> String {
    let root = Path::new(root);
    let dir = Path::new(dir);
    match dir.strip_prefix(root) {
        Ok(rel) => rel.to_string_lossy().into_owned(),
        Err(_) => dir.to_string_lossy().into_owned(),
    }
}

/// Validate skill name per Agent Skills spec. Returns array of validation
/// error messages (empty if valid).
fn validate_name(name: &str) -> Vec<String> {
    let mut errors = Vec::new();

    if name.chars().count() > MAX_NAME_LENGTH {
        errors.push(format!(
            "name exceeds {MAX_NAME_LENGTH} characters ({})",
            name.chars().count()
        ));
    }

    let valid_chars = !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
    if !valid_chars {
        errors.push(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
                .to_string(),
        );
    }

    if name.starts_with('-') || name.ends_with('-') {
        errors.push("name must not start or end with a hyphen".to_string());
    }

    if name.contains("--") {
        errors.push("name must not contain consecutive hyphens".to_string());
    }

    errors
}

/// Validate description per Agent Skills spec.
fn validate_description(description: Option<&str>) -> Vec<String> {
    let mut errors = Vec::new();
    match description {
        None => errors.push("description is required".to_string()),
        Some(desc) if desc.trim().is_empty() => errors.push("description is required".to_string()),
        Some(desc) => {
            let len = desc.chars().count();
            if len > MAX_DESCRIPTION_LENGTH {
                errors.push(format!(
                    "description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len})"
                ));
            }
        }
    }
    errors
}

fn create_skill_source_info(file_path: &str, base_dir: &str, source: &str) -> SourceInfo {
    match source {
        "user" => create_synthetic_source_info(
            file_path,
            "local",
            Some(SourceScope::User),
            None,
            Some(base_dir),
        ),
        "project" => create_synthetic_source_info(
            file_path,
            "local",
            Some(SourceScope::Project),
            None,
            Some(base_dir),
        ),
        "path" => create_synthetic_source_info(file_path, "local", None, None, Some(base_dir)),
        other => create_synthetic_source_info(file_path, other, None, None, Some(base_dir)),
    }
}

/// Load skills from a directory.
///
/// Discovery rules:
/// - if a directory contains SKILL.md, treat it as a skill root and do not recurse further
/// - otherwise, load direct .md children in the root
/// - recurse into subdirectories to find SKILL.md
pub fn load_skills_from_dir(options: LoadSkillsFromDirOptions) -> LoadSkillsResult {
    load_skills_from_dir_internal(options.dir, options.source, true, None)
}

fn load_skills_from_dir_internal(
    dir: &str,
    source: &str,
    include_root_files: bool,
    parent: Option<(&IgnoreMatcher, &str)>,
) -> LoadSkillsResult {
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();

    if !Path::new(dir).exists() {
        return LoadSkillsResult {
            skills,
            diagnostics,
        };
    }

    let (matcher, root): (IgnoreMatcher, String) = match parent {
        Some((parent_matcher, root)) => {
            let mut matcher = parent_matcher.clone();
            matcher.add_dir_rules(dir);
            (matcher, root.to_string())
        }
        None => {
            let mut matcher = IgnoreMatcher::new(dir);
            matcher.add_dir_rules(dir);
            let root = dir.to_string();
            (matcher, root)
        }
    };
    let gitignore = matcher.build();

    let Ok(entries) = std::fs::read_dir(dir) else {
        return LoadSkillsResult {
            skills,
            diagnostics,
        };
    };
    let entries: Vec<std::fs::DirEntry> = entries.filter_map(|entry| entry.ok()).collect();

    for entry in &entries {
        if entry.file_name() != "SKILL.md" {
            continue;
        }

        let full_path = join(dir, &entry.file_name().to_string_lossy());

        let mut is_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false);
        if entry.file_type().map(|t| t.is_symlink()).unwrap_or(false) {
            match std::fs::metadata(&full_path) {
                Ok(stats) => is_file = stats.is_file(),
                Err(_) => continue,
            }
        }

        let rel_path = to_posix_path(&path_relative(&root, &full_path));
        if !is_file || matcher.ignores(&gitignore, &rel_path, false) {
            continue;
        }

        let result = load_skill_from_file(&full_path, source);
        if let Some(skill) = result.0 {
            skills.push(skill);
        }
        diagnostics.extend(result.1);
        return LoadSkillsResult {
            skills,
            diagnostics,
        };
    }

    for entry in &entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }

        // Skip node_modules to avoid scanning dependencies
        if name == "node_modules" {
            continue;
        }

        let full_path = join(dir, &name);

        // For symlinks, check if they point to a directory and follow them
        let mut file_type = entry.file_type().ok();
        if file_type.as_ref().is_some_and(|t| t.is_symlink()) {
            match std::fs::metadata(&full_path) {
                Ok(stats) => file_type = Some(stats.file_type()),
                Err(_) => continue, // Broken symlink, skip it
            }
        }
        let Some(file_type) = file_type else { continue };

        let rel_path = to_posix_path(&path_relative(&root, &full_path));
        let ignore_path = if file_type.is_dir() {
            format!("{rel_path}/")
        } else {
            rel_path
        };
        if matcher.ignores(&gitignore, &ignore_path, file_type.is_dir()) {
            continue;
        }

        if file_type.is_dir() {
            let sub_result =
                load_skills_from_dir_internal(&full_path, source, false, Some((&matcher, &root)));
            skills.extend(sub_result.skills);
            diagnostics.extend(sub_result.diagnostics);
            continue;
        }

        if !file_type.is_file() || !include_root_files || !name.ends_with(".md") {
            continue;
        }

        let result = load_skill_from_file(&full_path, source);
        if let Some(skill) = result.0 {
            skills.push(skill);
        }
        diagnostics.extend(result.1);
    }

    LoadSkillsResult {
        skills,
        diagnostics,
    }
}

/// Load a single skill file: `None` plus any diagnostics when the file is
/// skipped (upstream `loadSkillFromFile`).
fn load_skill_from_file(file_path: &str, source: &str) -> (Option<Skill>, Vec<ResourceDiagnostic>) {
    let mut diagnostics = Vec::new();
    let is_declared_skill = Path::new(file_path)
        .file_name()
        .map(|n| n == "SKILL.md")
        .unwrap_or(false);

    let raw_content = match std::fs::read_to_string(file_path) {
        Ok(content) => content,
        Err(error) => {
            diagnostics.push(ResourceDiagnostic {
                r#type: DiagnosticType::Warning,
                message: error.to_string(),
                path: Some(file_path.to_string()),
                collision: None,
            });
            return (None, diagnostics);
        }
    };

    let frontmatter = match parse_frontmatter(&raw_content) {
        Ok(parsed) => parsed.frontmatter,
        Err(error) => {
            if is_declared_skill {
                diagnostics.push(ResourceDiagnostic {
                    r#type: DiagnosticType::Warning,
                    message: error.0,
                    path: Some(file_path.to_string()),
                    collision: None,
                });
            }
            return (None, diagnostics);
        }
    };

    let description = frontmatter.get("description").and_then(|v| v.as_str());
    let has_description = description.is_some_and(|d| !d.trim().is_empty());
    if !is_declared_skill && !has_description {
        return (None, diagnostics);
    }

    let skill_dir = Path::new(file_path)
        .parent()
        .unwrap_or(Path::new("."))
        .to_string_lossy()
        .into_owned();
    let parent_dir_name = Path::new(&skill_dir)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    // Validate description
    for error in validate_description(description) {
        diagnostics.push(ResourceDiagnostic {
            r#type: DiagnosticType::Warning,
            message: error,
            path: Some(file_path.to_string()),
            collision: None,
        });
    }

    // Use name from frontmatter, or fall back to parent directory name
    let frontmatter_name = frontmatter
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|n| !n.is_empty());
    let name = frontmatter_name.unwrap_or(&parent_dir_name).to_string();

    // Validate name
    for error in validate_name(&name) {
        diagnostics.push(ResourceDiagnostic {
            r#type: DiagnosticType::Warning,
            message: error,
            path: Some(file_path.to_string()),
            collision: None,
        });
    }

    // Still load the skill even with warnings, unless description is missing or empty.
    if !has_description {
        return (None, diagnostics);
    }

    (
        Some(Skill {
            description: description.unwrap_or_default().to_string(),
            name,
            file_path: file_path.to_string(),
            source_info: create_skill_source_info(file_path, &skill_dir, source),
            base_dir: skill_dir,
            disable_model_invocation: frontmatter.get("disable-model-invocation")
                == Some(&serde_json::Value::Bool(true)),
        }),
        diagnostics,
    )
}

/// Format skills for inclusion in a system prompt.
/// Uses XML format per Agent Skills standard.
/// See: <https://agentskills.io/integrate-skills>
///
/// Skills with `disableModelInvocation=true` are excluded from the prompt
/// (they can only be invoked explicitly via /skill:name commands).
pub fn format_skills_for_prompt(skills: &[Skill], file_read_tool: FileReadTool) -> String {
    let visible_skills: Vec<&Skill> = skills
        .iter()
        .filter(|s| !s.disable_model_invocation)
        .collect();

    if visible_skills.is_empty() {
        return String::new();
    }

    let mut lines = vec![
		"\n\nThe following skills provide specialized instructions for specific tasks.".to_string(),
		match file_read_tool {
			FileReadTool::Read => "Use the read tool to load a skill's file when the task matches its description.",
			FileReadTool::Bash => "Use bash to load a skill's file when the task matches its description.",
			FileReadTool::Indirect => "Load a skill's file when the task matches its description.",
		}
		.to_string(),
		"When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.".to_string(),
		String::new(),
		"<available_skills>".to_string(),
	];

    for skill in visible_skills {
        lines.push("  <skill>".to_string());
        lines.push(format!("    <name>{}</name>", escape_xml(&skill.name)));
        lines.push(format!(
            "    <description>{}</description>",
            escape_xml(&skill.description)
        ));
        lines.push(format!(
            "    <location>{}</location>",
            escape_xml(&skill.file_path)
        ));
        lines.push("  </skill>".to_string());
    }

    lines.push("</available_skills>".to_string());

    lines.join("\n")
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Load skills from all configured locations. Returns skills and any
/// validation diagnostics (upstream `loadSkills`).
pub fn load_skills(options: LoadSkillsOptions) -> LoadSkillsResult {
    let resolved_cwd = resolve_path_cwd(options.cwd);
    let resolved_agent_dir = match options.agent_dir {
        Some(dir) => resolve_path_cwd(dir),
        None => resolve_path_cwd(&get_agent_dir()),
    };

    let mut skill_map: IndexMap<String, Skill> = IndexMap::new();
    let mut real_path_set: HashSet<String> = HashSet::new();
    let mut all_diagnostics = Vec::new();
    let mut collision_diagnostics = Vec::new();

    macro_rules! add_skills {
        ($result:expr) => {{
            let result = $result;
            all_diagnostics.extend(result.diagnostics);
            for skill in result.skills {
                // Resolve symlinks to detect duplicate files
                let real_path = canonicalize_path(&skill.file_path);

                // Skip silently if we've already loaded this exact file (via symlink)
                if real_path_set.contains(&real_path) {
                    continue;
                }

                match skill_map.get(&skill.name) {
                    Some(existing) => collision_diagnostics.push(ResourceDiagnostic {
                        r#type: DiagnosticType::Collision,
                        message: format!("name \"{}\" collision", skill.name),
                        path: Some(skill.file_path.clone()),
                        collision: Some(ResourceCollision {
                            resource_type: ResourceType::Skill,
                            name: skill.name.clone(),
                            winner_path: existing.file_path.clone(),
                            loser_path: skill.file_path.clone(),
                            winner_source: None,
                            loser_source: None,
                        }),
                    }),
                    None => {
                        real_path_set.insert(real_path);
                        skill_map.insert(skill.name.clone(), skill);
                    }
                }
            }
        }};
    }

    if options.include_defaults {
        add_skills!(load_skills_from_dir_internal(
            &join(&resolved_agent_dir, "skills"),
            "user",
            true,
            None
        ));
        add_skills!(load_skills_from_dir_internal(
            &resolve_path(&join(CONFIG_DIR_NAME, "skills"), &resolved_cwd),
            "project",
            true,
            None
        ));
    }

    let user_skills_dir = join(&resolved_agent_dir, "skills");
    let project_skills_dir = resolve_path(&join(CONFIG_DIR_NAME, "skills"), &resolved_cwd);

    let is_under_path = |target: &str, root: &str| -> bool {
        let normalized_root = resolve_path_cwd(root);
        if target == normalized_root {
            return true;
        }
        let prefix = if normalized_root.ends_with(std::path::MAIN_SEPARATOR) {
            normalized_root.clone()
        } else {
            format!("{normalized_root}{}", std::path::MAIN_SEPARATOR)
        };
        target.starts_with(&prefix)
    };

    let get_source = |resolved_path: &str| -> &'static str {
        if !options.include_defaults {
            if is_under_path(resolved_path, &user_skills_dir) {
                return "user";
            }
            if is_under_path(resolved_path, &project_skills_dir) {
                return "project";
            }
        }
        "path"
    };

    for raw_path in options.skill_paths {
        // Upstream: resolvePath(rawPath, resolvedCwd, { trim: true })
        let resolved_path = resolve_path(raw_path.trim(), &resolved_cwd);
        if !Path::new(&resolved_path).exists() {
            all_diagnostics.push(ResourceDiagnostic {
                r#type: DiagnosticType::Warning,
                message: "skill path does not exist".to_string(),
                path: Some(resolved_path.clone()),
                collision: None,
            });
            continue;
        }

        match std::fs::metadata(&resolved_path) {
            Ok(stats) => {
                let source = get_source(&resolved_path);
                if stats.is_dir() {
                    add_skills!(load_skills_from_dir_internal(
                        &resolved_path,
                        source,
                        true,
                        None
                    ));
                } else if stats.is_file() && resolved_path.ends_with(".md") {
                    let (skill, file_diagnostics) = load_skill_from_file(&resolved_path, source);
                    if let Some(skill) = skill {
                        add_skills!(LoadSkillsResult {
                            skills: vec![skill],
                            diagnostics: file_diagnostics
                        });
                    } else {
                        all_diagnostics.extend(file_diagnostics);
                    }
                } else {
                    all_diagnostics.push(ResourceDiagnostic {
                        r#type: DiagnosticType::Warning,
                        message: "skill path is not a markdown file".to_string(),
                        path: Some(resolved_path.clone()),
                        collision: None,
                    });
                }
            }
            Err(error) => {
                all_diagnostics.push(ResourceDiagnostic {
                    r#type: DiagnosticType::Warning,
                    message: error.to_string(),
                    path: Some(resolved_path.clone()),
                    collision: None,
                });
            }
        }
    }

    let skills = skill_map.into_values().collect();
    let mut diagnostics = all_diagnostics;
    diagnostics.extend(collision_diagnostics);
    LoadSkillsResult {
        skills,
        diagnostics,
    }
}

#[cfg(test)]
#[path = "skills_tests.rs"]
mod skills_tests;
