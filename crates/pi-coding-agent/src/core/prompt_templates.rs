//! Port of `packages/coding-agent/src/core/prompt-templates.ts` — markdown
//! prompt templates: argument parsing, bash-style placeholder substitution,
//! directory scanning, and `/<name> <args>` expansion.
//!
//! Deviations:
//! - D57 — upstream's substitution regex is replaced by a hand-rolled scanner
//!   with identical match semantics (same alternation order and greedy digit
//!   parsing); see [`substitute_args`].
//! - D57b — diagnostic and description-truncation messages use Rust strings:
//!   I/O failures surface `std::io::Error` display strings instead of Node
//!   error messages, and string lengths use Unicode scalar counts where
//!   upstream uses UTF-16 units.

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::config::CONFIG_DIR_NAME;
use crate::core::diagnostics::{DiagnosticType, ResourceDiagnostic};
use crate::core::source_info::{create_synthetic_source_info, SourceInfo, SourceScope};
use crate::utils::frontmatter::parse_frontmatter;
use crate::utils::paths::{join, resolve_path, resolve_path_cwd};

/// Represents a prompt template loaded from a markdown file (upstream
/// `PromptTemplate`).
#[derive(Debug, Clone)]
pub struct PromptTemplate {
	pub name: String,
	pub description: String,
	pub argument_hint: Option<String>,
	pub content: String,
	pub source_info: SourceInfo,
	/// Absolute path to the template file.
	pub file_path: String,
}

/// Parse command arguments respecting quoted strings (bash-style). Returns
/// array of arguments (upstream `parseCommandArgs`).
pub fn parse_command_args(args_string: &str) -> Vec<String> {
	let mut args: Vec<String> = Vec::new();
	let mut current = String::new();
	let mut in_quote: Option<char> = None;

	for char in args_string.chars() {
		if let Some(quote) = in_quote {
			if char == quote {
				in_quote = None;
			} else {
				current.push(char);
			}
		} else if char == '"' || char == '\'' {
			in_quote = Some(char);
		} else if char.is_whitespace() {
			if !current.is_empty() {
				args.push(std::mem::take(&mut current));
			}
		} else {
			current.push(char);
		}
	}

	if !current.is_empty() {
		args.push(current);
	}

	args
}

/// Parse a `\d+` run starting at `bytes[index]`; returns the parsed number and
/// the end index, or `None` when no digits follow.
fn parse_digits(content: &str, index: usize) -> Option<(u64, usize)> {
	let bytes = content.as_bytes();
	let mut end = index;
	while end < bytes.len() && bytes[end].is_ascii_digit() {
		end += 1;
	}
	if end == index {
		return None;
	}
	content[index..end].parse().ok().map(|value| (value, end))
}

/// Substitute argument placeholders in template content (upstream
/// `substituteArgs`). Supports:
/// - `$1`, `$2`, ... for positional args
/// - `$@` and `$ARGUMENTS` for all args
/// - `${N:-default}` for positional arg N with default when missing/empty
/// - `${@:-default}` and `${ARGUMENTS:-default}` for all args with a default
///   when empty
/// - `${@:N}` for args from Nth onwards (bash-style slicing)
/// - `${@:N:L}` for L args starting from Nth
///
/// Note: Replacement happens on the template string only. Argument and
/// default values containing patterns like `$1`, `$@`, or `$ARGUMENTS` are NOT
/// recursively substituted.
///
/// Deviation D57: implemented as a scanner mirroring the upstream regex's
/// alternation order (`${...:-default}`, then `${@:N[:L]}`, then the simple
/// `$ARGUMENTS`/`$@`/`$N` forms) and its greedy `\d+` parsing; on a failed
/// candidate match the `$` is emitted literally and scanning advances one
/// character, exactly like the regex.
pub fn substitute_args(content: &str, args: &[String]) -> String {
	let all_args = args.join(" ");
	let mut out = String::new();
	let bytes = content.as_bytes();
	let mut i = 0;

	while i < bytes.len() {
		if bytes[i] != b'$' {
			// Copy the full (possibly multi-byte) character.
			let mut end = i + 1;
			while end < bytes.len() && !content.is_char_boundary(end) {
				end += 1;
			}
			out.push_str(&content[i..end]);
			i = end;
			continue;
		}

		// Alternative 1: `${target:-default}` with target `\d+|ARGUMENTS|@`.
		if bytes.get(i + 1) == Some(&b'{') {
			if let Some(close) = content[i + 2..].find('}').map(|offset| i + 2 + offset) {
				let inner = &content[i + 2..close];
				let matched = try_default_placeholder(inner, args, &all_args, &mut out)
					|| try_slice_placeholder(inner, args, &mut out);
				if matched {
					i = close + 1;
					continue;
				}
			}
		}

		// Alternative 2: simple `$ARGUMENTS`, `$@`, `$N`.
		if let Some(consumed) = try_simple_placeholder(content, i + 1, args, &all_args, &mut out) {
			i = consumed;
			continue;
		}

		// No pattern matched at this position: literal `$`.
		out.push('$');
		i += 1;
	}

	out
}

/// Try `${target:-default}` against `inner`; returns false when `inner` does
/// not have the shape.
fn try_default_placeholder(inner: &str, args: &[String], all_args: &str, out: &mut String) -> bool {
	let Some(sep) = inner.find(":-") else { return false };
	let target = &inner[..sep];
	let default_value = &inner[sep + 2..];
	let value = if target == "@" || target == "ARGUMENTS" {
		all_args.to_string()
	} else if !target.is_empty() && target.bytes().all(|b| b.is_ascii_digit()) {
		let index: u64 = target.parse().unwrap_or(0);
		args.get(index.saturating_sub(1) as usize).cloned().unwrap_or_default()
	} else {
		return false;
	};
	// Upstream: `value ? value : defaultValue` — empty/missing uses default.
	if value.is_empty() {
		out.push_str(default_value);
	} else {
		out.push_str(&value);
	}
	true
}

/// Try `${@:start[:length]}` against `inner`; returns false when `inner` does
/// not have the shape.
fn try_slice_placeholder(inner: &str, args: &[String], out: &mut String) -> bool {
	let Some(rest) = inner.strip_prefix("@:") else { return false };
	let Some((raw_start, after)) = parse_digits(rest, 0).map(|(value, end)| (value as i64, &rest[end..])) else {
		return false;
	};
	let length = if let Some(stripped) = after.strip_prefix(':') {
		match parse_digits(stripped, 0) {
			Some((value, end)) if end == stripped.len() => Some(value as usize),
			_ => return false,
		}
	} else if !after.is_empty() {
		return false;
	} else {
		None
	};
	// Convert to 0-indexed (user provides 1-indexed); treat 0 as 1 (bash
	// convention: args start at 1).
	let start = raw_start.saturating_sub(1).max(0) as usize;
	if let Some(length) = length {
		out.push_str(&args.iter().skip(start).take(length).cloned().collect::<Vec<_>>().join(" "));
	} else {
		out.push_str(&args.iter().skip(start).cloned().collect::<Vec<_>>().join(" "));
	}
	true
}

/// Try a simple placeholder at `content[index]` (just after `$`); returns the
/// new scan position on match.
fn try_simple_placeholder(content: &str, index: usize, args: &[String], all_args: &str, out: &mut String) -> Option<usize> {
	let rest = &content[index..];
	if let Some(stripped) = rest.strip_prefix("ARGUMENTS") {
		out.push_str(all_args);
		return Some(content.len() - stripped.len());
	}
	if rest.starts_with('@') {
		out.push_str(all_args);
		return Some(index + 1);
	}
	let (number, end) = parse_digits(content, index)?;
	// Upstream: `args[parseInt(simple, 10) - 1] ?? ""` — `$0` is index -1.
	let slot = number.checked_sub(1).map(|value| value as usize);
	if let Some(value) = slot.and_then(|index| args.get(index)) {
		out.push_str(value);
	}
	Some(end)
}

fn load_template_from_file(file_path: &str, source_info: SourceInfo) -> (Option<PromptTemplate>, Vec<ResourceDiagnostic>) {
	let mut diagnostics: Vec<ResourceDiagnostic> = Vec::new();

	let raw_content = match fs::read_to_string(file_path) {
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

	let parsed = match parse_frontmatter(&raw_content) {
		Ok(parsed) => parsed,
		Err(error) => {
			diagnostics.push(ResourceDiagnostic {
				r#type: DiagnosticType::Warning,
				message: error.0,
				path: Some(file_path.to_string()),
				collision: None,
			});
			return (None, diagnostics);
		}
	};

	let name = Path::new(file_path)
		.file_name()
		.map(|name| name.to_string_lossy().into_owned())
		.unwrap_or_default();
	let name = name.strip_suffix(".md").unwrap_or(&name).to_string();

	// Get description from frontmatter or first non-empty line.
	let mut description = parsed
		.frontmatter
		.get("description")
		.and_then(Value::as_str)
		.unwrap_or("")
		.to_string();
	if description.is_empty() {
		if let Some(first_line) = parsed.body.split('\n').find(|line| !line.trim().is_empty()) {
			// Truncate if too long. Deviation D57b: upstream truncates by
			// UTF-16 units; here by Unicode scalar count.
			if first_line.chars().count() > 60 {
				description = first_line.chars().take(60).collect();
				description.push_str("...");
			} else {
				description = first_line.to_string();
			}
		}
	}

	let argument_hint = parsed
		.frontmatter
		.get("argument-hint")
		.and_then(Value::as_str)
		.filter(|hint| !hint.is_empty())
		.map(str::to_string);

	(
		Some(PromptTemplate {
			name,
			description,
			argument_hint,
			content: parsed.body,
			source_info,
			file_path: file_path.to_string(),
		}),
		diagnostics,
	)
}

/// Result of scanning a directory for `.md` prompt templates.
fn load_templates_from_dir(
	dir: &str,
	get_source_info: &dyn Fn(&str) -> SourceInfo,
) -> (Vec<PromptTemplate>, Vec<ResourceDiagnostic>) {
	let mut templates: Vec<PromptTemplate> = Vec::new();
	let mut diagnostics: Vec<ResourceDiagnostic> = Vec::new();

	let entries = match fs::read_dir(dir) {
		Ok(entries) => entries,
		Err(_) => return (templates, diagnostics),
	};

	let mut paths: Vec<String> = entries
		.filter_map(|entry| entry.ok())
		.map(|entry| entry.path().to_string_lossy().into_owned())
		.collect();
	paths.sort();
	for full_path in paths {
		// For symlinks, check if they point to a file.
		let metadata = match fs::metadata(&full_path) {
			Ok(metadata) => metadata,
			// Broken symlink, skip it.
			Err(_) => continue,
		};
		if metadata.is_file() && full_path.ends_with(".md") {
			let (template, template_diagnostics) = load_template_from_file(&full_path, get_source_info(&full_path));
			if let Some(template) = template {
				templates.push(template);
			}
			diagnostics.extend(template_diagnostics);
		}
	}

	(templates, diagnostics)
}

/// Options for [`load_prompt_templates`] (upstream
/// `LoadPromptTemplatesOptions`).
#[derive(Debug, Clone, Default)]
pub struct LoadPromptTemplatesOptions {
	/// Working directory for project-local templates.
	pub cwd: String,
	/// Agent config directory for global templates.
	pub agent_dir: String,
	/// Explicit prompt template paths (files or directories).
	pub prompt_paths: Vec<String>,
	/// Include default prompt directories.
	pub include_defaults: bool,
}

/// Result of [`load_prompt_templates`] (upstream `LoadPromptTemplatesResult`).
#[derive(Debug, Clone, Default)]
pub struct LoadPromptTemplatesResult {
	pub templates: Vec<PromptTemplate>,
	pub diagnostics: Vec<ResourceDiagnostic>,
}

fn extend_result(
	templates: &mut Vec<PromptTemplate>,
	diagnostics: &mut Vec<ResourceDiagnostic>,
	result: (Vec<PromptTemplate>, Vec<ResourceDiagnostic>),
) {
	templates.extend(result.0);
	diagnostics.extend(result.1);
}

/// Load all prompt templates from (upstream `loadPromptTemplates`):
/// 1. Global: `agentDir/prompts/`
/// 2. Project: `cwd/{CONFIG_DIR_NAME}/prompts/`
/// 3. Explicit prompt paths
pub fn load_prompt_templates(options: &LoadPromptTemplatesOptions) -> LoadPromptTemplatesResult {
	let resolved_cwd = resolve_path_cwd(&options.cwd);
	let resolved_agent_dir = resolve_path_cwd(&options.agent_dir);

	let global_prompts_dir = join(&resolved_agent_dir, "prompts");
	let project_prompts_dir = resolve_path(&format!("{}/{}", resolved_cwd, CONFIG_DIR_NAME), &resolved_cwd);

	let is_under_path = |target: &str, root: &str| -> bool {
		let normalized_root = resolve_path_cwd(root);
		if target == normalized_root {
			return true;
		}
		let prefix = if normalized_root.ends_with(std::path::MAIN_SEPARATOR) {
			normalized_root.clone()
		} else {
			format!("{}{}", normalized_root, std::path::MAIN_SEPARATOR)
		};
		target.starts_with(&prefix)
	};

	let get_source_info = |resolved_path: &str| -> SourceInfo {
		if is_under_path(resolved_path, &global_prompts_dir) {
			return create_synthetic_source_info(
				resolved_path,
				"local",
				Some(SourceScope::User),
				None,
				Some(&global_prompts_dir),
			);
		}
		if is_under_path(resolved_path, &project_prompts_dir) {
			return create_synthetic_source_info(
				resolved_path,
				"local",
				Some(SourceScope::Project),
				None,
				Some(&project_prompts_dir),
			);
		}
		let base_dir = match fs::metadata(resolved_path) {
			Ok(metadata) if metadata.is_dir() => resolved_path.to_string(),
			_ => Path::new(resolved_path)
				.parent()
				.map(|parent| parent.to_string_lossy().into_owned())
				.unwrap_or_else(|| resolved_path.to_string()),
		};
		create_synthetic_source_info(resolved_path, "local", None, None, Some(&base_dir))
	};

	let mut templates: Vec<PromptTemplate> = Vec::new();
	let mut diagnostics: Vec<ResourceDiagnostic> = Vec::new();

	if options.include_defaults {
		extend_result(&mut templates, &mut diagnostics, load_templates_from_dir(&global_prompts_dir, &get_source_info));
		extend_result(&mut templates, &mut diagnostics, load_templates_from_dir(&project_prompts_dir, &get_source_info));
	}

	// 3. Load explicit prompt paths.
	for raw_path in &options.prompt_paths {
		// `resolvePath(rawPath, resolvedCwd, { trim: true })`.
		let resolved_path = resolve_path(raw_path.trim(), &resolved_cwd);
		let stats = match fs::metadata(&resolved_path) {
			Ok(stats) => stats,
			Err(_) => continue,
		};

		if stats.is_dir() {
			extend_result(&mut templates, &mut diagnostics, load_templates_from_dir(&resolved_path, &get_source_info));
		} else if stats.is_file() && resolved_path.ends_with(".md") {
			let (template, template_diagnostics) = load_template_from_file(&resolved_path, get_source_info(&resolved_path));
			if let Some(template) = template {
				templates.push(template);
			}
			diagnostics.extend(template_diagnostics);
		}
	}

	LoadPromptTemplatesResult {
		templates,
		diagnostics,
	}
}

/// Expand a prompt template if it matches a template name. Returns the
/// expanded content or the original text if not a template (upstream
/// `expandPromptTemplate`).
pub fn expand_prompt_template(text: &str, templates: &[PromptTemplate]) -> String {
	if !text.starts_with('/') {
		return text.to_string();
	}

	// `^\/([^\s]+)(?:\s+([\s\S]*))?$`
	let rest = &text[1..];
	let name_end = rest.find(char::is_whitespace).unwrap_or(rest.len());
	if name_end == 0 {
		return text.to_string();
	}
	let template_name = &rest[..name_end];
	let args_string = rest[name_end..].trim_start();

	if let Some(template) = templates.iter().find(|template| template.name == template_name) {
		let args = parse_command_args(args_string);
		return substitute_args(&template.content, &args);
	}

	text.to_string()
}

#[cfg(test)]
mod tests {
	use std::fs;
	use std::path::PathBuf;

	use super::*;
	use crate::config::get_agent_dir;
	use crate::core::source_info::{SourceInfo, SourceOrigin, SourceScope};

	fn args(values: &[&str]) -> Vec<String> {
		values.iter().map(|value| value.to_string()).collect()
	}

	fn substitute(content: &str, values: &[&str]) -> String {
		substitute_args(content, &args(values))
	}

	fn synthetic(path: &str) -> SourceInfo {
		create_synthetic_source_info(path, "local", Some(SourceScope::Temporary), Some(SourceOrigin::TopLevel), None)
	}

	fn write_file(path: &PathBuf, content: &str) {
		if let Some(parent) = path.parent() {
			let _ = fs::create_dir_all(parent);
		}
		fs::write(path, content).unwrap();
	}

	// ========================================================================
	// substituteArgs
	// ========================================================================

	#[test]
	fn substitute_replaces_arguments_with_all_args_joined() {
		assert_eq!(substitute("Test: $ARGUMENTS", &["a", "b", "c"]), "Test: a b c");
	}

	#[test]
	fn substitute_replaces_at_with_all_args_joined() {
		assert_eq!(substitute("Test: $@", &["a", "b", "c"]), "Test: a b c");
	}

	#[test]
	fn substitute_replaces_at_and_arguments_identically() {
		let values = ["foo", "bar", "baz"];
		assert_eq!(substitute("Test: $@", &values), substitute("Test: $ARGUMENTS", &values));
	}

	#[test]
	fn substitute_does_not_recursively_substitute_patterns_in_argument_values() {
		assert_eq!(substitute("$ARGUMENTS", &["$1", "$ARGUMENTS"]), "$1 $ARGUMENTS");
		assert_eq!(substitute("$@", &["$100", "$1"]), "$100 $1");
		assert_eq!(substitute("$ARGUMENTS", &["$100", "$1"]), "$100 $1");
	}

	#[test]
	fn substitute_supports_mixed_positional_and_arguments() {
		assert_eq!(substitute("$1: $ARGUMENTS", &["prefix", "a", "b"]), "prefix: prefix a b");
	}

	#[test]
	fn substitute_supports_mixed_positional_and_at() {
		assert_eq!(substitute("$1: $@", &["prefix", "a", "b"]), "prefix: prefix a b");
	}

	#[test]
	fn substitute_handles_empty_arguments_array_with_arguments() {
		assert_eq!(substitute("Test: $ARGUMENTS", &[]), "Test: ");
	}

	#[test]
	fn substitute_handles_empty_arguments_array_with_at() {
		assert_eq!(substitute("Test: $@", &[]), "Test: ");
	}

	#[test]
	fn substitute_handles_empty_arguments_array_with_positional() {
		assert_eq!(substitute("Test: $1", &[]), "Test: ");
	}

	#[test]
	fn substitute_handles_multiple_occurrences_of_arguments() {
		assert_eq!(substitute("$ARGUMENTS and $ARGUMENTS", &["a", "b"]), "a b and a b");
	}

	#[test]
	fn substitute_handles_multiple_occurrences_of_at() {
		assert_eq!(substitute("$@ and $@", &["a", "b"]), "a b and a b");
	}

	#[test]
	fn substitute_handles_mixed_occurrences_of_at_and_arguments() {
		assert_eq!(substitute("$@ and $ARGUMENTS", &["a", "b"]), "a b and a b");
	}

	#[test]
	fn substitute_handles_special_characters_in_arguments() {
		assert_eq!(substitute("$1 $2: $ARGUMENTS", &["arg100", "@user"]), "arg100 @user: arg100 @user");
	}

	#[test]
	fn substitute_handles_out_of_range_numbered_placeholders() {
		assert_eq!(substitute("$1 $2 $3 $4 $5", &["a", "b"]), "a b   ");
	}

	#[test]
	fn substitute_handles_unicode_characters() {
		assert_eq!(substitute("$ARGUMENTS", &["日本語", "🎉", "café"]), "日本語 🎉 café");
	}

	#[test]
	fn substitute_preserves_newlines_and_tabs_in_argument_values() {
		assert_eq!(substitute("$1 $2", &["line1\nline2", "tab\tthere"]), "line1\nline2 tab\tthere");
	}

	#[test]
	fn substitute_handles_consecutive_dollar_patterns() {
		assert_eq!(substitute("$1$2", &["a", "b"]), "ab");
	}

	#[test]
	fn substitute_handles_quoted_arguments_with_spaces() {
		assert_eq!(substitute("$ARGUMENTS", &["first arg", "second arg"]), "first arg second arg");
	}

	#[test]
	fn substitute_handles_single_argument_with_arguments() {
		assert_eq!(substitute("Test: $ARGUMENTS", &["only"]), "Test: only");
	}

	#[test]
	fn substitute_handles_single_argument_with_at() {
		assert_eq!(substitute("Test: $@", &["only"]), "Test: only");
	}

	#[test]
	fn substitute_handles_zero_index() {
		assert_eq!(substitute("$0", &["a", "b"]), "");
	}

	#[test]
	fn substitute_handles_decimal_number_only_integer_part_matches() {
		assert_eq!(substitute("$1.5", &["a"]), "a.5");
	}

	#[test]
	fn substitute_handles_arguments_as_part_of_word() {
		assert_eq!(substitute("pre$ARGUMENTS", &["a", "b"]), "prea b");
	}

	#[test]
	fn substitute_handles_at_as_part_of_word() {
		assert_eq!(substitute("pre$@", &["a", "b"]), "prea b");
	}

	#[test]
	fn substitute_handles_empty_arguments_in_middle_of_list() {
		assert_eq!(substitute("$ARGUMENTS", &["a", "", "c"]), "a  c");
	}

	#[test]
	fn substitute_handles_trailing_and_leading_spaces_in_arguments() {
		assert_eq!(substitute("$ARGUMENTS", &["  leading  ", "trailing  "]), "  leading   trailing  ");
	}

	#[test]
	fn substitute_handles_argument_containing_pattern_partially() {
		assert_eq!(substitute("Prefix $ARGUMENTS suffix", &["ARGUMENTS"]), "Prefix ARGUMENTS suffix");
	}

	#[test]
	fn substitute_handles_non_matching_patterns() {
		assert_eq!(substitute("$A $$ $ $ARGS", &["a"]), "$A $$ $ $ARGS");
	}

	#[test]
	fn substitute_handles_case_variations_case_sensitive() {
		assert_eq!(substitute("$arguments $Arguments $ARGUMENTS", &["a", "b"]), "$arguments $Arguments a b");
	}

	#[test]
	fn substitute_handles_both_syntaxes_with_same_result() {
		let values = ["x", "y", "z"];
		let result1 = substitute("$@ and $ARGUMENTS", &values);
		let result2 = substitute("$ARGUMENTS and $@", &values);
		assert_eq!(result1, result2);
		assert_eq!(result1, "x y z and x y z");
	}

	#[test]
	fn substitute_handles_very_long_argument_lists() {
		let values: Vec<String> = (0..100).map(|i| format!("arg{}", i)).collect();
		let result = substitute_args("$ARGUMENTS", &values);
		let joined = values.join(" ");
		assert_eq!(result, joined);
	}

	#[test]
	fn substitute_handles_numbered_placeholders_single_digit() {
		assert_eq!(substitute("$1 $2 $3", &["a", "b", "c"]), "a b c");
	}

	#[test]
	fn substitute_handles_numbered_placeholders_multiple_digits() {
		let values: Vec<String> = (0..15).map(|i| format!("val{}", i)).collect();
		assert_eq!(substitute_args("$10 $12 $15", &values), "val9 val11 val14");
	}

	#[test]
	fn substitute_handles_escaped_dollar_signs_literal_backslash_preserved() {
		assert_eq!(substitute("Price: \\$100", &[]), "Price: \\");
	}

	#[test]
	fn substitute_handles_mixed_numbered_and_wildcard_placeholders() {
		assert_eq!(
			substitute("$1: $@ ($ARGUMENTS)", &["first", "second", "third"]),
			"first: first second third (first second third)"
		);
	}

	#[test]
	fn substitute_handles_command_with_no_placeholders() {
		assert_eq!(substitute("Just plain text", &["a", "b"]), "Just plain text");
	}

	#[test]
	fn substitute_handles_command_with_only_placeholders() {
		assert_eq!(substitute("$1 $2 $@", &["a", "b", "c"]), "a b a b c");
	}

	// ========================================================================
	// substituteArgs - positional defaults
	// ========================================================================

	#[test]
	fn defaults_uses_default_when_positional_arg_missing() {
		assert_eq!(substitute("List exactly ${1:-7} next steps", &[]), "List exactly 7 next steps");
	}

	#[test]
	fn defaults_supports_defaults_for_all_arguments() {
		let template = "${@:-default}\n${ARGUMENTS:-default}";
		assert_eq!(substitute(template, &[]), "default\ndefault");
		assert_eq!(
			substitute(template, &["This", "would", "be", "the", "arguments"]),
			"This would be the arguments\nThis would be the arguments"
		);
	}

	#[test]
	fn defaults_uses_positional_arg_when_present() {
		assert_eq!(substitute("List exactly ${1:-7} next steps", &["3"]), "List exactly 3 next steps");
	}

	#[test]
	fn defaults_uses_default_when_positional_arg_empty() {
		assert_eq!(substitute("Mode: ${1:-brief}", &[""]), "Mode: brief");
	}

	#[test]
	fn defaults_supports_multiple_positional_defaults() {
		assert_eq!(substitute("${1:-7} ${2:-brief}", &[]), "7 brief");
		assert_eq!(substitute("${1:-7} ${2:-brief}", &["3"]), "3 brief");
		assert_eq!(substitute("${1:-7} ${2:-brief}", &["3", "verbose"]), "3 verbose");
	}

	#[test]
	fn defaults_does_not_recursively_substitute_patterns_in_arg_values() {
		assert_eq!(substitute("${1:-7}", &["$ARGUMENTS"]), "$ARGUMENTS");
		assert_eq!(substitute("${1:-7}", &["$1"]), "$1");
	}

	#[test]
	fn defaults_does_not_recursively_substitute_patterns_in_default_values() {
		assert_eq!(substitute("${1:-$ARGUMENTS}", &["a", "b"]), "a");
		assert_eq!(substitute("${3:-$ARGUMENTS}", &["a", "b"]), "$ARGUMENTS");
	}

	#[test]
	fn defaults_supports_defaults_with_spaces() {
		assert_eq!(substitute("${1:-seven steps}", &[]), "seven steps");
	}

	#[test]
	fn defaults_supports_out_of_range_positional_defaults() {
		assert_eq!(substitute("${3:-fallback}", &["a", "b"]), "fallback");
	}

	#[test]
	fn defaults_mixes_positional_defaults_with_existing_placeholders() {
		assert_eq!(substitute("$1 ${2:-x} $ARGUMENTS", &["a"]), "a x a");
	}

	// ========================================================================
	// substituteArgs - array slicing (bash-style)
	// ========================================================================

	#[test]
	fn slicing_slices_from_index() {
		assert_eq!(substitute("${@:2}", &["a", "b", "c", "d"]), "b c d");
		assert_eq!(substitute("${@:1}", &["a", "b", "c"]), "a b c");
		assert_eq!(substitute("${@:3}", &["a", "b", "c", "d"]), "c d");
	}

	#[test]
	fn slicing_slices_with_length() {
		assert_eq!(substitute("${@:2:2}", &["a", "b", "c", "d"]), "b c");
		assert_eq!(substitute("${@:1:1}", &["a", "b", "c"]), "a");
		assert_eq!(substitute("${@:3:1}", &["a", "b", "c", "d"]), "c");
		assert_eq!(substitute("${@:2:3}", &["a", "b", "c", "d", "e"]), "b c d");
	}

	#[test]
	fn slicing_handles_out_of_range_slices() {
		assert_eq!(substitute("${@:99}", &["a", "b"]), "");
		assert_eq!(substitute("${@:5}", &["a", "b"]), "");
		assert_eq!(substitute("${@:10:5}", &["a", "b"]), "");
	}

	#[test]
	fn slicing_handles_zero_length_slices() {
		assert_eq!(substitute("${@:2:0}", &["a", "b", "c"]), "");
		assert_eq!(substitute("${@:1:0}", &["a", "b"]), "");
	}

	#[test]
	fn slicing_handles_length_exceeding_array() {
		assert_eq!(substitute("${@:2:99}", &["a", "b", "c"]), "b c");
		assert_eq!(substitute("${@:1:10}", &["a", "b"]), "a b");
	}

	#[test]
	fn slicing_processes_slice_before_simple_at() {
		assert_eq!(substitute("${@:2} vs $@", &["a", "b", "c"]), "b c vs a b c");
		assert_eq!(substitute("First: ${@:1:1}, All: $@", &["x", "y", "z"]), "First: x, All: x y z");
	}

	#[test]
	fn slicing_does_not_recursively_substitute_slice_patterns_in_args() {
		assert_eq!(substitute("${@:1}", &["${@:2}", "test"]), "${@:2} test");
		assert_eq!(substitute("${@:2}", &["a", "${@:3}", "c"]), "${@:3} c");
	}

	#[test]
	fn slicing_handles_mixed_usage_with_positional_args() {
		assert_eq!(substitute("$1: ${@:2}", &["cmd", "arg1", "arg2"]), "cmd: arg1 arg2");
		assert_eq!(substitute("$1 $2 ${@:3}", &["a", "b", "c", "d"]), "a b c d");
	}

	#[test]
	fn slicing_treats_zero_as_all_args() {
		assert_eq!(substitute("${@:0}", &["a", "b", "c"]), "a b c");
	}

	#[test]
	fn slicing_handles_empty_args_array() {
		assert_eq!(substitute("${@:2}", &[]), "");
		assert_eq!(substitute("${@:1}", &[]), "");
	}

	#[test]
	fn slicing_handles_single_arg_array() {
		assert_eq!(substitute("${@:1}", &["only"]), "only");
		assert_eq!(substitute("${@:2}", &["only"]), "");
	}

	#[test]
	fn slicing_handles_slice_in_middle_of_text() {
		assert_eq!(substitute("Process ${@:2} with $1", &["tool", "file1", "file2"]), "Process file1 file2 with tool");
	}

	#[test]
	fn slicing_handles_multiple_slices_in_one_template() {
		assert_eq!(substitute("${@:1:1} and ${@:2}", &["a", "b", "c"]), "a and b c");
		assert_eq!(substitute("${@:1:2} vs ${@:3:2}", &["a", "b", "c", "d", "e"]), "a b vs c d");
	}

	#[test]
	fn slicing_handles_quoted_arguments_in_slices() {
		assert_eq!(substitute("${@:2}", &["cmd", "first arg", "second arg"]), "first arg second arg");
	}

	#[test]
	fn slicing_handles_special_characters_in_sliced_args() {
		assert_eq!(substitute("${@:2}", &["cmd", "$100", "@user", "#tag"]), "$100 @user #tag");
	}

	#[test]
	fn slicing_handles_unicode_in_sliced_args() {
		assert_eq!(substitute("${@:1}", &["日本語", "🎉", "café"]), "日本語 🎉 café");
	}

	#[test]
	fn slicing_combines_positional_slice_and_wildcard_placeholders() {
		let template = "Run $1 on ${@:2:2}, then process $@";
		let values = ["eslint", "file1.ts", "file2.ts", "file3.ts"];
		assert_eq!(
			substitute(template, &values),
			"Run eslint on file1.ts file2.ts, then process eslint file1.ts file2.ts file3.ts"
		);
	}

	#[test]
	fn slicing_handles_slice_with_no_spacing() {
		assert_eq!(substitute("prefix${@:2}suffix", &["a", "b", "c"]), "prefixb csuffix");
	}

	#[test]
	fn slicing_handles_large_slice_lengths_gracefully() {
		let values: Vec<String> = (0..10).map(|i| format!("arg{}", i + 1)).collect();
		assert_eq!(substitute_args("${@:5:100}", &values), "arg5 arg6 arg7 arg8 arg9 arg10");
	}

	// ========================================================================
	// parseCommandArgs
	// ========================================================================

	#[test]
	fn parse_parses_simple_space_separated_arguments() {
		assert_eq!(parse_command_args("a b c"), args(&["a", "b", "c"]));
	}

	#[test]
	fn parse_parses_double_quoted_arguments_with_spaces() {
		assert_eq!(parse_command_args("\"first arg\" second"), args(&["first arg", "second"]));
	}

	#[test]
	fn parse_parses_single_quoted_arguments() {
		assert_eq!(parse_command_args("'first arg' second"), args(&["first arg", "second"]));
	}

	#[test]
	fn parse_parses_mixed_quote_styles() {
		assert_eq!(
			parse_command_args("\"double\" 'single' \"double again\""),
			args(&["double", "single", "double again"])
		);
	}

	#[test]
	fn parse_handles_empty_string() {
		assert_eq!(parse_command_args(""), Vec::<String>::new());
	}

	#[test]
	fn parse_handles_extra_spaces() {
		assert_eq!(parse_command_args("a  b   c"), args(&["a", "b", "c"]));
	}

	#[test]
	fn parse_handles_tabs_as_separators() {
		assert_eq!(parse_command_args("a\tb\tc"), args(&["a", "b", "c"]));
	}

	#[test]
	fn parse_handles_quoted_empty_string() {
		// Empty quotes are skipped by current implementation.
		assert_eq!(parse_command_args("\"\" \" \""), args(&[" "]));
	}

	#[test]
	fn parse_handles_arguments_with_special_characters() {
		assert_eq!(parse_command_args("$100 @user #tag"), args(&["$100", "@user", "#tag"]));
	}

	#[test]
	fn parse_handles_unicode_characters() {
		assert_eq!(parse_command_args("日本語 🎉 café"), args(&["日本語", "🎉", "café"]));
	}

	#[test]
	fn parse_handles_newlines_in_quoted_arguments() {
		assert_eq!(parse_command_args("\"line1\nline2\" second"), args(&["line1\nline2", "second"]));
	}

	#[test]
	fn parse_treats_unquoted_newlines_as_separators() {
		assert_eq!(
			parse_command_args("label-2\n\nHere is some description #2."),
			args(&["label-2", "Here", "is", "some", "description", "#2."])
		);
	}

	#[test]
	fn parse_collapses_mixed_unquoted_whitespace() {
		assert_eq!(parse_command_args("a\n\n\tb  c"), args(&["a", "b", "c"]));
	}

	#[test]
	fn parse_handles_escaped_quotes_inside_quoted_strings() {
		// This implementation doesn't handle escaped quotes - backslash is literal.
		assert_eq!(parse_command_args("\"quoted \\\"text\\\"\""), args(&["quoted \\text\\"]));
	}

	#[test]
	fn parse_handles_trailing_spaces() {
		assert_eq!(parse_command_args("a b c   "), args(&["a", "b", "c"]));
	}

	#[test]
	fn parse_handles_leading_spaces() {
		assert_eq!(parse_command_args("   a b c"), args(&["a", "b", "c"]));
	}

	// ========================================================================
	// expandPromptTemplate
	// ========================================================================

	fn template_named(name: &str, content: &str) -> PromptTemplate {
		PromptTemplate {
			name: name.to_string(),
			description: "test".to_string(),
			argument_hint: None,
			content: content.to_string(),
			source_info: synthetic("/tmp/arg-test.md"),
			file_path: "/tmp/arg-test.md".to_string(),
		}
	}

	#[test]
	fn expand_splits_template_arguments_on_unquoted_newlines() {
		let templates = vec![template_named("arg-test", "- arg1: $1\n- rest: ${@:2}")];
		let result = expand_prompt_template("/arg-test label-2\n\nHere is some description #2.", &templates);
		assert_eq!(result, "- arg1: label-2\n- rest: Here is some description #2.");
	}

	#[test]
	fn expand_supports_template_command_separated_from_args_by_newline() {
		let templates = vec![template_named("arg-test", "arg1: $1")];
		let result = expand_prompt_template("/arg-test\nlabel-2", &templates);
		assert_eq!(result, "arg1: label-2");
	}

	// ========================================================================
	// parseCommandArgs + substituteArgs integration
	// ========================================================================

	#[test]
	fn integration_parses_and_substitutes_together() {
		let parsed = parse_command_args("Button \"onClick handler\" \"disabled support\"");
		let result = substitute_args("Create component $1 with features: $ARGUMENTS", &parsed);
		assert_eq!(result, "Create component Button with features: Button onClick handler disabled support");
	}

	#[test]
	fn integration_handles_the_readme_example() {
		let parsed = parse_command_args("Button \"onClick handler\" \"disabled support\"");
		let result = substitute_args("Create a React component named $1 with features: $ARGUMENTS", &parsed);
		assert_eq!(
			result,
			"Create a React component named Button with features: Button onClick handler disabled support"
		);
	}

	#[test]
	fn integration_produces_same_result_with_at_and_arguments() {
		let parsed = parse_command_args("feature1 feature2 feature3");
		let template1 = "Implement: $@";
		let template2 = "Implement: $ARGUMENTS";
		assert_eq!(substitute_args(template1, &parsed), substitute_args(template2, &parsed));
	}

	// ========================================================================
	// loadPromptTemplates - argument-hint frontmatter
	// ========================================================================

	struct TempDir(PathBuf);

	impl TempDir {
		fn new(label: &str) -> Self {
			let path = std::env::temp_dir().join(format!("pi-rs-test-{}-{}", label, std::process::id()));
			let _ = fs::remove_dir_all(&path);
			TempDir(path)
		}

		fn write_template(&self, name: &str, content: &str) {
			write_file(&self.0.join(format!("{}.md", name)), content);
		}

		fn path(&self) -> String {
			self.0.to_string_lossy().into_owned()
		}
	}

	impl Drop for TempDir {
		fn drop(&mut self) {
			let _ = fs::remove_dir_all(&self.0);
		}
	}

	fn load_from_test_dir(test_dir: &str) -> LoadPromptTemplatesResult {
		load_prompt_templates(&LoadPromptTemplatesOptions {
			cwd: std::env::current_dir().unwrap().to_string_lossy().into_owned(),
			agent_dir: get_agent_dir(),
			prompt_paths: vec![test_dir.to_string()],
			include_defaults: false,
		})
	}

	#[test]
	fn hint_parses_required_argument_hint_from_frontmatter() {
		let dir = TempDir::new("hint-pr");
		dir.write_template(
			"pr",
			"---\ndescription: Review PRs from URLs with structured issue and code analysis\nargument-hint: \"<PR-URL>\"\n---\nYou are given one or more GitHub PR URLs: $@",
		);

		let result = load_from_test_dir(&dir.path());
		let pr = result.templates.iter().find(|template| template.name == "pr").unwrap();
		assert_eq!(pr.argument_hint.as_deref(), Some("<PR-URL>"));
		assert_eq!(pr.description, "Review PRs from URLs with structured issue and code analysis");
	}

	#[test]
	fn hint_parses_optional_argument_hint_from_frontmatter() {
		let dir = TempDir::new("hint-wr");
		dir.write_template(
			"wr",
			"---\ndescription: Finish the current task end-to-end with changelog, commit, and push\nargument-hint: \"[instructions]\"\n---\nWrap it. Additional instructions: $ARGUMENTS",
		);

		let result = load_from_test_dir(&dir.path());
		let wr = result.templates.iter().find(|template| template.name == "wr").unwrap();
		assert_eq!(wr.argument_hint.as_deref(), Some("[instructions]"));
		assert_eq!(wr.description, "Finish the current task end-to-end with changelog, commit, and push");
	}

	#[test]
	fn hint_leaves_argument_hint_undefined_when_not_specified() {
		let dir = TempDir::new("hint-cl");
		dir.write_template(
			"cl",
			"---\ndescription: Audit changelog entries before release\n---\nAudit changelog entries for all commits since the last release.",
		);

		let result = load_from_test_dir(&dir.path());
		let cl = result.templates.iter().find(|template| template.name == "cl").unwrap();
		assert_eq!(cl.argument_hint, None);
	}

	#[test]
	fn hint_ignores_empty_argument_hint() {
		let dir = TempDir::new("hint-empty");
		dir.write_template(
			"empty-hint",
			"---\ndescription: A command with empty hint\nargument-hint: \"\"\n---\nDo something",
		);

		let result = load_from_test_dir(&dir.path());
		let template = result.templates.iter().find(|template| template.name == "empty-hint").unwrap();
		assert_eq!(template.argument_hint, None);
	}

	#[test]
	fn hint_preserves_argument_hint_with_special_characters() {
		let dir = TempDir::new("hint-is");
		dir.write_template(
			"is",
			"---\ndescription: Analyze GitHub issues (bugs or feature requests)\nargument-hint: \"<issue>\"\n---\nAnalyze GitHub issue(s): $ARGUMENTS",
		);

		let result = load_from_test_dir(&dir.path());
		let template = result.templates.iter().find(|template| template.name == "is").unwrap();
		assert_eq!(template.argument_hint.as_deref(), Some("<issue>"));
	}

	// ========================================================================
	// loadPromptTemplates - diagnostics
	// ========================================================================

	#[test]
	fn diagnostics_reports_invalid_frontmatter_and_keeps_valid_siblings() {
		// Regression test for #9354. Deviation D57b/D10: the crate's minimal
		// YAML parser (D10) accepts the upstream fixture
		// (`description: Broken: unquoted colon`) as a plain scalar, so the
		// invalid frontmatter here is a non-mapping document, which both
		// parsers reject.
		let dir = TempDir::new("invalid");
		let invalid_prompt_path = dir.0.join("invalid.md");
		write_file(&invalid_prompt_path, "---\n- not\n- a mapping\n---\nDo something.\n");
		write_file(&dir.0.join("valid.md"), "Valid prompt content.");

		let result = load_from_test_dir(&dir.path());

		let names: Vec<&str> = result.templates.iter().map(|template| template.name.as_str()).collect();
		assert_eq!(names, vec!["valid"]);
		assert_eq!(result.diagnostics.len(), 1);
		let diagnostic = &result.diagnostics[0];
		assert_eq!(diagnostic.r#type, DiagnosticType::Warning);
		assert_eq!(diagnostic.path.as_deref(), Some(invalid_prompt_path.to_string_lossy().as_ref()));
		assert!(diagnostic.message.contains("expected a mapping"), "message: {}", diagnostic.message);
	}
}
