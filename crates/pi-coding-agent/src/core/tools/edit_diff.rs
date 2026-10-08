//! Port of `packages/coding-agent/src/core/tools/edit-diff.ts` — shared diff
//! computation utilities for the edit and similar tools.
//!
//! Deviations:
//! - D37: the npm `diff` package is replaced by the `similar` crate.
//!   `generateUnifiedPatch` uses `similar::TextDiff::from_lines` +
//!   `UnifiedDiff` (hunk headers and `--- / +++` file headers match the npm
//!   `createTwoFilesPatch` `FILE_HEADERS_ONLY` format). `generateDiffString`
//!   rebuilds the npm `diffLines` part grouping by collecting maximal runs of
//!   same-kind changes from `iter_all_changes()`.
//! - `normalize("NFKC")` uses the `unicode-normalization` crate.
//! - `FuzzyMatchResult.index` was `-1` when not found upstream; the port uses
//!   `usize` set to `0` with `found: false` carrying the "not found" signal.
//! - Upstream throws `Error`s; the port returns `Result<_, String>` with the
//!   byte-identical message text.
//! - The `EditDiffResult | EditDiffError` union becomes
//!   [`EditDiffOutcome`].

use similar::ChangeTag;
use unicode_normalization::UnicodeNormalization;

use crate::core::tools::path_utils::resolve_to_cwd;
use crate::utils::text::split_bom;

/// Line ending detected for a file's content (`detectLineEnding` result).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineEnding {
	Crlf,
	Lf,
}

/// `detectLineEnding`: CRLF iff the first CRLF appears before the first LF.
pub fn detect_line_ending(content: &str) -> LineEnding {
	let crlf_idx = content.find("\r\n");
	let lf_idx = content.find('\n');
	let Some(lf_idx) = lf_idx else {
		return LineEnding::Lf;
	};
	let Some(crlf_idx) = crlf_idx else {
		return LineEnding::Lf;
	};
	if crlf_idx < lf_idx {
		LineEnding::Crlf
	} else {
		LineEnding::Lf
	}
}

/// `normalizeToLF`: convert CRLF and lone CR to LF.
pub fn normalize_to_lf(text: &str) -> String {
	text.replace("\r\n", "\n").replace('\r', "\n")
}

/// `restoreLineEndings`: convert LF back to the detected ending.
pub fn restore_line_endings(text: &str, ending: LineEnding) -> String {
	match ending {
		LineEnding::Crlf => text.replace('\n', "\r\n"),
		LineEnding::Lf => text.to_string(),
	}
}

/// Normalize text for fuzzy matching. Applies progressive transformations:
/// - NFKC normalization (upstream `.normalize("NFKC")`)
/// - Strip trailing whitespace from each line
/// - Normalize smart quotes to ASCII equivalents
/// - Normalize Unicode dashes/hyphens to ASCII hyphen
/// - Normalize special Unicode spaces to regular space
pub fn normalize_for_fuzzy_match(text: &str) -> String {
	let nfkc = text.nfkc().collect::<String>();
	nfkc
		// Strip trailing whitespace per line
		.split('\n')
		.map(|line| line.trim_end())
		.collect::<Vec<_>>()
		.join("\n")
		// Smart single quotes → '
		.replace(['\u{2018}', '\u{2019}', '\u{201A}', '\u{201B}'], "'")
		// Smart double quotes → "
		.replace(['\u{201C}', '\u{201D}', '\u{201E}', '\u{201F}'], "\"")
		// Various dashes/hyphens → -
		// U+2010 hyphen, U+2011 non-breaking hyphen, U+2012 figure dash,
		// U+2013 en-dash, U+2014 em-dash, U+2015 horizontal bar, U+2212 minus
		.replace(
			[
				'\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2015}', '\u{2212}',
			],
			"-",
		)
		// Special spaces → regular space
		// U+00A0 NBSP, U+2002-U+200A various spaces, U+202F narrow NBSP,
		// U+205F medium math space, U+3000 ideographic space
		.replace(
			[
				'\u{00A0}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}', '\u{2007}',
				'\u{2008}', '\u{2009}', '\u{200A}', '\u{202F}', '\u{205F}', '\u{3000}',
			],
			" ",
		)
}

/// `splitLinesWithEndings`: the upstream regex `/[^\n]*\n|[^\n]+/g` — lines
/// keep their `\n`; a final segment without `\n` is included; empty content
/// yields an empty vec (the regex never matches empty).
pub fn split_lines_with_endings(content: &str) -> Vec<&str> {
	let mut lines = Vec::new();
	let mut rest = content;
	while !rest.is_empty() {
		match rest.find('\n') {
			Some(idx) => {
				let (line, remainder) = rest.split_at(idx + 1);
				lines.push(line);
				rest = remainder;
			}
			None => {
				lines.push(rest);
				break;
			}
		}
	}
	lines
}

/// Byte range of one line in the content (`LineSpan`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineSpan {
	pub start: usize,
	pub end: usize,
}

/// An edit whose match location has been resolved (`MatchedEdit`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedEdit {
	pub edit_index: usize,
	pub match_index: usize,
	pub match_length: usize,
	pub new_text: String,
}

/// The fields of [`MatchedEdit`] needed to apply a replacement
/// (`TextReplacement` = `Pick<MatchedEdit, ...>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextReplacement {
	pub match_index: usize,
	pub match_length: usize,
	pub new_text: String,
}

impl From<&MatchedEdit> for TextReplacement {
	fn from(edit: &MatchedEdit) -> Self {
		TextReplacement {
			match_index: edit.match_index,
			match_length: edit.match_length,
			new_text: edit.new_text.clone(),
		}
	}
}

/// `getLineSpans`.
pub fn get_line_spans(content: &str) -> Vec<LineSpan> {
	let mut offset = 0;
	split_lines_with_endings(content)
		.into_iter()
		.map(|line| {
			let span = LineSpan {
				start: offset,
				end: offset + line.len(),
			};
			offset = span.end;
			span
		})
		.collect()
}

/// `getReplacementLineRange`: the inclusive line range a replacement touches.
pub fn get_replacement_line_range(lines: &[LineSpan], replacement: &TextReplacement) -> Result<(usize, usize), String> {
	let replacement_start = replacement.match_index;
	let replacement_end = replacement.match_index + replacement.match_length;

	let mut start_line = None;
	for (i, line) in lines.iter().enumerate() {
		if replacement_start >= line.start && replacement_start < line.end {
			start_line = Some(i);
			break;
		}
	}
	let Some(start_line) = start_line else {
		return Err("Replacement range is outside the base content.".to_string());
	};

	let mut end_line = start_line;
	while end_line < lines.len() && lines[end_line].end < replacement_end {
		end_line += 1;
	}
	if end_line >= lines.len() {
		return Err("Replacement range is outside the base content.".to_string());
	}

	Ok((start_line, end_line + 1))
}

/// `applyReplacements`: apply in reverse order so offsets remain stable;
/// `offset` shifts the match indices (used when splicing a sub-slice of the
/// base content).
pub fn apply_replacements(content: &str, replacements: &[TextReplacement], offset: usize) -> String {
	let mut result = content.to_string();
	for replacement in replacements.iter().rev() {
		let match_index = replacement.match_index - offset;
		result = format!(
			"{}{}{}",
			&result[..match_index],
			replacement.new_text,
			&result[match_index + replacement.match_length..]
		);
	}
	result
}

/// `applyReplacementsPreservingUnchangedLines`: apply replacements matched
/// against `baseContent` to `originalContent` while preserving unchanged line
/// blocks from the original.
///
/// This is useful when `baseContent` is a normalized view of the original.
/// Each replacement is widened to the lines it actually touches, those touched
/// lines are rewritten from the normalized base, and all other lines are
/// copied back from `originalContent`. The actual replacement ranges drive
/// preservation so duplicate normalized lines cannot be aligned to the wrong
/// occurrence.
pub fn apply_replacements_preserving_unchanged_lines(
	original_content: &str,
	base_content: &str,
	replacements: &[TextReplacement],
) -> Result<String, String> {
	let original_lines = split_lines_with_endings(original_content);
	let base_lines = get_line_spans(base_content);
	if original_lines.len() != base_lines.len() {
		return Err(
			"Cannot preserve unchanged lines because the base content has a different line count.".to_string(),
		);
	}

	struct Group {
		start_line: usize,
		end_line: usize,
		replacements: Vec<TextReplacement>,
	}

	let mut groups: Vec<Group> = Vec::new();
	let mut sorted_replacements: Vec<TextReplacement> = replacements.to_vec();
	sorted_replacements.sort_by_key(|replacement| replacement.match_index);
	for replacement in sorted_replacements {
		let range = get_replacement_line_range(&base_lines, &replacement)?;
		match groups.last_mut() {
			Some(current) if range.0 < current.end_line => {
				current.end_line = current.end_line.max(range.1);
				current.replacements.push(replacement);
			}
			_ => {
				groups.push(Group {
					start_line: range.0,
					end_line: range.1,
					replacements: vec![replacement],
				});
			}
		}
	}

	let mut original_line_index = 0;
	let mut result = String::new();
	for group in &groups {
		result += &original_lines[original_line_index..group.start_line].join("");

		let group_start_offset = base_lines[group.start_line].start;
		let group_end_offset = base_lines[group.end_line - 1].end;
		result += &apply_replacements(
			&base_content[group_start_offset..group_end_offset],
			&group.replacements,
			group_start_offset,
		);
		original_line_index = group.end_line;
	}
	result += &original_lines[original_line_index..].join("");

	Ok(result)
}

/// Result of fuzzy text lookup (`FuzzyMatchResult`). Upstream sets `index` to
/// `-1` when not found; the port stores `0` there and relies on
/// [`found`](Self::found).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzyMatchResult {
	/// Whether a match was found
	pub found: bool,
	/// The index where the match starts (in the content that should be used
	/// for replacement). `0` when not found (upstream `-1`; see docs above).
	pub index: usize,
	/// Length of the matched text
	pub match_length: usize,
	/// Whether fuzzy matching was used (false = exact match)
	pub used_fuzzy_match: bool,
	/// The content to use for replacement operations. When exact match:
	/// original content. When fuzzy match: normalized content.
	pub content_for_replacement: String,
}

/// One exact-text replacement (`Edit`). Wire names are `oldText`/`newText`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Edit {
	pub old_text: String,
	pub new_text: String,
}

/// Result of applying edits to normalized content (`AppliedEditsResult`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedEditsResult {
	pub base_content: String,
	pub new_content: String,
}

/// Find oldText in content, trying exact match first, then fuzzy match
/// (`fuzzyFindText`). When fuzzy matching is used, the returned
/// `content_for_replacement` is the fuzzy-normalized version of the content
/// (trailing whitespace stripped, Unicode quotes/dashes normalized to ASCII).
pub fn fuzzy_find_text(content: &str, old_text: &str) -> FuzzyMatchResult {
	// Try exact match first
	if let Some(exact_index) = content.find(old_text) {
		return FuzzyMatchResult {
			found: true,
			index: exact_index,
			match_length: old_text.len(),
			used_fuzzy_match: false,
			content_for_replacement: content.to_string(),
		};
	}

	// Try fuzzy match - work entirely in normalized space
	let fuzzy_content = normalize_for_fuzzy_match(content);
	let fuzzy_old_text = normalize_for_fuzzy_match(old_text);
	let fuzzy_index = fuzzy_content.find(&fuzzy_old_text);

	let Some(fuzzy_index) = fuzzy_index else {
		return FuzzyMatchResult {
			found: false,
			index: 0,
			match_length: 0,
			used_fuzzy_match: false,
			content_for_replacement: content.to_string(),
		};
	};

	// When fuzzy matching, return offsets in normalized space. Callers can use
	// the normalized content to compute replacements, then decide how much of
	// that normalized output should be written back.
	FuzzyMatchResult {
		found: true,
		index: fuzzy_index,
		match_length: fuzzy_old_text.len(),
		used_fuzzy_match: true,
		content_for_replacement: fuzzy_content,
	}
}

/// `countOccurrences`: occurrence count on the fuzzy-normalized content.
fn count_occurrences(content: &str, old_text: &str) -> usize {
	let fuzzy_content = normalize_for_fuzzy_match(content);
	let fuzzy_old_text = normalize_for_fuzzy_match(old_text);
	fuzzy_content.split(&fuzzy_old_text).count() - 1
}

/// `getNotFoundError`.
fn get_not_found_error(path: &str, edit_index: usize, total_edits: usize) -> String {
	if total_edits == 1 {
		return format!(
			"Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
		);
	}
	format!(
		"Could not find edits[{edit_index}] in {path}. The oldText must match exactly including all whitespace and newlines."
	)
}

/// `getDuplicateError`.
fn get_duplicate_error(path: &str, edit_index: usize, total_edits: usize, occurrences: usize) -> String {
	if total_edits == 1 {
		return format!(
			"Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
		);
	}
	format!(
		"Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
	)
}

/// `getEmptyOldTextError`.
fn get_empty_old_text_error(path: &str, edit_index: usize, total_edits: usize) -> String {
	if total_edits == 1 {
		return format!("oldText must not be empty in {path}.");
	}
	format!("edits[{edit_index}].oldText must not be empty in {path}.")
}

/// `getNoChangeError`.
fn get_no_change_error(path: &str, total_edits: usize) -> String {
	if total_edits == 1 {
		return format!(
			"No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
		);
	}
	format!("No changes made to {path}. The replacements produced identical content.")
}

/// Apply one or more exact-text replacements to LF-normalized content
/// (`applyEditsToNormalizedContent`).
///
/// All edits are matched against the same original content. Replacements are
/// then applied in reverse order so offsets remain stable. If any edit needs
/// fuzzy matching, the operation runs in fuzzy-normalized content space and
/// then overlays those line-level changes onto the original content so
/// unchanged line blocks keep their original bytes.
pub fn apply_edits_to_normalized_content(
	normalized_content: &str,
	edits: &[Edit],
	path: &str,
) -> Result<AppliedEditsResult, String> {
	let normalized_edits: Vec<Edit> = edits
		.iter()
		.map(|edit| Edit {
			old_text: normalize_to_lf(&edit.old_text),
			new_text: normalize_to_lf(&edit.new_text),
		})
		.collect();

	for (i, edit) in normalized_edits.iter().enumerate() {
		if edit.old_text.is_empty() {
			return Err(get_empty_old_text_error(path, i, normalized_edits.len()));
		}
	}

	let initial_matches: Vec<FuzzyMatchResult> = normalized_edits
		.iter()
		.map(|edit| fuzzy_find_text(normalized_content, &edit.old_text))
		.collect();
	let used_fuzzy_match = initial_matches.iter().any(|found| found.used_fuzzy_match);
	let replacement_base_content = if used_fuzzy_match {
		normalize_for_fuzzy_match(normalized_content)
	} else {
		normalized_content.to_string()
	};

	let mut matched_edits: Vec<MatchedEdit> = Vec::new();
	for (i, edit) in normalized_edits.iter().enumerate() {
		let match_result = fuzzy_find_text(&replacement_base_content, &edit.old_text);
		if !match_result.found {
			return Err(get_not_found_error(path, i, normalized_edits.len()));
		}

		let occurrences = count_occurrences(&replacement_base_content, &edit.old_text);
		if occurrences > 1 {
			return Err(get_duplicate_error(path, i, normalized_edits.len(), occurrences));
		}

		matched_edits.push(MatchedEdit {
			edit_index: i,
			match_index: match_result.index,
			match_length: match_result.match_length,
			new_text: edit.new_text.clone(),
		});
	}

	matched_edits.sort_by_key(|edit| edit.match_index);
	for i in 1..matched_edits.len() {
		let previous = &matched_edits[i - 1];
		let current = &matched_edits[i];
		if previous.match_index + previous.match_length > current.match_index {
			return Err(format!(
				"edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
				previous.edit_index, current.edit_index
			));
		}
	}

	let base_content = normalized_content;
	let replacements: Vec<TextReplacement> = matched_edits.iter().map(TextReplacement::from).collect();
	let new_content = if used_fuzzy_match {
		apply_replacements_preserving_unchanged_lines(normalized_content, &replacement_base_content, &replacements)?
	} else {
		apply_replacements(&replacement_base_content, &replacements, 0)
	};

	if base_content == new_content {
		return Err(get_no_change_error(path, normalized_edits.len()));
	}

	Ok(AppliedEditsResult {
		base_content: base_content.to_string(),
		new_content,
	})
}

/// Generate a standard unified patch (`generateUnifiedPatch`): upstream calls
/// `Diff.createTwoFilesPatch(path, path, ..., { headerOptions:
/// FILE_HEADERS_ONLY })`; the `similar` unified diff with a path header emits
/// the same `--- path` / `+++ path` lines (D37).
pub fn generate_unified_patch(path: &str, old_content: &str, new_content: &str, context_lines: usize) -> String {
	let diff = similar::TextDiff::from_lines(old_content, new_content);
	let mut unified = diff.unified_diff();
	unified.context_radius(context_lines).header(path, path);
	format!("{unified}")
}

/// Kind of a grouped run of same-kind line changes from `diffLines` (D37).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PartKind {
	Added,
	Removed,
	Equal,
}

/// One grouped run of same-kind line changes (npm `diffLines` part).
struct Part<'a> {
	kind: PartKind,
	lines: Vec<&'a str>,
}

/// Collect maximal runs of same-kind changes, mirroring npm `diffLines`
/// part grouping (removed and added lines form separate parts).
fn diff_line_parts<'a>(old_content: &'a str, new_content: &'a str) -> Vec<Part<'a>> {
	let diff = similar::TextDiff::from_lines(old_content, new_content);
	let mut parts: Vec<Part> = Vec::new();
	for change in diff.iter_all_changes() {
		let kind = match change.tag() {
			ChangeTag::Delete => PartKind::Removed,
			ChangeTag::Insert => PartKind::Added,
			ChangeTag::Equal => PartKind::Equal,
		};
		let mut lines: Vec<&str> = change.value().split('\n').collect();
		if lines.last() == Some(&"") {
			lines.pop();
		}
		match parts.last_mut() {
			Some(last) if last.kind == kind => last.lines.extend(lines),
			_ => parts.push(Part { kind, lines }),
		}
	}
	parts
}

/// Generate a display-oriented diff string with line numbers and context
/// (`generateDiffString`). Returns both the diff string and the first changed
/// line number (in the new file).
pub fn generate_diff_string(old_content: &str, new_content: &str, context_lines: usize) -> (String, Option<usize>) {
	let parts = diff_line_parts(old_content, new_content);
	let mut output: Vec<String> = Vec::new();

	let old_line_count = old_content.split('\n').count();
	let new_line_count = new_content.split('\n').count();
	let max_line_num = old_line_count.max(new_line_count);
	let line_num_width = max_line_num.to_string().len();

	let mut old_line_num: usize = 1;
	let mut new_line_num: usize = 1;
	let mut last_was_change = false;
	let mut first_changed_line: Option<usize> = None;

	let pad = |num: usize| format!("{num:>line_num_width$}");
	let pad_blank = " ".repeat(line_num_width);

	for (i, part) in parts.iter().enumerate() {
		let raw: &[&str] = &part.lines;

		let added = part.kind == PartKind::Added;
		let removed = part.kind == PartKind::Removed;

		if added || removed {
			// Capture the first changed line (in the new file)
			if first_changed_line.is_none() {
				first_changed_line = Some(new_line_num);
			}

			// Show the change
			for line in raw {
				if added {
					output.push(format!("+{} {line}", pad(new_line_num)));
					new_line_num += 1;
				} else {
					// removed
					output.push(format!("-{} {line}", pad(old_line_num)));
					old_line_num += 1;
				}
			}
			last_was_change = true;
		} else {
			// Context lines - only show a few before/after changes
			let next_part_is_change = i < parts.len() - 1
				&& matches!(parts[i + 1].kind, PartKind::Added | PartKind::Removed);
			let has_leading_change = last_was_change;
			let has_trailing_change = next_part_is_change;

			if has_leading_change && has_trailing_change {
				if raw.len() <= context_lines * 2 {
					for line in raw {
						output.push(format!(" {} {line}", pad(old_line_num)));
						old_line_num += 1;
						new_line_num += 1;
					}
				} else {
					let leading_lines = &raw[..context_lines];
					let trailing_lines = &raw[raw.len() - context_lines..];
					let skipped_lines = raw.len() - leading_lines.len() - trailing_lines.len();

					for line in leading_lines {
						output.push(format!(" {} {line}", pad(old_line_num)));
						old_line_num += 1;
						new_line_num += 1;
					}

					output.push(format!(" {pad_blank} ..."));
					old_line_num += skipped_lines;
					new_line_num += skipped_lines;

					for line in trailing_lines {
						output.push(format!(" {} {line}", pad(old_line_num)));
						old_line_num += 1;
						new_line_num += 1;
					}
				}
			} else if has_leading_change {
				let shown_lines = &raw[..context_lines.min(raw.len())];
				let skipped_lines = raw.len() - shown_lines.len();

				for line in shown_lines {
					output.push(format!(" {} {line}", pad(old_line_num)));
					old_line_num += 1;
					new_line_num += 1;
				}

				if skipped_lines > 0 {
					output.push(format!(" {pad_blank} ..."));
					old_line_num += skipped_lines;
					new_line_num += skipped_lines;
				}
			} else if has_trailing_change {
				let skipped_lines = raw.len().saturating_sub(context_lines);
				if skipped_lines > 0 {
					output.push(format!(" {pad_blank} ..."));
					old_line_num += skipped_lines;
					new_line_num += skipped_lines;
				}

				for line in &raw[skipped_lines..] {
					output.push(format!(" {} {line}", pad(old_line_num)));
					old_line_num += 1;
					new_line_num += 1;
				}
			} else {
				// Skip these context lines entirely
				old_line_num += raw.len();
				new_line_num += raw.len();
			}

			last_was_change = false;
		}
	}

	(output.join("\n"), first_changed_line)
}

/// Successful diff computation (`EditDiffResult`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditDiffResult {
	pub diff: String,
	pub first_changed_line: Option<usize>,
}

/// Failed diff computation (`EditDiffError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditDiffError {
	pub error: String,
}

/// Upstream `Promise<EditDiffResult | EditDiffError>` union; the port names
/// the success/error variants explicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditDiffOutcome {
	Result(EditDiffResult),
	Error(EditDiffError),
}

/// Map an `io::Error` onto the upstream Node error-code string used by the
/// `"code" in error` branch (`ENOENT`, `EACCES`, else `String(error)`).
fn error_code_message(error: &std::io::Error) -> String {
	match error.kind() {
		std::io::ErrorKind::NotFound => "Error code: ENOENT".to_string(),
		std::io::ErrorKind::PermissionDenied => "Error code: EACCES".to_string(),
		_ => error.to_string(),
	}
}

/// Compute the diff for one or more edit operations without applying them
/// (`computeEditsDiff`). Used for preview rendering in the TUI before the
/// tool executes.
pub async fn compute_edits_diff(path: &str, edits: &[Edit], cwd: &str) -> EditDiffOutcome {
	let absolute_path = resolve_to_cwd(path, cwd);

	// Check if file exists and is readable (upstream `access(path, R_OK)`).
	if let Err(error) = std::fs::metadata(&absolute_path).and_then(check_readable) {
		let error_message = error_code_message(&error);
		return EditDiffOutcome::Error(EditDiffError {
			error: format!("Could not edit file: {path}. {error_message}."),
		});
	}

	// Read the file
	let raw_content = match tokio::fs::read_to_string(&absolute_path).await {
		Ok(content) => content,
		// Upstream utf-8 decode failures surface through the same catch as a
		// bare `String(error)` message.
		Err(error) => {
			return EditDiffOutcome::Error(EditDiffError {
				error: error.to_string(),
			});
		}
	};

	// Strip BOM before matching (LLM won't include invisible BOM in oldText)
	let (_, content) = split_bom(&raw_content);
	let normalized_content = normalize_to_lf(content);
	let applied = match apply_edits_to_normalized_content(&normalized_content, edits, path) {
		Ok(applied) => applied,
		Err(message) => {
			return EditDiffOutcome::Error(EditDiffError { error: message });
		}
	};

	// Generate the diff
	let (diff, first_changed_line) = generate_diff_string(&applied.base_content, &applied.new_content, 4);
	EditDiffOutcome::Result(EditDiffResult {
		diff,
		first_changed_line,
	})
}

/// Check R_OK semantics on unix: the owner read bit must be set.
#[cfg(unix)]
fn check_readable(meta: std::fs::Metadata) -> std::io::Result<std::fs::Metadata> {
	use std::os::unix::fs::PermissionsExt;
	if meta.permissions().mode() & 0o400 == 0o400 {
		Ok(meta)
	} else {
		Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
	}
}

#[cfg(not(unix))]
fn check_readable(meta: std::fs::Metadata) -> std::io::Result<std::fs::Metadata> {
	Ok(meta)
}

/// Compute the diff for a single edit operation without applying it
/// (`computeEditDiff`). Kept as a convenience wrapper for single-edit callers.
pub async fn compute_edit_diff(path: &str, old_text: &str, new_text: &str, cwd: &str) -> EditDiffOutcome {
	compute_edits_diff(path, &[Edit {
		old_text: old_text.to_string(),
		new_text: new_text.to_string(),
	}], cwd)
	.await
}
