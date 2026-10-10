//! Port of `packages/durable/src/tools/edit-diff.ts` — shared diff computation for the edit tool.
//!
//! D1: the npm `diff` package is replaced by a hand-rolled unified-patch serializer (below) that
//! reproduces `Diff.createTwoFilesPatch(path, path, old, new, undefined, undefined,
//! { context: 4, headerOptions: FILE_HEADERS_ONLY })` byte-for-byte given the same line parts,
//! plus `similar::TextDiff::from_lines` for the underlying line diff. Hunk boundaries may differ
//! from npm `diff` in diff-ambiguous regions (different Myers implementations); the format and
//! semantics are identical.
//! D2: string indices are Rust byte offsets (`str`), not JS UTF-16 code units; all arithmetic is
//! internal to these functions, so results (strings) are identical.

use similar::{ChangeTag, TextDiff};

use super::unicode;

/// Line ending detected in a file's content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineEnding {
    Crlf,
    Lf,
}

pub fn detect_line_ending(content: &str) -> LineEnding {
    let crlf_idx = content.find("\r\n");
    let lf_idx = content.find('\n');
    match lf_idx {
        None => LineEnding::Lf,
        Some(lf) => match crlf_idx {
            None => LineEnding::Lf,
            Some(crlf) => {
                if crlf < lf {
                    LineEnding::Crlf
                } else {
                    LineEnding::Lf
                }
            }
        },
    }
}

pub fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

pub fn restore_line_endings(text: &str, ending: LineEnding) -> String {
    match ending {
        LineEnding::Crlf => text.replace('\n', "\r\n"),
        LineEnding::Lf => text.to_string(),
    }
}

/// Normalize text for fuzzy matching: NFKC, strip trailing whitespace per line, normalize smart
/// quotes, dashes/hyphens, and special Unicode spaces to ASCII.
pub fn normalize_for_fuzzy_match(text: &str) -> String {
    let normalized = unicode::nfkc(text);
    normalized
        .split('\n')
        .map(|line| line.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
            | '\u{2212}' => '-',
            '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
            other => other,
        })
        .collect()
}

/// `content.match(/[^\n]*\n|[^\n]+/g)`: lines with their newline terminators.
fn split_lines_with_endings(content: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut rest = content;
    while let Some(index) = rest.find('\n') {
        let (line, remainder) = rest.split_at(index + 1);
        lines.push(line);
        rest = remainder;
    }
    if !rest.is_empty() {
        lines.push(rest);
    }
    lines
}

#[derive(Debug, Clone, Copy)]
struct LineSpan {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
pub struct MatchedEdit {
    pub edit_index: usize,
    pub match_index: usize,
    pub match_length: usize,
    pub new_text: String,
}

#[derive(Debug, Clone)]
struct TextReplacement {
    match_index: usize,
    match_length: usize,
    new_text: String,
}

fn get_line_spans(content: &str) -> Vec<LineSpan> {
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

fn get_replacement_line_range(
    lines: &[LineSpan],
    replacement: &TextReplacement,
) -> Result<(usize, usize), String> {
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

fn apply_replacements(content: &str, replacements: &[TextReplacement], offset: usize) -> String {
    let mut result = content.to_string();
    for replacement in replacements.iter().rev() {
        let match_index = replacement.match_index - offset;
        result.replace_range(
            match_index..match_index + replacement.match_length,
            &replacement.new_text,
        );
    }
    result
}

/// Apply replacements matched against `baseContent` to `originalContent` while preserving
/// unchanged line blocks from the original.
pub fn apply_replacements_preserving_unchanged_lines(
    original_content: &str,
    base_content: &str,
    replacements: &[MatchedEdit],
) -> Result<String, String> {
    let original_lines = split_lines_with_endings(original_content);
    let base_lines = get_line_spans(base_content);
    if original_lines.len() != base_lines.len() {
        return Err(
            "Cannot preserve unchanged lines because the base content has a different line count."
                .to_string(),
        );
    }

    let as_replacement = |edit: &MatchedEdit| TextReplacement {
        match_index: edit.match_index,
        match_length: edit.match_length,
        new_text: edit.new_text.clone(),
    };
    let mut sorted_replacements: Vec<TextReplacement> =
        replacements.iter().map(as_replacement).collect();
    sorted_replacements.sort_by_key(|r| r.match_index);

    struct Group {
        start_line: usize,
        end_line: usize,
        replacements: Vec<TextReplacement>,
    }
    let mut groups: Vec<Group> = Vec::new();
    for replacement in &sorted_replacements {
        let (start_line, end_line) = get_replacement_line_range(&base_lines, replacement)?;
        match groups.last_mut() {
            Some(current) if start_line < current.end_line => {
                current.end_line = current.end_line.max(end_line);
                current.replacements.push(as_replacement(&MatchedEdit {
                    edit_index: 0,
                    match_index: replacement.match_index,
                    match_length: replacement.match_length,
                    new_text: replacement.new_text.clone(),
                }));
            }
            _ => groups.push(Group {
                start_line,
                end_line,
                replacements: vec![TextReplacement {
                    match_index: replacement.match_index,
                    match_length: replacement.match_length,
                    new_text: replacement.new_text.clone(),
                }],
            }),
        }
    }

    let mut original_line_index = 0;
    let mut result = String::new();
    for group in &groups {
        result.extend(
            original_lines[original_line_index..group.start_line]
                .iter()
                .copied(),
        );

        let group_start_offset = base_lines[group.start_line].start;
        let group_end_offset = base_lines[group.end_line - 1].end;
        result.push_str(&apply_replacements(
            &base_content[group_start_offset..group_end_offset],
            &group.replacements,
            group_start_offset,
        ));
        original_line_index = group.end_line;
    }
    result.extend(original_lines[original_line_index..].iter().copied());
    Ok(result)
}

/// Result of [`fuzzy_find_text`].
pub struct FuzzyMatchResult {
    pub found: bool,
    /// The index where the match starts (in `content_for_replacement`).
    pub index: usize,
    pub match_length: usize,
    /// Whether fuzzy matching was used (`false` = exact match).
    pub used_fuzzy_match: bool,
    /// The content to use for replacement operations: the original content for an exact match,
    /// the fuzzy-normalized content otherwise.
    pub content_for_replacement: String,
}

/// One targeted replacement.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Edit {
    pub old_text: String,
    pub new_text: String,
}

/// Result of [`apply_edits_to_normalized_content`].
#[derive(Debug)]
pub struct AppliedEditsResult {
    pub base_content: String,
    pub new_content: String,
}

/// Find `old_text` in `content`, trying exact match first, then fuzzy match.
pub fn fuzzy_find_text(content: &str, old_text: &str) -> FuzzyMatchResult {
    if let Some(exact_index) = content.find(old_text) {
        return FuzzyMatchResult {
            found: true,
            index: exact_index,
            match_length: old_text.len(),
            used_fuzzy_match: false,
            content_for_replacement: content.to_string(),
        };
    }

    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old_text = normalize_for_fuzzy_match(old_text);
    match fuzzy_content.find(&fuzzy_old_text) {
        None => FuzzyMatchResult {
            found: false,
            index: 0,
            match_length: 0,
            used_fuzzy_match: false,
            content_for_replacement: content.to_string(),
        },
        Some(fuzzy_index) => FuzzyMatchResult {
            found: true,
            index: fuzzy_index,
            match_length: fuzzy_old_text.len(),
            used_fuzzy_match: true,
            content_for_replacement: fuzzy_content,
        },
    }
}

/// Strip a UTF-8 BOM if present; return both the BOM (if any) and the text without it.
pub fn strip_bom(content: &str) -> (String, String) {
    match content.strip_prefix('\u{FEFF}') {
        Some(text) => ("\u{FEFF}".to_string(), text.to_string()),
        None => (String::new(), content.to_string()),
    }
}

fn count_occurrences(content: &str, old_text: &str) -> usize {
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old_text = normalize_for_fuzzy_match(old_text);
    if fuzzy_old_text.is_empty() {
        return 0;
    }
    fuzzy_content.matches(&fuzzy_old_text).count()
}

fn not_found_error(path: &str, edit_index: usize, total_edits: usize) -> String {
    if total_edits == 1 {
        format!(
			"Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
		)
    } else {
        format!(
			"Could not find edits[{edit_index}] in {path}. The oldText must match exactly including all whitespace and newlines."
		)
    }
}

fn duplicate_error(
    path: &str,
    edit_index: usize,
    total_edits: usize,
    occurrences: usize,
) -> String {
    if total_edits == 1 {
        format!(
			"Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
		)
    } else {
        format!(
			"Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
		)
    }
}

fn empty_old_text_error(path: &str, edit_index: usize, total_edits: usize) -> String {
    if total_edits == 1 {
        format!("oldText must not be empty in {path}.")
    } else {
        format!("edits[{edit_index}].oldText must not be empty in {path}.")
    }
}

fn no_change_error(path: &str, total_edits: usize) -> String {
    if total_edits == 1 {
        format!(
			"No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
		)
    } else {
        format!("No changes made to {path}. The replacements produced identical content.")
    }
}

/// Apply one or more exact-text replacements to LF-normalized content.
///
/// All edits are matched against the same original content; replacements apply in reverse order
/// so offsets remain stable. If any edit needs fuzzy matching, the operation runs in
/// fuzzy-normalized content space and then overlays those line-level changes onto the original
/// content so unchanged line blocks keep their original bytes.
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
            return Err(empty_old_text_error(path, i, normalized_edits.len()));
        }
    }

    let initial_matches: Vec<FuzzyMatchResult> = normalized_edits
        .iter()
        .map(|edit| fuzzy_find_text(normalized_content, &edit.old_text))
        .collect();
    let used_fuzzy_match = initial_matches.iter().any(|m| m.used_fuzzy_match);
    let replacement_base_content = if used_fuzzy_match {
        normalize_for_fuzzy_match(normalized_content)
    } else {
        normalized_content.to_string()
    };

    let mut matched_edits: Vec<MatchedEdit> = Vec::new();
    for (i, edit) in normalized_edits.iter().enumerate() {
        let match_result = fuzzy_find_text(&replacement_base_content, &edit.old_text);
        if !match_result.found {
            return Err(not_found_error(path, i, normalized_edits.len()));
        }

        let occurrences = count_occurrences(&replacement_base_content, &edit.old_text);
        if occurrences > 1 {
            return Err(duplicate_error(
                path,
                i,
                normalized_edits.len(),
                occurrences,
            ));
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
        let (previous, current) = (&matched_edits[i - 1], &matched_edits[i]);
        if previous.match_index + previous.match_length > current.match_index {
            return Err(format!(
				"edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
				previous.edit_index, current.edit_index
			));
        }
    }

    let base_content = normalized_content.to_string();
    let replacements: Vec<TextReplacement> = matched_edits
        .iter()
        .map(|edit| TextReplacement {
            match_index: edit.match_index,
            match_length: edit.match_length,
            new_text: edit.new_text.clone(),
        })
        .collect();
    let new_content = if used_fuzzy_match {
        apply_replacements_preserving_unchanged_lines(
            normalized_content,
            &replacement_base_content,
            &matched_edits,
        )?
    } else {
        apply_replacements(&replacement_base_content, &replacements, 0)
    };

    if base_content == new_content {
        return Err(no_change_error(path, normalized_edits.len()));
    }

    Ok(AppliedEditsResult {
        base_content,
        new_content,
    })
}

/// One grouped change region of the line diff: `Equal`, `Delete`, or `Insert`, with the raw
/// line values (newline terminators included, except possibly the file's last line).
enum DiffPart {
    Equal(Vec<String>),
    Delete(Vec<String>),
    Insert(Vec<String>),
}

fn diff_line_parts(old_content: &str, new_content: &str) -> Vec<DiffPart> {
    let diff = TextDiff::from_lines(old_content, new_content);
    let mut parts: Vec<DiffPart> = Vec::new();
    for change in diff.iter_all_changes() {
        let tag = change.tag();
        let push = |parts: &mut Vec<DiffPart>, value: String| {
            let target = match parts.last_mut() {
                Some(DiffPart::Equal(lines)) if tag == ChangeTag::Equal => Some(&mut *lines),
                Some(DiffPart::Delete(lines)) if tag == ChangeTag::Delete => Some(&mut *lines),
                Some(DiffPart::Insert(lines)) if tag == ChangeTag::Insert => Some(&mut *lines),
                _ => None,
            };
            match target {
                Some(lines) => lines.push(value),
                None => parts.push(match tag {
                    ChangeTag::Equal => DiffPart::Equal(vec![value]),
                    ChangeTag::Delete => DiffPart::Delete(vec![value]),
                    ChangeTag::Insert => DiffPart::Insert(vec![value]),
                }),
            }
        };
        push(&mut parts, change.value().to_string());
    }
    parts.push(DiffPart::Equal(Vec::new()));
    parts
}

/// Generate a standard unified patch: npm `diff`'s `createTwoFilesPatch(path, path, old, new,
/// undefined, undefined, { context, headerOptions: FILE_HEADERS_ONLY })` (D1).
pub fn generate_unified_patch(
    path: &str,
    old_content: &str,
    new_content: &str,
    context_lines: usize,
) -> String {
    let diff = diff_line_parts(old_content, new_content);
    let context = context_lines.max(0);

    let mut hunks: Vec<(usize, usize, usize, usize, Vec<String>)> = Vec::new();
    let mut old_range_start = 0usize;
    let mut new_range_start = 0usize;
    let mut cur_range: Vec<String> = Vec::new();
    let mut old_line = 1usize;
    let mut new_line = 1usize;

    for (i, part) in diff.iter().enumerate() {
        let (added, removed, lines): (bool, bool, &[String]) = match part {
            DiffPart::Equal(lines) => (false, false, lines),
            DiffPart::Insert(lines) => (true, false, lines),
            DiffPart::Delete(lines) => (false, true, lines),
        };

        if added || removed {
            if old_range_start == 0 {
                old_range_start = old_line;
                new_range_start = new_line;
                if i > 0 {
                    let prev_lines: &[String] = match &diff[i - 1] {
                        DiffPart::Equal(lines) => lines,
                        DiffPart::Delete(lines) => lines,
                        DiffPart::Insert(lines) => lines,
                    };
                    cur_range = if context > 0 {
                        prev_lines[prev_lines.len().saturating_sub(context)..]
                            .iter()
                            .map(|line| format!(" {line}"))
                            .collect()
                    } else {
                        Vec::new()
                    };
                    old_range_start -= cur_range.len();
                    new_range_start -= cur_range.len();
                }
            }
            for line in lines {
                cur_range.push(format!("{}{line}", if added { "+" } else { "-" }));
            }
            if added {
                new_line += lines.len();
            } else {
                old_line += lines.len();
            }
        } else if old_range_start != 0 {
            if lines.len() <= context * 2 && i + 2 < diff.len() {
                for line in lines {
                    cur_range.push(format!(" {line}"));
                }
            } else {
                let context_size = lines.len().min(context);
                for line in &lines[..context_size] {
                    cur_range.push(format!(" {line}"));
                }
                hunks.push((
                    old_range_start,
                    old_line - old_range_start + context_size,
                    new_range_start,
                    new_line - new_range_start + context_size,
                    std::mem::take(&mut cur_range),
                ));
                old_range_start = 0;
                new_range_start = 0;
            }
            old_line += lines.len();
            new_line += lines.len();
        } else {
            old_line += lines.len();
            new_line += lines.len();
        }
    }

    // Step 2: strip the trailing `\n` of each hunk line; a line without one gains
    // `\ No newline at end of file`.
    let mut ret: Vec<String> = Vec::new();
    ret.push(format!("--- {path}"));
    ret.push(format!("+++ {path}"));
    for (mut old_start, old_lines, mut new_start, new_lines, lines) in hunks {
        // Unified Diff Format quirk: if the chunk size is 0, the first number is one lower.
        if old_lines == 0 {
            old_start -= 1;
        }
        if new_lines == 0 {
            new_start -= 1;
        }
        ret.push(format!(
            "@@ -{old_start},{old_lines} +{new_start},{new_lines} @@"
        ));
        let mut skip_newline = false;
        for line in lines {
            if skip_newline {
                ret.push("\\ No newline at end of file".to_string());
                skip_newline = false;
            }
            match line.strip_suffix('\n') {
                Some(stripped) => ret.push(stripped.to_string()),
                None => {
                    ret.push(line);
                    skip_newline = true;
                }
            }
        }
        if skip_newline {
            ret.push("\\ No newline at end of file".to_string());
        }
    }

    let mut patch = ret.join("\n");
    patch.push('\n');
    patch
}

/// Generate a display-oriented diff string with line numbers and context; also returns the first
/// changed line number (in the new file).
pub fn generate_diff_string(
    old_content: &str,
    new_content: &str,
    context_lines: usize,
) -> (String, Option<usize>) {
    let parts = diff_line_parts(old_content, new_content);
    let mut output: Vec<String> = Vec::new();

    let old_lines = old_content.split('\n').count();
    let new_lines = new_content.split('\n').count();
    let max_line_num = old_lines.max(new_lines);
    let line_num_width = max_line_num.to_string().len();

    let mut old_line_num = 1usize;
    let mut new_line_num = 1usize;
    let mut last_was_change = false;
    let mut first_changed_line: Option<usize> = None;

    let trimmed = |raw: &[String]| -> Vec<String> {
        let mut raw = raw.to_vec();
        if raw.last().is_some_and(|line| line.is_empty()) {
            raw.pop();
        }
        raw.into_iter()
            .map(|line| line.strip_suffix('\n').unwrap_or(&line).to_string())
            .collect()
    };

    for (i, part) in parts.iter().enumerate() {
        let (added, removed, raw): (bool, bool, Vec<String>) = match part {
            DiffPart::Equal(lines) => (false, false, trimmed(lines)),
            DiffPart::Insert(lines) => (true, false, trimmed(lines)),
            DiffPart::Delete(lines) => (false, true, trimmed(lines)),
        };

        if added || removed {
            if first_changed_line.is_none() {
                first_changed_line = Some(new_line_num);
            }
            for line in &raw {
                if added {
                    output.push(format!(
                        "+{:>width$} {line}",
                        new_line_num,
                        width = line_num_width
                    ));
                    new_line_num += 1;
                } else {
                    output.push(format!(
                        "-{:>width$} {line}",
                        old_line_num,
                        width = line_num_width
                    ));
                    old_line_num += 1;
                }
            }
            last_was_change = true;
        } else {
            let next_part_is_change = i < parts.len() - 1
                && matches!(parts[i + 1], DiffPart::Insert(_) | DiffPart::Delete(_));
            let has_leading_change = last_was_change;
            let has_trailing_change = next_part_is_change;

            if has_leading_change && has_trailing_change {
                if raw.len() <= context_lines * 2 {
                    for line in &raw {
                        output.push(format!(
                            " {:>width$} {line}",
                            old_line_num,
                            width = line_num_width
                        ));
                        old_line_num += 1;
                        new_line_num += 1;
                    }
                } else {
                    let skipped_lines = raw.len() - context_lines * 2;
                    for line in &raw[..context_lines] {
                        output.push(format!(
                            " {:>width$} {line}",
                            old_line_num,
                            width = line_num_width
                        ));
                        old_line_num += 1;
                        new_line_num += 1;
                    }
                    output.push(format!(" {:>width$} ...", "", width = line_num_width));
                    old_line_num += skipped_lines;
                    new_line_num += skipped_lines;
                    for line in &raw[raw.len() - context_lines..] {
                        output.push(format!(
                            " {:>width$} {line}",
                            old_line_num,
                            width = line_num_width
                        ));
                        old_line_num += 1;
                        new_line_num += 1;
                    }
                }
            } else if has_leading_change {
                let skipped_lines = raw.len().saturating_sub(context_lines);
                for line in &raw[..context_lines.min(raw.len())] {
                    output.push(format!(
                        " {:>width$} {line}",
                        old_line_num,
                        width = line_num_width
                    ));
                    old_line_num += 1;
                    new_line_num += 1;
                }
                if skipped_lines > 0 {
                    output.push(format!(" {:>width$} ...", "", width = line_num_width));
                    old_line_num += skipped_lines;
                    new_line_num += skipped_lines;
                }
            } else if has_trailing_change {
                let skipped_lines = raw.len().saturating_sub(context_lines);
                if skipped_lines > 0 {
                    output.push(format!(" {:>width$} ...", "", width = line_num_width));
                    old_line_num += skipped_lines;
                    new_line_num += skipped_lines;
                }
                for line in &raw[skipped_lines.min(raw.len())..] {
                    output.push(format!(
                        " {:>width$} {line}",
                        old_line_num,
                        width = line_num_width
                    ));
                    old_line_num += 1;
                    new_line_num += 1;
                }
            } else {
                old_line_num += raw.len();
                new_line_num += raw.len();
            }

            last_was_change = false;
        }
    }

    (output.join("\n"), first_changed_line)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Minimal unified-patch applier matching npm `diff`'s `applyPatch` for patches this module
    /// produces: hunks apply at their recorded position (exact context, no fuzz), `\ No newline`
    /// markers honored.
    pub(crate) fn apply_patch(source: &str, patch: &str) -> String {
        let mut source_lines: Vec<String> = Vec::new();
        let terminated = !source.is_empty() && source.ends_with('\n');
        let body = source.strip_suffix('\n').unwrap_or(source);
        for line in body.split('\n') {
            source_lines.push(line.to_string());
        }
        let mut ends_without_newline = !terminated && !source_lines.is_empty();
        if source.is_empty() {
            source_lines.clear();
            ends_without_newline = false;
        }

        #[derive(Default)]
        struct Hunk {
            old_start: usize,
            old_count: usize,
            new_start: usize,
            new_count: usize,
            lines: Vec<String>,
        }
        let mut hunks: Vec<Hunk> = Vec::new();
        for line in patch.lines() {
            if let Some(rest) = line.strip_prefix("@@ -") {
                let Some((old_range, new_range)) = rest.split_once(" +") else {
                    continue;
                };
                let old_range = old_range.split(" @").next().unwrap_or(old_range);
                let new_range = new_range.split(" @").next().unwrap_or(new_range);
                let parse = |range: &str| -> (usize, usize) {
                    let (start, count) = range.split_once(',').unwrap_or((range, "1"));
                    (start.parse().unwrap(), count.parse().unwrap())
                };
                let (old_start, old_count) = parse(old_range);
                let (new_start, new_count) = parse(new_range);
                // npm diff reports a 0 range start as start-1.
                hunks.push(Hunk {
                    old_start: if old_count == 0 {
                        old_start + 1
                    } else {
                        old_start
                    },
                    old_count,
                    new_start: if new_count == 0 {
                        new_start + 1
                    } else {
                        new_start
                    },
                    new_count,
                    lines: Vec::new(),
                });
            } else if !hunks.is_empty()
                && (line.starts_with(' ')
                    || line.starts_with('+')
                    || line.starts_with('-')
                    || line.starts_with('\\'))
            {
                hunks.last_mut().unwrap().lines.push(line.to_string());
            }
        }

        let mut out: Vec<String> = Vec::new();
        let mut pos = 0usize;
        let mut out_ends_without_newline = false;
        for hunk in hunks {
            while pos + 1 < hunk.old_start {
                out.push(source_lines[pos].clone());
                pos += 1;
            }
            let mut hunk_old = 0usize;
            let mut no_final_newline = false;
            let mut plus_seen = false;
            for line in &hunk.lines {
                match line.chars().next() {
                    Some(' ') => {
                        out.push(source_lines[pos].clone());
                        pos += 1;
                        hunk_old += 1;
                    }
                    Some('-') => {
                        pos += 1;
                        hunk_old += 1;
                    }
                    Some('+') => {
                        plus_seen = true;
                        out.push(line[1..].to_string());
                    }
                    _ => {
                        // The `\ No newline` marker follows its own side's line: after the
                        // last `+` it applies to the result; before it, old file only.
                        no_final_newline = plus_seen;
                    }
                }
            }
            out_ends_without_newline = no_final_newline;
            let _ = hunk_old;
        }
        while pos < source_lines.len() {
            out.push(source_lines[pos].clone());
            pos += 1;
        }
        let _ = ends_without_newline;
        let mut result = out.join("\n");
        if !out.is_empty() && !out_ends_without_newline {
            result.push('\n');
        }
        result
    }

    #[test]
    fn detects_line_endings() {
        assert_eq!(detect_line_ending("a\r\nb"), LineEnding::Crlf);
        assert_eq!(detect_line_ending("a\nb\r\n"), LineEnding::Lf);
        assert_eq!(detect_line_ending("a"), LineEnding::Lf);
        assert_eq!(detect_line_ending(""), LineEnding::Lf);
    }

    #[test]
    fn normalizes_and_restores_line_endings() {
        assert_eq!(normalize_to_lf("a\r\nb\rc\nd"), "a\nb\nc\nd");
        assert_eq!(restore_line_endings("a\nb", LineEnding::Crlf), "a\r\nb");
        assert_eq!(restore_line_endings("a\nb", LineEnding::Lf), "a\nb");
    }

    #[test]
    fn fuzzy_matching() {
        // Exact match wins.
        let m = fuzzy_find_text("alpha beta", "beta");
        assert!(m.found && m.index == 6 && !m.used_fuzzy_match);
        // Smart quotes/dashes/spaces match fuzzily; replacement runs in normalized space.
        let m = fuzzy_find_text("“a–b\u{00A0}c”", "\"a-b c\"");
        assert!(m.found && m.used_fuzzy_match);
        assert_eq!(m.index, 0);
        assert_eq!(m.content_for_replacement, "\"a-b c\"");
        // No match.
        let m = fuzzy_find_text("abc", "xyz");
        assert!(!m.found);
    }

    #[test]
    fn strips_bom() {
        let (bom, text) = strip_bom("\u{FEFF}abc");
        assert_eq!((bom.as_str(), text.as_str()), ("\u{FEFF}", "abc"));
        let (bom, text) = strip_bom("abc");
        assert_eq!((bom.as_str(), text.as_str()), ("", "abc"));
    }

    #[test]
    fn applies_edits_and_reports_errors() {
        let result = apply_edits_to_normalized_content(
            "alpha\nbeta\ngamma\ndelta\n",
            &[
                Edit {
                    old_text: "alpha\n".into(),
                    new_text: "ALPHA\n".into(),
                },
                Edit {
                    old_text: "gamma\n".into(),
                    new_text: "GAMMA\n".into(),
                },
            ],
            "f",
        )
        .unwrap();
        assert_eq!(result.new_content, "ALPHA\nbeta\nGAMMA\ndelta\n");

        let err = apply_edits_to_normalized_content(
            "foo foo foo",
            &[Edit {
                old_text: "bar".into(),
                new_text: "baz".into(),
            }],
            "f",
        )
        .unwrap_err();
        assert!(
            err.starts_with("Could not find the exact text in f."),
            "{err}"
        );
        let err = apply_edits_to_normalized_content(
            "foo foo foo",
            &[Edit {
                old_text: "foo".into(),
                new_text: "x".into(),
            }],
            "f",
        )
        .unwrap_err();
        assert!(
            err.starts_with("Found 3 occurrences of the text in f."),
            "{err}"
        );
        let err = apply_edits_to_normalized_content(
            "a",
            &[Edit {
                old_text: "".into(),
                new_text: "x".into(),
            }],
            "f",
        )
        .unwrap_err();
        assert_eq!(err, "oldText must not be empty in f.");
        let err = apply_edits_to_normalized_content(
            "a",
            &[Edit {
                old_text: "a".into(),
                new_text: "a".into(),
            }],
            "f",
        )
        .unwrap_err();
        assert!(err.starts_with("No changes made to f."), "{err}");
        // Multi-edit wording.
        let err = apply_edits_to_normalized_content(
            "a\nb\n",
            &[
                Edit {
                    old_text: "a".into(),
                    new_text: "c".into(),
                },
                Edit {
                    old_text: "z".into(),
                    new_text: "z".into(),
                },
            ],
            "f",
        )
        .unwrap_err();
        assert_eq!(err, "Could not find edits[1] in f. The oldText must match exactly including all whitespace and newlines.");
    }

    #[test]
    fn rejects_overlapping_edits() {
        let err = apply_edits_to_normalized_content(
            "one\ntwo\nthree\n",
            &[
                Edit {
                    old_text: "one\ntwo\n".into(),
                    new_text: "ONE\nTWO\n".into(),
                },
                Edit {
                    old_text: "two\nthree\n".into(),
                    new_text: "TWO\nTHREE\n".into(),
                },
            ],
            "f",
        )
        .unwrap_err();
        assert!(err.contains("overlap"), "{err}");
    }

    #[test]
    fn unified_patch_round_trips_through_apply() {
        let cases: &[(&str, &str)] = &[
            ("alpha\nbeta\ngamma\ndelta\n", "ALPHA\nbeta\nGAMMA\ndelta\n"),
            ("one\n", "one\ntwo\nthree\n"),
            ("one\ntwo\nthree\n", "one\n"),
            (
                "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\n",
                "a\nb\nC\nd\ne\nf\ng\nh\ni\nj\nK\nl\n",
            ),
            ("", "new\n"),
            ("old\n", ""),
            ("no trailing newline", "no trailing newline!\n"),
            ("x\n", "x"),
        ];
        for (old, new) in cases {
            let patch = generate_unified_patch("f.txt", old, new, 4);
            assert!(patch.starts_with("--- f.txt\n+++ f.txt\n"), "{patch:?}");
            assert_eq!(apply_patch(old, &patch), *new, "patch was\n{patch}");
        }
        // Empty diff produces only the headers.
        assert_eq!(
            generate_unified_patch("f.txt", "a\n", "a\n", 4),
            "--- f.txt\n+++ f.txt\n"
        );
    }

    #[test]
    fn diff_string_format() {
        let (diff, first) = generate_diff_string(
            "alpha\nbeta\ngamma\ndelta\n",
            "ALPHA\nbeta\nGAMMA\ndelta\n",
            4,
        );
        assert_eq!(first, Some(1));
        assert!(diff.contains("ALPHA") && diff.contains("GAMMA"));
        for line in diff.split('\n').filter(|l| !l.is_empty()) {
            let (marker, rest) = line.split_at(1);
            assert!(matches!(marker, "+" | "-" | " "));
            let (num, text) = rest.split_once(' ').unwrap();
            if marker != " " || num.chars().all(|c| c.is_ascii_digit()) {
                assert!(num.chars().all(|c| c.is_ascii_digit()) && !text.is_empty());
            }
        }
    }

    #[test]
    fn preserving_unchanged_lines_overlays_normalized_space() {
        let original = "alpha  \nbeta\ngamma\n";
        let base = normalize_for_fuzzy_match(original);
        let replacement = TextReplacement {
            match_index: 0,
            match_length: "alpha".len(),
            new_text: "ALPHA".to_string(),
        };
        let result = apply_replacements_preserving_unchanged_lines(
            original,
            &base,
            &[MatchedEdit {
                edit_index: 0,
                match_index: 0,
                match_length: "alpha".len(),
                new_text: "ALPHA".to_string(),
            }],
        )
        .unwrap();
        // The touched line is rewritten from the normalized base; untouched lines keep original bytes.
        assert_eq!(result, "ALPHA\nbeta\ngamma\n");
        let _ = replacement;
    }
}
