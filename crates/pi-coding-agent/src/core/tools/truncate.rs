//! Port of `packages/coding-agent/src/core/tools/truncate.ts` — shared
//! truncation utilities for tool outputs.
//!
//! Byte counts are UTF-8 (`Buffer.byteLength`); `truncateLine` counts chars
//! where upstream uses UTF-16 `.length` (crate docs D23 convention).

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// Default line limit (`DEFAULT_MAX_LINES`).
pub const DEFAULT_MAX_LINES: usize = 2000;
/// Default byte limit (`DEFAULT_MAX_BYTES`, 50KB).
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;
/// Max chars per grep match line (`GREP_MAX_LINE_LENGTH`).
pub const GREP_MAX_LINE_LENGTH: usize = 500;

/// Result of a truncation pass (`TruncationResult`). Wire type: it rides in
/// tool `details` payloads to extensions and UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TruncationResult {
	/// The truncated content.
	pub content: String,
	/// Whether truncation occurred.
	pub truncated: bool,
	/// Which limit was hit: `"lines"`, `"bytes"`, or absent when not truncated.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub truncated_by: Option<TruncatedBy>,
	/// Total number of lines in the original content.
	#[serde(rename = "totalLines")]
	pub total_lines: usize,
	/// Total number of bytes in the original content.
	#[serde(rename = "totalBytes")]
	pub total_bytes: usize,
	/// Number of complete lines in the truncated output.
	#[serde(rename = "outputLines")]
	pub output_lines: usize,
	/// Number of bytes in the truncated output.
	#[serde(rename = "outputBytes")]
	pub output_bytes: usize,
	/// Whether the last line was partially truncated (tail truncation edge
	/// case).
	#[serde(rename = "lastLinePartial", default)]
	pub last_line_partial: bool,
	/// Whether the first line exceeded the byte limit (head truncation).
	#[serde(rename = "firstLineExceedsLimit", default)]
	pub first_line_exceeds_limit: bool,
	/// The max lines limit that was applied.
	#[serde(rename = "maxLines")]
	pub max_lines: usize,
	/// The max bytes limit that was applied.
	#[serde(rename = "maxBytes")]
	pub max_bytes: usize,
}

/// Which limit ended truncation (`TruncationResult.truncatedBy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TruncatedBy {
	#[serde(rename = "lines")]
	Lines,
	#[serde(rename = "bytes")]
	Bytes,
}

/// Options for the truncation passes (`TruncationOptions`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TruncationOptions {
	/// Maximum number of lines. Default: [`DEFAULT_MAX_LINES`].
	pub max_lines: Option<usize>,
	/// Maximum number of bytes. Default: [`DEFAULT_MAX_BYTES`].
	pub max_bytes: Option<usize>,
}

/// Split into lines the way the counters see them: a trailing newline does
/// not create an extra line (`splitLinesForCounting`).
fn split_lines_for_counting(content: &str) -> Vec<&str> {
	if content.is_empty() {
		return Vec::new();
	}
	let mut lines: Vec<&str> = content.split('\n').collect();
	if content.ends_with('\n') {
		lines.pop();
	}
	lines
}

/// Format bytes as human-readable size (`formatSize`): `123B`, `1.5KB`,
/// `2.0MB`.
pub fn format_size(bytes: usize) -> String {
	if bytes < 1024 {
		format!("{bytes}B")
	} else if bytes < 1024 * 1024 {
		format!("{:.1}KB", bytes as f64 / 1024.0)
	} else {
		format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
	}
}

/// Truncate content from the head, keeping the first N lines/bytes
/// (`truncateHead`). Suitable for file reads. Never returns partial lines; if
/// the first line exceeds the byte limit, returns empty content with
/// `firstLineExceedsLimit`.
pub fn truncate_head(content: &str, options: TruncationOptions) -> TruncationResult {
	let max_lines = options.max_lines.unwrap_or(DEFAULT_MAX_LINES);
	let max_bytes = options.max_bytes.unwrap_or(DEFAULT_MAX_BYTES);

	let total_bytes = content.len();
	let lines = split_lines_for_counting(content);
	let total_lines = lines.len();

	// Check if no truncation is needed.
	if total_lines <= max_lines && total_bytes <= max_bytes {
		return TruncationResult {
			content: content.to_string(),
			truncated: false,
			truncated_by: None,
			total_lines,
			total_bytes,
			output_lines: total_lines,
			output_bytes: total_bytes,
			last_line_partial: false,
			first_line_exceeds_limit: false,
			max_lines,
			max_bytes,
		};
	}

	// Check if the first line alone exceeds the byte limit.
	let first_line_bytes = lines[0].len();
	if first_line_bytes > max_bytes {
		return TruncationResult {
			content: String::new(),
			truncated: true,
			truncated_by: Some(TruncatedBy::Bytes),
			total_lines,
			total_bytes,
			output_lines: 0,
			output_bytes: 0,
			last_line_partial: false,
			first_line_exceeds_limit: true,
			max_lines,
			max_bytes,
		};
	}

	// Collect complete lines that fit.
	let mut output_lines_arr: Vec<&str> = Vec::new();
	let mut output_bytes_count = 0usize;
	let mut truncated_by = TruncatedBy::Lines;

	for (i, line) in lines.iter().enumerate().take(max_lines) {
		let line_bytes = line.len() + if i > 0 { 1 } else { 0 }; // +1 for newline

		if output_bytes_count + line_bytes > max_bytes {
			truncated_by = TruncatedBy::Bytes;
			break;
		}

		output_lines_arr.push(line);
		output_bytes_count += line_bytes;
	}

	// If we exited due to the line limit.
	if output_lines_arr.len() >= max_lines && output_bytes_count <= max_bytes {
		truncated_by = TruncatedBy::Lines;
	}

	let output_content = output_lines_arr.join("\n");
	let final_output_bytes = output_content.len();

	TruncationResult {
		content: output_content,
		truncated: true,
		truncated_by: Some(truncated_by),
		total_lines,
		total_bytes,
		output_lines: output_lines_arr.len(),
		output_bytes: final_output_bytes,
		last_line_partial: false,
		first_line_exceeds_limit: false,
		max_lines,
		max_bytes,
	}
}

/// Truncate content from the tail, keeping the last N lines/bytes
/// (`truncateTail`). Suitable for bash output. May return a partial first
/// line when the last line of the original content exceeds the byte limit.
pub fn truncate_tail(content: &str, options: TruncationOptions) -> TruncationResult {
	let max_lines = options.max_lines.unwrap_or(DEFAULT_MAX_LINES);
	let max_bytes = options.max_bytes.unwrap_or(DEFAULT_MAX_BYTES);

	let total_bytes = content.len();
	let lines = split_lines_for_counting(content);
	let total_lines = lines.len();

	// Check if no truncation is needed.
	if total_lines <= max_lines && total_bytes <= max_bytes {
		return TruncationResult {
			content: content.to_string(),
			truncated: false,
			truncated_by: None,
			total_lines,
			total_bytes,
			output_lines: total_lines,
			output_bytes: total_bytes,
			last_line_partial: false,
			first_line_exceeds_limit: false,
			max_lines,
			max_bytes,
		};
	}

	// Work backwards from the end.
	let mut output_lines_arr: Vec<Cow<'_, str>> = Vec::new();
	let mut output_bytes_count = 0usize;
	let mut truncated_by = TruncatedBy::Lines;
	let mut last_line_partial = false;

	for line in lines.iter().rev() {
		if output_lines_arr.len() >= max_lines {
			break;
		}
		let line_bytes = line.len() + if !output_lines_arr.is_empty() { 1 } else { 0 }; // +1 for newline

		if output_bytes_count + line_bytes > max_bytes {
			truncated_by = TruncatedBy::Bytes;
			// Edge case: if we haven't added ANY lines yet and this line
			// exceeds maxBytes, take the end of the line (partial).
			if output_lines_arr.is_empty() {
				let truncated_line = truncate_string_to_bytes_from_end(line, max_bytes);
				output_bytes_count = truncated_line.len();
				output_lines_arr.insert(0, Cow::Owned(truncated_line));
				last_line_partial = true;
			}
			break;
		}

		output_lines_arr.insert(0, Cow::Borrowed(*line));
		output_bytes_count += line_bytes;
	}

	// If we exited due to the line limit.
	if output_lines_arr.len() >= max_lines && output_bytes_count <= max_bytes {
		truncated_by = TruncatedBy::Lines;
	}

	let output_content = output_lines_arr.join("\n");
	let final_output_bytes = output_content.len();

	TruncationResult {
		content: output_content,
		truncated: true,
		truncated_by: Some(truncated_by),
		total_lines,
		total_bytes,
		output_lines: output_lines_arr.len(),
		output_bytes: final_output_bytes,
		last_line_partial,
		first_line_exceeds_limit: false,
		max_lines,
		max_bytes,
	}
}

/// Truncate a string to fit within a byte limit, keeping the end
/// (`truncateStringToBytesFromEnd`). Handles multi-byte UTF-8 characters:
/// skips leading continuation bytes of a cut sequence.
fn truncate_string_to_bytes_from_end(string: &str, max_bytes: usize) -> String {
	let buf = string.as_bytes();
	if buf.len() <= max_bytes {
		return string.to_string();
	}

	// Start from the end, skip maxBytes back.
	let mut start = buf.len() - max_bytes;

	// Find a valid UTF-8 boundary (start of a character).
	while start < buf.len() && (buf[start] & 0xc0) == 0x80 {
		start += 1;
	}

	String::from_utf8_lossy(&buf[start..]).into_owned()
}

/// Result of [`truncate_line`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncatedLine {
	pub text: String,
	pub was_truncated: bool,
}

/// Truncate a single line to max chars, adding a `[truncated]` suffix
/// (`truncateLine`). Used for grep match lines. Char counts mirror upstream
/// UTF-16 `.length` per crate docs D23 (differs only for astral text).
pub fn truncate_line(line: &str, max_chars: usize) -> TruncatedLine {
	if line.chars().count() <= max_chars {
		return TruncatedLine {
			text: line.to_string(),
			was_truncated: false,
		};
	}
	TruncatedLine {
		text: format!("{}... [truncated]", line.chars().take(max_chars).collect::<String>()),
		was_truncated: true,
	}
}

/// Result of [`truncate_middle`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiddleTruncationResult {
	/// The start and end of the content with a `…N chars truncated…` marker
	/// between them.
	pub content: String,
	pub truncated: bool,
	/// Characters left out.
	pub removed_chars: usize,
	pub total_bytes: usize,
	pub total_lines: usize,
}

/// Keep the start and the end of `content`, half of `maxBytes` each, and
/// replace the middle with a `…N chars truncated…` marker, like Codex does
/// for tool output (`truncateMiddle`). Cuts only at character boundaries.
pub fn truncate_middle(content: &str, max_bytes: usize) -> MiddleTruncationResult {
	let buf = content.as_bytes();
	let total_lines = split_lines_for_counting(content).len();
	if buf.len() <= max_bytes {
		return MiddleTruncationResult {
			content: content.to_string(),
			truncated: false,
			removed_chars: 0,
			total_bytes: buf.len(),
			total_lines,
		};
	}
	// Continuation bytes (10xxxxxx) are not character starts.
	let is_boundary = |index: usize| index >= buf.len() || (buf[index] & 0xc0) != 0x80;
	let mut head_end = max_bytes / 2;
	while head_end > 0 && !is_boundary(head_end) {
		head_end -= 1;
	}
	let mut tail_start = buf.len() - (max_bytes - max_bytes / 2);
	while tail_start < buf.len() && !is_boundary(tail_start) {
		tail_start += 1;
	}
	let head = String::from_utf8_lossy(&buf[..head_end]);
	let tail = String::from_utf8_lossy(&buf[tail_start..]);
	let removed_chars = String::from_utf8_lossy(&buf[head_end..tail_start]).chars().count();
	MiddleTruncationResult {
		content: format!("{head}…{removed_chars} chars truncated…{tail}"),
		truncated: true,
		removed_chars,
		total_bytes: buf.len(),
		total_lines,
	}
}
