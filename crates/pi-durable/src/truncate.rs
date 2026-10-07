//! Port of `packages/durable/src/truncate.ts` (verbatim logic).
//!
//! Truncation is based on two independent limits — whichever is hit first wins:
//! - Line limit (default: 2000 lines)
//! - Byte limit (default: 50KB)
//!
//! Never returns partial lines. Tool output streams are bounded by
//! `crate::harness::output` instead. (D1: `utf8ByteLength` collapses to
//! `str::len` — Rust strings are already UTF-8; JS's surrogate counting is a
//! UTF-16 artifact with identical byte results.)

use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_LINES: usize = 2000;
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024; // 50KB

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TruncationResult {
	/// The truncated content.
	pub content: String,
	/// Whether truncation occurred.
	pub truncated: bool,
	/// Which limit was hit: `"lines"`, `"bytes"`, or `None` if not truncated.
	pub truncated_by: Option<TruncationLimit>,
	/// Total number of lines in the original content.
	pub total_lines: usize,
	/// Total number of bytes in the original content.
	pub total_bytes: usize,
	/// Number of complete lines in the truncated output.
	pub output_lines: usize,
	/// Number of bytes in the truncated output.
	pub output_bytes: usize,
	/// Whether the last line was partially truncated (only for tail truncation edge case).
	pub last_line_partial: bool,
	/// Whether the first line exceeded the byte limit (for head truncation).
	pub first_line_exceeds_limit: bool,
	/// The max lines limit that was applied.
	pub max_lines: usize,
	/// The max bytes limit that was applied.
	pub max_bytes: usize,
}

/// Which independent limit fired first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TruncationLimit {
	Lines,
	Bytes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncationOptions {
	/// Maximum number of lines (default: 2000).
	pub max_lines: Option<usize>,
	/// Maximum number of bytes (default: 50KB).
	pub max_bytes: Option<usize>,
}

impl Default for TruncationOptions {
	fn default() -> Self {
		Self { max_lines: None, max_bytes: None }
	}
}

impl TruncationOptions {
	pub fn with_max_lines(max_lines: usize) -> Self {
		Self { max_lines: Some(max_lines), max_bytes: None }
	}
	pub fn with_max_bytes(max_bytes: usize) -> Self {
		Self { max_lines: None, max_bytes: Some(max_bytes) }
	}

	fn max_lines(&self) -> usize {
		self.max_lines.unwrap_or(DEFAULT_MAX_LINES)
	}
	fn max_bytes(&self) -> usize {
		self.max_bytes.unwrap_or(DEFAULT_MAX_BYTES)
	}
}

/// UTF-8 byte length of `content` (see module note D1).
pub fn utf8_byte_length(content: &str) -> usize {
	content.len()
}

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

/// Format bytes as human-readable size.
pub fn format_size(bytes: usize) -> String {
	if bytes < 1024 {
		format!("{bytes}B")
	} else if bytes < 1024 * 1024 {
		format!("{:.1}KB", bytes as f64 / 1024.0)
	} else {
		format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
	}
}

/// Truncate content from the head (keep first N lines/bytes).
/// Suitable for file reads where you want to see the beginning.
///
/// Never returns partial lines. If first line exceeds byte limit,
/// returns empty content with firstLineExceedsLimit=true.
pub fn truncate_head(content: &str, options: TruncationOptions) -> TruncationResult {
	truncate_head_of(
		content,
		Totals { lines: split_lines_for_counting(content).len(), bytes: utf8_byte_length(content) },
		options,
	)
}

/// `truncate_head` of a text known by a prefix and its totals (`lines` counted like
/// `truncate_head`, ignoring a trailing newline). The prefix must be the whole text, or longer
/// than `max_bytes + 1` UTF-8 bytes, or hold at least `max_lines` newlines; then the result
/// equals `truncate_head` of the whole text.
pub fn truncate_head_of(
	prefix: &str,
	totals: Totals,
	options: TruncationOptions,
) -> TruncationResult {
	let max_lines = options.max_lines();
	let max_bytes = options.max_bytes();

	let total_bytes = totals.bytes;
	let lines = split_lines_for_counting(prefix);
	let total_lines = totals.lines;

	// Check if no truncation needed
	if total_lines <= max_lines && total_bytes <= max_bytes {
		return TruncationResult {
			content: prefix.to_string(),
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

	// Check if first line alone exceeds byte limit
	let first_line_bytes = utf8_byte_length(lines[0]);
	if first_line_bytes > max_bytes {
		return TruncationResult {
			content: String::new(),
			truncated: true,
			truncated_by: Some(TruncationLimit::Bytes),
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

	// Collect complete lines that fit
	let mut output_lines_arr: Vec<&str> = Vec::new();
	let mut output_bytes_count = 0usize;
	let mut truncated_by = TruncationLimit::Lines;

	for (i, line) in lines.iter().enumerate().take(max_lines) {
		let line_bytes = utf8_byte_length(line) + if i > 0 { 1 } else { 0 }; // +1 for newline

		if output_bytes_count + line_bytes > max_bytes {
			truncated_by = TruncationLimit::Bytes;
			break;
		}

		output_lines_arr.push(line);
		output_bytes_count += line_bytes;
	}

	// Without a byte break, only omitted lines prove the line limit was reached; otherwise a
	// trailing newline exceeded bytes.
	if truncated_by != TruncationLimit::Bytes && output_lines_arr.len() < total_lines {
		truncated_by = TruncationLimit::Lines;
	} else if truncated_by != TruncationLimit::Bytes {
		truncated_by = TruncationLimit::Bytes;
	}

	let output_content = output_lines_arr.join("\n");
	let final_output_bytes = utf8_byte_length(&output_content);

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

/// Known prefix totals for [`truncate_head_of`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Totals {
	pub lines: usize,
	pub bytes: usize,
}
