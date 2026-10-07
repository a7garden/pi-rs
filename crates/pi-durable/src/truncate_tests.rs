//! Port of `test/env-truncate.test.ts`.
//!
//! Wiring note: `lib.rs` must declare `#[cfg(test)] mod truncate_tests;` (the module is
//! written for the integrator; `truncate.rs` itself is owned by another slice).
//!
//! Deviations:
//! - D1: the "runtime without Buffer" test spawns a Node child process to compute UTF-8 byte
//!   lengths without `Buffer`; Rust's `&str` is always UTF-8 with `len()` as the byte count,
//!   so the invariant is asserted directly (same inputs, same expected lengths and head).

use crate::truncate::{format_size, truncate_head, TruncationLimit, TruncationOptions};

fn options(max_lines: usize, max_bytes: usize) -> TruncationOptions {
	TruncationOptions { max_lines: Some(max_lines), max_bytes: Some(max_bytes) }
}

#[test]
fn reports_utf8_byte_counts_in_truncation_results() {
	let content = "aé🙂\nb";
	let result = truncate_head(content, options(10, 100));

	assert!(!result.truncated);
	assert_eq!(result.total_bytes, content.len());
	assert_eq!(result.output_bytes, content.len());
	assert_eq!(result.total_bytes, 9);
}

#[test]
fn counts_utf8_bytes_and_truncates_correctly() {
	// D1: the inputs of the upstream no-Buffer fixture, with its expected lengths and head.
	let inputs = ["", "ascii", "é", "中", "🙂", "\u{FFFD}", "aé🙂b", "߿ࠀ￿"];
	let lengths: Vec<usize> = inputs.iter().map(|input| input.len()).collect();
	assert_eq!(lengths, vec![0, 5, 2, 3, 4, 3, 8, 8]);

	let result = truncate_head("aé🙂\nb", options(10, 7));
	assert_eq!(result.content, "aé🙂");
	assert_eq!(result.output_bytes, 7);
	assert_eq!(result.truncated_by, Some(TruncationLimit::Bytes));
}

#[test]
fn does_not_count_a_trailing_newline_as_an_extra_line() {
	let content = "line\nline\nline\n";
	let head = truncate_head(content, options(3, 100));

	assert!(!head.truncated);
	assert_eq!(head.total_lines, 3);
	assert_eq!(head.output_lines, 3);
}

#[test]
fn truncates_head_by_line_limits() {
	let content = "one\ntwo\nthree\nfour";
	let result = truncate_head(content, options(2, 100));
	assert_eq!(result.content, "one\ntwo");
	assert!(result.truncated);
	assert_eq!(result.truncated_by, Some(TruncationLimit::Lines));
	assert_eq!(result.total_lines, 4);
	assert_eq!(result.output_lines, 2);
}

#[test]
fn reports_bytes_when_only_a_trailing_newline_exceeds_limits_at_the_line_cap() {
	let result = truncate_head("hello\nworld\n", options(2, 11));
	assert!(result.truncated);
	assert_eq!(result.truncated_by, Some(TruncationLimit::Bytes));
	assert_eq!(result.total_lines, 2);
	assert_eq!(result.output_lines, 2);
}

#[test]
fn truncates_head_on_utf8_byte_limits_without_partial_lines() {
	let content = "éé\nabc";
	let result = truncate_head(content, options(10, 4));

	assert_eq!(result.content, "éé");
	assert!(result.truncated);
	assert_eq!(result.truncated_by, Some(TruncationLimit::Bytes));
	assert_eq!(result.output_bytes, 4);
	assert!(!result.first_line_exceeds_limit);
}

#[test]
fn reports_head_truncation_when_the_first_line_exceeds_the_byte_limit() {
	let result = truncate_head("éé\nabc", options(10, 3));

	assert_eq!(result.content, "");
	assert!(result.truncated);
	assert_eq!(result.truncated_by, Some(TruncationLimit::Bytes));
	assert!(result.first_line_exceeds_limit);
}

#[test]
fn formats_sizes() {
	assert_eq!(format_size(1023), "1023B");
	assert_eq!(format_size(1536), "1.5KB");
	assert_eq!(format_size(3 * 1024 * 1024), "3.0MB");
}
