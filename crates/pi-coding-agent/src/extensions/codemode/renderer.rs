//! Port of `packages/coding-agent/src/extensions/codemode/renderer.ts` —
//! presentation for the codemode tool: nested-call rows with status,
//! duration, and model-call cost, plus the script output without the
//! "Script completed" header.
//!
//! # Deviation D92 — theme/TUI surface deferred
//!
//! The component assembly (`renderCall`/`renderResult` over pi-tui
//! `Container`/`Text`/`Spacer`, theme colors, `VisualLinePreview` wrapping,
//! `highlightCode`, and `keyHint` styling) is excluded by the crate-wide TUI
//! port decision (same scope as `core/tools/render_utils.rs` D32): the
//! definition's `render_call`/`render_result` stay `None`, and the theme-free
//! string surface is ported here so the renderer behavior is pinned and the
//! theme layer can re-assemble the components verbatim. The upstream renderer
//! tests are ported at the string level; the visual-wrap assertions of the
//! fourth test need pi-tui and are deferred (see the tests module).

use pi_agent::AgentToolResult;
use pi_ai::types::UserContent;

use crate::extensions::codemode::tool::{CodemodeNestedCall, CodemodeNestedCallStatus, CodemodeToolDetails};

const CODE_PREVIEW_LINES: usize = 10;
const CALL_PREVIEW_COUNT: usize = 8;
const OUTPUT_PREVIEW_LINES: usize = 5;
const COLLAPSED_ARGS_CHARS: usize = 80;

/// Expand hint of a collapsed preview (`expandHint`); the upstream
/// `keyHint("app.tools.expand", ...)` renders the user's keybinding, the
/// port carries the fallback verb (D92).
pub fn expand_hint(hidden: usize, noun: &str) -> String {
	format!("... ({hidden} more {noun}, to expand)")
}

/// `formatDuration`.
pub fn format_duration(ms: Option<f64>) -> String {
	let Some(ms) = ms else { return String::new() };
	if ms < 1000.0 {
		format!("{}ms", ms.round())
	} else {
		format!("{:.1}s", ms / 1000.0)
	}
}

/// `toPrecision(2)` for non-negative values (upstream `Number.prototype.toPrecision`).
fn to_precision(value: f64, digits: usize) -> String {
	if value == 0.0 {
		return format!("0.{}", "0".repeat(digits.saturating_sub(1)));
	}
	let decimals = digits.saturating_sub(1);
	let mut exponent = value.abs().log10().floor() as i32;
	let scaled = value.abs() / 10f64.powi(exponent);
	let mut significand = format!("{scaled:.decimals$}");
	// Rounding may carry into the next decade (`9.99.toPrecision(2)` is "10").
	if significand.parse::<f64>().map(|parsed| parsed >= 10.0).unwrap_or(false) {
		exponent += 1;
		significand = format!("{:.decimals$}", value.abs() / 10f64.powi(exponent));
	}
	expand_positional(&significand, exponent)
}

/// Expand a `d.ddd` significand with a decimal exponent into positional
/// notation, keeping trailing significant zeros like `toPrecision`.
fn expand_positional(significand: &str, exponent: i32) -> String {
	let negative = significand.starts_with('-');
	let digits: String = significand.chars().filter(|c| c.is_ascii_digit()).collect();
	let point = 1i32 + exponent;
	let mut out = String::new();
	if negative {
		out.push('-');
	}
	if point <= 0 {
		out.push_str("0.");
		for _ in 0..(-point) {
			out.push('0');
		}
		out.push_str(&digits);
	} else if point as usize >= digits.len() {
		out.push_str(&digits);
		for _ in 0..(point as usize - digits.len()) {
			out.push('0');
		}
	} else {
		out.push_str(&digits[..point as usize]);
		out.push('.');
		out.push_str(&digits[point as usize..]);
	}
	out
}

/// Cents for larger amounts, two significant digits for the fractions of a
/// cent classifier calls cost (`formatCost`).
pub fn format_cost(cost: f64) -> String {
	if cost >= 0.01 {
		format!("${cost:.2}")
	} else {
		format!("${}", to_precision(cost, 2))
	}
}

/// Status glyph of a nested call (`statusIcon`); the theme colors
/// (warning/success/error/muted) are deferred with the theme layer (D92).
pub fn status_glyph(status: CodemodeNestedCallStatus) -> &'static str {
	match status {
		CodemodeNestedCallStatus::Running => "…",
		CodemodeNestedCallStatus::Ok => "✓",
		CodemodeNestedCallStatus::Error => "✗",
		CodemodeNestedCallStatus::Cancelled => "⊘",
	}
}

/// One line per nested call (`formatCall`).
pub fn format_call(call: &CodemodeNestedCall, expanded: bool) -> String {
	let args = if !expanded && call.args.chars().count() > COLLAPSED_ARGS_CHARS {
		format!("{}...", call.args.chars().take(COLLAPSED_ARGS_CHARS - 3).collect::<String>())
	} else {
		call.args.clone()
	};
	let duration = format_duration(call.duration_ms);
	let mut line = format!("{} {}", status_glyph(call.status), call.name);
	if !args.is_empty() {
		line.push(' ');
		line.push_str(&args);
	}
	if !duration.is_empty() {
		line.push(' ');
		line.push_str(&duration);
	}
	// Upstream `if (call.cost)`: a zero cost is skipped.
	if let Some(cost) = call.cost.filter(|cost| *cost != 0.0) {
		line.push(' ');
		line.push_str(&format_cost(cost));
	}
	if expanded {
		if let Some(error) = &call.error {
			line.push_str("\n    ");
			line.push_str(&error.split('\n').collect::<Vec<_>>().join("\n    "));
		}
	}
	line
}

/// The nested-call rows of a result (`renderResult`'s calls block): the last
/// [`CALL_PREVIEW_COUNT`] calls when collapsed, an earlier-calls hint, and
/// the model-call total when more than one call reported cost.
pub fn format_calls_lines(calls: &[CodemodeNestedCall], expanded: bool) -> Vec<String> {
	if calls.is_empty() {
		return Vec::new();
	}
	let shown: Vec<&CodemodeNestedCall> = if expanded {
		calls.iter().collect()
	} else {
		calls.iter().rev().take(CALL_PREVIEW_COUNT).rev().collect()
	};
	let mut lines: Vec<String> = shown.iter().map(|call| format_call(call, expanded)).collect();
	if shown.len() < calls.len() {
		// Collapsed rows hide earlier calls, so the total covers every call.
		lines.insert(0, expand_hint(calls.len() - shown.len(), "earlier calls"));
	}
	let priced: Vec<f64> = calls.iter().filter_map(|call| call.cost).collect();
	if priced.len() > 1 {
		let total: f64 = priced.iter().sum();
		lines.push(format!("Model calls: {}", format_cost(total)));
	}
	lines
}

/// Whether `text` is the "Script completed/failed … Output:" header the tool
/// prepends (`SCRIPT_HEADER`).
pub fn is_script_header(text: &str) -> bool {
	let mut lines = text.split('\n');
	let (Some(first), Some(second), Some(third), Some(fourth), None) =
		(lines.next(), lines.next(), lines.next(), lines.next(), lines.next())
	else {
		return false;
	};
	if !(first == "Script completed" || first == "Script failed") {
		return false;
	}
	if !(second.starts_with("Wall time ") && second.ends_with(" seconds")) {
		return false;
	}
	let wall_time = &second["Wall time ".len()..second.len() - " seconds".len()];
	if wall_time.is_empty() || !wall_time.chars().all(|c| c.is_ascii_digit() || c == '.') {
		return false;
	}
	third == "Output:" && fourth.is_empty()
}

/// The script output of a result, without the "Script completed…Output:"
/// header. Rejected input (invalid options) has no header. Partial results
/// render nothing (`renderResult`'s output half; the theme styling and image
/// placeholders of upstream `getTextOutput` are deferred, D92).
pub fn result_output_text(result: &AgentToolResult, is_partial: bool) -> String {
	if is_partial {
		return String::new();
	}
	let content: &[UserContent] = &result.content;
	let has_header = match content.first() {
		Some(UserContent::Text(text)) => is_script_header(&text.text),
		_ => false,
	};
	let rest = if has_header { &content[1..] } else { content };
	let text = rest
		.iter()
		.filter_map(|block| match block {
			UserContent::Text(text) => Some(text.text.as_str()),
			_ => None,
		})
		.collect::<Vec<&str>>()
		.join("\n");
	text.trim().to_string()
}

/// Collapsed output footnote naming the spill file (`renderResult`'s
/// `Full output:` row).
pub fn full_output_note(details: &CodemodeToolDetails) -> Option<String> {
	details.full_output_path.as_ref().map(|path| format!("Full output: {path}"))
}

/// Code preview line budget of the call view (`CODE_PREVIEW_LINES`).
pub const CODE_PREVIEW_LINES_LIMIT: usize = CODE_PREVIEW_LINES;
/// Output preview line budget of the collapsed result view
/// (`OUTPUT_PREVIEW_LINES`).
pub const OUTPUT_PREVIEW_LINES_LIMIT: usize = OUTPUT_PREVIEW_LINES;

// Port of `packages/coding-agent/test/codemode-renderer.test.ts`, asserted on
// the theme-free string surface (D92). The fourth upstream case additionally
// pins pi-tui's visual (wrapped) line truncation at width 50; those line-count
// assertions are deferred with the theme layer — the portable half (the
// `Full output:` footnote of the collapsed preview) is asserted here.
#[cfg(test)]
mod renderer_tests {
	use super::*;
	use crate::extensions::codemode::execute::text_item;
	use serde_json::json;

	fn call(
		id: &str,
		name: &str,
		args: &str,
		status: CodemodeNestedCallStatus,
		duration_ms: Option<f64>,
		cost: Option<f64>,
	) -> CodemodeNestedCall {
		CodemodeNestedCall {
			id: id.to_string(),
			name: name.to_string(),
			args: args.to_string(),
			status,
			duration_ms,
			error: None,
			cost,
		}
	}

	fn result(content: Vec<UserContent>, calls: Vec<CodemodeNestedCall>) -> AgentToolResult {
		let details = CodemodeToolDetails { calls, full_output_path: None };
		AgentToolResult::new(content, serde_json::to_value(&details).unwrap_or(json!({})))
	}

	fn header_text() -> String {
		"Script completed\nWall time 0.1 seconds\nOutput:\n".to_string()
	}

	/// The rendered text: nested-call lines, then the output (the upstream
	/// Container + Spacer layout). The output block is only added when the
	/// trimmed output is non-empty (`if (output)` upstream).
	fn render_text(result: &AgentToolResult, calls: &[CodemodeNestedCall], expanded: bool) -> String {
		let mut lines = format_calls_lines(calls, expanded);
		let output = result_output_text(result, false);
		if !output.is_empty() {
			lines.push(String::new());
			lines.push(output);
		}
		lines.join("\n")
	}

	#[test]
	fn hides_the_script_header_and_shows_the_output() {
		let calls = vec![call("call/1", "read", r#"{"path":"a"}"#, CodemodeNestedCallStatus::Ok, Some(5.0), None)];
		let result = result(
			vec![text_item(header_text()), text_item("hello")],
			calls.clone(),
		);
		assert_eq!(render_text(&result, &calls, true), "✓ read {\"path\":\"a\"} 5ms\n\nhello");
	}

	#[test]
	fn shows_the_cost_of_model_calls_and_their_total() {
		let calls = vec![
			call(
				"call/models.classify/1",
				"models.classify",
				"scorer/judge",
				CodemodeNestedCallStatus::Ok,
				Some(5.0),
				Some(0.000012936),
			),
			call(
				"call/models.classify/2",
				"models.classify",
				"scorer/judge",
				CodemodeNestedCallStatus::Ok,
				Some(5.0),
				Some(0.02),
			),
			call(
				"call/models.classify/3",
				"models.classify",
				"scorer/judge",
				CodemodeNestedCallStatus::Ok,
				Some(5.0),
				None,
			),
		];
		let result = result(vec![text_item(header_text())], calls.clone());
		assert_eq!(
			render_text(&result, &calls, true),
			[
				"✓ models.classify scorer/judge 5ms $0.000013",
				"✓ models.classify scorer/judge 5ms $0.02",
				"✓ models.classify scorer/judge 5ms",
				"Model calls: $0.02",
			]
			.join("\n")
		);
	}

	#[test]
	fn shows_results_without_a_header_such_as_rejected_options() {
		let result = result(
			vec![text_item("The @options line must be followed by JavaScript source")],
			Vec::new(),
		);
		assert_eq!(
			result_output_text(&result, false),
			"The @options line must be followed by JavaScript source"
		);
	}

	#[test]
	fn collapsed_preview_names_the_full_output_file() {
		let details = CodemodeToolDetails {
			calls: Vec::new(),
			full_output_path: Some("/tmp/out.txt".to_string()),
		};
		assert_eq!(full_output_note(&details), Some("Full output: /tmp/out.txt".to_string()));
		// Deferred (D92): the upstream case also asserts that the collapsed
		// preview wraps to exactly 7 visual lines at width 50 — five of
		// "xxxx…" plus the expand hint — which needs pi-tui's wrapping.
	}
}

