//! Port of `packages/coding-agent/src/core/tools/render-utils.ts` — helpers
//! shared by the built-in tool renderers.
//!
//! Only the theme-free pure text helpers are ported (D32): the remaining
//! functions (`linkPath`, `formatToolCallWithArgs`, `invalidArgText`,
//! `renderToolPath`, `getTextOutput`) and the per-tool renderers under
//! `core/tools/renderers/` style text through the interactive `Theme` and
//! pi-tui components, which are excluded by the TUI port decision. They land
//! with the theme layer; `ToolDefinition::render_call`/`render_result`
//! stay `None` for the built-in tools.

/// `shortenPath`: collapse the home directory prefix to `~`.
pub fn shorten_path(path: &str) -> String {
	let home = crate::utils::paths::home_dir();
	if let Some(rest) = path.strip_prefix(&home) {
		return format!("~{rest}");
	}
	path.to_string()
}

/// `str`: pass strings through; `null`/`undefined` become `""`; anything else
/// is not renderable.
pub fn str(value: Option<&str>) -> Option<String> {
	value.map(str::to_string)
}

/// `replaceTabs`: expand tabs to three spaces for stable display alignment.
pub fn replace_tabs(text: &str) -> String {
	text.replace('\t', "   ")
}

/// `normalizeDisplayText`: strip carriage returns.
pub fn normalize_display_text(text: &str) -> String {
	text.replace('\r', "")
}
