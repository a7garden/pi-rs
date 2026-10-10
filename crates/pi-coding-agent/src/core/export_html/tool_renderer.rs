//! Tool HTML renderer for custom tools in HTML export.
//!
//! Port of upstream `packages/coding-agent/src/core/export-html/tool-renderer.ts`
//! (pure parts).
//!
//! # Deviations
//!
//! - **D81 — `createToolHtmlRenderer` component pipeline is deferred.**
//!   Upstream builds the renderer by invoking erased `ToolDefinition`
//!   `renderCall`/`renderResult` TUI renderers (D27/D32: the renderer layer
//!   and its `pi-tui` `Component` outputs were deferred with the excluded
//!   interactive mode) and converting the resulting ANSI lines to HTML via
//!   [`ansi_lines_to_html`]. Until that seam lands, custom-tool export
//!   falls back to structured result rendering (upstream behavior when a
//!   renderer returns `undefined`/throws); supply a custom
//!   [`ToolHtmlRenderer`](super::ToolHtmlRenderer) implementation to
//!   pre-render. The pure text helpers upstream uses on the rendered output —
//!   [`is_blank_rendered_line`] and [`trim_rendered_result_lines`] — are
//!   ported here so the whitespace contract stays testable.

/// Strip every `ESC[<digits/semicolons>m` sequence (upstream
/// `ANSI_ESCAPE_REGEX = /\x1b\[[\d;]*m/g` with `replace(_, "")`).
pub fn strip_sgr_sequences(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            let mut j = i + 2;
            while j < bytes.len() && (bytes[j].is_ascii_digit() || bytes[j] == b';') {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'm' {
                i = j + 1;
                continue;
            }
        }
        let ch = text[i..].chars().next().expect("index into char boundary");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Upstream `isBlankRenderedLine`: blank after removing SGR sequences.
pub fn is_blank_rendered_line(line: &str) -> bool {
    strip_sgr_sequences(line).trim().is_empty()
}

/// Upstream `trimRenderedResultLines`: drop blank rendered lines from both ends.
pub fn trim_rendered_result_lines(lines: &[String]) -> Vec<String> {
    let mut start = 0;
    let mut end = lines.len();
    while start < end && is_blank_rendered_line(&lines[start]) {
        start += 1;
    }
    while end > start && is_blank_rendered_line(&lines[end - 1]) {
        end -= 1;
    }
    lines[start..end].to_vec()
}
