//! Port of upstream `packages/coding-agent/src/core/export-html/index.ts` —
//! session → standalone HTML export.
//!
//! Upstream: `index.ts` plus the template assets (`template.html/css/js`,
//! `vendor/*.min.js`) embedded verbatim with `include_str!` (D80) instead of
//! read from a template directory at runtime.

pub mod ansi_to_html;
pub mod tool_renderer;

#[cfg(test)]
mod tests;

pub use ansi_to_html::{ansi_lines_to_html, ansi_to_html};

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use crate::config::APP_NAME;
use crate::core::agent_session::AgentStateSnapshot;
use crate::core::session_manager::{SessionEntry, SessionHeader, SessionManager};
use crate::theme::theme_json::{validate_theme_json, ThemeColorValue, ValidatedThemeJson};
use crate::theme::{get_available_themes_with_paths, get_theme_by_name};
use crate::utils::paths::{normalize_path, resolve_path, PathInputOptions};
use crate::utils::tui_compat::{color_to_hex, parse_color};

use system_theme_name::SYSTEM_THEME_NAME;

// ============================================================================
// Template assets (upstream: getExportTemplateDir() + fs.readFileSync)
// ============================================================================

const TEMPLATE_HTML: &str = include_str!("template.html");
const TEMPLATE_CSS: &str = include_str!("template.css");
const TEMPLATE_JS: &str = include_str!("template.js");
const MARKED_JS: &str = include_str!("vendor/marked.min.js");
const HIGHLIGHT_JS: &str = include_str!("vendor/highlight.min.js");

/// `SYSTEM_THEME_NAME` lives in a private submodule of [`crate::theme`]; the
/// public re-export is `SYSTEM_THEME`.
mod system_theme_name {
    pub use crate::theme::SYSTEM_THEME as SYSTEM_THEME_NAME;
}

// ============================================================================
/// Interface for rendering custom tools to HTML (upstream `ToolHtmlRenderer`
/// in `index.ts`; the concrete TUI-component implementation in
/// `tool-renderer.ts` is [`tool_renderer`]).
///
/// Used by agent-session to pre-render extension tool output.
pub trait ToolHtmlRenderer {
    /// Render a tool call to HTML. Returns None if tool has no custom renderer.
    fn render_call(&self, tool_call_id: &str, tool_name: &str, args: &Value) -> Option<String>;
    /// Render a tool result to collapsed/expanded HTML. Returns None if tool has
    /// no custom renderer.
    fn render_result(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        result: &[pi_ai::types::UserContent],
        details: Option<&Value>,
        is_error: bool,
    ) -> Option<RenderedToolResult>;
}

/// Collapsed/expanded HTML pair (upstream `{ collapsed?, expanded? }`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RenderedToolResult {
    pub collapsed: Option<String>,
    pub expanded: Option<String>,
}

/// Pre-rendered HTML for a custom tool call and result (upstream
/// `RenderedToolHtml`).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct RenderedToolHtml {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_html: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_html_collapsed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_html_expanded: Option<String>,
}

/// Export options (upstream `ExportOptions`).
#[derive(Clone, Default)]
pub struct ExportOptions {
    pub output_path: Option<String>,
    pub theme_name: Option<String>,
    /// Optional tool renderer for custom tools.
    pub tool_renderer: Option<std::sync::Arc<dyn ToolHtmlRenderer + Send + Sync>>,
}

impl std::fmt::Debug for ExportOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportOptions")
            .field("output_path", &self.output_path)
            .field("theme_name", &self.theme_name)
            .field("tool_renderer", &self.tool_renderer.is_some())
            .finish()
    }
}

// ============================================================================
// Color helpers (upstream private functions in index.ts)
// ============================================================================

/// Parsed RGB triple.
#[derive(Debug, Clone, Copy)]
struct Rgb {
    r: u32,
    g: u32,
    b: u32,
}

/// Parse a color string to RGB values. Supports hex (#RRGGBB) and rgb(r,g,b) formats.
fn parse_color_string(color: &str) -> Option<Rgb> {
    let color = color.trim();
    // /^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$/
    if let Some(hex) = color.strip_prefix('#') {
        if hex.len() == 6 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Some(Rgb {
                r: u32::from_str_radix(&hex[0..2], 16).ok()?,
                g: u32::from_str_radix(&hex[2..4], 16).ok()?,
                b: u32::from_str_radix(&hex[4..6], 16).ok()?,
            });
        }
        return None;
    }
    // /^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/
    let inner = color.strip_prefix("rgb")?.trim();
    let inner = inner.strip_prefix('(')?.strip_suffix(')')?;
    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() != 3 {
        return None;
    }
    let parse = |s: &str| -> Option<u32> {
        let s = s.trim();
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        s.parse::<u32>().ok()
    };
    Some(Rgb {
        r: parse(parts[0])?,
        g: parse(parts[1])?,
        b: parse(parts[2])?,
    })
}

/// Calculate relative luminance of a color (0-1, higher = lighter).
fn get_luminance(r: u32, g: u32, b: u32) -> f64 {
    let to_linear = |c: u32| {
        let s = c as f64 / 255.0;
        if s <= 0.03928 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * to_linear(r) + 0.7152 * to_linear(g) + 0.0722 * to_linear(b)
}

/// Adjust color brightness. Factor > 1 lightens, < 1 darkens.
fn adjust_brightness(color: &str, factor: f64) -> String {
    let Some(parsed) = parse_color_string(color) else {
        return color.to_string();
    };
    let adjust = |c: u32| ((c as f64 * factor).round() as i64).clamp(0, 255) as u32;
    format!(
        "rgb({}, {}, {})",
        adjust(parsed.r),
        adjust(parsed.g),
        adjust(parsed.b)
    )
}

/// Derived export background colors (upstream `deriveExportColors` return).
#[derive(Debug, Clone, PartialEq)]
struct DerivedExportColors {
    page_bg: String,
    card_bg: String,
    info_bg: String,
}

/// Derive export background colors from a base color (e.g., userMessageBg).
fn derive_export_colors(base_color: &str) -> DerivedExportColors {
    let Some(parsed) = parse_color_string(base_color) else {
        return DerivedExportColors {
            page_bg: "rgb(24, 24, 30)".to_string(),
            card_bg: "rgb(30, 30, 36)".to_string(),
            info_bg: "rgb(60, 55, 40)".to_string(),
        };
    };

    let luminance = get_luminance(parsed.r, parsed.g, parsed.b);
    let is_light = luminance > 0.5;
    let min255 = |v: u32| v.min(255);
    let clamp0 = |v: i64| v.max(0) as u32;

    if is_light {
        DerivedExportColors {
            page_bg: adjust_brightness(base_color, 0.96),
            card_bg: base_color.to_string(),
            info_bg: format!(
                "rgb({}, {}, {})",
                min255(parsed.r + 10),
                min255(parsed.g + 5),
                clamp0(parsed.b as i64 - 20)
            ),
        }
    } else {
        DerivedExportColors {
            page_bg: adjust_brightness(base_color, 0.7),
            card_bg: adjust_brightness(base_color, 0.85),
            info_bg: format!(
                "rgb({}, {}, {})",
                min255(parsed.r + 20),
                min255(parsed.g + 15),
                parsed.b
            ),
        }
    }
}

// ============================================================================
// Theme helpers (upstream theme.ts HTML-export helpers, deferred there as
// D56d — landed here because only export-html consumes them; D80)
// ============================================================================

/// Upstream `getResolvedThemeColors`: every token of the resolved theme as a
/// hex string. D80: token order follows the `BTreeMap` (alphabetical) rather
/// than upstream's JS object insertion order — cosmetic in generated CSS.
pub fn get_resolved_theme_colors(theme_name: Option<&str>) -> BTreeMap<String, String> {
    let name = theme_name.unwrap_or(SYSTEM_THEME_NAME);
    let theme = get_theme_by_name(name)
        .unwrap_or_else(|| get_theme_by_name(SYSTEM_THEME_NAME).expect("system theme is built in"));
    theme
        .colors()
        .into_iter()
        .map(|(token, color)| (token, color_to_hex(&color)))
        .collect()
}

/// Upstream `isLightTheme`.
pub fn is_light_theme(theme_name: Option<&str>) -> bool {
    let name = theme_name.unwrap_or(SYSTEM_THEME_NAME);
    let theme = get_theme_by_name(name)
        .unwrap_or_else(|| get_theme_by_name(SYSTEM_THEME_NAME).expect("system theme is built in"));
    theme.appearance() == "light"
}

/// Upstream `getThemeExportColors`: explicit `export:` colors from the theme
/// JSON, if specified. `None` per color that isn't set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ThemeExportColors {
    pub page_bg: Option<String>,
    pub card_bg: Option<String>,
    pub info_bg: Option<String>,
}

/// Upstream private `resolveVarRefs` (theme.ts): chase `{var}` style string
/// values through `vars` until a self-contained color form is reached.
fn resolve_var_refs(
    value: &ThemeColorValue,
    vars: &BTreeMap<String, ThemeColorValue>,
    visited: &mut std::collections::BTreeSet<String>,
) -> Result<String, String> {
    let ThemeColorValue::Str(value) = value else {
        // Numbers (palette indices) are handled by the caller.
        return Ok(value.as_display());
    };
    if value.is_empty()
        || value.starts_with('#')
        || value.to_ascii_lowercase().starts_with("oklch(")
        || value.to_ascii_lowercase().starts_with("okhsl(")
    {
        return Ok(value.clone());
    }
    if visited.contains(value) {
        return Err(format!("Circular variable reference detected: {value}"));
    }
    let Some(next) = vars.get(value) else {
        return Err(format!("Variable reference not found: {value}"));
    };
    visited.insert(value.clone());
    resolve_var_refs(next, vars, visited)
}

/// Load and validate the JSON document of a theme by name (upstream private
/// `loadThemeJson`): builtin dark/light are embedded (D56c), registered themes
/// are read from their source path.
fn load_theme_json_for_export(name: &str) -> Result<ValidatedThemeJson, String> {
    const DARK: &str = include_str!("../../theme/dark.json");
    const LIGHT: &str = include_str!("../../theme/light.json");
    let from_str = |label: &str, content: &str| -> Result<ValidatedThemeJson, String> {
        let json: Value =
            serde_json::from_str(content).map_err(|error| format!("{label}: {error}"))?;
        validate_theme_json(label, &json)
    };
    match name {
        "dark" => from_str("dark.json", DARK),
        "light" => from_str("light.json", LIGHT),
        other => {
            let info = get_available_themes_with_paths()
                .into_iter()
                .find(|info| info.name == other);
            let Some(path) = info.and_then(|info| info.path) else {
                return Err(format!(
                    "Theme \"{other}\" does not have a source path for export"
                ));
            };
            let content = std::fs::read_to_string(&path)
                .map_err(|error| format!("Failed to read theme file {path}: {error}"))?;
            let label = path.clone();
            from_str(&label, &content)
        }
    }
}

/// Upstream `getThemeExportColors`.
pub fn get_theme_export_colors(theme_name: Option<&str>) -> ThemeExportColors {
    let name = theme_name.unwrap_or(SYSTEM_THEME_NAME);
    if name == SYSTEM_THEME_NAME {
        return ThemeExportColors::default();
    }
    let Ok(theme_json) = load_theme_json_for_export(name) else {
        return ThemeExportColors::default();
    };
    let Some(export_section) = theme_json.export else {
        return ThemeExportColors::default();
    };
    let vars = theme_json.vars.clone().unwrap_or_default();
    // Export colors end up in CSS, which understands hex and oklch() values
    // directly but not okhsl().
    let resolve = |value: Option<&ThemeColorValue>| -> Option<String> {
        let value = value?;
        if let ThemeColorValue::Index(index) = value {
            return Some(color_to_hex(&crate::utils::tui_compat::indexed_color(
                *index,
            )));
        }
        let resolved = resolve_var_refs(value, &vars, &mut Default::default()).ok()?;
        if resolved.is_empty() {
            return None;
        }
        if resolved.to_ascii_lowercase().starts_with("okhsl(") {
            return parse_color(&resolved)
                .ok()
                .map(|color| color_to_hex(&color));
        }
        Some(resolved)
    };
    ThemeExportColors {
        page_bg: resolve(export_section.page_bg.as_ref()),
        card_bg: resolve(export_section.card_bg.as_ref()),
        info_bg: resolve(export_section.info_bg.as_ref()),
    }
}

/// Generate CSS custom property declarations from theme colors (upstream
/// `generateThemeVars`).
fn generate_theme_vars(theme_name: Option<&str>) -> String {
    let colors = get_resolved_theme_colors(theme_name);
    let mut lines: Vec<String> = Vec::new();
    for (key, value) in &colors {
        lines.push(format!("--{key}: {value};"));
    }

    // Use explicit theme export colors if available, otherwise derive from userMessageBg
    let theme_export = get_theme_export_colors(theme_name);
    let user_message_bg = colors
        .get("userMessageBg")
        .map(String::as_str)
        .unwrap_or("#343541");
    let derived_colors = derive_export_colors(user_message_bg);

    lines.push(format!(
        "--exportPageBg: {};",
        theme_export.page_bg.unwrap_or(derived_colors.page_bg)
    ));
    lines.push(format!(
        "--exportCardBg: {};",
        theme_export.card_bg.unwrap_or(derived_colors.card_bg)
    ));
    lines.push(format!(
        "--exportInfoBg: {};",
        theme_export.info_bg.unwrap_or(derived_colors.info_bg)
    ));

    lines.join("\n      ")
}

// ============================================================================
// Core HTML generation (upstream `generateHtml`)
// ============================================================================

/// Session payload embedded in the template (upstream `SessionData`).
#[derive(Debug, Clone, Serialize)]
pub struct SessionData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<SessionHeader>,
    pub entries: Vec<SessionEntry>,
    #[serde(rename = "leafId")]
    pub leaf_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    /// Pre-rendered HTML for custom tool calls/results, keyed by tool call ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered_tools: Option<BTreeMap<String, RenderedToolHtml>>,
}

/// Minimal base64 (RFC 4648, padded) — upstream `Buffer.toString("base64")`.
/// D80: hand-rolled to avoid a new crate dependency for one call site.
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Extract the current system prompt from a state snapshot (upstream reads
/// `state.systemPrompt` directly; D80: the Rust snapshot carries the
/// transcript, so the prompt is replayed from its system messages).
fn snapshot_system_prompt(state: &AgentStateSnapshot) -> Option<String> {
    let llm: Vec<pi_ai::types::Message> = state
        .messages
        .iter()
        .filter_map(|m| m.as_llm().cloned())
        .collect();
    if llm.is_empty() {
        return None;
    }
    Some(pi_ai::utils::transcript::get_current_system_prompt(&llm))
}

/// Tools declared on the snapshot, reduced to the template's shape
/// (`{ name, description, parameters }`).
fn snapshot_tools(state: &AgentStateSnapshot) -> Vec<Value> {
    state
        .tools
        .iter()
        .map(|agent_tool| {
            serde_json::json!({
                "name": agent_tool.tool.name,
                "description": agent_tool.tool.description,
                "parameters": agent_tool.tool.parameters,
            })
        })
        .collect()
}

/// Core HTML generation logic shared by both export functions (upstream
/// `generateHtml`).
fn generate_html(session_data: &SessionData, theme_name: Option<&str>) -> Result<String, String> {
    let session_data_json = serde_json::to_vec(session_data)
        .map_err(|error| format!("Failed to serialize session data: {error}"))?;

    // Base64 encode session data to avoid escaping issues
    let session_data_base64 = base64_encode(&session_data_json);

    let theme_vars = generate_theme_vars(theme_name);
    let colors = get_resolved_theme_colors(theme_name);
    let theme_export = get_theme_export_colors(theme_name);
    let derived_export_colors = derive_export_colors(
        colors
            .get("userMessageBg")
            .map(String::as_str)
            .unwrap_or("#343541"),
    );
    let body_bg = theme_export
        .page_bg
        .unwrap_or(derived_export_colors.page_bg);
    let container_bg = theme_export
        .card_bg
        .unwrap_or(derived_export_colors.card_bg);
    let info_bg = theme_export
        .info_bg
        .unwrap_or(derived_export_colors.info_bg);

    // Build the CSS with theme variables injected (upstream `String.replace`:
    // first occurrence only).
    let css = TEMPLATE_CSS
        .replacen("{{THEME_VARS}}", &theme_vars, 1)
        .replacen("{{BODY_BG}}", &body_bg, 1)
        .replacen("{{CONTAINER_BG}}", &container_bg, 1)
        .replacen("{{INFO_BG}}", &info_bg, 1);

    Ok(TEMPLATE_HTML
        .replacen("{{CSS}}", &css, 1)
        .replacen("{{JS}}", TEMPLATE_JS, 1)
        .replacen("{{SESSION_DATA}}", &session_data_base64, 1)
        .replacen("{{MARKED_JS}}", MARKED_JS, 1)
        .replacen("{{HIGHLIGHT_JS}}", HIGHLIGHT_JS, 1))
}

/// Tools rendered directly by the HTML template (not pre-rendered via the
/// TUI→ANSI→HTML pipeline).
const TEMPLATE_RENDERED_TOOLS: [&str; 5] = ["bash", "read", "write", "edit", "ls"];

fn is_template_rendered_tool(name: &str) -> bool {
    TEMPLATE_RENDERED_TOOLS.contains(&name)
}

/// Pre-render custom tools to HTML using their TUI renderers (upstream
/// `preRenderCustomTools`).
fn pre_render_custom_tools(
    entries: &[SessionEntry],
    tool_renderer: &dyn ToolHtmlRenderer,
) -> BTreeMap<String, RenderedToolHtml> {
    let mut rendered_tools: BTreeMap<String, RenderedToolHtml> = BTreeMap::new();

    for entry in entries {
        let SessionEntry::Message(message_entry) = entry else {
            continue;
        };
        let message = &message_entry.message;
        let pi_agent::types::AgentMessage::Llm(message) = message else {
            continue;
        };

        // Find tool calls in assistant messages
        if let pi_ai::types::Message::Assistant(msg) = message {
            for block in &msg.content {
                if let pi_ai::types::AssistantContent::ToolCall(tool_call) = block {
                    if is_template_rendered_tool(&tool_call.name) {
                        continue;
                    }
                    let args = Value::Object(tool_call.arguments.clone());
                    if let Some(call_html) =
                        tool_renderer.render_call(&tool_call.id, &tool_call.name, &args)
                    {
                        rendered_tools.insert(
                            tool_call.id.clone(),
                            RenderedToolHtml {
                                call_html: Some(call_html),
                                ..Default::default()
                            },
                        );
                    }
                }
            }
        }

        // Find tool results
        if let pi_ai::types::Message::ToolResult(msg) = message {
            let tool_name = msg.tool_name.as_str();
            // Only render if we have a pre-rendered call OR it's not template-rendered
            let existing = rendered_tools.get(&msg.tool_call_id).cloned();
            if existing.is_some() || !is_template_rendered_tool(tool_name) {
                if let Some(rendered) = tool_renderer.render_result(
                    &msg.tool_call_id,
                    tool_name,
                    &msg.content,
                    msg.details.as_ref(),
                    msg.is_error,
                ) {
                    let entry_out = rendered_tools.entry(msg.tool_call_id.clone()).or_default();
                    if let Some(existing) = existing {
                        *entry_out = existing;
                    }
                    entry_out.result_html_collapsed = rendered.collapsed;
                    entry_out.result_html_expanded = rendered.expanded;
                }
            }
        }
    }

    rendered_tools
}

// ============================================================================
// Export entry points (upstream `exportSessionToHtml` / `exportFromFile`)
// ============================================================================

/// Default output path (upstream inline `${APP_NAME}-session-${basename}.html`).
fn default_output_path(session_file: &str) -> String {
    // Node `basename(resolvedInputPath, ".jsonl")`: last path segment with a
    // trailing `.jsonl` suffix stripped (upstream writes cwd-relative).
    let base = session_file
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(session_file);
    let base = base.strip_suffix(".jsonl").unwrap_or(base);
    format!("{APP_NAME}-session-{base}.html")
}

fn write_output(path: &str, html: &str) -> Result<String, String> {
    let mut file =
        std::fs::File::create(path).map_err(|error| format!("Failed to write {path}: {error}"))?;
    file.write_all(html.as_bytes())
        .map_err(|error| format!("Failed to write {path}: {error}"))?;
    Ok(path.to_string())
}

/// Export session to HTML using SessionManager and AgentState (upstream
/// `exportSessionToHtml`; used by AgentSession's /export command).
pub fn export_session_to_html(
    sm: &SessionManager,
    state: Option<&AgentStateSnapshot>,
    options: ExportOptions,
) -> Result<String, String> {
    let session_file = sm
        .get_session_file()
        .map(str::to_string)
        .ok_or_else(|| "Cannot export in-memory session to HTML".to_string())?;
    if !Path::new(&session_file).exists() {
        return Err("Nothing to export yet - start a conversation first".to_string());
    }

    let entries = sm.get_entries();

    // Pre-render custom tools if a tool renderer is provided
    let mut rendered_tools: Option<BTreeMap<String, RenderedToolHtml>> = None;
    if let Some(tool_renderer) = &options.tool_renderer {
        let pre_rendered = pre_render_custom_tools(&entries, tool_renderer.as_ref());
        // Only include if we actually rendered something
        if !pre_rendered.is_empty() {
            rendered_tools = Some(pre_rendered);
        }
    }

    let session_data = SessionData {
        header: sm.get_header(),
        entries,
        leaf_id: sm.get_leaf_id().map(str::to_string),
        system_prompt: state.map(snapshot_system_prompt).flatten(),
        tools: state.map(snapshot_tools),
        rendered_tools,
    };

    let html = generate_html(&session_data, options.theme_name.as_deref())?;

    let output_path = match &options.output_path {
        Some(path) => normalize_path(path, &PathInputOptions::default()),
        None => default_output_path(&session_file),
    };
    write_output(&output_path, &html)
}

/// Export session file to HTML (standalone, without AgentState; upstream
/// `exportFromFile`, used by the CLI for exporting arbitrary session files).
pub fn export_from_file(input_path: &str, options: ExportOptions) -> Result<String, String> {
    let resolved_input_path = resolve_path(
        input_path,
        &std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy(),
    );
    if !Path::new(&resolved_input_path).exists() {
        return Err(format!("File not found: {resolved_input_path}"));
    }

    let sm = SessionManager::open(&resolved_input_path, None, None)
        .map_err(|error| error.to_string())?;

    let session_data = SessionData {
        header: sm.get_header(),
        entries: sm.get_entries(),
        leaf_id: sm.get_leaf_id().map(str::to_string),
        system_prompt: None,
        tools: None,
        rendered_tools: None,
    };

    let html = generate_html(&session_data, options.theme_name.as_deref())?;

    let output_path = match &options.output_path {
        Some(path) => normalize_path(path, &PathInputOptions::default()),
        None => default_output_path(&resolved_input_path),
    };
    write_output(&output_path, &html)
}
