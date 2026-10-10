//! Port of upstream `theme.ts` — the theme foundation: the [`Theme`] type with exact ANSI
//! rendering (`fg`/`bg`/`style`), theme JSON loading and variable resolution, appearance
//! detection, terminal color tracking, and the `system`/`dark`/`light` builtin themes.
//!
//! Upstream: `packages/coding-agent/src/modes/interactive/theme/theme.ts` (with `theme-json.ts`
//! in [`theme_json`] and `system-theme.ts` in [`system_theme`]).
//!
//! # Deviations
//!
//! - **D56 — capability detection lives in [`crate::utils::tui_compat`]**, a minimal port of the
//!   npm package `@earendil-works/pi-tui` that lands in a future `pi-tui` crate.
//! - **D56c — built-in themes are embedded.** `dark.json`/`light.json` are compiled in with
//!   `include_str!` instead of read from the package's theme directory at runtime, so ports that
//!   ship without the upstream package tree still resolve them. `theme-schema.json` ships
//!   alongside for editor support.
//! - **D56d — TUI/interactive-only surface is deferred** with the excluded interactive mode:
//!   the global `theme` proxy and `initTheme`/`setTheme`/`setThemeInstance`/`onThemeChange` and
//!   the custom-theme file watcher, the cli-highlight bridge (`highlightCode`,
//!   `getLanguageFromPath`, `buildCliHighlightTheme`), the TUI component themes
//!   (`getMarkdownTheme`, `getSelectListTheme`, `getEditorTheme`, `getSettingsListTheme`),
//!   the HTML-export helpers (`getResolvedThemeColors`, `isLightTheme`, `getThemeExportColors`),
//!   and the `ThinkingLevel`-typed accessors (`getThinkingBorderColor`, `getBashModeBorderColor`
//!   — use `fg("thinkingOff", ..)` / `fg("bashMode", ..)` directly). The registry
//!   (`setRegisteredThemes`, `getThemeByName`, `getAvailableThemes*`) and the terminal-color
//!   tracking (`setTerminalColors`, `setTerminalColorScheme`, `markTerminalColorsPending`) are
//!   ported because the resource loader drives them.
//! - **D56e — JS `chalk` text decorators** (`bold`/`italic`/…) are not ported; use
//!   [`crate::utils::ansi`].
#![allow(dead_code)]

pub mod system_theme;
pub mod theme_json;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::LazyLock;

use parking_lot::Mutex;

use crate::core::source_info::SourceInfo;
use crate::utils::text::strip_bom;
use crate::utils::tui_compat::{
    background_ansi, color_to_oklch, current_terminal_color_mode, foreground_ansi, mix_colors,
    parse_color, style_text_with_ansi, Color, TerminalColorMode, TerminalColors, TextAttributes,
};

use system_theme::{generate_system_theme_colors, SYSTEM_THEME_NAME};
use theme_json::{ThemeColorValue, ValidatedThemeJson};

pub use system_theme::SYSTEM_THEME_NAME as SYSTEM_THEME;
pub use theme_json::ThemeJsonValidator;

// ============================================================================
// Global theme state (upstream module-level lets)
// ============================================================================

struct GlobalState {
    /// The terminal's reported colors. Replaced (never mutated) on update, so themes can cache
    /// resolved colors by identity.
    terminal_colors: TerminalColors,
    /// While the terminal color query is in flight, the system theme renders in grayscale.
    terminal_colors_pending: bool,
    /// The terminal's last light/dark report (mode 2031). Only used while it has not reported a
    /// background.
    terminal_color_scheme: Option<&'static str>,
    registered_themes: BTreeMap<String, Theme>,
}

static GLOBAL_STATE: LazyLock<Mutex<GlobalState>> = LazyLock::new(|| {
    Mutex::new(GlobalState {
        terminal_colors: TerminalColors::default(),
        terminal_colors_pending: false,
        terminal_color_scheme: None,
        registered_themes: BTreeMap::new(),
    })
});

/// Serializes tests that touch the global theme state (upstream vitest runs these per-file).
pub fn theme_global_lock() -> parking_lot::MutexGuard<'static, ()> {
    static TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
    TEST_LOCK.lock()
}

/// Record the terminal's reported colors. Themes use the default colors for tokens set to ""
/// (terminal default); the system theme is generated from all of them. Ends the pending state.
pub fn set_terminal_colors(colors: &TerminalColors) {
    let mut state = GLOBAL_STATE.lock();
    state.terminal_colors = colors.clone();
    state.terminal_colors_pending = false;
}

/// Record the terminal's light/dark report, the fallback for terminals that do not report their
/// background.
pub fn set_terminal_color_scheme(scheme: Option<&'static str>) {
    GLOBAL_STATE.lock().terminal_color_scheme = scheme;
}

/// Render the system theme in grayscale until [`set_terminal_colors`] reports the terminal's
/// colors.
pub fn mark_terminal_colors_pending() {
    GLOBAL_STATE.lock().terminal_colors_pending = true;
}

// ============================================================================
// Color value resolution
// ============================================================================

/// Resolve variable references to their final literal color value.
fn resolve_var_refs(
    value: &ThemeColorValue,
    vars: &BTreeMap<String, ThemeColorValue>,
) -> Result<ThemeColorValue, String> {
    resolve_var_refs_visited(value, vars, &mut BTreeSet::new())
}

fn resolve_var_refs_visited(
    value: &ThemeColorValue,
    vars: &BTreeMap<String, ThemeColorValue>,
    visited: &mut BTreeSet<String>,
) -> Result<ThemeColorValue, String> {
    let text = match value {
        ThemeColorValue::Index(_) => return Ok(value.clone()),
        ThemeColorValue::Str(text) => text,
    };
    if text.is_empty() || text.starts_with('#') || is_ok_color(text) {
        return Ok(value.clone());
    }
    if visited.contains(text) {
        return Err(format!("Circular variable reference detected: {text}"));
    }
    let Some(target) = vars.get(text) else {
        return Err(format!("Variable reference not found: {text}"));
    };
    visited.insert(text.clone());
    resolve_var_refs_visited(target, vars, visited)
}

fn is_ok_color(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.starts_with("oklch(") || lower.starts_with("okhsl(")
}

fn resolve_theme_colors(
    colors: &BTreeMap<String, ThemeColorValue>,
    vars: &BTreeMap<String, ThemeColorValue>,
) -> Result<BTreeMap<String, ThemeColorValue>, String> {
    colors
        .iter()
        .map(|(key, value)| Ok((key.clone(), resolve_var_refs(value, vars)?)))
        .collect()
}

/// Fill the optional color tokens with their compatible fallbacks.
fn with_theme_color_fallbacks(
    colors: &BTreeMap<String, ThemeColorValue>,
) -> BTreeMap<String, ThemeColorValue> {
    let mut colors = colors.clone();
    let fallback = |colors: &BTreeMap<String, ThemeColorValue>, key: &str, from: &str| {
        colors.get(key).cloned().unwrap_or_else(|| {
            colors
                .get(from)
                .cloned()
                .unwrap_or(ThemeColorValue::Str(String::new()))
        })
    };
    colors.insert(
        "scrollbarTrack".to_string(),
        fallback(&colors, "scrollbarTrack", "muted"),
    );
    colors.insert(
        "scrollbarThumb".to_string(),
        fallback(&colors, "scrollbarThumb", "text"),
    );
    colors.insert(
        "thinkingMax".to_string(),
        fallback(&colors, "thinkingMax", "thinkingXhigh"),
    );
    colors.insert(
        "searchMatchBg".to_string(),
        fallback(&colors, "searchMatchBg", "selectedBg"),
    );
    colors.insert(
        "searchMatchText".to_string(),
        fallback(&colors, "searchMatchText", "text"),
    );
    colors
}

// ============================================================================
// Appearance & terminal default colors
// ============================================================================

const GUESSED_DEFAULT_DARK_FG: &str = "#e5e5e7";
const GUESSED_DEFAULT_DARK_BG: &str = "#000000";
const GUESSED_DEFAULT_LIGHT_FG: &str = "#000000";
const GUESSED_DEFAULT_LIGHT_BG: &str = "#ffffff";

/// Average OKLab lightness of the concrete colors; palette colors 0-15 follow the user's
/// terminal palette, so they say nothing about the theme.
fn average_lightness(colors: &[Color]) -> Option<f64> {
    let fixed: Vec<&Color> = colors
        .iter()
        .filter(|color| !matches!(color, Color::Indexed { index } if *index < 16))
        .collect();
    if fixed.is_empty() {
        return None;
    }
    Some(
        fixed
            .iter()
            .map(|color| color_to_oklch(color).0)
            .sum::<f64>()
            / fixed.len() as f64,
    )
}

/// Detect the background a theme is designed for from the lightness of its own colors.
fn detect_appearance(foregrounds: &[Color], backgrounds: &[Color]) -> Option<&'static str> {
    let fg = average_lightness(foregrounds);
    let bg = average_lightness(backgrounds);
    if let (Some(fg), Some(bg)) = (fg, bg) {
        return Some(if bg < fg { "dark" } else { "light" });
    }
    if let Some(bg) = bg {
        return Some(if bg < 0.5 { "dark" } else { "light" });
    }
    fg.map(|fg| if fg > 0.5 { "dark" } else { "light" })
}

// ============================================================================
// Theme
// ============================================================================

/// Color tokens (foreground slots), upstream `ThemeColor`.
pub const THEME_COLORS: &[&str] = &[
    "accent",
    "border",
    "borderAccent",
    "borderMuted",
    "success",
    "error",
    "warning",
    "muted",
    "dim",
    "text",
    "thinkingText",
    "scrollbarTrack",
    "scrollbarThumb",
    "searchMatchText",
    "userMessageText",
    "customMessageText",
    "customMessageLabel",
    "toolTitle",
    "toolOutput",
    "mdHeading",
    "mdLink",
    "mdLinkUrl",
    "mdCode",
    "mdCodeBlock",
    "mdCodeBlockBorder",
    "mdQuote",
    "mdQuoteBorder",
    "mdHr",
    "mdListBullet",
    "toolDiffAdded",
    "toolDiffRemoved",
    "toolDiffContext",
    "syntaxComment",
    "syntaxKeyword",
    "syntaxFunction",
    "syntaxVariable",
    "syntaxString",
    "syntaxNumber",
    "syntaxType",
    "syntaxOperator",
    "syntaxPunctuation",
    "thinkingOff",
    "thinkingMinimal",
    "thinkingLow",
    "thinkingMedium",
    "thinkingHigh",
    "thinkingXhigh",
    "thinkingMax",
    "bashMode",
];

/// Background tokens, upstream `ThemeBg`.
pub const THEME_BGS: &[&str] = &[
    "selectedBg",
    "searchMatchBg",
    "userMessageBg",
    "customMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
];

const BACKGROUND_TOKENS: &[&str] = &[
    "selectedBg",
    "searchMatchBg",
    "userMessageBg",
    "customMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
];

/// A foreground or background slot for [`Theme::style`].
#[derive(Debug, Clone)]
pub enum SlotValue {
    /// A theme token in its own slot.
    Token(String),
    /// A concrete color.
    Color(Color),
}

/// Options for [`Theme::style`] (upstream `ThemeStyle`; only text attributes plus fg/bg).
#[derive(Debug, Clone, Default)]
pub struct StyleOptions {
    pub fg: Option<SlotValue>,
    pub bg: Option<SlotValue>,
    pub attributes: TextAttributes,
}

#[derive(Debug, Clone, Default)]
pub struct ThemeOptions {
    pub name: Option<String>,
    pub source_path: Option<String>,
    pub source_info: Option<SourceInfo>,
    /// The background the theme is designed for; detected from its colors when omitted.
    pub appearance: Option<&'static str>,
    /// Foreground tokens to render faint (SGR 2).
    pub dim: Vec<String>,
}

/// A resolved theme: precomputed ANSI escape sequences per token, rendered in one color mode.
#[derive(Debug, Clone)]
pub struct Theme {
    pub name: Option<String>,
    pub source_path: Option<String>,
    pub source_info: Option<SourceInfo>,
    mode: TerminalColorMode,
    /// Precomputed escape sequences keep fg()/bg() on the render hot path to a lookup and concat.
    fg_ansi: BTreeMap<String, String>,
    bg_ansi: BTreeMap<String, String>,
    /// Tokens set to "" have no color of their own; `colors` fills them from the terminal defaults.
    concrete_colors: BTreeMap<String, Color>,
    default_foreground_tokens: Vec<String>,
    default_background_tokens: Vec<String>,
    /// Foreground tokens rendered faint (SGR 2) on top of their color.
    dim_tokens: BTreeSet<String>,
    own_appearance: Option<&'static str>,
}

impl Theme {
    pub fn new(
        fg_colors: &BTreeMap<String, ThemeColorValue>,
        bg_colors: &BTreeMap<String, ThemeColorValue>,
        mode: TerminalColorMode,
        options: ThemeOptions,
    ) -> Result<Self, String> {
        let mut fg_ansi = BTreeMap::new();
        let mut bg_ansi = BTreeMap::new();
        let mut concrete_colors = BTreeMap::new();
        let mut default_foreground_tokens = Vec::new();
        let mut default_background_tokens = Vec::new();
        let mut concrete_foregrounds: Vec<Color> = Vec::new();
        let mut concrete_backgrounds: Vec<Color> = Vec::new();

        // Fallbacks for optional tokens, as upstream does in the constructor.
        let fallback = |colors: &BTreeMap<String, ThemeColorValue>, key: &str, from: &str| {
            colors.get(key).cloned().unwrap_or_else(|| {
                colors
                    .get(from)
                    .cloned()
                    .unwrap_or(ThemeColorValue::Str(String::new()))
            })
        };
        let mut foregrounds = fg_colors.clone();
        foregrounds.insert(
            "scrollbarTrack".to_string(),
            fallback(fg_colors, "scrollbarTrack", "muted"),
        );
        foregrounds.insert(
            "scrollbarThumb".to_string(),
            fallback(fg_colors, "scrollbarThumb", "text"),
        );
        foregrounds.insert(
            "thinkingMax".to_string(),
            fallback(fg_colors, "thinkingMax", "thinkingXhigh"),
        );
        foregrounds.insert(
            "searchMatchText".to_string(),
            fallback(fg_colors, "searchMatchText", "text"),
        );
        let mut backgrounds = bg_colors.clone();
        backgrounds.insert(
            "searchMatchBg".to_string(),
            fallback(bg_colors, "searchMatchBg", "selectedBg"),
        );

        // Returns the escape sequence for the token's own slot.
        let mut add_token =
            |token: &str, value: &ThemeColorValue, is_background: bool| -> Result<String, String> {
                let ThemeColorValue::Str(text) = value else {
                    let index = match value {
                        ThemeColorValue::Index(index) => *index,
                        _ => unreachable!(),
                    };
                    let color = parse_color(&index.to_string())?;
                    concrete_colors.insert(token.to_string(), color);
                    if is_background {
                        concrete_backgrounds.push(color);
                    } else {
                        concrete_foregrounds.push(color);
                    }
                    return Ok(if is_background {
                        background_ansi(&color, mode)
                    } else {
                        foreground_ansi(&color, mode)
                    });
                };
                if text.is_empty() {
                    if is_background {
                        default_background_tokens.push(token.to_string());
                    } else {
                        default_foreground_tokens.push(token.to_string());
                    }
                    return Ok(if is_background {
                        "\x1b[49m".to_string()
                    } else {
                        "\x1b[39m".to_string()
                    });
                }
                let color = parse_color(text)?;
                concrete_colors.insert(token.to_string(), color);
                if is_background {
                    concrete_backgrounds.push(color);
                } else {
                    concrete_foregrounds.push(color);
                }
                Ok(if is_background {
                    background_ansi(&color, mode)
                } else {
                    foreground_ansi(&color, mode)
                })
            };

        for (token, value) in &foregrounds {
            fg_ansi.insert(token.clone(), add_token(token, value, false)?);
        }
        for (token, value) in &backgrounds {
            bg_ansi.insert(token.clone(), add_token(token, value, true)?);
        }

        let own_appearance = options
            .appearance
            .or_else(|| detect_appearance(&concrete_foregrounds, &concrete_backgrounds));
        Ok(Theme {
            name: options.name,
            source_path: options.source_path,
            source_info: options.source_info,
            mode,
            fg_ansi,
            bg_ansi,
            concrete_colors,
            default_foreground_tokens,
            default_background_tokens,
            dim_tokens: options.dim.into_iter().collect(),
            own_appearance,
        })
    }

    /// The background the theme is designed for: declared in the theme JSON, detected from its
    /// colors, or, for themes without usable colors, the terminal's appearance.
    pub fn appearance(&self) -> &'static str {
        self.own_appearance.unwrap_or_else(get_terminal_theme)
    }

    /// Concrete colors for all tokens. Tokens set to "" (terminal default) use the terminal's
    /// reported default colors, or a guess based on `appearance` when the terminal did not report
    /// them. Faint tokens are approximated by mixing their color toward the background.
    pub fn colors(&self) -> BTreeMap<String, Color> {
        // Resolve the appearance before taking the state lock: `get_terminal_theme()` locks it too.
        let appearance = self.own_appearance.unwrap_or_else(get_terminal_theme);
        let state = GLOBAL_STATE.lock();
        let terminal = &state.terminal_colors;
        let guess = if appearance == "dark" {
            (
                parse_color(GUESSED_DEFAULT_DARK_FG).unwrap(),
                parse_color(GUESSED_DEFAULT_DARK_BG).unwrap(),
            )
        } else {
            (
                parse_color(GUESSED_DEFAULT_LIGHT_FG).unwrap(),
                parse_color(GUESSED_DEFAULT_LIGHT_BG).unwrap(),
            )
        };
        let foreground = terminal
            .foreground
            .map(|rgb| Color::Rgb {
                r: rgb.r,
                g: rgb.g,
                b: rgb.b,
            })
            .unwrap_or(guess.0);
        let background = terminal
            .background
            .map(|rgb| Color::Rgb {
                r: rgb.r,
                g: rgb.g,
                b: rgb.b,
            })
            .unwrap_or(guess.1);
        let mut colors = self.concrete_colors.clone();
        for token in &self.default_foreground_tokens {
            colors.insert(token.clone(), foreground);
        }
        for token in &self.default_background_tokens {
            colors.insert(token.clone(), background);
        }
        for token in &self.dim_tokens {
            if let Some(color) = colors.get(token).cloned() {
                colors.insert(token.clone(), mix_colors(&color, &background, 0.4).unwrap());
            }
        }
        colors
    }

    /// A single resolved color token (upstream `theme.colors[token]`).
    pub fn color(&self, token: &str) -> Option<Color> {
        self.colors().get(token).cloned()
    }

    fn token_ansi(&self, ansi: &BTreeMap<String, String>, token: &str) -> Result<String, String> {
        ansi.get(token)
            .cloned()
            .ok_or_else(|| format!("Unknown theme color: {token}"))
    }

    /// Render `text` with token or concrete colors plus text attributes.
    pub fn style(&self, text: &str, options: &StyleOptions) -> Result<String, String> {
        let mut attributes = options.attributes.clone();
        if let Some(SlotValue::Token(token)) = &options.fg {
            if self.dim_tokens.contains(token) {
                attributes.dim = true;
            }
        }
        let fg_ansi = match &options.fg {
            None => None,
            Some(SlotValue::Token(token)) => Some(self.token_ansi(&self.fg_ansi, token)?),
            Some(SlotValue::Color(color)) => Some(foreground_ansi(color, self.mode)),
        };
        let bg_ansi = match &options.bg {
            None => None,
            Some(SlotValue::Token(token)) => Some(self.token_ansi(&self.bg_ansi, token)?),
            Some(SlotValue::Color(color)) => Some(background_ansi(color, self.mode)),
        };
        Ok(style_text_with_ansi(
            text,
            fg_ansi.as_deref(),
            bg_ansi.as_deref(),
            &attributes,
        ))
    }

    pub fn fg(&self, color: &str, text: &str) -> Result<String, String> {
        let ansi = self.token_ansi(&self.fg_ansi, color)?;
        if self.dim_tokens.contains(color) {
            return Ok(format!("{ansi}\x1b[2m{text}\x1b[22;39m"));
        }
        Ok(format!("{ansi}{text}\x1b[39m"))
    }

    pub fn bg(&self, color: &str, text: &str) -> Result<String, String> {
        let ansi = self.token_ansi(&self.bg_ansi, color)?;
        Ok(format!("{ansi}{text}\x1b[49m"))
    }

    /// Opening escape sequence for a foreground token. Faint tokens include SGR 2, which
    /// `\x1b[22m` closes.
    pub fn get_fg_ansi(&self, color: &str) -> Result<String, String> {
        let ansi = self.token_ansi(&self.fg_ansi, color)?;
        Ok(if self.dim_tokens.contains(color) {
            format!("{ansi}\x1b[2m")
        } else {
            ansi
        })
    }

    pub fn get_bg_ansi(&self, color: &str) -> Result<String, String> {
        self.token_ansi(&self.bg_ansi, color)
    }

    pub fn color_mode(&self) -> TerminalColorMode {
        self.mode
    }
}

// ============================================================================
// Theme loading
// ============================================================================

fn builtin_theme_json(name: &str) -> Option<serde_json::Value> {
    const DARK: &str = include_str!("dark.json");
    const LIGHT: &str = include_str!("light.json");
    let source = match name {
        "dark" => DARK,
        "light" => LIGHT,
        _ => return None,
    };
    serde_json::from_str(source).ok()
}

pub fn get_available_themes() -> Vec<String> {
    get_available_themes_with_paths()
        .into_iter()
        .map(|info| info.name)
        .collect()
}

#[derive(Debug, Clone)]
pub struct ThemeInfo {
    pub name: String,
    pub path: Option<String>,
}

pub fn get_available_themes_with_paths() -> Vec<ThemeInfo> {
    let state = GLOBAL_STATE.lock();
    let mut result: Vec<ThemeInfo> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut add_theme =
        |theme_info: ThemeInfo, result: &mut Vec<ThemeInfo>, seen: &mut BTreeSet<String>| {
            if seen.contains(&theme_info.name) {
                return;
            }
            seen.insert(theme_info.name.clone());
            result.push(theme_info);
        };

    // Built-in themes. The system theme is generated, so it has no file.
    add_theme(
        ThemeInfo {
            name: SYSTEM_THEME_NAME.to_string(),
            path: None,
        },
        &mut result,
        &mut seen,
    );
    for name in ["dark", "light"] {
        add_theme(
            ThemeInfo {
                name: name.to_string(),
                path: Some(format!("{name}.json")),
            },
            &mut result,
            &mut seen,
        );
    }

    // Registered (in-memory) themes.
    for (name, theme) in &state.registered_themes {
        add_theme(
            ThemeInfo {
                name: name.clone(),
                path: theme.source_path.clone(),
            },
            &mut result,
            &mut seen,
        );
    }

    // The system theme comes first: it is the default and adapts to every terminal.
    result.sort_by(|a, b| {
        if a.name == SYSTEM_THEME_NAME {
            std::cmp::Ordering::Less
        } else if b.name == SYSTEM_THEME_NAME {
            std::cmp::Ordering::Greater
        } else {
            a.name.cmp(&b.name)
        }
    });
    result
}

fn assert_theme_name_is_valid(name: &str) -> Result<(), String> {
    if name.contains('/') {
        return Err(format!(
			"Invalid theme name \"{name}\": theme names cannot contain \"/\" because it is reserved for automatic light/dark theme settings."
		));
    }
    Ok(())
}

fn parse_theme_json(label: &str, json: &serde_json::Value) -> Result<ValidatedThemeJson, String> {
    if let Some(validator) = *theme_json::validator_slot().lock() {
        return validator(label, json);
    }
    let Some(object) = json.as_object() else {
        return Err(format!(
            "Invalid theme \"{label}\": expected an object with a \"colors\" map."
        ));
    };
    if !object.contains_key("colors") {
        return Err(format!(
            "Invalid theme \"{label}\": expected an object with a \"colors\" map."
        ));
    }
    // Accept the document as-is (built-in themes), extracting the known fields leniently.
    let mut colors = BTreeMap::new();
    if let Some(colors_json) = object.get("colors").and_then(|value| value.as_object()) {
        for (key, value) in colors_json {
            let color = match value {
                serde_json::Value::String(text) => ThemeColorValue::Str(text.clone()),
                serde_json::Value::Number(number) => match number.as_u64() {
                    Some(index) if index <= 255 => ThemeColorValue::Index(index as u8),
                    _ => {
                        return Err(format!(
                            "Invalid theme \"{label}\": invalid color for \"{key}\"."
                        ))
                    }
                },
                _ => {
                    return Err(format!(
                        "Invalid theme \"{label}\": invalid color for \"{key}\"."
                    ))
                }
            };
            colors.insert(key.clone(), color);
        }
    }
    let vars = object
        .get("vars")
        .and_then(|value| value.as_object())
        .map(|vars| {
            vars.iter()
                .map(|(key, value)| {
                    let color = match value {
                        serde_json::Value::String(text) => ThemeColorValue::Str(text.clone()),
                        serde_json::Value::Number(number) => match number.as_u64() {
                            Some(index) if index <= 255 => ThemeColorValue::Index(index as u8),
                            _ => ThemeColorValue::Str(String::new()),
                        },
                        _ => ThemeColorValue::Str(String::new()),
                    };
                    (key.clone(), color)
                })
                .collect::<BTreeMap<_, _>>()
        });
    Ok(ValidatedThemeJson {
        name: object
            .get("name")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
        appearance: match object.get("appearance").and_then(|value| value.as_str()) {
            Some("dark") => Some("dark".to_string()),
            Some("light") => Some("light".to_string()),
            _ => None,
        },
        vars,
        colors,
        export: None,
    })
}

fn parse_theme_json_content(label: &str, content: &str) -> Result<ValidatedThemeJson, String> {
    let json: serde_json::Value = serde_json::from_str(strip_bom(content))
        .map_err(|error| format!("Failed to parse theme {label}: {error}"))?;
    parse_theme_json(label, &json)
}

pub fn load_theme_from_path(
    theme_path: &str,
    mode: Option<TerminalColorMode>,
) -> Result<Theme, String> {
    let content = std::fs::read_to_string(theme_path).map_err(|error| format!("{error}"))?;
    let theme_json = parse_theme_json_content(theme_path, &content)?;
    create_theme(&theme_json, mode, Some(theme_path.to_string()))
}

fn split_theme_colors(
    colors: &BTreeMap<String, ThemeColorValue>,
) -> (
    BTreeMap<String, ThemeColorValue>,
    BTreeMap<String, ThemeColorValue>,
) {
    let mut fg_colors = BTreeMap::new();
    let mut bg_colors = BTreeMap::new();
    for (key, value) in colors {
        if BACKGROUND_TOKENS.contains(&key.as_str()) {
            bg_colors.insert(key.clone(), value.clone());
        } else {
            fg_colors.insert(key.clone(), value.clone());
        }
    }
    (fg_colors, bg_colors)
}

fn create_theme(
    theme_json: &ValidatedThemeJson,
    mode: Option<TerminalColorMode>,
    source_path: Option<String>,
) -> Result<Theme, String> {
    let color_mode = mode.unwrap_or_else(current_terminal_color_mode);
    let resolved_colors = resolve_theme_colors(
        &with_theme_color_fallbacks(&theme_json.colors),
        &theme_json.vars.clone().unwrap_or_default(),
    )?;
    let (fg_colors, bg_colors) = split_theme_colors(&resolved_colors);
    Theme::new(
        &fg_colors,
        &bg_colors,
        color_mode,
        ThemeOptions {
            name: Some(theme_json.name.clone()),
            source_path,
            source_info: None,
            appearance: match theme_json.appearance.as_deref() {
                Some("dark") => Some("dark"),
                Some("light") => Some("light"),
                _ => None,
            },
            dim: Vec::new(),
        },
    )
}

/// Whether the terminal is dark or light, from everything it reported so far. See
/// [`detect_terminal_theme`].
pub fn get_terminal_theme() -> &'static str {
    let state = GLOBAL_STATE.lock();
    detect_terminal_theme(&state.terminal_colors, state.terminal_color_scheme, None)
}

/// Generate the system theme from the terminal's reported colors (grayscale while they are
/// pending).
fn create_system_theme(mode: Option<TerminalColorMode>) -> Theme {
    // Resolve the appearance before taking the state lock: `get_terminal_theme()` locks it too.
    let appearance_hint = get_terminal_theme();
    let state = GLOBAL_STATE.lock();
    let generated = generate_system_theme_colors(&system_theme::SystemThemeInput {
        foreground: state.terminal_colors.foreground,
        background: state.terminal_colors.background,
        palette: state.terminal_colors.palette.clone(),
        saturation: Some(if state.terminal_colors_pending {
            0.0
        } else {
            1.0
        }),
        appearance_hint: Some(appearance_hint),
    });
    let (fg_colors, bg_colors) = split_theme_colors(&generated.colors);
    Theme::new(
        &fg_colors,
        &bg_colors,
        mode.unwrap_or_else(current_terminal_color_mode),
        ThemeOptions {
            name: Some(SYSTEM_THEME_NAME.to_string()),
            source_path: None,
            source_info: None,
            appearance: generated.appearance,
            dim: generated.dim,
        },
    )
    .expect("system theme colors are valid")
}

fn load_theme(name: &str, mode: Option<TerminalColorMode>) -> Result<Theme, String> {
    // The system theme name is reserved: it takes precedence over custom themes of the same name.
    if name == SYSTEM_THEME_NAME {
        return Ok(create_system_theme(mode));
    }
    if let Some(registered) = GLOBAL_STATE.lock().registered_themes.get(name).cloned() {
        return Ok(registered);
    }
    if let Some(json) = builtin_theme_json(name) {
        return create_theme(&parse_theme_json(name, &json)?, mode, None);
    }
    let theme_path = format!("{}/{}.json", crate::config::get_custom_themes_dir(), name);
    if !Path::new(&theme_path).exists() {
        return Err(format!("Theme not found: {name}"));
    }
    let content = std::fs::read_to_string(&theme_path).map_err(|error| format!("{error}"))?;
    create_theme(&parse_theme_json_content(name, &content)?, mode, None)
}

pub fn get_theme_by_name(name: &str) -> Option<Theme> {
    load_theme(name, None).ok()
}

/// Replace the in-memory theme registry (upstream `setRegisteredThemes`).
pub fn set_registered_themes(themes: Vec<Theme>) -> Result<(), String> {
    let mut state = GLOBAL_STATE.lock();
    state.registered_themes.clear();
    for theme in themes {
        if let Some(name) = &theme.name {
            assert_theme_name_is_valid(name)?;
            state.registered_themes.insert(name.clone(), theme);
        }
    }
    Ok(())
}

// ============================================================================
// Theme setting helpers & terminal detection
// ============================================================================

/// Split a `light/dark` automatic theme setting into its two halves.
pub fn parse_auto_theme_setting(theme_setting: Option<&str>) -> Option<(String, String)> {
    let theme_setting = theme_setting?;
    let slash_index = theme_setting.find('/')?;
    if theme_setting[slash_index + 1..].contains('/') {
        return None;
    }
    let light_theme = theme_setting[..slash_index].trim().to_string();
    let dark_theme = theme_setting[slash_index + 1..].trim().to_string();
    if light_theme.is_empty() || dark_theme.is_empty() {
        return None;
    }
    Some((light_theme, dark_theme))
}

pub fn resolve_theme_setting(theme_setting: Option<&str>, terminal_theme: &str) -> Option<String> {
    if let Some((light_theme, dark_theme)) = parse_auto_theme_setting(theme_setting) {
        return Some(if terminal_theme == "light" {
            light_theme
        } else {
            dark_theme
        });
    }
    let theme_setting = theme_setting?;
    if theme_setting.contains('/') {
        return None;
    }
    Some(theme_setting.to_string())
}

/// Dark or light from the `COLORFGBG` environment variable some terminals set, or `None` without
/// a usable background index. The value is `fg;bg` or `fg;xpm;bg` (rxvt), where a field is an ANSI
/// color index or `default` when the color is not in the palette. The index refers to the
/// terminal's own palette, whose colors are unknown here, so it is classified by index like Vim
/// does: 0-6 and 8 (bright black, e.g. Solarized Dark's background) are dark, 7 and 9-15 are
/// light.
pub fn detect_color_fg_bg_theme(colorfgbg: Option<&str>) -> Option<&'static str> {
    let colorfgbg = colorfgbg?;
    let bg = colorfgbg.split(';').last()?.trim();
    if bg.is_empty() || bg.len() > 2 || !bg.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let index: usize = bg.parse().ok()?;
    if index > 15 {
        return None;
    }
    Some(if index <= 6 || index == 8 {
        "dark"
    } else {
        "light"
    })
}

/// Whether the terminal is dark or light. The background it renders decides, classified the same
/// way the system theme does. Without a reported background: the terminal's light/dark report,
/// then COLORFGBG, then dark.
pub fn detect_terminal_theme(
    colors: &TerminalColors,
    reported_scheme: Option<&'static str>,
    colorfgbg: Option<&str>,
) -> &'static str {
    if let Some(background) = &colors.background {
        return system_theme::terminal_appearance(background, colors.foreground.as_ref());
    }
    reported_scheme
        .or_else(|| detect_color_fg_bg_theme(colorfgbg))
        .unwrap_or("dark")
}

// ============================================================================
// Tests — 1:1 ports of upstream `theme-style.test.ts` and `theme-detection.test.ts`
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::system_theme::rgb_from_hex;
    use serde_json::json;

    /// Load a copy of a built-in theme from a temp file, modified by `edit`.
    fn load_theme_edited(
        base: &str,
        edit: impl FnOnce(&mut serde_json::Value),
    ) -> Result<Theme, String> {
        let mut theme_json: serde_json::Value = serde_json::from_str(if base == "dark" {
            include_str!("dark.json")
        } else {
            include_str!("light.json")
        })
        .unwrap();
        edit(&mut theme_json);
        let dir =
            std::env::temp_dir().join(format!("pi-theme-style-rs-{}-{}", std::process::id(), base));
        std::fs::create_dir_all(&dir).unwrap();
        let name = theme_json["name"].as_str().unwrap().to_string();
        let path = dir.join(format!("{name}.json"));
        std::fs::write(&path, serde_json::to_string(&theme_json).unwrap()).unwrap();
        load_theme_from_path(path.to_str().unwrap(), Some("truecolor"))
    }

    #[test]
    fn renders_theme_tokens_the_same_as_the_generic_text_styler() {
        let _guard = theme_global_lock();
        let theme = load_theme_edited("dark", |_| {}).unwrap();
        let fg = theme.color("success").unwrap();
        let bg = theme.color("toolSuccessBg").unwrap();
        let expected = style_text_with_ansi(
            "Ready",
            Some(&foreground_ansi(&fg, "truecolor")),
            Some(&background_ansi(&bg, "truecolor")),
            &TextAttributes {
                bold: true,
                ..Default::default()
            },
        );
        assert_eq!(
            theme
                .style(
                    "Ready",
                    &StyleOptions {
                        fg: Some(SlotValue::Token("success".to_string())),
                        bg: Some(SlotValue::Token("toolSuccessBg".to_string())),
                        attributes: TextAttributes {
                            bold: true,
                            ..Default::default()
                        },
                    }
                )
                .unwrap(),
            expected
        );
    }

    #[test]
    fn rejects_unknown_tokens_and_tokens_in_the_wrong_slot() {
        let _guard = theme_global_lock();
        let theme = load_theme_edited("dark", |_| {}).unwrap();
        assert_eq!(
            theme
                .style(
                    "x",
                    &StyleOptions {
                        fg: Some(SlotValue::Token("notAToken".to_string())),
                        ..Default::default()
                    }
                )
                .unwrap_err(),
            "Unknown theme color: notAToken"
        );
        assert_eq!(
            theme.fg("userMessageBg", "x").unwrap_err(),
            "Unknown theme color: userMessageBg"
        );
    }

    #[test]
    fn loads_oklch_and_okhsl_theme_values_including_through_variables() {
        let _guard = theme_global_lock();
        let theme = load_theme_edited("dark", |json| {
            json["colors"]["accent"] = json!("oklch(62% 0.1 200)");
        })
        .unwrap();
        assert_eq!(
            theme.color("accent").unwrap(),
            Color::Oklch {
                l: 0.62,
                c: 0.1,
                h: 200.0
            }
        );

        let theme = load_theme_edited("dark", |json| {
            json["vars"]["brand"] = json!("okhsl(250 60% 55%)");
            json["colors"]["accent"] = json!("brand");
            json["colors"]["error"] = json!("okhsl(20 90% 60%)");
        })
        .unwrap();
        let brand = parse_color("okhsl(250 60% 55%)").unwrap();
        assert_eq!(
            crate::utils::tui_compat::color_to_hex(&theme.color("accent").unwrap()),
            crate::utils::tui_compat::color_to_hex(&brand)
        );
        let error = parse_color("okhsl(20 90% 60%)").unwrap();
        assert_eq!(
            crate::utils::tui_compat::color_to_hex(&theme.color("error").unwrap()),
            crate::utils::tui_compat::color_to_hex(&error)
        );
    }

    #[test]
    fn detects_the_appearance_unless_it_is_declared() {
        let _guard = theme_global_lock();
        assert_eq!(
            load_theme_edited("dark", |_| {}).unwrap().appearance(),
            "dark"
        );
        assert_eq!(
            load_theme_edited("light", |_| {}).unwrap().appearance(),
            "light"
        );
        // Without a declaration, the appearance is detected from the theme's own colors.
        for base in ["dark", "light"] {
            let theme = load_theme_edited(base, |json| {
                json.as_object_mut().unwrap().remove("appearance");
            })
            .unwrap();
            assert_eq!(theme.appearance(), base);
        }
        let theme = load_theme_edited("dark", |json| {
            json["appearance"] = json!("light");
        })
        .unwrap();
        assert_eq!(theme.appearance(), "light");

        // Palette colors 0-15 follow the terminal palette, so such themes follow the terminal
        // background.
        let mut palette_json: serde_json::Value =
            serde_json::from_str(include_str!("dark.json")).unwrap();
        palette_json.as_object_mut().unwrap().remove("appearance");
        let keys: Vec<String> = palette_json["colors"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        for key in keys {
            palette_json["colors"][&key] = json!(if key.ends_with("Bg") { 0 } else { 7 });
        }
        let dir =
            std::env::temp_dir().join(format!("pi-theme-style-rs-{}-palette", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dark.json");
        std::fs::write(&path, serde_json::to_string(&palette_json).unwrap()).unwrap();
        let palette_only = load_theme_from_path(path.to_str().unwrap(), Some("truecolor")).unwrap();
        assert_eq!(palette_only.appearance(), "dark");
        set_terminal_colors(&TerminalColors {
            background: Some(rgb_from_hex("#fafafa")),
            ..Default::default()
        });
        assert_eq!(palette_only.appearance(), "light");
        set_terminal_colors(&TerminalColors::default());
    }

    #[test]
    fn renders_empty_tokens_as_terminal_defaults_and_reports_concrete_colors_for_them() {
        let _guard = theme_global_lock();
        let theme = load_theme_edited("dark", |json| {
            json["colors"]["text"] = json!("");
            json["colors"]["userMessageBg"] = json!("");
        })
        .unwrap();
        assert_eq!(theme.fg("text", "x").unwrap(), "\x1b[39mx\x1b[39m");
        assert_eq!(theme.bg("userMessageBg", "x").unwrap(), "\x1b[49mx\x1b[49m");
        assert_eq!(
            crate::utils::tui_compat::color_to_hex(&theme.color("text").unwrap()),
            "#e5e5e7"
        );
        assert_eq!(
            crate::utils::tui_compat::color_to_hex(&theme.color("userMessageBg").unwrap()),
            "#000000"
        );

        set_terminal_colors(&TerminalColors {
            foreground: Some(rgb_from_hex("#c8d2dc")),
            background: Some(rgb_from_hex("#0a141e")),
            ..Default::default()
        });
        assert_eq!(
            crate::utils::tui_compat::color_to_hex(&theme.color("text").unwrap()),
            "#c8d2dc"
        );
        assert_eq!(
            crate::utils::tui_compat::color_to_hex(&theme.color("userMessageBg").unwrap()),
            "#0a141e"
        );
        set_terminal_colors(&TerminalColors::default());
    }

    #[test]
    fn truecolor_and_256color_modes_render_exact_ansi_escapes() {
        let _guard = theme_global_lock();
        // Regression setup from upstream `resource-loader-theme.test.ts` #9973: userMessageBg is
        // #3c3544 and must render `\x1b[48;2;60;53;68m` in truecolor and `\x1b[48;5;59m` in 256
        // color mode.
        let theme = load_theme_edited("dark", |json| {
            json["name"] = json!("capability-test");
            json["colors"]["userMessageBg"] = json!("#3c3544");
        })
        .unwrap();
        assert_eq!(
            theme.bg("userMessageBg", "x").unwrap(),
            "\x1b[48;2;60;53;68mx\x1b[49m"
        );
        let theme_256 =
            load_theme_from_path(theme.source_path.as_deref().unwrap(), Some("256color")).unwrap();
        assert_eq!(
            theme_256.bg("userMessageBg", "x").unwrap(),
            "\x1b[48;5;59mx\x1b[49m"
        );
    }

    #[test]
    fn detect_color_fg_bg_classifies_the_last_field_by_palette_index_like_vim() {
        assert_eq!(detect_color_fg_bg_theme(Some("15;0")), Some("dark"));
        assert_eq!(detect_color_fg_bg_theme(Some("0;7;15")), Some("light"));
        // Solarized Dark's background is bright black.
        assert_eq!(detect_color_fg_bg_theme(Some("12;8")), Some("dark"));
        // rxvt writes "default" when the background is not a palette color.
        assert_eq!(detect_color_fg_bg_theme(Some("15;default")), None);
        assert_eq!(detect_color_fg_bg_theme(None), None);
    }

    #[test]
    fn detect_terminal_theme_prefers_background_then_scheme_then_colorfgbg_then_dark() {
        let env = Some("0;15");
        assert_eq!(
            detect_terminal_theme(
                &TerminalColors {
                    background: Some(rgb_from_hex("#080808")),
                    ..Default::default()
                },
                Some("light"),
                env
            ),
            "dark"
        );
        assert_eq!(
            detect_terminal_theme(&TerminalColors::default(), Some("dark"), env),
            "dark"
        );
        assert_eq!(
            detect_terminal_theme(&TerminalColors::default(), None, env),
            "light"
        );
        assert_eq!(
            detect_terminal_theme(&TerminalColors::default(), None, None),
            "dark"
        );
    }

    #[test]
    fn detect_terminal_theme_follows_the_foreground_when_text_is_readable_that_way() {
        let background = rgb_from_hex("#767676");
        assert_eq!(
            detect_terminal_theme(
                &TerminalColors {
                    background: Some(background),
                    ..Default::default()
                },
                None,
                None
            ),
            "light"
        );
        assert_eq!(
            detect_terminal_theme(
                &TerminalColors {
                    background: Some(background),
                    foreground: Some(rgb_from_hex("#ffffff")),
                    ..Default::default()
                },
                None,
                None
            ),
            "dark"
        );
        // White text cannot reach 4.5:1 on mid-gray.
        let mid_gray = rgb_from_hex("#808080");
        assert_eq!(
            detect_terminal_theme(
                &TerminalColors {
                    background: Some(mid_gray),
                    foreground: Some(rgb_from_hex("#ffffff")),
                    ..Default::default()
                },
                None,
                None
            ),
            "light"
        );
    }

    #[test]
    fn parses_and_resolves_automatic_theme_settings() {
        assert_eq!(
            parse_auto_theme_setting(Some("light/dark")),
            Some(("light".to_string(), "dark".to_string()))
        );
        assert_eq!(
            resolve_theme_setting(Some("dark"), "light").as_deref(),
            Some("dark")
        );
        assert_eq!(
            resolve_theme_setting(Some("light/dark"), "light").as_deref(),
            Some("light")
        );
        assert_eq!(
            resolve_theme_setting(Some("light/dark"), "dark").as_deref(),
            Some("dark")
        );
        assert_eq!(
            resolve_theme_setting(Some("light/dark/extra"), "dark"),
            None
        );
    }

    #[test]
    fn system_theme_is_listed_first_and_has_a_generated_name() {
        let _guard = theme_global_lock();
        let themes = get_available_themes();
        assert_eq!(themes[0], SYSTEM_THEME_NAME);
        assert!(themes.contains(&"dark".to_string()));
        assert!(themes.contains(&"light".to_string()));
    }
}
