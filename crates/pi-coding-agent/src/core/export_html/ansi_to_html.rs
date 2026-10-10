//! ANSI escape code to HTML converter.
//!
//! Port of upstream `packages/coding-agent/src/core/export-html/ansi-to-html.ts`.
//!
//! Converts terminal ANSI color/style codes to HTML with inline styles.
//! Supports:
//! - Standard foreground colors (30-37) and bright variants (90-97)
//! - Standard background colors (40-47) and bright variants (100-107)
//! - 256-color palette (38;5;N and 48;5;N)
//! - RGB true color (38;2;R;G;B and 48;2;R;G;B)
//! - Text styles: bold (1), dim (2), italic (3), underline (4)
//! - Reset (0)

/// Standard ANSI color palette (0-15)
const ANSI_COLORS: [&str; 16] = [
    "#000000", // 0: black
    "#800000", // 1: red
    "#008000", // 2: green
    "#808000", // 3: yellow
    "#000080", // 4: blue
    "#800080", // 5: magenta
    "#008080", // 6: cyan
    "#c0c0c0", // 7: white
    "#808080", // 8: bright black
    "#ff0000", // 9: bright red
    "#00ff00", // 10: bright green
    "#ffff00", // 11: bright yellow
    "#0000ff", // 12: bright blue
    "#ff00ff", // 13: bright magenta
    "#00ffff", // 14: bright cyan
    "#ffffff", // 15: bright white
];

/// Convert 256-color index to hex.
fn color256_to_hex(index: u32) -> String {
    // Standard colors (0-15)
    if index < 16 {
        return ANSI_COLORS[index as usize].to_string();
    }

    // Color cube (16-231): 6x6x6 = 216 colors
    if index < 232 {
        let cube_index = index - 16;
        let r = cube_index / 36;
        let g = (cube_index % 36) / 6;
        let b = cube_index % 6;
        let to_component = |n: u32| if n == 0 { 0 } else { 55 + n * 40 };
        let to_hex = |n: u32| format!("{:02x}", to_component(n));
        return format!("#{}{}{}", to_hex(r), to_hex(g), to_hex(b));
    }

    // Grayscale (232-255): 24 shades
    let gray = 8 + (index - 232) * 10;
    let gray_hex = format!("{:02x}", gray);
    format!("#{gray_hex}{gray_hex}{gray_hex}")
}

/// Escape HTML special characters (upstream `escapeHtml`).
pub fn escape_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#039;"),
            other => out.push(other),
        }
    }
    out
}

#[derive(Debug, Clone, Default, PartialEq)]
struct TextStyle {
    fg: Option<String>,
    bg: Option<String>,
    bold: bool,
    dim: bool,
    italic: bool,
    underline: bool,
}

fn style_to_inline_css(style: &TextStyle) -> String {
    let mut parts: Vec<&str> = Vec::new();
    let mut dynamic: Vec<String> = Vec::new();
    // Order matters for byte parity with upstream: color, background-color, then flags.
    if let Some(fg) = &style.fg {
        dynamic.push(format!("color:{fg}"));
    }
    if let Some(bg) = &style.bg {
        dynamic.push(format!("background-color:{bg}"));
    }
    if style.bold {
        parts.push("font-weight:bold");
    }
    if style.dim {
        parts.push("opacity:0.6");
    }
    if style.italic {
        parts.push("font-style:italic");
    }
    if style.underline {
        parts.push("text-decoration:underline");
    }
    let mut all = dynamic;
    all.extend(parts.into_iter().map(str::to_string));
    all.join(";")
}

fn has_style(style: &TextStyle) -> bool {
    style.fg.is_some()
        || style.bg.is_some()
        || style.bold
        || style.dim
        || style.italic
        || style.underline
}

/// Parse ANSI SGR (Select Graphic Rendition) codes and update style.
fn apply_sgr_code(params: &[u32], style: &mut TextStyle) {
    let mut i = 0;
    while i < params.len() {
        let code = params[i];

        if code == 0 {
            // Reset all
            *style = TextStyle::default();
        } else if code == 1 {
            style.bold = true;
        } else if code == 2 {
            style.dim = true;
        } else if code == 3 {
            style.italic = true;
        } else if code == 4 {
            style.underline = true;
        } else if code == 22 {
            // Reset bold/dim
            style.bold = false;
            style.dim = false;
        } else if code == 23 {
            style.italic = false;
        } else if code == 24 {
            style.underline = false;
        } else if (30..=37).contains(&code) {
            // Standard foreground colors
            style.fg = Some(ANSI_COLORS[(code - 30) as usize].to_string());
        } else if code == 38 {
            // Extended foreground color
            if params.get(i + 1) == Some(&5) && params.len() > i + 2 {
                // 256-color: 38;5;N
                style.fg = Some(color256_to_hex(params[i + 2]));
                i += 2;
            } else if params.get(i + 1) == Some(&2) && params.len() > i + 4 {
                // RGB: 38;2;R;G;B
                let (r, g, b) = (params[i + 2], params[i + 3], params[i + 4]);
                style.fg = Some(format!("rgb({r},{g},{b})"));
                i += 4;
            }
        } else if code == 39 {
            // Default foreground
            style.fg = None;
        } else if (40..=47).contains(&code) {
            // Standard background colors
            style.bg = Some(ANSI_COLORS[(code - 40) as usize].to_string());
        } else if code == 48 {
            // Extended background color
            if params.get(i + 1) == Some(&5) && params.len() > i + 2 {
                // 256-color: 48;5;N
                style.bg = Some(color256_to_hex(params[i + 2]));
                i += 2;
            } else if params.get(i + 1) == Some(&2) && params.len() > i + 4 {
                // RGB: 48;2;R;G;B
                let (r, g, b) = (params[i + 2], params[i + 3], params[i + 4]);
                style.bg = Some(format!("rgb({r},{g},{b})"));
                i += 4;
            }
        } else if code == 49 {
            // Default background
            style.bg = None;
        } else if (90..=97).contains(&code) {
            // Bright foreground colors
            style.fg = Some(ANS_COLORS_BRIGHT[(code - 90) as usize].to_string());
        } else if (100..=107).contains(&code) {
            // Bright background colors
            style.bg = Some(ANS_COLORS_BRIGHT[(code - 100) as usize].to_string());
        }
        // Ignore unrecognized codes

        i += 1;
    }
}

/// Bright palette entries (upstream indexes `ANSI_COLORS[8..16]`).
const ANS_COLORS_BRIGHT: [&str; 8] = [
    ANSI_COLORS[8],
    ANSI_COLORS[9],
    ANSI_COLORS[10],
    ANSI_COLORS[11],
    ANSI_COLORS[12],
    ANSI_COLORS[13],
    ANSI_COLORS[14],
    ANSI_COLORS[15],
];

/// One `ESC[<params>m` match: byte offset of the escape, its length, and the SGR
/// parameter string (upstream drives `ANSI_REGEX = /\x1b\[([\d;]*)m/g`).
struct SgrMatch {
    start: usize,
    end: usize,
    params: String,
}

/// Find the next SGR sequence at or after `from` (mirrors the global regex walk).
fn next_sgr(text: &str, from: usize) -> Option<SgrMatch> {
    let bytes = text.as_bytes();
    let mut i = from;
    while i + 1 < bytes.len() {
        if bytes[i] == 0x1b && bytes[i + 1] == b'[' {
            let mut j = i + 2;
            while j < bytes.len() && (bytes[j].is_ascii_digit() || bytes[j] == b';') {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'm' {
                return Some(SgrMatch {
                    start: i,
                    end: j + 1,
                    params: text[i + 2..j].to_string(),
                });
            }
        }
        i += 1;
    }
    None
}

/// Convert ANSI-escaped text to HTML with inline styles (upstream `ansiToHtml`).
pub fn ansi_to_html(text: &str) -> String {
    let mut style = TextStyle::default();
    let mut result = String::new();
    let mut last_index = 0;
    let mut in_span = false;

    let mut cursor = 0;
    while let Some(m) = next_sgr(text, cursor) {
        // Add text before this escape sequence
        let before_text = &text[last_index..m.start];
        if !before_text.is_empty() {
            result.push_str(&escape_html(before_text));
        }

        // Parse SGR parameters (`parseInt(p, 10) || 0`, empty param string → [0])
        let params: Vec<u32> = if m.params.is_empty() {
            vec![0]
        } else {
            m.params
                .split(';')
                .map(|p| p.parse::<u32>().unwrap_or(0))
                .collect()
        };

        // Close existing span if we have one
        if in_span {
            result.push_str("</span>");
            in_span = false;
        }

        // Apply the codes
        apply_sgr_code(&params, &mut style);

        // Open new span if we have any styling
        if has_style(&style) {
            result.push_str(&format!("<span style=\"{}\">", style_to_inline_css(&style)));
            in_span = true;
        }

        last_index = m.end;
        cursor = m.end;
    }

    // Add remaining text
    let remaining_text = &text[last_index..];
    if !remaining_text.is_empty() {
        result.push_str(&escape_html(remaining_text));
    }

    // Close any open span
    if in_span {
        result.push_str("</span>");
    }

    result
}

/// Convert an array of ANSI-escaped lines to HTML.
/// Each line is wrapped in a div element (upstream `ansiLinesToHtml`).
pub fn ansi_lines_to_html(lines: &[String]) -> String {
    lines
        .iter()
        .map(|line| {
            let inner = ansi_to_html(line);
            let body = if inner.is_empty() {
                "&nbsp;".to_string()
            } else {
                inner
            };
            format!("<div class=\"ansi-line\">{body}</div>")
        })
        .collect()
}
