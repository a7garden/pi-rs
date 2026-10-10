//! Port of upstream `system-theme.ts` — the `system` theme: pi's colors derived from the
//! terminal's own theme.
//!
//! Every token belongs to a color family (its hue) and has contrast rules: it must reach a
//! contrast level on the background and on the panels it is drawn on. Hue and saturation come
//! from the terminal's palette color for the family's ANSI slot, or from the family's own hue
//! when the terminal reports no palette. Lightness comes from the rules alone. Colors are built
//! in OKHSL, whose saturation is relative to the sRGB gamut, and fade toward gray near black and
//! white. A palette color never gains OKLCH chroma when it moves to another lightness, so pastel
//! palettes stay pastel.
//!
//! Depending on what the terminal reports, the theme is generated in one of three tiers:
//! - background and palette: hues from the palette, lightness from the background;
//! - background only: the families' own hues, lightness from the background;
//! - nothing: ANSI palette indices and the default colors, which the terminal renders itself.
#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::utils::tui_compat::{
    color_to_okhsl, color_to_oklch, color_to_rgb, okhsl_color, oklab_to_okhsl_lightness,
    oklch_color, parse_color, rgb_color, RgbColor,
};

use super::theme_json::ThemeColorValue;

pub const SYSTEM_THEME_NAME: &str = "system";

// ============================================================================
// Recipe: color families and their tokens
// ============================================================================

/// A family's OKHSL hue and saturation: `max` at mid lightness, falling toward `min` at black
/// and white.
#[derive(Debug, Clone, Copy)]
struct Family {
    hue: f64,
    saturation: (f64, f64), // (min, max)
    /// ANSI palette slot the family takes its hue and saturation from.
    slot: u8,
}

fn family(name: &str) -> Family {
    match name {
        "neutral" => Family {
            hue: 231.49,
            saturation: (0.02, 0.08),
            slot: 8,
        },
        "blue" => Family {
            hue: 231.49,
            saturation: (0.1, 0.68),
            slot: 4,
        },
        "green" => Family {
            hue: 158.68,
            saturation: (0.1, 0.76),
            slot: 2,
        },
        "red" => Family {
            hue: 20.0,
            saturation: (0.1, 0.92),
            slot: 1,
        },
        "yellow" => Family {
            hue: 82.36,
            saturation: (0.5, 1.0),
            slot: 3,
        },
        "orange" => Family {
            hue: 52.0,
            saturation: (0.12, 0.85),
            slot: 3,
        },
        "violet" => Family {
            hue: 295.0,
            saturation: (0.2, 0.6),
            slot: 5,
        },
        "calamine" => Family {
            hue: 202.43,
            saturation: (0.1, 0.74),
            slot: 6,
        },
        "thinkingSlate" => Family {
            hue: 231.49,
            saturation: (0.08, 0.2),
            slot: 4,
        },
        "thinkingBlue" => Family {
            hue: 231.49,
            saturation: (0.2, 0.45),
            slot: 4,
        },
        "thinkingPeriwinkle" => Family {
            hue: 263.25,
            saturation: (0.3, 0.6),
            slot: 6,
        },
        "thinkingViolet" => Family {
            hue: 295.0,
            saturation: (0.4, 0.75),
            slot: 5,
        },
        "thinkingMagenta" => Family {
            hue: 337.5,
            saturation: (0.5, 0.85),
            slot: 13,
        },
        "thinkingRed" => Family {
            hue: 20.0,
            saturation: (0.95, 1.0),
            slot: 1,
        },
        _ => unreachable!("unknown family {name}"),
    }
}

fn token_family(token: &str) -> &'static str {
    match token {
        "selectedBg" => "blue",
        "searchMatchBg" => "orange",
        "userMessageBg" => "blue",
        "customMessageBg" => "violet",
        "toolPendingBg" => "neutral",
        "toolSuccessBg" => "green",
        "toolErrorBg" => "red",

        "text" => "neutral",
        "userMessageText" => "neutral",
        "customMessageText" => "neutral",
        "toolTitle" => "neutral",
        "syntaxOperator" => "neutral",
        "syntaxPunctuation" => "neutral",
        "muted" => "neutral",
        "dim" => "neutral",
        "thinkingText" => "neutral",
        "toolOutput" => "neutral",
        "mdLinkUrl" => "neutral",
        "mdQuote" => "neutral",
        "mdQuoteBorder" => "neutral",
        "mdHr" => "neutral",
        "mdCodeBlockBorder" => "neutral",
        "toolDiffContext" => "neutral",
        "syntaxComment" => "neutral",
        "scrollbarTrack" => "neutral",
        "scrollbarThumb" => "neutral",
        "searchMatchText" => "neutral",
        "borderMuted" => "neutral",

        "accent" => "violet",
        "borderAccent" => "violet",
        "customMessageLabel" => "violet",
        "mdCode" => "violet",
        "mdListBullet" => "violet",
        "syntaxType" => "violet",
        "border" => "blue",
        "mdLink" => "blue",
        "syntaxKeyword" => "blue",
        "syntaxVariable" => "calamine",
        "success" => "green",
        "mdCodeBlock" => "green",
        "toolDiffAdded" => "green",
        "bashMode" => "green",
        "syntaxNumber" => "green",
        "error" => "red",
        "toolDiffRemoved" => "red",
        "warning" => "yellow",
        "mdHeading" => "yellow",
        "syntaxFunction" => "yellow",
        "syntaxString" => "orange",

        "thinkingOff" => "neutral",
        "thinkingMinimal" => "thinkingSlate",
        "thinkingLow" => "thinkingBlue",
        "thinkingMedium" => "thinkingPeriwinkle",
        "thinkingHigh" => "thinkingViolet",
        "thinkingXhigh" => "thinkingMagenta",
        "thinkingMax" => "thinkingRed",
        _ => unreachable!("unknown token {token}"),
    }
}

/// All tokens, in upstream `TOKEN_FAMILIES` order.
pub const TOKENS: &[&str] = &[
    "selectedBg",
    "searchMatchBg",
    "userMessageBg",
    "customMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
    "text",
    "userMessageText",
    "customMessageText",
    "toolTitle",
    "syntaxOperator",
    "syntaxPunctuation",
    "muted",
    "dim",
    "thinkingText",
    "toolOutput",
    "mdLinkUrl",
    "mdQuote",
    "mdQuoteBorder",
    "mdHr",
    "mdCodeBlockBorder",
    "toolDiffContext",
    "syntaxComment",
    "scrollbarTrack",
    "scrollbarThumb",
    "searchMatchText",
    "borderMuted",
    "accent",
    "borderAccent",
    "customMessageLabel",
    "mdCode",
    "mdListBullet",
    "syntaxType",
    "border",
    "mdLink",
    "syntaxKeyword",
    "syntaxVariable",
    "success",
    "mdCodeBlock",
    "toolDiffAdded",
    "bashMode",
    "syntaxNumber",
    "error",
    "toolDiffRemoved",
    "warning",
    "mdHeading",
    "syntaxFunction",
    "syntaxString",
    "thinkingOff",
    "thinkingMinimal",
    "thinkingLow",
    "thinkingMedium",
    "thinkingHigh",
    "thinkingXhigh",
    "thinkingMax",
];

/// Palette slots for tokens that would otherwise share a hue with a similar token.
fn token_slot(token: &str) -> Option<u8> {
    match token {
        "syntaxString" => Some(2),
        "syntaxNumber" => Some(5),
        "searchMatchBg" => Some(3),
        _ => None,
    }
}

// ============================================================================
// Contrast levels and rules
// ============================================================================

/// Target-lightness curves: a polynomial in the surface's OKLab lightness giving the OKLab
/// lightness a token needs on it. `reachable` is the range of surface lightness where the level
/// can be reached; beyond it the level is relaxed.
#[derive(Debug, Clone, Copy)]
struct Curve {
    coefficients: [f64; 6],
    reachable: (f64, f64),
}

fn level_curve(level: &str, appearance: &str) -> Curve {
    let dark = |coefficients: [f64; 6], reachable: (f64, f64)| Curve {
        coefficients,
        reachable,
    };
    match (level, appearance) {
        ("panel", "dark") => dark(
            [0.29131, -0.39746, 2.33185, -0.85524, -1.2076, 0.86276],
            (0.0, 0.979),
        ),
        ("panel", "light") => dark(
            [-3.74073, 27.94549, -78.44258, 112.6798, -79.60015, 22.11277],
            (0.348, 1.0),
        ),
        ("track", "dark") => dark(
            [0.39028, -0.23015, 0.83573, 2.43829, -4.38292, 2.01582],
            (0.0, 0.946),
        ),
        ("track", "light") => dark(
            [
                -5.24921, 38.37322, -107.28833, 152.10005, -106.17127, 29.18061,
            ],
            (0.368, 1.0),
        ),
        ("thinking0", "dark") => dark(
            [0.52988, -0.05809, -0.30924, 4.63567, -6.52933, 2.89108],
            (0.0, 0.873),
        ),
        ("thinking0", "light") => dark(
            [
                -28.27749, 182.85284, -469.62416, 603.15916, -384.59976, 97.35147,
            ],
            (0.51, 1.0),
        ),
        ("thinking1", "dark") => dark(
            [0.55278, -0.03667, -0.45659, 4.95347, -6.90265, 3.0706],
            (0.0, 0.858),
        ),
        ("thinking1", "light") => dark(
            [
                -37.10484, 235.86282, -596.62344, 754.3633, -474.00763, 118.3551,
            ],
            (0.535, 1.0),
        ),
        ("thinking2", "dark") => dark(
            [0.57486, -0.01765, -0.58987, 5.25227, -7.27175, 3.25532],
            (0.0, 0.842),
        ),
        ("thinking2", "light") => dark(
            [
                -59.89653, 377.05024, -945.07843, 1182.03145, -734.96375, 181.68658,
            ],
            (0.556, 1.0),
        ),
        ("thinking3", "dark") => dark(
            [0.59621, -0.00062, -0.71148, 5.53588, -7.6392, 3.44606],
            (0.0, 0.827),
        ),
        ("thinking3", "light") => dark(
            [
                -72.07122,
                445.84082,
                -1099.57352,
                1353.88793,
                -829.53392,
                202.26164,
            ],
            (0.58, 1.0),
        ),
        ("thinking4", "dark") => dark(
            [0.61691, 0.01462, -0.82288, 5.80651, -8.00641, 3.64333],
            (0.0, 0.811),
        ),
        ("thinking4", "light") => dark(
            [
                -110.14338,
                674.21488,
                -1645.75941,
                2004.32367,
                -1215.15899,
                293.3183,
            ],
            (0.6, 1.0),
        ),
        ("thinking5", "dark") => dark(
            [0.63702, 0.02826, -0.92498, 6.06465, -8.37246, 3.84651],
            (0.0, 0.795),
        ),
        ("thinking5", "light") => dark(
            [
                -175.47701,
                1063.54495,
                -2570.70594,
                3098.80776,
                -1860.15527,
                444.76392,
            ],
            (0.62, 1.0),
        ),
        ("thinking6", "dark") => dark(
            [0.65658, 0.04044, -1.01835, 6.30989, -8.73529, 4.05439],
            (0.0, 0.779),
        ),
        ("thinking6", "light") => dark(
            [
                -183.81712,
                1094.70055,
                -2602.68539,
                3088.71276,
                -1826.91131,
                430.75931,
            ],
            (0.643, 1.0),
        ),
        ("subtle", "dark") => dark(
            [0.56762, -0.02475, -0.5383, 5.12628, -7.10931, 3.17324],
            (0.0, 0.848),
        ),
        ("subtle", "light") => dark(
            [
                -232.85459,
                1376.54473,
                -3249.11801,
                3827.91186,
                -2248.29472,
                526.55751,
            ],
            (0.657, 1.0),
        ),
        ("thumb", "dark") => dark(
            [0.60323, 0.00278, -0.73328, 5.57157, -7.68067, 3.46933],
            (0.0, 0.823),
        ),
        ("thumb", "light") => dark(
            [
                -82.89897,
                511.01355,
                -1255.98095,
                1540.76821,
                -940.68087,
                228.58523,
            ],
            (0.586, 1.0),
        ),
        ("readable", "dark") => dark(
            [0.66937, 0.04704, -1.06871, 6.43941, -8.9332, 4.17229],
            (0.0, 0.77),
        ),
        ("readable", "light") => dark(
            [
                -1554.52576,
                8733.56817,
                -19604.93507,
                21977.72696,
                -12300.99599,
                2749.81288,
            ],
            (0.751, 1.0),
        ),
        ("emphasis", "dark") => dark(
            [0.7303, 0.07695, -1.31626, 7.1681, -10.14436, 4.92846],
            (0.0, 0.712),
        ),
        ("emphasis", "light") => dark(
            [
                -4948.31942,
                26870.91986,
                -58334.48399,
                63280.17197,
                -34298.01053,
                7430.30146,
            ],
            (0.811, 1.0),
        ),
        ("textOnPanel", "dark") => dark(
            [0.86713, 0.05232, -0.89428, 4.79014, -5.5432, 1.75023],
            (0.0, 0.542),
        ),
        ("textOnPanel", "light") => dark(
            [
                -8570.89457,
                43954.60805,
                -90084.00702,
                92220.6791,
                -47152.15802,
                9632.27113,
            ],
            (0.867, 1.0),
        ),
        ("text", "dark") => dark(
            [0.89242, 0.02311, -0.44862, 2.34417, -0.06084, -2.63844],
            (0.0, 0.5),
        ),
        ("text", "light") => dark(
            [
                -2004.67048,
                6664.47299,
                -6060.70202,
                -1792.61209,
                5133.82359,
                -1939.85583,
            ],
            (0.894, 1.0),
        ),
        _ => unreachable!("unknown level {level}/{appearance}"),
    }
}

const PANELS: &[&str] = &[
    "userMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
    "selectedBg",
    "searchMatchBg",
    "customMessageBg",
];

const THINKING: &[&str] = &[
    "thinkingOff",
    "thinkingMinimal",
    "thinkingLow",
    "thinkingMedium",
    "thinkingHigh",
    "thinkingXhigh",
    "thinkingMax",
];
const THINKING_LEVELS: &[&str] = &[
    "thinking0",
    "thinking1",
    "thinking2",
    "thinking3",
    "thinking4",
    "thinking5",
    "thinking6",
];

#[derive(Debug, Clone)]
struct Rule {
    token: &'static str,
    on: Vec<&'static str>,
    level: &'static str,
}

fn each(tokens: &[&'static str], on: &[&'static str], level: &'static str) -> Vec<Rule> {
    tokens
        .iter()
        .map(|token| Rule {
            token,
            on: on.to_vec(),
            level,
        })
        .collect()
}

fn rules() -> Vec<Rule> {
    let tool_panels = ["toolPendingBg", "toolSuccessBg", "toolErrorBg"];
    let mut result = Vec::new();
    result.extend(each(PANELS, &["background"], "panel"));
    result.push(Rule {
        token: "text",
        on: vec!["background"],
        level: "text",
    });
    result.push(Rule {
        token: "text",
        on: vec!["selectedBg"],
        level: "textOnPanel",
    });
    result.push(Rule {
        token: "userMessageText",
        on: vec!["userMessageBg"],
        level: "textOnPanel",
    });
    result.push(Rule {
        token: "toolTitle",
        on: tool_panels.to_vec(),
        level: "textOnPanel",
    });
    result.extend(each(
        &["accent", "success", "error", "warning"],
        &[
            "background",
            "selectedBg",
            "toolPendingBg",
            "toolSuccessBg",
            "toolErrorBg",
        ],
        "readable",
    ));
    result.push(Rule {
        token: "muted",
        on: vec![
            "background",
            "selectedBg",
            "customMessageBg",
            "toolPendingBg",
            "toolSuccessBg",
            "toolErrorBg",
        ],
        level: "readable",
    });
    result.push(Rule {
        token: "dim",
        on: vec![
            "background",
            "selectedBg",
            "customMessageBg",
            "toolPendingBg",
            "toolSuccessBg",
            "toolErrorBg",
        ],
        level: "subtle",
    });
    result.push(Rule {
        token: "thinkingText",
        on: vec!["background"],
        level: "readable",
    });
    result.push(Rule {
        token: "customMessageText",
        on: vec![
            "customMessageBg",
            "toolPendingBg",
            "toolSuccessBg",
            "toolErrorBg",
        ],
        level: "readable",
    });
    result.push(Rule {
        token: "customMessageLabel",
        on: vec![
            "background",
            "customMessageBg",
            "selectedBg",
            "toolPendingBg",
            "toolSuccessBg",
            "toolErrorBg",
        ],
        level: "readable",
    });
    result.push(Rule {
        token: "toolOutput",
        on: vec![
            "background",
            "toolPendingBg",
            "toolSuccessBg",
            "toolErrorBg",
        ],
        level: "readable",
    });
    result.extend(each(
        &[
            "mdHeading",
            "mdLink",
            "mdLinkUrl",
            "mdCode",
            "mdQuote",
            "mdCodeBlockBorder",
            "mdListBullet",
        ],
        &["background", "userMessageBg", "customMessageBg"],
        "readable",
    ));
    result.push(Rule {
        token: "mdCodeBlock",
        on: vec![
            "background",
            "userMessageBg",
            "customMessageBg",
            "toolPendingBg",
            "toolSuccessBg",
            "toolErrorBg",
        ],
        level: "readable",
    });
    result.extend(each(
        &["toolDiffAdded", "toolDiffRemoved", "toolDiffContext"],
        &[
            "background",
            "toolPendingBg",
            "toolSuccessBg",
            "toolErrorBg",
        ],
        "readable",
    ));
    result.extend(each(
        &[
            "syntaxComment",
            "syntaxKeyword",
            "syntaxFunction",
            "syntaxVariable",
            "syntaxString",
            "syntaxNumber",
            "syntaxType",
            "syntaxOperator",
            "syntaxPunctuation",
        ],
        &[
            "background",
            "userMessageBg",
            "customMessageBg",
            "toolPendingBg",
            "toolSuccessBg",
            "toolErrorBg",
        ],
        "readable",
    ));
    result.push(Rule {
        token: "searchMatchText",
        on: vec!["searchMatchBg"],
        level: "readable",
    });
    result.extend(each(
        &["bashMode", "border", "borderAccent"],
        &["background"],
        "readable",
    ));
    result.push(Rule {
        token: "borderMuted",
        on: vec!["background"],
        level: "subtle",
    });
    result.extend(each(
        &["mdQuoteBorder", "mdHr"],
        &[
            "background",
            "userMessageBg",
            "customMessageBg",
            "toolPendingBg",
            "toolSuccessBg",
            "toolErrorBg",
        ],
        "readable",
    ));
    result.push(Rule {
        token: "scrollbarTrack",
        on: vec!["background"],
        level: "track",
    });
    result.push(Rule {
        token: "scrollbarThumb",
        on: vec!["scrollbarTrack"],
        level: "thumb",
    });
    result.extend(
        THINKING
            .iter()
            .zip(THINKING_LEVELS)
            .map(|(token, level)| Rule {
                token,
                on: vec!["background"],
                level,
            }),
    );
    result
}

/// Tokens in dependency order: every surface before the tokens drawn on it.
fn solve_order() -> Vec<&'static str> {
    let all = rules();
    let mut order: Vec<&'static str> = Vec::new();
    fn visit(token: &'static str, all: &[Rule], order: &mut Vec<&'static str>) {
        if order.contains(&token) {
            return;
        }
        for rule in all {
            if rule.token != token {
                continue;
            }
            for surface in &rule.on {
                if *surface != "background" {
                    visit(surface, all, order);
                }
            }
        }
        order.push(token);
    }
    for rule in &all {
        visit(rule.token, &all, &mut order);
    }
    order
}

// ============================================================================
// Public API
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct SystemThemeInput {
    pub foreground: Option<RgbColor>,
    pub background: Option<RgbColor>,
    /// ANSI colors 0-15.
    pub palette: Option<Vec<RgbColor>>,
    /// Saturation multiplier from 0 (grayscale) to 1. The first frame renders in grayscale until
    /// colors arrive.
    pub saturation: Option<f64>,
    /// Appearance when the terminal did not report its background, e.g. from its light/dark
    /// report or COLORFGBG.
    pub appearance_hint: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct SystemThemeColors {
    /// Hex colors, ANSI palette indices, or "" for the terminal default.
    pub colors: BTreeMap<String, ThemeColorValue>,
    /// Foreground tokens rendered faint (SGR 2), for terminals that did not report colors.
    pub dim: Vec<String>,
    pub appearance: Option<&'static str>,
}

/// OKLab lightness of an sRGB color, 0-1.
fn oklab_lightness(color: &RgbColor) -> f64 {
    color_to_oklch(&rgb_color(color.r, color.g, color.b).unwrap()).0
}

/// WCAG 2 relative luminance.
pub fn relative_luminance(color: &RgbColor) -> f64 {
    let linear = |channel: f64| -> f64 {
        let value = channel / 255.0;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * linear(color.r) + 0.7152 * linear(color.g) + 0.0722 * linear(color.b)
}

/// WCAG 2 contrast ratio, 1-21.
pub fn wcag_contrast(first: &RgbColor, second: &RgbColor) -> f64 {
    let a = relative_luminance(first);
    let b = relative_luminance(second);
    (a.max(b) + 0.05) / (a.min(b) + 0.05)
}

/// Whether a terminal is dark or light, from its reported colors: the direction of its own
/// foreground when text can be readable that way, otherwise dark when white text has more
/// contrast on the background than black text.
pub fn terminal_appearance(background: &RgbColor, foreground: Option<&RgbColor>) -> &'static str {
    let white = RgbColor::new(255.0, 255.0, 255.0);
    let black = RgbColor::new(0.0, 0.0, 0.0);
    let white_contrast = wcag_contrast(&white, background);
    let black_contrast = wcag_contrast(&black, background);
    if let Some(foreground) = foreground {
        let foreground_l = oklab_lightness(&foreground);
        let background_l = oklab_lightness(background);
        if (foreground_l - background_l).abs() > 0.05 {
            let appearance = if foreground_l > background_l {
                "dark"
            } else {
                "light"
            };
            let best = if appearance == "dark" {
                white_contrast
            } else {
                black_contrast
            };
            if best >= TEXT_MINIMUM_WCAG_CONTRAST {
                return appearance;
            }
        }
    }
    if white_contrast >= black_contrast {
        "dark"
    } else {
        "light"
    }
}

// ============================================================================
// Generation
// ============================================================================

fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

fn hex_of(color: &RgbColor) -> String {
    let channel = |value: f64| format!("{:02x}", value.round() as i64);
    format!(
        "#{}{}{}",
        channel(color.r),
        channel(color.g),
        channel(color.b)
    )
}

/// Saturation weight at a lightness: a Gaussian (center 0.5, sigma 0.25), 0 at black and white,
/// 1 in the middle.
fn bell_weight(lightness: f64) -> f64 {
    let gaussian = |x: f64| (-(x - 0.5) * (x - 0.5) / (2.0 * 0.25 * 0.25)).exp();
    (gaussian(lightness) - gaussian(0.0)) / (1.0 - gaussian(0.0))
}

/// A family's saturation curve relative to its maximum: 1 at mid lightness, `min / max` at black
/// and white.
fn saturation_curve(family: &Family, lightness: f64) -> f64 {
    let (min, max) = family.saturation;
    let floor = if max > 0.0 { min / max } else { 1.0 };
    floor + (1.0 - floor) * bell_weight(lightness)
}

/// The target lightness for a level on a surface, or `None` where the level cannot be reached.
fn level_target(level: &str, appearance: &str, surface_l: f64) -> Option<f64> {
    let curve = level_curve(level, appearance);
    if surface_l < curve.reachable.0 || surface_l > curve.reachable.1 {
        return None;
    }
    Some(
        curve
            .coefficients
            .iter()
            .enumerate()
            .map(|(power, coefficient)| coefficient * surface_l.powi(power as i32))
            .sum(),
    )
}

/// A terminal color's OKHSL channels and its OKLCH chroma.
#[derive(Debug, Clone, Copy)]
struct SourceColor {
    h: f64,
    s: f64,
    l: f64,
    chroma: f64,
}

fn source_of(color: &RgbColor) -> SourceColor {
    let (h, s, l) = color_to_okhsl(&rgb_color(color.r, color.g, color.b).unwrap());
    let chroma = color_to_oklch(&rgb_color(color.r, color.g, color.b).unwrap()).1;
    SourceColor { h, s, l, chroma }
}

/// A source color's hue at another OKHSL lightness. Its saturation applies at its own lightness
/// and falls off toward black and white along the family's saturation curve, never rising above
/// it.
///
/// OKHSL saturation is relative to the most chroma sRGB allows at a lightness, so the same
/// saturation can mean more chroma elsewhere: Catppuccin Frappe's pink #f4b8e4 (chroma 0.089)
/// would become #eb76d1 (0.180) at the lightness the accent needs. Chroma is therefore also
/// capped at the source's, with the same falloff.
fn anchored(source: &SourceColor, family: &Family, lightness: f64, saturation: f64) -> RgbColor {
    let anchor = saturation_curve(family, source.l);
    let falloff = if anchor > 0.0 {
        (saturation_curve(family, lightness) / anchor).min(1.0)
    } else {
        1.0
    };
    let color = okhsl_color(source.h, source.s * falloff * saturation, lightness).unwrap();
    let cap = source.chroma * falloff * saturation;
    let (l, c, _) = color_to_oklch(&color);
    if c <= cap {
        color_to_rgb(&color)
    } else {
        color_to_rgb(&oklch_color(l, cap, source.h).unwrap())
    }
}

/// Move a text color toward white or black until it reaches the WCAG minimum on every surface.
fn with_text_contrast(color: &RgbColor, surfaces: &[RgbColor], lighter: bool) -> RgbColor {
    let meets = |candidate: &RgbColor| {
        surfaces
            .iter()
            .all(|surface| wcag_contrast(candidate, surface) >= TEXT_MINIMUM_WCAG_CONTRAST)
    };
    if meets(color) {
        return *color;
    }
    let rgb = rgb_color(color.r, color.g, color.b).unwrap();
    let (h, s, l) = color_to_okhsl(&rgb);
    let at = |lightness: f64| color_to_rgb(&okhsl_color(h, s, lightness).unwrap());
    let extreme = if lighter { 1.0 } else { 0.0 };
    if !meets(&at(extreme)) {
        return at(extreme);
    }
    let mut low = l;
    let mut high = extreme;
    for _ in 0..20 {
        let middle = (low + high) / 2.0;
        if meets(&at(middle)) {
            high = middle;
        } else {
            low = middle;
        }
    }
    at(high)
}

/// Colors for terminals that reported nothing: the terminal renders ANSI indices 0-15 and the
/// default colors with its own theme, so they fit any background. Neutral tokens below body text
/// are faint (SGR 2) instead of bright black, which some themes make nearly invisible. Panels
/// have no background.
fn indexed_colors(saturation: f64, appearance: Option<&'static str>) -> SystemThemeColors {
    let mut colors = BTreeMap::new();
    let mut dim = Vec::new();
    for token in TOKENS {
        if PANELS.contains(token) {
            colors.insert(token.to_string(), ThemeColorValue::Str(String::new()));
            continue;
        }
        let family_name = token_family(token);
        let neutral = family_name == "neutral";
        colors.insert(
            token.to_string(),
            if !neutral && saturation > 0.0 {
                ThemeColorValue::Index(
                    token_slot(token).unwrap_or_else(|| family(family_name).slot),
                )
            } else {
                ThemeColorValue::Str(String::new())
            },
        );
        if neutral && !FOREGROUND_TOKENS.contains(token) {
            dim.push(token.to_string());
        }
    }
    SystemThemeColors {
        colors,
        dim,
        appearance,
    }
}

/// Relaxation compresses levels stronger than this one toward it before weakening all levels.
fn readable_floor(appearance: &str) -> &'static str {
    if appearance == "dark" {
        "readable"
    } else {
        "subtle"
    }
}

/// Body text uses the terminal's foreground when it reaches this level, which is clearly stronger
/// than muted.
const FOREGROUND_LEVEL: &str = "emphasis";

/// Text-level tokens that take the terminal's foreground.
const FOREGROUND_TOKENS: &[&str] = &["text", "userMessageText", "toolTitle"];

/// WCAG 2 contrast ratio that body text must reach on the surfaces it is drawn on.
const TEXT_MINIMUM_WCAG_CONTRAST: f64 = 4.5;

/// Generate the system theme's colors from the terminal's reported colors.
pub fn generate_system_theme_colors(input: &SystemThemeInput) -> SystemThemeColors {
    let saturation = clamp(input.saturation.unwrap_or(1.0), 0.0, 1.0);
    let Some(background) = input.background else {
        return indexed_colors(saturation, input.appearance_hint);
    };
    let foreground = input.foreground;
    let palette: Option<Vec<SourceColor>> = input
        .palette
        .as_ref()
        .filter(|palette| palette.len() == 16)
        .map(|palette| palette.iter().map(source_of).collect());

    let appearance = terminal_appearance(&background, foreground.as_ref());
    let lighter = appearance == "dark";
    let extreme = if lighter { 1.0 } else { 0.0 };
    let background_l = oklab_lightness(&background);

    // A token's color at an OKLab lightness. With a palette, the palette color's saturation
    // applies at its own lightness and falls off toward black and white along the family's
    // curve, never rising above it.
    let paint = |token: &str, oklab_l: f64| -> RgbColor {
        let lightness = oklab_to_okhsl_lightness(oklab_l);
        let family = family(token_family(token));
        match &palette {
            None => {
                let (min, max) = family.saturation;
                color_to_rgb(
                    &okhsl_color(
                        family.hue,
                        (min + (max - min) * bell_weight(lightness)) * saturation,
                        lightness,
                    )
                    .unwrap(),
                )
            }
            Some(palette) => {
                let slot = token_slot(token).unwrap_or(family.slot);
                anchored(&palette[slot as usize], &family, lightness, saturation)
            }
        }
    };

    // The lightness a rule needs on a surface, relaxed by `t`: from 0 to 1, levels stronger than
    // the readable floor move toward it; from 1 to 2, all levels move toward the surface itself.
    let target = |level: &str, surface_l: f64, t: f64| -> Option<f64> {
        let reached = level_target(level, appearance, surface_l);
        if reached.is_none() && t == 0.0 {
            return None;
        }
        let distance = reached.unwrap_or(extreme) - surface_l;
        let floor = (level_target(readable_floor(appearance), appearance, surface_l)
            .unwrap_or(extreme))
            - surface_l;
        let compressed = if distance.abs() > floor.abs() {
            distance - (distance - floor) * t.min(1.0)
        } else {
            distance
        };
        Some(surface_l + compressed * (1.0 - (t - 1.0).max(0.0)))
    };

    // Keep a panel light enough (or dark enough) that white (or black) text still reaches the
    // body text minimum on it. This only matters for backgrounds near mid-gray, where it barely
    // does on the background.
    let extreme_text = if lighter {
        RgbColor::new(255.0, 255.0, 255.0)
    } else {
        RgbColor::new(0.0, 0.0, 0.0)
    };
    let readable =
        |color: &RgbColor| wcag_contrast(&extreme_text, color) >= TEXT_MINIMUM_WCAG_CONTRAST;
    let limit_panel = |token: &str, l: f64| -> RgbColor {
        let color = paint(token, l);
        if readable(&color) {
            return color;
        }
        let mut low = background_l;
        let mut high = l;
        for _ in 0..20 {
            let middle = (low + high) / 2.0;
            if readable(&paint(token, middle)) {
                low = middle;
            } else {
                high = middle;
            }
        }
        paint(token, low)
    };

    let order = solve_order();
    let all_rules = rules();
    let solve = |t: f64| -> Option<BTreeMap<String, RgbColor>> {
        let mut colors = BTreeMap::new();
        colors.insert("background".to_string(), background);
        for token in &order {
            let mut targets: Vec<f64> = Vec::new();
            for rule in &all_rules {
                if rule.token != *token {
                    continue;
                }
                for surface in &rule.on {
                    let surface_color = colors.get(*surface).unwrap_or(&background);
                    let value = target(rule.level, oklab_lightness(surface_color), t)?;
                    if !(0.0..=1.0).contains(&value) {
                        return None;
                    }
                    targets.push(value);
                }
            }
            if targets.is_empty() {
                continue;
            }
            let l = if lighter {
                targets.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            } else {
                targets.iter().cloned().fold(f64::INFINITY, f64::min)
            };
            let color = if PANELS.contains(token) {
                limit_panel(token, l)
            } else {
                paint(token, l)
            };
            colors.insert(token.to_string(), color);
        }
        Some(colors)
    };

    let mut relaxation = 0.0;
    let mut colors = solve(0.0);
    if colors.is_none() {
        // Mid-gray backgrounds cannot fit every level: relax as little as possible. Full
        // relaxation always fits.
        let mut low = 0.0;
        let mut high = 2.0;
        colors = solve(high);
        for _ in 0..20 {
            let middle = (low + high) / 2.0;
            let attempt = solve(middle);
            if attempt.is_some() {
                high = middle;
                colors = attempt;
            } else {
                low = middle;
            }
        }
        relaxation = high;
    }
    let solved = colors.unwrap_or_default();
    let surfaces_of = |token: &str| -> Vec<RgbColor> {
        all_rules
            .iter()
            .filter(|rule| rule.token == token)
            .flat_map(|rule| {
                rule.on
                    .iter()
                    .map(|surface| solved.get(*surface).unwrap_or(&background))
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .collect()
    };

    let mut result: BTreeMap<String, ThemeColorValue> = BTreeMap::new();
    for token in TOKENS {
        result.insert(
            token.to_string(),
            match solved.get(*token) {
                Some(color) => ThemeColorValue::Str(hex_of(color)),
                None => ThemeColorValue::Str(String::new()),
            },
        );
    }

    for token in FOREGROUND_TOKENS {
        let surfaces = surfaces_of(token);
        // Body text uses the terminal's own foreground where it is clearly stronger than muted
        // text; otherwise the foreground's hue at just enough lightness.
        let mut text = solved.get(*token).cloned();
        if let Some(foreground) = foreground {
            let targets: Vec<Option<f64>> = surfaces
                .iter()
                .map(|surface| target(FOREGROUND_LEVEL, oklab_lightness(surface), relaxation))
                .collect();
            if targets
                .iter()
                .all(|value| matches!(value, Some(value) if (0.0..=1.0).contains(value)))
            {
                let needed_values: Vec<f64> = targets.into_iter().flatten().collect();
                let needed = if lighter {
                    needed_values
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max)
                } else {
                    needed_values.iter().cloned().fold(f64::INFINITY, f64::min)
                };
                let foreground_l = oklab_lightness(&foreground);
                if (lighter && foreground_l >= needed) || (!lighter && foreground_l <= needed) {
                    result.insert(token.to_string(), ThemeColorValue::Str(String::new()));
                    continue;
                }
                text = Some(anchored(
                    &source_of(&foreground),
                    &family("neutral"),
                    oklab_to_okhsl_lightness(needed),
                    saturation,
                ));
            }
        }
        // Body text keeps at least 4.5:1 on the surfaces it is drawn on, even on relaxed mid-gray
        // backgrounds.
        if let Some(text) = text {
            result.insert(
                token.to_string(),
                ThemeColorValue::Str(hex_of(&with_text_contrast(&text, &surfaces, lighter))),
            );
        }
    }
    SystemThemeColors {
        colors: result,
        dim: Vec::new(),
        appearance: Some(appearance),
    }
}

/// Parse a hex string like upstream tests' `rgb()` helper.
pub fn rgb_from_hex(hex: &str) -> RgbColor {
    color_to_rgb(&parse_color(hex).expect("invalid hex"))
}

// ============================================================================
// Tests — 1:1 ports of upstream `system-theme.test.ts`
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::{
        get_theme_by_name, set_terminal_colors, theme_global_lock as system_theme_global_lock,
        SlotValue, StyleOptions,
    };
    use crate::utils::tui_compat::{
        color_to_hex, get_capabilities, get_terminal_color_mode, TerminalColors,
    };

    fn lightness(color: &RgbColor) -> f64 {
        color_to_oklch(&rgb_color(color.r, color.g, color.b).unwrap()).0
    }

    fn hue(color: &RgbColor) -> f64 {
        color_to_oklch(&rgb_color(color.r, color.g, color.b).unwrap()).2
    }

    fn chroma(color: &RgbColor) -> f64 {
        color_to_oklch(&rgb_color(color.r, color.g, color.b).unwrap()).1
    }

    fn dracula() -> SystemThemeInput {
        let base = [
            "#21222c", "#ff5555", "#50fa7b", "#f1fa8c", "#bd93f9", "#ff79c6", "#8be9fd", "#f8f8f2",
        ];
        let bright = [
            "#6272a4", "#ff6e6e", "#69ff94", "#ffffa5", "#d6acff", "#ff92df", "#a4ffff", "#ffffff",
        ];
        SystemThemeInput {
            background: Some(rgb_from_hex("#282a36")),
            foreground: Some(rgb_from_hex("#f8f8f2")),
            palette: Some(base.into_iter().chain(bright).map(rgb_from_hex).collect()),
            saturation: None,
            appearance_hint: None,
        }
    }

    /// Dark with a palette, light with an unreadable foreground, background only, and mid-gray.
    fn terminals() -> Vec<(&'static str, SystemThemeInput)> {
        vec![
            ("dracula", dracula()),
            (
                "solarizedLight",
                SystemThemeInput {
                    background: Some(rgb_from_hex("#fdf6e3")),
                    foreground: Some(rgb_from_hex("#657b83")),
                    ..Default::default()
                },
            ),
            (
                "backgroundOnly",
                SystemThemeInput {
                    background: Some(rgb_from_hex("#1e1e1e")),
                    ..Default::default()
                },
            ),
            (
                "midGray",
                SystemThemeInput {
                    background: Some(rgb_from_hex("#808080")),
                    foreground: Some(rgb_from_hex("#ffffff")),
                    ..Default::default()
                },
            ),
        ]
    }

    fn resolved(input: &SystemThemeInput, token: &str) -> RgbColor {
        let value = generate_system_theme_colors(&input).colors[token].as_display();
        if value.is_empty() {
            return if PANELS.contains(&token) {
                input.background.unwrap()
            } else {
                input.foreground.unwrap()
            };
        }
        color_to_rgb(&parse_color(&value).unwrap())
    }

    #[test]
    fn keeps_body_text_readable_on_background_and_panels() {
        let _guard = system_theme_global_lock();
        for (name, input) in terminals() {
            let text = resolved(&input, "text");
            for surface in [input.background.unwrap(), resolved(&input, "selectedBg")] {
                assert!(wcag_contrast(&text, &surface) >= 4.5, "{name}");
            }
            assert!(
                wcag_contrast(
                    &resolved(&input, "toolTitle"),
                    &resolved(&input, "toolErrorBg")
                ) >= 4.5,
                "{name}"
            );
        }
    }

    #[test]
    fn orders_foreground_roles_by_contrast_and_keeps_panels_close_to_the_background() {
        let _guard = system_theme_global_lock();
        for (name, input) in terminals() {
            let background = lightness(&input.background.unwrap());
            let offset = |token: &str| lightness(&resolved(&input, token)) - background;
            // On mid-gray the levels collapse to the strongest reachable color.
            if name != "midGray" {
                assert!(offset("text").abs() > offset("muted").abs(), "{name}");
                assert!(offset("muted").abs() > offset("dim").abs(), "{name}");
            }
            let lighter = generate_system_theme_colors(&input).appearance == Some("dark");
            for panel in [
                "userMessageBg",
                "toolPendingBg",
                "toolSuccessBg",
                "toolErrorBg",
                "selectedBg",
            ] {
                assert!(
                    wcag_contrast(&resolved(&input, panel), &input.background.unwrap()) < 2.0,
                    "{name} {panel}"
                );
                assert_eq!(offset(panel) > 0.0, lighter, "{name} {panel}");
            }
        }
    }

    #[test]
    fn uses_the_terminal_foreground_and_palette_hues() {
        let _guard = system_theme_global_lock();
        assert_eq!(
            generate_system_theme_colors(&dracula()).colors["text"].as_display(),
            ""
        );
        // Solarized's foreground is below 4.5:1 on its own background, so text is darkened.
        assert_ne!(
            generate_system_theme_colors(&terminals()[1].1).colors["text"].as_display(),
            ""
        );
        let error = resolved(&dracula(), "error");
        let palette_red = dracula().palette.unwrap()[1];
        assert!((hue(&error) - hue(&palette_red)).abs() < 8.0);
    }

    // https://github.com/earendil-works/pi/issues/10255
    #[test]
    fn keeps_pastel_palette_colors_pastel_at_other_lightnesses() {
        let _guard = system_theme_global_lock();
        let base = [
            "#51576d", "#e78284", "#a6d189", "#e5c890", "#8caaee", "#f4b8e4", "#81c8be", "#b5bfe2",
        ];
        let bright = [
            "#626880", "#e67172", "#8ec772", "#d9ba73", "#7b9ef0", "#f2a4db", "#5abfb5", "#a5adce",
        ];
        let frappe = SystemThemeInput {
            background: Some(rgb_from_hex("#303446")),
            foreground: Some(rgb_from_hex("#c6d0f5")),
            palette: Some(base.into_iter().chain(bright).map(rgb_from_hex).collect()),
            ..Default::default()
        };
        let pink = frappe.palette.as_ref().unwrap()[5];
        let accent = resolved(&frappe, "accent");
        // The accent is darker than the pink, but must not gain chroma (it was 2x before the cap).
        assert!(lightness(&accent) < lightness(&pink) - 0.05);
        assert!(chroma(&accent) <= chroma(&pink) * 1.03);
        for panel in ["userMessageBg", "customMessageBg"] {
            assert!(chroma(&resolved(&frappe, panel)) <= 0.1, "{panel}");
        }
    }

    #[test]
    fn renders_grayscale_at_zero_saturation() {
        let _guard = system_theme_global_lock();
        let input = SystemThemeInput {
            saturation: Some(0.0),
            ..dracula()
        };
        let colors = generate_system_theme_colors(&input).colors;
        let error = parse_color(&colors["error"].as_display()).unwrap();
        assert!(chroma(&color_to_rgb(&error)) < 0.005);
    }

    #[test]
    fn falls_back_to_palette_indices_and_faint_text_without_a_background() {
        let _guard = system_theme_global_lock();
        let generated = generate_system_theme_colors(&SystemThemeInput {
            appearance_hint: Some("light"),
            ..Default::default()
        });
        assert_eq!(generated.appearance, Some("light"));
        assert_eq!(
            (
                generated.colors["error"].as_display(),
                generated.colors["text"].as_display(),
                generated.colors["userMessageBg"].as_display()
            ),
            ("1".to_string(), String::new(), String::new())
        );
        assert!(generated.dim.contains(&"muted".to_string()));
        let grayscale = generate_system_theme_colors(&SystemThemeInput {
            saturation: Some(0.0),
            ..Default::default()
        });
        assert_eq!(grayscale.colors["error"].as_display(), "");
    }

    #[test]
    fn system_theme_is_generated_from_the_terminal_colors() {
        let _guard = system_theme_global_lock();
        assert_eq!(SYSTEM_THEME_NAME, "system");

        set_terminal_colors(&TerminalColors {
            foreground: Some(rgb_from_hex("#f8f8f2")),
            background: Some(rgb_from_hex("#282a36")),
            palette: Some(dracula().palette.unwrap()),
        });
        let theme = get_theme_by_name(SYSTEM_THEME_NAME).expect("system theme");
        assert_eq!(theme.appearance(), "dark");
        assert_eq!(theme.get_fg_ansi("text").unwrap(), "\x1b[39m");
        assert!(theme.get_fg_ansi("error").unwrap().starts_with("\x1b[38;"));
        // upstream afterEach: setTerminalColors({})
        set_terminal_colors(&TerminalColors::default());
    }

    #[test]
    fn system_theme_renders_faint_tokens_with_sgr_2() {
        let _guard = system_theme_global_lock();
        let theme = get_theme_by_name(SYSTEM_THEME_NAME).expect("system theme");
        assert_eq!(
            theme.fg("muted", "x").unwrap(),
            "\x1b[39m\x1b[2mx\x1b[22;39m"
        );
        assert_eq!(
            theme
                .style(
                    "x",
                    &StyleOptions {
                        fg: Some(SlotValue::Token("muted".to_string())),
                        ..Default::default()
                    }
                )
                .unwrap(),
            "\x1b[39m\x1b[2mx\x1b[22m\x1b[39m"
        );
    }

    #[test]
    fn color_mode_follows_capabilities() {
        let _guard = system_theme_global_lock();
        let mode = get_terminal_color_mode(get_capabilities());
        assert!(mode == "truecolor" || mode == "256color");
        // colorToHex sanity for the hex helper used across the suite
        assert_eq!(color_to_hex(&parse_color("#000000").unwrap()), "#000000");
    }
}
