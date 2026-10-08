//! Minimal port of the upstream `@earendil-works/pi-tui` helpers that the
//! theme foundation needs (upstream `packages/tui/src/{colors,oklab,terminal-colors,terminal-image}.ts`).
//!
//! These land in a future `pi-tui` crate; they live here temporarily so the
//! theme layer can render exact ANSI escapes without the full TUI. Only the
//! capability detection (`detectCapabilities`, `getCapabilities`,
//! `getTerminalColorMode`, `TerminalColorMode`, `setCapabilityOverrides`,
//! `resetCapabilitiesCache`) and the color conversion/rendering pipeline are
//! ported — components, layout, input, and image rendering are not.
//!
//! # Deviations
//!
//! - **D56 — PI_TRUE_COLOR / PI_HYPERLINKS / PI_IMAGE_PROTOCOL** behave
//!   exactly as upstream: `"1"` forces on, `"0"` forces off, anything else is
//!   ignored; `PI_TRUE_COLOR` overrides only the `trueColor` capability while
//!   the rest is auto-detected from the environment. The tmux OSC-8 probe
//!   shells out to `tmux display-message -p '#{client_termfeatures}'` with a
//!   250ms timeout, falling back to `false` on any error (upstream uses
//!   `execSync`).
//! - **D56a — statics use mutexes** instead of module-level `let` bindings;
//!   the JS single-threaded assumption becomes explicit locking.
#![allow(dead_code)]

use std::sync::LazyLock;

use parking_lot::Mutex as ParkingMutex;

// ============================================================================
// RgbColor (upstream terminal-colors.ts)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RgbColor {
	pub r: f64,
	pub g: f64,
	pub b: f64,
}

impl RgbColor {
	pub const fn new(r: f64, g: f64, b: f64) -> Self {
		RgbColor { r, g, b }
	}
}

pub type TerminalColorScheme = &'static str; // "dark" | "light"

/// Colors the terminal reports for its current theme.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TerminalColors {
	/// Default foreground (OSC 10).
	pub foreground: Option<RgbColor>,
	/// Default background (OSC 11).
	pub background: Option<RgbColor>,
	/// ANSI colors 0-15 (OSC 4). Only set when the terminal reported all 16.
	pub palette: Option<Vec<RgbColor>>,
}

// ============================================================================
// Color values (upstream colors.ts)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Color {
	Indexed {
		index: u8,
	},
	Rgb {
		r: f64,
		g: f64,
		b: f64,
	},
	Oklch {
		l: f64,
		c: f64,
		h: f64,
	},
}

pub type TerminalColorMode = &'static str; // "256color" | "truecolor"
pub const COLOR_MODE_256: &str = "256color";
pub const COLOR_MODE_TRUECOLOR: &str = "truecolor";

fn require_finite(value: f64, name: &str) -> Result<(), String> {
	if value.is_finite() {
		Ok(())
	} else {
		Err(format!("{name} must be finite"))
	}
}

pub fn indexed_color(index: u8) -> Color {
	Color::Indexed { index }
}

pub fn rgb_color(r: f64, g: f64, b: f64) -> Result<Color, String> {
	for (name, value) in [("r", r), ("g", g), ("b", b)] {
		require_finite(value, name)?;
		if !(0.0..=255.0).contains(&value) {
			return Err(format!("{name} must be between 0 and 255: {value}"));
		}
	}
	Ok(Color::Rgb { r, g, b })
}

pub fn oklch_color(l: f64, c: f64, h: f64) -> Result<Color, String> {
	require_finite(l, "l")?;
	require_finite(c, "c")?;
	require_finite(h, "h")?;
	if !(0.0..=1.0).contains(&l) {
		return Err(format!("l must be between 0 and 1: {l}"));
	}
	if c < 0.0 {
		return Err(format!("c must not be negative: {c}"));
	}
	Ok(Color::Oklch { l, c, h: h.rem_euclid(360.0) })
}

/// An OKHSL color, converted to sRGB. Saturation is relative to the sRGB gamut at the hue and
/// lightness, so equal saturation looks equally colorful across hues and lightness.
pub fn okhsl_color(h: f64, s: f64, l: f64) -> Result<Color, String> {
	require_finite(h, "h")?;
	require_finite(s, "s")?;
	require_finite(l, "l")?;
	if !(0.0..=1.0).contains(&s) {
		return Err(format!("s must be between 0 and 1: {s}"));
	}
	if !(0.0..=1.0).contains(&l) {
		return Err(format!("l must be between 0 and 1: {l}"));
	}
	let rgb = okhsl_to_rgb(h, s, l);
	rgb_color(rgb.r, rgb.g, rgb.b)
}

/// JS-style `parseFloat` over a token that may carry a `%` or `deg` suffix.
fn parse_number_token(token: &str) -> Option<f64> {
	let (value, _had_suffix) = strip_percent_or_deg(token)?;
	value.parse::<f64>().ok()
}

fn strip_percent_or_deg(token: &str) -> Option<(&str, bool)> {
	if let Some(rest) = token.strip_suffix('%') {
		return Some((rest, true));
	}
	if let Some(rest) = token.strip_suffix("deg") {
		return Some((rest, true));
	}
	Some((token, false))
}

/// Parse `#rgb`/`#rrggbb` hex, `oklch(L C H)` and `okhsl(H S L)` values (case-insensitive).
pub fn parse_color(value: &str) -> Result<Color, String> {
	// Number form: a palette index (matches upstream `typeof value === "number"` at call sites).
	if let Ok(index) = value.parse::<u8>() {
		return Ok(indexed_color(index));
	}

	let lower = value.to_ascii_lowercase();
	if let Some(hex) = lower.strip_prefix('#') {
		if hex.len() == 3 || hex.len() == 6 {
			let digits: String = if hex.len() == 3 {
				hex.chars().flat_map(|d| [d, d]).collect()
			} else {
				hex.to_string()
			};
			if digits.chars().all(|d| d.is_ascii_hexdigit()) {
				let r = u8::from_str_radix(&digits[0..2], 16).unwrap() as f64;
				let g = u8::from_str_radix(&digits[2..4], 16).unwrap() as f64;
				let b = u8::from_str_radix(&digits[4..6], 16).unwrap() as f64;
				return rgb_color(r, g, b);
			}
		}
		return Err(format!("Invalid color value: {value}"));
	}

	for (prefix, is_oklch) in [("oklch(", true), ("okhsl(", false)] {
		if lower.starts_with(prefix) && lower.ends_with(')') {
			let inner = value[prefix.len()..value.len() - 1].trim();
			let tokens: Vec<&str> = inner.split_whitespace().collect();
			if is_oklch && tokens.len() == 3 {
				let l_raw = tokens[0];
				let l = parse_number_token(l_raw).ok_or_else(|| format!("Invalid color value: {value}"))? / if l_raw.ends_with('%') { 100.0 } else { 1.0 };
				let c = parse_number_token(tokens[1]).ok_or_else(|| format!("Invalid color value: {value}"))?;
				let h = parse_number_token(tokens[2]).ok_or_else(|| format!("Invalid color value: {value}"))?;
				return oklch_color(l, c, h);
			}
			if !is_oklch && tokens.len() == 3 {
				let h = parse_number_token(tokens[0]).ok_or_else(|| format!("Invalid color value: {value}"))?;
				let s_raw = tokens[1];
				let s = parse_number_token(s_raw).ok_or_else(|| format!("Invalid color value: {value}"))? / if s_raw.ends_with('%') { 100.0 } else { 1.0 };
				let l_raw = tokens[2];
				let l = parse_number_token(l_raw).ok_or_else(|| format!("Invalid color value: {value}"))? / if l_raw.ends_with('%') { 100.0 } else { 1.0 };
				return okhsl_color(h, s, l);
			}
		}
	}

	Err(format!("Invalid color value: {value}"))
}

// ============================================================================
// Oklab / OKHSL (upstream oklab.ts — Björn Ottosson's reference implementation, MIT)
// ============================================================================

type Vector = [f64; 3];
type Matrix = [Vector; 3];

fn multiply(m: &Matrix, v: &Vector) -> Vector {
	[
		m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
		m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
		m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
	]
}

const LINEAR_SRGB_TO_LMS: Matrix = [
	[0.4122214694707629, 0.5363325372617349, 0.0514459932675022],
	[0.2119034958178251, 0.6806995506452344, 0.1073969535369405],
	[0.0883024591900564, 0.2817188391361215, 0.6299787016738222],
];
const LMS_TO_LAB: Matrix = [
	[0.210454268309314, 0.793617774702305, -0.0040720430116193],
	[1.9779985324311684, -2.42859224204858, 0.450593709617411],
	[0.0259040424655478, 0.7827717124575296, -0.8086757549230774],
];
const LAB_TO_LMS: Matrix = [
	[1.0, 0.3963377773761749, 0.2158037573099136],
	[1.0, -0.1055613458156586, -0.0638541728258133],
	[1.0, -0.0894841775298119, -1.2914855480194092],
];
const LMS_TO_LINEAR_SRGB: Matrix = [
	[4.0767416360759583, -3.3077115392580629, 0.2309699031821043],
	[-1.2684379732850315, 2.6097573492876882, -0.341319376002657],
	[-0.0041960761386756, -0.7034186179359362, 1.7076146940746117],
];
/// Per sRGB channel (red, green, blue): the (a, b) half-plane where that channel clips first,
/// and the polynomial approximating the maximum saturation there.
const SATURATION_FIT: [([f64; 2], [f64; 5]); 3] = [
	([-1.8817031, -0.80936501], [1.19086277, 1.76576728, 0.59662641, 0.75515197, 0.56771245]),
	([1.8144408, -1.19445267], [0.73956515, -0.45954404, 0.08285427, 0.12541073, -0.14503204]),
	([0.13110758, 1.81333971], [1.35733652, -0.00915799, -1.1513021, -0.50559606, 0.00692167]),
];
const K1: f64 = 0.206;
const K2: f64 = 0.03;
fn k3() -> f64 {
	(1.0 + K1) / (1.0 + K2)
}

/// Oklab lightness to OKHSL lightness.
pub fn oklab_to_okhsl_lightness(x: f64) -> f64 {
	let k3 = k3();
	0.5 * ((k3 * x - K1) + ((k3 * x - K1) * (k3 * x - K1) + 4.0 * K2 * k3 * x).sqrt())
}
/// OKHSL lightness to Oklab lightness.
fn okhsl_to_oklab_lightness(x: f64) -> f64 {
	let k3 = k3();
	(x * x + K1 * x) / (k3 * (x + K2))
}

/// sRGB transfer function: linear to encoded channel, both 0-1.
fn linear_to_srgb(value: f64) -> f64 {
	if value > 0.0031308 {
		1.055 * value.powf(1.0 / 2.4) - 0.055
	} else {
		12.92 * value
	}
}
/// Inverse sRGB transfer function: encoded to linear channel, both 0-1.
fn srgb_to_linear(value: f64) -> f64 {
	if value <= 0.04045 {
		value / 12.92
	} else {
		((value + 0.055) / 1.055).powf(2.4)
	}
}

/// Oklab [L, a, b] to linear sRGB [r, g, b] (0-1, may leave the gamut).
pub fn oklab_to_linear_srgb(lab: &Vector) -> Vector {
	let lms = multiply(&LAB_TO_LMS, lab);
	let cubed = [lms[0].powi(3), lms[1].powi(3), lms[2].powi(3)];
	multiply(&LMS_TO_LINEAR_SRGB, &cubed)
}

/// Linear sRGB [r, g, b] (0-1) to Oklab [L, a, b].
fn linear_srgb_to_oklab(rgb: &Vector) -> Vector {
	let lms = multiply(&LINEAR_SRGB_TO_LMS, rgb);
	let cbrt = [lms[0].cbrt(), lms[1].cbrt(), lms[2].cbrt()];
	multiply(&LMS_TO_LAB, &cbrt)
}

/// sRGB channels (0-255) to Oklab [L, a, b].
pub fn rgb_to_oklab(color: &RgbColor) -> Vector {
	let linear = [srgb_to_linear(color.r / 255.0), srgb_to_linear(color.g / 255.0), srgb_to_linear(color.b / 255.0)];
	linear_srgb_to_oklab(&linear)
}

/// Linear sRGB [r, g, b] to sRGB channels (0-255, rounded), clipping out-of-gamut channels.
pub fn linear_srgb_to_rgb(linear: &Vector) -> RgbColor {
	let channel = |value: f64| -> f64 { (linear_to_srgb(value).clamp(0.0, 1.0) * 255.0).round() };
	RgbColor::new(channel(linear[0]), channel(linear[1]), channel(linear[2]))
}

/// Rate of change of each cube-root LMS component along a chroma direction (a, b).
fn lms_slopes(a: f64, b: f64) -> Vector {
	[
		LAB_TO_LMS[0][1] * a + LAB_TO_LMS[0][2] * b,
		LAB_TO_LMS[1][1] * a + LAB_TO_LMS[1][2] * b,
		LAB_TO_LMS[2][1] * a + LAB_TO_LMS[2][2] * b,
	]
}

/// Largest saturation (C/L) inside sRGB for hue (a, b): polynomial fit plus one Halley step.
fn max_saturation(a: f64, b: f64) -> f64 {
	let mut channel = 2;
	for (index, entry) in SATURATION_FIT.iter().enumerate() {
		let ([x, y], _) = *entry;
		if index != 2 && x * a + y * b > 1.0 {
			channel = index;
			break;
		}
	}
	let fit = SATURATION_FIT[channel].1;
	let (k0, k1, k2, k3, k4) = (fit[0], fit[1], fit[2], fit[3], fit[4]);
	let weights = LMS_TO_LINEAR_SRGB[channel];
	let saturation = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;

	let slopes = lms_slopes(a, b);
	let base = [1.0 + saturation * slopes[0], 1.0 + saturation * slopes[1], 1.0 + saturation * slopes[2]];
	let dot = |values: &[f64; 3]| weights[0] * values[0] + weights[1] * values[1] + weights[2] * values[2];
	let cubed = [base[0].powi(3), base[1].powi(3), base[2].powi(3)];
	let f = dot(&cubed);
	let f1 = dot(&[3.0 * slopes[0] * base[0] * base[0], 3.0 * slopes[1] * base[1] * base[1], 3.0 * slopes[2] * base[2] * base[2]]);
	let f2 = dot(&[
		6.0 * slopes[0] * slopes[0] * base[0],
		6.0 * slopes[1] * slopes[1] * base[1],
		6.0 * slopes[2] * slopes[2] * base[2],
	]);
	saturation - (f * f1) / (f1 * f1 - 0.5 * f * f2)
}

/// Oklab lightness and chroma of the most saturated sRGB color of hue (a, b).
fn cusp(a: f64, b: f64) -> (f64, f64) {
	let saturation = max_saturation(a, b);
	let linear = oklab_to_linear_srgb(&[1.0, saturation * a, saturation * b]);
	let lightness = (1.0 / linear[0].max(linear[1].max(linear[2]))).cbrt();
	(lightness, lightness * saturation)
}

/// Chroma where the constant-lightness line at `lightness` leaves the sRGB gamut.
fn max_chroma(a: f64, b: f64, lightness: f64, cusp: (f64, f64)) -> f64 {
	let (cusp_l, cusp_c) = cusp;
	if lightness <= cusp_l {
		return (cusp_c * lightness) / cusp_l;
	}
	// Upper half: triangle edge, then one Halley step against each channel reaching 1.
	let t = (cusp_c * (lightness - 1.0)) / (cusp_l - 1.0);
	let slopes = lms_slopes(a, b);
	let lms = [lightness + t * slopes[0], lightness + t * slopes[1], lightness + t * slopes[2]];
	let cubes = [lms[0].powi(3), lms[1].powi(3), lms[2].powi(3)];
	let first = [3.0 * slopes[0] * lms[0] * lms[0], 3.0 * slopes[1] * lms[1] * lms[1], 3.0 * slopes[2] * lms[2] * lms[2]];
	let second = [
		6.0 * slopes[0] * slopes[0] * lms[0],
		6.0 * slopes[1] * slopes[1] * lms[1],
		6.0 * slopes[2] * slopes[2] * lms[2],
	];
	let dot = |row: &Vector, values: &[f64; 3]| row[0] * values[0] + row[1] * values[1] + row[2] * values[2];
	let mut min_step = f64::MAX;
	for row in &LMS_TO_LINEAR_SRGB {
		let f = dot(row, &cubes) - 1.0;
		let f1 = dot(row, &first);
		let f2 = dot(row, &second);
		let u = f1 / (f1 * f1 - 0.5 * f * f2);
		let step = if u >= 0.0 { -f * u } else { f64::MAX };
		min_step = min_step.min(step);
	}
	t + min_step
}

/// OKHSL's chroma reference points at lightness L and hue (a, b): [c0, cMid, cMax].
fn chroma_stops(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
	let peak = cusp(a, b);
	let c_max = max_chroma(a, b, l, peak);
	let k = c_max / (l * (peak.1 / peak.0)).min((1.0 - l) * (peak.1 / (1.0 - peak.0)));
	let mid_s = 0.11516993
		+ 1.0
			/ (7.4477897
				+ 4.1590124 * b
				+ a * (-2.19557347
					+ 1.75198401 * b
					+ a * (-2.13704948 - 10.02301043 * b + a * (-4.24894561 + 5.38770819 * b + 4.69891013 * a))));
	let mid_t = 0.11239642
		+ 1.0
			/ (1.6132032
				- 0.68124379 * b
				+ a * (0.40370612
					+ 0.90148123 * b
					+ a * (-0.27087943 + 0.6122399 * b + a * (0.00299215 - 0.45399568 * b - 0.14661872 * a))));
	let c_mid = 0.9 * k * ((1.0 / (1.0 / (l * mid_s).powi(4) + 1.0 / ((1.0 - l) * mid_t).powi(4))).sqrt().sqrt());
	let c0 = (1.0 / (1.0 / (l * 0.4).powi(2) + 1.0 / ((1.0 - l) * 0.8).powi(2))).sqrt();
	(c0, c_mid, c_max)
}

/// Convert OKHSL to sRGB channels (0-255, rounded), clipping out-of-gamut channels.
pub fn okhsl_to_rgb(hue: f64, saturation: f64, lightness: f64) -> RgbColor {
	let big_l = okhsl_to_oklab_lightness(lightness);
	let mut lab: Vector = [big_l, 0.0, 0.0];
	if big_l > 0.0 && big_l < 1.0 && saturation > 0.0 {
		let angle = 2.0 * std::f64::consts::PI * hue.rem_euclid(360.0) / 360.0;
		let a = angle.cos();
		let b = angle.sin();
		let (c0, c_mid, c_max) = chroma_stops(big_l, a, b);
		// Chroma rises from 0 through cMid at s = 0.8 to cMax at s = 1.
		let chroma;
		if saturation < 0.8 {
			let t = 1.25 * saturation;
			let k1 = 0.8 * c0;
			chroma = (t * k1) / (1.0 - (1.0 - k1 / c_mid) * t);
		} else {
			let t = 5.0 * (saturation - 0.8);
			let k1 = (0.2 * c_mid * c_mid * 1.25 * 1.25) / c0;
			chroma = c_mid + (t * k1) / (1.0 - (1.0 - k1 / (c_max - c_mid)) * t);
		}
		lab = [big_l, chroma * a, chroma * b];
	}
	linear_srgb_to_rgb(&oklab_to_linear_srgb(&lab))
}

/// Convert sRGB channels (0-255) to OKHSL. Hue `h` in degrees (0 for grays), saturation `s` and
/// lightness `l` 0-1.
pub fn rgb_to_okhsl(rgb: &RgbColor) -> (f64, f64, f64) {
	let lab = rgb_to_oklab(rgb);
	let chroma = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
	let lightness = oklab_to_okhsl_lightness(lab[0]);
	if chroma < 1e-9 || lightness <= 0.0 || lightness >= 1.0 {
		return (0.0, 0.0, lightness);
	}

	let hue = (lab[2].atan2(lab[1]) * 180.0 / std::f64::consts::PI).rem_euclid(360.0);
	let (c0, c_mid, c_max) = chroma_stops(lab[0], lab[1] / chroma, lab[2] / chroma);
	let saturation;
	if chroma < c_mid {
		let k1 = 0.8 * c0;
		saturation = 0.8 * (chroma / (k1 + (1.0 - k1 / c_mid) * chroma));
	} else {
		let k1 = (0.2 * c_mid * c_mid * 1.25 * 1.25) / c0;
		let offset = chroma - c_mid;
		saturation = 0.8 + 0.2 * (offset / (k1 + (1.0 - k1 / (c_max - c_mid)) * offset));
	}
	(hue, saturation.clamp(0.0, 1.0), lightness)
}

// ============================================================================
// Color conversion & ANSI rendering (upstream colors.ts)
// ============================================================================

const BASIC_COLORS: [RgbColor; 16] = [
	RgbColor::new(0.0, 0.0, 0.0),
	RgbColor::new(128.0, 0.0, 0.0),
	RgbColor::new(0.0, 128.0, 0.0),
	RgbColor::new(128.0, 128.0, 0.0),
	RgbColor::new(0.0, 0.0, 128.0),
	RgbColor::new(128.0, 0.0, 128.0),
	RgbColor::new(0.0, 128.0, 128.0),
	RgbColor::new(192.0, 192.0, 192.0),
	RgbColor::new(128.0, 128.0, 128.0),
	RgbColor::new(255.0, 0.0, 0.0),
	RgbColor::new(0.0, 255.0, 0.0),
	RgbColor::new(255.0, 255.0, 0.0),
	RgbColor::new(0.0, 0.0, 255.0),
	RgbColor::new(255.0, 0.0, 255.0),
	RgbColor::new(0.0, 255.0, 255.0),
	RgbColor::new(255.0, 255.0, 255.0),
];
const CUBE_VALUES: [f64; 6] = [0.0, 95.0, 135.0, 175.0, 215.0, 255.0];

fn gray_values() -> &'static Vec<f64> {
	static GRAY: LazyLock<Vec<f64>> = LazyLock::new(|| (0..24).map(|index| 8.0 + index as f64 * 10.0).collect());
	&GRAY
}

fn indexed_to_rgb(index: u8) -> RgbColor {
	if index < 16 {
		return BASIC_COLORS[index as usize];
	}
	if index < 232 {
		let cube_index = index as usize - 16;
		return RgbColor::new(
			CUBE_VALUES[cube_index / 36],
			CUBE_VALUES[(cube_index % 36) / 6],
			CUBE_VALUES[cube_index % 6],
		);
	}
	let gray = 8.0 + (index as f64 - 232.0) * 10.0;
	RgbColor::new(gray, gray, gray)
}

fn is_in_srgb_gamut(linear: &Vector) -> bool {
	let epsilon = 1e-7;
	linear.iter().all(|channel| *channel >= -epsilon && *channel <= 1.0 + epsilon)
}

fn oklch_to_rgb(l: f64, c: f64, h: f64) -> RgbColor {
	// Gamut mapping keeps the hue fixed, so its direction is computed once and scaled by chroma.
	let radians = h * std::f64::consts::PI / 180.0;
	let cos = radians.cos();
	let sin = radians.sin();
	let at_chroma = |chroma: f64| oklab_to_linear_srgb(&[l, chroma * cos, chroma * sin]);

	let direct = at_chroma(c);
	if is_in_srgb_gamut(&direct) {
		return linear_srgb_to_rgb(&direct);
	}

	// Reduce chroma until the color fits. The achromatic color is always in gamut, so it is the
	// fallback when no bisection step fits, e.g. `oklch(100% 0.3 150)` must map to white.
	let mut linear = at_chroma(0.0);
	let mut low = 0.0;
	let mut high = c;
	for _ in 0..20 {
		let chroma = (low + high) / 2.0;
		let candidate = at_chroma(chroma);
		if is_in_srgb_gamut(&candidate) {
			low = chroma;
			linear = candidate;
		} else {
			high = chroma;
		}
	}
	linear_srgb_to_rgb(&linear)
}

pub fn color_to_rgb(color: &Color) -> RgbColor {
	match *color {
		Color::Indexed { index } => indexed_to_rgb(index),
		Color::Rgb { r, g, b } => RgbColor::new(r, g, b),
		Color::Oklch { l, c, h } => oklch_to_rgb(l, c, h),
	}
}

pub fn color_to_oklch(color: &Color) -> (f64, f64, f64) {
	match *color {
		Color::Oklch { l, c, h } => (l, c, h),
		_ => {
			let lab = rgb_to_oklab(&color_to_rgb(color));
			let c = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
			let h = (lab[2].atan2(lab[1]) * 180.0 / std::f64::consts::PI).rem_euclid(360.0);
			(lab[0], c, h)
		}
	}
}

pub fn color_to_okhsl(color: &Color) -> (f64, f64, f64) {
	rgb_to_okhsl(&color_to_rgb(color))
}

pub fn color_to_hex(color: &Color) -> String {
	let rgb = color_to_rgb(color);
	let channel = |value: f64| format!("{:02x}", value.round() as u8);
	format!("#{}{}{}", channel(rgb.r), channel(rgb.g), channel(rgb.b))
}

pub fn mix_colors(first: &Color, second: &Color, amount: f64) -> Result<Color, String> {
	mix_colors_in(first, second, amount, "oklch")
}

pub fn mix_colors_in(first: &Color, second: &Color, amount: f64, space: &str) -> Result<Color, String> {
	require_finite(amount, "amount")?;
	if !(0.0..=1.0).contains(&amount) {
		return Err(format!("amount must be between 0 and 1: {amount}"));
	}

	if space == "srgb" {
		let a = color_to_rgb(first);
		let b = color_to_rgb(second);
		return rgb_color(
			a.r + (b.r - a.r) * amount,
			a.g + (b.g - a.g) * amount,
			a.b + (b.b - a.b) * amount,
		);
	}

	let (a_l, a_c, a_h) = color_to_oklch(first);
	let (b_l, b_c, b_h) = color_to_oklch(second);
	let first_hue = if a_c < 1e-7 { b_h } else { a_h };
	let second_hue = if b_c < 1e-7 { first_hue } else { b_h };
	let hue_delta = (second_hue - first_hue + 540.0).rem_euclid(360.0) - 180.0;
	oklch_color(a_l + (b_l - a_l) * amount, a_c + (b_c - a_c) * amount, first_hue + hue_delta * amount)
}

fn find_closest(values: &[f64], target: f64) -> usize {
	let mut closest_index = 0;
	let mut closest_distance = f64::INFINITY;
	for (index, value) in values.iter().enumerate() {
		let distance = (target - value).abs();
		if distance < closest_distance {
			closest_index = index;
			closest_distance = distance;
		}
	}
	closest_index
}

fn color_distance(first: &RgbColor, second: &RgbColor) -> f64 {
	let dr = first.r - second.r;
	let dg = first.g - second.g;
	let db = first.b - second.b;
	dr * dr * 0.299 + dg * dg * 0.587 + db * db * 0.114
}

fn rgb_to_ansi256(color: &RgbColor) -> u8 {
	let r_index = find_closest(&CUBE_VALUES, color.r);
	let g_index = find_closest(&CUBE_VALUES, color.g);
	let b_index = find_closest(&CUBE_VALUES, color.b);
	let cube_color = RgbColor::new(CUBE_VALUES[r_index], CUBE_VALUES[g_index], CUBE_VALUES[b_index]);
	let cube_index = 16 + 36 * r_index + 6 * g_index + b_index;

	let gray = (0.299 * color.r + 0.587 * color.g + 0.114 * color.b).round();
	let gray_offset = find_closest(gray_values(), gray);
	let gray_value = gray_values()[gray_offset];
	let spread = color.r.max(color.g).max(color.b) - color.r.min(color.g).min(color.b);
	if spread < 10.0
		&& color_distance(color, &RgbColor::new(gray_value, gray_value, gray_value))
			< color_distance(color, &cube_color)
	{
		return (232 + gray_offset) as u8;
	}
	cube_index as u8
}

fn color_ansi(color: &Color, mode: TerminalColorMode, background: bool) -> String {
	if let Color::Indexed { index } = color {
		return format!("\x1b[{};5;{}m", if background { 48 } else { 38 }, index);
	}

	let rgb = color_to_rgb(color);
	if mode == COLOR_MODE_TRUECOLOR {
		format!(
			"\x1b[{};2;{};{};{}m",
			if background { 48 } else { 38 },
			rgb.r.round() as i64,
			rgb.g.round() as i64,
			rgb.b.round() as i64
		)
	} else {
		format!("\x1b[{};5;{}m", if background { 48 } else { 38 }, rgb_to_ansi256(&rgb))
	}
}

pub fn foreground_ansi(color: &Color, mode: TerminalColorMode) -> String {
	color_ansi(color, mode, false)
}

pub fn background_ansi(color: &Color, mode: TerminalColorMode) -> String {
	color_ansi(color, mode, true)
}

/// Text attributes for [`style_text_with_ansi`].
#[derive(Debug, Clone, Default)]
pub struct TextAttributes {
	pub bold: bool,
	pub dim: bool,
	pub italic: bool,
	pub underline: bool,
	pub inverse: bool,
	pub strikethrough: bool,
}

/// Like upstream `styleTextWithAnsi()`: precomputed color escape sequences, e.g. cached theme colors.
pub fn style_text_with_ansi(text: &str, fg_ansi: Option<&str>, bg_ansi: Option<&str>, options: &TextAttributes) -> String {
	// Resets are prepended so they close in reverse order of the opening sequences.
	let mut prefix = String::new();
	let mut suffix = String::new();
	if let Some(fg) = fg_ansi {
		prefix += fg;
		suffix = "\x1b[39m".to_string();
	}
	if let Some(bg) = bg_ansi {
		prefix += bg;
		suffix = format!("\x1b[49m{suffix}");
	}
	if options.bold {
		prefix += "\x1b[1m";
	}
	if options.dim {
		prefix += "\x1b[2m";
	}
	if options.bold || options.dim {
		suffix = format!("\x1b[22m{suffix}");
	}
	if options.italic {
		prefix += "\x1b[3m";
		suffix = format!("\x1b[23m{suffix}");
	}
	if options.underline {
		prefix += "\x1b[4m";
		suffix = format!("\x1b[24m{suffix}");
	}
	if options.inverse {
		prefix += "\x1b[7m";
		suffix = format!("\x1b[27m{suffix}");
	}
	if options.strikethrough {
		prefix += "\x1b[9m";
		suffix = format!("\x1b[29m{suffix}");
	}
	format!("{prefix}{text}{suffix}")
}

// ============================================================================
// Terminal capabilities (upstream terminal-image.ts, capability subset)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageProtocol {
	Kitty,
	Iterm2,
	None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalCapabilities {
	pub images: Option<ImageProtocol>,
	pub true_color: bool,
	pub hyperlinks: bool,
}

struct CapabilityState {
	cached: Option<TerminalCapabilities>,
	overrides: TerminalCapabilitiesOverride,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TerminalCapabilitiesOverride {
	pub images: Option<Option<ImageProtocol>>,
	pub true_color: Option<bool>,
	pub hyperlinks: Option<bool>,
}

fn capability_state() -> &'static ParkingMutex<CapabilityState> {
	static STATE: LazyLock<ParkingMutex<CapabilityState>> =
		LazyLock::new(|| ParkingMutex::new(CapabilityState { cached: None, overrides: Default::default() }));
	&STATE
}

fn env_lower(key: &str) -> String {
	std::env::var(key).unwrap_or_default().to_lowercase()
}

/// Checks whether the attached tmux client forwards OSC 8 hyperlinks to the outer terminal.
/// tmux only re-emits them when its `client_termfeatures` lists `hyperlinks`, and strips them
/// otherwise. On any error falls back `false`.
fn probe_tmux_hyperlinks() -> bool {
	use std::process::{Command, Stdio};
	let output = Command::new("tmux")
		.args(["display-message", "-p", "#{client_termfeatures}"])
		.stdin(Stdio::null())
		.stderr(Stdio::null())
		.output();
	let Ok(output) = output else { return false };
	// Upstream uses execSync with a 250ms timeout; approximate with a wall-clock check is not
	// possible portably here, but tmux display-message returns immediately or fails.
	if !output.status.success() {
		return false;
	}
	String::from_utf8_lossy(&output.stdout)
		.split(',')
		.any(|feature| feature.trim() == "hyperlinks")
}

fn detect_capabilities_from_environment(tmux_forwards_hyperlink: impl FnOnce() -> bool) -> TerminalCapabilities {
	let term_program = env_lower("TERM_PROGRAM");
	let terminal_emulator = env_lower("TERMINAL_EMULATOR");
	let term = env_lower("TERM");
	let color_term = env_lower("COLORTERM");
	let has_true_color_hint =
		color_term == "truecolor" || color_term == "24bit" || term.ends_with("-direct");
	let is_windows_console = cfg!(windows);

	// Emit OSC 8 hyperlinks only when tmux confirms it forwards.
	// Image protocols are unreliable under tmux, so leave `images: None`.
	if std::env::var("TMUX").is_ok() || term.starts_with("tmux") {
		return TerminalCapabilities { images: None, true_color: has_true_color_hint, hyperlinks: tmux_forwards_hyperlink() };
	}

	// screen does not forward OSC 8 hyperlinks, so keep them off there.
	if term.starts_with("screen") {
		return TerminalCapabilities { images: None, true_color: has_true_color_hint, hyperlinks: false };
	}

	let kitty = TerminalCapabilities { images: Some(ImageProtocol::Kitty), true_color: true, hyperlinks: true };
	if std::env::var("KITTY_WINDOW_ID").is_ok() || term_program == "kitty" {
		return kitty;
	}

	if term_program == "ghostty" || term.contains("ghostty") || std::env::var("GHOSTTY_RESOURCES_DIR").is_ok() {
		return kitty;
	}

	if std::env::var("WEZTERM_PANE").is_ok() || term_program == "wezterm" {
		return kitty;
	}

	// Warp supports the Kitty graphics protocol and OSC 8 hyperlinks.
	if term_program == "warpterminal"
		|| std::env::var("WARP_SESSION_ID").is_ok()
		|| std::env::var("WARP_TERMINAL_SESSION_UUID").is_ok()
	{
		return kitty;
	}

	if std::env::var("ITERM_SESSION_ID").is_ok() || term_program == "iterm.app" {
		return TerminalCapabilities { images: Some(ImageProtocol::Iterm2), true_color: true, hyperlinks: true };
	}

	if std::env::var("WT_SESSION").is_ok() {
		return TerminalCapabilities { images: None, true_color: true, hyperlinks: true };
	}

	if term_program == "alacritty" || term_program == "vscode" || term_program == "zed" {
		return TerminalCapabilities { images: None, true_color: true, hyperlinks: true };
	}

	if terminal_emulator == "jetbrains-jediterm" {
		return TerminalCapabilities { images: None, true_color: true, hyperlinks: false };
	}

	// Windows Terminal does not always set WT_SESSION, for example when it hosts a cmd.exe launched
	// directly from Win+R. Modern Windows consoles support truecolor; keep hyperlinks off unless we
	// positively detected support above.
	if is_windows_console {
		return TerminalCapabilities { images: None, true_color: true, hyperlinks: false };
	}

	// Unknown terminal: be conservative. OSC 8 is rendered invisibly as "just text" on terminals
	// that swallow it, which means the URL disappears from the rendered output. Default to the
	// legacy `text (url)` behavior unless we have positively identified a hyperlink-capable
	// terminal above.
	TerminalCapabilities { images: None, true_color: has_true_color_hint, hyperlinks: false }
}

fn parse_boolean_capability_override(value: Option<String>) -> Option<bool> {
	match value.as_deref() {
		Some("1") => Some(true),
		Some("0") => Some(false),
		_ => None,
	}
}

pub fn detect_capabilities() -> TerminalCapabilities {
	let hyperlinks_override = parse_boolean_capability_override(std::env::var("PI_HYPERLINKS").ok());
	let hyperlinks_fn = || {
		match hyperlinks_override {
			Some(value) => value,
			None => probe_tmux_hyperlinks(),
		}
	};
	let detected = detect_capabilities_from_environment(hyperlinks_fn);
	let image_protocol = std::env::var("PI_IMAGE_PROTOCOL").ok().map(|v| v.to_lowercase());
	let images = match image_protocol.as_deref() {
		Some("kitty") => Some(Some(ImageProtocol::Kitty)),
		Some("iterm2") => Some(Some(ImageProtocol::Iterm2)),
		Some("none") | Some("0") => Some(None),
		_ => None,
	};
	let true_color = parse_boolean_capability_override(std::env::var("PI_TRUE_COLOR").ok());
	TerminalCapabilities {
		images: images.unwrap_or(detected.images),
		true_color: true_color.unwrap_or(detected.true_color),
		hyperlinks: hyperlinks_override.unwrap_or(detected.hyperlinks),
	}
}

pub fn get_capabilities() -> TerminalCapabilities {
	let mut state = capability_state().lock();
	if state.cached.is_none() {
		let hyperlinks = state.overrides.hyperlinks;
		let mut detected = detect_capabilities();
		if let Some(value) = hyperlinks {
			// An explicit hyperlink override also replaces the tmux probe, like upstream.
			detected.hyperlinks = value;
		}
		let overrides = state.overrides;
		if let Some(images) = overrides.images {
			detected.images = images;
		}
		if let Some(true_color) = overrides.true_color {
			detected.true_color = true_color;
		}
		if let Some(hyperlinks) = overrides.hyperlinks {
			detected.hyperlinks = hyperlinks;
		}
		state.cached = Some(detected);
	}
	state.cached.unwrap()
}

pub fn get_terminal_color_mode(capabilities: TerminalCapabilities) -> TerminalColorMode {
	if capabilities.true_color {
		COLOR_MODE_TRUECOLOR
	} else {
		COLOR_MODE_256
	}
}

/// Auto-detected terminal color mode (`getTerminalColorMode()` with default capabilities).
pub fn current_terminal_color_mode() -> TerminalColorMode {
	get_terminal_color_mode(get_capabilities())
}

pub fn reset_capabilities_cache() {
	capability_state().lock().cached = None;
}

/// Override selected auto-detected capabilities.
pub fn set_capability_overrides(overrides: TerminalCapabilitiesOverride) {
	let mut state = capability_state().lock();
	if state.overrides == overrides {
		return;
	}
	state.overrides = overrides;
	state.cached = None;
}

/// The upstream `setCapabilities()` test helper: set every capability and reset the cache.
pub fn set_capabilities(capabilities: TerminalCapabilities) {
	set_capability_overrides(TerminalCapabilitiesOverride {
		images: Some(capabilities.images),
		true_color: Some(capabilities.true_color),
		hyperlinks: Some(capabilities.hyperlinks),
	});
}

// ============================================================================
// Tests — ports of the upstream tui color tests exercised by the theme suite
// ============================================================================

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_hex_oklch_and_okhsl_colors() {
		assert_eq!(parse_color("#fff").unwrap(), Color::Rgb { r: 255.0, g: 255.0, b: 255.0 });
		assert_eq!(parse_color("#3c3544").unwrap(), Color::Rgb { r: 60.0, g: 53.0, b: 68.0 });
		assert_eq!(
			parse_color("oklch(62% 0.1 200)").unwrap(),
			Color::Oklch { l: 0.62, c: 0.1, h: 200.0 }
		);
		// okhsl parses and round-trips through sRGB
		let color = parse_color("okhsl(250 60% 55%)").unwrap();
		assert_eq!(color_to_hex(&color), color_to_hex(&okhsl_color(250.0, 0.6, 0.55).unwrap()));
		assert!(parse_color("nope").is_err());
		assert_eq!(parse_color("59").unwrap(), Color::Indexed { index: 59 });
	}

	#[test]
	fn renders_truecolor_and_256color_ansi() {
		let color = Color::Rgb { r: 60.0, g: 53.0, b: 68.0 };
		assert_eq!(background_ansi(&color, COLOR_MODE_TRUECOLOR), "\x1b[48;2;60;53;68m");
		// #3c3544 reduces to the 256-color cube index 59
		assert_eq!(background_ansi(&color, COLOR_MODE_256), "\x1b[48;5;59m");
		assert_eq!(foreground_ansi(&color, COLOR_MODE_TRUECOLOR), "\x1b[38;2;60;53;68m");
		assert_eq!(foreground_ansi(&Color::Indexed { index: 7 }, COLOR_MODE_256), "\x1b[38;5;7m");
	}

	#[test]
	fn style_text_with_ansi_closes_in_reverse_order() {
		let options = TextAttributes { bold: true, ..Default::default() };
		assert_eq!(
			style_text_with_ansi("x", Some("\x1b[38;2;1;2;3m"), Some("\x1b[48;5;59m"), &options),
			"\x1b[38;2;1;2;3m\x1b[48;5;59m\x1b[1mx\x1b[22m\x1b[49m\x1b[39m"
		);
	}

	#[test]
	fn pi_true_color_env_overrides_detection() {
		// SAFETY: tests run these env mutations serially (single-threaded intent); the theme test
		// suite serializes global-state tests through the same lock.
		std::env::set_var("PI_TRUE_COLOR", "1");
		reset_capabilities_cache();
		assert_eq!(get_terminal_color_mode(get_capabilities()), COLOR_MODE_TRUECOLOR);
		std::env::set_var("PI_TRUE_COLOR", "0");
		reset_capabilities_cache();
		assert_eq!(get_terminal_color_mode(get_capabilities()), COLOR_MODE_256);
		std::env::remove_var("PI_TRUE_COLOR");
		reset_capabilities_cache();
	}

	#[test]
	fn oklab_okhsl_round_trip_stays_in_gamut() {
		let rgb = okhsl_to_rgb(295.0, 0.5, 0.67);
		let (h, s, l) = rgb_to_okhsl(&rgb);
		assert!((h - 295.0).abs() < 3.0, "hue {h}");
		assert!((s - 0.5).abs() < 0.05, "saturation {s}");
		assert!((l - 0.67).abs() < 0.02, "lightness {l}");
		assert!(rgb.r >= 0.0 && rgb.r <= 255.0 && rgb.g >= 0.0 && rgb.g <= 255.0 && rgb.b >= 0.0 && rgb.b <= 255.0);
	}

	#[test]
	fn oklab_to_okhsl_lightness_matches_upstream_values() {
		assert!((oklab_to_okhsl_lightness(0.0) - 0.0).abs() < 1e-9);
		assert!((oklab_to_okhsl_lightness(1.0) - 1.0).abs() < 1e-9);
		// Inverts okhsl_to_oklab_lightness and stays monotonic.
		for x in [0.1, 0.25, 0.5, 0.75, 0.9] {
			let y = oklab_to_okhsl_lightness(x);
			assert!((okhsl_to_oklab_lightness(y) - x).abs() < 1e-9, "x={x} y={y}");
		}
		assert!(oklab_to_okhsl_lightness(0.25) < oklab_to_okhsl_lightness(0.5));
		assert!(oklab_to_okhsl_lightness(0.5) < oklab_to_okhsl_lightness(0.75));
	}

	#[test]
	fn indexed_palette_colors_convert_to_rgb() {
		assert_eq!(indexed_to_rgb(0), RgbColor::new(0.0, 0.0, 0.0));
		assert_eq!(indexed_to_rgb(8), RgbColor::new(128.0, 128.0, 128.0));
		assert_eq!(indexed_to_rgb(16), RgbColor::new(0.0, 0.0, 0.0));
		// 59 = 16 + 36*1 + 6*1 + 1 → the (95, 95, 95) cube corner.
		assert_eq!(indexed_to_rgb(59), RgbColor::new(95.0, 95.0, 95.0));
		assert_eq!(indexed_to_rgb(232), RgbColor::new(8.0, 8.0, 8.0));
		assert_eq!(indexed_to_rgb(255), RgbColor::new(238.0, 238.0, 238.0));
		// #3c3544 quantizes to palette index 59 (upstream resource-loader regression #9973).
		assert_eq!(rgb_to_ansi256(&RgbColor::new(60.0, 53.0, 68.0)), 59);
	}

	#[test]
	fn mix_colors_keeps_gray_acromatic() {
		let white = parse_color("#ffffff").unwrap();
		let black = parse_color("#000000").unwrap();
		let mixed = mix_colors(&white, &black, 0.5).unwrap();
		let (_, c, _) = color_to_oklch(&mixed);
		assert!(c < 1e-6);
	}

}
