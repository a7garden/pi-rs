//! Port of upstream `theme-json.ts` — theme JSON validation, kept out of
//! `theme.rs` on purpose.
//!
//! Validating user-authored theme files needs a schema engine (upstream pulls
//! in typebox, ~17 MB of module graph); palette lookup does not, so a
//! presentation that only uses built-in themes should never pay for it.
//! `interactive-mode` installs this validator; anything that does not simply
//! skips validation, as built-in themes already do.
//!
//! # Deviations
//!
//! - **D56b — hand-written schema validation replaces typebox.** The accepted
//!   grammar and the error-message shape (`Invalid theme "<label>":` +
//!   `Missing required color tokens:` list + `Other errors:` list, and the
//!   reserved-"/" name error) are 1:1 with upstream, but non-required typebox
//!   error wording is approximated (typebox messages are not a stable API).
//!   The validator seam takes a `fn` pointer instead of a JS closure.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde_json::Value;

/// One color value in a theme document: a string (hex/OKLCH/OKHSL/var ref/"" terminal default)
/// or a 256-color palette index.
#[derive(Debug, Clone, PartialEq)]
pub enum ThemeColorValue {
	Str(String),
	Index(u8),
}

impl ThemeColorValue {
	pub fn as_display(&self) -> String {
		match self {
			ThemeColorValue::Str(value) => value.clone(),
			ThemeColorValue::Index(index) => index.to_string(),
		}
	}
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ThemeExportSection {
	pub page_bg: Option<ThemeColorValue>,
	pub card_bg: Option<ThemeColorValue>,
	pub info_bg: Option<ThemeColorValue>,
}

/// The validated shape of one theme document (upstream `ValidatedThemeJson`).
#[derive(Debug, Clone, Default)]
pub struct ValidatedThemeJson {
	pub name: String,
	pub appearance: Option<String>,
	pub vars: Option<BTreeMap<String, ThemeColorValue>>,
	pub colors: BTreeMap<String, ThemeColorValue>,
	pub export: Option<ThemeExportSection>,
}

/// Color tokens that may be omitted (they get compatible fallbacks in `theme.rs`).
const OPTIONAL_COLORS: &[&str] = &[
	"scrollbarTrack",
	"scrollbarThumb",
	"searchMatchBg",
	"searchMatchText",
	"thinkingMax",
];

/// Required color tokens, in upstream schema order.
pub const REQUIRED_COLORS: &[&str] = &[
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
	"selectedBg",
	"userMessageBg",
	"userMessageText",
	"customMessageBg",
	"customMessageText",
	"customMessageLabel",
	"toolPendingBg",
	"toolSuccessBg",
	"toolErrorBg",
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
	"bashMode",
];

pub type ThemeJsonValidator = fn(label: &str, json: &Value) -> Result<ValidatedThemeJson, String>;

static THEME_JSON_VALIDATOR: LazyLock<parking_lot::Mutex<Option<ThemeJsonValidator>>> =
	LazyLock::new(|| parking_lot::Mutex::new(None));

/// Install full theme validation. Without it, documents are accepted as-is, which is what
/// built-in themes already do: validating user-authored JSON needs a schema engine, and a
/// presentation that only uses built-in themes should not pay for it.
pub fn set_theme_json_validator(validator: Option<ThemeJsonValidator>) {
	*THEME_JSON_VALIDATOR.lock() = validator;
}

/// The installed validator, if any (used by `theme.rs`'s `parseThemeJson`).
pub fn validator_slot() -> &'static parking_lot::Mutex<Option<ThemeJsonValidator>> {
	&THEME_JSON_VALIDATOR
}

fn parse_color_value(value: &Value) -> Result<ThemeColorValue, String> {
	match value {
		Value::String(text) => Ok(ThemeColorValue::Str(text.clone())),
		Value::Number(number) => {
			if let Some(index) = number.as_u64() {
				if index <= 255 {
					return Ok(ThemeColorValue::Index(index as u8));
				}
			}
			Err("expected a color value: a string or an integer from 0 to 255".to_string())
		}
		_ => Err("expected a color value: a string or an integer from 0 to 255".to_string()),
	}
}

fn parse_color_map(value: &Value) -> Result<BTreeMap<String, ThemeColorValue>, String> {
	let map = value
		.as_object()
		.ok_or_else(|| "expected an object".to_string())?;
	let mut result = BTreeMap::new();
	for (key, value) in map {
		result.insert(key.clone(), parse_color_value(value)?);
	}
	Ok(result)
}

/// Validate one theme document, returning an error message that names the offending tokens.
pub fn validate_theme_json(label: &str, json: &Value) -> Result<ValidatedThemeJson, String> {
	let mut missing_colors: Vec<&'static str> = Vec::new();
	let mut other_errors: Vec<String> = Vec::new();

	let object_error = |path: &str, message: &str| format!("  - {path}: {message}");

	let Some(object) = json.as_object() else {
		return Err(format!(
			"Invalid theme \"{label}\":\n\nOther errors:\n{}",
			object_error("/", "expected an object")
		));
	};

	// `$schema`: optional string.
	if let Some(schema) = object.get("$schema") {
		if schema.as_str().is_none() {
			other_errors.push(object_error("/$schema", "expected a string"));
		}
	}

	// `name`: required string.
	let name = match object.get("name") {
		Some(Value::String(name)) => name.clone(),
		Some(_) => {
			other_errors.push(object_error("/name", "expected a string"));
			String::new()
		}
		None => {
			missing_colors.push("name");
			String::new()
		}
	};

	// `appearance`: optional "dark" | "light".
	if let Some(appearance) = object.get("appearance") {
		match appearance.as_str() {
			Some("dark") | Some("light") => {}
			_ => other_errors.push(object_error("/appearance", "expected \"dark\" or \"light\"")),
		}
	}

	// `vars`: optional record of color values.
	let vars = match object.get("vars") {
		Some(Value::Object(_)) => Some(parse_color_map(object.get("vars").unwrap())?),
		Some(_) => {
			other_errors.push(object_error("/vars", "expected an object"));
			None
		}
		None => None,
	};

	// `colors`: required object of color values.
	let colors = match object.get("colors") {
		Some(value @ Value::Object(_)) => parse_color_map(value)?,
		Some(_) => {
			other_errors.push(object_error("/colors", "expected an object"));
			BTreeMap::new()
		}
		None => {
			missing_colors.extend(REQUIRED_COLORS);
			BTreeMap::new()
		}
	};
	for required in REQUIRED_COLORS {
		if !colors.contains_key(*required) {
			missing_colors.push(required);
		}
	}
	for key in colors.keys() {
		if !REQUIRED_COLORS.contains(&key.as_str()) && !OPTIONAL_COLORS.contains(&key.as_str()) {
			// Upstream's schema allows additional color properties; keep them.
			continue;
		}
	}

	// `export`: optional object with optional color values.
	let export = match object.get("export") {
		Some(Value::Object(export)) => {
			let mut section = ThemeExportSection::default();
			if let Some(value) = export.get("pageBg") {
				section.page_bg = Some(parse_color_value(value)?);
			}
			if let Some(value) = export.get("cardBg") {
				section.card_bg = Some(parse_color_value(value)?);
			}
			if let Some(value) = export.get("infoBg") {
				section.info_bg = Some(parse_color_value(value)?);
			}
			Some(section)
		}
		Some(_) => {
			other_errors.push(object_error("/export", "expected an object"));
			None
		}
		None => None,
	};

	if !missing_colors.is_empty() || !other_errors.is_empty() {
		missing_colors.sort_unstable();
		missing_colors.dedup();
		let mut error_message = format!("Invalid theme \"{label}\":\n");
		if !missing_colors.is_empty() {
			error_message += "\nMissing required color tokens:\n";
			error_message += &missing_colors
				.iter()
				.map(|color| format!("  - {color}"))
				.collect::<Vec<_>>()
				.join("\n");
			error_message += "\n\nPlease add these colors to your theme's \"colors\" object.";
			error_message += "\nSee the built-in themes (dark.json, light.json) for reference values.";
		}
		if !other_errors.is_empty() {
			error_message += &format!("\n\nOther errors:\n{}", other_errors.join("\n"));
		}
		return Err(error_message);
	}

	if name.contains('/') {
		return Err(format!(
			"Invalid theme name \"{name}\": theme names cannot contain \"/\" because it is reserved for automatic light/dark theme settings."
		));
	}

	let appearance = object.get("appearance").and_then(|value| value.as_str()).map(str::to_string);
	Ok(ValidatedThemeJson { name, appearance, vars, colors, export })
}

// ============================================================================
// Tests — 1:1 with the upstream theme-json validation contract
// ============================================================================

#[cfg(test)]
mod tests {
	use super::*;
	use serde_json::json;

	fn minimal_valid_theme() -> Value {
		let mut colors = serde_json::Map::new();
		for token in REQUIRED_COLORS {
			colors.insert(token.to_string(), json!("#000000"));
		}
		json!({ "name": "test", "colors": colors })
	}

	#[test]
	fn accepts_a_valid_theme_document() {
		let theme = validate_theme_json("t", &minimal_valid_theme()).unwrap();
		assert_eq!(theme.name, "test");
		assert_eq!(theme.appearance, None);
		assert_eq!(theme.colors.len(), REQUIRED_COLORS.len());
		assert_eq!(theme.colors["accent"], ThemeColorValue::Str("#000000".to_string()));
	}

	#[test]
	fn accepts_palette_indices_optional_colors_vars_and_export() {
		let mut json = minimal_valid_theme();
		json["colors"]["scrollbarTrack"] = json!(5);
		json["colors"]["scrollbarThumb"] = json!(7);
		json["colors"]["thinkingMax"] = json!(9);
		json["colors"]["searchMatchBg"] = json!(11);
		json["colors"]["searchMatchText"] = json!(13);
		json["appearance"] = json!("dark");
		json["vars"] = json!({ "brand": "okhsl(250 60% 55%)", "idx": 12 });
		json["export"] = json!({ "pageBg": "#ffffff" });
		let theme = validate_theme_json("t", &json).unwrap();
		assert_eq!(theme.appearance.as_deref(), Some("dark"));
		assert_eq!(theme.vars.as_ref().unwrap()["idx"], ThemeColorValue::Index(12));
		assert_eq!(theme.export.as_ref().unwrap().page_bg, Some(ThemeColorValue::Str("#ffffff".to_string())));
	}

	#[test]
	fn names_every_missing_required_color_token() {
		let mut json = minimal_valid_theme();
		json["colors"].as_object_mut().unwrap().remove("accent");
		json["colors"].as_object_mut().unwrap().remove("bashMode");
		json["colors"].as_object_mut().unwrap().remove("text");
		let error = validate_theme_json("my-theme", &json).unwrap_err();
		assert!(error.starts_with("Invalid theme \"my-theme\":\n"), "{error}");
		assert!(error.contains("Missing required color tokens:"), "{error}");
		assert!(error.contains("  - accent"), "{error}");
		assert!(error.contains("  - bashMode"), "{error}");
		assert!(error.contains("  - text"), "{error}");
		assert!(error.contains("Please add these colors to your theme's \"colors\" object."), "{error}");
		assert!(error.contains("See the built-in themes (dark.json, light.json) for reference values."), "{error}");
	}

	#[test]
	fn reports_a_colors_object_of_the_wrong_type() {
		let json = json!({ "name": "t", "colors": 42 });
		let error = validate_theme_json("t", &json).unwrap_err();
		assert!(error.contains("/colors"), "{error}");
	}

	#[test]
	fn rejects_out_of_range_palette_indices() {
		let mut json = minimal_valid_theme();
		json["colors"]["accent"] = json!(256);
		let error = validate_theme_json("t", &json).unwrap_err();
		assert!(!error.is_empty());
	}

	#[test]
	fn rejects_theme_names_containing_a_slash() {
		let mut json = minimal_valid_theme();
		json["name"] = json!("light/dark");
		let error = validate_theme_json("t", &json).unwrap_err();
		assert_eq!(
			error,
			"Invalid theme name \"light/dark\": theme names cannot contain \"/\" because it is reserved for automatic light/dark theme settings."
		);
	}

	#[test]
	fn validator_seam_installs_and_removes() {
		// The validator slot is global; serialize against other theme tests like upstream per-file vitest.
		let _guard = crate::theme::theme_global_lock();
		fn rejecting(_label: &str, _json: &Value) -> Result<ValidatedThemeJson, String> {
			Err("nope".to_string())
		}
		set_theme_json_validator(Some(rejecting));
		let dir = std::env::temp_dir().join(format!("pi-theme-validator-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let path = dir.join("seam.json");
		std::fs::write(&path, minimal_valid_theme().to_string()).unwrap();
		// The seam intercepts `parseThemeJson`, i.e. `loadThemeFromPath`, not the raw validator.
		assert_eq!(crate::theme::load_theme_from_path(path.to_str().unwrap(), None).unwrap_err(), "nope");
		set_theme_json_validator(None);
		assert!(crate::theme::load_theme_from_path(path.to_str().unwrap(), None).is_ok());
		let _ = std::fs::remove_file(&path);
	}
}
