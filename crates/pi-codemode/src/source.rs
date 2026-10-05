//! Mirrors upstream `src/source.ts`.
//!
//! Codemode source format: JavaScript, optionally preceded by one options line.
//!
//! ```js
//! // @options: {"max_output_tokens": 2000, "timeout_ms": 30000}
//! const text = await tools.read({ path: "package.json" });
//! text(JSON.parse(text).name);
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Prefix of the optional first-line options directive.
pub const CODEMODE_OPTIONS_PREFIX: &str = "// @options:";

const SUPPORTED_FIELDS: [&str; 2] = ["max_output_tokens", "timeout_ms"];
const SUPPORTED_FIELDS_TEXT: &str = "`max_output_tokens` and `timeout_ms`";
/// Largest delay `setTimeout` supports, which bounds `timeout_ms`.
const MAX_TIMEOUT_MS: f64 = 2_147_483_647.0;

/// Lark grammar for providers with grammar-constrained tool input. It only fixes the shape of the
/// options line; the options JSON and the code are checked by [`parse_codemode_source`].
pub const CODEMODE_SOURCE_GRAMMAR: &str = r"
start: options_source | plain_source
options_source: OPTIONS_LINE NEWLINE SOURCE
plain_source: SOURCE

OPTIONS_LINE: /[ \t]*\/\/ @options:[^\r\n]*/
NEWLINE: /\r?\n/
SOURCE: /[\s\S]+/
";

/// Error raised for empty input and invalid options. The `Display` message matches the
/// upstream `CodemodeSourceError` strings used in tests.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct CodemodeSourceError {
    pub message: String,
}

impl CodemodeSourceError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

/// Per-script options parsed from the options line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodemodeSourceOptions {
    /// Token budget for the script's output.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub max_output_tokens: Option<u64>,
    /// Hard deadline for the whole script in milliseconds, including tool calls.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub timeout_ms: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedCodemodeSource {
    /// The script with the options line replaced by an empty line, so line numbers are unchanged.
    pub code: String,
    pub options: CodemodeSourceOptions,
}

/// Upstream `Number.isSafeInteger(value) && value >= 0`. The value arrives from
/// `serde_json`, so it is already a finite number; integers beyond the safe range
/// or fractional values are rejected.
fn is_safe_integer(value: &Value) -> bool {
    match value {
        Value::Number(n) => {
            let v = n.as_f64().unwrap_or(f64::NAN);
            v.fract() == 0.0 && v.abs() <= 9_007_199_254_740_991.0
        }
        _ => false,
    }
}

fn parse_options(directive: &str) -> Result<CodemodeSourceOptions, CodemodeSourceError> {
    if directive.is_empty() {
        return Err(CodemodeSourceError::new(format!(
            "@options must be a JSON object with supported fields {SUPPORTED_FIELDS_TEXT}"
        )));
    }
    let value: Value = serde_json::from_str(directive).map_err(|error| {
        CodemodeSourceError::new(format!(
            "@options must be valid JSON with supported fields {SUPPORTED_FIELDS_TEXT}: {error}"
        ))
    })?;
    let Value::Object(fields) = value else {
        return Err(CodemodeSourceError::new(format!(
            "@options must be a JSON object with supported fields {SUPPORTED_FIELDS_TEXT}"
        )));
    };
    for key in fields.keys() {
        if !SUPPORTED_FIELDS.contains(&key.as_str()) {
            return Err(CodemodeSourceError::new(format!(
                "@options only supports {SUPPORTED_FIELDS_TEXT}; got `{key}`"
            )));
        }
    }
    let mut options = CodemodeSourceOptions::default();
    if let Some(max_output_tokens) = fields.get("max_output_tokens") {
        if max_output_tokens.is_null() {
            // JSON `undefined` has no representation; upstream only sets the field when present,
            // and `null` is not a safe integer, so it is rejected like any other bad value.
            return Err(CodemodeSourceError::new(
                "@options field `max_output_tokens` must be a non-negative safe integer",
            ));
        }
        if !is_safe_integer(max_output_tokens) {
            return Err(CodemodeSourceError::new(
                "@options field `max_output_tokens` must be a non-negative safe integer",
            ));
        }
        let value = max_output_tokens.as_f64().unwrap();
        if value < 0.0 {
            return Err(CodemodeSourceError::new(
                "@options field `max_output_tokens` must be a non-negative safe integer",
            ));
        }
        options.max_output_tokens = Some(value as u64);
    }
    if let Some(timeout_ms) = fields.get("timeout_ms") {
        let valid = is_safe_integer(timeout_ms)
            && timeout_ms.as_f64().map_or(false, |v| v > 0.0 && v <= MAX_TIMEOUT_MS);
        if !valid {
            return Err(CodemodeSourceError::new(format!(
                "@options field `timeout_ms` must be a positive integer up to {MAX_TIMEOUT_MS}"
            )));
        }
        options.timeout_ms = Some(timeout_ms.as_f64().unwrap() as u32);
    }
    Ok(options)
}

/// Split an optional first-line `// @options: {...}` from the script. Throws
/// [`CodemodeSourceError`] for empty input and invalid options.
pub fn parse_codemode_source(input: &str) -> Result<ParsedCodemodeSource, CodemodeSourceError> {
    if input.trim().is_empty() {
        return Err(CodemodeSourceError::new(
            "Expected JavaScript source text (non-empty). Provide JS only, optionally with a first line `// @options: {\"max_output_tokens\": 1000}`.",
        ));
    }
    let newline = input.find('\n');
    let first_line_full = match newline {
        Some(index) => &input[..index],
        None => input,
    };
    let first_line = first_line_full.strip_suffix('\r').unwrap_or(first_line_full);
    let trimmed = first_line.trim_start();
    if !trimmed.starts_with(CODEMODE_OPTIONS_PREFIX) {
        return Ok(ParsedCodemodeSource { code: input.to_string(), options: CodemodeSourceOptions::default() });
    }
    let code = match newline {
        Some(index) => input[index..].to_string(),
        None => String::new(),
    };
    if code.trim().is_empty() {
        return Err(CodemodeSourceError::new(
            "The @options line must be followed by JavaScript source on subsequent lines",
        ));
    }
    let directive = trimmed[CODEMODE_OPTIONS_PREFIX.len()..].trim();
    Ok(ParsedCodemodeSource { code, options: parse_options(directive)? })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_plain_code_unchanged() {
        let parsed = parse_codemode_source("text('hi')").unwrap();
        assert_eq!(parsed.code, "text('hi')");
        assert_eq!(parsed.options, CodemodeSourceOptions::default());
        let parsed = parse_codemode_source("// just a comment\nreturn 1").unwrap();
        assert_eq!(parsed.code, "// just a comment\nreturn 1");
        assert_eq!(parsed.options, CodemodeSourceOptions::default());
    }

    #[test]
    fn parses_the_options_line_and_keeps_line_numbers() {
        let parsed = parse_codemode_source("// @options: {\"timeout_ms\": 10}\nconst a = 1;\ntext(a)").unwrap();
        assert_eq!(parsed.code, "\nconst a = 1;\ntext(a)");
        assert_eq!(parsed.options.timeout_ms, Some(10));
        assert_eq!(parsed.options.max_output_tokens, None);

        let parsed =
            parse_codemode_source("  // @options:{\"max_output_tokens\":0,\"timeout_ms\":1500}\r\ntext(1)").unwrap();
        assert_eq!(parsed.options.max_output_tokens, Some(0));
        assert_eq!(parsed.options.timeout_ms, Some(1500));

        let parsed = parse_codemode_source("// @options: {}\ntext(1)").unwrap();
        assert_eq!(parsed.code, "\ntext(1)");
        assert_eq!(parsed.options, CodemodeSourceOptions::default());
    }

    #[test]
    fn only_treats_the_first_line_as_an_options_line() {
        let input = "text(1)\n// @options: {\"timeout_ms\": 1}";
        let parsed = parse_codemode_source(input).unwrap();
        assert_eq!(parsed.code, input);
        assert_eq!(parsed.options, CodemodeSourceOptions::default());
        assert_eq!(parse_codemode_source("// @optionsx {}\ntext(1)").unwrap().options, CodemodeSourceOptions::default());
    }

    #[test]
    fn rejects_empty_input_and_invalid_options() {
        // (input, exact message); the JSON-parse case keeps only the shared prefix because
        // the engine-specific suffix differs (serde_json vs V8).
        let cases: Vec<(&str, &str, bool)> = vec![
            ("", "Expected JavaScript source text (non-empty). Provide JS only, optionally with a first line `// @options: {\"max_output_tokens\": 1000}`.", true),
            ("  \n", "Expected JavaScript source text (non-empty). Provide JS only, optionally with a first line `// @options: {\"max_output_tokens\": 1000}`.", true),
            ("// @options:\ntext(1)", "@options must be a JSON object with supported fields `max_output_tokens` and `timeout_ms`", true),
            ("// @options: {timeout_ms: 1}\ntext(1)", "@options must be valid JSON with supported fields", false),
            ("// @options: [1]\ntext(1)", "@options must be a JSON object with supported fields `max_output_tokens` and `timeout_ms`", true),
            ("// @options: {\"yield\": 1}\ntext(1)", "@options only supports `max_output_tokens` and `timeout_ms`; got `yield`", true),
            ("// @options: {\"max_output_tokens\": 1.5}\ntext(1)", "@options field `max_output_tokens` must be a non-negative safe integer", true),
            ("// @options: {\"timeout_ms\": 0}\ntext(1)", "@options field `timeout_ms` must be a positive integer up to 2147483647", true),
            ("// @options: {\"timeout_ms\": 1}", "The @options line must be followed by JavaScript source on subsequent lines", true),
            ("// @options: {\"timeout_ms\": 1}\n  \n", "The @options line must be followed by JavaScript source on subsequent lines", true),
        ];
        for (input, message, exact) in cases {
            let error = parse_codemode_source(input).unwrap_err();
            if exact {
                assert_eq!(error.message, message, "input: {input:?}");
            } else {
                assert!(error.message.contains(message), "input: {input:?}, got: {}", error.message);
            }
        }
    }
}
