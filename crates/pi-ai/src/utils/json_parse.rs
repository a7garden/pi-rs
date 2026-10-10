//! Port of `packages/ai/src/utils/json-parse.ts`.
//!
//! Upstream falls back to the `partial-json` package for streaming
//! partial-object parsing; the Rust port implements the same suffix-truncation
//! semantics natively (crate docs, deviation D9): close open strings/arrays/
//! objects greedily so partial tool-call arguments parse as they stream.

const VALID_JSON_ESCAPES: [char; 8] = ['"', '\\', '/', 'b', 'f', 'n', 'r', 't'];

fn is_control_character(ch: char) -> bool {
    (ch as u32) <= 0x1f
}

fn escape_control_character(ch: char) -> String {
    match ch {
        '\u{8}' => "\\b".to_string(),
        '\u{c}' => "\\f".to_string(),
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        other => format!("\\u{:04x}", other as u32),
    }
}

/// Repairs malformed JSON string literals by:
/// - escaping raw control characters inside strings
/// - doubling backslashes before invalid escape characters
pub fn repair_json(json: &str) -> String {
    let mut repaired = String::with_capacity(json.len());
    let mut in_string = false;
    let mut chars = json.chars().peekable();

    while let Some(ch) = chars.next() {
        if !in_string {
            repaired.push(ch);
            if ch == '"' {
                in_string = true;
            }
            continue;
        }

        if ch == '"' {
            repaired.push(ch);
            in_string = false;
            continue;
        }

        if ch == '\\' {
            match chars.peek() {
                None => repaired.push_str("\\\\"),
                Some(&next_char) => {
                    if next_char == 'u' {
                        // Look ahead four hex digits after the `u`.
                        let rest: String = chars.clone().skip(1).take(4).collect();
                        if rest.len() == 4 && rest.chars().all(|c| c.is_ascii_hexdigit()) {
                            repaired.push_str("\\u");
                            repaired.push_str(&rest);
                            // Consume `u` + digits.
                            chars.next();
                            for _ in 0..4 {
                                chars.next();
                            }
                            continue;
                        }
                    }
                    if VALID_JSON_ESCAPES.contains(&next_char) {
                        repaired.push('\\');
                        repaired.push(next_char);
                        chars.next();
                        continue;
                    }
                    repaired.push_str("\\\\");
                }
            }
            continue;
        }

        if is_control_character(ch) {
            repaired.push_str(&escape_control_character(ch));
        } else {
            repaired.push(ch);
        }
    }

    repaired
}

/// `JSON.parse` with a [`repair_json`] retry when the raw parse fails and a
/// repair changes the text. Errors carry the serde message.
pub fn parse_json_with_repair<T: serde::de::DeserializeOwned>(
    json: &str,
) -> Result<T, ParseJsonError> {
    match serde_json::from_str(json) {
        Ok(value) => Ok(value),
        Err(original) => {
            let repaired_json = repair_json(json);
            if repaired_json != json {
                return serde_json::from_str(&repaired_json)
                    .map_err(|error| ParseJsonError(error.to_string()));
            }
            Err(ParseJsonError(original.to_string()))
        }
    }
}

/// Attempts to parse potentially incomplete JSON during streaming
/// (`parseStreamingJson`). Always returns a valid `Value`, even if the JSON is
/// incomplete: repair parse, then partial parse, then `{}`.
pub fn parse_streaming_json(partial_json: Option<&str>) -> serde_json::Value {
    let Some(partial_json) = partial_json else {
        return serde_json::json!({});
    };
    if partial_json.trim().is_empty() {
        return serde_json::json!({});
    }

    if let Ok(value) = parse_json_with_repair::<serde_json::Value>(partial_json) {
        return value;
    }
    if let Ok(value) = partial_parse(partial_json) {
        return value;
    }
    if let Ok(value) = partial_parse(&repair_json(partial_json)) {
        return value;
    }
    serde_json::json!({})
}

/// Parse a possibly-truncated JSON document by closing open containers and
/// strings (the `partial-json` "default" allow-list behavior upstream relies
/// on).
pub fn partial_parse(input: &str) -> Result<serde_json::Value, ParseJsonError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ParseJsonError("empty input".to_string()));
    }
    let completed = complete_partial(trimmed);
    serde_json::from_str(&completed).map_err(|error| ParseJsonError(error.to_string()))
}

/// Close open strings, arrays, and objects at the deepest unclosed position.
fn complete_partial(input: &str) -> String {
    let mut in_string = false;
    let mut escaped = false;
    let mut stack: Vec<char> = Vec::new();
    let mut end = input.len();

    let bytes = input.as_bytes();
    let mut iter = input.char_indices();
    while let Some((index, ch)) = iter.next() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' | '[' => stack.push(ch),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
        let _ = index;
        let _ = bytes;
    }

    if in_string {
        if escaped {
            // Trailing backslash inside a string: drop it.
            end = end.saturating_sub(1);
        }
        // Unterminated string: also drop any trailing escape or partial
        // unicode escape before closing the quote.
        end = trim_partial_escape(input, end);
    }

    let mut out = String::from(&input[..end]);
    // Drop a dangling comma or operator before closing containers (only
    // meaningful outside a string — when inside one, nothing dangles).
    if !in_string {
        while let Some(last) = out.chars().last() {
            if matches!(last, ',' | ':') {
                out.pop();
            } else {
                break;
            }
        }
    }
    // Close an open string FIRST, then its containers, so the result is valid
    // JSON (`{"path":"READ` → `{"path":"READ"}`).
    if in_string {
        out.push('"');
    }
    for open in stack.into_iter().rev() {
        out.push(match open {
            '{' => '}',
            '[' => ']',
            other => other,
        });
    }
    out
}

/// Remove a trailing incomplete escape sequence (`\`, `\u`, `\uX`, ...) from a
/// string cut mid-literal.
fn trim_partial_escape(input: &str, end: usize) -> usize {
    let bytes = input.as_bytes();
    let mut cut = end;
    let mut backslashes = 0usize;
    let mut scan = cut;
    while scan > 0 {
        let ch = bytes[scan - 1];
        if ch == b'\\' {
            backslashes += 1;
            scan -= 1;
        } else {
            break;
        }
    }
    if backslashes % 2 == 1 {
        cut = scan;
    } else {
        // Partial `\u` escape: `\u` followed by 1-3 hex digits at the end.
        let mut u_pos = None;
        let mut scan = cut;
        while scan > 0 && cut - scan < 5 {
            let ch = bytes[scan - 1];
            if ch == b'u' && cut - scan <= 4 {
                u_pos = Some(scan - 1);
                break;
            }
            if !(ch as char).is_ascii_hexdigit() {
                break;
            }
            scan -= 1;
        }
        if let Some(u_pos) = u_pos {
            // Only when the `\` is not itself escaped.
            let mut bs = 0usize;
            let mut scan = u_pos;
            while scan > 0 && bytes[scan - 1] == b'\\' {
                bs += 1;
                scan -= 1;
            }
            if bs % 2 == 0 {
                cut = u_pos;
            }
        }
    }
    cut
}

/// Parse failure surface (`JSON.parse` throwing).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ParseJsonError(String);

impl ParseJsonError {
    pub fn message(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod partial_parse_tests {
    use super::*;

    #[test]
    fn completes_partial_strings_inside_containers() {
        assert_eq!(
            parse_streaming_json(Some("{\"path\":\"READ")),
            serde_json::json!({"path": "READ"})
        );
        assert_eq!(
            parse_streaming_json(Some("{\"path\":\"READ\", \"items\":[\"a\"")),
            serde_json::json!({"path": "READ", "items": ["a"]})
        );
        assert_eq!(
            parse_streaming_json(Some("{\"a\": 1,")),
            serde_json::json!({"a": 1})
        );
        assert_eq!(parse_streaming_json(Some("{\"a\":")), serde_json::json!({}));
    }
}
