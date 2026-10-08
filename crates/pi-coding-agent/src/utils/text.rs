//! Port of `packages/coding-agent/src/utils/text.ts` — BOM splitting and
//! JSON-with-comments cleanup (upstream `stripJsonComments` is shared with
//! [`super::json`]).

/// Split a leading UTF-8 byte order mark from decoded text.
pub fn split_bom(content: &str) -> (&str, &str) {
	if let Some(text) = content.strip_prefix('\u{FEFF}') {
		("\u{FEFF}", text)
	} else {
		("", content)
	}
}

/// Remove a leading UTF-8 byte order mark from decoded text.
pub fn strip_bom(content: &str) -> &str {
	split_bom(content).1
}

/// Strip `//` line comments and trailing commas from JSON, leaving string
/// literals untouched. Mirrors the upstream two-pass regex:
/// comments (outside string literals) are removed first, then trailing commas
/// followed only by whitespace up to a closing `}` or `]`.
pub fn strip_json_comments(input: &str) -> String {
	let without_comments = strip_line_comments(input);
	strip_trailing_commas(&without_comments)
}

/// Pass 1: remove `// …` line comments while copying string literals verbatim.
fn strip_line_comments(input: &str) -> String {
	let mut out = String::with_capacity(input.len());
	let mut chars = input.chars().peekable();
	let mut in_string = false;
	while let Some(c) = chars.next() {
		if in_string {
			out.push(c);
			if c == '\\' {
				if let Some(escaped) = chars.next() {
					out.push(escaped);
				}
			} else if c == '"' {
				in_string = false;
			}
			continue;
		}
		if c == '"' {
			in_string = true;
			out.push(c);
			continue;
		}
		if c == '/' && chars.peek() == Some(&'/') {
			chars.next();
			for skipped in chars.by_ref() {
				if skipped == '\n' {
					out.push('\n');
					break;
				}
			}
			continue;
		}
		out.push(c);
	}
	out
}

/// Pass 2: drop a `,` when only whitespace separates it from `}` or `]`.
fn strip_trailing_commas(input: &str) -> String {
	let mut out = String::with_capacity(input.len());
	let mut chars = input.chars().peekable();
	let mut in_string = false;
	while let Some(c) = chars.next() {
		if in_string {
			out.push(c);
			if c == '\\' {
				if let Some(escaped) = chars.next() {
					out.push(escaped);
				}
			} else if c == '"' {
				in_string = false;
			}
			continue;
		}
		if c == '"' {
			in_string = true;
			out.push(c);
			continue;
		}
		if c == ',' {
			let mut lookahead = chars.clone();
			let mut trailing = false;
			while let Some(&ws) = lookahead.peek() {
				if ws == '}' || ws == ']' {
					trailing = true;
					break;
				}
				if ws.is_whitespace() {
					lookahead.next();
				} else {
					break;
				}
			}
			if trailing {
				continue;
			}
		}
		out.push(c);
	}
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn splits_leading_bom() {
		assert_eq!(split_bom("\u{FEFF}hello"), ("\u{FEFF}", "hello"));
		assert_eq!(split_bom("hello"), ("", "hello"));
		assert_eq!(strip_bom("\u{FEFF}text"), "text");
		assert_eq!(strip_bom("text"), "text");
	}

	#[test]
	fn strips_line_comments_outside_strings() {
		assert_eq!(strip_json_comments("{\"a\": 1 // c\n}"), "{\"a\": 1 \n}");
		assert_eq!(strip_json_comments("{\"url\": \"http://x//y\"}"), "{\"url\": \"http://x//y\"}");
		assert_eq!(strip_json_comments("// only a comment"), "");
		assert_eq!(strip_json_comments("\"// not a comment\""), "\"// not a comment\"");
		assert_eq!(strip_json_comments("\"escaped \\\" quote // x\""), "\"escaped \\\" quote // x\"");
	}

	#[test]
	fn strips_trailing_commas() {
		assert_eq!(strip_json_comments("[1, 2, 3,]"), "[1, 2, 3]");
		assert_eq!(strip_json_comments("{\"a\": 1,\n\"b\": 2,\n}"), "{\"a\": 1,\n\"b\": 2\n}");
		assert_eq!(strip_json_comments("{\"a\": [1, 2,], \"b\": 3}"), "{\"a\": [1, 2], \"b\": 3}");
		assert_eq!(strip_json_comments("[1, \"2,\"]"), "[1, \"2,\"]");
		assert_eq!(strip_json_comments("{\"a\": 1, \"b\": 2}"), "{\"a\": 1, \"b\": 2}");
	}
}
