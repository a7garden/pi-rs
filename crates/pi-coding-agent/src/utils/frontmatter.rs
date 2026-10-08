//! Port of `packages/coding-agent/src/utils/frontmatter.ts` — extract and
//! parse `---`-delimited YAML frontmatter.
//!
//! Deviation D10 — the upstream `yaml` package is replaced by a hand-rolled
//! minimal YAML subset parser covering frontmatter-shaped documents:
//! block mappings, block sequences, block scalars (`|`, `|-`, `|+`, `>`,
//! `>-`, `>+`), single/double-quoted and plain scalars, flow sequences/maps,
//! and comments. Values surface as [`serde_json::Value`] (upstream returns
//! arbitrary JS values); a non-mapping document root is an error where the
//! `yaml` package would return it unchecked. Error messages carry
//! `at line N, column M` positions like the `yaml` package, but the
//! surrounding text is this parser's own.

use serde_json::{Map, Value};

/// Failure to parse frontmatter YAML (upstream propagates the `yaml`
/// package's exception).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct FrontmatterError(pub String);

/// A parsed frontmatter document: the `frontmatter` mapping plus the
/// remaining `body`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedFrontmatter {
	pub frontmatter: Map<String, Value>,
	pub body: String,
}

fn normalize_newlines(value: &str) -> String {
	value.replace("\r\n", "\n").replace('\r', "\n")
}

fn extract_frontmatter(content: &str) -> (Option<String>, String) {
	let normalized = normalize_newlines(&super::text::strip_bom(content));

	if !normalized.starts_with("---") {
		return (None, normalized);
	}

	let Some(end_index) = normalized[3..].find("\n---").map(|index| index + 3) else {
		return (None, normalized);
	};

	(Some(normalized[4..end_index].to_string()), normalized[end_index + 4..].trim().to_string())
}

/// Parse `---` frontmatter into a mapping plus the trimmed body.
pub fn parse_frontmatter(content: &str) -> Result<ParsedFrontmatter, FrontmatterError> {
	let (yaml_string, body) = extract_frontmatter(content);
	let Some(yaml_string) = yaml_string else {
		return Ok(ParsedFrontmatter { frontmatter: Map::new(), body });
	};
	let parsed = parse_yaml(&yaml_string)?;
	let frontmatter = match parsed {
		Value::Null => Map::new(),
		Value::Object(map) => map,
		other => return Err(FrontmatterError(format!("Invalid YAML frontmatter: expected a mapping, got {other}"))),
	};
	Ok(ParsedFrontmatter { frontmatter, body })
}

/// Remove frontmatter, returning the trimmed body.
pub fn strip_frontmatter(content: &str) -> Result<String, FrontmatterError> {
	Ok(parse_frontmatter(content)?.body)
}

/// A non-blank, non-comment line with its 1-based number and indent.
struct Line {
	number: usize,
	indent: usize,
	content: String,
}

fn scan_lines(text: &str) -> Vec<Line> {
	let mut lines = Vec::new();
	for (index, raw) in text.split('\n').enumerate() {
		let trimmed = raw.trim_end();
		let indent = trimmed.len() - trimmed.trim_start().len();
		let content = trimmed.trim_start();
		if content.is_empty() || content.starts_with('#') {
			continue;
		}
		lines.push(Line { number: index + 1, indent, content: content.to_string() });
	}
	lines
}

/// Parse a frontmatter YAML document (the minimal subset; see D10).
fn parse_yaml(text: &str) -> Result<Value, FrontmatterError> {
	let lines = scan_lines(text);
	if lines.is_empty() {
		return Ok(Value::Null);
	}
	let mut pos = 0;
	parse_block(&lines, &mut pos, lines[0].indent)
}

fn parse_block(lines: &[Line], pos: &mut usize, indent: usize) -> Result<Value, FrontmatterError> {
	if lines[*pos].content.starts_with("- ") || lines[*pos].content == "-" {
		parse_sequence(lines, pos, indent)
	} else if lines[*pos].content.contains(':') {
		parse_mapping(lines, pos, indent)
	} else {
		let line = &lines[*pos];
		*pos += 1;
		parse_scalar(&line.content, line.number, 1)
	}
}

fn parse_sequence(lines: &[Line], pos: &mut usize, indent: usize) -> Result<Value, FrontmatterError> {
	let mut items = Vec::new();
	while *pos < lines.len() && lines[*pos].indent == indent {
		let line = &lines[*pos];
		let rest = line.content.strip_prefix("-").unwrap().trim_start();
		if rest.is_empty() {
			*pos += 1;
			if *pos < lines.len() && lines[*pos].indent > indent {
				let child_indent = lines[*pos].indent;
				items.push(parse_block(lines, pos, child_indent)?);
			} else {
				items.push(Value::Null);
			}
			continue;
		}
		let dash_offset = line.content.len() - rest.len();
		// Re-parse as a virtual line holding the item content.
		let item = parse_inline_item(lines, pos, indent, rest, dash_offset)?;
		items.push(item);
	}
	if *pos < lines.len() && lines[*pos].indent > indent {
		return Err(FrontmatterError(format!(
			"Unexpected indentation at line {}, column {}",
			lines[*pos].number,
			lines[*pos].indent + 1
		)));
	}
	Ok(Value::Array(items))
}

/// Parse one `- <rest>` item, treating `key: value` items as single-pair
/// mappings that absorb continuation keys aligned at the item's value indent.
fn parse_inline_item(
	lines: &[Line],
	pos: &mut usize,
	indent: usize,
	rest: &str,
	dash_offset: usize,
) -> Result<Value, FrontmatterError> {
	let line_number = lines[*pos].number;
	let item_indent = indent + dash_offset;
	let item_line_number = line_number;
	if let Some((key, value_part)) = split_key(rest) {
		let mut map = Map::new();
		let value = if value_part.is_empty() {
			*pos += 1;
			child_value(lines, pos, item_indent, item_line_number)?
		} else if is_block_scalar_header(value_part) {
			*pos += 1;
			parse_block_scalar(lines, pos, indent + dash_offset, value_part)?
		} else {
			*pos += 1;
			parse_scalar(value_part, item_line_number, item_indent + 1)?
		};
		map.insert(key.to_string(), value);
		while *pos < lines.len() && lines[*pos].indent == item_indent && split_key(&lines[*pos].content).is_some() {
			parse_mapping_entry(lines, pos, item_indent, &mut map)?;
		}
		Ok(Value::Object(map))
	} else {
		*pos += 1;
		parse_scalar(rest, item_line_number, item_indent + 1)
	}
}

fn parse_mapping(lines: &[Line], pos: &mut usize, indent: usize) -> Result<Value, FrontmatterError> {
	let mut map = Map::new();
	while *pos < lines.len() && lines[*pos].indent == indent {
		parse_mapping_entry(lines, pos, indent, &mut map)?;
	}
	if *pos < lines.len() && lines[*pos].indent > indent {
		return Err(FrontmatterError(format!(
			"Unexpected indentation at line {}, column {}",
			lines[*pos].number,
			lines[*pos].indent + 1
		)));
	}
	Ok(Value::Object(map))
}

/// Parse a single `key: …` entry into `map`; returns the parsed value.
fn parse_mapping_entry(
	lines: &[Line],
	pos: &mut usize,
	indent: usize,
	map: &mut Map<String, Value>,
) -> Result<Value, FrontmatterError> {
	let line = &lines[*pos];
	let line_number = line.number;
	let Some((key, value_part)) = split_key(&line.content) else {
		return Err(FrontmatterError(format!(
			"Invalid mapping entry at line {}, column {}",
			line.number,
			line.indent + 1
		)));
	};
	let value = if value_part.is_empty() {
		*pos += 1;
		child_value(lines, pos, indent + 1, line_number)?
	} else if is_block_scalar_header(value_part) {
		*pos += 1;
		parse_block_scalar(lines, pos, indent, value_part)?
	} else {
		let column = line.indent + line.content.find(':').unwrap_or(0) + 3;
		*pos += 1;
		parse_scalar(value_part, line_number, column)?
	};
	map.insert(key.to_string(), value.clone());
	Ok(value)
}

/// Parse the value of a key whose own line ends after `key:`.
fn child_value(lines: &[Line], pos: &mut usize, min_indent: usize, _line_number: usize) -> Result<Value, FrontmatterError> {
	if *pos >= lines.len() || lines[*pos].indent < min_indent {
		return Ok(Value::Null);
	}
	let child_indent = lines[*pos].indent;
	parse_block(lines, pos, child_indent)
}

fn split_key(content: &str) -> Option<(&str, &str)> {
	let bytes = content.as_bytes();
	if bytes.first() == Some(&b'"') || bytes.first() == Some(&b'\'') {
		let quote = bytes[0];
		let mut index = 1;
		while index < bytes.len() {
			if bytes[index] == quote {
				if quote == b'\'' && bytes.get(index + 1) == Some(&b'\'') {
					index += 2;
					continue;
				}
				if quote == b'"' && bytes[index - 1] == b'\\' {
					index += 1;
					continue;
				}
				break;
			}
			index += 1;
		}
		if index >= bytes.len() {
			return None;
		}
		let rest = &content[index + 1..];
		let rest_trimmed = rest.trim_start();
		if !rest_trimmed.starts_with(':') {
			return None;
		}
		return Some((&content[1..index], rest_trimmed[1..].trim_start()));
	}
	let colon = content.find(':')?;
	let after = &content[colon + 1..];
	if !after.is_empty() && !after.starts_with(' ') {
		// Plain scalars may contain colons when not followed by a space only
		// at the top level of a URL etc.; YAML requires `key: ` here.
		if content[colon + 1..].contains(' ') {
			return None;
		}
		return None;
	}
	Some((content[..colon].trim(), after.trim_start()))
}

fn is_block_scalar_header(value: &str) -> bool {
	value == "|" || value == ">" || value == "|-" || value == ">-" || value == "|+" || value == ">+"
}

fn parse_block_scalar(lines: &[Line], pos: &mut usize, key_indent: usize, header: &str) -> Result<Value, FrontmatterError> {
	let folded = header.starts_with('>');
	let chomp_strip = header.ends_with('-');
	let chomp_keep = header.ends_with('+');

	let block_indent = if *pos < lines.len() && lines[*pos].indent > key_indent {
		lines[*pos].indent
	} else {
		return Ok(Value::String(String::new()));
	};

	let mut collected: Vec<(usize, String)> = Vec::new();
	while *pos < lines.len() && (lines[*pos].indent >= block_indent || lines[*pos].content.is_empty()) {
		let line = &lines[*pos];
		let text = if line.content.is_empty() {
			String::new()
		} else if line.indent >= block_indent {
			line.content[line.indent - block_indent.min(line.indent)..].to_string()
		} else {
			break;
		};
		collected.push((line.indent, text));
		*pos += 1;
	}
	// Trim trailing blank lines (handled per chomping below).
	while collected.last().is_some_and(|(_, text)| text.is_empty()) {
		collected.pop();
	}
	let blank_count = collected.iter().filter(|(_, text)| text.is_empty()).count();
	let mut body = String::new();
	if folded {
		for (index, (_, text)) in collected.iter().enumerate() {
			if index > 0 {
				body.push(if text.is_empty() { '\n' } else { ' ' });
			}
			body.push_str(text);
		}
		body.push('\n');
	} else {
		for (_, text) in &collected {
			body.push_str(text);
			body.push('\n');
		}
	}
	if chomp_strip {
		while body.ends_with('\n') {
			body.pop();
		}
	} else if chomp_keep {
		for _ in 0..blank_count.max(0) {
			body.push('\n');
		}
	}
	let _ = folded;
	Ok(Value::String(body))
}

fn parse_scalar(raw: &str, line: usize, column: usize) -> Result<Value, FrontmatterError> {
	let trimmed = raw.trim();
	if trimmed.starts_with('"') {
		let body = trimmed.strip_prefix('"').unwrap();
		let Some(end) = find_unescaped(body, '"') else {
			return Err(FrontmatterError(format!(
				"Unexpected end of quoted scalar at line {line}, column {column}"
			)));
		};
		return Ok(Value::String(unescape_double(body[..end].to_string())));
	}
	if trimmed.starts_with('\'') {
		let body = trimmed.strip_prefix('\'').unwrap();
		let Some(end) = find_single_quote_end(body) else {
			return Err(FrontmatterError(format!(
				"Unexpected end of quoted scalar at line {line}, column {column}"
			)));
		};
		return Ok(Value::String(body[..end].replace("''", "'")));
	}
	if trimmed.starts_with('[') {
		return parse_flow(trimmed, line, column);
	}
	if trimmed.starts_with('{') {
		return parse_flow(trimmed, line, column);
	}
	// Plain scalar: strip a trailing `# comment`.
	let mut plain = trimmed;
	if let Some(hash) = plain.find(" #") {
		plain = plain[..hash].trim_end();
	}
	Ok(plain_to_value(plain))
}

fn plain_to_value(plain: &str) -> Value {
	match plain {
		"" | "~" | "null" | "Null" | "NULL" => Value::Null,
		"true" | "True" | "TRUE" => Value::Bool(true),
		"false" | "False" | "FALSE" => Value::Bool(false),
		_ => {
			if let Ok(int) = plain.parse::<i64>() {
				return Value::Number(int.into());
			}
			if let Ok(float) = plain.parse::<f64>() {
				if float.is_finite() {
					return serde_json::Number::from_f64(float).map(Value::Number).unwrap_or(Value::Null);
				}
			}
			Value::String(plain.to_string())
		}
	}
}

fn find_unescaped(text: &str, quote: char) -> Option<usize> {
	let bytes = text.as_bytes();
	let mut index = 0;
	while index < bytes.len() {
		if bytes[index] == b'\\' {
			// Skip the escaped byte, then re-align to the next char boundary
			// so `text[index..]` below never slices mid-character.
			index += 2;
			while index < bytes.len() && !text.is_char_boundary(index) {
				index += 1;
			}
			continue;
		}
		if text[index..].starts_with(quote) {
			return Some(index);
		}
		// Advance by one full character; byte-wise +1 would land inside a
		// multi-byte character and panic on the slice above.
		index += text[index..].chars().next().unwrap().len_utf8();
	}
	None
}

fn find_single_quote_end(text: &str) -> Option<usize> {
	let mut index = 0;
	while let Some(found) = text[index..].find('\'') {
		let at = index + found;
		if text[at..].starts_with("''") {
			index = at + 2;
			continue;
		}
		return Some(at);
	}
	None
}

fn unescape_double(text: String) -> String {
	let mut out = String::with_capacity(text.len());
	let mut chars = text.chars();
	while let Some(c) = chars.next() {
		if c != '\\' {
			out.push(c);
			continue;
		}
		match chars.next() {
			Some('n') => out.push('\n'),
			Some('t') => out.push('\t'),
			Some('r') => out.push('\r'),
			Some('0') => out.push('\0'),
			Some(other) => out.push(other),
			None => out.push('\\'),
		}
	}
	out
}

/// Minimal flow-style parser for `[a, b]` and `{a: b}` collections.
fn parse_flow(text: &str, line: usize, column: usize) -> Result<Value, FrontmatterError> {
	// Unterminated collections report at the position just past the text,
	// like the `yaml` package.
	let end_column = column + text.chars().count();
	let mut chars = text.chars().peekable();
	let value = parse_flow_value(&mut chars, line, column, end_column)?;
	while chars.peek().is_some_and(|c| c.is_whitespace()) {
		chars.next();
	}
	if chars.peek().is_some() {
		return Err(FrontmatterError(format!(
			"Unexpected content at line {line}, column {column}"
		)));
	}
	Ok(value)
}

fn parse_flow_value(
	chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
	line: usize,
	column: usize,
	end_column: usize,
) -> Result<Value, FrontmatterError> {
	while chars.peek().is_some_and(|c| c.is_whitespace()) {
		chars.next();
	}
	let Some(&c) = chars.peek() else {
		return Err(FrontmatterError(format!(
			"Unexpected end of flow collection at line {line}, column {end_column}"
		)));
	};
	match c {
		'[' => {
			chars.next();
			let mut items = Vec::new();
			loop {
				while chars.peek().is_some_and(|c| c.is_whitespace()) {
					chars.next();
				}
				match chars.peek() {
					None => {
						return Err(FrontmatterError(format!(
							"Unexpected end of flow sequence at line {line}, column {end_column}"
						)))
					}
					Some(']') => {
						chars.next();
						return Ok(Value::Array(items));
					}
					Some(',') => {
						chars.next();
						continue;
					}
					Some(_) => items.push(parse_flow_value(chars, line, column, end_column)?),
				}
			}
		}
		'{' => {
			chars.next();
			let mut map = Map::new();
			loop {
				while chars.peek().is_some_and(|c| c.is_whitespace()) {
					chars.next();
				}
				match chars.peek() {
					None => {
						return Err(FrontmatterError(format!(
							"Unexpected end of flow mapping at line {line}, column {end_column}"
						)))
					}
					Some('}') => {
						chars.next();
						return Ok(Value::Object(map));
					}
					Some(',') => {
						chars.next();
						continue;
					}
					Some(_) => {
						let key = parse_flow_scalar_text(chars, ":,}}")?;
						while chars.peek().is_some_and(|c| c.is_whitespace()) {
							chars.next();
						}
						if chars.peek() != Some(&':') {
							return Err(FrontmatterError(format!(
								"Expected ':' in flow mapping at line {line}, column {column}"
							)));
						}
						chars.next();
						let value = parse_flow_value(chars, line, column, end_column)?;
						map.insert(key, value);
					}
				}
			}
		}
		_ => {
			let text = parse_flow_scalar_text(chars, ",]}")?;
			Ok(plain_to_value(text.trim()))
		}
	}
}

/// Read a scalar run of characters until one of `terminators` (or EOF).
fn parse_flow_scalar_text(
	chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
	terminators: &str,
) -> Result<String, FrontmatterError> {
	let mut out = String::new();
	while let Some(&c) = chars.peek() {
		if terminators.contains(c) {
			break;
		}
		out.push(c);
		chars.next();
	}
	Ok(out)
}

#[cfg(test)]
mod tests {
	use super::*;
	use serde_json::json;

	#[test]
	fn parses_keys_strips_quotes_and_returns_body() {
		let input = "---\nname: \"skill-name\"\ndescription: 'A desc'\nfoo-bar: value\n---\n\nBody text";
		let parsed = parse_frontmatter(input).unwrap();
		assert_eq!(parsed.frontmatter.get("name"), Some(&json!("skill-name")));
		assert_eq!(parsed.frontmatter.get("description"), Some(&json!("A desc")));
		assert_eq!(parsed.frontmatter.get("foo-bar"), Some(&json!("value")));
		assert_eq!(parsed.body, "Body text");
	}

	#[test]
	fn normalizes_newlines_and_handles_crlf() {
		let input = "---\r\nname: test\r\n---\r\nLine one\r\nLine two";
		let parsed = parse_frontmatter(input).unwrap();
		assert_eq!(parsed.body, "Line one\nLine two");
	}

	#[test]
	fn errors_on_invalid_yaml_frontmatter() {
		let input = "---\nfoo: [bar\n---\nBody";
		let error = parse_frontmatter(input).unwrap_err();
		assert!(
			error.to_string().contains("at line 1, column 10"),
			"unexpected error: {error}"
		);
	}

	#[test]
	fn parses_multiline_block_yaml_syntax() {
		let input = "---\ndescription: |\n  Line one\n  Line two\n---\n\nBody";
		let parsed = parse_frontmatter(input).unwrap();
		assert_eq!(parsed.frontmatter.get("description"), Some(&json!("Line one\nLine two\n")));
		assert_eq!(parsed.body, "Body");
	}

	#[test]
	fn returns_original_content_when_frontmatter_is_missing_or_unterminated() {
		let no_frontmatter = "Just text\nsecond line";
		let missing_end = "---\nname: test\nBody without terminator";
		let result_no_frontmatter = parse_frontmatter(no_frontmatter).unwrap();
		let result_missing_end = parse_frontmatter(missing_end).unwrap();
		assert_eq!(result_no_frontmatter.body, "Just text\nsecond line");
		assert_eq!(result_missing_end.body, "---\nname: test\nBody without terminator");
		assert!(result_missing_end.frontmatter.is_empty());
	}

	#[test]
	fn returns_empty_object_for_empty_or_comment_only_frontmatter() {
		let input = "---\n# just a comment\n---\nBody";
		let parsed = parse_frontmatter(input).unwrap();
		assert_eq!(parsed.frontmatter, Map::new());
		assert_eq!(parsed.body, "Body");
	}

	#[test]
	fn strip_frontmatter_removes_frontmatter_and_trims_body() {
		let input = "---\nkey: value\n---\n\nBody\n";
		assert_eq!(strip_frontmatter(input).unwrap(), "Body");
	}

	#[test]
	fn strip_frontmatter_returns_body_when_no_frontmatter_present() {
		let input = "\n  No frontmatter body  \n";
		assert_eq!(strip_frontmatter(input).unwrap(), "\n  No frontmatter body  \n");
	}
}
