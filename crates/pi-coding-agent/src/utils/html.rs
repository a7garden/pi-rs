//! Port of `packages/coding-agent/src/utils/html.ts` — decode HTML entities
//! (`&amp;`, `&#65;`, `&#x41;`) at a position within an HTML string.
//!
//! Deviation D21 — upstream `String.fromCodePoint` throws `RangeError` for
//! lone surrogate code points (`0xD800..=0xDFFF`); [`decode_html_entity`]
//! returns `None` instead of panicking.

/// A decoded entity: the replacement `text` and the consumed `length`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedHtmlEntity {
	pub text: String,
	pub length: usize,
}

fn decode_code_point(code_point: i64) -> Option<String> {
	if !(0..=0x10FFFF).contains(&code_point) {
		return None;
	}
	// `String.fromCodePoint` rejects lone surrogates (it throws); see D21.
	let ch = char::from_u32(code_point as u32)?;
	Some(ch.to_string())
}

/// Decode a named or numeric character reference body (`amp`, `#65`,
/// `#x41`) — without the surrounding `&`/`;`.
pub fn decode_html_entity(entity: &str) -> Option<String> {
	match entity {
		"amp" => return Some("&".to_string()),
		"lt" => return Some("<".to_string()),
		"gt" => return Some(">".to_string()),
		"quot" => return Some("\"".to_string()),
		"apos" => return Some("'".to_string()),
		_ => {}
	}

	if entity.starts_with("#x") || entity.starts_with("#X") {
		return parse_js_int(&entity[2..], 16).and_then(decode_code_point);
	}

	if entity.starts_with('#') {
		return parse_js_int(&entity[1..], 10).and_then(decode_code_point);
	}

	None
}

/// `Number.parseInt` semantics: optional sign, then leading digits in `radix`;
/// `None` when there are no digits.
fn parse_js_int(text: &str, radix: u32) -> Option<i64> {
	let text = text.trim_start();
	let (negative, rest) = match text.strip_prefix('-') {
		Some(rest) => (true, rest),
		None => (false, text.strip_prefix('+').unwrap_or(text)),
	};
	let digits: String = rest.chars().take_while(|c| c.is_digit(radix)).collect();
	if digits.is_empty() {
		return None;
	}
	let value = u32::from_str_radix(&digits, radix).ok()? as i64;
	Some(if negative { -value } else { value })
}

/// Decode the entity starting at `&`-position `index` in `html`.
pub fn decode_html_entity_at(html: &str, index: usize) -> Option<DecodedHtmlEntity> {
	let rest = html.get(index..)?;
	if !rest.starts_with('&') {
		return None;
	}
	let semicolon_index = rest[1..].find(';')? + 1 + index;
	if semicolon_index - index > 16 {
		return None;
	}

	let entity = &html[index + 1..semicolon_index];
	let decoded = decode_html_entity(entity)?;

	Some(DecodedHtmlEntity { text: decoded, length: semicolon_index - index + 1 })
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn decodes_named_entities() {
		assert_eq!(decode_html_entity("amp").as_deref(), Some("&"));
		assert_eq!(decode_html_entity("lt").as_deref(), Some("<"));
		assert_eq!(decode_html_entity("gt").as_deref(), Some(">"));
		assert_eq!(decode_html_entity("quot").as_deref(), Some("\""));
		assert_eq!(decode_html_entity("apos").as_deref(), Some("'"));
		assert_eq!(decode_html_entity("nbsp"), None);
	}

	#[test]
	fn decodes_numeric_entities() {
		assert_eq!(decode_html_entity("#65").as_deref(), Some("A"));
		assert_eq!(decode_html_entity("#x41").as_deref(), Some("A"));
		assert_eq!(decode_html_entity("#X41").as_deref(), Some("A"));
		assert_eq!(decode_html_entity("#0").as_deref(), Some("\u{0}"));
		assert_eq!(decode_html_entity("#1114111").as_deref(), Some("\u{10FFFF}"));
	}

	#[test]
	fn rejects_invalid_entities() {
		assert_eq!(decode_html_entity("#x"), None);
		assert_eq!(decode_html_entity("#xzz"), None);
		assert_eq!(decode_html_entity("#-1"), None);
		assert_eq!(decode_html_entity("#99999999"), None);
		assert_eq!(decode_html_entity("#xD800"), None);
		assert_eq!(decode_html_entity("#55296"), None);
	}

	#[test]
	fn decodes_entities_at_positions() {
		let html = "a&amp;b &#x41;";
		let decoded = decode_html_entity_at(html, 1).unwrap();
		assert_eq!(decoded.text, "&");
		assert_eq!(decoded.length, 5);
		assert_eq!(decode_html_entity_at(html, 8).unwrap().text, "A");
		assert_eq!(decode_html_entity_at(html, 0), None);
		assert_eq!(decode_html_entity_at("a&amp", 1), None);
		assert_eq!(decode_html_entity_at("a&unknown;", 1), None);
		assert_eq!(decode_html_entity_at("a&toolongentityname;", 1), None);
	}
}
