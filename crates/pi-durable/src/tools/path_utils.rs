//! Port of `packages/durable/src/tools/path-utils.ts` — tool path normalization and resolution.

use std::collections::HashSet;

use pi_chord::context::Context;

use crate::env::{get_or_throw, ExecutionEnv};

use super::unicode;

/// The characters of upstream's `UNICODE_SPACES` regex.
fn is_unicode_space(c: char) -> bool {
	matches!(c, '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}')
}

pub fn normalize_tool_path(path: &str) -> String {
	let normalized: String = path.chars().map(|c| if is_unicode_space(c) { ' ' } else { c }).collect();
	match normalized.strip_prefix('@') {
		Some(rest) => rest.to_string(),
		None => normalized,
	}
}

pub async fn resolve_tool_path(env: &dyn ExecutionEnv, path: &str, context: &Context) -> crate::env::Result<String> {
	env.absolute_path(&normalize_tool_path(path), context).await
}

/// Upstream `resolved.replace(/ (AM|PM)\./gi, "\u{202F}$1.")`.
fn replace_am_pm_dot(resolved: &str) -> String {
	let bytes = resolved.as_bytes();
	let am_pm_at = |i: usize| -> Option<&[u8]> {
		if bytes.get(i) != Some(&b' ') || i + 4 > bytes.len() {
			return None;
		}
		let (a, m) = (bytes[i + 1].to_ascii_uppercase(), bytes[i + 2].to_ascii_uppercase());
		let matches = (a == b'A' && m == b'M' || a == b'P' && m == b'M') && bytes[i + 3] == b'.';
		matches.then(|| &bytes[i + 1..i + 4])
	};
	let mut out = String::with_capacity(resolved.len());
	let mut i = 0;
	while i < resolved.len() {
		match am_pm_at(i) {
			Some(am_pm) => {
				out.push('\u{202F}');
				out.push_str(std::str::from_utf8(am_pm).unwrap());
				i += 4;
			}
			None => {
				let ch = resolved[i..].chars().next().unwrap();
				out.push(ch);
				i += ch.len_utf8();
			}
		}
	}
	out
}

/// macOS/HFS+ name variants of the same file, tried in order (JS `Set` dedup: first occurrence
/// wins).
fn read_variants(resolved: &str) -> Vec<String> {
	let nfd = unicode::nfd(resolved);
	let apostrophes = |text: &str| text.replace('\'', "\u{2019}");
	let mut variants = vec![
		resolved.to_string(),
		replace_am_pm_dot(resolved),
		nfd.clone(),
		apostrophes(resolved),
		apostrophes(&nfd),
	];
	let mut seen: HashSet<String> = HashSet::new();
	variants.retain(|variant| seen.insert(variant.clone()));
	variants
}

/// Resolve a `read` tool path, tolerating macOS name-normalization variants of the same file.
pub async fn resolve_read_tool_path(env: &dyn ExecutionEnv, path: &str, context: &Context) -> crate::env::Result<String> {
	let resolved = resolve_tool_path(env, path, context).await?;
	for variant in read_variants(&resolved) {
		if get_or_throw(env.exists(&variant, context).await) {
			return Ok(variant);
		}
	}
	Ok(resolved)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn normalizes_unicode_spaces_and_at_prefix() {
		assert_eq!(normalize_tool_path("\u{00A0}a\u{3000}b\u{202F}c\u{200A}d@"), " a b c d@");
		assert_eq!(normalize_tool_path("@file.txt"), "file.txt");
		assert_eq!(normalize_tool_path("plain.txt"), "plain.txt");
		// Only the leading '@' is stripped.
		assert_eq!(normalize_tool_path("a@b"), "a@b");
		// Not normalized: other whitespace and near-miss ranges.
		assert_eq!(normalize_tool_path("\u{200B}x\u{2001}"), "\u{200B}x ");
	}

	#[test]
	fn builds_deduplicated_read_variants() {
		// ASCII paths collapse to the plain variant.
		assert_eq!(read_variants("/tmp/file.txt"), vec!["/tmp/file.txt".to_string()]);
		// Decomposable names produce an NFD variant.
		let variants = read_variants("caf\u{00E9}.txt");
		assert_eq!(variants[0], "caf\u{00E9}.txt");
		assert_eq!(variants[1], "caf\u{0065}\u{0301}.txt");
		// Apostrophes map to typographic ones (plain and NFD).
		let variants = read_variants("it's.txt");
		assert!(variants.contains(&"it's.txt".to_string()));
		assert!(variants.contains(&"it\u{2019}s.txt".to_string()));
		// " AM." gets a narrow no-break space.
		let variants = read_variants("10 AM.txt");
		assert!(variants.contains(&"10\u{202F}AM.txt".to_string()));
	}

	#[test]
	fn replaces_am_pm_case_insensitively() {
		assert_eq!(replace_am_pm_dot("a AM. b pm. c Am."), "a\u{202F}AM. b\u{202F}pm. c\u{202F}Am.");
		assert_eq!(replace_am_pm_dot("no matches"), "no matches");
		assert_eq!(replace_am_pm_dot("AM."), "AM.");
		// "am." at byte 0: no leading space, untouched.
		assert_eq!(replace_am_pm_dot("am."), "am.");
	}
}
