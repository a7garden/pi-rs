//! Port of `packages/ai/src/utils/hash.ts`.

/// Fast deterministic hash to shorten long strings (`shortHash`). Bit-exact
/// port of the upstream JS implementation (including `Math.imul` 32-bit
/// wrapping semantics and base36 output), so hashes match upstream values.
pub fn short_hash(input: &str) -> String {
	let mut h1: u32 = 0xdead_beef;
	let mut h2: u32 = 0x41c6_ce57;
	// JS `charCodeAt` yields UTF-16 code units.
	for ch in input.encode_utf16() {
		h1 = imul(h1 ^ ch as u32, 2654435761);
		h2 = imul(h2 ^ ch as u32, 1597334677);
	}
	h1 = imul(h1 ^ (h1 >> 16), 2246822507) ^ imul(h2 ^ (h2 >> 13), 3266489909);
	h2 = imul(h2 ^ (h2 >> 16), 2246822507) ^ imul(h1 ^ (h1 >> 13), 3266489909);
	format!("{}{}", to_base36(h2), to_base36(h1))
}

/// JS `Math.imul`: 32-bit wrapping multiply.
fn imul(a: u32, b: u32) -> u32 {
	a.wrapping_mul(b)
}

fn to_base36(value: u32) -> String {
	let mut out = Vec::new();
	let mut n = value;
	if n == 0 {
		return "0".to_string();
	}
	while n > 0 {
		let digit = (n % 36) as u8;
		out.push(if digit < 10 { b'0' + digit } else { b'a' + digit - 10 });
		n /= 36;
	}
	out.reverse();
	String::from_utf8(out).expect("base36 is ascii")
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Values computed with the upstream JS implementation.
	#[test]
	fn matches_upstream_values() {
		assert_eq!(short_hash(""), "k4n83c7h0j2b");
		assert_eq!(short_hash("hello"), "1h6qa0qrowduu");
		assert_eq!(short_hash("claude-opus-4-7"), "cblpga1y4x1mb");
		assert_eq!(short_hash("héllo world 🙈"), "1aaut8z5rn00q");
	}
}
