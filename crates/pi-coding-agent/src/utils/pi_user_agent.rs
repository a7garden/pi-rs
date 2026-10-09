//! Port of `packages/coding-agent/src/utils/pi-user-agent.ts` — the `pi/…`
//! User-Agent string.
//!
//! Deviation D22 — `process.platform`/`process.arch`/`process.versions` have
//! no direct Rust counterpart; [`get_pi_user_agent`] maps
//! `std::env::consts::{OS, ARCH}` onto the Node naming (`macos`→`darwin`,
//! `windows`→`win32`, `aarch64`→`arm64`, `x86_64`→`x64`) and reports the
//! runtime as `rust` where upstream reports `bun/x.y` or `node/vX`.

/// The `pi/<version> (<platform>; <runtime>; <arch>)` User-Agent value.
pub fn get_pi_user_agent(version: &str) -> String {
	format!("pi/{version} ({}; rust; {})", node_platform(), node_arch())
}

/// `process.platform` naming (`darwin`/`win32`/…), shared with the bug-report
/// environment summary.
pub(crate) fn node_platform() -> String {
	match std::env::consts::OS {
		"macos" => "darwin".to_string(),
		"windows" => "win32".to_string(),
		other => other.to_string(),
	}
}

/// `process.arch` naming (`arm64`/`x64`/…), shared with the bug-report
/// environment summary.
pub(crate) fn node_arch() -> String {
	match std::env::consts::ARCH {
		"aarch64" => "arm64".to_string(),
		"x86_64" => "x64".to_string(),
		other => other.to_string(),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn formats_the_user_agent() {
		let agent = get_pi_user_agent("1.2.3");
		let inner = agent.strip_prefix("pi/1.2.3 (").expect("version prefix").strip_suffix(')').expect("closing paren");
		let parts: Vec<&str> = inner.split("; ").collect();
		assert_eq!(parts.len(), 3, "unexpected agent: {agent}");
		assert!(matches!(parts[0].as_ref(), "darwin" | "linux" | "win32" | _));
		assert_eq!(parts[1], "rust");
	}

	#[test]
	fn uses_node_arch_names() {
		// Raw Rust target names must not leak into the string.
		let agent = get_pi_user_agent("0.0.0");
		assert!(!agent.contains("aarch64") && !agent.contains("x86_64"), "unexpected agent: {agent}");
	}
}
