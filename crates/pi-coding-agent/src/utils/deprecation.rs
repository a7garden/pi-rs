//! Port of `packages/coding-agent/src/utils/deprecation.ts` — emit a
//! `Deprecation warning:` once per unique message.

use std::collections::HashSet;
use std::sync::LazyLock;

use parking_lot::Mutex;

static EMITTED_DEPRECATION_WARNINGS: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// Chalk `chalk.yellow` equivalent used by upstream `console.warn`.
fn yellow(text: &str) -> String {
	format!("\u{1b}[33m{text}\u{1b}[0m")
}

/// Warn about a deprecated API once per unique message.
pub fn warn_deprecation(message: &str) {
	let mut emitted = EMITTED_DEPRECATION_WARNINGS.lock();
	if !emitted.insert(message.to_string()) {
		return;
	}
	drop(emitted);
	eprintln!("{}", yellow(&format!("Deprecation warning: {message}")));
}

/// Clear deprecation warning state. Exported for tests.
pub fn clear_deprecation_warnings_for_tests() {
	EMITTED_DEPRECATION_WARNINGS.lock().clear();
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn warns_once_per_message() {
		clear_deprecation_warnings_for_tests();
		warn_deprecation("test: use x");
		warn_deprecation("test: use x");
		warn_deprecation("test: use y");
		let emitted = EMITTED_DEPRECATION_WARNINGS.lock();
		assert_eq!(emitted.len(), 2);
		assert!(emitted.contains("test: use x"));
		assert!(emitted.contains("test: use y"));
		drop(emitted);
		clear_deprecation_warnings_for_tests();
		assert!(EMITTED_DEPRECATION_WARNINGS.lock().is_empty());
	}
}
