//! Port of the config-path surface of `packages/coding-agent/src/config.ts`
//! — the parts session-manager depends on (`APP_NAME`, `getAgentDir`,
//! `getSessionsDir`) plus the adjacent constants exported from the package
//! entry (`CONFIG_DIR_NAME`, `VERSION`, env-var names).
//!
//! Upstream derives `APP_NAME`/`CONFIG_DIR_NAME` from `package.json`
//! `piConfig` overrides at runtime; the Rust port pins the defaults (`pi`,
//! `.pi`). Not ported yet: install-method detection, self-update commands,
//! package-asset paths, share-viewer URL, and the remaining `~/.pi/agent/*`
//! path helpers (see crate docs for the module checklist).

use crate::utils::paths::{home_dir, join, normalize_path, PathInputOptions};

/// Upstream `APP_NAME` (`pkg.piConfig?.name || "pi"`).
pub const APP_NAME: &str = "pi";

/// Upstream `APP_TITLE` (`piConfig ? APP_NAME : "π"`).
pub const APP_TITLE: &str = "π";

/// Upstream `CONFIG_DIR_NAME` (`pkg.piConfig?.configDir || ".pi"`).
pub const CONFIG_DIR_NAME: &str = ".pi";

/// Upstream `VERSION` (`pkg.version`); pinned to the workspace port version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// e.g., PI_CODING_AGENT_DIR (upstream `ENV_AGENT_DIR`).
pub const ENV_AGENT_DIR: &str = "PI_CODING_AGENT_DIR";

/// e.g., PI_CODING_AGENT_SESSION_DIR (upstream `ENV_SESSION_DIR`).
pub const ENV_SESSION_DIR: &str = "PI_CODING_AGENT_SESSION_DIR";

/// Upstream `expandTildePath` (a `normalizePath` wrapper).
pub fn expand_tilde_path(path: &str) -> String {
	normalize_path(path, &PathInputOptions::default())
}

/// Get the agent config directory (e.g., ~/.pi/agent/) — `PI_CODING_AGENT_DIR`
/// override wins (`getAgentDir`).
pub fn get_agent_dir() -> String {
	if let Ok(env_dir) = std::env::var(ENV_AGENT_DIR) {
		if !env_dir.is_empty() {
			return expand_tilde_path(&env_dir);
		}
	}
	join(&home_dir(), &format!("{CONFIG_DIR_NAME}/agent"))
}

/// Get path to sessions directory (`getSessionsDir`).
pub fn get_sessions_dir() -> String {
	join(&get_agent_dir(), "sessions")
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn agent_dir_env_override_wins() {
		// SAFETY: single-threaded test touching a distinct env key.
		std::env::set_var(ENV_AGENT_DIR, "/tmp/pi-agent-dir-override");
		assert_eq!(get_agent_dir(), "/tmp/pi-agent-dir-override");
		std::env::remove_var(ENV_AGENT_DIR);
		assert_eq!(get_sessions_dir(), join(&get_agent_dir(), "sessions"));
	}
}
