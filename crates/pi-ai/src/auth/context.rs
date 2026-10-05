//! Port of `packages/ai/src/auth/context.ts`.
//!
//! Upstream reads `process.env` and probes file existence via `node:fs` with a
//! `node:os` homedir expansion. The Rust port uses `std::env` and
//! `std::path::Path::exists` directly; `~` expansion uses the `HOME`
//! environment variable (no `dirs` dependency — assignment note). Browser
//! fallbacks (undefined `process.env`, always-false `fileExists`) have no Rust
//! equivalent and are dropped. `~` expansion uses the `HOME` environment
//! variable (no `dirs` dependency — assignment note) (deviation D15).
use std::sync::Arc;

use crate::auth::types::{AuthContext, BoxFuture};

/// Default auth context: env vars from `process.env`, file existence via the
/// filesystem (`defaultProviderAuthContext`).
pub fn default_provider_auth_context() -> Arc<dyn AuthContext> {
	Arc::new(DefaultProviderAuthContext)
}

struct DefaultProviderAuthContext;

impl AuthContext for DefaultProviderAuthContext {
	fn env<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Option<String>> {
		Box::pin(async move { std::env::var(name).ok().filter(|value| !value.trim().is_empty()) })
	}

	fn file_exists<'a>(&'a self, path: &'a str) -> BoxFuture<'a, bool> {
		Box::pin(async move { std::path::Path::new(&expand_tilde(path)).exists() })
	}
}

/// Upstream: `resolved.startsWith("~")` → `os.homedir() + resolved.slice(1)`.
/// With `HOME` unset the path is used as-is (access then fails → `false`),
/// matching upstream behavior when homedir resolution fails.
fn expand_tilde(path: &str) -> String {
	if let Some(rest) = path.strip_prefix('~') {
		if let Some(home) = std::env::var_os("HOME") {
			return format!("{}{rest}", home.to_string_lossy());
		}
	}
	path.to_string()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[tokio::test]
	async fn env_returns_set_variable_and_none_for_blank() {
		let ctx = default_provider_auth_context();
		// Names chosen to not collide with the ambient environment.
		std::env::set_var("PI_AI_TEST_AUTH_ENV_SET", "value");
		std::env::set_var("PI_AI_TEST_AUTH_ENV_BLANK", "   ");
		std::env::remove_var("PI_AI_TEST_AUTH_ENV_MISSING");
		assert_eq!(ctx.env("PI_AI_TEST_AUTH_ENV_SET").await.as_deref(), Some("value"));
		assert_eq!(ctx.env("PI_AI_TEST_AUTH_ENV_BLANK").await, None);
		assert_eq!(ctx.env("PI_AI_TEST_AUTH_ENV_MISSING").await, None);
	}

	#[tokio::test]
	async fn file_exists_checks_plain_paths() {
		let ctx = default_provider_auth_context();
		let mut existing = std::env::temp_dir();
		existing.push("pi-ai-auth-context-test");
		std::fs::write(&existing, b"x").unwrap();
		assert!(ctx.file_exists(existing.to_str().unwrap()).await);
		assert!(!ctx.file_exists("/nonexistent/pi-ai-auth-context-test").await);
		std::fs::remove_file(&existing).unwrap();
	}

	#[tokio::test]
	async fn file_exists_expands_leading_tilde() {
		let home = std::env::temp_dir().join("pi-ai-auth-context-home");
		std::fs::create_dir_all(&home).unwrap();
		let marker = home.join("marker-file");
		std::fs::write(&marker, b"x").unwrap();
		std::env::set_var("HOME", &home);
		let ctx = default_provider_auth_context();
		assert!(ctx.file_exists("~/marker-file").await);
		// Bare `~` resolves to the home directory itself.
		assert!(ctx.file_exists("~").await);
		// `~`-prefixed names concatenate like upstream (`homedir() + slice(1)`).
		assert!(!ctx.file_exists("~/missing-file").await);
		std::fs::remove_file(&marker).unwrap();
		std::fs::remove_dir(&home).unwrap();
	}
}
