//! Port of `packages/coding-agent/src/experimental/coordinator-entry.ts` —
//! the executable entrypoint of the internal coordinator process.
//!
//! Upstream is a `#!/usr/bin/env node` script that consumes the internal
//! spawn role and runs the coordinator with the remaining argv. The Rust
//! port exposes [`run_coordinator_entry`] for the internal-process
//! entrypoint (a future bin target or dispatcher); the module-level
//! "is this file the executed entry" probe is a JS-runtime seam and is not
//! ported (D138/D140).
//!
//! # Deviations
//!
//! - **D140 — coordinator entrypoint.** Upstream prints the error via
//!   `console.error` and `process.exit(1)`; the port returns the error
//!   string from [`run_coordinator_entry`] so the host binary decides how
//!   to surface it (same contract as the D102 mode-dispatcher seam).

use super::coordinator::run_coordinator_process;
use super::process::consume_internal_process_role;

/// Consume the internal role and run the coordinator process with `args`
/// (`process.argv.slice(2)`). Errors when the invocation was not an
/// internal coordinator spawn.
pub async fn run_coordinator_entry(args: &[String]) -> Result<(), String> {
	let role = consume_internal_process_role()?;
	if role != Some(super::process::InternalProcessRole::Coordinator) {
		return Err("Coordinator entrypoint requires an internal coordinator invocation".into());
	}
	run_coordinator_process(args).await
}

#[cfg(test)]
mod tests {
	use super::*;

	#[tokio::test]
	async fn entry_requires_internal_coordinator_role() {
		let _guard = crate::experimental::TEST_GUARD.lock();
		std::env::set_var(super::super::process::INTERNAL_PROCESS_ENV, "session-worker");
		let error = run_coordinator_entry(&[]).await.unwrap_err();
		assert_eq!(error, "Coordinator entrypoint requires an internal coordinator invocation");
		std::env::remove_var(super::super::process::INTERNAL_PROCESS_ENV);
	}

	#[tokio::test]
	async fn entry_requires_both_socket_paths() {
		let _guard = crate::experimental::TEST_GUARD.lock();
		std::env::set_var(super::super::process::INTERNAL_PROCESS_ENV, "coordinator");
		let error = run_coordinator_entry(&["/tmp/only-public.sock".to_string()])
			.await
			.unwrap_err();
		assert_eq!(error, "Coordinator requires public and control socket paths");
		std::env::remove_var(super::super::process::INTERNAL_PROCESS_ENV);
	}
}
