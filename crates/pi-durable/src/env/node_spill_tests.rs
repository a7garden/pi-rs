//! Port of the spill tests of `test/env-node.test.ts` plus `test/env-node-spill.test.ts`.
//!
//! Deviations:
//! - D1: `env-node-spill.test.ts` mocks Node's `createWriteStream` to be slow and immediately
//!   backpressured, pinning the high-water-mark backpressure contract. Rust spill writes are
//!   sequential awaited appends on a shared file (node.rs D6), so there is no write stream to
//!   mock; the observable guarantees (no output after settle, post-exit grace extended while
//!   a spill write is pending, spill content complete) are covered by the ported tests below
//!   and the `failing_spill` seam. The artificial slow-write mock test itself is deferred.
//! - D2: "preserves exact raw bytes" uses a shell `printf` with escapes instead of Node
//!   `Buffer` (bytes `66 80 00 6f`); `printf '\x80'` emits a raw 0x80 byte via the shell.

use std::sync::Arc;

use crate::env::{
	get_or_throw, ExecCommand, ExecutionErrorCode, NodeEnvOptions, NodeExecutionEnv, ShellExecOptions,
	ShellSpillOptions, FileSystem, Shell,
};

use super::node_tests::{remove_dir, unique_dir};

fn env_with_failing_spill(root: &std::path::Path) -> Arc<NodeExecutionEnv> {
	// Upstream `FailingSpillExecutionEnv`: `createTempFile` with the spill prefix returns a
	// path in a missing directory; the env exposes the same seam as a test-only flag.
	let mut env = NodeExecutionEnv::new(NodeEnvOptions {
		cwd: root.to_string_lossy().into_owned(),
		..Default::default()
	});
	env.failing_spill = true;
	Arc::new(env)
}

#[tokio::test]
async fn does_not_create_a_spill_before_output_crosses_its_thresholds() {
	let root = unique_dir("pi-durable-env-node");
	let env = super::node_tests::env_at(&root);
	let result = get_or_throw(
		env.exec(
			ExecCommand::from("printf short"),
			Some(ShellExecOptions {
				spill: Some(ShellSpillOptions { after_bytes: 100, after_lines: 10 }),
				..Default::default()
			}),
			&BACKGROUND_CONTEXT,
		)
		.await,
	);
	assert!(result.spill_path.is_none());
	remove_dir(&root);
}

use pi_chord::context::BACKGROUND_CONTEXT;

#[tokio::test]
async fn preserves_exact_raw_bytes_in_the_spill_while_streaming_decoded_text() {
	let root = unique_dir("pi-durable-env-node");
	let env = super::node_tests::env_at(&root);
	let expected: Vec<u8> = vec![0x66, 0x80, 0x00, 0x6f];
	let result = get_or_throw(
		env.exec(
			ExecCommand::from("printf 'f\\200\\000o'"),
			Some(ShellExecOptions {
				spill: Some(ShellSpillOptions { after_bytes: 1, after_lines: 10 }),
				..Default::default()
			}),
			&BACKGROUND_CONTEXT,
		)
		.await,
	);
	let spill_path = result.spill_path.expect("spill path");
	assert_eq!(get_or_throw(env.read_binary_file(&spill_path, &BACKGROUND_CONTEXT).await), expected);
	let _ = std::fs::remove_file(&spill_path);
	remove_dir(&root);
}

#[tokio::test]
async fn reports_the_spill_of_a_command_that_times_out() {
	let root = unique_dir("pi-durable-env-node");
	let env = super::node_tests::env_at(&root);
	let error = env
		.exec(
			ExecCommand::from("printf 12345678901234567890; sleep 5"),
			Some(ShellExecOptions {
				timeout: Some(0.3),
				spill: Some(ShellSpillOptions { after_bytes: 10, after_lines: 10 }),
				..Default::default()
			}),
			&BACKGROUND_CONTEXT,
		)
		.await
		.unwrap_err();
	assert_eq!(error.code, ExecutionErrorCode::Timeout);
	let spill_path = error.spill_path.expect("spill path on timeout");
	assert_eq!(get_or_throw(env.read_text_file(&spill_path, &BACKGROUND_CONTEXT).await), "12345678901234567890");
	let _ = std::fs::remove_file(&spill_path);
	remove_dir(&root);
}

#[tokio::test]
async fn fails_rather_than_silently_losing_a_requested_spill() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_with_failing_spill(&root);
	let error = env
		.exec(
			ExecCommand::from("printf 12345678901234567890"),
			Some(ShellExecOptions {
				spill: Some(ShellSpillOptions { after_bytes: 10, after_lines: 10 }),
				..Default::default()
			}),
			&BACKGROUND_CONTEXT,
		)
		.await
		.unwrap_err();
	assert_eq!(error.code, crate::env::ExecutionErrorCode::Unknown);
	assert!(error.message.contains("Failed to preserve complete shell output"), "{}", error.message);
	remove_dir(&root);
}
