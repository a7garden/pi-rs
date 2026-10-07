//! Port of `test/env-node-conformance.test.ts` — the file's own `describe` blocks
//! ("watch limits" and "readers"). The `registerEnvConformance` suites need
//! `packages/durable/src/testing/env-conformance.ts`, whose Rust port is owned by the
//! `testing` slice; those suites are deferred to the D-ledger below until it lands.
//!
//! Deferred (ledger):
//! - L1: `registerEnvConformance(... "NodeExecutionEnv conformance")` — full filesystem/
//!   reader/watch conformance matrix; needs `crate::testing` (placeholder owned elsewhere).
//! - L2: `registerEnvConformance(... "NodeExecutionEnv conformance with polling watches")` —
//!   same, with `NodeWatchOptions { mode: Polling, poll_interval_ms: Some(100) }`.

use std::sync::Arc;
use std::time::Duration;

use pi_chord::context::BACKGROUND_CONTEXT;

use crate::env::{
	get_or_throw, FileErrorCode, NodeEnvOptions, NodeExecutionEnv, RemoveOptions, WatchChange, WatchTarget,
	FileSystem,
};

use super::node_tests::{env_at, remove_dir, unique_dir};

fn env_with_watch(root: &std::path::Path, watch: crate::env::NodeWatchOptions) -> Arc<NodeExecutionEnv> {
	Arc::new(NodeExecutionEnv::new(NodeEnvOptions { cwd: root.to_string_lossy().into_owned(), watch, ..Default::default() }))
}

/// Collects watch changes until `predicate` matches or `timeout` elapses.
async fn wait_for_change(
	changes: &Arc<std::sync::Mutex<Vec<WatchChange>>>,
	predicate: impl Fn(&WatchChange) -> bool,
	timeout: Duration,
) -> bool {
	let deadline = tokio::time::Instant::now() + timeout;
	while tokio::time::Instant::now() < deadline {
		if changes.lock().expect("changes lock").iter().any(&predicate) {
			return true;
		}
		tokio::time::sleep(Duration::from_millis(20)).await;
	}
	changes.lock().expect("changes lock").iter().any(&predicate)
}

#[tokio::test]
async fn refuses_a_tree_over_the_directory_budget_and_stops_with_an_error_when_one_grows_past_it() {
	let root = unique_dir("pi-durable-env-node-watch");
	let mut watch = crate::env::NodeWatchOptions::default();
	watch.max_directories = Some(3);
	let env = env_with_watch(&root, watch);
	get_or_throw(env.create_dir("tree/a/b", None, &BACKGROUND_CONTEXT).await);
	get_or_throw(env.create_dir("tree/c", None, &BACKGROUND_CONTEXT).await);
	let refused = match env
		.watch(
			&[WatchTarget { path: "tree".to_string(), recursive: true, exclude: None }],
			Arc::new(|_| {}),
			&BACKGROUND_CONTEXT,
		)
		.await
	{
		Err(error) => error,
		Ok(_) => panic!("watch over the budget must fail"),
	};
	assert_eq!(refused.code, FileErrorCode::Invalid);

	get_or_throw(
		env.remove("tree/c", Some(RemoveOptions { recursive: Some(true), force: None }), &BACKGROUND_CONTEXT).await,
	);
	let changes: Arc<std::sync::Mutex<Vec<WatchChange>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
	let sink = Arc::clone(&changes);
	let watcher = get_or_throw(
		env.watch(
			&[WatchTarget { path: "tree".to_string(), recursive: true, exclude: None }],
			Arc::new(move |change| sink.lock().expect("changes lock").push(change)),
			&BACKGROUND_CONTEXT,
		)
		.await,
	);
	get_or_throw(env.create_dir("tree/d", None, &BACKGROUND_CONTEXT).await);
	let stopped = wait_for_change(&changes, |change| matches!(change, WatchChange::Error(error) if error.code == FileErrorCode::Invalid), Duration::from_secs(3)).await;
	assert!(stopped, "watcher stops with an error when the tree grows past the budget");
	watcher.close(&BACKGROUND_CONTEXT).await;
	remove_dir(&root);
}

#[tokio::test]
async fn reads_ranges_spanning_several_internal_chunks_exactly() {
	let root = unique_dir("pi-durable-env-node-readers");
	let env = env_at(&root);
	let length = 2 * 1024 * 1024 + 1024 * 1024 / 2;
	let bytes: Vec<u8> = (0..length).map(|index| ((index * 31) % 251) as u8).collect();
	std::fs::write(root.join("big.bin"), &bytes).unwrap();
	let reader = get_or_throw(env.open_binary_reader("big.bin", None, &BACKGROUND_CONTEXT).await);
	let all = get_or_throw(reader.read(0, length + 10, &BACKGROUND_CONTEXT).await);
	assert_eq!(all.len(), length);
	assert_eq!(all, bytes);
	let middle = get_or_throw(reader.read(1024 * 1024 - 3, 7, &BACKGROUND_CONTEXT).await);
	assert_eq!(middle, bytes[1024 * 1024 - 3..1024 * 1024 + 4]);
	reader.close(&BACKGROUND_CONTEXT).await;
	remove_dir(&root);
}

#[cfg(unix)]
#[tokio::test]
async fn refuses_a_fifo_without_waiting_for_a_writer() {
	let root = unique_dir("pi-durable-env-node-readers");
	let env = env_at(&root);
	std::process::Command::new("mkfifo").arg(root.join("pipe")).status().expect("mkfifo");
	let error = match env.open_binary_reader("pipe", None, &BACKGROUND_CONTEXT).await {
		Err(error) => error,
		Ok(_) => panic!("a FIFO must be refused"),
	};
	assert_eq!(error.code, FileErrorCode::Invalid);
	remove_dir(&root);
}

#[cfg(unix)]
#[tokio::test]
async fn refuses_a_fifo_even_with_no_follow_when_it_is_not_a_symlink() {
	// Companion to the upstream test pinning that `noFollow` only refuses symlinks.
	let root = unique_dir("pi-durable-env-node-readers");
	let env = env_at(&root);
	std::process::Command::new("mkfifo").arg(root.join("pipe")).status().expect("mkfifo");
	let error = match env
		.open_binary_reader("pipe", Some(crate::env::OpenBinaryOptions { no_follow: true }), &BACKGROUND_CONTEXT)
		.await
	{
		Err(error) => error,
		Ok(_) => panic!("a FIFO must be refused"),
	};
	assert_eq!(error.code, FileErrorCode::Invalid);
	remove_dir(&root);
}
