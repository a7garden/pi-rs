//! Port of `test/env-node.test.ts` — filesystem and text-line-reader parts of
//! `NodeExecutionEnv`. The shell tests live in `node_shell_tests`, the spill tests in
//! `node_spill_tests` (mirroring the upstream file split), and the conformance harness
//! (`registerEnvConformance`) tests in `node_conformance_tests`.
//!
//! Deviations:
//! - D1: tests that spy on Node `FileHandle` internals (`sync` mock, read gating) have no
//!   counterpart without Node's prototype patching; the observable contracts they pin
//!   (flush works, reads abortable) are covered by their neighbors where expressible.
//! - D2: invalid truncation sizes that `u64` cannot represent (negative, fractional, NaN,
//!   infinite, > `MAX_SAFE_INTEGER`) are skipped; the missing-file case is kept.
//! - D3: the legacy-WSL stdin-transport `exec` test relies on mutating `process.platform`
//!   at runtime; the transport selection is a compile-time `cfg` here (node.rs D2) and the
//!   selection logic is covered by `get_bash_shell_config` unit tests instead.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use pi_chord::context::{with_abort_signal, BACKGROUND_CONTEXT};
use pi_chord::context::Context;

use crate::env::{
	get_or_throw, CreateDirOptions, FileErrorCode, FileKind, NodeEnvOptions, NodeExecutionEnv, ReadTextLinesOptions,
	RemoveOptions, TempFileOptions, TextLine, FileContent, FileSystem,
};

pub(super) fn unique_dir(prefix: &str) -> PathBuf {
	let dir = std::env::temp_dir().join(format!(
		"{prefix}-{}-{}",
		std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis(),
		crate::env::node::random_suffix()
	));
	std::fs::create_dir_all(&dir).expect("create temp dir");
	dir
}

pub(super) fn remove_dir(dir: &Path) {
	let _ = std::fs::remove_dir_all(dir);
}

pub(super) fn env_at(root: &Path) -> Arc<NodeExecutionEnv> {
	Arc::new(NodeExecutionEnv::new(NodeEnvOptions {
		cwd: root.to_string_lossy().into_owned(),
		..Default::default()
	}))
}

pub(super) fn aborted_context() -> Context {
	let controller = pi_chord::context::AbortController::new();
	controller.abort_default();
	with_abort_signal(controller.signal(), BACKGROUND_CONTEXT.clone())
}

// ── NodeExecutionEnv filesystem ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn reads_writes_lists_and_removes_files_and_directories() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	assert_eq!(
		get_or_throw(env.absolute_path("nested/child", &BACKGROUND_CONTEXT).await),
		root.join("nested/child").to_string_lossy().into_owned()
	);
	assert_eq!(
		get_or_throw(env.join_path(&[root.to_string_lossy().into_owned(), "nested".into(), "child".into()], &BACKGROUND_CONTEXT).await),
		root.join("nested/child").to_string_lossy().into_owned()
	);
	get_or_throw(env.create_dir("nested/child", None, &BACKGROUND_CONTEXT).await);
	get_or_throw(env.write_file("nested/child/file.txt", FileContent::from("hel"), &BACKGROUND_CONTEXT).await);
	get_or_throw(env.append_file("nested/child/file.txt", FileContent::from("lo"), &BACKGROUND_CONTEXT).await);
	assert_eq!(
		get_or_throw(env.read_text_file("nested/child/file.txt", &BACKGROUND_CONTEXT).await),
		"hello"
	);
	assert_eq!(
		get_or_throw(
			env.read_text_lines("nested/child/file.txt", Some(ReadTextLinesOptions { max_lines: Some(1) }), &BACKGROUND_CONTEXT)
				.await
		),
		vec!["hello"]
	);
	assert_eq!(
		get_or_throw(env.read_binary_file("nested/child/file.txt", &BACKGROUND_CONTEXT).await),
		b"hello".to_vec()
	);

	let entries = get_or_throw(env.list_dir("nested/child", &BACKGROUND_CONTEXT).await);
	assert_eq!(entries.len(), 1);
	assert_eq!(entries[0].name, "file.txt");
	assert_eq!(entries[0].path, root.join("nested/child/file.txt").to_string_lossy().into_owned());
	assert_eq!(entries[0].kind, FileKind::File);
	assert_eq!(entries[0].size, 5);

	assert!(get_or_throw(env.exists("nested/child/file.txt", &BACKGROUND_CONTEXT).await));
	get_or_throw(env.remove("nested/child/file.txt", None, &BACKGROUND_CONTEXT).await);
	assert!(!get_or_throw(env.exists("nested/child/file.txt", &BACKGROUND_CONTEXT).await));
	remove_dir(&root);
}

#[tokio::test]
async fn expands_home_relative_paths_and_file_urls() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	let home = homedir_of(&env);
	assert_eq!(
		get_or_throw(env.absolute_path("~/pi-node-env-test", &BACKGROUND_CONTEXT).await),
		home.join("pi-node-env-test").to_string_lossy().into_owned()
	);
	let file_path = root.join("file with spaces.txt");
	assert_eq!(
		get_or_throw(
			env.absolute_path(
				&format!("file://{}", file_path.to_string_lossy()),
				&BACKGROUND_CONTEXT
			)
			.await
		),
		file_path.to_string_lossy().into_owned()
	);
	remove_dir(&root);
}

#[tokio::test]
async fn returns_file_info_for_files_directories_and_symlinks_without_following() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(env.create_dir("dir", Some(CreateDirOptions { recursive: Some(true) }), &BACKGROUND_CONTEXT).await);
	get_or_throw(env.write_file("dir/file.txt", FileContent::from("hello"), &BACKGROUND_CONTEXT).await);
	std::os::unix::fs::symlink(root.join("dir/file.txt"), root.join("file-link")).unwrap();
	std::os::unix::fs::symlink(root.join("dir"), root.join("dir-link")).unwrap();

	let dir = get_or_throw(env.file_info("dir", &BACKGROUND_CONTEXT).await);
	assert_eq!((dir.name.as_str(), dir.kind), ("dir", FileKind::Directory));
	let file = get_or_throw(env.file_info("dir/file.txt", &BACKGROUND_CONTEXT).await);
	assert_eq!((file.name.as_str(), file.kind, file.size), ("file.txt", FileKind::File, 5));
	for link in ["file-link", "dir-link"] {
		let info = get_or_throw(env.file_info(link, &BACKGROUND_CONTEXT).await);
		assert_eq!(info.kind, FileKind::Symlink, "{link}");
	}
	assert_eq!(
		get_or_throw(env.canonical_path("file-link", &BACKGROUND_CONTEXT).await),
		std::fs::canonicalize(root.join("dir/file.txt")).unwrap().to_string_lossy().into_owned()
	);
	remove_dir(&root);
}

#[tokio::test]
async fn lists_symlinks_as_symlinks() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(env.write_file("target.txt", FileContent::from("hello"), &BACKGROUND_CONTEXT).await);
	std::os::unix::fs::symlink(root.join("target.txt"), root.join("link.txt")).unwrap();

	let mut entries: Vec<(String, FileKind)> = get_or_throw(env.list_dir(".", &BACKGROUND_CONTEXT).await)
		.into_iter()
		.map(|entry| (entry.name, entry.kind))
		.collect();
	entries.sort_by(|a, b| a.0.cmp(&b.0));
	assert_eq!(entries, vec![("link.txt".to_string(), FileKind::Symlink), ("target.txt".to_string(), FileKind::File)]);
	remove_dir(&root);
}

#[tokio::test]
async fn stops_reading_text_lines_at_the_requested_limit() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(env.write_file("file.txt", FileContent::from("one\ntwo\nthree"), &BACKGROUND_CONTEXT).await);
	assert_eq!(
		get_or_throw(
			env.read_text_lines("file.txt", Some(ReadTextLinesOptions { max_lines: Some(1) }), &BACKGROUND_CONTEXT)
				.await
		),
		vec!["one"]
	);
	remove_dir(&root);
}

#[tokio::test]
async fn returns_file_error_for_missing_paths_and_keeps_exists_false() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	let error = env.file_info("missing.txt", &BACKGROUND_CONTEXT).await.unwrap_err();
	assert_eq!(error.code, FileErrorCode::NotFound);
	assert_eq!(error.path.as_deref(), Some(root.join("missing.txt").to_string_lossy().as_ref()));
	assert!(!get_or_throw(env.exists("missing.txt", &BACKGROUND_CONTEXT).await));
	remove_dir(&root);
}

#[tokio::test]
async fn returns_file_error_for_listing_non_directories() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(env.write_file("file.txt", FileContent::from("hello"), &BACKGROUND_CONTEXT).await);
	let error = env.list_dir("file.txt", &BACKGROUND_CONTEXT).await.unwrap_err();
	assert_eq!(error.code, FileErrorCode::NotDirectory);
	remove_dir(&root);
}

#[tokio::test]
async fn appends_to_new_files_and_creates_parent_directories() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(env.append_file("new/nested/file.txt", FileContent::from("a"), &BACKGROUND_CONTEXT).await);
	get_or_throw(env.append_file("new/nested/file.txt", FileContent::from("b"), &BACKGROUND_CONTEXT).await);
	assert_eq!(get_or_throw(env.read_text_file("new/nested/file.txt", &BACKGROUND_CONTEXT).await), "ab");
	remove_dir(&root);
}

#[tokio::test]
async fn atomically_renames_a_file_and_replaces_the_destination() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(env.write_file("source.txt", FileContent::from("new"), &BACKGROUND_CONTEXT).await);
	get_or_throw(env.write_file("destination.txt", FileContent::from("old"), &BACKGROUND_CONTEXT).await);
	get_or_throw(env.rename_file("source.txt", "destination.txt", &BACKGROUND_CONTEXT).await);
	assert!(!get_or_throw(env.exists("source.txt", &BACKGROUND_CONTEXT).await));
	assert_eq!(get_or_throw(env.read_text_file("destination.txt", &BACKGROUND_CONTEXT).await), "new");
	remove_dir(&root);
}

#[tokio::test]
async fn reports_the_source_path_when_rename_fails_because_the_source_is_missing() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(env.write_file("destination.txt", FileContent::from("unchanged"), &BACKGROUND_CONTEXT).await);
	let error = env.rename_file("missing-source.txt", "destination.txt", &BACKGROUND_CONTEXT).await.unwrap_err();
	assert_eq!(error.code, FileErrorCode::NotFound);
	assert_eq!(error.path.as_deref(), Some(root.join("missing-source.txt").to_string_lossy().as_ref()));
	assert_eq!(get_or_throw(env.read_text_file("destination.txt", &BACKGROUND_CONTEXT).await), "unchanged");
	remove_dir(&root);
}

#[tokio::test]
async fn creates_temporary_directories_and_files() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	let temp_dir = get_or_throw(env.create_temp_dir(Some("node-env-test-"), &BACKGROUND_CONTEXT).await);
	assert!(Path::new(&temp_dir).is_dir());
	let temp_file = get_or_throw(
		env.create_temp_file(
			Some(TempFileOptions { prefix: Some("prefix-".into()), suffix: Some(".txt".into()) }),
			&BACKGROUND_CONTEXT,
		)
		.await
	);
	assert!(Path::new(&temp_file).is_file());
	assert!(temp_file.ends_with(".txt"));
	assert_eq!(get_or_throw(env.read_text_file(&temp_file, &BACKGROUND_CONTEXT).await), "");
	remove_dir(&root);
	remove_dir(&PathBuf::from(&temp_dir));
}

#[tokio::test]
async fn honors_create_dir_recursive_false_and_remove_recursive_force_options() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	let error = env
		.create_dir("missing/child", Some(CreateDirOptions { recursive: Some(false) }), &BACKGROUND_CONTEXT)
		.await
		.unwrap_err();
	assert_eq!(error.code, FileErrorCode::NotFound);

	get_or_throw(env.write_file("dir/child/file.txt", FileContent::from("hello"), &BACKGROUND_CONTEXT).await);
	assert!(
		env.remove("dir", Some(RemoveOptions { recursive: Some(false), force: None }), &BACKGROUND_CONTEXT)
			.await
			.is_err()
	);
	get_or_throw(
		env.remove("dir", Some(RemoveOptions { recursive: Some(true), force: None }), &BACKGROUND_CONTEXT).await,
	);
	assert!(!get_or_throw(env.exists("dir", &BACKGROUND_CONTEXT).await));

	assert!(
		env.remove("missing", Some(RemoveOptions { recursive: None, force: Some(false) }), &BACKGROUND_CONTEXT)
			.await
			.is_err()
	);
	get_or_throw(
		env.remove("missing", Some(RemoveOptions { recursive: None, force: Some(true) }), &BACKGROUND_CONTEXT).await,
	);
	remove_dir(&root);
}

#[tokio::test]
async fn returns_aborted_results_without_side_effects_for_pre_aborted_operations() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(env.write_file("file.txt", FileContent::from("hello"), &BACKGROUND_CONTEXT).await);
	let context = aborted_context();

	assert!(env.read_text_file("file.txt", &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.read_text_lines("file.txt", None, &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.read_binary_file("file.txt", &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(matches!(env.open_text_line_reader("file.txt", &context).await, Err(error) if error.code == FileErrorCode::Aborted));
	assert!(env.write_file("other.txt", FileContent::from("hello"), &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.append_file("file.txt", FileContent::from(" world"), &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.truncate_file("file.txt", 1, &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.flush_file("file.txt", &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.rename_file("file.txt", "renamed.txt", &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.file_info("file.txt", &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.list_dir(".", &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.canonical_path("file.txt", &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.exists("file.txt", &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.create_dir("dir", None, &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.remove("file.txt", None, &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.create_temp_dir(None, &context).await.unwrap_err().code == FileErrorCode::Aborted);
	assert!(env.create_temp_file(None, &context).await.unwrap_err().code == FileErrorCode::Aborted);

	assert_eq!(get_or_throw(env.read_text_file("file.txt", &BACKGROUND_CONTEXT).await), "hello");
	assert_eq!(
		get_or_throw(env.list_dir(".", &BACKGROUND_CONTEXT).await)
			.into_iter()
			.map(|entry| entry.name)
			.collect::<Vec<_>>(),
		vec!["file.txt"]
	);
	remove_dir(&root);
}

#[tokio::test]
async fn truncates_and_extends_files_to_exact_byte_sizes() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(
		env.write_file("file.bin", FileContent::from(vec![0x61, 0xc3, 0xa9, 0x0a, 0x62, 0x0a]), &BACKGROUND_CONTEXT)
			.await,
	);

	// Truncation is byte-exact even when the boundary splits a UTF-8 sequence.
	get_or_throw(env.truncate_file("file.bin", 2, &BACKGROUND_CONTEXT).await);
	assert_eq!(get_or_throw(env.read_binary_file("file.bin", &BACKGROUND_CONTEXT).await), vec![0x61, 0xc3]);

	get_or_throw(env.truncate_file("file.bin", 4, &BACKGROUND_CONTEXT).await);
	assert_eq!(get_or_throw(env.read_binary_file("file.bin", &BACKGROUND_CONTEXT).await), vec![0x61, 0xc3, 0, 0]);

	get_or_throw(env.truncate_file("file.bin", 0, &BACKGROUND_CONTEXT).await);
	assert_eq!(get_or_throw(env.file_info("file.bin", &BACKGROUND_CONTEXT).await).size, 0);
	remove_dir(&root);
}

#[tokio::test]
async fn never_creates_missing_files_when_truncating() {
	// D2: sizes that `u64` cannot represent (negative, fractional, NaN, infinite, beyond
	// `MAX_SAFE_INTEGER`) are skipped; the missing-file contract is kept.
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(env.write_file("file.txt", FileContent::from("hello"), &BACKGROUND_CONTEXT).await);
	let error = env.truncate_file("missing.txt", 0, &BACKGROUND_CONTEXT).await.unwrap_err();
	assert_eq!(error.code, FileErrorCode::NotFound);
	assert_eq!(error.path.as_deref(), Some(root.join("missing.txt").to_string_lossy().as_ref()));
	assert!(!get_or_throw(env.exists("missing.txt", &BACKGROUND_CONTEXT).await));
	remove_dir(&root);
}

#[tokio::test]
async fn flushes_existing_files_without_changing_content_and_reports_missing_paths() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(env.write_file("file.txt", FileContent::from("durable"), &BACKGROUND_CONTEXT).await);
	get_or_throw(env.flush_file("file.txt", &BACKGROUND_CONTEXT).await);
	assert_eq!(get_or_throw(env.read_text_file("file.txt", &BACKGROUND_CONTEXT).await), "durable");

	let error = env.flush_file("missing.txt", &BACKGROUND_CONTEXT).await.unwrap_err();
	assert_eq!(error.code, FileErrorCode::NotFound);
	assert_eq!(error.path.as_deref(), Some(root.join("missing.txt").to_string_lossy().as_ref()));
	assert!(!get_or_throw(env.exists("missing.txt", &BACKGROUND_CONTEXT).await));

	get_or_throw(env.create_dir("dir", None, &BACKGROUND_CONTEXT).await);
	assert_eq!(env.flush_file("dir", &BACKGROUND_CONTEXT).await.unwrap_err().code, FileErrorCode::IsDirectory);
	remove_dir(&root);
}

#[tokio::test]
async fn cleanup_is_best_effort() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	env.cleanup(&BACKGROUND_CONTEXT).await;
	remove_dir(&root);
}

// ── NodeExecutionEnv text line reader ────────────────────────────────────────────────────────

#[tokio::test]
async fn reports_whether_each_line_was_newline_terminated_and_preserves_carriage_returns() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(env.write_file("lines.txt", FileContent::from("one\r\n\ntwo\npartial"), &BACKGROUND_CONTEXT).await);
	let reader = get_or_throw(env.open_text_line_reader("lines.txt", &BACKGROUND_CONTEXT).await);
	let mut lines = Vec::new();
	while let Some(line) = get_or_throw(reader.read_line(&BACKGROUND_CONTEXT).await) {
		lines.push(line);
	}
	assert_eq!(
		lines,
		vec![
			TextLine { text: "one\r".to_string(), terminated: true },
			TextLine { text: String::new(), terminated: true },
			TextLine { text: "two".to_string(), terminated: true },
			TextLine { text: "partial".to_string(), terminated: false },
		]
	);
	assert!(get_or_throw(reader.read_line(&BACKGROUND_CONTEXT).await).is_none());
	reader.close(&BACKGROUND_CONTEXT).await;
	remove_dir(&root);
}

#[tokio::test]
async fn returns_no_lines_for_an_empty_file_and_one_terminated_empty_line_for_a_lone_newline() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(env.write_file("empty.txt", FileContent::from(""), &BACKGROUND_CONTEXT).await);
	get_or_throw(env.write_file("newline.txt", FileContent::from("\n"), &BACKGROUND_CONTEXT).await);

	let empty = get_or_throw(env.open_text_line_reader("empty.txt", &BACKGROUND_CONTEXT).await);
	assert!(get_or_throw(empty.read_line(&BACKGROUND_CONTEXT).await).is_none());
	empty.close(&BACKGROUND_CONTEXT).await;

	let newline = get_or_throw(env.open_text_line_reader("newline.txt", &BACKGROUND_CONTEXT).await);
	assert_eq!(get_or_throw(newline.read_line(&BACKGROUND_CONTEXT).await), Some(TextLine { text: String::new(), terminated: true }));
	assert!(get_or_throw(newline.read_line(&BACKGROUND_CONTEXT).await).is_none());
	newline.close(&BACKGROUND_CONTEXT).await;
	remove_dir(&root);
}

#[tokio::test]
async fn decodes_multi_byte_characters_split_across_read_chunks_and_lines_longer_than_one_chunk() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	// The reader uses 64 KiB chunks; place a four-byte character across the first boundary.
	let first = format!("{}😀tail", "a".repeat(64 * 1024 - 2));
	let second = "é".repeat(100_000);
	get_or_throw(env.write_file("large.txt", FileContent::from(format!("{first}\n{second}")), &BACKGROUND_CONTEXT).await);
	let reader = get_or_throw(env.open_text_line_reader("large.txt", &BACKGROUND_CONTEXT).await);
	assert_eq!(
		get_or_throw(reader.read_line(&BACKGROUND_CONTEXT).await),
		Some(TextLine { text: first, terminated: true })
	);
	assert_eq!(
		get_or_throw(reader.read_line(&BACKGROUND_CONTEXT).await),
		Some(TextLine { text: second, terminated: false })
	);
	assert!(get_or_throw(reader.read_line(&BACKGROUND_CONTEXT).await).is_none());
	reader.close(&BACKGROUND_CONTEXT).await;
	remove_dir(&root);
}

#[tokio::test]
async fn rejects_reads_after_close_and_closes_idempotently() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	get_or_throw(env.write_file("lines.txt", FileContent::from("one\n"), &BACKGROUND_CONTEXT).await);
	let reader = get_or_throw(env.open_text_line_reader("lines.txt", &BACKGROUND_CONTEXT).await);
	reader.close(&BACKGROUND_CONTEXT).await;
	reader.close(&BACKGROUND_CONTEXT).await;
	let error = reader.read_line(&BACKGROUND_CONTEXT).await.unwrap_err();
	assert_eq!(error.code, FileErrorCode::Invalid);
	assert_eq!(error.path.as_deref(), Some(root.join("lines.txt").to_string_lossy().as_ref()));
	remove_dir(&root);
}

#[tokio::test]
async fn reports_missing_files_when_opening_a_reader() {
	let root = unique_dir("pi-durable-env-node");
	let env = env_at(&root);
	let error = match env.open_text_line_reader("missing.txt", &BACKGROUND_CONTEXT).await {
		Err(error) => error,
		Ok(_) => panic!("missing file must not open"),
	};
	assert_eq!(error.code, FileErrorCode::NotFound);
	assert_eq!(error.path.as_deref(), Some(root.join("missing.txt").to_string_lossy().as_ref()));
	remove_dir(&root);
}

/// Node's `os.homedir()` for the home-expansion test.
fn homedir_of(_env: &NodeExecutionEnv) -> PathBuf {
	std::env::home_dir().unwrap_or_else(|| PathBuf::from("/"))
}
