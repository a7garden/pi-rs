//! Port of `packages/durable/src/testing/env-conformance.ts` — the
//! runner-independent `ExecutionEnv` conformance cases, run against any
//! [`ExecutionEnv`] through a provider-supplied fresh environment.
//!
//! Deviations (module ledger):
//! - D1: upstream passes negative offsets and fractional lengths to the
//!   binary reader to assert `invalid` rejections; the Rust
//!   `BinaryReader::read` signature (`u64`, `usize`) makes those inputs
//!   unrepresentable, so those two checks are dropped and the remaining
//!   invalid/aborted paths are kept.
//! - D2: upstream's string command form maps to `ExecCommand::Shell`; the
//!   argv form builds `ExecCommand::Argv([shell..., script, ...])`.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use pi_chord::context::{with_cancel, Context, BACKGROUND_CONTEXT};
use serde_json::{json, Value};

use crate::env::{
	DirPage, ExecCommand, ExecutionEnv, ExecutionError, FileContent, FileError, OpenBinaryOptions,
	ScanLinesOptions, ShellExecOptions, ShellExecResult, ShellOutputInfo, ShellOutputSkip,
	ShellOutputWindow, TextDecoder, WatchChange, WatchTarget,
};

use super::assertions::to_value;
use super::types::{ConformanceAssertions, EnvConformanceCase, EnvConformanceOptions};

pub(crate) fn ctx() -> &'static Context {
	&BACKGROUND_CONTEXT
}

/// A context whose abort signal fired before the call (upstream `abortedContext`).
fn aborted_context() -> Context {
	let cancelled = with_cancel(BACKGROUND_CONTEXT.clone());
	cancelled.cancel_default();
	cancelled.context
}

fn file_error_code<T>(result: &Result<T, FileError>) -> &'static str {
	match result {
		Ok(_) => panic!("expected an error result"),
		Err(error) => error.code.as_str(),
	}
}

fn exec_error_code(result: &Result<ShellExecResult, ExecutionError>) -> &'static str {
	match result {
		Ok(_) => panic!("expected an error result"),
		Err(error) => error.code.as_str(),
	}
}

async fn read_all(
	env: &Arc<dyn ExecutionEnv>,
	path: &str,
	max_entries: usize,
) -> (Vec<Vec<crate::env::FileInfo>>, bool) {
	let reader = env
		.open_dir_reader(path, ctx())
		.await
		.expect("dir reader opens");
	let mut pages = Vec::new();
	let mut done = false;
	for _ in 0..1000 {
		let next = reader
			.next(max_entries, ctx())
			.await
			.expect("dir page reads");
		done = next.done;
		pages.push(next.entries);
		if done {
			break;
		}
	}
	reader.close(ctx()).await;
	(pages, done)
}

/// Whether a change reports `path`: an overflow, or a reported path at or above it.
fn covers(change: &WatchChange, path: &str) -> bool {
	match change {
		WatchChange::Overflow => true,
		WatchChange::Paths(paths) => paths.iter().any(|reported| {
			path == reported
				|| path.starts_with(&format!("{reported}/"))
				|| path.starts_with(&format!("{reported}\\"))
		}),
		WatchChange::Error(_) => false,
	}
}

struct WatchHelpers {
	env: Arc<dyn ExecutionEnv>,
	changes: Arc<Mutex<Vec<WatchChange>>>,
}

impl WatchHelpers {
	fn lock(&self) -> MutexGuard<'_, Vec<WatchChange>> {
		self.changes.lock().expect("watch changes lock")
	}

	async fn absolute(&self, path: &str) -> String {
		self.env
			.absolute_path(path, ctx())
			.await
			.expect("absolute path")
	}

	/// Wait up to three seconds for a change, after the call, that reports `path`.
	async fn expect_change(&self, path: &str, change: impl Future<Output = ()>) {
		let target = self.absolute(path).await;
		let from = self.lock().len();
		change.await;
		let deadline = Instant::now() + Duration::from_secs(3);
		loop {
			let found = {
				let changes = self.lock();
				changes[from..].iter().any(|entry| covers(entry, &target))
			};
			if found {
				return;
			}
			if Instant::now() > deadline {
				let reported = self.lock()[from..].to_vec();
				panic!("No change reported {target}; got {reported:?}");
			}
			tokio::time::sleep(Duration::from_millis(20)).await;
		}
	}
}

/// Watch `targets` while `run` changes files, with helpers that wait for a change.
async fn watching(
	env: &Arc<dyn ExecutionEnv>,
	targets: Vec<WatchTarget>,
	run: impl FnOnce(WatchHelpers) -> Pin<Box<dyn Future<Output = ()> + Send>>,
) {
	let changes: Arc<Mutex<Vec<WatchChange>>> = Arc::new(Mutex::new(Vec::new()));
	let sink = changes.clone();
	let watcher = env
		.watch(
			&targets,
			Arc::new(move |change| sink.lock().expect("watch changes lock").push(change)),
			ctx(),
		)
		.await
		.expect("watcher starts");
	run(WatchHelpers {
		env: env.clone(),
		changes,
	})
	.await;
	watcher.close(ctx()).await;
}

async fn write_file(env: &Arc<dyn ExecutionEnv>, path: &str, content: &str) {
	env.write_file(path, FileContent::from(content), ctx())
		.await
		.expect("file writes");
}

/// Collect stdout and stderr streams of one `exec` call (upstream `execCollect`).
async fn exec_collect(
	env: &Arc<dyn ExecutionEnv>,
	command: ExecCommand,
	cwd: Option<&str>,
) -> (Result<ShellExecResult, ExecutionError>, String, String) {
	let stdout: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
	let stderr: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
	let sink_stdout = stdout.clone();
	let sink_stderr = stderr.clone();
	let result = env
		.exec(
			command,
			Some(ShellExecOptions {
				cwd: cwd.map(str::to_string),
				on_output: Some(Arc::new(
					move |text: &str, info: &ShellOutputInfo| match info.stream {
						crate::env::OutputStream::Stdout => {
							sink_stdout.lock().expect("stdout lock").push_str(text)
						}
						crate::env::OutputStream::Stderr => {
							sink_stderr.lock().expect("stderr lock").push_str(text)
						}
					},
				)),
				..Default::default()
			}),
			ctx(),
		)
		.await;
	let collect = |lock: &Mutex<String>| lock.lock().expect("output lock").clone();
	(result, collect(&stdout), collect(&stderr))
}

/// Build the upstream argv form `[...shell, script, ...args]`.
fn argv(shell: &[String], script: &str, args: &[&str]) -> ExecCommand {
	let mut command: Vec<String> = shell.to_vec();
	command.push(script.to_string());
	command.extend(args.iter().map(|argument| argument.to_string()));
	ExecCommand::Argv(command)
}

type EnvTest = fn(
	Arc<dyn ConformanceAssertions>,
	Arc<dyn ExecutionEnv>,
	Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>>;

fn case(
	options: &EnvConformanceOptions,
	name: &str,
	shell: Arc<Vec<String>>,
	test: EnvTest,
) -> EnvConformanceCase {
	case_with_timeout(options, name, None, shell, test)
}

/// Watch cases wait up to three seconds per step, longer than test runners allow by default.
fn watch_case(
	options: &EnvConformanceOptions,
	name: &str,
	shell: Arc<Vec<String>>,
	test: EnvTest,
) -> EnvConformanceCase {
	case_with_timeout(options, name, Some(30_000), shell, test)
}

fn case_with_timeout(
	options: &EnvConformanceOptions,
	name: &str,
	timeout_ms: Option<u64>,
	shell: Arc<Vec<String>>,
	test: EnvTest,
) -> EnvConformanceCase {
	let with_env = options.with_env.clone();
	let assertions = options.assertions.clone();
	EnvConformanceCase {
		name: name.to_string(),
		timeout_ms,
		run: Arc::new(move || {
			let with_env = with_env.clone();
			let assertions = assertions.clone();
			let shell = shell.clone();
			Box::pin(async move {
				with_env(Box::new(move |env| {
					let assertions = assertions.clone();
					let shell = shell.clone();
					Box::pin(async move { test(assertions, env, shell).await })
				}))
				.await;
			})
		}),
	}
}

/// Creates runner-independent cases for an [`ExecutionEnv`]. `with_env` must
/// call and await its callback exactly once per case with an environment whose
/// `cwd` is a fresh, empty, writable directory.
pub fn create_env_conformance(options: EnvConformanceOptions) -> Vec<EnvConformanceCase> {
	let shell: Arc<Vec<String>> = Arc::new(
		options
			.shell
			.clone()
			.unwrap_or_else(|| vec!["sh".to_string(), "-c".to_string()]),
	);
	let symlinks = options.symlinks.unwrap_or(true);
	let mut cases = vec![
		case(
			&options,
			"binary reader reads byte ranges of the opened file",
			shell.clone(),
			case_binary_reader_ranges,
		),
		case(
			&options,
			"binary reader scans lines like decoding the whole file",
			shell.clone(),
			case_binary_reader_scan_lines,
		),
		case(
			&options,
			"binary reader keeps reading the file it opened after a rename",
			shell.clone(),
			case_binary_reader_rename,
		),
		case(
			&options,
			"binary reader refuses directories, missing files and aborted opens",
			shell.clone(),
			case_binary_reader_refusals,
		),
		case(
			&options,
			"directory reader pages every entry exactly once",
			shell.clone(),
			case_dir_reader_pages,
		),
		case(
			&options,
			"directory reader reports the end and refuses use after close",
			shell.clone(),
			case_dir_reader_end_and_close,
		),
		case(
			&options,
			"directory reader refuses missing paths and files",
			shell.clone(),
			case_dir_reader_refusals,
		),
		case(
			&options,
			"directory reader skips entries removed during enumeration",
			shell.clone(),
			case_dir_reader_removed_entries,
		),
		watch_case(
			&options,
			"watch reports a missing file's creation, changes, replacement and removal",
			shell.clone(),
			case_watch_file_lifecycle,
		),
		watch_case(
			&options,
			"watch reports a missing target whose ancestors are created",
			shell.clone(),
			case_watch_missing_ancestors,
		),
		watch_case(
			&options,
			"watch follows directories created together with their contents",
			shell.clone(),
			case_watch_recursive_directory,
		),
		watch_case(
			&options,
			"watch keeps watching a path whose parent is renamed and recreated",
			shell.clone(),
			case_watch_parent_renamed,
		),
		watch_case(
			&options,
			"watch skips excluded entries and reports a rename out of them",
			shell.clone(),
			case_watch_excludes,
		),
		watch_case(
			&options,
			"watch stops reporting once closed",
			shell.clone(),
			case_watch_closed,
		),
		case(
			&options,
			"argv exec passes arguments to the program without shell parsing",
			shell.clone(),
			case_exec_argv,
		),
		case(
			&options,
			"exec reports the stream of every chunk in both forms",
			shell.clone(),
			case_exec_streams,
		),
		case(
			&options,
			"argv exec honors cwd and exit codes",
			shell.clone(),
			case_exec_cwd,
		),
		case(
			&options,
			"argv exec reports missing programs and empty argv as spawn errors",
			shell.clone(),
			case_exec_spawn_errors,
		),
		case(
			&options,
			"windowed exec keeps the exact tail and counts what it skips",
			shell.clone(),
			case_exec_window,
		),
		case(
			&options,
			"argv exec distinguishes timeout from abort",
			shell.clone(),
			case_exec_timeout_abort,
		),
	];
	if symlinks {
		cases.push(case(
			&options,
			"binary reader follows symlinks unless noFollow refuses the final one",
			shell,
			case_symlinks,
		));
	}
	cases
}

fn case_binary_reader_ranges(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		write_file(&env, "data.txt", "hello world").await;
		let reader = env
			.open_binary_reader("data.txt", None, ctx())
			.await
			.expect("binary reader opens");
		let info = reader.info(ctx()).await.expect("reader info");
		assert.partial_deep_equal(
			&to_value(&info),
			&json!({ "name": "data.txt", "kind": "file", "size": 11 }),
		);
		let decode = |bytes: &[u8]| String::from_utf8_lossy(bytes).into_owned();
		assert.strict_equal(
			&json!(decode(&reader.read(0, 5, ctx()).await.expect("read 0..5"))),
			&json!("hello"),
		);
		assert.strict_equal(
			&json!(decode(&reader.read(6, 100, ctx()).await.expect("read 6.."))),
			&json!("world"),
		);
		assert.strict_equal(
			&json!(reader.read(11, 4, ctx()).await.expect("read at end").len()),
			&json!(0),
		);
		assert.strict_equal(
			&json!(reader
				.read(50, 1, ctx())
				.await
				.expect("read past end")
				.len()),
			&json!(0),
		);
		assert.strict_equal(
			&json!(reader
				.read(3, 0, ctx())
				.await
				.expect("read zero length")
				.len()),
			&json!(0),
		);
		// D1: negative offsets and fractional lengths are unrepresentable here.
		let aborted = reader
			.read(0, 1, &aborted_context())
			.await
			.map(|bytes| bytes.len());
		assert.strict_equal(&json!(file_error_code(&aborted)), &json!("aborted"));
		reader.close(ctx()).await;
		reader.close(ctx()).await;
		let closed = reader.read(0, 1, ctx()).await.map(|bytes| bytes.len());
		assert.strict_equal(&json!(file_error_code(&closed)), &json!("invalid"));
		let info = reader
			.info(ctx())
			.await
			.map(|found: crate::env::FileInfo| found.size);
		assert.strict_equal(&json!(file_error_code(&info)), &json!("invalid"));
	})
}

fn case_binary_reader_scan_lines(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		// A byte-order mark, an invalid sequence before a newline, an empty line, a later U+FEFF, and no final newline.
		let bytes: Vec<u8> = vec![
			0xef, 0xbb, 0xbf, 0x61, 0x0a, 0xe2, 0x82, 0x0a, 0x0a, 0xef, 0xbb, 0xbf, 0x62, 0x0a,
			0xc3, 0xa9,
		];
		env.write_file("lines.txt", FileContent::Binary(bytes.clone()), ctx())
			.await
			.expect("file writes");
		let lines: Vec<String> = TextDecoder::new()
			.decode(Some(&bytes))
			.split('\n')
			.map(str::to_string)
			.collect();
		let reader = env
			.open_binary_reader("lines.txt", None, ctx())
			.await
			.expect("binary reader opens");
		let ranges: Vec<(u64, Option<u64>)> = vec![
			(0, None),
			(0, Some(1)),
			(1, Some(3)),
			(2, Some(3)),
			(3, None),
			(4, Some(9)),
		];
		for (start_line, end_line) in ranges {
			let scan = reader
				.scan_lines(
					ScanLinesOptions {
						start_line,
						end_line,
					},
					ctx(),
				)
				.await
				.expect("scan lines");
			let selected: Vec<String> = match end_line {
				// Upstream slices past EOF like `Array.slice`; clamp for Rust.
				Some(end) => {
					lines[start_line as usize..usize::min(end as usize, lines.len())].to_vec()
				}
				None => lines[start_line as usize..].to_vec(),
			};
			// Upstream strips a mark only at offset 0 (`ignoreBOM: from > 0`).
			// Upstream's JS `decode()` flushes a truncated trailing sequence to
			// U+FFFD; the Rust decoder buffers unless flushed with `None`.
			let range = |from: u64, to: u64| -> String {
				let slice = &bytes[from as usize..to as usize];
				let mut decoder = if from == 0 {
					TextDecoder::new()
				} else {
					crate::env::range_decoder()
				};
				let mut text = decoder.decode(Some(slice));
				text.push_str(&decoder.decode(None));
				text
			};
			let joined = selected.join("\n");
			assert.strict_equal(&json!(scan.newlines), &json!(lines.len() as u64 - 1));
			assert.strict_equal(&json!(range(scan.start, scan.end)), &json!(joined));
			assert.strict_equal(&json!(joined.as_bytes().len()), &json!(scan.selected_bytes));
			assert.strict_equal(
				&json!(range(scan.start, scan.first_line_end)),
				&json!(lines[start_line as usize]),
			);
			assert.strict_equal(
				&json!(lines[start_line as usize].as_bytes().len()),
				&json!(scan.first_line_bytes),
			);
		}
		let past_end = reader
			.scan_lines(
				ScanLinesOptions {
					start_line: 9,
					end_line: None,
				},
				ctx(),
			)
			.await
			.expect("scan past the end");
		assert.partial_deep_equal(
			&json!({ "start": past_end.start, "end": past_end.end, "selectedBytes": past_end.selected_bytes }),
			&json!({ "start": bytes.len(), "end": bytes.len(), "selectedBytes": 0 }),
		);
		let invalid = reader
			.scan_lines(
				ScanLinesOptions {
					start_line: 2,
					end_line: Some(2),
				},
				ctx(),
			)
			.await
			.map(|_| 0);
		assert.strict_equal(&json!(file_error_code(&invalid)), &json!("invalid"));
		reader.close(ctx()).await;
	})
}

fn case_binary_reader_rename(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		write_file(&env, "a.txt", "one").await;
		let reader = env
			.open_binary_reader("a.txt", None, ctx())
			.await
			.expect("binary reader opens");
		env.rename_file("a.txt", "b.txt", ctx())
			.await
			.expect("rename");
		write_file(&env, "a.txt", "two").await;
		let bytes = reader
			.read(0, 10, ctx())
			.await
			.expect("read through rename");
		assert.strict_equal(
			&json!(String::from_utf8_lossy(&bytes).into_owned()),
			&json!("one"),
		);
		reader.close(ctx()).await;
	})
}

fn case_binary_reader_refusals(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		env.create_dir("dir", None, ctx())
			.await
			.expect("dir creates");
		write_file(&env, "file.txt", "x").await;
		let refused = env.open_binary_reader("dir", None, ctx()).await.map(|_| 0);
		assert.strict_equal(&json!(file_error_code(&refused)), &json!("is_directory"));
		let missing = env
			.open_binary_reader("missing.txt", None, ctx())
			.await
			.map(|_| 0);
		assert.strict_equal(&json!(file_error_code(&missing)), &json!("not_found"));
		let aborted = env
			.open_binary_reader("file.txt", None, &aborted_context())
			.await
			.map(|_| 0);
		assert.strict_equal(&json!(file_error_code(&aborted)), &json!("aborted"));
	})
}

fn case_dir_reader_pages(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let names = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"];
		for name in names {
			write_file(&env, name, name).await;
		}
		env.create_dir("sub", None, ctx())
			.await
			.expect("dir creates");
		let (pages, done) = read_all(&env, ".", 2).await;
		assert.ok(done, "directory reader reached the end");
		for page in &pages {
			assert.ok(page.len() <= 2, "page within maxEntries");
		}
		let mut names_found: Vec<String> = pages
			.iter()
			.flatten()
			.map(|entry| entry.name.clone())
			.collect();
		names_found.sort();
		let mut expected: Vec<String> = names.iter().map(|name| name.to_string()).collect();
		expected.push("sub".to_string());
		expected.sort();
		assert.deep_equal(&json!(names_found), &json!(expected));
		let entries: Vec<Value> = pages.iter().flatten().map(to_value).collect();
		let sub = entries
			.iter()
			.find(|entry| entry["name"] == "sub")
			.expect("sub directory listed");
		assert.strict_equal(&sub["kind"], &json!("directory"));
		let file = entries
			.iter()
			.find(|entry| entry["name"] == "a.txt")
			.expect("a.txt listed");
		assert.partial_deep_equal(file, &json!({ "kind": "file", "size": 5 }));
	})
}

fn case_dir_reader_end_and_close(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		env.create_dir("empty", None, ctx())
			.await
			.expect("dir creates");
		let reader = env
			.open_dir_reader("empty", ctx())
			.await
			.expect("dir reader opens");
		let page = reader.next(10, ctx()).await.expect("dir page");
		assert.deep_equal(
			&json!({ "entries": page.entries, "done": page.done }),
			&json!({ "entries": [], "done": true }),
		);
		let page = reader.next(10, ctx()).await.expect("dir page");
		assert.deep_equal(
			&json!({ "entries": page.entries, "done": page.done }),
			&json!({ "entries": [], "done": true }),
		);
		let invalid = reader.next(0, ctx()).await.map(|_: DirPage| 0);
		assert.strict_equal(&json!(file_error_code(&invalid)), &json!("invalid"));
		let aborted = reader.next(1, &aborted_context()).await.map(|_: DirPage| 0);
		assert.strict_equal(&json!(file_error_code(&aborted)), &json!("aborted"));
		reader.close(ctx()).await;
		reader.close(ctx()).await;
		let closed = reader.next(1, ctx()).await.map(|_: DirPage| 0);
		assert.strict_equal(&json!(file_error_code(&closed)), &json!("invalid"));
	})
}

fn case_dir_reader_refusals(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		write_file(&env, "file.txt", "x").await;
		let missing = env.open_dir_reader("missing", ctx()).await.map(|_| 0);
		assert.strict_equal(&json!(file_error_code(&missing)), &json!("not_found"));
		let file = env.open_dir_reader("file.txt", ctx()).await.map(|_| 0);
		assert.strict_equal(&json!(file_error_code(&file)), &json!("not_directory"));
		let aborted = env
			.open_dir_reader(".", &aborted_context())
			.await
			.map(|_| 0);
		assert.strict_equal(&json!(file_error_code(&aborted)), &json!("aborted"));
	})
}

fn case_dir_reader_removed_entries(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		env.create_dir("dir", None, ctx())
			.await
			.expect("dir creates");
		for name in ["x", "y", "z"] {
			write_file(&env, &format!("dir/{name}"), name).await;
		}
		let reader = env
			.open_dir_reader("dir", ctx())
			.await
			.expect("dir reader opens");
		for name in ["x", "y", "z"] {
			env.remove(&format!("dir/{name}"), None, ctx())
				.await
				.expect("remove");
		}
		let mut entries: Vec<crate::env::FileInfo> = Vec::new();
		for _ in 0..10 {
			let next = reader.next(10, ctx()).await.expect("dir page");
			let done = next.done;
			entries.extend(next.entries);
			if done {
				break;
			}
		}
		reader.close(ctx()).await;
		assert.deep_equal(&json!(entries.len()), &json!(0));
	})
}

fn case_watch_file_lifecycle(
	_assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		watching(
			&env,
			vec![WatchTarget {
				path: "AGENTS.md".to_string(),
				..Default::default()
			}],
			|helpers| {
				Box::pin(async move {
					let change = write_file(&helpers.env, "AGENTS.md", "one");
					helpers.expect_change("AGENTS.md", change).await;
					let change = write_file(&helpers.env, "AGENTS.md", "two!");
					helpers.expect_change("AGENTS.md", change).await;
					// Editors replace a file by renaming a new one over it.
					let replace = async {
						write_file(&helpers.env, "AGENTS.md.tmp", "three").await;
						helpers
							.env
							.rename_file("AGENTS.md.tmp", "AGENTS.md", ctx())
							.await
							.expect("rename");
					};
					helpers.expect_change("AGENTS.md", replace).await;
					let change = write_file(&helpers.env, "AGENTS.md", "four");
					helpers.expect_change("AGENTS.md", change).await;
					let remove = async {
						helpers
							.env
							.remove("AGENTS.md", None, ctx())
							.await
							.expect("remove");
					};
					helpers.expect_change("AGENTS.md", remove).await;
				})
			},
		)
		.await;
	})
}

fn case_watch_missing_ancestors(
	_assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		watching(
			&env,
			vec![WatchTarget {
				path: "a/b/c/AGENTS.md".to_string(),
				..Default::default()
			}],
			|helpers| {
				Box::pin(async move {
					let change = write_file(&helpers.env, "a/b/c/AGENTS.md", "x");
					helpers.expect_change("a/b/c/AGENTS.md", change).await;
				})
			},
		)
		.await;
	})
}

fn case_watch_recursive_directory(
	_assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		env.create_dir("skills", None, ctx())
			.await
			.expect("dir creates");
		watching(
			&env,
			vec![WatchTarget {
				path: "skills".to_string(),
				recursive: true,
				..Default::default()
			}],
			|helpers| {
				Box::pin(async move {
					// Written before any watcher on the new directories can exist.
					let change = write_file(&helpers.env, "skills/a/b/SKILL.md", "one");
					helpers.expect_change("skills/a/b/SKILL.md", change).await;
					let change = write_file(&helpers.env, "skills/a/b/SKILL.md", "two!");
					helpers.expect_change("skills/a/b/SKILL.md", change).await;
					let change = write_file(&helpers.env, "skills/a/b/c/SKILL.md", "deeper");
					helpers.expect_change("skills/a/b/c/SKILL.md", change).await;
				})
			},
		)
		.await;
	})
}

fn case_watch_parent_renamed(
	_assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		write_file(&env, "proj/.pi/skills/x.md", "x").await;
		watching(
			&env,
			vec![WatchTarget {
				path: "proj/.pi/skills".to_string(),
				recursive: true,
				..Default::default()
			}],
			|helpers| {
				Box::pin(async move {
					let rename = async {
						helpers
							.env
							.rename_file("proj/.pi", "proj/old", ctx())
							.await
							.expect("rename");
					};
					helpers.expect_change("proj/.pi/skills", rename).await;
					let change = write_file(&helpers.env, "proj/.pi/skills/y.md", "y");
					helpers.expect_change("proj/.pi/skills/y.md", change).await;
					let change = write_file(&helpers.env, "proj/.pi/skills/y.md", "yy");
					helpers.expect_change("proj/.pi/skills/y.md", change).await;
				})
			},
		)
		.await;
	})
}

fn case_watch_excludes(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		env.create_dir("skills", None, ctx())
			.await
			.expect("dir creates");
		let targets = vec![WatchTarget {
			path: "skills".to_string(),
			recursive: true,
			exclude: Some(crate::env::WatchExclude {
				hidden: true,
				names: vec!["node_modules".to_string()],
			}),
		}];
		watching(&env, targets, move |helpers| {
			Box::pin(async move {
				write_file(&helpers.env, "skills/node_modules/dep/SKILL.md", "dep").await;
				write_file(&helpers.env, "skills/.SKILL.md.tmp", "draft").await;
				let rename = async {
					helpers
						.env
						.rename_file("skills/.SKILL.md.tmp", "skills/SKILL.md", ctx())
						.await
						.expect("rename");
				};
				helpers.expect_change("skills/SKILL.md", rename).await;
				let hidden = vec![
					helpers.absolute("skills/node_modules").await,
					helpers.absolute("skills/.SKILL.md.tmp").await,
				];
				let changes: Vec<WatchChange> = helpers.lock().clone();
				for change in &changes {
					if let WatchChange::Paths(paths) = change {
						for path in paths {
							assert.ok(
								!hidden.iter().any(|excluded| {
									path == excluded || path.starts_with(excluded.as_str())
								}),
								&format!("excluded {path}"),
							);
						}
					}
				}
			})
		})
		.await;
	})
}

fn case_watch_closed(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let changes: Arc<Mutex<Vec<WatchChange>>> = Arc::new(Mutex::new(Vec::new()));
		let sink = changes.clone();
		let watcher = env
			.watch(
				&[WatchTarget {
					path: "file.txt".to_string(),
					..Default::default()
				}],
				Arc::new(move |change| sink.lock().expect("watch changes lock").push(change)),
				ctx(),
			)
			.await
			.expect("watcher starts");
		assert.ok(
			matches!(
				watcher.mode(),
				crate::env::WatchMode::Native | crate::env::WatchMode::Polling
			),
			"watcher reports its mode",
		);
		watcher.close(ctx()).await;
		watcher.close(ctx()).await;
		write_file(&env, "file.txt", "x").await;
		tokio::time::sleep(Duration::from_millis(300)).await;
		let reported = changes.lock().expect("watch changes lock").clone();
		assert.deep_equal(&json!(reported.len()), &json!(0));
	})
}

fn case_exec_argv(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let hostile = "it's $(touch pwned) `touch pwned` *; touch pwned";
		let command = argv(
			&shell,
			"printf \"%s|%s\" \"$1\" \"$2\"",
			&["argv0", hostile, "a b"],
		);
		let (result, stdout, _stderr) = exec_collect(&env, command, None).await;
		assert.strict_equal(&json!(result.expect("exec succeeds").exit_code), &json!(0));
		assert.strict_equal(&json!(stdout), &json!(format!("{hostile}|a b")));
		let pwned = env.exists("pwned", ctx()).await.expect("exists");
		assert.strict_equal(&json!(pwned), &json!(false));
	})
}

fn case_exec_streams(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let script = "printf out; printf err >&2; printf more";
		let (result, stdout, stderr) = exec_collect(&env, argv(&shell, script, &[]), None).await;
		assert.strict_equal(&json!(result.expect("exec succeeds").exit_code), &json!(0));
		assert.strict_equal(&json!(stdout), &json!("outmore"));
		assert.strict_equal(&json!(stderr), &json!("err"));
		let (result, stdout, stderr) =
			exec_collect(&env, ExecCommand::Shell(script.to_string()), None).await;
		assert.strict_equal(&json!(result.expect("exec succeeds").exit_code), &json!(0));
		assert.strict_equal(&json!(stdout), &json!("outmore"));
		assert.strict_equal(&json!(stderr), &json!("err"));
	})
}

fn case_exec_cwd(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		env.create_dir("sub", None, ctx())
			.await
			.expect("dir creates");
		let (result, _stdout, _stderr) = exec_collect(
			&env,
			argv(&shell, "printf x > made.txt; exit 3", &[]),
			Some("sub"),
		)
		.await;
		assert.strict_equal(&json!(result.expect("exec succeeds").exit_code), &json!(3));
		let made = env
			.read_text_file("sub/made.txt", ctx())
			.await
			.expect("read made.txt");
		assert.strict_equal(&json!(made), &json!("x"));
	})
}

fn case_exec_spawn_errors(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let missing = env
			.exec(
				ExecCommand::Argv(vec!["pi-durable-conformance-missing-program".to_string()]),
				None,
				ctx(),
			)
			.await;
		assert.strict_equal(&json!(exec_error_code(&missing)), &json!("spawn_error"));
		let empty = env.exec(ExecCommand::Argv(Vec::new()), None, ctx()).await;
		assert.strict_equal(&json!(exec_error_code(&empty)), &json!("spawn_error"));
	})
}

fn case_exec_window(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let lines = 2000u64;
		let window = ShellOutputWindow {
			max_bytes: 200,
			max_lines: 5,
			min_interval_ms: 0,
			bytes_per_second: 1_000_000_000.0,
		};
		let state: Arc<Mutex<(u64, u64, String)>> = Arc::new(Mutex::new((0, 0, String::new())));
		let sink = state.clone();
		let result = env
			.exec(
				argv(
					&shell,
					&format!("i=0; while [ $i -lt {lines} ]; do echo line-$i; i=$((i+1)); done"),
					&[],
				),
				Some(ShellExecOptions {
					window: Some(window),
					on_output: Some(Arc::new(move |text: &str, info: &ShellOutputInfo| {
						let (mut bytes, mut newlines, mut tail) =
							std::mem::take(&mut *sink.lock().expect("window state lock"));
						if let Some(ShellOutputSkip {
							bytes: skipped_bytes,
							newlines: skipped_newlines,
							..
						}) = info.skipped
						{
							bytes += skipped_bytes;
							newlines += skipped_newlines;
							let after = text.as_bytes().len() as u64;
							let after_newlines = text.split('\n').count() as u64 - 1;
							assert!(
								after > window.max_bytes as u64
									|| after_newlines > window.max_lines as u64,
								"a skip is followed by more than the window"
							);
							tail.clear();
						}
						bytes += text.as_bytes().len() as u64;
						newlines += text.split('\n').count() as u64 - 1;
						tail.push_str(text);
						*sink.lock().expect("window state lock") = (bytes, newlines, tail);
					})),
					..Default::default()
				}),
				ctx(),
			)
			.await
			.expect("exec succeeds");
		assert.strict_equal(&json!(result.exit_code), &json!(0));
		let expected: String = (0..lines).map(|index| format!("line-{index}\n")).collect();
		let (bytes, newlines, tail) = state.lock().expect("window state lock").clone();
		assert.strict_equal(&json!(bytes), &json!(expected.as_bytes().len() as u64));
		assert.strict_equal(&json!(newlines), &json!(lines));
		let expected_tail: String = (lines - window.max_lines as u64..lines)
			.map(|index| format!("line-{index}\n"))
			.collect();
		assert.ok(
			tail.ends_with(&expected_tail),
			"the delivered output ends with the tail",
		);
	})
}

fn case_exec_timeout_abort(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		let timed_out = env
			.exec(
				argv(&shell, "sleep 2", &[]),
				Some(ShellExecOptions {
					timeout: Some(0.1),
					..Default::default()
				}),
				ctx(),
			)
			.await;
		assert.strict_equal(&json!(exec_error_code(&timed_out)), &json!("timeout"));
		let cancelled = with_cancel(BACKGROUND_CONTEXT.clone());
		let running = {
			let env = env.clone();
			let shell = shell.clone();
			let context = cancelled.context.clone();
			tokio::spawn(
				async move { env.exec(argv(&shell, "sleep 2", &[]), None, &context).await },
			)
		};
		tokio::time::sleep(Duration::from_millis(100)).await;
		cancelled.cancel_default();
		let aborted = running.await.expect("exec task joins");
		assert.strict_equal(&json!(exec_error_code(&aborted)), &json!("aborted"));
	})
}

fn case_symlinks(
	assert: Arc<dyn ConformanceAssertions>,
	env: Arc<dyn ExecutionEnv>,
	_shell: Arc<Vec<String>>,
) -> Pin<Box<dyn Future<Output = ()> + Send>> {
	Box::pin(async move {
		write_file(&env, "target.txt", "target").await;
		env.create_dir("sub", None, ctx())
			.await
			.expect("dir creates");
		write_file(&env, "sub/inner.txt", "inner").await;
		let (linked, _stdout, _stderr) = exec_collect(
			&env,
			ExecCommand::Shell("ln -s target.txt link.txt && ln -s sub dirlink".to_string()),
			None,
		)
		.await;
		assert.strict_equal(&json!(linked.expect("ln succeeds").exit_code), &json!(0));

		let followed = env
			.open_binary_reader("link.txt", None, ctx())
			.await
			.expect("link follows");
		let bytes = followed
			.read(0, 10, ctx())
			.await
			.expect("read through link");
		assert.strict_equal(
			&json!(String::from_utf8_lossy(&bytes).into_owned()),
			&json!("target"),
		);
		followed.close(ctx()).await;

		let refused = env
			.open_binary_reader(
				"link.txt",
				Some(OpenBinaryOptions { no_follow: true }),
				ctx(),
			)
			.await
			.map(|_| 0);
		assert.strict_equal(&json!(file_error_code(&refused)), &json!("invalid"));

		// Only the final component is refused; earlier symlinked directories still resolve.
		let inner = env
			.open_binary_reader(
				"dirlink/inner.txt",
				Some(OpenBinaryOptions { no_follow: true }),
				ctx(),
			)
			.await
			.expect("dirlink resolves");
		let bytes = inner.read(0, 10, ctx()).await.expect("read inner");
		assert.strict_equal(
			&json!(String::from_utf8_lossy(&bytes).into_owned()),
			&json!("inner"),
		);
		inner.close(ctx()).await;
	})
}
