//! Port of `packages/env/test/differential.test.ts` — `RemoteExecutionEnv` must give the
//! same answers as `NodeExecutionEnv` for identical operations. This file carries the random
//! file-operation-sequences case; the read/bash tool case lives in `tests/tool_differential.rs`.

use std::sync::atomic::{AtomicUsize, Ordering};

use pi_chord::context::BACKGROUND_CONTEXT;
use pi_durable::env::{
    CreateDirOptions, FileContent, FileError, FileInfo, FileSystem, NodeEnvOptions,
    OpenBinaryOptions, ReadTextLinesOptions, RemoveOptions, ScanLinesOptions,
};
use pi_durable::env::node::NodeExecutionEnv;
use pi_env::{Connection, ConnectionOptions, RemoteExecutionEnv};
use serde_json::{json, Value};

static NEXT: AtomicUsize = AtomicUsize::new(0);

const NAMES: [&str; 8] = ["a.txt", "b", "dir", "dir/c.txt", "dir/sub", "dir/sub/d.bin", "missing", "a.txt/x"];

/// Upstream `random(seed)`: the same xorshift32 the TypeScript test uses.
fn random(seed: u32) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state.wrapping_add(0x6d2b79f5);
        let mut t = state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        (t ^ (t >> 14)) as f64 / 4294967296.
    }
}

fn name(next: &mut impl FnMut() -> f64) -> &'static str {
    NAMES[(next() * NAMES.len() as f64) as usize]
}

fn flag(next: &mut impl FnMut() -> f64) -> bool {
    next() < 0.5
}

fn small(next: &mut impl FnMut() -> f64, limit: u64) -> u64 {
    (next() * limit as f64) as u64
}

/// Upstream `CONTENTS`: empty text, plain text, BOM + CRLF line, fixed bytes, 70k `x`s —
/// paired with the label fragment upstream's `pick`/`JSON.stringify` produce for it.
fn content(next: &mut impl FnMut() -> f64) -> (FileContent, String) {
    match (next() * 5.) as usize {
        0 => (FileContent::Text(String::new()), "\"\"".into()),
        1 => (FileContent::Text("hello\n".into()), "\"hello\\n\"".into()),
        2 => (FileContent::Text("\u{feff}bom\r\nline".into()), "\"\\ufeffbom\\r\\nline\"".into()),
        3 => (FileContent::Binary(vec![0xe2, 0x82, 0x0a, 0xff, 0xef, 0xbb, 0xbf]), "\"<bytes>\"".into()),
        _ => (FileContent::Text("x".repeat(70_000)), "\"<70000 chars>\"".into()),
    }
}

/// A `Result` reduced the way upstream's `normalize` does: failures keep only their code.
fn ok_or_code<TValue>(result: Result<TValue, FileError>, success: impl FnOnce(TValue) -> Value) -> Value {
    match result {
        Ok(value) => json!({ "ok": true, "value": success(value) }),
        Err(error) => json!({ "ok": false, "code": error.code.as_str() }),
    }
}

/// A binary read as its hex digest (upstream turns `Uint8Array`s into `{bytes: hex}`).
fn bytes_result(result: Result<Vec<u8>, FileError>) -> Value {
    ok_or_code(result, |bytes| json!(bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>()))
}

/// A `FileInfo` with `mtimeMs` replaced by a constant and the root spelled `<root>`.
fn info_value(root: &str, info: FileInfo) -> Value {
    json!({
        "name": info.name,
        "path": info.path.replace(root, "<root>"),
        "kind": info.kind,
        "size": info.size,
        "mtimeMs": "mtime",
    })
}

#[derive(Clone, Copy)]
enum Kind {
    WriteFile,
    AppendFile,
    ReadTextFile,
    ReadBinaryFile,
    ReadTextLines,
    FileInfo,
    Exists,
    ListDir,
    CreateDir,
    Remove,
    RenameFile,
    TruncateFile,
    FlushFile,
    CanonicalPath,
    OpenBinaryReader,
}

/// A random operation with its arguments fixed, and a label for failure messages
/// (upstream `randomOperation`).
struct Operation {
    label: String,
    kind: Kind,
    a: String,
    b: String,
    content: FileContent,
    flag_a: bool,
    flag_b: bool,
    small_a: u64,
    small_b: u64,
}

fn random_operation(next: &mut impl FnMut() -> f64) -> Operation {
    let mut operation = Operation {
        label: String::new(),
        kind: Kind::WriteFile,
        a: String::new(),
        b: String::new(),
        content: FileContent::Text(String::new()),
        flag_a: false,
        flag_b: false,
        small_a: 0,
        small_b: 0,
    };
    match (next() * 15.) as usize {
        0 => {
            operation.kind = Kind::WriteFile;
            operation.a = name(next).to_string();
            let (content, label) = content(next);
            operation.content = content;
            operation.label = format!("writeFile({:?}, {label})", operation.a);
        }
        1 => {
            operation.kind = Kind::AppendFile;
            operation.a = name(next).to_string();
            let (content, label) = content(next);
            operation.content = content;
            operation.label = format!("appendFile({:?}, {label})", operation.a);
        }
        2 => {
            operation.kind = Kind::ReadTextFile;
            operation.a = name(next).to_string();
            operation.label = format!("readTextFile({:?})", operation.a);
        }
        3 => {
            operation.kind = Kind::ReadBinaryFile;
            operation.a = name(next).to_string();
            operation.label = format!("readBinaryFile({:?})", operation.a);
        }
        4 => {
            operation.kind = Kind::ReadTextLines;
            operation.a = name(next).to_string();
            operation.small_a = small(next, 4);
            operation.label = format!("readTextLines({:?}, {{maxLines: {}}})", operation.a, operation.small_a);
        }
        5 => {
            operation.kind = Kind::FileInfo;
            operation.a = name(next).to_string();
            operation.label = format!("fileInfo({:?})", operation.a);
        }
        6 => {
            operation.kind = Kind::Exists;
            operation.a = name(next).to_string();
            operation.label = format!("exists({:?})", operation.a);
        }
        7 => {
            operation.kind = Kind::ListDir;
            operation.a = name(next).to_string();
            operation.label = format!("listDir({:?})", operation.a);
        }
        8 => {
            operation.kind = Kind::CreateDir;
            operation.a = name(next).to_string();
            operation.flag_a = flag(next);
            operation.label = format!("createDir({:?}, {})", operation.a, operation.flag_a);
        }
        9 => {
            operation.kind = Kind::Remove;
            operation.a = name(next).to_string();
            operation.flag_a = flag(next);
            operation.flag_b = flag(next);
            operation.label = format!("remove({:?}, {}, {})", operation.a, operation.flag_a, operation.flag_b);
        }
        10 => {
            operation.kind = Kind::RenameFile;
            operation.a = name(next).to_string();
            operation.b = name(next).to_string();
            operation.label = format!("renameFile({:?}, {:?})", operation.a, operation.b);
        }
        11 => {
            operation.kind = Kind::TruncateFile;
            operation.a = name(next).to_string();
            operation.small_a = small(next, 10);
            operation.label = format!("truncateFile({:?}, {})", operation.a, operation.small_a);
        }
        12 => {
            operation.kind = Kind::FlushFile;
            operation.a = name(next).to_string();
            operation.label = format!("flushFile({:?})", operation.a);
        }
        13 => {
            operation.kind = Kind::CanonicalPath;
            operation.a = name(next).to_string();
            operation.label = format!("canonicalPath({:?})", operation.a);
        }
        _ => {
            operation.kind = Kind::OpenBinaryReader;
            operation.a = name(next).to_string();
            operation.flag_a = flag(next);
            operation.small_a = small(next, 8);
            operation.small_b = small(next, 8);
            operation.label = format!(
                "openBinaryReader({:?}, {}, {}, {})",
                operation.a, operation.flag_a, operation.small_a, operation.small_b
            );
        }
    }
    operation
}

/// Run one fixed operation against one environment, reducing the result for comparison.
async fn run(env: &dyn FileSystem, operation: &Operation, root: &str) -> Value {
    let a = operation.a.as_str();
    match operation.kind {
        Kind::WriteFile => ok_or_code(env.write_file(a, operation.content.clone(), &BACKGROUND_CONTEXT).await, |_| json!(null)),
        Kind::AppendFile => ok_or_code(env.append_file(a, operation.content.clone(), &BACKGROUND_CONTEXT).await, |_| json!(null)),
        Kind::ReadTextFile => ok_or_code(env.read_text_file(a, &BACKGROUND_CONTEXT).await, Value::String),
        Kind::ReadBinaryFile => bytes_result(env.read_binary_file(a, &BACKGROUND_CONTEXT).await),
        Kind::ReadTextLines => ok_or_code(
            env.read_text_lines(a, Some(ReadTextLinesOptions { max_lines: Some(operation.small_a as usize) }), &BACKGROUND_CONTEXT).await,
            |lines| json!(lines),
        ),
        Kind::FileInfo => ok_or_code(env.file_info(a, &BACKGROUND_CONTEXT).await, |info| info_value(root, info)),
        Kind::Exists => ok_or_code(env.exists(a, &BACKGROUND_CONTEXT).await, |exists| json!(exists)),
        Kind::ListDir => ok_or_code(env.list_dir(a, &BACKGROUND_CONTEXT).await, |entries| {
            Value::Array(entries.into_iter().map(|entry| info_value(root, entry)).collect())
        }),
        Kind::CreateDir => ok_or_code(env.create_dir(a, Some(CreateDirOptions { recursive: Some(operation.flag_a) }), &BACKGROUND_CONTEXT).await, |_| json!(null)),
        Kind::Remove => ok_or_code(
            env.remove(a, Some(RemoveOptions { recursive: Some(operation.flag_a), force: Some(operation.flag_b) }), &BACKGROUND_CONTEXT).await,
            |_| json!(null),
        ),
        Kind::RenameFile => ok_or_code(env.rename_file(a, &operation.b, &BACKGROUND_CONTEXT).await, |_| json!(null)),
        Kind::TruncateFile => ok_or_code(env.truncate_file(a, operation.small_a, &BACKGROUND_CONTEXT).await, |_| json!(null)),
        Kind::FlushFile => ok_or_code(env.flush_file(a, &BACKGROUND_CONTEXT).await, |_| json!(null)),
        Kind::CanonicalPath => ok_or_code(env.canonical_path(a, &BACKGROUND_CONTEXT).await, |path| json!(path.replace(root, "<root>"))),
        Kind::OpenBinaryReader => match env.open_binary_reader(a, Some(OpenBinaryOptions { no_follow: operation.flag_a }), &BACKGROUND_CONTEXT).await {
            Ok(reader) => {
                let read = bytes_result(reader.read(operation.small_a, operation.small_b as usize, &BACKGROUND_CONTEXT).await);
                let scan = ok_or_code(
                    reader.scan_lines(ScanLinesOptions { start_line: 0, end_line: None }, &BACKGROUND_CONTEXT).await,
                    |scan| json!(scan),
                );
                reader.close(&BACKGROUND_CONTEXT).await;
                json!([read, scan])
            }
            Err(error) => json!({ "ok": false, "code": error.code.as_str() }),
        },
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn gives_the_same_results_for_random_file_operation_sequences() {
    let connection = Connection::new(ConnectionOptions::new([env!("CARGO_BIN_EXE_pi-env").to_string()]));
    for seed in 1u32..=60 {
        let index = NEXT.fetch_add(1, Ordering::SeqCst);
        let base = std::env::temp_dir();
        let local_dir = base.join(format!("pi-env-diff-local-{}-{seed}-{index}", std::process::id()));
        let remote_dir = base.join(format!("pi-env-diff-remote-{}-{seed}-{index}", std::process::id()));
        std::fs::create_dir_all(&local_dir).unwrap();
        std::fs::create_dir_all(&remote_dir).unwrap();
        let local_root = local_dir.to_string_lossy().into_owned();
        let remote_root = remote_dir.to_string_lossy().into_owned();
        let local = NodeExecutionEnv::new(NodeEnvOptions { cwd: local_root.clone(), ..Default::default() });
        let remote = RemoteExecutionEnv::new(connection.clone(), "pi-env:differential", remote_root.clone());
        let mut next = random(seed);
        for step in 0..25 {
            let operation = random_operation(&mut next);
            let expected = run(&local, &operation, &local_root).await;
            let actual = run(&remote, &operation, &remote_root).await;
            assert_eq!(actual, expected, "seed {seed} step {step} {}", operation.label);
        }
        let _ = std::fs::remove_dir_all(&local_dir);
        let _ = std::fs::remove_dir_all(&remote_dir);
    }
    connection.close().await;
}
