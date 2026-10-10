//! Port of `packages/durable/src/env/node.ts` — the local `ExecutionEnv` on `std::fs`/`tokio::fs`.
//!
//! Function names keep the upstream snake_case mapping; error strings that tests assert are
//! verbatim. Upstream async (fs/promises) maps to tokio::fs; the child-process orchestration
//! (`waitForChildProcess`, spill streams) maps to tokio tasks with the same observable
//! sequencing: no output after settle, post-exit stdio grace of 100 ms extended while a spill
//! write is pending, spill completion awaited before the result settles.
//!
//! Deviations:
//! - D1: Node errno strings (libuv messages like "ENOENT: no such file or directory, open …")
//!   become `std::io::Error` Display strings; the error `code` classification (not_found,
//!   permission_denied, …) and `path` are preserved, and tests assert codes/paths, not OS text.
//! - D2: `process.platform` is a compile-time `cfg`, not a mutable global. Platform selection is
//!   carried through a `Platform` parameter on the shell-config functions so the legacy-WSL
//!   transport tests can exercise the Windows branch on any host; `platform()` reads `cfg!` at
//!   the call sites.
//! - D3: `killProcessTree` maps to `kill(-pid, SIGKILL)` via a direct `extern "C" { fn kill }`
//!   declaration (the tree is spawned in its own process group); Windows keeps the upstream
//!   `taskkill /F /T` spawn.
//! - D4: `randomUUID` becomes a local v4 generator (no UUID dependency); `mkdtemp` is reproduced
//!   with 6 random [A-Za-z0-9] suffix chars like libuv.
//! - D5: JS "safe integer" validation of `offset`/`length`/`size` collapses into the Rust
//!   `u64`/`usize` types (negative and fractional inputs are unrepresentable); the remaining
//!   validations and their messages are kept.
//! - D6: spill output writes are sequential awaited appends on a shared file instead of a
//!   Node WriteStream with high-water-mark backpressure; ordering, completeness, and the
//!   fail-loudly contract are preserved (the artificial slow-write mock test is deferred —
//!   see `node_spill_tests`).
//! - D7: `readFile` with `encoding: "utf8"` decodes with WHATWG replacement semantics
//!   (`from_utf8_lossy`), matching Node's StringDecoder.
//! - D8: `file://` URL conversion handles the `file:///path` form with percent-decoding; other
//!   authorities or malformed URLs stay ordinary paths, matching the upstream catch.
//! - D9: post-spawn child "error" events have no tokio counterpart (spawn errors surface from
//!   `spawn()` itself); `waitForChildProcess`'s rejection path collapses into the spawn error.
//! - D10: `statfs`-based unreliable-filesystem detection lives in `node_watch.rs` (D2 there).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use async_trait::async_trait;
use pi_chord::context::Context;
use tokio::io::{AsyncReadExt as _, AsyncSeekExt as _, AsyncWriteExt as _, SeekFrom};
use tokio::sync::Mutex as AsyncMutex;

use crate::env::decode::StreamDecoder;
use crate::env::line_scan::LineScanner;
use crate::env::node_watch::NodeFileWatcher;
use crate::env::{
    abort_error_like, err, ok, to_file_error, BinaryReader, CreateDirOptions, DirPage, DirReader,
    ExecCommand, ExecutionError, ExecutionErrorCode, FileContent, FileError, FileErrorCode,
    FileInfo, FileKind, FileSystem, FileWatcher, LineScan, NodeWatchOptions, OpenBinaryOptions,
    OutputStream, ReadTextLinesOptions, RemoveOptions, Result, ScanLinesOptions, Shell,
    ShellExecOptions, ShellExecResult, ShellOutputInfo, ShellSpillOptions, TempFileOptions,
    TextLine, TextLineReader, WatchCallback, WatchTarget,
};

const MAX_TIMEOUT_MS: f64 = 2_147_483_647.0;
const MAX_TIMEOUT_SECONDS: f64 = MAX_TIMEOUT_MS / 1000.0;
const EXIT_STDIO_GRACE_MS: u64 = 100;
/// Largest single read of a `BinaryReader`, so a huge `length` allocates only as much as the file yields.
const BINARY_READ_CHUNK: usize = 1024 * 1024;
const TEXT_CHUNK: usize = 64 * 1024;

/// Stable Unix `open(2)` bits used for nonblocking binary opens (`O_NONBLOCK`, `O_NOFOLLOW`).
/// The values differ per platform: Linux uses 0o4000/0o400, Darwin 0x4/0x100.
#[cfg(unix)]
const O_NONBLOCK_FLAG: i32 = if cfg!(target_os = "macos") {
    0x4
} else {
    0o4000
};
#[cfg(unix)]
const O_NOFOLLOW_FLAG: i32 = if cfg!(target_os = "macos") {
    0x100
} else {
    0o400
};

/// Compile-time stand-in for `process.platform` (see D2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Platform {
    Win32,
    Darwin,
    Linux,
    Other,
}

pub(crate) fn platform() -> Platform {
    if cfg!(windows) {
        Platform::Win32
    } else if cfg!(target_os = "macos") {
        Platform::Darwin
    } else if cfg!(target_os = "linux") {
        Platform::Linux
    } else {
        Platform::Other
    }
}

pub(crate) type BoxError = Arc<dyn std::error::Error + Send + Sync>;

fn resolve_timeout_ms(timeout: Option<f64>) -> std::result::Result<Option<f64>, ExecutionError> {
    let Some(timeout) = timeout else {
        return Ok(None);
    };
    if !timeout.is_finite() || timeout <= 0.0 {
        return Err(ExecutionError::new(
            ExecutionErrorCode::Timeout,
            "Invalid timeout: must be a finite number of seconds",
        ));
    }
    let timeout_ms = timeout * 1000.0;
    if timeout_ms > MAX_TIMEOUT_MS {
        return Err(ExecutionError::new(
            ExecutionErrorCode::Timeout,
            format!("Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds"),
        ));
    }
    Ok(Some(timeout_ms))
}

fn homedir() -> String {
    std::env::home_dir()
        .map(|home| home.to_string_lossy().into_owned())
        .unwrap_or_else(|| "/".to_string())
}

/// Minimal `file://` URL → path (upstream `fileURLToPath`); see D8.
fn file_url_to_path(url: &str) -> Option<String> {
    let rest = url.strip_prefix("file://")?;
    let (host, path) = rest.split_once('/')?;
    let path = format!("/{path}");
    if !host.is_empty() && host != "localhost" {
        return None;
    }
    let mut bytes = Vec::with_capacity(path.len());
    let raw = path.as_bytes();
    let mut index = 0;
    while index < raw.len() {
        if raw[index] == b'%' && index + 2 < raw.len() {
            let high = (raw[index + 1] as char).to_digit(16);
            let low = (raw[index + 2] as char).to_digit(16);
            if let (Some(high), Some(low)) = (high, low) {
                bytes.push((high * 16 + low) as u8);
                index += 3;
                continue;
            }
        }
        bytes.push(raw[index]);
        index += 1;
    }
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

fn resolve_path(cwd: &str, path: &str) -> String {
    let mut normalized = path.to_string();
    if normalized == "~" {
        normalized = homedir();
    } else if normalized.starts_with("~/") || (cfg!(windows) && normalized.starts_with("~\\")) {
        normalized = Path::new(&homedir())
            .join(&normalized[2..])
            .to_string_lossy()
            .into_owned();
    } else if normalized.starts_with("file://") {
        if let Some(converted) = file_url_to_path(&normalized) {
            normalized = converted;
        }
        // Keep malformed URLs as ordinary paths so filesystem methods preserve their non-throwing contract.
    }
    let candidate = Path::new(&normalized);
    let resolved = if candidate.is_absolute() {
        std::path::absolute(candidate)
    } else {
        std::path::absolute(Path::new(cwd).join(candidate))
    };
    resolved
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or(normalized)
}

/// Upstream `path.join(...parts)` — concatenates and normalizes lexically without touching the filesystem.
fn join_path_parts(parts: &[String]) -> String {
    let mut joined = PathBuf::new();
    for part in parts {
        joined.push(part);
    }
    let absolute = joined.is_absolute();
    let mut normalized = PathBuf::new();
    if absolute {
        normalized.push("/");
    }
    let mut parents: Vec<String> = Vec::new();
    for component in joined.components() {
        match component {
            std::path::Component::ParentDir => match parents.pop() {
                Some(_) => {}
                None if !absolute => normalized.push(".."),
                None => {}
            },
            std::path::Component::CurDir => {}
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {}
            std::path::Component::Normal(part) => parents.push(part.to_string_lossy().into_owned()),
        }
    }
    for part in parents {
        normalized.push(part);
    }
    let result = normalized.to_string_lossy().into_owned();
    if result.is_empty() {
        return ".".to_string();
    }
    result
}

fn file_kind_from_metadata(metadata: &std::fs::Metadata) -> Option<FileKind> {
    let file_type = metadata.file_type();
    if file_type.is_file() {
        Some(FileKind::File)
    } else if file_type.is_dir() {
        Some(FileKind::Directory)
    } else if file_type.is_symlink() {
        Some(FileKind::Symlink)
    } else {
        None
    }
}

fn file_info_from_metadata(
    path: &str,
    metadata: &std::fs::Metadata,
) -> Result<FileInfo, FileError> {
    let Some(kind) = file_kind_from_metadata(metadata) else {
        return err(FileError::new(
            FileErrorCode::Invalid,
            "Unsupported file type",
            Some(path.to_string()),
        ));
    };
    let mtime_ms = metadata
        .modified()
        .ok()
        .map(|mtime| {
            mtime
                .duration_since(std::time::UNIX_EPOCH)
                .map(|since| since.as_secs_f64() * 1000.0)
        })
        .transpose()
        .unwrap_or(Some(0.0))
        .unwrap_or(0.0);
    ok(FileInfo {
        name: Path::new(path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string()),
        path: path.to_string(),
        kind,
        size: metadata.len(),
        mtime_ms,
    })
}

fn path_exists(path: &str) -> bool {
    Path::new(path).metadata().is_ok()
}

fn spawn_error(error: std::io::Error) -> ExecutionError {
    ExecutionError::new(ExecutionErrorCode::SpawnError, error.to_string())
        .with_cause(abort_error_like(&error))
}

/// SIGKILL to the process group, falling back to the direct child (upstream `killProcessTree`).
pub(crate) fn kill_process_tree(pid: u32) {
    kill_process_tree_on(platform(), pid);
}

pub(crate) fn kill_process_tree_on(target: Platform, pid: u32) {
    match target {
        Platform::Win32 => {
            let system_root =
                std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
            let taskkill = Path::new(&system_root)
                .join("System32")
                .join("taskkill.exe");
            // A failed spawn is ignored, matching the consumed "error" event upstream.
            let _ = std::process::Command::new(taskkill)
                .args(["/F", "/T", "/PID", &pid.to_string()])
                .output();
        }
        _ => {
            // POSIX: the child runs in its own process group (spawn with process_group(0)), so a
            // group-wide SIGKILL reaches descendants; a direct kill covers a reaped group id.
            let pid = pid as i32;
            unsafe {
                if kill(-pid, SIGKILL) != 0 {
                    let _ = kill(pid, SIGKILL);
                }
            }
        }
    }
}

#[cfg(unix)]
mod posix_kill {
    const SIGKILL_VALUE: i32 = 9;
    extern "C" {
        pub fn kill(pid: i32, sig: i32) -> i32;
    }
    pub const SIGKILL: i32 = SIGKILL_VALUE;
}

#[cfg(unix)]
use posix_kill::{kill, SIGKILL};

async fn run_command(command: &str, args: &[&str], timeout_ms: u64) -> (String, Option<i32>) {
    let mut child = match tokio::process::Command::new(command)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return (String::new(), None),
    };
    let Some(stdout) = child.stdout.take() else {
        return (String::new(), None);
    };
    let read = tokio::spawn(async move {
        let mut stdout = stdout;
        let mut buffer = String::new();
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            match stdout.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(read) => buffer.push_str(&String::from_utf8_lossy(&buf[..read])),
            }
        }
        buffer
    });
    let status = match tokio::time::timeout(Duration::from_millis(timeout_ms), child.wait()).await {
        Ok(Ok(status)) => Some(status.code().unwrap_or(-1)),
        _ => {
            let _ = child.start_kill();
            None
        }
    };
    let stdout = read.await.unwrap_or_default();
    let first_match = stdout.trim().lines().next().map(str::to_string);
    match first_match {
        Some(first) if path_exists(&first) => (stdout, status),
        _ => (stdout, status),
    }
}

async fn find_bash_on_path() -> Option<String> {
    let (stdout, status) = if platform() == Platform::Win32 {
        run_command("where", &["bash.exe"], 5000).await
    } else {
        run_command("which", &["bash"], 5000).await
    };
    if status != Some(0) || stdout.is_empty() {
        return None;
    }
    let first_match = stdout.trim().lines().next().map(str::to_string);
    match first_match {
        Some(matched) if path_exists(&matched) => Some(matched),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ShellConfig {
    shell: String,
    args: Vec<String>,
    command_transport: Option<CommandTransport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandTransport {
    Stdin,
}

fn is_legacy_wsl_bash_path(path: &str) -> bool {
    let normalized = path.replace('/', "\\").to_lowercase();
    let Some((drive, rest)) = normalized.split_once(':') else {
        return false;
    };
    let mut drive = drive.chars();
    if !(drive
        .next()
        .is_some_and(|letter| letter.is_ascii_lowercase())
        && drive.next().is_none())
    {
        return false;
    }
    let Some(suffix) = rest.strip_prefix("\\windows\\") else {
        return false;
    };
    let Some(shell) = suffix
        .strip_prefix("system32\\")
        .or_else(|| suffix.strip_prefix("sysnative\\"))
    else {
        return false;
    };
    shell == "bash.exe"
}

fn get_bash_shell_config(platform: Platform, shell: &str) -> ShellConfig {
    if platform == Platform::Win32 && is_legacy_wsl_bash_path(shell) {
        ShellConfig {
            shell: shell.to_string(),
            args: vec!["-s".to_string()],
            command_transport: Some(CommandTransport::Stdin),
        }
    } else {
        ShellConfig {
            shell: shell.to_string(),
            args: vec!["-c".to_string()],
            command_transport: None,
        }
    }
}

async fn get_shell_config(
    platform: Platform,
    custom_shell_path: Option<&str>,
) -> std::result::Result<ShellConfig, ExecutionError> {
    if let Some(custom) = custom_shell_path {
        if path_exists(custom) {
            return Ok(get_bash_shell_config(platform, custom));
        }
        return Err(ExecutionError::new(
            ExecutionErrorCode::ShellUnavailable,
            format!("Custom shell path not found: {custom}"),
        ));
    }
    if platform == Platform::Win32 {
        let mut candidates: Vec<String> = Vec::new();
        if let Ok(program_files) = std::env::var("ProgramFiles") {
            candidates.push(
                Path::new(&program_files)
                    .join("Git")
                    .join("bin")
                    .join("bash.exe")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        if let Ok(program_files_x86) = std::env::var("ProgramFiles(x86)") {
            candidates.push(
                Path::new(&program_files_x86)
                    .join("Git")
                    .join("bin")
                    .join("bash.exe")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        for candidate in &candidates {
            if path_exists(candidate) {
                return Ok(get_bash_shell_config(platform, candidate));
            }
        }
        if let Some(bash_on_path) = find_bash_on_path().await {
            return Ok(get_bash_shell_config(platform, &bash_on_path));
        }
        return Err(ExecutionError::new(
			ExecutionErrorCode::ShellUnavailable,
			format!(
				"No bash shell found. Options:\n  1. Install Git for Windows: https://git-scm.com/download/win\n  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n  3. Configure an explicit shellPath\n\nSearched Git Bash in:\n{}",
				candidates.iter().map(|path| format!("  {path}")).collect::<Vec<_>>().join("\n")
			),
		));
    }
    if path_exists("/bin/bash") {
        return Ok(get_bash_shell_config(platform, "/bin/bash"));
    }
    if let Some(bash_on_path) = find_bash_on_path().await {
        return Ok(get_bash_shell_config(platform, &bash_on_path));
    }
    Ok(ShellConfig {
        shell: "sh".to_string(),
        args: vec!["-c".to_string()],
        command_transport: None,
    })
}

fn get_shell_env(
    shell_env: Option<&HashMap<String, String>>,
    extra_env: Option<&HashMap<String, String>>,
    inherit_env: bool,
) -> HashMap<String, String> {
    let mut merged: HashMap<String, String> = HashMap::new();
    // Upstream: `if (!inheritEnv) return { ...extraEnv }` — neither the process env nor the
    // configured `shellEnv` applies when inheritance is off.
    if !inherit_env {
        if let Some(extra_env) = extra_env {
            for (key, value) in extra_env {
                merged.insert(key.clone(), value.clone());
            }
        }
        return merged;
    }
    for (key, value) in std::env::vars() {
        merged.insert(key, value);
    }
    if let Some(shell_env) = shell_env {
        for (key, value) in shell_env {
            merged.insert(key.clone(), value.clone());
        }
    }
    if let Some(extra_env) = extra_env {
        for (key, value) in extra_env {
            merged.insert(key.clone(), value.clone());
        }
    }
    merged
}

fn closed_result<TValue>(what: &str, path: &str) -> Result<TValue, FileError> {
    err(FileError::new(
        FileErrorCode::Invalid,
        format!("{what} is closed"),
        Some(path.to_string()),
    ))
}

fn symlink_refused(path: &str, cause: Option<BoxError>) -> FileError {
    let error = FileError::new(
        FileErrorCode::Invalid,
        "Refusing to follow a symbolic link",
        Some(path.to_string()),
    );
    match cause {
        Some(cause) => error.with_cause(cause),
        None => error,
    }
}

/// A random v4 UUID (upstream `randomUUID`); see D4.
fn random_uuid() -> String {
    let mut bytes = [0u8; 16];
    if Path::new("/dev/urandom").exists() {
        if let Ok(mut urandom) = std::fs::File::open("/dev/urandom") {
            let _ = std::io::Read::read_exact(&mut urandom, &mut bytes);
        }
    }
    if bytes == [0u8; 16] {
        // Fallback entropy: time, pid, address, and a counter.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or(0) as u64;
        let mut seed =
            nanos ^ (std::process::id() as u64).rotate_left(32) ^ (&bytes as *const _ as u64);
        for byte in bytes.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (seed >> 33) as u8;
        }
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!(
        "&{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )[1..]
        .to_string()
}

pub(crate) fn random_suffix() -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut bytes = [0u8; 6];
    if Path::new("/dev/urandom").exists() {
        if let Ok(mut urandom) = std::fs::File::open("/dev/urandom") {
            let _ = std::io::Read::read_exact(&mut urandom, &mut bytes);
        }
    }
    if bytes.iter().all(|byte| *byte == 0) {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos() as u64)
            .unwrap_or(0);
        let mut seed = nanos ^ (std::process::id() as u64).rotate_left(17);
        for byte in bytes.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (seed >> 33) as u8;
        }
    }
    bytes
        .iter()
        .map(|byte| ALPHABET[(*byte as usize) % ALPHABET.len()] as char)
        .collect()
}

/// Upstream `mkdtemp`: creates the directory, which persists until removed by the caller.
async fn mkdtemp(prefix: &str) -> std::io::Result<String> {
    let base = std::env::temp_dir();
    loop {
        let candidate = base.join(format!("{prefix}{}", random_suffix()));
        match tokio::fs::create_dir(&candidate).await {
            Ok(()) => return Ok(candidate.to_string_lossy().into_owned()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
}

/// Strict LF reader; Node readline does not report whether its final line was newline-terminated.
struct ReaderState {
    file: tokio::fs::File,
    decoder: StreamDecoder,
    buffered: String,
    byte_offset: u64,
    ended: bool,
    closed: bool,
}

pub struct NodeTextLineReader {
    path: String,
    state: AsyncMutex<ReaderState>,
}

impl NodeTextLineReader {
    pub(crate) fn new(file: tokio::fs::File, path: String) -> Self {
        Self {
            path,
            state: AsyncMutex::new(ReaderState {
                file,
                decoder: StreamDecoder::new(),
                buffered: String::new(),
                byte_offset: 0,
                ended: false,
                closed: false,
            }),
        }
    }
}

#[async_trait]
impl TextLineReader for NodeTextLineReader {
    async fn read_line(&self, context: &Context) -> Result<Option<TextLine>, FileError> {
        if let Some(aborted) = abort_result(context, Some(&self.path)) {
            return Err(aborted);
        }
        let mut state = self.state.lock().await;
        if state.closed {
            return err(FileError::new(
                FileErrorCode::Invalid,
                "Text line reader is closed",
                Some(self.path.clone()),
            ));
        }
        let mut chunk = vec![0u8; TEXT_CHUNK];
        loop {
            if let Some(newline) = state.buffered.find('\n') {
                let text = state.buffered[..newline].to_string();
                state.buffered.drain(..newline + 1);
                return ok(Some(TextLine {
                    text,
                    terminated: true,
                }));
            }
            if state.ended {
                if state.buffered.is_empty() {
                    return ok(None);
                }
                let text = std::mem::take(&mut state.buffered);
                return ok(Some(TextLine {
                    text,
                    terminated: false,
                }));
            }
            // Explicit positions allow an aborted read to be retried without skipping bytes.
            let byte_offset = state.byte_offset;
            state
                .file
                .seek(SeekFrom::Start(byte_offset))
                .await
                .map_err(|error| to_file_error(&error, Some(&self.path)))?;
            let bytes_read = state
                .file
                .read(&mut chunk)
                .await
                .map_err(|error| to_file_error(&error, Some(&self.path)))?;
            // Abort the read's results, not its byte position (upstream checks before advancing).
            if let Some(aborted) = abort_result(context, Some(&self.path)) {
                return Err(aborted);
            }
            state.byte_offset += bytes_read as u64;
            let decoded = if bytes_read == 0 {
                state.ended = true;
                state.decoder.decode(None)
            } else {
                state.decoder.decode(Some(&chunk[..bytes_read]))
            };
            state.buffered.push_str(&decoded);
        }
    }

    async fn close(&self, _context: &Context) {
        let mut state = self.state.lock().await;
        if state.closed {
            return;
        }
        state.closed = true;
        state.buffered.clear();
        // Closing is best-effort, including after cancellation or an earlier I/O failure.
        let _ = state.file.shutdown().await;
    }
}

struct BinaryState {
    file: tokio::fs::File,
    closed: bool,
}

pub struct NodeBinaryReader {
    path: String,
    state: AsyncMutex<BinaryState>,
}

impl NodeBinaryReader {
    pub(crate) fn new(file: tokio::fs::File, path: String) -> Self {
        Self {
            path,
            state: AsyncMutex::new(BinaryState {
                file,
                closed: false,
            }),
        }
    }
}

#[async_trait]
impl BinaryReader for NodeBinaryReader {
    async fn info(&self, context: &Context) -> Result<FileInfo, FileError> {
        if let Some(aborted) = abort_result(context, Some(&self.path)) {
            return Err(aborted);
        }
        let state = self.state.lock().await;
        if state.closed {
            return closed_result("Binary reader", &self.path);
        }
        let metadata = state
            .file
            .metadata()
            .await
            .map_err(|error| to_file_error(&error, Some(&self.path)))?;
        file_info_from_metadata(&self.path, &metadata)
    }

    async fn read(
        &self,
        offset: u64,
        length: usize,
        context: &Context,
    ) -> Result<Vec<u8>, FileError> {
        if let Some(aborted) = abort_result(context, Some(&self.path)) {
            return Err(aborted);
        }
        let mut state = self.state.lock().await;
        if state.closed {
            return closed_result("Binary reader", &self.path);
        }
        let mut chunks: Vec<Vec<u8>> = Vec::new();
        let mut total = 0usize;
        while total < length {
            let chunk_size = std::cmp::min(length - total, BINARY_READ_CHUNK);
            let mut chunk = vec![0u8; chunk_size];
            state
                .file
                .seek(SeekFrom::Start(offset + total as u64))
                .await
                .map_err(|error| to_file_error(&error, Some(&self.path)))?;
            let bytes_read = state
                .file
                .read(&mut chunk)
                .await
                .map_err(|error| to_file_error(&error, Some(&self.path)))?;
            if let Some(aborted) = abort_result(context, Some(&self.path)) {
                return Err(aborted);
            }
            if bytes_read == 0 {
                break;
            }
            chunk.truncate(bytes_read);
            total += bytes_read;
            chunks.push(chunk);
        }
        if chunks.len() == 1 {
            return ok(chunks.pop().unwrap_or_default());
        }
        let mut bytes = Vec::with_capacity(total);
        for chunk in chunks {
            bytes.extend_from_slice(&chunk);
        }
        ok(bytes)
    }

    async fn scan_lines(
        &self,
        options: ScanLinesOptions,
        context: &Context,
    ) -> Result<LineScan, FileError> {
        if let Some(aborted) = abort_result(context, Some(&self.path)) {
            return Err(aborted);
        }
        let mut state = self.state.lock().await;
        if state.closed {
            return closed_result("Binary reader", &self.path);
        }
        let mut scanner = match LineScanner::new(options.start_line, options.end_line) {
            Ok(scanner) => scanner,
            Err(_) => {
                return err(FileError::new(
                    FileErrorCode::Invalid,
                    "Invalid line range",
                    Some(self.path.clone()),
                ))
            }
        };
        let mut chunk = vec![0u8; TEXT_CHUNK];
        let mut position = 0u64;
        loop {
            state
                .file
                .seek(SeekFrom::Start(position))
                .await
                .map_err(|error| to_file_error(&error, Some(&self.path)))?;
            let bytes_read = state
                .file
                .read(&mut chunk)
                .await
                .map_err(|error| to_file_error(&error, Some(&self.path)))?;
            if let Some(aborted) = abort_result(context, Some(&self.path)) {
                return Err(aborted);
            }
            if bytes_read == 0 {
                return ok(scanner.finish());
            }
            scanner.push(&chunk[..bytes_read]);
            position += bytes_read as u64;
        }
    }

    async fn close(&self, _context: &Context) {
        let mut state = self.state.lock().await;
        if state.closed {
            return;
        }
        state.closed = true;
        let _ = state.file.shutdown().await;
    }
}

struct DirState {
    entries: tokio::fs::ReadDir,
    done: bool,
    closed: bool,
}

pub struct NodeDirReader {
    path: String,
    state: AsyncMutex<DirState>,
}

impl NodeDirReader {
    pub(crate) fn new(entries: tokio::fs::ReadDir, path: String) -> Self {
        Self {
            path,
            state: AsyncMutex::new(DirState {
                entries,
                done: false,
                closed: false,
            }),
        }
    }
}

#[async_trait]
impl DirReader for NodeDirReader {
    async fn next(&self, max_entries: usize, context: &Context) -> Result<DirPage, FileError> {
        if let Some(aborted) = abort_result(context, Some(&self.path)) {
            return Err(aborted);
        }
        let mut state = self.state.lock().await;
        if state.closed {
            return closed_result("Directory reader", &self.path);
        }
        if max_entries == 0 {
            return err(FileError::new(
                FileErrorCode::Invalid,
                "maxEntries must be a positive safe integer",
                Some(self.path.clone()),
            ));
        }
        let mut entries: Vec<FileInfo> = Vec::new();
        while !state.done && entries.len() < max_entries {
            let Some(entry) = state
                .entries
                .next_entry()
                .await
                .map_err(|error| to_file_error(&error, Some(&self.path)))?
            else {
                state.done = true;
                break;
            };
            if let Some(aborted) = abort_result(context, Some(&self.path)) {
                return Err(aborted);
            }
            let entry_path = Path::new(&self.path)
                .join(entry.file_name())
                .to_string_lossy()
                .into_owned();
            let metadata = match tokio::fs::symlink_metadata(&entry_path).await {
                Ok(metadata) => metadata,
                // Removed between enumeration and lstat: not part of the listing any more.
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return err(to_file_error(&error, Some(&entry_path))),
            };
            if let Ok(info) = file_info_from_metadata(&entry_path, &metadata) {
                entries.push(info);
            }
        }
        ok(DirPage {
            entries,
            done: state.done,
        })
    }

    async fn close(&self, _context: &Context) {
        let mut state = self.state.lock().await;
        if state.closed {
            return;
        }
        state.closed = true;
    }
}

fn abort_result(context: &Context, path: Option<&str>) -> Option<FileError> {
    context
        .abort_signal()
        .filter(|signal| signal.aborted())
        .map(|_| FileError::aborted(path.map(str::to_string)))
}

fn open_text_options() -> tokio::fs::OpenOptions {
    let mut options = tokio::fs::OpenOptions::new();
    options.read(true);
    options
}

fn open_write_options(append: bool) -> tokio::fs::OpenOptions {
    let mut options = tokio::fs::OpenOptions::new();
    options
        .write(true)
        .create(true)
        .truncate(!append)
        .append(append);
    options
}

fn open_read_write_options() -> tokio::fs::OpenOptions {
    let mut options = tokio::fs::OpenOptions::new();
    options.read(true).write(true);
    options
}

/// The local `ExecutionEnv`: every local environment sees the same files (`node:local`).
pub struct NodeExecutionEnv {
    cwd: String,
    shell_path: Option<String>,
    shell_env: Option<HashMap<String, String>>,
    watch_options: NodeWatchOptions,
    active_child_pids: StdMutex<HashSet<u32>>,
    /// Test seam mirroring upstream `FailingSpillExecutionEnv`: `createTempFile` with the spill
    /// prefix returns a path in a missing directory (see `node_spill_tests`).
    #[cfg(test)]
    pub(crate) failing_spill: bool,
}

impl NodeExecutionEnv {
    pub fn new(options: crate::env::NodeEnvOptions) -> Self {
        Self {
            cwd: options.cwd,
            shell_path: options.shell_path,
            shell_env: options.shell_env,
            watch_options: options.watch,
            active_child_pids: StdMutex::new(HashSet::new()),
            #[cfg(test)]
            failing_spill: false,
        }
    }
}

#[async_trait]
impl FileSystem for NodeExecutionEnv {
    fn id(&self) -> &str {
        "node:local"
    }

    fn cwd(&self) -> &str {
        &self.cwd
    }

    async fn absolute_path(&self, path: &str, _context: &Context) -> Result<String, FileError> {
        ok(resolve_path(&self.cwd, path))
    }

    async fn join_path(&self, parts: &[String], _context: &Context) -> Result<String, FileError> {
        ok(join_path_parts(parts))
    }

    async fn read_text_file(&self, path: &str, context: &Context) -> Result<String, FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        match tokio::fs::read(&resolved).await {
            Ok(bytes) => ok(String::from_utf8_lossy(&bytes).into_owned()),
            Err(error) => err(to_file_error(&error, Some(&resolved))),
        }
    }

    async fn open_text_line_reader(
        &self,
        path: &str,
        context: &Context,
    ) -> Result<Arc<dyn TextLineReader>, FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        match open_text_options().open(&resolved).await {
            Ok(file) => {
                if let Some(aborted) = abort_result(context, Some(&resolved)) {
                    return Err(aborted);
                }
                ok(Arc::new(NodeTextLineReader::new(file, resolved)))
            }
            Err(error) => err(to_file_error(&error, Some(&resolved))),
        }
    }

    async fn read_text_lines(
        &self,
        path: &str,
        options: Option<ReadTextLinesOptions>,
        context: &Context,
    ) -> Result<Vec<String>, FileError> {
        if options.and_then(|options| options.max_lines) == Some(0) {
            return ok(Vec::new());
        }
        let opened = self.open_text_line_reader(path, context).await?;
        let mut lines: Vec<String> = Vec::new();
        let result = loop {
            if let Some(max_lines) = options.and_then(|options| options.max_lines) {
                if lines.len() >= max_lines {
                    break Ok(lines);
                }
            }
            match opened.read_line(context).await {
                Ok(Some(line)) => lines.push(line.text),
                Ok(None) => break Ok(lines),
                Err(error) => break Err(error),
            }
        };
        opened.close(context).await;
        result
    }

    async fn read_binary_file(&self, path: &str, context: &Context) -> Result<Vec<u8>, FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        match tokio::fs::read(&resolved).await {
            Ok(bytes) => ok(bytes),
            Err(error) => err(to_file_error(&error, Some(&resolved))),
        }
    }

    async fn open_binary_reader(
        &self,
        path: &str,
        options: Option<OpenBinaryOptions>,
        context: &Context,
    ) -> Result<Arc<dyn BinaryReader>, FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        let no_follow = options.is_some_and(|options| options.no_follow);
        let mut open = tokio::fs::OpenOptions::new();
        open.read(true);
        #[cfg(unix)]
        {
            // Nonblocking, so opening a FIFO does not wait for a writer. The flag values are the
            // stable Unix open(2) bits (O_NONBLOCK = 04000, O_NOFOLLOW = 0400); no libc dependency.
            let mut flags: i32 = O_NONBLOCK_FLAG;
            if no_follow {
                flags |= O_NOFOLLOW_FLAG;
            }
            open.custom_flags(flags);
        }
        #[cfg(windows)]
        {
            // Windows has no O_NOFOLLOW; check the final component before opening (not race-free).
            if no_follow {
                match tokio::fs::symlink_metadata(&resolved).await {
                    Ok(metadata) if metadata.file_type().is_symlink() => {
                        return err(symlink_refused(&resolved, None))
                    }
                    Ok(_) => {}
                    Err(error) => return err(to_file_error(&error, Some(&resolved))),
                }
            }
        }
        match open.open(&resolved).await {
            Ok(file) => {
                let metadata = match file.metadata().await {
                    Ok(metadata) => metadata,
                    Err(error) => return err(to_file_error(&error, Some(&resolved))),
                };
                if !metadata.is_file() {
                    return err(if metadata.is_dir() {
                        FileError::new(
                            FileErrorCode::IsDirectory,
                            "EISDIR: illegal operation on a directory, read",
                            Some(resolved.clone()),
                        )
                    } else {
                        FileError::new(
                            FileErrorCode::Invalid,
                            "Not a regular file",
                            Some(resolved.clone()),
                        )
                    });
                }
                if let Some(aborted) = abort_result(context, Some(&resolved)) {
                    return Err(aborted);
                }
                ok(Arc::new(NodeBinaryReader::new(file, resolved)))
            }
            Err(error) => {
                // O_NOFOLLOW reports a final-component symlink as ELOOP (Linux errno 40, Darwin 62;
                // `ErrorKind::FilesystemLoop` is still unstable, so match the raw errnos).
                if no_follow && matches!(error.raw_os_error(), Some(40) | Some(62)) {
                    return err(symlink_refused(&resolved, Some(abort_error_like(&error))));
                }
                err(to_file_error(&error, Some(&resolved)))
            }
        }
    }

    async fn write_file(
        &self,
        path: &str,
        content: FileContent,
        context: &Context,
    ) -> Result<(), FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        if let Err(error) = write_content(&resolved, &content, false).await {
            return err(to_file_error(&error, Some(&resolved)));
        }
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        ok(())
    }

    async fn append_file(
        &self,
        path: &str,
        content: FileContent,
        context: &Context,
    ) -> Result<(), FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        if let Err(error) = write_content(&resolved, &content, true).await {
            return err(to_file_error(&error, Some(&resolved)));
        }
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        ok(())
    }

    async fn truncate_file(
        &self,
        path: &str,
        size: u64,
        context: &Context,
    ) -> Result<(), FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        let result = async {
            let mut file = open_read_write_options()
                .open(&resolved)
                .await
                .map_err(|error| to_file_error(&error, Some(&resolved)))?;
            file.set_len(size)
                .await
                .map_err(|error| to_file_error(&error, Some(&resolved)))?;
            let aborted = abort_result(context, Some(&resolved));
            file.shutdown().await.ok();
            match aborted {
                Some(aborted) => Err(aborted),
                None => Ok(()),
            }
        };
        match result.await {
            Ok(()) => ok(()),
            // The async block maps I/O errors onto `FileError` (including is_directory) already.
            Err(error) => err(error),
        }
    }

    async fn flush_file(&self, path: &str, context: &Context) -> Result<(), FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        let result = async {
            let mut file = open_read_write_options()
                .open(&resolved)
                .await
                .map_err(|error| to_file_error(&error, Some(&resolved)))?;
            // POSIX refuses to open a directory for writing; Windows opens it, so check explicitly.
            if file
                .metadata()
                .await
                .map(|metadata| metadata.is_dir())
                .unwrap_or(false)
            {
                return Err(FileError::new(
                    FileErrorCode::IsDirectory,
                    "Is a directory",
                    Some(resolved.clone()),
                ));
            }
            file.sync_all()
                .await
                .map_err(|error| to_file_error(&error, Some(&resolved)))?;
            let aborted = abort_result(context, Some(&resolved));
            file.shutdown().await.ok();
            match aborted {
                Some(aborted) => Err(aborted),
                None => Ok(()),
            }
        };
        match result.await {
            Ok(()) => ok(()),
            // The async block maps I/O errors onto `FileError` already.
            Err(error) => err(error),
        }
    }

    async fn rename_file(
        &self,
        source_path: &str,
        destination_path: &str,
        context: &Context,
    ) -> Result<(), FileError> {
        let source = resolve_path(&self.cwd, source_path);
        let destination = resolve_path(&self.cwd, destination_path);
        if let Some(aborted) = abort_result(context, Some(&destination)) {
            return Err(aborted);
        }
        match tokio::fs::rename(&source, &destination).await {
            Ok(()) => ok(()),
            Err(error) => err(to_file_error(&error, Some(&source))),
        }
    }

    async fn file_info(&self, path: &str, context: &Context) -> Result<FileInfo, FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        match tokio::fs::symlink_metadata(&resolved).await {
            Ok(metadata) => file_info_from_metadata(&resolved, &metadata),
            Err(error) => err(to_file_error(&error, Some(&resolved))),
        }
    }

    async fn list_dir(&self, path: &str, context: &Context) -> Result<Vec<FileInfo>, FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        let mut entries = match tokio::fs::read_dir(&resolved).await {
            Ok(entries) => entries,
            Err(error) => return err(to_file_error(&error, Some(&resolved))),
        };
        let mut named: Vec<(std::ffi::OsString, FileInfo)> = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| to_file_error(&error, Some(&resolved)))?
        {
            if let Some(aborted) = abort_result(context, Some(&resolved)) {
                return Err(aborted);
            }
            let entry_path = Path::new(&resolved)
                .join(entry.file_name())
                .to_string_lossy()
                .into_owned();
            match tokio::fs::symlink_metadata(&entry_path).await {
                Ok(metadata) => {
                    if let Ok(info) = file_info_from_metadata(&entry_path, &metadata) {
                        named.push((entry.file_name(), info));
                    }
                }
                Err(error) => return err(to_file_error(&error, Some(&entry_path))),
            }
        }
        // Node's `readdir` scans with libuv, whose scandir order on POSIX is alphasort over raw
        // name bytes; sort the same way so listings are order-identical (D-note: on non-Unix the
        // encoded bytes are the comparison key, since `FileInfo.path` is lossy UTF-8).
        named.sort_by(|(a, _), (b, _)| a.as_encoded_bytes().cmp(b.as_encoded_bytes()));
        ok(named.into_iter().map(|(_, info)| info).collect())
    }

    async fn open_dir_reader(
        &self,
        path: &str,
        context: &Context,
    ) -> Result<Arc<dyn DirReader>, FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        match tokio::fs::read_dir(&resolved).await {
            Ok(entries) => {
                if let Some(aborted) = abort_result(context, Some(&resolved)) {
                    return Err(aborted);
                }
                ok(Arc::new(NodeDirReader::new(entries, resolved)))
            }
            Err(error) => err(to_file_error(&error, Some(&resolved))),
        }
    }

    async fn watch(
        &self,
        targets: &[WatchTarget],
        on_change: WatchCallback,
        context: &Context,
    ) -> Result<Arc<dyn crate::env::FileWatcher>, FileError> {
        if let Some(aborted) = abort_result(context, None) {
            return Err(aborted);
        }
        let cwd = self.cwd.clone();
        match NodeFileWatcher::open(
            targets,
            move |path| resolve_path(&cwd, path),
            on_change,
            self.watch_options.clone(),
        )
        .await
        {
            Ok(watcher) => {
                if let Some(aborted) = abort_result(context, None) {
                    watcher.close(context).await;
                    return Err(aborted);
                }
                Ok(Arc::new(watcher))
            }
            Err(error) => Err(error),
        }
    }

    async fn canonical_path(&self, path: &str, context: &Context) -> Result<String, FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        match tokio::fs::canonicalize(&resolved).await {
            Ok(path) => ok(path.to_string_lossy().into_owned()),
            Err(error) => err(to_file_error(&error, Some(&resolved))),
        }
    }

    async fn exists(&self, path: &str, context: &Context) -> Result<bool, FileError> {
        match self.file_info(path, context).await {
            Ok(_) => ok(true),
            Err(error) if error.code == FileErrorCode::NotFound => ok(false),
            Err(error) => Err(error),
        }
    }

    async fn create_dir(
        &self,
        path: &str,
        options: Option<CreateDirOptions>,
        context: &Context,
    ) -> Result<(), FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        let recursive = options
            .and_then(|options| options.recursive)
            .unwrap_or(true);
        let result = if recursive {
            tokio::fs::create_dir_all(&resolved).await
        } else {
            tokio::fs::create_dir(&resolved).await
        };
        match result {
            Ok(()) => ok(()),
            Err(error) => err(to_file_error(&error, Some(&resolved))),
        }
    }

    async fn remove(
        &self,
        path: &str,
        options: Option<RemoveOptions>,
        context: &Context,
    ) -> Result<(), FileError> {
        let resolved = resolve_path(&self.cwd, path);
        if let Some(aborted) = abort_result(context, Some(&resolved)) {
            return Err(aborted);
        }
        let recursive = options
            .and_then(|options| options.recursive)
            .unwrap_or(false);
        let force = options.and_then(|options| options.force).unwrap_or(false);
        let result = remove_path(&resolved, recursive, force).await;
        match result {
            Ok(()) => ok(()),
            Err(error) => {
                if force && error.kind() == std::io::ErrorKind::NotFound {
                    return ok(());
                }
                err(to_file_error(&error, Some(&resolved)))
            }
        }
    }

    async fn create_temp_dir(
        &self,
        prefix: Option<&str>,
        context: &Context,
    ) -> Result<String, FileError> {
        if let Some(aborted) = abort_result(context, None) {
            return Err(aborted);
        }
        let prefix = prefix.unwrap_or("tmp-");
        match mkdtemp(prefix).await {
            Ok(dir) => ok(dir),
            Err(error) => err(to_file_error(&error, None)),
        }
    }

    async fn create_temp_file(
        &self,
        options: Option<TempFileOptions>,
        context: &Context,
    ) -> Result<String, FileError> {
        #[cfg_attr(not(test), allow(unused_variables))]
        let prefix = options.as_ref().and_then(|options| options.prefix.clone());
        #[cfg(test)]
        if self.failing_spill && prefix.as_deref() == Some("pi-output-") {
            return ok(Path::new(&self.cwd)
                .join("missing")
                .join("spill.log")
                .to_string_lossy()
                .into_owned());
        }
        let dir = self.create_temp_dir(Some("tmp-"), context).await?;
        let options = options.unwrap_or_default();
        let file_path = Path::new(&dir)
            .join(format!(
                "{}{}{}",
                options.prefix.unwrap_or_default(),
                random_uuid(),
                options.suffix.unwrap_or_default()
            ))
            .to_string_lossy()
            .into_owned();
        match tokio::fs::File::create(&file_path).await {
            Ok(_) => ok(file_path),
            Err(error) => err(to_file_error(&error, Some(&file_path))),
        }
    }

    async fn cleanup(&self, _context: &Context) {
        let pids = lock(&self.active_child_pids).drain().collect::<Vec<_>>();
        for pid in pids {
            kill_process_tree(pid);
        }
    }
}

async fn write_content(resolved: &str, content: &FileContent, append: bool) -> std::io::Result<()> {
    if let Some(parent) = Path::new(resolved).parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let bytes: std::borrow::Cow<[u8]> = match content {
        FileContent::Text(text) => std::borrow::Cow::Borrowed(text.as_bytes()),
        FileContent::Binary(bytes) => std::borrow::Cow::Borrowed(bytes),
    };
    let mut file = open_write_options(append).open(resolved).await?;
    file.write_all(&bytes).await?;
    file.flush().await?;
    Ok(())
}

async fn remove_path(resolved: &str, recursive: bool, force: bool) -> std::io::Result<()> {
    let metadata = match tokio::fs::symlink_metadata(resolved).await {
        Ok(metadata) => metadata,
        Err(error) if force && error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        if !recursive {
            // Node `rm` on a directory without `recursive` fails with EISDIR.
            return Err(std::io::Error::new(
                std::io::ErrorKind::IsADirectory,
                "Path is a directory",
            ));
        }
        tokio::fs::remove_dir_all(resolved).await
    } else {
        tokio::fs::remove_file(resolved).await
    }
}

fn to_file_error_io(error: &std::io::Error, resolved: &str) -> FileError {
    to_file_error(error, Some(resolved))
}

#[async_trait]
impl Shell for NodeExecutionEnv {
    async fn exec(
        &self,
        command: ExecCommand,
        options: Option<ShellExecOptions>,
        context: &Context,
    ) -> std::result::Result<ShellExecResult, ExecutionError> {
        self.exec_impl(command, options, context).await
    }

    async fn cleanup(&self, _context: &Context) {
        let pids = lock(&self.active_child_pids).drain().collect::<Vec<_>>();
        for pid in pids {
            kill_process_tree(pid);
        }
    }
}

struct ExecShared {
    /// The spawned child's pid, killed on callback or spill failure (upstream `onAbort`).
    pid: Option<u32>,
    timed_out: AtomicBool,
    callback_error: StdMutex<Option<ExecutionError>>,
    on_output: Option<Arc<dyn Fn(&str, &crate::env::ShellOutputInfo) + Send + Sync>>,
    stdout_decoder: AsyncMutex<StreamDecoder>,
    stderr_decoder: AsyncMutex<StreamDecoder>,
    spill: Option<ShellSpillOptions>,
    spill_started: AtomicBool,
    spill_pending_writes: AtomicU32,
    spill_error: StdMutex<Option<ExecutionError>>,
    spill_state: AsyncMutex<SpillState>,
    spill_prefix: StdMutex<Vec<Vec<u8>>>,
    seen_bytes: AtomicU64,
    seen_newlines: AtomicU64,
}

struct SpillState {
    file: Option<tokio::fs::File>,
    path: Option<String>,
}

/// Locks a std mutex, recovering from poisoning (a panicking callback must not wedge `exec`).
fn lock<T>(mutex: &StdMutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn kill_pid(pid: Option<u32>) {
    if let Some(pid) = pid {
        kill_process_tree(pid);
    }
}

impl ExecShared {
    /// No output reaches the caller after `exec` settled or the callback already failed;
    /// a panicking callback becomes a `callback_error` (upstream: a throwing one) and kills
    /// the command.
    fn emit(&self, text: &str, stream: OutputStream) {
        if text.is_empty() || self.on_output.is_none() || lock(&self.callback_error).is_some() {
            return;
        }
        let info = ShellOutputInfo::for_stream(stream);
        let on_output = self.on_output.as_ref().expect("checked above");
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| on_output(text, &info)));
        if result.is_err() {
            let mut slot = lock(&self.callback_error);
            if slot.is_none() {
                *slot = Some(ExecutionError::new(
                    ExecutionErrorCode::CallbackError,
                    "callback failed",
                ));
            }
            kill_pid(self.pid);
        }
    }

    fn fail_spill(&self, message: String) {
        let mut slot = lock(&self.spill_error);
        if slot.is_none() {
            *slot = Some(ExecutionError::new(
                ExecutionErrorCode::Unknown,
                format!("Failed to preserve complete shell output: {message}"),
            ));
            drop(slot);
            kill_pid(self.pid);
        }
    }

    fn decoder(&self, stream: OutputStream) -> &AsyncMutex<StreamDecoder> {
        match stream {
            OutputStream::Stdout => &self.stdout_decoder,
            OutputStream::Stderr => &self.stderr_decoder,
        }
    }

    /// Appends one chunk to the spill file. Sequential awaited appends on the shared file keep
    /// ordering and completeness (see node.rs D6).
    async fn spill_write(&self, chunk: Vec<u8>) {
        if chunk.is_empty() {
            return;
        }
        self.spill_pending_writes.fetch_add(1, Ordering::SeqCst);
        let mut state = self.spill_state.lock().await;
        if let Some(file) = state.file.as_mut() {
            if let Err(error) = file.write_all(&chunk).await {
                self.spill_pending_writes.fetch_sub(1, Ordering::SeqCst);
                self.fail_spill(error.to_string());
                return;
            }
        }
        self.spill_pending_writes.fetch_sub(1, Ordering::SeqCst);
    }
}

impl NodeExecutionEnv {
    /// Starts the spill lazily (upstream `startSpill`): the first chunk that crosses either
    /// threshold opens the temporary file and drains the buffered prefix into it.
    async fn exec_start_spill(&self, shared: &Arc<ExecShared>, chunk: Vec<u8>, context: &Context) {
        if !shared.spill_started.swap(true, Ordering::SeqCst) {
            let created = self
                .create_temp_file(
                    Some(TempFileOptions {
                        prefix: Some("pi-output-".to_string()),
                        suffix: Some(".log".to_string()),
                    }),
                    context,
                )
                .await;
            let path = match created {
                Ok(path) => path,
                Err(error) => {
                    shared.fail_spill(error.message.clone());
                    return;
                }
            };
            match open_write_options(true).open(&path).await {
                Ok(file) => {
                    *shared.spill_state.lock().await = SpillState {
                        file: Some(file),
                        path: Some(path),
                    };
                }
                Err(error) => {
                    shared.fail_spill(error.to_string());
                    return;
                }
            }
        }
        shared.spill_write(chunk).await;
    }

    /// Upstream `feed`: emit the decoded chunk, then decide whether it belongs in the spill.
    async fn exec_feed(
        &self,
        shared: &Arc<ExecShared>,
        stream: OutputStream,
        chunk: Vec<u8>,
        context: &Context,
    ) {
        let text = shared.decoder(stream).lock().await.decode(Some(&chunk));
        shared.emit(&text, stream);
        let Some(spill) = shared.spill else { return };
        if chunk.is_empty() {
            return;
        }
        if shared.spill_started.load(Ordering::SeqCst) {
            shared.spill_write(chunk).await;
            return;
        }
        // Output seen before the spill starts: counted against the thresholds and buffered for the spill's prefix.
        shared
            .seen_bytes
            .fetch_add(chunk.len() as u64, Ordering::SeqCst);
        let mut newlines = shared.seen_newlines.fetch_add(
            chunk.iter().filter(|byte| **byte == 0x0a).count() as u64,
            Ordering::SeqCst,
        );
        newlines += chunk.iter().filter(|byte| **byte == 0x0a).count() as u64;
        let lines = newlines + if chunk[chunk.len() - 1] == 0x0a { 0 } else { 1 };
        let seen_bytes = shared.seen_bytes.load(Ordering::SeqCst);
        if seen_bytes <= spill.after_bytes && lines <= spill.after_lines {
            lock(&shared.spill_prefix).push(chunk);
            return;
        }
        let prefix_chunks = std::mem::take(&mut *lock(&shared.spill_prefix));
        for prefix in prefix_chunks {
            self.exec_start_spill(shared, prefix, context).await;
        }
        self.exec_start_spill(shared, chunk, context).await;
    }

    /// Upstream `finishSpill`: wait for pending writes, then flush and close the file.
    async fn exec_finish_spill(shared: &ExecShared) {
        loop {
            if shared.spill_pending_writes.load(Ordering::SeqCst) == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let mut state = shared.spill_state.lock().await;
        if let Some(file) = state.file.as_mut() {
            if let Err(error) = file.flush().await {
                shared.fail_spill(error.to_string());
            }
            if let Err(error) = file.sync_all().await {
                shared.fail_spill(error.to_string());
            }
        }
        state.file = None;
    }

    async fn exec_impl(
        &self,
        command: ExecCommand,
        options: Option<ShellExecOptions>,
        context: &Context,
    ) -> std::result::Result<ShellExecResult, ExecutionError> {
        let options = options.unwrap_or_default();
        let signal = context.abort_signal();
        if signal.as_ref().is_some_and(|signal| signal.aborted()) {
            return Err(ExecutionError::new(ExecutionErrorCode::Aborted, "aborted"));
        }
        let timeout_ms = match resolve_timeout_ms(options.timeout)? {
            Some(timeout_ms) => timeout_ms as u64,
            None => u64::MAX,
        };
        let cwd = match options.cwd.as_deref() {
            Some(cwd) => resolve_path(&self.cwd, cwd),
            None => self.cwd.clone(),
        };
        // A string runs through the shell; an argv array runs its program directly, without shell parsing.
        let (program, args, stdin_command) = match &command {
            ExecCommand::Shell(text) => {
                let config = get_shell_config(platform(), self.shell_path.as_deref()).await?;
                match config.command_transport {
                    Some(CommandTransport::Stdin) => {
                        (config.shell, config.args, Some(text.clone()))
                    }
                    None => {
                        let mut args = config.args;
                        args.push(text.clone());
                        (config.shell, args, None)
                    }
                }
            }
            ExecCommand::Argv(argv) => {
                let Some(first) = argv.first() else {
                    return Err(ExecutionError::new(
                        ExecutionErrorCode::SpawnError,
                        "Empty argv: no program to run",
                    ));
                };
                (first.clone(), argv[1..].to_vec(), None)
            }
        };
        if !path_exists(&cwd) {
            return Err(ExecutionError::new(
                ExecutionErrorCode::SpawnError,
                format!("Working directory does not exist: {cwd}\nCannot execute bash commands."),
            ));
        }

        let mut cmd = tokio::process::Command::new(&program);
        cmd.args(&args)
            .current_dir(&cwd)
            .stdin(if stdin_command.is_some() {
                std::process::Stdio::piped()
            } else {
                std::process::Stdio::null()
            })
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .env_clear();
        for (key, value) in get_shell_env(
            self.shell_env.as_ref(),
            options.env.as_ref(),
            options.inherit_env.unwrap_or(true),
        ) {
            cmd.env(key, value);
        }
        // Own process group, so aborting or timing out kills the whole tree (upstream `detached`).
        #[cfg(unix)]
        cmd.process_group(0);
        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(error) => return Err(spawn_error(error)),
        };
        let pid = child.id();
        if let Some(pid) = pid {
            lock(&self.active_child_pids).insert(pid);
        }

        let shared = Arc::new(ExecShared {
            pid,
            timed_out: AtomicBool::new(false),
            callback_error: StdMutex::new(None),
            on_output: options.on_output.clone(),
            stdout_decoder: AsyncMutex::new(StreamDecoder::new()),
            stderr_decoder: AsyncMutex::new(StreamDecoder::new()),
            spill: options.spill,
            spill_started: AtomicBool::new(false),
            spill_pending_writes: AtomicU32::new(0),
            spill_error: StdMutex::new(None),
            spill_state: AsyncMutex::new(SpillState {
                file: None,
                path: None,
            }),
            spill_prefix: StdMutex::new(Vec::new()),
            seen_bytes: AtomicU64::new(0),
            seen_newlines: AtomicU64::new(0),
        });

        if let Some(stdin_command) = &stdin_command {
            if let Some(mut stdin) = child.stdin.take() {
                // A failed write to the closed pipe is ignored, like upstream `stdin.on("error", () => {})`.
                let _ = stdin.write_all(stdin_command.as_bytes()).await;
                let _ = stdin.shutdown().await;
            }
        }
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let mut wait_task = tokio::spawn(async move { child.wait().await });

        let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<(OutputStream, Vec<u8>)>(1024);
        fn spawn_reader(
            mut pipe: impl tokio::io::AsyncRead + Unpin + Send + 'static,
            stream: OutputStream,
            tx: tokio::sync::mpsc::Sender<(OutputStream, Vec<u8>)>,
        ) -> tokio::task::JoinHandle<()> {
            tokio::spawn(async move {
                let mut buffer = vec![0u8; TEXT_CHUNK];
                loop {
                    match pipe.read(&mut buffer).await {
                        Ok(0) | Err(_) => break,
                        Ok(read) => {
                            if tx.send((stream, buffer[..read].to_vec())).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            })
        }
        let mut reader_tasks = Vec::new();
        if let Some(pipe) = stdout {
            reader_tasks.push(spawn_reader(pipe, OutputStream::Stdout, chunk_tx.clone()));
        }
        if let Some(pipe) = stderr {
            reader_tasks.push(spawn_reader(pipe, OutputStream::Stderr, chunk_tx.clone()));
        }
        drop(chunk_tx);

        // No timeout: cap the deadline far from `Instant` overflow while effectively unbounded.
        let deadline = tokio::time::Instant::now()
            + Duration::from_millis(timeout_ms.min(30 * 24 * 60 * 60 * 1000));
        let mut exit_status: Option<std::process::ExitStatus> = None;
        let mut killed = false;

        // Read chunks as they arrive; after exit, settle once both streams end, or once no
        // chunk arrived within the post-exit stdio grace (extended while a spill write is
        // pending) — a descendant holding stdio open must not hang `exec`.
        loop {
            if exit_status.is_some() {
                let quiet = tokio::time::sleep(Duration::from_millis(EXIT_STDIO_GRACE_MS));
                tokio::select! {
                    maybe = chunk_rx.recv() => {
                        match maybe {
                            Some((stream, chunk)) => {
                                self.exec_feed(&shared, stream, chunk, context).await;
                                continue;
                            }
                            None => {
                                if exit_status.is_none() {
                                    if let Ok(result) = (&mut wait_task).await {
                                        exit_status = result.ok();
                                    }
                                }
                                break;
                            }
                        }
                    }
                    () = quiet => {
                        if shared.spill_pending_writes.load(Ordering::SeqCst) > 0 {
                            // A pending spill write extends the grace (upstream `spillIsDraining`).
                            continue;
                        }
                        break;
                    }
                }
            }
            tokio::select! {
                maybe = chunk_rx.recv() => {
                    match maybe {
                        Some((stream, chunk)) => self.exec_feed(&shared, stream, chunk, context).await,
                        None => {
                            // Both streams ended; the exit status may not be reaped yet, and a
                            // successful exit must not surface as `unwrap_or(1)`'s fallback.
                            if exit_status.is_none() {
                                if let Ok(result) = (&mut wait_task).await {
                                    exit_status = result.ok();
                                }
                            }
                            break;
                        }
                    }
                }
                () = tokio::time::sleep_until(deadline), if exit_status.is_none() && !killed => {
                    shared.timed_out.store(true, Ordering::SeqCst);
                    killed = true;
                    kill_pid(pid);
                }
                () = async {
                    match &signal {
                        Some(signal) => {
                            let _ = signal.aborted_future().await;
                        }
                        None => std::future::pending::<()>().await,
                    }
                }, if exit_status.is_none() && !killed => {
                    killed = true;
                    kill_pid(pid);
                }
                result = async { (&mut wait_task).await }, if exit_status.is_none() => {
                    exit_status = result.ok().and_then(|status| status.ok());
                    if killed {
                        // Already killed through timeout/abort; keep draining output in the grace loop.
                        continue;
                    }
                }
            }
        }
        if exit_status.is_none() {
            // The wait task may still be running; give the killed process a moment to be reaped.
            let _ = tokio::time::timeout(Duration::from_secs(1), async { (&mut wait_task).await })
                .await;
        }
        for task in reader_tasks {
            task.abort();
        }

        Self::exec_finish_spill(&shared).await;
        // The decoders' held-back tails flush after the streams end (upstream `decode()`).
        let stdout_tail = shared.stdout_decoder.lock().await.decode(None);
        shared.emit(&stdout_tail, OutputStream::Stdout);
        let stderr_tail = shared.stderr_decoder.lock().await.decode(None);
        shared.emit(&stderr_tail, OutputStream::Stderr);

        if let Some(callback_error) = lock(&shared.callback_error).clone() {
            if let Some(pid) = pid {
                lock(&self.active_child_pids).remove(&pid);
            }
            return Err(callback_error);
        }
        let interrupted = if shared.timed_out.load(Ordering::SeqCst) {
            Some(ExecutionError::new(
                ExecutionErrorCode::Timeout,
                format!(
                    "timeout:{}",
                    options
                        .timeout
                        .map_or_else(|| "undefined".to_string(), |timeout| format!("{timeout}"))
                ),
            ))
        } else if signal.as_ref().is_some_and(|signal| signal.aborted()) {
            Some(ExecutionError::new(ExecutionErrorCode::Aborted, "aborted"))
        } else {
            None
        };
        if let Some(mut interrupted) = interrupted {
            if let Some(path) = shared.spill_state.lock().await.path.clone() {
                interrupted.set_spill_path(path);
            }
            if let Some(pid) = pid {
                lock(&self.active_child_pids).remove(&pid);
            }
            return Err(interrupted);
        }
        if let Some(spill_error) = lock(&shared.spill_error).clone() {
            if let Some(pid) = pid {
                lock(&self.active_child_pids).remove(&pid);
            }
            return Err(spill_error);
        }
        // A process killed by a signal (e.g. OOM killer) has no exit code; map it to the
        // conventional 128 + signal number so callers do not mistake it for success.
        let exit_code = exit_status
            .and_then(|status| {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt as _;
                    status
                        .code()
                        .or_else(|| status.signal().map(|signal_number| 128 + signal_number))
                }
                #[cfg(not(unix))]
                {
                    status.code()
                }
            })
            .unwrap_or(1);
        let spill_path = shared.spill_state.lock().await.path.clone();
        if let Some(pid) = pid {
            lock(&self.active_child_pids).remove(&pid);
        }
        Ok(ShellExecResult {
            exit_code,
            spill_path,
        })
    }
}
