//! Port of `packages/durable/src/env/index.ts` (plus re-exports of `decode.ts`/`line-scan.ts`).
//!
//! Expected failures are returned as `Result` values, never thrown: upstream's `Result<T, E>`
//! type collapses into [`std::result::Result`] with a `FileError` default, and `ok`/`err`
//! become `Ok`/`Err` (kept for call-site parity). `getOrThrow` panics — Rust callers should
//! propagate instead of throwing across the boundary.
//!
//! The [`FileSystem`] and [`Shell`] capabilities are object-safe async traits; [`ExecutionEnv`]
//! is their upstream composition, usable as `Arc<dyn ExecutionEnv>`.
//!
//! Deviations:
//! - D1: `getOrUndefined(result)` collapses into Rust's own `Result::ok()`; not re-exported.
//! - D2: `toError(unknown)` (normalizing thrown non-Error values) has no Rust counterpart —
//!   all fallible operations return typed errors already; omitted.
//! - D3: upstream `Context`/`AbortSignal` values from `@earendil-works/chord` map to
//!   `pi_chord::context::Context` passed by reference; `context.abortSignal` checks become
//!   `context.abort_signal()`.
//! - D4: `WatchChange` is a Rust enum (`Paths`/`Overflow`/`Error`) with serde shapes matching
//!   the upstream union members (`{paths: [...]}` / `{overflow: true}` / `{error: FileError}`).

pub mod decode;
pub mod line_scan;
pub mod node;
pub mod node_watch;

#[cfg(test)]
mod line_scan_tests;
#[cfg(test)]
mod node_conformance_tests;
#[cfg(test)]
mod node_shell_tests;
#[cfg(test)]
mod node_spill_tests;
#[cfg(test)]
mod node_tests;

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use pi_chord::context::Context;

pub use decode::{range_decoder, starts_with_bom, StreamDecoder, TextDecoder};
pub use line_scan::{LineScan, LineScanner, RangeError};
pub use node::NodeExecutionEnv;
pub use node_watch::{NodeFileWatcher, NodeWatchOptions};

/// A shared, cloneable error cause (upstream attaches the thrown error as `cause`).
pub(crate) type SharedError = Arc<dyn std::error::Error + Send + Sync>;

/// Wraps an I/O error as a shared cause. Upstream keeps the thrown error as the new error's
/// `cause`; abort-like failures (an operation cancelled through the context) surface as their
/// own `FileError`s before I/O runs, so an I/O error here is always a genuine OS failure.
pub(crate) fn abort_error_like(error: &std::io::Error) -> SharedError {
    // `std::io::Error` is not `Clone`; recreate it from the OS error (or the raw payload).
    if let Some(code) = error.raw_os_error() {
        return Arc::new(std::io::Error::from_raw_os_error(code));
    }
    Arc::new(std::io::Error::other(error.to_string()))
}

/// Upstream `toFileError` (node.ts): maps an I/O error onto the portable codes, keeping the
/// OS message and the offending path.
pub(crate) fn to_file_error(error: &std::io::Error, fallback_path: Option<&str>) -> FileError {
    let code = match error.kind() {
        std::io::ErrorKind::NotFound => FileErrorCode::NotFound,
        std::io::ErrorKind::PermissionDenied => FileErrorCode::PermissionDenied,
        std::io::ErrorKind::NotADirectory => FileErrorCode::NotDirectory,
        std::io::ErrorKind::IsADirectory => FileErrorCode::IsDirectory,
        std::io::ErrorKind::InvalidInput => FileErrorCode::Invalid,
        _ => FileErrorCode::Unknown,
    };
    FileError::new(code, error.to_string(), fallback_path.map(str::to_string))
        .with_cause(abort_error_like(error))
}

/// Upstream `NodeExecutionEnvOptions`.
#[derive(Debug, Clone, Default)]
pub struct NodeEnvOptions {
    /// Working directory; defaults to the process's current directory.
    pub cwd: String,
    /// Explicit shell executable, overriding platform defaults.
    pub shell_path: Option<String>,
    /// Replaces the inherited environment for commands when set.
    pub shell_env: Option<HashMap<String, String>>,
    /// Watch configuration (see [`NodeWatchOptions`]).
    pub watch: NodeWatchOptions,
}

impl NodeEnvOptions {
    pub fn new() -> Self {
        Self {
            cwd: std::env::current_dir()
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_else(|_| ".".to_string()),
            ..Default::default()
        }
    }
}

/// Result of a fallible operation. Expected failures are returned instead of thrown.
pub type Result<TValue, TError = FileError> = std::result::Result<TValue, TError>;

/// Upstream `getOrThrow`: unwrap, panicking with the error on failure. Rust callers should
/// prefer `?`/`match`; this exists for test parity with upstream call sites.
pub fn get_or_throw<TValue, TError>(result: Result<TValue, TError>) -> TValue
where
    TError: fmt::Debug,
{
    result.expect("get_or_throw on an error result")
}

/// Upstream `ok`: the `Ok` constructor of [`Result`].
pub fn ok<TValue>(value: TValue) -> Result<TValue> {
    Ok(value)
}

/// Upstream `err`: the `Err` constructor of [`Result`].
pub fn err<TValue>(error: FileError) -> Result<TValue, FileError> {
    Err(error)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileKind {
    File,
    Directory,
    Symlink,
}

impl fmt::Display for FileKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            FileKind::File => "file",
            FileKind::Directory => "directory",
            FileKind::Symlink => "symlink",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileErrorCode {
    Aborted,
    NotFound,
    PermissionDenied,
    NotDirectory,
    IsDirectory,
    Invalid,
    NotSupported,
    Unknown,
}

impl FileErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            FileErrorCode::Aborted => "aborted",
            FileErrorCode::NotFound => "not_found",
            FileErrorCode::PermissionDenied => "permission_denied",
            FileErrorCode::NotDirectory => "not_directory",
            FileErrorCode::IsDirectory => "is_directory",
            FileErrorCode::Invalid => "invalid",
            FileErrorCode::NotSupported => "not_supported",
            FileErrorCode::Unknown => "unknown",
        }
    }
}

impl fmt::Display for FileErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Upstream `FileError` (an `Error` subclass): `message` is the [`Display`] text, `path` the
/// offending path when known, `cause` the wrapped underlying error.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct FileError {
    pub code: FileErrorCode,
    pub message: String,
    pub path: Option<String>,
    #[source]
    pub cause: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

impl FileError {
    pub fn new(code: FileErrorCode, message: impl Into<String>, path: Option<String>) -> Self {
        Self {
            code,
            message: message.into(),
            path,
            cause: None,
        }
    }

    pub fn with_cause(mut self, cause: Arc<dyn std::error::Error + Send + Sync>) -> Self {
        self.cause = Some(cause);
        self
    }

    pub fn aborted(path: Option<String>) -> Self {
        Self::new(FileErrorCode::Aborted, "aborted", path)
    }

    /// The upstream JSON shape of a serialized `FileError` (code/message/path).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "code": self.code.as_str(),
            "message": self.message,
            "path": self.path,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionErrorCode {
    Aborted,
    Timeout,
    ShellUnavailable,
    SpawnError,
    CallbackError,
    Unknown,
}

impl ExecutionErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExecutionErrorCode::Aborted => "aborted",
            ExecutionErrorCode::Timeout => "timeout",
            ExecutionErrorCode::ShellUnavailable => "shell_unavailable",
            ExecutionErrorCode::SpawnError => "spawn_error",
            ExecutionErrorCode::CallbackError => "callback_error",
            ExecutionErrorCode::Unknown => "unknown",
        }
    }
}

impl fmt::Display for ExecutionErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Upstream `ExecutionError` (an `Error` subclass).
#[derive(Debug, Clone)]
pub struct ExecutionError {
    pub code: ExecutionErrorCode,
    pub message: String,
    pub cause: Option<Arc<dyn std::error::Error + Send + Sync>>,
    /// Spill file of a command that timed out or was aborted after its output crossed the spill thresholds.
    pub spill_path: Option<String>,
}

impl ExecutionError {
    pub fn new(code: ExecutionErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            cause: None,
            spill_path: None,
        }
    }

    pub fn with_cause(mut self, cause: Arc<dyn std::error::Error + Send + Sync>) -> Self {
        self.cause = Some(cause);
        self
    }

    pub fn set_spill_path(&mut self, path: String) {
        self.spill_path = Some(path);
    }
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause
            .as_ref()
            .map(|cause| cause.as_ref() as &(dyn std::error::Error + 'static))
    }
}

/// Upstream `FileInfo`. `mtime_ms` keeps the upstream `mtimeMs` sub-millisecond precision.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileInfo {
    pub name: String,
    pub path: String,
    pub kind: FileKind,
    pub size: u64,
    pub mtime_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextLine {
    pub text: String,
    pub terminated: bool,
}

/// Upstream `TextLineReader` (strict LF reader).
#[async_trait]
pub trait TextLineReader: Send + Sync {
    async fn read_line(&self, context: &Context) -> Result<Option<TextLine>, FileError>;
    async fn close(&self, context: &Context);
}

/// Positional reads from one opened regular file; all calls see the same file even if its path is renamed.
#[async_trait]
pub trait BinaryReader: Send + Sync {
    /// Metadata of the opened file, not of whatever its path names now.
    async fn info(&self, context: &Context) -> Result<FileInfo, FileError>;
    /// Up to `length` bytes at `offset`; fewer only at end of file.
    async fn read(
        &self,
        offset: u64,
        length: usize,
        context: &Context,
    ) -> Result<Vec<u8>, FileError>;
    /// One pass over the file that locates lines `[startLine, endLine)` (`endLine` absent: to the
    /// end), 0-based, where line `k` starts after the `k`-th newline byte.
    async fn scan_lines(
        &self,
        options: ScanLinesOptions,
        context: &Context,
    ) -> Result<LineScan, FileError>;
    async fn close(&self, context: &Context);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ScanLinesOptions {
    pub start_line: u64,
    /// `None`: to the end (upstream `Number.POSITIVE_INFINITY`).
    pub end_line: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OpenBinaryOptions {
    /// Refuse a symbolic link as the final path component with `invalid`; earlier components still resolve.
    pub no_follow: bool,
}

/// A file or directory to watch. It may be missing; creating it is a change.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WatchTarget {
    pub path: String,
    /// Watch everything below a directory, not only its entries. Symbolic links below it are not followed.
    pub recursive: bool,
    /// Entries below `path` that are neither watched nor reported: names starting with `.`, or these names.
    pub exclude: Option<WatchExclude>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WatchExclude {
    pub hidden: bool,
    pub names: Vec<String>,
}

/// What changed: something at or below each path may have changed (a directory path covers its
/// whole subtree). Calls may be spurious; a change is never missed while the watcher is healthy.
#[derive(Debug, Clone)]
pub enum WatchChange {
    Paths(Vec<String>),
    /// Coverage was uncertain for a while (lost events, reconnect); rescan everything that is watched.
    Overflow,
    /// The watcher stopped, for example because the watched tree grew past the environment's
    /// limit; no calls follow.
    Error(FileError),
}

impl WatchChange {
    /// The wire shape of the upstream union member (`{paths}`, `{overflow}`, `{error}`).
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            WatchChange::Paths(paths) => serde_json::json!({ "paths": paths }),
            WatchChange::Overflow => serde_json::json!({ "overflow": true }),
            WatchChange::Error(error) => serde_json::json!({ "error": error.to_json() }),
        }
    }
}

/// Reliability of a [`FileWatcher`] (upstream `FileWatcher.mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchMode {
    /// Changes are reported within about two seconds.
    Native,
    /// The environment compares snapshots, because the file system does not report changes
    /// reliably (network and FUSE file systems); a change undone between two snapshots can be missed.
    Polling,
}

/// Callback receiving what changed.
pub type WatchCallback = Arc<dyn Fn(WatchChange) + Send + Sync>;

#[async_trait]
pub trait FileWatcher: Send + Sync {
    fn mode(&self) -> WatchMode;
    /// Stop watching; no `onChange` call starts after this resolves. Idempotent.
    async fn close(&self, context: &Context);
}

/// Pages of one directory's entries.
#[async_trait]
pub trait DirReader: Send + Sync {
    /// Up to `max_entries` entries in the order the file system returns them, continuing where
    /// the previous call stopped. `done` marks the end; it may come with the last entries or
    /// with an empty page. An entry that disappears before its metadata is read is skipped, as
    /// are entries of unsupported kinds. After a failed or aborted call, close the reader.
    async fn next(&self, max_entries: usize, context: &Context) -> Result<DirPage, FileError>;
    async fn close(&self, context: &Context);
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DirPage {
    pub entries: Vec<FileInfo>,
    pub done: bool,
}

/// Portable filesystem capability. Operations return failures rather than throwing.
#[async_trait]
pub trait FileSystem: Send + Sync {
    /// The file namespace: equal ids see the same files at the same paths, whatever their `cwd`.
    fn id(&self) -> &str;
    fn cwd(&self) -> &str;
    async fn absolute_path(&self, path: &str, context: &Context) -> Result<String, FileError>;
    async fn join_path(&self, parts: &[String], context: &Context) -> Result<String, FileError>;
    async fn read_text_file(&self, path: &str, context: &Context) -> Result<String, FileError>;
    async fn open_text_line_reader(
        &self,
        path: &str,
        context: &Context,
    ) -> Result<Arc<dyn TextLineReader>, FileError>;
    async fn read_text_lines(
        &self,
        path: &str,
        options: Option<ReadTextLinesOptions>,
        context: &Context,
    ) -> Result<Vec<String>, FileError>;
    async fn read_binary_file(&self, path: &str, context: &Context) -> Result<Vec<u8>, FileError>;
    /// Open a regular file for bounded positional reads. A directory fails with `is_directory`,
    /// other non-regular files with `invalid`. With `noFollow`, a symbolic link as the final
    /// path component fails with `invalid` instead of being followed; earlier components are
    /// still resolved.
    async fn open_binary_reader(
        &self,
        path: &str,
        options: Option<OpenBinaryOptions>,
        context: &Context,
    ) -> Result<Arc<dyn BinaryReader>, FileError>;
    async fn write_file(
        &self,
        path: &str,
        content: FileContent,
        context: &Context,
    ) -> Result<(), FileError>;
    async fn append_file(
        &self,
        path: &str,
        content: FileContent,
        context: &Context,
    ) -> Result<(), FileError>;
    /// Truncate or extend a file to exactly `size` bytes.
    async fn truncate_file(
        &self,
        path: &str,
        size: u64,
        context: &Context,
    ) -> Result<(), FileError>;
    /// Flush file contents and metadata needed to retrieve them from an open file handle.
    async fn flush_file(&self, path: &str, context: &Context) -> Result<(), FileError>;
    async fn rename_file(
        &self,
        source_path: &str,
        destination_path: &str,
        context: &Context,
    ) -> Result<(), FileError>;
    async fn file_info(&self, path: &str, context: &Context) -> Result<FileInfo, FileError>;
    async fn list_dir(&self, path: &str, context: &Context) -> Result<Vec<FileInfo>, FileError>;
    async fn open_dir_reader(
        &self,
        path: &str,
        context: &Context,
    ) -> Result<Arc<dyn DirReader>, FileError>;
    /// Report changes to files and directories, for hosts that load resources from the
    /// environment. When the returned watcher exists, coverage is established: a host that
    /// watches before it loads cannot miss a change made during the load. See [`WatchChange`]
    /// for what is reported and [`FileWatcher::mode`] for how reliably.
    async fn watch(
        &self,
        targets: &[WatchTarget],
        on_change: WatchCallback,
        context: &Context,
    ) -> Result<Arc<dyn FileWatcher>, FileError>;
    async fn canonical_path(&self, path: &str, context: &Context) -> Result<String, FileError>;
    async fn exists(&self, path: &str, context: &Context) -> Result<bool, FileError>;
    async fn create_dir(
        &self,
        path: &str,
        options: Option<CreateDirOptions>,
        context: &Context,
    ) -> Result<(), FileError>;
    async fn remove(
        &self,
        path: &str,
        options: Option<RemoveOptions>,
        context: &Context,
    ) -> Result<(), FileError>;
    async fn create_temp_dir(
        &self,
        prefix: Option<&str>,
        context: &Context,
    ) -> Result<String, FileError>;
    async fn create_temp_file(
        &self,
        options: Option<TempFileOptions>,
        context: &Context,
    ) -> Result<String, FileError>;
    async fn cleanup(&self, context: &Context);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReadTextLinesOptions {
    /// `None`: no limit.
    pub max_lines: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileContent {
    Text(String),
    Binary(Vec<u8>),
}

impl From<String> for FileContent {
    fn from(value: String) -> Self {
        FileContent::Text(value)
    }
}

impl From<&str> for FileContent {
    fn from(value: &str) -> Self {
        FileContent::Text(value.to_string())
    }
}

impl From<Vec<u8>> for FileContent {
    fn from(value: Vec<u8>) -> Self {
        FileContent::Binary(value)
    }
}

impl From<&[u8]> for FileContent {
    fn from(value: &[u8]) -> Self {
        FileContent::Binary(value.to_vec())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CreateDirOptions {
    /// Upstream default: `true`.
    pub recursive: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RemoveOptions {
    /// Upstream default: `false`.
    pub recursive: Option<bool>,
    /// Upstream default: `false`.
    pub force: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TempFileOptions {
    pub prefix: Option<String>,
    pub suffix: Option<String>,
}

/// Spill the complete output to a temporary file once it exceeds either threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShellSpillOptions {
    pub after_bytes: u64,
    /// Complete or partial lines.
    pub after_lines: u64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShellExecResult {
    pub exit_code: i32,
    /// Temporary file holding the complete raw output, when the spill thresholds were exceeded.
    pub spill_path: Option<String>,
}

#[derive(Clone, Default)]
pub struct ShellExecOptions {
    pub cwd: Option<String>,
    pub env: Option<HashMap<String, String>>,
    /// Upstream default: `true`.
    pub inherit_env: Option<bool>,
    /// Seconds; fractional timeouts are honored.
    pub timeout: Option<f64>,
    /// Every decoded chunk of stdout and stderr as it arrives, in arrival order, with the stream
    /// it came from: raw, unbounded, and unthrottled. Each stream is decoded separately, so a
    /// character split across chunks survives. A panicking callback becomes a `callback_error`
    /// (upstream: a throwing one).
    pub on_output: Option<Arc<dyn Fn(&str, &ShellOutputInfo) + Send + Sync>>,
    pub spill: Option<ShellSpillOptions>,
    /// The caller keeps only this tail of the output, so the environment may omit output outside
    /// it and report the omission as `info.skipped`. Without it, every chunk is delivered. The
    /// local `NodeExecutionEnv` delivers every chunk; hosts apply windows themselves (upstream
    /// parity — see `ShellOutputWindow`).
    pub window: Option<ShellOutputWindow>,
}

/// The tail of the combined output a caller keeps, and how often it samples it. An environment
/// that transfers output over a slow link uses it to omit what the caller would drop anyway and
/// to send no faster than the caller commits.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShellOutputWindow {
    /// UTF-8 bytes of decoded text kept at the end of the output.
    pub max_bytes: usize,
    /// Lines kept at the end of the output.
    pub max_lines: usize,
    /// Minimum pause between the caller's samples of the output.
    pub min_interval_ms: u64,
    /// Each sample also pauses the caller in proportion to its size at this rate.
    pub bytes_per_second: f64,
}

/// Output an environment omitted, measured on the decoded text `onOutput` would have received:
/// every U+FFFD counts as three bytes, and no sanitizing is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShellOutputSkip {
    /// UTF-8 byte length of the omitted text.
    pub bytes: u64,
    /// Newlines (U+000A) in the omitted text.
    pub newlines: u64,
    /// Whether the omitted text ends with a newline.
    pub ends_with_newline: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputStream {
    #[default]
    Stdout,
    Stderr,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShellOutputInfo {
    pub stream: OutputStream,
    /// Output omitted immediately before this chunk, only with `window`. The chunk then holds
    /// all output after the omission up to its end, and that is more than the window by at least
    /// one byte or one line. The omission and such a chunk may span both streams in arrival
    /// order; `stream` then names the chunk's last stream. Callers that need the streams apart
    /// do not pass `window`.
    pub skipped: Option<ShellOutputSkip>,
}

impl ShellOutputInfo {
    /// Upstream `{ stream }` info without an omission.
    pub fn for_stream(stream: OutputStream) -> Self {
        Self {
            stream,
            skipped: None,
        }
    }
}

/// Run a command. A string runs through the environment's shell. An array runs `command[0]`
/// directly with the rest as its arguments, without a shell, so they reach the program
/// unparsed. Aborting the context or a timeout kills only this command's processes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecCommand {
    Shell(String),
    Argv(Vec<String>),
}

impl From<String> for ExecCommand {
    fn from(value: String) -> Self {
        ExecCommand::Shell(value)
    }
}

impl From<&str> for ExecCommand {
    fn from(value: &str) -> Self {
        ExecCommand::Shell(value.to_string())
    }
}

impl From<Vec<String>> for ExecCommand {
    fn from(value: Vec<String>) -> Self {
        ExecCommand::Argv(value)
    }
}

impl From<&[&str]> for ExecCommand {
    fn from(value: &[&str]) -> Self {
        ExecCommand::Argv(value.iter().map(|part| part.to_string()).collect())
    }
}

#[async_trait]
pub trait Shell: Send + Sync {
    async fn exec(
        &self,
        command: ExecCommand,
        options: Option<ShellExecOptions>,
        context: &Context,
    ) -> std::result::Result<ShellExecResult, ExecutionError>;
    /// Kill every command this environment still runs; for its owner's shutdown, never for one request.
    async fn cleanup(&self, context: &Context);
}

/// Upstream `interface ExecutionEnv extends FileSystem, Shell {}`.
pub trait ExecutionEnv: FileSystem + Shell {}

impl<T> ExecutionEnv for T where T: FileSystem + Shell {}
