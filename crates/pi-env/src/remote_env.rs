//! An `ExecutionEnv` on another machine, reached through a pi-env daemon (upstream
//! `packages/env/src/remote-env.ts`). Results match `NodeExecutionEnv` running on that machine: the
//! daemon performs the system calls, and this module applies Node's path, error, and result rules of
//! the remote system (see [`crate::paths`]).

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use pi_chord::context::{AbortController, AbortSignal, Context};
use pi_durable::env::{
    BinaryReader, CreateDirOptions, DirPage, DirReader, ExecCommand, ExecutionError,
    ExecutionErrorCode, FileContent, FileError, FileErrorCode, FileInfo, FileKind, FileSystem,
    FileWatcher, LineScan, OpenBinaryOptions, ReadTextLinesOptions, RemoveOptions,
    ScanLinesOptions, Shell, ShellExecOptions, ShellExecResult, ShellOutputInfo, ShellOutputSkip,
    StreamDecoder, TempFileOptions, TextLine, TextLineReader, WatchCallback, WatchChange,
    WatchTarget,
};
use serde_json::{Map, Value, json};

use crate::{
    connection::{Connection, Json, RemoteError, Reply, RequestOptions},
    errors::{abort_error, aborted, to_file_error},
    paths::{Paths, file_url_to_path},
    remote_watch::RemoteWatcher,
};

const MAX_TIMEOUT_MS: f64 = 2_147_483_647.0;
const MAX_TIMEOUT_SECONDS: f64 = MAX_TIMEOUT_MS / 1000.0;
/// Bytes per `pread` request of a binary reader; several are in flight at once.
const READ_CHUNK: usize = 256 * 1024;
const READ_DEPTH: usize = 8;
/// Node's `writeFile` writes at most this much per call and checks for an abort before each.
const WRITE_CHUNK: usize = 512 * 1024;
const WRITE_DEPTH: usize = 8;
/// Node's `readFile`: chunk of known-size reads, chunk of unknown-size reads, and the largest file it reads.
const READ_FILE_CHUNK: usize = 512 * 1024;
const READ_FILE_UNKNOWN_CHUNK: usize = 64 * 1024;
const READ_FILE_MAX: u64 = 2_u64.pow(31) - 1;
/// `NodeTextLineReader`'s read size.
const LINE_CHUNK: usize = 64 * 1024;

#[derive(Clone, Default)]
pub struct RemoteExecutionEnvOptions {
    pub connection: Option<Connection>,
    /// The file namespace: equal ids see the same files (`FileSystem.id`), e.g. `pi-env:<env name>`.
    pub id: String,
    pub cwd: String,
    pub shell_path: Option<String>,
    /// Added to the remote environment of every command that inherits it, like `shellEnv`.
    pub shell_env: Option<HashMap<String, String>>,
    /// How the daemon watches, like `NodeExecutionEnv`'s `watch` option.
    pub watch: crate::remote_watch::RemoteWatchOptions,
}

/// An info record from the daemon.
type RemoteFileInfo = Json;
type RemoteDirEntry = Json;

/// Node's path rules of the remote system.
pub async fn remote_path(connection: &Connection) -> Result<Paths, RemoteError> {
    Ok(Paths::for_os(&connection.info().await?.os))
}

fn to_info(path: &str, remote: &RemoteFileInfo, paths: Paths) -> Result<FileInfo, FileError> {
    if remote.get("kind").and_then(Value::as_str) == Some("other") {
        return Err(FileError::new(
            FileErrorCode::Invalid,
            "Unsupported file type",
            Some(path.to_string()),
        ));
    }
    let kind = match remote.get("kind").and_then(Value::as_str) {
        Some("file") => FileKind::File,
        Some("directory") => FileKind::Directory,
        Some("symlink") => FileKind::Symlink,
        _ => {
            return Err(FileError::new(
                FileErrorCode::Invalid,
                "Unsupported file type",
                Some(path.to_string()),
            ));
        }
    };
    Ok(FileInfo {
        name: paths.basename(path),
        path: path.to_string(),
        kind,
        size: remote.get("size").and_then(Value::as_u64).unwrap_or(0),
        mtime_ms: remote
            .get("mtimeSec")
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            * 1000.0
            + remote
                .get("mtimeNsec")
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
                / 1e6,
    })
}

/// A daemon handle: valid only in the session that opened it.
#[derive(Clone)]
struct Handle {
    env: RemoteExecutionEnv,
    id: u64,
    session: u64,
    path: String,
}

impl Handle {
    async fn request(
        &self,
        op: &str,
        fields: Json,
        options: RequestOptions,
    ) -> Result<Reply, RemoteError> {
        let mut fields = fields;
        fields.insert("handle".into(), json!(self.id));
        let mut options = options;
        options.session = Some(self.session);
        self.env.connection.request(op, fields, options).await
    }

    async fn pread(&self, offset: Option<u64>, length: usize) -> Result<Vec<u8>, RemoteError> {
        let mut fields = Map::new();
        if let Some(offset) = offset {
            fields.insert("offset".into(), json!(offset));
        }
        fields.insert("length".into(), json!(length));
        Ok(self
            .request("pread", fields, RequestOptions::default())
            .await?
            .payload)
    }

    async fn close(&self) {
        let _ = self
            .request("close", Map::new(), RequestOptions::default())
            .await;
    }
}

/// Read `length` bytes from `offset` as consecutive reads of at most `chunk` bytes, several in flight.
/// Ends early at the end of the file. A short read continues from where it ended, as sequential reads
/// would. Returns `None` when aborted.
async fn read_range<R, Fut>(
    read: R,
    offset: u64,
    length: u64,
    chunk: usize,
    aborted: impl Fn() -> bool + Send,
) -> Result<Option<(Vec<Vec<u8>>, usize)>, RemoteError>
where
    R: Fn(u64, usize) -> Fut + Send,
    Fut: Future<Output = Result<Vec<u8>, RemoteError>> + Send,
{
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut total = 0usize;
    let mut next = offset;
    let mut in_flight: std::collections::VecDeque<(usize, Fut)> = Default::default();
    while total < length as usize {
        while in_flight.len() < READ_DEPTH && next < offset + length {
            let size = ((offset + length - next) as usize).min(chunk);
            in_flight.push_back((size, read(next, size)));
            next += size as u64;
        }
        let (size, bytes) = in_flight.pop_front().unwrap();
        let bytes = bytes.await?;
        if aborted() {
            return Ok(None);
        }
        if bytes.is_empty() {
            break;
        }
        total += bytes.len();
        chunks.push(bytes);
        if total < length as usize && chunks.last().is_some_and(|last| last.len() < size) {
            // Reads already in flight assumed a full read; continue right after this one.
            in_flight.clear();
            next = offset + total as u64;
        }
    }
    Ok(Some((chunks, total)))
}

fn concat(chunks: Vec<Vec<u8>>, total: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(total);
    for chunk in chunks {
        bytes.extend_from_slice(&chunk);
    }
    bytes
}

fn early(context: &Context, path: Option<&str>, closed: bool, what: &str) -> Option<FileError> {
    if aborted(context.abort_signal().as_ref()) {
        return Some(abort_error(path));
    }
    closed.then(|| {
        FileError::new(
            FileErrorCode::Invalid,
            format!("{what} is closed"),
            path.map(str::to_owned),
        )
    })
}

struct RemoteBinaryReader {
    handle: Handle,
    closed: parking_lot::Mutex<bool>,
}

#[async_trait]
impl BinaryReader for RemoteBinaryReader {
    async fn info(&self, context: &Context) -> Result<FileInfo, FileError> {
        if let Some(error) = early(
            context,
            Some(&self.handle.path),
            *self.closed.lock(),
            "Binary reader",
        ) {
            return Err(error);
        }
        let paths = remote_path(&self.handle.env.connection)
            .await
            .map_err(|error| to_file_error(&error, Some(&self.handle.path)))?;
        let reply = self
            .handle
            .request("fstat", Map::new(), RequestOptions::default())
            .await
            .map_err(|error| to_file_error(&error, Some(&self.handle.path)))?;
        to_info(&self.handle.path, &reply.json, paths)
    }

    async fn read(
        &self,
        offset: u64,
        length: usize,
        context: &Context,
    ) -> Result<Vec<u8>, FileError> {
        if let Some(error) = early(
            context,
            Some(&self.handle.path),
            *self.closed.lock(),
            "Binary reader",
        ) {
            return Err(error);
        }
        let signal = context.abort_signal();
        let read = read_range(
            |at, size| self.handle.pread(Some(at), size),
            offset,
            length as u64,
            READ_CHUNK,
            || signal.as_ref().is_some_and(AbortSignal::aborted),
        )
        .await;
        match read {
            Ok(Some((chunks, total))) => Ok(concat(chunks, total)),
            Ok(None) => Err(FileError::aborted(Some(self.handle.path.clone()))),
            Err(error) => Err(to_file_error(&error, Some(&self.handle.path))),
        }
    }

    async fn scan_lines(
        &self,
        options: ScanLinesOptions,
        context: &Context,
    ) -> Result<LineScan, FileError> {
        if let Some(error) = early(
            context,
            Some(&self.handle.path),
            *self.closed.lock(),
            "Binary reader",
        ) {
            return Err(error);
        }
        let mut fields = Map::new();
        fields.insert("startLine".into(), json!(options.start_line));
        if let Some(end_line) = options.end_line {
            fields.insert("endLine".into(), json!(end_line));
        }
        let mut request = RequestOptions::default();
        request.signal = context.abort_signal();
        let reply = self
            .handle
            .request("scanLines", fields, request)
            .await
            .map_err(|error| to_file_error(&error, Some(&self.handle.path)))?;
        serde_json::from_value(Value::Object(reply.json)).map_err(|_| {
            FileError::new(
                FileErrorCode::Invalid,
                "Invalid line scan",
                Some(self.handle.path.clone()),
            )
        })
    }

    async fn close(&self, _context: &Context) {
        if *self.closed.lock() {
            return;
        }
        *self.closed.lock() = true;
        self.handle.close().await;
    }
}

struct RemoteDirReader {
    handle: Handle,
    done: parking_lot::Mutex<bool>,
    closed: parking_lot::Mutex<bool>,
}

#[async_trait]
impl DirReader for RemoteDirReader {
    async fn next(&self, max_entries: usize, context: &Context) -> Result<DirPage, FileError> {
        let path = self.handle.path.clone();
        if let Some(error) = early(
            context,
            Some(&path),
            *self.closed.lock(),
            "Directory reader",
        ) {
            return Err(error);
        }
        let paths = remote_path(&self.handle.env.connection)
            .await
            .map_err(|error| to_file_error(&error, Some(&path)))?;
        let mut entries = Vec::new();
        while !*self.done.lock() && entries.len() < max_entries {
            let mut fields = Map::new();
            fields.insert("max".into(), json!(max_entries - entries.len()));
            let reply = self
                .handle
                .request("readdir", fields, RequestOptions::default())
                .await
                .map_err(|error| to_file_error(&error, Some(&path)))?;
            let json = reply.json;
            if aborted(context.abort_signal().as_ref()) {
                return Err(FileError::aborted(Some(path)));
            }
            for entry in json
                .get("entries")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
            {
                let Some(entry) = entry.as_object() else {
                    continue;
                };
                let name = entry
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let entry_path = paths.resolve(&[&path, &name], "", &HashMap::new());
                if let Some(error) = entry.get("error") {
                    // Removed between enumeration and lstat: not part of the listing any more.
                    if error.get("code").and_then(Value::as_str) == Some("ENOENT") {
                        continue;
                    }
                    let error =
                        RemoteError::from_fields(error.as_object().cloned().unwrap_or_default());
                    return Err(to_file_error(&error, Some(&entry_path)));
                }
                if let Ok(info) = to_info(&entry_path, &entry, paths) {
                    entries.push(info);
                }
            }
            *self.done.lock() = json.get("done").and_then(Value::as_bool).unwrap_or(false);
        }
        Ok(DirPage {
            entries,
            done: *self.done.lock(),
        })
    }

    async fn close(&self, _context: &Context) {
        if *self.closed.lock() {
            return;
        }
        *self.closed.lock() = true;
        self.handle.close().await;
    }
}

/// `NodeTextLineReader`: positional reads of a file opened like `open(path, "r")`.
struct RemoteTextLineReader {
    handle: Handle,
    decoder: parking_lot::Mutex<StreamDecoder>,
    offset: parking_lot::Mutex<u64>,
    buffered: parking_lot::Mutex<String>,
    ended: parking_lot::Mutex<bool>,
    closed: parking_lot::Mutex<bool>,
}

#[async_trait]
impl TextLineReader for RemoteTextLineReader {
    async fn read_line(&self, context: &Context) -> Result<Option<TextLine>, FileError> {
        let path = self.handle.path.clone();
        if let Some(error) = early(
            context,
            Some(&path),
            *self.closed.lock(),
            "Text line reader",
        ) {
            return Err(error);
        }
        loop {
            {
                let mut buffered = self.buffered.lock();
                if let Some(newline) = buffered.find('\n') {
                    let text = buffered[..newline].to_string();
                    buffered.drain(..newline + 1);
                    return Ok(Some(TextLine {
                        text,
                        terminated: true,
                    }));
                }
                if *self.ended.lock() {
                    if buffered.is_empty() {
                        return Ok(None);
                    }
                    let text = std::mem::take(&mut *buffered);
                    return Ok(Some(TextLine {
                        text,
                        terminated: false,
                    }));
                }
            }
            let offset = *self.offset.lock();
            let bytes = self.handle.pread(Some(offset), LINE_CHUNK).await;
            let bytes = match bytes {
                Ok(bytes) => bytes,
                Err(error) => return Err(to_file_error(&error, Some(&path))),
            };
            if aborted(context.abort_signal().as_ref()) {
                return Err(FileError::aborted(Some(path)));
            }
            *self.offset.lock() += bytes.len() as u64;
            let mut buffered = self.buffered.lock();
            if bytes.is_empty() {
                buffered.push_str(&self.decoder.lock().decode(None));
                *self.ended.lock() = true;
            } else {
                buffered.push_str(&self.decoder.lock().decode(Some(&bytes)));
            }
        }
    }

    async fn close(&self, _context: &Context) {
        if *self.closed.lock() {
            return;
        }
        *self.closed.lock() = true;
        self.buffered.lock().clear();
        self.handle.close().await;
    }
}

/// An `ExecutionEnv` on another machine, reached through a pi-env daemon.
#[derive(Clone)]
pub struct RemoteExecutionEnv {
    id: String,
    cwd: String,
    pub connection: Connection,
    shell_path: Option<String>,
    shell_env: Option<HashMap<String, String>>,
    watch_options: crate::remote_watch::RemoteWatchOptions,
    /// Running commands this environment started (request id → session), for `cleanup()`.
    running: Arc<parking_lot::Mutex<HashMap<u32, u64>>>,
}

impl RemoteExecutionEnv {
    pub fn new(
        connection: Connection,
        id: impl Into<String>,
        cwd: impl Into<String>,
    ) -> RemoteExecutionEnv {
        RemoteExecutionEnv {
            id: id.into(),
            cwd: cwd.into(),
            connection,
            shell_path: None,
            shell_env: None,
            watch_options: Default::default(),
            running: Arc::new(parking_lot::Mutex::new(HashMap::new())),
        }
    }

    pub fn with_options(
        options: RemoteExecutionEnvOptions,
    ) -> Result<RemoteExecutionEnv, FileError> {
        Ok(RemoteExecutionEnv {
            connection: options.connection.ok_or_else(|| {
                FileError::new(FileErrorCode::Invalid, "connection is required", None)
            })?,
            id: options.id,
            cwd: options.cwd,
            shell_path: options.shell_path,
            shell_env: options.shell_env,
            watch_options: options.watch,
            running: Arc::new(parking_lot::Mutex::new(HashMap::new())),
        })
    }

    /// Node's `resolvePath` on the remote system, with its home directory and working directories.
    async fn resolve(&self, path: &str) -> Result<String, RemoteError> {
        let info = self.connection.info().await?;
        let windows = info.os == "windows";
        let paths = Paths::for_os(&info.os);
        let mut normalized = path.to_string();
        if normalized == "~" {
            normalized = info.home.clone();
        } else if normalized.starts_with("~/") || (windows && normalized.starts_with("~\\")) {
            normalized = paths.join(&[&info.home, &normalized[2..]]);
        } else if let Some(rest) = normalized.strip_prefix("file://") {
            if let Some(converted) = file_url_to_path(rest, windows) {
                // Malformed URLs stay ordinary paths, as Node does.
                normalized = converted;
            }
        }
        if paths.is_absolute(&normalized) {
            return Ok(paths.resolve(&[&normalized], &info.cwd, &info.drive_cwds));
        }
        if windows && normalized.len() >= 2 && normalized.as_bytes()[1] == b':' {
            let drive = &normalized[..2];
            let rooted = normalized
                .as_bytes()
                .get(2)
                .is_some_and(|byte| *byte == b'\\' || *byte == b'/');
            if !rooted && !drive.eq_ignore_ascii_case(&self.cwd[..self.cwd.len().min(2)]) {
                // A drive-relative path on another drive: Node resolves it against that drive's working
                // directory (`=D:`), else its own working directory if on that drive, else the drive's root.
                let mut base = info
                    .drive_cwds
                    .get(&drive.to_uppercase())
                    .cloned()
                    .unwrap_or_else(|| info.cwd.clone());
                if !base[..base.len().min(2)].eq_ignore_ascii_case(drive)
                    && base.as_bytes().get(2) == Some(&b'\\')
                {
                    base = format!("{drive}\\");
                }
                return Ok(paths.resolve(&[&base, &normalized], &info.cwd, &info.drive_cwds));
            }
        }
        Ok(paths.resolve(&[&self.cwd, &normalized], &info.cwd, &info.drive_cwds))
    }

    /// Resolve, check for abort, run; `check == "both"` checks again after the operation ran.
    async fn file_op<T, F, Fut>(
        &self,
        path: &str,
        context: &Context,
        check: Check,
        run: F,
    ) -> Result<T, FileError>
    where
        F: FnOnce(String) -> Fut,
        Fut: Future<Output = Result<T, FileError>>,
    {
        let resolved = self
            .resolve(path)
            .await
            .map_err(|error| to_file_error(&error, Some(path)))?;
        if aborted(context.abort_signal().as_ref()) {
            return Err(abort_error(Some(resolved.as_str())));
        }
        let value = run(resolved.clone()).await;
        let value = value.map_err(|error| FileError {
            path: error.path.or_else(|| Some(resolved.clone())),
            ..error
        });
        if check == Check::Both && aborted(context.abort_signal().as_ref()) {
            return Err(abort_error(Some(resolved.as_str())));
        }
        value
    }

    async fn open_handle(
        &self,
        resolved: &str,
        mut json: Json,
    ) -> Result<(Handle, Json), FileError> {
        let reply = self
            .connection
            .request(
                "open",
                {
                    json.insert("path".into(), json!(resolved));
                    json
                },
                RequestOptions::default(),
            )
            .await
            .map_err(|error| to_file_error(&error, Some(resolved)))?;
        let id = reply
            .json
            .get("handle")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        Ok((
            Handle {
                env: self.clone(),
                id,
                session: reply.session,
                path: resolved.to_string(),
            },
            reply.json,
        ))
    }

    /// Node's `readFile`, with or without UTF-8 decoding, over a file opened like `open(path, "r")`.
    async fn read_file(
        &self,
        path: &str,
        encoding: bool,
        context: &Context,
    ) -> Result<Vec<u8>, FileError> {
        let signal = context.abort_signal();
        let aborted_now = || signal.as_ref().is_some_and(AbortSignal::aborted);
        self.file_op(path, context, Check::Before, async |resolved| {
            let (handle, json) = self.open_handle(&resolved, Json::new()).await?;
            let result = async {
                if let Some(stat_error) = json.get("statError") {
                    let error = RemoteError::from_fields(
                        stat_error.as_object().cloned().unwrap_or_default(),
                    );
                    return Err(to_file_error(&error, Some(&resolved)));
                }
                if aborted_now() {
                    return Err(FileError::new(
                        FileErrorCode::Aborted,
                        "The operation was aborted",
                        Some(resolved.clone()),
                    ));
                }
                let stat = json.get("stat").cloned().unwrap_or_default();
                let size = if stat.get("kind").and_then(Value::as_str) == Some("file") {
                    stat.get("size").and_then(Value::as_u64).unwrap_or(0)
                } else {
                    0
                };
                if size > READ_FILE_MAX {
                    return Err(FileError::new(
                        FileErrorCode::Unknown,
                        format!("File size ({size}) is greater than 2 GiB"),
                        Some(resolved.clone()),
                    ));
                }
                if size == 0 {
                    // Unknown size: sequential reads until the end, as for FIFOs and devices.
                    let mut decoder = StreamDecoder::new();
                    let mut bytes = Vec::new();
                    let mut text = String::new();
                    loop {
                        if aborted_now() {
                            return Err(FileError::new(
                                FileErrorCode::Aborted,
                                "The operation was aborted",
                                Some(resolved.clone()),
                            ));
                        }
                        let chunk = handle
                            .pread(None, READ_FILE_UNKNOWN_CHUNK)
                            .await
                            .map_err(|error| to_file_error(&error, Some(&resolved)))?;
                        if chunk.is_empty() {
                            break;
                        }
                        bytes.extend_from_slice(&chunk);
                        if encoding {
                            text.push_str(&decoder.decode(Some(&chunk)));
                        }
                    }
                    return Ok(if encoding {
                        if bytes.is_empty() {
                            Vec::new()
                        } else {
                            text.push_str(&decoder.decode(None));
                            text.into_bytes()
                        }
                    } else {
                        bytes
                    });
                }
                if !encoding {
                    // A single read of the whole size, or reads of 512 KiB until the size or the end of the file.
                    if size <= READ_FILE_CHUNK as u64 {
                        return handle
                            .pread(Some(0), size as usize)
                            .await
                            .map_err(|error| to_file_error(&error, Some(&resolved)));
                    }
                    let read = read_range(
                        |at, length| handle.pread(Some(at), length),
                        0,
                        size,
                        READ_FILE_CHUNK,
                        aborted_now,
                    )
                    .await;
                    return match read {
                        Ok(Some((chunks, total))) => Ok(concat(chunks, total)),
                        Ok(None) => Err(FileError::new(
                            FileErrorCode::Aborted,
                            "The operation was aborted",
                            Some(resolved.clone()),
                        )),
                        Err(error) => Err(to_file_error(&error, Some(&resolved))),
                    };
                }
                // Decoding: reads of min(size, 512 KiB) until the size, a short read, or the end of the file.
                let length = size.min(READ_FILE_CHUNK as u64) as usize;
                let mut decoder = StreamDecoder::new();
                let mut text = String::new();
                let mut total = 0usize;
                let mut first = true;
                let mut in_flight: std::collections::VecDeque<(usize, _)> = Default::default();
                let mut next = 0u64;
                loop {
                    while in_flight.len() < READ_DEPTH && next < size {
                        let request = length.min((size - next) as usize);
                        in_flight.push_back((request, handle.pread(Some(next), request)));
                        next += request as u64;
                    }
                    let (request, bytes) = in_flight.pop_front().unwrap();
                    let bytes = bytes
                        .await
                        .map_err(|error| to_file_error(&error, Some(&resolved)))?;
                    if aborted_now() {
                        return Err(FileError::new(
                            FileErrorCode::Aborted,
                            "The operation was aborted",
                            Some(resolved.clone()),
                        ));
                    }
                    total += bytes.len();
                    if bytes.is_empty() || total == size as usize || bytes.len() != request {
                        return Ok(if first {
                            bytes
                        } else {
                            let mut text = std::mem::take(&mut text);
                            text.push_str(&decoder.decode(Some(&bytes)));
                            text.push_str(&decoder.decode(None));
                            text.into_bytes()
                        });
                    }
                    if first {
                        text = String::from_utf8_lossy(&bytes).into_owned();
                        first = false;
                    } else {
                        text.push_str(&decoder.decode(Some(&bytes)));
                    }
                }
            }
            .await;
            handle.close().await;
            result
        })
        .await
    }

    /// Node's `writeFile` (abort checked before each 512 KiB write) or `appendFile` (no checks, one
    /// after): the first request creates parents, opens, and writes the first chunk; later chunks go
    /// to the same open file.
    async fn write(
        &self,
        path: &str,
        content: FileContent,
        append: bool,
        context: &Context,
    ) -> Result<(), FileError> {
        let bytes = match content {
            FileContent::Text(text) => text.into_bytes(),
            FileContent::Binary(bytes) => bytes,
        };
        let signal = if append { None } else { context.abort_signal() };
        self.file_op(
            path,
            context,
            if append { Check::Both } else { Check::Before },
            async |resolved| {
                let first = bytes.len().min(WRITE_CHUNK);
                let keep = bytes.len() > WRITE_CHUNK;
                let mut fields = Map::new();
                fields.insert("path".into(), json!(resolved));
                fields.insert("append".into(), json!(append));
                fields.insert("keep".into(), json!(keep));
                let reply = self
                    .connection
                    .request(
                        "write",
                        fields,
                        RequestOptions {
                            payload: bytes[..first].to_vec(),
                            ..Default::default()
                        },
                    )
                    .await
                    .map_err(|error| to_file_error(&error, Some(&resolved)))?;
                if !keep {
                    return Ok(());
                }
                let handle = Handle {
                    env: self.clone(),
                    id: reply
                        .json
                        .get("handle")
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                    session: reply.session,
                    path: resolved.clone(),
                };
                let mut in_flight: std::collections::VecDeque<_> = Default::default();
                let result: Result<(), FileError> = async {
                    let mut offset = WRITE_CHUNK;
                    while offset < bytes.len() {
                        if signal.as_ref().is_some_and(AbortSignal::aborted) {
                            // A chunk that failed before this check fails the write, as it would in sequence.
                            let mut failure: Option<FileError> = None;
                            for written in in_flight.drain(..) {
                                if let Err(error) = written.await
                                    && failure.is_none()
                                {
                                    failure = Some(error);
                                }
                            }
                            if let Some(error) = failure {
                                return Err(error);
                            }
                            return Err(FileError::new(
                                FileErrorCode::Aborted,
                                "The operation was aborted",
                                Some(resolved.clone()),
                            ));
                        }
                        let end = (offset + WRITE_CHUNK).min(bytes.len());
                        let resolved_chunk = resolved.clone();
                        let written = handle.request(
                            "writeChunk",
                            Map::new(),
                            RequestOptions {
                                payload: bytes[offset..end].to_vec(),
                                ..Default::default()
                            },
                        );
                        let written = async move {
                            written
                                .await
                                .map(|_| ())
                                .map_err(|error| to_file_error(&error, Some(&resolved_chunk)))
                        };
                        in_flight.push_back(written);
                        if in_flight.len() >= WRITE_DEPTH {
                            in_flight.pop_front().unwrap().await?;
                        }
                        offset = end;
                    }
                    // The first failure; later chunks are refused by the daemon, so the file has no gaps.
                    while let Some(written) = in_flight.pop_front() {
                        written.await?;
                    }
                    Ok(())
                }
                .await;
                while let Some(written) = in_flight.pop_front() {
                    let _ = written.await;
                }
                handle.close().await;
                result
            },
        )
        .await
    }

    async fn open_dir_handle(&self, resolved: &str) -> Result<Handle, FileError> {
        let mut fields = Map::new();
        fields.insert("path".into(), json!(resolved));
        let reply = self
            .connection
            .request("opendir", fields, RequestOptions::default())
            .await
            .map_err(|error| to_file_error(&error, Some(resolved)))?;
        Ok(Handle {
            env: self.clone(),
            id: reply
                .json
                .get("handle")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            session: reply.session,
            path: resolved.to_string(),
        })
    }

    async fn op(
        &self,
        op: &str,
        fields: Json,
        resolved: Option<&str>,
        context: &Context,
    ) -> Result<(Json, Vec<u8>), FileError> {
        let mut request = RequestOptions::default();
        request.signal = context.abort_signal();
        self.connection
            .request(op, fields, request)
            .await
            .map(|reply| (reply.json, reply.payload))
            .map_err(|error| to_file_error(&error, resolved))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Check {
    Before,
    Both,
}

#[async_trait]
impl FileSystem for RemoteExecutionEnv {
    fn id(&self) -> &str {
        &self.id
    }

    fn cwd(&self) -> &str {
        &self.cwd
    }

    async fn absolute_path(&self, path: &str, _context: &Context) -> Result<String, FileError> {
        self.resolve(path)
            .await
            .map_err(|error| to_file_error(&error, Some(path)))
    }

    async fn join_path(&self, parts: &[String], _context: &Context) -> Result<String, FileError> {
        let paths = remote_path(&self.connection)
            .await
            .map_err(|error| to_file_error(&error, None))?;
        let references: Vec<&str> = parts.iter().map(String::as_str).collect();
        Ok(paths.join(&references))
    }

    async fn read_text_file(&self, path: &str, context: &Context) -> Result<String, FileError> {
        let bytes = self.read_file(path, true, context).await?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    async fn open_text_line_reader(
        &self,
        path: &str,
        context: &Context,
    ) -> Result<Arc<dyn TextLineReader>, FileError> {
        let handle = self
            .file_op(path, context, Check::Before, async |resolved| {
                let (handle, _) = self
                    .open_handle(&resolved, {
                        let mut fields = Map::new();
                        fields.insert("mode".into(), json!("read"));
                        fields
                    })
                    .await?;
                Ok(handle)
            })
            .await?;
        if aborted(context.abort_signal().as_ref()) {
            handle.close().await;
            return Err(abort_error(Some(handle.path.as_str())));
        }
        Ok(Arc::new(RemoteTextLineReader {
            handle,
            decoder: parking_lot::Mutex::new(StreamDecoder::new()),
            offset: parking_lot::Mutex::new(0),
            buffered: parking_lot::Mutex::new(String::new()),
            ended: parking_lot::Mutex::new(false),
            closed: parking_lot::Mutex::new(false),
        }))
    }

    async fn read_text_lines(
        &self,
        path: &str,
        options: Option<ReadTextLinesOptions>,
        context: &Context,
    ) -> Result<Vec<String>, FileError> {
        if options
            .and_then(|options| options.max_lines)
            .is_some_and(|max| max == 0)
        {
            return Ok(Vec::new());
        }
        let reader = self.open_text_line_reader(path, context).await?;
        let result: Result<Vec<String>, FileError> = async {
            let mut lines = Vec::new();
            while options
                .and_then(|options| options.max_lines)
                .is_none_or(|max| lines.len() < max)
            {
                let Some(line) = reader.read_line(context).await? else {
                    break;
                };
                lines.push(line.text);
            }
            Ok(lines)
        }
        .await;
        reader.close(context).await;
        result
    }

    async fn read_binary_file(&self, path: &str, context: &Context) -> Result<Vec<u8>, FileError> {
        self.read_file(path, false, context).await
    }

    async fn open_binary_reader(
        &self,
        path: &str,
        options: Option<OpenBinaryOptions>,
        context: &Context,
    ) -> Result<Arc<dyn BinaryReader>, FileError> {
        let no_follow = options.is_some_and(|options| options.no_follow);
        let handle = self
            .file_op(path, context, Check::Before, async |resolved| {
                let (handle, _) = self
                    .open_handle(&resolved, {
                        let mut fields = Map::new();
                        fields.insert("noFollow".into(), json!(no_follow));
                        fields
                    })
                    .await?;
                Ok(handle)
            })
            .await?;
        if aborted(context.abort_signal().as_ref()) {
            handle.close().await;
            return Err(abort_error(Some(handle.path.as_str())));
        }
        Ok(Arc::new(RemoteBinaryReader {
            handle,
            closed: parking_lot::Mutex::new(false),
        }))
    }

    async fn write_file(
        &self,
        path: &str,
        content: FileContent,
        context: &Context,
    ) -> Result<(), FileError> {
        self.write(path, content, false, context).await
    }

    async fn append_file(
        &self,
        path: &str,
        content: FileContent,
        context: &Context,
    ) -> Result<(), FileError> {
        self.write(path, content, true, context).await
    }

    async fn truncate_file(
        &self,
        path: &str,
        size: u64,
        context: &Context,
    ) -> Result<(), FileError> {
        self.file_op(path, context, Check::Both, async |resolved| {
            let mut fields = Map::new();
            fields.insert("path".into(), json!(resolved));
            fields.insert("size".into(), json!(size));
            self.op("truncate", fields, Some(&resolved), context)
                .await
                .map(|_| ())
        })
        .await
    }

    async fn flush_file(&self, path: &str, context: &Context) -> Result<(), FileError> {
        self.file_op(path, context, Check::Both, async |resolved| {
            let mut fields = Map::new();
            fields.insert("path".into(), json!(resolved));
            self.op("fsync", fields, Some(&resolved), context)
                .await
                .map(|_| ())
        })
        .await
    }

    async fn rename_file(
        &self,
        source_path: &str,
        destination_path: &str,
        context: &Context,
    ) -> Result<(), FileError> {
        let source = self
            .resolve(source_path)
            .await
            .map_err(|error| to_file_error(&error, Some(source_path)))?;
        let destination = self
            .resolve(destination_path)
            .await
            .map_err(|error| to_file_error(&error, Some(source_path)))?;
        if aborted(context.abort_signal().as_ref()) {
            return Err(abort_error(Some(destination.as_str())));
        }
        let mut fields = Map::new();
        fields.insert("path".into(), json!(source));
        fields.insert("to".into(), json!(destination));
        let mut request = RequestOptions::default();
        request.signal = context.abort_signal();
        self.connection
            .request("rename", fields, request)
            .await
            .map(|_| ())
            .map_err(|error| to_file_error(&error, Some(&source)))
    }

    async fn file_info(&self, path: &str, context: &Context) -> Result<FileInfo, FileError> {
        let paths = remote_path(&self.connection)
            .await
            .map_err(|error| to_file_error(&error, Some(path)))?;
        self.file_op(path, context, Check::Before, async |resolved| {
            let mut fields = Map::new();
            fields.insert("path".into(), json!(resolved));
            let (json, _) = self.op("lstat", fields, Some(&resolved), context).await?;
            to_info(&resolved, &json, paths)
        })
        .await
    }

    async fn list_dir(&self, path: &str, context: &Context) -> Result<Vec<FileInfo>, FileError> {
        let paths = remote_path(&self.connection)
            .await
            .map_err(|error| to_file_error(&error, Some(path)))?;
        self.file_op(path, context, Check::Before, async |resolved| {
            let handle = self.open_dir_handle(&resolved).await?;
            let result: Result<Vec<RemoteDirEntry>, FileError> = async {
                let mut entries = Vec::new();
                let mut done = false;
                while !done {
                    let mut fields = Map::new();
                    fields.insert("max".into(), json!(1000));
                    let reply = handle
                        .request("readdir", fields, RequestOptions::default())
                        .await
                        .map_err(|error| to_file_error(&error, Some(&resolved)))?;
                    let json = reply.json;
                    entries.extend(
                        json.get("entries")
                            .and_then(Value::as_array)
                            .into_iter()
                            .flatten()
                            .filter_map(Value::as_object)
                            .cloned(),
                    );
                    done = json.get("done").and_then(Value::as_bool).unwrap_or(false);
                }
                Ok(entries)
            }
            .await;
            handle.close().await;
            let mut entries = result?;
            // Node's `readdir` fails on any entry it cannot lstat. libuv sorts names by their bytes on
            // POSIX; on Windows it keeps the file system's order, which the daemon reports.
            if paths == Paths::Posix {
                entries.sort_by_key(|entry| {
                    entry
                        .get("raw")
                        .and_then(Value::as_array)
                        .map(|bytes| {
                            bytes
                                .iter()
                                .filter_map(Value::as_u64)
                                .map(|byte| byte as u8)
                                .collect::<Vec<u8>>()
                        })
                        .unwrap_or_else(|| {
                            entry
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .as_bytes()
                                .to_vec()
                        })
                });
            }
            let mut infos = Vec::new();
            for entry in entries {
                if aborted(context.abort_signal().as_ref()) {
                    return Err(FileError::aborted(Some(resolved.clone())));
                }
                let name = entry
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let entry_path = paths.resolve(&[&resolved, &name], "", &HashMap::new());
                if let Some(error) = entry.get("error") {
                    let error =
                        RemoteError::from_fields(error.as_object().cloned().unwrap_or_default());
                    return Err(to_file_error(&error, Some(&entry_path)));
                }
                if let Ok(info) = to_info(&entry_path, &entry, paths) {
                    infos.push(info);
                }
            }
            Ok(infos)
        })
        .await
    }

    async fn open_dir_reader(
        &self,
        path: &str,
        context: &Context,
    ) -> Result<Arc<dyn DirReader>, FileError> {
        let handle = self
            .file_op(path, context, Check::Before, async |resolved| {
                self.open_dir_handle(&resolved).await
            })
            .await?;
        if aborted(context.abort_signal().as_ref()) {
            handle.close().await;
            return Err(abort_error(Some(handle.path.as_str())));
        }
        Ok(Arc::new(RemoteDirReader {
            handle,
            done: parking_lot::Mutex::new(false),
            closed: parking_lot::Mutex::new(false),
        }))
    }

    async fn canonical_path(&self, path: &str, context: &Context) -> Result<String, FileError> {
        self.file_op(path, context, Check::Before, async |resolved| {
            let mut fields = Map::new();
            fields.insert("path".into(), json!(resolved));
            let (json, _) = self
                .op("realpath", fields, Some(&resolved), context)
                .await?;
            Ok(json
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string())
        })
        .await
    }

    async fn exists(&self, path: &str, context: &Context) -> Result<bool, FileError> {
        match self.file_info(path, context).await {
            Ok(_) => Ok(true),
            Err(error) if error.code == FileErrorCode::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    async fn create_dir(
        &self,
        path: &str,
        options: Option<CreateDirOptions>,
        context: &Context,
    ) -> Result<(), FileError> {
        let recursive = options
            .and_then(|options| options.recursive)
            .unwrap_or(true);
        self.file_op(path, context, Check::Before, async |resolved| {
            let mut fields = Map::new();
            fields.insert("path".into(), json!(resolved));
            fields.insert("recursive".into(), json!(recursive));
            self.op("mkdir", fields, Some(&resolved), context)
                .await
                .map(|_| ())
        })
        .await
    }

    async fn remove(
        &self,
        path: &str,
        options: Option<RemoveOptions>,
        context: &Context,
    ) -> Result<(), FileError> {
        let options = options.unwrap_or_default();
        let recursive = options.recursive.unwrap_or(false);
        let force = options.force.unwrap_or(false);
        self.file_op(path, context, Check::Before, async |resolved| {
            let mut fields = Map::new();
            fields.insert("path".into(), json!(resolved));
            fields.insert("recursive".into(), json!(recursive));
            fields.insert("force".into(), json!(force));
            self.op("rm", fields, Some(&resolved), context)
                .await
                .map(|_| ())
        })
        .await
    }

    async fn create_temp_dir(
        &self,
        prefix: Option<&str>,
        context: &Context,
    ) -> Result<String, FileError> {
        if aborted(context.abort_signal().as_ref()) {
            return Err(abort_error(None));
        }
        let info = self
            .connection
            .info()
            .await
            .map_err(|error| to_file_error(&error, None))?;
        let paths = Paths::for_os(&info.os);
        let mut fields = Map::new();
        fields.insert(
            "path".into(),
            json!(paths.join(&[&info.tmpdir, prefix.unwrap_or("tmp-")])),
        );
        let (json, _) = self.op("mkdtemp", fields, None, context).await?;
        Ok(json
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string())
    }

    async fn create_temp_file(
        &self,
        options: Option<TempFileOptions>,
        context: &Context,
    ) -> Result<String, FileError> {
        let dir = self.create_temp_dir(Some("tmp-"), context).await?;
        let paths = remote_path(&self.connection)
            .await
            .map_err(|error| to_file_error(&error, None))?;
        let name = format!(
            "{}{}{}",
            options
                .as_ref()
                .and_then(|options| options.prefix.as_deref())
                .unwrap_or(""),
            unique()?,
            options
                .as_ref()
                .and_then(|options| options.suffix.as_deref())
                .unwrap_or("")
        );
        let file_path = paths.join(&[&dir, &name]);
        // Node's `writeFile(filePath, "")`, which does not create parents.
        let mut fields = Map::new();
        fields.insert("path".into(), json!(file_path));
        fields.insert("append".into(), json!(false));
        fields.insert("parents".into(), json!(false));
        self.op("write", fields, Some(&file_path), context)
            .await
            .map(|_| ())?;
        Ok(file_path)
    }

    async fn cleanup(&self, _context: &Context) {
        // Kill without aborting, so the commands settle with their killed status, as `NodeExecutionEnv` does.
        self.kill_running();
    }

    async fn watch(
        &self,
        targets: &[WatchTarget],
        on_change: WatchCallback,
        context: &Context,
    ) -> Result<Arc<dyn FileWatcher>, FileError> {
        if aborted(context.abort_signal().as_ref()) {
            return Err(abort_error(None));
        }
        let mut resolved_targets = Vec::with_capacity(targets.len());
        for target in targets {
            let mut target = target.clone();
            target.path = self
                .resolve(&target.path)
                .await
                .map_err(|error| to_file_error(&error, Some(&target.path)))?;
            resolved_targets.push(target);
        }
        if aborted(context.abort_signal().as_ref()) {
            return Err(abort_error(None));
        }
        let watcher = RemoteWatcher::open(
            self.connection.clone(),
            &resolved_targets,
            on_change,
            self.watch_options,
        )
        .await
        .map_err(|error| to_file_error(&error, None))?;
        if aborted(context.abort_signal().as_ref()) {
            watcher.close().await;
            return Err(abort_error(None));
        }
        Ok(Arc::new(watcher))
    }
}

impl RemoteExecutionEnv {
    /// Kill every command this environment still runs, without aborting it; both `cleanup`s share it.
    fn kill_running(&self) {
        for (id, session) in self.running.lock().drain() {
            self.connection.kill(id, session);
        }
    }

    async fn exec_impl(
        &self,
        command: ExecCommand,
        options: Option<ShellExecOptions>,
        context: &Context,
    ) -> Result<ShellExecResult, ExecutionError> {
        let options = options.unwrap_or_default();
        let signal = context.abort_signal();
        if signal.as_ref().is_some_and(AbortSignal::aborted) {
            return Err(ExecutionError::new(ExecutionErrorCode::Aborted, "aborted"));
        }
        let timeout = options.timeout;
        if let Some(timeout) = timeout {
            if !timeout.is_finite() || timeout <= 0.0 {
                return Err(ExecutionError::new(
                    ExecutionErrorCode::Timeout,
                    "Invalid timeout: must be a finite number of seconds",
                ));
            }
            if timeout * 1000.0 > MAX_TIMEOUT_MS {
                return Err(ExecutionError::new(
                    ExecutionErrorCode::Timeout,
                    format!("Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds"),
                ));
            }
        }
        let inherit_env = options.inherit_env.unwrap_or(true);
        let cwd = match &options.cwd {
            Some(cwd) => self.resolve(cwd).await.map_err(|error| {
                ExecutionError::new(ExecutionErrorCode::Unknown, error.to_string())
            })?,
            None => self.cwd.clone(),
        };
        if signal.as_ref().is_some_and(AbortSignal::aborted) {
            return Err(ExecutionError::new(ExecutionErrorCode::Aborted, "aborted"));
        }
        let mut env: Map<String, Value> = Map::new();
        let empty: HashMap<String, String> = HashMap::new();
        let inherited: &HashMap<String, String> = if inherit_env {
            self.shell_env.as_ref().unwrap_or(&empty)
        } else {
            &empty
        };
        let overrides: &HashMap<String, String> = options.env.as_ref().unwrap_or(&empty);
        for (key, value) in inherited.iter().chain(overrides.iter()) {
            env.insert(key.clone(), json!(value));
        }
        let callback_error: Arc<parking_lot::Mutex<Option<ExecutionError>>> =
            Arc::new(parking_lot::Mutex::new(None));
        let settled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let running: Arc<parking_lot::Mutex<Option<(u32, u64)>>> =
            Arc::new(parking_lot::Mutex::new(None));
        // A failing `on_output` aborts the command; the request settles with the daemon's answer.
        let controller = AbortController::new();
        let external = signal;
        let internal = controller.signal();
        let combined = match external {
            Some(external) => Some(AbortSignal::any([external, internal])),
            None => Some(internal),
        };
        let on_output = options.on_output.clone();
        let event_callback: Option<Arc<dyn Fn(Json, Vec<u8>) + Send + Sync>> =
            on_output.map(|on_output| -> Arc<dyn Fn(Json, Vec<u8>) + Send + Sync> {
                let callback_error = callback_error.clone();
                let settled = settled.clone();
                Arc::new(move |event: Json, payload: Vec<u8>| {
                    if settled.load(std::sync::atomic::Ordering::SeqCst)
                        || callback_error.lock().is_some()
                    {
                        return;
                    }
                    if event.get("kind").and_then(Value::as_str) != Some("output") {
                        return;
                    }
                    let text = String::from_utf8_lossy(&payload).into_owned();
                    if text.is_empty() {
                        return;
                    }
                    let mut info = ShellOutputInfo::default();
                    info.stream = if event.get("stream").and_then(Value::as_str) == Some("stderr") {
                        pi_durable::env::OutputStream::Stderr
                    } else {
                        pi_durable::env::OutputStream::Stdout
                    };
                    if let Some(skipped) = event.get("skipped") {
                        info.skipped = Some(ShellOutputSkip {
                            bytes: skipped.get("bytes").and_then(Value::as_u64).unwrap_or(0),
                            newlines: skipped.get("newlines").and_then(Value::as_u64).unwrap_or(0),
                            ends_with_newline: skipped
                                .get("endsWithNewline")
                                .and_then(Value::as_bool)
                                .unwrap_or(false),
                        });
                    }
                    let delivered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        on_output(&text, &info)
                    }));
                    if let Err(panic) = delivered {
                        let message = panic
                            .downcast_ref::<&str>()
                            .map(|message| message.to_string())
                            .or_else(|| panic.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "output callback failed".to_string());
                        *callback_error.lock() = Some(ExecutionError::new(
                            ExecutionErrorCode::CallbackError,
                            message,
                        ));
                        controller.abort_default();
                    }
                })
            });
        let mut fields = Map::new();
        match &command {
            ExecCommand::Shell(command) => {
                fields.insert("command".into(), json!(command));
            }
            ExecCommand::Argv(argv) => {
                fields.insert("argv".into(), json!(argv));
            }
        }
        fields.insert("cwd".into(), json!(cwd));
        fields.insert("env".into(), Value::Object(env));
        fields.insert("inheritEnv".into(), json!(inherit_env));
        if let Some(shell_path) = &self.shell_path {
            fields.insert("shellPath".into(), json!(shell_path));
        }
        if let Some(timeout) = timeout {
            fields.insert("timeoutMs".into(), json!(timeout * 1000.0));
        }
        if let Some(spill) = options.spill {
            fields.insert(
                "spill".into(),
                json!({ "afterBytes": spill.after_bytes, "afterLines": spill.after_lines }),
            );
        }
        if let Some(window) = options.window {
            fields.insert(
                "window".into(),
                json!({
                    "maxBytes": window.max_bytes,
                    "maxLines": window.max_lines,
                    "minIntervalMs": window.min_interval_ms,
                    "bytesPerSecond": window.bytes_per_second,
                }),
            );
        }
        let running_map = self.running.clone();
        let on_start_callback = running.clone();
        let request = RequestOptions {
            signal: combined,
            on_event: event_callback,
            on_start: Some(Arc::new(move |id: u32, session: u64| {
                *on_start_callback.lock() = Some((id, session));
                running_map.lock().insert(id, session);
            })),
            ..Default::default()
        };
        let result = self.connection.request("exec", fields, request).await;
        settled.store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some((id, _)) = running.lock().take() {
            self.running.lock().remove(&id);
        }
        if let Some(error) = callback_error.lock().clone() {
            return Err(error);
        }
        match result {
            Ok(reply) => {
                let json = reply.json;
                let mut result = ShellExecResult {
                    exit_code: json
                        .get("exitCode")
                        .and_then(Value::as_i64)
                        .unwrap_or_default() as i32,
                    spill_path: None,
                };
                if let Some(spill_path) = json.get("spillPath").and_then(Value::as_str) {
                    result.spill_path = Some(spill_path.to_string());
                }
                Ok(result)
            }
            Err(error) => {
                let mut failure = match error.code.as_str() {
                    "timeout" => ExecutionError::new(
                        ExecutionErrorCode::Timeout,
                        format!("timeout:{:?}", timeout.unwrap_or_default()),
                    ),
                    "aborted" => ExecutionError::new(ExecutionErrorCode::Aborted, "aborted"),
                    "shell_unavailable" => ExecutionError::new(
                        ExecutionErrorCode::ShellUnavailable,
                        error.message.clone(),
                    ),
                    "spawn_error" => {
                        ExecutionError::new(ExecutionErrorCode::SpawnError, error.message.clone())
                    }
                    _ => ExecutionError::new(ExecutionErrorCode::Unknown, error.message.clone()),
                };
                if let Some(spill_path) = error.fields.get("spillPath").and_then(Value::as_str) {
                    failure.set_spill_path(spill_path.to_string());
                }
                Err(failure)
            }
        }
    }
}

fn unique() -> Result<String, FileError> {
    let mut bytes = [0u8; 16];
    getrandom::getrandom(&mut bytes).map_err(|error| {
        FileError::new(
            FileErrorCode::Unknown,
            format!("Failed to generate random bytes: {error}"),
            None,
        )
    })?;
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    Ok(format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    ))
}

#[async_trait]
impl Shell for RemoteExecutionEnv {
    async fn exec(
        &self,
        command: ExecCommand,
        options: Option<ShellExecOptions>,
        context: &Context,
    ) -> std::result::Result<ShellExecResult, ExecutionError> {
        self.exec_impl(command, options, context).await
    }

    async fn cleanup(&self, _context: &Context) {
        // Kill without aborting, so the commands settle with their killed status, as `NodeExecutionEnv` does.
        self.kill_running();
    }
}
