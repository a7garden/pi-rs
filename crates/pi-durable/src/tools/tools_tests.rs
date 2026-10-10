//! Port of `packages/durable/test/tools.test.ts` — same assertions, adapted to the erased Rust
//! seam: a thrown tool error becomes an error result (`isError` + `tool_error` diagnostic; see
//! `tools` module D1), so `rejects.toThrow(message)` assertions check that diagnostic message.
//! The fake API implements `ToolExecutionApi` with the environment, collected output, and
//! diagnostics; environment subclasses become [`TestEnv`] hook variants.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use pi_chord::context::{with_abort_signal, AbortController, Context, BACKGROUND_CONTEXT};
use serde_json::{json, Value};
use tokio::sync::Notify;

use crate::env::node::NodeExecutionEnv;
use crate::env::{
    get_or_throw, BinaryReader, ExecCommand, ExecutionEnv, ExecutionError, ExecutionErrorCode,
    FileContent, FileError, FileInfo, FileSystem, LineScan, ScanLinesOptions, Shell,
    ShellExecOptions, ShellExecResult, ShellOutputInfo, ShellOutputSkip, ShellOutputWindow,
};
use crate::harness::api::{
    ToolDiagnostic, ToolExecutionApi, ToolExecutionResult, ToolRegistration,
};
use crate::harness::types::RegistrySnapshot;
use crate::truncate::DEFAULT_MAX_LINES;

use super::edit::prepare_edit_arguments;
use super::file_mutation_queue::with_file_mutation_queue;
use super::image::detect_supported_image_mime_type;
use super::{
    create_bash_tool, create_edit_tool, create_power_shell_tool, create_read_tool,
    create_write_tool, BashExecution, BashToolOptions, PowerShellToolOptions,
};

// -- harness -----------------------------------------------------------------

fn create_temp_dir() -> String {
    let dir = std::env::temp_dir().join(format!(
        "pi-durable-tools-{}-{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos(),
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir.to_string_lossy().into_owned()
}

pub(crate) fn node_env(cwd: String) -> NodeExecutionEnv {
    NodeExecutionEnv::new(crate::env::NodeEnvOptions {
        cwd,
        ..Default::default()
    })
}

fn node_env_with_shell_env(cwd: String, shell_env: Vec<(&str, &str)>) -> NodeExecutionEnv {
    let _ = cwd;
    NodeExecutionEnv::new(crate::env::NodeEnvOptions {
        cwd,
        shell_env: Some(
            shell_env
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        ),
        ..Default::default()
    })
}

fn some_env<E: ExecutionEnv + 'static>(env: Arc<E>) -> Option<Arc<dyn ExecutionEnv>> {
    Some(env)
}

fn get<T>(result: crate::env::Result<T>) -> T {
    get_or_throw(result)
}

async fn write(env: &dyn FileSystem, path: &str, content: impl Into<FileContent>) {
    get(env
        .write_file(path, content.into(), &BACKGROUND_CONTEXT)
        .await);
}

async fn read_text(env: &dyn FileSystem, path: &str) -> String {
    get(env.read_text_file(path, &BACKGROUND_CONTEXT).await)
}

/// Test environment wrapping [`NodeExecutionEnv`] with upstream's subclass overrides.
struct TestEnv {
    inner: NodeExecutionEnv,
    kind: EnvKind,
}

enum EnvKind {
    Plain,
    /// `SlowReadExecutionEnv`: `readTextFile` delays 20ms.
    SlowRead,
    /// `BlockingWriteExecutionEnv`.
    BlockingWrite(Arc<BlockingState>),
    /// `BlockingEditExecutionEnv`.
    BlockingEdit(Arc<BlockingState>),
    /// `TimeoutOutputExecutionEnv`: emits `DEFAULT_MAX_LINES + 1` lines, spills, then times out.
    TimeoutOutput,
    /// `SkippingEnv`: records the received window, emits one chunk with a skip, succeeds.
    Skipping(Arc<SkipShared>),
    /// `ProgramsEnv`: only the listed programs exist; records each command, prints `output`.
    Programs(Arc<ProgramsShared>),
    /// `GrowingLogExecutionEnv`: every reader `read` appends to the log, as a busy writer would.
    GrowingLog(String),
    /// `OtherFileSystem`: a distinct file-system id over the same files.
    OtherFs(Arc<BlockingState>),
}

/// Shared state of `EnvKind::Programs`.
struct ProgramsShared {
    installed: Vec<String>,
    output: String,
    exit_code: i32,
    commands: Mutex<Vec<ExecCommand>>,
}

struct SkipShared {
    skip: ShellOutputSkip,
    received_window: Mutex<Option<ShellOutputWindow>>,
}

struct BlockingState {
    first_write_started: Notify,
    finish_first_write: Notify,
    second_write_started: AtomicBool,
    first_edit_write_settled: AtomicBool,
}

impl BlockingState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            first_write_started: Notify::new(),
            finish_first_write: Notify::new(),
            second_write_started: AtomicBool::new(false),
            first_edit_write_settled: AtomicBool::new(false),
        })
    }
}

impl TestEnv {
    fn plain(cwd: String) -> Self {
        Self {
            inner: node_env(cwd),
            kind: EnvKind::Plain,
        }
    }

    fn cwd(&self) -> &str {
        self.inner.cwd()
    }

    async fn write_file_inner(
        &self,
        path: &str,
        content: FileContent,
        context: &Context,
    ) -> Result<(), FileError> {
        self.inner.write_file(path, content, context).await
    }
}

#[async_trait::async_trait]
impl FileSystem for TestEnv {
    fn id(&self) -> &str {
        match &self.kind {
            EnvKind::OtherFs(_) => "other",
            _ => self.inner.id(),
        }
    }

    fn cwd(&self) -> &str {
        self.inner.cwd()
    }

    async fn absolute_path(&self, path: &str, context: &Context) -> crate::env::Result<String> {
        self.inner.absolute_path(path, context).await
    }

    async fn join_path(&self, parts: &[String], context: &Context) -> crate::env::Result<String> {
        self.inner.join_path(parts, context).await
    }

    async fn read_text_file(&self, path: &str, context: &Context) -> crate::env::Result<String> {
        if matches!(self.kind, EnvKind::SlowRead) {
            delay(20).await;
        }
        self.inner.read_text_file(path, context).await
    }

    async fn open_text_line_reader(
        &self,
        path: &str,
        context: &Context,
    ) -> crate::env::Result<Arc<dyn crate::env::TextLineReader>> {
        self.inner.open_text_line_reader(path, context).await
    }

    async fn read_text_lines(
        &self,
        path: &str,
        options: Option<crate::env::ReadTextLinesOptions>,
        context: &Context,
    ) -> crate::env::Result<Vec<String>> {
        self.inner.read_text_lines(path, options, context).await
    }

    async fn read_binary_file(&self, path: &str, context: &Context) -> crate::env::Result<Vec<u8>> {
        self.inner.read_binary_file(path, context).await
    }

    async fn open_binary_reader(
        &self,
        path: &str,
        options: Option<crate::env::OpenBinaryOptions>,
        context: &Context,
    ) -> crate::env::Result<Arc<dyn crate::env::BinaryReader>> {
        let reader = self
            .inner
            .open_binary_reader(path, options, context)
            .await?;
        if let EnvKind::GrowingLog(log_path) = &self.kind {
            return Ok(Arc::new(GrowingReader {
                inner: reader,
                log_path: log_path.clone(),
            }));
        }
        Ok(reader)
    }

    async fn write_file(
        &self,
        path: &str,
        content: FileContent,
        context: &Context,
    ) -> crate::env::Result<()> {
        match &self.kind {
            EnvKind::BlockingWrite(state) => {
                if content == FileContent::Text("first\n".to_string()) {
                    state.first_write_started.notify_one();
                    state.finish_first_write.notified().await;
                } else if content == FileContent::Text("second\n".to_string()) {
                    state.second_write_started.store(true, Ordering::SeqCst);
                }
                self.inner.write_file(path, content, context).await
            }
            EnvKind::BlockingEdit(state) => {
                if content == FileContent::Text("ALPHA\nbeta\n".to_string()) {
                    state.first_write_started.notify_one();
                    state.finish_first_write.notified().await;
                    let result = self
                        .inner
                        .write_file(path, content, &BACKGROUND_CONTEXT)
                        .await;
                    state.first_edit_write_settled.store(true, Ordering::SeqCst);
                    return result;
                }
                if content == FileContent::Text("ALPHA\nBETA\n".to_string())
                    || content == FileContent::Text("alpha\nBETA\n".to_string())
                {
                    state.second_write_started.store(true, Ordering::SeqCst);
                }
                self.inner.write_file(path, content, context).await
            }
            EnvKind::OtherFs(state) => {
                if content == FileContent::Text("second\n".to_string()) {
                    state.second_write_started.store(true, Ordering::SeqCst);
                }
                self.inner.write_file(path, content, context).await
            }
            _ => self.inner.write_file(path, content, context).await,
        }
    }

    async fn append_file(
        &self,
        path: &str,
        content: FileContent,
        context: &Context,
    ) -> crate::env::Result<()> {
        self.inner.append_file(path, content, context).await
    }

    async fn truncate_file(
        &self,
        path: &str,
        size: u64,
        context: &Context,
    ) -> crate::env::Result<()> {
        self.inner.truncate_file(path, size, context).await
    }

    async fn flush_file(&self, path: &str, context: &Context) -> crate::env::Result<()> {
        self.inner.flush_file(path, context).await
    }

    async fn rename_file(
        &self,
        source: &str,
        destination: &str,
        context: &Context,
    ) -> crate::env::Result<()> {
        self.inner.rename_file(source, destination, context).await
    }

    async fn file_info(
        &self,
        path: &str,
        context: &Context,
    ) -> crate::env::Result<crate::env::FileInfo> {
        self.inner.file_info(path, context).await
    }

    async fn list_dir(
        &self,
        path: &str,
        context: &Context,
    ) -> crate::env::Result<Vec<crate::env::FileInfo>> {
        self.inner.list_dir(path, context).await
    }

    async fn open_dir_reader(
        &self,
        path: &str,
        context: &Context,
    ) -> crate::env::Result<Arc<dyn crate::env::DirReader>> {
        self.inner.open_dir_reader(path, context).await
    }

    async fn watch(
        &self,
        targets: &[crate::env::WatchTarget],
        on_change: crate::env::WatchCallback,
        context: &Context,
    ) -> crate::env::Result<Arc<dyn crate::env::FileWatcher>> {
        self.inner.watch(targets, on_change, context).await
    }

    async fn canonical_path(&self, path: &str, context: &Context) -> crate::env::Result<String> {
        self.inner.canonical_path(path, context).await
    }

    async fn exists(&self, path: &str, context: &Context) -> crate::env::Result<bool> {
        self.inner.exists(path, context).await
    }

    async fn create_dir(
        &self,
        path: &str,
        options: Option<crate::env::CreateDirOptions>,
        context: &Context,
    ) -> crate::env::Result<()> {
        self.inner.create_dir(path, options, context).await
    }

    async fn remove(
        &self,
        path: &str,
        options: Option<crate::env::RemoveOptions>,
        context: &Context,
    ) -> crate::env::Result<()> {
        self.inner.remove(path, options, context).await
    }

    async fn create_temp_dir(
        &self,
        prefix: Option<&str>,
        context: &Context,
    ) -> crate::env::Result<String> {
        self.inner.create_temp_dir(prefix, context).await
    }

    async fn create_temp_file(
        &self,
        options: Option<crate::env::TempFileOptions>,
        context: &Context,
    ) -> crate::env::Result<String> {
        self.inner.create_temp_file(options, context).await
    }

    async fn cleanup(&self, context: &Context) {
        <NodeExecutionEnv as FileSystem>::cleanup(&self.inner, context).await;
    }
}

#[async_trait::async_trait]
impl Shell for TestEnv {
    async fn exec(
        &self,
        command: ExecCommand,
        options: Option<ShellExecOptions>,
        context: &Context,
    ) -> Result<ShellExecResult, ExecutionError> {
        match &self.kind {
            EnvKind::TimeoutOutput => {
                let truncated_output_lines = DEFAULT_MAX_LINES + 1;
                let output = format!(
                    "{}\n",
                    (1..=truncated_output_lines)
                        .map(|index| format!("line-{index}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                let spill_path = get(self
                    .inner
                    .create_temp_file(
                        Some(crate::env::TempFileOptions {
                            prefix: Some("timeout-".to_string()),
                            suffix: Some(".log".to_string()),
                        }),
                        context,
                    )
                    .await);
                self.inner
                    .write_file(&spill_path, output.as_str().into(), context)
                    .await
                    .map_err(|error| {
                        ExecutionError::new(ExecutionErrorCode::Unknown, error.to_string())
                    })?;
                if let Some(options) = &options {
                    if let Some(on_output) = &options.on_output {
                        on_output(
                            &output,
                            &ShellOutputInfo::for_stream(crate::env::OutputStream::Stdout),
                        );
                    }
                }
                let mut error = ExecutionError::new(
                    ExecutionErrorCode::Timeout,
                    format!(
                        "timeout:{}",
                        options
                            .as_ref()
                            .and_then(|options| options.timeout)
                            .map(js_number)
                            .unwrap_or_default()
                    ),
                );
                error.set_spill_path(spill_path);
                Err(error)
            }
            EnvKind::Skipping(shared) => {
                *shared.received_window.lock() =
                    options.as_ref().and_then(|options| options.window);
                if let Some(options) = &options {
                    if let Some(on_output) = &options.on_output {
                        on_output(
                            "tail\n",
                            &ShellOutputInfo {
                                stream: crate::env::OutputStream::Stdout,
                                skipped: Some(shared.skip),
                            },
                        );
                    }
                }
                Ok(ShellExecResult {
                    exit_code: 0,
                    spill_path: None,
                })
            }
            EnvKind::Programs(shared) => {
                shared.commands.lock().push(command.clone());
                let program = match &command {
                    ExecCommand::Argv(argv) => argv.first().cloned().unwrap_or_default(),
                    ExecCommand::Shell(_) => String::new(),
                };
                if !shared.installed.contains(&program) {
                    return Err(ExecutionError::new(
                        ExecutionErrorCode::SpawnError,
                        format!("spawn {program} ENOENT"),
                    ));
                }
                if let Some(options) = &options {
                    if let Some(on_output) = &options.on_output {
                        on_output(
                            &shared.output,
                            &ShellOutputInfo::for_stream(crate::env::OutputStream::Stdout),
                        );
                    }
                }
                Ok(ShellExecResult {
                    exit_code: shared.exit_code,
                    spill_path: None,
                })
            }
            _ => self.inner.exec(command, options, context).await,
        }
    }

    async fn cleanup(&self, context: &Context) {
        <NodeExecutionEnv as Shell>::cleanup(&self.inner, context).await;
    }
}

/// A binary reader whose every `read` appends to the log first (upstream's overridden
/// `openBinaryReader` in the growing-log test).
struct GrowingReader {
    inner: Arc<dyn BinaryReader>,
    log_path: String,
}

#[async_trait::async_trait]
impl BinaryReader for GrowingReader {
    async fn info(&self, context: &Context) -> Result<FileInfo, FileError> {
        self.inner.info(context).await
    }

    async fn read(
        &self,
        offset: u64,
        length: usize,
        context: &Context,
    ) -> Result<Vec<u8>, FileError> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&self.log_path)
            .expect("open growing log");
        file.write_all(b"more\n").expect("append to growing log");
        self.inner.read(offset, length, context).await
    }

    async fn scan_lines(
        &self,
        options: ScanLinesOptions,
        context: &Context,
    ) -> Result<LineScan, FileError> {
        self.inner.scan_lines(options, context).await
    }

    async fn close(&self, context: &Context) {
        self.inner.close(context).await
    }
}

struct NoSnapshot;

impl std::fmt::Debug for NoSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NoSnapshot")
    }
}

impl RegistrySnapshot for NoSnapshot {
    fn installed(&self) -> Vec<Arc<crate::harness::types::Extension>> {
        Vec::new()
    }
    fn extension(&self, _name: &str) -> Option<Arc<crate::harness::types::Extension>> {
        None
    }
    fn tools(&self) -> Vec<(Arc<crate::harness::types::Extension>, Arc<ToolRegistration>)> {
        Vec::new()
    }
    fn sections(
        &self,
    ) -> Vec<(
        Arc<crate::harness::types::Extension>,
        crate::harness::types::PromptSection,
    )> {
        Vec::new()
    }
    fn tasks(&self) -> Vec<crate::harness::types::AnyTask> {
        Vec::new()
    }
    fn task(&self, _name: &str) -> Option<crate::harness::types::AnyTask> {
        None
    }
}

/// A minimal execution API: the environment, collected output and diagnostics, and nothing durable.
pub(crate) struct FakeApi {
    env: Option<Arc<dyn ExecutionEnv>>,
    window: Option<ShellOutputWindow>,
    output: Mutex<Vec<(String, Option<ShellOutputSkip>)>>,
    diagnostics: Mutex<Vec<ToolDiagnostic>>,
}

impl FakeApi {
    fn new(env: Option<Arc<dyn ExecutionEnv>>) -> Arc<Self> {
        Arc::new(Self {
            env,
            window: None,
            output: Mutex::new(Vec::new()),
            diagnostics: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait::async_trait]
impl ToolExecutionApi for FakeApi {
    fn task_id(&self) -> crate::types::TaskId {
        crate::types::TaskId(1)
    }
    fn conversation_id(&self) -> crate::types::ConversationId {
        crate::types::ConversationId(1)
    }
    fn call_id(&self) -> &str {
        "call"
    }
    fn registry(&self) -> Arc<dyn crate::harness::types::RegistrySnapshot> {
        Arc::new(NoSnapshot)
    }
    async fn snapshot(
        &self,
        _token: &crate::documents::DocToken,
        _owner: Option<u64>,
        _key: Option<&str>,
        _context: &Context,
    ) -> Result<Option<crate::types::JsonObject>, String> {
        Ok(None)
    }
    async fn snapshot_as_of(
        &self,
        _token: &crate::documents::DocToken,
        _conversation_id: u64,
        _key: Option<&str>,
        _at: crate::types::EntryId,
        _context: &Context,
    ) -> Result<Option<crate::types::JsonObject>, String> {
        Ok(None)
    }
    async fn watch_doc(
        &self,
        _token: &crate::documents::DocToken,
        _owner: Option<u64>,
        _key: Option<&str>,
        _context: &Context,
    ) -> Result<Option<crate::tasks::DocumentWatch>, String> {
        Ok(None)
    }
    async fn agent(&self, _context: &Context) -> Result<crate::harness::types::Agent, String> {
        Err("fake api has no agent".to_string())
    }
    fn models(&self) -> Arc<dyn pi_ai::Models> {
        pi_ai::create_models(None)
    }
    fn env(&self) -> Option<Arc<dyn ExecutionEnv>> {
        self.env.clone()
    }
    fn output(&self, chunk: &[u8], skipped: Option<ShellOutputSkip>) {
        self.output
            .lock()
            .push((String::from_utf8_lossy(chunk).into_owned(), skipped));
    }
    fn output_window(&self) -> Option<ShellOutputWindow> {
        self.window
    }
    fn diagnostic(&self, diagnostic: ToolDiagnostic) {
        self.diagnostics.lock().push(diagnostic);
    }
    async fn details(&self, _value: Value, _context: &Context) -> Result<(), String> {
        Ok(())
    }
    async fn commit(
        &self,
        _change: &mut crate::session::CommitValueCallback<'_>,
        _context: &Context,
    ) -> Result<Value, String> {
        Err("fake api cannot commit".to_string())
    }
    async fn memo(&self, _name: &str, _context: &Context) -> Result<Option<Value>, String> {
        Ok(None)
    }
    async fn memo_put(
        &self,
        _name: &str,
        candidate: Value,
        _context: &Context,
    ) -> Result<Value, String> {
        Ok(candidate)
    }
    async fn create_task(
        &self,
        _task: &crate::harness::types::AnyTask,
        _input: Value,
        _options: crate::types::TaskOptions,
        _context: &Context,
    ) -> Result<crate::types::TaskId, String> {
        Err("fake api cannot create tasks".to_string())
    }
    async fn get_task(
        &self,
        _id: crate::types::TaskId,
        _context: &Context,
    ) -> Result<Option<crate::types::TaskRecord>, String> {
        Ok(None)
    }
    async fn wait_for_task(
        &self,
        _id: crate::types::TaskId,
        _context: &Context,
    ) -> Result<crate::harness::types::SettledTask, String> {
        Err("fake api cannot wait for tasks".to_string())
    }
    async fn conversation(
        &self,
        _id: crate::types::ConversationId,
        _context: &Context,
    ) -> Result<Option<Arc<dyn crate::session::ConversationHandle>>, String> {
        Ok(None)
    }
}

/// Runs a tool against a caller-built `FakeApi` (upstream's custom-`api` test variant).
async fn run_api(
    tool: &ToolRegistration,
    args: Value,
    api: &Arc<FakeApi>,
    context: Context,
) -> (
    Vec<(String, Option<ShellOutputSkip>)>,
    ToolExecutionResult,
    Vec<ToolDiagnostic>,
) {
    let result = (tool.execute)(args, Arc::clone(api) as Arc<dyn ToolExecutionApi>, context).await;
    (
        api.output.lock().clone(),
        result,
        api.diagnostics.lock().clone(),
    )
}

fn parts_text(output: &[(String, Option<ShellOutputSkip>)]) -> String {
    output.iter().map(|(text, _)| text.as_str()).collect()
}

async fn run(
    tool: &ToolRegistration,
    args: Value,
    env: Option<Arc<dyn ExecutionEnv>>,
    context: Context,
) -> (
    Vec<(String, Option<ShellOutputSkip>)>,
    ToolExecutionResult,
    Vec<ToolDiagnostic>,
) {
    let api = FakeApi::new(env);
    let result = (tool.execute)(args, api.clone(), context).await;
    let output = api.output.lock().clone();
    let reported = api.diagnostics.lock().clone();
    (output, result, reported)
}

/// The error message of a failed run: upstream's thrown `error.message` (module D1).
fn error_message(result: &ToolExecutionResult) -> String {
    result
        .diagnostics
        .iter()
        .flatten()
        .find(|diagnostic| diagnostic.code.as_deref() == Some("tool_error"))
        .map(|diagnostic| diagnostic.message.clone())
        .unwrap_or_default()
}

fn text_output(result: &ToolExecutionResult) -> String {
    match &result.content {
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// The diagnostics of a tool result (upstream `result.diagnostics`).
fn diagnostic_text(result: &ToolExecutionResult) -> String {
    result
        .diagnostics
        .iter()
        .flatten()
        .map(|diagnostic| diagnostic.message.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn delay(ms: u64) -> impl Future<Output = ()> {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(ms);
    async move { tokio::time::sleep_until(deadline).await }
}

fn js_number(value: f64) -> String {
    format!("{value}")
}

fn details_of(result: &ToolExecutionResult) -> &Value {
    result.details.as_ref().expect("details set")
}

// -- tests -------------------------------------------------------------------

const TRUNCATED_OUTPUT_LINES: usize = DEFAULT_MAX_LINES + 1;

/// A 1×1 PNG (upstream's base64 fixture, decoded).
const PNG_BYTES: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0,
    0, 0, 31, 21, 196, 137, 0, 0, 0, 13, 73, 68, 65, 84, 120, 156, 99, 96, 96, 96, 248, 15, 0, 1,
    4, 1, 0, 95, 229, 195, 75, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

#[tokio::test(flavor = "multi_thread")]
async fn tools_fail_with_an_ordinary_error_when_no_environment_is_configured() {
    let (output, result, reported) = run(
        &create_read_tool(),
        json!({ "path": "x" }),
        None,
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(result.is_error.unwrap_or(false));
    assert_eq!(
        error_message(&result),
        "No execution environment is configured"
    );
    assert!(output.is_empty());
    assert_eq!(result.diagnostics.as_ref().map(Vec::len), Some(1));
}

#[test]
fn read_detects_the_complete_gif_signatures() {
    for signature in ["GIF87a", "GIF89a"] {
        assert_eq!(
            detect_supported_image_mime_type(signature.as_bytes()),
            Some("image/gif")
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn read_reads_text_with_offsets_and_limits_and_reports_continuation_as_a_diagnostic() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    write(
        env.as_ref(),
        "test.txt",
        (1..=100)
            .map(|index| format!("Line {index}"))
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .await;
    let (_, result, reported) = run(
        &create_read_tool(),
        json!({ "path": "test.txt", "offset": 41, "limit": 20 }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(!result.is_error.unwrap_or(false));
    let output_text = text_output(&result);
    assert!(!output_text.contains("Line 40"));
    assert!(output_text.contains("Line 41"));
    assert!(output_text.contains("Line 60"));
    assert!(!output_text.contains("Line 61"));
    assert!(!output_text.contains("more lines"));
    assert_eq!(
        diagnostic_text(&result),
        "40 more lines in file. Use offset=61 to continue."
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn read_truncates_large_text_by_line_count() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    write(
        env.as_ref(),
        "large.txt",
        (1..=2500)
            .map(|index| format!("Line {index}"))
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .await;
    let (_, result, reported) = run(
        &create_read_tool(),
        json!({ "path": "large.txt" }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert_eq!(
        diagnostic_text(&result),
        "Showing lines 1-2000 of 2500. Use offset=2001 to continue."
    );
    assert_eq!(
        result.diagnostics.as_ref().unwrap()[0].code.as_deref(),
        Some("truncated")
    );
    let truncation = &details_of(&result)["truncation"];
    assert_eq!(truncation["truncated"], json!(true));
    assert_eq!(truncation["truncatedBy"], json!("lines"));
    assert_eq!(truncation["totalLines"], json!(2500));
    assert_eq!(truncation["outputLines"], json!(2000));
}

#[tokio::test(flavor = "multi_thread")]
async fn read_does_not_count_a_trailing_newline_as_an_extra_line_at_the_truncation_limit() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    write(
        env.as_ref(),
        "exact.txt",
        format!(
            "{}\n",
            (0..2000).map(|_| "x").collect::<Vec<_>>().join("\n")
        ),
    )
    .await;
    let (_, result, reported) = run(
        &create_read_tool(),
        json!({ "path": "exact.txt" }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(result.details.is_none());
    assert!(reported.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn read_shows_the_start_of_a_line_longer_than_the_byte_limit() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    write(
        env.as_ref(),
        "long.txt",
        format!("{}\nnext\n", "é".repeat(40_000)),
    )
    .await;
    let (_, result, reported) = run(
        &create_read_tool(),
        json!({ "path": "long.txt" }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    // Two-byte characters: the cut lands on a character boundary at or below the limit.
    assert_eq!(text_output(&result), "é".repeat(25_600));
    assert_eq!(
		diagnostic_text(&result),
		"Line 1 is 78.1KB, exceeds the 50.0KB limit; showing its first 50.0KB. Use bash: sed -n '1p' long.txt | tail -c +51201"
	);
    let truncation = &details_of(&result)["truncation"];
    assert_eq!(truncation["truncated"], json!(true));
    assert_eq!(truncation["firstLineExceedsLimit"], json!(true));
    assert_eq!(truncation["outputBytes"], json!(51_200));
    assert_eq!(truncation["outputLines"], json!(1));
    assert!(truncation.get("content").is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn read_rejects_offsets_beyond_the_file() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    write(env.as_ref(), "short.txt", "one\ntwo\nthree").await;
    let (_, result, _) = run(
        &create_read_tool(),
        json!({ "path": "short.txt", "offset": 100 }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert_eq!(
        error_message(&result),
        "Offset 100 is beyond end of file (3 lines total)"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn read_reports_images_by_content_as_unsupported() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    write(env.as_ref(), "image.txt", PNG_BYTES.to_vec()).await;
    let (_, result, reported) = run(
        &create_read_tool(),
        json!({ "path": "image.txt" }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert_eq!(result.content, Some(json!([])));
    assert_eq!(result.is_error, Some(true));
    assert_eq!(
        diagnostic_text(&result),
        "image.txt is an image (image/png); reading images is not supported"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn write_writes_files_and_creates_parent_directories() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    let (_, result, _) = run(
        &create_write_tool(),
        json!({ "path": "nested/dir/file.txt", "content": "hello" }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert_eq!(
        text_output(&result),
        "Successfully wrote to nested/dir/file.txt"
    );
    assert_eq!(
        read_text(env.as_ref(), "nested/dir/file.txt").await,
        "hello"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn write_keeps_the_mutation_queue_locked_until_an_aborted_write_settles() {
    let state = BlockingState::new();
    let env: Arc<TestEnv> = Arc::new(TestEnv {
        inner: node_env(create_temp_dir()),
        kind: EnvKind::BlockingWrite(Arc::clone(&state)),
    });
    let tool = create_write_tool();
    let controller = AbortController::new();
    let spawn_tool = tool.clone();
    let abort_signal = controller.signal();
    let spawn_env = env.clone();
    let first_write = tokio::spawn(async move {
        run(
            &spawn_tool,
            json!({ "path": "file.txt", "content": "first\n" }),
            some_env(spawn_env),
            with_abort_signal(abort_signal, BACKGROUND_CONTEXT.clone()),
        )
        .await
    });
    state.first_write_started.notified().await;
    controller.abort_default();
    let spawn_tool = tool.clone();
    let spawn_env = env.clone();
    let second_write = tokio::spawn(async move {
        run(
            &spawn_tool,
            json!({ "path": "file.txt", "content": "second\n" }),
            some_env(spawn_env),
            BACKGROUND_CONTEXT.clone(),
        )
        .await
    });
    delay(20).await;
    assert!(!state.second_write_started.load(Ordering::SeqCst));
    state.finish_first_write.notify_one();
    let (_, first_result, _) = first_write.await.expect("joined");
    assert!(first_result.is_error.unwrap_or(false));
    let (_, second_result, _) = second_write.await.expect("joined");
    assert!(!second_result.is_error.unwrap_or(false));
    assert_eq!(read_text(env.as_ref(), "file.txt").await, "second\n");
}

#[tokio::test(flavor = "multi_thread")]
async fn edit_applies_disjoint_edits_and_returns_both_diff_formats() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    let original = "alpha\nbeta\ngamma\ndelta\n";
    write(env.as_ref(), "edit.txt", original).await;
    let (_, result, _) = run(
        &create_edit_tool(),
        json!({
            "path": "edit.txt",
            "edits": [
                { "oldText": "alpha\n", "newText": "ALPHA\n" },
                { "oldText": "gamma\n", "newText": "GAMMA\n" },
            ],
        }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    let details = details_of(&result);
    assert_eq!(
        text_output(&result),
        "Successfully replaced 2 block(s) in edit.txt."
    );
    assert!(details["diff"].as_str().expect("diff").contains("ALPHA"));
    assert!(details["diff"].as_str().expect("diff").contains("GAMMA"));
    assert_eq!(
        crate::tools::edit_diff::tests::apply_patch(
            original,
            details["patch"].as_str().expect("patch")
        ),
        "ALPHA\nbeta\nGAMMA\ndelta\n"
    );
    assert_eq!(
        read_text(env.as_ref(), "edit.txt").await,
        "ALPHA\nbeta\nGAMMA\ndelta\n"
    );
}

#[test]
fn edit_repairs_edits_sent_as_various_shapes_without_mutating_them() {
    let prepare = prepare_edit_arguments;
    let edit = json!({ "oldText": "a", "newText": "b" });
    let as_string = json!({ "path": "f", "edits": json!([edit]).to_string() });
    let prepared = prepare(&as_string);
    assert_eq!(prepared, json!({ "path": "f", "edits": [edit] }));
    // The call's arguments stay unchanged.
    assert_eq!(as_string["edits"], json!([edit]).to_string());
    assert_eq!(
        prepare(&json!({ "path": "f", "edits": edit.to_string() })),
        json!({ "path": "f", "edits": [edit] })
    );
    assert_eq!(
        prepare(&json!({ "path": "f", "edits": edit })),
        json!({ "path": "f", "edits": [edit] })
    );
    assert_eq!(
        prepare(&json!({ "path": "f", "edits": [edit], "oldText": "c", "newText": "d" })),
        json!({ "path": "f", "edits": [edit, { "oldText": "c", "newText": "d" }] })
    );
    assert_eq!(
        prepare(&json!({ "path": "f", "edits": "not json" })),
        json!({ "path": "f", "edits": "not json" })
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn edit_matches_all_edits_against_the_original_and_rejects_overlaps() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    write(env.as_ref(), "edit.txt", "one\ntwo\nthree\n").await;
    let (_, result, _) = run(
        &create_edit_tool(),
        json!({
            "path": "edit.txt",
            "edits": [
                { "oldText": "one\ntwo\n", "newText": "ONE\nTWO\n" },
                { "oldText": "two\nthree\n", "newText": "TWO\nTHREE\n" },
            ],
        }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(error_message(&result).contains("overlap"));
    assert_eq!(
        read_text(env.as_ref(), "edit.txt").await,
        "one\ntwo\nthree\n"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn edit_rejects_missing_and_duplicate_target_text() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    write(env.as_ref(), "edit.txt", "foo foo foo").await;
    let (_, missing, _) = run(
        &create_edit_tool(),
        json!({ "path": "edit.txt", "edits": [{ "oldText": "bar", "newText": "baz" }] }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(error_message(&missing).contains("Could not find the exact text"));
    let (_, duplicate, _) = run(
        &create_edit_tool(),
        json!({ "path": "edit.txt", "edits": [{ "oldText": "foo", "newText": "bar" }] }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(error_message(&duplicate).contains("Found 3 occurrences"));
}

#[tokio::test(flavor = "multi_thread")]
async fn edit_keeps_the_mutation_queue_locked_until_an_aborted_edit_write_settles() {
    let state = BlockingState::new();
    let env: Arc<TestEnv> = Arc::new(TestEnv {
        inner: node_env(create_temp_dir()),
        kind: EnvKind::BlockingEdit(Arc::clone(&state)),
    });
    write(env.as_ref(), "file.txt", "alpha\nbeta\n").await;
    let tool = create_edit_tool();
    let controller = AbortController::new();
    let spawn_tool = tool.clone();
    let abort_signal = controller.signal();
    let spawn_env = env.clone();
    let first_edit = tokio::spawn(async move {
        run(
            &spawn_tool,
            json!({ "path": "file.txt", "edits": [{ "oldText": "alpha", "newText": "ALPHA" }] }),
            some_env(spawn_env),
            with_abort_signal(abort_signal, BACKGROUND_CONTEXT.clone()),
        )
        .await
    });
    state.first_write_started.notified().await;
    controller.abort_default();
    let spawn_tool = tool.clone();
    let spawn_env = env.clone();
    let second_edit = tokio::spawn(async move {
        run(
            &spawn_tool,
            json!({ "path": "file.txt", "edits": [{ "oldText": "beta", "newText": "BETA" }] }),
            some_env(spawn_env),
            BACKGROUND_CONTEXT.clone(),
        )
        .await
    });
    delay(20).await;
    assert!(!state.second_write_started.load(Ordering::SeqCst));
    state.finish_first_write.notify_one();
    let (_, first_result, _) = first_edit.await.expect("joined");
    assert_eq!(error_message(&first_result), "Operation aborted");
    let (_, second_result, _) = second_edit.await.expect("joined");
    assert!(!second_result.is_error.unwrap_or(false));
    assert!(state.first_edit_write_settled.load(Ordering::SeqCst));
    assert_eq!(read_text(env.as_ref(), "file.txt").await, "ALPHA\nBETA\n");
}

#[tokio::test(flavor = "multi_thread")]
async fn edit_serializes_concurrent_edits_through_canonical_and_symlink_paths() {
    let env: Arc<TestEnv> = Arc::new(TestEnv {
        inner: node_env(create_temp_dir()),
        kind: EnvKind::SlowRead,
    });
    write(env.as_ref(), "target.txt", "alpha\nbeta\ngamma\n").await;
    std::os::unix::fs::symlink("target.txt", format!("{}/link.txt", env.cwd())).expect("symlink");
    let tool = create_edit_tool();
    let (first, second) = tokio::join!(
        run(
            &tool,
            json!({ "path": "target.txt", "edits": [{ "oldText": "alpha", "newText": "ALPHA" }] }),
            some_env(env.clone()),
            BACKGROUND_CONTEXT.clone()
        ),
        run(
            &tool,
            json!({ "path": "link.txt", "edits": [{ "oldText": "beta", "newText": "BETA" }] }),
            some_env(env.clone()),
            BACKGROUND_CONTEXT.clone()
        ),
    );
    assert!(
        !first.1.is_error.unwrap_or(false),
        "{}",
        error_message(&first.1)
    );
    assert!(
        !second.1.is_error.unwrap_or(false),
        "{}",
        error_message(&second.1)
    );
    assert_eq!(
        read_text(env.as_ref(), "target.txt").await,
        "ALPHA\nBETA\ngamma\n"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn edit_serializes_edits_of_one_file_across_environment_objects_of_one_file_system() {
    let dir = create_temp_dir();
    let first: Arc<TestEnv> = Arc::new(TestEnv {
        inner: node_env(dir.clone()),
        kind: EnvKind::SlowRead,
    });
    let second: Arc<TestEnv> = Arc::new(TestEnv {
        inner: node_env(dir),
        kind: EnvKind::SlowRead,
    });
    write(first.as_ref(), "file.txt", "alpha\nbeta\n").await;
    let tool = create_edit_tool();
    let (first_run, second_run) = tokio::join!(
        run(
            &tool,
            json!({ "path": "file.txt", "edits": [{ "oldText": "alpha", "newText": "ALPHA" }] }),
            some_env(first.clone()),
            BACKGROUND_CONTEXT.clone()
        ),
        run(
            &tool,
            json!({ "path": "file.txt", "edits": [{ "oldText": "beta", "newText": "BETA" }] }),
            some_env(second.clone()),
            BACKGROUND_CONTEXT.clone()
        ),
    );
    assert!(
        !first_run.1.is_error.unwrap_or(false),
        "{}",
        error_message(&first_run.1)
    );
    assert!(
        !second_run.1.is_error.unwrap_or(false),
        "{}",
        error_message(&second_run.1)
    );
    assert_eq!(read_text(first.as_ref(), "file.txt").await, "ALPHA\nBETA\n");
}

#[tokio::test(flavor = "multi_thread")]
async fn edit_serializes_a_new_file_created_through_a_symlinked_directory_with_its_canonical_path()
{
    let state = BlockingState::new();
    let env: Arc<TestEnv> = Arc::new(TestEnv {
        inner: node_env(create_temp_dir()),
        kind: EnvKind::BlockingWrite(Arc::clone(&state)),
    });
    std::fs::create_dir(format!("{}/real", env.cwd())).expect("real dir");
    std::os::unix::fs::symlink(format!("{}/real", env.cwd()), format!("{}/link", env.cwd()))
        .expect("symlink");
    let tool = create_write_tool();
    let spawn_tool = tool.clone();
    let spawn_env = env.clone();
    let first = tokio::spawn(async move {
        run(
            &spawn_tool,
            json!({ "path": "link/new.txt", "content": "first\n" }),
            some_env(spawn_env),
            BACKGROUND_CONTEXT.clone(),
        )
        .await
    });
    state.first_write_started.notified().await;
    let spawn_tool = tool.clone();
    let spawn_env = env.clone();
    let second = tokio::spawn(async move {
        run(
            &spawn_tool,
            json!({ "path": "real/new.txt", "content": "second\n" }),
            some_env(spawn_env),
            BACKGROUND_CONTEXT.clone(),
        )
        .await
    });
    delay(20).await;
    assert!(!state.second_write_started.load(Ordering::SeqCst));
    state.finish_first_write.notify_one();
    let (_, first_result, _) = first.await.expect("joined");
    assert!(
        !first_result.is_error.unwrap_or(false),
        "{}",
        error_message(&first_result)
    );
    let (_, second_result, _) = second.await.expect("joined");
    assert!(
        !second_result.is_error.unwrap_or(false),
        "{}",
        error_message(&second_result)
    );
    assert_eq!(read_text(env.as_ref(), "real/new.txt").await, "second\n");
}

#[tokio::test(flavor = "multi_thread")]
async fn mutation_queue_keys_a_missing_file_whose_name_contains_a_backslash_like_the_created_file()
{
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    let created = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let first = {
        let created = Arc::clone(&created);
        let release = Arc::clone(&release);
        let env = env.clone();
        tokio::spawn(async move {
            let env_for_mutation = env.clone();
            with_file_mutation_queue(
                env_for_mutation,
                "a\\b.txt",
                async move {
                    write(env.as_ref(), "a\\b.txt", "first\n").await;
                    created.notify_one();
                    release.notified().await;
                },
                &BACKGROUND_CONTEXT,
            )
            .await;
        })
    };
    created.notified().await;
    let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let second = {
        let entered = Arc::clone(&entered);
        let env = env.clone();
        tokio::spawn(async move {
            let env_for_mutation = env.clone();
            with_file_mutation_queue(
                env_for_mutation,
                "a\\b.txt",
                async move {
                    entered.store(true, Ordering::SeqCst);
                },
                &BACKGROUND_CONTEXT,
            )
            .await;
        })
    };
    delay(20).await;
    assert!(!entered.load(Ordering::SeqCst));
    release.notify_one();
    let (first, second) = tokio::join!(first, second);
    first.expect("joined");
    second.expect("joined");
    assert!(entered.load(Ordering::SeqCst));
}

#[tokio::test(flavor = "multi_thread")]
async fn mutation_queue_does_not_serialize_the_same_path_on_different_file_systems() {
    let state = BlockingState::new();
    let dir = create_temp_dir();
    let local: Arc<TestEnv> = Arc::new(TestEnv {
        inner: node_env(dir.clone()),
        kind: EnvKind::BlockingWrite(Arc::clone(&state)),
    });
    let other: Arc<TestEnv> = Arc::new(TestEnv {
        inner: node_env(dir),
        kind: EnvKind::OtherFs(Arc::clone(&state)),
    });
    let tool = create_write_tool();
    let spawn_tool = tool.clone();
    let blocked = tokio::spawn(async move {
        run(
            &spawn_tool,
            json!({ "path": "file.txt", "content": "first\n" }),
            some_env(local.clone()),
            BACKGROUND_CONTEXT.clone(),
        )
        .await
    });
    state.first_write_started.notified().await;
    let (_, other_result, _) = run(
        &tool,
        json!({ "path": "file.txt", "content": "second\n" }),
        some_env(other.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(!other_result.is_error.unwrap_or(false));
    assert!(state.second_write_started.load(Ordering::SeqCst));
    state.finish_first_write.notify_one();
    let (_, blocked_result, _) = blocked.await.expect("joined");
    assert!(
        !blocked_result.is_error.unwrap_or(false),
        "{}",
        error_message(&blocked_result)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn edit_edits_regular_files_through_symlinks() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    write(env.as_ref(), "target.txt", "before\n").await;
    std::os::unix::fs::symlink("target.txt", format!("{}/link.txt", env.cwd())).expect("symlink");
    let (_, result, _) = run(
        &create_edit_tool(),
        json!({ "path": "link.txt", "edits": [{ "oldText": "before", "newText": "after" }] }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(
        !result.is_error.unwrap_or(false),
        "{}",
        error_message(&result)
    );
    // Read through the inner environment, not the symlink.
    let (_, read_result, _) = run(
        &create_read_tool(),
        json!({ "path": "target.txt" }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert_eq!(text_output(&read_result), "after\n");
}

#[tokio::test(flavor = "multi_thread")]
async fn edit_preserves_bom_and_crlf_line_endings() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    write(env.as_ref(), "edit.txt", "\u{FEFF}one\r\ntwo\r\n").await;
    let (_, result, _) = run(
        &create_edit_tool(),
        json!({ "path": "edit.txt", "edits": [{ "oldText": "two", "newText": "TWO" }] }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(
        !result.is_error.unwrap_or(false),
        "{}",
        error_message(&result)
    );
    assert_eq!(
        read_text(env.as_ref(), "edit.txt").await,
        "\u{FEFF}one\r\nTWO\r\n"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn bash_passes_the_retained_window_to_the_environment_and_forwards_what_it_skipped() {
    let window = ShellOutputWindow {
        max_bytes: 4,
        max_lines: 1,
        min_interval_ms: 100,
        bytes_per_second: 1024.0,
    };
    let skip = ShellOutputSkip {
        bytes: 6,
        newlines: 2,
        ends_with_newline: true,
    };
    let shared = Arc::new(SkipShared {
        skip,
        received_window: Mutex::new(None),
    });
    let env = Arc::new(TestEnv {
        inner: node_env(create_temp_dir()),
        kind: EnvKind::Skipping(Arc::clone(&shared)),
    });
    let api = Arc::new(FakeApi {
        env: Some(env),
        window: Some(window),
        output: Mutex::new(Vec::new()),
        diagnostics: Mutex::new(Vec::new()),
    });
    let (calls, result, _) = run_api(
        &create_bash_tool(None),
        json!({ "command": "anything" }),
        &api,
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(
        !result.is_error.unwrap_or(false),
        "{}",
        error_message(&result)
    );
    assert_eq!(*shared.received_window.lock(), Some(window));
    assert_eq!(calls, vec![("tail\n".to_string(), Some(skip))]);
}

#[tokio::test(flavor = "multi_thread")]
async fn bash_streams_combined_stdout_and_stderr_and_returns_no_content_of_its_own() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    let (output, result, _) = run(
        &create_bash_tool(None),
        json!({ "command": "printf out; printf err >&2" }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    let combined: String = output.iter().map(|(text, _)| text.as_str()).collect();
    assert!(combined.contains("out"));
    assert!(combined.contains("err"));
    assert!(result.content.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn bash_throws_on_nonzero_exits_and_timeouts_after_streaming_the_output() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    let tool = create_bash_tool(None);
    let (output, failed, _) = run(
        &tool,
        json!({ "command": "printf failed; exit 7" }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert_eq!(error_message(&failed), "Command exited with code 7");
    let combined: String = output.iter().map(|(text, _)| text.as_str()).collect();
    assert_eq!(combined, "failed");
    let (_, slow, _) = run(
        &tool,
        json!({ "command": "sleep 2", "timeout": 0.01 }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert_eq!(error_message(&slow), "Command timed out after 0.01 seconds");
}

#[tokio::test(flavor = "multi_thread")]
async fn bash_reports_the_spill_of_a_command_that_times_out() {
    let env = Arc::new(TestEnv {
        inner: node_env(create_temp_dir()),
        kind: EnvKind::TimeoutOutput,
    });
    let (_, failed, reported) = run(
        &create_bash_tool(None),
        json!({ "command": "emit-output-then-time-out", "timeout": 0.05 }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert_eq!(
        error_message(&failed),
        "Command timed out after 0.05 seconds"
    );
    let full_output_message = &reported[0].message;
    let full_output_path = full_output_message
        .strip_prefix("Full output: ")
        .expect("spill diagnostic names the spill file");
    let full_output = read_text(env.as_ref(), full_output_path).await;
    assert!(full_output.contains("line-1\nline-2"));
    assert!(full_output.contains(&format!(
        "line-{DEFAULT_MAX_LINES}\nline-{TRUNCATED_OUTPUT_LINES}"
    )));
}

#[tokio::test(flavor = "multi_thread")]
async fn bash_prepares_command_cwd_and_an_explicit_environment_with_the_calls_api() {
    let received_env: Arc<Mutex<Option<Arc<dyn ExecutionEnv>>>> = Arc::new(Mutex::new(None));
    let received_signal: Arc<Mutex<Option<pi_chord::context::AbortSignal>>> =
        Arc::new(Mutex::new(None));
    let env = Arc::new(node_env_with_shell_env(
        create_temp_dir(),
        vec![("PI_BASH_PREPARE_INHERITED", "inherited")],
    ));
    get(env.create_dir("workspace", None, &BACKGROUND_CONTEXT).await);
    let workspace = format!("{}/workspace", env.cwd());
    let controller = AbortController::new();
    let captured_env = Arc::clone(&received_env);
    let captured_signal = Arc::clone(&received_signal);
    let prepare_workspace = workspace.clone();
    let tool = create_bash_tool(Some(BashToolOptions {
        command_prefix: Some("prefix=ready".to_string()),
        prepare: Some(Arc::new(
            move |execution: &mut BashExecution,
                  api: Arc<dyn ToolExecutionApi>,
                  call_context: Context| {
                let captured_env = Arc::clone(&captured_env);
                let captured_signal = Arc::clone(&captured_signal);
                *captured_env.lock() = api.env();
                *captured_signal.lock() = call_context.abort_signal();
                execution.cwd = prepare_workspace.clone();
                execution.env.insert(
                    "PI_BASH_PREPARE_EXPLICIT".to_string(),
                    "explicit".to_string(),
                );
                execution.inherit_env = false;
                execution.command += &format!(
				"\n: > prepared-cwd\nprintf '%s:%s:%s' \"$prefix\" \"${{PI_BASH_PREPARE_INHERITED-}}\" \"$PI_BASH_PREPARE_EXPLICIT\"\nprintf ':%s' \"$PWD\""
			);
                Box::pin(async {}) as super::bash::FutureBox
            },
        )),
    }));
    let (output, result, _) = run(
        &tool,
        json!({ "command": ":" }),
        Some(env.clone() as Arc<dyn ExecutionEnv>),
        with_abort_signal(controller.signal(), BACKGROUND_CONTEXT.clone()),
    )
    .await;
    assert!(
        !result.is_error.unwrap_or(false),
        "{}",
        error_message(&result)
    );
    assert!(received_env.lock().is_some());
    assert!(!received_signal
        .lock()
        .as_ref()
        .expect("signal captured")
        .aborted());
    let pwd = get(env.canonical_path(&workspace, &BACKGROUND_CONTEXT).await);
    assert_eq!(parts_text(&output), format!("ready::explicit:{pwd}"));
    assert!(get(env
        .exists(&format!("{workspace}/prepared-cwd"), &BACKGROUND_CONTEXT)
        .await));
}

#[tokio::test(flavor = "multi_thread")]
async fn bash_supports_command_prefixes() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    let (output, result, _) = run(
        &create_bash_tool(Some(BashToolOptions {
            command_prefix: Some("value=hello".to_string()),
            prepare: None,
        })),
        json!({ "command": "printf $value" }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(
        !result.is_error.unwrap_or(false),
        "{}",
        error_message(&result)
    );
    assert_eq!(
        output
            .iter()
            .map(|(text, _)| text.as_str())
            .collect::<String>(),
        "hello"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn bash_streams_every_byte_and_spills_complete_output_beyond_the_default_limits() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    let (output, result, reported) = run(
        &create_bash_tool(None),
        json!({ "command": "i=1; while [ $i -le 3000 ]; do echo line-$i; i=$((i + 1)); done" }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(
        !result.is_error.unwrap_or(false),
        "{}",
        error_message(&result)
    );
    let expected: String = (1..=3000).map(|index| format!("line-{index}\n")).collect();
    assert_eq!(parts_text(&output), expected);
    let full_output_path = reported[0]
        .message
        .strip_prefix("Full output: ")
        .expect("spill diagnostic names the spill file");
    assert_eq!(read_text(env.as_ref(), full_output_path).await, expected);
}

#[tokio::test(flavor = "multi_thread")]
async fn bash_does_not_spill_output_within_the_limits() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    let (_, result, reported) = run(
        &create_bash_tool(None),
        json!({ "command": "printf small" }),
        some_env(env.clone()),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(!result.is_error.unwrap_or(false));
    assert!(reported.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn read_reads_a_log_that_grows_while_it_is_read() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    write(env.as_ref(), "app.log", "one\ntwo\n").await;
    let log_path = format!("{}/app.log", env.cwd());
    // Every read appends, as a busy writer would; the scanned lines stay as they were.
    let growing = Arc::new(TestEnv {
        inner: node_env(env.cwd().to_string()),
        kind: EnvKind::GrowingLog(log_path),
    });
    let (output, result, _) = run(
        &create_read_tool(),
        json!({ "path": "app.log", "limit": 2 }),
        some_env(growing),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(
        !result.is_error.unwrap_or(false),
        "{}",
        error_message(&result)
    );
    assert_eq!(text_output(&result), "one\ntwo");
    assert_eq!(output.len(), 0);
}

/// An environment where only the listed programs exist; it records each command and prints `output`.
fn programs_env(
    installed: &[&str],
    output: &str,
    exit_code: i32,
    cwd: String,
) -> (Arc<TestEnv>, Arc<ProgramsShared>) {
    let shared = Arc::new(ProgramsShared {
        installed: installed
            .iter()
            .map(|program| program.to_string())
            .collect(),
        output: output.to_string(),
        exit_code,
        commands: Mutex::new(Vec::new()),
    });
    let env = Arc::new(TestEnv {
        inner: node_env(cwd),
        kind: EnvKind::Programs(Arc::clone(&shared)),
    });
    (env, shared)
}

#[tokio::test(flavor = "multi_thread")]
async fn powershell_runs_the_command_with_pwsh_as_one_argument_forcing_utf8_output() {
    let (env, shared) = programs_env(&["pwsh"], "héllo\n", 0, create_temp_dir());
    let (output, result, _) = run(
        &create_power_shell_tool(None),
        json!({ "command": "Write-Output 'héllo'" }),
        some_env(env),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(
        !result.is_error.unwrap_or(false),
        "{}",
        error_message(&result)
    );
    assert_eq!(parts_text(&output), "héllo\n");
    assert_eq!(
        *shared.commands.lock(),
        vec![ExecCommand::Argv(vec![
			"pwsh".to_string(),
			"-NoProfile".to_string(),
			"-NonInteractive".to_string(),
			"-ExecutionPolicy".to_string(),
			"Bypass".to_string(),
			"-Command".to_string(),
			"try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\nWrite-Output 'héllo'"
				.to_string(),
		])]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn powershell_falls_back_to_windows_powershell_and_reports_the_last_start_failure() {
    let (env, shared) = programs_env(&["powershell"], "ok", 0, create_temp_dir());
    let (output, result, _) = run(
        &create_power_shell_tool(Some(PowerShellToolOptions {
            command_prefix: Some("$x = 1".to_string()),
            prepare: None,
            programs: None,
        })),
        json!({ "command": "$x" }),
        some_env(env),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(
        !result.is_error.unwrap_or(false),
        "{}",
        error_message(&result)
    );
    assert_eq!(parts_text(&output), "ok");
    let firsts: Vec<String> = shared
        .commands
        .lock()
        .iter()
        .map(|command| match command {
            ExecCommand::Argv(argv) => argv.first().cloned().unwrap_or_default(),
            ExecCommand::Shell(shell) => shell.clone(),
        })
        .collect();
    assert_eq!(firsts, vec!["pwsh".to_string(), "powershell".to_string()]);
    let commands = shared.commands.lock();
    let script = match &commands[1] {
        ExecCommand::Argv(argv) => argv.last().expect("script argument").clone(),
        ExecCommand::Shell(_) => panic!("powershell runs argv, not a shell string"),
    };
    let mut lines = script.lines().rev();
    assert_eq!(lines.next(), Some("$x"));
    assert_eq!(lines.next(), Some("$x = 1"));
    assert!(lines.next().is_some());

    let (env, _) = programs_env(&[], "", 0, create_temp_dir());
    let (_, failed, _) = run(
        &create_power_shell_tool(None),
        json!({ "command": "1" }),
        some_env(env),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert_eq!(error_message(&failed), "spawn powershell ENOENT");
}

#[tokio::test(flavor = "multi_thread")]
async fn powershell_throws_on_a_nonzero_exit_after_streaming_the_output() {
    let (env, _) = programs_env(&["pwsh"], "partial", 3, create_temp_dir());
    let (_, failed, _) = run(
        &create_power_shell_tool(None),
        json!({ "command": "exit 3" }),
        some_env(env),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert_eq!(error_message(&failed), "Command exited with code 3");
}

#[cfg(windows)]
#[tokio::test(flavor = "multi_thread")]
async fn powershell_runs_real_powershell_with_utf8_output() {
    let env: Arc<TestEnv> = Arc::new(TestEnv::plain(create_temp_dir()));
    let (output, result, _) = run(
        &create_power_shell_tool(None),
        json!({ "command": "Write-Output ('h' + [char]0xe9 + 'llo'); exit 0" }),
        some_env(env),
        BACKGROUND_CONTEXT.clone(),
    )
    .await;
    assert!(
        !result.is_error.unwrap_or(false),
        "{}",
        error_message(&result)
    );
    assert_eq!(parts_text(&output).trim(), "héllo");
}
