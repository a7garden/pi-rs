//! Mirrors upstream `src/runtime/host.ts`: the sandbox and per-execution host logic.
//!
//! Upstream spawns a Node `Worker` per execution and talks the [`protocol`](super::protocol)
//! messages to it. The JS-specific surface — spawning the worker, posting messages,
//! terminating it — is captured by the [`CodeModeWorkerFactory`] / [`WorkerHandle`] traits so
//! the core stays pure Rust and is testable against in-process fakes; pi-runtime implements
//! them over real worker threads and QuickJS VMs.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Map, Value};
use tokio::sync::mpsc;

use crate::identifier::to_codemode_identifier;
use crate::runtime::protocol::{
    CallTarget, DoneOutcome, GlobalInfo, HostToWorkerMessage, ToolInfo, WorkerData,
    WorkerToHostMessage,
};
use crate::types::{
    AbortSignal, CodemodeCall, CodemodeCallStatus, CodemodeError, CodemodeErrorKind,
    CodemodeExecuteOptions, CodemodeOutputItem, CodemodeResult, CodemodeStoreWrites, CodemodeTool,
    CodemodeToolContext,
};
use crate::wasm::{UnavailableWasmLoader, WasmLoader};

const DEFAULT_TIMEOUT_MS: f64 = 300_000.0;

fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' || c == '$' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
}

fn is_reserved_global(name: &str) -> bool {
    matches!(
        name,
        "tools"
            | "ALL_TOOLS"
            | "console"
            | "text"
            | "image"
            | "exit"
            | "globalThis"
            | "store"
            | "load"
    )
}

/// Snapshot for `load()`: key to JSON text. Keys with non-JSON-serializable values are
/// skipped, matching upstream `serializeStore` (`JSON.stringify` returning `undefined`).
fn serialize_store(store: Option<&Map<String, Value>>) -> Map<String, Value> {
    let mut serialized = Map::new();
    for (key, value) in store.into_iter().flatten() {
        if let Ok(json) = serde_json::to_string(value) {
            serialized.insert(key.clone(), Value::String(json));
        }
    }
    serialized
}

/// The prelude that serializes values for the host runs in the same VM as the script, so a
/// script that patches built-ins (for example `Array.prototype.toJSON`) can make it send
/// malformed data. Payloads that do not decode carry this error, and the execution fails as
/// a sandbox error. Upstream `BridgeError`.
struct BridgeError(String);

fn parse_bridge_json(json: &str, what: &str) -> Result<Value, BridgeError> {
    serde_json::from_str(json).map_err(|_| BridgeError(format!("{what} is not valid JSON")))
}

fn parse_store_writes(json: &str) -> Result<CodemodeStoreWrites, BridgeError> {
    // Upstream wire shape: `[key, json]` pairs for `store()` and bare `[key]` for deletions.
    let entries = parse_bridge_json(json, "store writes")?;
    let Value::Array(entries) = entries else {
        return Err(BridgeError("store writes are not an array".to_string()));
    };
    let mut writes = CodemodeStoreWrites::default();
    for entry in entries {
        let (key, value) = match entry {
            Value::Array(values) => match values.as_slice() {
                [Value::String(key)] => (key.clone(), None),
                [Value::String(key), Value::String(value)] => (key.clone(), Some(value.clone())),
                _ => {
                    return Err(BridgeError(
                        "store writes contain a malformed entry".to_string(),
                    ))
                }
            },
            _ => {
                return Err(BridgeError(
                    "store writes contain a malformed entry".to_string(),
                ))
            }
        };
        match value {
            None => writes.delete.push(key),
            Some(value) => {
                // `JSON.stringify(key)` renders the key as JSON, quotes included.
                let what = format!(
                    "store value for {}",
                    serde_json::to_string(&key).unwrap_or_else(|_| key.clone())
                );
                writes.set.insert(key, parse_bridge_json(&value, &what)?);
            }
        }
    }
    Ok(writes)
}

fn parse_script_error(error_json: &str) -> Result<CodemodeError, BridgeError> {
    let parsed = parse_bridge_json(error_json, "script error")?;
    if !parsed.is_object() && !parsed.is_array() {
        // JS typeof calls arrays objects, but they fail the field checks below; everything
        // else (including null, which upstream also excludes explicitly) is not an object.
        return Err(BridgeError("script error is not an object".to_string()));
    }
    let Value::Object(fields) = parsed else {
        return Err(BridgeError("script error is malformed".to_string()));
    };
    // Only absent fields are optional; `null` fails the upstream `typeof … !== "string"` checks.
    let string_field = |name: &str| -> Result<Option<String>, BridgeError> {
        match fields.get(name) {
            None => Ok(None),
            Some(Value::String(value)) => Ok(Some(value.clone())),
            Some(_) => Err(BridgeError("script error is malformed".to_string())),
        }
    };
    let Some(message) = string_field("message")? else {
        return Err(BridgeError("script error is malformed".to_string()));
    };
    let name = string_field("name")?;
    let stack = string_field("stack")?;
    let mut error = CodemodeError::new(CodemodeErrorKind::Script, message);
    if let Some(name) = name {
        error = error.with_name(name);
    }
    if let Some(stack) = stack {
        error = error.with_stack(stack);
    }
    Ok(error)
}

/// Sandbox message for a decode failure on the worker bridge, naming the likely cause.
fn bridge_broken(reason: &str) -> CodemodeError {
    CodemodeError::new(
        CodemodeErrorKind::Sandbox,
        format!("Sandbox bridge broken: {reason}. The script may have modified built-ins such as a prototype's toJSON."),
    )
}

fn format_timeout(timeout_ms: f64) -> String {
    if timeout_ms.fract() == 0.0 {
        format!("{}", timeout_ms as u64)
    } else {
        format!("{timeout_ms}")
    }
}

/// A spawned per-execution worker. The host posts [`HostToWorkerMessage`]s, receives raw
/// payloads, and can terminate the worker. Worker-to-host payloads arrive exactly as the
/// worker posted them (`serde_json::Value`); the host validates each against the protocol,
/// so a malformed payload breaks the bridge instead of being dropped (upstream checks
/// `isWorkerToHostMessage` on every `message` event). `interrupt` is the Rust
/// counterpart of the upstream shared `Int32Array`: the host sets it to non-zero right
/// before termination, and the VM's interrupt handler polls it so a script spinning in wasm
/// can be stopped.
pub struct RunningWorker {
    pub to_worker: mpsc::Sender<HostToWorkerMessage>,
    pub from_worker: mpsc::Receiver<Value>,
    pub interrupt: Arc<AtomicI32>,
    pub handle: Box<dyn WorkerHandle>,
}

/// Termination surface of a spawned worker.
pub trait WorkerHandle: Send {
    /// Terminate the worker; resolves once the worker thread/VM is gone. Upstream:
    /// `worker.terminate()`.
    fn terminate(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;

    /// Exit code reported when the worker's message channel closes before the script
    /// settles. Defaults to 1 (an abnormal exit), matching upstream workers that died on
    /// an uncaught error.
    fn exit_code(&self) -> i32 {
        1
    }
}

/// Spawns one worker per execution. Rust counterpart of `new Worker(workerUrl, { workerData })`.
pub trait CodeModeWorkerFactory: Send + Sync {
    fn spawn(&self, data: WorkerData) -> Result<RunningWorker, String>;
}

/// Options for [`CodemodeSandbox::new`], mirroring upstream `CodemodeSandboxOptions`.
#[derive(Default)]
pub struct CodemodeSandboxOptions {
    pub tools: Vec<CodemodeTool>,
    /// Functions exposed as top-level identifiers instead of on `tools`. They behave like
    /// tools (JSON round trip, promise result) but are not recorded in `result.calls`. Names
    /// must be identifiers and may not shadow the built-in globals (`tools`, `ALL_TOOLS`,
    /// `console`, `text`, `image`, `exit`, `store`, `load`).
    pub globals: Vec<CodemodeTool>,
    /// Overall deadline per execution, including time spent in tools. `f64::INFINITY`
    /// disables the deadline; the execution then only ends when the script settles or is
    /// aborted. Default: 300000.
    pub timeout_ms: Option<f64>,
    /// Maximum memory the QuickJS VM may allocate. Default: no limit beyond wasm32's 4 GiB
    /// address space.
    pub memory_limit_bytes: Option<u64>,
    /// Loads the compiled `quickjs-wasi/quickjs.wasm`. Default: [`UnavailableWasmLoader`]
    /// (the real loader belongs to pi-runtime).
    pub wasm_loader: Option<Arc<dyn WasmLoader>>,
    /// Spawns the per-execution worker. Default: [`NoWorkerFactory`] (the real worker
    /// belongs to pi-runtime).
    pub worker_factory: Option<Arc<dyn CodeModeWorkerFactory>>,
}

/// The crate-default worker factory: worker processes belong to a runtime implementation.
/// Executions report its error through the sandbox-error path (`Failed to start worker: …`),
/// mirroring upstream behavior when the worker file is missing.
pub struct NoWorkerFactory;

impl CodeModeWorkerFactory for NoWorkerFactory {
    fn spawn(&self, _data: WorkerData) -> Result<RunningWorker, String> {
        Err(
            "no worker implementation; configure `CodemodeSandboxOptions::worker_factory`"
                .to_string(),
        )
    }
}

/// Cancellation delivered to the execution loop: the error `finish` records.
type CancelMessage = (CodemodeErrorKind, String);

/// One in-flight execution: its cancel channel and a completion signal for `close()`.
struct RunningEntry {
    cancel: mpsc::Sender<CancelMessage>,
    done: tokio::sync::oneshot::Receiver<()>,
}

struct SandboxInner {
    tools_by_name: Mutex<Vec<(String, CodemodeTool)>>,
    globals_by_name: Mutex<Vec<(String, CodemodeTool)>>,
    timeout_ms: f64,
    memory_limit_bytes: Option<u64>,
    wasm_loader: Arc<dyn WasmLoader>,
    worker_factory: Arc<dyn CodeModeWorkerFactory>,
    closed: Mutex<bool>,
    running: Mutex<Vec<RunningEntry>>,
}

impl std::fmt::Debug for CodemodeSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodemodeSandbox").finish_non_exhaustive()
    }
}

/// Runs JavaScript in a QuickJS VM (a separate wasm instance) inside a worker thread. The
/// script sees `tools.<name>(args)` for every registered tool, `ALL_TOOLS`, the output
/// helpers `text`, `image`, `exit`, and `console.*`, `store`/`load`, and the configured
/// globals; nothing else (no timers, `fetch`, `process`, `require`, modules).
///
/// Each `execute()` gets its own worker and VM; the sandbox only holds the tool table and
/// defaults. `close()` aborts in-flight executions. `execute` must be called from within a
/// tokio runtime (it spawns the per-execution task).
#[derive(Clone)]
pub struct CodemodeSandbox {
    inner: Arc<SandboxInner>,
}

impl CodemodeSandbox {
    /// Creates a sandbox. Returns `Err` with the upstream error strings when a global name
    /// is invalid/reserved/conflicting.
    pub fn new(options: CodemodeSandboxOptions) -> Result<Self, String> {
        let mut tools_by_name: Vec<(String, CodemodeTool)> = Vec::new();
        for tool in options.tools {
            if tools_by_name.iter().any(|(name, _)| name == &tool.name) {
                return Err(format!("Tool \"{}\" is already registered", tool.name));
            }
            tools_by_name.push((tool.name.clone(), tool));
        }
        let mut globals_by_name: Vec<(String, CodemodeTool)> = Vec::new();
        let mut namespaces: Vec<String> = Vec::new();
        for global in options.globals {
            let parts: Vec<&str> = global.name.split('.').collect();
            if parts.len() > 2
                || !parts.iter().all(|part| is_identifier(part))
                || is_reserved_global(parts[0])
            {
                return Err(format!("Invalid global name \"{}\"", global.name));
            }
            if globals_by_name.iter().any(|(name, _)| name == &global.name) {
                return Err(format!("Global \"{}\" is already registered", global.name));
            }
            if parts.len() == 2 {
                namespaces.push(parts[0].to_string());
            }
            globals_by_name.push((global.name.clone(), global));
        }
        for namespace in namespaces {
            if globals_by_name.iter().any(|(name, _)| name == &namespace) {
                return Err(format!(
                    "Global \"{namespace}\" conflicts with the namespace \"{namespace}\""
                ));
            }
        }
        Ok(Self {
            inner: Arc::new(SandboxInner {
                tools_by_name: Mutex::new(tools_by_name),
                globals_by_name: Mutex::new(globals_by_name),
                timeout_ms: options.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS),
                memory_limit_bytes: options.memory_limit_bytes,
                wasm_loader: options
                    .wasm_loader
                    .unwrap_or_else(|| Arc::new(UnavailableWasmLoader)),
                worker_factory: options
                    .worker_factory
                    .unwrap_or_else(|| Arc::new(NoWorkerFactory)),
                closed: Mutex::new(false),
                running: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Returns `Err` if a tool with the same name is already registered.
    pub fn register_tool(&self, tool: CodemodeTool) -> Result<(), String> {
        let mut tools = self.inner.tools_by_name.lock();
        if tools.iter().any(|(name, _)| name == &tool.name) {
            return Err(format!("Tool \"{}\" is already registered", tool.name));
        }
        tools.push((tool.name.clone(), tool));
        Ok(())
    }

    pub fn unregister_tool(&self, name: &str) -> bool {
        let mut tools = self.inner.tools_by_name.lock();
        let before = tools.len();
        tools.retain(|(existing, _)| existing != name);
        tools.len() != before
    }

    /// Registered tools, in registration order.
    pub fn tools(&self) -> Vec<CodemodeTool> {
        self.inner
            .tools_by_name
            .lock()
            .iter()
            .map(|(_, tool)| tool.clone())
            .collect()
    }

    /// Registered globals, in registration order.
    pub fn globals(&self) -> Vec<CodemodeTool> {
        self.inner
            .globals_by_name
            .lock()
            .iter()
            .map(|(_, tool)| tool.clone())
            .collect()
    }

    /// `code` is an async function body: `return` and top-level `await` work. Never fails
    /// for script failures; those come back as the error variant of [`CodemodeResult`]. The
    /// `Err(String)` return is reserved for a closed sandbox, mirroring the upstream
    /// rejected promise.
    pub fn execute(
        &self,
        code: String,
        options: CodemodeExecuteOptions,
    ) -> impl Future<Output = Result<CodemodeResult, String>> + Send + '_ {
        let inner = Arc::clone(&self.inner);
        async move {
            if *inner.closed.lock() {
                return Err("Sandbox is closed".to_string());
            }
            let (cancel_tx, cancel_rx) = mpsc::channel::<CancelMessage>(4);
            let execution = Execution {
                code: code.into(),
                tools: Arc::new(
                    inner
                        .tools_by_name
                        .lock()
                        .iter()
                        .map(|(name, tool)| (name.clone(), tool.clone()))
                        .collect(),
                ),
                globals: Arc::new(
                    inner
                        .globals_by_name
                        .lock()
                        .iter()
                        .map(|(name, tool)| (name.clone(), tool.clone()))
                        .collect(),
                ),
                timeout_ms: options.timeout_ms.unwrap_or(inner.timeout_ms),
                signal: options.signal,
                memory_limit_bytes: inner.memory_limit_bytes,
                store: serialize_store(options.store.as_ref()),
                wasm_loader: Arc::clone(&inner.wasm_loader),
                worker_factory: Arc::clone(&inner.worker_factory),
                cancel_tx: cancel_tx.clone(),
                cancel_rx,
            };
            let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
            let running_entry = RunningEntry {
                cancel: cancel_tx,
                done: done_rx,
            };
            inner.running.lock().push(running_entry);
            let handle = tokio::spawn(async move {
                let result = execution.run().await;
                let _ = done_tx.send(());
                result
            });
            Ok(handle.await.unwrap_or_else(|error| CodemodeResult::Err {
                ok: crate::types::False,
                error: CodemodeError::new(
                    CodemodeErrorKind::Sandbox,
                    format!("execution task failed: {error}"),
                ),
                output: Vec::new(),
                calls: Vec::new(),
            }))
        }
    }

    /// Aborts in-flight executions (they resolve with `kind: "aborted"`) and rejects new ones.
    pub async fn close(&self) {
        *self.inner.closed.lock() = true;
        let running: Vec<RunningEntry> = std::mem::take(&mut *self.inner.running.lock());
        let mut completions = Vec::new();
        for entry in running {
            let RunningEntry { cancel, done } = entry;
            let _ = cancel
                .send((CodemodeErrorKind::Aborted, "Sandbox closed".to_string()))
                .await;
            completions.push(done);
        }
        for done in completions {
            let _ = done.await;
        }
    }
}

struct PendingCall {
    /// Index into `ExecutionState::calls` for tool calls (globals are not recorded).
    record_index: Option<usize>,
    controller: AbortSignal,
    started_at: std::time::Instant,
}

struct ExecutionState {
    output: Mutex<Vec<CodemodeOutputItem>>,
    calls: Mutex<Vec<CodemodeCall>>,
    pending: Mutex<HashMap<u64, PendingCall>>,
}

struct Execution {
    code: String,
    tools: Arc<HashMap<String, CodemodeTool>>,
    globals: Arc<HashMap<String, CodemodeTool>>,
    timeout_ms: f64,
    signal: Option<AbortSignal>,
    memory_limit_bytes: Option<u64>,
    store: Map<String, Value>,
    wasm_loader: Arc<dyn WasmLoader>,
    worker_factory: Arc<dyn CodeModeWorkerFactory>,
    cancel_tx: mpsc::Sender<CancelMessage>,
    cancel_rx: mpsc::Receiver<CancelMessage>,
}

impl Execution {
    async fn run(self) -> CodemodeResult {
        // Deadline.
        if self.timeout_ms.is_finite() {
            let cancel_tx = self.cancel_tx.clone();
            let timeout_ms = self.timeout_ms;
            let message = format!(
                "Execution timed out after {} ms",
                format_timeout(timeout_ms)
            );
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs_f64(timeout_ms / 1000.0)).await;
                let _ = cancel_tx.send((CodemodeErrorKind::Timeout, message)).await;
            });
        }

        // Caller's signal.
        if let Some(signal) = self.signal.clone() {
            let cancel_tx = self.cancel_tx.clone();
            tokio::spawn(async move {
                signal.wait_aborted().await;
                let reason = signal
                    .reason()
                    .unwrap_or_else(|| "Execution aborted".to_string());
                let _ = cancel_tx.send((CodemodeErrorKind::Aborted, reason)).await;
            });
        }

        // Load the wasm (upstream: `options.wasm.then(start, error => finish(sandbox))`).
        let wasm = match self.wasm_loader.load().await {
            Ok(wasm) => wasm,
            Err(error) => {
                return self
                    .finish(
                        &Arc::new(ExecutionState::empty()),
                        None,
                        Some(CodemodeError::new(
                            CodemodeErrorKind::Sandbox,
                            format!("Failed to load QuickJS: {error}"),
                        )),
                        None,
                    )
                    .await;
            }
        };

        let data = WorkerData {
            code: self.code.clone(),
            tools: self
                .tools
                .values()
                .map(|tool| ToolInfo {
                    name: tool.name.clone(),
                    js_name: to_codemode_identifier(&tool.name),
                    description: tool.description.clone().unwrap_or_default(),
                })
                .collect(),
            globals: self
                .globals
                .values()
                .map(|global| GlobalInfo {
                    name: global.name.clone(),
                    spread: global.spread,
                })
                .collect(),
            wasm: Some(wasm),
            memory_limit_bytes: self.memory_limit_bytes,
            store: self.store.clone(),
            interrupt: Arc::new(AtomicI32::new(0)),
        };

        match self.worker_factory.spawn(data) {
            Ok(worker) => self.drive(worker).await,
            Err(error) => {
                self.finish(
                    &Arc::new(ExecutionState::empty()),
                    None,
                    Some(CodemodeError::new(
                        CodemodeErrorKind::Sandbox,
                        format!("Failed to start worker: {error}"),
                    )),
                    None,
                )
                .await
            }
        }
    }

    /// Message loop until done/crash/cancel/worker exit.
    async fn drive(mut self, mut worker: RunningWorker) -> CodemodeResult {
        let state = Arc::new(ExecutionState::empty());

        loop {
            tokio::select! {
                biased;
                cancel = self.cancel_rx.recv() => {
                    match cancel {
                        Some((kind, message)) => {
                            return self
                                .finish(&state, Some(&mut worker), Some(CodemodeError::new(kind, message)), None)
                                .await;
                        }
                        None => unreachable!("the execution holds its own cancel sender"),
                    }
                }
                message = worker.from_worker.recv() => {
                    // A queued cancellation preempts worker messages, so a done that races an
                    // abort still reports the abort (upstream checks `finished` per message).
                    if let Ok((kind, message)) = self.cancel_rx.try_recv() {
                        return self
                            .finish(&state, Some(&mut worker), Some(CodemodeError::new(kind, message)), None)
                            .await;
                    }
                    match message {
                        Some(raw) => {
                            // An exception here would be uncaught in the host process and leave
                            // the execution unsettled (upstream wraps `handleMessage` in
                            // try/catch and reports every failure through `finish`).
                            let message = match serde_json::from_value::<WorkerToHostMessage>(raw) {
                                Ok(message) => message,
                                Err(_) => {
                                    return self
                                        .finish(&state, Some(&mut worker), Some(bridge_broken("unknown message from the worker")), None)
                                        .await;
                                }
                            };
                            match self.on_message(&state, &mut worker, message).await {
                                Some(Ok(result)) => return result,
                                Some(Err(error)) => {
                                    return self.finish(&state, Some(&mut worker), Some(bridge_broken(&error.0)), None).await;
                                }
                                None => {}
                            }
                        }
                        None => {
                            let code = worker.handle.exit_code();
                            return self
                                .finish(
                                    &state,
                                    Some(&mut worker),
                                    Some(CodemodeError::new(
                                        CodemodeErrorKind::Sandbox,
                                        format!("Worker exited with code {code} before the script settled"),
                                    )),
                                    None,
                                )
                                .await;
                        }
                    }
                }
            }
        }
    }

    /// Handles one worker message; `Some(Ok(result))` ends the execution, `Some(Err)` is a
    /// broken bridge (upstream `handleMessage` throwing `BridgeError` inside the listener).
    async fn on_message(
        &self,
        state: &Arc<ExecutionState>,
        worker: &mut RunningWorker,
        message: WorkerToHostMessage,
    ) -> Option<Result<CodemodeResult, BridgeError>> {
        match message {
            WorkerToHostMessage::Output { item } => {
                state.output.lock().push(item);
                None
            }
            WorkerToHostMessage::Call {
                id,
                target,
                name,
                args,
            } => {
                if state.pending.lock().contains_key(&id) {
                    return Some(Err(BridgeError(format!("duplicate call id {id}"))));
                }
                self.handle_call(state, &worker.to_worker, id, target, name, args);
                None
            }
            WorkerToHostMessage::Done { ok, outcome } => {
                self.on_done(state, worker, ok, outcome).await
            }
            WorkerToHostMessage::Crash { message } => Some(Ok(self
                .finish(
                    state,
                    Some(worker),
                    Some(CodemodeError::new(CodemodeErrorKind::Sandbox, message)),
                    None,
                )
                .await)),
        }
    }

    async fn on_done(
        &self,
        state: &Arc<ExecutionState>,
        worker: &mut RunningWorker,
        ok: bool,
        outcome: DoneOutcome,
    ) -> Option<Result<CodemodeResult, BridgeError>> {
        // Decode everything before finish(), which must not throw once it starts.
        if !ok {
            let DoneOutcome::Err { error } = outcome else {
                // Missing `error` field: upstream parses `undefined` and JSON.parse throws.
                return Some(Err(BridgeError(
                    "script error is not valid JSON".to_string(),
                )));
            };
            let script_error = match parse_script_error(&error) {
                Ok(error) => error,
                Err(error) => return Some(Err(error)),
            };
            return Some(Ok(self
                .finish(state, Some(worker), Some(script_error), None)
                .await));
        }
        let DoneOutcome::Ok { value, writes } = outcome else {
            // Missing `value` field: upstream parses `undefined` and JSON.parse throws.
            return Some(Err(BridgeError(
                "return value is not valid JSON".to_string(),
            )));
        };
        let value = match value {
            None => None,
            Some(json) => match parse_bridge_json(&json, "return value") {
                Ok(value) => Some(value),
                Err(error) => return Some(Err(error)),
            },
        };
        let writes = match parse_store_writes(&writes) {
            Ok(writes) => writes,
            Err(error) => return Some(Err(error)),
        };
        Some(Ok(self
            .finish(state, Some(worker), None, Some((value, writes)))
            .await))
    }

    fn handle_call(
        &self,
        state: &Arc<ExecutionState>,
        to_worker: &mpsc::Sender<HostToWorkerMessage>,
        id: u64,
        target: CallTarget,
        name: String,
        args: Option<String>,
    ) {
        let is_tool = target == CallTarget::Tool;
        let record_index = if is_tool {
            let mut calls = state.calls.lock();
            calls.push(CodemodeCall {
                name: name.clone(),
                status: CodemodeCallStatus::Cancelled,
                duration_ms: 0.0,
            });
            Some(calls.len() - 1)
        } else {
            None
        };
        let controller = AbortSignal::new();
        let started_at = std::time::Instant::now();
        state.pending.lock().insert(
            id,
            PendingCall {
                record_index,
                controller: controller.clone(),
                started_at,
            },
        );

        let state = Arc::clone(state);
        let to_worker = to_worker.clone();
        let tools = Arc::clone(&self.tools);
        let globals = Arc::clone(&self.globals);
        tokio::spawn(async move {
            let table = if is_tool { &tools } else { &globals };
            let label = if is_tool { "tool" } else { "global" };
            let (status, reply) = match dispatch(table, label, &name, args, controller).await {
                Ok(payload) => (
                    CodemodeCallStatus::Ok,
                    HostToWorkerMessage::Result {
                        id,
                        ok: true,
                        payload,
                    },
                ),
                Err(message) => (
                    CodemodeCallStatus::Error,
                    HostToWorkerMessage::Result {
                        id,
                        ok: false,
                        payload: Some(message),
                    },
                ),
            };
            // Already cancelled by finish(): the record keeps "cancelled" and the
            // worker is gone or going.
            let pending = state.pending.lock().remove(&id);
            if pending.is_none() {
                return;
            }
            if let (Some(index), Some(pending)) = (record_index, pending) {
                let mut calls = state.calls.lock();
                if let Some(record) = calls.get_mut(index) {
                    record.status = status;
                    record.duration_ms = pending.started_at.elapsed().as_secs_f64() * 1000.0;
                }
            }
            let _ = to_worker.send(reply).await;
        });
    }

    /// Winds down the execution: cancels pending calls (their records keep "cancelled" but
    /// get durations), sets the interrupt flag, and terminates the worker.
    async fn finish(
        &self,
        state: &Arc<ExecutionState>,
        worker: Option<&mut RunningWorker>,
        error: Option<CodemodeError>,
        ok: Option<(Option<Value>, CodemodeStoreWrites)>,
    ) -> CodemodeResult {
        {
            let mut pending = state.pending.lock();
            for (_, call) in pending.drain() {
                call.controller.abort("execution finished");
                if let Some(index) = call.record_index {
                    let mut calls = state.calls.lock();
                    if let Some(record) = calls.get_mut(index) {
                        record.duration_ms = call.started_at.elapsed().as_secs_f64() * 1000.0;
                    }
                }
            }
        }

        let result = match (error, ok) {
            (Some(error), _) => CodemodeResult::Err {
                ok: crate::types::False,
                error,
                output: std::mem::take(&mut *state.output.lock()),
                calls: std::mem::take(&mut *state.calls.lock()),
            },
            (None, Some((value, store_writes))) => CodemodeResult::Ok {
                ok: crate::types::True,
                value,
                output: std::mem::take(&mut *state.output.lock()),
                calls: std::mem::take(&mut *state.calls.lock()),
                store_writes,
            },
            (None, None) => unreachable!("finish requires an error or an ok payload"),
        };

        if let Some(worker) = worker {
            // The VM's interrupt handler polls this; the worker is terminated right after.
            worker.interrupt.store(1, Ordering::SeqCst);
            worker.handle.terminate().await;
        }
        result
    }
}

impl ExecutionState {
    fn empty() -> Self {
        Self {
            output: Mutex::new(Vec::new()),
            calls: Mutex::new(Vec::new()),
            pending: Mutex::new(HashMap::new()),
        }
    }
}

async fn dispatch(
    table: &HashMap<String, CodemodeTool>,
    label: &str,
    name: &str,
    args: Option<String>,
    controller: AbortSignal,
) -> Result<Option<String>, String> {
    let tool = table
        .get(name)
        .ok_or_else(|| format!("Unknown {label} \"{name}\""))?;
    let args: Option<Value> = match args {
        None => None,
        Some(json) => Some(serde_json::from_str(&json).map_err(|error| error.to_string())?),
    };
    let context = CodemodeToolContext { signal: controller };
    let value = (tool.execute)(args, context).await?;
    match value {
        None => Ok(None),
        Some(value) => Ok(Some(
            serde_json::to_string(&value).map_err(|error| error.to_string())?,
        )),
    }
}
