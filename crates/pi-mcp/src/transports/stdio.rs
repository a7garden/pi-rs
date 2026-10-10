//! Port of `packages/mcp/src/transports/stdio.ts`.
//!
//! Spawns the MCP server as a child process speaking newline-delimited JSON-RPC on
//! stdio. Port deviations from upstream:
//! - `cross-spawn` becomes [`tokio::process::Command`]; on Unix the child runs in its
//!   own process group (upstream `detached`) and the whole group is signaled; the
//!   Windows `taskkill /T /F` tree-kill has no direct equivalent here, so Windows
//!   falls back to signaling the direct child only (upstream also prefers the group).
//! - The host-exit safety net (upstream `process.once("exit")`) is installed via
//!   `atexit(3)` on Unix: live process groups receive a final SIGTERM.

use parking_lot::Mutex;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Mutex as AsyncMutex;

use crate::protocol::jsonrpc::{parse_json_rpc_message, JsonRpcMessage, McpError};
use crate::transports::transport::{
    CloseListener, ErrorListener, McpTransport, MessageListener, TransportEvents, Unsubscribe,
    DEFAULT_MAX_MESSAGE_BYTES,
};

const DEFAULT_MAX_STDERR_BYTES: usize = 64 * 1024;
const DEFAULT_CLOSE_TIMEOUT_MS: u64 = 2_000;
/// How long a server gets to exit on its own after stdin closes, before it is sent SIGTERM.
const STDIN_CLOSE_GRACE_MS: u64 = 500;

const SIGTERM: i32 = 15;
const SIGKILL: i32 = 9;

pub type OnStderr = Arc<dyn Fn(&str) + Send + Sync>;

#[derive(Clone)]
pub struct StdioTransportOptions {
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    pub env: Option<Vec<(String, String)>>,
    /// Inherit the parent environment (default true, upstream `inheritEnv !== false`).
    pub inherit_env: Option<bool>,
    /// `"pipe"` (default, captured) or `"inherit"`.
    pub stderr: StdioMode,
    pub on_stderr: Option<OnStderr>,
    pub max_message_bytes: Option<usize>,
    pub max_stderr_bytes: Option<usize>,
    /// Time to wait for the server to exit after SIGTERM before sending SIGKILL. Default: 2000.
    pub close_timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdioMode {
    Pipe,
    Inherit,
}

impl Default for StdioTransportOptions {
    fn default() -> Self {
        StdioTransportOptions {
            command: String::new(),
            args: Vec::new(),
            cwd: None,
            env: None,
            inherit_env: None,
            stderr: StdioMode::Pipe,
            on_stderr: None,
            max_message_bytes: None,
            max_stderr_bytes: None,
            close_timeout_ms: None,
        }
    }
}

struct StdioState {
    events: TransportEvents,
    child: AsyncMutex<Option<tokio::process::Child>>,
    stdin: AsyncMutex<Option<tokio::process::ChildStdin>>,
    stdout_reader: AsyncMutex<Option<tokio::task::JoinHandle<()>>>,
    stdout_buffer: Mutex<Vec<u8>>,
    stderr_buffer: Mutex<Vec<u8>>,
    finished: AtomicBool,
    started: AtomicBool,
    closed: AtomicBool,
    pid: AtomicU64,
    max_message_bytes: usize,
    max_stderr_bytes: usize,
}

impl StdioState {
    fn pid(&self) -> Option<u32> {
        u32::try_from(self.pid.load(Ordering::SeqCst)).ok()
    }
}

/// Reported once the process and its stdout are done (upstream child `close` event).
fn finish_once(state: &StdioState) {
    if state.finished.swap(true, Ordering::SeqCst) {
        return;
    }
    #[cfg(unix)]
    process_groups::untrack(state.pid());
    let mut buffer = state.stdout_buffer.lock();
    if !buffer.iter().all(u8::is_ascii_whitespace) {
        state.events.emit_error(McpError::other(
            "MCP stdio server closed with an incomplete JSON-RPC message",
        ));
    }
    buffer.clear();
    state.events.emit_close();
}

/// Port of upstream `StdioTransport`.
pub struct StdioTransport {
    pub options: StdioTransportOptions,
    state: Arc<StdioState>,
}

impl StdioTransport {
    pub fn new(options: StdioTransportOptions) -> StdioTransport {
        let state = Arc::new(StdioState {
            events: TransportEvents::new(),
            child: AsyncMutex::new(None),
            stdin: AsyncMutex::new(None),
            stdout_reader: AsyncMutex::new(None),
            stdout_buffer: Mutex::new(Vec::new()),
            stderr_buffer: Mutex::new(Vec::new()),
            finished: AtomicBool::new(false),
            started: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            pid: AtomicU64::new(0),
            max_message_bytes: options
                .max_message_bytes
                .unwrap_or(DEFAULT_MAX_MESSAGE_BYTES),
            max_stderr_bytes: options.max_stderr_bytes.unwrap_or(DEFAULT_MAX_STDERR_BYTES),
        });
        StdioTransport { options, state }
    }

    pub fn pid(&self) -> Option<u32> {
        self.state.pid()
    }

    /// Everything captured from the server's stderr so far (upstream `get stderr()`).
    pub fn stderr(&self) -> String {
        let buffer = self.state.stderr_buffer.lock();
        String::from_utf8_lossy(&buffer).into_owned()
    }

    fn events(&self) -> &TransportEvents {
        &self.state.events
    }

    fn handle_stdout_chunk(&self, state: &StdioState, chunk: &[u8]) {
        let mut buffer = state.stdout_buffer.lock();
        buffer.extend_from_slice(chunk);
        loop {
            let Some(newline) = buffer.iter().position(|b| *b == 0x0a) else {
                if buffer.len() > state.max_message_bytes {
                    buffer.clear();
                    state.events.emit_error(McpError::other(format!(
                        "MCP stdio message exceeds {} bytes",
                        state.max_message_bytes
                    )));
                }
                return;
            };
            let line: Vec<u8> = buffer.drain(..newline).collect();
            buffer.drain(..1); // the newline
            if line.len() > state.max_message_bytes {
                state.events.emit_error(McpError::other(format!(
                    "MCP stdio message exceeds {} bytes",
                    state.max_message_bytes
                )));
                continue;
            }
            let mut text = String::from_utf8_lossy(&line).into_owned();
            if text.ends_with('\r') {
                text.pop();
            }
            if text.trim().is_empty() {
                continue;
            }
            let parsed = serde_json::from_str::<serde_json::Value>(&text)
                .map_err(|e| McpError::other(e.to_string()))
                .and_then(parse_json_rpc_message);
            match parsed {
                Ok(message) => state.events.emit_message(message),
                Err(error) => state.events.emit_error(error),
            }
        }
    }

    fn handle_stderr_chunk(&self, state: &StdioState, chunk: &[u8]) {
        {
            let mut buffer = state.stderr_buffer.lock();
            buffer.extend_from_slice(chunk);
            if buffer.len() > state.max_stderr_bytes {
                let overflow = buffer.len() - state.max_stderr_bytes;
                buffer.drain(..overflow);
            }
        }
        if let Some(on_stderr) = &self.options.on_stderr {
            on_stderr(&String::from_utf8_lossy(chunk));
        }
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn start(&self) -> Result<(), McpError> {
        if self.state.started.swap(true, Ordering::SeqCst) {
            return Err(McpError::other("MCP stdio transport already started"));
        }
        if self.state.closed.load(Ordering::SeqCst) {
            return Err(McpError::connection_closed("MCP connection closed"));
        }
        let mut command = tokio::process::Command::new(&self.options.command);
        command
            .args(&self.options.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(match self.options.stderr {
                StdioMode::Pipe => Stdio::piped(),
                StdioMode::Inherit => Stdio::inherit(),
            });
        if let Some(cwd) = &self.options.cwd {
            command.current_dir(cwd);
        }
        if self.options.inherit_env == Some(false) {
            command.env_clear();
        }
        if let Some(env) = &self.options.env {
            for (name, value) in env {
                command.env(name, value);
            }
        }
        // Own process group, so closing the transport can terminate the server's children too.
        #[cfg(unix)]
        command.process_group(0);
        #[cfg(windows)]
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW, upstream `windowsHide`.

        let mut child = command
            .spawn()
            .map_err(|e| McpError::other(e.to_string()))?;
        let pid = child.id().unwrap_or_default();
        self.state.pid.store(u64::from(pid), Ordering::SeqCst);
        #[cfg(unix)]
        process_groups::track(pid);

        let stdin = child.stdin.take().expect("stdin piped");
        let mut stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take();

        *self.state.child.lock().await = Some(child);
        *self.state.stdin.lock().await = Some(stdin);

        // stdout reader: newline-delimited JSON-RPC (upstream `handleStdout` + `close` handling).
        let reader_state = self.state.clone();
        let transport_for_stdout = StdioTransport {
            options: self.options.clone(),
            state: self.state.clone(),
        };
        let reader = tokio::spawn(async move {
            let mut chunk = [0u8; 8192];
            loop {
                match stdout.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => transport_for_stdout.handle_stdout_chunk(&reader_state, &chunk[..n]),
                }
            }
            finish_once(&reader_state);
        });
        *self.state.stdout_reader.lock().await = Some(reader);

        // stderr reader (upstream `handleStderr`).
        if let Some(mut stderr) = stderr {
            let stderr_state = self.state.clone();
            let transport_for_stderr = StdioTransport {
                options: self.options.clone(),
                state: self.state.clone(),
            };
            tokio::spawn(async move {
                let mut chunk = [0u8; 8192];
                loop {
                    match stderr.read(&mut chunk).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            transport_for_stderr.handle_stderr_chunk(&stderr_state, &chunk[..n])
                        }
                    }
                }
            });
        }
        Ok(())
    }

    async fn send(&self, message: JsonRpcMessage) -> Result<(), McpError> {
        if !self.state.started.load(Ordering::SeqCst) || self.state.closed.load(Ordering::SeqCst) {
            return Err(McpError::connection_closed("MCP connection closed"));
        }
        let mut stdin = self.state.stdin.lock().await;
        let Some(stdin) = stdin.as_mut() else {
            return Err(McpError::connection_closed("MCP connection closed"));
        };
        let mut payload =
            serde_json::to_vec(&message).map_err(|e| McpError::other(e.to_string()))?;
        payload.push(b'\n');
        stdin
            .write_all(&payload)
            .await
            .map_err(|e| McpError::other(e.to_string()))?;
        stdin
            .flush()
            .await
            .map_err(|e| McpError::other(e.to_string()))
    }

    async fn close(&self) -> Result<(), McpError> {
        if self.state.closed.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let child = self.state.child.lock().await.take();
        let Some(mut child) = child else {
            finish_once(&self.state);
            return Ok(());
        };
        // Shutdown per the spec: close stdin and let the server exit, then SIGTERM, then SIGKILL.
        *self.state.stdin.lock().await = None;

        let close_timeout_ms = self
            .options
            .close_timeout_ms
            .unwrap_or(DEFAULT_CLOSE_TIMEOUT_MS);
        let grace = STDIN_CLOSE_GRACE_MS.min(close_timeout_ms);
        let mut exited = child
            .try_wait()
            .map_err(|e| McpError::other(e.to_string()))?
            .is_some();
        if !exited {
            match tokio::time::timeout(std::time::Duration::from_millis(grace), child.wait()).await
            {
                Ok(Ok(_)) => exited = true,
                Ok(Err(e)) => return Err(McpError::other(e.to_string())),
                Err(_) => kill_process_tree(self.state.pid(), SIGTERM),
            }
        }
        if !exited {
            match tokio::time::timeout(
                std::time::Duration::from_millis(close_timeout_ms),
                child.wait(),
            )
            .await
            {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => return Err(McpError::other(e.to_string())),
                Err(_) => kill_process_tree(self.state.pid(), SIGKILL),
            }
        }
        // Upstream also SIGTERMs the group once the server closed, so children that
        // ignored stdin closing do not outlive it.
        kill_process_tree(self.state.pid(), SIGTERM);
        // Let the stdout reader observe EOF and report any trailing partial message
        // (upstream resolves on the child `close` event, i.e. after stdio drained).
        let reader = self.state.stdout_reader.lock().await.take();
        if let Some(reader) = reader {
            let _ = reader.await;
        } else {
            finish_once(&self.state);
        }
        Ok(())
    }

    fn on_message(&self, listener: MessageListener) -> Unsubscribe {
        self.events().on_message(listener)
    }

    fn on_error(&self, listener: ErrorListener) -> Unsubscribe {
        self.events().on_error(listener)
    }

    fn on_close(&self, listener: CloseListener) -> Unsubscribe {
        self.events().on_close(listener)
    }
}

/// Port of upstream `killProcessTree` (Unix half): signal the whole process group so
/// wrappers like `npx` or `uvx` do not leave the server behind, falling back to the
/// direct child. Windows has no group here; the direct child is killed instead.
fn kill_process_tree(pid: Option<u32>, signal: i32) {
    let Some(pid) = pid else { return };
    #[cfg(unix)]
    {
        unsafe {
            if libc::kill(-(pid as i32), signal) == 0 {
                return;
            }
            // The group is gone or was never created; fall back to the direct child.
            let _ = libc::kill(pid as i32, signal);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (pid, signal);
    }
}

/// Process groups of running servers, killed if the host exits without closing them
/// (upstream `liveProcessGroups` + `process.once("exit")`).
#[cfg(unix)]
mod process_groups {
    use parking_lot::Mutex;
    use std::collections::HashSet;
    use std::sync::Once;

    static LIVE: Mutex<Option<HashSet<i32>>> = Mutex::new(None);
    static HOOK: Once = Once::new();

    extern "C" fn on_exit() {
        if let Some(groups) = LIVE.lock().as_ref() {
            for pid in groups {
                unsafe {
                    let _ = libc::kill(-*pid, libc::SIGTERM);
                }
            }
        }
    }

    pub fn track(pid: u32) {
        HOOK.call_once(|| unsafe {
            libc::atexit(on_exit);
        });
        if let Ok(pid) = i32::try_from(pid) {
            LIVE.lock().get_or_insert_with(HashSet::new).insert(pid);
        }
    }

    pub fn untrack(pid: Option<u32>) {
        if let Some(pid) = pid.and_then(|p| i32::try_from(p).ok()) {
            if let Some(groups) = LIVE.lock().as_mut() {
                groups.remove(&pid);
            }
        }
    }
}
