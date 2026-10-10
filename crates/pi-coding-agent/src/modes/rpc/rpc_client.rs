//! Port of `packages/coding-agent/src/modes/rpc/rpc-client.ts` — RPC client
//! for programmatic access to the coding agent.
//!
//! Spawns the agent in RPC mode and provides a typed API for all operations.
//!
//! Deviation **D94** — upstream spawns `node <cliPath>`; the client keeps
//! that default (`program: "node"`) but the program is configurable. A
//! custom program with no `cli_path` is spawned directly (no script
//! argument; D109 — the live-CLI suite runs the built `pi` binary). The
//! private-`send` test seam becomes a crate-internal transport override
//! (upstream tests poke `privateClient.send`); aggregates without a
//! `Deserialize` port (compaction, tree, commands) surface as
//! [`serde_json::Value`] (D93). The instance is shared behind `Arc`
//! internally, so `start`/`stop` take `&self` (upstream mutates instance
//! fields directly).

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};

#[cfg(test)]
use futures_util::future::BoxFuture;
use tokio::io::AsyncWriteExt;
use tokio::sync::oneshot;

use pi_agent::types::AgentThinkingLevel;

use crate::core::bash_executor::BashResult;
use crate::core::session_manager::SessionEntry;

use super::jsonl::{serialize_json_line, JsonlLineReader};
use super::rpc_types::RpcResponse;

/// Options for the RPC client (`RpcClientOptions`).
#[derive(Clone, Default)]
pub struct RpcClientOptions {
    /// Path to the agent CLI entry point (default: `dist/cli.js`).
    pub cli_path: Option<String>,
    /// Program executing the CLI (default: `node`; D94).
    pub program: Option<String>,
    /// Working directory for the agent.
    pub cwd: Option<String>,
    /// Additional environment variables.
    pub env: Option<HashMap<String, String>>,
    /// Provider to use.
    pub provider: Option<String>,
    /// Model ID to use.
    pub model: Option<String>,
    /// Additional CLI arguments.
    pub args: Vec<String>,
}

/// Available-model summary (`ModelInfo`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub provider: String,
    pub id: String,
    pub context_window: u64,
    pub reasoning: bool,
}

/// Agent event listener (`RpcEventListener`); the event is the wire JSON.
pub type RpcEventListener = Arc<dyn Fn(&JsonValue) + Send + Sync>;

/// Test transport seam (D94): replaces the child-process round trip with a
/// canned handler. Crate-internal; production leaves it unset.
#[cfg(test)]
pub(crate) type SendOverride =
    Box<dyn Fn(JsonValue) -> BoxFuture<'static, Result<RpcResponse, String>> + Send + Sync>;

/// Reader-loop detach flag.
struct DetachFlag(Mutex<bool>);

impl DetachFlag {
    fn new() -> Self {
        DetachFlag(Mutex::new(false))
    }

    fn detach(&self) {
        *self.0.lock() = true;
    }

    fn is_detached(&self) -> bool {
        *self.0.lock()
    }
}

/// State shared between the client handle and its reader tasks.
struct Core {
    options: RpcClientOptions,
    /// Owned by the exit watcher once `start` spawns it; `stop` signals
    /// through `child_pid` + `exited` instead of touching the handle.
    child: tokio::sync::Mutex<Option<tokio::process::Child>>,
    child_pid: Mutex<Option<u32>>,
    exited: tokio::sync::Notify,
    stdin: Mutex<Option<tokio::process::ChildStdin>>,
    listeners: Mutex<Vec<(u64, RpcEventListener)>>,
    listener_seq: AtomicU64,
    pending_requests: Mutex<HashMap<String, oneshot::Sender<Result<RpcResponse, String>>>>,
    request_id: AtomicU32,
    stderr: Mutex<String>,
    exit_error: Mutex<Option<String>>,
    stop_reading: Mutex<Option<Arc<DetachFlag>>>,
    #[cfg(test)]
    send_override: Mutex<Option<SendOverride>>,
}

/// Client for a spawned RPC-mode agent (`RpcClient`).
#[derive(Clone)]
pub struct RpcClient {
    core: Arc<Core>,
}

// Response payload structs (upstream method return types).

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClearedQueue {
    pub steering: Vec<String>,
    #[serde(rename = "followUp")]
    pub follow_up: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cancelled {
    pub cancelled: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRef {
    pub provider: String,
    pub id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CycledModel {
    pub model: ModelRef,
    #[serde(rename = "thinkingLevel")]
    pub thinking_level: String,
    #[serde(rename = "isScoped")]
    pub is_scoped: bool,
}

/// Parse a wire thinking-level name (`AgentThinkingLevel` is Serialize-only).
pub fn parse_thinking_level(level: &str) -> Result<AgentThinkingLevel, String> {
    match level {
        "off" => Ok(AgentThinkingLevel::Off),
        "minimal" => Ok(AgentThinkingLevel::Minimal),
        "low" => Ok(AgentThinkingLevel::Low),
        "medium" => Ok(AgentThinkingLevel::Medium),
        "high" => Ok(AgentThinkingLevel::High),
        "xhigh" => Ok(AgentThinkingLevel::Xhigh),
        "max" => Ok(AgentThinkingLevel::Max),
        other => Err(format!("Unknown thinking level: {other}")),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportedHtml {
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Forked {
    pub text: Option<String>,
    pub cancelled: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ForkMessage {
    pub entry_id: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entries {
    pub entries: Vec<SessionEntry>,
    #[serde(rename = "leafId")]
    pub leaf_id: Option<String>,
}

fn disposition_of(response: RpcResponse) -> Result<String, String> {
    let data = response.into_data()?;
    Ok(data
        .get("disposition")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_string())
}

/// Unix termination signal of an exit status (`signalCode`).
#[cfg(unix)]
fn unix_signal(status: &std::process::ExitStatus) -> Option<String> {
    use std::os::unix::process::ExitStatusExt;
    status.signal().map(|signal| signal.to_string())
}

#[cfg(not(unix))]
fn unix_signal(_status: &std::process::ExitStatus) -> Option<String> {
    None
}

impl RpcClient {
    pub fn new(options: RpcClientOptions) -> Self {
        RpcClient {
            core: Arc::new(Core {
                options,
                child: tokio::sync::Mutex::new(None),
                child_pid: Mutex::new(None),
                exited: tokio::sync::Notify::new(),
                stdin: Mutex::new(None),
                listeners: Mutex::new(Vec::new()),
                listener_seq: AtomicU64::new(0),
                pending_requests: Mutex::new(HashMap::new()),
                request_id: AtomicU32::new(0),
                stderr: Mutex::new(String::new()),
                exit_error: Mutex::new(None),
                stop_reading: Mutex::new(None),
                #[cfg(test)]
                send_override: Mutex::new(None),
            }),
        }
    }

    /// Get collected stderr output (useful for debugging).
    pub fn get_stderr(&self) -> String {
        self.core.stderr.lock().clone()
    }

    /// Install the crate-internal transport override (D94 test seam).
    #[cfg(test)]
    pub(crate) fn set_send_override(&self, override_fn: SendOverride) {
        *self.core.send_override.lock() = Some(override_fn);
    }

    /// Start the RPC agent process.
    pub async fn start(&self) -> Result<(), String> {
        if self.core.child.lock().await.is_some() {
            return Err("Client already started".to_string());
        }

        *self.core.exit_error.lock() = None;

        let cli_path = self.core.options.cli_path.clone();
        let program = self
            .core
            .options
            .program
            .clone()
            .unwrap_or_else(|| "node".to_string());
        // D109: a custom program with no explicit script path is the CLI
        // itself (the live-CLI tests spawn the built `pi` binary directly);
        // the default `node` program keeps the upstream script default.
        let script = match cli_path {
            Some(path) => Some(path),
            None if self.core.options.program.is_some() => None,
            None => Some("dist/cli.js".to_string()),
        };
        let mut args: Vec<String> = script.into_iter().collect();
        args.push("--mode".to_string());
        args.push("rpc".to_string());

        if let Some(provider) = &self.core.options.provider {
            args.push("--provider".to_string());
            args.push(provider.clone());
        }
        if let Some(model) = &self.core.options.model {
            args.push("--model".to_string());
            args.push(model.clone());
        }
        args.extend(self.core.options.args.iter().cloned());

        let mut command = tokio::process::Command::new(&program);
        command
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // A client dropped without `stop` kills the agent instead of
        // orphaning it (upstream teardown relies on vitest process cleanup).
        command.kill_on_drop(true);
        if let Some(cwd) = &self.core.options.cwd {
            command.current_dir(cwd);
        }
        if let Some(env) = &self.core.options.env {
            for (key, value) in env {
                command.env(key, value);
            }
        }

        let mut child = command
            .spawn()
            .map_err(|error| format!("Agent process error: {error}"))?;
        *self.core.stdin.lock() = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr_pipe = child.stderr.take();
        *self.core.child_pid.lock() = child.id();
        *self.core.child.lock().await = Some(child);

        // Collect stderr for debugging (upstream tees to process.stderr).
        if let Some(mut stderr_pipe) = stderr_pipe {
            let core = Arc::clone(&self.core);
            tokio::spawn(async move {
                use tokio::io::AsyncReadExt;
                let mut buffer = [0u8; 8192];
                loop {
                    match stderr_pipe.read(&mut buffer).await {
                        Ok(0) | Err(_) => break,
                        Ok(read) => {
                            core.stderr
                                .lock()
                                .push_str(&String::from_utf8_lossy(&buffer[..read]));
                            use std::io::Write;
                            let _ = std::io::stderr().write_all(&buffer[..read]);
                        }
                    }
                }
            });
        }

        // Exit watcher: owns the Child, records the error, rejects every
        // pending request (`createProcessExitError` / `rejectPendingRequests`).
        let core = Arc::clone(&self.core);
        tokio::spawn(async move {
            let (code, signal, spawn_error) = {
                let mut child = core.child.lock().await.take();
                match child.as_mut() {
                    Some(child) => match child.wait().await {
                        Ok(status) => (status.code(), unix_signal(&status), None),
                        Err(error) => (None, None, Some(error.to_string())),
                    },
                    None => (None, None, Some("child already collected".to_string())),
                }
            };
            let message = if let Some(spawn_error) = spawn_error {
                format!(
                    "Agent process error: {spawn_error}. Stderr: {}",
                    core.stderr.lock().clone()
                )
            } else {
                format!(
                    "Agent process exited (code={} signal={}). Stderr: {}",
                    code.map(|code| code.to_string())
                        .unwrap_or_else(|| "null".to_string()),
                    signal.unwrap_or_else(|| "null".to_string()),
                    core.stderr.lock().clone(),
                )
            };
            *core.exit_error.lock() = Some(message.clone());
            core.exited.notify_waiters();
            let pending: Vec<_> = core.pending_requests.lock().drain().collect();
            for (_, sender) in pending {
                let _ = sender.send(Err(message.clone()));
            }
        });

        // Strict JSONL reader for stdout.
        let stop_flag = Arc::new(DetachFlag::new());
        if let Some(stdout) = stdout {
            let core = Arc::clone(&self.core);
            let flag = Arc::clone(&stop_flag);
            tokio::spawn(async move {
                // Strict JSONL decoder fed from the async child stream
                // (upstream: `attachJsonlLineReader(childProcess.stdout)`).
                use tokio::io::AsyncReadExt;
                let mut decoder = JsonlLineReader::new();
                let mut buffer = [0u8; 8192];
                let mut stdout = stdout;
                let forward = |core: &Arc<Core>, line: String| {
                    if !flag.is_detached() {
                        handle_line(core, &line);
                    }
                };
                loop {
                    match stdout.read(&mut buffer).await {
                        Ok(0) | Err(_) => break,
                        Ok(read) => decoder.feed(&buffer[..read], &mut |line| forward(&core, line)),
                    }
                }
                decoder.finish(&mut |line| forward(&core, line));
            });
        }
        *self.core.stop_reading.lock() = Some(stop_flag);

        // Wait a moment for process to initialize.
        tokio::time::sleep(Duration::from_millis(100)).await;

        if let Some(error) = self.core.exit_error.lock().clone() {
            return Err(error);
        }
        Ok(())
    }

    /// Stop the RPC agent process (`SIGTERM`, escalating after 1s).
    pub async fn stop(&self) {
        let Some(pid) = *self.core.child_pid.lock() else {
            return;
        };

        if let Some(stop_reading) = self.core.stop_reading.lock().take() {
            stop_reading.detach();
        }
        // `process.kill("SIGTERM")`; the exit watcher reaps the handle.
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGTERM);
        }
        // Wait for the watcher to observe the exit, escalating after 1s.
        if tokio::time::timeout(Duration::from_secs(1), self.core.exited.notified())
            .await
            .is_err()
        {
            unsafe {
                libc::kill(pid as libc::pid_t, libc::SIGKILL);
            }
            let _ = tokio::time::timeout(Duration::from_secs(1), self.core.exited.notified()).await;
        }

        *self.core.child_pid.lock() = None;
        *self.core.stdin.lock() = None;
        self.core.pending_requests.lock().clear();
    }

    /// Subscribe to agent events; returns the unsubscribe closure.
    pub fn on_event(&self, listener: RpcEventListener) -> Box<dyn FnOnce() + Send> {
        let id = self.core.listener_seq.fetch_add(1, Ordering::SeqCst);
        self.core.listeners.lock().push((id, listener));
        let core = Arc::clone(&self.core);
        Box::new(move || {
            core.listeners
                .lock()
                .retain(|(existing, _)| *existing != id);
        })
    }

    // =========================================================================
    // Command Methods
    // =========================================================================

    /// Send a prompt to the agent. Returns the disposition after acceptance;
    /// use [`RpcClient::on_event`] to receive streaming events.
    pub async fn prompt(
        &self,
        message: impl Into<String>,
        images: Option<JsonValue>,
        streaming_behavior: Option<&str>,
    ) -> Result<String, String> {
        let mut command = json!({ "type": "prompt", "message": message.into() });
        if let Some(images) = images {
            command["images"] = images;
        }
        if let Some(streaming_behavior) = streaming_behavior {
            command["streamingBehavior"] = json!(streaming_behavior);
        }
        let response = self.send(command).await?;
        disposition_of(response)
    }

    /// Queue a steering message to interrupt the agent mid-run.
    pub async fn steer(
        &self,
        message: impl Into<String>,
        images: Option<JsonValue>,
    ) -> Result<String, String> {
        let mut command = json!({ "type": "steer", "message": message.into() });
        if let Some(images) = images {
            command["images"] = images;
        }
        let response = self.send(command).await?;
        disposition_of(response)
    }

    /// Queue a follow-up message to be processed after the agent finishes.
    pub async fn follow_up(
        &self,
        message: impl Into<String>,
        images: Option<JsonValue>,
    ) -> Result<String, String> {
        let mut command = json!({ "type": "follow_up", "message": message.into() });
        if let Some(images) = images {
            command["images"] = images;
        }
        let response = self.send(command).await?;
        disposition_of(response)
    }

    /// Abort current operation.
    pub async fn abort(&self) -> Result<(), String> {
        self.send(json!({ "type": "abort" })).await.map(|_| ())
    }

    /// Clear queued steering and follow-up messages, returning their text.
    pub async fn clear_queue(&self) -> Result<ClearedQueue, String> {
        let response = self.send(json!({ "type": "clear_queue" })).await?;
        let data = response.into_data()?;
        serde_json::from_value(data).map_err(|error| error.to_string())
    }

    /// Start a new session, optionally with parent tracking.
    pub async fn new_session(&self, parent_session: Option<&str>) -> Result<Cancelled, String> {
        let response = self
            .send(json!({ "type": "new_session", "parentSession": parent_session }))
            .await?;
        let data = response.into_data()?;
        serde_json::from_value(data).map_err(|error| error.to_string())
    }

    /// Get current session state.
    pub async fn get_state(&self) -> Result<JsonValue, String> {
        let response = self.send(json!({ "type": "get_state" })).await?;
        response.into_data()
    }

    /// Set model by provider and ID.
    pub async fn set_model(&self, provider: &str, model_id: &str) -> Result<ModelRef, String> {
        let response = self
            .send(json!({ "type": "set_model", "provider": provider, "modelId": model_id }))
            .await?;
        let data = response.into_data()?;
        serde_json::from_value(data).map_err(|error| error.to_string())
    }

    /// Cycle to next model.
    pub async fn cycle_model(&self) -> Result<Option<CycledModel>, String> {
        let response = self.send(json!({ "type": "cycle_model" })).await?;
        let data = response.into_data()?;
        if data.is_null() {
            return Ok(None);
        }
        serde_json::from_value(data)
            .map(Some)
            .map_err(|error| error.to_string())
    }

    /// Get list of available models.
    pub async fn get_available_models(&self) -> Result<Vec<ModelInfo>, String> {
        let response = self.send(json!({ "type": "get_available_models" })).await?;
        let data = response.into_data()?;
        let models = data.get("models").cloned().unwrap_or(JsonValue::Null);
        serde_json::from_value(models).map_err(|error| error.to_string())
    }

    /// Set thinking level.
    pub async fn set_thinking_level(&self, level: AgentThinkingLevel) -> Result<(), String> {
        self.send(json!({ "type": "set_thinking_level", "level": level }))
            .await
            .map(|_| ())
    }

    /// Cycle thinking level.
    pub async fn cycle_thinking_level(&self) -> Result<Option<AgentThinkingLevel>, String> {
        let response = self.send(json!({ "type": "cycle_thinking_level" })).await?;
        let data = response.into_data()?;
        if data.is_null() {
            return Ok(None);
        }
        let level = data
            .get("level")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        parse_thinking_level(level).map(Some)
    }

    /// Get list of available thinking levels for the current model.
    pub async fn get_available_thinking_levels(&self) -> Result<Vec<AgentThinkingLevel>, String> {
        let response = self
            .send(json!({ "type": "get_available_thinking_levels" }))
            .await?;
        let data = response.into_data()?;
        let levels = data
            .get("levels")
            .and_then(JsonValue::as_array)
            .map(|levels| {
                levels
                    .iter()
                    .filter_map(|level| level.as_str())
                    .map(parse_thinking_level)
                    .collect::<Result<Vec<_>, _>>()
            })
            .unwrap_or_else(|| Ok(Vec::new()))?;
        Ok(levels)
    }

    /// Set steering mode.
    pub async fn set_steering_mode(&self, mode: &str) -> Result<(), String> {
        self.send(json!({ "type": "set_steering_mode", "mode": mode }))
            .await
            .map(|_| ())
    }

    /// Set follow-up mode.
    pub async fn set_follow_up_mode(&self, mode: &str) -> Result<(), String> {
        self.send(json!({ "type": "set_follow_up_mode", "mode": mode }))
            .await
            .map(|_| ())
    }

    /// Compact session context.
    pub async fn compact(&self, custom_instructions: Option<&str>) -> Result<JsonValue, String> {
        let response = self
            .send(json!({ "type": "compact", "customInstructions": custom_instructions }))
            .await?;
        response.into_data()
    }

    /// Set auto-compaction enabled/disabled.
    pub async fn set_auto_compaction(&self, enabled: bool) -> Result<(), String> {
        self.send(json!({ "type": "set_auto_compaction", "enabled": enabled }))
            .await
            .map(|_| ())
    }

    /// Set auto-retry enabled/disabled.
    pub async fn set_auto_retry(&self, enabled: bool) -> Result<(), String> {
        self.send(json!({ "type": "set_auto_retry", "enabled": enabled }))
            .await
            .map(|_| ())
    }

    /// Abort in-progress retry.
    pub async fn abort_retry(&self) -> Result<(), String> {
        self.send(json!({ "type": "abort_retry" }))
            .await
            .map(|_| ())
    }

    /// Execute a bash command.
    pub async fn bash(&self, command: impl Into<String>) -> Result<BashResult, String> {
        let response = self
            .send(json!({ "type": "bash", "command": command.into() }))
            .await?;
        let data = response.into_data()?;
        serde_json::from_value(data).map_err(|error| error.to_string())
    }

    /// Abort running bash command.
    pub async fn abort_bash(&self) -> Result<(), String> {
        self.send(json!({ "type": "abort_bash" })).await.map(|_| ())
    }

    /// Get session statistics.
    pub async fn get_session_stats(&self) -> Result<JsonValue, String> {
        let response = self.send(json!({ "type": "get_session_stats" })).await?;
        response.into_data()
    }

    /// Export session to HTML.
    pub async fn export_html(&self, output_path: Option<&str>) -> Result<ExportedHtml, String> {
        let response = self
            .send(json!({ "type": "export_html", "outputPath": output_path }))
            .await?;
        let data = response.into_data()?;
        serde_json::from_value(data).map_err(|error| error.to_string())
    }

    /// Switch to a different session file.
    pub async fn switch_session(&self, session_path: &str) -> Result<Cancelled, String> {
        let response = self
            .send(json!({ "type": "switch_session", "sessionPath": session_path }))
            .await?;
        let data = response.into_data()?;
        serde_json::from_value(data).map_err(|error| error.to_string())
    }

    /// Fork from a specific message.
    pub async fn fork(&self, entry_id: &str) -> Result<Forked, String> {
        let response = self
            .send(json!({ "type": "fork", "entryId": entry_id }))
            .await?;
        let data = response.into_data()?;
        serde_json::from_value(data).map_err(|error| error.to_string())
    }

    /// Clone the current active branch into a new session.
    pub async fn clone_session(&self) -> Result<Cancelled, String> {
        let response = self.send(json!({ "type": "clone" })).await?;
        let data = response.into_data()?;
        serde_json::from_value(data).map_err(|error| error.to_string())
    }

    /// Get messages available for forking.
    pub async fn get_fork_messages(&self) -> Result<Vec<ForkMessage>, String> {
        let response = self.send(json!({ "type": "get_fork_messages" })).await?;
        let data = response.into_data()?;
        let messages = data.get("messages").cloned().unwrap_or(JsonValue::Null);
        serde_json::from_value(messages).map_err(|error| error.to_string())
    }

    /// Get session entries in append order, optionally only those after
    /// `since`.
    pub async fn get_entries(&self, since: Option<&str>) -> Result<Entries, String> {
        let response = self
            .send(json!({ "type": "get_entries", "since": since }))
            .await?;
        let data = response.into_data()?;
        serde_json::from_value(data).map_err(|error| error.to_string())
    }

    /// Get the session entry tree.
    pub async fn get_tree(&self) -> Result<JsonValue, String> {
        let response = self.send(json!({ "type": "get_tree" })).await?;
        response.into_data()
    }

    /// Get text of last assistant message.
    pub async fn get_last_assistant_text(&self) -> Result<Option<String>, String> {
        let response = self
            .send(json!({ "type": "get_last_assistant_text" }))
            .await?;
        let data = response.into_data()?;
        let text = data.get("text").cloned().unwrap_or(JsonValue::Null);
        serde_json::from_value(text).map_err(|error| error.to_string())
    }

    /// Set the session display name.
    pub async fn set_session_name(&self, name: impl Into<String>) -> Result<(), String> {
        self.send(json!({ "type": "set_session_name", "name": name.into() }))
            .await
            .map(|_| ())
    }

    /// Get all messages in the session.
    pub async fn get_messages(&self) -> Result<Vec<pi_agent::types::AgentMessage>, String> {
        let response = self.send(json!({ "type": "get_messages" })).await?;
        let data = response.into_data()?;
        let messages = data.get("messages").cloned().unwrap_or(JsonValue::Null);
        serde_json::from_value(messages).map_err(|error| error.to_string())
    }

    /// Get available commands (extension commands, prompt templates, skills).
    pub async fn get_commands(&self) -> Result<JsonValue, String> {
        let response = self.send(json!({ "type": "get_commands" })).await?;
        response.into_data()
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Wait for agent to become idle (no streaming); resolves on
    /// `agent_settled`.
    pub async fn wait_for_idle(&self, timeout: Duration) -> Result<(), String> {
        let (sender, receiver) = oneshot::channel::<()>();
        let sender = Mutex::new(Some(sender));
        let listener: RpcEventListener = Arc::new(move |event| {
            if event.get("type").and_then(JsonValue::as_str) == Some("agent_settled") {
                if let Some(sender) = sender.lock().take() {
                    let _ = sender.send(());
                }
            }
        });
        let unsubscribe = self.on_event(listener);
        let outcome = match tokio::time::timeout(timeout, receiver).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) | Err(_) => Err(format!(
                "Timeout waiting for agent to become idle. Stderr: {}",
                self.get_stderr()
            )),
        };
        unsubscribe();
        outcome
    }

    /// Collect events until agent becomes idle.
    pub async fn collect_events(&self, timeout: Duration) -> Result<Vec<JsonValue>, String> {
        let events: Arc<Mutex<Vec<JsonValue>>> = Arc::new(Mutex::new(Vec::new()));
        let (sender, receiver) = oneshot::channel::<Vec<JsonValue>>();
        let listener: RpcEventListener = {
            let events = Arc::clone(&events);
            let sender = Mutex::new(Some(sender));
            Arc::new(move |event| {
                events.lock().push(event.clone());
                if event.get("type").and_then(JsonValue::as_str) == Some("agent_settled") {
                    if let Some(sender) = sender.lock().take() {
                        let _ = sender.send(events.lock().clone());
                    }
                }
            })
        };
        let unsubscribe = self.on_event(listener);
        let outcome = match tokio::time::timeout(timeout, receiver).await {
            Ok(Ok(events)) => Ok(events),
            Ok(Err(_)) | Err(_) => Err(format!(
                "Timeout collecting events. Stderr: {}",
                self.get_stderr()
            )),
        };
        unsubscribe();
        outcome
    }

    /// Send prompt and wait for completion, returning all events.
    pub async fn prompt_and_wait(
        &self,
        message: impl Into<String>,
        images: Option<JsonValue>,
        timeout: Duration,
    ) -> Result<Vec<JsonValue>, String> {
        let collect = self.collect_events(timeout);
        let prompt = self.prompt(message, images, None);
        let (collect, prompt) = tokio::join!(collect, prompt);
        prompt?;
        collect
    }

    // =========================================================================
    // Internal
    // =========================================================================

    /// `send`: assign the correlation id, forward the command, await the
    /// response with a 30s timeout.
    pub(crate) async fn send(&self, command: JsonValue) -> Result<RpcResponse, String> {
        #[cfg(test)]
        #[cfg(test)]
        if let Some(override_fn) = self.core.send_override.lock().as_ref() {
            return override_fn(command).await;
        }

        if self.core.child_pid.lock().is_none() || self.core.stdin.lock().is_none() {
            return Err("Client not started".to_string());
        }
        if let Some(exit_error) = self.core.exit_error.lock().clone() {
            return Err(exit_error);
        }
        if self.core.child.lock().await.is_none() {
            // The exit watcher took the handle: the process already exited
            // and recorded the error.
            if let Some(exit_error) = self.core.exit_error.lock().clone() {
                return Err(exit_error);
            }
        }

        let id = format!(
            "req_{}",
            self.core.request_id.fetch_add(1, Ordering::SeqCst) + 1
        );
        let mut full_command = command;
        if let Some(object) = full_command.as_object_mut() {
            object.insert("id".to_string(), json!(id));
        }

        let (sender, receiver) = oneshot::channel::<Result<RpcResponse, String>>();
        self.core.pending_requests.lock().insert(id.clone(), sender);

        let write_result = match self.core.stdin.lock().as_mut() {
            Some(stdin) => {
                stdin
                    .write_all(serialize_json_line(&full_command).as_bytes())
                    .await
            }
            None => Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "stdin closed",
            )),
        };
        if let Err(error) = write_result {
            if let Some(pending) = self.core.pending_requests.lock().remove(&id) {
                let _ = pending.send(Err(format!(
                    "Agent process stdin error: {error}. Stderr: {}",
                    self.get_stderr()
                )));
            }
        }

        let stderr = self.get_stderr();
        let command_type = full_command
            .get("type")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .to_string();
        match tokio::time::timeout(Duration::from_secs(30), receiver).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err("Agent process dropped the response channel".to_string()),
            Err(_) => {
                self.core.pending_requests.lock().remove(&id);
                Err(format!(
                    "Timeout waiting for response to {command_type}. Stderr: {stderr}"
                ))
            }
        }
    }
}

/// `handleLine`: route responses to pending requests, events to listeners.
fn handle_line(core: &Core, line: &str) {
    let Ok(data) = serde_json::from_str::<JsonValue>(line) else {
        // Ignore non-JSON lines.
        return;
    };

    // Check if it's a response to a pending request.
    let is_response = data.get("type").and_then(JsonValue::as_str) == Some("response");
    if is_response {
        if let Some(id) = data
            .get("id")
            .and_then(JsonValue::as_str)
            .map(str::to_string)
        {
            let pending = core.pending_requests.lock().remove(&id);
            if let Some(pending) = pending {
                let response: Result<RpcResponse, String> =
                    serde_json::from_value(data).map_err(|error| error.to_string());
                let _ = pending.send(response);
                return;
            }
        }
    }

    // Otherwise it's an event. Iterate a snapshot so listeners that
    // unsubscribe during dispatch do not cause later listeners to miss this
    // event.
    for (_, listener) in core.listeners.lock().clone() {
        listener(&data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Client whose transport is replaced by a recording stub (upstream:
    /// `privateClient.send = vi.fn(...)`; D94).
    fn recording_client_with(response: JsonValue) -> (RpcClient, Arc<Mutex<Vec<JsonValue>>>) {
        let client = RpcClient::new(RpcClientOptions::default());
        let sent: Arc<Mutex<Vec<JsonValue>>> = Arc::new(Mutex::new(Vec::new()));
        let sent_for_fn = Arc::clone(&sent);
        let response_for_fn = response;
        client.set_send_override(Box::new(move |command| {
            sent_for_fn.lock().push(command);
            let response: RpcResponse = serde_json::from_value(response_for_fn.clone()).unwrap();
            Box::pin(async move { Ok(response) })
        }));
        (client, sent)
    }

    #[tokio::test]
    async fn sends_the_clear_queue_rpc_command() {
        let (client, sent) = recording_client_with(json!({
            "type": "response",
            "command": "clear_queue",
            "success": true,
            "data": { "steering": ["Change direction"], "followUp": ["Summarize when finished"] },
        }));

        let result = client.clear_queue().await.unwrap();

        assert_eq!(*sent.lock(), vec![json!({ "type": "clear_queue" })]);
        assert_eq!(
            result,
            ClearedQueue {
                steering: vec!["Change direction".to_string()],
                follow_up: vec!["Summarize when finished".to_string()],
            }
        );
    }

    #[tokio::test]
    async fn sends_the_clone_rpc_command() {
        let (client, sent) = recording_client_with(json!({
            "type": "response",
            "command": "clone",
            "success": true,
            "data": { "cancelled": false },
        }));

        let result = client.clone_session().await.unwrap();

        assert_eq!(*sent.lock(), vec![json!({ "type": "clone" })]);
        assert_eq!(result, Cancelled { cancelled: false });
    }

    #[tokio::test]
    async fn rejects_an_in_flight_request_when_the_child_process_exits() {
        let dir = std::env::temp_dir().join(format!(
            "pi-rpc-client-exit-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("child.mjs");
        std::fs::write(
			&script,
			"process.stdin.once(\"data\", () => {\n\tprocess.exit(43);\n});\nprocess.stdin.resume();\n",
		)
		.unwrap();

        let client = RpcClient::new(RpcClientOptions {
            cli_path: Some(script.to_string_lossy().into_owned()),
            ..Default::default()
        });
        client.start().await.unwrap();

        let error = client.get_commands().await.unwrap_err();

        assert!(
            error.contains("Agent process exited (code=43 signal=null)"),
            "got: {error}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
