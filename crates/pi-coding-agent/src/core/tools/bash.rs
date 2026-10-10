//! Port of `upstream/pi/packages/coding-agent/src/core/tools/bash.ts`.
//!
//! Deviations:
//! - Node `child_process.spawn` options (cwd/env/detached/windowsHide/stdio)
//!   become `tokio::process::Command` configuration in this module; the
//!   [`crate::utils::child_process::spawn_process`] seam lacks those knobs.
//! - `os.constants.signals` mapping (128 + signal number) reuses the
//!   semantics of [`crate::utils::child_process`]: `wait_for_child_process`
//!   already folds signal terminations into `ChildResult.code` as
//!   `128 + signal`; a `None` code maps to `1` (upstream
//!   `exitCode ?? (signalCode ? 128 + sig : 1)`).
//! - `getShellConfig` module-function spying in upstream tests becomes
//!   constructor injection: [`create_local_shell_operations`] /
//!   [`create_local_bash_operations`] take an injected
//!   `Arc<dyn Fn() -> Result<ShellConfig, String> + Send + Sync>` resolver.
//! - Test-command substitutions (`process.execPath` → `sh`) live in the
//!   central test suite, not here.
//! - Renderers are deferred (D32); [`BASH_UPDATE_THROTTLE_MS`] is defined
//!   here with its upstream value from `renderers/bash.ts`.
//! - This module defines its own [`BashOperations`] trait. Note that
//!   `crate::core::extensions::types` currently holds a `BashOperations`
//!   stand-in for `user_bash`; that stand-in gets replaced when the
//!   extensions runner ports.
//! - `spawn()` errors surface as `Err(io::Error::to_string())` (upstream
//!   throws the Node spawn error).
//! - `timeout:${timeout}` renders the raw seconds input with JavaScript
//!   number formatting via [`format_timeout_value`].

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use pi_agent::{AgentTool, AgentToolResult, AgentToolUpdateCallback, BoxFuture};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::core::extensions::types::{ExtensionContext, ExtensionToolContext, ToolDefinition};
use crate::core::tools::output_accumulator::{
    OutputAccumulator, OutputAccumulatorOptions, OutputSnapshot,
};
use crate::core::tools::tool_definition_wrapper::wrap_tool_definition;
use crate::core::tools::truncate::{
    format_size, TruncatedBy, TruncationResult, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES,
};
use crate::utils::abort::AbortSignal;
use crate::utils::child_process::wait_for_child_process;
use crate::utils::shell::{
    get_shell_config, get_shell_env, kill_process_tree, track_detached_child_pid,
    untrack_detached_child_pid, CommandTransport, ShellConfig,
};
use pi_ai::types::{ConstrainedSamplingConfig, ConstrainedSamplingSetting, StrictSchemaMode};

const MAX_TIMEOUT_MS: f64 = 2_147_483_647.0;
/// Output limit of `structuredContent.output`, which programmatic callers such as codemode scripts receive.
const STRUCTURED_OUTPUT_MAX_BYTES: usize = 1024 * 1024;
const MAX_TIMEOUT_SECONDS: f64 = MAX_TIMEOUT_MS / 1000.0;

/// Throttle for streaming tool updates (`BASH_UPDATE_THROTTLE_MS`, upstream
/// value from `core/tools/renderers/bash.ts`; renderers deferred, D32).
pub(crate) const BASH_UPDATE_THROTTLE_MS: u64 = 100;

fn resolve_timeout_ms(timeout: Option<f64>) -> Result<Option<f64>, String> {
    let Some(timeout) = timeout else {
        return Ok(None);
    };
    if !timeout.is_finite() || timeout <= 0.0 {
        return Err("Invalid timeout: must be a finite number of seconds".to_string());
    }
    let timeout_ms = timeout * 1000.0;
    if timeout_ms > MAX_TIMEOUT_MS {
        return Err(format!(
            "Invalid timeout: maximum is {} seconds",
            MAX_TIMEOUT_SECONDS
        ));
    }
    Ok(Some(timeout_ms))
}

/// Render a timeout value the way JavaScript stringifies numbers in
/// `timeout:${timeout}` (integers without a decimal point).
fn format_timeout_value(timeout: f64) -> String {
    if timeout.fract() == 0.0 && timeout.abs() < 1e15 {
        format!("{}", timeout as i64)
    } else {
        format!("{timeout}")
    }
}

/// `bashSchema` (`Type.Object`) — JSON Schema for the tool parameters.
pub fn bash_tool_input_schema() -> JsonValue {
    json!({
        "type": "object",
        "properties": {
            "command": { "type": "string", "description": "Shell command to execute" },
            "timeout": { "type": "number", "description": "Timeout in seconds (optional, no default timeout)" }
        },
        "required": ["command"]
    })
}

/// `bashToolSystemPromptContribution`.
pub const BASH_TOOL_SYSTEM_PROMPT_SNIPPET: &str = "Execute bash commands (ls, grep, find, etc.)";
pub const BASH_TOOL_SYSTEM_PROMPT_GUIDELINES: &[&str] =
    &["You can inspect PI_* environment variables for current model and session details."];

/// `BashToolInput` (`Static<typeof bashSchema>`).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BashToolInput {
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<f64>,
}

/// `bashOutputSchema` — JSON Schema for the machine-readable result.
/// Property names keep the upstream snake_case spelling.
pub fn bash_tool_output_schema() -> JsonValue {
    json!({
        "type": "object",
        "properties": {
            "output": { "type": "string", "description": "Combined stdout and stderr, possibly truncated" },
            "truncated": { "type": "boolean" },
            "full_output_path": { "type": "string", "description": "Full output, when truncated" },
            "exit_code": { "type": "number" },
            "wall_time_seconds": { "type": "number" }
        },
        "required": ["output", "truncated", "exit_code", "wall_time_seconds"]
    })
}

/// `BashToolDetails`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashToolDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
}

/// Pluggable operations for the bash tool.
///
/// Override these to delegate command execution to remote systems (for example SSH).
pub trait BashOperations: Send + Sync {
    /// Execute a command and stream output.
    ///
    /// Report signal terminations as `128 + signal number`; a `None` exit
    /// code is treated as a failed command.
    fn exec(
        &self,
        command: &str,
        cwd: &str,
        options: BashExecOptions,
    ) -> BoxFuture<'static, Result<BashExecOutcome, String>>;
}

/// Options of [`BashOperations::exec`].
pub struct BashExecOptions {
    /// Streamed stdout/stderr chunks.
    pub on_data: Arc<dyn Fn(Vec<u8>) + Send + Sync>,
    /// Abort signal; when aborted the process tree is killed and `"aborted"` is reported.
    pub signal: Option<AbortSignal>,
    /// Timeout in seconds.
    pub timeout: Option<f64>,
    /// Environment override; `None` falls back to [`get_shell_env`].
    pub env: Option<BTreeMap<String, String>>,
}

/// Outcome of [`BashOperations::exec`].
pub struct BashExecOutcome {
    pub exit_code: Option<i32>,
}

/// Adapter implementing [`BashOperations`] from a closure.
pub(crate) struct FnOperations<F>(pub(crate) F)
where
    F: Fn(&str, &str, BashExecOptions) -> BoxFuture<'static, Result<BashExecOutcome, String>>
        + Send
        + Sync;

impl<F> BashOperations for FnOperations<F>
where
    F: Fn(&str, &str, BashExecOptions) -> BoxFuture<'static, Result<BashExecOutcome, String>>
        + Send
        + Sync,
{
    fn exec(
        &self,
        command: &str,
        cwd: &str,
        options: BashExecOptions,
    ) -> BoxFuture<'static, Result<BashExecOutcome, String>> {
        (self.0)(command, cwd, options)
    }
}

/// Shared process execution used by the built-in shell tools
/// (`createLocalShellOperations`).
pub fn create_local_shell_operations(
    shell_name: &str,
    resolve_shell_config: Arc<dyn Fn() -> Result<ShellConfig, String> + Send + Sync>,
) -> Arc<dyn BashOperations> {
    let shell_name = shell_name.to_string();
    Arc::new(FnOperations(
        move |command: &str, cwd: &str, options: BashExecOptions| {
            let shell_name = shell_name.clone();
            let resolve_shell_config = Arc::clone(&resolve_shell_config);
            let command = command.to_string();
            let cwd = cwd.to_string();
            Box::pin(async move {
                let timeout_ms = resolve_timeout_ms(options.timeout)?;
                if options.signal.as_ref().is_some_and(AbortSignal::aborted) {
                    return Err("aborted".to_string());
                }
                let shell_config = resolve_shell_config()?;
                if std::fs::metadata(&cwd).is_err() {
                    return Err(format!(
					"Working directory does not exist: {cwd}\nCannot execute {shell_name} commands."
				));
                }

                let command_from_stdin =
                    shell_config.command_transport == Some(CommandTransport::Stdin);
                let mut cmd = tokio::process::Command::new(&shell_config.shell);
                if command_from_stdin {
                    cmd.args(&shell_config.args);
                } else {
                    cmd.args(shell_config.args.iter().chain(std::iter::once(&command)));
                }
                cmd.current_dir(&cwd)
                    .env_clear()
                    .envs(
                        options
                            .env
                            .clone()
                            .unwrap_or_else(|| get_shell_env().into_iter().collect()),
                    )
                    .stdin(if command_from_stdin {
                        std::process::Stdio::piped()
                    } else {
                        std::process::Stdio::null()
                    })
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .kill_on_drop(false);
                #[cfg(unix)]
                {
                    // `detached: true` on POSIX.
                    cmd.process_group(0);
                }
                #[cfg(windows)]
                {
                    const CREATE_NO_WINDOW: u32 = 0x0800_0000; // `windowsHide: true`
                    cmd.creation_flags(CREATE_NO_WINDOW);
                }

                let mut child = cmd.spawn().map_err(|e| e.to_string())?;
                if command_from_stdin {
                    if let Some(mut stdin) = child.stdin.take() {
                        // Ignore write errors the way the upstream stdin "error" no-op handler does.
                        let _ = stdin.write_all(command.as_bytes()).await;
                        let _ = stdin.shutdown().await;
                    }
                }
                let pid = child.id();
                if let Some(pid) = pid {
                    track_detached_child_pid(pid);
                }

                // Stream stdout and stderr through `on_data`.
                let on_data = options.on_data.clone();
                let mut readers = Vec::new();
                let mut streams: Vec<Box<dyn tokio::io::AsyncRead + Unpin + Send>> = Vec::new();
                if let Some(stdout) = child.stdout.take() {
                    streams.push(Box::new(stdout));
                }
                if let Some(stderr) = child.stderr.take() {
                    streams.push(Box::new(stderr));
                }
                for stream in streams {
                    let on_data = Arc::clone(&on_data);
                    readers.push(tokio::spawn(async move {
                        let mut stream = stream;
                        let mut buf = [0u8; 65_536];
                        loop {
                            match stream.read(&mut buf).await {
                                Ok(0) | Err(_) => break,
                                Ok(n) => on_data(buf[..n].to_vec()),
                            }
                        }
                    }));
                }

                // Wait for termination; kill the process tree on abort or timeout.
                let mut timed_out = false;
                let mut wait_fut = Box::pin(wait_for_child_process(child));
                let mut timeout_sleep = timeout_ms
                    .map(|ms| Box::pin(tokio::time::sleep(Duration::from_millis(ms as u64))));
                let mut result: Option<std::io::Result<crate::utils::child_process::ChildResult>> =
                    None;
                while result.is_none() {
                    tokio::select! {
                        biased;
                        res = &mut wait_fut => result = Some(res),
                        _ = async {
                            match &options.signal {
                                Some(signal) => signal.aborted_future().await,
                                None => std::future::pending().await,
                            }
                        }, if options.signal.as_ref().is_some_and(|s| !s.aborted()) => {
                            if let Some(pid) = pid {
                                kill_process_tree(pid);
                            }
                        },
                        _ = async {
                            match timeout_sleep.as_mut() {
                                Some(sleep) => sleep.as_mut().await,
                                None => std::future::pending().await,
                            }
                        }, if !timed_out && timeout_ms.is_some() => {
                            timed_out = true;
                            if let Some(pid) = pid {
                                kill_process_tree(pid);
                            }
                        },
                    }
                }

                // Drain the readers so all streamed data is delivered before returning.
                for reader in readers {
                    let _ = reader.await;
                }

                let outcome = (|| {
                    // Handle shell spawn/wait errors.
                    let result = match result {
                        Some(Ok(result)) => result,
                        Some(Err(e)) => return Err(e.to_string()),
                        None => return Ok(None),
                    };
                    if options.signal.as_ref().is_some_and(AbortSignal::aborted) {
                        return Err("aborted".to_string());
                    }
                    if timed_out {
                        return Err(format!(
                            "timeout:{}",
                            format_timeout_value(options.timeout.unwrap_or_default())
                        ));
                    }
                    // A signal-killed shell has no exit code. `ChildResult.code` already
                    // carries the 128 + signal convention; a remaining `None` maps to 1 so
                    // callers do not mistake the termination for a successful command.
                    Ok(result.code.or(Some(1)))
                })();
                if let Some(pid) = pid {
                    untrack_detached_child_pid(pid);
                }
                let exit_code = outcome?;
                Ok(BashExecOutcome { exit_code })
            })
        },
    )) as Arc<dyn BashOperations>
}

/// Create bash operations using pi's built-in local shell execution backend.
///
/// This is useful for extensions that intercept user_bash and still want pi's
/// standard local shell behavior while wrapping or rewriting commands.
pub fn create_local_bash_operations(shell_path: Option<String>) -> Arc<dyn BashOperations> {
    create_local_shell_operations(
        "bash",
        Arc::new(move || get_shell_config(shell_path.as_deref())),
    )
}

/// `BashSpawnContext`.
#[derive(Debug, Clone)]
pub struct BashSpawnContext {
    pub command: String,
    pub cwd: String,
    pub env: BTreeMap<String, String>,
}

/// `BashSpawnHook`.
pub type BashSpawnHook = Arc<dyn Fn(BashSpawnContext) -> BashSpawnContext + Send + Sync>;

fn resolve_spawn_context(
    command: String,
    cwd: String,
    spawn_hook: Option<BashSpawnHook>,
    expose_session_environment: bool,
    ctx: Option<&ExtensionContext>,
) -> BashSpawnContext {
    let mut env: BTreeMap<String, String> = get_shell_env().into_iter().collect();
    for key in [
        "PI_SESSION_ID",
        "PI_SESSION_FILE",
        "PI_PROVIDER",
        "PI_MODEL",
        "PI_REASONING_LEVEL",
    ] {
        env.remove(key);
    }
    if expose_session_environment {
        if let Some(ctx) = ctx {
            env.insert(
                "PI_SESSION_ID".to_string(),
                ctx.session_manager.lock().get_session_id().to_string(),
            );
            if let Some(session_file) = ctx
                .session_manager
                .lock()
                .get_session_file()
                .map(str::to_string)
            {
                env.insert("PI_SESSION_FILE".to_string(), session_file.to_string());
            }
            if let Some(model) = &ctx.model {
                env.insert("PI_PROVIDER".to_string(), model.base.provider.clone());
                env.insert("PI_MODEL".to_string(), model.base.id.clone());
            }
            if let Some(level) = ctx.thinking_level {
                // Upstream assigns the string value of the thinking level.
                let serialized = serde_json::to_value(level)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_string))
                    .unwrap_or_default();
                env.insert("PI_REASONING_LEVEL".to_string(), serialized);
            }
        }
    }
    let base_context = BashSpawnContext { command, cwd, env };
    match spawn_hook {
        Some(spawn_hook) => spawn_hook(base_context),
        None => base_context,
    }
}

/// `BashToolOptions`.
#[derive(Default, Clone)]
pub struct BashToolOptions {
    /// Custom operations for command execution. Default: local shell
    pub operations: Option<Arc<dyn BashOperations>>,
    /// Command prefix prepended to every command (for example shell setup commands)
    pub command_prefix: Option<String>,
    /// Optional explicit shell path from settings
    pub shell_path: Option<String>,
    /// Expose current Pi session metadata as PI_* environment variables. Default: true
    pub expose_session_environment: Option<bool>,
    /// Hook to adjust command, cwd, or env before execution
    pub spawn_hook: Option<BashSpawnHook>,
}

/// `ShellToolConfig` (lifetime-free).
pub struct ShellToolConfig {
    pub name: &'static str,
    pub label: &'static str,
    pub shell_name: &'static str,
    pub prompt: &'static str,
    pub prompt_snippet: &'static str,
    pub prompt_guidelines: Option<&'static [&'static str]>,
    pub temp_file_prefix: &'static str,
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or_default()
}

/// Throttle state for streaming updates (upstream `updateTimer`/`updateDirty`/
/// `lastUpdateAt` closure variables).
struct UpdateThrottle {
    update_dirty: AtomicBool,
    last_update_at: Mutex<f64>,
    update_timer: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl UpdateThrottle {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            update_dirty: AtomicBool::new(false),
            last_update_at: Mutex::new(0.0),
            update_timer: Mutex::new(None),
        })
    }

    fn clear_timer(&self) {
        if let Some(handle) = self.update_timer.lock().take() {
            handle.abort();
        }
    }
}

async fn emit_output_update(
    output: &Arc<parking_lot::Mutex<OutputAccumulator>>,
    on_update: &AgentToolUpdateCallback,
    throttle: &UpdateThrottle,
) {
    if !throttle.update_dirty.swap(false, Ordering::SeqCst) {
        return;
    }
    *throttle.last_update_at.lock() = now_ms();
    let snapshot = output.lock().snapshot(true);
    let details = BashToolDetails {
        truncation: if snapshot.truncation.truncated {
            Some(snapshot.truncation.clone())
        } else {
            None
        },
        full_output_path: snapshot.full_output_path.clone(),
    };
    on_update(AgentToolResult::new(
        vec![pi_ai::types::UserContent::from(
            pi_ai::types::TextContent::new(snapshot.content),
        )],
        serde_json::to_value(&details).unwrap_or(JsonValue::Null),
    ));
}

async fn schedule_output_update(
    output: &Arc<parking_lot::Mutex<OutputAccumulator>>,
    on_update: &AgentToolUpdateCallback,
    throttle: &Arc<UpdateThrottle>,
) {
    throttle.update_dirty.store(true, Ordering::SeqCst);
    let delay = BASH_UPDATE_THROTTLE_MS as f64 - (now_ms() - *throttle.last_update_at.lock());
    if delay <= 0.0 {
        throttle.clear_timer();
        emit_output_update(output, on_update, throttle).await;
        return;
    }
    let mut timer = throttle.update_timer.lock();
    if timer.is_none() {
        let throttle = Arc::clone(throttle);
        let output = Arc::clone(output);
        let on_update = Arc::clone(on_update);
        *timer = Some(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs_f64(delay / 1000.0)).await;
            *throttle.update_timer.lock() = None;
            emit_output_update(&output, &on_update, &throttle).await;
        }));
    }
}

/// `finishOutput`: stop accepting output, flush the final throttled update,
/// take a snapshot, and close the temp file.
async fn finish_output(
    output: &Arc<parking_lot::Mutex<OutputAccumulator>>,
    on_update: Option<&AgentToolUpdateCallback>,
    throttle: &UpdateThrottle,
) -> OutputSnapshot {
    output.lock().finish();
    throttle.clear_timer();
    if let Some(on_update) = on_update {
        emit_output_update(output, on_update, throttle).await;
    }
    let mut guard = output.lock();
    let snapshot = guard.snapshot(true);
    guard.close_temp_file();
    snapshot
}

/// `formatOutput`.
fn format_output(
    output: &Arc<parking_lot::Mutex<OutputAccumulator>>,
    snapshot: &OutputSnapshot,
    empty_text: &str,
) -> String {
    let truncation = &snapshot.truncation;
    let mut text = if snapshot.content.is_empty() {
        empty_text.to_string()
    } else {
        snapshot.content.clone()
    };
    if truncation.truncated {
        let full_output_path = snapshot.full_output_path.as_deref().unwrap_or("undefined");
        let start_line = truncation.total_lines - truncation.output_lines + 1;
        let end_line = truncation.total_lines;
        if truncation.last_line_partial {
            let last_line_size = format_size(output.lock().get_last_line_bytes());
            text += &format!(
                "\n\n[Showing last {} of line {} (line is {}). Full output: {}]",
                format_size(truncation.output_bytes),
                end_line,
                last_line_size,
                full_output_path
            );
        } else if truncation.truncated_by == Some(TruncatedBy::Lines) {
            text += &format!(
                "\n\n[Showing lines {start_line}-{end_line} of {}. Full output: {}]",
                truncation.total_lines, full_output_path
            );
        } else {
            text += &format!(
                "\n\n[Showing lines {start_line}-{end_line} of {} ({} limit). Full output: {}]",
                truncation.total_lines,
                format_size(DEFAULT_MAX_BYTES),
                full_output_path
            );
        }
    }
    text
}

fn details_for(snapshot: &OutputSnapshot) -> Option<BashToolDetails> {
    if snapshot.truncation.truncated {
        Some(BashToolDetails {
            truncation: Some(snapshot.truncation.clone()),
            full_output_path: snapshot.full_output_path.clone(),
        })
    } else {
        // Upstream `details: undefined` serializes as JSON null.
        None
    }
}

fn append_status(text: &str, status: &str) -> String {
    if text.is_empty() {
        status.to_string()
    } else {
        format!("{text}\n\n{status}")
    }
}

/// `createShellToolDefinition`.
pub fn create_shell_tool_definition(
    cwd: &str,
    config: ShellToolConfig,
    options: Option<BashToolOptions>,
) -> ToolDefinition {
    let options = options.unwrap_or_default();
    let ops = options
        .operations
        .unwrap_or_else(|| create_local_bash_operations(options.shell_path.clone()));
    let command_prefix = options.command_prefix;
    let expose_session_environment = options.expose_session_environment.unwrap_or(true);
    let spawn_hook = options.spawn_hook;
    let cwd = cwd.to_string();
    let temp_file_prefix = config.temp_file_prefix;

    let mut definition = ToolDefinition::new(
		config.name.to_string(),
		config.label.to_string(),
		format!(
			"Execute a {} command in the current working directory. Returns stdout and stderr. Output is truncated to last {} lines or {}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
			config.shell_name,
			DEFAULT_MAX_LINES,
			DEFAULT_MAX_BYTES / 1024
		),
		bash_tool_input_schema(),
	);
    definition.prompt_snippet = Some(config.prompt_snippet.to_string());
    definition.prompt_guidelines = if expose_session_environment {
        config
            .prompt_guidelines
            .map(|guidelines| guidelines.iter().map(|s| s.to_string()).collect())
    } else {
        None
    };
    definition.output_schema = Some(bash_tool_output_schema());
    definition.constrained_sampling = Some(ConstrainedSamplingSetting::Config(
        ConstrainedSamplingConfig::JsonSchema {
            strict: StrictSchemaMode::Prefer,
        },
    ));

    definition.execute = Arc::new(
        move |_tool_call_id: String,
              args: JsonValue,
              signal: Option<AbortSignal>,
              on_update: Option<AgentToolUpdateCallback>,
              ctx: Option<ExtensionToolContext>| {
            let ops = Arc::clone(&ops);
            let command_prefix = command_prefix.clone();
            let spawn_hook = spawn_hook.clone();
            let cwd = cwd.clone();
            let temp_file_prefix = temp_file_prefix.to_string();
            Box::pin(async move {
                let input: BashToolInput =
                    serde_json::from_value(args).map_err(|e| format!("Invalid arguments: {e}"))?;
                let resolved_command = match &command_prefix {
                    Some(prefix) => format!("{prefix}\n{}", input.command),
                    None => input.command.clone(),
                };
                let ctx: Option<&ExtensionContext> = ctx.as_deref();
                let effective_cwd = ctx
                    .map(|c| c.cwd.clone())
                    .filter(|c| !c.is_empty())
                    .unwrap_or(cwd);
                let spawn_context = resolve_spawn_context(
                    resolved_command,
                    effective_cwd,
                    spawn_hook,
                    expose_session_environment,
                    ctx,
                );

                let output = Arc::new(parking_lot::Mutex::new(OutputAccumulator::new(
                    OutputAccumulatorOptions {
                        temp_file_prefix: Some(temp_file_prefix),
                        ..Default::default()
                    },
                )));
                let accepting_output = Arc::new(AtomicBool::new(true));
                let throttle = UpdateThrottle::new();

                if let Some(on_update) = &on_update {
                    on_update(AgentToolResult::new(Vec::new(), JsonValue::Null));
                }

                let started_at = Instant::now();

                // `handleData` gate + append + throttled update scheduling.
                // The append is synchronous: upstream's `onData` handlers run
                // on the same turn of the event loop, so data pushed while
                // `ops.exec` runs is always captured before the final
                // snapshot (async spawns would race the finish path).
                let handle_data = {
                    let output = Arc::clone(&output);
                    let accepting_output = Arc::clone(&accepting_output);
                    let throttle = Arc::clone(&throttle);
                    let on_update = on_update.clone();
                    Arc::new(move |data: Vec<u8>| {
                        if !accepting_output.load(Ordering::SeqCst) {
                            return;
                        }
                        output.lock().append(&data);
                        if let Some(on_update) = on_update.clone() {
                            let output = Arc::clone(&output);
                            let throttle = Arc::clone(&throttle);
                            tokio::spawn(async move {
                                schedule_output_update(&output, &on_update, &throttle).await;
                            });
                        }
                    })
                };

                let exec_result = ops
                    .exec(
                        &spawn_context.command,
                        &spawn_context.cwd,
                        BashExecOptions {
                            on_data: handle_data,
                            signal: signal.clone(),
                            timeout: input.timeout,
                            env: Some(spawn_context.env.clone()),
                        },
                    )
                    .await;

                let exit_code = match exec_result {
                    Ok(outcome) => outcome.exit_code,
                    Err(message) => {
                        let snapshot = finish_output(&output, on_update.as_ref(), &throttle).await;
                        let text = format_output(&output, &snapshot, "");
                        if message == "aborted" {
                            return Err(append_status(&text, "Command aborted"));
                        }
                        if message.starts_with("timeout:") {
                            let timeout_secs = message.split(':').nth(1).unwrap_or("");
                            return Err(append_status(
                                &text,
                                &format!("Command timed out after {timeout_secs} seconds"),
                            ));
                        }
                        return Err(message);
                    }
                };

                let snapshot = finish_output(&output, on_update.as_ref(), &throttle).await;
                let details = details_for(&snapshot);
                let output_text = format_output(&output, &snapshot, "(no output)");
                let Some(exit_code) = exit_code else {
                    return Err(append_status(
                        &output_text,
                        "Command terminated without an exit code",
                    ));
                };
                let wall_time_seconds =
                    (started_at.elapsed().as_secs_f64() * 1000.0 / 100.0).round() / 10.0;
                let full_output_future =
                    output.lock().read_full_output(STRUCTURED_OUTPUT_MAX_BYTES);
                let full_output = full_output_future.await;
                let mut structured = json!({
                    "output": full_output.content,
                    "truncated": full_output.truncated,
                    "exit_code": exit_code,
                    "wall_time_seconds": wall_time_seconds,
                });
                if full_output.truncated {
                    if let Some(full_output_path) = &snapshot.full_output_path {
                        structured["full_output_path"] = json!(full_output_path);
                    }
                }

                let content = if exit_code != 0 {
                    append_status(
                        &output_text,
                        &format!("Command exited with code {exit_code}"),
                    )
                } else {
                    output_text
                };
                let mut result = AgentToolResult::new(
                    vec![pi_ai::types::UserContent::from(
                        pi_ai::types::TextContent::new(content),
                    )],
                    serde_json::to_value(&details).unwrap_or(JsonValue::Null),
                );
                result.structured_content = Some(structured);
                if exit_code != 0 {
                    result.is_error = Some(true);
                }
                Ok(result)
            })
        },
    );
    definition
}

/// `bashToolConfig`.
pub fn bash_tool_config() -> ShellToolConfig {
    ShellToolConfig {
        name: "bash",
        label: "bash",
        shell_name: "bash",
        prompt: "$",
        prompt_snippet: BASH_TOOL_SYSTEM_PROMPT_SNIPPET,
        prompt_guidelines: Some(BASH_TOOL_SYSTEM_PROMPT_GUIDELINES),
        temp_file_prefix: "pi-bash",
    }
}

pub fn create_bash_tool_definition(cwd: &str, options: Option<BashToolOptions>) -> ToolDefinition {
    create_shell_tool_definition(cwd, bash_tool_config(), options)
}

pub fn create_bash_tool(cwd: &str, options: Option<BashToolOptions>) -> AgentTool {
    let definition = create_bash_tool_definition(cwd, options);
    wrap_tool_definition(definition, None)
}
