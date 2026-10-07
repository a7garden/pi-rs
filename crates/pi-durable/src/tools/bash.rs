//! Port of `packages/durable/src/tools/bash.ts` — run a command through the environment's shell.
//!
//! D1: upstream throws inside `execute`; the harness converts a throw into an error result. The
//! erased seam has no throwing path (`ToolExecute` returns a result), so each tool wraps a
//! fallible `execute` returning `Result<ToolExecutionResult, String>`; the mod-level
//! `wrap_execute` produces the same `{isError, diagnostics:[tool_error]}` shape upstream's
//! harness builds. The `aborted` rethrow (harness fails the task) collapses to the same error
//! result here; abort supervision stays with the harness slice (D2).

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use pi_chord::context::Context;
use serde_json::{json, Value};

use crate::env::{
	ExecCommand, ExecutionErrorCode, ShellExecOptions, ShellOutputInfo, ShellSpillOptions,
};
use crate::harness::api::{
	define_tool, ToolDiagnostic, ToolDiagnosticSeverity, ToolExecute, ToolExecutionApi,
	ToolExecutionResult, ToolRegistration,
};
use crate::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES};

use super::env::require_env;

const MAX_TIMEOUT_SECONDS: f64 = 2_147_483_647.0 / 1000.0;

pub fn bash_parameters() -> Value {
	json!({
		"type": "object",
		"properties": {
			"command": { "type": "string", "description": "Bash command to execute" },
			"timeout": { "type": "number", "description": "Timeout in seconds (optional, no default timeout)" },
		},
		"required": ["command"],
	})
}

/// Upstream `BashToolInput` (`Static<typeof bashSchema>`); erased to validated JSON (D1).
pub type BashToolInput = Value;

#[derive(Debug, Clone)]
pub struct BashExecution {
	pub command: String,
	pub cwd: String,
	pub env: HashMap<String, String>,
	pub inherit_env: bool,
}

/// Upstream `BashPrepare`: `(execution, api, context) => void | Promise<void>`.
pub type BashPrepare = Arc<
	dyn Fn(&mut BashExecution, Arc<dyn ToolExecutionApi>, Context) -> FutureBox + Send + Sync,
>;
pub type FutureBox = std::pin::Pin<Box<dyn Future<Output = ()> + Send>>;

#[derive(Clone, Default)]
pub struct BashToolOptions {
	pub command_prefix: Option<String>,
	pub prepare: Option<BashPrepare>,
}

fn validate_timeout(timeout: Option<f64>) -> Result<(), String> {
	let Some(timeout) = timeout else { return Ok(()) };
	if !timeout.is_finite() || timeout <= 0.0 {
		return Err("Invalid timeout: must be a finite number of seconds".to_string());
	}
	if timeout > MAX_TIMEOUT_SECONDS {
		return Err(format!("Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds"));
	}
	Ok(())
}

/// Fallible body of one `bash` call (see module D1).
pub async fn execute(
	args: Value,
	options: Option<BashToolOptions>,
	api: Arc<dyn ToolExecutionApi>,
	context: Context,
) -> Result<ToolExecutionResult, String> {
	let command = args.get("command").and_then(Value::as_str).unwrap_or_default().to_string();
	let timeout = args.get("timeout").and_then(Value::as_f64);
	validate_timeout(timeout)?;
	let env = require_env(api.as_ref())?;
	let mut execution = BashExecution {
		command: match options.as_ref().and_then(|options| options.command_prefix.clone()) {
			Some(prefix) => format!("{prefix}\n{command}"),
			None => command,
		},
		cwd: env.cwd().to_string(),
		env: HashMap::new(),
		inherit_env: true,
	};
	if let Some(prepare) = options.as_ref().and_then(|options| options.prepare.clone()) {
		prepare(&mut execution, Arc::clone(&api), context.clone()).await;
	}
	let mut exec_options = ShellExecOptions {
		cwd: Some(execution.cwd.clone()),
		env: Some(execution.env.clone()),
		inherit_env: Some(execution.inherit_env),
		timeout,
		on_output: Some({
			let api = Arc::clone(&api);
			Arc::new(move |text: &str, info: &ShellOutputInfo| {
				api.output(text.as_bytes(), info.skipped);
			})
		}),
		spill: Some(ShellSpillOptions {
			after_bytes: DEFAULT_MAX_BYTES as u64,
			after_lines: DEFAULT_MAX_LINES as u64,
		}),
		..Default::default()
	};
	// An environment may then omit output outside the retained tail and report the omission.
	exec_options.window = api.output_window();
	let result = env.exec(ExecCommand::Shell(execution.command.clone()), Some(exec_options), &context).await;
	let spill_path = match &result {
		Ok(value) => value.spill_path.clone(),
		Err(error) => error.spill_path.clone(),
	};
	if let Some(spill_path) = spill_path {
		api.diagnostic(ToolDiagnostic {
			severity: ToolDiagnosticSeverity::Info,
			code: Some("full_output".to_string()),
			message: format!("Full output: {spill_path}"),
		});
	}
	let value = match result {
		Ok(value) => value,
		Err(error) => {
			if error.code == ExecutionErrorCode::Aborted && context.abort_signal().is_some_and(|signal| signal.aborted())
			{
				return Err(error.message);
			}
			if error.code == ExecutionErrorCode::Timeout {
				return Err(format!("Command timed out after {} seconds", timeout.unwrap_or_default()));
			}
			if error.code == ExecutionErrorCode::Aborted {
				return Err("Command aborted".to_string());
			}
			return Err(error.message);
		}
	};
	if value.exit_code != 0 {
		return Err(format!("Command exited with code {}", value.exit_code));
	}
	Ok(ToolExecutionResult::default())
}

/// Runs a command through the environment's shell. Its output streams to `api.output()`, where
/// the Harness keeps the tail within the default limits; the result content is that retained
/// output. The retained window goes to the environment, which may omit output outside it and
/// report how much it omitted, so dropped counts stay exact. Output beyond the limits is spilled
/// to a file whose path is reported as a diagnostic. A nonzero exit or timeout becomes an error
/// result that still carries the output and diagnostics.
pub fn create_bash_tool(options: Option<BashToolOptions>) -> ToolRegistration {
	let execute_fn: ToolExecute = Arc::new(move |args, api, context| {
		let options = options.clone();
		Box::pin(async move {
			match execute(args, options, Arc::clone(&api), context).await {
				Ok(result) => result,
				Err(message) => ToolExecutionResult {
					is_error: Some(true),
					diagnostics: Some(vec![ToolDiagnostic {
						severity: ToolDiagnosticSeverity::Error,
						code: Some("tool_error".to_string()),
						message,
					}]),
					..Default::default()
				},
			}
		})
	});
define_tool(ToolRegistration {
	tool: pi_ai::Tool {
		name: "bash".to_string(),
		description: format!(
			"Execute a bash command in the current working directory. Returns combined stdout and stderr. Output is truncated to last {} lines or {}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
			DEFAULT_MAX_LINES,
			DEFAULT_MAX_BYTES / 1024
		),
		parameters: bash_parameters(),
		constrained_sampling: None,
	},
	replay: None,
	execution_mode: None,
	prepare_arguments: None,
	output_limits: Some(crate::harness::api::ToolOutputLimits {
		retain: Some(crate::harness::api::OutputRetain::Tail),
		..Default::default()
	}),
	execute: execute_fn,
})
}
