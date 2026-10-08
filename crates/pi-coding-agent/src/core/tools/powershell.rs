//! Port of `upstream/pi/packages/coding-agent/src/core/tools/powershell.ts`.
//!
//! The PowerShell tool aliases the bash tool surfaces: `PowerShellOperations`
//! = [`crate::core::tools::bash::BashOperations`], and the spawn/detail/input
//! types are type aliases; only the UTF-8 output prefix, the tool config, and
//! the default operations wrapper differ from the bash port.

use std::sync::Arc;

use crate::core::extensions::types::ToolDefinition;
use crate::core::tools::bash::{
	create_local_shell_operations, create_shell_tool_definition, BashOperations, BashSpawnContext, BashSpawnHook,
	BashToolDetails, BashToolInput, FnOperations,
};
use crate::core::tools::tool_definition_wrapper::wrap_tool_definition;
use crate::utils::shell::get_powershell_config;
use pi_agent::AgentTool;

const UTF8_OUTPUT_PREFIX: &str = "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\n";

/// `powershellToolSystemPromptContribution`.
pub const POWERSHELL_TOOL_SYSTEM_PROMPT_SNIPPET: &str = "Execute PowerShell commands";
pub const POWERSHELL_TOOL_SYSTEM_PROMPT_GUIDELINES: &[&str] =
	&["You can inspect PI_* environment variables for current model and session details."];

/// `PowerShellOperations` aliases the bash trait object (upstream: `type PowerShellOperations = BashOperations`).
pub type PowerShellOperations = dyn BashOperations + 'static;
pub type PowerShellSpawnContext = BashSpawnContext;
pub type PowerShellSpawnHook = BashSpawnHook;
pub type PowerShellToolDetails = BashToolDetails;
pub type PowerShellToolInput = BashToolInput;

/// `PowerShellToolOptions` (subset of [`crate::core::tools::bash::BashToolOptions`]).
#[derive(Default, Clone)]
pub struct PowerShellToolOptions {
	/// Custom operations for command execution. Default: local PowerShell
	pub operations: Option<Arc<dyn BashOperations>>,
	/// Expose current Pi session metadata as PI_* environment variables. Default: true
	pub expose_session_environment: Option<bool>,
	/// Hook to adjust command, cwd, or env before execution
	pub spawn_hook: Option<PowerShellSpawnHook>,
}

pub fn create_local_power_shell_operations() -> Arc<PowerShellOperations> {
	let operations = create_local_shell_operations("PowerShell", Arc::new(get_powershell_config));
	Arc::new(FnOperations(move |command: &str, cwd: &str, options| {
		let operations = Arc::clone(&operations);
		let command = format!("{UTF8_OUTPUT_PREFIX}{command}");
		operations.exec(&command, cwd, options)
	})) as Arc<PowerShellOperations>
}

/// `powershellToolConfig`.
pub fn powershell_tool_config() -> crate::core::tools::bash::ShellToolConfig {
	crate::core::tools::bash::ShellToolConfig {
		name: "powershell",
		label: "powershell",
		shell_name: "PowerShell",
		prompt: "PS>",
		prompt_snippet: POWERSHELL_TOOL_SYSTEM_PROMPT_SNIPPET,
		prompt_guidelines: Some(POWERSHELL_TOOL_SYSTEM_PROMPT_GUIDELINES),
		temp_file_prefix: "pi-powershell",
	}
}

pub fn create_power_shell_tool_definition(cwd: &str, options: Option<PowerShellToolOptions>) -> ToolDefinition {
	let options = options.unwrap_or_default();
	create_shell_tool_definition(
		cwd,
		powershell_tool_config(),
		Some(crate::core::tools::bash::BashToolOptions {
			operations: Some(options.operations.unwrap_or_else(create_local_power_shell_operations)),
			expose_session_environment: options.expose_session_environment,
			spawn_hook: options.spawn_hook,
			..Default::default()
		}),
	)
}

pub fn create_power_shell_tool(cwd: &str, options: Option<PowerShellToolOptions>) -> AgentTool {
	let definition = create_power_shell_tool_definition(cwd, options);
	wrap_tool_definition(definition, None)
}
