//! Port of `packages/coding-agent/src/core/tools/index.ts` — the tool
//! composition root: per-name factories and the coding/read-only/all sets.
//!
//! `Tool` = [`pi_agent::AgentTool`]; `ToolDef` = erased
//! [`ToolDefinition`] (D27).

use pi_agent::AgentTool;

use crate::core::extensions::types::ToolDefinition;

use super::bash::{create_bash_tool, create_bash_tool_definition, BashToolOptions};
use super::edit::{create_edit_tool, create_edit_tool_definition, EditToolOptions};
use super::find::{create_find_tool, create_find_tool_definition, FindToolOptions};
use super::grep::{create_grep_tool, create_grep_tool_definition, GrepToolOptions};
use super::ls::{create_ls_tool, create_ls_tool_definition, LsToolOptions};
use super::powershell::{
    create_power_shell_tool, create_power_shell_tool_definition, PowerShellToolOptions,
};
use super::read::{create_read_tool, create_read_tool_definition, ReadToolOptions};
use super::write::{create_write_tool, create_write_tool_definition, WriteToolOptions};

/// A wrapped tool for the agent runtime (`Tool`).
pub type Tool = AgentTool;
/// A tool definition (`ToolDef`).
pub type ToolDef = ToolDefinition;

/// The built-in tool names (`ToolName`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolName {
    Read,
    Bash,
    PowerShell,
    Edit,
    Write,
    Grep,
    Find,
    Ls,
}

/// Every built-in tool name (`allToolNames`).
pub const ALL_TOOL_NAMES: [ToolName; 8] = [
    ToolName::Read,
    ToolName::Bash,
    ToolName::PowerShell,
    ToolName::Edit,
    ToolName::Write,
    ToolName::Grep,
    ToolName::Find,
    ToolName::Ls,
];

impl ToolName {
    /// Upstream string form.
    pub fn as_str(&self) -> &'static str {
        match self {
            ToolName::Read => "read",
            ToolName::Bash => "bash",
            ToolName::PowerShell => "powershell",
            ToolName::Edit => "edit",
            ToolName::Write => "write",
            ToolName::Grep => "grep",
            ToolName::Find => "find",
            ToolName::Ls => "ls",
        }
    }

    /// Parse a tool name; the upstream `Set` membership check.
    pub fn from_name(name: &str) -> Option<ToolName> {
        match name {
            "read" => Some(ToolName::Read),
            "bash" => Some(ToolName::Bash),
            "powershell" => Some(ToolName::PowerShell),
            "edit" => Some(ToolName::Edit),
            "write" => Some(ToolName::Write),
            "grep" => Some(ToolName::Grep),
            "find" => Some(ToolName::Find),
            "ls" => Some(ToolName::Ls),
            _ => None,
        }
    }
}

/// Per-tool options (`ToolsOptions`).
#[derive(Default)]
pub struct ToolsOptions {
    pub read: Option<ReadToolOptions>,
    pub bash: Option<BashToolOptions>,
    pub powershell: Option<PowerShellToolOptions>,
    pub write: Option<WriteToolOptions>,
    pub edit: Option<EditToolOptions>,
    pub grep: Option<GrepToolOptions>,
    pub find: Option<FindToolOptions>,
    pub ls: Option<LsToolOptions>,
}

/// Create a tool definition by name (`createToolDefinition`).
///
/// # Errors
/// `Unknown tool name: {name}` for names outside [`ALL_TOOL_NAMES`] — but the
/// [`ToolName`] enum makes that unreachable; upstream's string switch becomes
/// this total match.
pub fn create_tool_definition(
    tool_name: ToolName,
    cwd: &str,
    options: Option<&ToolsOptions>,
) -> ToolDef {
    match tool_name {
        ToolName::Read => create_read_tool_definition(
            cwd,
            options.and_then(|options| options.read.as_ref().cloned()),
        ),
        ToolName::Bash => create_bash_tool_definition(
            cwd,
            options.and_then(|options| options.bash.as_ref().cloned()),
        ),
        ToolName::PowerShell => create_power_shell_tool_definition(
            cwd,
            options.and_then(|options| options.powershell.as_ref().cloned()),
        ),
        ToolName::Edit => create_edit_tool_definition(
            cwd,
            options.and_then(|options| options.edit.as_ref().cloned()),
        ),
        ToolName::Write => create_write_tool_definition(
            cwd,
            options.and_then(|options| options.write.as_ref().cloned()),
        ),
        ToolName::Grep => create_grep_tool_definition(
            cwd,
            options.and_then(|options| options.grep.as_ref().cloned()),
        ),
        ToolName::Find => create_find_tool_definition(
            cwd,
            options.and_then(|options| options.find.as_ref().cloned()),
        ),
        ToolName::Ls => create_ls_tool_definition(
            cwd,
            options.and_then(|options| options.ls.as_ref().cloned()),
        ),
    }
}

/// Create a wrapped tool by name (`createTool`).
pub fn create_tool(tool_name: ToolName, cwd: &str, options: Option<&ToolsOptions>) -> Tool {
    match tool_name {
        ToolName::Read => create_read_tool(
            cwd,
            options.and_then(|options| options.read.as_ref().cloned()),
        ),
        ToolName::Bash => create_bash_tool(
            cwd,
            options.and_then(|options| options.bash.as_ref().cloned()),
        ),
        ToolName::PowerShell => create_power_shell_tool(
            cwd,
            options.and_then(|options| options.powershell.as_ref().cloned()),
        ),
        ToolName::Edit => create_edit_tool(
            cwd,
            options.and_then(|options| options.edit.as_ref().cloned()),
        ),
        ToolName::Write => create_write_tool(
            cwd,
            options.and_then(|options| options.write.as_ref().cloned()),
        ),
        ToolName::Grep => create_grep_tool(
            cwd,
            options.and_then(|options| options.grep.as_ref().cloned()),
        ),
        ToolName::Find => create_find_tool(
            cwd,
            options.and_then(|options| options.find.as_ref().cloned()),
        ),
        ToolName::Ls => create_ls_tool(
            cwd,
            options.and_then(|options| options.ls.as_ref().cloned()),
        ),
    }
}

/// The coding tools: read, bash, edit, write (`createCodingToolDefinitions`).
pub fn create_coding_tool_definitions(cwd: &str, options: Option<&ToolsOptions>) -> Vec<ToolDef> {
    vec![
        create_read_tool_definition(
            cwd,
            options.and_then(|options| options.read.as_ref().cloned()),
        ),
        create_bash_tool_definition(
            cwd,
            options.and_then(|options| options.bash.as_ref().cloned()),
        ),
        create_edit_tool_definition(
            cwd,
            options.and_then(|options| options.edit.as_ref().cloned()),
        ),
        create_write_tool_definition(
            cwd,
            options.and_then(|options| options.write.as_ref().cloned()),
        ),
    ]
}

/// The read-only tools: read, grep, find, ls (`createReadOnlyToolDefinitions`).
pub fn create_read_only_tool_definitions(
    cwd: &str,
    options: Option<&ToolsOptions>,
) -> Vec<ToolDef> {
    vec![
        create_read_tool_definition(
            cwd,
            options.and_then(|options| options.read.as_ref().cloned()),
        ),
        create_grep_tool_definition(
            cwd,
            options.and_then(|options| options.grep.as_ref().cloned()),
        ),
        create_find_tool_definition(
            cwd,
            options.and_then(|options| options.find.as_ref().cloned()),
        ),
        create_ls_tool_definition(
            cwd,
            options.and_then(|options| options.ls.as_ref().cloned()),
        ),
    ]
}

/// Every tool definition keyed by name (`createAllToolDefinitions`).
pub fn create_all_tool_definitions(cwd: &str, options: Option<&ToolsOptions>) -> [ToolDef; 8] {
    [
        create_read_tool_definition(
            cwd,
            options.and_then(|options| options.read.as_ref().cloned()),
        ),
        create_bash_tool_definition(
            cwd,
            options.and_then(|options| options.bash.as_ref().cloned()),
        ),
        create_power_shell_tool_definition(
            cwd,
            options.and_then(|options| options.powershell.as_ref().cloned()),
        ),
        create_edit_tool_definition(
            cwd,
            options.and_then(|options| options.edit.as_ref().cloned()),
        ),
        create_write_tool_definition(
            cwd,
            options.and_then(|options| options.write.as_ref().cloned()),
        ),
        create_grep_tool_definition(
            cwd,
            options.and_then(|options| options.grep.as_ref().cloned()),
        ),
        create_find_tool_definition(
            cwd,
            options.and_then(|options| options.find.as_ref().cloned()),
        ),
        create_ls_tool_definition(
            cwd,
            options.and_then(|options| options.ls.as_ref().cloned()),
        ),
    ]
}

/// The coding tools (`createCodingTools`).
pub fn create_coding_tools(cwd: &str, options: Option<&ToolsOptions>) -> Vec<Tool> {
    vec![
        create_read_tool(
            cwd,
            options.and_then(|options| options.read.as_ref().cloned()),
        ),
        create_bash_tool(
            cwd,
            options.and_then(|options| options.bash.as_ref().cloned()),
        ),
        create_edit_tool(
            cwd,
            options.and_then(|options| options.edit.as_ref().cloned()),
        ),
        create_write_tool(
            cwd,
            options.and_then(|options| options.write.as_ref().cloned()),
        ),
    ]
}

/// The read-only tools (`createReadOnlyTools`).
pub fn create_read_only_tools(cwd: &str, options: Option<&ToolsOptions>) -> Vec<Tool> {
    vec![
        create_read_tool(
            cwd,
            options.and_then(|options| options.read.as_ref().cloned()),
        ),
        create_grep_tool(
            cwd,
            options.and_then(|options| options.grep.as_ref().cloned()),
        ),
        create_find_tool(
            cwd,
            options.and_then(|options| options.find.as_ref().cloned()),
        ),
        create_ls_tool(
            cwd,
            options.and_then(|options| options.ls.as_ref().cloned()),
        ),
    ]
}

/// Every tool keyed by name (`createAllTools`).
pub fn create_all_tools(cwd: &str, options: Option<&ToolsOptions>) -> [Tool; 8] {
    [
        create_read_tool(
            cwd,
            options.and_then(|options| options.read.as_ref().cloned()),
        ),
        create_bash_tool(
            cwd,
            options.and_then(|options| options.bash.as_ref().cloned()),
        ),
        create_power_shell_tool(
            cwd,
            options.and_then(|options| options.powershell.as_ref().cloned()),
        ),
        create_edit_tool(
            cwd,
            options.and_then(|options| options.edit.as_ref().cloned()),
        ),
        create_write_tool(
            cwd,
            options.and_then(|options| options.write.as_ref().cloned()),
        ),
        create_grep_tool(
            cwd,
            options.and_then(|options| options.grep.as_ref().cloned()),
        ),
        create_find_tool(
            cwd,
            options.and_then(|options| options.find.as_ref().cloned()),
        ),
        create_ls_tool(
            cwd,
            options.and_then(|options| options.ls.as_ref().cloned()),
        ),
    ]
}
