//! Port of `upstream/pi/packages/coding-agent/test/tool-system-prompt-contributions.test.ts`
//! (1:1) — each built-in tool definition stays aligned with its system-prompt
//! contribution, and the shell tools' session-environment guidance is
//! conditional on `exposeSessionEnvironment`.

use super::bash::{
    create_bash_tool_definition, BashToolOptions, BASH_TOOL_SYSTEM_PROMPT_GUIDELINES,
    BASH_TOOL_SYSTEM_PROMPT_SNIPPET,
};
use super::edit::{
    create_edit_tool_definition, EDIT_TOOL_SYSTEM_PROMPT_GUIDELINES,
    EDIT_TOOL_SYSTEM_PROMPT_SNIPPET,
};
use super::find::{create_find_tool_definition, FIND_TOOL_SYSTEM_PROMPT_SNIPPET};
use super::grep::{create_grep_tool_definition, GREP_TOOL_SYSTEM_PROMPT_SNIPPET};
use super::ls::{create_ls_tool_definition, LS_TOOL_SYSTEM_PROMPT_SNIPPET};
use super::powershell::{
    create_power_shell_tool_definition, PowerShellToolOptions,
    POWERSHELL_TOOL_SYSTEM_PROMPT_GUIDELINES, POWERSHELL_TOOL_SYSTEM_PROMPT_SNIPPET,
};
use super::read::{
    create_read_tool_definition, READ_TOOL_SYSTEM_PROMPT_GUIDELINES,
    READ_TOOL_SYSTEM_PROMPT_SNIPPET,
};
use super::write::{
    create_write_tool_definition, WRITE_TOOL_SYSTEM_PROMPT_GUIDELINES,
    WRITE_TOOL_SYSTEM_PROMPT_SNIPPET,
};
use crate::core::extensions::types::ToolDefinition;

fn check_alignment(definition: &ToolDefinition, snippet: &str, guide: Option<Vec<String>>) {
    assert_eq!(definition.prompt_snippet.as_deref(), Some(snippet));
    assert_eq!(definition.prompt_guidelines, guide);
}

fn guidelines(constants: &[&str]) -> Option<Vec<String>> {
    Some(constants.iter().map(|g| g.to_string()).collect())
}

#[test]
fn keeps_the_read_tool_definition_aligned_with_its_contribution() {
    let definition = create_read_tool_definition("/workspace", None);
    check_alignment(
        &definition,
        READ_TOOL_SYSTEM_PROMPT_SNIPPET,
        guidelines(READ_TOOL_SYSTEM_PROMPT_GUIDELINES.as_slice()),
    );
}

#[test]
fn keeps_the_bash_tool_definition_aligned_with_its_contribution() {
    let definition = create_bash_tool_definition("/workspace", None);
    check_alignment(
        &definition,
        BASH_TOOL_SYSTEM_PROMPT_SNIPPET,
        guidelines(BASH_TOOL_SYSTEM_PROMPT_GUIDELINES),
    );
}

#[test]
fn keeps_the_powershell_tool_definition_aligned_with_its_contribution() {
    let definition = create_power_shell_tool_definition("/workspace", None);
    check_alignment(
        &definition,
        POWERSHELL_TOOL_SYSTEM_PROMPT_SNIPPET,
        guidelines(POWERSHELL_TOOL_SYSTEM_PROMPT_GUIDELINES),
    );
}

#[test]
fn keeps_the_edit_tool_definition_aligned_with_its_contribution() {
    let definition = create_edit_tool_definition("/workspace", None);
    check_alignment(
        &definition,
        EDIT_TOOL_SYSTEM_PROMPT_SNIPPET,
        guidelines(EDIT_TOOL_SYSTEM_PROMPT_GUIDELINES.as_slice()),
    );
}

#[test]
fn keeps_the_write_tool_definition_aligned_with_its_contribution() {
    let definition = create_write_tool_definition("/workspace", None);
    check_alignment(
        &definition,
        WRITE_TOOL_SYSTEM_PROMPT_SNIPPET,
        guidelines(WRITE_TOOL_SYSTEM_PROMPT_GUIDELINES.as_slice()),
    );
}

#[test]
fn keeps_the_grep_tool_definition_aligned_with_its_contribution() {
    let definition = create_grep_tool_definition("/workspace", None);
    check_alignment(&definition, GREP_TOOL_SYSTEM_PROMPT_SNIPPET, None);
}

#[test]
fn keeps_the_find_tool_definition_aligned_with_its_contribution() {
    let definition = create_find_tool_definition("/workspace", None);
    check_alignment(&definition, FIND_TOOL_SYSTEM_PROMPT_SNIPPET, None);
}

#[test]
fn keeps_the_ls_tool_definition_aligned_with_its_contribution() {
    let definition = create_ls_tool_definition("/workspace", None);
    check_alignment(&definition, LS_TOOL_SYSTEM_PROMPT_SNIPPET, None);
}

#[test]
fn keeps_bash_session_environment_guidance_conditional() {
    let definition = create_bash_tool_definition(
        "/workspace",
        Some(BashToolOptions {
            expose_session_environment: Some(false),
            ..Default::default()
        }),
    );
    assert!(definition.prompt_guidelines.is_none());
}

#[test]
fn keeps_powershell_session_environment_guidance_conditional() {
    let definition = create_power_shell_tool_definition(
        "/workspace",
        Some(PowerShellToolOptions {
            expose_session_environment: Some(false),
            ..Default::default()
        }),
    );
    assert!(definition.prompt_guidelines.is_none());
}
