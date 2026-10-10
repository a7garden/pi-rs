//! Port of `packages/coding-agent/src/core/slash-commands.ts` — the
//! slash-command surface types and the built-in command catalog shown in the
//! TUI.

use crate::core::source_info::SourceInfo;

/// Where a slash command comes from (upstream `SlashCommandSource`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlashCommandSource {
    Extension,
    Prompt,
    Skill,
}

/// A registered slash command (upstream `SlashCommandInfo`).
#[derive(Debug, Clone)]
pub struct SlashCommandInfo {
    pub name: String,
    pub description: Option<String>,
    pub source: SlashCommandSource,
    pub source_info: SourceInfo,
}

/// A built-in slash command entry (upstream `BuiltinSlashCommand`).
#[derive(Debug, Clone)]
pub struct BuiltinSlashCommand {
    pub name: &'static str,
    pub description: &'static str,
    pub argument_hint: Option<&'static str>,
}

const fn builtin(
    name: &'static str,
    description: &'static str,
    argument_hint: Option<&'static str>,
) -> BuiltinSlashCommand {
    BuiltinSlashCommand {
        name,
        description,
        argument_hint,
    }
}

/// Built-in slash commands (upstream `BUILTIN_SLASH_COMMANDS`).
pub const BUILTIN_SLASH_COMMANDS: &[BuiltinSlashCommand] = &[
    builtin("settings", "Open settings menu", None),
    builtin(
        "model",
        "Select model (opens selector UI)",
        Some("<provider/model>"),
    ),
    builtin("tree", "Navigate session tree (switch branches)", None),
    builtin("thinking", "Set thinking level", Some("<level>")),
    builtin(
        "scoped-models",
        "Enable/disable models for Ctrl+P cycling",
        None,
    ),
    builtin(
        "export",
        "Export session (HTML default, or specify path: .html/.jsonl)",
        None,
    ),
    builtin(
        "import",
        "Import and resume a session from a JSONL file",
        None,
    ),
    builtin("share", "Share session as a secret GitHub gist", None),
    builtin(
        "bug",
        "Report a bug to the Pi developers",
        Some("<description>"),
    ),
    builtin("copy", "Copy last agent message to clipboard", None),
    builtin("name", "Set session display name", None),
    builtin("session", "Show session info and stats", None),
    builtin("changelog", "Show changelog entries", None),
    builtin("hotkeys", "Show all keyboard shortcuts", None),
    builtin(
        "fork",
        "Create a new fork from a previous user message",
        None,
    ),
    builtin(
        "clone",
        "Duplicate the current session at the current position",
        None,
    ),
    builtin(
        "trust",
        "Save project trust decision for future sessions",
        None,
    ),
    builtin(
        "login",
        "Configure provider authentication",
        Some("<provider>"),
    ),
    builtin("logout", "Remove provider authentication", None),
    builtin("new", "Start a new session", None),
    builtin("compact", "Manually compact the session context", None),
    builtin("resume", "Resume a different session", None),
    builtin(
        "reload",
        "Reload keybindings, extensions, skills, prompts, themes, and context files",
        None,
    ),
    // Upstream: `Quit ${APP_NAME}`; APP_NAME is "pi" (config.rs D-derivation).
    builtin("quit", "Quit pi", None),
];
