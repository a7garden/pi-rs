//! CLI argument parsing and help display (`cli/args.ts`).
//!
//! \*\*D100 — hand-rolled parser over clap.** The upstream parser is a
//! hand-written loop with idiosyncratic observable behavior (`-p` consumes
//! one non-flag message, unknown `--flags` are captured for extensions,
//! `--mode` value validation emits diagnostics instead of failing). Porting
//! the loop 1:1 matches upstream exactly; clap would need a shim layer that
//! reimplements every quirk anyway.

use std::collections::BTreeMap;

use crate::config::{APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR, ENV_SESSION_DIR};

/// Upstream `Mode` (`"text" | "json" | "rpc"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Text,
    Json,
    Rpc,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Text => "text",
            Mode::Json => "json",
            Mode::Rpc => "rpc",
        }
    }
}

/// Value captured for an unknown (potentially extension) flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnknownFlagValue {
    Bool,
    Str(String),
}

/// Diagnostic severity (`"warning" | "error"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticType {
    Warning,
    Error,
}

/// Parse diagnostic (`{ type, message }`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub diagnostic_type: DiagnosticType,
    pub message: String,
}

impl Diagnostic {
    fn error(message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            diagnostic_type: DiagnosticType::Error,
            message: message.into(),
        }
    }

    fn warning(message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            diagnostic_type: DiagnosticType::Warning,
            message: message.into(),
        }
    }
}

/// Parsed CLI arguments (`Args`).
#[derive(Debug, Clone, Default)]
pub struct Args {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub system_prompt: Option<String>,
    pub append_system_prompt: Option<Vec<String>>,
    pub thinking: Option<pi_agent::types::AgentThinkingLevel>,
    pub continue_: bool,
    pub resume: bool,
    pub help: bool,
    pub version: bool,
    pub mode: Option<Mode>,
    pub name: Option<String>,
    pub no_session: bool,
    pub session: Option<String>,
    pub session_id: Option<String>,
    pub fork: Option<String>,
    pub session_dir: Option<String>,
    pub models: Option<Vec<String>>,
    pub tools: Option<Vec<String>>,
    pub exclude_tools: Option<Vec<String>>,
    pub no_tools: bool,
    pub no_builtin_tools: bool,
    pub extensions: Option<Vec<String>>,
    pub no_extensions: bool,
    pub no_mcp: bool,
    pub print: bool,
    pub export: Option<String>,
    pub no_skills: bool,
    pub skills: Option<Vec<String>>,
    pub prompt_templates: Option<Vec<String>>,
    pub no_prompt_templates: bool,
    pub themes: Option<Vec<String>>,
    pub use_theme: Option<String>,
    pub no_themes: bool,
    pub no_context_files: bool,
    /// `None` = absent, `Some(None)` = `true`, `Some(Some(pattern))` = search string.
    pub list_models: Option<Option<String>>,
    pub offline: bool,
    pub tui_mode: Option<crate::core::settings_manager::TuiMode>,
    pub verbose: bool,
    /// `None` = absent, `Some(true)` = `--approve`, `Some(false)` = `--no-approve`.
    pub project_trust_override: Option<bool>,
    pub messages: Vec<String>,
    pub file_args: Vec<String>,
    /// Unknown flags (potentially extension flags) — flag name to value.
    pub unknown_flags: BTreeMap<String, UnknownFlagValue>,
    pub diagnostics: Vec<Diagnostic>,
}

pub const VALID_THINKING_LEVELS: [&str; 7] =
    ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

pub fn is_valid_thinking_level(level: &str) -> bool {
    VALID_THINKING_LEVELS.contains(&level)
}

fn parse_thinking_level(level: &str) -> Option<pi_agent::types::AgentThinkingLevel> {
    use pi_agent::types::AgentThinkingLevel;
    Some(match level {
        "off" => AgentThinkingLevel::Off,
        "minimal" => AgentThinkingLevel::Minimal,
        "low" => AgentThinkingLevel::Low,
        "medium" => AgentThinkingLevel::Medium,
        "high" => AgentThinkingLevel::High,
        "xhigh" => AgentThinkingLevel::Xhigh,
        "max" => AgentThinkingLevel::Max,
        _ => return None,
    })
}

pub fn normalize_session_name(value: &str) -> Option<String> {
    let name = value.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

fn split_comma_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|name| !name.is_empty())
        .collect()
}

/// Parse CLI arguments (`parseArgs`).
pub fn parse_args(args: &[String]) -> Args {
    let mut result = Args::default();

    let mut i = 0usize;
    while i < args.len() {
        let arg = args[i].as_str();

        if arg == "--" {
            for positional_arg in &args[i + 1..] {
                if positional_arg.starts_with('@') {
                    result.file_args.push(positional_arg[1..].to_string());
                } else {
                    result.messages.push(positional_arg.clone());
                }
            }
            break;
        } else if arg == "--help" || arg == "-h" {
            result.help = true;
        } else if arg == "--version" || arg == "-v" {
            result.version = true;
        } else if arg == "--mode" {
            let mode = args.get(i + 1).map(|s| s.as_str());
            match mode {
                None => {
                    result
                        .diagnostics
                        .push(Diagnostic::error("--mode requires text, json, or rpc"));
                }
                Some(mode) if mode.starts_with('-') => {
                    result
                        .diagnostics
                        .push(Diagnostic::error("--mode requires text, json, or rpc"));
                }
                Some("text") => {
                    result.mode = Some(Mode::Text);
                    i += 1;
                }
                Some("json") => {
                    result.mode = Some(Mode::Json);
                    i += 1;
                }
                Some("rpc") => {
                    result.mode = Some(Mode::Rpc);
                    i += 1;
                }
                Some(other) => {
                    result.diagnostics.push(Diagnostic::error(format!(
                        "Invalid mode \"{other}\". Valid values: text, json, rpc"
                    )));
                    i += 1;
                }
            }
        } else if arg == "--continue" || arg == "-c" {
            result.continue_ = true;
        } else if arg == "--resume" || arg == "-r" {
            result.resume = true;
        } else if arg == "--provider" && i + 1 < args.len() {
            i += 1;
            result.provider = Some(args[i].clone());
        } else if arg == "--model" && i + 1 < args.len() {
            i += 1;
            result.model = Some(args[i].clone());
        } else if arg == "--api-key" && i + 1 < args.len() {
            i += 1;
            result.api_key = Some(args[i].clone());
        } else if arg == "--system-prompt" && i + 1 < args.len() {
            i += 1;
            result.system_prompt = Some(args[i].clone());
        } else if arg == "--append-system-prompt" && i + 1 < args.len() {
            i += 1;
            result
                .append_system_prompt
                .get_or_insert_with(Vec::new)
                .push(args[i].clone());
        } else if arg == "--name" || arg == "-n" {
            if i + 1 < args.len() {
                i += 1;
                result.name = Some(args[i].clone());
            } else {
                result
                    .diagnostics
                    .push(Diagnostic::error("--name requires a value"));
            }
        } else if arg == "--no-session" {
            result.no_session = true;
        } else if arg == "--session" && i + 1 < args.len() {
            i += 1;
            result.session = Some(args[i].clone());
        } else if arg == "--session-id" && i + 1 < args.len() {
            i += 1;
            result.session_id = Some(args[i].clone());
        } else if arg == "--fork" && i + 1 < args.len() {
            i += 1;
            result.fork = Some(args[i].clone());
        } else if arg == "--session-dir" && i + 1 < args.len() {
            i += 1;
            result.session_dir = Some(args[i].clone());
        } else if arg == "--models" && i + 1 < args.len() {
            i += 1;
            result.models = Some(split_comma_list(&args[i]));
        } else if arg == "--no-tools" || arg == "-nt" {
            result.no_tools = true;
        } else if arg == "--no-builtin-tools" || arg == "-nbt" {
            result.no_builtin_tools = true;
        } else if (arg == "--tools" || arg == "-t") && i + 1 < args.len() {
            i += 1;
            let tools = split_comma_list(&args[i]);
            if let Some(error) = crate::core::settings_manager::get_tool_list_error(&tools) {
                result
                    .diagnostics
                    .push(Diagnostic::error(format!("{arg}: {error}")));
            } else {
                result.tools = Some(tools);
            }
        } else if (arg == "--exclude-tools" || arg == "-xt") && i + 1 < args.len() {
            i += 1;
            result.exclude_tools = Some(split_comma_list(&args[i]));
        } else if arg == "--thinking" && i + 1 < args.len() {
            i += 1;
            let level = args[i].as_str();
            if let Some(level) = parse_thinking_level(level) {
                result.thinking = Some(level);
            } else {
                result.diagnostics.push(Diagnostic::warning(format!(
                    "Invalid thinking level \"{level}\". Valid values: {}",
                    VALID_THINKING_LEVELS.join(", ")
                )));
            }
        } else if arg == "--print" || arg == "-p" {
            result.print = true;
            if let Some(next) = args.get(i + 1) {
                let next = next.as_str();
                if !next.starts_with('@') && (!next.starts_with('-') || next.starts_with("---")) {
                    result.messages.push(next.to_string());
                    i += 1;
                }
            }
        } else if arg == "--export" && i + 1 < args.len() {
            i += 1;
            result.export = Some(args[i].clone());
        } else if (arg == "--extension" || arg == "-e") && i + 1 < args.len() {
            i += 1;
            result
                .extensions
                .get_or_insert_with(Vec::new)
                .push(args[i].clone());
        } else if arg == "--no-extensions" || arg == "-ne" {
            result.no_extensions = true;
        } else if arg == "--no-mcp" {
            result.no_mcp = true;
        } else if arg == "--skill" && i + 1 < args.len() {
            i += 1;
            result
                .skills
                .get_or_insert_with(Vec::new)
                .push(args[i].clone());
        } else if arg == "--prompt-template" && i + 1 < args.len() {
            i += 1;
            result
                .prompt_templates
                .get_or_insert_with(Vec::new)
                .push(args[i].clone());
        } else if arg == "--theme" && i + 1 < args.len() {
            i += 1;
            result
                .themes
                .get_or_insert_with(Vec::new)
                .push(args[i].clone());
        } else if arg == "--use-theme" {
            let theme_name = args.get(i + 1);
            match theme_name {
                None => {
                    result
                        .diagnostics
                        .push(Diagnostic::error("--use-theme requires a theme name"));
                }
                Some(theme_name) if theme_name.starts_with('-') => {
                    result
                        .diagnostics
                        .push(Diagnostic::error("--use-theme requires a theme name"));
                }
                Some(theme_name) => {
                    result.use_theme = Some(theme_name.clone());
                    i += 1;
                }
            }
        } else if arg == "--no-skills" || arg == "-ns" {
            result.no_skills = true;
        } else if arg == "--no-prompt-templates" || arg == "-np" {
            result.no_prompt_templates = true;
        } else if arg == "--no-themes" {
            result.no_themes = true;
        } else if arg == "--no-context-files" || arg == "-nc" {
            result.no_context_files = true;
        } else if arg == "--list-models" {
            // Check if next arg is a search pattern (not a flag or file arg)
            if let Some(next) = args.get(i + 1) {
                if !next.starts_with('-') && !next.starts_with('@') {
                    result.list_models = Some(Some(next.clone()));
                    i += 1;
                    continue;
                }
            }
            result.list_models = Some(None);
        } else if arg == "--tui-mode" {
            let mode = args.get(i + 1);
            match mode.map(|m| m.as_str()) {
                Some("regular") => {
                    result.tui_mode = Some(crate::core::settings_manager::TuiMode::Regular);
                    i += 1;
                }
                Some("fullscreen") => {
                    result.tui_mode = Some(crate::core::settings_manager::TuiMode::Fullscreen);
                    i += 1;
                }
                None => {
                    result.diagnostics.push(Diagnostic::error(
                        "--tui-mode requires regular or fullscreen",
                    ));
                }
                Some(other) if other.starts_with('-') => {
                    result.diagnostics.push(Diagnostic::error(
                        "--tui-mode requires regular or fullscreen",
                    ));
                }
                Some(other) => {
                    i += 1;
                    result.diagnostics.push(Diagnostic::error(format!(
                        "Invalid TUI mode \"{other}\". Valid values: regular, fullscreen"
                    )));
                }
            }
        } else if arg == "--verbose" {
            result.verbose = true;
        } else if arg == "--approve" || arg == "-a" {
            result.project_trust_override = Some(true);
        } else if arg == "--no-approve" || arg == "-na" {
            result.project_trust_override = Some(false);
        } else if arg == "--offline" {
            result.offline = true;
        } else if let Some(stripped) = arg.strip_prefix('@') {
            result.file_args.push(stripped.to_string()); // Remove @ prefix
        } else if let Some(stripped) = arg.strip_prefix("--") {
            if let Some(eq_index) = stripped.find('=') {
                result.unknown_flags.insert(
                    stripped[..eq_index].to_string(),
                    UnknownFlagValue::Str(stripped[eq_index + 1..].to_string()),
                );
            } else {
                let flag_name = stripped.to_string();
                let next = args.get(i + 1);
                match next {
                    Some(next) if !next.starts_with('-') && !next.starts_with('@') => {
                        result
                            .unknown_flags
                            .insert(flag_name, UnknownFlagValue::Str(next.clone()));
                        i += 1;
                    }
                    _ => {
                        result
                            .unknown_flags
                            .insert(flag_name, UnknownFlagValue::Bool);
                    }
                }
            }
        } else if arg.starts_with('-') {
            result
                .diagnostics
                .push(Diagnostic::error(format!("Unknown option: {arg}")));
        } else {
            result.messages.push(arg.to_string());
        }

        i += 1;
    }

    result
}

/// Extension-registered CLI flag (`ExtensionFlag`).
pub struct ExtensionFlagInfo {
    pub name: String,
    pub is_string: bool,
    pub description: Option<String>,
    pub extension_path: String,
}

impl ExtensionFlagInfo {
    /// Build from a loaded extension's registered flag.
    pub fn from_extension_flag(
        name: &str,
        flag: &crate::core::extensions::types::ExtensionFlag,
        extension_path: &str,
    ) -> ExtensionFlagInfo {
        ExtensionFlagInfo {
            name: name.to_string(),
            is_string: flag.flag_type == crate::core::extensions::types::ExtensionFlagType::String,
            description: flag.description.clone(),
            extension_path: extension_path.to_string(),
        }
    }
}

/// Print help text (`printHelp`).
///
/// Upstream colors via chalk; the port prints uncolored text (no TTY
/// styling dependency).
pub fn print_help(extension_flags: &[ExtensionFlagInfo]) {
    let extension_flags_text = if !extension_flags.is_empty() {
        let mut text = String::from("\nExtension CLI Flags:\n");
        let lines: Vec<String> = extension_flags
            .iter()
            .map(|flag| {
                let value = if flag.is_string { " <value>" } else { "" };
                let description = flag
                    .description
                    .clone()
                    .unwrap_or_else(|| format!("Registered by {}", flag.extension_path));
                format!("{:30}{}", format!("  --{}{value}", flag.name), description)
            })
            .collect();
        text.push_str(&lines.join("\n"));
        text.push('\n');
        text
    } else {
        String::new()
    };

    println!(
        r#"{APP_NAME} - AI coding assistant with read, bash, edit, write tools

Usage:
  {APP_NAME} [options] [--] [@files...] [messages...]

Commands:
  {APP_NAME} install <source> [-l]     Install extension source and add to settings
  {APP_NAME} remove <source> [-l]      Remove extension source from settings
  {APP_NAME} uninstall <source> [-l]   Alias for remove
  {APP_NAME} update [source|self|pi]   Update pi, extensions, or model catalogs
  {APP_NAME} list                      List installed extensions from settings
  {APP_NAME} config [-l]               Open TUI to enable/disable package resources (Tab switches scope)
  {APP_NAME} auth <command>            Print credentials or check provider readiness
  {APP_NAME} mcp <command>             Check MCP servers, sign in to or out of OAuth servers
  {APP_NAME} <command> --help          Show help for install/remove/uninstall/update/list/config/auth/mcp

Options:
  --provider <name>              Provider to search for --model (requires --model)
  --model <pattern>              Model pattern or ID (supports "provider/id" and optional ":<thinking>")
  --api-key <key>                API key (defaults to env vars)
  --system-prompt <text>         System prompt (default: coding assistant prompt)
  --append-system-prompt <text>  Append text or file contents to the system prompt (can be used multiple times)
  --mode <mode>                  Output mode: text (default), json, or rpc
  --print, -p                    Non-interactive mode: process prompt and exit
  --continue, -c                 Continue previous session
  --resume, -r                   Select a session to resume
  --session <path|id>            Use specific session file or partial UUID
  --session-id <id>              Use exact project session ID, creating it if missing
  --fork <path|id>               Fork specific session file or partial UUID into a new session
  --session-dir <dir>            Directory for session storage and lookup
  --no-session                   Don't save session (ephemeral)
  --name, -n <name>              Set session display name
  --models <patterns>            Comma-separated model patterns for Ctrl+P cycling
                                 Supports globs (anthropic/*, *sonnet*) and fuzzy matching
  --no-tools, -nt                Disable all tools by default (built-in and extension)
  --no-builtin-tools, -nbt       Disable built-in tools by default but keep extension/custom tools enabled
  --tools, -t <tools>            Comma-separated allowlist of tool names or patterns (*) to enable
                                 Keeps MCP tools unless an entry starts with mcp__
                                 Only +name/-name entries add to or remove from the defaults
  --exclude-tools, -xt <tools>   Comma-separated denylist of tool names or patterns (*) to disable
                                 Applies to all tools, MCP tools included
  --thinking <level>             Set thinking level: off, minimal, low, medium, high, xhigh, max
  --extension, -e <path>         Load an extension file or builtin:<name> (can be used multiple times)
  --no-extensions, -ne           Disable extension discovery and built-in extensions (explicit -e paths still work)
  --no-mcp                       Disable built-in MCP support: no servers connect and no MCP tools
  --skill <path>                 Load a skill file or directory (can be used multiple times)
  --no-skills, -ns               Disable skills discovery and loading
  --prompt-template <path>       Load a prompt template file or directory (can be used multiple times)
  --no-prompt-templates, -np     Disable prompt template discovery and loading
  --theme <path>                 Load a theme file or directory (can be used multiple times)
  --use-theme <name[/name]>      Set the initial interactive theme for this run
  --no-themes                    Disable theme discovery and loading
  --no-context-files, -nc        Disable AGENTS.md and CLAUDE.md discovery and loading
  --export <file>                Export session file to HTML and exit
  --list-models [search]         List available models (with optional fuzzy search)
  --verbose                      Force verbose startup (overrides quietStartup setting)
  --tui-mode <mode>              TUI mode: fullscreen (default) or regular
  --approve, -a                  Trust project-local files for this run
  --no-approve, -na              Ignore project-local files for this run
  --offline                      Disable startup network operations (same as PI_OFFLINE=1)
  --                             End option parsing; treat remaining arguments as messages/files
  --help, -h                     Show this help
  --version, -v                  Show version number

Extensions can register additional flags (e.g., --plan from plan-mode extension).{extension_flags_text}

Examples:
  # Print a provider API key for an external client
  {APP_NAME} auth print-api-key --provider openai

  # Print an OAuth bearer token for an external client (refreshes if expired)
  {APP_NAME} auth print-bearer-token --provider openai-codex

  # Interactive mode
  {APP_NAME}

  # Interactive mode with initial prompt
  {APP_NAME} "List all .ts files in src/"

  # Include files in initial message
  {APP_NAME} @prompt.md @image.png "What color is the sky?"

  # Non-interactive mode (process and exit)
  {APP_NAME} -p "List all .ts files in src/"

  # Prompt beginning with a dash
  {APP_NAME} -p -- "- Summarize these points"

  # Multiple messages (interactive)
  {APP_NAME} "Read package.json" "What dependencies do we have?"

  # Continue previous session
  {APP_NAME} --continue "What did we discuss?"

  # Start a named session
  {APP_NAME} --name "Refactor auth module"

  # Use different model
  {APP_NAME} --provider openai --model gpt-4o-mini "Help me refactor this code"

  # Use model with provider prefix (no --provider needed)
  {APP_NAME} --model openai/gpt-4o "Help me refactor this code"

  # Use model with thinking level shorthand
  {APP_NAME} --model sonnet:high "Solve this complex problem"

  # Limit model cycling to specific models
  {APP_NAME} --models claude-sonnet,claude-haiku,gpt-4o

  # Limit to a specific provider with glob pattern
  {APP_NAME} --models "github-copilot/*"

  # Cycle models with fixed thinking levels
  {APP_NAME} --models sonnet:high,haiku:low

  # Start with a specific thinking level
  {APP_NAME} --thinking high "Solve this complex problem"

  # Read-only mode (no file modifications possible)
  {APP_NAME} --tools read,grep,find,ls -p "Review the code in src/"

  # Add codemode to the default tools
  {APP_NAME} --tools +codemode

  # Codemode with only the tools of one MCP server
  {APP_NAME} --tools read,bash,codemode,'mcp__radius__*'

  # Disable one tool while keeping the rest available
  {APP_NAME} --exclude-tools ask_question

  # Export a session file to HTML
  {APP_NAME} --export ~/{CONFIG_DIR_NAME}/agent/sessions/--path--/session.jsonl
  {APP_NAME} --export session.jsonl output.html

Environment Variables:
  ANTHROPIC_AUTH_TOKEN             - Anthropic bearer auth token
  ANTHROPIC_API_KEY                - Anthropic Claude API key
  ANTHROPIC_OAUTH_TOKEN            - Anthropic OAuth token (alternative to API key)
  ANT_LING_API_KEY                 - Ant Ling API key
  OPENAI_API_KEY                   - OpenAI GPT API key
  AZURE_OPENAI_API_KEY             - Azure OpenAI/Cognitive Services API key
  AZURE_OPENAI_BASE_URL            - Azure OpenAI/Cognitive Services base URL
  AZURE_OPENAI_RESOURCE_NAME       - Azure OpenAI resource name (alternative to base URL)
  AZURE_OPENAI_API_VERSION         - Azure OpenAI API version (default: v1)
  AZURE_OPENAI_DEPLOYMENT_NAME_MAP - Azure OpenAI model=deployment map (comma-separated)
  DEEPSEEK_API_KEY                 - DeepSeek API key
  NVIDIA_API_KEY                   - NVIDIA NIM API key
  GEMINI_API_KEY                   - Google Gemini API key
  GROQ_API_KEY                     - Groq API key
  CEREBRAS_API_KEY                 - Cerebras API key
  XAI_API_KEY                      - xAI Grok API key
  FIREWORKS_API_KEY                - Fireworks API key
  TOGETHER_API_KEY                 - Together AI API key
  BASETEN_API_KEY                  - Baseten API key
  OPENROUTER_API_KEY               - OpenRouter API key
  AI_GATEWAY_API_KEY               - Vercel AI Gateway API key
  ZAI_API_KEY                      - ZAI Coding Plan API key (Global)
  ZAI_CODING_CN_API_KEY            - ZAI Coding Plan API key (China)
  MISTRAL_API_KEY                  - Mistral API key
  MINIMAX_API_KEY                  - MiniMax API key
  MOONSHOT_API_KEY                 - Moonshot AI API key
  OPENCODE_API_KEY                 - OpenCode Zen/OpenCode Go API key
  KIMI_API_KEY                     - Kimi For Coding API key
  META_API_KEY                     - Meta Model API key
  CLOUDFLARE_API_KEY               - Cloudflare API token (Workers AI and AI Gateway)
  CLOUDFLARE_ACCOUNT_ID            - Cloudflare account id (required for both)
  CLOUDFLARE_GATEWAY_ID            - Cloudflare AI Gateway slug (required for AI Gateway)
  QWEN_TOKEN_PLAN_API_KEY          - Qwen Token Plan API key (international region)
  QWEN_TOKEN_PLAN_CN_API_KEY       - Qwen Token Plan API key (China region)
  XIAOMI_API_KEY                   - Xiaomi MiMo API key (api.xiaomimimo.com billing)
  XIAOMI_TOKEN_PLAN_CN_API_KEY     - Xiaomi MiMo Token Plan API key (China region)
  XIAOMI_TOKEN_PLAN_AMS_API_KEY    - Xiaomi MiMo Token Plan API key (Amsterdam region)
  XIAOMI_TOKEN_PLAN_SGP_API_KEY    - Xiaomi MiMo Token Plan API key (Singapore region)
  AWS_PROFILE                      - AWS profile for Amazon Bedrock
  AWS_ACCESS_KEY_ID                - AWS access key for Amazon Bedrock
  AWS_SECRET_ACCESS_KEY            - AWS secret key for Amazon Bedrock
  AWS_BEARER_TOKEN_BEDROCK         - Bedrock API key (bearer token)
  AWS_REGION                       - AWS region for Amazon Bedrock (e.g., us-east-1)
  {:32} - Config directory (default: ~/{CONFIG_DIR_NAME}/agent)
  {:32} - Session storage directory (overridden by --session-dir)
  PI_PACKAGE_DIR                   - Override package directory (for Nix/Guix store paths)
  PI_OFFLINE                       - Disable startup network operations when set to 1/true/yes
  PI_TELEMETRY                     - Override install telemetry when set to 1/true/yes or 0/false/no
  PI_SHARE_VIEWER_URL              - Base URL for /share command (default: https://pi.dev/session/)

Built-in Tool Names:
  read       - Read file contents
  bash       - Execute bash commands
  powershell - Execute PowerShell commands on Windows
  edit       - Edit files with find/replace
  write      - Write files (creates/overwrites)
  grep       - Search file contents (read-only, off by default)
  find       - Find files by glob pattern (read-only, off by default)
  ls         - List directory contents (read-only, off by default)
"#,
        ENV_AGENT_DIR, ENV_SESSION_DIR,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Args {
        let owned: Vec<String> = list.iter().map(|s| s.to_string()).collect();
        parse_args(&owned)
    }

    // ---- --version flag ----

    #[test]
    fn parses_version_flag() {
        let result = args(&["--version"]);
        assert!(result.version);
    }

    #[test]
    fn parses_v_shorthand() {
        let result = args(&["-v"]);
        assert!(result.version);
    }

    #[test]
    fn version_takes_precedence_over_other_args() {
        let result = args(&["--version", "--help", "some message"]);
        assert!(result.version);
        assert!(result.help);
        assert!(result.messages.contains(&"some message".to_string()));
    }

    // ---- --help flag ----

    #[test]
    fn parses_help_flag() {
        let result = args(&["--help"]);
        assert!(result.help);
    }

    #[test]
    fn parses_h_shorthand() {
        let result = args(&["-h"]);
        assert!(result.help);
    }

    // ---- --print flag ----

    #[test]
    fn parses_print_flag() {
        let result = args(&["--print"]);
        assert!(result.print);
    }

    #[test]
    fn parses_p_shorthand() {
        let result = args(&["-p"]);
        assert!(result.print);
    }

    #[test]
    fn parses_prompt_after_p_even_when_it_starts_with_yaml_frontmatter() {
        let prompt = "---\ntitle: hello\n---\nSay hi.";
        let result = args(&["-p", prompt]);
        assert!(result.print);
        assert_eq!(result.messages, vec![prompt.to_string()]);
        assert!(result.unknown_flags.is_empty());
    }

    #[test]
    fn does_not_consume_options_after_p_as_prompts() {
        let result = args(&["-p", "--provider", "openai", "Say hi."]);
        assert!(result.print);
        assert_eq!(result.provider.as_deref(), Some("openai"));
        assert_eq!(result.messages, vec!["Say hi.".to_string()]);
    }

    // ---- --continue / --resume ----

    #[test]
    fn parses_continue_flag() {
        assert!(args(&["--continue"]).continue_);
        assert!(args(&["-c"]).continue_);
    }

    #[test]
    fn parses_resume_flag() {
        assert!(args(&["--resume"]).resume);
        assert!(args(&["-r"]).resume);
    }

    // ---- flags with values ----

    #[test]
    fn parses_value_flags() {
        assert_eq!(
            args(&["--provider", "openai"]).provider.as_deref(),
            Some("openai")
        );
        assert_eq!(
            args(&["--model", "gpt-4o"]).model.as_deref(),
            Some("gpt-4o")
        );
        assert_eq!(
            args(&["--api-key", "sk-test-key"]).api_key.as_deref(),
            Some("sk-test-key")
        );
        assert_eq!(
            args(&["--system-prompt", "You are a helpful assistant"])
                .system_prompt
                .as_deref(),
            Some("You are a helpful assistant")
        );
        assert_eq!(
            args(&["--append-system-prompt", "Additional context"]).append_system_prompt,
            Some(vec!["Additional context".to_string()])
        );
        assert_eq!(
            args(&[
                "--append-system-prompt",
                "Context A",
                "--append-system-prompt",
                "Context B"
            ])
            .append_system_prompt,
            Some(vec!["Context A".to_string(), "Context B".to_string()])
        );
        assert_eq!(
            args(&["--session", "/path/to/session.jsonl"])
                .session
                .as_deref(),
            Some("/path/to/session.jsonl")
        );
        assert_eq!(
            args(&["--session-id", "orchestrated-session"])
                .session_id
                .as_deref(),
            Some("orchestrated-session")
        );
        let fork = args(&["--fork", "1234abcd"]);
        assert_eq!(fork.fork.as_deref(), Some("1234abcd"));
        assert!(fork.messages.is_empty());
        assert_eq!(
            args(&["--export", "session.jsonl"]).export.as_deref(),
            Some("session.jsonl")
        );
        assert!(args(&["--thinking", "high"]).thinking.is_some());
        assert_eq!(
            args(&["--models", "gpt-4o,claude-sonnet,gemini-pro"]).models,
            Some(vec![
                "gpt-4o".to_string(),
                "claude-sonnet".to_string(),
                "gemini-pro".to_string()
            ])
        );
    }

    // Issue #10334
    #[test]
    fn ignores_empty_entries_in_models() {
        assert_eq!(
            args(&["--models", "gpt-4o, ,claude-sonnet,"]).models,
            Some(vec!["gpt-4o".to_string(), "claude-sonnet".to_string()])
        );
    }

    // Issue #9045: --mode flag
    #[test]
    fn parses_mode_values() {
        for (value, expected) in [
            ("text", Mode::Text),
            ("json", Mode::Json),
            ("rpc", Mode::Rpc),
        ] {
            let result = args(&["--mode", value]);
            assert_eq!(result.mode, Some(expected));
            assert!(result.diagnostics.is_empty());
        }
    }

    #[test]
    fn rejects_invalid_mode_value() {
        for mode in ["yaml", ""] {
            let result = args(&["--mode", mode, "--version"]);
            assert_eq!(result.mode, None);
            assert!(result.version);
            assert!(result.messages.is_empty());
            assert!(result.unknown_flags.is_empty());
            assert_eq!(
                result.diagnostics,
                vec![Diagnostic::error(format!(
                    "Invalid mode \"{mode}\". Valid values: text, json, rpc"
                ))]
            );
        }
    }

    #[test]
    fn reports_a_missing_mode_value() {
        let result = args(&["--mode"]);
        assert_eq!(result.mode, None);
        assert!(result.unknown_flags.is_empty());
        assert_eq!(
            result.diagnostics,
            vec![Diagnostic::error("--mode requires text, json, or rpc")]
        );
    }

    #[test]
    fn does_not_consume_another_option_as_a_mode_value() {
        let result = args(&["--mode", "--version"]);
        assert_eq!(result.mode, None);
        assert!(result.version);
        assert!(result.unknown_flags.is_empty());
        assert_eq!(
            result.diagnostics,
            vec![Diagnostic::error("--mode requires text, json, or rpc")]
        );
    }

    #[test]
    fn reports_an_invalid_mode_value_after_a_valid_one() {
        let result = args(&["--mode", "json", "--mode", "yaml"]);
        assert_eq!(
            result.diagnostics,
            vec![Diagnostic::error(
                "Invalid mode \"yaml\". Valid values: text, json, rpc"
            )]
        );
    }

    // ---- --name flag ----

    #[test]
    fn parses_name_flag() {
        assert_eq!(
            args(&["--name", "my-session"]).name.as_deref(),
            Some("my-session")
        );
        assert_eq!(
            args(&["-n", "quick-session"]).name.as_deref(),
            Some("quick-session")
        );
    }

    #[test]
    fn preserves_empty_values_for_main_validation() {
        assert_eq!(args(&["--name", ""]).name.as_deref(), Some(""));
    }

    #[test]
    fn normalizes_display_names_and_rejects_whitespace_only_values() {
        assert_eq!(
            normalize_session_name("  named session  ").as_deref(),
            Some("named session")
        );
        assert_eq!(normalize_session_name("   "), None);
    }

    #[test]
    fn reports_missing_name_value() {
        let result = args(&["--name"]);
        assert_eq!(
            result.diagnostics,
            vec![Diagnostic::error("--name requires a value")]
        );
    }

    #[test]
    fn name_works_alongside_other_flags() {
        let result = args(&[
            "--name",
            "named-run",
            "--print",
            "--model",
            "gpt-4o",
            "hello",
        ]);
        assert_eq!(result.name.as_deref(), Some("named-run"));
        assert!(result.print);
        assert_eq!(result.model.as_deref(), Some("gpt-4o"));
        assert_eq!(result.messages, vec!["hello".to_string()]);
    }

    // ---- --no-session flag ----

    #[test]
    fn parses_no_session_flag() {
        assert!(args(&["--no-session"]).no_session);
    }

    #[test]
    fn preserves_custom_session_ids_for_non_persisting_commands() {
        let r = args(&["--session-id", "ephemeral-id", "--help"]);
        assert_eq!(r.session_id.as_deref(), Some("ephemeral-id"));
        assert!(r.help);

        let r = args(&["--session-id", "ephemeral-id", "--list-models"]);
        assert_eq!(r.session_id.as_deref(), Some("ephemeral-id"));
        assert_eq!(r.list_models, Some(None));

        let r = args(&["--session-id", "ephemeral-id", "--no-session"]);
        assert_eq!(r.session_id.as_deref(), Some("ephemeral-id"));
        assert!(r.no_session);
    }

    // ---- --extension / --no-extensions ----

    #[test]
    fn parses_extension_flags() {
        assert_eq!(
            args(&["--extension", "./my-extension.ts"]).extensions,
            Some(vec!["./my-extension.ts".to_string()])
        );
        assert_eq!(
            args(&["-e", "./my-extension.ts"]).extensions,
            Some(vec!["./my-extension.ts".to_string()])
        );
        assert_eq!(
            args(&["--extension", "./ext1.ts", "-e", "./ext2.ts"]).extensions,
            Some(vec!["./ext1.ts".to_string(), "./ext2.ts".to_string()])
        );
    }

    #[test]
    fn parses_no_extensions_with_explicit_e_flags() {
        let result = args(&["--no-extensions", "-e", "foo.ts", "-e", "bar.ts"]);
        assert!(result.no_extensions);
        assert_eq!(
            result.extensions,
            Some(vec!["foo.ts".to_string(), "bar.ts".to_string()])
        );
    }

    // ---- list-ish repeatable flags ----

    #[test]
    fn parses_skill_prompt_template_theme_flags() {
        assert_eq!(
            args(&["--skill", "./skill-dir"]).skills,
            Some(vec!["./skill-dir".to_string()])
        );
        assert_eq!(
            args(&["--skill", "./skill-a", "--skill", "./skill-b"]).skills,
            Some(vec!["./skill-a".to_string(), "./skill-b".to_string()])
        );
        assert_eq!(
            args(&["--prompt-template", "./prompts"]).prompt_templates,
            Some(vec!["./prompts".to_string()])
        );
        assert_eq!(
            args(&["--prompt-template", "./one", "--prompt-template", "./two"]).prompt_templates,
            Some(vec!["./one".to_string(), "./two".to_string()])
        );
        assert_eq!(
            args(&["--theme", "./theme.json"]).themes,
            Some(vec!["./theme.json".to_string()])
        );
        assert_eq!(
            args(&["--theme", "./dark.json", "--theme", "./light.json"]).themes,
            Some(vec!["./dark.json".to_string(), "./light.json".to_string()])
        );
    }

    // ---- --use-theme flag ----

    #[test]
    fn parses_use_theme() {
        assert_eq!(
            args(&["--use-theme", "light"]).use_theme.as_deref(),
            Some("light")
        );
    }

    #[test]
    fn reports_when_the_theme_name_value_is_missing() {
        let result = args(&["--use-theme", "--print"]);
        assert_eq!(result.use_theme, None);
        assert!(result.print);
        assert_eq!(
            result.diagnostics,
            vec![Diagnostic::error("--use-theme requires a theme name")]
        );
    }

    // ---- negation flags ----

    #[test]
    fn parses_negation_flags() {
        assert!(args(&["--no-skills"]).no_skills);
        assert!(args(&["--no-prompt-templates"]).no_prompt_templates);
        assert!(args(&["--no-themes"]).no_themes);
        assert!(args(&["--no-context-files"]).no_context_files);
        assert!(args(&["-nc"]).no_context_files);
    }

    // ---- project approval flags ----

    #[test]
    fn parses_project_approval_flags() {
        assert_eq!(args(&["--approve"]).project_trust_override, Some(true));
        assert_eq!(args(&["-a"]).project_trust_override, Some(true));
        assert_eq!(args(&["--no-approve"]).project_trust_override, Some(false));
        assert_eq!(args(&["-na"]).project_trust_override, Some(false));
    }

    // ---- --verbose / --offline ----

    #[test]
    fn parses_verbose_and_offline_flags() {
        assert!(args(&["--verbose"]).verbose);
        assert!(args(&["--offline"]).offline);
    }

    // ---- --tui-mode flag ----

    #[test]
    fn parses_tui_modes() {
        assert_eq!(
            args(&["--tui-mode", "regular"]).tui_mode,
            Some(crate::core::settings_manager::TuiMode::Regular)
        );
        assert_eq!(
            args(&["--tui-mode", "fullscreen"]).tui_mode,
            Some(crate::core::settings_manager::TuiMode::Fullscreen)
        );
    }

    #[test]
    fn rejects_invalid_tui_modes() {
        let result = args(&["--tui-mode", "other"]);
        assert_eq!(
            result.diagnostics,
            vec![Diagnostic::error(
                "Invalid TUI mode \"other\". Valid values: regular, fullscreen"
            )]
        );
    }

    #[test]
    fn requires_a_tui_mode() {
        let result = args(&["--tui-mode"]);
        assert_eq!(
            result.diagnostics,
            vec![Diagnostic::error(
                "--tui-mode requires regular or fullscreen"
            )]
        );
    }

    #[test]
    fn does_not_recognize_the_old_ui_mode_flag() {
        let result = args(&["--ui-mode", "fullscreen"]);
        assert_eq!(result.tui_mode, None);
        assert_eq!(
            result.unknown_flags.get("ui-mode"),
            Some(&UnknownFlagValue::Str("fullscreen".to_string()))
        );
    }

    // ---- tool flags ----

    #[test]
    fn parses_tool_flags() {
        assert!(args(&["--no-tools"]).no_tools);
        assert!(args(&["-nt"]).no_tools);
        assert!(args(&["--no-builtin-tools"]).no_builtin_tools);
        assert!(args(&["-nbt"]).no_builtin_tools);
        assert_eq!(
            args(&["--tools", "read,bash"]).tools,
            Some(vec!["read".to_string(), "bash".to_string()])
        );
        assert_eq!(
            args(&["-t", "read,bash"]).tools,
            Some(vec!["read".to_string(), "bash".to_string()])
        );
        assert_eq!(
            args(&["--exclude-tools", "read,bash"]).exclude_tools,
            Some(vec!["read".to_string(), "bash".to_string()])
        );
        assert_eq!(
            args(&["-xt", "read,bash"]).exclude_tools,
            Some(vec!["read".to_string(), "bash".to_string()])
        );

        let result = args(&["--no-tools", "--tools", "read,bash"]);
        assert!(result.no_tools);
        assert_eq!(
            result.tools,
            Some(vec!["read".to_string(), "bash".to_string()])
        );

        let result = args(&["--no-builtin-tools", "--tools", "read,bash"]);
        assert!(result.no_builtin_tools);
        assert_eq!(
            result.tools,
            Some(vec!["read".to_string(), "bash".to_string()])
        );
    }

    #[test]
    fn parses_no_mcp_flag() {
        let result = args(&["--no-mcp"]);
        assert!(result.no_mcp);
        assert!(result.unknown_flags.is_empty());
    }

    #[test]
    fn parses_plus_and_minus_tool_modifiers() {
        let result = args(&["-t", "+codemode,-write"]);
        assert_eq!(
            result.tools,
            Some(vec!["+codemode".to_string(), "-write".to_string()])
        );
        assert!(result.diagnostics.is_empty());
    }

    #[test]
    fn rejects_tool_names_mixed_with_modifiers() {
        let result = args(&["--tools", "read,+codemode"]);
        assert_eq!(result.tools, None);
        assert_eq!(
            result.diagnostics,
            vec![Diagnostic::error(
                "--tools: tool names cannot be mixed with +name or -name entries"
            )]
        );
    }

    #[test]
    fn rejects_patterns_in_tool_modifiers() {
        let result = args(&["-t", "+mcp__radius__*"]);
        assert_eq!(result.tools, None);
        assert_eq!(
            result.diagnostics,
            vec![Diagnostic::error(
                "-t: +name and -name entries take exact tool names, not patterns: +mcp__radius__*"
            )]
        );
    }

    // ---- messages and file args ----

    #[test]
    fn parses_plain_text_messages() {
        assert_eq!(
            args(&["hello", "world"]).messages,
            vec!["hello".to_string(), "world".to_string()]
        );
    }

    #[test]
    fn parses_file_arguments() {
        assert_eq!(
            args(&["@README.md", "@src/main.ts"]).file_args,
            vec!["README.md".to_string(), "src/main.ts".to_string()]
        );
    }

    #[test]
    fn parses_mixed_messages_and_file_args() {
        let result = args(&["@file.txt", "explain this", "@image.png"]);
        assert_eq!(
            result.file_args,
            vec!["file.txt".to_string(), "image.png".to_string()]
        );
        assert_eq!(result.messages, vec!["explain this".to_string()]);
    }

    #[test]
    fn captures_unknown_long_flags_with_string_values() {
        let result = args(&["--unknown-flag", "message"]);
        assert!(result.messages.is_empty());
        assert_eq!(
            result.unknown_flags.get("unknown-flag"),
            Some(&UnknownFlagValue::Str("message".to_string()))
        );
    }

    #[test]
    fn captures_unknown_boolean_long_flags() {
        let result = args(&["--unknown-flag"]);
        assert_eq!(
            result.unknown_flags.get("unknown-flag"),
            Some(&UnknownFlagValue::Bool)
        );
    }

    #[test]
    fn captures_unknown_long_flags_with_equals_syntax() {
        let result = args(&["--unknown-flag=value"]);
        assert_eq!(
            result.unknown_flags.get("unknown-flag"),
            Some(&UnknownFlagValue::Str("value".to_string()))
        );
    }

    // ---- complex combinations ----

    #[test]
    fn parses_multiple_flags_together() {
        let result = args(&[
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet",
            "--print",
            "--thinking",
            "high",
            "@prompt.md",
            "Do the task",
        ]);
        assert_eq!(result.provider.as_deref(), Some("anthropic"));
        assert_eq!(result.model.as_deref(), Some("claude-sonnet"));
        assert!(result.print);
        assert!(result.thinking.is_some());
        assert_eq!(result.file_args, vec!["prompt.md".to_string()]);
        assert_eq!(result.messages, vec!["Do the task".to_string()]);
    }

    #[test]
    fn thinking_level_round_trips() {
        for level in VALID_THINKING_LEVELS {
            assert!(is_valid_thinking_level(level));
        }
        assert!(!is_valid_thinking_level("ultra"));
    }
}
