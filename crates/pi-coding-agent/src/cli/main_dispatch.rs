//! `main.ts` — CLI entry orchestration, minus interactive-mode dispatch.
//!
//! Mode dispatch seam (**D102**): upstream ends `main()` by constructing
//! `InteractiveMode`/`runPrintMode`/`runRpcMode` from `modes/`. The modes
//! module ports separately; this module consumes it through the
//! [`ModeDispatcher`] trait installed with [`set_mode_dispatcher`]. Without
//! a dispatcher, print/json/rpc modes exit with an error naming the seam;
//! the `pi` binary target (D109) installs the production runners.
//!
//! Further documented omissions (see crate docs D103/D104):
//! - `takeOverStdout`/`restoreStdout`, http dispatcher/proxy configuration,
//!   timing probes, theme init, and `setCapabilityOverrides` are
//!   interactive/TUI-adjacent infra and are not invoked.
//! - `install`/`remove`/`update`/`list`/`config`/`mcp` subcommands
//!   (package-manager-cli.ts, mcp cli) are deferred modules (D104).
//! - `--export` is deferred with HTML export (D79).
//! - `--resume` needs the TUI session picker (excluded); it reports an
//!   error like other unavailable interactive surfaces.
//! - Project trust uses `--approve`/`--no-approve`; the `ProjectTrustStore`
//!   and trust-manager modules are deferred (D103).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use pi_agent::types::AgentThinkingLevel;
use pi_ai::models::Models as _;

use crate::cli::args::{
	parse_args, print_help, normalize_session_name, Args, DiagnosticType, ExtensionFlagInfo, Mode,
	UnknownFlagValue,
};
use crate::cli::auth_check::{
	check_provider_auth, create_auth_check_model_runtime, get_provider_credential, AuthCheckResult,
	AuthCheckReason, AuthCheckStatus,
};
use crate::cli::auth_command::{
	get_auth_command_name, get_auth_command_usage, is_auth_command_help, parse_auth_command,
	print_auth_command_help, validate_auth_command_args, AuthCommandError, AuthCommandKind,
};
use crate::cli::credential_print::resolve_credential_for_print;
use crate::cli::file_processor::{process_file_arguments, ProcessFileOptions};
use crate::cli::initial_message::{build_initial_message, InitialMessageInput};
use crate::cli::list_models::list_models;
use crate::core::agent_session::ScopedModelEntry;
use crate::core::agent_session_runtime::{
	create_agent_session_runtime, AgentSessionRuntime, CreateAgentSessionRuntimeFactory,
	CreateAgentSessionRuntimeOptions, CreateAgentSessionRuntimeResult,
};
use crate::core::agent_session_services::{
	create_agent_session_from_services, create_agent_session_services, AgentSessionRuntimeDiagnostic,
	CreateAgentSessionFromServicesOptions, CreateAgentSessionServicesOptions, DiagnosticType as RuntimeDiagnosticType,
	ResourceLoaderOptionsOverride,
};
use crate::core::resource_loader::ResourceLoaderReloadOptions;
use crate::core::auth_storage::{AuthStorage, ReadOnlyAuthStorage};
use crate::core::model_resolver::{resolve_cli_model, resolve_model_scope_with_diagnostics, ResolveCliModelOptions};
use crate::core::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::core::sdk::NoToolsMode;
use crate::core::session_cwd::{get_missing_session_cwd_issue, MissingSessionCwdError};
use crate::core::session_manager::{assert_valid_session_id, NewSessionOptions, SessionManager};
use crate::core::settings_manager::{SettingsManager, SettingsManagerCreateOptions};
use crate::migrations::run_migrations;
use crate::{config, migrations};

const EXTENSION_LOAD_FAILURE_HINT: &str = "Hint: Start without extensions using \"pi -ne\".";

/// Effective application mode (`AppMode` from core/project-trust.ts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
	Print,
	Json,
	Rpc,
	Interactive,
}

impl AppMode {
	pub fn as_str(self) -> &'static str {
		match self {
			AppMode::Print => "print",
			AppMode::Json => "json",
			AppMode::Rpc => "rpc",
			AppMode::Interactive => "interactive",
		}
	}
}

/// Options for the print-mode runner (`runPrintMode` options).
pub struct PrintModeOptions {
	pub json: bool,
	pub messages: Vec<String>,
	pub initial_message: Option<String>,
	pub initial_images: Vec<pi_ai::types::ImageContent>,
}

/// Installed print/rpc mode runners (**D102** seam). The runtime is passed
/// by value: the production installer (the `pi` binary, D109) wraps it in
/// the owning [`crate::modes::AgentSessionRuntimeHandle`].
pub trait ModeDispatcher: Send + Sync {
	/// Run print mode; returns the process exit code (`runPrintMode`).
	fn run_print_mode(&self, runtime: AgentSessionRuntime, options: PrintModeOptions) -> i32;
	/// Run RPC mode until the connection closes (`runRpcMode`).
	fn run_rpc_mode(&self, runtime: AgentSessionRuntime);
}

static MODE_DISPATCHER: OnceLock<Box<dyn ModeDispatcher>> = OnceLock::new();

/// Install the modes-module dispatcher. First installation wins.
pub fn set_mode_dispatcher(dispatcher: Box<dyn ModeDispatcher>) {
	let _ = MODE_DISPATCHER.set(dispatcher);
}

fn mode_dispatcher() -> Option<&'static dyn ModeDispatcher> {
	MODE_DISPATCHER.get().map(|boxed| boxed.as_ref())
}

/// `readPipedStdin` — read all content from piped stdin; `None` on a TTY.
async fn read_piped_stdin() -> Option<String> {
	// If stdin is a TTY, we're running interactively — don't read stdin.
	if !use_stdin_pipe() {
		return None;
	}
	let mut buffer = String::new();
	match tokio::io::AsyncReadExt::read_to_string(&mut tokio::io::stdin(), &mut buffer).await {
		Ok(_) => {
			let trimmed = buffer.trim();
			if trimmed.is_empty() { None } else { Some(trimmed.to_string()) }
		}
		Err(_) => None,
	}
}

#[cfg(unix)]
fn use_stdin_pipe() -> bool {
	// SAFETY: `isatty` on fd 0 has no preconditions.
	unsafe { libc::isatty(0) == 0 }
}

#[cfg(not(unix))]
fn use_stdin_pipe() -> bool {
	true
}

#[cfg(unix)]
fn stdout_is_tty() -> bool {
	// SAFETY: `isatty` on fd 1 has no preconditions.
	unsafe { libc::isatty(1) == 1 }
}

#[cfg(not(unix))]
fn stdout_is_tty() -> bool {
	true
}

fn report_diagnostics(diagnostics: &[AgentSessionRuntimeDiagnostic]) {
	for diagnostic in diagnostics {
		let prefix = match diagnostic.diagnostic_type {
			RuntimeDiagnosticType::Error => "Error: ",
			RuntimeDiagnosticType::Warning => "Warning: ",
			_ => "",
		};
		eprintln!("{prefix}{}", diagnostic.message);
	}
}

fn is_truthy_env_flag(value: Option<&str>) -> bool {
	match value {
		Some(value) => value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes"),
		None => false,
	}
}

/// `resolveAppMode`.
pub fn resolve_app_mode(parsed: &Args, stdin_is_tty: bool, stdout_is_tty: bool) -> AppMode {
	match parsed.mode {
		Some(Mode::Rpc) => return AppMode::Rpc,
		Some(Mode::Json) => return AppMode::Json,
		_ => {}
	}
	if parsed.print || !stdin_is_tty || !stdout_is_tty {
		return AppMode::Print;
	}
	AppMode::Interactive
}

fn to_print_output_mode(app_mode: AppMode) -> bool {
	app_mode == AppMode::Json
}

/// `runAuthCommand` — handles `pi auth ...`; `Ok(true)` means handled.
/// Returns the process exit code as `Err` when upstream would set
/// `process.exitCode` and return.
async fn run_auth_command(args: &[String]) -> Result<bool, i32> {
	if is_auth_command_help(args) {
		print_auth_command_help();
		return Ok(true);
	}

	let command = match parse_auth_command(args) {
		Err(error) => {
			eprintln!("Error: {error}");
			return Err(1);
		}
		Ok(None) => return Ok(false),
		Ok(Some(command)) => command,
	};

	let parsed = parse_args(&command.args);
	if let Some((option, _)) = parsed.unknown_flags.iter().next() {
		eprintln!("Error: Unknown option --{option} for \"{}\".", get_auth_command_name(command.kind));
		eprintln!("Use \"{} --help\" or \"{}\".", config::APP_NAME, get_auth_command_usage(command.kind));
		return Err(1);
	}
	if !parsed.diagnostics.is_empty() {
		let message = parsed
			.diagnostics
			.iter()
			.map(|diagnostic| diagnostic.message.clone())
			.collect::<Vec<_>>()
			.join("\n");
		eprintln!("Error: {message}");
		return Err(if command.kind == AuthCommandKind::Check { 2 } else { 1 });
	}

	let outcome: Result<(AuthCheckResult, Option<String>), AuthCommandError> = async {
		if command.kind != AuthCommandKind::Check {
			let model_runtime = ModelRuntime::create(CreateModelRuntimeOptions {
				allow_model_network: false,
				model_refresh_timeout_ms: Some(15_000),
				..Default::default()
			})
			.await
			.map_err(|_| AuthCommandError("Failed to resolve credential".to_string()))?;
			let credential = resolve_credential_for_print(
				&parsed,
				&model_runtime,
				command.kind,
				command.min_expiry_ms,
				None,
			)
			.await?;
			println!("{credential}");
			std::process::exit(if command.kind == AuthCommandKind::Check { 2 } else { 1 });
		}

		let requested_auth = validate_auth_command_args(&parsed, command.kind)?;
		let outcome = async {
			let (credentials, model_runtime) = if command.no_refresh {
				let storage = Arc::new(ReadOnlyAuthStorage::new(&config::get_auth_path()));
				let runtime = create_auth_check_model_runtime(storage)
					.await
					.map_err(|_| AuthCommandError("Failed to create model runtime".to_string()))?;
				(None, runtime)
			} else {
				let credentials = Arc::new(AuthStorage::new(None));
				let runtime = create_auth_check_model_runtime(credentials.clone())
					.await
					.map_err(|_| AuthCommandError("Failed to create model runtime".to_string()))?;
				(Some(credentials), runtime)
			};
			let mut result = check_provider_auth(&parsed, model_runtime.as_ref(), !command.no_refresh).await?;
			let mut credential: Option<String> = None;
			if command.credentials && result.status == AuthCheckStatus::Ready {
				if let Some(credentials) = &credentials {
					credential = get_provider_credential(
						&result.provider,
						model_runtime.as_ref(),
						credentials,
						!command.no_refresh,
					)
					.await;
					if credential.is_none() {
						result = AuthCheckResult::new(
							AuthCheckStatus::NotReady,
							result.provider.clone(),
							Some(AuthCheckReason::CredentialNotAvailable),
						);
					}
				}
			}
			Ok::<_, AuthCommandError>((result, credential))
		}
		.await;
		match outcome {
			Ok(pair) => Ok(pair),
			Err(_) => Ok((
				AuthCheckResult::new(
					AuthCheckStatus::Invalid,
					requested_auth.0.or(requested_auth.1).unwrap_or_default(),
					Some(AuthCheckReason::InvalidState),
				),
				None,
			)),
		}
	}
	.await;

	let (result, credential) = match outcome {
		Ok(pair) => pair,
		Err(error) => {
			eprintln!("Error: {error}");
			return Err(if command.kind == AuthCommandKind::Check { 2 } else { 1 });
		}
	};

	let output = if command.json {
		let mut object = serde_json::Map::new();
		object.insert(
			"status".to_string(),
			serde_json::Value::String(
				match result.status {
					AuthCheckStatus::Ready => "ready",
					AuthCheckStatus::NotReady => "not_ready",
					AuthCheckStatus::Invalid => "invalid",
				}
				.to_string(),
			),
		);
		object.insert("provider".to_string(), serde_json::Value::String(result.provider.clone()));
		if let Some(reason) = result.reason {
			object.insert(
				"reason".to_string(),
				serde_json::Value::String(
					match reason {
						AuthCheckReason::ProviderNotFound => "provider_not_found",
						AuthCheckReason::CredentialsNotConfigured => "credentials_not_configured",
						AuthCheckReason::CredentialNotAvailable => "credential_not_available",
						AuthCheckReason::InvalidState => "invalid_state",
					}
					.to_string(),
				),
			);
		}
		if let Some(auth_type) = result.auth_type {
			object.insert(
				"authType".to_string(),
				serde_json::Value::String(if auth_type == pi_ai::auth::types::AuthType::ApiKey {
					"api_key".to_string()
				} else {
					"oauth".to_string()
				}),
			);
		}
		if let Some(credential) = &credential {
			object.insert("credentials".to_string(), serde_json::Value::String(credential.clone()));
		}
		serde_json::Value::Object(object).to_string()
	} else {
		credential.unwrap_or_else(|| {
			match result.status {
				AuthCheckStatus::Ready => "ready",
				AuthCheckStatus::NotReady => "not_ready",
				AuthCheckStatus::Invalid => "invalid",
			}
			.to_string()
		})
	};
	println!("{output}");
	std::process::exit(match result.status {
		AuthCheckStatus::Ready => 0,
		AuthCheckStatus::NotReady => 1,
		AuthCheckStatus::Invalid => 2,
	});
}

/// `prepareInitialMessage`.
async fn prepare_initial_message(
	parsed: &mut Args,
	stdin_content: Option<String>,
) -> (Option<String>, Vec<pi_ai::types::ImageContent>) {
	if parsed.file_args.is_empty() {
		let result = build_initial_message(InitialMessageInput {
			parsed,
			file_text: None,
			file_images: Vec::new(),
			stdin_content,
		});
		return (result.initial_message, result.initial_images.unwrap_or_default());
	}

	// AgentSession resizes these after extension hooks select the request
	// model.
	let processed = match process_file_arguments(&parsed.file_args, Some(ProcessFileOptions { auto_resize_images: false }))
		.await
	{
		Ok(processed) => processed,
		Err(error) => {
			eprintln!("{error}");
			std::process::exit(1);
		}
	};
	let result = build_initial_message(InitialMessageInput {
		parsed,
		file_text: Some(processed.text),
		file_images: processed.images,
		stdin_content,
	});
	(result.initial_message, result.initial_images.unwrap_or_default())
}

/// Session resolution result (`ResolvedSession`).
enum ResolvedSession {
	/// Direct file path.
	Path(String),
	/// Found in current project.
	Local(String),
	/// Found in a different project.
	Global { path: String, cwd: String },
	/// Not found anywhere.
	NotFound(String),
}

fn find_local_session_by_exact_id(session_id: &str, cwd: &str, session_dir: Option<&str>) -> Option<String> {
	SessionManager::find_by_id(cwd, session_id, session_dir).ok().flatten()
}

/// `resolveSessionPath`.
async fn resolve_session_path(session_arg: &str, cwd: &str, session_dir: Option<&str>) -> ResolvedSession {
	// If it looks like a file path, resolve it before handing it to the
	// session manager.
	if session_arg.contains('/') || session_arg.contains('\\') || session_arg.ends_with(".jsonl") {
		return ResolvedSession::Path(crate::utils::paths::resolve_path(session_arg, cwd));
	}

	// Exact IDs only require reading session headers. Fall back to the full
	// metadata listing for prefix matches.
	if let Some(path) = find_local_session_by_exact_id(session_arg, cwd, session_dir) {
		return ResolvedSession::Local(path);
	}

	if let Ok(local_sessions) = SessionManager::list(cwd, session_dir, None, None).await {
		if let Some(local_match) = local_sessions.into_iter().find(|s| s.id.starts_with(session_arg)) {
			return ResolvedSession::Local(local_match.path);
		}
	}

	// Try global search across all projects
	if let Ok(all_sessions) = SessionManager::list_all(None, None).await {
		let global_match = all_sessions
			.iter()
			.find(|s| s.id == session_arg)
			.or_else(|| all_sessions.iter().find(|s| s.id.starts_with(session_arg)));
		if let Some(global_match) = global_match {
			return ResolvedSession::Global {
				path: global_match.path.clone(),
				cwd: global_match.cwd.clone(),
			};
		}
	}

	ResolvedSession::NotFound(session_arg.to_string())
}

/// `promptConfirm` — yes/no confirmation from stdin.
async fn prompt_confirm(message: &str) -> bool {
	use tokio::io::AsyncBufReadExt;
	eprintln!("{message} [y/N] ");
	let mut line = String::new();
	let _ = tokio::io::BufReader::new(tokio::io::stdin()).read_line(&mut line).await;
	let answer = line.trim().to_lowercase();
	answer == "y" || answer == "yes"
}

fn fail(message: &str) -> ! {
	eprintln!("Error: {message}");
	std::process::exit(1);
}

fn validate_fork_flags(parsed: &Args) {
	if parsed.fork.is_none() {
		return;
	}

	let conflicting_flags: Vec<&str> = [
		parsed.session.as_ref().map(|_| "--session"),
		parsed.continue_.then_some("--continue"),
		parsed.resume.then_some("--resume"),
		parsed.no_session.then_some("--no-session"),
	]
	.into_iter()
	.flatten()
	.collect();

	if !conflicting_flags.is_empty() {
		fail(&format!("--fork cannot be combined with {}", conflicting_flags.join(", ")));
	}
}

fn validate_session_id_flags(parsed: &Args) {
	let Some(session_id) = &parsed.session_id else { return };

	let conflicting_flags: Vec<&str> = [
		parsed.session.as_ref().map(|_| "--session"),
		parsed.continue_.then_some("--continue"),
		parsed.resume.then_some("--resume"),
	]
	.into_iter()
	.flatten()
	.collect();

	if !conflicting_flags.is_empty() {
		fail(&format!("--session-id cannot be combined with {}", conflicting_flags.join(", ")));
	}

	if let Err(error) = assert_valid_session_id(session_id) {
		fail(&error.to_string());
	}
}

/// `createSessionManager`.
pub async fn create_session_manager(
	parsed: &Args,
	cwd: &str,
	session_dir: Option<&str>,
	_settings_manager: &SettingsManager,
) -> SessionManager {
	let new_session_options = |id: &Option<String>| {
		id.clone().map(|id| NewSessionOptions { id: Some(id), parent_session: None })
	};

	if parsed.no_session || parsed.help || parsed.list_models.is_some() {
		return SessionManager::in_memory(Some(cwd), new_session_options(&parsed.session_id), None)
			.unwrap_or_else(|error| fail(&error.to_string()));
	}

	if let Some(fork) = parsed.fork.clone() {
		if let Some(session_id) = &parsed.session_id {
			if find_local_session_by_exact_id(session_id, cwd, session_dir).is_some() {
				fail(&format!("Session already exists with id '{session_id}'"));
			}
		}

		match resolve_session_path(&fork, cwd, session_dir).await {
			ResolvedSession::Path(path) | ResolvedSession::Local(path) | ResolvedSession::Global { path, .. } => {
				return SessionManager::fork_from(
					&path,
					cwd,
					session_dir,
					new_session_options(&parsed.session_id),
				)
				.unwrap_or_else(|error| fail(&error.to_string()));
			}
			ResolvedSession::NotFound(arg) => {
				fail(&format!("No session found matching '{arg}'"));
			}
		}
	}

	if let Some(session) = parsed.session.clone() {
		match resolve_session_path(&session, cwd, session_dir).await {
			ResolvedSession::Path(path) | ResolvedSession::Local(path) => {
				return SessionManager::open(&path, session_dir, None)
					.unwrap_or_else(|error| fail(&error.to_string()));
			}
			ResolvedSession::Global { path, cwd: session_cwd } => {
				println!("Session found in different project: {session_cwd}");
				if !prompt_confirm("Fork this session into current directory?").await {
					println!("Aborted.");
					std::process::exit(0);
				}
				return SessionManager::fork_from(&path, cwd, session_dir, None)
					.unwrap_or_else(|error| fail(&error.to_string()));
			}
			ResolvedSession::NotFound(arg) => {
				fail(&format!("No session found matching '{arg}'"));
			}
		}
	}

	if parsed.resume {
		// The TUI session picker is excluded with interactive mode; `--resume`
		// has no non-interactive selection surface.
		fail("--resume requires the interactive session picker (not available in this build).");
	}

	if parsed.continue_ {
		return SessionManager::continue_recent(cwd, session_dir)
			.unwrap_or_else(|error| fail(&error.to_string()));
	}

	if let Some(session_id) = &parsed.session_id {
		if let Some(path) = find_local_session_by_exact_id(session_id, cwd, session_dir) {
			return SessionManager::open(&path, session_dir, None)
				.unwrap_or_else(|error| fail(&error.to_string()));
		}
		eprintln!("Warning: No project session found with id '{session_id}'; creating a new session with that id.");
	}

	SessionManager::create(cwd, session_dir, new_session_options(&parsed.session_id))
		.unwrap_or_else(|error| fail(&error.to_string()))
}

/// Result of [`build_session_options`] (upstream return object).
pub struct BuiltSessionOptions {
	pub model: Option<pi_ai::types::Model>,
	pub thinking_level: Option<AgentThinkingLevel>,
	pub scoped_models: Vec<ScopedModelEntry>,
	pub tools: Option<Vec<String>>,
	pub exclude_tools: Option<Vec<String>>,
	pub no_tools: Option<NoToolsMode>,
	pub cli_thinking_from_model: bool,
	pub diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
}

/// `buildSessionOptions`.
pub fn build_session_options(
	parsed: &Args,
	scoped_models: Vec<ScopedModelEntry>,
	has_existing_session: bool,
	model_runtime: &ModelRuntime,
	settings_manager: &SettingsManager,
) -> BuiltSessionOptions {
	let mut options = BuiltSessionOptions {
		model: None,
		thinking_level: None,
		scoped_models: Vec::new(),
		tools: None,
		exclude_tools: None,
		no_tools: None,
		cli_thinking_from_model: false,
		diagnostics: Vec::new(),
	};

	// Model from CLI
	// - supports --provider <name> --model <pattern>
	// - supports --model <provider>/<pattern>
	if parsed.provider.is_some() && parsed.model.is_none() {
		options.diagnostics.push(AgentSessionRuntimeDiagnostic {
			diagnostic_type: RuntimeDiagnosticType::Error,
			message: format!(
				"--provider requires --model (for example: --provider {} --model <pattern>)",
				parsed.provider.clone().unwrap_or_default()
			),
		});
	}
	if let Some(cli_model) = &parsed.model {
		let resolved = resolve_cli_model(ResolveCliModelOptions {
			cli_provider: parsed.provider.as_deref(),
			cli_model: Some(cli_model),
			cli_thinking: parsed.thinking,
			model_runtime,
		});
		if let Some(warning) = &resolved.warning {
			options.diagnostics.push(AgentSessionRuntimeDiagnostic {
				diagnostic_type: RuntimeDiagnosticType::Warning,
				message: warning.clone(),
			});
		}
		if let Some(error) = &resolved.error {
			options.diagnostics.push(AgentSessionRuntimeDiagnostic {
				diagnostic_type: RuntimeDiagnosticType::Error,
				message: error.clone(),
			});
		}
		if let Some(model) = resolved.model {
			options.model = Some(model);
			// Allow "--model <pattern>:<thinking>" as a shorthand. Explicit
			// --thinking still takes precedence (applied later).
			if parsed.thinking.is_none() {
				if let Some(thinking_level) = resolved.thinking_level {
					options.thinking_level = Some(thinking_level);
					options.cli_thinking_from_model = true;
				}
			}
		}
	}

	if options.model.is_none() && !scoped_models.is_empty() && !has_existing_session {
		// Check if saved default is in scoped models - use it if so, otherwise
		// first scoped model
		let saved_provider = settings_manager.get_default_provider();
		let saved_model_id = settings_manager.get_default_model();
		let saved_model = match (&saved_provider, &saved_model_id) {
			(Some(provider), Some(model_id)) => ModelRuntime::get_model(model_runtime, provider, model_id),
			_ => None,
		};
		let saved_in_scope =
			saved_model.and_then(|saved| scoped_models.iter().find(|sm| sm.model == saved).cloned());

		if let Some(saved_in_scope) = saved_in_scope {
			options.model = Some(saved_in_scope.model);
			// Use thinking level from scoped model config if explicitly set
			if parsed.thinking.is_none() {
				options.thinking_level = saved_in_scope.thinking_level;
			}
		} else {
			let first = scoped_models[0].clone();
			options.model = Some(first.model);
			// Use thinking level from first scoped model if explicitly set
			if parsed.thinking.is_none() {
				options.thinking_level = first.thinking_level;
			}
		}
	}

	// Thinking level from CLI (takes precedence over scoped model thinking
	// levels set above)
	if let Some(thinking) = parsed.thinking {
		options.thinking_level = Some(thinking);
	}

	// Scoped models for Ctrl+P cycling
	if !scoped_models.is_empty() {
		options.scoped_models = scoped_models;
	}

	// API key from CLI - handled by the caller before session creation.

	// Tools
	if parsed.no_tools {
		options.no_tools = Some(NoToolsMode::All);
	} else if parsed.no_builtin_tools {
		options.no_tools = Some(NoToolsMode::Builtin);
	}
	if let Some(tools) = &parsed.tools {
		options.tools = Some(tools.clone());
	}
	if let Some(exclude_tools) = &parsed.exclude_tools {
		options.exclude_tools = Some(exclude_tools.clone());
	}

	options
}

fn resolve_cli_paths(cwd: &str, paths: Option<&Vec<String>>) -> Option<Vec<String>> {
	paths.map(|paths| {
		paths
			.iter()
			.map(|value| {
				if crate::utils::paths::is_local_path(value) {
					crate::utils::paths::resolve_path(value, cwd)
				} else {
					value.clone()
				}
			})
			.collect()
	})
}

fn collect_settings_diagnostics(settings_manager: &mut SettingsManager) -> Vec<AgentSessionRuntimeDiagnostic> {
	settings_manager
		.drain_errors()
		.into_iter()
		.map(|error| AgentSessionRuntimeDiagnostic {
			diagnostic_type: RuntimeDiagnosticType::Warning,
			message: error.error,
		})
		.collect()
}

fn deduplicate_diagnostics(diagnostics: Vec<AgentSessionRuntimeDiagnostic>) -> Vec<AgentSessionRuntimeDiagnostic> {
	let mut seen = std::collections::BTreeSet::new();
	let mut result = Vec::new();
	for diagnostic in diagnostics {
		let key = (diagnostic.diagnostic_type.as_str(), diagnostic.message.clone());
		if seen.insert(key) {
			result.push(diagnostic);
		}
	}
	result
}

fn print_mode_unavailable(mode: AppMode) -> ! {
	eprintln!(
		"Error: {} mode requires the modes module, which is not linked into this build (D102).",
		mode.as_str()
	);
	std::process::exit(1);
}

/// `MainOptions`.
#[derive(Default)]
pub struct MainOptions;

/// `main` — CLI orchestration minus interactive dispatch.
pub async fn main(args: Vec<String>, _options: Option<MainOptions>) {
	let offline_mode = args.iter().any(|arg| arg == "--offline")
		|| is_truthy_env_flag(std::env::var("PI_OFFLINE").ok().as_deref());
	if offline_mode {
		std::env::set_var("PI_OFFLINE", "1");
		std::env::set_var("PI_SKIP_VERSION_CHECK", "1");
	}

	if let Err(exit_code) = run_auth_command(&args).await {
		std::process::exit(exit_code);
	}

	let cwd = std::env::current_dir()
		.map(|dir| dir.to_string_lossy().to_string())
		.unwrap_or_else(|_| ".".to_string());
	let agent_dir = config::get_agent_dir();
	let _bootstrap_settings_manager = SettingsManager::create(&cwd, &agent_dir, SettingsManagerCreateOptions::default());

	// Upstream `install/remove/update/list/config/mcp` subcommands live in
	// deferred modules (D104).
	if let Some(first) = args.first() {
		if matches!(first.as_str(), "install" | "remove" | "uninstall" | "update" | "list" | "config" | "mcp") {
			eprintln!("Error: \"{first}\" is not available in this build (package/mcp CLI modules are deferred).");
			std::process::exit(1);
		}
	}

	let mut parsed = parse_args(&args);
	if !parsed.diagnostics.is_empty() {
		for diagnostic in &parsed.diagnostics {
			match diagnostic.diagnostic_type {
				DiagnosticType::Error => eprintln!("Error: {}", diagnostic.message),
				DiagnosticType::Warning => eprintln!("Warning: {}", diagnostic.message),
			}
		}
		if parsed.diagnostics.iter().any(|d| d.diagnostic_type == DiagnosticType::Error) {
			std::process::exit(1);
		}
	}

	if parsed.version {
		println!("{}", config::VERSION);
		std::process::exit(0);
	}

	if parsed.export.is_some() {
		// HTML export is deferred (D79).
		eprintln!("Error: --export requires the HTML export module, which is deferred (D79).");
		std::process::exit(1);
	}

	let app_mode = resolve_app_mode(&parsed, !use_stdin_pipe(), stdout_is_tty());

	if parsed.mode == Some(Mode::Rpc) && !parsed.file_args.is_empty() {
		eprintln!("Error: @file arguments are not supported in RPC mode");
		std::process::exit(1);
	}

	validate_fork_flags(&parsed);
	validate_session_id_flags(&parsed);

	// Run migrations (pass cwd for project-local migrations)
	let migration_result = run_migrations(&cwd);
	let deprecation_warnings = migration_result.deprecation_warnings;

	let mut startup_settings_manager =
		SettingsManager::create(&cwd, &agent_dir, SettingsManagerCreateOptions::default());
	let startup_settings_diagnostics = collect_settings_diagnostics(&mut startup_settings_manager);

	if app_mode == AppMode::Interactive {
		// Interactive mode is excluded; the startup selector, first-time
		// setup, and theme overrides are TUI surfaces.
		eprintln!(
			"Error: interactive mode is not available in this build (the TUI is excluded; use --print, --mode json, or --mode rpc)."
		);
		std::process::exit(1);
	}

	// Decide the final runtime cwd before creating cwd-bound runtime services.
	let env_session_dir = std::env::var(config::ENV_SESSION_DIR).ok();
	let session_dir = parsed
		.session_dir
		.as_ref()
		.map(|dir| crate::utils::paths::normalize_path(dir, &crate::utils::paths::PathInputOptions::default()))
		.or_else(|| env_session_dir.as_ref().map(|dir| config::expand_tilde_path(dir)))
		.or_else(|| startup_settings_manager.get_session_dir());
	let session_dir_ref = session_dir.as_deref();
	let mut session_manager = create_session_manager(&parsed, &cwd, session_dir_ref, &startup_settings_manager).await;
	let missing_session_cwd_issue = get_missing_session_cwd_issue(&session_manager, &cwd);
	if let Some(issue) = missing_session_cwd_issue {
		// Interactive prompting is excluded; non-interactive modes report the
		// upstream `MissingSessionCwdError` text.
		eprintln!("Error: {}", MissingSessionCwdError::new(issue));
		std::process::exit(1);
	}
	if let Some(name) = &parsed.name {
		let Some(name) = normalize_session_name(name) else {
			eprintln!("Error: --name requires a non-empty value");
			std::process::exit(1);
		};
		let _ = session_manager.append_session_info(&name);
	}

	// Project trust: `--approve`/`--no-approve` override; the trust store and
	// trust-requiring resource detection are deferred (D103), so project-local
	// files are trusted unless overridden.
	let project_trusted = parsed.project_trust_override.unwrap_or(true);
	let resolved_extension_paths = resolve_cli_paths(&cwd, parsed.extensions.as_ref());
	let resolved_skill_paths = resolve_cli_paths(&cwd, parsed.skills.as_ref());
	let resolved_prompt_template_paths = resolve_cli_paths(&cwd, parsed.prompt_templates.as_ref());
	let resolved_theme_paths = resolve_cli_paths(&cwd, parsed.themes.as_ref());

	let session_manager_cwd = session_manager.get_cwd().to_string();
	let session_manager = Arc::new(Mutex::new(session_manager));
	let parsed_for_factory = parsed.clone();
	let _ = &mut parsed; // messages are consumed after runtime creation, like upstream

	let create_runtime: CreateAgentSessionRuntimeFactory = Arc::new(
		move |options: CreateAgentSessionRuntimeOptions| {
			let parsed = parsed_for_factory.clone();
			let resolved_extension_paths = resolved_extension_paths.clone();
			let resolved_skill_paths = resolved_skill_paths.clone();
			let resolved_prompt_template_paths = resolved_prompt_template_paths.clone();
			let resolved_theme_paths = resolved_theme_paths.clone();
			Box::pin(async move {
				let runtime_settings_manager = SettingsManager::create(
					&options.cwd,
					&options.agent_dir,
					SettingsManagerCreateOptions { project_trusted: Some(project_trusted) },
				);
				let unknown_flags: HashMap<String, crate::core::extensions::types::FlagValue> = parsed
					.unknown_flags
					.iter()
					.map(|(name, value)| {
						(
							name.clone(),
							match value {
								UnknownFlagValue::Bool => crate::core::extensions::types::FlagValue::Bool(true),
								UnknownFlagValue::Str(value) => {
									crate::core::extensions::types::FlagValue::Str(value.clone())
								}
							},
						)
					})
					.collect();
				let services = create_agent_session_services(CreateAgentSessionServicesOptions {
					cwd: options.cwd.clone(),
					agent_dir: Some(options.agent_dir.clone()),
					settings_manager: Some(runtime_settings_manager),
					model_runtime: None,
					model_runtime_signal: None,
					extension_flag_values: Some(unknown_flags),
					resource_loader_options: ResourceLoaderOptionsOverride {
						additional_extension_paths: resolved_extension_paths.unwrap_or_default(),
						additional_skill_paths: resolved_skill_paths.unwrap_or_default(),
						additional_prompt_template_paths: resolved_prompt_template_paths.unwrap_or_default(),
						additional_theme_paths: resolved_theme_paths.unwrap_or_default(),
						no_extensions: parsed.no_extensions,
						no_skills: parsed.no_skills,
						no_prompt_templates: parsed.no_prompt_templates,
						no_themes: parsed.no_themes,
						no_context_files: parsed.no_context_files,
						system_prompt: parsed.system_prompt.clone(),
						append_system_prompt: parsed.append_system_prompt.clone(),
						..Default::default()
					},
					resource_loader_reload_options: ResourceLoaderReloadOptions::default(),
				})
				.await;

				let model_runtime = services.model_runtime.clone();
				let settings_manager = services.settings_manager.clone();
				let mut diagnostics: Vec<AgentSessionRuntimeDiagnostic> = services.diagnostics.clone();
				diagnostics.extend(collect_settings_diagnostics(&mut settings_manager.lock()));
				let extensions_result = {
					let loader = services.resource_loader.lock();
					(loader.get_extensions().errors.clone(), loader.get_extensions().warnings.clone())
				};
				for error in &extensions_result.0 {
					diagnostics.push(AgentSessionRuntimeDiagnostic {
						diagnostic_type: RuntimeDiagnosticType::Error,
						message: format!("Failed to load extension \"{}\": {}", error.path, error.error),
					});
				}
				for warning in extensions_result.1.iter().flatten() {
					diagnostics.push(AgentSessionRuntimeDiagnostic {
						diagnostic_type: RuntimeDiagnosticType::Warning,
						message: format!("Extension package \"{}\": {}", warning.path, warning.error),
					});
				}

				let model_patterns =
					parsed.models.clone().or_else(|| settings_manager.lock().get_enabled_models());
				let scoped_models: Vec<ScopedModelEntry> = match &model_patterns {
					Some(patterns) if !patterns.is_empty() => {
						resolve_model_scope_with_diagnostics(patterns, model_runtime.as_ref(), None)
							.await
							.map(|result| {
								result
									.scoped_models
									.into_iter()
									.map(|scoped| ScopedModelEntry {
										model: scoped.model,
										thinking_level: scoped.thinking_level,
									})
									.collect()
							})
							.unwrap_or_default()
					}
					_ => Vec::new(),
				};
				let has_existing_session = !options.session_manager.lock().build_session_context().messages.is_empty();
				let mut session_options = build_session_options(
					&parsed,
					scoped_models,
					has_existing_session,
					&model_runtime,
					&settings_manager.lock(),
				);
				diagnostics.append(&mut session_options.diagnostics);

				if let Some(api_key) = &parsed.api_key {
					match &session_options.model {
						None => diagnostics.push(AgentSessionRuntimeDiagnostic {
							diagnostic_type: RuntimeDiagnosticType::Error,
							message:
								"--api-key requires a model to be specified via --model, --provider/--model, or --models"
									.to_string(),
						}),
						Some(model) => {
							let _ = model_runtime
								.set_runtime_api_key(&model.base.provider, api_key, None)
								.await;
						}
					}
				}

				let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
					services: services.clone(),
					session_manager: options.session_manager.clone(),
					session_start_event: options.session_start_event,
					model: session_options.model,
					thinking_level: session_options.thinking_level,
					scoped_models: session_options.scoped_models,
					tools: session_options.tools,
					exclude_tools: session_options.exclude_tools,
					no_tools: session_options.no_tools,
					custom_tools: Vec::new(),
				})
				.await?;

				Ok(CreateAgentSessionRuntimeResult {
					session: created.session,
					services,
					diagnostics,
					model_fallback_message: created.model_fallback_message,
				})
			})
		},
	);

	let runtime = create_agent_session_runtime(
		create_runtime,
		CreateAgentSessionRuntimeOptions {
			cwd: session_manager_cwd,
			agent_dir,
			session_manager,
			session_start_event: None,
			project_trust_context: None,
		},
	)
	.await
	.unwrap_or_else(|error| fail(&error));

	if parsed.help {
		report_diagnostics(&startup_settings_diagnostics);
		let extension_flags: Vec<ExtensionFlagInfo> = runtime
			.services()
			.resource_loader
			.lock()
			.get_extensions()
			.extensions
			.iter()
			.flat_map(|extension| {
				extension
					.flags
					.iter()
					.map(|(name, flag)| {
						ExtensionFlagInfo::from_extension_flag(name, flag, &extension.path)
					})
					.collect::<Vec<_>>()
			})
			.collect();
		print_help(&extension_flags);
		std::process::exit(0);
	}

	if let Some(list_models_value) = &parsed.list_models {
		report_diagnostics(&startup_settings_diagnostics);
		list_models(&runtime.services().model_runtime, list_models_value.as_deref(), None).await;
		std::process::exit(0);
	}

	// Read piped stdin content (if any) - skip for RPC mode which uses stdin
	// for JSON-RPC
	let stdin_content = if app_mode != AppMode::Rpc { read_piped_stdin().await } else { None };

	let (initial_message, initial_images) = prepare_initial_message(&mut parsed, stdin_content).await;

	// Show deprecation warnings (non-interactive variant prints and returns).
	if !deprecation_warnings.is_empty() {
		migrations::show_deprecation_warnings(&deprecation_warnings);
	}

	let startup_diagnostics = deduplicate_diagnostics(
		startup_settings_diagnostics
			.iter()
			.chain(runtime.diagnostics().iter())
			.cloned()
			.collect(),
	);
	report_diagnostics(&startup_diagnostics);
	let has_runtime_errors = runtime
		.diagnostics()
		.iter()
		.any(|diagnostic| diagnostic.diagnostic_type == RuntimeDiagnosticType::Error);
	if has_runtime_errors {
		if runtime
			.diagnostics()
			.iter()
			.any(|diagnostic| diagnostic.message.contains("Failed to load extension"))
		{
			eprintln!("{EXTENSION_LOAD_FAILURE_HINT}");
		}
		std::process::exit(1);
	}

	if runtime.session().model().is_none() {
		eprintln!("Error: {}", crate::core::auth_guidance::format_no_models_available_message());
		std::process::exit(1);
	}

	match app_mode {
		AppMode::Rpc => match mode_dispatcher() {
			Some(dispatcher) => dispatcher.run_rpc_mode(runtime),
			None => print_mode_unavailable(AppMode::Rpc),
		},
		AppMode::Interactive => {
			// Excluded (see module docs).
			eprintln!("Error: interactive mode is not available in this build.");
			std::process::exit(1);
		}
		AppMode::Print | AppMode::Json => {
			let options = PrintModeOptions {
				json: to_print_output_mode(app_mode),
				messages: parsed.messages,
				initial_message,
				initial_images,
			};
			let exit_code = mode_dispatcher()
				.map(|dispatcher| dispatcher.run_print_mode(runtime, options))
				.unwrap_or_else(|| print_mode_unavailable(app_mode));
			if exit_code != 0 {
				std::process::exit(exit_code);
			}
		}
	}
}
