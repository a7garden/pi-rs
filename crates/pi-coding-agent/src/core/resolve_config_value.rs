//! Port of `packages/coding-agent/src/core/resolve-config-value.ts` —
//! resolution of configuration values that may be shell commands
//! (`!cmd`), environment-variable templates (`$VAR`/`${VAR}`), or literals.
//! Used by `auth-storage.ts` and the model layer (`provider-composer.ts`,
//! `model-registry.ts`).
//!
//! # Deviations
//!
//! - **Sync child processes.** Node `execSync`/`spawnSync` become
//!   [`std::process::Command`] with a 50 ms `try_wait` poll up to the
//!   upstream 10 s timeout (then kill). The default-shell path runs `sh -c`
//!   (POSIX) / `cmd /C` (Windows), matching Node's `execSync` shells.
//! - **Platform seam.** The Windows-only configured-shell branch is gated on
//!   `cfg!(windows)`; the stdin-transport path is exercised directly in tests
//!   through a shell-config override hook (upstream mocks `getShellConfig`
//!   and swaps `process.platform`).
//! - **`||` falsiness.** `resolveEnvConfigValue`'s `env?.[name] ||
//!   process.env[name]` treats empty strings as unset; the port replicates
//!   that (an empty value falls through to the process environment).

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use pi_ai::types::ProviderEnv;

use crate::utils::shell::{get_shell_config, CommandTransport, ShellConfig};

/// Credential-scoped environment overrides (`Record<string, string> |
/// undefined` upstream). Shared alias: [`pi_ai::types::ProviderEnv`].

/// Cache for shell command results (persists for process lifetime).
static COMMAND_RESULT_CACHE: LazyLock<Mutex<HashMap<String, Option<String>>>> = LazyLock::new(Mutex::default);

/// Test seam: when set, replaces [`get_shell_config`] (upstream tests mock the
/// module function; crate docs deviation for this module).
static SHELL_CONFIG_OVERRIDE: RwLock<Option<ShellConfig>> = RwLock::new(None);

/// Test-only seam replacing the upstream `getShellConfig` module mock.
pub fn set_shell_config_override(config: Option<ShellConfig>) {
	*SHELL_CONFIG_OVERRIDE.write() = config;
}

fn configured_shell() -> Option<ShellConfig> {
	if let Some(config) = SHELL_CONFIG_OVERRIDE.read().clone() {
		return Some(config);
	}
	get_shell_config(None).ok()
}

/// `^[A-Za-z_][A-Za-z0-9_]*$`.
fn is_env_var_name(name: &str) -> bool {
	let mut chars = name.chars();
	match chars.next() {
		Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
		_ => return false,
	}
	chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Longest `^[A-Za-z_][A-Za-z0-9_]*` prefix length (may be 0).
fn env_var_name_prefix_len(value: &str) -> usize {
	let mut len = 0;
	for c in value.chars() {
		if c.is_ascii_alphanumeric() || c == '_' {
			len += c.len_utf8();
		} else {
			break;
		}
	}
	len
}

#[derive(Debug, Clone, PartialEq)]
enum TemplatePart {
	Literal(String),
	Env { name: String },
}

#[derive(Debug, Clone, PartialEq)]
enum ConfigValueReference {
	Command { config: String },
	Template { parts: Vec<TemplatePart> },
}

fn append_literal(parts: &mut Vec<TemplatePart>, value: &str) {
	if value.is_empty() {
		return;
	}
	if let Some(TemplatePart::Literal(previous)) = parts.last_mut() {
		previous.push_str(value);
		return;
	}
	parts.push(TemplatePart::Literal(value.to_string()));
}

fn parse_config_value_template(config: &str) -> Vec<TemplatePart> {
	let mut parts: Vec<TemplatePart> = Vec::new();
	let bytes = config.as_bytes();
	let mut index = 0;

	while index < config.len() {
		let dollar_index = match config[index..].find('$') {
			Some(offset) => index + offset,
			None => {
				append_literal(&mut parts, &config[index..]);
				break;
			}
		};

		append_literal(&mut parts, &config[index..dollar_index]);
		let next_char = bytes.get(dollar_index + 1).copied();

		if next_char == Some(b'$') || next_char == Some(b'!') {
			append_literal(&mut parts, &(next_char.unwrap() as char).to_string());
			index = dollar_index + 2;
			continue;
		}

		if next_char == Some(b'{') {
			let end_index = config[dollar_index + 2..].find('}').map(|offset| dollar_index + 2 + offset);
			let Some(end_index) = end_index else {
				append_literal(&mut parts, "$");
				index = dollar_index + 1;
				continue;
			};

			let name = &config[dollar_index + 2..end_index];
			if is_env_var_name(name) {
				parts.push(TemplatePart::Env { name: name.to_string() });
			} else {
				append_literal(&mut parts, &config[dollar_index..end_index + 1]);
			}
			index = end_index + 1;
			continue;
		}

		let prefix_len = env_var_name_prefix_len(&config[dollar_index + 1..]);
		if prefix_len > 0 {
			parts.push(TemplatePart::Env {
				name: config[dollar_index + 1..dollar_index + 1 + prefix_len].to_string(),
			});
			index = dollar_index + 1 + prefix_len;
			continue;
		}

		append_literal(&mut parts, "$");
		index = dollar_index + 1;
	}

	parts
}

fn parse_config_value_reference(config: &str) -> ConfigValueReference {
	if config.starts_with('!') {
		return ConfigValueReference::Command { config: config.to_string() };
	}
	ConfigValueReference::Template { parts: parse_config_value_template(config) }
}

/// `env?.[name] || process.env[name] || undefined` — empty strings fall
/// through, matching JS `||` falsiness.
fn resolve_env_config_value(name: &str, env: Option<&ProviderEnv>) -> Option<String> {
	env.and_then(|env| env.get(name))
		.filter(|value| !value.is_empty())
		.cloned()
		.or_else(|| std::env::var(name).ok().filter(|value| !value.is_empty()))
}

fn get_template_env_var_names(parts: &[TemplatePart]) -> Vec<String> {
	let mut names: Vec<String> = Vec::new();
	for part in parts {
		let TemplatePart::Env { name } = part else { continue };
		if names.contains(name) {
			continue;
		}
		names.push(name.clone());
	}
	names
}

fn resolve_template(parts: &[TemplatePart], env: Option<&ProviderEnv>) -> Option<String> {
	let mut resolved = String::new();
	for part in parts {
		match part {
			TemplatePart::Literal(value) => resolved.push_str(value),
			TemplatePart::Env { name } => {
				let env_value = resolve_env_config_value(name, env)?;
				resolved.push_str(&env_value);
			}
		}
	}
	Some(resolved)
}

/// The single env-var name a pure `$VAR`/`${VAR}` reference resolves to
/// (`getConfigValueEnvVarName`).
pub fn get_config_value_env_var_name(config: &str) -> Option<String> {
	match parse_config_value_reference(config) {
		ConfigValueReference::Command { .. } => None,
		ConfigValueReference::Template { parts } => match parts.as_slice() {
			[TemplatePart::Env { name }] => Some(name.clone()),
			_ => None,
		},
	}
}

/// Every env-var name referenced by a template value
/// (`getConfigValueEnvVarNames`); empty for commands.
pub fn get_config_value_env_var_names(config: &str) -> Vec<String> {
	match parse_config_value_reference(config) {
		ConfigValueReference::Command { .. } => Vec::new(),
		ConfigValueReference::Template { parts } => get_template_env_var_names(&parts),
	}
}

/// Referenced env-var names that resolve to nothing in `env` or the process
/// environment (`getMissingConfigValueEnvVarNames`).
pub fn get_missing_config_value_env_var_names(config: &str, env: Option<&ProviderEnv>) -> Vec<String> {
	get_config_value_env_var_names(config)
		.into_iter()
		.filter(|name| resolve_env_config_value(name, env).is_none())
		.collect()
}

/// Whether the value is a `!command` reference (`isCommandConfigValue`).
pub fn is_command_config_value(config: &str) -> bool {
	matches!(parse_config_value_reference(config), ConfigValueReference::Command { .. })
}

/// Whether every referenced env var resolves (`isConfigValueConfigured`).
pub fn is_config_value_configured(config: &str, env: Option<&ProviderEnv>) -> bool {
	get_missing_config_value_env_var_names(config, env).is_empty()
}

/// Run the command through the user's configured shell
/// (`executeWithConfiguredShell`): `(executed, value)`.
fn execute_with_configured_shell(command: &str) -> (bool, Option<String>) {
	let Some(shell_config) = configured_shell() else {
		return (false, None);
	};
	let command_from_stdin = matches!(shell_config.command_transport, Some(CommandTransport::Stdin));

	let mut cmd = Command::new(&shell_config.shell);
	cmd.args(&shell_config.args);
	if !command_from_stdin {
		cmd.arg(command);
	}
	cmd.stdin(if command_from_stdin { Stdio::piped() } else { Stdio::null() })
		.stdout(Stdio::piped())
		.stderr(Stdio::null());

	let mut child = match cmd.spawn() {
		Ok(child) => child,
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => return (false, None),
		Err(_) => return (true, None),
	};

	if command_from_stdin {
		if let Some(mut stdin) = child.stdin.take() {
			let _ = stdin.write_all(command.as_bytes());
		}
	}

	let stdout = drain_after_exit(&mut child);
	match stdout {
		Some(value) => (true, (!value.is_empty()).then_some(value)),
		None => (true, None),
	}
}

/// Drain the child's piped stdout on a helper thread while waiting for exit
/// with the upstream 10 s timeout; `None` when the command fails or is killed
/// on timeout (the child is killed first, then the drain EOFs).
fn drain_after_exit(child: &mut std::process::Child) -> Option<String> {
	let stdout_pipe = child.stdout.take();
	let reader = std::thread::spawn(move || {
		let mut buffer = String::new();
		if let Some(mut pipe) = stdout_pipe {
			use std::io::Read;
			let _ = pipe.read_to_string(&mut buffer);
		}
		buffer
	});

	let status = wait_with_timeout(child, Duration::from_secs(10));
	let stdout = reader.join().unwrap_or_default();
	match status {
		Some(status) if status.success() => {
			let value = stdout.trim().to_string();
			(!value.is_empty()).then_some(value)
		}
		_ => None,
	}
}

/// Poll `try_wait` until exit or timeout; `None` on timeout (child killed).
fn wait_with_timeout(child: &mut std::process::Child, timeout: Duration) -> Option<std::process::ExitStatus> {
	let deadline = Instant::now() + timeout;
	loop {
		match child.try_wait() {
			Ok(Some(status)) => return Some(status),
			Ok(None) if Instant::now() < deadline => {
				std::thread::sleep(Duration::from_millis(50));
			}
			Ok(None) => {
				let _ = child.kill();
				let _ = child.wait();
				return None;
			}
			Err(_) => return None,
		}
	}
}

/// Node `execSync` default shell: `/bin/sh -c` (POSIX) / `cmd /C` (Windows).
fn execute_with_default_shell(command: &str) -> Option<String> {
	let mut cmd = if cfg!(windows) {
		let mut cmd = Command::new("cmd");
		cmd.args(["/C", command]);
		cmd
	} else {
		let mut cmd = Command::new("sh");
		cmd.args(["-c", command]);
		cmd
	};
	cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());

	let mut child = cmd.spawn().ok()?;
	drain_after_exit(&mut child)
}

fn execute_command_uncached(command_config: &str) -> Option<String> {
	let command = &command_config[1..];
	if cfg!(windows) {
		let (executed, value) = execute_with_configured_shell(command);
		if executed {
			value
		} else {
			execute_with_default_shell(command)
		}
	} else {
		execute_with_default_shell(command)
	}
}

fn execute_command(command_config: &str) -> Option<String> {
	if let Some(cached) = COMMAND_RESULT_CACHE.lock().get(command_config) {
		return cached.clone();
	}
	let result = execute_command_uncached(command_config);
	COMMAND_RESULT_CACHE.lock().insert(command_config.to_string(), result.clone());
	result
}

/// Resolve a config value (API key, header value, etc.) to an actual value
/// (`resolveConfigValue`):
/// - `!cmd` executes the rest as a shell command and uses trimmed stdout
///   (cached for the process lifetime)
/// - `$ENV_VAR`/`${ENV_VAR}` interpolate the named environment variable
///   (credential-scoped `env` first, then the process environment)
/// - in non-command values, `$$` escapes a literal `$` and `$!` a literal `!`
/// - otherwise the value is a literal
pub fn resolve_config_value(config: &str, env: Option<&ProviderEnv>) -> Option<String> {
	match parse_config_value_reference(config) {
		ConfigValueReference::Command { config } => execute_command(&config),
		ConfigValueReference::Template { parts } => resolve_template(&parts, env),
	}
}

/// Like [`resolve_config_value`] but never uses the command cache
/// (`resolveConfigValueUncached`).
pub fn resolve_config_value_uncached(config: &str, env: Option<&ProviderEnv>) -> Option<String> {
	match parse_config_value_reference(config) {
		ConfigValueReference::Command { config } => execute_command_uncached(&config),
		ConfigValueReference::Template { parts } => resolve_template(&parts, env),
	}
}

/// Like [`resolve_config_value_uncached`] but fails with the upstream error
/// strings when the value cannot be resolved (`resolveConfigValueOrThrow`).
pub fn resolve_config_value_or_throw(
	config: &str,
	description: &str,
	env: Option<&ProviderEnv>,
) -> Result<String, String> {
	if let Some(resolved_value) = resolve_config_value_uncached(config, env) {
		return Ok(resolved_value);
	}

	match parse_config_value_reference(config) {
		ConfigValueReference::Command { config } => {
			return Err(format!(
				"Failed to resolve {description} from shell command: {}",
				&config[1..]
			));
		}
		ConfigValueReference::Template { .. } => {
			let missing_env_vars = get_missing_config_value_env_var_names(config, env);
			if missing_env_vars.len() == 1 {
				return Err(format!(
					"Failed to resolve {description} from environment variable: {}",
					missing_env_vars[0]
				));
			}
			if missing_env_vars.len() > 1 {
				return Err(format!(
					"Failed to resolve {description} from environment variables: {}",
					missing_env_vars.join(", ")
				));
			}
		}
	}

	Err(format!("Failed to resolve {description}"))
}

/// Resolve all header values using the same resolution logic as API keys
/// (`resolveHeaders`). Entries resolving to nothing are dropped; an empty
/// result reads as absent.
pub fn resolve_headers(
	headers: Option<&BTreeMap<String, String>>,
	env: Option<&ProviderEnv>,
) -> Option<BTreeMap<String, String>> {
	let headers = headers?;
	let mut resolved = BTreeMap::new();
	for (key, value) in headers {
		if let Some(resolved_value) = resolve_config_value(value, env) {
			resolved.insert(key.clone(), resolved_value);
		}
	}
	(!resolved.is_empty()).then_some(resolved)
}

/// [`resolve_headers`] with upstream throw semantics per entry
/// (`resolveHeadersOrThrow`).
pub fn resolve_headers_or_throw(
	headers: Option<&BTreeMap<String, String>>,
	description: &str,
	env: Option<&ProviderEnv>,
) -> Result<Option<BTreeMap<String, String>>, String> {
	let Some(headers) = headers else {
		return Ok(None);
	};
	let mut resolved = BTreeMap::new();
	for (key, value) in headers {
		resolved.insert(key.clone(), resolve_config_value_or_throw(value, &format!("{description} header \"{key}\""), env)?);
	}
	Ok((!resolved.is_empty()).then_some(resolved))
}

/// Clear the config value command cache (`clearConfigValueCache`).
pub fn clear_config_value_cache() {
	COMMAND_RESULT_CACHE.lock().clear();
}

#[cfg(test)]
#[path = "resolve_config_value_tests.rs"]
mod resolve_config_value_tests;
