//! `cli/auth-command.ts` — `auth` subcommand parsing.

use super::args::Args;
use crate::config::APP_NAME;

/// `AuthCommandKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthCommandKind {
	Check,
	ApiKey,
	BearerToken,
}

/// Parsed `auth` command (`AuthCommand`).
#[derive(Debug, Clone)]
pub struct AuthCommand {
	pub kind: AuthCommandKind,
	pub args: Vec<String>,
	pub json: bool,
	pub credentials: bool,
	pub no_refresh: bool,
	pub min_expiry_ms: Option<u64>,
}

/// `AuthCommandError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthCommandError(pub String);

impl std::fmt::Display for AuthCommandError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.0)
	}
}

impl std::error::Error for AuthCommandError {}

fn auth_command_usage(kind: AuthCommandKind) -> String {
	match kind {
		AuthCommandKind::Check => {
			format!("{APP_NAME} auth check --provider <provider> [--json] [--credentials] [--no-refresh]")
		}
		AuthCommandKind::ApiKey => format!("{APP_NAME} auth print-api-key --provider <provider> [--model <model>]"),
		AuthCommandKind::BearerToken => format!(
			"{APP_NAME} auth print-bearer-token --provider <provider> [--model <model>] [--min-expiry <duration>]"
		),
	}
}

pub fn get_auth_command_name(kind: AuthCommandKind) -> &'static str {
	match kind {
		AuthCommandKind::Check => "auth check",
		AuthCommandKind::ApiKey => "auth print-api-key",
		AuthCommandKind::BearerToken => "auth print-bearer-token",
	}
}

pub fn get_auth_command_usage(kind: AuthCommandKind) -> String {
	auth_command_usage(kind)
}

pub fn is_auth_command_help(args: &[String]) -> bool {
	args.first().map(|s| s.as_str()) == Some("auth")
		&& (args.get(1).is_none()
			|| args.get(1).map(|s| s.as_str()) == Some("help")
			|| args.iter().any(|a| a == "--help" || a == "-h"))
}

pub fn print_auth_command_help() {
	println!(
		"Usage:
  pi auth print-api-key [--provider <provider>] [--model <model>]
  pi auth print-bearer-token [--provider <provider>] [--model <model>] [--min-expiry <duration>]
  pi auth check [--provider <provider>] [--model <model>] [--json] [--credentials] [--no-refresh]

Auth commands require at least one of --provider or --model. Checks refresh expired OAuth credentials by default; --no-refresh prevents this. --credentials emits the credential, or includes it in JSON output."
	);
}

/// Parse an `auth` subcommand (`parseAuthCommand`). Returns `None` when the
/// first argument is not `auth`.
pub fn parse_auth_command(args: &[String]) -> Result<Option<AuthCommand>, AuthCommandError> {
	if args.first().map(|s| s.as_str()) != Some("auth") {
		return Ok(None);
	}

	let kind = match args.get(1).map(|s| s.as_str()) {
		Some("check") => AuthCommandKind::Check,
		Some("print-api-key") => AuthCommandKind::ApiKey,
		Some("print-bearer-token") => AuthCommandKind::BearerToken,
		other => {
			return Err(AuthCommandError(format!(
				"Unknown auth command \"{}\". Use \"{APP_NAME} auth print-api-key\", \"{APP_NAME} auth print-bearer-token\", or \"{APP_NAME} auth check\".",
				other.unwrap_or("")
			)));
		}
	};

	let mut command_args: Vec<String> = Vec::new();
	let mut json = false;
	let mut credentials = false;
	let mut no_refresh = false;
	let mut min_expiry_ms: Option<u64> = None;

	let mut index = 2usize;
	while index < args.len() {
		let arg = args[index].as_str();
		if arg == "--min-expiry" {
			if kind != AuthCommandKind::BearerToken {
				return Err(AuthCommandError("--min-expiry is only supported by print-bearer-token".to_string()));
			}
			index += 1;
			let value = args.get(index).map(|s| s.as_str());
			let parsed = parse_min_expiry(value);
			let (amount, unit_ms) = match parsed {
				Some(parsed) => parsed,
				None => {
					return Err(AuthCommandError("--min-expiry must use a duration such as 30m or 1h".to_string()));
				}
			};
			min_expiry_ms = Some(amount * unit_ms);
			index += 1;
			continue;
		}
		if arg == "--json" || arg == "--credentials" || arg == "--no-refresh" {
			if kind != AuthCommandKind::Check {
				return Err(AuthCommandError(format!("{arg} is only supported by auth check")));
			}
			if arg == "--json" {
				json = true;
			} else if arg == "--credentials" {
				credentials = true;
			} else {
				no_refresh = true;
			}
			index += 1;
			continue;
		}
		command_args.push(arg.to_string());
		index += 1;
	}

	Ok(Some(AuthCommand { kind, args: command_args, json, credentials, no_refresh, min_expiry_ms }))
}

/// Parse `<amount><ms|s|m|h>` into `(amount, unit milliseconds)`.
fn parse_min_expiry(value: Option<&str>) -> Option<(u64, u64)> {
	let value = value?;
	let digits_end = value.find(|c: char| !c.is_ascii_digit())?;
	if digits_end == 0 {
		return None;
	}
	let (amount, unit) = value.split_at(digits_end);
	let amount: u64 = amount.parse().ok()?;
	let unit_ms = match unit {
		"ms" => 1,
		"s" => 1_000,
		"m" => 60_000,
		"h" => 3_600_000,
		_ => return None,
	};
	Some((amount, unit_ms))
}

/// Validate the parsed `Args` for an auth command (`validateAuthCommandArgs`).
pub fn validate_auth_command_args(
	args: &Args,
	kind: AuthCommandKind,
) -> Result<(Option<String>, Option<String>), AuthCommandError> {
	let provider = args.provider.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
	let model = args.model.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
	if let Some((option, _)) = args.unknown_flags.iter().next() {
		return Err(AuthCommandError(format!(
			"Unknown option --{option} for \"{}\".",
			get_auth_command_name(kind)
		)));
	}
	if args.api_key.is_some() || !args.messages.is_empty() || !args.file_args.is_empty() {
		return Err(AuthCommandError("Auth commands only accept --provider and --model".to_string()));
	}
	if kind == AuthCommandKind::Check {
		if provider.is_none() && model.is_none() {
			return Err(AuthCommandError(
				"Auth checks require --provider <provider> or --model <model>".to_string(),
			));
		}
		return Ok((provider, model));
	}
	if provider.is_none() && model.is_none() {
		return Err(AuthCommandError(
			"Credential printing requires --provider <provider> or --model <model>".to_string(),
		));
	}
	Ok((provider, model))
}

/// Extract the printable credential from an auth result
/// (`getAuthCredential`): the API key, else the bearer token in the
/// `authorization` header.
pub fn get_auth_credential(auth: Option<&pi_ai::auth::types::AuthResult>) -> Option<String> {
	let auth = auth?;
	if let Some(api_key) = &auth.auth.api_key {
		return Some(api_key.clone());
	}
	if let Some(headers) = &auth.auth.headers {
		for (name, value) in headers {
			if name.eq_ignore_ascii_case("authorization") {
				if let Some(value) = value {
					if let Some(token) = bearer_token(value) {
						return Some(token);
					}
				}
			}
		}
	}
	None
}

fn bearer_token(header: &str) -> Option<String> {
	let mut parts = header.splitn(2, ' ');
	let scheme = parts.next()?;
	let rest = parts.next()?;
	if scheme.eq_ignore_ascii_case("bearer") && !rest.trim().is_empty() {
		Some(rest.trim().to_string())
	} else {
		None
	}
}

#[cfg(test)]
mod tests {
	//! Parse/validate cases from `test/auth-check.test.ts` and
	//! `test/credential-print.test.ts` (the runtime-backed cases live with
	//! [`super::super::auth_check`]).
	use super::*;

	fn args(list: &[&str]) -> Vec<String> {
		list.iter().map(|s| s.to_string()).collect()
	}

	#[test]
	fn accepts_optional_json_credentials_and_no_refresh() {
		let command = parse_auth_command(&args(&["auth", "check", "--provider", "openai"]))
			.unwrap()
			.unwrap();
		assert!(matches!(command.kind, AuthCommandKind::Check));
		assert_eq!(command.args, vec!["--provider".to_string(), "openai".to_string()]);
		assert!(!command.json && !command.credentials && !command.no_refresh);
		assert_eq!(command.min_expiry_ms, None);

		let command = parse_auth_command(&args(&[
			"auth",
			"check",
			"--json",
			"--credentials",
			"--no-refresh",
			"--provider",
			"openai",
		]))
		.unwrap()
		.unwrap();
		assert!(command.json && command.credentials && command.no_refresh);
	}

	#[test]
	fn parses_credential_commands() {
		let command = parse_auth_command(&args(&["auth", "print-api-key", "--provider", "openai"]))
			.unwrap()
			.unwrap();
		assert!(matches!(command.kind, AuthCommandKind::ApiKey));

		let command = parse_auth_command(&args(&["auth", "print-bearer-token"])).unwrap().unwrap();
		assert!(matches!(command.kind, AuthCommandKind::BearerToken));

		let command =
			parse_auth_command(&args(&["auth", "print-bearer-token", "--min-expiry", "30m"]))
				.unwrap()
				.unwrap();
		assert!(matches!(command.kind, AuthCommandKind::BearerToken));
		assert_eq!(command.min_expiry_ms, Some(30 * 60_000));
		assert!(command.args.is_empty());
	}

	#[test]
	fn rejects_min_expiry_on_api_key() {
		let error = parse_auth_command(&args(&["auth", "print-api-key", "--min-expiry", "30m"]))
			.unwrap_err();
		assert!(error.0.contains("only supported by print-bearer-token"), "{error}");
	}

	#[test]
	fn is_help_detection() {
		assert!(is_auth_command_help(&args(&["auth", "--help"])));
		assert!(is_auth_command_help(&args(&["auth", "print-api-key", "--help"])));
		assert!(is_auth_command_help(&args(&["auth", "print-bearer-token", "-h"])));
		assert!(is_auth_command_help(&args(&["auth", "check", "--help"])));
		assert!(is_auth_command_help(&args(&["auth"])));
		assert!(!is_auth_command_help(&args(&["auth", "check"])));
	}

	#[test]
	fn rejects_unknown_auth_command() {
		let error = parse_auth_command(&args(&["auth", "unknown"])).unwrap_err();
		assert!(error.0.contains("Unknown auth command \"unknown\""), "{error}");
	}

	#[test]
	fn non_auth_input_parses_to_none() {
		assert!(parse_auth_command(&args(&["--provider", "openai"])).unwrap().is_none());
	}

	#[test]
	fn validates_check_requires_provider_or_model() {
		let error =
			validate_auth_command_args(&crate::cli::args::parse_args(&[]), AuthCommandKind::Check)
				.unwrap_err();
		assert!(error.0.contains("Auth checks require --provider"), "{error}");

		let error = validate_auth_command_args(
			&crate::cli::args::parse_args(&[]),
			AuthCommandKind::ApiKey,
		)
		.unwrap_err();
		assert!(error.0.contains("Credential printing requires --provider"), "{error}");
	}

	#[test]
	fn rejects_messages_and_unknown_flags() {
		let parsed = crate::cli::args::parse_args(&args(&["hello"]));
		let error =
			validate_auth_command_args(&parsed, AuthCommandKind::Check).unwrap_err();
		assert!(error.0.contains("only accept --provider and --model"), "{error}");

		let parsed = crate::cli::args::parse_args(&args(&["--unknown-flag"]));
		let error =
			validate_auth_command_args(&parsed, AuthCommandKind::Check).unwrap_err();
		assert!(error.0.contains("Unknown option --unknown-flag"), "{error}");
	}

	#[test]
	fn trims_provider_and_model() {
		let parsed = crate::cli::args::parse_args(&args(&["--provider", " openai "]));
		let (provider, model) =
			validate_auth_command_args(&parsed, AuthCommandKind::Check).unwrap();
		assert_eq!(provider.as_deref(), Some("openai"));
		assert_eq!(model, None);
	}

	#[test]
	fn extracts_credential_from_auth_result() {
		use pi_ai::auth::types::{AuthResult, ModelAuth};
		let auth = AuthResult {
			auth: ModelAuth {
				api_key: Some("test-key".to_string()),
				headers: None,
				base_url: None,
			},
			env: None,
			source: None,
		};
		assert_eq!(get_auth_credential(Some(&auth)).as_deref(), Some("test-key"));

		let auth = AuthResult {
			auth: ModelAuth {
				api_key: None,
				headers: Some(
					[("Authorization".to_string(), Some("Bearer header-token".to_string()))]
						.into_iter()
						.collect(),
				),
				base_url: None,
			},
			env: None,
			source: None,
		};
		assert_eq!(get_auth_credential(Some(&auth)).as_deref(), Some("header-token"));
		assert_eq!(get_auth_credential(None), None);
	}
}
