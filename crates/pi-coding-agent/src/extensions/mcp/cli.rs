//! Port of `packages/coding-agent/src/extensions/mcp/cli.ts`: `pi mcp` —
//! add, remove, and check MCP servers and sign in to them outside a session.
//! Agents run it; the user only approves access in the browser. Running
//! sessions pick up new credentials on their next turn.
//!
//! # Port deviations
//!
//! - **D81.** `chalk` styling is dropped (plain-text output), matching the
//!   TUI exclusion; the help text is the same modulo styling.
//! - **D88.** Tests drive the stdio fixture through `/bin/sh` (upstream runs
//!   `node …/stdio-server.mjs`); OS spawn errors surface the platform text,
//!   so assertions check the connection state rather than `ENOENT` strings.
//! - **D89.** Upstream `ProjectTrustStore` is ported as a local minimal
//!   reader of `trust.json` (nearest-ancestor entry) until the core
//!   trust-manager module lands; `waitForRedirectUrl` reads stdin through
//!   tokio instead of `node:readline`.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::config::{get_agent_dir, CONFIG_DIR_NAME, APP_NAME};
use crate::extensions::mcp::config::{
	add_mcp_server_config, get_mcp_tool_exposure, load_mcp_config, remove_mcp_server_config, update_mcp_server_config,
	LoadedMcpConfig, McpExposure, McpServerConfigPatch, McpServerEntry,
};
use crate::extensions::mcp::log::McpServerLog;
use crate::extensions::mcp::oauth::{
	sign_in_mcp_server, McpOAuthCredentialStore, McpSignInOptions, McpSignInPrompt, McpOAuthSettings,
	SIGN_IN_CANCELLED_MESSAGE,
};
use crate::extensions::mcp::runtime::{
	create_default_transport, McpCredentials, McpServerConnection, McpServerConnectionOptions,
};

const HELP: &str = "Usage:
  pi mcp add <server> [options] -- <command> [args...]
  pi mcp add <server> [options] --url <url>
  pi mcp remove <server> [-l]
  pi mcp list [--json]
  pi mcp login <server> [--timeout <seconds>]
  pi mcp logout <server>

Configure and check MCP servers and sign in to OAuth servers without starting a session.
Reads ~/.pi/agent/mcp.json and, in trusted projects, .pi/mcp.json.

Commands:
  add <server>            Add or replace a server in mcp.json
  remove <server>         Remove a server from mcp.json
  list                    Show state, tools, and errors (exits 1 on failure)
  login <server>          Sign in through the browser
  logout <server>         Delete the stored OAuth credentials

Options for add and remove:
  -l, --local             Use .pi/mcp.json in the current project instead of the global file

Options for add:
  --url <url>             Streamable HTTP server URL (instead of a command)
  --env <KEY=VALUE>       Environment variable for a stdio server (repeatable)
  --cwd <dir>             Working directory for a stdio server
  --header <KEY=VALUE>    HTTP header (repeatable)
  --bearer-token-env-var <NAME>
                          Send \"Authorization: Bearer ${NAME}\"
  --oauth-client-id <id>  Pre-registered OAuth client id
  --oauth-client-secret <secret>
                          OAuth client secret (may be ${NAME} or !command)
  --oauth-callback-port <port>
                          Fixed OAuth callback port
  --oauth-client-name <name>
                          Client name sent when registering with the OAuth server
  --exposure <mode>       codemode (default), deferred, direct, or hidden
  --description <text>    What the server offers, shown in the system prompt

Other options:
  --json                  Print the list as JSON
  --timeout <seconds>     How long login waits for the browser (default: 300)";

const HELP_HINT: &str = "Use \"pi mcp --help\" for usage.";

const DEFAULT_LOGIN_TIMEOUT_SECONDS: f64 = 300.0;

/// Upstream `McpCommandOptions`.
#[derive(Clone)]
pub struct McpCommandOptions {
	pub cwd: String,
	pub agent_dir: String,
	/// Defaults to `mcp-auth.json` in the agent directory.
	pub credentials: Option<Arc<dyn McpCredentials>>,
	/// Defaults to the platform browser.
	pub open_url: Option<Arc<dyn Fn(&str) + Send + Sync>>,
	/// Defaults to console output.
	pub log: Option<Arc<dyn Fn(&str) + Send + Sync>>,
	pub error: Option<Arc<dyn Fn(&str) + Send + Sync>>,
	/// Test seam: overrides the project-trust decision (upstream reads
	/// `trust.json` through `ProjectTrustStore`).
	pub project_trusted: Option<bool>,
}

#[derive(Default)]
struct ServerReport {
	name: String,
	scope: String,
	source: String,
	r#override: Option<String>,
	enabled: bool,
	exposure: String,
	transport: String,
	state: String,
	tools: Vec<String>,
	tool_exposure: BTreeMap<String, String>,
	resources: Option<usize>,
	resource_templates: Option<usize>,
	error: Option<String>,
}

fn create_connection(
	entry: McpServerEntry,
	options: &McpCommandOptions,
	credentials: &Arc<dyn McpCredentials>,
) -> Arc<McpServerConnection> {
	McpServerConnection::new(McpServerConnectionOptions {
		entry,
		cwd: options.cwd.clone(),
		create_transport: Arc::new(create_default_transport),
		credentials: Arc::clone(credentials),
		provider_token: None,
		on_tools: Arc::new(|_| {}),
		on_change: None,
		log: Some(Arc::new(parking_lot::Mutex::new(McpServerLog::new(format!(
			"{}/mcp.log",
			options.agent_dir
		))))),
	})
}

/// Short spellings of options. `-l`/`--local` match `pi install`
/// (`OPTION_ALIASES`).
fn option_alias(arg: &str) -> &str {
	match arg {
		"-l" => "--local",
		other => other,
	}
}

#[derive(Default)]
struct ParsedOptions {
	positional: Vec<String>,
	values: BTreeMap<String, Option<String>>,
	/// Values of `list` options, in order.
	lists: BTreeMap<String, Vec<String>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OptionKind {
	Flag,
	Value,
	List,
}

/// Parse `--name value` options; returns `None` and reports unknown ones.
/// `--` ends the options, as does reaching `max_positionals` positional
/// arguments: the remaining arguments are positional, so a command's own
/// options (`add <server> <command> --flag`) are passed through
/// (`parseOptions`).
fn parse_options(
	args: &[String],
	known: &BTreeMap<&'static str, OptionKind>,
	error: &dyn Fn(&str),
	max_positionals: usize,
) -> Option<ParsedOptions> {
	let mut parsed = ParsedOptions::default();
	let mut index = 0usize;
	while index < args.len() {
		let arg = option_alias(&args[index]).to_string();
		if arg == "--" || parsed.positional.len() >= max_positionals {
			let start = if arg == "--" { index + 1 } else { index };
			parsed.positional.extend(args[start..].iter().cloned());
			break;
		}
		if !arg.starts_with("--") {
			parsed.positional.push(arg);
			index += 1;
			continue;
		}
		let name = arg.trim_start_matches('-').to_string();
		let Some(kind) = known.get(name.as_str()).copied() else {
			error(&format!("Unknown option {arg}.\n{HELP_HINT}"));
			return None;
		};
		match kind {
			OptionKind::Flag => {
				parsed.values.insert(name, None);
			}
			OptionKind::Value | OptionKind::List => {
				index += 1;
				let Some(value) = args.get(index) else {
					error(&format!("{arg} needs a value."));
					return None;
				};
				if kind == OptionKind::List {
					parsed.lists.entry(name).or_default().push(value.clone());
				} else {
					parsed.values.insert(name, Some(value.clone()));
				}
			}
		}
		index += 1;
	}
	Some(parsed)
}

/// Minimal `trust.json` reader standing in for upstream `ProjectTrustStore`
/// (D89): decisions keyed by normalized path, nearest ancestor wins.
fn project_trust(agent_dir: &str, cwd: &str) -> Option<bool> {
	let trust_path = format!("{agent_dir}/trust.json");
	let Ok(text) = std::fs::read_to_string(&trust_path) else {
		return None;
	};
	let Ok(Value::Object(data)) = serde_json::from_str::<Value>(&text) else {
		return None;
	};
	let mut current = std::path::Path::new(cwd);
	loop {
		let key = current.to_string_lossy().into_owned();
		if let Some(decision) = data.get(&key).and_then(|value| value.as_str()) {
			return Some(decision == "trusted");
		}
		if !current.to_path_buf().pop() {
			return None;
		}
	}
}

/// Whether the project's `.pi` directory carries config that requires trust
/// (upstream `requiresTrustDecision` reduced to the mcp.json case the CLI
/// needs).
fn project_has_trusted_config(cwd: &str) -> bool {
	std::path::Path::new(&format!("{cwd}/{CONFIG_DIR_NAME}/mcp.json")).exists()
}

/// Run `pi mcp <args>` and return the exit code (`runMcpCommand`).
pub async fn run_mcp_command(args: &[String], options: &McpCommandOptions) -> i32 {
	let log: Arc<dyn Fn(&str) + Send + Sync> =
		options.log.clone().unwrap_or_else(|| Arc::new(|line| println!("{line}")));
	let error: Arc<dyn Fn(&str) + Send + Sync> = options
		.error
		.clone()
		.unwrap_or_else(|| Arc::new(|line| eprintln!("{line}")));
	let Some(command) = args.first() else {
		log(HELP);
		return 0;
	};
	if command == "help" || args.iter().any(|arg| arg == "--help" || arg == "-h") {
		log(HELP);
		return 0;
	}

	let project_config = format!("{}/{}/mcp.json", options.cwd, CONFIG_DIR_NAME);
	if command == "add" || command == "remove" {
		return if command == "add" {
			add_command(&args[1..], &project_config, options, &log, &error)
		} else {
			remove_command(&args[1..], &project_config, options, &log, &error)
		};
	}
	let project_trusted = options
		.project_trusted
		.unwrap_or_else(|| project_trust(&options.agent_dir, &options.cwd).unwrap_or(false));
	let loaded = load_mcp_config(&options.agent_dir, &options.cwd, project_trusted);
	let untrusted_note = if !project_trusted && std::path::Path::new(&project_config).exists() {
		Some(format!(
			"{project_config} is ignored because the project is not trusted. Start {APP_NAME} in the project to trust it."
		))
	} else {
		None
	};
	let credentials: Arc<dyn McpCredentials> = options
		.credentials
		.clone()
		.unwrap_or_else(|| Arc::new(McpOAuthCredentialStore::open_default()));

	match command.as_str() {
		"list" => {
			let mut known = BTreeMap::new();
			known.insert("json", OptionKind::Flag);
			let Some(parsed) = parse_options(&args[1..], &known, error.as_ref(), usize::MAX) else {
				return 1;
			};
			if !parsed.positional.is_empty() {
				error(&format!("Usage: {APP_NAME} mcp list [--json]\n{HELP_HINT}"));
				return 1;
			}
			list_command(
				&loaded,
				parsed.values.contains_key("json"),
				untrusted_note.as_deref(),
				options,
				&credentials,
				&log,
			)
			.await
		}
		"login" | "logout" => {
			let mut known = BTreeMap::new();
			if command == "login" {
				known.insert("timeout", OptionKind::Value);
			}
			let Some(parsed) = parse_options(&args[1..], &known, error.as_ref(), usize::MAX) else {
				return 1;
			};
			let [name, extra @ ..] = parsed.positional.as_slice() else {
				error(&format!("Usage: {APP_NAME} mcp {command} <server>\n{HELP_HINT}"));
				return 1;
			};
			if !extra.is_empty() {
				error(&format!("Usage: {APP_NAME} mcp {command} <server>\n{HELP_HINT}"));
				return 1;
			}
			let Some(entry) = loaded.servers.iter().find(|server| server.name == *name).cloned() else {
				let configured = loaded
					.servers
					.iter()
					.map(|server| server.name.clone())
					.collect::<Vec<_>>()
					.join(", ");
				error(&format!(
					"No MCP server named \"{name}\".{} Configured: {}.",
					untrusted_note.as_deref().map(|note| format!(" {note}")).unwrap_or_default(),
					if configured.is_empty() { "none".to_string() } else { configured },
				));
				return 1;
			};
			let connection = create_connection(entry.clone(), options, &credentials);
			let Some(url) = connection.oauth_url() else {
				error(&format!(
					"MCP server \"{name}\" does not use OAuth. Only HTTP servers without an Authorization header do."
				));
				return 1;
			};
			if command == "logout" {
				let removed = credentials.remove(name, &url);
				log(
					if removed {
						format!("Signed out of MCP server \"{name}\".")
					} else {
						format!("No stored credentials for MCP server \"{name}\".")
					}
					.as_str(),
				);
				return 0;
			}
			let timeout = parsed
				.values
				.get("timeout")
				.and_then(|value| value.clone())
				.map(|value| value.parse::<f64>())
				.transpose()
				.unwrap_or(Some(DEFAULT_LOGIN_TIMEOUT_SECONDS));
			let Some(timeout) = timeout.filter(|timeout| *timeout > 0.0) else {
				error("--timeout must be a positive number of seconds.");
				return 1;
			};
			let result = login_command(&entry, &connection, &url, (timeout * 1000.0) as u64, options, &credentials, &log, &error)
				.await;
			connection.close().await;
			result
		}
		other => {
			error(&format!("Unknown mcp command \"{other}\".\n{HELP_HINT}"));
			1
		}
	}
}

/// Parse `KEY=VALUE` pairs of a repeatable option into a record
/// (`parsePairs`). Returns `None` after reporting a malformed pair.
fn parse_pairs(option: &str, pairs: Option<&Vec<String>>, error: &dyn Fn(&str)) -> Option<BTreeMap<String, String>> {
	let Some(pairs) = pairs else {
		return Some(BTreeMap::new());
	};
	let mut record = BTreeMap::new();
	for pair in pairs {
		let Some(separator) = pair.find('=') else {
			error(&format!("--{option} expects KEY=VALUE, got \"{pair}\"."));
			return None;
		};
		if separator == 0 {
			error(&format!("--{option} expects KEY=VALUE, got \"{pair}\"."));
			return None;
		}
		record.insert(pair[..separator].to_string(), pair[separator + 1..].to_string());
	}
	Some(record)
}

fn add_command(
	args: &[String],
	project_config: &str,
	options: &McpCommandOptions,
	log: &Arc<dyn Fn(&str) + Send + Sync>,
	error: &Arc<dyn Fn(&str) + Send + Sync>,
) -> i32 {
	let usage = format!("Usage: {APP_NAME} mcp add <server> [options] (--url <url> | -- <command> [args...])\n{HELP_HINT}");
	let mut known = BTreeMap::new();
	known.insert("local", OptionKind::Flag);
	known.insert("url", OptionKind::Value);
	known.insert("env", OptionKind::List);
	known.insert("cwd", OptionKind::Value);
	known.insert("header", OptionKind::List);
	known.insert("bearer-token-env-var", OptionKind::Value);
	known.insert("oauth-client-id", OptionKind::Value);
	known.insert("oauth-client-secret", OptionKind::Value);
	known.insert("oauth-callback-port", OptionKind::Value);
	known.insert("oauth-client-name", OptionKind::Value);
	known.insert("exposure", OptionKind::Value);
	known.insert("description", OptionKind::Value);
	let Some(parsed) = parse_options(args, &known, error.as_ref(), 2) else {
		return 1;
	};
	let [name, command @ ..] = parsed.positional.as_slice() else {
		error(&usage);
		return 1;
	};
	let name = name.clone();
	let url = parsed.values.get("url").cloned().flatten();
	if url.is_none() == command.is_empty() {
		error(&usage);
		return 1;
	}
	let value = |option: &str| parsed.values.get(option).cloned().flatten();
	let exposure = value("exposure");
	let http_only = [
		"header",
		"bearer-token-env-var",
		"oauth-client-id",
		"oauth-client-secret",
		"oauth-callback-port",
		"oauth-client-name",
	];
	let stdio_only = ["env", "cwd"];
	let misplaced = (if url.is_none() { http_only.as_slice() } else { stdio_only.as_slice() })
		.iter()
		.find(|option| parsed.values.contains_key(**option) || parsed.lists.contains_key(**option));
	if let Some(misplaced) = misplaced {
		error(&format!(
			"--{misplaced} only applies to {}.",
			if url.is_none() { "HTTP servers (--url)" } else { "stdio servers" }
		));
		return 1;
	}

	let config: Value;
	if let Some(url) = url {
		let Some(headers) = parse_pairs("header", parsed.lists.get("header"), error.as_ref()) else {
			return 1;
		};
		let mut headers: BTreeMap<String, String> = headers;
		if let Some(bearer) = value("bearer-token-env-var") {
			headers.insert("Authorization".to_string(), format!("Bearer ${{{bearer}}}"));
		}
		let mut oauth = Map::new();
		if let Some(client_id) = value("oauth-client-id") {
			oauth.insert("clientId".to_string(), json!(client_id));
		}
		if let Some(client_secret) = value("oauth-client-secret") {
			oauth.insert("clientSecret".to_string(), json!(client_secret));
		}
		if let Some(port) = value("oauth-callback-port") {
			oauth.insert("callbackPort".to_string(), json!(port));
		}
		if let Some(client_name) = value("oauth-client-name") {
			oauth.insert("clientName".to_string(), json!(client_name));
		}
		let mut object = serde_json::Map::new();
		object.insert("url".to_string(), json!(url));
		if !headers.is_empty() {
			object.insert("headers".to_string(), serde_json::to_value(&headers).expect("headers serialization is infallible"));
		}
		if !oauth.is_empty() {
			object.insert("oauth".to_string(), Value::Object(oauth));
		}
		config = Value::Object(object);
	} else {
		let Some(env) = parse_pairs("env", parsed.lists.get("env"), error.as_ref()) else {
			return 1;
		};
		let [executable, command_args @ ..] = command else {
			error(&usage);
			return 1;
		};
		let mut object = serde_json::Map::new();
		object.insert("command".to_string(), json!(executable));
		if !command_args.is_empty() {
			object.insert("args".to_string(), json!(command_args));
		}
		if !env.is_empty() {
			object.insert("env".to_string(), serde_json::to_value(&env).expect("env serialization is infallible"));
		}
		if let Some(cwd) = value("cwd") {
			object.insert("cwd".to_string(), json!(cwd));
		}
		config = Value::Object(object);
	}
	let mut config = config;
	if let Some(exposure) = exposure.as_ref() {
		config["exposure"] = json!(exposure);
	}
	if let Some(description) = value("description") {
		config["description"] = json!(description);
	}
	let validated = match crate::extensions::mcp::config::validate_mcp_server_config(&name, &config) {
		Ok(validated) => validated,
		Err(message) => {
			error(&message);
			return 1;
		}
	};

	let project = parsed.values.contains_key("local");
	let path = if project {
		project_config.to_string()
	} else {
		format!("{}/mcp.json", options.agent_dir)
	};
	let scope = if project { "project" } else { "global" };
	let replaced = match add_mcp_server_config(&path, &name, &validated) {
		Ok(replaced) => replaced,
		Err(add_error) => {
			error(&format!("Could not update {path}: {add_error}"));
			return 1;
		}
	};
	log(
		if replaced {
			format!("Replaced {scope} MCP server \"{name}\" in {path}.")
		} else {
			format!("Added {scope} MCP server \"{name}\" in {path}.")
		}
		.as_str(),
	);
	if project && !options.project_trusted.unwrap_or_else(|| project_trust(&options.agent_dir, &options.cwd).unwrap_or(false)) {
		log(&format!(
			"The project is not trusted, so {path} is ignored until you start {APP_NAME} in the project and trust it."
		));
	}
	// HTTP servers without an Authorization header may use OAuth.
	let may_need_sign_in = validated.get("url").is_some()
		&& !validated
			.get("headers")
			.and_then(|headers| headers.as_object())
			.map(|headers| {
				headers
					.keys()
					.any(|header| header.eq_ignore_ascii_case("authorization"))
			})
			.unwrap_or(false);
	log(
		if may_need_sign_in {
			format!("Check it with: {APP_NAME} mcp list. If it requires sign-in: {APP_NAME} mcp login {name}")
		} else {
			format!("Check it with: {APP_NAME} mcp list")
		}
		.as_str(),
	);
	0
}

fn remove_command(
	args: &[String],
	project_config: &str,
	options: &McpCommandOptions,
	log: &Arc<dyn Fn(&str) + Send + Sync>,
	error: &Arc<dyn Fn(&str) + Send + Sync>,
) -> i32 {
	let mut known = BTreeMap::new();
	known.insert("local", OptionKind::Flag);
	let Some(parsed) = parse_options(args, &known, error.as_ref(), usize::MAX) else {
		return 1;
	};
	let [name, extra @ ..] = parsed.positional.as_slice() else {
		error(&format!("Usage: {APP_NAME} mcp remove <server> [-l]\n{HELP_HINT}"));
		return 1;
	};
	if !extra.is_empty() {
		error(&format!("Usage: {APP_NAME} mcp remove <server> [-l]\n{HELP_HINT}"));
		return 1;
	}
	let name = name.clone();
	let project = parsed.values.contains_key("local");
	let global_config = format!("{}/mcp.json", options.agent_dir);
	let path = if project { project_config.to_string() } else { global_config };
	let scope = if project { "project" } else { "global" };
	let removed = match remove_mcp_server_config(&path, &name) {
		Ok(removed) => removed,
		Err(remove_error) => {
			error(&format!("Could not update {path}: {remove_error}"));
			return 1;
		}
	};
	if removed {
		log(&format!("Removed {scope} MCP server \"{name}\" from {path}."));
		return 0;
	}
	let other = load_mcp_config(&options.agent_dir, &options.cwd, true)
		.servers
		.into_iter()
		.find(|server| server.name == name && server.scope.map(|server_scope| server_scope.as_str() != scope).unwrap_or(true));
	let hint = other
		.map(|other| {
			format!(
				" It is defined in {}{}.",
				other.source,
				if other.scope == Some(crate::extensions::mcp::config::McpServerScope::Project) {
					"; use --local"
				} else {
					"; omit --local"
				}
			)
		})
		.unwrap_or_default();
	error(&format!("No {scope} MCP server named \"{name}\" in {path}.{hint}"));
	1
}

async fn list_command(
	loaded: &LoadedMcpConfig,
	as_json: bool,
	untrusted_note: Option<&str>,
	options: &McpCommandOptions,
	credentials: &Arc<dyn McpCredentials>,
	log: &Arc<dyn Fn(&str) + Send + Sync>,
) -> i32 {
	let mut reports: Vec<ServerReport> = Vec::new();
	for entry in &loaded.servers {
		let mut report = ServerReport {
			name: entry.name.clone(),
			scope: entry.scope.map(|scope| scope.as_str().to_string()).unwrap_or_else(|| "global".to_string()),
			source: entry.source.clone(),
			r#override: entry.r#override.clone(),
			enabled: entry.config.get("enabled").and_then(|value| value.as_bool()).unwrap_or(true),
			exposure: entry
				.config
				.get("exposure")
				.and_then(|value| value.as_str())
				.unwrap_or("codemode")
				.to_string(),
			transport: describe_transport(entry),
			state: "disabled".to_string(),
			tools: Vec::new(),
			..Default::default()
		};
		if !report.enabled {
			reports.push(report);
			continue;
		}
		let connection = create_connection(entry.clone(), options, credentials);
		let _ = connection.get_client().await;
		report.state = connection.state().map(|state| state.as_str().to_string()).unwrap_or_else(|| "connecting".to_string());
		report.tools = connection.tools().iter().map(|tool| tool.name.clone()).collect();
		for tool in &report.tools {
			let exposure = get_mcp_tool_exposure(&entry.config, tool);
			if exposure.as_str() != report.exposure {
				report.tool_exposure.insert(tool.clone(), exposure.as_str().to_string());
			}
		}
		if connection.has_resources() {
			report.resources = Some(connection.resources().len());
			report.resource_templates = Some(connection.resource_templates().len());
		}
		if report.state != "connected" {
			report.error = connection.error();
		}
		connection.close().await;
		reports.push(report);
	}
	let failed =
		!loaded.errors.is_empty() || reports.iter().any(|report| report.enabled && report.state != "connected");

	if as_json {
		let servers: Vec<Value> = reports
			.iter()
			.map(|report| {
				let mut object = json!({
					"name": report.name,
					"scope": report.scope,
					"source": report.source,
					"enabled": report.enabled,
					"exposure": report.exposure,
					"transport": report.transport,
					"state": report.state,
					"tools": report.tools,
				});
				if let Some(override_path) = &report.r#override {
					object["override"] = json!(override_path);
				}
				if !report.tool_exposure.is_empty() {
					object["toolExposure"] = serde_json::to_value(&report.tool_exposure).expect("map serialization is infallible");
				}
				if let Some(resources) = report.resources {
					object["resources"] = json!(resources);
					object["resourceTemplates"] = json!(report.resource_templates.unwrap_or(0));
				}
				if let Some(error) = &report.error {
					object["error"] = json!(error);
				}
				object
			})
			.collect();
		let mut payload = json!({ "servers": servers, "errors": loaded.errors });
		if let Some(note) = untrusted_note {
			payload["note"] = json!(note);
		}
		log(&serde_json::to_string_pretty(&payload).expect("payload serialization is infallible"));
		return if failed { 1 } else { 0 };
	}
	if reports.is_empty() && loaded.errors.is_empty() {
		log(&format!(
			"No MCP servers configured. Add them to {}/mcp.json or .pi/mcp.json.",
			options.agent_dir
		));
	}
	for report in &reports {
		let state = match report.state.as_str() {
			"connected" => format!("connected, {} tool{}", report.tools.len(), if report.tools.len() == 1 { "" } else { "s" }),
			"needs-auth" => "needs sign-in".to_string(),
			other => other.to_string(),
		};
		log(&format!("{}: {} ({}, {})", report.name, state, report.exposure, report.scope));
		log(&format!("  {}", report.transport));
		if let Some(override_path) = &report.r#override {
			log(&format!("  project override: {override_path}"));
		}
		if report.state == "needs-auth" {
			log(&format!("  sign in with: {APP_NAME} mcp login {}", report.name));
		}
		if !report.tools.is_empty() {
			let tools = report
				.tools
				.iter()
				.map(|tool| match report.tool_exposure.get(tool) {
					Some(exposure) => format!("{tool} [{exposure}]"),
					None => tool.clone(),
				})
				.collect::<Vec<_>>()
				.join(", ");
			log(&format!("  tools: {tools}"));
		}
		if let Some(resources) = report.resources {
			log(&format!(
				"  resources: {resources}, URI templates: {}",
				report.resource_templates.unwrap_or(0)
			));
		}
		if let Some(error_text) = &report.error {
			log(&format!("  {}", error_text.replace('\n', "\n  ")));
		}
	}
	for config_error in &loaded.errors {
		log(&format!("config error: {config_error}"));
	}
	if let Some(note) = untrusted_note {
		log(note);
	}
	if failed { 1 } else { 0 }
}

fn describe_transport(entry: &McpServerEntry) -> String {
	let config = &entry.config;
	if let Some(url) = config.get("url").and_then(|value| value.as_str()) {
		return url.to_string();
	}
	let command = config.get("command").and_then(|value| value.as_str()).unwrap_or_default();
	let args = config
		.get("args")
		.and_then(|args| args.as_array())
		.map(|args| {
			args.iter()
				.filter_map(|arg| arg.as_str())
				.collect::<Vec<_>>()
				.join(" ")
		})
		.unwrap_or_default();
	if args.is_empty() {
		command.to_string()
	} else {
		format!("{command} {args}")
	}
}

async fn login_command(
	entry: &McpServerEntry,
	connection: &Arc<McpServerConnection>,
	url: &str,
	timeout_ms: u64,
	options: &McpCommandOptions,
	credentials: &Arc<dyn McpCredentials>,
	log: &Arc<dyn Fn(&str) + Send + Sync>,
	error: &Arc<dyn Fn(&str) + Send + Sync>,
) -> i32 {
	let name = &entry.name;
	// Connecting first answers whether a sign-in is needed and records the
	// server's challenge.
	if connection.get_client().await.is_ok() {
		log(&format!(
			"Already signed in to MCP server \"{name}\" ({} tools).",
			connection.tools().len()
		));
		return 0;
	}
	if connection.state() != Some(crate::extensions::mcp::runtime::ServerState::NeedsAuth) {
		error(&format!(
			"MCP server \"{name}\" failed to connect: {}",
			connection.error().unwrap_or_else(|| "unknown error".to_string())
		));
		return 1;
	}

	let open_url = options.open_url.clone().unwrap_or_else(|| Arc::new(|url| {
		let _ = std::process::Command::new("open").arg(url).spawn();
	}));
	let interactive = options.open_url.is_none();
	let prompt = McpSignInPrompt {
		show_authorization_url: Arc::new({
			let name = name.clone();
			let log = Arc::clone(log);
			let open_url = Arc::clone(&open_url);
			move |url| {
				log(&format!("Sign in to MCP server \"{name}\" in your browser:\n{url}"));
				open_url(url.as_str());
			}
		}),
		prompt_for_redirect_url: Arc::new(move |signal| {
			Box::pin(wait_for_redirect_url(signal, timeout_ms, interactive))
		}),
	};
	if let Err(sign_in_error) = sign_in_mcp_server(McpSignInOptions {
		server_url: url.to_string(),
		store: credentials.for_server(name, url),
		settings: connection.oauth_settings().unwrap_or_default(),
		challenge: connection.challenge(),
		prompt,
	})
	.await
	{
		if sign_in_error.to_string() == SIGN_IN_CANCELLED_MESSAGE {
			error(&format!(
				"Sign-in to MCP server \"{name}\" was cancelled or not completed within {} seconds.",
				(timeout_ms as f64 / 1000.0).round() as i64
			));
		} else {
			error(&format!(
				"Sign-in to MCP server \"{name}\" failed: {sign_in_error}"
			));
		}
		return 1;
	}
	if let Err(connect_error) = connection.after_sign_in().await {
		error(&format!("Signed in, but {connect_error}"));
		return 1;
	}
	log(&format!(
		"Signed in to MCP server \"{name}\" ({} tools).",
		connection.tools().len()
	));
	0
}

/// The pasted redirect URL in a terminal; otherwise only the browser callback
/// can finish the sign-in. Returns `None` (cancelling the sign-in) after
/// `timeout_ms`, or when the callback arrived (`waitForRedirectUrl`).
async fn wait_for_redirect_url(
	signal: pi_mcp::client::AbortSignal,
	timeout_ms: u64,
	interactive: bool,
) -> Option<String> {
	let _ = signal;
	if !interactive {
		return None;
	}
	let line = tokio::select! {
		line = tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), read_stdin_line()) => line.ok().flatten(),
		_ = tokio::time::sleep(std::time::Duration::from_millis(timeout_ms)) => None,
	};
	line.filter(|line| !line.trim().is_empty())
}

async fn read_stdin_line() -> Option<String> {
	use tokio::io::AsyncBufReadExt;
	let stdin = tokio::io::stdin();
	let mut lines = tokio::io::BufReader::new(stdin).lines();
	lines.next_line().await.ok().flatten()
}

/// The path of the global config, for tests (`join(agentDir, "mcp.json")`).
pub fn global_config_path(agent_dir: &str) -> String {
	format!("{agent_dir}/mcp.json")
}

/// Update helper re-exported for tests (upstream `updateMcpServerConfig` via
/// `/mcp`).
pub fn update_config_for_test(path: &str, name: &str, patch: McpServerConfigPatch, r#override: bool) -> Result<(), String> {
	update_mcp_server_config(path, name, patch, r#override)
}

