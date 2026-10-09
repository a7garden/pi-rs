//! Port of `test/mcp-command.test.ts` (`pi mcp`). The stdio fixture is a
//! `/bin/sh` JSON-RPC loop (upstream runs `node …/stdio-server.mjs` — D88).

use std::sync::Arc;

use serde_json::{json, Value};

use super::cli::*;
use super::config::load_mcp_config;

const FIXTURE_SCRIPT: &str = "while IFS= read -r line; do\n\
	id=$(printf '%s' \"$line\" | sed -n 's/.*\"id\":\\([0-9]*\\).*/\\1/p')\n\
	[ -z \"$id\" ] && continue\n\
	case \"$line\" in\n\
	*initialize*) result='{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"fixture\",\"version\":\"1\"}}';;\n\
	*tools/list*) result='{\"tools\":[{\"name\":\"echo\",\"inputSchema\":{\"type\":\"object\"}}]}';;\n\
	*) result='{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}';;\n\
	esac\n\
	printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":%s}\\n' \"$id\" \"$result\"\n\
done";

fn fixture_server() -> Value {
	json!({ "command": "/bin/sh", "args": ["-c", FIXTURE_SCRIPT] })
}

fn broken_server() -> Value {
	json!({ "command": "pi-test-missing-mcp-server" })
}

fn parked_server() -> Value {
	json!({ "command": "/bin/sh", "args": ["-c", FIXTURE_SCRIPT], "enabled": false })
}

fn bad_server() -> Value {
	json!({ "args": ["no command"] })
}

fn temp_dir(prefix: &str) -> String {
	static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
	let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
	let dir = std::env::temp_dir()
		.join(format!("{prefix}{}-{id}", std::process::id()))
		.to_string_lossy()
		.into_owned();
	std::fs::create_dir_all(&dir).expect("temp dir");
	dir
}

struct Run {
	exit_code: i32,
	output: String,
	agent_dir: String,
}

async fn run(args: &[&str], servers: Option<Value>) -> Run {
	let agent_dir = temp_dir("pi-mcp-command-");
	run_in(&agent_dir, args, servers).await
}

async fn run_in(agent_dir: &str, args: &[&str], servers: Option<Value>) -> Run {
	if let Some(servers) = servers {
		std::fs::write(
			format!("{agent_dir}/mcp.json"),
			serde_json::to_string(&json!({ "mcpServers": servers })).expect("serialize"),
		)
		.expect("write config");
	}
	let output: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
	let sink = Arc::clone(&output);
	let error_sink = Arc::clone(&output);
	let options = McpCommandOptions {
		cwd: agent_dir.to_string(),
		agent_dir: agent_dir.to_string(),
		credentials: None,
		open_url: Some(Arc::new(|_url| {})),
		log: Some(Arc::new(move |line| sink.lock().expect("lock").push(line.to_string()))),
		error: Some(Arc::new(move |line| error_sink.lock().expect("lock").push(line.to_string()))),
		project_trusted: Some(false),
	};
	let args: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
	let exit_code = run_mcp_command(&args, &options).await;
	let joined = output.lock().expect("lock").join("\n");
	Run {
		exit_code,
		output: joined,
		agent_dir: agent_dir.to_string(),
	}
}

fn read_config(path: &str) -> Value {
	serde_json::from_str(&std::fs::read_to_string(path).expect("read config")).expect("parse config")
}

/// Port of "lists servers with their state, tools, and errors, and fails while
/// anything is wrong" (the spawn-error text is OS-reported — D88 — so the
/// `broken` assertion checks the state line only).
#[tokio::test(flavor = "multi_thread")]
async fn lists_servers_with_their_state_tools_and_errors() {
	let result = run(
		&["list"],
		Some(json!({
			"fixture": fixture_server(),
			"broken": broken_server(),
			"parked": parked_server(),
			"bad": bad_server()
		})),
	)
	.await;
	assert_eq!(result.exit_code, 1);
	assert!(result.output.contains("fixture: connected, 1 tool (codemode, global)"), "output: {}", result.output);
	assert!(result.output.contains("  tools: echo"));
	assert!(result.output.contains("broken: failed (codemode, global)"));
	assert!(result.output.contains("parked: disabled (codemode, global)"));
	assert!(result.output.contains("config error: "));
	assert!(result.output.contains("server \"bad\" needs either \"command\""));

	let ok = run(&["list"], Some(json!({ "fixture": fixture_server() }))).await;
	assert_eq!(ok.exit_code, 0);
}

/// Port of "prints JSON for scripts".
#[tokio::test(flavor = "multi_thread")]
async fn prints_json_for_scripts() {
	let result = run(
		&["list", "--json"],
		Some(json!({ "fixture": fixture_server(), "parked": parked_server() })),
	)
	.await;
	assert_eq!(result.exit_code, 0, "output: {}", result.output);
	let parsed: Value = serde_json::from_str(&result.output).expect("parse json");
	let summary: Vec<(String, String, Vec<String>)> = parsed["servers"]
		.as_array()
		.expect("servers")
		.iter()
		.map(|server| {
			(
				server["name"].as_str().expect("name").to_string(),
				server["state"].as_str().expect("state").to_string(),
				server["tools"]
					.as_array()
					.expect("tools")
					.iter()
					.map(|tool| tool.as_str().expect("tool").to_string())
					.collect(),
			)
		})
		.collect();
	assert_eq!(
		summary,
		vec![
			("fixture".to_string(), "connected".to_string(), vec!["echo".to_string()]),
			("parked".to_string(), "disabled".to_string(), Vec::new()),
		]
	);
}

/// Port of "rejects unknown servers and servers without OAuth for login and
/// logout".
#[tokio::test(flavor = "multi_thread")]
async fn rejects_unknown_servers_and_servers_without_oauth() {
	let servers = json!({
		"fixture": fixture_server(),
		"broken": broken_server(),
		"parked": parked_server(),
		"bad": bad_server()
	});
	let missing = run(&["login", "nope"], Some(servers.clone())).await;
	assert_eq!(missing.exit_code, 1);
	assert!(
		missing
			.output
			.contains("No MCP server named \"nope\". Configured: fixture, broken, parked."),
		"output: {}",
		missing.output
	);
	let no_oauth = run(&["logout", "fixture"], Some(servers.clone())).await;
	assert_eq!(no_oauth.exit_code, 1);
	assert!(
		no_oauth
			.output
			.contains("MCP server \"fixture\" does not use OAuth. Only HTTP servers without an Authorization header do.")
	);
	let unknown = run(&["frobnicate"], Some(servers)).await;
	assert_eq!(unknown.exit_code, 1);
}

/// Port of "adds stdio servers and passes options after the command through".
#[tokio::test(flavor = "multi_thread")]
async fn adds_stdio_servers_and_passes_options_through() {
	let added = run(
		&["add", "--env", "A=1", "--env", "B=x=y", "files", "--", "npx", "-y", "server", "--root", "."],
		None,
	)
	.await;
	assert_eq!(added.exit_code, 0, "output: {}", added.output);
	assert!(added.output.contains("Added global MCP server \"files\""));
	assert_eq!(
		read_config(&global_config_path(&added.agent_dir)),
		json!({
			"mcpServers": {
				"files": { "command": "npx", "args": ["-y", "server", "--root", "."], "env": { "A": "1", "B": "x=y" } }
			}
		})
	);

	// Without `--`, options after the command belong to the command too.
	let replaced = run_in(&added.agent_dir, &["add", "files", "node", "server.js", "--port", "1"], None).await;
	assert!(replaced.output.contains("Replaced global MCP server \"files\""));
	assert_eq!(
		read_config(&global_config_path(&added.agent_dir)),
		json!({
			"mcpServers": { "files": { "command": "node", "args": ["server.js", "--port", "1"] } }
		})
	);
}

/// Port of "adds HTTP servers and keeps other content of the file".
#[tokio::test(flavor = "multi_thread")]
async fn adds_http_servers_and_keeps_other_content() {
	let result = run(
		&[
			"add",
			"docs",
			"--url",
			"https://example.com/mcp",
			"--bearer-token-env-var",
			"DOCS_TOKEN",
			"--header",
			"X-Team=core",
			"--exposure",
			"direct",
			"--description",
			"Product docs",
		],
		Some(json!({ "fixture": fixture_server() })),
	)
	.await;
	assert_eq!(result.exit_code, 0, "output: {}", result.output);
	assert!(!result.output.contains("mcp login"));
	assert_eq!(
		read_config(&global_config_path(&result.agent_dir)),
		json!({
			"mcpServers": {
				"fixture": fixture_server(),
				"docs": {
					"url": "https://example.com/mcp",
					"headers": { "X-Team": "core", "Authorization": "Bearer ${DOCS_TOKEN}" },
					"exposure": "direct",
					"description": "Product docs"
				}
			}
		})
	);

	let oauth = run_in(
		&result.agent_dir,
		&[
			"add",
			"sentry",
			"--url",
			"https://mcp.sentry.dev/mcp",
			"--oauth-client-id",
			"pi",
			"--oauth-client-name",
			"Claude Code",
		],
		None,
	)
	.await;
	assert!(oauth.output.contains("If it requires sign-in: pi mcp login sentry"));
	let config = read_config(&global_config_path(&result.agent_dir));
	assert_eq!(
		config["mcpServers"]["sentry"],
		json!({ "url": "https://mcp.sentry.dev/mcp", "oauth": { "clientId": "pi", "clientName": "Claude Code" } })
	);
}

/// Port of "rejects invalid add invocations without writing".
#[tokio::test(flavor = "multi_thread")]
async fn rejects_invalid_add_invocations_without_writing() {
	let cases: Vec<Vec<&str>> = vec![
		vec!["add", "x"],
		vec!["add", "x", "--url", "https://example.com", "--", "cmd"],
		vec!["add", "bad name", "--", "cmd"],
		vec!["add", "x", "--url", "ftp://example.com"],
		vec!["add", "x", "--env", "A=1", "--url", "https://example.com"],
		vec!["add", "x", "--header", "A=1", "--", "cmd"],
		vec!["add", "x", "--env", "NOVALUE", "--", "cmd"],
		vec!["add", "x", "--exposure", "loud", "--", "cmd"],
	];
	for args in &cases {
		let result = run(args, None).await;
		assert_eq!(result.exit_code, 1, "args: {}", args.join(" "));
		assert!(
			!std::path::Path::new(&global_config_path(&result.agent_dir)).exists(),
			"args {}: config written",
			args.join(" ")
		);
	}
}

/// Port of "adds and removes project servers".
#[tokio::test(flavor = "multi_thread")]
async fn adds_and_removes_project_servers() {
	let added = run(&["add", "-l", "local", "--", "node", "server.js"], None).await;
	assert!(added.output.contains("The project is not trusted"), "output: {}", added.output);
	let project_config = format!("{}/.pi/mcp.json", added.agent_dir);
	assert_eq!(
		read_config(&project_config),
		json!({ "mcpServers": { "local": { "command": "node", "args": ["server.js"] } } })
	);

	let wrong_scope = run_in(&added.agent_dir, &["remove", "local"], None).await;
	assert_eq!(wrong_scope.exit_code, 1);
	assert!(
		wrong_scope
			.output
			.contains(&format!("It is defined in {project_config}; use --local.")),
		"output: {}",
		wrong_scope.output
	);

	let removed = run_in(&added.agent_dir, &["remove", "local", "--local"], None).await;
	assert_eq!(removed.exit_code, 0);
	assert!(removed.output.contains("Removed project MCP server \"local\""));
	assert_eq!(read_config(&project_config), json!({ "mcpServers": {} }));
}

/// Port of "removes global servers".
#[tokio::test(flavor = "multi_thread")]
async fn removes_global_servers() {
	let servers = json!({
		"fixture": fixture_server(),
		"broken": broken_server(),
		"parked": parked_server(),
		"bad": bad_server()
	});
	let result = run(&["remove", "broken"], Some(servers)).await;
	assert_eq!(result.exit_code, 0, "output: {}", result.output);
	let config = read_config(&global_config_path(&result.agent_dir));
	let keys: Vec<&str> = config["mcpServers"]
		.as_object()
		.expect("mcpServers")
		.keys()
		.map(String::as_str)
		.collect();
	assert_eq!(keys, vec!["fixture", "parked", "bad"]);
	let missing = run_in(&result.agent_dir, &["remove", "broken"], None).await;
	assert_eq!(missing.exit_code, 1);
	assert!(missing.output.contains("No global MCP server named \"broken\""));
}

/// The loaded-config seam the command shares with the extension: listing an
/// unknown agent dir reports no servers.
#[tokio::test]
async fn load_config_reports_empty_without_files() {
	let agent_dir = temp_dir("pi-mcp-command-empty-");
	let loaded = load_mcp_config(&agent_dir, &agent_dir, false);
	assert!(loaded.servers.is_empty());
	assert!(loaded.errors.is_empty());
}

