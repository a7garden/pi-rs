//! Port of `packages/coding-agent/src/extensions/mcp/index.ts` — the built-in
//! MCP integration entry.
//!
//! The upstream file is a JS module evaluated by the extension loader: its
//! `createMcpExtension()` factory registers tools, commands, renderers, and
//! event handlers through the extension API. The Rust port keeps the
//! *factory surface* native ([`create_mcp_extension`] over
//! [`ExtensionFactory`](crate::core::extensions::types::ExtensionFactory));
//! the JS module evaluation itself stays behind the D40
//! [`core::extensions::loader`] seam.
//!
//! # Files
//!
//! | Upstream | Port | Status |
//! |---|---|---|
//! | `index.ts` | this module + [`factory`] | ported (JS eval → D40) |
//! | `config.ts` | [`crate::extensions::mcp::config`] | ported |
//! | `oauth.ts` | [`crate::extensions::mcp::oauth`] | ported (glue; flow in pi-mcp) |
//! | `tools.ts` | [`crate::extensions::mcp::tools`] | ported (renderers → D81) |
//! | `resources.ts` | [`crate::extensions::mcp::resources`] | ported |
//! | `runtime.ts` | [`crate::extensions::mcp::runtime`] | ported |
//! | `cli.ts` | [`crate::extensions::mcp::cli`] | ported (chalk → D81) |
//! | `log.ts` | [`crate::extensions::mcp::log`] | ported |
//! | `ui.ts` | — | deferred (TUI-only, D81) |
//! | `cli.lazy.ts` / `runtime.lazy.ts` | — | folded (Rust linking needs no lazy loader, D81) |
//!
//! # Deviations (D80–D89)
//!
//! - **D80.** `index.ts` JS module evaluation defers to the D40
//!   `ExtensionModuleLoader`; `create_mcp_extension` is the native factory.
//!   The `before_agent_start` handler computes the `mcp_servers` section, but
//!   the runner's erased `BeforeAgentStartEventResult` cannot yet carry
//!   section mutations (only `systemPrompt`), so section publishing waits on
//!   runner support; the pure renderer ([`render_servers_section`]) is ported
//!   and tested.
//! - **D81.** TUI-only surface defers: `ui.ts` and `showMcpManager`, the
//!   tool renderers (`createMcpToolRenderers`), the `/mcp` TUI manager
//!   screens, and chalk styling in the CLI. `cli.lazy.ts`/`runtime.lazy.ts`
//!   fold into plain module imports.
//! - **D82.** `mcp.json` rewrites use a local pretty printer over
//!   order-preserving `serde_json` (see config.rs).
//! - **D83.** The generic, non-object-safe `AuthStorageBackend` erases behind
//!   [`runtime::McpCredentials`] / `Arc<dyn McpOAuthServerStore>`.
//! - **D84.** The refresh lock is a create-exclusive lock file with mtime
//!   staleness (proper-lockfile constants preserved).
//! - **D85.** `URL` → the `url` crate; `[::1]` matched literally.
//! - **D86.** Local base64url/hex encoders (no base64 dependency).
//! - **D87.** OAuth callback pages render plain text.
//! - **D88.** Test fixtures: in-module tokio HTTP OAuth server; `/bin/sh`
//!   stdio JSON-RPC fixture.
//! - **D89.** JS string semantics: section budgets and the 64-character tool
//!   name cap count UTF-8 bytes where upstream counts UTF-16 units (ASCII
//!   equivalent; the limits are generous). `McpToolDetails` is erased JSON.
//!
//! # Deferred tests
//!
//! - `test/suite/agent-session-mcp*.test.ts` — AgentSession↔MCP integration
//!   exercises the TUI/RPC session hosts; deferred with the session-host
//!   ports.
//! - TUI manager flows (`/mcp` menus, tools listing screens) — D81.
//! - `refreshes once …` asserts upstream's lock-dir emptiness through the
//!   shared lock directory; the Rust refresh path shares one process lock
//!   (D84) and the test asserts the single refresh + lock release directly.

/// Name of the system prompt section that lists the servers whose tools are
/// not declared (`MCP_SERVERS_SECTION`).
pub const MCP_SERVERS_SECTION: &str = "mcp_servers";
/// Characters of a server description in the section, as Codex allows for
/// deferred namespaces.
const MAX_SERVER_DESCRIPTION_CHARS: usize = 250;
/// Characters of the whole section. Descriptions shrink to fit; when the
/// server lines alone do not fit, the last servers are left out and counted in
/// a closing line.
pub const MAX_SERVERS_SECTION_CHARS: usize = 4096;

use serde_json::Value;

use crate::extensions::mcp::config::{
	get_mcp_tool_exposure, mcp_namespace, McpExposure, McpServerEntry,
};

/// Exposure descriptions used in `/mcp` (`EXPOSURE_DESCRIPTIONS`).
pub fn exposure_description(exposure: McpExposure) -> &'static str {
	match exposure {
		McpExposure::Codemode => "called from codemode scripts, which find them with searchTools()",
		McpExposure::Deferred => "not declared until tool_search loads them, then called directly; no codemode needed",
		McpExposure::Direct => "declared to the model like built-in tools",
		McpExposure::Hidden => unreachable!("EXPOSURE_DESCRIPTIONS excludes hidden"),
	}
}

/// Tool exposure of an MCP exposure (re-exported from tools).
pub use crate::extensions::mcp::tools::to_tool_exposure;

pub fn is_enabled(entry: &McpServerEntry) -> bool {
	entry.config.get("enabled").and_then(|value| value.as_bool()).unwrap_or(true)
}

pub fn exposure_of(entry: &McpServerEntry) -> McpExposure {
	entry
		.config
		.get("exposure")
		.and_then(|value| value.as_str())
		.and_then(McpExposure::parse)
		.unwrap_or(McpExposure::Codemode)
}

/// Exposures the server's tools can have, known from its config before it
/// connects (`configuredExposures`).
pub fn configured_exposures(entry: &McpServerEntry) -> Vec<McpExposure> {
	let mut exposures = vec![exposure_of(entry)];
	if let Some(tool_exposure) = entry.config.get("toolExposure").and_then(|value| value.as_object()) {
		for exposure in tool_exposure.values() {
			if let Some(exposure) = exposure.as_str().and_then(McpExposure::parse) {
				exposures.push(exposure);
			}
		}
	}
	exposures
}

/// Whether some of the server's tools are declared to the model, so the first
/// prompt waits for them (`hasDirectTools`).
pub fn has_direct_tools(entry: &McpServerEntry) -> bool {
	configured_exposures(entry).contains(&McpExposure::Direct)
}

/// Whether some of the server's tools are reached through codemode or
/// tool_search (`hasIndirectTools`).
pub fn has_indirect_tools(entry: &McpServerEntry) -> bool {
	let exposures = configured_exposures(entry);
	exposures.contains(&McpExposure::Codemode) || exposures.contains(&McpExposure::Deferred)
}

/// The section's first line. It explains only the ways of reaching tools that
/// the listed servers use (`serversSectionIntro`).
fn servers_section_intro(reaches: &[&str]) -> String {
	let mut intro = "MCP servers whose tools are not declared to you.".to_string();
	if reaches.contains(&"codemode") {
		intro += " Call the tools of `codemode` servers from codemode scripts.";
	}
	if reaches.contains(&"tool_search") {
		intro += " Load the tools of `tool_search` servers with `tool_search`.";
	}
	intro
}

fn truncate(text: &str, max: usize) -> String {
	if text.chars().count() <= max {
		return text.to_string();
	}
	if max <= 1 {
		return String::new();
	}
	let mut cut: String = text.chars().take(max - 1).collect();
	while cut.ends_with(char::is_whitespace) {
		cut.pop();
	}
	format!("{cut}\u{2026}")
}

/// What the `mcp_servers` section needs of a server (`McpServerListing`).
#[derive(Debug, Clone, PartialEq)]
pub struct McpServerListing {
	pub entry: McpServerEntry,
	/// Server instructions from `initialize`, once connected.
	pub instructions: Option<String>,
}

/// First line of the configured description, or of the server instructions
/// once connected (`serverSummary`).
fn server_summary(server: &McpServerListing) -> String {
	let text = server
		.entry
		.config
		.get("description")
		.and_then(|value| value.as_str())
		.map(str::trim)
		.filter(|description| !description.is_empty())
		.map(str::to_string)
		.or_else(|| server.instructions.clone())
		.unwrap_or_default();
	text.lines().next().unwrap_or_default().trim().to_string()
}

/// The `mcp_servers` section: every enabled server with codemode or deferred
/// tools, with how its tools are reached and a one-line summary. The model
/// learns of the servers from it, since neither codemode nor tool_search lists
/// them. `None` when there are no such servers (`renderServersSection`).
pub fn render_servers_section(servers: &[McpServerListing]) -> Option<String> {
	let mut listed: Vec<&McpServerListing> = servers
		.iter()
		.filter(|server| is_enabled(&server.entry) && has_indirect_tools(&server.entry))
		.collect();
	listed.sort_by(|a, b| a.entry.name.to_lowercase().cmp(&b.entry.name.to_lowercase()));
	if listed.is_empty() {
		return None;
	}
	let reaches: Vec<&str> = listed
		.iter()
		.map(|server| {
			if configured_exposures(&server.entry).contains(&McpExposure::Codemode) {
				"codemode"
			} else {
				"tool_search"
			}
		})
		.collect();
	let intro = servers_section_intro(&reaches);
	let heads: Vec<String> = listed
		.iter()
		.enumerate()
		.map(|(index, server)| format!("- {} ({})", mcp_namespace(&server.entry.name), reaches[index]))
		.collect();
	let omitted = |count: usize| -> Vec<String> {
		if count > 0 {
			vec![format!(
				"- \u{2026} {} more server{}; find their tools with searchTools()",
				count,
				if count == 1 { "" } else { "s" }
			)]
		} else {
			Vec::new()
		}
	};
	// Characters of the intro, the first `kept` server lines without
	// descriptions, and the omission line.
	let size = |kept: usize| -> usize {
		let mut lines = vec![intro.clone()];
		lines.extend(heads.iter().take(kept).cloned());
		lines.extend(omitted(listed.len() - kept));
		lines.join("\n").chars().count()
	};
	let mut kept = listed.len();
	while kept > 0 && size(kept) > MAX_SERVERS_SECTION_CHARS {
		kept -= 1;
	}
	// Each description also takes a ": " separator.
	let per_server = if kept == 0 {
		0
	} else {
		MAX_SERVER_DESCRIPTION_CHARS.min(
			(MAX_SERVERS_SECTION_CHARS.saturating_sub(size(kept)) / kept).saturating_sub(2),
		)
	};
	let mut lines: Vec<String> = listed
		.iter()
		.take(kept)
		.enumerate()
		.map(|(index, server)| {
			let summary = if per_server > 0 { truncate(&server_summary(server), per_server) } else { String::new() };
			if summary.is_empty() {
				heads[index].clone()
			} else {
				format!("{}: {}", heads[index], summary)
			}
		})
		.collect();
	lines.insert(0, intro);
	lines.extend(omitted(listed.len() - kept));
	Some(lines.join("\n"))
}

/// Whether a codemode script needs the server: it names the server's
/// namespace, or searches, enumerates, or describes tools or namespaces, which
/// may name the server in other forms (`scriptNeedsServer`).
pub fn script_needs_server(code: &str, server: &str) -> bool {
	if code
		.split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
		.any(|word| matches!(word, "searchTools" | "describeNamespace" | "describeTool" | "ALL_TOOLS"))
	{
		return true;
	}
	code.contains(&mcp_namespace(server))
}

/// Short state for lists and the startup report (`describeState`).
pub fn describe_state(
	server: &McpServerListing,
	connection: Option<&crate::extensions::mcp::runtime::McpServerConnection>,
	with_error: bool,
) -> String {
	if !is_enabled(&server.entry) {
		return "disabled".to_string();
	}
	let Some(connection) = connection else {
		return "starting".to_string();
	};
	use crate::extensions::mcp::runtime::ServerState;
	match connection.state() {
		Some(ServerState::NeedsAuth) => "needs sign-in".to_string(),
		Some(ServerState::Failed) => {
			if with_error {
				format!(
					"failed: {}",
					connection
						.error()
						.unwrap_or_else(|| "unknown error".to_string())
						.lines()
						.next()
						.unwrap_or("unknown error")
						.to_string()
				)
			} else {
				"failed".to_string()
			}
		}
		Some(ServerState::Connected) => {
			let tools = connection.tools().len();
			let resources = connection.resources().len();
			let resource_count = if resources > 0 {
				format!(" \u{b7} {} resource{}", resources, if resources == 1 { "" } else { "s" })
			} else {
				String::new()
			};
			format!("connected \u{b7} {} tool{}{}", tools, if tools == 1 { "" } else { "s" }, resource_count)
		}
		Some(ServerState::Connecting) => "connecting\u{2026}".to_string(),
		state => state.map(|state| state.as_str().to_string()).unwrap_or_else(|| "starting".to_string()),
	}
}

/// Servers that need the user first (`attentionRank`).
pub fn attention_rank(server: &McpServerListing, connection: Option<&crate::extensions::mcp::runtime::McpServerConnection>) -> u8 {
	use crate::extensions::mcp::runtime::ServerState;
	if !is_enabled(&server.entry) {
		return 5;
	}
	match connection.and_then(|connection| connection.state()) {
		Some(ServerState::NeedsAuth) => 0,
		Some(ServerState::Failed) => 1,
		Some(ServerState::Disconnected) => 2,
		Some(ServerState::Connected) => 4,
		_ => 3,
	}
}

/// Transport of a server entry, for details lines (`describeTransport`).
pub fn describe_transport(entry: &McpServerEntry) -> String {
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

/// Tool names a server's exposure resolves to for one tool (used by the
/// factory and the CLI list).
pub fn tool_exposure_of(entry: &McpServerEntry, tool: &str) -> McpExposure {
	get_mcp_tool_exposure(&entry.config, tool)
}

/// Config-shape witness: a server entry with an `exposure` override carries
/// the same JSON shape `render_servers_section` reads.
#[allow(unused)]
fn _config_shape_witness(config: &Value) -> bool {
	config.get("exposure").is_some()
}

#[cfg(test)]
mod index_tests {
	//! Port of `test/mcp-extension.test.ts` `describe("MCP servers section")`.

	use super::*;

	fn server(name: &str, description: Option<&str>, exposure: Option<McpExposure>) -> McpServerListing {
		let mut config = serde_json::json!({ "command": "x" });
		if let Some(description) = description {
			config["description"] = serde_json::json!(description);
		}
		if let Some(exposure) = exposure {
			config["exposure"] = serde_json::json!(exposure.as_str());
		}
		McpServerListing {
			entry: McpServerEntry {
				name: name.to_string(),
				config,
				source: "test".to_string(),
				scope: None,
				r#override: None,
			},
			instructions: None,
		}
	}

	/// Port of "lists servers with how their tools are reached and the first
	/// line of their description".
	#[test]
	fn lists_servers_with_how_their_tools_are_reached_and_the_first_line_of_their_description() {
		let mut plain = server("plain", None, None);
		plain.instructions = Some("From instructions.".to_string());
		let listings = [
			server("docs", Some("Docs search.\nMore."), None),
			server("later", None, Some(McpExposure::Deferred)),
			server("direct", Some("Declared."), Some(McpExposure::Direct)),
			plain,
		];
		let section = render_servers_section(&listings).expect("section");
		let lines: Vec<&str> = section.lines().skip(1).collect();
		assert_eq!(
			lines,
			vec![
				"- mcp__docs (codemode): Docs search.",
				"- mcp__later (tool_search)",
				"- mcp__plain (codemode): From instructions.",
			]
		);
		let only = [server("direct", Some("Declared."), Some(McpExposure::Direct))];
		assert_eq!(render_servers_section(&only), None);
	}

	/// Port of "shortens descriptions to fit the size limit".
	#[test]
	fn shortens_descriptions_to_fit_the_size_limit() {
		let servers: Vec<McpServerListing> = (0..40)
			.map(|index| server(&format!("server{index}"), Some(&"x".repeat(400)), None))
			.collect();
		let section = render_servers_section(&servers).expect("section");
		assert!(section.chars().count() <= MAX_SERVERS_SECTION_CHARS);
		assert_eq!(section.lines().count(), 41);
		assert!(section.contains("- mcp__server39 (codemode): x"));
	}

	/// Port of "leaves out the last servers when their names alone do not fit".
	#[test]
	fn leaves_out_the_last_servers_when_their_names_alone_do_not_fit() {
		let servers: Vec<McpServerListing> = (0..200)
			.map(|index| server(&format!("server-with-a-long-name-{index}"), Some("desc"), None))
			.collect();
		let section = render_servers_section(&servers).expect("section");
		assert!(section.chars().count() <= MAX_SERVERS_SECTION_CHARS);
		let lines: Vec<&str> = section.lines().collect();
		let last = lines.last().expect("last line");
		assert!(
			last.starts_with("- \u{2026} ") && last.ends_with(" more servers; find their tools with searchTools()"),
			"unexpected last line: {last}"
		);
		let omitted: usize = last
			.split_whitespace()
			.find_map(|word| word.parse::<usize>().ok())
			.expect("omitted count");
		assert_eq!(lines.len() - 2 + omitted, 200);
	}

	/// Port of the `scriptNeedsServer` behaviors the `tool_call` handler
	/// relies on.
	#[test]
	fn script_needs_server_matches_namespaces_and_search_calls() {
		assert!(script_needs_server("await searchTools(\"docs\")", "docs"));
		assert!(script_needs_server("describeNamespace(\"mcp__docs\")", "docs"));
		assert!(script_needs_server("mcp__docs.search()", "docs"));
		assert!(!script_needs_server("mcp__other.search()", "docs"));
		assert!(script_needs_server("ALL_TOOLS", "anything"));
	}
}
