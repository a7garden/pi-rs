//! The native [`ExtensionFactory`] of the MCP extension (upstream
//! `createMcpExtension()` in `index.ts`; the JS module evaluation defers to the
//! D40 loader — D80).
//!
//! Registers the `mcp` command and the session lifetime handlers
//! (`session_start`, `before_agent_start`, `tool_call`, `turn_start`,
//! `mcp_servers_change`, `session_shutdown`) on the extension API: connecting
//! configured and extension-registered servers, activating codemode /
//! tool_search for indirect exposures, registering connection tools and the
//! resource tools, and reporting servers that need attention.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures_util::future::BoxFuture;
use parking_lot::Mutex;
use serde_json::{json, Value};

use pi_mcp::client::McpRequestOptions;

use crate::config::{get_agent_dir, CONFIG_DIR_NAME};
use crate::core::extensions::types::{
	ExtensionApi, ExtensionContext, ExtensionFactory, McpServerConfig, NotifyKind, RegisteredCommandOptions,
	RegisteredMcpServer, ToolDefinition, ToolInfo, ToolNamespace,
};
use crate::extensions::mcp::config::{
	get_mcp_tool_exposure, load_mcp_config, mcp_namespace, update_mcp_server_config, LoadedMcpConfig,
	McpExposure, McpServerConfigPatch, McpServerEntry, McpServerScope,
};
use crate::extensions::mcp::index::{
	configured_exposures, describe_state, exposure_of, has_direct_tools, is_enabled, script_needs_server,
	MCP_SERVERS_SECTION,
};
use crate::extensions::mcp::oauth::{
	sign_in_mcp_server, McpOAuthCredentialStore, McpSignInOptions, McpSignInPrompt,
};
use crate::extensions::mcp::resources::create_mcp_resource_tool_definitions;
use crate::extensions::mcp::runtime::{
	create_default_transport, McpCredentials, McpServerConnection, McpServerConnectionOptions, ServerState,
};
use crate::extensions::mcp::tools::{
	create_mcp_tool_definition, McpToolCaller, READ_MCP_RESOURCE_TOOL,
};

const DEFAULT_STARTUP_WAIT_MS: u64 = 10_000;

pub const RESOURCE_TOOL_NAMES: [&str; 3] = [
	"list_mcp_resources",
	"list_mcp_resource_templates",
	READ_MCP_RESOURCE_TOOL,
];

pub const CODEMODE_TOOL_NAME: &str = "codemode";
pub const TOOL_SEARCH_TOOL_NAME: &str = "tool_search";

/// Upstream `McpExtensionOptions`.
#[derive(Clone, Default)]
pub struct McpExtensionOptions {
	/// Defaults to reading `mcp.json` from the agent directory and the trusted
	/// project.
	pub load_config: Option<Arc<dyn Fn(&ExtensionContext) -> LoadedMcpConfig + Send + Sync>>,
	/// Defaults to stdio and streamable HTTP transports built from the server
	/// config.
	pub create_transport: Option<crate::extensions::mcp::runtime::McpTransportFactory>,
	/// Defaults to `mcp-auth.json` in the agent directory.
	pub credentials: Option<Arc<dyn McpCredentials>>,
	/// File server log messages are appended to. Defaults to `mcp.log` in the
	/// agent directory.
	pub log_path: Option<String>,
	/// Opens the OAuth authorization URL. Defaults to the platform browser.
	pub open_url: Option<Arc<dyn Fn(&str) + Send + Sync>>,
	/// Saves `/mcp` changes to the server's config file.
	pub update_config: Option<Arc<dyn Fn(&McpServerEntry, &McpServerConfigPatch) + Send + Sync>>,
	/// How long the first prompt waits for servers with `direct` tools that
	/// are still connecting at startup. Default: 10000.
	pub startup_wait_ms: Option<u64>,
}

/// One configured server. Disabled servers have no connection (upstream
/// `McpServer`).
#[derive(Clone)]
struct ExtensionServer {
	entry: McpServerEntry,
	connection: Option<Arc<McpServerConnection>>,
	/// For servers extensions registered: the config as registered, to detect
	/// re-registrations.
	registered_config: Option<String>,
	/// Result of the last `/mcp` action that failed, shown in the manager.
	message: Option<String>,
}

/// pi tool name → the `<server>\0<tool>` it was assigned to, so names stay
/// unique and stable (upstream `toolOwners`).
type ToolOwners = HashMap<String, String>;
/// Tool names currently offered by each server.
type ServerTools = HashMap<String, HashSet<String>>;
/// Last definition registered under each tool name, to re-register withdrawn
/// tools as hidden.
type Definitions = HashMap<String, ToolDefinition>;

/// Shared mutable extension state (upstream factory closure state).
struct McpExtensionState {
	options: McpExtensionOptions,
	inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
	servers: Vec<ExtensionServer>,
	/// Servers from `mcp.json`, which take precedence over registered servers
	/// of the same name.
	configured_entries: Vec<McpServerEntry>,
	config_errors: Vec<String>,
	/// The trusted project's `mcp.json`, where `/mcp` saves project overrides
	/// of global servers.
	project_config: Option<String>,
	/// Registered servers that `mcp.json` overrides, shown in `/mcp`.
	overridden: Vec<String>,
	/// Between session_start and session_shutdown.
	session_active: bool,
	auto_enable_codemode: bool,
	/// Whether the "codemode tools unreachable" warning was shown.
	warned_unreachable: bool,
	/// Whether a prompt already waited for the startup connections.
	waited_for_startup: bool,
	/// Bumped on every session start and shutdown so a late connection is
	/// dropped.
	generation: u64,
	/// Working directory of the session, for stdio servers.
	session_cwd: String,
	startup_wait_ms: u64,
	tool_owners: ToolOwners,
	server_tools: ServerTools,
	definitions: Definitions,
	/// Exposure the resource tools were last registered with.
	resource_tools_exposure: Option<McpExposure>,
}

/// Whether a registered tool is the codemode / tool_search tool. The two
/// sibling extension packages own the canonical predicates
/// (`isCodemodeTool`/`isToolSearchTool` upstream); matching by name keeps this
/// module decoupled from their internals (D80).
fn is_codemode_tool(tool: &ToolInfo) -> bool {
	tool.name == CODEMODE_TOOL_NAME
}

fn is_tool_search_tool(tool: &ToolInfo) -> bool {
	tool.name == TOOL_SEARCH_TOOL_NAME
}

/// The tool-call adapter of one connected server (`McpToolCaller` over
/// [`McpServerConnection`]).
struct ConnectionCaller(Arc<McpServerConnection>);

impl McpToolCaller for ConnectionCaller {
	fn call_tool(
		&self,
		name: &str,
		args: Value,
		options: McpRequestOptions,
	) -> BoxFuture<'static, Result<pi_mcp::protocol::content::CallToolResult, pi_mcp::protocol::jsonrpc::McpError>> {
		let connection = Arc::clone(&self.0);
		let name = name.to_string();
		Box::pin(async move {
			connection
				.call_tool(&name, args, options)
				.await
				.map_err(pi_mcp::protocol::jsonrpc::McpError::other)
		})
	}
}


/// `ui.notify(message, kind)` convenience (the Rust `ExtensionUi::notify`
/// takes an owned message and an `Option<NotifyKind>`).
fn notify(ui: &Arc<dyn crate::core::extensions::types::ExtensionUi>, message: impl std::fmt::Display, kind: NotifyKind) {
	ui.notify(message.to_string(), Some(kind));
}

/// The `mcp` extension factory (`createMcpExtension`). See the
/// [`module docs`](super) for file status and deviations.
pub fn create_mcp_extension(options: McpExtensionOptions) -> ExtensionFactory {
	let startup_wait_ms = options.startup_wait_ms.unwrap_or(DEFAULT_STARTUP_WAIT_MS);
	let state = Arc::new(McpExtensionState {
		options,
		inner: Mutex::new(Inner { startup_wait_ms, ..Default::default() }),
	});
	Arc::new(move |pi: Arc<dyn ExtensionApi>| {
		let state = Arc::clone(&state);
		Box::pin(async move { register(state, pi).await })
	})
}

fn default_load_config(state: &McpExtensionState, ctx: &ExtensionContext) -> LoadedMcpConfig {
	if let Some(load_config) = &state.options.load_config {
		return load_config(ctx);
	}
	load_mcp_config(&get_agent_dir(), &ctx.cwd, ctx.is_project_trusted())
}

fn credentials_of(state: &McpExtensionState) -> Arc<dyn McpCredentials> {
	state
		.options
		.credentials
		.clone()
		.unwrap_or_else(|| Arc::new(McpOAuthCredentialStore::open_default()))
}

fn log_of(state: &McpExtensionState) -> Arc<Mutex<crate::extensions::mcp::log::McpServerLog>> {
	Arc::new(Mutex::new(crate::extensions::mcp::log::McpServerLog::new(
		state
			.options
			.log_path
			.clone()
			.unwrap_or_else(|| format!("{}/mcp.log", get_agent_dir())),
	)))
}

fn default_update_config(entry: &McpServerEntry, patch: &McpServerConfigPatch) {
	if let Some(path) = entry.r#override.as_deref().or(Some(entry.source.as_str())) {
		let _ = update_mcp_server_config(path, &entry.name, *patch, entry.r#override.is_some());
	}
}

async fn register(state: Arc<McpExtensionState>, pi: Arc<dyn ExtensionApi>) -> Result<(), String> {
	// -- session_start ------------------------------------------------------

	pi.on(
		"session_start",
		{
			let state = Arc::clone(&state);
			let pi = Arc::clone(&pi);
			Arc::new(move |_event, ctx| {
				let state = Arc::clone(&state);
				let pi = Arc::clone(&pi);
				Box::pin(async move {
					start_session(&state, &pi, &ctx).await;
					Ok(None)
				})
			})
		},
	);

	// -- before_agent_start -------------------------------------------------

	pi.on(
		"before_agent_start",
		{
			let state = Arc::clone(&state);
			Arc::new(move |event, _ctx| {
				let state = Arc::clone(&state);
				Box::pin(async move {
					// The first prompt waits for servers whose tools are
					// declared to the model, but not indefinitely: a slow or
					// hanging server must not hold up the prompt
					// (`waitForDirectServers`). The `mcp_servers` section is
					// rendered by `render_servers_section`; the runner's
					// erased result cannot carry section mutations yet (D80).
					wait_for_direct_servers(&state).await;
					let _ = event;
					Ok(None)
				})
			})
		},
	);

	// -- tool_call ----------------------------------------------------------

	pi.on(
		"tool_call",
		{
			let state = Arc::clone(&state);
			Arc::new(move |event, ctx| {
				let state = Arc::clone(&state);
				Box::pin(async move {
					wait_for_needed_servers(&state, &event, ctx.signal.clone()).await;
					Ok(None)
				})
			})
		},
	);

	// -- turn_start ---------------------------------------------------------

	pi.on(
		"turn_start",
		{
			let state = Arc::clone(&state);
			Arc::new(move |_event, _ctx| {
				let state = Arc::clone(&state);
				Box::pin(async move {
					reconnect_signed_in(&state).await;
					Ok(None)
				})
			})
		},
	);

	// -- mcp_servers_change -------------------------------------------------

	pi.on(
		"mcp_servers_change",
		{
			let state = Arc::clone(&state);
			let pi = Arc::clone(&pi);
			Arc::new(move |_event, ctx| {
				let state = Arc::clone(&state);
				let pi = Arc::clone(&pi);
				Box::pin(async move {
					servers_changed(&state, &pi, &ctx).await;
					Ok(None)
				})
			})
		},
	);

	// -- session_shutdown ---------------------------------------------------

	pi.on(
		"session_shutdown",
		{
			let state = Arc::clone(&state);
			Arc::new(move |_event, _ctx| {
				let state = Arc::clone(&state);
				Box::pin(async move {
					shutdown(&state).await;
					Ok(None)
				})
			})
		},
	);

	// -- /mcp command -------------------------------------------------------

	pi.register_command(
		"mcp",
		RegisteredCommandOptions {
			description: Some(
				"Manage MCP servers: sign in, reconnect, enable or disable, and change exposure".to_string(),
			),
			// Argument completions feed the TUI autocomplete (opaque stand-in
			// in this port — D81); the command surface itself is complete.
			get_argument_completions: None,
			handler: {
				let state = Arc::clone(&state);
				Arc::new(move |args, ctx| {
					let state = Arc::clone(&state);
					Box::pin(async move { run_command(&state, &args, &ctx).await })
				})
			},
		},
	);
	Ok(())
}

fn connections(state: &McpExtensionState) -> Vec<Arc<McpServerConnection>> {
	state
		.inner
		.lock()
		.servers
		.iter()
		.filter_map(|server| server.connection.clone())
		.collect()
}

fn find_server(state: &McpExtensionState, name: &str) -> Option<ExtensionServer> {
	state
		.inner
		.lock()
		.servers
		.iter()
		.find(|server| server.entry.name == name)
		.cloned()
}

/// Servers extensions registered, except names `mcp.json` defines, which take
/// precedence (`registeredServers`).
fn registered_servers(state: &McpExtensionState, pi: &Arc<dyn ExtensionApi>) -> (Vec<ExtensionServer>, Vec<String>) {
	let configured_entries = state.inner.lock().configured_entries.clone();
	let mut registered = Vec::new();
	let mut overridden = Vec::new();
	for RegisteredMcpServer { name, config, extension_path } in pi.get_mcp_servers() {
		let configured = configured_entries
			.iter()
			.find(|entry| mcp_namespace(entry.name.as_str()) == mcp_namespace(name.as_str()))
			.cloned();
		if let Some(configured) = configured {
			overridden.push(format!(
				"\"{name}\" registered by {extension_path} is overridden by \"{}\" in {}",
				configured.name, configured.source
			));
			continue;
		}
		let registered_config = serde_json::to_string(&config.0).unwrap_or_default();
		registered.push(ExtensionServer {
			entry: McpServerEntry {
				name,
				config: config.0,
				source: extension_path,
				scope: Some(McpServerScope::Extension),
				r#override: None,
			},
			connection: None,
			registered_config: Some(registered_config),
			message: None,
		});
	}
	(registered, overridden)
}

/// Activate the codemode / tool_search tool the servers' exposures ask for
/// (`ensureDiscoveryActive`).
fn ensure_discovery_active(state: &McpExtensionState, pi: &Arc<dyn ExtensionApi>, ctx: &ExtensionContext) {
	let (needs_codemode, needs_tool_search, auto_enable, warned) = {
		let inner = state.inner.lock();
		let mut exposures: Vec<McpExposure> = Vec::new();
		for server in &inner.servers {
			if is_enabled(&server.entry) {
				exposures.extend(configured_exposures(&server.entry));
			}
		}
		(
			exposures.contains(&McpExposure::Codemode),
			exposures.contains(&McpExposure::Deferred),
			inner.auto_enable_codemode,
			inner.warned_unreachable,
		)
	};
	if !needs_codemode && !needs_tool_search {
		return;
	}
	// Other extensions' tools of the same names cannot reach MCP tools, so
	// never activate them.
	let tools = pi.get_all_tools();
	let has_codemode = tools.iter().any(is_codemode_tool);
	let has_tool_search = tools.iter().any(is_tool_search_tool);
	let active = pi.get_active_tools();
	let mut activate: Vec<String> = Vec::new();
	if needs_codemode && has_codemode && auto_enable && !active.iter().any(|name| name == CODEMODE_TOOL_NAME) {
		activate.push(CODEMODE_TOOL_NAME.to_string());
	}
	if needs_tool_search && has_tool_search && !active.iter().any(|name| name == TOOL_SEARCH_TOOL_NAME) {
		activate.push(TOOL_SEARCH_TOOL_NAME.to_string());
	}
	if !activate.is_empty() {
		let mut next = active.clone();
		next.extend(activate.iter().cloned());
		pi.set_active_tools(next);
	}
	let reachable_codemode = active
		.iter()
		.chain(activate.iter())
		.any(|name| name == CODEMODE_TOOL_NAME);
	let reachable_tool_search = active
		.iter()
		.chain(activate.iter())
		.any(|name| name == TOOL_SEARCH_TOOL_NAME);
	if (has_codemode && reachable_codemode) || (has_tool_search && reachable_tool_search) {
		return;
	}
	if warned {
		return;
	}
	state.inner.lock().warned_unreachable = true;
	let reason = if needs_codemode && has_codemode && !auto_enable {
		" (autoEnableCodemode is false)"
	} else {
		""
	};
	notify(
		&ctx.ui,
		format!(
			"MCP tools are only reachable from the codemode or tool_search tool, but neither is active{reason}; they cannot be called."
		),
		NotifyKind::Warning,
	);
}

/// Register the tools a connected server offers (`registerTools`).
fn register_tools(
	state: &McpExtensionState,
	pi: &Arc<dyn ExtensionApi>,
	connection: &Arc<McpServerConnection>,
) {
	let server = connection.name().to_string();
	let entry = find_server(state, &server)
		.map(|found| found.entry)
		.unwrap_or_else(|| connection.entry().clone());
	let description = entry
		.config
		.get("description")
		.and_then(|value| value.as_str())
		.map(str::trim)
		.filter(|description| !description.is_empty())
		.map(str::to_string);
	let namespace = ToolNamespace {
		name: mcp_namespace(&server),
		description,
		instructions: connection.instructions(),
	};
	let timeout_ms = connection.timeout_ms();
	let previous: HashSet<String> = state
		.inner
		.lock()
		.server_tools
		.get(&server)
		.cloned()
		.unwrap_or_default();
	let caller = Arc::new(ConnectionCaller(Arc::clone(connection)));
	// Like Codex, all tools whose names sanitize to the same name get the hash
	// suffix, so which one would keep the plain name does not depend on the
	// order of the list.
	let plain: Vec<String> = {
		let mut seen = HashSet::new();
		let mut names = Vec::new();
		for tool in connection.tools() {
			let name = crate::extensions::mcp::tools::create_mcp_tool_name(&server, &tool.name, |_| false);
			if seen.insert(name.clone()) {
				names.push(name);
			}
		}
		names
	};
	let mut current: HashSet<String> = HashSet::new();
	let mut definitions: Vec<ToolDefinition> = Vec::new();
	for tool in connection.tools() {
		let owner = format!("{server}\0{}", tool.name);
		let name = crate::extensions::mcp::tools::create_mcp_tool_name(&server, &tool.name, |candidate| {
			let inner = state.inner.lock();
			inner
				.tool_owners
				.get(candidate)
				.is_some_and(|existing| existing != &owner)
				|| current.contains(candidate)
				|| plain.iter().filter(|plain| **plain == candidate).count() > 1
		});
		let exposure = get_mcp_tool_exposure(&entry.config, &tool.name);
		let get_client: Arc<
			dyn Fn() -> BoxFuture<'static, Result<Arc<dyn McpToolCaller>, String>> + Send + Sync,
		> = {
			let caller = Arc::clone(&caller);
			Arc::new(move || {
				let caller = Arc::clone(&caller);
				Box::pin(async move { Ok(caller as Arc<dyn McpToolCaller>) })
			})
		};
		let definition = create_mcp_tool_definition(crate::extensions::mcp::tools::McpToolDefinitionOptions {
			server: server.clone(),
			tool,
			name: name.clone(),
			exposure,
			namespace: namespace.clone(),
			timeout_ms,
			get_client,
			readable_resources: Some({
				let connection = Arc::clone(connection);
				let hidden = exposure_of(&entry) == McpExposure::Hidden;
				Arc::new(move || connection.has_resources() && !hidden)
			}),
		});
		current.insert(name.clone());
		definitions.push(definition);
	}
	{
		let mut inner = state.inner.lock();
		for definition in &definitions {
			inner.definitions.insert(definition.name.clone(), definition.clone());
			inner.tool_owners.insert(definition.name.clone(), format!("{server}\0{}", definition.name));
		}
		inner.server_tools.insert(server.clone(), current);
	}
	for definition in definitions {
		pi.register_tool(definition);
	}
	// Tools cannot be unregistered, so tools the server dropped are
	// re-registered as hidden. When the server offers them again they are
	// registered with their configured exposure above.
	let withdrawn: Vec<ToolDefinition> = {
		let inner = state.inner.lock();
		previous
			.iter()
			.filter(|name| !inner.server_tools.get(&server).is_some_and(|current| current.contains(*name)))
			.filter_map(|name| inner.definitions.get(name).cloned())
			.collect()
	};
	for mut definition in withdrawn {
		definition.exposure = Some(crate::core::extensions::types::ToolExposure::Hidden);
		pi.register_tool(definition);
	}
	sync_resource_tools(state, pi);
}

/// Make a disabled server's tools unreachable (`hideTools`).
fn hide_tools(state: &McpExtensionState, pi: &Arc<dyn ExtensionApi>, server: &str) {
	let hidden: Vec<ToolDefinition> = {
		let mut inner = state.inner.lock();
		let names: Vec<String> = inner
			.server_tools
			.get(server)
			.cloned()
			.unwrap_or_default()
			.into_iter()
			.collect();
		let hidden = names
			.iter()
			.filter_map(|name| inner.definitions.get(name).cloned())
			.collect();
		inner.server_tools.insert(server.to_string(), HashSet::new());
		hidden
	};
	for mut definition in hidden {
		definition.exposure = Some(crate::core::extensions::types::ToolExposure::Hidden);
		pi.register_tool(definition);
	}
	sync_resource_tools(state, pi);
}

/// Enabled servers with resources whose exposure is not `hidden`
/// (`serversWithResources`).
fn servers_with_resources(state: &McpExtensionState) -> Vec<ExtensionServer> {
	state
		.inner
		.lock()
		.servers
		.iter()
		.filter(|server| {
			server
				.connection
				.as_ref()
				.map(|connection| connection.has_resources())
				.unwrap_or(false)
				&& is_enabled(&server.entry)
				&& exposure_of(&server.entry) != McpExposure::Hidden
		})
		.cloned()
		.collect()
}

/// Register the resource tools with the widest exposure of the servers they
/// reach (`syncResourceTools`).
fn sync_resource_tools(state: &McpExtensionState, pi: &Arc<dyn ExtensionApi>) {
	let servers = servers_with_resources(state);
	let exposures: HashSet<McpExposure> = servers.iter().map(|server| exposure_of(&server.entry)).collect();
	let exposure = [McpExposure::Direct, McpExposure::Codemode, McpExposure::Deferred]
		.into_iter()
		.find(|candidate| exposures.contains(candidate));
	let next = exposure.unwrap_or(McpExposure::Hidden);
	{
		let inner = state.inner.lock();
		if inner.resource_tools_exposure == Some(next)
			|| (inner.resource_tools_exposure.is_none() && next == McpExposure::Hidden)
		{
			return;
		}
	}
	let was_direct = state.inner.lock().resource_tools_exposure == Some(McpExposure::Direct);
	state.inner.lock().resource_tools_exposure = Some(next);
	let connection_servers: Vec<Arc<McpServerConnection>> = servers
		.iter()
		.filter_map(|server| server.connection.clone())
		.collect();
	let definitions = create_mcp_resource_tool_definitions(
		next,
		Arc::new(move || {
			connection_servers
				.iter()
				.map(|connection| {
					Arc::new(crate::extensions::mcp::runtime::ConnectionResourceServer(Arc::clone(connection)))
						as Arc<dyn crate::extensions::mcp::resources::McpResourceServer>
				})
				.collect::<Vec<_>>()
		}),
	);
	for definition in &definitions {
		pi.register_tool(definition.clone());
	}
	if was_direct {
		let names: HashSet<String> = definitions.iter().map(|definition| definition.name.clone()).collect();
		let active = pi.get_active_tools();
		pi.set_active_tools(active.into_iter().filter(|name| !names.contains(name)).collect());
	}
}

/// Create the server's connection (`createConnection`).
fn create_connection(
	state: &Arc<McpExtensionState>,
	pi: &Arc<dyn ExtensionApi>,
	server: &ExtensionServer,
) -> Arc<McpServerConnection> {
	let credentials = credentials_of(state);
	let log = log_of(state);
	let create_transport = state
		.options
		.create_transport
		.clone()
		.unwrap_or_else(|| Arc::new(create_default_transport));
	let on_tools = {
		let state = Arc::downgrade(state);
		let pi = Arc::downgrade(pi);
		Arc::new(move |connection: &Arc<McpServerConnection>| {
			let (Some(state), Some(pi)) = (state.upgrade(), pi.upgrade()) else {
				return;
			};
			register_tools(&state, &pi, connection);
		})
	};
	let on_change = {
		let pi_for_change = pi.clone();
		let _ = pi_for_change;
		None::<Arc<dyn Fn(&McpServerConnection) + Send + Sync>>
	};
	McpServerConnection::new(McpServerConnectionOptions {
		entry: server.entry.clone(),
		cwd: state.inner.lock().session_cwd.clone(),
		create_transport,
		credentials,
		provider_token: None,
		on_tools,
		on_change,
		log: Some(log),
	})
}

async fn start_session(
	state: &Arc<McpExtensionState>,
	pi: &Arc<dyn ExtensionApi>,
	ctx: &ExtensionContext,
) {
	let loaded = default_load_config(state, ctx);
	{
		let mut inner = state.inner.lock();
		inner.config_errors = loaded.errors.clone();
		inner.project_config = loaded.project_config.clone();
		inner.auto_enable_codemode = loaded.auto_enable_codemode.unwrap_or(true);
		inner.warned_unreachable = false;
		inner.waited_for_startup = false;
		inner.session_cwd = ctx.cwd.clone();
		inner.session_active = true;
		inner.generation += 1;
		inner.configured_entries = loaded.servers.clone();
		let (registered, overridden) = registered_servers(state, pi);
		inner.overridden = overridden;
		inner.servers = loaded
			.servers
			.iter()
			.map(|entry| ExtensionServer {
				entry: entry.clone(),
				connection: None,
				registered_config: None,
				message: None,
			})
			.chain(registered)
			.collect();
	}
	ensure_discovery_active(state, pi, ctx);
	let enabled: Vec<ExtensionServer> = {
		let inner = state.inner.lock();
		inner
			.servers
			.iter()
			.filter(|server| is_enabled(&server.entry))
			.cloned()
			.collect()
	};
	if enabled.is_empty() {
		report_problems(state, ctx, None);
		return;
	}
	// Connections start in the background; failures show in their state.
	for server in enabled {
		let connection = create_connection(state, pi, &server);
		{
			let mut inner = state.inner.lock();
			if let Some(slot) = inner
				.servers
				.iter_mut()
				.find(|candidate| candidate.entry.name == server.entry.name)
			{
				slot.connection = Some(Arc::clone(&connection));
			}
		}
		let connection = Arc::clone(&connection);
		tokio::spawn(async move {
			let _ = connection.get_client().await;
		});
	}
}

/// Wait for servers still connecting, until they settle or `signal` aborts
/// (`waitForServers`).
async fn wait_for_servers(state: &McpExtensionState, signal: Option<crate::utils::abort::AbortSignal>) {
	let pending: Vec<Arc<McpServerConnection>> = state
		.inner
		.lock()
		.servers
		.iter()
		.filter_map(|server| server.connection.clone())
		.filter(|connection| connection.state() != Some(ServerState::Connected))
		.collect();
	if pending.is_empty() {
		return;
	}
	let wait = async move {
		for connection in pending {
			let _ = connection.get_client().await;
		}
	};
	match signal {
		Some(signal) => {
			tokio::select! {
				() = wait => {}
				() = signal.aborted_future() => {}
			}
		}
		None => wait.await,
	}
}

/// Wait for direct servers until the startup budget elapses
/// (`waitForDirectServers`).
async fn wait_for_direct_servers(state: &McpExtensionState) {
	{
		let mut inner = state.inner.lock();
		if inner.waited_for_startup {
			return;
		}
		inner.waited_for_startup = true;
	}
	let wait_ms = state.inner.lock().startup_wait_ms;
	let pending: Vec<Arc<McpServerConnection>> = state
		.inner
		.lock()
		.servers
		.iter()
		.filter(|server| is_enabled(&server.entry) && has_direct_tools(&server.entry))
		.filter_map(|server| server.connection.clone())
		.filter(|connection| connection.state() != Some(ServerState::Connected))
		.collect();
	if pending.is_empty() {
		return;
	}
	let wait = async move {
		for connection in pending {
			let _ = connection.get_client().await;
		}
	};
	let _ = tokio::time::timeout(std::time::Duration::from_millis(wait_ms), wait).await;
}

/// A codemode script waits for the servers it names; tool_search and the
/// resource tools wait for all of them (`tool_call` handler).
async fn wait_for_needed_servers(
	state: &McpExtensionState,
	event: &Value,
	signal: Option<crate::utils::abort::AbortSignal>,
) {
	let tool_name = event
		.get("toolName")
		.or_else(|| event.get("tool_name"))
		.and_then(|value| value.as_str())
		.unwrap_or_default()
		.to_string();
	let pending: Vec<ExtensionServer> = {
		let inner = state.inner.lock();
		inner
			.servers
			.iter()
			.filter(|server| {
				is_enabled(&server.entry)
					&& server
						.connection
						.as_ref()
						.map(|connection| connection.state() != Some(ServerState::Connected))
						.unwrap_or(false)
			})
			.cloned()
			.collect()
	};
	if pending.is_empty() {
		return;
	}
	let input = event.get("input").cloned().unwrap_or(Value::Null);
	let waiting: Vec<ExtensionServer> = if tool_name == CODEMODE_TOOL_NAME {
		let code = input
			.get("code")
			.and_then(|value| value.as_str())
			.unwrap_or_default()
			.to_string();
		pending
			.into_iter()
			.filter(|server| script_needs_server(&code, &server.entry.name))
			.collect()
	} else if tool_name == TOOL_SEARCH_TOOL_NAME || RESOURCE_TOOL_NAMES.contains(&tool_name.as_str()) {
		pending
	} else {
		return;
	};
	if waiting.is_empty() {
		return;
	}
	let wait = async move {
		for server in waiting {
			if let Some(connection) = server.connection {
				let _ = connection.get_client().await;
			}
		}
	};
	match signal {
		Some(signal) => {
			tokio::select! {
				() = wait => {}
				() = signal.aborted_future() => {}
			}
		}
		None => wait.await,
	}
}

/// Pick up sign-ins done outside the session, such as `pi mcp login` run by
/// the agent (`reconnectSignedIn`).
async fn reconnect_signed_in(state: &McpExtensionState) {
	let credentials = credentials_of(state);
	let pending: Vec<Arc<McpServerConnection>> = {
		let inner = state.inner.lock();
		inner
			.servers
			.iter()
			.filter_map(|server| server.connection.clone())
			.filter(|connection| connection.state() == Some(ServerState::NeedsAuth))
			.collect()
	};
	for connection in pending {
		let Some(url) = connection.oauth_url() else {
			continue;
		};
		if credentials.tokens(connection.name(), &url).is_some() {
			let _ = connection.reconnect().await;
		}
	}
}

/// Servers registered or unregistered during the session connect or disconnect
/// right away (`mcp_servers_change` handler).
async fn servers_changed(state: &Arc<McpExtensionState>, pi: &Arc<dyn ExtensionApi>, ctx: &ExtensionContext) {
	if !state.inner.lock().session_active {
		return;
	}
	let generation = state.inner.lock().generation;
	let (registered, overridden) = registered_servers(state, pi);
	{
		let mut inner = state.inner.lock();
		inner.overridden = overridden;
		// Unregistered servers and re-registered ones with a new config are
		// dropped; the latter come back below.
		let registered_names: HashSet<String> = registered.iter().map(|server| server.entry.name.clone()).collect();
		let removed: Vec<String> = inner
			.servers
			.iter()
			.filter(|server| {
				server.entry.scope == Some(McpServerScope::Extension)
					&& !registered
						.iter()
						.any(|next| next.entry.name == server.entry.name && next.registered_config == server.registered_config)
			})
			.map(|server| server.entry.name.clone())
			.collect();
		if !removed.is_empty() {
			inner
				.servers
				.retain(|candidate| !removed.contains(&candidate.entry.name));
		}
		let _ = registered_names;
		drop(inner);
		for name in &removed {
			hide_tools(state, pi, name);
		}
		let mut inner = state.inner.lock();
		for server in registered {
			if !inner.servers.iter().any(|existing| existing.entry.name == server.entry.name) {
				inner.servers.push(server);
			}
		}
	}
	ensure_discovery_active(state, pi, ctx);
	let closing = connections(state);
	let connecting: Vec<ExtensionServer> = {
		let inner = state.inner.lock();
		inner
			.servers
			.iter()
			.filter(|server| is_enabled(&server.entry) && server.connection.is_none())
			.cloned()
			.collect()
	};
	for server in &connecting {
		let connection = create_connection(state, pi, server);
		let mut inner = state.inner.lock();
		if let Some(slot) = inner
			.servers
			.iter_mut()
			.find(|candidate| candidate.entry.name == server.entry.name)
		{
			slot.connection = Some(Arc::clone(&connection));
		}
	}
	for server in &connecting {
		if let Some(connection) = server.connection.as_ref() {
			let connection = Arc::clone(connection);
			tokio::spawn(async move {
				let _ = connection.get_client().await;
			});
		}
	}
	if state.inner.lock().generation != generation {
		for server in &connecting {
			if let Some(connection) = server.connection.as_ref() {
				connection.close().await;
			}
		}
		return;
	}
	let _ = closing;
	report_problems(state, ctx, Some(&connecting));
}

async fn shutdown(state: &Arc<McpExtensionState>) {
	let closing = connections(state);
	{
		let mut inner = state.inner.lock();
		inner.session_active = false;
		inner.generation += 1;
		inner.servers.clear();
	}
	for connection in closing {
		connection.close().await;
	}
}

/// One message for everything that needs the user after startup
/// (`reportProblems`).
fn report_problems(state: &McpExtensionState, ctx: &ExtensionContext, only: Option<&[ExtensionServer]>) {
	let mut lines: Vec<String> = Vec::new();
	{
		let inner = state.inner.lock();
		if only.is_none() {
			for error in &inner.config_errors {
				lines.push(format!("config: {error}"));
			}
		}
		for server in only.map(|only| only.to_vec()).unwrap_or_else(|| inner.servers.clone()) {
			let connection_state = server.connection.as_ref().and_then(|connection| connection.state());
			if matches!(connection_state, Some(ServerState::NeedsAuth) | Some(ServerState::Failed)) {
				lines.push(format!(
					"{}: {}",
					server.entry.name,
					describe_state(
						&crate::extensions::mcp::index::McpServerListing {
							entry: server.entry.clone(),
							instructions: server.connection.as_ref().and_then(|connection| connection.instructions()),
						},
						server.connection.as_deref(),
						true
					)
				));
			}
		}
	}
	if lines.is_empty() {
		return;
	}
	notify(
		&ctx.ui,
		format!(
			"MCP servers need attention:\n{}\nRun /mcp to fix.",
			lines.iter().map(|line| format!("  {line}")).collect::<Vec<_>>().join("\n")
		),
		NotifyKind::Warning,
	);
}

// ---------------------------------------------------------------------------
// `/mcp` command
// ---------------------------------------------------------------------------

const MCP_USAGE: &str =
	"Usage: /mcp, /mcp login [server], /mcp logout [server], /mcp reconnect [server]";

fn uses_oauth(server: &ExtensionServer) -> bool {
	server.connection.as_ref().and_then(|connection| connection.oauth_url()).is_some()
}

fn format_status(state: &McpExtensionState) -> String {
	let inner = state.inner.lock();
	if inner.servers.is_empty() && inner.config_errors.is_empty() && inner.overridden.is_empty() {
		return format!(
			"No MCP servers configured. Add them to {}/mcp.json or .pi/mcp.json.",
			get_agent_dir()
		);
	}
	let mut lines: Vec<String> = Vec::new();
	for server in &inner.servers {
		let name = &server.entry.name;
		let exposure = exposure_of(&server.entry);
		let connection = server.connection.as_ref();
		if connection.and_then(|connection| connection.state()) == Some(ServerState::NeedsAuth) {
			lines.push(format!("{name}: needs sign-in, run /mcp login {name} ({})", exposure.as_str()));
			continue;
		}
		let tools = match connection {
			Some(connection) if connection.state() == Some(ServerState::Connected) => {
				format!(", {} tools", connection.tools().len())
			}
			_ => String::new(),
		};
		let state = if !is_enabled(&server.entry) {
			"disabled".to_string()
		} else {
			match connection.and_then(|connection| connection.state()) {
				Some(ServerState::Disconnected) => "disconnected, reconnects on next call".to_string(),
				Some(state) => state.as_str().to_string(),
				None => "starting".to_string(),
			}
		};
		let error = connection
			.filter(|connection| connection.state() != Some(ServerState::Connected))
			.and_then(|connection| connection.error())
			.map(|error| format!("\n    {}", error.replace('\n', "\n    ")))
			.unwrap_or_default();
		lines.push(format!("{name}: {state}{tools} ({}){error}", exposure.as_str()));
	}
	for error in &inner.config_errors {
		lines.push(format!("config error: {error}"));
	}
	for line in &inner.overridden {
		lines.push(format!("overridden: {line}"));
	}
	lines.join("\n")
}

async fn run_command(
	state: &Arc<McpExtensionState>,
	args: &str,
	ctx: &crate::core::extensions::types::ExtensionCommandContext,
) {
	let parts: Vec<&str> = args.trim().split_whitespace().filter(|part| !part.is_empty()).collect();
	let Some(action) = parts.first().copied() else {
		// The TUI manager defers (D81); non-TUI prints the status.
		notify(&ctx.ui, format_status(state), NotifyKind::Info);
		return;
	};
	if parts.len() > 2 {
		notify(&ctx.ui, MCP_USAGE, NotifyKind::Warning);
		return;
	}
	match action {
		"login" => {
			let Some(server) = pick_server(state, parts.get(1).copied(), ctx, oauth_pick()).await else {
				return;
			};
			login_command(state, &server, ctx).await;
		}
		"logout" => {
			let Some(server) = pick_server(state, parts.get(1).copied(), ctx, oauth_pick()).await else {
				return;
			};
			let credentials = credentials_of(state);
			let removed = server
				.connection
				.as_ref()
				.and_then(|connection| connection.oauth_url())
				.map(|url| credentials.remove(&server.entry.name, &url))
				.unwrap_or(false);
			notify(
				&ctx.ui,
				if removed {
					format!("Signed out of MCP server \"{}\".", server.entry.name)
				} else {
					format!("No stored credentials for MCP server \"{}\".", server.entry.name)
				
				},
				NotifyKind::Info,
			);
		}
		"reconnect" => {
			let Some(server) = pick_server(
				state,
				parts.get(1).copied(),
				ctx,
				PickOptions {
					eligible: Box::new(|server| server.connection.is_some()),
					preferred: Box::new(|server| {
						matches!(
							server.connection.as_ref().and_then(|connection| connection.state()),
							Some(ServerState::Failed | ServerState::Disconnected)
						)
					}),
					none: "No enabled MCP server to reconnect.".to_string(),
				},
			)
			.await else {
				return;
			};
			let Some(connection) = server.connection.as_ref() else {
				notify(&ctx.ui, format!("MCP server \"{}\" is disabled.", server.entry.name), NotifyKind::Error);
				return;
			};
			match connection.reconnect().await {
				Ok(()) => notify(
					&ctx.ui,
					format!("Reconnected to MCP server \"{}\".", server.entry.name),
					NotifyKind::Info,
				),
				Err(error) => notify(&ctx.ui, error, NotifyKind::Error),
			}
		}
		_ => notify(&ctx.ui, MCP_USAGE, NotifyKind::Warning),
	}
}

struct PickOptions {
	eligible: Box<dyn Fn(&ExtensionServer) -> bool + Send + Sync>,
	preferred: Box<dyn Fn(&ExtensionServer) -> bool + Send + Sync>,
	none: String,
}

fn oauth_pick() -> PickOptions {
	PickOptions {
		eligible: Box::new(uses_oauth),
		preferred: Box::new(|server| {
			server.connection.as_ref().and_then(|connection| connection.state()) == Some(ServerState::NeedsAuth)
		}),
		none: "No enabled MCP server uses OAuth. Only HTTP servers without an Authorization header do.".to_string(),
	}
}

/// Resolve the server for a subcommand, asking when the name is omitted and
/// ambiguous (`pickServer`).
async fn pick_server(
	state: &Arc<McpExtensionState>,
	name: Option<&str>,
	ctx: &crate::core::extensions::types::ExtensionCommandContext,
	options: PickOptions,
) -> Option<ExtensionServer> {
	if let Some(name) = name {
		match find_server(state, name) {
			None => {
				notify(
					&ctx.ui,
					format!("No MCP server named \"{name}\"."),
					NotifyKind::Error,
				);
				None
			}
			Some(server) if !(options.eligible)(&server) => {
				notify(&ctx.ui, options.none, NotifyKind::Error);
				None
			}
			server => server,
		}
	} else {
		let candidates: Vec<ExtensionServer> = {
			let inner = state.inner.lock();
			inner
				.servers
				.iter()
				.filter(|server| (options.eligible)(server))
				.cloned()
				.collect()
		};
		if candidates.is_empty() {
			notify(&ctx.ui, options.none, NotifyKind::Info);
			return None;
		}
		if candidates.len() == 1 {
			return candidates.into_iter().next();
		}
		let preferred: Vec<ExtensionServer> = candidates
			.iter()
			.filter(|server| (options.preferred)(server))
			.cloned()
			.collect();
		if preferred.len() == 1 {
			return preferred.into_iter().next();
		}
		let names: Vec<String> = candidates.iter().map(|server| server.entry.name.clone()).collect();
		let choice = ctx.ctx.ui.select("MCP server".to_string(), names, None).await;
		candidates.into_iter().find(|server| Some(&server.entry.name) == choice.as_ref())
	}
}

async fn login_command(
	state: &Arc<McpExtensionState>,
	server: &ExtensionServer,
	ctx: &crate::core::extensions::types::ExtensionCommandContext,
) {
	let name = &server.entry.name;
	if !ctx.has_ui {
		notify(
		&ctx.ui,
		format!("Signing in to MCP server \"{name}\" requires interactive mode."),
		NotifyKind::Error,
	);
		return;
	}
	let Some(connection) = server.connection.as_ref() else {
		notify(&ctx.ui, format!("MCP server \"{name}\" does not use OAuth."), NotifyKind::Error);
		return;
	};
	let Some(url) = connection.oauth_url() else {
		notify(&ctx.ui, format!("MCP server \"{name}\" does not use OAuth."), NotifyKind::Error);
		return;
	};
	let open_url = state.options.open_url.clone().unwrap_or_else(|| {
		Arc::new(|url| {
			let _ = std::process::Command::new("open").arg(url).spawn();
		})
	});
	let failure = match sign_in_mcp_server(McpSignInOptions {
		server_url: url.clone(),
		store: credentials_of(state).for_server(name, &url),
		settings: connection.oauth_settings().unwrap_or_default(),
		challenge: connection.challenge(),
		prompt: McpSignInPrompt {
			show_authorization_url: Arc::new(move |authorization_url| {
				open_url(authorization_url.as_str());
			}),
			prompt_for_redirect_url: Arc::new(|signal| {
				Box::pin(async move {
					// The interactive paste prompt needs a terminal input
					// channel; without the TUI sign-in screen (D81) the
					// sign-in completes only through the browser callback.
					signal.cancelled().await;
					None
				})
			}),
		},
	})
	.await
	{
		Ok(()) => None,
		Err(error) if error.to_string() == "Sign-in cancelled" => Some("Sign-in cancelled.".to_string()),
		Err(error) => Some(format!("Sign-in failed: {error}")),
	};
	if let Some(failure) = failure {
		notify(&ctx.ui, failure, NotifyKind::Error);
		return;
	}
	// The challenge that asked for this sign-in is answered.
	if let Err(error) = connection.after_sign_in().await {
		notify(&ctx.ui, format!("Signed in, but {error}"), NotifyKind::Error);
	}
}

/// Config patch applied by `/mcp` actions (kept for the save path of
/// `setEnabled`/`setExposure`, which the TUI manager drives — D81).
#[allow(unused)]
fn save_config_patch(enabled: bool) -> McpServerConfigPatch {
	McpServerConfigPatch { enabled: Some(enabled), exposure: None }
}

#[allow(unused)]
fn _witnesses() {
	let _: fn() -> McpServerConfig = || McpServerConfig(serde_json::json!({}));
	let _: fn(&str, &str) -> String = |cwd, entry| format!("{cwd}/{CONFIG_DIR_NAME}/{entry}");
	let _ = json!({});
}
