//! Port of `experimental/client-runtime.ts` — opens the live server/session
//! service namespaces backing one experimental (non-interactive) presentation.
//!
//! Upstream surfaces map as follows: `ClientRuntimeRoute` becomes
//! [`ClientRuntimeRoute`], `ClientRuntimeServer`/`ActivatedClientRuntimeServer`
//! become the flat [`ClientRuntimeServer`]/[`ActivatedClientRuntimeServer`]
//! structs, and the `services.use(Service)` lookups become the typed facades
//! ([`SessionDirectoryFacade`], [`RemoteSessionManagement`],
//! [`BuiltinSessionManagement`], [`PresentationPluginsFacade`],
//! [`ModelsFacade`], [`AgentControllerFacade`], [`TranscriptFacade`]) over
//! chord's [`pi_chord::types::ServiceClientHandle`].
//!
//! Deviations:
//! - **D181 — facade seams.** Upstream interface extension
//!   (`ActivatedClientRuntimeServer extends ClientRuntimeServer`) becomes a
//!   flat struct; opened namespaces are kept alive by holding their
//!   `Box<dyn RemoteServices>` (upstream garbage collection) and disposed by
//!   dropping them. `directory.state.value!` hydration assertions become
//!   errors, and the reconnect/clients dispose steps (infallible in Rust)
//!   contribute no failures to the upstream `Promise.allSettled` collection.
//! - **D151 adjustment** — `AuthInput` still lives in
//!   [`crate::experimental::radius_auth`]; `ClientCommand` comes from
//!   [`crate::experimental::commands`] (D177).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures_util::future::join_all;
use pi_chord::context::{Context, BACKGROUND_CONTEXT};
use pi_chord::services::errors::{error, AnyError, AggregateError};
use pi_chord::types::{RemoteServiceSourceOpenOptions, RemoteServices, ServiceClientHandle};
use pi_chord::JsonValue;
use pi_client::unix::{
	create_unix_transport_factory, discover_unix_servers, DiscoverUnixServersOptions, UnixServerRoute,
	UnixTransportOptions,
};
use pi_client::{connect_client, Client, ClientOptions, ServerError};

use crate::experimental::commands::{ClientCommand, TransportAddress};
use crate::experimental::radius_auth::RadiusRelayAuthResolver;
use crate::experimental::radius_relay::{
	create_radius_client_transport_factory, RadiusClientReconnect, RadiusClientTransportOptions,
};
use crate::experimental::server::{
	activate_server, resolve_server_directory, resolve_session_directory, ActivateServerOptions, ENV_SERVER_ID,
};
use crate::experimental::services::agent_controller::{
	members as agent_members, AgentOperationResponse, AgentPromptRequest, AgentPromptResult,
	AGENT_CONTROLLER,
};
use crate::experimental::services::connection::{
	create_server_service_source, create_session_service_source, ServerServiceSource, ServiceSourceOptions,
	SessionServiceSource,
};
use crate::experimental::services::models::{ModelsState, MODELS};
use crate::experimental::services::models_provider::members as models_members;
use crate::experimental::services::plugins::{
	members as plugin_members, PrepareSessionRequest, PRESENTATION_PLUGINS,
};
use crate::experimental::services::sessions::{
	SessionCreateOptions, SessionDirectoryState, SessionSummary, SESSION_DIRECTORY, SESSION_MANAGEMENT,
};
use crate::experimental::services::transcript::TRANSCRIPT;

/// Upstream `ClientRuntimeRoute`:
/// `{ transport: "unix" } & UnixServerRoute | { transport: "radius", serverId }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientRuntimeRoute {
	Unix(UnixServerRoute),
	Radius { server_id: String },
}

impl ClientRuntimeRoute {
	pub fn server_id(&self) -> &str {
		match self {
			ClientRuntimeRoute::Unix(route) => &route.server_id,
			ClientRuntimeRoute::Radius { server_id } => server_id,
		}
	}

	/// The Unix socket path, present only for the `unix` transport.
	pub fn unix_path(&self) -> Option<&str> {
		match self {
			ClientRuntimeRoute::Unix(route) => Some(&route.path),
			ClientRuntimeRoute::Radius { .. } => None,
		}
	}
}

/// Upstream `ClientRuntimeServer`.
#[derive(Clone)]
pub struct ClientRuntimeServer {
	pub route: ClientRuntimeRoute,
	pub client: Client,
	pub server: Arc<dyn ServerServiceSource>,
	pub session: Arc<dyn SessionServiceSource>,
}

/// Upstream `ClientRuntime`: the opened routes plus grouped disposal.
pub struct ClientRuntime {
	pub servers: Vec<ClientRuntimeServer>,
	reconnectors: parking_lot::Mutex<Vec<RadiusClientReconnect>>,
	server_sources: parking_lot::Mutex<Vec<Arc<dyn ServerServiceSource>>>,
	session_sources: parking_lot::Mutex<Vec<Arc<dyn SessionServiceSource>>>,
	clients: parking_lot::Mutex<Vec<Client>>,
	disposed: AtomicBool,
}

impl ClientRuntime {
	/// Upstream `runtime.dispose()` — dispose reconnectors, service sources,
	/// and clients (all concurrently), aggregating the collected failures.
	pub async fn dispose(&self) -> Result<(), AnyError> {
		if self.disposed.swap(true, Ordering::SeqCst) {
			return Ok(());
		}
		let reconnectors = std::mem::take(&mut *self.reconnectors.lock());
		let server_sources = std::mem::take(&mut *self.server_sources.lock());
		let session_sources = std::mem::take(&mut *self.session_sources.lock());
		let clients = std::mem::take(&mut *self.clients.lock());
		// Reconnector and client disposal is infallible here (D181); only the
		// service sources contribute to the settled failure list.
		join_all(reconnectors.iter().map(|reconnector| reconnector.dispose())).await;
		let _ = join_all(clients.iter().map(|client| client.dispose())).await;
		let source_results = join_all(
			server_sources
				.iter()
				.map(|source| source.dispose(&BACKGROUND_CONTEXT))
				.chain(session_sources.iter().map(|source| source.dispose(&BACKGROUND_CONTEXT))),
		)
		.await;
		let errors: Vec<AnyError> = source_results
			.into_iter()
			.filter_map(|result| match result {
				Ok(()) => None,
				Err(dispose_error) => Some(dispose_error),
			})
			.collect();
		match errors.len() {
			0 => Ok(()),
			1 => Err(errors.into_iter().next().expect("one error")),
			_ => Err(AggregateError::new("Failed to dispose experimental client runtime", errors)),
		}
	}
}

/// Upstream `OpenClientRuntimeOptions`.
#[derive(Debug, Clone, Default)]
pub struct OpenClientRuntimeOptions {
	/// Directory searched when `--connect` is omitted. Defaults to
	/// `PI_SERVER_DIR` or `~/.pi/server`.
	pub directory: Option<String>,
}

fn noop_assert_access() -> Arc<dyn Fn() -> Result<(), AnyError> + Send + Sync> {
	Arc::new(|| Ok(()))
}

fn noop_on_error() -> Arc<dyn Fn(AnyError) + Send + Sync> {
	Arc::new(|_: AnyError| {})
}

/// One remote `SessionManagement` handle (upstream `use(SessionManagement)`).
#[derive(Clone)]
pub struct RemoteSessionManagement {
	handle: Arc<dyn ServiceClientHandle>,
}

impl RemoteSessionManagement {
	pub(crate) fn new(handle: Arc<dyn ServiceClientHandle>) -> Self {
		Self { handle }
	}

	async fn invoke(&self, member: &str, args: Vec<JsonValue>, context: &Context) -> Result<Option<JsonValue>, AnyError> {
		self.handle.invoke(member, args, context).await
	}

	/// Upstream `management.create(options, context)`.
	pub async fn create(
		&self,
		options: SessionCreateOptions,
		context: &Context,
	) -> Result<SessionSummary, AnyError> {
		let request = serde_json::to_value(options).map_err(|serde_error| error(serde_error.to_string()))?;
		let result = self.invoke("create", vec![request], context).await?;
		let value = result.ok_or_else(|| error("create did not return a Session summary"))?;
		serde_json::from_value(value).map_err(|serde_error| error(serde_error.to_string()))
	}

	/// Upstream `management.remove(sessionId, context)`.
	pub async fn remove(&self, session_id: &str, context: &Context) -> Result<(), AnyError> {
		self.invoke("remove", vec![JsonValue::String(session_id.to_string())], context)
			.await?;
		Ok(())
	}

	/// Upstream `management.attach(sessionId, context)`.
	pub async fn attach(&self, session_id: &str, context: &Context) -> Result<(), AnyError> {
		self.invoke("attach", vec![JsonValue::String(session_id.to_string())], context)
			.await?;
		Ok(())
	}

	/// Upstream `management.detach(context)`.
	pub async fn detach(&self, context: &Context) -> Result<(), AnyError> {
		self.invoke("detach", Vec::new(), context).await?;
		Ok(())
	}
}

/// Upstream's inline `management` wrapper in `activateBuiltinClientServices`:
/// remote management plus the Session attachment waits.
pub struct BuiltinSessionManagement {
	remote: RemoteSessionManagement,
	client: Client,
	session: Arc<dyn SessionServiceSource>,
}

impl BuiltinSessionManagement {
	pub fn remote(&self) -> &RemoteSessionManagement {
		&self.remote
	}

	/// `create` delegates straight to the remote management.
	pub async fn create(
		&self,
		options: SessionCreateOptions,
		context: &Context,
	) -> Result<SessionSummary, AnyError> {
		self.remote.create(options, context).await
	}

	pub async fn remove(&self, session_id: &str, context: &Context) -> Result<(), AnyError> {
		let removes_current_attachment = self
			.client
			.attachment()
			.map(|attachment: pi_client::types::SessionTarget| attachment.session_id)
			.as_deref()
			== Some(session_id);
		self.remote.remove(session_id, context).await?;
		if removes_current_attachment {
			self.session.when_detached(context).await?;
		}
		Ok(())
	}

	pub async fn attach(&self, session_id: &str, context: &Context) -> Result<(), AnyError> {
		self.remote.attach(session_id, context).await?;
		self.session.when_attached(session_id, context).await
	}

	pub async fn detach(&self, context: &Context) -> Result<(), AnyError> {
		self.remote.detach(context).await?;
		self.session.when_detached(context).await
	}
}

/// Client-side `use(SessionDirectory)`.
pub struct SessionDirectoryFacade {
	handle: Arc<dyn ServiceClientHandle>,
}

impl SessionDirectoryFacade {
	pub(crate) fn new(handle: Arc<dyn ServiceClientHandle>) -> Self {
		Self { handle }
	}

	/// Upstream `directory.state.value!`.
	pub fn state(&self) -> Result<SessionDirectoryState, AnyError> {
		let state = self
			.handle
			.state("state")?
			.ok_or_else(|| error("Session directory state is missing"))?;
		let value = state.value().ok_or_else(|| error("Session directory state is not hydrated"))?;
		serde_json::from_value(value.clone()).map_err(|serde_error| error(serde_error.to_string()))
	}
}

/// Client-side `use(PresentationPlugins)`.
pub struct PresentationPluginsFacade {
	handle: Arc<dyn ServiceClientHandle>,
}

impl PresentationPluginsFacade {
	pub(crate) fn new(handle: Arc<dyn ServiceClientHandle>) -> Self {
		Self { handle }
	}

	/// Upstream `plugins.prepareSession(request, context)`.
	pub async fn prepare_session(
		&self,
		request: PrepareSessionRequest,
		context: &Context,
	) -> Result<(), AnyError> {
		let request = serde_json::to_value(request).map_err(|serde_error| error(serde_error.to_string()))?;
		self.handle.invoke(plugin_members::PREPARE_SESSION, vec![request], context).await?;
		Ok(())
	}
}

/// Client-side `use(Models)`.
pub struct ModelsFacade {
	handle: Arc<dyn ServiceClientHandle>,
}

impl ModelsFacade {
	pub(crate) fn new(handle: Arc<dyn ServiceClientHandle>) -> Self {
		Self { handle }
	}

	/// Upstream `models.state.value!`.
	pub fn state(&self) -> Result<ModelsState, AnyError> {
		let state = self.handle.state(models_members::STATE)?.ok_or_else(|| error("Models state is missing"))?;
		let value = state.value().ok_or_else(|| error("Models state is not hydrated"))?;
		serde_json::from_value(value.clone()).map_err(|serde_error| error(serde_error.to_string()))
	}
}

/// Client-side `use(AgentController)`.
pub struct AgentControllerFacade {
	handle: Arc<dyn ServiceClientHandle>,
}

impl AgentControllerFacade {
	pub(crate) fn new(handle: Arc<dyn ServiceClientHandle>) -> Self {
		Self { handle }
	}

	/// Upstream `agent.prompt(request, context)`.
	pub async fn prompt(
		&self,
		request: AgentPromptRequest,
		context: &Context,
	) -> Result<AgentOperationResponse, AnyError> {
		let request = serde_json::to_value(request).map_err(|serde_error| error(serde_error.to_string()))?;
		let result = self.handle.invoke(agent_members::PROMPT, vec![request], context).await?;
		let value = result.ok_or_else(|| error("prompt did not return a response"))?;
		serde_json::from_value(value).map_err(|serde_error| error(serde_error.to_string()))
	}

	/// Upstream `agent.waitForPrompt(operationId, context)`.
	pub async fn wait_for_prompt(
		&self,
		operation_id: &str,
		context: &Context,
	) -> Result<AgentPromptResult, AnyError> {
		let result = self
			.handle
			.invoke(
				agent_members::WAIT_FOR_PROMPT,
				vec![JsonValue::String(operation_id.to_string())],
				context,
			)
			.await?;
		let value = result.ok_or_else(|| error("waitForPrompt did not return a result"))?;
		serde_json::from_value(value).map_err(|serde_error| error(serde_error.to_string()))
	}
}

/// Client-side `use(Transcript)`.
pub struct TranscriptFacade {
	handle: Arc<dyn ServiceClientHandle>,
}

impl TranscriptFacade {
	pub(crate) fn new(handle: Arc<dyn ServiceClientHandle>) -> Self {
		Self { handle }
	}

	/// The raw `Transcript` service handle (the durable view state is typed at
	/// the worker seam, D144).
	pub fn handle(&self) -> &Arc<dyn ServiceClientHandle> {
		&self.handle
	}
}

/// Upstream `ActivatedClientRuntimeServer` (flattened, D181).
pub struct ActivatedClientRuntimeServer {
	pub route: ClientRuntimeRoute,
	pub client: Client,
	pub server: Arc<dyn ServerServiceSource>,
	pub session: Arc<dyn SessionServiceSource>,
	pub directory: SessionDirectoryFacade,
	pub management: BuiltinSessionManagement,
	pub plugins: PresentationPluginsFacade,
	pub models: ModelsFacade,
	pub agent: AgentControllerFacade,
	pub transcript: TranscriptFacade,
	// Keeps the opened namespaces alive (upstream garbage collection, D181).
	_server_services: Box<dyn RemoteServices>,
	_session_services: Box<dyn RemoteServices>,
}

/// `routeFromExplicitPath(path)`: derive `<uuidv4-server-id>.sock` routes.
pub fn route_from_explicit_path(path: &str) -> Result<UnixServerRoute, AnyError> {
	let name = std::path::Path::new(path)
		.file_name()
		.and_then(|name| name.to_str())
		.unwrap_or_default();
	let server_id = name.strip_suffix(".sock").unwrap_or_default();
	if !pi_protocol::is_server_id(server_id) {
		return Err(error("--connect path must end with <uuidv4-server-id>.sock"));
	}
	Ok(UnixServerRoute { server_id: server_id.to_string(), path: path.to_string() })
}

async fn connect_route_client(
	route: &ClientRuntimeRoute,
	command: &ClientCommand,
) -> Result<Client, AnyError> {
	let transport_factory = match route {
		ClientRuntimeRoute::Unix(unix) => create_unix_transport_factory(UnixTransportOptions {
			path: unix.path.clone(),
			max_pending_bytes: None,
		})?,
		ClientRuntimeRoute::Radius { server_id } => create_radius_client_transport_factory(
			RadiusClientTransportOptions {
				server_id: server_id.clone(),
				auth: Arc::new(RadiusRelayAuthResolver::new(command.auth.clone())),
				web_socket_factory: None,
			},
		),
	};
	connect_client(ClientOptions {
		transport_factory,
		server_id: route.server_id().to_string(),
		max_frame_length: None,
		on_listener_error: None,
	})
	.await
}

/// Open one route: connect (or reuse the auto-activated client), publish the
/// service sources, and install the Radius reattach hook.
async fn open_route(
	runtime: &mut ClientRuntime,
	route: ClientRuntimeRoute,
	command: &ClientCommand,
	directory: &str,
	activated_client: &mut Option<Client>,
) -> Result<(), AnyError> {
	let client = match activated_client.take() {
		Some(client) => client,
		None => match connect_route_client(&route, command).await {
			Ok(client) => client,
			Err(connect_error) => {
				let recoverable = command.connect.is_none()
					&& matches!(route, ClientRuntimeRoute::Unix(_))
					&& connect_error
						.downcast_ref::<ServerError>()
						.is_some_and(|server_error| server_error.code == "version");
				if !recoverable {
					return Err(connect_error);
				}
				// Stale socket: activate a fresh server for the same ID
				// (upstream `activateServer` fallback on `ServerError code "version"`).
				let activated = activate_server(&ActivateServerOptions {
					directory: directory.to_string(),
					requested_server_id: Some(route.server_id().to_string()),
					session_dir: resolve_session_directory(None),
					provider: None,
					model: None,
				})
				.await?;
				activated.client
			}
		},
	};
	runtime.clients.lock().push(client.clone());
	let server = create_server_service_source(client.clone(), ServiceSourceOptions::default());
	runtime.server_sources.lock().push(Arc::clone(&server));
	if let ClientRuntimeRoute::Radius { .. } = &route {
		let reconnect_services = Arc::new(
			server.open(RemoteServiceSourceOpenOptions {
				services: vec![SESSION_MANAGEMENT.id().to_string()],
				assert_access: noop_assert_access(),
				on_error: noop_on_error(),
			})
			.await?,
		);
		let reconnect_management =
			RemoteSessionManagement::new(reconnect_services.use_service(SESSION_MANAGEMENT.id())?);
		let reconnect = RadiusClientReconnect::new(
			client.clone(),
			Arc::new(move |session_id: String| {
				let reconnect_services = Arc::clone(&reconnect_services);
				let reconnect_management = reconnect_management.clone();
				Box::pin(async move {
					reconnect_services.ready(&BACKGROUND_CONTEXT).await?;
					reconnect_management.attach(&session_id, &BACKGROUND_CONTEXT).await
				})
			}),
		)?;
		runtime.reconnectors.lock().push(reconnect);
	}
	let session = create_session_service_source(client.clone(), ServiceSourceOptions::default());
	runtime.session_sources.lock().push(Arc::clone(&session));
	runtime
		.servers
		.push(ClientRuntimeServer { route, client, server, session });
	Ok(())
}

/// Upstream `openClientRuntime(command, options)` — open live server/session
/// service namespaces for one experimental presentation.
pub async fn open_client_runtime(
	command: &ClientCommand,
	options: OpenClientRuntimeOptions,
) -> Result<ClientRuntime, AnyError> {
	if command.auth.is_some() && !matches!(command.connect, Some(TransportAddress::Radius { .. })) {
		return Err(error("Authentication is only supported for experimental Radius connections"));
	}
	if command.provider.is_some() && command.model.is_none() {
		return Err(error("Server model provider requires a model"));
	}
	if command.connect.is_some() && command.model.is_some() {
		return Err(error("Model selection is only valid when automatically activating a new server"));
	}
	if matches!(command.connect, Some(TransportAddress::Radius { .. })) && command.plugin_packages.is_some() {
		return Err(error("Plugin package paths can only be configured on a local Unix server"));
	}
	let directory = resolve_server_directory(options.directory.as_deref());
	let mut activated_client: Option<Client> = None;
	let routes: Vec<ClientRuntimeRoute> = match &command.connect {
		Some(TransportAddress::Radius { server_id }) => vec![ClientRuntimeRoute::Radius { server_id: server_id.clone() }],
		Some(TransportAddress::Unix { path }) => vec![ClientRuntimeRoute::Unix(route_from_explicit_path(path)?)],
		None => {
			let discovered = discover_unix_servers(DiscoverUnixServersOptions {
				directory: directory.clone(),
				timeout_ms: None,
			})
			.await?;
			let mut routes: Vec<ClientRuntimeRoute> = discovered.into_iter().map(ClientRuntimeRoute::Unix).collect();
			if !routes.is_empty() && command.model.is_some() {
				return Err(error("Model selection is only valid when automatically activating a new server"));
			}
			if routes.is_empty() {
				let activated = activate_server(&ActivateServerOptions {
					directory: directory.clone(),
					requested_server_id: std::env::var(ENV_SERVER_ID).ok(),
					session_dir: resolve_session_directory(None),
					provider: command.provider.clone(),
					model: command.model.clone(),
				})
				.await?;
				routes = vec![ClientRuntimeRoute::Unix(activated.route)];
				activated_client = Some(activated.client);
			}
			routes
		}
	};
	if command.plugin_packages.is_some() && routes.len() != 1 {
		return Err(error("Plugin selection requires exactly one local server"));
	}

	let mut runtime = ClientRuntime {
		servers: Vec::new(),
		reconnectors: parking_lot::Mutex::new(Vec::new()),
		server_sources: parking_lot::Mutex::new(Vec::new()),
		session_sources: parking_lot::Mutex::new(Vec::new()),
		clients: parking_lot::Mutex::new(Vec::new()),
		disposed: AtomicBool::new(false),
	};
	let mut start_error: Option<AnyError> = None;
	for route in routes {
		if let Err(route_error) =
			open_route(&mut runtime, route, command, &directory, &mut activated_client).await
		{
			start_error = Some(route_error);
			break;
		}
	}
	if let Some(start_error) = start_error {
		// Upstream: dispose, and wrap both failures in an AggregateError when
		// the cleanup itself fails too.
		return match runtime.dispose().await {
			Ok(()) => Err(start_error),
			Err(cleanup_error) => Err(AggregateError::new(
				"Experimental client startup and cleanup failed",
				vec![start_error, cleanup_error],
			)),
		};
	}
	Ok(runtime)
}

/// Upstream `activateBuiltinClientServices(server)` — acquire and connect the
/// built-in service facades used by the non-interactive client.
pub async fn activate_builtin_client_services(
	server: ClientRuntimeServer,
) -> Result<ActivatedClientRuntimeServer, AnyError> {
	let server_services = server.server.open(RemoteServiceSourceOpenOptions {
		services: vec![
			SESSION_DIRECTORY.id().to_string(),
			SESSION_MANAGEMENT.id().to_string(),
			PRESENTATION_PLUGINS.id().to_string(),
		],
		assert_access: noop_assert_access(),
		on_error: noop_on_error(),
	})
	.await?;
	let session_services = server.session.open(RemoteServiceSourceOpenOptions {
		services: vec![
			MODELS.id().to_string(),
			AGENT_CONTROLLER.id().to_string(),
			TRANSCRIPT.id().to_string(),
		],
		assert_access: noop_assert_access(),
		on_error: noop_on_error(),
	})
	.await?;
	let directory = SessionDirectoryFacade::new(server_services.use_service(SESSION_DIRECTORY.id())?);
	let remote_management = RemoteSessionManagement::new(server_services.use_service(SESSION_MANAGEMENT.id())?);
	let management = BuiltinSessionManagement {
		remote: remote_management,
		client: server.client.clone(),
		session: Arc::clone(&server.session),
	};
	let plugins = PresentationPluginsFacade::new(server_services.use_service(PRESENTATION_PLUGINS.id())?);
	let models = ModelsFacade::new(session_services.use_service(MODELS.id())?);
	let agent = AgentControllerFacade::new(session_services.use_service(AGENT_CONTROLLER.id())?);
	let transcript = TranscriptFacade::new(session_services.use_service(TRANSCRIPT.id())?);
	tokio::try_join!(
		server_services.ready(&BACKGROUND_CONTEXT),
		session_services.ready(&BACKGROUND_CONTEXT)
	)?;
	Ok(ActivatedClientRuntimeServer {
		route: server.route,
		client: server.client,
		server: server.server,
		session: server.session,
		directory,
		management,
		plugins,
		models,
		agent,
		transcript,
		_server_services: server_services,
		_session_services: session_services,
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	use crate::experimental::radius_auth::AuthInput;

	/// `Result::unwrap_err` without a `Debug` bound on the ok type (D181:
	/// `ClientRuntime` is not `Debug`).
	fn expect_failure(result: Result<ClientRuntime, AnyError>) -> AnyError {
		match result {
			Ok(_) => panic!("expected an error"),
			Err(failure) => failure,
		}
	}

	#[test]
	fn explicit_path_routes_require_socket_ids() {
		let server_id = "00000000-0000-4000-8000-000000000001";
		let route = route_from_explicit_path(&format!("/tmp/pi/{server_id}.sock")).unwrap();
		assert_eq!(route.server_id, server_id);
		assert_eq!(route.path, format!("/tmp/pi/{server_id}.sock"));
		assert!(route_from_explicit_path("/tmp/pi/not-a-uuid.sock").is_err());
		assert!(route_from_explicit_path("/tmp/pi/no-socket").is_err());
		assert!(route_from_explicit_path("/tmp/pi/").is_err());
	}

	#[tokio::test]
	async fn open_client_runtime_validates_option_combinations() {
		let command = ClientCommand {
			auth: Some(AuthInput::Token { token: "t".into() }),
			connect: Some(TransportAddress::Unix { path: "/tmp/x.sock".into() }),
			..Default::default()
		};
		let failure = expect_failure(open_client_runtime(&command, OpenClientRuntimeOptions::default()).await);
		assert_eq!(
			failure.to_string(),
			"Authentication is only supported for experimental Radius connections"
		);

		let command = ClientCommand { provider: Some("p".into()), ..Default::default() };
		let failure = expect_failure(open_client_runtime(&command, OpenClientRuntimeOptions::default()).await);
		assert_eq!(failure.to_string(), "Server model provider requires a model");

		let command = ClientCommand {
			connect: Some(TransportAddress::Unix { path: "/tmp/x.sock".into() }),
			model: Some("m".into()),
			..Default::default()
		};
		let failure = expect_failure(open_client_runtime(&command, OpenClientRuntimeOptions::default()).await);
		assert_eq!(failure.to_string(), "Model selection is only valid when automatically activating a new server");

		let command = ClientCommand {
			connect: Some(TransportAddress::Radius { server_id: "00000000-0000-4000-8000-000000000001".into() }),
			plugin_packages: Some(vec!["/p".into()]),
			..Default::default()
		};
		let failure = expect_failure(open_client_runtime(&command, OpenClientRuntimeOptions::default()).await);
		assert_eq!(failure.to_string(), "Plugin package paths can only be configured on a local Unix server");
	}

	#[tokio::test]
	async fn invalid_explicit_path_fails_without_disposing() {
		let command = ClientCommand {
			connect: Some(TransportAddress::Unix { path: "/tmp/not-a-server.sock".into() }),
			..Default::default()
		};
		let failure = expect_failure(open_client_runtime(&command, OpenClientRuntimeOptions::default()).await);
		assert_eq!(failure.to_string(), "--connect path must end with <uuidv4-server-id>.sock");
	}

	#[test]
	fn route_accessors_expose_transport_details() {
		let unix = ClientRuntimeRoute::Unix(UnixServerRoute {
			server_id: "00000000-0000-4000-8000-000000000001".into(),
			path: "/tmp/pi.sock".into(),
		});
		assert_eq!(unix.server_id(), "00000000-0000-4000-8000-000000000001");
		assert_eq!(unix.unix_path(), Some("/tmp/pi.sock"));
		let radius = ClientRuntimeRoute::Radius { server_id: "00000000-0000-4000-8000-000000000002".into() };
		assert_eq!(radius.server_id(), "00000000-0000-4000-8000-000000000002");
		assert_eq!(radius.unix_path(), None);
	}
}
