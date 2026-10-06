//! Port of `packages/mcp/src/client.ts`.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc};
use {parking_lot::Mutex};
use std::time::Duration;

use serde_json::{json, Map, Value};

use crate::protocol::content::CallToolResult;
use crate::protocol::jsonrpc::{
	is_json_rpc_id, is_object, JsonRpcId, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest,
	JsonRpcResponse, McpError, JSONRPC_VERSION, JSON_RPC_ERROR_CODES_INTERNAL_ERROR,
	JSON_RPC_ERROR_CODES_METHOD_NOT_FOUND,
};
use crate::protocol::types::{
	ClientCapabilities, Implementation, InitializeResult, ListResourceTemplatesResult,
	ListResourcesResult, ProgressNotification, ReadResourceResult, Resource, ResourceTemplate, Root,
	ServerCapabilities, Tool, LATEST_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS,
};
use crate::transports::transport::{McpTransport, MessageListener, Unsubscribe};

const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 30_000;
const MAX_LIST_PAGES: usize = 1_000;

pub type RequestHandlerFuture = Pin<Box<dyn Future<Output = Result<Value, McpError>> + Send>>;
pub type RequestHandler = Arc<dyn Fn(Value, &RequestContext) -> RequestHandlerFuture + Send + Sync>;
pub type RootsProviderFuture = Pin<Box<dyn Future<Output = Result<Vec<Root>, McpError>> + Send>>;
pub type RootsProvider = Arc<dyn Fn() -> RootsProviderFuture + Send + Sync>;

/// Cancellation primitive standing in for the platform `AbortController`/`AbortSignal`
/// (upstream receives signals from the host runtime; pi-runtime adapts its own).
#[derive(Clone, Default)]
pub struct AbortSignal {
	inner: Arc<SignalInner>,
}

#[derive(Default)]
struct SignalInner {
	reason: tokio::sync::watch::Sender<Option<String>>,
}

impl AbortSignal {
	pub fn aborted(&self) -> bool {
		self.inner.reason.borrow().is_some()
	}

	/// The abort reason, if aborted (upstream `signal.reason`).
	pub fn reason(&self) -> Option<String> {
		self.inner.reason.borrow().clone()
	}

	/// Resolves once aborted.
	pub async fn cancelled(&self) {
		let _ = self
			.inner
			.reason
			.subscribe()
			.wait_for(|reason| reason.is_some())
			.await;
	}
}

/// Upstream `AbortController`.
#[derive(Clone, Default)]
pub struct AbortController {
	signal: AbortSignal,
}

impl AbortController {
	pub fn new() -> AbortController {
		AbortController { signal: AbortSignal::default() }
	}

	pub fn signal(&self) -> AbortSignal {
		self.signal.clone()
	}

	pub fn abort(&self, reason: Option<String>) {
		let _ = self.signal.inner.reason.send(reason);
	}
}

/// Context handed to request handlers (upstream `{ signal: AbortSignal }`).
pub struct RequestContext {
	pub signal: AbortSignal,
}

pub type ProgressListener = Arc<dyn Fn(&ProgressNotification) + Send + Sync>;
pub type ErrorListener = Arc<dyn Fn(&McpError) + Send + Sync>;
pub type CloseListener = Arc<dyn Fn() + Send + Sync>;
pub type NotificationListener = Arc<dyn Fn(&Value) + Send + Sync>;

/// Upstream `McpClientOptions` (an `Implementation` plus extras).
pub struct McpClientOptions {
	pub name: String,
	pub version: String,
	pub title: Option<String>,
	pub capabilities: Option<ClientCapabilities>,
	/// Must be one of [`SUPPORTED_PROTOCOL_VERSIONS`] (upstream: `SupportedProtocolVersion`).
	pub protocol_version: Option<String>,
	pub request_timeout_ms: Option<u64>,
	pub roots: Option<Roots>,
}

/// Upstream `roots?: readonly Root[] | (() => ...)`.
pub enum Roots {
	Static(Vec<Root>),
	Dynamic(RootsProvider),
}

impl McpClientOptions {
	pub fn new(name: impl Into<String>, version: impl Into<String>) -> McpClientOptions {
		McpClientOptions {
			name: name.into(),
			version: version.into(),
			title: None,
			capabilities: None,
			protocol_version: None,
			request_timeout_ms: None,
			roots: None,
		}
	}
}

pub struct PendingRequest {
	resolve: parking_lot::Mutex<Option<tokio::sync::oneshot::Sender<Result<Value, McpError>>>>,
	timeout_ms: u64,
	cancellable: bool,
	on_progress: Option<ProgressListener>,
	progress_token: Option<JsonRpcId>,
	rearm: Arc<tokio::sync::Notify>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ClientState {
	Idle,
	Connecting,
	Connected,
	Closed,
}

impl ClientState {
	/// The upstream `ClientState` string.
	pub fn as_str(&self) -> &'static str {
		match self {
			ClientState::Idle => "idle",
			ClientState::Connecting => "connecting",
			ClientState::Connected => "connected",
			ClientState::Closed => "closed",
		}
	}
}

struct ClientStateInner {
	state: ClientState,
	transport: Option<Arc<dyn McpTransport>>,
	server_info: Option<Implementation>,
	server_capabilities: Option<ServerCapabilities>,
	instructions: Option<String>,
	protocol_version: Option<String>,
}

#[derive(Default)]
struct Listeners {
	notification: HashMap<String, Vec<(u64, NotificationListener)>>,
	errors: Vec<(u64, ErrorListener)>,
	closes: Vec<(u64, CloseListener)>,
	next_listener_id: u64,
}

struct ClientInner {
	options: McpClientOptions,
	state: Mutex<ClientStateInner>,
	next_request_id: AtomicU64,
	pending: Mutex<HashMap<JsonRpcId, Arc<PendingRequest>>>,
	progress_requests: Mutex<HashMap<JsonRpcId, JsonRpcId>>,
	incoming: Mutex<HashMap<JsonRpcId, AbortController>>,
	request_handlers: Mutex<HashMap<String, RequestHandler>>,
	listeners: Mutex<Listeners>,
	disposers: Mutex<Vec<Unsubscribe>>,
}

/// Standalone Model Context Protocol client (upstream `McpClient`).
pub struct McpClient {
	inner: Arc<ClientInner>,
}

/// Port of upstream `validateInitializeResult` — structural validation, then decode.
fn validate_initialize_result(value: &Value) -> Result<InitializeResult, McpError> {
	let ok = is_object(value)
		&& value.get("protocolVersion").and_then(Value::as_str).is_some()
		&& value.get("capabilities").map(is_object).unwrap_or(false)
		&& value.get("serverInfo").map(is_object).unwrap_or(false)
		&& value
			.get("serverInfo")
			.and_then(|info| info.get("name"))
			.and_then(Value::as_str)
			.is_some()
		&& value
			.get("serverInfo")
			.and_then(|info| info.get("version"))
			.and_then(Value::as_str)
			.is_some()
		&& value.get("instructions").map(Value::is_string).unwrap_or(true);
	if !ok {
		return Err(McpError::invalid("Invalid MCP initialize result"));
	}
	serde_json::from_value(value.clone()).map_err(|_| McpError::invalid("Invalid MCP initialize result"))
}

/// Port of upstream `validateListPage`: one page of a paginated list, items checked
/// by `is_item`, unknown fields preserved.
fn validate_list_page(
	method: &str,
	key: &str,
	value: &Value,
	is_item: impl Fn(&Map<String, Value>) -> bool,
) -> Result<(Vec<Map<String, Value>>, Option<String>), McpError> {
	let items = value.get(key);
	if !is_object(value) || !items.map(Value::is_array).unwrap_or(false) {
		return Err(McpError::invalid(format!("Invalid MCP {method} result")));
	}
	let mut checked = Vec::new();
	for item in items.and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]) {
		match item.as_object() {
			Some(item) if is_item(item) => checked.push(item.clone()),
			_ => return Err(McpError::invalid(format!("Invalid entry in MCP {method} result"))),
		}
	}
	// Some servers end pagination with `null` or `""` instead of omitting the cursor.
	let next_cursor = match value.get("nextCursor") {
		None | Some(Value::Null) => None,
		Some(Value::String(s)) if s.is_empty() => None,
		Some(Value::String(s)) => Some(s.clone()),
		Some(_) => return Err(McpError::invalid(format!("Invalid MCP {method} cursor"))),
	};
	Ok((checked, next_cursor))
}

fn is_tool(tool: &Map<String, Value>) -> bool {
	tool.get("name").and_then(Value::as_str).is_some()
		&& tool.get("inputSchema").map(is_object).unwrap_or(false)
}

/// `name` is required by the spec, but some servers omit it; the URI stands in.
fn is_resource(resource: &Map<String, Value>) -> bool {
	resource.get("uri").and_then(Value::as_str).is_some()
		&& resource
			.get("name")
			.map(|name| name.is_null() || name.is_string())
			.unwrap_or(true)
}

fn is_resource_template(template: &Map<String, Value>) -> bool {
	template.get("uriTemplate").and_then(Value::as_str).is_some()
		&& template
			.get("name")
			.map(|name| name.is_null() || name.is_string())
			.unwrap_or(true)
}

fn to_resource(item: &Map<String, Value>) -> Resource {
	let mut item = item.clone();
	if !item.contains_key("name") {
		let uri = item.get("uri").cloned().unwrap_or(Value::Null);
		item.insert("name".to_string(), uri);
	}
	serde_json::from_value(Value::Object(item)).expect("validated resource")
}

fn to_resource_template(item: &Map<String, Value>) -> ResourceTemplate {
	let mut item = item.clone();
	if !item.contains_key("name") {
		let uri_template = item.get("uriTemplate").cloned().unwrap_or(Value::Null);
		item.insert("name".to_string(), uri_template);
	}
	serde_json::from_value(Value::Object(item)).expect("validated resource template")
}

fn parse_tool(item: &Map<String, Value>) -> Tool {
	serde_json::from_value(Value::Object(item.clone())).expect("validated tool")
}

/// Port of upstream `validateReadResourceResult`.
fn validate_read_resource_result(value: &Value) -> Result<ReadResourceResult, McpError> {
	let contents = value.get("contents");
	if !is_object(value) || !contents.map(Value::is_array).unwrap_or(false) {
		return Err(McpError::invalid("Invalid MCP resources/read result"));
	}
	for entry in contents.and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]) {
		let ok = is_object(entry)
			&& entry.get("uri").and_then(Value::as_str).is_some()
			&& (entry.get("text").and_then(Value::as_str).is_some()
				|| entry.get("blob").and_then(Value::as_str).is_some());
		if !ok {
			return Err(McpError::invalid("Invalid contents in MCP resources/read result"));
		}
	}
	serde_json::from_value(value.clone())
		.map_err(|_| McpError::invalid("Invalid MCP resources/read result"))
}

/// `content` is required by the spec, but servers that only return `structuredContent`
/// omit it (the SDK defaults it too).
fn validate_call_tool_result(value: &Value) -> Result<CallToolResult, McpError> {
	let content = value.get("content");
	if !is_object(value) || content.map(|c| !c.is_array()).unwrap_or(false) {
		return Err(McpError::invalid("Invalid MCP tools/call result"));
	}
	if value
		.get("structuredContent")
		.map(|s| !is_object(s))
		.unwrap_or(false)
	{
		return Err(McpError::invalid("Invalid MCP tools/call structured content"));
	}
	#[derive(serde::Deserialize)]
	#[serde(rename_all = "camelCase")]
	struct Shape {
		content: Option<Vec<crate::protocol::content::ContentBlock>>,
		structured_content: Option<Map<String, Value>>,
		is_error: Option<bool>,
		#[serde(rename = "_meta")]
		meta: Option<Map<String, Value>>,
	}
	let shape: Shape = serde_json::from_value(value.clone())
		.map_err(|_| McpError::invalid("Invalid MCP tools/call result"))?;
	Ok(CallToolResult {
		content: shape.content.unwrap_or_default(),
		structured_content: shape.structured_content,
		is_error: shape.is_error,
		meta: shape.meta,
	})
}

/// Per-request options (upstream `McpRequestOptions`).
#[derive(Default, Clone)]
pub struct McpRequestOptions {
	/// Upstream `signal?: AbortSignal`.
	pub signal: Option<AbortSignal>,
	pub timeout_ms: Option<u64>,
	pub on_progress: Option<ProgressListener>,
}

impl McpRequestOptions {
	pub fn timeout_ms(mut self, timeout_ms: u64) -> McpRequestOptions {
		self.timeout_ms = Some(timeout_ms);
		self
	}

	pub fn signal(mut self, signal: AbortSignal) -> McpRequestOptions {
		self.signal = Some(signal);
		self
	}

	pub fn on_progress(mut self, on_progress: ProgressListener) -> McpRequestOptions {
		self.on_progress = Some(on_progress);
		self
	}
}

impl McpClient {
	/// Upstream `new McpClient(options)`.
	pub fn new(mut options: McpClientOptions) -> McpClient {
		let mut request_handlers: HashMap<String, RequestHandler> = HashMap::new();
		request_handlers.insert(
			"ping".to_string(),
			Arc::new(|_params: Value, _context: &RequestContext| {
				Box::pin(async { Ok(json!({})) }) as RequestHandlerFuture
			}),
		);
		if let Some(roots) = options.roots.take() {
			request_handlers.insert(
				"roots/list".to_string(),
				Arc::new(move |_params: Value, _context: &RequestContext| {
					match &roots {
						Roots::Static(roots) => {
							let roots = roots.clone();
							Box::pin(async move { Ok(json!({ "roots": roots })) }) as RequestHandlerFuture
						}
						Roots::Dynamic(provider) => {
							let future = provider();
							Box::pin(async move {
								future.await.map(|roots| json!({ "roots": roots }))
							}) as RequestHandlerFuture
						}
					}
				}),
			);
		}
		McpClient {
			inner: Arc::new(ClientInner {
				options,
				state: Mutex::new(ClientStateInner {
					state: ClientState::Idle,
					transport: None,
					server_info: None,
					server_capabilities: None,
					instructions: None,
					protocol_version: None,
				}),
				next_request_id: AtomicU64::new(1),
				pending: Mutex::new(HashMap::new()),
				progress_requests: Mutex::new(HashMap::new()),
				incoming: Mutex::new(HashMap::new()),
				request_handlers: Mutex::new(request_handlers),
				listeners: Mutex::new(Listeners::default()),
				disposers: Mutex::new(Vec::new()),
			}),
		}
	}

	fn state_lock(&self) -> parking_lot::MutexGuard<'_, ClientStateInner> {
		self.inner.state.lock()
	}

	pub fn connection_state(&self) -> ClientState {
		self.state_lock().state
	}

	pub fn server_info(&self) -> Option<Implementation> {
		self.state_lock().server_info.clone()
	}

	pub fn server_capabilities(&self) -> Option<ServerCapabilities> {
		self.state_lock().server_capabilities.clone()
	}

	pub fn instructions(&self) -> Option<String> {
		self.state_lock().instructions.clone()
	}

	pub fn protocol_version(&self) -> Option<String> {
		self.state_lock().protocol_version.clone()
	}

	/// Connect a transport and initialize the session (upstream `connect`).
	pub async fn connect(
		&self,
		transport: Arc<dyn McpTransport>,
	) -> Result<InitializeResult, McpError> {
		{
			let mut state = self.state_lock();
			if state.state != ClientState::Idle {
				return Err(McpError::other(format!(
					"Cannot connect MCP client in {} state",
					state.state.as_str()
				)));
			}
			state.state = ClientState::Connecting;
			state.transport = Some(transport.clone());
		}

		let inner = Arc::clone(&self.inner);
		*self.inner.disposers.lock() = vec![
			transport.on_message(message_listener(Arc::downgrade(&inner))),
			// Transport errors are reported only. Pending requests fail when the transport closes.
			transport.on_error(error_listener(Arc::downgrade(&inner))),
			transport.on_close(close_listener(Arc::downgrade(&inner))),
		];

		let result: Result<InitializeResult, McpError> = async {
			transport.start().await?;
			let mut capabilities = self.inner.options.capabilities.clone().unwrap_or_default();
			if self.inner.options.roots.is_some() && capabilities.roots.is_none() {
				capabilities.roots = Some(Default::default());
			}
			let mut client_info = json!({
				"name": self.inner.options.name,
				"version": self.inner.options.version,
			});
			if let Some(title) = &self.inner.options.title {
				client_info["title"] = json!(title);
			}
			let params = json!({
				"protocolVersion": self
					.inner
					.options
					.protocol_version
					.clone()
					.unwrap_or_else(|| LATEST_PROTOCOL_VERSION.to_string()),
				"capabilities": capabilities,
				"clientInfo": client_info,
			});
			let params = params.as_object().cloned().expect("object literal");
			let result_value = self
				.request_internal("initialize", Some(params), &McpRequestOptions::default(), true)
				.await?;
			let result = validate_initialize_result(&result_value)?;
			if !SUPPORTED_PROTOCOL_VERSIONS.contains(&result.protocol_version.as_str()) {
				return Err(McpError::other(format!(
					"MCP server selected unsupported protocol version {}",
					result.protocol_version
				)));
			}
			{
				let mut state = self.state_lock();
				state.protocol_version = Some(result.protocol_version.clone());
				state.server_info = Some(result.server_info.clone());
				state.server_capabilities = Some(result.capabilities.clone());
				state.instructions = result.instructions.clone();
			}
			transport.set_protocol_version(&result.protocol_version);
			self.notify_internal("notifications/initialized", None, true).await?;
			Ok(result)
		}
		.await;

		match result {
			Ok(result) => {
				self.state_lock().state = ClientState::Connected;
				Ok(result)
			}
			Err(error) => {
				let _ = self.close().await;
				Err(error)
			}
		}
	}

	/// Upstream `request`: send a request and wait for its response.
	pub async fn request(
		&self,
		method: &str,
		params: Option<Map<String, Value>>,
		options: &McpRequestOptions,
	) -> Result<Value, McpError> {
		self.request_internal(method, params, options, false).await
	}

	pub async fn notify(
		&self,
		method: &str,
		params: Option<Map<String, Value>>,
	) -> Result<(), McpError> {
		self.notify_internal(method, params, false).await
	}

	/// Register a server-to-client request handler; returns an unsubscribe handle
	/// (upstream `setRequestHandler`).
	pub fn set_request_handler<F>(&self, method: &str, handler: F) -> Unsubscribe
	where
		F: Fn(Value, &RequestContext) -> RequestHandlerFuture + Send + Sync + 'static,
	{
		self.set_request_handler_arc(method, Arc::new(handler))
	}

	pub fn set_request_handler_arc(&self, method: &str, handler: RequestHandler) -> Unsubscribe {
		self.inner
			.request_handlers
			.lock()
			
			.insert(method.to_string(), handler.clone());
		let inner = Arc::clone(&self.inner);
		let method = method.to_string();
		Box::new(move || {
			let mut handlers = inner
				.request_handlers
				.lock()
				;
			if handlers
				.get(&method)
				.map(|current| Arc::ptr_eq(current, &handler))
				.unwrap_or(false)
			{
				handlers.remove(&method);
			}
		})
	}

	pub fn on_notification(&self, method: &str, listener: NotificationListener) -> Unsubscribe {
		let id = {
			let mut listeners =
				self.inner.listeners.lock();
			let id = listeners.next_listener_id;
			listeners.next_listener_id += 1;
			listeners.notification.entry(method.to_string()).or_default().push((id, listener));
			id
		};
		let inner = Arc::clone(&self.inner);
		let method = method.to_string();
		Box::new(move || {
			let mut listeners = inner.listeners.lock();
			if let Some(entries) = listeners.notification.get_mut(&method) {
				entries.retain(|(existing, _)| *existing != id);
				if entries.is_empty() {
					listeners.notification.remove(&method);
				}
			}
		})
	}

	pub fn on_error(&self, listener: ErrorListener) -> Unsubscribe {
		let id = {
			let mut listeners =
				self.inner.listeners.lock();
			let id = listeners.next_listener_id;
			listeners.next_listener_id += 1;
			listeners.errors.push((id, listener));
			id
		};
		let inner = Arc::clone(&self.inner);
		Box::new(move || {
			inner
				.listeners
				.lock()
				
				.errors
				.retain(|(existing, _)| *existing != id);
		})
	}

	/// Called once when the connection closes, whether the transport dropped or `close()` was called.
	pub fn on_close(&self, listener: CloseListener) -> Unsubscribe {
		let id = {
			let mut listeners =
				self.inner.listeners.lock();
			let id = listeners.next_listener_id;
			listeners.next_listener_id += 1;
			listeners.closes.push((id, listener));
			id
		};
		let inner = Arc::clone(&self.inner);
		Box::new(move || {
			inner
				.listeners
				.lock()
				
				.closes
				.retain(|(existing, _)| *existing != id);
		})
	}

	pub async fn ping(&self, options: &McpRequestOptions) -> Result<(), McpError> {
		self.request("ping", None, options).await.map(|_| ())
	}

	pub async fn list_tools(&self, options: &McpRequestOptions) -> Result<Vec<Tool>, McpError> {
		let items = self.list_all("tools/list", "tools", is_tool, options).await?;
		Ok(items.iter().map(parse_tool).collect())
	}

	/// Every resource, following `nextCursor` through all pages.
	pub async fn list_resources(&self, options: &McpRequestOptions) -> Result<Vec<Resource>, McpError> {
		let items = self
			.list_all("resources/list", "resources", is_resource, options)
			.await?;
		Ok(items.iter().map(to_resource).collect())
	}

	/// One page of resources, starting at `cursor`.
	pub async fn list_resources_page(
		&self,
		cursor: Option<&str>,
		options: &McpRequestOptions,
	) -> Result<ListResourcesResult, McpError> {
		let page = self
			.list_page("resources/list", "resources", is_resource, cursor, options)
			.await?;
		Ok(ListResourcesResult {
			resources: page.0.iter().map(to_resource).collect(),
			next_cursor: page.1,
			meta: None,
		})
	}

	/// Every resource template, following `nextCursor` through all pages.
	pub async fn list_resource_templates(
		&self,
		options: &McpRequestOptions,
	) -> Result<Vec<ResourceTemplate>, McpError> {
		let items = self
			.list_all("resources/templates/list", "resourceTemplates", is_resource_template, options)
			.await?;
		Ok(items.iter().map(to_resource_template).collect())
	}

	/// One page of resource templates, starting at `cursor`.
	pub async fn list_resource_templates_page(
		&self,
		cursor: Option<&str>,
		options: &McpRequestOptions,
	) -> Result<ListResourceTemplatesResult, McpError> {
		let page = self
			.list_page(
				"resources/templates/list",
				"resourceTemplates",
				is_resource_template,
				cursor,
				options,
			)
			.await?;
		Ok(ListResourceTemplatesResult {
			resource_templates: page.0.iter().map(to_resource_template).collect(),
			next_cursor: page.1,
			meta: None,
		})
	}

	pub async fn read_resource(
		&self,
		uri: &str,
		options: &McpRequestOptions,
	) -> Result<ReadResourceResult, McpError> {
		let mut params = Map::new();
		params.insert("uri".to_string(), json!(uri));
		let result = self.request("resources/read", Some(params), options).await?;
		validate_read_resource_result(&result)
	}

	async fn list_page(
		&self,
		method: &str,
		key: &str,
		is_item: impl Fn(&Map<String, Value>) -> bool + Copy,
		cursor: Option<&str>,
		options: &McpRequestOptions,
	) -> Result<(Vec<Map<String, Value>>, Option<String>), McpError> {
		let params = cursor.map(|cursor| {
			let mut params = Map::new();
			params.insert("cursor".to_string(), json!(cursor));
			params
		});
		let value = self.request(method, params, options).await?;
		validate_list_page(method, key, &value, is_item)
	}

	/// Every item of a paginated list method (upstream `listAll`).
	async fn list_all(
		&self,
		method: &str,
		key: &str,
		is_item: impl Fn(&Map<String, Value>) -> bool + Copy,
		options: &McpRequestOptions,
	) -> Result<Vec<Map<String, Value>>, McpError> {
		let mut items: Vec<Map<String, Value>> = Vec::new();
		let mut cursors: HashSet<String> = HashSet::new();
		let mut cursor: Option<String> = None;
		for _page_number in 0..MAX_LIST_PAGES {
			let page = self
				.list_page(method, key, is_item, cursor.as_deref(), options)
				.await?;
			items.extend(page.0);
			let Some(next_cursor) = page.1 else {
				return Ok(items);
			};
			if cursors.contains(&next_cursor) {
				return Err(McpError::other(format!(
					"MCP {method} returned duplicate cursor: {next_cursor}"
				)));
			}
			cursors.insert(next_cursor.clone());
			cursor = Some(next_cursor);
		}
		Err(McpError::other(format!("MCP {method} exceeded {MAX_LIST_PAGES} pages")))
	}

	pub async fn call_tool(
		&self,
		name: &str,
		args: Option<Map<String, Value>>,
		options: &McpRequestOptions,
	) -> Result<CallToolResult, McpError> {
		let mut params = Map::new();
		params.insert("name".to_string(), json!(name));
		if let Some(args) = args {
			params.insert("arguments".to_string(), Value::Object(args));
		}
		let result = self.request("tools/call", Some(params), options).await?;
		validate_call_tool_result(&result)
	}

	/// Close the connection (upstream `close`).
	pub async fn close(&self) -> Result<(), McpError> {
		let transport = {
			let mut state = self.state_lock();
			state.transport.take()
		};
		self.dispose_transport_listeners();
		self.inner.mark_closed(McpError::connection_closed("MCP connection closed"));
		if let Some(transport) = transport {
			transport.close().await?;
		}
		Ok(())
	}

	async fn request_internal(
		&self,
		method: &str,
		params: Option<Map<String, Value>>,
		options: &McpRequestOptions,
		allow_connecting: bool,
	) -> Result<Value, McpError> {
		let transport = self.require_transport(allow_connecting)?;
		if options.signal.as_ref().map(|signal| signal.aborted()).unwrap_or(false) {
			return Err(McpError::abort("MCP request aborted"));
		}
		let id = JsonRpcId::number(self.inner.next_request_id.fetch_add(1, Ordering::SeqCst));
		let progress_token = options.on_progress.as_ref().map(|_| id.clone());
		let request_params = progress_token.as_ref().map(|progress_token| {
			let mut params = params.clone().unwrap_or_default();
			let mut meta = params
				.get("_meta")
				.and_then(Value::as_object)
				.cloned()
				.unwrap_or_default();
			meta.insert(
				"progressToken".to_string(),
				serde_json::to_value(progress_token).expect("serializable id"),
			);
			params.insert("_meta".to_string(), Value::Object(meta));
			params
		});
		let params = request_params.or(params);
		let message = JsonRpcRequest {
			jsonrpc: JSONRPC_VERSION.to_string(),
			id: id.clone(),
			method: method.to_string(),
			params: params.map(Value::Object),
		};

		let (resolve, response) = tokio::sync::oneshot::channel::<Result<Value, McpError>>();
		let timeout_ms = options
			.timeout_ms
			.or(self.inner.options.request_timeout_ms)
			.unwrap_or(DEFAULT_REQUEST_TIMEOUT_MS);
		let entry = Arc::new(PendingRequest {
			resolve: parking_lot::Mutex::new(Some(resolve)),
			timeout_ms,
			cancellable: method != "initialize",
			on_progress: options.on_progress.clone(),
			progress_token: progress_token.clone(),
			rearm: Arc::new(tokio::sync::Notify::new()),
		});
		self.inner
			.pending
			.lock()
			
			.insert(id.clone(), entry.clone());
		if let Some(progress_token) = &progress_token {
			self.inner
				.progress_requests
				.lock()
				
				.insert(progress_token.clone(), id.clone());
		}
		if entry.timeout_ms > 0 {
			self.inner.spawn_timeout_watcher(id.clone(), Arc::clone(&entry));
		}

		// Upstream: `transport.send(message).catch((error) => this.cancelPending(id, error, false))`.
		let send_inner = Arc::clone(&self.inner);
		let send_id = id.clone();
		let send_transport = Arc::clone(&transport);
		tokio::spawn(async move {
			if let Err(error) = send_transport.send(JsonRpcMessage::Request(message)).await {
				send_inner.cancel_pending(&send_id, error, false, None);
			}
		});

		let mut cancelled = std::pin::pin!(async {
			match &options.signal {
				Some(signal) => signal.cancelled().await,
				None => std::future::pending::<()>().await,
			}
		});
		tokio::select! {
			result = response => {
				result.unwrap_or_else(|_| Err(McpError::connection_closed("MCP connection closed")))
			}
			() = &mut cancelled => {
				let reason = options
					.signal
					.as_ref()
					.and_then(|signal| signal.reason())
					.unwrap_or_else(|| "Aborted".to_string());
				let error = McpError::abort("MCP request aborted");
				self.inner.cancel_pending(&id, error.clone(), entry.cancellable, Some(reason));
				Err(error)
			}
		}
	}

	async fn notify_internal(
		&self,
		method: &str,
		params: Option<Map<String, Value>>,
		allow_connecting: bool,
	) -> Result<(), McpError> {
		let transport = self.require_transport(allow_connecting)?;
		transport
			.send(JsonRpcMessage::Notification(JsonRpcNotification {
				jsonrpc: JSONRPC_VERSION.to_string(),
				method: method.to_string(),
				params: params.map(Value::Object),
			}))
			.await
	}

	fn require_transport(&self, allow_connecting: bool) -> Result<Arc<dyn McpTransport>, McpError> {
		let state = self.state_lock();
		if state.transport.is_some()
			&& (state.state == ClientState::Connected
				|| (allow_connecting && state.state == ClientState::Connecting))
		{
			return Ok(state.transport.clone().expect("checked above"));
		}
		Err(McpError::connection_closed(format!("MCP client is {}", state.state.as_str())))
	}

	fn dispose_transport_listeners(&self) {
		let disposers =
			std::mem::take(&mut *self.inner.disposers.lock());
		for dispose in disposers {
			dispose();
		}
	}
}

impl ClientInner {
	/// Timeout handling (upstream `armTimeout`): restart the per-request deadline.
	fn arm_timeout(&self, entry: &PendingRequest) {
		if entry.timeout_ms == 0 {
			return;
		}
		entry.rearm.notify_one();
	}

	/// Per-request timer task standing in for upstream's `setTimeout` entry timer.
	fn spawn_timeout_watcher(self: &Arc<Self>, id: JsonRpcId, entry: Arc<PendingRequest>) {
		let inner = Arc::clone(self);
		tokio::spawn(async move {
			loop {
				tokio::select! {
					() = tokio::time::sleep(Duration::from_millis(entry.timeout_ms)) => {
						inner.cancel_pending(
							&id,
							McpError::timeout(entry.timeout_ms),
							entry.cancellable,
							Some("Request timed out".to_string()),
						);
						return;
					}
					() = entry.rearm.notified() => {
						// Progress renewed the deadline; restart the sleep.
					}
				}
			}
		});
	}

	/// Bookkeeping when an entry settles (upstream `removePending`, minus the map
	/// entry itself, which each caller removes).
	fn remove_pending_side(&self, entry: &Arc<PendingRequest>) {
		if let Some(progress_token) = &entry.progress_token {
			self.progress_requests
				.lock()
				
				.remove(progress_token);
		}
	}

	/// Upstream `cancelPending`.
	fn cancel_pending(self: &Arc<Self>, id: &JsonRpcId, error: McpError, notify_server: bool, reason: Option<String>) {
		let entry = {
			let mut pending = self.pending.lock();
			pending.remove(id)
		};
		let Some(entry) = entry else { return };
		self.remove_pending_side(&entry);
		if let Some(resolve) = entry.resolve.lock().take() {
			let _ = resolve.send(Err(error));
		}
		if notify_server {
			if let Some(transport) = self.transport_snapshot() {
				let mut params = Map::new();
				params.insert("requestId".to_string(), serde_json::to_value(id).expect("serializable id"));
				if let Some(reason) = reason.filter(|reason| !reason.is_empty()) {
					params.insert("reason".to_string(), json!(reason));
				}
				let inner = Arc::clone(self);
				tokio::spawn(async move {
					if let Err(send_error) = transport
						.send(JsonRpcMessage::Notification(JsonRpcNotification {
							jsonrpc: JSONRPC_VERSION.to_string(),
							method: "notifications/cancelled".to_string(),
							params: Some(Value::Object(params)),
						}))
						.await
					{
						inner.emit_error(send_error);
					}
				});
			}
		}
	}

	fn transport_snapshot(&self) -> Option<Arc<dyn McpTransport>> {
		self.state
			.lock()
			
			.transport
			.clone()
	}

	fn emit_error(&self, error: McpError) {
		let listeners: Vec<crate::client::ErrorListener> = {
			let listeners = self.listeners.lock();
			listeners.errors.iter().map(|(_, listener)| listener.clone()).collect()
		};
		for listener in listeners {
			listener(&error);
		}
	}

	fn emit_error_ref(&self, error: &McpError) {
		self.emit_error(error.clone());
	}

	/// Upstream `handleMessage`.
	fn handle_message(inner: Arc<ClientInner>, message: JsonRpcMessage) {
		match message {
			JsonRpcMessage::Response(response) => inner.handle_response(response),
			JsonRpcMessage::Request(request) => {
				tokio::spawn(async move {
					ClientInner::handle_request(inner, request).await;
				});
			}
			JsonRpcMessage::Notification(notification) => {
				inner.handle_notification(&notification.method, notification.params.as_ref());
			}
		}
	}

	/// Upstream `handleResponse`.
	fn handle_response(&self, message: JsonRpcResponse) {
		let id = message.id().clone();
		let entry = {
			let mut pending = self.pending.lock();
			pending.remove(&id)
		};
		let Some(entry) = entry else {
			self.emit_error(McpError::other(format!(
				"Received response for unknown MCP request {id}"
			)));
			return;
		};
		self.remove_pending_side(&entry);
		match message {
			JsonRpcResponse::Success(success) => {
				if let Some(resolve) =
					entry.resolve.lock().take()
				{
					let _ = resolve.send(Ok(success.result));
				}
			}
			JsonRpcResponse::Error(error) => {
				let rejection = McpError::JsonRpc {
					code: error.error.code,
					message: error.error.message,
					data: error.error.data,
				};
				if let Some(resolve) =
					entry.resolve.lock().take()
				{
					let _ = resolve.send(Err(rejection));
				}
			}
		}
	}

	/// Upstream `handleRequest` (serving server-to-client requests).
	async fn handle_request(inner: Arc<ClientInner>, message: JsonRpcRequest) {
		let transport = match inner.transport_snapshot() {
			Some(transport) => transport,
			None => return,
		};
		let handler = inner
			.request_handlers
			.lock()
			
			.get(&message.method)
			.cloned();
		let Some(handler) = handler else {
			let error = transport
				.send(JsonRpcMessage::Response(JsonRpcResponse::Error(crate::protocol::jsonrpc::JsonRpcErrorResponse {
					jsonrpc: JSONRPC_VERSION.to_string(),
					id: message.id.clone(),
					error: crate::protocol::jsonrpc::JsonRpcErrorObject {
						code: JSON_RPC_ERROR_CODES_METHOD_NOT_FOUND,
						message: format!("Method not found: {}", message.method),
						data: None,
					},
				})))
				.await;
			if let Err(error) = error {
				inner.emit_error(error);
			}
			return;
		};
		let controller = AbortController::new();
		inner
			.incoming
			.lock()
			
			.insert(message.id.clone(), controller.clone());
		let context = RequestContext { signal: controller.signal() };
		let params = message.params.clone().unwrap_or(Value::Null);
		let result = handler(params, &context).await;
		let response = match result {
			// Upstream sends `result ?? {}`.
			Ok(result) if !result.is_null() => {
				JsonRpcMessage::Response(JsonRpcResponse::Success(crate::protocol::jsonrpc::JsonRpcSuccessResponse {
					jsonrpc: JSONRPC_VERSION.to_string(),
					id: message.id.clone(),
					result,
				}))
			}
			Ok(_) => JsonRpcMessage::Response(JsonRpcResponse::Success(
				crate::protocol::jsonrpc::JsonRpcSuccessResponse {
					jsonrpc: JSONRPC_VERSION.to_string(),
					id: message.id.clone(),
					result: json!({}),
				},
			)),
			Err(error) => {
				let response_error = match &error {
					McpError::JsonRpc { code, message, data } => crate::protocol::jsonrpc::JsonRpcErrorObject {
						code: *code,
						message: message.clone(),
						data: data.clone(),
					},
					other => crate::protocol::jsonrpc::JsonRpcErrorObject {
						code: JSON_RPC_ERROR_CODES_INTERNAL_ERROR,
						message: other.to_string(),
						data: None,
					},
				};
				JsonRpcMessage::Response(JsonRpcResponse::Error(crate::protocol::jsonrpc::JsonRpcErrorResponse {
					jsonrpc: JSONRPC_VERSION.to_string(),
					id: message.id.clone(),
					error: response_error,
				}))
			}
		};
		inner
			.incoming
			.lock()
			
			.remove(&message.id);
		if let Err(send_error) = transport.send(response).await {
			inner.emit_error(send_error);
		}
	}

	/// Upstream `handleNotification`.
	fn handle_notification(&self, method: &str, params: Option<&Value>) {
		let params = params.cloned().unwrap_or(Value::Null);
		if method == "notifications/progress" {
			self.handle_progress(&params);
		} else if method == "notifications/cancelled" {
			self.handle_cancelled(&params);
		}
		let listeners: Vec<NotificationListener> = {
			let listeners = self.listeners.lock();
			listeners
				.notification
				.get(method)
				.map(|entries| entries.iter().map(|(_, listener)| listener.clone()).collect())
				.unwrap_or_default()
		};
		for listener in listeners {
			listener(&params);
		}
	}

	/// Upstream `handleProgress`.
	fn handle_progress(&self, params: &Value) {
		let token = params.get("progressToken").filter(|token| is_json_rpc_id(token));
		let progress = params.get("progress").and_then(Value::as_f64);
		let (Some(token), Some(_)) = (token, progress) else {
			return;
		};
		let Ok(token) = serde_json::from_value::<JsonRpcId>(token.clone()) else {
			return;
		};
		let request_id = self
			.progress_requests
			.lock()
			
			.get(&token)
			.cloned();
		let Some(request_id) = request_id else { return };
		let entry = {
			let pending = self.pending.lock();
			pending.get(&request_id).cloned()
		};
		let Some(entry) = entry else { return };
		self.arm_timeout(&entry);
		if let Some(on_progress) = &entry.on_progress {
			if let Ok(notification) = serde_json::from_value::<ProgressNotification>(params.clone()) {
				on_progress(&notification);
			}
		}
	}

	/// Upstream `handleCancelled`.
	fn handle_cancelled(&self, params: &Value) {
		if !is_object(params) {
			return;
		}
		let Some(request_id) = params.get("requestId").filter(|id| is_json_rpc_id(id)) else {
			return;
		};
		let Ok(request_id) = serde_json::from_value::<JsonRpcId>(request_id.clone()) else {
			return;
		};
		let controller = self
			.incoming
			.lock()
			
			.get(&request_id)
			.cloned();
		if let Some(controller) = controller {
			controller.abort(params.get("reason").and_then(Value::as_str).map(str::to_string));
		}
	}

	fn handle_transport_close(&self) {
		self.mark_closed(McpError::connection_closed("MCP connection closed"));
	}

	/// Idempotent: rejects in-flight requests, aborts server requests we are serving,
	/// and flips the state. Close listeners fire once (upstream `markClosed`).
	fn mark_closed(&self, error: McpError) {
		let was_closed = {
			let mut state = self.state.lock();
			let was_closed = state.state == ClientState::Closed;
			state.state = ClientState::Closed;
			was_closed
		};
		self.reject_pending(&error);
		{
			let mut incoming = self.incoming.lock();
			for (_, controller) in incoming.drain() {
				controller.abort(Some(error.to_string()));
			}
		}
		if was_closed {
			return;
		}
		let listeners: Vec<CloseListener> = {
			let listeners = self.listeners.lock();
			listeners.closes.iter().map(|(_, listener)| listener.clone()).collect()
		};
		for listener in listeners {
			listener();
		}
	}

	fn reject_pending(&self, error: &McpError) {
		let entries: Vec<Arc<PendingRequest>> = {
			let mut pending = self.pending.lock();
			pending.drain().map(|(_, entry)| entry).collect()
		};
		for entry in entries {
			self.remove_pending_side(&entry);
			if let Some(resolve) = entry.resolve.lock().take() {
				let _ = resolve.send(Err(error.clone()));
			}
		}
	}
}

fn message_listener(inner: std::sync::Weak<ClientInner>) -> MessageListener {
	Arc::new(move |message| {
		let Some(inner) = inner.upgrade() else { return };
		ClientInner::handle_message(inner, message);
	})
}

fn error_listener(inner: std::sync::Weak<ClientInner>) -> crate::transports::transport::ErrorListener {
	Arc::new(move |error| {
		let Some(inner) = inner.upgrade() else { return };
		inner.emit_error_ref(error);
	})
}

fn close_listener(inner: std::sync::Weak<ClientInner>) -> crate::transports::transport::CloseListener {
	Arc::new(move || {
		let Some(inner) = inner.upgrade() else { return };
		inner.handle_transport_close();
	})
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::protocol::jsonrpc::{JsonRpcErrorResponse, JsonRpcErrorObject, JsonRpcSuccessResponse};
	use crate::testing::create_in_memory_transport_pair;
	use crate::transports::in_memory::InMemoryTransport;
	use futures_util::future::BoxFuture;
	use std::collections::HashMap;
	use parking_lot::Mutex as StdMutex;
	use std::task::{Context, Poll};

	type TestHandler =
		Arc<dyn Fn(&JsonRpcRequest) -> BoxFuture<'static, Result<Value, McpError>> + Send + Sync>;

	struct TestServer {
		transport: Arc<InMemoryTransport>,
		messages: Arc<StdMutex<Vec<JsonRpcMessage>>>,
		handlers: Arc<StdMutex<HashMap<String, TestHandler>>>,
	}

	impl TestServer {
		fn set_handler<F>(&self, method: &str, handler: F)
		where
			F: Fn(&JsonRpcRequest) -> BoxFuture<'static, Result<Value, McpError>> + Send + Sync + 'static,
		{
			self.handlers
				.lock()
				
				.insert(method.to_string(), Arc::new(handler));
		}

		fn wire_messages(&self) -> Vec<Value> {
			self.messages
				.lock()
				
				.iter()
				.map(|message| serde_json::to_value(message).expect("serializable"))
				.collect()
		}

		fn has_notification(&self, method: &str) -> bool {
			self.messages
				.lock()
				
				.iter()
				.any(|message| {
					matches!(message, JsonRpcMessage::Notification(notification)
						if notification.method == method)
				})
		}
	}

	fn success(id: JsonRpcId, result: Value) -> JsonRpcMessage {
		JsonRpcMessage::Response(JsonRpcResponse::Success(JsonRpcSuccessResponse {
			jsonrpc: JSONRPC_VERSION.to_string(),
			id,
			result,
		}))
	}

	fn static_value(
		value: Value,
	) -> impl Fn(&JsonRpcRequest) -> BoxFuture<'static, Result<Value, McpError>> + Send + Sync + 'static
	{
		move |_request| {
			let value = value.clone();
			Box::pin(async move { Ok(value) })
		}
	}

	async fn create_server() -> (Arc<InMemoryTransport>, TestServer) {
		let pair = create_in_memory_transport_pair();
		let handlers: Arc<StdMutex<HashMap<String, TestHandler>>> =
			Arc::new(StdMutex::new(HashMap::new()));
		let messages: Arc<StdMutex<Vec<JsonRpcMessage>>> = Arc::new(StdMutex::new(Vec::new()));
		let handler_map = Arc::clone(&handlers);
		let message_log = Arc::clone(&messages);
		let server_transport = Arc::clone(&pair.server);
		pair.server.on_message(Arc::new(move |message| {
			message_log
				.lock()
				
				.push(message.clone());
			let JsonRpcMessage::Request(request) = message else { return };
			let handlers = Arc::clone(&handler_map);
			let server = Arc::clone(&server_transport);
			tokio::spawn(async move {
				let handler = handlers
					.lock()
					
					.get(&request.method)
					.cloned();
				let response = match handler {
					None => Err(McpError::json_rpc(
						JSON_RPC_ERROR_CODES_METHOD_NOT_FOUND,
						format!("Method not found: {}", request.method),
						None,
					)),
					Some(handler) => handler(&request).await,
				};
				let reply = match response {
					Ok(result) => success(request.id.clone(), result),
					Err(error) => JsonRpcMessage::Response(JsonRpcResponse::Error(JsonRpcErrorResponse {
						jsonrpc: JSONRPC_VERSION.to_string(),
						id: request.id.clone(),
						error: JsonRpcErrorObject {
							code: error.code().unwrap_or(JSON_RPC_ERROR_CODES_INTERNAL_ERROR),
							message: error.to_string(),
							data: match &error {
								McpError::JsonRpc { data, .. } => data.clone(),
								_ => None,
							},
						},
					})),
				};
				let _ = server.send(reply).await;
			});
		}));
		pair.server.start().await.expect("server start");
		let test_server = TestServer { transport: pair.server, messages, handlers };
		test_server.set_handler(
			"initialize",
			|_request| {
				Box::pin(async {
					Ok(json!({
						"protocolVersion": LATEST_PROTOCOL_VERSION,
						"capabilities": { "tools": { "listChanged": true } },
						"serverInfo": { "name": "test-server", "version": "1.0.0" },
						"instructions": "Use test tools.",
					}))
				})
			},
		);
		(pair.client, test_server)
	}

	async fn connect() -> (McpClient, TestServer) {
		let (client_transport, server) = create_server().await;
		let client = McpClient::new(McpClientOptions::new("test-client", "2.0.0"));
		client.connect(client_transport).await.expect("connect");
		(client, server)
	}

	/// Let in-memory deliveries and spawned handlers settle.
	async fn flush() {
		for _ in 0..50 {
			tokio::task::yield_now().await;
			tokio::time::sleep(Duration::from_millis(2)).await;
		}
	}

	/// Drive a call future up to its first await so the request is on the wire,
	/// mirroring upstream's synchronous promise-executor start.
	fn start<T>(future: &mut std::pin::Pin<&mut (dyn Future<Output = T> + Send)>) {
		let waker = futures_util::task::noop_waker();
		let mut context = Context::from_waker(&waker);
		let _ = future.as_mut().poll(&mut context);
	}

	// Port of client.test.ts "initializes the connection before exposing server information".
	#[tokio::test]
	async fn initializes_the_connection_before_exposing_server_information() {
		let (client, server) = connect().await;
		assert_eq!(client.connection_state(), ClientState::Connected);
		assert_eq!(client.protocol_version().as_deref(), Some(LATEST_PROTOCOL_VERSION));
		assert_eq!(
			client.server_info(),
			Some(Implementation { name: "test-server".into(), version: "1.0.0".into(), title: None })
		);
		assert_eq!(
			serde_json::to_value(&client.server_capabilities()).unwrap(),
			json!({ "tools": { "listChanged": true } })
		);
		assert_eq!(client.instructions().as_deref(), Some("Use test tools."));
		flush().await;
		assert_eq!(
			server.wire_messages(),
			vec![
				json!({
					"jsonrpc": "2.0",
					"id": 1,
					"method": "initialize",
					"params": {
						"protocolVersion": LATEST_PROTOCOL_VERSION,
						"capabilities": {},
						"clientInfo": { "name": "test-client", "version": "2.0.0" },
					},
				}),
				json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
			]
		);
		client.close().await.expect("close");
	}

	// Port of client.test.ts "paginates tools and preserves protocol tool definitions".
	#[tokio::test]
	async fn paginates_tools_and_preserves_protocol_tool_definitions() {
		let (client, server) = connect().await;
		server.set_handler("tools/list", |request| {
			let cursor = request
				.params
				.as_ref()
				.and_then(|params| params.get("cursor"))
				.and_then(Value::as_str)
				.map(str::to_string);
			Box::pin(async move {
				if cursor.is_none() {
					Ok(json!({
						"tools": [{ "name": "search", "description": "Search", "inputSchema": { "type": "object" } }],
						"nextCursor": "page-2",
					}))
				} else {
					// Some servers end pagination with an empty cursor instead of omitting it.
					Ok(json!({
						"tools": [{
							"name": "read",
							"inputSchema": { "type": "object" },
							"outputSchema": { "type": "object" },
							"annotations": { "readOnlyHint": true },
						}],
						"nextCursor": "",
					}))
				}
			})
		});
		let tools = client.list_tools(&McpRequestOptions::default()).await.expect("list tools");
		assert_eq!(
			serde_json::to_value(&tools).unwrap(),
			json!([
				{ "name": "search", "description": "Search", "inputSchema": { "type": "object" } },
				{
					"name": "read",
					"inputSchema": { "type": "object" },
					"outputSchema": { "type": "object" },
					"annotations": { "readOnlyHint": true },
				},
			])
		);
		client.close().await.expect("close");
	}

	// Port of client.test.ts "lists and reads resources".
	#[tokio::test]
	async fn lists_and_reads_resources() {
		let (client, server) = connect().await;
		server.set_handler("resources/list", |request| {
			let cursor = request
				.params
				.as_ref()
				.and_then(|params| params.get("cursor"))
				.and_then(Value::as_str)
				.map(str::to_string);
			Box::pin(async move {
				if cursor.is_none() {
					Ok(json!({
						"resources": [{ "uri": "file:///a", "name": "a", "mimeType": "text/plain" }],
						"nextCursor": "2",
					}))
				} else {
					Ok(json!({ "resources": [{ "uri": "file:///b" }] }))
				}
			})
		});
		server.set_handler(
			"resources/templates/list",
			static_value(json!({
				"resourceTemplates": [{ "uriTemplate": "repo://{owner}/{repo}", "name": "repo" }],
			})),
		);
		server.set_handler("resources/read", |request| {
			let uri = request
				.params
				.as_ref()
				.and_then(|params| params.get("uri"))
				.and_then(Value::as_str)
				.unwrap_or_default()
				.to_string();
			Box::pin(async move { Ok(json!({ "contents": [{ "uri": uri, "text": "hello" }] })) })
		});
		// A missing name falls back to the URI.
		let resources = client.list_resources(&McpRequestOptions::default()).await.expect("list resources");
		assert_eq!(
			serde_json::to_value(&resources).unwrap(),
			json!([
				{ "uri": "file:///a", "name": "a", "mimeType": "text/plain" },
				{ "uri": "file:///b", "name": "file:///b" },
			])
		);
		let templates =
			client.list_resource_templates(&McpRequestOptions::default()).await.expect("templates");
		assert_eq!(
			serde_json::to_value(&templates).unwrap(),
			json!([{ "uriTemplate": "repo://{owner}/{repo}", "name": "repo" }])
		);
		// Single pages pass the cursor through.
		let page = client.list_resources_page(None, &McpRequestOptions::default()).await.expect("page 1");
		assert_eq!(
			serde_json::to_value(&page).unwrap(),
			json!({
				"resources": [{ "uri": "file:///a", "name": "a", "mimeType": "text/plain" }],
				"nextCursor": "2",
			})
		);
		let page =
			client.list_resources_page(Some("2"), &McpRequestOptions::default()).await.expect("page 2");
		assert_eq!(
			serde_json::to_value(&page).unwrap(),
			json!({ "resources": [{ "uri": "file:///b", "name": "file:///b" }] })
		);
		let read = client.read_resource("file:///a", &McpRequestOptions::default()).await.expect("read");
		assert_eq!(
			serde_json::to_value(&read).unwrap(),
			json!({ "contents": [{ "uri": "file:///a", "text": "hello" }] })
		);

		server
			.set_handler("resources/read", static_value(json!({ "contents": [{ "uri": "file:///a" }] })));
		let error = client
			.read_resource("file:///a", &McpRequestOptions::default())
			.await
			.expect_err("invalid contents");
		assert_eq!(error.to_string(), "Invalid contents in MCP resources/read result");
		server.set_handler("resources/list", static_value(json!({ "resources": [{ "name": "no uri" }] })));
		let error = client
			.list_resources(&McpRequestOptions::default())
			.await
			.expect_err("invalid entry");
		assert_eq!(error.to_string(), "Invalid entry in MCP resources/list result");
		client.close().await.expect("close");
	}

	// Port of client.test.ts "returns structured tool content and surfaces JSON-RPC errors".
	#[tokio::test]
	async fn returns_structured_tool_content_and_surfaces_json_rpc_errors() {
		let (client, server) = connect().await;
		server.set_handler("tools/call", |request| {
			let name = request
				.params
				.as_ref()
				.and_then(|params| params.get("name"))
				.and_then(Value::as_str)
				.unwrap_or_default()
				.to_string();
			let count = request
				.params
				.as_ref()
				.and_then(|params| params.pointer("/arguments/count"))
				.cloned()
				.unwrap_or(Value::Null);
			Box::pin(async move {
				if name == "fail" {
					return Err(McpError::json_rpc(1234, "tool failed", Some(json!({ "retryable": false }))));
				}
				Ok(json!({
					"content": [{ "type": "text", "text": "ok" }],
					"structuredContent": { "count": count },
				}))
			})
		});
		let mut args = serde_json::Map::new();
		args.insert("count".to_string(), json!(3));
		let result =
			client.call_tool("count", Some(args), &McpRequestOptions::default()).await.expect("call");
		assert_eq!(
			serde_json::to_value(&result).unwrap(),
			json!({
				"content": [{ "type": "text", "text": "ok" }],
				"structuredContent": { "count": 3 },
			})
		);
		let error = client
			.call_tool("fail", None, &McpRequestOptions::default())
			.await
			.expect_err("tool failed");
		match error {
			McpError::JsonRpc { code, message, data } => {
				assert_eq!(code, 1234);
				assert_eq!(message, "tool failed");
				assert_eq!(data, Some(json!({ "retryable": false })));
			}
			other => panic!("expected JSON-RPC error, got {other:?}"),
		}
		client.close().await.expect("close");
	}

	// Port of client.test.ts "renews the timeout on progress" (vitest fake timers → paused clock).
	#[tokio::test(start_paused = true)]
	async fn renews_the_timeout_on_progress() {
		let (client, server) = connect().await;
		let progress_transport = Arc::clone(&server.transport);
		server.set_handler("tools/call", move |request| {
			let progress_transport = Arc::clone(&progress_transport);
			let token = request
				.params
				.as_ref()
				.and_then(|params| params.pointer("/_meta/progressToken"))
				.cloned()
				.expect("progress token");
			Box::pin(async move {
				tokio::spawn(async move {
					tokio::time::sleep(Duration::from_millis(40)).await;
					let _ = progress_transport
						.send(JsonRpcMessage::Notification(JsonRpcNotification {
							jsonrpc: JSONRPC_VERSION.to_string(),
							method: "notifications/progress".to_string(),
							params: Some(json!({
								"progressToken": token,
								"progress": 1,
								"total": 2,
							})),
						}))
						.await;
				});
				tokio::time::sleep(Duration::from_millis(80)).await;
				Ok(json!({ "content": [{ "type": "text", "text": "done" }] }))
			})
		});
		let progress: Arc<StdMutex<Vec<ProgressNotification>>> = Arc::new(StdMutex::new(Vec::new()));
		let progress_listener = Arc::clone(&progress);
		let options = McpRequestOptions::default()
			.timeout_ms(50)
			.on_progress(Arc::new(move |progress: &ProgressNotification| {
				progress_listener
					.lock()
					
					.push(progress.clone());
			}));
		let call = client.call_tool("slow", None, &options);
		let result = call.await.expect("call completes before renewed timeout");
		assert_eq!(
			serde_json::to_value(&result).unwrap(),
			json!({ "content": [{ "type": "text", "text": "done" }] })
		);
		let progress = progress.lock().clone();
		assert_eq!(progress.len(), 1);
		assert_eq!(
			serde_json::to_value(&progress[0]).unwrap(),
			json!({ "progressToken": 2, "progress": 1, "total": 2 })
		);
		client.close().await.expect("close");
	}

	// Port of client.test.ts "cancels aborted and timed-out requests".
	#[tokio::test]
	async fn cancels_aborted_and_timed_out_requests() {
		let (client, server) = connect().await;
		server.set_handler(
			"tools/call",
			|_request| Box::pin(async { std::future::pending::<Result<Value, McpError>>().await }),
		);
		let controller = AbortController::new();
		let aborted_options = McpRequestOptions::default().signal(controller.signal());
		let mut aborted = Box::pin(client.call_tool("wait", None, &aborted_options));
		{
			let mut started: std::pin::Pin<&mut (dyn Future<Output = Result<CallToolResult, McpError>> + Send)> =
				aborted.as_mut();
			start(&mut started);
		}
		controller.abort(Some("stop".to_string()));
		let error = aborted.await.expect_err("aborted");
		assert!(matches!(error, McpError::Abort { .. }));
		flush().await;
		assert!(server.wire_messages().contains(&json!({
			"jsonrpc": "2.0",
			"method": "notifications/cancelled",
			"params": { "requestId": 2, "reason": "stop" },
		})));

		let error = client
			.call_tool("wait", None, &McpRequestOptions::default().timeout_ms(5))
			.await
			.expect_err("timed out");
		assert!(matches!(error, McpError::Timeout { .. }));
		client.close().await.expect("close");
	}

	// Port of client.test.ts "reports transport errors without failing pending requests".
	#[tokio::test]
	async fn reports_transport_errors_without_failing_pending_requests() {
		let (client_transport, server) = create_server().await;
		let client = McpClient::new(McpClientOptions::new("test-client", "1.0.0"));
		client.connect(client_transport.clone()).await.expect("connect");
		let errors: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
		let error_listener = Arc::clone(&errors);
		client.on_error(Arc::new(move |error: &McpError| {
			error_listener
				.lock()
				
				.push(error.to_string());
		}));
		let release = Arc::new(tokio::sync::Notify::new());
		let release_for_handler = Arc::clone(&release);
		server.set_handler("tools/call", move |_request| {
			let release = Arc::clone(&release_for_handler);
			Box::pin(async move {
				release.notified().await;
				Ok(json!({ "content": [] }))
			})
		});
		let wait_options = McpRequestOptions::default();
		let call = client.call_tool("wait", None, &wait_options);
		let mut call = std::pin::pin!(call);
		{
			let mut started: std::pin::Pin<
				&mut (dyn Future<Output = Result<CallToolResult, McpError>> + Send),
			> = call.as_mut();
			start(&mut started);
		}
		flush().await;
		client_transport.emit_error(McpError::other("stray log line"));
		release.notify_one();
		let result = call.await.expect("call still resolves");
		assert_eq!(serde_json::to_value(&result).unwrap(), json!({ "content": [] }));
		assert_eq!(
			errors.lock().clone(),
			vec!["stray log line".to_string()]
		);
		client.close().await.expect("close");
	}

	// Port of client.test.ts "accepts servers that answer with an older protocol version".
	#[tokio::test]
	async fn accepts_servers_that_answer_with_an_older_protocol_version() {
		let (client_transport, server) = create_server().await;
		server.set_handler("initialize", |_request| {
			Box::pin(async {
				Ok(json!({
					"protocolVersion": "2024-11-05",
					"capabilities": {},
					"serverInfo": { "name": "old-server", "version": "0.1.0" },
				}))
			})
		});
		let client = McpClient::new(McpClientOptions::new("test-client", "1.0.0"));
		client.connect(client_transport).await.expect("connect");
		assert_eq!(client.protocol_version().as_deref(), Some("2024-11-05"));
		client.close().await.expect("close");

		let (unsupported_transport, unsupported) = create_server().await;
		unsupported.set_handler("initialize", |_request| {
			Box::pin(async {
				Ok(json!({
					"protocolVersion": "1999-01-01",
					"capabilities": {},
					"serverInfo": { "name": "ancient-server", "version": "0.1.0" },
				}))
			})
		});
		let rejected = McpClient::new(McpClientOptions::new("test-client", "1.0.0"));
		let error = rejected.connect(unsupported_transport).await.expect_err("rejected");
		assert!(error.to_string().contains("unsupported protocol version"), "{error}");
		assert_eq!(rejected.connection_state(), ClientState::Closed);
	}

	// Port of client.test.ts "defaults missing tool result content to an empty list".
	#[tokio::test]
	async fn defaults_missing_tool_result_content_to_an_empty_list() {
		let (client, server) = connect().await;
		server.set_handler("tools/call", static_value(json!({ "structuredContent": { "ok": true } })));
		let result = client.call_tool("structured", None, &McpRequestOptions::default()).await.expect("call");
		assert_eq!(
			serde_json::to_value(&result).unwrap(),
			json!({ "content": [], "structuredContent": { "ok": true } })
		);
		server.set_handler("tools/call", static_value(json!({ "content": "not a list" })));
		let error = client
			.call_tool("broken", None, &McpRequestOptions::default())
			.await
			.expect_err("invalid result");
		assert_eq!(error.to_string(), "Invalid MCP tools/call result");
		client.close().await.expect("close");
	}

	// Port of client.test.ts "does not send notifications/cancelled for a timed-out initialize".
	#[tokio::test]
	async fn does_not_send_notifications_cancelled_for_a_timed_out_initialize() {
		let (client_transport, server) = create_server().await;
		server.set_handler(
			"initialize",
			|_request| Box::pin(async { std::future::pending::<Result<Value, McpError>>().await }),
		);
		let mut options = McpClientOptions::new("test-client", "1.0.0");
		options.request_timeout_ms = Some(5);
		let client = McpClient::new(options);
		let error = client.connect(client_transport).await.expect_err("timed out");
		assert!(matches!(error, McpError::Timeout { .. }));
		flush().await;
		assert!(!server.has_notification("notifications/cancelled"));
	}

	// Port of client.test.ts "notifies close listeners once when the transport drops".
	#[tokio::test]
	async fn notifies_close_listeners_once_when_the_transport_drops() {
		let (client, server) = connect().await;
		let closes = Arc::new(StdMutex::new(0usize));
		let close_listener = Arc::clone(&closes);
		client.on_close(Arc::new(move || {
			*close_listener.lock() += 1;
		}));
		server.set_handler(
			"tools/call",
			|_request| Box::pin(async { std::future::pending::<Result<Value, McpError>>().await }),
		);
		let wait_options = McpRequestOptions::default();
		let pending = client.call_tool("wait", None, &wait_options);
		let mut pending = std::pin::pin!(pending);
		{
			let mut started: std::pin::Pin<
				&mut (dyn Future<Output = Result<CallToolResult, McpError>> + Send),
			> = pending.as_mut();
			start(&mut started);
		}
		flush().await;
		server.transport.close().await.expect("close transport");
		let error = pending.await.expect_err("pending fails");
		assert_eq!(error.to_string(), "MCP connection closed");
		assert_eq!(client.connection_state(), ClientState::Closed);
		client.close().await.expect("close");
		assert_eq!(*closes.lock(), 1);
	}

	// Port of client.test.ts "answers roots/list and dispatches notifications".
	#[tokio::test]
	async fn answers_roots_list_and_dispatches_notifications() {
		let (client_transport, server) = create_server().await;
		let mut options = McpClientOptions::new("test-client", "1.0.0");
		options.roots = Some(Roots::Static(vec![Root {
			uri: "file:///workspace".to_string(),
			name: Some("workspace".to_string()),
		}]));
		let client = McpClient::new(options);
		client.connect(client_transport).await.expect("connect");
		let changes = Arc::new(StdMutex::new(Vec::<Value>::new()));
		let change_listener = Arc::clone(&changes);
		client.on_notification(
			"notifications/tools/list_changed",
			Arc::new(move |params: &Value| {
				change_listener
					.lock()
					
					.push(params.clone());
			}),
		);
		server
			.transport
			.send(success_startup_request())
			.await
			.expect("send roots/list");
		server
			.transport
			.send(JsonRpcMessage::Notification(JsonRpcNotification {
				jsonrpc: JSONRPC_VERSION.to_string(),
				method: "notifications/tools/list_changed".to_string(),
				params: None,
			}))
			.await
			.expect("send notification");
		flush().await;
		assert!(server.wire_messages().contains(&json!({
			"jsonrpc": "2.0",
			"id": "roots",
			"result": { "roots": [{ "uri": "file:///workspace", "name": "workspace" }] },
		})));
		assert_eq!(
			changes.lock().clone(),
			vec![Value::Null]
		);
		client.close().await.expect("close");
	}

	fn success_startup_request() -> JsonRpcMessage {
		JsonRpcMessage::Request(JsonRpcRequest {
			jsonrpc: JSONRPC_VERSION.to_string(),
			id: JsonRpcId::String("roots".to_string()),
			method: "roots/list".to_string(),
			params: None,
		})
	}

	// Port of client.test.ts "silences warnings for unused helper imports" — the
	// upstream suite also exercises Poll-driven starts; `start` uses these types.
	#[allow(unused)]
	fn poll_witness(_: Poll<()>) {}
}
