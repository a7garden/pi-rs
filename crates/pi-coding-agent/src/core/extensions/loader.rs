//! Port of `packages/coding-agent/src/core/extensions/loader.ts` — the
//! extension loader. **Stand-in layer (D43).**
//!
//! Upstream loads TypeScript/JavaScript extension modules through `jiti` and
//! invokes their default-exported factory with a `pi` API object. Loading
//! arbitrary TS/JS is impossible in Rust until the pi-runtime JS bridge
//! exists, so this module ports the *host-side shell* of the loader: the
//! module cache registry, the shared runtime construction with throwing
//! action stubs, the `Extension` record shape, and the load pipeline —
//! without any JS execution.
//!
//! ## D43 — the loader is a stand-in until pi-runtime lands
//!
//! What the stand-in does:
//! - `create_extension_runtime()` mirrors upstream
//!   `createExtensionRuntime()`: a shared [`ExtensionRuntime`] whose action
//!   methods panic with upstream's "not initialized" message, whose
//!   `assertActive`/`invalidate`/event-bus tracking behave 1:1, and whose
//!   pending provider/native-provider/virtual-model registrations queue
//!   exactly like upstream's pre-bind runtime.
//! - `clear_extension_cache()` clears a real global cache registry
//!   (resolved path → cached module factory + mtime) and bumps the
//!   generation, like upstream's `extensionCache` + `extensionCacheCwd` +
//!   `extensionCacheGeneration`. Changing the cwd between loads clears the
//!   cache; entries are keyed by `(path, mtime)`.
//! - `load_extensions_cached()` walks the paths with upstream's pipeline:
//!   resolve against the cwd, consult the cache by `(path, mtime)`, and
//!   collect [`Extension`] records and [`LoadRecord`] errors. A path that
//!   does not exist yields the stand-in error
//!   `Extension path does not exist: <path>` (upstream surfaces the failure
//!   as `Failed to load extension: Cannot find module …` from the jiti
//!   import; the stand-in detects it earlier with the same `{ path, error }`
//!   record shape). An existing file yields a stand-in [`Extension`] record
//!   with empty handler/tool/command collections — upstream's shape after a
//!   factory that registers nothing.
//! - `load_extension_from_factory()` mirrors upstream: it builds the
//!   [`Extension`] skeleton (path, resolved path, synthetic source info,
//!   empty collections) and invokes the factory through an
//!   [`ExtensionApiHandle`] that performs upstream's registration-side
//!   behavior — `on` (with working unsubscribe), `registerTool` (with the
//!   object-parameter-schema guard), `registerCommand`,
//!   `registerShortcut`, `registerFlag` (with default-type validation and
//!   pending flag values), the renderer registrations, `getFlag`, and the
//!   pre-bind queuing provider/virtual-model registrations. A factory `Err`
//!   maps to `Err(message)` and marks the API failed (upstream
//!   `discard()`); a later call on a failed API panics with upstream's
//!   "failed to load" message.
//!
//! What the stand-in does *not* do (lands with pi-runtime):
//! - No TS/JS module resolution, compilation, or import; cached factories
//!   are stored as `Option<ExtensionFactory>` (`None` until the bridge can
//!   produce real factories from files).
//! - Action methods (`sendMessage`, `getSettings`, …) are throwing stubs;
//!   `exec` returns a failed [`ExecResult`], `setModel` a future resolving
//!   `false` (upstream rejects the promise).
//! - `registerMcpServer`/`unregisterMcpServer`/`getMcpServers` are no-ops —
//!   [`McpServerRegistry`] is an opaque stand-in (D29).
//! - Queued registrations accumulate in the load-owned
//!   [`RuntimeQueues`] block and are drained into
//!   `ExtensionRuntimeState.pending_*` by `load_extensions_cached`; the
//!   runner port consumes them from there.
//! - An extension's `registerVirtualModel` route is context-aware; the
//!   pending queue's context-free [`VirtualModelDefinition`] carries a
//!   route stub that panics like the other pre-bind stubs until the runner
//!   supplies a bound context.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::SystemTime;

use parking_lot::Mutex;
use pi_agent::{AgentThinkingLevel, BoxFuture};
use pi_ai::types::Model;
use pi_ai::Provider as AiProvider;
use serde_json::Value as JsonValue;

use super::types::{
	CustomMessageDraft, EntryRenderer, EventBus,
	EventBusHandler, ExecOptions, ExecResult, Extension, ExtensionApi, ExtensionApiHandle, ExtensionFactory,
	ExtensionFlag, ExtensionFlagType, ExtensionRuntime, ExtensionRuntimeState, ExtensionShortcut,
	ExtensionVirtualModel, FlagOptions, FlagValue, HandlerFn, InlineExtension, KeyId, LoadExtensionsResult,
	LoadRecord, MarkdownTransformer, McpServerConfig, McpServerRegistry, MessageRenderer, ModelRoute,
	ModelRouteRequest, PendingNativeProviderRegistration, PendingProviderRegistration,
	PendingVirtualModelRegistration, ProviderConfig, RegisteredCommand, RegisteredCommandOptions,
	RegisteredMcpServer, RegisteredTool, SendMessageOptions, SendUserMessageContent, SendUserMessageOptions,
	ShortcutOptions, SlashCommandInfo, ToolDefinition, ToolInfo, ToolRendererResolver, Unsubscribe,
	VirtualModelDefinition,
};
use crate::core::settings_manager::Settings;
use crate::core::source_info::{create_synthetic_source_info, get_synthetic_path_source, is_synthetic_path};
use crate::utils::paths::{resolve_path, resolve_path_cwd};

/// Upstream's pre-bind failure message (thrown by action methods).
const NOT_INITIALIZED: &str =
	"Extension runtime not initialized. Action methods cannot be called during extension loading.";

/// Upstream's default staleness message (`invalidate()` without argument).
const STALE_DEFAULT: &str = "This extension ctx is stale after session replacement or reload. Do not use a captured pi or command ctx after ctx.newSession(), ctx.fork(), ctx.switchSession(), or ctx.reload(). For newSession, fork, and switchSession, move post-replacement work into withSession and use the ctx passed to withSession. For reload, do not use the old ctx after await ctx.reload().";

// ============================================================================
// Module cache (`extensionCache` / `extensionCacheCwd` /
// `extensionCacheGeneration`)
// ============================================================================

/// A cached module factory keyed by resolved path; the mtime is part of the
/// cache key (upstream relies on jiti's `moduleCache: false` plus its own
/// path-keyed factory cache; the stand-in invalidates on file change).
struct CacheEntry {
	mtime: Option<SystemTime>,
	factory: Option<ExtensionFactory>,
}

/// Global cache registry (`extensionCache` + cwd + generation).
#[derive(Default)]
struct ExtensionCache {
	cwd: Option<String>,
	generation: u64,
	entries: HashMap<String, CacheEntry>,
}

static EXTENSION_CACHE: LazyLock<Mutex<ExtensionCache>> = LazyLock::new(Mutex::default);

/// Clear the global module cache (`clearExtensionCache`).
pub fn clear_extension_cache() {
	let mut cache = EXTENSION_CACHE.lock();
	cache.entries.clear();
	cache.cwd = None;
	cache.generation += 1;
}

/// Cache-scope token (`ExtensionCacheToken`): loads only reuse entries
/// captured under the same cwd and generation.
#[derive(Clone)]
struct CacheToken {
	cwd: String,
	generation: u64,
}

/// Bind the cache to `cwd`, clearing it when the cwd changed
/// (`useExtensionCacheCwd`).
fn use_extension_cache_cwd(cwd: &str) -> CacheToken {
	let mut cache = EXTENSION_CACHE.lock();
	if cache.cwd.as_deref().is_some_and(|previous| previous != cwd) {
		cache.entries.clear();
		cache.cwd = None;
		cache.generation += 1;
	}
	cache.cwd = Some(cwd.to_string());
	CacheToken { cwd: cwd.to_string(), generation: cache.generation }
}

fn is_current_cache_token(token: &CacheToken) -> bool {
	let cache = EXTENSION_CACHE.lock();
	cache.generation == token.generation && cache.cwd.as_deref() == Some(token.cwd.as_str())
}

/// Look up a cached factory by `(resolved path, mtime)`
/// (`loadExtensionModule`'s cache hit path).
fn cached_factory(path: &str, mtime: Option<SystemTime>, token: &CacheToken) -> Option<ExtensionFactory> {
	if !is_current_cache_token(token) {
		return None;
	}
	let cache = EXTENSION_CACHE.lock();
	let entry = cache.entries.get(path)?;
	if entry.mtime == mtime {
		entry.factory.clone()
	} else {
		None
	}
}

/// Store a factory under `(resolved path, mtime)` when the token is still
/// current (`extensionCache.set`).
fn store_cache_entry(path: &str, mtime: Option<SystemTime>, factory: Option<ExtensionFactory>, token: &CacheToken) {
	if !is_current_cache_token(token) {
		return;
	}
	EXTENSION_CACHE
		.lock()
		.entries
		.insert(path.to_string(), CacheEntry { mtime, factory });
}

fn file_mtime(path: &str) -> Option<SystemTime> {
	std::fs::metadata(path).ok()?.modified().ok()
}

// ============================================================================
// Shared runtime state (`createExtensionRuntime`)
// ============================================================================

/// Mutable state shared by the runtime vtable closures (`state` +
/// `eventBusUnsubscribers` upstream).
#[derive(Default)]
struct RuntimeShared {
	stale_message: Option<String>,
	/// Active tracked event-bus subscriptions; drained by `invalidate`.
	tracked: Vec<Arc<TrackedUnsubscribe>>,
}

/// A tracked subscription: callable once, by its owner or `invalidate`.
struct TrackedUnsubscribe {
	active: std::sync::atomic::AtomicBool,
	inner: Mutex<Option<Unsubscribe>>,
}

impl TrackedUnsubscribe {
	/// Fire exactly once; `false` when already fired.
	fn fire(&self) -> bool {
		if self.active.swap(false, std::sync::atomic::Ordering::SeqCst) {
			if let Some(unsubscribe) = self.inner.lock().take() {
				unsubscribe();
			}
			true
		} else {
			false
		}
	}
}

/// Pending pre-bind registrations queued by the loader API; drained into
/// `ExtensionRuntimeState.pending_*` by the load functions (see D43).
#[derive(Default)]
struct RuntimeQueues {
	providers: Vec<PendingProviderRegistration>,
	native_providers: Vec<PendingNativeProviderRegistration>,
	virtual_models: Vec<PendingVirtualModelRegistration>,
}

/// Drain `queues` into the runtime state's pending vectors.
fn flush_queues(queues: &Mutex<RuntimeQueues>, state: &ExtensionRuntimeState) {
	let mut queues = queues.lock();
	*state.pending_provider_registrations.write() = std::mem::take(&mut queues.providers);
	*state.pending_native_provider_registrations.write() = std::mem::take(&mut queues.native_providers);
	*state.pending_virtual_model_registrations.write() = std::mem::take(&mut queues.virtual_models);
}

/// Create a runtime with throwing stubs for action methods
/// (`createExtensionRuntime`). `runner.bindCore()` replaces these with real
/// implementations once the runner lands.
pub fn create_extension_runtime() -> ExtensionRuntime {
	create_extension_runtime_shared().0
}

/// [`create_extension_runtime`] plus the queue block its loader API writes
/// to, so the loader can drain registrations into the state snapshots.
fn create_extension_runtime_shared() -> (ExtensionRuntime, Arc<Mutex<RuntimeQueues>>) {
	let queues = Arc::new(Mutex::new(RuntimeQueues::default()));
	let shared = Arc::new(Mutex::new(RuntimeShared::default()));

	let assert_active: Arc<dyn Fn() -> Result<(), String> + Send + Sync> = {
		let shared = Arc::clone(&shared);
		Arc::new(move || match shared.lock().stale_message.as_deref() {
			Some(message) => Err(message.to_string()),
			None => Ok(()),
		})
	};
	let invalidate: Arc<dyn Fn(Option<String>) + Send + Sync> = {
		let shared = Arc::clone(&shared);
		Arc::new(move |message: Option<String>| {
			let mut shared = shared.lock();
			if shared.stale_message.is_some() {
				return;
			}
			shared.stale_message = Some(message.unwrap_or_else(|| STALE_DEFAULT.to_string()));
			for tracked in shared.tracked.drain(..) {
				tracked.fire();
			}
		})
	};
	let track_event_bus_subscription: Arc<dyn Fn(Unsubscribe) -> Unsubscribe + Send + Sync> = {
		let shared = Arc::clone(&shared);
		Arc::new(move |unsubscribe: Unsubscribe| {
			let tracked = Arc::new(TrackedUnsubscribe {
				active: std::sync::atomic::AtomicBool::new(true),
				inner: Mutex::new(Some(unsubscribe)),
			});
			shared.lock().tracked.push(Arc::clone(&tracked));
			let owner = Arc::clone(&tracked);
			Box::new(move || {
				owner.fire();
			}) as Unsubscribe
		})
	};
	let register_provider: Arc<dyn Fn(String, ProviderConfig, Option<String>) + Send + Sync> = {
		let queues = Arc::clone(&queues);
		Arc::new(move |name: String, config: ProviderConfig, extension_path: Option<String>| {
			queues.lock().providers.push(PendingProviderRegistration {
				name,
				config,
				extension_path: extension_path.unwrap_or_else(|| "<unknown>".to_string()),
			});
		})
	};
	let register_native_provider: Arc<dyn Fn(Arc<dyn AiProvider>, Option<String>) + Send + Sync> = {
		let queues = Arc::clone(&queues);
		Arc::new(move |provider: Arc<dyn AiProvider>, extension_path: Option<String>| {
			queues.lock().native_providers.push(PendingNativeProviderRegistration {
				provider: Some(provider),
				extension_path: extension_path.unwrap_or_else(|| "<unknown>".to_string()),
			});
		})
	};
	let unregister_provider: Arc<dyn Fn(String, Option<String>) + Send + Sync> = {
		let queues = Arc::clone(&queues);
		Arc::new(move |name: String, _extension_path: Option<String>| {
			let mut queues = queues.lock();
			queues.providers.retain(|registration| registration.name != name);
			queues
				.native_providers
				.retain(|registration| !registration.provider.as_ref().is_some_and(|p| p.id() == name));
		})
	};
	let register_virtual_model: Arc<dyn Fn(VirtualModelDefinition, Option<String>) + Send + Sync> = {
		let queues = Arc::clone(&queues);
		Arc::new(move |definition: VirtualModelDefinition, extension_path: Option<String>| {
			queues.lock().virtual_models.push(PendingVirtualModelRegistration {
				definition: Some(definition),
				extension_path: extension_path.unwrap_or_else(|| "<unknown>".to_string()),
			});
		})
	};
	let unregister_virtual_model: Arc<dyn Fn(String, String) + Send + Sync> = {
		let queues = Arc::clone(&queues);
		Arc::new(move |provider: String, id: String| {
			queues.lock().virtual_models.retain(|registration| match &registration.definition {
				Some(definition) => definition.provider != provider || definition.id != id,
				None => true,
			});
		})
	};

	let state = ExtensionRuntimeState {
		flag_values: parking_lot::RwLock::new(HashMap::new()),
		pending_provider_registrations: parking_lot::RwLock::new(Vec::new()),
		pending_native_provider_registrations: parking_lot::RwLock::new(Vec::new()),
		pending_virtual_model_registrations: parking_lot::RwLock::new(Vec::new()),
		create_context: Arc::new(|| panic!("{NOT_INITIALIZED}")),
		assert_active,
		invalidate,
		track_event_bus_subscription,
		register_provider,
		register_native_provider,
		unregister_provider,
		mcp_servers: Arc::new(McpServerRegistry::default()),
		register_virtual_model,
		unregister_virtual_model,
	};
	(
		ExtensionRuntime {
			state: Arc::new(state),
			actions: super::types::ExtensionActions::default(),
			core: Arc::new(super::types::RuntimeCore::default()),
		},
		queues,
	)
}

// ============================================================================
// The loader API (`createExtensionAPI`): registration methods write to the
// extension; action methods are throwing stubs (D43).
// ============================================================================

struct ApiInner {
	extension: Extension,
	pending_flag_values: HashMap<String, FlagValue>,
	failed: bool,
}

/// Subscriptions made through `on`: the same live registry the runner
/// reads (`extension.handlers`, D40).
type HandlerRegistry = Arc<Mutex<HashMap<String, Vec<HandlerFn>>>>;

fn extension_handlers(extension: &Extension) -> HandlerRegistry {
	Arc::clone(&extension.handlers)
}

/// Per-extension loader API (`createExtensionAPI`'s `api` + load state).
struct LoaderApi {
	inner: Mutex<ApiInner>,
	handlers: HandlerRegistry,
	path: String,
	runtime: ExtensionRuntime,
	queues: Arc<Mutex<RuntimeQueues>>,
	event_bus: Arc<dyn EventBus>,
}

impl LoaderApi {
	fn assert_active(&self) {
		{
			let inner = self.inner.lock();
			if inner.failed {
				panic!("Extension \"{}\" failed to load and its API is no longer active.", self.path);
			}
		}
		if let Err(message) = (self.runtime.state.assert_active)() {
			panic!("{message}");
		}
	}

	/// Copy the extension out after a successful factory run (`commit`).
	/// Returns the extension plus the pending flag defaults captured during
	/// load (upstream promotes them into `runtime.flagValues` unless the CLI
	/// already set a value).
	fn commit(&self) -> (Extension, HashMap<String, FlagValue>) {
		let mut inner = self.inner.lock();
		let pending_flag_values = std::mem::take(&mut inner.pending_flag_values);
		(inner.extension.clone(), pending_flag_values)
	}

	/// Mark the API failed after a throwing factory (`discard`).
	fn discard(&self) {
		self.inner.lock().failed = true;
		self.handlers.lock().clear();
	}
	
}

impl ExtensionApi for LoaderApi {
	fn on(&self, event: &str, handler: HandlerFn) -> Unsubscribe {
		self.assert_active();
		let mut handlers = self.handlers.lock();
		let list = handlers.entry(event.to_string()).or_default();
		list.push(Arc::clone(&handler));
		// Upstream removes the exact registered wrapper; with erased `Arc`
		// handlers, removal is `Arc` pointer equality.
		let registry = Arc::clone(&self.handlers);
		let event = event.to_string();
		Box::new(move || {
			let mut handlers = registry.lock();
			if let Some(list) = handlers.get_mut(&event) {
				list.retain(|registered| !Arc::ptr_eq(registered, &handler));
				if list.is_empty() {
					handlers.remove(&event);
				}
			}
		})
	}

	fn register_tool(&self, tool: ToolDefinition) {
		self.assert_active();
		if !tool.parameters.is_object() {
			panic!(
				"Tool \"{}\" registered by extension \"{}\" must define an object parameter schema.",
				tool.name, self.path
			);
		}
		let mut inner = self.inner.lock();
		let source_info = inner.extension.source_info.clone();
		inner
			.extension
			.tools
			.insert(tool.name.clone(), RegisteredTool { definition: tool, source_info });
		// `runtime.refreshTools()` is a no-op pre-bind (upstream comment).
	}

	fn register_command(&self, name: &str, options: RegisteredCommandOptions) {
		self.assert_active();
		if name.is_empty() {
			panic!(
				"Command registered by extension \"{}\" must have a non-empty string name. Use pi.registerCommand(\"name\", {{ description, handler }}).",
				self.path
			);
		}
		let mut inner = self.inner.lock();
		let source_info = inner.extension.source_info.clone();
		inner.extension.commands.insert(
			name.to_string(),
			RegisteredCommand {
				name: name.to_string(),
				source_info,
				description: options.description,
				get_argument_completions: options.get_argument_completions,
				handler: options.handler,
			},
		);
	}

	fn register_shortcut(&self, shortcut: KeyId, options: ShortcutOptions) {
		self.assert_active();
		let mut inner = self.inner.lock();
		inner.extension.shortcuts.insert(
			shortcut.clone(),
			ExtensionShortcut {
				shortcut,
				description: options.description,
				handler: options.handler,
				extension_path: self.path.clone(),
			},
		);
	}

	fn register_flag(&self, name: &str, options: FlagOptions) {
		self.assert_active();
		let (description, flag_type, default) = match &options {
			FlagOptions::Boolean { description, default } => {
				(description.clone(), ExtensionFlagType::Boolean, default.map(FlagValue::Bool))
			}
			FlagOptions::String { description, default } => {
				(description.clone(), ExtensionFlagType::String, default.clone().map(FlagValue::Str))
			}
		};
		if let Some(default) = &default {
			let matches = match (default, flag_type) {
				(FlagValue::Bool(_), ExtensionFlagType::Boolean)
				| (FlagValue::Str(_), ExtensionFlagType::String) => true,
				_ => false,
			};
			if !matches {
				let expected = if flag_type == ExtensionFlagType::Boolean { "boolean" } else { "string" };
				let got = match default {
					FlagValue::Bool(_) => "boolean",
					FlagValue::Str(_) => "string",
				};
				panic!("Invalid default for flag \"{name}\": expected {expected}, got {got}");
			}
		}
		let mut inner = self.inner.lock();
		inner.extension.flags.insert(
			name.to_string(),
			ExtensionFlag {
				name: name.to_string(),
				description,
				flag_type,
				default: default.clone(),
				extension_path: self.path.clone(),
			},
		);
		if let Some(default) = default {
			if !self.runtime.state.has_flag_value(name) && !inner.pending_flag_values.contains_key(name) {
				inner.pending_flag_values.insert(name.to_string(), default);
			}
		}
	}

	fn get_flag(&self, name: &str) -> Option<FlagValue> {
		self.assert_active();
		let inner = self.inner.lock();
		if !inner.extension.flags.contains_key(name) {
			return None;
		}
		if let Some(value) = self.runtime.state.get_flag_value(name) {
			return Some(value.clone());
		}
		inner.pending_flag_values.get(name).cloned()
	}

	fn register_message_renderer(&self, custom_type: &str, renderer: MessageRenderer) {
		self.assert_active();
		self.inner.lock().extension.message_renderers.insert(custom_type.to_string(), renderer);
	}

	fn register_markdown_transformer(&self, transformer: MarkdownTransformer) {
		self.assert_active();
		self.inner.lock().extension.markdown_transformer = Some(transformer);
	}

	fn register_entry_renderer(&self, custom_type: &str, renderer: EntryRenderer) {
		self.assert_active();
		self.inner
			.lock()
			.extension
			.entry_renderers
			.get_or_insert_with(HashMap::new)
			.insert(custom_type.to_string(), renderer);
	}

	fn register_tool_renderer(&self, resolver: ToolRendererResolver) {
		self.assert_active();
		self.inner.lock().extension.tool_renderers.get_or_insert_with(Vec::new).push(resolver);
	}

	fn send_message(&self, _message: CustomMessageDraft, _options: Option<SendMessageOptions>) {
		panic!("{NOT_INITIALIZED}");
	}

	fn send_user_message(&self, _content: SendUserMessageContent, _options: Option<SendUserMessageOptions>) {
		panic!("{NOT_INITIALIZED}");
	}

	fn append_entry(&self, _custom_type: &str, _data: Option<JsonValue>) {
		panic!("{NOT_INITIALIZED}");
	}

	fn set_session_name(&self, _name: String) {
		panic!("{NOT_INITIALIZED}");
	}

	fn get_session_name(&self) -> Option<String> {
		panic!("{NOT_INITIALIZED}");
	}

	fn set_label(&self, _entry_id: &str, _label: Option<String>) {
		panic!("{NOT_INITIALIZED}");
	}

	fn exec(&self, _command: &str, _args: &[String], _options: Option<ExecOptions>) -> BoxFuture<'static, ExecResult> {
		self.assert_active();
		Box::pin(std::future::ready(ExecResult {
			stdout: String::new(),
			stderr: NOT_INITIALIZED.to_string(),
			code: 1,
			killed: false,
		}))
	}

	fn get_active_tools(&self) -> Vec<String> {
		panic!("{NOT_INITIALIZED}");
	}

	fn get_all_tools(&self) -> Vec<ToolInfo> {
		panic!("{NOT_INITIALIZED}");
	}

	fn get_settings(&self) -> Settings {
		panic!("{NOT_INITIALIZED}");
	}

	fn set_active_tools(&self, _tool_names: Vec<String>) {
		panic!("{NOT_INITIALIZED}");
	}

	fn get_commands(&self) -> Vec<SlashCommandInfo> {
		panic!("{NOT_INITIALIZED}");
	}

	fn set_model(&self, _model: Model) -> BoxFuture<'static, bool> {
		Box::pin(std::future::ready(false))
	}

	fn get_thinking_level(&self) -> AgentThinkingLevel {
		panic!("{NOT_INITIALIZED}");
	}

	fn set_thinking_level(&self, _level: AgentThinkingLevel) {
		panic!("{NOT_INITIALIZED}");
	}

	fn register_provider(&self, name: &str, config: ProviderConfig) {
		self.assert_active();
		self.queues.lock().providers.push(PendingProviderRegistration {
			name: name.to_string(),
			config,
			extension_path: self.path.clone(),
		});
	}

	fn register_native_provider(&self, provider: Arc<dyn AiProvider>) {
		self.assert_active();
		self.queues.lock().native_providers.push(PendingNativeProviderRegistration {
			provider: Some(provider),
			extension_path: self.path.clone(),
		});
	}

	fn unregister_provider(&self, name: &str) {
		self.assert_active();
		let mut queues = self.queues.lock();
		queues.providers.retain(|registration| registration.name != name);
		queues
			.native_providers
			.retain(|registration| !registration.provider.as_ref().is_some_and(|p| p.id() == name));
	}

	fn register_mcp_server(&self, _name: &str, _config: McpServerConfig) {
		self.assert_active();
		// `McpServerRegistry` is an opaque stand-in (D29); no-op until the
		// mcp-servers module lands.
	}

	fn unregister_mcp_server(&self, _name: &str) {
		self.assert_active();
	}

	fn get_mcp_servers(&self) -> Vec<RegisteredMcpServer> {
		self.assert_active();
		Vec::new()
	}

	fn register_virtual_model(&self, model: ExtensionVirtualModel) {
		self.assert_active();
		let ExtensionVirtualModel {
			provider,
			id,
			name,
			thinking_levels,
			context_window,
			max_tokens,
			input,
			route,
		} = model;
		self.queues.lock().virtual_models.push(PendingVirtualModelRegistration {
			definition: Some(VirtualModelDefinition {
				provider,
				id,
				name,
				thinking_levels,
				context_window,
				max_tokens,
				input,
				// The extension route is context-aware; the pending queue's
				// context-free definition carries a pre-bind stub (D43).
				route: {
					let _route = route;
					Arc::new(|_request: ModelRouteRequest| -> BoxFuture<'static, ModelRoute> {
						Box::pin(async { panic!("{NOT_INITIALIZED}") })
					})
				},
			}),
			extension_path: self.path.clone(),
		});
	}

	fn unregister_virtual_model(&self, provider: &str, id: &str) {
		self.assert_active();
		self.queues
			.lock()
			.virtual_models
			.retain(|registration| match &registration.definition {
				Some(definition) => definition.provider != provider || definition.id != id,
				None => true,
			});
	}

	fn events(&self) -> &dyn EventBus {
		self.assert_active();
		self.event_bus.as_ref()
	}
}

// ============================================================================
// Extension records and the load pipeline
// ============================================================================

/// Create an `Extension` object with empty collections (`createExtension`).
fn create_extension(extension_path: &str, resolved_path: &str) -> Extension {
	let source = get_synthetic_path_source(extension_path).unwrap_or_else(|| "local".to_string());
	let base_dir = if is_synthetic_path(extension_path) {
		None
	} else {
		std::path::Path::new(resolved_path)
			.parent()
			.and_then(|parent| parent.to_str())
			.map(str::to_string)
	};
	Extension {
		path: extension_path.to_string(),
		resolved_path: resolved_path.to_string(),
		hidden: None,
		replaceable: None,
		source_info: create_synthetic_source_info(extension_path, &source, None, None, base_dir.as_deref()),
		handlers: Arc::new(Mutex::new(HashMap::new())),
		tools: HashMap::new(),
		message_renderers: HashMap::new(),
		tool_renderers: None,
		markdown_transformer: None,
		entry_renderers: None,
		commands: HashMap::new(),
		flags: HashMap::new(),
		shortcuts: HashMap::new(),
	}
}

/// Result of one load step (`loadExtension`'s `{ extension, error }`).
struct LoadedExtension {
	extension: Option<Extension>,
	error: Option<String>,
}

/// Stand-in `loadExtension`: resolve the path, fail with the stand-in
/// missing-path error, reuse the cached factory entry by `(path, mtime)`,
/// and produce a stand-in `Extension` record — no JS execution (D43).
fn load_extension(
	extension_path: &str,
	cwd: &str,
	cache_token: Option<&CacheToken>,
) -> LoadedExtension {
	let resolved_path = resolve_path(extension_path, cwd);
	if !std::path::Path::new(&resolved_path).exists() {
		return LoadedExtension {
			extension: None,
			error: Some(format!("Extension path does not exist: {extension_path}")),
		};
	}

	let mtime = file_mtime(&resolved_path);
	if let Some(token) = cache_token {
		// A cache hit skips the (stand-in) module import, exactly like
		// upstream's cached-factory path.
		let _factory = cached_factory(&resolved_path, mtime, token);
		store_cache_entry(&resolved_path, mtime, None, token);
	}

	// Stand-in: no module to import, hence no factory to run — the record is
	// upstream's `Extension` shape with empty collections (D43).
	LoadedExtension { extension: Some(create_extension(extension_path, &resolved_path)), error: None }
}

/// Create an `Extension` from a factory function
/// (`loadExtensionFromFactory`). Builds the skeleton, invokes the factory
/// through an [`ExtensionApiHandle`], and maps a factory `Err` to
/// `Err(message)`.
pub async fn load_extension_from_factory(
	factory: ExtensionFactory,
	cwd: &str,
	event_bus: Arc<dyn EventBus>,
	runtime: Option<ExtensionRuntime>,
	extension_path: &str,
) -> Result<Extension, String> {
	let resolved_cwd = resolve_path_cwd(cwd);
	let (runtime, queues) = match runtime {
		Some(runtime) => (runtime, Arc::new(Mutex::new(RuntimeQueues::default()))),
		None => create_extension_runtime_shared(),
	};

	let extension = create_extension(extension_path, extension_path);
	let handlers = Arc::clone(&extension_handlers(&extension));
	let api = Arc::new(LoaderApi {
		inner: Mutex::new(ApiInner {
			extension,
			pending_flag_values: HashMap::new(),
			failed: false,
		}),
		handlers,
		path: extension_path.to_string(),
		runtime: runtime.clone(),
		queues: Arc::clone(&queues),
		event_bus,
	});
	let handle: ExtensionApiHandle = api.clone();

	// Factories panic on invalid registrations (upstream throws); map the
	// panic to the upstream error-string channel.
	let joined = tokio::spawn(factory(handle)).await;
	let factory_result = match joined {
		Ok(result) => result,
		Err(join_error) => {
			let message = match join_error.try_into_panic() {
				Ok(payload) => payload
					.downcast_ref::<String>()
					.cloned()
					.or_else(|| payload.downcast_ref::<&str>().map(|message| message.to_string()))
					.unwrap_or_else(|| "extension factory panicked".to_string()),
				Err(_) => "extension factory task cancelled".to_string(),
			};
			api.discard();
			return Err(message);
		}
	};

	match factory_result {
		Ok(()) => {
			let (committed, pending_flag_values) = api.commit();
			// Drain the API-owned queues into the loader-owned runtime
			// snapshots (upstream: the runner reads
			// `pendingProviderRegistrations` from the shared runtime).
			flush_queues(&queues, &api.runtime.state);
			for (name, value) in pending_flag_values {
				api.runtime.state.flag_values.write().entry(name).or_insert(value);
			}
			Ok(committed)
		}
		Err(error) => {
			api.discard();
			Err(error)
		}
	}
}

/// Load extensions from paths, reusing cached module factories
/// (`loadExtensionsCached`).
pub async fn load_extensions_cached(
	paths: &[String],
	cwd: &str,
	event_bus: Arc<dyn EventBus>,
	runtime: Option<ExtensionRuntime>,
) -> LoadExtensionsResult {
	let resolved_cwd = resolve_path_cwd(cwd);
	let cache_token = use_extension_cache_cwd(&resolved_cwd);
	let (mut runtime, queues) = match runtime {
		Some(runtime) => (runtime, Arc::new(Mutex::new(RuntimeQueues::default()))),
		None => create_extension_runtime_shared(),
	};

	let mut extensions = Vec::new();
	let mut errors = Vec::new();
	for extension_path in paths {
		let loaded = load_extension(extension_path, &resolved_cwd, Some(&cache_token));
		if let Some(error) = loaded.error {
			errors.push(LoadRecord { path: extension_path.clone(), error });
			continue;
		}
		if let Some(extension) = loaded.extension {
			extensions.push(extension);
		}
	}

	flush_queues(&queues, &runtime.state);
	LoadExtensionsResult { extensions, errors, warnings: None, runtime }
}

/// Inline extension constructor parity helper: an [`InlineExtension::Factory`]
/// is loaded through [`load_extension_from_factory`] with the `<inline>`
/// path, matching upstream's default `extensionPath`.
pub fn inline_extension_factory(inline: &InlineExtension) -> Option<&ExtensionFactory> {
	match inline {
		InlineExtension::Factory(factory) => Some(factory),
		InlineExtension::Named { factory, .. } => Some(factory),
	}
}

// ============================================================================
// Tests — stand-in lifecycle: cache behavior, missing-path errors, factory
// invocation success/failure.
// ============================================================================

#[cfg(test)]
mod tests {
	use super::*;
	use crate::core::extensions::types::EventBusHandler;

	struct TestEventBus {
		emitted: Mutex<Vec<(String, JsonValue)>>,
	}

	impl EventBus for TestEventBus {
		fn emit(&self, channel: &str, data: JsonValue) {
			self.emitted.lock().push((channel.to_string(), data));
		}
		fn on(&self, _channel: &str, _handler: EventBusHandler) -> Unsubscribe {
			Box::new(|| {})
		}
	}

	fn test_event_bus() -> Arc<dyn EventBus> {
		Arc::new(TestEventBus { emitted: Mutex::new(Vec::new()) })
	}

	fn ok_factory(register: bool) -> ExtensionFactory {
		Arc::new(move |api: ExtensionApiHandle| {
			Box::pin(async move {
				if register {
					api.register_tool(ToolDefinition::new(
						"echo",
						"Echo",
						"Echoes.",
						serde_json::json!({"type": "object"}),
					));
					api.register_provider("acme", ProviderConfig::default());
				}
				Ok(())
			}) as BoxFuture<'static, Result<(), String>>
		})
	}

	fn err_factory(message: &'static str) -> ExtensionFactory {
		Arc::new(move |_api: ExtensionApiHandle| {
			Box::pin(async move { Err(message.to_string()) }) as BoxFuture<'static, Result<(), String>>
		})
	}

	fn runtime() -> ExtensionRuntime {
		create_extension_runtime()
	}

	/// The cache registry is global — serialize cache-touching tests.
	static CACHE_TEST_LOCK: Mutex<()> = Mutex::new(());

	#[test]
	fn missing_path_yields_stand_in_error_record() {
		let _guard = CACHE_TEST_LOCK.lock();
		let result = tokio::runtime::Runtime::new().unwrap().block_on(load_extensions_cached(
			&["./nope/missing.ts".to_string()],
			".",
			test_event_bus(),
			Some(runtime()),
		));
		assert!(result.extensions.is_empty());
		assert_eq!(result.errors.len(), 1);
		assert_eq!(result.errors[0].path, "./nope/missing.ts");
		assert_eq!(result.errors[0].error, "Extension path does not exist: ./nope/missing.ts");
	}

	#[test]
	fn existing_path_produces_stand_in_record_and_cache_entry() {
		let _guard = CACHE_TEST_LOCK.lock();
		let dir = std::env::temp_dir().join(format!("pi-loader-test-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let file = dir.join("ext.ts");
		std::fs::write(&file, "export default () => {}").unwrap();
		let path = file.to_string_lossy().to_string();

		let first = tokio::runtime::Runtime::new().unwrap().block_on(load_extensions_cached(
			&[path.clone()],
			&dir.to_string_lossy(),
			test_event_bus(),
			Some(runtime()),
		));
		assert!(first.errors.is_empty());
		assert_eq!(first.extensions.len(), 1);
		assert_eq!(first.extensions[0].path, path);
		assert!(first.extensions[0].tools.is_empty());
		assert!(first.runtime.pending_providers().is_empty());

		// The entry is cached under (path, mtime).
		assert!(EXTENSION_CACHE.lock().entries.contains_key(&path));

		std::fs::remove_dir_all(&dir).ok();
	}

	#[test]
	fn clear_extension_cache_bumps_generation_and_clears_entries() {
		let _guard = CACHE_TEST_LOCK.lock();
		let dir = std::env::temp_dir().join(format!("pi-loader-gen-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let path = dir.join("ext.ts").to_string_lossy().to_string();
		std::fs::write(&path, "export default () => {}").unwrap();
		let _ = tokio::runtime::Runtime::new().unwrap().block_on(load_extensions_cached(
			&[path.clone()],
			&dir.to_string_lossy(),
			test_event_bus(),
			Some(runtime()),
		));

		let before = EXTENSION_CACHE.lock().generation;
		clear_extension_cache();
		{
			let cache = EXTENSION_CACHE.lock();
			assert!(cache.entries.is_empty());
			assert_eq!(cache.generation, before + 1);
		}

		std::fs::remove_dir_all(&dir).ok();
	}

	#[test]
	fn factory_success_collects_registrations() {
		let extension = tokio::runtime::Runtime::new().unwrap().block_on(load_extension_from_factory(
			ok_factory(true),
			".",
			test_event_bus(),
			None,
			"<inline>",
		))
		.unwrap();

		assert_eq!(extension.path, "<inline>");
		assert_eq!(extension.resolved_path, "<inline>");
		assert!(extension.tools.contains_key("echo"));
		assert_eq!(extension.tools["echo"].source_info.path, "<inline>");
	}

	#[test]
	fn factory_failure_maps_to_err() {
		let error = tokio::runtime::Runtime::new().unwrap().block_on(load_extension_from_factory(
			err_factory("boom"),
			".",
			test_event_bus(),
			None,
			"<inline>",
		))
		.unwrap_err();
		assert_eq!(error, "boom");
	}

	#[test]
	fn register_tool_rejects_non_object_schema() {
		let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
			let rt = tokio::runtime::Runtime::new().unwrap();
			rt.block_on(load_extension_from_factory(
				Arc::new(|api: ExtensionApiHandle| {
					Box::pin(async move {
						api.register_tool(ToolDefinition::new("bad", "Bad", "Bad.", serde_json::json!("nope")));
						Ok(())
					}) as BoxFuture<'static, Result<(), String>>
				}),
				".",
				test_event_bus(),
				None,
				"<inline>",
			))
		}));
		let error = result
			.ok()
			.and_then(|outcome| outcome.err())
			.expect("non-object parameter schema must fail the load");
		assert_eq!(
			error,
			"Tool \"bad\" registered by extension \"<inline>\" must define an object parameter schema."
		);
	}

	#[test]
	fn action_methods_panic_before_runner_binds() {
		let runtime = runtime();
		let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
			(runtime.state.create_context)();
		}));
		assert!(result.is_err());
		assert!((runtime.state.assert_active)().is_ok());
		(runtime.state.invalidate)(None);
		assert!((runtime.state.assert_active)().is_err());
	}

	#[test]
	fn runtime_tracks_event_bus_subscriptions_until_invalidation() {
		let runtime = runtime();
		let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
		let flag = Arc::clone(&fired);
		let unsubscribe = (runtime.state.track_event_bus_subscription)(Box::new(move || {
			flag.store(true, std::sync::atomic::Ordering::SeqCst);
		}));
		unsubscribe();
		assert!(fired.load(std::sync::atomic::Ordering::SeqCst));

		// A subscription still active when `invalidate` runs is fired by it.
		let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
		let flag = Arc::clone(&fired);
		let _tracked = (runtime.state.track_event_bus_subscription)(Box::new(move || {
			flag.store(true, std::sync::atomic::Ordering::SeqCst);
		}));
		assert!((runtime.state.assert_active)().is_ok());
		(runtime.state.invalidate)(Some("stale".to_string()));
		assert!((runtime.state.assert_active)().is_err());
		assert!(fired.load(std::sync::atomic::Ordering::SeqCst));
	}

	#[test]
	fn inline_extension_factory_extracts_both_variants() {
		assert!(inline_extension_factory(&InlineExtension::Factory(ok_factory(false))).is_some());
		assert!(inline_extension_factory(&InlineExtension::Named {
			name: "named".to_string(),
			factory: ok_factory(false),
			hidden: None,
			replaceable: None,
			builtin: None,
		})
		.is_some());
	}
}
