//! Port of `packages/coding-agent/src/core/extensions/loader.ts` — the
//! extension loader.
//!
//! Upstream loads TypeScript/JavaScript extension modules through `jiti` and
//! invokes their default-exported factory with a `pi` API object. The
//! host-side shell landed first as a stand-in (D43); the module seam is now
//! bridged to pi-runtime's embedded deno_core host (D107): file-backed
//! `.ts`/`.js` extensions are imported by [`pi_runtime::extensions::DenoExtensionHost`]
//! and their factory's registration calls route back into the same loader
//! API inline factories use, so file extensions produce real
//! [`Extension`] records (handlers, tools, commands, flags, shortcuts).
//!
//! ## D43 — host-side shell (pre-bridge)
//!
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
//!   does not exist yields `Extension path does not exist: <path>` (upstream
//!   surfaces the failure as `Failed to load extension: Cannot find module …`
//!   from the jiti import; the port detects it earlier with the same
//!   `{ path, error }` record shape).
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
//! ## D107 — the module seam bridges to the JS host
//!
//! - A cache miss imports the file through the shared [`DenoExtensionHost`]
//!   (`ExtensionHost::load`) and runs its default-exported factory. The JS
//!   `pi` proxy's registration calls (`on`, `registerTool`,
//!   `registerCommand`, `registerShortcut`, `registerFlag`,
//!   `registerProvider`, `unregisterProvider`, `registerMcpServer`,
//!   `unregisterMcpServer`) cross the JSON boundary as slot references and
//!   route into the loader API, so the resulting [`Extension`] record is
//!   indistinguishable from an inline factory's. A cache hit re-runs the
//!   stored factory via `ExtensionHost::reload` without re-importing the
//!   module (upstream's jiti `moduleCache: false` + path-keyed factory
//!   cache).
//! - Load failures wrap as `Failed to load extension: <message>` like
//!   upstream's `loadExtension` catch.
//! - Function- or component-valued registrations cannot cross the JSON
//!   boundary and error the load: renderer registrations (TUI `Component`
//!   pipeline, D56), `registerVirtualModel`/`registerNativeProvider`
//!   (context-bound route/AbortSignal values, runner-bind work), and
//!   `getArgumentCompletions`/`prepareArguments`/`namespace`/`annotations`.
//!   Tool `execute` maps the upstream `AgentToolResult` JSON shape; the
//!   abort signal / update callback / tool context arguments arrive `None`
//!   until the runner binds them.
//! - Action methods from JS (`sendMessage`, `getSettings`, …) report
//!   upstream's "not initialized" error pre-bind; `getFlag` works (it reads
//!   load state). The `pi.events` bus property and `ctx.*` proxies route
//!   through the same channel and land with the runner-bind work.
//! - The bare `typebox` specifier resolves to a minimal schema shim (the
//!   embedded runtime has no npm installs; D108).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;
use std::time::SystemTime;

use indexmap::IndexMap;
use parking_lot::Mutex;
use pi_agent::{AgentThinkingLevel, AgentToolResult, BoxFuture, ToolExecutionMode};
use pi_ai::types::Model;
use pi_ai::Provider as AiProvider;
use pi_runtime::extensions::{DenoExtensionHost, DenoExtensionHostOptions, ExtensionBackend, ExtensionHost, ExtensionHostError};
use serde_json::Value as JsonValue;

use super::types::{
	CustomMessageDraft, EntryRenderer, EventBus,
	EventBusHandler, ExecOptions, ExecResult, Extension, ExtensionApi, ExtensionApiHandle, ExtensionContext, ExtensionFactory,
	ExtensionFlag, ExtensionFlagType, ExtensionRuntime, ExtensionRuntimeState, ExtensionShortcut, ExtensionToolContext,
	ExtensionVirtualModel, FlagOptions, FlagValue, HandlerFn, InlineExtension, KeyId, LoadExtensionsResult,
	LoadRecord, MarkdownTransformer, McpServerConfig, McpServerRegistry, MessageRenderer, ModelRoute,
	ModelRouteRequest, PendingNativeProviderRegistration, PendingProviderRegistration,
	PendingVirtualModelRegistration, ProviderConfig, RegisteredCommand, RegisteredCommandOptions,
	RegisteredMcpServer, RegisteredTool, SendMessageOptions, SendUserMessageContent, SendUserMessageOptions,
	ShortcutOptions, SlashCommandInfo, ToolDefinition, ToolExecute, ToolExposure, ToolInfo, ToolRendererResolver, Unsubscribe,
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

/// Test-only gate serializing global extension-cache mutation (tests) with
/// cache-reading tests in this module (`clearExtensionCache` may run at any
/// time upstream; the ported tests keep the assertions deterministic).
#[cfg(test)]
pub(crate) fn cache_test_gate() -> parking_lot::MutexGuard<'static, ()> {
	static CACHE_TEST_LOCK: Mutex<()> = Mutex::new(());
	CACHE_TEST_LOCK.lock()
}

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
// JS host bridge (D107): file-backed extension modules load through
// pi_runtime's embedded deno_core host. Registration calls from JS cross the
// JSON boundary as slot references (`{"__piSlot": n}`) and route into the
// same loader API inline factories use.
// ============================================================================

/// Bare-specifier JS shim for the `typebox` schema toolkit (upstream
/// extensions import it as a real npm dependency; the embedded runtime has no
/// npm installs — D108). Covers the schema combinators ported extensions use.
const TYPEBOX_SHIM: &str = r#"
const Optional = (schema) => schema;
export const Type = {
	Unknown: () => ({}),
	Any: () => ({}),
	Optional,
	String: (options) => ({ type: "string", ...(options ?? {}) }),
	Number: (options) => ({ type: "number", ...(options ?? {}) }),
	Boolean: (options) => ({ type: "boolean", ...(options ?? {}) }),
	Literal: (value) => ({ const: value }),
	Array: (items, options) => ({ type: "array", items, ...(options ?? {}) }),
	Object: (properties, options) => ({
		type: "object",
		properties: properties ?? {},
		...(options ?? {}),
	}),
	Record: (_keys, values) => ({ type: "object", additionalProperties: values }),
	Union: (options) => ({ anyOf: options ?? [] }),
};
export default { Type };
"#;

/// One live bridge binding: the loader API a host key's JS registration calls
/// route into, plus the unsubscribes its `on` calls returned.
struct BridgeTarget {
	extension_path: String,
	api: ExtensionApiHandle,
	unsubscribers: Mutex<HashMap<u32, Unsubscribe>>,
}

/// Live bridge bindings (host key → target). The host serializes requests on
/// its dedicated thread, so a key has at most one binding at a time.
static BRIDGE_TARGETS: LazyLock<Mutex<HashMap<String, Arc<BridgeTarget>>>> = LazyLock::new(Mutex::default);

static BRIDGE_KEY_SEQ: AtomicU64 = AtomicU64::new(0);

fn bridge_next_key() -> String {
	format!("bridge-{}", BRIDGE_KEY_SEQ.fetch_add(1, Ordering::Relaxed))
}

fn bridge_bind_target(key: &str, extension_path: &str, api: ExtensionApiHandle) {
	BRIDGE_TARGETS.lock().insert(
		key.to_string(),
		Arc::new(BridgeTarget {
			extension_path: extension_path.to_string(),
			api,
			unsubscribers: Mutex::new(HashMap::new()),
		}),
	);
}

fn bridge_unbind_target(key: &str) {
	BRIDGE_TARGETS.lock().remove(key);
}

/// Test-only module-import counter (resolved path → host `load` count).
/// Upstream counts module-level side effects through a `globalThis` key; the
/// embedded host exposes no eval seam, so the bridge counts its own imports —
/// the same observable (one import per path per cache generation).
#[cfg(test)]
static BRIDGE_IMPORT_COUNTS: LazyLock<Mutex<HashMap<String, u32>>> = LazyLock::new(Mutex::default);

/// Read [`BRIDGE_IMPORT_COUNTS`] for one resolved path.
#[cfg(test)]
pub(crate) fn bridge_import_count(resolved_path: &str) -> u32 {
	BRIDGE_IMPORT_COUNTS.lock().get(resolved_path).copied().unwrap_or(0)
}

fn bridge_target(key: &str) -> Result<Arc<BridgeTarget>, String> {
	BRIDGE_TARGETS
		.lock()
		.get(key)
		.cloned()
		.ok_or_else(|| format!("No extension is loading under host key {key}"))
}

/// Run a loader-API call arriving from JS, mapping a Rust panic (upstream
/// throws) to an op error so the JS factory rejects with the same message.
fn bridge_catch<R>(f: impl FnOnce() -> R) -> Result<R, String> {
	std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|payload| {
		payload
			.downcast_ref::<String>()
			.cloned()
			.or_else(|| payload.downcast_ref::<&str>().map(|message| message.to_string()))
			.unwrap_or_else(|| "extension registration failed".to_string())
	})
}

/// The `{"__piSlot": n}` slot reference of a slotified function argument.
fn bridge_slot(value: Option<&JsonValue>) -> Option<u32> {
	value
		.and_then(|value| value.get("__piSlot"))
		.and_then(JsonValue::as_u64)
		.and_then(|slot| u32::try_from(slot).ok())
}

/// Upstream `typeof` names for flag-default mismatch errors.
fn bridge_json_type(value: &JsonValue) -> &'static str {
	match value {
		JsonValue::Bool(_) => "boolean",
		JsonValue::Number(_) => "number",
		JsonValue::String(_) => "string",
		JsonValue::Array(_) | JsonValue::Object(_) | JsonValue::Null => "object",
	}
}

/// Event handler crossing the bridge: the payload serializes into the host
/// slot and the JSON result comes back (the `ctx` argument stays on the JS
/// side, which supplies its own live proxy).
fn bridge_event_handler(host: Arc<dyn ExtensionHost>, key: String, slot: u32) -> HandlerFn {
	Arc::new(move |event: JsonValue, _ctx: ExtensionContext| {
		let (host, key) = (Arc::clone(&host), key.clone());
		Box::pin(async move { host.call_slot(&key, slot, event).await.map_err(|error| error.message) })
	})
}

/// Command handler crossing the bridge (`handler(args, ctx)`); handler
/// failures log like upstream's unhandled rejection.
fn bridge_command_handler(
	host: Arc<dyn ExtensionHost>,
	key: String,
	slot: u32,
) -> super::types::CommandHandler {
	Arc::new(move |args: String, _ctx: super::types::ExtensionCommandContext| {
		let (host, key) = (Arc::clone(&host), key.clone());
		Box::pin(async move {
			if let Err(error) = host.call_slot(&key, slot, JsonValue::String(args)).await {
				eprintln!("pi: extension command handler failed: {}", error.message);
			}
		})
	})
}

/// Shortcut handler crossing the bridge (`handler(ctx)`).
fn bridge_shortcut_handler(
	host: Arc<dyn ExtensionHost>,
	key: String,
	slot: u32,
) -> Arc<dyn Fn(ExtensionContext) -> BoxFuture<'static, ()> + Send + Sync> {
	Arc::new(move |_ctx: ExtensionContext| {
		let (host, key) = (Arc::clone(&host), key.clone());
		Box::pin(async move {
			if let Err(error) = host.call_slot(&key, slot, JsonValue::Null).await {
				eprintln!("pi: extension shortcut handler failed: {}", error.message);
			}
		})
	})
}

/// Tool `execute` crossing the bridge; the abort signal, update callback, and
/// tool-context arguments arrive `None` until the runner binds them (D107).
fn bridge_tool_execute(host: Arc<dyn ExtensionHost>, key: String, slot: u32) -> ToolExecute {
	Arc::new(
		move |_tool_call_id: String, args: JsonValue, _signal, _on_update, _ctx: Option<ExtensionToolContext>| {
			let (host, key) = (Arc::clone(&host), key.clone());
			Box::pin(async move {
				let value = host.call_slot(&key, slot, args).await.map_err(|error| error.message)?;
				bridge_tool_result(value)
			})
		},
	)
}

/// Map a JS execute result onto [`AgentToolResult`] (upstream
/// `AgentToolResult` object shape; other shapes error the tool call).
fn bridge_tool_result(value: Option<JsonValue>) -> Result<AgentToolResult, String> {
	let Some(object) = value.filter(|value| !value.is_null()) else {
		return Ok(AgentToolResult::new(Vec::new(), JsonValue::Null));
	};
	if !object.is_object() {
		return Err("Tool execute must return an AgentToolResult object.".to_string());
	}
	let content = match object.get("content") {
		Some(value) => serde_json::from_value::<Vec<pi_ai::types::UserContent>>(value.clone())
			.map_err(|error| format!("Tool result content is not valid content: {error}"))?,
		None => Vec::new(),
	};
	Ok(AgentToolResult {
		content,
		details: object.get("details").cloned().unwrap_or(JsonValue::Null),
		structured_content: object.get("structuredContent").filter(|value| !value.is_null()).cloned(),
		usage: object
			.get("usage")
			.filter(|value| !value.is_null())
			.and_then(|value| serde_json::from_value(value.clone()).ok()),
		is_error: object.get("isError").and_then(JsonValue::as_bool),
		terminate: object.get("terminate").and_then(JsonValue::as_bool),
	})
}

/// Rust side of the bridge ([`ExtensionBackend`]): routes JS `pi` calls into
/// the loader API bound to the extension's host key.
struct BridgeBackend;

/// `registerCommand`-style methods arrive as a JSON argument array.
fn bridge_args(payload: &JsonValue, method: &str) -> Result<Vec<JsonValue>, String> {
	let Ok(args) = serde_json::from_value::<Vec<JsonValue>>(payload.clone()) else {
		return Err(format!("{method} registration must carry an argument array."));
	};
	Ok(args)
}

#[async_trait::async_trait]
impl ExtensionBackend for BridgeBackend {
	async fn call(&self, ext: &str, method: &str, args: JsonValue) -> Result<JsonValue, String> {
		let target = bridge_target(ext)?;
		match method {
			"getFlag" => {
				let name = args.as_str().ok_or("getFlag requires a string name")?;
				let value = bridge_catch(|| target.api.get_flag(name))?;
				Ok(value
					.map(|value| serde_json::to_value(value).unwrap_or(JsonValue::Null))
					.unwrap_or(JsonValue::Null))
			}
			_ => Err(NOT_INITIALIZED.to_string()),
		}
	}

	fn register(&self, ext: &str, method: &str, payload: JsonValue) -> Result<JsonValue, String> {
		let target = bridge_target(ext)?;
		match method {
			"on" => {
				let event = payload
					.get("event")
					.and_then(JsonValue::as_str)
					.ok_or("on() requires an event name")?
					.to_string();
				let slot = bridge_slot(payload.get("handler"))
					.ok_or_else(|| format!("Event handler for \"{event}\" must be a function."))?;
				let host = extension_host();
				let key = ext.to_string();
				let unsubscribe = bridge_catch(|| target.api.on(&event, bridge_event_handler(host, key, slot)))?;
				target.unsubscribers.lock().insert(slot, unsubscribe);
				Ok(JsonValue::Null)
			}
			"unsubscribe" => {
				let slot =
					bridge_slot(payload.get("handler")).ok_or("unsubscribe() requires a handler slot")?;
				if let Some(unsubscribe) = target.unsubscribers.lock().remove(&slot) {
					bridge_catch(move || unsubscribe())?;
				}
				Ok(JsonValue::Null)
			}
			"registerTool" => {
				let name = payload
					.get("name")
					.and_then(JsonValue::as_str)
					.filter(|name| !name.is_empty())
					.ok_or_else(|| {
						format!("Tool registered by extension \"{}\" must have a non-empty string name.", target.extension_path)
					})?
					.to_string();
				let label = payload.get("label").and_then(JsonValue::as_str).unwrap_or(&name).to_string();
				let description =
					payload.get("description").and_then(JsonValue::as_str).unwrap_or_default().to_string();
				let parameters = payload.get("parameters").cloned().unwrap_or(JsonValue::Null);
				let mut tool = ToolDefinition::new(name.clone(), label, description, parameters);
				tool.prompt_snippet = payload.get("promptSnippet").and_then(JsonValue::as_str).map(str::to_string);
				tool.prompt_guidelines = payload
					.get("promptGuidelines")
					.and_then(JsonValue::as_array)
					.map(|items| items.iter().filter_map(JsonValue::as_str).map(str::to_string).collect::<Vec<_>>());
				tool.output_schema = payload.get("outputSchema").filter(|value| !value.is_null()).cloned();
				tool.exposure = match payload.get("exposure").filter(|value| !value.is_null()) {
					Some(value) => Some(serde_json::from_value::<ToolExposure>(value.clone())
						.map_err(|error| format!("Invalid exposure for tool \"{name}\": {error}"))?),
					None => None,
				};
				tool.execution_mode = match payload.get("executionMode").and_then(JsonValue::as_str) {
					Some("sequential") => Some(ToolExecutionMode::Sequential),
					Some("parallel") => Some(ToolExecutionMode::Parallel),
					Some(other) => {
						return Err(format!("Invalid executionMode \"{other}\" for tool \"{name}\"."))
					}
					None => None,
				};
				tool.default_active = payload.get("defaultActive").and_then(JsonValue::as_bool);
				if let Some(slot) = bridge_slot(payload.get("execute")) {
					tool.execute = bridge_tool_execute(extension_host(), ext.to_string(), slot);
				}
				bridge_catch(move || target.api.register_tool(tool))?;
				Ok(JsonValue::Null)
			}
			"registerCommand" => {
				let args = bridge_args(&payload, method)?;
				let name = args
					.first()
					.and_then(JsonValue::as_str)
					.ok_or_else(|| {
						format!(
							"Command registered by extension \"{}\" must have a non-empty string name. Use pi.registerCommand(\"name\", {{ description, handler }}).",
							target.extension_path
						)
					})?
					.to_string();
				let options = args.get(1).cloned().unwrap_or(JsonValue::Null);
				let slot = bridge_slot(options.get("handler")).ok_or_else(|| {
					format!("Command \"/{name}\" registered by extension \"{}\" must define handler().", target.extension_path)
				})?;
				let description = options.get("description").and_then(JsonValue::as_str).map(str::to_string);
				let host = extension_host();
				let key = ext.to_string();
				let handler = bridge_command_handler(host, key, slot);
				bridge_catch(move || {
					target
						.api
						.register_command(&name, RegisteredCommandOptions { description, get_argument_completions: None, handler })
				})?;
				Ok(JsonValue::Null)
			}
			"registerShortcut" => {
				let args = bridge_args(&payload, method)?;
				let shortcut = args
					.first()
					.and_then(JsonValue::as_str)
					.ok_or("registerShortcut requires a key id")?
					.to_string();
				let options = args.get(1).cloned().unwrap_or(JsonValue::Null);
				let slot = bridge_slot(options.get("handler"))
					.ok_or_else(|| format!("Shortcut \"{shortcut}\" must define handler()."))?;
				let description = options.get("description").and_then(JsonValue::as_str).map(str::to_string);
				let host = extension_host();
				let key = ext.to_string();
				let handler = bridge_shortcut_handler(host, key, slot);
				bridge_catch(move || {
					target.api.register_shortcut(shortcut.clone(), ShortcutOptions { description, handler })
				})?;
				Ok(JsonValue::Null)
			}
			"registerFlag" => {
				let args = bridge_args(&payload, method)?;
				let name = args
					.first()
					.and_then(JsonValue::as_str)
					.ok_or("registerFlag requires a flag name")?
					.to_string();
				let options = args.get(1).cloned().unwrap_or(JsonValue::Null);
				let flag_type = options
					.get("type")
					.and_then(JsonValue::as_str)
					.ok_or_else(|| format!("Flag \"{name}\" must declare type \"boolean\" or \"string\"."))?;
				let description = options.get("description").and_then(JsonValue::as_str).map(str::to_string);
				let default = options.get("default").filter(|value| !value.is_null());
				let flag_options = match flag_type {
					"boolean" => {
						if let Some(value) = default {
							if !value.is_boolean() {
								return Err(format!(
									"Invalid default for flag \"{name}\": expected boolean, got {}",
									bridge_json_type(value)
								));
							}
						}
						FlagOptions::Boolean { description, default: default.and_then(JsonValue::as_bool) }
					}
					"string" => {
						if let Some(value) = default {
							if !value.is_string() {
								return Err(format!(
									"Invalid default for flag \"{name}\": expected string, got {}",
									bridge_json_type(value)
								));
							}
						}
						FlagOptions::String { description, default: default.and_then(JsonValue::as_str).map(str::to_string) }
					}
					other => {
						return Err(format!(
							"Invalid type \"{other}\" for flag \"{name}\": expected \"boolean\" or \"string\"."
						))
					}
				};
				bridge_catch(move || target.api.register_flag(&name, flag_options))?;
				Ok(JsonValue::Null)
			}
			"registerProvider" => {
				let args = bridge_args(&payload, method)?;
				let name = args
					.first()
					.and_then(JsonValue::as_str)
					.ok_or("registerProvider requires a provider name")?
					.to_string();
				let config: ProviderConfig =
					serde_json::from_value(args.get(1).cloned().unwrap_or(JsonValue::Null))
						.map_err(|error| format!("Invalid provider config for \"{name}\": {error}"))?;
				bridge_catch(move || target.api.register_provider(&name, config))?;
				Ok(JsonValue::Null)
			}
			"unregisterProvider" => {
				let name = payload.as_str().ok_or("unregisterProvider requires a provider name")?;
				bridge_catch(|| target.api.unregister_provider(name))?;
				Ok(JsonValue::Null)
			}
			"registerMcpServer" => {
				let args = bridge_args(&payload, method)?;
				let name = args
					.first()
					.and_then(JsonValue::as_str)
					.ok_or("registerMcpServer requires a server name")?
					.to_string();
				let config = McpServerConfig(args.get(1).cloned().unwrap_or(JsonValue::Null));
				bridge_catch(move || target.api.register_mcp_server(&name, config))?;
				Ok(JsonValue::Null)
			}
			"unregisterMcpServer" => {
				let name = payload.as_str().ok_or("unregisterMcpServer requires a server name")?;
				bridge_catch(|| target.api.unregister_mcp_server(name))?;
				Ok(JsonValue::Null)
			}
			// Function- or component-valued registrations cannot cross the
			// JSON boundary (D107).
			"registerMessageRenderer" | "registerEntryRenderer" | "registerMarkdownTransformer"
			| "registerToolRenderer" => Err(
				"Renderer registration from file-backed extensions is not supported over the JSON bridge (D107): the renderer pipeline is TUI-bound (D56)."
					.to_string(),
			),
			"registerVirtualModel" => Err(
				"registerVirtualModel from file-backed extensions is not supported over the JSON bridge (D107): the route callback carries a live context (runner-bind work)."
					.to_string(),
			),
			"registerNativeProvider" => Err(
				"registerProvider(provider) from file-backed extensions is not supported over the JSON bridge (D107): native provider objects cannot cross the JSON boundary."
					.to_string(),
			),
			other => Err(format!("Unknown extension API method \"{other}\".")),
		}
	}
}

/// The shared bridge host: one embedded JS runtime per process; loads
/// serialize on its dedicated thread. Registers the `typebox` shim (D108).
static EXTENSION_HOST: LazyLock<Arc<dyn ExtensionHost>> = LazyLock::new(|| {
	let options = DenoExtensionHostOptions {
		virtual_sources: vec![("typebox".to_string(), TYPEBOX_SHIM.to_string())],
		..Default::default()
	};
	Arc::new(DenoExtensionHost::new(Arc::new(BridgeBackend), options))
});

fn extension_host() -> Arc<dyn ExtensionHost> {
	Arc::clone(&*EXTENSION_HOST)
}

/// Import `resolved_path` through the JS host and run its default-exported
/// factory against a fresh loader API. Returns the extension plus the cached
/// factory that re-runs the same JS factory without re-importing the module
/// (upstream `extensionCache.set(path, factory)`).
async fn load_extension_via_host(
	resolved_path: &str,
	extension_path: &str,
	event_bus: Arc<dyn EventBus>,
	runtime: Option<ExtensionRuntime>,
) -> Result<(Extension, ExtensionFactory), String> {
	let host = extension_host();
	let source_key = bridge_next_key();
	let load_factory: ExtensionFactory = {
		let host = Arc::clone(&host);
		let entry = resolved_path.to_string();
		let source_key = source_key.clone();
		let extension_path = extension_path.to_string();
		Arc::new(move |api: ExtensionApiHandle| {
			let host = Arc::clone(&host);
			let (entry, key, extension_path) = (entry.clone(), source_key.clone(), extension_path.clone());
			Box::pin(async move {
				bridge_bind_target(&key, &extension_path, api);
				let result = host.load(Path::new(&entry), &key).await;
				bridge_unbind_target(&key);
				result.map(|_| ()).map_err(|error: ExtensionHostError| error.message)
			}) as BoxFuture<'static, Result<(), String>>
		})
	};
	// Count the module import (upstream counts module-level side effects;
	// the host exposes no eval seam, so the bridge counts its own imports).
	#[cfg(test)]
	{
		*BRIDGE_IMPORT_COUNTS.lock().entry(resolved_path.to_string()).or_insert(0) += 1;
	}
	let extension = initialize_extension(load_factory, extension_path, resolved_path, event_bus, runtime).await?;
	let cached_factory: ExtensionFactory = {
		let host = Arc::clone(&host);
		let source_key = source_key.clone();
		let extension_path = extension_path.to_string();
		Arc::new(move |api: ExtensionApiHandle| {
			let host = Arc::clone(&host);
			let (source_key, key, extension_path) =
				(source_key.clone(), bridge_next_key(), extension_path.clone());
			Box::pin(async move {
				bridge_bind_target(&key, &extension_path, api);
				let result = host.reload(&source_key, &key).await;
				bridge_unbind_target(&key);
				result.map(|_| ()).map_err(|error: ExtensionHostError| error.message)
			}) as BoxFuture<'static, Result<(), String>>
		})
	};
	Ok((extension, cached_factory))
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
		tools: IndexMap::new(),
		message_renderers: HashMap::new(),
		tool_renderers: None,
		markdown_transformer: None,
		entry_renderers: None,
		commands: IndexMap::new(),
		flags: IndexMap::new(),
		shortcuts: IndexMap::new(),
	}
}

/// Result of one load step (`loadExtension`'s `{ extension, error }`).
struct LoadedExtension {
	extension: Option<Extension>,
	error: Option<String>,
}

/// `loadExtension`: resolve the path, fail early on a missing path, reuse
/// the cached factory entry by `(path, mtime)`, and otherwise import the
/// module through the JS host bridge (D107).
async fn load_extension(
	extension_path: &str,
	cwd: &str,
	event_bus: Arc<dyn EventBus>,
	runtime: Option<ExtensionRuntime>,
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
		// A cache hit re-runs the cached factory without re-importing the
		// module (upstream `loadExtensionModule`'s cached-factory path).
		if let Some(factory) = cached_factory(&resolved_path, mtime, token) {
			return match initialize_extension(factory, extension_path, &resolved_path, event_bus, runtime).await {
				Ok(extension) => LoadedExtension { extension: Some(extension), error: None },
				Err(message) => LoadedExtension {
					extension: None,
					error: Some(format!("Failed to load extension: {message}")),
				},
			};
		}
	}

	// Cache miss: import the module through the JS host and run its
	// default-exported factory (D107).
	match load_extension_via_host(&resolved_path, extension_path, event_bus, runtime).await {
		Ok((extension, cached_factory)) => {
			if let Some(token) = cache_token {
				store_cache_entry(&resolved_path, mtime, Some(cached_factory), token);
			}
			LoadedExtension { extension: Some(extension), error: None }
		}
		Err(message) => LoadedExtension {
			extension: None,
			error: Some(format!("Failed to load extension: {message}")),
		},
	}
}

/// Invoke `factory` against a fresh loader API and collect the resulting
/// extension (`initializeExtension`). A factory `Err` — or panic — maps to
/// `Err(message)` and marks the API failed (`discard`).
async fn initialize_extension(
	factory: ExtensionFactory,
	extension_path: &str,
	resolved_path: &str,
	event_bus: Arc<dyn EventBus>,
	runtime: Option<ExtensionRuntime>,
) -> Result<Extension, String> {
	let (runtime, queues) = match runtime {
		Some(runtime) => (runtime, Arc::new(Mutex::new(RuntimeQueues::default()))),
		None => create_extension_runtime_shared(),
	};

	let extension = create_extension(extension_path, resolved_path);
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
	// Upstream resolves the cwd and passes it as initializeExtension's `cwd`;
	// the port's loader API carries no cwd (D43 scope note).
	let _resolved_cwd = resolve_path_cwd(cwd);
	initialize_extension(factory, extension_path, extension_path, event_bus, runtime).await
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
	let (runtime, queues) = match runtime {
		Some(runtime) => (runtime, Arc::new(Mutex::new(RuntimeQueues::default()))),
		None => create_extension_runtime_shared(),
	};

	let mut extensions = Vec::new();
	let mut errors = Vec::new();
	for extension_path in paths {
		let loaded = load_extension(
			extension_path,
			&resolved_cwd,
			Arc::clone(&event_bus),
			Some(runtime.clone()),
			Some(&cache_token),
		)
		.await;
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


	#[test]
	fn missing_path_yields_stand_in_error_record() {
		let _guard = cache_test_gate();
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
		let _guard = cache_test_gate();
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

		// The entry is cached under (path, mtime). Any parallel test loading
		// extensions from a different cwd legitimately clears the global
		// cache (`useExtensionCacheCwd`), so re-load and re-check a few
		// times instead of asserting against racing clears.
		let mut cached = EXTENSION_CACHE.lock().entries.contains_key(&path);
		for _ in 0..50 {
			if cached {
				break;
			}
			std::thread::sleep(std::time::Duration::from_millis(10));
			let _ = tokio::runtime::Runtime::new().unwrap().block_on(load_extensions_cached(
				&[path.clone()],
				&dir.to_string_lossy(),
				test_event_bus(),
				Some(runtime()),
			));
			cached = EXTENSION_CACHE.lock().entries.contains_key(&path);
		}
		assert!(cached);

		std::fs::remove_dir_all(&dir).ok();
	}

	#[test]
	fn clear_extension_cache_bumps_generation_and_clears_entries() {
		let _guard = cache_test_gate();
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

#[cfg(test)]
mod runtime_freshness_probe {
	// Probe: two sequential cached loads sharing one factory must yield two
	// independent `Extension`s whose handlers both dispatch.
	use crate::core::resource_loader::{DefaultResourceLoader, ResourceLoaderReloadOptions};
	use super::*;
	use crate::core::extensions::runner::ExtensionRunner;
	use crate::core::extensions::types::ExtensionContext;
	use std::sync::atomic::{AtomicU32, Ordering};

	struct ProbeBus;
	impl EventBus for ProbeBus {
		fn emit(&self, _channel: &str, _data: JsonValue) {}
		fn on(&self, _channel: &str, _handler: EventBusHandler) -> Unsubscribe {
			Box::new(|| {})
		}
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn second_cached_load_dispatches_session_start() {
		let dir = std::env::temp_dir().join(format!("loader-probe-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let cwd = dir.to_string_lossy().to_string();

		let counter = Arc::new(AtomicU32::new(0));
		let counter_for_fn = Arc::clone(&counter);
		let factory: ExtensionFactory = Arc::new(move |_pi: ExtensionApiHandle| {
			let counter = Arc::clone(&counter_for_fn);
			Box::pin(async move {
				_pi.on(
					"session_start",
					Arc::new(move |_event: JsonValue, _ctx: ExtensionContext| {
						let counter = Arc::clone(&counter);
						Box::pin(async move {
							counter.fetch_add(1, Ordering::SeqCst);
							Ok(None)
						})
					}),
				);
				Ok(())
			})
		});

		let factory_for_first = Arc::clone(&factory);
		let cwd_for_task = cwd.clone();
		let first = tokio::task::spawn_blocking(move || {
			let mut loader = DefaultResourceLoader::new(crate::core::resource_loader::DefaultResourceLoaderOptions {
				cwd: cwd_for_task.clone(),
				agent_dir: cwd_for_task.clone(),
				extension_factories: vec![crate::core::extensions::types::InlineExtension::Factory(factory_for_first)],
				..Default::default()
			});
			loader.reload(ResourceLoaderReloadOptions::default());
			loader
		})
		.await
		.unwrap();
		let first_result = first.get_extensions().clone();
		let first_extensions = first_result.extensions.clone();
		let factory_for_second = Arc::clone(&factory);
		let cwd_for_task = cwd.clone();
		let second = tokio::task::spawn_blocking(move || {
			let mut loader = DefaultResourceLoader::new(crate::core::resource_loader::DefaultResourceLoaderOptions {
				cwd: cwd_for_task.clone(),
				agent_dir: cwd_for_task.clone(),
				extension_factories: vec![crate::core::extensions::types::InlineExtension::Factory(factory_for_second)],
				..Default::default()
			});
			loader.reload(ResourceLoaderReloadOptions::default());
			loader
		})
		.await
		.unwrap();
		let second_result = second.get_extensions().clone();
		let second_extensions = second_result.extensions.clone();

		let session_manager = Arc::new(parking_lot::Mutex::new(
			crate::core::session_manager::SessionManager::in_memory(None, None, None).unwrap(),
		));
		let model_runtime = crate::core::model_runtime::ModelRuntime::create(Default::default()).await.unwrap();
		let model_registry = Arc::new(crate::core::model_registry::ModelRegistry::new(model_runtime));
		let make_runner = |extensions: Vec<crate::core::extensions::types::Extension>| {
			let runner = ExtensionRunner::new(
				extensions,
				create_extension_runtime(),
				&cwd,
				Arc::clone(&session_manager),
				Arc::clone(&model_registry),
			);
			runner
		};
		let runner_one = Arc::new(make_runner(first_extensions));
		let runner_two = Arc::new(make_runner(second_extensions));

		runner_one.emit("session_start", JsonValue::Null).await;
		let after_first = counter.load(Ordering::SeqCst);
		runner_two.emit("session_start", JsonValue::Null).await;
		let after_second = counter.load(Ordering::SeqCst);

		assert!(after_first >= 1, "first load dispatch: {after_first}");
		assert!(
			after_second > after_first,
			"second load must dispatch into its own fresh handlers (first={after_first}, second={after_second})"
		);
	}
}
