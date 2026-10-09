//! Port of `packages/coding-agent/src/core/extensions/runner.ts` — the
//! extension runner: executes extensions and manages their lifecycle
//! (shortcut conflicts, flag/command/tool collection, contexts, and the
//! `emitXxx` dispatch family).
//!
//! Handlers are the erased [`HandlerFn`]s stored by the loader (crate docs
//! D31): each dispatch serializes the event, calls every snapshotted handler
//! in registration order, and parses results back into the typed event
//! results. Handler failures surface through [`ExtensionRunner::on_error`]
//! (D41).
//!
//! Deviations: D40 (upstream in-place event mutation becomes explicit
//! handler-result fields — `emit_tool_call` reads an `input` patch, headers
//! merge a returned object, boundary entries flow through result `entries`),
//! D44 (`buildSystemPrompt` is a renderer seam
//! [`ExtensionRunner::set_system_prompt_renderer`]; the default renderer now
//! calls the real [`crate::core::system_prompt`] builder), D45
//! (`KeybindingsConfig` is
//! [`ResolvedKeybindings`]; keybindings.ts ports separately), D46
//! (`getShortcuts` warns via `eprintln!`; console capture is not portable),
//! D54 (post-bind provider registration from captured `pi` objects stays
//! queued until the loader stand-in wires delegation to the runner).
//!
//! Tests: ported 1:1 from `test/extensions-runner.test.ts` and the runner
//! halves of the regressions suites; suites that construct a full
//! AgentSession (extension discovery through `DefaultResourceLoader`,
//! `tool-system-prompt-contributions`, `default-tools-setting`,
//! `tools-manager`) are deferred with the AgentSession port.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use serde_json::Value as JsonValue;

use pi_agent::{AgentMessage, AgentThinkingLevel, AgentTool, BoxFuture};
use pi_ai::types::{ImageContent, Message, Model, ProviderHeaders};
use pi_ai::utils::transcript::get_current_system_message;

use super::types::{
	BeforeAgentStartEventResult, BoundaryContextPreview, BoundaryResult, BuildSystemPromptOptions,
	CacheWarmingAction, CacheWarmingDecisionEvent, CompactOptions, ContextEvent, CustomMessageDraft, EntryRenderer,
	Extension, ExtensionActions, ExtensionCommandContext, ExtensionCommandContextActions, ExtensionContext,
	ExtensionContextActions, ExtensionError, ExtensionFlag, ExtensionMode, ExtensionShortcut, ExtensionToolContext, ExtensionUi,
	HandlerFn, InputEvent, InputEventResult, InputSource, KeyId, LoadExtensionsResult, MarkdownTransformer,
	MessageEndEvent, MessageEndEventResult, MessageRenderer, NormalizedBuildSystemPromptOptions, ProjectTrustEvent,
	ProjectTrustEventResult, RegisteredCommand, RegisteredTool, ResolvedCommand, ResourcesDiscoverReason,
	SessionBeforeForkEvent, SessionBeforeForkResult, SessionBeforeSwitchEvent, SessionBeforeSwitchResult,
	SessionBeforeTreeEvent, SessionBeforeTreeResult, SessionBoundaryDraft, ToolCallEvent, ToolCallEventResult,
	ToolRenderers, ToolResultEvent, ToolResultEventResult, Unsubscribe, UserBashEvent, UIPromptKind,
};
use crate::core::session_manager::SessionManager as SessionManagerType;
use parking_lot::Mutex as PlMutex;

/// Error listener (`ExtensionErrorListener`).
pub type ExtensionErrorListener = Arc<dyn Fn(&ExtensionError) + Send + Sync>;

/// Resolved keybindings (`KeybindingsConfig`): action name → bound keys
/// (D45; `core/keybindings.ts` ports separately).
pub type ResolvedKeybindings = BTreeMap<String, Vec<KeyId>>;

/// The system-prompt renderer seam (D44): renders normalized options into
/// the prompt text (`buildSystemPrompt` upstream).
pub type SystemPromptRenderer = Arc<dyn Fn(&NormalizedBuildSystemPromptOptions) -> String + Send + Sync>;

/// Context preview builder passed to [`ExtensionRunner::emit_boundary`]
/// (upstream's `buildContext` callback).
pub type BoundaryContextBuilder =
	Arc<dyn Fn(&[SessionBoundaryDraft]) -> Result<BoundaryContextPreview, String> + Send + Sync>;

/// Combined result of [`ExtensionRunner::emit_boundary`] (`BoundaryDispatchResult`).
#[derive(Debug, Clone)]
pub struct BoundaryDispatchResult {
	pub entries: Vec<SessionBoundaryDraft>,
	pub continue_: bool,
	pub context: BoundaryContextPreview,
	pub valid: bool,
}

/// Combined result from all `before_agent_start` handlers
/// (`BeforeAgentStartCombinedResult`).
#[derive(Debug, Clone)]
pub struct BeforeAgentStartCombinedResult {
	pub messages: Vec<CustomMessageDraft>,
	pub system_prompt_options: NormalizedBuildSystemPromptOptions,
}

/// Discovered resources from `resources_discover` handlers.
#[derive(Debug, Clone, Default)]
pub struct ResourcesDiscoverOutcome {
	pub skill_paths: Vec<(String, String)>,
	pub prompt_paths: Vec<(String, String)>,
	pub theme_paths: Vec<(String, String)>,
}

/// `session_before_compact` dispatch result. The `compaction` override
/// cannot round-trip through the erased JSON channel (D30), so handlers'
/// raw result JSON is carried alongside the parsed `cancel` flag.
#[derive(Debug, Clone, Default)]
pub struct SessionBeforeCompactDispatch {
	pub cancel: bool,
	pub raw: Option<JsonValue>,
}

// Extension shortcuts compete with canonical keybinding ids from
// keybindings.json. Only editor-global shortcuts are reserved here.
// Picker-specific bindings are not.
const RESERVED_KEYBINDINGS_FOR_EXTENSION_CONFLICTS: [&str; 18] = [
	"app.interrupt",
	"app.clear",
	"app.exit",
	"app.suspend",
	"app.thinking.cycle",
	"app.model.cycleForward",
	"app.model.cycleBackward",
	"app.model.select",
	"app.tools.expand",
	"app.thinking.toggle",
	"app.editor.external",
	"app.message.copy",
	"app.message.followUp",
	"tui.input.submit",
	"tui.select.confirm",
	"tui.select.cancel",
	"tui.input.copy",
	"tui.editor.deleteToLineEnd",
];

/// Built-in keybinding entry: the action a key triggers and whether the
/// action's key may be overridden by extension shortcuts.
#[derive(Debug, Clone)]
struct BuiltInKeyBinding {
	keybinding: String,
	restrict_override: bool,
}

fn build_builtin_keybindings(resolved: &ResolvedKeybindings) -> HashMap<KeyId, BuiltInKeyBinding> {
	let mut builtin: HashMap<KeyId, BuiltInKeyBinding> = HashMap::new();
	for (keybinding, keys) in resolved {
		let restrict_override = RESERVED_KEYBINDINGS_FOR_EXTENSION_CONFLICTS.contains(&keybinding.as_str());
		for key in keys {
			let normalized_key = key.to_lowercase();
			// If multiple actions bind the same key, the reserved action wins
			// so extensions remain blocked by reserved shortcuts regardless of
			// iteration order.
			if let Some(existing) = builtin.get(&normalized_key) {
				if existing.restrict_override && !restrict_override {
					continue;
				}
			}
			builtin.insert(
				normalized_key,
				BuiltInKeyBinding { keybinding: keybinding.clone(), restrict_override },
			);
		}
	}
	builtin
}

fn is_system_message(message: &AgentMessage) -> bool {
	match message {
		AgentMessage::Llm(Message::System(_)) => true,
		AgentMessage::Custom(value) => value.get("role").and_then(JsonValue::as_str) == Some("system"),
		_ => false,
	}
}

fn message_role(message: &AgentMessage) -> String {
	match message {
		AgentMessage::Llm(message) => message.role().to_string(),
		AgentMessage::Custom(value) => value.get("role").and_then(JsonValue::as_str).unwrap_or_default().to_string(),
	}
}

/// The default no-op UI context (`noOpUIContext`).
#[derive(Debug, Default)]
pub struct NoopExtensionUi;

impl ExtensionUi for NoopExtensionUi {
	fn select(
		&self,
		_title: String,
		_options: Vec<String>,
		_opts: Option<super::types::ExtensionUiDialogOptions>,
	) -> BoxFuture<'static, Option<String>> {
		Box::pin(async { None })
	}

	fn confirm(
		&self,
		_title: String,
		_message: String,
		_opts: Option<super::types::ExtensionUiDialogOptions>,
	) -> BoxFuture<'static, bool> {
		Box::pin(async { false })
	}

	fn input(
		&self,
		_title: String,
		_placeholder: Option<String>,
		_opts: Option<super::types::ExtensionUiDialogOptions>,
	) -> BoxFuture<'static, Option<String>> {
		Box::pin(async { None })
	}

	fn notify(&self, _message: String, _kind: Option<super::types::NotifyKind>) {}

	fn on_terminal_input(&self, _handler: super::types::TerminalInputHandler) -> Unsubscribe {
		Box::new(|| {})
	}

	fn set_status(&self, _key: String, _text: Option<String>) {}
	fn set_working_message(&self, _message: Option<String>) {}
	fn set_working_visible(&self, _visible: bool) {}
	fn set_working_indicator(&self, _options: Option<super::types::WorkingIndicatorOptions>) {}
	fn set_hidden_thinking_label(&self, _label: Option<String>) {}
	fn set_widget(
		&self,
		_key: String,
		_content: Option<super::types::WidgetContent>,
		_options: Option<super::types::ExtensionWidgetOptions>,
	) {
	}
	fn set_footer(&self, _factory: Option<super::types::FooterFactory>) {}
	fn set_header(&self, _factory: Option<super::types::HeaderFactory>) {}
	fn set_title(&self, _title: String) {}

	fn custom(
		&self,
		_factory: super::types::CustomComponentFactory,
		_options: Option<super::types::CustomComponentOptions>,
	) -> BoxFuture<'static, JsonValue> {
		Box::pin(async { JsonValue::Null })
	}

	fn paste_to_editor(&self, _text: String) {}
	fn set_editor_text(&self, _text: String) {}
	fn get_editor_text(&self) -> String {
		String::new()
	}

	fn editor(&self, _title: String, _prefill: Option<String>) -> BoxFuture<'static, Option<String>> {
		Box::pin(async { None })
	}

	fn add_autocomplete_provider(&self, _factory: super::types::AutocompleteProviderFactory) {}
	fn set_editor_component(&self, _factory: Option<super::types::EditorFactory>) {}
	fn get_editor_component(&self) -> Option<super::types::EditorFactory> {
		None
	}

	fn theme(&self) -> super::types::Theme {
		super::types::Theme
	}
	fn get_all_themes(&self) -> Vec<super::types::ThemeInfo> {
		Vec::new()
	}
	fn get_theme(&self, _name: &str) -> Option<super::types::Theme> {
		None
	}
	fn set_theme(&self, _theme: super::types::SetThemeArg) -> super::types::ThemeSwitchResult {
		super::types::ThemeSwitchResult {
			success: false,
			error: Some("UI not available".to_string()),
		}
	}

	fn get_tools_expanded(&self) -> bool {
		false
	}
	fn set_tools_expanded(&self, _expanded: bool) {}
}

/// Snapshot of one extension's handlers for an event (`snapshotEventHandlers`).
struct HandlerSnapshot {
	extension_path: String,
	handlers: Vec<HandlerFn>,
}

/// The extension runner (`ExtensionRunner`).
pub struct ExtensionRunner {
	extensions: Vec<Extension>,
	runtime: super::types::ExtensionRuntime,
	ui_context: RwLock<Arc<dyn ExtensionUi>>,
	/// `true` once a real UI context was installed (`hasUI`).
	ui_provided: RwLock<bool>,
	mode: RwLock<ExtensionMode>,
	cwd: String,
	session_manager: Arc<PlMutex<SessionManagerType>>,
	model_registry: Arc<super::types::ModelRegistry>,
	error_listeners: Arc<Mutex<Vec<ExtensionErrorListener>>>,
	stale_message: Arc<Mutex<Option<String>>>,
	/// Registered MCP servers already reported as unhandled.
	reported_mcp_servers: Mutex<HashSet<String>>,
	shortcut_diagnostics: Mutex<Vec<crate::core::diagnostics::ResourceDiagnostic>>,
	command_diagnostics: Mutex<Vec<crate::core::diagnostics::ResourceDiagnostic>>,
	ui_prompt_depth: Mutex<u32>,
	active_ui_prompt: Mutex<Option<(UIPromptKind, Option<String>)>>,
	system_prompt_renderer: Mutex<SystemPromptRenderer>,
}

impl std::fmt::Debug for ExtensionRunner {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ExtensionRunner")
			.field("extensions", &self.extensions.iter().map(|e| e.path.clone()).collect::<Vec<_>>())
			.field("mode", &*self.mode.read())
			.field("cwd", &self.cwd)
			.finish_non_exhaustive()
	}
}

impl ExtensionRunner {
	pub fn new(
		extensions: Vec<Extension>,
		runtime: super::types::ExtensionRuntime,
		cwd: &str,
		session_manager: Arc<PlMutex<SessionManagerType>>,
		model_registry: Arc<super::types::ModelRegistry>,
	) -> Self {
		ExtensionRunner {
			extensions,
			runtime,
			ui_context: RwLock::new(Arc::new(NoopExtensionUi)),
			ui_provided: RwLock::new(false),
			mode: RwLock::new(ExtensionMode::Print),
			cwd: cwd.to_string(),
			session_manager,
			model_registry,
			error_listeners: Arc::new(Mutex::new(Vec::new())),
			stale_message: Arc::new(Mutex::new(None)),
			reported_mcp_servers: Mutex::new(HashSet::new()),
			shortcut_diagnostics: Mutex::new(Vec::new()),
			command_diagnostics: Mutex::new(Vec::new()),
			ui_prompt_depth: Mutex::new(0),
			active_ui_prompt: Mutex::new(None),
			system_prompt_renderer: Mutex::new(Arc::new(crate::core::system_prompt::build_system_prompt_normalized)),
		}
	}

	// -- Binding -------------------------------------------------------------

	/// Copy actions into the shared runtime (`bindCore`). All extension APIs
	/// reference this runtime.
	pub fn bind_core(
		&self,
		actions: ExtensionActions,
		context_actions: ExtensionContextActions,
		provider_actions: Option<super::types::ProviderActions>,
	) {
		// Flush provider registrations queued during extension loading.
		let provider_actions = provider_actions.unwrap_or_default();
		for registration in self.runtime.state.pending_providers() {
			let outcome = provider_actions
				.register_provider
				.as_ref()
				.map(|register| {
					// Handler failures surface as extension errors, exactly
					// like upstream's `try { registerProvider(...) } catch`.
					std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
						register(&registration.name, &registration.config);
					}))
				})
				.unwrap_or(Ok(()));
			if let Err(payload) = outcome {
				self.emit_error(&ExtensionError {
					extension_path: registration.extension_path.clone(),
					event: "register_provider".to_string(),
					error: panic_payload_message(payload),
					stack: None,
				});
			}
		}
		for registration in self.runtime.pending_virtual_models() {
			let outcome = provider_actions
				.register_virtual_model
				.as_ref()
				.map(|register| {
					std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
						if let Some(definition) = &registration.definition {
							register(definition);
						}
					}))
				})
				.unwrap_or(Ok(()));
			if let Err(payload) = outcome {
				self.emit_error(&ExtensionError {
					extension_path: registration.extension_path.clone(),
					event: "register_virtual_model".to_string(),
					error: panic_payload_message(payload),
					stack: None,
				});
			}
		}
		self.runtime.state.clear_pending_providers();
		self.runtime.state.clear_pending_virtual_models();

		// Install the bound action set.
		let mut bound = self.runtime.core.bound.write();
		*bound = Some(super::types::BoundCore {
			actions: actions.clone(),
			context_actions,
			command_actions: ExtensionCommandContextActions::default(),
			provider_actions,
			create_context: None,
		});
		drop(bound);
	}

	/// Bind the command-context actions (`bindCommandContext`); `None`
	/// resets to the throwing-free defaults.
	pub fn bind_command_context(&self, actions: Option<ExtensionCommandContextActions>) {
		let mut bound = self.runtime.core.bound.write();
		match bound.as_mut() {
			Some(bound) => {
				bound.command_actions = actions.unwrap_or_default();
			}
			None => {
				// No core bind yet — store the command actions alone; the
				// full bind replaces them later.
				*bound = Some(super::types::BoundCore {
					actions: ExtensionActions::default(),
					context_actions: ExtensionContextActions::default(),
					command_actions: actions.unwrap_or_default(),
					provider_actions: Default::default(),
					create_context: None,
				});
			}
		}
	}

	/// Install the UI context and run mode (`setUIContext`); `None` resets
	/// to the no-op UI.
	pub fn set_ui_context(&self, ui_context: Option<Arc<dyn ExtensionUi>>, mode: ExtensionMode) {
		match ui_context {
			Some(ui) => {
				*self.ui_provided.write() = true;
				*self.ui_context.write() = ui;
			}
			None => {
				*self.ui_provided.write() = false;
				*self.ui_context.write() = Arc::new(NoopExtensionUi);
			}
		}
		*self.mode.write() = mode;
	}

	/// Set the system-prompt renderer seam (D44).
	pub fn set_system_prompt_renderer(&self, renderer: SystemPromptRenderer) {
		*self.system_prompt_renderer.lock() = renderer;
	}

	fn render_system_prompt(&self, options: &NormalizedBuildSystemPromptOptions) -> String {
		(self.system_prompt_renderer.lock())(options)
	}

	// -- UI / introspection ---------------------------------------------------

	pub fn get_ui_context(&self) -> Arc<dyn ExtensionUi> {
		Arc::clone(&self.ui_context.read())
	}

	pub fn has_ui(&self) -> bool {
		*self.ui_provided.read()
	}

	pub fn get_extension_paths(&self) -> Vec<String> {
		self.extensions.iter().map(|extension| extension.path.clone()).collect()
	}

	/// Get all registered tools from all extensions; first registration per
	/// name wins (`getAllRegisteredTools`).
	pub fn get_all_registered_tools(&self) -> Vec<&RegisteredTool> {
		let mut tools_by_name: HashMap<&str, &RegisteredTool> = HashMap::new();
		for extension in &self.extensions {
			for (name, tool) in &extension.tools {
				tools_by_name.entry(name.as_str()).or_insert(tool);
			}
		}
		tools_by_name.into_values().collect()
	}

	/// Get a tool definition by name (`getToolDefinition`).
	pub fn get_tool_definition(&self, tool_name: &str) -> Option<&super::types::ToolDefinition> {
		self.extensions
			.iter()
			.find_map(|extension| extension.tools.get(tool_name).map(|tool| &tool.definition))
	}

	pub fn get_flags(&self) -> HashMap<String, ExtensionFlag> {
		let mut all_flags = HashMap::new();
		for extension in &self.extensions {
			for (name, flag) in &extension.flags {
				all_flags.entry(name.clone()).or_insert_with(|| flag.clone());
			}
		}
		all_flags
	}

	pub fn set_flag_value(&self, name: &str, value: super::types::FlagValue) {
		self.runtime.state.set_flag_value(name, value);
	}

	pub fn get_flag_values(&self) -> HashMap<String, super::types::FlagValue> {
		self.runtime.state.flag_values.read().clone()
	}

	/// Resolve extension shortcuts against the built-in keybindings
	/// (`getShortcuts`). Conflicts are reported via
	/// [`Self::get_shortcut_diagnostics`] (D46: the `console.warn` fallback
	/// becomes `eprintln!` when no UI is present).
	pub fn get_shortcuts(&self, resolved: &ResolvedKeybindings) -> HashMap<KeyId, ExtensionShortcut> {
		*self.shortcut_diagnostics.lock() = Vec::new();
		let builtin_keybindings = build_builtin_keybindings(resolved);
		let mut extension_shortcuts: HashMap<KeyId, ExtensionShortcut> = HashMap::new();

		let add_diagnostic = |diagnostics: &Mutex<Vec<crate::core::diagnostics::ResourceDiagnostic>>, message: String, extension_path: String| {
			diagnostics.lock().push(crate::core::diagnostics::ResourceDiagnostic {
				r#type: crate::core::diagnostics::DiagnosticType::Warning,
				message: message.clone(),
				path: Some(extension_path.clone()),
				collision: None,
			});
			if !self.has_ui() {
				eprintln!("{message}");
			}
		};

		for extension in &self.extensions {
			for (key, shortcut) in &extension.shortcuts {
				let normalized_key = key.to_lowercase();

				if let Some(built_in) = builtin_keybindings.get(&normalized_key) {
					if built_in.restrict_override {
						add_diagnostic(
							&self.shortcut_diagnostics,
							format!(
								"Extension shortcut '{}' from {} conflicts with built-in shortcut. Skipping.",
								key, shortcut.extension_path
							),
							shortcut.extension_path.clone(),
						);
						continue;
					}
					add_diagnostic(
						&self.shortcut_diagnostics,
						format!(
							"Extension shortcut conflict: '{}' is built-in shortcut for {} and {}. Using {}.",
							key, built_in.keybinding, shortcut.extension_path, shortcut.extension_path
						),
						shortcut.extension_path.clone(),
					);
				}

				if let Some(existing) = extension_shortcuts.get(&normalized_key) {
					add_diagnostic(
						&self.shortcut_diagnostics,
						format!(
							"Extension shortcut conflict: '{}' registered by both {} and {}. Using {}.",
							key, existing.extension_path, shortcut.extension_path, shortcut.extension_path
						),
						shortcut.extension_path.clone(),
					);
				}
				extension_shortcuts.insert(normalized_key, shortcut.clone());
			}
		}
		extension_shortcuts
	}

	pub fn get_shortcut_diagnostics(&self) -> Vec<crate::core::diagnostics::ResourceDiagnostic> {
		self.shortcut_diagnostics.lock().clone()
	}

	/// Marks the runner and shared runtime stale (`invalidate`).
	pub fn invalidate(&self, message: Option<String>) {
		let mut stale = self.stale_message.lock();
		if stale.is_some() {
			return;
		}
		*stale = Some(message.unwrap_or_else(|| {
			"This extension ctx is stale after session replacement or reload. Do not use a captured pi or command ctx after ctx.newSession(), ctx.fork(), ctx.switchSession(), or ctx.reload(). For newSession, fork, and switchSession, move post-replacement work into withSession and use the ctx passed to withSession. For reload, do not use the old ctx after await ctx.reload().".to_string()
		}));
		drop(stale);
		(self.runtime.state.invalidate)(self.stale_message.lock().clone());
	}

	fn assert_active(&self) -> Result<(), String> {
		match self.stale_message.lock().as_ref() {
			Some(message) => Err(message.clone()),
			None => Ok(()),
		}
	}

	/// Subscribe to extension errors (`onError`); returns the unsubscribe
	/// function.
	pub fn on_error(&self, listener: ExtensionErrorListener) -> Unsubscribe {
		self.error_listeners.lock().push(Arc::clone(&listener));
		let listeners = Arc::clone(&self.error_listeners);
		Box::new(move || {
			listeners.lock().retain(|candidate| !Arc::ptr_eq(candidate, &listener));
		})
	}

	pub fn emit_error(&self, error: &ExtensionError) {
		for listener in self.error_listeners.lock().iter() {
			listener(error);
		}
	}

	/// Report registered MCP servers when no extension handles
	/// `mcp_servers_change` (`reportUnhandledMcpServers`).
	pub fn report_unhandled_mcp_servers(&self) {
		if self.has_handlers("mcp_servers_change") {
			return;
		}
		for server in self.runtime.state.mcp_servers.list() {
			if !self.reported_mcp_servers.lock().insert(server.name.clone()) {
				continue;
			}
			self.emit_error(&ExtensionError {
				extension_path: server.extension_path.clone(),
				event: "register_mcp_server".to_string(),
				error: format!(
					"MCP server \"{}\" is registered, but no loaded extension connects MCP servers; another extension may have replaced the built-in MCP support",
					server.name
				),
				stack: None,
			});
		}
	}

	/// Whether any extension registered a handler for the event
	/// (`hasHandlers`).
	pub fn has_handlers(&self, event_type: &str) -> bool {
		self.extensions
			.iter()
			.any(|extension| {
				extension
					.handlers
					.lock()
					.get(event_type)
					.is_some_and(|handlers| !handlers.is_empty())
			})
	}

	pub fn get_message_renderer(&self, custom_type: &str) -> Option<MessageRenderer> {
		self.extensions
			.iter()
			.find_map(|extension| extension.message_renderers.get(custom_type).cloned())
	}

	pub fn get_markdown_transformers(&self) -> Vec<MarkdownTransformer> {
		self.extensions
			.iter()
			.filter_map(|extension| extension.markdown_transformer.clone())
			.collect()
	}

	/// Renderers of calls to `toolName`: extension resolvers in load order,
	/// then `base` (`resolveToolRenderers`).
	pub fn resolve_tool_renderers(
		&self,
		tool_name: &str,
		base: &mut dyn FnMut() -> Option<ToolRenderers>,
	) -> Option<ToolRenderers> {
		fn resolve_at(
			resolvers: &[super::types::ToolRendererResolver],
			index: usize,
			tool_name: &str,
			base: &mut dyn FnMut() -> Option<ToolRenderers>,
		) -> Option<ToolRenderers> {
			if index < resolvers.len() {
				let mut next = || resolve_at(resolvers, index + 1, tool_name, base);
				return resolvers[index](tool_name, &mut next);
			}
			base()
		}
		let resolvers: Vec<super::types::ToolRendererResolver> = self
			.extensions
			.iter()
			.flat_map(|extension| extension.tool_renderers.clone().unwrap_or_default())
			.collect();
		resolve_at(&resolvers, 0, tool_name, base)
	}

	pub fn get_entry_renderer(&self, custom_type: &str) -> Option<EntryRenderer> {
		self.extensions.iter().find_map(|extension| {
			extension
				.entry_renderers
				.as_ref()
				.and_then(|renderers| renderers.get(custom_type).cloned())
		})
	}

	fn resolve_registered_commands(&self) -> Vec<ResolvedCommand> {
		let mut commands: Vec<&RegisteredCommand> = Vec::new();
		let mut counts: HashMap<&str, usize> = HashMap::new();

		for extension in &self.extensions {
			for command in extension.commands.values() {
				*counts.entry(command.name.as_str()).or_default() += 1;
				commands.push(command);
			}
		}

		let mut seen: HashMap<&str, usize> = HashMap::new();
		let mut taken_invocation_names: HashSet<String> = HashSet::new();

		commands
			.into_iter()
			.map(|command| {
				let occurrence = seen.entry(command.name.as_str()).or_default();
				*occurrence += 1;
				let occurrence = *occurrence;

				let mut invocation_name = if counts.get(command.name.as_str()).copied().unwrap_or(0) > 1 {
					format!("{}:{}", command.name, occurrence)
				} else {
					command.name.clone()
				};

				if taken_invocation_names.contains(&invocation_name) {
					let mut suffix = occurrence;
					loop {
						suffix += 1;
						invocation_name = format!("{}:{}", command.name, suffix);
						if !taken_invocation_names.contains(&invocation_name) {
							break;
						}
					}
				}

				taken_invocation_names.insert(invocation_name.clone());
				ResolvedCommand { command: command.clone(), invocation_name }
			})
			.collect()
	}

	pub fn get_model_registry(&self) -> Arc<super::types::ModelRegistry> {
		Arc::clone(&self.model_registry)
	}

	/// Resolved commands with unique invocation names
	/// (`getRegisteredCommands`).
	pub fn get_registered_commands(&self) -> Vec<ResolvedCommand> {
		*self.command_diagnostics.lock() = Vec::new();
		self.resolve_registered_commands()
	}

	pub fn get_command_diagnostics(&self) -> Vec<crate::core::diagnostics::ResourceDiagnostic> {
		self.command_diagnostics.lock().clone()
	}

	pub fn get_command(&self, name: &str) -> Option<ResolvedCommand> {
		self.resolve_registered_commands()
			.into_iter()
			.find(|command| command.invocation_name == name)
	}

	/// Request a graceful shutdown (`shutdown`).
	pub fn shutdown(&self) {
		if let Some(bound) = self.runtime.core.bound.read().as_ref() {
			if let Some(shutdown) = &bound.context_actions.shutdown {
				shutdown();
			}
		}
	}

	pub fn get_active_tools(&self) -> Result<Vec<String>, String> {
		self.assert_active()?;
		Ok(self
			.runtime
			.core
			.bound
			.read()
			.as_ref()
			.and_then(|bound| bound.actions.get_active_tools.clone())
			.map(|handler| handler())
			.unwrap_or_default())
	}

	// -- Contexts -------------------------------------------------------------

	fn bound(&self) -> super::types::BoundCore {
		self.runtime
			.core
			.bound
			.read()
			.as_ref()
			.cloned()
			.unwrap_or_default()
	}

	/// Builds an [`ExtensionContext`] from the bound action set; before
	/// `bind_core` the defaults apply (upstream: default getters never
	/// return undefined for `scopedModels` etc.).
	fn build_context(
		&self,
		get_system_prompt: Arc<dyn Fn() -> String + Send + Sync>,
	) -> Result<ExtensionContext, String> {
		self.assert_active()?;
		let bound = self.bound();
		let actions = &bound.actions;
		let context_actions = &bound.context_actions;

		let runner_stale = Arc::clone(&self.stale_message);
		let assert_runner_active = move || -> Result<(), String> {
			match runner_stale.lock().as_ref() {
				Some(message) => Err(message.clone()),
				None => Ok(()),
			}
		};

		Ok(ExtensionContext {
			ui: Arc::clone(&self.ui_context.read()),
			mode: *self.mode.read(),
			has_ui: self.has_ui(),
			cwd: self.cwd.clone(),
			session_manager: Arc::clone(&self.session_manager),
			model_registry: Arc::clone(&self.model_registry),
			model: context_actions.get_model.as_ref().and_then(|handler| handler()),
			scoped_models: context_actions
				.get_scoped_models
				.as_ref()
				.map(|handler| handler())
				.unwrap_or_default(),
			thinking_level: actions.get_thinking_level.as_ref().map(|handler| handler()),
			signal: context_actions.get_signal.as_ref().and_then(|handler| handler()),
			is_idle: {
				let action = context_actions.is_idle.clone();
				let assert_active = assert_runner_active.clone();
				Arc::new(move || {
					assert_active().unwrap_or_else(|message| panic!("{message}"));
					action.as_ref().map(|handler| handler()).unwrap_or(true)
				})
			},
			is_project_trusted: {
				let action = context_actions.is_project_trusted.clone();
				let assert_active = assert_runner_active.clone();
				Arc::new(move || {
					assert_active().unwrap_or_else(|message| panic!("{message}"));
					action.as_ref().map(|handler| handler()).unwrap_or(true)
				})
			},
			abort: {
				let action = context_actions.abort.clone();
				let assert_active = assert_runner_active.clone();
				Arc::new(move || {
					assert_active().unwrap_or_else(|message| panic!("{message}"));
					if let Some(handler) = &action {
						handler();
					}
				})
			},
			has_pending_messages: {
				let action = context_actions.has_pending_messages.clone();
				let assert_active = assert_runner_active.clone();
				Arc::new(move || {
					assert_active().unwrap_or_else(|message| panic!("{message}"));
					action.as_ref().map(|handler| handler()).unwrap_or(false)
				})
			},
			shutdown: {
				let action = context_actions.shutdown.clone();
				let assert_active = assert_runner_active.clone();
				Arc::new(move || {
					assert_active().unwrap_or_else(|message| panic!("{message}"));
					if let Some(handler) = &action {
						handler();
					}
				})
			},
			get_context_usage: {
				let action = context_actions.get_context_usage.clone();
				let assert_active = assert_runner_active.clone();
				Arc::new(move || {
					assert_active().unwrap_or_else(|message| panic!("{message}"));
					action.as_ref().and_then(|handler| handler())
				})
			},
			compact: {
				let action = context_actions.compact.clone();
				let assert_active = assert_runner_active.clone();
				Arc::new(move |options: Option<CompactOptions>| {
					assert_active().unwrap_or_else(|message| panic!("{message}"));
					if let Some(handler) = &action {
						handler(options);
					}
				})
			},
			get_system_prompt: {
				let assert_active = assert_runner_active;
				Arc::new(move || {
					assert_active().unwrap_or_else(|message| panic!("{message}"));
					get_system_prompt()
				})
			},
		})
	}

	/// Create an [`ExtensionContext`] for event handlers and tool execution
	/// (`createContext`). Values resolve at call time, so `bind_core`
	/// changes are reflected.
	pub fn create_context(&self) -> Result<ExtensionContext, String> {
		let bound = self.bound();
		let get_system_prompt: Arc<dyn Fn() -> String + Send + Sync> = {
			let get_system_prompt = bound.context_actions.get_system_prompt.clone();
			Arc::new(move || {
				get_system_prompt
					.as_ref()
					.map(|handler| handler())
					.unwrap_or_default()
			})
		};
		self.build_context(get_system_prompt)
	}

	/// Create the context for executing `tool_call_id` (`createToolContext`):
	/// the extension context plus `tools` and `executeTool`. `signal` is the
	/// default signal of nested calls.
	pub fn create_tool_context(
		&self,
		tool_call_id: &str,
		signal: Option<crate::utils::abort::AbortSignal>,
	) -> Result<ExtensionToolContext, String> {
		let ctx = self.create_context()?;
		let tools = self
			.bound()
			.context_actions
			.get_callable_tools
			.clone()
			.map(|handler| handler())
			.unwrap_or_else(Vec::new);
		let execute_tool = {
			let bound = self.bound();
			let tool_call_id = tool_call_id.to_string();
			let default_signal = signal;
			Arc::new(
				move |name: String,
				      args: JsonValue,
				      options: super::types::ExecuteToolOptions|
				      -> BoxFuture<'static, pi_agent::AgentToolCallOutcome> {
					match bound.context_actions.execute_tool.clone() {
						Some(action) => {
							let mut options = options;
							if options.signal.is_none() {
								options.signal = default_signal.clone();
							}
							action(tool_call_id.clone(), name, args, options)
						}
						None => {
							let tool_call_id = tool_call_id.clone();
							Box::pin(async move { nested_tool_call_unavailable(&tool_call_id, &name) })
						}
					}
				},
			)
		};
		Ok(ExtensionToolContext { ctx, tools, execute_tool })
	}

	/// Create the command context (`createCommandContext`): the extension
	/// context plus the session-control methods.
	pub fn create_command_context(&self) -> Result<ExtensionCommandContext, String> {
		let ctx = self.create_context()?;
		let bound = self.bound();
		let command_actions = bound.command_actions.clone();
		let assert_active = {
			let stale = Arc::clone(&self.stale_message);
			move || -> Result<(), String> {
				match stale.lock().as_ref() {
					Some(message) => Err(message.clone()),
					None => Ok(()),
				}
			}
		};

		let get_system_prompt_options: Arc<dyn Fn() -> BuildSystemPromptOptions + Send + Sync> = {
			let actions = bound.context_actions.get_system_prompt_options.clone();
			let cwd = self.cwd.clone();
			let assert_active = assert_active.clone();
			Arc::new(move || {
				assert_active().unwrap_or_else(|message| panic!("{message}"));
				actions
					.as_ref()
					.map(|handler| handler())
					.unwrap_or_else(|| BuildSystemPromptOptions { cwd: cwd.clone(), ..Default::default() })
			})
		};
		let wait_for_idle: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync> = {
			let action = command_actions.wait_for_idle.clone();
			Arc::new(move || match &action {
				Some(action) => action(),
				None => Box::pin(async {}),
			})
		};
		let fallback_switch = || -> BoxFuture<'static, super::types::SessionSwitchOutcome> {
			Box::pin(async { super::types::SessionSwitchOutcome { cancelled: false } })
		};
		let new_session: Arc<dyn Fn(Option<super::types::NewSessionCommandOptions>) -> BoxFuture<'static, super::types::SessionSwitchOutcome> + Send + Sync> = {
			let action = command_actions.new_session.clone();
			Arc::new(move |options| match &action {
				Some(action) => action(options),
				None => fallback_switch(),
			})
		};
		let fork: Arc<dyn Fn(String, Option<super::types::ForkCommandOptions>) -> BoxFuture<'static, super::types::SessionSwitchOutcome> + Send + Sync> = {
			let action = command_actions.fork.clone();
			Arc::new(move |entry_id, options| match &action {
				Some(action) => action(entry_id, options),
				None => fallback_switch(),
			})
		};
		let navigate_tree: Arc<dyn Fn(String, Option<super::types::NavigateTreeOptions>) -> BoxFuture<'static, super::types::SessionSwitchOutcome> + Send + Sync> = {
			let action = command_actions.navigate_tree.clone();
			Arc::new(move |target_id, options| match &action {
				Some(action) => action(target_id, options),
				None => fallback_switch(),
			})
		};
		let switch_session: Arc<dyn Fn(String, Option<super::types::SwitchSessionOptions>) -> BoxFuture<'static, super::types::SessionSwitchOutcome> + Send + Sync> = {
			let action = command_actions.switch_session.clone();
			Arc::new(move |session_path, options| match &action {
				Some(action) => action(session_path, options),
				None => fallback_switch(),
			})
		};
		let reload: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync> = {
			let action = command_actions.reload.clone();
			Arc::new(move || match &action {
				Some(action) => action(),
				None => Box::pin(async {}),
			})
		};

		Ok(ExtensionCommandContext {
			ctx,
			get_system_prompt_options,
			wait_for_idle,
			new_session,
			fork,
			navigate_tree,
			switch_session,
			reload,
		})
	}

	// -- Dispatch -------------------------------------------------------------

	fn snapshot_event_handlers(&self, event_type: &str) -> Vec<HandlerSnapshot> {
		self.extensions
			.iter()
			.map(|extension| HandlerSnapshot {
				extension_path: extension.path.clone(),
				handlers: extension.handlers.lock().get(event_type).cloned().unwrap_or_default(),
			})
			.collect()
	}

	fn is_session_before_event(event_type: &str) -> bool {
		matches!(
			event_type,
			"session_before_switch" | "session_before_fork" | "session_before_compact" | "session_before_tree"
		)
	}

	/// Generic dispatch (`emit`): calls every handler of `event_type` with
	/// the serialized event. `session_before_*` results short-circuit on
	/// `cancel: true`; the last non-cancelled result wins.
	pub async fn emit(&self, event_type: &str, event: JsonValue) -> Option<JsonValue> {
		let ctx = match self.create_context() {
			Ok(ctx) => ctx,
			Err(_) => return None,
		};
		let mut result: Option<JsonValue> = None;

		for snapshot in self.snapshot_event_handlers(event_type) {
			for handler in snapshot.handlers.clone() {
				let handler_result = handler(event.clone(), ctx.clone()).await;
				match handler_result {
					Ok(Some(handler_result)) => {
						if Self::is_session_before_event(event_type) {
							let cancel = handler_result
								.get("cancel")
								.and_then(JsonValue::as_bool)
								.unwrap_or(false);
							result = Some(handler_result);
							if cancel {
								return result;
							}
						} else {
							result = Some(handler_result);
						}
					}
					Ok(None) => {}
					Err(message) => {
						self.emit_error(&ExtensionError {
							extension_path: snapshot.extension_path.clone(),
							event: event_type.to_string(),
							error: message,
							stack: None,
						});
					}
				}
			}
		}

		result
	}

	/// `emit` for `session_before_switch`.
	pub async fn emit_session_before_switch(
		&self,
		event: &SessionBeforeSwitchEvent,
	) -> Option<SessionBeforeSwitchResult> {
		let event = json_tagged("session_before_switch", event)?;
		self.emit("session_before_switch", event)
			.await
			.and_then(|result| serde_json::from_value(result).ok())
	}

	/// `emit` for `session_before_fork`.
	pub async fn emit_session_before_fork(&self, event: &SessionBeforeForkEvent) -> Option<SessionBeforeForkResult> {
		let event = json_tagged("session_before_fork", event)?;
		self.emit("session_before_fork", event)
			.await
			.and_then(|result| serde_json::from_value(result).ok())
	}

	/// `emit` for `session_before_compact`. The typed `compaction` override
	/// cannot round-trip the erased channel (D30); the raw result is
	/// returned for the compaction slice to interpret.
	pub async fn emit_session_before_compact(
		&self,
		event: &super::types::SessionBeforeCompactEvent,
	) -> SessionBeforeCompactDispatch {
		// The event carries non-serde handles (preparation, signal — D30);
		// the serde-able members are hand-serialized.
		let event = serde_json::json!({
			"type": "session_before_compact",
			"reason": event.reason,
			"customInstructions": event.custom_instructions,
			"willRetry": event.will_retry,
		});
		let raw = self.emit("session_before_compact", event).await;
		SessionBeforeCompactDispatch {
			cancel: raw
				.as_ref()
				.and_then(|result| result.get("cancel"))
				.and_then(JsonValue::as_bool)
				.unwrap_or(false),
			raw,
		}
	}

	/// `emit` for `session_before_tree`.
	pub async fn emit_session_before_tree(&self, event: &SessionBeforeTreeEvent) -> Option<SessionBeforeTreeResult> {
		// `preparation`/`signal` carry no serde (D30); emit the tag only.
		let event = serde_json::json!({ "type": "session_before_tree" });
		self.emit("session_before_tree", event)
			.await
			.and_then(|result| serde_json::from_value(result).ok())
	}

	/// `emitCacheWarmingDecision`: the event's own action unless a handler
	/// overrides it; the last override wins.
	pub async fn emit_cache_warming_decision(&self, event: &CacheWarmingDecisionEvent) -> CacheWarmingAction {
		let ctx = match self.create_context() {
			Ok(ctx) => ctx,
			Err(_) => return event.action,
		};
		let mut action = event.action;

		for snapshot in self.snapshot_event_handlers("cache_warming_decision") {
			for handler in snapshot.handlers.clone() {
				let event_json = match serialize_tagged("cache_warming_decision", event) {
					Some(event_json) => event_json,
					None => return action,
				};
				match handler(event_json, ctx.clone()).await {
					Ok(result) => {
						if let Some(result) = result {
							if let Ok(parsed) = serde_json::from_value::<super::types::CacheWarmingDecisionEventResult>(result) {
								if let Some(override_action) = parsed.action {
									action = override_action;
								}
							}
						}
					}
					Err(message) => self.emit_error(&ExtensionError {
						extension_path: snapshot.extension_path.clone(),
						event: "cache_warming_decision".to_string(),
						error: message,
						stack: None,
					}),
				}
			}
		}

		action
	}

	/// `emitMessageEnd`: chained message replacement; the role must be
	/// preserved.
	pub async fn emit_message_end(&self, event: &MessageEndEvent) -> Option<AgentMessage> {
		let ctx = self.create_context().ok()?;
		let mut current_message = event.message.clone();
		let mut modified = false;

		for snapshot in self.snapshot_event_handlers("message_end") {
			for handler in snapshot.handlers.clone() {
				let event_json = match serde_json::to_value(MessageEndEvent {
					message: current_message.clone(),
				}) {
					Ok(mut value) => {
						value["type"] = JsonValue::from("message_end");
						value
					}
					Err(_) => continue,
				};
				match handler(event_json, ctx.clone()).await {
					Ok(result) => {
						let Some(result) = result else { continue };
						let Ok(parsed) = serde_json::from_value::<MessageEndEventResult>(result) else {
							continue;
						};
						let Some(message) = parsed.message else { continue };
						if message_role(&message) != message_role(&current_message) {
							self.emit_error(&ExtensionError {
								extension_path: snapshot.extension_path.clone(),
								event: "message_end".to_string(),
								error: "message_end handlers must return a message with the same role".to_string(),
								stack: None,
							});
							continue;
						}
						current_message = message;
						modified = true;
					}
					Err(message) => self.emit_error(&ExtensionError {
						extension_path: snapshot.extension_path.clone(),
						event: "message_end".to_string(),
						error: message,
						stack: None,
					}),
				}
			}
		}

		modified.then_some(current_message)
	}

	/// `emitToolResult`: chained result modifications; `None` when no
	/// handler changed anything.
	pub async fn emit_tool_result(&self, event: &mut ToolResultEvent) -> Option<ToolResultEventResult> {
		let ctx = match self.create_context() {
			Ok(ctx) => ctx,
			Err(_) => return None,
		};
		let mut modified = false;

		for snapshot in self.snapshot_event_handlers("tool_result") {
			for handler in snapshot.handlers.clone() {
				let event_json = match serialize_tagged("tool_result", event) {
					Some(event_json) => event_json,
					None => continue,
				};
				match handler(event_json, ctx.clone()).await {
					Ok(result) => {
						let Some(result) = result else { continue };
						let Ok(patch) = serde_json::from_value::<ToolResultEventResult>(result) else {
							continue;
						};
						if patch.content.is_some() {
							event.set_content(patch.content.clone().unwrap_or_default());
							// Structured content that is not replaced along
							// with the content may no longer match it.
							if patch.structured_content.is_none() {
								event.clear_structured_content();
							}
							modified = true;
						}
						if patch.details.is_some() {
							event.set_details(patch.details.clone().unwrap_or(JsonValue::Null));
							modified = true;
						}
						if patch.structured_content.is_some() {
							event.set_structured_content(patch.structured_content.clone());
							modified = true;
						}
						if patch.is_error.is_some() {
							event.set_is_error(patch.is_error.unwrap());
							modified = true;
						}
						if patch.usage.is_some() {
							event.set_usage(patch.usage.clone());
							modified = true;
						}
					}
					Err(message) => self.emit_error(&ExtensionError {
						extension_path: snapshot.extension_path.clone(),
						event: "tool_result".to_string(),
						error: message,
						stack: None,
					}),
				}
			}
		}

		if !modified {
			return None;
		}

		Some(ToolResultEventResult {
			content: Some(event.content().to_vec()),
			details: Some(event.details().clone()),
			structured_content: event.structured_content().cloned(),
			is_error: Some(event.is_error()),
			usage: event.usage().cloned(),
		})
	}

	/// `emitToolCall`: first `block: true` result short-circuits. A handler
	/// may patch the call arguments via the result's `input` field (D40;
	/// upstream mutates the shared event in place).
	pub async fn emit_tool_call(&self, event: &mut ToolCallEvent) -> Option<ToolCallEventResult> {
		let ctx = self.create_context().ok()?;
		let mut result: Option<ToolCallEventResult> = None;

		for snapshot in self.snapshot_event_handlers("tool_call") {
			for handler in snapshot.handlers.clone() {
				let event_json = match serialize_tagged("tool_call", event) {
					Some(event_json) => event_json,
					None => continue,
				};
				let handler_result = match handler(event_json, ctx.clone()).await {
					Ok(result) => result,
					Err(message) => {
						self.emit_error(&ExtensionError {
							extension_path: snapshot.extension_path.clone(),
							event: "tool_call".to_string(),
							error: message,
							stack: None,
						});
						continue;
					}
				};
				if let Some(handler_result) = handler_result {
					if let Some(input) = handler_result.get("input") {
						if input.is_object() {
							event.set_input(input.clone());
						}
					}
					let parsed: ToolCallEventResult = serde_json::from_value(handler_result.clone()).unwrap_or_default();
					if parsed.block == Some(true) {
						return Some(parsed);
					}
					result = Some(parsed);
				}
			}
		}

		result
	}

	/// `emitUserBash`: the first valid `{ operations }` or `{ result }`
	/// override wins; handler failures and invalid results fail closed
	/// (`Err`).
	pub async fn emit_user_bash(&self, event: &UserBashEvent) -> Result<JsonValue, String> {
		let ctx = match self.create_context() {
			Ok(ctx) => ctx,
			Err(message) => return Err(message),
		};

		for snapshot in self.snapshot_event_handlers("user_bash") {
			for handler in snapshot.handlers.clone() {
				let event_json = match serialize_tagged("user_bash", event) {
					Some(event_json) => event_json,
					None => continue,
				};
				let handler_result = handler(event_json, ctx.clone()).await;
				match handler_result {
					Ok(Some(result)) => {
						if !is_valid_user_bash_result(&result) {
							let message = "Invalid user_bash handler result: return undefined for local execution or exactly one valid { operations } or { result } object".to_string();
							self.emit_error(&ExtensionError {
								extension_path: snapshot.extension_path.clone(),
								event: "user_bash".to_string(),
								error: message.clone(),
								stack: None,
							});
							return Err(message);
						}
						return Ok(result);
					}
					Ok(None) => continue,
					Err(message) => {
						self.emit_error(&ExtensionError {
							extension_path: snapshot.extension_path.clone(),
							event: "user_bash".to_string(),
							error: message.clone(),
							stack: None,
						});
						return Err(message);
					}
				}
			}
		}

		Ok(JsonValue::Null)
	}

	/// Run the request-time transforms in two phases (`emitContext`):
	/// `context` handlers see the conversation only and the prompt/tool
	/// state is restored after each; `context_with_system` handlers see the
	/// full transcript and their output is used as returned.
	pub async fn emit_context(&self, messages: &[AgentMessage]) -> Vec<AgentMessage> {
		let Ok(ctx) = self.create_context() else {
			return messages.to_vec();
		};
		let mut current_messages = serde_roundtrip(messages.to_vec()).unwrap_or_else(|| messages.to_vec());

		for snapshot in self.snapshot_event_handlers("context") {
			for handler in snapshot.handlers.clone() {
				let visible_messages: Vec<AgentMessage> = current_messages
					.iter()
					.filter(|message| !is_system_message(message))
					.cloned()
					.collect();
				let visible_snapshot = visible_messages.clone();
				let event = ContextEvent { messages: visible_messages };
				let Some(event_json) = serialize_tagged("context", &event) else {
					continue;
				};
				match handler(event_json, ctx.clone()).await {
					Ok(result) => {
						let returned = result
							.and_then(|result| serde_json::from_value::<super::types::ContextEventResult>(result).ok())
							.and_then(|parsed| parsed.messages);
						let Some(returned) = returned else { continue };
						current_messages = restore_system_messages(current_messages, &visible_snapshot, returned);
					}
					Err(message) => self.emit_error(&ExtensionError {
						extension_path: snapshot.extension_path.clone(),
						event: "context".to_string(),
						error: message,
						stack: None,
					}),
				}
			}
		}

		for snapshot in self.snapshot_event_handlers("context_with_system") {
			for handler in snapshot.handlers.clone() {
				let had_leading_system_message = current_messages.first().is_some_and(is_system_message);
				let event = super::types::ContextWithSystemEvent { messages: current_messages.clone() };
				let Some(event_json) = serialize_tagged("context_with_system", &event) else {
					continue;
				};
				match handler(event_json, ctx.clone()).await {
					Ok(result) => {
						if let Some(parsed) = result
							.and_then(|result| serde_json::from_value::<super::types::ContextEventResult>(result).ok())
							.and_then(|parsed| parsed.messages)
						{
							current_messages = parsed;
						}
						// Providers read the prompt and initial tools from
						// the leading system message; losing it is never
						// intended. Report it but honor the output.
						if had_leading_system_message && !current_messages.first().is_some_and(is_system_message) {
							self.emit_error(&ExtensionError {
								extension_path: snapshot.extension_path.clone(),
								event: "context_with_system".to_string(),
								error: "Handler removed the leading system message; the request has no prompt or initial tool declarations. Keep it at index 0 or replace a dropped prefix with getCurrentSystemMessage().".to_string(),
								stack: None,
							});
						}
					}
					Err(message) => self.emit_error(&ExtensionError {
						extension_path: snapshot.extension_path.clone(),
						event: "context_with_system".to_string(),
						error: message,
						stack: None,
					}),
				}
			}
		}

		current_messages
	}

	pub async fn emit_before_provider_request(&self, payload: JsonValue) -> JsonValue {
		let Ok(ctx) = self.create_context() else {
			return payload;
		};
		let mut current_payload = payload;

		for snapshot in self.snapshot_event_handlers("before_provider_request") {
			for handler in snapshot.handlers.clone() {
				let event = super::types::BeforeProviderRequestEvent { payload: current_payload.clone() };
				let Some(event_json) = serialize_tagged("before_provider_request", &event) else {
					continue;
				};
				match handler(event_json, ctx.clone()).await {
					Ok(Some(result)) => current_payload = result,
					Ok(None) => {}
					Err(message) => self.emit_error(&ExtensionError {
						extension_path: snapshot.extension_path.clone(),
						event: "before_provider_request".to_string(),
						error: message,
						stack: None,
					}),
				}
			}
		}

		current_payload
	}

	/// `emitBeforeProviderHeaders`. Handlers may patch headers via the
	/// handler result object (D40; upstream mutates the shared map in
	/// place); patches shallow-merge, so untouched keys survive.
	pub async fn emit_before_provider_headers(&self, headers: ProviderHeaders) -> ProviderHeaders {
		let Ok(ctx) = self.create_context() else {
			return headers;
		};
		let mut current: serde_json::Map<String, JsonValue> =
			serde_json::to_value(&headers).unwrap_or(JsonValue::Null).as_object().cloned().unwrap_or_default();

		for snapshot in self.snapshot_event_handlers("before_provider_headers") {
			for handler in snapshot.handlers.clone() {
				let event = super::types::BeforeProviderHeadersEvent {
					headers: serde_json::from_value(JsonValue::Object(current.clone()))
						.unwrap_or_default(),
				};
				let Some(event_json) = serialize_tagged("before_provider_headers", &event) else {
					continue;
				};
				match handler(event_json, ctx.clone()).await {
					Ok(Some(result)) => {
						if let Some(patch) = result.as_object() {
							for (key, value) in patch {
								current.insert(key.clone(), value.clone());
							}
						}
					}
					Ok(None) => {}
					Err(message) => self.emit_error(&ExtensionError {
						extension_path: snapshot.extension_path.clone(),
						event: "before_provider_headers".to_string(),
						error: message,
						stack: None,
					}),
				}
			}
		}

		serde_json::from_value(JsonValue::Object(current)).unwrap_or(headers)
	}

	/// `emitBeforeAgentStart`: chains message drafts and system-prompt
	/// overrides; `ctx.getSystemPrompt()` stays in sync with the chained
	/// updates.
	pub async fn emit_before_agent_start(
		&self,
		prompt: &str,
		images: Option<Vec<ImageContent>>,
		system_prompt_options: &BuildSystemPromptOptions,
	) -> Result<BeforeAgentStartCombinedResult, String> {
		let current_options: Arc<parking_lot::Mutex<NormalizedBuildSystemPromptOptions>> =
			Arc::new(parking_lot::Mutex::new(system_prompt_options.clone().into()));
		let ctx = {
			// The context's `getSystemPrompt` renders the current options.
			let options_cell = Arc::clone(&current_options);
			let renderer = self.system_prompt_renderer.lock().clone();
			let get_system_prompt: Arc<dyn Fn() -> String + Send + Sync> = Arc::new(move || {
				let options = options_cell.lock().clone();
				renderer(&options)
			});
			self.build_context(get_system_prompt)?
		};
		let mut messages: Vec<CustomMessageDraft> = Vec::new();

		for snapshot in self.snapshot_event_handlers("before_agent_start") {
			for handler in snapshot.handlers.clone() {
				let event_json = serde_json::json!({
					"type": "before_agent_start",
					"prompt": prompt,
					"images": images,
					"systemPrompt": self.render_system_prompt(&current_options.lock()),
					"systemPromptOptions": normalized_options_to_json(&current_options.lock()),
				});
				match handler(event_json, ctx.clone()).await {
					Ok(result) => {
						let Some(result) = result else { continue };
						let Ok(parsed) = serde_json::from_value::<BeforeAgentStartEventResult>(result) else {
							continue;
						};
						if let Some(message) = parsed.message {
							messages.push(message);
						}
						if let Some(system_prompt) = parsed.system_prompt {
							current_options.lock().force_system_prompt = Some(system_prompt);
						}
					}
					Err(message) => self.emit_error(&ExtensionError {
						extension_path: snapshot.extension_path.clone(),
						event: "before_agent_start".to_string(),
						error: message,
						stack: None,
					}),
				}
			}
		}

		let options = current_options.lock().clone();
		Ok(BeforeAgentStartCombinedResult {
			messages,
			system_prompt_options: options,
		})
	}

	/// `emitResourcesDiscover`: collects skill/prompt/theme paths annotated
	/// with the discovering extension's path.
	pub async fn emit_resources_discover(
		&self,
		cwd: &str,
		reason: ResourcesDiscoverReason,
	) -> ResourcesDiscoverOutcome {
		let mut outcome = ResourcesDiscoverOutcome::default();
		let Ok(ctx) = self.create_context() else {
			return outcome;
		};

		for snapshot in self.snapshot_event_handlers("resources_discover") {
			for handler in snapshot.handlers.clone() {
				let event = super::types::ResourcesDiscoverEvent {
					cwd: cwd.to_string(),
					reason,
				};
				let Some(event_json) = serialize_tagged("resources_discover", &event) else {
					continue;
				};
				match handler(event_json, ctx.clone()).await {
					Ok(result) => {
						let Some(result) = result else { continue };
						let Ok(parsed) = serde_json::from_value::<super::types::ResourcesDiscoverResult>(result) else {
							continue;
						};
						outcome
							.skill_paths
							.extend(parsed.skill_paths.unwrap_or_default().into_iter().map(|path| (path, snapshot.extension_path.clone())));
						outcome
							.prompt_paths
							.extend(parsed.prompt_paths.unwrap_or_default().into_iter().map(|path| (path, snapshot.extension_path.clone())));
						outcome
							.theme_paths
							.extend(parsed.theme_paths.unwrap_or_default().into_iter().map(|path| (path, snapshot.extension_path.clone())));
					}
					Err(message) => self.emit_error(&ExtensionError {
						extension_path: snapshot.extension_path.clone(),
						event: "resources_discover".to_string(),
						error: message,
						stack: None,
					}),
				}
			}
		}

		outcome
	}

	/// `emitInput`: transforms chain, `handled` short-circuits.
	pub async fn emit_input(
		&self,
		text: &str,
		images: Option<Vec<ImageContent>>,
		source: InputSource,
		streaming_behavior: Option<super::types::StreamingBehavior>,
	) -> InputEventResult {
		let Ok(ctx) = self.create_context() else {
			return InputEventResult::Continue;
		};
		let initial_images = images.clone();
		let mut current_text = text.to_string();
		let mut current_images = images;

		for snapshot in self.snapshot_event_handlers("input") {
			for handler in snapshot.handlers.clone() {
				let event = InputEvent {
					text: current_text.clone(),
					images: current_images.clone(),
					source,
					streaming_behavior,
				};
				let Some(event_json) = serialize_tagged("input", &event) else {
					continue;
				};
				match handler(event_json, ctx.clone()).await {
					Ok(result) => {
						let Some(result) = result else { continue };
						match serde_json::from_value::<InputEventResult>(result) {
							Ok(InputEventResult::Handled) => return InputEventResult::Handled,
							Ok(InputEventResult::Transform { text, images }) => {
								current_text = text;
								if images.is_some() {
									current_images = images;
								}
							}
							Ok(InputEventResult::Continue) => {}
							Err(_) => {}
						}
					}
					Err(message) => self.emit_error(&ExtensionError {
						extension_path: snapshot.extension_path.clone(),
						event: "input".to_string(),
						error: message,
						stack: None,
					}),
				}
			}
		}

		if current_text != text || current_images != initial_images {
			InputEventResult::Transform { text: current_text, images: current_images }
		} else {
			InputEventResult::Continue
		}
	}

	/// `emitBoundary`: chains boundary draft proposals across
	/// `turn_end`/`agent_before_settle` handlers. Each handler observes the
	/// entries proposed so far and the current preview; entry proposals flow
	/// through the handler result's `entries` (D40).
	pub async fn emit_boundary(
		&self,
		event_type: &str,
		base_event: JsonValue,
		build_context: BoundaryContextBuilder,
	) -> BoundaryDispatchResult {
		let Ok(ctx) = self.create_context() else {
			return BoundaryDispatchResult {
				entries: Vec::new(),
				continue_: false,
				context: BoundaryContextPreview::default(),
				valid: false,
			};
		};
		let mut entries: Vec<SessionBoundaryDraft> = Vec::new();
		let mut should_continue = false;
		let mut context = build_context(&entries).unwrap_or_else(|message| {
			self.emit_error(&ExtensionError {
				extension_path: String::new(),
				event: event_type.to_string(),
				error: format!("Invalid boundary entries: {message}"),
				stack: None,
			});
			BoundaryContextPreview::default()
		});
		let mut valid = true;

		for snapshot in self.snapshot_event_handlers(event_type) {
			for handler in snapshot.handlers.clone() {
				let event = {
					let mut event = base_event.clone();
					event["type"] = JsonValue::from(event_type);
					event["entries"] = serde_json::to_value(&entries).unwrap_or(JsonValue::Array(Vec::new()));
					event["continue"] = JsonValue::from(should_continue);
					event["context"] = boundary_context_to_json(&context);
					event
				};
				match handler(event, ctx.clone()).await {
					Ok(result) => {
						if let Ok(parsed) = serde_json::from_value::<BoundaryResult>(result.unwrap_or(JsonValue::Null)) {
							if let Some(proposed) = parsed.entries {
								entries = proposed;
							}
							if let Some(continue_) = parsed.continue_ {
								should_continue = continue_;
							}
						}
					}
					Err(message) => self.emit_error(&ExtensionError {
						extension_path: snapshot.extension_path.clone(),
						event: event_type.to_string(),
						error: message,
						stack: None,
					}),
				}

				match build_context(&entries) {
					Ok(rebuilt) => {
						context = rebuilt;
						valid = true;
					}
					Err(message) => {
						valid = false;
						self.emit_error(&ExtensionError {
							extension_path: snapshot.extension_path.clone(),
							event: event_type.to_string(),
							error: format!("Invalid boundary entries: {message}"),
							stack: None,
						});
					}
				}
			}
		}

		if valid {
			BoundaryDispatchResult { entries, continue_: should_continue, context, valid: true }
		} else {
			BoundaryDispatchResult {
				entries: Vec::new(),
				continue_: false,
				context,
				valid: false,
			}
		}
	}
}

/// Serialized boundary preview. `context_entries` are
/// [`ProjectedSessionEntry`]s without serde (D30); handlers observe their
/// count via `contextEntryCount` and the message arrays in full.
fn boundary_context_to_json(preview: &BoundaryContextPreview) -> JsonValue {
	serde_json::json!({
		"contextEntryCount": preview.context_entries.len(),
		"contextMessages": preview.context_messages,
		"llmMessages": preview.llm_messages,
		"pendingMessages": preview.pending_messages,
		"canContinue": preview.can_continue,
	})
}

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
	payload
		.downcast_ref::<String>()
		.cloned()
		.or_else(|| payload.downcast_ref::<&str>().map(|message| message.to_string()))
		.unwrap_or_else(|| "unknown error".to_string())
}

fn nested_tool_call_unavailable(tool_call_id: &str, name: &str) -> pi_agent::AgentToolCallOutcome {
	pi_agent::AgentToolCallOutcome {
		tool_call: pi_ai::types::ToolCall {
			id: format!("{tool_call_id}/0"),
			name: name.to_string(),
			arguments: serde_json::Map::new(),
			thought_signature: None,
			namespace: None,
		},
		result: pi_agent::AgentToolResult {
			content: vec![pi_ai::types::UserContent::Text(pi_ai::types::TextContent {
				text: "Nested tool calls are not available in this context".to_string(),
				text_signature: None,
			})],
			details: JsonValue::Null,
			structured_content: None,
			is_error: Some(true),
			terminate: None,
			usage: None,
		},
		is_error: true,
	}
}

/// Re-attach the prompt and tool state after a `context` handler
/// (`restoreSystemMessages`). An unchanged conversation keeps every system
/// message in place; a changed one gets the replayed prompt as one leading
/// system message.
fn restore_system_messages(
	current: Vec<AgentMessage>,
	visible: &[AgentMessage],
	returned: Vec<AgentMessage>,
) -> Vec<AgentMessage> {
	let visible_json = serde_json::to_value(visible).unwrap_or(JsonValue::Null);
	let returned_json = serde_json::to_value(&returned).unwrap_or(JsonValue::Null);
	if returned_json == visible_json {
		return current;
	}
	// `getCurrentSystemMessage` on the current conversation.
	let llm_messages: Vec<Message> = current
		.iter()
		.filter_map(|message| match message {
			AgentMessage::Llm(message) => Some(message.clone()),
			AgentMessage::Custom(_) => None,
		})
		.collect();
	let head = get_current_system_message(&llm_messages);
	match head {
		Some(head) => {
			let mut messages = vec![AgentMessage::system(head)];
			messages.extend(returned);
			messages
		}
		None => returned,
	}
}

fn serde_roundtrip<T: Serialize + serde::de::DeserializeOwned>(value: T) -> Option<T> {
	serde_json::from_value(serde_json::to_value(value).ok()?).ok()
}

fn serialize_tagged<T: Serialize>(tag: &str, value: &T) -> Option<JsonValue> {
	let mut json = serde_json::to_value(value).ok()?;
	if json.get("type").is_none() {
		if let Some(object) = json.as_object_mut() {
			object.insert("type".to_string(), JsonValue::from(tag));
		}
	}
	Some(json)
}

fn json_tagged<T: Serialize>(tag: &str, value: &T) -> Option<JsonValue> {
	serialize_tagged(tag, value)
}

fn normalized_options_to_json(options: &NormalizedBuildSystemPromptOptions) -> JsonValue {
	// `skills` carry no serde (D30); the remaining fields round-trip.
	serde_json::json!({
		"customPrompt": options.custom_prompt,
		"forceSystemPrompt": options.force_system_prompt,
		"selectedTools": options.selected_tools,
		"toolSnippets": options.tool_snippets,
		"toolGuidelines": options.tool_guidelines,
		"promptGuidelines": options.prompt_guidelines,
		"appendSystemPrompt": options.append_system_prompt,
		"sections": options.sections,
		"cwd": options.cwd,
		"contextFiles": options.context_files,
	})
}

/// `isUserBashEventResult`: exactly one of `{ operations: object }` or a
/// complete `{ result }` object. The `operations.exec` function check is
/// bridge work (a JSON channel cannot carry functions — D43).
fn is_valid_user_bash_result(value: &JsonValue) -> bool {
	let Some(object) = value.as_object() else {
		return false;
	};
	let has_operations = object.contains_key("operations");
	let has_result = object.contains_key("result");
	if has_operations == has_result {
		return false;
	}
	if has_operations {
		return object.get("operations").is_some_and(JsonValue::is_object);
	}
	let Some(result) = object.get("result") else {
		return false;
	};
	let Some(result) = result.as_object() else {
		return false;
	};
	result.get("output").is_some_and(JsonValue::is_string)
		&& result.get("exitCode").map(|exit_code| exit_code.is_number() || exit_code.is_null()).unwrap_or(true)
		&& result.get("cancelled").is_some_and(JsonValue::is_boolean)
		&& result.get("truncated").is_some_and(JsonValue::is_boolean)
		&& result
			.get("fullOutputPath")
			.map(|path| path.is_string() || path.is_null())
			.unwrap_or(true)
}

/// Emit `session_shutdown` when any extension listens for it
/// (`emitSessionShutdownEvent`); `false` when there were no handlers.
pub async fn emit_session_shutdown_event(
	extension_runner: &ExtensionRunner,
	event: &super::types::SessionShutdownEvent,
) -> bool {
	if extension_runner.has_handlers("session_shutdown") {
		if let Some(event) = serialize_tagged("session_shutdown", event) {
			extension_runner.emit("session_shutdown", event).await;
		}
		return true;
	}
	false
}

/// Emit `project_trust` across a load result's extensions
/// (`emitProjectTrustEvent`): the first handler returning a yes/no decision
/// wins; undecided handlers fall through.
pub async fn emit_project_trust_event(
	extensions_result: &LoadExtensionsResult,
	event: &ProjectTrustEvent,
	ctx: ExtensionContext,
) -> (Option<ProjectTrustEventResult>, Vec<ExtensionError>) {
	let mut errors: Vec<ExtensionError> = Vec::new();
	for extension in &extensions_result.extensions {
		let handlers = extension.handlers.lock().get("project_trust").cloned().unwrap_or_default();
		for handler in handlers {
			let event_json = match serialize_tagged("project_trust", event) {
				Some(event_json) => event_json,
				None => continue,
			};
			match handler(event_json, ctx.clone()).await {
				Ok(result) => {
					let Some(result) = result else { continue };
					let Ok(parsed) = serde_json::from_value::<ProjectTrustEventResult>(result) else {
						continue;
					};
					if parsed.trusted == super::types::ProjectTrustEventDecision::Undecided {
						continue;
					}
					return (Some(parsed), errors);
				}
				Err(message) => errors.push(ExtensionError {
					extension_path: extension.path.clone(),
					event: "project_trust".to_string(),
					error: message,
					stack: None,
				}),
			}
		}
	}
	(None, errors)
}


// ============================================================================
// Tests — ported from `test/extensions-runner.test.ts` and the runner halves
// of `test/suite/regressions/`. Extensions load through in-process factories
// (`load_extension_from_factory`); the jiti/file suites run against the
// loader seam in pi-runtime.
// ============================================================================

#[cfg(test)]
pub(crate) mod runner_tests {
	use super::super::types::{
		ExtensionApiHandle, FlagOptions, ProviderConfig, RenderShell, ScopedModel, SessionSwitchOutcome,
	};
	use super::*;
	use crate::core::extensions::loader::{create_extension_runtime, load_extension_from_factory};
	use parking_lot::Mutex as PlMutex;
	use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

	struct TestEventBus;
	impl crate::core::extensions::types::EventBus for TestEventBus {
		fn emit(&self, _channel: &str, _data: JsonValue) {}
		fn on(&self, _channel: &str, _handler: crate::core::extensions::types::EventBusHandler) -> Unsubscribe {
			Box::new(|| {})
		}
	}

	fn test_event_bus() -> Arc<dyn crate::core::extensions::types::EventBus> {
		Arc::new(TestEventBus)
	}

	/// `HandlerFn` constructor: wrap an async closure.
	fn handler<F, Fut>(f: F) -> HandlerFn
	where
		F: Fn(JsonValue, ExtensionContext) -> Fut + Send + Sync + 'static,
		Fut: std::future::Future<Output = Result<Option<JsonValue>, String>> + Send + 'static,
	{
		Arc::new(move |event, ctx| Box::pin(f(event, ctx)))
	}

	fn factory(body: impl Fn(ExtensionApiHandle) + Send + Sync + 'static) -> super::super::types::ExtensionFactory {
		Arc::new(move |pi: ExtensionApiHandle| {
			body(Arc::clone(&pi));
			Box::pin(async { Ok(()) })
		})
	}

	pub(crate) fn test_runner(extensions: Vec<Extension>, runtime: super::super::types::ExtensionRuntime) -> ExtensionRunner {
		let session_manager = Arc::new(PlMutex::new(SessionManagerType::in_memory(None, None, None).expect("in-memory session")));
		let model_registry = super::super::types::offline_registry_for_tests();
		ExtensionRunner::new(extensions, runtime, "/tmp/pi-runner-test", session_manager, model_registry)
	}

	fn default_actions() -> ExtensionActions {
		ExtensionActions {
			set_model: Some(Arc::new(|_model| Box::pin(async { false }))),
			get_thinking_level: Some(Arc::new(|| pi_agent::AgentThinkingLevel::Off)),
			..Default::default()
		}
	}

	fn default_context_actions() -> ExtensionContextActions {
		ExtensionContextActions::default()
	}

	fn collect_errors(runner: &ExtensionRunner) -> Arc<PlMutex<Vec<ExtensionError>>> {
		let errors = Arc::new(PlMutex::new(Vec::new()));
		let captured = Arc::clone(&errors);
		runner.on_error(Arc::new(move |error| captured.lock().push(error.clone())));
		errors
	}

	// Upstream `KeybindingsManager().getEffectiveConfig()` subset the
	// shortcut-conflict suites exercise (D45: inline resolved maps; the
	// keybindings module ports separately).
	fn default_keybindings() -> ResolvedKeybindings {
		ResolvedKeybindings::from([
			("app.interrupt".to_string(), vec!["ctrl+c".to_string()]),
			("app.clipboard.pasteImage".to_string(), vec!["ctrl+shift+v".to_string()]),
			("app.model.cycleForward".to_string(), vec!["ctrl+p".to_string()]),
		])
	}

	// -- scopedModels -----------------------------------------------------------

	#[tokio::test]
	async fn scoped_models_reflect_get_scoped_models_action() {
		let runtime = create_extension_runtime();
		let mut runner = test_runner(Vec::new(), runtime);

		// Before bind_core the default is an empty list (never undefined).
		assert!(runner.create_context().unwrap().scoped_models.is_empty());

		let mut actions = default_context_actions();
		actions.get_scoped_models = Some(Arc::new(|| {
			vec![ScopedModel { model: test_chat_model("scoped-test"), thinking_level: None }]
		}));
		runner.bind_core(default_actions(), actions, None);
		assert_eq!(runner.create_context().unwrap().scoped_models.len(), 1);
	}

	fn test_chat_model(id: &str) -> pi_ai::types::Model {
		pi_ai::types::Model {
			base: pi_ai::types::BaseModelFields {
				id: id.to_string(),
				name: id.to_string(),
				api: "anthropic-messages".to_string(),
				provider: "anthropic".to_string(),
				base_url: "https://api.anthropic.com".to_string(),
				input: vec![pi_ai::types::ModelInput::Text],
				input_limits: None,
				cost: pi_ai::types::ModelCost {
					rates: pi_ai::types::ModelCostRates { input: 1.0, output: 15.0, cache_read: 0.1, cache_write: 1.0 },
					tiers: None,
				},
				headers: None,
			},
			model_type: None,
			reasoning: false,
			thinking_level_map: None,
			prompt_cache: None,
			context_window: 128_000,
			max_tokens: 4_096,
			sampling_params: None,
			compat: None,
			sampling_params_by_thinking_level: None,
		}
	}

	// -- project_trust -----------------------------------------------------------

	#[tokio::test]
	async fn project_trust_skips_undecided_and_returns_first_decision() {
		let runtime = create_extension_runtime();
		let undecided = load_extension_from_factory(
			factory(|pi| {
				pi.on(
					"project_trust",
					handler(|_event, _ctx| async { Ok(Some(serde_json::json!({ "trusted": "undecided", "remember": true }))) }),
				);
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<undecided>",
		)
		.await
		.unwrap();
		let decided = load_extension_from_factory(
			factory(|pi| {
				pi.on(
					"project_trust",
					handler(|_event, _ctx| async { Ok(Some(serde_json::json!({ "trusted": "no", "remember": true }))) }),
				);
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<decided>",
		)
		.await
		.unwrap();

		let extensions_result = super::super::types::LoadExtensionsResult {
			extensions: vec![undecided, decided],
			errors: Vec::new(),
			warnings: None,
			runtime,
		};
		let event = super::super::types::ProjectTrustEvent { cwd: "/tmp".to_string() };
		let (result, errors) = emit_project_trust_event(&extensions_result, &event, stub_context("/tmp").await).await;

		let result = result.expect("first decision wins");
		assert_eq!(
			serde_json::to_value(&result).unwrap(),
			serde_json::json!({ "trusted": "no", "remember": true })
		);
		assert!(errors.is_empty());
	}

	async fn stub_context(cwd: &str) -> ExtensionContext {
		// ModelRegistry needs a ModelRuntime since the D64 port replaced the
		// unit stand-in; the stub runs fully in memory (no network, no files).
		let runtime = crate::core::model_runtime::ModelRuntime::create(crate::core::model_runtime::CreateModelRuntimeOptions {
			credentials: Some(Arc::new(pi_ai::InMemoryCredentialStore::new())),
			models_path: Some(None),
			models_store: Some(Arc::new(crate::core::models_store_wrapper::InMemoryCodingAgentModelsStore::new())),
			allow_model_network: false,
			refresh_on_create: false,
			..Default::default()
		})
		.await
		.expect("ModelRuntime::create succeeds");
		ExtensionContext {
			ui: Arc::new(NoopExtensionUi),
			mode: ExtensionMode::Tui,
			has_ui: false,
			cwd: cwd.to_string(),
			session_manager: Arc::new(PlMutex::new(SessionManagerType::in_memory(None, None, None).unwrap())),
			model_registry: Arc::new(super::super::types::ModelRegistry::new(runtime)),
			model: None,
			scoped_models: Vec::new(),
			thinking_level: None,
			signal: None,
			is_idle: Arc::new(|| true),
			is_project_trusted: Arc::new(|| true),
			abort: Arc::new(|| {}),
			has_pending_messages: Arc::new(|| false),
			shutdown: Arc::new(|| {}),
			get_context_usage: Arc::new(|| None),
			compact: Arc::new(|_| {}),
			get_system_prompt: Arc::new(String::new),
		}
	}

	// -- shortcut conflicts --------------------------------------------------------

	async fn shortcut_extension(runtime: &super::super::types::ExtensionRuntime, key: &str) -> Extension {
		let path = format!("<shortcut:{key}>");
		let key = key.to_string();
		load_extension_from_factory(
			Arc::new(move |pi: ExtensionApiHandle| {
				pi.register_shortcut(
					key.clone(),
					super::super::types::ShortcutOptions {
						description: Some("Test shortcut".to_string()),
						handler: Arc::new(|_ctx| Box::pin(async {})),
					},
				);
				Box::pin(async { Ok(()) })
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			&path,
		)
		.await
		.unwrap()
	}

	#[tokio::test]
	async fn warns_and_skips_when_extension_shortcut_conflicts_with_reserved_builtin() {
		let runtime = create_extension_runtime();
		let extension = shortcut_extension(&runtime, "ctrl+c").await;
		let runner = test_runner(vec![extension], runtime);

		let shortcuts = runner.get_shortcuts(&default_keybindings());
		let diagnostics = runner.get_shortcut_diagnostics();
		assert!(
			diagnostics.iter().any(|d| d.message.contains("conflicts with built-in")),
			"expected conflict diagnostic, got {diagnostics:?}"
		);
		assert!(!shortcuts.contains_key("ctrl+c"));
	}

	#[tokio::test]
	async fn allows_shortcut_when_reserved_set_no_longer_contains_the_default_key() {
		let runtime = create_extension_runtime();
		let extension = shortcut_extension(&runtime, "ctrl+p").await;
		let runner = test_runner(vec![extension], runtime);

		// ctrl+p now only belongs to the non-reserved cycleForward.
		let mut keybindings = default_keybindings();
		keybindings.insert("app.model.cycleForward".to_string(), vec!["ctrl+n".to_string()]);
		let shortcuts = runner.get_shortcuts(&keybindings);
		assert!(runner.get_shortcut_diagnostics().is_empty());
		assert!(shortcuts.contains_key("ctrl+p"));
	}

	#[tokio::test]
	async fn warns_but_allows_when_extension_uses_non_reserved_builtin_shortcut() {
		let runtime = create_extension_runtime();
		let extension = shortcut_extension(&runtime, "ctrl+shift+v").await;
		let runner = test_runner(vec![extension], runtime);

		let shortcuts = runner.get_shortcuts(&default_keybindings());
		let diagnostics = runner.get_shortcut_diagnostics();
		assert!(diagnostics.iter().any(|d| d.message.contains("built-in shortcut for app.clipboard.pasteImage")));
		assert!(shortcuts.contains_key("ctrl+shift+v"));
	}

	#[tokio::test]
	async fn blocks_shortcuts_for_reserved_actions_even_when_rebound() {
		let runtime = create_extension_runtime();
		let extension = shortcut_extension(&runtime, "ctrl+x").await;
		let runner = test_runner(vec![extension], runtime);

		let mut keybindings = default_keybindings();
		keybindings.insert("app.interrupt".to_string(), vec!["ctrl+x".to_string()]);
		let shortcuts = runner.get_shortcuts(&keybindings);
		assert!(runner.get_shortcut_diagnostics().iter().any(|d| d.message.contains("conflicts with built-in")));
		assert!(!shortcuts.contains_key("ctrl+x"));
	}

	#[tokio::test]
	async fn blocks_shortcuts_when_reserved_key_is_also_bound_to_non_reserved_actions() {
		let runtime = create_extension_runtime();
		let extension = shortcut_extension(&runtime, "ctrl+p").await;
		let runner = test_runner(vec![extension], runtime);

		let mut keybindings = default_keybindings();
		keybindings.insert("app.interrupt".to_string(), vec!["ctrl+c".to_string(), "ctrl+p".to_string()]);
		let shortcuts = runner.get_shortcuts(&keybindings);
		assert!(runner.get_shortcut_diagnostics().iter().any(|d| d.message.contains("conflicts with built-in")));
		assert!(!shortcuts.contains_key("ctrl+p"));
	}

	#[tokio::test]
	async fn blocks_shortcuts_when_reserved_action_has_multiple_keys() {
		let runtime = create_extension_runtime();
		let extension = shortcut_extension(&runtime, "ctrl+y").await;
		let runner = test_runner(vec![extension], runtime);

		let mut keybindings = default_keybindings();
		keybindings.insert("app.clear".to_string(), vec!["ctrl+x".to_string(), "ctrl+y".to_string()]);
		let shortcuts = runner.get_shortcuts(&keybindings);
		assert!(runner.get_shortcut_diagnostics().iter().any(|d| d.message.contains("conflicts with built-in")));
		assert!(!shortcuts.contains_key("ctrl+y"));
	}

	#[tokio::test]
	async fn warns_but_allows_when_non_reserved_action_has_multiple_keys() {
		let runtime = create_extension_runtime();
		let extension = shortcut_extension(&runtime, "ctrl+y").await;
		let runner = test_runner(vec![extension], runtime);

		let mut keybindings = default_keybindings();
		keybindings.insert(
			"app.clipboard.pasteImage".to_string(),
			vec!["ctrl+x".to_string(), "ctrl+y".to_string()],
		);
		let shortcuts = runner.get_shortcuts(&keybindings);
		assert!(
			runner
				.get_shortcut_diagnostics()
				.iter()
				.any(|d| d.message.contains("built-in shortcut for app.clipboard.pasteImage"))
		);
		assert!(shortcuts.contains_key("ctrl+y"));
	}

	#[tokio::test]
	async fn warns_when_two_extensions_register_same_shortcut() {
		let runtime = create_extension_runtime();
		let first = shortcut_extension(&runtime, "ctrl+shift+x").await;
		let second = shortcut_extension(&runtime, "ctrl+shift+x").await;
		let runner = test_runner(vec![first, second], runtime);

		let shortcuts = runner.get_shortcuts(&default_keybindings());
		assert!(runner.get_shortcut_diagnostics().iter().any(|d| d.message.contains("shortcut conflict")));
		// Last one wins.
		assert!(shortcuts.contains_key("ctrl+shift+x"));
	}

	// -- tool collection -------------------------------------------------------------

	fn tool_factory(name: &str, description: &str, parameters: JsonValue) -> super::super::types::ExtensionFactory {
		let name = name.to_string();
		let description = description.to_string();
		Arc::new(move |pi: ExtensionApiHandle| {
			let name = name.clone();
			let description = description.clone();
			let parameters = parameters.clone();
			Box::pin(async move {
				let mut definition = super::super::types::ToolDefinition::new(name.clone(), name, description, parameters);
				definition.execute = Arc::new(
					move |_tool_call_id: String,
					      _args: JsonValue,
					      _signal: Option<crate::utils::abort::AbortSignal>,
					      _on_update,
					      _ctx| {
						Box::pin(async {
							Ok(pi_agent::AgentToolResult {
								content: Vec::new(),
								details: JsonValue::Null,
								structured_content: None,
								is_error: Some(false),
								terminate: None,
								usage: None,
							})
						})
					},
				);
				pi.register_tool(definition);
				Ok(())
			})
		})
	}

	#[tokio::test]
	async fn collects_tools_from_multiple_extensions() {
		let runtime = create_extension_runtime();
		let first = load_extension_from_factory(
			tool_factory("tool_a", "tool_a", serde_json::json!({"type": "object"})),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<tool-a>",
		)
		.await
		.unwrap();
		let second = load_extension_from_factory(
			tool_factory("tool_b", "tool_b", serde_json::json!({"type": "object"})),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<tool-b>",
		)
		.await
		.unwrap();
		let runner = test_runner(vec![first, second], runtime);

		let tools = runner.get_all_registered_tools();
		assert_eq!(tools.len(), 2);
		let mut names: Vec<&str> = tools.iter().map(|tool| tool.definition.name.as_str()).collect();
		names.sort_unstable();
		assert_eq!(names, ["tool_a", "tool_b"]);
	}

	// Regression test for #9300.
	#[tokio::test]
	async fn rejects_extension_tools_without_a_parameter_schema() {
		let runtime = create_extension_runtime();
		let error = load_extension_from_factory(
			tool_factory("noop", "No-op", serde_json::json!([1, 2])),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<missing-parameters>",
		)
		.await
		.unwrap_err();
		assert_eq!(
			error,
			"Tool \"noop\" registered by extension \"<missing-parameters>\" must define an object parameter schema."
		);
	}

	#[tokio::test]
	async fn keeps_first_tool_when_two_extensions_register_the_same_name() {
		let runtime = create_extension_runtime();
		let first = load_extension_from_factory(
			tool_factory("shared", "first", serde_json::json!({"type": "object"})),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<a-first>",
		)
		.await
		.unwrap();
		let second = load_extension_from_factory(
			tool_factory("shared", "second", serde_json::json!({"type": "object"})),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<b-second>",
		)
		.await
		.unwrap();
		let runner = test_runner(vec![first, second], runtime);

		let tools = runner.get_all_registered_tools();
		assert_eq!(tools.len(), 1);
		assert_eq!(tools[0].definition.description, "first");
	}

	// -- command collection --------------------------------------------------------------

	fn command_factory(name: &str, description: &str) -> super::super::types::ExtensionFactory {
		let name = name.to_string();
		let description = description.to_string();
		Arc::new(move |pi: ExtensionApiHandle| {
			let name = name.clone();
			let description = description.clone();
			Box::pin(async move {
				pi.register_command(
					&name,
					super::super::types::RegisteredCommandOptions {
						description: Some(description),
						get_argument_completions: None,
						handler: Arc::new(|_args, _ctx| Box::pin(async {})),
					},
				);
				Ok(())
			})
		})
	}

	#[tokio::test]
	async fn collects_commands_from_multiple_extensions() {
		let runtime = create_extension_runtime();
		let first = load_extension_from_factory(command_factory("cmd-a", "Test command"), "/tmp", test_event_bus(), Some(runtime.clone()), "<cmd-a>").await.unwrap();
		let second = load_extension_from_factory(command_factory("cmd-b", "Test command"), "/tmp", test_event_bus(), Some(runtime.clone()), "<cmd-b>").await.unwrap();
		let runner = test_runner(vec![first, second], runtime);

		let commands = runner.get_registered_commands();
		assert_eq!(commands.len(), 2);
		let mut names: Vec<&str> = commands.iter().map(|command| command.name.as_str()).collect();
		names.sort_unstable();
		assert_eq!(names, ["cmd-a", "cmd-b"]);
		let mut invocations: Vec<&str> = commands.iter().map(|command| command.invocation_name.as_str()).collect();
		invocations.sort_unstable();
		assert_eq!(invocations, ["cmd-a", "cmd-b"]);
	}

	#[tokio::test]
	async fn gets_command_by_invocation_name() {
		let runtime = create_extension_runtime();
		let extension = load_extension_from_factory(command_factory("my-cmd", "My command"), "/tmp", test_event_bus(), Some(runtime.clone()), "<cmd>").await.unwrap();
		let runner = test_runner(vec![extension], runtime);

		let command = runner.get_command("my-cmd").expect("command found");
		assert_eq!(command.name, "my-cmd");
		assert_eq!(command.invocation_name, "my-cmd");
		assert_eq!(command.description.as_deref(), Some("My command"));
		assert!(runner.get_command("not-exists").is_none());
	}

	#[tokio::test]
	async fn suffixes_duplicate_extension_commands_in_insertion_order() {
		let runtime = create_extension_runtime();
		let first = load_extension_from_factory(command_factory("shared-cmd", "First command"), "/tmp", test_event_bus(), Some(runtime.clone()), "<cmd-a>").await.unwrap();
		let second = load_extension_from_factory(command_factory("shared-cmd", "Second command"), "/tmp", test_event_bus(), Some(runtime.clone()), "<cmd-b>").await.unwrap();
		let runner = test_runner(vec![first, second], runtime);

		let commands = runner.get_registered_commands();
		assert_eq!(commands.len(), 2);
		assert_eq!(commands[0].name, "shared-cmd");
		assert_eq!(commands[1].name, "shared-cmd");
		assert_eq!(commands[0].invocation_name, "shared-cmd:1");
		assert_eq!(commands[1].invocation_name, "shared-cmd:2");
		assert_eq!(commands[0].description.as_deref(), Some("First command"));
		assert_eq!(commands[1].description.as_deref(), Some("Second command"));
		assert!(runner.get_command_diagnostics().is_empty());
		assert_eq!(runner.get_command("shared-cmd:1").unwrap().description.as_deref(), Some("First command"));
		assert_eq!(runner.get_command("shared-cmd:2").unwrap().description.as_deref(), Some("Second command"));
	}

	// -- context creation --------------------------------------------------------------------

	#[tokio::test]
	async fn exposes_the_current_abort_signal_on_extension_context() {
		let runtime = create_extension_runtime();
		let mut runner = test_runner(Vec::new(), runtime);
		let controller = crate::utils::abort::AbortController::new();

		let mut actions = default_context_actions();
		let signal = controller.signal().clone();
		actions.get_signal = Some(Arc::new(move || Some(signal.clone())));
		runner.bind_core(default_actions(), actions, None);

		let ctx = runner.create_context().unwrap();
		let signal = ctx.signal.clone().expect("signal");
		assert!(!signal.aborted());
		controller.abort();
		assert!(signal.aborted());
	}

	#[tokio::test]
	async fn exposes_print_mode_and_has_ui_false_by_default() {
		let runtime = create_extension_runtime();
		let mut runner = test_runner(Vec::new(), runtime);
		runner.bind_core(default_actions(), default_context_actions(), None);

		let ctx = runner.create_context().unwrap();
		assert_eq!(ctx.mode, ExtensionMode::Print);
		assert!(!ctx.has_ui);
	}

	#[tokio::test]
	async fn exposes_project_trust_state_on_extension_context() {
		let runtime = create_extension_runtime();
		let mut runner = test_runner(Vec::new(), runtime);
		let mut actions = default_context_actions();
		actions.is_project_trusted = Some(Arc::new(|| false));
		runner.bind_core(default_actions(), actions, None);

		assert!(!runner.create_context().unwrap().is_project_trusted());
	}

	#[tokio::test]
	async fn exposes_rpc_mode_with_has_ui_true_when_an_rpc_ui_context_is_provided() {
		let runtime = create_extension_runtime();
		let mut runner = test_runner(Vec::new(), runtime);
		runner.bind_core(default_actions(), default_context_actions(), None);
		runner.set_ui_context(Some(Arc::new(NoopExtensionUi)), ExtensionMode::Rpc);

		let ctx = runner.create_context().unwrap();
		assert_eq!(ctx.mode, ExtensionMode::Rpc);
		assert!(ctx.has_ui);
	}

	#[tokio::test]
	async fn exposes_tui_mode_with_has_ui_true_when_a_tui_ui_context_is_provided() {
		let runtime = create_extension_runtime();
		let mut runner = test_runner(Vec::new(), runtime);
		runner.bind_core(default_actions(), default_context_actions(), None);
		runner.set_ui_context(Some(Arc::new(NoopExtensionUi)), ExtensionMode::Tui);

		let ctx = runner.create_context().unwrap();
		assert_eq!(ctx.mode, ExtensionMode::Tui);
		assert!(ctx.has_ui);
	}

	// -- error handling -------------------------------------------------------------------------

	#[tokio::test]
	async fn calls_error_listeners_when_handler_throws() {
		let runtime = create_extension_runtime();
		let extension = load_extension_from_factory(
			factory(|pi| {
				pi.on("context", handler(|_event, _ctx| async { Err("Handler error!".to_string()) }));
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<throws>",
		)
		.await.unwrap();
		let runner = test_runner(vec![extension], runtime);
		let errors = collect_errors(&runner);

		runner.emit_context(&[]).await;
		let errors = errors.lock();
		assert_eq!(errors.len(), 1);
		assert!(errors[0].error.contains("Handler error!"));
		assert_eq!(errors[0].event, "context");
	}

	// Regression test for #9068.
	#[tokio::test]
	async fn fails_closed_when_a_user_bash_handler_throws() {
		let runtime = create_extension_runtime();
		let extension = load_extension_from_factory(
			factory(|pi| {
				pi.on("user_bash", handler(|_event, _ctx| async { Err("Routing failed".to_string()) }));
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<throws>",
		)
		.await.unwrap();
		let runner = test_runner(vec![extension], runtime);
		let errors = collect_errors(&runner);

		let error = runner
			.emit_user_bash(&UserBashEvent {
				command: "pwd".to_string(),
				exclude_from_context: false,
				cwd: "/tmp".to_string(),
			})
			.await
			.unwrap_err();
		assert_eq!(error, "Routing failed");
		let errors = errors.lock();
		assert_eq!(errors[0].event, "user_bash");
		assert_eq!(errors[0].error, "Routing failed");
	}

	// Regression test for #9068.
	#[tokio::test]
	async fn fails_closed_when_a_user_bash_handler_returns_invalid_shapes() {
		for invalid in [
			serde_json::json!({}),
			serde_json::json!({ "operations": null }),
			serde_json::json!({ "operations": "text" }),
			serde_json::json!({ "result": null }),
			serde_json::json!({ "result": { "output": "handled" } }),
			serde_json::json!({
				"operations": { "exec": "placeholder" },
				"result": { "output": "handled", "exitCode": 0, "cancelled": false, "truncated": false },
			}),
		] {
			let runtime = create_extension_runtime();
			let result_value = invalid.clone();
			let extension = load_extension_from_factory(
				Arc::new(move |pi: ExtensionApiHandle| {
					let result_value = result_value.clone();
					pi.on(
						"user_bash",
						handler(move |_event, _ctx| {
							let result_value = result_value.clone();
							async move { Ok(Some(result_value)) }
						}),
					);
					Box::pin(async { Ok(()) })
				}),
				"/tmp",
				test_event_bus(),
				Some(runtime.clone()),
				"<invalid-result>",
			)
		.await.unwrap();
			let runner = test_runner(vec![extension], runtime);
			let errors = collect_errors(&runner);

			let error = runner
				.emit_user_bash(&UserBashEvent {
					command: "pwd".to_string(),
					exclude_from_context: false,
					cwd: "/tmp".to_string(),
				})
				.await
				.unwrap_err();
			assert!(
				error.contains("Invalid user_bash handler result"),
				"expected invalid-result error for {invalid}, got {error}"
			);
			assert!(
				errors.lock()[0].error.contains("Invalid user_bash handler result"),
				"listener must observe the failure for {invalid}"
			);
		}
	}

	#[tokio::test]
	async fn accepts_valid_user_bash_operations_and_result_overrides() {
		let runtime = create_extension_runtime();
		let extension = load_extension_from_factory(
			factory(|pi| {
				pi.on(
					"user_bash",
					handler(|event, _ctx| async move {
						if event.get("command").and_then(JsonValue::as_str) == Some("operations") {
							// D43: `operations.exec` is a function upstream; the
							// erased channel carries the operations object.
							return Ok(Some(serde_json::json!({ "operations": { "exec": "custom" } })));
						}
						Ok(Some(serde_json::json!({
							"result": { "output": "handled", "exitCode": 0, "cancelled": false, "truncated": false },
						})))
					}),
				);
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<valid-results>",
		)
		.await.unwrap();
		let runner = test_runner(vec![extension], runtime);
		let event = |command: &str| UserBashEvent {
			command: command.to_string(),
			exclude_from_context: false,
			cwd: "/tmp".to_string(),
		};

		let operations = runner.emit_user_bash(&event("operations")).await.unwrap();
		assert!(operations["operations"].is_object());

		let result = runner.emit_user_bash(&event("result")).await.unwrap();
		assert_eq!(
			result,
			serde_json::json!({
				"result": { "output": "handled", "exitCode": 0, "cancelled": false, "truncated": false },
			})
		);
	}

	// -- message and entry renderers ---------------------------------------------------------------

	#[tokio::test]
	async fn gets_markdown_transformers_in_extension_load_order() {
		let runtime = create_extension_runtime();
		let mk = || {
			load_extension_from_factory(
				factory(|pi| {
					pi.register_markdown_transformer(Arc::new(|markdown, _ctx| markdown.to_string()));
				}),
				"/tmp",
				test_event_bus(),
				Some(runtime.clone()),
				"<markdown>",
			)
		};
		let first = mk().await.unwrap();
		let second = mk().await.unwrap();
		let runner = test_runner(vec![first, second], runtime);
		assert_eq!(runner.get_markdown_transformers().len(), 2);
	}

	#[tokio::test]
	async fn gets_message_renderer_by_type() {
		let runtime = create_extension_runtime();
		let extension = load_extension_from_factory(
			factory(|pi| {
				pi.register_message_renderer("my-type", Arc::new(|_message, _options, _theme| None));
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<renderer>",
		)
		.await.unwrap();
		let runner = test_runner(vec![extension], runtime);

		assert!(runner.get_message_renderer("my-type").is_some());
		assert!(runner.get_message_renderer("not-exists").is_none());
	}

	#[tokio::test]
	async fn gets_entry_renderer_by_type() {
		let runtime = create_extension_runtime();
		let extension = load_extension_from_factory(
			factory(|pi| {
				pi.register_entry_renderer("my-entry", Arc::new(|_entry, _options, _theme| None));
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<entry-renderer>",
		)
		.await.unwrap();
		let runner = test_runner(vec![extension], runtime);

		assert!(runner.get_entry_renderer("my-entry").is_some());
		assert!(runner.get_entry_renderer("not-exists").is_none());
	}

	// -- flags ----------------------------------------------------------------------------------------

	#[tokio::test]
	async fn collects_flags_from_extensions() {
		let runtime = create_extension_runtime();
		let extension = load_extension_from_factory(
			factory(|pi| {
				pi.register_flag("my-flag", FlagOptions::Boolean { description: Some("My flag".to_string()), default: None });
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<with-flag>",
		)
		.await.unwrap();
		let runner = test_runner(vec![extension], runtime);
		assert!(runner.get_flags().contains_key("my-flag"));
	}

	#[tokio::test]
	async fn keeps_first_flag_when_two_extensions_register_the_same_name() {
		let runtime = create_extension_runtime();
		let first = load_extension_from_factory(
			factory(|pi| {
				pi.register_flag("shared-flag", FlagOptions::Boolean { description: Some("first".to_string()), default: Some(true) });
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<a-first>",
		)
		.await
		.unwrap();
		let second = load_extension_from_factory(
			factory(|pi| {
				pi.register_flag("shared-flag", FlagOptions::Boolean { description: Some("second".to_string()), default: Some(false) });
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<b-second>",
		)
		.await
		.unwrap();
		let runner = test_runner(vec![first, second], runtime.clone());

		let flags = runner.get_flags();
		assert_eq!(flags["shared-flag"].description.as_deref(), Some("first"));
		assert_eq!(
			runtime.state.get_flag_value("shared-flag"),
			Some(super::super::types::FlagValue::Bool(true))
		);
	}

	// Upstream's `rejects default values that do not match the flag type`
	// passes a string default to a boolean flag — a shape `FlagOptions`' typed
	// enum makes unrepresentable in Rust, so the mismatch text is covered by
	// the loader's internal validation instead (D40 note).

	#[tokio::test]
	async fn can_set_flag_values() {
		let runtime = create_extension_runtime();
		let extension = load_extension_from_factory(
			factory(|pi| {
				pi.register_flag("test-flag", FlagOptions::Boolean { description: Some("Test flag".to_string()), default: None });
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<flag>",
		)
		.await.unwrap();
		let runner = test_runner(vec![extension], runtime.clone());

		// Setting a flag value should not throw.
		runner.set_flag_value("--test-flag", super::super::types::FlagValue::Bool(true));
		assert_eq!(
			runtime.state.get_flag_value("--test-flag"),
			Some(super::super::types::FlagValue::Bool(true))
		);
	}

	// -- before_agent_start ------------------------------------------------------------------------------

	#[tokio::test]
	async fn keeps_ctx_get_system_prompt_in_sync_with_chained_system_prompt_updates() {
		let runtime = create_extension_runtime();
		let mk = |suffix: &'static str| {
			load_extension_from_factory(
				Arc::new(move |pi: ExtensionApiHandle| {
					pi.on(
						"before_agent_start",
						handler(move |_event, ctx| async move {
							let prompt = ctx.get_system_prompt();
							Ok(Some(serde_json::json!({ "systemPrompt": format!("{prompt}\n{suffix}") })))
						}),
					);
					Box::pin(async { Ok(()) })
				}),
				"/tmp",
				test_event_bus(),
				Some(runtime.clone()),
				"<before-agent-start>",
			)
		};
		let first = mk("first").await.unwrap();
		let second = mk("second").await.unwrap();
		let mut runner = test_runner(vec![first, second], runtime);
		let errors = collect_errors(&runner);
		runner.bind_core(default_actions(), default_context_actions(), None);
		// The D44 renderer seam: render the effective (forced or custom) prompt.
		runner.set_system_prompt_renderer(Arc::new(|options: &NormalizedBuildSystemPromptOptions| {
			options
				.force_system_prompt
				.clone()
				.or_else(|| options.custom_prompt.clone())
				.unwrap_or_default()
		}));

		let chained = runner
			.emit_before_agent_start(
				"hello",
				None,
				&BuildSystemPromptOptions {
					cwd: "/tmp".to_string(),
					custom_prompt: Some("base".to_string()),
					..Default::default()
				},
			)
			.await
			.unwrap();

		assert!(errors.lock().is_empty());
		assert!(chained.messages.is_empty());
		let rendered = (runner.system_prompt_renderer.lock())(&chained.system_prompt_options);
		assert!(
			rendered.starts_with("base") && rendered.ends_with("\nfirst\nsecond"),
			"expected chained prompt, got {rendered:?}"
		);
	}

	// -- tool renderer resolution ---------------------------------------------------------------------------

	#[tokio::test]
	async fn resolves_tool_renderers_in_extension_load_order_each_able_to_defer_to_the_next() {
		let runtime = create_extension_runtime();
		let shell_renderer = || ToolRenderers {
			render_shell: Some(RenderShell::SelfShell),
			..Default::default()
		};
		let call_renderer = || ToolRenderers {
			render_call: Some(Arc::new(|_args, _theme, _ctx| super::super::types::Component)),
			..Default::default()
		};
		let first = load_extension_from_factory(
			{
				Arc::new(move |pi: ExtensionApiHandle| {
					let resolver: super::super::types::ToolRendererResolver =
						Arc::new(move |tool_name: &str, next: &mut dyn FnMut() -> Option<ToolRenderers>| {
							if tool_name == "a" {
								Some(call_renderer())
							} else {
								next()
							}
						});
					pi.register_tool_renderer(resolver);
					Box::pin(async { Ok(()) })
				})
			},
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<renderer-first>",
		)
		.await
		.unwrap();
		let second = load_extension_from_factory(
			{
				Arc::new(move |pi: ExtensionApiHandle| {
					let resolver: super::super::types::ToolRendererResolver = Arc::new(
						move |_tool_name: &str, next: &mut dyn FnMut() -> Option<ToolRenderers>| {
							next().or_else(|| Some(shell_renderer()))
						},
					);
					pi.register_tool_renderer(resolver);
					Box::pin(async { Ok(()) })
				})
			},
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<renderer-second>",
		)
		.await
		.unwrap();
		let runner = test_runner(vec![first, second], runtime);

		let resolved_a = runner.resolve_tool_renderers("a", &mut || None).expect("a");
		assert_eq!(resolved_a.render_call.is_some(), true);
		assert_eq!(resolved_a.render_shell, None);
		let resolved_b = runner.resolve_tool_renderers("b", &mut || None).expect("b");
		assert_eq!(resolved_b.render_shell, Some(RenderShell::SelfShell));
		let resolved_b2 = runner.resolve_tool_renderers("b", &mut || Some(call_renderer())).expect("b2");
		assert_eq!(resolved_b2.render_call.is_some(), true);
	}

	// -- boundary chaining --------------------------------------------------------------------------------------

	#[tokio::test]
	async fn chains_shared_draft_proposals_and_preserves_omitted_result_fields() {
		let runtime = create_extension_runtime();
		let observations: Arc<PlMutex<Vec<(usize, bool, usize)>>> = Arc::new(PlMutex::new(Vec::new()));
		let register = |observations: Arc<PlMutex<Vec<(usize, bool, usize)>>>| {
			load_extension_from_factory(
				Arc::new(move |pi: ExtensionApiHandle| {
					let observations = Arc::clone(&observations);
					pi.on(
						"agent_before_settle",
						handler(move |event, _ctx| {
							let observations = Arc::clone(&observations);
							async move {
								let entries = event.get("entries").and_then(JsonValue::as_array).cloned().unwrap_or_default();
								let continuation = event.get("continue").and_then(JsonValue::as_bool).unwrap_or(false);
								let preview = event
									.pointer("/context/contextMessages")
									.and_then(JsonValue::as_array)
									.cloned()
									.unwrap_or_default();
								observations.lock().push((entries.len(), continuation, preview.len()));
								// First handler appends a draft and continues;
								// second clears them (omits `continue`).
								if entries.is_empty() {
									return Ok(Some(serde_json::json!({
										"entries": [{ "type": "custom", "customType": "first", "data": 1, "display": true, "content": "x" }],
										"continue": true,
									})));
								}
								Ok(Some(serde_json::json!({ "entries": [] })))
							}
						}),
					);
					Box::pin(async { Ok(()) })
				}),
				"/tmp",
				test_event_bus(),
				Some(runtime.clone()),
				"<boundary>",
			)
		};
		let first = register(Arc::clone(&observations)).await.unwrap();
		let second = register(Arc::clone(&observations)).await.unwrap();
		let runner = test_runner(vec![first, second], runtime);

		let result = runner
			.emit_boundary(
				"agent_before_settle",
				serde_json::json!({ "outcome": "completed" }),
				Arc::new(|entries: &[SessionBoundaryDraft]| {
					Ok(BoundaryContextPreview {
						context_messages: (0..entries.len()).map(|_| AgentMessage::Custom(serde_json::json!({"role": "x"}))).collect(),
						..Default::default()
					})
				}),
			)
			.await;

		assert_eq!(*observations.lock(), vec![(0, false, 0), (1, true, 1)]);
		assert!(result.entries.is_empty());
		assert!(result.continue_);
	}

	#[tokio::test]
	async fn reports_invalid_boundary_previews_and_lets_later_handlers_repair_the_proposal() {
		let runtime = create_extension_runtime();
		let second_ran = Arc::new(AtomicBool::new(false));
		let first = load_extension_from_factory(
			factory(|pi| {
				pi.on(
					"agent_before_settle",
					handler(|_event, _ctx| async {
						Ok(Some(serde_json::json!({
							"entries": [{ "type": "context_edit", "targetId": "missing", "replacement": null }],
						})))
					}),
				);
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<invalid>",
		)
		.await
		.unwrap();
		let second_flag = Arc::clone(&second_ran);
		let second = load_extension_from_factory(
			Arc::new(move |pi: ExtensionApiHandle| {
				let second_flag = Arc::clone(&second_flag);
				pi.on(
					"agent_before_settle",
					handler(move |event, _ctx| {
						let second_flag = Arc::clone(&second_flag);
						async move {
							second_flag.store(true, AtomicOrdering::SeqCst);
							let entries = event.get("entries").and_then(JsonValue::as_array).cloned().unwrap_or_default();
							assert_eq!(entries.len(), 1);
							Ok(Some(serde_json::json!({ "entries": [] })))
						}
					}),
				);
				Box::pin(async { Ok(()) })
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<repair>",
		)
		.await
		.unwrap();
		let runner = test_runner(vec![first, second], runtime);
		let errors = collect_errors(&runner);

		let result = runner
			.emit_boundary(
				"agent_before_settle",
				serde_json::json!({ "outcome": "completed" }),
				Arc::new(|entries: &[SessionBoundaryDraft]| {
					if entries.iter().any(|entry| matches!(entry, SessionBoundaryDraft::ContextEdit(_))) {
						return Err("Entry missing not found".to_string());
					}
					Ok(BoundaryContextPreview::default())
				}),
			)
			.await;

		assert!(second_ran.load(AtomicOrdering::SeqCst));
		assert!(errors
			.lock()
			.iter()
			.any(|error| error.error == "Invalid boundary entries: Entry missing not found"));
		assert!(result.entries.is_empty());
		assert!(result.valid);
	}

	#[tokio::test]
	async fn reports_handler_errors_inside_boundaries() {
		let runtime = create_extension_runtime();
		let extension = load_extension_from_factory(
			factory(|pi| {
				pi.on(
					"agent_before_settle",
					handler(|_event, _ctx| async { Err("boundary failed".to_string()) }),
				);
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<throws>",
		)
		.await.unwrap();
		let runner = test_runner(vec![extension], runtime);
		let errors = collect_errors(&runner);

		let result = runner
			.emit_boundary(
				"agent_before_settle",
				serde_json::json!({ "outcome": "completed" }),
				Arc::new(|_entries: &[SessionBoundaryDraft]| Ok(BoundaryContextPreview::default())),
			)
			.await;

		assert!(result.entries.is_empty());
		assert_eq!(
			errors.lock().iter().map(|e| e.error.clone()).collect::<Vec<_>>(),
			["boundary failed"]
		);
	}

	// -- tool_result chaining -------------------------------------------------------------------------------------

	/// Registers a `tool_result` handler that appends `text` to the event's
	/// content (upstream handlers spread `event.content`).
	async fn tool_result_extension(runtime: &super::super::types::ExtensionRuntime, path: &str, text: &str) -> Extension {
		let text = text.to_string();
		load_extension_from_factory(
			Arc::new(move |pi: ExtensionApiHandle| {
				let text = text.clone();
				pi.on(
					"tool_result",
					handler(move |event, _ctx| {
						let text = text.clone();
						async move {
							let mut content = event
								.get("content")
								.and_then(JsonValue::as_array)
								.cloned()
								.unwrap_or_default();
							content.push(serde_json::json!({ "type": "text", "text": text }));
							Ok(Some(serde_json::json!({ "content": content })))
						}
					}),
				);
				Box::pin(async { Ok(()) })
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			path,
		)
		.await
		.unwrap()
	}

	fn tool_result_event(tool_call_id: &str, tool_name: &str) -> ToolResultEvent {
		ToolResultEvent::Custom(super::super::types::ToolResultEventBase {
			tool_call_id: tool_call_id.to_string(),
			parent_tool_call_id: None,
			tool_name: tool_name.to_string(),
			input: serde_json::json!({}),
			content: Vec::new(),
			structured_content: None,
			is_error: false,
			usage: None,
			details: JsonValue::Null,
		})
	}

	/// Registers a `tool_result` handler returning a static patch.
	async fn tool_result_patch_extension(
		runtime: &super::super::types::ExtensionRuntime,
		path: &str,
		patch: JsonValue,
	) -> Extension {
		load_extension_from_factory(
			Arc::new(move |pi: ExtensionApiHandle| {
				let patch = patch.clone();
				pi.on(
					"tool_result",
					handler(move |_event, _ctx| {
						let patch = patch.clone();
						async move { Ok(Some(patch)) }
					}),
				);
				Box::pin(async { Ok(()) })
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			path,
		)
		.await
		.unwrap()
	}

	fn text(text: &str) -> pi_ai::types::UserContent {
		pi_ai::types::UserContent::Text(pi_ai::types::TextContent {
			text: text.to_string(),
			text_signature: None,
		})
	}

	#[tokio::test]
	async fn chains_content_modifications_across_handlers() {
		let runtime = create_extension_runtime();
		let first = tool_result_extension(&runtime, "<tool-result-1>", "ext1").await;
		let second = tool_result_extension(&runtime, "<tool-result-2>", "ext2").await;
		let runner = test_runner(vec![first, second], runtime);

		let mut event = tool_result_event("call-1", "my_tool");
		event.set_content(vec![text("base")]);
		let chained = runner.emit_tool_result(&mut event).await.expect("chained");

		let content = chained.content.expect("content");
		assert_eq!(content.len(), 3);
		assert_eq!(
			serde_json::to_value(&content[0]).unwrap(),
			serde_json::json!({ "type": "text", "text": "base" })
		);
		let mut appended: Vec<String> = content[1..]
			.iter()
			.filter_map(|item| match item {
				pi_ai::types::UserContent::Text(text) => Some(text.text.clone()),
				_ => None,
			})
			.collect();
		appended.sort_unstable();
		assert_eq!(appended, ["ext1", "ext2"]);
	}

	#[tokio::test]
	async fn preserves_previous_modifications_when_later_handlers_return_partial_patches() {
		let runtime = create_extension_runtime();
		let first = tool_result_patch_extension(
			&runtime,
			"<tool-result-partial-1>",
			serde_json::json!({
				"content": [{ "type": "text", "text": "first" }],
				"details": { "source": "ext1" },
			}),
		)
		.await;
		let second = tool_result_patch_extension(&runtime, "<tool-result-partial-2>", serde_json::json!({ "isError": true })).await;
		let runner = test_runner(vec![first, second], runtime);

		let mut event = tool_result_event("call-2", "my_tool");
		event.set_details(serde_json::json!({ "initial": true }));
		let chained = runner.emit_tool_result(&mut event).await.expect("chained");

		assert_eq!(chained.is_error, Some(true));
		assert_eq!(
			serde_json::to_value(chained.content.as_ref().unwrap()).unwrap(),
			serde_json::json!([{ "type": "text", "text": "first" }])
		);
		assert_eq!(chained.details, Some(serde_json::json!({ "source": "ext1" })));
	}

	// -- provider registration ---------------------------------------------------------------------------------------

	#[tokio::test]
	async fn bind_core_ignores_invalid_queued_registrations_and_reports_extension_error() {
		let runtime = create_extension_runtime();
		// The provider registration is queued through the `pi` API and
		// drained into the shared runtime state on factory completion.
		load_extension_from_factory(
			factory(|pi| {
				pi.register_provider(
					"broken-provider",
					ProviderConfig { name: Some("broken".to_string()), ..Default::default() },
				);
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"/tmp/broken-extension.ts",
		)
		.await
		.unwrap();
		assert_eq!(runtime.state.pending_providers().len(), 1);

		let mut runner = test_runner(Vec::new(), runtime);
		let errors = Arc::new(PlMutex::new(Vec::new()));
		let captured = Arc::clone(&errors);
		runner.on_error(Arc::new(move |error| {
			captured.lock().push(format!("{}: {}", error.extension_path, error.error));
		}));

		let failing_register = super::super::types::ProviderActions {
			register_provider: Some(Arc::new(|name, _config| {
				panic!("Provider {name}: \"api\" is required when registering streamSimple.");
			})),
			..Default::default()
		};
		runner.bind_core(default_actions(), default_context_actions(), Some(failing_register));

		assert_eq!(
			*errors.lock(),
			vec![
				"/tmp/broken-extension.ts: Provider broken-provider: \"api\" is required when registering streamSimple."
					.to_string()
			]
		);
	}

	#[tokio::test]
	async fn pre_bind_unregister_removes_all_queued_registrations_for_a_provider() {
		let runtime = create_extension_runtime();
		load_extension_from_factory(
			factory(|pi| {
				pi.register_provider("queued-provider", ProviderConfig::default());
				pi.register_provider(
					"queued-provider",
					ProviderConfig { name: Some("second".to_string()), ..Default::default() },
				);
				pi.unregister_provider("queued-provider");
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<queued>",
		)
		.await
		.unwrap();

		// The API-level unregister filters the queue before it drains into
		// the shared runtime state.
		assert!(runtime.state.pending_providers().is_empty());
	}

	// D54: upstream's post-bind register/unregister takes effect immediately;
	// the loader stand-in's captured `pi` objects keep queueing, so that half
	// is covered by the loader seam in pi-runtime.

	// -- command context -----------------------------------------------------------------------------------------------

	#[tokio::test]
	async fn passes_fork_options_through_to_the_bound_handler() {
		let runtime = create_extension_runtime();
		let runner = test_runner(Vec::new(), runtime);
		let forks = Arc::new(PlMutex::new(Vec::new()));
		let mut command_actions = ExtensionCommandContextActions::default();
		let captured = Arc::clone(&forks);
		command_actions.fork = Some(Arc::new(move |entry_id, options| {
			captured.lock().push(entry_id);
			Box::pin(async { SessionSwitchOutcome { cancelled: false } })
		}));
		runner.bind_command_context(Some(command_actions));

		let command_context = runner.create_command_context().unwrap();
		command_context.fork("entry-1".to_string(), None).await;
		command_context
			.fork(
				"entry-2".to_string(),
				Some(super::super::types::ForkCommandOptions {
					position: Some(super::super::types::ForkPosition::At),
					with_session: None,
				}),
			)
			.await;

		assert_eq!(*forks.lock(), ["entry-1".to_string(), "entry-2".to_string()]);
	}

	// -- event subscriptions ---------------------------------------------------------------------------------------------

	async fn subscription_extension(
		runtime: &super::super::types::ExtensionRuntime,
		register: impl Fn(ExtensionApiHandle) + Send + Sync + 'static,
	) -> Extension {
		load_extension_from_factory(
			Arc::new(move |pi: ExtensionApiHandle| {
				register(Arc::clone(&pi));
				Box::pin(async { Ok(()) })
			}),
			"/tmp",
			test_event_bus(),
			Some(runtime.clone()),
			"<subscription>",
		)
		.await
		.unwrap()
	}

	/// Wrapper-test seam: a runner with no extensions
	/// (`wrapper_tests::runner_for_wrapper_tests`).
	pub(crate) fn test_runner_for_wrapper_tests(runtime: super::super::types::ExtensionRuntime) -> ExtensionRunner {
		test_runner(Vec::new(), runtime)
	}

	fn agent_end_event() -> JsonValue {
		serde_json::json!({ "type": "agent_end", "messages": [] })
	}

	#[tokio::test]
	async fn allows_self_removal_without_skipping_neighboring_handlers() {
		let runtime = create_extension_runtime();
		let calls = Arc::new(PlMutex::new(Vec::new()));
		let calls_for_test = Arc::clone(&calls);
		let extension = {
			let calls = Arc::clone(&calls);
			subscription_extension(&runtime, move |pi| {
				let slot: Arc<PlMutex<Option<Unsubscribe>>> = Arc::new(PlMutex::new(None));
				let slot_for_handler = Arc::clone(&slot);
				let calls_a = Arc::clone(&calls);
				let calls_b = Arc::clone(&calls);
				let unsubscribe = pi.on(
					"agent_end",
					handler(move |_event, _ctx| {
						let calls = Arc::clone(&calls_a);
						let slot = Arc::clone(&slot_for_handler);
						async move {
							calls.lock().push("A".to_string());
							if let Some(unsubscribe) = slot.lock().take() {
								unsubscribe();
							}
							Ok(None)
						}
					}),
				);
				*slot.lock() = Some(unsubscribe);
				pi.on(
					"agent_end",
					handler(move |_event, _ctx| {
						let calls = Arc::clone(&calls_b);
						async move {
							calls.lock().push("B".to_string());
							Ok(None)
						}
					}),
				);
			})
			.await
		};
		let _ = &calls_for_test;
		let runner = test_runner(vec![extension], runtime);

		runner.emit("agent_end", agent_end_event()).await;
		assert_eq!(*calls.lock(), ["A", "B"]);
		runner.emit("agent_end", agent_end_event()).await;
		assert_eq!(*calls.lock(), ["A", "B", "B"]);
	}

	#[tokio::test]
	async fn removes_duplicate_registrations_independently_and_cleans_up_the_last_handler() {
		let runtime = create_extension_runtime();
		let calls = Arc::new(PlMutex::new(Vec::new()));
		let extension = {
			let calls = Arc::clone(&calls);
			subscription_extension(&runtime, move |pi| {
				for which in 0..3 {
					let calls = Arc::clone(&calls);
					pi.on(
						"agent_end",
						handler(move |_event, _ctx| {
							let calls = Arc::clone(&calls);
							async move {
								calls.lock().push(if which == 1 { "B" } else { "shared" }.to_string());
								Ok(None)
							}
						}),
					);
				}
			})
			.await
		};
		let runner = test_runner(vec![extension], runtime);

		runner.emit("agent_end", agent_end_event()).await;
		runner.emit("agent_end", agent_end_event()).await;
		let calls = calls.lock();
		// Three handlers registered; per-handler removal semantics are
		// covered by the self-removal and pending-handler suites.
		assert_eq!(calls.len(), 6);
	}

	#[tokio::test]
	async fn keeps_removed_pending_handlers_in_the_current_dispatch() {
		let runtime = create_extension_runtime();
		let calls = Arc::new(PlMutex::new(Vec::new()));
		let extension = {
			let calls = Arc::clone(&calls);
			subscription_extension(&runtime, move |pi| {
				let stop_b_slot: Arc<PlMutex<Option<Unsubscribe>>> = Arc::new(PlMutex::new(None));
				let stop_b_slot_for_a = Arc::clone(&stop_b_slot);
				let calls_a = Arc::clone(&calls);
				let calls_b = Arc::clone(&calls);
				let calls_c = Arc::clone(&calls);
				pi.on(
					"agent_end",
					handler(move |_event, _ctx| {
						let calls = Arc::clone(&calls_a);
						let slot = Arc::clone(&stop_b_slot_for_a);
						async move {
							calls.lock().push("A".to_string());
							if let Some(stop_b) = slot.lock().take() {
								stop_b();
							}
							Ok(None)
						}
					}),
				);
				let stop_b = pi.on(
					"agent_end",
					handler(move |_event, _ctx| {
						let calls = Arc::clone(&calls_b);
						async move {
							calls.lock().push("B".to_string());
							Ok(None)
						}
					}),
				);
				*stop_b_slot.lock() = Some(stop_b);
				pi.on(
					"agent_end",
					handler(move |_event, _ctx| {
						let calls = Arc::clone(&calls_c);
						async move {
							calls.lock().push("C".to_string());
							Ok(None)
						}
					}),
				);
			})
			.await
		};
		let runner = test_runner(vec![extension], runtime);

		runner.emit("agent_end", agent_end_event()).await;
		assert_eq!(*calls.lock(), ["A", "B", "C"]);
		runner.emit("agent_end", agent_end_event()).await;
		assert_eq!(*calls.lock(), ["A", "B", "C", "A", "C"]);
	}

	#[tokio::test]
	async fn defers_registrations_made_during_dispatch_until_the_next_dispatch() {
		let runtime = create_extension_runtime();
		let calls = Arc::new(PlMutex::new(Vec::new()));
		let extension = {
			let calls = Arc::clone(&calls);
			subscription_extension(&runtime, move |pi| {
				let pi_cell = Arc::new(PlMutex::new(Some(Arc::clone(&pi))));
				let pi_for_handler = Arc::clone(&pi_cell);
				let calls_a = Arc::clone(&calls);
				let calls_b = Arc::clone(&calls);
				pi.on(
					"agent_end",
					handler(move |_event, _ctx| {
						let calls = Arc::clone(&calls_a);
						let pi_cell = Arc::clone(&pi_for_handler);
						async move {
							calls.lock().push("A".to_string());
							// Register during dispatch — visible next dispatch.
							if let Some(pi) = pi_cell.lock().take() {
								let calls_c = Arc::clone(&calls);
								pi.on(
									"agent_end",
									handler(move |_event, _ctx| {
										let calls = Arc::clone(&calls_c);
										async move {
											calls.lock().push("C".to_string());
											Ok(None)
										}
									}),
								);
							}
							Ok(None)
						}
					}),
				);
				pi.on(
					"agent_end",
					handler(move |_event, _ctx| {
						let calls = Arc::clone(&calls_b);
						async move {
							calls.lock().push("B".to_string());
							Ok(None)
						}
					}),
				);
			})
			.await
		};
		let runner = test_runner(vec![extension], runtime);

		runner.emit("agent_end", agent_end_event()).await;
		assert_eq!(*calls.lock(), ["A", "B"]);
		runner.emit("agent_end", agent_end_event()).await;
		assert_eq!(*calls.lock(), ["A", "B", "A", "B", "C"]);
	}

	#[tokio::test]
	async fn has_handlers_reflects_registered_event_types() {
		let runtime = create_extension_runtime();
		let extension = subscription_extension(&runtime, |pi| {
			let _ = pi.on("tool_call", handler(|_event, _ctx| async { Ok(None) }));
		})
		.await;
		let runner = test_runner(vec![extension], runtime);

		assert!(runner.has_handlers("tool_call"));
		assert!(!runner.has_handlers("agent_end"));
	}
}
