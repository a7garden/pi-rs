//! Port of `packages/coding-agent/src/core/extensions/types.ts` — the
//! extension system type foundation: UI/command/tool contexts, tool
//! definitions, every extension event and event result, the `pi.*` API
//! surface, provider-registration types, and the loader/runtime shapes.
//!
//! # Representation overview (crate docs D27–D31)
//!
//! - **Data types are serde structs/enums** with camelCase renames and
//!   absent-vs-`null` mapped to `Option` + `skip_serializing_if`, matching
//!   the wire shapes the JS extension bridge (pi-runtime) exchanges.
//! - **Callbacks are erased `Arc<dyn Fn>` fields/aliases** with
//!   `BoxFuture<'static, _>` for async (crate conventions). Handlers take the
//!   context **by value**: every context is a bag of `Arc` handles, so the
//!   clone is cheap and futures stay `'static`.
//! - **Context interfaces became vtable structs**: [`ExtensionContext`] is a
//!   struct of data fields plus `Arc<dyn Fn>` action fields with forwarding
//!   methods; "extends" contexts embed their base and `Deref` to it (D28).
//!   The `ExtensionUIContext` interface became the object-safe [`ExtensionUi`]
//!   trait; each mode implements it (upstream: interactive/RPC/print each
//!   provide one).
//! - **`ToolDefinition<TParams, TDetails, TState>` is erased** to a struct
//!   whose params/details/state are `serde_json::Value`, exactly like
//!   pi-agent's `AgentTool` (D27). Params are JSON Schema documents
//!   (`pi-ai` D6 precedent). A follow-up slice ports `core/tools/*` against
//!   this shape.
//!
//! Types whose upstream home module is not ported yet are defined in the
//! *Dependency stand-ins* section and replaced by re-exports when those
//! modules land (D29).
//!
//! Tests: upstream has no `types.test.ts`; the runtime functions (type
//! guards, `isToolCallEventType`, `defineTool`) and the hand-rolled event
//! serialization are covered by the `types_tests` module at the bottom of
//! this file.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use pi_agent::{
	AgentMessage, AgentThinkingLevel, AgentTool, AgentToolCallOutcome, AgentToolResult, AgentToolUpdateCallback,
	BoxFuture, ToolExecutionMode,
};
use pi_ai::auth::types::OAuthCredentials;
use pi_ai::types::{
	AssistantMessage, AssistantMessageEvent, ChatType, ClassifierTypeTag, ConstrainedSamplingSetting, ImageContent,
	ImageTypeTag, Message, Model, ModelCompat, ModelCost, ModelInput, ModelInputLimits, ModelOutput, ModelPromptCache,
	ModelThinkingLevel, ProviderHeaders, SamplingParams, SimpleStreamOptions, ThinkingLevelMap, ToolResultMessage,
	Usage, UserContentField,
};
use pi_ai::{AssistantMessageEventStream, Provider as AiProvider, ProviderClassifier, ProviderImages, RefreshModelsContext, TranscriptContext};

use crate::core::compaction::{CompactionPreparation, CompactionResult};
use crate::core::messages::CustomMessage;
use crate::core::session_manager::{
	BranchSummaryEntry, CompactionEntry, ContextEditReplacement, CustomEntry, ProjectedSessionEntry, SessionEntry,
	SessionManager,
};
use crate::core::settings_manager::Settings;
use crate::core::skills::Skill;
use crate::core::source_info::SourceInfo;
use crate::utils::abort::AbortSignal;

// ============================================================================
// Dependency stand-ins (crate docs D29)
//
// Upstream types.ts imports these from core modules that are not ported yet.
// Each stand-in keeps the upstream name and wire shape; the module that ports
// the home file replaces it with a `pub use` re-export. Opaque unit structs
// mark runtime handles whose surface belongs to the home module.
// ============================================================================

/// Upstream `core/slash-commands.ts` `SlashCommandSource` (module ported
/// separately).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlashCommandSource {
	#[serde(rename = "extension")]
	Extension,
	#[serde(rename = "prompt")]
	Prompt,
	#[serde(rename = "skill")]
	Skill,
}

/// Upstream `core/slash-commands.ts` `SlashCommandInfo` (module ported
/// separately).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SlashCommandInfo {
	pub name: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub description: Option<String>,
	pub source: SlashCommandSource,
	pub source_info: SourceInfo,
}

/// Upstream `core/exec.ts` `ExecOptions` (module ported separately).
#[derive(Debug, Clone, Default)]
pub struct ExecOptions {
	/// Signal to cancel the command.
	pub signal: Option<AbortSignal>,
	/// Timeout in milliseconds.
	pub timeout: Option<u64>,
	/// Working directory.
	pub cwd: Option<String>,
}

/// Upstream `core/exec.ts` `ExecResult` (module ported separately).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecResult {
	pub stdout: String,
	pub stderr: String,
	pub code: i32,
	pub killed: bool,
}

/// Upstream `core/model-resolver.ts` `ScopedModel` — the real port landed
/// ([`crate::core::model_resolver`]), replacing the earlier stand-in (D29).
pub use crate::core::model_resolver::ScopedModel;

/// Upstream `core/bash-executor.ts` `BashResult` — the real port landed
/// ([`crate::core::bash_executor`]), replacing the earlier stand-in (D29).
pub use crate::core::bash_executor::BashResult;

/// Upstream `core/cache-warmer.ts` `CacheWarmingAction` — the real port
/// landed ([`crate::core::cache_warmer`]), replacing the earlier stand-in
/// (D29, D67).
pub use crate::core::cache_warmer::{CacheWarmingAction, CacheWarmingDecisionEvent, CacheWarmingDecisionEventResult};

/// Upstream `core/system-prompt.ts` `BuildSystemPromptOptions` (module ported
/// separately). Plain struct: the `skills` member is [`Skill`], which carries
/// no serde yet, so serialization is bridge work (D30).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BuildSystemPromptOptions {
	/// Custom system prompt (replaces the default prefix).
	pub custom_prompt: Option<String>,
	/// Exact full prompt replacement set by a `before_agent_start` handler.
	pub force_system_prompt: Option<String>,
	/// Tools to include in the prompt. Default: `[read, bash, edit, write]`.
	pub selected_tools: Option<Vec<String>>,
	/// Optional one-line tool snippets keyed by tool name.
	pub tool_snippets: Option<BTreeMap<String, String>>,
	/// Guideline bullets contributed by each tool, keyed by tool name.
	pub tool_guidelines: Option<BTreeMap<String, Vec<String>>>,
	/// Additional guideline bullets appended to the default rules.
	pub prompt_guidelines: Option<Vec<String>>,
	/// Text appended from user configuration before project context.
	pub append_system_prompt: Option<String>,
	/// Additional XML-wrapped prompt sections keyed by tag name.
	pub sections: Option<BTreeMap<String, String>>,
	/// Working directory.
	pub cwd: String,
	/// Pre-loaded context files.
	pub context_files: Option<Vec<ContextFileEntry>>,
	/// Pre-loaded skills.
	pub skills: Option<Vec<Skill>>,
}

/// `BuildSystemPromptOptions.contextFiles` entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextFileEntry {
	pub path: String,
	pub content: String,
}

/// Upstream `core/system-prompt.ts` `NormalizedBuildSystemPromptOptions`:
/// every list/map field is present after normalization. Plain struct — see
/// [`BuildSystemPromptOptions`] (D30).
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedBuildSystemPromptOptions {
	pub custom_prompt: Option<String>,
	pub force_system_prompt: Option<String>,
	pub selected_tools: Vec<String>,
	pub tool_snippets: BTreeMap<String, String>,
	pub tool_guidelines: BTreeMap<String, Vec<String>>,
	pub prompt_guidelines: Vec<String>,
	pub append_system_prompt: String,
	pub sections: BTreeMap<String, String>,
	pub cwd: String,
	pub context_files: Option<Vec<ContextFileEntry>>,
	pub skills: Option<Vec<Skill>>,
}

impl From<BuildSystemPromptOptions> for NormalizedBuildSystemPromptOptions {
	fn from(options: BuildSystemPromptOptions) -> Self {
		NormalizedBuildSystemPromptOptions {
			custom_prompt: options.custom_prompt,
			force_system_prompt: options.force_system_prompt,
			selected_tools: options.selected_tools.unwrap_or_else(|| {
				["read", "bash", "edit", "write"]
					.iter()
					.map(|tool| (*tool).to_string())
					.collect()
			}),
			tool_snippets: options.tool_snippets.unwrap_or_default(),
			tool_guidelines: options.tool_guidelines.unwrap_or_default(),
			prompt_guidelines: options.prompt_guidelines.unwrap_or_default(),
			append_system_prompt: options.append_system_prompt.unwrap_or_default(),
			sections: options.sections.unwrap_or_default(),
			cwd: options.cwd,
			context_files: options.context_files,
			skills: options.skills,
		}
	}
}

/// Upstream `core/mcp-servers.ts` `McpServerConfig` (module ported
/// separately): the stdio/http server config union, carried as raw JSON until
/// the home module lands.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct McpServerConfig(pub JsonValue);

/// Upstream `core/mcp-servers.ts` `RegisteredMcpServer` (module ported
/// separately). `config` stays raw JSON until `McpServerConfig` is ported.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RegisteredMcpServer {
	pub name: String,
	pub config: McpServerConfig,
	/// Path of the extension that registered the server.
	pub extension_path: String,
}

/// Upstream `core/mcp-servers.ts` `McpServerRegistry` (module ported
/// separately): name → registered-server map owned by the runner. Minimal
/// in-crate storage for the loader/runner (register/unregister/list with
/// extension-path ownership); change listeners and connection lifecycle land
/// with the full module port (D46).
#[derive(Debug, Default)]
pub struct McpServerRegistry {
	servers: parking_lot::Mutex<Vec<RegisteredMcpServer>>,
}

impl Clone for McpServerRegistry {
	fn clone(&self) -> Self {
		McpServerRegistry { servers: parking_lot::Mutex::new(self.servers.lock().clone()) }
	}
}

impl McpServerRegistry {
	pub fn register(&self, server: RegisteredMcpServer) {
		self.servers.lock().push(server);
	}

	/// Removes the server; `true` when it was registered by `extension_path`.
	pub fn unregister(&self, name: &str, extension_path: &str) -> bool {
		let mut servers = self.servers.lock();
		let before = servers.len();
		servers.retain(|server| !(server.name == name && server.extension_path == extension_path));
		servers.len() != before
	}

	pub fn get(&self, name: &str) -> Option<RegisteredMcpServer> {
		self.servers.lock().iter().find(|s| s.name == name).cloned()
	}

	pub fn list(&self) -> Vec<RegisteredMcpServer> {
		self.servers.lock().clone()
	}
}

/// Upstream `core/model-registry.ts` `ModelRegistry`: a synchronous facade
/// over `ModelRuntime` (API-key resolution, provider lookup, streaming).
/// Real port landed at [`crate::core::model_registry`]; this re-export
/// replaces the earlier unit stand-in (D29, D64).
pub use crate::core::model_registry::ModelRegistry;

/// Offline [`ModelRegistry`] for tests: a [`ModelRuntime`] over an in-memory
/// credential store with no models.json and no refresh (test fixture).
#[cfg(test)]
pub(crate) fn offline_registry_for_tests() -> std::sync::Arc<ModelRegistry> {
	use std::sync::{Arc, OnceLock};
	static TEST_REGISTRY: OnceLock<Arc<ModelRegistry>> = OnceLock::new();
	TEST_REGISTRY
		.get_or_init(|| {
			// Build off-thread: callers run inside tokio workers.
			std::thread::spawn(|| {
				let runtime = tokio::runtime::Builder::new_current_thread()
					.build()
					.expect("test runtime");
				runtime.block_on(async {
					let rt = crate::core::model_runtime::ModelRuntime::create(
						crate::core::model_runtime::CreateModelRuntimeOptions {
							credentials: Some(Arc::new(
								pi_ai::auth::credential_store::InMemoryCredentialStore::new(),
							)),
							models_path: Some(None),
							refresh_on_create: false,
							..Default::default()
						},
					)
					.await
					.expect("offline model runtime");
					Arc::new(ModelRegistry::new(rt))
				})
			})
			.join()
			.expect("offline registry builder")
		})
		.clone()
}

// ---------------------------------------------------------------------------
// TUI stand-ins (the `pi-tui` package and the `modes/interactive` theme layer
// are excluded by port decision). These carry the renderer signatures 1:1;
// ported (non-interactive) modes never invoke them.
// ---------------------------------------------------------------------------

/// A rendered TUI component (`pi-tui` `Component`). Opaque: interactive mode
/// is excluded, so no ported code constructs one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Component;

/// The interactive TUI (`pi-tui` `TUI`). Opaque stand-in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tui;

/// Editor styling theme (`pi-tui` `EditorTheme`). Opaque stand-in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EditorTheme;

/// A custom editor component (`pi-tui` `EditorComponent`). Opaque stand-in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EditorComponent;

/// Autocomplete provider (`pi-tui` `AutocompleteProvider`). Opaque stand-in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AutocompleteProvider;

/// Autocomplete suggestion (`pi-tui` `AutocompleteItem`). Opaque stand-in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AutocompleteItem;

/// Overlay visibility handle (`pi-tui` `OverlayHandle`). Opaque stand-in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OverlayHandle;

/// Overlay positioning/sizing options (`pi-tui` `OverlayOptions`). Opaque
/// stand-in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OverlayOptions;

/// Key id (`pi-tui` `KeyId`): the wire form is a string like `"ctrl+d"`.
pub type KeyId = String;

/// Interactive theme (`modes/interactive/theme/theme.ts` `Theme`). Opaque
/// stand-in: the theme layer is TUI-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Theme;

/// Upstream `core/footer-data-provider.ts` `ReadonlyFooterDataProvider`
/// (module ported separately). Opaque stand-in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReadonlyFooterDataProvider;

/// Upstream `core/keybindings.ts` `KeybindingsManager` (module ported
/// separately). Opaque stand-in: keybinding dispatch is interactive-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KeybindingsManager;

/// Upstream `core/keybindings.ts` `AppKeybinding`: a keybinding action name
/// (upstream a string literal union; carried as a string until the module
/// ports).
pub type AppKeybinding = String;

/// Unsubscribe handle returned by event/UI subscriptions (`() => void`).
pub type Unsubscribe = Box<dyn FnOnce() + Send>;

/// Upstream `core/event-bus.ts` `EventBus` (module ported separately): the
/// shared pub/sub for extension communication.
pub trait EventBus: Send + Sync {
	/// Broadcast `data` on `channel`.
	fn emit(&self, channel: &str, data: JsonValue);
	/// Subscribe to `channel`; returns an unsubscribe function.
	fn on(&self, channel: &str, handler: EventBusHandler) -> Unsubscribe;
}

/// Handler of an [`EventBus`] subscription.
pub type EventBusHandler = Arc<dyn Fn(JsonValue) + Send + Sync>;

/// Upstream `core/tools/bash.ts` `BashOperations` — the real trait landed
/// with the tools port; the earlier stand-in trait is replaced by a
/// re-export (D29).
pub use crate::core::tools::bash::{BashExecOptions as BashOperationsExecOptions, BashExecOutcome as BashOperationsExecOutcome, BashOperations};

// ---------------------------------------------------------------------------
// Virtual-model stand-ins (upstream `core/virtual-models.ts`, module ported
/// separately). Only the route-callable surface needed by
/// [`ExtensionVirtualModel`] is carried here.
// ---------------------------------------------------------------------------

/// Why a virtual-model route runs (upstream `ModelRouteReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelRouteReason {
	#[serde(rename = "user")]
	User,
	#[serde(rename = "continuation")]
	Continuation,
	#[serde(rename = "retry")]
	Retry,
	#[serde(rename = "direct")]
	Direct,
}

/// Physical model + thinking level reference (`{ model, thinkingLevel? }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RouteModelRef {
	pub model: Model,
	#[serde(rename = "thinkingLevel")]
	pub thinking_level: Option<ModelThinkingLevel>,
}

/// Upstream `ModelRouteRequest<TState>` with `TState` erased to JSON (D27).
/// Plain struct: carries the live `AbortSignal` (D30).
#[derive(Debug, Clone)]
pub struct ModelRouteRequest {
	/// The selected virtual model.
	pub model: Model,
	/// The selected thinking level; its meaning is up to the router.
	pub thinking_level: ModelThinkingLevel,
	pub reason: ModelRouteReason,
	/// Physical model/thinking level of the latest successful response.
	pub previous: Option<RouteModelRef>,
	/// For `retry`: the failed request, which `messages` no longer contains.
	pub failed: Option<RouteFailedRequest>,
	/// Router state last returned on this session branch.
	pub state: Option<JsonValue>,
	/// Conversation for this request, including system messages.
	pub messages: Vec<Message>,
	pub signal: Option<AbortSignal>,
}

/// The `failed` member of [`ModelRouteRequest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RouteFailedRequest {
	pub model: Model,
	#[serde(rename = "thinkingLevel")]
	pub thinking_level: Option<ModelThinkingLevel>,
	/// The failed assistant message; carries `stopReason`/`errorMessage`.
	pub message: AssistantMessage,
}

/// Upstream `ModelRoute<TState>`: the physical model and thinking level for
/// one request (TState erased to JSON, D27).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRoute {
	pub model: Model,
	#[serde(rename = "thinkingLevel")]
	pub thinking_level: ModelThinkingLevel,
	/// New router state; returning `request.state` (or `None`) keeps the
	/// current state.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub state: Option<JsonValue>,
}

/// Upstream `VirtualModelDefinition<TState>` (module ported separately): the
/// registration payload queued by [`ExtensionRuntimeState`]. TState erased to
/// JSON (D27). Plain struct: carries the `route` callback (D31).
#[derive(Clone)]
pub struct VirtualModelDefinition {
	/// Provider the virtual model is listed under.
	pub provider: String,
	/// Model id; must not collide with a physical model of `provider`.
	pub id: String,
	pub name: String,
	/// Thinking levels offered for selection. Default: `["off"]`.
	pub thinking_levels: Option<Vec<ModelThinkingLevel>>,
	/// Limits shown before the first response.
	pub context_window: Option<u64>,
	pub max_tokens: Option<u64>,
	/// Input types accepted for selection. Default: text and images.
	pub input: Option<Vec<ModelInput>>,
	/// Pick the physical model and thinking level for one request.
	pub route: VirtualModelRouteFn,
}

/// Router callback of a [`VirtualModelDefinition`] (no extension context).
pub type VirtualModelRouteFn =
	Arc<dyn Fn(ModelRouteRequest) -> BoxFuture<'static, ModelRoute> + Send + Sync>;

impl std::fmt::Debug for VirtualModelDefinition {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("VirtualModelDefinition")
			.field("provider", &self.provider)
			.field("id", &self.id)
			.field("name", &self.name)
			.finish_non_exhaustive()
	}
}

// ============================================================================
// UI Context
// ============================================================================

/// Options for extension UI dialogs (`ExtensionUIDialogOptions`).
#[derive(Debug, Clone, Default)]
pub struct ExtensionUiDialogOptions {
	/// Signal to programmatically dismiss the dialog.
	pub signal: Option<AbortSignal>,
	/// Timeout in milliseconds; the dialog auto-dismisses with a countdown.
	pub timeout: Option<u64>,
}

/// Placement for extension widgets (`WidgetPlacement`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WidgetPlacement {
	#[serde(rename = "aboveEditor")]
	AboveEditor,
	#[serde(rename = "belowEditor")]
	BelowEditor,
}

/// Options for extension widgets (`ExtensionWidgetOptions`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtensionWidgetOptions {
	/// Where the widget is rendered. Default: `aboveEditor`.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub placement: Option<WidgetPlacement>,
}

/// Result value of a [`TerminalInputHandler`]
/// (`{ consume?: boolean; data?: string } | undefined`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalInputAck {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub consume: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub data: Option<String>,
}

/// Raw terminal input listener for extensions (`TerminalInputHandler`).
/// Returning `None` leaves the input untouched.
pub type TerminalInputHandler = Arc<dyn Fn(&str) -> Option<TerminalInputAck> + Send + Sync>;

/// Working indicator configuration (`WorkingIndicatorOptions`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkingIndicatorOptions {
	/// Animation frames; `[]` hides the indicator, custom frames render verbatim.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub frames: Option<Vec<String>>,
	/// Frame interval in milliseconds.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub interval_ms: Option<u64>,
}

/// Notification severity of [`ExtensionUi::notify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotifyKind {
	#[serde(rename = "info")]
	Info,
	#[serde(rename = "warning")]
	Warning,
	#[serde(rename = "error")]
	Error,
}

/// Theme name/path pair of [`ExtensionUi::get_all_themes`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThemeInfo {
	pub name: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub path: Option<String>,
}

/// Argument of [`ExtensionUi::set_theme`] (`string | Theme`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetThemeArg {
	Name(String),
	Theme(Theme),
}

/// Result of [`ExtensionUi::set_theme`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThemeSwitchResult {
	pub success: bool,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub error: Option<String>,
}

/// Widget content of [`ExtensionUi::set_widget`] (string array or component
/// factory overloads).
pub enum WidgetContent {
	Text(Vec<String>),
	Factory(WidgetFactory),
}

/// `(tui, theme) => Component & { dispose?() }` widget factory.
pub type WidgetFactory = Arc<dyn Fn(&Tui, &Theme) -> Component + Send + Sync>;

/// `(tui, theme, footerData) => Component` footer factory.
pub type FooterFactory = Arc<dyn Fn(&Tui, &Theme, &ReadonlyFooterDataProvider) -> Component + Send + Sync>;

/// `(tui, theme) => Component` header factory.
pub type HeaderFactory = Arc<dyn Fn(&Tui, &Theme) -> Component + Send + Sync>;

/// `done(result)` callback of a custom component factory; the result value is
/// erased to JSON (D27).
pub type CustomComponentDone = Box<dyn FnOnce(JsonValue) + Send>;

/// Factory of [`ExtensionUi::custom`]
/// (`(tui, theme, keybindings, done) => Component | Promise<Component>`).
pub type CustomComponentFactory =
	Arc<dyn Fn(&Tui, &Theme, &KeybindingsManager, CustomComponentDone) -> BoxFuture<'static, Component> + Send + Sync>;

/// Options of [`ExtensionUi::custom`].
#[derive(Default)]
pub struct CustomComponentOptions {
	pub overlay: Option<bool>,
	/// Overlay options, static or computed per update.
	pub overlay_options: Option<OverlayOptionsArg>,
	/// Called with the overlay handle after the overlay is shown.
	pub on_handle: Option<Arc<dyn Fn(OverlayHandle) + Send + Sync>>,
}

/// `OverlayOptions | (() => OverlayOptions)`.
pub enum OverlayOptionsArg {
	Static(OverlayOptions),
	Dynamic(Arc<dyn Fn() -> OverlayOptions + Send + Sync>),
}

/// Wrap the current autocomplete provider with additional behavior
/// (`AutocompleteProviderFactory`).
pub type AutocompleteProviderFactory =
	Arc<dyn Fn(Arc<AutocompleteProvider>) -> Arc<AutocompleteProvider> + Send + Sync>;

/// `(tui, theme, keybindings) => EditorComponent` (`EditorFactory`).
pub type EditorFactory = Arc<dyn Fn(&Tui, &EditorTheme, &KeybindingsManager) -> EditorComponent + Send + Sync>;

/// UI context for extensions to request interactive UI
/// (`ExtensionUIContext`). Each mode (interactive, RPC, print) provides its
/// own implementation; TUI-only surface is carried by the opaque stand-in
/// types (D29).
pub trait ExtensionUi: Send + Sync {
	/// Show a selector and return the user's choice.
	fn select(
		&self,
		title: String,
		options: Vec<String>,
		opts: Option<ExtensionUiDialogOptions>,
	) -> BoxFuture<'static, Option<String>>;

	/// Show a confirmation dialog.
	fn confirm(&self, title: String, message: String, opts: Option<ExtensionUiDialogOptions>)
		-> BoxFuture<'static, bool>;

	/// Show a text input dialog.
	fn input(
		&self,
		title: String,
		placeholder: Option<String>,
		opts: Option<ExtensionUiDialogOptions>,
	) -> BoxFuture<'static, Option<String>>;

	/// Show a notification to the user.
	fn notify(&self, message: String, kind: Option<NotifyKind>);

	/// Listen to raw terminal input (interactive mode only); returns an
	/// unsubscribe function.
	fn on_terminal_input(&self, handler: TerminalInputHandler) -> Unsubscribe;

	/// Set status text in the footer/status bar; `None` clears.
	fn set_status(&self, key: String, text: Option<String>);

	/// Set the working/loading message shown during streaming; `None`
	/// restores the default.
	fn set_working_message(&self, message: Option<String>);

	/// Show or hide the built-in interactive working loader row.
	fn set_working_visible(&self, visible: bool);

	/// Configure the interactive working indicator (`None` restores the
	/// default animated spinner).
	fn set_working_indicator(&self, options: Option<WorkingIndicatorOptions>);

	/// Set the label shown for hidden thinking blocks; `None` restores the
	/// default.
	fn set_hidden_thinking_label(&self, label: Option<String>);

	/// Set a widget above or below the editor; `None` removes it.
	fn set_widget(&self, key: String, content: Option<WidgetContent>, options: Option<ExtensionWidgetOptions>);

	/// Set a custom footer component; `None` restores the built-in footer.
	fn set_footer(&self, factory: Option<FooterFactory>);

	/// Set a custom header component; `None` restores the built-in header.
	fn set_header(&self, factory: Option<HeaderFactory>);

	/// Set the terminal window/tab title.
	fn set_title(&self, title: String);

	/// Show a custom component with keyboard focus; the typed result `T` is
	/// erased to JSON (D27).
	fn custom(&self, factory: CustomComponentFactory, options: Option<CustomComponentOptions>)
		-> BoxFuture<'static, JsonValue>;

	/// Paste text into the editor, triggering paste handling.
	fn paste_to_editor(&self, text: String);

	/// Set the text in the core input editor.
	fn set_editor_text(&self, text: String);

	/// Get the current text from the core input editor.
	fn get_editor_text(&self) -> String;

	/// Show a multi-line editor for text editing.
	fn editor(&self, title: String, prefill: Option<String>) -> BoxFuture<'static, Option<String>>;

	/// Stack additional autocomplete behavior on top of the built-in provider.
	fn add_autocomplete_provider(&self, factory: AutocompleteProviderFactory);

	/// Set a custom editor component; `None` restores the default editor.
	fn set_editor_component(&self, factory: Option<EditorFactory>);

	/// Get the currently configured custom editor factory.
	fn get_editor_component(&self) -> Option<EditorFactory>;

	/// Get the current theme for styling.
	fn theme(&self) -> Theme;

	/// Get all available themes with their names and file paths.
	fn get_all_themes(&self) -> Vec<ThemeInfo>;

	/// Load a theme by name without switching to it.
	fn get_theme(&self, name: &str) -> Option<Theme>;

	/// Set the current theme by name or theme object.
	fn set_theme(&self, theme: SetThemeArg) -> ThemeSwitchResult;

	/// Get current tool output expansion state.
	fn get_tools_expanded(&self) -> bool;

	/// Set tool output expansion state.
	fn set_tools_expanded(&self, expanded: bool);
}

// ============================================================================
// Extension Context
// ============================================================================

/// Estimated context usage (`ContextUsage`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextUsage {
	/// Estimated context tokens, or `None` if unknown (e.g. right after
	/// compaction, before the next LLM response).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub tokens: Option<u64>,
	#[serde(rename = "contextWindow")]
	pub context_window: u64,
	/// Context usage as percentage of the context window.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub percent: Option<f64>,
}

/// Callbacks of a triggered compaction (`CompactOptions`).
#[derive(Default)]
pub struct CompactOptions {
	pub custom_instructions: Option<String>,
	pub on_complete: Option<Arc<dyn Fn(CompactionResult) + Send + Sync>>,
	pub on_error: Option<Arc<dyn Fn(String) + Send + Sync>>,
}

/// Context passed to extension event handlers (`ExtensionMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtensionMode {
	#[serde(rename = "tui")]
	Tui,
	#[serde(rename = "rpc")]
	Rpc,
	#[serde(rename = "json")]
	Json,
	#[serde(rename = "print")]
	Print,
}

/// Context passed to extension event handlers (`ExtensionContext`).
///
/// Data fields carry the session state; the behavioral members are `Arc`
/// vtable fields populated by the mode's runner (upstream
/// `ExtensionContextActions`) with forwarding methods below (D28).
#[derive(Clone)]
pub struct ExtensionContext {
	/// UI methods for user interaction.
	pub ui: Arc<dyn ExtensionUi>,
	/// Current run mode; use `Tui` to guard terminal-only UI.
	pub mode: ExtensionMode,
	/// Whether dialog-capable UI is available (true in TUI and RPC modes).
	pub has_ui: bool,
	/// Current working directory.
	pub cwd: String,
	/// Session manager (upstream `ReadonlySessionManager` — D28). Shared
	/// mutable in Rust (agent-session D74); lock for reads and writes.
	pub session_manager: Arc<parking_lot::Mutex<SessionManager>>,
	/// Model registry for API key resolution.
	pub model_registry: Arc<ModelRegistry>,
	/// Current model (may be `None`).
	pub model: Option<Model>,
	/// Models scoped to this session (`--models` / `enabledModels`).
	pub scoped_models: Vec<ScopedModel>,
	/// Current thinking level, when provided by the session runtime.
	pub thinking_level: Option<AgentThinkingLevel>,
	/// The current abort signal, or `None` when the agent is not streaming.
	pub signal: Option<AbortSignal>,

	// Action implementations (upstream `ExtensionContextActions`).
	pub(crate) is_idle: Arc<dyn Fn() -> bool + Send + Sync>,
	pub(crate) is_project_trusted: Arc<dyn Fn() -> bool + Send + Sync>,
	pub(crate) abort: Arc<dyn Fn() + Send + Sync>,
	pub(crate) has_pending_messages: Arc<dyn Fn() -> bool + Send + Sync>,
	pub(crate) shutdown: Arc<dyn Fn() + Send + Sync>,
	pub(crate) get_context_usage: Arc<dyn Fn() -> Option<ContextUsage> + Send + Sync>,
	pub(crate) compact: Arc<dyn Fn(Option<CompactOptions>) + Send + Sync>,
	pub(crate) get_system_prompt: Arc<dyn Fn() -> String + Send + Sync>,
}

impl std::fmt::Debug for ExtensionContext {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ExtensionContext")
			.field("mode", &self.mode)
			.field("has_ui", &self.has_ui)
			.field("cwd", &self.cwd)
			.field("model", &self.model)
			.field("thinking_level", &self.thinking_level)
			.finish_non_exhaustive()
	}
}

impl ExtensionContext {
	/// Whether the agent is idle (not streaming).
	pub fn is_idle(&self) -> bool {
		(self.is_idle)()
	}

	/// Whether project-local trust is active for this context.
	pub fn is_project_trusted(&self) -> bool {
		(self.is_project_trusted)()
	}

	/// Abort the current agent operation.
	pub fn abort(&self) {
		(self.abort)()
	}

	/// Whether there are queued messages waiting.
	pub fn has_pending_messages(&self) -> bool {
		(self.has_pending_messages)()
	}

	/// Gracefully shutdown pi and exit.
	pub fn shutdown(&self) {
		(self.shutdown)()
	}

	/// Get current context usage for the active model.
	pub fn get_context_usage(&self) -> Option<ContextUsage> {
		(self.get_context_usage)()
	}

	/// Trigger compaction without awaiting completion.
	pub fn compact(&self, options: Option<CompactOptions>) {
		(self.compact)(options)
	}

	/// Get the current effective system prompt.
	pub fn get_system_prompt(&self) -> String {
		(self.get_system_prompt)()
	}
}

/// Options for [`ExtensionToolContext::execute_tool`] (`ExecuteToolOptions`).
#[derive(Default)]
pub struct ExecuteToolOptions {
	/// Defaults to the calling tool's signal.
	pub signal: Option<AbortSignal>,
	/// Receives partial results of the nested tool, in addition to
	/// `tool_execution_update` events.
	pub on_update: Option<AgentToolUpdateCallback>,
}

/// `{ cancelled: boolean }` result of session control commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSwitchOutcome {
	pub cancelled: bool,
}

/// `setup(withSession)` callback option of session control commands.
pub type WithSessionCallback = Arc<dyn Fn(ReplacedSessionContext) -> BoxFuture<'static, ()> + Send + Sync>;

/// Options of [`ExtensionCommandContext::new_session`].
#[derive(Default)]
pub struct NewSessionCommandOptions {
	/// Parent session for the new branch.
	pub parent_session: Option<String>,
	/// Run before the switch commits.
	pub setup: Option<Arc<dyn Fn(Arc<SessionManager>) -> BoxFuture<'static, ()> + Send + Sync>>,
	/// Run with a fresh command context bound to the replacement session.
	pub with_session: Option<WithSessionCallback>,
}

/// Position of a fork relative to the entry (`"before" | "at"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForkPosition {
	#[serde(rename = "before")]
	Before,
	#[serde(rename = "at")]
	At,
}

/// Options of [`ExtensionCommandContext::fork`].
#[derive(Default)]
pub struct ForkCommandOptions {
	pub position: Option<ForkPosition>,
	pub with_session: Option<WithSessionCallback>,
}

/// Options of [`ExtensionCommandContext::navigate_tree`].
#[derive(Default)]
pub struct NavigateTreeOptions {
	pub summarize: Option<bool>,
	pub custom_instructions: Option<String>,
	pub replace_instructions: Option<bool>,
	pub label: Option<String>,
}

/// Options of [`ExtensionCommandContext::switch_session`].
#[derive(Default)]
pub struct SwitchSessionOptions {
	pub with_session: Option<WithSessionCallback>,
}

/// Extended context for command handlers (`ExtensionCommandContext`):
/// session control methods only safe in user-initiated commands (D28).
#[derive(Clone)]
pub struct ExtensionCommandContext {
	/// Base extension context.
	pub ctx: ExtensionContext,
	pub(crate) get_system_prompt_options: Arc<dyn Fn() -> BuildSystemPromptOptions + Send + Sync>,
	pub(crate) wait_for_idle: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>,
	pub(crate) new_session: Arc<dyn Fn(Option<NewSessionCommandOptions>) -> BoxFuture<'static, SessionSwitchOutcome> + Send + Sync>,
	pub(crate) fork: Arc<dyn Fn(String, Option<ForkCommandOptions>) -> BoxFuture<'static, SessionSwitchOutcome> + Send + Sync>,
	pub(crate) navigate_tree: Arc<dyn Fn(String, Option<NavigateTreeOptions>) -> BoxFuture<'static, SessionSwitchOutcome> + Send + Sync>,
	pub(crate) switch_session: Arc<dyn Fn(String, Option<SwitchSessionOptions>) -> BoxFuture<'static, SessionSwitchOutcome> + Send + Sync>,
	pub(crate) reload: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>,
}

impl std::fmt::Debug for ExtensionCommandContext {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ExtensionCommandContext").field("ctx", &self.ctx).finish()
	}
}

impl std::ops::Deref for ExtensionCommandContext {
	type Target = ExtensionContext;

	fn deref(&self) -> &Self::Target {
		&self.ctx
	}
}

impl ExtensionCommandContext {
	/// Get the current base system-prompt construction options.
	pub fn get_system_prompt_options(&self) -> BuildSystemPromptOptions {
		(self.get_system_prompt_options)()
	}

	/// Wait for the agent to finish streaming.
	pub fn wait_for_idle(&self) -> BoxFuture<'static, ()> {
		(self.wait_for_idle)()
	}

	/// Start a new session, optionally with initialization.
	pub fn new_session(&self, options: Option<NewSessionCommandOptions>) -> BoxFuture<'static, SessionSwitchOutcome> {
		(self.new_session)(options)
	}

	/// Fork from a specific entry, creating a new session file.
	pub fn fork(&self, entry_id: String, options: Option<ForkCommandOptions>) -> BoxFuture<'static, SessionSwitchOutcome> {
		(self.fork)(entry_id, options)
	}

	/// Navigate to a different point in the session tree.
	pub fn navigate_tree(
		&self,
		target_id: String,
		options: Option<NavigateTreeOptions>,
	) -> BoxFuture<'static, SessionSwitchOutcome> {
		(self.navigate_tree)(target_id, options)
	}

	/// Switch to a different session file.
	pub fn switch_session(
		&self,
		session_path: String,
		options: Option<SwitchSessionOptions>,
	) -> BoxFuture<'static, SessionSwitchOutcome> {
		(self.switch_session)(session_path, options)
	}

	/// Reload extensions, skills, prompts, themes, and context files.
	pub fn reload(&self) -> BoxFuture<'static, ()> {
		(self.reload)()
	}
}

/// Content of [`ReplacedSessionContext::send_user_message`]-style calls
/// (`string | (TextContent | ImageContent)[]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SendUserMessageContent {
	Text(String),
	Blocks(Vec<pi_ai::types::UserContent>),
}

/// The `Pick<CustomMessage, ...>` draft sent by `sendMessage` handlers
/// (`{ customType, content, display, details }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessageDraft {
	#[serde(rename = "customType")]
	pub custom_type: String,
	pub content: UserContentField,
	pub display: bool,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub details: Option<JsonValue>,
}

/// Options of `sendMessage` (`{ triggerTurn?, deliverAs? }`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendMessageOptions {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub trigger_turn: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub deliver_as: Option<DeliveryTarget>,
}

/// Where a queued message is delivered (`"steer" | "followUp" | "nextTurn"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryTarget {
	#[serde(rename = "steer")]
	Steer,
	#[serde(rename = "followUp")]
	FollowUp,
	#[serde(rename = "nextTurn")]
	NextTurn,
}

/// Options of `sendUserMessage` (`{ deliverAs?, expandPromptTemplates? }`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendUserMessageOptions {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub deliver_as: Option<DeliveryTarget>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub expand_prompt_templates: Option<bool>,
}

/// Message sender of [`ReplacedSessionContext`].
pub type SendMessageHandler =
	Arc<dyn Fn(CustomMessageDraft, Option<SendMessageOptions>) -> BoxFuture<'static, ()> + Send + Sync>;

/// User-message sender of [`ReplacedSessionContext`].
pub type SendUserMessageHandler =
	Arc<dyn Fn(SendUserMessageContent, Option<SendUserMessageOptions>) -> BoxFuture<'static, ()> + Send + Sync>;

/// Fresh command-capable context bound to the replacement session after a
/// session switch (`ReplacedSessionContext`), passed to `withSession()`
/// callbacks (D28).
#[derive(Clone)]
pub struct ReplacedSessionContext {
	/// Command context.
	pub ctx: ExtensionCommandContext,
	pub(crate) send_message: SendMessageHandler,
	pub(crate) send_user_message: SendUserMessageHandler,
}

impl std::fmt::Debug for ReplacedSessionContext {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ReplacedSessionContext").field("ctx", &self.ctx).finish()
	}
}

impl std::ops::Deref for ReplacedSessionContext {
	type Target = ExtensionCommandContext;

	fn deref(&self) -> &Self::Target {
		&self.ctx
	}
}

impl ReplacedSessionContext {
	/// Send a custom message to the (new) session.
	pub fn send_message(&self, message: CustomMessageDraft, options: Option<SendMessageOptions>) -> BoxFuture<'static, ()> {
		(self.send_message)(message, options)
	}

	/// Send a user message to the agent.
	pub fn send_user_message(
		&self,
		content: SendUserMessageContent,
		options: Option<SendUserMessageOptions>,
	) -> BoxFuture<'static, ()> {
		(self.send_user_message)(content, options)
	}
}

/// Execute-tool callback of [`ExtensionToolContext`]. Never fails: unknown
/// tools, validation errors, blocked calls, and thrown errors come back as
/// `is_error: true` results.
pub type ExecuteToolFn =
	Arc<dyn Fn(String, JsonValue, ExecuteToolOptions) -> BoxFuture<'static, AgentToolCallOutcome> + Send + Sync>;

/// Context passed to tool `execute()` in a session (`ExtensionToolContext`):
/// the extension context plus `executeTool()` for running other tools through
/// the same validation, hooks, and permission checks as model-issued calls.
#[derive(Clone)]
pub struct ExtensionToolContext {
	/// Base extension context.
	pub ctx: ExtensionContext,
	/// Tools [`execute_tool`](Self::execute_tool) can call.
	pub tools: Vec<AgentTool>,
	pub(crate) execute_tool: ExecuteToolFn,
}

impl std::fmt::Debug for ExtensionToolContext {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ExtensionToolContext")
			.field("ctx", &self.ctx)
			.field("tools", &[self.tools.iter().map(|tool| tool.tool.name.clone())])
			.finish()
	}
}

impl std::ops::Deref for ExtensionToolContext {
	type Target = ExtensionContext;

	fn deref(&self) -> &Self::Target {
		&self.ctx
	}
}

impl ExtensionToolContext {
	/// Run another tool. The call gets the id `<calling id>/<n>`; it does not
	/// appear in the transcript.
	pub fn execute_tool(&self, name: String, args: JsonValue, options: ExecuteToolOptions) -> BoxFuture<'static, AgentToolCallOutcome> {
		(self.execute_tool)(name, args, options)
	}
}

// ============================================================================
// Tool Types
// ============================================================================

/// Rendering options for tool results (`ToolRenderResultOptions`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolRenderResultOptions {
	/// Whether the result view is expanded.
	pub expanded: bool,
	/// Whether this is a partial/streaming result.
	#[serde(rename = "isPartial")]
	pub is_partial: bool,
}

/// Context passed to tool renderers (`ToolRenderContext<TState, TArgs>`) with
/// `TState`/`TArgs` erased to JSON (D27).
#[derive(Clone)]
pub struct ToolRenderContext {
	/// Current tool call arguments; shared across call/result renders for the
	/// same tool call.
	pub args: JsonValue,
	/// Unique id for this tool execution; stable across call/result renders.
	pub tool_call_id: String,
	/// Invalidate just this tool execution component for redraw.
	pub invalidate: Arc<dyn Fn() + Send + Sync>,
	/// Previously returned component for this render slot, if any.
	pub last_component: Option<Component>,
	/// Shared renderer state for this tool row.
	pub state: JsonValue,
	/// Working directory for this tool execution.
	pub cwd: String,
	/// Whether the tool execution has started.
	pub execution_started: bool,
	/// Whether the tool call arguments are complete.
	pub args_complete: bool,
	/// Whether the tool result is partial/streaming.
	pub is_partial: bool,
	/// Whether the result view is expanded.
	pub expanded: bool,
	/// Whether inline images are currently shown in the TUI.
	pub show_images: bool,
	/// Whether the current result is an error.
	pub is_error: bool,
}

impl std::fmt::Debug for ToolRenderContext {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ToolRenderContext")
			.field("tool_call_id", &self.tool_call_id)
			.field("cwd", &self.cwd)
			.finish_non_exhaustive()
	}
}

/// How the model reaches a tool (`ToolExposure`).
///
/// - `direct`: declared to the model while active, and callable while active.
/// - `model_only`: declared to the model while active, never callable.
/// - `codemode`: callable whenever registered; not declared unless activated.
/// - `deferred`: like `codemode`, but codemode tools do not list it.
/// - `hidden`: registered but unreachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolExposure {
	#[serde(rename = "direct")]
	Direct,
	#[serde(rename = "model-only")]
	ModelOnly,
	#[serde(rename = "codemode")]
	Codemode,
	#[serde(rename = "deferred")]
	Deferred,
	#[serde(rename = "hidden")]
	Hidden,
}

impl Default for ToolExposure {
	fn default() -> Self {
		ToolExposure::Direct
	}
}

/// Hints about what a tool does, with the meaning of MCP tool annotations
/// (`ToolAnnotations`). Unverified author hints; permission extensions use
/// them to decide which calls to confirm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolAnnotations {
	/// The tool does not modify its environment.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub read_only_hint: Option<bool>,
	/// The tool may delete or overwrite data.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub destructive_hint: Option<bool>,
	/// Repeating a call with the same arguments has no further effect.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub idempotent_hint: Option<bool>,
	/// The tool reaches an open world of external entities.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub open_world_hint: Option<bool>,
}

/// A group of related tools, such as the tools of one MCP server
/// (`ToolNamespace`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolNamespace {
	/// For example `mcp__docs`.
	pub name: String,
	/// Short summary shown once with the group in model-facing tool listings.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub description: Option<String>,
	/// Longer usage guidance, such as MCP server instructions.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub instructions: Option<String>,
}

/// The tools of a session as `ToolDefinition.prepareLoadout` sees them
/// (`ToolLoadout`); the lookup methods are vtable fields (D28).
#[derive(Clone)]
pub struct ToolLoadout {
	/// Tools declared to the model (the active tools), in order.
	pub declared: Vec<AgentTool>,
	/// Tools callable through `ctx.executeTool()`.
	pub callable: Vec<AgentTool>,
	/// Every registered tool.
	pub registered: Vec<AgentTool>,
	pub(crate) get_exposure: Arc<dyn Fn(&str) -> ToolExposure + Send + Sync>,
	pub(crate) get_namespace: Arc<dyn Fn(&str) -> Option<ToolNamespace> + Send + Sync>,
}

impl std::fmt::Debug for ToolLoadout {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ToolLoadout")
			.field("declared", &self.declared)
			.field("callable", &self.callable)
			.field("registered", &self.registered)
			.finish_non_exhaustive()
	}
}

impl ToolLoadout {
	/// Exposure of the named tool.
	pub fn get_exposure(&self, name: &str) -> ToolExposure {
		(self.get_exposure)(name)
	}

	/// Namespace of the named tool.
	pub fn get_namespace(&self, name: &str) -> Option<ToolNamespace> {
		(self.get_namespace)(name)
	}
}

/// Changes `ToolDefinition.prepareLoadout` makes to what the model sees
/// (`ToolLoadoutChanges`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolLoadoutChanges {
	/// Model-facing descriptions of declared tools, by tool name.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub descriptions: Option<BTreeMap<String, String>>,
	/// Declared tools whose declarations requests leave out. They stay active
	/// and callable, and the transcript still declares them.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub hidden_declarations: Option<Vec<String>>,
}

/// Whether `ToolExecutionComponent` renders the standard colored shell or the
/// tool renders its own framing (`renderShell`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderShell {
	#[serde(rename = "default")]
	Default,
	#[serde(rename = "self")]
	SelfShell,
}

/// Execute callback of [`ToolDefinition`]. The `Err` path mirrors an
/// upstream-throwing execute (error tool result), matching
/// [`pi_agent::AgentToolExecute`]. Args are the validated JSON object; the
/// result details are erased JSON (D27). The context is `None` when the tool
/// runs outside a session (plain [`pi_agent::AgentTool`], direct calls);
/// tools fall back to their constructor cwd (`ctx?.cwd || cwd` upstream).
pub type ToolExecute = Arc<
	dyn Fn(
			String,
			JsonValue,
			Option<AbortSignal>,
			Option<AgentToolUpdateCallback>,
			Option<ExtensionToolContext>,
		) -> BoxFuture<'static, Result<AgentToolResult, String>>
		+ Send
		+ Sync,
>;

/// Custom rendering for tool call display (`renderCall`); args are the
/// validated JSON object (D27).
pub type ToolRenderCall = Arc<dyn Fn(JsonValue, &Theme, &ToolRenderContext) -> Component + Send + Sync>;

/// Custom rendering for tool result display (`renderResult`).
pub type ToolRenderResult =
	Arc<dyn Fn(&AgentToolResult, &ToolRenderResultOptions, &Theme, &ToolRenderContext) -> Component + Send + Sync>;

/// Adjusts how the loadout is presented to the model while the tool is active
/// (`prepareLoadout`); called whenever the active tools change.
pub type PrepareLoadoutFn = Arc<dyn Fn(&ToolLoadout) -> Option<ToolLoadoutChanges> + Send + Sync>;

/// Tool definition for `registerTool()` (`ToolDefinition<TParams, TDetails,
/// TState>`), with the three type parameters erased (D27):
///
/// - `TParams` → JSON Schema document in [`parameters`](Self::parameters)
///   (`pi-ai` D6 precedent); `execute` receives the validated JSON object.
/// - `TDetails` → erased JSON in [`AgentToolResult::details`], exactly like
///   [`pi_agent::AgentTool`].
/// - `TState` → erased JSON in [`ToolRenderContext::state`].
#[derive(Clone)]
pub struct ToolDefinition {
	/// Tool name (used in LLM tool calls).
	pub name: String,
	/// Human-readable label for UI.
	pub label: String,
	/// Description for the LLM.
	pub description: String,
	/// Optional one-line snippet for the *Available tools* section of the
	/// default system prompt; custom tools are omitted from that section when
	/// absent.
	pub prompt_snippet: Option<String>,
	/// Optional guideline bullets appended to the default system prompt
	/// Guidelines section when this tool is active.
	pub prompt_guidelines: Option<Vec<String>>,
	/// Parameter schema (JSON Schema document; upstream TypeBox — D27).
	pub parameters: JsonValue,
	/// Optional provider-side constrained sampling request; `Disabled` is
	/// equivalent to leaving it unset.
	pub constrained_sampling: Option<ConstrainedSamplingSetting>,
	/// Controls whether the tool result renders the standard shell or the
	/// tool's own framing.
	pub render_shell: Option<RenderShell>,
	/// Compatibility shim to prepare raw tool call arguments before schema
	/// validation; must return an object conforming to `parameters`.
	pub prepare_arguments: Option<Arc<dyn Fn(JsonValue) -> JsonValue + Send + Sync>>,
	/// JSON Schema of `structuredContent` in successful results. Tools that
	/// declare it should always set `structuredContent`; codemode scripts then
	/// receive it instead of the text content.
	pub output_schema: Option<JsonValue>,
	/// How the model reaches the tool. Default: [`ToolExposure::Direct`].
	pub exposure: Option<ToolExposure>,
	/// Group the tool belongs to, for example its MCP server.
	pub namespace: Option<ToolNamespace>,
	/// Hints about what the tool does, for example from an MCP server.
	pub annotations: Option<ToolAnnotations>,
	/// Whether registering the tool activates it. Default: `true` for `direct`
	/// and `model_only` tools; other exposures are never activated on
	/// registration.
	pub default_active: Option<bool>,
	/// Adjust the model-facing loadout while this tool is active.
	pub prepare_loadout: Option<PrepareLoadoutFn>,
	/// Per-tool execution mode override; `None` applies the default.
	pub execution_mode: Option<ToolExecutionMode>,
	/// Execute the tool.
	pub execute: ToolExecute,
	/// Custom rendering for tool call display.
	pub render_call: Option<ToolRenderCall>,
	/// Custom rendering for tool result display.
	pub render_result: Option<ToolRenderResult>,
}

impl std::fmt::Debug for ToolDefinition {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ToolDefinition")
			.field("name", &self.name)
			.field("label", &self.label)
			.field("description", &self.description)
			.field("exposure", &self.exposure)
			.field("execution_mode", &self.execution_mode)
			.finish_non_exhaustive()
	}
}

impl ToolDefinition {
	/// A definition with only the required members; everything else defaults
	/// to absent. (Convenience for the 15-field literal.)
	pub fn new(name: impl Into<String>, label: impl Into<String>, description: impl Into<String>, parameters: JsonValue) -> Self {
		ToolDefinition {
			name: name.into(),
			label: label.into(),
			description: description.into(),
			prompt_snippet: None,
			prompt_guidelines: None,
			parameters,
			constrained_sampling: None,
			render_shell: None,
			prepare_arguments: None,
			output_schema: None,
			exposure: None,
			namespace: None,
			annotations: None,
			default_active: None,
			prepare_loadout: None,
			execution_mode: None,
			execute: Arc::new(|_, _, _, _, _| Box::pin(async { Err("tool has no execute".to_string()) })),
			render_call: None,
			render_result: None,
		}
	}
}

/// The renderer members of a tool definition (`ToolRenderers` =
/// `Pick<ToolDefinition, "renderShell" | "renderCall" | "renderResult">`).
#[derive(Clone, Default)]
pub struct ToolRenderers {
	pub render_shell: Option<RenderShell>,
	pub render_call: Option<ToolRenderCall>,
	pub render_result: Option<ToolRenderResult>,
}

impl std::fmt::Debug for ToolRenderers {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ToolRenderers")
			.field("render_shell", &self.render_shell)
			.finish_non_exhaustive()
	}
}

/// Chooses how calls to a tool are drawn, including tools that are not
/// registered (`ToolRendererResolver`). `next()` returns the renderers the
/// remaining resolvers, then the registered tool, would use.
pub type ToolRendererResolver =
	Arc<dyn Fn(&str, &mut dyn FnMut() -> Option<ToolRenderers>) -> Option<ToolRenderers> + Send + Sync>;

/// Preserve parameter inference for standalone tool definitions (`defineTool`).
///
/// Upstream this is a TS type-inference shim for assignments and arrays where
/// contextual typing would widen params; the Rust port keeps the identity
/// function for 1:1 API surface (D27).
pub fn define_tool<T>(tool: T) -> T {
	tool
}

// ============================================================================
// Startup/Resource Events
// ============================================================================

/// Fired before project trust is resolved (`ProjectTrustEvent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "project_trust")]
pub struct ProjectTrustEvent {
	pub cwd: String,
}

/// Project trust decision (`ProjectTrustEventDecision`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectTrustEventDecision {
	#[serde(rename = "yes")]
	Yes,
	#[serde(rename = "no")]
	No,
	#[serde(rename = "undecided")]
	Undecided,
}

/// Result of a project trust handler (`ProjectTrustEventResult`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectTrustEventResult {
	pub trusted: ProjectTrustEventDecision,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub remember: Option<bool>,
}

impl Default for ProjectTrustEventDecision {
	fn default() -> Self {
		ProjectTrustEventDecision::Undecided
	}
}

/// Context of a project trust handler (`ProjectTrustContext`). The `ui`
/// handle is the full [`ExtensionUi`]; trust handlers only call
/// `select`/`confirm`/`input`/`notify` (upstream narrows the type — D28).
#[derive(Clone)]
pub struct ProjectTrustContext {
	pub cwd: String,
	pub mode: ExtensionMode,
	pub has_ui: bool,
	pub ui: Arc<dyn ExtensionUi>,
}

impl std::fmt::Debug for ProjectTrustContext {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ProjectTrustContext")
			.field("cwd", &self.cwd)
			.field("mode", &self.mode)
			.field("has_ui", &self.has_ui)
			.finish()
	}
}

/// Handler of the `project_trust` event (`ProjectTrustHandler`).
pub type ProjectTrustHandler =
	Arc<dyn Fn(ProjectTrustEvent, ProjectTrustContext) -> BoxFuture<'static, ProjectTrustEventResult> + Send + Sync>;

/// Why [`ResourcesDiscoverEvent`] fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourcesDiscoverReason {
	#[serde(rename = "startup")]
	Startup,
	#[serde(rename = "reload")]
	Reload,
}

/// Fired after session_start so extensions can provide additional resource
/// paths (`ResourcesDiscoverEvent`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "resources_discover", rename_all = "camelCase")]
pub struct ResourcesDiscoverEvent {
	pub cwd: String,
	pub reason: ResourcesDiscoverReason,
}

/// Result from a resources_discover handler (`ResourcesDiscoverResult`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesDiscoverResult {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub skill_paths: Option<Vec<String>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub prompt_paths: Option<Vec<String>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub theme_paths: Option<Vec<String>>,
}

/// Fired when an extension registers or unregisters an MCP server after the
/// extensions are bound (`McpServersChangeEvent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "mcp_servers_change", rename_all = "camelCase")]
pub struct McpServersChangeEvent {
	/// Every registered server after the change.
	pub servers: Vec<RegisteredMcpServer>,
}

// ============================================================================
// Session Events
// ============================================================================

/// Why a session start happened (`SessionStartEvent.reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStartReason {
	#[serde(rename = "startup")]
	Startup,
	#[serde(rename = "reload")]
	Reload,
	#[serde(rename = "new")]
	New,
	#[serde(rename = "resume")]
	Resume,
	#[serde(rename = "fork")]
	Fork,
}

/// Fired when a session is started, loaded, or reloaded (`SessionStartEvent`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "session_start", rename_all = "camelCase")]
pub struct SessionStartEvent {
	pub reason: SessionStartReason,
	/// Previously active session file; present for new/resume/fork.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub previous_session_file: Option<String>,
}

/// Fired when the current session metadata changes (`SessionInfoChangedEvent`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "session_info_changed", rename_all = "camelCase")]
pub struct SessionInfoChangedEvent {
	/// Current normalized session name; `None` when cleared.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub name: Option<String>,
}

/// Why a session switch was requested (`SessionBeforeSwitchEvent.reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionSwitchReason {
	#[serde(rename = "new")]
	New,
	#[serde(rename = "resume")]
	Resume,
}

/// Fired before switching to another session; can be cancelled
/// (`SessionBeforeSwitchEvent`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "session_before_switch", rename_all = "camelCase")]
pub struct SessionBeforeSwitchEvent {
	pub reason: SessionSwitchReason,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub target_session_file: Option<String>,
}

/// Fired before forking a session; can be cancelled (`SessionBeforeForkEvent`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "session_before_fork", rename_all = "camelCase")]
pub struct SessionBeforeForkEvent {
	pub entry_id: String,
	pub position: ForkPosition,
}

/// What triggered a compaction (`reason` on compact events).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactReason {
	#[serde(rename = "manual")]
	Manual,
	#[serde(rename = "threshold")]
	Threshold,
	#[serde(rename = "overflow")]
	Overflow,
}

/// Fired before context compaction; can be cancelled or customized
/// (`SessionBeforeCompactEvent`). Plain struct: carries the live `AbortSignal`
/// and the non-serde [`CompactionPreparation`] (D30).
#[derive(Debug, Clone)]
pub struct SessionBeforeCompactEvent {
	pub preparation: CompactionPreparation,
	pub branch_entries: Vec<SessionEntry>,
	/// What triggered the compaction: manual `/compact`, the context
	/// threshold, or context overflow recovery.
	pub custom_instructions: Option<String>,
	pub reason: CompactReason,
	/// True when the aborted turn is retried after this compaction.
	pub will_retry: bool,
	pub signal: AbortSignal,
}

/// Fired after context compaction succeeds (`SessionCompactEvent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "session_compact", rename_all = "camelCase")]
pub struct SessionCompactEvent {
	pub compaction_entry: CompactionEntry,
	#[serde(rename = "fromExtension", default)]
	pub from_extension: bool,
	pub reason: CompactReason,
	#[serde(rename = "willRetry", default)]
	pub will_retry: bool,
}

/// Fired after context compaction fails or is aborted
/// (`SessionCompactFailedEvent`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "session_compact_failed", rename_all = "camelCase")]
pub struct SessionCompactFailedEvent {
	pub reason: CompactReason,
	/// Error text when compaction failed for a non-abort reason.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub error_message: Option<String>,
	/// True when compaction was cancelled or aborted.
	#[serde(default)]
	pub aborted: bool,
	#[serde(rename = "willRetry", default)]
	pub will_retry: bool,
	/// True when the failing compaction content came from a
	/// `session_before_compact` handler.
	#[serde(rename = "fromExtension", default)]
	pub from_extension: bool,
}

/// Why an extension runtime is torn down (`SessionShutdownEvent.reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionShutdownReason {
	#[serde(rename = "quit")]
	Quit,
	#[serde(rename = "reload")]
	Reload,
	#[serde(rename = "new")]
	New,
	#[serde(rename = "resume")]
	Resume,
	#[serde(rename = "fork")]
	Fork,
}

/// Fired before an extension runtime is torn down due to quit, reload, or
/// session replacement (`SessionShutdownEvent`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "session_shutdown", rename_all = "camelCase")]
pub struct SessionShutdownEvent {
	pub reason: SessionShutdownReason,
	/// Destination session file when shutting down due to session replacement.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub target_session_file: Option<String>,
}

/// Preparation data for tree navigation (`TreePreparation`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TreePreparation {
	pub target_id: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub old_leaf_id: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub common_ancestor_id: Option<String>,
	pub entries_to_summarize: Vec<SessionEntry>,
	#[serde(rename = "userWantsSummary", default)]
	pub user_wants_summary: bool,
	/// Custom instructions for summarization.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub custom_instructions: Option<String>,
	/// If true, `customInstructions` replaces the default prompt instead of
	/// being appended.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub replace_instructions: Option<bool>,
	/// Label to attach to the branch summary entry.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub label: Option<String>,
}

/// Fired before navigating in the session tree; can be cancelled
/// (`SessionBeforeTreeEvent`). Plain struct: carries the live `AbortSignal`
/// (D30).
#[derive(Debug, Clone)]
pub struct SessionBeforeTreeEvent {
	pub preparation: TreePreparation,
	pub signal: AbortSignal,
}

/// Fired after navigating in the session tree (`SessionTreeEvent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "session_tree", rename_all = "camelCase")]
pub struct SessionTreeEvent {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub new_leaf_id: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub old_leaf_id: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub summary_entry: Option<BranchSummaryEntry>,
	#[serde(rename = "fromExtension", default, skip_serializing_if = "Option::is_none")]
	pub from_extension: Option<bool>,
}

/// Session lifecycle events (`SessionEvent` union); discriminated by the
/// `type` tag. Plain enum: two members carry live `AbortSignal`s, so the
/// union itself is not serde-derived (D30) — the bridge dispatches on member
/// payloads, each of which serializes on its own where wire-representable.
#[derive(Debug, Clone)]
pub enum SessionEvent {
	Start(SessionStartEvent),
	InfoChanged(SessionInfoChangedEvent),
	BeforeSwitch(SessionBeforeSwitchEvent),
	BeforeFork(SessionBeforeForkEvent),
	BeforeCompact(SessionBeforeCompactEvent),
	Compact(SessionCompactEvent),
	CompactFailed(SessionCompactFailedEvent),
	Shutdown(SessionShutdownEvent),
	BeforeTree(SessionBeforeTreeEvent),
	Tree(SessionTreeEvent),
}

// ============================================================================
// Agent Events
// ============================================================================

/// Fired before each LLM call; can modify messages (`ContextEvent`). The
/// prompt and tool state belong to Pi and are restored after handlers run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "context", rename_all = "camelCase")]
pub struct ContextEvent {
	/// The conversation without system messages.
	pub messages: Vec<AgentMessage>,
}

/// Fired before each LLM call, after every `context` handler has run and Pi
/// has restored prompt and tool state (`ContextWithSystemEvent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "context_with_system", rename_all = "camelCase")]
pub struct ContextWithSystemEvent {
	/// Full transcript including system messages; sent as returned.
	pub messages: Vec<AgentMessage>,
}

/// Fired before a provider request is sent; can replace the payload
/// (`BeforeProviderRequestEvent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "before_provider_request", rename_all = "camelCase")]
pub struct BeforeProviderRequestEvent {
	pub payload: JsonValue,
}

/// Fired after request headers are assembled, before the provider HTTP call
/// (`BeforeProviderHeadersEvent`). Handlers mutate `headers` in place; a
/// `None` value deletes that header.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "before_provider_headers", rename_all = "camelCase")]
pub struct BeforeProviderHeadersEvent {
	pub headers: ProviderHeaders,
}

/// Fired after a provider response is received and before the response stream
/// is consumed (`AfterProviderResponseEvent`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "after_provider_response", rename_all = "camelCase")]
pub struct AfterProviderResponseEvent {
	pub status: u16,
	pub headers: BTreeMap<String, String>,
}

/// Fired for a parsed provider stream event before Pi normalizes it
/// (`ProviderStreamEvent`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "provider_stream_event", rename_all = "camelCase")]
pub struct ProviderStreamEvent {
	/// Owning provider id.
	pub provider: String,
	/// API id (upstream `Api`, a string union — `pi-ai` D6).
	pub api: String,
	pub model: String,
	pub data: JsonValue,
}

/// Fired after user submits prompt but before agent loop
/// (`BeforeAgentStartEvent`). Plain struct: `systemPromptOptions` is the
/// non-serde [`NormalizedBuildSystemPromptOptions`] (D30).
#[derive(Debug, Clone, PartialEq)]
pub struct BeforeAgentStartEvent {
	/// The raw user prompt text (after expansion).
	pub prompt: String,
	/// Images attached to the user prompt, if any.
	pub images: Option<Vec<ImageContent>>,
	/// The current system prompt, rendered from `systemPromptOptions` and
	/// earlier handler changes.
	pub system_prompt: String,
	/// Mutable prompt sections; later handlers observe earlier mutations.
	pub system_prompt_options: NormalizedBuildSystemPromptOptions,
}

/// Fired when an agent loop starts (`AgentStartEvent`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "agent_start")]
pub struct AgentStartEvent {}

/// Fired when an agent loop ends (`AgentEndEvent`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "agent_end", rename_all = "camelCase")]
pub struct AgentEndEvent {
	pub messages: Vec<AgentMessage>,
}

/// How an agent run ended (`AgentActivityOutcome`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentActivityOutcome {
	#[serde(rename = "completed")]
	Completed,
	#[serde(rename = "aborted")]
	Aborted,
	#[serde(rename = "error")]
	Error,
}

impl Default for AgentActivityOutcome {
	fn default() -> Self {
		AgentActivityOutcome::Completed
	}
}

/// Draft of a custom entry (`CustomEntryDraft`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "custom", rename_all = "camelCase")]
pub struct CustomEntryDraft {
	#[serde(rename = "customType")]
	pub custom_type: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub data: Option<JsonValue>,
}

/// Draft of a custom message entry (`CustomMessageEntryDraft`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "custom_message", rename_all = "camelCase")]
pub struct CustomMessageEntryDraft {
	#[serde(rename = "customType")]
	pub custom_type: String,
	pub content: UserContentField,
	#[serde(default)]
	pub display: bool,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub details: Option<JsonValue>,
}

/// Draft of a context edit entry (`ContextEditEntryDraft`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "context_edit", rename_all = "camelCase")]
pub struct ContextEditEntryDraft {
	pub target_id: String,
	pub replacement: ContextEditReplacement,
}

/// Draft of a compaction entry (`CompactionEntryDraft`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "compaction", rename_all = "camelCase")]
pub struct CompactionEntryDraft {
	pub summary: String,
	/// `None` (JSON `null`) creates a self-retaining compaction that keeps no
	/// preceding entries. Serialized as `null` when absent, never omitted.
	#[serde(rename = "firstKeptEntryId", default)]
	pub first_kept_entry_id: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub details: Option<JsonValue>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub usage: Option<Usage>,
}

/// Entries a boundary handler may append (`SessionBoundaryDraft`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SessionBoundaryDraft {
	#[serde(rename = "custom")]
	Custom(CustomEntryDraft),
	#[serde(rename = "custom_message")]
	CustomMessage(CustomMessageEntryDraft),
	#[serde(rename = "context_edit")]
	ContextEdit(ContextEditEntryDraft),
	#[serde(rename = "compaction")]
	Compaction(CompactionEntryDraft),
}

/// Preview of the context a boundary decision affects
/// (`BoundaryContextPreview`). Plain struct: [`ProjectedSessionEntry`] carries
/// no serde (D30).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BoundaryContextPreview {
	pub context_entries: Vec<ProjectedSessionEntry>,
	pub context_messages: Vec<AgentMessage>,
	pub llm_messages: Vec<Message>,
	pub pending_messages: Vec<AgentMessage>,
	pub can_continue: bool,
}

/// Shared state of turn/settle boundary events (`BoundaryState`). Plain
/// struct — see [`BoundaryContextPreview`] (D30).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BoundaryState {
	pub entries: Vec<SessionBoundaryDraft>,
	/// Upstream `continue` (Rust keyword — suffixed field name).
	pub continue_: bool,
	pub context: BoundaryContextPreview,
	pub outcome: AgentActivityOutcome,
}

/// What a boundary handler may change (`BoundaryResult`). Omitted fields stay
/// as they are.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundaryResult {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub entries: Option<Vec<SessionBoundaryDraft>>,
	#[serde(rename = "continue", default, skip_serializing_if = "Option::is_none")]
	pub continue_: Option<bool>,
}

/// Fired before final settlement; may append entries and ensure one next
/// provider request (`AgentBeforeSettleEvent`, extends `BoundaryState`). Plain
/// struct — see [`BoundaryState`] (D30).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentBeforeSettleEvent {
	pub boundary: BoundaryState,
}

/// Fired after an agent run has fully settled with no automatic retry,
/// compaction, or queued continuation pending (`AgentSettledEvent`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "agent_settled")]
pub struct AgentSettledEvent {}

/// Kind of blocking UI prompt (`UIPromptKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UIPromptKind {
	#[serde(rename = "select")]
	Select,
	#[serde(rename = "confirm")]
	Confirm,
	#[serde(rename = "input")]
	Input,
	#[serde(rename = "editor")]
	Editor,
	#[serde(rename = "custom")]
	Custom,
}

/// The constant `reason: "ui_prompt"` member of the UI prompt events.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum UIPromptReason {
	#[serde(rename = "ui_prompt")]
	#[default]
	UiPrompt,
}

/// Fired when Pi starts waiting on a blocking user-facing extension UI prompt
/// (`UIPromptStartEvent`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "ui_prompt_start", rename_all = "camelCase")]
pub struct UIPromptStartEvent {
	#[serde(default)]
	pub reason: UIPromptReason,
	pub kind: UIPromptKind,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub title: Option<String>,
}

/// Fired when Pi is no longer waiting on a blocking UI prompt
/// (`UIPromptEndEvent`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "ui_prompt_end", rename_all = "camelCase")]
pub struct UIPromptEndEvent {
	#[serde(default)]
	pub reason: UIPromptReason,
	pub kind: UIPromptKind,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub title: Option<String>,
}

/// Fired at the start of each turn (`TurnStartEvent`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "turn_start", rename_all = "camelCase")]
pub struct TurnStartEvent {
	pub turn_index: u32,
	/// Unix-epoch milliseconds (crate docs D2).
	pub timestamp: i64,
}

/// Fired at the end of each turn (`TurnEndEvent`, extends `BoundaryState`).
/// Plain struct — see [`BoundaryState`] (D30).
#[derive(Debug, Clone, PartialEq)]
pub struct TurnEndEvent {
	pub boundary: BoundaryState,
	pub turn_index: u32,
	pub message: AgentMessage,
	pub tool_results: Vec<ToolResultMessage>,
	pub message_entry_id: String,
	pub tool_result_entry_ids: Vec<String>,
}

/// Fired when a message starts (`MessageStartEvent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "message_start", rename_all = "camelCase")]
pub struct MessageStartEvent {
	pub message: AgentMessage,
}

/// Fired during assistant message streaming (`MessageUpdateEvent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "message_update", rename_all = "camelCase")]
pub struct MessageUpdateEvent {
	pub message: AgentMessage,
	#[serde(rename = "assistantMessageEvent")]
	pub assistant_message_event: AssistantMessageEvent,
}

/// Fired when a message ends (`MessageEndEvent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "message_end", rename_all = "camelCase")]
pub struct MessageEndEvent {
	pub message: AgentMessage,
}

/// Fired when a tool starts executing (`ToolExecutionStartEvent`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "tool_execution_start", rename_all = "camelCase")]
pub struct ToolExecutionStartEvent {
	pub tool_call_id: String,
	pub tool_name: String,
	pub args: JsonValue,
	/// Set when another tool (e.g. a codemode script) made this call.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub parent_tool_call_id: Option<String>,
}

/// Fired during tool execution with partial/streaming output
/// (`ToolExecutionUpdateEvent`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "tool_execution_update", rename_all = "camelCase")]
pub struct ToolExecutionUpdateEvent {
	pub tool_call_id: String,
	pub tool_name: String,
	pub args: JsonValue,
	pub partial_result: JsonValue,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub parent_tool_call_id: Option<String>,
}

/// Fired when a tool finishes executing (`ToolExecutionEndEvent`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "tool_execution_end", rename_all = "camelCase")]
pub struct ToolExecutionEndEvent {
	pub tool_call_id: String,
	pub tool_name: String,
	pub result: JsonValue,
	#[serde(rename = "isError", default)]
	pub is_error: bool,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub parent_tool_call_id: Option<String>,
}

// ============================================================================
// Model Events
// ============================================================================

/// Why a model was selected (`ModelSelectSource`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSelectSource {
	#[serde(rename = "set")]
	Set,
	#[serde(rename = "cycle")]
	Cycle,
	#[serde(rename = "restore")]
	Restore,
}

/// Fired when a new model is selected (`ModelSelectEvent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "model_select", rename_all = "camelCase")]
pub struct ModelSelectEvent {
	pub model: Model,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub previous_model: Option<Model>,
	pub source: ModelSelectSource,
}

/// Fired when a new thinking level is selected (`ThinkingLevelSelectEvent`).
/// Serialize only: [`AgentThinkingLevel`] has no `Deserialize` upstream-side
/// (pi-agent D6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename = "thinking_level_select", rename_all = "camelCase")]
pub struct ThinkingLevelSelectEvent {
	#[serde(rename = "level")]
	pub level: AgentThinkingLevel,
	#[serde(rename = "previousLevel")]
	pub previous_level: AgentThinkingLevel,
}

// ============================================================================
// User Bash Events
// ============================================================================

/// Fired when user executes a bash command via `!` or `!!` prefix
/// (`UserBashEvent`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "user_bash", rename_all = "camelCase")]
pub struct UserBashEvent {
	/// The command to execute.
	pub command: String,
	/// True if the `!!` prefix was used (excluded from LLM context).
	#[serde(rename = "excludeFromContext", default)]
	pub exclude_from_context: bool,
	/// Current working directory.
	pub cwd: String,
}

// ============================================================================
// Input Events
// ============================================================================

/// Source of user input (`InputSource`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputSource {
	#[serde(rename = "interactive")]
	Interactive,
	#[serde(rename = "rpc")]
	Rpc,
	#[serde(rename = "extension")]
	Extension,
}

/// How input is delivered during streaming
/// (`InputEvent.streamingBehavior`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamingBehavior {
	#[serde(rename = "steer")]
	Steer,
	#[serde(rename = "followUp")]
	FollowUp,
}

/// Fired when user input is received, before agent processing (`InputEvent`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "input", rename_all = "camelCase")]
pub struct InputEvent {
	/// The input text.
	pub text: String,
	/// Attached images, if any.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub images: Option<Vec<ImageContent>>,
	/// Where the input came from.
	pub source: InputSource,
	/// How the input will be delivered during streaming, or `None` when idle.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub streaming_behavior: Option<StreamingBehavior>,
}

/// Result from an input event handler (`InputEventResult`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "camelCase")]
pub enum InputEventResult {
	#[default]
	#[serde(rename = "continue")]
	Continue,
	#[serde(rename = "transform")]
	Transform {
		text: String,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		images: Option<Vec<ImageContent>>,
	},
	#[serde(rename = "handled")]
	Handled,
}

// ============================================================================
// Tool Events
// ============================================================================

/// Per-tool `tool_call` event variants (`ToolCallEvent`). All variants carry
/// the same wire shape:
/// `{ type: "tool_call", toolCallId, parentToolCallId?, toolName, input }`.
///
/// The typed `input` payloads (upstream `BashToolInput`, `ReadToolInput`, …
/// from `core/tools/*.ts`) are erased to JSON until the `core/tools` port
/// lands (D30); the guards only discriminate by `toolName`, matching
/// upstream's runtime guards.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolCallEvent {
	Bash(ToolCallEventBase),
	PowerShell(ToolCallEventBase),
	Read(ToolCallEventBase),
	Edit(ToolCallEventBase),
	Write(ToolCallEventBase),
	Grep(ToolCallEventBase),
	Find(ToolCallEventBase),
	Ls(ToolCallEventBase),
	/// Any non-builtin tool (`CustomToolCallEvent`).
	Custom(ToolCallEventBase),
}

/// Base payload of every [`ToolCallEvent`] variant (`ToolCallEventBase` plus
/// `toolName`/`input`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCallEventBase {
	/// The call's id. For nested calls (`parentToolCallId` set) pi assigns
	/// `<parent id>/<n>`; such ids never appear in the transcript.
	pub tool_call_id: String,
	/// Set when another tool (e.g. a codemode script) issued this call.
	pub parent_tool_call_id: Option<String>,
	/// The tool name (`"bash"`, `"read"`, … or the custom tool's name).
	pub tool_name: String,
	/// Tool call arguments. Mutable in upstream handlers: mutate in place to
	/// patch arguments before execution (the Rust bridge carries patches back
	/// in the handler result, like pi-agent D8).
	pub input: JsonValue,
}

impl ToolCallEvent {
	/// The variant's `toolName`.
	pub fn tool_name(&self) -> &str {
		match self {
			ToolCallEvent::Bash(event) => &event.tool_name,
			ToolCallEvent::PowerShell(event) => &event.tool_name,
			ToolCallEvent::Read(event) => &event.tool_name,
			ToolCallEvent::Edit(event) => &event.tool_name,
			ToolCallEvent::Write(event) => &event.tool_name,
			ToolCallEvent::Grep(event) => &event.tool_name,
			ToolCallEvent::Find(event) => &event.tool_name,
			ToolCallEvent::Ls(event) => &event.tool_name,
			ToolCallEvent::Custom(event) => &event.tool_name,
		}
	}

	/// The base payload.
	pub fn base(&self) -> &ToolCallEventBase {
		match self {
			ToolCallEvent::Bash(event)
			| ToolCallEvent::PowerShell(event)
			| ToolCallEvent::Read(event)
			| ToolCallEvent::Edit(event)
			| ToolCallEvent::Write(event)
			| ToolCallEvent::Grep(event)
			| ToolCallEvent::Find(event)
			| ToolCallEvent::Ls(event)
			| ToolCallEvent::Custom(event) => event,
		}
	}

	fn base_mut(&mut self) -> &mut ToolCallEventBase {
		match self {
			ToolCallEvent::Bash(event)
			| ToolCallEvent::PowerShell(event)
			| ToolCallEvent::Read(event)
			| ToolCallEvent::Edit(event)
			| ToolCallEvent::Write(event)
			| ToolCallEvent::Grep(event)
			| ToolCallEvent::Find(event)
			| ToolCallEvent::Ls(event)
			| ToolCallEvent::Custom(event) => event,
		}
	}

	/// Patch the call arguments (runner port D40; upstream handlers mutate
	/// the shared `input` in place).
	pub fn set_input(&mut self, input: JsonValue) {
		self.base_mut().input = input;
	}

	/// A `bash` tool call event.
	pub fn bash(tool_call_id: impl Into<String>, input: JsonValue) -> Self {
		ToolCallEvent::Bash(ToolCallEventBase {
			tool_call_id: tool_call_id.into(),
			parent_tool_call_id: None,
			tool_name: "bash".to_string(),
			input,
		})
	}

	/// A custom (non-builtin) tool call event.
	pub fn custom(tool_call_id: impl Into<String>, tool_name: impl Into<String>, input: JsonValue) -> Self {
		ToolCallEvent::Custom(ToolCallEventBase {
			tool_call_id: tool_call_id.into(),
			parent_tool_call_id: None,
			tool_name: tool_name.into(),
			input,
		})
	}
}

/// Wire shape shared by all [`ToolCallEvent`] variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolCallEventWire {
	#[serde(rename = "type")]
	event_type: ToolCallEventTypeTag,
	pub tool_call_id: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub parent_tool_call_id: Option<String>,
	pub tool_name: String,
	pub input: JsonValue,
}

/// The constant `"tool_call"` type tag.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCallEventTypeTag {
	#[serde(rename = "tool_call")]
	#[default]
	ToolCall,
}

impl Serialize for ToolCallEvent {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		let base = self.base();
		ToolCallEventWire {
			event_type: ToolCallEventTypeTag::ToolCall,
			tool_call_id: base.tool_call_id.clone(),
			parent_tool_call_id: base.parent_tool_call_id.clone(),
			tool_name: base.tool_name.clone(),
			input: base.input.clone(),
		}
		.serialize(serializer)
	}
}

/// Per-tool `tool_result` event variants (`ToolResultEvent`). Wire shape:
/// `{ type: "tool_result", toolCallId, parentToolCallId?, input, content,
/// structuredContent?, isError, usage?, toolName, details }`.
///
/// The typed `details` payloads (upstream `BashToolDetails`, `EditToolDetails`,
/// … from `core/tools/*.ts`) are erased to JSON until the `core/tools` port
/// lands (D30).
#[derive(Debug, Clone, PartialEq)]
pub enum ToolResultEvent {
	Bash(ToolResultEventBase),
	PowerShell(ToolResultEventBase),
	Read(ToolResultEventBase),
	Edit(ToolResultEventBase),
	Write(ToolResultEventBase),
	Grep(ToolResultEventBase),
	Find(ToolResultEventBase),
	Ls(ToolResultEventBase),
	/// Any non-builtin tool (`CustomToolResultEvent`).
	Custom(ToolResultEventBase),
}

/// Base payload of every [`ToolResultEvent`] variant (`ToolResultEventBase`
/// plus `toolName`/`details`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolResultEventBase {
	/// The call's id; `<parent id>/<n>` for nested calls.
	pub tool_call_id: String,
	/// Set when another tool (e.g. a codemode script) issued this call.
	pub parent_tool_call_id: Option<String>,
	pub tool_name: String,
	pub input: JsonValue,
	/// Model-facing content.
	pub content: Vec<pi_ai::types::UserContent>,
	/// Machine-readable result for tools that declare an `outputSchema`.
	/// Handlers that redact `content` should also replace this; replacing
	/// `content` alone drops it.
	pub structured_content: Option<JsonValue>,
	pub is_error: bool,
	/// Usage from the tool execution itself, if available.
	pub usage: Option<Usage>,
	/// Per-tool details (erased JSON; D30).
	pub details: JsonValue,
}

impl ToolResultEvent {
	/// The variant's `toolName`.
	pub fn tool_name(&self) -> &str {
		match self {
			ToolResultEvent::Bash(event) => &event.tool_name,
			ToolResultEvent::PowerShell(event) => &event.tool_name,
			ToolResultEvent::Read(event) => &event.tool_name,
			ToolResultEvent::Edit(event) => &event.tool_name,
			ToolResultEvent::Write(event) => &event.tool_name,
			ToolResultEvent::Grep(event) => &event.tool_name,
			ToolResultEvent::Find(event) => &event.tool_name,
			ToolResultEvent::Ls(event) => &event.tool_name,
			ToolResultEvent::Custom(event) => &event.tool_name,
		}
	}

	/// The base payload.
	pub fn base(&self) -> &ToolResultEventBase {
		match self {
			ToolResultEvent::Bash(event)
			| ToolResultEvent::PowerShell(event)
			| ToolResultEvent::Read(event)
			| ToolResultEvent::Edit(event)
			| ToolResultEvent::Write(event)
			| ToolResultEvent::Grep(event)
			| ToolResultEvent::Find(event)
			| ToolResultEvent::Ls(event)
			| ToolResultEvent::Custom(event) => event,
		}
	}

	fn base_mut(&mut self) -> &mut ToolResultEventBase {
		match self {
			ToolResultEvent::Bash(event)
			| ToolResultEvent::PowerShell(event)
			| ToolResultEvent::Read(event)
			| ToolResultEvent::Edit(event)
			| ToolResultEvent::Write(event)
			| ToolResultEvent::Grep(event)
			| ToolResultEvent::Find(event)
			| ToolResultEvent::Ls(event)
			| ToolResultEvent::Custom(event) => event,
		}
	}

	/// Result content (`tool_result` chaining; runner port D40).
	pub fn content(&self) -> &[pi_ai::types::UserContent] {
		&self.base().content
	}

	pub fn set_content(&mut self, content: Vec<pi_ai::types::UserContent>) {
		self.base_mut().content = content;
	}

	pub fn structured_content(&self) -> Option<&JsonValue> {
		self.base().structured_content.as_ref()
	}

	pub fn set_structured_content(&mut self, structured_content: Option<JsonValue>) {
		self.base_mut().structured_content = structured_content;
	}

	/// Clears structured content (replacing `content` without a replacement
	/// drops it — upstream semantics).
	pub fn clear_structured_content(&mut self) {
		self.base_mut().structured_content = None;
	}

	pub fn is_error(&self) -> bool {
		self.base().is_error
	}

	pub fn set_is_error(&mut self, is_error: bool) {
		self.base_mut().is_error = is_error;
	}

	pub fn usage(&self) -> Option<&Usage> {
		self.base().usage.as_ref()
	}

	pub fn set_usage(&mut self, usage: Option<Usage>) {
		self.base_mut().usage = usage;
	}

	pub fn details(&self) -> &JsonValue {
		&self.base().details
	}

	pub fn set_details(&mut self, details: JsonValue) {
		self.base_mut().details = details;
	}
}

/// Wire shape shared by all [`ToolResultEvent`] variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolResultEventWire {
	#[serde(rename = "type")]
	event_type: ToolResultEventTypeTag,
	pub tool_call_id: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub parent_tool_call_id: Option<String>,
	pub tool_name: String,
	pub input: JsonValue,
	pub content: Vec<pi_ai::types::UserContent>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub structured_content: Option<JsonValue>,
	#[serde(rename = "isError", default)]
	pub is_error: bool,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub usage: Option<Usage>,
	/// Absent mirrors upstream `details: undefined` (`write` always).
	#[serde(default, skip_serializing_if = "JsonValue::is_null")]
	pub details: JsonValue,
}

/// The constant `"tool_result"` type tag.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolResultEventTypeTag {
	#[serde(rename = "tool_result")]
	#[default]
	ToolResult,
}

impl Serialize for ToolResultEvent {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		let base = self.base();
		ToolResultEventWire {
			event_type: ToolResultEventTypeTag::ToolResult,
			tool_call_id: base.tool_call_id.clone(),
			parent_tool_call_id: base.parent_tool_call_id.clone(),
			tool_name: base.tool_name.clone(),
			input: base.input.clone(),
			content: base.content.clone(),
			structured_content: base.structured_content.clone(),
			is_error: base.is_error,
			usage: base.usage.clone(),
			details: base.details.clone(),
		}
		.serialize(serializer)
	}
}

/// Whether `event` is the `bash` tool result (`isBashToolResult`). The TS
/// type guards are runtime tool-name checks; the Rust predicates mirror that
/// exactly (a custom event named `bash` passes, like upstream).
pub fn is_bash_tool_result(event: &ToolResultEvent) -> bool {
	event.tool_name() == "bash"
}

/// Whether `event` is the `powershell` tool result (`isPowerShellToolResult`).
pub fn is_power_shell_tool_result(event: &ToolResultEvent) -> bool {
	event.tool_name() == "powershell"
}

/// Whether `event` is the `read` tool result (`isReadToolResult`).
pub fn is_read_tool_result(event: &ToolResultEvent) -> bool {
	event.tool_name() == "read"
}

/// Whether `event` is the `edit` tool result (`isEditToolResult`).
pub fn is_edit_tool_result(event: &ToolResultEvent) -> bool {
	event.tool_name() == "edit"
}

/// Whether `event` is the `write` tool result (`isWriteToolResult`).
pub fn is_write_tool_result(event: &ToolResultEvent) -> bool {
	event.tool_name() == "write"
}

/// Whether `event` is the `grep` tool result (`isGrepToolResult`).
pub fn is_grep_tool_result(event: &ToolResultEvent) -> bool {
	event.tool_name() == "grep"
}

/// Whether `event` is the `find` tool result (`isFindToolResult`).
pub fn is_find_tool_result(event: &ToolResultEvent) -> bool {
	event.tool_name() == "find"
}

/// Whether `event` is the `ls` tool result (`isLsToolResult`).
pub fn is_ls_tool_result(event: &ToolResultEvent) -> bool {
	event.tool_name() == "ls"
}

/// Type guard for narrowing `ToolCallEvent` by tool name
/// (`isToolCallEventType`). The TS overload set collapses to one runtime
/// name check; use [`ToolCallEvent::tool_name`] plus a `match` for the typed
/// narrowing the TS declarations provide statically.
pub fn is_tool_call_event_type(tool_name: &str, event: &ToolCallEvent) -> bool {
	event.tool_name() == tool_name
}

/// Union of all extension event types (`ExtensionEvent`). Upstream this is a
/// type-level union used for handler typing; the Rust enum mirrors it for
/// dispatch code. Not serde-derived and not `PartialEq`: members carry live
/// handles and the tool unions do not deserialize (D30).
#[derive(Debug, Clone)]
pub enum ExtensionEvent {
	ProjectTrust(ProjectTrustEvent),
	ResourcesDiscover(ResourcesDiscoverEvent),
	McpServersChange(McpServersChangeEvent),
	Session(SessionEvent),
	Context(ContextEvent),
	ContextWithSystem(ContextWithSystemEvent),
	CacheWarmingDecision(CacheWarmingDecisionEvent),
	BeforeProviderRequest(BeforeProviderRequestEvent),
	BeforeProviderHeaders(BeforeProviderHeadersEvent),
	AfterProviderResponse(AfterProviderResponseEvent),
	ProviderStream(ProviderStreamEvent),
	BeforeAgentStart(BeforeAgentStartEvent),
	AgentStart(AgentStartEvent),
	AgentEnd(AgentEndEvent),
	AgentBeforeSettle(AgentBeforeSettleEvent),
	AgentSettled(AgentSettledEvent),
	UIPromptStart(UIPromptStartEvent),
	UIPromptEnd(UIPromptEndEvent),
	TurnStart(TurnStartEvent),
	TurnEnd(TurnEndEvent),
	MessageStart(MessageStartEvent),
	MessageUpdate(MessageUpdateEvent),
	MessageEnd(MessageEndEvent),
	ToolExecutionStart(ToolExecutionStartEvent),
	ToolExecutionUpdate(ToolExecutionUpdateEvent),
	ToolExecutionEnd(ToolExecutionEndEvent),
	ModelSelect(ModelSelectEvent),
	ThinkingLevelSelect(ThinkingLevelSelectEvent),
	UserBash(UserBashEvent),
	Input(InputEvent),
	ToolCall(ToolCallEvent),
	ToolResult(ToolResultEvent),
}

// ============================================================================
// Event Results
// ============================================================================

/// Result from a `context` handler (`ContextEventResult`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextEventResult {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub messages: Option<Vec<AgentMessage>>,
}

/// Result from a `turn_end` handler (`TurnEndEventResult`).
pub type TurnEndEventResult = BoundaryResult;

/// Result from an `agent_before_settle` handler
/// (`AgentBeforeSettleEventResult`).
pub type AgentBeforeSettleEventResult = BoundaryResult;

/// Result from a `before_provider_request` handler
/// (`BeforeProviderRequestEventResult = unknown`).
pub type BeforeProviderRequestEventResult = JsonValue;

/// Result from a `tool_call` handler (`ToolCallEventResult`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallEventResult {
	/// Block tool execution; modify arguments via the handler result's args
	/// patch (upstream mutates `event.input` in place — pi-agent D8 style).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub block: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub reason: Option<String>,
	/// Hint that the agent should stop after the current tool batch when this
	/// call is blocked. Early termination only happens when every finalized
	/// tool result in the batch sets this to true.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub terminate: Option<bool>,
}

/// Result from a `user_bash` handler (`UserBashEventResult`): either custom
/// operations to execute with, or a full replacement result.
pub enum UserBashEventResult {
	/// Custom operations to use for execution.
	Operations(Arc<dyn BashOperations>),
	/// Full replacement: the extension handled execution itself.
	Result(BashResult),
}

/// Changes a `tool_result` handler makes (`ToolResultEventResult`). Omitted
/// fields stay as they are, except that replacing `content` without returning
/// `structuredContent` drops the structured content, because it may no longer
/// match. Return it along with `content` to keep it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultEventResult {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub content: Option<Vec<pi_ai::types::UserContent>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub details: Option<JsonValue>,
	#[serde(rename = "structuredContent", default, skip_serializing_if = "Option::is_none")]
	pub structured_content: Option<JsonValue>,
	#[serde(rename = "isError", default, skip_serializing_if = "Option::is_none")]
	pub is_error: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub usage: Option<Usage>,
}

/// Result from a `message_end` handler (`MessageEndEventResult`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageEndEventResult {
	/// Replace the finalized message; must keep the original role.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub message: Option<AgentMessage>,
}

/// Result from a `before_agent_start` handler (`BeforeAgentStartEventResult`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BeforeAgentStartEventResult {
	/// Custom message entry appended before the run.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub message: Option<CustomMessageDraft>,
	/// Replace the complete system prompt for this turn; later handlers
	/// observe this exact override.
	#[serde(rename = "systemPrompt", default, skip_serializing_if = "Option::is_none")]
	pub system_prompt: Option<String>,
}

/// Result from a `session_before_switch` handler (`SessionBeforeSwitchResult`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionBeforeSwitchResult {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cancel: Option<bool>,
}

/// Result from a `session_before_fork` handler (`SessionBeforeForkResult`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionBeforeForkResult {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cancel: Option<bool>,
	#[serde(rename = "skipConversationRestore", default, skip_serializing_if = "Option::is_none")]
	pub skip_conversation_restore: Option<bool>,
}

/// Result from a `session_before_compact` handler
/// (`SessionBeforeCompactResult`). Plain struct: [`CompactionResult`] carries
/// no serde (D30).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionBeforeCompactResult {
	pub cancel: Option<bool>,
	pub compaction: Option<CompactionResult>,
}

/// Branch summary override of [`SessionBeforeTreeResult`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TreeSummaryOverride {
	pub summary: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub details: Option<JsonValue>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub usage: Option<Usage>,
}

/// Result from a `session_before_tree` handler (`SessionBeforeTreeResult`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionBeforeTreeResult {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cancel: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub summary: Option<TreeSummaryOverride>,
	/// Override custom instructions for summarization.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub custom_instructions: Option<String>,
	/// Override whether `customInstructions` replaces the default prompt.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub replace_instructions: Option<bool>,
	/// Override label to attach to the branch summary entry.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub label: Option<String>,
}

// ============================================================================
// Message and Entry Rendering
// ============================================================================

/// Options for message renderers (`MessageRenderOptions`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageRenderOptions {
	pub expanded: bool,
	/// Horizontal padding configured by the `outputPad` setting.
	pub output_pad: u32,
}

/// Message type of a markdown transform (`MarkdownTransformContext.messageType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarkdownTransformMessageType {
	#[serde(rename = "user")]
	User,
	#[serde(rename = "assistant")]
	Assistant,
	#[serde(rename = "assistant-thinking")]
	AssistantThinking,
}

/// Context of a markdown transform (`MarkdownTransformContext`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MarkdownTransformContext {
	#[serde(rename = "messageType")]
	pub message_type: MarkdownTransformMessageType,
	#[serde(rename = "isStreaming", default)]
	pub is_streaming: bool,
	#[serde(rename = "availableWidth")]
	pub available_width: u32,
}

/// Transformer for user/assistant markdown before rendering
/// (`MarkdownTransformer`).
pub type MarkdownTransformer = Arc<dyn Fn(&str, &MarkdownTransformContext) -> String + Send + Sync>;

/// Options for entry renderers (`EntryRenderOptions`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryRenderOptions {
	pub expanded: bool,
}

/// Custom renderer for `CustomMessageEntry` (`MessageRenderer<T>`); the
/// details payload `T` is erased in the ported [`CustomMessage`] (D27).
pub type MessageRenderer =
	Arc<dyn Fn(&CustomMessage, &MessageRenderOptions, &Theme) -> Option<Component> + Send + Sync>;

/// Custom renderer for `CustomEntry` (`EntryRenderer<T>`); the data payload
/// `T` is erased in the ported [`CustomEntry`] (D27).
pub type EntryRenderer = Arc<dyn Fn(&CustomEntry, &EntryRenderOptions, &Theme) -> Option<Component> + Send + Sync>;

// ============================================================================
// Command Registration
// ============================================================================

/// `getArgumentCompletions` callback of a registered command.
pub type ArgumentCompletionsFn =
	Arc<dyn Fn(&str) -> BoxFuture<'static, Option<Vec<AutocompleteItem>>> + Send + Sync>;

/// Handler of a registered command.
pub type CommandHandler = Arc<dyn Fn(String, ExtensionCommandContext) -> BoxFuture<'static, ()> + Send + Sync>;

/// A registered custom command (`RegisteredCommand`).
#[derive(Clone)]
pub struct RegisteredCommand {
	pub name: String,
	pub source_info: SourceInfo,
	pub description: Option<String>,
	pub get_argument_completions: Option<ArgumentCompletionsFn>,
	pub handler: CommandHandler,
}

impl std::fmt::Debug for RegisteredCommand {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("RegisteredCommand")
			.field("name", &self.name)
			.field("description", &self.description)
			.finish_non_exhaustive()
	}
}

/// Registration options (upstream `Omit<RegisteredCommand, "name" |
/// "sourceInfo">`).
#[derive(Clone)]
pub struct RegisteredCommandOptions {
	pub description: Option<String>,
	pub get_argument_completions: Option<ArgumentCompletionsFn>,
	pub handler: CommandHandler,
}

impl std::fmt::Debug for RegisteredCommandOptions {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("RegisteredCommandOptions")
			.field("description", &self.description)
			.finish_non_exhaustive()
	}
}

/// A command resolved against an invocation name (`ResolvedCommand`).
#[derive(Clone, Debug)]
pub struct ResolvedCommand {
	/// The registered command.
	pub command: RegisteredCommand,
	/// The name actually invoked (e.g. including a prefix).
	pub invocation_name: String,
}

impl std::ops::Deref for ResolvedCommand {
	type Target = RegisteredCommand;

	fn deref(&self) -> &Self::Target {
		&self.command
	}
}

// ============================================================================
// Extension API
// ============================================================================

/// Handler function type for events (`ExtensionHandler<E, R>`). The upstream
/// `Promise<R | void> | R | void` return becomes `Option<R>`; `None` mirrors
/// `void`/`undefined`.
pub type ExtensionHandler<E, R> = Arc<dyn Fn(E, ExtensionContext) -> BoxFuture<'static, Option<R>> + Send + Sync>;

/// Erased handler stored by the runtime (`HandlerFn`): the event payload is
/// its serialized form and the result its serialized result (or `None` for
/// `void`). Upstream overloads collapse to this single erased form, matching
/// the `Map<string, HandlerFn[]>` storage (D31). Handler failures are an
/// explicit `Err` — upstream throws across the `await` (D41).
pub type HandlerFn = Arc<
	dyn Fn(JsonValue, ExtensionContext) -> BoxFuture<'static, Result<Option<JsonValue>, String>> + Send + Sync,
>;

/// The `pi` object passed to extension factories (`ExtensionAPI`).
///
/// The upstream `on` overload set collapses to one erased subscription
/// keyed by the upstream event-name strings (D31). Callbacks take owned
/// contexts (cheap `Arc` clones) so futures stay `'static`.
pub trait ExtensionApi: Send + Sync {
	// -- Event Subscription --------------------------------------------------

	/// Subscribe to an event by name (e.g. `"session_start"`); returns an
	/// unsubscribe function.
	fn on(&self, event: &str, handler: HandlerFn) -> Unsubscribe;

	// -- Tool Registration ---------------------------------------------------

	/// Register a tool that the LLM can call (`registerTool`).
	fn register_tool(&self, tool: ToolDefinition);

	// -- Command, Shortcut, Flag Registration --------------------------------

	/// Register a custom command (`registerCommand`).
	fn register_command(&self, name: &str, options: RegisteredCommandOptions);

	/// Register a keyboard shortcut (`registerShortcut`).
	fn register_shortcut(&self, shortcut: KeyId, options: ShortcutOptions);

	/// Register a CLI flag (`registerFlag`).
	fn register_flag(&self, name: &str, options: FlagOptions);

	/// Get the value of a registered CLI flag (`getFlag`).
	fn get_flag(&self, name: &str) -> Option<FlagValue>;

	// -- Message Rendering ---------------------------------------------------

	/// Register a custom renderer for `CustomMessageEntry`.
	fn register_message_renderer(&self, custom_type: &str, renderer: MessageRenderer);

	/// Register a transformer for user/assistant markdown before rendering.
	fn register_markdown_transformer(&self, transformer: MarkdownTransformer);

	/// Register a custom renderer for `CustomEntry`; custom entries do not
	/// participate in LLM context.
	fn register_entry_renderer(&self, custom_type: &str, renderer: EntryRenderer);

	/// Choose how tool calls are drawn; resolvers run in extension load order.
	fn register_tool_renderer(&self, resolver: ToolRendererResolver);

	// -- Actions -------------------------------------------------------------

	/// Send a custom message to the session (`sendMessage`).
	fn send_message(&self, message: CustomMessageDraft, options: Option<SendMessageOptions>);

	/// Send a user message to the agent; always triggers a turn
	/// (`sendUserMessage`).
	fn send_user_message(&self, content: SendUserMessageContent, options: Option<SendUserMessageOptions>);

	/// Append a custom entry to the session for state persistence, not sent
	/// to the LLM (`appendEntry`).
	fn append_entry(&self, custom_type: &str, data: Option<JsonValue>);

	// -- Session Metadata ----------------------------------------------------

	/// Set the session display name (`setSessionName`).
	fn set_session_name(&self, name: String);

	/// Get the current session name, if set (`getSessionName`).
	fn get_session_name(&self) -> Option<String>;

	/// Set or clear a label on an entry (`setLabel`).
	fn set_label(&self, entry_id: &str, label: Option<String>);

	/// Execute a shell command (`exec`; upstream resolves always, never
	/// rejects).
	fn exec(&self, command: &str, args: &[String], options: Option<ExecOptions>) -> BoxFuture<'static, ExecResult>;

	/// Names of the active tools — the tools declared to the model
	/// (`getActiveTools`).
	fn get_active_tools(&self) -> Vec<String>;

	/// All configured tools with parameter schema, prompt guidelines,
	/// exposure, and source metadata (`getAllTools`).
	fn get_all_tools(&self) -> Vec<ToolInfo>;

	/// A copy of the effective settings — global and project settings merged,
	/// with overrides (`getSettings`).
	fn get_settings(&self) -> Settings;

	/// Set the active tools by name; unknown and `hidden` tools are ignored
	/// (`setActiveTools`).
	fn set_active_tools(&self, tool_names: Vec<String>);

	/// Available slash commands in the current session (`getCommands`).
	fn get_commands(&self) -> Vec<SlashCommandInfo>;

	// -- Model and Thinking Level ---------------------------------------------

	/// Set the model for the current session without changing the configured
	/// default (`setModel`); `false` when authentication is not configured.
	fn set_model(&self, model: Model) -> BoxFuture<'static, bool>;

	/// Current thinking level (`getThinkingLevel`).
	fn get_thinking_level(&self) -> AgentThinkingLevel;

	/// Set the thinking level (clamped to model capabilities) for the current
	/// session (`setThinkingLevel`).
	fn set_thinking_level(&self, level: AgentThinkingLevel);

	// -- Provider Registration -------------------------------------------------

	/// Register or override a provider from a config
	/// (`registerProvider(name, config)` overload).
	fn register_provider(&self, name: &str, config: ProviderConfig);

	/// Register a native pi-ai provider
	/// (`registerProvider(provider)` overload).
	fn register_native_provider(&self, provider: Arc<dyn AiProvider>);

	/// Unregister a previously registered provider (`unregisterProvider`);
	/// no-op when not registered.
	fn unregister_provider(&self, name: &str);

	// -- MCP Servers ------------------------------------------------------------

	/// Register an MCP server for this session (`registerMcpServer`); the
	/// config mirrors an `mcpServers` entry in `mcp.json`.
	fn register_mcp_server(&self, name: &str, config: McpServerConfig);

	/// Remove an MCP server this extension registered and close its
	/// connection (`unregisterMcpServer`).
	fn unregister_mcp_server(&self, name: &str);

	/// Every MCP server registered by extensions (`getMcpServers`).
	fn get_mcp_servers(&self) -> Vec<RegisteredMcpServer>;

	// -- Virtual Models -----------------------------------------------------------

	/// Register a virtual model (`registerVirtualModel`).
	fn register_virtual_model(&self, model: ExtensionVirtualModel);

	/// Remove a virtual model (`unregisterVirtualModel`).
	fn unregister_virtual_model(&self, provider: &str, id: &str);

	// -- Misc -----------------------------------------------------------------

	/// Shared event bus for extension communication (`events`).
	fn events(&self) -> &dyn EventBus;
}

/// Options of [`ExtensionApi::register_shortcut`].
#[derive(Clone)]
pub struct ShortcutOptions {
	pub description: Option<String>,
	pub handler: Arc<dyn Fn(ExtensionContext) -> BoxFuture<'static, ()> + Send + Sync>,
}

impl std::fmt::Debug for ShortcutOptions {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ShortcutOptions").field("description", &self.description).finish()
	}
}

/// CLI flag declaration (`registerFlag` options union: boolean or string).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlagOptions {
	Boolean {
		description: Option<String>,
		default: Option<bool>,
	},
	String {
		description: Option<String>,
		default: Option<String>,
	},
}

/// A CLI flag value (`boolean | string`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FlagValue {
	Bool(bool),
	Str(String),
}

// ============================================================================
// Provider Registration Types
// ============================================================================

/// Custom `streamSimple` handler (`ProviderConfig.streamSimple`): streams a
/// normalized transcript for a custom API.
pub type StreamSimpleFn =
	Arc<dyn Fn(Model, TranscriptContext, Option<SimpleStreamOptions>) -> AssistantMessageEventStream + Send + Sync>;

/// `refreshModels` callback (`ProviderConfig.refreshModels`); errors mirror
/// upstream rejections.
pub type RefreshModelsFn =
	Arc<dyn Fn(RefreshModelsContext) -> BoxFuture<'static, Result<Vec<ProviderModelConfig>, String>> + Send + Sync>;

/// OAuth callbacks (`login(callbacks)`) — upstream `OAuthLoginCallbacks`,
/// carried by `pi-ai`'s compat types.
pub type OAuthLoginCallbacks = pi_ai::compat::extension_oauth_types::OAuthLoginCallbacks;

/// OAuth provider configuration of [`ProviderConfig`] (`oauth` member). The
/// `id` is set automatically from the provider name.
pub struct ProviderOAuthConfig {
	/// Display name for the provider in login UI.
	pub name: String,
	/// Whether access through this auth method is backed by a subscription.
	pub is_subscription: Option<bool>,
	/// Deprecated upstream; retained for source compatibility.
	pub uses_callback_server: Option<bool>,
	/// Run the login flow; returns credentials to persist.
	pub login: Arc<dyn Fn(OAuthLoginCallbacks) -> BoxFuture<'static, Result<OAuthCredentials, String>> + Send + Sync>,
	/// Refresh expired credentials.
	pub refresh_token:
		Arc<dyn Fn(OAuthCredentials, AbortSignal) -> BoxFuture<'static, Result<OAuthCredentials, String>> + Send + Sync>,
	/// Convert credentials to an API key string for the provider.
	pub get_api_key: Arc<dyn Fn(&OAuthCredentials) -> String + Send + Sync>,
	/// Legacy synchronous credential-dependent model projection.
	pub modify_models: Option<Arc<dyn Fn(Vec<Model>, &OAuthCredentials) -> Vec<Model> + Send + Sync>>,
}

impl std::fmt::Debug for ProviderOAuthConfig {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ProviderOAuthConfig").field("name", &self.name).finish_non_exhaustive()
	}
}

impl Clone for ProviderOAuthConfig {
	fn clone(&self) -> Self {
		ProviderOAuthConfig {
			name: self.name.clone(),
			is_subscription: self.is_subscription,
			uses_callback_server: self.uses_callback_server,
			login: self.login.clone(),
			refresh_token: self.refresh_token.clone(),
			get_api_key: self.get_api_key.clone(),
			modify_models: self.modify_models.clone(),
		}
	}
}

/// Configuration for registering a provider via `registerProvider(name,
/// config)` (`ProviderConfig`). Callback members are host-side only and
/// skipped by serde — the JS bridge adapts JS functions before constructing
/// the struct (D31).
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
	/// Display name for the provider in UI.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub name: Option<String>,
	/// Base URL for the API endpoint; required when defining models.
	#[serde(rename = "baseUrl", default, skip_serializing_if = "Option::is_none")]
	pub base_url: Option<String>,
	/// API key literal, env interpolation (`$ENV_VAR`/`${ENV_VAR}`), or
	/// leading `!command`.
	#[serde(rename = "apiKey", default, skip_serializing_if = "Option::is_none")]
	pub api_key: Option<String>,
	/// API type; required at provider or model level when defining models
	/// (upstream `Api`, a string union — `pi-ai` D6).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub api: Option<String>,
	/// Optional `streamSimple` handler for custom APIs. `#[serde(skip)]` (D31).
	#[serde(skip)]
	pub stream_simple: Option<StreamSimpleFn>,
	/// Image-generation implementations keyed by image API. `#[serde(skip)]`
	/// (D31).
	#[serde(skip)]
	pub images: Option<BTreeMap<String, Arc<dyn ProviderImages>>>,
	/// Classifier implementations keyed by classifier API. `#[serde(skip)]`
	/// (D31).
	#[serde(skip)]
	pub classifiers: Option<BTreeMap<String, Arc<dyn ProviderClassifier>>>,
	/// Custom headers to include in requests.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub headers: Option<BTreeMap<String, String>>,
	/// If true, adds `Authorization: Bearer` with the resolved API key.
	#[serde(rename = "authHeader", default, skip_serializing_if = "Option::is_none")]
	pub auth_header: Option<bool>,
	/// Models to register; replaces all existing models for this provider.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub models: Option<Vec<ProviderModelConfig>>,
	/// Refresh this provider's model list. `#[serde(skip)]` (D31).
	#[serde(skip)]
	pub refresh_models: Option<RefreshModelsFn>,
	/// OAuth provider for `/login` support. `#[serde(skip)]` (D31).
	#[serde(skip)]
	pub oauth: Option<ProviderOAuthConfig>,
}

impl std::fmt::Debug for ProviderConfig {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ProviderConfig")
			.field("name", &self.name)
			.field("base_url", &self.base_url)
			.field("api", &self.api)
			.field("models", &self.models.as_ref().map(Vec::len))
			.finish_non_exhaustive()
	}
}

/// Shared model-config members (`ProviderModelConfigBase`); flattened into
/// each [`ProviderModelConfig`] variant.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderModelConfigBase {
	/// Model ID.
	pub id: String,
	/// Display name.
	pub name: String,
	/// API type override for this model.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub api: Option<String>,
	/// API endpoint URL override for this model.
	#[serde(rename = "baseUrl", default, skip_serializing_if = "Option::is_none")]
	pub base_url: Option<String>,
	/// Supported input types.
	#[serde(default)]
	pub input: Vec<ModelInput>,
	/// Provider input limits and cache-safe image preprocessing metadata.
	#[serde(rename = "inputLimits", default, skip_serializing_if = "Option::is_none")]
	pub input_limits: Option<ModelInputLimits>,
	/// Per-million-token cost rates and optional request-wide pricing tiers.
	#[serde(default)]
	pub cost: Option<ModelCost>,
	/// Custom headers for this model.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub headers: Option<BTreeMap<String, String>>,
}

/// Chat model configuration (`ProviderChatModelConfig`); omitted `type` is
/// normalized to `"chat"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderChatModelConfig {
	#[serde(flatten)]
	pub base: ProviderModelConfigBase,
	/// The optional `"chat"` discriminator.
	#[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
	pub model_type: Option<ChatType>,
	/// API type override.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub api: Option<String>,
	/// Whether the model supports extended thinking.
	#[serde(default)]
	pub reasoning: bool,
	/// Maps pi thinking levels to provider/model-specific values; `None` marks
	/// a level unsupported.
	#[serde(rename = "thinkingLevelMap", default, skip_serializing_if = "Option::is_none")]
	pub thinking_level_map: Option<ThinkingLevelMap>,
	/// Best-effort prompt cache lifetime in seconds per retention tier.
	#[serde(rename = "promptCache", default, skip_serializing_if = "Option::is_none")]
	pub prompt_cache: Option<ModelPromptCache>,
	/// Maximum context window size in tokens.
	#[serde(rename = "contextWindow", default)]
	pub context_window: u64,
	/// Maximum output tokens.
	#[serde(rename = "maxTokens", default)]
	pub max_tokens: u64,
	#[serde(rename = "samplingParams", default, skip_serializing_if = "Option::is_none")]
	pub sampling_params: Option<SamplingParams>,
	/// OpenAI compatibility settings.
	#[serde(rename = "compat", default, skip_serializing_if = "Option::is_none")]
	pub compat: Option<ModelCompat>,
}

/// Image-generation model configuration (`ProviderImageModelConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderImageModelConfig {
	#[serde(flatten)]
	pub base: ProviderModelConfigBase,
	#[serde(rename = "type")]
	pub model_type: ImageTypeTag,
	/// API type override.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub api: Option<String>,
	#[serde(default)]
	pub output: Vec<ModelOutput>,
}

/// Structured classifier model configuration (`ProviderClassifierModelConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderClassifierModelConfig {
	#[serde(flatten)]
	pub base: ProviderModelConfigBase,
	#[serde(rename = "type")]
	pub model_type: ClassifierTypeTag,
	/// API type override.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub api: Option<String>,
	/// Maximum context window size in tokens.
	#[serde(rename = "contextWindow", default)]
	pub context_window: u64,
}

/// Configuration for a model within a provider (`ProviderModelConfig` union).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProviderModelConfig {
	Chat(ProviderChatModelConfig),
	Image(ProviderImageModelConfig),
	Classifier(ProviderClassifierModelConfig),
}

/// Route callback of [`ExtensionVirtualModel`]: like
/// [`VirtualModelRouteFn`] but with the extension context.
pub type ExtensionModelRouteFn =
	Arc<dyn Fn(ModelRouteRequest, ExtensionContext) -> BoxFuture<'static, ModelRoute> + Send + Sync>;

/// Virtual model registered via `registerVirtualModel()`
/// (`ExtensionVirtualModel<TState>`); the data members of
/// [`VirtualModelDefinition`] plus a context-aware `route` (TState erased to
/// JSON, D27).
#[derive(Clone)]
pub struct ExtensionVirtualModel {
	/// Provider the virtual model is listed under.
	pub provider: String,
	/// Model id; must not collide with a physical model of `provider`.
	pub id: String,
	pub name: String,
	/// Thinking levels offered for selection. Default: `["off"]`.
	pub thinking_levels: Option<Vec<ModelThinkingLevel>>,
	/// Limits shown before the first response.
	pub context_window: Option<u64>,
	pub max_tokens: Option<u64>,
	/// Input types accepted for selection.
	pub input: Option<Vec<ModelInput>>,
	/// Pick the physical model and thinking level for one request.
	pub route: ExtensionModelRouteFn,
}

impl std::fmt::Debug for ExtensionVirtualModel {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ExtensionVirtualModel")
			.field("provider", &self.provider)
			.field("id", &self.id)
			.field("name", &self.name)
			.finish_non_exhaustive()
	}
}

/// Extension factory function (`ExtensionFactory`); sync and async
/// initialization both become a future. `Err` mirrors an upstream throw and
/// becomes a load error.
pub type ExtensionFactory = Arc<dyn Fn(ExtensionApiHandle) -> BoxFuture<'static, Result<(), String>> + Send + Sync>;

/// Handle of the `pi` API passed to factories (shared [`ExtensionApi`]).
pub type ExtensionApiHandle = Arc<dyn ExtensionApi>;

/// An extension registered inline by the host (`InlineExtension`).
pub enum InlineExtension {
	Factory(ExtensionFactory),
	Named {
		/// Display name shown as `<inline:name>` (or `builtin:name` with
		/// `builtin`) in the startup list and errors.
		name: String,
		factory: ExtensionFactory,
		/// Omit from the startup Extensions list.
		hidden: Option<bool>,
		/// Yield name conflicts to another extension instead of reporting.
		replaceable: Option<bool>,
		/// Register as the `builtin:<name>` extension resource instead of an
		/// inline extension.
		builtin: Option<bool>,
	},
}

// ============================================================================
// Loaded Extension Types
// ============================================================================

/// A tool registered by an extension (`RegisteredTool`).
#[derive(Clone)]
pub struct RegisteredTool {
	pub definition: ToolDefinition,
	pub source_info: SourceInfo,
}

impl std::fmt::Debug for RegisteredTool {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("RegisteredTool")
			.field("definition", &self.definition)
			.finish_non_exhaustive()
	}
}

/// Tool info returned by `getAllTools` (`ToolInfo`). Serialize only:
/// [`SourceInfo`] has no `Deserialize` (crate docs D1 wire direction).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolInfo {
	pub name: String,
	pub description: String,
	/// Parameter schema (JSON Schema document — D27).
	pub parameters: JsonValue,
	#[serde(rename = "promptGuidelines", default, skip_serializing_if = "Option::is_none")]
	pub prompt_guidelines: Option<Vec<String>>,
	pub exposure: ToolExposure,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub namespace: Option<ToolNamespace>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub annotations: Option<ToolAnnotations>,
	pub source_info: SourceInfo,
}

/// A registered CLI flag (`ExtensionFlag`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtensionFlag {
	pub name: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub description: Option<String>,
	/// Boolean or string flag.
	#[serde(rename = "type")]
	pub flag_type: ExtensionFlagType,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub default: Option<FlagValue>,
	pub extension_path: String,
}

/// Flag kind (`ExtensionFlag.type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtensionFlagType {
	#[serde(rename = "boolean")]
	Boolean,
	#[serde(rename = "string")]
	String,
}

/// A registered keyboard shortcut (`ExtensionShortcut`).
#[derive(Clone)]
pub struct ExtensionShortcut {
	pub shortcut: KeyId,
	pub description: Option<String>,
	pub handler: Arc<dyn Fn(ExtensionContext) -> BoxFuture<'static, ()> + Send + Sync>,
	pub extension_path: String,
}

impl std::fmt::Debug for ExtensionShortcut {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ExtensionShortcut")
			.field("shortcut", &self.shortcut)
			.field("description", &self.description)
			.finish_non_exhaustive()
	}
}

/// Append-entry action (`AppendEntryHandler`).
pub type AppendEntryHandler = Arc<dyn Fn(&str, Option<JsonValue>) + Send + Sync>;
/// Session-name actions (`Set/GetSessionNameHandler`).
pub type SetSessionNameHandler = Arc<dyn Fn(String) + Send + Sync>;
pub type GetSessionNameHandler = Arc<dyn Fn() -> Option<String> + Send + Sync>;
/// Active-tools actions (`Get/SetActiveToolsHandler`).
pub type GetActiveToolsHandler = Arc<dyn Fn() -> Vec<String> + Send + Sync>;
pub type SetActiveToolsHandler = Arc<dyn Fn(Vec<String>) + Send + Sync>;
/// All-tools action (`GetAllToolsHandler`).
pub type GetAllToolsHandler = Arc<dyn Fn() -> Vec<ToolInfo> + Send + Sync>;
/// Settings action (`GetSettingsHandler`).
pub type GetSettingsHandler = Arc<dyn Fn() -> Settings + Send + Sync>;
/// Commands action (`GetCommandsHandler`).
pub type GetCommandsHandler = Arc<dyn Fn() -> Vec<SlashCommandInfo> + Send + Sync>;
/// Tools-refresh action (`RefreshToolsHandler`).
pub type RefreshToolsHandler = Arc<dyn Fn() + Send + Sync>;
/// Model action (`SetModelHandler`).
pub type SetModelHandler = Arc<dyn Fn(Model) -> BoxFuture<'static, bool> + Send + Sync>;
/// Thinking-level actions (`Get/SetThinkingLevelHandler`).
pub type GetThinkingLevelHandler = Arc<dyn Fn() -> AgentThinkingLevel + Send + Sync>;
pub type SetThinkingLevelHandler = Arc<dyn Fn(AgentThinkingLevel) + Send + Sync>;
/// Label action (`SetLabelHandler`).
pub type SetLabelHandler = Arc<dyn Fn(&str, Option<String>) + Send + Sync>;

/// Shared state created by the loader (`ExtensionRuntimeState`); flag values
/// default during registration and CLI values arrive after. Vtable members
/// are populated by the loader/runner (D28). Runtime clones share the
/// data fields through interior mutability so drains after load are visible
/// to the caller's handle (D40).
pub struct ExtensionRuntimeState {
	/// Registered flag values.
	pub flag_values: parking_lot::RwLock<HashMap<String, FlagValue>>,
	/// Legacy provider-config registrations queued during load, processed when
	/// the runner binds.
	pub pending_provider_registrations: parking_lot::RwLock<Vec<PendingProviderRegistration>>,
	/// Native pi-ai provider registrations queued during load.
	pub pending_native_provider_registrations: parking_lot::RwLock<Vec<PendingNativeProviderRegistration>>,
	/// Virtual model registrations queued during load.
	pub pending_virtual_model_registrations: parking_lot::RwLock<Vec<PendingVirtualModelRegistration>>,
	// NOTE: the closure/Arc fields follow; `Clone` is derived via the shared
	// `Arc`s — see the explicit derive-able tail below. (parking_lot locks are
	// cloned by value through their guards above.)
	/// Create an extension context; panics before the runner binds.
	pub create_context: Arc<dyn Fn() -> ExtensionContext + Send + Sync>,
	/// Panics via `Err` message when this instance is stale after runtime
	/// replacement (`assertActive`).
	pub assert_active: Arc<dyn Fn() -> Result<(), String> + Send + Sync>,
	/// Marks this instance stale (`invalidate`).
	pub invalidate: Arc<dyn Fn(Option<String>) + Send + Sync>,
	/// Retain an event-bus subscription until this runtime is invalidated.
	pub track_event_bus_subscription: Arc<dyn Fn(Unsubscribe) -> Unsubscribe + Send + Sync>,
	/// Register/unregister a provider: queues before bind, calls the model
	/// registry directly after.
	pub register_provider: Arc<dyn Fn(String, ProviderConfig, Option<String>) + Send + Sync>,
	pub register_native_provider: Arc<dyn Fn(Arc<dyn AiProvider>, Option<String>) + Send + Sync>,
	pub unregister_provider: Arc<dyn Fn(String, Option<String>) + Send + Sync>,
	/// Servers registered with `registerMcpServer`.
	pub mcp_servers: Arc<McpServerRegistry>,
	pub register_virtual_model: Arc<dyn Fn(VirtualModelDefinition, Option<String>) + Send + Sync>,
	pub unregister_virtual_model: Arc<dyn Fn(String, String) + Send + Sync>,
}

impl Clone for ExtensionRuntimeState {
	fn clone(&self) -> Self {
		ExtensionRuntimeState {
			flag_values: parking_lot::RwLock::new(self.flag_values.read().clone()),
			pending_provider_registrations: parking_lot::RwLock::new(self.pending_provider_registrations.read().clone()),
			pending_native_provider_registrations: parking_lot::RwLock::new(
				self.pending_native_provider_registrations.read().clone(),
			),
			pending_virtual_model_registrations: parking_lot::RwLock::new(
				self.pending_virtual_model_registrations.read().clone(),
			),
			create_context: Arc::clone(&self.create_context),
			assert_active: Arc::clone(&self.assert_active),
			invalidate: Arc::clone(&self.invalidate),
			track_event_bus_subscription: Arc::clone(&self.track_event_bus_subscription),
			register_provider: Arc::clone(&self.register_provider),
			register_native_provider: Arc::clone(&self.register_native_provider),
			unregister_provider: Arc::clone(&self.unregister_provider),
			mcp_servers: Arc::clone(&self.mcp_servers),
			register_virtual_model: Arc::clone(&self.register_virtual_model),
			unregister_virtual_model: Arc::clone(&self.unregister_virtual_model),
		}
	}
}

impl std::fmt::Debug for ExtensionRuntimeState {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ExtensionRuntimeState")
			.field("flag_values", &self.flag_values.read().keys().collect::<Vec<_>>())
			.field(
				"pending_provider_registrations",
				&self.pending_provider_registrations.read().len(),
			)
			.finish_non_exhaustive()
	}
}

impl ExtensionRuntimeState {
	/// `flagValues.get(name)`.
	pub fn get_flag_value(&self, name: &str) -> Option<FlagValue> {
		self.flag_values.read().get(name).cloned()
	}

	/// Whether a flag has any value (`flagValues.has(name)`).
	pub fn has_flag_value(&self, name: &str) -> bool {
		self.flag_values.read().contains_key(name)
	}

	/// `flagValues.set(name, value)`.
	pub fn set_flag_value(&self, name: &str, value: FlagValue) {
		self.flag_values.write().insert(name.to_string(), value);
	}

	/// Queued legacy provider registrations (drained on bind).
	pub fn pending_providers(&self) -> Vec<PendingProviderRegistration> {
		self.pending_provider_registrations.read().clone()
	}

	/// Drops the queued legacy provider/native registrations.
	pub fn clear_pending_providers(&self) {
		*self.pending_provider_registrations.write() = Vec::new();
		*self.pending_native_provider_registrations.write() = Vec::new();
	}

	/// Queued virtual-model registrations (drained on bind).
	pub fn pending_virtual_models(&self) -> Vec<PendingVirtualModelRegistration> {
		self.pending_virtual_model_registrations.read().clone()
	}

	/// Drops the queued virtual-model registrations.
	pub fn clear_pending_virtual_models(&self) {
		*self.pending_virtual_model_registrations.write() = Vec::new();
	}
}

/// Queued legacy provider registration (`{ name, config, extensionPath }`).
#[derive(Debug, Clone, Default)]
pub struct PendingProviderRegistration {
	pub name: String,
	pub config: ProviderConfig,
	pub extension_path: String,
}

/// Queued native provider registration.
#[derive(Clone, Default)]
pub struct PendingNativeProviderRegistration {
	pub provider: Option<Arc<dyn AiProvider>>,
	pub extension_path: String,
}

impl std::fmt::Debug for PendingNativeProviderRegistration {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("PendingNativeProviderRegistration")
			.field("extension_path", &self.extension_path)
			.finish_non_exhaustive()
	}
}

/// Queued virtual model registration.
#[derive(Clone, Default)]
pub struct PendingVirtualModelRegistration {
	pub definition: Option<VirtualModelDefinition>,
	pub extension_path: String,
}

impl std::fmt::Debug for PendingVirtualModelRegistration {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("PendingVirtualModelRegistration")
			.field("extension_path", &self.extension_path)
			.finish_non_exhaustive()
	}
}

/// Action implementations for `pi.*` API methods (`ExtensionActions`),
/// provided to `runner.initialize()` and copied into the shared runtime.
#[derive(Clone, Default)]
pub struct ExtensionActions {
	pub send_message: Option<SendMessageHandler>,
	pub send_user_message: Option<SendUserMessageHandler>,
	pub append_entry: Option<AppendEntryHandler>,
	pub set_session_name: Option<SetSessionNameHandler>,
	pub get_session_name: Option<GetSessionNameHandler>,
	pub set_label: Option<SetLabelHandler>,
	pub get_active_tools: Option<GetActiveToolsHandler>,
	pub get_all_tools: Option<GetAllToolsHandler>,
	pub get_settings: Option<GetSettingsHandler>,
	pub set_active_tools: Option<SetActiveToolsHandler>,
	pub refresh_tools: Option<RefreshToolsHandler>,
	pub get_commands: Option<GetCommandsHandler>,
	pub set_model: Option<SetModelHandler>,
	pub get_thinking_level: Option<GetThinkingLevelHandler>,
	pub set_thinking_level: Option<SetThinkingLevelHandler>,
}

impl std::fmt::Debug for ExtensionActions {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ExtensionActions").finish_non_exhaustive()
	}
}

/// Actions for [`ExtensionContext`] (`ctx.*` in event handlers); required by
/// all modes. `None` members mirror upstream optional actions
/// (`getSystemPromptOptions`, `executeTool`, `getCallableTools`).
#[derive(Clone, Default)]
pub struct ExtensionContextActions {
	pub get_model: Option<Arc<dyn Fn() -> Option<Model> + Send + Sync>>,
	pub get_scoped_models: Option<Arc<dyn Fn() -> Vec<ScopedModel> + Send + Sync>>,
	pub is_idle: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
	pub is_project_trusted: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
	pub get_signal: Option<Arc<dyn Fn() -> Option<AbortSignal> + Send + Sync>>,
	pub abort: Option<Arc<dyn Fn() + Send + Sync>>,
	pub has_pending_messages: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
	pub shutdown: Option<Arc<dyn Fn() + Send + Sync>>,
	pub get_context_usage: Option<Arc<dyn Fn() -> Option<ContextUsage> + Send + Sync>>,
	pub compact: Option<Arc<dyn Fn(Option<CompactOptions>) + Send + Sync>>,
	pub get_system_prompt: Option<Arc<dyn Fn() -> String + Send + Sync>>,
	pub get_system_prompt_options: Option<Arc<dyn Fn() -> BuildSystemPromptOptions + Send + Sync>>,
	/// Backs [`ExtensionToolContext::execute_tool`]; without it nested calls
	/// fail.
	pub execute_tool: Option<Arc<dyn Fn(String, String, JsonValue, ExecuteToolOptions) -> BoxFuture<'static, AgentToolCallOutcome> + Send + Sync>>,
	/// Backs [`ExtensionToolContext::tools`].
	pub get_callable_tools: Option<Arc<dyn Fn() -> Vec<AgentTool> + Send + Sync>>,
}

impl std::fmt::Debug for ExtensionContextActions {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ExtensionContextActions").finish_non_exhaustive()
	}
}

/// Actions for [`ExtensionCommandContext`] (`ctx.*` in command handlers);
/// only needed for interactive mode where extension commands are invokable.
#[derive(Clone, Default)]
pub struct ExtensionCommandContextActions {
	pub wait_for_idle: Option<Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>>,
	pub new_session: Option<Arc<dyn Fn(Option<NewSessionCommandOptions>) -> BoxFuture<'static, SessionSwitchOutcome> + Send + Sync>>,
	pub fork: Option<Arc<dyn Fn(String, Option<ForkCommandOptions>) -> BoxFuture<'static, SessionSwitchOutcome> + Send + Sync>>,
	pub navigate_tree: Option<Arc<dyn Fn(String, Option<NavigateTreeOptions>) -> BoxFuture<'static, SessionSwitchOutcome> + Send + Sync>>,
	pub switch_session: Option<Arc<dyn Fn(String, Option<SwitchSessionOptions>) -> BoxFuture<'static, SessionSwitchOutcome> + Send + Sync>>,
	pub reload: Option<Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>>,
}

impl std::fmt::Debug for ExtensionCommandContextActions {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ExtensionCommandContextActions").finish_non_exhaustive()
	}
}

/// Full runtime = state + actions + bind core (`ExtensionRuntime`). Created
/// by the loader with throwing action stubs, completed by
/// `runner.bind_core()`.
#[derive(Debug, Clone)]
pub struct ExtensionRuntime {
	pub state: Arc<ExtensionRuntimeState>,
	pub actions: ExtensionActions,
	/// Shared bind state the runner fills via `bind_core` (D40).
	pub core: Arc<RuntimeCore>,
}

impl std::ops::Deref for ExtensionRuntime {
	type Target = ExtensionRuntimeState;

	fn deref(&self) -> &Self::Target {
		&self.state
	}
}

/// Provider registration callbacks a mode can install via `bind_core`
/// (upstream `providerActions`). When absent, queued registrations are
/// drained without a registry call (the model registry is an opaque stub
/// until that module ports).
#[derive(Clone, Default)]
pub struct ProviderActions {
	pub register_provider: Option<Arc<dyn Fn(&str, &ProviderConfig) + Send + Sync>>,
	pub register_native_provider: Option<Arc<dyn Fn(&Arc<dyn pi_ai::Provider>) + Send + Sync>>,
	pub unregister_provider: Option<Arc<dyn Fn(&str) + Send + Sync>>,
	pub register_virtual_model: Option<Arc<dyn Fn(&VirtualModelDefinition) + Send + Sync>>,
	pub unregister_virtual_model: Option<Arc<dyn Fn(&str, &str) + Send + Sync>>,
}

impl std::fmt::Debug for ProviderActions {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ProviderActions").finish_non_exhaustive()
	}
}

/// The action set `bind_core` installs (`bindCore`'s arguments), readable by
/// the loader closures and the runner's context builders.
#[derive(Clone, Default)]
pub struct BoundCore {
	pub actions: ExtensionActions,
	pub context_actions: ExtensionContextActions,
	pub command_actions: ExtensionCommandContextActions,
	pub provider_actions: ProviderActions,
	pub create_context: Option<Arc<dyn Fn() -> Result<ExtensionContext, String> + Send + Sync>>,
}

/// Shared bind state behind every [`ExtensionRuntime`] clone (D40).
#[derive(Default)]
pub struct RuntimeCore {
	pub bound: parking_lot::RwLock<Option<BoundCore>>,
}

impl std::fmt::Debug for RuntimeCore {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("RuntimeCore")
			.field("bound", &self.bound.read().is_some())
			.finish()
	}
}

/// Loaded extension with all registered items (`Extension`).
#[derive(Clone)]
pub struct Extension {
	pub path: String,
	pub resolved_path: String,
	pub hidden: Option<bool>,
	/// See [`InlineExtension`].
	pub replaceable: Option<bool>,
	pub source_info: SourceInfo,
	/// Erased handlers by event name (`Map<string, HandlerFn[]>`; D31).
	/// Shared with the loader API so subscriptions/unsubscriptions after
	/// commit are visible to the runner (D40).
	pub handlers: Arc<parking_lot::Mutex<HashMap<String, Vec<HandlerFn>>>>,
	pub tools: HashMap<String, RegisteredTool>,
	pub message_renderers: HashMap<String, MessageRenderer>,
	pub tool_renderers: Option<Vec<ToolRendererResolver>>,
	pub markdown_transformer: Option<MarkdownTransformer>,
	pub entry_renderers: Option<HashMap<String, EntryRenderer>>,
	pub commands: HashMap<String, RegisteredCommand>,
	pub flags: HashMap<String, ExtensionFlag>,
	pub shortcuts: HashMap<KeyId, ExtensionShortcut>,
}

impl std::fmt::Debug for Extension {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Extension")
			.field("path", &self.path)
			.field("resolved_path", &self.resolved_path)
			.field("handlers", &self.handlers.lock().keys().collect::<Vec<_>>())
			.field("tools", &self.tools.keys().collect::<Vec<_>>())
			.field("commands", &self.commands.keys().collect::<Vec<_>>())
			.finish_non_exhaustive()
	}
}

/// Error/warning record of extension loading (`LoadExtensionsResult.errors/
/// warnings` members).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadRecord {
	pub path: String,
	pub error: String,
}

/// Result of loading extensions (`LoadExtensionsResult`).
#[derive(Debug, Clone)]
pub struct LoadExtensionsResult {
	pub extensions: Vec<Extension>,
	pub errors: Vec<LoadRecord>,
	pub warnings: Option<Vec<LoadRecord>>,
	/// Shared runtime — actions are throwing stubs until
	/// `runner.initialize()`.
	pub runtime: ExtensionRuntime,
}

// ============================================================================
// Extension Error
// ============================================================================

/// An error raised inside an extension handler (`ExtensionError`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtensionError {
	pub extension_path: String,
	pub event: String,
	pub error: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub stack: Option<String>,
}

// ============================================================================
// Tests — upstream has no dedicated types.test.ts; these cover the runtime
// functions (guards, `isToolCallEventType`, `defineTool`) and the hand-rolled
// event serialization shapes (D30).
// ============================================================================

#[cfg(test)]
mod types_tests {
	use super::*;
	use serde_json::json;

	fn call_event(tool_name: &str) -> ToolCallEvent {
		ToolCallEvent::custom("call-1", tool_name, json!({}))
	}

	fn result_event(tool_name: &str) -> ToolResultEvent {
		ToolResultEvent::Custom(ToolResultEventBase {
			tool_call_id: "call-1".to_string(),
			parent_tool_call_id: None,
			tool_name: tool_name.to_string(),
			input: json!({}),
			content: Vec::new(),
			structured_content: None,
			is_error: false,
			usage: None,
			details: JsonValue::Null,
		})
	}

	#[test]
	fn tool_result_guards_compare_tool_names() {
		assert!(is_bash_tool_result(&result_event("bash")));
		assert!(is_power_shell_tool_result(&result_event("powershell")));
		assert!(is_read_tool_result(&result_event("read")));
		assert!(is_edit_tool_result(&result_event("edit")));
		assert!(is_write_tool_result(&result_event("write")));
		assert!(is_grep_tool_result(&result_event("grep")));
		assert!(is_find_tool_result(&result_event("find")));
		assert!(is_ls_tool_result(&result_event("ls")));
		// Name-only checks, like upstream: a custom event named `bash` passes.
		assert!(is_bash_tool_result(&result_event("bash")));
		assert!(!is_bash_tool_result(&result_event("grep")));
		assert!(!is_ls_tool_result(&result_event("find")));
	}

	#[test]
	fn is_tool_call_event_type_matches_names() {
		assert!(is_tool_call_event_type("bash", &call_event("bash")));
		assert!(is_tool_call_event_type("my_tool", &call_event("my_tool")));
		assert!(!is_tool_call_event_type("bash", &call_event("grep")));
	}

	#[test]
	fn tool_call_event_serializes_flat_wire_shape() {
		let event = ToolCallEvent::Bash(ToolCallEventBase {
			tool_call_id: "call-1".to_string(),
			parent_tool_call_id: None,
			tool_name: "bash".to_string(),
			input: json!({"command": "ls"}),
		});
		let value = serde_json::to_value(&event).unwrap();
		assert_eq!(
			value,
			json!({
				"type": "tool_call",
				"toolCallId": "call-1",
				"toolName": "bash",
				"input": {"command": "ls"},
			})
		);

		let nested = ToolCallEvent::Custom(ToolCallEventBase {
			tool_call_id: "call-1/0".to_string(),
			parent_tool_call_id: Some("call-1".to_string()),
			tool_name: "my_tool".to_string(),
			input: json!({}),
		});
		let value = serde_json::to_value(&nested).unwrap();
		assert_eq!(value["parentToolCallId"], json!("call-1"));
	}

	#[test]
	fn tool_result_event_serializes_flat_wire_shape() {
		let event = ToolResultEvent::Write(ToolResultEventBase {
			tool_call_id: "call-2".to_string(),
			parent_tool_call_id: None,
			tool_name: "write".to_string(),
			input: json!({}),
			content: Vec::new(),
			structured_content: None,
			is_error: true,
			usage: None,
			// `WriteToolResultEvent.details` is `undefined` upstream: absent.
			details: JsonValue::Null,
		});
		let value = serde_json::to_value(&event).unwrap();
		assert_eq!(
			value,
			json!({
				"type": "tool_result",
				"toolCallId": "call-2",
				"toolName": "write",
				"input": {},
				"content": [],
				"isError": true,
			})
		);
		assert!(value.get("details").is_none());
		assert!(value.get("structuredContent").is_none());
	}

	#[test]
	fn tool_call_event_type_tags_are_constant() {
		let value = serde_json::to_value(&call_event("bash")).unwrap();
		assert_eq!(value["type"], json!("tool_call"));
		let value = serde_json::to_value(&result_event("bash")).unwrap();
		assert_eq!(value["type"], json!("tool_result"));
	}

	#[test]
	fn session_event_members_serialize_with_type_tags() {
		let value = serde_json::to_value(&SessionStartEvent {
			reason: SessionStartReason::Resume,
			previous_session_file: Some("/tmp/s.jsonl".to_string()),
		})
		.unwrap();
		assert_eq!(
			value,
			json!({"type": "session_start", "reason": "resume", "previousSessionFile": "/tmp/s.jsonl"})
		);

		let value = serde_json::to_value(&SessionCompactFailedEvent {
			reason: CompactReason::Overflow,
			error_message: None,
			aborted: true,
			will_retry: false,
			from_extension: false,
		})
		.unwrap();
		assert_eq!(value["type"], json!("session_compact_failed"));
		assert_eq!(value["aborted"], json!(true));
		assert!(value.get("errorMessage").is_none());
	}

	#[test]
	fn define_tool_is_identity() {
		let definition = ToolDefinition::new("greet", "Greet", "Says hello", json!({"type": "object"}));
		let wrapped = define_tool(definition.clone());
		assert_eq!(wrapped.name, definition.name);
		assert_eq!(wrapped.label, definition.label);
		assert_eq!(wrapped.parameters, definition.parameters);
	}

	#[test]
	fn normalized_system_prompt_options_fill_defaults() {
		let normalized: NormalizedBuildSystemPromptOptions = BuildSystemPromptOptions {
			cwd: "/tmp".to_string(),
			..Default::default()
		}
		.into();
		assert_eq!(normalized.selected_tools, ["read", "bash", "edit", "write"]);
		assert!(normalized.tool_snippets.is_empty());
		assert!(normalized.tool_guidelines.is_empty());
		assert!(normalized.prompt_guidelines.is_empty());
		assert_eq!(normalized.append_system_prompt, "");
		assert_eq!(normalized.cwd, "/tmp");
	}

	#[test]
	fn input_event_result_is_tagged_by_action() {
		let value = serde_json::to_value(&InputEventResult::Continue).unwrap();
		assert_eq!(value, json!({"action": "continue"}));
		let value = serde_json::to_value(&InputEventResult::Transform {
			text: "hi".to_string(),
			images: None,
		})
		.unwrap();
		assert_eq!(value, json!({"action": "transform", "text": "hi"}));
	}
}
