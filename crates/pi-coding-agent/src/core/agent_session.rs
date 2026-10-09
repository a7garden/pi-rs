//! Port of `packages/coding-agent/src/core/agent-session.ts` — the
//! [`AgentSession`] class shared by every run mode: agent state access,
//! event subscription with session persistence, model/thinking management,
//! compaction (manual + automatic), bash execution, retry, tree navigation,
//! stats, and the extension-runtime lifecycle.
//!
//! The standalone [`parse_skill_block`] surface keeps its previous home in
//! this module (consumers: interactive-mode, index, export-html upstream).
//!
//! # Deviations
//! - **D74 — shared mutable `SessionManager`**: upstream hands every consumer
//!   the same JS object. Rust `SessionManager` methods take `&mut self`, so
//!   the runner and the extension context hold
//!   `Arc<Mutex<SessionManager>>` (upstream `ReadonlySessionManager` stays a
//!   convention, D28) and `ExtensionRunner::bind_core`/`set_ui_context` take
//!   `&self` (their writes already go through interior-mutable shared state).
//! - **D75 — typed session event enum**: the TS `AgentSessionEvent` union
//!   (core `AgentEvent` pass-throughs plus session-specific payloads and the
//!   `WithParentToolCallId` conditional) collapses into [`AgentSessionEvent`],
//!   a Rust enum; `tool_execution_*` variants carry
//!   `parent_tool_call_id: Option<String>`.
//! - **D76 — message identity by value**: upstream keys `_entryIdsByMessage`
//!   (WeakMap) and `_boundaryDispatchedMessages` (WeakSet) by object
//!   identity. `AgentMessage` is a value type in Rust, so entries are
//!   located by structural equality (branch scan, then projection index) and
//!   dispatched boundaries are remembered by serialized fingerprint.
//! - **D77 — `message_end` replacement**: upstream mutates the finalized
//!   message object in place so agent state, listeners, and persistence see
//!   the extension replacement. Rust finds the structurally-equal message in
//!   agent state and swaps it.
//! - **D78 — auto-compaction test seam**: upstream tests monkey-patch the
//!   private `_runAutoCompaction` via `vi.spyOn`; the port routes the call
//!   through an internal override slot that only the in-module tests set.
//! - **D79 — bug-report summary defer**: `bug-report.ts` is not ported yet,
//!   so [`AgentSession::summarize_for_bug_report`] returns a fixed `Err`
//!   describing the missing module. `export-html/` has since landed in
//!   [`core::export_html`] (D80/D81), so [`AgentSession::export_to_html`] is
//!   wired to it. Every other member is ported.
//!
//! `agent-session-runtime.ts` and `agent-session-services.ts` (the
//! `AgentSessionRuntime`/services layer) are separate ports; the upstream
//! suites that construct sessions through the runtime belong to them.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;
use pi_agent::types::{
	AgentLoopTurnUpdate, AgentRequestUpdate, AgentToolCallOutcome, AfterToolCallContext, AfterToolCallResult,
	BeforeToolCallContext, BeforeToolCallResult, BoxFuture, PrepareRequestContext, PromptInput, QueueMode,
};
use pi_agent::{Agent, AgentEvent, AgentMessage, AgentOptions, AgentThinkingLevel, AgentTool};
use pi_ai::models::{clamp_thinking_level, get_supported_thinking_levels, models_are_equal};
use pi_ai::types::{
	AnyModel, AssistantMessage, Context, ImageContent, Message, Model, ModelThinkingLevel, ProviderEnv,
	ProviderHeaders, SimpleStreamOptions, SystemMessage, TextContent, ToolCall, ToolResultMessage,
	TranscriptContext, Usage, UserContent, UserContentField,
};
use pi_ai::utils::abort::{AbortController as AiAbortController, AbortSignal as AiAbortSignal};
use pi_ai::utils::overflow::{is_context_overflow, is_recoverable_length};
use pi_ai::utils::retry::{is_retryable_assistant_error, retry_delay_ms, RetryCallbacks, RetryPolicy};
use serde_json::{json, Value as JsonValue};

use crate::core::auth_guidance::{format_no_api_key_found_message, format_no_model_selected_message};
use crate::core::bash_executor::{execute_bash_with_operations, BashResult};
use crate::core::cache_warmer::{CacheWarmer, CacheWarmingStatus};
use crate::core::compaction::{
	calculate_context_tokens, collect_entries_for_branch_summary, compact as compact_context,
	estimate_context_tokens, estimate_projected_context_tokens, estimate_tokens, generate_branch_summary,
	prepare_compaction, should_compact, CompactionPreparation, CompactionResult, CompactionSettings,
};
use crate::core::defaults::{DEFAULT_THINKING_LEVEL, THINKING_LEVEL_OPTIONS};
use crate::core::event_bus::create_event_bus;
use crate::core::extensions::loader::{create_extension_runtime, load_extension_from_factory};
use crate::core::extensions::runner::{emit_session_shutdown_event, BoundaryContextBuilder, ExtensionErrorListener, ExtensionRunner};
use crate::core::extensions::types::{
	AgentActivityOutcome, BeforeAgentStartEvent, BoundaryContextPreview, CompactOptions,
	CompactReason, CompactionEntryDraft, ContextEditEntryDraft, ContextUsage, CustomEntryDraft,
	CustomMessageDraft, CustomMessageEntryDraft, ExecuteToolOptions, ExtensionActions, ExtensionCommandContextActions,
	ExtensionContextActions, ExtensionError, ExtensionMode, ExtensionUi,
	InputEventResult, InputSource, ProviderActions, SessionBeforeCompactEvent,
	SessionBeforeTreeEvent, SessionBeforeTreeResult, SessionBoundaryDraft, SessionStartEvent, SessionStartReason,
	ToolDefinition, ToolExposure, ToolInfo, ToolLoadout, ToolLoadoutChanges, TreePreparation,
	Unsubscribe, VirtualModelDefinition,
};
use crate::core::extensions::wrapper::wrap_registered_tools;
use crate::core::messages::{convert_to_llm, BashExecutionMessage, CustomMessage};
use crate::core::model_registry::ModelRegistry;
use crate::core::model_resolver::ScopedModel;
use crate::core::model_runtime::{ModelRuntime, ModelRuntimeAuthOverrides};
use crate::core::nested_tool_calls::{NestedToolCallHost, NestedToolCallOptions, NestedToolCallRunner, NestedToolExecutionEvent, NestedUpdateForwarder};
use crate::core::prompt_templates::{expand_prompt_template, PromptTemplate};
use crate::core::resource_loader::{DefaultResourceLoader, ResourceExtensionPaths, ResourceLoaderReloadOptions};
use crate::core::session_export::export_session_to_jsonl;
use crate::core::session_manager::{
	get_latest_compaction_entry, BranchSummaryEntry, CompactionEntry, ContextEditEntry, SessionEntry, SessionManager,
	SessionProjection,
};
use crate::core::settings_manager::{CacheWarmingMode, Settings, SettingsManager, DEFAULT_TOOL_NAMES, SteeringMode};
use crate::core::extensions::types::SlashCommandInfo;
use crate::core::source_info::{create_synthetic_source_info, is_synthetic_path, BUILTIN_PATH_PREFIX, SourceInfo};
use crate::core::system_prompt::{
	build_system_prompt_normalized, build_system_prompt_sections, diff_system_prompt_sections,
	normalize_build_system_prompt_options,
};
use crate::core::tools::bash::{create_local_bash_operations, BashOperations};
use crate::core::tools::index::create_all_tool_definitions;
use crate::core::tools::tool_definition_wrapper::create_tool_definition_from_agent_tool;
use crate::core::usage_totals::{add_usage_to_totals, combine_usage, create_usage_totals};
use crate::core::virtual_models::{
	find_latest_response, get_branch_selection, get_virtual_model_state, is_virtual_model, is_virtual_model_api,
	VirtualModelStateData, VIRTUAL_MODEL_STATE_ENTRY,
};
use crate::utils::abort::{AbortController, AbortSignal};
use crate::utils::frontmatter::strip_frontmatter;
use crate::utils::sleep::sleep;

// ============================================================================
// Skill Block Parsing
// ============================================================================

/// Parsed skill block from a user message (`ParsedSkillBlock`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedSkillBlock {
	pub name: String,
	pub location: String,
	pub content: String,
	pub user_message: Option<String>,
}

/// Parse a skill block from message text (`parseSkillBlock`). Returns `None`
/// if the text doesn't contain a skill block.
pub fn parse_skill_block(text: &str) -> Option<ParsedSkillBlock> {
	// Upstream regex:
	// /^<skill name="([^"]+)" location="([^"]+)">\n([\s\S]*?)\n<\/skill>(?:\n\n([\s\S]+))?$/
	let rest = text.strip_prefix("<skill ")?;
	let (name, rest) = match_attr(rest, "name")?;
	let rest = rest.strip_prefix(' ')?;
	let (location, rest) = match_attr(rest, "location")?;
	let rest = rest.strip_prefix(">\n")?;
	let (content, tail) = split_until_close(rest)?;
	let user_message = tail
		.strip_prefix("\n\n")
		.map(str::trim)
		.filter(|value| !value.is_empty())
		.map(str::to_string);
	Some(ParsedSkillBlock { name, location, content: content.to_string(), user_message })
}

/// Match `key="value"` at the start of `input`, returning the value and the
/// remainder (`([^"]+)` capture — at least one character).
fn match_attr<'a>(input: &'a str, key: &str) -> Option<(String, &'a str)> {
	let rest = input.strip_prefix(key)?.strip_prefix("=\"")?;
	let end = rest.find('"')?;
	if end == 0 {
		return None;
	}
	Some((rest[..end].to_string(), &rest[end + 1..]))
}

/// Match the `([\s\S]*?)\n<\/skill>(?:\n\n([\s\S]+))?$` tail: the shortest
/// content terminated by `\n</skill>` at end-of-input, with an optional
/// trailing user message.
fn split_until_close(input: &str) -> Option<(&str, &str)> {
	let mut search_from = 0;
	while let Some(offset) = input[search_from..].find("\n</skill>") {
		let content_end = search_from + offset;
		let tail = &input[content_end + "\n</skill>".len()..];
		// The regex anchors `</skill>` to the end of the message: only an
		// optional `\n\n<user message>` may follow.
		if tail.is_empty() || tail.starts_with("\n\n") {
			return Some((&input[..content_end], tail));
		}
		search_from = content_end + 1;
	}
	None
}

// ============================================================================
// Types
// ============================================================================

fn without_deleted_headers(headers: Option<ProviderHeaders>) -> Option<ProviderHeaders> {
	headers.map(|headers| headers.into_iter().filter(|(_, value)| value.is_some()).collect())
}

/// Canonical `streamSimple`-backed [`StreamFn`] (upstream: the `streamSimple`
/// function object itself). Agent constructors that want the default
/// provider dispatch install this instance so
/// [`AgentSession`] identity checks (`streamFunction === streamSimple`)
/// hold.
pub fn simple_stream_fn() -> pi_agent::types::StreamFn {
	static SIMPLE: std::sync::LazyLock<pi_agent::types::StreamFn> = std::sync::LazyLock::new(|| {
		Arc::new(move |model: Model, context: TranscriptContext, options: Option<SimpleStreamOptions>| {
			Box::pin(async move { Ok(pi_ai::compat::stream_simple(&model, &context_to_request_context(&context), options)) })
		})
	});
	Arc::clone(&SIMPLE)
}

fn context_to_request_context(context: &TranscriptContext) -> Context {
	Context { system_prompt: None, messages: context.messages().to_vec(), tools: None }
}

/// Models to cycle through with Ctrl+P (upstream `scopedModels` entries).
#[derive(Debug, Clone)]
pub struct ScopedModelEntry {
	pub model: Model,
	pub thinking_level: Option<AgentThinkingLevel>,
}

/// Session-specific events that extend the core agent events (D75).
#[derive(Debug, Clone)]
pub enum AgentSessionEvent {
	// Core agent events, passed through (with `willRetry` on `agent_end` and
	// `parentToolCallId` on nested tool executions).
	AgentStart,
	AgentEnd { messages: Vec<AgentMessage>, will_retry: bool },
	TurnStart,
	TurnEnd { message: AgentMessage, tool_results: Vec<ToolResultMessage> },
	MessageStart { message: AgentMessage },
	MessageUpdate { message: AgentMessage, assistant_message_event: pi_ai::types::AssistantMessageEvent },
	MessageEnd { message: AgentMessage },
	ToolExecutionStart {
		tool_call_id: String,
		tool_name: String,
		args: JsonValue,
		parent_tool_call_id: Option<String>,
	},
	ToolExecutionUpdate {
		tool_call_id: String,
		tool_name: String,
		args: JsonValue,
		partial_result: JsonValue,
		parent_tool_call_id: Option<String>,
	},
	ToolExecutionEnd {
		tool_call_id: String,
		tool_name: String,
		result: JsonValue,
		is_error: bool,
		parent_tool_call_id: Option<String>,
	},

	// Session-specific events.
	AgentSettled,
	QueueUpdate { steering: Vec<String>, follow_up: Vec<String> },
	CompactionStart { reason: CompactReason },
	EntryAppended { entry: SessionEntry },
	SessionInfoChanged { name: Option<String> },
	ThinkingLevelChanged { level: AgentThinkingLevel },
	CompactionEnd {
		reason: CompactReason,
		result: Option<CompactionResult>,
		aborted: bool,
		will_retry: bool,
		error_message: Option<String>,
	},
	AutoRetryStart { attempt: u32, max_attempts: u32, delay_ms: u64, error_message: String },
	AutoRetryEnd { success: bool, attempt: u32, final_error: Option<String> },
	SummarizationRetryScheduled { attempt: u32, max_attempts: u32, delay_ms: u64, error_message: String },
	SummarizationRetryAttemptStart { source: SummarizationRetrySource },
	SummarizationRetryFinished,
	BashExecutionUpdate { id: Option<String>, delta: String },
}

/// Source context of a summarization retry (`source` member).
#[derive(Debug, Clone, PartialEq)]
pub enum SummarizationRetrySource {
	BranchSummary,
	Compaction { reason: CompactReason },
}

/// Listener for agent session events (upstream listeners are synchronous).
pub type AgentSessionEventListener = Arc<dyn Fn(AgentSessionEvent) + Send + Sync>;

/// Boxes a deferred `prompt` re-entry as a `Send` future, breaking the
/// recursive async-fn `Send` inference cycle (mechanical, not a deviation).
fn boxed_prompt(
	session: AgentSession,
	text: String,
	options: Option<PromptOptions>,
) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send>> {
	Box::pin(async move {
		let session = &session;
		session.prompt(text, options).await
	})
}

/// Options for [`AgentSession::prompt`].
#[derive(Default)]
pub struct PromptOptions {
	/// Whether to dispatch extension commands and expand skill commands and
	/// prompt templates (default: true).
	pub expand_prompt_templates: Option<bool>,
	/// Image attachments.
	pub images: Option<Vec<ImageContent>>,
	/// When streaming, how to queue the message. Required if streaming.
	pub streaming_behavior: Option<StreamingBehavior>,
	/// Source of input for extension input event handlers.
	pub source: Option<InputSource>,
	/// RPC-mode hook observing how an accepted prompt was dispatched. Not
	/// called when the prompt is rejected.
	pub preflight_result: Option<Arc<dyn Fn(PromptDisposition) + Send + Sync>>,
}

/// How input is delivered during streaming (`"steer" | "followUp"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingBehavior {
	Steer,
	FollowUp,
}

/// Disposition of queued input (`QueuedInputDisposition`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueuedInputDisposition {
	Handled,
	Queued,
}

/// Disposition of a prompt (`PromptDisposition`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptDisposition {
	Handled,
	Queued,
	Started,
}

/// Options for model/thinking mutations.
#[derive(Default, Clone, Copy)]
pub struct ModelMutationOptions {
	/// Persist the new value to global defaults. Defaults to session-only.
	pub persist: bool,
}

/// Result from [`AgentSession::cycle_model`].
#[derive(Debug, Clone)]
pub struct ModelCycleResult {
	pub model: Model,
	pub thinking_level: AgentThinkingLevel,
	/// Whether cycling through scoped models (`--models` flag) or all
	/// available.
	pub is_scoped: bool,
}

/// Session statistics for `/session`.
#[derive(Debug, Clone)]
pub struct SessionStats {
	pub session_file: Option<String>,
	pub session_id: String,
	pub user_messages: usize,
	pub assistant_messages: usize,
	pub tool_calls: usize,
	pub tool_results: usize,
	pub total_messages: usize,
	pub tokens: SessionTokens,
	pub cost: f64,
	pub context_usage: Option<ContextUsage>,
}

/// Token totals of [`SessionStats`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SessionTokens {
	pub input: u64,
	pub output: u64,
	pub cache_read: u64,
	pub cache_write: u64,
	pub total: u64,
}

/// Extension bindings applied by [`AgentSession::bind_extensions`].
#[derive(Default)]
pub struct ExtensionBindings {
	pub ui_context: Option<Arc<dyn ExtensionUi>>,
	pub mode: Option<ExtensionMode>,
	pub command_context_actions: Option<ExtensionCommandContextActions>,
	pub abort_handler: Option<Arc<dyn Fn() + Send + Sync>>,
	pub shutdown_handler: Option<ShutdownHandler>,
	pub on_error: Option<ExtensionErrorListener>,
}

/// Callback invoked on graceful shutdown (upstream `ShutdownHandler`).
pub type ShutdownHandler = Arc<dyn Fn() + Send + Sync>;

/// Mutable ref the `Agent` uses to reach the current
/// [`ExtensionRunner`] (upstream `{ current?: ExtensionRunner }`).
pub type ExtensionRunnerRef = Arc<parking_lot::RwLock<Option<Arc<ExtensionRunner>>>>;

#[derive(Clone)]
struct ToolDefinitionEntry {
	definition: ToolDefinition,
	source_info: SourceInfo,
}

fn estimate_messages_tokens(messages: &[AgentMessage]) -> u64 {
	messages.iter().map(|message| estimate_tokens(message) as u64).sum()
}

/// Serialize a message for the boundary-dispatched fingerprint set (D76).
fn message_fingerprint(message: &AgentMessage) -> String {
	serde_json::to_string(message).unwrap_or_else(|_| format!("{:?}", message.timestamp()))
}


/// `AgentThinkingLevel` → `ModelThinkingLevel` (pi-agent D6 bridge).
fn agent_level_to_model_level(level: AgentThinkingLevel) -> ModelThinkingLevel {
	match level {
		AgentThinkingLevel::Off => ModelThinkingLevel::Off,
		AgentThinkingLevel::Minimal => ModelThinkingLevel::Minimal,
		AgentThinkingLevel::Low => ModelThinkingLevel::Low,
		AgentThinkingLevel::Medium => ModelThinkingLevel::Medium,
		AgentThinkingLevel::High => ModelThinkingLevel::High,
		AgentThinkingLevel::Xhigh => ModelThinkingLevel::Xhigh,
		AgentThinkingLevel::Max => ModelThinkingLevel::Max,
	}
}

/// `ModelThinkingLevel` → `AgentThinkingLevel` (`off` collapses to `off`).
fn model_level_to_agent_level(level: ModelThinkingLevel) -> AgentThinkingLevel {
	match level {
		ModelThinkingLevel::Off => AgentThinkingLevel::Off,
		ModelThinkingLevel::Minimal => AgentThinkingLevel::Minimal,
		ModelThinkingLevel::Low => AgentThinkingLevel::Low,
		ModelThinkingLevel::Medium => AgentThinkingLevel::Medium,
		ModelThinkingLevel::High => AgentThinkingLevel::High,
		ModelThinkingLevel::Xhigh => AgentThinkingLevel::Xhigh,
		ModelThinkingLevel::Max => AgentThinkingLevel::Max,
	}
}

/// Transcript-level thinking level for a session thinking level (`off` →
/// `None`).
fn agent_level_to_reasoning(level: AgentThinkingLevel) -> Option<pi_ai::types::ThinkingLevel> {
	match level {
		AgentThinkingLevel::Off => None,
		AgentThinkingLevel::Minimal => Some(pi_ai::types::ThinkingLevel::Minimal),
		AgentThinkingLevel::Low => Some(pi_ai::types::ThinkingLevel::Low),
		AgentThinkingLevel::Medium => Some(pi_ai::types::ThinkingLevel::Medium),
		AgentThinkingLevel::High => Some(pi_ai::types::ThinkingLevel::High),
		AgentThinkingLevel::Xhigh => Some(pi_ai::types::ThinkingLevel::Xhigh),
		AgentThinkingLevel::Max => Some(pi_ai::types::ThinkingLevel::Max),
	}
}

/// Upstream `error.message` of a thrown value.
fn thrown_message(error: &pi_ai::utils::thrown::ThrownValue) -> String {
	match error {
		pi_ai::utils::thrown::ThrownValue::Text(message) => message.clone(),
		pi_ai::utils::thrown::ThrownValue::Error(error) => error.to_string(),
	}
}

/// Upstream `ThinkingLevel` string form (session transcripts store the
/// string).
fn thinking_level_str(level: AgentThinkingLevel) -> String {
	match level {
		AgentThinkingLevel::Off => "off".to_string(),
		AgentThinkingLevel::Minimal => "minimal".to_string(),
		AgentThinkingLevel::Low => "low".to_string(),
		AgentThinkingLevel::Medium => "medium".to_string(),
		AgentThinkingLevel::High => "high".to_string(),
		AgentThinkingLevel::Xhigh => "xhigh".to_string(),
		AgentThinkingLevel::Max => "max".to_string(),
	}
}

/// `NormalizedBuildSystemPromptOptions` → `BuildSystemPromptOptions` (the
/// runner seam takes the un-normalized shape; D30).
fn build_options_from_normalized(
	options: &crate::core::extensions::types::NormalizedBuildSystemPromptOptions,
) -> crate::core::extensions::types::BuildSystemPromptOptions {
	crate::core::extensions::types::BuildSystemPromptOptions {
		custom_prompt: options.custom_prompt.clone(),
		force_system_prompt: options.force_system_prompt.clone(),
		selected_tools: Some(options.selected_tools.clone()),
		tool_snippets: Some(options.tool_snippets.clone()),
		tool_guidelines: Some(options.tool_guidelines.clone()),
		prompt_guidelines: Some(options.prompt_guidelines.clone()).filter(|guidelines| !guidelines.is_empty()),
		append_system_prompt: Some(options.append_system_prompt.clone()).filter(|prompt| !prompt.is_empty()),
		sections: Some(options.sections.clone()).filter(|sections| !sections.is_empty()),
		cwd: options.cwd.clone(),
		context_files: options.context_files.clone(),
		skills: options.skills.clone(),
	}
}

/// Exposure lookup closure for [`ToolLoadout`] (weak — the loadout outlives
/// no particular session).
fn exposure_closure(
	inner: &AgentSessionInner,
) -> Arc<dyn Fn(&str) -> ToolExposure + Send + Sync> {
	let weak = inner.self_weak.clone();
	Arc::new(move |name: &str| {
		weak.upgrade().map(|inner| inner.get_tool_exposure(name)).unwrap_or(ToolExposure::Direct)
	})
}


/// An all-defaults `NormalizedBuildSystemPromptOptions` (the type carries no
/// `Default` because every member is present after normalization).
fn empty_normalized_prompt_options() -> crate::core::extensions::types::NormalizedBuildSystemPromptOptions {
	crate::core::extensions::types::NormalizedBuildSystemPromptOptions {
		custom_prompt: None,
		force_system_prompt: None,
		selected_tools: Vec::new(),
		tool_snippets: Default::default(),
		tool_guidelines: Default::default(),
		prompt_guidelines: Vec::new(),
		append_system_prompt: String::new(),
		sections: Default::default(),
		cwd: String::new(),
		context_files: None,
		skills: None,
	}
}

/// Serialize a `CustomMessageDraft` onto the transcript as an opaque custom
/// message (pi-coding-agent D4).
fn custom_draft_to_agent_message(draft: &CustomMessageDraft) -> AgentMessage {
	AgentMessage::Custom(serde_json::to_value(draft).unwrap_or(JsonValue::Null))
}

fn compact_reason_from(reason: &str) -> Option<CompactReason> {
	match reason {
		"manual" => Some(CompactReason::Manual),
		"threshold" => Some(CompactReason::Threshold),
		"overflow" => Some(CompactReason::Overflow),
		_ => None,
	}
}

// ============================================================================
// AgentSession
// ============================================================================

/// Core abstraction for agent lifecycle and session management
/// (`AgentSession`). Cheap to clone; all methods take `&self`.
#[derive(Clone)]
pub struct AgentSession {
	pub(crate) inner: Arc<AgentSessionInner>,
}

pub(crate) struct AgentSessionInner {
	/// Handle to self, set right after the outer `Arc` is created.
	self_weak: std::sync::Weak<AgentSessionInner>,
	pub(crate) agent: Arc<Agent>,
	pub(crate) session_manager: Arc<Mutex<SessionManager>>,
	pub(crate) settings_manager: Arc<Mutex<SettingsManager>>,

	scoped_models: Mutex<Vec<ScopedModelEntry>>,

	unsubscribe_agent: Mutex<Option<Unsubscribe>>,
	event_listeners: Mutex<Vec<AgentSessionEventListener>>,
	is_agent_run_active: std::sync::atomic::AtomicBool,
	agent_run_abort_requested: std::sync::atomic::AtomicBool,
	/// Watch channel over the idle state; `waitForIdle` awaits changes.
	idle_tx: tokio::sync::watch::Sender<bool>,

	/// Pending steering messages for UI display. Removed when delivered.
	steering_messages: Mutex<Vec<String>>,
	/// Pending follow-up messages for UI display. Removed when delivered.
	follow_up_messages: Mutex<Vec<String>>,
	/// Messages queued alongside the next user prompt ("asides").
	pending_next_turn_messages: Mutex<Vec<CustomMessage>>,
	/// Context-only custom messages queued during a run.
	pending_custom_messages: Mutex<Vec<CustomMessage>>,

	// Compaction state
	compaction_abort_controller: Mutex<Option<AbortController>>,
	auto_compaction_abort_controller: Mutex<Option<(u64, AbortController)>>,
	overflow_recovery_attempted: std::sync::atomic::AtomicBool,

	// Branch summarization state
	branch_summary_abort_controller: Mutex<Option<AbortController>>,

	// Retry state
	retry_abort_controller: Mutex<Option<AbortController>>,
	retry_attempt: std::sync::atomic::AtomicU32,
	/// Failed response that the next request repeats (upstream
	/// `_failedResponse`).
	failed_response: Mutex<Option<AssistantMessage>>,

	// Bash execution state (id-keyed: `AbortSignal` exposes no identity)
	bash_abort_controllers: Mutex<HashMap<u64, AbortController>>,
	next_abort_id: std::sync::atomic::AtomicU64,
	pending_bash_messages: Mutex<Vec<BashExecutionMessage>>,

	// Extension system
	extension_runner: parking_lot::RwLock<Option<Arc<ExtensionRunner>>>,
	turn_index: std::sync::atomic::AtomicU32,
	boundary_dispatched_messages: Mutex<HashSet<String>>,
	last_assistant_message: Mutex<Option<AssistantMessage>>,
	last_assistant_tool_results: Mutex<Vec<ToolResultMessage>>,
	last_activity_outcome: Mutex<AgentActivityOutcome>,
	is_before_settle: std::sync::atomic::AtomicBool,
	abort_during_before_settle: std::sync::atomic::AtomicBool,
	is_emitting_agent_settled: std::sync::atomic::AtomicBool,
	deferred_settled_actions: Mutex<Vec<BoxFuture<'static, ()>>>,

	resource_loader: Arc<Mutex<DefaultResourceLoader>>,
	custom_tools: Vec<ToolDefinition>,
	base_tool_definitions: Mutex<HashMap<String, ToolDefinition>>,
	cwd: String,
	extension_runner_ref: Option<ExtensionRunnerRef>,
	initial_active_tool_names: Option<Vec<String>>,
	/// Restored/reloaded tools that are not registered yet (upstream
	/// `_pendingToolNames`).
	pending_tool_names: Mutex<HashSet<String>>,
	uses_default_tools: bool,
	allowed_tool_names: Option<HashSet<String>>,
	excluded_tool_names: Option<HashSet<String>>,
	base_tools_override: Option<Vec<AgentTool>>,
	session_start_event: SessionStartEvent,
	extension_ui_context: Mutex<Option<Arc<dyn ExtensionUi>>>,
	extension_mode: Mutex<ExtensionMode>,
	extension_command_context_actions: Mutex<Option<ExtensionCommandContextActions>>,
	extension_abort_handler: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
	extension_shutdown_handler: Mutex<Option<ShutdownHandler>>,
	extension_error_listener: Mutex<Option<ExtensionErrorListener>>,
	extension_error_unsubscriber: Mutex<Option<Unsubscribe>>,

	model_runtime: Arc<ModelRuntime>,
	cache_warmer: Option<CacheWarmer>,

	// Tool registry for extension getTools/setTools
	tool_registry: Mutex<HashMap<String, AgentTool>>,
	/// Created on the first `ctx.execute_tool()` call.
	nested_tool_calls: Mutex<Option<Arc<NestedToolCallRunner>>>,
	/// Tools whose declarations `prepareLoadout` hooks hide from requests.
	hidden_declarations: Mutex<HashSet<String>>,
	tool_definitions: Mutex<HashMap<String, ToolDefinitionEntry>>,
	tool_prompt_snippets: Mutex<HashMap<String, String>>,
	tool_prompt_guidelines: Mutex<HashMap<String, Vec<String>>>,

	base_system_prompt_options: Mutex<Option<crate::core::extensions::types::NormalizedBuildSystemPromptOptions>>,
	/// Prompt options after before_agent_start mutations for the active run.
	run_system_prompt_options: Mutex<Option<crate::core::extensions::types::NormalizedBuildSystemPromptOptions>>,

	// D78: auto-compaction test seam (upstream: vi.spyOn on the private).
	run_auto_compaction_override:
		Mutex<Option<Arc<dyn Fn(CompactReason, bool) -> BoxFuture<'static, bool> + Send + Sync>>>,
	#[cfg(test)]
	run_auto_compaction_calls: Mutex<Vec<(String, bool)>>,
}

/// `Agent.state` surface used by callers (upstream `AgentSession.state`).
pub struct AgentStateSnapshot {
	pub model: Model,
	pub thinking_level: AgentThinkingLevel,
	pub tools: Vec<AgentTool>,
	pub messages: Vec<AgentMessage>,
}

impl AgentSession {
	/// Construct a session (`new AgentSession(config)`). The instance is
	/// returned behind an `Arc` because subscriptions and extension actions
	/// capture it.
	pub fn new(config: AgentSessionConfig) -> Arc<AgentSession> {
		let (idle_tx, _) = tokio::sync::watch::channel(false);
		let inner = Arc::new(AgentSessionInner {
			self_weak: std::sync::Weak::new(),
			agent: Arc::clone(&config.agent),
			session_manager: config.session_manager,
			settings_manager: config.settings_manager,
			scoped_models: Mutex::new(config.scoped_models),
			unsubscribe_agent: Mutex::new(None),
			event_listeners: Mutex::new(Vec::new()),
			is_agent_run_active: std::sync::atomic::AtomicBool::new(false),
			agent_run_abort_requested: std::sync::atomic::AtomicBool::new(false),
			idle_tx,
			steering_messages: Mutex::new(Vec::new()),
			follow_up_messages: Mutex::new(Vec::new()),
			pending_next_turn_messages: Mutex::new(Vec::new()),
			pending_custom_messages: Mutex::new(Vec::new()),
			compaction_abort_controller: Mutex::new(None),
			auto_compaction_abort_controller: Mutex::new(None),
			overflow_recovery_attempted: std::sync::atomic::AtomicBool::new(false),
			branch_summary_abort_controller: Mutex::new(None),
			retry_abort_controller: Mutex::new(None),
			retry_attempt: std::sync::atomic::AtomicU32::new(0),
			failed_response: Mutex::new(None),
			bash_abort_controllers: Mutex::new(HashMap::new()),
			next_abort_id: std::sync::atomic::AtomicU64::new(1),
			pending_bash_messages: Mutex::new(Vec::new()),
			extension_runner: parking_lot::RwLock::new(None),
			turn_index: std::sync::atomic::AtomicU32::new(0),
			boundary_dispatched_messages: Mutex::new(HashSet::new()),
			last_assistant_message: Mutex::new(None),
			last_assistant_tool_results: Mutex::new(Vec::new()),
			last_activity_outcome: Mutex::new(AgentActivityOutcome::Completed),
			is_before_settle: std::sync::atomic::AtomicBool::new(false),
			abort_during_before_settle: std::sync::atomic::AtomicBool::new(false),
			is_emitting_agent_settled: std::sync::atomic::AtomicBool::new(false),
			deferred_settled_actions: Mutex::new(Vec::new()),
			resource_loader: config.resource_loader,
			custom_tools: config.custom_tools,
			base_tool_definitions: Mutex::new(HashMap::new()),
			cwd: config.cwd,
			extension_runner_ref: config.extension_runner_ref,
			initial_active_tool_names: config.initial_active_tool_names,
			pending_tool_names: Mutex::new(HashSet::new()),
			uses_default_tools: config.uses_default_tools,
			allowed_tool_names: config.allowed_tool_names.map(|names| names.into_iter().collect()),
			excluded_tool_names: config.excluded_tool_names.map(|names| names.into_iter().collect()),
			base_tools_override: config.base_tools_override,
			session_start_event: config
				.session_start_event
				.unwrap_or(SessionStartEvent { reason: SessionStartReason::Startup, previous_session_file: None }),
			extension_ui_context: Mutex::new(None),
			extension_mode: Mutex::new(ExtensionMode::Print),
			extension_command_context_actions: Mutex::new(None),
			extension_abort_handler: Mutex::new(None),
			extension_shutdown_handler: Mutex::new(None),
			extension_error_listener: Mutex::new(None),
			extension_error_unsubscriber: Mutex::new(None),
			model_runtime: config.model_runtime,
			cache_warmer: config.cache_warmer,
			tool_registry: Mutex::new(HashMap::new()),
			nested_tool_calls: Mutex::new(None),
			hidden_declarations: Mutex::new(HashSet::new()),
			tool_definitions: Mutex::new(HashMap::new()),
			tool_prompt_snippets: Mutex::new(HashMap::new()),
			tool_prompt_guidelines: Mutex::new(HashMap::new()),
			base_system_prompt_options: Mutex::new(None),
			run_system_prompt_options: Mutex::new(None),
			run_auto_compaction_override: Mutex::new(None),
			#[cfg(test)]
			run_auto_compaction_calls: Mutex::new(Vec::new()),
		});

		let session = AgentSession { inner: Arc::clone(&inner) };
		// SAFETY: write-once of the self handle before the instance escapes;
		// no other thread can observe `inner` yet.
		let mut_self: &mut AgentSessionInner = unsafe { &mut *(Arc::as_ptr(&inner) as *mut AgentSessionInner) };
		mut_self.self_weak = Arc::downgrade(&inner);

		// Always subscribe to agent events for internal handling (session
		// persistence, extensions, auto-compaction, retry logic).
		let weak = Arc::downgrade(&inner);
		let unsubscribe = inner.agent.subscribe(move |event, _signal| {
			let weak = weak.clone();
			Box::pin(async move {
				if let Some(inner) = weak.upgrade() {
					inner.handle_agent_event(event).await;
				}
			})
		});
		*inner.unsubscribe_agent.lock() = Some(unsubscribe);

		inner.install_agent_tool_hooks();
		inner.install_agent_next_turn_refresh();
		inner.install_agent_request_projection();
		inner.install_agent_boundary_hooks();
		inner.install_hidden_declarations_projection();
		inner.install_agent_forced_prompt_projection();

		if let Some(cache_warmer) = &inner.cache_warmer {
			let weak = Arc::downgrade(&inner);
			cache_warmer.set_on_warmed(Box::new(move |entry| {
				if let Some(inner) = weak.upgrade() {
					inner.emit(AgentSessionEvent::EntryAppended { entry: SessionEntry::Usage(entry.clone()) });
				}
			}));
		}

		inner.build_runtime(BuildRuntimeOptions {
			active_tool_names: inner.initial_active_tool_names.clone(),
			flag_values: None,
			include_all_extension_tools: true,
		});
		if inner.initial_active_tool_names.is_none() {
			inner.restore_tools_from_transcript();
		}

		Arc::new(session)
	}

	/// The underlying agent (upstream public `agent` field).
	pub fn agent(&self) -> &Arc<Agent> {
		&self.inner.agent
	}

	/// The session manager (upstream public `sessionManager` field).
	pub fn session_manager(&self) -> &Arc<Mutex<SessionManager>> {
		&self.inner.session_manager
	}

	/// The settings manager (upstream public `settingsManager` field).
	pub fn settings_manager(&self) -> &Arc<Mutex<SettingsManager>> {
		&self.inner.settings_manager
	}

	/// Canonical model/auth runtime used by coding-agent internals.
	pub fn model_runtime(&self) -> &Arc<ModelRuntime> {
		&self.inner.model_runtime
	}

	/// The extension runner.
	pub fn extension_runner(&self) -> Arc<ExtensionRunner> {
		self.inner.extension_runner()
	}

	// =========================================================================
	// Event Subscription
	// =========================================================================

	/// Subscribe to agent events. Session persistence is handled internally
	/// (saves messages on message_end). Returns an unsubscribe function for
	/// this listener.
	pub fn subscribe(&self, listener: impl Fn(AgentSessionEvent) + Send + Sync + 'static) -> Unsubscribe {
		let listener: AgentSessionEventListener = Arc::new(listener);
		self.inner.event_listeners.lock().push(Arc::clone(&listener));
		let weak = Arc::downgrade(&self.inner);
		Box::new(move || {
			if let Some(inner) = weak.upgrade() {
				inner.event_listeners.lock().retain(|existing| !Arc::ptr_eq(existing, &listener));
			}
		})
	}

	/// Remove all listeners and disconnect from the agent. Call this when
	/// completely done with the session.
	pub fn dispose(&self) {
		let inner = &self.inner;
		// Dispose must succeed even if an abort hook throws: each step is
		// independent and cannot panic (abort only flips flags/signals).
		inner.abort_retry();
		inner.abort_compaction();
		inner.abort_branch_summary();
		inner.abort_bash();
		inner.agent.abort();

		if let Some(runner) = inner.extension_runner.read().clone() {
			runner.invalidate(Some(
				"This extension ctx is stale after session replacement or reload. Do not use a captured pi or command ctx after ctx.newSession(), ctx.fork(), ctx.switchSession(), or ctx.reload(). For newSession, fork, and switchSession, move post-replacement work into withSession and use the ctx passed to withSession. For reload, do not use the old ctx after await ctx.reload().".to_string(),
			));
		}
		if let Some(unsubscribe) = inner.unsubscribe_agent.lock().take() {
			unsubscribe();
		}
		*inner.event_listeners.lock() = Vec::new();
		if let Some(cache_warmer) = &inner.cache_warmer {
			cache_warmer.set_on_warmed(Box::new(|_| {}));
			cache_warmer.cancel();
		}
		let session_id = inner.session_manager.lock().get_session_id().to_string();
		let _ = pi_ai::session_resources::cleanup_session_resources(Some(session_id.as_str()));
	}

	// =========================================================================
	// Read-only State Access
	// =========================================================================

	/// Refresh the public finalized transcript from the canonical session
	/// projection.
	pub fn refresh_context(&self) {
		self.inner.refresh_finalized_context();
	}

	/// Full agent state snapshot (upstream `get state()`).
	pub fn state(&self) -> AgentStateSnapshot {
		let state = self.inner.agent.lock_state();
		AgentStateSnapshot {
			model: state.model.clone(),
			thinking_level: state.thinking_level,
			tools: state.tools.clone(),
			messages: state.messages.clone(),
		}
	}

	/// Current cache-warming state and the policy inputs that produced it.
	pub fn cache_warming_status(&self) -> Option<CacheWarmingStatus> {
		self.inner.cache_warmer.as_ref().map(|warmer| warmer.status())
	}

	/// Persist the cache-warming mode and immediately reconcile active
	/// warming.
	pub fn set_cache_warming_mode(&self, mode: CacheWarmingMode) {
		self.inner.settings_manager.lock().set_cache_warming_mode(mode);
		if let Some(cache_warmer) = &self.inner.cache_warmer {
			cache_warmer.on_mode_changed();
		}
	}

	/// Current model (may be `None` if not yet selected).
	pub fn model(&self) -> Option<Model> {
		Some(self.inner.agent.model()).filter(|model| !model.base.id.is_empty())
	}

	/// Re-resolve the active model from the model registry
	/// (`_refreshCurrentModelFromRegistry`; the SDK create path calls this
	/// after applying extension provider registrations).
	pub fn refresh_current_model_from_registry(&self) {
		self.inner.refresh_current_model_from_registry();
	}

	/// Current thinking level.
	pub fn thinking_level(&self) -> AgentThinkingLevel {
		self.inner.agent.thinking_level()
	}

	/// Under a virtual selection, the physical model and thinking level of
	/// the latest successful response.
	pub fn routed_model(&self) -> Option<(Model, Option<AgentThinkingLevel>)> {
		let model = self.model()?;
		if !is_virtual_model(&model) {
			return None;
		}
		let llm: Vec<Message> = self
			.inner
			.agent
			.messages()
			.iter()
			.filter_map(|message| message.as_llm().cloned())
			.collect();
		let latest = find_latest_response(&llm)?;
		let physical = self.inner.model_runtime.get_physical_model(&latest.provider, &latest.model)?;
		Some((physical, latest.thinking_level.map(model_level_to_agent_level)))
	}

	/// Whether the session is currently processing an agent run or post-run
	/// continuation.
	pub fn is_streaming(&self) -> bool {
		self.inner.is_agent_run_active.load(std::sync::atomic::Ordering::SeqCst)
	}

	/// Whether the session has no active agent run, compaction, branch
	/// summary, retry, or queued continuation.
	pub fn is_idle(&self) -> bool {
		!self.is_streaming() && !self.is_compacting()
	}

	/// Current effective system prompt, including changes not yet sent to the
	/// model.
	pub fn system_prompt(&self) -> String {
		let inner = &self.inner;
		let options = inner
			.run_system_prompt_options
			.lock()
			.clone()
			.or_else(|| inner.base_system_prompt_options.lock().clone());
		options
			.map(|options| build_system_prompt_normalized(&options))
			.unwrap_or_default()
	}

	/// Current retry attempt (0 if not retrying).
	pub fn retry_attempt(&self) -> u32 {
		self.inner.retry_attempt.load(std::sync::atomic::Ordering::SeqCst)
	}

	/// Names of currently active tools — the tools declared to the model.
	pub fn get_active_tool_names(&self) -> Vec<String> {
		self.inner.agent.tools().iter().map(|tool| tool.name().to_string()).collect()
	}

	/// Names of the tools tools can call through `ctx.executeTool()`.
	pub fn get_callable_tool_names(&self) -> Vec<String> {
		self.inner.get_callable_tools(None).iter().map(|tool| tool.name().to_string()).collect()
	}

	/// All configured tools with name, description, parameter schema, prompt
	/// guidelines, and source metadata.
	pub fn get_all_tools(&self) -> Vec<ToolInfo> {
		// Snapshot under one lock: `get_tool_exposure` re-locks
		// `tool_definitions`, so calling it while holding the guard
		// self-deadlocks (parking_lot Mutex is not reentrant).
		let entries: Vec<(ToolDefinition, crate::core::source_info::SourceInfo)> = self
			.inner
			.tool_definitions
			.lock()
			.values()
			.map(|entry| (entry.definition.clone(), entry.source_info.clone()))
			.collect();
		entries
			.into_iter()
			.map(|(definition, source_info)| ToolInfo {
				name: definition.name.clone(),
				description: definition.description.clone(),
				parameters: definition.parameters.clone(),
				prompt_guidelines: definition.prompt_guidelines.clone(),
				exposure: definition.exposure.unwrap_or(ToolExposure::Direct),
				namespace: definition.namespace.clone(),
				annotations: definition.annotations,
				source_info,
			})
			.collect()
	}

	/// The erased tool definition registered under `name`, if any.
	pub fn get_tool_definition(&self, name: &str) -> Option<ToolDefinition> {
		self.inner.tool_definitions.lock().get(name).map(|entry| entry.definition.clone())
	}

	/// Set active tools by name. Only tools in the registry can be enabled.
	/// Unknown and hidden tool names are ignored. Also rebuilds the system
	/// prompt to reflect the new tool set. Changes take effect on the next
	/// agent turn.
	pub fn set_active_tools_by_name(&self, tool_names: Vec<String>) {
		let previous = self.get_active_tool_names();
		self.inner.set_active_tools(tool_names);
		// A loadout that deactivates a tool replaces the restored one, whose
		// pending tools are dropped. One that only adds tools, like activating
		// tool_search, keeps them.
		let active: HashSet<String> = self.get_active_tool_names().into_iter().collect();
		if previous.iter().any(|name| !active.contains(name)) {
			self.inner.pending_tool_names.lock().clear();
		}
	}

	/// Whether compaction or branch summarization is currently running.
	pub fn is_compacting(&self) -> bool {
		self.inner.compaction_abort_controller.lock().is_some()
			|| self.inner.auto_compaction_abort_controller.lock().is_some()
			|| self.inner.branch_summary_abort_controller.lock().is_some()
	}

	/// All messages including custom types like bash executions.
	pub fn messages(&self) -> Vec<AgentMessage> {
		self.inner.agent.messages()
	}

	/// Current steering mode.
	pub fn steering_mode(&self) -> QueueMode {
		self.inner.agent.steering_mode()
	}

	/// Current follow-up mode.
	pub fn follow_up_mode(&self) -> QueueMode {
		self.inner.agent.follow_up_mode()
	}

	/// Current session file path, or `None` if sessions are disabled.
	pub fn session_file(&self) -> Option<String> {
		self.inner.session_manager.lock().get_session_file().map(str::to_string)
	}

	/// Current session ID.
	pub fn session_id(&self) -> String {
		self.inner.session_manager.lock().get_session_id().to_string()
	}

	/// Current session display name, if set.
	pub fn session_name(&self) -> Option<String> {
		self.inner.session_manager.lock().get_session_name()
	}

	/// Scoped models for cycling (from `--models` flag).
	pub fn scoped_models(&self) -> Vec<ScopedModelEntry> {
		self.inner.scoped_models.lock().clone()
	}

	/// Update scoped models for cycling.
	pub fn set_scoped_models(&self, scoped_models: Vec<ScopedModelEntry>) {
		*self.inner.scoped_models.lock() = scoped_models;
	}

	/// File-based prompt templates.
	pub fn prompt_templates(&self) -> Vec<PromptTemplate> {
		self.inner.resource_loader.lock().get_prompts().items
	}

	// =========================================================================
	// Prompting
	// =========================================================================

	/// Send a prompt to the agent (`prompt`). Handles extension commands,
	/// expands skill commands and prompt templates by default, queues via
	/// `steer`/`followUp` while streaming, and validates model + auth before
	/// sending when not streaming.
	pub async fn prompt(&self, text: impl Into<String>, options: Option<PromptOptions>) -> Result<(), String> {
		let options = options.unwrap_or_default();
		if self.inner.is_emitting_agent_settled.load(std::sync::atomic::Ordering::SeqCst) {
			let session = self.clone();
			let text = text.into();
			self.inner.deferred_settled_actions.lock().push(Box::pin(async move {
				let _ = boxed_prompt(session, text, Some(options)).await;
			}));
			return Ok(());
		}
		let inner = &self.inner;
		let expand_prompt_templates = options.expand_prompt_templates.unwrap_or(true);
		let text = text.into();
		let preflight_result = options.preflight_result.clone();
		let call_preflight = |disposition: PromptDisposition| {
			if let Some(preflight) = &preflight_result {
				preflight(disposition);
			}
		};

		// Handle extension commands first (execute immediately, even during
		// streaming). Extension commands manage their own LLM interaction via
		// `pi.sendMessage()`.
		if expand_prompt_templates && text.starts_with('/') {
			if inner.try_execute_extension_command(&text).await {
				call_preflight(PromptDisposition::Handled);
				return Ok(());
			}
		}

		if inner.compaction_abort_controller.lock().is_some() {
			return Err(
				"Cannot submit a prompt while compaction is in progress. Wait for compaction to finish and retry."
					.to_string(),
			);
		}

		// Emit input event for extension interception (before skill/template
		// expansion).
		let processed_input = inner
			.run_input_handlers(
				text,
				options.images.clone(),
				options.source.unwrap_or(InputSource::Interactive),
				if self.is_streaming() { options.streaming_behavior } else { None },
			)
			.await;
		let Some(processed_input) = processed_input else {
			call_preflight(PromptDisposition::Handled);
			return Ok(());
		};
		let (current_text, current_images) = processed_input;

		// Expand skill commands (/skill:name args) and prompt templates
		// (/template args).
		let mut expanded_text = current_text;
		if expand_prompt_templates {
			expanded_text = inner.expand_skill_command(&expanded_text);
			expanded_text = expand_prompt_template(&expanded_text, &self.prompt_templates());
		}

		// If streaming, queue via steer() or followUp() based on option.
		if self.is_streaming() {
			let Some(streaming_behavior) = options.streaming_behavior else {
				return Err(
					"Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') to queue the message."
						.to_string(),
				);
			};
			let images = if current_images.is_empty() { None } else { Some(current_images.clone()) };
			if streaming_behavior == StreamingBehavior::FollowUp {
				inner.queue_follow_up(expanded_text, images).await;
			} else {
				inner.queue_steer(expanded_text, images).await;
			}
			call_preflight(PromptDisposition::Queued);
			return Ok(());
		}

		// Flush any pending bash and custom messages before the new prompt.
		inner.flush_pending_bash_messages();
		inner.flush_pending_custom_messages();

		// Validate model.
		let Some(model) = self.model() else {
			return Err(format_no_model_selected_message());
		};

		let has_configured_auth = inner.model_runtime.has_configured_auth(model.provider())
			|| inner.model_runtime.get_provider_auth_status(model.provider()).configured;
		if !has_configured_auth {
			if inner.model_runtime.is_using_oauth(model.provider()) {
				return Err(format!(
					"Authentication failed for \"{}\". Credentials may have expired or network is unavailable. Run '/login {}' to re-authenticate.",
					model.provider(),
					model.provider()
				));
			}
			return Err(format_no_api_key_found_message(model.provider()));
		}

		// Check if we need to compact before sending (catches aborted
		// responses). The user's new prompt is sent below, so do not call
		// `agent.continue_()` here.
		if let Some(last_assistant) = inner.find_last_assistant_message() {
			inner.check_compaction(&last_assistant, false, Vec::new()).await;
		}

		// Emit before_agent_start before normalizing images so
		// extension-driven model selection determines the resize profile used
		// for the request and history.
		let selected_tools_before = inner
			.base_system_prompt_options
			.lock()
			.clone()
			.map(|options| options.selected_tools)
			.unwrap_or_default();
		let runner = inner.extension_runner();
		let build_options = build_options_from_normalized(
			&inner
				.base_system_prompt_options
				.lock()
				.clone()
				.unwrap_or_else(empty_normalized_prompt_options),
		);
		let result = runner
			.emit_before_agent_start(
				&expanded_text,
				if current_images.is_empty() { None } else { Some(current_images.clone()) },
				&build_options,
			)
			.await
			.map_err(|error| format!("before_agent_start handler failed: {error}"))?;
		let mut system_prompt_options = result.system_prompt_options;
		// Handlers may edit event.systemPromptOptions.selectedTools or call
		// setActiveTools(), which updates the live loadout instead. An
		// explicit edit wins; otherwise the live loadout is authoritative, so
		// a setActiveTools() call is not undone here.
		let handler_edited_tools = system_prompt_options.selected_tools.len() != selected_tools_before.len()
			|| system_prompt_options
				.selected_tools
				.iter()
				.zip(selected_tools_before.iter())
				.any(|(name, before)| name != before);
		if !handler_edited_tools {
			system_prompt_options.selected_tools = self.get_active_tool_names();
		}

		let normalized = inner.normalize_prompt_images(&current_images).await;
		let user_text = if normalized.hints.is_empty() {
			expanded_text
		} else {
			format!("{}\n\n{}", expanded_text, normalized.hints.join("\n"))
		};

		// Build messages only after hooks and image normalization completed.
		let mut messages: Vec<AgentMessage> = Vec::new();
		let mut user_content: Vec<UserContent> = vec![UserContent::Text(TextContent::new(user_text))];
		user_content.extend(normalized.images.into_iter().map(UserContent::Image));
		messages.push(AgentMessage::user(pi_ai::types::UserMessage {
			content: UserContentField::Blocks(user_content),
			timestamp: crate_now_ms(),
		}));

		// Inject any pending "nextTurn" messages as context alongside the
		// user message.
		for message in inner.pending_next_turn_messages.lock().drain(..) {
			messages.push(custom_message_to_agent_message(&message));
		}

		for message in &result.messages {
			messages.push(custom_draft_to_agent_message(message));
		}
		let update_message = inner.prepare_prompt_and_tool_loadout(&mut system_prompt_options, None);
		*inner.run_system_prompt_options.lock() = Some(system_prompt_options);
		if let Some(update_message) = update_message {
			messages.insert(0, AgentMessage::system(update_message));
		}

		call_preflight(PromptDisposition::Started);
		inner.run_agent_prompt(messages).await;
		Ok(())
	}

	/// Try to execute an extension command. Returns true if a command was
	/// found and executed.
	pub async fn try_execute_extension_command(&self, text: &str) -> bool {
		self.inner.try_execute_extension_command(text).await
	}

	/// Queue a steering message while the agent is running. Delivered after
	/// the current assistant turn finishes executing its tool calls, before
	/// the next LLM call. Expands skill commands and prompt templates.
	/// Errors on extension commands.
	pub async fn steer(
		&self,
		text: impl Into<String>,
		images: Option<Vec<ImageContent>>,
		source: Option<InputSource>,
	) -> Result<QueuedInputDisposition, String> {
		self.inner
			.queue_user_input(text.into(), images, StreamingBehavior::Steer, source.unwrap_or(InputSource::Interactive))
			.await
	}

	/// Queue a follow-up message to be processed after the agent finishes.
	/// Delivered only when the agent has no more tool calls or steering
	/// messages. Expands skill commands and prompt templates. Errors on
	/// extension commands.
	pub async fn follow_up(
		&self,
		text: impl Into<String>,
		images: Option<Vec<ImageContent>>,
		source: Option<InputSource>,
	) -> Result<QueuedInputDisposition, String> {
		self.inner
			.queue_user_input(text.into(), images, StreamingBehavior::FollowUp, source.unwrap_or(InputSource::Interactive))
			.await
	}

	/// Send a custom message to the session. Creates a
	/// `CustomMessageEntry` (`sendCustomMessage`).
	pub async fn send_custom_message(
		&self,
		message: CustomMessageDraft,
		options: Option<SendMessageOptions>,
	) -> Result<(), String> {
		let inner = &self.inner;
		let mut app_message = CustomMessage {
			role: crate::core::messages::CustomRole::Custom,
			custom_type: message.custom_type,
			content: message.content,
			display: message.display,
			details: message.details,
			timestamp: crate_now_ms(),
		};
		let options = options.unwrap_or_default();
		if options.deliver_as == Some(DeliveryTarget::NextTurn) {
			inner.pending_next_turn_messages.lock().push(app_message);
		} else if self.is_streaming() && options.trigger_turn != Some(false) {
			let agent_message = custom_message_to_agent_message(&app_message);
			if options.deliver_as == Some(DeliveryTarget::FollowUp) {
				inner.agent.follow_up(agent_message);
			} else {
				inner.agent.steer(agent_message);
			}
		} else if options.trigger_turn == Some(true) {
			let message = custom_message_to_agent_message(&app_message);
			if inner.is_emitting_agent_settled.load(std::sync::atomic::Ordering::SeqCst) {
				let deferred_inner = Arc::clone(&self.inner);
				inner
					.deferred_settled_actions
					.lock()
					.push(Box::pin(async move {
						deferred_inner.run_agent_prompt(vec![message]).await;
					}));
				return Ok(());
			}
			inner.run_agent_prompt(vec![message]).await;
		} else if self.is_streaming() {
			// Appending now would put the message between an assistant tool
			// call and its result, which providers that validate message
			// order reject on replay. Defer to the end of the turn.
			inner.pending_custom_messages.lock().push(app_message);
		} else {
			inner.append_custom_message(&mut app_message);
		}
		Ok(())
	}

	/// Send a user message to the agent. Always triggers a turn
	/// (`sendUserMessage`).
	pub async fn send_user_message(
		&self,
		content: crate::core::extensions::types::SendUserMessageContent,
		options: Option<crate::core::extensions::types::SendUserMessageOptions>,
	) -> Result<(), String> {
		let options = options.unwrap_or_default();
		let deliver_as = options.deliver_as.map(|target| match target {
			crate::core::extensions::types::DeliveryTarget::Steer => StreamingBehavior::Steer,
			crate::core::extensions::types::DeliveryTarget::FollowUp => StreamingBehavior::FollowUp,
			crate::core::extensions::types::DeliveryTarget::NextTurn => StreamingBehavior::Steer,
		});
		// Normalize content to text string + optional images.
		let (text, images) = match content {
			crate::core::extensions::types::SendUserMessageContent::Text(text) => (text, None),
			crate::core::extensions::types::SendUserMessageContent::Blocks(parts) => {
				let mut text_parts: Vec<String> = Vec::new();
				let mut images: Vec<ImageContent> = Vec::new();
				for part in parts {
					match part {
						UserContent::Text(text) => text_parts.push(text.text),
						UserContent::Image(image) => images.push(image),
						_ => {}
					}
				}
				let text = text_parts.join("\n");
				let images = if images.is_empty() { None } else { Some(images) };
				(text, images)
			}
		};

		self.prompt(
			text,
			Some(PromptOptions {
				expand_prompt_templates: Some(options.expand_prompt_templates.unwrap_or(false)),
				streaming_behavior: deliver_as,
				images,
				source: Some(InputSource::Extension),
				..PromptOptions::default()
			}),
		)
		.await
	}

	/// Clear all queued messages and return them. Useful for restoring to the
	/// editor when the user aborts.
	pub fn clear_queue(&self) -> (Vec<String>, Vec<String>) {
		let inner = &self.inner;
		let steering = inner.steering_messages.lock().drain(..).collect();
		let follow_up = inner.follow_up_messages.lock().drain(..).collect();
		inner.agent.clear_all_queues();
		inner.emit_queue_update();
		(steering, follow_up)
	}

	/// Number of pending messages (includes both steering and follow-up).
	pub fn pending_message_count(&self) -> usize {
		let inner = &self.inner;
		inner.steering_messages.lock().len() + inner.follow_up_messages.lock().len()
	}

	/// Pending steering messages.
	pub fn get_steering_messages(&self) -> Vec<String> {
		self.inner.steering_messages.lock().clone()
	}

	/// Pending follow-up messages.
	pub fn get_follow_up_messages(&self) -> Vec<String> {
		self.inner.follow_up_messages.lock().clone()
	}

	/// The resource loader.
	pub fn resource_loader(&self) -> Arc<Mutex<DefaultResourceLoader>> {
		Arc::clone(&self.inner.resource_loader)
	}

	/// Abort the current operation and wait for the agent to become idle.
	pub async fn abort(&self) {
		let inner = &self.inner;
		if self.is_streaming() {
			inner.agent_run_abort_requested.store(true, std::sync::atomic::Ordering::SeqCst);
		}
		inner.abort_retry();
		inner.abort_compaction();
		inner.abort_branch_summary();
		if inner.is_before_settle.load(std::sync::atomic::Ordering::SeqCst) {
			inner.abort_during_before_settle.store(true, std::sync::atomic::Ordering::SeqCst);
		}
		inner.agent.abort();
		self.wait_for_idle().await;
	}

	/// Resolve when the session is idle (no agent run, compaction, retry, or
	/// branch summary in flight).
	pub async fn wait_for_idle(&self) {
		let mut rx = self.inner.idle_tx.subscribe();
		while !self.is_idle() {
			if rx.changed().await.is_err() {
				return;
			}
		}
	}

	// =========================================================================
	// Model Management
	// =========================================================================

	/// Set model directly. Validates that auth is configured and saves to the
	/// session transcript. Persists to global defaults only when
	/// `options.persist` is true.
	pub async fn set_model(&self, model: Model, options: ModelMutationOptions) -> Result<(), String> {
		let inner = &self.inner;
		let status = inner.model_runtime.get_provider_auth_status(model.provider());
		if !status.configured {
			return Err(format!("No API key for {}/{}", model.provider(), model.id()));
		}

		let previous_model = self.model();
		let thinking_level = self.get_thinking_level_for_model_switch(Some(&model), None);
		inner.agent.set_model(model.clone());
		inner
			.session_manager
			.lock()
			.append_model_change(model.provider(), model.id())
			.map_err(|error| error.to_string())?;
		if options.persist {
			inner.settings_manager.lock().set_default_model_and_provider(model.provider(), model.id());
			self.add_persisted_default_to_non_empty_scope(&model);
		}

		// Apply thinking level for the new model. Per-model thinking level
		// overrides take priority over the global default. Model persistence
		// does not implicitly rewrite the global thinking default.
		self.set_thinking_level(thinking_level, options);

		self.inner.emit_model_select(model, previous_model, ModelSelectSource::Set).await;
		Ok(())
	}

	fn add_persisted_default_to_non_empty_scope(&self, model: &Model) {
		let mut scoped = self.inner.scoped_models.lock();
		if scoped.is_empty() {
			return;
		}
		if scoped.iter().any(|entry| models_are_equal(Some(&AnyModel::Chat(entry.model.clone())), Some(&AnyModel::Chat(model.clone())))) {
			return;
		}
		scoped.push(ScopedModelEntry { model: model.clone(), thinking_level: None });
		drop(scoped);

		let enabled_models = self.inner.settings_manager.lock().get_enabled_models();
		let Some(enabled_models) = enabled_models else { return };
		if enabled_models.is_empty() {
			return;
		}
		let model_reference = format!("{}/{}", model.provider(), model.id());
		if enabled_models.iter().any(|pattern| pattern.to_lowercase() == model_reference.to_lowercase()) {
			return;
		}
		let mut next = enabled_models;
		next.push(model_reference);
		self.inner.settings_manager.lock().set_enabled_models(Some(&next));
	}

	/// Cycle to next/previous model. Uses scoped models (from `--models`
	/// flag) if available, otherwise all available models.
	pub async fn cycle_model(
		&self,
		direction: CycleDirection,
		options: ModelMutationOptions,
	) -> Result<Option<ModelCycleResult>, String> {
		if !self.inner.scoped_models.lock().is_empty() {
			self.cycle_scoped_model(direction, options).await
		} else {
			self.cycle_available_model(direction, options).await
		}
	}

	async fn cycle_scoped_model(
		&self,
		direction: CycleDirection,
		options: ModelMutationOptions,
	) -> Result<Option<ModelCycleResult>, String> {
		let inner = &self.inner;
		let available: HashSet<String> = inner
			.model_runtime
			.get_available_snapshot()
			.into_iter()
			.map(|model| format!("{}\u{0}{}", model.provider(), model.id()))
			.collect();
		let all_scoped = inner.scoped_models.lock().clone();
		let scoped_models: Vec<ScopedModelEntry> = all_scoped
			.into_iter()
			.filter(|scoped| available.contains(&format!("{}\u{0}{}", scoped.model.provider(), scoped.model.id())))
			.collect();
		if scoped_models.len() <= 1 {
			return Ok(None);
		}

		let current_model = self.model();
		let mut current_index = scoped_models
			.iter()
			.position(|scoped| Some(scoped.model.clone()) == current_model)
			.unwrap_or(0);
		let len = scoped_models.len();
		let next_index = match direction {
			CycleDirection::Forward => (current_index + 1) % len,
			CycleDirection::Backward => (current_index + len - 1) % len,
		};
		let _ = &mut current_index;
		let next = &scoped_models[next_index];
		let thinking_level = self.get_thinking_level_for_model_switch(Some(&next.model), next.thinking_level);

		// Apply model.
		inner.agent.set_model(next.model.clone());
		inner
			.session_manager
			.lock()
			.append_model_change(next.model.provider(), next.model.id())
			.map_err(|error| error.to_string())?;
		if options.persist {
			inner
				.settings_manager
				.lock()
				.set_default_model_and_provider(next.model.provider(), next.model.id());
			self.add_persisted_default_to_non_empty_scope(&next.model);
		}

		// Apply thinking level for the new model: an explicit scoped model
		// thinking level overrides defaults; per-model overrides take
		// priority over the global default. `set_thinking_level` clamps to
		// model capabilities. Model persistence does not implicitly rewrite
		// the global thinking default.
		self.set_thinking_level(thinking_level, options);

		self.inner.emit_model_select(next.model.clone(), current_model, ModelSelectSource::Cycle).await;

		Ok(Some(ModelCycleResult { model: next.model.clone(), thinking_level: self.thinking_level(), is_scoped: true }))
	}

	async fn cycle_available_model(
		&self,
		direction: CycleDirection,
		options: ModelMutationOptions,
	) -> Result<Option<ModelCycleResult>, String> {
		let inner = &self.inner;
		let available_models = inner.model_runtime.get_available_snapshot();
		if available_models.len() <= 1 {
			return Ok(None);
		}

		let current_model = self.model();
		let current_index = available_models
			.iter()
			.position(|model| Some(model.clone()) == current_model)
			.unwrap_or(0);
		let len = available_models.len();
		let next_index = match direction {
			CycleDirection::Forward => (current_index + 1) % len,
			CycleDirection::Backward => (current_index + len - 1) % len,
		};
		let next_model = available_models[next_index].clone();

		let thinking_level = self.get_thinking_level_for_model_switch(Some(&next_model), None);
		inner.agent.set_model(next_model.clone());
		inner
			.session_manager
			.lock()
			.append_model_change(next_model.provider(), next_model.id())
			.map_err(|error| error.to_string())?;
		if options.persist {
			inner.settings_manager.lock().set_default_model_and_provider(next_model.provider(), next_model.id());
			self.add_persisted_default_to_non_empty_scope(&next_model);
		}

		// Apply thinking level for the new model. Model persistence does not
		// implicitly rewrite the global thinking default.
		self.set_thinking_level(thinking_level, options);

		self.inner.emit_model_select(next_model.clone(), current_model, ModelSelectSource::Cycle).await;

		Ok(Some(ModelCycleResult { model: next_model, thinking_level: self.thinking_level(), is_scoped: false }))
	}

	// =========================================================================
	// Thinking Level Management
	// =========================================================================

	/// Set thinking level. Clamps to model capabilities based on available
	/// thinking levels. Saves the clamped level to the session transcript
	/// only if the level actually changes. Persists the requested level to
	/// global defaults only when `options.persist` is true.
	pub fn set_thinking_level(&self, level: AgentThinkingLevel, options: ModelMutationOptions) {
		let inner = &self.inner;
		let available_levels = self.get_available_thinking_levels();
		let effective_level = if available_levels.contains(&level) {
			level
		} else {
			inner.clamp_thinking_level(level, &available_levels)
		};

		// Only persist if actually changing.
		let previous_level = inner.agent.thinking_level();
		let is_changing = effective_level != previous_level;

		inner.agent.set_thinking_level(effective_level);

		if options.persist {
			inner.settings_manager.lock().set_default_thinking_level(level);
		}

		if is_changing {
			let _ = inner
				.session_manager
				.lock()
				.append_thinking_level_change(&thinking_level_str(effective_level));
			inner.emit(AgentSessionEvent::ThinkingLevelChanged { level: effective_level });
			let runner = inner.extension_runner();
			let event = json!({
				"type": "thinking_level_select",
				"level": effective_level,
				"previousLevel": previous_level,
			});
			// Fire-and-forget like upstream `void this._extensionRunner.emit(...)`.
			tokio::spawn(async move {
				let _ = runner.emit("thinking_level_select", event).await;
			});
		}
	}

	/// Cycle to next thinking level. Returns `None` if the model doesn't
	/// support thinking.
	pub fn cycle_thinking_level(&self, options: ModelMutationOptions) -> Option<AgentThinkingLevel> {
		if !self.supports_thinking() {
			return None;
		}

		let levels = self.get_available_thinking_levels();
		let current_index = levels.iter().position(|level| *level == self.thinking_level()).unwrap_or(0);
		let next_level = levels[(current_index + 1) % levels.len()];

		self.set_thinking_level(next_level, options);
		Some(next_level)
	}

	/// Available thinking levels for current model. The provider clamps to
	/// what the specific model supports internally.
	pub fn get_available_thinking_levels(&self) -> Vec<AgentThinkingLevel> {
		let Some(model) = self.model() else {
			return THINKING_LEVEL_OPTIONS.to_vec();
		};
		get_supported_thinking_levels(&model)
			.into_iter()
			.map(model_level_to_agent_level)
			.collect()
	}

	/// Whether the current model supports thinking/reasoning.
	pub fn supports_thinking(&self) -> bool {
		self.model().map(|model| model.reasoning).unwrap_or(false)
	}

	fn get_thinking_level_for_model_switch(
		&self,
		target_model: Option<&Model>,
		explicit_level: Option<AgentThinkingLevel>,
	) -> AgentThinkingLevel {
		if let Some(explicit_level) = explicit_level {
			return explicit_level;
		}
		// Per-model default takes priority when switching to a model that has
		// one.
		if let Some(target_model) = target_model {
			let per_model = self
				.inner
				.settings_manager
				.lock()
				.get_model_thinking_level(target_model.provider(), target_model.id());
			if let Some(per_model) = per_model {
				return per_model;
			}
		}
		self.inner
			.settings_manager
			.lock()
			.get_default_thinking_level()
			.or(Some(self.thinking_level()))
			.unwrap_or(DEFAULT_THINKING_LEVEL)
	}

	// =========================================================================
	// Queue Mode Management
	// =========================================================================

	fn sync_queue_modes_from_settings(&self) {
		let settings = self.inner.settings_manager.lock();
		let steering = steering_mode_to_queue(settings.get_steering_mode());
		let follow_up = steering_mode_to_queue(settings.get_follow_up_mode());
		drop(settings);
		self.inner.agent.set_steering_mode(steering);
		self.inner.agent.set_follow_up_mode(follow_up);
	}

	/// Set steering message mode. Saves to settings.
	pub fn set_steering_mode(&self, mode: QueueMode) {
		self.inner.agent.set_steering_mode(mode);
		self.inner.settings_manager.lock().set_steering_mode(queue_to_steering_mode(mode));
	}

	/// Set follow-up message mode. Saves to settings.
	pub fn set_follow_up_mode(&self, mode: QueueMode) {
		self.inner.agent.set_follow_up_mode(mode);
		self.inner.settings_manager.lock().set_follow_up_mode(queue_to_steering_mode(mode));
	}

	// =========================================================================
	// Compaction
	// =========================================================================

	/// Manually compact the session context (`compact`). Used by `/compact`,
	/// RPC, and extensions. Aborts the current agent operation first; manual
	/// compaction never retries or continues the interrupted agent turn.
	pub async fn compact(&self, custom_instructions: Option<String>) -> Result<CompactionResult, String> {
		let inner = &self.inner;
		self.abort().await;
		let controller = AbortController::new();
		*inner.compaction_abort_controller.lock() = Some(controller.clone());
		inner.emit(AgentSessionEvent::CompactionStart { reason: CompactReason::Manual });
		let mut from_extension = false;
		let mut cancelled_by_extension = false;

		let result = async {
			let Some(model) = self.model() else {
				return Err(format_no_model_selected_message());
			};

			let settings = inner.compaction_settings(Some(&model))?;
			let path_entries = inner.session_manager.lock().get_branch();

			let Some(preparation) = prepare_compaction(&path_entries, &settings) else {
				// Check why we can't compact.
				let last_entry = path_entries.last();
				if matches!(last_entry, Some(SessionEntry::Compaction(_))) {
					return Err("Already compacted".to_string());
				}
				return Err("Nothing to compact (session too small)".to_string());
			};

			let mut extension_compaction: Option<CompactionResult> = None;

			let runner = inner.extension_runner();
			if runner.has_handlers("session_before_compact") {
				let event = SessionBeforeCompactEvent {
					preparation: preparation.clone(),
					branch_entries: path_entries.clone(),
					custom_instructions: custom_instructions.clone(),
					reason: CompactReason::Manual,
					will_retry: false,
					signal: controller.signal().clone(),
				};
				let dispatch = runner.emit_session_before_compact(&event).await;
				if dispatch.cancel {
					cancelled_by_extension = true;
					return Err("Compaction cancelled".to_string());
				}
				if let Some(raw) = dispatch.raw {
					if let Ok(result) = serde_json::from_value::<SessionBeforeCompactRawResult>(raw) {
						if result.cancel.unwrap_or(false) {
							cancelled_by_extension = true;
							return Err("Compaction cancelled".to_string());
						}
						if let Some(compaction) = result.compaction {
							extension_compaction = Some(compaction.into_compaction_result());
							from_extension = true;
						}
					}
				}
			}

			let compaction_content = if let Some(extension_compaction) = extension_compaction {
				// Extension provided compaction content.
				extension_compaction
			} else {
				// Shared default summary generator, also used by automatic
				// compaction.
				let signal = bridge_crate_signal_to_ai(Some(controller.signal().clone()));
				inner
					.run_default_compaction(&preparation, &model, custom_instructions.as_deref(), signal.as_ref(), CompactReason::Manual)
					.await?
			};
			let summary = compaction_content.summary;
			let first_kept_entry_id = compaction_content.first_kept_entry_id;
			let tokens_before = compaction_content.tokens_before;
			let usage = compaction_content.usage;
			let details = compaction_content.details.map(|details| serde_json::to_value(details).unwrap_or(JsonValue::Null));

			if controller.signal().aborted() {
				return Err("Compaction cancelled".to_string());
			}

			inner
				.session_manager
				.lock()
				.append_compaction(&summary, Some(&first_kept_entry_id), tokens_before, details.clone(), Some(from_extension), usage.clone())
				.map_err(|error| error.to_string())?;
			let new_entries = inner.session_manager.lock().get_entries();
			inner.refresh_finalized_context();
			let estimated_tokens_after =
				estimate_messages_tokens(&inner.session_manager.lock().build_session_projection().messages);

			// Get the saved compaction entry for the extension event.
			let saved_compaction_entry =
				new_entries.iter().find_map(|entry| match entry {
					SessionEntry::Compaction(compaction) if compaction.summary == summary => Some(compaction.clone()),
					_ => None,
				});

			if let Some(saved_compaction_entry) = saved_compaction_entry {
				let event = json!({
					"type": "session_compact",
					"compactionEntry": saved_compaction_entry,
					"fromExtension": from_extension,
					"reason": "manual",
					"willRetry": false,
				});
				let _ = runner.emit("session_compact", event).await;
			}

			let compaction_result = CompactionResult {
				summary,
				first_kept_entry_id,
				tokens_before,
				estimated_tokens_after: Some(estimated_tokens_after),
				usage,
				details: details.and_then(|details| serde_json::from_value(details).ok()),
			};
			Ok(compaction_result)
		}
		.await;

		// compaction_end listeners may submit queued prompts, so expose idle
		// state before notifying them.
		inner.clear_manual_compaction_state();
		match result {
			Ok(compaction_result) => {
				inner.emit(AgentSessionEvent::CompactionEnd {
					reason: CompactReason::Manual,
					result: Some(compaction_result.clone()),
					aborted: false,
					will_retry: false,
					error_message: None,
				});
				Ok(compaction_result)
			}
			Err(error) => {
				let aborted = controller.signal().aborted() || cancelled_by_extension;
				let error_message = if aborted { None } else { Some(format!("Compaction failed: {error}")) };
				inner.emit(AgentSessionEvent::CompactionEnd {
					reason: CompactReason::Manual,
					result: None,
					aborted,
					will_retry: false,
					error_message: error_message.clone(),
				});
				let runner = inner.extension_runner();
				let failed = crate::core::extensions::types::SessionCompactFailedEvent {
					reason: CompactReason::Manual,
					error_message: error_message.clone(),
					aborted,
					will_retry: false,
					from_extension,
				};
				let _ = runner.emit("session_compact_failed", serde_json::to_value(failed).unwrap_or(JsonValue::Null)).await;
				Err(error)
			}
		}
	}

	/// Cancel in-progress compaction (manual or auto).
	pub fn abort_compaction(&self) {
		self.inner.abort_compaction();
	}

	/// Cancel in-progress branch summarization.
	pub fn abort_branch_summary(&self) {
		self.inner.abort_branch_summary();
	}

	/// Toggle auto-compaction setting.
	pub fn set_auto_compaction_enabled(&self, enabled: bool) {
		self.inner.settings_manager.lock().set_compaction_enabled(enabled);
	}

	/// Whether auto-compaction is enabled.
	pub fn auto_compaction_enabled(&self) -> bool {
		self.inner.settings_manager.lock().get_compaction_enabled()
	}

	// =========================================================================
	// Extension lifecycle
	// =========================================================================

	/// Bind extension runtime bindings and emit the session start event
	/// (`bindExtensions`).
	pub async fn bind_extensions(&self, bindings: ExtensionBindings) -> Result<(), String> {
		let inner = &self.inner;
		if bindings.ui_context.is_some() {
			*inner.extension_ui_context.lock() = bindings.ui_context;
		}
		if bindings.mode.is_some() {
			*inner.extension_mode.lock() = bindings.mode.unwrap();
		}
		if bindings.command_context_actions.is_some() {
			*inner.extension_command_context_actions.lock() = bindings.command_context_actions;
		}
		if bindings.abort_handler.is_some() {
			*inner.extension_abort_handler.lock() = bindings.abort_handler;
		}
		if bindings.shutdown_handler.is_some() {
			*inner.extension_shutdown_handler.lock() = bindings.shutdown_handler;
		}
		if bindings.on_error.is_some() {
			*inner.extension_error_listener.lock() = bindings.on_error;
		}

		inner.apply_extension_bindings(&inner.extension_runner());
		let runner = inner.extension_runner();
		runner
			.emit("session_start", serde_json::to_value(inner.session_start_event.clone()).unwrap_or(JsonValue::Null))
			.await;
		runner.report_unhandled_mcp_servers();
		self.extend_resources_from_extensions(match inner.session_start_event.reason {
			SessionStartReason::Reload => "reload",
			_ => "startup",
		})
		.await
	}

	async fn extend_resources_from_extensions(&self, reason: &str) -> Result<(), String> {
		let inner = &self.inner;
		let runner = inner.extension_runner();
		if !runner.has_handlers("resources_discover") {
			return Ok(());
		}

		let reason = match reason {
			"reload" => crate::core::extensions::types::ResourcesDiscoverReason::Reload,
			_ => crate::core::extensions::types::ResourcesDiscoverReason::Startup,
		};
		let outcome = runner.emit_resources_discover(&inner.cwd, reason).await;
		if outcome.skill_paths.is_empty() && outcome.prompt_paths.is_empty() && outcome.theme_paths.is_empty() {
			return Ok(());
		}

		let extension_paths = ResourceExtensionPaths {
			skill_paths: self.build_extension_resource_paths(&outcome.skill_paths),
			prompt_paths: self.build_extension_resource_paths(&outcome.prompt_paths),
			theme_paths: self.build_extension_resource_paths(&outcome.theme_paths),
		};

		inner.resource_loader.lock().extend_resources(extension_paths);
		let active = self.get_active_tool_names();
		inner.rebuild_system_prompt(&active);
		Ok(())
	}

	fn build_extension_resource_paths(
		&self,
		entries: &[(String, String)],
	) -> Vec<crate::core::resource_loader::ResourcePathEntry> {
		entries
			.iter()
			.map(|(path, extension_path)| crate::core::resource_loader::ResourcePathEntry {
				path: path.clone(),
				metadata: crate::core::source_info::PathMetadata {
					source: self.get_extension_source_label(extension_path),
					scope: crate::core::source_info::SourceScope::Temporary,
					origin: crate::core::source_info::SourceOrigin::TopLevel,
					base_dir: if is_synthetic_path(extension_path) {
						None
					} else {
						Some(
							std::path::Path::new(extension_path)
								.parent()
								.map(|parent| parent.to_string_lossy().to_string())
								.unwrap_or_default(),
						)
					},
					package_root: None,
				},
			})
			.collect()
	}

	fn get_extension_source_label(&self, extension_path: &str) -> String {
		if is_synthetic_path(extension_path) {
			return format!("extension:{}", extension_path.replace(['<', '>'], ""));
		}
		let base = std::path::Path::new(extension_path)
			.file_name()
			.map(|name| name.to_string_lossy().to_string())
			.unwrap_or_default();
		let name = base
			.strip_suffix(".ts")
			.or_else(|| base.strip_suffix(".js"))
			.unwrap_or(&base);
		format!("extension:{name}")
	}

	/// Reload extensions, settings, skills, prompts, themes, and context
	/// files (`reload`).
	pub async fn reload(&self, before_session_start: Option<BoxFuture<'static, ()>>) -> Result<(), String> {
		let inner = &self.inner;
		let old_runner = inner.extension_runner();
		let previous_flag_values = old_runner.get_flag_values();
		let _ = emit_session_shutdown_event(
			&old_runner,
			&crate::core::extensions::types::SessionShutdownEvent {
				reason: crate::core::extensions::types::SessionShutdownReason::Reload,
				target_session_file: None,
			},
		)
		.await;
		old_runner.invalidate(None);
		let previous_default_tools: HashSet<String> = if inner.uses_default_tools {
			inner
				.settings_manager
				.lock()
				.get_default_tools()
				.unwrap_or_else(|| DEFAULT_TOOL_NAMES.iter().map(|name| name.to_string()).collect())
				.into_iter()
				.collect()
		} else {
			HashSet::new()
		};
		inner.settings_manager.lock().reload();
		self.sync_queue_modes_from_settings();
		pi_ai::compat::reset_api_providers();
		inner.resource_loader.lock().reload(ResourceLoaderReloadOptions::default());
		// Activate tools newly added to defaultTools. Removed ones stay
		// active, and tools disabled during the session stay disabled unless
		// the setting newly adds them.
		let added_default_tools: Vec<String> = if inner.uses_default_tools {
			inner
				.settings_manager
				.lock()
				.get_default_tools()
				.unwrap_or_else(|| DEFAULT_TOOL_NAMES.iter().map(|name| name.to_string()).collect())
				.into_iter()
				.filter(|name| !previous_default_tools.contains(name))
				.collect()
		} else {
			Vec::new()
		};
		// Tools the new extensions register later, such as MCP tools, are
		// pending until then.
		{
			let mut pending = inner.pending_tool_names.lock();
			for name in self.get_active_tool_names() {
				pending.insert(name);
			}
		}
		inner.build_runtime(BuildRuntimeOptions {
			active_tool_names: Some(
				self.get_active_tool_names().into_iter().chain(added_default_tools).collect(),
			),
			flag_values: Some(previous_flag_values),
			include_all_extension_tools: true,
		});

		let has_bindings = inner.extension_ui_context.lock().is_some()
			|| inner.extension_command_context_actions.lock().is_some()
			|| inner.extension_shutdown_handler.lock().is_some()
			|| inner.extension_error_listener.lock().is_some();
		if has_bindings {
			if let Some(before_session_start) = before_session_start {
				before_session_start.await;
			}
			let runner = inner.extension_runner();
			runner
				.emit(
					"session_start",
					serde_json::to_value(SessionStartEvent {
						reason: SessionStartReason::Reload,
						previous_session_file: None,
					})
					.unwrap_or(JsonValue::Null),
				)
				.await;
			runner.report_unhandled_mcp_servers();
			self.extend_resources_from_extensions("reload").await?;
		}
		Ok(())
	}

	// =========================================================================
	// Auto-Retry
	// =========================================================================

	/// Whether auto-retry is currently in progress.
	pub fn is_retrying(&self) -> bool {
		self.inner.retry_abort_controller.lock().is_some()
	}

	/// Whether auto-retry is enabled.
	pub fn auto_retry_enabled(&self) -> bool {
		self.inner.settings_manager.lock().get_retry_enabled()
	}

	/// Toggle auto-retry setting.
	pub fn set_auto_retry_enabled(&self, enabled: bool) {
		self.inner.settings_manager.lock().set_retry_enabled(enabled);
	}

	/// Abort an in-progress retry (`AgentSession.abortRetry`).
	pub fn abort_retry(&self) {
		self.inner.abort_retry();
	}

	// =========================================================================
	// Bash Execution
	// =========================================================================

	/// Execute a bash command. Adds the result to agent context and session
	/// (`executeBash`).
	pub async fn execute_bash(
		&self,
		command: impl Into<String>,
		on_chunk: Option<Arc<dyn Fn(&str) + Send + Sync>>,
		options: Option<ExecuteBashOptions>,
	) -> Result<BashResult, String> {
		let inner = &self.inner;
		let options = options.unwrap_or_default();
		let controller = AbortController::new();
		let controller_id = inner
			.next_abort_id
			.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
		inner.bash_abort_controllers.lock().insert(controller_id, controller.clone());

		// Apply command prefix if configured (e.g. `shopt -s expand_aliases`
		// for alias support).
		let (prefix, shell_path) = {
			let settings = inner.settings_manager.lock();
			(settings.get_shell_command_prefix(), settings.get_shell_path())
		};
		let command = command.into();
		let resolved_command = match &prefix {
			Some(prefix) => format!("{prefix}\n{command}"),
			None => command.clone(),
		};

		let operations = options.operations.clone().unwrap_or_else(|| create_local_bash_operations(shell_path.clone()));
		let emit_update = {
			let inner = Arc::clone(&self.inner);
			let id = options.id.clone();
			Arc::new(move |delta: &str| {
				if let Some(on_chunk) = &on_chunk {
					on_chunk(&delta);
				}
				inner.emit(AgentSessionEvent::BashExecutionUpdate { id: id.clone(), delta: delta.to_string() });
			})
		};

		let cwd = inner.session_manager.lock().get_cwd().to_string();
		let result = execute_bash_with_operations(
			&resolved_command,
			&cwd,
			operations,
			Some(crate::core::bash_executor::BashExecutorOptions {
				on_chunk: Some(emit_update),
				signal: Some(controller.signal().clone()),
			}),
		)
		.await;

		inner.bash_abort_controllers.lock().remove(&controller_id);
		let result = result.map_err(|error| error)?;
		self.record_bash_result(&command, &result, options.exclude_from_context);
		Ok(result)
	}

	/// Record a bash execution result in session history. Used by
	/// `execute_bash` and by extensions that handle bash execution
	/// themselves.
	pub fn record_bash_result(&self, command: &str, result: &BashResult, exclude_from_context: Option<bool>) {
		let inner = &self.inner;
		let bash_message = BashExecutionMessage {
			role: crate::core::messages::BashExecutionRole::BashExecution,
			command: command.to_string(),
			output: result.output.clone(),
			exit_code: result.exit_code.map(i64::from),
			cancelled: result.cancelled,
			truncated: result.truncated,
			full_output_path: result.full_output_path.clone(),
			timestamp: crate_now_ms(),
			exclude_from_context,
		};

		// If the agent is streaming, defer adding to avoid breaking
		// tool_use/tool_result ordering.
		if self.is_streaming() {
			inner.pending_bash_messages.lock().push(bash_message);
		} else {
			append_bash_message(&inner, bash_message);
			inner.refresh_finalized_context();
		}
	}

	/// Cancel running bash commands.
	pub fn abort_bash(&self) {
		self.inner.abort_bash();
	}

	/// Whether a bash command is currently running.
	pub fn is_bash_running(&self) -> bool {
		!self.inner.bash_abort_controllers.lock().is_empty()
	}

	/// Whether bash messages are waiting to be flushed.
	pub fn has_pending_bash_messages(&self) -> bool {
		!self.inner.pending_bash_messages.lock().is_empty()
	}

	// =========================================================================
	// Session Management
	// =========================================================================

	/// Set a display name for the current session.
	pub fn set_session_name(&self, name: &str) {
		let inner = &self.inner;
		let _ = inner.session_manager.lock().append_session_info(name);
		let event = AgentSessionEvent::SessionInfoChanged { name: self.session_name() };
		inner.emit(event.clone());
		let runner = inner.extension_runner();
		tokio::spawn(async move {
			if let AgentSessionEvent::SessionInfoChanged { name } = event {
				let _ = runner
					.emit("session_info_changed", json!({ "type": "session_info_changed", "name": name }))
					.await;
			}
		});
	}

	// =========================================================================
	// Tree Navigation
	// =========================================================================

	/// Navigate to a different node in the session tree (`navigateTree`).
	/// Unlike fork, this stays in the same file.
	pub async fn navigate_tree(
		&self,
		target_id: &str,
		options: Option<NavigateTreeOptions>,
	) -> Result<NavigateTreeResult, String> {
		let inner = &self.inner;
		let options = options.unwrap_or_default();
		if self.is_streaming() {
			return Err("Wait for the current response to finish before navigating the session tree.".to_string());
		}
		if self.is_compacting() {
			return Err(
				"Wait for the current compaction or tree navigation to finish before navigating the session tree."
					.to_string(),
			);
		}

		let old_leaf_id = inner.session_manager.lock().get_leaf_id().map(str::to_string);

		// No-op if already at target.
		if Some(target_id) == old_leaf_id.as_deref() {
			return Ok(NavigateTreeResult { editor_text: None, cancelled: false, aborted: false, summary_entry: None });
		}

		// Model required for summarization.
		if options.summarize.unwrap_or(false) && self.model().is_none() {
			return Err("No model available for summarization".to_string());
		}

		let target_entry = inner
			.session_manager
			.lock()
			.get_entry(target_id)
			.cloned()
			.ok_or_else(|| format!("Entry {target_id} not found"))?;

		// Collect entries to summarize (from old leaf to common ancestor).
		let collected = collect_entries_for_branch_summary(&inner.session_manager.lock(), old_leaf_id.as_deref(), target_id);
		let entries_to_summarize = collected.entries;
		let common_ancestor_id = collected.common_ancestor_id;

		// Prepare event data — mutable so extensions can override.
		let mut custom_instructions = options.custom_instructions.clone();
		let mut replace_instructions = options.replace_instructions;
		let mut label = options.label.clone();

		let preparation = TreePreparation {
			target_id: target_id.to_string(),
			old_leaf_id: old_leaf_id.clone(),
			common_ancestor_id,
			entries_to_summarize: entries_to_summarize.clone(),
			user_wants_summary: options.summarize.unwrap_or(false),
			custom_instructions: custom_instructions.clone(),
			replace_instructions,
			label: label.clone(),
		};

		// Set up abort controller for summarization.
		let controller = AbortController::new();
		*inner.branch_summary_abort_controller.lock() = Some(controller.clone());

		let result = async {
			let mut extension_summary: Option<TreeSummaryOverrideValue> = None;
			let mut from_extension = false;

			// Emit session_before_tree event.
			let runner = inner.extension_runner();
			if runner.has_handlers("session_before_tree") {
				let event = SessionBeforeTreeEvent {
					preparation: preparation.clone(),
					signal: controller.signal().clone(),
				};
				let result = runner.emit_session_before_tree(&event).await;
				if let Some(result) = result {
					if result.cancel.unwrap_or(false) {
						return Ok(NavigateTreeResult {
							editor_text: None,
							cancelled: true,
							aborted: false,
							summary_entry: None,
						});
					}
					if result.summary.is_some() && options.summarize.unwrap_or(false) {
						extension_summary = result.summary.clone();
						from_extension = true;
					}
					// Allow extensions to override instructions and label.
					if result.custom_instructions.is_some() {
						custom_instructions = result.custom_instructions.clone();
					}
					if result.replace_instructions.is_some() {
						replace_instructions = result.replace_instructions;
					}
					if result.label.is_some() {
						label = result.label.clone();
					}
				}
			}

			// Run default summarizer if needed.
			let mut summary_text: Option<String> = None;
			let mut summary_details: Option<JsonValue> = None;
			let mut summary_usage: Option<Usage> = None;
			if options.summarize.unwrap_or(false) && !entries_to_summarize.is_empty() && extension_summary.is_none() {
				let model = self.model().expect("checked above");
				let signal = bridge_crate_signal_to_ai_value(controller.signal());
				let auth = inner.get_summarization_request_auth(&model, Some(&signal)).await?;
				let branch_summary_settings = inner.settings_manager.lock().get_branch_summary_reserve_tokens();
				let result = generate_branch_summary(
					&entries_to_summarize,
					crate::core::compaction::GenerateBranchSummaryOptions {
						model: auth.model,
						api_key: auth.api_key.as_deref(),
						headers: auth.headers.as_ref(),
						env: auth.env.as_ref(),
						signal: signal.clone(),
						custom_instructions: custom_instructions.as_deref(),
						replace_instructions: replace_instructions.unwrap_or(false),
						reserve_tokens: Some(branch_summary_settings),
						stream_fn: Some(Arc::clone(&inner.agent.stream_function)),
						retry: Some(&inner.retry_policy()),
						callbacks: Some(&inner.summarization_retry_callbacks(SummarizationRetrySource::BranchSummary)),
					},
				)
				.await;
				if result.aborted {
					return Ok(NavigateTreeResult {
						editor_text: None,
						cancelled: true,
						aborted: true,
						summary_entry: None,
					});
				}
				if let Some(error) = result.error {
					return Err(error);
				}
				summary_text = result.summary.clone();
				summary_usage = result.usage;
				summary_details = Some(json!({
					"readFiles": result.read_files,
					"modifiedFiles": result.modified_files,
				}));
			} else if let Some(extension_summary) = extension_summary {
				summary_text = Some(extension_summary.summary);
				summary_details = extension_summary.details;
				summary_usage = extension_summary.usage;
			}

			// Determine the new leaf position based on target type.
			let (new_leaf_id, editor_text): (Option<String>, Option<String>) = match &target_entry {
				SessionEntry::Message(message_entry) => {
					let message = &message_entry.message;
					if let AgentMessage::Llm(Message::User(user)) = message {
						// User message: leaf = parent (None if root), text
						// goes to the editor.
						(
							message_entry.base.parent_id.clone(),
							Some(user_content_text_for_editor(&user.content)),
						)
					} else {
						// Non-user message: leaf = selected node.
						(Some(target_id.to_string()), None)
					}
				}
				SessionEntry::CustomMessage(custom_entry) => (
					// Custom message: leaf = parent (None if root), text goes
					// to the editor.
					custom_entry.base.parent_id.clone(),
					Some(user_content_field_text_for_editor(&custom_entry.content)),
				),
				_ => (Some(target_id.to_string()), None),
			};

			// Switch leaf (with or without summary). The summary is attached
			// at the navigation target position (newLeafId), not the old
			// branch.
			let mut summary_entry: Option<BranchSummaryEntry> = None;
			if let Some(summary_text) = summary_text.clone() {
				// Create summary at target position (None for root).
				let summary_id = inner
					.session_manager
					.lock()
					.branch_with_summary(
						new_leaf_id.as_deref(),
						&summary_text,
						summary_details.clone(),
						Some(from_extension),
						summary_usage,
					)
					.map_err(|error| error.to_string())?;
				summary_entry = match inner.session_manager.lock().get_entry(&summary_id) {
					Some(SessionEntry::BranchSummary(entry)) => Some(entry.clone()),
					_ => None,
				};

				// Attach label to the summary entry.
				if let Some(label) = &label {
					let _ = inner.session_manager.lock().append_label_change(&summary_id, Some(label.clone()));
				}
			} else if new_leaf_id.is_none() {
				// No summary, navigating to root — reset leaf.
				inner.session_manager.lock().reset_leaf();
			} else {
				// No summary, navigating to non-root.
				inner
					.session_manager
					.lock()
					.branch(new_leaf_id.as_deref().expect("non-root leaf"))
					.map_err(|error| error.to_string())?;
			}

			// Attach label to the target entry when not summarizing (no
			// summary entry to label).
			if let Some(label) = &label {
				if summary_text.is_none() {
					let _ = inner.session_manager.lock().append_label_change(target_id, Some(label.clone()));
				}
			}

			// Update finalized context from the canonical session projection.
			inner.refresh_finalized_context();
			inner.restore_tools_from_transcript();

			// Emit session_tree event.
			let new_leaf = inner.session_manager.lock().get_leaf_id().map(str::to_string);
			let event = json!({
				"type": "session_tree",
				"newLeafId": new_leaf,
				"oldLeafId": old_leaf_id,
				"summaryEntry": summary_entry,
				"fromExtension": if summary_text.is_some() { Some(from_extension) } else { None },
			});
			let _ = runner.emit("session_tree", event).await;

			Ok(NavigateTreeResult { editor_text, cancelled: false, aborted: false, summary_entry })
		}
		.await;

		*inner.branch_summary_abort_controller.lock() = None;
		inner.update_idle_watch();
		result
	}

	/// All user messages from the session, for the fork selector.
	pub fn get_user_messages_for_forking(&self) -> Vec<(String, String)> {
		let inner = &self.inner;
		let entries = inner.session_manager.lock().get_entries();
		let mut result = Vec::new();
		for entry in entries {
			let SessionEntry::Message(message_entry) = &entry else { continue };
			let AgentMessage::Llm(Message::User(user)) = &message_entry.message else { continue };
			let text = user_content_text_for_editor(&user.content);
			if !text.is_empty() {
				result.push((entry.id().to_string(), text));
			}
		}
		result
	}

	// =========================================================================
	// Stats
	// =========================================================================

	/// Session statistics. Aggregates over ALL session entries (including
	/// history compacted away), so token/cost totals reflect what was
	/// actually billed across the session.
	pub fn get_session_stats(&self) -> SessionStats {
		let inner = &self.inner;
		let mut user_messages = 0usize;
		let mut assistant_messages = 0usize;
		let mut tool_results = 0usize;
		let mut total_messages = 0usize;
		let mut tool_calls = 0usize;
		let mut usage_totals = create_usage_totals();

		for entry in inner.session_manager.lock().get_entries() {
			match &entry {
				SessionEntry::Usage(usage_entry) => {
					add_usage_to_totals(&mut usage_totals, &usage_entry.usage);
				}
				SessionEntry::BranchSummary(branch) => {
					if let Some(usage) = &branch.usage {
						add_usage_to_totals(&mut usage_totals, usage);
					}
				}
				SessionEntry::Compaction(compaction) => {
					if let Some(usage) = &compaction.usage {
						add_usage_to_totals(&mut usage_totals, usage);
					}
				}
				_ => {}
			}
			let SessionEntry::Message(message_entry) = &entry else { continue };
			total_messages += 1;
			match &message_entry.message {
				AgentMessage::Llm(Message::User(_)) => user_messages += 1,
				AgentMessage::Llm(Message::ToolResult(tool_result)) => {
					tool_results += 1;
					if let Some(usage) = &tool_result.usage {
						add_usage_to_totals(&mut usage_totals, usage);
					}
				}
				AgentMessage::Llm(Message::Assistant(assistant)) => {
					assistant_messages += 1;
					tool_calls += assistant.content.iter().filter(|block| matches!(block, pi_ai::types::AssistantContent::ToolCall(_))).count();
					add_usage_to_totals(&mut usage_totals, &assistant.usage);
				}
				_ => {}
			}
		}

		let tokens = SessionTokens {
			input: usage_totals.input,
			output: usage_totals.output,
			cache_read: usage_totals.cache_read,
			cache_write: usage_totals.cache_write,
			total: usage_totals.input + usage_totals.output + usage_totals.cache_read + usage_totals.cache_write,
		};
		SessionStats {
			session_file: self.session_file(),
			session_id: self.session_id(),
			user_messages,
			assistant_messages,
			tool_calls,
			tool_results,
			total_messages,
			tokens,
			cost: usage_totals.cost,
			context_usage: self.get_context_usage(),
		}
	}

	/// Estimated context usage against the limits model
	/// (`getContextUsage`).
	pub fn get_context_usage(&self) -> Option<ContextUsage> {
		let inner = &self.inner;
		let model = inner.limits_model()?;
		let context_window = model.context_window;
		if context_window == 0 {
			return None;
		}

		// After compaction, the last assistant usage reflects pre-compaction
		// context size. We can only trust usage from an assistant that
		// responded after the latest compaction. If no such assistant exists,
		// context token count is unknown until the next LLM response.
		let projection = inner.session_manager.lock().build_session_projection();
		let branch = inner.session_manager.lock().get_branch();
		let latest_compaction = get_latest_compaction_entry(&branch);

		if let Some(latest_compaction) = latest_compaction {
			let projected_assistants: HashSet<String> = projection
				.entries
				.iter()
				.filter(|entry| {
					entry.messages.iter().any(|message| {
						matches!(message, AgentMessage::Llm(Message::Assistant(assistant))
							if assistant.stop_reason != pi_ai::types::StopReason::Aborted
								&& assistant.stop_reason != pi_ai::types::StopReason::Error
								&& calculate_context_tokens(&assistant.usage) > 0)
					})
				})
				.map(|entry| entry.source_entry.id().to_string())
				.collect();
			let compaction_index = branch.iter().position(|entry| entry.id() == latest_compaction.base.id);
			let has_post_compaction_usage = match compaction_index {
				Some(index) => branch[index + 1..].iter().any(|entry| projected_assistants.contains(entry.id())),
				None => false,
			};
			if !has_post_compaction_usage {
				return Some(ContextUsage { tokens: None, context_window, percent: None });
			}
		}

		// Upstream: the last assistant usage is the authoritative context
		// size; the projection estimate is only the fallback (e.g. right
		// after a user turn, before the next response).
		let last_assistant_usage = branch.iter().rev().find_map(|entry| match entry {
			crate::core::session_manager::SessionEntry::Message(message_entry) => match &message_entry.message {
				AgentMessage::Llm(Message::Assistant(assistant))
					if assistant.stop_reason != pi_ai::types::StopReason::Aborted
						&& assistant.stop_reason != pi_ai::types::StopReason::Error =>
				{
					Some(&assistant.usage)
				}
				_ => None,
			},
			_ => None,
		});
		let tokens = match last_assistant_usage {
			Some(usage) if calculate_context_tokens(usage) > 0 => calculate_context_tokens(usage),
			_ => estimate_projected_context_tokens(&projection, &branch).tokens,
		};
		let percent = (tokens as f64 / context_window as f64) * 100.0;
		Some(ContextUsage { tokens: Some(tokens), context_window, percent: Some(percent) })
	}

	// =========================================================================
	// Export & Utilities
	// =========================================================================

	/// Export session to HTML (upstream `AgentSession.exportToHtml`; the
	/// `export-html/` module is now ported — see [`core::export_html`]).
	pub async fn export_to_html(&self, output_path: Option<&str>, theme_name: Option<&str>) -> Result<String, String> {
		let snapshot = self.state();
		super::export_html::export_session_to_html(
			&self.inner.session_manager.lock(),
			Some(&snapshot),
			super::export_html::ExportOptions {
				output_path: output_path.map(str::to_string),
				theme_name: theme_name.map(str::to_string),
				tool_renderer: None,
			},
		)
	}

	/// Export the current session branch to a JSONL file. Writes the session
	/// header followed by all entries on the current branch path.
	pub fn export_to_jsonl(&self, output_path: Option<&str>) -> Result<String, String> {
		let manager = self.inner.session_manager.lock();
		export_session_to_jsonl(&manager, output_path, None).map_err(|error| error.to_string())
	}

	/// Ask the current model to describe what went wrong in this session for
	/// a bug report (D79: `bug-report.ts` is not ported yet).
	pub async fn summarize_for_bug_report(&self, _hint: Option<&str>, _signal: Option<&AiAbortSignal>) -> Result<String, String> {
		let _ = self;
		Err("bug_report summary is not ported yet (upstream bug-report.ts defers; D79)".to_string())
	}

	/// Text content of the last assistant message (`getLastAssistantText`).
	/// Useful for the /copy command.
	pub fn get_last_assistant_text(&self) -> Option<String> {
		let messages = self.messages();
		let last_assistant = messages.into_iter().rev().find_map(|message| match message {
			AgentMessage::Llm(Message::Assistant(assistant)) => {
				// Skip aborted messages with no content.
				if assistant.stop_reason == pi_ai::types::StopReason::Aborted && assistant.content.is_empty() {
					None
				} else {
					Some(assistant)
				}
			}
			_ => None,
		})?;

		let mut text = String::new();
		for content in &last_assistant.content {
			if let pi_ai::types::AssistantContent::Text(block) = content {
				text.push_str(&block.text);
			}
		}
		let trimmed = text.trim().to_string();
		if trimmed.is_empty() {
			None
		} else {
			Some(trimmed)
		}
	}

	// =========================================================================
	// Extension System
	// =========================================================================

	/// Fresh command context whose `sendMessage`/`sendUserMessage` route to
	/// this session (`createReplacedSessionContext`).
	pub fn create_replaced_session_context(&self) -> Result<crate::core::extensions::types::ReplacedSessionContext, String> {
		let inner = &self.inner;
		let context = inner.extension_runner().create_command_context()?;
		let session = self.clone();
		let send_message: crate::core::extensions::types::SendMessageHandler = Arc::new(move |message, options| {
			let session = session.clone();
			let options = options.map(|options| SendMessageOptions {
				trigger_turn: options.trigger_turn,
				deliver_as: options.deliver_as.map(|target| match target {
					crate::core::extensions::types::DeliveryTarget::Steer => DeliveryTarget::Steer,
					crate::core::extensions::types::DeliveryTarget::FollowUp => DeliveryTarget::FollowUp,
					crate::core::extensions::types::DeliveryTarget::NextTurn => DeliveryTarget::NextTurn,
				}),
			});
			Box::pin(async move {
				let _ = session.send_custom_message(message, options).await;
			})
		});
		let session = self.clone();
		let send_user_message: crate::core::extensions::types::SendUserMessageHandler = Arc::new(move |content, options| {
			let session = session.clone();
			Box::pin(async move {
				let _ = session.send_user_message(content, options).await;
			})
		});
		Ok(crate::core::extensions::types::ReplacedSessionContext {
			ctx: context,
			send_message,
			send_user_message,
		})
	}

	/// Whether extensions have handlers for a specific event type.
	pub fn has_extension_handlers(&self, event_type: &str) -> bool {
		self.inner.extension_runner().has_handlers(event_type)
	}
}

// ============================================================================
// Supporting option/result types
// ============================================================================

/// Options of [`AgentSession::execute_bash`].
#[derive(Default)]
pub struct ExecuteBashOptions {
	/// If true, command output won't be sent to the LLM (`!!` prefix).
	pub exclude_from_context: Option<bool>,
	/// Identifier included in bash execution update events.
	pub id: Option<String>,
	/// Custom bash operations for remote execution.
	pub operations: Option<Arc<dyn BashOperations>>,
}

/// Options of [`AgentSession::send_custom_message`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SendMessageOptions {
	pub trigger_turn: Option<bool>,
	pub deliver_as: Option<DeliveryTarget>,
}

/// Where a queued message is delivered (`"steer" | "followUp" | "nextTurn"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryTarget {
	Steer,
	FollowUp,
	NextTurn,
}

/// Options of [`AgentSession::navigate_tree`].
#[derive(Debug, Clone, Default)]
pub struct NavigateTreeOptions {
	pub summarize: Option<bool>,
	pub custom_instructions: Option<String>,
	pub replace_instructions: Option<bool>,
	pub label: Option<String>,
}

/// Result of [`AgentSession::navigate_tree`].
#[derive(Debug, Clone, Default)]
pub struct NavigateTreeResult {
	pub editor_text: Option<String>,
	pub cancelled: bool,
	pub aborted: bool,
	pub summary_entry: Option<BranchSummaryEntry>,
}

/// Model cycling direction (`"forward" | "backward"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleDirection {
	Forward,
	Backward,
}

/// Model selection event source (`"set" | "cycle" | "restore"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSelectSource {
	Set,
	Cycle,
	Restore,
}

/// Wire shape of a `session_before_compact` handler result
/// (`SessionBeforeCompactResult`).
#[derive(Debug, Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SessionBeforeCompactRawResult {
	cancel: Option<bool>,
	compaction: Option<SessionBeforeCompactRawCompaction>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SessionBeforeCompactRawCompaction {
	summary: String,
	first_kept_entry_id: String,
	tokens_before: u64,
	usage: Option<Usage>,
	details: Option<JsonValue>,
}

impl SessionBeforeCompactRawCompaction {
	fn into_compaction_result(self) -> CompactionResult {
		CompactionResult {
			summary: self.summary,
			first_kept_entry_id: self.first_kept_entry_id,
			tokens_before: self.tokens_before,
			estimated_tokens_after: None,
			usage: self.usage,
			details: self.details.and_then(|details| serde_json::from_value(details).ok()),
		}
	}
}

type TreeSummaryOverrideValue = crate::core::extensions::types::TreeSummaryOverride;

fn user_content_text_for_editor(content: &UserContentField) -> String {
	match content {
		UserContentField::Text(text) => text.clone(),
		UserContentField::Blocks(blocks) => blocks
			.iter()
			.filter_map(|block| match block {
				UserContent::Text(text) => Some(text.text.clone()),
				_ => None,
			})
			.collect::<Vec<_>>()
			.join("\n"),
	}
}

fn custom_message_to_agent_message(message: &CustomMessage) -> AgentMessage {
	custom_message_to_wire(message)
}

fn custom_message_to_wire(message: &CustomMessage) -> AgentMessage {
	// Custom messages travel as opaque JSON on the transcript (pi-coding-agent D4).
	serde_json::from_value(
		serde_json::to_value(CustomMessage {
			role: message.role,
			custom_type: message.custom_type.clone(),
			content: message.content.clone(),
			display: message.display,
			details: message.details.clone(),
			timestamp: message.timestamp,
		})
		.unwrap_or(JsonValue::Null),
	)
	.unwrap_or(AgentMessage::Custom(JsonValue::Null))
}

fn crate_now_ms() -> i64 {
	std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map(|duration| duration.as_millis() as i64)
		.unwrap_or(0)
}

// ============================================================================
// Constructor options
// ============================================================================

/// Configuration for [`AgentSession::new`] (`AgentSessionConfig`).
pub struct AgentSessionConfig {
	pub agent: Arc<Agent>,
	pub session_manager: Arc<Mutex<SessionManager>>,
	pub settings_manager: Arc<Mutex<SettingsManager>>,
	pub cwd: String,
	/// Models to cycle through with Ctrl+P (from `--models` flag).
	pub scoped_models: Vec<ScopedModelEntry>,
	/// Resource loader for extensions, skills, prompts, themes, context
	/// files, and the system prompt.
	pub resource_loader: Arc<Mutex<DefaultResourceLoader>>,
	/// SDK custom tools registered outside extensions.
	pub custom_tools: Vec<ToolDefinition>,
	/// Canonical model/auth runtime used by coding-agent internals.
	pub model_runtime: Arc<ModelRuntime>,
	/// Keeps the prompt cache entry of the last session request warm.
	pub cache_warmer: Option<CacheWarmer>,
	/// Initial active built-in tool names. Default: [read, bash, edit, write].
	pub initial_active_tool_names: Option<Vec<String>>,
	/// Whether the initial tools come from the `defaultTools` setting. When
	/// true, reload activates tools newly added to the setting. Tools removed
	/// from it stay active.
	pub uses_default_tools: bool,
	/// Optional allowlist of tool names. When provided, only these tool names
	/// are exposed.
	pub allowed_tool_names: Option<Vec<String>>,
	/// Optional denylist of tool names. When provided, these tool names are
	/// not exposed.
	pub excluded_tool_names: Option<Vec<String>>,
	/// Override base tools (useful for custom runtimes). Synthesized into
	/// minimal tool definitions internally.
	pub base_tools_override: Option<Vec<AgentTool>>,
	/// Mutable ref used by the agent to access the current
	/// [`ExtensionRunner`].
	pub extension_runner_ref: Option<ExtensionRunnerRef>,
	/// Session start event metadata emitted when extensions bind to this
	/// runtime.
	pub session_start_event: Option<SessionStartEvent>,
}

impl Default for AgentSessionConfig {
	fn default() -> Self {
		unimplemented!("AgentSessionConfig has required members; construct it literally")
	}
}

// ============================================================================
// AgentSessionInner — private implementation
// ============================================================================

struct RequestAuth {
	model: Model,
	api_key: Option<String>,
	headers: Option<ProviderHeaders>,
	env: Option<pi_ai::types::ProviderEnv>,
	thinking_level: Option<AgentThinkingLevel>,
}

impl AgentSessionInner {
	// ==== infrastructure ====

	pub(crate) fn extension_runner(&self) -> Arc<ExtensionRunner> {
		self.extension_runner.read().clone().expect("extension runner is built during AgentSession::new")
	}

	fn update_idle_watch(&self) {
		let is_idle = !self.is_agent_run_active.load(std::sync::atomic::Ordering::SeqCst)
			&& self.compaction_abort_controller.lock().is_none()
			&& self.auto_compaction_abort_controller.lock().is_none()
			&& self.branch_summary_abort_controller.lock().is_none();
		let _ = self.idle_tx.send(is_idle);
	}

	pub(crate) fn emit(&self, event: AgentSessionEvent) {
		let listeners = self.event_listeners.lock().clone();
		for listener in listeners {
			listener(event.clone());
		}
	}

	fn emit_queue_update(&self) {
		self.emit(AgentSessionEvent::QueueUpdate {
			steering: self.steering_messages.lock().clone(),
			follow_up: self.follow_up_messages.lock().clone(),
		});
	}

	async fn emit_session_compact_failed(&self, event: crate::core::extensions::types::SessionCompactFailedEvent) {
		let runner = self.extension_runner();
		if runner.has_handlers("session_compact_failed") {
			let _ = runner
				.emit("session_compact_failed", serde_json::to_value(event).unwrap_or(JsonValue::Null))
				.await;
		}
	}

	/// Bridge the crate-local abort signal (extension events, D9) onto the
	/// pi-ai signal the compaction functions take.
	fn bridge_compaction_signal(&self, signal: &AbortSignal) -> AiAbortSignal {
		let bridged = AiAbortSignal::new();
		let forward_from = signal.clone();
		let target = bridged.clone();
		tokio::spawn(async move {
			forward_from.aborted_future().await;
			AiAbortSignal::abort(&target);
		});
		bridged
	}

	fn retry_policy(&self) -> RetryPolicy {
		let settings = self.settings_manager.lock();
		let (enabled, max_retries, base_delay_ms, max_agent_delay_ms) = settings.get_retry_settings();
		RetryPolicy { enabled, max_retries: max_retries as u32, base_delay_ms, max_agent_delay_ms: Some(max_agent_delay_ms) }
	}

	fn compaction_settings(&self, model: Option<&Model>) -> Result<CompactionSettings, String> {
		let model_ref = model.map(|model| (model.provider(), model.id()));
		let (enabled, reserve_tokens, keep_recent_tokens) = self.settings_manager.lock().get_compaction_settings(model_ref)?;
		Ok(CompactionSettings { enabled, reserve_tokens, keep_recent_tokens })
	}

	fn summarization_retry_callbacks(&self, source: SummarizationRetrySource) -> RetryCallbacks {
		let weak = self.self_weak.clone();
		RetryCallbacks {
			on_retry_scheduled: {
				let weak = weak.clone();
				Some(Arc::new(move |attempt: u32, max_attempts: u32, delay_ms: u64, error_message: &str| {
					let weak = weak.clone();
					let error_message = error_message.to_string();
					Box::pin(async move {
						if let Some(inner) = weak.upgrade() {
							inner.emit(AgentSessionEvent::SummarizationRetryScheduled {
								attempt,
								max_attempts,
								delay_ms,
								error_message,
							});
						}
					})
				}))
			},
			on_retry_attempt_start: {
				let weak = weak.clone();
				let source = source.clone();
				Some(Arc::new(move || {
					let weak = weak.clone();
					let source = source.clone();
					Box::pin(async move {
						if let Some(inner) = weak.upgrade() {
							inner.emit(AgentSessionEvent::SummarizationRetryAttemptStart { source });
						}
					})
				}))
			},
			on_retry_finished: {
				let weak = weak.clone();
				Some(Arc::new(move |_success: bool, _attempt: u32, _final_error: Option<&str>| {
					let weak = weak.clone();
					Box::pin(async move {
						if let Some(inner) = weak.upgrade() {
							inner.emit(AgentSessionEvent::SummarizationRetryFinished);
						}
					})
				}))
			},
		}
	}

	// ==== auth resolution ====

	async fn get_required_request_auth(&self, model: &Model, signal: Option<&AiAbortSignal>) -> Result<RequestAuth, String> {
		let result = match self.model_runtime.get_auth_for_model(&pi_ai::types::AnyModel::Chat(model.clone()), Some(
			crate::core::model_runtime::ModelRuntimeAuthOverrides { signal: signal.cloned(), ..Default::default() },
		))
		.await
		{
			Ok(result) => result,
			Err(error) => {
				if thrown_message(&error).contains("authHeader requires a resolved API key") {
					return Err(format_no_api_key_found_message(model.provider()));
				}
				return Err(thrown_message(&error));
			}
		};
		if let Some(result) = result {
			if result.auth.api_key.is_some() || result.auth.headers.is_some() {
				let request_model = match &result.auth.base_url {
					Some(base_url) => {
						let mut model = model.clone();
						model.base.base_url = base_url.clone();
						model
					}
					None => model.clone(),
				};
				return Ok(RequestAuth {
					model: request_model,
					api_key: result.auth.api_key,
					headers: without_deleted_headers(result.auth.headers),
					env: result.env,
					thinking_level: None,
				});
			}
		}

		if self.model_runtime.is_using_oauth(model.provider()) {
			return Err(format!(
				"Authentication failed for \"{}\". Credentials may have expired or network is unavailable. Run '/login {}' to re-authenticate.",
				model.provider(),
				model.provider()
			));
		}
		Err(format_no_api_key_found_message(model.provider()))
	}

	async fn get_summarization_request_auth(
		&self,
		selected_model: &Model,
		signal: Option<&AiAbortSignal>,
	) -> Result<RequestAuth, String> {
		// Route a virtual model first: summaries size their input and output
		// from the model they get.
		let (model, thinking_level) = if is_virtual_model(selected_model) {
			let route = self
				.model_runtime
				.resolve_model(
					selected_model,
					&convert_to_llm(&self.agent.messages()),
					crate::core::virtual_models::RouteRequestOptions {
						reason: crate::core::virtual_models::ModelRouteReason::Direct,
						thinking_level: agent_level_to_model_level(self.agent.thinking_level()),
						failed: None,
						state: None,
						signal: signal.cloned(),
					},
				)
				.await
				.map_err(|error| thrown_message(&error))?;
			(route.model, Some(self.agent.thinking_level()))
		} else {
			(selected_model.clone(), Some(self.agent.thinking_level()))
		};

		if Arc::ptr_eq(&self.agent.stream_function, &simple_stream_fn()) {
			let mut auth = self.get_required_request_auth(&model, signal).await?;
			auth.thinking_level = thinking_level;
			return Ok(auth);
		}

		match self
			.model_runtime
			.get_auth_for_model(&pi_ai::types::AnyModel::Chat(model.clone()), Some(
				crate::core::model_runtime::ModelRuntimeAuthOverrides { signal: signal.cloned(), ..Default::default() },
			))
			.await
		{
			Ok(Some(result)) => {
				let request_model = match &result.auth.base_url {
					Some(base_url) => {
						let mut model = model.clone();
						model.base.base_url = base_url.clone();
						model
					}
					None => model.clone(),
				};
				Ok(RequestAuth {
					model: request_model,
					api_key: result.auth.api_key,
					headers: without_deleted_headers(result.auth.headers),
					env: result.env,
					thinking_level,
				})
			}
			Ok(None) => Ok(RequestAuth { model, api_key: None, headers: None, env: None, thinking_level }),
			Err(error) => {
				if signal.map(|signal| signal.aborted()).unwrap_or(false) {
					return Err(thrown_message(&error));
				}
				Ok(RequestAuth { model, api_key: None, headers: None, env: None, thinking_level })
			}
		}
	}

	/// The model whose limits apply to `message`, or `None` when the message
	/// came from another model. Under a virtual selection, that is the
	/// physical model that produced it.
	fn model_for_message(&self, message: &AssistantMessage) -> Option<Model> {
		let model = self.agent.model();
		if is_virtual_model(&model) {
			return self.model_runtime.get_physical_model(&message.provider, &message.model);
		}
		(model.provider() == message.provider && model.id() == message.model).then_some(model)
	}

	/// Record the selection on the current branch when the branch implies
	/// another one, so a resume restores it (`_recordSelection`).
	fn record_selection(&self) {
		let model = self.agent.model();
		let runtime = Arc::clone(&self.model_runtime);
		let mut get_model = move |provider: &str, model_id: &str| pi_ai::Models::get_model(&*runtime, provider, model_id);
		let branch = self.session_manager.lock().get_branch();
		let recorded = get_branch_selection(&branch, &mut get_model);
		let Some(recorded) = recorded else { return };
		if recorded.provider == model.provider() && recorded.model_id == model.id() {
			return;
		}
		let recorded_model = get_model(&recorded.provider, &recorded.model_id);
		if !is_virtual_model(&model) && !recorded_model.map(|model| is_virtual_model(&model)).unwrap_or(false) {
			return;
		}
		let _ = self
			.session_manager
			.lock()
			.append_model_change(model.provider(), model.id());
	}

	/// The model whose limits apply to the conversation.
	fn limits_model(&self) -> Option<Model> {
			self.model_option()
	}

	fn model_option(&self) -> Option<Model> {
		let model = self.agent.model();
		(!model.base.id.is_empty()).then_some(model)
	}

	fn clamp_thinking_level(&self, level: AgentThinkingLevel, _available_levels: &[AgentThinkingLevel]) -> AgentThinkingLevel {
		if self.model_option().is_none() {
			return AgentThinkingLevel::Off;
		}
		model_level_to_agent_level(clamp_thinking_level(&self.agent.model(), agent_level_to_model_level(level)))
	}

	// ==== agent hook installation ====

	/// Install tool hooks once on the agent instance (`_installAgentToolHooks`).
	pub(crate) fn install_agent_tool_hooks(&self) {
		let weak = self.self_weak.clone();
		*self.agent.before_tool_call.write() = Some(Arc::new(move |context: BeforeToolCallContext, _signal| {
			let weak = weak.clone();
			Box::pin(async move {
				match weak.upgrade() {
					Some(inner) => inner.before_tool_call(&context).await,
					None => Ok(None),
				}
			})
		}));
		let weak = self.self_weak.clone();
		*self.agent.after_tool_call.write() = Some(Arc::new(move |context: AfterToolCallContext, _signal| {
			let weak = weak.clone();
			Box::pin(async move {
				match weak.upgrade() {
					Some(inner) => inner.after_tool_call(&context).await,
					None => Ok(None),
				}
			})
		}));
	}

	/// `tool_call` handlers (`_beforeToolCall`).
	async fn before_tool_call(&self, context: &BeforeToolCallContext) -> Result<Option<BeforeToolCallResult>, String> {
		let runner = self.extension_runner();
		if !runner.has_handlers("tool_call") {
			return Ok(None);
		}

		let mut event = crate::core::extensions::types::ToolCallEvent::Custom(crate::core::extensions::types::ToolCallEventBase {
			tool_call_id: context.tool_call.id.clone(),
			parent_tool_call_id: None,
			tool_name: context.tool_call.name.clone(),
			input: context.args.clone(),
		});
		// The runner routes handler failures to `emitError` itself; the
		// blocking decision travels through the result (upstream rethrows
		// thrown handlers — D74 runner seam).
		let result = runner.emit_tool_call(&mut event).await;
		// Handler argument patches travel through the mutated event input
		// (extensions/types D8 note).
		let patched_args = event.base().input.clone();
		Ok(result.map(|result| BeforeToolCallResult {
			block: result.block,
			reason: result.reason,
			terminate: result.terminate,
			args: (patched_args != context.args).then_some(patched_args),
		}))
	}

	/// `tool_result` handlers and image normalization (`_afterToolCall`).
	async fn after_tool_call(&self, context: &AfterToolCallContext) -> Result<Option<AfterToolCallResult>, String> {
		let runner = self.extension_runner();
		let hook_result = if runner.has_handlers("tool_result") {
			let mut event = crate::core::extensions::types::ToolResultEvent::Custom(
				crate::core::extensions::types::ToolResultEventBase {
					tool_call_id: context.tool_call.id.clone(),
					parent_tool_call_id: None,
					tool_name: context.tool_call.name.clone(),
					input: context.args.clone(),
					content: context.result.content.clone(),
					structured_content: context.result.structured_content.clone(),
					is_error: context.is_error,
					usage: context.result.usage.clone(),
					details: context.result.details.clone(),
				},
			);
			runner.emit_tool_result(&mut event).await
		} else {
			None
		};

		let content = hook_result
			.as_ref()
			.map(|result| result.content.clone().unwrap_or_else(|| context.result.content.clone()))
			.unwrap_or_else(|| context.result.content.clone());
		// Runs after the extension hook so images injected or replaced by
		// extensions are normalized too.
		let resize_options = self
			.limits_model()
			.and_then(|model| model.base.input_limits)
			.and_then(|limits| limits.images)
			.map(|images| images.resize);
		let as_tool_content: Vec<crate::utils::tool_result_images::ToolResultContent> = content
			.iter()
			.filter_map(|block| match block {
				UserContent::Text(text) => Some(crate::utils::tool_result_images::ToolResultContent::Text(text.clone())),
				UserContent::Image(image) => Some(crate::utils::tool_result_images::ToolResultContent::Image(image.clone())),
				_ => None,
			})
			.collect();
		let auto_resize_images = self.settings_manager.lock().get_image_auto_resize();
		let normalized_content = crate::utils::tool_result_images::normalize_tool_result_images(
			as_tool_content,
			Some(&crate::utils::tool_result_images::NormalizeToolResultImagesOptions {
				auto_resize_images: Some(auto_resize_images),
				resize_options: resize_options.flatten(),
			}),
		)
		.await
		.into_iter()
		.map(|block| match block {
			crate::utils::tool_result_images::ToolResultContent::Text(text) => UserContent::Text(text),
			crate::utils::tool_result_images::ToolResultContent::Image(image) => UserContent::Image(image),
		})
		.collect::<Vec<UserContent>>();

		let hook = match &hook_result {
			Some(hook) => hook,
			None => {
				if normalized_content == content {
					return Ok(None);
				}
				return Ok(Some(AfterToolCallResult {
					content: Some(normalized_content),
					details: None,
					structured_content: None,
					is_error: None,
					usage: None,
					terminate: None,
				}));
			}
		};

		Ok(Some(AfterToolCallResult {
			content: Some(normalized_content),
			details: hook.details.clone(),
			structured_content: hook.structured_content.clone(),
			is_error: hook.is_error,
			usage: hook.usage.clone(),
			terminate: None,
		}))
	}

	/// Run a call that the tool call `parent_tool_call_id` made through
	/// `ctx.executeTool()` (`_executeNestedToolCall`).
	async fn execute_nested_tool_call(
		&self,
		parent_tool_call_id: &str,
		name: &str,
		args: Option<serde_json::Map<String, JsonValue>>,
		options: ExecuteToolOptions,
	) -> AgentToolCallOutcome {
		{
			let mut nested = self.nested_tool_calls.lock();
			if nested.is_none() {
				let weak = self.self_weak.clone();
				*nested = Some(Arc::new(NestedToolCallRunner::new(Arc::new(NestedHost { inner: weak }))));
			}
		}
		let nested = Arc::clone(self.nested_tool_calls.lock().as_ref().expect("just set"));
		nested.execute(parent_tool_call_id, name, args, NestedToolCallOptions { signal: options.signal, on_update: options.on_update }).await
	}

	/// Whether `projection` exceeds the compaction threshold of `model`.
	fn exceeds_compaction_threshold(&self, model: &Model, projection: &SessionProjection) -> bool {
		if model.context_window == 0 {
			return false;
		}
		let settings = match self.compaction_settings(Some(model)) {
			Ok(settings) => settings,
			Err(_) => return false,
		};
		let branch = self.session_manager.lock().get_branch();
		should_compact(estimate_projected_context_tokens(projection, &branch).tokens, model.context_window, &settings)
	}

	async fn compact_before_next_assistant_response(&self, _context: &pi_agent::AgentContext) -> Vec<AgentMessage> {
		let projection = self.session_manager.lock().build_session_projection();
		// A virtual selection is checked in prepareRequest, against the model
		// the request is routed to.
		let model = self.model_option();
		let should_run = match &model {
			Some(model) => !is_virtual_model(model) && self.exceeds_compaction_threshold(model, &projection),
			None => false,
		};
		if !should_run {
			return projection.messages;
		}
		self.run_auto_compaction(CompactReason::Threshold, false).await;
		self.session_manager.lock().build_session_projection().messages
	}

	/// `prepareRequest` projection + virtual routing (`_installAgentRequestProjection`).
	pub(crate) fn install_agent_request_projection(&self) {
		let previous = self.agent.prepare_request.read().clone();
		let weak = self.self_weak.clone();
		*self.agent.prepare_request.write() = Some(Arc::new(move |request: PrepareRequestContext, signal| {
			let weak = weak.clone();
			let previous = previous.clone();
			Box::pin(async move {
				let Some(inner) = weak.upgrade() else { return None };
				inner.prepare_request_with_projection(request, signal, previous).await
			})
		}));
	}

	async fn prepare_request_with_projection(
		&self,
		request: PrepareRequestContext,
		signal: Option<AiAbortSignal>,
		previous: Option<pi_agent::types::PrepareRequestHook>,
	) -> Option<AgentRequestUpdate> {
		let failed = self.failed_response.lock().take();
		let mut prepare = || async {
			let projection = self.session_manager.lock().build_session_projection();
			let canonical_context = pi_agent::types::PrepareRequestContext {
				context: pi_agent::AgentContext {
					messages: projection.messages.clone(),
					tools: Some(self.agent.tools()),
				},
				model: self.agent.model(),
				thinking_level: self.agent.thinking_level(),
			};
			let previous = match &previous {
				Some(hook) => hook(canonical_context.clone(), signal.clone()).await,
				None => None,
			};
			(previous, canonical_context, projection)
		};
		let (mut previous_result, mut canonical, projection) = prepare().await;
		let model = previous_result
			.as_ref()
			.and_then(|update| update.model.clone())
			.unwrap_or_else(|| self.agent.model());
		let thinking_level = previous_result
			.as_ref()
			.and_then(|update| update.thinking_level)
			.unwrap_or_else(|| self.agent.thinking_level());
		if !is_virtual_model(&model) {
			return Some(previous_result.unwrap_or(AgentRequestUpdate {
				context: Some(pi_agent::AgentContext {
					messages: canonical.context.messages,
					tools: canonical.context.tools,
				}),
				model: None,
				thinking_level: None,
			}));
		}

		// The selection stays in agent state; only this request uses the
		// routed model. A routing failure rejects, which ends the run with an
		// error response. Only messages the user wrote start a turn.
		let messages = &canonical.context.messages;
		let last_response = messages.iter().rposition(|message| message.role() == "assistant").map(|index| index as i64).unwrap_or(-1);
		let user_turn = messages
			.iter()
			.skip((last_response + 1) as usize)
			.any(|message| message.role() == "user");
		let branch_for_state = self.session_manager.lock().get_branch();
		let state = get_virtual_model_state(&branch_for_state, model.provider(), model.id());
		let state_for_request = state.clone();
		let route = self
			.model_runtime
			.resolve_model(
				&model,
				&convert_to_llm(messages),
				crate::core::virtual_models::RouteRequestOptions {
					reason: if failed.is_some() {
						crate::core::virtual_models::ModelRouteReason::Retry
					} else if user_turn {
						crate::core::virtual_models::ModelRouteReason::User
					} else {
						crate::core::virtual_models::ModelRouteReason::Continuation
					},
					thinking_level: agent_level_to_model_level(thinking_level),
					failed,
					state: state_for_request,
					signal: signal.clone(),
				},
			)
			.await
			.ok()?;
		if route.state.is_some() && route.state != state.clone() {
			let data = VirtualModelStateData {
				provider: model.provider().to_string(),
				model_id: model.id().to_string(),
				state: route.state.clone().unwrap_or(JsonValue::Null),
			};
			let entry_id = self
				.session_manager
				.lock()
				.append_custom_entry(VIRTUAL_MODEL_STATE_ENTRY, Some(serde_json::to_value(data).unwrap_or(JsonValue::Null)))
				.ok()?;
			if let Some(entry) = self.session_manager.lock().get_entry(&entry_id).cloned() {
				self.emit(AgentSessionEvent::EntryAppended { entry });
			}
		}
		// The route stands: the router already decided this request. The
		// state entry does not change the projection.
		if self.exceeds_compaction_threshold(&route.model, &projection) {
			self.run_auto_compaction(CompactReason::Threshold, false).await;
			let (previous_next, canonical_next, _) = prepare().await;
			previous_result = previous_next;
			canonical = canonical_next;
		}
		Some(AgentRequestUpdate {
			context: Some(pi_agent::AgentContext {
				messages: canonical.context.messages,
				tools: canonical.context.tools,
			}),
			model: previous_result.as_ref().and_then(|update| update.model.clone()).or(Some(route.model)),
			thinking_level: previous_result
				.as_ref()
				.and_then(|update| update.thinking_level)
				.or(Some(model_level_to_agent_level(route.thinking_level))),
		})
	}

	/// Turn-end boundary dispatch via the agent's `finishTurn` hook
	/// (`_installAgentBoundaryHooks`).
	pub(crate) fn install_agent_boundary_hooks(&self) {
		let previous = self.agent.finish_turn.read().clone();
		let weak = self.self_weak.clone();
		*self.agent.finish_turn.write() = Some(Arc::new(move |turn: pi_agent::types::AgentTurnContext, signal| {
			let weak = weak.clone();
			let previous = previous.clone();
			Box::pin(async move {
				let Some(inner) = weak.upgrade() else {
					return match previous {
						Some(hook) => hook(turn, signal).await,
						None => None,
					};
				};
				inner.finish_turn_with_boundary(turn, signal, previous).await
			})
		}));
	}

	async fn finish_turn_with_boundary(
		&self,
		turn: pi_agent::types::AgentTurnContext,
		signal: Option<AiAbortSignal>,
		previous: Option<pi_agent::types::FinishTurn>,
	) -> Option<pi_agent::types::AgentTurnDecision> {
		let message = turn.message.clone();
		self.boundary_dispatched_messages
			.lock()
			.insert(message_fingerprint(&AgentMessage::Llm(Message::Assistant(message.clone()))));
		let extension_continue = self.dispatch_turn_end_boundary(&message, &turn.tool_results).await;
		let previous_decision = match &previous {
			Some(hook) => hook(turn, signal).await,
			None => None,
		};
		if previous_decision == Some(pi_agent::types::AgentTurnDecision::End) {
			return previous_decision;
		}
		if extension_continue || previous_decision == Some(pi_agent::types::AgentTurnDecision::Continue) {
			return Some(pi_agent::types::AgentTurnDecision::Continue);
		}
		None
	}

	/// Compaction before the next assistant response + prompt/loadout refresh
	/// (`_installAgentNextTurnRefresh`).
	pub(crate) fn install_agent_next_turn_refresh(&self) {
		let previous = self.agent.prepare_next_turn_with_context.read().clone();
		let weak = self.self_weak.clone();
		*self.agent.prepare_next_turn_with_context.write() = Some(Arc::new(move |turn, signal| {
			let weak = weak.clone();
			let previous = previous.clone();
			Box::pin(async move {
				let Some(inner) = weak.upgrade() else {
					return match previous {
						Some(hook) => hook(turn, signal).await,
						None => None,
					};
				};
				inner.prepare_next_turn_with_refresh(turn, signal, previous).await
			})
		}));
	}

	async fn prepare_next_turn_with_refresh(
		&self,
		turn: pi_agent::types::PrepareNextTurnContext,
		signal: Option<AiAbortSignal>,
		previous: Option<pi_agent::types::PrepareNextTurnWithContextHook>,
	) -> Option<AgentLoopTurnUpdate> {
		let messages = self.compact_before_next_assistant_response(&turn.context).await;
		let turn = pi_agent::types::PrepareNextTurnContext {
			context: pi_agent::AgentContext { messages, tools: turn.context.tools },
			message: turn.message,
			tool_results: turn.tool_results,
			new_messages: turn.new_messages,
		};
		let previous_snapshot = match &previous {
			Some(hook) => hook(turn.clone(), signal).await,
			None => None,
		};
		let next_context = previous_snapshot
			.as_ref()
			.and_then(|update| update.context.clone())
			.unwrap_or(turn.context.clone());
		let run_options = self
			.run_system_prompt_options
			.lock()
			.clone()
			.or_else(|| self.base_system_prompt_options.lock().clone())
			.unwrap_or_else(empty_normalized_prompt_options);
		let base = self
			.base_system_prompt_options
			.lock()
			.clone()
			.unwrap_or_else(empty_normalized_prompt_options);
		let mut options = run_options.clone();
		options.selected_tools = self.agent.tools().iter().map(|tool| tool.name().to_string()).collect();
		let mut tool_snippets = base.tool_snippets.clone();
		tool_snippets.extend(run_options.tool_snippets.clone());
		options.tool_snippets = tool_snippets;
		let mut tool_guidelines = base.tool_guidelines.clone();
		for (name, guidelines) in run_options.tool_guidelines.clone() {
			tool_guidelines.insert(name, guidelines);
		}
		options.tool_guidelines = tool_guidelines;

		let update_message =
			self.prepare_prompt_and_tool_loadout(&mut options, Some(&next_context.messages));
		if std::env::var("PI_DBG").is_ok() {
			let llm: Vec<pi_ai::types::Message> = next_context.messages.iter().filter_map(|m| m.as_llm().cloned()).collect();
			eprintln!("DBG next_turn: roles={:?} current_sys={:?} update_message={:?}", next_context.messages.iter().map(|m| m.role().to_string()).collect::<Vec<_>>(), pi_ai::utils::transcript::get_current_system_message(&llm).map(|s| (s.sections.clone(), s.tools_added.clone())), update_message.is_some());
		}
		// Keep session.systemPrompt and ctx.getSystemPrompt() in step with
		// what the provider sees.
		*self.run_system_prompt_options.lock() = Some(options);

		Some(AgentLoopTurnUpdate {
			context: Some(pi_agent::AgentContext {
				messages: next_context.messages,
				tools: Some(self.agent.tools()),
			}),
			messages: {
				let mut messages = previous_snapshot.as_ref().map(|update| update.messages.clone()).unwrap_or_default();
				if let Some(message) = update_message {
					messages.push(AgentMessage::Llm(Message::System(message)));
				}
				messages
			},
			model: Some(self.agent.model()),
			thinking_level: Some(self.agent.thinking_level()),
		})
	}

	/// Remove declarations that `prepareLoadout` hooks hide
	/// (`_installHiddenDeclarationsProjection`).
	pub(crate) fn install_hidden_declarations_projection(&self) {
		let previous = self.agent.transform_context.read().clone();
		let weak = self.self_weak.clone();
		*self.agent.transform_context.write() = Some(Arc::new(move |messages: Vec<AgentMessage>, signal| {
			let weak = weak.clone();
			let previous = previous.clone();
			Box::pin(async move {
				let transformed = match &previous {
					Some(hook) => hook(messages, signal).await,
					None => messages,
				};
				let Some(inner) = weak.upgrade() else { return transformed };
				let hidden = inner.hidden_declarations.lock().clone();
				if hidden.is_empty() {
					return transformed;
				}
				transformed
					.into_iter()
					.map(|message| match message {
						AgentMessage::Llm(Message::System(system)) => {
							if system.tools_added.is_none() && system.tools_removed.is_none() {
								return AgentMessage::Llm(Message::System(system));
							}
							let mut system = system;
							let added: Vec<_> = system
								.tools_added
								.take()
								.unwrap_or_default()
								.into_iter()
								.filter(|tool| !hidden.contains(&tool.name))
								.collect();
							let removed: Vec<_> = system
								.tools_removed
								.take()
								.unwrap_or_default()
								.into_iter()
								.filter(|tool| !hidden.contains(&tool.name))
								.collect();
							if !added.is_empty() {
								system.tools_added = Some(added);
							}
							if !removed.is_empty() {
								system.tools_removed = Some(removed);
							}
							AgentMessage::Llm(Message::System(system))
						}
						other => other,
					})
					.collect()
			})
		}));
	}

	/// Send a forced prompt as the provider's leading system prompt without
	/// recording it (`_installAgentForcedPromptProjection`).
	pub(crate) fn install_agent_forced_prompt_projection(&self) {
		let previous = self.agent.transform_context.read().clone();
		let weak = self.self_weak.clone();
		*self.agent.transform_context.write() = Some(Arc::new(move |messages: Vec<AgentMessage>, signal| {
			let weak = weak.clone();
			let previous = previous.clone();
			Box::pin(async move {
				let transformed = match &previous {
					Some(hook) => hook(messages, signal).await,
					None => messages,
				};
				let Some(inner) = weak.upgrade() else { return transformed };
				let forced = inner
					.run_system_prompt_options
					.lock()
					.clone()
					.and_then(|options| options.force_system_prompt);
				let Some(forced) = forced else { return transformed };
				let current = pi_ai::utils::transcript::get_current_system_message(
					&transformed.iter().filter_map(|message| message.as_llm().cloned()).collect::<Vec<_>>(),
				);
				let mut head = SystemMessage {
					content: pi_ai::types::SystemContent::Text(forced),
					sections: None,
					tools_added: current.as_ref().and_then(|system| system.tools_added.clone()),
					tools_removed: None,
					timestamp: current.as_ref().map(|system| system.timestamp).unwrap_or_else(crate_now_ms),
				};
				head.sections = None;
				let mut next = vec![AgentMessage::Llm(Message::System(head))];
				next.extend(transformed.into_iter().filter(|message| message.role() != "system"));
				next
			})
		}));
	}

	// ==== boundary plumbing ====

	pub(crate) fn refresh_finalized_context(&self) {
		let projection = self.session_manager.lock().build_session_projection();
		self.agent.set_messages(projection.messages);
	}

	fn apply_boundary_drafts(
		&self,
		manager: &Mutex<SessionManager>,
		drafts: &[SessionBoundaryDraft],
	) -> Vec<SessionEntry> {
		let mut appended = Vec::new();
		let mut manager = manager.lock();
		for draft in drafts {
			let entry_id = match draft {
				SessionBoundaryDraft::Custom(custom) => manager
					.append_custom_entry(&custom.custom_type, custom.data.clone())
					.expect("boundary draft append"),
				SessionBoundaryDraft::CustomMessage(custom_message) => manager
					.append_custom_message_entry(
						&custom_message.custom_type,
						custom_message.content.clone(),
						custom_message.display,
						custom_message.details.clone(),
					)
					.expect("boundary draft append"),
				SessionBoundaryDraft::ContextEdit(context_edit) => manager
					.append_context_edit(&context_edit.target_id, Some(context_edit_replacement_to_editable(&context_edit.replacement)))
					.expect("boundary draft append"),
				SessionBoundaryDraft::Compaction(compaction) => {
					let tokens_before = estimate_projected_context_tokens(
						&manager.build_session_projection(),
						&manager.get_branch(),
					)
					.tokens;
					manager
						.append_compaction(
							&compaction.summary,
							compaction.first_kept_entry_id.as_deref(),
							tokens_before,
							compaction.details.clone(),
							None,
							compaction.usage.clone(),
						)
						.expect("boundary draft append")
				}
			};
			if let Some(entry) = manager.get_entry(&entry_id) {
				appended.push(entry.clone());
			}
		}
		appended
	}

	fn get_pending_boundary_messages(&self) -> Vec<AgentMessage> {
		self.agent
			.peek_queued_messages()
			.into_iter()
			.chain(self.pending_custom_messages.lock().iter().map(custom_message_to_agent_message))
			.collect()
	}

	fn build_boundary_context(
		&self,
		drafts: &[SessionBoundaryDraft],
		boundary: &str,
	) -> Result<BoundaryContextPreview, String> {
		let preview_manager = self.create_boundary_preview_manager(drafts)?;
		let projection = preview_manager.build_session_projection();
		let pending_messages = self.get_pending_boundary_messages();
		let llm_messages = convert_to_llm(&projection.messages);
		let final_role = llm_messages.last().map(|message| message.role());
		let has_non_system_context = llm_messages.iter().any(|message| message.role() != "system");
		let context_can_continue = has_non_system_context && final_role != Some("assistant");
		let pending_custom_context = !self.pending_custom_messages.lock().is_empty();
		Ok(BoundaryContextPreview {
			context_entries: projection.entries,
			context_messages: projection.messages,
			llm_messages,
			pending_messages,
			can_continue: context_can_continue
				|| pending_custom_context
				|| (boundary == "turn_end"
					&& self.agent.has_queued_messages())
				|| (boundary == "agent_before_settle"
					&& final_role == Some("assistant")
					&& self.agent.has_queued_messages()),
		})
	}

	fn create_boundary_preview_manager(&self, drafts: &[SessionBoundaryDraft]) -> Result<SessionManager, String> {
		let manager = self.session_manager.lock();
		let header = manager.get_header().ok_or_else(|| "Session header is missing".to_string())?;
		let branch = manager.get_branch();
		drop(manager);
		let mut file_entries = Vec::with_capacity(branch.len() + 1);
		file_entries.push(crate::core::session_manager::FileEntry::Header(header));
		for entry in branch {
			file_entries.push(crate::core::session_manager::FileEntry::Entry(entry));
		}
		let preview = SessionManager::in_memory(Some(&self.cwd), None, Some(file_entries))
			.map_err(|error| error.to_string())?;
		let mut preview = preview;
		let _ = self.apply_boundary_drafts_manager(&mut preview, drafts);
		Ok(preview)
	}

	fn apply_boundary_drafts_manager(&self, manager: &mut SessionManager, drafts: &[SessionBoundaryDraft]) -> Vec<SessionEntry> {
		let mut appended = Vec::new();
		for draft in drafts {
			let entry_id = match draft {
				SessionBoundaryDraft::Custom(custom) => manager
					.append_custom_entry(&custom.custom_type, custom.data.clone())
					.expect("boundary draft append"),
				SessionBoundaryDraft::CustomMessage(custom_message) => manager
					.append_custom_message_entry(
						&custom_message.custom_type,
						custom_message.content.clone(),
						custom_message.display,
						custom_message.details.clone(),
					)
					.expect("boundary draft append"),
				SessionBoundaryDraft::ContextEdit(context_edit) => manager
					.append_context_edit(&context_edit.target_id, Some(context_edit_replacement_to_editable(&context_edit.replacement)))
					.expect("boundary draft append"),
				SessionBoundaryDraft::Compaction(compaction) => {
					let tokens_before = estimate_projected_context_tokens(
						&manager.build_session_projection(),
						&manager.get_branch(),
					)
					.tokens;
					manager
						.append_compaction(
							&compaction.summary,
							compaction.first_kept_entry_id.as_deref(),
							tokens_before,
							compaction.details.clone(),
							None,
							compaction.usage.clone(),
						)
						.expect("boundary draft append")
				}
			};
			if let Some(entry) = manager.get_entry(&entry_id) {
				appended.push(entry.clone());
			}
		}
		appended
	}

	fn commit_boundary_drafts(&self, drafts: &[SessionBoundaryDraft]) {
		let appended = self.apply_boundary_drafts(&self.session_manager, drafts);
		self.refresh_finalized_context();
		for entry in appended {
			self.emit(AgentSessionEvent::EntryAppended { entry });
		}
	}

	fn report_invalid_boundary_continuation(&self, event: &str) {
		self.extension_runner().emit_error(&ExtensionError {
			extension_path: "<boundary>".to_string(),
			event: event.to_string(),
			error: format!("{event} requested continuation without runnable model context"),
			stack: None,
		});
	}

	async fn dispatch_turn_end_boundary(&self, message: &AssistantMessage, tool_results: &[ToolResultMessage]) -> bool {
		let outcome = match message.stop_reason {
			pi_ai::types::StopReason::Aborted => AgentActivityOutcome::Aborted,
			pi_ai::types::StopReason::Error => AgentActivityOutcome::Error,
			_ => AgentActivityOutcome::Completed,
		};
		*self.last_activity_outcome.lock() = outcome;
		let message_entry_id = self.find_persisted_message_entry_id(&AgentMessage::Llm(Message::Assistant(message.clone())));
		let runner = self.extension_runner();
		if !runner.has_handlers("turn_end") {
			return false;
		}
		let Some(message_entry_id) = message_entry_id else {
			runner.emit_error(&ExtensionError {
				extension_path: "<boundary>".to_string(),
				event: "turn_end".to_string(),
				error: "turn_end could not resolve the persisted assistant entry ID".to_string(),
				stack: None,
			});
			return false;
		};
		let tool_result_entry_ids: Vec<String> = tool_results
			.iter()
			.filter_map(|result| self.find_persisted_message_entry_id(&AgentMessage::Llm(Message::ToolResult(result.clone()))))
			.collect();
		let turn_index = self.turn_index.load(std::sync::atomic::Ordering::SeqCst);
		let weak = self.self_weak.clone();
		let build_context: BoundaryContextBuilder = Arc::new(move |drafts: &[SessionBoundaryDraft]| {
			match weak.upgrade() {
				Some(inner) => inner.build_boundary_context(drafts, "turn_end"),
				None => Err("session dropped".to_string()),
			}
		});
		let boundary = runner
			.emit_boundary(
				"turn_end",
				json!({
					"type": "turn_end",
					"turnIndex": turn_index,
					"message": AgentMessage::Llm(Message::Assistant(message.clone())),
					"toolResults": tool_results,
					"messageEntryId": message_entry_id,
					"toolResultEntryIds": tool_result_entry_ids,
					"outcome": outcome,
				}),
				build_context,
			)
			.await;
		self.commit_boundary_drafts(&boundary.entries);
		if boundary.continue_ && !self.build_boundary_context(&[], "turn_end").map(|context| context.can_continue).unwrap_or(false) {
			self.report_invalid_boundary_continuation("turn_end");
			return false;
		}
		boundary.continue_
	}

	// ==== agent event handling ====

	pub(crate) async fn handle_agent_event(&self, event: AgentEvent) {
		// Record the calls a tool made through ctx.executeTool() and their
		// usage on its result message.
		if let Some(nested) = self.nested_tool_calls.lock().clone() {
			match &event {
				AgentEvent::MessageStart { message } => {
					if let AgentMessage::Llm(Message::ToolResult(tool_result)) = message {
						if let Some(summary) = nested.take_record(&tool_result.tool_call_id) {
							let mut tool_result = tool_result.clone();
							tool_result.nested_calls = summary.calls;
							tool_result.usage = match (tool_result.usage.clone(), summary.usage) {
								(Some(existing), Some(nested_usage)) => Some(combine_usage(&existing, &nested_usage)),
								(None, Some(nested_usage)) => Some(nested_usage),
								(existing, None) => existing,
							};
							self.replace_agent_state_message(
								message,
								&AgentMessage::Llm(Message::ToolResult(tool_result)),
							);
						}
					}
				}
				AgentEvent::AgentEnd { .. } => nested.clear(),
				_ => {}
			}
		}

		// When a user message starts, check if it's from either queue and
		// remove it BEFORE emitting. This ensures the UI sees the updated
		// queue state.
		if let AgentEvent::MessageStart { message } = &event {
			if let AgentMessage::Llm(Message::User(user)) = message {
				self.overflow_recovery_attempted.store(false, std::sync::atomic::Ordering::SeqCst);
				let message_text = user_content_field_text_for_editor(&user.content);
				if !message_text.is_empty() {
					// Check steering queue first, then follow-up queue.
					let mut steering = self.steering_messages.lock();
					if let Some(index) = steering.iter().position(|text| *text == message_text) {
						steering.remove(index);
						drop(steering);
						self.emit_queue_update();
					} else {
						drop(steering);
						let mut follow_up = self.follow_up_messages.lock();
						if let Some(index) = follow_up.iter().position(|text| *text == message_text) {
							follow_up.remove(index);
							drop(follow_up);
							self.emit_queue_update();
						}
					}
				}
			}
		}

		// Emit to extensions first, then notify public listeners.
		self.emit_extension_event(&event).await;
		match &event {
			AgentEvent::AgentEnd { messages } => self.emit(AgentSessionEvent::AgentEnd {
				messages: messages.clone(),
				will_retry: self.will_retry_after_agent_end(messages),
			}),
			AgentEvent::AgentStart => self.emit(AgentSessionEvent::AgentStart),
			AgentEvent::TurnStart => self.emit(AgentSessionEvent::TurnStart),
			AgentEvent::TurnEnd { message, tool_results } => {
				self.emit(AgentSessionEvent::TurnEnd { message: message.clone(), tool_results: tool_results.clone() })
			}
			AgentEvent::MessageStart { message } => self.emit(AgentSessionEvent::MessageStart { message: message.clone() }),
			AgentEvent::MessageUpdate { message, assistant_message_event } => self.emit(AgentSessionEvent::MessageUpdate {
				message: message.clone(),
				assistant_message_event: assistant_message_event.clone(),
			}),
			AgentEvent::MessageEnd { message } => self.emit(AgentSessionEvent::MessageEnd { message: message.clone() }),
			AgentEvent::ToolExecutionStart { tool_call_id, tool_name, args } => self.emit(AgentSessionEvent::ToolExecutionStart {
				tool_call_id: tool_call_id.clone(),
				tool_name: tool_name.clone(),
				args: args.clone(),
				parent_tool_call_id: None,
			}),
			AgentEvent::ToolExecutionUpdate { tool_call_id, tool_name, args, partial_result } => {
				self.emit(AgentSessionEvent::ToolExecutionUpdate {
					tool_call_id: tool_call_id.clone(),
					tool_name: tool_name.clone(),
					args: args.clone(),
					partial_result: serde_json::to_value(partial_result).unwrap_or(JsonValue::Null),
					parent_tool_call_id: None,
				})
			}
			AgentEvent::ToolExecutionEnd { tool_call_id, tool_name, result, is_error } => {
				self.emit(AgentSessionEvent::ToolExecutionEnd {
					tool_call_id: tool_call_id.clone(),
					tool_name: tool_name.clone(),
					result: serde_json::to_value(result).unwrap_or(JsonValue::Null),
					is_error: *is_error,
					parent_tool_call_id: None,
				})
			}
		}

		// Handle session persistence.
		if let AgentEvent::MessageEnd { message } = &event {
			let mut entry_id: Option<String> = None;
			match message {
				// Custom messages from extensions persist as
				// CustomMessageEntry.
				AgentMessage::Custom(_) => {
					if let Some(custom) = crate::core::messages::custom_value_from_agent(message).cloned() {
						if let Ok(draft) = serde_json::from_value::<CustomMessage>(custom.clone()) {
							entry_id = self
								.session_manager
								.lock()
								.append_custom_message_entry(&draft.custom_type, draft.content, draft.display, draft.details)
								.ok();
						}
					}
				}
				AgentMessage::Llm(_) => {
					entry_id = self.session_manager.lock().append_message(message.clone()).ok();
				}
			}
			let _ = entry_id;

			if let AgentMessage::Llm(Message::Assistant(assistant)) = message {
				*self.last_assistant_message.lock() = Some(assistant.clone());
				if assistant.stop_reason != pi_ai::types::StopReason::Error
					&& assistant.stop_reason != pi_ai::types::StopReason::Length
				{
					self.overflow_recovery_attempted.store(false, std::sync::atomic::Ordering::SeqCst);
				}

				// Reset retry counter immediately on successful assistant
				// response. This prevents accumulation across multiple LLM
				// calls within a turn.
				let retry_attempt = self.retry_attempt.load(std::sync::atomic::Ordering::SeqCst);
				if assistant.stop_reason != pi_ai::types::StopReason::Error && retry_attempt > 0 {
					self.emit(AgentSessionEvent::AutoRetryEnd { success: true, attempt: retry_attempt, final_error: None });
					self.retry_attempt.store(0, std::sync::atomic::Ordering::SeqCst);
				}
			}
		}

		// A turn ends after its assistant message and every tool result has
		// been appended, so this is the first point in the run where a
		// context-only custom message can be inserted without landing between
		// a tool call and its result.
		if let AgentEvent::TurnEnd { tool_results, .. } = &event {
			*self.last_assistant_tool_results.lock() = tool_results.clone();
			self.flush_pending_custom_messages();
		}
	}

	fn will_retry_after_agent_end(&self, messages: &[AgentMessage]) -> bool {
		if self.agent_run_abort_requested.load(std::sync::atomic::Ordering::SeqCst) {
			return false;
		}
		let settings = self.settings_manager.lock().get_retry_settings();
		let (enabled, max_retries, _, _) = settings;
		let retry_attempt = self.retry_attempt.load(std::sync::atomic::Ordering::SeqCst);
		if !enabled || retry_attempt >= max_retries as u32 {
			return false;
		}

		for message in messages.iter().rev() {
			if let AgentMessage::Llm(Message::Assistant(assistant)) = message {
				return self.is_retryable_error(assistant);
			}
		}
		false
	}

	fn find_persisted_message_entry_id(&self, message: &AgentMessage) -> Option<String> {
		let branch = self.session_manager.lock().get_branch();
		for entry in branch.iter().rev() {
			if let SessionEntry::Message(message_entry) = entry {
				if &message_entry.message == message {
					return Some(entry.id().to_string());
				}
			}
		}

		let messages = self.agent.messages();
		let message_index = messages.iter().position(|existing| existing == message)?;
		let projection = self.session_manager.lock().build_session_projection();
		let mut projected_index = 0usize;
		for entry in &projection.entries {
			for _ in &entry.messages {
				if projected_index == message_index {
					return Some(entry.source_entry.id().to_string());
				}
				projected_index += 1;
			}
		}
		None
	}

	fn omit_recovery_attempt(&self, message: &AssistantMessage, tool_results: &[AgentMessage]) {
		let mut targets = vec![AgentMessage::Llm(Message::Assistant(message.clone()))];
		targets.extend(tool_results.iter().cloned());
		let target_ids: Vec<Option<String>> = targets.iter().map(|target| self.find_persisted_message_entry_id(target)).collect();
		let agent_messages = self.agent.messages();
		let unresolved_projected_target = targets.iter().zip(target_ids.iter()).any(|(target, target_id)| {
			target_id.is_none() && agent_messages.iter().any(|existing| existing == target)
		});
		if unresolved_projected_target {
			panic!("Cannot persist recovery omission because a projected message has no source entry");
		}
		for target_id in target_ids.into_iter().flatten() {
			// Drop the append guard before locking again: in edition 2021 the
			// scrutinee temporary of `if let` lives to the end of the whole
			// statement, so nesting the second `lock()` in the body would
			// re-acquire the same non-reentrant mutex and deadlock.
			let appended = self.session_manager.lock().append_context_edit(&target_id, None);
			if let Ok(edit_id) = appended {
				let entry = self.session_manager.lock().get_entry(&edit_id).cloned();
				if let Some(entry) = entry {
					self.emit(AgentSessionEvent::EntryAppended { entry });
				}
			}
		}
		self.refresh_finalized_context();
	}

	fn find_last_assistant_message(&self) -> Option<AssistantMessage> {
		self.agent.messages().into_iter().rev().find_map(|message| match message {
			AgentMessage::Llm(Message::Assistant(assistant)) => Some(assistant),
			_ => None,
		})
	}

	/// Replace the structurally-equal message in agent state (D77: upstream
	/// mutates the object in place).
	fn replace_agent_state_message(&self, target: &AgentMessage, replacement: &AgentMessage) {
		if target == replacement {
			return;
		}
		let mut state = self.agent.lock_state();
		if let Some(index) = state.messages.iter().position(|message| message == target) {
			state.messages[index] = replacement.clone();
		}
	}

	/// Emit extension events based on agent events (`_emitExtensionEvent`).
	async fn emit_extension_event(&self, event: &AgentEvent) {
		let runner = self.extension_runner();
		match event {
			AgentEvent::AgentStart => {
				self.turn_index.store(0, std::sync::atomic::Ordering::SeqCst);
				let _ = runner.emit("agent_start", json!({ "type": "agent_start" })).await;
			}
			AgentEvent::AgentEnd { messages } => {
				let _ = runner
					.emit("agent_end", json!({ "type": "agent_end", "messages": messages }))
					.await;
			}
			AgentEvent::TurnStart => {
				let extension_event = json!({
					"type": "turn_start",
					"turnIndex": self.turn_index.load(std::sync::atomic::Ordering::SeqCst),
					"timestamp": crate_now_ms(),
				});
				let _ = runner.emit("turn_start", extension_event).await;
			}
			AgentEvent::TurnEnd { message, tool_results } => {
				let dispatched = match message {
					AgentMessage::Llm(Message::Assistant(assistant)) => {
						self.boundary_dispatched_messages
							.lock()
							.remove(&message_fingerprint(&AgentMessage::Llm(Message::Assistant(assistant.clone()))))
					}
					_ => true,
				};
				if let AgentMessage::Llm(Message::Assistant(assistant)) = message {
					if !dispatched {
						self.dispatch_turn_end_boundary(assistant, tool_results).await;
					}
				}
				self.turn_index.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
			}
			AgentEvent::MessageStart { message } => {
				let _ = runner
					.emit("message_start", json!({ "type": "message_start", "message": message }))
					.await;
			}
			AgentEvent::MessageUpdate { message, assistant_message_event } => {
				let _ = runner
					.emit(
						"message_update",
						json!({
							"type": "message_update",
							"message": message,
							"assistantMessageEvent": assistant_message_event,
						}),
					)
					.await;
			}
			AgentEvent::MessageEnd { message } => {
				let replacement = runner
					.emit_message_end(&crate::core::extensions::types::MessageEndEvent { message: message.clone() })
					.await;
				if let Some(replacement) = replacement {
					// Untyped extension handlers can return messages with
					// null/missing content; normalize so it never enters
					// agent state or session history.
					let normalized = normalize_replacement_message(replacement);
					let target = message.clone();
					self.replace_agent_state_message(&target, &normalized);
				}
			}
			AgentEvent::ToolExecutionStart { tool_call_id, tool_name, args } => {
				let _ = runner
					.emit(
						"tool_execution_start",
						json!({
							"type": "tool_execution_start",
							"toolCallId": tool_call_id,
							"toolName": tool_name,
							"args": args,
						}),
					)
					.await;
			}
			AgentEvent::ToolExecutionUpdate { tool_call_id, tool_name, args, partial_result } => {
				let _ = runner
					.emit(
						"tool_execution_update",
						json!({
							"type": "tool_execution_update",
							"toolCallId": tool_call_id,
							"toolName": tool_name,
							"args": args,
							"partialResult": partial_result,
						}),
					)
					.await;
			}
			AgentEvent::ToolExecutionEnd { tool_call_id, tool_name, result, is_error } => {
				let _ = runner
					.emit(
						"tool_execution_end",
						json!({
							"type": "tool_execution_end",
							"toolCallId": tool_call_id,
							"toolName": tool_name,
							"result": result,
							"isError": is_error,
						}),
					)
					.await;
			}
		}
	}

	// ==== run lifecycle ====

	async fn run_agent_prompt(&self, messages: Vec<AgentMessage>) {
		self.agent_run_abort_requested.store(false, std::sync::atomic::Ordering::SeqCst);
		// Compaction before the prompt may have scheduled a retry; the new
		// prompt replaces it.
		self.failed_response.lock().take();
		self.record_selection();
		// The run records the loadout in the transcript; restored tools that
		// did not register by now are dropped, so a tool that never registers
		// does not stay pending.
		self.pending_tool_names.lock().clear();
		self.is_agent_run_active.store(true, std::sync::atomic::Ordering::SeqCst);
		self.update_idle_watch();

		let agent = Arc::clone(&self.agent);
		let result = agent.prompt(PromptInput::Messages(messages)).await;

		if result.is_ok() {
			loop {
				if self.agent_run_abort_requested.load(std::sync::atomic::Ordering::SeqCst) {
					break;
				}
				let should_continue = self.handle_post_agent_run().await;
				if should_continue {
					if self.agent_run_abort_requested.load(std::sync::atomic::Ordering::SeqCst) {
						break;
					}
					let _ = Arc::clone(&self.agent).continue_().await;
					continue;
				}
				if self.agent_run_abort_requested.load(std::sync::atomic::Ordering::SeqCst)
					|| !self.run_before_settle_boundary().await
				{
					break;
				}
				if self.agent_run_abort_requested.load(std::sync::atomic::Ordering::SeqCst) {
					break;
				}
				let _ = Arc::clone(&self.agent).continue_().await;
			}
		}

		if self.agent_run_abort_requested.load(std::sync::atomic::Ordering::SeqCst) {
			self.finish_cancelled_retry();
		}
		self.failed_response.lock().take();
		*self.run_system_prompt_options.lock() = None;
		self.flush_pending_bash_messages();
		self.flush_pending_custom_messages();
		self.emit_agent_settled().await;
	}

	async fn handle_post_agent_run(&self) -> bool {
		let message = self.last_assistant_message.lock().take();
		let tool_results = std::mem::take(&mut *self.last_assistant_tool_results.lock());
		if self.agent_run_abort_requested.load(std::sync::atomic::Ordering::SeqCst) {
			self.finish_cancelled_retry();
			return false;
		}
		let Some(message) = message else {
			return self.agent.has_queued_messages();
		};

		if self.is_retryable_error(&message) && self.prepare_retry(&message).await {
			if self.agent_run_abort_requested.load(std::sync::atomic::Ordering::SeqCst) {
				self.finish_cancelled_retry();
			}
			*self.failed_response.lock() = Some(message);
			return !self.agent_run_abort_requested.load(std::sync::atomic::Ordering::SeqCst);
		}
		if self.agent_run_abort_requested.load(std::sync::atomic::Ordering::SeqCst) {
			self.finish_cancelled_retry();
			return false;
		}

		let retry_attempt = self.retry_attempt.load(std::sync::atomic::Ordering::SeqCst);
		if message.stop_reason == pi_ai::types::StopReason::Error && retry_attempt > 0 {
			self.emit(AgentSessionEvent::AutoRetryEnd {
				success: false,
				attempt: retry_attempt,
				final_error: message.error_message.clone(),
			});
			self.retry_attempt.store(0, std::sync::atomic::Ordering::SeqCst);
		}

		let tool_results_as_messages: Vec<AgentMessage> =
			tool_results.iter().map(|result| AgentMessage::Llm(Message::ToolResult(result.clone()))).collect();
		if self.check_compaction(&message, true, tool_results_as_messages).await {
			return !self.agent_run_abort_requested.load(std::sync::atomic::Ordering::SeqCst);
		}

		// The low-level loop drains both queues before agent_end. Messages
		// queued by agent_end handlers require a fresh run before
		// pre-settlement handlers fire.
		!self.agent_run_abort_requested.load(std::sync::atomic::Ordering::SeqCst) && self.agent.has_queued_messages()
	}

	async fn run_before_settle_boundary(&self) -> bool {
		let runner = self.extension_runner();
		if !runner.has_handlers("agent_before_settle") {
			return self.agent.has_queued_messages();
		}
		self.is_before_settle.store(true, std::sync::atomic::Ordering::SeqCst);
		self.abort_during_before_settle.store(false, std::sync::atomic::Ordering::SeqCst);
		let result: Result<bool, String> = async {
			let outcome = *self.last_activity_outcome.lock();
			let weak = self.self_weak.clone();
			let build_context: BoundaryContextBuilder = Arc::new(move |drafts: &[SessionBoundaryDraft]| {
				match weak.upgrade() {
					Some(inner) => inner.build_boundary_context(drafts, "agent_before_settle"),
					None => Err("session dropped".to_string()),
				}
			});
			let result = runner
				.emit_boundary(
					"agent_before_settle",
					json!({ "type": "agent_before_settle", "outcome": outcome }),
					build_context,
				)
				.await;
			self.commit_boundary_drafts(&result.entries);
			self.flush_pending_custom_messages();
			let final_context = self.build_boundary_context(&[], "agent_before_settle")?;
			if self.abort_during_before_settle.load(std::sync::atomic::Ordering::SeqCst) {
				return Ok(false);
			}
			let should_continue = result.continue_ || self.agent.has_queued_messages();
			if should_continue && !final_context.can_continue {
				if result.continue_ {
					self.report_invalid_boundary_continuation("agent_before_settle");
				}
				return Ok(false);
			}
			Ok(should_continue)
		}
		.await;
		self.is_before_settle.store(false, std::sync::atomic::Ordering::SeqCst);
		result.unwrap_or_else(|_| self.agent.has_queued_messages())
	}

	async fn emit_agent_settled(&self) {
		if let Some(cache_warmer) = &self.cache_warmer {
			cache_warmer.on_agent_settled();
		}
		self.is_agent_run_active.store(false, std::sync::atomic::Ordering::SeqCst);
		self.is_emitting_agent_settled.store(true, std::sync::atomic::Ordering::SeqCst);
		let runner = self.extension_runner();
		let _ = runner.emit("agent_settled", json!({ "type": "agent_settled" })).await;
		self.emit(AgentSessionEvent::AgentSettled);
		self.is_emitting_agent_settled.store(false, std::sync::atomic::Ordering::SeqCst);
		self.update_idle_watch();

		let deferred = std::mem::take(&mut *self.deferred_settled_actions.lock());
		for action in deferred {
			action.await;
		}
	}

	// ==== input handling ====

	async fn run_input_handlers(
		&self,
		text: String,
		images: Option<Vec<ImageContent>>,
		source: InputSource,
		streaming_behavior: Option<StreamingBehavior>,
	) -> Option<(String, Vec<ImageContent>)> {
		let runner = self.extension_runner();
		if !runner.has_handlers("input") {
			return Some((text, images.unwrap_or_default()));
		}

		let input_result = runner
			.emit_input(
				&text,
				images,
				source,
				streaming_behavior.map(|behavior| match behavior {
					StreamingBehavior::Steer => crate::core::extensions::types::StreamingBehavior::Steer,
					StreamingBehavior::FollowUp => crate::core::extensions::types::StreamingBehavior::FollowUp,
				}),
			)
			.await;
		match input_result {
			InputEventResult::Handled => None,
			InputEventResult::Transform { text, images } => Some((text, images.unwrap_or_default())),
			InputEventResult::Continue => Some((text, Vec::new())),
		}
	}

	async fn normalize_prompt_images(&self, images: &[ImageContent]) -> NormalizedImages {
		if images.is_empty() {
			return NormalizedImages { images: Vec::new(), hints: Vec::new() };
		}

		let mut normalized_images: Vec<ImageContent> = Vec::new();
		let mut hints: Vec<String> = Vec::new();
		let resize_options = self
			.limits_model()
			.and_then(|model| model.base.input_limits)
			.and_then(|limits| limits.images)
			.map(|images| images.resize);
		for image in images {
			let bytes = crate::utils::image_convert::base64_decode(&image.data);
			let auto_resize_images = self.settings_manager.lock().get_image_auto_resize();
			let resize = resize_options.clone().flatten().map(|options| crate::utils::image_resize::ImageResizeOptions {
				max_width: options.max_width.map(|value| value as u32),
				max_height: options.max_height.map(|value| value as u32),
				max_bytes: options.max_bytes,
				jpeg_quality: options.jpeg_quality,
			});
			let processed = crate::utils::image_process::process_image(
				&bytes,
				&image.mime_type,
				Some(&crate::utils::image_process::ProcessImageOptions {
					auto_resize_images: Some(auto_resize_images),
					resize_options: resize,
				}),
			)
			.await;
			match processed {
				crate::utils::image_process::ProcessImageResult::Ok { data, mime_type, hints: processed_hints } => {
					normalized_images.push(ImageContent { data, mime_type });
					hints.extend(processed_hints);
				}
				crate::utils::image_process::ProcessImageResult::Err { message } => hints.push(message),
			}
		}
		NormalizedImages { images: normalized_images, hints }
	}

	async fn try_execute_extension_command(&self, text: &str) -> bool {
		// Parse command name and args.
		let (command_name, args) = match text.find(' ') {
			Some(space_index) => (&text[1..space_index], &text[space_index + 1..]),
			None => (&text[1..], ""),
		};

		let runner = self.extension_runner();
		let Some(command) = runner.get_command(command_name) else {
			return false;
		};

		// Get command context from the extension runner (includes session
		// control methods).
		let ctx = match runner.create_command_context() {
			Ok(ctx) => ctx,
			Err(message) => {
				runner.emit_error(&ExtensionError {
					extension_path: format!("command:{command_name}"),
					event: "command".to_string(),
					error: message,
					stack: None,
				});
				return true;
			}
		};

		(command.handler)(args.to_string(), ctx).await;
		true
	}

	/// Expand skill commands (`/skill:name args`) to their full content
	/// (`_expandSkillCommand`).
	fn expand_skill_command(&self, text: &str) -> String {
		if !text.starts_with("/skill:") {
			return text.to_string();
		}

		let (skill_name, args) = match text.find(' ') {
			Some(space_index) => (&text[7..space_index], text[space_index + 1..].trim()),
			None => (&text[7..], ""),
		};

		let skill = self
			.resource_loader
			.lock()
			.get_skills()
			.items
			.into_iter()
			.find(|skill| skill.name == skill_name);
		let Some(skill) = skill else {
			return text.to_string(); // Unknown skill, pass through.
		};

		match std::fs::read_to_string(&skill.file_path) {
			Ok(content) => {
				let body = strip_frontmatter(&content).map(|stripped| stripped).unwrap_or(content).trim().to_string();
				let skill_block = format!(
					"<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
					skill.name, skill.file_path, skill.base_dir, body
				);
				if args.is_empty() {
					skill_block
				} else {
					format!("{skill_block}\n\n{args}")
				}
			}
			Err(error) => {
				// Emit error like extension commands do.
				self.extension_runner().emit_error(&ExtensionError {
					extension_path: skill.file_path.clone(),
					event: "skill_expansion".to_string(),
					error: error.to_string(),
					stack: None,
				});
				text.to_string() // Return original on error.
			}
		}
	}

	async fn queue_user_input(
		&self,
		text: String,
		images: Option<Vec<ImageContent>>,
		behavior: StreamingBehavior,
		source: InputSource,
	) -> Result<QueuedInputDisposition, String> {
		if text.starts_with('/') {
			self.throw_if_extension_command(&text)?;
		}

		let processed_input = self
			.run_input_handlers(
				text,
				images,
				source,
				if self.is_agent_run_active.load(std::sync::atomic::Ordering::SeqCst) { Some(behavior) } else { None },
			)
			.await;
		let Some((text, images)) = processed_input else {
			return Ok(QueuedInputDisposition::Handled);
		};

		let expanded_text = self.expand_skill_command(&text);
		let expanded_text = expand_prompt_template(&expanded_text, &self.prompt_templates_public());
		let images = if images.is_empty() { None } else { Some(images) };

		if behavior == StreamingBehavior::Steer {
			self.queue_steer(expanded_text, images).await;
		} else {
			self.queue_follow_up(expanded_text, images).await;
		}
		Ok(QueuedInputDisposition::Queued)
	}

	fn prompt_templates_public(&self) -> Vec<PromptTemplate> {
		self.resource_loader.lock().get_prompts().items
	}

	async fn queue_steer(&self, text: String, images: Option<Vec<ImageContent>>) {
		self.steering_messages.lock().push(text.clone());
		self.emit_queue_update();
		let mut content: Vec<UserContent> = vec![UserContent::Text(TextContent::new(text))];
		if let Some(images) = images {
			content.extend(images.into_iter().map(UserContent::Image));
		}
		self.agent.steer(AgentMessage::Llm(Message::User(pi_ai::types::UserMessage {
			content: UserContentField::Blocks(content),
			timestamp: crate_now_ms(),
		})));
	}

	async fn queue_follow_up(&self, text: String, images: Option<Vec<ImageContent>>) {
		self.follow_up_messages.lock().push(text.clone());
		self.emit_queue_update();
		let mut content: Vec<UserContent> = vec![UserContent::Text(TextContent::new(text))];
		if let Some(images) = images {
			content.extend(images.into_iter().map(UserContent::Image));
		}
		self.agent.follow_up(AgentMessage::Llm(Message::User(pi_ai::types::UserMessage {
			content: UserContentField::Blocks(content),
			timestamp: crate_now_ms(),
		})));
	}

	fn throw_if_extension_command(&self, text: &str) -> Result<(), String> {
		let command_name = match text.find(' ') {
			Some(space_index) => &text[1..space_index],
			None => &text[1..],
		};
		if self.extension_runner().get_command(command_name).is_some() {
			return Err(format!(
				"Extension command \"/{command_name}\" cannot be queued. Use prompt() or execute the command when not streaming."
			));
		}
		Ok(())
	}

	fn append_custom_message(&self, app_message: &mut CustomMessage) {
		let _ = self.session_manager.lock().append_custom_message_entry(
			&app_message.custom_type,
			app_message.content.clone(),
			app_message.display,
			app_message.details.clone(),
		);
		self.refresh_finalized_context();
		let agent_message = custom_message_to_agent_message(app_message);
		self.emit(AgentSessionEvent::MessageStart { message: agent_message.clone() });
		self.emit(AgentSessionEvent::MessageEnd { message: agent_message });
	}

	fn flush_pending_custom_messages(&self) {
		let pending = std::mem::take(&mut *self.pending_custom_messages.lock());
		for mut app_message in pending {
			self.append_custom_message(&mut app_message);
		}
	}

	fn flush_pending_bash_messages(&self) {
		let pending = std::mem::take(&mut *self.pending_bash_messages.lock());
		if pending.is_empty() {
			return;
		}
		for bash_message in pending {
			append_bash_message(self, bash_message);
		}
		self.refresh_finalized_context();
	}

	// ==== model events ====

	async fn emit_model_select(&self, next_model: Model, previous_model: Option<Model>, source: ModelSelectSource) {
		let equal = match &previous_model {
			Some(previous) => models_are_equal(
				Some(&pi_ai::types::AnyModel::Chat(previous.clone())),
				Some(&pi_ai::types::AnyModel::Chat(next_model.clone())),
			),
			None => false,
		};
		if equal {
			return;
		}
		let event = json!({
			"type": "model_select",
			"model": next_model,
			"previousModel": previous_model,
			"source": match source {
				ModelSelectSource::Set => "set",
				ModelSelectSource::Cycle => "cycle",
				ModelSelectSource::Restore => "restore",
			},
		});
		let _ = self.extension_runner().emit("model_select", event).await;
	}

	// ==== compaction internals ====

	async fn run_default_compaction(
		&self,
		preparation: &CompactionPreparation,
		model: &Model,
		custom_instructions: Option<&str>,
		signal: Option<&AiAbortSignal>,
		reason: CompactReason,
	) -> Result<CompactionResult, String> {
		// Resolve the request only when Pi summarizes itself: routing may call
		// models or fail.
		let request = self.get_summarization_request_auth(model, signal).await?;
		compact_context(
			preparation,
			&request.model,
			request.api_key.as_deref(),
			request.headers.as_ref(),
			custom_instructions,
			signal,
			request.thinking_level,
			Some(Arc::clone(&self.agent.stream_function)),
			request.env.as_ref(),
			Some(&self.retry_policy()),
			Some(&self.summarization_retry_callbacks(SummarizationRetrySource::Compaction { reason })),
			None, // sessionId
		)
		.await
	}

	fn clear_manual_compaction_state(&self) {
		*self.compaction_abort_controller.lock() = None;
		self.update_idle_watch();
	}

	async fn check_compaction(&self, assistant_message: &AssistantMessage, skip_aborted_check: bool, tool_results: Vec<AgentMessage>) -> bool {
		let settings = match self.compaction_settings(self.model_option().as_ref()) {
			Ok(settings) => settings,
			Err(_) => return false,
		};
		if !settings.enabled {
			return false;
		}

		// Skip if message was aborted (user cancelled).
		if skip_aborted_check && assistant_message.stop_reason == pi_ai::types::StopReason::Aborted {
			return false;
		}

		// Skip the overflow check if the message came from a different model.
		let message_model = self.model_for_message(assistant_message);
		let same_model = message_model.is_some();
		let context_window = match &message_model {
			Some(model) => model.context_window,
			None => self.model_option().map(|model| model.context_window).unwrap_or(0),
		};

		// Skip compaction checks if this assistant message is older than the
		// latest compaction boundary.
		let branch_for_compaction = self.session_manager.lock().get_branch();
		let compaction_entry = get_latest_compaction_entry(&branch_for_compaction);
		if let Some(compaction_entry) = compaction_entry {
			if assistant_message.timestamp <= parse_iso_timestamp(&compaction_entry.base.timestamp) {
				return false;
			}
		}

		// Automatic cases 1 and 2: context overflow.
		let current_projection = self.session_manager.lock().build_session_projection();
		let assistant_entry_id =
			self.find_persisted_message_entry_id(&AgentMessage::Llm(Message::Assistant(assistant_message.clone())));
		let assistant_is_projected = match &assistant_entry_id {
			None => true,
			Some(entry_id) => current_projection.entries.iter().any(|entry| {
				entry.source_entry.id() == entry_id
					&& entry.messages.iter().any(|message| message.role() == "assistant")
			}),
		};
		let branch = self.session_manager.lock().get_branch();
		let assistant_index = assistant_entry_id
			.as_deref()
			.and_then(|entry_id| branch.iter().position(|entry| entry.id() == entry_id));
		let entries_after_assistant: Vec<SessionEntry> = match assistant_index {
			Some(index) => branch[index + 1..].to_vec(),
			None => Vec::new(),
		};
		let has_post_assistant_context_edit = entries_after_assistant
			.iter()
			.any(|entry| matches!(entry, SessionEntry::ContextEdit(_)));
		let latest_assistant_edit = entries_after_assistant.iter().rev().find_map(|entry| match entry {
			SessionEntry::ContextEdit(context_edit) if Some(context_edit.target_id.as_str()) == assistant_entry_id.as_deref() => {
				Some(context_edit)
			}
			_ => None,
		});
		let assistant_retained_for_explicit_recovery = assistant_entry_id.is_none()
			|| (!entries_after_assistant.iter().any(|entry| matches!(entry, SessionEntry::Compaction(_)))
				&& !matches!(&latest_assistant_edit.map(|edit| &edit.replacement), Some(crate::core::session_manager::ContextEditReplacement::Null)));
		let assistant_usage_matches_projection = assistant_is_projected && !has_post_assistant_context_edit;
		let explicit_overflow =
			assistant_message.stop_reason == pi_ai::types::StopReason::Error && is_context_overflow(assistant_message, None);
		let context_overflow = same_model
			&& ((explicit_overflow && assistant_retained_for_explicit_recovery)
				|| (assistant_usage_matches_projection && is_context_overflow(assistant_message, Some(context_window))));
		let recoverable_length = same_model
			&& assistant_is_projected
			&& message_model
				.as_ref()
				.map(|model| is_recoverable_length(assistant_message, model.max_tokens))
				.unwrap_or(false);
		if context_overflow || recoverable_length {
			let will_retry = assistant_message.stop_reason != pi_ai::types::StopReason::Stop;

			// Case 2: the response completed successfully. Compact, but do
			// not retry because agent.continue() cannot continue from a
			// completed assistant response.
			if !will_retry {
				return self.run_auto_compaction(CompactReason::Overflow, false).await;
			}

			if self.overflow_recovery_attempted.load(std::sync::atomic::Ordering::SeqCst) {
				let error_message = if context_overflow {
					"Context overflow recovery failed after one compact-and-retry attempt. Try reducing context or switching to a larger-context model."
				} else {
					"Truncated response recovery failed after one compact-and-retry attempt."
				}
				.to_string();
				self.emit(AgentSessionEvent::CompactionEnd {
					reason: CompactReason::Overflow,
					result: None,
					aborted: false,
					will_retry: false,
					error_message: Some(error_message.clone()),
				});
				self.emit_session_compact_failed(crate::core::extensions::types::SessionCompactFailedEvent {
					reason: CompactReason::Overflow,
					error_message: Some(error_message),
					aborted: false,
					will_retry: false,
					from_extension: false,
				})
				.await;
				return false;
			}

			// Persistently omit the selected final attempt before post-run
			// recovery compaction.
			self.overflow_recovery_attempted.store(true, std::sync::atomic::Ordering::SeqCst);
			self.omit_recovery_attempt(assistant_message, &tool_results);
			let retry = self.run_auto_compaction(CompactReason::Overflow, will_retry).await;
			if retry {
				*self.failed_response.lock() = Some(assistant_message.clone());
			}
			return retry;
		}

		// Case 3: threshold compaction without retry. For error messages or
		// all-zero usage messages, estimate from the last valid response.
		let projection = current_projection;
		let has_context_edits = projection.entries.iter().any(|entry| matches!(entry.source_entry, SessionEntry::ContextEdit(_)));
		let direct_context_tokens = calculate_context_tokens(&assistant_message.usage);
		let context_tokens = if has_context_edits {
			estimate_projected_context_tokens(&projection, &branch).tokens
		} else if assistant_message.stop_reason == pi_ai::types::StopReason::Error || direct_context_tokens == 0 {
			let messages = self.agent.messages();
			let estimate = estimate_context_tokens(&messages);
			// Without provider usage, estimate.tokens is the pure
			// message-size estimate. Only usage-backed estimates need the
			// stale pre-compaction check.
			if let Some(last_usage_index) = estimate.last_usage_index {
				if let Some(compaction_entry) = compaction_entry {
					if let Some(AgentMessage::Llm(Message::Assistant(usage_msg))) = messages.get(last_usage_index) {
						if usage_msg.timestamp <= parse_iso_timestamp(&compaction_entry.base.timestamp) {
							return false;
						}
					}
				}
			}
			estimate.tokens
		} else {
			direct_context_tokens
		};
		if should_compact(context_tokens, context_window, &settings) {
			return self.run_auto_compaction(CompactReason::Threshold, false).await;
		}
		false
	}

	/// Execute threshold or overflow compaction (`_runAutoCompaction`).
	async fn run_auto_compaction(&self, reason: CompactReason, will_retry: bool) -> bool {
		// D78: test seam mirrors upstream's `vi.spyOn` on this method.
		#[cfg(test)]
		{
			self.run_auto_compaction_calls.lock().push((format!("{reason:?}"), will_retry));
		}
		let override_fn = self.run_auto_compaction_override.lock().clone();
		if let Some(override_fn) = override_fn {
			return override_fn(reason, will_retry).await;
		}

		let model = self.model_option();
		let settings = self.compaction_settings(model.as_ref()).ok();
		let mut controller: Option<(u64, AbortController)> = None;
		let mut started = false;
		let mut from_extension = false;
		let mut cancelled_by_extension = false;

		let result = async {
			let Some(model) = model else {
				return Ok(false);
			};
			let Some(settings) = settings else {
				return Ok(false);
			};

			let path_entries = self.session_manager.lock().get_branch();
			let Some(preparation) = prepare_compaction(&path_entries, &settings) else {
				return Ok(false);
			};

			let abort_controller = AbortController::new();
			let abort_id = self.next_abort_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
			controller = Some((abort_id, abort_controller.clone()));
			*self.auto_compaction_abort_controller.lock() = Some((abort_id, abort_controller.clone()));
			started = true;
			self.update_idle_watch();
			self.emit(AgentSessionEvent::CompactionStart { reason });
			if abort_controller.signal().aborted() {
				return Err("Compaction cancelled".to_string());
			}

			let mut extension_compaction: Option<CompactionResult> = None;

			let runner = self.extension_runner();
			if runner.has_handlers("session_before_compact") {
				let event = SessionBeforeCompactEvent {
					preparation: preparation.clone(),
					branch_entries: path_entries.clone(),
					custom_instructions: None,
					reason,
					will_retry,
					signal: abort_controller.signal().clone(),
				};
				let dispatch = runner.emit_session_before_compact(&event).await;
				if dispatch.cancel {
					cancelled_by_extension = true;
					return Err("Compaction cancelled".to_string());
				}
				if let Some(raw) = dispatch.raw {
					if let Ok(result) = serde_json::from_value::<SessionBeforeCompactRawResult>(raw) {
						if result.cancel.unwrap_or(false) {
							cancelled_by_extension = true;
							return Err("Compaction cancelled".to_string());
						}
						if let Some(compaction) = result.compaction {
							extension_compaction = Some(compaction.into_compaction_result());
							from_extension = true;
						}
					}
				}
			}
			if abort_controller.signal().aborted() {
				return Err("Compaction cancelled".to_string());
			}

			let compaction_content = if let Some(extension_compaction) = extension_compaction {
				extension_compaction
			} else {
				let signal = bridge_crate_signal_to_ai(Some(abort_controller.signal().clone()));
				self
					.run_default_compaction(&preparation, &model, None, signal.as_ref(), reason)
					.await?
			};
			let summary = compaction_content.summary;
			let first_kept_entry_id = compaction_content.first_kept_entry_id;
			let tokens_before = compaction_content.tokens_before;
			let usage = compaction_content.usage;
			let details = compaction_content.details.map(|details| serde_json::to_value(details).unwrap_or(JsonValue::Null));
			if abort_controller.signal().aborted() {
				return Err("Compaction cancelled".to_string());
			}

			self.session_manager
				.lock()
				.append_compaction(&summary, Some(&first_kept_entry_id), tokens_before, details.clone(), Some(from_extension), usage.clone())
				.map_err(|error| error.to_string())?;
			let new_entries = self.session_manager.lock().get_entries();
			self.refresh_finalized_context();
			let estimated_tokens_after =
				estimate_messages_tokens(&self.session_manager.lock().build_session_projection().messages);

			let saved_compaction_entry = new_entries.iter().find_map(|entry| match entry {
				SessionEntry::Compaction(compaction) if compaction.summary == summary => Some(compaction.clone()),
				_ => None,
			});

			if let Some(saved_compaction_entry) = saved_compaction_entry {
				let event = json!({
					"type": "session_compact",
					"compactionEntry": saved_compaction_entry,
					"fromExtension": from_extension,
					"reason": serde_json::to_value(reason).unwrap_or(JsonValue::Null),
					"willRetry": will_retry,
				});
				let _ = runner.emit("session_compact", event).await;
			}

			let result = CompactionResult {
				summary,
				first_kept_entry_id,
				tokens_before,
				estimated_tokens_after: Some(estimated_tokens_after),
				usage,
				details: details.and_then(|details| serde_json::from_value(details).ok()),
			};
			self.emit(AgentSessionEvent::CompactionEnd { reason, result: Some(result), aborted: false, will_retry, error_message: None });

			if will_retry {
				return Ok(true);
			}

			// Auto-compaction can complete while follow-up/steering/custom
			// messages are waiting. Continue once so queued messages are
			// delivered.
			Ok(self.agent.has_queued_messages())
		}
		.await;

		let outcome = match result {
			Ok(continue_) => continue_,
			Err(error) => {
				let aborted = controller
					.as_ref()
					.map(|(_, controller)| controller.signal().aborted())
					.unwrap_or(false)
					|| cancelled_by_extension;
				if started {
					let error_message = if aborted {
						None
					} else {
						match reason {
							CompactReason::Overflow => Some(format!("Context overflow recovery failed: {error}")),
							_ => Some(format!("Auto-compaction failed: {error}")),
						}
					};
					self.emit(AgentSessionEvent::CompactionEnd {
						reason,
						result: None,
						aborted,
						will_retry: false,
						error_message: error_message.clone(),
					});
					self.emit_session_compact_failed(crate::core::extensions::types::SessionCompactFailedEvent {
						reason,
						error_message,
						aborted,
						will_retry: false,
						from_extension,
					})
					.await;
				}
				false
			}
		};

		let same = match (&*self.auto_compaction_abort_controller.lock(), &controller) {
			(Some((current_id, _)), Some((abort_id, _))) => current_id == abort_id,
			_ => false,
		};
		if same {
			*self.auto_compaction_abort_controller.lock() = None;
		}
		self.update_idle_watch();
		outcome
	}

	// ==== retry internals ====

	fn is_retryable_error(&self, message: &AssistantMessage) -> bool {
		// Context overflow is handled by compaction, not retry.
		let context_window = self
			.model_for_message(message)
			.or_else(|| self.model_option())
			.map(|model| model.context_window)
			.unwrap_or(0);
		if is_context_overflow(message, Some(context_window)) {
			return false;
		}
		is_retryable_assistant_error(message)
	}

	fn finish_cancelled_retry(&self) {
		let retry_attempt = self.retry_attempt.load(std::sync::atomic::Ordering::SeqCst);
		if retry_attempt == 0 {
			return;
		}
		self.retry_attempt.store(0, std::sync::atomic::Ordering::SeqCst);
		self.emit(AgentSessionEvent::AutoRetryEnd {
			success: false,
			attempt: retry_attempt,
			final_error: Some("Retry cancelled".to_string()),
		});
	}

	/// Prepare a retryable error for continuation with exponential backoff.
	/// Returns true if the caller should continue the agent.
	async fn prepare_retry(&self, message: &AssistantMessage) -> bool {
		let (enabled, max_retries, base_delay_ms, max_agent_delay_ms) = self.settings_manager.lock().get_retry_settings();
		if !enabled {
			return false;
		}

		let attempt = self.retry_attempt.load(std::sync::atomic::Ordering::SeqCst) + 1;
		self.retry_attempt.store(attempt, std::sync::atomic::Ordering::SeqCst);

		if attempt > max_retries as u32 {
			// Preserve the completed attempt count so post-run handling can
			// emit the final failure.
			self.retry_attempt.store(attempt - 1, std::sync::atomic::Ordering::SeqCst);
			return false;
		}

		let policy = RetryPolicy {
			enabled,
			max_retries: max_retries as u32,
			base_delay_ms,
			max_agent_delay_ms: Some(max_agent_delay_ms),
		};
		let delay_ms = retry_delay_ms(&policy, attempt);

		self.emit(AgentSessionEvent::AutoRetryStart {
			attempt,
			max_attempts: max_retries as u32,
			delay_ms,
			error_message: message.error_message.clone().unwrap_or_else(|| "Unknown error".to_string()),
		});

		// Keep the failed attempt in raw history while durably omitting it
		// from the model projection.
		self.omit_recovery_attempt(message, &[]);

		// Wait with exponential backoff (abortable).
		let controller = AbortController::new();
		*self.retry_abort_controller.lock() = Some(controller.clone());
		let signal = controller.signal().clone();
		let slept = sleep(delay_ms, Some(&signal)).await;
		*self.retry_abort_controller.lock() = None;
		if slept.is_err() {
			// Aborted during sleep — emit end event so the UI can clean up.
			self.finish_cancelled_retry();
			return false;
		}
		true
	}

	pub(crate) fn abort_retry(&self) {
		if let Some(controller) = self.retry_abort_controller.lock().as_ref() {
			controller.abort();
		}
	}

	pub(crate) fn abort_compaction(&self) {
		if let Some(controller) = self.compaction_abort_controller.lock().as_ref() {
			controller.abort();
		}
		if let Some((_, controller)) = self.auto_compaction_abort_controller.lock().as_ref() {
			controller.abort();
		}
	}

	pub(crate) fn abort_branch_summary(&self) {
		if let Some(controller) = self.branch_summary_abort_controller.lock().as_ref() {
			controller.abort();
		}
	}

	pub(crate) fn abort_bash(&self) {
		for controller in self.bash_abort_controllers.lock().values() {
			controller.abort();
		}
	}

	// ==== tool registry ====

	fn refresh_tool_registry(&self, options: RefreshToolRegistryOptions) {
		// Tools that were already activated on registration. A tool whose
		// exposure changes to `direct` or `model-only` (for example from
		// `hidden`) is activated like a new tool.
		let previous_activated_on_registration: HashSet<String> = self
			.tool_registry
			.lock()
			.keys()
			.filter(|name| self.is_activated_on_registration(name))
			.cloned()
			.collect();
		let previous_active_tool_names =
				self.agent.tools().iter().map(|tool| tool.name().to_string()).collect::<Vec<_>>();

		let runner = self.extension_runner();
		let registered_tools: Vec<crate::core::extensions::types::RegisteredTool> = runner
			.get_all_registered_tools()
			.into_iter()
			.cloned()
			.collect();
		let mut all_custom_tools: Vec<crate::core::extensions::types::RegisteredTool> = registered_tools;
		let mut all_custom_tools_for_wrap = Vec::new();
		for definition in &self.custom_tools {
			all_custom_tools.push(crate::core::extensions::types::RegisteredTool {
				definition: definition.clone(),
				source_info: create_synthetic_source_info(
					&format!("<sdk:{}>", definition.name),
					"sdk",
					None,
					None,
					None,
				),
			});
		}
		all_custom_tools.retain(|tool| self.is_allowed_tool(&tool.definition.name));
		all_custom_tools_for_wrap = all_custom_tools.clone();

		let mut definition_registry: HashMap<String, ToolDefinitionEntry> = HashMap::new();
		for (name, definition) in self.base_tool_definitions.lock().iter() {
			if !self.is_allowed_tool(name) {
				continue;
			}
			definition_registry.insert(
				name.clone(),
				ToolDefinitionEntry {
					definition: definition.clone(),
					source_info: create_synthetic_source_info(&format!("{BUILTIN_PATH_PREFIX}{name}"), "builtin", None, None, None),
				},
			);
		}
		for tool in all_custom_tools {
			definition_registry.insert(
				tool.definition.name.clone(),
				ToolDefinitionEntry { definition: tool.definition.clone(), source_info: tool.source_info.clone() },
			);
		}
		*self.tool_definitions.lock() = definition_registry.clone();

		let snippets: HashMap<String, String> = definition_registry
			.values()
			.filter_map(|entry| {
				self.normalize_prompt_snippet(entry.definition.prompt_snippet.as_deref()).map(|snippet| (entry.definition.name.clone(), snippet))
			})
			.collect();
		*self.tool_prompt_snippets.lock() = snippets;
		let guidelines: HashMap<String, Vec<String>> = definition_registry
			.values()
			.filter_map(|entry| {
				let normalized = self.normalize_prompt_guidelines(entry.definition.prompt_guidelines.as_deref());
				(!normalized.is_empty()).then(|| (entry.definition.name.clone(), normalized))
			})
			.collect();
		*self.tool_prompt_guidelines.lock() = guidelines;

		let wrapped_extension_tools = wrap_registered_tools(all_custom_tools_for_wrap, Arc::clone(&runner));
		let wrapped_built_in_tools = wrap_registered_tools(
			self.base_tool_definitions
				.lock()
				.values()
				.filter(|definition| self.is_allowed_tool(&definition.name))
				.map(|definition| crate::core::extensions::types::RegisteredTool {
					definition: definition.clone(),
					source_info: create_synthetic_source_info(
						&format!("{BUILTIN_PATH_PREFIX}{}", definition.name),
						"builtin",
						None,
						None,
						None,
					),
				})
				.collect(),
			Arc::clone(&runner),
		);

		let mut tool_registry: HashMap<String, AgentTool> =
			wrapped_built_in_tools.into_iter().map(|tool| (tool.name().to_string(), tool)).collect();
		for tool in wrapped_extension_tools {
			tool_registry.insert(tool.name().to_string(), tool);
		}
		*self.tool_registry.lock() = tool_registry;

		let mut next_active_tool_names: Vec<String> = match &options.active_tool_names {
			Some(names) => names.clone(),
			None => previous_active_tool_names.clone(),
		}
		.into_iter()
		.filter(|name| self.is_allowed_tool(name))
		.collect();

		if let Some(allowed_tool_names) = &self.allowed_tool_names {
			for tool_name in self.tool_registry.lock().keys() {
				// Naming a tool activates it even when it is not active by
				// default.
				if allowed_tool_names.contains(tool_name) && self.is_declarable(tool_name) {
					next_active_tool_names.push(tool_name.clone());
				}
			}
		} else if options.include_all_extension_tools {
			for tool in self.tool_registry.lock().keys() {
				if self.is_activated_on_registration(tool) {
					next_active_tool_names.push(tool.clone());
				}
			}
		} else if options.active_tool_names.is_none() {
			for tool_name in self.tool_registry.lock().keys() {
				if !previous_activated_on_registration.contains(tool_name) && self.is_activated_on_registration(tool_name) {
					next_active_tool_names.push(tool_name.clone());
				}
			}
		}
		// Pending tools that are registered now become active.
		next_active_tool_names.extend(self.pending_tool_names.lock().iter().cloned());

		let deduped: Vec<String> = {
			let mut seen = HashSet::new();
			next_active_tool_names.into_iter().filter(|name| seen.insert(name.clone())).collect()
		};
		self.set_active_tools(deduped);
	}

	/// Whether activating the tool declares it to the model.
	fn is_declarable(&self, name: &str) -> bool {
		matches!(self.get_tool_exposure(name), ToolExposure::Direct | ToolExposure::ModelOnly)
	}

	/// Whether registering the tool activates it, which declares it to the
	/// model.
	fn is_activated_on_registration(&self, name: &str) -> bool {
		if !self.is_declarable(name) {
			return false;
		}
		self.tool_definitions
			.lock()
			.get(name)
			.map(|entry| entry.definition.default_active != Some(false))
			.unwrap_or(false)
	}

	fn is_allowed_tool(&self, name: &str) -> bool {
		(self.allowed_tool_names.as_ref().map(|allowed| allowed.contains(name)).unwrap_or(true))
			&& !self.excluded_tool_names.as_ref().map(|excluded| excluded.contains(name)).unwrap_or(false)
	}

	fn get_tool_exposure(&self, name: &str) -> ToolExposure {
		self.tool_definitions
			.lock()
			.get(name)
			.and_then(|entry| entry.definition.exposure)
			.unwrap_or(ToolExposure::Direct)
	}

	/// Tools callable through `ctx.executeTool()`: the active `direct` tools
	/// and every registered `codemode` or `deferred` tool.
	pub(crate) fn get_callable_tools(&self, active: Option<HashSet<String>>) -> Vec<AgentTool> {
		let active = active.unwrap_or_else(|| self.agent.tools().iter().map(|tool| tool.name().to_string()).collect::<HashSet<_>>());
		self.tool_registry
			.lock()
			.values()
			.filter(|tool| {
				let exposure = self.get_tool_exposure(tool.name());
				exposure == ToolExposure::Codemode
					|| exposure == ToolExposure::Deferred
					|| (exposure == ToolExposure::Direct && active.contains(tool.name()))
			})
			.cloned()
			.collect()
	}

	pub(crate) fn set_active_tools(&self, tool_names: Vec<String>) {
		let tools = self.apply_tool_loadout(&tool_names);
		let mut pending = self.pending_tool_names.lock();
		for tool in &tools {
			pending.remove(tool.name());
		}
		drop(pending);
		let names: Vec<String> = tools.iter().map(|tool| tool.name().to_string()).collect();
		self.rebuild_system_prompt(&names);
	}

	/// Set the agent's tools for the given active tool names and return them
	/// (`_applyToolLoadout`).
	fn apply_tool_loadout(&self, tool_names: &[String]) -> Vec<AgentTool> {
		let registry = self.tool_registry.lock().clone();
		let mut seen = HashSet::new();
		let tools: Vec<AgentTool> = tool_names
			.iter()
			.filter(|name| seen.insert((*name).clone()))
			.filter_map(|name| {
				let tool = registry.get(name)?;
				(self.get_tool_exposure(name) != ToolExposure::Hidden).then(|| tool.clone())
			})
			.collect();

		let definition_registry = self.tool_definitions.lock().clone();
		let hooks: Vec<&ToolDefinitionEntry> = tools
			.iter()
			.filter_map(|tool| {
				let entry = definition_registry.get(tool.name())?;
				entry.definition.prepare_loadout.as_ref().map(|_| entry)
			})
			.collect();
		let mut hidden: HashSet<String> = HashSet::new();
		let mut declared = tools.clone();
		if !hooks.is_empty() {
			let loadout = ToolLoadout {
				declared: tools.clone(),
				callable: self.get_callable_tools(Some(tools.iter().map(|tool| tool.name().to_string()).collect())),
				registered: registry.values().cloned().collect(),
				get_exposure: exposure_closure(self),
				get_namespace: {
					let definition_registry = definition_registry.clone();
					Arc::new(move |name: &str| definition_registry.get(name).and_then(|entry| entry.definition.namespace.clone()))
				},
			};
			let mut descriptions: HashMap<String, String> = HashMap::new();
			for entry in &hooks {
				let Some(prepare_loadout) = &entry.definition.prepare_loadout else { continue };
				match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prepare_loadout(&loadout))) {
					Ok(changes) => {
						if let Some(changes) = changes {
							if let Some(change_descriptions) = changes.descriptions {
								for (name, description) in change_descriptions {
									descriptions.insert(name, description);
								}
							}
							if let Some(hidden_declarations) = changes.hidden_declarations {
								for name in hidden_declarations {
									hidden.insert(name);
								}
							}
						}
					}
					Err(payload) => {
						self.extension_runner().emit_error(&ExtensionError {
							extension_path: entry.source_info.path.clone(),
							event: "prepare_loadout".to_string(),
							error: panic_payload_message(payload),
							stack: None,
						});
					}
				}
			}
			declared = tools
				.into_iter()
				.map(|mut tool| match descriptions.get(tool.name()) {
					Some(description) => {
						tool.tool.description = description.clone();
						tool
					}
					None => tool,
				})
				.collect();
		}
		*self.hidden_declarations.lock() = hidden;
		self.agent.set_tools(declared.clone());
		declared
	}

	fn normalize_prompt_snippet(&self, text: Option<&str>) -> Option<String> {
		let text = text?;
		let one_line = text
			.split(['\r', '\n'])
			.collect::<Vec<_>>()
			.join(" ")
			.split_whitespace()
			.collect::<Vec<_>>()
			.join(" ");
		let trimmed = one_line.trim().to_string();
		(!trimmed.is_empty()).then_some(trimmed)
	}

	fn normalize_prompt_guidelines(&self, guidelines: Option<&[String]>) -> Vec<String> {
		let Some(guidelines) = guidelines else { return Vec::new() };
		if guidelines.is_empty() {
			return Vec::new();
		}
		let mut unique: Vec<String> = Vec::new();
		for guideline in guidelines {
			let normalized = guideline.trim().to_string();
			if !normalized.is_empty() && !unique.contains(&normalized) {
				unique.push(normalized);
			}
		}
		unique
	}

	pub(crate) fn rebuild_system_prompt(&self, tool_names: &[String]) {
		let registry = self.tool_registry.lock();
		let valid_tool_names: Vec<String> = tool_names
			.iter()
			.filter(|name| registry.contains_key(*name))
			.cloned()
			.collect();
		let snippets = self.tool_prompt_snippets.lock().clone();
		let hidden = self.hidden_declarations.lock().clone();
		let mut tool_snippets = std::collections::BTreeMap::new();
		for name in registry.keys() {
			// Tools without a snippet are not listed. Hidden tools are only
			// callable through another tool.
			if let Some(snippet) = snippets.get(name) {
				if !hidden.contains(name) {
					tool_snippets.insert(name.clone(), snippet.clone());
				}
			}
		}
		drop(registry);

		let loader = self.resource_loader.lock();
		let loader_system_prompt = loader.get_system_prompt();
		let loader_append_system_prompt = loader.get_append_system_prompt();
		let append_system_prompt = if loader_append_system_prompt.is_empty() {
			String::new()
		} else {
			loader_append_system_prompt.join("\n\n")
		};
		let loaded_skills = loader.get_skills().items;
		let loaded_context_files: Vec<crate::core::extensions::types::ContextFileEntry> = loader
			.get_agents_files()
			.into_iter()
			.map(|file| crate::core::extensions::types::ContextFileEntry { path: file.path, content: file.content })
			.collect();
		drop(loader);

		*self.base_system_prompt_options.lock() = Some(normalize_build_system_prompt_options(
			&crate::core::extensions::types::BuildSystemPromptOptions {
				cwd: self.cwd.clone(),
				skills: Some(loaded_skills),
				context_files: Some(loaded_context_files),
				custom_prompt: loader_system_prompt,
				append_system_prompt: if append_system_prompt.is_empty() { None } else { Some(append_system_prompt) },
				selected_tools: Some(valid_tool_names),
				tool_snippets: Some(tool_snippets),
				tool_guidelines: Some(self.tool_prompt_guidelines.lock().clone().into_iter().collect()),
				..Default::default()
			},
		));
	}

	/// Apply a prompt and tool loadout for the next request
	/// (`_preparePromptAndToolLoadout`).
	fn prepare_prompt_and_tool_loadout(
		&self,
		options: &mut crate::core::extensions::types::NormalizedBuildSystemPromptOptions,
		messages: Option<&[AgentMessage]>,
	) -> Option<SystemMessage> {
		let applied = self.apply_tool_loadout(&options.selected_tools);
		options.selected_tools = applied.iter().map(|tool| tool.name().to_string()).collect();
		// The tool list must match the declarations the request carries, so
		// hidden tools are not listed.
		let hidden = self.hidden_declarations.lock().clone();
		options.tool_snippets.retain(|name, _| !hidden.contains(name));
		let messages = messages
			.map(|messages| messages.to_vec())
			.unwrap_or_else(|| self.agent.messages());
		let llm: Vec<Message> = messages.iter().filter_map(|message| message.as_llm().cloned()).collect();
		let current_sections = pi_ai::utils::transcript::get_current_system_message(&llm)
			.and_then(|system| system.sections)
			.unwrap_or_default();
		let sections = build_system_prompt_sections(&build_options_from_normalized(options)).ok()?;
		let patch = diff_system_prompt_sections(&current_sections, &sections)?;
		Some(SystemMessage {
			content: pi_ai::types::SystemContent::Text(String::new()),
			sections: Some(patch),
			tools_added: None,
			tools_removed: None,
			timestamp: crate_now_ms(),
		})
	}

	/// Restore the active tool loadout declared by the session transcript
	/// (`_restoreToolsFromTranscript`).
	pub(crate) fn restore_tools_from_transcript(&self) {
		self.pending_tool_names.lock().clear();
		let context = self.session_manager.lock().build_session_context();
		let llm: Vec<Message> = context.messages.iter().filter_map(|message| message.as_llm().cloned()).collect();
		let Some(current) = pi_ai::utils::transcript::get_current_system_message(&llm) else {
			return;
		};
		let names: Vec<String> = current.tools_added.unwrap_or_default().iter().map(|tool| tool.name.clone()).collect();
		{
			let mut pending = self.pending_tool_names.lock();
			for name in &names {
				if self.is_allowed_tool(name) {
					pending.insert(name.clone());
				}
			}
		}
		self.set_active_tools(names);
	}

	// ==== runtime construction ====

	pub(crate) fn build_runtime(&self, options: BuildRuntimeOptions) {
		let (auto_resize_images, shell_command_prefix, shell_path) = {
			let settings = self.settings_manager.lock();
			(settings.get_image_auto_resize(), settings.get_shell_command_prefix(), settings.get_shell_path())
		};
		let base_tool_definitions: Vec<(String, ToolDefinition)> = match &self.base_tools_override {
			Some(overrides) => overrides
				.iter()
				.map(|tool| (tool.name().to_string(), create_tool_definition_from_agent_tool(tool)))
				.collect(),
			None => create_all_tool_definitions(
				&self.cwd,
				Some(&crate::core::tools::index::ToolsOptions {
					read: Some(crate::core::tools::read::ReadToolOptions { auto_resize_images: Some(auto_resize_images), ..Default::default() }),
					bash: Some(crate::core::tools::bash::BashToolOptions {
						command_prefix: shell_command_prefix.clone(),
						shell_path: shell_path.clone(),
						..Default::default()
					}),
					..Default::default()
				}),
			)
			.into_iter()
			.map(|definition| (definition.name.clone(), definition))
			.collect(),
		};
		*self.base_tool_definitions.lock() = base_tool_definitions.into_iter().collect();

		let extensions_result = {
			let loader = self.resource_loader.lock();
			loader.get_extensions().clone()
		};
		if let Some(flag_values) = &options.flag_values {
			for (name, value) in flag_values {
				extensions_result.runtime.state.set_flag_value(name, value.clone());
			}
		}

		let model_registry = Arc::new(ModelRegistry::new(Arc::clone(&self.model_runtime)));
		let runner = Arc::new(ExtensionRunner::new(
			extensions_result.extensions,
			extensions_result.runtime,
			&self.cwd,
			Arc::clone(&self.session_manager),
			model_registry,
		));
		if let Some(extension_runner_ref) = &self.extension_runner_ref {
			*extension_runner_ref.write() = Some(Arc::clone(&runner));
		}
		self.bind_extension_core(&runner);
		self.apply_extension_bindings(&runner);
		*self.extension_runner.write() = Some(runner);

		let default_active_tool_names: Vec<String> = match &self.base_tools_override {
			Some(overrides) => overrides.iter().map(|tool| tool.name().to_string()).collect(),
			None => ["read", "bash", "edit", "write"].iter().map(|name| name.to_string()).collect(),
		};
		let base_active_tool_names = options.active_tool_names.unwrap_or(default_active_tool_names);
		self.refresh_tool_registry(RefreshToolRegistryOptions {
			active_tool_names: Some(base_active_tool_names),
			include_all_extension_tools: options.include_all_extension_tools,
		});
	}

	pub(crate) fn apply_extension_bindings(&self, runner: &ExtensionRunner) {
		runner.set_ui_context(self.extension_ui_context.lock().clone(), *self.extension_mode.lock());
		runner.bind_command_context(self.extension_command_context_actions.lock().clone());

		if let Some(unsubscriber) = self.extension_error_unsubscriber.lock().take() {
			unsubscriber();
		}
		if let Some(listener) = self.extension_error_listener.lock().clone() {
			*self.extension_error_unsubscriber.lock() = Some(runner.on_error(listener));
		}
	}

	pub(crate) fn refresh_current_model_from_registry(&self) {
		let current_model = self.model_option();
		let Some(current_model) = current_model else { return };
		let refreshed_model = pi_ai::Models::get_model(&*self.model_runtime, current_model.provider(), current_model.id());
		match refreshed_model {
			Some(refreshed_model) if refreshed_model != current_model => {
				self.agent.set_model(refreshed_model);
			}
			_ => {}
		}
	}

	fn bind_extension_core(&self, runner: &ExtensionRunner) {
		let weak_for_commands = self.self_weak.clone();
		let get_commands = move || -> Vec<SlashCommandInfo> {
			let Some(inner) = weak_for_commands.upgrade() else { return Vec::new() };
			let session = AgentSession { inner };
			let runner = session.extension_runner();
			let extension_commands: Vec<SlashCommandInfo> = runner
				.get_registered_commands()
				.into_iter()
				.map(|command| {
					let source_info = command.source_info.clone();
					let description = command.description.clone();
					SlashCommandInfo {
						name: command.invocation_name,
						description,
						source: crate::core::extensions::types::SlashCommandSource::Extension,
						source_info,
					}
				})
				.collect();

			let templates: Vec<SlashCommandInfo> = session
				.inner.prompt_templates_public()
				.into_iter()
				.map(|template| SlashCommandInfo {
					name: template.name,
					description: Some(template.description),
					source: crate::core::extensions::types::SlashCommandSource::Prompt,
					source_info: template.source_info,
				})
				.collect();

			let skills: Vec<SlashCommandInfo> = session
				.inner.resource_loader
				.lock()
				.get_skills()
				.items
				.into_iter()
				.map(|skill| SlashCommandInfo {
					name: format!("skill:{}", skill.name),
					description: Some(skill.description),
					source: crate::core::extensions::types::SlashCommandSource::Skill,
					source_info: skill.source_info,
				})
				.collect();

			extension_commands.into_iter().chain(templates).chain(skills).collect()
		};

		let actions = ExtensionActions {
			send_message: Some(Arc::new({
				let weak = self.self_weak.clone();
				move |message, options| {
				let options = options.map(|options| SendMessageOptions {
					trigger_turn: options.trigger_turn,
					deliver_as: options.deliver_as.map(|target| match target {
						crate::core::extensions::types::DeliveryTarget::Steer => DeliveryTarget::Steer,
						crate::core::extensions::types::DeliveryTarget::FollowUp => DeliveryTarget::FollowUp,
						crate::core::extensions::types::DeliveryTarget::NextTurn => DeliveryTarget::NextTurn,
					}),
				});
				let weak = weak.clone();
				Box::pin(async move {
					let Some(inner) = weak.upgrade() else { return };
					let runner = inner.extension_runner();
					let session = AgentSession { inner };
					if let Err(error) = session.send_custom_message(message, options).await {
						runner.emit_error(&ExtensionError {
							extension_path: "<runtime>".to_string(),
							event: "send_message".to_string(),
							error,
							stack: None,
						});
					}
				})
			}})),
			send_user_message: Some(Arc::new({
				let weak = self.self_weak.clone();
				move |content, options| {
				let weak = weak.clone();
				Box::pin(async move {
					let Some(inner) = weak.upgrade() else { return };
					let runner = inner.extension_runner();
					let session = AgentSession { inner };
					if let Err(error) = session.send_user_message(content, options).await {
						runner.emit_error(&ExtensionError {
							extension_path: "<runtime>".to_string(),
							event: "send_user_message".to_string(),
							error,
							stack: None,
						});
					}
				})
			}})),
			append_entry: Some(Arc::new({
				let weak = self.self_weak.clone();
				move |custom_type: &str, data: Option<JsonValue>| {
				let Some(inner) = weak.upgrade() else { return };
				let entry_id = match inner.session_manager.lock().append_custom_entry(custom_type, data) {
					Ok(entry_id) => entry_id,
					Err(_) => return,
				};
				let entry = inner.session_manager.lock().get_entry(&entry_id).cloned();
				if let Some(entry) = entry {
					inner.emit(AgentSessionEvent::EntryAppended { entry });
				}
			}})),
			set_session_name: Some(Arc::new({ let weak = self.self_weak.clone(); move |name| {
				if let Some(inner) = weak.upgrade() {
					AgentSession { inner }.set_session_name(&name);
				}
			}})),
			get_session_name: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade().and_then(|inner| inner.session_manager.lock().get_session_name())
			}})),
			set_label: Some(Arc::new({ let weak = self.self_weak.clone(); move |entry_id: &str, label: Option<String>| {
				if let Some(inner) = weak.upgrade() {
					let _ = inner.session_manager.lock().append_label_change(entry_id, label);
				}
			}})),
			get_active_tools: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade().map(|inner| AgentSession { inner }.get_active_tool_names()).unwrap_or_default()
			}})),
			get_all_tools: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade().map(|inner| AgentSession { inner }.get_all_tools()).unwrap_or_default()
			}})),
			get_settings: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade()
					.map(|inner| inner.settings_manager.lock().get_settings())
					.unwrap_or_else(Settings::new)
			}})),
			set_active_tools: Some(Arc::new({ let weak = self.self_weak.clone(); move |tool_names| {
				if let Some(inner) = weak.upgrade() {
					AgentSession { inner }.set_active_tools_by_name(tool_names);
				}
			}})),
			refresh_tools: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				if let Some(inner) = weak.upgrade() {
					inner.refresh_tool_registry(RefreshToolRegistryOptions { active_tool_names: None, include_all_extension_tools: false });
				}
			}})),
			get_commands: Some(Arc::new(move || get_commands())),
			set_model: Some(Arc::new({ let weak = self.self_weak.clone(); move |model| {
				let weak = weak.clone();
				Box::pin(async move {
					let Some(inner) = weak.upgrade() else { return false };
					if !inner.model_runtime.has_configured_auth(model.provider()) {
						return false;
					}
					AgentSession { inner }.set_model(model, ModelMutationOptions::default()).await.is_ok()
				})
			}})),
			get_thinking_level: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade().map(|inner| inner.agent.thinking_level()).unwrap_or(AgentThinkingLevel::Off)
			}})),
			set_thinking_level: Some(Arc::new({ let weak = self.self_weak.clone(); move |level| {
				if let Some(inner) = weak.upgrade() {
					AgentSession { inner }.set_thinking_level(level, ModelMutationOptions::default());
				}
			}})),
		};

		let context_actions = ExtensionContextActions {
			get_model: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade().and_then(|inner| inner.model_option())
			}})),
			get_scoped_models: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade()
					.map(|inner| {
						inner
							.scoped_models
							.lock()
							.iter()
							.map(|entry| crate::core::model_resolver::ScopedModel {
								model: entry.model.clone(),
								thinking_level: entry.thinking_level,
							})
							.collect()
					})
					.unwrap_or_default()
			}})),
			is_idle: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade().map(|inner| AgentSession { inner }.is_idle()).unwrap_or(true)
			}})),
			is_project_trusted: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade()
					.map(|inner| inner.settings_manager.lock().is_project_trusted())
					.unwrap_or(false)
			}})),
			get_signal: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade()
					.and_then(|inner| inner.agent.signal())
					.map(|signal| {
						let controller = crate::utils::abort::AbortController::new();
						let bridged = controller.signal().clone();
						tokio::spawn(async move {
							signal.aborted_future().await;
							controller.abort();
						});
						bridged
					})
			}})),
			abort: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				let Some(inner) = weak.upgrade() else { return };
				let abort_handler = inner.extension_abort_handler.lock().clone();
				if let Some(abort_handler) = abort_handler {
					abort_handler();
					return;
				}
				tokio::spawn(async move {
					AgentSession { inner }.abort().await;
				});
			}})),
			has_pending_messages: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade().map(|inner| AgentSession { inner }.pending_message_count() > 0).unwrap_or(false)
			}})),
			shutdown: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				if let Some(inner) = weak.upgrade() {
					if let Some(shutdown_handler) = inner.extension_shutdown_handler.lock().clone() {
						shutdown_handler();
					}
				}
			}})),
			get_context_usage: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade().and_then(|inner| AgentSession { inner }.get_context_usage())
			}})),
			compact: Some(Arc::new({ let weak = self.self_weak.clone(); move |options: Option<CompactOptions>| {
				let Some(inner) = weak.upgrade() else { return };
				let session = AgentSession { inner };
				let custom_instructions = options.as_ref().and_then(|options| options.custom_instructions.clone());
				tokio::spawn(async move {
					match session.compact(custom_instructions).await {
						Ok(result) => {
							if let Some(options) = options {
								if let Some(on_complete) = &options.on_complete {
									on_complete(result);
								}
							}
						}
						Err(error) => {
							if let Some(options) = options {
								if let Some(on_error) = &options.on_error {
									on_error(error);
								}
							}
						}
					}
				});
			}})),
			get_system_prompt: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade().map(|inner| AgentSession { inner }.system_prompt()).unwrap_or_default()
			}})),
			get_system_prompt_options: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade()
					.map(|inner| {
						build_options_from_normalized(
							&inner
								.base_system_prompt_options
								.lock()
								.clone()
								.unwrap_or_else(empty_normalized_prompt_options),
						)
					})
					.unwrap_or_default()
			}})),
			execute_tool: Some(Arc::new({ let weak = self.self_weak.clone(); move |caller_id: String, name: String, args: JsonValue, options: ExecuteToolOptions| {
				let weak = weak.clone();
				Box::pin(async move {
					match weak.upgrade() {
						Some(inner) => {
							let object_args = args.as_object().cloned();
							inner.execute_nested_tool_call(&caller_id, &name, object_args, options).await
						}
						None => AgentToolCallOutcome {
							tool_call: ToolCall {
								id: caller_id,
								name,
								arguments: serde_json::Map::new(),
								thought_signature: None,
								namespace: None,
							},
							result: crate_error_tool_result("session dropped"),
							is_error: true,
						},
					}
				})
			}})),
			get_callable_tools: Some(Arc::new({ let weak = self.self_weak.clone(); move || {
				weak.upgrade().map(|inner| inner.get_callable_tools(None)).unwrap_or_default()
			}})),
		};

		let provider_actions = ProviderActions {
			register_provider: Some(Arc::new({ let weak = self.self_weak.clone(); move |name: &str, config: &crate::core::extensions::types::ProviderConfig| {
				if let Some(inner) = weak.upgrade() {
					let _ = inner
						.model_runtime
						.register_provider(name, crate::core::provider_composer::ProviderConfigInput::from(config.clone()));
					inner.refresh_current_model_from_registry();
				}
			}})),
			register_native_provider: Some(Arc::new({ let weak = self.self_weak.clone(); move |provider: &Arc<dyn pi_ai::Provider>| {
				if let Some(inner) = weak.upgrade() {
					let _ = inner.model_runtime.register_native_provider(Arc::clone(provider));
					inner.refresh_current_model_from_registry();
				}
			}})),
			unregister_provider: Some(Arc::new({ let weak = self.self_weak.clone(); move |name: &str| {
				if let Some(inner) = weak.upgrade() {
					inner.model_runtime.unregister_provider(name);
					inner.refresh_current_model_from_registry();
				}
			}})),
			register_virtual_model: Some(Arc::new({ let weak = self.self_weak.clone(); move |definition: &VirtualModelDefinition| {
				if let Some(inner) = weak.upgrade() {
					let _ = inner.model_runtime.register_extension_virtual_model(definition.clone());
					inner.refresh_current_model_from_registry();
				}
			}})),
			unregister_virtual_model: Some(Arc::new({ let weak = self.self_weak.clone(); move |provider: &str, id: &str| {
				if let Some(inner) = weak.upgrade() {
					inner.model_runtime.unregister_virtual_model(provider, id);
					inner.refresh_current_model_from_registry();
				}
			}})),
		};

		runner.bind_core(actions, context_actions, Some(provider_actions));
	}
}

// ============================================================================
// Small helpers shared above
// ============================================================================

struct RefreshToolRegistryOptions {
	active_tool_names: Option<Vec<String>>,
	include_all_extension_tools: bool,
}

struct BuildRuntimeOptions {
	active_tool_names: Option<Vec<String>>,
	flag_values: Option<HashMap<String, crate::core::extensions::types::FlagValue>>,
	include_all_extension_tools: bool,
}

struct NormalizedImages {
	images: Vec<ImageContent>,
	hints: Vec<String>,
}

/// The nested-call host backed by the session's tool pipeline (D68).
struct NestedHost {
	inner: std::sync::Weak<AgentSessionInner>,
}

impl NestedToolCallHost for NestedHost {
	fn get_tools(&self) -> Vec<AgentTool> {
		match self.inner.upgrade() {
			Some(inner) => inner.get_callable_tools(None),
			None => Vec::new(),
		}
	}

	fn is_sequential(&self) -> bool {
		match self.inner.upgrade() {
			Some(inner) => inner.agent.tool_execution == pi_agent::types::ToolExecutionMode::Sequential,
			None => false,
		}
	}

	fn run_tool_call(
		&self,
		tool_call: ToolCall,
		parent_tool_call_id: &str,
		signal: Option<crate::utils::abort::AbortSignal>,
		on_update: NestedUpdateForwarder,
	) -> BoxFuture<'static, AgentToolCallOutcome> {
		let inner = self.inner.upgrade();
		Box::pin(async move {
			let Some(inner) = inner else {
				return AgentToolCallOutcome { tool_call, result: crate_error_tool_result("session dropped"), is_error: true };
			};
			let inner_arc = Arc::clone(&inner);
			let Some(assistant_message) = inner.find_last_assistant_message() else {
				return AgentToolCallOutcome {
					tool_call,
					result: crate_error_tool_result("No assistant message issued this call"),
					is_error: true,
				};
			};
			pi_agent::run_tool_call(
				tool_call,
				pi_agent::RunToolCallOptions {
					tools: inner.get_callable_tools(None),
					assistant_message,
					context: pi_agent::AgentContext {
						messages: inner.agent.messages(),
						tools: Some(inner.agent.tools()),
					},
					signal: bridge_crate_signal_to_ai(signal),
					on_update: Some(on_update),
					before_tool_call: {
						let inner = Arc::clone(&inner_arc);
						Some(Arc::new(move |context: BeforeToolCallContext, _signal| {
							let inner = Arc::clone(&inner);
							Box::pin(async move { inner.before_tool_call(&context).await })
						}))
					},
					after_tool_call: {
						let inner = Arc::clone(&inner_arc);
						Some(Arc::new(move |context: AfterToolCallContext, _signal| {
							let inner = Arc::clone(&inner);
							Box::pin(async move { inner.after_tool_call(&context).await })
						}))
					},
				},
			)
			.await
		})
	}

	fn emit(&self, event: NestedToolExecutionEvent) {
		let Some(inner) = self.inner.upgrade() else { return };
		let runner = inner.extension_runner();
		let (event_type, value): (&str, JsonValue) = match &event {
			NestedToolExecutionEvent::Start { tool_call_id, tool_name, args, parent_tool_call_id } => (
				"tool_execution_start",
				json!({ "type": "tool_execution_start", "toolCallId": tool_call_id, "toolName": tool_name, "args": args, "parentToolCallId": parent_tool_call_id }),
			),
			NestedToolExecutionEvent::Update { tool_call_id, tool_name, args, partial_result, parent_tool_call_id } => (
				"tool_execution_update",
				json!({ "type": "tool_execution_update", "toolCallId": tool_call_id, "toolName": tool_name, "args": args, "partialResult": partial_result, "parentToolCallId": parent_tool_call_id }),
			),
			NestedToolExecutionEvent::End { tool_call_id, tool_name, result, is_error, parent_tool_call_id } => (
				"tool_execution_end",
				json!({ "type": "tool_execution_end", "toolCallId": tool_call_id, "toolName": tool_name, "result": result, "isError": is_error, "parentToolCallId": parent_tool_call_id }),
			),
		};
		// Mirror the nested event both into the extension runner and the
		// public listeners (with `parentToolCallId`, D75).
		let listener_event = match &event {
			NestedToolExecutionEvent::Start { tool_call_id, tool_name, args, parent_tool_call_id } => AgentSessionEvent::ToolExecutionStart {
				tool_call_id: tool_call_id.clone(),
				tool_name: tool_name.clone(),
				args: args.clone(),
				parent_tool_call_id: Some(parent_tool_call_id.clone()),
			},
			NestedToolExecutionEvent::Update { tool_call_id, tool_name, args, partial_result, parent_tool_call_id } => {
				AgentSessionEvent::ToolExecutionUpdate {
					tool_call_id: tool_call_id.clone(),
					tool_name: tool_name.clone(),
					args: args.clone(),
					partial_result: serde_json::to_value(partial_result).unwrap_or(JsonValue::Null),
					parent_tool_call_id: Some(parent_tool_call_id.clone()),
				}
			}
			NestedToolExecutionEvent::End { tool_call_id, tool_name, result, is_error, parent_tool_call_id } => {
				AgentSessionEvent::ToolExecutionEnd {
					tool_call_id: tool_call_id.clone(),
					tool_name: tool_name.clone(),
					result: serde_json::to_value(result).unwrap_or(JsonValue::Null),
					is_error: *is_error,
					parent_tool_call_id: Some(parent_tool_call_id.clone()),
				}
			}
		};
		inner.emit(listener_event);
		tokio::spawn(async move {
			let _ = runner.emit(event_type, value).await;
		});
	}
}

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
	if let Some(message) = payload.downcast_ref::<&str>() {
		return (*message).to_string();
	}
	if let Some(message) = payload.downcast_ref::<String>() {
		return message.clone();
	}
	"extension handler panicked".to_string()
}

fn crate_error_tool_result(message: &str) -> pi_agent::AgentToolResult {
	pi_agent::AgentToolResult {
		content: vec![UserContent::Text(TextContent::new(message.to_string()))],
		details: json!({}),
		structured_content: None,
		is_error: Some(true),
		terminate: None,
		usage: None,
	}
}

fn normalize_replacement_message(replacement: AgentMessage) -> AgentMessage {
	let is_llm = matches!(replacement, AgentMessage::Llm(_));
	if !is_llm {
		return replacement;
	}
	match replacement {
		AgentMessage::Llm(Message::User(mut user)) => {
			if matches!(user.content, UserContentField::Blocks(ref blocks) if blocks.is_empty())
				|| matches!(user.content, UserContentField::Text(ref text) if text.is_empty())
			{
				user.content = UserContentField::Blocks(Vec::new());
			}
			AgentMessage::Llm(Message::User(user))
		}
		AgentMessage::Llm(Message::Assistant(mut assistant)) => {
			if assistant.content.is_empty() {
				// Keep empty content: assistant content has no null shape.
			}
			AgentMessage::Llm(Message::Assistant(assistant))
		}
		AgentMessage::Llm(Message::ToolResult(mut tool_result)) => {
			if tool_result.content.is_empty() {
				tool_result.content = Vec::new();
			}
			AgentMessage::Llm(Message::ToolResult(tool_result))
		}
		other => other,
	}
}

pub(crate) fn bridge_crate_signal_to_ai_value(signal: &crate::utils::abort::AbortSignal) -> AiAbortSignal {
	let bridged = AiAbortSignal::new();
	let target = bridged.clone();
	let forward_from = signal.clone();
	tokio::spawn(async move {
		forward_from.aborted_future().await;
		AiAbortSignal::abort(&target);
	});
	bridged
}

pub(crate) fn bridge_crate_signal_to_ai(signal: Option<crate::utils::abort::AbortSignal>) -> Option<AiAbortSignal> {
	let signal = signal?;
	let bridged = AiAbortSignal::new();
	let target = bridged.clone();
	tokio::spawn(async move {
		signal.aborted_future().await;
		AiAbortSignal::abort(&target);
	});
	Some(bridged)
}

/// Persist a bash execution as an opaque custom message on the transcript
/// (pi-coding-agent D4: custom roles travel as JSON).
fn append_bash_message(inner: &AgentSessionInner, bash_message: BashExecutionMessage) {
	let wire = serde_json::to_value(&bash_message).unwrap_or(JsonValue::Null);
	let _ = inner.session_manager.lock().append_message(AgentMessage::Custom(wire));
}

fn context_edit_replacement_to_editable(
	replacement: &crate::core::session_manager::ContextEditReplacement,
) -> crate::core::session_manager::ContextEditableContent {
	match replacement {
		crate::core::session_manager::ContextEditReplacement::Content { content } => content.clone(),
		crate::core::session_manager::ContextEditReplacement::Null => {
			crate::core::session_manager::ContextEditableContent::Text(String::new())
		}
	}
}

fn parse_iso_timestamp(timestamp: &str) -> i64 {
	crate::utils::time::parse_timestamp_ms(timestamp).unwrap_or(0)
}

fn steering_mode_to_queue(mode: SteeringMode) -> QueueMode {
	match mode {
		SteeringMode::All => QueueMode::All,
		SteeringMode::OneAtATime => QueueMode::OneAtATime,
	}
}

fn queue_to_steering_mode(mode: QueueMode) -> SteeringMode {
	match mode {
		QueueMode::All => SteeringMode::All,
		QueueMode::OneAtATime => SteeringMode::OneAtATime,
	}
}

fn user_content_field_text_for_editor(content: &UserContentField) -> String {
	match content {
		UserContentField::Text(text) => text.clone(),
		UserContentField::Blocks(blocks) => blocks
			.iter()
			.filter_map(|block| match block {
				UserContent::Text(text) => Some(text.text.clone()),
				_ => None,
			})
			.collect::<Vec<_>>()
			.join("\n"),
	}
}

// ============================================================================
// Tests (ported upstream agent-session*.test.ts suites)
// ============================================================================

#[cfg(test)]
#[path = "agent_session_tests.rs"]
mod agent_session_tests;
