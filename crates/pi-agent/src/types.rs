//! Port of `packages/agent/src/types.ts` — agent-level types layered on
//! pi-ai: the [`StreamFn`] transport seam, tool/hook contracts, the
//! [`AgentMessage`] transcript union, [`AgentLoopConfig`], and
//! [`AgentEvent`].

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use pi_ai::types::{
	AssistantMessage, AssistantMessageEvent, ImageContent, JsonValue, Message, Model, SimpleStreamOptions,
	SystemMessage, Tool, ToolCall, ToolResultMessage, TranscriptContext, Usage, UserMessage,
};
use pi_ai::utils::abort::AbortSignal;
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use serde::{Deserialize, Serialize};


/// Boxed sendable future (mirrors `pi_ai::http::BoxFuture`).
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// `(event) => Promise<void> | void` sink of the low-level loop.
pub type AgentEventSink = Arc<dyn Fn(AgentEvent) -> BoxFuture<'static, ()> + Send + Sync>;

/// Agent thinking level including the `off` state (upstream
/// `ThinkingLevel`). See crate docs D6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentThinkingLevel {
	Off,
	Minimal,
	Low,
	Medium,
	High,
	#[serde(rename = "xhigh")]
	Xhigh,
	#[serde(rename = "max")]
	Max,
}

impl AgentThinkingLevel {
	/// `off` maps to `None` (upstream `reasoning === "off" ? undefined`).
	pub fn to_reasoning(self) -> Option<pi_ai::types::ThinkingLevel> {
		match self {
			AgentThinkingLevel::Off => None,
			AgentThinkingLevel::Minimal => Some(pi_ai::types::ThinkingLevel::Minimal),
			AgentThinkingLevel::Low => Some(pi_ai::types::ThinkingLevel::Low),
			AgentThinkingLevel::Medium => Some(pi_ai::types::ThinkingLevel::Medium),
			AgentThinkingLevel::High => Some(pi_ai::types::ThinkingLevel::High),
			AgentThinkingLevel::Xhigh => Some(pi_ai::types::ThinkingLevel::Xhigh),
			AgentThinkingLevel::Max => Some(pi_ai::types::ThinkingLevel::Max),
		}
	}

	pub fn from_reasoning(reasoning: Option<pi_ai::types::ThinkingLevel>) -> AgentThinkingLevel {
		use pi_ai::types::ThinkingLevel as T;
		match reasoning {
			None => AgentThinkingLevel::Off,
			Some(T::Minimal) => AgentThinkingLevel::Minimal,
			Some(T::Low) => AgentThinkingLevel::Low,
			Some(T::Medium) => AgentThinkingLevel::Medium,
			Some(T::High) => AgentThinkingLevel::High,
			Some(T::Xhigh) => AgentThinkingLevel::Xhigh,
			Some(T::Max) => AgentThinkingLevel::Max,
		}
	}
}

/// Stream function used by the agent loop. `streamSimple` satisfies this
/// shape.
///
/// The loop passes a normalized transcript: the system prompt and tool
/// declarations are carried by the transcript's system messages, never by
/// top-level prompt or tool fields.
///
/// Contract:
/// - Must not return `Err` for request/model/runtime failures (upstream:
///   must not throw). The upstream synchronous-throw path maps to `Err`
///   (crate docs D2).
/// - Must resolve to an `AssistantMessageEventStream`.
/// - Failures must be encoded in the returned stream via protocol events and
///   a final `AssistantMessage` with stopReason "error" or "aborted" and
///   errorMessage.
pub type StreamFn = Arc<
	dyn Fn(Model, TranscriptContext, Option<SimpleStreamOptions>) -> BoxFuture<'static, Result<AssistantMessageEventStream, String>>
		+ Send
		+ Sync,
>;

/// Configuration for how tool calls from a single assistant message are
/// executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolExecutionMode {
	Sequential,
	Parallel,
}

/// Controls how many queued user messages are injected at a drain point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum QueueMode {
	All,
	OneAtATime,
}

/// A single tool call content block emitted by an assistant message
/// (upstream `AgentToolCall`).
pub type AgentToolCall = ToolCall;

/// Result returned from `beforeToolCall`.
///
/// `block: true` prevents the tool from executing; the loop emits an error
/// tool result instead. `reason` becomes the text shown in that error result
/// (a default blocked message when omitted).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BeforeToolCallResult {
	pub block: Option<bool>,
	pub reason: Option<String>,
	/// Hint that the agent should stop after the current tool batch when this
	/// call is blocked. Early termination only happens when every finalized
	/// tool result in the batch sets this to true.
	pub terminate: Option<bool>,
	/// Replacement validated arguments for execution (crate docs D8: upstream
	/// hooks mutate the shared `args` object in place; Rust carries the
	/// mutation back through this field instead).
	pub args: Option<JsonValue>,
}

/// Partial override returned from `afterToolCall`. Merge semantics are
/// field-by-field (see upstream doc comment on `AfterToolCallResult`); there
/// is no deep merge for `content`, `details`, or `usage`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AfterToolCallResult {
	pub content: Option<Vec<pi_ai::types::UserContent>>,
	pub details: Option<JsonValue>,
	pub structured_content: Option<JsonValue>,
	pub is_error: Option<bool>,
	/// Usage from the final tool execution itself, if available. Not used for
	/// main LLM context accounting.
	pub usage: Option<Usage>,
	/// Hint that the agent should stop after the current tool batch.
	pub terminate: Option<bool>,
}

/// Context passed to `beforeToolCall`.
#[derive(Debug, Clone)]
pub struct BeforeToolCallContext {
	/// The assistant message that requested the tool call.
	pub assistant_message: AssistantMessage,
	/// The raw tool call block from `assistantMessage.content`.
	pub tool_call: AgentToolCall,
	/// Validated tool arguments for the target tool schema.
	pub args: JsonValue,
	/// Current agent context at the time the tool call is prepared.
	pub context: AgentContext,
}

/// Context passed to `afterToolCall`.
#[derive(Debug, Clone)]
pub struct AfterToolCallContext {
	pub assistant_message: AssistantMessage,
	pub tool_call: AgentToolCall,
	pub args: JsonValue,
	/// The executed tool result before any `afterToolCall` overrides.
	pub result: AgentToolResult,
	/// Whether the executed tool result is currently treated as an error.
	pub is_error: bool,
	pub context: AgentContext,
}

/// Context passed to completed-turn callbacks.
#[derive(Debug, Clone)]
pub struct AgentTurnContext {
	/// The assistant message that completed the turn.
	pub message: AssistantMessage,
	/// Tool result messages emitted for the completed turn.
	pub tool_results: Vec<ToolResultMessage>,
	/// Current agent context after the turn's messages have been appended.
	pub context: AgentContext,
	/// Messages this loop invocation returns if it exits at this point.
	pub new_messages: Vec<AgentMessage>,
}

/// Decision returned by `FinishTurn`. `None` preserves normal scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentTurnDecision {
	Continue,
	End,
}

/// Called after a completed assistant turn and all of its tool-result
/// messages, but before `turn_end`.
pub type FinishTurn = Arc<
	dyn Fn(AgentTurnContext, Option<AbortSignal>) -> BoxFuture<'static, Option<AgentTurnDecision>> + Send + Sync,
>;

/// Replacement runtime state used by the agent loop before starting another
/// provider request (`AgentLoopTurnUpdate`).
#[derive(Debug, Clone, Default)]
pub struct AgentLoopTurnUpdate {
	pub context: Option<AgentContext>,
	/// Messages to append before the next provider request, with normal
	/// lifecycle events.
	pub messages: Vec<AgentMessage>,
	pub model: Option<Model>,
	/// `None` keeps the current level; `Some(None)` is the `off` request.
	pub thinking_level: Option<AgentThinkingLevel>,
}

/// Runtime state available immediately before a conversational provider
/// request (`PrepareRequestContext`).
#[derive(Debug, Clone)]
pub struct PrepareRequestContext {
	pub context: AgentContext,
	pub model: Model,
	pub thinking_level: AgentThinkingLevel,
}

/// Replacement runtime state for the provider request being prepared
/// (`AgentRequestUpdate`).
#[derive(Debug, Clone, Default)]
pub struct AgentRequestUpdate {
	pub context: Option<AgentContext>,
	pub model: Option<Model>,
	pub thinking_level: Option<AgentThinkingLevel>,
}

/// Called immediately before every conversational provider request,
/// including the first.
pub type PrepareRequestHook = Arc<
	dyn Fn(PrepareRequestContext, Option<AbortSignal>) -> BoxFuture<'static, Option<AgentRequestUpdate>>
		+ Send
		+ Sync,
>;

pub type PrepareNextTurnContext = AgentTurnContext;

/// Called after `turn_end` when the loop will continue, immediately before
/// the next turn starts.
pub type PrepareNextTurnHook =
	Arc<dyn Fn(PrepareNextTurnContext) -> BoxFuture<'static, Option<AgentLoopTurnUpdate>> + Send + Sync>;

/// Legacy signal-only `prepareNextTurn` (upstream `prepareNextTurn(signal)`).
pub type PrepareNextTurnWithContextHook = Arc<
	dyn Fn(PrepareNextTurnContext, Option<AbortSignal>) -> BoxFuture<'static, Option<AgentLoopTurnUpdate>>
		+ Send
		+ Sync,
>;

/// Returns steering messages to inject into the conversation mid-run.
pub type GetQueueMessages = Arc<dyn Fn() -> BoxFuture<'static, Vec<AgentMessage>> + Send + Sync>;

/// Converts `AgentMessage`s to LLM-compatible `Message`s before each LLM
/// call. Contract: must not return `Err` (upstream: must not throw).
pub type ConvertToLlm = Arc<dyn Fn(Vec<AgentMessage>) -> BoxFuture<'static, Vec<Message>> + Send + Sync>;

/// Optional transform applied to the context before `convertToLlm`.
pub type TransformContext =
	Arc<dyn Fn(Vec<AgentMessage>, Option<AbortSignal>) -> BoxFuture<'static, Vec<AgentMessage>> + Send + Sync>;

/// Resolves an API key dynamically for each LLM call.
pub type GetApiKey = Arc<dyn Fn(&str) -> BoxFuture<'static, Option<String>> + Send + Sync>;

/// Called before a tool is executed, after arguments have been validated.
/// An `Err` (upstream: throw) becomes an error tool result.
pub type BeforeToolCallHook = Arc<
	dyn Fn(BeforeToolCallContext, Option<AbortSignal>) -> BoxFuture<'static, Result<Option<BeforeToolCallResult>, String>>
		+ Send
		+ Sync,
>;

/// Called after a tool finishes executing, before `tool_execution_end` and
/// tool-result message events are emitted. An `Err` (upstream: throw)
/// replaces the result with an error tool result.
pub type AfterToolCallHook = Arc<
	dyn Fn(AfterToolCallContext, Option<AbortSignal>) -> BoxFuture<'static, Result<Option<AfterToolCallResult>, String>>
		+ Send
		+ Sync,
>;

/// Loop-level error (upstream thrown `Error`s crossing the async boundary).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct AgentError(pub String);

/// One message in the agent transcript (`AgentMessage`): an LLM message or a
/// custom app message (upstream declaration merging → opaque JSON value,
/// typically an object carrying its own `role`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AgentMessage {
	Llm(Message),
	Custom(JsonValue),
}

impl AgentMessage {
	/// `AgentMessage::Llm(Message::System(...))`.
	pub fn system(message: SystemMessage) -> AgentMessage {
		AgentMessage::Llm(Message::System(message))
	}

	/// `AgentMessage::Llm(Message::User(...))`.
	pub fn user(message: UserMessage) -> AgentMessage {
		AgentMessage::Llm(Message::User(message))
	}

	/// `AgentMessage::Llm(Message::Assistant(...))`.
	pub fn assistant(message: AssistantMessage) -> AgentMessage {
		AgentMessage::Llm(Message::Assistant(message))
	}

	/// `AgentMessage::Llm(Message::ToolResult(...))`.
	pub fn tool_result(message: ToolResultMessage) -> AgentMessage {
		AgentMessage::Llm(Message::ToolResult(message))
	}

	/// Upstream `message.role` as a plain string (custom messages read their
	/// JSON `role` field, `"custom"` when absent).
	pub fn role(&self) -> &str {
		match self {
			AgentMessage::Llm(message) => message.role(),
			AgentMessage::Custom(value) => value.get("role").and_then(|role| role.as_str()).unwrap_or("custom"),
		}
	}

	/// The LLM message underneath, if any (custom messages return `None`).
	pub fn as_llm(&self) -> Option<&Message> {
		match self {
			AgentMessage::Llm(message) => Some(message),
			AgentMessage::Custom(_) => None,
		}
	}

	/// Upstream timestamp access: LLM messages carry their own; custom
	/// messages read a numeric `timestamp` field (`0` when absent).
	pub fn timestamp(&self) -> i64 {
		match self {
			AgentMessage::Llm(message) => match message {
				Message::System(system) => system.timestamp,
				Message::User(user) => user.timestamp,
				Message::Assistant(assistant) => assistant.timestamp,
				Message::ToolResult(tool_result) => tool_result.timestamp,
			},
			AgentMessage::Custom(value) => value.get("timestamp").and_then(|value| value.as_i64()).unwrap_or(0),
		}
	}
}

impl From<Message> for AgentMessage {
	fn from(value: Message) -> Self {
		AgentMessage::Llm(value)
	}
}

impl From<SystemMessage> for AgentMessage {
	fn from(value: SystemMessage) -> Self {
		AgentMessage::Llm(Message::System(value))
	}
}

impl From<UserMessage> for AgentMessage {
	fn from(value: UserMessage) -> Self {
		AgentMessage::Llm(Message::User(value))
	}
}

impl From<AssistantMessage> for AgentMessage {
	fn from(value: AssistantMessage) -> Self {
		AgentMessage::Llm(Message::Assistant(value))
	}
}

impl From<ToolResultMessage> for AgentMessage {
	fn from(value: ToolResultMessage) -> Self {
		AgentMessage::Llm(Message::ToolResult(value))
	}
}

/// Final or partial result produced by a tool (`AgentToolResult`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgentToolResult {
	/// Text or image content returned to the model.
	pub content: Vec<pi_ai::types::UserContent>,
	/// Arbitrary structured details for logs or UI rendering.
	#[serde(skip_serializing_if = "JsonValue::is_null")]
	pub details: JsonValue,
	/// Machine-readable result matching the tool's `outputSchema`, for
	/// programmatic callers. Not sent to the model.
	#[serde(rename = "structuredContent", skip_serializing_if = "Option::is_none")]
	pub structured_content: Option<JsonValue>,
	/// Usage from the final tool execution itself, if available.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub usage: Option<Usage>,
	/// Report a failure without throwing.
	#[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
	pub is_error: Option<bool>,
	/// Hint that the agent should stop after the current tool batch.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub terminate: Option<bool>,
}

impl AgentToolResult {
	pub fn new(content: Vec<pi_ai::types::UserContent>, details: JsonValue) -> AgentToolResult {
		AgentToolResult {
			content,
			details,
			structured_content: None,
			usage: None,
			is_error: None,
			terminate: None,
		}
	}
}

/// Final outcome of a tool call after hooks ran (`AgentToolCallOutcome`).
#[derive(Debug, Clone)]
pub struct AgentToolCallOutcome {
	pub tool_call: AgentToolCall,
	pub result: AgentToolResult,
	pub is_error: bool,
}

/// Callback used by tools to stream partial execution updates
/// (`AgentToolUpdateCallback`). Calls made after the tool settles are
/// ignored by the loop.
pub type AgentToolUpdateCallback = Arc<dyn Fn(AgentToolResult) + Send + Sync>;

/// Tool execution entry point (`AgentTool.execute`). Fails with the
/// upstream-throw path (`Err(message)` → error tool result).
pub type AgentToolExecute = Arc<
	dyn Fn(
			String,
			serde_json::Map<String, JsonValue>,
			Option<AbortSignal>,
			Option<AgentToolUpdateCallback>,
		) -> BoxFuture<'static, Result<AgentToolResult, String>>
		+ Send
		+ Sync,
>;

/// Recovery policy for an effect whose durable intent exists but whose
/// outcome is unknown (`AgentTool.replay`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolReplayPolicy {
	Never,
	Safe,
}

/// Tool definition used by the agent runtime (`AgentTool`).
#[derive(Clone)]
pub struct AgentTool {
	/// Provider declaration (name, description, JSON Schema parameters).
	pub tool: Tool,
	/// Human-readable label for UI display.
	pub label: String,
	/// Optional compatibility shim for raw tool-call arguments before schema
	/// validation. Must return an object matching `parameters`.
	pub prepare_arguments: Option<Arc<dyn Fn(JsonValue) -> JsonValue + Send + Sync>>,
	/// JSON Schema of `structuredContent` in successful results.
	pub output_schema: Option<JsonValue>,
	/// Execute the tool call.
	pub execute: AgentToolExecute,
	/// Recovery policy (upstream `replay`).
	pub replay: Option<ToolReplayPolicy>,
	/// Per-tool execution mode override.
	pub execution_mode: Option<ToolExecutionMode>,
}

impl std::fmt::Debug for AgentTool {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("AgentTool")
			.field("name", &self.tool.name)
			.field("label", &self.label)
			.field("execution_mode", &self.execution_mode)
			.finish_non_exhaustive()
	}
}

impl AgentTool {
	/// Convenience constructor: label defaults to the tool name (upstream
	/// tests spell out every field; production callers rarely do).
	pub fn new(
		name: impl Into<String>,
		description: impl Into<String>,
		parameters: JsonValue,
		label: impl Into<String>,
		execute: AgentToolExecute,
	) -> AgentTool {
		AgentTool {
			tool: Tool {
				name: name.into(),
				description: description.into(),
				parameters,
				constrained_sampling: None,
			},
			label: label.into(),
			prepare_arguments: None,
			output_schema: None,
			execute,
			replay: None,
			execution_mode: None,
		}
	}

	/// The provider-facing name (`tool.name`).
	pub fn name(&self) -> &str {
		&self.tool.name
	}
}

/// Context snapshot passed into the low-level agent loop (`AgentContext`).
#[derive(Debug, Clone, Default)]
pub struct AgentContext {
	/// Transcript visible to the model.
	pub messages: Vec<AgentMessage>,
	/// Tools available for execution in this run.
	pub tools: Option<Vec<AgentTool>>,
}

/// Events emitted by the Agent for UI updates (`AgentEvent`).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
	#[serde(rename = "agent_start")]
	AgentStart,
	#[serde(rename = "agent_end")]
	AgentEnd { messages: Vec<AgentMessage> },
	/// A turn is one assistant response + any tool calls/results.
	#[serde(rename = "turn_start")]
	TurnStart,
	#[serde(rename = "turn_end")]
	TurnEnd { message: AgentMessage, #[serde(rename = "toolResults")] tool_results: Vec<ToolResultMessage> },
	/// Emitted for system, user, assistant, and toolResult messages.
	#[serde(rename = "message_start")]
	MessageStart { message: AgentMessage },
	/// Only emitted for assistant messages during streaming.
	#[serde(rename = "message_update")]
	MessageUpdate { message: AgentMessage, #[serde(rename = "assistantMessageEvent")] assistant_message_event: AssistantMessageEvent },
	#[serde(rename = "message_end")]
	MessageEnd { message: AgentMessage },
	#[serde(rename = "tool_execution_start")]
	ToolExecutionStart {
		#[serde(rename = "toolCallId")]
		tool_call_id: String,
		#[serde(rename = "toolName")]
		tool_name: String,
		args: JsonValue,
	},
	#[serde(rename = "tool_execution_update")]
	ToolExecutionUpdate {
		#[serde(rename = "toolCallId")]
		tool_call_id: String,
		#[serde(rename = "toolName")]
		tool_name: String,
		args: JsonValue,
		#[serde(rename = "partialResult")]
		partial_result: AgentToolResult,
	},
	#[serde(rename = "tool_execution_end")]
	ToolExecutionEnd {
		#[serde(rename = "toolCallId")]
		tool_call_id: String,
		#[serde(rename = "toolName")]
		tool_name: String,
		result: AgentToolResult,
		#[serde(rename = "isError")]
		is_error: bool,
	},
}

impl AgentEvent {
	/// Upstream `event.type`.
	pub fn event_type(&self) -> &'static str {
		match self {
			AgentEvent::AgentStart => "agent_start",
			AgentEvent::AgentEnd { .. } => "agent_end",
			AgentEvent::TurnStart => "turn_start",
			AgentEvent::TurnEnd { .. } => "turn_end",
			AgentEvent::MessageStart { .. } => "message_start",
			AgentEvent::MessageUpdate { .. } => "message_update",
			AgentEvent::MessageEnd { .. } => "message_end",
			AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
			AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
			AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
		}
	}
}

/// Agent-loop public configuration (`AgentLoopConfig`). Upstream this
/// extends `SimpleStreamOptions`; Rust carries the options bag as a field.
#[derive(Clone)]
pub struct AgentLoopConfig {
	pub model: Model,
	/// Upstream inherited `SimpleStreamOptions` fields (`reasoning`,
	/// `apiKey`, `sessionId`, `transport`, `thinkingBudgets`,
	/// `maxRetryDelayMs`, `onPayload`, …) flow to the stream fn unchanged.
	pub simple: SimpleStreamOptions,
	pub convert_to_llm: ConvertToLlm,
	pub transform_context: Option<TransformContext>,
	pub get_api_key: Option<GetApiKey>,
	pub finish_turn: Option<FinishTurn>,
	pub prepare_request: Option<PrepareRequestHook>,
	pub prepare_next_turn: Option<PrepareNextTurnHook>,
	pub get_steering_messages: Option<GetQueueMessages>,
	pub get_follow_up_messages: Option<GetQueueMessages>,
	/// Tool execution mode. Default "parallel".
	pub tool_execution: Option<ToolExecutionMode>,
	pub before_tool_call: Option<BeforeToolCallHook>,
	pub after_tool_call: Option<AfterToolCallHook>,
}

impl std::fmt::Debug for AgentLoopConfig {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("AgentLoopConfig")
			.field("model", &self.model.id())
			.field("tool_execution", &self.tool_execution)
			.finish_non_exhaustive()
	}
}

impl Default for AgentLoopConfig {
	fn default() -> AgentLoopConfig {
		AgentLoopConfig {
			model: default_model(),
			simple: SimpleStreamOptions::default(),
			convert_to_llm: Arc::new(|messages| Box::pin(async move { default_convert_to_llm(&messages) })),
			transform_context: None,
			get_api_key: None,
			finish_turn: None,
			prepare_request: None,
			prepare_next_turn: None,
			get_steering_messages: None,
			get_follow_up_messages: None,
			tool_execution: None,
			before_tool_call: None,
			after_tool_call: None,
		}
	}
}

/// Upstream `defaultConvertToLlm`: pass through standard LLM messages.
pub fn default_convert_to_llm(messages: &[AgentMessage]) -> Vec<Message> {
	messages.iter().filter_map(|message| message.as_llm().cloned()).collect()
}

/// Convenience closure constructors (upstream object-literal hooks).
pub fn convert_to_llm(
	f: impl Fn(Vec<AgentMessage>) -> BoxFuture<'static, Vec<Message>> + Send + Sync + 'static,
) -> ConvertToLlm {
	Arc::new(f)
}
pub fn transform_context(
	f: impl Fn(Vec<AgentMessage>, Option<AbortSignal>) -> BoxFuture<'static, Vec<AgentMessage>> + Send + Sync + 'static,
) -> TransformContext {
	Arc::new(f)
}
pub fn get_api_key(
	f: impl Fn(&str) -> BoxFuture<'static, Option<String>> + Send + Sync + 'static,
) -> GetApiKey {
	Arc::new(f)
}
pub fn prepare_request(
	f: impl Fn(PrepareRequestContext, Option<AbortSignal>) -> BoxFuture<'static, Option<AgentRequestUpdate>>
		+ Send
		+ Sync
		+ 'static,
) -> PrepareRequestHook {
	Arc::new(f)
}
pub fn prepare_next_turn(
	f: impl Fn(PrepareNextTurnContext) -> BoxFuture<'static, Option<AgentLoopTurnUpdate>> + Send + Sync + 'static,
) -> PrepareNextTurnHook {
	Arc::new(f)
}
pub fn prepare_next_turn_with_context(
	f: impl Fn(PrepareNextTurnContext, Option<AbortSignal>) -> BoxFuture<'static, Option<AgentLoopTurnUpdate>>
		+ Send
		+ Sync
		+ 'static,
) -> PrepareNextTurnWithContextHook {
	Arc::new(f)
}
pub fn get_steering_messages(
	f: impl Fn() -> BoxFuture<'static, Vec<AgentMessage>> + Send + Sync + 'static,
) -> GetQueueMessages {
	Arc::new(f)
}
pub fn get_follow_up_messages(
	f: impl Fn() -> BoxFuture<'static, Vec<AgentMessage>> + Send + Sync + 'static,
) -> GetQueueMessages {
	Arc::new(f)
}
pub fn finish_turn(
	f: impl Fn(AgentTurnContext, Option<AbortSignal>) -> BoxFuture<'static, Option<AgentTurnDecision>> + Send + Sync + 'static,
) -> FinishTurn {
	Arc::new(f)
}
pub fn before_tool_call(
	f: impl Fn(BeforeToolCallContext, Option<AbortSignal>) -> BoxFuture<'static, Result<Option<BeforeToolCallResult>, String>>
		+ Send
		+ Sync
		+ 'static,
) -> BeforeToolCallHook {
	Arc::new(f)
}
pub fn after_tool_call(
	f: impl Fn(AfterToolCallContext, Option<AbortSignal>) -> BoxFuture<'static, Result<Option<AfterToolCallResult>, String>>
		+ Send
		+ Sync
		+ 'static,
) -> AfterToolCallHook {
	Arc::new(f)
}
pub fn before_tool_call_context(assistant_message: AssistantMessage, tool_call: AgentToolCall, args: JsonValue, context: AgentContext) -> BeforeToolCallContext {
	BeforeToolCallContext { assistant_message, tool_call, args, context }
}
pub fn after_tool_call_context(assistant_message: AssistantMessage, tool_call: AgentToolCall, args: JsonValue, result: AgentToolResult, is_error: bool, context: AgentContext) -> AfterToolCallContext {
	AfterToolCallContext { assistant_message, tool_call, args, result, is_error, context }
}

/// Upstream `DEFAULT_MODEL` (`agent.ts`): placeholder model for agents
/// created without one.
pub fn default_model() -> Model {
	Model {
		base: pi_ai::types::BaseModelFields {
			id: "unknown".into(),
			name: "unknown".into(),
			api: "unknown".into(),
			provider: "unknown".into(),
			base_url: String::new(),
			input: Vec::new(),
			input_limits: None,
			cost: pi_ai::types::ModelCost {
				rates: pi_ai::types::ModelCostRates { input: 0.0, output: 0.0, cache_read: 0.0, cache_write: 0.0 },
				tiers: None,
			},
			headers: None,
		},
		model_type: None,
		reasoning: false,
		thinking_level_map: None,
		prompt_cache: None,
		context_window: 0,
		max_tokens: 0,
		sampling_params: None,
		sampling_params_by_thinking_level: None,
		compat: None,
	}
}

/// Upstream `EMPTY_USAGE`.
pub fn empty_usage() -> Usage {
	Usage::default()
}

/// `Date.now()` equivalent: Unix milliseconds.
pub fn now_ms() -> i64 {
	std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map(|duration| duration.as_millis() as i64)
		.unwrap_or(0)
}

/// Prompt input accepted by [`crate::agent::Agent::prompt`] (upstream
/// overloads `prompt(string, images?)` / `prompt(message)` /
/// `prompt(messages[])`).
#[derive(Debug, Clone)]
pub enum PromptInput {
	Text { text: String, images: Vec<ImageContent> },
	Messages(Vec<AgentMessage>),
}

impl From<&str> for PromptInput {
	fn from(value: &str) -> Self {
		PromptInput::Text { text: value.to_string(), images: Vec::new() }
	}
}

impl From<String> for PromptInput {
	fn from(value: String) -> Self {
		PromptInput::Text { text: value, images: Vec::new() }
	}
}

impl From<Vec<AgentMessage>> for PromptInput {
	fn from(value: Vec<AgentMessage>) -> Self {
		PromptInput::Messages(value)
	}
}

impl From<AgentMessage> for PromptInput {
	fn from(value: AgentMessage) -> Self {
		PromptInput::Messages(vec![value])
	}
}

impl From<Message> for PromptInput {
	fn from(value: Message) -> Self {
		PromptInput::Messages(vec![AgentMessage::Llm(value)])
	}
}

