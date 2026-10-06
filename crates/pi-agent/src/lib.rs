//! Port of @earendil-works/pi-agent (upstream: packages/agent) — the agent
//! loop: multi-turn orchestration on top of pi-ai, tool execution with
//! before/after hooks, steering/follow-up queues, and the stateful [`Agent`]
//! wrapper. Public re-exports mirror upstream `src/index.ts`.
//!
//! # Module map (upstream `packages/agent/src` → this crate)
//!
//! `types.ts` → [`types`]; `agent-loop.ts` → [`agent_loop`]; `agent.ts` →
//! [`agent`]; `proxy.ts` → [`proxy`]; `stream-fn.ts` → [`stream_fn`].
//!
//! # Reading this crate
//!
//! * [`types::StreamFn`] is the transport seam (upstream stream-fn/type):
//!   `Models.streamSimple` satisfies it. Failures are encoded in the
//!   returned [`AssistantMessageEventStream`], never by `Err`, except the
//!   upstream-synchronous-throw path (see D2).
//! * [`agent_loop::agent_loop`] returns the same push-based
//!   `EventStream<AgentEvent, Vec<AgentMessage>>` shape upstream does; it is
//!   drained as a `futures_core::Stream` and resolved with `.result()`.
//! * [`agent::Agent`] keeps upstream semantics: listeners are awaited in
//!   subscription order and settle before `prompt`/`continue_` resolve;
//!   `agent_end` is the final event of a run.
//!
//! # Deviations from upstream
//!
//! - **D1 — JS property accessors become methods/guards.** `AgentState`
//!   accessor properties (`state.messages`, `state.tools`, …) become
//!   [`agent::Agent::lock_state`] plus accessor methods; assignment-copy
//!   semantics are irrelevant under Rust ownership (the stored vector is
//!   always owned by the state).
//! - **D2 — synchronous throws become `Result`/`Err(String)`.** Upstream
//!   `StreamFn` "may throw", and `Agent.prompt`/`continue`/`reset` throw for
//!   misuse. Rust expresses these as `Err`; messages are byte-identical.
//!   `stream_fn::get_default_stream_fn` panics with the upstream error
//!   message when unset (it would be an unreachable misconfiguration
//!   upstream, thrown at the same point).
//! - **D3 — `queueMicrotask` producers push eagerly.** Mock/fake stream fns
//!   in tests push events synchronously before returning; `EventStream`
//!   buffers, so consumer-visible order and result resolution are identical.
//! - **D4 — `partialJson` is loop-local.** Proxy tool-call deltas carry a
//!   side map `contentIndex → partial json` instead of an extra field on
//!   [`pi_ai::types::ToolCall`]; reconstructed `arguments` are identical.
//! - **D5 — `fetch` seam.** `streamProxy` takes `options.fetch`
//!   ([`pi_ai::http::FetchFn`], reqwest default) instead of `globalThis.fetch`;
//!   tests inject recording fakes. Non-OK status text uses `""` for
//!   `response.statusText` (Rust has no reason phrases).
//! - **D6 — agent `ThinkingLevel` includes `off`.** Upstream
//!   `ThinkingLevel = "off" | …`; pi-ai's `ThinkingLevel` has no `off`.
//!   [`types::AgentThinkingLevel`] carries it and maps to
//!   `Option<pi_ai ThinkingLevel>` (`off` → `None`, exactly the loop's
//!   `reasoning ?? "off"` round-trip).
//! - **D7 — parallel tool execution uses `join_all` on the loop task**
//!   instead of independent promises; interleaving and completion-order
//!   event emission are preserved by cooperative polling.
//! - **D8 — `beforeToolCall` argument mutation is an explicit override.**
//!   Upstream hooks mutate the shared `args` object in place; Rust passes
//!   arguments by value, so [`types::BeforeToolCallResult`] carries an
//!   optional `args` replacement applied without revalidation (same
//!   observable behavior as the upstream "execute mutated beforeToolCall
//!   args" test).
//!
//! `e2e.test.ts` upstream only exercises the faux provider (offline); no
//! live-endpoint cases exist to defer.

pub mod agent;
pub mod agent_loop;
pub mod proxy;
pub mod stream_fn;
pub mod types;

pub use agent::{Agent, AgentInitialState, AgentOptions, MutableAgentState};
pub use agent_loop::{
	agent_loop, agent_loop_continue, run_agent_loop, run_agent_loop_continue, run_tool_call, AgentEventStream,
	RunToolCallOptions, ToolUpdateSink,
};
pub use proxy::{stream_proxy, ProxyAssistantMessageEvent, ProxyStreamOptions};
pub use stream_fn::{get_default_stream_fn, set_default_stream_fn};
pub use types::{
	after_tool_call, after_tool_call_context, before_tool_call, before_tool_call_context, convert_to_llm,
	default_convert_to_llm, default_model, empty_usage, finish_turn, get_api_key, get_follow_up_messages,
	get_steering_messages, now_ms, prepare_next_turn, prepare_next_turn_with_context, prepare_request,
	transform_context, AfterToolCallContext, AfterToolCallResult, AgentContext, AgentError, AgentEvent, AgentEventSink,
	AgentLoopConfig, AgentLoopTurnUpdate, AgentMessage, AgentRequestUpdate, AgentThinkingLevel, AgentTool,
	AgentToolCall, AgentToolCallOutcome, AgentToolExecute, AgentToolResult, AgentToolUpdateCallback, AgentTurnContext,
	AgentTurnDecision, BeforeToolCallContext, BeforeToolCallResult, BoxFuture, ConvertToLlm, FinishTurn, GetApiKey,
	GetQueueMessages, PrepareNextTurnContext, PrepareNextTurnHook, PrepareNextTurnWithContextHook,
	PrepareRequestContext, PrepareRequestHook, PromptInput, QueueMode, StreamFn, ToolExecutionMode, ToolReplayPolicy,
	TransformContext,
};
