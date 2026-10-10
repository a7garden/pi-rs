//! Port of `packages/agent/src/agent.ts` — the stateful [`Agent`] wrapper
//! around the low-level loop: owns the transcript, emits lifecycle events,
//! executes tools via the loop, and exposes queueing APIs for steering and
//! follow-up messages.

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::Mutex;
use pi_ai::types::{
    AssistantMessage, Message, Model, SimpleStreamOptions, StopReason, ToolResultMessage, Usage,
};
use pi_ai::utils::abort::{AbortController, AbortSignal};
use pi_ai::utils::transcript::{
    create_initial_system_message, get_current_system_message, get_current_system_prompt,
    to_tool_declaration,
};

use crate::agent_loop::{run_agent_loop, run_agent_loop_continue};
use crate::stream_fn::get_default_stream_fn;
use crate::types::{
    default_convert_to_llm, now_ms, prepare_next_turn, AgentContext, AgentError, AgentEvent,
    AgentEventSink, AgentLoopConfig, AgentMessage, AgentThinkingLevel, AgentTool, BoxFuture,
    PromptInput, QueueMode, ToolExecutionMode,
};

/// Initial state for [`Agent`]. `system_prompt` and `tools` become the
/// leading system message unless `messages` already starts with one.
#[derive(Default)]
pub struct AgentInitialState {
    pub system_prompt: Option<String>,
    pub model: Option<Model>,
    pub thinking_level: Option<AgentThinkingLevel>,
    pub tools: Vec<AgentTool>,
    pub messages: Vec<AgentMessage>,
}

/// Options for constructing an [`Agent`] (upstream `AgentOptions`).
pub struct AgentOptions {
    pub stream_fn: Option<crate::types::StreamFn>,
    pub initial_state: AgentInitialState,
    pub convert_to_llm: Option<crate::types::ConvertToLlm>,
    pub transform_context: Option<crate::types::TransformContext>,
    pub get_api_key: Option<crate::types::GetApiKey>,
    pub on_payload: Option<pi_ai::types::PayloadHook>,
    pub on_response: Option<pi_ai::types::ResponseHook>,
    pub on_provider_stream_event: Option<pi_ai::types::ProviderStreamEventHook>,
    pub before_tool_call: Option<crate::types::BeforeToolCallHook>,
    pub after_tool_call: Option<crate::types::AfterToolCallHook>,
    pub finish_turn: Option<crate::types::FinishTurn>,
    pub prepare_request: Option<crate::types::PrepareRequestHook>,
    pub prepare_next_turn: Option<crate::types::PrepareNextTurnWithContextHook>,
    pub prepare_next_turn_with_context: Option<crate::types::PrepareNextTurnWithContextHook>,
    pub steering_mode: Option<QueueMode>,
    pub follow_up_mode: Option<QueueMode>,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<pi_ai::types::ThinkingBudgets>,
    pub transport: Option<pi_ai::types::Transport>,
    pub max_retry_delay_ms: Option<u64>,
    pub tool_execution: Option<ToolExecutionMode>,
}

impl Default for AgentOptions {
    fn default() -> AgentOptions {
        AgentOptions {
            stream_fn: None,
            initial_state: AgentInitialState::default(),
            convert_to_llm: None,
            transform_context: None,
            get_api_key: None,
            on_payload: None,
            on_response: None,
            on_provider_stream_event: None,
            before_tool_call: None,
            after_tool_call: None,
            finish_turn: None,
            prepare_request: None,
            prepare_next_turn: None,
            prepare_next_turn_with_context: None,
            steering_mode: None,
            follow_up_mode: None,
            session_id: None,
            thinking_budgets: None,
            transport: None,
            max_retry_delay_ms: None,
            tool_execution: None,
        }
    }
}

struct PendingMessageQueue {
    messages: Vec<AgentMessage>,
    mode: QueueMode,
}

impl PendingMessageQueue {
    fn new(mode: QueueMode) -> PendingMessageQueue {
        PendingMessageQueue {
            messages: Vec::new(),
            mode,
        }
    }

    fn enqueue(&mut self, message: AgentMessage) {
        self.messages.push(message);
    }

    fn has_items(&self) -> bool {
        !self.messages.is_empty()
    }

    fn peek(&self) -> Vec<AgentMessage> {
        match self.mode {
            QueueMode::All => self.messages.clone(),
            QueueMode::OneAtATime => self.messages.first().into_iter().cloned().collect(),
        }
    }

    fn drain(&mut self) -> Vec<AgentMessage> {
        let drained = self.peek();
        self.messages.drain(..drained.len());
        drained
    }

    fn clear(&mut self) {
        self.messages.clear();
    }
}

/// Runtime-owned agent state (upstream `MutableAgentState`); see
/// [`Agent::lock_state`].
pub struct MutableAgentState {
    pub model: Model,
    pub thinking_level: AgentThinkingLevel,
    pub tools: Vec<AgentTool>,
    pub messages: Vec<AgentMessage>,
    pub is_streaming: bool,
    pub streaming_message: Option<AgentMessage>,
    pub pending_tool_calls: HashSet<String>,
    pub error_message: Option<String>,
}

impl MutableAgentState {
    /// Current system prompt, replayed from the transcript's system
    /// messages. Read-only upstream: to change the prompt, append a system
    /// message with `content` or `sections`.
    pub fn system_prompt(&self) -> String {
        let llm: Vec<Message> = self
            .messages
            .iter()
            .filter_map(|message| message.as_llm().cloned())
            .collect();
        get_current_system_prompt(&llm)
    }
}

struct ActiveRun {
    idle: Arc<tokio::sync::Notify>,
    controller: AbortController,
}

type AgentListener = Arc<dyn Fn(AgentEvent, AbortSignal) -> BoxFuture<'static, ()> + Send + Sync>;

/// Stateful wrapper around the low-level agent loop.
///
/// `Agent` owns the current transcript, emits lifecycle events, executes
/// tools, and exposes queueing APIs for steering and follow-up messages.
/// Construct behind an [`Arc`]: `subscribe` hands listeners a weak
/// reference so listener closures can call back into the agent.
pub struct Agent {
    state: Arc<Mutex<MutableAgentState>>,
    listeners: Mutex<Vec<AgentListener>>,
    steering_queue: Arc<Mutex<PendingMessageQueue>>,
    follow_up_queue: Arc<Mutex<PendingMessageQueue>>,

    pub convert_to_llm: crate::types::ConvertToLlm,
    pub transform_context: parking_lot::RwLock<Option<crate::types::TransformContext>>,
    pub stream_function: crate::types::StreamFn,
    pub get_api_key: Option<crate::types::GetApiKey>,
    pub on_payload: Option<pi_ai::types::PayloadHook>,
    pub on_response: Option<pi_ai::types::ResponseHook>,
    pub on_provider_stream_event: Option<pi_ai::types::ProviderStreamEventHook>,
    pub before_tool_call: parking_lot::RwLock<Option<crate::types::BeforeToolCallHook>>,
    pub after_tool_call: parking_lot::RwLock<Option<crate::types::AfterToolCallHook>>,
    pub finish_turn: parking_lot::RwLock<Option<crate::types::FinishTurn>>,
    pub prepare_request: parking_lot::RwLock<Option<crate::types::PrepareRequestHook>>,
    pub prepare_next_turn:
        parking_lot::RwLock<Option<crate::types::PrepareNextTurnWithContextHook>>,
    pub prepare_next_turn_with_context:
        parking_lot::RwLock<Option<crate::types::PrepareNextTurnWithContextHook>>,
    active_run: Mutex<Option<ActiveRun>>,
    session_id: Mutex<Option<String>>,
    /// Optional per-level thinking token budgets forwarded to the stream
    /// function.
    pub thinking_budgets: Option<pi_ai::types::ThinkingBudgets>,
    /// Preferred transport forwarded to the stream function.
    pub transport: pi_ai::types::Transport,
    /// Optional cap for provider-requested retry delays.
    pub max_retry_delay_ms: Option<u64>,
    /// Tool execution strategy for assistant messages that contain multiple
    /// tool calls.
    pub tool_execution: ToolExecutionMode,
}

impl Agent {
    pub fn new(options: AgentOptions) -> Agent {
        let state = Arc::new(Mutex::new(create_mutable_agent_state(
            &options.initial_state,
        )));
        let mut options = options;
        let stream_function = options
            .stream_fn
            .take()
            .unwrap_or_else(get_default_stream_fn);
        Agent {
            state,
            listeners: Mutex::new(Vec::new()),
            steering_queue: Arc::new(Mutex::new(PendingMessageQueue::new(
                options.steering_mode.unwrap_or(QueueMode::OneAtATime),
            ))),
            follow_up_queue: Arc::new(Mutex::new(PendingMessageQueue::new(
                options.follow_up_mode.unwrap_or(QueueMode::OneAtATime),
            ))),
            convert_to_llm: options.convert_to_llm.take().unwrap_or_else(|| {
                Arc::new(|messages| Box::pin(async move { default_convert_to_llm(&messages) }))
            }),
            transform_context: parking_lot::RwLock::new(options.transform_context.take()),
            stream_function,
            get_api_key: options.get_api_key.take(),
            on_payload: options.on_payload.take(),
            on_response: options.on_response.take(),
            on_provider_stream_event: options.on_provider_stream_event.take(),
            before_tool_call: parking_lot::RwLock::new(options.before_tool_call.take()),
            after_tool_call: parking_lot::RwLock::new(options.after_tool_call.take()),
            finish_turn: parking_lot::RwLock::new(options.finish_turn.take()),
            prepare_request: parking_lot::RwLock::new(options.prepare_request.take()),
            prepare_next_turn: parking_lot::RwLock::new(options.prepare_next_turn.take()),
            prepare_next_turn_with_context: parking_lot::RwLock::new(
                options.prepare_next_turn_with_context.take(),
            ),
            active_run: Mutex::new(None),
            session_id: Mutex::new(options.session_id.take()),
            thinking_budgets: options.thinking_budgets.take(),
            transport: options.transport.unwrap_or(pi_ai::types::Transport::Auto),
            max_retry_delay_ms: options.max_retry_delay_ms,
            tool_execution: options
                .tool_execution
                .unwrap_or(ToolExecutionMode::Parallel),
        }
    }

    // ==== state accessors (upstream property accessors; crate docs D1) ====

    /// Direct access to the runtime-owned state. Guards must not be held
    /// across `.await` points.
    pub fn lock_state(&self) -> parking_lot::MutexGuard<'_, MutableAgentState> {
        self.state.lock()
    }

    pub fn system_prompt(&self) -> String {
        self.state.lock().system_prompt()
    }

    pub fn model(&self) -> Model {
        self.state.lock().model.clone()
    }

    pub fn set_model(&self, model: Model) {
        self.state.lock().model = model;
    }

    pub fn thinking_level(&self) -> AgentThinkingLevel {
        self.state.lock().thinking_level
    }

    pub fn set_thinking_level(&self, level: AgentThinkingLevel) {
        self.state.lock().thinking_level = level;
    }

    pub fn tools(&self) -> Vec<AgentTool> {
        self.state.lock().tools.clone()
    }

    pub fn set_tools(&self, tools: Vec<AgentTool>) {
        self.state.lock().tools = tools;
    }

    pub fn messages(&self) -> Vec<AgentMessage> {
        self.state.lock().messages.clone()
    }

    pub fn set_messages(&self, messages: Vec<AgentMessage>) {
        self.state.lock().messages = messages;
    }

    pub fn is_streaming(&self) -> bool {
        self.state.lock().is_streaming
    }

    pub fn streaming_message(&self) -> Option<AgentMessage> {
        self.state.lock().streaming_message.clone()
    }

    pub fn pending_tool_calls(&self) -> HashSet<String> {
        self.state.lock().pending_tool_calls.clone()
    }

    pub fn error_message(&self) -> Option<String> {
        self.state.lock().error_message.clone()
    }

    /// Session identifier forwarded to providers for cache-aware backends.
    pub fn session_id(&self) -> Option<String> {
        self.session_id.lock().clone()
    }

    pub fn set_session_id(&self, session_id: Option<String>) {
        *self.session_id.lock() = session_id;
    }

    /// Subscribe to agent lifecycle events.
    ///
    /// Listener promises are awaited in subscription order and are included
    /// in the current run's settlement. Listeners also receive the active
    /// abort signal for the current run. `agent_end` is the final emitted
    /// event for a run, but the agent does not become idle until all awaited
    /// listeners for that event have settled.
    pub fn subscribe(
        self: &Arc<Self>,
        listener: impl Fn(AgentEvent, AbortSignal) -> BoxFuture<'static, ()> + Send + Sync + 'static,
    ) -> Box<dyn FnOnce() + Send> {
        let listener: AgentListener = Arc::new(listener);
        self.listeners.lock().push(listener.clone());
        let weak = Arc::downgrade(self);
        Box::new(move || {
            let Some(agent) = weak.upgrade() else { return };
            let mut listeners = agent.listeners.lock();
            if let Some(position) = listeners
                .iter()
                .position(|existing| Arc::ptr_eq(existing, &listener))
            {
                listeners.remove(position);
            }
        })
    }

    /// Controls how queued steering messages are drained.
    pub fn steering_mode(&self) -> QueueMode {
        self.steering_queue.lock().mode
    }

    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.steering_queue.lock().mode = mode;
    }

    /// Controls how queued follow-up messages are drained.
    pub fn follow_up_mode(&self) -> QueueMode {
        self.follow_up_queue.lock().mode
    }

    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.follow_up_queue.lock().mode = mode;
    }

    /// Queue a message to be injected after the current assistant turn
    /// finishes.
    pub fn steer(&self, message: impl Into<AgentMessage>) {
        self.steering_queue.lock().enqueue(message.into());
    }

    /// Queue a message to run only after the agent would otherwise stop.
    pub fn follow_up(&self, message: impl Into<AgentMessage>) {
        self.follow_up_queue.lock().enqueue(message.into());
    }

    /// Remove all queued steering messages.
    pub fn clear_steering_queue(&self) {
        self.steering_queue.lock().clear();
    }

    /// Remove all queued follow-up messages.
    pub fn clear_follow_up_queue(&self) {
        self.follow_up_queue.lock().clear();
    }

    /// Remove all queued steering and follow-up messages.
    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    /// Returns true when either queue still contains pending messages.
    pub fn has_queued_messages(&self) -> bool {
        self.steering_queue.lock().has_items() || self.follow_up_queue.lock().has_items()
    }

    /// Preview the messages selected for the next turn without consuming
    /// them.
    pub fn peek_queued_messages(&self) -> Vec<AgentMessage> {
        let steering = self.steering_queue.lock().peek();
        if !steering.is_empty() {
            steering
        } else {
            self.follow_up_queue.lock().peek()
        }
    }

    /// Active abort signal for the current run, if any.
    pub fn signal(&self) -> Option<AbortSignal> {
        self.active_run
            .lock()
            .as_ref()
            .map(|run| run.controller.signal())
    }

    /// Abort the current run, if one is active.
    pub fn abort(&self) {
        if let Some(run) = self.active_run.lock().as_ref() {
            run.controller.abort();
        }
    }

    /// Resolve when the current run and all awaited event listeners have
    /// finished. This resolves after `agent_end` listeners settle.
    pub async fn wait_for_idle(&self) {
        let notify = match self.active_run.lock().as_ref() {
            Some(run) => run.idle.clone(),
            None => return,
        };
        notify.notified().await;
    }

    /// Clear conversation state and queues while retaining the replayed
    /// prompt/tool baseline.
    pub fn reset(&self) -> Result<(), AgentError> {
        if self.active_run.lock().is_some() {
            return Err(AgentError(
                "Agent is already processing. Wait for completion before resetting.".into(),
            ));
        }

        let mut state = self.state.lock();
        let llm: Vec<Message> = state
            .messages
            .iter()
            .filter_map(|message| message.as_llm().cloned())
            .collect();
        state.messages = get_current_system_message(&llm)
            .map(|baseline| vec![AgentMessage::system(baseline)])
            .unwrap_or_default();
        state.is_streaming = false;
        state.streaming_message = None;
        state.pending_tool_calls = HashSet::new();
        state.error_message = None;
        drop(state);
        self.clear_follow_up_queue();
        self.clear_steering_queue();
        Ok(())
    }

    /// Start a new prompt from text (with optional images), a single message,
    /// or a batch of messages.
    pub async fn prompt(self: Arc<Self>, input: impl Into<PromptInput>) -> Result<(), AgentError> {
        if self.active_run.lock().is_some() {
            return Err(AgentError(
				"Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
					.into(),
			));
        }
        let messages = self.normalize_prompt_input(input.into());
        self.run_prompt_messages(messages, false).await
    }

    /// Continue from the current transcript. The last message must be a user
    /// or tool-result message.
    pub async fn continue_(self: Arc<Self>) -> Result<(), AgentError> {
        if self.active_run.lock().is_some() {
            return Err(AgentError(
                "Agent is already processing. Wait for completion before continuing.".into(),
            ));
        }

        // Block-scoped so the guard cannot live across the awaits below
        // (`drop(state)` alone does not end the generator's may-use range).
        let (last_message, all_system, has_messages) = {
            let state = self.state.lock();
            (
                state.messages.last().cloned(),
                !state.messages.is_empty()
                    && state
                        .messages
                        .iter()
                        .all(|message| message.role() == "system"),
                !state.messages.is_empty(),
            )
        };
        if !has_messages || all_system {
            return Err(AgentError("No messages to continue from".into()));
        }

        if last_message.as_ref().map(|message| message.role()) == Some("assistant") {
            let queued_steering = self.steering_queue.lock().drain();
            if !queued_steering.is_empty() {
                return self.run_prompt_messages(queued_steering, true).await;
            }

            let queued_follow_ups = self.follow_up_queue.lock().drain();
            if !queued_follow_ups.is_empty() {
                return self.run_prompt_messages(queued_follow_ups, false).await;
            }

            return Err(AgentError(
                "Cannot continue from message role: assistant".into(),
            ));
        }

        self.run_continuation().await
    }

    fn normalize_prompt_input(&self, input: PromptInput) -> Vec<AgentMessage> {
        match input {
            PromptInput::Messages(messages) => messages,
            PromptInput::Text { text, images } => {
                let mut content: Vec<pi_ai::types::UserContent> =
                    vec![pi_ai::types::UserContent::Text(
                        pi_ai::types::TextContent::new(text),
                    )];
                content.extend(images.into_iter().map(pi_ai::types::UserContent::Image));
                vec![AgentMessage::user(pi_ai::types::UserMessage {
                    content: pi_ai::types::UserContentField::Blocks(content),
                    timestamp: now_ms(),
                })]
            }
        }
    }

    async fn run_prompt_messages(
        self: &Arc<Self>,
        messages: Vec<AgentMessage>,
        skip_initial_steering_poll: bool,
    ) -> Result<(), AgentError> {
        let agent = Arc::clone(self);
        let skip = Arc::new(std::sync::atomic::AtomicBool::new(
            skip_initial_steering_poll,
        ));
        self.run_with_lifecycle(move |signal| {
            let agent = agent.clone();
            Box::pin(async move {
                let loop_config = agent.create_loop_config(skip);
                let context = agent.create_context_snapshot();
                let stream_fn = agent.stream_function.clone();
                let emit = agent.event_sink();
                run_agent_loop(
                    messages,
                    context,
                    loop_config,
                    emit,
                    Some(signal),
                    Some(stream_fn),
                )
                .await?;
                Ok(())
            })
        })
        .await
    }

    async fn run_continuation(self: &Arc<Self>) -> Result<(), AgentError> {
        let agent = Arc::clone(self);
        self.run_with_lifecycle(move |signal| {
            let agent = agent.clone();
            Box::pin(async move {
                let loop_config =
                    agent.create_loop_config(Arc::new(std::sync::atomic::AtomicBool::new(false)));
                let context = agent.create_context_snapshot();
                let stream_fn = agent.stream_function.clone();
                let emit = agent.event_sink();
                run_agent_loop_continue(context, loop_config, emit, Some(signal), Some(stream_fn))
                    .await?;
                Ok(())
            })
        })
        .await
    }

    fn create_context_snapshot(&self) -> AgentContext {
        let state = self.state.lock();
        AgentContext {
            messages: state.messages.clone(),
            tools: Some(state.tools.clone()),
        }
    }

    fn create_loop_config(
        self: &Arc<Self>,
        skip_initial_steering_poll: Arc<std::sync::atomic::AtomicBool>,
    ) -> AgentLoopConfig {
        let state = self.state.lock();
        let mut config = AgentLoopConfig {
            model: state.model.clone(),
            simple: SimpleStreamOptions {
                reasoning: state.thinking_level.to_reasoning(),
                ..SimpleStreamOptions::default()
            },
            convert_to_llm: self.convert_to_llm.clone(),
            transform_context: self.transform_context.read().clone(),
            get_api_key: self.get_api_key.clone(),
            finish_turn: self.finish_turn.read().clone(),
            prepare_request: self.prepare_request.read().clone(),
            prepare_next_turn: None,
            get_steering_messages: None,
            get_follow_up_messages: None,
            tool_execution: Some(self.tool_execution),
            before_tool_call: self.before_tool_call.read().clone(),
            after_tool_call: self.after_tool_call.read().clone(),
        };
        config.simple.base.session_id = self.session_id();
        config.simple.base.on_payload = self.on_payload.clone();
        config.simple.base.on_response = self.on_response.clone();
        config.simple.base.on_provider_stream_event = self.on_provider_stream_event.clone();
        config.simple.base.transport = Some(self.transport);
        config.simple.thinking_budgets = self.thinking_budgets.clone();
        config.simple.base.max_retry_delay_ms = self.max_retry_delay_ms;

        if self.prepare_next_turn_with_context.read().is_some()
            || self.prepare_next_turn.read().is_some()
        {
            let agent = Arc::clone(self);
            let with_context = self.prepare_next_turn_with_context.read().clone();
            let signal_only = self.prepare_next_turn.read().clone();
            config.prepare_next_turn = Some(prepare_next_turn(move |context| {
                let agent = agent.clone();
                let with_context = with_context.clone();
                let signal_only = signal_only.clone();
                Box::pin(async move {
                    if let Some(hook) = with_context {
                        return hook(context, agent.signal()).await;
                    }
                    if let Some(hook) = signal_only {
                        return hook(context, agent.signal()).await;
                    }
                    None
                })
            }));
        }

        {
            let queue = self.steering_queue.clone();
            config.get_steering_messages = Some(crate::types::get_steering_messages(move || {
                let queue = queue.clone();
                let skip = skip_initial_steering_poll.clone();
                Box::pin(async move {
                    if skip.swap(false, std::sync::atomic::Ordering::SeqCst) {
                        return Vec::new();
                    }
                    queue.lock().drain()
                })
            }));
        }
        {
            let queue = self.follow_up_queue.clone();
            config.get_follow_up_messages = Some(crate::types::get_follow_up_messages(move || {
                let queue = queue.clone();
                Box::pin(async move { queue.lock().drain() })
            }));
        }
        config
    }

    fn event_sink(self: &Arc<Self>) -> AgentEventSink {
        let agent = Arc::clone(self);
        Arc::new(move |event| {
            let agent = agent.clone();
            Box::pin(async move {
                agent.process_events(event).await;
            })
        })
    }

    async fn run_with_lifecycle(
        self: &Arc<Self>,
        executor: impl FnOnce(AbortSignal) -> BoxFuture<'static, Result<(), AgentError>>,
    ) -> Result<(), AgentError> {
        {
            let mut active = self.active_run.lock();
            if active.is_some() {
                return Err(AgentError("Agent is already processing.".into()));
            }
            *active = Some(ActiveRun {
                idle: Arc::new(tokio::sync::Notify::new()),
                controller: AbortController::new(),
            });
        }

        {
            let mut state = self.state.lock();
            state.is_streaming = true;
            state.streaming_message = None;
            state.error_message = None;
        }

        let signal = self
            .active_run
            .lock()
            .as_ref()
            .expect("just set")
            .controller
            .signal();
        let aborted = signal.aborted();
        match executor(signal).await {
            Ok(()) => {}
            // Upstream swallows the executor error after emitting the failure
            // lifecycle; prompt()/continue() resolve either way.
            Err(error) => {
                self.handle_run_failure(error, aborted).await;
            }
        }
        self.finish_run();
        Ok(())
    }

    async fn handle_run_failure(&self, error: AgentError, aborted: bool) {
        let model = self.state.lock().model.clone();
        let failure_message = AssistantMessage {
            content: vec![pi_ai::types::AssistantContent::Text(
                pi_ai::types::TextContent::new(""),
            )],
            api: model.api().to_string(),
            provider: model.provider().to_string(),
            model: model.id().to_string(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            thinking_level: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: if aborted {
                StopReason::Aborted
            } else {
                StopReason::Error
            },
            deferred: None,
            error_message: Some(error.0.clone()),
            raw_stop_reason: None,
            end_turn: None,
            timestamp: now_ms(),
            duration_ms: None,
        };
        self.process_events(AgentEvent::MessageStart {
            message: AgentMessage::assistant(failure_message.clone()),
        })
        .await;
        self.process_events(AgentEvent::MessageEnd {
            message: AgentMessage::assistant(failure_message.clone()),
        })
        .await;
        self.process_events(AgentEvent::TurnEnd {
            message: AgentMessage::assistant(failure_message.clone()),
            tool_results: Vec::<ToolResultMessage>::new(),
        })
        .await;
        self.process_events(AgentEvent::AgentEnd {
            messages: vec![AgentMessage::assistant(failure_message)],
        })
        .await;
    }

    fn finish_run(&self) {
        {
            let mut state = self.state.lock();
            state.is_streaming = false;
            state.streaming_message = None;
            state.pending_tool_calls = HashSet::new();
        }
        if let Some(run) = self.active_run.lock().take() {
            run.idle.notify_one();
        }
    }

    /// Reduce internal state for a loop event, then await listeners.
    ///
    /// `agent_end` only means no further loop events will be emitted. The
    /// run is considered idle later, after all awaited listeners for
    /// `agent_end` finish and `finish_run()` clears runtime-owned state.
    async fn process_events(&self, event: AgentEvent) {
        {
            let mut state = self.state.lock();
            match &event {
                AgentEvent::MessageStart { message }
                | AgentEvent::MessageUpdate { message, .. } => {
                    state.streaming_message = Some(message.clone());
                }
                AgentEvent::MessageEnd { message } => {
                    state.streaming_message = None;
                    state.messages.push(message.clone());
                }
                AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                    state.pending_tool_calls.insert(tool_call_id.clone());
                }
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                    state.pending_tool_calls.remove(tool_call_id);
                }
                AgentEvent::TurnEnd { message, .. } => {
                    if let AgentMessage::Llm(pi_ai::types::Message::Assistant(assistant)) = message
                    {
                        if let Some(error_message) = &assistant.error_message {
                            state.error_message = Some(error_message.clone());
                        }
                    }
                }
                AgentEvent::AgentEnd { .. } => {
                    state.streaming_message = None;
                }
                AgentEvent::AgentStart
                | AgentEvent::TurnStart
                | AgentEvent::ToolExecutionUpdate { .. } => {}
            }
        }

        let signal = self
            .active_run
            .lock()
            .as_ref()
            .map(|run| run.controller.signal())
            .expect("Agent listener invoked outside active run");
        let listeners = self.listeners.lock().clone();
        for listener in listeners {
            listener(event.clone(), signal.clone()).await;
        }
    }
}

fn create_mutable_agent_state(initial_state: &AgentInitialState) -> MutableAgentState {
    let mut messages = initial_state.messages.clone();
    let initial_message = create_initial_system_message(
        initial_state.system_prompt.as_deref(),
        Some(
            &initial_state
                .tools
                .iter()
                .map(|tool| to_tool_declaration(&tool.tool))
                .collect::<Vec<_>>(),
        ),
    );
    let first_is_system = matches!(
        messages.first(),
        Some(AgentMessage::Llm(Message::System(_)))
    );
    if !first_is_system {
        if let Some(initial_message) = initial_message {
            messages.insert(0, AgentMessage::system(initial_message));
        }
    }

    MutableAgentState {
        model: initial_state
            .model
            .clone()
            .unwrap_or_else(crate::types::default_model),
        thinking_level: initial_state
            .thinking_level
            .unwrap_or(AgentThinkingLevel::Off),
        tools: initial_state.tools.clone(),
        messages,
        is_streaming: false,
        streaming_message: None,
        pending_tool_calls: HashSet::new(),
        error_message: None,
    }
}
