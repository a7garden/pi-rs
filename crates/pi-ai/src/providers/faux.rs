//! Port of `packages/ai/src/providers/faux.ts` — the canonical scripted test
//! provider.
//!
//! Deviations:
//! - D18: upstream `randomId` uses `Date.now()` + `Math.random()`; Rust uses
//!   `utils::diagnostics::now_ms()` + `getrandom` bytes rendered in base36.
//! - D19: `estimateTokens` uses `chars().count()` instead of JavaScript's
//!   UTF-16 `.length` (identical for the ASCII strings the tests use).
//! - D20: response factories return `Result<AssistantMessage, ThrownValue>`
//!   instead of throwing; a thrown factory error resolves to the same
//!   scripted error path (message text identical).
//! - D21: the module-level `fauxProvider()` convenience plus the global
//!   api-registry (`registerFauxProvider`, compat.ts) is out of scope;
//!   [`faux_provider`] returns the [`create_provider`]-built
//!   `Arc<dyn Provider>` alongside the [`FauxCore`].

use parking_lot::Mutex;
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use futures_util::future::BoxFuture;

use crate::models::{
    create_provider, ApiDispatch, CreateProviderOptions, Provider, ProviderStreams,
};
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, CacheRetention,
    DeferredCancelOptions, DeferredFetchOptions, DeferredHandle, DeferredRequest, Message, Model,
    ModelCost, ModelCostRates, ModelInput, ModelInputLimits, ProviderResponse, SimpleStreamOptions,
    StopReason, StreamDoneReason, StreamErrorReason, StreamOptions, TextContent, ThinkingContent,
    ToolCall, ToolResultMessage, TranscriptContext, Usage, UsageCost, UserContent,
};
use crate::utils::abort::AbortSignal;
use crate::utils::diagnostics::now_ms;
use crate::utils::event_stream::{done_event, error_event, AssistantMessageEventStream};
use crate::utils::text::get_system_message_text;
use crate::utils::thrown::ThrownValue;

const DEFAULT_API: &str = "faux";
const DEFAULT_PROVIDER: &str = "faux";
const DEFAULT_MODEL_ID: &str = "faux-1";
const DEFAULT_MODEL_NAME: &str = "Faux Model";
const DEFAULT_BASE_URL: &str = "http://localhost:0";
const DEFAULT_MIN_TOKEN_SIZE: usize = 3;
const DEFAULT_MAX_TOKEN_SIZE: usize = 5;

fn default_usage() -> Usage {
    Usage {
        input: 0,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 0,
        cost: UsageCost::default(),
    }
}

fn now() -> i64 {
    now_ms()
}

/// Upstream `FauxModelDefinition`.
#[derive(Debug, Clone, Default)]
pub struct FauxModelDefinition {
    pub id: String,
    pub name: Option<String>,
    pub reasoning: Option<bool>,
    pub input: Option<Vec<ModelInput>>,
    pub input_limits: Option<ModelInputLimits>,
    pub cost: Option<ModelCost>,
    pub context_window: Option<u64>,
    pub max_tokens: Option<u64>,
}

/// One scripted assistant content block (`FauxContentBlock`).
#[derive(Debug, Clone, PartialEq)]
pub enum FauxContentBlock {
    Text(TextContent),
    Thinking(ThinkingContent),
    ToolCall(ToolCall),
}

impl From<TextContent> for FauxContentBlock {
    fn from(value: TextContent) -> Self {
        FauxContentBlock::Text(value)
    }
}

impl From<ThinkingContent> for FauxContentBlock {
    fn from(value: ThinkingContent) -> Self {
        FauxContentBlock::Thinking(value)
    }
}

impl From<ToolCall> for FauxContentBlock {
    fn from(value: ToolCall) -> Self {
        FauxContentBlock::ToolCall(value)
    }
}

/// `string | FauxContentBlock | FauxContentBlock[]` accepted by
/// [`faux_assistant_message`].
#[derive(Debug, Clone, PartialEq)]
pub enum FauxContent {
    Text(String),
    Blocks(Vec<FauxContentBlock>),
}

impl From<String> for FauxContent {
    fn from(value: String) -> Self {
        FauxContent::Text(value)
    }
}

impl From<&str> for FauxContent {
    fn from(value: &str) -> Self {
        FauxContent::Text(value.to_string())
    }
}

impl From<Vec<FauxContentBlock>> for FauxContent {
    fn from(value: Vec<FauxContentBlock>) -> Self {
        FauxContent::Blocks(value)
    }
}

impl From<FauxContentBlock> for FauxContent {
    fn from(value: FauxContentBlock) -> Self {
        FauxContent::Blocks(vec![value])
    }
}

/// Options bag of [`faux_assistant_message`] (upstream inline object).
#[derive(Debug, Clone, Default)]
pub struct FauxMessageOptions {
    pub stop_reason: Option<StopReason>,
    pub deferred: Option<DeferredHandle>,
    pub error_message: Option<String>,
    pub response_id: Option<String>,
    pub timestamp: Option<i64>,
}

/// Upstream `fauxText`.
pub fn faux_text(text: impl Into<String>) -> TextContent {
    TextContent::new(text)
}

/// Upstream `fauxThinking`.
pub fn faux_thinking(thinking: impl Into<String>) -> ThinkingContent {
    ThinkingContent::new(thinking)
}

/// Upstream `fauxToolCall`. `id` replaces the random call id when supplied.
pub fn faux_tool_call(
    name: impl Into<String>,
    arguments: crate::types::JsonObject,
    id: Option<&str>,
) -> ToolCall {
    ToolCall {
        id: id.map(str::to_string).unwrap_or_else(|| random_id("tool")),
        name: name.into(),
        arguments,
        thought_signature: None,
        namespace: None,
    }
}

fn normalize_faux_assistant_content(content: FauxContent) -> Vec<AssistantContent> {
    match content {
        FauxContent::Text(text) => vec![AssistantContent::Text(faux_text(text))],
        FauxContent::Blocks(blocks) => blocks
            .into_iter()
            .map(|block| match block {
                FauxContentBlock::Text(text) => AssistantContent::Text(text),
                FauxContentBlock::Thinking(thinking) => AssistantContent::Thinking(thinking),
                FauxContentBlock::ToolCall(tool_call) => AssistantContent::ToolCall(tool_call),
            })
            .collect(),
    }
}

/// Upstream `fauxAssistantMessage`.
pub fn faux_assistant_message(
    content: impl Into<FauxContent>,
    options: FauxMessageOptions,
) -> AssistantMessage {
    let mut message = AssistantMessage {
        content: normalize_faux_assistant_content(content.into()),
        api: DEFAULT_API.to_string(),
        provider: DEFAULT_PROVIDER.to_string(),
        model: DEFAULT_MODEL_ID.to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        usage: default_usage(),
        stop_reason: options.stop_reason.unwrap_or(StopReason::Stop),
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: options.timestamp.unwrap_or_else(now),
        duration_ms: None,
    };
    message.deferred = options.deferred;
    message.error_message = options.error_message;
    message.response_id = options.response_id;
    message
}

/// Upstream `FauxProviderState` (shared, interior-mutable).
#[derive(Default)]
pub struct FauxProviderState {
    call_count: AtomicU64,
    deferred_fetch_count: AtomicU64,
    cancelled_deferred: Mutex<Vec<DeferredHandle>>,
}

impl FauxProviderState {
    pub fn call_count(&self) -> u64 {
        self.call_count.load(Ordering::SeqCst)
    }

    pub fn deferred_fetch_count(&self) -> u64 {
        self.deferred_fetch_count.load(Ordering::SeqCst)
    }

    pub fn cancelled_deferred(&self) -> Vec<DeferredHandle> {
        self.cancelled_deferred.lock().clone()
    }
}

/// Upstream `FauxResponseFactory`. Errors are the port of thrown exceptions
/// (D20).
pub type FauxResponseFactory = Arc<
    dyn for<'a> Fn(
            &'a TranscriptContext,
            Option<&'a SimpleStreamOptions>,
            &'a FauxProviderState,
            &'a Model,
        ) -> BoxFuture<'a, Result<AssistantMessage, ThrownValue>>
        + Send
        + Sync,
>;

/// One scripted response (`FauxResponseStep`).
#[derive(Clone)]
pub enum FauxResponseStep {
    Message(AssistantMessage),
    Factory(FauxResponseFactory),
}

impl From<AssistantMessage> for FauxResponseStep {
    fn from(value: AssistantMessage) -> Self {
        FauxResponseStep::Message(value)
    }
}

impl FauxResponseStep {
    /// Wrap a synchronous factory (upstream passes plain functions).
    pub fn sync(
        factory: impl Fn(
                &TranscriptContext,
                Option<&SimpleStreamOptions>,
                &FauxProviderState,
                &Model,
            ) -> AssistantMessage
            + Send
            + Sync
            + 'static,
    ) -> FauxResponseStep {
        FauxResponseStep::Factory(Arc::new(move |context, options, state, model| {
            Box::pin(std::future::ready(Ok(factory(
                context, options, state, model,
            ))))
        }))
    }

    /// Wrap a throwing factory (upstream `() => { throw ... }`).
    pub fn failing(message: impl Into<String>) -> FauxResponseStep {
        let message = message.into();
        FauxResponseStep::Factory(Arc::new(move |_context, _options, _state, _model| {
            Box::pin(std::future::ready(Err(ThrownValue::message(
                message.clone(),
            ))))
        }))
    }
}

/// Deferred tuning of [`RegisterFauxProviderOptions`].
#[derive(Debug, Clone, Default)]
pub struct FauxDeferredOptions {
    /// Number of fetches that return the original handle before the scripted
    /// response becomes ready.
    pub pending_fetches: Option<usize>,
    pub poll_after_ms: Option<u64>,
}

/// Token pacing of [`RegisterFauxProviderOptions`].
#[derive(Debug, Clone, Default)]
pub struct FauxTokenSize {
    pub min: Option<usize>,
    pub max: Option<usize>,
}

/// Upstream `RegisterFauxProviderOptions`.
#[derive(Debug, Clone, Default)]
pub struct RegisterFauxProviderOptions {
    pub api: Option<String>,
    pub provider: Option<String>,
    pub models: Vec<FauxModelDefinition>,
    pub deferred: Option<FauxDeferredOptions>,
    pub tokens_per_second: Option<f64>,
    pub token_size: Option<FauxTokenSize>,
}

/// Upstream `estimateTokens` (D19).
fn estimate_tokens(text: &str) -> u64 {
    (text.chars().count() as u64).div_ceil(4)
}

/// Upstream `randomId` (D18).
fn random_id(prefix: &str) -> String {
    let mut bytes = [0u8; 8];
    let _ = getrandom::getrandom(&mut bytes);
    let mut value = u64::from_le_bytes(bytes);
    let mut suffix = String::new();
    while suffix.len() < 4 {
        let digit = (value % 36) as u32;
        value /= 36;
        let ch = if digit < 10 {
            char::from(b'0' + digit as u8)
        } else {
            char::from(b'a' + (digit - 10) as u8)
        };
        suffix.push(ch);
    }
    format!("{prefix}:{:x}:{suffix}", now_ms())
}

fn assistant_content_to_text(content: &[AssistantContent]) -> String {
    content
        .iter()
        .map(|block| match block {
            AssistantContent::Text(text) => text.text.clone(),
            AssistantContent::Thinking(thinking) => thinking.thinking.clone(),
            AssistantContent::ToolCall(tool_call) => {
                format!(
                    "{}:{}",
                    tool_call.name,
                    serde_json::to_string(&tool_call.arguments).expect("arguments serialize")
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_result_to_text(message: &ToolResultMessage) -> String {
    let mut parts = vec![message.tool_name.clone()];
    for block in &message.content {
        if let UserContent::Text(text) = block {
            parts.push(text.text.clone());
        } else if let UserContent::Image(image) = block {
            parts.push(format!("[image:{}:{}]", image.mime_type, image.data.len()));
        }
    }
    parts.join("\n")
}

fn message_to_text(message: &Message) -> String {
    match message {
        Message::System(system) => {
            let mut parts = vec![get_system_message_text(system)];
            for tool in system.tools_removed.iter().flatten() {
                parts.push(format!(
                    "tool-:{}",
                    serde_json::to_string(tool).expect("tool reference serializes")
                ));
            }
            for tool in system.tools_added.iter().flatten() {
                parts.push(format!(
                    "tool+:{}",
                    serde_json::to_string(tool).expect("tool serializes")
                ));
            }
            parts
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        }
        Message::User(user) => match &user.content {
            crate::types::UserContentField::Text(text) => text.clone(),
            crate::types::UserContentField::Blocks(blocks) => blocks
                .iter()
                .map(|block| match block {
                    UserContent::Text(text) => text.text.clone(),
                    UserContent::Image(image) => {
                        format!("[image:{}:{}]", image.mime_type, image.data.len())
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        },
        Message::Assistant(assistant) => assistant_content_to_text(&assistant.content),
        Message::ToolResult(tool_result) => tool_result_to_text(tool_result),
    }
}

/// Length of the prompt text that joins `messages` with blank lines
/// (`joinedLength`).
fn joined_length(messages: &[String], count: usize) -> usize {
    let mut length = if count > 0 { (count - 1) * 2 } else { 0 };
    for message in messages.iter().take(count) {
        length += message.chars().count();
    }
    length
}

/// Length of the common prefix of the two joined prompts. Equal messages are
/// compared whole; characters are compared only from the first message that
/// differs (`commonPromptPrefixLength`).
fn common_prompt_prefix_length(previous: &[String], current: &[String]) -> usize {
    let mut index = 0;
    while index < previous.len() && index < current.len() && previous[index] == current[index] {
        index += 1;
    }
    let rest = |messages: &[String]| -> String {
        if index == messages.len() {
            String::new()
        } else {
            format!(
                "{}{}",
                if index > 0 { "\n\n" } else { "" },
                messages[index..].join("\n\n")
            )
        }
    };
    joined_length(previous, index) + common_prefix_length(&rest(previous), &rest(current))
}

fn common_prefix_length(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let length = a.len().min(b.len());
    let mut index = 0;
    while index < length && a[index] == b[index] {
        index += 1;
    }
    index
}

/// Upstream `withUsageEstimate`.
fn with_usage_estimate(
    mut message: AssistantMessage,
    context: &TranscriptContext,
    options: Option<&SimpleStreamOptions>,
    prompt_cache: &Mutex<HashMap<String, Vec<String>>>,
) -> AssistantMessage {
    // One text per message; the whole prompt joins them with blank lines.
    let prompt: Vec<String> = context
        .messages()
        .iter()
        .map(|message| format!("{}:{}", message.role(), message_to_text(message)))
        .collect();
    let prompt_length = joined_length(&prompt, prompt.len());
    let prompt_tokens = (prompt_length as u64).div_ceil(4);
    let output_tokens = estimate_tokens(&assistant_content_to_text(&message.content));
    let mut input = prompt_tokens;
    let mut cache_read = 0;
    let mut cache_write = 0;
    let session_id = options.and_then(|options| options.session_id.clone());
    let cache_retention_none = matches!(
        options.and_then(|options| options.cache_retention),
        Some(CacheRetention::None)
    );

    if let Some(session_id) = session_id {
        if !cache_retention_none {
            let mut cache = prompt_cache.lock();
            let previous_prompt = cache.get(&session_id).cloned();
            match previous_prompt {
                Some(previous_prompt) => {
                    let cached_chars = common_prompt_prefix_length(&previous_prompt, &prompt);
                    cache_read = (cached_chars as u64).div_ceil(4);
                    cache_write = ((prompt_length - cached_chars) as u64).div_ceil(4);
                    input = prompt_tokens.saturating_sub(cache_read);
                }
                None => {
                    cache_write = prompt_tokens;
                }
            }
            cache.insert(session_id, prompt);
        }
    }

    message.usage = Usage {
        input,
        output: output_tokens,
        cache_read,
        cache_write,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: input + output_tokens + cache_read + cache_write,
        cost: UsageCost::default(),
    };
    message
}

/// Upstream `splitStringByTokenSize`.
fn split_string_by_token_size(
    text: &str,
    min_token_size: usize,
    max_token_size: usize,
) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut index = 0;
    let text: Vec<char> = text.chars().collect();
    while index < text.len() {
        let spread = (max_token_size - min_token_size + 1) as u64;
        let token_size = min_token_size as u64 + faux_rand_u64() % spread;
        let char_size = (token_size * 4).max(1) as usize;
        let end = (index + char_size).min(text.len());
        chunks.push(text[index..end].iter().collect());
        index = end;
    }
    if chunks.is_empty() {
        vec![String::new()]
    } else {
        chunks
    }
}

fn faux_rand_u64() -> u64 {
    let mut bytes = [0u8; 8];
    let _ = getrandom::getrandom(&mut bytes);
    u64::from_le_bytes(bytes)
}

/// Upstream `cloneMessage` (`structuredClone` plus re-stamping).
fn clone_message(
    message: &AssistantMessage,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    let mut cloned = message.clone();
    cloned.api = api.to_string();
    cloned.provider = provider.to_string();
    cloned.model = model_id.to_string();
    if cloned.timestamp == 0 {
        cloned.timestamp = now();
    }
    cloned
}

fn create_deferred_message(model: &Model, handle: DeferredHandle) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api().to_string(),
        provider: model.provider().to_string(),
        model: model.id().to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        usage: default_usage(),
        stop_reason: StopReason::Deferred,
        deferred: Some(handle),
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now(),
        duration_ms: None,
    }
}

fn create_error_message(
    error: &ThrownValue,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    let error_message = match error {
        ThrownValue::Error(error) => error.to_string(),
        ThrownValue::Text(text) => text.clone(),
    };
    AssistantMessage {
        content: Vec::new(),
        api: api.to_string(),
        provider: provider.to_string(),
        model: model_id.to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        usage: default_usage(),
        stop_reason: StopReason::Error,
        deferred: None,
        error_message: Some(error_message),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now(),
        duration_ms: None,
    }
}

fn create_aborted_message(partial: &AssistantMessage) -> AssistantMessage {
    let mut aborted = partial.clone();
    aborted.stop_reason = StopReason::Aborted;
    aborted.error_message = Some("Request was aborted".to_string());
    aborted.timestamp = now();
    aborted
}

async fn schedule_chunk(chunk: &str, tokens_per_second: Option<f64>) {
    match tokens_per_second {
        Some(tokens_per_second) if tokens_per_second > 0.0 => {
            let delay_ms = (estimate_tokens(chunk) as f64 / tokens_per_second) * 1000.0;
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms as u64)).await;
        }
        _ => tokio::task::yield_now().await,
    }
}

/// Upstream `streamWithDeltas`. Only failure path: a scripted message without
/// a terminal stop reason (D20 converts the upstream throw).
async fn stream_with_deltas(
    stream: &AssistantMessageEventStream,
    message: AssistantMessage,
    min_token_size: usize,
    max_token_size: usize,
    tokens_per_second: Option<f64>,
    signal: Option<&AbortSignal>,
) -> Result<(), ThrownValue> {
    let mut partial = message.clone();
    partial.content = Vec::new();
    partial.stop_reason = StopReason::Pending;

    let aborted_now = |partial: &AssistantMessage| {
        let aborted = create_aborted_message(partial);
        stream.push(error_event(aborted.clone(), StreamErrorReason::Aborted));
        stream.end(Some(aborted));
    };

    if signal.is_some_and(|signal| signal.aborted()) {
        aborted_now(&partial);
        return Ok(());
    }

    stream.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });

    for index in 0..message.content.len() {
        if signal.is_some_and(|signal| signal.aborted()) {
            aborted_now(&partial);
            return Ok(());
        }

        match message.content[index].clone() {
            AssistantContent::Thinking(block) => {
                partial
                    .content
                    .push(AssistantContent::Thinking(ThinkingContent::new("")));
                stream.push(AssistantMessageEvent::ThinkingStart {
                    content_index: index as u32,
                    partial: partial.clone(),
                });
                for chunk in
                    split_string_by_token_size(&block.thinking, min_token_size, max_token_size)
                {
                    schedule_chunk(&chunk, tokens_per_second).await;
                    if signal.is_some_and(|signal| signal.aborted()) {
                        aborted_now(&partial);
                        return Ok(());
                    }
                    if let Some(AssistantContent::Thinking(entry)) = partial.content.get_mut(index)
                    {
                        entry.thinking.push_str(&chunk);
                    }
                    stream.push(AssistantMessageEvent::ThinkingDelta {
                        content_index: index as u32,
                        delta: chunk,
                        partial: partial.clone(),
                    });
                }
                stream.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: index as u32,
                    content: block.thinking,
                    partial: partial.clone(),
                });
            }
            AssistantContent::Text(block) => {
                partial
                    .content
                    .push(AssistantContent::Text(TextContent::new("")));
                stream.push(AssistantMessageEvent::TextStart {
                    content_index: index as u32,
                    partial: partial.clone(),
                });
                for chunk in split_string_by_token_size(&block.text, min_token_size, max_token_size)
                {
                    schedule_chunk(&chunk, tokens_per_second).await;
                    if signal.is_some_and(|signal| signal.aborted()) {
                        aborted_now(&partial);
                        return Ok(());
                    }
                    if let Some(AssistantContent::Text(entry)) = partial.content.get_mut(index) {
                        entry.text.push_str(&chunk);
                    }
                    stream.push(AssistantMessageEvent::TextDelta {
                        content_index: index as u32,
                        delta: chunk,
                        partial: partial.clone(),
                    });
                }
                stream.push(AssistantMessageEvent::TextEnd {
                    content_index: index as u32,
                    content: block.text,
                    partial: partial.clone(),
                });
            }
            AssistantContent::ToolCall(block) => {
                partial.content.push(AssistantContent::ToolCall(ToolCall {
                    id: block.id.clone(),
                    name: block.name.clone(),
                    arguments: Default::default(),
                    thought_signature: None,
                    namespace: None,
                }));
                stream.push(AssistantMessageEvent::ToolcallStart {
                    content_index: index as u32,
                    partial: partial.clone(),
                });
                let serialized =
                    serde_json::to_string(&block.arguments).expect("tool call arguments serialize");
                for chunk in split_string_by_token_size(&serialized, min_token_size, max_token_size)
                {
                    schedule_chunk(&chunk, tokens_per_second).await;
                    if signal.is_some_and(|signal| signal.aborted()) {
                        aborted_now(&partial);
                        return Ok(());
                    }
                    stream.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: index as u32,
                        delta: chunk,
                        partial: partial.clone(),
                    });
                }
                if let Some(AssistantContent::ToolCall(entry)) = partial.content.get_mut(index) {
                    entry.arguments = block.arguments.clone();
                }
                stream.push(AssistantMessageEvent::ToolcallEnd {
                    content_index: index as u32,
                    tool_call: block.clone(),
                    partial: partial.clone(),
                });
            }
        }
    }

    if message.stop_reason == StopReason::Pending {
        return Err(ThrownValue::message(
            "Faux response ended without a stop reason",
        ));
    }

    match message.stop_reason {
        StopReason::Pending => unreachable!("pending handled above"),
        StopReason::Error | StopReason::Aborted => {
            let reason = if message.stop_reason == StopReason::Error {
                StreamErrorReason::Error
            } else {
                StreamErrorReason::Aborted
            };
            stream.push(error_event(message.clone(), reason));
            stream.end(Some(message));
        }
        StopReason::Stop => {
            stream.push(done_event(message.clone(), StreamDoneReason::Stop));
            stream.end(Some(message));
        }
        StopReason::Length => {
            stream.push(done_event(message.clone(), StreamDoneReason::Length));
            stream.end(Some(message));
        }
        StopReason::ToolUse => {
            stream.push(done_event(message.clone(), StreamDoneReason::ToolUse));
            stream.end(Some(message));
        }
        StopReason::Deferred => {
            stream.push(done_event(message.clone(), StreamDoneReason::Deferred));
            stream.end(Some(message));
        }
    }
    Ok(())
}

struct DeferredEntry {
    handle: DeferredHandle,
    step: FauxResponseStep,
    context: TranscriptContext,
    options: Option<SimpleStreamOptions>,
    model: Model,
    pending_fetches: usize,
    cancelled: bool,
    final_message: Option<AssistantMessage>,
}

/// Lock-free snapshot of a deferred entry awaiting its final resolution.
struct FauxDeferredRequest {
    step: FauxResponseStep,
    context: TranscriptContext,
    options: Option<SimpleStreamOptions>,
    model: Model,
    existing_final: Option<AssistantMessage>,
}

struct FauxCoreInner {
    api: String,
    provider: String,
    min_token_size: usize,
    max_token_size: usize,
    tokens_per_second: Option<f64>,
    deferred_options: Option<FauxDeferredOptions>,
    models: Vec<Model>,
    state: FauxProviderState,
    pending_responses: Mutex<Vec<FauxResponseStep>>,
    prompt_cache: Mutex<HashMap<String, Vec<String>>>,
    deferred_responses: Mutex<HashMap<String, DeferredEntry>>,
}

/// Upstream `createFauxCore(...)` result.
#[derive(Clone)]
pub struct FauxCore {
    inner: Arc<FauxCoreInner>,
}

impl FauxCore {
    /// Upstream `createFauxCore(options)`.
    pub fn new(options: RegisterFauxProviderOptions) -> FauxCore {
        let api = options
            .api
            .clone()
            .unwrap_or_else(|| random_id(DEFAULT_API));
        let provider = options
            .provider
            .clone()
            .unwrap_or_else(|| DEFAULT_PROVIDER.to_string());
        let token_size = options.token_size.clone().unwrap_or_default();
        let requested_min = token_size.min.unwrap_or(DEFAULT_MIN_TOKEN_SIZE);
        let requested_max = token_size.max.unwrap_or(DEFAULT_MAX_TOKEN_SIZE);
        let min_token_size = requested_min.clamp(1, requested_max).max(1);
        let max_token_size = min_token_size.max(requested_max);
        let model_definitions = if !options.models.is_empty() {
            options.models
        } else {
            vec![FauxModelDefinition {
                id: DEFAULT_MODEL_ID.to_string(),
                name: Some(DEFAULT_MODEL_NAME.to_string()),
                reasoning: Some(false),
                input: Some(vec![ModelInput::Text, ModelInput::Image]),
                cost: Some(ModelCost::default()),
                context_window: Some(128_000),
                max_tokens: Some(16_384),
                input_limits: None,
            }]
        };
        let models = model_definitions
            .into_iter()
            .map(|definition| Model {
                base: crate::types::BaseModelFields {
                    id: definition.id.clone(),
                    name: definition.name.unwrap_or_else(|| definition.id.clone()),
                    api: api.clone(),
                    provider: provider.clone(),
                    base_url: DEFAULT_BASE_URL.to_string(),
                    input: definition
                        .input
                        .unwrap_or_else(|| vec![ModelInput::Text, ModelInput::Image]),
                    input_limits: definition.input_limits,
                    cost: definition.cost.unwrap_or_else(|| ModelCost {
                        rates: ModelCostRates::default(),
                        tiers: None,
                    }),
                    headers: None,
                },
                model_type: None,
                reasoning: definition.reasoning.unwrap_or(false),
                thinking_level_map: None,
                prompt_cache: None,
                context_window: definition.context_window.unwrap_or(128_000),
                max_tokens: definition.max_tokens.unwrap_or(16_384),
                sampling_params: None,
                sampling_params_by_thinking_level: None,
                compat: None,
            })
            .collect();

        FauxCore {
            inner: Arc::new(FauxCoreInner {
                api,
                provider,
                min_token_size,
                max_token_size,
                tokens_per_second: options.tokens_per_second,
                deferred_options: options.deferred,
                models,
                state: FauxProviderState::default(),
                pending_responses: Mutex::new(Vec::new()),
                prompt_cache: Mutex::new(HashMap::new()),
                deferred_responses: Mutex::new(HashMap::new()),
            }),
        }
    }

    pub fn api(&self) -> &str {
        &self.inner.api
    }

    pub fn provider(&self) -> &str {
        &self.inner.provider
    }

    pub fn models(&self) -> Vec<Model> {
        self.inner.models.clone()
    }

    /// Upstream `getModel()` — the first model.
    pub fn first_model(&self) -> Option<Model> {
        self.inner.models.first().cloned()
    }

    /// Upstream `getModel(id)`.
    pub fn get_model(&self, model_id: &str) -> Option<Model> {
        self.inner
            .models
            .iter()
            .find(|model| model.id() == model_id)
            .cloned()
    }

    pub fn state(&self) -> &FauxProviderState {
        &self.inner.state
    }

    pub fn set_responses(&self, responses: Vec<FauxResponseStep>) {
        *self.inner.pending_responses.lock() = responses;
    }

    pub fn append_responses(&self, responses: Vec<FauxResponseStep>) {
        self.inner.pending_responses.lock().extend(responses);
    }

    pub fn get_pending_response_count(&self) -> usize {
        self.inner.pending_responses.lock().len()
    }

    async fn resolve_response(
        &self,
        step: &FauxResponseStep,
        context: &TranscriptContext,
        stream_options: Option<&SimpleStreamOptions>,
        request_model: &Model,
    ) -> Result<AssistantMessage, ThrownValue> {
        let resolved = match step {
            FauxResponseStep::Message(message) => message.clone(),
            FauxResponseStep::Factory(factory) => {
                factory(context, stream_options, &self.inner.state, request_model).await?
            }
        };
        let cloned = clone_message(
            &resolved,
            &self.inner.api,
            &self.inner.provider,
            request_model.id(),
        );
        Ok(with_usage_estimate(
            cloned,
            context,
            stream_options,
            &self.inner.prompt_cache,
        ))
    }

    /// Upstream `stream` (`StreamFunction<string, SimpleStreamOptions>`).
    pub fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        stream_options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        let outer = AssistantMessageEventStream::new();
        let step = {
            let mut pending = self.inner.pending_responses.lock();
            if pending.is_empty() {
                None
            } else {
                Some(pending.remove(0))
            }
        };
        self.inner.state.call_count.fetch_add(1, Ordering::SeqCst);

        let inner = Arc::clone(&self.inner);
        let task_stream = outer.clone();
        let model = model.clone();
        let context = context.clone();
        let stream_options = stream_options.cloned();
        tokio::spawn(async move {
            let outer = task_stream;
            let result: Result<(), ThrownValue> = async {
                if let Some(on_response) = stream_options
                    .as_ref()
                    .and_then(|options| options.base.base.on_response.clone())
                {
                    on_response(
                        ProviderResponse {
                            status: 200,
                            headers: BTreeMap::new(),
                        },
                        model.clone(),
                    )
                    .await;
                }

                let step = match step {
                    Some(step) => step,
                    None => {
                        let mut message = create_error_message(
                            &ThrownValue::message("No more faux responses queued"),
                            &inner.api,
                            &inner.provider,
                            model.id(),
                        );
                        message = with_usage_estimate(
                            message,
                            &context,
                            stream_options.as_ref(),
                            &inner.prompt_cache,
                        );
                        outer.push(error_event(message.clone(), StreamErrorReason::Error));
                        outer.end(Some(message));
                        return Ok(());
                    }
                };

                let deferred_enabled = stream_options
                    .as_ref()
                    .and_then(|options| options.deferred.as_ref())
                    .is_some_and(DeferredRequest::is_enabled);
                if deferred_enabled {
                    let handle = DeferredHandle {
                        provider: model.provider().to_string(),
                        model_id: model.id().to_string(),
                        api: model.api().to_string(),
                        id: random_id("deferred"),
                        expires_at: None,
                        poll_after_ms: inner
                            .deferred_options
                            .as_ref()
                            .and_then(|options| options.poll_after_ms),
                        data: None,
                    };
                    let entry = DeferredEntry {
                        handle: handle.clone(),
                        step,
                        context: context.clone(),
                        options: stream_options.clone(),
                        model: model.clone(),
                        pending_fetches: inner
                            .deferred_options
                            .as_ref()
                            .and_then(|options| options.pending_fetches)
                            .unwrap_or(0),
                        cancelled: false,
                        final_message: None,
                    };
                    inner
                        .deferred_responses
                        .lock()
                        .insert(handle.id.clone(), entry);
                    let deferred_stream = stream_with_deltas(
                        &outer,
                        create_deferred_message(&model, handle),
                        inner.min_token_size,
                        inner.max_token_size,
                        inner.tokens_per_second,
                        stream_options
                            .as_ref()
                            .and_then(|options| options.base.base.signal.as_ref()),
                    );
                    let _ = deferred_stream.await;
                    return Ok(());
                }

                let core = FauxCore {
                    inner: Arc::clone(&inner),
                };
                let message = core
                    .resolve_response(&step, &context, stream_options.as_ref(), &model)
                    .await?;
                // Upstream: a throw inside the delta-emission loop (e.g. a
                // response ending without a stop reason) propagates to the
                // task's error handler below, which emits the terminal error.
                stream_with_deltas(
                    &outer,
                    message,
                    inner.min_token_size,
                    inner.max_token_size,
                    inner.tokens_per_second,
                    stream_options
                        .as_ref()
                        .and_then(|options| options.base.base.signal.as_ref()),
                )
                .await?;
                Ok(())
            }
            .await;
            if let Err(error) = result {
                let message = create_error_message(&error, &inner.api, &inner.provider, model.id());
                outer.push(error_event(message.clone(), StreamErrorReason::Error));
                outer.end(Some(message));
            }
        });

        outer
    }

    /// Upstream `fetchDeferred`.
    pub fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        fetch_options: Option<&DeferredFetchOptions>,
    ) -> AssistantMessageEventStream {
        let outer = AssistantMessageEventStream::new();
        self.inner
            .state
            .deferred_fetch_count
            .fetch_add(1, Ordering::SeqCst);

        let inner = Arc::clone(&self.inner);
        let task_stream = outer.clone();
        let model = model.clone();
        let handle = handle.clone();
        let fetch_options = fetch_options.cloned();
        tokio::spawn(async move {
            let outer = task_stream;
            let result: Result<(), ThrownValue> = async {
                if let Some(on_response) = fetch_options
                    .as_ref()
                    .and_then(|options| options.base.on_response.clone())
                {
                    on_response(
                        ProviderResponse {
                            status: 200,
                            headers: BTreeMap::new(),
                        },
                        model.clone(),
                    )
                    .await;
                }

                // Validate and apply pending-fetch accounting under a short
                // lock; the final resolution awaits, so it happens outside
                // the guard (upstream mutates the shared entry object but
                // never holds anything across an await).
                let action = {
                    let mut entries = inner.deferred_responses.lock();
                    let entry = match entries.get_mut(&handle.id) {
                        Some(entry)
                            if entry.handle.provider == handle.provider
                                && entry.handle.model_id == handle.model_id
                                && entry.handle.api == handle.api =>
                        {
                            entry
                        }
                        _ => {
                            return Err(ThrownValue::message(format!(
                                "Unknown faux deferred response: {}",
                                handle.id
                            )))
                        }
                    };
                    if entry.cancelled {
                        return Err(ThrownValue::message(format!(
                            "Faux deferred response was cancelled: {}",
                            handle.id
                        )));
                    }
                    if entry.pending_fetches > 0 {
                        entry.pending_fetches -= 1;
                        None
                    } else {
                        Some(FauxDeferredRequest {
                            step: entry.step.clone(),
                            context: entry.context.clone(),
                            options: entry.options.clone(),
                            model: entry.model.clone(),
                            existing_final: entry.final_message.clone(),
                        })
                    }
                };

                enum Action {
                    Pending,
                    Final(AssistantMessage),
                }
                let action = match action {
                    None => Action::Pending,
                    Some(request) => {
                        let final_message = match request.existing_final {
                            Some(final_message) => final_message,
                            None => {
                                let core = FauxCore {
                                    inner: Arc::clone(&inner),
                                };
                                let mut submission_options = request.options.clone();
                                if let Some(options) = submission_options.as_mut() {
                                    options.deferred = None;
                                    options.base.base.signal = None;
                                    options.base.base.on_response = None;
                                }
                                let resolved = core
                                    .resolve_response(
                                        &request.step,
                                        &request.context,
                                        submission_options.as_ref(),
                                        &request.model,
                                    )
                                    .await;
                                let message = match resolved {
                                    Ok(message) => message,
                                    Err(error) => create_error_message(
                                        &error,
                                        &inner.api,
                                        &inner.provider,
                                        request.model.id(),
                                    ),
                                };
                                inner
                                    .deferred_responses
                                    .lock()
                                    .get_mut(&handle.id)
                                    .expect("entry still registered")
                                    .final_message = Some(message.clone());
                                message
                            }
                        };
                        Action::Final(final_message)
                    }
                };

                match action {
                    Action::Pending => {
                        let message = create_deferred_message(&model, handle);
                        let _ = stream_with_deltas(
                            &outer,
                            message,
                            inner.min_token_size,
                            inner.max_token_size,
                            inner.tokens_per_second,
                            fetch_options
                                .as_ref()
                                .and_then(|options| options.base.signal.as_ref()),
                        )
                        .await;
                    }
                    Action::Final(message) => {
                        let _ = stream_with_deltas(
                            &outer,
                            message,
                            inner.min_token_size,
                            inner.max_token_size,
                            inner.tokens_per_second,
                            fetch_options
                                .as_ref()
                                .and_then(|options| options.base.signal.as_ref()),
                        )
                        .await;
                    }
                }
                Ok(())
            }
            .await;
            if let Err(error) = result {
                let message = create_error_message(&error, &inner.api, &inner.provider, model.id());
                outer.push(error_event(message.clone(), StreamErrorReason::Error));
                outer.end(Some(message));
            }
        });

        outer
    }

    /// Upstream `cancelDeferred`.
    pub async fn cancel_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        cancel_options: Option<&DeferredCancelOptions>,
    ) {
        self.inner
            .state
            .cancelled_deferred
            .lock()
            .push(handle.clone());
        if let Some(entry) = self.inner.deferred_responses.lock().get_mut(&handle.id) {
            entry.cancelled = true;
        }
        if let Some(on_response) =
            cancel_options.and_then(|options| options.base.on_response.clone())
        {
            on_response(
                ProviderResponse {
                    status: 200,
                    headers: BTreeMap::new(),
                },
                model.clone(),
            )
            .await;
        }
    }
}

/// [`ProviderStreams`] adapter over a [`FauxCore`].
#[derive(Clone)]
struct FauxStreams {
    core: FauxCore,
}

impl ProviderStreams for FauxStreams {
    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        self.core.stream(
            model,
            context,
            Some(&SimpleStreamOptions::from(options.clone())),
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.core.stream(model, context, Some(options))
    }

    fn has_fetch_deferred(&self) -> bool {
        true
    }

    fn has_cancel_deferred(&self) -> bool {
        true
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: &DeferredFetchOptions,
    ) -> Option<AssistantMessageEventStream> {
        Some(self.core.fetch_deferred(model, handle, Some(options)))
    }

    fn cancel_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: &DeferredCancelOptions,
    ) -> Option<
        Pin<Box<dyn Future<Output = Result<(), crate::utils::thrown::ThrownValue>> + Send + '_>>,
    > {
        let core = self.core.clone();
        let model = model.clone();
        let handle = handle.clone();
        let options = options.clone();
        Some(Box::pin(async move {
            core.cancel_deferred(&model, &handle, Some(&options)).await;
            Ok(())
        }))
    }
}

/// Upstream `FauxProviderHandle`.
pub struct FauxProviderHandle {
    /// The `createProvider`-built provider.
    pub provider: Arc<dyn Provider>,
    /// The scripted core (api id, models, queue control, state).
    pub core: FauxCore,
}

impl FauxProviderHandle {
    pub fn api(&self) -> &str {
        self.core.api()
    }

    pub fn models(&self) -> Vec<Model> {
        self.core.models()
    }

    pub fn first_model(&self) -> Option<Model> {
        self.core.first_model()
    }

    pub fn get_model(&self, model_id: &str) -> Option<Model> {
        self.core.get_model(model_id)
    }

    pub fn state(&self) -> &FauxProviderState {
        self.core.state()
    }

    pub fn set_responses(&self, responses: Vec<FauxResponseStep>) {
        self.core.set_responses(responses);
    }

    pub fn append_responses(&self, responses: Vec<FauxResponseStep>) {
        self.core.append_responses(responses);
    }

    pub fn get_pending_response_count(&self) -> usize {
        self.core.get_pending_response_count()
    }
}

/// The faux api-key `resolve`: ambient-only, always resolves to an empty
/// auth (upstream `{ auth: {} }`).
fn resolve_faux<'a>(
    _input: crate::auth::types::ApiKeyAuthInput<'a>,
) -> BoxFuture<'a, Result<Option<crate::auth::types::AuthResult>, ThrownValue>> {
    Box::pin(async move { Ok(Some(crate::auth::types::AuthResult::default())) })
}

/// Upstream `fauxProvider(options)`: builds the faux core and wraps it in a
/// real provider via [`create_provider`].
pub fn faux_provider(options: RegisterFauxProviderOptions) -> FauxProviderHandle {
    let core = FauxCore::new(options);
    let provider = create_provider(CreateProviderOptions {
        id: core.provider().to_string(),
        name: None,
        base_url: None,
        headers: None,
        auth: crate::auth::types::ProviderAuth {
            api_key: Some(crate::auth::types::ApiKeyAuth {
                name: "Faux".to_string(),
                login: None,
                check: None,
                resolve: Box::new(resolve_faux),
            }),
            oauth: None,
        },
        models: core
            .models()
            .into_iter()
            .map(crate::types::AnyModel::Chat)
            .collect(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
        api: Some(ApiDispatch::Single(Arc::new(FauxStreams {
            core: core.clone(),
        }))),
        images: Default::default(),
        classifiers: Default::default(),
    })
    .expect("faux provider configuration is valid");
    FauxProviderHandle { provider, core }
}

#[cfg(test)]
mod tests {
    // Port of test/faux-provider.test.ts. The "unregisters the provider" case
    // is NOT ported: it exercises the global api-registry (`registerApiProvider`
    // / compat.ts), which is out of scope for this module. The token-estimation
    // test's expected serialization is adjusted to the actual upstream
    // `serializeContext` output (`system:sys\ntool+:...` as one segment); the
    // upstream test file's `tools:` literal cannot be produced by the upstream
    // implementation either.
    use super::*;
    use crate::types::{
        AssistantContent, Context, ImageContent, StopReason, Tool, UserContentField, UserMessage,
    };
    use crate::utils::abort::AbortController;
    use crate::utils::transcript::normalize_context;
    use futures_util::StreamExt;

    fn faux_user_message(text: &str, timestamp: i64) -> Message {
        Message::User(UserMessage {
            content: UserContentField::Text(text.to_string()),
            timestamp,
        })
    }

    async fn complete(
        core: &FauxCore,
        model: &Model,
        context: &Context,
        options: Option<SimpleStreamOptions>,
    ) -> AssistantMessage {
        core.stream(model, &normalize_context(context), options.as_ref())
            .into_result()
            .await
    }

    async fn collect_events(
        core: &FauxCore,
        model: &Model,
        context: &Context,
        options: Option<SimpleStreamOptions>,
    ) -> Vec<AssistantMessageEvent> {
        let mut events = Vec::new();
        let mut stream = core.stream(model, &normalize_context(context), options.as_ref());
        while let Some(event) = stream.next().await {
            events.push(event);
        }
        events
    }

    fn message_of_content(content: impl Into<FauxContent>) -> AssistantMessage {
        faux_assistant_message(content, FauxMessageOptions::default())
    }

    fn event_types(events: &[AssistantMessageEvent]) -> Vec<&'static str> {
        events
            .iter()
            .map(|event| match event {
                AssistantMessageEvent::Start { .. } => "start",
                AssistantMessageEvent::TextStart { .. } => "text_start",
                AssistantMessageEvent::TextDelta { .. } => "text_delta",
                AssistantMessageEvent::TextEnd { .. } => "text_end",
                AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
                AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
                AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
                AssistantMessageEvent::ToolcallStart { .. } => "toolcall_start",
                AssistantMessageEvent::ToolcallDelta { .. } => "toolcall_delta",
                AssistantMessageEvent::ToolcallEnd { .. } => "toolcall_end",
                AssistantMessageEvent::Done { .. } => "done",
                AssistantMessageEvent::Error { .. } => "error",
            })
            .collect()
    }

    #[tokio::test]
    async fn registers_a_custom_provider_and_estimates_usage() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![message_of_content("hello world").into()]);

        let context = Context {
            system_prompt: Some("Be concise.".to_string()),
            messages: vec![faux_user_message("hi there", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");

        let response = complete(&core, &model, &context, None).await;
        assert_eq!(
            response.content,
            vec![AssistantContent::Text(TextContent::new("hello world"))]
        );
        assert!(response.usage.input > 0);
        assert!(response.usage.output > 0);
        assert_eq!(
            response.usage.total_tokens,
            response.usage.input + response.usage.output
        );
        assert_eq!(core.state().call_count(), 1);
    }

    #[tokio::test]
    async fn supports_helper_blocks_for_text_thinking_and_tool_calls() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![faux_assistant_message(
            FauxContent::Blocks(vec![
                FauxContentBlock::Thinking(faux_thinking("think")),
                FauxContentBlock::ToolCall(faux_tool_call(
                    "echo",
                    serde_json::json!({ "text": "hi" })
                        .as_object()
                        .expect("object")
                        .clone(),
                    // Fixed id: the assertion compares full blocks and the
                    // default id is random (upstream test uses a fixed id
                    // for the same reason).
                    Some("tool_fixed"),
                )),
                FauxContentBlock::Text(faux_text("done")),
            ]),
            FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into()]);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");
        let response = complete(&core, &model, &context, None).await;

        assert_eq!(
            response.content,
            vec![
                AssistantContent::Thinking(ThinkingContent::new("think")),
                AssistantContent::ToolCall(faux_tool_call(
                    "echo",
                    serde_json::json!({ "text": "hi" })
                        .as_object()
                        .expect("object")
                        .clone(),
                    Some("tool_fixed")
                )),
                AssistantContent::Text(TextContent::new("done")),
            ]
        );
        assert_eq!(response.stop_reason, StopReason::ToolUse);
    }

    #[tokio::test]
    async fn supports_multiple_models_with_per_model_reasoning_and_model_aware_factories() {
        let core = FauxCore::new(RegisterFauxProviderOptions {
            models: vec![
                FauxModelDefinition {
                    id: "faux-fast".to_string(),
                    name: Some("Faux Fast".to_string()),
                    reasoning: Some(false),
                    ..Default::default()
                },
                FauxModelDefinition {
                    id: "faux-thinker".to_string(),
                    name: Some("Faux Thinker".to_string()),
                    reasoning: Some(true),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        core.set_responses(vec![
            FauxResponseStep::sync(|_context, _options, _state, model| {
                faux_assistant_message(
                    format!("{}:{}", model.id(), model.reasoning),
                    FauxMessageOptions::default(),
                )
            }),
            FauxResponseStep::sync(|_context, _options, _state, model| {
                faux_assistant_message(
                    format!("{}:{}", model.id(), model.reasoning),
                    FauxMessageOptions::default(),
                )
            }),
        ]);

        let model_ids: Vec<String> = core
            .models()
            .iter()
            .map(|model| model.id().to_string())
            .collect();
        assert_eq!(
            model_ids,
            vec!["faux-fast".to_string(), "faux-thinker".to_string()]
        );
        assert_eq!(core.first_model().expect("first").id(), "faux-fast");
        assert!(!core.get_model("faux-fast").expect("fast").reasoning);
        assert!(core.get_model("faux-thinker").expect("thinker").reasoning);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let fast = complete(
            &core,
            &core.get_model("faux-fast").expect("fast"),
            &context,
            None,
        )
        .await;
        let thinker = complete(
            &core,
            &core.get_model("faux-thinker").expect("thinker"),
            &context,
            None,
        )
        .await;

        assert_eq!(
            fast.content,
            vec![AssistantContent::Text(TextContent::new("faux-fast:false"))]
        );
        assert_eq!(
            thinker.content,
            vec![AssistantContent::Text(TextContent::new(
                "faux-thinker:true"
            ))]
        );
    }

    #[tokio::test]
    async fn rewrites_api_provider_and_model_on_returned_messages() {
        let core = FauxCore::new(RegisterFauxProviderOptions {
            api: Some("faux:test".to_string()),
            provider: Some("faux-provider".to_string()),
            models: vec![FauxModelDefinition {
                id: "faux-model".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        });
        core.set_responses(vec![message_of_content("hello").into()]);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("model");
        let response = complete(&core, &model, &context, None).await;

        assert_eq!(response.api, "faux:test");
        assert_eq!(response.provider, "faux-provider");
        assert_eq!(response.model, "faux-model");
    }

    #[tokio::test]
    async fn consumes_queued_responses_in_order_and_errors_when_exhausted() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![
            message_of_content("first").into(),
            message_of_content("second").into(),
        ]);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");

        let first = complete(&core, &model, &context, None).await;
        let second = complete(&core, &model, &context, None).await;
        let exhausted = complete(&core, &model, &context, None).await;

        assert_eq!(
            first.content,
            vec![AssistantContent::Text(TextContent::new("first"))]
        );
        assert_eq!(
            second.content,
            vec![AssistantContent::Text(TextContent::new("second"))]
        );
        assert_eq!(exhausted.stop_reason, StopReason::Error);
        assert_eq!(
            exhausted.error_message.as_deref(),
            Some("No more faux responses queued")
        );
        assert_eq!(core.get_pending_response_count(), 0);
        assert_eq!(core.state().call_count(), 3);
    }

    #[tokio::test]
    async fn can_replace_and_append_queued_responses() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![message_of_content("first").into()]);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");

        let response = complete(&core, &model, &context, None).await;
        assert_eq!(
            response.content,
            vec![AssistantContent::Text(TextContent::new("first"))]
        );
        assert_eq!(core.get_pending_response_count(), 0);

        core.set_responses(vec![message_of_content("second").into()]);
        assert_eq!(core.get_pending_response_count(), 1);
        let response = complete(&core, &model, &context, None).await;
        assert_eq!(
            response.content,
            vec![AssistantContent::Text(TextContent::new("second"))]
        );

        core.append_responses(vec![
            message_of_content("third").into(),
            message_of_content("fourth").into(),
        ]);
        assert_eq!(core.get_pending_response_count(), 2);
        let response = complete(&core, &model, &context, None).await;
        assert_eq!(
            response.content,
            vec![AssistantContent::Text(TextContent::new("third"))]
        );
        let response = complete(&core, &model, &context, None).await;
        assert_eq!(
            response.content,
            vec![AssistantContent::Text(TextContent::new("fourth"))]
        );
        assert_eq!(core.get_pending_response_count(), 0);
    }

    #[tokio::test]
    async fn supports_async_response_factories() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![FauxResponseStep::Factory(Arc::new(
            |context: &TranscriptContext,
             _options: Option<&SimpleStreamOptions>,
             state: &FauxProviderState,
             _model: &Model|
             -> BoxFuture<'_, Result<AssistantMessage, ThrownValue>> {
                Box::pin(async move {
                    Ok(faux_assistant_message(
                        format!("{}:{}", context.messages().len(), state.call_count()),
                        FauxMessageOptions::default(),
                    ))
                })
            },
        ))]);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");
        let response = complete(&core, &model, &context, None).await;

        assert_eq!(
            response.content,
            vec![AssistantContent::Text(TextContent::new("1:1"))]
        );
    }

    #[tokio::test]
    async fn emits_an_error_when_a_response_factory_throws() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![FauxResponseStep::failing("boom")]);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");
        let events = collect_events(&core, &model, &context, None).await;

        assert_eq!(events.len(), 1);
        let AssistantMessageEvent::Error { error, reason } = &events[0] else {
            panic!("expected error event, got {:?}", event_types(&events));
        };
        assert_eq!(*reason, StreamErrorReason::Error);
        assert_eq!(error.stop_reason, StopReason::Error);
        assert_eq!(error.error_message.as_deref(), Some("boom"));
    }

    #[tokio::test]
    async fn rejects_a_queued_response_without_a_terminal_stop_reason() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![faux_assistant_message(
            "partial",
            FauxMessageOptions {
                stop_reason: Some(StopReason::Pending),
                ..Default::default()
            },
        )
        .into()]);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");
        let events = collect_events(&core, &model, &context, None).await;

        assert!(!events
            .iter()
            .any(|event| matches!(event, AssistantMessageEvent::Done { .. })));
        let terminal = events.last().expect("terminal event");
        let AssistantMessageEvent::Error { error, .. } = terminal else {
            panic!(
                "expected terminal error event, got {:?}",
                event_types(&events)
            );
        };
        assert_eq!(error.stop_reason, StopReason::Error);
        assert_eq!(
            error.error_message.as_deref(),
            Some("Faux response ended without a stop reason")
        );
    }

    #[tokio::test]
    async fn estimates_prompt_and_output_tokens_from_serialized_context() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![message_of_content("done").into()]);

        let tool = Tool {
            name: "echo".to_string(),
            description: "Echo back text".to_string(),
            parameters: serde_json::json!({ "type": "object", "properties": { "text": { "type": "string" } } }),
            constrained_sampling: None,
        };
        let context = Context {
            system_prompt: Some("sys".to_string()),
            messages: vec![
                Message::User(UserMessage {
                    content: UserContentField::Blocks(vec![
                        crate::types::UserContent::Text(TextContent::new("hello")),
                        crate::types::UserContent::Image(ImageContent::new("abcd", "image/png")),
                    ]),
                    timestamp: 1,
                }),
                Message::Assistant(message_of_content("prior")),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "tool-1".to_string(),
                    tool_name: "echo".to_string(),
                    content: vec![crate::types::UserContent::Text(TextContent::new(
                        "tool out",
                    ))],
                    details: None,
                    usage: None,
                    nested_calls: None,
                    is_error: false,
                    duration_ms: None,
                    timestamp: 2,
                }),
            ],
            tools: Some(vec![tool.clone()]),
        };

        let model = core.first_model().expect("default model");
        let response = complete(&core, &model, &context, None).await;

        let tool_json = serde_json::to_string(&tool).expect("tool serializes");
        let prompt_text = [
            format!("system:sys\ntool+:{tool_json}"),
            "user:hello\n[image:image/png:4]".to_string(),
            "assistant:prior".to_string(),
            "toolResult:echo\ntool out".to_string(),
        ]
        .join("\n\n");
        let expected_prompt_tokens = estimate_tokens(&prompt_text);
        let expected_output_tokens = estimate_tokens("done");

        assert_eq!(response.usage.input, expected_prompt_tokens);
        assert_eq!(response.usage.output, expected_output_tokens);
        assert_eq!(response.usage.cache_read, 0);
        assert_eq!(response.usage.cache_write, 0);
        assert_eq!(
            response.usage.total_tokens,
            expected_prompt_tokens + expected_output_tokens
        );
    }

    fn simple_options(
        session_id: Option<&str>,
        retention: Option<CacheRetention>,
    ) -> SimpleStreamOptions {
        SimpleStreamOptions {
            base: StreamOptions {
                session_id: session_id.map(str::to_string),
                cache_retention: retention,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn does_not_share_cache_across_sessions_or_requests_without_session_id() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![
            message_of_content("first").into(),
            message_of_content("second").into(),
            message_of_content("third").into(),
        ]);

        let mut context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hello", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");

        let first = complete(
            &core,
            &model,
            &context,
            Some(simple_options(
                Some("session-1"),
                Some(CacheRetention::Short),
            )),
        )
        .await;
        assert!(first.usage.cache_write > 0);
        context.messages.push(Message::Assistant(first));
        context.messages.push(faux_user_message("follow up", 2));

        let second = complete(
            &core,
            &model,
            &context,
            Some(simple_options(
                Some("session-2"),
                Some(CacheRetention::Short),
            )),
        )
        .await;
        assert_eq!(second.usage.cache_read, 0);
        assert!(second.usage.cache_write > 0);

        let third = complete(&core, &model, &context, None).await;
        assert_eq!(third.usage.cache_read, 0);
        assert_eq!(third.usage.cache_write, 0);
    }

    #[tokio::test]
    async fn simulates_prompt_caching_per_session_id() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![
            message_of_content("first").into(),
            message_of_content("second").into(),
        ]);

        let mut context = Context {
            system_prompt: Some("Be concise.".to_string()),
            messages: vec![faux_user_message("hello", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");

        let first = complete(
            &core,
            &model,
            &context,
            Some(simple_options(
                Some("session-1"),
                Some(CacheRetention::Short),
            )),
        )
        .await;
        assert_eq!(first.usage.cache_read, 0);
        assert!(first.usage.cache_write > 0);

        context.messages.push(Message::Assistant(first));
        context.messages.push(faux_user_message("follow up", 2));

        let second = complete(
            &core,
            &model,
            &context,
            Some(simple_options(
                Some("session-1"),
                Some(CacheRetention::Short),
            )),
        )
        .await;
        assert!(second.usage.cache_read > 0);
        assert!(second.usage.input + second.usage.cache_read > second.usage.input);
    }

    // Port of test/faux-provider.test.ts "counts cached characters up to the
    // first difference in the joined prompt" (v1.1.0).
    #[tokio::test]
    async fn counts_cached_characters_up_to_the_first_difference_in_the_joined_prompt() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![
            message_of_content("a").into(),
            message_of_content("b").into(),
            message_of_content("c").into(),
        ]);
        let model = core.first_model().expect("default model");
        let user = |content: &str, timestamp: i64| faux_user_message(content, timestamp);

        // Prompt texts: "user:hello world" (16 characters), then 16 + 2 +
        // "user:next" (9) = 27.
        let mut context = Context {
            system_prompt: None,
            messages: vec![user("hello world", 1)],
            tools: None,
        };
        let _first = complete(
            &core,
            &model,
            &context,
            Some(simple_options(
                Some("session-1"),
                Some(CacheRetention::Short),
            )),
        )
        .await;
        context.messages.push(user("next", 2));
        let extended = complete(
            &core,
            &model,
            &context,
            Some(simple_options(
                Some("session-1"),
                Some(CacheRetention::Short),
            )),
        )
        .await;
        assert_eq!(
            (
                extended.usage.input,
                extended.usage.cache_read,
                extended.usage.cache_write
            ),
            (3, 4, 3),
            "unexpected usage {extended:?}"
        );

        // The first message now differs after "user:hello w" (12 characters):
        // "user:hello wide" + 2 + 9 = 26.
        context.messages[0] = user("hello wide", 1);
        let edited = complete(
            &core,
            &model,
            &context,
            Some(simple_options(
                Some("session-1"),
                Some(CacheRetention::Short),
            )),
        )
        .await;
        assert_eq!(
            (
                edited.usage.input,
                edited.usage.cache_read,
                edited.usage.cache_write
            ),
            (4, 3, 4),
            "unexpected usage {edited:?}"
        );
    }

    #[tokio::test]
    async fn does_not_simulate_caching_when_cache_retention_is_none() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![
            message_of_content("first").into(),
            message_of_content("second").into(),
        ]);

        let mut context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hello", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");

        let _first = complete(
            &core,
            &model,
            &context,
            Some(simple_options(
                Some("session-1"),
                Some(CacheRetention::None),
            )),
        )
        .await;
        context
            .messages
            .push(Message::Assistant(message_of_content("first")));
        context.messages.push(faux_user_message("follow up", 2));
        let second = complete(
            &core,
            &model,
            &context,
            Some(simple_options(
                Some("session-1"),
                Some(CacheRetention::None),
            )),
        )
        .await;
        assert_eq!(second.usage.cache_read, 0);
        assert_eq!(second.usage.cache_write, 0);
    }

    #[tokio::test]
    async fn streams_thinking_text_and_partial_tool_call_deltas() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![faux_assistant_message(
            FauxContent::Blocks(vec![
                FauxContentBlock::Thinking(faux_thinking("thinking text")),
                FauxContentBlock::Text(faux_text("answer text")),
                FauxContentBlock::ToolCall(faux_tool_call(
                    "echo",
                    serde_json::json!({ "text": "hi", "count": 12 })
                        .as_object()
                        .expect("object")
                        .clone(),
                    Some("tool-1"),
                )),
            ]),
            FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into()]);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");
        let events = collect_events(&core, &model, &context, None).await;

        let types = event_types(&events);
        assert!(types.contains(&"thinking_start"));
        assert!(types.contains(&"thinking_delta"));
        assert!(types.contains(&"text_start"));
        assert!(types.contains(&"text_delta"));
        assert!(types.contains(&"toolcall_start"));
        assert!(types.contains(&"toolcall_delta"));
        assert!(types.contains(&"toolcall_end"));

        let deltas: String = events
            .iter()
            .filter_map(|event| match event {
                AssistantMessageEvent::ToolcallDelta { delta, .. } => Some(delta.clone()),
                _ => None,
            })
            .collect();
        assert!(deltas.len() > 1);
        let parsed: serde_json::Value =
            serde_json::from_str(&deltas).expect("deltas concatenate to JSON");
        assert_eq!(parsed, serde_json::json!({ "text": "hi", "count": 12 }));
    }

    #[tokio::test]
    async fn streams_an_exact_event_order_for_fixed_size_chunks() {
        let core = FauxCore::new(RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(1),
                max: Some(1),
            }),
            ..Default::default()
        });
        core.set_responses(vec![faux_assistant_message(
            FauxContent::Blocks(vec![
                FauxContentBlock::Thinking(faux_thinking("go")),
                FauxContentBlock::Text(faux_text("ok")),
                FauxContentBlock::ToolCall(faux_tool_call(
                    "echo",
                    Default::default(),
                    Some("tool-1"),
                )),
            ]),
            FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into()]);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");
        let events = collect_events(&core, &model, &context, None).await;

        let AssistantMessageEvent::Start { partial } = &events[0] else {
            panic!("expected start event first, got {:?}", event_types(&events));
        };
        assert_eq!(partial.stop_reason, StopReason::Pending);
        assert_eq!(
            event_types(&events),
            vec![
                "start",
                "thinking_start",
                "thinking_delta",
                "thinking_end",
                "text_start",
                "text_delta",
                "text_end",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_end",
                "done",
            ]
        );
    }

    #[tokio::test]
    async fn streams_multiple_tool_calls_in_one_message() {
        let core = FauxCore::new(RegisterFauxProviderOptions::default());
        core.set_responses(vec![faux_assistant_message(
            FauxContent::Blocks(vec![
                FauxContentBlock::ToolCall(faux_tool_call(
                    "echo",
                    serde_json::json!({ "text": "one" })
                        .as_object()
                        .expect("object")
                        .clone(),
                    Some("tool-1"),
                )),
                FauxContentBlock::ToolCall(faux_tool_call(
                    "echo",
                    serde_json::json!({ "text": "two" })
                        .as_object()
                        .expect("object")
                        .clone(),
                    Some("tool-2"),
                )),
            ]),
            FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into()]);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");
        let events = collect_events(&core, &model, &context, None).await;

        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AssistantMessageEvent::ToolcallStart { .. }))
                .count(),
            2
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AssistantMessageEvent::ToolcallEnd { .. }))
                .count(),
            2
        );
    }

    fn scripted_error_message(error_message: &str, stop_reason: StopReason) -> AssistantMessage {
        let mut message = message_of_content("partial");
        message.stop_reason = stop_reason;
        message.error_message = Some(error_message.to_string());
        message
    }

    #[tokio::test]
    async fn streams_an_explicit_assistant_error_message_as_a_terminal_error() {
        let core = FauxCore::new(RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(2),
                max: Some(2),
            }),
            ..Default::default()
        });
        core.set_responses(vec![scripted_error_message(
            "upstream failed",
            StopReason::Error,
        )
        .into()]);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");
        let events = collect_events(&core, &model, &context, None).await;

        assert_eq!(
            event_types(&events),
            vec!["start", "text_start", "text_delta", "text_end", "error"]
        );
        let AssistantMessageEvent::Error { reason, error } = events.last().expect("terminal")
        else {
            unreachable!("asserted above");
        };
        assert_eq!(*reason, StreamErrorReason::Error);
        assert_eq!(error.stop_reason, StopReason::Error);
        assert_eq!(error.error_message.as_deref(), Some("upstream failed"));
    }

    #[tokio::test]
    async fn streams_an_explicit_assistant_aborted_message_as_a_terminal_error() {
        let core = FauxCore::new(RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(2),
                max: Some(2),
            }),
            ..Default::default()
        });
        core.set_responses(vec![scripted_error_message(
            "Request was aborted",
            StopReason::Aborted,
        )
        .into()]);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");
        let events = collect_events(&core, &model, &context, None).await;

        assert_eq!(
            event_types(&events),
            vec!["start", "text_start", "text_delta", "text_end", "error"]
        );
        let AssistantMessageEvent::Error { reason, error } = events.last().expect("terminal")
        else {
            unreachable!("asserted above");
        };
        assert_eq!(*reason, StreamErrorReason::Aborted);
        assert_eq!(error.stop_reason, StopReason::Aborted);
        assert_eq!(error.error_message.as_deref(), Some("Request was aborted"));
    }

    #[tokio::test]
    async fn supports_aborting_before_the_first_chunk() {
        let core = FauxCore::new(RegisterFauxProviderOptions {
            tokens_per_second: Some(50.0),
            token_size: Some(FauxTokenSize {
                min: Some(3),
                max: Some(3),
            }),
            ..Default::default()
        });
        core.set_responses(vec![message_of_content("abcdefghijklmnopqrstuvwxyz").into()]);

        let controller = AbortController::new();
        controller.abort();
        let options = SimpleStreamOptions {
            base: StreamOptions {
                base: crate::types::ProviderRequestOptions {
                    signal: Some(controller.signal()),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");
        let events = collect_events(&core, &model, &context, Some(options)).await;

        assert_eq!(events.len(), 1);
        let AssistantMessageEvent::Error { reason, error } = &events[0] else {
            panic!("expected error event, got {:?}", event_types(&events));
        };
        assert_eq!(*reason, StreamErrorReason::Aborted);
        assert_eq!(error.stop_reason, StopReason::Aborted);
    }

    #[tokio::test]
    async fn supports_aborting_mid_text_stream_when_paced() {
        let core = FauxCore::new(RegisterFauxProviderOptions {
            tokens_per_second: Some(100.0),
            token_size: Some(FauxTokenSize {
                min: Some(3),
                max: Some(3),
            }),
            ..Default::default()
        });
        core.set_responses(vec![message_of_content("abcdefghijklmnopqrstuvwxyz").into()]);

        let controller = AbortController::new();
        let options = SimpleStreamOptions {
            base: StreamOptions {
                base: crate::types::ProviderRequestOptions {
                    signal: Some(controller.signal()),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");

        let mut events: Vec<AssistantMessageEvent> = Vec::new();
        let mut text_delta_count = 0;
        let mut stream = core.stream(&model, &normalize_context(&context), Some(&options));
        while let Some(event) = stream.next().await {
            let is_text_delta = matches!(event, AssistantMessageEvent::TextDelta { .. });
            events.push(event);
            if is_text_delta {
                text_delta_count += 1;
                controller.abort();
            }
        }

        assert_eq!(text_delta_count, 1);
        let types = event_types(&events);
        assert!(types.contains(&"text_start"));
        assert!(types.contains(&"text_delta"));
        assert!(types.contains(&"error"));
        assert!(!types.contains(&"text_end"));
    }

    #[tokio::test]
    async fn supports_aborting_mid_thinking_stream_when_paced() {
        let core = FauxCore::new(RegisterFauxProviderOptions {
            tokens_per_second: Some(100.0),
            token_size: Some(FauxTokenSize {
                min: Some(3),
                max: Some(3),
            }),
            ..Default::default()
        });
        let mut message = message_of_content("ignored");
        message.content = vec![AssistantContent::Thinking(faux_thinking(
            "abcdefghijklmnopqrstuvwxyz",
        ))];
        core.set_responses(vec![message.into()]);

        let controller = AbortController::new();
        let options = SimpleStreamOptions {
            base: StreamOptions {
                base: crate::types::ProviderRequestOptions {
                    signal: Some(controller.signal()),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");

        let mut thinking_delta_count = 0;
        let mut stream = core.stream(&model, &normalize_context(&context), Some(&options));
        while let Some(event) = stream.next().await {
            let is_thinking_delta = matches!(event, AssistantMessageEvent::ThinkingDelta { .. });
            drop(event);
            if is_thinking_delta {
                thinking_delta_count += 1;
                controller.abort();
            }
        }

        assert_eq!(thinking_delta_count, 1);
    }

    #[tokio::test]
    async fn supports_aborting_mid_toolcall_stream_when_paced() {
        let core = FauxCore::new(RegisterFauxProviderOptions {
            tokens_per_second: Some(100.0),
            token_size: Some(FauxTokenSize {
                min: Some(3),
                max: Some(3),
            }),
            ..Default::default()
        });
        let mut message = message_of_content("done");
        message.content = vec![AssistantContent::ToolCall(ToolCall {
            id: "tool-1".to_string(),
            name: "echo".to_string(),
            arguments:
                serde_json::json!({ "text": "abcdefghijklmnopqrstuvwxyz", "count": 123456789 })
                    .as_object()
                    .expect("object")
                    .clone(),
            thought_signature: None,
            namespace: None,
        })];
        message.stop_reason = StopReason::ToolUse;
        core.set_responses(vec![message.into()]);

        let controller = AbortController::new();
        let options = SimpleStreamOptions {
            base: StreamOptions {
                base: crate::types::ProviderRequestOptions {
                    signal: Some(controller.signal()),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = core.first_model().expect("default model");

        let mut toolcall_delta_count = 0;
        let mut stream = core.stream(&model, &normalize_context(&context), Some(&options));
        while let Some(event) = stream.next().await {
            let is_toolcall_delta = matches!(event, AssistantMessageEvent::ToolcallDelta { .. });
            drop(event);
            if is_toolcall_delta {
                toolcall_delta_count += 1;
                controller.abort();
            }
        }

        assert_eq!(toolcall_delta_count, 1);
    }

    #[tokio::test]
    async fn builds_a_provider_through_create_provider() {
        let handle = faux_provider(RegisterFauxProviderOptions::default());
        assert_eq!(handle.provider.id(), "faux");
        assert_eq!(handle.api().starts_with("faux"), true);
        assert_eq!(handle.models().len(), 1);
        handle.set_responses(vec![message_of_content("hello").into()]);

        let context = Context {
            system_prompt: None,
            messages: vec![faux_user_message("hi", 1)],
            tools: None,
        };
        let model = handle.first_model().expect("default model");
        let response = complete(&handle.core, &model, &context, None).await;
        assert_eq!(
            response.content,
            vec![AssistantContent::Text(TextContent::new("hello"))]
        );
        assert_eq!(handle.provider.get_models()[0].id(), model.id());
    }
}
