//! Port of `packages/ai/src/utils/assistant-message-frame.ts`.
//!
//! Compact, replayable assistant-message progress. Terminal settlement is
//! intentionally excluded and must be persisted separately.
//! [`AssistantMessageFrameEncoder`] turns an assistant event stream into
//! frames; [`reduce_assistant_message_frames`] replays frames into the final
//! [`AssistantMessage`].
//!
//! Deviations (crate docs ledger):
//! - **D19 — the frame union is a serde-tagged enum** (`type` discriminator,
//!   camelCase fields, optional metadata omitted when absent) and encoder /
//!   reducer misuse errors are `Result<_, FrameError>` carrying the exact
//!   upstream error strings (upstream throws).
//! - **D19a — string offsets are Unicode scalar values** (`char` counts)
//!   where upstream counts UTF-16 code units; identical for BMP stream text.
//! - **D19b — the `contentIndex` safety-integer check is static** (`u32`
//!   cannot be negative or unsafe), so the `Invalid assistant message frame
//!   contentIndex` throw is unreachable and dropped.
//! - **D19c — block-kind mismatch errors inside frame variants are
//!   unreachable** (Rust variants are typed); the encoder-side checks against
//!   the live `partial` remain.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, StopReason, TextContent,
    ThinkingContent, ToolCall,
};
use crate::utils::json_parse::parse_streaming_json;

/// Upstream `AssistantMessageFrame` union.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AssistantMessageFrame {
    #[serde(rename = "start")]
    Start { partial: AssistantMessage },
    #[serde(rename = "text_start")]
    TextStart {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        content: TextContent,
    },
    #[serde(rename = "text_delta")]
    TextDelta {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "text_end")]
    TextEnd {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        content: String,
        #[serde(
            rename = "textSignature",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        text_signature: Option<String>,
    },
    #[serde(rename = "thinking_start")]
    ThinkingStart {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        content: ThinkingContent,
    },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "thinking_end")]
    ThinkingEnd {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        content: String,
        #[serde(
            rename = "thinkingSignature",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        thinking_signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        redacted: Option<bool>,
    },
    #[serde(rename = "toolcall_start")]
    ToolcallStart {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        #[serde(rename = "toolCall")]
        tool_call: ToolCall,
    },
    #[serde(rename = "toolcall_checkpoint")]
    ToolcallCheckpoint {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        json: String,
    },
    #[serde(rename = "toolcall_delta")]
    ToolcallDelta {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "toolcall_end")]
    ToolcallEnd {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        id: String,
        name: String,
        arguments: serde_json::Map<String, serde_json::Value>,
        #[serde(
            rename = "thoughtSignature",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        thought_signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
    },
}

impl AssistantMessageFrame {
    /// Upstream `frame.type`.
    pub fn frame_type(&self) -> &'static str {
        match self {
            AssistantMessageFrame::Start { .. } => "start",
            AssistantMessageFrame::TextStart { .. } => "text_start",
            AssistantMessageFrame::TextDelta { .. } => "text_delta",
            AssistantMessageFrame::TextEnd { .. } => "text_end",
            AssistantMessageFrame::ThinkingStart { .. } => "thinking_start",
            AssistantMessageFrame::ThinkingDelta { .. } => "thinking_delta",
            AssistantMessageFrame::ThinkingEnd { .. } => "thinking_end",
            AssistantMessageFrame::ToolcallStart { .. } => "toolcall_start",
            AssistantMessageFrame::ToolcallCheckpoint { .. } => "toolcall_checkpoint",
            AssistantMessageFrame::ToolcallDelta { .. } => "toolcall_delta",
            AssistantMessageFrame::ToolcallEnd { .. } => "toolcall_end",
        }
    }
}

/// Encoder/reducer misuse error; the message is the exact upstream `Error`
/// text (deviation D19).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct FrameError(pub String);

/// Per-block encoder bookkeeping (upstream `EncoderBlockState`).
#[derive(Debug, Clone, PartialEq)]
enum EncoderBlockState {
    Text {
        covered_chars: usize,
        delta_chars: usize,
    },
    Thinking {
        covered_chars: usize,
        delta_chars: usize,
    },
    ToolCall {
        caught_up: bool,
        catchup_json: String,
        snapshot_arguments: String,
    },
}

impl EncoderBlockState {
    fn kind(&self) -> BlockKind {
        match self {
            EncoderBlockState::Text { .. } => BlockKind::Text,
            EncoderBlockState::Thinking { .. } => BlockKind::Thinking,
            EncoderBlockState::ToolCall { .. } => BlockKind::ToolCall,
        }
    }
}

/// Per-block reducer bookkeeping (upstream `ReducerBlockState`).
#[derive(Debug, Clone)]
struct ReducerBlockState {
    kind: BlockKind,
    ended: bool,
    json: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    ToolCall,
}

impl BlockKind {
    fn as_str(self) -> &'static str {
        match self {
            BlockKind::Text => "text",
            BlockKind::Thinking => "thinking",
            BlockKind::ToolCall => "toolCall",
        }
    }
}

fn event_type(event: &AssistantMessageEvent) -> &'static str {
    match event {
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
    }
}

fn clone_text_content(content: &TextContent) -> TextContent {
    TextContent {
        text: content.text.clone(),
        text_signature: content.text_signature.clone(),
    }
}

fn clone_thinking_content(content: &ThinkingContent) -> ThinkingContent {
    ThinkingContent {
        thinking: content.thinking.clone(),
        thinking_signature: content.thinking_signature.clone(),
        redacted: content.redacted.clone(),
    }
}

fn clone_tool_call(tool_call: &ToolCall) -> ToolCall {
    ToolCall {
        id: tool_call.id.clone(),
        name: tool_call.name.clone(),
        arguments: tool_call.arguments.clone(),
        thought_signature: tool_call.thought_signature.clone(),
        namespace: tool_call.namespace.clone(),
    }
}

fn clone_start_message(message: &AssistantMessage) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: message.api.clone(),
        provider: message.provider.clone(),
        model: message.model.clone(),
        response_model: message.response_model.clone(),
        response_id: message.response_id.clone(),
        provider_thinking_level: message.provider_thinking_level.clone(),
        thinking_level: None,
        diagnostics: message.diagnostics.clone(),
        usage: message.usage.clone(),
        stop_reason: StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: message.timestamp,
        duration_ms: None,
    }
}

fn block_kind(block: &AssistantContent) -> BlockKind {
    match block {
        AssistantContent::Text(_) => BlockKind::Text,
        AssistantContent::Thinking(_) => BlockKind::Thinking,
        AssistantContent::ToolCall(_) => BlockKind::ToolCall,
    }
}

fn event_content_block<'a>(
    partial: &'a AssistantMessage,
    event_type: &str,
    content_index: u32,
) -> Result<&'a AssistantContent, FrameError> {
    // The upstream `assertContentIndex` safety-integer check is static for `u32`
    // (deviation D19b).
    partial.content.get(content_index as usize).ok_or_else(|| {
        FrameError(format!(
            "{} event has no content block at index {}",
            event_type, content_index
        ))
    })
}

/// `JSON.stringify(arguments)`; serde JSON values always serialize, so the
/// upstream "not JSON-serializable" throw is unreachable (deviation D19c).
fn serialized_arguments_value(value: &serde_json::Value) -> String {
    serde_json::to_string(value).expect("JSON values serialize")
}

fn serialized_arguments(arguments: &serde_json::Map<String, serde_json::Value>) -> String {
    serialized_arguments_value(&serde_json::Value::Object(arguments.clone()))
}

static EMPTY_PARSED_TOOL_ARGUMENTS: LazyLock<String> =
    LazyLock::new(|| serialized_arguments_value(&parse_streaming_json(Some(""))));

fn char_len(text: &str) -> usize {
    text.chars().count()
}

/// Upstream `isJsonPrefix`: whether `snapshot` is a parsed prefix of `current`.
fn is_json_prefix(snapshot: &serde_json::Value, current: &serde_json::Value) -> bool {
    match snapshot {
        serde_json::Value::String(snapshot_text) => current
            .as_str()
            .map(|current_text| current_text.starts_with(snapshot_text.as_str()))
            .unwrap_or(false),
        serde_json::Value::Array(snapshot_items) => match current.as_array() {
            Some(current_items) if snapshot_items.len() <= current_items.len() => snapshot_items
                .iter()
                .zip(current_items.iter())
                .all(|(snapshot_item, current_item)| is_json_prefix(snapshot_item, current_item)),
            _ => false,
        },
        serde_json::Value::Object(snapshot_fields) => match current.as_object() {
            Some(current_fields) => snapshot_fields.iter().all(|(key, value)| {
                current_fields
                    .get(key)
                    .map(|current| is_json_prefix(value, current))
                    .unwrap_or(false)
            }),
            None => false,
        },
        other => other == current,
    }
}

/// Encodes one assistant stream. `partial` remains a shared live accumulator;
/// the encoder uses per-block offsets to avoid replaying deltas already visible
/// when an older queued event is consumed.
///
/// Upstream `AssistantMessageFrameEncoder`.
#[derive(Debug, Default)]
pub struct AssistantMessageFrameEncoder {
    started: bool,
    terminal: bool,
    blocks: BTreeMap<u32, EncoderBlockState>,
}

impl AssistantMessageFrameEncoder {
    pub fn new() -> AssistantMessageFrameEncoder {
        AssistantMessageFrameEncoder::default()
    }

    /// Encode one stream event; `Ok(None)` when the event produces no frame.
    pub fn encode(
        &mut self,
        event: &AssistantMessageEvent,
    ) -> Result<Option<AssistantMessageFrame>, FrameError> {
        if self.terminal {
            return Err(FrameError(format!(
                "Assistant message event {} follows a terminal event",
                event_type(event)
            )));
        }

        match event {
            AssistantMessageEvent::Start { partial } => {
                if self.started {
                    return Err(FrameError(
                        "Assistant message stream contains more than one start event".into(),
                    ));
                }
                self.started = true;
                return Ok(Some(AssistantMessageFrame::Start {
                    partial: clone_start_message(partial),
                }));
            }
            AssistantMessageEvent::Done { .. } => {
                if !self.started {
                    return Err(FrameError(
                        "Assistant message done event appears before start".into(),
                    ));
                }
                self.terminal = true;
                return Ok(None);
            }
            AssistantMessageEvent::Error { .. } => {
                self.terminal = true;
                return Ok(None);
            }
            _ => {}
        }

        if !self.started {
            return Err(FrameError(format!(
                "Assistant message {} event appears before start",
                event_type(event)
            )));
        }

        match event {
            AssistantMessageEvent::TextStart {
                content_index,
                partial,
            } => {
                let content = event_content_block(partial, "text_start", *content_index)?;
                let AssistantContent::Text(content) = content else {
                    return Err(FrameError(format!(
                        "text_start event points to {} block at index {}",
                        block_kind(content).as_str(),
                        content_index
                    )));
                };
                self.start_block(
                    *content_index,
                    EncoderBlockState::Text {
                        covered_chars: char_len(&content.text),
                        delta_chars: 0,
                    },
                )?;
                Ok(Some(AssistantMessageFrame::TextStart {
                    content_index: *content_index,
                    content: clone_text_content(content),
                }))
            }
            AssistantMessageEvent::TextDelta {
                content_index,
                delta,
                ..
            } => self.encode_text_delta(*content_index, delta, BlockKind::Text),
            AssistantMessageEvent::TextEnd {
                content_index,
                content,
                partial,
            } => {
                let block = event_content_block(partial, "text_end", *content_index)?;
                let AssistantContent::Text(text_content) = block else {
                    return Err(FrameError(format!(
                        "text_end event points to {} block at index {}",
                        block_kind(block).as_str(),
                        content_index
                    )));
                };
                self.end_block(*content_index, BlockKind::Text)?;
                Ok(Some(AssistantMessageFrame::TextEnd {
                    content_index: *content_index,
                    content: content.clone(),
                    text_signature: text_content.text_signature.clone(),
                }))
            }
            AssistantMessageEvent::ThinkingStart {
                content_index,
                partial,
            } => {
                let content = event_content_block(partial, "thinking_start", *content_index)?;
                let AssistantContent::Thinking(content) = content else {
                    return Err(FrameError(format!(
                        "thinking_start event points to {} block at index {}",
                        block_kind(content).as_str(),
                        content_index
                    )));
                };
                self.start_block(
                    *content_index,
                    EncoderBlockState::Thinking {
                        covered_chars: char_len(&content.thinking),
                        delta_chars: 0,
                    },
                )?;
                Ok(Some(AssistantMessageFrame::ThinkingStart {
                    content_index: *content_index,
                    content: clone_thinking_content(content),
                }))
            }
            AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
                ..
            } => self.encode_text_delta(*content_index, delta, BlockKind::Thinking),
            AssistantMessageEvent::ThinkingEnd {
                content_index,
                content,
                partial,
            } => {
                let block = event_content_block(partial, "thinking_end", *content_index)?;
                let AssistantContent::Thinking(thinking_content) = block else {
                    return Err(FrameError(format!(
                        "thinking_end event points to {} block at index {}",
                        block_kind(block).as_str(),
                        content_index
                    )));
                };
                self.end_block(*content_index, BlockKind::Thinking)?;
                Ok(Some(AssistantMessageFrame::ThinkingEnd {
                    content_index: *content_index,
                    content: content.clone(),
                    thinking_signature: thinking_content.thinking_signature.clone(),
                    redacted: thinking_content.redacted,
                }))
            }
            AssistantMessageEvent::ToolcallStart {
                content_index,
                partial,
            } => {
                let content = event_content_block(partial, "toolcall_start", *content_index)?;
                let AssistantContent::ToolCall(content) = content else {
                    return Err(FrameError(format!(
                        "toolcall_start event points to {} block at index {}",
                        block_kind(content).as_str(),
                        content_index
                    )));
                };
                let snapshot_arguments = serialized_arguments(&content.arguments);
                let caught_up = snapshot_arguments == EMPTY_PARSED_TOOL_ARGUMENTS.as_str();
                self.start_block(
                    *content_index,
                    EncoderBlockState::ToolCall {
                        caught_up,
                        catchup_json: String::new(),
                        snapshot_arguments: if caught_up {
                            String::new()
                        } else {
                            snapshot_arguments
                        },
                    },
                )?;
                Ok(Some(AssistantMessageFrame::ToolcallStart {
                    content_index: *content_index,
                    tool_call: clone_tool_call(content),
                }))
            }
            AssistantMessageEvent::ToolcallDelta {
                content_index,
                delta,
                ..
            } => {
                let state = self.block_mut(*content_index, BlockKind::ToolCall)?;
                let EncoderBlockState::ToolCall {
                    caught_up,
                    catchup_json,
                    snapshot_arguments,
                } = state
                else {
                    return Err(FrameError("Unreachable tool-call encoder state".into()));
                };
                if *caught_up {
                    return Ok(if delta.is_empty() {
                        None
                    } else {
                        Some(AssistantMessageFrame::ToolcallDelta {
                            content_index: *content_index,
                            delta: delta.clone(),
                        })
                    });
                }
                catchup_json.push_str(delta);
                let arguments_value = parse_streaming_json(Some(catchup_json.as_str()));
                if serialized_arguments_value(&arguments_value) != snapshot_arguments.as_str() {
                    // Legacy grammar calls include the initial input in toolcall_start, but their
                    // JSON delta stream still begins at an empty input. Its parsed arguments can
                    // therefore extend, rather than exactly reproduce, the start snapshot.
                    let snapshot_value = parse_streaming_json(Some(snapshot_arguments.as_str()));
                    if !is_json_prefix(&snapshot_value, &arguments_value) {
                        return Ok(None);
                    }
                }
                *caught_up = true;
                *snapshot_arguments = String::new();
                let json = std::mem::take(catchup_json);
                Ok(if json.is_empty() {
                    None
                } else {
                    Some(AssistantMessageFrame::ToolcallCheckpoint {
                        content_index: *content_index,
                        json,
                    })
                })
            }
            AssistantMessageEvent::ToolcallEnd {
                content_index,
                tool_call,
                partial,
            } => {
                let content = event_content_block(partial, "toolcall_end", *content_index)?;
                if block_kind(content) != BlockKind::ToolCall {
                    return Err(FrameError(format!(
                        "toolcall_end event points to {} block at index {}",
                        block_kind(content).as_str(),
                        content_index
                    )));
                }
                self.end_block(*content_index, BlockKind::ToolCall)?;
                Ok(Some(AssistantMessageFrame::ToolcallEnd {
                    content_index: *content_index,
                    id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                    thought_signature: tool_call.thought_signature.clone(),
                    namespace: tool_call.namespace.clone(),
                }))
            }
            AssistantMessageEvent::Start { .. }
            | AssistantMessageEvent::Done { .. }
            | AssistantMessageEvent::Error { .. } => {
                Err(FrameError("Unreachable terminal event in encoder".into()))
            }
        }
    }

    fn start_block(
        &mut self,
        content_index: u32,
        state: EncoderBlockState,
    ) -> Result<(), FrameError> {
        if self.blocks.contains_key(&content_index) {
            return Err(FrameError(format!(
                "Assistant message block {} starts more than once",
                content_index
            )));
        }
        self.blocks.insert(content_index, state);
        Ok(())
    }

    fn block_mut(
        &mut self,
        content_index: u32,
        kind: BlockKind,
    ) -> Result<&mut EncoderBlockState, FrameError> {
        let Some(state) = self.blocks.get_mut(&content_index) else {
            return Err(FrameError(format!(
                "Assistant message {} block {} has not started",
                kind.as_str(),
                content_index
            )));
        };
        if state.kind() != kind {
            return Err(FrameError(format!(
                "Assistant message block {} is {}, not {}",
                content_index,
                state.kind().as_str(),
                kind.as_str()
            )));
        }
        Ok(state)
    }

    fn end_block(&mut self, content_index: u32, kind: BlockKind) -> Result<(), FrameError> {
        self.block_mut(content_index, kind)?;
        self.blocks.remove(&content_index);
        Ok(())
    }

    fn encode_text_delta(
        &mut self,
        content_index: u32,
        delta: &str,
        kind: BlockKind,
    ) -> Result<Option<AssistantMessageFrame>, FrameError> {
        let state = self.block_mut(content_index, kind)?;
        let (covered_chars, delta_chars) = match state {
            EncoderBlockState::Text {
                covered_chars,
                delta_chars,
            }
            | EncoderBlockState::Thinking {
                covered_chars,
                delta_chars,
            } => (covered_chars, delta_chars),
            EncoderBlockState::ToolCall { .. } => {
                return Err(FrameError("Unreachable text encoder state".into()))
            }
        };
        let delta_start = *delta_chars;
        *delta_chars += char_len(delta);
        let covered = covered_chars.saturating_sub(delta_start);
        if covered >= char_len(delta) {
            return Ok(None);
        }
        let uncovered = if covered == 0 {
            delta.to_string()
        } else {
            delta.chars().skip(covered).collect()
        };
        Ok(Some(match kind {
            BlockKind::Text => AssistantMessageFrame::TextDelta {
                content_index,
                delta: uncovered,
            },
            BlockKind::Thinking => AssistantMessageFrame::ThinkingDelta {
                content_index,
                delta: uncovered,
            },
            BlockKind::ToolCall => return Err(FrameError("Unreachable text encoder state".into())),
        }))
    }
}

fn append_block(
    message: &mut AssistantMessage,
    states: &mut BTreeMap<u32, ReducerBlockState>,
    content_index: u32,
    block: AssistantContent,
    state: ReducerBlockState,
) -> Result<(), FrameError> {
    if content_index as usize != message.content.len() {
        let reason = if (content_index as usize) < message.content.len() {
            "already exists"
        } else {
            "would leave a gap"
        };
        return Err(FrameError(format!(
            "Cannot start assistant message block at index {}: {}",
            content_index, reason
        )));
    }
    message.content.push(block);
    states.insert(content_index, state);
    Ok(())
}

fn active_block<'a>(
    message: &'a mut AssistantMessage,
    states: &'a mut BTreeMap<u32, ReducerBlockState>,
    content_index: u32,
    expected_kind: BlockKind,
    frame_type: &str,
) -> Result<(&'a mut AssistantContent, &'a mut ReducerBlockState), FrameError> {
    let state = states.get_mut(&content_index);
    let block = message.content.get_mut(content_index as usize);
    let (Some(block), Some(state)) = (block, state) else {
        return Err(FrameError(format!(
            "{} frame has no started block at index {}",
            frame_type, content_index
        )));
    };
    if state.kind != expected_kind || block_kind(block) != expected_kind {
        return Err(FrameError(format!(
            "{} frame expected {} block at index {}, found {}",
            frame_type,
            expected_kind.as_str(),
            content_index,
            block_kind(block).as_str()
        )));
    }
    if state.ended {
        return Err(FrameError(format!(
            "{} frame follows the end of block at index {}",
            frame_type, content_index
        )));
    }
    Ok((block, state))
}

/// Replay compact frames without mutating them. Returns `Ok(None)` when the
/// iterator contains no start frame.
///
/// Upstream `reduceAssistantMessageFrames`.
pub fn reduce_assistant_message_frames<I>(frames: I) -> Result<Option<AssistantMessage>, FrameError>
where
    I: IntoIterator<Item = AssistantMessageFrame>,
{
    let mut message: Option<AssistantMessage> = None;
    let mut frame_before_start: Option<&'static str> = None;
    let mut states: BTreeMap<u32, ReducerBlockState> = BTreeMap::new();

    for frame in frames {
        if let AssistantMessageFrame::Start { partial } = frame {
            if message.is_some() {
                return Err(FrameError(
                    "Assistant message frame sequence contains more than one start frame".into(),
                ));
            }
            if let Some(before) = frame_before_start {
                return Err(FrameError(format!(
                    "{} frame appears before the start frame",
                    before
                )));
            }
            message = Some(partial);
            continue;
        }
        let Some(current) = message.as_mut() else {
            if frame_before_start.is_none() {
                frame_before_start = Some(frame.frame_type());
            }
            continue;
        };

        match frame {
            AssistantMessageFrame::Start { .. } => unreachable!("handled above"),
            AssistantMessageFrame::TextStart {
                content_index,
                content,
            } => {
                append_block(
                    current,
                    &mut states,
                    content_index,
                    AssistantContent::Text(content),
                    ReducerBlockState {
                        kind: BlockKind::Text,
                        ended: false,
                        json: String::new(),
                    },
                )?;
            }
            AssistantMessageFrame::TextDelta {
                content_index,
                delta,
            } => {
                let (block, _) = active_block(
                    current,
                    &mut states,
                    content_index,
                    BlockKind::Text,
                    "text_delta",
                )?;
                let AssistantContent::Text(text) = block else {
                    return Err(FrameError("Unreachable text frame state".into()));
                };
                text.text.push_str(&delta);
            }
            AssistantMessageFrame::TextEnd {
                content_index,
                content,
                text_signature,
            } => {
                let (block, state) = active_block(
                    current,
                    &mut states,
                    content_index,
                    BlockKind::Text,
                    "text_end",
                )?;
                let AssistantContent::Text(text) = block else {
                    return Err(FrameError("Unreachable text frame state".into()));
                };
                text.text = content;
                text.text_signature = text_signature;
                state.ended = true;
            }
            AssistantMessageFrame::ThinkingStart {
                content_index,
                content,
            } => {
                append_block(
                    current,
                    &mut states,
                    content_index,
                    AssistantContent::Thinking(content),
                    ReducerBlockState {
                        kind: BlockKind::Thinking,
                        ended: false,
                        json: String::new(),
                    },
                )?;
            }
            AssistantMessageFrame::ThinkingDelta {
                content_index,
                delta,
            } => {
                let (block, _) = active_block(
                    current,
                    &mut states,
                    content_index,
                    BlockKind::Thinking,
                    "thinking_delta",
                )?;
                let AssistantContent::Thinking(thinking) = block else {
                    return Err(FrameError("Unreachable thinking frame state".into()));
                };
                thinking.thinking.push_str(&delta);
            }
            AssistantMessageFrame::ThinkingEnd {
                content_index,
                content,
                thinking_signature,
                redacted,
            } => {
                let (block, state) = active_block(
                    current,
                    &mut states,
                    content_index,
                    BlockKind::Thinking,
                    "thinking_end",
                )?;
                let AssistantContent::Thinking(thinking) = block else {
                    return Err(FrameError("Unreachable thinking frame state".into()));
                };
                thinking.thinking = content;
                thinking.thinking_signature = thinking_signature;
                thinking.redacted = redacted;
                state.ended = true;
            }
            AssistantMessageFrame::ToolcallStart {
                content_index,
                tool_call,
            } => {
                append_block(
                    current,
                    &mut states,
                    content_index,
                    AssistantContent::ToolCall(tool_call),
                    ReducerBlockState {
                        kind: BlockKind::ToolCall,
                        ended: false,
                        json: String::new(),
                    },
                )?;
            }
            AssistantMessageFrame::ToolcallCheckpoint {
                content_index,
                json,
            } => {
                let (block, state) = active_block(
                    current,
                    &mut states,
                    content_index,
                    BlockKind::ToolCall,
                    "toolcall_checkpoint",
                )?;
                let AssistantContent::ToolCall(tool_call) = block else {
                    return Err(FrameError("Unreachable tool-call checkpoint state".into()));
                };
                state.json = json.clone();
                tool_call.arguments = parsed_arguments(&json);
            }
            AssistantMessageFrame::ToolcallDelta {
                content_index,
                delta,
            } => {
                let (block, state) = active_block(
                    current,
                    &mut states,
                    content_index,
                    BlockKind::ToolCall,
                    "toolcall_delta",
                )?;
                let AssistantContent::ToolCall(_) = block else {
                    return Err(FrameError("Unreachable tool-call frame state".into()));
                };
                state.json.push_str(&delta);
            }
            AssistantMessageFrame::ToolcallEnd {
                content_index,
                id,
                name,
                arguments,
                thought_signature,
                namespace,
            } => {
                let (block, state) = active_block(
                    current,
                    &mut states,
                    content_index,
                    BlockKind::ToolCall,
                    "toolcall_end",
                )?;
                let AssistantContent::ToolCall(tool_call) = block else {
                    return Err(FrameError("Unreachable tool-call frame state".into()));
                };
                tool_call.id = id;
                tool_call.name = name;
                tool_call.arguments = arguments;
                tool_call.thought_signature = thought_signature;
                tool_call.namespace = namespace;
                state.ended = true;
            }
        }
    }

    let Some(mut message) = message else {
        return Ok(None);
    };
    for (content_index, state) in &states {
        if state.kind != BlockKind::ToolCall || state.ended || state.json.is_empty() {
            continue;
        }
        let block = message
            .content
            .get_mut(*content_index as usize)
            .ok_or_else(|| FrameError("Unreachable tool-call frame state".into()))?;
        let AssistantContent::ToolCall(tool_call) = block else {
            return Err(FrameError("Unreachable tool-call frame state".into()));
        };
        tool_call.arguments = parsed_arguments(&state.json);
    }

    Ok(Some(message))
}

fn parsed_arguments(json: &str) -> serde_json::Map<String, serde_json::Value> {
    match parse_streaming_json(Some(json)) {
        serde_json::Value::Object(arguments) => arguments,
        _ => serde_json::Map::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AssistantMessageDiagnostic, StreamDoneReason, StreamErrorReason, Usage, UsageCost,
    };
    use serde_json::json;

    fn usage() -> Usage {
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

    fn seed() -> AssistantMessage {
        AssistantMessage {
            content: Vec::new(),
            api: "test-api".to_string(),
            provider: "test-provider".to_string(),
            model: "test-model".to_string(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            thinking_level: None,
            diagnostics: None,
            usage: usage(),
            stop_reason: StopReason::Pending,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 1,
            duration_ms: None,
        }
    }

    fn frame(
        encoder: &mut AssistantMessageFrameEncoder,
        event: AssistantMessageEvent,
    ) -> AssistantMessageFrame {
        encoder
            .encode(&event)
            .expect("encoding succeeds")
            .expect("expected event to produce a frame")
    }

    fn encoded(
        encoder: &mut AssistantMessageFrameEncoder,
        event: AssistantMessageEvent,
    ) -> Option<AssistantMessageFrame> {
        encoder.encode(&event).expect("encoding succeeds")
    }

    fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: arguments.as_object().cloned().expect("object arguments"),
            thought_signature: None,
            namespace: None,
        }
    }

    fn start_frame(partial: AssistantMessage) -> AssistantMessageFrame {
        AssistantMessageFrame::Start { partial }
    }

    fn expect_error(result: Result<Option<AssistantMessage>, FrameError>, needle: &str) {
        let error = result.expect_err("expected a frame error");
        assert!(
            error.to_string().contains(needle),
            "`{error}` should contain `{needle}`"
        );
    }

    fn expect_encode_error(
        result: Result<Option<AssistantMessageFrame>, FrameError>,
        needle: &str,
    ) {
        let error = result.expect_err("expected an encode error");
        assert!(
            error.to_string().contains(needle),
            "`{error}` should contain `{needle}`"
        );
    }

    #[test]
    fn uses_authoritative_text_end_content_and_signature() {
        let mut partial = seed();
        let mut encoder = AssistantMessageFrameEncoder::new();
        let mut frames = vec![frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        )];
        partial
            .content
            .push(AssistantContent::Text(TextContent::new("Hello ")));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::TextStart {
                content_index: 0,
                partial: partial.clone(),
            },
        ));
        partial.content[0] = AssistantContent::Text(TextContent {
            text: "Hello world".to_string(),
            text_signature: Some("sig-text".to_string()),
        });
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "incorrect".to_string(),
                partial: partial.clone(),
            },
        ));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::TextEnd {
                content_index: 0,
                content: "Hello world".to_string(),
                partial,
            },
        ));

        assert_eq!(
            frames.last(),
            Some(&AssistantMessageFrame::TextEnd {
                content_index: 0,
                content: "Hello world".to_string(),
                text_signature: Some("sig-text".to_string()),
            })
        );
        let reduced = reduce_assistant_message_frames(frames)
            .expect("reduces")
            .expect("message");
        assert_eq!(
            reduced.content,
            vec![AssistantContent::Text(TextContent {
                text: "Hello world".to_string(),
                text_signature: Some("sig-text".to_string()),
            })]
        );
    }

    #[test]
    fn preserves_provider_thinking_level_from_the_stream_start() {
        let mut partial = seed();
        partial.provider_thinking_level = Some("high".to_string());
        let mut encoder = AssistantMessageFrameEncoder::new();
        let start = frame(&mut encoder, AssistantMessageEvent::Start { partial });

        let AssistantMessageFrame::Start { partial } = &start else {
            panic!("expected a start frame");
        };
        assert_eq!(partial.provider_thinking_level.as_deref(), Some("high"));
        let reduced = reduce_assistant_message_frames(vec![start])
            .expect("reduces")
            .expect("message");
        assert_eq!(reduced.provider_thinking_level.as_deref(), Some("high"));
    }

    #[test]
    fn preserves_initial_and_final_thinking_metadata_including_redaction() {
        let mut partial = seed();
        let mut encoder = AssistantMessageFrameEncoder::new();
        let mut frames = vec![frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        )];
        partial
            .content
            .push(AssistantContent::Thinking(ThinkingContent {
                thinking: "[redacted]".to_string(),
                thinking_signature: Some("encrypted-start".to_string()),
                redacted: Some(true),
            }));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ThinkingStart {
                content_index: 0,
                partial: partial.clone(),
            },
        ));
        partial.content[0] = AssistantContent::Thinking(ThinkingContent {
            thinking: "[redacted]".to_string(),
            thinking_signature: Some("encrypted-final".to_string()),
            redacted: Some(true),
        });
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ThinkingEnd {
                content_index: 0,
                content: "[redacted]".to_string(),
                partial,
            },
        ));

        assert_eq!(
            frames.last(),
            Some(&AssistantMessageFrame::ThinkingEnd {
                content_index: 0,
                content: "[redacted]".to_string(),
                thinking_signature: Some("encrypted-final".to_string()),
                redacted: Some(true),
            })
        );
        let reduced = reduce_assistant_message_frames(frames)
            .expect("reduces")
            .expect("message");
        assert_eq!(
            reduced.content[0],
            AssistantContent::Thinking(ThinkingContent {
                thinking: "[redacted]".to_string(),
                thinking_signature: Some("encrypted-final".to_string()),
                redacted: Some(true),
            })
        );
    }

    #[test]
    fn parses_unfinished_tool_json_once_and_uses_authoritative_completed_arguments() {
        let initial_frames: Vec<AssistantMessageFrame> = vec![
            start_frame(seed()),
            AssistantMessageFrame::ToolcallStart {
                content_index: 0,
                tool_call: tool_call("initial-id", "write", json!({})),
            },
            AssistantMessageFrame::ToolcallDelta {
                content_index: 0,
                delta: r#"{"path":"READ"#.to_string(),
            },
        ];

        let reduced = reduce_assistant_message_frames(initial_frames.clone())
            .expect("reduces")
            .expect("message");
        assert_eq!(
            reduced.content[0],
            AssistantContent::ToolCall(tool_call("initial-id", "write", json!({"path": "READ"})))
        );

        let mut complete_frames = initial_frames;
        complete_frames.push(AssistantMessageFrame::ToolcallDelta {
            content_index: 0,
            delta: r#"ME.md","lines":[1,2]}"#.to_string(),
        });
        complete_frames.push(AssistantMessageFrame::ToolcallEnd {
            content_index: 0,
            id: "final-id".to_string(),
            name: "write_file".to_string(),
            arguments: json!({"path": "final.md", "lines": [3]})
                .as_object()
                .cloned()
                .expect("object"),
            thought_signature: Some("thought".to_string()),
            namespace: Some("files".to_string()),
        });
        let reduced = reduce_assistant_message_frames(complete_frames)
            .expect("reduces")
            .expect("message");
        let mut expected = tool_call(
            "final-id",
            "write_file",
            json!({"path": "final.md", "lines": [3]}),
        );
        expected.thought_signature = Some("thought".to_string());
        expected.namespace = Some("files".to_string());
        assert_eq!(reduced.content[0], AssistantContent::ToolCall(expected));
    }

    #[test]
    fn reconciles_queued_text_events_against_one_advanced_live_partial_without_duplicate_content() {
        let mut partial = seed();
        // Upstream events hold the live `partial` by reference and are encoded only
        // after all deltas landed, so every event observes the final snapshot here.
        partial
            .content
            .push(AssistantContent::Text(TextContent::new("Hello world")));
        let events: Vec<AssistantMessageEvent> = vec![
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
            AssistantMessageEvent::TextStart {
                content_index: 0,
                partial: partial.clone(),
            },
            AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "Hel".to_string(),
                partial: partial.clone(),
            },
            AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "lo".to_string(),
                partial: partial.clone(),
            },
            AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: " ".to_string(),
                partial: partial.clone(),
            },
            AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "world".to_string(),
                partial,
            },
        ];

        let mut encoder = AssistantMessageFrameEncoder::new();
        let frames: Vec<AssistantMessageFrame> = events
            .into_iter()
            .filter_map(|event| encoded(&mut encoder, event))
            .collect();

        let types: Vec<&'static str> = frames.iter().map(|frame| frame.frame_type()).collect();
        assert_eq!(types, vec!["start", "text_start"]);
        let AssistantMessageFrame::Start { partial } = &frames[0] else {
            panic!("expected a start frame");
        };
        assert_eq!(partial.content, Vec::new());
        assert_eq!(partial.stop_reason, StopReason::Pending);
        let reduced = reduce_assistant_message_frames(frames)
            .expect("reduces")
            .expect("message");
        assert_eq!(
            reduced.content,
            vec![AssistantContent::Text(TextContent::new("Hello world"))]
        );
    }

    #[test]
    fn trims_only_the_covered_prefix_when_a_start_snapshot_lands_inside_a_delta() {
        let mut partial = seed();
        let mut encoder = AssistantMessageFrameEncoder::new();
        let mut frames = vec![frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        )];
        partial
            .content
            .push(AssistantContent::Text(TextContent::new("Hel")));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::TextStart {
                content_index: 0,
                partial: partial.clone(),
            },
        ));
        assert_eq!(
            encoded(
                &mut encoder,
                AssistantMessageEvent::TextDelta {
                    content_index: 0,
                    delta: "He".to_string(),
                    partial: partial.clone()
                }
            ),
            None
        );
        let remainder = encoded(
            &mut encoder,
            AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "llo".to_string(),
                partial,
            },
        )
        .expect("expected uncovered text delta");
        frames.push(remainder.clone());

        assert_eq!(
            remainder,
            AssistantMessageFrame::TextDelta {
                content_index: 0,
                delta: "lo".to_string()
            }
        );
        let reduced = reduce_assistant_message_frames(frames)
            .expect("reduces")
            .expect("message");
        assert_eq!(
            reduced.content,
            vec![AssistantContent::Text(TextContent::new("Hello"))]
        );
    }

    #[test]
    fn checkpoints_queued_tool_json_without_replaying_covered_deltas() {
        let mut partial = seed();
        let mut tool_call_value = tool_call("call", "write", json!({}));
        let mut events: Vec<AssistantMessageEvent> = vec![AssistantMessageEvent::Start {
            partial: partial.clone(),
        }];
        // toolcall_start is encoded after the arguments advanced, so the encoder
        // snapshots the final arguments (upstream live-reference semantics).
        tool_call_value.arguments = json!({"path": "README.md"})
            .as_object()
            .cloned()
            .expect("object");
        partial
            .content
            .push(AssistantContent::ToolCall(tool_call_value));
        events.push(AssistantMessageEvent::ToolcallStart {
            content_index: 0,
            partial: partial.clone(),
        });
        events.push(AssistantMessageEvent::ToolcallDelta {
            content_index: 0,
            delta: r#"{"path":"READ"#.to_string(),
            partial: partial.clone(),
        });
        events.push(AssistantMessageEvent::ToolcallDelta {
            content_index: 0,
            delta: r#"ME.md"}"#.to_string(),
            partial,
        });

        let mut encoder = AssistantMessageFrameEncoder::new();
        let frames: Vec<AssistantMessageFrame> = events
            .into_iter()
            .filter_map(|event| encoded(&mut encoder, event))
            .collect();
        let types: Vec<&'static str> = frames.iter().map(|frame| frame.frame_type()).collect();
        assert_eq!(
            types,
            vec!["start", "toolcall_start", "toolcall_checkpoint"]
        );
        assert_eq!(
            frames.last(),
            Some(&AssistantMessageFrame::ToolcallCheckpoint {
                content_index: 0,
                json: r#"{"path":"README.md"}"#.to_string(),
            })
        );
        let reduced = reduce_assistant_message_frames(frames)
            .expect("reduces")
            .expect("message");
        assert_eq!(
            reduced.content,
            vec![AssistantContent::ToolCall(tool_call(
                "call",
                "write",
                json!({"path": "README.md"})
            ))]
        );
    }

    #[test]
    fn resumes_legacy_grammar_tool_json_from_initial_arguments() {
        let mut partial = seed();
        let mut encoder = AssistantMessageFrameEncoder::new();
        let mut frames = vec![frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        )];
        let mut tool_call_value = tool_call("call", "bash", json!({"input": "a"}));
        partial
            .content
            .push(AssistantContent::ToolCall(tool_call_value.clone()));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ToolcallStart {
                content_index: 0,
                partial: partial.clone(),
            },
        ));
        tool_call_value.arguments = json!({"input": "ab"}).as_object().cloned().expect("object");
        partial.content[0] = AssistantContent::ToolCall(tool_call_value.clone());
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ToolcallDelta {
                content_index: 0,
                delta: r#"{"input":"ab"#.to_string(),
                partial: partial.clone(),
            },
        ));
        tool_call_value.arguments = json!({"input": "abc"})
            .as_object()
            .cloned()
            .expect("object");
        partial.content[0] = AssistantContent::ToolCall(tool_call_value);
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ToolcallDelta {
                content_index: 0,
                delta: r#"c"}"#.to_string(),
                partial,
            },
        ));

        assert_eq!(
            &frames[2..],
            &[
                AssistantMessageFrame::ToolcallCheckpoint {
                    content_index: 0,
                    json: r#"{"input":"ab"#.to_string(),
                },
                AssistantMessageFrame::ToolcallDelta {
                    content_index: 0,
                    delta: r#"c"}"#.to_string(),
                },
            ]
        );
        let reduced = reduce_assistant_message_frames(frames)
            .expect("reduces")
            .expect("message");
        assert_eq!(
            reduced.content,
            vec![AssistantContent::ToolCall(tool_call(
                "call",
                "bash",
                json!({"input": "abc"})
            ))]
        );
    }

    #[test]
    fn streams_tool_json_compactly_from_an_empty_argument_start() {
        let mut partial = seed();
        let mut encoder = AssistantMessageFrameEncoder::new();
        let mut frames = vec![frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        )];
        let mut tool_call_value = tool_call("call", "bash", json!({}));
        partial
            .content
            .push(AssistantContent::ToolCall(tool_call_value.clone()));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ToolcallStart {
                content_index: 0,
                partial: partial.clone(),
            },
        ));
        tool_call_value.arguments = json!({"command": "ls -la /tmp"})
            .as_object()
            .cloned()
            .expect("object");
        partial.content[0] = AssistantContent::ToolCall(tool_call_value);
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ToolcallDelta {
                content_index: 0,
                delta: r#"{"command":"ls -la /tmp"}"#.to_string(),
                partial,
            },
        ));

        assert_eq!(
            frames.last(),
            Some(&AssistantMessageFrame::ToolcallDelta {
                content_index: 0,
                delta: r#"{"command":"ls -la /tmp"}"#.to_string(),
            })
        );
        let reduced = reduce_assistant_message_frames(frames)
            .expect("reduces")
            .expect("message");
        assert_eq!(
            reduced.content[0],
            AssistantContent::ToolCall(tool_call(
                "call",
                "bash",
                json!({"command": "ls -la /tmp"})
            ))
        );
    }

    #[test]
    fn accepts_a_pre_generation_error_but_rejects_success_or_updates_before_start() {
        let mut failed = seed();
        failed.stop_reason = StopReason::Error;
        failed.error_message = Some("setup failed".to_string());
        assert_eq!(
            AssistantMessageFrameEncoder::new().encode(&AssistantMessageEvent::Error {
                reason: StreamErrorReason::Error,
                error: failed,
            }),
            Ok(None)
        );

        let mut completed = seed();
        completed.stop_reason = StopReason::Stop;
        expect_encode_error(
            AssistantMessageFrameEncoder::new().encode(&AssistantMessageEvent::Done {
                reason: StreamDoneReason::Stop,
                message: completed,
            }),
            "done event appears before start",
        );
        expect_encode_error(
            AssistantMessageFrameEncoder::new().encode(&AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "x".to_string(),
                partial: seed(),
            }),
            "text_delta event appears before start",
        );
    }

    #[test]
    fn treats_end_signature_metadata_including_absence_as_authoritative() {
        let mut stale_text = TextContent::new("");
        stale_text.text_signature = Some("stale-text".to_string());
        let mut stale_thinking = ThinkingContent::new("");
        stale_thinking.thinking_signature = Some("stale-thinking".to_string());
        stale_thinking.redacted = Some(true);
        let mut stale_tool_call = tool_call("call", "read", json!({}));
        stale_tool_call.thought_signature = Some("stale-tool".to_string());
        stale_tool_call.namespace = Some("stale-namespace".to_string());

        let frames: Vec<AssistantMessageFrame> = vec![
            start_frame(seed()),
            AssistantMessageFrame::TextStart {
                content_index: 0,
                content: stale_text,
            },
            AssistantMessageFrame::TextEnd {
                content_index: 0,
                content: String::new(),
                text_signature: None,
            },
            AssistantMessageFrame::ThinkingStart {
                content_index: 1,
                content: stale_thinking,
            },
            AssistantMessageFrame::ThinkingEnd {
                content_index: 1,
                content: String::new(),
                thinking_signature: Some(String::new()),
                redacted: Some(false),
            },
            AssistantMessageFrame::ToolcallStart {
                content_index: 2,
                tool_call: stale_tool_call,
            },
            AssistantMessageFrame::ToolcallEnd {
                content_index: 2,
                id: "call".to_string(),
                name: "read".to_string(),
                arguments: json!({}).as_object().cloned().expect("object"),
                thought_signature: None,
                namespace: None,
            },
        ];

        let reduced = reduce_assistant_message_frames(frames)
            .expect("reduces")
            .expect("message");
        assert_eq!(
            reduced.content,
            vec![
                AssistantContent::Text(TextContent::new("")),
                AssistantContent::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: Some(String::new()),
                    redacted: Some(false),
                }),
                AssistantContent::ToolCall(tool_call("call", "read", json!({}))),
            ]
        );
    }

    #[test]
    fn stores_authoritative_final_arguments_in_toolcall_end_frames() {
        let mut partial = seed();
        let mut tool_call_value = tool_call("call-1", "read", json!({"path": "README.md"}));
        tool_call_value.thought_signature = Some("thought".to_string());
        tool_call_value.namespace = Some("files".to_string());
        partial
            .content
            .push(AssistantContent::ToolCall(tool_call_value.clone()));

        let mut encoder = AssistantMessageFrameEncoder::new();
        frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        );
        frame(
            &mut encoder,
            AssistantMessageEvent::ToolcallStart {
                content_index: 0,
                partial: partial.clone(),
            },
        );
        let end = frame(
            &mut encoder,
            AssistantMessageEvent::ToolcallEnd {
                content_index: 0,
                tool_call: tool_call_value.clone(),
                partial,
            },
        );
        assert_eq!(
            end,
            AssistantMessageFrame::ToolcallEnd {
                content_index: 0,
                id: "call-1".to_string(),
                name: "read".to_string(),
                arguments: json!({"path": "README.md"})
                    .as_object()
                    .cloned()
                    .expect("object"),
                thought_signature: Some("thought".to_string()),
                namespace: Some("files".to_string()),
            }
        );
    }

    #[test]
    fn whitelists_public_block_fields_from_provider_shaped_partials() {
        // Provider-shaped scratch fields (index/partialJson/streamIndex/outputIndex)
        // cannot exist on the typed Rust structs; the whitelist behavior is
        // structural, so assert the encoded frames carry exactly the public fields.
        let mut partial = seed();
        partial.content.push(AssistantContent::Text(TextContent {
            text: "visible".to_string(),
            text_signature: Some("text-sig".to_string()),
        }));
        partial
            .content
            .push(AssistantContent::Thinking(ThinkingContent {
                thinking: "reasoning".to_string(),
                thinking_signature: Some("thinking-sig".to_string()),
                redacted: Some(false),
            }));
        let mut tool_call_value = tool_call("call", "run", json!({"value": 1}));
        tool_call_value.thought_signature = Some("tool-sig".to_string());
        tool_call_value.namespace = Some("tools".to_string());
        partial
            .content
            .push(AssistantContent::ToolCall(tool_call_value));

        let mut encoder = AssistantMessageFrameEncoder::new();
        let start = frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        );
        let text_start = frame(
            &mut encoder,
            AssistantMessageEvent::TextStart {
                content_index: 0,
                partial: partial.clone(),
            },
        );
        let thinking_start = frame(
            &mut encoder,
            AssistantMessageEvent::ThinkingStart {
                content_index: 1,
                partial: partial.clone(),
            },
        );
        let tool_start = frame(
            &mut encoder,
            AssistantMessageEvent::ToolcallStart {
                content_index: 2,
                partial,
            },
        );

        let AssistantMessageFrame::Start { partial } = &start else {
            panic!("expected a start frame");
        };
        assert_eq!(partial.content, Vec::new());
        let AssistantMessageFrame::TextStart { content, .. } = &text_start else {
            panic!("expected a text start frame");
        };
        assert_eq!(content.text, "visible");
        assert_eq!(content.text_signature.as_deref(), Some("text-sig"));
        let AssistantMessageFrame::ThinkingStart { content, .. } = &thinking_start else {
            panic!("expected a thinking start frame");
        };
        assert_eq!(content.thinking, "reasoning");
        let AssistantMessageFrame::ToolcallStart { tool_call, .. } = &tool_start else {
            panic!("expected a toolcall start frame");
        };
        assert_eq!(
            tool_call.arguments,
            json!({"value": 1}).as_object().cloned().expect("object")
        );
    }

    #[test]
    fn supports_interleaved_streams_by_content_index() {
        let frames: Vec<AssistantMessageFrame> = vec![
            start_frame(seed()),
            AssistantMessageFrame::TextStart {
                content_index: 0,
                content: TextContent::new(""),
            },
            AssistantMessageFrame::ToolcallStart {
                content_index: 1,
                tool_call: tool_call("call", "lookup", json!({})),
            },
            AssistantMessageFrame::ThinkingStart {
                content_index: 2,
                content: ThinkingContent::new(""),
            },
            AssistantMessageFrame::TextDelta {
                content_index: 0,
                delta: "answer".to_string(),
            },
            AssistantMessageFrame::ToolcallDelta {
                content_index: 1,
                delta: r#"{"query":"pi"}"#.to_string(),
            },
            AssistantMessageFrame::ThinkingDelta {
                content_index: 2,
                delta: "check".to_string(),
            },
            AssistantMessageFrame::ToolcallEnd {
                content_index: 1,
                id: "call".to_string(),
                name: "lookup".to_string(),
                arguments: json!({"query": "pi"}).as_object().cloned().expect("object"),
                thought_signature: None,
                namespace: None,
            },
            AssistantMessageFrame::TextEnd {
                content_index: 0,
                content: "answer".to_string(),
                text_signature: None,
            },
            AssistantMessageFrame::ThinkingEnd {
                content_index: 2,
                content: "check".to_string(),
                thinking_signature: None,
                redacted: None,
            },
        ];

        let reduced = reduce_assistant_message_frames(frames)
            .expect("reduces")
            .expect("message");
        assert_eq!(
            reduced.content,
            vec![
                AssistantContent::Text(TextContent::new("answer")),
                AssistantContent::ToolCall(tool_call("call", "lookup", json!({"query": "pi"}))),
                AssistantContent::Thinking(ThinkingContent::new("check")),
            ]
        );
    }

    #[test]
    fn snapshots_mutable_event_data_and_keeps_reduction_pure() {
        let mut partial = seed();
        partial.diagnostics = Some(vec![AssistantMessageDiagnostic {
            diagnostic_type: "test".to_string(),
            timestamp: 2,
            error: None,
            details: Some(
                json!({"value": "original"})
                    .as_object()
                    .cloned()
                    .expect("object"),
            ),
        }]);
        let mut encoder = AssistantMessageFrameEncoder::new();
        let start = frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        );
        if let Some(details) = partial.diagnostics.as_mut().expect("diagnostics")[0]
            .details
            .as_mut()
        {
            details.insert("value".to_string(), json!("mutated"));
        }
        partial.usage.cost.total = 99.0;

        let mut tool_call_value =
            tool_call("call", "run", json!({"nested": {"value": "original"}}));
        partial
            .content
            .push(AssistantContent::ToolCall(tool_call_value.clone()));
        let tool_start = frame(
            &mut encoder,
            AssistantMessageEvent::ToolcallStart {
                content_index: 0,
                partial,
            },
        );
        let tool_start_snapshot = tool_start.clone();
        tool_call_value.arguments = json!({"nested": {"value": "mutated"}})
            .as_object()
            .cloned()
            .expect("object");

        let reduced = reduce_assistant_message_frames(vec![start, tool_start])
            .expect("reduces")
            .expect("message");
        assert_eq!(
            reduced.diagnostics.as_ref().expect("diagnostics")[0]
                .details
                .as_ref()
                .expect("details")["value"],
            json!("original")
        );
        assert_eq!(reduced.usage.cost.total, 0.0);
        assert_eq!(
            reduced.content[0],
            AssistantContent::ToolCall(tool_call(
                "call",
                "run",
                json!({"nested": {"value": "original"}})
            ))
        );

        // The reduced message is a private snapshot: mutating its tool call must not
        // affect the encoded frame.
        let mut reduced = reduced;
        let AssistantContent::ToolCall(mutated) = &mut reduced.content[0] else {
            panic!("expected a tool call");
        };
        mutated.arguments = json!({"nested": "changed-output"})
            .as_object()
            .cloned()
            .expect("object");
        let AssistantMessageFrame::ToolcallStart { tool_call, .. } = tool_start_snapshot else {
            panic!("expected a toolcall start frame");
        };
        assert_eq!(
            tool_call.arguments,
            json!({"nested": {"value": "original"}})
                .as_object()
                .cloned()
                .expect("object")
        );
    }

    #[test]
    fn omits_terminal_events_because_settlement_is_separate() {
        let mut message = seed();
        let mut completed = AssistantMessageFrameEncoder::new();
        completed
            .encode(&AssistantMessageEvent::Start {
                partial: message.clone(),
            })
            .expect("encoding succeeds");
        message.stop_reason = StopReason::Stop;
        assert_eq!(
            completed.encode(&AssistantMessageEvent::Done {
                reason: StreamDoneReason::Stop,
                message: message.clone(),
            }),
            Ok(None)
        );
        message.stop_reason = StopReason::Error;
        message.error_message = Some("failed".to_string());
        assert_eq!(
            AssistantMessageFrameEncoder::new().encode(&AssistantMessageEvent::Error {
                reason: StreamErrorReason::Error,
                error: message,
            }),
            Ok(None)
        );
    }

    #[test]
    fn returns_none_when_there_is_no_start_frame() {
        assert_eq!(reduce_assistant_message_frames(Vec::new()), Ok(None));
        assert_eq!(
            reduce_assistant_message_frames(vec![AssistantMessageFrame::TextDelta {
                content_index: 0,
                delta: "x".to_string(),
            }]),
            Ok(None)
        );
    }

    #[test]
    fn rejects_frames_before_start_wrong_block_kinds_duplicate_ends_and_index_gaps() {
        expect_error(
            reduce_assistant_message_frames(vec![
                AssistantMessageFrame::TextDelta {
                    content_index: 0,
                    delta: "x".to_string(),
                },
                start_frame(seed()),
            ]),
            "before the start frame",
        );
        expect_error(
            reduce_assistant_message_frames(vec![
                start_frame(seed()),
                AssistantMessageFrame::ToolcallStart {
                    content_index: 0,
                    tool_call: tool_call("call", "run", json!({})),
                },
                AssistantMessageFrame::TextDelta {
                    content_index: 0,
                    delta: "wrong".to_string(),
                },
            ]),
            "expected text block",
        );
        expect_error(
            reduce_assistant_message_frames(vec![
                start_frame(seed()),
                AssistantMessageFrame::TextStart {
                    content_index: 0,
                    content: TextContent::new(""),
                },
                AssistantMessageFrame::TextEnd {
                    content_index: 0,
                    content: String::new(),
                    text_signature: None,
                },
                AssistantMessageFrame::TextEnd {
                    content_index: 0,
                    content: String::new(),
                    text_signature: None,
                },
            ]),
            "follows the end",
        );
        expect_error(
            reduce_assistant_message_frames(vec![
                start_frame(seed()),
                AssistantMessageFrame::TextStart {
                    content_index: 1,
                    content: TextContent::new(""),
                },
            ]),
            "would leave a gap",
        );
    }

    #[test]
    fn rejects_conversion_events_whose_content_index_points_to_the_wrong_block_kind() {
        let mut partial = seed();
        let mut encoder = AssistantMessageFrameEncoder::new();
        encoder
            .encode(&AssistantMessageEvent::Start {
                partial: partial.clone(),
            })
            .expect("encoding succeeds");
        partial
            .content
            .push(AssistantContent::Thinking(ThinkingContent::new("")));
        expect_encode_error(
            encoder.encode(&AssistantMessageEvent::TextStart {
                content_index: 0,
                partial,
            }),
            "text_start event points to thinking block",
        );
    }
}
