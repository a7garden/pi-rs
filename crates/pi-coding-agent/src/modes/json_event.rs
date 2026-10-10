//! Port of `packages/coding-agent/src/modes/json-event.ts` plus the wire
//! projection for [`AgentSessionEvent`] (upstream serializes the typed TS
//! event objects directly with `JSON.stringify`).
//!
//! Upstream types (`JsonAgentSessionEvent`, the `WithoutPartial` mapped
//! types) are type-level only — the Rust wire surface is
//! [`serde_json::Value`].
//!
//! Deviation **D93** — [`AgentSessionEvent`] is a typed enum (D75), not a
//! plain object, so the pass-through branch of `toJsonEvent` goes through the
//! explicit [`event_to_json`] projection; field names mirror the upstream
//! emit sites in `core/agent-session.ts` (`followUp`, `contentIndex`,
//! `willRetry`, …). Key order is preserved (`serde_json/preserve_order`).

use serde_json::{json, Map, Value as JsonValue};

use pi_agent::types::AgentMessage;
use pi_ai::types::{AssistantContent, Message, Usage};

use crate::core::agent_session::{AgentSessionEvent, SummarizationRetrySource};
use crate::core::extensions::types::CompactReason;

/// `Usage` as JSON (`message.usage` on the wire).
fn usage_to_json(usage: &Usage) -> JsonValue {
    serde_json::to_value(usage).unwrap_or(JsonValue::Null)
}

/// Drop `partial` from an assistant message event, mirroring the upstream
/// `const { partial: _partial, ...deltaEvent } = event` spread.
fn strip_partial(event: &JsonValue) -> JsonValue {
    let mut stripped = event.clone();
    if let Some(object) = stripped.as_object_mut() {
        object.remove("partial");
    }
    stripped
}

/// `toJsonAssistantMessageEvent`: remove cumulative snapshots; `toolcall_start`
/// gains constant-size `id`/`toolName` metadata.
fn to_json_assistant_message_event(
    event: &pi_ai::types::AssistantMessageEvent,
) -> Result<JsonValue, String> {
    if let pi_ai::types::AssistantMessageEvent::ToolcallStart {
        content_index,
        partial,
    } = event
    {
        let tool_call = partial.content.get(*content_index as usize);
        let Some(AssistantContent::ToolCall(tool_call)) = tool_call else {
            return Err(format!(
                "toolcall_start content at index {content_index} is not a tool call"
            ));
        };
        let serialized = serde_json::to_value(event).unwrap_or(JsonValue::Null);
        let mut delta_event = strip_partial(&serialized);
        let object = delta_event
            .as_object_mut()
            .expect("serialized event object");
        object.insert("id".to_string(), JsonValue::String(tool_call.id.clone()));
        object.insert(
            "toolName".to_string(),
            JsonValue::String(tool_call.name.clone()),
        );
        return Ok(delta_event);
    }

    let serialized = serde_json::to_value(event).unwrap_or(JsonValue::Null);
    if serialized.get("partial").is_none() {
        return Ok(serialized);
    }
    Ok(strip_partial(&serialized))
}

/// `toJsonEvent`: remove cumulative assistant snapshots from streaming wire
/// events. `message_start` provides the initial message, deltas build it, and
/// `message_end` provides the final authoritative message. Cumulative usage,
/// tool-call ids, and tool names remain available because their size is
/// constant.
pub fn to_json_event(event: &AgentSessionEvent) -> Result<JsonValue, String> {
    let AgentSessionEvent::MessageUpdate {
        message,
        assistant_message_event,
    } = event
    else {
        return Ok(event_to_json(event));
    };
    if !matches!(message, AgentMessage::Llm(Message::Assistant(_))) {
        return Err("message_update message is not an assistant message".to_string());
    }

    let usage = message_usage(message);
    Ok(json!({
        "type": "message_update",
        "usage": usage_to_json(&usage),
        "assistantMessageEvent": to_json_assistant_message_event(assistant_message_event)?,
    }))
}

/// Usage of an LLM message (assistant messages carry it; other roles default
/// to zero, matching upstream's assistant-only guarantee).
fn message_usage(message: &AgentMessage) -> Usage {
    if let AgentMessage::Llm(Message::Assistant(assistant)) = message {
        return assistant.usage.clone();
    }
    Usage::default()
}

/// Wire projection of every non-`message_update` session event (upstream
/// emits these object literals directly from `core/agent-session.ts`).
pub fn event_to_json(event: &AgentSessionEvent) -> JsonValue {
    match event {
        AgentSessionEvent::AgentStart => json!({ "type": "agent_start" }),
        AgentSessionEvent::AgentEnd {
            messages,
            will_retry,
        } => {
            json!({ "type": "agent_end", "messages": messages_to_json(messages), "willRetry": will_retry })
        }
        AgentSessionEvent::TurnStart => json!({ "type": "turn_start" }),
        AgentSessionEvent::TurnEnd {
            message,
            tool_results,
        } => json!({
            "type": "turn_end",
            "message": message_to_json(message),
            "toolResults": tool_results_to_json(tool_results),
        }),
        AgentSessionEvent::MessageStart { message } => {
            json!({ "type": "message_start", "message": message_to_json(message) })
        }
        AgentSessionEvent::MessageUpdate { .. } => {
            // Handled by `to_json_event` (the wire drops the cumulative
            // snapshot); the full event shape is only projected here for
            // completeness.
            json!({
                "type": "message_update",
                "message": message_to_json(&match event {
                    AgentSessionEvent::MessageUpdate { message, .. } => message.clone(),
                    _ => unreachable!(),
                }),
            })
        }
        AgentSessionEvent::MessageEnd { message } => {
            json!({ "type": "message_end", "message": message_to_json(message) })
        }
        AgentSessionEvent::ToolExecutionStart {
            tool_call_id,
            tool_name,
            args,
            parent_tool_call_id,
        } => {
            let mut object = Map::new();
            object.insert("type".into(), json!("tool_execution_start"));
            object.insert("toolCallId".into(), json!(tool_call_id));
            object.insert("toolName".into(), json!(tool_name));
            object.insert("args".into(), args.clone());
            insert_parent_tool_call_id(&mut object, parent_tool_call_id.as_deref());
            JsonValue::Object(object)
        }
        AgentSessionEvent::ToolExecutionUpdate {
            tool_call_id,
            tool_name,
            args,
            partial_result,
            parent_tool_call_id,
        } => {
            let mut object = Map::new();
            object.insert("type".into(), json!("tool_execution_update"));
            object.insert("toolCallId".into(), json!(tool_call_id));
            object.insert("toolName".into(), json!(tool_name));
            object.insert("args".into(), args.clone());
            object.insert("partialResult".into(), partial_result.clone());
            insert_parent_tool_call_id(&mut object, parent_tool_call_id.as_deref());
            JsonValue::Object(object)
        }
        AgentSessionEvent::ToolExecutionEnd {
            tool_call_id,
            tool_name,
            result,
            is_error,
            duration_ms,
            parent_tool_call_id,
        } => {
            let mut object = Map::new();
            object.insert("type".into(), json!("tool_execution_end"));
            object.insert("toolCallId".into(), json!(tool_call_id));
            object.insert("toolName".into(), json!(tool_name));
            object.insert("result".into(), result.clone());
            object.insert("isError".into(), json!(is_error));
            if let Some(duration_ms) = duration_ms {
                object.insert("durationMs".into(), json!(duration_ms));
            }
            insert_parent_tool_call_id(&mut object, parent_tool_call_id.as_deref());
            JsonValue::Object(object)
        }
        AgentSessionEvent::AgentSettled { aborted } => {
            json!({ "type": "agent_settled", "aborted": aborted })
        }
        AgentSessionEvent::QueueUpdate {
            steering,
            follow_up,
        } => json!({
            "type": "queue_update",
            "steering": steering,
            "followUp": follow_up,
        }),
        AgentSessionEvent::CompactionStart { reason } => {
            json!({ "type": "compaction_start", "reason": compact_reason_to_json(*reason) })
        }
        AgentSessionEvent::EntryAppended { entry } => {
            json!({ "type": "entry_appended", "entry": entry_to_json(entry) })
        }
        AgentSessionEvent::SessionInfoChanged { name } => {
            json!({ "type": "session_info_changed", "name": name })
        }
        AgentSessionEvent::ThinkingLevelChanged { level } => {
            json!({ "type": "thinking_level_changed", "level": level })
        }
        AgentSessionEvent::CompactionEnd {
            reason,
            result,
            aborted,
            will_retry,
            error_message,
        } => {
            let mut object = Map::new();
            object.insert("type".into(), json!("compaction_end"));
            object.insert("reason".into(), compact_reason_to_json(*reason));
            object.insert(
                "result".into(),
                match result {
                    Some(result) => compaction_result_to_json(result),
                    None => JsonValue::Null,
                },
            );
            object.insert("aborted".into(), json!(aborted));
            object.insert("willRetry".into(), json!(will_retry));
            if let Some(error_message) = error_message {
                object.insert("errorMessage".into(), json!(error_message));
            }
            JsonValue::Object(object)
        }
        AgentSessionEvent::AutoRetryStart {
            attempt,
            max_attempts,
            delay_ms,
            error_message,
        } => json!({
            "type": "auto_retry_start",
            "attempt": attempt,
            "maxAttempts": max_attempts,
            "delayMs": delay_ms,
            "errorMessage": error_message,
        }),
        AgentSessionEvent::AutoRetryEnd {
            success,
            attempt,
            final_error,
        } => {
            let mut object = Map::new();
            object.insert("type".into(), json!("auto_retry_end"));
            object.insert("success".into(), json!(success));
            object.insert("attempt".into(), json!(attempt));
            if let Some(final_error) = final_error {
                object.insert("finalError".into(), json!(final_error));
            }
            JsonValue::Object(object)
        }
        AgentSessionEvent::SummarizationRetryScheduled {
            attempt,
            max_attempts,
            delay_ms,
            error_message,
        } => json!({
            "type": "summarization_retry_scheduled",
            "attempt": attempt,
            "maxAttempts": max_attempts,
            "delayMs": delay_ms,
            "errorMessage": error_message,
        }),
        AgentSessionEvent::SummarizationRetryAttemptStart { source } => {
            let mut object = Map::new();
            object.insert("type".into(), json!("summarization_retry_attempt_start"));
            if let JsonValue::Object(source) = retry_source_to_json(source) {
                object.extend(source);
            }
            JsonValue::Object(object)
        }
        AgentSessionEvent::SummarizationRetryFinished => {
            json!({ "type": "summarization_retry_finished" })
        }
        AgentSessionEvent::BashExecutionUpdate { id, delta } => {
            let mut object = Map::new();
            object.insert("type".into(), json!("bash_execution_update"));
            if let Some(id) = id {
                object.insert("id".into(), json!(id));
            }
            object.insert("delta".into(), json!(delta));
            JsonValue::Object(object)
        }
    }
}

fn insert_parent_tool_call_id(
    object: &mut Map<String, JsonValue>,
    parent_tool_call_id: Option<&str>,
) {
    if let Some(parent_tool_call_id) = parent_tool_call_id {
        object.insert("parentToolCallId".into(), json!(parent_tool_call_id));
    }
}

fn retry_source_to_json(source: &SummarizationRetrySource) -> JsonValue {
    match source {
        SummarizationRetrySource::BranchSummary => json!({ "source": "branchSummary" }),
        SummarizationRetrySource::Compaction { reason } => {
            json!({ "source": "compaction", "reason": compact_reason_to_json(*reason) })
        }
    }
}

/// `CompactReason` wire names.
pub fn compact_reason_to_json(reason: CompactReason) -> JsonValue {
    serde_json::to_value(reason).unwrap_or(JsonValue::Null)
}

fn messages_to_json(messages: &[AgentMessage]) -> JsonValue {
    JsonValue::Array(messages.iter().map(message_to_json).collect())
}

fn tool_results_to_json(tool_results: &[pi_ai::types::ToolResultMessage]) -> JsonValue {
    JsonValue::Array(
        tool_results
            .iter()
            .map(|result| serde_json::to_value(result).unwrap_or(JsonValue::Null))
            .collect(),
    )
}

fn message_to_json(message: &AgentMessage) -> JsonValue {
    serde_json::to_value(message).unwrap_or(JsonValue::Null)
}

fn entry_to_json(entry: &crate::core::session_manager::SessionEntry) -> JsonValue {
    serde_json::to_value(entry).unwrap_or(JsonValue::Null)
}

/// `CompactionResult` wire shape (upstream object literal; D93 projection —
/// the Rust struct is not `Serialize`).
pub fn compaction_result_to_json(result: &crate::core::compaction::CompactionResult) -> JsonValue {
    let mut object = Map::new();
    object.insert("summary".into(), json!(result.summary));
    object.insert("firstKeptEntryId".into(), json!(result.first_kept_entry_id));
    object.insert("tokensBefore".into(), json!(result.tokens_before));
    if let Some(estimated_tokens_after) = result.estimated_tokens_after {
        object.insert("estimatedTokensAfter".into(), json!(estimated_tokens_after));
    }
    if let Some(usage) = &result.usage {
        object.insert("usage".into(), usage_to_json(usage));
    }
    if let Some(details) = &result.details {
        object.insert(
            "details".into(),
            serde_json::to_value(details).unwrap_or(JsonValue::Null),
        );
    }
    JsonValue::Object(object)
}

#[cfg(test)]
mod tests {
    //! Ports of:
    //! - `test/suite/regressions/7290-json-stream-linear.test.ts`
    //! - `test/suite/regressions/7911-json-stream-usage.test.ts`
    //! - `test/suite/regressions/7925-toolcall-start-metadata.test.ts`
    //!
    //! Each drives a live `AgentSession` (suite `createHarness` upstream) and
    //! projects its `message_update` events through [`to_json_event`].

    use std::collections::VecDeque;
    use std::sync::Arc;

    use parking_lot::Mutex;

    use serde_json::{json, Value as JsonValue};

    use pi_agent::types::StreamFn;
    use pi_ai::types::{
        AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
        Message, StopReason, StreamDoneReason, ToolCall,
    };

    use crate::core::agent_session::{
        AgentSession, AgentSessionConfig, AgentSessionEvent, ScopedModelEntry,
    };
    use crate::core::session_manager::SessionManager;
    use crate::core::settings_manager::{Settings, SettingsManager, SettingsManagerCreateOptions};

    use super::to_json_event;
    use crate::modes::test_support::{
        assistant_message, loader_with_factories, temp_dir, test_model_runtime, text_model,
    };

    /// Stream events for one queued response: the faux provider streams real
    /// deltas (`text_start`/`text_delta`/`text_end`, `toolcall_start`) so
    /// `message_update` events occur at all.
    fn stream_events_for(response: &AssistantMessage) -> Vec<AssistantMessageEvent> {
        let mut events = vec![AssistantMessageEvent::Start {
            partial: AssistantMessage {
                content: Vec::new(),
                ..response.clone()
            },
        }];
        for (index, content) in response.content.iter().enumerate() {
            let content_index = index as u32;
            let mut partial = AssistantMessage {
                content: Vec::new(),
                ..response.clone()
            };
            match content {
                AssistantContent::Text(text) => {
                    events.push(AssistantMessageEvent::TextStart {
                        content_index,
                        partial: partial.clone(),
                    });
                    partial.content.push(AssistantContent::Text(text.clone()));
                    events.push(AssistantMessageEvent::TextDelta {
                        content_index,
                        delta: text.text.clone(),
                        partial: partial.clone(),
                    });
                    events.push(AssistantMessageEvent::TextEnd {
                        content_index,
                        content: text.text.clone(),
                        partial,
                    });
                }
                AssistantContent::ToolCall(tool_call) => {
                    partial
                        .content
                        .push(AssistantContent::ToolCall(tool_call.clone()));
                    events.push(AssistantMessageEvent::ToolcallStart {
                        content_index,
                        partial,
                    });
                }
                AssistantContent::Thinking(_) => {}
            }
        }
        events.push(AssistantMessageEvent::Done {
            reason: StreamDoneReason::Stop,
            message: response.clone(),
        });
        events
    }

    /// `harness.setResponses`: each run consumes the next queued response.
    fn queue_stream_fn(responses: Vec<AssistantMessage>) -> StreamFn {
        let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        Arc::new(move |_model, _context, _options| {
            let queue = Arc::clone(&queue);
            Box::pin(async move {
                let response = queue.lock().pop_front().expect("queued response");
                let stream = AssistantMessageEventStream::new();
                for event in stream_events_for(&response) {
                    stream.push(event);
                }
                Ok(stream)
            })
        })
    }

    async fn harness(responses: Vec<AssistantMessage>) -> Arc<AgentSession> {
        let dir = temp_dir("json-event");
        let model = text_model(200_000);
        let loader = loader_with_factories(&dir, Vec::new()).await;
        let model_runtime = test_model_runtime(model.clone(), true).await;
        let agent = Arc::new(pi_agent::Agent::new(pi_agent::AgentOptions {
            stream_fn: Some(queue_stream_fn(responses)),
            initial_state: pi_agent::AgentInitialState {
                system_prompt: Some("Test".to_string()),
                model: Some(model),
                thinking_level: None,
                tools: Vec::new(),
                messages: Vec::new(),
            },
            ..Default::default()
        }));
        AgentSession::new(AgentSessionConfig {
            agent,
            session_manager: Arc::new(Mutex::new(
                SessionManager::in_memory(None, None, None).unwrap(),
            )),
            settings_manager: Arc::new(Mutex::new(SettingsManager::in_memory(
                &Settings::new(),
                SettingsManagerCreateOptions::default(),
            ))),
            cwd: dir,
            scoped_models: Vec::<ScopedModelEntry>::new(),
            resource_loader: loader,
            custom_tools: Vec::new(),
            model_runtime,
            cache_warmer: None,
            initial_active_tool_names: None,
            uses_default_tools: false,
            default_tool_modifiers: Vec::new(),
            allowed_tool_names: None,
            excluded_tool_names: None,
            base_tools_override: None,
            extension_runner_ref: None,
            session_start_event: None,
        })
    }

    async fn prompt_collecting_updates(
        session: &Arc<AgentSession>,
        prompt: &str,
    ) -> Vec<JsonValue> {
        let updates: Arc<Mutex<Vec<AgentSessionEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&updates);
        let unsubscribe = session.subscribe(move |event| {
            if matches!(event, AgentSessionEvent::MessageUpdate { .. }) {
                sink.lock().push(event);
            }
        });
        session.prompt(prompt, None).await.unwrap();
        unsubscribe();
        let guard = updates.lock();
        guard
            .iter()
            .map(|event| to_json_event(event).expect("message_update projection"))
            .collect()
    }

    /// `measureUpdateBytes`: delta-only updates whose serialized size scales
    /// linearly with the streamed text.
    async fn measure_update_bytes(text: &str) -> usize {
        let session = harness(vec![assistant_message(text)]).await;

        let updates = prompt_collecting_updates(&session, "respond").await;
        assert!(!updates.is_empty());
        let mut total = 0;
        for update in &updates {
            assert!(
                update.get("message").is_none(),
                "wire update must drop the message snapshot"
            );
            let event = update
                .get("assistantMessageEvent")
                .expect("assistant event member");
            assert!(
                event.get("partial").is_none(),
                "wire event must drop the partial snapshot"
            );
            total += serde_json::to_string(update).unwrap().len();
        }
        total
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn emits_delta_only_message_updates_whose_size_scales_linearly() {
        let small_bytes = measure_update_bytes(&"x".repeat(2_000)).await;
        let large_bytes = measure_update_bytes(&"x".repeat(4_000)).await;

        assert!(large_bytes > small_bytes);
        assert!(
            (large_bytes as f64) / (small_bytes as f64) < 2.2,
            "large {large_bytes} vs small {small_bytes}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn includes_cumulative_usage_without_cumulative_message_snapshots() {
        // #7911 requires a response whose usage is populated.
        let response = AssistantMessage {
            usage: super::super::test_support::usage(10),
            ..assistant_message("hello")
        };
        let session = harness(vec![response]).await;
        let updates: Arc<Mutex<Vec<AgentSessionEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&updates);
        let unsubscribe = session.subscribe(move |event| {
            if matches!(event, AgentSessionEvent::MessageUpdate { .. }) {
                sink.lock().push(event);
            }
        });
        session.prompt("respond", None).await.unwrap();
        unsubscribe();

        // #7290's delta-only wire projection dropped fixed-size metadata with
        // the snapshots; usage must survive.
        let update = updates
            .lock()
            .iter()
            .find(|event| match event {
                AgentSessionEvent::MessageUpdate { message, .. } => match message {
                    pi_agent::types::AgentMessage::Llm(Message::Assistant(assistant)) => {
                        assistant.usage.total_tokens > 0
                    }
                    _ => false,
                },
                _ => false,
            })
            .expect("an assistant update with populated usage")
            .clone();
        let (message, assistant_event) = match &update {
            AgentSessionEvent::MessageUpdate {
                message,
                assistant_message_event,
            } => (message, assistant_message_event),
            _ => unreachable!(),
        };
        let usage = match message {
            pi_agent::types::AgentMessage::Llm(Message::Assistant(assistant)) => {
                assistant.usage.clone()
            }
            _ => unreachable!(),
        };

        let wire_update = to_json_event(&update).unwrap();
        assert_eq!(
            wire_update.get("usage"),
            Some(&serde_json::to_value(&usage).unwrap())
        );
        assert!(wire_update.get("message").is_none());
        assert!(wire_update
            .get("assistantMessageEvent")
            .and_then(|event| event.get("partial"))
            .is_none());
        let _ = assistant_event;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn includes_the_tool_call_id_and_name_without_cumulative_snapshots() {
        let tool_call = ToolCall {
            id: "call_7925".to_string(),
            name: "write".to_string(),
            arguments: serde_json::from_value(
                json!({ "path": "output.txt", "content": "x".repeat(100) }),
            )
            .unwrap(),
            thought_signature: None,
            namespace: None,
        };
        let mut tool_use_message = assistant_message("");
        tool_use_message.content = vec![AssistantContent::ToolCall(tool_call)];
        tool_use_message.stop_reason = StopReason::ToolUse;

        let session = harness(vec![tool_use_message, assistant_message("done")]).await;
        let updates = prompt_collecting_updates(&session, "write a file").await;

        let update = updates
            .iter()
            .find(|update| {
                update
                    .pointer("/assistantMessageEvent/type")
                    .and_then(JsonValue::as_str)
                    == Some("toolcall_start")
            })
            .expect("toolcall_start assistant update");

        assert_eq!(
            update,
            &json!({
                "type": "message_update",
                "usage": update.get("usage").unwrap(),
                "assistantMessageEvent": {
                    "type": "toolcall_start",
                    "contentIndex": 0,
                    "id": "call_7925",
                    "toolName": "write",
                },
            })
        );
    }
}
